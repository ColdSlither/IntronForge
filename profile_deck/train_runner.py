"""Trainer bridge: drive LoRAlab TrainerStudio's SDXL pipeline from IntronForge.

LoRAlab (https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio, MIT)
stays a separate, commit-pinned component with its own venv. This module runs
its file-driven scripts (1_pre_cache_sdxl.py -> 2_train_lora_sdxl.py) as
subprocesses with CWD set to the LoRAlab repo, captures their console into
per-job logs, and copies the finished LoRA into Forge's Lora dir where the
deck's scan/enrichment picks it up. Their settings files are backed up
before each run and restored after, so their own launcher stays usable.

Credit obligation: see THIRD_PARTY_NOTICES.md at the project root.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

LORALAB_DIR = Path(os.environ.get("LORALAB_DIR", "/path/to/AcademiaSD_LoRAlab-TrainerStudio"))
PINNED_COMMIT = "49e89e8e13790a7a0539093d660a23317d86f73c"
LORA_OUT_DIR = Path(os.environ.get("LORA_OUT_DIR",
                                   "/path/to/forge/models/Lora"))
TRAININGS = Path(__file__).resolve().parent / "trainings"
DATASETS = Path(__file__).resolve().parent / "datasets"
FORGE = "http://127.0.0.1:7860"

PRESETS = {
    "sdxl_base": "SDXL Base 1.0 (downloads from HF on first run)",
    "pony_v6": "Pony Diffusion V6 XL (downloads from HF; score-prefix lane)",
    "illustrious_v01": "Illustrious XL v0.1 (downloads from HF)",
    "juggernaut_xi": "Juggernaut XI (downloads from HF)",
    "custom": "custom checkpoint file (no download)",
}

THEIR_SETTINGS = ["pre_cache_settings_sdxl.json", "train_settings_sdxl.json"]

_LOCK = threading.Lock()
JOB = {"state": "idle", "id": None, "phase": None, "dataset": None, "preset": None,
       "name": None, "started": None, "finished": None, "error": None,
       "step": None, "exported": None}
_log_lines = []
_proc = {"p": None}


class Busy(Exception):
    pass


def _log(line: str):
    _log_lines.append(line)
    del _log_lines[:-60]
    jid = JOB.get("id")
    if jid:
        p = TRAININGS / jid / "run.log"
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def forge_busy() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(FORGE + "/sdapi/v1/progress", timeout=5) as r:
            d = json.loads(r.read())
        return bool(d.get("state", {}).get("job_count")) or float(d.get("progress") or 0) > 0.001
    except Exception:
        return False


def commit_drift() -> str | None:
    try:
        head = subprocess.run(["git", "-C", str(LORALAB_DIR), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        return head if head and head != PINNED_COMMIT else None
    except Exception:
        return None


def status() -> dict:
    out = dict(JOB)
    out["log"] = list(_log_lines[-25:])
    out["pinned_commit"] = PINNED_COMMIT
    out["commit_drift"] = commit_drift()
    out["lora_out_dir"] = str(LORA_OUT_DIR)
    return out


def stop() -> dict:
    p = _proc["p"]
    if p and p.poll() is None:
        p.terminate()  # their trainer saves state on signal
        return {"stopping": True, "id": JOB.get("id")}
    return {"stopping": False, "id": JOB.get("id")}


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", (text or "").strip().lower()).strip("-")[:50]
    return s or "trained_lora"


def start(dataset: str, preset: str, name: str, trigger: str = "",
          custom_checkpoint: str = "", overrides: dict | None = None) -> dict:
    if not _LOCK.acquire(blocking=False):
        raise Busy(f"a training job is already running ({JOB.get('id')})")
    try:
        ds = DATASETS / dataset
        if (not dataset or "/" in dataset or not ds.is_dir()):
            raise ValueError(f"dataset {dataset!r} not found (export one from the library first)")
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}")
        venv_py = LORALAB_DIR / "venv" / "bin" / "python"
        if not venv_py.is_file():
            raise ValueError(f"LoRAlab venv not found at {venv_py}")
        if preset == "custom":
            if not Path(custom_checkpoint).is_file():
                raise ValueError("custom preset needs an existing checkpoint file path")
        if forge_busy():
            raise Busy("Forge is mid-render on the shared GPU; wait for it to finish "
                       "or stop the render before training")
        jid = f"train_{time.strftime('%Y%m%d_%H%M%S')}"
        job_dir = TRAININGS / jid
        (job_dir / "out").mkdir(parents=True, exist_ok=True)
        JOB.update({"state": "running", "id": jid, "phase": "pre-cache",
                    "dataset": dataset, "preset": preset, "name": _slug(name),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "finished": None, "error": None, "step": None, "exported": None})
        threading.Thread(target=_run_job, args=(job_dir, ds, preset, trigger,
                                                custom_checkpoint, overrides or {}),
                         daemon=True).start()
        return status()
    except Exception:
        _LOCK.release()
        raise


def _write_their_settings(job_dir: Path, ds: Path, preset: str, trigger: str,
                          custom_checkpoint: str, overrides: dict):
    sdir = LORALAB_DIR / "settings"
    sdir.mkdir(parents=True, exist_ok=True)
    cache = str(job_dir / "cache")
    pre = {"model_preset": preset, "custom_checkpoint": str(custom_checkpoint or ""),
           "dataset_path": str(ds), "cache_dir": cache, "project_name": "",
           "trigger_word": trigger or ""}
    train = {"cache_dir": cache, "output_dir": str(job_dir / "out"), "project_name": "",
             "trigger_word": trigger or "", "preview_every": 0, "save_every": 0,
             "precision": "nf4", "total_steps": 1500}
    for k, v in (overrides or {}).items():
        if v is not None:
            train[k] = v
    (sdir / "pre_cache_settings_sdxl.json").write_text(json.dumps(pre, indent=1))
    (sdir / "train_settings_sdxl.json").write_text(json.dumps(train, indent=1))


def _backup_their_settings(job_dir: Path) -> dict:
    sdir = LORALAB_DIR / "settings"
    saved = {}
    for fname in THEIR_SETTINGS:
        p = sdir / fname
        if p.is_file():
            content = p.read_text(encoding="utf-8")
            saved[fname] = content
            (job_dir / f"backup_{fname}").write_text(content, encoding="utf-8")
    return saved


def _restore_their_settings(saved: dict):
    sdir = LORALAB_DIR / "settings"
    sdir.mkdir(parents=True, exist_ok=True)
    for fname in THEIR_SETTINGS:
        p = sdir / fname
        if fname in saved:
            p.write_text(saved[fname], encoding="utf-8")
        else:
            p.unlink(missing_ok=True)


def _parse_step(line: str):
    m = re.search(r"(?:paso|step)\s+(\d+)", line, re.I)
    if m:
        try:
            JOB["step"] = int(m.group(1))
        except ValueError:
            pass


def _run_job(job_dir: Path, ds: Path, preset: str, trigger: str,
             custom_checkpoint: str, overrides: dict):
    venv_py = LORALAB_DIR / "venv" / "bin" / "python"
    t0 = time.time()
    saved = _backup_their_settings(job_dir)
    try:
        _write_their_settings(job_dir, ds, preset, trigger, custom_checkpoint, overrides)
        for phase, script in (("pre-cache", "scripts/1_pre_cache_sdxl.py"),
                              ("train", "scripts/2_train_lora_sdxl.py")):
            JOB["phase"] = phase
            _log(f"[{phase}] starting {script}")
            p = subprocess.Popen([str(venv_py), script], cwd=str(LORALAB_DIR),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", bufsize=1)
            _proc["p"] = p
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    _log(line)
                    _parse_step(line)
            rc = p.wait()
            _proc["p"] = None
            if rc != 0:
                raise RuntimeError(f"{phase} exited with code {rc}")
        final = job_dir / "out" / "SDXL_FINAL_LoRA.safetensors"
        if not final.is_file():
            raise RuntimeError("trainer finished but no SDXL_FINAL_LoRA.safetensors found")
        LORA_OUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = LORA_OUT_DIR / f"{JOB['name']}.safetensors"
        n = 0
        while dest.exists():
            n += 1
            dest = LORA_OUT_DIR / f"{JOB['name']}_{n}.safetensors"
        shutil.copy2(final, dest)
        JOB.update({"state": "done", "phase": "exported",
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "exported": dest.name})
        _log(f"[export] copied to {dest} (restart Forge or refresh models to see it in the deck)")
    except Exception as ex:
        JOB.update({"state": "error", "error": f"{type(ex).__name__}: {ex}"[:300],
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        _log(f"[error] {JOB['error']}")
    finally:
        _restore_their_settings(saved)
        (job_dir / "result.json").write_text(json.dumps(status(), indent=1, default=str),
                                             encoding="utf-8")
        _proc["p"] = None
        _LOCK.release()
