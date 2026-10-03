"""Profile Deck: local web app tunneling into Forge Neo's REST API.

Bind 127.0.0.1 only. Run with the Forge venv python:
    /path/to/forge/venv/bin/python -m uvicorn app:app \
        --host 127.0.0.1 --port 7877
"""
import json
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import forge_client
import lora_store

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "profiles"
OUTPUTS = ROOT / "outputs"
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

app = FastAPI(title="Profile Deck")
app.mount("/loras-files", StaticFiles(directory=str(lora_store.LORA_DIR)), name="loras")
app.mount("/lora-previews", StaticFiles(directory=str(lora_store.PREVIEWS)), name="lora-previews")


def profile_path(character: str, name: str) -> Path:
    if not (KEY_RE.match(character) and KEY_RE.match(name)):
        raise HTTPException(400, "profile keys must match [a-z0-9][a-z0-9._-]*")
    return PROFILES / character / f"{name}.json"


def load_profile(character: str, name: str) -> dict:
    p = profile_path(character, name)
    if not p.is_file():
        raise HTTPException(404, f"profile {character}/{name} not found")
    return json.loads(p.read_text())


def list_profiles() -> dict:
    out = {}
    for char_dir in sorted(PROFILES.iterdir()):
        if not char_dir.is_dir():
            continue
        names = sorted(p.stem for p in char_dir.glob("*.json"))
        if names:
            out[char_dir.name] = names
    return out


class GenerateReq(BaseModel):
    character: str
    name: str
    overrides: dict = {}
    suffix: str = ""
    profile: dict | None = None  # ephemeral form state; never persisted


class BatchReq(BaseModel):
    character: str
    name: str
    lines: list[str]
    seed_mode: str = "increment"  # fixed | increment | random
    base_seed: int | None = None
    profile: dict | None = None  # ephemeral form state; never persisted


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


UNIVERSAL = ROOT / "universal.json"


def load_universal() -> dict:
    return json.loads(UNIVERSAL.read_text())


def apply_universal(profile: dict) -> dict:
    """Ratio fields set to null in a profile inherit the universal pins."""
    u = load_universal()
    b = profile.setdefault("base", {})
    for k in ("steps", "cfg_scale", "distilled_cfg_scale"):
        if b.get(k) is None:
            b[k] = u[k]
    h = profile.get("hires")
    if isinstance(h, dict):
        for k, uk in (("scale", "hr_scale"), ("steps", "hr_steps"),
                      ("denoise", "hr_denoise"), ("cfg", "hr_cfg")):
            if h.get(k) is None:
                h[k] = u[uk]
    for t in (profile.get("detailer") or {}).get("tabs", []):
        if isinstance(t, dict):
            for k in ("confidence", "denoise", "padding"):
                if t.get(k) is None:
                    t[k] = u[k]
    return profile


@app.get("/api/universal")
def get_universal():
    return load_universal()


@app.put("/api/universal")
def put_universal(u: dict):
    UNIVERSAL.write_text(json.dumps(u, indent=1))
    return {"saved": "universal", "locked": bool(u.get("locked"))}


@app.get("/api/options")
def options():
    """Live dropdown data from Forge: checkpoints, samplers, schedulers,
    upscalers (latent + GAN), and ADetailer detector models."""
    import urllib.request

    def get(path):
        try:
            with urllib.request.urlopen(forge_client.FORGE + path, timeout=10) as r:
                return json.loads(r.read())
        except Exception:
            return []

    checkpoints = [m.get("title") for m in get("/sdapi/v1/sd-models") if m.get("title")]
    samplers = [s.get("name") for s in get("/sdapi/v1/samplers") if s.get("name")]
    schedulers = [s.get("label") or s.get("name")
                  for s in get("/sdapi/v1/schedulers") if s.get("label") or s.get("name")]
    upscalers = [u.get("name") for u in get("/sdapi/v1/latent-upscale-modes") if u.get("name")]
    upscalers += [u.get("name") for u in get("/sdapi/v1/upscalers") if u.get("name")]
    detectors = get("/adetailer/v1/ad_model").get("ad_model", [])
    return {"checkpoints": checkpoints, "samplers": samplers,
            "schedulers": schedulers, "upscalers": upscalers,
            "detectors": detectors}


@app.get("/api/loras")
def loras():
    return lora_store.scan()


@app.post("/api/loras/enrich")
def enrich_lora(req: dict):
    try:
        e = lora_store.enrich(req.get("path", ""))
    except ValueError as ex:
        raise HTTPException(404, str(ex))
    except Exception as ex:
        raise HTTPException(502, f"civitai lookup failed: {ex}")
    e = dict(e)
    if e.get("preview_file"):
        e["preview"] = "/lora-previews/" + Path(e["preview_file"]).name
    return e


@app.post("/api/ingest")
def ingest(req: dict):
    """PNG metadata -> draft profile. Forge infotext and ComfyUI graphs."""
    import base64
    import io

    import ingest as ingest_mod
    from PIL import Image

    b64 = req.get("b64", "")
    if not b64:
        raise HTTPException(400, "missing b64 png data")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        info = img.info
    except Exception as ex:
        raise HTTPException(400, f"unreadable png: {ex}")
    if info.get("parameters"):
        parsed = ingest_mod.parse_infotext(info["parameters"])
    elif info.get("prompt"):
        try:
            parsed = ingest_mod.parse_comfy(info["prompt"])
        except Exception as ex:
            raise HTTPException(400, f"comfyui graph parse failed: {ex}")
    else:
        raise HTTPException(400, "no generation metadata in this png")

    checkpoints = []
    upscalers = []
    try:
        import urllib.request
        with urllib.request.urlopen(forge_client.FORGE + "/sdapi/v1/sd-models", timeout=10) as r:
            checkpoints = [m.get("title") for m in json.loads(r.read()) if m.get("title")]
        with urllib.request.urlopen(forge_client.FORGE + "/sdapi/v1/upscalers", timeout=10) as r:
            upscalers += [u.get("name") for u in json.loads(r.read()) if u.get("name")]
        with urllib.request.urlopen(forge_client.FORGE + "/sdapi/v1/latent-upscale-modes", timeout=10) as r:
            upscalers += [u.get("name") for u in json.loads(r.read()) if u.get("name")]
    except Exception:
        pass

    draft = ingest_mod.build_draft(parsed, checkpoints, upscalers)
    stem = re.sub(r"[^a-z0-9_-]+", "_", Path(req.get("filename", "ingested.png")).stem.lower()).strip("_")[:40] or "ingested"
    return {"profile": draft["profile"], "meta": draft["meta"],
            "suggest": {"character": "ingested", "name": stem}}


@app.get("/api/profiles")
def profiles():
    return list_profiles()


@app.get("/api/profile/{character}/{name}")
def get_profile(character: str, name: str):
    return load_profile(character, name)


@app.put("/api/profile/{character}/{name}")
def put_profile(character: str, name: str, profile: dict):
    p = profile_path(character, name)
    if p.is_file():
        # the lock is a property of the file on disk, never of the submitted form
        try:
            existing = json.loads(p.read_text())
        except Exception:
            raise HTTPException(409, "existing profile is unreadable; refusing to overwrite")
        if existing.get("locked"):
            raise HTTPException(409, "profile is locked; saving creates a variant")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=1))
    return {"saved": f"{character}/{name}"}


@app.post("/api/profile/{character}/{name}/lock")
def lock_profile(character: str, name: str, body: dict):
    p = profile_path(character, name)
    if not p.is_file():
        raise HTTPException(404, f"profile {character}/{name} not found")
    data = json.loads(p.read_text())
    data["locked"] = bool(body.get("locked"))
    p.write_text(json.dumps(data, indent=1))
    return {"character": character, "name": name, "locked": data["locked"]}


@app.post("/api/profile/{character}/{name}/variant")
def variant_profile(character: str, name: str, profile: dict):
    profile_path(character, name)  # validates keys
    profile["variant_of"] = profile.get("variant_of", name)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    new_name = f"{name}_{stamp}"
    n = 1
    while (PROFILES / character / f"{new_name}.json").is_file():
        n += 1
        new_name = f"{name}_{stamp}_{n}"
    out = PROFILES / character / f"{new_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1))
    return {"saved": f"{character}/{new_name}", "name": new_name}


@app.post("/api/generate")
def generate(req: GenerateReq):
    profile = apply_universal(req.profile if req.profile else load_profile(req.character, req.name))
    overrides = dict(req.overrides)
    if req.suffix:
        overrides["prompt"] = profile["base"]["prompt"].rstrip() + (
            ("\n" if "<lora:" in profile["base"]["prompt"] else ", ")
            + req.suffix.strip())
    payload = forge_client.build_payload(profile, overrides)
    t0 = time.time()
    result = forge_client.generate(payload)
    ticket = forge_client.save_result(
        OUTPUTS / req.character / req.name, f"{req.character}/{req.name}",
        result["image"], result["info"], payload, time.time() - t0)
    ticket["url"] = f"/outputs/{req.character}/{req.name}/{ticket['file']}"
    return ticket


@app.post("/api/batch")
def batch(req: BatchReq):
    profile = apply_universal(req.profile if req.profile else load_profile(req.character, req.name))
    base_seed = req.base_seed if req.base_seed is not None \
        else profile["base"].get("seed", -1)
    tickets = []
    for i, line in enumerate(req.lines[:64]):
        if req.seed_mode == "fixed":
            seed = base_seed
        elif req.seed_mode == "random":
            seed = -1
        else:
            seed = base_seed + i if base_seed >= 0 else -1
        payload = forge_client.build_payload(
            profile, {"seed": seed, "prompt": line and (
                profile["base"]["prompt"].rstrip() + (
                    "\n" if "<lora:" in profile["base"]["prompt"] else ", ")
                + line.strip()) or profile["base"]["prompt"]})
        t0 = time.time()
        try:
            result = forge_client.generate(payload)
        except Exception as e:  # keep the batch alive, log the failure
            tickets.append({"line": i, "error": str(e)[:200]})
            continue
        ticket = forge_client.save_result(
            OUTPUTS / req.character / req.name, f"{req.character}/{req.name}",
            result["image"], result["info"], payload, time.time() - t0)
        ticket["url"] = f"/outputs/{req.character}/{req.name}/{ticket['file']}"
        tickets.append(ticket)
    return tickets


app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
