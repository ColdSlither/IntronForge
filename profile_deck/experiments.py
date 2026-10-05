"""Experiment harness for IntronForge.

Productizes the session's ladder investigations (tools/harness.py and
step_ladder.py) into a first-class, persisted, agent-readable feature.

Core flow:
1. Accept a base profile + axes (e.g., detailer tab denoise values).
2. Expand the axes into a matrix of override dictionaries.
3. Render each cell with the same fixed seed (unless seed is itself an axis).
4. Detect the face in each render, crop it, and compute metrics.
5. Persist a JSON results table under experiments/<id>/.
6. Let the agent/user read the table and optionally apply the winning cell.
"""
import base64
import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import forge_client


def _signature(profile: dict, axes: dict, fixed_seed: int | None,
               metrics: list[str]) -> str:
    """Deterministic signature for cache/duplicate detection."""
    payload = json.dumps({"profile": profile, "axes": axes,
                          "fixed_seed": fixed_seed, "metrics": metrics},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _expand_axes(axes: dict) -> list[dict]:
    """Turn {'a': [1,2], 'b': [3,4]} into four override dicts in
    deterministic order (sorted keys, then product)."""
    import itertools
    keys = sorted(axes.keys())
    values = [axes[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def _normalize_path(path: str) -> str:
    """Bracket syntax to dotted: 'detailer.tabs[0].denoise' -> 'detailer.tabs.0.denoise'."""
    return re.sub(r"\[(\d+)\]", r".\1", path.strip())


def _normalize_axes(axes: dict) -> dict:
    """Validate axis input shape; return axes keyed by normalized paths."""
    if not isinstance(axes, dict) or not axes:
        raise ValueError("axes must be a non-empty object mapping paths to value lists")
    out = {}
    for path, values in axes.items():
        if not isinstance(path, str) or not path.strip():
            raise ValueError("axis path must be a non-empty string")
        if not isinstance(values, list) or not values:
            raise ValueError(f"axis {path!r} needs a non-empty list of values")
        out[_normalize_path(path)] = values
    return out


def _resolve_path(obj, path: str):
    """Walk a dotted path without mutating; None when any segment is missing."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            if not part.isdigit() or int(part) >= len(cur):
                return None
            cur = cur[int(part)]
        else:
            return None
    return cur


def _validate_axes(profile: dict, axes: dict) -> None:
    """An axis path that does not resolve would silently change nothing:
    every cell would render identical to baseline and the metrics table
    would report the axis has no effect. Refuse instead."""
    for path in axes:
        if _resolve_path(profile, path) is None:
            raise ValueError(
                f"axis path {path!r} does not resolve in the profile "
                "(use e.g. detailer.tabs.0.denoise)")


def _pin_seed(profile: dict, fixed_seed: int | None) -> int:
    """One seed for baseline and every cell: the request seed, else the
    profile seed, else a fresh draw. An unpinned (-1/missing) seed would
    randomize every render and turn drift into seed noise. A negative
    fixed_seed means 'unspecified' in deck convention and falls through."""
    if fixed_seed is not None and int(fixed_seed) >= 0:
        return int(fixed_seed)
    try:
        seed = int((profile.get("base") or {}).get("seed"))
    except (TypeError, ValueError):
        seed = -1
    return seed if seed >= 0 else random.randrange(2 ** 32)


_METRIC_ALIASES = {
    "face": "face", "lab": "face", "texture_energy": "face",
    "drift": "drift", "lab_drift": "drift", "deltae": "drift",
    "ssim_vs_baseline": "drift",
}


def _canonical_metrics(names: list[str] | None) -> list[str]:
    """Map requested metric names (including the roadmap's aliases) to the
    canonical groups that are actually computed: 'face' and 'drift'."""
    if names is None:
        return ["face", "drift"]
    out = []
    for n in names:
        canonical = _METRIC_ALIASES.get(n)
        if canonical is None:
            raise ValueError(
                f"unknown metric {n!r}; known: face, drift "
                "(aliases: lab, texture_energy, lab_drift, deltae, ssim_vs_baseline)")
        if canonical not in out:
            out.append(canonical)
    return out or ["face", "drift"]


def _set_nested(obj: dict, path: str, value) -> None:
    """Set obj['a']['b'][0]['c'] given path 'a.b.0.c'."""
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = cur.setdefault(p, {})
    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = value
    else:
        cur[last] = value


def _apply_overrides(profile: dict, overrides: dict) -> dict:
    """Deep-copy a profile and apply dotted-path overrides."""
    p = copy.deepcopy(profile)
    for path, value in overrides.items():
        _set_nested(p, _normalize_path(path), value)
    return p


_FACE_MODELS: dict[str, object] = {}


def _load_face_model(model_path: Path):
    """Load the detector once per process; reloading per cell costs
    seconds of disk+init per render for no benefit."""
    key = str(model_path)
    if key not in _FACE_MODELS:
        from ultralytics import YOLO
        _FACE_MODELS[key] = YOLO(key)
    return _FACE_MODELS[key]


def _detect_face_box(image: Image.Image, model_path: Path,
                     conf: float = 0.3) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) for the first detected face, or None."""
    if not model_path.is_file():
        return None
    try:
        model = _load_face_model(model_path)
    except Exception:
        return None
    results = model.predict(np.asarray(image), conf=conf, verbose=False)
    if not results or not results[0].boxes:
        return None
    x0, y0, x1, y1 = [int(v) for v in results[0].boxes.xyxy[0].tolist()]
    return x0, y0, x1, y1


def _shrink_box(x0, y0, x1, y1, w, h, frac):
    """Inset a box by frac on each side, clamped to image bounds."""
    dx, dy = int((x1 - x0) * frac), int((y1 - y0) * frac)
    return max(0, x0 + dx), max(0, y0 + dy), min(w, x1 - dx), min(h, y1 - dy)


def _srgb_to_lab(arr: np.ndarray) -> np.ndarray:
    """arr float 0..1 RGB -> Lab (D65)."""
    a = np.clip(arr, 0, 1)
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    A = 500 * (f[..., 0] - f[..., 1])
    B = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], axis=-1)


def _texture_energy(gray: np.ndarray) -> float:
    """Mean absolute Laplacian on a grayscale array (0..1)."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(gray, (3, 3))
    return float(np.abs((win * k).sum(axis=(-2, -1))).mean())


def _mean_abs_diff(a_img: Image.Image, b_img: Image.Image) -> float:
    """Mean absolute RGB difference of two renders, min-cropped.
    Pairwise neighbor delta, ported from tools/step_ladder.py."""
    a = np.asarray(a_img.convert("RGB"), dtype=np.float64)
    b = np.asarray(b_img.convert("RGB"), dtype=np.float64)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return round(float(np.abs(a[:h, :w] - b[:h, :w]).mean()), 4)


def _compute_metrics(image: Image.Image, baseline: Image.Image | None,
                     face_model: Path | None,
                     names: list[str] | None = None) -> dict:
    """Compute requested metrics for one render. names are canonical
    groups ('face', 'drift'); see _canonical_metrics for aliases."""
    want = set(names or ["face", "drift"])
    arr = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    h, w = arr.shape[:2]
    metrics = {}

    # Face crop metrics
    if "face" in want:
        box = _detect_face_box(image, face_model) if face_model else None
        if box:
            x0, y0, x1, y1 = box
            bx0, by0, bx1, by1 = _shrink_box(x0, y0, x1, y1, w, h, 0.15)
            lab = _srgb_to_lab(arr)
            box_lab = lab[by0:by1, bx0:bx1]
            L = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
            metrics["face"] = {
                "bbox": [x0, y0, x1, y1],
                "L": round(float(box_lab[..., 0].mean()), 2),
                "a": round(float(box_lab[..., 1].mean()), 2),
                "b": round(float(box_lab[..., 2].mean()), 2),
                "texture": round(_texture_energy(L[by0:by1, bx0:bx1]), 5),
            }
        else:
            metrics["face"] = None

    # Whole-image drift vs baseline
    if "drift" in want and baseline is not None:
        barr = np.asarray(baseline.convert("RGB"), dtype=np.float64) / 255.0
        bh, bw = barr.shape[:2]
        if (bh, bw) != (h, w):
            barr = np.asarray(Image.fromarray((barr * 255).astype(np.uint8))
                              .resize((w, h), Image.LANCZOS), dtype=np.float64) / 255.0
        diff = np.abs(arr - barr).mean(axis=2)
        drift = {
            "mean_abs": round(float(diff.mean()), 4),
            "p95": round(float(np.percentile(diff, 95)), 4),
        }
        # Simple SSIM-ish structural similarity (placeholder; can swap for
        # skimage.structural_similarity later if available).
        mu_a = arr.mean(axis=(0, 1))
        mu_b = barr.mean(axis=(0, 1))
        var_a = ((arr - mu_a) ** 2).mean(axis=(0, 1))
        var_b = ((barr - mu_b) ** 2).mean(axis=(0, 1))
        cov = ((arr - mu_a) * (barr - mu_b)).mean(axis=(0, 1))
        ssim = ((2 * mu_a * mu_b + 0.01) * (2 * cov + 0.03)) / \
               ((mu_a ** 2 + mu_b ** 2 + 0.01) * (var_a + var_b + 0.03))
        drift["ssim"] = round(float(ssim.mean()), 4)

        # CIELAB dE inside the changed mask, ported from tools/harness.py:
        # changed = max-channel diff > 8/255, mask dilated 7px so the
        # measurement includes the change edges, not just interiors.
        diff_max = np.abs(arr - barr).max(axis=2)
        changed = diff_max > 8 / 255.0
        drift["changed_frac"] = round(float(changed.mean()), 4)
        if changed.any():
            mask = np.asarray(Image.fromarray((changed * 255).astype(np.uint8))
                              .filter(ImageFilter.MaxFilter(7))) > 0
            d = _srgb_to_lab(arr)[mask] - _srgb_to_lab(barr)[mask]
            de = np.sqrt((d ** 2).sum(axis=1))
            drift["dE_mean"] = round(float(de.mean()), 2)
            drift["dE_p95"] = round(float(np.percentile(de, 95)), 2)
            drift["dL"] = round(float(d[:, 0].mean()), 2)
            drift["da"] = round(float(d[:, 1].mean()), 2)
            drift["db"] = round(float(d[:, 2].mean()), 2)
        else:
            drift["dE_mean"] = drift["dE_p95"] = 0.0
            drift["dL"] = drift["da"] = drift["db"] = 0.0
        metrics["drift"] = drift

    return metrics


class Experiment:
    def __init__(self, exp_root: Path, profile: dict, character: str, name: str,
                 axes: dict, fixed_seed: int | None = None,
                 metrics: list[str] | None = None):
        self.exp_root = exp_root
        self.profile = profile
        self.character = character
        self.name = name
        self.axes = _normalize_axes(axes)
        self.fixed_seed = _pin_seed(profile, fixed_seed)
        self.metric_names = _canonical_metrics(metrics)
        self.cells = _expand_axes(self.axes)
        if len(self.cells) > 64:
            raise ValueError(
                f"experiment matrix has {len(self.cells)} cells; the cap is 64")
        _validate_axes(profile, self.axes)
        self.signature = _signature(profile, self.axes, self.fixed_seed,
                                    self.metric_names)
        self.id = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
        self.dir: Path | None = None
        self.baseline = None
        self.error = None
        self.results = []
        self.status = "pending"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "signature": self.signature,
            "character": self.character,
            "name": self.name,
            "axes": self.axes,
            "fixed_seed": self.fixed_seed,
            "metrics": self.metric_names,
            "cells": len(self.cells),
            "baseline": self.baseline,
            "results": self.results,
        }

    def save(self):
        # Atomic write: GET /api/experiment/{id} polls this file while the
        # background run rewrites it after every cell.
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "experiment.json.tmp"
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(self.dir / "experiment.json")

    def find_cache(self) -> Path | None:
        """Return the most recent cached experiment dir with the same signature.
        Before allocation self.dir is None, so we do not filter by id yet."""
        if not self.exp_root.exists():
            return None
        own_dir = self.dir
        best = None
        for d in self.exp_root.iterdir():
            if not d.is_dir():
                continue
            if own_dir is not None and d.resolve() == own_dir.resolve():
                continue
            j = d / "experiment.json"
            if not j.is_file():
                continue
            try:
                data = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("signature") == self.signature and data.get("status") == "done":
                if best is None or d.stat().st_mtime > best.stat().st_mtime:
                    best = d
        return best

    def run(self, outputs_dir: Path,
            face_model: Path = Path("/path/to/forge/models/adetailer/face_yolov8n.pt")):
        # Check for a completed cache BEFORE creating our own directory, so a
        # duplicate signature does not leave an empty experiment folder.
        cached = self.find_cache()
        if cached:
            data = json.loads((cached / "experiment.json").read_text(encoding="utf-8"))
            self.id = data.get("id", self.id)
            self.dir = cached
            self.results = data.get("results", [])
            self.baseline = data.get("baseline")
            self.status = "done"
            self.save()
            return self

        # No cache: allocate the directory only when we are about to render.
        self.dir = self.exp_root / self.id
        n = 0
        while self.dir.exists() or (self.exp_root / (self.id + ".json")).exists():
            n += 1
            self.id = f"exp_{time.strftime('%Y%m%d_%H%M%S')}_{n}"
            self.dir = self.exp_root / self.id

        self.status = "running"
        self.error = None
        self.save()

        try:
            # Render baseline first (the unmodified profile).
            baseline_profile = copy.deepcopy(self.profile)
            baseline_profile.setdefault("base", {})["seed"] = self.fixed_seed
            baseline_payload = forge_client.build_payload(baseline_profile)
            t0 = time.time()
            baseline_result = forge_client.generate(baseline_payload)
            baseline_wall = time.time() - t0
            baseline_image = baseline_result["image"]
            baseline_ticket = forge_client.save_result(
                outputs_dir, f"{self.character}/{self.name}",
                baseline_image, baseline_result["info"], baseline_payload, baseline_wall,
                ticket_profile=f"{self.character}/{self.name}",
                extra={"experiment_id": self.id, "cell": "baseline"})
            self.baseline = baseline_ticket
            self.save()

            prev_image = None
            for cell in self.cells:
                cell_profile = _apply_overrides(baseline_profile, cell)
                cell_payload = forge_client.build_payload(cell_profile)
                t0 = time.time()
                cell_result = forge_client.generate(cell_payload)
                wall = time.time() - t0
                cell_ticket = forge_client.save_result(
                    outputs_dir, f"{self.character}/{self.name}",
                    cell_result["image"], cell_result["info"], cell_payload, wall,
                    ticket_profile=f"{self.character}/{self.name}",
                    extra={"experiment_id": self.id, "cell": cell})
                metrics = _compute_metrics(cell_result["image"], baseline_image,
                                           face_model, self.metric_names)
                # Neighbor delta along a single-axis ladder (roadmap 4.4);
                # meaningless across an unordered multi-axis product.
                if len(self.axes) == 1 and prev_image is not None:
                    metrics["delta_prev"] = _mean_abs_diff(cell_result["image"],
                                                           prev_image)
                prev_image = cell_result["image"]
                self.results.append({
                    "cell": cell,
                    "ticket": cell_ticket,
                    "metrics": metrics,
                })
                self.save()

            self.status = "done"
        except Exception as ex:
            # Persist the failure so a wedged 'running' status can't lie
            # forever and the signature cache never matches a broken run.
            self.status = "error"
            self.error = f"{type(ex).__name__}: {ex}"[:300]
        self.save()
        return self


def load_experiment(exp_dir: Path) -> dict:
    return json.loads((exp_dir / "experiment.json").read_text(encoding="utf-8"))


def list_experiments(exp_root: Path) -> list[dict]:
    if not exp_root.exists():
        return []
    out = []
    for d in sorted(exp_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "experiment.json").is_file():
            try:
                out.append(load_experiment(d))
            except Exception:
                pass
    return out
