#!/usr/bin/env python
"""ADetailer skin tone drift experiment harness.

Drives Forge Neo REST API (/sdapi/v1/txt2img) on port 7860.
Base ticket: winner ticket from vault (NIXES, Euler a Karras 24 steps, CFG 7,
seed 3940102728, 768x1024, hires off for matrix runs).
One variable per run, same seed everywhere. ad_save_images_before is ON so
each detailer run also writes its pre-detailer image to the output dir.

Usage: harness.py run <variant_name>
       harness.py analyze <variant_name> [<variant_name> ...]
"""
import base64
import json
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CROPS = RESULTS / "crops"
OUTDIR = Path("/home/rell/sd-webui-forge-neo/output/txt2img-images")
BASE = "http://127.0.0.1:7860"

WINNER_POS = """score_9, score_8_up, score_7_up, score_6_up, source_cartoon, 1girl, solo,
85mm portrait, medium shot, three-quarter view, eye-level,
standing, relaxed posture, weight shifted,
direct eye contact looking at viewer,
soft confident expression,
single catchlight,
BREAK
(mature woman:1.1), (in her mid 30s:1.2), slim dancer body, long legs,
long red hair, loose tendrils framing face,
(blue eyes:1.3), adult face,
(white buttoned blouse:1.4), (fitted pencil skirt:1.3), gold belt, delicate gold necklace with pendant,
BREAK
modern office interior, daytime, diffused window light, softbox key camera left,
<lora:incase_style_v3_ponyxl:0.8>"""

WINNER_NEG = """score_6, score_5, score_4, lowres, bad anatomy, bad hands, signature, watermarks, ugly, error, extra limb, missing limbs, bad art, bad painting, bad photo, bad image, deformed body, merged limbs, badly drawn face, ugly face, cross-eyed, young woman, 20s, child, teen, teenage, underage, youthful, baby face, chibi, masculine features, blue streaks in hair, fit body, flat chest, cars, automobiles, harsh sunlight, overhead fluorescent, hard shadows, high contrast, overexposed highlights, multiple catchlights, monochrome, sketch"""

CHECKPOINT = "Pony/NIXES_v5.5.43.safetensors"
SEED = 3940102728

# ADetailer tab-1 args, banked starting ticket. Keys match ADetailerArgs.
def ad_args(**over):
    d = {
        "ad_model": "face_yolov8n.pt",
        "ad_model_classes": "",
        "ad_tab_enable": True,
        "ad_prompt": "",
        "ad_negative_prompt": "",
        "ad_confidence": 0.3,
        "ad_mask_min_ratio": 0.0,
        "ad_mask_max_ratio": 1.0,
        "ad_mask_k": 0,
        "ad_mask_filter_method": "Area",
        "ad_dilate_erode": 4,
        "ad_x_offset": 0,
        "ad_y_offset": 0,
        "ad_mask_merge_invert": "None",
        "ad_mask_blur": 4,
        "ad_denoising_strength": 0.45,
        "ad_inpaint_only_masked": True,
        "ad_inpaint_only_masked_padding": 32,
        "ad_use_inpaint_width_height": False,
        "ad_inpaint_width": 512,
        "ad_inpaint_height": 512,
        "ad_use_steps": False,
        "ad_steps": 20,
        "ad_use_cfg_scale": False,
        "ad_cfg_scale": 4.0,
        "ad_use_checkpoint": False,
        "ad_checkpoint": None,
        "ad_use_vae": False,
        "ad_vae": None,
        "ad_use_modules": False,
        "ad_modules": None,
        "ad_use_sampler": False,
        "ad_sampler": "Use same sampler",
        "ad_scheduler": "Use same scheduler",
        "ad_use_noise_multiplier": False,
        "ad_noise_multiplier": 1.0,
        "ad_restore_face": False,
        "ad_controlnet_model": "None",
        "ad_controlnet_module": "None",
        "ad_controlnet_weight": 1.0,
        "ad_controlnet_guidance_start_end": [0.0, 1.0],
        "is_api": True,
    }
    d.update(over)
    return d


VARIANTS = {
    # baseline: no detailer pass at all
    "base_off": dict(ad=None),
    # banked starting ticket as-is
    "base_on": dict(ad=ad_args()),
    # denoise ladder (commanded 0.3-0.6, plus 0.0 pipeline diagnostic)
    "dn000": dict(ad=ad_args(ad_denoising_strength=0.0)),
    "dn030": dict(ad=ad_args(ad_denoising_strength=0.3)),
    "dn040": dict(ad=ad_args(ad_denoising_strength=0.4)),
    "dn050": dict(ad=ad_args(ad_denoising_strength=0.5)),
    "dn060": dict(ad=ad_args(ad_denoising_strength=0.6)),
    # suspect 1: forced external VAE on base and detailer.
    # Neo's per-request sd_vae override goes through reload_vae_weights,
    # which needs a file PATH, not the display name.
    "vae_ext": dict(ad=ad_args(), overrides={"sd_vae": "/home/rell/sd-webui-forge-neo/models/VAE/sdxl_vae.safetensors"}),
    # suspect 3 (inverted): default already carries the LoRA into the
    # detailer pass via inherited prompt, so the test is the detailer
    # WITHOUT it: same prompt text, lora token stripped.
    "lora_ad_no": dict(ad=ad_args(ad_prompt=WINNER_POS.replace(
        "\n<lora:incase_style_v3_ponyxl:0.8>", ""))),
    # suspect 4: detailer CFG lowered to 4.0
    "cfg40": dict(ad=ad_args(ad_use_cfg_scale=True, ad_cfg_scale=4.0)),
    # suspect 6: alternate detectors
    "det_v8s": dict(ad=ad_args(ad_model="face_yolov8s.pt")),
    "det_mp": dict(ad=ad_args(ad_model="mediapipe_face_short.tflite")),
    # combos: LoRA stripped from detailer prompt + lowest useful denoise
    "comboA": dict(ad=ad_args(
        ad_prompt=WINNER_POS.replace("\n<lora:incase_style_v3_ponyxl:0.8>", ""),
        ad_denoising_strength=0.4)),
    "comboB": dict(ad=ad_args(
        ad_prompt=WINNER_POS.replace("\n<lora:incase_style_v3_ponyxl:0.8>", ""),
        ad_denoising_strength=0.3)),
    # full winner ticket (hires fix on) with the combo detailer
    "final": dict(ad=ad_args(
        ad_prompt=WINNER_POS.replace("\n<lora:incase_style_v3_ponyxl:0.8>", ""),
        ad_denoising_strength=0.4),
        t2i={
            "enable_hr": True,
            "hr_scale": 2.0,
            "hr_upscaler": "RealESRGAN_x4plus_anime_6B",
            "hr_second_pass_steps": 15,
            "denoising_strength": 0.5,
            "hr_cfg": 4.5,
            "hr_distilled_cfg": 3.0,
        }),
}


def build_payload(name):
    v = VARIANTS[name]
    p = {
        "prompt": WINNER_POS,
        "negative_prompt": WINNER_NEG,
        "seed": SEED,
        "steps": 24,
        "cfg_scale": 7,
        "sampler_name": "Euler a",
        "scheduler": "Karras",
        "width": 768,
        "height": 1024,
        "batch_size": 1,
        "n_iter": 1,
        "save_images": True,
        "send_images": True,
        "hr_additional_modules": ["Use same choices"],
        "override_settings": {"sd_model_checkpoint": CHECKPOINT},
    }
    if name == "base_off":
        p["alwayson_scripts"] = {"ADetailer": {"args": [ad_args(ad_model="None")]}}
    else:
        p["alwayson_scripts"] = {"ADetailer": {"args": [v["ad"]]}}
    if v.get("overrides"):
        p["override_settings"].update(v["overrides"])
    if v.get("t2i"):
        p.update(v["t2i"])
    return p


def api_post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def snapshot_dir():
    s = set()
    if OUTDIR.exists():
        for d in OUTDIR.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    s.add((f.stat().st_mtime_ns, str(f)))
    return s


def new_files(before_snap):
    s = snapshot_dir()
    return [Path(x[1]) for x in sorted(s - before_snap)]


def run(name):
    payload = build_payload(name)
    snap = snapshot_dir()
    t0 = time.time()
    resp = api_post("/sdapi/v1/txt2img", payload)
    dt = time.time() - t0
    info = json.loads(resp["info"])
    img = Image.open(BytesIO(base64.b64decode(resp["images"][0]))).convert("RGB")
    after_path = CROPS / f"{name}_after_full.png"
    img.save(after_path)
    # ADetailer before-image(s) land on disk when a detailer pass ran
    befores = [f for f in new_files(snap) if "-ad-before" in f.name]
    before_path = None
    if befores:
        before_path = CROPS / f"{name}_before_full.png"
        Image.open(befores[0]).convert("RGB").save(before_path)
        befores[0].rename(befores[0])  # keep original on disk
    rec = {
        "variant": name,
        "seed": info["seed"],
        "infotext": info["infotexts"][0],
        "before_file": str(before_path) if before_path else None,
        "after_file": str(after_path),
        "wall_s": round(dt, 1),
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(f"[{name}] done in {dt:.0f}s  before={'yes' if before_path else 'no'}")


# ---------------- analysis ----------------

def srgb_to_lab(arr):
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


def analyze(name):
    rec = json.loads((RESULTS / f"{name}.json").read_text())
    after = np.asarray(Image.open(rec["after_file"]), dtype=np.float64) / 255.0
    if not rec["before_file"]:
        (RESULTS / f"{name}_metrics.json").write_text(json.dumps({"variant": name, "detailer": False}))
        print(f"[{name}] no detailer pass, stored as baseline")
        return
    before = np.asarray(Image.open(rec["before_file"]), dtype=np.float64) / 255.0
    h, w = before.shape[:2]
    if after.shape[:2] != (h, w):
        after = np.asarray(Image.open(rec["after_file"]).resize((w, h), Image.LANCZOS), dtype=np.float64) / 255.0

    diff = np.abs(before - after).max(axis=2)
    changed = diff > 8 / 255.0
    frac = changed.mean()
    mask_img = Image.fromarray((changed * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
    mask = np.asarray(mask_img) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        print(f"[{name}] nothing changed")
        return
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

    lb = srgb_to_lab(before)
    la = srgb_to_lab(after)
    sel = mask
    d = la[sel] - lb[sel]
    de = np.sqrt((d ** 2).sum(axis=1))
    met = {
        "variant": name,
        "detailer": True,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "changed_frac": round(float(frac), 4),
        "pix": int(sel.sum()),
        "L_before": round(float(lb[sel][:, 0].mean()), 2),
        "L_after": round(float(la[sel][:, 0].mean()), 2),
        "a_before": round(float(lb[sel][:, 1].mean()), 2),
        "a_after": round(float(la[sel][:, 1].mean()), 2),
        "b_before": round(float(lb[sel][:, 2].mean()), 2),
        "b_after": round(float(la[sel][:, 2].mean()), 2),
        "dL": round(float(d[:, 0].mean()), 2),
        "da": round(float(d[:, 1].mean()), 2),
        "db": round(float(d[:, 2].mean()), 2),
        "dE_mean": round(float(de.mean()), 2),
        "dE_p95": round(float(np.percentile(de, 95)), 2),
    }
    (RESULTS / f"{name}_metrics.json").write_text(json.dumps(met, indent=1))

    # side-by-side crop: before | after, 2x zoom
    my, mx = int((y1 - y0) * 0.25) + 8, int((x1 - x0) * 0.25) + 8
    cy0, cy1 = max(0, y0 - my), min(h, y1 + my)
    cx0, cx1 = max(0, x0 - mx), min(w, x1 + mx)
    cb = (before[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)
    ca = (after[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)
    strip = np.concatenate([cb, np.full((cb.shape[0], 6, 3), 255, np.uint8), ca], axis=1)
    Image.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.NEAREST).save(
        CROPS / f"{name}_compare.png")
    print(json.dumps(met))


if __name__ == "__main__":
    cmd = sys.argv[1]
    CROPS.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:]
    if cmd == "run":
        for n in names:
            run(n)
    elif cmd == "analyze":
        for n in names:
            analyze(n)
    else:
        raise SystemExit(f"unknown cmd {cmd}")
