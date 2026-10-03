"""Forge client for the Profile Deck.

Builds /sdapi/v1/txt2img payloads from profiles and calls the Forge Neo API.
Owns the empirically derived rules from the 2026-10-03 investigation:
- ADetailer args must go as dicts (ADetailerArgs) at the script's args slot.
- Detailer prompts inherit the base prompt including <lora:...> tokens, which
  darkens faces; the deck strips them automatically unless a tab overrides.
- A per-request sd_vae override must be a file PATH in this Forge build.
"""
import base64
import json
import re
import time
import urllib.request
from io import BytesIO
from pathlib import Path

FORGE = "http://127.0.0.1:7860"
CHECKPOINT_DEFAULT = "YOUR_CHECKPOINT.safetensors"  # EDIT: your checkpoint
LORA_TOKEN = re.compile(r"\s*<lora:[^>]*>\s*")
LORA_PARSE = re.compile(r"<lora:([^:>]+):([^>]*)>")


STYLES_PATH = Path("/path/to/forge/extensions/StyleSelectorXL/sdxl_styles.json")  # EDIT if you use StyleSelectorXL
CN_IMAGES = Path(__file__).resolve().parent / "controlnet_images"
CN_MODES = {"Balanced": 0, "My prompt is more important": 1, "ControlNet is more important": 2}
CN_RESIZE = {"Just Resize": 0, "Crop and Resize": 1, "Resize and Fill": 2}


def load_styles() -> list:
    try:
        return json.loads(STYLES_PATH.read_text())
    except Exception:
        return []


def apply_style(prompt: str, negative: str, style_name: str | None) -> tuple[str, str]:
    """StyleSelectorXL semantics: {prompt} template wraps the positive,
    style negative is prepended to the negative."""
    if not style_name:
        return prompt, negative
    for template in load_styles():
        if template.get("name") == style_name and "{prompt}" in template.get("prompt", ""):
            wrapped = template["prompt"].replace("{prompt}", prompt)
            sneg = template.get("negative_prompt", "")
            new_neg = f"{sneg}, {negative}" if (sneg and negative) else (sneg or negative)
            return wrapped, new_neg
    return prompt, negative


def merge_lora_tokens(prompt: str, loras: list) -> str:
    """Rebuild a prompt's lora tokens: existing tokens keep their order,
    profile.loras entries set the weight and append if new."""
    tokens = [(m.group(1), m.group(2)) for m in LORA_PARSE.finditer(prompt)]
    order = [n for n, _ in tokens]
    weights = dict(tokens)
    for l in loras or []:
        n = (l.get("name") or "").strip()
        if not n:
            continue
        if n not in weights:
            order.append(n)
        weights[n] = str(l.get("weight", 0.8))
    stripped = LORA_TOKEN.sub(" ", prompt)
    stripped = re.sub(r",\s*\n\s*(BREAK|\Z)", r"\n\1", stripped)
    stripped = re.sub(r",\s*$", "", stripped).strip()
    if not order:
        return stripped
    tail = ",\n".join(f"<lora:{n}:{weights[n]}>" for n in order)
    return (stripped + "\n" + tail) if stripped else tail

AD_DEFAULTS = {
    "ad_model": "None",
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
    "ad_denoising_strength": 0.4,
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


def strip_lora(prompt: str) -> str:
    """Remove <lora:...> tokens and any dangling comma left behind."""
    stripped = LORA_TOKEN.sub(" ", prompt)
    stripped = re.sub(r",\s*\n\s*(BREAK|\Z)", r"\n\1", stripped)
    stripped = re.sub(r",\s*$", "", stripped)
    return stripped.strip().rstrip(",").strip()


def build_payload(profile: dict, overrides: dict | None = None) -> dict:
    o = overrides or {}
    base = profile["base"]
    payload = {
        "prompt": o.get("prompt", base["prompt"]),
        "negative_prompt": base.get("negative_prompt", ""),
        "seed": int(o.get("seed", base.get("seed", -1))),
        "steps": int(o.get("steps", base.get("steps", 24))),
        "cfg_scale": float(o.get("cfg_scale", base.get("cfg_scale", 6))),
        "distilled_cfg_scale": float(base.get("distilled_cfg_scale", 3.0)),
        "sampler_name": base.get("sampler", "Euler a"),
        "scheduler": base.get("scheduler", "Karras"),
        "width": int(base.get("width", 768)),
        "height": int(base.get("height", 1024)),
        "batch_size": 1,
        "n_iter": 1,
        "save_images": True,
        "send_images": True,
        "hr_additional_modules": ["Use same choices"],
        "alwayson_scripts": {},
        "override_settings": {
            "sd_model_checkpoint": base.get("checkpoint", CHECKPOINT_DEFAULT)
        },
    }
    payload["prompt"] = merge_lora_tokens(payload["prompt"], profile.get("loras"))

    hires = profile.get("hires") or {}
    if hires.get("enabled"):
        payload.update({
            "enable_hr": True,
            "hr_scale": float(hires.get("scale", 2.0)),
            "hr_upscaler": hires.get("upscaler", "RealESRGAN_x4plus_anime_6B"),
            "hr_second_pass_steps": int(hires.get("steps", 10)),
            "denoising_strength": float(hires.get("denoise", 0.45)),
            "hr_cfg": float(hires.get("cfg", 4.5)),
        })

    detailer = profile.get("detailer") or {}
    tabs = [t for t in detailer.get("tabs", []) if t.get("enabled")]
    if tabs:
        base_prompt = payload["prompt"]
        args = []
        for tab in tabs:
            ad = dict(AD_DEFAULTS)
            ad.update({
                "ad_model": tab["model"],
                "ad_confidence": float(tab.get("confidence", 0.3)),
                "ad_denoising_strength": float(tab.get("denoise", 0.4)),
                "ad_inpaint_only_masked_padding": int(tab.get("padding", 32)),
            })
            if tab.get("prompt_override"):
                ad["ad_prompt"] = tab["prompt_override"]
            elif detailer.get("strip_lora_from_prompt", True):
                ad["ad_prompt"] = strip_lora(base_prompt)
            if tab.get("negative_override"):
                ad["ad_negative_prompt"] = tab["negative_override"]
            args.append(ad)
        payload["alwayson_scripts"]["ADetailer"] = {"args": args}
    else:
        payload["alwayson_scripts"]["ADetailer"] = {"args": [dict(AD_DEFAULTS, ad_model="None")]}

    # style wrap LAST so detailer prompts derive from the pre-style prompt
    payload["prompt"], payload["negative_prompt"] = apply_style(
        payload["prompt"], payload["negative_prompt"], profile.get("style", "base"))

    cn_units = (profile.get("controlnet") or {}).get("units") or []
    cn_args = []
    for u in cn_units:
        if not u.get("enabled"):
            continue
        model = u.get("model") or "None"
        if model == "None":
            continue  # enabled unit without a model asserts server-side
        img_file = CN_IMAGES / Path(u.get("image_file", "")).name
        if not img_file.is_file():
            continue
        cn_args.append({
            "enabled": True,
            "module": u.get("module") or "None",
            "model": model,
            "weight": float(u.get("weight", 1.0)),
            "image": base64.b64encode(img_file.read_bytes()).decode(),
            "guidance_start": float(u.get("guidance_start", 0.0)),
            "guidance_end": float(u.get("guidance_end", 1.0)),
            "control_mode": CN_MODES.get(u.get("control_mode"), 0),
            "resize_mode": CN_RESIZE.get(u.get("resize_mode"), 1),
            "pixel_perfect": bool(u.get("pixel_perfect", False)),
        })
        if len(cn_args) >= 3:
            break
    if cn_args:
        payload["alwayson_scripts"]["ControlNet"] = {"args": cn_args}

    return payload


def generate(payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        FORGE + "/sdapi/v1/txt2img",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    image = Image_open(resp["images"][0])
    info = json.loads(resp["info"])
    return {"image": image, "info": info}


def Image_open(b64: str):
    from PIL import Image
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def save_result(out_dir: Path, profile_key: str, image, info: dict,
                payload: dict, wall_s: float) -> dict:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    seed = info.get("seed", -1)
    stem = f"{stamp}_{seed}"
    png = out_dir / f"{stem}.png"
    image.save(png)
    ticket = {
        "profile": profile_key,
        "file": png.name,
        "seed": seed,
        "wall_s": round(wall_s, 1),
        "effective_settings": effective_settings(payload),
        "forge_infotext": info.get("infotexts", [""])[0],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(ticket, indent=1))
    return ticket


def effective_settings(payload: dict) -> dict:
    """The exact settings that reached the engine, for the sidecar ticket."""
    ad = payload.get("alwayson_scripts", {}).get("ADetailer", {}).get("args", [])
    return {
        "prompt": payload["prompt"],
        "negative_prompt": payload["negative_prompt"],
        "checkpoint": payload["override_settings"].get("sd_model_checkpoint"),
        "size": f"{payload['width']}x{payload['height']}",
        "sampler": payload["sampler_name"],
        "scheduler": payload["scheduler"],
        "steps": payload["steps"],
        "cfg_scale": payload["cfg_scale"],
        "distilled_cfg_scale": payload["distilled_cfg_scale"],
        "seed": payload["seed"],
        "hires": {k: payload[k] for k in (
            "enable_hr", "hr_scale", "hr_upscaler", "hr_second_pass_steps",
            "denoising_strength", "hr_cfg") if k in payload} or None,
        "detailer_tabs": [
            {k: v for k, v in a.items() if k in (
                "ad_model", "ad_confidence", "ad_denoising_strength",
                "ad_inpaint_only_masked_padding", "ad_prompt")}
            for a in ad
        ],
    }
