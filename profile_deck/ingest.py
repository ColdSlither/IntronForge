"""PNG ingest: turn archived generation metadata into a draft profile.

Two source dialects, both normalized to the deck's Forge-compliant
profile schema:
- A1111/Forge infotext (the "parameters" chunk).
- ComfyUI graph JSON (the "prompt" chunk), walked for the pieces a
  character profile needs: prompts, checkpoint, sampler chain, latent
  size, loras, hires.

Per the story spec: only character-prompt-necessary fields are pulled.
Universal-ratio fields (steps, cfg, distilled, hires scale/steps/denoise/
cfg, detailer conf/denoise/pad) are set to null so the profile inherits
the universal pins.
"""
import json
import re

LORA_PARSE = re.compile(r"<lora:([^:>]+):([^>]*)>")

SAMPLER_MAP = {
    "euler_ancestral": "Euler a", "euler": "Euler", "euler_ancestral_cfg_pp": "Euler a CFG++",
    "dpmpp_2m": "DPM++ 2M", "dpmpp_sde": "DPM++ SDE", "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "DPM++ 3M SDE", "dpmpp_2s_ancestral": "DPM++ 2s a RF",
    "dpmpp_2m_cfg_pp": "DPM++ 2M CFG++", "ddim": "DDIM", "uni_pc": "UniPC",
    "lcm": "LCM", "heun": "Heun", "dpm_2": "DPM2", "lms": "LMS",
    "res_multistep": "Res Multistep", "restart": "Restart",
}
SCHEDULER_MAP = {
    "karras": "Karras", "exponential": "Exponential", "sgm_uniform": "SGM Uniform",
    "simple": "Simple", "ddim_uniform": "DDIM Uniform", "beta": "Beta",
    "linear_quadratic": "Linear Quadratic", "kl_optimal": "KL Optimal",
    "normal": "Automatic",
}

# A1111 display names -> Forge's installed upscaler names.
UPSCALER_ALIASES = {
    "R-ESRGAN 4x+ Anime6B": "RealESRGAN_x4plus_anime_6B",
    "R-ESRGAN 4x+": "RealESRGAN_x4plus",
    "R-ESRGAN 2x+": "RealESRGAN_x2plus",
    "ESRGAN_4x": "RealESRGAN_x4plus",
}


def normalize_upscaler(name: str | None, installed: list[str] | None) -> tuple[str | None, str | None]:
    """Map an infotext upscaler name to an installed one, or fall back."""
    fallback = "RealESRGAN_x4plus_anime_6B"
    if not name:
        return fallback, None
    if not installed or name in installed:
        return name, None
    alias = UPSCALER_ALIASES.get(name)
    if alias and alias in installed:
        return alias, None
    return fallback, f"hires upscaler {name!r} not installed; using {fallback}"

SETTINGS_SPLIT = re.compile(r'([A-Za-z0-9 _\-]+?):\s*("(?:[^"\\]|\\.)*"|[^,]*)(?:,|$)')


def detect_base(prompt_text: str) -> str | None:
    t = prompt_text.lower()
    if "score_9" in t or "source_cartoon" in t or "source_anime" in t:
        return "Pony"
    if "masterwork" in t or "best_quality" in t or "masterpiece" in t or "best quality" in t:
        return "Illustrious"
    return None


def parse_settings(s: str) -> dict:
    out = {}
    for m in SETTINGS_SPLIT.finditer(s):
        k = m.group(1).strip()
        v = m.group(2).strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1].replace('\\"', '"').replace("\\n", "\n")
        out[k] = v
    return out


def parse_infotext(text: str) -> dict:
    """A1111/Forge parameters chunk -> parsed pieces."""
    prompt, negative, settings_text = text, "", ""
    m = None
    for m in re.finditer(r"\nSteps: ", text):
        pass
    if m:
        settings_text = text[m.start() + 1:]
        head = text[:m.start()]
    else:
        head = text
    neg_m = re.search(r"\nNegative prompt: ", head + "\n")
    if neg_m:
        prompt = (head + "\n")[:neg_m.start()].strip("\n")
        negative = (head + "\n")[neg_m.end():].strip("\n")
    else:
        prompt = head.strip("\n")
    s = parse_settings(settings_text)
    return {"format": "a1111", "prompt": prompt, "negative_prompt": negative,
            "settings": s}


def a1111_tabs(s: dict) -> list:
    """Old ADetailer infotext keys -> deck detailer tabs."""
    tabs = []
    for suffix in ("", " 2nd", " 3rd"):
        model = s.get(f"ADetailer model{suffix}")
        if not model or model == "None":
            continue
        tabs.append({
            "enabled": True,
            "model": model,
            "prompt_override": s.get(f"ADetailer prompt{suffix}") or None,
            "negative_override": s.get(f"ADetailer negative prompt{suffix}") or None,
            "confidence": float(s[f"ADetailer confidence{suffix}"]) if s.get(f"ADetailer confidence{suffix}") else None,
            "denoise": float(s[f"ADetailer denoising strength{suffix}"]) if s.get(f"ADetailer denoising strength{suffix}") else None,
            "padding": int(float(s[f"ADetailer inpaint padding{suffix}"])) if s.get(f"ADetailer inpaint padding{suffix}") else None,
        })
    return tabs


def _walk_model_chain(g: dict, node_id) -> tuple[list, str | None]:
    """Follow model inputs upstream: collect loras, find checkpoint."""
    loras, ckpt = [], None
    seen = set()
    cur = node_id
    while cur is not None and cur not in seen:
        seen.add(cur)
        node = g.get(cur)
        if not node:
            break
        ct = node.get("class_type", "")
        ins = node.get("inputs", {})
        if ct == "LoraLoader" or "lora" in ct.lower() and "loader" in ct.lower():
            if isinstance(ins.get("lora_name"), str):
                loras.append({"name": ins["lora_name"], "weight": ins.get("strength_model", 0.8)})
            cur = _link_source(ins.get("model"))
        elif "checkpointloader" in ct.lower():
            ckpt = ins.get("ckpt_name")
            cur = None
        elif ct == "Reroute":
            cur = _link_source(ins.get("source") or next(iter(
                (v for v in ins.values() if isinstance(v, list) and len(v) == 2), None)))
        else:
            cur = _link_source(ins.get("model"))
    return loras, ckpt


def _link_source(v):
    if isinstance(v, list) and len(v) == 2:
        return v[0]
    return None


def _is_sampler_node(n: dict) -> bool:
    """Structural signature: real samplers carry steps/cfg/seed/sampler_name
    regardless of class name (KSampler, KSamplerAdvanced, KSampler_A1111...).
    Link-wired values (e.g. seed from a Primitive node) count as present."""
    ins = n.get("inputs", {})

    def present(v):
        return (isinstance(v, (int, float, str))
                or (isinstance(v, list) and len(v) == 2))

    return (present(ins.get("steps")) and present(ins.get("cfg"))
            and present(ins.get("seed"))
            and isinstance(ins.get("sampler_name"), str)
            and isinstance(ins.get("scheduler"), str))


def parse_comfy(graph_text: str) -> dict:
    """ComfyUI prompt graph -> parsed pieces (Forge-normalized)."""
    g = json.loads(graph_text)
    notes = []
    samplers = {nid: n for nid, n in g.items()
                if _is_sampler_node(n)
                and "detailer" not in n.get("class_type", "").lower()}
    if not samplers:
        raise ValueError("no KSampler node found in ComfyUI graph")

    def latent_is_empty(nid):
        return g.get(str(nid), {}).get("class_type") in (
            "EmptyLatentImage", "EmptySD3LatentImage")

    def rank(item):
        nid, n = item
        empty = 1 if latent_is_empty(_link_source(n.get("inputs", {}).get("latent_image"))) else 0
        dn = n.get("inputs", {}).get("denoise")
        high = 1 if (dn is None or (isinstance(dn, (int, float)) and dn >= 0.9)) else 0
        return (empty, high)

    ordered = sorted(samplers.items(), key=rank, reverse=True)
    primary_id, primary = ordered[0], ordered[0][1]
    hires = None
    for nid, n in ordered[1:]:
        if isinstance(n.get("inputs", {}).get("denoise"), (int, float)) and n["inputs"]["denoise"] < 0.9:
            hires = (nid, n)
            break
        if hires is None:
            hires = (nid, n)
    ins = primary.get("inputs", {})

    def text_of(link):
        src = _link_source(link)
        node = g.get(str(src), {})
        if "cliptextencode" in node.get("class_type", "").lower():
            t = node.get("inputs", {}).get("text")
            return t if isinstance(t, str) else None
        return None

    positive = text_of(ins.get("positive")) or ""
    negative = text_of(ins.get("negative")) or ""
    loras, ckpt = _walk_model_chain(g, _link_source(ins.get("model")))
    size = {}
    lat = _link_source(ins.get("latent_image"))
    lat_node = g.get(str(lat), {})
    if lat_node.get("class_type") in ("EmptyLatentImage", "EmptySD3LatentImage"):
        size = {"width": lat_node.get("inputs", {}).get("width"),
                "height": lat_node.get("inputs", {}).get("height")}
    else:
        notes.append("latent is not empty (img2img); size unknown")

    sampler = SAMPLER_MAP.get(ins.get("sampler_name"), str(ins.get("sampler_name")))
    scheduler = SCHEDULER_MAP.get(ins.get("scheduler"), str(ins.get("scheduler")))
    out = {
        "format": "comfy",
        "prompt": positive,
        "negative_prompt": negative,
        "settings": {
            "Steps": ins.get("steps"), "CFG scale": ins.get("cfg"),
            "Seed": ins.get("seed"), "Sampler": sampler, "Schedule type": scheduler,
            "Denoising strength": ins.get("denoise"),
            **({"Size": f"{size['width']}x{size['height']}"} if size else {}),
        },
        "loras": loras, "checkpoint": ckpt, "size": size or None,
        "hires": None, "tabs": [], "notes": notes,
    }
    if hires:
        h_id, h_node = hires
        h_in = h_node.get("inputs", {})
        out["hires"] = {
            "enabled": True,
            "upscaler": None,
            "denoise": h_in.get("denoise"),
            "steps": h_in.get("steps"),
            "cfg": h_in.get("cfg"),
        }
        lat_src = _link_source(h_in.get("latent_image"))
        lat_up = g.get(str(lat_src), {})
        if lat_up.get("class_type") in ("LatentUpscale", "UpscaleLatent"):
            out["hires"]["upscaler"] = lat_up.get("inputs", {}).get("upscale_method")
        for nid, n in g.items():
            if n.get("class_type") == "UpscaleModelLoader":
                name = n.get("inputs", {}).get("model_name") or ""
                out["hires"]["upscaler"] = re.sub(r"\.(pth|safetensors)$", "", name)
                break
        notes.append(f"hires pass detected on node {h_id}")
    for nid, n in g.items():
        if "detailer" in n.get("class_type", "").lower():
            notes.append(f"detailer node found ({n['class_type']}) but not parsed yet")
    return out


def build_draft(parsed: dict, checkpoints: list[str] | None = None,
                upscalers: list[str] | None = None) -> dict:
    """Parsed pieces -> draft profile (universal-ratio fields null)."""
    prompt = parsed["prompt"]
    loras = [{"name": n, "weight": float(w) if w else 0.8}
             for n, w in LORA_PARSE.findall(prompt)]
    if not loras and parsed.get("loras"):
        loras = parsed["loras"]
    prompt = LORA_PARSE.sub(" ", prompt)
    prompt = re.sub(r",\s*\n\s*(BREAK|\Z)", r"\n\1", prompt)
    prompt = re.sub(r",\s*$", "", prompt).strip()
    seen, deduped = set(), []
    for l in loras:
        if l["name"] in seen:
            continue
        seen.add(l["name"])
        deduped.append(l)
    loras = deduped
    s = parsed.get("settings", {})
    try:
        seed = int(float(s.get("Seed")))
    except (TypeError, ValueError):
        seed = -1

    ckpt = parsed.get("checkpoint") or s.get("Model")
    resolved = None
    if ckpt:
        base = ckpt.replace("\\", "/").split("/")[-1].lower()
        for c in checkpoints or []:
            if c.split("/")[-1].lower() == base:
                resolved = c
                break
    size = parsed.get("size") or {}
    if not size and s.get("Size"):
        m = re.match(r"(\d+)x(\d+)", s["Size"])
        if m:
            size = {"width": int(m.group(1)), "height": int(m.group(2))}

    tabs = parsed.get("tabs") or a1111_tabs(s) or []
    notes = list(parsed.get("notes", []))
    hires = parsed.get("hires")
    if hires is None and s.get("Hires upscaler"):
        hires = {"enabled": True, "upscaler": s.get("Hires upscaler")}
    if hires and hires.get("upscaler"):
        up, note = normalize_upscaler(hires["upscaler"], upscalers)
        hires["upscaler"] = up
        if note:
            notes.append(note)

    if ckpt and not resolved:
        notes.append(f"checkpoint {ckpt!r} not matched to an installed model; "
                     f"pick one in the editor before generating")
    for k in ("Steps", "CFG scale", "Hires steps", "Denoising strength"):
        if k in s:
            notes.append(f"{k} {s[k]} inherited from universal ratios")

    profile = {
        "locked": False,
        "loras": loras,
        "base": {
            "prompt": prompt,
            "negative_prompt": parsed.get("negative_prompt", ""),
            "checkpoint": resolved or ckpt or "YOUR_CHECKPOINT.safetensors",
            "vae": "Automatic",
            "width": size.get("width") or 768,
            "height": size.get("height") or 1024,
            "sampler": s.get("Sampler") or "Euler a",
            "scheduler": s.get("Schedule type") or "Karras",
            "steps": None, "cfg_scale": None, "distilled_cfg_scale": None,
            "seed": seed,
        },
        "hires": {
            "enabled": bool(hires and hires.get("enabled")),
            "upscaler": (hires or {}).get("upscaler") or "RealESRGAN_x4plus_anime_6B",
            "scale": None, "steps": None, "denoise": None, "cfg": None,
        },
        "detailer": {
            "strip_lora_from_prompt": True,
            "tabs": tabs,
        },
        "notes": f"ingested {parsed['format']} png" + (f"; base family {detect_base(parsed['prompt'])}" if detect_base(parsed['prompt']) else ""),
    }
    meta = {
        "format": parsed["format"],
        "base_family": detect_base(parsed["prompt"]),
        "checkpoint_resolved": bool(resolved),
        "notes": notes,
    }
    return {"profile": profile, "meta": meta}
