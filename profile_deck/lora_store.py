"""LoRA store: disk scan, enrichment cache, CivitAI lookups.

The CivitAI API key is read from Forge's own config.json (custom_api_key,
written by sd-civitai-browser-plus). It is used only for metadata GET
requests and is never logged or echoed. Enrichment results are cached in
lora_cache.json so each model is hashed and fetched at most once.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

LORA_DIR = Path("/home/rell/sd-webui-forge-neo/models/Lora")
FORGE_CONFIG = Path("/home/rell/sd-webui-forge-neo/config.json")
CACHE = Path(__file__).resolve().parent / "lora_cache.json"
PREVIEWS = Path(__file__).resolve().parent / "lora_previews"
PREVIEWS.mkdir(exist_ok=True)


def _cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(c: dict):
    CACHE.write_text(json.dumps(c, indent=1))


def _api_key() -> str | None:
    try:
        return json.loads(FORGE_CONFIG.read_text()).get("custom_api_key") or None
    except Exception:
        return None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan() -> list:
    c = _cache()
    entries = []
    for p in sorted(LORA_DIR.rglob("*.safetensors")):
        rel = p.relative_to(LORA_DIR).as_posix()
        e = c.get(rel, {})
        preview = None
        if e.get("preview_file"):
            preview = "/lora-previews/" + Path(e["preview_file"]).name
        else:
            pv = p.with_name(p.stem + ".preview.png")
            if pv.is_file():
                preview = "/loras-files/" + pv.relative_to(LORA_DIR).as_posix()
        entries.append({
            "name": p.stem,
            "path": rel,
            "base_model": e.get("baseModel"),
            "trained_words": e.get("trainedWords", []),
            "civitai_url": e.get("civitai_url"),
            "version": e.get("version_name"),
            "preview": preview,
            "enriched": bool(e.get("civitai_url")),
        })
    return entries


def enrich(rel: str) -> dict:
    p = LORA_DIR / rel
    if not p.is_file():
        raise ValueError(f"lora not found: {rel}")
    c = _cache()
    e = c.get(rel, {})
    if e.get("civitai_url"):
        return e
    e.setdefault("sha256", _sha256(p))

    req = urllib.request.Request(
        f"https://civitai.com/api/v1/model-versions/by-hash/{e['sha256']}",
        headers={"User-Agent": "profile-deck/1.0"})
    key = _api_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=30) as r:
        v = json.loads(r.read())

    PREVIEWS.mkdir(exist_ok=True)
    for img in v.get("images", []):
        u = img.get("url")
        if not u:
            continue
        ext = ".jpg" if ".jpg" in u.lower() or ".jpeg" in u.lower() else ".png"
        dest = PREVIEWS / (p.stem + ext)
        try:
            urllib.request.urlretrieve(u, dest)
            e["preview_file"] = dest.name
            break
        except Exception:
            continue

    e.update({
        "version_name": v.get("name"),
        "model_name": (v.get("model") or {}).get("name"),
        "baseModel": v.get("baseModel"),
        "trainedWords": v.get("trainedWords", []),
        "civitai_url": f"https://civitai.com/models/{v.get('modelId')}?modelVersionId={v.get('id')}",
    })
    c[rel] = e
    _save_cache(c)
    return e
