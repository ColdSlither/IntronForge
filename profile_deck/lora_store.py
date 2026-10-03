"""LoRA store: disk scan, enrichment cache, CivitAI lookups.

The CivitAI API key is read from Forge's own config.json (custom_api_key,
written by sd-civitai-browser-plus). It is used only for metadata GET
requests and is never logged or echoed. Enrichment results are cached in
lora_cache.json so each model is hashed and fetched at most once.
"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

LORA_DIR = Path("/path/to/forge/models/Lora")  # EDIT: your Forge LoRA folder
FORGE_CONFIG = Path("/path/to/forge/config.json")  # EDIT: holds your CivitAI key
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


def _hash_of(p: Path, rel: str) -> str:
    """Reuse the ComfyUI-side sidecar hash when present (saves a full read)."""
    side = p.with_name(p.stem + ".metadata.json")
    if side.is_file():
        try:
            h = json.loads(side.read_text()).get("sha256")
            if h and re.fullmatch(r"[0-9a-f]{64}", str(h)):
                return str(h)
        except Exception:
            pass
    return _sha256(p)


def scan() -> list:
    c = _cache()
    entries = []
    for p in sorted(LORA_DIR.rglob("*.safetensors")):
        rel = p.relative_to(LORA_DIR).as_posix()
        if rel.startswith(".cache") or p.name.startswith("._"):
            continue
        e = c.get(rel, {})
        preview = None
        if e.get("preview_file"):
            preview = "/lora-previews/" + Path(e["preview_file"]).name
        else:
            pv = p.with_name(p.stem + ".preview.png")
            if pv.is_file():
                preview = "/loras-files/" + pv.relative_to(LORA_DIR).as_posix()
        parent = p.parent.relative_to(LORA_DIR).as_posix()
        top = parent.split("/")[0] if "/" in parent else parent
        folder = "(root)" if top == "." else top
        entries.append({
            "name": p.stem,
            "path": rel,
            "folder": folder,
            "base_model": e.get("baseModel"),
            "trained_words": e.get("trainedWords", []),
            "civitai_url": e.get("civitai_url"),
            "version": e.get("version_name"),
            "preview": preview,
            "enriched": bool(e.get("civitai_url")),
        })
    return entries


def enrich_all() -> dict:
    """Fetch missing previews for already-cached entries, then enrich the rest."""
    ok = skipped = failed = 0
    errors = []
    c = _cache()
    for rel, e in c.items():
        if e.get("civitai_url") and not e.get("preview_file"):
            try:
                _fetch_preview(LORA_DIR / rel, e)
                _save_cache(c)
                ok += 1
            except Exception as ex:
                failed += 1
                errors.append(f"{Path(rel).stem}: preview {str(ex)[:60]}")
        elif not e.get("civitai_url"):
            skipped += 1
    r2 = enrich_pass()
    return {"ok": ok + r2["ok"], "skipped": skipped + r2["skipped"],
            "failed": failed + r2["failed"], "errors": (errors + r2["errors"])[:10],
            "total": ok + skipped + failed + r2["total"]}


def enrich_pass() -> dict:
    ok = skipped = failed = 0
    errors = []
    for e in scan():
        if e["enriched"]:
            skipped += 1
            continue
        try:
            enrich(e["path"])
            ok += 1
        except Exception as ex:
            failed += 1
            errors.append(f"{e['name']}: {str(ex)[:80]}")
    return {"ok": ok, "skipped": skipped, "failed": failed,
            "errors": errors[:10], "total": ok + skipped + failed}


def _fetch_preview(p: Path, e: dict) -> bool:
    """Download the first available version image into the preview cache."""
    req = urllib.request.Request(
        f"https://civitai.com/api/v1/model-versions/by-hash/{e['sha256']}",
        headers={"User-Agent": "profile-deck/1.0"})
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
            rq = urllib.request.Request(u, headers={
                "User-Agent": "profile-deck/1.0", "Accept": "image/*"})
            with urllib.request.urlopen(rq, timeout=30) as r:
                dest.write_bytes(r.read())
            e["preview_file"] = dest.name
            return True
        except Exception:
            continue
    return False


def enrich(rel: str) -> dict:
    p = LORA_DIR / rel
    if not p.is_file():
        raise ValueError(f"lora not found: {rel}")
    c = _cache()
    e = c.get(rel, {})
    if e.get("civitai_url"):
        return e
    e.setdefault("sha256", _hash_of(p, rel))

    try:
        if _fetch_preview(p, e):
            pass
    except Exception:
        pass

    req = urllib.request.Request(
        f"https://civitai.com/api/v1/model-versions/by-hash/{e['sha256']}",
        headers={"User-Agent": "profile-deck/1.0"})
    key = _api_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=30) as r:
        v = json.loads(r.read())

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
