"""LoRA store: disk scan, enrichment cache, CivitAI lookups.

The CivitAI API key is read from Forge's own config.json (custom_api_key,
written by sd-civitai-browser-plus). It is used only for metadata GET
requests and is never logged or echoed. Enrichment results are cached in
lora_cache.json so each model is hashed and fetched at most once.
"""
import hashlib
import json
import os
import threading
import re
import urllib.error
import urllib.request
from pathlib import Path

CIVITAI = "https://civitai.com"

LORA_DIR = Path("/path/to/forge/models/Lora")
FORGE_CONFIG = Path("/path/to/forge/config.json")
CACHE = Path(__file__).resolve().parent / "lora_cache.json"
PREVIEWS = Path(__file__).resolve().parent / "lora_previews"
PREVIEWS.mkdir(exist_ok=True)
_CACHE_LOCK = threading.Lock()


def _cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(c: dict):
    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(c, indent=1))
    os.replace(tmp, CACHE)


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
    try:
        if side.is_file() and side.stat().st_mtime <= p.stat().st_mtime:
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
            "no_civitai": bool(e.get("no_civitai")),
        })
    return entries


def enrich_all() -> dict:
    """Fetch missing previews for already-cached entries, then enrich the rest."""
    ok = skipped = failed = 0
    errors = []
    c = _cache()
    for rel, e in list(c.items()):
        if e.get("civitai_url") and not e.get("preview_file"):
            try:
                _fetch_preview(LORA_DIR / rel, e)
                with _CACHE_LOCK:
                    fresh = _cache()
                    fresh.setdefault(rel, {}).update(
                        {k: v for k, v in e.items() if k in ("preview_file", "no_preview")})
                    _save_cache(fresh)
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
    ok = skipped = failed = no_match = 0
    errors = []
    for e in scan():
        if e["enriched"]:
            skipped += 1
            continue
        if e.get("no_civitai"):
            no_match += 1  # known self-trained/delisted: not a failure
            continue
        try:
            enrich(e["path"])
            ok += 1
        except ValueError as ex:
            no_match += 1  # clean no-match from this pass, don't repeat it
            _ = str(ex)
        except Exception as ex:
            failed += 1
            errors.append(f"{e['name']}: {str(ex)[:80]}")
    return {"ok": ok, "skipped": skipped, "failed": failed,
            "no_match": no_match,
            "errors": errors[:10], "total": ok + skipped + failed + no_match}


def _fetch_preview(p: Path, e: dict) -> bool:
    """Download the first available version image into the preview cache."""
    if e.get("no_preview"):
        return False
    req = _auth_req(f"https://civitai.com/api/v1/model-versions/by-hash/{e['sha256']}")
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
    e["no_preview"] = True
    return False


def enrich(rel: str) -> dict:
    p = LORA_DIR / rel
    if not p.is_file():
        raise ValueError(f"lora not found: {rel}")
    with _CACHE_LOCK:
        c = _cache()
        e = c.get(rel, {})
        if e.get("civitai_url"):
            return e
    e.setdefault("sha256", _hash_of(p, rel))

    if e.get("no_civitai"):
        raise ValueError("no CivitAI match \u2014 self-trained or removed from the site")

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
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            v = json.loads(r.read())
    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            e["no_civitai"] = True
            c[rel] = e
            _save_cache(c)
            raise ValueError("no CivitAI match \u2014 self-trained or removed from the site")
        raise

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


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(_auth_req(url), timeout=30) as r:
        return json.loads(r.read())


def parse_ref(ref: str):
    """CivitAI URL or bare model ID -> ("model"|"version", id)."""
    ref = (ref or "").strip()
    if ref.isdigit():
        return "model", int(ref)
    vid = re.search(r"[?&]modelVersionId=(\d+)", ref)
    if vid:
        return "version", int(vid.group(1))
    mid = re.search(r"/models/(\d+)", ref)
    if mid:
        return "model", int(mid.group(1))
    raise ValueError("paste a civitai model link (civitai.com/models/ID) or a bare model ID")


def _auth_req(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "profile-deck/1.0"})
    key = _api_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    return req


def _preview_from_version(dest: Path, e: dict, v: dict) -> bool:
    PREVIEWS.mkdir(exist_ok=True)
    for img in v.get("images", []):
        u = img.get("url")
        if not u:
            continue
        ext = ".jpg" if ".jpg" in u.lower() or ".jpeg" in u.lower() else ".png"
        pth = PREVIEWS / (dest.stem + ext)
        try:
            rq = urllib.request.Request(u, headers={
                "User-Agent": "profile-deck/1.0", "Accept": "image/*"})
            with urllib.request.urlopen(rq, timeout=30) as r:
                pth.write_bytes(r.read())
            e["preview_file"] = pth.name
            return True
        except Exception:
            continue
    e["no_preview"] = True
    return False


def import_from_civitai(ref: str, progress=None) -> dict:
    """Download a LoRA from CivitAI into the store, with metadata + preview."""
    kind, iid = parse_ref(ref)
    if kind == "model":
        model = _get_json(f"{CIVITAI}/api/v1/models/{iid}")
        versions = model.get("modelVersions") or []
        if not versions:
            raise ValueError("model has no downloadable versions")
        v = versions[0]
        v["modelId"] = model.get("id")
    else:
        v = _get_json(f"{CIVITAI}/api/v1/model-versions/{iid}")

    files = [f for f in v.get("files", [])
             if (f.get("name") or "").lower().endswith((".safetensors", ".sft"))
             or f.get("type") == "Model"]
    if not files:
        raise ValueError("no safetensors file on this version")
    fmeta = files[0]
    fname = os.path.basename(fmeta.get("name") or "imported.safetensors")
    fname = re.sub(r'[\\/:*?"<>|]', "", fname).strip() or "imported.safetensors"
    dest = LORA_DIR / fname
    rel = dest.relative_to(LORA_DIR).as_posix()

    c = _cache()
    e = c.get(rel, {})
    want_url = f"https://civitai.com/models/{v.get('modelId')}?modelVersionId={v.get('id')}"
    dl = fmeta.get("downloadUrl") or f"{CIVITAI}/api/download/models/{v.get('id')}"
    if dest.exists():
        if e.get("civitai_url") == want_url:
            return {"exists": True, "name": dest.stem, "path": rel,
                    "message": "already installed and enriched"}
        want_kb = fmeta.get("sizeKB") or 0
        if want_kb and abs(dest.stat().st_size / 1024 - want_kb) < 3:
            e.setdefault("sha256", _sha256(dest))
            dl = None  # same bytes already on disk from a prior partial import
    size = 0
    if dl is not None:
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(_auth_req(dl), timeout=120) as r, open(part, "wb") as out:
                total = int(r.headers.get("Content-Length") or 0)
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    out.write(chunk)
                    if progress:
                        try:
                            progress(size, total, dest.stem)
                        except Exception:
                            pass
            part.rename(dest)
        except Exception:
            part.unlink(missing_ok=True)
            raise
    else:
        size = dest.stat().st_size

    e["sha256"] = _sha256(dest)
    e.update({
        "version_name": v.get("name"),
        "model_name": (v.get("model") or {}).get("name"),
        "baseModel": v.get("baseModel"),
        "trainedWords": v.get("trainedWords", []),
        "civitai_url": f"https://civitai.com/models/{v.get('modelId')}?modelVersionId={v.get('id')}",
        "size_mb": round(size / (1 << 20), 1),
    })
    if _preview_from_version(dest, e, v):
        pass
    with _CACHE_LOCK:
        c = _cache()
        c[rel] = e
        _save_cache(c)
    return {"imported": True, "name": dest.stem, "path": rel,
            "size_mb": e["size_mb"], "base_model": e["baseModel"],
            "trained_words": e["trainedWords"],
            "preview": ("/lora-previews/" + e["preview_file"]) if e.get("preview_file") else None,
            "civitai_url": e["civitai_url"]}
