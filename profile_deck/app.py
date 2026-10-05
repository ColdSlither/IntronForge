"""Profile Deck: local web app tunneling into Forge Neo's REST API.

Bind 127.0.0.1 only. Run with the Forge venv python:
    /path/to/forge/venv/bin/python -m uvicorn app:app \
        --host 127.0.0.1 --port 7877
"""
import json
import re
import time
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import forge_client
import linter
import lora_store
import tags as tag_assist
import translator

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "profiles"
OUTPUTS = ROOT / "outputs"
EXPERIMENTS = ROOT / "experiments"
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

app = FastAPI(title="IntronForge")
app.mount("/loras-files", StaticFiles(directory=str(lora_store.LORA_DIR)), name="loras")
app.mount("/lora-previews", StaticFiles(directory=str(lora_store.PREVIEWS)), name="lora-previews")
CN_IMAGES = ROOT / "controlnet_images"
CN_IMAGES.mkdir(exist_ok=True)
app.mount("/cn-images", StaticFiles(directory=str(CN_IMAGES)), name="cn-images")


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


class ExperimentReq(BaseModel):
    character: str
    name: str
    axes: dict
    fixed_seed: int | None = None
    metrics: list[str] | None = None
    profile: dict | None = None  # ephemeral form state; never persisted


class ExperimentApplyReq(BaseModel):
    cell: dict


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
    for k, d in (("steps", 24), ("cfg_scale", 6), ("distilled_cfg_scale", 3.0)):
        if b.get(k) is None:
            b[k] = u.get(k, d)
    h = profile.get("hires")
    if isinstance(h, dict):
        for k, uk, d in (("scale", "hr_scale", 2.0), ("steps", "hr_steps", 10),
                         ("denoise", "hr_denoise", 0.45), ("cfg", "hr_cfg", 4.5)):
            if h.get(k) is None:
                h[k] = u.get(uk, d)
    for t in (profile.get("detailer") or {}).get("tabs", []):
        if isinstance(t, dict):
            for k, d in (("confidence", 0.3), ("denoise", 0.4), ("padding", 32)):
                if t.get(k) is None:
                    t[k] = u.get(k, d)
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
    r = get("/adetailer/v1/ad_model")
    detectors = r.get("ad_model", []) if isinstance(r, dict) else []
    return {"checkpoints": checkpoints, "samplers": samplers,
            "schedulers": schedulers, "upscalers": upscalers,
            "detectors": detectors}


@app.get("/api/loras")
def loras():
    return lora_store.scan()


@app.post("/api/loras/enrich-all")
def enrich_all():
    """Fetch previews for cached entries missing them, enrich the rest."""
    import time
    t0 = time.time()
    result = lora_store.enrich_all()
    result["wall_s"] = round(time.time() - t0, 1)
    return result


@app.post("/api/loras/enrich")
def enrich_lora(req: dict):
    try:
        e = lora_store.enrich(req.get("path", ""))
    except ValueError as ex:
        msg = str(ex)
        if "no CivitAI match" in msg:
            # a known no-match is state, not an error: self-trained or
            # delisted models are already flagged and skipped by bulk runs
            return {"no_civitai": True, "name": Path(req.get("path", "")).stem,
                    "message": "no CivitAI match (self-trained or delisted); "
                               "flagged, bulk runs skip it"}
        raise HTTPException(404, msg)
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
        # Sidecar fallback: a downstream tool may have stripped the PNG text
        # chunks. Look for a companion `.intronforge.json` next to the file,
        # then anywhere in the outputs tree (the browser only reports the
        # bare file name, so deck renders are found by stem).
        sidecar = ingest_mod.read_sidecar(req.get("filename", ""))
        if not sidecar.get("parameters"):
            stem = Path(req.get("filename", "")).stem.split(".")[0]
            if stem and not set(stem) & set("[]*?"):
                for hit in OUTPUTS.rglob(f"{stem}.intronforge.json"):
                    sidecar = ingest_mod.read_sidecar(str(hit))
                    if sidecar.get("parameters"):
                        break
        if sidecar.get("parameters"):
            parsed = ingest_mod.parse_infotext(sidecar["parameters"])
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


@app.post("/api/sections/parse")
def sections_parse(req: dict):
    import sections as section_mod
    return section_mod.parse(req.get("prompt", ""))


@app.post("/api/sections/compose")
def sections_compose(req: dict):
    import sections as section_mod
    return {"prompt": section_mod.compose(req.get("sections", {}))}


@app.post("/api/lint")
def lint(req: dict):
    return linter.lint(req.get("prompt", ""), req.get("negative_prompt", ""),
                       req.get("protected", []))


@app.post("/api/lint/apply")
def lint_apply(req: dict):
    return linter.apply(req.get("prompt", ""), req.get("negative_prompt", ""),
                        req.get("selected", []))


@app.post("/api/translate")
def translate(req: dict):
    """Normalize a prompt to the configured lane. Receipt only; no side effects."""
    return translator.translate(req.get("prompt", ""),
                                req.get("negative_prompt", ""),
                                req.get("loras", []))


SNIPPETS = ROOT / "snippets.json"


@app.get("/api/snippets")
def snippets_list(section: str = ""):
    data = json.loads(SNIPPETS.read_text()) if SNIPPETS.is_file() else {"snippets": []}
    out = data.get("snippets", [])
    if section:
        out = [s for s in out if s.get("section") == section]
    return out


@app.post("/api/snippets")
def snippets_save(req: dict):
    data = json.loads(SNIPPETS.read_text()) if SNIPPETS.is_file() else {"snippets": []}
    import time as _t
    snip = {
        "id": f"snap_{int(_t.time()*1000)}",
        "name": (req.get("name") or "").strip()[:60] or "snippet",
        "section": req.get("section", ""),
        "character": req.get("character"),
        "lines": [l for l in (req.get("lines") or []) if l.strip()],
    }
    if not snip["lines"]:
        raise HTTPException(400, "snippet has no lines")
    data.setdefault("snippets", []).append(snip)
    SNIPPETS.write_text(json.dumps(data, indent=1))
    return {"saved": snip["id"], "name": snip["name"]}


@app.delete("/api/snippets/{snippet_id}")
def snippets_delete(snippet_id: str):
    data = json.loads(SNIPPETS.read_text()) if SNIPPETS.is_file() else {"snippets": []}
    before = len(data.get("snippets", []))
    data["snippets"] = [s for s in data.get("snippets", []) if s.get("id") != snippet_id]
    SNIPPETS.write_text(json.dumps(data, indent=1))
    return {"deleted": before - len(data["snippets"])}


@app.get("/api/tags/suggest")
def tags_suggest(q: str = ""):
    """Local danbooru autocomplete over the on-disk tag dump."""
    return tag_assist.suggest(q)


@app.get("/api/styles")
def styles():
    return [{"name": s.get("name")} for s in forge_client.load_styles() if s.get("name")]


@app.get("/api/controlnet/options")
def controlnet_options():
    import urllib.request
    out = {"models": [], "modules": []}
    for key, path in (("models", "/controlnet/model_list"), ("modules", "/controlnet/module_list")):
        try:
            with urllib.request.urlopen(forge_client.FORGE + path, timeout=10) as r:
                out[key] = json.loads(r.read()).get(path.rsplit("/", 1)[-1], [])
        except Exception:
            pass
    return out


@app.post("/api/controlnet-image")
def controlnet_image(req: dict):
    import base64 as b64mod

    from PIL import Image as PILImage
    try:
        data = b64mod.b64decode(str(req.get("b64", "")).split(",")[-1])
    except Exception:
        raise HTTPException(400, "unreadable image data")
    key = f"{req.get('character', 'x')}_{req.get('name', 'y')}_u{req.get('unit', 1)}"
    safe = re.sub(r"[^a-z0-9_-]", "_", key.lower())
    p = ROOT / "controlnet_images" / f"{safe}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    try:
        im = PILImage.open(p)
        im.thumbnail((1024, 1024))
        im.save(p)
    except Exception:
        p.unlink(missing_ok=True)
        raise HTTPException(400, "not a valid image")
    return {"image_file": p.name, "url": f"/cn-images/{p.name}"}


@app.get("/api/outputs/{character}/{name}")
def outputs_list(character: str, name: str):
    if not (KEY_RE.match(character) and KEY_RE.match(name)):
        raise HTTPException(400, "bad profile key")
    d = OUTPUTS / character / name
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
    out = []
    for f in files:
        seed = None
        side = f.with_suffix(".json")
        if side.is_file():
            try:
                seed = json.loads(side.read_text()).get("seed")
            except Exception:
                pass
        out.append({"file": f.name, "seed": seed,
                    "url": f"/outputs/{character}/{name}/{f.name}"})
    return out


THUMBS = ROOT / "thumbs"
THUMBS.mkdir(exist_ok=True)
import store as index_store


def _lib_index() -> dict:
    return index_store.load()


def _save_lib_index(items: dict):
    index_store.save(items)


def _lib_key(character: str, name: str, file: str) -> str:
    return f"{character}/{name}/{file}"


def _validated_lib_path(character: str, name: str, file: str) -> Path:
    if not (KEY_RE.match(character) and KEY_RE.match(name)):
        raise HTTPException(400, "bad profile key")
    if "/" in file or "\\" in file or not file.lower().endswith(".png"):
        raise HTTPException(400, "bad file name")
    p = OUTPUTS / character / name / file
    if not p.is_file():
        raise HTTPException(404, "render not found")
    return p


@app.get("/api/library")
def library(character: str = "", name: str = "", limit: int = 120, offset: int = 0,
            min_rating: int = 0, favorites: int = 0, tag: str = "", folder: str = ""):
    """Every render across all characters/profiles, newest first, merged
    with the curation index (rating/favorite/tags/folders)."""
    index = _lib_index()
    entries = []
    for char_dir in sorted(OUTPUTS.iterdir()):
        if not char_dir.is_dir():
            continue
        if character and char_dir.name != character:
            continue
        for prof_dir in sorted(char_dir.iterdir()):
            if not prof_dir.is_dir():
                continue
            if name and prof_dir.name != name:
                continue
            for f in prof_dir.glob("*.png"):
                key = _lib_key(char_dir.name, prof_dir.name, f.name)
                meta = index.get(key, {})
                rating = meta.get("rating", 0)
                if min_rating and rating < min_rating:
                    continue
                if favorites and not meta.get("favorite"):
                    continue
                if folder and folder not in meta.get("folders", []):
                    continue
                tags = meta.get("tags", [])
                wd14 = meta.get("wd14", [])
                tag_l = tag.lower()
                if tag and tag_l not in [t.lower() for t in tags] and tag_l not in [t.lower() for t in wd14]:
                    continue
                side = f.with_suffix(".json")
                seed = None
                if side.is_file():
                    try:
                        seed = json.loads(side.read_text()).get("seed")
                    except Exception:
                        pass
                entries.append({
                    "character": char_dir.name,
                    "name": prof_dir.name,
                    "file": f.name,
                    "seed": seed,
                    "mtime": int(f.stat().st_mtime),
                    "rating": rating,
                    "favorite": bool(meta.get("favorite")),
                    "tags": tags,
                    "folders": meta.get("folders", []),
                    "title": meta.get("title", ""),
                    "wd14": wd14[:8],
                    "url": f"/outputs/{char_dir.name}/{prof_dir.name}/{f.name}",
                })
    entries.sort(key=lambda e: (-(e["favorite"]), -e["rating"], -e["mtime"]))
    total = len(entries)
    return {"total": total, "entries": entries[offset:offset + limit]}


@app.post("/api/library/rate")
def library_rate(req: dict):
    """Curation state lives in the library index; sidecars stay write-once."""
    character, name, file = req.get("character", ""), req.get("name", ""), req.get("file", "")
    _validated_lib_path(character, name, file)
    items = _lib_index()
    key = _lib_key(character, name, file)
    meta = items.get(key, {"rating": 0, "favorite": False, "tags": []})
    if "rating" in req:
        try:
            r = int(req["rating"])
        except (TypeError, ValueError):
            raise HTTPException(400, "rating must be a number 0-5")
        if not 0 <= r <= 5:
            raise HTTPException(400, "rating must be 0-5")
        meta["rating"] = r
    if "favorite" in req:
        meta["favorite"] = bool(req["favorite"])
    if "tags" in req:
        tags = req["tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise HTTPException(400, "tags must be a list of strings")
        meta["tags"] = [t.strip() for t in tags if t.strip()][:20]
    if "title" in req:
        title = req["title"]
        if title is not None and not isinstance(title, str):
            raise HTTPException(400, "title must be a string")
        meta["title"] = (title or "").strip()[:80]
    def _apply(items):
        items[key] = meta
    index_store.merge_and_save(_apply)
    return {"key": key, **meta}


@app.get("/api/library/thumb")
def library_thumb(character: str, name: str, file: str):
    """256px cached thumbnail; originals never leave the outputs mount."""
    if not (KEY_RE.match(character) and KEY_RE.match(name)):
        raise HTTPException(400, "bad profile key")
    if "/" in file or "\\" in file or not file.lower().endswith(".png"):
        raise HTTPException(400, "bad file name")
    src = OUTPUTS / character / name / file
    if not src.is_file():
        raise HTTPException(404, "render not found")
    import hashlib
    th = THUMBS / (hashlib.md5(f"{character}/{name}/{file}".encode()).hexdigest() + ".jpg")
    if not th.is_file():
        from PIL import Image as PILImage
        im = PILImage.open(src).convert("RGB")
        im.thumbnail((256, 256))
        im.save(th, "JPEG", quality=80)
    return FileResponse(th, media_type="image/jpeg")


@app.post("/api/library/index-hashes")
def library_index_hashes():
    """Compute perceptual hashes for every render missing one."""
    import phash as phash_mod
    items = _lib_index()
    indexed = ok = 0
    for char_dir in OUTPUTS.iterdir():
        if not char_dir.is_dir():
            continue
        for prof_dir in char_dir.iterdir():
            if not prof_dir.is_dir():
                continue
            for f in prof_dir.glob("*.png"):
                key = _lib_key(char_dir.name, prof_dir.name, f.name)
                meta = items.setdefault(key, {"rating": 0, "favorite": False, "tags": []})
                indexed += 1
                if meta.get("phash"):
                    continue
                try:
                    meta["phash"] = phash_mod.phash(f)
                    ok += 1
                except Exception:
                    continue
    _save_lib_index(items)
    return {"total": indexed, "hashed": ok}


def _hash_map() -> dict:
    return {k: v.get("phash") for k, v in _lib_index().items() if v.get("phash")}


@app.get("/api/library/similar")
def library_similar(character: str, name: str, file: str, threshold: int = 12):
    import phash as phash_mod
    _validated_lib_path(character, name, file)
    hm = _hash_map()
    target = hm.get(_lib_key(character, name, file))
    if not target:
        raise HTTPException(400, "target has no hash; build hashes first")
    out = []
    for key, h in hm.items():
        if key == _lib_key(character, name, file):
            continue
        dist = phash_mod.hamming(target, h)
        if dist <= threshold:
            c, n, f = key.split("/", 2)
            out.append({"character": c, "name": n, "file": f, "distance": dist,
                        "url": f"/outputs/{c}/{n}/{f}",
                        "thumb": f"/api/library/thumb?character={c}&name={n}&file={urllib.parse.quote(f)}"})
    out.sort(key=lambda x: x["distance"])
    return {"target": target, "matches": out[:24]}


@app.get("/api/library/duplicates")
def library_duplicates(threshold: int = 6):
    """Greedy clustering: near-duplicate groups across the whole library."""
    import phash as phash_mod
    hm = _hash_map()
    groups, assigned = [], set()
    for key, h in sorted(hm.items()):
        if key in assigned:
            continue
        group = {"key": key, "members": []}
        for other, oh in hm.items():
            if other == key or other in assigned:
                continue
            if phash_mod.hamming(h, oh) <= threshold:
                group["members"].append(other)
                assigned.add(other)
        if group["members"]:
            assigned.add(key)
            group["members"].insert(0, key)
            group["thumbs"] = [f"/api/library/thumb?character={c}&name={n}&file={urllib.parse.quote(f)}"
                               for c, n, f in [m.split("/", 2) for m in group["members"][:8]]]
            groups.append(group)
    return {"groups": groups, "hashed": len(hm)}


@app.get("/api/library/characters")
def library_characters():
    out = {}
    for char_dir in sorted(OUTPUTS.iterdir()):
        if char_dir.is_dir():
            out[char_dir.name] = sorted(p.name for p in char_dir.iterdir() if p.is_dir())
    return out


# ---------------- library curation: bulk delete, virtual folders ----------------

FOLDERS_PATH = ROOT / "library_folders.json"


def _load_folder_names() -> list:
    try:
        return json.loads(FOLDERS_PATH.read_text()).get("folders", [])
    except Exception:
        return []


def _save_folder_names(names: list):
    with index_store.INDEX_LOCK:
        tmp = FOLDERS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"folders": names}, indent=1))
        tmp.replace(FOLDERS_PATH)


def _folder_name(req: dict) -> str:
    name = req.get("name") if isinstance(req.get("name"), str) else ""
    name = (name or "").strip()
    if not name or "/" in name or "\\" in name or len(name) > 30:
        raise HTTPException(400, "folder name must be 1-30 chars, no slashes")
    return name


def _item_keys(req: dict) -> list[str]:
    items = req.get("items")
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "items must be a non-empty list")
    if len(items) > 256:
        raise HTTPException(400, "at most 256 items per call")
    keys = []
    for it in items:
        character, name, file = it.get("character", ""), it.get("name", ""), it.get("file", "")
        _validated_lib_path(character, name, file)
        keys.append(_lib_key(character, name, file))
    return keys


def _delete_render_files(character: str, name: str, file: str):
    """Remove one render and everything derived from it: png, both sidecars,
    index entry (caller strips via merge), cached thumbnail."""
    import hashlib
    p = OUTPUTS / character / name / file
    for suffix in (".png", ".json", ".intronforge.json"):
        p.with_suffix(suffix).unlink(missing_ok=True)
    thumb = THUMBS / (hashlib.md5(f"{character}/{name}/{file}".encode()).hexdigest() + ".jpg")
    thumb.unlink(missing_ok=True)


@app.post("/api/library/delete")
def library_delete(req: dict):
    """Bulk-delete checked renders: files, sidecars, index entries, thumbs."""
    keys = _item_keys(req or {})
    def _apply(items):
        for k in keys:
            items.pop(k, None)
    index_store.merge_and_save(_apply)
    for k in keys:
        c, n, f = k.split("/", 2)
        _delete_render_files(c, n, f)
    return {"deleted": len(keys)}


@app.get("/api/library/folders")
def library_folders():
    index = _lib_index()
    counts = {}
    for meta in index.values():
        for f in meta.get("folders", []):
            counts[f] = counts.get(f, 0) + 1
    names = _load_folder_names()
    # membership can exist for folders created before a registry rollback
    for f in counts:
        if f not in names:
            names.append(f)
    return {"folders": [{"name": n, "items": counts.get(n, 0)} for n in sorted(names)]}


@app.post("/api/library/folders")
def library_folders_create(req: dict):
    name = _folder_name(req or {})
    names = _load_folder_names()
    if name in names:
        raise HTTPException(409, f"folder {name!r} already exists")
    if len(names) >= 50:
        raise HTTPException(400, "folder cap reached (50)")
    _save_folder_names(names + [name])
    return {"created": name}


@app.post("/api/library/folders/rename")
def library_folders_rename(req: dict):
    old = _folder_name(req or {})
    new = (req.get("to") if isinstance(req.get("to"), str) else "") or ""
    new = new.strip()
    if not new or "/" in new or "\\" in new or len(new) > 30:
        raise HTTPException(400, "new name must be 1-30 chars, no slashes")
    names = _load_folder_names()
    if old not in names:
        raise HTTPException(404, f"folder {old!r} not found")
    if new != old and new in names:
        raise HTTPException(409, f"folder {new!r} already exists")
    def _apply(items):
        for meta in items.values():
            fs = meta.get("folders") or []
            if old in fs:
                meta["folders"] = [new if f == old else f for f in fs]
    index_store.merge_and_save(_apply)
    _save_folder_names([new if n == old else n for n in names])
    return {"renamed": old, "to": new}


@app.post("/api/library/folders/delete")
def library_folders_delete(req: dict):
    """Remove the folder and its memberships; renders stay on disk."""
    name = _folder_name(req or {})
    names = _load_folder_names()
    if name not in names:
        raise HTTPException(404, f"folder {name!r} not found")
    affected = 0
    def _apply(items):
        nonlocal affected
        for meta in items.values():
            fs = meta.get("folders") or []
            if name in fs:
                affected += 1
                meta["folders"] = [f for f in fs if f != name]
    index_store.merge_and_save(_apply)
    _save_folder_names([n for n in names if n != name])
    return {"deleted": name, "items_affected": affected}


@app.post("/api/library/assign")
def library_assign(req: dict):
    """Add or remove checked renders' membership in a virtual folder."""
    req = req or {}
    folder = req.get("folder") if isinstance(req.get("folder"), str) else ""
    folder = folder.strip()
    if not folder or "/" in folder or "\\" in folder or len(folder) > 30:
        raise HTTPException(400, "folder name must be 1-30 chars, no slashes")
    if folder not in _load_folder_names():
        raise HTTPException(404, f"folder {folder!r} not found")
    add = bool(req.get("add", True))
    keys = _item_keys(req)
    def _apply(items):
        for k in keys:
            meta = items.setdefault(k, {"rating": 0, "favorite": False, "tags": []})
            fs = meta.setdefault("folders", [])
            if add and folder not in fs:
                fs.append(folder)
            elif not add:
                meta["folders"] = [f for f in fs if f != folder]
    index_store.merge_and_save(_apply)
    return {"folder": folder, "changed": len(keys), "added": add}


DATASETS = ROOT / "datasets"


@app.post("/api/library/export-dataset")
def library_export_dataset(req: dict):
    """Copy checked renders into a training dataset folder, one caption .txt
    per image. The feeder for the LoRAlab training lane: originals stay in
    the library, and the selection itself is the filter. Re-exporting the
    same name is idempotent (files are overwritten in place)."""
    import shutil
    req = req or {}
    name = req.get("name") if isinstance(req.get("name"), str) else ""
    name = name.strip()
    if not name or not KEY_RE.match(name):
        raise HTTPException(400, "dataset name must match [a-z0-9][a-z0-9_-]*")
    mode = req.get("caption") if isinstance(req.get("caption"), str) else "prompt"
    if mode not in ("prompt", "wd14", "none"):
        raise HTTPException(400, "caption must be prompt, wd14, or none")
    keys = _item_keys(req)
    index = _lib_index()
    out_dir = DATASETS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _clean_caption(text: str) -> str:
        text = forge_client.strip_lora(text or "")
        text = re.sub(r"\s*BREAK\s*", ", ", text)
        text = re.sub(r"\s*\n\s*", ", ", text)
        return re.sub(r"\s*,\s*,\s*", ", ", text).strip(" ,")[:2000]

    items, fallbacks = [], 0
    for k in keys:
        c, n, f = k.split("/", 2)
        caption = ""
        if mode == "wd14":
            caption = ", ".join((index.get(k) or {}).get("wd14", []))
        if not caption and mode in ("prompt", "wd14"):
            side = OUTPUTS / c / n / Path(f).with_suffix(".json")
            if side.is_file():
                try:
                    t = json.loads(side.read_text())
                    caption = (t.get("effective_settings") or {}).get("prompt", "")
                except Exception:
                    caption = ""
            if caption and mode == "wd14":
                fallbacks += 1
        if mode != "none":
            caption = _clean_caption(caption)
        src = OUTPUTS / c / n / f
        dest_name = f
        dst = out_dir / dest_name
        if dst.exists() and dst.stat().st_size != src.stat().st_size:
            dest_name = f"{c}_{f}"
            dst = out_dir / dest_name
        shutil.copy2(src, dst)
        if mode != "none":
            dst.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        else:
            dst.with_suffix(".txt").unlink(missing_ok=True)
        items.append({"key": k, "file": dest_name})
    receipt = {"dataset": name, "path": str(out_dir), "caption": mode,
               "exported": len(items), "wd14_fallbacks": fallbacks,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": items}
    tmp = out_dir / "dataset.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    tmp.replace(out_dir / "dataset.json")
    return receipt


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
    try:
        data = json.loads(p.read_text())
    except Exception:
        raise HTTPException(409, "profile unreadable; refusing lock change")
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
        result["image"], result["info"], payload, time.time() - t0,
        ticket_profile=f"{req.character}/{req.name}")
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
            result["image"], result["info"], payload, time.time() - t0,
            ticket_profile=f"{req.character}/{req.name}")
        ticket["url"] = f"/outputs/{req.character}/{req.name}/{ticket['file']}"
        tickets.append(ticket)
    return tickets


app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")


import watcher as lib_watcher

lib_watcher.start()


@app.get("/api/watcher")
def watcher_status():
    return lib_watcher.status()


@app.get("/api/stats")
def stats_endpoint(character: str = "", name: str = ""):
    import stats as stats_mod
    return stats_mod.collect(character, name)


@app.post("/api/library/wd14-tag-all")
def library_wd14_tag_all(req: dict = None):
    """Batch WD14-tag renders missing the wd14 field. Long-running."""
    import wd14 as wd14_mod
    if not wd14_mod.available():
        raise HTTPException(400, "wd14 model not downloaded")
    req = req or {}
    character = req.get("character", "")
    try:
        limit = int(req.get("limit", 500))
    except (TypeError, ValueError):
        raise HTTPException(400, "limit must be a number")
    items = _lib_index()
    tagged = skipped = failed = 0
    for char_dir in OUTPUTS.iterdir():
        if not char_dir.is_dir():
            continue
        if character and char_dir.name != character:
            continue
        for prof_dir in char_dir.iterdir():
            if not prof_dir.is_dir():
                continue
            for f in prof_dir.glob("*.png"):
                key = _lib_key(char_dir.name, prof_dir.name, f.name)
                meta = items.setdefault(key, {"rating": 0, "favorite": False, "tags": []})
                if meta.get("wd14"):
                    skipped += 1
                    continue
                if tagged >= limit:
                    _save_lib_index(items)
                    return {"tagged": tagged, "skipped": skipped, "failed": failed,
                            "note": f"limit {limit} reached; run again for more"}
                try:
                    meta["wd14"] = wd14_mod.tag_image(f)["tags"]
                    tagged += 1
                    if tagged % 8 == 0:
                        _save_lib_index(items)
                except Exception:
                    failed += 1
    _save_lib_index(items)
    return {"tagged": tagged, "skipped": skipped, "failed": failed}


@app.get("/api/library/wd14-status")
def library_wd14_status():
    import wd14 as wd14_mod
    items = _lib_index()
    tagged = sum(1 for v in items.values() if v.get("wd14"))
    return {"model_present": wd14_mod.available(), "tagged": tagged,
            "renders": sum(1 for c in OUTPUTS.iterdir() if c.is_dir()
                           for p in c.iterdir() if p.is_dir()
                           for f in p.glob("*.png"))}


@app.post("/api/profile/{character}/{name}/rename")
def rename_profile(character: str, name: str, req: dict):
    """Rename an unlocked profile: file, outputs dir, and library-index
    curation keys move together. Locked profiles refuse."""
    new_name = ((req.get("new_name") or "").strip()
                if isinstance(req.get("new_name"), str) else "")
    if not KEY_RE.match(new_name):
        raise HTTPException(400, "new name must match [a-z0-9][a-z0-9_-]*")
    if new_name == name:
        raise HTTPException(400, "name unchanged")
    old_p = profile_path(character, name)
    new_p = profile_path(character, new_name)
    if not old_p.is_file():
        raise HTTPException(404, f"profile {character}/{name} not found")
    if new_p.exists():
        raise HTTPException(409, "target profile name already exists")
    try:
        data = json.loads(old_p.read_text())
    except Exception:
        raise HTTPException(409, "profile unreadable; refusing rename")
    if data.get("locked"):
        raise HTTPException(409, "profile is locked; unlock before renaming")
    old_out = OUTPUTS / character / name
    new_out = OUTPUTS / character / new_name
    if old_out.is_dir() and new_out.exists():
        raise HTTPException(409, "target outputs folder already exists")
    import os
    outputs_moved = old_out.is_dir()
    os.rename(old_p, new_p)
    if outputs_moved:
        os.rename(old_out, new_out)
    items = _lib_index()
    prefix_old, prefix_new = f"{character}/{name}/", f"{character}/{new_name}/"
    migrated = {
        (prefix_new + k[len(prefix_old):]): v
        for k, v in items.items() if k.startswith(prefix_old)
    }
    if migrated:
        items = {k: v for k, v in items.items() if not k.startswith(prefix_old)}
        items.update(migrated)
        _save_lib_index(items)
    return {"renamed": f"{character}/{new_name}",
            "outputs_moved": outputs_moved, "index_keys_migrated": len(migrated)}


@app.delete("/api/profile/{character}/{name}")
def delete_profile(character: str, name: str):
    """Delete an unlocked profile JSON. Renders and tickets stay on disk."""
    p = profile_path(character, name)
    if not p.is_file():
        raise HTTPException(404, f"profile {character}/{name} not found")
    try:
        data = json.loads(p.read_text())
    except Exception:
        raise HTTPException(409, "profile unreadable; refusing delete")
    if data.get("locked"):
        raise HTTPException(409, "profile is locked; unlock before deleting")
    p.unlink()
    return {"deleted": f"{character}/{name}",
            "outputs_kept": (OUTPUTS / character / name).is_dir()}

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")





# --- async lora import jobs with progress ---
import threading

_IMPORT_LOCK = threading.Lock()
_IMPORT_JOB = {"state": "idle", "pct": 0, "mb": 0, "total_mb": 0,
               "name": None, "error": None, "result": None}


def _run_import(ref: str):
    def progress(size, total, name):
        _IMPORT_JOB.update({
            "state": "downloading", "name": name,
            "mb": round(size / (1 << 20), 1),
            "total_mb": round(total / (1 << 20), 1) if total else 0,
            "pct": min(100, int(size * 100 / total)) if total else -1,
        })

    try:
        _IMPORT_JOB.update({"state": "resolving", "pct": 0, "error": None})
        r = lora_store.import_from_civitai(ref, progress=progress)
        _IMPORT_JOB.update({"state": "done", "pct": 100, "result": r})
    except ValueError as ex:
        _IMPORT_JOB.update({"state": "error", "error": str(ex)[:200]})
    except Exception as ex:
        _IMPORT_JOB.update({"state": "error", "error": f"{type(ex).__name__}: {str(ex)[:160]}"})


@app.post("/api/loras/import")
def loras_import(req: dict):
    """Start an import job; poll /api/loras/import/status for progress."""
    ref = (req or {}).get("ref", "")
    if not isinstance(ref, str) or not ref.strip():
        raise HTTPException(400, "paste a civitai model link (civitai.com/models/ID) or a bare model ID")
    if not _IMPORT_LOCK.acquire(blocking=False):
        raise HTTPException(409, "an import is already running")
    try:
        _IMPORT_JOB.update({"state": "starting", "pct": 0, "mb": 0, "total_mb": 0,
                            "name": None, "error": None, "result": None})
        threading.Thread(target=_guard, args=(ref,), daemon=True).start()
    except Exception:
        _IMPORT_LOCK.release()
        raise
    return {"job": "current"}


def _guard(ref):
    try:
        _run_import(ref)
    finally:
        _IMPORT_LOCK.release()


@app.get("/api/loras/import/status")
def loras_import_status():
    return dict(_IMPORT_JOB)


@app.get("/api/loras/key-status")
def loras_key_status():
    return {"key_present": bool(lora_store._api_key()),
            "source": "forge config.json (custom_api_key)"}


# ---------------- experiment harness ----------------

EXP_ID_RE = re.compile(r"^exp_\d{8}_\d{6}(?:_\d+)?$")

_EXP_JOB_LOCK = threading.Lock()
_EXP_JOB = {"id": None, "state": "idle", "character": None, "name": None,
            "error": None}


def _run_experiment(exp, outputs_dir: Path):
    try:
        exp.run(outputs_dir)
        _EXP_JOB.update({"state": exp.status, "error": exp.error})
    except Exception as ex:  # run() traps its own errors; belt and braces
        _EXP_JOB.update({"state": "error",
                         "error": f"{type(ex).__name__}: {str(ex)[:200]}"})


def _experiment_guard(exp, outputs_dir: Path):
    try:
        _run_experiment(exp, outputs_dir)
    finally:
        _EXP_JOB_LOCK.release()


@app.post("/api/experiment")
def experiment_start(req: ExperimentReq):
    """Start a controlled experiment in the background; returns immediately
    with the experiment id. Poll GET /api/experiment/{id} for progress:
    the results table grows one row per completed cell."""
    import experiments as exp_mod
    if not (KEY_RE.match(req.character) and KEY_RE.match(req.name)):
        raise HTTPException(400, "bad profile key")
    profile = apply_universal(req.profile if req.profile else load_profile(req.character, req.name))
    if not _EXP_JOB_LOCK.acquire(blocking=False):
        raise HTTPException(
            409, f"experiment {_EXP_JOB.get('id')} is still running; "
                 f"poll GET /api/experiment/{_EXP_JOB.get('id')}")
    try:
        try:
            exp = exp_mod.Experiment(
                EXPERIMENTS, profile, req.character, req.name,
                req.axes, req.fixed_seed, req.metrics)
        except ValueError as ex:
            raise HTTPException(400, str(ex))
        _EXP_JOB.update({"id": exp.id, "state": "starting",
                         "character": req.character, "name": req.name,
                         "error": None})
        threading.Thread(target=_experiment_guard,
                         args=(exp, OUTPUTS / req.character / req.name),
                         daemon=True).start()
    except Exception:
        _EXP_JOB_LOCK.release()
        raise
    return exp.to_dict()


@app.get("/api/experiments")
def experiments_list():
    import experiments as exp_mod
    return {"experiments": exp_mod.list_experiments(EXPERIMENTS)}


def _experiment_dir(exp_id: str) -> Path:
    if not EXP_ID_RE.match(exp_id):
        raise HTTPException(400, "bad experiment id")
    d = EXPERIMENTS / exp_id
    if not d.is_dir():
        raise HTTPException(404, f"experiment {exp_id} not found")
    return d


@app.get("/api/experiment/{exp_id}")
def experiment_get(exp_id: str):
    import experiments as exp_mod
    return exp_mod.load_experiment(_experiment_dir(exp_id))


@app.post("/api/experiment/{exp_id}/apply")
def experiment_apply(exp_id: str, req: ExperimentApplyReq):
    """Apply a winning cell's settings to the source profile.
    This is the explicit user-accept gate: agents propose, users apply."""
    import experiments as exp_mod
    data = exp_mod.load_experiment(_experiment_dir(exp_id))
    if data.get("status") != "done":
        raise HTTPException(409, "experiment not finished")
    if req.cell not in [r.get("cell") for r in data.get("results", [])]:
        raise HTTPException(400, "cell is not one of this experiment's result cells")
    profile = load_profile(data["character"], data["name"])
    if profile.get("locked"):
        raise HTTPException(409, "profile is locked; unlock before applying")
    profile = exp_mod._apply_overrides(profile, req.cell)
    p = profile_path(data["character"], data["name"])
    p.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    return {"saved": f"{data['character']}/{data['name']}", "cell": req.cell}


# ---------------- assist mailbox (agent harness connection) ----------------
# The deck files requests and shows proposals; an agent harness (Hermes is
# the reference) does the model work under its own provider and money rules.
# The deck never calls out and cannot spend.

ASSIST_DIR = ROOT / "assist"
ASSIST_ID_RE = re.compile(r"^asst_\d{8}_\d{6}(?:_\d+)?$")
ASSIST_KINDS = {"format-for-lane", "general"}
ASSIST_LOCK = threading.Lock()


def _assist_path(asst_id: str) -> Path:
    if not ASSIST_ID_RE.match(asst_id):
        raise HTTPException(400, "bad assist id")
    p = ASSIST_DIR / f"{asst_id}.json"
    if not p.is_file():
        raise HTTPException(404, f"assist request {asst_id} not found")
    return p


def _assist_write(rec: dict):
    ASSIST_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ASSIST_DIR / (rec["id"] + ".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(ASSIST_DIR / f"{rec['id']}.json")


@app.post("/api/assist")
def assist_create(req: dict):
    """File an assist request for an agent harness to answer."""
    req = req or {}
    kind = req.get("kind") if isinstance(req.get("kind"), str) else ""
    if kind not in ASSIST_KINDS:
        raise HTTPException(400, f"kind must be one of {sorted(ASSIST_KINDS)}")
    prompt_text = req.get("prompt") if isinstance(req.get("prompt"), str) else ""
    if not prompt_text.strip():
        raise HTTPException(400, "prompt is required")
    note = req.get("note") if isinstance(req.get("note"), str) else ""
    lane = req.get("lane") if isinstance(req.get("lane"), str) else ""
    rec = {
        "id": f"asst_{time.strftime('%Y%m%d_%H%M%S')}",
        "kind": kind,
        "status": "open",
        "character": str(req.get("character") or "")[:60],
        "name": str(req.get("name") or "")[:60],
        "lane": lane[:30],
        "prompt": prompt_text[:20000],
        "negative_prompt": str(req.get("negative_prompt") or "")[:20000],
        "note": note[:500],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "proposal": None,
    }
    with ASSIST_LOCK:
        n = 0
        while (ASSIST_DIR / f"{rec['id']}.json").exists():
            n += 1
            rec["id"] = f"asst_{time.strftime('%Y%m%d_%H%M%S')}_{n}"
        _assist_write(rec)
    return rec


@app.get("/api/assist")
def assist_list(status: str = "", limit: int = 20):
    out = []
    if ASSIST_DIR.is_dir():
        for f in sorted(ASSIST_DIR.glob("asst_*.json"), reverse=True):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if status and rec.get("status") != status:
                continue
            out.append(rec)
    return {"requests": out[:max(1, min(limit, 100))]}


@app.get("/api/assist/{asst_id}")
def assist_get(asst_id: str):
    return json.loads(_assist_path(asst_id).read_text(encoding="utf-8"))


@app.post("/api/assist/{asst_id}/propose")
def assist_propose(asst_id: str, req: dict):
    """Harness answer: a proposed prompt plus rationale. Open requests only."""
    req = req or {}
    prompt_text = req.get("prompt") if isinstance(req.get("prompt"), str) else ""
    if not prompt_text.strip():
        raise HTTPException(400, "proposal prompt is required")
    with ASSIST_LOCK:
        rec = json.loads(_assist_path(asst_id).read_text(encoding="utf-8"))
        if rec.get("status") != "open":
            raise HTTPException(409, f"request is {rec.get('status')}, not open")
        rec["proposal"] = {
            "prompt": prompt_text[:20000],
            "negative_prompt": str(req.get("negative_prompt") or "")[:20000],
            "rationale": str(req.get("rationale") or "")[:2000],
            "proposed_by": str(req.get("proposed_by") or "unknown")[:100],
            "proposed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rec["status"] = "proposed"
        _assist_write(rec)
    return rec


@app.post("/api/assist/{asst_id}/resolve")
def assist_resolve(asst_id: str, req: dict):
    """User decision on a proposal: accepted or declined."""
    decision = (req or {}).get("decision")
    if decision not in ("accepted", "declined"):
        raise HTTPException(400, "decision must be accepted or declined")
    with ASSIST_LOCK:
        rec = json.loads(_assist_path(asst_id).read_text(encoding="utf-8"))
        if rec.get("status") != "proposed":
            raise HTTPException(409, f"request is {rec.get('status')}, not proposed")
        rec["status"] = decision
        rec["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _assist_write(rec)
    return rec


# ---------------- intradepot: markdown knowledge base ----------------

import depot as depot_mod


def _depot_note_or_404(note_id: str) -> dict:
    try:
        return depot_mod.load_note(note_id)
    except FileNotFoundError:
        raise HTTPException(404, f"note {note_id} not found")
    except ValueError:
        raise HTTPException(400, "bad note id")


@app.get("/api/depot")
def depot_list(q: str = "", tag: str = "", type: str = ""):
    """Search notes; q matches title, body, tags, and links."""
    return {"notes": depot_mod.list_notes(q, tag, type),
            "tags": sorted({t for r in depot_mod.list_notes() for t in r["tags"]})}


@app.get("/api/depot/backlinks/{target}")
def depot_backlinks(target: str):
    return {"target": target, "notes": depot_mod.backlinks(target)}


@app.get("/api/depot/{note_id}")
def depot_get(note_id: str):
    return _depot_note_or_404(note_id)


@app.post("/api/depot")
def depot_save(req: dict):
    """Create or update a note. id optional (slugged from title)."""
    req = req or {}
    title = str(req.get("title") or "").strip()[:120]
    if not title:
        raise HTTPException(400, "title is required")
    tags = req.get("tags") if isinstance(req.get("tags"), list) else []
    links = req.get("links") if isinstance(req.get("links"), list) else []
    extra = {}
    for key in ("hires", "detailer_tabs"):
        if req.get(key) is not None:
            extra[key] = req[key]
    try:
        return depot_mod.write_note(str(req.get("id") or title), title,
                                    str(req.get("type") or "note"), tags, links,
                                    str(req.get("body") or ""), extra)
    except ValueError as ex:
        raise HTTPException(400, str(ex))


@app.delete("/api/depot/{note_id}")
def depot_delete(note_id: str):
    try:
        depot_mod.delete_note(note_id)
    except FileNotFoundError:
        raise HTTPException(404, f"note {note_id} not found")
    except ValueError:
        raise HTTPException(400, "bad note id")
    return {"deleted": note_id}


@app.post("/api/depot/{note_id}/apply-recipe")
def depot_apply_recipe(note_id: str, req: dict):
    """Apply a recipe note's hires/detailer blocks to a profile.
    The explicit user-accept gate: same rule as experiment apply."""
    note = _depot_note_or_404(note_id)
    fm = note["frontmatter"]
    hires = fm.get("hires") if isinstance(fm.get("hires"), dict) else None
    tabs = fm.get("detailer_tabs") if isinstance(fm.get("detailer_tabs"), list) else None
    if not hires and not tabs:
        raise HTTPException(400, "note carries no recipe (needs hires or detailer_tabs)")
    req = req or {}
    character, name = req.get("character", ""), req.get("name", "")
    profile = load_profile(character, name)  # validates keys, 404s
    if profile.get("locked"):
        raise HTTPException(409, "profile is locked; unlock before applying")
    changed = {}
    if hires:
        profile.setdefault("hires", {}).update(hires)
        changed["hires"] = sorted(hires.keys())
    if tabs:
        profile.setdefault("detailer", {})["tabs"] = tabs
        changed["detailer_tabs"] = len(tabs)
    p = profile_path(character, name)
    p.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    return {"applied": f"{character}/{name}", "note": note_id, "changed": changed}


@app.post("/api/depot/ingest-guide")
def depot_ingest_guide(req: dict):
    """Saved HTML page (Civitai guide folders in the project root) -> guide note."""
    path = str((req or {}).get("path") or "").strip()
    if not path:
        raise HTTPException(400, "path is required")
    try:
        return depot_mod.ingest_guide(path)
    except ValueError as ex:
        raise HTTPException(400, str(ex))


# ---------------- trainer bridge (LoRAlab integration) ----------------

import train_runner


@app.get("/api/train/presets")
def train_presets():
    return {"presets": train_runner.PRESETS,
            "pinned_commit": train_runner.PINNED_COMMIT,
            "commit_drift": train_runner.commit_drift()}


@app.get("/api/train/datasets")
def train_datasets():
    out = []
    if DATASETS.is_dir():
        for d in sorted(DATASETS.iterdir(), key=lambda x: -x.stat().st_mtime):
            if d.is_dir():
                pngs = len(list(d.glob("*.png")))
                receipt = {}
                rj = d / "dataset.json"
                if rj.is_file():
                    try:
                        r = json.loads(rj.read_text())
                        receipt = {"caption": r.get("caption"), "created": r.get("created")}
                    except Exception:
                        pass
                out.append({"name": d.name, "images": pngs, **receipt})
    return {"datasets": out}


@app.post("/api/train/start")
def train_start(req: dict):
    """Start a LoRA training run through the LoRAlab bridge."""
    req = req or {}
    name = req.get("name") if isinstance(req.get("name"), str) else ""
    name = name.strip()
    if not name:
        raise HTTPException(400, "name the output lora")
    overrides = req.get("overrides") if isinstance(req.get("overrides"), dict) else {}
    try:
        return train_runner.start(req.get("dataset", ""), req.get("preset", ""),
                                   name, req.get("trigger_word", ""),
                                   req.get("custom_checkpoint", ""), overrides)
    except train_runner.Busy as ex:
        raise HTTPException(409, str(ex))
    except ValueError as ex:
        raise HTTPException(400, str(ex))


@app.get("/api/train/status")
def train_status():
    return train_runner.status()


@app.post("/api/train/stop")
def train_stop():
    return train_runner.stop()
