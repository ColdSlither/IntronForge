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
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

app = FastAPI(title="Profile Deck")
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
    """Normalize a prompt to the user checkpoint lane. Receipt only; no side effects."""
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
    data = b64mod.b64decode(req.get("b64", "").split(",")[-1])
    key = f"{req.get('character', 'x')}_{req.get('name', 'y')}_u{req.get('unit', 1)}"
    safe = re.sub(r"[^a-z0-9_-]", "_", key.lower())
    p = ROOT / "controlnet_images" / f"{safe}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    im = PILImage.open(p)
    im.thumbnail((1024, 1024))
    im.save(p)
    return {"image_file": p.name, "url": f"/cn-images/{p.name}"}


@app.get("/api/outputs/{character}/{name}")
def outputs_list(character: str, name: str):
    if not (KEY_RE.match(character) and KEY_RE.match(name)):
        raise HTTPException(400, "bad profile key")
    d = OUTPUTS / character / name
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [{"file": f.name, "url": f"/outputs/{character}/{name}/{f.name}"} for f in files]


THUMBS = ROOT / "thumbs"
THUMBS.mkdir(exist_ok=True)
LIB_INDEX = ROOT / "library_index.json"


def _lib_index() -> dict:
    try:
        return json.loads(LIB_INDEX.read_text()).get("items", {})
    except Exception:
        return {}


def _save_lib_index(items: dict):
    LIB_INDEX.write_text(json.dumps({"items": items}, indent=1))


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
            min_rating: int = 0, favorites: int = 0, tag: str = ""):
    """Every render across all characters/profiles, newest first, merged
    with the curation index (rating/favorite/tags)."""
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
        r = int(req["rating"])
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
    items[key] = meta
    _save_lib_index(items)
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
            c, n, f = key.split("/", 2)
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
    limit = int(req.get("limit", 500))
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
                    return {"tagged": tagged, "skipped": skipped, "failed": failed,
                            "note": f"limit {limit} reached; run again for more"}
                try:
                    meta["wd14"] = wd14_mod.tag_image(f)["tags"]
                    tagged += 1
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
    new_name = (req.get("new_name") or "").strip()
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
