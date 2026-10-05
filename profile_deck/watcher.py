"""Background watcher: pre-warm the library for new renders.

The library listing scans OUTPUTS live, so registration is inherent.
This poller does the heavy pre-computation a new render needs: cached
thumbnail and perceptual hash. Idempotent by design — both artifacts
are cached on disk, so a restart simply picks up where it stopped and
never duplicates entries.

Files younger than QUIET_SECONDS are skipped until the next poll so a
render still being written is never half-hashed.
"""
import json
import threading
import time
from pathlib import Path

import phash as phash_mod

POLL_SECONDS = 8
QUIET_SECONDS = 10

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
THUMBS = ROOT / "thumbs"

import store as index_store

_state = {"running": False, "last_scan": None, "processed": 0, "errors": 0}


def _scan_once() -> int:
    updates = {}  # key -> phash, merged under the shared lock at save time
    processed = 0
    now = time.time()
    for char_dir in OUTPUTS.iterdir():
        if not char_dir.is_dir():
            continue
        for prof_dir in char_dir.iterdir():
            if not prof_dir.is_dir():
                continue
            for f in prof_dir.glob("*.png"):
                try:
                    if now - f.stat().st_mtime < QUIET_SECONDS:
                        continue
                except OSError:
                    continue
                key = f"{char_dir.name}/{prof_dir.name}/{f.name}"
                need_thumb = not _thumb_exists(char_dir.name, prof_dir.name, f.name)
                items = index_store.load()
                meta = items.get(key, {})
                need_phash = not meta.get("phash")
                if not need_thumb and not need_phash:
                    continue
                if need_thumb:
                    try:
                        _make_thumb(f, char_dir.name, prof_dir.name, f.name)
                    except Exception:
                        _state["errors"] += 1
                if need_phash:
                    try:
                        updates[key] = phash_mod.phash(f)
                    except Exception:
                        _state["errors"] += 1
                processed += 1
    if updates:
        def _apply(items):
            for k, h in updates.items():
                items.setdefault(k, {"rating": 0, "favorite": False, "tags": []})["phash"] = h
        index_store.merge_and_save(_apply)
    return processed


def _thumb_name(character: str, name: str, file: str) -> Path:
    import hashlib
    return THUMBS / (hashlib.md5(f"{character}/{name}/{file}".encode()).hexdigest() + ".jpg")


def _thumb_exists(character: str, name: str, file: str) -> bool:
    return _thumb_name(character, name, file).is_file()


def _make_thumb(f: Path, character: str, name: str, file: str):
    from PIL import Image
    THUMBS.mkdir(exist_ok=True)
    dest = _thumb_name(character, name, file)
    if dest.is_file():
        return
    im = Image.open(f).convert("RGB")
    im.thumbnail((256, 256))
    im.save(dest, "JPEG", quality=80)


def watch_loop():
    _state["running"] = True
    while True:
        try:
            n = _scan_once()
            _state["last_scan"] = time.strftime("%H:%M:%S")
            if n:
                _state["processed"] += n
        except Exception:
            _state["errors"] += 1
        time.sleep(POLL_SECONDS)


def start():
    t = threading.Thread(target=watch_loop, daemon=True)
    t.start()
    return t


def status() -> dict:
    return {**_state, "poll_seconds": POLL_SECONDS}
