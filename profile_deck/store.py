"""Shared library-index store: one lock, atomic writes.

Both the request handlers (app.py) and the background watcher mutate
library_index.json; this module is the single synchronization point.
Saves are atomic (temp file + os.replace) so a crash mid-write can never
leave invalid JSON for the loader to swallow into an empty index.
"""
import json
import os
import threading
from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parent / "library_index.json"
INDEX_LOCK = threading.Lock()

_DEFAULT = {"items": {}}


def load() -> dict:
    with INDEX_LOCK:
        try:
            return json.loads(INDEX_PATH.read_text()).get("items", {})
        except Exception:
            return {}


def save(items: dict):
    with INDEX_LOCK:
        _atomic_write(items)


def merge_and_save(mutator):
    """Run mutator(items_dict) under the lock and persist atomically.
    Use for read-modify-write sequences so concurrent writers merge."""
    with INDEX_LOCK:
        try:
            items = json.loads(INDEX_PATH.read_text()).get("items", {})
        except Exception:
            items = {}
        mutator(items)
        _atomic_write(items)


def _atomic_write(items: dict):
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"items": items}, indent=1))
    os.replace(tmp, INDEX_PATH)
