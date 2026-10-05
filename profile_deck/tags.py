"""Danbooru tag assist: local autocomplete over the on-disk tag dump.

Data: danbooru_2025-09-01.csv (183k rows; tag, category, count, aliases
pipe-separated) from the EasyUseAnima install. No network at runtime.

Suggest order: exact tag -> exact alias -> prefix matches -> substring
matches -> alias substrings -> nearest-prefix fallback for multi-word
phrases ("green ballroom gown" -> the green_* tag space). All ranked by
post count.
"""
import csv
import re
from pathlib import Path

CSV_PATH = Path("/path/to/danbooru/danbooru_2025-09-01.csv")
CATEGORY_NAMES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}
LIMIT = 12

_tags = []
_by_tag = {}
_by_alias = {}
_loaded = False


def load():
    global _tags, _by_tag, _by_alias, _loaded
    if _loaded:
        return
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            tag = row[0].strip()
            if not tag:
                continue
            try:
                count = int(row[2].strip())
            except ValueError:
                count = 0
            cat = row[1].strip()
            rows.append((tag, cat, count))
            low = tag.lower()
            if low not in _by_tag:
                _by_tag[low] = rows[-1]
            if len(row) > 3:
                for alias in row[3].split(","):
                    alias = alias.strip().lower()
                    if alias and alias not in _by_alias:
                        _by_alias[alias] = low
    rows.sort(key=lambda r: -r[2])
    _tags = rows
    _loaded = True


def _query_norm(q: str) -> str:
    return re.sub(r"[\s]+", "_", (q or "").strip().lower()).strip("_")


def _entry(tag):
    t, cat, n = _by_tag.get(tag.lower(), (tag, "0", 0))
    return {"tag": t, "category": CATEGORY_NAMES.get(int(cat) if str(cat).isdigit() else 0, "general"),
            "count": n, "alias_of": None}


def suggest(q: str, limit: int = LIMIT) -> dict:
    load()
    raw = (q or "").strip()
    qu = _query_norm(raw)
    if not qu:
        return {"query": raw, "results": []}
    results, seen = [], set()

    def push(tag, alias_of=None):
        if tag in seen or len(results) >= limit:
            return
        e = _entry(tag)
        e["alias_of"] = alias_of
        results.append(e)
        seen.add(tag)

    # 1. exact tag or exact alias
    if qu in _by_tag:
        push(_by_tag[qu][0])
    elif qu in _by_alias:
        push(_by_alias[qu], alias_of=qu)

    # 2. prefix matches, count-ranked (the dump is pre-sorted)
    for tag, cat, n in _tags:
        if len(results) >= limit:
            break
        if tag.lower().startswith(qu):
            push(tag)

    # 3. substring matches, count-ranked
    if len(results) < limit:
        for tag, cat, n in _tags:
            if len(results) >= limit:
                break
            if qu in tag.lower():
                push(tag)

    # 4. alias substrings
    if len(results) < limit:
        for alias, canon in _by_alias.items():
            if len(results) >= limit:
                break
            if qu in alias and canon not in seen:
                push(canon, alias_of=alias)

    # 5. nearest-prefix fallback for multi-word phrases: drop trailing
    #    segments until something NEW matches ("green_ballroom_gown" -> "green")
    if len(results) < limit and "_" in qu:
        parts = qu.split("_")
        for take in range(len(parts) - 1, 0, -1):
            q2 = "_".join(parts[:take])
            before = len(results)
            for tag, cat, n in _tags:
                if len(results) >= limit:
                    break
                if tag.lower().startswith(q2 + "_") and tag not in seen:
                    push(tag)
            if len(results) > before:
                break

    return {"query": raw, "results": results[:limit]}
