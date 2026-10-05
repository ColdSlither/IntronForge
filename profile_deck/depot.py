"""IntronDepot: markdown knowledge base for IntronForge.

Notes are plain markdown files under profile_deck/depot/ with simple
frontmatter (type, tags, links; recipe blocks as JSON values), so the folder
opens directly as an Obsidian vault and the same links and tags resolve
there. Backlinks are derived from note links, never written into profiles.

Note types: guide (ingested or hand-written), prompt (insertable into the
prompt bar), recipe (hires/detailer blocks applicable to a profile behind an
explicit user confirm), note (everything else).

Snippet bridge decision (2026-10-04): the snippet library stays separate.
Snippets are structured prompt blocks consumed by the sections composer
through its own endpoint; depot notes are knowledge documents. Revisit only
if depot prompt-notes earn real usage.
"""
import json
import re
import time
from pathlib import Path

DEPOT = Path(__file__).resolve().parent / "depot"
NOTE_TYPES = {"guide", "prompt", "recipe", "note"}
SLUG_RE = re.compile(r"[^a-z0-9_-]+")
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
TITLE_RE = re.compile(r"^#\s+(.+)$", re.M)


def _clean_list_value(v) -> str:
    return re.sub(r"[\[\],\n]", "", str(v)).strip()


def _slug(text: str, fallback: str = "note") -> str:
    s = SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")[:60]
    return s or fallback


def _note_path(note_id: str) -> Path:
    if (not note_id or "/" in note_id or "\\" in note_id
            or note_id.startswith(".") or ".." in note_id):
        raise ValueError("bad note id")
    return DEPOT / f"{note_id}.md"


def parse_note(text: str) -> dict:
    """Tolerant frontmatter parse: key: value with [list] and JSON values."""
    fm, body = {}, text
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            body = parts[1]
            for line in parts[0][4:].splitlines():
                if ":" not in line or line.startswith((" ", "\t")):
                    continue
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if not k:
                    continue
                if v.startswith("{") or v.startswith("["):
                    try:
                        fm[k] = json.loads(v)
                    except Exception:
                        if v.startswith("[") and v.endswith("]"):
                            fm[k] = [x.strip().strip('"') for x in v[1:-1].split(",") if x.strip()]
                        else:
                            fm[k] = v
                else:
                    fm[k] = v.strip('"')
    m = TITLE_RE.search(body)
    title = m.group(1).strip() if m else ""
    links = set(str(x) for x in (fm.get("links") or []))
    links.update(WIKILINK_RE.findall(body))
    return {"frontmatter": fm, "title": title, "body": body.strip("\n"),
            "links": sorted(l for l in links if l)}


def load_note(note_id: str) -> dict:
    p = _note_path(note_id)
    if not p.is_file():
        raise FileNotFoundError(f"note {note_id} not found")
    parsed = parse_note(p.read_text(encoding="utf-8"))
    return {"id": note_id, **parsed, "mtime": int(p.stat().st_mtime)}


def write_note(note_id: str, title: str, note_type: str, tags, links,
               body: str, extra_fm: dict | None = None) -> dict:
    if note_type not in NOTE_TYPES:
        raise ValueError(f"type must be one of {sorted(NOTE_TYPES)}")
    note_id = _slug(note_id or title)
    fm = [f"type: {note_type}",
          "tags: [" + ", ".join(_clean_list_value(t) for t in (tags or []) if str(t).strip()) + "]",
          "links: [" + ", ".join(_clean_list_value(l) for l in (links or []) if str(l).strip()) + "]",
          "created: " + time.strftime("%Y-%m-%d")]
    for k, v in (extra_fm or {}).items():
        fm.append(f"{k}: {json.dumps(v)}")
    text = "---\n" + "\n".join(fm) + "\n---\n\n" + (body or "").strip() + "\n"
    DEPOT.mkdir(parents=True, exist_ok=True)
    p = _note_path(note_id)
    tmp = DEPOT / (note_id + ".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return load_note(note_id)


def delete_note(note_id: str) -> bool:
    p = _note_path(note_id)
    if not p.is_file():
        raise FileNotFoundError(f"note {note_id} not found")
    p.unlink()
    return True


def list_notes(q: str = "", tag: str = "", note_type: str = "") -> list:
    rows = []
    if not DEPOT.is_dir():
        return rows
    for p in sorted(DEPOT.glob("*.md"), key=lambda x: -x.stat().st_mtime):
        try:
            n = parse_note(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tags = [str(t) for t in (n["frontmatter"].get("tags") or [])]
        t = n["frontmatter"].get("type", "note")
        if note_type and t != note_type:
            continue
        if tag and tag not in tags:
            continue
        if q:
            hay = " ".join([p.stem, n["title"], n["body"],
                            " ".join(tags), " ".join(n["links"])]).lower()
            if q.lower() not in hay:
                continue
        snippet = re.sub(r"\s+", " ", re.sub(r"^#.*$", "", n["body"])).strip()[:140]
        rows.append({"id": p.stem, "title": n["title"] or p.stem, "type": t,
                     "tags": tags, "links": n["links"],
                     "mtime": int(p.stat().st_mtime), "snippet": snippet})
    return rows


def backlinks(target: str) -> list:
    return [r for r in list_notes() if target in r["links"]]


def ingest_guide(path_str: str) -> dict:
    """Saved CivitAI (or any) HTML page -> guide note. Keeps the source path
    in the body so the receipt says where the text came from."""
    p = Path(path_str.strip())
    if p.is_dir():
        htmls = sorted(p.glob("*.html"))
        if not htmls:
            raise ValueError("folder has no html file")
        p = htmls[0]
    if not p.is_file():
        raise ValueError(f"not a file: {p}")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for sel in soup(["script", "style", "nav", "aside", "header", "footer", "noscript"]):
        sel.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string
             else (soup.h1.get_text().strip() if soup.h1 else p.stem))
    title = re.sub(r"\s*_\s*Civitai\s*$", "", title).strip() or p.stem
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
    if len(text) < 300:
        raise ValueError("page has no extractable text (js-only page?); "
                         "paste the content into a note instead")
    tags = ["civitai"]
    tl = title.lower() + text[:4000].lower()
    if "pony" in tl or "score_9" in tl:
        tags.append("pony")
    if "prompt" in tl:
        tags.append("prompting")
    body = f"# {title}\n\nsource: `{p}`\ningested {time.strftime('%Y-%m-%d')}\n\n{text[:100000]}\n"
    existing = _slug(title)
    note_id = existing
    n = 0
    while _note_path(note_id).is_file():
        n += 1
        note_id = f"{existing}-{n}"
    return write_note(note_id, title, "guide", tags, [], body)
