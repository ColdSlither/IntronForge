"""Prompt statistics over the library's sidecar tickets.

Aggregates token frequencies from every render's recorded effective
prompt (positive and negative separated), per character and overall.
Quality-grammar tokens (score_*, source_*) are counted as grammar, not
content, and excluded from the top-tag lists. Feeds the danbooru assist
with the user's real usage instead of a generic corpus.
"""
import json
import re
from collections import Counter
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"

GRAMMAR = re.compile(r"^(score_\d(_up)?|source_\w+)$", re.I)


def _tokens(text: str):
    for seg in re.split(r"\s*BREAK\s*", text or ""):
        for tok in seg.split(","):
            t = re.sub(r"[()]+", "", tok)
            t = t.split(":")[0]
            t = re.sub(r"[\s-]+", " ", t.strip().lower()).strip()
            t = t.replace(" ", "_")
            if t and not GRAMMAR.match(t) and not t.startswith("<lora"):
                yield t


def collect(character: str = "", name: str = "") -> dict:
    root = OUTPUTS
    chars = {}
    overall_pos = Counter()
    overall_neg = Counter()
    total = 0
    for char_dir in sorted(root.iterdir()):
        if not char_dir.is_dir():
            continue
        if character and char_dir.name != character:
            continue
        c_pos, c_neg = Counter(), Counter()
        c_renders = 0
        for prof_dir in sorted(char_dir.iterdir()):
            if not prof_dir.is_dir():
                continue
            if name and prof_dir.name != name:
                continue
            for side in prof_dir.glob("*.json"):
                try:
                    d = json.loads(side.read_text())
                    eff = d.get("effective_settings", {})
                    pos, neg = eff.get("prompt", ""), eff.get("negative_prompt", "")
                except Exception:
                    continue
                c_renders += 1
                total += 1
                for t in _tokens(pos):
                    c_pos[t] += 1
                    overall_pos[t] += 1
                for t in _tokens(neg):
                    c_neg[t] += 1
                    overall_neg[t] += 1
        if c_renders:
            chars[char_dir.name] = {
                "renders": c_renders,
                "profiles": len([p for p in prof_dir.parent.iterdir() if p.is_dir()]),
                "top_positive": [{"tag": t, "count": c} for t, c in c_pos.most_common(25)],
                "top_negative": [{"tag": t, "count": c} for t, c in c_neg.most_common(15)],
            }
    return {
        "total_renders": total,
        "characters": chars,
        "overall_top_positive": [{"tag": t, "count": c} for t, c in overall_pos.most_common(40)],
        "overall_top_negative": [{"tag": t, "count": c} for t, c in overall_neg.most_common(20)],
    }
