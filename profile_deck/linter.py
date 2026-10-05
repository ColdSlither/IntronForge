"""Prompt redundancy linter, adapted from PromptRedundancyAssistant.

Scoring (capped at 100):
- 20 pts per exact duplicate tag beyond the first
- 12 pts per semantic-overlap member beyond the first (curated table)
- 3 pts per repeated-word excess (same significant word in 3+ tags)
- 12 pts per contradictory color on the same attribute (keeps first)
- 6 pts per generic tag subsumed by a more specific one present

Protected tokens (LoRA names, trigger words) are never flagged.
The negative prompt is checked for exact duplicates only.
lint() returns findings; apply() performs selected removals server-side
so the prompt rebuild stays in one place.
"""
import re

DUPE_POINTS = 20
OVERLAP_POINTS = 12
REPEAT_WORD_POINTS = 3
COLOR_CONFLICT_POINTS = 12
SUBSUMED_POINTS = 6
SCORE_CAP = 100

OVERLAP_GROUPS = [
    {"masterpiece", "best quality", "high quality", "highres", "high res", "highest quality"},
    {"looking at viewer", "eye contact"},
    {"smiling", "grin", "grinning"},
    {"huge breasts", "big breasts", "large breasts", "gigantic breasts"},
    {"beautiful", "stunning", "gorgeous", "cutest girl in the world"},
    {"detailed face", "detailed eyes", "intricate detail"},
]

COLORS = {"black", "white", "red", "orange", "yellow", "green", "blue",
          "purple", "pink", "brown", "grey", "gray", "blonde", "blond",
          "silver", "aqua", "cyan", "auburn", "crimson"}
COLOR_ATTR = re.compile(r"\b(hair|eyes?|eyebrows|pubic hair)\b")

SUBSUME_BASES = ["dress", "shirt", "skirt", "shoes", "boots", "hair", "gloves", "lingerie"]


def _norm(token: str) -> str:
    t = re.sub(r"[()]+", "", token or "")
    t = t.split(":")[0]
    return re.sub(r"[\s-]+", " ", t.lower()).strip()


def _tokenize(text: str):
    out = []
    for seg in re.split(r"\s*BREAK\s*", text or ""):
        for line in seg.split("\n"):
            for tok in line.split(","):
                if tok.strip():
                    out.append(tok.strip())
    return out


def _is_protected(raw: str, protected) -> bool:
    n = _norm(raw)
    for p in protected or []:
        pn = _norm(p)
        if pn and (pn in n or n in pn):
            return True
    return False


def lint(prompt: str, negative: str, protected) -> dict:
    findings = []
    protected = list(protected or [])

    toks = _tokenize(prompt)
    prot = [t for t in toks if _is_protected(t, protected)]
    work = [t for t in toks if not _is_protected(t, protected)]
    norms = [_norm(t) for t in work]

    # 1. exact duplicates
    seen, dupe_groups = {}, {}
    for raw, n in zip(work, norms):
        seen.setdefault(n, raw)
        if n in seen and n not in dupe_groups:
            dupe_groups[n] = {"first": seen[n], "dupes": []}
        elif n in dupe_groups:
            dupe_groups[n]["dupes"].append(raw)
    for n, g in dupe_groups.items():
        if g["dupes"]:
            findings.append({
                "op": "remove", "kind": "duplicate",
                "target": ", ".join(g["dupes"]),
                "reason": f"exact duplicate of '{g['first']}'",
                "points": DUPE_POINTS * len(g["dupes"]),
                "remove_norms": {n: len(g["dupes"])},
            })

    # 2. semantic overlap (curated groups)
    present = {}
    for raw, n in zip(work, norms):
        present.setdefault(n, raw)
    for group in OVERLAP_GROUPS:
        members = [n for n in present if n in group]
        if len(members) > 1:
            members.sort(key=lambda n: -len(n))  # keep the most specific
            for n in members[1:]:
                findings.append({
                    "op": "remove", "kind": "overlap",
                    "target": present[n],
                    "reason": f"overlaps with '{present[members[0]]}'",
                    "points": OVERLAP_POINTS,
                    "remove_norms": {n: 1},
                })

    # 3. repeated significant words
    word_hits = {}
    for raw, n in zip(work, norms):
        for w in set(n.split()):
            if len(w) >= 4:
                word_hits.setdefault(w, []).append((raw, n))
    for w, hits in word_hits.items():
        if len(hits) >= 3:
            findings.append({
                "op": "flag", "kind": "repeated-word",
                "target": f"'{w}' in {len(hits)} tags",
                "reason": ", ".join(r for r, _ in hits[:6]),
                "points": REPEAT_WORD_POINTS * (len(hits) - 2),
                "remove_norms": {},
            })

    # 4. contradictory colors on the same attribute
    attr_colors = {}
    for raw, n in zip(work, norms):
        if COLOR_ATTR.search(n):
            for c in COLORS:
                if re.search(rf"\b{c}\b", n):
                    attr = COLOR_ATTR.search(n).group(1)
                    attr_colors.setdefault(attr, []).append((c, raw))
                    break
    for attr, lst in attr_colors.items():
        colors = []
        for c, raw in lst:
            if c not in colors:
                colors.append(c)
        if len(colors) > 1:
            findings.append({
                "op": "flag", "kind": "color-conflict",
                "target": f"{attr}: {' vs '.join(colors)}",
                "reason": "contradictory colors; first declaration wins unless heterochromia is intended",
                "points": COLOR_CONFLICT_POINTS * (len(colors) - 1),
                "remove_norms": {},
            })

    # 5. generic subsumed by specific
    normset = set(norms)
    for base in SUBSUME_BASES:
        if base in normset:
            specific = [n for n in normset if n != base and n.endswith(" " + base)]
            if specific:
                findings.append({
                    "op": "flag", "kind": "subsumed",
                    "target": base,
                    "reason": f"generic tag subsumed by: {', '.join(sorted(specific)[:4])}",
                    "points": SUBSUMED_POINTS,
                    "remove_norms": {},
                })

    # negative prompt: duplicates only
    neg_toks = _tokenize(negative)
    neg_norms = [_norm(t) for t in neg_toks]
    neg_seen, neg_dupes = {}, []
    for raw, n in zip(neg_toks, neg_norms):
        if n in neg_seen:
            neg_dupes.append(raw)
        else:
            neg_seen[n] = raw
    if neg_dupes:
        findings.append({
            "op": "remove", "kind": "negative-duplicate",
            "target": ", ".join(neg_dupes),
            "reason": "exact duplicates in the negative prompt",
            "points": DUPE_POINTS, "remove_norms": {},
            "negative": True, "remove_negative_norms":
                {n: sum(1 for x in neg_norms if x == n) - 1
                 for n in set(neg_norms)
                 if neg_norms.count(n) > 1},
        })

    score = min(SCORE_CAP, sum(f["points"] for f in findings))
    protected_skipped = [t for t in prot]
    findings.sort(key=lambda f: -f["points"])
    return {"score": score, "findings": findings,
            "protected_skipped": protected_skipped}


def _strip_tokens(text: str, removals: dict) -> str:
    """Remove removals[norm] EXTRA occurrences per token, keeping the
    first. Budget is global across segments and lines (the deck's prompt
    format is line-per-tag, so a per-line budget would delete every
    occurrence instead of deduping)."""
    remaining = dict(removals)
    kept_once = set()
    segments = re.split(r"\s*BREAK\s*", text or "")
    out_segments = []
    for seg in segments:
        out_lines = []
        for line in seg.split("\n"):
            kept = []
            for tok in line.split(","):
                if not tok.strip():
                    continue
                n = _norm(tok)
                if n in kept_once and remaining.get(n, 0) > 0:
                    remaining[n] -= 1
                    continue
                kept.append(tok.strip())
                kept_once.add(n)
            if kept:
                out_lines.append(", ".join(kept) + ",")
        out_segments.append("\n".join(out_lines))
    result = "\nBREAK\n".join(out_segments)
    return re.sub(r"\n?BREAK\s*$", "", result).rstrip(",").rstrip()


def apply(prompt: str, negative: str, selected: list) -> dict:
    """selected: the findings (op remove) chosen by the user."""
    pos_removals, neg_removals = {}, {}
    for f in selected or []:
        if f.get("negative"):
            for k, v in (f.get("remove_negative_norms") or {}).items():
                neg_removals[k] = max(neg_removals.get(k, 0), v)
        else:
            for k, v in (f.get("remove_norms") or {}).items():
                pos_removals[k] = pos_removals.get(k, 0) + v
    new_prompt = _strip_tokens(prompt, pos_removals) if pos_removals else prompt
    new_negative = _strip_tokens(negative, neg_removals) if neg_removals else negative
    return {"prompt": new_prompt, "negative_prompt": new_negative}
