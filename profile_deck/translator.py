"""Prompt translator: normalize archived prompts to the deck's lane.

Target lane (user-declared): YOUR_CHECKPOINT + YOUR_STYLE_LORA @ 0.8,
source_cartoon, score scaffold, 1girl/solo convention.

Design: a rules table for the mechanical 80 percent, a receipt (changes
list) for the judgment tail, and NO side effects. The UI shows the diff
and applies only on explicit confirm.

BREAK-safe: prompts are processed per segment (BREAK separates labeled
sections in the deck's format); the scaffold and character-count tags go
in the first segment only.
"""
import re

SCORE_SCAFFOLD = "score_9, score_8_up, score_7_up, score_6_up"
LANE_SOURCE = "source_cartoon"
LANE_LORA = "YOUR_STYLE_LORA"
LANE_WEIGHT = 0.8

LORA_PARSE = re.compile(r"<(?:lora|lyco):([^:>]+):([^>]*)>")

CHARACTER_COUNT_TAGS = {
    "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
    "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys",
    "solo", "multiple_girls", "multiple_boys", "duo_focus",
}

MULTIPLE_TAGS = {
    "2girls", "3girls", "4girls", "5girls", "6+girls",
    "2boys", "3boys", "4boys", "5boys", "6+boys",
    "multiple_girls", "multiple_boys", "duo_focus",
}

# tokens that look like character counts but are not danbooru tags
BAD_COUNT_PATTERNS = re.compile(r"^\d*\s*(woman|women|man|men|people|person|female|male)$")

# Illustrious/NoobAI/SDXL quality tokens that carry no weight on Pony.
ILLUSTRIOUS_STRIP = {
    "masterpiece", "best quality", "high quality", "high resolution",
    "very aesthetic", "absurdres", "newest", "professional",
    "official art", "finely detailed", "extremely detailed",
    "worst quality", "normal quality", "lowres", "junk of quality",
}

# Style-lora variants that all normalize to the canonical lane lora.
# Forms are fully separator-collapsed (spaces, underscores, hyphens).
LANE_LORA_VARIANTS = set()  # EDIT: add your style-lora variant names (separator-collapsed)

# Style carriers the lane lora replaces (separator-collapsed forms).
STYLE_DENYLIST = set()  # EDIT: add style-carrier lora names to strip on your lane


def _norm(name: str) -> str:
    """Collapse all separator classes; used for lora-name matching."""
    return re.sub(r"[\s_-]+", " ", (name or "").strip().lower()).strip()


def _spaceless(s: str) -> str:
    return re.sub(r"[\s_-]+", "", (s or "").lower())


def _tag_norm(token: str) -> str:
    """Prompt-token form: drop emphasis brackets and :weight suffixes.
    Underscores are PRESERVED (danbooru tags are underscored)."""
    t = re.sub(r"[()]+", "", token or "")
    t = t.split(":")[0]
    return re.sub(r"[\s-]+", " ", (t or "").strip().lower()).strip()


def _spaceless_tag(token: str) -> str:
    return _spaceless(_tag_norm(token))


def detect_source(prompt: str) -> str:
    t = prompt.lower()
    if "score_9" in t or "source_cartoon" in t or "source_anime" in t:
        return "pony"
    if "masterpiece" in t or "best quality" in t or "masterwork" in t or "best_quality" in t:
        return "illustrious"
    return "sdxl"


def _segment_tokens(prompt: str):
    """Split into BREAK segments, then comma tokens per segment."""
    segments = re.split(r"\s*BREAK\s*", prompt)
    return [[t.strip() for t in seg.split(",") if t.strip()] for seg in segments]


def translate(prompt: str, negative: str, loras: list) -> dict:
    changes = []
    prompt, negative = prompt or "", negative or ""
    src = detect_source(prompt)

    # loras: inline tags join the passed list, then leave the prompt text
    def _w(w):
        try:
            return float(w)
        except (TypeError, ValueError):
            return 0.8

    all_loras = [{"name": n, "weight": _w(w)} for n, w in LORA_PARSE.findall(prompt)]
    for l in loras or []:
        if l.get("name"):
            all_loras.append({"name": l["name"], "weight": _w(l.get("weight", 0.8))})

    segments = _segment_tokens(prompt)
    loose_tags = []
    for seg in segments:
        keep = []
        for t in seg:
            if LORA_PARSE.fullmatch(t.strip()):
                loose_tags.append(t)
            else:
                keep.append(t)
        seg[:] = keep
    if loose_tags:
        changes.append({"op": "strip", "target": ", ".join(loose_tags),
                        "reason": "lora tags move to the profile's lora list", "apply": True})

    flat = [(_tag_norm(t), t) for seg in segments for t in seg]

    # 1. quality tokens: always stripped on the lane (any source family)
    illu = {_spaceless(s) for s in ILLUSTRIOUS_STRIP}
    removed = [orig for low, orig in flat if _spaceless(low) in illu]
    if removed:
        segments = [[t for t in seg if _spaceless(_tag_norm(t)) not in illu]
                    for seg in segments]
        changes.append({"op": "strip", "target": ", ".join(removed),
                        "reason": "quality tokens carry no weight on the Pony lane",
                        "apply": True})

    # 2. score scaffold (exact score_9 token; score_9_up alone is not a scaffold)
    flat = [(_tag_norm(t), t) for seg in segments for t in seg]
    has_score = any(low == "score_9" for low, _ in flat)
    if not has_score:
        present = {low for low, _ in flat}
        add = [s for s in SCORE_SCAFFOLD.split(", ") if s not in present]
        if LANE_SOURCE not in present:
            add.append(LANE_SOURCE)
        if add:
            segments[0] = add + segments[0]
            changes.append({"op": "add", "target": ", ".join(add),
                            "reason": "the lane's quality grammar (score scaffold + source_cartoon)",
                            "apply": True})
    else:
        swapped = removed_anime = False
        for seg in segments:
            has_cartoon = any(_tag_norm(t) == "source_cartoon" for t in seg)
            keep = []
            for t in seg:
                if _tag_norm(t) == "source_anime":
                    if has_cartoon:
                        removed_anime = True  # cartoon already present: drop the anime tag
                        continue
                    t = LANE_SOURCE
                    swapped = True
                keep.append(t)
            seg[:] = keep
        if swapped:
            changes.append({"op": "normalize", "target": "source_anime -> source_cartoon",
                            "reason": "the deck lane renders cartoon; source_anime pulls the anime prior",
                            "apply": True})
        if removed_anime:
            changes.append({"op": "strip", "target": "source_anime",
                            "reason": "duplicate lane source; source_cartoon already present",
                            "apply": True})
        if not any(_tag_norm(t) == "source_cartoon" for seg in segments for t in seg):
            insert_at = 0
            for i, t in enumerate(segments[0]):
                tn = _tag_norm(t)
                if tn.startswith("score_") or tn.startswith("source_"):
                    insert_at = i + 1
                else:
                    break
            segments[0].insert(insert_at, LANE_SOURCE)
            changes.append({"op": "add", "target": LANE_SOURCE,
                            "reason": "the lane source was missing", "apply": True})
        tail = [low for low, _ in flat
                if low in ("score_5_up", "score_4_up", "score_3_up", "score_2_up", "score_1_up")]
        if tail:
            changes.append({"op": "flag", "target": ", ".join(sorted(set(tail))),
                            "reason": "non-standard score tail; the lane scaffold stops at score_6_up",
                            "apply": False})

    # 3. character-count tags (first segment only)
    flat = [(_tag_norm(t), t) for seg in segments for t in seg]
    found = [low for low, _ in flat if low in CHARACTER_COUNT_TAGS]
    bad_counts = [orig for low, orig in flat
                  if low not in CHARACTER_COUNT_TAGS and BAD_COUNT_PATTERNS.match(low)]
    if not found:
        insert_at = 0
        for i, (low, _) in enumerate(flat):
            if low.startswith("score_") or low.startswith("source_"):
                insert_at = i + 1
        segments[0][insert_at:insert_at] = ["solo", "1girl"]
        changes.append({"op": "add", "target": "1girl, solo",
                        "reason": "no character-count tag found; deck convention",
                        "apply": True})
    elif "solo" in found and any(m in found for m in MULTIPLE_TAGS):
        changes.append({"op": "flag", "target": "solo alongside multiple-character tags",
                        "reason": "contradictory character-count tags; review before generating",
                        "apply": False})
    if bad_counts:
        changes.append({"op": "flag", "target": ", ".join(bad_counts),
                        "reason": "not danbooru character-count tags (1girl/1boy); consider 1girl",
                        "apply": False})

    translated_prompt = "\nBREAK\n".join(", ".join(seg) for seg in segments)

    # 4. loras: normalize style-lora variants, strip style carriers, keep the rest
    final_loras, lane_lora_present, kept = [], False, 0
    style_tokens_stripped = []
    for l in all_loras:
        n = _norm(l.get("name"))
        ns = _spaceless(l.get("name"))
        if not n:
            continue
        # archived names often carry the file extension; the tables are
        # extension-free, so strip before matching
        n = re.sub(r"\.safetensors$", "", n).strip()
        ns = re.sub(r"safetensors$", "", ns)
        if n == _norm(LANE_LORA) or ns in {_spaceless(v) for v in LANE_LORA_VARIANTS}:
            if not lane_lora_present:
                final_loras.insert(0, {"name": LANE_LORA, "weight": LANE_WEIGHT})
                lane_lora_present = True
            try:
                had_w = float(l.get("weight", LANE_WEIGHT))
            except (TypeError, ValueError):
                had_w = LANE_WEIGHT
            if abs(had_w - LANE_WEIGHT) > 1e-9:
                changes.append({"op": "normalize",
                                "target": f"lora {LANE_LORA} weight {had_w} -> {LANE_WEIGHT}",
                                "reason": "the lane style lora runs at the tuned weight",
                                "apply": True})
            if n != _norm(LANE_LORA):
                changes.append({"op": "normalize",
                                "target": f"lora {l.get('name')} -> {LANE_LORA} @ {LANE_WEIGHT}",
                                "reason": "all style-lora variants normalize to the lane style lora",
                                "apply": True})
            continue
        if n in STYLE_DENYLIST or ns in {_spaceless(s) for s in STYLE_DENYLIST}:
            changes.append({"op": "strip", "target": f"lora {l.get('name')}",
                            "reason": "style carrier; the lane lora carries the style",
                            "apply": True})
            continue
        final_loras.append(l)
        kept += 1
    if not lane_lora_present:
        final_loras.insert(0, {"name": LANE_LORA, "weight": LANE_WEIGHT})
        changes.append({"op": "add", "target": f"lora {LANE_LORA} @ {LANE_WEIGHT}",
                        "reason": "the lane's style lora", "apply": True})
    if kept:
        changes.append({"op": "keep", "target": f"{kept} lora(s)",
                        "reason": "character/detail/utility loras pass through", "apply": True})

    # 5. style-lora names riding as bare prompt tokens
    flat2 = [(_tag_norm(t), _spaceless_tag(t), t) for seg in segments for t in seg]
    style_tokens = [orig for low, sp, orig in flat2
                    if low in STYLE_DENYLIST or sp in {_spaceless(s) for s in STYLE_DENYLIST}]
    if style_tokens:
        kill = {_tag_norm(t) for t in style_tokens}
        segments = [[t for t in seg if _tag_norm(t) not in kill] for seg in segments]
        style_tokens_stripped = style_tokens
        changes.append({"op": "strip", "target": ", ".join(style_tokens),
                        "reason": "style lora names as prompt tokens; the lane lora carries the style",
                        "apply": True})
        translated_prompt = "\nBREAK\n".join(", ".join(seg) for seg in segments)

    return {
        "source": src,
        "original_prompt": prompt,
        "prompt": translated_prompt,
        "negative_prompt": negative,
        "loras": final_loras,
        "changes": changes,
    }
