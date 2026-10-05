"""Section composer: the user's prompt format as structure.

Six logical sections (user-declared, from Dorian_White's Pony Prompts
for Dummies, adapted; matches the banked example prompts):

  quality    1. Quality and Style
  keystone   2. Keystone, Composition, Posture, Gesture, Action, Supporting
  character  3. Body, Hair, Face, Clothes and Details
  secondary  4. Secondary Character
  setting    5. Setting
  lighting   6. Lighting Detail

Compose groups them into BREAK blocks the way the banked prompts do:
[quality+keystone] BREAK [character] BREAK [secondary] BREAK [setting+lighting].

Parsing is line-level and conservative: lines are classified by content
heuristics; unclassified lines take positional defaults (first segment
leans quality, last leans setting, middle leans character). Empty BREAK
segments collapse on round-trip: legacy prompts with doubled or trailing
BREAKs normalize to the six-section layout. The prompt string remains
the single source of truth; sections are a structured view, not a
storage format.
"""
import re

SECTION_ORDER = ["quality", "keystone", "character", "secondary", "setting", "lighting"]
SECTION_LABELS = {
    "quality": "1 · Quality and Style",
    "keystone": "2 · Keystone, Composition, Posture, Action",
    "character": "3 · Body, Hair, Face, Clothes, Details",
    "secondary": "4 · Secondary Character",
    "setting": "5 · Setting",
    "lighting": "6 · Lighting Detail",
}

_RE = {
    "score": re.compile(r"\bscore_\d", re.I),
    "source": re.compile(r"\bsource_\w+", re.I),
    "count": re.compile(r"^\(?\d\s*(girl|boy|women|men|other)s?\)?$", re.I),
    "solo": re.compile(r"\bsolo\b", re.I),
    "shot": re.compile(r"\b(portrait|medium shot|full body|close-?up|cowboy shot|\d+mm|"
                       r"three-?quarter|from (above|below|side|behind)|dutch angle|"
                       r"eye-?level|pov|wide (shot|angle)|focus|dynamic angle|angle)\b", re.I),
    "posture": re.compile(r"\b(standing|sitting|kneeling|lying|leaning|walking|"
                          r"posture|weight shifted|arms? (up|crossed|behind)|hands? on|"
                          r"head (tilt|rest)|eye contact|looking at viewer|smil(?:e|ing)|"
                          r"expression|catchlight|smirk|frown|laughing)\b", re.I),
    "body": re.compile(r"\b(breasts?|curvy|slim|voluptuous|athletic|plump|muscular|"
                       r"skin tone|pale skin|dark-?skinned|tan(?!k)|freckles|"
                       r"long legs|narrow waist|wide hips|thick thighs|dancer body|"
                       r"mature woman|aged up|mid 30s|\d0s\b|milf|body type)\b", re.I),
    "hair": re.compile(r"\b(hair|bangs|ponytail|braid|twintails|curls|tendrils|updo)\b", re.I),
    "face": re.compile(r"\b(eyes?:\d|blue eyes|green eyes|brown eyes|amber eyes|hazel|"
                       r"face|lips|makeup|moles?|pointed ears|horns)\b", re.I),
    "clothes": re.compile(r"\b(blouse|shirt|skirt|dress|pants|jeans|shorts|bikini|"
                          r"lingerie|wearing|outfit|jacket|coat|sweater|top|"
                          r"belt|necklace|earrings|jewelry|pendant|glasses|boots|heels|"
                          r"stockings|pantyhose|gloves|hat)\b", re.I),
    "charrole": re.compile(r"\b(mature woman|1girl|1boy|woman|man|female|male)\b", re.I),
    "secondary_hint": re.compile(r"\b(1boy|male|man|second (girl|woman)|another (girl|woman)|"
                                 r"mmf|threesome|couple)\b", re.I),
    "setting": re.compile(r"\b(interior|outdoors?|outside|inside|room|office|beach|street|"
                          r"bedroom|kitchen|garden|forest|city|background|"
                          r"daytime|night(?:time)?|evening|morning|sunset|sunrise|"
                          r"living room|desk|couch|sofa|bed\b|window)\b", re.I),
    "lighting": re.compile(r"\b(light|lighting|lamp|softbox|rim light|backlit|"
                           r"golden hour|glow|shadows?|sunlight|candle|neon)\b", re.I),
}

# order matters: earlier rules win ties
_LINE_RULES = [
    ("quality", re.compile(r"(" + _RE["score"].pattern + r"|" + _RE["source"].pattern + r")", re.I)),
    ("keystone", _RE["shot"]),
    ("keystone", _RE["posture"]),
    ("character", _RE["hair"]),
    ("character", _RE["face"]),
    ("character", _RE["clothes"]),
    ("character", _RE["body"]),
    ("setting", _RE["setting"]),
    ("lighting", _RE["lighting"]),
]


def _classify_line(line: str, seg_index: int, seg_count: int) -> str:
    low = line.lower()
    for name, rx in _LINE_RULES:
        if rx.search(low):
            return name
    if _RE["count"].match(low.strip()) or _RE["solo"].search(low):
        return "keystone"
    if _RE["charrole"].search(low):
        return "character"
    # positional defaults: first segment starts as quality, last leans setting
    if seg_index == 0:
        return "quality"
    if seg_index == seg_count - 1:
        return "setting"
    return "character"


def parse(prompt: str) -> dict:
    """Prompt text -> six named sections (lists of lines)."""
    prompt = prompt or ""
    segments = re.split(r"\s*BREAK\s*", prompt)
    sections = {k: [] for k in SECTION_ORDER}
    seen_character = False
    for si, seg in enumerate(segments):
        lines = [l.strip() for l in seg.split("\n") if l.strip()]
        # split each physical line into logical comma-lines? No: keep the
        # author's line structure; classification is per line.
        last = None
        for line in lines:
            # secondary detection: a character-classified line with male/role
            # hints in a segment AFTER the first character segment
            name = _classify_line(line, si, len(segments))
            if name == "character":
                if seen_character and _RE["secondary_hint"].search(line.lower()):
                    name = "secondary"
                else:
                    seen_character = True
            elif name == "setting" and last == "character" and not sections["setting"]:
                pass
            sections[name].append(line)
            last = name
    return {"sections": sections}


def compose(sections: dict) -> str:
    """Six sections -> BREAK-grouped prompt text (banked-prompt layout)."""
    def clean(name):
        out = []
        for l in (sections.get(name) or []):
            l = (l or "").strip().strip(",").strip()
            if l:
                out.append(l)
        return out

    blocks = []
    block_a = clean("quality") + clean("keystone")
    if block_a:
        blocks.append(block_a)
    block_b = clean("character")
    if block_b:
        blocks.append(block_b)
    block_c = clean("secondary")
    if block_c:
        blocks.append(block_c)
    block_d = clean("setting") + clean("lighting")
    if block_d:
        blocks.append(block_d)
    return "\nBREAK\n".join(",\n".join(b) + "," for b in blocks)
