"""WD14 tagger: local ONNX inference over the library's renders.

Model: SmilingWolf/wd-swinv2-tagger-v3 (448px, stored in wd14/).
Preprocess: RGB, pad to square on white, resize to 448. Thresholds:
0.35 general, 0.35 character, rating heads excluded. Tags land in the
library index under "wd14" (separate from user tags), so library tag
filters search both while curation stays clean.
"""
import csv
import threading
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent / "wd14"
MODEL = MODEL_DIR / "model.onnx"
TAGS = MODEL_DIR / "selected_tags.csv"
SIZE = 448
GENERAL_THRESHOLD = 0.35
CHAR_THRESHOLD = 0.35
MAX_TAGS = 40

_session = None
_tag_names = None


def available() -> bool:
    return MODEL.is_file() and TAGS.is_file()


_LOAD_LOCK = threading.Lock()


def _load():
    global _session, _tag_names, _categories
    if _session is not None:
        return
    with _LOAD_LOCK:
        if _session is not None:
            return
        import onnxruntime
        names, cats = [], []
        with open(TAGS, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row[0] == "tag_id":
                    continue
                names.append(row[1])
                cats.append(int(row[2]))
        session = onnxruntime.InferenceSession(
            str(MODEL), providers=["CPUExecutionProvider"])
        # publish names/categories BEFORE the session flag so a concurrent
        # reader that passes the _session check never sees empty tags
        _tag_names, _categories = names, cats
        _session = session


def tag_image(path: Path) -> dict:
    _load()
    img = Image.open(path).convert("RGBA")
    size = max(img.size)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2), mask=img)
    arr = np.asarray(canvas.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float32)
    # BGR, as the model expects
    arr = arr[:, :, ::-1]
    out = _session.run(None, {(_session.get_inputs()[0].name): arr[None, ...]})[0][0]

    general, character = [], []
    for name, cat, conf in zip(_tag_names, _categories, out):
        if cat == 0 and conf >= GENERAL_THRESHOLD:
            general.append((name, float(conf)))
        elif cat == 4 and conf >= CHAR_THRESHOLD:
            character.append((name, float(conf)))
        # cat 9 is the rating head (general/sensitive/nsfw/explicit): excluded
    general.sort(key=lambda x: -x[1])
    character.sort(key=lambda x: -x[1])
    tags = ([n for n, _ in general] + [n for n, _ in character])[:MAX_TAGS]
    return {"tags": tags, "top": [{"tag": n, "conf": round(c, 3)} for n, c in general[:12]]}
