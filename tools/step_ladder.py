#!/usr/bin/env python
"""Steps ladder: find where base sampling plateaus on Euler a Karras.

Renders the same seed and prompt at several step counts, detects the face
in each render, and scores: texture energy (mean |Laplacian| on L) inside
the face box, Lab skin tone in the box, and how much the image changes
between neighboring step counts. Also builds a labeled face-crop strip.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from harness import api_post, srgb_to_lab  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LADDER = RESULTS / "ladder"
LADDER.mkdir(parents=True, exist_ok=True)

POS = """score_9, score_8_up, score_7_up, score_6_up, source_cartoon, 1girl, solo,
35mm portrait, medium shot, three-quarter view, eye-level,
standing, relaxed posture, weight shifted,
direct eye contact looking at viewer,
soft confident expression,
single catchlight,
BREAK
(mature woman:1.1), (in her mid 30s:1.5), nordic, heart shaped face, slim dancer body, long legs,
long red hair, loose tendrils framing face,
(bright blue eyes:1.3), adult face, voluptuous sagging breasts,
(white buttoned blouse:1.4), (fitted pencil skirt:1.3), gold belt, delicate gold necklace with pendant,
BREAK
modern office interior, daytime, diffused window light, softbox key camera left,
<lora:incase_style_v3_ponyxl:0.8>"""

NEG = """score_6, score_5, score_4, lowres, bad anatomy, bad hands, signature, watermarks, ugly, error, extra limb, missing limbs, bad art, bad painting, bad photo, bad image, deformed body, merged limbs, badly drawn face, ugly face, cross-eyed, young woman, 20s, child, teen, teenage, underage, youthful, baby face, chibi, masculine features, blue streaks in hair, fit body, flat chest, cars, automobiles, harsh sunlight, overhead fluorescent, hard shadows, high contrast, overexposed highlights, multiple catchlights, monochrome, sketch"""

STEPS = [14, 16, 18, 20, 22, 24]
SEED = 3940102728


def render(steps):
    p = {
        "prompt": POS,
        "negative_prompt": NEG,
        "seed": SEED,
        "steps": steps,
        "cfg_scale": 6,
        "distilled_cfg_scale": 3.0,
        "sampler_name": "Euler a",
        "scheduler": "Karras",
        "width": 768,
        "height": 1365,
        "batch_size": 1,
        "n_iter": 1,
        "save_images": True,
        "send_images": True,
        "hr_additional_modules": ["Use same choices"],
        "override_settings": {"sd_model_checkpoint": "Pony/NIXES_v5.5.43.safetensors"},
        "alwayson_scripts": {"ADetailer": {"args": [{
            "ad_model": "None", "ad_tab_enable": False, "is_api": True}]}},
    }
    resp = api_post("/sdapi/v1/txt2img", p)
    import base64
    from io import BytesIO
    img = Image.open(BytesIO(base64.b64decode(resp["images"][0]))).convert("RGB")
    out = LADDER / f"steps_{steps:02d}.png"
    img.save(out)
    return out


def laplacian_mean(gray):
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(gray, (3, 3))
    return float(np.abs((win * k).sum(axis=(-2, -1))).mean())


def analyze(paths):
    from ultralytics import YOLO
    model = YOLO("/home/rell/sd-webui-forge-neo/models/adetailer/face_yolov8n.pt")

    rows, boxes, crops = [], [], []
    for p in steps_labels(paths):
        path, steps = p
        img = np.asarray(Image.open(path), dtype=np.float64)
        h, w = img.shape[:2]
        r = model.predict(path, conf=0.3, verbose=False)[0]
        if len(r.boxes) == 0:
            rows.append({"steps": steps, "face": False})
            boxes.append(None)
            crops.append(None)
            continue
        x0, y0, x1, y1 = [int(v) for v in r.boxes.xyxy[0].tolist()]
        bx0, by0, bx1, by1 = shrink(x0, y0, x1, y1, w, h, 0.15)
        lab = srgb_to_lab(img / 255.0)
        box = lab[by0:by1, bx0:bx1]
        L = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]) / 255.0
        rows.append({
            "steps": steps,
            "face": True,
            "bbox": [x0, y0, x1, y1],
            "texture": round(laplacian_mean(L[by0:by1, bx0:bx1]), 5),
            "L": round(float(box[..., 0].mean()), 2),
            "a": round(float(box[..., 1].mean()), 2),
            "b": round(float(box[..., 2].mean()), 2),
        })
        boxes.append((bx0, by0, bx1, by1))
        crop = img[by0:by1, bx0:bx1].astype(np.uint8)
        crops.append((steps, crop))

    # pairwise change between neighboring step counts
    for i in range(1, len(paths)):
        ia = np.asarray(Image.open(paths[i - 1]), dtype=np.float64)
        ib = np.asarray(Image.open(paths[i]), dtype=np.float64)
        h = min(ia.shape[0], ib.shape[0]); w = min(ia.shape[1], ib.shape[1])
        d = float(np.abs(ia[:h, :w] - ib[:h, :w]).mean())
        rows[i]["delta_prev"] = round(d, 2)

    # contact sheet
    present = [c for c in crops if c is not None]
    if present:
        scale = 2
        ch = max(c.shape[0] for _, c in present) * scale
        cw = sum(c.shape[1] * scale + 8 for _, c in present) + 8
        sheet = Image.new("RGB", (cw, ch + 26), (20, 20, 20))
        x = 8
        dr = ImageDraw.Draw(sheet)
        for steps, crop in crops:
            if crop is None:
                continue
            c = Image.fromarray(crop).resize((crop.shape[1] * scale, crop.shape[0] * scale), Image.LANCZOS)
            sheet.paste(c, (x, 24))
            dr.text((x + 4, 6), f"steps {steps}", fill=(240, 240, 240))
            x += c.width + 8
        sheet.save(LADDER / "face_strip.png")

    (LADDER / "ladder.json").write_text(json.dumps(rows, indent=1))
    return rows


def steps_labels(paths):
    return list(zip(paths, STEPS))


def shrink(x0, y0, x1, y1, w, h, f):
    dx, dy = int((x1 - x0) * f), int((y1 - y0) * f)
    return max(0, x0 + dx), max(0, y0 + dy), min(w, x1 - dx), min(h, y1 - dy)


if __name__ == "__main__":
    paths = []
    for s in STEPS:
        paths.append(render(s))
        print(f"rendered steps={s}", flush=True)
    rows = analyze(paths)
    hdr = f"{'steps':>5} {'texture':>9} {'L':>6} {'a':>6} {'b':>6} {'delta_prev':>10}"
    print(hdr)
    for r in rows:
        if not r.get("face"):
            print(f"{r['steps']:>5}  no face detected")
            continue
        print(f"{r['steps']:>5} {r['texture']:>9.5f} {r['L']:>6.2f} {r['a']:>6.2f} {r['b']:>6.2f} "
              f"{r.get('delta_prev',''):>10}")
