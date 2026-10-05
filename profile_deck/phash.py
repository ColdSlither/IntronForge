"""Perceptual hashing for the library: DCT pHash in pure numpy/PIL.

64-bit hash from the top-left 8x8 DCT coefficients of a 32x32 grayscale
downscale. Similar images -> small Hamming distance (0 = identical,
<= 6 near-duplicate, <= 12 same scene/pose).
"""
from pathlib import Path

import numpy as np
from PIL import Image

_DCT = None


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    r = np.arange(n).reshape(1, -1)
    m = np.sqrt(2.0 / n) * np.cos((2 * r + 1) * k * np.pi / (2 * n))
    m[0] /= np.sqrt(2.0)
    return m


def phash(path: Path) -> str:
    global _DCT
    if _DCT is None:
        _DCT = _dct_matrix(32)
    img = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float64)
    d = _DCT @ a @ _DCT.T
    top = d[:8, :8].flatten()
    med = np.median(top[1:])
    v = 0
    for b in (top > med):
        v = (v << 1) | int(b)
    return f"{v:016x}"


def hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")
