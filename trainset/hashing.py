"""Image fingerprints for leak checks between training sources and the eval suite.

- sha256 of the file bytes: byte-identical copies.
- pHash (64 bit): DCT of a 32×32 grey thumbnail, with the 8×8 lowest frequencies thresholded at their median (DC
  excluded). Survives resizing and recompression. Hamming distance ≤ 8 is treated as a near-duplicate.
- Tile hashes: pHash of every tile in a 2×2 and a 3×3 grid. An eval image that is a crop of a training scene (or the
  reverse) then shares a hash with one of the other side's tiles. Images under 256 px a side get no tile hashes,
  because their tiles are too small to fingerprint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # eval scenes reach hundreds of megapixels; we only read local, already-vetted files

_N = 32
_DCT = np.array([[np.cos(np.pi * (2 * x + 1) * u / (2 * _N)) for x in range(_N)] for u in range(_N)])
GRIDS = (2, 3)
MIN_TILE_SIDE = 256


def phash(image: Image.Image) -> int:
    grey = np.asarray(image.convert("L").resize((_N, _N), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_DCT @ grey @ _DCT.T)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    return int(np.packbits(bits.astype(np.uint8), bitorder="little").view("<u8")[0])


def tiles(image: Image.Image, grid: int):
    w, h = image.size
    for i in range(grid):
        for j in range(grid):
            box = (j * w // grid, i * h // grid, (j + 1) * w // grid, (i + 1) * h // grid)
            yield i, j, image.crop(box)


def fingerprint(path: str | Path) -> dict:
    """{"sha256", "size": [w, h], "hashes": [[grid, i, j, phash], ...]}; grid 1 is the whole image."""
    data = Path(path).read_bytes()
    out = {"sha256": hashlib.sha256(data).hexdigest()}
    with Image.open(path) as im:
        out["size"] = list(im.size)  # the file's own size, before any reduced decode
        im.draft("RGB", (2048, 2048))  # JPEG: decode at reduced size; hashes only need a thumbnail per tile
        im = im.convert("RGB")
        hashes = [[1, 0, 0, phash(im)]]
        if min(im.size) >= MIN_TILE_SIDE:
            for g in GRIDS:
                hashes += [[g, i, j, phash(t)] for i, j, t in tiles(im, g)]
    out["hashes"] = hashes
    return out


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise Hamming distance between uint64 arrays."""
    x = np.bitwise_xor(a.astype(np.uint64), b.astype(np.uint64))
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
