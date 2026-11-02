"""Read xBD from the HF mirror kshitijrajsharma/xview2-xbd (test + hold parquet; user decision 2026-10-16).

Each mirror row is one scene: t1_image / t2_image (pre / post RGB PNG, 1024x1024), t1_mask (pre: 0 background,
1 building) and t2_mask (post: 0 background, 1 no damage, 2 minor, 3 major, 4 destroyed), image_name like
"test/images/socal-fire_00000823".
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow.parquet as pq

from bench.common import RAW

XBD = RAW / "xbd" / "data"


def scenes(columns: list[str] | None = None, splits: tuple[str, ...] = ("test", "hold")) -> Iterator[dict]:
    for split in splits:
        for f in sorted(XBD.glob(f"{split}-*.parquet")):
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                for row in pf.read_row_group(g, columns=columns).to_pylist():
                    row["split"] = split
                    row["scene"] = row["image_name"].rsplit("/", 1)[-1]
                    yield row


def images_by_original_name(names: set[str]) -> dict[str, bytes]:
    """Original xBD file names ("socal-fire_00000823_post_disaster.png") -> PNG bytes from the mirror."""
    wanted: dict[str, list[tuple[str, str]]] = {}
    for name in names:
        stem = name.rsplit(".", 1)[0]
        for suffix, column in (("_post_disaster", "t2_image"), ("_pre_disaster", "t1_image")):
            if stem.endswith(suffix):
                wanted.setdefault(stem[: -len(suffix)], []).append((name, column))
                break
        else:
            raise ValueError(f"not an xBD image name: {name}")
    out: dict[str, bytes] = {}
    for row in scenes(columns=["image_name", "t1_image", "t2_image"]):
        for name, column in wanted.get(row["image_name"].rsplit("/", 1)[-1], []):
            out[name] = row[column]["bytes"]
    missing = names - set(out)
    if missing:
        raise FileNotFoundError(f"{len(missing)} xBD images not in the mirror's test/hold splits, e.g. {sorted(missing)[:3]}")
    return out
