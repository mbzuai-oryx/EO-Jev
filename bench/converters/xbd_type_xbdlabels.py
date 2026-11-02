"""xBD disaster type with xBD's own labels (fire, flooding, wind, earthquake, tsunami, volcano), the answer format the
xBD-trained models and the dataset's own metadata use. Same 933 test scenes, images and question as
xbd_disaster_type.jsonl (which uses event-based labels: four hurricanes = "hurricane", plus "tornado"); options are
shuffled per scene with the same seeding.

Checked against the released xBD test metadata (disaster_type in xview2_test.json): 933 / 933 scenes agree.

    python -m bench.converters.xbd_type_xbdlabels
"""

from __future__ import annotations

import random

from bench.common import CONVERTED, read_jsonl, validate_row, write_jsonl
from bench.converters.xbd_damage import DISASTER_TYPES_XBD, EVENT_TYPE_XBD


def main() -> None:
    src = CONVERTED / "xbd" / "xbd_disaster_type.jsonl"
    out = []
    for row in read_jsonl(src):
        scene = row["group_id"].split(":", 1)[1]
        options = DISASTER_TYPES_XBD[:]
        random.Random(scene).shuffle(options)
        q = dict(row["questions"]["q"], criteria={o: None for o in options})
        r = {**row, "id": row["id"].replace("xBD-type:", "xBD-type-xbd:"), "benchmark": "xBD-disaster-type-xbd",
             "questions": {"q": q}, "answer": {"q": EVENT_TYPE_XBD[row["meta"]["event"]]},
             "meta": {**row["meta"], "label_set": "xbd"}}
        validate_row(r)
        out.append(r)
    write_jsonl(CONVERTED / "xbd" / "xbd_disaster_type_xbdlabels.jsonl", out)
    print(f"wrote {len(out)} rows")


if __name__ == "__main__":
    main()
