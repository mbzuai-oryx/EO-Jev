"""C20 RSHR-Bench (arXiv 2512.17319; 1,932 MCQ on 1,377 ultra-high-resolution images, 2.6-806 MP) -> rows.

Source: HF `RL-MIND/RSHR-Bench` @ f1e95190ca04845618aab220e274cd9b9e349693 (Apache-2.0): 197 parquet shards with the
original-resolution images embedded (225.6 GB). 1,704 ids; 1,932 questions = 1,590 single-turn + 114 three-turn
dialogues (342 turns). 1,861 are 4-option, 71 are 3-option (MRJC 20, MRJCS 50, one MTFP turn); 1,862 have one
image, 50 two (future prediction) and 20 three (multi-region joint contrast). Categories keep the paper's split
(Table 2), which `task_family` carries as a prefix because the two are reported separately (text-only models beat
VLMs on reasoning):
  perception_*  colour, shape, orientation (`detection`), object classification, relation, object grounding,
                regional grounding, object counting, regional counting                         (1,370 questions)
  reasoning_*   anomaly detection, future prediction (2 images), multi-region joint contrast (3 images),
                MRJC single-image (box options), object state judgement; and the multi-turn MAD / MTFP / MOSJ
                (task_family reasoning_multiturn_*)                                                (562 questions)

ADAPTED (resolution); report it as "not the published protocol". Two input variants, built as in xlrs.py:
  v0 "downscale": each image resized (Lanczos) to <= 1,638,400 px.
  v1 "tiles":     the v0 overview plus its four quadrants, each resized to <= 1,638,400 px (5 images per source
                  image, 10 / 15 for the 2- / 3-image items); the state names the layout.
Native pixel coordinates (1,040 questions): 720 questions give a reference box in the question text (colour,
shape, orientation, classification, regional counting; MRJC one box per image) and 320 give box options (object
grounding 200, regional grounding 70, MRJCS 50). Every "[x1, y1, x2, y2]" in the question and options is RESCALED
to the shown overview image (x * w_overview / w_native, y likewise, rounded), and every stated "resolution: W x H"
is replaced by the overview's size, so the option keys and the gold letter are unchanged; the state adds one line
saying which image the coordinates refer to (in v1: Image 1, the whole image, with the quadrant split points).
Boxes were not drawn on the images. Original question / options are in meta. MRJC items 411, 413 and 414 state one
resolution for three images of different sizes: there each image's overview size is listed ("1280 x 1280 (Image A),
..."). Item 180 states 4097 x 4096 for a 4096 x 4096 image (replaced like all others). 84 greyscale sources -> RGB.
Multi-turn: every turn is its own row (as the paper scores each turn: per-turn accuracy per category, plus a
dialogue-level exact match, MTEM@1 = all 3 turns correct; group by meta.dialogue_id). Earlier turns go into the
state as question + gold answer text (teacher forcing; the paper does not say what history it used).
Kept: all 1,932 questions, question and option wording (apart from the rescaled numbers), option order A-D.
Item 779 (object_counting) has four identical options "0": as in xlrs.py each text gets its letter appended
("0 (A)"), and meta.duplicate_options flags it.

    MODULE=bench.converters.rshr ARGS="--workers 5" sbatch slurm/convert_cpu.sbatch   (225 GB in; ~40 min, <32 GB)
    python -m bench.converters.rshr     (resumable: rendered images are listed in converted/rshr/images/_index.jsonl
                                         and skipped; tiles are written atomically)
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")  # give freed column buffers back to the OS at once
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from bench.common import CONVERTED, DATASETS, MAX_PIXELS, validate_row, write_jsonl  # noqa: E402
from bench.converters.mme_rw_rs import rel, render_tiles, rendered, tile_files, tile_sizes  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SNAP = (Path(os.environ.get("EOJEV_DATASET_HUB", DATASETS / "hf_hub")) / "datasets--RL-MIND--RSHR-Bench" / "snapshots"
        / "f1e95190ca04845618aab220e274cd9b9e349693")
OUT = CONVERTED / "rshr"
INDEX = OUT / "images" / "_index.jsonl"
META_COLS = ["id", "turn_id", "category", "question", "answer", "option_a", "option_b", "option_c", "option_d",
             "options", "image_count", "image_1_path", "image_2_path", "image_3_path", "source_images",
             "source_image_sizes"]
BIG = 400_000_000  # encoded bytes; the 26 "200m" images (~806 MP, ~1.1 GB PNG) need ~5 GB each to read and decode


def image_key(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", path.removeprefix("images/").rsplit(".", 1)[0])


def load_items() -> list[dict]:
    items = []
    for f in sorted(glob.glob(str(SNAP / "data" / "test-*.parquet"))):
        for i, r in enumerate(pq.read_table(f, columns=META_COLS).to_pylist()):
            r["_shard"], r["_row"] = f, i
            items.append(r)
    return items


# ---- images ------------------------------------------------------------------------------------------------------

def _render_shard(task: tuple[str, list[tuple[int, int, str]]]) -> list[dict]:
    """Decode and tile the given (row, slot, source path) images of one shard, one image at a time."""
    shard, wanted = task
    out = []
    for slot in sorted({s for _, s, _ in wanted}):
        col = pq.read_table(shard, columns=[f"image_{slot}"]).column(0)  # one column of one shard (~13 rows)
        blobs = [(row, path, col[row]["bytes"].as_py()) for row, s, path in wanted if s == slot]
        del col  # keep only the wanted images' bytes while decoding
        while blobs:
            row, path, data = blobs.pop(0)
            t0 = time.time()
            files = tile_files(OUT, image_key(path))
            with Image.open(io.BytesIO(data)) as im:
                if rendered(files):  # tiles of a killed run that never reached the index: read the header only
                    info = {"orig_px": list(im.size), "mode": im.mode, "format": im.format}
                else:
                    im.load()
                    info = render_tiles(im, files)
            del data
            out.append({"path": path, **info, "bytes_shard": shard.rsplit("/", 1)[1], "row": row, "slot": slot,
                        "sec": round(time.time() - t0, 1)})
    return out


def render_images(items: list[dict], workers: int) -> dict[str, dict]:
    """Render each source image once (from its first occurrence); returns {path: info} from the index."""
    done = {}
    if INDEX.exists():
        for line in INDEX.open():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # a line cut off by a killed job: that image is looked at again
                continue
            if rendered(tile_files(OUT, image_key(d["path"]))):
                done[d["path"]] = d
    first: dict[str, tuple[str, int, int]] = {}
    nbytes: dict[str, int] = {}
    for r in items:
        for k, (p, b) in enumerate(zip(r["source_images"], r["source_image_sizes"])):
            first.setdefault(p, (r["_shard"], r["_row"], k + 1))
            nbytes[p] = b
    todo = defaultdict(list)
    for p, (shard, row, slot) in first.items():
        if p not in done:
            todo[shard].append((row, slot, p))
    n_todo = sum(len(v) for v in todo.values())
    print(f"images: {len(first)} referenced, {len(done)} already rendered, {n_todo} to render "
          f"from {len(todo)} shards", flush=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    # memory (32 GB job): shards holding a >400 MB image go last, two at a time
    big = {s for s, w in todo.items() if any(nbytes[p] > BIG for _, _, p in w)}
    t0, n = time.time(), 0
    for phase, n_workers in (("normal", workers), ("big", min(workers, 2))):
        tasks = sorted((s, w) for s, w in todo.items() if (s in big) == (phase == "big"))
        if not tasks:
            continue
        print(f"phase {phase}: {len(tasks)} shards, {n_workers} workers", flush=True)
        with mp.get_context("spawn").Pool(n_workers, maxtasksperchild=4) as pool, INDEX.open("a") as idx:
            for res in pool.imap_unordered(_render_shard, tasks):
                for d in res:
                    n += 1
                    done[d["path"]] = d
                    idx.write(json.dumps(d) + "\n")
                    print(f"[{n}/{n_todo}] {d['path']} {d['orig_px']} {d['mode']} {d['sec']}s "
                          f"(elapsed {time.time() - t0:.0f}s)", flush=True)
                idx.flush()
    missing = set(first) - set(done)
    if missing:
        raise RuntimeError(f"{len(missing)} images not rendered: {sorted(missing)[:5]}")
    return done


# ---- rows --------------------------------------------------------------------------------------------------------

# original category -> (split, task name, paper code); the split follows the paper's Table 2
CATS = {
    "color": ("perception", "color", "COL"),
    "shape": ("perception", "shape", "SHP"),
    "detection": ("perception", "orientation", "DET"),
    "object_clasfication": ("perception", "object_classification", "OC"),
    "relation": ("perception", "relation", "REL"),
    "object_grounding": ("perception", "object_grounding", "OGD"),
    "regional_grouding": ("perception", "regional_grounding", "RG"),
    "object_counting": ("perception", "object_counting", "OCN"),
    "regionla_counting": ("perception", "regional_counting", "RCN"),
    "anomaly_detection": ("reasoning", "anomaly_detection", "AD"),
    "future_prediction_two_image": ("reasoning", "future_prediction", "FP"),
    "multi_region_join_contrast": ("reasoning", "multi_region_joint_contrast", "MRJC"),
    "multi_region_join_contrast_singal": ("reasoning", "multi_region_joint_contrast_single", "MRJCS"),
    "object_state_judgement": ("reasoning", "object_state_judgement", "OSJ"),
    "multturn_anomaly_detection": ("reasoning", "multiturn_anomaly_detection", "MAD"),
    "multitrun-future_prediction": ("reasoning", "multiturn_future_prediction", "MTFP"),
    "multiturn_object_state_judgement": ("reasoning", "multiturn_object_state_judgement", "MOSJ"),
}
BOX = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")
NUM = re.compile(r"\d+")
RES = re.compile(r"(?i)(resolution\s*:?\s*)(\d+)(\s*x\s*)(\d+)")
OWNER = re.compile(r"Image ([A-C])'s box:\s*$")  # MRJC: "Image B's box:[...]" refers to the 2nd image
LAYOUT = ("Each source image is given as 5 images: the whole image, then its top-left, top-right, "
          "bottom-left and bottom-right quarters in more detail.")  # xlrs.py's v1 wording
NAMES = ["Image A", "Image B", "Image C"]


class Frames:
    """Native -> overview coordinate maps for the images of one item."""

    def __init__(self, infos: list[dict]):
        self.native = [tuple(i["orig_px"]) for i in infos]
        self.over = [tuple(tile_sizes(*n)["overview_px"]) for n in self.native]
        self.boxes: list[tuple[int, list[int], list[int]]] = []  # (image, native box, rescaled box)

    def box(self, m: re.Match, k: int) -> str:
        (w, h), (ow, oh) = self.native[k], self.over[k]
        vals = [int(v) for v in NUM.findall(m.group(0))]
        new = [min(ow, round(v * ow / w)) if i % 2 == 0 else min(oh, round(v * oh / h)) for i, v in enumerate(vals)]
        self.boxes.append((k, vals, new))
        it = iter(new)
        return NUM.sub(lambda _: str(next(it)), m.group(0))  # keep the published spacing

    def size_text(self, sep: str = " x ") -> str:
        """The overview size; for several images of different sizes, each one named (Image A, B, C)."""
        if len(set(self.over)) == 1:
            return f"{self.over[0][0]}{sep}{self.over[0][1]}"
        return ", ".join(f"{w}{sep}{h} ({name})" for (w, h), name in zip(self.over, NAMES))

    def rescale(self, text: str) -> str:
        """Every [x1, y1, x2, y2] to overview pixels of the image it refers to; every stated resolution to the
        overview size (MRJC states one resolution for its three images; if they differ, each size is given)."""
        def sub_box(m: re.Match) -> str:
            owner = OWNER.search(text[max(0, m.start() - 20): m.start()])
            return self.box(m, "ABC".index(owner.group(1)) if owner else 0)

        def sub_res(m: re.Match) -> str:
            return f"{m.group(1)}{self.size_text(m.group(3))}"

        return RES.sub(sub_res, BOX.sub(sub_box, text))


def build_rows(items: list[dict], infos: dict[str, dict]) -> None:
    from bench.specialists.prompting import render  # the text every model sees

    dialogues = defaultdict(list)
    for r in items:
        dialogues[r["id"]].append(r)
    for d in dialogues.values():
        d.sort(key=lambda r: -1 if r["turn_id"] is None else r["turn_id"])
    rows = {"v0": [], "v1": []}
    stats = Counter()
    for r in items:
        split, task, code = CATS[r["category"]]
        letters = [l for l in "ABCD" if r[f"option_{l.lower()}"] is not None]
        published = {l: r[f"option_{l.lower()}"] for l in letters}
        assert published == json.loads(r["options"]) and r["answer"] in letters, r["id"]
        srcs = r["source_images"]
        frames = Frames([infos[p] for p in srcs])
        question = frames.rescale(r["question"]).strip()
        texts = [frames.rescale(published[l]).strip() for l in letters]
        coords = ("options" if any(BOX.search(published[l]) for l in letters)
                  else "question" if BOX.search(r["question"]) else None)
        dup = len(set(texts)) != len(texts)
        if dup:  # item 779: four identical options; keep them distinct as xlrs.py does
            texts = [f"{t} ({l})" for l, t in zip(letters, texts)]
            stats["duplicate_options"] += 1
        gold = texts[letters.index(r["answer"])]
        multi_turn = r["turn_id"] is not None
        n = len(srcs)
        # earlier turns of the dialogue, with their gold answers (teacher forcing)
        history = ""
        if multi_turn:
            earlier = [t for t in dialogues[r["id"]] if t["turn_id"] < r["turn_id"]]
            if earlier:
                lines = [f"Q: {t['question'].strip()}\nA: {t['option_' + t['answer'].lower()].strip()}" for t in earlier]
                history = (f"This is turn {r['turn_id'] + 1} of a conversation about the image. Earlier turns, with "
                           "their correct answers:\n" + "\n".join(lines))
        ow, oh = frames.over[0]
        (w, h) = frames.native[0]
        names_v0 = {1: "", 2: "Image 1 is Image A and Image 2 is Image B.",
                    3: "Images 1, 2 and 3 are Image A, Image B and Image C."}[n]
        names_v1 = {1: "", 2: "Images 1-5 show Image A and images 6-10 Image B.",
                    3: "Images 1-5 show Image A, images 6-10 Image B and images 11-15 Image C."}[n]
        if coords is None:
            coord_v0 = coord_v1 = ""
        elif n == 2:
            raise ValueError(f"{r['id']}: coordinates in a two-image item")
        elif n == 1:
            coord_v0 = f"Box coordinates [x1, y1, x2, y2] are pixel positions in the image as shown ({ow} x {oh} px)."
            coord_v1 = (f"Box coordinates [x1, y1, x2, y2] are pixel positions in Image 1, the whole image "
                        f"({ow} x {oh} px); its quarters split at x = {round(w // 2 * ow / w)} and "
                        f"y = {round(h // 2 * oh / h)}.")
        else:
            coord_v0 = (f"Box coordinates [x1, y1, x2, y2] are pixel positions in each image as shown "
                        f"({frames.size_text()} px).")
            coord_v1 = (f"Box coordinates [x1, y1, x2, y2] are pixel positions in the whole image of each source "
                        f"(Images 1, 6 and 11, {frames.size_text()} px).")
        files = [tile_files(OUT, image_key(p)) for p in srcs]
        rid = f"RSHR:{r['id']}" + (f":t{r['turn_id']}" if multi_turn else "")
        base = {
            "group_id": f"RSHR:{image_key(srcs[0])}",
            "benchmark": "RSHR-Bench",
            "split": "test",
            "task_family": f"{split}_{task}",
            "skills": [split, code] + (["multi_turn"] if multi_turn else []) + (["multi_image"] if n > 1 else []),
            "questions": {"q": {"type": "choice", "instructions": question, "criteria": {t: None for t in texts}}},
            "answer": {"q": gold},
            "meta": {"rshr_id": r["id"], "turn_id": r["turn_id"], "category": r["category"], "paper_code": code,
                     "rshr_split": split, "multi_turn": multi_turn,
                     "dialogue_id": f"RSHR:{r['id']}" if multi_turn else None,
                     "n_turns": len(dialogues[r["id"]]) if multi_turn else 1,
                     "sources": srcs, "orig_px": [list(x) for x in frames.native],
                     "overview_px": [list(x) for x in frames.over],
                     "resize_factor": [tile_sizes(*x)["resize_factor"] for x in frames.native],
                     "quarter_resize_factor": [tile_sizes(*x)["quarter_resize_factor"] for x in frames.native],
                     "coords": coords,
                     "boxes_native": [b[1] for b in frames.boxes], "boxes_shown": [b[2] for b in frames.boxes],
                     "box_image": [b[0] for b in frames.boxes], "gold_letter": r["answer"], "n_options": len(letters),
                     "duplicate_options": dup, "published_question": r["question"], "published_options": published},
        }
        state_v0 = "\n".join(s for s in (names_v0, coord_v0, history) if s)
        state_v1 = "\n".join(s for s in (" ".join(s for s in (LAYOUT, names_v1) if s), coord_v1, history) if s)
        v0 = {**base, "id": rid + ":v0", "images": [rel(f[0]) for f in files], "state": state_v0,
              "meta": {**base["meta"], "variant": "downscale"}}
        v1 = {**base, "id": rid + ":v1", "images": [rel(x) for f in files for x in f], "state": state_v1,
              "meta": {**base["meta"], "variant": "overview+quadrants"}}
        for v in (v0, v1):
            validate_row(v)
        rows["v0"].append(v0)
        rows["v1"].append(v1)
        stats[f"coords_{coords}"] += 1
        # box sanity: inside the native image, positive area before and after rescaling
        for k, nb, sb in frames.boxes:
            W, H = frames.native[k]
            if not (0 <= nb[0] < nb[2] <= W and 0 <= nb[1] < nb[3] <= H):
                stats["box_outside_native_image"] += 1
                print("  box outside the image:", rid, nb, (W, H))
            if not (sb[0] < sb[2] and sb[1] < sb[3]):
                stats["box_collapsed_after_rescale"] += 1
        for m in RES.finditer(r["question"]):
            if (int(m.group(2)), int(m.group(4))) not in frames.native:
                stats["stated_resolution_differs"] += 1
                print("  stated resolution differs:", rid, m.group(0), frames.native)
    write_jsonl(OUT / "rshr_v0_downscale.jsonl", rows["v0"])
    write_jsonl(OUT / "rshr_v1_tiles.jsonl", rows["v1"])

    r0 = rows["v0"]
    print("RSHR-Bench:", len(r0), "rows (1 choice question each) per variant; dropped 0")
    print("  split:", dict(Counter(r["meta"]["rshr_split"] for r in r0)))
    print("  task_family:", dict(Counter(r["task_family"] for r in r0)))
    print("  options:", dict(Counter(r["meta"]["n_options"] for r in r0)), "| images per row:",
          dict(Counter(len(r["images"]) for r in r0)), "(v1:", dict(Counter(len(r["images"]) for r in rows["v1"])), ")")
    print("  gold letter:", dict(sorted(Counter(r["meta"]["gold_letter"] for r in r0).items())))
    print("  multi-turn rows:", sum(r["meta"]["multi_turn"] for r in r0), "in",
          len({r["meta"]["dialogue_id"] for r in r0 if r["meta"]["multi_turn"]}), "dialogues")
    print("  checks:", dict(stats))
    px = sorted(i["orig_px"][0] * i["orig_px"][1] for i in infos.values())
    print(f"  images: {len(infos)} sources -> {5 * len(infos)} PNGs; {sum(p > MAX_PIXELS for p in px)} downscaled; "
          f"source px min {px[0]:,} median {px[len(px) // 2]:,} max {px[-1]:,}; modes",
          dict(Counter(i["mode"] for i in infos.values())))
    for rid in ("RSHR:870:v0", "RSHR:400:v1", "RSHR:1620:t2:v0"):  # grounding, 3-image MRJC, multi-turn
        row = next(r for r in rows[rid[-2:]] if r["id"] == rid)
        item = render(row, "q")
        print(f"\n--- {rid} ({row['task_family']}), gold {row['meta']['gold_letter']}; images: {row['images']}")
        print(item.text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--images-only", action="store_true")
    args = ap.parse_args()
    items = load_items()
    assert len(items) == 1932, len(items)
    paths = {p for r in items for p in r["source_images"]}
    assert len({image_key(p) for p in paths}) == len(paths), "image keys collide"
    infos = render_images(items, args.workers)
    if args.images_only:
        return
    build_rows(items, infos)


if __name__ == "__main__":
    main()
