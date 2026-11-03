"""C25 RS-Neg (negation understanding in RS MLLMs; arXiv 2606.20177, "ECCV'26") -> rows.
AS-PUBLISHED full sets + an ADAPTED 5k checkpoint subset.

Source: HF mumu-0011/RS-Neg @ 27053292a498fdebefd8f83aff52c23258371a8e (annotations only, Apache-2.0); evaluation
protocol from github.com/mumu011/NeFo @ 06b52eefa6cd53f71545d6e109e41f02c7cb7620 (README + evaluation/*.py, read,
not run). Images resolved through raw/rs-neg/image_index.json (manifest §7): NWPU-RESISC45 (HF timm/resisc45
parquets), VRSBench Images_val.zip, RSICD (HF arampacha/rsicd parquets), Sydney-Captions (HF isaaccorley zip, .tif),
UCMerced zip (.tif), AID zip (.jpg).

Files (all as published unless stated):
  rs_neg_vqa_negated.jsonl  9,205 `noul`: VQA_Merged/merged_vqa_tlm_yesno.jsonl, the gold file of NeFo's negative-VQA
                            evaluation (evaluate_vqa_negative.py); task_family = its `type` (object_negative 5,675,
                            attribute_negative 2,379, state_negative 1,151); answers Yes 4,666 / No 4,539.
  rs_neg_mcq_negated.jsonl  5,675 four-option `choice`: MCQ_Merged/merged_caption_negated_llava_format.jsonl (NeFo
                            README's MCQ evaluation input); options in published order and text (letter prefixes
                            stripped), task_family = mcq_<type> (object 4,502, attribute 791, state 382).
  rs_neg_ucm_negated.jsonl  2,100 `choice`: GeoChat-Bench/UCmerced_negated_formatted.jsonl, 21 negated classes
                            ("not X but Y") in each item's published order; task_family "scene_cls".
  rs_neg_aid_negated.jsonl  3,000 `choice`: GeoChat-Bench/aid_negated_formatted.jsonl, 30 negated classes.
  rs_neg_5k.jsonl           ADAPTED: a seeded (seed 0) 5,000-row subset of the VQA + MCQ files, stratified by
                            (file, type) with proportional allocation (largest remainder); rows identical to the full
                            files. For per-checkpoint tracking only; cite published numbers against the full files.
Prompts: `instructions` is the published user text with "<image>" removed and stripped, exactly as NeFo's
evaluation_with_finetuned_model.py builds it, except that the MCQ's lettered "Options:" block is not repeated in
`instructions` (the runners render the same four options as "A. ... D." in the same order); the verbatim published
prompt is in meta.published_prompt. VQA questions keep their published suffix "Please answer start with yes or no".
NeFo scores: VQA by the first yes/no in the answer; MCQ by the extracted letter; scene classification by exact
(lower-cased, trailing-period-stripped) match to "not X but Y" or to "Y" alone (score_accuracy.py). We score the
chosen option.
Not identical: RSICD (224 x 224) and the UCM images below 65,536 px are upscaled (bicubic) to MIN_PIXELS for d3,
factor in meta; Sydney (.tif) and UCM (.tif) images are re-encoded as PNG (same pixels); the runners add their own
option listing / answer suffix (for scene classification the class list then appears twice in the text).
Skipped on purpose: Ref_VRS (2,484 negated referring-expression grounding items; box output, IoU-scored, no closed
form) and FloodNet (1,037 transfer items: yes/no + complex counting on FloodNet *train* images; a separate transfer
set in the paper). The non-negated control files (merged_vqa_tlm_yesno_pos, UCmerced_formatted, aid_formatted) are
not converted.

    python -m bench.converters.rs_neg
"""

from __future__ import annotations

import glob
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from bench.common import CONVERTED, DATASETS, RAW, validate_row, write_jsonl
from bench.converters.perception_utils import sample_text, store_fitted, stratified, summary

REVISION = "mumu-0011/RS-Neg@27053292a498fdebefd8f83aff52c23258371a8e"
SNAP = DATASETS / "hf_hub" / "datasets--mumu-0011--RS-Neg" / "snapshots" / "27053292a498fdebefd8f83aff52c23258371a8e"
HUB = DATASETS / "hf_hub"
OUT = CONVERTED / "rs_neg"
MCQ_RE = re.compile(r"^(?P<pre>.*?)\n<image>\nOptions: \n(?P<opts>A\. .*)$", re.S)
OPT_RE = re.compile(r"^([A-D])\. (.*)$")
CLS_RE = re.compile(r"^<image>(?P<lead>Classify the given image in one of the following classes\. Classes: )"
                    r"(?P<classes>.*?)\.?\s*\nAnswer in one word or a short phrase\.$", re.S)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


class Images:
    """Resolve an RS-Neg image reference via raw/rs-neg/image_index.json and store it under converted/rs_neg."""

    def __init__(self) -> None:
        self.index = json.load(open(RAW / "rs-neg" / "image_index.json"))["index"]
        self.done: dict[str, dict] = {}
        self._resisc = self._rsicd = None
        self._zips: dict[str, zipfile.ZipFile] = {}
        self.stats = Counter()

    def _zip(self, path: Path) -> zipfile.ZipFile:
        if str(path) not in self._zips:
            self._zips[str(path)] = zipfile.ZipFile(path)
        return self._zips[str(path)]

    def _bytes(self, ref: str) -> bytes:
        src, *loc = self.index[ref]
        if src == "resisc45-parquet":
            if self._resisc is None:  # only the referenced images (2,921 of 31,500)
                want = {v[1] for v in self.index.values() if v[0] == src}
                self._resisc = {}
                for f in sorted(glob.glob(str(RAW / "resisc45" / "data" / "*.parquet"))):
                    for r in pq.read_table(f, columns=["image"]).to_pylist():
                        if r["image"]["path"] in want:
                            self._resisc[r["image"]["path"]] = r["image"]["bytes"]
            return self._resisc[loc[0]]
        if src == "rsicd-hf-parquet":
            if self._rsicd is None:  # only the referenced images
                want = {(v[1], v[2]) for v in self.index.values() if v[0] == src}
                self._rsicd = {}
                for f in sorted(glob.glob(str(HUB / "datasets--arampacha--rsicd" / "snapshots" / "*" / "data" / "*.parquet"))):
                    for r in pq.read_table(f, columns=["filename", "image"]).to_pylist():
                        if (Path(f).name, r["filename"]) in want:
                            self._rsicd[(Path(f).name, r["filename"])] = r["image"]["bytes"]
            return self._rsicd[(loc[0], loc[1])]
        if src == "vrsbench-Images_val.zip":
            return self._zip(RAW / "vrsbench" / "Images_val.zip").read(loc[0])
        if src == "sydney-zip":
            z = glob.glob(str(HUB / "datasets--isaaccorley--Sydney-Captions" / "snapshots" / "*" / "Sydney_captions.zip"))[0]
            return self._zip(Path(z)).read(loc[0])
        if src == "ucmerced-zip":
            return self._zip(RAW / "ucmerced" / "UCMerced_LandUse.zip").read(loc[0])
        if src == "aid-zip":
            return self._zip(RAW / "aid" / "AID.zip").read(loc[0])
        raise ValueError(f"{ref}: unsupported source {src}")

    def get(self, ref: str, prefix: str = "") -> dict:
        key = prefix + ref
        if key not in self.done:
            src = self.index[ref][0]
            stem = OUT / "images" / (prefix + str(Path(ref).with_suffix("")))
            info = store_fitted(self._bytes(ref), stem)
            info["image_source"] = src
            self.done[key] = info
            self.stats["images"] += 1
            self.stats[f"from {src}"] += 1
            if info["resize_factor"] > 1:
                self.stats["upscaled"] += 1
            elif info["resize_factor"] < 1:
                self.stats["downscaled"] += 1
            if info["reencoded"]:
                self.stats["re-encoded PNG"] += 1
        return self.done[key]


def image_meta(info: dict) -> dict:
    return {"orig_px": info["orig_px"], "px": info["px"], "resize_factor": info["resize_factor"],
            "image_source": info["image_source"]}


def vqa_rows(images: Images) -> list[dict]:
    rows = []
    for x in jsonl(SNAP / "VQA_Merged" / "merged_vqa_tlm_yesno.jsonl"):
        prompt = x["messages"][0]["content"]
        text = prompt.replace("<image>", "").strip()
        gt = x["ground_truth"].strip()
        if gt not in ("Yes", "No") or x["messages"][1]["content"].strip() != gt or len(x["images"]) != 1:
            raise ValueError(f"VQA {x['question_id']}: unexpected record")
        info = images.get(x["images"][0])
        rows.append({
            "id": f"RS-Neg-VQA:{x['question_id']}",
            "group_id": f"RS-Neg:{x['images'][0]}",
            "benchmark": "RS-Neg-VQA",
            "split": "test",
            "task_family": x["type"],
            "skills": ["negation", "hallucination"],
            "images": [info["path"]],
            "state": "",
            "questions": {"q": {"type": "noul", "instructions": text}},
            "answer": {"q": gt == "Yes"},
            "meta": {"source": REVISION, "source_file": "VQA_Merged/merged_vqa_tlm_yesno.jsonl",
                     "question_id": x["question_id"], "image": x["images"][0], "qa_type": x["type"],
                     "published_prompt": text, "published_answer": gt, "as_published": True, **image_meta(info)},
        })
    return rows


def mcq_rows(images: Images) -> list[dict]:
    mcq_type = {it["filename"]: it.get("mcq_type") for it in json.load(open(SNAP / "MCQ_Merged" / "merged_mcq.json"))
                if isinstance(it, dict) and it.get("filename")}
    rows, type_mismatch = [], 0
    for x in jsonl(SNAP / "MCQ_Merged" / "merged_caption_negated_llava_format.jsonl"):
        prompt = x["messages"][0]["content"]
        m = MCQ_RE.match(prompt)
        if not m:
            raise ValueError(f"MCQ {x['question_id']}: unparsed prompt {prompt[:200]!r}")
        parsed = [OPT_RE.match(line) for line in m.group("opts").split("\n")]
        if len(parsed) != 4 or not all(parsed) or [p.group(1) for p in parsed] != list("ABCD"):
            raise ValueError(f"MCQ {x['question_id']}: options {m.group('opts')!r}")
        options = [p.group(2).strip() for p in parsed]
        duplicated = len(set(options)) != 4
        if duplicated:  # 7 published items repeat the gold text in a distractor: tag repeats with their letter
            options = [f"{o} ({l})" if options.count(o) > 1 else o for l, o in zip("ABCD", options)]
        gold = options["ABCD".index(x["ground_truth"].strip())]
        if mcq_type.get(x["question_id"]) not in (None, x["type"]):
            type_mismatch += 1
        ref = x["images"] if isinstance(x["images"], str) else x["images"][0]
        info = images.get(ref)
        rows.append({
            "id": f"RS-Neg-MCQ:{x['question_id']}",
            "group_id": f"RS-Neg:{ref}",
            "benchmark": "RS-Neg-MCQ",
            "split": "test",
            "task_family": f"mcq_{x['type']}",
            "skills": ["negation", "caption_matching"],
            "images": [info["path"]],
            "state": "",
            "questions": {"q": {"type": "choice", "instructions": m.group("pre").strip(),
                                "criteria": {o: None for o in options}}},
            "answer": {"q": gold},
            "meta": {"source": REVISION, "source_file": "MCQ_Merged/merged_caption_negated_llava_format.jsonl",
                     "question_id": x["question_id"], "image": ref, "mcq_type": x["type"],
                     "published_prompt": prompt.replace("<image>", "").strip(),
                     "published_answer_letter": x["ground_truth"].strip(), "as_published": True, **image_meta(info),
                     **({"duplicate_options": "published options repeat the gold text; repeats tagged (A)..(D)"}
                        if duplicated else {})},
        })
    if type_mismatch:
        print(f"  note: {type_mismatch} MCQ items whose `type` differs from merged_mcq.json mcq_type")
    return rows


def scene_rows(images: Images, fname: str, bench: str, prefix: str) -> list[dict]:
    rows = []
    for x in jsonl(SNAP / "GeoChat-Bench" / fname):
        prompt = x["messages"][0]["content"]
        m = CLS_RE.match(prompt)
        if not m:
            raise ValueError(f"{fname} {x['question_id']}: unparsed prompt {prompt[:200]!r}")
        classes = [c.strip() for c in m.group("classes").split(", ")]
        gt = x["ground_truth"].strip()
        if gt not in classes or len(set(classes)) != len(classes):
            raise ValueError(f"{fname} {x['question_id']}: ground truth {gt!r} not among {len(classes)} classes")
        info = images.get(x["images"][0], prefix)
        rows.append({
            "id": f"{bench}:{x['question_id']}",
            "group_id": f"{bench}:{x['question_id']}",
            "benchmark": bench,
            "split": "test",
            "task_family": "scene_cls",
            "skills": ["negation", "lulc"],
            "images": [info["path"]],
            "state": "",
            "questions": {"scene": {"type": "choice", "instructions": prompt.replace("<image>", "").strip(),
                                    "criteria": {c: None for c in classes}}},
            "answer": {"scene": gt},
            "meta": {"source": REVISION, "source_file": f"GeoChat-Bench/{fname}", "question_id": x["question_id"],
                     "image": x["images"][0], "published_prompt": prompt.replace("<image>", "").strip(),
                     "n_classes": len(classes), "published_scoring": "exact match to 'not X but Y' or to 'Y'",
                     "as_published": True, **image_meta(info)},
        })
    return rows


def main() -> None:
    images = Images()
    outputs = {
        "rs_neg_vqa_negated.jsonl": vqa_rows(images),
        "rs_neg_mcq_negated.jsonl": mcq_rows(images),
        "rs_neg_ucm_negated.jsonl": scene_rows(images, "UCmerced_negated_formatted.jsonl", "RS-Neg-UCM-neg", "ucm/"),
        "rs_neg_aid_negated.jsonl": scene_rows(images, "aid_negated_formatted.jsonl", "RS-Neg-AID-neg", "aid/"),
    }
    pool = outputs["rs_neg_vqa_negated.jsonl"] + outputs["rs_neg_mcq_negated.jsonl"]
    for r in pool:
        r["_stratum"] = f"{r['benchmark']}:{r['task_family']}"
    subset = stratified(pool, "_stratum", 5000, random.Random(0))
    for r in pool:
        r.pop("_stratum")
    order = {r["id"]: i for i, r in enumerate(pool)}
    subset = sorted(({**r, "meta": {**r["meta"], "as_published": False, "subset": "5k stratified (seed 0)"}}
                     for r in subset), key=lambda r: order[r["id"]])
    outputs["rs_neg_5k.jsonl"] = subset
    for fname, rows in outputs.items():
        for r in rows:
            validate_row(r)
        write_jsonl(OUT / fname, rows)
        print(summary(fname, rows))
    print("images:", dict(images.stats), "| dropped: 0")
    for fname in ("rs_neg_vqa_negated.jsonl", "rs_neg_mcq_negated.jsonl", "rs_neg_aid_negated.jsonl"):
        print(f"== samples from {fname}")
        print(sample_text(outputs[fname], 1 if fname != "rs_neg_mcq_negated.jsonl" else 2, seed=3))


if __name__ == "__main__":
    main()
