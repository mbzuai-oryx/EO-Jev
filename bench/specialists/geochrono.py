"""GeoChrono (eo-geochrono env): Qwen3-VL-4B-Instruct + merged LoRA + TempEnc ("OTTC") patched into the forward pass.

Loading follows third_party/GeoChrono/ottc-novel/test/model_ottc.py (base -> merge LoRA -> inject TempEnc -> load
ottc_weights.pt), and the prompt follows its harness (test/models/qwen3vl.py, test/system_prompt.py): an expert
system prompt, frames resized to 1024 x 1024 and each preceded by its year label ("2006:"), then the question and the
answer instruction. CDVQA pairs carry no dates, so their frames are labelled "Before:" / "After:".
TempEnc treats all frames of one forward pass as a single location's time series, so the batch size is 1.
"""

from __future__ import annotations

import json
import os
import re
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from bench import common
from bench.specialists.hf_vlm import HFVLM
from bench.specialists.prompting import LETTER_SUFFIX, YESNO_SUFFIX, Item

THIRD_PARTY = str(common.THIRD_PARTY / "GeoChrono")
INPUT_FORMAT = ("You will receive a temporal sequence of remote sensing images. Each image is preceded by its year "
                "label in the format:\nYYYY: [image]\nImages are arranged in chronological order from earliest to "
                "latest.")
LAND_COVER = ("Land cover is classified into five categories:\n"
              "1. vegetation - areas covered by plants, grass, trees, or crops\n"
              "2. buildings - man-made structures such as houses, factories, or commercial buildings\n"
              "3. non-vegetated surface - bare soil, roads, parking lots, or other exposed ground\n"
              "4. water - rivers, lakes, ponds, or other water bodies\n"
              "5. playground - sports fields, recreational areas, or athletic facilities")
SYSTEM = ("You are an expert in remote sensing image analysis specializing in temporal land cover change detection.\n\n"
          f"{INPUT_FORMAT}\n\n{LAND_COVER}")
LETTER_RULE = ('IMPORTANT: Respond with ONLY the option letter followed by a period (e.g., "A." or "B."). '
               "Do not provide explanations or reasoning.")
YESNO_RULE = "IMPORTANT: Respond with ONLY yes or no. Do not provide explanations or reasoning."


class GeoChrono(HFVLM):
    def __init__(self, path: str, *, base: str, image_size: int = 1024, max_new_tokens: int = 128, **_):
        sys.path.insert(0, os.path.join(THIRD_PARTY, "ottc-novel"))
        from ottc_wrapper import inject_ottc  # noqa: PLC0415 - the authors' TempEnc code
        from peft import PeftModel  # noqa: PLC0415

        model = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda:0",
                                                                attn_implementation="sdpa")
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
        with open(os.path.join(path, "ottc_config.json")) as f:
            inject_ottc(model, json.load(f))
        model.ottc.load_state_dict(torch.load(os.path.join(path, "ottc_weights.pt"), map_location="cpu"))
        model.ottc.to(device="cuda:0", dtype=torch.bfloat16)
        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(base)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        self.system, self.answer_prefix, self.answer_end = SYSTEM, "", "<|im_end|>"
        self.gen_kwargs = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None, "use_cache": True}
        self.max_new_tokens, self.batch_size, self.name_batch = max_new_tokens, 1, 1
        self.image_size = image_size
        self._letter_ids = {}

    def _labels(self, item: Item) -> list[str]:
        years = re.findall(r"Image \d+ = (\d{4})", item.text)
        if len(years) == len(item.images):
            return years
        if len(item.images) == 2:
            return ["Before", "After"]
        return [f"Image {i + 1}" for i in range(len(item.images))]

    def prompt(self, item: Item, prefill: str = "") -> str:
        text = item.text.replace(LETTER_SUFFIX, LETTER_RULE).replace(YESNO_SUFFIX, YESNO_RULE)
        content = []
        for i, lab in enumerate(self._labels(item)):
            content += [{"type": "text", "text": f"{'' if i == 0 else chr(10)}{lab}:"}, {"type": "image"}]
        content.append({"type": "text", "text": "\n\n" + text})
        msgs = [{"role": "system", "content": [{"type": "text", "text": self.system}]},
                {"role": "user", "content": content}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill

    def _inputs(self, prompts, image_lists):
        sized = [[im.resize((self.image_size, self.image_size), Image.LANCZOS) if im.size != (self.image_size,) * 2 else im
                  for im in ims] for ims in image_lists]
        return super()._inputs(prompts, sized)
