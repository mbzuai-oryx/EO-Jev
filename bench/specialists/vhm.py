"""VHM-7B (eo-vhm env: the authors' LLaVA-style code, transformers 4.31).

As in RSEvalKit's VHM wrapper (third_party/RSEvalKit/vhm_model.py): conversation template "v1", "<image>\\n" before
the question, CLIP-336 preprocessing with padding to square, fp16, greedy decoding with up to 512 new tokens, and
"Answer with the option's letter from the given choices directly." for multiple choice.
logprob: one forward pass over the templated prompt, next-token probabilities over the letters / yes-no.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from bench import common
from bench.specialists.prompting import LETTER_SUFFIX, Item, first_token_ids

THIRD_PARTY = str(common.THIRD_PARTY / "VHM")


class VHM:
    def __init__(self, path: str, *, max_new_tokens: int = 512, **_):
        sys.path.insert(0, THIRD_PARTY)
        from vhm.model.builder import load_pretrained_model  # noqa: PLC0415 - the authors' loader

        # model_name must contain "vhm" for the loader to build the multimodal class
        self.tok, model, self.image_processor, _ = load_pretrained_model(
            model_path=path, model_base=None, model_name="vhm_7B", device="cpu", device_map="cpu")
        self.model = model.cuda().eval()
        self.max_new_tokens = max_new_tokens
        self._ids: dict[str, list[int]] = {}

    def _inputs(self, item: Item):
        from vhm.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: PLC0415
        from vhm.conversation import conv_templates  # noqa: PLC0415
        from vhm.mm_utils import process_images, tokenizer_image_token  # noqa: PLC0415

        assert len(item.images) == 1, "VHM takes one image"
        images = process_images(item.images, self.image_processor, SimpleNamespace(image_aspect_ratio="pad"))
        text = item.text.replace(LETTER_SUFFIX, "Answer with the option's letter from the given choices directly.")
        img = DEFAULT_IMAGE_TOKEN
        if self.model.config.mm_use_im_start_end:
            img = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], img + "\n" + text)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return ids.unsqueeze(0).cuda(), images.to("cuda", dtype=torch.float16), conv

    @torch.inference_mode()
    def score(self, items: list[Item]) -> list[list[float]]:
        out = []
        for it in items:
            assert it.mode != "names", "class-name scoring is not used for VHM"
            ids, images, _ = self._inputs(it)
            p = torch.softmax(self.model(input_ids=ids, images=images).logits[0, -1].float(), dim=-1)
            row = []
            for lab in it.labels:
                if lab not in self._ids:
                    self._ids[lab] = first_token_ids(self.tok, lab)
                row.append(float(p[self._ids[lab]].sum()))
            out.append(row)
        return out

    @torch.inference_mode()
    def generate(self, items: list[Item]) -> list[str]:
        from vhm.conversation import SeparatorStyle  # noqa: PLC0415
        from vhm.mm_utils import KeywordsStoppingCriteria  # noqa: PLC0415

        out = []
        for it in items:
            ids, images, conv = self._inputs(it)
            stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            gen = self.model.generate(ids, images=images, do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
                                      stopping_criteria=[KeywordsStoppingCriteria([stop], self.tok, ids)])
            out.append(self.tok.decode(gen[0, ids.shape[1]:]).strip().split("</s>")[0])
        return out
