"""EarthDial (InternVL2-4B / Phi-3; loaded with the authors' package, third_party/EarthDial/src, as the checkpoint's
own inference.py does).

As in the authors' RSVQA evaluation defaults (src/earthdial/eval/rs_vqa/vqa_test.py: no --dynamic, --num-beams 5):
one 448 x 448 image per input (InternVL tiling off), ImageNet normalisation, model.chat() with 5-beam decoding.
Several images (not needed for RSVQA-LR) are passed as separate image groups.
Both answer modes go through bench.specialists.capture.CaptureAdapter.
"""

from __future__ import annotations

import contextlib
import io
import sys

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

from bench import common
from bench.specialists.capture import CaptureAdapter
from bench.specialists.prompting import Item

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
THIRD_PARTY = str(common.THIRD_PARTY / "EarthDial" / "src")


class EarthDial(CaptureAdapter):
    def __init__(self, path: str, *, num_beams: int = 5, max_new_tokens: int = 64, **_):
        sys.path.insert(0, THIRD_PARTY)
        from earthdial.model.internvl_chat import InternVLChatModel  # noqa: PLC0415 - the authors' model code

        self.model = InternVLChatModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                       low_cpu_mem_usage=True).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        cfg = self.model.config
        self.size = cfg.force_image_size or cfg.vision_config.image_size
        self.transform = T.Compose([T.Resize((self.size, self.size), interpolation=InterpolationMode.BICUBIC),
                                    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.lm = self.model.language_model
        self.num_beams, self.max_new_tokens = num_beams, max_new_tokens
        self.answer_prefix = ""
        # end of the assistant turn in the Phi-3 conversation template
        self.answer_end = "<|end|>" if "<|end|>" in self.tok.get_vocab() else self.tok.eos_token
        self.answer_context = "<|assistant|>\n"  # the answer is written right after the assistant turn opener

    def _predict(self, item: Item) -> str:
        pixel_values = torch.stack([self.transform(im) for im in item.images]).to(torch.bfloat16).cuda()
        # a native question carries its own task tokens and <image> placeholders (bench/specialists/native.py)
        question = item.text if item.native else "\n".join("<image>" for _ in item.images) + "\n" + item.text
        gen = dict(max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=self.num_beams, min_new_tokens=1)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.model.chat(self.tok, pixel_values, question, gen, num_patches_list=[1] * len(item.images))
