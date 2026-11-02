"""TerraScope and EarthMind (Sa2VA code shipped with the checkpoints; eo-sa2va env, transformers 4.42.3).

Prompt layout and prediction function per model, as in the authors' evaluation scripts:
- TerraScope: R1-style instruction + "<image>\\n" + question and options; predict_forward_with_grounding (one image)
  or predict_forward_with_grounding_multi (pre/post pairs) - projects/terrascope/evaluation/infer_*.py.
- EarthMind: "<image>" + question and options + "only output the correct option."; predict_forward (one image or
  same-size frames) or, for EarthMind4B_multi, predict_forward_multi(image=SAR, rgb_image=optical) -
  projects/llava_sam2/evaluation/infer_earthmindbench_mcq.py.
Both answer modes go through bench.specialists.capture.CaptureAdapter.
"""

from __future__ import annotations

import contextlib
import io

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.specialists.capture import CaptureAdapter
from bench.specialists.prompting import LETTER_SUFFIX, Item

TERRASCOPE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The Assistant first thinks about the reasoning "
    "process in their mind and then provides the user a concise final answer in a "
    "short word or phrase. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>\n\n"
)


class Sa2VA(CaptureAdapter):
    def __init__(self, path: str, *, kind: str, answer_prefix: str = "", answer_end: str | None = None,
                 max_new_tokens: int = 2048, **_):
        assert kind in ("terrascope", "earthmind", "earthmind_multi")
        self.kind = kind
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model.preparing_for_generation(tokenizer=self.tok, max_new_tokens=max_new_tokens)
        self.lm = self.model.language_model
        self.answer_prefix = answer_prefix
        tpl = self.model.template
        self.answer_end = answer_end if answer_end is not None else (
            (tpl.get("STOP_WORDS") or [None])[0] or tpl.get("SUFFIX", "").strip() or self.tok.eos_token)

    def _text(self, item: Item) -> str:
        if self.kind == "terrascope":  # the authors' layout has no answer-format line after the options
            return TERRASCOPE_PROMPT + "<image>\n" + item.text.replace("\n" + LETTER_SUFFIX, "\n")
        return "<image>" + item.text.replace(LETTER_SUFFIX, "only output the correct option.")

    def _predict(self, item: Item) -> str:
        m, text, ims = self.model, self._text(item), item.images
        with contextlib.redirect_stdout(io.StringIO()):  # the remote code prints per-step logs
            if self.kind == "terrascope":
                if len(ims) == 1:
                    out = m.predict_forward_with_grounding(image=ims[0], text=text, tokenizer=self.tok,
                                                           max_tokens_per_seg=8)
                else:
                    out = m.predict_forward_with_grounding_multi(image_list=ims, text=text, tokenizer=self.tok,
                                                                 max_tokens_per_seg=8)
            elif self.kind == "earthmind_multi":
                assert len(ims) == 2, "EarthMind multi takes [optical, SAR]"
                out = m.predict_forward_multi(image=ims[1], rgb_image=ims[0], text=text, tokenizer=self.tok)
            elif len(ims) == 1:
                out = m.predict_forward(image=ims[0], text=text, tokenizer=self.tok)
            else:
                out = m.predict_forward(video=ims, text=text, tokenizer=self.tok)
        return out.get("prediction", "")
