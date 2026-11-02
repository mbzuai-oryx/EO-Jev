"""TEOChat (Irvin et al., ICLR 2025; eo-teochat env: the authors' Video-LLaVA-based code, third_party/TEOChat,
transformers 4.31).

As in the authors' evaluation (scripts/eval_teochat.sh -> videollava/eval/eval.py load_model and
videollava/eval/inference.py run_inference_single):
- load_pretrained_model with --load_8bit (the merged checkpoint quantised by bitsandbytes, as TEOChat was trained on
  an 8-bit base); the LanguageBind video tower is dropped after loading, images go through the image tower only.
- LanguageBind image processor: shortest side to 224 (bicubic), centre crop 224, CLIP normalisation; no padding
  (their eval does not pad, although config.json says image_aspect_ratio "pad").
- conversation template "v1" (Vicuna: system prompt, "USER: ... ASSISTANT:", turns end with "</s>");
  --chronological_prefix: "times:" -> "times in chronological order:"; --prompt_strategy interleave: the <video> tag
  becomes "Image 1: <image>Image 2: <image>..." with the frames in temporal order.
- up to 256 new tokens, stop at "</s>". Their eval samples at temperature 0.2; we decode greedily unless a
  temperature is given.

Our (non-native) questions get the framing sentence TEOChatlas puts before the images: "These are two satellite
images capturing the same location at different times: <video> \\n" for a pair (its xBD eval files), "This is a
satellite image: <video>\\n" for one image and "This is a sequence of satellite images capturing the same location at
different times: <video> \\n" for more; the letter instruction uses the LLaVA-1.5 wording Video-LLaVA was tuned on
(TEOChatlas itself has no lettered questions). Native questions (bench/specialists/native.py) are TEOChatlas
questions that carry their own <video> tag and are used as they are.

Both answer modes go through bench.specialists.capture.CaptureAdapter. The authors' generate path,
model.generate(input_ids, images=...), builds the prompt embeddings inside LlavaLlamaForCausalLM.forward and hands
them to LlamaForCausalLM.forward through super(), which an instance-level patch cannot see. _predict therefore builds
the embeddings with that same function (prepare_inputs_labels_for_multimodal) and generates from them with
self.lm.generate(inputs_embeds=...), as later LLaVA releases' generate() does: same embeddings, positions and
attention mask, so the same tokens.
"""

from __future__ import annotations

import os
import sys

import torch

from bench import common
from bench.specialists.capture import CaptureAdapter
from bench.specialists.prompting import LETTER_SUFFIX, Item

THIRD_PARTY = str(common.THIRD_PARTY / "TEOChat")
LETTER_RULE = "Answer with the option's letter from the given choices directly."
LEAD = {1: "This is a satellite image: <video>\n",
        2: "These are two satellite images capturing the same location at different times: <video> \n"}
LEAD_SEQUENCE = "This is a sequence of satellite images capturing the same location at different times: <video> \n"


class TEOChat(CaptureAdapter):
    def __init__(self, path: str, *, max_new_tokens: int = 256, load_8bit: bool = True,
                 temperature: float | None = None, **_):
        sys.path.insert(0, THIRD_PARTY)
        from videollava.model import LlavaConfig  # noqa: PLC0415 - the authors' code
        from videollava.model.builder import load_pretrained_model  # noqa: PLC0415
        from videollava.utils import disable_torch_init  # noqa: PLC0415

        disable_torch_init()  # as eval.load_model
        # The LanguageBind towers are named by repo id in config.json and loaded with
        # from_pretrained(cache_dir=config.cache_dir), which defaults to ./cache_dir: point it at the HF cache.
        cfg = LlavaConfig.from_pretrained(path)
        cfg.cache_dir = os.environ.get("HF_HUB_CACHE")
        # model_name must contain "teochat" (or "llava") for the loader to build the multimodal model and its towers
        self.tok, self.model, processor, _ = load_pretrained_model(path, None, "teochat", load_8bit=load_8bit,
                                                                   device="cuda:0", config=cfg)
        self.model.model.video_tower = None  # as eval.load_model
        self.model.eval()
        self.processor = processor["image"]
        self.lm = self.model  # its forward(inputs_embeds=...) skips the image splicing and runs the LLaMA LM
        self.answer_prefix, self.answer_end = "", "</s>"
        # no answer_context: TEOChat answers with a leading space ("ASSISTANT: Major damage." -> "▁Major"), which is
        # exactly how the tokenizer encodes an answer on its own; "ASSISTANT:" as context would drop that marker
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.dtype = torch.float16  # image tensors, as run_inference_single

    def _messages(self, item: Item) -> list[tuple[str, str | None]]:
        """(user, assistant) turns; the images go with the first user turn."""
        msgs = [*getattr(item, "turns", []), (item.text, None)]
        if getattr(item, "native", False):
            return msgs
        lead = LEAD.get(len(item.images), LEAD_SEQUENCE)
        msgs[-1] = (msgs[-1][0].replace(LETTER_SUFFIX, LETTER_RULE), None)
        msgs[0] = (lead + msgs[0][0], msgs[0][1])
        return msgs

    def _predict(self, item: Item) -> str:
        from videollava.constants import IMAGE_TOKEN_INDEX  # noqa: PLC0415
        from videollava.conversation import conv_templates  # noqa: PLC0415
        from videollava.eval.inference import replace_video_token  # noqa: PLC0415
        from videollava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token  # noqa: PLC0415

        conv = conv_templates["v1"].copy()
        for user, reply in self._messages(item):
            conv.append_message(conv.roles[0], user)
            conv.append_message(conv.roles[1], reply)
        prompt = conv.get_prompt().replace("times:", "times in chronological order:")
        prompt = replace_video_token(prompt, item.images, "interleave")
        if prompt.count("<image>") != len(item.images):
            raise ValueError(f"{prompt.count('<image>')} image placeholders for {len(item.images)} images")
        dev = self.model.device
        images = [self.processor.preprocess(im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
                  .to(dev, dtype=self.dtype) for im in item.images]
        ids = tokenizer_image_token(prompt, self.tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(dev)
        embeds = self.model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images)[4]
        # transformers 4.31 starts inputs_embeds generation from one BOS id that is never fed to the model (the
        # embeddings are); passing it explicitly fixes where the new tokens start in the output
        start = torch.full((1, 1), self.tok.bos_token_id, dtype=torch.long, device=dev)
        stop = KeywordsStoppingCriteria([conv.sep2], self.tok, start)
        sampling = (dict(do_sample=True, temperature=self.temperature) if self.temperature
                    else dict(do_sample=False, num_beams=1))
        out = self.lm.generate(input_ids=start, inputs_embeds=embeds,
                               attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=dev),
                               max_new_tokens=self.max_new_tokens, use_cache=True, stopping_criteria=[stop],
                               **sampling)
        return self.tok.decode(out[0, start.shape[1]:]).replace("</s>", "").strip()
