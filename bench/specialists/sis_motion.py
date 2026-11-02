"""SIS-Motion (Zou et al., "Self in Space", arXiv 2607.12477; eo-sismotion env: the authors' motion code,
third_party/Self-in-Space/motion, transformers 4.57.1).

Qwen2.5-VL-7B-Instruct + LoRA (r=32 on q/k/v/o_proj, merged) + a motion branch: a frozen VideoFlow MOFNet
(MOF_kitti.pth, run in FP32 on frames resized to 320 x 480) predicts optical flow; the forward flow becomes
[magnitude, dx, dy] pseudo-images that go through the same frozen Qwen vision encoder, and the trained `visual_flow`
connector (LayerNorm -> Linear -> GELU -> Linear) adds the result to the video tokens before the language model.
All of this is the authors' MotionMLLMForConditionalGeneration (motion/src/uav/model/motion_mllm.py).

As in the authors' evaluation (motion/run_eval.sh -> motion/src/eval/eval.py load_model / run_inference, with frames
prepared like motion/src/eval/dataset_utils.py SISBenchDataset):
- loading, in their order: MotionMLLMConfig from the base config + sis_motion_config.json's motion_config and
  connector_type; MotionMLLMForConditionalGeneration.from_pretrained(base, bf16); accelerate hooks removed from the
  motion modules; MOF_kitti.pth loaded into the MOFNet, which then runs in FP32; connector_weights.pt in bf16; the LoRA
  adapter merged (PeftModel.from_pretrained + merge_and_unload). The MOFNet weights are copied into the bf16 model
  before the FP32 cast, so they are bf16-rounded, exactly as in their training and evaluation.
- video processor: 128*28*28 .. 512*28*28 pixels per frame, fps 2, 8 .. 32 frames (eval.py overrides).
- the frames go in as one video (a list of PIL frames) at the start of the user turn, then the text; the stock
  Qwen2.5-VL chat template (its default system prompt) with the generation prompt.
- video_tchw: the same frames as float [0, 255] tensors, bilinearly resized to 320 x 480, re-indexed with
  np.linspace to the processor's frame count (grid_t * temporal_patch_size) when it differs.
- forward and generate under torch.autocast(bf16) and no_grad; greedy decoding (do_sample=False with the base
  generation_config, as theirs), up to 128 new tokens, decoded without special tokens.
Differences from theirs: sdpa attention instead of flash_attention_2 (as our other adapters; attn_implementation
overrides); the text is our common question format (bench/specialists/prompting.py: the frame-time state, question,
lettered options and LETTER_SUFFIX) instead of their "{question}\\n{A. ..}\\nPlease select the correct answer.".
SIS-Motion was trained to answer an MCQ with the letter alone.

logprob: the authors' forward (MotionMLLMForConditionalGeneration.forward with video_tchw, i.e. MOFNet -> flow
pseudo-images -> ViT -> connector -> fused video tokens -> LoRA-merged LM) on the full prompt; the option
probabilities are read from the logits at the last position, the next token after the assistant turn opener (plus
answer_prefix, empty by default), summed over each label's first-token ids. bench/specialists/capture.py's input
capture does not fit this model: Qwen2.5-VL's language model needs the 3-D M-RoPE positions that its forward computes
from input_ids, video_grid_thw and second_per_grid_ts, which a bare inputs_embeds call would replace by 1-D positions.
Class-name scoring (mode "names") is not implemented; SIS-Bench is lettered.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from bench import common
from bench.specialists.prompting import Item, first_token_ids

THIRD_PARTY = str(common.THIRD_PARTY / "Self-in-Space" / "motion")
MOTION_HW = (320, 480)              # dataset_utils.py: frames pre-resized for the motion encoder
VIDEO_MIN_PX, VIDEO_MAX_PX = 128 * 28 * 28, 512 * 28 * 28  # eval.py load_model video processor overrides
MIN_MOTION_FRAMES = 3               # MOFNet needs a previous and a next frame around each middle frame
BASE_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"


def _import_authors():
    """The authors' model classes. motion_encoder.py imports deepspeed at module level (for its ZeRO-3 loading
    path only); if deepspeed is absent or broken, a stub stands in, which nothing in inference calls."""
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)
    try:
        import deepspeed  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001 - ImportError, or deepspeed failing on import (e.g. accelerator detection)
        import types  # noqa: PLC0415
        sys.modules["deepspeed"] = types.ModuleType("deepspeed")
    from src.uav.model.motion_mllm import MotionMLLMConfig, MotionMLLMForConditionalGeneration  # noqa: PLC0415
    return MotionMLLMConfig, MotionMLLMForConditionalGeneration


def check_mofnet_weights(mofnet: torch.nn.Module, ckpt_path: str) -> int:
    """The authors load MOF_kitti.pth with strict=False and only print mismatches; fail instead, since a timm version
    that renames the twins encoder's modules would silently leave it random. Returns the number of tensors."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt = {k[len("module."):] for k in sd if k.startswith("module.")}
    model = set(mofnet.state_dict())
    missing, unexpected = sorted(model - ckpt), sorted(ckpt - model)
    if missing or unexpected:
        raise RuntimeError(f"MOF_kitti.pth does not match MOFNet: missing {missing[:8]} unexpected {unexpected[:8]}")
    return len(ckpt)


class SISMotion:
    def __init__(self, path: str, *, base: str, attn_implementation: str = "sdpa", max_new_tokens: int = 128,
                 answer_prefix: str = "", device: str = "cuda:0", **_):
        from accelerate.hooks import remove_hook_from_module  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        MotionMLLMConfig, MotionMLLMForConditionalGeneration = _import_authors()
        with open(os.path.join(path, "sis_motion_config.json")) as f:
            manifest = json.load(f)
        with open(os.path.join(path, "adapter_config.json")) as f:
            adapter_base = json.load(f)["base_model_name_or_path"]
        if {adapter_base, manifest.get("base_model_name_or_path", BASE_REPO)} != {BASE_REPO}:
            raise ValueError(f"this adapter is written for a {BASE_REPO} base, the package names {adapter_base}")
        if manifest["connector_type"] != "visual_flow":
            raise ValueError(f"unexpected connector_type {manifest['connector_type']}")

        # eval.py load_model, LoRA-checkpoint branch
        config = MotionMLLMConfig.from_pretrained(base, motion_config=manifest["motion_config"],
                                                  connector_config={"connector_type": manifest["connector_type"]})
        model = MotionMLLMForConditionalGeneration.from_pretrained(
            base, config=config, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16,
            device_map={"": device})
        remove_hook_from_module(model.motion_encoder, recurse=True)
        remove_hook_from_module(model.connector, recurse=True)
        mof_path = os.path.join(path, manifest.get("videoflow_checkpoint", "MOF_kitti.pth"))
        check_mofnet_weights(model.motion_encoder.mofnet, mof_path)
        model.motion_encoder.load_pretrained_weights(mof_path)
        model.motion_encoder.to(device=device, dtype=torch.float32)
        conn = torch.load(os.path.join(path, manifest.get("connector_weights", "connector_weights.pt")),
                          map_location="cpu", weights_only=True)
        model.connector.load_state_dict(conn)  # strict
        model.connector.to(device=device, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, path)
        lora_b = [m.weight for n, m in model.named_modules() if n.endswith("lora_B.default")]
        if len(lora_b) != 4 * config.text_config.num_hidden_layers or any(not w.abs().sum() for w in lora_b):
            raise RuntimeError(f"LoRA adapter not fully loaded: {len(lora_b)} lora_B layers")
        self.model = model.merge_and_unload().eval()
        print(f"SIS-Motion: base {base}, {len(lora_b)} LoRA layers merged, MOFNet {mof_path}, "
              f"attn {attn_implementation}", flush=True)

        self.processor = AutoProcessor.from_pretrained(base)
        vp = self.processor.video_processor  # eval.py: align the video processor with the LoRA training settings
        vp.min_pixels, vp.max_pixels, vp.fps = VIDEO_MIN_PX, VIDEO_MAX_PX, 2.0
        if hasattr(vp, "min_frames"):
            vp.min_frames = 8
        if hasattr(vp, "max_frames"):
            vp.max_frames = 32
        if isinstance(getattr(vp, "size", None), dict):
            vp.size["shortest_edge"], vp.size["longest_edge"] = VIDEO_MIN_PX, VIDEO_MAX_PX
        self.tok = self.processor.tokenizer
        self.device = torch.device(device)
        self.max_new_tokens, self.answer_prefix = max_new_tokens, answer_prefix
        self._ids: dict[str, list[int]] = {}

    # ---------------------------------------------------------------- inputs
    def _video_tchw(self, frames: list, n_used: int) -> torch.Tensor:
        """dataset_utils.py _load_frames + eval.py run_inference frame alignment."""
        t = []
        for im in frames:  # per frame (= their batched interpolate on a stack of equal-size frames)
            x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float()[None]
            if tuple(x.shape[-2:]) != MOTION_HW:
                x = F.interpolate(x, size=MOTION_HW, mode="bilinear", align_corners=False)
            t.append(x[0])
        tchw = torch.stack(t)
        if tchw.shape[0] != n_used:
            tchw = tchw[np.linspace(0, tchw.shape[0] - 1, n_used, dtype=int)]
        return tchw

    def _inputs(self, item: Item, prefill: str = "") -> dict:
        if item.turns or item.native:
            raise ValueError("SIS-Motion adapter: no native format / multi-turn dialogue")
        frames = [im.convert("RGB") for im in item.images]
        if frames and len(frames) < MIN_MOTION_FRAMES:
            raise ValueError(f"SIS-Motion needs >= {MIN_MOTION_FRAMES} frames for optical flow, got {len(frames)}")
        content = ([{"type": "video", "video": frames}] if frames else []) + [{"type": "text", "text": item.text}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True) + prefill
        inputs = self.processor(text=[text], videos=[frames] if frames else None, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        if frames:
            tps = getattr(self.processor.video_processor, "temporal_patch_size", 2)
            n_used = int(inputs["video_grid_thw"][:, 0].sum().item()) * tps
            inputs["video_tchw"] = [self._video_tchw(frames, n_used)]  # forward() moves it to the MOFNet's device
        return inputs

    def _label_ids(self, label: str) -> list[int]:
        if label not in self._ids:
            self._ids[label] = first_token_ids(self.tok, label)
        return self._ids[label]

    # ---------------------------------------------------------------- logprob mode
    @torch.no_grad()
    def _score_one(self, item: Item) -> list[float]:
        if item.mode == "names":
            raise NotImplementedError("SIS-Motion adapter scores letters / yes-no only")
        inputs = self._inputs(item, self.answer_prefix)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(**inputs, use_cache=False, logits_to_keep=1)
        p = torch.softmax(out.logits[0, -1].float(), dim=-1)
        return [float(p[self._label_ids(lab)].sum()) for lab in item.labels]

    def score(self, items: list[Item]) -> list[list[float]]:
        return [self._score_one(it) for it in items]

    # ---------------------------------------------------------------- generate mode
    @torch.no_grad()
    def generate(self, items: list[Item]) -> list[str]:
        out = []
        for it in items:
            inputs = self._inputs(it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            new = ids[:, inputs["input_ids"].shape[1]:]
            out.append(self.processor.batch_decode(new, skip_special_tokens=True)[0])
        return out
