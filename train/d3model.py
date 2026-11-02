"""d3-flash as a trainable model: the shipped runtime's forward pass plus LoRA adapters.

Everything that decides what the model computes comes from the checkpoint's own d3_runtime.py (as in
bench/runners/d3.py); nothing is re-implemented:
- backbone: Qwen3_5Model in BF16 with SDPA, the vision patch embedding run as a matrix product
  (linearize_patch_embed), and noncausal full attention: the 8 softmax-attention layers see the whole prompt,
  while the Gated DeltaNet layers stay causal;
- prompts: normalize_question, then render (text) or image_messages (images first) through the checkpoint's
  tokenizer / processor, left-padded, with images kept between 65,536 and 1,638,400 px;
- readout: the hidden state at the last prompt position, FP32 @ readout.T, over the question's own codes only,
  divided by the checkpoint temperature.

train/parity.py checks this against the runtime itself (gate R0).

Trainable: LoRA (rank r, scale alpha / r) on the text tower's linear layers:
- attention q / k / v / o;
- Gated DeltaNet in_proj_qkv / in_proj_z / out_proj;
- MLP gate / up / down.

B starts at zero, so step 0 is exactly d3-flash. Layers whose smaller side is not larger than r (in_proj_a / in_proj_b,
32 outputs) are skipped. Embeddings, norms, the patch embedding and the readout stay frozen.

Vision adapters (add_vision_lora; used by EO-Jev), same B = 0 start and the same
alpha / r scale as the text adapters, named "visual.<path>":
- the merger (connector): linear_fc1 / linear_fc2, which map each 2x2 patch block to the language model's width;
- the vision blocks: attn.qkv / attn.proj and mlp.linear_fc1 / linear_fc2 in all 27 blocks. The tower then runs in
  train mode, so its blocks use activation checkpointing too.
Without them the vision tower stays frozen.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from bench.common import D3_FLASH

sys.path.insert(0, str(D3_FLASH))
import d3_runtime as rt  # noqa: E402 - shipped with the checkpoint

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj")
VISION_PREFIX = "visual."
VISION_BLOCK_TARGETS = (("attn", ("qkv", "proj")), ("mlp", ("linear_fc1", "linear_fc2")))
MERGER_TARGETS = ("linear_fc1", "linear_fc2")


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) * B A x. A and B are FP32 master weights, applied in the input's dtype."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        dev = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype)) * self.scale

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return (self.base.weight.float() + self.scale * (self.lora_b @ self.lora_a)).to(self.base.weight.dtype)


@dataclass
class Example:
    """One question of one row, ready to run: images are RGB PIL images (none for a text request)."""

    key: str
    state: Any
    question: dict
    images: list = field(default_factory=list)
    target: list[float] | None = None
    ordinal: bool = False
    group: str = ""


def n_options(question: dict) -> int:
    return 2 if question["type"] == "noul" else len(question["criteria"])


def gold_index(question: dict, gold: Any) -> int:
    if question["type"] == "noul":
        return int(bool(gold))
    if question["type"] == "score":
        return int(gold)
    return list(question["criteria"]).index(gold)


def answer_vector(question: dict, answer: dict) -> list[float]:
    """A runtime answer as probabilities in option order (choice: criteria order; noul: false, true; score: levels)."""
    if question["type"] == "noul":
        return [1.0 - answer["noul"], answer["noul"]]
    if question["type"] == "score":
        return [answer["probabilities"][str(i)] for i in range(len(question["criteria"]))]
    return [answer["probabilities"][k] for k in question["criteria"]]


class D3Trainable(nn.Module):
    def __init__(self, root: str | Path = D3_FLASH, device: str = "cuda:0"):
        super().__init__()
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        self.root = Path(root)
        self.config = json.loads((self.root / "decision_config.json").read_text())
        if self.config.get("attention_mode") != "noncausal_full_attention" or self.config.get("pooling", "last") != "last":
            raise ValueError("expected d3-flash's noncausal full attention and last-token pooling")
        self.temperature = float(self.config.get("temperature", 1.0))
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.root))
        self.tokenizer.padding_side = "left"
        codes, token_ids = rt.answer_codes(self.tokenizer)
        if codes != self.config["codes"] or token_ids != self.config["token_ids"]:
            raise ValueError("answer codes differ from decision_config.json")
        self.codes, self.pad_id = codes, self.tokenizer.pad_token_id
        torch.manual_seed(20260919)  # as the runtime, before loading
        self.backbone = rt.backbone_class(rt.backbone_type(self.root)).from_pretrained(
            str(self.root), dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": str(self.device)})
        self.backbone.requires_grad_(False)
        if not rt.linearize_patch_embed(self.backbone):
            raise RuntimeError("vision patch embedding not found")
        rt.enable_noncausal_full_attention(self.backbone.language_model)
        readout = load_file(str(self.root / "readout.safetensors"))["weight"]
        self.register_buffer("readout", readout.to(self.device, torch.float32), persistent=False)
        self.processor = rt.load_processor(self.root)
        self.lora: dict[str, LoRALinear] = {}
        self.vision_blocks_trainable = False

    # ------------------------------------------------------------------ adapters

    def _locate(self, name: str) -> tuple[nn.Module, str]:
        """(parent module, child name) of an adapter name: "visual.<path>" in the vision tower, else in the text tower."""
        if name.startswith(VISION_PREFIX):
            root, name = self.backbone.visual, name[len(VISION_PREFIX):]
        else:
            root = self.backbone.language_model
        parent_name, _, child = name.rpartition(".")
        return (root.get_submodule(parent_name) if parent_name else root), child

    def _wrap(self, name: str, r: int, alpha: float) -> None:
        parent, child = self._locate(name)
        wrapped = LoRALinear(getattr(parent, child), r, alpha)
        setattr(parent, child, wrapped)
        self.lora[name] = wrapped
        if name.startswith(VISION_PREFIX + "blocks."):
            self.vision_blocks_trainable = True

    def add_vision_lora(self, merger_r: int = 0, blocks_r: int = 0, scale: float = 2.0) -> int:
        """Adapters on the merger and / or the vision blocks (alpha = scale * r); returns their parameter count."""
        before = set(self.lora)
        if merger_r:
            for name in MERGER_TARGETS:
                self._wrap(f"{VISION_PREFIX}merger.{name}", merger_r, scale * merger_r)
        if blocks_r:
            for i in range(len(self.backbone.visual.blocks)):
                for sub, names in VISION_BLOCK_TARGETS:
                    for name in names:
                        self._wrap(f"{VISION_PREFIX}blocks.{i}.{sub}.{name}", blocks_r, scale * blocks_r)
        return sum(p.numel() for n in set(self.lora) - before for p in (self.lora[n].lora_a, self.lora[n].lora_b))

    def lora_groups(self) -> dict[str, list[nn.Parameter]]:
        """Adapter parameters by part: text, merger, vision blocks (for per-part learning rates)."""
        out = {"text": [], "merger": [], "vision": []}
        for name, m in self.lora.items():
            part = ("merger" if name.startswith(VISION_PREFIX + "merger.") else "vision"
                    if name.startswith(VISION_PREFIX) else "text")
            out[part] += [m.lora_a, m.lora_b]
        return out

    def add_lora(self, r: int = 32, alpha: float = 64.0, targets: tuple[str, ...] = LORA_TARGETS) -> int:
        """Wrap the targeted text-tower linears; returns the number of trainable parameters."""
        text = self.backbone.language_model
        for parent_name, parent in list(text.named_modules()):
            for name, child in list(parent.named_children()):
                if name in targets and isinstance(child, nn.Linear) and min(child.in_features, child.out_features) > r:
                    wrapped = LoRALinear(child, r, alpha)
                    setattr(parent, name, wrapped)
                    self.lora[f"{parent_name}.{name}" if parent_name else name] = wrapped
        if not self.lora:
            raise RuntimeError("no LoRA target found")
        return sum(p.numel() for p in self.lora_parameters())

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.lora.values() for p in (m.lora_a, m.lora_b)]

    def lora_state(self) -> dict[str, torch.Tensor]:
        out = {}
        for name, m in self.lora.items():
            out[f"{name}.lora_a"] = m.lora_a.detach().float().cpu().contiguous()
            out[f"{name}.lora_b"] = m.lora_b.detach().float().cpu().contiguous()
        return out

    def save_lora(self, path: str | Path, meta: dict) -> None:
        from safetensors.torch import save_file

        first = next(iter(self.lora.values()))
        save_file(self.lora_state(), str(path), metadata={
            "format": "eojev-d3-lora/1", "r": str(first.r), "alpha": str(first.scale * first.r),
            "meta": json.dumps(meta)})

    def load_lora(self, path: str | Path) -> dict:
        """Attach the adapter stored at path; returns its metadata. Without adapters yet, every layer named in the file
        gets one, with the rank of its stored A matrix and the file's alpha / r scale (so text-only, vision and merged
        adapters all load)."""
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt") as f:
            info = f.metadata()
        state = load_file(str(path))
        names = {k.rsplit(".", 1)[0] for k in state}
        if not self.lora:
            scale = float(info["alpha"]) / int(info["r"])
            for name in sorted(names):
                r = state[f"{name}.lora_a"].shape[0]
                self._wrap(name, r, scale * r)
        if names != set(self.lora):
            raise ValueError(f"adapter covers {len(names)} layers, the model has {len(self.lora)}")
        with torch.no_grad():
            for name, m in self.lora.items():
                m.lora_a.copy_(state[f"{name}.lora_a"])
                m.lora_b.copy_(state[f"{name}.lora_b"])
        return info

    @torch.no_grad()
    def merge_lora(self) -> None:
        """Fold every adapter into its base weight and drop the adapters (BF16 result, as the package stores it)."""
        for name, m in self.lora.items():
            m.base.weight.copy_(m.merged_weight())
            parent, child = self._locate(name)
            setattr(parent, child, m.base)
        self.lora = {}
        self.vision_blocks_trainable = False

    def enable_checkpointing(self) -> None:
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen tower: eval mode keeps its blocks out of activation checkpointing; with block adapters it trains
        self.backbone.visual.train(mode and self.vision_blocks_trainable)
        return self

    # ------------------------------------------------------------------ forward

    def render(self, ex: Example) -> tuple[str, int]:
        return render_prompt(self.tokenizer, self.processor, self.codes, ex)

    def _text_hidden(self, texts: list[str]) -> torch.Tensor:
        """As D3.logits: tokenized without special tokens, left-padded with the pad id."""
        seqs = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):] = torch.as_tensor(s, dtype=torch.long)
            mask[i, width - len(s):] = 1
        with rt.sdpa_backends(self.device):
            out = self.backbone(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), use_cache=False)
        return out.last_hidden_state[:, -1]

    def _image_hidden(self, texts: list[str], images: list) -> torch.Tensor:
        """As D3.image_logits: the processor expands the placeholders, resizes and pads; the backbone gets every tensor."""
        encoded = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        with rt.sdpa_backends(self.device):
            out = self.backbone(**inputs, use_cache=False)
        return out.last_hidden_state[:, -1]

    def logits(self, examples: list[Example]) -> tuple[torch.Tensor, list[int]]:
        """Masked code logits (FP32, [B, 255]) in example order, and each question's option count.
        Text-only and image questions run as two batches, as the runtime never mixes them."""
        rendered = [self.render(ex) for ex in examples]
        hidden: list[torch.Tensor | None] = [None] * len(examples)
        text_idx = [i for i, ex in enumerate(examples) if not ex.images]
        image_idx = [i for i, ex in enumerate(examples) if ex.images]
        if text_idx:
            for i, h in zip(text_idx, self._text_hidden([rendered[i][0] for i in text_idx])):
                hidden[i] = h
        if image_idx:
            images = [im for i in image_idx for im in examples[i].images]
            for i, h in zip(image_idx, self._image_hidden([rendered[i][0] for i in image_idx], images)):
                hidden[i] = h
        logits = torch.stack(hidden).float() @ self.readout.T
        counts = [c for _, c in rendered]
        limit = torch.as_tensor(counts, device=self.device)[:, None]
        invalid = torch.arange(rt.MAX_OPTIONS, device=self.device)[None] >= limit
        return logits.masked_fill(invalid, float("-inf")), counts

    def probabilities(self, examples: list[Example]) -> list[list[float]]:
        logits, counts = self.logits(examples)
        probs = (logits / self.temperature).softmax(-1).cpu().tolist()
        return [p[:c] for p, c in zip(probs, counts)]


def render_prompt(tokenizer, processor, codes: list[str], ex: Example) -> tuple[str, int]:
    """The runtime's prompt for one question (D3.text / D3.image_text) and its number of options."""
    q = rt.normalize_question(ex.question)
    if ex.images:
        text = processor.apply_chat_template(
            rt.image_messages("d3", ex.state, q.rendered, codes, len(ex.images)),
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if text.count(rt.IMAGE_TOKEN) != len(ex.images):
            raise ValueError(f"{ex.key}: the prompt contains a literal image placeholder")
    else:
        text = rt.render(tokenizer, "d3", ex.state, q.rendered, codes)
    return text, len(q.keys)


def load_images(paths: list[Path]) -> list:
    """RGB PIL images exactly as the runtime decodes a path (no EXIF rotation, no resize)."""
    return [rt.load_image(str(p)) for p in paths]
