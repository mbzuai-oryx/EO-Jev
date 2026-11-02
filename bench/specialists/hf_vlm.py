"""Adapter for specialists that load with stock transformers (Qwen2-VL, Qwen2.5-VL, Qwen3-VL and GLM-4.1V bases).

    adapter = HFVLM(path, system=..., answer_prefix="<answer>", answer_end="</answer>")
    adapter.score(items)      -> list of probability lists over item.labels   (logprob mode)
    adapter.generate(items)   -> list of generated strings                    (generate mode)

answer_prefix is appended after the chat template's generation prompt in logprob mode, so the next token is the
answer itself (for reasoning models this skips the reasoning, e.g. "<think>\n\n</think>\n<answer>"); answer_end is
what follows a complete answer (closing tag or end-of-turn token) and is scored with class names so that a name
that is a prefix of another ("railway" / "railway station") is not favoured.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from bench.specialists.prompting import Item, first_token_ids


class HFVLM:
    def __init__(self, path: str, *, system: str | None = None, answer_prefix: str = "", answer_end: str | None = None,
                 gen_kwargs: dict | None = None, max_new_tokens: int = 1024, batch_size: int = 8,
                 name_batch: int = 16, processor_kwargs: dict | None = None, adapter: str | None = None,
                 load_in_4bit: bool = False):
        self.processor = AutoProcessor.from_pretrained(path, **(processor_kwargs or {}))
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        quant = None
        if load_in_4bit:  # as LLaMA-Factory's quantization_bit=4 (QLoRA adapters were trained on this)
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
            quantization_config=quant).eval()
        if adapter:  # a LoRA adapter on top of the base checkpoint at `path`
            from peft import PeftModel  # noqa: PLC0415
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
            n_lora = sum(1 for name, _ in self.model.named_modules() if name.endswith("lora_A"))
            if not n_lora:
                raise RuntimeError(f"adapter {adapter} injected no LoRA layers (module names do not match)")
            print(f"adapter {adapter}: {n_lora} LoRA layers", flush=True)
        self.system, self.answer_prefix = system, answer_prefix
        self.answer_end = answer_end if answer_end is not None else self._end_of_turn()
        # use_cache explicitly: some checkpoints ship generation_config use_cache=false (RSThinker), which makes
        # every decoding step recompute the whole sequence
        self.gen_kwargs = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None, "use_cache": True,
                           **(gen_kwargs or {})}
        self.max_new_tokens, self.batch_size, self.name_batch = max_new_tokens, batch_size, name_batch
        self._letter_ids: dict[str, list[int]] = {}

    # ---------------------------------------------------------------- prompt building
    def _end_of_turn(self) -> str:
        tpl = self.tok.chat_template or getattr(self.processor, "chat_template", "") or ""
        for tok in ("<|im_end|>", "<|user|>", "<|endoftext|>"):
            if tok in tpl:
                return tok
        return self.tok.eos_token

    def prompt(self, item: Item, prefill: str = "") -> str:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": self.system}]})
        images = [{"type": "image"} for _ in item.images]
        for user, assistant in item.turns:  # earlier turns of the model's own dialogue format (native.py)
            msgs.append({"role": "user", "content": images + [{"type": "text", "text": user}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant}]})
            images = []
        msgs.append({"role": "user", "content": images + [{"type": "text", "text": item.text}]})
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill

    def _inputs(self, prompts: list[str], image_lists: list[list]):
        images = [im for ims in image_lists for im in ims]
        return self.processor(text=prompts, images=images or None, padding=True, return_tensors="pt").to("cuda:0")

    # ---------------------------------------------------------------- logprob mode
    def _token_ids(self, label: str) -> list[int]:
        if label not in self._letter_ids:
            self._letter_ids[label] = first_token_ids(self.tok, label)
        return self._letter_ids[label]

    @torch.inference_mode()
    def _first_token(self, items: list[Item]) -> list[list[float]]:
        prompts = [self.prompt(it, self.answer_prefix) for it in items]
        inputs = self._inputs(prompts, [it.images for it in items])
        logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float()
        probs = torch.softmax(logits, dim=-1)
        out = []
        for it, p in zip(items, probs):
            out.append([float(p[self._token_ids(lab)].sum()) for lab in it.labels])
        return out

    @torch.inference_mode()
    def _names(self, item: Item) -> list[float]:
        """log P(name + answer_end | prompt) for every class name, as softmax-normalised probabilities."""
        prompt = self.prompt(item, self.answer_prefix)
        base = self._inputs([prompt], [item.images])
        p_ids = base["input_ids"][0]
        extra = {k: v for k, v in base.items() if k not in ("input_ids", "attention_mask")}
        conts = [self.tok.encode(name + self.answer_end, add_special_tokens=False) for name in item.options]
        logps = []
        for s in range(0, len(conts), self.name_batch):
            chunk = conts[s: s + self.name_batch]
            n, longest = len(chunk), max(len(c) for c in chunk)
            total = len(p_ids) + longest
            ids = torch.full((n, total), self.tok.pad_token_id, dtype=torch.long, device="cuda:0")
            mask = torch.zeros((n, total), dtype=torch.long, device="cuda:0")
            for i, c in enumerate(chunk):  # left padding: every sequence ends at the last position
                seq = torch.cat([p_ids, torch.tensor(c, device="cuda:0")])
                ids[i, total - len(seq):] = seq
                mask[i, total - len(seq):] = 1
            batch_extra = {}
            for k, v in extra.items():
                if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, len(p_ids)):
                    # per-token side inputs (e.g. multimodal token types): pad exactly like input_ids
                    side = torch.zeros((n, total), dtype=v.dtype, device=v.device)
                    for i, c in enumerate(chunk):
                        start = total - len(p_ids) - len(c)
                        side[i, start: start + len(p_ids)] = v[0]
                    batch_extra[k] = side
                else:  # image tensors: repeat once per sequence
                    batch_extra[k] = torch.cat([v] * n, dim=0) if torch.is_tensor(v) else v
            keep = longest + 1
            logits = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=keep, **batch_extra).logits.float()
            lp = torch.log_softmax(logits, dim=-1)
            for i, c in enumerate(chunk):
                pos = torch.arange(keep - len(c) - 1, keep - 1, device="cuda:0")
                logps.append(float(lp[i, pos, torch.tensor(c, device="cuda:0")].sum()))
        lp = torch.tensor(logps)
        return torch.softmax(lp, dim=0).tolist()

    def score(self, items: list[Item]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(items)
        first = [i for i, it in enumerate(items) if it.mode != "names"]
        for s in range(0, len(first), self.batch_size):
            idx = first[s: s + self.batch_size]
            for i, p in zip(idx, self._first_token([items[i] for i in idx])):
                out[i] = p
        for i, it in enumerate(items):
            if it.mode == "names":
                out[i] = self._names(it)
        return out  # type: ignore[return-value]

    # ---------------------------------------------------------------- generate mode
    @torch.inference_mode()
    def generate(self, items: list[Item]) -> list[str]:
        out = []
        for s in range(0, len(items), self.batch_size):
            chunk = items[s: s + self.batch_size]
            inputs = self._inputs([self.prompt(it) for it in chunk], [it.images for it in chunk])
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id, **self.gen_kwargs)
            new = gen[:, inputs["input_ids"].shape[1]:]
            pad = self.tok.pad_token or ""
            # keep special tokens (some models tag reasoning with them) but drop the batch padding
            out += [t.replace(pad, "") if pad else t for t in self.tok.batch_decode(new, skip_special_tokens=False)]
        return out
