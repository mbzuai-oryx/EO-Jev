"""Base adapter for specialists that ship their own inference code (TerraScope, EarthMind, EarthDial, VHM).

Subclasses set self.lm (the language model inside the VLM), self.tok, self.answer_prefix, self.answer_end and
implement _predict(item) -> generated text, calling the authors' own prediction function.

generate: _predict(item) as is.
logprob: _predict(item) is run until the language model receives its input embeddings (image features spliced into
the templated prompt); that call is intercepted, the answer prefix is appended, and the option probabilities are read
from one forward pass. The visual input path is therefore exactly the authors' code in both modes.
"""

from __future__ import annotations

import torch

from bench.specialists.prompting import Item, first_token_ids


class _Captured(Exception):
    pass


class CaptureAdapter:
    lm: torch.nn.Module
    tok: object
    answer_prefix: str = ""
    answer_end: str = ""
    answer_context: str = ""  # end of the prompt text, for tokenising scored answers in context (_cont_ids)
    name_batch: int = 16

    def _predict(self, item: Item) -> str:  # pragma: no cover - subclasses
        raise NotImplementedError

    # ---------------------------------------------------------------- input capture
    def _prompt_embeds(self, item: Item) -> torch.Tensor:
        """Input embeddings the model builds for this item, shape (1, L, C)."""
        lm, got = self.lm, {}
        orig_forward = lm.forward

        def grab(*args, **kw):
            emb = kw.get("inputs_embeds")
            if emb is not None and emb.shape[1] > 1:
                got["emb"] = emb.detach()
                raise _Captured
            return orig_forward(*args, **kw)

        def grab_generate(*args, **kw):
            got["emb"] = kw["inputs_embeds"].detach()
            raise _Captured

        saved = {k: lm.__dict__[k] for k in ("forward", "generate") if k in lm.__dict__}  # e.g. accelerate hooks
        lm.forward, lm.generate = grab, grab_generate
        try:
            self._predict(item)
        except _Captured:
            pass
        finally:
            for k in ("forward", "generate"):
                if k in saved:
                    setattr(lm, k, saved[k])
                else:
                    delattr(lm, k)
        if "emb" not in got:
            raise RuntimeError("prompt embeddings were not captured")
        return got["emb"]

    # ---------------------------------------------------------------- logprob mode
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids.to(self.lm.get_input_embeddings().weight.device))

    def _cont_ids(self, text: str) -> list[int]:
        """Token ids of `text` as it continues the prompt. SentencePiece tokenizers (LLaMA, Phi-3) start a string with
        a word marker ("▁destroyed") that the model never writes after its turn opener (EarthDial writes "destroy",
        "ed"); so `text` is tokenised after answer_context, the prompt's last characters, and that prefix dropped."""
        ctx = self.answer_context
        if not ctx:
            return self.tok.encode(text, add_special_tokens=False)
        base = self.tok.encode(ctx, add_special_tokens=False)
        full = self.tok.encode(ctx + text, add_special_tokens=False)
        if full[: len(base)] != base:
            raise ValueError(f"answer_context {ctx!r} does not tokenise as a prefix of {ctx + text!r}")
        return full[len(base):]

    def _label_ids(self, label: str) -> list[int]:
        cache = self.__dict__.setdefault("_ids", {})
        if label not in cache:
            cache[label] = first_token_ids(self.tok, label)
        return cache[label]

    @torch.inference_mode()
    def _score_one(self, item: Item) -> list[float]:
        emb = self._prompt_embeds(item)
        dev = emb.device
        if self.answer_prefix:
            prefix = torch.tensor([self.tok.encode(self.answer_prefix, add_special_tokens=False)], device=dev)
            emb = torch.cat([emb, self._embed(prefix).to(emb.dtype)], dim=1)
        if item.mode != "names":
            mask = torch.ones(emb.shape[:2], device=dev, dtype=torch.long)
            p = torch.softmax(self.lm(inputs_embeds=emb, attention_mask=mask).logits[0, -1].float(), dim=-1)
            return [float(p[self._label_ids(lab)].sum()) for lab in item.labels]
        conts = [self._cont_ids(n + self.answer_end) for n in item.options]
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        logps, L = [], emb.shape[1]
        for s in range(0, len(conts), self.name_batch):
            chunk = conts[s: s + self.name_batch]
            ids = torch.full((len(chunk), max(len(c) for c in chunk)), pad, dtype=torch.long, device=dev)
            for i, c in enumerate(chunk):  # right padding: padded positions come after every scored token
                ids[i, : len(c)] = torch.tensor(c, device=dev)
            x = torch.cat([emb.expand(len(chunk), -1, -1), self._embed(ids).to(emb.dtype)], dim=1)
            mask = torch.ones(x.shape[:2], device=dev, dtype=torch.long)
            lp = torch.log_softmax(self.lm(inputs_embeds=x, attention_mask=mask).logits[:, L - 1:].float(), dim=-1)
            for i, c in enumerate(chunk):
                logps.append(float(lp[i, torch.arange(len(c), device=dev), torch.tensor(c, device=dev)].sum()))
        return torch.softmax(torch.tensor(logps), dim=0).tolist()

    def score(self, items: list[Item]) -> list[list[float]]:
        return [self._score_one(it) for it in items]

    # ---------------------------------------------------------------- generate mode
    @torch.inference_mode()
    def generate(self, items: list[Item]) -> list[str]:
        return [self._predict(it) for it in items]
