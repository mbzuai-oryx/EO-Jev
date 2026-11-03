"""Model-independent prompts and answer parsing for the specialist models.

Every specialist sees the same text for a question, built from our converted rows. The only differences between
models are their chat templates, system prompts and answer tags, which live in the adapters:
- choice with up to 26 options: lettered options, answered with a letter.
- scene classification (and any choice with more than 26 options): the class list, answered with the class name,
  which is how these models are trained and evaluated in their own papers.
- noul: answered yes or no.
- score: the ordered levels as lettered options.

Two answer modes (bench/runners/vlm.py):
- logprob: the answer is prefilled up to the point where the model writes it (e.g. after "<answer>"). The
  probability of each option is read from the next-token distribution over the letters / yes-no, or, for class names,
  from the probability of the whole name followed by the end of the answer.
- generate: greedy decoding with the model's own reasoning format, then parse_answer() maps the text to an option.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_SUFFIX = "Answer with the option's letter from the given choices."
YESNO_SUFFIX = "Answer yes or no."
NAMES_SUFFIX = "Answer with the class name only."


@dataclass
class Item:
    """One question of one row, rendered for a model."""

    row_id: str
    qid: str
    qtype: str            # choice | noul | score
    mode: str             # letters | yesno | names
    text: str             # user text; images are passed separately and placed before it
    options: list[str]    # option texts (choice), level names (score), [] for noul
    images: list = field(default_factory=list)
    native: bool = False  # text is already in the model's own format (bench/specialists/native.py): use it as is
    turns: list = field(default_factory=list)  # earlier (user, assistant) turns; images go with the first user turn

    @property
    def labels(self) -> list[str]:
        """The strings the model is expected to answer with, one per option."""
        if self.mode == "letters":
            return list(LETTERS[: len(self.options)])
        if self.mode == "yesno":
            return ["yes", "no"]
        return list(self.options)


def render(row: dict, qid: str) -> Item:
    q = row["questions"][qid]
    head = row.get("state", "").strip()
    head = head + "\n" if head else ""
    instr = q["instructions"].strip()
    if q["type"] == "noul":
        return Item(row["id"], qid, "noul", "yesno", f"{head}{instr}\n{YESNO_SUFFIX}", [])
    options = list(q["criteria"])
    if q["type"] == "choice" and (row.get("task_family") == "scene_cls" or len(options) > len(LETTERS)):
        text = f"{head}{instr}\nChoose one of the following classes: {', '.join(options)}.\n{NAMES_SUFFIX}"
        return Item(row["id"], qid, "choice", "names", text, options)
    listing = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return Item(row["id"], qid, q["type"], "letters", f"{head}{instr}\n{listing}\n{LETTER_SUFFIX}", options)


def to_answer(item: Item, probs: list[float]) -> dict:
    """Probabilities over item.labels -> the answer record format bench/metrics/score.py reads."""
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]
    if item.qtype == "noul":
        return {"type": "noul", "noul": probs[0]}
    if item.qtype == "score":
        return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
                "probabilities": {str(i): p for i, p in enumerate(probs)}}
    best = max(range(len(probs)), key=probs.__getitem__)
    return {"type": "choice", "choice": item.options[best], "probabilities": dict(zip(item.options, probs))}


UNPARSED = "__unparsed__"


def to_answer_generated(item: Item, index: int | None) -> dict:
    """A parsed generated answer -> one-hot answer record. Unparsed answers count as wrong for choice (an extra
    sentinel option takes all the mass), as "yes" with p=0.5 for noul, and as level 0 for score; the runner records
    every parse failure so their rate is reported."""
    if index is None:
        if item.qtype == "choice":
            probs = {o: 0.0 for o in item.options} | {UNPARSED: 1.0}
            return {"type": "choice", "choice": UNPARSED, "probabilities": probs, "unparsed": True}
        if item.qtype == "noul":
            return {"type": "noul", "noul": 0.5, "unparsed": True}
        n = len(item.options)
        return {**to_answer(item, [1.0] * n), "unparsed": True}
    onehot = [0.0] * len(item.labels)
    onehot[index] = 1.0
    return to_answer(item, onehot)


_TAG_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_SPECIAL = re.compile(r"<\|[^|>]*\|>|</?(?:s|eos|bos)>|<\|?endoftext\|?>")


def answer_span(text: str) -> str:
    """The part of a generation that holds the final answer."""
    text = _SPECIAL.sub("", text)
    tagged = _TAG_ANSWER.findall(text)
    if tagged:
        return tagged[-1].strip()
    rest = _THINK.sub("", text).strip()
    if "<think>" in rest.lower():  # unfinished reasoning: nothing to parse
        return ""
    return rest or text.strip()


YES_FORMS = ["yes", "Yes", "YES"]
NO_FORMS = ["no", "No", "NO"]


def first_token_ids(tok, label: str) -> list[int]:
    """Token ids that start an answer `label` ("A", "yes", ...), with and without a leading space. Leading tokens that
    decode to whitespace only (SentencePiece's bare "▁") are skipped, so they never count as the label."""
    forms = {"yes": YES_FORMS, "no": NO_FORMS}.get(label, [label])
    ids = set()
    for f in forms:
        for s in (f, " " + f):
            for t in tok.encode(s, add_special_tokens=False):
                if tok.decode([t]).strip():
                    ids.add(t)
                    break
        # SentencePiece tokenizers (LLaMA, Phi-3) encode "B" as "▁B", but after a newline the model writes the bare
        # piece "B"; add it when it is a vocabulary entry (BPE tokenizers already got it from encode above)
        t = tok.convert_tokens_to_ids(f)
        if isinstance(t, int) and t != tok.unk_token_id and tok.decode([t]).strip() == f:
            ids.add(t)
    return sorted(ids)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_answer(item: Item, text: str) -> int | None:
    """Index into item.labels, or None if the answer cannot be read."""
    span = answer_span(text)
    if not span:
        return None
    if item.mode == "yesno":
        m = re.search(r"\b(yes|no|true|false)\b", span, re.I)
        return None if m is None else (0 if m.group(1).lower() in ("yes", "true") else 1)
    if item.mode == "letters":
        letters = item.labels
        pats = [r"^\s*\(?([A-Z])\)?(?:[.):,\s]|$)",                       # "B", "B.", "(B) ..."
                r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\b"]   # "the answer is B"
        if len(span) <= 40:  # short answers only: in prose, "I" or "A" are words, not options
            pats.append(r"\b([A-Z])\b")
        for pat in pats:
            for m in re.finditer(pat, span):
                if m.group(1) in letters:
                    return letters.index(m.group(1))
        names = _match_name(item.options, span)  # answered with the option text instead of its letter
        return names
    return _match_name(item.options, span)


def _match_name(options: list[str], span: str) -> int | None:
    norm_opts = [_norm(o) for o in options]
    s = _norm(span)
    if s in norm_opts:
        return norm_opts.index(s)
    contained = [(len(o), i) for i, o in enumerate(norm_opts) if o and re.search(rf"\b{re.escape(o)}\b", s)]
    if contained:
        return max(contained)[1]   # longest option mentioned ("railway station" over "railway")
    close = difflib.get_close_matches(s, norm_opts, n=1, cutoff=0.75)
    if close:
        return norm_opts.index(close[0])
    # the answer names part of exactly one option ("sea" for EuroSAT's "sea or lake")
    part_of = [i for i, o in enumerate(norm_opts) if s and re.search(rf"\b{re.escape(s)}\b", o)]
    return part_of[0] if len(part_of) == 1 else None
