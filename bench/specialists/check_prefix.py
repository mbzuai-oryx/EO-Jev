"""Answer-prefix check for a specialist adapter (run this before a full logprob run): for the first rows of a rows file,
the top next tokens after the registered answer prefix (and common variants) and the probability mass on the option
labels. A label mass near 0 means the prefix does not end where the model writes its answer (TerraScope before the fix:
"\n" after <answer>) or the label token ids are wrong (SentencePiece "B" vs "▁B").

    PYTHONPATH=. python -m bench.specialists.check_prefix <model> <rows.jsonl> [n]
"""
import sys

import torch
from PIL import Image

from bench.common import CONVERTED
from bench.specialists import load
from bench.specialists.prompting import render

model_name, rows_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
VARIANTS = ["", "<answer>", "<think></think><answer>", "<think> </think><answer>", "<think>\n\n</think>\n<answer>",
            "<think>\n\n</think>\n\n<answer>", "<think>\n\n</think>\n<answer>\n"]

m = load(model_name)
capture = hasattr(m, "_prompt_embeds")
vhm = not hasattr(m, "answer_prefix")  # VHM: the authors' conv template, no prefix
if vhm:
    m.answer_prefix, m.answer_end = "", ""
prefixes = [m.answer_prefix] + [v for v in VARIANTS if v != m.answer_prefix]
print(f"model={model_name} registered prefix={m.answer_prefix!r} end={m.answer_end!r} capture={capture}")


@torch.inference_mode()
def next_probs(it, emb_cache, pre):
    if capture:
        if "emb" not in emb_cache:
            emb_cache["emb"] = m._prompt_embeds(it)
        emb = emb_cache["emb"]
        if pre:
            ids = torch.tensor([m.tok.encode(pre, add_special_tokens=False)], device=emb.device)
            emb = torch.cat([emb, m._embed(ids).to(emb.dtype)], dim=1)
        mask = torch.ones(emb.shape[:2], device=emb.device, dtype=torch.long)
        return torch.softmax(m.lm(inputs_embeds=emb, attention_mask=mask).logits[0, -1].float(), dim=-1)
    if vhm:
        ids, images, _ = m._inputs(it)
        return torch.softmax(m.model(input_ids=ids, images=images).logits[0, -1].float(), dim=-1)
    inputs = m._inputs([m.prompt(it, pre)], [it.images])
    return torch.softmax(m.model(**inputs, logits_to_keep=1).logits[0, -1].float(), dim=-1)


from bench.specialists.prompting import first_token_ids
label_ids = (lambda lab: first_token_ids(m.tok, lab)) if vhm else (m._label_ids if capture else m._token_ids)
rows = [json.loads(l) for l, _ in zip(open(rows_path), range(n))]
for row in rows:
    ims = []
    for rel in row["images"]:
        with Image.open(CONVERTED / rel) as im:
            ims.append(im.convert("RGB"))
    for qid in list(row["questions"])[:2]:
        it = render(row, qid)
        it.images = ims
        print(f"=== {row['id']} {qid} mode={it.mode} gold={row['answer'][qid]} labels={it.labels[:6]}")
        cache = {}
        for k, pre in enumerate(prefixes):
            if k > 0 and (it.mode == "names" or vhm):
                break
            p = next_probs(it, cache, pre)
            top = torch.topk(p, 6)
            mass = sum(float(p[label_ids(lab)].sum()) for lab in it.labels) if it.mode != "names" else float("nan")
            toks = ", ".join(f"{m.tok.decode([int(i)])!r}:{float(v):.2f}" for v, i in zip(top.values, top.indices))
            print(f"  {'*' if k == 0 else ' '} prefix={pre!r:36s} label_mass={mass:.3f}  top: {toks}")
            if it.mode == "names" and not capture and not vhm:  # what the model writes after the prefix
                with torch.inference_mode():
                    inputs = m._inputs([m.prompt(it, pre)], [it.images])
                    g = m.model.generate(**inputs, max_new_tokens=12, do_sample=False, temperature=None, top_p=None,
                                         top_k=None, use_cache=True)
                print("    greedy:", repr(m.tok.decode(g[0, inputs["input_ids"].shape[1]:])),
                      "| scored:", [round(x, 3) for x in m._names(it)][:10], "| options:", it.options[:10])
