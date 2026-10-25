"""Where do the [CLM] steps attend? Step-pass attention over the cached document, per layer.

For each eval doc, run n_gold steps, match them to the gold claims (Hungarian), and report per layer, averaged over
heads, steps and docs:
  mass_sink / mass_eos / mass_body / mass_prompt / mass_steps
                 share of attention on the doc's first token, its <|endoftext|>, the other doc tokens ("body"),
                 the soft prompt, and the steps themselves (earlier steps + self)
  body_entropy   entropy of a step's attention over the body (renormalized, heads averaged) / log(#body tokens)
  pair_js        mean pairwise Jensen-Shannon divergence (base 2, 0 = same, 1 = disjoint) between steps' body
                 distributions
  pair_jaccard   mean pairwise Jaccard of steps' top-20 body tokens
  own_claim      body attention on tokens unique to the step's matched claim (not in the doc's other claims), minus
                 the mean of the same over the doc's other claims (> 0 = a step reads its own claim's words)

  python -m embedrag.analyze_attention --ckpts dir1 dir2 --init_strides 1 24 --eval_file data/claimemb/small.dev.jsonl
"""
import argparse
import json
import os

import numpy as np
import torch

from .config import QWEN3_EMB, ClaimEmbConfig
from .data import ClaimCollator
from .model import ClaimEmb, hungarian_targets

METRICS = ["mass_sink", "mass_eos", "mass_body", "mass_prompt", "mass_steps",
           "body_entropy", "pair_js", "pair_jaccard", "own_claim"]


def step_attentions(model, cache, doc_attention_mask, n):
    """model.steps with attention weights: list over layers of (heads, P + n, K)."""
    orig, box = model.backbone.forward, {}

    def forward(*args, **kwargs):
        out = orig(*args, output_attentions=True, **kwargs)
        box["att"] = out.attentions
        return out

    model.backbone.forward = forward
    try:
        step_mask = torch.ones(1, n, dtype=torch.bool, device=doc_attention_mask.device)
        e, _ = model.steps(cache, doc_attention_mask, step_mask)
    finally:
        del model.backbone.forward
    assert box["att"] is not None and box["att"][0] is not None, "no attention weights (eager attention needed)"
    return e, [a[0].float() for a in box["att"]]


def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: (a * (a.clamp_min(1e-12) / b.clamp_min(1e-12)).log2()).sum()
    return float((kl(p, m) + kl(q, m)) / 2)


@torch.no_grad()
def analyze_doc(model, collator, ex, max_steps):
    dev = model.step_query.device
    claims = ex["claims"][:max_steps]
    n = len(claims)
    d = {k: v.to(dev) for k, v in collator.docs([ex["text"]]).items()}
    t = model.teacher(**{k.replace("claim_", ""): v.to(dev) for k, v in collator.claims(claims).items()})
    _, cache = model.read_doc(**d)
    e, atts = step_attentions(model, cache, d["doc_attention_mask"], n)
    L = d["doc_input_ids"].size(1)
    P = 0 if model.soft_prompt is None else model.soft_prompt.size(0)
    y = hungarian_targets(e, t, torch.arange(n, device=dev), torch.zeros(n, dtype=torch.long, device=dev)).tolist()

    body_ids = d["doc_input_ids"][0, 1:L - 1]
    sets = [set(collator.tokenizer(c, add_special_tokens=False)["input_ids"]) for c in claims]
    uniq = [torch.isin(body_ids, torch.tensor(sorted(s - set().union(*(sets[:k] + sets[k + 1:]))) or [-1],
                                              device=dev)) for k, s in enumerate(sets)]

    out = {m: [] for m in METRICS}
    for a in atts:
        a = a[:, P:]  # step rows: (heads, n, L + P + n)
        out["mass_sink"].append(float(a[..., 0].mean()))
        out["mass_eos"].append(float(a[..., L - 1].mean()))
        out["mass_body"].append(float(a[..., 1:L - 1].sum(-1).mean()))
        out["mass_prompt"].append(float(a[..., L:L + P].sum(-1).mean()))
        out["mass_steps"].append(float(a[..., L + P:].sum(-1).mean()))
        b = a[..., 1:L - 1].mean(0)
        p = b / b.sum(-1, keepdim=True).clamp_min(1e-12)  # (n, body)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1) / np.log(p.size(-1))
        out["body_entropy"].append(float(ent.mean()))
        if n < 2:
            for m in ("pair_js", "pair_jaccard", "own_claim"):
                out[m].append(float("nan"))
            continue
        top = [set(r.topk(min(20, p.size(-1))).indices.tolist()) for r in p]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        out["pair_js"].append(np.mean([js(p[i], p[j]) for i, j in pairs]))
        out["pair_jaccard"].append(np.mean([len(top[i] & top[j]) / len(top[i] | top[j]) for i, j in pairs]))
        own = [float(p[i][uniq[y[i]]].sum()) - np.mean([float(p[i][uniq[k]].sum()) for k in range(n) if k != y[i]])
               for i in range(n)]
        out["own_claim"].append(np.mean(own))
    return out


def analyze(model, collator, docs, max_steps):
    model.backbone.config._attn_implementation = "eager"  # sdpa returns no attention weights
    per_doc = [analyze_doc(model, collator, ex, max_steps) for ex in docs]
    return {m: np.nanmean([r[m] for r in per_doc], 0).tolist() for m in METRICS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="*", default=[], help="ClaimEmb checkpoints")
    p.add_argument("--init_strides", nargs="*", type=int, default=[], help="also analyze untrained models")
    p.add_argument("--backbone", default=QWEN3_EMB, help="for --init_strides")
    p.add_argument("--eval_file", required=True)
    p.add_argument("--num_docs", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.eval_file) as f:
        docs = [json.loads(l) for _, l in zip(range(args.num_docs), f)]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [(f"init-stride{s}", lambda s=s: ClaimEmb(ClaimEmbConfig(backbone=args.backbone, step_stride=s)))
            for s in args.init_strides]
    runs += [(os.path.basename(c.rstrip("/")), lambda c=c: ClaimEmb.from_pretrained(c)) for c in args.ckpts]

    res = {}
    for name, load in runs:
        model = load().to(dev).eval()
        res[name] = analyze(model, ClaimCollator(model.tokenizer, model.cfg.max_claims), docs, model.cfg.max_claims)
        del model
        torch.cuda.empty_cache()
        print(name, {m: round(float(np.mean(v)), 4) for m, v in res[name].items()}, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=1)

    n_layers = len(next(iter(res.values()))["pair_js"])
    show = sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})
    for m in METRICS:
        print(f"\n{m:24s}" + "".join(f"  L{l:<5d}" for l in show) + "  mean")
        for name, r in res.items():
            print(f"  {name:22s}" + "".join(f"  {r[m][l]:.4f}" for l in show) + f"  {np.mean(r[m]):.4f}")


if __name__ == "__main__":
    main()
