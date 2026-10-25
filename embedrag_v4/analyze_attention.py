"""Where do v4 grid queries attend in the document? Maps each query's attention back onto doc tokens.

For each eval doc: the grid queries are Hungarian-matched to the doc's gold claims (as in training); attention
of each query over the doc tokens (heads averaged; renormalized over doc tokens, the first token excluded) per layer:
  sink          share of the query's attention on the first token (the attention sink)
  near_ptr      doc attention within +-16 tokens of the query's pointer p_t
  near_ptr_lift near_ptr / (share of doc tokens in that window): 1 = no preference for the pointer's surroundings
  dist_ptr      attention-weighted |token - p_t| / doc length
  own_span      matched queries: doc attention on the matched claim's aligned span
  own_lift      own_span / (span length / doc length)
  own_top1      matched queries: the matched claim's span gets the most attention among the doc's claim spans
Per doc, also: |p_t - end of the matched claim's span| / doc length (do queries take claims near their pointer?)
Closeness: per gold claim, cos to its matched query ("matched") and the best cos over the kept queries ("kept_best")
(full lists saved, for distributions).
Examples: for the first --n_examples docs, each kept query's pointer, matched claim and top attended words
(attention averaged over layers --example_layers).

  python -m embedrag_v4.analyze_attention --ckpts ckptA ckptB --init_stride 16 \\
      --eval_file data/claimemb/rag25.dev.regions.jsonl --num_docs 100 --output out.json
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .config import QWEN3_EMB, ClaimEmbConfig
from .data import ClaimCollator
from .model import ClaimEmb

METRICS = ["sink", "near_ptr", "near_ptr_lift", "dist_ptr", "own_span", "own_lift", "own_top1"]


@torch.no_grad()
def analyze_doc(model, collator, ex, window, example_layers, n_top):
    dev = next(model.parameters()).device
    b = {k: v.to(dev) for k, v in collator([ex]).items()}
    n_doc = int(b["doc_attention_mask"].sum()) - 1
    spans, gmask = model.grid_spans(b["doc_attention_mask"])
    _, cache = model.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
    e, hq, atts = model.queries(cache, b["doc_attention_mask"], spans, gmask, return_hidden=True, return_attentions=True)
    t = model.teacher(b["claim_input_ids"], b["claim_attention_mask"])
    rows, cols = linear_sum_assignment(-(e @ t.T).float().cpu().numpy())
    match = dict(zip(rows.tolist(), cols.tolist()))
    ptr = spans[0, gmask[0], 1].tolist()
    cspans = b["claim_spans"][0, b["claim_mask"][0]].tolist()
    L = b["doc_input_ids"].size(1)
    off = L - n_doc - 1                                                   # left padding; doc tokens [off, off + n_doc)
    pos = torch.arange(n_doc, device=dev).float()

    out = {m: [] for m in METRICS}
    maps = []
    for a in atts:
        a = a[0, :, : len(ptr)].float().mean(0)                           # (Q, L + T), heads averaged
        sink = a[:, off]
        d = a[:, off + 1: off + n_doc]                                    # doc tokens, first excluded
        d = d / d.sum(-1, keepdim=True).clamp_min(1e-12)
        d = torch.cat([torch.zeros_like(d[:, :1]), d], 1)                 # back to doc token index
        maps.append(d)
        near, lift, dist, own, ownl, top1 = [], [], [], [], [], []
        for q, p in enumerate(ptr):
            lo, hi = max(1, p - window), min(n_doc - 1, p + window)
            frac = (hi - lo + 1) / max(n_doc - 1, 1)
            near.append(float(d[q, lo:hi + 1].sum()))
            lift.append(near[-1] / frac)
            dist.append(float((d[q] * (pos - p).abs()).sum()) / n_doc)
            if q in match:
                s, e_ = cspans[match[q]]
                own.append(float(d[q, s:e_ + 1].sum()))
                ownl.append(own[-1] / ((e_ - s + 1) / n_doc))
                per = [float(d[q, s2:e2 + 1].sum()) for s2, e2 in cspans]
                top1.append(float(int(np.argmax(per)) == match[q]))
        for k, v in zip(METRICS, [float(sink.mean()), np.mean(near), np.mean(lift), np.mean(dist),
                                  np.mean(own) if own else np.nan, np.mean(ownl) if ownl else np.nan,
                                  np.mean(top1) if top1 else np.nan]):
            out[k].append(v)
    offset = [abs(ptr[q] - cspans[c][1]) / n_doc for q, c in match.items()]
    keep = torch.ones(len(ptr), dtype=torch.bool, device=dev)
    if model.keep_head is not None:                                       # as generate(): has-claim > 0, at least one
        logits = model.keep_head(hq).squeeze(-1)
        keep = logits > 0
        keep[logits.argmax()] = True
    sim = (e @ t.T).float()
    inv = {c: q for q, c in match.items()}
    close = [{"claim": ex["claims"][c], "matched": float(sim[inv[c], c]) if c in inv else None,
              "kept_best": float(sim[keep, c].max())} for c in range(len(cspans))]

    example = None
    if n_top:
        ids = b["doc_input_ids"][0, off: off + n_doc].tolist()
        tok = model.tokenizer
        avg = torch.stack([maps[l] for l in example_layers]).mean(0)
        example = []
        for q, p in enumerate(ptr):
            top = avg[q].topk(min(n_top, n_doc)).indices.sort().values.tolist()
            example.append({"query": q, "pointer": p, "pointer_word": tok.decode(ids[p]), "kept": bool(keep[q]),
                            "matched_claim": ex["claims"][match[q]] if q in match else None,
                            "matched_cos": float(sim[q, match[q]]) if q in match else None,
                            "top_words": [tok.decode(ids[i]).strip() for i in top]})
    return out, offset, example, close


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="*", default=[])
    p.add_argument("--init_stride", type=int, default=0, help="also analyze an untrained grid model (0 = skip)")
    p.add_argument("--backbone", default=QWEN3_EMB)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--num_docs", type=int, default=100)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--n_examples", type=int, default=2)
    p.add_argument("--n_top", type=int, default=12)
    p.add_argument("--example_layers", default="14-21")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lo, hi = map(int, args.example_layers.split("-"))
    docs = [json.loads(l) for _, l in zip(range(args.num_docs), open(args.eval_file))]
    runs = [(os.path.basename(c.rstrip("/")), lambda c=c: ClaimEmb.from_pretrained(c)) for c in args.ckpts]
    if args.init_stride:
        runs.append(("untrained", lambda: ClaimEmb(ClaimEmbConfig(backbone=args.backbone, query="grid",
                                                                  query_pos="end", stride=args.init_stride))))
    res = {}
    for name, load in runs:
        model = load().to(dev).eval()
        model.backbone.config._attn_implementation = "eager"             # sdpa returns no attention weights
        collator = ClaimCollator(model.tokenizer, model.cfg.max_claims)
        per, offs, ex, close = [], [], [], []
        for i, d in enumerate(docs):
            o, off, e, c = analyze_doc(model, collator, d, args.window, range(lo, hi + 1),
                                       args.n_top if i < args.n_examples else 0)
            per.append(o); offs += off; close += c
            if e: ex.append({"claims": d["claims"], "queries": e})
        res[name] = {"layers": {m: np.nanmean([r[m] for r in per], 0).tolist() for m in METRICS},
                     "ptr_to_matched_claim_end": {"mean": float(np.mean(offs)), "median": float(np.median(offs))},
                     "closeness": close, "examples": ex}
        print(name, {m: round(float(np.nanmean(v)), 3) for m, v in res[name]["layers"].items()},
              res[name]["ptr_to_matched_claim_end"],
              {k: round(float(np.mean([x[k] for x in close if x[k] is not None])), 3) for k in ("matched", "kept_best")},
              flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=1)

    nl = len(next(iter(res.values()))["layers"]["sink"])
    show = sorted({0, nl // 4, nl // 2, 3 * nl // 4, nl - 1})
    for m in METRICS:
        print(f"\n{m:14s}" + "".join(f"  L{l:<5d}" for l in show) + "  mean")
        for name, r in res.items():
            v = r["layers"][m]
            print(f"  {name[:22]:22s}" + "".join(f"  {v[l]:.3f}" for l in show) + f"  {np.nanmean(v):.3f}")
    for name, r in res.items():
        for d in r["examples"][:1]:
            print(f"\n=== {name}: example doc ({len(d['claims'])} claims)")
            for q in d["queries"]:
                if q["kept"]:
                    print(f"  q{q['query']:<2d} ptr {q['pointer']:3d} [{q['pointer_word']!s:>10}]  top: {' '.join(q['top_words'])}")
                    print(f"        matched: {str(q['matched_claim'])[:100]}")


if __name__ == "__main__":
    main()
