"""Claim-embedding fidelity on held-out documents (pool = every gold claim of the eval docs, teacher vectors).

  kept_queries  the queries the has-claim head keeps (the real system); all_queries  every grid query
  for each: gold claims Hungarian-matched to the doc's vectors -> match_cos, match_R@1 (the matched claim is the
  vector's top-1 in the pool), matched_frac; and over all vectors: count_mae, n_pred, coverage (per gold claim, best
  cos over its doc's vectors), claim_hit@1 (gold claim is some vector's top-1), precision@1 (a vector's top-1 is from
  its own doc), distinct@1 (distinct top-1 claims / vectors)
  doc_vector baseline: the doc vector alone, scored the same way

  python -m claimemb.evaluate --ckpt ckpt/claimemb --eval_file data/claimemb/rag25.dev.jsonl --num_docs 300
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .data import ClaimCollator
from .model import ClaimEmb


def batched(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def match_metrics(vecs, pool, ranges):
    """Hungarian assignment of each doc's gold claims to its vectors (one-to-one, max total cos)."""
    cos, r1, n_matched = [], [], 0
    for v, (a, b) in zip(vecs, ranges):
        sim = (pool[a:b] @ v.T).float().cpu().numpy()                    # (claims, vectors)
        rows, cols = linear_sum_assignment(-sim)
        n_matched += len(rows)
        cos += sim[rows, cols].tolist()
        r1 += ((v[cols] @ pool.T).argmax(-1).cpu().numpy() == a + rows).astype(float).tolist()
    return {"match_cos": np.mean(cos), "match_R@1": np.mean(r1), "matched_frac": n_matched / int(ranges[-1][1])}


def free_run_metrics(vecs, pool, owner, ranges):
    """vecs[d]: (n_d, H) vectors for doc d; pool: (C, H) gold claims, owner[c] = doc of claim c."""
    cover, hit, prec, distinct = [], [], [], []
    for d, (v, (a, b)) in enumerate(zip(vecs, ranges)):
        top1 = (v @ pool.T).argmax(-1)
        cover += (v @ pool[a:b].T).max(0).values.tolist()
        hit += [float(c in set(top1.tolist())) for c in range(a, b)]
        prec += (owner[top1] == d).float().tolist()
        distinct.append(len(set(top1.tolist())) / len(v))
    return {"coverage": np.mean(cover), "claim_hit@1": np.mean(hit), "precision@1": np.mean(prec),
            "distinct@1": np.mean(distinct)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--num_docs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClaimEmb.from_pretrained(args.ckpt, torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    collator = ClaimCollator(model.tokenizer, model.cfg.max_claims)
    docs = [json.loads(l) for _, l in zip(range(args.num_docs), open(args.eval_file))]
    claims = [c for d in docs for c in d["claims"]]
    owner = torch.tensor([i for i, d in enumerate(docs) for _ in d["claims"]], device=dev)
    bounds = np.cumsum([0] + [len(d["claims"]) for d in docs])
    ranges = list(zip(bounds[:-1], bounds[1:]))
    pool = torch.cat([model.teacher(**{k.replace("claim_", ""): v.to(dev) for k, v in collator.claims(b).items()})
                      for b in batched(claims, 64)])

    doc_vecs, kept, every = [], [], []
    for b in batched([d["text"] for d in docs], args.batch_size):
        inp = {k: v.to(dev) for k, v in collator.docs(b).items()}
        dv, vecs, _ = model.generate(**inp)
        doc_vecs += list(dv.unsqueeze(1)); kept += vecs
        every += model.generate(**inp, all_queries=True)[1]
    n_gold = np.diff(bounds)
    count = lambda vs: {"count_mae": np.mean(np.abs(np.array([len(v) for v in vs]) - n_gold)),
                        "n_pred": np.mean([len(v) for v in vs]), "n_gold": np.mean(n_gold)}
    res = {"kept_queries": {**match_metrics(kept, pool, ranges), **count(kept), **free_run_metrics(kept, pool, owner, ranges)},
           "all_queries": {**match_metrics(every, pool, ranges), **count(every), **free_run_metrics(every, pool, owner, ranges)},
           "doc_vector": free_run_metrics(doc_vecs, pool, owner, ranges)}
    res = {k: {m: round(float(x), 4) for m, x in v.items()} for k, v in res.items()}
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
    if os.environ.get("WANDB_RUN_ID"):  # attach to the training run's summary (set by the slurm script)
        import wandb
        wandb.init(id=os.environ["WANDB_RUN_ID"], resume="allow")
        wandb.run.summary.update({f"fidelity/{k}/{m}": x for k, v in res.items() for m, x in v.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
