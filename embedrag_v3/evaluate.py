"""Claim-embedding fidelity on held-out documents (no queries / qrels needed).

The pool is every gold claim of the eval documents, embedded by the frozen teacher. Reported:

  teacher-forced (n gold steps):
    step_cos      cos(e_t, gold claim t)
    step_R@1/10   gold claim t is in the top-1/10 of the pool for e_t
    match_cos/R@1 the same after one-to-one Hungarian matching of steps to the doc's claims (order-free)
  free-running (stop head decides n):
    count_mae     |n_pred - n_gold|
    coverage      per gold claim, max cos over its document's generated vectors
    claim_hit@1   gold claim is top-1 in the pool for at least one of its document's vectors
    precision@1   generated vector's top-1 pool claim belongs to its own document
    distinct@1    distinct top-1 claims / n_pred (low = steps repeat the same claim)
  doc_vector baseline: the frozen teacher's single document vector, scored the same way.

  python -m embedrag_v3.evaluate --ckpt ckpt/claimemb_v3 --eval_file data/claimemb/eng.dev.jsonl
"""
import argparse
import json
import os

import numpy as np
import torch

from .config import QWEN3, QWEN3_EMB, ClaimEmbConfig
from .data import ClaimCollator
from .model import ClaimEmb, hungarian_targets


def batched(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


@torch.no_grad()
def embed_claims(model, collator, texts, bsz=64):
    dev = model.step_query.device
    out = [model.teacher(**{k.replace("claim_", ""): v.to(dev) for k, v in collator.claims(b).items()})
           for b in batched(texts, bsz)]
    return torch.cat(out)


@torch.no_grad()
def embed_docs(model, collator, texts, bsz):
    """The teacher's doc vectors (= the docs_emb index), for the baseline."""
    dev = model.step_query.device
    return torch.cat([model.teacher(**{k: v.to(dev) for k, v in collator.teacher_texts(b, collator.max_doc_len, "").items()})
                      for b in batched(texts, bsz)])


def free_run_metrics(vecs, pool, pool_owner, gold_ranges):
    """vecs[d]: (n_d, H) vectors for doc d; pool: (C, H) gold claims, pool_owner[c] = doc of claim c."""
    cover, hit, prec, distinct = [], [], [], []
    for d, (v, (a, b)) in enumerate(zip(vecs, gold_ranges)):
        top1 = (v @ pool.T).argmax(-1)
        cover += (v @ pool[a:b].T).max(0).values.tolist()
        hit += [float(c in set(top1.tolist())) for c in range(a, b)]
        prec += (pool_owner[top1] == d).float().tolist()
        distinct.append(len(set(top1.tolist())) / len(v))
    return {"coverage": np.mean(cover), "claim_hit@1": np.mean(hit), "precision@1": np.mean(prec),
            "distinct@1": np.mean(distinct)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="ClaimEmb checkpoint; omit to report only the baseline")
    p.add_argument("--student", default=QWEN3)
    p.add_argument("--teacher", default=QWEN3_EMB)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--num_docs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_doc_len", type=int, default=1024)
    p.add_argument("--output", default=None, help="write the metrics json here")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = ClaimEmb.from_pretrained(args.ckpt, dtype) if args.ckpt \
        else ClaimEmb(ClaimEmbConfig(args.student, args.teacher), dtype)
    model = model.to(dev).eval()
    collator = ClaimCollator(model.tokenizer, model.teacher_tokenizer, model.cfg.doc_template, model.cfg.max_claims,
                             args.max_doc_len)

    docs = [json.loads(l) for l in open(args.eval_file) if l.strip()][: args.num_docs]
    claims = [c for d in docs for c in d["claims"]]
    owner = torch.tensor([i for i, d in enumerate(docs) for _ in d["claims"]], device=dev)
    bounds = np.cumsum([0] + [len(d["claims"]) for d in docs])
    ranges = list(zip(bounds[:-1], bounds[1:]))
    pool = embed_claims(model, collator, claims)
    print(f"{len(docs)} docs, {len(claims)} pool claims", flush=True)

    res = {"doc_vector": free_run_metrics(list(embed_docs(model, collator, [d["text"] for d in docs],
                                                           args.batch_size).unsqueeze(1)),
                                          pool, owner, ranges)}
    if args.ckpt:
        # teacher-forced: exactly the n gold claim steps
        cos, r1, r10, mcos, mr1 = [], [], [], [], []
        for idx in batched(list(range(len(docs))), args.batch_size):
            b = collator([docs[i] for i in idx])
            b = {k: v.to(dev) for k, v in b.items()}
            cache = model.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
            e, _ = model.steps(cache, b["doc_attention_mask"], b["step_mask"])
            e, y = e[b["slot_claim"] >= 0], b["slot_claim"][b["slot_claim"] >= 0] + int(ranges[idx[0]][0])
            cos += (e * pool[y]).sum(-1).tolist()
            rank = ((e @ pool.T) > (e * pool[y]).sum(-1, keepdim=True)).sum(-1)
            r1 += (rank < 1).float().tolist()
            r10 += (rank < 10).float().tolist()
            doc = b["step_mask"].nonzero()[:, 0][b["slot_claim"] >= 0]
            ym = hungarian_targets(e, pool, y, doc)
            mcos += (e * pool[ym]).sum(-1).tolist()
            mr1 += ((e @ pool.T).argmax(-1) == ym).float().tolist()
        res["teacher_forced"] = {"step_cos": np.mean(cos), "step_R@1": np.mean(r1), "step_R@10": np.mean(r10),
                                 "match_cos": np.mean(mcos), "match_R@1": np.mean(mr1)}

        vecs = [v for b in batched([d["text"] for d in docs], args.batch_size)
                for v in model.generate(**{k: x.to(dev) for k, x in collator.docs(b).items()})]
        n_gold = np.diff(bounds)
        res["free_run"] = {"count_mae": np.mean(np.abs(np.array([len(v) for v in vecs]) - n_gold)),
                           "n_pred": np.mean([len(v) for v in vecs]), "n_gold": np.mean(n_gold),
                           **free_run_metrics(vecs, pool, owner, ranges)}

    res = {k: {m: round(float(x), 4) for m, x in v.items()} for k, v in res.items()}
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
    if os.environ.get("WANDB_RUN_ID"):  # attach to the training run's summary (set by the slurm script)
        import wandb
        wandb.init(id=os.environ["WANDB_RUN_ID"], resume="allow")
        wandb.run.summary.update({f"fidelity/{k}/{m}": x for k, v in res.items() for m, x in v.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
