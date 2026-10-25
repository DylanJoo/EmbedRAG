"""Claim-embedding fidelity of ClaimEmb v4 on held-out documents (pool = all gold claims, teacher vectors).

  oracle regions (the aligned gold spans; how good the queries are with a perfect locator):
    match_cos / match_R@1   cos(e_k, gold claim k) / gold claim k is e_k's top-1 in the pool
  predicted regions (CRF): count_mae, n_pred, coverage, claim_hit@1, precision@1, distinct@1 (as v2/v3), and
    region P/R/F1   predicted spans vs the trimmed gold spans the CRF is trained on (exact match)
  untrained reference (--ckpt omitted): the same with an untrained model (query mask/residual as given)
  grid: "all_queries" = every grid query, gold claims Hungarian-matched to them (match_cos / match_R@1 /
    matched_frac, as reference_chunks); "kept_queries" = the queries the has-claim head keeps (the real system)

  python -m embedrag_v4.evaluate --ckpt ckpt/v4 --eval_file data/claimemb/rag25.dev.regions.jsonl --num_docs 300
"""
import argparse
import json
import os

import numpy as np
import torch

from embedrag_v3.evaluate import batched, free_run_metrics
from embedrag_v3.reference_chunks import match_metrics

from .config import QWEN3_EMB, ClaimEmbConfig
from .crf import spans_to_tags, tags_to_spans
from .data import ClaimCollator
from .model import ClaimEmb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--backbone", default=QWEN3_EMB)
    p.add_argument("--query", default="mask", help="without --ckpt")
    p.add_argument("--eval_file", required=True, help="*.regions.jsonl")
    p.add_argument("--num_docs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = ClaimEmb.from_pretrained(args.ckpt, dtype) if args.ckpt else \
        ClaimEmb(ClaimEmbConfig(backbone=args.backbone, query=args.query), dtype)
    model = model.to(dev).eval()
    collator = ClaimCollator(model.tokenizer, model.cfg.max_claims)

    docs = [json.loads(l) for _, l in zip(range(args.num_docs), open(args.eval_file))]
    claims = [c for d in docs for c in d["claims"][: model.cfg.max_claims]]
    owner = torch.tensor([i for i, d in enumerate(docs) for _ in d["claims"][: model.cfg.max_claims]], device=dev)
    bounds = np.cumsum([0] + [min(len(d["claims"]), model.cfg.max_claims) for d in docs])
    ranges = list(zip(bounds[:-1], bounds[1:]))
    pool = torch.cat([model.teacher(**{k.replace("claim_", ""): v.to(dev) for k, v in collator.claims(b).items()})
                      for b in batched(claims, 64)])

    n_gold = np.diff(bounds)
    if model.cfg.query == "grid":
        all_vecs, kept = [], []
        for idx in batched(list(range(len(docs))), args.batch_size):
            b = {k: v.to(dev) for k, v in collator.docs([docs[i]["text"] for i in idx]).items()}
            with torch.no_grad():
                _, cache = model.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
                spans, gmask = model.grid_spans(b["doc_attention_mask"])
                e = model.queries(cache, b["doc_attention_mask"], spans, gmask)
            all_vecs += list(torch.split(e, gmask.sum(1).tolist()))
            kept += model.generate(b["doc_input_ids"], b["doc_attention_mask"])[1]
        cnt = lambda vs: {"count_mae": np.mean(np.abs(np.array([len(v) for v in vs]) - n_gold)),
                          "n_pred": np.mean([len(v) for v in vs]), "n_gold": np.mean(n_gold)}
        res = {"all_queries": {**match_metrics(all_vecs, pool, ranges), **cnt(all_vecs),
                               **free_run_metrics(all_vecs, pool, owner, ranges)},
               "kept_queries": {**match_metrics(kept, pool, ranges), **cnt(kept),
                                **free_run_metrics(kept, pool, owner, ranges)}}
        finish(res, args)
        return

    cos, r1, oracle_vecs, pred_vecs, tp, n_pred_sp, n_gold_sp = [], [], [], [], 0, 0, 0
    for idx in batched(list(range(len(docs))), args.batch_size):
        b = {k: v.to(dev) for k, v in collator([docs[i] for i in idx]).items()}
        with torch.no_grad():
            h, cache = model.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
            e = model.queries(cache, b["doc_attention_mask"], b["claim_spans"], b["claim_mask"])
        y = torch.arange(int(ranges[idx[0]][0]), int(ranges[idx[-1]][1]), device=dev)
        cos += (e * pool[y]).sum(-1).tolist()
        r1 += ((e @ pool.T).argmax(-1) == y).float().tolist()
        oracle_vecs += list(torch.split(e, b["claim_mask"].sum(1).tolist()))
        _, vecs, regions = model.generate(b["doc_input_ids"], b["doc_attention_mask"])
        pred_vecs += vecs
        for i, r in zip(idx, regions):   # region P/R/F1 against the trimmed gold spans (what B/I/O can express)
            n = len(model.tokenizer(docs[i]["text"], add_special_tokens=False)["input_ids"][: collator.max_doc_len])
            gold = set(tags_to_spans(spans_to_tags([tuple(s) for s in docs[i]["spans"]], n)))
            tp += len(gold & set(r)); n_pred_sp += len(r); n_gold_sp += len(gold)

    prec, rec = tp / max(n_pred_sp, 1), tp / max(n_gold_sp, 1)
    res = {
        "oracle_regions": {"match_cos": np.mean(cos), "match_R@1": np.mean(r1),
                           **free_run_metrics(oracle_vecs, pool, owner, ranges)},
        "predicted_regions": {"count_mae": np.mean(np.abs(np.array([len(v) for v in pred_vecs]) - n_gold)),
                              "n_pred": np.mean([len(v) for v in pred_vecs]), "n_gold": np.mean(n_gold),
                              "region_P": prec, "region_R": rec, "region_F1": 2 * prec * rec / max(prec + rec, 1e-9),
                              **free_run_metrics(pred_vecs, pool, owner, ranges)},
    }
    finish(res, args)


def finish(res, args):
    res = {k: {m: round(float(x), 4) for m, x in v.items()} for k, v in res.items()}
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
    if os.environ.get("WANDB_RUN_ID"):
        import wandb
        wandb.init(id=os.environ["WANDB_RUN_ID"], resume="allow")
        wandb.run.summary.update({f"fidelity/{k}/{m}": x for k, v in res.items() for m, x in v.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
