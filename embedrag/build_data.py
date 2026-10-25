"""Build train/dev jsonl of {"id", "text", "claims": [...]} from RAGTIME processed-claims files.

Keeps documents that fit in `max_doc_len` backbone tokens and have 1..max_claims claims; claims stay
in document order. The split is by a hash of the doc id, so it is stable across runs and corpora.

  python -m embedrag.build_data --input ~/scratch/ragtime2/eng-docs.processed-claims.jsonl.gz \
      --output data/claimemb/eng --num_docs 200000
  -> data/claimemb/eng.train.jsonl, data/claimemb/eng.dev.jsonl
"""
import argparse
import ast
import gzip
import hashlib
import json
import os

from transformers import AutoTokenizer

from .config import QWEN3_EMB


def claims_of(row):
    s = row["statements"]
    return ast.literal_eval(s) if isinstance(s, str) else s  # stored as a stringified python list


def is_dev(doc_id, dev_frac):
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 10_000 < dev_frac * 10_000


def batches(path, size):
    rows = []
    with gzip.open(os.path.expanduser(path), "rt") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == size:
                yield rows
                rows = []
    if rows:
        yield rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="prefix; writes <prefix>.train.jsonl and <prefix>.dev.jsonl")
    p.add_argument("--tokenizer", default=QWEN3_EMB)
    p.add_argument("--num_docs", type=int, default=200_000, help="kept docs (train + dev)")
    p.add_argument("--dev_frac", type=float, default=0.005)
    p.add_argument("--max_doc_len", type=int, default=1024)
    p.add_argument("--max_claims", type=int, default=32)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    outs = {s: open(f"{args.output}.{s}.jsonl", "w") for s in ("train", "dev")}
    n = {"train": 0, "dev": 0, "skipped": 0}
    for rows in batches(args.input, 1000):
        lens = [len(x) for x in tok([r["text"] for r in rows])["input_ids"]]
        for r, length in zip(rows, lens):
            claims = [c.strip() for c in claims_of(r) if c.strip()]
            if length > args.max_doc_len or not 1 <= len(claims) <= args.max_claims:
                n["skipped"] += 1
                continue
            split = "dev" if is_dev(r["id"], args.dev_frac) else "train"
            outs[split].write(json.dumps({"id": r["id"], "text": r["text"], "claims": claims}) + "\n")
            n[split] += 1
        if n["train"] + n["dev"] >= args.num_docs:
            break
    for f in outs.values():
        f.close()
    print(f"wrote {n['train']} train / {n['dev']} dev docs ({n['skipped']} skipped) -> {args.output}.*.jsonl")


if __name__ == "__main__":
    main()
