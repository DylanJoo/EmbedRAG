"""Build train/dev jsonl of {"id", "text", "claims": [...]} from the TREC RAG 2025 passages (data/rag25/).

Inputs (copied from ~/ClaimEmbeddings, see data/rag25/README.md):
  rag25_top100_corpus.jsonl              {"docid", "title", "text", ...}: MS MARCO v2.1 segments, top-100 for 105 queries
  rag25_claims.Qwen3-32B.limitall.jsonl  {"docid", "claims", "finish_reason"}: atomic, decontextualized claims
The title is prepended to the text ("<title>\\n<text>") since the claims were extracted with it (to resolve
references). Filters and split as embedrag.build_data: <= max_doc_len teacher tokens, 1..max_claims claims, dev by
doc-id hash.

  python -m embedrag_v3.build_rag25 --output data/claimemb/rag25 --tokenizer <Qwen3-Embedding path>
  -> data/claimemb/rag25.train.jsonl, data/claimemb/rag25.dev.jsonl
"""
import argparse
import json
import os

from transformers import AutoTokenizer

from embedrag.build_data import is_dev

from .config import QWEN3_EMB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/rag25/rag25_top100_corpus.jsonl")
    p.add_argument("--claims", default="data/rag25/rag25_claims.Qwen3-32B.limitall.jsonl")
    p.add_argument("--output", default="data/claimemb/rag25")
    p.add_argument("--tokenizer", default=QWEN3_EMB)
    p.add_argument("--dev_frac", type=float, default=0.1)
    p.add_argument("--max_doc_len", type=int, default=1024)
    p.add_argument("--max_claims", type=int, default=32)
    args = p.parse_args()

    corpus = {d["docid"]: d for d in map(json.loads, open(args.corpus))}
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    outs = {s: open(f"{args.output}.{s}.jsonl", "w") for s in ("train", "dev")}
    n = {"train": 0, "dev": 0, "skipped": 0}
    for r in map(json.loads, open(args.claims)):
        doc = corpus[r["docid"]]
        text = f"{doc['title']}\n{doc['text']}" if doc.get("title") else doc["text"]
        claims = [c.strip() for c in r["claims"] if c.strip()]
        if r["finish_reason"] != "stop" or len(tok(text)["input_ids"]) > args.max_doc_len \
                or not 1 <= len(claims) <= args.max_claims:
            n["skipped"] += 1
            continue
        split = "dev" if is_dev(r["docid"], args.dev_frac) else "train"
        outs[split].write(json.dumps({"id": r["docid"], "text": text, "claims": claims}) + "\n")
        n[split] += 1
    for f in outs.values():
        f.close()
    print(f"wrote {n['train']} train / {n['dev']} dev docs ({n['skipped']} skipped) -> {args.output}.*.jsonl")


if __name__ == "__main__":
    main()
