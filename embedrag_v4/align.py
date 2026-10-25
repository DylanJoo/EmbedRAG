"""Align each gold claim to a token span of its document (the claim's "region"), offline.

Tokens are the embedding model's tokens of the raw doc text (no special tokens), i.e. the positions the doc
occupies in its input "{doc}<|endoftext|>". A token "hits" a claim when the doc word it belongs to (lowercased
\\w+) is one of the claim's words. Each claim's span [s, e] maximizes a token-level F1:
    precision = hits in span / span length,   recall = min(1, hits in span / claim length in tokens)
over spans up to max_span tokens (ties: the shorter span).

  python -m embedrag_v4.align --input data/claimemb/rag25.train.jsonl --tokenizer <Qwen3-Embedding path>
  -> data/claimemb/rag25.train.regions.jsonl  (+ "spans": [[s, e], ...] per doc), and overlap statistics
"""
import argparse
import json
import os
import re

import numpy as np
from transformers import AutoTokenizer

WORD = re.compile(r"\w+")


def token_words(text, offsets):
    """Lowercased doc word each token overlaps (None for tokens without a word, e.g. punctuation/space)."""
    words = [(m.start(), m.end(), m.group().lower()) for m in WORD.finditer(text)]
    out, j = [], 0
    for a, b in offsets:
        while j < len(words) and words[j][1] <= a:
            j += 1
        out.append(words[j][2] if j < len(words) and words[j][0] < b else None)
    return out


def best_span(hits, claim_len, max_span):
    """hits (L,) 0/1 -> (s, e) maximizing token F1 (see module doc), or None without any hit."""
    if not hits.any():
        return None
    L = len(hits)
    csum = np.concatenate([[0], np.cumsum(hits)])
    best, arg = -1.0, None
    for w in range(1, min(max_span, L) + 1):
        h = csum[w:] - csum[:-w]                       # hits of spans [s, s + w - 1]
        p, r = h / w, np.minimum(1.0, h / claim_len)
        f1 = np.where(h > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)
        s = int(f1.argmax())
        if f1[s] > best + 1e-9:
            best, arg = float(f1[s]), (s, s + w - 1)
    return arg


def align_doc(tok, text, claims, max_doc_len, max_span):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    tw = token_words(text, enc["offset_mapping"][:max_doc_len])
    spans = []
    for c in claims:
        cw = set(WORD.findall(c.lower()))
        hits = np.array([w is not None and w in cw for w in tw], dtype=float)
        n = max(1, len(tok(c, add_special_tokens=False)["input_ids"]))
        spans.append(best_span(hits, n, max_span) or (0, 0))
    return spans


def overlap_stats(all_spans):
    pairs = over_pairs = claims = claims_over = 0
    for spans in all_spans:
        for i, (s, e) in enumerate(spans):
            o = [j for j, (s2, e2) in enumerate(spans) if j != i and s <= e2 and s2 <= e]
            claims += 1
            claims_over += bool(o)
            over_pairs += sum(j > i for j in o)
        pairs += len(spans) * (len(spans) - 1) // 2
    return {"claims": claims, "claims_overlapping_another": claims_over / max(claims, 1),
            "overlapping_pairs_in_doc": over_pairs / max(pairs, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="jsonl {id, text, claims}")
    p.add_argument("--output", default=None, help="default: <input without .jsonl>.regions.jsonl")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max_doc_len", type=int, default=1024)
    p.add_argument("--max_span", type=int, default=96)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = args.output or args.input.replace(".jsonl", ".regions.jsonl")
    all_spans = []
    tmp = f"{out}.tmp{os.getpid()}"  # renamed when complete: concurrent jobs never read or share a partial file
    with open(tmp, "w") as f:
        for line in open(args.input):
            d = json.loads(line)
            d["spans"] = [list(s) for s in align_doc(tok, d["text"], d["claims"], args.max_doc_len, args.max_span)]
            all_spans.append(d["spans"])
            f.write(json.dumps(d) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"output": out, **overlap_stats(all_spans)}, indent=2))


if __name__ == "__main__":
    main()
