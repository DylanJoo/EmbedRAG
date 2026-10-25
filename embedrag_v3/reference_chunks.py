"""Training-free reference: embed each document's own pieces with the frozen teacher and score them like ClaimEmb.

Pieces per document (the first max_doc_len teacher tokens, as the model sees):
  sentences        split on . ! ? followed by whitespace; each embedded alone, as the teacher embeds a claim
                   (raw text + its end token <|endoftext|>, pooled at that end token)
  windows          fixed token windows (--window tokens, every --window_stride tokens), embedded like sentences
Context x end-token ablation for sentences (pooled hidden state, normalized):
  sentences_noeos  the sentence alone, pooled at its own last token (no end token)
  in_context       one pass over the whole doc, pooled at each sentence's last token (sees only the doc prefix)
  in_context_eos   the doc prefix up to each sentence + <|endoftext|>, pooled at that end token
One doc pass, then one <|endoftext|> per sentence appended to it (each sees only what its mask allows, not the others):
  pos_eos          the end token at the position right after its sentence, seeing the whole doc
  mask_eos         the end token at the position right after its sentence, seeing only its sentence (+ the first
                   token, the attention sink); the sentence's token states still carry earlier context

Reported with the same definitions as evaluate.py (pool = all gold claims of the eval docs):
  match (instead of teacher-forced): Hungarian one-to-one assignment of a doc's gold claims to its pieces
    match_cos       mean cos of matched (claim, piece) pairs
    match_R@1       a matched piece's top-1 in the pool is its assigned claim
    matched_frac    share of gold claims that got a piece (< 1 when a doc has fewer pieces than claims)
  free_run (all pieces = the generated vectors): count_mae, n_pred, coverage, claim_hit@1, precision@1, distinct@1
  More pieces mean more chances to hit a claim: compare at similar n_pred.

  python -m embedrag_v3.reference_chunks --teacher Qwen/Qwen3-Embedding-0.6B \\
      --eval_file data/claimemb/n30k.dev.jsonl --num_docs 300 --output scratch/claimemb3-n30k/reference_chunks.json
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel, AutoTokenizer

from .config import QWEN3_EMB
from .data import encode, pad
from .evaluate import batched, free_run_metrics
from .model import last_token


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def sentence_ids(tok, text, max_doc_len):
    """Doc -> per-sentence token ids (sentence i > 0 with its leading space), cut at max_doc_len tokens in total."""
    out, n = [], 0
    for i, s in enumerate(sentences(text)):
        ids = tok((" " if i else "") + s, add_special_tokens=False)["input_ids"][: max_doc_len - n]
        if not ids:
            break
        out.append(ids)
        n += len(ids)
    return out


def end_id(tok):
    """The token the embedding model appends and pools at (<|endoftext|> for Qwen3-Embedding)."""
    return (tok("")["input_ids"] or [tok.eos_token_id])[-1]


@torch.no_grad()
def pool_at(model, seqs, at, pad_id, bsz=32):
    """Right-padded token id lists -> normalized hidden states at index at[i] of each sequence."""
    dev, out = next(model.parameters()).device, []
    for i in range(0, len(seqs), bsz):
        b = seqs[i:i + bsz]
        ids = pad(b, pad_id).to(dev)
        mask = pad([[1] * len(s) for s in b], 0).to(dev)
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        idx = torch.tensor(at[i:i + bsz], device=dev)
        out.append(F.normalize(h[torch.arange(len(b), device=dev), idx].float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def appended_eos(model, doc, spans, eos, only_span):
    """One pass over doc, then one end token per (start, end) span at position end + 1, all appended in one call.

    Each end token attends to its span (+ token 0) if only_span, else to the whole doc; never to the other end tokens.
    """
    dev = next(model.parameters()).device
    out = model(input_ids=torch.tensor([doc], device=dev), use_cache=True)
    L, S = len(doc), len(spans)
    keep = torch.zeros(S, L + S, dtype=torch.bool, device=dev)
    for i, (s, e) in enumerate(spans):
        if only_span:
            keep[i, 0] = True
            keep[i, s:e + 1] = True
        else:
            keep[i, :L] = True
        keep[i, L + i] = True
    dtype = next(model.parameters()).dtype
    mask = torch.zeros(1, 1, S, L + S, dtype=dtype, device=dev).masked_fill(~keep, torch.finfo(dtype).min)
    h = model(input_ids=torch.full((1, S), eos, device=dev), attention_mask=mask,
              position_ids=torch.tensor([[e + 1 for _, e in spans]], device=dev),
              past_key_values=out.past_key_values, use_cache=True).last_hidden_state[0]
    return F.normalize(h.float(), dim=-1)


def ablation_vectors(model, tok, text, mode, max_doc_len, max_piece_len):
    """Context x end-token variants of sentence vectors for one doc -> (n_sentences, H)."""
    sents = sentence_ids(tok, text, max_doc_len)
    eos = end_id(tok)
    if mode == "sentences_noeos":
        seqs = [s[:max_piece_len] for s in sents]
        return pool_at(model, seqs, [len(s) - 1 for s in seqs], tok.pad_token_id)
    ends = np.cumsum([len(s) for s in sents]) - 1           # last token of each sentence in the doc
    doc = [t for s in sents for t in s]
    if mode in ("pos_eos", "mask_eos"):
        spans = [(int(e) - len(s) + 1, int(e)) for s, e in zip(sents, ends)]
        return appended_eos(model, doc, spans, eos, only_span=mode == "mask_eos")
    if mode == "in_context":                                   # one pass; causal, so each end sees its prefix
        dev = next(model.parameters()).device
        h = model(input_ids=torch.tensor([doc], device=dev)).last_hidden_state[0]
        return F.normalize(h[torch.tensor(ends, device=dev)].float(), dim=-1)
    seqs = [doc[: e + 1] + [eos] for e in ends]              # in_context_eos: prefix + end token
    return pool_at(model, seqs, [len(s) - 1 for s in seqs], tok.pad_token_id, bsz=8)


def windows(ids, size, stride):
    """Token windows covering ids; the last window ends at the last token."""
    if len(ids) <= size:
        return [ids]
    starts = list(range(0, len(ids) - size, stride)) + [len(ids) - size]
    return [ids[s:s + size] for s in starts]


def pieces(tok, text, mode, max_doc_len, size, stride):
    """Doc (first max_doc_len teacher tokens) -> list of piece texts."""
    ids = tok(text, add_special_tokens=False)["input_ids"][:max_doc_len]
    doc = tok.decode(ids)
    if mode == "sentences":
        return sentences(doc) or [doc]
    return [tok.decode(w) for w in windows(ids, size, stride)]


@torch.no_grad()
def embed(model, tok, texts, max_len, bsz=64):
    """Frozen embedding LM on texts (with its end token), last-token pooled + normalized."""
    dev, out = next(model.parameters()).device, []
    for b in batched(texts, bsz):
        seqs = [encode(tok, t, max_len) for t in b]
        ids = pad(seqs, tok.pad_token_id).to(dev)
        mask = pad([[1] * len(s) for s in seqs], 0).to(dev)
        out.append(F.normalize(last_token(model(input_ids=ids, attention_mask=mask).last_hidden_state, mask).float(), dim=-1))
    return torch.cat(out)


def match_metrics(vecs, pool, ranges):
    """Hungarian assignment of each doc's gold claims to its piece vectors (one-to-one, max total cos)."""
    cos, r1, n_matched = [], [], 0
    for v, (a, b) in zip(vecs, ranges):
        sim = (pool[a:b] @ v.T).float().cpu().numpy()          # (claims, pieces)
        rows, cols = linear_sum_assignment(-sim)
        n_matched += len(rows)
        cos += sim[rows, cols].tolist()
        top1 = (v[cols] @ pool.T).argmax(-1).cpu().numpy()
        r1 += (top1 == a + rows).astype(float).tolist()
    return {"match_cos": np.mean(cos), "match_R@1": np.mean(r1), "matched_frac": n_matched / int(ranges[-1][1])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=QWEN3_EMB)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--num_docs", type=int, default=300)
    p.add_argument("--max_doc_len", type=int, default=1024)
    p.add_argument("--max_claim_len", type=int, default=64)
    p.add_argument("--max_piece_len", type=int, default=256, help="longer sentences are cut (keeping the end token)")
    p.add_argument("--modes", nargs="+", default=["sentences", "windows"],
                   choices=["sentences", "windows", "sentences_noeos", "in_context", "in_context_eos", "pos_eos",
                            "mask_eos"])
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--window_stride", type=int, default=16)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModel.from_pretrained(args.teacher, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                      attn_implementation="sdpa").to(dev).eval()

    with open(args.eval_file) as f:
        docs = [json.loads(l) for _, l in zip(range(args.num_docs), f)]
    claims = [c for d in docs for c in d["claims"]]
    owner = torch.tensor([i for i, d in enumerate(docs) for _ in d["claims"]], device=dev)
    bounds = np.cumsum([0] + [len(d["claims"]) for d in docs])
    ranges = list(zip(bounds[:-1], bounds[1:]))
    pool = embed(model, tok, claims, args.max_claim_len)
    print(f"{len(docs)} docs, {len(claims)} pool claims", flush=True)

    res = {}
    for mode in args.modes:
        if mode in ("sentences", "windows"):
            ps = [pieces(tok, d["text"], mode, args.max_doc_len, args.window, args.window_stride) for d in docs]
            flat = embed(model, tok, [x for p_ in ps for x in p_], args.max_piece_len)
            vecs = list(torch.split(flat, [len(p_) for p_ in ps]))
        else:
            with torch.no_grad():
                vecs = [ablation_vectors(model, tok, d["text"], mode, args.max_doc_len, args.max_piece_len)
                        for d in docs]
        n_gold = np.diff(bounds)
        res[mode] = {
            "match": match_metrics(vecs, pool, ranges),
            "free_run": {"count_mae": np.mean(np.abs(np.array([len(v) for v in vecs]) - n_gold)),
                         "n_pred": np.mean([len(v) for v in vecs]), "n_gold": np.mean(n_gold),
                         **free_run_metrics(vecs, pool, owner, ranges)},
        }
    res = {m: {k: {x: round(float(y), 4) for x, y in v.items()} for k, v in r.items()} for m, r in res.items()}
    res["settings"] = {"window": args.window, "window_stride": args.window_stride, "num_docs": len(docs)}
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
