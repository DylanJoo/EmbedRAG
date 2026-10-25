"""Collator: documents (+ claim regions from align.py) -> model inputs; gold claims -> teacher inputs.

One jsonl line = one document with its claims and their aligned token spans:
    {"id", "text", "claims": [...], "spans": [[s, e], ...]}   # s, e: doc token positions (inclusive)
"spans" is optional (query=grid doesn't use regions): without it every claim gets (0, 0) and all tags are O.

The doc input is the embedding model's own: "<doc tokens><|endoftext|>" (left-padded), so the state at the end token is
the doc vector. Spans index the doc tokens (= positions in the unpadded input). Per batch:
  claim_spans (B, T, 2)  each doc's claim regions (padding: 0, 0), with claim_mask (B, T)
  tags        (B, L)     B/I/O tags of the doc tokens (overlaps trimmed, see crf.spans_to_tags), right-padded,
                         with tag_mask (B, L); the end token is not tagged
"""
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from .crf import spans_to_tags


def pad(seqs, value, left=False):
    n = max([len(s) for s in seqs] + [1])
    return torch.tensor([([value] * (n - len(s)) + s) if left else (s + [value] * (n - len(s))) for s in seqs])


def encode(tok, text, max_len):
    """Tokenize the way the embedding model was trained (with its end token), keeping the end token."""
    ids = tok(text)["input_ids"]
    return ids if len(ids) <= max_len else ids[: max_len - 1] + ids[-1:]


def end_id(tok):
    return (tok("")["input_ids"] or [tok.eos_token_id])[-1]


@dataclass
class ClaimCollator:
    tokenizer: Any
    max_claims: int = 32
    max_doc_len: int = 1024
    max_claim_len: int = 64

    def docs(self, texts):
        eos = end_id(self.tokenizer)
        seqs = [self.tokenizer(t, add_special_tokens=False)["input_ids"][: self.max_doc_len] + [eos] for t in texts]
        return {"doc_input_ids": pad(seqs, self.tokenizer.pad_token_id, left=True),
                "doc_attention_mask": pad([[1] * len(s) for s in seqs], 0, left=True)}

    def claims(self, texts):
        seqs = [encode(self.tokenizer, t, self.max_claim_len) for t in texts]
        return {"claim_input_ids": pad(seqs, self.tokenizer.pad_token_id),
                "claim_attention_mask": pad([[1] * len(s) for s in seqs], 0)}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        docs = self.docs([ex["text"] for ex in features])
        n_doc = (docs["doc_attention_mask"].sum(1) - 1).tolist()      # doc tokens, without the end token
        claims, spans, tags = [], [], []
        for ex, n in zip(features, n_doc):
            k = min(len(ex["claims"]), self.max_claims)
            sp = [(min(s, n - 1), min(e, n - 1)) for s, e in ex.get("spans", [(0, 0)] * k)[:k]]
            claims += ex["claims"][:k]
            spans.append(sp)
            tags.append(spans_to_tags(sp, n) if "spans" in ex else [0] * n)
        T = max(len(s) for s in spans)
        return {
            **docs,
            **self.claims(claims),
            "claim_spans": torch.tensor([[list(x) for x in s] + [[0, 0]] * (T - len(s)) for s in spans]),
            "claim_mask": pad([[True] * len(s) for s in spans], False),
            "tags": pad(tags, 0),
            "tag_mask": pad([[True] * len(t) for t in tags], False),
        }
