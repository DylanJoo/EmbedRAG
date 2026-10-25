"""Collator: documents -> the embedding model's own input; gold claims -> teacher inputs.

One jsonl line = one document with its claims (embedrag/build_data.py, embedrag_v3/build_rag25.py):
    {"id": "...", "text": "...", "claims": ["claim 1", "claim 2", ...]}

The doc input is "<doc tokens><|endoftext|>" (left-padded), so the state at the end token is the doc vector.
"""
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


def pad(seqs, value, left=False):
    n = max([len(s) for s in seqs] + [1])
    return torch.tensor([([value] * (n - len(s)) + s) if left else (s + [value] * (n - len(s))) for s in seqs])


def encode(tok, text, max_len):
    """Tokenize the way the embedding model was trained (with its end token), keeping the end token."""
    ids = tok(text)["input_ids"]
    return ids if len(ids) <= max_len else ids[: max_len - 1] + ids[-1:]


def end_id(tok):
    """The token the embedding model appends and pools at (<|endoftext|> for Qwen3-Embedding)."""
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
        claims = [ex["claims"][: self.max_claims] for ex in features]
        return {
            **self.docs([ex["text"] for ex in features]),
            **self.claims([c for cs in claims for c in cs]),
            "n_claims": torch.tensor([len(cs) for cs in claims]),
        }
