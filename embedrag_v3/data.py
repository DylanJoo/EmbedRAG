"""Collator: documents -> student prompt inputs + claim-step masks; gold claims -> teacher inputs.

Same jsonl as v2 (embedrag/build_data.py):
    {"id": "...", "text": "...", "claims": ["claim 1", "claim 2", ...]}   # claims in document order

The student reads doc_template with the doc filled in (left-padded; only the doc is truncated); then [CLM]
steps follow it: step t (t <= n) is trained to embed claim t, and steps n+1 .. n+1+extra_stop_steps to say
stop (capped at max_claims + 1 steps). Claims (and docs, for the baseline) are tokenized for the teacher, the
way the embedding model was trained. Per-step tensors are flat, in row-major order of step_mask.
"""
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from .config import DOC_TEMPLATE


def pad(seqs, value, left=False):
    n = max([len(s) for s in seqs] + [1])
    return torch.tensor([([value] * (n - len(s)) + s) if left else (s + [value] * (n - len(s))) for s in seqs])


def encode(tok, text, max_len):
    """Tokenize the way the embedding model was trained (with its end token), keeping the end token."""
    ids = tok(text)["input_ids"]
    return ids if len(ids) <= max_len else ids[: max_len - 1] + ids[-1:]


@dataclass
class ClaimCollator:
    tokenizer: Any          # student
    teacher_tokenizer: Any
    doc_template: str = DOC_TEMPLATE
    max_claims: int = 32
    max_doc_len: int = 1024
    max_claim_len: int = 64
    extra_stop_steps: int = 0

    def __post_init__(self):
        before, after = self.doc_template.split("{doc}")
        self._before, self._after = (self.tokenizer(s, add_special_tokens=False)["input_ids"] for s in (before, after))

    def docs(self, texts):
        """Student prompts: template around each doc, the doc truncated to max_doc_len tokens."""
        seqs = [self._before + self.tokenizer(t, add_special_tokens=False)["input_ids"][: self.max_doc_len]
                + self._after for t in texts]
        return {"doc_input_ids": pad(seqs, self.tokenizer.pad_token_id, left=True),
                "doc_attention_mask": pad([[1] * len(s) for s in seqs], 0, left=True)}

    def teacher_texts(self, texts, max_len, prefix):
        seqs = [encode(self.teacher_tokenizer, t, max_len) for t in texts]
        return {f"{prefix}input_ids": pad(seqs, self.teacher_tokenizer.pad_token_id),
                f"{prefix}attention_mask": pad([[1] * len(s) for s in seqs], 0)}

    def claims(self, texts):
        return self.teacher_texts(texts, self.max_claim_len, "claim_")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        claims, n_steps, slot_claim, stop = [], [], [], []
        for ex in features:
            cs, base = ex["claims"][: self.max_claims], len(claims)
            fits = len(ex["claims"]) <= self.max_claims  # a truncated list has no true stop step
            n_stop = min(1 + self.extra_stop_steps, self.max_claims + 1 - len(cs)) if fits else 0
            claims += cs
            n_steps.append(len(cs) + n_stop)
            slot_claim += [base + t for t in range(len(cs))] + [-1] * n_stop
            stop += [0.0] * len(cs) + [1.0] * n_stop
        return {
            **self.docs([ex["text"] for ex in features]),
            "step_mask": pad([[True] * n for n in n_steps], False),
            **self.claims(claims),
            "slot_claim": torch.tensor(slot_claim),  # gold claim index per step, -1 = stop step
            "stop_labels": torch.tensor(stop),
        }
