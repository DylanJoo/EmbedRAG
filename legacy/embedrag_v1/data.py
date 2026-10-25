"""Unified data schema + collator.

One jsonl line = one example:
    {"task": "qa",                      # free-form tag (ae | cont | qa | summ | rel | single ...)
     "instruction": "Question: ...",     # may contain "{context}"; otherwise context is prepended
     "passages": ["...", "..."],         # each passage -> num_mem embeddings
     "target": "..." | [["[1] ", 2.0], ["summary ...", 1.0]],   # optional per-segment loss weights
     "meta": {...}}                      # optional, used by RL rewards / evaluation

The decoder sees passage i as "[i] <mem x k>", so outputs can cite passages as [i].
The teacher (for distillation) sees the same prompt with the passage text instead of <mem>.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .backbone import ChatFormat
from .model import MEM_ID

PLACEHOLDER = "<|ctx:{}|>"
PLACEHOLDER_RE = re.compile(r"<\|ctx:(\d+)\|>")


def user_text(example, items):
    ctx = "\n".join(f"[{i + 1}] {x}" for i, x in enumerate(items))
    instr = example["instruction"]
    if "{context}" in instr:
        return instr.replace("{context}", ctx)
    return f"Context:\n{ctx}\n\n{instr}" if items else instr


def target_segments(target):
    if target is None:
        return []
    if isinstance(target, str):
        return [(target, 1.0)]
    return [(t, float(w)) for t, w in target]


def pad(seqs, value, left=False):
    n = max([len(s) for s in seqs] + [1])
    rows = [([value] * (n - len(s)) + s) if left else (s + [value] * (n - len(s))) for s in seqs]
    return torch.tensor(rows)


@dataclass
class EmbedRAGCollator:
    tokenizer: Any                       # decoder tokenizer
    enc_tokenizer: Any                   # encoder tokenizer (same object if shared)
    chat: ChatFormat
    num_mem: int = 8
    max_passage_len: int = 256
    max_target_len: int = 512
    passage_prefix: str = ""             # optional encoder-side instruction, e.g. "Passage: "
    with_teacher: bool = False           # also build full-text teacher inputs (distillation)
    for_generation: bool = False         # no targets, left-padded decoder inputs

    def _ids(self, tok, text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def _passage(self, text):
        tok = self.enc_tokenizer
        ids = self._ids(tok, self.passage_prefix + text)[: self.max_passage_len]
        return ([tok.bos_token_id] if tok.bos_token_id is not None else []) + ids

    def _teacher_text(self, text):
        # truncate like the encoder does, so teacher and student see the same content
        return self.tokenizer.decode(self._ids(self.tokenizer, text)[: self.max_passage_len])

    def _student_prompt(self, example):
        n = len(example.get("passages", []))
        rendered = self.chat.prompt(user_text(example, [PLACEHOLDER.format(i) for i in range(n)]))
        parts = PLACEHOLDER_RE.split(rendered)  # [text, idx, text, idx, ..., text]
        ids = []
        for j, part in enumerate(parts):
            if j % 2 == 0:
                ids += self._ids(self.tokenizer, part)
            else:
                assert int(part) == j // 2, "passages must appear once, in order"
                ids += [MEM_ID] * self.num_mem
        return ids

    def _target(self, example):
        ids, weights = [], []
        for text, w in target_segments(example.get("target")) + [(self.chat.eot, 1.0)]:
            t = self._ids(self.tokenizer, text)
            ids, weights = ids + t, weights + [w] * len(t)
        return ids[: self.max_target_len], weights[: self.max_target_len]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        passages = [self._passage(p) for ex in features for p in ex.get("passages", [])]
        batch = {
            "enc_input_ids": pad(passages, self.enc_tokenizer.pad_token_id, left=True),
            "enc_attention_mask": pad([[1] * len(p) for p in passages], 0, left=True),
        }
        if not passages:
            batch = {k: v[:0] for k, v in batch.items()}

        prompts = [self._student_prompt(ex) for ex in features]
        if self.for_generation:
            batch["dec_input_ids"] = pad(prompts, pad_id, left=True)
            batch["dec_attention_mask"] = pad([[1] * len(p) for p in prompts], 0, left=True)
            return batch

        targets = [self._target(ex) for ex in features]
        seqs = [p + t for p, (t, _) in zip(prompts, targets)]
        batch["dec_input_ids"] = pad(seqs, pad_id)
        batch["dec_attention_mask"] = pad([[1] * len(s) for s in seqs], 0)
        batch["labels"] = pad([[-100] * len(p) + t for p, (t, _) in zip(prompts, targets)], -100)
        batch["label_weights"] = pad([[0.0] * len(p) + w for p, (_, w) in zip(prompts, targets)], 0.0).float()

        if self.with_teacher:
            t_prompts = [
                self._ids(self.tokenizer, self.chat.prompt(
                    user_text(ex, [self._teacher_text(p) for p in ex.get("passages", [])])))
                for ex in features
            ]
            t_seqs = [p + t for p, (t, _) in zip(t_prompts, targets)]
            batch["teacher_input_ids"] = pad(t_seqs, pad_id)
            batch["teacher_attention_mask"] = pad([[1] * len(s) for s in t_seqs], 0)
            batch["teacher_labels"] = pad([[-100] * len(p) + t for p, (t, _) in zip(t_prompts, targets)], -100)
        return batch
