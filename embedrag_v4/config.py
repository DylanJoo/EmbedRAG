"""ClaimEmb v4 configs: one embedding LM, <|endoftext|> queries that back-track to claim regions.

  ClaimEmbConfig : architecture, saved with every checkpoint (claimemb_config.json)
  DataArgs       : data files and collator settings          (train.py)
  TrainArgs      : HF TrainingArguments + loss weights        (train.py)
"""
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

QWEN3_EMB = "Qwen/Qwen3-Embedding-0.6B"


@dataclass
class ClaimEmbConfig:
    backbone: str = field(default=QWEN3_EMB, metadata={"help": "embedding LM (last-token pooling); also the teacher"})
    query: str = field(default="mask", metadata={"help": "how a claim query back-tracks to its region: mask (attends "
                                                        "only to the region + token 0) | residual (sees the whole "
                                                        "doc; input = <|endoftext|> + b(distance back to region end)) "
                                                        "| grid (residual queries at fixed positions every `stride` "
                                                        "doc tokens, no regions; Hungarian-matched to claims)"})
    stride: int = field(default=16, metadata={"help": "grid: one query per `stride` doc tokens"})
    max_queries: int = field(default=64, metadata={"help": "grid: max queries per doc"})
    query_pos: str = field(default="region", metadata={"help": "query position id: region (right after the region "
                                                              "end) | end (the doc end token's, like the doc vector)"})
    residual_dim: int = field(default=128, metadata={"help": "sinusoidal features of the back-track distance"})
    max_claims: int = 32
    lora_r: int = field(default=16, metadata={"help": "query LoRA rank; 0 = no LoRA (only the residual/heads train)"})
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class ModelArgs(ClaimEmbConfig):
    init_from: Optional[str] = field(default=None, metadata={"help": "ClaimEmb v4 checkpoint to start from"})


@dataclass
class DataArgs:
    train_files: str = field(default=None, metadata={"help": "comma-separated *.regions.jsonl from embedrag_v4.align"})
    eval_files: Optional[str] = None
    max_doc_len: int = 1024
    max_claim_len: int = 64


@dataclass
class TrainArgs(TrainingArguments):
    output_dir: str = "./temp"
    remove_unused_columns: bool = False
    report_to: Optional[str] = "none"
    distill_weight: float = field(default=1.0, metadata={"help": "claim-claim score distillation (KL)"})
    cos_weight: float = 1.0
    tag_weight: float = field(default=1.0, metadata={"help": "CRF segmentation loss (the region locator); grid: unused"})
    keep_weight: float = field(default=1.0, metadata={"help": "grid: BCE of the has-claim head"})
    match_warmup_steps: int = field(default=100, metadata={"help": "grid: proportional assignment of claims to grid "
                                                                    "queries for the first N steps, then Hungarian"})
    temperature: float = field(default=0.05, metadata={"help": "softmax temperature of teacher and student scores"})
