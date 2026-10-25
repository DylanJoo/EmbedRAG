"""ClaimEmb configs.

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
    stride: int = field(default=16, metadata={"help": "one query per `stride` doc tokens"})
    max_queries: int = field(default=64, metadata={"help": "max queries per doc"})
    residual_dim: int = field(default=128, metadata={"help": "sinusoidal features of the back-track distance"})
    max_claims: int = field(default=32, metadata={"help": "max gold claims per doc used in training"})
    lora_r: int = field(default=16, metadata={"help": "query LoRA rank; 0 = no LoRA (only the residual and head train)"})
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class ModelArgs(ClaimEmbConfig):
    init_from: Optional[str] = field(default=None, metadata={"help": "ClaimEmb checkpoint to start from"})


@dataclass
class DataArgs:
    train_files: str = field(default=None, metadata={"help": "comma-separated jsonl of {id, text, claims}"})
    max_doc_len: int = 1024
    max_claim_len: int = 64


@dataclass
class TrainArgs(TrainingArguments):
    output_dir: str = "./temp"
    remove_unused_columns: bool = False
    report_to: Optional[str] = "none"
    distill_weight: float = field(default=1.0, metadata={"help": "claim-claim score distillation (KL)"})
    cos_weight: float = 1.0
    keep_weight: float = field(default=1.0, metadata={"help": "BCE of the has-claim head"})
    temperature: float = field(default=0.05, metadata={"help": "softmax temperature of teacher and student scores"})
    match_warmup_steps: int = field(default=100, metadata={"help": "proportional claim -> query assignment for the "
                                                                    "first N steps, then Hungarian"})
