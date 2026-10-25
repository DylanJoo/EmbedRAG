"""All ClaimEmb configs in one place. Defaults = the current best recipe.

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
    max_claims: int = field(default=32, metadata={"help": "max claim steps per document"})
    step_stride: int = field(default=24, metadata={"help": "position gap between [CLM] steps (step t at doc_len + "
                                                          "stride * t), e.g. ~ mean claim length in tokens"})
    soft_prompt: str = field(default="", metadata={"help": "text whose token embeddings init learned prompt "
                                                           "vectors between the doc and the steps ('' = none)"})
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class ModelArgs(ClaimEmbConfig):
    init_from: Optional[str] = field(default=None, metadata={"help": "ClaimEmb checkpoint to start from"})


@dataclass
class DataArgs:
    train_files: str = field(default=None, metadata={"help": "comma-separated jsonl from build_data.py"})
    eval_files: Optional[str] = None
    max_doc_len: int = 1024
    max_claim_len: int = 64
    extra_stop_steps: int = field(default=2, metadata={"help": "steps past the last claim also labeled stop"})


@dataclass
class TrainArgs(TrainingArguments):
    output_dir: str = "./temp"
    remove_unused_columns: bool = False
    report_to: Optional[str] = "none"
    distill_weight: float = field(default=1.0, metadata={"help": "claim-claim score distillation (KL)"})
    cos_weight: float = 1.0
    stop_weight: float = 1.0
    stop_pos_weight: float = field(default=3.0, metadata={"help": "BCE weight on stop labels (class balance)"})
    temperature: float = field(default=0.05, metadata={"help": "softmax temperature of teacher and student scores"})
    match: str = field(default="hungarian", metadata={"help": "step-to-claim targets: order (step t = claim t) | "
                                                              "hungarian (one-to-one, max total cos, per doc)"})
    match_warmup_steps: int = field(default=100, metadata={"help": "with match=hungarian: use document order for "
                                                                    "the first N steps, then switch"})
