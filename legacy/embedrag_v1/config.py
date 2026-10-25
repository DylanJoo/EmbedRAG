"""All EmbedRAG configs in one place.

  EmbedRAGConfig : architecture, saved with every checkpoint (embedrag_config.json)
  ModelArgs      : EmbedRAGConfig + what to init from / train         (train.py)
  DataArgs       : data files and collator settings                    (train.py)
  TrainArgs      : HF TrainingArguments + loss weights                 (train.py)
  RLArgs         : GRPO settings                                       (rl.py)
"""
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class EmbedRAGConfig:
    decoder: str = "meta-llama/Llama-3.2-1B-Instruct"
    encoder: Optional[str] = field(default=None, metadata={"help": "None = share the decoder backbone"})
    pooling: str = field(default="memory", metadata={"help": "memory (learned query slots) | mean (chunk mean-pool)"})
    num_mem: int = field(default=8, metadata={"help": "embeddings per passage"})
    enc_layer: int = field(default=-1, metadata={"help": "which encoder hidden layer to read"})
    encoder_lora: bool = True
    decoder_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    attn_implementation: str = "sdpa"


@dataclass
class ModelArgs(EmbedRAGConfig):
    init_from: Optional[str] = field(
        default=None, metadata={"help": "EmbedRAG checkpoint to start from"}
    )
    train_encoder: bool = True
    train_projector: bool = True
    train_decoder: bool = False


@dataclass
class DataArgs:
    train_files: str = field(default=None, metadata={"help": "comma-separated unified jsonl files"})
    eval_files: Optional[str] = None
    tasks: Optional[str] = field(default=None, metadata={"help": "comma-separated task filter"})
    max_passage_len: int = 256
    max_target_len: int = 512
    passage_prefix: str = ""
    system_prompt: Optional[str] = None
    use_chat_template: bool = True


@dataclass
class TrainArgs(TrainingArguments):
    output_dir: str = "./temp"
    remove_unused_columns: bool = False
    report_to: Optional[str] = "none"
    ce_weight: float = 1.0
    kd_weight: float = 0.0
    kd_temperature: float = 1.0


@dataclass
class RLArgs:
    init_from: str = None
    train_files: str = None
    output_dir: str = "./temp-rl"
    tasks: Optional[str] = None
    steps: int = 500
    prompts_per_step: int = 4
    group_size: int = 8
    micro_batch: int = 4
    lr: float = 5e-6
    beta: float = 0.02
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    max_passage_len: int = 256
    reward_weights: str = field(default="label=1,cite=1,f1=0.5,len=0.2")
    use_chat_template: bool = True
    save_steps: int = 100
    log_steps: int = 5
    seed: int = 42
