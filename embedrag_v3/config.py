"""All ClaimEmb v3 configs in one place.

  ClaimEmbConfig : architecture, saved with every checkpoint (claimemb_config.json)
  DataArgs       : data files and collator settings          (train.py)
  TrainArgs      : HF TrainingArguments + loss weights        (train.py)
"""
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

QWEN3 = "Qwen/Qwen3-0.6B"
QWEN3_EMB = "Qwen/Qwen3-Embedding-0.6B"
# Qwen3 chat format, non-thinking (empty think block); the doc is read by the frozen student, steps follow
DOC_TEMPLATE = ("<|im_start|>user\n{doc}\n\nList the claims made in the document above.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
# named templates (--doc_template <name>); "doc_eos" keeps the embedding model's own doc input ({doc}<|endoftext|>)
# untouched and only switches to generation after it, so with student = teacher the doc states (and the doc
# vector at <|endoftext|>) are exactly the index's
TEMPLATES = {"chat": DOC_TEMPLATE, "doc": "{doc}<|endoftext|>", "doc_eos": "{doc}<|endoftext|><|im_start|>assistant\n"}


@dataclass
class ClaimEmbConfig:
    student: str = field(default=QWEN3, metadata={"help": "decoder LM that reads the doc and runs the [CLM] steps"})
    teacher: str = field(default=QWEN3_EMB, metadata={"help": "frozen embedding LM (last-token pooling): "
                                                              "gold claim vectors and the doc-vector baseline"})
    doc_template: str = field(default=DOC_TEMPLATE, metadata={"help": "student input around the doc ({doc}), or a "
                                                                      "name in TEMPLATES (chat | doc | doc_eos)"})
    hide_doc_end: bool = field(default=False, metadata={"help": "steps can't attend to the prompt's last token (with "
                                                                "template doc: the <|endoftext|> the doc vector is read at)"})
    step_init: str = field(default="-", metadata={"help": "text whose (last) token embedding inits [CLM]"})
    max_claims: int = field(default=32, metadata={"help": "max claim steps per document"})
    step_stride: int = field(default=24, metadata={"help": "position gap between [CLM] steps (step t at "
                                                           "prompt_len + stride * t)"})
    step_weights: str = field(default="lora", metadata={"help": "weights the [CLM] steps run with: lora (adapter on "
                                                                "the frozen student) | full (a trainable full copy)"})
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def __post_init__(self):
        self.doc_template = TEMPLATES.get(self.doc_template, self.doc_template)


@dataclass
class ModelArgs(ClaimEmbConfig):
    init_from: Optional[str] = field(default=None, metadata={"help": "ClaimEmb v3 checkpoint to start from"})


@dataclass
class DataArgs:
    train_files: str = field(default=None, metadata={"help": "comma-separated jsonl from embedrag.build_data"})
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
