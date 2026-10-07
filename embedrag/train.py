"""SFT for EmbedRAG (stage 1 alignment / stage 2 task tuning), via HF Trainer.

Stage 1 (align):  --train_encoder --train_projector, decoder frozen, data = pretrain (ae/cont)
Stage 2 (task):   --init_from <stage1> [--decoder_lora --train_decoder] --kd_weight 1, data = mds/qa
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Optional

import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments

from .backbone import ChatFormat
from .data import EmbedRAGCollator
from .model import EmbedRAG, EmbedRAGConfig


@dataclass
class ModelArgs(EmbedRAGConfig):
    init_from: Optional[str] = field(default=None, metadata={"help": "EmbedRAG checkpoint to start from"})
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


class EmbedRAGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parts = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(**inputs)
        for k in ("ce", "kd"):
            if k in out:
                self._parts[k].append(out[k].item())
        return (out["loss"], out) if return_outputs else out["loss"]

    def log(self, logs, *args, **kwargs):
        logs.update({k: sum(v) / len(v) for k, v in self._parts.items() if v})
        self._parts.clear()
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        self.model.save(output_dir or self.args.output_dir)  # adapters + projector only


def load_data(files, tasks):
    # plain records: `target` / `meta` vary in type across tasks, which Arrow (HF datasets) can't mix
    keep = set(tasks.split(",")) if tasks else None
    rows = [json.loads(l) for f in files.split(",") for l in open(f) if l.strip()]
    return [r for r in rows if keep is None or r["task"] in keep]


def main():
    margs, dargs, targs = HfArgumentParser((ModelArgs, DataArgs, TrainArgs)).parse_args_into_dataclasses()
    cfg_keys = {f.name for f in fields(EmbedRAGConfig)}
    cfg = {k: getattr(margs, k) for k in cfg_keys}
    dtype = torch.bfloat16 if targs.bf16 else torch.float32

    if margs.init_from:  # keep the saved architecture; only allow adding a decoder adapter
        model = EmbedRAG.from_pretrained(margs.init_from, dtype, **({"decoder_lora": True} if margs.decoder_lora else {}))
    else:
        model = EmbedRAG(EmbedRAGConfig(**cfg), dtype)
    model.set_trainable(margs.train_encoder, margs.train_projector, margs.train_decoder)
    model.ce_weight, model.kd_weight, model.kd_temperature = targs.ce_weight, targs.kd_weight, targs.kd_temperature

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.2f}M")

    collator = EmbedRAGCollator(
        tokenizer=model.tokenizer, enc_tokenizer=model.enc_tokenizer,
        chat=ChatFormat(model.tokenizer, dargs.use_chat_template, dargs.system_prompt),
        num_mem=model.cfg.num_mem, max_passage_len=dargs.max_passage_len,
        max_target_len=dargs.max_target_len, passage_prefix=dargs.passage_prefix,
        with_teacher=targs.kd_weight > 0,
    )
    trainer = EmbedRAGTrainer(
        model=model, args=targs, data_collator=collator,
        train_dataset=load_data(dargs.train_files, dargs.tasks),
        eval_dataset=load_data(dargs.eval_files, dargs.tasks) if dargs.eval_files else None,
    )
    trainer.train()
    trainer.save_model(targs.output_dir)


if __name__ == "__main__":
    main()
