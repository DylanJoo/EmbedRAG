"""Train ClaimEmb v4 (query LoRA [+ back-track residual] + CRF region locator) via HF Trainer.

  python -m embedrag_v4.train --backbone <Qwen3-Embedding> --query mask \
      --train_files data/claimemb/rag25.train.regions.jsonl --bf16 --output_dir ckpt/v4
"""
import json
from collections import defaultdict
from dataclasses import asdict, fields

import torch
from transformers import HfArgumentParser, Trainer, TrainerCallback

from .config import ClaimEmbConfig, DataArgs, ModelArgs, TrainArgs
from .data import ClaimCollator
from .model import ClaimEmb


class ClaimEmbTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parts = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(**inputs)
        for k in ("distill", "cos", "tag"):
            self._parts[k].append(out[k].item())
        return (out["loss"], out) if return_outputs else out["loss"]

    def log(self, logs, *args, **kwargs):
        logs.update({k: sum(v) / len(v) for k, v in self._parts.items() if v})
        self._parts.clear()
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        self.model.save(output_dir or self.args.output_dir)  # LoRA (+ residual) + tag head + CRF only


class MatchWarmup(TrainerCallback):
    """grid: proportional claim -> query assignment for the first `steps` steps, then Hungarian."""

    def __init__(self, model, steps):
        self.model, self.steps = model, steps

    def on_step_begin(self, args, state, control, **kwargs):
        self.model.match = "hungarian" if state.global_step >= self.steps else "order"


class ConfigToWandb(TrainerCallback):
    def __init__(self, config):
        self.config = config

    def on_train_begin(self, args, state, control, **kwargs):
        import wandb
        if wandb.run is not None:
            wandb.config.update(self.config, allow_val_change=True)


def load_data(files):
    return [json.loads(l) for f in files.split(",") for l in open(f) if l.strip()]


def main():
    margs, dargs, targs = HfArgumentParser((ModelArgs, DataArgs, TrainArgs)).parse_args_into_dataclasses()
    dtype = torch.bfloat16 if targs.bf16 else torch.float32
    if margs.init_from:
        model = ClaimEmb.from_pretrained(margs.init_from, dtype)
    else:
        model = ClaimEmb(ClaimEmbConfig(**{f.name: getattr(margs, f.name) for f in fields(ClaimEmbConfig)}), dtype)
    model.distill_weight, model.cos_weight, model.tag_weight = targs.distill_weight, targs.cos_weight, targs.tag_weight
    model.temperature, model.keep_weight = targs.temperature, targs.keep_weight
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    collator = ClaimCollator(model.tokenizer, model.cfg.max_claims, dargs.max_doc_len, dargs.max_claim_len)
    trainer = ClaimEmbTrainer(model=model, args=targs, data_collator=collator,
                              train_dataset=load_data(dargs.train_files),
                              eval_dataset=load_data(dargs.eval_files) if dargs.eval_files else None)
    if model.cfg.query == "grid" and targs.match_warmup_steps:
        trainer.add_callback(MatchWarmup(model, targs.match_warmup_steps))
    if "wandb" in targs.report_to:
        trainer.add_callback(ConfigToWandb({**asdict(model.cfg), **asdict(dargs)}))
    trainer.train()
    trainer.save_model(targs.output_dir)


if __name__ == "__main__":
    main()
