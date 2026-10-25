"""Train ClaimEmb (claim LoRA + [CLM] step input + stop head) via HF Trainer.

  accelerate launch -m embedrag.train --train_files data/claimemb/eng.train.jsonl \
      --bf16 --per_device_train_batch_size 32 --learning_rate 1e-4 --output_dir ckpt/claimemb
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
        for k in ("distill", "cos", "stop"):
            self._parts[k].append(out[k].item())
        return (out["loss"], out) if return_outputs else out["loss"]

    def log(self, logs, *args, **kwargs):
        logs.update({k: sum(v) / len(v) for k, v in self._parts.items() if v})
        self._parts.clear()
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        self.model.save(output_dir or self.args.output_dir)  # LoRA + [CLM] + stop head only


class ConfigToWandb(TrainerCallback):
    """HF's wandb callback logs only TrainingArguments; add the model and data settings to the run config."""

    def __init__(self, config):
        self.config = config

    def on_train_begin(self, args, state, control, **kwargs):
        import wandb
        if wandb.run is not None:
            wandb.config.update(self.config, allow_val_change=True)


class MatchWarmup(TrainerCallback):
    """Train with document-order targets for the first `steps` steps, then switch to `match`."""

    def __init__(self, model, match, steps):
        self.model, self.match, self.steps = model, match, steps

    def on_step_begin(self, args, state, control, **kwargs):
        self.model.match = self.match if state.global_step >= self.steps else "order"


def load_data(files):
    return [json.loads(l) for f in files.split(",") for l in open(f) if l.strip()]


def main():
    margs, dargs, targs = HfArgumentParser((ModelArgs, DataArgs, TrainArgs)).parse_args_into_dataclasses()
    dtype = torch.bfloat16 if targs.bf16 else torch.float32
    if margs.init_from:
        model = ClaimEmb.from_pretrained(margs.init_from, dtype)
    else:
        model = ClaimEmb(ClaimEmbConfig(**{f.name: getattr(margs, f.name) for f in fields(ClaimEmbConfig)}), dtype)
    model.distill_weight, model.cos_weight, model.stop_weight = targs.distill_weight, targs.cos_weight, targs.stop_weight
    model.temperature, model.stop_pos_weight, model.match = targs.temperature, targs.stop_pos_weight, targs.match

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.2f}M")

    collator = ClaimCollator(model.tokenizer, model.cfg.max_claims, dargs.max_doc_len, dargs.max_claim_len,
                             dargs.extra_stop_steps)
    trainer = ClaimEmbTrainer(
        model=model,
        args=targs,
        data_collator=collator,
        train_dataset=load_data(dargs.train_files),
        eval_dataset=load_data(dargs.eval_files) if dargs.eval_files else None,
    )
    if targs.match_warmup_steps:
        trainer.add_callback(MatchWarmup(model, targs.match, targs.match_warmup_steps))
    if "wandb" in targs.report_to:
        trainer.add_callback(ConfigToWandb({**asdict(model.cfg), **asdict(dargs)}))
    trainer.train()
    trainer.save_model(targs.output_dir)


if __name__ == "__main__":
    main()
