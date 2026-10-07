"""GRPO on top of an SFT'd EmbedRAG checkpoint (single GPU).

Policy = decoder LoRA ("decoder"); encoder + projector are frozen so passage embeddings are fixed
inputs. Reference = frozen copy of the SFT decoder adapter ("decoder_ref"), or the base decoder if
the checkpoint had no decoder adapter. Loss per token (on-policy, one update per batch):
    -A * exp(logp - logp.detach()) + beta * KL_k3(ref || policy),   A = group-normalized reward

  python -m embedrag.rl --init_from ckpt/stage2 --train_files data/embedrag/mds.jsonl --tasks summ
"""
import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import HfArgumentParser

from .backbone import ChatFormat
from .data import EmbedRAGCollator
from .model import CONFIG_NAME, EmbedRAG, set_adapter
from .rewards import compute_reward

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def completion_mask(ids, eos_ids):
    """1 for tokens up to and including the first EOS."""
    is_eos = torch.isin(ids, torch.tensor(eos_ids, device=ids.device))
    after = (is_eos.long().cumsum(-1) - is_eos.long()) > 0
    return (~after).long()


def token_logps(model, emb, mask, comp_ids, adapter):
    T = comp_ids.size(1)
    h = model.decoder_hidden(emb, mask, adapter)[:, -T - 1:-1]
    return torch.gather(F.log_softmax(model.logits(h), -1), -1, comp_ids.unsqueeze(-1)).squeeze(-1)


def main():
    args = HfArgumentParser(RLArgs).parse_args_into_dataclasses()[0]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.reward_weights.split(","))}

    with open(os.path.join(args.init_from, CONFIG_NAME)) as f:
        had_decoder = json.load(f)["decoder_lora"]
    model = EmbedRAG.from_pretrained(args.init_from, decoder_lora=True).to(DEVICE).eval()  # eval: no LoRA dropout
    ref = None
    if had_decoder:
        model.add_adapter("decoder_ref", model.decoder, copy_from="decoder", trainable=False)
        ref = "decoder_ref"
    model.set_trainable(encoder=False, projector=False, decoder=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    tok = model.tokenizer
    collator = EmbedRAGCollator(tok, model.enc_tokenizer, ChatFormat(tok, args.use_chat_template),
                                num_mem=model.cfg.num_mem, max_passage_len=args.max_passage_len,
                                for_generation=True)
    eos = model.decoder.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    keep = set(args.tasks.split(",")) if args.tasks else None
    data = [json.loads(l) for fn in args.train_files.split(",") for l in open(fn) if l.strip()]
    data = [ex for ex in data if keep is None or ex["task"] in keep]

    G = args.group_size
    for step in range(1, args.steps + 1):
        exs = random.sample(data, args.prompts_per_step)
        batch = {k: v.to(DEVICE) for k, v in collator(exs).items()}

        # --- rollout
        with torch.no_grad():
            mem = model.encode(batch["enc_input_ids"], batch["enc_attention_mask"]) \
                if batch["enc_input_ids"].numel() else None
            p_emb = model.embed_decoder_inputs(batch["dec_input_ids"], mem).repeat_interleave(G, 0)
            p_mask = batch["dec_attention_mask"].repeat_interleave(G, 0)
            set_adapter(model.decoder, "decoder")
            comp = model.decoder.generate(inputs_embeds=p_emb, attention_mask=p_mask, do_sample=True,
                                          temperature=args.temperature, top_p=args.top_p,
                                          max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
        c_mask = completion_mask(comp, eos)
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        scored = [compute_reward(t, exs[i // G], weights) for i, t in enumerate(texts)]
        r = torch.tensor([s for s, _ in scored], device=comp.device).view(-1, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)

        # --- policy update (micro-batched; token-level normalization over the whole step)
        n_tok = c_mask.sum().clamp(min=1)
        kl_sum = 0.0
        for s in range(0, comp.size(0), args.micro_batch):
            sl = slice(s, s + args.micro_batch)
            ids, m = comp[sl], c_mask[sl]
            emb = torch.cat([p_emb[sl], model.decoder.get_input_embeddings()(ids)], 1)
            mask = torch.cat([p_mask[sl], m], 1)
            logp = token_logps(model, emb, mask, ids, "decoder")
            with torch.no_grad():
                ref_logp = token_logps(model, emb, mask, ids, ref)
            d = ref_logp - logp
            kl = d.exp() - d - 1
            loss_tok = -adv[sl, None] * torch.exp(logp - logp.detach()) + args.beta * kl
            ((loss_tok * m).sum() / n_tok).backward()
            kl_sum += (kl.detach() * m).sum().item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_steps == 0:
            parts = {}
            for _, p in scored:
                for k, v in p.items():
                    parts.setdefault(k, []).append(v)
            print(json.dumps({"step": step, "reward": round(r.mean().item(), 4),
                              **{k: round(sum(v) / len(v), 4) for k, v in parts.items()},
                              "kl": round(kl_sum / n_tok.item(), 5),
                              "gen_len": round(c_mask.sum(1).float().mean().item(), 1)}), flush=True)
        if step % args.save_steps == 0 or step == args.steps:
            model.save(os.path.join(args.output_dir, f"step-{step}"))


if __name__ == "__main__":
    main()
