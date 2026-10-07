"""EmbedRAG: an encoder LM compresses each passage into k embeddings; a decoder LM reads them as soft tokens.

    passage ──(encoder LM + LoRA "encoder")──> hidden states ──pool──> (k, H_enc)
            ──projector──> (k, H_dec) ──spliced into decoder inputs_embeds at MEM_ID slots
    prompt + [mem x k per passage] ──(decoder LM [+ LoRA "decoder"])──> answer

Encoder and decoder are either one shared backbone with two LoRA adapters (default; one copy of
the weights in memory) or two different backbones (`encoder=` another model, e.g. a small Qwen).
"""
import contextlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file, save_file

from .backbone import hidden_size, load_backbone, logit_softcap, lora_target_regex

MEM_ID = -1  # sentinel token id in decoder inputs, replaced by passage embeddings
CONFIG_NAME, WEIGHTS_NAME = "embedrag_config.json", "embedrag_weights.safetensors"


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


def set_adapter(model, name):
    """Switch the active LoRA in-place (None = base model) without PEFT's requires_grad side effects."""
    if not hasattr(model, "_embedrag_tuners"):
        model._embedrag_tuners = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    for m in model._embedrag_tuners:
        m._disable_adapters = name is None
        if name is not None:
            m._active_adapter = [name]
    model._embedrag_adapter = name


@contextlib.contextmanager
def use_adapter(model, name):
    prev = getattr(model, "_embedrag_adapter", None)
    set_adapter(model, name)
    try:
        yield
    finally:
        set_adapter(model, prev)


def checkpoint_context_fn(model):
    """Gradient checkpointing recomputes layers during backward, when another adapter may be active
    (shared encoder/decoder backbone). Capture the adapter at forward time and restore it on recompute."""
    def fn():
        return contextlib.nullcontext(), use_adapter(model, getattr(model, "_embedrag_adapter", None))
    return fn


class Projector(nn.Module):
    """MLP into the decoder embedding space; output norm is pinned to the typical token-embedding norm."""

    def __init__(self, d_in, d_out, init_norm):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)
        self.scale = nn.Parameter(torch.tensor(float(init_norm)))

    def forward(self, x):
        y = self.fc2(F.gelu(self.fc1(self.ln(x.float()))))
        return F.normalize(y, dim=-1) * self.scale


class EmbedRAG(nn.Module):
    def __init__(self, cfg: EmbedRAGConfig, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.decoder, self.tokenizer = load_backbone(cfg.decoder, dtype, cfg.attn_implementation)
        if cfg.encoder:
            self.encoder, self.enc_tokenizer = load_backbone(cfg.encoder, dtype, cfg.attn_implementation)
        else:
            self.encoder, self.enc_tokenizer = None, self.tokenizer
        self.softcap = logit_softcap(self.decoder)
        # runtime loss weights (set by the training script, not saved)
        self.ce_weight, self.kd_weight, self.kd_temperature = 1.0, 0.0, 1.0

        if cfg.encoder_lora:
            self.add_adapter("encoder", self.enc_model)
        if cfg.decoder_lora:
            self.add_adapter("decoder", self.decoder)

        enc_emb = self.enc_model.get_input_embeddings()
        dec_emb = self.decoder.get_input_embeddings()
        with torch.no_grad():
            g = torch.Generator().manual_seed(0)
            ids = torch.randint(0, enc_emb.num_embeddings, (max(cfg.num_mem, 1),), generator=g)
            mem_init = enc_emb(ids.to(enc_emb.weight.device)).float()
            ids = torch.randint(0, dec_emb.num_embeddings, (1024,), generator=g)
            dec_norm = dec_emb(ids.to(dec_emb.weight.device)).float().norm(dim=-1).median()
        # learned query slots appended to each passage (pooling == "memory")
        self.mem_queries = nn.Parameter(mem_init[: cfg.num_mem].clone()) if cfg.pooling == "memory" else None
        self.projector = Projector(hidden_size(self.enc_model), hidden_size(self.decoder), dec_norm)

    # ---------------------------------------------------------------- setup
    @property
    def enc_model(self):
        return self.encoder if self.encoder is not None else self.decoder

    def add_adapter(self, name, model, copy_from=None, trainable=True):
        lora = LoraConfig(
            r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha, lora_dropout=self.cfg.lora_dropout,
            target_modules=lora_target_regex(), bias="none",
        )
        inject_adapter_in_model(lora, model, adapter_name=name)
        for m in model.modules():
            if isinstance(m, BaseTunerLayer) and name in m.lora_A:
                if copy_from is not None:
                    m.lora_A[name].load_state_dict(m.lora_A[copy_from].state_dict())
                    m.lora_B[name].load_state_dict(m.lora_B[copy_from].state_dict())
                for p in list(m.lora_A[name].parameters()) + list(m.lora_B[name].parameters()):
                    p.requires_grad_(trainable)

    def set_trainable(self, encoder=True, projector=True, decoder=True):
        """Freeze/unfreeze components per training stage (other adapters, e.g. RL refs, stay frozen)."""
        for n, p in self.named_parameters():
            if "lora_" in n:  # e.g. decoder.model.layers.0.self_attn.q_proj.lora_A.encoder.weight
                adapter = n.split("lora_", 1)[1].split(".")[1]
                p.requires_grad_({"encoder": encoder, "decoder": decoder}.get(adapter, False))
            elif n.startswith(("projector", "mem_queries")):
                p.requires_grad_(projector)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        for m in [self.decoder] + ([self.encoder] if self.encoder is not None else []):
            kw = {"use_reentrant": False, "context_fn": checkpoint_context_fn(m)}
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)
            m.config.use_cache = False

    # ---------------------------------------------------------------- encoder
    def encode(self, enc_input_ids, enc_attention_mask):
        """Left-padded passages (P, L) -> passage embeddings in decoder space (P, k, H_dec)."""
        model, k = self.enc_model, self.cfg.num_mem
        set_adapter(model, "encoder")
        emb = model.get_input_embeddings()(enc_input_ids)
        mask = enc_attention_mask
        if self.cfg.pooling == "memory":
            q = self.mem_queries.to(emb.dtype).unsqueeze(0).expand(emb.size(0), -1, -1)
            emb = torch.cat([emb, q], 1)
            mask = torch.cat([mask, mask.new_ones(mask.size(0), k)], 1)
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
        out = model(inputs_embeds=emb, attention_mask=mask, position_ids=pos,
                    output_hidden_states=True, use_cache=False, logits_to_keep=1)
        h = out.hidden_states[self.cfg.enc_layer]
        if self.cfg.pooling == "memory":
            h = h[:, -k:]
        else:  # mean-pool k contiguous chunks of the real tokens
            rank = (mask.long().cumsum(-1) - 1).clamp(min=0)
            length = mask.sum(-1, keepdim=True).clamp(min=1)
            chunk = (rank * k // length).clamp(max=k - 1)
            assign = F.one_hot(chunk, k).to(h.dtype) * mask.unsqueeze(-1).to(h.dtype)
            h = torch.einsum("plk,plh->pkh", assign, h) / assign.sum(1).clamp(min=1).unsqueeze(-1)
        return self.projector(h)

    # ---------------------------------------------------------------- decoder
    def embed_decoder_inputs(self, input_ids, mem=None):
        emb = self.decoder.get_input_embeddings()(input_ids.clamp(min=0))
        slots = input_ids == MEM_ID
        if mem is None:
            assert not slots.any(), "MEM slots present but no passage embeddings given"
            return emb
        assert slots.sum() == mem.shape[0] * mem.shape[1], "MEM slot count != passages x num_mem"
        return emb.masked_scatter(slots.unsqueeze(-1), mem.to(emb.dtype))

    def decoder_hidden(self, inputs_embeds, attention_mask, adapter="decoder"):
        set_adapter(self.decoder, adapter)
        pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        out = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=pos,
                           output_hidden_states=True, use_cache=False, logits_to_keep=1)
        return out.hidden_states[-1]

    def logits(self, h):
        logits = self.decoder.get_output_embeddings()(h).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def target_logits(self, h, labels):
        """Logits only at positions that predict a label (saves the full-vocab projection)."""
        sel = labels[:, 1:] != -100
        return self.logits(h[:, :-1][sel]), labels[:, 1:][sel], sel

    # ---------------------------------------------------------------- training
    def forward(self, enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask, labels,
                label_weights=None, teacher_input_ids=None, teacher_attention_mask=None, teacher_labels=None):
        mem = self.encode(enc_input_ids, enc_attention_mask) if enc_input_ids.numel() else None
        h = self.decoder_hidden(self.embed_decoder_inputs(dec_input_ids, mem), dec_attention_mask)
        logits, y, sel = self.target_logits(h, labels)
        w = label_weights[:, 1:][sel] if label_weights is not None else torch.ones_like(y, dtype=logits.dtype)
        ce = (F.cross_entropy(logits, y, reduction="none") * w).sum() / w.sum().clamp(min=1e-6)
        out = {"ce": ce.detach()}
        loss = self.ce_weight * ce

        if self.kd_weight > 0 and teacher_input_ids is not None:
            # teacher = the frozen base decoder reading the passages as plain text
            with torch.no_grad():
                th = self.decoder_hidden(self.embed_decoder_inputs(teacher_input_ids), teacher_attention_mask, None)
                t_logits, t_y, _ = self.target_logits(th, teacher_labels)
            assert torch.equal(t_y, y), "teacher/student targets must be identical"
            T = self.kd_temperature
            t_logp = F.log_softmax(t_logits / T, -1)
            kd = (t_logp.exp() * (t_logp - F.log_softmax(logits / T, -1))).sum(-1)
            kd = (kd * w).sum() / w.sum().clamp(min=1e-6) * T * T
            out["kd"] = kd.detach()
            loss = loss + self.kd_weight * kd

        out["loss"] = loss
        return out

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask, **gen_kwargs):
        """Decoder inputs must be left-padded. Returns only the newly generated token ids."""
        mem = self.encode(enc_input_ids, enc_attention_mask) if enc_input_ids.numel() else None
        emb = self.embed_decoder_inputs(dec_input_ids, mem)
        set_adapter(self.decoder, "decoder")
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        return self.decoder.generate(inputs_embeds=emb, attention_mask=dec_attention_mask, **gen_kwargs)

    # ---------------------------------------------------------------- io
    def adapter_state_dict(self):
        # frozen RL reference adapters ("*_ref") are not saved
        keep = lambda k: ("lora_" in k and "_ref." not in k) or k.startswith(("projector.", "mem_queries"))
        return {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items() if keep(k)}

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        save_file(self.adapter_state_dict(), os.path.join(out_dir, WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, ckpt_dir, dtype=torch.bfloat16, **overrides):
        """Load a saved checkpoint. `overrides` may add components (e.g. decoder_lora=True for stage 2)."""
        with open(os.path.join(ckpt_dir, CONFIG_NAME)) as f:
            cfg = EmbedRAGConfig(**{**json.load(f), **overrides})
        model = cls(cfg, dtype)
        missing, unexpected = model.load_state_dict(load_file(os.path.join(ckpt_dir, WEIGHTS_NAME)), strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        return model
