"""ClaimEmb: an embedding LM that reads a document once and writes out its claim embeddings step by step.

    <doc tokens><|endoftext|>  ──(Qwen3-Embedding, frozen, no grad)──>  doc vector + KV cache
    [CLM] [CLM] ... [CLM]      ──(same backbone + LoRA "claim", reading that cache)──>  h_1 ... h_T
        e_t = normalize(h_t)              claim embedding, in the backbone's own retrieval space
        p_t = sigmoid(stop_head(h_t))     "no claims left"; generation ends at the first p_t > 0.5

The LoRA is active only on the [CLM] steps, so the document's states are the frozen model's: the doc vector
equals the precomputed index vector exactly, and training never backpropagates through document tokens.
Each step attends to the document and to all earlier steps. [CLM] is one learned vector shared by all steps,
initialized from <|endoftext|>, the token the backbone already pools at. Steps sit `step_stride` positions apart.
With `step_weights="full"` the steps instead run through a trainable full copy of the backbone.
An optional soft prompt (learned vectors, init from the tokens of `soft_prompt`) goes between the doc and the steps.

The teacher is the frozen backbone embedding each gold claim's text. Steps are matched to gold claims in
document order, or one-to-one by Hungarian matching (`match`), and trained by score distillation: each step's
scores over the batch's claims should match the scores its gold claim vector gives them (see `distill_loss`).
"""
import contextlib
import copy
import json
import os
from dataclasses import asdict, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file, save_file
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel, AutoTokenizer

from .config import ClaimEmbConfig

CONFIG_NAME, WEIGHTS_NAME = "claimemb_config.json", "claimemb_weights.safetensors"
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def hungarian_targets(e, t, y, doc):
    """Re-assign gold claims to steps one-to-one within each doc, maximizing total cos (order-free targets).

    e (S, H) step vectors, t (C, H) gold claim vectors, y (S,) document-order targets, doc (S,) doc of each step.
    """
    y = y.clone()
    sim = (e.detach() @ t.T).float().cpu()
    for d in doc.unique():
        idx = (doc == d).nonzero().squeeze(1)
        claims = y[idx]
        rows, cols = linear_sum_assignment(-sim[idx.cpu()][:, claims.cpu()].numpy())
        y[idx[rows]] = claims[cols]
    return y


def distill_loss(e, t, y, temperature):
    """Listwise score distillation: KL(softmax(t_y . t / tau) || softmax(e . t / tau)), averaged over steps.

    e (S, H) step vectors, t (C, H) candidate vectors (the batch's gold claims), y (S,) each step's gold claim.
    The teacher scores are the gold claim vector's own scores, so the loss is 0 exactly at e = t_y: unlike
    one-hot InfoNCE, near-duplicate claims in a tightly related document are not pushed apart, while in a
    dispersed document the target is nearly one-hot.
    """
    target = F.softmax(t[y] @ t.T / temperature, dim=-1)
    return F.kl_div(F.log_softmax(e @ t.T / temperature, dim=-1), target, reduction="batchmean")


def last_token(h, mask):
    """Right-padded (B, L, H) -> hidden state at each row's last real token (B, H)."""
    return h[torch.arange(h.size(0), device=h.device), mask.sum(-1) - 1]


class ClaimEmb(nn.Module):
    def __init__(self, cfg: ClaimEmbConfig, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.backbone, dtype=dtype, attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Define training configs (set from TrainArgs in train.py)
        self.distill_weight, self.cos_weight, self.stop_weight, self.temperature = 1.0, 1.0, 1.0, 0.05
        self.stop_pos_weight = 1.0
        self.match = "order"  # or "hungarian"

        # Define step weights: a lora on the frozen backbone (active on [CLM] steps only; see `lora`), or a full
        # trainable copy of the backbone; either way documents and teacher go through the frozen original
        if cfg.step_weights == "full":
            # fp32 master weights: small full-model updates would round away in bf16
            self.step_backbone = copy.deepcopy(self.backbone).float().requires_grad_(True)
        else:
            lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                              target_modules=LORA_MODULES, bias="none")
            inject_adapter_in_model(lora, self.backbone, adapter_name="claim")
            for n, p in self.backbone.named_parameters():
                p.requires_grad_("lora_" in n)
            self.step_backbone = None
        self._tuners = [m for m in self.backbone.modules() if isinstance(m, BaseTunerLayer)]

        # Define step input: init [CLM] as the end token the backbone already pools at
        emb, H = self.backbone.get_input_embeddings(), self.backbone.config.hidden_size
        end_id = (self.tokenizer("")["input_ids"] or [self.tokenizer.eos_token_id])[-1]
        self.step_query = nn.Parameter(emb.weight[end_id].detach().float().unsqueeze(0))
        self.stop_head = nn.Linear(H, 1)

        # Define soft prompt (optional): learned vectors between the doc and the steps, init from a text's tokens
        ids = self.tokenizer(cfg.soft_prompt, add_special_tokens=False)["input_ids"] if cfg.soft_prompt else []
        self.soft_prompt = nn.Parameter(emb.weight[ids].detach().float()) if ids else None

    # ---------------------------------------------------------------- setup
    @contextlib.contextmanager
    def lora(self, on):
        if not self._tuners:  # step_weights="full": no adapter to switch
            yield
            return
        prev = self._tuners[0]._disable_adapters
        for m in self._tuners:
            m._disable_adapters = not on
        try:
            yield
        finally:
            for m in self._tuners:
                m._disable_adapters = prev

    # ---------------------------------------------------------------- frozen passes
    @torch.no_grad()
    def teacher(self, input_ids, attention_mask):
        """Frozen backbone on right-padded texts, last-token pooled + normalized (= the index vectors)."""
        with self.lora(False):
            h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(last_token(h, attention_mask).float(), dim=-1)

    @torch.no_grad()
    def read_doc(self, doc_input_ids, doc_attention_mask):
        """Frozen backbone on left-padded docs -> (doc vectors (B, H), KV cache)."""
        pos = (doc_attention_mask.cumsum(-1) - 1).clamp(min=0)
        with self.lora(False):
            out = self.backbone(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                                position_ids=pos, use_cache=True)
        return F.normalize(out.last_hidden_state[:, -1].float(), dim=-1), out.past_key_values

    # ---------------------------------------------------------------- student
    def steps(self, cache, doc_attention_mask, step_mask):
        """[CLM] steps after cached docs -> (claim vectors, stop logits) for the real steps (S, H), (S,).

        step_mask (B, T) marks real steps (right-padded); outputs are in row-major order of step_mask.
        """
        B, T = step_mask.shape
        P = 0 if self.soft_prompt is None else self.soft_prompt.size(0)
        x = self.step_query.expand(B, T, -1)
        if P:  # soft prompt between the doc and the steps, trained with them
            x = torch.cat([self.soft_prompt.expand(B, P, -1), x], 1)
        n = doc_attention_mask.sum(-1, keepdim=True)
        pos = torch.cat([n + torch.arange(P, device=x.device),
                         n + P + self.cfg.step_stride * torch.arange(T, device=x.device)], 1)
        mask = torch.cat([doc_attention_mask, step_mask.new_ones(B, P).long(), step_mask.long()], 1)
        net, full = self.backbone, self.step_backbone is not None
        if full:  # the fp32 step copy runs in fp32 (no autocast) on an fp32 copy of the doc cache
            net = self.step_backbone
            for layer in cache.layers:
                layer.keys, layer.values = layer.keys.float(), layer.values.float()
        with self.lora(True), torch.autocast(x.device.type, enabled=not full and torch.is_autocast_enabled(x.device.type)):
            h = net(
                inputs_embeds=x.to(net.dtype),
                attention_mask=mask,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True
            ).last_hidden_state[:, P:][step_mask].float()
        return F.normalize(h, dim=-1), self.stop_head(h).squeeze(-1)

    # ---------------------------------------------------------------- training
    def forward(
        self, 
        doc_input_ids, 
        doc_attention_mask, 
        step_mask, 
        claim_input_ids, 
        claim_attention_mask,
        slot_claim, 
        stop_labels
    ):
        t = self.teacher(claim_input_ids, claim_attention_mask)  # (C, H)
        _, cache = self.read_doc(doc_input_ids, doc_attention_mask)
        e, stop_logits = self.steps(cache, doc_attention_mask, step_mask)

        m = slot_claim >= 0
        e, y = e[m], slot_claim[m]
        if self.match == "hungarian":
            y = hungarian_targets(e, t, y, step_mask.nonzero()[:, 0][m])
        distill = distill_loss(e, t, y, self.temperature)    # candidates: same-doc + in-batch claims
        cos = (1 - (e * t[y]).sum(-1)).mean()                 # = MSE / 2 on unit vectors
        stop = F.binary_cross_entropy_with_logits(stop_logits, stop_labels.float(),
                                                  pos_weight=stop_logits.new_tensor(self.stop_pos_weight))
        loss = self.distill_weight * distill + self.cos_weight * cos + self.stop_weight * stop
        return {"loss": loss, "distill": distill.detach(), "cos": cos.detach(), "stop": stop.detach()}

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, doc_input_ids, doc_attention_mask, max_steps=None):
        """Left-padded docs -> (doc vectors (B, H), list of (n_i, H) claim vectors).

        n_i ends at the first stop (at least 1 claim). One pass over each document; step inputs don't depend
        on earlier outputs, so all steps run in one call on its cache.
        """
        B, T = doc_input_ids.size(0), max_steps or self.cfg.max_claims + 1
        doc, cache = self.read_doc(doc_input_ids, doc_attention_mask)
        e, s = self.steps(cache, doc_attention_mask, torch.ones(B, T, dtype=torch.bool, device=doc.device))
        e, s = e.view(B, T, -1), s.view(B, T)
        claims = []
        for e_i, s_i in zip(e, s):
            hit = (s_i[1:] > 0).nonzero()
            claims.append(e_i[: int(hit[0]) + 1 if len(hit) else T])
        return doc, claims

    # ---------------------------------------------------------------- io
    def trainable_state_dict(self):
        keep = lambda k: "lora_" in k or k.startswith(("step_backbone", "step_query", "stop_head", "soft_prompt"))
        cast = lambda k, v: v.to(self.backbone.dtype) if k.startswith("step_backbone") else v  # fp32 only to train
        return {k: cast(k, v).detach().contiguous().cpu() for k, v in self.state_dict().items() if keep(k)}

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        save_file(self.trainable_state_dict(), os.path.join(out_dir, WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, ckpt_dir, dtype=torch.bfloat16, **overrides):
        with open(os.path.join(ckpt_dir, CONFIG_NAME)) as f:
            saved = json.load(f)
        known = {f.name for f in fields(ClaimEmbConfig)}  # older checkpoints carry removed options
        cfg = ClaimEmbConfig(**{k: v for k, v in {**saved, **overrides}.items() if k in known})
        model = cls(cfg, dtype)
        _, unexpected = model.load_state_dict(load_file(os.path.join(ckpt_dir, WEIGHTS_NAME)), strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        return model
