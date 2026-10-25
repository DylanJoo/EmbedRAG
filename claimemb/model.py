"""ClaimEmb: claim embeddings as end-token queries that point back into the document.

    <doc tokens><|endoftext|>        ──(Qwen3-Embedding, frozen, no grad)──>  doc vector (exactly the index) + cache
    <|endoftext|> + b(d_t), t = 1..K ──(same model + LoRA "claim", appended)──>  claim vectors, has-claim scores

Queries sit on a grid of pointers p_t = min(stride * t - 1, last doc token) over the doc tokens. Query t is the end
token the embedding model pools at, plus a back-track residual b(d_t) of the distance d_t from the doc's end token
back to p_t (a zero-initialized linear map of sinusoidal features). Every query sees the whole doc (but not the doc's
own end token, nor the other queries) at the doc end token's position, so with b = 0 each query is exactly the doc
vector: only the residual tells queries apart. The LoRA, active on the queries only, learns to read and compress the
text around each pointer.

Training: each doc's gold claims (embedded by the frozen backbone, the teacher) are matched to its queries one-to-one,
by proportional position for the first steps and then by Hungarian matching on cosine. Matched queries are trained
by score distillation + cosine; a has-claim head learns which queries hold a claim and decides which are kept at
inference.
"""
import contextlib
import json
import math
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
from .data import end_id

CONFIG_NAME, WEIGHTS_NAME = "claimemb_config.json", "claimemb_weights.safetensors"
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def last_token(h, mask):
    """Right-padded (B, L, H) -> hidden state at each row's last real token (B, H)."""
    return h[torch.arange(h.size(0), device=h.device), mask.sum(-1) - 1]


def distill_loss(e, t, y, temperature):
    """Listwise score distillation: KL(softmax(t_y . t / tau) || softmax(e . t / tau)), averaged over queries.

    e (S, H) query vectors, t (C, H) the batch's gold claim vectors, y (S,) each query's gold claim. The teacher
    scores are the gold claim vector's own, so the loss is 0 exactly at e = t_y.
    """
    target = F.softmax(t[y] @ t.T / temperature, dim=-1)
    return F.kl_div(F.log_softmax(e @ t.T / temperature, dim=-1), target, reduction="batchmean")


def sinusoid(d, dim):
    """(...,) distances -> (..., dim) sin/cos features."""
    f = torch.exp(-math.log(10000.0) * torch.arange(dim // 2, device=d.device) / (dim // 2))
    a = d.float().unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1)


class ClaimEmb(nn.Module):
    def __init__(self, cfg: ClaimEmbConfig, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.backbone, dtype=dtype, attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        H = self.backbone.config.hidden_size

        # Define training configs (set from TrainArgs in train.py)
        self.distill_weight, self.cos_weight, self.keep_weight, self.temperature = 1.0, 1.0, 1.0, 0.05
        self.match = "hungarian"  # or "order": proportional warm start

        # Define lora (active on the queries only; see `lora`); lora_r = 0: the queries run the frozen model
        if cfg.lora_r > 0:
            lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                              target_modules=LORA_MODULES, bias="none")
            inject_adapter_in_model(lora, self.backbone, adapter_name="claim")
            for n, p in self.backbone.named_parameters():
                p.requires_grad_("lora_" in n)
        self._tuners = [m for m in self.backbone.modules() if isinstance(m, BaseTunerLayer)]

        # Define the queries: the end token + a zero-initialized back-track residual, and the has-claim head
        self.end_id = end_id(self.tokenizer)
        self.residual = nn.Linear(cfg.residual_dim, H)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.keep_head = nn.Linear(H, 1)

    # ---------------------------------------------------------------- setup
    @contextlib.contextmanager
    def lora(self, on):
        if not self._tuners:  # lora_r = 0
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
        """Frozen backbone on left-padded docs -> (token states (B, L, H), KV cache). The doc vector is h[:, -1]."""
        pos = (doc_attention_mask.cumsum(-1) - 1).clamp(min=0)
        with self.lora(False):
            out = self.backbone(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                                position_ids=pos, use_cache=True)
        return out.last_hidden_state, out.past_key_values

    # ---------------------------------------------------------------- queries
    def pointers(self, doc_attention_mask):
        """Grid pointers p_t = min(stride * (t + 1) - 1, last doc token) (B, K), with mask (B, K)."""
        n_doc = (doc_attention_mask.sum(1) - 1).tolist()
        cnt = [min(self.cfg.max_queries, -(-n // self.cfg.stride)) for n in n_doc]
        K, dev = max(cnt), doc_attention_mask.device
        p = [[min(self.cfg.stride * (t + 1) - 1, n - 1) for t in range(c)] + [0] * (K - c) for n, c in zip(n_doc, cnt)]
        return torch.tensor(p, device=dev), torch.arange(K, device=dev) < torch.tensor(cnt, device=dev).unsqueeze(1)

    def queries(self, cache, doc_attention_mask, ptr, qmask, return_attentions=False):
        """End-token queries for pointers ptr (B, K), appended to the doc cache.

        Returns claim vectors (S, H) and hidden states (S, H) for the real queries (row-major over qmask), plus
        per-layer attention weights (B, heads, K, L + K) if return_attentions (needs eager attention)."""
        B, K = qmask.shape
        L, dev = doc_attention_mask.size(1), doc_attention_mask.device
        n = doc_attention_mask.sum(1, keepdim=True)                       # input length incl. the end token
        x = self.backbone.get_input_embeddings().weight[self.end_id].float().expand(B, K, -1)
        x = x + self.residual(sinusoid((n - 1) - ptr, self.cfg.residual_dim))
        keep = (doc_attention_mask.bool() & (torch.arange(L, device=dev) < L - 1)).unsqueeze(1).expand(B, K, L)
        keep = torch.cat([keep & qmask.unsqueeze(-1), torch.eye(K, dtype=torch.bool, device=dev).expand(B, K, K)], -1)
        dtype = self.backbone.dtype
        mask = torch.zeros(B, 1, K, L + K, dtype=dtype, device=dev).masked_fill(~keep.unsqueeze(1), torch.finfo(dtype).min)
        with self.lora(True):
            out = self.backbone(inputs_embeds=x.to(dtype), attention_mask=mask, position_ids=(n - 1).expand(B, K),
                                past_key_values=cache, use_cache=True, output_attentions=return_attentions)
        h = out.last_hidden_state[qmask].float()
        res = (F.normalize(h, dim=-1), h)
        return res + (out.attentions,) if return_attentions else res

    # ---------------------------------------------------------------- training
    def targets(self, e, t, q_cnt, c_cnt):
        """Assign each doc's claims to its queries one-to-one -> (query idx, claim idx) over the flat tensors."""
        qi, ci, q0, c0 = [], [], 0, 0
        sim = (e.detach() @ t.T).float().cpu().numpy()
        for nq, nc in zip(q_cnt, c_cnt):
            if self.match == "order":                                   # claim k -> the proportional query
                rows = sorted({min(nq - 1, int((k + 0.5) * nq / nc)) for k in range(nc)})
                pairs = list(zip(rows, range(len(rows))))
            else:
                r, c = linear_sum_assignment(-sim[q0:q0 + nq, c0:c0 + nc])
                pairs = list(zip(r, c))
            qi += [q0 + a for a, _ in pairs]
            ci += [c0 + b for _, b in pairs]
            q0, c0 = q0 + nq, c0 + nc
        return torch.tensor(qi, device=e.device), torch.tensor(ci, device=e.device)

    def forward(self, doc_input_ids, doc_attention_mask, claim_input_ids, claim_attention_mask, n_claims):
        t = self.teacher(claim_input_ids, claim_attention_mask)          # (C, H)
        _, cache = self.read_doc(doc_input_ids, doc_attention_mask)
        ptr, qmask = self.pointers(doc_attention_mask)
        e, h = self.queries(cache, doc_attention_mask, ptr, qmask)
        qi, ci = self.targets(e, t, qmask.sum(1).tolist(), n_claims.tolist())
        distill = distill_loss(e[qi], t, ci, self.temperature)
        cos = (1 - (e[qi] * t[ci]).sum(-1)).mean()
        labels = torch.zeros(len(e), device=e.device)
        labels[qi] = 1.0
        keep = F.binary_cross_entropy_with_logits(self.keep_head(h).squeeze(-1), labels)
        loss = self.distill_weight * distill + self.cos_weight * cos + self.keep_weight * keep
        return {"loss": loss, "distill": distill.detach(), "cos": cos.detach(), "keep": keep.detach()}

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, doc_input_ids, doc_attention_mask, all_queries=False):
        """Left-padded docs -> (doc vectors (B, H), list of (n_i, H) claim vectors, list of their pointers).

        Keeps the queries the has-claim head accepts (at least one per doc), or every query if all_queries."""
        h, cache = self.read_doc(doc_input_ids, doc_attention_mask)
        ptr, qmask = self.pointers(doc_attention_mask)
        e, hq = self.queries(cache, doc_attention_mask, ptr, qmask)
        logits = self.keep_head(hq).squeeze(-1)
        vecs, ptrs, i = [], [], 0
        for b, c in enumerate(qmask.sum(1).tolist()):
            k = torch.ones(c, dtype=torch.bool, device=e.device) if all_queries else logits[i:i + c] > 0
            k[logits[i:i + c].argmax()] = True
            vecs.append(e[i:i + c][k])
            ptrs.append(ptr[b, :c][k].tolist())
            i += c
        return F.normalize(h[:, -1].float(), dim=-1), vecs, ptrs

    # ---------------------------------------------------------------- io
    def trainable_state_dict(self):
        keep = lambda k: "lora_" in k or k.startswith(("residual", "keep_head"))
        return {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items() if keep(k)}

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        save_file(self.trainable_state_dict(), os.path.join(out_dir, WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, ckpt_dir, dtype=torch.bfloat16, **overrides):
        with open(os.path.join(ckpt_dir, CONFIG_NAME)) as f:
            saved = json.load(f)
        known = {f.name for f in fields(ClaimEmbConfig)}
        cfg = ClaimEmbConfig(**{k: v for k, v in {**saved, **overrides}.items() if k in known})
        model = cls(cfg, dtype)
        _, unexpected = model.load_state_dict(load_file(os.path.join(ckpt_dir, WEIGHTS_NAME)), strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        return model
