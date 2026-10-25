"""ClaimEmb v4: claim embeddings as <|endoftext|> queries that back-track to claim regions of a document.

    <doc tokens><|endoftext|>   ──(Qwen3-Embedding, frozen, no grad)──>  doc vector (exactly the index) + KV cache
    <|endoftext|> per claim     ──(same model + LoRA "claim", appended to that cache)──>  claim vectors
    tag head + CRF on the doc's token states                                         ──>  claim regions (B/I/O)

Each query is the end token the embedding model pools at; queries never see each other. How a query back-tracks
to its region [s, e] (`query`):
  mask      it attends only to the region and to token 0 (the attention sink). Untrained, with sentence regions,
            this already matches embedding each sentence alone on distinguishability.
  residual  it sees the whole doc, and its input is <|endoftext|> + b(d), d = distance back from the doc's end token
            to the region end, b = a zero-initialized linear map of sinusoidal features of d: b = 0 is the original
            end token (the doc vector).
  grid      residual queries at fixed positions every `stride` doc tokens (no regions, no locator), matched to the
            gold claims one-to-one by Hungarian matching (after a proportional warm start); a has-claim head on each
            query decides which ones are kept at inference.
Its position id is right after the region (`query_pos=region`) or the doc end token's (`end`). A residual query
doesn't see the doc's own end token, so with b = 0 and query_pos=end it reproduces the doc vector exactly.

Training uses the aligned gold regions (overlaps allowed) for the queries, and their B/I/O tags (overlaps
trimmed) for the CRF; at inference the CRF's regions drive the queries.
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
from scipy.optimize import linear_sum_assignment
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

from embedrag_v3.model import distill_loss, last_token

from .config import ClaimEmbConfig
from .crf import CRF, tags_to_spans
from .data import end_id

CONFIG_NAME, WEIGHTS_NAME = "claimemb_config.json", "claimemb_weights.safetensors"
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sinusoid(d, dim):
    """(..., ) distances -> (..., dim) sin/cos features."""
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
        self.distill_weight, self.cos_weight, self.tag_weight, self.temperature = 1.0, 1.0, 1.0, 0.05
        self.keep_weight, self.match = 1.0, "hungarian"  # grid: has-claim loss; "order" = proportional warm start

        # Define lora (active on the claim queries only; see `lora`); lora_r = 0: none, the queries run the frozen model
        if cfg.lora_r > 0:
            lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                              target_modules=LORA_MODULES, bias="none")
            inject_adapter_in_model(lora, self.backbone, adapter_name="claim")
            for n, p in self.backbone.named_parameters():
                p.requires_grad_("lora_" in n)
        self._tuners = [m for m in self.backbone.modules() if isinstance(m, BaseTunerLayer)]

        # Define the query: the end token, plus (residual) a zero-initialized back-track embedding
        self.end_id = end_id(self.tokenizer)
        self.residual = nn.Linear(cfg.residual_dim, H) if cfg.query in ("residual", "grid") else None
        self.keep_head = nn.Linear(H, 1) if cfg.query == "grid" else None
        if self.residual is not None:
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

        # Define the region locator: B/I/O emissions from the doc's token states + a linear-chain CRF
        self.tag_head = nn.Linear(H, 3)
        self.crf = CRF(3)

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

    # ---------------------------------------------------------------- locator
    def emissions(self, h, doc_attention_mask):
        """Doc token states (left-padded, end token last) -> right-padded B/I/O emissions (B, Lmax, 3), mask."""
        B, L, _ = h.shape
        n = doc_attention_mask.sum(1) - 1                                  # doc tokens per row, without the end token
        Lmax = int(n.max())
        idx = (L - 1 - n).unsqueeze(1) + torch.arange(Lmax, device=h.device)   # first doc token = L - 1 - n
        mask = torch.arange(Lmax, device=h.device) < n.unsqueeze(1)
        g = h.gather(1, idx.clamp(max=L - 2).unsqueeze(-1).expand(-1, -1, h.size(-1)))
        return self.tag_head(g.float()), mask

    @torch.no_grad()
    def predict_regions(self, h, doc_attention_mask):
        """Viterbi B/I/O -> list of [(s, e), ...] per doc (at most max_claims; the whole doc if none)."""
        em, mask = self.emissions(h, doc_attention_mask)
        out = []
        for tags, n in zip(self.crf.decode(em, mask), mask.sum(1).tolist()):
            spans = tags_to_spans(tags)[: self.cfg.max_claims]
            out.append(spans or [(0, n - 1)])
        return out

    # ---------------------------------------------------------------- claim queries
    def grid_spans(self, doc_attention_mask):
        """Grid positions p_t = min(stride * (t + 1) - 1, last doc token) as (p, p) spans (B, T, 2), with mask (B, T)."""
        n_doc = (doc_attention_mask.sum(1) - 1).tolist()
        cnt = [min(self.cfg.max_queries, -(-n // self.cfg.stride)) for n in n_doc]
        T = max(cnt)
        p = [[min(self.cfg.stride * (t + 1) - 1, n - 1) for t in range(c)] + [0] * (T - c) for n, c in zip(n_doc, cnt)]
        dev = doc_attention_mask.device
        p = torch.tensor(p, device=dev)
        return torch.stack([p, p], -1), torch.arange(T, device=dev) < torch.tensor(cnt, device=dev).unsqueeze(1)

    def queries(self, cache, doc_attention_mask, claim_spans, claim_mask, return_hidden=False, return_attentions=False):
        """One end-token query per claim region, appended to the doc cache -> claim vectors (S, H), row-major.

        return_attentions: also return per-layer attention weights (B, heads, T, L + T) (needs eager attention)."""
        B, T = claim_mask.shape
        L, dev = doc_attention_mask.size(1), doc_attention_mask.device
        n = doc_attention_mask.sum(1, keepdim=True)                        # input length incl. the end token
        off = L - n                                                        # left padding
        s, e = claim_spans[..., 0], claim_spans[..., 1]

        x = self.backbone.get_input_embeddings().weight[self.end_id].float().expand(B, T, -1)
        if self.residual is not None:
            x = x + self.residual(sinusoid((n - 1) - e, self.cfg.residual_dim))
        pos = e + 1 if self.cfg.query_pos == "region" else (n - 1).expand(B, T)

        cols = torch.arange(L, device=dev)
        if self.cfg.query == "mask":
            keep = (cols >= (off + s).unsqueeze(-1)) & (cols <= (off + e).unsqueeze(-1))
            keep |= cols == off.unsqueeze(-1)                              # token 0: the attention sink
        else:                                                              # the doc, without its own end token
            keep = (doc_attention_mask.bool() & (cols < L - 1)).unsqueeze(1).expand(B, T, L)
        keep = torch.cat([keep & claim_mask.unsqueeze(-1), torch.eye(T, dtype=torch.bool, device=dev).expand(B, T, T)], -1)
        dtype = self.backbone.dtype
        mask = torch.zeros(B, 1, T, L + T, dtype=dtype, device=dev).masked_fill(~keep.unsqueeze(1),
                                                                                torch.finfo(dtype).min)
        with self.lora(True):
            out = self.backbone(inputs_embeds=x.to(dtype), attention_mask=mask, position_ids=pos,
                                past_key_values=cache, use_cache=True, output_attentions=return_attentions)
        h = out.last_hidden_state[claim_mask].float()
        res = (F.normalize(h, dim=-1),) + ((h,) if return_hidden else ()) + ((out.attentions,) if return_attentions else ())
        return res if len(res) > 1 else res[0]

    def grid_targets(self, e, t, q_cnt, c_cnt):
        """Assign each doc's claims to its grid queries one-to-one -> (query idx, claim idx) over the flat tensors."""
        qi, ci, q0, c0 = [], [], 0, 0
        sim = (e.detach() @ t.T).float().cpu().numpy()
        for nq, nc in zip(q_cnt, c_cnt):
            if self.match == "order":                                   # claim k -> the proportional grid slot
                rows = sorted({min(nq - 1, int((k + 0.5) * nq / nc)) for k in range(nc)})
                pairs = list(zip(rows, range(len(rows))))
            else:
                r, c = linear_sum_assignment(-sim[q0:q0 + nq, c0:c0 + nc])
                pairs = list(zip(r, c))
            qi += [q0 + a for a, _ in pairs]
            ci += [c0 + b for _, b in pairs]
            q0, c0 = q0 + nq, c0 + nc
        dev = e.device
        return torch.tensor(qi, device=dev), torch.tensor(ci, device=dev)

    def grid_forward(self, t, cache, doc_attention_mask, claim_mask):
        spans, gmask = self.grid_spans(doc_attention_mask)
        e, h = self.queries(cache, doc_attention_mask, spans, gmask, return_hidden=True)
        qi, ci = self.grid_targets(e, t, gmask.sum(1).tolist(), claim_mask.sum(1).tolist())
        distill = distill_loss(e[qi], t, ci, self.temperature)
        cos = (1 - (e[qi] * t[ci]).sum(-1)).mean()
        labels = torch.zeros(len(e), device=e.device)
        labels[qi] = 1.0
        keep = F.binary_cross_entropy_with_logits(self.keep_head(h).squeeze(-1), labels)
        loss = self.distill_weight * distill + self.cos_weight * cos + self.keep_weight * keep
        return {"loss": loss, "distill": distill.detach(), "cos": cos.detach(), "tag": keep.detach()}

    # ---------------------------------------------------------------- training
    def forward(self, doc_input_ids, doc_attention_mask, claim_input_ids, claim_attention_mask,
                claim_spans, claim_mask, tags, tag_mask):
        t = self.teacher(claim_input_ids, claim_attention_mask)          # (C, H), row-major over claim_mask
        h, cache = self.read_doc(doc_input_ids, doc_attention_mask)
        if self.cfg.query == "grid":                                       # no regions: "tag" logs the has-claim BCE
            return self.grid_forward(t, cache, doc_attention_mask, claim_mask)
        e = self.queries(cache, doc_attention_mask, claim_spans, claim_mask)
        y = torch.arange(len(t), device=t.device)                         # each query is its claim's
        distill = distill_loss(e, t, y, self.temperature)
        cos = (1 - (e * t).sum(-1)).mean()
        em, mask = self.emissions(h, doc_attention_mask)
        tag = self.crf.nll(em, tags[:, : em.size(1)], mask & tag_mask[:, : em.size(1)])
        loss = self.distill_weight * distill + self.cos_weight * cos + self.tag_weight * tag
        return {"loss": loss, "distill": distill.detach(), "cos": cos.detach(), "tag": tag.detach()}

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, doc_input_ids, doc_attention_mask):
        """Left-padded docs -> (doc vectors (B, H), list of (n_i, H) claim vectors, list of regions)."""
        h, cache = self.read_doc(doc_input_ids, doc_attention_mask)
        if self.cfg.query == "grid":                                       # keep queries the has-claim head accepts
            spans, gmask = self.grid_spans(doc_attention_mask)
            e, hq = self.queries(cache, doc_attention_mask, spans, gmask, return_hidden=True)
            keep = self.keep_head(hq).squeeze(-1)
            vecs, regions, i = [], [], 0
            for b, c in enumerate(gmask.sum(1).tolist()):
                k = keep[i:i + c] > 0
                k[keep[i:i + c].argmax()] = True                          # at least one claim
                vecs.append(e[i:i + c][k])
                regions.append([tuple(x) for x in spans[b, :c][k.cpu()].tolist()])
                i += c
            return F.normalize(h[:, -1].float(), dim=-1), vecs, regions
        regions = self.predict_regions(h, doc_attention_mask)
        T = max(len(r) for r in regions)
        spans = torch.tensor([[list(x) for x in r] + [[0, 0]] * (T - len(r)) for r in regions], device=h.device)
        cmask = torch.tensor([[True] * len(r) + [False] * (T - len(r)) for r in regions], device=h.device)
        e = self.queries(cache, doc_attention_mask, spans, cmask)
        return F.normalize(h[:, -1].float(), dim=-1), list(torch.split(e, [len(r) for r in regions])), regions

    # ---------------------------------------------------------------- io
    def trainable_state_dict(self):
        keep = lambda k: "lora_" in k or k.startswith(("residual", "keep_head", "tag_head", "crf"))
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
