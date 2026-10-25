"""How does a plain Qwen3 decoder attend to the document while writing claims, and do the [CLM] steps do the same?

Decoder pass: the frozen student (LoRA off = plain Qwen3-0.6B) reads doc_template(doc) followed by the gold claims
teacher-forced as a bullet list ("-" + " claim" + "\\n" per claim). Two kinds of query rows per claim:
  first  the "-" token, which predicts the claim's first word (the role a [CLM] step plays)
  body   the claim's own tokens (each predicts the next word: copying / paraphrasing)
Step pass (with --ckpt, and for the untrained init): the [CLM] steps on the same prompt, matched to the gold claims
by Hungarian matching, as rows "step".

The doc is split into sentences ("blocks"); each claim's source sentence = max word-F1 with the claim. Per layer,
averaged over heads, rows and docs:
  mass_sink / mass_template / mass_doc / mass_prev / mass_own
              share of attention on the first token, the template, the doc, earlier claims (decoder: their tokens;
              steps: earlier steps), and the row's own claim so far (decoder: its "-" and tokens; steps: itself)
  src_share   share of the row's doc attention that falls on its claim's source sentence
  src_lift    src_share / (source sentence length / doc length): 1 = no preference
  src_top1    the source sentence gets the most doc attention (head-averaged); chance = mean 1/#sentences
Per head: src_top1 per (layer, head), to find "locator" (first) and "copy" (body) heads.
Order: Spearman between claim index and the expected sentence index of its attention (all layers/heads),
next to the same for the source sentences themselves (how ordered the gold claims are).

  python -m embedrag_v3.analyze_decoder --ckpt scratch/claimemb3-n30k/r16-s42 \\
      --eval_file data/claimemb/small.dev.jsonl --output scratch/claimemb3-n30k/decoder_attention.json
"""
import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from scipy.stats import spearmanr

from .config import QWEN3, QWEN3_EMB, ClaimEmbConfig
from .model import ClaimEmb, hungarian_targets

ROWS = ["first", "body", "step_init", "step"]
METRICS = ["mass_sink", "mass_template", "mass_doc", "mass_prev", "mass_own", "src_share", "src_lift", "src_top1"]
WORD = re.compile(r"\w+")


def word_f1(a, b):
    a, b = Counter(WORD.findall(a.lower())), Counter(WORD.findall(b.lower()))
    common = sum((a & b).values())
    if not common:
        return 0.0
    p, r = common / sum(b.values()), common / sum(a.values())
    return 2 * p * r / (p + r)


def build(tok, template, ex, max_doc_len, max_claims):
    """Token ids and spans: template prefix, doc sentence by sentence, template suffix, then the claims."""
    before, after = (tok(s, add_special_tokens=False)["input_ids"] for s in template.split("{doc}"))
    sents, doc, spans = [], [], []
    for i, s in enumerate(re.split(r"(?<=[.!?])\s+", ex["text"].strip())):
        ids = tok((" " if i else "") + s, add_special_tokens=False)["input_ids"]
        if not ids or len(doc) + len(ids) > max_doc_len:
            break
        spans.append((len(before) + len(doc), len(before) + len(doc) + len(ids)))
        sents.append(s)
        doc += ids
    prompt = before + doc + after
    claims = ex["claims"][:max_claims]
    dash, nl = tok("-", add_special_tokens=False)["input_ids"], tok("\n", add_special_tokens=False)["input_ids"]
    ids, claim_spans = list(prompt), []
    for c in claims:
        lo = len(ids)
        ids += dash + tok(" " + c, add_special_tokens=False)["input_ids"]
        claim_spans.append((lo, lo + len(dash), len(ids)))  # (dash, first claim token, end)
        ids += nl
    src = [int(np.argmax([word_f1(c, s) for s in sents])) for c in claims]
    src_f1 = [max(word_f1(c, s) for s in sents) for c in claims]
    return dict(prompt=prompt, ids=ids, doc=(len(before), len(before) + len(doc)), sent_spans=spans,
                sents=sents, claims=claims, claim_spans=claim_spans, src=src, src_f1=src_f1)


def sentence_shares(a, b):
    """a (H, R, K) attention rows -> (H, R, S) share of each row's doc attention per sentence, and doc mass (H, R)."""
    lo, hi = b["doc"]
    doc = a[..., lo:hi]
    mass = doc.sum(-1)
    seg = torch.cat([torch.full((e - s,), i, device=a.device) for i, (s, e) in enumerate(b["sent_spans"])])
    S = len(b["sent_spans"])
    sums = torch.zeros(*doc.shape[:-1], S, device=a.device).index_add_(-1, seg, doc)
    return sums / mass.unsqueeze(-1).clamp_min(1e-12), mass


def row_metrics(a, b, rows, own, prev, ks):
    """a (H, Rall, K); rows: query indices; own/prev: per row lists of key ranges; ks: claim index per row."""
    a = a[:, rows]
    share, mass = sentence_shares(a, b)
    lo, hi = b["doc"]
    sizes = torch.tensor([e - s for s, e in b["sent_spans"]], device=a.device, dtype=torch.float) / (hi - lo)
    src = torch.tensor([b["src"][k] for k in ks], device=a.device)
    r = torch.arange(len(rows), device=a.device)
    src_share = share[:, r, src]                                 # (H, R)
    span_mass = lambda rngs: torch.stack([a[:, i, s:e].sum(-1) if e > s else a.new_zeros(a.size(0))
                                          for i, (s, e) in enumerate(rngs)], 1)
    hm = share.mean(0)                                           # head-averaged (R, S)
    out = {
        "mass_sink": a[..., 0].mean(), "mass_doc": mass.mean(),
        "mass_template": (a[..., :len(b["prompt"])].sum(-1) - mass - a[..., 0]).mean(),
        "mass_prev": span_mass(prev).mean(), "mass_own": span_mass(own).mean(),
        "src_share": src_share.mean(), "src_lift": (src_share / sizes[src]).mean(),
        "src_top1": (hm.argmax(-1) == src).float().mean(),
    }
    head_top1 = (share.argmax(-1) == src).float().mean(-1)     # (H,)
    return {k: float(v) for k, v in out.items()}, head_top1.cpu().numpy(), hm


def capture(net):
    """Run `net` with output_attentions, keep them in box["att"]."""
    orig, box = net.forward, {}

    def forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        out = orig(*args, **kwargs)
        box["att"] = out.attentions
        return out
    net.forward = forward
    return box, lambda: delattr(net, "forward")


@torch.no_grad()
def analyze_doc(model, b, init_model=None):
    """Decoder rows (first, body) from `model`'s plain student; step rows from `init_model` (untrained) and from
    `model` when it is a different, trained model."""
    dev = model.step_query.device
    res, n = {}, len(b["claims"])
    ids = torch.tensor([b["ids"]], device=dev)
    with model.lora(False):
        att = model.student(input_ids=ids, output_attentions=True).attentions
    firsts = [s[0] for s in b["claim_spans"]]
    bodies = [(i, k) for k, (d0, c0, e) in enumerate(b["claim_spans"]) for i in range(c0, e)]
    prev_f = [(b["claim_spans"][0][0], d0) for d0, _, _ in b["claim_spans"]]   # earlier claims (incl. newlines)
    own_f = [(d0, d0 + 1) for d0, _, _ in b["claim_spans"]]
    rows_b, ks_b = [i for i, _ in bodies], [k for _, k in bodies]
    prev_b = [prev_f[k] for k in ks_b]
    own_b = [(b["claim_spans"][k][0], i + 1) for i, k in bodies]
    for name, rows, own, prev, ks in [("first", firsts, own_f, prev_f, list(range(n))),
                                      ("body", rows_b, own_b, prev_b, ks_b)]:
        res[name] = [row_metrics(a[0].float(), b, rows, own, prev, ks) for a in att]

    # [CLM] steps on the same prompt, matched to the gold claims
    for name, m in [("step_init", init_model), ("step", model if model is not init_model else None)]:
        if m is None:
            continue
        p = torch.tensor([b["prompt"]], device=dev)
        mask = torch.ones_like(p)
        cache = m.read_doc(p, mask)
        net = m.student if m.step_student is None else m.step_student
        box, restore = capture(net)
        try:
            e, _ = m.steps(cache, mask, torch.ones(1, n, dtype=torch.bool, device=dev))
        finally:
            restore()
        t = m.teacher(**{k.replace("claim_", ""): v.to(dev) for k, v in teacher_inputs(m, b["claims"]).items()})
        y = hungarian_targets(e, t, torch.arange(n, device=dev), torch.zeros(n, dtype=torch.long, device=dev)).tolist()
        P = len(b["prompt"])
        res[name] = [row_metrics(a[0].float(), b, list(range(n)), [(P + i, P + i + 1) for i in range(n)],
                                 [(P, P + i) for i in range(n)], y) for a in box["att"]]
        res[name + "_y"] = y
    return res


def teacher_inputs(m, claims):
    from .data import ClaimCollator
    return ClaimCollator(m.tokenizer, m.teacher_tokenizer, m.cfg.doc_template).claims(claims)


def expected_sentence(hms):
    """hms: per layer (R, S) head-averaged shares -> (R,) expected sentence index (all layers averaged)."""
    h = torch.stack(hms).mean(0)
    return (h * torch.arange(h.size(1), device=h.device)).sum(-1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="v3 checkpoint (its student is the decoder; omit = untrained)")
    p.add_argument("--student", default=QWEN3, help="without --ckpt (a local path when offline)")
    p.add_argument("--teacher", default=QWEN3_EMB)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--num_docs", type=int, default=100)
    p.add_argument("--max_doc_len", type=int, default=1024)
    p.add_argument("--n_examples", type=int, default=4, help="docs whose claim x sentence maps are saved")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClaimEmb.from_pretrained(args.ckpt).to(dev).eval() if args.ckpt else None
    # untrained model on the same (local) student/teacher: hub ids fail offline in transformers 4.57
    cfg = ClaimEmbConfig(**{k: v for k, v in vars(model.cfg).items()}) if model else ClaimEmbConfig(args.student, args.teacher)
    cfg.step_weights = "lora"
    init = ClaimEmb(cfg).to(dev).eval()
    model = model or init
    for m in {id(init): init, id(model): model}.values():
        m.student.config._attn_implementation = "eager"  # sdpa returns no attention weights
        if m.step_student is not None:
            m.step_student.config._attn_implementation = "eager"

    with open(args.eval_file) as f:
        docs = [json.loads(l) for _, l in zip(range(args.num_docs), f)]
    acc = {r: {m: [] for m in METRICS} for r in ROWS}
    heads = {r: [] for r in ROWS}
    order = {r: [] for r in ["first", "body", "step", "gold_src"]}
    chance, f1s, examples = [], [], []
    for di, ex in enumerate(docs):
        b = build(model.tokenizer, model.cfg.doc_template, ex, args.max_doc_len, model.cfg.max_claims)
        if len(b["claims"]) < 2 or len(b["sents"]) < 2:
            continue
        res = analyze_doc(model, b, init_model=init)
        chance.append(1 / len(b["sents"]))
        f1s += b["src_f1"]
        for r in ROWS:
            if r not in res:
                continue
            for m in METRICS:
                acc[r][m].append([lay[0][m] for lay in res[r]])
            heads[r].append(np.stack([lay[1] for lay in res[r]]))
        # order: claim index vs expected attended sentence (decoder rows by claim; steps by matched claim)
        n = len(b["claims"])
        order["gold_src"].append(spearmanr(range(n), b["src"])[0])
        order["first"].append(spearmanr(range(n), expected_sentence([lay[2] for lay in res["first"]]))[0])
        if "step" in res:
            ex_s = expected_sentence([lay[2] for lay in res["step"]])
            order["step"].append(spearmanr(res["step_y"], ex_s)[0])
        if len(examples) < args.n_examples:
            mean_map = lambda r: torch.stack([lay[2] for lay in res[r]]).mean(0).cpu().numpy().round(4).tolist()
            examples.append({"doc": di, "sents": b["sents"], "claims": b["claims"], "src": b["src"],
                             "first": mean_map("first"),
                             "step": mean_map("step") if "step" in res else None,
                             "step_y": res.get("step_y")})
        if di % 20 == 0:
            print(f"doc {di}", flush=True)

    out = {"n_docs": len(chance), "chance_top1": float(np.mean(chance)), "src_f1_mean": float(np.mean(f1s)),
           "layers": {r: {m: np.nanmean(acc[r][m], 0).tolist() for m in METRICS} for r in ROWS if acc[r]["mass_doc"]},
           "head_top1": {r: np.mean(heads[r], 0).tolist() for r in ROWS if heads[r]},
           "order_spearman": {r: float(np.nanmean(v)) for r, v in order.items() if v},
           "examples": examples, "ckpt": args.ckpt}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)

    print(f"\n{out['n_docs']} docs; chance src_top1 {out['chance_top1']:.3f}; mean source word-F1 {out['src_f1_mean']:.3f}")
    L = len(out["layers"]["first"]["mass_doc"])
    show = sorted({0, L // 4, L // 2, 3 * L // 4, L - 1})
    for m in METRICS:
        print(f"\n{m:14s}" + "".join(f"  L{l:<5d}" for l in show) + "  mean")
        for r, v in out["layers"].items():
            print(f"  {r:12s}" + "".join(f"  {v[m][l]:.4f}" for l in show) + f"  {np.mean(v[m]):.4f}")
    for r, h in out["head_top1"].items():
        h = np.array(h)
        top = np.dstack(np.unravel_index(np.argsort(-h, axis=None)[:5], h.shape))[0]
        print(f"top src_top1 heads [{r}]: " + ", ".join(f"L{l}H{k}={h[l, k]:.2f}" for l, k in top))
    print("order spearman:", {k: round(v, 3) for k, v in out["order_spearman"].items()})


if __name__ == "__main__":
    main()
