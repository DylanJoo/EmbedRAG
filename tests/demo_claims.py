"""Can a frozen LLM read a claim back from its single pre-computed claim embedding?

Option 2 of using claim embeddings as EmbedRAG memory: skip the passage encoder and project each
claim's retrieval vector (tevatron index, Qwen3-Embedding-0.6B) straight into the decoder as one soft
token. Only the projector trains; the decoder (Qwen3-0.6B) stays frozen. Prints gold / before / after
for training claims and held-out claims.

  python -m tests.demo_claims --reps '~/scratch/ragtime2/Qwen3-Embedding-0.6B/claims_emb/claims_emb.eng-docs-0.pkl' \
      --claims ~/scratch/ragtime2/claims_flat/eng-docs.claims.jsonl.gz
"""
import argparse
import gzip
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F

from legacy.embedrag_v1 import ChatFormat, EmbedRAG, EmbedRAGCollator, EmbedRAGConfig
from legacy.embedrag_v1.build_data import AE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_subset(reps_path, claims_path, n, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["vecs"], list(z["ids"]), list(z["texts"])
    print(f"loading {reps_path} (large pickle, ~1-2 min)", flush=True)
    with open(os.path.expanduser(reps_path), "rb") as f:
        reps, lookup = pickle.load(f)
    reps, lookup = np.asarray(reps)[: 4 * n], list(lookup)[: 4 * n]
    want, texts = {cid: i for i, cid in enumerate(lookup)}, {}
    with gzip.open(os.path.expanduser(claims_path), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                texts[r["id"]] = r["text"]
                if len(texts) == len(want):
                    break
    # keep claims of a readable length
    keep = [i for i, cid in enumerate(lookup) if cid in texts and 6 <= len(texts[cid].split()) <= 30][:n]
    vecs, ids = reps[keep].astype(np.float32), [lookup[i] for i in keep]
    texts = [texts[c] for c in ids]
    np.savez(cache, vecs=vecs, ids=np.array(ids, dtype=object), texts=np.array(texts, dtype=object))
    return vecs, ids, texts


def batch(collator, texts, gen=False):
    exs = [{"instruction": AE.format(1), "passages": [t], "target": t} for t in texts]
    b = (collator.gen if gen else collator)(exs)
    b.pop("enc_input_ids"), b.pop("enc_attention_mask")  # the claim vector replaces the encoder
    return {k: v.to(DEVICE) for k, v in b.items()}


def loss_fn(m, b, mem):
    h = m.decoder_hidden(m.embed_decoder_inputs(b["dec_input_ids"], mem), b["dec_attention_mask"])
    logits, y, _ = m.target_logits(h, b["labels"])
    return F.cross_entropy(logits, y)


@torch.no_grad()
def decode(m, collator, vecs, texts):
    m.eval()
    b = batch(collator, texts, gen=True)
    mem = m.projector(vecs)[:, None]
    out = m.decoder.generate(inputs_embeds=m.embed_decoder_inputs(b["dec_input_ids"], mem),
                             attention_mask=b["dec_attention_mask"], max_new_tokens=48, do_sample=False,
                             pad_token_id=m.tokenizer.pad_token_id)
    return [s.strip() for s in m.tokenizer.batch_decode(out, skip_special_tokens=True)]


def token_f1(a, b):
    a, b = a.lower().split(), b.lower().split()
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    return 0.0 if not common else 2 * common / (len(a) + len(b))


def report(title, gold, before, after):
    print(f"\n===== {title} =====")
    for g, x, y in zip(gold, before, after):
        print(f"gold  : {g}\nbefore: {x}\nafter : {y}\n")
    f = lambda preds: np.mean([token_f1(p, g) for p, g in zip(preds, gold)])
    print(f"token-F1 vs gold   before {f(before):.2f}   after {f(after):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", required=True)
    p.add_argument("--claims", required=True)
    p.add_argument("--decoder", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_show", type=int, default=6)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cache", default="claims_subset.npz")
    args = p.parse_args()

    torch.manual_seed(0)
    vecs, ids, texts = load_subset(args.reps, args.claims, args.n_train + 64, args.cache)
    vecs = torch.tensor(vecs, device=DEVICE)
    train_idx, test_idx = list(range(args.n_train)), list(range(args.n_train, len(texts)))
    print(f"{len(train_idx)} train claims, {len(test_idx)} held-out, vector dim {vecs.shape[1]}")

    # pooling=mean -> no memory slots; no LoRA; the encoder is never called (claim vectors instead)
    m = EmbedRAG(EmbedRAGConfig(decoder=args.decoder, pooling="mean", num_mem=1, encoder_lora=False),
                 torch.bfloat16).to(DEVICE)
    m.projector.float()
    assert m.projector.ln.normalized_shape[0] == vecs.shape[1], "decoder hidden size != claim vector dim"
    m.set_trainable(encoder=False, projector=True, decoder=False)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable: projector only, {n_train / 1e6:.1f}M params; decoder frozen")

    chat = ChatFormat(m.tokenizer)
    collator = EmbedRAGCollator(m.tokenizer, m.tokenizer, chat, num_mem=1, max_target_len=64)
    collator.gen = EmbedRAGCollator(m.tokenizer, m.tokenizer, chat, num_mem=1, for_generation=True)

    show_tr, show_te = train_idx[: args.n_show], test_idx[: args.n_show]
    before_tr = decode(m, collator, vecs[show_tr], [texts[i] for i in show_tr])
    before_te = decode(m, collator, vecs[show_te], [texts[i] for i in show_te])

    opt = torch.optim.AdamW(m.projector.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    rng = random.Random(0)
    for step in range(args.steps):
        m.train()
        idx = rng.sample(train_idx, args.bsz)
        b = batch(collator, [texts[i] for i in idx])
        loss = loss_fn(m, b, m.projector(vecs[idx])[:, None])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                bt = batch(collator, [texts[i] for i in test_idx[:32]])
                test = loss_fn(m.eval(), bt, m.projector(vecs[test_idx[:32]])[:, None]).item()
            print(f"step {step:4d}  train ce {loss.item():.3f}  held-out ce {test:.3f}", flush=True)

    report("TRAINING claims", [texts[i] for i in show_tr], before_tr,
           decode(m, collator, vecs[show_tr], [texts[i] for i in show_tr]))
    report("HELD-OUT claims (never trained on)", [texts[i] for i in show_te], before_te,
           decode(m, collator, vecs[show_te], [texts[i] for i in show_te]))


if __name__ == "__main__":
    main()
