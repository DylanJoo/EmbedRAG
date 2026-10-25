"""End-to-end EmbedRAG demo on a tiny random Llama (CPU, ~2 min).

Each passage is compressed to k=4 vectors (mean pooling: one vector per chunk of tokens). We train on
`ae`: "Reconstruct passage [1]" from its 4 vectors. (`ae_multi`, picking passage [j] out of several, is
too hard for this toy model to learn in minutes; the unit tests cover its plumbing.) Passages are random
word strings drawn fresh every step, so the decoder cannot
memorize them: the only way to lower the loss is to read the passage out of its 4 vectors. At the end we
swap the passages and check that the output follows them (the counterfactual check from DESIGN.md).
(With the 8 fixed sentences an earlier version used, the loss went down while the model ignored the
vectors and recited its favourite sentence: exactly the failure that check is meant to catch.)

A real pretrained LM does this with only LoRA + projector (stage 1, decoder frozen); a tiny random LM has
no language ability for LoRA to adapt, so here all of its weights train.

  python -m tests.demo_tiny
"""
import random
import tempfile

import torch

from legacy.embedrag_v1 import MEM_ID, ChatFormat, EmbedRAG, EmbedRAGCollator, EmbedRAGConfig
from legacy.embedrag_v1.build_data import AE
from tests.tiny import WORDS, make_tiny

STEPS, N_PSG, N_WORDS = 800, 1, 4


def passage(rng):
    return " ".join(rng.choice(WORDS) for _ in range(N_WORDS))


def examples(rng, n):
    for _ in range(n):
        group = [passage(rng) for _ in range(N_PSG)]
        j = rng.randrange(N_PSG)
        yield {"task": "ae" if N_PSG == 1 else "ae_multi", "instruction": AE.format(j + 1), "passages": group, "target": group[j]}


def show(m, collator, exs):
    m.eval()
    out = m.generate(**collator(exs), max_new_tokens=N_WORDS + 2, do_sample=False)
    hits = 0
    for ex, pred in zip(exs, m.tokenizer.batch_decode(out, skip_special_tokens=True)):
        hits += pred.strip() == ex["target"]
        print(f"  {ex['instruction'][:27]} | passages={ex['passages']}\n    gold: {ex['target']}\n    pred: {pred}")
    print(f"  exact match: {hits}/{len(exs)}")


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    rng = random.Random(0)
    path = make_tiny(tempfile.mkdtemp() + "/tiny")
    m = EmbedRAG(EmbedRAGConfig(decoder=path, pooling="mean", num_mem=4, attn_implementation="eager"), torch.float32)
    tok = m.tokenizer
    train = EmbedRAGCollator(tok, m.enc_tokenizer, ChatFormat(tok), num_mem=4)
    gen = EmbedRAGCollator(tok, m.enc_tokenizer, ChatFormat(tok), num_mem=4, for_generation=True)

    b = train(list(examples(rng, 1)))
    ids = b["dec_input_ids"][0]
    print("decoder input (MEM = one compressed vector):")
    print("  " + " ".join("MEM" if i == MEM_ID else tok.convert_ids_to_tokens(int(i)) for i in ids))
    print()

    held = list(examples(random.Random(123), 4))
    print("before training:")
    show(m, gen, held)

    m.set_trainable(encoder=True, projector=True, decoder=False)
    for p in m.decoder.parameters():  # toy only: the shared backbone itself trains
        p.requires_grad_(True)
    m.projector.scale.data.fill_(1.0)  # toy only: random-init token embeddings are tiny (norm ~0.2)
    print(f"trainable params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,} "
          f"of {sum(p.numel() for p in m.parameters()):,}")
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for step in range(STEPS):
        m.train()
        loss = m(**train(list(examples(rng, 16))))["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == STEPS - 1:
            print(f"step {step:3d}  ce {loss.item():.3f}")

    print("\nafter training (unseen random passages):")
    show(m, gen, held)

    print("\ncounterfactual: same instructions, new passages -> output should follow the passages")
    rng2 = random.Random(7)
    show(m, gen, [{**ex, "passages": (ps := [passage(rng2) for _ in range(N_PSG)]),
                   "target": ps[int(ex["instruction"].split("[")[1][0]) - 1]} for ex in held])


if __name__ == "__main__":
    main()
