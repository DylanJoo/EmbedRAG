"""Generate with a trained EmbedRAG checkpoint and score with the RL rewards.

  python -m embedrag.infer --ckpt ckpt/stage2 --input data/embedrag/mds.test.jsonl --output out.jsonl
"""
import argparse
import json

import torch

from .backbone import ChatFormat
from .data import EmbedRAGCollator
from .model import EmbedRAG
from .rewards import REWARDS, compute_reward

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_passage_len", type=int, default=256)
    p.add_argument("--no_chat_template", action="store_true")
    args = p.parse_args()

    model = EmbedRAG.from_pretrained(args.ckpt).to(DEVICE).eval()
    tok = model.tokenizer
    collator = EmbedRAGCollator(tok, model.enc_tokenizer, ChatFormat(tok, not args.no_chat_template),
                                num_mem=model.cfg.num_mem, max_passage_len=args.max_passage_len,
                                for_generation=True)
    data = [json.loads(l) for l in open(args.input) if l.strip()]
    weights = {k: 1.0 for k in REWARDS}
    with open(args.output, "w") as f:
        for s in range(0, len(data), args.batch_size):
            exs = data[s:s + args.batch_size]
            batch = {k: v.to(DEVICE) for k, v in collator(exs).items()}
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
            for ex, text in zip(exs, tok.batch_decode(out, skip_special_tokens=True)):
                _, scores = compute_reward(text, ex, weights)
                f.write(json.dumps({**ex, "prediction": text, "scores": scores}) + "\n")


if __name__ == "__main__":
    main()
