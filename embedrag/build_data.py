"""Build unified-schema jsonl (see embedrag/data.py) for each training stage.

  pretrain : raw passages {"text"|"contents"}         -> ae / multi-passage ae / continuation
  mds      : old data/compose_pairs.py output         -> summ (cited, with distractors) / rel / single
  qa       : {"question", "passages"|"ctxs", "answer"|"answers"} -> qa

Example:
  python -m embedrag.build_data pretrain --input corpus.jsonl --output data/embedrag/pretrain.jsonl
  python -m embedrag.build_data mds --input data/mds-5k-greedy-1.jsonl --output data/embedrag/mds.jsonl
"""
import argparse
import json
import random

AE = "Reconstruct passage [{}] word for word."
CONT = "Continue the text of passage [1]."
SUMM = ("Summarize the context based on the topic. For each passage, write its number (e.g. [1]) followed "
        "by a summary of the topic-relevant content, `unrelated.` if it is irrelevant, or `redundant.` "
        "if its information is already covered. Topic: {}")
REL = ("For each passage, write its number followed by `relevant.`, `unrelated.` or `redundant.` "
       "with respect to the topic. Topic: {}")
SINGLE = "Summarize passage [1] based on the topic, or write `unrelated.` if it is irrelevant. Topic: {}"
QA = "Answer the question using the context. Cite supporting passages as [i]. Question: {}"


def words(text, start, n):
    return " ".join(text.split()[start:start + n])


def build_pretrain(rows, args, rng):
    texts = [r.get("text") or r.get("contents") for r in rows]
    texts = [t for t in texts if t and len(t.split()) >= args.ctx_words // 2]
    for t in texts:
        ctx = words(t, 0, args.ctx_words)
        yield {"task": "ae", "instruction": AE.format(1), "passages": [ctx], "target": ctx}
        nxt = words(t, args.ctx_words, args.tgt_words)
        if nxt:
            yield {"task": "cont", "instruction": CONT, "passages": [ctx], "target": nxt}
        if args.multi_ae > 1:  # fine-grained: address one passage among several
            group = [ctx] + [words(x, 0, args.ctx_words) for x in rng.sample(texts, args.multi_ae - 1)]
            rng.shuffle(group)
            j = group.index(ctx)
            yield {"task": "ae_multi", "instruction": AE.format(j + 1), "passages": group, "target": ctx}


def build_mds(rows, args, rng):
    for r in rows:
        topic = r["topic"] if isinstance(r["topic"], str) else " ".join(r["topic"])
        items = [(d, " ".join(c[: args.psgs_per_doc]), "relevant")
                 for d, c in zip(r["doc_ctxs"][: args.max_docs], r["comp_ctxs"])]
        items += [(d, "unrelated.", "unrelated") for d in r.get("distract_ctxs", [])[: args.num_distractors]]
        items += [(d, "redundant.", "redundant") for d in r.get("redundant_ctxs", [])[: args.num_redundant]]
        if not items:
            continue
        rng.shuffle(items)
        passages = [d for d, _, _ in items]
        labels = {k: [i + 1 for i, (_, _, l) in enumerate(items) if l == k]
                  for k in ("relevant", "unrelated", "redundant")}
        meta = {**labels, "reference": " ".join(f"[{i + 1}] {s}" for i, (_, s, _) in enumerate(items))}

        summ = [seg for i, (_, s, _) in enumerate(items) for seg in ([f"[{i + 1}] ", args.cite_weight], [s + " ", 1.0])]
        yield {"task": "summ", "instruction": SUMM.format(topic), "passages": passages, "target": summ, "meta": meta}

        rel = [seg for i, (_, _, l) in enumerate(items) for seg in ([f"[{i + 1}] ", args.cite_weight], [l + ". ", 1.0])]
        yield {"task": "rel", "instruction": REL.format(topic), "passages": passages, "target": rel, "meta": meta}

        for d, s, l in items:
            if l != "redundant":  # redundancy is undefined for a single passage
                yield {"task": "single", "instruction": SINGLE.format(topic), "passages": [d], "target": s}


def build_qa(rows, args, rng):
    for r in rows:
        ctxs = r.get("passages") or [c.get("text", c) if isinstance(c, dict) else c for c in r.get("ctxs", [])]
        answers = r.get("answers") or [r["answer"]]
        yield {"task": "qa", "instruction": QA.format(r["question"]), "passages": ctxs[: args.max_docs],
               "target": answers[0], "meta": {"answers": answers}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["pretrain", "mds", "qa"])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ctx_words", type=int, default=128)
    p.add_argument("--tgt_words", type=int, default=64)
    p.add_argument("--multi_ae", type=int, default=4)
    p.add_argument("--max_docs", type=int, default=5)
    p.add_argument("--psgs_per_doc", type=int, default=1)
    p.add_argument("--num_distractors", type=int, default=2)
    p.add_argument("--num_redundant", type=int, default=1)
    p.add_argument("--cite_weight", type=float, default=2.0, help="loss weight on [i] citation tokens")
    args = p.parse_args()

    rng = random.Random(args.seed)
    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    build = {"pretrain": build_pretrain, "mds": build_mds, "qa": build_qa}[args.mode]
    n = 0
    with open(args.output, "w") as f:
        for ex in build(rows, args, rng):
            f.write(json.dumps(ex) + "\n")
            n += 1
    print(f"wrote {n} examples -> {args.output}")


if __name__ == "__main__":
    main()
