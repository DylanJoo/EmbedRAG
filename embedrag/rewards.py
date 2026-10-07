"""Rule-based rewards for RL. Each returns a value in [0, 1] or None when not applicable to the example.

Uses `meta` from build_data.py: relevant / unrelated / redundant passage ids, reference, answers.
An LLM-judge reward (faithfulness to the passages) can be added with the same signature.
"""
import re
from collections import Counter

SEG_RE = re.compile(r"\[(\d+)\]([^\[]*)")


def parse_segments(text):
    """'[1] foo [2] unrelated.' -> {1: 'foo', 2: 'unrelated.'} (first occurrence wins)."""
    out = {}
    for i, seg in SEG_RE.findall(text):
        out.setdefault(int(i), seg.strip())
    return out


def predicted_label(seg):
    s = seg.lower()
    return "unrelated" if s.startswith("unrelated") else "redundant" if s.startswith("redundant") else "relevant"


def label_accuracy(text, meta):
    """Per-passage decision (relevant / unrelated / redundant) accuracy."""
    gold = {i: k for k in ("relevant", "unrelated", "redundant") for i in meta.get(k, [])}
    if not gold:
        return None
    segs = parse_segments(text)
    return sum(i in segs and predicted_label(segs[i]) == k for i, k in gold.items()) / len(gold)


def citation_f1(text, meta):
    """Cited-as-relevant passages vs. gold relevant passages; penalizes citing non-existent ids."""
    if "relevant" not in meta:
        return None
    gold = set(meta["relevant"])
    cited = {i for i, seg in parse_segments(text).items() if predicted_label(seg) == "relevant"}
    if not gold and not cited:
        return 1.0
    tp = len(gold & cited)
    return 0.0 if tp == 0 else 2 * tp / (len(gold) + len(cited))


def token_f1(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    common = sum((Counter(p) & Counter(r)).values())
    return 0.0 if common == 0 else 2 * common / (len(p) + len(r))


def answer_f1(text, meta):
    refs = meta.get("answers") or ([meta["reference"]] if "reference" in meta else None)
    return max(token_f1(text, r) for r in refs) if refs else None


def length_ok(text, meta, max_words=300):
    return 1.0 if len(text.split()) <= max_words else 0.0


REWARDS = {"label": label_accuracy, "cite": citation_f1, "f1": answer_f1, "len": length_ok}


def compute_reward(text, example, weights):
    """Weighted mean over the applicable rewards; returns (total, per-reward dict)."""
    meta = example.get("meta") or {}
    parts = {k: REWARDS[k](text, meta) for k, w in weights.items() if w > 0}
    parts = {k: v for k, v in parts.items() if v is not None}
    total = sum(weights[k] * v for k, v in parts.items()) / max(sum(weights[k] for k in parts), 1e-6)
    return total, parts
