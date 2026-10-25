"""Linear-chain CRF over B/I/O tags for claim segmentation, and spans <-> tags.

Tags: O = 0 (no claim), B = 1 (a claim starts), I = 2 (inside a claim). B/I tags can't express overlapping
spans, so `spans_to_tags` trims each span to start after the previous one (dropping spans left empty).
"""
import torch
import torch.nn as nn

O, B, I = 0, 1, 2


def spans_to_tags(spans, length):
    """[(s, e), ...] (inclusive) -> list of `length` tags; overlaps trimmed in start order."""
    tags, prev_end = [O] * length, -1
    for s, e in sorted(spans):
        s, e = max(s, prev_end + 1), min(e, length - 1)
        if s > e:
            continue
        tags[s] = B
        for i in range(s + 1, e + 1):
            tags[i] = I
        prev_end = e
    return tags


def tags_to_spans(tags):
    """Tags -> [(s, e), ...]; an I after O starts a new span."""
    spans, start = [], None
    for i, t in enumerate(list(tags) + [O]):
        if t == B or (t == I and start is None) or t == O:
            if start is not None:
                spans.append((start, i - 1))
                start = None
            if t != O:
                start = i
    return spans


class CRF(nn.Module):
    """Linear-chain CRF on right-padded emissions (B, L, K) with mask (B, L) (valid tokens are a prefix)."""

    def __init__(self, num_tags=3):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(num_tags, num_tags))  # trans[i, j]: score of tag i -> tag j
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))

    def nll(self, emissions, tags, mask):
        """Negative log-likelihood of the gold tags per token, averaged over the batch (a per-sequence sum would
        scale with doc length and swamp the other losses)."""
        mask = mask.bool()
        return ((self._log_partition(emissions, mask) - self._score(emissions, tags, mask)) / mask.sum(1)).mean()

    def _score(self, em, tags, mask):
        Bsz, L, _ = em.shape
        ar = torch.arange(Bsz, device=em.device)
        s = self.start[tags[:, 0]] + em[ar, 0, tags[:, 0]]
        for t in range(1, L):
            step = self.trans[tags[:, t - 1], tags[:, t]] + em[ar, t, tags[:, t]]
            s = s + step * mask[:, t]
        last = tags[ar, mask.sum(1) - 1]
        return s + self.end[last]

    def _log_partition(self, em, mask):
        a = self.start + em[:, 0]                                   # (B, K)
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(a.unsqueeze(2) + self.trans + em[:, t].unsqueeze(1), dim=1)
            a = torch.where(mask[:, t:t + 1], nxt, a)
        return torch.logsumexp(a + self.end, dim=1)

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi -> list of tag lists (valid length per row)."""
        mask = mask.bool()
        score, back = self.start + emissions[:, 0], []
        for t in range(1, emissions.size(1)):
            cand = score.unsqueeze(2) + self.trans                    # (B, K_prev, K)
            best, idx = cand.max(1)
            nxt = best + emissions[:, t]
            back.append(torch.where(mask[:, t:t + 1], idx, torch.arange(idx.size(1), device=idx.device)))
            score = torch.where(mask[:, t:t + 1], nxt, score)
        score = score + self.end
        out = []
        for b in range(emissions.size(0)):
            n = int(mask[b].sum())
            tag = int(score[b].argmax())
            path = [tag]
            for t in range(n - 1, 0, -1):
                tag = int(back[t - 1][b, tag])
                path.append(tag)
            out.append(path[::-1])
        return out
