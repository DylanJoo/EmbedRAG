"""Unit tests for the main ClaimEmb package (grid end-token queries) on a tiny random Qwen3 (CPU).

  python -m unittest tests.test_claimemb_grid -v
"""
import tempfile
import unittest

import torch
import torch.nn.functional as F

from claimemb import ClaimCollator, ClaimEmb, ClaimEmbConfig
from claimemb.evaluate import free_run_metrics, match_metrics
from tests.test_claimemb import DOCS, make_tiny


class TinyCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = make_tiny(cls.tmp.name + "/tiny")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def model(self, **kw):
        torch.manual_seed(0)
        return ClaimEmb(ClaimEmbConfig(**{"backbone": self.path, "max_claims": 4, "lora_dropout": 0.0, "stride": 3,
                                          **kw}), torch.float32).eval()

    def batch(self, m):
        return ClaimCollator(m.tokenizer, 4)(DOCS)


class TestQueries(TinyCase):
    def test_pointers(self):
        m = self.model()
        b = self.batch(m)
        ptr, qmask = m.pointers(b["doc_attention_mask"])
        for row, n in enumerate((b["doc_attention_mask"].sum(1) - 1).tolist()):
            self.assertEqual(ptr[row, qmask[row]].tolist(), [min(3 * (t + 1) - 1, n - 1) for t in range(-(-n // 3))])

    def test_zero_residual_is_the_doc_vector(self):
        m = self.model()
        b = self.batch(m)
        h, cache = m.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
        ptr, qmask = m.pointers(b["doc_attention_mask"])
        e, _ = m.queries(cache, b["doc_attention_mask"], ptr, qmask)
        torch.testing.assert_close(e, F.normalize(h[:, -1], dim=-1)[qmask.nonzero()[:, 0]], atol=1e-5, rtol=1e-4)

    def test_queries_do_not_see_each_other(self):
        m = self.model()
        with torch.no_grad():
            m.residual.weight.normal_()
        b = self.batch(m)
        run = lambda p: m.queries(m.read_doc(b["doc_input_ids"], b["doc_attention_mask"])[1], b["doc_attention_mask"],
                                  p, qmask)[0]
        ptr, qmask = m.pointers(b["doc_attention_mask"])
        e1 = run(ptr)
        ptr2 = ptr.clone()
        ptr2[1, 0] = 1                                                    # move doc 1's first pointer only
        e2 = run(ptr2)
        row = int(qmask[0].sum())
        keep = [i for i in range(len(e1)) if i != row]
        torch.testing.assert_close(e1[keep], e2[keep])
        self.assertFalse(torch.allclose(e1[row], e2[row]))


class TestTraining(TinyCase):
    def test_targets(self):
        m = self.model()
        t = torch.eye(4)                                                  # doc 0: 3 queries / 2 claims; doc 1: 2 / 2
        e = torch.stack([t[2], t[0], t[1], t[3], t[2]])
        m.match = "hungarian"
        self.assertEqual(sorted(zip(*[x.tolist() for x in m.targets(e, t, [3, 2], [2, 2])])),
                         [(1, 0), (2, 1), (3, 3), (4, 2)])
        m.match = "order"
        self.assertEqual(list(zip(*[x.tolist() for x in m.targets(e, t, [3, 2], [2, 2])])),
                         [(0, 0), (2, 1), (3, 2), (4, 3)])

    def test_loss_grads_save_load(self):
        for r in (16, 0):
            m = self.model(lora_r=r).train()
            b = self.batch(m)
            for match in ("order", "hungarian"):
                m.match = match
                out = m(**b)
                self.assertTrue(all(torch.isfinite(out[k]) for k in ("loss", "distill", "cos", "keep")))
                out["loss"].backward()
            grads = {n.split(".")[0] if "lora_" not in n else "lora" for n, p in m.named_parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0}
            self.assertEqual(grads, {"residual", "keep_head"} | ({"lora"} if r else set()))
            m.eval()
            with tempfile.TemporaryDirectory() as d:
                m.save(d)
                rl = ClaimEmb.from_pretrained(d, torch.float32).eval()
                v1, v2 = rl.generate(b["doc_input_ids"], b["doc_attention_mask"]), m.generate(b["doc_input_ids"], b["doc_attention_mask"])
                torch.testing.assert_close(torch.cat(v1[1]), torch.cat(v2[1]))
                self.assertEqual(v1[2], v2[2])
                self.assertTrue(all(len(v) >= 1 for v in v1[1]))

    def test_overfits(self):
        m = self.model().train()
        b = self.batch(m)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3)
        first = m(**b)
        for _ in range(60):
            opt.zero_grad()
            m(**b)["loss"].backward()
            opt.step()
        last = m.eval()(**b)
        self.assertLess(last["cos"], first["cos"] * 0.5)
        self.assertLess(last["keep"], first["keep"])


class TestMetrics(unittest.TestCase):
    def test_perfect_vectors(self):
        pool = torch.eye(5)
        vecs, ranges = [pool[:2], pool[2:]], [(0, 2), (2, 5)]
        m = match_metrics(vecs, pool, ranges)
        self.assertEqual((m["match_cos"], m["match_R@1"], m["matched_frac"]), (1.0, 1.0, 1.0))
        self.assertEqual(free_run_metrics(vecs, pool, torch.tensor([0, 0, 1, 1, 1]), ranges)["claim_hit@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
