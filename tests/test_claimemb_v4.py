"""Unit tests for ClaimEmb v4 (back-tracking end-token queries + CRF locator) on a tiny random Qwen3 (CPU).

  python -m unittest tests.test_claimemb_v4 -v
"""
import itertools
import json
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from embedrag_v4 import ClaimCollator, ClaimEmb, ClaimEmbConfig
from embedrag_v4.align import align_doc, best_span
from embedrag_v4.crf import CRF, spans_to_tags, tags_to_spans
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
        return ClaimEmb(ClaimEmbConfig(**{"backbone": self.path, "max_claims": 4, "lora_dropout": 0.0, **kw}),
                        torch.float32).eval()

    def docs(self, m):
        return [{**d, "spans": [list(s) for s in align_doc(m.tokenizer, d["text"], d["claims"], 1024, 32)]}
                for d in DOCS]


class TestCRF(unittest.TestCase):
    def test_tags_and_spans(self):
        self.assertEqual(spans_to_tags([(1, 3), (2, 5), (3, 4)], 7), [0, 1, 2, 2, 1, 2, 0])  # trimmed, (3, 4) dropped
        self.assertEqual(tags_to_spans([0, 1, 2, 2, 1, 2, 0, 2]), [(1, 3), (4, 5), (7, 7)])

    def test_partition_and_viterbi_match_brute_force(self):
        torch.manual_seed(0)
        crf = CRF(3)
        with torch.no_grad():
            for p in crf.parameters():
                p.normal_()
        em, mask = torch.randn(2, 4, 3), torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]).bool()
        for b in range(2):
            n = int(mask[b].sum())
            paths = list(itertools.product(range(3), repeat=n))
            def score(path):
                s = crf.start[path[0]] + em[b, 0, path[0]] + crf.end[path[-1]]
                return s + sum(crf.trans[path[t - 1], path[t]] + em[b, t, path[t]] for t in range(1, n))
            scores = torch.stack([score(pth) for pth in paths])
            gold = torch.tensor([list(paths[0]) + [0] * (4 - n)])
            nll = crf.nll(em[b:b + 1], gold, mask[b:b + 1])
            torch.testing.assert_close(nll, (torch.logsumexp(scores, 0) - scores[0]) / n, atol=1e-5, rtol=1e-5)
            self.assertEqual(crf.decode(em[b:b + 1], mask[b:b + 1])[0], list(paths[int(scores.argmax())]))


class TestAlign(TinyCase):
    def test_best_span_and_doc(self):
        self.assertEqual(best_span(np.array([0, 1, 1, 0, 1, 0, 0.]), 3, 8), (1, 4))
        self.assertIsNone(best_span(np.zeros(4), 3, 8))
        m = self.model()
        d = self.docs(m)[1]
        ids = m.tokenizer(d["text"], add_special_tokens=False)["input_ids"]
        for c, (s, e) in zip(d["claims"], d["spans"]):
            self.assertEqual(m.tokenizer.decode(ids[s:e + 1]), c)  # tiny docs contain each claim verbatim


class TestModel(TinyCase):
    def test_mask_query_is_the_end_token_on_its_prefix(self):
        """mask query on region [0, e] at position e + 1 = the plain model on <doc[:e+1]><end> (LoRA B = 0)."""
        m = self.model(query="mask")
        b = ClaimCollator(m.tokenizer, 4)(self.docs(m))
        b["claim_spans"][0, 0] = torch.tensor([0, 2])                   # first query of each doc: region [0, e]
        b["claim_spans"][1, 0] = torch.tensor([0, 4])
        h, cache = m.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
        e = m.queries(cache, b["doc_attention_mask"], b["claim_spans"], b["claim_mask"])
        ref = AutoModel.from_pretrained(self.path, dtype=torch.float32).eval()
        firsts = [0, int(b["claim_mask"][0].sum())]
        for i, (row, end) in enumerate(zip(firsts, [2, 4])):
            ids = m.tokenizer(DOCS[i]["text"], add_special_tokens=False)["input_ids"][: end + 1] + [m.end_id]
            want = F.normalize(ref(input_ids=torch.tensor([ids])).last_hidden_state[0, -1], dim=-1)
            torch.testing.assert_close(e[row], want, atol=1e-5, rtol=1e-4)

    def test_zero_residual_is_the_doc_vector(self):
        """residual query, b = 0 (init), position = the doc end token's: exactly the doc vector."""
        m = self.model(query="residual", query_pos="end")
        b = ClaimCollator(m.tokenizer, 4)(self.docs(m))
        h, cache = m.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
        e = m.queries(cache, b["doc_attention_mask"], b["claim_spans"], b["claim_mask"])
        doc = F.normalize(h[:, -1], dim=-1)
        owner = b["claim_mask"].nonzero()[:, 0]
        torch.testing.assert_close(e, doc[owner], atol=1e-5, rtol=1e-4)

    def test_queries_do_not_see_each_other(self):
        for q in ("mask", "residual"):
            m = self.model(query=q)
            with torch.no_grad():
                if m.residual is not None:
                    m.residual.weight.normal_()
            b = ClaimCollator(m.tokenizer, 4)(self.docs(m))
            run = lambda sp: m.queries(*m.read_doc(b["doc_input_ids"], b["doc_attention_mask"])[1:],
                                       b["doc_attention_mask"], sp, b["claim_mask"])
            e1 = run(b["claim_spans"])
            sp = b["claim_spans"].clone()
            sp[1, 0] = torch.tensor([0, 1])                              # move doc 1's first region only
            e2 = run(sp)
            row = int(b["claim_mask"][0].sum())
            keep = [i for i in range(len(e1)) if i != row]
            torch.testing.assert_close(e1[keep], e2[keep])
            self.assertFalse(torch.allclose(e1[row], e2[row]))

    def test_loss_grads_save_load_generate(self):
        for q in ("mask", "residual"):
            m = self.model(query=q).train()
            docs = self.docs(m)
            b = ClaimCollator(m.tokenizer, 4)(docs)
            out = m(**b)
            self.assertTrue(all(torch.isfinite(out[k]) for k in ("loss", "distill", "cos", "tag")))
            out["loss"].backward()
            grads = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
            self.assertTrue(any("lora_B" in n for n in grads))
            self.assertTrue({"tag_head.weight", "crf.trans"} <= grads)
            self.assertEqual("residual.weight" in grads, q == "residual")
            self.assertFalse(any("lora_" not in n and n.startswith("backbone") for n in grads))
            m.eval()
            with tempfile.TemporaryDirectory() as d:
                m.save(d)
                r = ClaimEmb.from_pretrained(d, torch.float32).eval()
                dv, vecs, regions = r.generate(b["doc_input_ids"], b["doc_attention_mask"])
                dv2, vecs2, regions2 = m.generate(b["doc_input_ids"], b["doc_attention_mask"])
                self.assertEqual(regions, regions2)
                torch.testing.assert_close(torch.cat(vecs), torch.cat(vecs2))
                self.assertEqual(len(vecs), 2)

    def test_crf_learns_the_regions(self):
        m = self.model()
        b = ClaimCollator(m.tokenizer, 4)(self.docs(m))
        opt = torch.optim.Adam(list(m.tag_head.parameters()) + list(m.crf.parameters()), lr=0.1)
        h, _ = m.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
        for _ in range(150):
            em, mask = m.emissions(h, b["doc_attention_mask"])
            loss = m.crf.nll(em, b["tags"], mask & b["tag_mask"])
            opt.zero_grad(); loss.backward(); opt.step()
        gold = [tags_to_spans(t[: int(k.sum())].tolist()) for t, k in zip(b["tags"], b["tag_mask"])]
        self.assertEqual(m.predict_regions(h, b["doc_attention_mask"]), gold)


class TestGrid(TinyCase):
    def test_grid_queries(self):
        m = self.model(query="grid", query_pos="end", stride=3)
        b = ClaimCollator(m.tokenizer, 4)(self.docs(m))
        spans, gmask = m.grid_spans(b["doc_attention_mask"])
        n_doc = (b["doc_attention_mask"].sum(1) - 1).tolist()
        for row, n in enumerate(n_doc):                                  # positions 2, 5, ..., capped at the last token
            want = [min(3 * (t + 1) - 1, n - 1) for t in range(-(-n // 3))]
            self.assertEqual(spans[row, gmask[row], 1].tolist(), want)
        h, cache = m.read_doc(b["doc_input_ids"], b["doc_attention_mask"])
        e = m.queries(cache, b["doc_attention_mask"], spans, gmask)       # b = 0: every query is the doc vector
        owner = gmask.nonzero()[:, 0]
        torch.testing.assert_close(e, F.normalize(h[:, -1], dim=-1)[owner], atol=1e-5, rtol=1e-4)
        m.train()
        for match in ("order", "hungarian"):
            m.match = match
            out = m(**b)
            out["loss"].backward()
        grads = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        self.assertTrue({"residual.weight", "keep_head.weight"} <= grads)
        self.assertTrue(any("lora_B" in n for n in grads))
        self.assertFalse(any(n.startswith(("tag_head", "crf")) for n in grads))
        m.eval()
        with tempfile.TemporaryDirectory() as d:
            m.save(d)
            r = ClaimEmb.from_pretrained(d, torch.float32).eval()
            v1, v2 = r.generate(b["doc_input_ids"], b["doc_attention_mask"])[1], \
                m.generate(b["doc_input_ids"], b["doc_attention_mask"])[1]
            torch.testing.assert_close(torch.cat(v1), torch.cat(v2))
            self.assertTrue(all(len(v) >= 1 for v in v1))

    def test_grid_without_lora(self):
        """lora_r = 0: only the residual and the has-claim head train."""
        m = self.model(query="grid", query_pos="end", stride=3, lora_r=0).train()
        self.assertEqual(m._tuners, [])
        m(**ClaimCollator(m.tokenizer, 4)(self.docs(m)))["loss"].backward()
        grads = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        self.assertEqual({n.split(".")[0] for n in grads}, {"residual", "keep_head"})

    def test_attention_analysis_runs(self):
        from embedrag_v4.analyze_attention import METRICS, analyze_doc
        m = self.model(query="grid", query_pos="end", stride=3)
        m.backbone.config._attn_implementation = "eager"
        out, offs, ex, close = analyze_doc(m, ClaimCollator(m.tokenizer, 4), self.docs(m)[1], 2, range(0, 2), 3)
        self.assertEqual(len(close), 3)
        self.assertTrue(all(-1 <= c["kept_best"] <= 1 and c["matched"] is not None for c in close))
        self.assertEqual(set(out), set(METRICS))
        self.assertEqual(len(out["sink"]), m.backbone.config.num_hidden_layers)
        self.assertTrue(all(0 <= x <= 1 for x in out["near_ptr"] + out["own_span"] + out["sink"]))
        self.assertEqual(len(offs), 3)                                    # 3 claims, each matched to a query
        self.assertTrue(all(len(q["top_words"]) == 3 for q in ex))

    def test_grid_without_spans(self):
        """grid trains on plain {text, claims} docs (no aligned spans)."""
        m = self.model(query="grid", query_pos="end", stride=3)
        b = ClaimCollator(m.tokenizer, 4)([{k: v for k, v in d.items() if k != "spans"} for d in DOCS])
        self.assertEqual(int(b["tags"].sum()), 0)
        self.assertTrue(torch.isfinite(m(**b)["loss"]))

    def test_grid_targets(self):
        m = self.model(query="grid")
        t = torch.eye(4)                                                  # doc 0: 3 queries / 2 claims; doc 1: 2 / 2
        e = torch.stack([t[2], t[0], t[1], t[3], t[2]])
        m.match = "hungarian"
        qi, ci = m.grid_targets(e, t, [3, 2], [2, 2])
        self.assertEqual(sorted(zip(qi.tolist(), ci.tolist())), [(1, 0), (2, 1), (3, 3), (4, 2)])
        m.match = "order"                                                 # claim k -> slot int((k + .5) * nq / nc)
        qi, ci = m.grid_targets(e, t, [3, 2], [2, 2])
        self.assertEqual(list(zip(qi.tolist(), ci.tolist())), [(0, 0), (2, 1), (3, 2), (4, 3)])


if __name__ == "__main__":
    unittest.main()
