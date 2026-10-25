"""Unit tests for ClaimEmb v3 (decoder student + separate embedding teacher) on tiny random Qwen3s (CPU).

  python -m unittest tests.test_claimemb_v3 -v
"""
import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, Qwen3ForCausalLM

from embedrag_v3 import ClaimCollator, ClaimEmb, ClaimEmbConfig
from tests.test_claimemb import DOCS, TEXTS, make_tiny

TEMPLATE = "the {doc} . sat"  # tiny vocab: a prefix and a suffix around the doc


def make_tiny_teacher(path):
    """Same tiny architecture and tokenizer as the student, different weights."""
    make_tiny(path)
    torch.manual_seed(1)
    Qwen3ForCausalLM(Qwen3ForCausalLM.config_class.from_pretrained(path)).save_pretrained(path)
    return path


class TinyCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.student = make_tiny(cls.tmp.name + "/student")
        cls.teacher = make_tiny_teacher(cls.tmp.name + "/teacher")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def model(self, **kw):
        torch.manual_seed(0)
        kw = {"student": self.student, "teacher": self.teacher, "doc_template": TEMPLATE, "step_init": "cat",
              "max_claims": 4, "step_stride": 3, "lora_dropout": 0.0, **kw}
        cfg = ClaimEmbConfig(**kw)
        return ClaimEmb(cfg, torch.float32).eval()

    @staticmethod
    def collator(m, **kw):
        return ClaimCollator(m.tokenizer, m.teacher_tokenizer, m.cfg.doc_template, max_claims=4, **kw)

    def run_steps(self, m, texts, T):
        d = self.collator(m).docs(texts)
        cache = m.read_doc(**d)
        e, s = m.steps(cache, d["doc_attention_mask"], torch.ones(len(texts), T, dtype=torch.bool))
        return e.view(len(texts), T, -1), s.view(len(texts), T)


class TestCollator(TinyCase):
    def test_template_and_truncation(self):
        tok = AutoTokenizer.from_pretrained(self.student)
        c = ClaimCollator(tok, tok, TEMPLATE, max_doc_len=3)
        d = c.docs(["the cat sat on the mat", "dog"])
        ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        row0 = d["doc_input_ids"][0][d["doc_attention_mask"][0].bool()].tolist()
        self.assertEqual(row0, ids("the") + ids("the cat sat") + ids(". sat"))  # only the doc is cut
        self.assertEqual(d["doc_attention_mask"][1].tolist()[:2], [0, 0])  # left-padded
        claims = c.claims(["the cat sat"])["claim_input_ids"][0].tolist()
        self.assertEqual(claims[-1], tok.eos_token_id)  # teacher inputs keep the embedding end token


class TestDocEosTemplate(TinyCase):
    def test_doc_part_is_the_teachers_input(self):
        """doc_eos with student = teacher: the prompt starts with exactly the teacher's doc tokens (+ end token),
        so the frozen doc states, and the state at the end token, equal the index's doc vector."""
        from embedrag_v3.config import TEMPLATES
        self.assertEqual(ClaimEmbConfig(doc_template="doc_eos").doc_template, TEMPLATES["doc_eos"])
        m = self.model(doc_template="{doc}<end> sat", student=self.teacher)  # tiny vocab: <end> = its end token
        c = self.collator(m)
        d = c.docs(TEXTS)
        t = c.teacher_texts(TEXTS, 1024, "")
        for i in range(len(TEXTS)):
            got = d["doc_input_ids"][i][d["doc_attention_mask"][i].bool()].tolist()
            want = t["input_ids"][i][t["attention_mask"][i].bool()].tolist()
            self.assertEqual(got[: len(want)], want)
            self.assertEqual(got[len(want):], m.tokenizer("sat", add_special_tokens=False)["input_ids"])
            n = len(want)
            with m.lora(False):
                h = m.student(input_ids=torch.tensor([got])).last_hidden_state[0, n - 1]
            torch.testing.assert_close(F.normalize(h, dim=-1), m.teacher(t["input_ids"][i:i + 1, :n],
                                                                         t["attention_mask"][i:i + 1, :n])[0])


class TestHideDocEnd(TinyCase):
    def test_steps_ignore_the_last_prompt_token(self):
        """hide_doc_end: steps don't depend on the prompt's last token; without it they do."""
        for hide in (True, False):
            outs = []
            for tpl in ("{doc} cat", "{doc} sat"):              # same doc, different last prompt token
                m = self.model(doc_template=tpl, hide_doc_end=hide)
                outs.append(self.run_steps(m, TEXTS, 3)[0])
            if hide:
                torch.testing.assert_close(outs[0], outs[1])
            else:
                self.assertFalse(torch.allclose(outs[0], outs[1]))


class TestModel(TinyCase):
    def test_teacher_is_plain_embedding_model(self):
        m = self.model()
        ref = AutoModel.from_pretrained(self.teacher, dtype=torch.float32).eval()
        b = self.collator(m).claims(["the cat sat", "a bird sang a song"])
        got = m.teacher(b["claim_input_ids"], b["claim_attention_mask"])
        for i in range(2):
            n = int(b["claim_attention_mask"][i].sum())
            want = F.normalize(ref(input_ids=b["claim_input_ids"][i:i + 1, :n]).last_hidden_state[0, -1], dim=-1)
            torch.testing.assert_close(got[i], want, atol=1e-5, rtol=1e-4)

    def test_init_step_is_plain_student_on_prompt(self):
        """LoRA B = 0 and proj = identity: step 1 = the plain student on <prompt><step_init token>."""
        m = self.model()
        e, _ = self.run_steps(m, TEXTS, 1)
        ref = AutoModel.from_pretrained(self.student, dtype=torch.float32).eval()
        tok, cat = m.tokenizer, m.tokenizer("cat", add_special_tokens=False)["input_ids"]
        for i, t in enumerate(TEXTS):
            ids = torch.tensor([tok(TEMPLATE.format(doc=t), add_special_tokens=False)["input_ids"] + cat])
            want = F.normalize(ref(input_ids=ids).last_hidden_state[0, -1], dim=-1)
            torch.testing.assert_close(e[i, 0], want, atol=1e-5, rtol=1e-4)

    def test_save_load_roundtrip(self):
        m = self.model()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_B" in n or n.startswith("proj"):
                    p.add_(0.01 * torch.randn_like(p))
        with tempfile.TemporaryDirectory() as d:
            m.save(d)
            r = ClaimEmb.from_pretrained(d, torch.float32).eval()
            torch.testing.assert_close(self.run_steps(r, TEXTS, 3)[0], self.run_steps(m, TEXTS, 3)[0])


class TestTraining(TinyCase):
    def test_loss_and_grads(self):
        m = self.model().train()
        out = m(**self.collator(m)(DOCS))
        self.assertTrue(all(torch.isfinite(out[k]) for k in ("loss", "distill", "cos", "stop")))
        out["loss"].backward()
        grads = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        for name in ("step_query", "proj.weight", "stop_head.weight"):
            self.assertIn(name, grads)
        self.assertTrue(any("lora_B" in n for n in grads))
        self.assertFalse(any(n.startswith("teacher_model") for n in grads))
        self.assertFalse(any("lora_" not in n and n.startswith("student") for n in grads))

    def test_overfits(self):
        m = self.model().train()
        b = self.collator(m)(DOCS)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3)
        first = m(**b)
        for _ in range(60):
            opt.zero_grad()
            m(**b)["loss"].backward()
            opt.step()
        last = m.eval()(**b)
        self.assertLess(last["cos"], first["cos"] * 0.5)
        self.assertLess(last["stop"], first["stop"])

    def test_generate(self):
        m = self.model()
        claims = m.generate(**self.collator(m).docs(TEXTS))
        self.assertEqual(len(claims), 2)
        self.assertTrue(all(c.shape[1] == m.teacher_model.config.hidden_size for c in claims))


class TestAnalyzeDecoder(TinyCase):
    def test_spans_and_metrics(self):
        from embedrag_v3.analyze_decoder import ROWS, analyze_doc, build
        m, init = self.model(), self.model()
        m.student.config._attn_implementation = init.student.config._attn_implementation = "eager"
        b = build(m.tokenizer, TEMPLATE, DOCS[1], 1024, 4)
        self.assertEqual(len(b["sents"]), 4)
        self.assertEqual(b["src"], [0, 1, 2])  # each claim is verbatim one sentence
        lo, hi = b["doc"]
        self.assertEqual((b["sent_spans"][0][0], b["sent_spans"][-1][1]), (lo, hi))
        res = analyze_doc(m, b, init_model=init)
        self.assertEqual(set(res), {"first", "body", "step_init", "step", "step_init_y", "step_y"})
        self.assertEqual(set(analyze_doc(init, b, init_model=init)), {"first", "body", "step_init", "step_init_y"})
        for r in ("first", "body", "step_init", "step"):
            lay = res[r][0][0]
            total = lay["mass_sink"] + lay["mass_template"] + lay["mass_doc"] + lay["mass_prev"] + lay["mass_own"]
            self.assertAlmostEqual(total, 1.0, places=4)  # the five masses partition each row's attention


class TestReferenceChunks(TinyCase):
    def test_pieces_and_metrics(self):
        from embedrag_v3.evaluate import free_run_metrics
        from embedrag_v3.reference_chunks import embed, match_metrics, pieces, windows
        self.assertEqual(windows(list(range(10)), 4, 3), [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]])
        self.assertEqual(windows([1, 2], 4, 3), [[1, 2]])
        tok = AutoTokenizer.from_pretrained(self.teacher)
        self.assertEqual(len(pieces(tok, DOCS[1]["text"], "sentences", 1024, 4, 2)), 4)
        model = AutoModel.from_pretrained(self.teacher, dtype=torch.float32).eval()
        claims = [c for d in DOCS for c in d["claims"]]
        pool = embed(model, tok, claims, 64)
        ref = AutoModel.from_pretrained(self.teacher, dtype=torch.float32).eval()
        want = F.normalize(ref(**tok(claims[0], return_tensors="pt")).last_hidden_state[0, -1], dim=-1)
        torch.testing.assert_close(pool[0], want, atol=1e-5, rtol=1e-4)  # = how the teacher embeds a claim
        # pieces that are exactly the claims: perfect match and hits
        vecs = [pool[:2], pool[2:]]
        m = match_metrics(vecs, pool, [(0, 2), (2, 5)])
        self.assertAlmostEqual(m["match_cos"], 1.0, places=5)
        self.assertEqual((m["match_R@1"], m["matched_frac"]), (1.0, 1.0))
        owner = torch.tensor([0, 0, 1, 1, 1])
        self.assertEqual(free_run_metrics(vecs, pool, owner, [(0, 2), (2, 5)])["claim_hit@1"], 1.0)
        # context x end-token variants: one vector per sentence, each the plain teacher at the right token
        from embedrag_v3.reference_chunks import ablation_vectors, end_id, sentence_ids
        text = DOCS[1]["text"]
        sents = sentence_ids(tok, text, 1024)
        doc = [t for x in sents for t in x]
        e1 = len(sents[0]) + len(sents[1]) - 1                      # last token of sentence 2 in the doc
        plain = lambda ids, i: F.normalize(ref(input_ids=torch.tensor([ids])).last_hidden_state[0, i], dim=-1)
        want = {"sentences_noeos": plain(sents[1], -1), "in_context": plain(doc, e1),
                "in_context_eos": plain(doc[: e1 + 1] + [end_id(tok)], -1)}
        for mode, w in want.items():
            v = ablation_vectors(model, tok, text, mode, 1024, 256)
            self.assertEqual(len(v), len(sents))
            torch.testing.assert_close(v[1], w, atol=1e-5, rtol=1e-4)
        # appended end tokens: the first sentence's masked one and the last sentence's full-view one both equal
        # the prefix + end token pass (same visible tokens, same positions)
        ice = ablation_vectors(model, tok, text, "in_context_eos", 1024, 256)
        torch.testing.assert_close(ablation_vectors(model, tok, text, "mask_eos", 1024, 256)[0], ice[0],
                                   atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(ablation_vectors(model, tok, text, "pos_eos", 1024, 256)[-1], ice[-1],
                                   atol=1e-5, rtol=1e-4)
        # a masked middle sentence differs from its full-view version (it can't pool from other sentences)
        self.assertFalse(torch.allclose(ablation_vectors(model, tok, text, "mask_eos", 1024, 256)[1],
                                        ablation_vectors(model, tok, text, "pos_eos", 1024, 256)[1]))
        # fewer pieces than claims: only some claims matched
        self.assertAlmostEqual(match_metrics([pool[:1], pool[2:3]], pool, [(0, 2), (2, 5)])["matched_frac"], 2 / 5)


if __name__ == "__main__":
    unittest.main()
