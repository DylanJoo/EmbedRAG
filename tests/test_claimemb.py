"""Unit tests for ClaimEmb on a tiny random Qwen3 (CPU, under a minute).

  python -m unittest tests.test_claimemb -v
"""
import json
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoModel, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from embedrag import ClaimCollator, ClaimEmb, ClaimEmbConfig

WORDS = "the cat sat on mat dog ran in park sun is hot today rain fell all night bird sang a song".split()
DOCS = [
    {"id": "a", "text": "the cat sat on the mat . the dog ran in the park",
     "claims": ["the cat sat on the mat", "the dog ran in the park"]},
    {"id": "b", "text": "the sun is hot today . rain fell all night . a bird sang a song . the cat sat",
     "claims": ["the sun is hot today", "rain fell all night", "a bird sang a song"]},
]
TEXTS = [d["text"] for d in DOCS]


def make_tiny(path):
    """Word-level tokenizer that appends <end> like Qwen3-Embedding appends <|endoftext|>."""
    vocab = {t: i for i, t in enumerate(["<pad>", "<end>", "<unk>", "."] + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="$A <end>", special_tokens=[("<end>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<end>", unk_token="<unk>")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                      pad_token_id=0, eos_token_id=1, bos_token_id=1)
    os.makedirs(path, exist_ok=True)
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


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
        cfg = ClaimEmbConfig(backbone=self.path, max_claims=4, lora_dropout=0.0, **kw)
        return ClaimEmb(cfg, torch.float32).eval()

    @staticmethod
    def perturb_lora(m):
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.05)

    def plain(self):
        """The untouched backbone, for reference outputs."""
        return AutoModel.from_pretrained(self.path, dtype=torch.float32).eval()

    def run_steps(self, m, texts, T):
        d = ClaimCollator(m.tokenizer).docs(texts)
        doc, cache = m.read_doc(**d)
        e, s = m.steps(cache, d["doc_attention_mask"], torch.ones(len(texts), T, dtype=torch.bool))
        return doc, e.view(len(texts), T, -1), s.view(len(texts), T)


class TestCollator(TinyCase):
    def test_steps_and_labels(self):
        m = self.model()
        b = ClaimCollator(m.tokenizer, max_claims=4)(DOCS)
        self.assertEqual(b["step_mask"].sum(1).tolist(), [3, 4])  # n claims + 1 stop step
        self.assertEqual(b["slot_claim"].tolist(), [0, 1, -1, 2, 3, 4, -1])
        self.assertEqual(b["stop_labels"].tolist(), [0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(b["claim_input_ids"].size(0), 5)
        # docs are left-padded and end with the end token, right where the steps start
        self.assertEqual(b["doc_attention_mask"][0, 0].item(), 0)
        self.assertTrue((b["doc_input_ids"][:, -1] == m.tokenizer.eos_token_id).all())

    def test_extra_stop_steps(self):
        m = self.model()
        b = ClaimCollator(m.tokenizer, max_claims=4, extra_stop_steps=2)(DOCS)
        # doc a: 2 claims + 3 stops; doc b: 3 claims + 2 stops (capped at max_claims + 1 = 5 steps)
        self.assertEqual(b["step_mask"].sum(1).tolist(), [5, 5])
        self.assertEqual(b["stop_labels"].tolist(), [0, 0, 1, 1, 1, 0, 0, 0, 1, 1])

    def test_truncated_claims_have_no_stop_step(self):
        m = self.model()
        b = ClaimCollator(m.tokenizer, max_claims=2)([DOCS[1]])
        self.assertEqual(b["slot_claim"].tolist(), [0, 1])
        self.assertEqual(b["stop_labels"].tolist(), [0, 0])


class TestFrozenPasses(TinyCase):
    def test_doc_vector_is_exact_despite_lora(self):
        """The doc pass never uses the LoRA: doc vectors equal the plain backbone's, padded or not."""
        m = self.model()
        self.perturb_lora(m)
        doc, _ = m.read_doc(**ClaimCollator(m.tokenizer).docs(TEXTS))
        ref = self.plain()
        for i, t in enumerate(TEXTS):
            ids = m.tokenizer(t, return_tensors="pt")["input_ids"]
            want = F.normalize(ref(input_ids=ids).last_hidden_state[0, -1], dim=-1)
            torch.testing.assert_close(doc[i], want, atol=1e-5, rtol=1e-4)

    def test_teacher_ignores_lora(self):
        m = self.model()
        c = ClaimCollator(m.tokenizer).claims([d["claims"][0] for d in DOCS])
        before = m.teacher(c["claim_input_ids"], c["claim_attention_mask"])
        self.perturb_lora(m)
        after = m.teacher(c["claim_input_ids"], c["claim_attention_mask"])
        torch.testing.assert_close(before, after)
        torch.testing.assert_close(after.norm(dim=-1), torch.ones(2))


class TestSteps(TinyCase):
    def test_step_one_at_init_is_one_more_end_token(self):
        """LoRA is zero and [CLM] = <end> at init: step 1 = the plain backbone on '<doc> <end> <end>'."""
        m = self.model()
        _, e, _ = self.run_steps(m, TEXTS, 1)
        ref = self.plain()
        for i, t in enumerate(TEXTS):
            ids = m.tokenizer(t, return_tensors="pt")["input_ids"]
            ids = torch.cat([ids, ids[:, -1:]], 1)
            want = F.normalize(ref(input_ids=ids).last_hidden_state[0, -1], dim=-1)
            torch.testing.assert_close(e[i, 0], want, atol=1e-5, rtol=1e-4)

    def test_step_stride_positions(self):
        """At init, step t with stride s = the plain backbone on '<doc> <end> <end>...' at positions n + s*t."""
        m = self.model(step_stride=5)
        _, e, _ = self.run_steps(m, TEXTS, 3)
        ref = self.plain()
        for i, t in enumerate(TEXTS):
            ids = m.tokenizer(t, return_tensors="pt")["input_ids"]
            n = ids.size(1)
            ids = torch.cat([ids, ids[:, -1:].expand(1, 3)], 1)
            pos = torch.cat([torch.arange(n), n + 5 * torch.arange(3)]).unsqueeze(0)
            want = F.normalize(ref(input_ids=ids, position_ids=pos).last_hidden_state[0, n:], dim=-1)
            torch.testing.assert_close(e[i], want, atol=1e-5, rtol=1e-4)

    def test_padding_invariant(self):
        for kw in ({"step_stride": 1}, {"step_stride": 5}, {"step_stride": 5, "soft_prompt": "the cat sat"}):
            m = self.model(**kw)
            self.perturb_lora(m)
            _, e_batch, s_batch = self.run_steps(m, TEXTS, 3)
            for i, t in enumerate(TEXTS):
                _, e_one, s_one = self.run_steps(m, [t], 3)
                torch.testing.assert_close(e_batch[i], e_one[0], atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(s_batch[i], s_one[0], atol=1e-5, rtol=1e-4)

    def test_lora_changes_steps_only(self):
        m = self.model()
        doc0, e0, _ = self.run_steps(m, TEXTS, 2)
        self.perturb_lora(m)
        doc1, e1, _ = self.run_steps(m, TEXTS, 2)
        torch.testing.assert_close(doc0, doc1)
        self.assertGreater((e0 - e1).abs().max().item(), 1e-3)

    def test_steps_are_causal(self):
        """Step t never sees later steps, so one parallel call equals step-by-step generation."""
        for kw in ({}, {"soft_prompt": "the cat sat", "step_stride": 5}):
            m = self.model(**kw)
            self.perturb_lora(m)
            _, e3, s3 = self.run_steps(m, TEXTS, 3)
            _, e5, s5 = self.run_steps(m, TEXTS, 5)
            torch.testing.assert_close(e3, e5[:, :3], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(s3, s5[:, :3], atol=1e-5, rtol=1e-4)

    def test_soft_prompt_at_init_is_prompt_text(self):
        """Prompt init = embeddings of its text: step 1 = the plain backbone on '<doc> <end> the cat sat <end>'."""
        m = self.model(soft_prompt="the cat sat", step_stride=5)
        self.assertEqual(m.soft_prompt.shape[0], 3)
        doc, e, _ = self.run_steps(m, TEXTS, 1)
        torch.testing.assert_close(doc, self.run_steps(self.model(), TEXTS, 1)[0])  # doc vector untouched
        ref = self.plain()
        p = m.tokenizer("the cat sat", add_special_tokens=False, return_tensors="pt")["input_ids"]
        for i, t in enumerate(TEXTS):
            ids = m.tokenizer(t, return_tensors="pt")["input_ids"]
            ids = torch.cat([ids, p, ids[:, -1:]], 1)
            want = F.normalize(ref(input_ids=ids).last_hidden_state[0, -1], dim=-1)
            torch.testing.assert_close(e[i, 0], want, atol=1e-5, rtol=1e-4)


class TestTraining(TinyCase):
    def test_loss_and_grads(self):
        m = self.model(soft_prompt="the cat sat").train()
        out = m(**ClaimCollator(m.tokenizer, max_claims=4)(DOCS))
        self.assertTrue(all(torch.isfinite(out[k]) for k in ("loss", "distill", "cos", "stop")))
        out["loss"].backward()
        grads = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        self.assertIn("step_query", grads)
        self.assertIn("soft_prompt", grads)
        self.assertTrue(any(n.startswith("stop_head") for n in grads))
        self.assertTrue(any("lora_B" in n for n in grads))
        self.assertFalse(any("lora_" not in n and n.startswith("backbone") for n in grads))

    def test_overfits(self):
        """A few steps on two docs should pull each step toward its own claim and learn to stop."""
        m = self.model().train()
        b = ClaimCollator(m.tokenizer, max_claims=4)(DOCS)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3)
        first = m(**b)
        for _ in range(60):
            opt.zero_grad()
            m(**b)["loss"].backward()
            opt.step()
        last = m.eval()(**b)
        self.assertLess(last["distill"], first["distill"] * 0.5)
        self.assertLess(last["cos"], first["cos"])
        self.assertLess(last["stop"], first["stop"])


class TestDistill(unittest.TestCase):
    def test_zero_at_teacher_and_positive_elsewhere(self):
        from embedrag.model import distill_loss
        t = F.normalize(torch.randn(6, 8), dim=-1)
        t[1] = F.normalize(t[0] + 0.05 * torch.randn(8), dim=0)  # near-duplicate claims: no penalty for closeness
        y = torch.tensor([0, 1, 2, 3])
        self.assertLess(distill_loss(t[y], t, y, 0.05).item(), 1e-6)
        self.assertGreater(distill_loss(t[[1, 0, 2, 3]], t, y, 0.05).item(), 0)


class TestHungarian(TinyCase):
    def test_reassigns_within_doc_only(self):
        from embedrag.model import hungarian_targets
        t = torch.eye(4)  # 4 claims: doc 0 has claims 0,1; doc 1 has claims 2,3
        e = torch.stack([t[1], t[0], t[2], t[3]])  # doc 0's steps are swapped, doc 1's are in order
        y = torch.tensor([0, 1, 2, 3])
        self.assertEqual(hungarian_targets(e, t, y, torch.tensor([0, 0, 1, 1])).tolist(), [1, 0, 2, 3])

    def test_match_warmup_switches(self):
        from types import SimpleNamespace
        from embedrag.train import MatchWarmup
        m = SimpleNamespace(match="hungarian")
        cb = MatchWarmup(m, "hungarian", 100)
        cb.on_step_begin(None, SimpleNamespace(global_step=0), None)
        self.assertEqual(m.match, "order")
        cb.on_step_begin(None, SimpleNamespace(global_step=100), None)
        self.assertEqual(m.match, "hungarian")

    def test_hungarian_never_lowers_cos(self):
        m = self.model().train()
        m.match = "hungarian"
        b = ClaimCollator(m.tokenizer, max_claims=4)(DOCS)
        out = m(**b)
        m.match = "order"
        self.assertLessEqual(out["cos"].item(), m(**b)["cos"].item() + 1e-6)


class TestGenerate(TinyCase):
    def test_generate_stops_and_returns_doc_vectors(self):
        m = self.model()
        d = ClaimCollator(m.tokenizer).docs(TEXTS)
        with torch.no_grad():
            m.stop_head.bias.fill_(50.0)
            doc, claims = m.generate(**d)
            self.assertEqual([len(v) for v in claims], [1, 1])  # always >= 1 claim
            m.stop_head.bias.fill_(-50.0)
            self.assertEqual([len(v) for v in m.generate(**d)[1]], [5, 5])  # max_claims + 1
        torch.testing.assert_close(doc, m.read_doc(**d)[0])

    def test_save_load(self):
        m = self.model()
        self.perturb_lora(m)
        _, e, _ = self.run_steps(m, TEXTS, 2)
        with tempfile.TemporaryDirectory() as d:
            m.save(d)
            cfg_file = os.path.join(d, "claimemb_config.json")  # older checkpoints list since-removed options
            with open(cfg_file) as f:
                cfg = json.load(f)
            with open(cfg_file, "w") as f:
                json.dump({**cfg, "per_step_tokens": False, "feedback": False, "coverage_layer": 0}, f)
            m2 = ClaimEmb.from_pretrained(d, torch.float32).eval()
        torch.testing.assert_close(e, self.run_steps(m2, TEXTS, 2)[1])


if __name__ == "__main__":
    unittest.main()
