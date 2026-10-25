"""Unit tests for EmbedRAG on a tiny random Llama (CPU, a few seconds).

  python -m unittest tests.test_embedrag -v
"""
import argparse
import random
import tempfile
import unittest

import torch

from legacy.embedrag_v1 import MEM_ID, ChatFormat, EmbedRAG, EmbedRAGCollator, EmbedRAGConfig
from legacy.embedrag_v1 import build_data, rewards
from tests.tiny import make_tiny

K = 4
EXAMPLES = [
    {"task": "ae", "instruction": build_data.AE.format(2),
     "passages": ["the cat sat on the mat", "the dog ran in the park today"],
     "target": "the dog ran in the park today"},
    {"task": "rel", "instruction": build_data.REL.format("the cat"),
     "passages": ["the cat sat", "rain fell all night", "the cat sat on the mat"],
     "target": [["[1] ", 2.0], ["relevant. ", 1.0], ["[2] ", 2.0], ["unrelated. ", 1.0],
                ["[3] ", 2.0], ["redundant. ", 1.0]]},
]


class TinyModelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = make_tiny(cls.tmp.name + "/tiny")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def model(self, **kw):
        torch.manual_seed(0)
        cfg = EmbedRAGConfig(decoder=self.path, num_mem=K, attn_implementation="eager", lora_dropout=0.0, **kw)
        return EmbedRAG(cfg, torch.float32).eval()

    def collator(self, m, **kw):
        return EmbedRAGCollator(m.tokenizer, m.enc_tokenizer, ChatFormat(m.tokenizer), num_mem=K, **kw)

    @staticmethod
    def perturb_lora(m):
        """LoRA B is zero-initialized; make the adapters non-trivial so they actually change outputs."""
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.02)


class TestCollator(TinyModelCase):
    def test_mem_slots_and_labels(self):
        m = self.model()
        b = self.collator(m, with_teacher=True)(EXAMPLES)
        n_psg = [len(ex["passages"]) for ex in EXAMPLES]
        # every passage becomes exactly k MEM slots in its own example's prompt
        self.assertEqual(b["enc_input_ids"].shape[0], sum(n_psg))
        self.assertEqual((b["dec_input_ids"] == MEM_ID).sum(1).tolist(), [n * K for n in n_psg])
        # passages are left-padded for the encoder
        self.assertTrue((b["enc_attention_mask"][:, -1] == 1).all())
        # loss only on the target: no label on any prompt / MEM position
        self.assertTrue((b["labels"][b["dec_input_ids"] == MEM_ID] == -100).all())
        # citation tokens carry the upweighted loss
        self.assertEqual(b["label_weights"].max().item(), 2.0)
        # teacher reads text (no MEM slots) but predicts the very same target tokens
        self.assertFalse((b["teacher_input_ids"] == MEM_ID).any())
        for i in range(len(EXAMPLES)):
            s, t = b["labels"][i], b["teacher_labels"][i]
            self.assertEqual(s[s != -100].tolist(), t[t != -100].tolist())

    def test_generation_batch_is_left_padded(self):
        m = self.model()
        b = self.collator(m, for_generation=True)(EXAMPLES)
        self.assertNotIn("labels", b)
        self.assertTrue((b["dec_attention_mask"][:, -1] == 1).all())


class TestEncoder(TinyModelCase):
    def check_encode(self, pooling):
        m = self.model(pooling=pooling)
        self.perturb_lora(m)
        b = self.collator(m)(EXAMPLES)
        with torch.no_grad():
            mem = m.encode(b["enc_input_ids"], b["enc_attention_mask"])
            # shape (P, k, H_dec), every vector has the projector's norm
            self.assertEqual(tuple(mem.shape), (5, K, m.decoder.config.hidden_size))
            torch.testing.assert_close(mem.norm(dim=-1), torch.full(mem.shape[:2], m.projector.scale.item()))
            # a passage's embedding does not depend on padding or on the other passages in the batch
            for i in range(mem.shape[0]):
                ids = b["enc_input_ids"][i][b["enc_attention_mask"][i].bool()][None]
                alone = m.encode(ids, torch.ones_like(ids))
                torch.testing.assert_close(alone[0], mem[i], atol=1e-5, rtol=1e-4)

    def test_memory_pooling(self):
        self.check_encode("memory")

    def test_mean_pooling(self):
        self.check_encode("mean")
        self.assertIsNone(self.model(pooling="mean").mem_queries)

    def test_splice_order(self):
        m = self.model()
        ids = torch.tensor([[5, MEM_ID, MEM_ID, 6, MEM_ID, MEM_ID]])
        mem = torch.randn(2, 2, m.decoder.config.hidden_size)
        emb = m.embed_decoder_inputs(ids, mem)
        torch.testing.assert_close(emb[0, [1, 2, 4, 5]], mem.reshape(4, -1))  # passage 1, then passage 2
        with self.assertRaises(AssertionError):  # slot count must match passages x k
            m.embed_decoder_inputs(ids, mem[:1])


class TestTraining(TinyModelCase):
    def test_trainable_parts_per_stage(self):
        m = self.model(decoder_lora=True)
        self.perturb_lora(m)
        b = self.collator(m)(EXAMPLES)
        for enc, dec in [(True, False), (False, True)]:  # stage 1 align / decoder-only
            m.set_trainable(encoder=enc, projector=True, decoder=dec)
            m.zero_grad(set_to_none=True)
            m.train()
            m(**b)["loss"].backward()
            got = {n.split("lora_", 1)[1].split(".")[1] for n, p in m.named_parameters()
                   if "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0}
            self.assertEqual(got, {a for a, on in [("encoder", enc), ("decoder", dec)] if on})
            self.assertIsNotNone(m.projector.fc1.weight.grad)
            self.assertIsNotNone(m.mem_queries.grad)
            # the frozen base weights never get gradients
            self.assertIsNone(m.decoder.model.layers[0].self_attn.q_proj.base_layer.weight.grad)

    def test_decoder_reads_the_passages(self):
        m = self.model()
        self.perturb_lora(m)
        c = self.collator(m)
        swapped = [{**EXAMPLES[0], "passages": ["bird sang a song", "fish swam in the river"]}]
        with torch.no_grad():
            l1 = m(**c(EXAMPLES[:1]))["loss"]
            l2 = m(**c(swapped))["loss"]
        self.assertGreater(abs(l1 - l2).item(), 1e-4)

    def test_kd_loss(self):
        m = self.model()
        m.kd_weight = 1.0
        out = m(**self.collator(m, with_teacher=True)(EXAMPLES))
        self.assertIn("kd", out)
        self.assertGreaterEqual(out["kd"].item(), 0.0)
        torch.testing.assert_close(out["loss"].detach(), out["ce"] + out["kd"])

    def test_loss_decreases(self):
        # a frozen *random* decoder can barely be steered through k vectors, so also train its LoRA here
        m = self.model(decoder_lora=True)
        m.set_trainable(encoder=True, projector=True, decoder=True)
        b = self.collator(m)(EXAMPLES)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        m.train()
        losses = []
        for _ in range(30):
            loss = m(**b)["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(min(losses[-5:]), 0.85 * losses[0])

    def test_generate(self):
        m = self.model()
        b = self.collator(m, for_generation=True)(EXAMPLES)
        out = m.generate(**b, max_new_tokens=5, do_sample=False)
        self.assertEqual(out.shape[0], len(EXAMPLES))
        self.assertLessEqual(out.shape[1], 5)

    def test_save_load_roundtrip(self):
        m = self.model(decoder_lora=True)
        self.perturb_lora(m)
        b = self.collator(m)(EXAMPLES)
        with tempfile.TemporaryDirectory() as d:
            m.save(d)
            m2 = EmbedRAG.from_pretrained(d, torch.float32).eval()
        with torch.no_grad():
            torch.testing.assert_close(m2(**b)["loss"], m(**b)["loss"])


class TestRewards(unittest.TestCase):
    META = {"relevant": [1, 3], "unrelated": [2], "redundant": [4]}

    def test_parse_segments(self):
        self.assertEqual(rewards.parse_segments("[1] cats sit. [2] unrelated. [1] again"),
                         {1: "cats sit.", 2: "unrelated."})

    def test_label_accuracy_and_citation_f1(self):
        perfect = "[1] cats. [2] unrelated. [3] dogs. [4] redundant."
        self.assertEqual(rewards.label_accuracy(perfect, self.META), 1.0)
        self.assertEqual(rewards.citation_f1(perfect, self.META), 1.0)
        half = "[1] cats. [2] unrelated. [3] unrelated. [4] redundant."
        self.assertEqual(rewards.label_accuracy(half, self.META), 0.75)
        self.assertAlmostEqual(rewards.citation_f1(half, self.META), 2 / 3)
        self.assertIsNone(rewards.label_accuracy("x", {}))

    def test_compute_reward_skips_inapplicable(self):
        total, parts = rewards.compute_reward("paris", {"meta": {"answers": ["Paris"]}},
                                              {"f1": 1.0, "label": 1.0})
        self.assertEqual(parts, {"f1": 1.0})
        self.assertEqual(total, 1.0)


class TestBuildData(unittest.TestCase):
    def test_mds_labels_match_passages(self):
        row = {"topic": "cats", "doc_ctxs": ["d1", "d2"], "comp_ctxs": [["c1"], ["c2"]],
               "distract_ctxs": ["x1", "x2"], "redundant_ctxs": ["r1"]}
        args = argparse.Namespace(max_docs=5, psgs_per_doc=1, num_distractors=2, num_redundant=1, cite_weight=2.0)
        exs = list(build_data.build_mds([row], args, random.Random(0)))
        rel = next(e for e in exs if e["task"] == "rel")
        kind = {"d": "relevant", "x": "unrelated", "r": "redundant"}
        for k in ("relevant", "unrelated", "redundant"):
            self.assertEqual(rel["meta"][k], [i + 1 for i, p in enumerate(rel["passages"]) if kind[p[0]] == k])
        # gold reference scores perfectly under the RL rewards
        self.assertEqual(rewards.label_accuracy(rel["meta"]["reference"], rel["meta"]), 1.0)
        # one `single` example per non-redundant passage
        self.assertEqual(sum(e["task"] == "single" for e in exs), 4)


if __name__ == "__main__":
    unittest.main()
