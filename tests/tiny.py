"""A tiny random Llama + word-level tokenizer, so EmbedRAG can be tested on CPU with no downloads."""
import os

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from legacy.embedrag_v1 import build_data

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
WORDS = ("the cat sat on mat dog ran in park sun is hot today rain fell all night "
         "bird sang a song fish swam river tree grew tall moon rose over hill "
         "paris capital of france berlin germany what where who").split()
TEXTS = [build_data.AE, build_data.CONT, build_data.SUMM, build_data.REL, build_data.SINGLE, build_data.QA,
         "Context: relevant. unrelated. redundant. Question: Topic:", " ".join(map(str, range(21)))]


def vocab():
    pre = pre_tokenizers.Whitespace()
    toks = {t for s in TEXTS + WORDS for t, _ in pre.pre_tokenize_str(s)}
    return {t: i for i, t in enumerate(SPECIALS + sorted(toks))}


def make_tiny(path, seed=0):
    v = vocab()
    tk = Tokenizer(models.WordLevel(v, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<s>",
                                  eos_token="</s>", unk_token="<unk>")
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(v), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    os.makedirs(path, exist_ok=True)
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path
