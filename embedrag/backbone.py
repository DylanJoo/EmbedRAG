"""Uniform wrapper over HF decoder-only LMs (Llama-3, Qwen2.5/3, Gemma-3).

Everything family-specific lives here, so the rest of the package only needs:
  load_backbone / hidden_size / lora_target_regex / logit_softcap / ChatFormat.

Notes per family:
  - Gemma-3 (>=4b) loads as Gemma3ForConditionalGeneration; text sizes sit in `config.text_config`,
    and the vision tower must be excluded from LoRA. Its embedding layer applies the sqrt(d) scale
    itself, so `get_input_embeddings()(ids)` already yields what the decoder expects.
  - Qwen3 templates take `enable_thinking`; we always disable thinking (ignored by other templates).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def text_config(config):
    return getattr(config, "text_config", None) or config


def hidden_size(model):
    return text_config(model.config).hidden_size


def logit_softcap(model):
    return getattr(text_config(model.config), "final_logit_softcapping", None)


def lora_target_regex():
    # full-match on module names; skip any vision / multimodal modules (Gemma-3)
    return r"^(?!.*(vision|multi_modal)).*\.(" + "|".join(LORA_MODULES) + r")$"


def load_backbone(name_or_path, dtype=torch.bfloat16, attn_implementation="sdpa"):
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=dtype, attn_implementation=attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


class ChatFormat:
    """Renders prompts with the model's own chat template and finds the end-of-turn suffix."""

    def __init__(self, tokenizer, use_chat_template=True, system=None):
        self.tok = tokenizer
        self.use_chat = use_chat_template and tokenizer.chat_template is not None
        self.system = system
        self.eot = self._find_eot()

    def _messages(self, user):
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        return msgs + [{"role": "user", "content": user}]

    def prompt(self, user):
        if not self.use_chat:
            return (self.tok.bos_token or "") + (f"{self.system}\n\n" if self.system else "") + user + "\n"
        return self.tok.apply_chat_template(
            self._messages(user), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _find_eot(self):
        if not self.use_chat:
            return self.tok.eos_token
        probe = "XQXQXQ"
        full = self.tok.apply_chat_template(
            self._messages("hi") + [{"role": "assistant", "content": probe}],
            tokenize=False, enable_thinking=False,
        )
        return full[full.rfind(probe) + len(probe):]
