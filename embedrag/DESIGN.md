# EmbedRAG design

An encoder LM compresses each passage into k vectors. A decoder LM reads those vectors as soft tokens. Both LMs are adapted with LoRA.

## 1. Encoder embedding (`model.py: encode`)
| Option | Setting | Notes |
|---|---|---|
| Memory slots (default) | `pooling=memory, num_mem=k` | Appends k learned query vectors to the passage and reads their hidden states (as in ICAE/PISCO). |
| Chunk mean | `pooling=mean` | Mean-pools k contiguous token chunks (REFRAG-like). Needs no new parameters. |
| Layer | `enc_layer=-1` | Intermediate layers may transfer better; worth sweeping. |
| Projector | LN → MLP → unit-normalize × learned scale | The scale is initialized to the decoder's median token-embedding norm. This matters for Gemma, whose embeddings are scaled by sqrt(d). |

Compression ratio = `max_passage_len / num_mem`. Sweep k ∈ {1, 4, 8, 16, 32}; k=1 is xRAG-like.

## 2. PEFT
- **Shared backbone (default):** one copy of the weights carries the LoRA adapters `encoder` and `decoder`, switched per call with `set_adapter`.
- **Separate encoder:** `--encoder <smaller LM>`, e.g. Qwen3-0.6B encoding for Gemma-3-4b. The projector bridges the different hidden sizes.
- **Trainable parts by stage:** stage 1 trains `encoder`, the projector and the memory slots, with the decoder frozen. This keeps the decoder's general ability intact and makes the embeddings a "language" the frozen decoder already reads. Stage 2 can optionally add the `decoder` adapter.

## 3. Training data (`build_data.py`)
| Stage | Tasks | Purpose |
|---|---|---|
| 1 align | `ae` (reconstruct), `cont` (continue), `ae_multi` (reconstruct passage [j] of n) | Embeddings keep the content; the decoder can tell passages apart |
| 2 task | `qa`, `summ` (cited multi-doc summary with distractors/redundant), `rel`, `single` | Use the content |
| 3 RL | `summ`, `qa` with `meta` labels | Optimize what CE misses |

## 4. Objectives
- **CE** on target tokens, with per-segment weights (`target: [[text, w], ...]`). Citation tokens `[i]` are upweighted (`--cite_weight`).
- **KD** (`--kd_weight`): the teacher is the frozen base decoder given the same prompt with the passages as plain text. The loss is the token-level KL on the same targets, so the student learns to match full-text RAG behavior. PISCO reports that distilling from the teacher matters more than gold labels.
- **Fine-grained supervision:** decisions per passage (`rel`: relevant / unrelated / redundant), passage addressing (`ae_multi`), and single-passage summaries (`single`).

## 5. RL (`rl.py`, GRPO)
- **Policy:** the `decoder` LoRA. The encoder and projector are frozen, so the embeddings are fixed inputs. The reference is a frozen copy of the SFT adapter.
- **Rewards (`rewards.py`):**
  - per-passage label accuracy
  - citation F1 (cited ids vs. gold relevant ids)
  - answer token-F1
  - length cap

  An LLM-judge faithfulness reward fits the same interface.
- **Next steps:** let RL also update the encoder (needs a reference encoder), and add a learned policy that chooses which passages to expand to full text (REFRAG).

## 6. Things to verify
- Baselines: no context (lower bound) and full-text RAG with the same LM (upper bound).
- Whether the decoder actually reads the embeddings. Test with counterfactual or unseen-fact passages; if outputs don't change with the passages, the model is answering from its own knowledge.
