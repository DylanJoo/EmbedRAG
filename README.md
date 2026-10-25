# consolidated-retrieval-context

## Structure
```
claimemb/            # MAIN ClaimEmb: grid end-token queries + back-track residual (Qwen3-Embedding + LoRA)
  model.py           #   ClaimEmb: doc read frozen (exact doc vector), <|endoftext|> + b(distance) queries on a
                     #   stride grid, has-claim head; Hungarian-matched training, score distillation + cos
  data.py            #   collator: {id, text, claims} -> doc input + teacher claim inputs
  train.py           #   HF Trainer (proportional -> Hungarian matching warm start)
  evaluate.py        #   claim-vector fidelity: kept / all queries, doc-vector baseline
embedrag_v4/         # v4 experiments: end-token queries with regions (mask / residual / grid), CRF locator,
                     #   claim-span alignment (align.py), grid attention analysis (analyze_attention.py)
embedrag_v3/         # v3 experiments: separate student/teacher (Qwen3-0.6B or Qwen3-Embedding), doc templates,
                     #   training-free baselines (reference_chunks.py), decoder attention (analyze_decoder.py),
                     #   rag25 builder (build_rag25.py)
embedrag/            # v2 ClaimEmb: a document's claim embeddings, one step at a time (Qwen3-Embedding + LoRA)
  model.py           #   ClaimEmb: [CLM] steps, stop head, teacher = backbone with LoRA off, score distillation + cos + stop
  data.py            #   collator: doc + claim-step slots; gold claims -> teacher inputs
  build_data.py      #   RAGTIME processed-claims -> {id, text, claims} train/dev jsonl
  train.py           #   HF Trainer
  evaluate.py        #   claim-vector fidelity vs gold claims (+ single doc-vector baseline)
  DESIGN.md          #   design notes
legacy/embedrag_v1/  # v1 (archived): passage embeddings as LM context (encoder LoRA -> projector -> decoder LoRA)
  backbone.py        #   Llama / Qwen / Gemma-3 loading, chat template, LoRA targets
  model.py           #   EmbedRAG: encode (memory slots | chunk mean), projector, CE + KD loss, generate
  data.py            #   unified jsonl schema + collator (student w/ embeddings, teacher w/ text)
  build_data.py      #   pretrain (ae/cont) | mds (summ/rel/single) | qa  -> unified jsonl
  train.py           #   SFT stages: align / task (HF Trainer)
  rl.py, rewards.py  #   GRPO on decoder LoRA; label / citation / F1 / length rewards
  infer.py           #   generation + scoring
  DESIGN.md          #   design notes
models/              # OLD (reference): fidt5.py (FiD-T5), modeling_llama.py (CEPE-style cross-attn), archived/
data/                # OLD: compose_pairs.py (mds pairs), collator.py (FiD collators)
llm/ prompts/ tools/ # OLD: NeuCLIR RAG pipeline (vLLM, PLAID-X search, post-citation, eval)
train.py, standard.py, multidoc_summarize.py, legacy/summarize.py   # OLD entry points
slurm_scripts/       # claimemb_grid.sh runs the main package; claimemb.sh v2, claimemb_v3.sh v3, claimemb_v4.sh v4;
                     #   embedrag.sh runs v1; others are OLD
tests/               # unit tests on tiny random Qwen3 models: test_claimemb_grid.py (main), test_claimemb.py (v2),
                     #   test_claimemb_v3.py, test_claimemb_v4.py
```
