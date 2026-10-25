# consolidated-retrieval-context

## Structure
```
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
slurm_scripts/       # claimemb.sh {data|train|eval} (LUMI) runs v2; embedrag.sh runs v1; others are OLD
```
