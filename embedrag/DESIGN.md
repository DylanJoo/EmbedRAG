# ClaimEmb design (v2)

An embedding LM reads a document and writes out its claim embeddings one step at a time, directly in a
retrieval space. Today, claim-level retrieval needs an LLM to extract claims and then one embedding pass
per claim; ClaimEmb produces all of a document's claim vectors in one forward pass.

v1 (passage embeddings decoded as soft tokens by a decoder LM) is archived in `legacy/embedrag_v1/`. It
asked the embeddings to be both retrievable and decodable without the source text, and those goals compete.
v2 trains for retrieval only. Decoding the claim vectors back into text is a later, separate study (§6).

## 1. Model (`model.py`)
```
<doc tokens><|endoftext|>   ──(Qwen3-Embedding-0.6B, frozen, no grad)──>  doc vector + KV cache
[CLM] [CLM] ... [CLM]       ──(same backbone + LoRA "claim", reading the cache)──>  h_1 ... h_T
    e_t = normalize(h_t)              claim t's embedding
    p_t = sigmoid(stop_head(h_t))     no claims left; generation ends at the first p_t > 0.5 (t >= 2)
```
- **LoRA only on the [CLM] steps.** Document tokens go through the frozen model, so the doc vector is
  exactly the index vector (`docs_emb`), and training never backpropagates through the ~1k document
  tokens: only the ~13-35 steps carry gradients (no gradient checkpointing).
- **[CLM]** is one learned vector shared by all steps, initialized from `<|endoftext|>` (the token the
  backbone pools at). Steps attend to the document and to all earlier steps; their inputs don't depend on
  earlier outputs, so all steps run in one call. At init (LoRA = 0), step 1 = the plain model on
  `<doc><|endoftext|><|endoftext|>`.
- **Stop head:** linear on h_t; claim counts vary (median 12, p95 31).
- **Trainable:** LoRA r=16 on all attention+MLP projections (10.1M), [CLM] (1024), stop head (1025).

## 2. Teacher and objective
- **Teacher:** the frozen backbone embedding each gold claim's text. It reproduces the precomputed RAGTIME
  index vectors exactly (cos = 1.0000), so claim vectors share the space of `queries_emb` / `docs_emb` /
  `claims_emb`; the query side stays frozen. Inputs have no prefix: raw text + `<|endoftext|>`.
- **Matching (`--match`, default hungarian):** each doc's steps are assigned to its gold claims one-to-one,
  maximizing total cos (scipy). The first `--match_warmup_steps` (100) use document order (step t = claim t);
  without that warm start Hungarian is seed-unstable.
- **Losses:** score distillation + cosine(e_t, its claim) (= MSE/2 on unit vectors) + BCE stop.
  Distillation is listwise KL between the student scores softmax(e_t · t_j / τ) and the teacher scores
  softmax(t_y · t_j / τ) over the batch's claims (same doc + in-batch), τ = 0.05: the gold claim vector's own
  scores, so the optimum is exactly e_t = t_y. It replaced one-hot InfoNCE, whose optimum is *not* e_t = t_y
  when a document's claims are close to each other (it pushes near-duplicates apart, fighting the cos term),
  and whose strength therefore varied with how tightly a document's claims are related. MSE alone
  collapses to the doc's average claim. Now only claim-claim scores are distilled; doc-claim (and later
  query-claim) scores can be added as extra candidates in `distill_loss`.
- **Stop labels:** the step after the last claim and `--extra_stop_steps` (2) more are labeled stop,
  weighted `--stop_pos_weight` (3). This cut the count error from 7.7 to ~2.

Tried and removed: a separate [CLM]
per step, feeding back the previous claim vector (ignored by the model), and a coverage attention bias over
document tokens (single-seed gain, assumes document-order targets).

## 3. Data (`build_data.py`)
RAGTIME `eng-docs.processed-claims.jsonl.gz` (1.0M docs, `statements` = LLM-extracted claims in doc order)
→ `{"id", "text", "claims"}`. Keeps docs with ≤ 1024 tokens (77%) and 1–32 claims; dev split by doc-id hash.

## 4. Evaluation (`evaluate.py`, no qrels needed)
Pool = all gold claims of the dev docs (teacher vectors).
- **Teacher-forced** (n gold steps): `step_cos`, `step_R@1/10` (gold claim t ranks top-k for e_t, document
  order) and `match_cos`, `match_R@1` (after Hungarian matching; the fair view for Hungarian-trained models).
- **Free-running** (stop head): `claim_hit@1` (share of gold claims that are some vector's top-1: the main
  metric), `precision@1` (top-1 claim is from its own doc), `distinct@1` (steps don't repeat), `count_mae`,
  `coverage` (per gold claim, max cos over its doc's vectors).
- **Baseline:** the frozen single doc vector, scored the same way (claim_hit@1 0.09).

## 5. Next: downstream retrieval (needs RAGTIME qrels)
Query → doc by max-sim over each doc's vectors, with frozen Qwen3-Embedding query vectors. Compare: the doc
vector (lower bound), max-sim over gold claim vectors (upper bound), ClaimEmb, and k fixed memory slots
without the stop head (tests whether the autoregressive steps matter). Then add query→claim score distillation.

## 6. Later: decoding claim vectors
A Qwen3-0.6B decoder LoRA reads e_t via a projector **together with the source document**, so it only has to
locate and state the claim. Note that `tests/demo_claims.py` (projector only, no document) recovers the topic
but not the details (token-F1 ≈ 0.07). A cheap probe once training is done: run that projector on ClaimEmb
vectors vs gold claim vectors; similar quality means retrieval training didn't hurt decodability.
