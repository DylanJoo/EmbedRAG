#!/bin/bash -l
#SBATCH --job-name=claimemb3
#SBATCH --account=project_465002532
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# ClaimEmb v3: Qwen3-0.6B student, frozen Qwen3-Embedding-0.6B teacher (same data as v2).
#   sbatch slurm_scripts/claimemb_v3.sh small                                   # ~3k docs, 400 steps
#   NDOCS=30000 DATA_NAME=n30k EVAL_DOCS=300 STEPS=1500 sbatch slurm_scripts/claimemb_v3.sh small
# BSZ: batch size (default 16)
# TEMPLATE: doc_template name (chat | doc_eos) or text with {doc}; default chat. doc_eos with STUDENT=$TEACHER
#   keeps the doc input identical to the embedding model's ({doc}<|endoftext|>) and only adds the generation cue
# HIDE_DOC_END=1: steps can't attend to the prompt's last token (with TEMPLATE=doc: the doc vector's <|endoftext|>)
# SCHED: lr schedule (default constant_with_warmup; e.g. cosine)
# NAME: run name; STEP_WEIGHTS: lora | full (use a small LR, e.g. 2e-5); LORA_R (alpha = 2 * rank)
# STRIDE (default 24), EXTRA_STOP / STOP_PW (2 / 3), MATCH / MATCH_WARMUP (hungarian / 100), DISTILL / COS (1 / 1)

ROOT=/pfs/lustrep1/scratch/project_465002532/personal/dylan
SIF=/scratch/project_465002532/images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0-AY-GRPO.sif
HUB=/users/judylan1/.cache/huggingface/hub
STUDENT=${STUDENT:-$HUB/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}
TEACHER=${TEACHER:-$HUB/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}
export WANDB_PROJECT=${WANDB_PROJECT:-claimemb} WANDB_DIR=${ROOT}/scratch/wandb
mkdir -p $WANDB_DIR
curl -s --max-time 10 -o /dev/null https://api.wandb.ai || export WANDB_MODE=offline

run() {
    singularity exec -B /pfs,/scratch,/tmp,/users $SIF bash -c \
        "source /opt/miniconda3/bin/activate pytorch; export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1; cd ${ROOT}/EmbedRAG; $*"
}

case $1 in
small)
    DATA_NAME=${DATA_NAME:-small}; EVAL_DOCS=${EVAL_DOCS:-100}; NAME=${NAME:-v3-s${SEED:-42}}
    DATA=data/claimemb/$DATA_NAME   # built by v2: sbatch slurm_scripts/claimemb.sh small
    OUT=${ROOT}/scratch/claimemb3-$DATA_NAME${TAG:+-$TAG}/$NAME
    LR=${LR:-1e-3}; STEPS=${STEPS:-400}
    export WANDB_RUN_ID=v3-$DATA_NAME-$NAME-${SLURM_JOB_ID} WANDB_RUN_GROUP=v3-${TAG:-$DATA_NAME}
    run python -m embedrag_v3.train --student $STUDENT --teacher $TEACHER \
        --distill_weight ${DISTILL:-1} --cos_weight ${COS:-1} \
        --extra_stop_steps ${EXTRA_STOP:-2} --stop_pos_weight ${STOP_PW:-3} \
        --match ${MATCH:-hungarian} --match_warmup_steps ${MATCH_WARMUP:-100} --step_stride ${STRIDE:-24} \
        ${TEMPLATE:+--doc_template $TEMPLATE} ${HIDE_DOC_END:+--hide_doc_end} --step_weights ${STEP_WEIGHTS:-lora} --lora_r ${LORA_R:-16} --lora_alpha $((2 * ${LORA_R:-16})) \
        --seed ${SEED:-42} --report_to wandb --run_name v3-$DATA_NAME-$NAME \
        --train_files $DATA.train.jsonl --bf16 --per_device_train_batch_size ${BSZ:-16} --learning_rate $LR \
        --lr_scheduler_type ${SCHED:-constant_with_warmup} --warmup_steps 20 --max_steps $STEPS --logging_steps 20 \
        --save_strategy no --output_dir $OUT
    run python -m embedrag_v3.evaluate --ckpt $OUT --eval_file $DATA.dev.jsonl --num_docs $EVAL_DOCS \
        --output $OUT/fidelity.json ;;
eval)   # CKPTS="dir1 dir2" EVAL_FILE=data/claimemb/n30k.dev.jsonl EVAL_DOCS=300 sbatch ... eval
    for c in ${CKPTS:-$CKPT}; do
        run python -m embedrag_v3.evaluate --ckpt $c --eval_file ${EVAL_FILE:-data/claimemb/small.dev.jsonl} \
            --num_docs ${EVAL_DOCS:-100} --output $c/fidelity${EVAL_SUFFIX}.json
    done ;;
attn)   # plain-decoder vs [CLM]-step attention over doc sentences: CKPT=dir OUT_JSON=... sbatch -t 00:30:00 ... attn
    run python -m embedrag_v3.analyze_decoder ${CKPT:+--ckpt $CKPT} \
        --eval_file ${EVAL_FILE:-data/claimemb/small.dev.jsonl} --num_docs ${EVAL_DOCS:-100} --output $OUT_JSON ;;
ref)    # training-free reference (teacher embeds doc sentences / token windows): OUT_JSON=... sbatch -t 00:20:00 ... ref
        # MODES: any of sentences windows sentences_noeos in_context in_context_eos pos_eos mask_eos (default: sentences windows)
    run python -m embedrag_v3.reference_chunks --teacher $TEACHER \
        --eval_file ${EVAL_FILE:-data/claimemb/n30k.dev.jsonl} --num_docs ${EVAL_DOCS:-300} \
        --window ${WINDOW:-32} --window_stride ${WINDOW_STRIDE:-16} ${MODES:+--modes $MODES} --output $OUT_JSON ;;
esac
