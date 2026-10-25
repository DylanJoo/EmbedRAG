#!/bin/bash -l
#SBATCH --job-name=claimemb
#SBATCH --account=project_465002532
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# ClaimEmb (main package claimemb/): grid end-token queries with a back-track residual on Qwen3-Embedding.
#   sbatch slurm_scripts/claimemb_grid.sh train                           # rag25, 600 steps
#   DATA_NAME=n30k STEPS=750 STRIDE=24 sbatch slurm_scripts/claimemb_grid.sh train
#   CKPT=dir sbatch -t 00:15:00 slurm_scripts/claimemb_grid.sh eval
# DATA_NAME (rag25), STRIDE (16), LORA_R (16; 0 = residual only), STEPS (600), BSZ (32), LR (1e-3), SCHED (cosine),
# SEED (42), EVAL_DOCS (300), NAME, TAG

ROOT=/pfs/lustrep1/scratch/project_465002532/personal/dylan
SIF=/scratch/project_465002532/images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0-AY-GRPO.sif
BACKBONE=${BACKBONE:-/users/judylan1/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}
export WANDB_PROJECT=${WANDB_PROJECT:-claimemb} WANDB_DIR=${ROOT}/scratch/wandb
mkdir -p $WANDB_DIR
curl -s --max-time 10 -o /dev/null https://api.wandb.ai || export WANDB_MODE=offline

run() {
    singularity exec -B /pfs,/scratch,/tmp,/users $SIF bash -c \
        "source /opt/miniconda3/bin/activate pytorch; export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1; cd ${ROOT}/EmbedRAG; $*"
}

DATA_NAME=${DATA_NAME:-rag25}; DATA=data/claimemb/$DATA_NAME; EVAL_DOCS=${EVAL_DOCS:-300}

case $1 in
train)
    NAME=${NAME:-grid-stride${STRIDE:-16}-r${LORA_R:-16}-s${SEED:-42}}
    OUT=${ROOT}/scratch/claimemb-main-$DATA_NAME${TAG:+-$TAG}/$NAME
    export WANDB_RUN_ID=main-$DATA_NAME-$NAME-${SLURM_JOB_ID} WANDB_RUN_GROUP=main-${TAG:-$DATA_NAME}
    run python -m claimemb.train --backbone $BACKBONE --stride ${STRIDE:-16} \
        --lora_r ${LORA_R:-16} --lora_alpha $((2 * ${LORA_R:-16})) \
        --seed ${SEED:-42} --report_to wandb --run_name main-$DATA_NAME-$NAME \
        --train_files $DATA.train.jsonl --bf16 --per_device_train_batch_size ${BSZ:-32} \
        --learning_rate ${LR:-1e-3} --lr_scheduler_type ${SCHED:-cosine} --warmup_steps 20 --max_steps ${STEPS:-600} \
        --logging_steps 20 --save_strategy no --output_dir $OUT
    run python -m claimemb.evaluate --ckpt $OUT --eval_file $DATA.dev.jsonl --num_docs $EVAL_DOCS \
        --output $OUT/fidelity.json ;;
eval)
    run python -m claimemb.evaluate --ckpt $CKPT --eval_file ${EVAL_FILE:-$DATA.dev.jsonl} --num_docs $EVAL_DOCS \
        --output $CKPT/fidelity${EVAL_SUFFIX:-}.json ;;
esac
