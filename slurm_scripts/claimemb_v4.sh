#!/bin/bash -l
#SBATCH --job-name=claimemb4
#SBATCH --account=project_465002532
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# ClaimEmb v4: <|endoftext|> queries back-tracking to claim regions + CRF region locator (Qwen3-Embedding only).
#   QUERY=mask sbatch slurm_scripts/claimemb_v4.sh train                 # regions built on first use
#   QUERY=residual QUERY_POS=end sbatch slurm_scripts/claimemb_v4.sh train
#   QUERY=mask sbatch -t 00:20:00 slurm_scripts/claimemb_v4.sh ref       # untrained model, oracle regions
#   QUERY=grid QUERY_POS=end STRIDE=16 sbatch slurm_scripts/claimemb_v4.sh train   # residual queries on a fixed grid
# LORA_R (16; 0 = no LoRA), DATA_NAME (default rag25), STEPS (600), BSZ (32), LR (1e-3), SCHED (cosine), SEED (42), EVAL_DOCS (300), NAME, TAG

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
QUERY=${QUERY:-mask}; QUERY_POS=${QUERY_POS:-region}
SPLITS="dev"; [ "$1" = train ] && SPLITS="train dev"   # ref/attn only need dev regions
SUFFIX=regions.jsonl
if [ "$QUERY" = grid ] && [ "$1" = train ]; then SUFFIX=jsonl; SPLITS=""; fi   # grid uses no regions
for split in $SPLITS; do
    [ -f $DATA.$split.regions.jsonl ] || run python -m embedrag_v4.align --input $DATA.$split.jsonl --tokenizer $BACKBONE
done

case $1 in
train)
    NAME=${NAME:-$QUERY-$QUERY_POS${STRIDE:+-stride$STRIDE}${LORA_R:+-r$LORA_R}-s${SEED:-42}}
    OUT=${ROOT}/scratch/claimemb4-$DATA_NAME${TAG:+-$TAG}/$NAME
    export WANDB_RUN_ID=v4-$DATA_NAME-$NAME-${SLURM_JOB_ID} WANDB_RUN_GROUP=v4-${TAG:-$DATA_NAME}
    run python -m embedrag_v4.train --backbone $BACKBONE --query $QUERY --query_pos $QUERY_POS --stride ${STRIDE:-16} \
        --lora_r ${LORA_R:-16} --lora_alpha $((2 * ${LORA_R:-16})) \
        --seed ${SEED:-42} --report_to wandb --run_name v4-$DATA_NAME-$NAME \
        --train_files $DATA.train.$SUFFIX --bf16 --per_device_train_batch_size ${BSZ:-32} \
        --learning_rate ${LR:-1e-3} --lr_scheduler_type ${SCHED:-cosine} --warmup_steps 20 --max_steps ${STEPS:-600} \
        --logging_steps 20 --save_strategy no --output_dir $OUT
    run python -m embedrag_v4.evaluate --ckpt $OUT --eval_file $DATA.dev.$SUFFIX --num_docs $EVAL_DOCS \
        --output $OUT/fidelity.json ;;
ref)
    run python -m embedrag_v4.evaluate --backbone $BACKBONE --query $QUERY --eval_file $DATA.dev.regions.jsonl \
        --num_docs $EVAL_DOCS --output ${ROOT}/scratch/claimemb4-$DATA_NAME/reference-untrained-$QUERY.json ;;
attn)   # where grid queries attend + per-claim closeness: CKPTS="dir1 dir2" OUT_JSON=... sbatch -t 00:30:00 ... attn
    run python -m embedrag_v4.analyze_attention --ckpts ${CKPTS:-} --backbone $BACKBONE --init_stride ${INIT_STRIDE:-16} \
        --eval_file $DATA.dev.regions.jsonl --num_docs ${EVAL_DOCS:-100} --output $OUT_JSON ;;
esac
