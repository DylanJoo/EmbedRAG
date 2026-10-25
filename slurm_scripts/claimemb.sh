#!/bin/bash -l
#SBATCH --job-name=claimemb
#SBATCH --account=project_465002532
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# usage: sbatch slurm_scripts/claimemb.sh {data|train|eval}
# loss ablation (DISTILL = claim-claim score distillation weight, COS = cosine anchor weight):
#   NAME=cos DISTILL=0 COS=1 sbatch slurm_scripts/claimemb.sh train
#   NAME=distill DISTILL=1 COS=0 sbatch slurm_scripts/claimemb.sh train
#   NAME=both DISTILL=1 COS=1 sbatch slurm_scripts/claimemb.sh train

ROOT=/pfs/lustrep1/scratch/project_465002532/personal/dylan
SIF=/scratch/project_465002532/images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0-AY-GRPO.sif
# local snapshot: offline tokenizer loading by hub id fails in transformers 4.57
BACKBONE=${BACKBONE:-/users/judylan1/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}
NAME=${NAME:-both}
DISTILL=${DISTILL:-1}
COS=${COS:-1}
DATA=data/claimemb/eng
OUT=${ROOT}/scratch/claimemb/${NAME}
export WANDB_PROJECT=${WANDB_PROJECT:-claimemb} WANDB_DIR=${ROOT}/scratch/wandb
mkdir -p $WANDB_DIR
# compute nodes without internet: log offline, then `wandb sync $WANDB_DIR/wandb/offline-run-*`
curl -s --max-time 10 -o /dev/null https://api.wandb.ai || export WANDB_MODE=offline

run() {
    singularity exec -B /pfs,/scratch,/tmp,/users $SIF bash -c \
        "source /opt/miniconda3/bin/activate pytorch; export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1; cd ${ROOT}/EmbedRAG; $*"
}

case $1 in
data)
    run python -m embedrag.build_data --input ~/scratch/ragtime2/eng-docs.processed-claims.jsonl.gz \
        --output $DATA --tokenizer $BACKBONE --num_docs 200000 ;;
train)
    # no gradient checkpointing: only the [CLM] steps are backpropagated (docs are read frozen)
    export WANDB_RUN_ID=${NAME}-${SLURM_JOB_ID}  # evaluate.py adds its metrics to this run
    run python -m embedrag.train --backbone $BACKBONE --distill_weight $DISTILL --cos_weight $COS \
        --report_to wandb --run_name $NAME \
        --train_files $DATA.train.jsonl --bf16 \
        --per_device_train_batch_size 32 --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 \
        --max_steps 5000 --logging_steps 20 --save_steps 2000 --dataloader_num_workers 2 \
        --output_dir $OUT
    run python -m embedrag.evaluate --ckpt $OUT --eval_file $DATA.dev.jsonl --output $OUT/fidelity.json ;;
eval)   # CKPTS="dir1 dir2" EVAL_FILE=data/claimemb/small.dev.jsonl EVAL_DOCS=100 sbatch -p dev-g -t 00:15:00 ... eval
    for c in ${CKPTS:-$CKPT}; do
        run python -m embedrag.evaluate --ckpt $c --eval_file ${EVAL_FILE:-$DATA.dev.jsonl} \
            --num_docs ${EVAL_DOCS:-500} --output $c/fidelity${EVAL_SUFFIX}.json
    done ;;
attn)   # where [CLM] steps attend: CKPTS="dir1 dir2" INIT_STRIDES="1 24" OUT_JSON=... sbatch -p dev-g -t 00:30:00 ... attn
    run python -m embedrag.analyze_attention --backbone $BACKBONE --ckpts ${CKPTS:-} --init_strides ${INIT_STRIDES:-} \
        --eval_file ${EVAL_FILE:-data/claimemb/small.dev.jsonl} --num_docs ${EVAL_DOCS:-100} --output $OUT_JSON ;;
small)  # quick runs on ~3k docs, fits dev-g; defaults = the current best recipe:
        #   sbatch -p dev-g -t 00:30:00 slurm_scripts/claimemb.sh small
        # CFGS entries: name,distill_weight,cos_weight      e.g. CFGS="cos,0,1 distill,1,0 both,1,1"
        # EXTRA_STOP / STOP_PW: extra stop-labeled steps / stop class weight (defaults 2 / 3)
        # MATCH / MATCH_WARMUP: order | hungarian / document-order steps first (defaults hungarian / 100)
        # STRIDE: position gap between [CLM] steps (default 24); PROMPT: soft-prompt init text (default none)
        # BSZ: per-device batch size (default 16)
        # LORA_R: lora rank (default 16; alpha = 2 * rank)
        # bigger subset: NDOCS=30000 DATA_NAME=n30k EVAL_DOCS=300 STEPS=1500 sbatch -p dev-g -t 01:00:00 ...
    NDOCS=${NDOCS:-3000}; DATA_NAME=${DATA_NAME:-small}; EVAL_DOCS=${EVAL_DOCS:-100}
    SMALL=data/claimemb/$DATA_NAME
    BASE=${ROOT}/scratch/claimemb-$DATA_NAME/baseline.json
    SOUT=${ROOT}/scratch/claimemb-$DATA_NAME${TAG:+-$TAG}
    LR=${LR:-1e-3}; SCHED=${SCHED:-cosine}; STEPS=${STEPS:-400}; CFGS=${CFGS:-"both,1,1"}
    [ -f $SMALL.dev.jsonl ] || run python -m embedrag.build_data \
        --input ~/scratch/ragtime2/eng-docs.processed-claims.jsonl.gz \
        --output $SMALL --tokenizer $BACKBONE --num_docs $NDOCS --dev_frac 0.05
    [ -f $BASE ] || run python -m embedrag.evaluate --backbone $BACKBONE \
        --eval_file $SMALL.dev.jsonl --num_docs $EVAL_DOCS --output $BASE
    for cfg in $CFGS; do
        IFS=, read -r name distill cos <<< "$cfg"
        set -- $name $distill $cos
        # run name = what the run is; TAG = which batch of runs (wandb group)
        export WANDB_RUN_ID=$DATA_NAME-lr$LR-$1-${SLURM_JOB_ID} WANDB_RUN_GROUP=${TAG:-$DATA_NAME}
        run python -m embedrag.train --backbone $BACKBONE --distill_weight $2 --cos_weight $3 \
            --extra_stop_steps ${EXTRA_STOP:-2} --stop_pos_weight ${STOP_PW:-3} \
            --match ${MATCH:-hungarian} --match_warmup_steps ${MATCH_WARMUP:-100} --step_stride ${STRIDE:-24} --soft_prompt "'${PROMPT:-}'" \
            --lora_r ${LORA_R:-16} --lora_alpha $((2 * ${LORA_R:-16})) \
            --seed ${SEED:-42} \
            --report_to wandb --run_name $DATA_NAME-lr$LR-$1 \
            --train_files $SMALL.train.jsonl --bf16 --per_device_train_batch_size ${BSZ:-16} --learning_rate $LR \
            --lr_scheduler_type $SCHED --warmup_steps 20 --max_steps $STEPS --logging_steps 20 --save_strategy no \
            --output_dir $SOUT/$1
        run python -m embedrag.evaluate --ckpt $SOUT/$1 --eval_file $SMALL.dev.jsonl --num_docs $EVAL_DOCS \
            --output $SOUT/$1/fidelity.json
    done ;;
esac
