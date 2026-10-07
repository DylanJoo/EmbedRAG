#!/bin/sh
#SBATCH --job-name=embedrag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# usage: sbatch slurm_scripts/embedrag.sh {data|align|task|rl}
# DEC: meta-llama/Llama-3.2-1B-Instruct | Qwen/Qwen3-4B | google/gemma-3-4b-it

source ~/.bashrc
enter_conda
conda activate crc
cd ~/EmbedRAG

DEC=${DEC:-meta-llama/Llama-3.2-1B-Instruct}
NAME=${NAME:-$(basename $DEC)-k8}
OUT=${MODEL_DIR}/embedrag/${NAME}
COMMON="--bf16 --gradient_checkpointing --per_device_train_batch_size 4 --gradient_accumulation_steps 4
        --lr_scheduler_type cosine --warmup_ratio 0.03 --logging_steps 10 --save_steps 1000 --report_to wandb"

case $1 in
data)
    python -m embedrag.build_data pretrain --input ${CORPUS:-data/corpus.jsonl} --output data/embedrag/pretrain.jsonl
    python -m embedrag.build_data mds --input data/mds-5k-greedy-1.jsonl --output data/embedrag/mds.jsonl ;;
align)  # stage 1: encoder LoRA + projector, frozen decoder
    accelerate launch --config_file configs/default_train.yaml -m embedrag.train $COMMON \
        --decoder $DEC --num_mem 8 --pooling memory \
        --train_files data/embedrag/pretrain.jsonl \
        --learning_rate 1e-4 --max_steps 20000 --run_name ${NAME}-align --output_dir ${OUT}/align ;;
task)   # stage 2: + decoder LoRA, CE + distillation from full-text teacher
    accelerate launch --config_file configs/default_train.yaml -m embedrag.train $COMMON \
        --init_from ${OUT}/align --decoder_lora --train_decoder --kd_weight 1.0 \
        --train_files data/embedrag/mds.jsonl,data/embedrag/pretrain.jsonl --tasks summ,rel,single,ae_multi \
        --learning_rate 5e-5 --max_steps 10000 --run_name ${NAME}-task --output_dir ${OUT}/task ;;
rl)     # stage 3: GRPO on decoder LoRA
    python -m embedrag.rl --init_from ${OUT}/task --train_files data/embedrag/mds.jsonl --tasks summ \
        --output_dir ${OUT}/rl ;;
esac
