#!/bin/bash
# Note: If this file is called from loop.sh, then --job-name and --output below will be overridden

#SBATCH --job-name=dmpo
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:H200:8
#SBATCH --time=02:00:00
# max 16 GPU hours, i.e., time <= 16h / num of GPUs
#SBATCH --mem-per-gpu=80G
# maximum GPU RAM, 141G for H200, 94G for H100
# in the current setting, 40G is enough for num_replicates=2 and 80G is enough for num_replicates=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --output=../outputs/%j.%x/.log

NUM_PROCESSES=8 # this is the number of GPUs

# If this file is called from loop.sh, then receive environment variables from it;
# Otherwise, generate randomly or based on the current timestamp
export WANDB_RUN_ID="${WANDB_RUN_ID:-"$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c8)"}"
echo "WandB Run ID: ${WANDB_RUN_ID}"

RUN_NAME="${RUN_NAME:-"$(date '+%m%d%H%M%S')"}"
echo "Run name: ${RUN_NAME}"



DATASET="${DATASET:-"gsm8k"}"

LOG_DIR="${LOG_DIR:-"../outputs/${DATASET}/${RUN_NAME}"}"
mkdir -p "${LOG_DIR}"
echo "Training logs and checkpoints will be saved to: ${LOG_DIR}"

MIN_PORT=10000; MAX_PORT=65535; RANDOM_PORT=$(( RANDOM % (MAX_PORT - MIN_PORT + 1) + MIN_PORT ))

# Check if there are previous checkpoints. If found, resume training from the last one
LAST_CHECKPOINT=$(ls -d "${LOG_DIR}/checkpoint-"* 2>/dev/null | sort -V | tail -n 1)
if [ -n "${LAST_CHECKPOINT}" ]; then
    echo "Found the last checkpoint: ${LAST_CHECKPOINT}, resuming training from this checkpoint."
    RESUME_FLAG="--resume_from_checkpoint ${LAST_CHECKPOINT}"
else
    echo "No previous checkpoints found, starting new training."
    RESUME_FLAG=""
fi

srun accelerate launch \
    --config_file accelerate.yaml \
    --num_processes $NUM_PROCESSES \
    --main_process_port $RANDOM_PORT dmpo_train.py \
    --config dmpo_train_config.yaml \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir $LOG_DIR \
    ${RESUME_FLAG} \
    --advantage_centering true \
    --advantage_centering_unbias false \
    --advantage_centering_neg true \
    --compute_ref_log_prob_elbo true \
    --compute_ref_log_prob_elbo_size 4 \
    --centering_strength 1.0 \
    --alpha 0.04 \
    --num_generations 16 \
    --num_iterations 8 \
    --per_device_train_batch_size 4 \
    --sync_ref_model true \
    --ref_model_sync_steps 64 \
    --generation_batch_size 8 \
    --save_total_limit 10 \
    --loss_mask_non_eos false \
    --loss "wdce" \
    --use_fast_sampler "fast_dllm" \
    --sampler "pd_cache_prefix" \
    --temperature 0.2 \
    --pretrained_model_path "GSAI-ML/LLaDA-8B-Instruct"

# Note: append additional training arguments above, but avoid changing the config file