#!/bin/bash
# Note: If this file is called from loop.sh, then --job-name and --output below will be overridden

#SBATCH --job-name=eval
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:H100:4
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --output=../eval_outputs/%j.%x/.log

# Configuration variables

# GPU_IDS=(0 1 2 3 4 5 6 7)
GPU_IDS=(0 1 2 3)

MIN_PORT=10000; MAX_PORT=65535; MASTER_PORT=$(( RANDOM % (MAX_PORT - MIN_PORT + 1) + MIN_PORT ))

RUN_NAME="" # To be specified
checkpoint_path="" # To be specified
output_dir="../eval_results/${RUN_NAME}_full_steps"
mkdir -p "${output_dir}"
echo "Evaluation results will be saved to: ${output_dir}"


# Arrays of tasks and generation lengths
TASKS=("gsm8k" "math" "countdown" "sudoku")
GEN_LENGTHS=(128 256 512)

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
    
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
    
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --output_dir "${output_dir}" \
      --model_path "GSAI-ML/LLaDA-8B-Instruct" \
      --checkpoint_path "${checkpoint_path}" \
      --full_steps
  done
done


echo "All evaluations completed!"
