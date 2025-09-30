#!/bin/bash

# Generate a unique WandB Run ID that will be shared by all subsequent submissions,
# ensuring that all training information is recorded in the same WandB Run;
# Or receive the manually specified Run ID to resume recording

export WANDB_RUN_ID="${WANDB_RUN_ID:-"$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c8)"}"
echo "WandB Run ID: ${WANDB_RUN_ID}"

export RUN_NAME="${RUN_NAME:-"$(date '+%m%d%H%M%S')"}"
echo "Run name: ${RUN_NAME}"


# Set log directory for storing training logs and checkpoints
export LOG_DIR="/storage/ice-shared/ae3530b/yzhu738/PCE-outputs/math500/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
echo "Training logs and checkpoints will be saved to: ${LOG_DIR}"

# Set relevant environment variables
export ROOT_DIR="/home/hice1/yzhu738/scratch"
export WANDB_ENTITY="yzhu738"
export WANDB_PROJECT="pce-mdm-d1"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export TORCH_HOME="${ROOT_DIR}/.cache/torch"
export TMPDIR="${ROOT_DIR}/.tmp"
export WANDB_DIR="${ROOT_DIR}/.wandb"
export WANDB_CACHE_DIR="${ROOT_DIR}/.cache/wandb"
export WANDB_CONFIG_DIR="${ROOT_DIR}/.config/wandb"
export WANDB_DATA_DIR="${ROOT_DIR}/.cache/wandb-data/"
export WANDB_ARTIFACT_DIR="${ROOT_DIR}/.artifacts"
export TRANSFORMERS_VERBOSITY="detail" # "detail", "debug", "info", "warning" (default), "error", "critical"

# Set the dataset for training, default gsm8k
DATASET="${DATASET:-"math"}"

# Set the sbatch file name to loop, default run_pce.sh
SBATCH_FILE="${SBATCH_FILE:-"run_dmpo.sh"}"

# Set maximum number of loops (default 5) and start training
MAX_LOOPS="${MAX_LOOPS:-15}"
for COUNTER in $(seq 1 $MAX_LOOPS); do
    echo "-----------------------------------------------------------------"
    echo "Submission round ${COUNTER}/${MAX_LOOPS}"

    sbatch --wait \
           --job-name="${RUN_NAME}" \
           --output="${LOG_DIR}/%j.log" \
           --export=ALL,WANDB_RESUME="allow",DATASET="${DATASET}" \
           ${SBATCH_FILE}

    # --wait indicates to wait for the job to complete

    sleep 5 # Add a short delay to ensure the job status is updated

    # Now check the status of the last submission to determine the exit reason
    echo "sbatch task interrupted. Checking if it is a timeout issue..."

    # Check the status of the most recent job
    LAST_JOB_ID=$(sacct -n -X --format=JobID,State --name="${RUN_NAME}" --user="$USER" | tail -n 1 | awk '{print $1}')
    LAST_JOB_STATE=$(sacct -n -X --format=JobID,State --name="${RUN_NAME}" --user="$USER" | tail -n 1 | awk '{print $2}')

    # If the job is in a TIMEOUT state, submit a new job; otherwise, exit the loop
    if [ "${LAST_JOB_STATE}" == "TIMEOUT" ]; then
        echo "Job status: TIMEOUT. Submit the job again."
    elif [ "${LAST_JOB_STATE}" == "COMPLETED" ]; then
        echo "Job status: COMPLETED. This may still be a TIMEOUT. Submit the job again."
    elif [ -z "${LAST_JOB_ID}" ]; then # rarely happens
        echo "No recent job record found. Exiting the loop."
        break
    else # Including CANCELLED, FAILED, etc.
        echo "Job status: ${LAST_JOB_STATE}. Exiting the loop."
        break
    fi
    
done