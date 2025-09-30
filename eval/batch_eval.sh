#!/bin/bash

# Batch evaluation script for multiple checkpoints with job queue management
# Usage: ./batch_eval.sh RUN_NAME MAX_CONCURRENT_JOBS CHECKPOINT_ITERATIONS...
# Example: ./batch_eval.sh 0904231537 3 1000 2000 3000 4000 5000

# Check if minimum arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 RUN_NAME MAX_CONCURRENT_JOBS CHECKPOINT_ITERATION1 [CHECKPOINT_ITERATION2 ...]"
    echo "Example: $0 0904231537 3 1000 2000 3000 4000 5000"
    exit 1
fi

# Parse command line arguments
RUN_NAME="$1"
MAX_CONCURRENT_JOBS="$2"
shift 2
CHECKPOINT_ITERATIONS=("$@")

echo "Starting batch evaluation for run: $RUN_NAME"
echo "Maximum concurrent jobs: $MAX_CONCURRENT_JOBS"
echo "Checkpoint iterations to evaluate: ${CHECKPOINT_ITERATIONS[*]}"

# Function to count running jobs with our specific pattern
count_running_jobs() {
    local run_name="$1"
    # Count jobs that match our naming pattern and are in RUNNING or PENDING state
    squeue -u $(whoami) -h -o "%j %t" | grep -E "^eval_${run_name}_[0-9]+ (R|PD)$" | wc -l
}

# Function to wait for job slots to become available
wait_for_slot() {
    local run_name="$1"
    local max_jobs="$2"
    
    while [ $(count_running_jobs "$run_name") -ge $max_jobs ]; do
        echo "$(date): Maximum concurrent jobs ($max_jobs) reached. Waiting for slot..."
        sleep 30  # Check every 30 seconds
    done
}

# Function to submit evaluation job for a specific checkpoint
submit_eval_job() {
    local run_name="$1"
    local checkpoint_iter="$2"
    local checkpoint_path="../outputs/${run_name}/checkpoint-${checkpoint_iter}"
    local output_dir="../eval_results/${run_name}"
    local job_name="eval_${run_name}_${checkpoint_iter}"
    local log_file="../eval_outputs/${job_name}.log"
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "Warning: Checkpoint directory $checkpoint_path does not exist. Skipping iteration $checkpoint_iter"
        return 1
    fi
    
    # Create output directories if they don't exist
    mkdir -p "$output_dir"
    mkdir -p "../eval_outputs"
    
    echo "$(date): Submitting evaluation job for checkpoint iteration $checkpoint_iter"
    
    # Create a temporary script with the specific checkpoint configuration
    local temp_script="/tmp/eval_${run_name}_${checkpoint_iter}.sh"
    
    # Copy the base evaluation script and modify it
    sed -e "s|^RUN_NAME=.*|RUN_NAME=\"$run_name\"|" \
        -e "s|checkpoint_path=.*|checkpoint_path=\"$checkpoint_path\"|" \
        -e "s|output_dir=.*|output_dir=\"$output_dir\"|" \
        run_eval_1.sh > "$temp_script"
    
    # Submit the job with custom name and output
    sbatch --job-name="$job_name" \
           --output="$log_file" \
           "$temp_script"
    
    local job_status=$?
    
    # Clean up temporary script
    rm -f "$temp_script"
    
    if [ $job_status -eq 0 ]; then
        echo "$(date): Successfully submitted job $job_name"
    else
        echo "$(date): Failed to submit job $job_name"
        return 1
    fi
}

# Main execution loop
echo "$(date): Starting batch evaluation process"

for checkpoint_iter in "${CHECKPOINT_ITERATIONS[@]}"; do
    # Wait for a slot to become available
    wait_for_slot "$RUN_NAME" "$MAX_CONCURRENT_JOBS"
    
    # Submit the evaluation job
    submit_eval_job "$RUN_NAME" "$checkpoint_iter"
    
    # Small delay to prevent overwhelming the scheduler
    sleep 2
done

echo "$(date): All evaluation jobs have been submitted"
echo "Monitor job status with: squeue -u \$(whoami)"
echo "Check logs in: ../eval_outputs/"
echo "Results will be saved in: ../eval_results/${RUN_NAME}/"

# Optional: Wait for all jobs to complete
read -p "Do you want to wait for all jobs to complete? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "$(date): Waiting for all evaluation jobs to complete..."
    
    while [ $(count_running_jobs "$RUN_NAME") -gt 0 ]; do
        running_count=$(count_running_jobs "$RUN_NAME")
        echo "$(date): $running_count evaluation jobs still running..."
        sleep 60  # Check every minute
    done
    
    echo "$(date): All evaluation jobs completed!"
    echo "Check results in: ../eval_results/${RUN_NAME}/"
fi
