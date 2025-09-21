#!/bin/bash

# Script to submit a sequence of training jobs with dependencies
# Usage: ./submit_job_sequence.sh JOB_NAME MODEL TASK CONFIG_SUFFIX ACCELERATOR GPU_TYPE NUM_JOBS [DEPENDENCY]

# Check if correct number of arguments is provided
if [ $# -lt 7 ] || [ $# -gt 8 ]; then
    echo "Usage: $0 JOB_NAME MODEL TASK CONFIG_SUFFIX ACCELERATOR GPU_TYPE NUM_JOBS [DEPENDENCY]"
    echo "Example: $0 'my-training-seq' 'Qwen2.5-Math-7B' 'grpo' 'oatcloud' 'zero2_oatcloud' 'h100' 5"
    echo "Example with dependency: $0 'my-training-seq' 'Qwen2.5-Math-7B' 'grpo' 'oatcloud' 'zero2_oatcloud' 'h100' 5 12345"
    exit 1
fi

# Parse arguments
JOB_NAME="$1"
MODEL="$2"
TASK="$3"
CONFIG_SUFFIX="$4"
ACCELERATOR="$5"
GPU_TYPE="$6"
NUM_JOBS="$7"
DEPENDENCY="${8:-}"  # Optional dependency for the first job

# Validate GPU_TYPE
if [[ "$GPU_TYPE" != "h100" && "$GPU_TYPE" != "a100" ]]; then
    echo "Error: GPU_TYPE must be either 'h100' or 'a100'"
    exit 1
fi

# Validate NUM_JOBS
if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]] || [ "$NUM_JOBS" -lt 1 ]; then
    echo "Error: NUM_JOBS must be a positive integer"
    exit 1
fi

# Determine which training script to use based on GPU type
if [[ "$GPU_TYPE" == "h100" ]]; then
    TRAIN_SCRIPT="train_oatcloud_4h100.sh"
elif [[ "$GPU_TYPE" == "a100" ]]; then
    TRAIN_SCRIPT="train_oatcloud_4a100.sh"
fi

echo "=== Job Sequence Submission ==="
echo "Job Name: $JOB_NAME"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Config Suffix: $CONFIG_SUFFIX"
echo "Accelerator: $ACCELERATOR"
echo "GPU Type: $GPU_TYPE"
echo "Training Script: $TRAIN_SCRIPT"
echo "Number of Jobs: $NUM_JOBS"
if [ -n "$DEPENDENCY" ]; then
    echo "Initial Dependency: $DEPENDENCY"
else
    echo "Initial Dependency: None"
fi
echo "================================"

# Initialize variables for dependency chain
PREVIOUS_JOB_ID="$DEPENDENCY"

# Submit jobs in sequence with dependencies
for ((i=1; i<=$NUM_JOBS; i++)); do
    # Generate a random 4-digit seed
    SEED=$(shuf -i 1000-9999 -n 1)
    
    # Create job name with sequence number
    SEQUENCE_JOB_NAME="${JOB_NAME}-${i}"
    
    echo "Submitting job $i/$NUM_JOBS with seed $SEED..."
    
    # Prepare sbatch command
    SBATCH_CMD="sbatch"
    
    # Add dependency if this is not the first job or if there's an initial dependency
    if [ -n "$PREVIOUS_JOB_ID" ]; then
        SBATCH_CMD="$SBATCH_CMD --dependency=$PREVIOUS_JOB_ID"
        if [ "$i" -eq 1 ] && [ -n "$DEPENDENCY" ]; then
            echo "  Adding initial dependency on job $PREVIOUS_JOB_ID"
        elif [ "$i" -gt 1 ]; then
            echo "  Adding dependency on previous job $PREVIOUS_JOB_ID"
        fi
    fi
    
    # Add job name
    SBATCH_CMD="$SBATCH_CMD --job-name=\"$SEQUENCE_JOB_NAME\" --exclude=oat18,oat17"
    
    # Submit the job
    echo "  Command: $SBATCH_CMD $TRAIN_SCRIPT $MODEL $TASK $CONFIG_SUFFIX $ACCELERATOR $SEED"
    
    # Execute the sbatch command and capture the job ID
    JOB_OUTPUT=$($SBATCH_CMD "$TRAIN_SCRIPT" "$MODEL" "$TASK" "$CONFIG_SUFFIX" "$ACCELERATOR" "$SEED")
    
    # Extract job ID from output (format: "Submitted batch job 12345")
    NEW_JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+')
    
    if [ -n "$NEW_JOB_ID" ]; then
        echo "  ✓ Job submitted successfully with ID: $NEW_JOB_ID"
        PREVIOUS_JOB_ID="$NEW_JOB_ID"
    else
        echo "  ✗ Failed to submit job or extract job ID"
        echo "  Output: $JOB_OUTPUT"
        exit 1
    fi
    
    echo ""
done

echo "=== Job Sequence Complete ==="
echo "All $NUM_JOBS jobs have been submitted with dependencies."
echo "First job ID: $PREVIOUS_JOB_ID"
echo "Last job ID: $PREVIOUS_JOB_ID"
echo "Use 'squeue -u $USER' to monitor job status"
echo "Use 'scancel $PREVIOUS_JOB_ID' to cancel the entire sequence"
