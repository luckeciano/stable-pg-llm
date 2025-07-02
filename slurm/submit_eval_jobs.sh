#!/bin/bash
TASK_NAME=$1 # "eval_llm"
MODEL_ID_PREFIX=$2 # "Qwen-2.5-7B-GRPO-NoBaseline-FisherMaskGlobal-1e-11"
MAX_STEPS=$3 # 11
FIRST_STEP=$4 # 100
INTERVAL=$5 # 10
WANDB_ENTITY=$6 # "max-ent-llms"
WANDB_PROJECT=$7 # "PolicyGradientStability"
HF_PREFIX=$8 # "luckeciano"
WANDB_PREFIX=$9 # "/scratch-ssd/lucelo/grad-stability/data"
MAX_MODEL_LENGTH=${10} # 4096
GPU_TYPE=${11} # "h100"
TENSOR_PARALLEL=${12:-False}
TRUST_REMOTE_CODE=${13:-False}
SYSTEM_PROMPT=${14}

echo "Fetching model revisions..."
# Run the Python script and capture both stdout and stderr
PYTHON_OUTPUT=$(python ~/maxent-rl-r1/src/open_r1/utils/list_revisions.py $MODEL_ID_PREFIX $MAX_STEPS $FIRST_STEP $INTERVAL $WANDB_ENTITY $WANDB_PROJECT $HF_PREFIX $WANDB_PREFIX 2>&1)

# Extract the model revisions (lines that contain 4 colons)
MODEL_REVISIONS=$(echo "$PYTHON_OUTPUT" | grep -E "^[^:]*:[^:]*:[^:]*:[^:]*$")

# Print the Python script output (excluding the model revision lines)
echo "$PYTHON_OUTPUT" | grep -v -E "^[^:]*:[^:]*:[^:]*:[^:]*$"

# Group revisions by model
declare -A model_revisions
for entry in $MODEL_REVISIONS; do
    if [[ $entry == *:*:*:* ]]; then
        MODEL_ID=$(echo $entry | cut -d':' -f1)
        STEP=$(echo $entry | cut -d':' -f2)
        MODEL_REVISION=$(echo $entry | cut -d':' -f3)
        WANDB_RUN_PATH=$(echo $entry | cut -d':' -f4)
        model_revisions["$MODEL_ID"]="${model_revisions["$MODEL_ID"]} $STEP:$MODEL_REVISION:$WANDB_RUN_PATH"
    fi
done

# Iterate per model, then per revision
for MODEL_ID in "${!model_revisions[@]}"; do
    echo "Processing model: $MODEL_ID"
    PREV_JOB_ID=""
    
    for entry in ${model_revisions["$MODEL_ID"]}; do
        STEP=$(echo $entry | cut -d':' -f1)
        MODEL_REVISION=$(echo $entry | cut -d':' -f2)
        WANDB_RUN_PATH=$(echo $entry | cut -d':' -f3)
        echo "Preparing job for MODEL=$MODEL_ID, STEP=$STEP, REVISION=$MODEL_REVISION, WANDB_PATH=$WANDB_RUN_PATH"

        CMD="sbatch --exclude=oat18"
        if [ -n "$PREV_JOB_ID" ]; then
            CMD+=" --dependency=afterany:$PREV_JOB_ID"
        fi

        if [ "$GPU_TYPE" == "h100" ]; then
            JOB_SUBMIT_OUTPUT=$($CMD evaluate_revision_h100.sh \
                "$TASK_NAME" "$MODEL_ID" "$MODEL_REVISION" "$STEP" \
                "$WANDB_RUN_PATH" "$MAX_MODEL_LENGTH" \
                "$TENSOR_PARALLEL" "$TRUST_REMOTE_CODE" "$SYSTEM_PROMPT")
        else
            JOB_SUBMIT_OUTPUT=$($CMD evaluate_revision.sh \
                "$TASK_NAME" "$MODEL_ID" "$MODEL_REVISION" "$STEP" \
                "$WANDB_RUN_PATH" "$MAX_MODEL_LENGTH" \
                "$TENSOR_PARALLEL" "$TRUST_REMOTE_CODE" "$SYSTEM_PROMPT")
        fi

        echo "$JOB_SUBMIT_OUTPUT"
        PREV_JOB_ID=$(echo "$JOB_SUBMIT_OUTPUT" | grep -oP '\d+')
    done
done
