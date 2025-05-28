#!/bin/bash
TASK_NAME=$1
MODEL_ID=$2
MAX_STEPS=$3
FIRST_STEP=$4
INTERVAL=$5
WANDB_RUN_PATH=$6
MAX_MODEL_LENGTH=$7
TENSOR_PARALLEL=${8:-False}
TRUST_REMOTE_CODE=${9:-False}
SYSTEM_PROMPT=$10

echo "Fetching model revisions..."
MODEL_REVISIONS=$(python ~/maxent-rl-r1/src/open_r1/utils/list_revisions.py $MODEL_ID $MAX_STEPS $FIRST_STEP $INTERVAL)

PREV_JOB_ID=""

for entry in $MODEL_REVISIONS; do
    STEP=$(echo $entry | cut -d':' -f1)
    MODEL_REVISION=$(echo $entry | cut -d':' -f2)
    echo "Preparing job for STEP=$STEP, REVISION=$MODEL_REVISION"

    CMD="sbatch --exclude=oat18"
    if [ -n "$PREV_JOB_ID" ]; then
        CMD+=" --dependency=afterany:$PREV_JOB_ID"
    fi

    JOB_SUBMIT_OUTPUT=$($CMD evaluate_revision.sh \
        "$TASK_NAME" "$MODEL_ID" "$MODEL_REVISION" "$STEP" \
        "$WANDB_RUN_PATH" "$MAX_MODEL_LENGTH" \
        "$TENSOR_PARALLEL" "$TRUST_REMOTE_CODE" "$SYSTEM_PROMPT")

    echo "$JOB_SUBMIT_OUTPUT"
    PREV_JOB_ID=$(echo "$JOB_SUBMIT_OUTPUT" | grep -oP '\d+')
done
