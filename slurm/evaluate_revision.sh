#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --job-name="eval-revision"
#SBATCH --partition=normal
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err
#SBATCH --requeue
#SBATCH --time=00:30:00

# ========== Job Setup ==========
JOBID=$SLURM_JOB_ID
HOSTNAME=$(hostname)

# ========== Environment ==========
source ./.env
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets
export HF_HOME=$HOME/.cache/huggingface
export TMPDIR=/scratch/${USER}/tmp
mkdir -p $TMPDIR

rm -rf ~/.cache/pip
export PIP_NO_CACHE_DIR=1
export PIP_DEFAULT_TIMEOUT=100
export PYTHONWARNINGS="ignore::DeprecationWarning"

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/maxent-rl-r1/environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate maxent-r1-lucelo

cd ~/maxent-rl-r1
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -e ".[dev]"
pip install flash-attn==2.7.4.post1 --no-build-isolation

nvidia-smi
huggingface-cli login --token $HUGGINGFACE_WRITETOKEN

# ========== Arguments ==========
TASK_NAME=$1
MODEL_ID=$2
MODEL_REVISION=$3
STEP=$4
WANDB_RUN_PATH=$5
MAX_MODEL_LENGTH=$6
TENSOR_PARALLEL=${7:-False}
TRUST_REMOTE_CODE=${8:-False}
SYSTEM_PROMPT=$9

# print all arguments
echo "[$JOBID] Arguments: $@"

NUM_GPUS=$(nvidia-smi -L | wc -l)
TASKS="custom|aime25|0|0,custom|aime24|0|0,custom|math_500|0|0,custom|gpqa:diamond|0|0,custom|minervamath|0|0,custom|gsm8k|0|0,custom|amc23|0|0,custom|olympiadbench|0|0"
# TASKS="custom|math_500|0|0"
MODEL_NAME=$(echo $MODEL_ID | sed 's/\//_/g')
ACCELERATE_USE_DEEPSPEED=false
HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ENABLE_HF_TRANSFER=0

# ========== Retry Management ==========
ATTEMPT_FILE="/scratch-ssd/$USER/retry_logs/${MODEL_NAME}_${STEP}.attempt"
mkdir -p "$(dirname "$ATTEMPT_FILE")"
ATTEMPT_COUNT=$(cat "$ATTEMPT_FILE" 2>/dev/null || echo 0)
((ATTEMPT_COUNT++))
echo "$ATTEMPT_COUNT" > "$ATTEMPT_FILE"

if [ "$ATTEMPT_COUNT" -gt 3 ]; then
    echo "[$JOBID] Max retry attempts reached for $MODEL_REVISION (step $STEP)"
    exit 1
fi

echo "[$JOBID] Attempt $ATTEMPT_COUNT for $MODEL_REVISION (step $STEP)"

# ========== Build Model Args ==========
if [ "$TENSOR_PARALLEL" = "True" ]; then
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    MODEL_ARGS="pretrained=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH,temperature:0.6,top_p:0.95}"
else
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    MODEL_ARGS="pretrained=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH,temperature:0.6,top_p:0.95}"
fi

OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME"
echo "[$JOBID] Saving results to $OUTPUT_DIR"

echo "HF_HUB_ENABLE_HF_TRANSFER: $HF_HUB_ENABLE_HF_TRANSFER"

# ========== Run Evaluation ==========
if [[ $TASKS == *"custom"* ]]; then
    lighteval vllm "$MODEL_ARGS" $TASKS \
        --custom-tasks "src/open_r1/evaluate.py" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR \
        --save-details \
        ${SYSTEM_PROMPT:+--system-prompt "$SYSTEM_PROMPT"}
else
    lighteval vllm "$MODEL_ARGS" $TASKS \
        --use-chat-template \
        --output-dir $OUTPUT_DIR \
        --save-details \
        ${SYSTEM_PROMPT:+--system-prompt "$SYSTEM_PROMPT"}
fi

EXIT_CODE=$?

# ========== Post-Eval ==========
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[$JOBID] Logging results to WandB..."
    python ~/maxent-rl-r1/src/open_r1/utils/log_to_wandb.py $WANDB_RUN_PATH "$OUTPUT_DIR/results/" $STEP
else
    echo "[$JOBID] Evaluation failed with exit code $EXIT_CODE. Requeuing..."
    scontrol requeue $JOBID
fi

echo "[$JOBID] Cleaning up output directory..."
rm -rf "$OUTPUT_DIR"

sleep 10

exit $EXIT_CODE


