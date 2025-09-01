#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name="eval-revision"
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err

# ========== Job Setup ==========
JOBID=$SLURM_JOB_ID
HOSTNAME=$(hostname)

cleanup_orphans() {
    echo "[$JOBID] Cleaning orphaned processes on $HOSTNAME..."

    # Kill ray/vllm/eval-related processes
    pkill -u $USER -f vllm
    pkill -u $USER -f lighteval
    pkill -u $USER -f evaluate.py
    pkill -u $USER -f ray::
    pkill -u $USER -f raylet
    pkill -u $USER -f gcs_server
    pkill -u $USER -f "python.*ray"
    pkill -u $USER -f nccl

    # Clean ray temp if needed
    RAY_TMP="/tmp/ray"
    [ -d "$RAY_TMP" ] && rm -rf $RAY_TMP/*

    # Clean torch NCCL envs
    export NCCL_ASYNC_ERROR_HANDLING=1
    sleep 5
}

# Early cleanup before starting the job
cleanup_orphans

# Register exit-time cleanup
trap cleanup_orphans EXIT

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
pip install flash-attn --no-build-isolation

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
TASKS="custom|aime25|0|0,custom|aime24|0|0,custom|math_500|0|0,custom|gsm8k|0|0"
# TASKS="custom|math_500|0|0"
MODEL_NAME=$(echo $MODEL_ID | sed 's/\//_/g')
ACCELERATE_USE_DEEPSPEED=false
HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_ARGS="pretrained=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH,temperature:0.6,top_p:0.95}"


OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME"
echo "[$JOBID] Saving results to $OUTPUT_DIR"

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

exit 0


