#!/bin/bash
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name="act-pm"
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err

source ./.env
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/
export TMPDIR=/scratch/${USER}/tmp
mkdir -p $TMPDIR

rm -rf ~/.cache/pip
export PIP_NO_CACHE_DIR=1
export PIP_DEFAULT_TIMEOUT=100
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Setup environment
/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/maxent-rl-r1/environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate maxent-r1

cd ~/maxent-rl-r1
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -e ".[dev]"
pip install flash-attn --no-build-isolation

nvidia-smi
huggingface-cli login --token $HUGGINGFACE_WRITETOKEN

# Required args
TASK_NAME=$1         # e.g. "eval_llm"
MODEL_ID=$2          # e.g. "Qwen/Qwen2.5-Math-7B"
MAX_STEPS=$3         # e.g. 5000
WANDB_RUN_PATH=$4    # e.g. "my-entity/my-project/abc123"
MAX_MODEL_LENGTH=$5  # e.g. 32768

# Optional args
[ -z "$6" ] && TENSOR_PARALLEL=False || TENSOR_PARALLEL=$6
[ -z "$7" ] && TRUST_REMOTE_CODE=False || TRUST_REMOTE_CODE=$7
SYSTEM_PROMPT=$8

NUM_GPUS=$(nvidia-smi -L | wc -l)
# TASKS="custom|aime24|0|0,custom|math_500|0|0,custom|gpqa:diamond|0|0"
TASKS="custom|aime25|0|0"
# TASKS="custom|aime25|0|0,custom|aime24|0|0,custom|math_500|0|0,custom|gpqa:diamond|0|0,custom|minervamath|0|0,custom|gsm8k|0|0,custom|amc23|0|0"
MODEL_NAME=$(echo $MODEL_ID | sed 's/\//_/g')
ACCELERATE_USE_DEEPSPEED=false
HF_HUB_ENABLE_HF_TRANSFER=1

echo "Fetching model revisions..."
MODEL_REVISIONS=$(python ~/maxent-rl-r1/src/open_r1/utils/list_revisions.py $MODEL_ID $MAX_STEPS)
echo "Model revisions: $MODEL_REVISIONS"

for entry in $MODEL_REVISIONS; do
    STEP=$(echo $entry | cut -d':' -f1)
    MODEL_REVISION=$(echo $entry | cut -d':' -f2)
    echo "Running evaluation for step $STEP, revision $MODEL_REVISION"

    if [ "$TENSOR_PARALLEL" = "True" ]; then
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        MODEL_ARGS="pretrained=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH,temperature:0.6,top_p:0.95}"
    else
        MODEL_ARGS="pretrained=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH,temperature:0.6,top_p:0.95}"
    fi

    OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME"
    echo "Saving eval results to $OUTPUT_DIR"

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

    echo "Logging to WandB..."
    python ~/maxent-rl-r1/src/open_r1/utils/log_to_wandb.py $WANDB_RUN_PATH "$OUTPUT_DIR/results/" $STEP

    echo "Cleaning up $OUTPUT_DIR..."
    rm -rf $OUTPUT_DIR

    # echo "Cleaning GPU memory..."
    # python -c "import torch; torch.cuda.empty_cache()"
    # sleep 10  # allow VLLM subprocesses to exit cleanly
done

echo "All evaluations completed."
