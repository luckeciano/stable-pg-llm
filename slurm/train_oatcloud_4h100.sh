#!/bin/bash
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --job-name="r1-token-entropy"
#SBATCH --partition=h100
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err

source ./.env
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets
export HF_HOME=$HOME/.cache/huggingface

export TMPDIR=/scratch/${USER}/tmp
mkdir -p $TMPDIR
BUILD_DIR=/scratch-ssd/${USER}/conda_envs/pip-build

# Clean pip cache and set pip configurations
rm -rf ~/.cache/pip
export PIP_NO_CACHE_DIR=1
export PIP_DEFAULT_TIMEOUT=100
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Check number of GPUs
echo "Number of GPUs: $(SLURM_JOB_GPUS)"

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/maxent-rl-r1/environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate maxent-r1-lucelo

# Installing flash-attn
pip install flash-attn --no-build-isolation

cd ~/maxent-rl-r1
echo "pwd: $(pwd)"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -e ".[dev]"

echo $TMPDIR

nvidia-smi

huggingface-cli login --token $HUGGINGFACE_WRITETOKEN

echo "START TIME: $(date)"

MODEL=$1 #e.g., Qwen2.5-Math-7B
TASK=$2 #e.g., grpo
CONFIG_SUFFIX=$3 #e.g., oatcloud
ACCELERATOR=$4 #e.g., zero2_oatcloud
SEED=$5 #e.g., 42
OPTIONAL_ARGS=""

# Training setup
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
# Due to conflicts between Accelerate's DeepSpeed configs and Transformers' TrainingArguments, we need to parse the gradient accumulation steps from the config file to ensure they match
CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')

# Check if we are running vLLM during training to adjust the world size
if grep -q 'use_vllm:\s*true' "$CONFIG_FILE"; then
    USE_VLLM="true"
else
    USE_VLLM="false"
fi

if [[ "$USE_VLLM" == "true" ]]; then
    WORLD_SIZE=$(($WORLD_SIZE - 1))
fi

# Split the string into individual arguments
IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"

# Loop through the arguments and find the one with "--gradient_accumulation_steps"
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "--gradient_accumulation_steps="* ]]; then
        # Extract the value after the equals sign
        GRAD_ACC_STEPS="${arg#*=}"
        break  # Exit the loop once we find the desired argument
    fi
done

echo "Gradient accumulation steps: $GRAD_ACC_STEPS"
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Generate a random port between 29500 and 29750
MASTER_PORT=$(shuf -i 29500-29750 -n 1)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NUM_NODES: $NUM_NODES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "GRAD_ACC_STEPS: $GRAD_ACC_STEPS"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "OPTIONAL_ARGS: $OPTIONAL_ARGS"
echo "ACCELERATOR: $ACCELERATOR"
echo "TASK: $TASK"
echo "SLURM_PROCID: $SLURM_PROCID"

export CMD=" \
    src/open_r1/$TASK.py --config $CONFIG_FILE $OPTIONAL_ARGS --seed $SEED
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "

# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# Need to disable P2P to avoid hanging broadcast in A100s
export NCCL_P2P_DISABLE=1


# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"
