# Stabilizing Policy Gradients For Sample-Efficient Reinforcement Learning in LLM Reasoning

Status: Under submission (active development)

## Building the environment

Our code has been successfully tested on 4×80GB A100/H100 GPUs with CUDA 12.9. The following commands will create a Conda environment with all the required dependencies:

```bash
  conda env create -n capo
  conda activate capo
  pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Run the Code

We provide bash scripts for running our training and evaluation in SLURM-based clusters.

### Training
```bash
bash slurm/submit_job_sequence.sh <JOB_NAME> <MODEL> grpo <RECIPE> oatcloud zero2_oatcloud <GPU_TYPE> <NUM_SEEDS>
```

For reproducing the experiments in the paper, use the following recipes:

- GRPO: `standard_grpo`
- CAPO: `config_grpo_efficient_grpo_adam_fisher_mask_token_1e-4_hessian_mask_token_0.01`
- Dr.GRPO: `drgrpo_efficient_base`
- REINFORCE: `grpo_efficient_nobaseline_adam`
- Dr.CAPO: `config_drgrpo_efficient_adam_fisher_mask_token_1e-3_hessian_mask_token_5e-4`
- ReinCAPO: `config_grpo_efficient_nobaseline_adam_fisher_mask_token_1e-5_hessian_mask_token_0.1`

### Evaluation
During evaluation, we cross-reference wandlogs and HF checkpoints. After evaluation, you will see eval metrics in the wandb logs.

```bash
bash submit_eval_jobs.sh <JOB_NAME> <MODEL> <FINAL_CHECKPOINT__STEP> <INITIAL_CHECKPOINT_STEP> <STEPS_INTERVAL> <WANDB_ENTITY> <WANDB_PROJECT> <WANDB_USER> <EXP_PREFIX> <TOTAL_SEQ_LENGTH> <GPU_TYPE>
```