#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import wandb

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)

from transformers import TrainerCallback
import wandb
import torch

class AdamStatsLogger(TrainerCallback):
    def __init__(self):
        self.trainer = None

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not control.should_log or self.trainer is None:
            return

        optimizer = self.trainer.optimizer
        eps = optimizer.param_groups[0]["eps"]

        m_vals = []
        v_vals = []
        effective_lrs = []

        for group in optimizer.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None or param not in optimizer.state:
                    continue

                state_dict = optimizer.state[param]
                m = state_dict.get("exp_avg", None)
                v = state_dict.get("exp_avg_sq", None)

                if m is not None and v is not None:
                    # Adam's effective learning rate: lr * m_t / (sqrt(v_t) + eps)
                    effective_lr = (lr * m / (v.sqrt() + eps)).detach().flatten()
                    
                    m_vals.append(m.detach().flatten())
                    v_vals.append(v.detach().flatten())
                    effective_lrs.append(effective_lr)

        if not m_vals:
            return  # nothing to log

        m_cat = torch.cat(m_vals)
        v_cat = torch.cat(v_vals)
        effective_lrs_cat = torch.cat(effective_lrs)

        # Log the stats: min, max, mean for moments and effective learning rates
        self.trainer._metrics["adam_stats/m_t_mean"] = m_cat.mean().item()
        self.trainer._metrics["adam_stats/m_t_min"] = m_cat.min().item()
        self.trainer._metrics["adam_stats/m_t_max"] = m_cat.max().item()
        self.trainer._metrics["adam_stats/v_t_mean"] = v_cat.mean().item()
        self.trainer._metrics["adam_stats/v_t_min"] = v_cat.min().item()
        self.trainer._metrics["adam_stats/v_t_max"] = v_cat.max().item()
        self.trainer._metrics["adam_stats/lr_effective_mean"] = effective_lrs_cat.mean().item()
        self.trainer._metrics["adam_stats/lr_effective_min"] = effective_lrs_cat.min().item()
        self.trainer._metrics["adam_stats/lr_effective_max"] = effective_lrs_cat.max().item()

        self.log_lm_head_lr(optimizer, model.lm_head.in_features, model.lm_head.out_features)

    def log_lm_head_lr(self, optimizer, in_features, out_features):
        expected_numel = in_features * out_features

        for group in optimizer.param_groups:
            flat_param = group['params'][0]  # ZeRO-2 flattens all into one
            if flat_param.numel() < expected_numel:
                raise ValueError("Flattened param tensor too small")
            
            eps = group["eps"]
            state_dict = optimizer.state[flat_param]
            m = state_dict.get("exp_avg", None)
            v = state_dict.get("exp_avg_sq", None)

            # Adam's effective learning rate: lr * m_t / (sqrt(v_t) + eps)
            effective_lr = (group["lr"] * m / (v.sqrt() + eps)).detach().flatten()

            effective_lr = effective_lr[:expected_numel]

            self.trainer._metrics["adam_stats/lm_head/lr_effective_mean"] = effective_lr.mean().item()
            self.trainer._metrics["adam_stats/lm_head/lr_effective_min"] = effective_lr.min().item()
            self.trainer._metrics["adam_stats/lm_head/lr_effective_max"] = effective_lr.max().item()
            self.trainer._metrics["adam_stats/lm_head/lr_effective_std"] = effective_lr.std().item()
            return

CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
}


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks
