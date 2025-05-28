# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    eval_dataset_ratio: Optional[int] = field(
        default=1.0,
        metadata={"help": "The ratio of the eval dataset to use for evaluation."},
    )

    # Actor-Critic
    num_value_tokens: int = field(default=5, metadata={"help": "The number of value tokens to use."})
    value_type: str = field(default="digit", metadata={"help": "The type of value tokens to use. Possible values: digit, token."})
    value_inference_strategy: str = field(default="marginalization", metadata={"help": "The strategy to use for value inference. Possible values: marginalization, mode."})
    value_loss_weight: float = field(default=0.01, metadata={"help": "The weight of the value loss."})
    value_loss: str = field(default="hard_label", metadata={"help": "The loss function to use for the value function. Possible values: hard_label, soft_label."})
    normalize_advantages: bool = field(default=False, metadata={"help": "Whether to normalize the advantages."})
    advantage_target_std: float = field(default=1.0, metadata={"help": "The target std of the advantages."})
    anneal_advantage_std: bool = field(default=False, metadata={"help": "Whether to anneal the advantage std."})
    reward_intervals: list[tuple[float, float]] = field(default=None, metadata={"help": "The intervals for the rewards."})

    # Smooth GRPO
    smooth_logprobs: bool = field(default=False, metadata={"help": "Whether to smooth the logprobs."})
    softplus_alpha: float = field(default=0.0, metadata={"help": "The alpha for the softplus function."})

    # MaxEnt RL
    entropy_alpha: float = field(default=0.0, metadata={"help": "The alpha for the entropy loss."})
    entropy_estimator: str = field(default="logprobs", metadata={"help": "The estimator for the entropy loss. Possible values: logprobs, entropy."})



@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
