from collections import defaultdict
import time
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import wandb
from tqdm import tqdm
from packaging import version
from torch.utils.data import DataLoader, Sampler


import transformers
import torch.nn.functional as F
from transformers import Trainer
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import EvalLoopOutput, has_length, denumpify_detensorize
from transformers.utils import logging
from transformers import LogitsProcessorList, LogitsProcessor
from accelerate.utils import broadcast_object_list, gather, gather_object

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import (
    pad,
    selective_log_softmax,
)

from open_r1.modifiable_grpo_trainer import ModifiableGRPOTrainer, RepeatRandomSampler


logger = logging.get_logger(__name__)

class MinProbabilityLogitsProcessor(LogitsProcessor):
    def __init__(self, min_prob: float):
        if not 0.0 < min_prob < 1.0:
            raise ValueError("min_prob must be between 0 and 1.")
        self.min_prob = min_prob

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)
        # Create a mask for tokens with probabilities below the threshold
        mask = probs < self.min_prob
        # Set logits of masked tokens to -inf
        scores = scores.masked_fill(mask, float('-inf'))
        return scores
    
DEFAULT_STATS = {
    'mean': lambda x: x.mean().item(),
    'max': lambda x: x.max().item(),
    'min': lambda x: x.min().item(),
    'p25': lambda x: x.quantile(0.25).item(),
    'p75': lambda x: x.quantile(0.75).item(),
    'median': lambda x: x.median().item(),
    'var': lambda x: x.var().item(),
}
    
class GRPOEntropyTrainer(ModifiableGRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics["num_completions_train"] = []
        self._metrics["num_completions_eval"] = []
        self._metrics["generated_tokens_train"] = []
        self._metrics["generated_tokens_eval"] = []
        # self.logits_processor = LogitsProcessorList([MinProbabilityLogitsProcessor(0.01)])
        self.entropy_alpha = kwargs['args'].entropy_alpha
        self.smooth_logprobs = kwargs['args'].smooth_logprobs
        self.softplus_alpha = kwargs['args'].softplus_alpha
        self.entropy_estimator = kwargs['args'].entropy_estimator

    def _get_and_smooth_token_logps(self, model, input_ids, attention_mask, logits_to_keep, mode, return_hidden_states=False, return_entropy=False):
        per_token_logps, entropy, embeddings, probs, action_one_hot = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, return_hidden_states=return_hidden_states, return_entropy=return_entropy)

        if self.smooth_logprobs:
            self._metrics[mode]["non_smoothed_logprobs/min"].append(self.accelerator.gather_for_metrics(per_token_logps.min()).mean().item())

            # Smooth the logprobs
            per_token_logps = self._smooth_logprobs(per_token_logps)

            self._metrics[mode]["smoothed_logprobs/min"].append(self.accelerator.gather_for_metrics(per_token_logps.min()).mean().item())
        
        return per_token_logps, entropy, embeddings, probs, action_one_hot

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, return_hidden_states=False, return_entropy=False):
        """Get per-token log probabilities and embeddings from the model.
        
        Args:
            model: The model to get log probabilities and embeddings from
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep from the end
            
        Returns:
            tuple: (log_probs, embeddings) where:
                - log_probs: Per-token log probabilities (B, L)
                - embeddings: Last hidden state embeddings (B, L, H)
        """
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1, output_hidden_states=True)
        logits = outputs.logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        # compute logprobs
        logits = logits[:, -logits_to_keep:]
        log_probs = selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

        # compute softmax entropy
        entropy, probs, action_one_hot = None, None, None
        if return_entropy:
            entropy, probs, top_k_indices = self._compute_entropy(logits, k = self.generation_config.top_k)

            # Expand input_ids to match the shape of top_k_indices
            input_ids_expanded = input_ids.unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            # Compare input_ids with top_k_indices
            action_one_hot = (input_ids_expanded == top_k_indices).float()  # (batch_size, seq_len, num_tokens)


        embeddings = None
        if return_hidden_states:
            # hidden states are (num_layers, batch, sequence, hidden)
            embeddings = torch.stack([x for x in outputs.hidden_states])  # (num_layers, B, L, H)
        
        return log_probs, entropy, embeddings, probs, action_one_hot

    # override the _generate_and_score_completions method to pass logprobs to the entropy reward function
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        # Compute prompt IDs and mask
        prompt_ids, prompt_mask, prompts_text = self._compute_prompt_ids_and_mask(inputs)

        # Generate completions
        prompt_completion_ids, completion_ids = self._generate_completions(prompt_ids, prompt_mask, prompts_text, prompts, device)

        # Compute completion mask
        completion_mask = self._compute_completion_mask(completion_ids, device)

        # Compute reference logprobs
        ref_per_token_logps, old_per_token_logps, old_last_token_embeddings = \
            self._compute_ref_per_token_logps(prompt_completion_ids, completion_ids, prompt_mask, completion_mask)

        # Decode completions
        completions, completions_text = self._decode_completions(completion_ids, inputs, prompts)

        # Compute rewards
        rewards_per_func, correct_responses = \
            self._compute_rewards(completions, inputs, prompts, completion_ids, old_per_token_logps, old_last_token_embeddings, device)
                
        # Log the metrics - mode
        mode = "eval" if self.control.should_evaluate else "train"

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Find index of accuracy reward function
        accuracy_idx = [i for i, func in enumerate(self.reward_funcs) if func.__name__ == "accuracy_reward"][0]
        accuracy_reward = rewards_per_func[:, accuracy_idx]

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        gathered_advantages = self._compute_advantages(rewards, mode)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        
        advantages = gathered_advantages[process_slice]

        table = self._log_stats(
            stats_dict={
                'rewards_per_func': rewards_per_func,
                'rewards': rewards,
                'advantages': gathered_advantages,
                'old_per_token_logps': old_per_token_logps,
                'prompts_text': prompts_text,
                'completions_text': completions_text,
                'completion_mask': completion_mask,
                'correct_responses': correct_responses,
            },
            mode=mode
        )

        final_dict = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "accuracy_reward": accuracy_reward,
        }

        if table is not None:
            # merge table and final_dict
            final_dict['table'] = table

        return final_dict

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps, entropy, hidden_states, probs, action_one_hot = self._get_and_smooth_token_logps(self.model, input_ids, attention_mask, logits_to_keep, mode, \
                                                                       return_entropy=True, return_hidden_states=True)
        
        self._compute_and_log_gradient_norm(hidden_states, per_token_logps, probs, inputs["advantages"], completion_mask, mode)
        self._compute_and_log_gradient_direction(hidden_states, action_one_hot, probs, inputs["advantages"], completion_mask, mode)
        self._compute_and_log_softmax_probs_stats(probs, completion_mask, mode)

        gathered_entropy = self._gather_masked_tensor_across_processes(entropy, completion_mask)
        self._accumulate_stats(
            data=gathered_entropy,
            metric_name='policy_entropy',
            mode=mode,
            stats=DEFAULT_STATS,
        )

        
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = self._smooth_logprobs(inputs["ref_per_token_logps"]) if self.smooth_logprobs else inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        if self.num_iterations > 1 and self.smooth_logprobs:
            old_per_token_logps = self._smooth_logprobs(inputs["old_per_token_logps"])
        elif self.num_iterations > 1:
            old_per_token_logps = inputs["old_per_token_logps"]
        else:
            old_per_token_logps = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.entropy_alpha != 0.0:
            if self.entropy_estimator == "logprobs":
                per_token_loss = per_token_loss + self.entropy_alpha * per_token_logps # H = E[-log(p)], so we sum the logprobs in the loss
            elif self.entropy_estimator == "softmax":
                per_token_loss = per_token_loss - self.entropy_alpha * entropy # Here is the entropy directly, so we subtract it from the loss
            else:
                raise ValueError(f"Invalid entropy estimator: {self.entropy_estimator}")

        loss = self._compute_final_loss(per_token_loss, completion_mask)

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if self.entropy_alpha != 0.0:
            if self.entropy_estimator == "logprobs":
                entropy_loss = ((self.entropy_alpha * per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).float()
            elif self.entropy_estimator == "softmax":
                entropy_loss = -((self.entropy_alpha * entropy * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).float()

            self._accumulate_stats(
                data=entropy_loss,
                metric_name='entropy_loss',
                mode=mode,
            )

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
    

    def _compute_and_log_softmax_probs_stats(self, softmax_probs, completion_mask, mode):
        # Compute sharpness
        with torch.inference_mode():
            sharpness = self.estimate_sharpness_from_probs(softmax_probs)
            gathered_sharpness = self._gather_masked_tensor_across_processes(sharpness, completion_mask)
            self._accumulate_stats(
                data=gathered_sharpness,
                metric_name='policy_sharpness',
                mode=mode,
            )

            # Compute average softmax probability
            gathered_softmax_probs = self._gather_masked_tensor3d_across_processes(softmax_probs, completion_mask)
            mean_softmax_probs = torch.mean(gathered_softmax_probs, dim=0)
            if self.accelerator.is_main_process:
                    # sample 10000 random actions following the distribution of the mean softmax probabilities
                    random_actions = torch.multinomial(mean_softmax_probs, 10000, replacement=True)
                    # Convert to numpy and compute histogram with fixed bins from 0 to 49
                    actions_np = random_actions.detach().cpu().numpy()
                    hist_values, hist_bins = np.histogram(actions_np, bins=50, range=(0,50), density=True)
                    wandb.log({
                        "actions/density": wandb.Histogram(np_histogram=(hist_values, hist_bins))
                    })

    def estimate_sharpness_from_probs(self, probs: torch.Tensor, threshold: float = 1e-4, default_sharpness: float = 10.0) -> torch.Tensor:
        """
        Estimate sharpness (1/tau) using log-linear regression on log(pi) vs. index,
        ignoring actions where pi < threshold. If support < 2, return default sharpness.

        Args:
            probs (torch.Tensor): Softmax probabilities, shape (batch, seq_len, num_actions).
            threshold (float): Probability cutoff for including an action in the fit.
            default_sharpness (float): Fallback value when support is too small.

        Returns:
            torch.Tensor: Estimated sharpness (1/tau), shape (batch, seq_len)
        """
        B, T, K = probs.shape
        device = probs.device
        dtype = probs.dtype

        indices = torch.arange(K, device=device, dtype=dtype).view(1, 1, -1)  # (1, 1, K)
        log_probs = torch.log(probs + 1e-12)  # (B, T, K)

        mask = (probs > threshold).float()  # (B, T, K)
        support = mask.sum(dim=-1)  # (B, T)
        valid = support >= 2  # (B, T)

        mask_sum = support.clamp(min=1.0).unsqueeze(-1)  # (B, T, 1)
        x_mean = (indices * mask).sum(dim=-1, keepdim=True) / mask_sum
        y_mean = (log_probs * mask).sum(dim=-1, keepdim=True) / mask_sum

        x_centered = indices - x_mean
        y_centered = log_probs - y_mean

        cov_xy = (mask * x_centered * y_centered).sum(dim=-1) / mask_sum.squeeze(-1)
        var_x = (mask * x_centered**2).sum(dim=-1) / mask_sum.squeeze(-1)

        sharpness = -cov_xy / (var_x + 1e-12)  # Avoid divide-by-zero

        # Where not valid (support < 2), assign default sharpness
        sharpness = torch.where(valid, sharpness, torch.full_like(sharpness, default_sharpness))
        return sharpness

    def _compute_and_log_gradient_direction(self, hidden_states, one_hot_actions, softmax_probs, advantages, completion_mask, mode):
        """
        Returns:
            grad_directions: Tensor of shape (B, T, D) - normalized gradient direction vectors
        """

        with torch.inference_mode():
            T = completion_mask.shape[1]

            # Compute (e_a - pi): shape (B, T, K)
            delta = one_hot_actions - softmax_probs  # (B, T, K)

            # Multiply by advantages: shape (B, T, K)
            weighted_delta = advantages.unsqueeze(-1).unsqueeze(-1) * delta  # (B, T, K)

            # Features
            features = hidden_states[-1, :, -T:, :]

            # Compute gradient: weighted sum of features per token
            # (B, T, K) @ (B, T, D) → (B, T, D)
            gradients = torch.einsum('btk,btd->btd', weighted_delta, features)

            # ---- 1. Full variance of gradients across samples (per token) ----
            gathered_gradients = self._gather_masked_tensor3d_across_processes(gradients, completion_mask)
            grad_mean = gathered_gradients.mean(dim=0)
            grad_var = torch.norm(gathered_gradients - grad_mean, dim=-1) ** 2  # (B, T)
            self._accumulate_stats(
                data=grad_var,
                metric_name='per_token_full_gradient_variance',
                mode=mode,
                stats={ 'variance': lambda x: x.mean().item(),
                    'max_squared_error': lambda x: x.max().item() },
            )
            
            # ---- 2. Variance of policy error term delta = (e - pi) across samples ----
            gathered_delta = self._gather_masked_tensor3d_across_processes(delta, completion_mask)
            delta_mean = gathered_delta.mean(dim=0)
            delta_var = torch.norm(gathered_delta - delta_mean, dim=-1) ** 2  # (B, T)
            self._accumulate_stats(
                data=delta_var,
                metric_name='policy_error_vector_variance',
                mode=mode,
                stats={ 'metric': lambda x: x.mean().item(),
                    'max_squared_error': lambda x: x.max().item() },
            )

            # ---- 3. Variance of hidden states across samples ----
            gathered_features = self._gather_masked_tensor3d_across_processes(features, completion_mask)
            h_mean = gathered_features.mean(dim=0)
            h_var = torch.norm(gathered_features - h_mean, dim=-1) ** 2  # (B, T)
            self._accumulate_stats(
                data=h_var,
                metric_name='feature_vector_variance',
                mode=mode,
                stats={ 'metric': lambda x: x.mean().item(),
                    'max_squared_error': lambda x: x.max().item() },
            )

            # ---- 4. Sentence-level gradient variance across samples ----
            grad_mean_per_sentence = gradients.mean(dim=1)  # (B, D)
            gathered_sentence_gradients = self.accelerator.gather(grad_mean_per_sentence)
            global_mean = gathered_sentence_gradients.mean(dim=0)  # (1, D)
            grad_var_per_sentence = torch.norm(gathered_sentence_gradients - global_mean, dim=-1) ** 2  # (B,)
            self._accumulate_stats(
                data=grad_var_per_sentence,
                metric_name='sentence_full_gradient_variance',
                mode=mode,
                stats={ 'metric': lambda x: x.mean().item(),
                    'max_squared_error': lambda x: x.max().item(),
                    'p75': lambda x: x.quantile(0.75).item(),
                    'p90': lambda x: x.quantile(0.9).item(),
                    'p95': lambda x: x.quantile(0.95).item(),
                    'p99': lambda x: x.quantile(0.99).item(), },
            )

            # Compute per sentence average gradient norm
            # Compute averages over groups of N elements
            N = self.num_generations
            num_groups = len(gathered_sentence_gradients) // N
            grad_norm_per_state = gathered_sentence_gradients.reshape(num_groups, N, -1) # (B, N, D)
            mu_s = grad_norm_per_state.mean(dim=1) # (B, D)
            
            # Action level variance
            sq_sigma_s = (torch.norm(grad_norm_per_state - mu_s.unsqueeze(1), dim=-1) ** 2).mean(dim=1) # (B,)

            # State level variance
            avg_mu_s = mu_s.mean(dim=0) # (D,)
            sq_mu_error = torch.norm(mu_s - avg_mu_s, dim=-1) ** 2

            self._accumulate_stats(
                data=sq_sigma_s,
                metric_name='action_level_variance_full_gradient',
                mode=mode,
                stats={ 'metric': lambda x: x.mean().item() } # action level variance is the mean of the variance
            )

            self._accumulate_stats(
                data=sq_mu_error,
                metric_name='state_level_variance_full_gradient',
                mode=mode,
                stats={'metric': lambda x: x.mean().item() }
            )

    def _compute_and_log_gradient_norm(self, hidden_states, logprobs, softmax_probs, advantages, completion_mask, mode):
        # Compute feature norm
        with torch.inference_mode():
            seq_len = logprobs.shape[1]
            feature_norm = torch.norm(hidden_states[-1, :, -seq_len:, :], dim=-1)
            gathered_feature_norm = self._gather_masked_tensor_across_processes(feature_norm, completion_mask)
            self._accumulate_stats(
                data=gathered_feature_norm,
                metric_name='per_token_feature_norm',
                mode=mode,
                stats=DEFAULT_STATS,
            )

            policy_norm_sq = (softmax_probs**2).sum(dim=-1)
            probs = torch.exp(logprobs)
            policy_error_norm = torch.abs(1 - 2 * probs + policy_norm_sq)

            gathered_policy_error_norm = self._gather_masked_tensor_across_processes(policy_error_norm, completion_mask)
            self._accumulate_stats(
                data=gathered_policy_error_norm,
                metric_name='per_token_policy_error_norm',
                mode=mode
            )

            grad_norm = torch.abs(advantages.unsqueeze(1)) * torch.sqrt(policy_error_norm) * feature_norm


            mean_grad_norm = (grad_norm * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            gathered_mean_grad_norm = self.accelerator.gather(mean_grad_norm)
            self._accumulate_stats(
                data=gathered_mean_grad_norm,
                metric_name='per_sentence_gradient_norm',
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p85': lambda x: x.quantile(0.85).item(),
                    'p90': lambda x: x.quantile(0.9).item(),
                    'p95': lambda x: x.quantile(0.95).item(),
                    'p99': lambda x: x.quantile(0.99).item(),
                },
            )

            # Compute per sentence average gradient norm
            # Compute averages over groups of N elements
            N = self.num_generations
            num_groups = len(gathered_mean_grad_norm) // N
            grad_norm_per_state = gathered_mean_grad_norm.reshape(num_groups, N)
            mu_s = grad_norm_per_state.mean(dim=1)
            sq_sigma_s = grad_norm_per_state.var(dim=1)

            self._accumulate_stats(
                data=sq_sigma_s,
                metric_name='action_level_variance',
                mode=mode,
                stats={ 'metric': lambda x: x.mean().item() } # action level variance is the mean of the variance
            )

            self._accumulate_stats(
                data=mu_s,
                metric_name='state_level_variance',
                mode=mode,
                stats={'metric': lambda x: x.var().item() } # state level variance is the variance of the mean
            )


            gathered_all_grad_norm = self._gather_masked_tensor_across_processes(grad_norm, completion_mask)
            self._accumulate_stats(
                data=gathered_all_grad_norm,
                metric_name='per_token_gradient_norm',
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p1': lambda x: x.quantile(0.01).item(),
                    'p5': lambda x: x.quantile(0.05).item(),
                    'p10': lambda x: x.quantile(0.1).item(),
                }
            )
    
    def _smooth_logprobs(self, logprobs):
        return torch.nn.functional.softplus(logprobs +  self.softplus_alpha) - self.softplus_alpha
    
    def _compute_prompt_ids_and_mask(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the prompt IDs and attention mask from the inputs.

        Args:
            inputs (dict): The input dictionary containing the prompt and completion IDs.
        """
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)  # Call Trainer's _prepare_inputs directly
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        return prompt_ids, prompt_mask, prompts_text
    
    def _generate_completions(self, prompt_ids, prompt_mask, prompts_text, prompts, device):
        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        return prompt_completion_ids, completion_ids
    
    def _compute_completion_mask(self, completion_ids, device):
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        return completion_mask

    def _compute_ref_per_token_logps(self, prompt_completion_ids, completion_ids, prompt_mask, completion_mask):
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # TODO extract hidden states here too for other reward functions
        with torch.inference_mode():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            # if self.num_iterations > 1:
            old_per_token_logps, _, old_last_token_embeddings, _, _ = self._get_per_token_logps(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep, return_hidden_states=True
            )
            # else:
                # old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps, _, _, _, _ = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps, _, _, _, _ = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        return ref_per_token_logps, old_per_token_logps, old_last_token_embeddings
    
    def _decode_completions(self, completion_ids, inputs, prompts):
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        return completions, completions_text
    
    def _compute_rewards(self, completions, inputs, prompts, completion_ids, old_per_token_logps, old_last_token_embeddings, device):
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = Trainer._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            elif "entropy" in reward_func.__name__:
                # Pass logprobs to the entropy reward function
                # TODO hidden states might be too large to pass to the reward function after gathering
                output_reward_func = reward_func(
                    prompts=prompts, 
                    completions=completions, 
                    logprobs=old_per_token_logps, 
                    hidden_states=old_last_token_embeddings, 
                    completion_ids=completion_ids,
                    num_generations=self.num_generations
                )
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                # Check which responses are correct based on accuracy reward
                if reward_func.__name__ == "accuracy_reward":
                    correct_responses = torch.tensor(output_reward_func, dtype=torch.bool, device=device)

        return rewards_per_func, correct_responses
    
    def _compute_advantages(self, rewards, mode):
        if mode == "eval" or self.num_generations == 1:
            logger.warning("Evaluation or num_generations = 1, returning rewards as advantages")
            return rewards
        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) 

        self._accumulate_stats(
            data=std_grouped_rewards,
            metric_name='grouped_std_rewards',
            mode=mode,
            stats={'mean': lambda x: x.mean().item()},
        )

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)    
        return advantages
    
    def _compute_final_loss(self, per_token_loss, completion_mask):
        z = completion_mask.sum(dim=1)
        per_sentence_loss = per_token_loss * completion_mask
        loss_1 = per_sentence_loss.sum(dim=1) / z
        loss_avg_1 = loss_1.mean()

        gather_loss_1 = self.accelerator.gather_for_metrics(loss_1)
        mode = "eval" if self.control.should_evaluate else "train"

        self._accumulate_stats(
            data=gather_loss_1,
            metric_name='policy_loss',
            mode=mode,
        )

        # Old, buggy loss
        loss_2 = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        return loss_avg_1

    
    def _log_stats(self, stats_dict, mode):
        rewards_per_func = stats_dict['rewards_per_func']
        rewards = stats_dict['rewards']
        advantages = stats_dict['advantages']
        old_per_token_logps = stats_dict['old_per_token_logps']
        prompts_text = stats_dict['prompts_text']
        completions_text = stats_dict['completions_text']
        completion_mask = stats_dict['completion_mask']
        correct_responses = stats_dict['correct_responses']

        completion_length = self.accelerator.gather(completion_mask.sum(1)).float()
        correct_responses = self.accelerator.gather(correct_responses)

        num_generation_tokens = completion_length.sum()
        self._metrics[f"generated_tokens_{mode}"].append(num_generation_tokens.item())
        self._metrics[f"num_completions_{mode}"].append(completion_length.shape[0])

        self._metrics[mode]["num_completions"] = (torch.tensor(sum(self._metrics[f"num_completions_{mode}"]), device=self.accelerator.device), {'total': lambda x: x.item()})
        self._metrics[mode]["generated_tokens"] = (torch.tensor(sum(self._metrics[f"generated_tokens_{mode}"]), device=self.accelerator.device), {'total': lambda x: x.item()})

        # Log advantages
        self._accumulate_stats(
            data=advantages,
            metric_name='advantages',
            mode=mode,
        )

        # Log stats for each reward function
        for i, reward_func in enumerate(self.reward_funcs):
            self._accumulate_stats(
                data=rewards_per_func[:, i],
                metric_name=reward_func.__name__,
                mode=mode,
                groups={'correct': correct_responses, 'incorrect': ~correct_responses},
            )

        self._accumulate_stats(
            data=completion_length,
            metric_name='completion_length',
            mode=mode,
            groups={'correct': correct_responses, 'incorrect': ~correct_responses},
        )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._accumulate_stats(
                data=rewards_per_func[:, i],
                metric_name=f"rewards/{reward_func_name}",
                mode=mode,
            )

        self._accumulate_stats(
            data=rewards,
            metric_name="reward",
            mode=mode,
        )

        # save logprobs to wandb
        if old_per_token_logps is not None:
            # Compute per-sequence mean log probs
            mean_log_probs = (old_per_token_logps * completion_mask).sum(1) / completion_mask.sum(1)
            gather_mean_log_probs = self.accelerator.gather(mean_log_probs)

            # Compute per-token log probs, masked by completion_mask (variable length)
            gather_all_log_probs = self._gather_masked_tensor_across_processes(old_per_token_logps, completion_mask)

            # Accumulate stats
            self._accumulate_stats(
                data=gather_mean_log_probs,
                metric_name='mean_logprobs',
                mode=mode,
                stats={
                    'mean': lambda x: x.mean().item(),
                    'var': lambda x: x.var().item()
                },
            )

            self._accumulate_stats(
                data=gather_all_log_probs,
                metric_name='all_logprobs',
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p1': lambda x: x.quantile(0.01).item(),
                    'p5': lambda x: x.quantile(0.05).item(),
                    'p10': lambda x: x.quantile(0.1).item(),
                }
            )

            # Plot histogram of logprobs in wandb
            if self.accelerator.is_main_process:
                wandb.log({
                    "logprobs/histogram": wandb.Histogram(
                        gather_all_log_probs.cpu().numpy()
                    )
                })

        prompts_to_log = gather_object(prompts_text)
        completions_to_log = gather_object(completions_text)
        rewards_to_log = rewards.tolist()

        # For logging
        table = {
            "step": [str(self.state.global_step)] * len(rewards),
            "prompt": prompts_to_log,
            "completion": completions_to_log,
            "reward": rewards.tolist(),
            "correct": correct_responses.tolist(),
            "advantages": advantages.tolist(),
        }

        # Add individual reward function values
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            table[f"reward_{reward_func_name}"] = rewards_per_func[:, i].tolist()

        
        if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
            import pandas as pd                    
            df = pd.DataFrame(table)
            
            if self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return table
        
    def _accumulate_stats(self, data, metric_name, mode, groups=None, stats=DEFAULT_STATS):
        """Accumulate statistics for the given data.
        
        Args:
            data (torch.Tensor): Tensor containing the data to analyze
            metric_name (str): Base name for the metric (e.g. 'completion_length')
            mode (str): Either "train" or "eval"
            groups (dict, optional): Dictionary mapping group names to boolean masks for data grouping
        """
        # Save data in metrics dictionary, grouped by mode and groups
        # If metric_name is not in metrics, create it
        if metric_name not in self._metrics[mode]:
            self._metrics[mode][metric_name] = (data, stats)
        else:
            self._metrics[mode][metric_name] = (torch.cat([self._metrics[mode][metric_name][0], data]), stats)

        if groups is not None:
            for group_name, mask in groups.items():
                if f"{metric_name}/{group_name}" not in self._metrics[mode]:
                    self._metrics[mode][f"{metric_name}/{group_name}"] = (data[mask], stats)
                else:
                    self._metrics[mode][f"{metric_name}/{group_name}"] = (torch.cat([self._metrics[mode][f"{metric_name}/{group_name}"][0], data[mask]]), stats)
        
    def _compute_and_log_stats(self, data, metric_name, mode, stats):
        """Compute and log statistics for the given data.
        
        Args:
            data (torch.Tensor): Tensor containing the data to analyze
            metric_name (str): Base name for the metric (e.g. 'completion_length')
            mode (str): Either "train" or "eval"
            groups (dict, optional): Dictionary mapping group names to boolean masks for data grouping
            stats (dict, optional): Dictionary mapping stat names to computation functions.
                                  Defaults to mean, max, min, p25, p75, median
        """        
        # Skip empty data
        if data.numel() == 0:
            return {}
            
        # Compute and log overall statistics
        metrics = {}
        for stat_name, stat_func in stats.items():
            metric_key = f"{metric_name}/{stat_name}" if stat_name != 'mean' else f"{metric_name}"
            metrics[metric_key] = stat_func(data)

        return metrics
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        metrics = {}
        observed_num_examples = 0

        # Initialize lists to collect data across all processes
        all_tables = []
        all_accuracy_rewards = []

        with torch.no_grad():
            for step, inputs in tqdm(
                enumerate(dataloader),
                desc="Evaluation",
                disable=not self.accelerator.is_local_main_process,
                total=len(dataloader) if hasattr(dataloader, "__len__") else None
            ):
                # Update observed examples count
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prediction step
                step_metrics = self._prepare_inputs(inputs)
                
                # Collect tables and accuracy rewards
                if 'table' in step_metrics:
                    all_tables.append(step_metrics['table'])
                all_accuracy_rewards.append(step_metrics['accuracy_reward'])

            # Gather data from all processes
            gathered_tables = gather_object(all_tables)
            gathered_accuracy_rewards = torch.cat(all_accuracy_rewards)
            
            # Handle single GPU case where gather returns a single tensor instead of a list
            if not isinstance(gathered_tables, list):
                gathered_tables = [gathered_tables]

            # Process gathered data on main process
            if self.accelerator.is_main_process:
                # Flatten the gathered tables
                full_table = {}
                for tables in gathered_tables:
                    # Handle case where tables might be a single dict instead of a list
                    tables_list = tables if isinstance(tables, list) else [tables]
                    for table in tables_list:
                        for key, values in table.items():
                            if key in full_table:
                                full_table[key].extend(values)
                            else:
                                full_table[key] = list(values)

                # Log to wandb
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    df = pd.DataFrame(full_table)
                    wandb.log({"eval_set_completions": wandb.Table(dataframe=df)})

            # Log the length of accuracy rewards per GPU and total
            logger.info(f"TotalLength of accuracy rewards: {gathered_accuracy_rewards.shape}")

            # Compute metrics using gathered rewards
            metrics['eval_accuracy_reward'] = gathered_accuracy_rewards.mean().item()
            # if metrics['eval_accuracy_reward'] < 0.4:
            #     raise ValueError("Accuracy reward is too low, aborting training")

        torch.cuda.empty_cache()
        return EvalLoopOutput(
            metrics=metrics,
            predictions=None,
            label_ids=None,
            num_samples=observed_num_examples
        )
    

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self._metrics.get('eval') and self._metrics['eval'] else "train"

        metrics = {}
        for key, val in self._metrics[mode].items():
            if isinstance(val, tuple) and len(val) == 2:
                metrics.update(self._compute_and_log_stats(val[0], key, mode, val[1]))
            else:
                metrics.update({key: sum(val) / len(val)})  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        # Move any adam_stats metrics to the main metrics dict
        for key, value in self._metrics.items():
            if key.startswith('adam_stats/'):
                metrics[key] = value

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            Trainer.log(self, logs, start_time)
        else:  # transformers<=4.46
            Trainer.log(self, logs)
        self._metrics[mode].clear()

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=1, # No repeats for evaluation
            seed=self.args.seed,
        )
    
    def _compute_entropy(self, logits, chunk_size=128, k=200):
        # Compute the entropy of the top k logits
        top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)
        log_probs = F.log_softmax(top_k_logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy, probs, top_k_indices

    def _gather_masked_tensor_across_processes(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Gather a tensor across all processes, masking by a boolean/int mask (same shape as tensor),
        and return a 1D tensor of all valid (masked) values across all processes.

        Args:
            tensor (torch.Tensor): The tensor to gather and mask (e.g., per-token logprobs).
            mask (torch.Tensor): The mask tensor (same shape as tensor), 1 for valid, 0 for invalid.

        Returns:
            torch.Tensor: 1D tensor of all valid values across all processes.
        """
        # Flatten and mask locally
        flat_tensor = tensor.flatten()[mask.flatten() == 1].float()

        # Gather the number of valid tokens from each process
        local_length = torch.tensor([flat_tensor.shape[0]], device=flat_tensor.device)
        gathered_lengths = self.accelerator.gather(local_length).flatten()  # Shape: [num_processes]

        # Pad local tensor to max length across processes
        max_length = gathered_lengths.max().item()
        padded_flat_tensor = torch.nn.functional.pad(
            flat_tensor,
            (0, max_length - flat_tensor.shape[0]),
            value=0.0  # Pad with zeros (or NaN if you prefer)
        )

        # Gather padded tensors (same shape across processes)
        gathered_padded_flat_tensor = self.accelerator.gather(padded_flat_tensor)

        # Reshape to [num_processes, max_length]
        gathered_padded_flat_tensor = gathered_padded_flat_tensor.reshape(-1, max_length)

        # Create mask to remove padding
        positions = torch.arange(max_length, device=gathered_lengths.device).unsqueeze(0)  # [1, max_length]
        mask = positions < gathered_lengths.unsqueeze(1)  # [num_processes, max_length], bool

        # Flatten mask and gathered tensor
        mask = mask.flatten()
        gathered_padded_flat_tensor = gathered_padded_flat_tensor.flatten()

        # Apply mask to get all valid values
        gathered_valid_values = gathered_padded_flat_tensor[mask]
        return gathered_valid_values


    def _gather_masked_tensor3d_across_processes(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Gather a 3D tensor across all processes, masking by a boolean/int mask (same shape as tensor without K),
        and return a 2D tensor of all valid (masked) values across all processes.

        Args:
            tensor (torch.Tensor): The tensor to gather and mask, shape (B, T, K).
            mask (torch.Tensor): The mask tensor, shape (B, T), 1 for valid, 0 for invalid.

        Returns:
            torch.Tensor: 2D tensor of all valid values across all processes, shape (num_valid_tokens, K)
        """
        B, T, K = tensor.shape
        tensor_2d = tensor.reshape(-1, K)  # (B*T, K)
        mask_1d = mask.reshape(-1)         # (B*T,)
        flat_tensor = tensor_2d[mask_1d == 1].float()  # (num_valid, K)

        # Gather the number of valid tokens from each process
        local_length = torch.tensor([flat_tensor.shape[0]], device=flat_tensor.device)
        gathered_lengths = self.accelerator.gather(local_length).flatten()  # Shape: [num_processes]

        # Pad local tensor to max length across processes
        max_length = gathered_lengths.max().item()
        pad_rows = max_length - flat_tensor.shape[0]
        if pad_rows > 0:
            padded_flat_tensor = torch.cat(
                [flat_tensor, torch.zeros((pad_rows, K), dtype=flat_tensor.dtype, device=flat_tensor.device)],
                dim=0
            )
        else:
            padded_flat_tensor = flat_tensor

        # Gather padded tensors (same shape across processes)
        gathered_padded_flat_tensor = self.accelerator.gather(padded_flat_tensor)

        # Reshape to [num_processes, max_length, K]
        gathered_padded_flat_tensor = gathered_padded_flat_tensor.reshape(-1, max_length, K)

        # Create mask to remove padding
        positions = torch.arange(max_length, device=gathered_lengths.device).unsqueeze(0)  # [1, max_length]
        mask_valid = positions < gathered_lengths.unsqueeze(1)  # [num_processes, max_length], bool

        # Flatten mask and gathered tensor
        mask_valid = mask_valid.flatten()
        gathered_padded_flat_tensor = gathered_padded_flat_tensor.reshape(-1, K)

        # Apply mask to get all valid values
        gathered_valid_values = gathered_padded_flat_tensor[mask_valid]
        return gathered_valid_values
