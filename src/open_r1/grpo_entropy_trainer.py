from collections import defaultdict
import time
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import wandb
import io
import tempfile
import os
from tqdm import tqdm
from packaging import version
from torch.utils.data import DataLoader, Sampler
from open_r1 import sparse_grad_ops
from open_r1.sparse_optim_models import SparseAdamModel, SparseAdamWModel, SparseSGDModel


import transformers
import torch.nn.functional as F
from transformers import Trainer
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import EvalLoopOutput, has_length, denumpify_detensorize
from transformers.utils import logging
from transformers import LogitsProcessorList, LogitsProcessor
from accelerate.utils import broadcast_object_list, gather, gather_object
from trl.extras.profiling import profiling_decorator

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
        self.advantage_target_std = kwargs['args'].advantage_target_std

        self.hessian_token_lambda = kwargs['args'].hessian_token_lambda
        self.fisher_token_lambda = kwargs['args'].fisher_token_lambda
        self.hessian_sentence_lambda = kwargs['args'].hessian_sentence_lambda
        self.fisher_sentence_lambda = kwargs['args'].fisher_sentence_lambda
        self.hessian_global_lambda = kwargs['args'].hessian_global_lambda
        self.fisher_global_lambda = kwargs['args'].fisher_global_lambda

        self.hessian_token_mask_tau = kwargs['args'].hessian_token_mask_tau
        self.fisher_token_mask_tau = kwargs['args'].fisher_token_mask_tau
        self.hessian_sentence_mask_tau = kwargs['args'].hessian_sentence_mask_tau
        self.fisher_sentence_mask_tau = kwargs['args'].fisher_sentence_mask_tau
        self.hessian_global_mask_tau = kwargs['args'].hessian_global_mask_tau
        self.fisher_global_mask_tau = kwargs['args'].fisher_global_mask_tau
        self.curvature_masking = any([self.hessian_token_mask_tau, self.fisher_token_mask_tau, self.hessian_sentence_mask_tau, \
                                      self.fisher_sentence_mask_tau, self.hessian_global_mask_tau, self.fisher_global_mask_tau])
        
        self.sequential_masking = kwargs['args'].sequential_masking
        self.hessian_symmetric_clipping = kwargs['args'].hessian_symmetric_clipping

        if kwargs['args'].optim_model_type == "adam":
            self.sparse_optim_model = SparseAdamModel(adam_beta1=kwargs['args'].adam_beta1, adam_beta2=kwargs['args'].adam_beta2, adam_epsilon=kwargs['args'].adam_epsilon)
        elif kwargs['args'].optim_model_type == "adamw":
            self.sparse_optim_model = SparseAdamWModel(adam_beta1=kwargs['args'].adam_beta1, adam_beta2=kwargs['args'].adam_beta2, adam_epsilon=kwargs['args'].adam_epsilon, weight_decay=kwargs['args'].weight_decay)
        else:
            self.sparse_optim_model = SparseSGDModel()

        self.capo_only = kwargs['args'].capo_only
        if self.capo_only and any([self.fisher_sentence_mask_tau,  self.hessian_sentence_mask_tau, self.fisher_global_mask_tau, self.hessian_global_mask_tau]):
            raise ValueError("The 'capo_only' flag is designed for evaluating the token-level CAPO, not the sentence-level or global-level CAPO.")


    def _get_and_smooth_token_logps(self, model, input_ids, attention_mask, logits_to_keep, mode, return_hidden_states=False, return_entropy=False):
        per_token_logps, entropy, embeddings, probs, action_one_hot, top_k_token_ids = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, return_hidden_states=return_hidden_states, return_entropy=return_entropy)

        if self.smooth_logprobs:
            self._metrics[mode]["non_smoothed_logprobs/min"].append(self.accelerator.gather_for_metrics(per_token_logps.min()).mean().item())

            # Smooth the logprobs
            per_token_logps = self._smooth_logprobs(per_token_logps)

            self._metrics[mode]["smoothed_logprobs/min"].append(self.accelerator.gather_for_metrics(per_token_logps.min()).mean().item())
        
        return per_token_logps, entropy, embeddings, probs, action_one_hot, top_k_token_ids

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
        entropy, probs, action_one_hot, top_k_indices = None, None, None, None
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
        
        return log_probs, entropy, embeddings, probs, action_one_hot, top_k_indices

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

    @profiling_decorator
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

        per_token_logps, entropy, embeddings, probs, action_one_hot, top_k_token_ids = self._get_and_smooth_token_logps(self.model, input_ids, attention_mask, logits_to_keep, mode, \
                                                                       return_entropy=True, return_hidden_states=True)
        seq_len = top_k_token_ids.size(1)
        hidden_states = embeddings[-1, :, -seq_len:, :]

        self._compute_and_log_softmax_probs_stats(probs, completion_mask, mode)
        self._log_feature_stats(per_token_logps, hidden_states, probs, completion_mask, mode)
        del embeddings
        torch.cuda.empty_cache()

        ##### Compute and log gradients and curvatures #####
        full_update_term, approx_kl = self._compute_and_log_gradients_linear_model(hidden_states, per_token_logps, probs, action_one_hot, inputs, completion_mask, top_k_token_ids, mode, update_optim_model=False)

        ###### Re-evaluate gradients and curvatures after curvature masking ######
        if self.curvature_masking:
            if self.sequential_masking:
                curvature_mask, masks_hessian, masks_fisher = self._compute_curvature_sequential_mask(full_update_term, approx_kl, completion_mask, hidden_states, per_token_logps, probs, action_one_hot, inputs, top_k_token_ids, mode, update_optim_model=True)
                updated_mask = completion_mask * curvature_mask
            else:
                curvature_mask_hessian, masks_hessian = self._compute_hessian_curvature_mask(full_update_term, completion_mask)
                curvature_mask_fisher, masks_fisher = self._compute_fisher_curvature_mask(approx_kl, completion_mask)
                updated_mask = completion_mask * curvature_mask_hessian * curvature_mask_fisher

                # Evaluate the gradients and curvatures after the curvature masking
                self._compute_and_log_gradients_linear_model(hidden_states, per_token_logps, probs, action_one_hot, inputs, updated_mask, top_k_token_ids, mode, prefix="masked", update_optim_model=True)
        
        torch.cuda.empty_cache()
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

        curvature_estimator = {}
        masks = {}
        if full_update_term is not None:
            curvature_estimator["hessian"] = full_update_term
        if approx_kl is not None:
            curvature_estimator["fisher"] = approx_kl
        if self.curvature_masking:
            masks["hessian"] = masks_hessian
            masks["fisher"] = masks_fisher
        else:
            updated_mask = completion_mask
        
        loss = self._compute_final_loss(per_token_loss, completion_mask, updated_mask, masks, curvature_estimator, completion_ids)

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
    

    @profiling_decorator
    @torch.no_grad()
    def _compute_and_log_gradients_linear_model(self, hidden_states, per_token_logps, probs, action_one_hot, inputs, completion_mask, top_k_token_ids, mode, prefix="", update_optim_model=True):
        
        ######################## Compute gradients under the linear logit model ########################
        full_update_term = {}
        approx_kl = {}

        ###### Token Level Gradients and Curvatures ######
        per_token_gradient = self._compute_token_level_gradients(hidden_states, probs, action_one_hot, inputs["advantages"], completion_mask, top_k_token_ids)
        effective_per_token_gradients = self._compute_effective_token_gradients(per_token_gradient, top_k_token_ids)
        token_grad_norm_sq = self._compute_grad_norm_sq(per_token_gradient, effective_per_token_gradients, None, None, "token")
        _, full_update_term['token'] = self._compute_and_log_hessian_coefficient(hidden_states, action_one_hot, probs, inputs["advantages"], completion_mask, \
                                                        effective_per_token_gradients, token_grad_norm_sq, "token", mode, prefix)
        _, approx_kl['token'] = self._compute_and_log_fisher_curvature(hidden_states, action_one_hot, probs, inputs["advantages"], completion_mask, \
                                                        effective_per_token_gradients, "token", mode=mode, prefix=prefix)
        del effective_per_token_gradients, token_grad_norm_sq
        torch.cuda.empty_cache()

        if self.capo_only:
            if update_optim_model:
                self._capo_only_update_adam_moments(per_token_gradient, hidden_states.shape[-1], top_k_token_ids, completion_mask, hidden_states.dtype)
                torch.cuda.empty_cache()    
            return full_update_term, approx_kl

        
        ###### Per Sentence Gradients and Curvatures ######
        #### 1. Sparsify the gradients
        dicts_per_sentence = self._sparsify_sentence_gradients(per_token_gradient, hidden_states.shape[-1], top_k_token_ids, completion_mask, hidden_states.dtype)
        global_dicts = self._sparsify_global_gradients(per_token_gradient, hidden_states.shape[-1], top_k_token_ids, completion_mask, hidden_states.dtype)
        
        #### 2. Log per token gradients to free up memory
        self._log_gradients_stats(per_token_gradient, dicts_per_sentence, completion_mask, mode, prefix)
        del per_token_gradient
        torch.cuda.empty_cache()

        #### 3. Compute per sentence gradients
        effective_g_sent, sentence_grad_norm_sq = self._compute_all_sentence_gradients(dicts_per_sentence, hidden_states.shape[-1], top_k_token_ids, hidden_states.dtype)
        _, full_update_term['sentence'] = self._compute_and_log_hessian_coefficient(hidden_states, action_one_hot, probs, inputs["advantages"], completion_mask, \
                                                        effective_g_sent, sentence_grad_norm_sq, "sentence", mode, prefix)
        _, approx_kl['sentence'] = self._compute_and_log_fisher_curvature(hidden_states, action_one_hot, probs, inputs["advantages"], completion_mask, \
                                                        effective_g_sent, "sentence", mode, prefix)

        del effective_g_sent, sentence_grad_norm_sq, dicts_per_sentence
        torch.cuda.empty_cache()

        ###### Global Gradients and Curvatures ######
        effective_g_global, global_grad_norm_sq, final_global_dict = self._compute_all_global_gradients(global_dicts, hidden_states.shape[-1], top_k_token_ids, hidden_states.dtype)
        _, full_update_term['global'] = self._compute_and_log_hessian_coefficient(hidden_states, action_one_hot, probs, inputs["advantages"], completion_mask, \
                                                        effective_g_global, global_grad_norm_sq, "global", mode, prefix)
        _, approx_kl['global'] = self._compute_and_log_fisher_curvature(hidden_states, action_one_hot, probs, inputs["advantages"], completion_mask, \
                                                        effective_g_global, "global", mode, prefix)

        
        if update_optim_model:
            self.sparse_optim_model.update_effective_gradient_moments(final_global_dict)

        del effective_g_global, global_grad_norm_sq, global_dicts, final_global_dict

        torch.cuda.empty_cache()

        return full_update_term, approx_kl

    @profiling_decorator
    def _compute_token_level_gradients(self, h, softmax_probs, one_hot_actions, advantages, completion_mask, token_ids):
        """
        This function computes the gradients per token.
        The idea of this function is: per token, we can simply implement the gradient as g = A * (v ⊗ h).
        """

        # === 1. Extract final layer hidden states and policy info ===
        pi = softmax_probs                                 # [B, T, V]
        e_a = one_hot_actions.to(h.dtype)                  # [B, T, V]
        v = e_a - pi                                       # [B, T, V]
        A = advantages.unsqueeze(1).to(h.dtype)            # [B, 1]

        # === 2. Compute token-level gradients: g = A * (v ⊗ h) ===
        g = A.unsqueeze(-1).unsqueeze(-1) * v.unsqueeze(-1) * h.unsqueeze(2)  # [B, T, V, H]

        return g
    
    @profiling_decorator
    def _sparsify_sentence_gradients(self, g, hidden_dim, token_ids, mask, grad_dtype):
        """
        This function sparsifies the gradients based on the token_ids and mask.
        It the dictionary of gradients per token at a sentence and global level.

        We need to average the gradients across the tokens, and each token spans a different subspace of tokens based on the top-k generation.
        We cannot create a large gradient tensor for the full vocab (~150k tokens), so we build dictionaries to exploit sparsity.
        """
        ######################################################################################
        # === 1. Compute gradients for each sentence ===
        
        # Explanation: Each per token gradient spans a subspace for their top-K tokens (K = 50). This subspace differ across tokens.
        # Thus, we need to map the same logits (tokens) to the same gradients. 
        # Since spanning a full vocab tensor is infeasible, we build a dictionary to exploit sparsity.

        # === Dimensions ===
        # B: batch_size, T: seq_len, V: top-K tokens, H: hidden_dim
        batch_size, seq_len, topk_size = token_ids.shape

        dicts_per_sentence = []  # List of dicts: one per sentence

        # For each sentence, build dict and compute mean gradients
        for b in range(batch_size):
            # Mask for valid tokens in this sentence
            sentence_mask = mask[b].unsqueeze(1).expand(seq_len, topk_size).reshape(-1)  # [T*V]

            # Flatten sentence views
            ids_flat = token_ids[b].reshape(-1)          # [T*V]
            g_flat   = g[b].reshape(-1, hidden_dim)      # [T*V, H]

            # Filter out masked positions
            token_ids_valid = ids_flat[sentence_mask > 0]  # [N_valid]
            g_valid = g_flat[sentence_mask > 0]                  # [N_valid, H]

            # Add this check
            if token_ids_valid.numel() == 0:
                dict_b = {}
                dicts_per_sentence.append(dict_b)
                continue

            # Build dictionary for this sentence

            # Group by token id: unique ids, inverse mapping of each row -> group, and counts
            uniq_ids, inverse, counts = torch.unique(token_ids_valid, return_inverse=True, return_counts=True)

            # Sum gradients per group with index_add_
            sums = torch.zeros((uniq_ids.numel(), hidden_dim), dtype=g_valid.dtype, device=self.accelerator.device)
            sums.index_add_(0, inverse, g_valid)  # sums[g] += g_valid[i] where g = inverse[i]

            means = sums / counts.unsqueeze(1)
            # Convert to a small dict (sparse) for this sentence
            dict_b = {int(k.item()): means[i] for i, k in enumerate(uniq_ids)}
            dicts_per_sentence.append(dict_b)

        return dicts_per_sentence
    
    @profiling_decorator
    def _sparsify_global_gradients(self, g, hidden_dim, token_ids, mask, grad_dtype):
        """
        This function sparsifies the gradients based on the token_ids and mask.
        It the dictionary of gradients per token at a sentence and global level.
        """
        # === Dimensions ===
        # B: batch_size, T: seq_len, V: top-K tokens, H: hidden_dim
        batch_size, seq_len, topk_size = token_ids.shape

        # === Build global dictionary ===

        # Group by token id: unique ids, inverse mapping of each row -> group, and counts# Build a single flat validity mask for all B*T*V
        valid_all = mask.unsqueeze(2).expand(batch_size, seq_len, topk_size).reshape(-1)  # [B*T*V]

        ids_all = token_ids.reshape(-1)                 # [B*T*V]
        g_all   = g.reshape(-1, hidden_dim)             # [B*T*V, H]

        ids_valid_all = ids_all[valid_all > 0]          # [N_all]
        g_valid_all   = g_all[valid_all > 0]            # [N_all, H]

        if ids_valid_all.numel() == 0:
            return {}

        uniq_ids_g, inverse_g, counts_g = torch.unique(
            ids_valid_all, return_inverse=True, return_counts=True
        )
        sums_g = torch.zeros((uniq_ids_g.numel(), hidden_dim), dtype=g_valid_all.dtype, device=self.accelerator.device)
        sums_g.index_add_(0, inverse_g, g_valid_all)

        # You can also return sums+counts if you prefer to delay the division.
        # Here I return means for parity with per-sentence dicts.
        global_dict = {int(k.item()): (sums_g[i], int(counts_g[i].item()))
                       for i, k in enumerate(uniq_ids_g)}
        return global_dict
        
    @profiling_decorator
    def _compute_all_sentence_gradients(self, dicts_per_sentence, hidden_dim, token_ids, grad_dtype):
        """
        This function computes the gradients per sentence, including effective gradients.
        """
        effective_g_sent_dicts = self.sparse_optim_model.compute_effective_sentence_gradients(dicts_per_sentence)
        effective_g_sent = self._compute_sentence_gradients(effective_g_sent_dicts, hidden_dim, token_ids, grad_dtype)
        sentence_grad_norm_sq = self._compute_grad_norm_sq(None, effective_g_sent, dicts_per_sentence, effective_g_sent_dicts, "sentence")

        del effective_g_sent_dicts
        return effective_g_sent, sentence_grad_norm_sq
    
    @profiling_decorator
    def _compute_sentence_gradients(self, dicts_per_sentence, hidden_dim, token_ids, grad_dtype):
        """
        This function computes the gradients per sentence.
        While computing the curvatures later on, to once again exploit sparsity, we need to map the gradients back to their corresponding subspaces.
        """
        # === Dimensions ===
        # B: batch_size, T: seq_len, V: top-K tokens, H: hidden_dim
        batch_size, seq_len, topk_size = token_ids.shape

        # We now the mean gradient per logit for each sentence.
        # Now we wanna map this gradients back to their corresponding subspaces
        # We start from token_ids and map each token_id to its corresponding gradient
        # We do this for each token in the batch, returning a grad tensor of shape (B, T, V, H)
        
        # Preallocate output buffer [B, T*V, H] filled with zeros
        grad_tensor_sentence_flat = torch.zeros((batch_size, seq_len * topk_size, hidden_dim), dtype=grad_dtype, device=self.accelerator.device)

        # Flatten token_ids for easier processing
        # Shape: [B, T*V], where each row corresponds to all tokens in a sentence
        token_ids_flat = token_ids.view(batch_size, seq_len * topk_size)  # [B, T*V]

        for b in range(batch_size):
            dict_b = dicts_per_sentence[b] # dictionary {token_id: mean_gradient}
            if not dict_b:
                continue  # skip if no valid tokens

            # Convert dict keys (token IDs) to a tensor [U]
            # and stack values (mean gradients) into a tensor [U, H]
            unique_token_ids_b = torch.tensor(list(dict_b.keys()), device=self.accelerator.device, dtype=torch.long)  # [num_unique_tokens]
            mean_grads_b = torch.stack(list(dict_b.values()))  # [num_unique_tokens, H]

            # Sort token IDs so we can perform fast search with bucketize
            # sorted_ids[i] matches row i in sorted_grads
            sorted_ids, perm = torch.sort(unique_token_ids_b)   # [U]
            sorted_grads = mean_grads_b.index_select(0, perm)   # [U, H]

            # All token IDs in this sentence [T*V]
            token_ids_b_flat = token_ids_flat[b]  # [T*V]

            # === Core operation: map token_ids_b_flat -> gradients ===
            # For each token, bucketize finds the insertion index into sorted_ids
            # "right=True" ensures exact matches map to their true index + 1
            ins_idx = torch.bucketize(token_ids_b_flat, sorted_ids, right=True)  # [T*V], values in [0..U]

            # Example:
            #   sorted_ids = [7, 10, 42]
            #   token_ids_b_flat = [10, 10, 1, 7, 42]
            #   ins_idx = [2, 2, 0, 1, 3]
            
            # Mark tokens that are >= sorted_ids[0], i.e. not before the first ID
            valid_rows = ins_idx > 0

            if valid_rows.any():
                # Candidate index is one step back (since ins_idx is insertion point)
                candidate_idx = ins_idx.clamp(min=1) - 1

                # Continuing example:
                #   candidate_idx = [1, 1, -1, 0, 2]
                
                # Double-check matches: keep only tokens that actually equal sorted_ids[candidate_idx]
                same = sorted_ids.index_select(0, candidate_idx[valid_rows]) == token_ids_b_flat[valid_rows]

                # Example:
                #   sorted_ids[candidate_idx] = [10, 10, 7, 42]
                #   token_ids_b_flat[valid_rows] = [10, 10, 7, 42]
                #   same = [True, True, True, True]

                # Build final mask of valid positions where token exists in dict_b
                final_valid = torch.zeros_like(valid_rows)
                final_valid[valid_rows] = same

                # Gather gradient rows from sorted_grads for all valid tokens
                mapped_indices = candidate_idx[final_valid]  # [M]
                # Example: mapped_indices = [1, 1, 0, 2]


                gathered_grads = sorted_grads.index_select(0, mapped_indices)  # [M, H]
                # Example: sorted_grads = [g7, g10, g42]
                #          gathered_grads = [g10, g10, g7, g42]

                # Assign gathered grads into the output buffer only at valid positions
                grad_tensor_sentence_flat[b, final_valid] = gathered_grads

                del final_valid, same, mapped_indices, gathered_grads, candidate_idx
            
            del unique_token_ids_b, mean_grads_b, sorted_ids, perm, sorted_grads, token_ids_b_flat, ins_idx, valid_rows

        sentence_gradient = grad_tensor_sentence_flat.view(batch_size, seq_len, topk_size, hidden_dim)  # [B, T, V, H]
        del grad_tensor_sentence_flat
        torch.cuda.empty_cache()
         
        return sentence_gradient
        
    @profiling_decorator
    def _compute_all_global_gradients(self, global_dict, hidden_dim, token_ids, grad_dtype):
        """
        This function computes the gradients per global level, including effective gradients.
        """
        """
        This function computes the gradients per global level.
        While computing the curvatures later on, to once again exploit sparsity, we need to map the gradients back to their corresponding subspaces.
        """
        gather_fn = self.accelerator.gather
        # === Dimensions ===
        # B: batch_size, T: seq_len, V: top-K tokens, H: hidden_dim
        batch_size, seq_len, topk_size = token_ids.shape
        num_keys = self.accelerator.gather(torch.tensor(len(global_dict.keys()), device=self.accelerator.device))
        max_keys = max(num_keys)

        if max_keys == 0:
            final_global_dict = {}
            global_gradient = torch.zeros((batch_size, seq_len, topk_size, hidden_dim), dtype=grad_dtype, device=self.accelerator.device)
            # norm sq is a zero for the full batch
            global_grad_norm_sq = torch.zeros(1, dtype=grad_dtype, device=self.accelerator.device)
            return global_gradient, global_grad_norm_sq, final_global_dict

        global_dict_tensor, key_count_tensor = self._list_of_dicts_to_tensor([global_dict], max_num_tokens=max_keys, hidden_dim=hidden_dim, grad_dtype=grad_dtype)
        gathered_global_dict_tensor = gather_fn(global_dict_tensor)
        gathered_key_count_tensor = gather_fn(key_count_tensor)

        # Aggregate the global dictionary, returning the average gradient per token_id
        final_global_dict = self.aggregate_global_dict(gathered_global_dict_tensor, gathered_key_count_tensor, num_keys)
        del gathered_global_dict_tensor, gathered_key_count_tensor, global_dict_tensor, key_count_tensor
        torch.cuda.empty_cache()
        
        if self.capo_only:
            return final_global_dict

        effective_final_global_dict = self.sparse_optim_model.compute_effective_global_gradients(final_global_dict)
        effective_g_global = self._compute_global_gradients(effective_final_global_dict, hidden_dim, token_ids, grad_dtype)
        global_grad_norm_sq = self._compute_grad_norm_sq(None, effective_g_global, final_global_dict, effective_final_global_dict, "global")
        del effective_final_global_dict
        torch.cuda.empty_cache()

        return effective_g_global, global_grad_norm_sq, final_global_dict
    
    @profiling_decorator
    def _compute_global_gradients(self, final_global_dict, hidden_dim, token_ids, grad_dtype):
        batch_size, seq_len, topk_size = token_ids.shape
        if not final_global_dict:
            global_gradient = torch.zeros((batch_size, seq_len, topk_size, hidden_dim), dtype=grad_dtype, device=self.accelerator.device)
            return global_gradient

        ############ We now map the global gradient back to the dense representation
        global_gradient = sparse_grad_ops.densify_gradient_dict(final_global_dict, token_ids, grad_dtype, self.accelerator.device)
        return global_gradient

    
    # gathered_global_dict_tensor: iterable of 2D tensors shaped [N_i, D]
    # num_keys: iterable of ints (valid rows in each corresponding tensor)
    # Layout per row: [key, value_0, ..., value_{V-1}, count]
    # Types: key should be integer-like; count/value are tensors (float/long as you need).

    def aggregate_global_dict(self, gathered_global_dict_tensor, gathered_key_count_tensor, num_keys):
        # 1) Concatenate only the valid slices once
        values = torch.cat([t[:n] for t, n in zip(gathered_global_dict_tensor, num_keys)], dim=0)
        valid_key_count_tensors = torch.cat([t[:n] for t, n in zip(gathered_key_count_tensor, num_keys)], dim=0)
        
        # 2) Extract keys and counts based on tensor structure
        has_counts = valid_key_count_tensors.shape[-1] > 1
        
        # Always extract keys (first column)
        keys = valid_key_count_tensors[:, 0]
        
        # Extract counts only when available
        cnts = valid_key_count_tensors[:, 1] if has_counts else None

        # 3) Group by key and sum using vectorized index_add
        uniq_keys, inv = torch.unique(keys, sorted=False, return_inverse=True)

        # Initialize accumulators
        K = uniq_keys.shape[0]
        V = values.shape[1]
        values_sum = torch.zeros(K, V, device=values.device, dtype=values.dtype)
        
        # Accumulate values
        values_sum.index_add_(0, inv, values)
        
        # 4) Handle counts if available
        if has_counts:
            cnts_sum = torch.zeros(K, device=cnts.device, dtype=cnts.dtype)
            cnts_sum.index_add_(0, inv, cnts)
            
            # Build dict by averaging the gradients
            final_global_dict = {
                int(k.item()): values_sum[i] /  cnts_sum[i]
                for i, k in enumerate(uniq_keys) if cnts_sum[i] > 0
            }
        else:
            # Build dict without counts
            final_global_dict = {
                int(k.item()): values_sum[i]
                for i, k in enumerate(uniq_keys)
            }
                
        return final_global_dict

    @profiling_decorator
    def _capo_only_update_adam_moments(self, per_token_gradient, hidden_states_shape, top_k_token_ids, completion_mask, hidden_states_dtype):
        """
        This function performs a meta-step for the CAPO-only mode.
        It computes the global gradients and updates the effective gradient moments.
        """

        global_dicts = self._sparsify_global_gradients(per_token_gradient, hidden_states_shape, top_k_token_ids, completion_mask, hidden_states_dtype)
        final_global_dict = self._compute_all_global_gradients(global_dicts, hidden_states_shape, top_k_token_ids, hidden_states_dtype)
        self.sparse_optim_model.update_effective_gradient_moments(final_global_dict)

    @profiling_decorator
    def _compute_effective_token_gradients(self, per_token_gradient, top_k_token_ids):
        return self.sparse_optim_model.compute_effective_token_gradients(per_token_gradient, top_k_token_ids, self.accelerator.device)
        
    @profiling_decorator
    def _compute_gradients(self, h, per_token_logps, softmax_probs, one_hot_actions, advantages, completion_mask, token_ids):
        """
        This function computes the gradients per token, per sentence, and per global token.
        It also returns the dictionary of gradients per token at a sentence and global level.

        The idea of this function is: per token, we can simply implement the gradient as g = A * (v ⊗ h).
        However, for sentence/global level we need to average the gradients across the tokens, and each token spans a different subspace of tokens based on the top-k generation.
        We cannot create a large gradient tensor for the full vocab (~150k tokens), so we build dictionaries to exploit sparsity.

        Furthermore, while computing the curvatures later on, to once again exploit sparsity, we need to map the gradients back to their corresponding subspaces.
        """
        gather_fn = self.accelerator.gather
        # === Dimensions ===
        # B: batch_size, T: seq_len, V: top-K tokens, H: hidden_dim
        batch_size, seq_len, topk_size = token_ids.shape
        hidden_dim = h.shape[-1] # Hidden State is of shape [B, T, H]

        # === 1. Extract final layer hidden states and policy info ===
        pi = softmax_probs                                 # [B, T, V]
        e_a = one_hot_actions.to(h.dtype)                  # [B, T, V]
        v = e_a - pi                                       # [B, T, V]
        A = advantages.unsqueeze(1).to(h.dtype)            # [B, 1]
        mask = completion_mask.to(h.dtype)                 # [B, T]

        # === 2. Compute token-level gradients: g = A * (v ⊗ h) ===
        g = A.unsqueeze(-1).unsqueeze(-1) * v.unsqueeze(-1) * h.unsqueeze(2)  # [B, T, V, H]

        ######################################################################################
        # === 3. Compute gradients for each sentence ===
        
        # Explanation: Each per token gradient spans a subspace for their top-K tokens (K = 50). This subspace differ across tokens.
        # Thus, we need to map the same logits (tokens) to the same gradients. 
        # Since spanning a full vocab tensor is infeasible, we build a dictionary to exploit sparsity.

        dicts_per_sentence = []  # List of dicts: one per sentence
        global_dict = {}

        # For each sentence, build dict and compute mean gradients
        for b in range(batch_size):
            # Mask for valid tokens in this sentence
            sentence_mask = mask[b, :]  # [T]

            # Slice tensors for sentence
            token_ids_b = token_ids[b]      # [T, V]
            g_b = g[b]                      # [T, V, H]

            token_ids_flat = token_ids_b.view(-1)  # [T*V]
            g_flat = g_b.view(-1, hidden_dim)      # [T*V, H]

            # Filter out masked positions
            mask_b_expanded = sentence_mask.view(-1, 1).expand(-1, topk_size).reshape(-1)  # [T*V]
            valid_mask = mask_b_expanded > 0

            token_ids_valid = token_ids_flat[valid_mask]  # [N_valid]
            g_valid = g_flat[valid_mask]                  # [N_valid, H]

            # === Build dictionary for this sentence
            dict_b = {}
            unique_token_ids = token_ids_valid.unique()

            for token_id in unique_token_ids:
                token_mask = token_ids_valid == token_id
                grads_for_token = g_valid[token_mask]  # [N_occurrences, H]
                sum_grad = grads_for_token.sum(dim=0)  # [H]
                cnt = grads_for_token.size(0)
                dict_b[token_id.item()] = sum_grad / cnt

                # Add grads_for_token to global dictionary. If the token_id is already in the global dictionary, we stack the gradients
                if token_id.item() in global_dict:
                    prev_sum_grad, prev_cnt = global_dict[token_id.item()]
                    global_dict[token_id.item()] = (prev_sum_grad + sum_grad, prev_cnt + cnt)
                else:
                    global_dict[token_id.item()] = (sum_grad, cnt)

            dicts_per_sentence.append(dict_b)

        # We now know the mean gradient per logit for each sentence.
        # Now we wanna map this gradients back to their corresponding subspaces
        # We start from token_ids and map each token_id to its corresponding gradient
        # We do this for each token in the batch, returning a grad tensor of shape (B, T, V, H)
        grad_tensor_sentence_flat = torch.zeros((batch_size, seq_len * topk_size, hidden_dim), dtype=h.dtype, device=h.device)

        # Flatten token_ids for indexing
        token_ids_flat = token_ids.view(batch_size, -1)  # [B, T*V]

        for b in range(batch_size):
            dict_b = dicts_per_sentence[b]
            if not dict_b:
                continue  # skip if no valid tokens

            # Stack token IDs and mean gradients for this batch
            unique_token_ids_b = torch.tensor(list(dict_b.keys()), device=h.device, dtype=torch.long)  # [num_unique_tokens]
            mean_grads_b = torch.stack(list(dict_b.values()))  # [num_unique_tokens, H]

            # Create a mapping from token_id -> position in unique_token_ids_b
            token_id_to_idx = {tid.item(): idx for idx, tid in enumerate(unique_token_ids_b)}

            # Map token_ids_flat[b] -> mean_grad
            token_ids_b_flat = token_ids_flat[b]  # [T*V]
            # Create a mask for which tokens have a mean gradient
            mapped_indices = torch.tensor(
                [token_id_to_idx.get(tid.item(), -1) for tid in token_ids_b_flat],
                device=h.device
            )  # [T*V], -1 means no gradient

            # Prepare assignment buffer
            assigned_grads = torch.zeros((token_ids_b_flat.size(0), hidden_dim),
                                        dtype=mean_grads_b.dtype,
                                        device=mean_grads_b.device)

            valid_rows = mapped_indices >= 0
            if valid_rows.any():
                assigned_grads[valid_rows] = mean_grads_b[mapped_indices[valid_rows]]

            grad_tensor_sentence_flat[b] = assigned_grads

        sentence_gradient = grad_tensor_sentence_flat.view(batch_size, seq_len, topk_size, hidden_dim)  # [B, T, V, H]
        ######################################################################################

        # === 4. Compute global gradient ===
        gathered_global_dict = gather_object([global_dict])
        

        # First, merge all the global dictionaries. If the token_id is already in the global dictionary, we stack the gradients
        final_global_dict = {}
        for gathered_dict in gathered_global_dict:
            for token_id, (sum_grad, cnt) in gathered_dict.items():
                # Ensure both sum_grad and count are tensors on the *local* device
                sum_grad = sum_grad.to(h.device, non_blocking=True)
                cnt = torch.as_tensor(cnt, device=h.device)
                if token_id in final_global_dict:
                    prev_sum_grad, prev_cnt = final_global_dict[token_id]
                    final_global_dict[token_id] = (prev_sum_grad + sum_grad, prev_cnt + cnt)
                else:
                    final_global_dict[token_id] = (sum_grad, cnt)

        # Compute the mean of the gradients in the global dictionary
        total_tokens = 0
        for token_id in final_global_dict:
            sum_grad, cnt = final_global_dict[token_id]
            total_tokens += cnt
            final_global_dict[token_id] = sum_grad / cnt

        # We use the same logic as before to map the global gradient back to the sentence_gradient tensor
        grad_tensor_global_flat = torch.zeros((batch_size * seq_len * topk_size, hidden_dim), dtype=h.dtype, device=h.device)

        # Flatten token_ids for indexing
        token_ids_flat_global = token_ids.flatten()  # [B*T*V]

        unique_token_ids_global = torch.tensor(list(final_global_dict.keys()), device=h.device, dtype=torch.long)  # [num_unique_tokens]
        mean_grads_global = torch.stack(list(final_global_dict.values()))  # [num_unique_tokens, H]

        for j, tid in enumerate(unique_token_ids_global):
            rows = torch.nonzero(token_ids_flat_global == tid, as_tuple=False).squeeze(-1)
            if rows.numel() > 0:
                grad_tensor_global_flat.index_copy_(0, rows, mean_grads_global[j].expand(rows.size(0), -1))

        global_gradient = grad_tensor_global_flat.view(batch_size, seq_len, topk_size, hidden_dim)  # [B, T, V, H]
        ######################################################################################
        
        del grad_tensor_sentence_flat, grad_tensor_global_flat, token_ids_flat, token_ids_flat_global, unique_token_ids_global, mean_grads_global
        return g, sentence_gradient, global_gradient, dicts_per_sentence, final_global_dict
            

    @profiling_decorator
    def _compute_hessian_curvature_mask(self, full_update_term, completion_mask):
        curvature_mask = torch.ones_like(completion_mask)
        if full_update_term is None:
            return curvature_mask, { "total": curvature_mask }

        masks = {}
        curvature_mask = self._compute_hessian_token_mask(full_update_term, curvature_mask, masks)
        curvature_mask = self._compute_hessian_sentence_mask(full_update_term, curvature_mask, masks)
        curvature_mask = self._compute_hessian_global_mask(full_update_term, curvature_mask, masks)

        masks["total"] = curvature_mask
        return curvature_mask, masks
    
    @profiling_decorator
    def _compute_fisher_curvature_mask(self, approx_kl, completion_mask):
        curvature_mask = torch.ones_like(completion_mask)

        if approx_kl is None:
            return curvature_mask, { "total": curvature_mask }

        masks = {}
        curvature_mask = self._compute_fisher_token_mask(approx_kl, curvature_mask, masks)
        curvature_mask = self._compute_fisher_sentence_mask(approx_kl, curvature_mask, masks)
        curvature_mask = self._compute_fisher_global_mask(approx_kl, curvature_mask, masks)

        masks["total"] = curvature_mask
        return curvature_mask, masks

    def _compute_curvature_sequential_mask(self, full_update_term, approx_kl, completion_mask, \
                                        hidden_states, per_token_logps, probs, action_one_hot, inputs, top_k_token_ids, mode, update_optim_model):
        curvature_mask = torch.ones_like(completion_mask)
        if full_update_term is None:
            return curvature_mask, { "total": curvature_mask }, { "total": curvature_mask }

        masks_hessian = { "total": curvature_mask.clone() }
        masks_fisher = { "total": curvature_mask.clone() }

        token_update = self.hessian_token_mask_tau != 0.0 or self.fisher_token_mask_tau != 0.0
        sentence_update = self.hessian_sentence_mask_tau != 0.0 or self.fisher_sentence_mask_tau != 0.0
        global_update = self.hessian_global_mask_tau != 0.0 or self.fisher_global_mask_tau != 0.0

        # clone completion_mask
        updated_mask = completion_mask.clone()
        if token_update:
            curvature_mask_hessian_token = self._compute_hessian_token_mask(full_update_term, updated_mask, masks_hessian)
            curvature_mask_fisher_token = self._compute_fisher_token_mask(approx_kl, updated_mask, masks_fisher)
            updated_mask = updated_mask * curvature_mask_hessian_token * curvature_mask_fisher_token

            # Evaluate the gradients and curvatures after the curvature masking if needed
            last_masking = False if sentence_update or global_update else True
            self._compute_and_log_gradients_linear_model(hidden_states, per_token_logps, probs, action_one_hot, inputs, updated_mask, top_k_token_ids, mode, prefix="after_token_mask", update_optim_model=(update_optim_model and last_masking))

        if sentence_update:
            curvature_mask_hessian_sentence = self._compute_hessian_sentence_mask(full_update_term, updated_mask, masks_hessian)
            curvature_mask_fisher_sentence = self._compute_fisher_sentence_mask(approx_kl, updated_mask, masks_fisher)
            updated_mask = updated_mask * curvature_mask_hessian_sentence * curvature_mask_fisher_sentence

            # Evaluate the gradients and curvatures after the curvature masking
            last_masking = False if global_update else True
            self._compute_and_log_gradients_linear_model(hidden_states, per_token_logps, probs, action_one_hot, inputs, updated_mask, top_k_token_ids, mode, prefix="after_sentence_mask", update_optim_model=(update_optim_model and last_masking))

        if global_update:
            curvature_mask_hessian_global = self._compute_hessian_global_mask(full_update_term, updated_mask, masks_hessian)
            curvature_mask_fisher_global = self._compute_fisher_global_mask(approx_kl, updated_mask, masks_fisher)
            updated_mask = updated_mask * curvature_mask_hessian_global * curvature_mask_fisher_global

            # Evaluate the gradients and curvatures after the curvature masking
            self._compute_and_log_gradients_linear_model(hidden_states, per_token_logps, probs, action_one_hot, inputs, updated_mask, top_k_token_ids, mode, prefix="after_global_mask", update_optim_model=update_optim_model)

        return updated_mask, masks_hessian, masks_fisher
    
    def _compute_hessian_token_mask(self, full_update_term, curvature_mask, masks):
        if self.hessian_token_mask_tau != 0.0:
            if self.hessian_symmetric_clipping:
                curvature_mask_token = ((full_update_term["token"] < self.hessian_token_mask_tau) * (full_update_term["token"] > -self.hessian_token_mask_tau))
            else:
                curvature_mask_token = ((full_update_term["token"] < self.hessian_token_mask_tau) * (full_update_term["token"] >= 0.0))
            masks["token"] = curvature_mask_token
            masks["total"] = masks["total"] * curvature_mask_token if "total" in masks else curvature_mask_token
            curvature_mask = curvature_mask * curvature_mask_token
        return curvature_mask

    def _compute_hessian_sentence_mask(self, full_update_term, curvature_mask, masks):
        if self.hessian_sentence_mask_tau != 0.0:
            if self.hessian_symmetric_clipping:
                curvature_mask_sentence = ((full_update_term["sentence"] < self.hessian_sentence_mask_tau) * (full_update_term["sentence"] > -self.hessian_sentence_mask_tau)).unsqueeze(-1).expand_as(curvature_mask)
            else:
                curvature_mask_sentence = ((full_update_term["sentence"] < self.hessian_sentence_mask_tau) * (full_update_term["sentence"] >= 0.0)).unsqueeze(-1).expand_as(curvature_mask)
            masks["sentence"] = curvature_mask_sentence
            masks["total"] = masks["total"] * curvature_mask_sentence if "total" in masks else curvature_mask_sentence
            curvature_mask = curvature_mask * curvature_mask_sentence
        return curvature_mask

    def _compute_hessian_global_mask(self, full_update_term, curvature_mask, masks):
        if self.hessian_global_mask_tau != 0.0:
            if self.hessian_symmetric_clipping:
                curvature_mask_global = ((full_update_term["global"] < self.hessian_global_mask_tau) * (full_update_term["global"] > -self.hessian_global_mask_tau)).unsqueeze(-1).expand_as(curvature_mask)
            else:
                curvature_mask_global = ((full_update_term["global"] < self.hessian_global_mask_tau) * (full_update_term["global"] >= 0.0)).unsqueeze(-1).expand_as(curvature_mask)
            masks["global"] = curvature_mask_global
            masks["total"] = masks["total"] * curvature_mask_global if "total" in masks else curvature_mask_global
            curvature_mask = curvature_mask * curvature_mask_global
        return curvature_mask

    def _compute_fisher_token_mask(self, approx_kl, curvature_mask, masks):
        if self.fisher_token_mask_tau != 0.0:
            curvature_mask_token = (approx_kl["token"] < self.fisher_token_mask_tau)
            masks["token"] = curvature_mask_token
            masks["total"] = masks["total"] * curvature_mask_token if "total" in masks else curvature_mask_token
            curvature_mask = curvature_mask * curvature_mask_token
        return curvature_mask

    def _compute_fisher_sentence_mask(self, approx_kl, curvature_mask, masks):
        if self.fisher_sentence_mask_tau != 0.0:
            curvature_mask_sentence = (approx_kl["sentence"] < self.fisher_sentence_mask_tau).unsqueeze(-1).expand_as(curvature_mask)
            masks["sentence"] = curvature_mask_sentence
            masks["total"] = masks["total"] * curvature_mask_sentence if "total" in masks else curvature_mask_sentence
            curvature_mask = curvature_mask * curvature_mask_sentence
        return curvature_mask

    def _compute_fisher_global_mask(self, approx_kl, curvature_mask, masks):
        if self.fisher_global_mask_tau != 0.0:
            curvature_mask_global = (approx_kl["global"] < self.fisher_global_mask_tau).unsqueeze(-1).expand_as(curvature_mask)
            masks["global"] = curvature_mask_global
            masks["total"] = masks["total"] * curvature_mask_global if "total" in masks else curvature_mask_global
            curvature_mask = curvature_mask * curvature_mask_global
        return curvature_mask

    @profiling_decorator
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


    # TODO: Remove or fix this function - currently implementing the wrong thing
    def _compute_and_log_gradient_direction(self, features, one_hot_actions, softmax_probs, advantages, completion_mask, mode):
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
            
    @profiling_decorator
    def _log_feature_stats(self, logprobs, features, softmax_probs, completion_mask, mode):
        # Compute feature norm
        with torch.inference_mode():
            seq_len = logprobs.shape[1]
            feature_norm = torch.norm(features, dim=-1)
            gathered_feature_norm = self._gather_masked_tensor_across_processes(feature_norm, completion_mask)
            self._accumulate_stats(
                data=gathered_feature_norm,
                metric_name='per_token_feature_norm',
                mode=mode,
                stats=DEFAULT_STATS,
            )

            # ---- Variance of hidden states across samples ----
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

            policy_norm_sq = (softmax_probs**2).sum(dim=-1)
            probs = torch.exp(logprobs)
            policy_error_norm = torch.abs(1 - 2 * probs + policy_norm_sq)

            gathered_policy_error_norm = self._gather_masked_tensor_across_processes(policy_error_norm, completion_mask)
            self._accumulate_stats(
                data=gathered_policy_error_norm,
                metric_name='per_token_policy_error_norm',
                mode=mode
            )

    @profiling_decorator
    def _log_gradients_stats(self, per_token_gradient, dicts_per_sentence, completion_mask, mode, prefix=""):
        batch_size, seq_len, topk_size, hidden_dim = per_token_gradient.shape
        with torch.inference_mode():
            # Log per token gradient norm
            token_grad = per_token_gradient.view(batch_size, seq_len, -1) # (B, T, V*H)
            norm_per_token_grad = torch.norm(token_grad, dim=-1) # (B, T)
            gathered_per_token_gradient = self._gather_masked_tensor_across_processes(norm_per_token_grad, completion_mask)
            self._accumulate_stats(
                data=gathered_per_token_gradient,
                metric_name=f'{prefix}_per_token_gradient_norm' if prefix else "per_token_gradient_norm",
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p1': lambda x: x.quantile(0.01).item(),
                    'p5': lambda x: x.quantile(0.05).item(),
                    'p10': lambda x: x.quantile(0.1).item(),
                }
            )

            # Log per sentence gradient norm
            norms = []
            for d in dicts_per_sentence:
                # If d is empty, append a 0 tensor
                if not d:
                    norms.append(torch.tensor(0.0, device=self.accelerator.device, dtype=per_token_gradient.dtype))
                    continue
                values = list(d.values())
                stacked_values = torch.stack(values)  # [num_tokens, hidden_dim]
                norm_sq = torch.norm(stacked_values, dim=-1).pow(2).sum()  # More efficient
                norms.append(norm_sq.sqrt())

            norm_tensor = torch.stack(norms)
            gathered_norm_tensor = self.accelerator.gather(norm_tensor).float()
            self._accumulate_stats(
                data=gathered_norm_tensor,
                metric_name=f'{prefix}_per_sentence_gradient_norm' if prefix else "per_sentence_gradient_norm",
                mode=mode,
                stats=DEFAULT_STATS,
            )


            # Log per sentence gradient full variance
            # mean of squared L2 Norm: compute average of gathered_norm_tensor square
            mean_sq_l2_norm = (gathered_norm_tensor**2).mean()
            # We need to compute the norm of the mean gradient
            N = len(dicts_per_sentence)

            all_keys = set()
            for d in dicts_per_sentence:
                all_keys.update(d.keys())
            
            # # Pre-allocate sum_v with proper device and dtype
            # first_value = next(iter(next(d for d in dicts_per_sentence if d).values()))
            # sum_v = {k: torch.zeros_like(first_value) for k in all_keys}
            

            # Reducing dicts_per_sentence into a single dict
            pairs = [(k, v) for d in dicts_per_sentence for k, v in d.items()]
            if not pairs:
                sum_v = {}
            else:
                vals = torch.stack([v for _, v in pairs], dim=0).to(self.accelerator.device)    # [M, D]
                keys = torch.tensor([k for k, _ in pairs], device=self.accelerator.device, dtype=torch.long)
                uniq, inv = torch.unique(keys, sorted=False, return_inverse=True)
                add = torch.zeros(uniq.numel(), vals.size(1), device=self.accelerator.device, dtype=vals.dtype)
                add.index_add_(0, inv, vals)
                sum_v = {k.item(): add[i] for i, k in enumerate(uniq)}

            dict_keys = self.accelerator.gather(torch.tensor(len(sum_v.keys()), device=self.accelerator.device))
            max_keys = max(dict_keys)

            if max_keys == 0:
                all_sum_v = {}
            else:
                # Convert to tensor
                sum_v_tensor, sum_v_key_count_tensor = self._list_of_dicts_to_tensor([sum_v], max_num_tokens=max_keys, hidden_dim=hidden_dim, grad_dtype=per_token_gradient.dtype, key_count_dim=1)
                
                gathered_sum_v = self.accelerator.gather(sum_v_tensor)
                gathered_sum_v_key = self.accelerator.gather(sum_v_key_count_tensor)
                all_N = sum(self.accelerator.gather(torch.tensor(N, device=self.accelerator.device)))

                all_sum_v = self.aggregate_global_dict(gathered_sum_v, gathered_sum_v_key, dict_keys)

            if not all_sum_v:
                norm_sq_mean_v = 0.0
            else:
                mean_v_tensor = torch.stack(list(all_sum_v.values())) / all_N
                norm_sq_mean_v = torch.sum(mean_v_tensor ** 2)

            sentence_gradient_variance = (mean_sq_l2_norm - norm_sq_mean_v).unsqueeze(0)

            self._accumulate_stats(
                data=sentence_gradient_variance.float(),
                metric_name=f'{prefix}_sentence_full_gradient_variance' if prefix else "sentence_full_gradient_variance",
                mode=mode,
                stats={ 'metric': lambda x: x.mean().item(),
                    'max_squared_error': lambda x: x.max().item(),
                    'p75': lambda x: x.quantile(0.75).item(),
                    'p90': lambda x: x.quantile(0.9).item(),
                    'p95': lambda x: x.quantile(0.95).item(),
                    'p99': lambda x: x.quantile(0.99).item(), },
            )

    @profiling_decorator
    def _compute_and_log_hessian_coefficient(self, h, one_hot_actions, softmax_probs, advantages, completion_mask, 
                                             effective_g, grad_norm_sq, granularity, mode, prefix=""):
        """
        Compute the hessian coefficient for each token in the sequence.
        """
        def compute_hessian_token_level(G, A, v, h, pi, mask, gather_fn):
            # Step 1: Compute u = G @ h
            # G: (B, T, V, H), h: (B, T, H) => u: (B, T, V)
            u = torch.einsum('btvh,bth->btv', G, h)  # u = G h

            # Step 2: Compute per-sample contributions
            per_token_hessian = compute_per_sample_hessian_contributions(u, A, v, pi)

            # Step 3: Masked sentence-level average
            return per_token_hessian
        
        def compute_hessian_sentence_level(g_sent, A, v, h, pi, mask, gather_fn):
            # Compute per-sample Hessian contributions
            contrib = compute_hessian_token_level(g_sent, A, v, h, pi, mask, gather_fn)

            # Masked sentence-level average
            contrib = contrib * mask         # (B, T)
            sentence_sum = contrib.sum(dim=1)        # (B,)
            sentence_len = mask.sum(dim=1).clamp(min=1.0)  # (B,)
            hessian_per_sentence = sentence_sum / sentence_len  # (B,)
            
            return hessian_per_sentence
        
        def compute_hessian_global_level(g_global, A, v, h, pi, mask, gather_fn):
            # Compute per-sample Hessian contributions
            contrib = compute_hessian_token_level(g_global, A, v, h, pi, mask, gather_fn)

            # Masked sentence-level average
            contrib = contrib * mask         # (B, T)
            sentence_sum = contrib.sum(dim=1)        # (B,)
            sentence_len = mask.sum(dim=1).clamp(min=1.0)  # (B,)
            hessian_per_sentence = sentence_sum / sentence_len  # (B,)

            # Global-level average
            hessian_global = gather_fn(hessian_per_sentence).mean() # (1,)
            
            return hessian_global
        
        def compute_per_sample_hessian_contributions(u, A, v, pi):
            # Step 2: First term: (vᵗ u)^2
            dot = (v * u).sum(dim=-1)  # (B, T)
            first_term = dot ** 2      # (B, T)

            # Step 3: Compute Fisher matrix F = diag(pi) - pi @ piᵗ
            F = torch.diag_embed(pi) - pi.unsqueeze(-1) @ pi.unsqueeze(-2)    # (B, T, V, V)

            # Step 4: Compute uᵗ F u = Tr(F u uᵗ)
            u_unsq = u.unsqueeze(-1)                               # (B, T, V, 1)
            F_u = torch.matmul(F, u_unsq).squeeze(-1)              # (B, T, V)
            second_term = (F_u * u).sum(dim=-1)                    # (B, T)

            # Step 5: Final term: A * (first - second)
            delta = first_term - second_term                       # (B, T)
            contrib = A * delta                        # (B, T)
            return contrib

        def compute_hessian_inference_wrapper(compute_fn, allow_grad):
            if allow_grad:
                return compute_fn()
            else:
                with torch.inference_mode():
                    return compute_fn()
                
        # Helper function to encapsulate the calculations and logging
        def _compute_hessian_coefficients(effective_g, grad_norm_sq, hessian_lambda, hessian_fn, granularity, log_gather_fn=None):
            gather_fn = self.accelerator.gather
            # === Dimensions ===
            # B: batch_size, T: seq_len, V: vocab_size, H: hidden_dim, D: V * H
            batch_size, seq_len, vocab_size = softmax_probs.shape
            hidden_dim = h.shape[-1]
            D = vocab_size * hidden_dim  # flattened feature dimension

            # === 1. Extract final layer hidden states and policy info ===
            pi = softmax_probs                                 # [B, T, V]
            e_a = one_hot_actions.to(h.dtype)                  # [B, T, V]
            v = e_a - pi                                       # [B, T, V]
            A = advantages.unsqueeze(1).to(h.dtype)            # [B, 1]
            mask = completion_mask.to(h.dtype)                 # [B, T]

            # === 2. Compute hessian coefficient ===
            hessian = compute_hessian_inference_wrapper(
                lambda: hessian_fn(effective_g, A, v, h, pi, mask, gather_fn),
                hessian_lambda != 0.0
            )

            # === 3. Compute full update term ===
            lr = self.lr_scheduler.get_last_lr()[0]

            full_update_term = lr * grad_norm_sq + 0.5 * hessian * lr**2

            # === 4. Log stats ===
            if granularity == "global":
                hessian = hessian.unsqueeze(0).float()
                full_update_term = full_update_term.unsqueeze(0).float()
            log_hessian_stats(hessian, full_update_term, mode, f"{prefix}_{granularity}" if prefix else f"{granularity}", gather_fn=log_gather_fn)
            return hessian, full_update_term

        # # Helper function to encapsulate the calculations and logging
        # def _compute_hessian_coefficients():
        #     gather_fn = self.accelerator.gather
        #     # === Dimensions ===
        #     # B: batch_size, T: seq_len, V: vocab_size, H: hidden_dim, D: V * H
        #     batch_size, seq_len, vocab_size = softmax_probs.shape
        #     hidden_dim = h.shape[-1]
        #     D = vocab_size * hidden_dim  # flattened feature dimension

        #     # === 1. Extract final layer hidden states and policy info ===
        #     pi = softmax_probs                                 # [B, T, V]
        #     e_a = one_hot_actions.to(h.dtype)                  # [B, T, V]
        #     v = e_a - pi                                       # [B, T, V]
        #     A = advantages.unsqueeze(1).to(h.dtype)            # [B, 1]
        #     mask = completion_mask.to(h.dtype)                 # [B, T]
            

        #     # === 2. Compute effective gradients ===
        #     effective_g_token, _ = self._compute_effective_gradients(g_token)
        #     effective_g_sent, effective_g_dicts_per_sentence = self._compute_effective_gradients(g_sent, dicts_per_sentence)
        #     effective_g_global, effective_g_global_dict = self._compute_effective_gradients(g_global, final_global_dict)

        #     # === 3. Compute hessian coefficient ===
        #     hessian_token = compute_hessian_inference_wrapper(
        #         lambda: compute_hessian_token_level(A, v, h, pi, effective_g_token),
        #         self.hessian_token_lambda != 0.0
        #     )
        #     hessian_sentence = compute_hessian_inference_wrapper(
        #         lambda: compute_hessian_sentence_level(effective_g_sent, A, v, h, pi, mask, gather_fn),
        #         self.hessian_sentence_lambda != 0.0
        #     )
        #     hessian_global = compute_hessian_inference_wrapper(
        #         lambda: compute_hessian_global_level(effective_g_global, A, v, h, pi, mask, gather_fn),
        #         self.hessian_global_lambda != 0.0
        #     )

        #     # === 4. Compute full update term ===
        #     lr = self.lr_scheduler.get_last_lr()[0]
            
        #     # Token-level grad norm: simple dot product, as the logits map directly from g_token to effective_g_token. Using bmm to avoid materializing big tensors.
        #     B, T, V, H = g_token.shape
        #     bt = B * T
        #     token_grad_norm_sq = torch.bmm(
        #         g_token.reshape(bt, 1, V * H).contiguous(),
        #         effective_g_token.reshape(bt, V * H, 1).contiguous()
        #     ).reshape(B, T)
            
        #     # Sentence-level grad norm: requires sparse dot product over dict of sequences
        #     sentence_grad_norm_sq = self._sparse_dot_product_sentence_level(dicts_per_sentence, effective_g_dicts_per_sentence)
        #     global_grad_norm_sq = self._sparse_dot_product_global_level(final_global_dict, effective_g_global_dict)

        #     full_token_update_term = lr * token_grad_norm_sq + 0.5 * hessian_token * lr**2
        #     full_sentence_update_term = lr * sentence_grad_norm_sq + 0.5 * hessian_sentence * lr**2
        #     full_global_update_term = lr * global_grad_norm_sq + 0.5 * hessian_global * lr**2

        #     # === 5. Log stats ===
        #     log_hessian_stats(hessian_global.unsqueeze(0).float(), full_global_update_term, mode, f"{prefix}_global" if prefix else "global")
        #     log_hessian_stats(hessian_sentence, full_sentence_update_term, mode, f"{prefix}_per_sentence" if prefix else "per_sentence", gather_fn=lambda t: self.accelerator.gather(t).float())
        #     log_hessian_stats(hessian_token, full_token_update_term, mode, f"{prefix}_per_token" if prefix else "per_token", gather_fn=lambda t: self._gather_masked_tensor_across_processes(t, completion_mask).float())

        #     hessian_stats = {
        #         "global": hessian_global,
        #         "sentence": hessian_sentence,
        #         "token": hessian_token,
        #     }
        #     full_update_stats = {
        #         "global": full_global_update_term,
        #         "sentence": full_sentence_update_term,
        #         "token": full_token_update_term,
        #     }
        #     return hessian_stats, full_update_stats
        
        def log_hessian_stats(hessian_coeff, full_update_term, mode, metric_prefix, gather_fn=None):
            if gather_fn is not None:
                gathered_hessian_coeff = gather_fn(hessian_coeff)
                gathered_full_update_term = gather_fn(full_update_term)
            else:
                gathered_hessian_coeff = hessian_coeff
                gathered_full_update_term = full_update_term
            
            self._accumulate_stats(
                data=gathered_hessian_coeff.float(),
                metric_name=f'{metric_prefix}_hessian_coeff',
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p99': lambda x: x.quantile(0.99).item(),
                }
            )

            self._accumulate_stats(
                data=torch.abs(gathered_hessian_coeff).float(),
                metric_name=f'{metric_prefix}_hessian_coeff_abs',
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p99': lambda x: x.quantile(0.99).item(),
                }
            )

            # # log quadractic coefficient
            # lr = self.lr_scheduler.get_last_lr()[0]
            # quadratic_coeff = 0.5 * gathered_hessian_coeff * lr**2
            # self._accumulate_stats(
            #     data=quadratic_coeff.float(),
            #     metric_name=f'{metric_prefix}_hessian_coeff_quadratic_coeff',
            #     mode=mode,
            #     stats={
            #         **DEFAULT_STATS,
            #         'p85': lambda x: x.quantile(0.85).item(),
            #         'p90': lambda x: x.quantile(0.9).item(),
            #         'p95': lambda x: x.quantile(0.95).item(),
            #         'p99': lambda x: x.quantile(0.99).item(),
            #     }
            # )

            # log full update term
            self._accumulate_stats(
                data=gathered_full_update_term.float(),
                metric_name=f'{metric_prefix}_full_update_term',
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p85': lambda x: x.quantile(0.85).item(),
                    'p90': lambda x: x.quantile(0.9).item(),
                    'p95': lambda x: x.quantile(0.95).item(),
                    'p99': lambda x: x.quantile(0.99).item(),
                }
            )

        def _get_hessian_lambda_and_fn(granularity):
            if granularity == "token":
                return self.hessian_token_lambda, compute_hessian_token_level, lambda t: self._gather_masked_tensor_across_processes(t, completion_mask)
            elif granularity == "sentence":
                return self.hessian_sentence_lambda, compute_hessian_sentence_level, lambda t: self.accelerator.gather(t)
            elif granularity == "global":
                return self.hessian_global_lambda, compute_hessian_global_level, None

        hessian_lambda, hessian_fn, log_gather_fn = _get_hessian_lambda_and_fn(granularity)

        if self.hessian_global_lambda == 0.0 and self.hessian_sentence_lambda == 0.0 and self.hessian_token_lambda == 0.0:
            # If lambda is 0, compute in inference mode to save resources
            with torch.inference_mode():
                hessian_stats, full_update_stats = _compute_hessian_coefficients(effective_g, grad_norm_sq, hessian_lambda, hessian_fn, granularity, log_gather_fn)
        else:
            # If lambda is non-zero, compute normally to allow for gradients
            hessian_stats, full_update_stats = _compute_hessian_coefficients(effective_g, grad_norm_sq, hessian_lambda, hessian_fn, granularity, log_gather_fn)
            
        return hessian_stats, full_update_stats
    

    

    @profiling_decorator
    def _compute_and_log_fisher_curvature(self, h, one_hot_actions, softmax_probs, advantages, completion_mask, \
                                            effective_g, granularity, mode, prefix=""):
        """
        Compute the Fisher curvature for each token in the sequence. We compute g^T F g, where F is the Fisher information matrix.
        """
        def compute_fisher_token_level(G, v, h, mask, gather_fn):
            """
            Per-token Fisher curvature.
            
            Args:
                v: [B, T, V]
                h: [B, T, H]
                G: [B, T, V, H]
            
            Returns:
                curvature: [B, T]
            """
            # einsum over batch and time
            s = torch.einsum('btv,btvh,bth->bt', v, G, h)
            return s ** 2
        
        def compute_fisher_sentence_level(G, v, h, mask, gather_fn):
            """
            Per-sentence Fisher curvature.
            
            Args:
                v: [B, T, V]
                h: [B, T, H]
                G: [B, T, V, H]
            
            Returns:
                curvature: [B]
            """
            # einsum over batch only
            s = torch.einsum('btv,btvh,bth->bt', v, G, h)  # [B, T]

            fisher = s ** 2
            # mask and average over time
            fisher = (fisher * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B]
            return fisher  # [B]
        
        def compute_fisher_global_level(g_global, v, h, mask, gather_fn):
            """
            Global Fisher curvature with a single gradient direction.
            
            Args:
                v: [B, T, V]
                h: [B, T, H]
                g_flat: [V * H]
            
            Returns:
                curvature: scalar
            """
            # einsum globally
            s = torch.einsum('btv,btvh,bth->bt', v, g_global, h)  # [B, T]

            fisher = s ** 2
            fisher = (fisher * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B]
            fisher = gather_fn(fisher).mean()

            return fisher

        def compute_fisher_inference_wrapper(compute_fn, allow_grad):
            if allow_grad:
                return compute_fn()
            else:
                with torch.inference_mode():
                    return compute_fn()

        # Helper function to encapsulate the calculations and logging
        
        def _compute_fisher_curvatures(fisher_lambda, fisher_fn, granularity, log_gather_fn=None):
            gather_fn = self.accelerator.gather
            # === Dimensions ===
            # B: batch_size, T: seq_len, V: vocab_size, H: hidden_dim, D: V * H
            batch_size, seq_len, vocab_size = softmax_probs.shape
            hidden_dim = h.shape[-1]
            D = vocab_size * hidden_dim  # flattened feature dimension

            # === 1. Extract final layer hidden states and policy info ===
            pi = softmax_probs                                 # [B, T, V]
            e_a = one_hot_actions.to(h.dtype)                  # [B, T, V]
            v = e_a - pi                                       # [B, T, V]
            A = advantages.unsqueeze(1).to(h.dtype)            # [B, 1]
            mask = completion_mask.to(h.dtype)                 # [B, T]
            mask_f = mask.float().to(h.dtype)

            # === 2. Compute hessian coefficient ===
            fisher = compute_fisher_inference_wrapper(
                lambda: fisher_fn(effective_g, v, h, mask, gather_fn),
                fisher_lambda != 0.0
            )

            # === 3. Compute full update term ===
            lr = self.lr_scheduler.get_last_lr()[0]
            kl_term = 0.5 * fisher * lr**2

            # === 4. Log stats ===
            if granularity == "global":
                fisher = fisher.unsqueeze(0).float()
                kl_term = kl_term.unsqueeze(0).float()
            log_fisher_stats(fisher, kl_term, mode, f"{prefix}_{granularity}" if prefix else f"{granularity}", gather_fn=log_gather_fn)

            return fisher, kl_term


            # # === 3. Compute Fisher curvature ===
            # fisher_token = compute_fisher_inference_wrapper(
            #     lambda: compute_fisher_token_level(effective_g_token, v, h),
            #     self.fisher_token_lambda != 0.0
            # )
            # fisher_sentence = compute_fisher_inference_wrapper(
            #     lambda: compute_fisher_sentence_level(effective_g_sent, v, h, mask),
            #     self.fisher_sentence_lambda != 0.0
            # )
            # fisher_global = compute_fisher_inference_wrapper(
            #     lambda: compute_fisher_global_level(effective_g_global, v, h, mask, gather_fn),
            #     self.fisher_global_lambda != 0.0
            # )

            # # KL divergence
            # lr = self.lr_scheduler.get_last_lr()[0]
            # kl_token = 0.5 * fisher_token * lr**2
            # kl_sentence = 0.5 * fisher_sentence * lr**2
            # kl_global = 0.5 * fisher_global * lr**2

            # === 5. Log stats ===
            # log_fisher_stats(fisher_global.unsqueeze(0).float(), kl_global, mode, f"{prefix}_global" if prefix else "global")
            # log_fisher_stats(fisher_sentence, kl_sentence, mode, f"{prefix}_per_sentence" if prefix else "per_sentence", gather_fn=lambda t: self.accelerator.gather(t).float())
            # log_fisher_stats(fisher_token, kl_token, mode, f"{prefix}_per_token" if prefix else "per_token", gather_fn=lambda t: self._gather_masked_tensor_across_processes(t, completion_mask).float())

            # curvature_stats = {
            #     "global": fisher_global,
            #     "sentence": fisher_sentence,
            #     "token": fisher_token,
            # }
            
            # kl_stats = {
            #     "global": kl_global,
            #     "sentence": kl_sentence,
            #     "token": kl_token,
            # }
            # return curvature_stats, kl_stats
        
        def log_fisher_stats(fisher_curvature, kl_div, mode, metric_prefix, gather_fn=None):
            if gather_fn is not None:
                gathered_fisher_curvature = gather_fn(fisher_curvature)
                gathered_kl_div = gather_fn(kl_div)
            else:
                gathered_fisher_curvature = fisher_curvature
                gathered_kl_div = kl_div
            
            self._accumulate_stats(
                data=gathered_fisher_curvature.float(),
                metric_name=f'{metric_prefix}_fisher_curvature',
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p85': lambda x: x.quantile(0.85).item(),
                    'p90': lambda x: x.quantile(0.9).item(),
                    'p95': lambda x: x.quantile(0.95).item(),
                    'p99': lambda x: x.quantile(0.99).item(),
                }
            )

            self._accumulate_stats(
                data=gathered_kl_div.float(),
                metric_name=f'{metric_prefix}_fisher_kl_divergence',
                mode=mode,
                stats={
                    **DEFAULT_STATS,
                    'p85': lambda x: x.quantile(0.85).item(),
                    'p90': lambda x: x.quantile(0.9).item(),
                    'p95': lambda x: x.quantile(0.95).item(),
                    'p99': lambda x: x.quantile(0.99).item(),
                }
            )

        def _get_fisher_lambda_and_fn(granularity):
            if granularity == "token":
                return self.fisher_token_lambda, compute_fisher_token_level, lambda t: self._gather_masked_tensor_across_processes(t, completion_mask)
            elif granularity == "sentence":
                return self.fisher_sentence_lambda, compute_fisher_sentence_level, lambda t: self.accelerator.gather(t)
            elif granularity == "global":
                return self.fisher_global_lambda, compute_fisher_global_level, None

        fisher_lambda, fisher_fn, log_gather_fn = _get_fisher_lambda_and_fn(granularity)

        if self.fisher_global_lambda == 0.0 and self.fisher_sentence_lambda == 0.0 and self.fisher_token_lambda == 0.0:
            # If lambda is 0, compute in inference mode to save resources
            with torch.inference_mode():
                fisher_curvatures, approx_kl = _compute_fisher_curvatures(fisher_lambda, fisher_fn, granularity, log_gather_fn)
        else:
            # If lambda is non-zero, compute normally to allow for gradients
            fisher_curvatures, approx_kl = _compute_fisher_curvatures(fisher_lambda, fisher_fn, granularity, log_gather_fn)
            
        return fisher_curvatures, approx_kl
    
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
            old_per_token_logps, _, old_last_token_embeddings, _, _, _ = self._get_per_token_logps(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep, return_hidden_states=True
            )
            # else:
                # old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps, _, _, _, _, _ = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps, _, _, _, _, _ = self._get_per_token_logps(
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
                reward_kwargs['dataset'] = self.train_dataset if self.control.should_evaluate else self.eval_dataset
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

        if self.advantage_target_std is None or self.advantage_target_std == 0.0:
            advantages = rewards - mean_grouped_rewards
        else:
            std_grouped_rewards = std_grouped_rewards * self.advantage_target_std
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        return advantages
    
    def _compute_final_loss(self, per_token_loss, completion_mask, updated_mask, masks, curvature_estimator, completion_ids):
        mode = "eval" if self.control.should_evaluate else "train"
        if self.curvature_masking:
            masks_hessian = masks['hessian']
            masks_fisher = masks['fisher']
            self._log_masking_stats(masks_hessian, masks_fisher, completion_mask, completion_ids, mode)
            
            # Final mask
            completion_mask = completion_mask * updated_mask

        # --- Token-level adjustments ---
        if self.hessian_token_lambda != 0.0:
            per_token_loss = per_token_loss + self.hessian_token_lambda * curvature_estimator['hessian']["token"]
        if self.fisher_token_lambda != 0.0:
            per_token_loss = per_token_loss + self.fisher_token_lambda * curvature_estimator['fisher']["token"]

        # --- Sentence-level loss calculation ---
        z = completion_mask.sum(dim=1).clamp(min=1)
        per_sentence_loss = (per_token_loss * completion_mask).sum(dim=1) / z

        # --- Sentence and Global-level adjustments ---
        if self.hessian_sentence_lambda != 0.0:
            per_sentence_loss = per_sentence_loss + self.hessian_sentence_lambda * curvature_estimator['hessian']["sentence"]
        if self.fisher_sentence_lambda != 0.0:
            per_sentence_loss = per_sentence_loss + self.fisher_sentence_lambda * curvature_estimator['fisher']["sentence"]
                
        global_loss = per_sentence_loss.mean()
        if self.hessian_global_lambda != 0.0:
            global_loss = global_loss + self.hessian_global_lambda * curvature_estimator['hessian']["global"]
        if self.fisher_global_lambda != 0.0:
            global_loss = global_loss + self.fisher_global_lambda * curvature_estimator['fisher']["global"]

        gather_loss_1 = self.accelerator.gather_for_metrics(per_sentence_loss)
        
        self._accumulate_stats(
            data=gather_loss_1,
            metric_name='policy_loss',
            mode=mode,
        )

        return global_loss

    def _log_masking_stats(self, masks_hessian, masks_fisher, completion_mask, completion_ids, mode):
        full_mask = { "total": masks_fisher['total'] * masks_hessian['total'] }

        for curvature_estimator in ["hessian", "fisher", "full"]:
            masks = masks_fisher if curvature_estimator == "fisher" else masks_hessian if curvature_estimator == "hessian" else full_mask
            if masks is None:
                continue
            for level in ["token", "sentence", "global", "total"]:
                if level in masks:
                    curvature_mask = masks[level].bool()
                    total_tokens = completion_mask.sum().clamp(min=1)
                    clipped_tokens = (~curvature_mask * completion_mask).sum()
                    gathered_clipped_tokens = self.accelerator.gather_for_metrics(clipped_tokens).sum()
                    gathered_total_tokens = self.accelerator.gather_for_metrics(total_tokens).sum()
                    ratio = (gathered_clipped_tokens / (gathered_total_tokens + 1e-8)).item()
                    metric_key = f"curvature_clip_ratio_{level}_{curvature_estimator}"
                    if metric_key not in self._metrics[mode]:
                        self._metrics[mode][metric_key] = []
                    self._metrics[mode][metric_key].append(ratio)
        # TODO: fix logging masking completions
        # self._log_masked_completions(self.processing_class,completion_ids, full_mask['total'], str(self.state.epoch))
        # self._log_masked_completions(self.processing_class,completion_ids, masks_fisher['total'], f"fisher_{str(self.state.epoch)}")
        # self._log_masked_completions(self.processing_class,completion_ids, masks_hessian['total'], f"hessian_{str(self.state.epoch)}")

    def _log_masked_completions(self, tokenizer, completion_ids, completion_mask, name_suffix=""):
        """
        Logs three text files via wandb.Artifact:
        1. Masked token IDs
        2. Masked tokens (decoded)
        3. Fully masked sentences (if any rows are fully masked)
        """
        # 1. Compute original lengths before padding
        orig_lengths = torch.full((completion_ids.size(0),), completion_ids.size(1),
                                dtype=torch.int64, device=self.accelerator.device).contiguous()


        padded_completion_ids = self.accelerator.pad_across_processes(completion_ids, dim=1, padding_value=tokenizer.pad_token_id).to(torch.int64)
        padded_completion_mask = self.accelerator.pad_across_processes(completion_mask, dim=1, padding_value=0).to(torch.int64)
        

        gathered_completion_ids = self.accelerator.gather(padded_completion_ids)
        gathered_completion_mask = self.accelerator.gather(padded_completion_mask)
        gathered_orig_lengths = self.accelerator.gather(orig_lengths)

        logger.info(f"orig_lengths: {orig_lengths}")
        logger.info(f"gathered_orig_lengths: {gathered_orig_lengths}")

        B, T = gathered_completion_ids.shape

        # Create in-memory string buffers
        buf_ids = io.StringIO()
        buf_tokens = io.StringIO()
        buf_sentences = io.StringIO()

        vocab_size = tokenizer.vocab_size

        for b in range(B):
            mask = gathered_completion_mask[b]
            ids = gathered_completion_ids[b]
            L = int(gathered_orig_lengths[b].item())

            ids_row = gathered_completion_ids[b, :L]
            mask_row = gathered_completion_mask[b, :L]

            # Get tokens that are masked and not pad
            masked_ids = ids_row[mask_row == 0]
            masked_ids = masked_ids[masked_ids != tokenizer.pad_token_id]

            for token_id in masked_ids:
                token_id_val = token_id.item()
                buf_ids.write(f"{token_id_val}\n")
                
                # Validate token ID is within valid range before decoding
                if 0 <= token_id_val < vocab_size:
                    try:
                        decoded_token = tokenizer.decode([token_id_val], skip_special_tokens=True)
                        buf_tokens.write(decoded_token + "\n")
                    except Exception as e:
                        # Log the error and skip this token
                        buf_tokens.write(f"[INVALID_TOKEN_{token_id_val}]\n")
                else:
                    buf_tokens.write(f"[OUT_OF_RANGE_TOKEN_{token_id_val}]\n")

            if mask.sum().item() == 0:
                try:
                    # Filter out invalid token IDs before decoding the full sequence
                    valid_ids = [id_val.item() for id_val in ids if 0 <= id_val.item() < vocab_size]
                    if valid_ids:
                        decoded_sentence = tokenizer.decode(valid_ids, skip_special_tokens=True)
                        buf_sentences.write(decoded_sentence + "\n")
                    else:
                        buf_sentences.write("[NO_VALID_TOKENS]\n")
                except Exception as e:
                    buf_sentences.write(f"[DECODE_ERROR: {str(e)}]\n")


        # Rewind buffers
        buf_ids.seek(0)
        buf_tokens.seek(0)
        buf_sentences.seek(0)

        # Write to temp files
        if self.accelerator.is_main_process:
            temp_files = []
            def write_temp_file(buf, filename):
                tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
                tmp.write(buf.read())
                tmp.flush()
                temp_files.append((tmp.name, filename))

            write_temp_file(buf_ids, f"masked_token_ids_{name_suffix}.txt")
            write_temp_file(buf_tokens, f"masked_tokens_{name_suffix}.txt")
            write_temp_file(buf_sentences, f"fully_masked_sentences_{name_suffix}.txt")

            # Create and log artifact
            artifact = wandb.Artifact(f"masked_tokens_{name_suffix}", type="mask_data")
            for path, filename in temp_files:
                artifact.add_file(path, name=filename)

            wandb.run.log_artifact(artifact)

            # Clean up
            for path, _ in temp_files:
                os.remove(path)
    
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
        local_length = torch.tensor([flat_tensor.shape[0]], device=self.accelerator.device)
        gathered_lengths = self.accelerator.gather(local_length).flatten()  # Shape: [num_processes]

        # Pad local tensor to max length across processes
        max_length = gathered_lengths.max().item()

        # if max_length is 0, return an empty tensor
        if max_length == 0:
            return torch.tensor([], device=flat_tensor.device)

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


    def _compute_grad_norm_sq(self, g, effective_g, sparse_g_dict, effective_g_dict, granularity):
            if granularity == "token":
                # Token-level grad norm: simple dot product, as the logits map directly from g_token to effective_g_token. Using bmm to avoid materializing big tensors.
                B, T, V, H = g.shape
                bt = B * T
                token_grad_norm_sq = torch.bmm(
                    g.reshape(bt, 1, V * H).contiguous(),
                    effective_g.reshape(bt, V * H, 1).contiguous()
                ).reshape(B, T)
                return token_grad_norm_sq
            
            if granularity == "sentence":
                return self._sparse_dot_product_sentence_level(sparse_g_dict, effective_g_dict, effective_g.dtype)
            elif granularity == "global":
                return self._sparse_dot_product_global_level(sparse_g_dict, effective_g_dict, effective_g.dtype)

    def _sparse_dot_product_sentence_level(self, dicts_per_sentence, effective_g_dicts_per_sentence, grad_dtype):
        """
        Compute the sparse dot product between g_sent and effective_g_sent,
        using the dictionary of sequences.
        """
        # Initialize a dictionary to store the dot products for each sentence   
        sentence_dot_products = []

        # Iterate over each sentence in the batch
        for i, (dict_i, effective_g_dict_i) in enumerate(zip(dicts_per_sentence, effective_g_dicts_per_sentence)):
            # Match keys in dict_i with keys in effective_g_dict_i
            matched_keys = dict_i.keys() & effective_g_dict_i.keys()
            if matched_keys:
                val = sum(torch.dot(effective_g_dict_i[k], dict_i[k]) for k in matched_keys)
            else:
                # both dicts empty or no overlap -> scalar 0 tensor
                val = torch.tensor(0.0, device=self.accelerator.device, dtype=grad_dtype)
            # Compute the dot product between the effective gradient and the dictionary
            sentence_dot_products.append(val)

        # Stack the sentence dot products into a 1D tensor
        sentence_dot_products = torch.stack(sentence_dot_products)

        return sentence_dot_products

    def _sparse_dot_product_global_level(self, g_global_dict, effective_g_global_dict, grad_dtype):
        """
        Compute the sparse dot product between g_global and effective_g_global,
        using the final global dictionary.
        """
        # Match keys in g_global with keys in effective_g_global
        matched_keys = g_global_dict.keys() & effective_g_global_dict.keys()
        if matched_keys:
            val = sum(torch.dot(g_global_dict[key], effective_g_global_dict[key]) for key in matched_keys)
        else:
            # both dicts empty or no overlap -> scalar 0 tensor
            val = torch.tensor(0.0, device=self.accelerator.device, dtype=grad_dtype)
        # Compute the dot product between the effective gradient and the dictionary
        return val

    def _dict_to_tensor(self, d):
        """
        Convert dict[int -> torch.Tensor(N,)] 
        into torch.Tensor(num_keys, N+1).
        """
        key_count_rows = []
        grad_rows = []
        
        for k, v in d.items():
            if isinstance(v, tuple):
                grad_rows.append(v[0])
                key_count_rows.append([k, v[1]])
            else:
                grad_rows.append(v)
                key_count_rows.append([k])
        grad_tensor = torch.stack(grad_rows)
        key_count_tensor = torch.tensor(key_count_rows, dtype=torch.long, device=grad_tensor.device)
        return grad_tensor, key_count_tensor
    
    
    def _list_of_dicts_to_tensor(self, list_of_dicts, max_num_tokens, hidden_dim, grad_dtype, pad_value=0.0, key_count_dim=2):
        """
        Convert a list of dict[int -> tensor(N,)] into a padded tensor of shape
        (batch_size, max_num_tokens, N+1).
        """
        batch_grad_tensors = []
        batch_key_count_tensors = []
        for d in list_of_dicts:
            # Convert dict to tensor
            if not d:
                # create grad tensor full padded with pad_value
                grad_tensor = torch.full((max_num_tokens, hidden_dim), pad_value, device=self.accelerator.device, dtype=grad_dtype)
                key_count_tensor = torch.full((max_num_tokens, key_count_dim), pad_value, device=self.accelerator.device, dtype=torch.long)
            else:
                grad_tensor, key_count_tensor = self._dict_to_tensor(d)  # (num_keys, N+1)
                num_keys, dim = grad_tensor.shape
            
                # Pad or truncate to max_num_tokens
                if num_keys < max_num_tokens:
                    pad_size = (0, 0, 0, max_num_tokens - num_keys)  # (left,right,top,bottom)
                    grad_tensor = torch.nn.functional.pad(grad_tensor, pad_size, value=pad_value)
                    key_count_tensor = torch.nn.functional.pad(key_count_tensor, pad_size, value=pad_value)
                elif num_keys > max_num_tokens:
                    grad_tensor = grad_tensor[:max_num_tokens]
                    key_count_tensor = key_count_tensor[:max_num_tokens]
            
            batch_grad_tensors.append(grad_tensor)
            batch_key_count_tensors.append(key_count_tensor)
        
        return torch.stack(batch_grad_tensors), torch.stack(batch_key_count_tensors)  # (batch_size, max_num_tokens, N+1)
