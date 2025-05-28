import torch
import torch.nn.functional as F
from typing import Any, Union
from accelerate.utils import gather
from trl.models import unwrap_model_for_generation
from open_r1.grpo_entropy_trainer import GRPOEntropyTrainer
from copy import deepcopy

class ActorCriticTrainer(GRPOEntropyTrainer):
    """
    Actor-Critic trainer class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_value_tokens = kwargs['args'].num_value_tokens
        self.value_type = kwargs['args'].value_type
        self.value_inference_strategy = kwargs['args'].value_inference_strategy
        self.value_loss_weight = kwargs['args'].value_loss_weight
        self.normalize_advantages = kwargs['args'].normalize_advantages
        self.advantage_target_std = kwargs['args'].advantage_target_std
        self.reward_intervals = kwargs['args'].reward_intervals
        self.value_loss = kwargs['args'].value_loss
        
        if self.num_value_tokens % 2 == 0:
                raise ValueError(f"num_value_tokens must be odd for digit value type.")
        
        if self.value_type == "digit":
            if self.num_value_tokens >= 10 and self.num_value_tokens > 1:
                raise ValueError(f"num_value_tokens must be less than 10 and greater than 1 for digit value type.")  
            self.target_tokens = [f"{i}" for i in range(1, self.num_value_tokens + 1)]
            min_reward, max_reward, interval_size, possible_values = self._compute_reward_interval()
            representation = "".join([f"Token {token} represents {value}\n" for token, value in zip(self.target_tokens, possible_values)])
            self.critic_prompt = f"Analyze the following problem carefully and provide a value function prediction between 1 and {self.num_value_tokens}. Here is the value each token represents: {representation}."
        elif self.value_type == "token":
            self.target_tokens = [f"<VF_{i}>" for i in range(1, self.num_value_tokens + 1)]
            self.processing_class.add_tokens(self.target_tokens)
            # Resize token embeddings
            # self.model.resize_token_embeddings(len(self.processing_class))
            representation = "".join([f"Token {token} represents {value}\n" for token, value in zip(self.target_tokens, possible_values)])
            self.critic_prompt = f"Analyze the following problem carefully and provide a value function prediction between <VF_1> and <VF_{self.num_value_tokens}>. Here is the value each token represents: {representation}."
        else:
            raise ValueError(f"Invalid value type: {self.value_type}")
        
        self.target_ids = torch.tensor([self.processing_class.convert_tokens_to_ids(token) for token in self.target_tokens], device=self.accelerator.device)

    
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        # Compute prompt IDs and mask
        prompt_ids, prompt_mask, prompts_text = self._compute_prompt_ids_and_mask(inputs)

        # Generate completions
        prompt_completion_ids, completion_ids = self._generate_completions(prompt_ids, prompt_mask, prompts_text, prompts, device)

        # Generate critic completions
        values, values_log_probs, pred_value_log_probs = self._compute_values(inputs, device)

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
        
        # Find index of accuracy reward function
        accuracy_idx = [i for i, func in enumerate(self.reward_funcs) if func.__name__ == "accuracy_reward"][0]
        accuracy_reward = rewards_per_func[:, accuracy_idx]
                
        # Log the metrics - mode
        mode = "eval" if self.control.should_evaluate else "train"

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        advantages = self._compute_advantages_from_critic(rewards, values)

        table = self._log_stats(
            stats_dict={
                'rewards_per_func': self.accelerator.gather(rewards_per_func),
                'rewards': self.accelerator.gather(rewards),
                'advantages': self.accelerator.gather(advantages),
                'pred_value_log_probs': self.accelerator.gather(pred_value_log_probs),
                'values': self.accelerator.gather(values.float()),
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
            "values_log_probs": values_log_probs,
            "rewards": rewards,
            "accuracy_reward": self.accelerator.gather(accuracy_reward),
        }

        if table is not None:
            final_dict['table'] = table

        return final_dict

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "eval" if self.control.should_evaluate else "train"
        policy_loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        per_token_logps, entropy, _, _, _ = self._get_and_smooth_token_logps(self.model, input_ids, attention_mask, logits_to_keep, mode, return_entropy=self.entropy_estimator == "softmax")

        rewards = inputs["rewards"]

        if self.entropy_alpha > 0.0:
            estimator = per_token_logps if self.entropy_estimator == "logprobs" else entropy
            avg_entropy = (estimator * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            rewards = rewards + self.entropy_alpha * avg_entropy

        values_log_probs = inputs["values_log_probs"]

        # Recompute value predictions with gradients enabled
        if self.value_loss == "hard_label":
            target_value = self._compute_target_value(rewards)
            value_loss = self._compute_value_loss(values_log_probs, target_value)
        elif self.value_loss == "soft_label":
            value_loss = self._compute_gaussian_soft_value_loss(values_log_probs, rewards)
        else:
            raise ValueError(f"Invalid value loss: {self.value_loss}")

        # Log the metrics
        self._metrics[mode]["value_loss"].append(self.accelerator.gather_for_metrics(value_loss).mean().item())
        
        total_loss = policy_loss + self.value_loss_weight * value_loss
        return total_loss
    
    # Format into conversation
    def _make_critic_inputs(self, inputs):
        critic_inputs = deepcopy(inputs)
        for input in critic_inputs:
            prompt = []
            prompt.append({"role": "system", "content": self.critic_prompt})
            prompt.append({"role": "user", "content": input["problem"] + "\n\nVALUE: "})
            input["prompt"] = prompt
        return critic_inputs
    
    def _compute_values(self, inputs, device):
        prompts = [x["prompt"] for x in inputs]

        # Compute critic inputs
        critic_inputs = self._make_critic_inputs(inputs)

        # Compute prompt IDs and mask
        prompt_ids, prompt_mask, prompts_text = self._compute_prompt_ids_and_mask(critic_inputs)
        
        # Generate critic completions
        values_log_probs = self._compute_next_token_logits(prompt_ids, prompt_mask, device)

        # Select the target value token
        values, pred_value_log_probs = self._infer_values(values_log_probs, self.value_inference_strategy)

        return values, values_log_probs, pred_value_log_probs
    
    def _compute_next_token_logits(self, input_ids, attention_mask, device):
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            outputs = unwrapped_model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
        # Get indices of the last non-padding tokens in each sequence
        last_token_indices = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last token index
        
        # Gather logits for the last tokens of each sequence
        batch_indices = torch.arange(logits.size(0), device=device)
        last_token_logits = logits[batch_indices, last_token_indices, :]
        target_logits = last_token_logits[:, self.target_ids]

        target_log_probs = F.log_softmax(target_logits, dim=-1)  # Convert logits to log probabilities
        
        return target_log_probs
    
    def _infer_values(self, log_probs, type="marginalization"):
        min_reward, max_reward, interval_size, possible_values = self._compute_reward_interval()
        
        mode_values = log_probs.argmax(dim=-1)
        pred_value_log_probs = torch.gather(log_probs, dim=1, index=mode_values.unsqueeze(-1)).squeeze(-1)
        if type == "mode":
            values = min_reward + mode_values * interval_size
            return values, pred_value_log_probs
        
        elif type == "marginalization":
            # Compute marginalization
            values_per_prob = torch.exp(log_probs) * possible_values
            inferred_values = torch.sum(values_per_prob, dim=-1)
            return inferred_values, pred_value_log_probs
        else:
            raise ValueError(f"Invalid value inference strategy: {type}")

    
    def _compute_target_value(self, rewards):
        min_reward, max_reward, interval_size, possible_values = self._compute_reward_interval()
        # Assign rewards to the closest target value
        target_value = torch.argmin(torch.abs(rewards.unsqueeze(-1) - possible_values), dim=-1)

        return target_value
    
    def _compute_value_scalar(self, value):
        min_reward, max_reward, interval_size, possible_values = self._compute_reward_interval()

        # Convert value to scalar
        return min_reward + value * interval_size
    
    def _compute_value_loss(self, log_probs, target_ids):
        loss = F.nll_loss(log_probs, target_ids)
        return loss
    
    def _compute_soft_value_loss(self, log_probs, target_ids, smoothing=0.1):
        n_classes = log_probs.size(-1)
        
        # Create one-hot targets
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, target_ids.unsqueeze(1), 1.0 - smoothing)

        loss = -torch.sum(true_dist * log_probs, dim=1).mean()
        return loss
    
    def _compute_gaussian_soft_value_loss(self, log_probs, target_scalars):
        """
        log_probs: [batch_size, num_classes] - log-probabilities (log_softmax applied)
        target_scalars: [batch_size] - ground truth scalar values
        sigma: standard deviation for Gaussian smoothing
        """
        batch_size, num_classes = log_probs.shape
        device = log_probs.device
        min_reward, max_reward, bin_width, target_values = self._compute_reward_interval()

        # Expand to shape [batch_size, num_classes] to match targets
        target_values = target_values.unsqueeze(0).expand(batch_size, -1)  # [B, C]

        # Expand target ids to match shape
        target_scalars = target_scalars.unsqueeze(1).float()  # [B, 1]

        # Compute squared distance from the target class
        squared_dist = (target_values - target_scalars) ** 2  # [B, C]

        
        sigma = 0.33 * bin_width
        # Compute unnormalized Gaussian
        unnormalized = torch.exp(-squared_dist / (2 * sigma ** 2))  # [B, C]

        # Normalize to get valid probability distribution
        smoothed_targets = unnormalized / unnormalized.sum(dim=1, keepdim=True)  # [B, C]

        # Cross-entropy between smoothed target and predicted log_probs
        loss = -torch.sum(smoothed_targets * log_probs, dim=1).mean()

        return loss

    def _compute_reward_interval(self):
        # We assume each reward function is bounded
        min_reward = 0.0
        max_reward = 0.0
        for interval, weight in zip(self.reward_intervals, self.reward_weights):
            min_reward += interval[0] * weight
            max_reward += interval[1] * weight
        
        # if self.entropy_alpha > 0.0:
        #     # min reward does not change as log1 = 0
        #     max_reward += 1.0 # this limits the expressivity of the value function, but in practice it is fine

        interval_size = (max_reward - min_reward) / (len(self.target_tokens) - 1)
        possible_values = min_reward + torch.arange(len(self.target_tokens), device=self.accelerator.device) * interval_size

        return min_reward, max_reward, interval_size, possible_values
    
    def _compute_advantages_from_critic(self, rewards, value_scalar):
        advantages = rewards - value_scalar.detach()

        if self.normalize_advantages:
            gathered_advantages = self.accelerator.gather(advantages)
            advantages = (advantages - gathered_advantages.mean()) / (gathered_advantages.std() + 1e-8)

        advantages = advantages * self.advantage_target_std

        return advantages
    
    def _log_stats(self, stats_dict, mode):
        table = super()._log_stats(stats_dict, mode)

        values = stats_dict['values'].float()
        pred_value_log_probs = stats_dict['pred_value_log_probs'].float()

        self._accumulate_stats(
            data=values,
            metric_name='values',
            mode=mode,
        )

        self._accumulate_stats(
            data=pred_value_log_probs,
            metric_name='pred_value_log_probs',
            mode=mode,
        )

        return table
        

        
        
        
        

    