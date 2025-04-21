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
        # self.target_tokens = [f"<VF_{i}>" for i in range(10)]
        # self.processing_class.add_tokens(self.target_tokens)
        self.target_tokens = [f"{i}" for i in range(1, 6)]

    
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

        target_value = self._compute_target_value(rewards, self.reward_weights)

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
            "target_value": target_value,
            "accuracy_reward": self.accelerator.gather(accuracy_reward),
        }

        if table is not None:
            final_dict['table'] = table

        return final_dict

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        policy_loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        values_log_probs = inputs["values_log_probs"]
        target_value = inputs["target_value"]

        # Recompute value predictions with gradients enabled
        value_loss = self._compute_value_loss(values_log_probs, target_value)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["value_loss"].append(self.accelerator.gather_for_metrics(value_loss).mean().item())
        
        total_loss = policy_loss + 0.01 * value_loss
        return total_loss
    
    def _compute_value_loss(self, log_probs, target_ids):
        loss = F.nll_loss(log_probs, target_ids)
        return loss
    
    # Format into conversation
    def _make_critic_inputs(self, inputs):
        critic_prompt = "Analyze the following problem carefully and provide a value function prediction between 1 and 5. 1 is minimum value, 5 is maximum value. 3 is the midpoint and means zero reward."
        critic_inputs = deepcopy(inputs)
        for input in critic_inputs:
            prompt = []
            prompt.append({"role": "system", "content": critic_prompt})
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
        values, pred_value_log_probs = self._infer_values(values_log_probs)

        return values, values_log_probs, pred_value_log_probs
    
    def _compute_next_token_logits(self, input_ids, attention_mask, device):
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            outputs = unwrapped_model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
        # Get indices of the last non-padding tokens in each sequence
        last_token_indices = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last token index

        target_ids = torch.tensor([self.processing_class.convert_tokens_to_ids(token) for token in self.target_tokens], device=device)
        
        # Gather logits for the last tokens of each sequence
        batch_indices = torch.arange(logits.size(0), device=device)
        last_token_logits = logits[batch_indices, last_token_indices, :]
        target_logits = last_token_logits[:, target_ids]

        target_log_probs = F.log_softmax(target_logits, dim=-1)  # Convert logits to log probabilities
        
        return target_log_probs
    
    def _infer_values(self, log_probs, type="marginalization"):
        min_reward, max_reward = self._compute_reward_interval(self.reward_weights)
        # Split interval [min_reward, max_reward] into len(self.target_tokens) intervals
        interval_size = (max_reward - min_reward) / (len(self.target_tokens) - 1)
        possible_values = min_reward + torch.arange(len(self.target_tokens), device=log_probs.device) * interval_size
        
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

    
    def _compute_target_value(self, rewards, reward_weights):
        min_reward, max_reward = self._compute_reward_interval(reward_weights)
        
        # Split interval [min_reward, max_reward] into len(self.target_tokens) intervals
        interval_size = (max_reward - min_reward) / (len(self.target_tokens) - 1)
        target_values = torch.arange(min_reward, max_reward, interval_size, device=rewards.device)

        # Assign rewards to the closest target value
        target_value = torch.argmin(torch.abs(rewards.unsqueeze(-1) - target_values), dim=-1)

        return target_value
    
    def _compute_value_scalar(self, value, reward_weights):
        min_reward, max_reward = self._compute_reward_interval(reward_weights)

        # Split interval (min_reward, max_reward) into len(self.target_tokens) intervals
        interval_size = (max_reward - min_reward) / (len(self.target_tokens) - 1)

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
    
    #TODO
    def _compute_gaussian_soft_value_loss(self, log_probs, target_ids, smoothing=0.1):
        """
        log_probs: [batch_size, num_classes] - log-probabilities (log_softmax applied)
        target_ids: [batch_size] - ground truth class indices
        sigma: standard deviation for Gaussian smoothing
        """
        batch_size, num_classes = log_probs.shape
        device = log_probs.device

        # Create class index vector: [0, 1, ..., num_classes-1]
        class_range = torch.arange(num_classes, device=device).float()  # [num_classes]

        # Expand to shape [batch_size, num_classes] to match targets
        class_range = class_range.unsqueeze(0).expand(batch_size, -1)  # [B, C]

        # Expand target ids to match shape
        target_ids = target_ids.unsqueeze(1).float()  # [B, 1]

        # Compute squared distance from the target class
        squared_dist = (class_range - target_ids) ** 2  # [B, C]

        # Compute unnormalized Gaussian
        unnormalized = torch.exp(-squared_dist / (2 * sigma ** 2))  # [B, C]

        # Normalize to get valid probability distribution
        smoothed_targets = unnormalized / unnormalized.sum(dim=1, keepdim=True)  # [B, C]

        # Cross-entropy between smoothed target and predicted log_probs
        loss = -torch.sum(smoothed_targets * log_probs, dim=1).mean()

        return loss

    def _compute_reward_interval(self, reward_weights):
        # We assume each reward function is bounded between -1 and 1
        min_reward = -1.0 * reward_weights.sum()
        max_reward = 1.0 * reward_weights.sum()
        return min_reward, max_reward
    
    def _compute_advantages_from_critic(self, rewards, value_scalar):
        advantages = rewards - value_scalar.detach()
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
        

        
        
        
        

    