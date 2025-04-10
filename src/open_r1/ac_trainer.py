import torch
import torch.nn.functional as F
from typing import Any, Union
from accelerate.utils import gather
from open_r1.grpo_entropy_trainer import GRPOEntropyTrainer
from copy import deepcopy

class ActorCriticTrainer(GRPOEntropyTrainer):
    """
    Actor-Critic trainer class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_tokens = [f"<VF_{i}>" for i in range(10)]
        self.processing_class.add_tokens(self.target_tokens)

    
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
        values = self._compute_values(inputs, device)

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

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        gathered_advantages = self._compute_advantages(rewards, mode)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        self._log_stats(
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

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    # Format into conversation
    def _make_critic_inputs(self, inputs):
        critic_prompt = "<VALUE_TASK>"
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
        target_log_probs = self._compute_next_token_logits(prompt_ids, prompt_mask, device)

        # Select the target value token
        values = target_log_probs.argmax(dim=-1)

        return values, target_log_probs
    
    def _compute_next_token_logits(self, input_ids, attention_mask, device):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
        # Get indices of the last non-padding tokens in each sequence
        last_token_indices = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last token index
        
        # Gather logits for the last tokens of each sequence
        batch_indices = torch.arange(logits.size(0), device=device)
        last_token_logits = logits[batch_indices, last_token_indices, :]

        log_probs = F.log_softmax(last_token_logits, dim=-1)  # Convert logits to log probabilities
    
        # Convert target tokens to their corresponding IDs
        target_ids = torch.tensor([self.processing_class.convert_tokens_to_ids(token) for token in self.target_tokens], device=device)
        
        # Gather the log probabilities of the target tokens in a vectorized way
        target_log_probs = log_probs[:, target_ids]
        
        return target_log_probs
    
    def _compute_value_loss(self, log_probs, target_ids):
        device = log_probs.device
        loss = F.nll_loss(log_probs, target_ids)
        return loss

    