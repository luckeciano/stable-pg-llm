from collections import defaultdict
from typing import Any, Union
import torch
import torch.nn as nn
import wandb

from transformers import Trainer
from accelerate.utils import broadcast_object_list, gather, gather_object

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import (
    pad,
    selective_log_softmax,
)

from open_r1.modifiable_grpo_trainer import ModifiableGRPOTrainer


class GRPOEntropyTrainer(ModifiableGRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics["train_stats"] = defaultdict(list)
        self._metrics["eval_stats"] = defaultdict(list)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, return_hidden_states=False):
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
        logits = logits[:, -logits_to_keep:]
        log_probs = selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens
        if return_hidden_states:
            # hidden states are (num_layers, batch, sequence, hidden)
            embeddings = torch.stack([x for x in outputs.hidden_states])  # (num_layers, B, L, H)
            return log_probs, embeddings
        else:
            return log_probs

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

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        gathered_advantages = self._compute_advantages(rewards, mode)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        
        advantages = gathered_advantages[process_slice]

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = self._compute_final_loss(per_token_loss, completion_mask)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
    
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
            old_per_token_logps, old_last_token_embeddings = self._get_per_token_logps(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep, return_hidden_states=True
            )
            # else:
                # old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
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
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) 

        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        self._compute_and_log_stats(
            data=std_grouped_rewards,
            metric_name='grouped_std_rewards',
            mode=mode,
        )

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)    
        return advantages
    
    def _compute_final_loss(self, per_token_loss, completion_mask):
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        return loss
    
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

        # Log advantages
        self._compute_and_log_stats(
            data=advantages,
            metric_name='advantages',
            mode=mode,
        )

        # Log stats for each reward function
        for i, reward_func in enumerate(self.reward_funcs):
            self._compute_and_log_stats(
                data=rewards_per_func[:, i],
                metric_name=reward_func.__name__,
                mode=mode,
                groups={'correct': correct_responses, 'incorrect': ~correct_responses},
            )

        self._compute_and_log_stats(
            data=completion_length,
            metric_name='completion_length',
            mode=mode,
            groups={'correct': correct_responses, 'incorrect': ~correct_responses},
        )

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())

        # save logprobs to wandb
        if old_per_token_logps is not None:
            mean_logprobs = self.accelerator.gather_for_metrics(old_per_token_logps.mean(1)).float().mean().item()
            sum_logprobs = self.accelerator.gather_for_metrics(old_per_token_logps.sum(1)).float().mean().item()
            self._metrics[mode]["logprobs/mean"].append(mean_logprobs)
            self._metrics[mode]["logprobs/sum"].append(sum_logprobs)



        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                # if is_rich_available():
                    # print_prompt_completions_sample(
                    #     prompts_to_log,
                    #     completions_to_log,
                    #     rewards_to_log,
                    #     self.state.global_step,
                    # )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                        "correct": correct_responses.tolist(),
                    }

                    # Add individual reward function values
                    for i, reward_func in enumerate(self.reward_funcs):
                        if isinstance(reward_func, nn.Module):
                            reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                        else:
                            reward_func_name = reward_func.__name__
                        table[f"reward_{reward_func_name}"] = rewards_per_func[:, i].tolist()
                        
                    df = pd.DataFrame(table)

                    wandb.log({"completions": wandb.Table(dataframe=df)})
    
    def _compute_and_log_stats(self, data, metric_name, mode, groups=None, stats=None):
        """Compute and log statistics for the given data.
        
        Args:
            data (torch.Tensor): Tensor containing the data to analyze
            metric_name (str): Base name for the metric (e.g. 'completion_length')
            mode (str): Either "train" or "eval"
            groups (dict, optional): Dictionary mapping group names to boolean masks for data grouping
            stats (dict, optional): Dictionary mapping stat names to computation functions.
                                  Defaults to mean, max, min, p25, p75, median
        """
        # Default statistics if none provided
        if stats is None:
            stats = {
                'mean': lambda x: x.mean().item(),
                'max': lambda x: x.max().item(),
                'min': lambda x: x.min().item(),
                'p25': lambda x: x.quantile(0.25).item(),
                'p75': lambda x: x.quantile(0.75).item(),
                'median': lambda x: x.median().item()
            }
        
        # Compute and log overall statistics
        for stat_name, stat_func in stats.items():
            metric_key = f"{metric_name}/{stat_name}" if stat_name != 'mean' else metric_name
            self._metrics[mode][metric_key].append(stat_func(data))
        
        # If groups are provided, compute statistics for each group
        if groups is not None:
            for group_name, mask in groups.items():
                group_data = data[mask]
                if len(group_data) > 0:  # Only compute stats if group has data
                    for stat_name, stat_func in stats.items():
                        metric_key = f"{metric_name}/{group_name}/{stat_name}" if stat_name != 'mean' else f"{metric_name}/{group_name}"
                        self._metrics[mode][metric_key].append(stat_func(group_data))
