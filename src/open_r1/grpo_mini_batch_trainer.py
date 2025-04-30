from open_r1.grpo_entropy_trainer import GRPOEntropyTrainer
from transformers.utils import logging
import torch

logger = logging.get_logger(__name__)   

class GRPOMiniBatchTrainer(GRPOEntropyTrainer):
    """
    GRPO trainer class.
    """
    
    def _compute_advantages(self, rewards, mode):
        if mode == "eval" or self.num_generations == 1:
            logger.warning("Evaluation or num_generations = 1, returning rewards as advantages")
            return rewards
        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) 

        # Create a mask that is self.num_generations for the first generation and 0 for the rest
        mask = torch.zeros(self.num_generations, device=self.accelerator.device)
        mask[0] = self.num_generations

        # Repeat the mask vector until it becomes a tensor of the same size as rewards
        mask = mask.repeat(len(rewards) // self.num_generations)

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
        advantages = advantages * mask # Only the first generation is used for training
        return advantages
    