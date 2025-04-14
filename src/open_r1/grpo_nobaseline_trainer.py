from open_r1.grpo_entropy_trainer import GRPOEntropyTrainer

class GRPONoBaselineTrainer(GRPOEntropyTrainer):
    """
    GRPO No Baseline trainer class.
    """
    
    def _compute_advantages(self, rewards, mean_grouped_rewards, std_grouped_rewards): 
        return rewards
    