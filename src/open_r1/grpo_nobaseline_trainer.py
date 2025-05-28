from open_r1.grpo_entropy_trainer import GRPOEntropyTrainer

class GRPONoBaselineTrainer(GRPOEntropyTrainer):
    """
    GRPO No Baseline trainer class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalize_advantages = kwargs['args'].normalize_advantages
    
    def _compute_advantages(self, rewards, mode):
        # No need to gather here as it is already done in the outer function
        if self.normalize_advantages:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards

        return advantages
    