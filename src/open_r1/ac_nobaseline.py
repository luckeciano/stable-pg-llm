from open_r1.ac_trainer import ActorCriticTrainer

class ActorCriticNoBaselineTrainer(ActorCriticTrainer):
    """
    "Actor-Critic" No Baseline trainer class: Just Policy Gradient without Critic, but uniformly from prompt distribution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_loss_weight = 0.0

    def _compute_advantages_from_critic(self, rewards, value_scalar): 
        return rewards
    