from open_r1.ac_trainer import ActorCriticTrainer
import numpy as np

class ActorCriticNoBaselineTrainer(ActorCriticTrainer):
    """
    "Actor-Critic" No Baseline trainer class: Just Policy Gradient without Critic, but uniformly from prompt distribution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_loss_weight = 0.0
        self.normalize_advantages = kwargs['args'].normalize_advantages
        self.advantage_target_std = kwargs['args'].advantage_target_std
        self.anneal_advantage_std = kwargs['args'].anneal_advantage_std

    def _compute_advantages_from_critic(self, rewards, value_scalar):
        if self.anneal_advantage_std:
            self.advantage_target_std = self._compute_advantage_target_std_schedule()

        if self.normalize_advantages:
            gathered_rewards = self.accelerator.gather(rewards)
            advantages = (rewards - gathered_rewards.mean()) / (gathered_rewards.std() + 1e-8)
            advantages = advantages * self.advantage_target_std
        else:
            advantages = rewards

        return advantages    

    def _compute_advantage_target_std_schedule(self, start_var=1.0, end_var=0.4):
        # Increase the decay rate so the curve drops faster
        step = self.state.global_step
        total_steps = self.state.max_steps
        decay_rate = np.log(start_var / end_var) / (total_steps / 3)  # decay in first third
        steps = np.arange(total_steps + 1)
        variances = start_var * np.exp(-decay_rate * steps)
        stds = np.sqrt(variances)
        return stds[step]
    