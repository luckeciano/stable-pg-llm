from open_r1.grpo_entropy_trainer import GRPOEntropyTrainer

class DrGRPOTrainer(GRPOEntropyTrainer):
    """
    DrGRPO trainer class.
    """
    def _compute_advantages(self, rewards, mode): 
        if mode == "eval" or self.num_generations == 1:
            return rewards
        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        return advantages

    def _compute_final_loss(self, per_token_loss, completion_mask, updated_mask, masks, curvature_estimator, completion_ids):
        mode = "eval" if self.control.should_evaluate else "train"
        if self.curvature_masking:
            masks_hessian = masks['hessian']
            masks_fisher = masks['fisher']
            self._log_masking_stats(masks_hessian, masks_fisher, completion_mask, completion_ids, mode)
            
            # Final mask
            completion_mask = completion_mask * updated_mask


        # --- Sentence-level loss calculation ---
        per_sentence_loss = (per_token_loss * completion_mask).sum(dim=1) / self.max_completion_length

        global_loss = per_sentence_loss.mean()

        gather_loss_1 = self.accelerator.gather_for_metrics(per_sentence_loss)
        
        self._accumulate_stats(
            data=gather_loss_1,
            metric_name='policy_loss',
            mode=mode,
        )

        return global_loss
    