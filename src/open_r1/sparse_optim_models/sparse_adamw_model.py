from __future__ import annotations
from typing import Dict, Hashable, List
from open_r1 import sparse_grad_ops
from .sparse_adam_model import SparseAdamModel
import torch

class SparseAdamWModel(SparseAdamModel):
    """
    Keeps AdamW moments per logit key.

    Typical loop per step:
        adam_grads = opt.compute_effective_gradients(grads)  # read-only
        opt.update_effective_gradient_moments(grads)         # in-place state update
        # then update your weights with adam_grads
    """
    def __init__(self, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(adam_beta1, adam_beta2, adam_epsilon)
        self.weight_decay = weight_decay

    @torch.no_grad()
    def compute_effective_token_gradients(self, per_token_grads: torch.Tensor, token_ids: torch.Tensor, device: torch.device) -> Dict[Hashable, torch.Tensor]:
        raise NotImplementedError("compute_effective_token_gradients is not implemented for SparseAdamWModel")

    @torch.no_grad()
    def compute_effective_gradients_from_dict(self, dict: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        raise NotImplementedError("compute_effective_gradients_from_dict is not implemented for SparseAdamWModel")

    @torch.no_grad()
    def update_effective_gradient_moments(self, grads_by_logit: Dict[Hashable, torch.Tensor]) -> None:
        raise NotImplementedError("update_effective_gradient_moments is not implemented for SparseAdamWModel")
