from __future__ import annotations
from typing import Dict, Hashable, List
import torch

class SparseSGDModel:
    """
    Keeps SGD moments per logit key.

    Essentially, this model is a no-op.
    """
    @torch.no_grad()
    def compute_effective_token_gradients(self, per_token_grads: torch.Tensor, token_ids: torch.Tensor, device: torch.device) -> Dict[Hashable, torch.Tensor]:
        return per_token_grads

    @torch.no_grad()
    def compute_effective_gradients_from_dict(self, dict: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        return dict
    
    @torch.no_grad()
    def compute_effective_global_gradients(self, global_dict: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        return self.compute_effective_gradients_from_dict(global_dict)

    @torch.no_grad()
    def compute_effective_sentence_gradients(self, sentence_dicts: List[Dict[Hashable, torch.Tensor]]) -> List[Dict[Hashable, torch.Tensor]]:
        return sentence_dicts

    @torch.no_grad()
    def update_effective_gradient_moments(self, grads_by_logit: Dict[Hashable, torch.Tensor]) -> None:
        pass
