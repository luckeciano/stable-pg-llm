from __future__ import annotations
from typing import Dict, Hashable, List
from open_r1 import sparse_grad_ops
from .sparse_sgd_model import SparseSGDModel
import torch

class SparseAdamModel(SparseSGDModel):
    """
    Keeps Adam moments per logit key.

    Typical loop per step:
        adam_grads = opt.compute_effective_gradients(grads)  # read-only
        opt.update_effective_gradient_moments(grads)         # in-place state update
        # then update your weights with adam_grads
    """
    def __init__(self, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-8):
        self.beta1 = float(adam_beta1)
        self.beta2 = float(adam_beta2)
        self.eps   = float(adam_epsilon)

        self.m: Dict[Hashable, torch.Tensor] = {}
        self.v: Dict[Hashable, torch.Tensor] = {}
        self.t: int                          = 0

    @torch.no_grad()
    def batch_compute_effective_token_gradients(self, per_token_grads: torch.Tensor, token_ids: torch.Tensor, device: torch.device) -> Dict[Hashable, torch.Tensor]:
        """
        Compute Adam-scaled gradients for this step *without* mutating internal state.
        Uses m_{t+1}, v_{t+1}, t+1 formed from current state and provided grads.
        Everything in-place to save memory.
        """
        per_token_m = sparse_grad_ops.densify_gradient_dict(self.m, token_ids, per_token_grads.dtype, device)
        per_token_v = sparse_grad_ops.densify_gradient_dict(self.v, token_ids, per_token_grads.dtype, device)
        t_next = self.t + 1

        if per_token_m is None or per_token_v is None:
            return per_token_grads

        # Pre-compute bias correction factors to avoid repeated computation
        beta1_correction = 1.0 - self.beta1 ** t_next
        beta2_correction = 1.0 - self.beta2 ** t_next

        # prospective moments - all in-place operations
        per_token_m.mul_(self.beta1).add_(per_token_grads, alpha=(1.0 - self.beta1))
        per_token_v.mul_(self.beta2).addcmul_(per_token_grads, per_token_grads, value=(1.0 - self.beta2))

        # bias correction @ t_next - in-place
        per_token_m.div_(beta1_correction)
        per_token_v.div_(beta2_correction)

        # Compute final Adam gradient in-place to avoid extra tensor allocation
        per_token_v.sqrt_().add_(self.eps)
        per_token_m.div_(per_token_v)
        
        # per_token_m now contains the final Adam gradient
        # Clean up v tensor immediately
        del per_token_v
        torch.cuda.empty_cache()
        return per_token_m

    @torch.no_grad()
    def compute_effective_token_gradients(self, per_token_grads_batch: torch.Tensor, token_ids_batch: torch.Tensor, device: torch.device, batch_size: int = 12) -> torch.Tensor:
        """
        Compute Adam-scaled gradients for a batch of sequences *without* mutating internal state.
        Uses m_{t+1}, v_{t+1}, t+1 formed from current state and provided grads.
        Processes sequences in batches to manage memory usage.
        
        Args:
            per_token_grads_batch: Tensor of shape [total_tokens, ...] containing gradients for all tokens
            token_ids_batch: Tensor of shape [total_tokens] containing token IDs for all tokens
            device: Device to perform computations on
            batch_size: Number of tokens to process in each batch
            
        Returns:
            Tensor of shape [total_tokens, ...] containing Adam-scaled gradients
        """
        total_tokens = per_token_grads_batch.shape[0]
        result_list = []
        
        # Process in batches
        for start_idx in range(0, total_tokens, batch_size):
            end_idx = min(start_idx + batch_size, total_tokens)
            
            # Extract batch
            per_token_grads = per_token_grads_batch[start_idx:end_idx]
            token_ids = token_ids_batch[start_idx:end_idx]
            
            # Use the existing logic for this batch
            batch_result = self.batch_compute_effective_token_gradients(per_token_grads, token_ids, device)
            
            # Add to result list
            result_list.append(batch_result)
            
            # Clean up batch result to free memory
            del batch_result
        
        # Concatenate all batch results
        result_grads = torch.cat(result_list, dim=0)
        
        # Clean up the list
        del result_list
        torch.cuda.empty_cache()
        return result_grads

    @torch.no_grad()
    def compute_effective_gradients_from_dict(self, dict: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """
        For global gradients, we compute Adam-scaled gradients using parallel tensor operations.
        This is much faster than the original dict iteration approach.
        
        For keys not in self.m and self.v, we treat the moments as zeros (m=0, v=0),
        meaning all keys go through Adam processing.
        """
        if not self.m or not self.v or not dict:
            return dict
        
        # Pre-compute bias correction factors once
        t_next = self.t + 1
        beta1_correction = 1.0 - self.beta1 ** t_next
        beta2_correction = 1.0 - self.beta2 ** t_next
        
        # Get device and dtype from the first gradient tensor
        first_grad = next(iter(dict.values()))
        device = first_grad.device
        dtype = first_grad.dtype
        
        # Convert dict to parallel tensors
        keys = list(dict.keys())
        grads_tensor = torch.stack([dict[key] for key in keys]).to(device, dtype=dtype)  # [num_keys, ...]
        
        # Get corresponding m and v tensors, treating missing keys as zeros
        m_tensor = torch.zeros_like(grads_tensor)
        v_tensor = torch.zeros_like(grads_tensor)
        
        for i, key in enumerate(keys):
            if key in self.m and key in self.v:
                m_tensor[i] = self.m[key]
                v_tensor[i] = self.v[key]
            # For missing keys, m_tensor[i] and v_tensor[i] remain zeros
        
        # Perform Adam updates in parallel for ALL keys (including those with zero moments)
        # Update moments in-place for all keys
        m_next = m_tensor.clone()
        v_next = v_tensor.clone()
        
        # Adam moment updates in parallel
        m_next.mul_(self.beta1).add_(grads_tensor, alpha=(1.0 - self.beta1))
        v_next.mul_(self.beta2).addcmul_(grads_tensor, grads_tensor, value=(1.0 - self.beta2))
        
        # Apply bias correction in parallel
        m_hat = m_next.div_(beta1_correction)
        v_hat = v_next.div_(beta2_correction)
        
        # Final Adam gradient computation in parallel
        adam_grads = m_hat.div_(v_hat.sqrt_().add_(self.eps))
        
        # Update the result tensor
        grads_tensor = adam_grads
        
        # Clean up temporary tensors
        del m_next, v_next, m_hat, v_hat, adam_grads
        
        # Convert back to dict format
        adam_grad = {key: grads_tensor[i] for i, key in enumerate(keys)}
        
        # Clean up
        del grads_tensor, m_tensor, v_tensor
        torch.cuda.empty_cache()
        return adam_grad
    
    @torch.no_grad()
    def compute_effective_global_gradients(self, global_dict: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """
        For global gradients, we iterate over the tokens in global_dict and compute the effective gradient for each token.
        """
        return self.compute_effective_gradients_from_dict(global_dict)

    @torch.no_grad()
    def compute_effective_sentence_gradients(self, sentence_dicts: List[Dict[Hashable, torch.Tensor]]) -> List[Dict[Hashable, torch.Tensor]]:
        """
        For sentence gradients, we iterate over the sentence dicts.
        """
        adam_grad = []
        for sentence_dict in sentence_dicts:
            result = self.compute_effective_gradients_from_dict(sentence_dict)
            adam_grad.append(result)
        
        torch.cuda.empty_cache()
        return adam_grad

    @torch.no_grad()
    def update_effective_gradient_moments(self, grads_by_logit: Dict[Hashable, torch.Tensor]) -> None:
        """
        Update (m, v) and increment t in-place using the provided grads.
        """
        self.t += 1
        for key, g in grads_by_logit.items():
            self._ensure_state(key, g)

            m = self.m[key]
            v = self.v[key]

            # m_t <- beta1*m_{t-1} + (1-beta1)*g_t
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            # v_t <- beta2*v_{t-1} + (1-beta2)*g_t^2
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

    def _ensure_state(self, key: Hashable, g: torch.Tensor) -> None:
        if key not in self.m:
            self.m[key] = torch.zeros_like(g, memory_format=torch.preserve_format)
            self.v[key] = torch.zeros_like(g, memory_format=torch.preserve_format)
