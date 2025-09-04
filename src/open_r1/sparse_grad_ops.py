import torch

def densify_gradient_dict(grad_dict, token_ids, grad_dtype, device):
    if not grad_dict:
        return None
    
    batch_size, seq_len, topk_size = token_ids.shape
    # 0) Flatten token_ids for indexing
    token_ids_flat_global = token_ids.flatten()  # [B*T*V]

    # 1) Build tensors of keys and grads
    keys = torch.tensor(list(grad_dict.keys()), device=device, dtype=torch.long)   # [U]
    grads = torch.stack(list(grad_dict.values())).to(device, dtype=grad_dtype)     # [U, H]

    # 2) Sort keys once, and align grads to that order
    keys_sorted, order = torch.sort(keys)                       # [U]
    grads_sorted = grads.index_select(0, order)                 # [U, H]

    # 3) Map each token id to its index in keys_sorted via searchsorted (O(N log U))
    idx = torch.searchsorted(keys_sorted, token_ids_flat_global)  # [N]
    # (If some token_ids might be missing, guard it)
    valid = (idx < keys_sorted.numel()) & (keys_sorted[idx] == token_ids_flat_global)
    # 4) Gather in one shot
    grad_tensor_global_flat = torch.zeros(
        (token_ids_flat_global.numel(), grads_sorted.size(1)),
        dtype=grad_dtype, device=device
    )
    grad_tensor_global_flat[valid] = grads_sorted.index_select(0, idx[valid])

    global_gradient = grad_tensor_global_flat.view(batch_size, seq_len, topk_size, -1)
    
    del grad_tensor_global_flat, token_ids_flat_global, keys, grads, keys_sorted, order, idx, valid, grads_sorted
    torch.cuda.empty_cache()
    return global_gradient