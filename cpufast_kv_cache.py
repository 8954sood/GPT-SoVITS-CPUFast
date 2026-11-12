"""Allocate head-major decode capacity while copying only valid prompt KV."""
import torch


@torch.jit.script
def head_major_prefix_cache(prefix: torch.Tensor, heads: int, decode_steps: int):
    if prefix.dim() != 3 or heads <= 0 or prefix.shape[2] % heads or decode_steps < 0:
        raise ValueError("Invalid prompt cache geometry")
    batch, length, width = prefix.shape
    dim = width // heads
    result = prefix.new_empty((batch, heads, length + decode_steps, dim))
    result[:, :, :length, :].copy_(prefix.reshape(batch, length, heads, dim).transpose(1, 2))
    return result
