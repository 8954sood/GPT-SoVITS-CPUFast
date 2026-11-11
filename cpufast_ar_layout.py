"""Experimental FP32 row-major GEMV caches for the pinned CPUFast transformer.

Import only after CPUFast's GPT_SoVITS directory is on sys.path. No checkpoint
tensor is cast, quantized, pruned, or replaced. These caches are for immutable,
loaded evaluation models; rebuild after loading another checkpoint.
"""

import torch
import torch.nn.functional as F
from AR.models.t2s_model import T2SBlock


@torch.jit.script
class RowBlock:
    def __init__(self, block: T2SBlock):
        self.block = block
        self.qkv = block.qkv_w_t.transpose(0, 1).contiguous()
        self.out = block.out_w_t.transpose(0, 1).contiguous()
        self.ff1 = block.mlp.w1_t.transpose(0, 1).contiguous()
        self.ff2 = block.mlp.w2_t.transpose(0, 1).contiguous()

    def process_prompt(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        max_decode_steps: int,
        padding: torch.Tensor | None,
        torch_sdpa: bool,
    ):
        return self.block.process_prompt(x, mask, max_decode_steps, padding, torch_sdpa)

    def decode_next_token(
        self,
        x: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cache_len: int,
        mask: torch.Tensor = None,
        torch_sdpa: bool = True,
    ):
        if x.shape[0] * x.shape[1] != 1 or not torch_sdpa:
            return self.block.decode_next_token(
                x, keys, values, cache_len, mask, torch_sdpa
            )
        b = self.block
        q, k, v = F.linear(x, self.qkv, b.qkv_b).chunk(3, dim=-1)
        next_len = cache_len + k.shape[1]
        keys[:, cache_len:next_len] = k
        values[:, cache_len:next_len] = v
        q = q.view(1, 1, b.num_heads, -1).transpose(1, 2)
        k = keys[:, :next_len].view(1, next_len, b.num_heads, -1).transpose(1, 2)
        v = values[:, :next_len].view(1, next_len, b.num_heads, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, ~mask if mask is not None else None)
        a = a.transpose(1, 2).reshape(1, 1, -1)
        x = F.layer_norm(
            x + F.linear(a, self.out, b.out_b),
            [b.hidden_dim],
            b.norm_w1,
            b.norm_b1,
            b.norm_eps1,
        )
        hidden = F.relu(F.linear(x, self.ff1, b.mlp.b1))
        x = F.layer_norm(
            x + F.linear(hidden, self.ff2, b.mlp.b2),
            [b.hidden_dim],
            b.norm_w2,
            b.norm_b2,
            b.norm_eps2,
        )
        return x, keys, values


@torch.jit.script
class RowTransformer:
    def __init__(self, blocks: list[RowBlock]):
        self.blocks = blocks

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        max_decode_steps: int,
        padding_mask: torch.Tensor | None = None,
        torch_sdpa: bool = True,
    ):
        keys = torch.jit.annotate(list[torch.Tensor], [])
        values = torch.jit.annotate(list[torch.Tensor], [])
        length = 0
        for block in self.blocks:
            x, k, v, length = block.process_prompt(
                x, attn_mask, max_decode_steps, padding_mask, torch_sdpa
            )
            keys.append(k)
            values.append(v)
        return x, keys, values, length

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: list[torch.Tensor],
        v_cache: list[torch.Tensor],
        cache_len: int,
        attn_mask: torch.Tensor = None,
        torch_sdpa: bool = True,
    ):
        for i in range(len(self.blocks)):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(
                x, k_cache[i], v_cache[i], cache_len, attn_mask, torch_sdpa
            )
        return x, k_cache, v_cache, cache_len + 1


def build(original):
    return RowTransformer([RowBlock(block) for block in original.blocks])
