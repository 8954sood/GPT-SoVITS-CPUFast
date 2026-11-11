"""Lossless coefficient packing with FP32 ARM GEMV, for immutable CPUFast models."""

import torch
import torch.nn.functional as F
from AR.models.t2s_model import T2SBlock

import cpufast_cpp_ops


@torch.jit.script
def pack(weight_t: torch.Tensor):
    w = weight_t.transpose(0, 1)
    return (
        w.to(torch.float16)
        .reshape(w.shape[0] // 8, 8, w.shape[1])
        .transpose(1, 2)
        .contiguous()
    )


@torch.jit.script
class CppLosslessBlock:
    def __init__(
        self, block: T2SBlock, native_attention: bool = False, head_major: bool = False
    ):
        self.head_major = head_major
        self.native_attention = native_attention
        self.block = block
        self.qkv = pack(block.qkv_w_t)
        self.out = pack(block.out_w_t)
        self.ff1 = pack(block.mlp.w1_t)
        self.ff2 = pack(block.mlp.w2_t)

    def process_prompt(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        max_decode_steps: int,
        padding: torch.Tensor | None,
        torch_sdpa: bool,
    ):
        x, k, v, length = self.block.process_prompt(
            x, mask, max_decode_steps, padding, torch_sdpa
        )
        if self.head_major and x.shape[0] == 1:
            b, l, width = k.shape
            k = k.view(b, l, self.block.num_heads, -1).transpose(1, 2).contiguous()
            v = v.view(b, l, self.block.num_heads, -1).transpose(1, 2).contiguous()
        return x, k, v, length

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
        q, k, v = torch.ops.cpufast_cpp.lossless_linear(
            x.contiguous(), self.qkv, b.qkv_b
        ).chunk(3, dim=-1)
        next_len = cache_len + k.shape[1]
        if self.head_major and keys.dim() == 4:
            keys[:, :, cache_len:next_len] = k.view(1, 1, b.num_heads, -1).transpose(
                1, 2
            )
            values[:, :, cache_len:next_len] = v.view(1, 1, b.num_heads, -1).transpose(
                1, 2
            )
            a = F.scaled_dot_product_attention(
                q.view(1, 1, b.num_heads, -1).transpose(1, 2),
                keys[:, :, :next_len],
                values[:, :, :next_len],
                ~mask if mask is not None else None,
            )
            a = a.transpose(1, 2).reshape(1, 1, -1)
        else:
            keys[:, cache_len:next_len] = k
            values[:, cache_len:next_len] = v
            if self.native_attention:
                a = torch.ops.cpufast_cpp.token_attention(
                    q, keys, values, next_len, b.num_heads, mask
                )
            else:
                q = q.view(1, 1, b.num_heads, -1).transpose(1, 2)
                k = (
                    keys[:, :next_len]
                    .view(1, next_len, b.num_heads, -1)
                    .transpose(1, 2)
                )
                v = (
                    values[:, :next_len]
                    .view(1, next_len, b.num_heads, -1)
                    .transpose(1, 2)
                )
                a = F.scaled_dot_product_attention(
                    q, k, v, ~mask if mask is not None else None
                )
                a = a.transpose(1, 2).reshape(1, 1, -1)
        x = F.layer_norm(
            x
            + torch.ops.cpufast_cpp.lossless_linear(a.contiguous(), self.out, b.out_b),
            [b.hidden_dim],
            b.norm_w1,
            b.norm_b1,
            b.norm_eps1,
        )
        hidden = F.relu(
            torch.ops.cpufast_cpp.lossless_linear(x.contiguous(), self.ff1, b.mlp.b1)
        )
        x = F.layer_norm(
            x
            + torch.ops.cpufast_cpp.lossless_linear(
                hidden.contiguous(), self.ff2, b.mlp.b2
            ),
            [b.hidden_dim],
            b.norm_w2,
            b.norm_b2,
            b.norm_eps2,
        )
        return x, keys, values


@torch.jit.script
class CppLosslessTransformer:
    def __init__(self, blocks: list[CppLosslessBlock]):
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


def build(original, native_attention=False, head_major=False):
    import hashlib

    audit = []
    before = hashlib.sha256()
    for index, block in enumerate(original.blocks):
        for name, weight in [
            ("qkv", block.qkv_w_t),
            ("out", block.out_w_t),
            ("ff1", block.mlp.w1_t),
            ("ff2", block.mlp.w2_t),
        ]:
            exact = torch.equal(weight, weight.half().float())
            if not exact or weight.shape[1] % 8 or weight.shape[0] % 4:
                raise ValueError(
                    "No lossless ARM packing is possible for this checkpoint"
                )
            before.update(weight.contiguous().numpy().tobytes())
            audit.append(
                {"block": index, "matrix": name, "fp16_roundtrip_exact": exact}
            )
    runtime = CppLosslessTransformer(
        [
            CppLosslessBlock(block, native_attention, head_major)
            for block in original.blocks
        ]
    )
    after = hashlib.sha256()
    for cached, block in zip(runtime.blocks, original.blocks):
        for name, weight in [
            ("qkv", block.qkv_w_t),
            ("out", block.out_w_t),
            ("ff1", block.mlp.w1_t),
            ("ff2", block.mlp.w2_t),
        ]:
            packed = getattr(cached, name)
            restored = (
                packed.transpose(1, 2)
                .reshape(weight.shape[1], weight.shape[0])
                .t()
                .float()
            )
            assert torch.equal(restored, weight)
            after.update(weight.contiguous().numpy().tobytes())
    assert before.hexdigest() == after.hexdigest()
    return runtime, {
        "matrices": audit,
        "original_digest_before": before.hexdigest(),
        "original_digest_after": after.hexdigest(),
        "all_coefficients_exact": True,
        "arithmetic": "FP32 multiplication and accumulation",
        "native_workers": cpufast_cpp_ops.WORKERS,
    }
