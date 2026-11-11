"""Hash authoritative AR tensors, including those stored in TorchScript classes."""

import hashlib


def ar_digest(transformer):
    digest = hashlib.sha256()
    for index, cached in enumerate(transformer.blocks):
        block = cached.block if hasattr(cached, "block") else cached
        values = [
            ("qkv_w", block.qkv_w_t),
            ("qkv_b", block.qkv_b),
            ("out_w", block.out_w_t),
            ("out_b", block.out_b),
            ("ff1_w", block.mlp.w1_t),
            ("ff1_b", block.mlp.b1),
            ("ff2_w", block.mlp.w2_t),
            ("ff2_b", block.mlp.b2),
            ("norm_w1", block.norm_w1),
            ("norm_b1", block.norm_b1),
            ("norm_w2", block.norm_w2),
            ("norm_b2", block.norm_b2),
        ]
        for name, value in values:
            tensor = value.detach().cpu().contiguous()
            digest.update(
                str((index, name, tensor.dtype, tuple(tensor.shape))).encode()
            )
            digest.update(tensor.numpy().tobytes())
        digest.update(str((block.norm_eps1, block.norm_eps2)).encode())
    return digest.hexdigest()
