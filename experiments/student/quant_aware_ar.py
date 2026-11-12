"""CPU training surrogate for FP16-representable weights and W8A8 decode.

Prompt rows stay FP32. Only rows after prediction_start+1 use dynamic per-token
activation and per-output weight quantization. This does not emulate native
attention/normalization bitwise and is never an inference/default engine.
"""

import torch
import torch.nn.functional as F

from experiments.student.distill_ar import ARStudent


def half_ste(value):
    rounded = value.half().float()
    if not torch.isfinite(rounded).all():
        raise ValueError("Nonfinite half-representable training matrix")
    return value + (rounded - value).detach()


def decode_linear(module, value, prefix):
    weight = half_ste(module.weight)
    prompt = F.linear(value[:, :prefix], weight, module.bias)
    if prefix == value.shape[1]:
        return prompt
    current = value[:, prefix:]
    with torch.no_grad():
        maximum = current.detach().abs().amax(-1, keepdim=True)
        sx = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        sw = (weight.detach().abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
        xq = torch.round(current.detach() * sx.reciprocal()).clamp(-127, 127)
        wq = torch.round(weight.detach() / sw).clamp(-127, 127)
        # Float accumulation of integral products is a training surrogate, not
        # a claim of equality to every possible native int32 dot product.
        exact_grid = F.linear(xq, wq) * sx * sw.T
        if module.bias is not None:
            exact_grid = exact_grid + module.bias
    x_ste = current + (xq * sx - current).detach()
    w_ste = weight + (wq * sw - weight).detach()
    surrogate = F.linear(x_ste, w_ste, module.bias)
    semantic = surrogate + (exact_grid - surrogate).detach()
    return torch.cat([prompt, semantic], dim=1)


def block_forward(block, value, blocked, prefix):
    batch, length, width = value.shape
    q, k, v = decode_linear(block.qkv, value, prefix).chunk(3, -1)
    q, k, v = [
        tensor.view(batch, length, block.heads, width // block.heads).transpose(1, 2)
        for tensor in (q, k, v)
    ]
    attended = F.scaled_dot_product_attention(q, k, v, ~blocked)
    attended = attended.transpose(1, 2).reshape(batch, length, width)
    value = block.norm1(value + decode_linear(block.out, attended, prefix))
    hidden = F.relu(decode_linear(block.ff1, value, prefix))
    return block.norm2(value + decode_linear(block.ff2, hidden, prefix))


class QuantAwareARStudent(ARStudent):
    def _forward(self, value, blocked, start, anchors):
        prefix = start + 1
        if not 0 < prefix <= value.shape[1]:
            raise ValueError("Teacher trace must identify the actual prefill boundary")
        captured = {}
        for original, block in zip(self.keep, self.blocks):
            value = block_forward(block, value, blocked, prefix)
            if original in anchors:
                captured[str(original)] = value[:, start:].clone()
        if len(captured) != len(anchors):
            raise ValueError("Feature anchors must be retained original layers")
        logits = F.linear(value[:, start:], half_ste(self.predictor.weight))
        return logits.reshape(-1, self.predictor.out_features), captured

    def forward(self, x, blocked, start):
        return self._forward(x, blocked, start, ())[0]

    def forward_with_features(self, x, blocked, start, anchors):
        return self._forward(x, blocked, start, anchors)
