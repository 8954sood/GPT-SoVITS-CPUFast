"""Optional convolution epilogues for the original residual block operation order."""

import types
import torch
import torch.nn.functional as F
from cpufast_fft import FFTConv1d
from cpufast_winograd import WinogradConv1d


def install_fused_blocks(blocks):
    count = 0
    for block in blocks:
        if not all(
            isinstance(c, (FFTConv1d, WinogradConv1d))
            for c in (*block.convs1, *block.convs2)
        ):
            continue
        original = block.forward

        def forward(self, x, x_mask=None, _original=original):
            if x_mask is not None or self.training or torch.is_grad_enabled():
                return _original(x, x_mask)
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = c1(F.leaky_relu(x, 0.1), activate=True)
                x = c2(xt, residual=x)
            return x

        block.forward = types.MethodType(forward, block)
        count += 1
    return count
