"""Value-preserving FP32 execution for one-token classifier projection."""

import torch
import torch.nn.functional as F


class NativePredictor(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)
        self._key = self._packed = self._packed_bias = None
        self.train(original.training)

    def forward(self, x):
        if (
            self.training
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cpu")
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or self.weight.device.type != "cpu"
            or self.weight.dtype != torch.float32
            or x.numel() != self.weight.shape[1]
        ):
            return F.linear(x, self.weight, self.bias)
        key = tuple(
            (id(p), p._version, p.data_ptr(), p.dtype, p.device)
            for p in self.parameters()
        )
        if self._key != key:
            w = self.weight.detach()
            half = w.half()
            if not torch.equal(half.float(), w):
                return F.linear(x, self.weight, self.bias)
            n, k = w.shape
            padded = (n + 7) // 8 * 8
            self._packed = (
                F.pad(half, (0, 0, 0, padded - n))
                .reshape(padded // 8, 8, k)
                .transpose(1, 2)
                .contiguous()
            )
            bias = self.bias.detach() if self.bias is not None else w.new_zeros(n)
            self._packed_bias = F.pad(bias, (0, padded - n)).contiguous()
            self._key = key
        return torch.ops.cpufast_cpp.lossless_linear(
            x.contiguous(), self._packed, self._packed_bias
        )[..., : self.weight.shape[0]]
