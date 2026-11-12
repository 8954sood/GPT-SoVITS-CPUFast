"""Optional small-channel FP32 tap GEMM for the V2ProPlus decoder."""

import torch
import torch.nn.functional as F

from cpufast_fft import FFTConv1d


class TapConv1d(torch.nn.Module):
    """Preserve decoder weights; use a disposable, version-checked tap cache."""

    def __init__(self, original):
        super().__init__()
        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)
        self.padding = original.padding
        self.dilation = original.dilation
        self._cache_key = self._packed = None
        self.train(original.training)

    def forward(self, x, *, activate=False, residual=None):
        if (
            self.training
            or torch.is_grad_enabled()
            or x.ndim != 3
            or x.shape[0] != 1
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or self.weight.dtype != torch.float32
            or torch.is_autocast_enabled("cpu")
            or any(p.is_inference() for p in self.parameters())
        ):
            y = F.conv1d(
                x, self.weight, self.bias, padding=self.padding, dilation=self.dilation
            )
        else:
            key = (id(self.weight), self.weight._version, self.weight.data_ptr())
            if key != self._cache_key:
                self._packed = self.weight.detach().permute(2, 0, 1).contiguous()
                self._cache_key = key
            length = (
                x.shape[-1]
                + 2 * self.padding
                - self.dilation * (self.weight.shape[-1] - 1)
            )
            padded = F.pad(x, (self.padding, self.padding)).squeeze(0)
            y = torch.empty(
                (self.weight.shape[0], length), dtype=x.dtype, device=x.device
            )
            torch.mm(self._packed[0], padded[:, :length], out=y)
            for j in range(1, self.weight.shape[-1]):
                start = j * self.dilation
                torch.addmm(
                    y, self._packed[j], padded[:, start : start + length], out=y
                )
            if self.bias is not None:
                y.add_(self.bias[:, None])
            y = y.unsqueeze(0)
        if activate:
            y = F.leaky_relu(y, 0.1)
        return y if residual is None else y + residual


class PackedSmallTap(TapConv1d):
    """Channels-last FP32 conv for the measured 48-channel, 7-tap geometry."""

    def forward(self, x, *, activate=False, residual=None):
        if (
            self.training
            or torch.is_grad_enabled()
            or x.ndim != 3
            or x.shape[0] != 1
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or self.weight.dtype != torch.float32
            or torch.is_autocast_enabled("cpu")
            or any(p.is_inference() for p in self.parameters())
        ):
            return super().forward(x, activate=activate, residual=residual)
        key = (id(self.weight), self.weight._version, self.weight.data_ptr())
        if key != self._cache_key:
            self._packed = (
                self.weight.detach()
                .unsqueeze(2)
                .contiguous(memory_format=torch.channels_last)
            )
            self._cache_key = key
        y = F.conv2d(
            x.unsqueeze(2).contiguous(memory_format=torch.channels_last),
            self._packed,
            self.bias,
            padding=(0, self.padding),
            dilation=(1, self.dilation),
        ).squeeze(2)
        if activate:
            y = F.leaky_relu(y, 0.1)
        return y if residual is None else y + residual


def install_small_taps(blocks, max_channels=48, *, packed_48=False):
    """Replace eligible FFT layers; call after fused block installation."""
    count = 0
    for block in blocks:
        for group in (block.convs1, block.convs2):
            for index, layer in enumerate(group):
                if (
                    isinstance(layer, FFTConv1d)
                    and layer.weight.shape[1] <= max_channels
                ):
                    use_packed = (
                        packed_48
                        and layer.weight.shape[1] == 48
                        and layer.weight.shape[0] == 48
                        and layer.weight.shape[-1] == 7
                        and layer.dilation == 1
                    )
                    group[index] = (
                        PackedSmallTap(layer) if use_packed else TapConv1d(layer)
                    )
                    count += 1
    return count
