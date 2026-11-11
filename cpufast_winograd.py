"""FP32 F(4,3) Winograd convolution, with polyphase handling of dilation."""

import torch
import torch.nn.functional as F


class WinogradConv1d(torch.nn.Module):
    def __init__(self, original, native=False):
        super().__init__()
        if (
            original.weight.shape[-1] != 3
            or original.groups != 1
            or original.stride != (1,)
        ):
            raise ValueError("Expected stride-one, three-tap, ungrouped convolution")
        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)
        self.padding = original.padding[0]
        self.dilation = original.dilation[0]
        self._key = self._transformed = None
        self.native = native
        self.train(original.training)

    def forward(self, x, *, activate=False, residual=None):
        if (
            self.training
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cpu")
            or x.ndim != 3
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or self.weight.device.type != "cpu"
            or self.weight.dtype != torch.float32
            or any(p.is_inference() for p in self.parameters())
        ):
            y = F.conv1d(
                x, self.weight, self.bias, padding=self.padding, dilation=self.dilation
            )
            if activate:
                y = F.leaky_relu(y, 0.1)
            return y if residual is None else y + residual
        key = tuple(
            (id(p), p._version, p.data_ptr(), p.dtype, p.device)
            for p in self.parameters()
        )
        if key != self._key:
            w0, w1, w2 = self.weight.detach().unbind(-1)
            self._transformed = torch.stack(
                (
                    w0 * 0.25,
                    -(w0 + w1 + w2) / 6,
                    (-w0 + w1 - w2) / 6,
                    w0 / 24 + w1 / 12 + w2 / 6,
                    w0 / 24 - w1 / 12 + w2 / 6,
                    w2,
                )
            ).contiguous()
            self._key = key
        d = self.dilation
        padded = F.pad(x, (self.padding, self.padding))
        desired = padded.shape[-1] - 2 * d
        if desired <= 0:
            return F.conv1d(x, self.weight, self.bias, padding=self.padding, dilation=d)
        padded = F.pad(padded, (0, (-padded.shape[-1]) % d))
        n, c, _ = padded.shape
        streams = padded.reshape(n, c, -1, d).permute(0, 3, 1, 2).reshape(n * d, c, -1)
        if self.native:
            transformed = torch.ops.cpufast_cpp.winograd_input(streams.contiguous())
            products = torch.bmm(self._transformed, transformed)
            return torch.ops.cpufast_cpp.winograd_output(
                products, n, d, desired, self.bias, activate, residual
            )
        length = streams.shape[-1] - 2
        windows = F.pad(streams, (0, (-length) % 4)).unfold(-1, 6, 4)
        a, b, c0, d0, e, f = windows.unbind(-1)
        transformed = (
            torch.stack(
                (
                    4 * a - 5 * c0 + e,
                    -4 * b - 4 * c0 + d0 + e,
                    4 * b - 4 * c0 - d0 + e,
                    -2 * b - c0 + 2 * d0 + e,
                    2 * b - c0 - 2 * d0 + e,
                    4 * b - 5 * d0 + f,
                )
            )
            .permute(0, 2, 1, 3)
            .reshape(6, x.shape[1], -1)
            .contiguous()
        )
        products = torch.bmm(self._transformed, transformed)
        m0, m1, m2, m3, m4, m5 = products.unbind(0)
        outputs = torch.stack(
            (
                m0 + m1 + m2 + m3 + m4,
                m1 - m2 + 2 * (m3 - m4),
                m1 + m2 + 4 * (m3 + m4),
                m1 - m2 + 8 * (m3 - m4) + m5,
            ),
            -1,
        )
        tiles = windows.shape[-2]
        outputs = (
            outputs.reshape(self.weight.shape[0], n * d, tiles, 4)
            .permute(1, 0, 2, 3)
            .reshape(n * d, self.weight.shape[0], -1)[..., :length]
        )
        outputs = (
            outputs.reshape(n, d, self.weight.shape[0], length)
            .permute(0, 2, 3, 1)
            .reshape(n, self.weight.shape[0], -1)[..., :desired]
        )
        if self.bias is not None:
            outputs = outputs + self.bias[None, :, None]
        if activate:
            outputs = F.leaky_relu(outputs, 0.1)
        return outputs if residual is None else outputs + residual


def install_winograd(model, *, native=False):
    from cpufast_optimizations import PackedConv1d

    count = 0
    for name, child in list(model.named_children()):
        if (
            isinstance(child, (torch.nn.Conv1d, PackedConv1d))
            and not getattr(child, "transposed", False)
            and child.weight.shape[-1] == 3
        ):
            setattr(model, name, WinogradConv1d(child, native=native))
            count += 1
        elif not isinstance(child, WinogradConv1d):
            count += install_winograd(child, native=native)
    return count
