"""Experimental FP32 Cook-Toom transforms; preserve authoritative parameters."""

from fractions import Fraction
import torch
import torch.nn.functional as F
from cpufast_winograd import WinogradConv1d
from cpufast_optimizations import PhaseUpsample1d


def matrices(m, r):
    alpha = m + r - 1
    points = [
        Fraction(0),
        Fraction(1),
        Fraction(-1),
        Fraction(2),
        Fraction(-2),
        Fraction(1, 2),
        Fraction(-1, 2),
        Fraction(3),
        Fraction(-3),
        Fraction(3, 2),
        Fraction(-3, 2),
        Fraction(1, 4),
        Fraction(-1, 4),
        Fraction(4),
        Fraction(-4),
    ]
    points = points[: alpha - 1]
    if len(points) != alpha - 1:
        raise ValueError("Unsupported transform size")

    def poly(roots):
        p = [Fraction(1)]
        for root in roots:
            q = [Fraction(0)] * (len(p) + 1)
            for i, a in enumerate(p):
                q[i] -= root * a
                q[i + 1] += a
            p = q
        return p

    P = poly(points)
    B = []
    G = []
    for t in points:
        other = [x for x in points if x != t]
        p = poly(other)
        denom = Fraction(1)
        for x in other:
            denom *= t - x
        B.append(p + [Fraction(0)])
        G.append([t**j / denom for j in range(r)])
    B.append(P)
    G.append([Fraction(0)] * (r - 1) + [Fraction(1)])
    A = [[t**j for t in points] + [Fraction(int(j == m - 1))] for j in range(m)]
    return tuple(
        torch.tensor([[float(x) for x in row] for row in z]) for z in (B, G, A)
    )


class CookToom:
    def __init__(self, weight, m=6, native=False):
        self.native = native
        self.m = m
        self.r = weight.shape[-1]
        self.B, self.G, self.A = matrices(m, self.r)
        self.weights = torch.einsum("pr,oir->poi", self.G, weight).contiguous()

    def __call__(self, x, padding, dilation=1):
        d = dilation
        m = self.m
        n, c, _ = x.shape
        padded = F.pad(x, (padding, padding))
        desired = padded.shape[-1] - (self.r - 1) * d
        padded = F.pad(padded, (0, (-padded.shape[-1]) % d))
        streams = padded.reshape(n, c, -1, d).permute(0, 3, 1, 2).reshape(n * d, c, -1)
        length = streams.shape[-1] - self.r + 1
        if self.native:
            transformed = torch.ops.cpufast_cpp.cook_input(
                streams.contiguous(), self.B, m
            )
            products = torch.bmm(self.weights, transformed)
            return torch.ops.cpufast_cpp.cook_output(products, self.A, n, d, desired)
        windows = F.pad(streams, (0, (-length) % m)).unfold(-1, m + self.r - 1, m)
        transformed = (
            F.linear(windows, self.B)
            .permute(3, 1, 0, 2)
            .reshape(m + self.r - 1, c, -1)
            .contiguous()
        )
        products = torch.bmm(self.weights, transformed)
        result = F.linear(products.permute(1, 2, 0), self.A)
        tiles = windows.shape[-2]
        channels = self.weights.shape[1]
        result = (
            result.reshape(channels, n * d, tiles, m)
            .permute(1, 0, 2, 3)
            .reshape(n * d, channels, -1)[..., :length]
        )
        return (
            result.reshape(n, d, channels, length)
            .permute(0, 2, 3, 1)
            .reshape(n, channels, -1)[..., :desired]
        )


class CookToomConv1d(torch.nn.Module):
    def __init__(self, original, m=6, native=False):
        self.native_cook = native
        super().__init__()
        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)
        self.padding = (
            original.padding
            if isinstance(original.padding, int)
            else original.padding[0]
        )
        self.dilation = (
            original.dilation
            if isinstance(original.dilation, int)
            else original.dilation[0]
        )
        self.m = m
        self._runner = None
        self._cook_key = None
        self.train(original.training)

    def forward(self, x):
        if (
            self.training
            or torch.is_grad_enabled()
            or x.dtype != torch.float32
            or x.device.type != "cpu"
        ):
            return F.conv1d(
                x, self.weight, self.bias, padding=self.padding, dilation=self.dilation
            )
        key = tuple((id(p), p._version, p.data_ptr()) for p in self.parameters())
        if key != self._cook_key:
            self._runner = CookToom(self.weight.detach(), self.m, self.native_cook)
            self._cook_key = key
        y = self._runner(x, self.padding, self.dilation)
        return y if self.bias is None else y + self.bias[None, :, None]


class CookToomPhase(PhaseUpsample1d):
    def __init__(self, original, m=4, native=False):
        self.native_cook = native
        # Original is the original ConvTranspose1d (before phase installation).
        super().__init__(original)
        self.m = m
        self._runner = None
        self._cook_key = None

    def forward(self, x):
        if (
            self.training
            or torch.is_grad_enabled()
            or x.dtype != torch.float32
            or x.device.type != "cpu"
        ):
            return super().forward(x)
        if self.taps == 1:
            return super().forward(x)
        key = tuple((id(p), p._version, p.data_ptr()) for p in self.parameters())
        stride = self.stride[0]
        if key != self._cook_key:
            ci, co, _ = self.weight.shape
            weight = (
                self.weight.detach()
                .reshape(ci, co, self.taps, stride)
                .permute(1, 3, 0, 2)
                .flip(-1)
                .reshape(co * stride, ci, self.taps)
            )
            self._runner = CookToom(weight, self.m, self.native_cook)
            self._cook_key = key
        output = self._runner(x, self.taps - 1)
        if self.bias is not None:
            output = output + self.bias.repeat_interleave(stride)[None, :, None]
        n, _, length = output.shape
        result = (
            output.reshape(n, self.weight.shape[1], stride, length)
            .permute(0, 1, 3, 2)
            .reshape(n, self.weight.shape[1], length * stride)
        )
        p = self.padding[0]
        return result[..., p : result.shape[-1] - p] if p else result
