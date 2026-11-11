"""FP32 overlap-save convolution with disposable spectral coefficient caches."""

import torch
import torch.nn.functional as F


class FFTConv1d(torch.nn.Module):
    """Compute the original cross-correlation; preserve all parameter objects."""

    def __init__(self, original, n_fft=None, backend="torch"):
        super().__init__()
        if (
            original.stride != (1,)
            or original.groups != 1
            or getattr(original, "transposed", False)
            or isinstance(original.padding, str)
        ):
            raise ValueError(
                "FFT convolution requires stride-one ungrouped convolution"
            )
        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)
        self.padding = original.padding[0]
        self.dilation = original.dilation[0]
        self.effective_kernel = self.dilation * (self.weight.shape[-1] - 1) + 1
        self.n_fft = n_fft or (128 if self.weight.shape[1] >= 256 else 256)
        if self.n_fft < self.effective_kernel:
            raise ValueError("FFT size is smaller than the effective kernel")
        self.hop = self.n_fft - self.effective_kernel + 1
        self._key = self._wr = self._wi = self._ws = None
        self.backend = backend
        self._packed = None
        self._native = None
        self.train(original.training)

    def forward(self, x):
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
            return F.conv1d(
                x, self.weight, self.bias, padding=self.padding, dilation=self.dilation
            )
        key = tuple(
            (id(p), p._version, p.data_ptr(), p.dtype, p.device)
            for p in self.parameters()
        )
        if self._key != key:
            w = self.weight.detach()
            kernel = w.new_zeros(w.shape[0], w.shape[1], self.effective_kernel)
            kernel[..., :: self.dilation] = w
            spectrum = torch.fft.rfft(kernel.flip(-1), n=self.n_fft).permute(2, 0, 1)
            self._wr = spectrum.real.contiguous()
            self._wi = spectrum.imag.contiguous()
            self._ws = self._wr + self._wi
            if self.backend == "fused":
                self._packed = torch.cat(
                    (self._wr, self._wi, self._ws), dim=0
                ).contiguous()
                self._wr = self._wi = self._ws = None
            elif self.backend == "native":
                if w.shape[0] % 12:
                    raise ValueError(
                        "Native spectral GEMM requires output channels divisible by 12"
                    )
                coefficients = torch.cat((self._wr, self._wi, self._ws), dim=0)
                self._packed = (
                    coefficients.reshape(
                        coefficients.shape[0], w.shape[0] // 12, 12, w.shape[1]
                    )
                    .transpose(2, 3)
                    .contiguous()
                )
                self._wr = self._wi = self._ws = None
            elif self.backend not in ("torch", "fused"):
                raise ValueError("Unknown spectral GEMM backend")
            self._key = key
        padded = F.pad(x, (self.padding, self.padding))
        output_length = padded.shape[-1] - self.effective_kernel + 1
        if output_length <= 0:
            return F.conv1d(
                x, self.weight, self.bias, padding=self.padding, dilation=self.dilation
            )
        blocks = (padded.shape[-1] + self.hop - 1) // self.hop
        windows = F.pad(
            padded, (self.effective_kernel - 1, blocks * self.hop - padded.shape[-1])
        ).unfold(-1, self.n_fft, self.hop)
        transformed_raw = torch.fft.rfft(windows, n=self.n_fft)
        if self.backend == "fused":
            packed_input = torch.ops.cpufast_cpp.fft_split(transformed_raw.contiguous())
            products = torch.bmm(self._packed, packed_input)
            product = torch.ops.cpufast_cpp.fft_combine(products)
        else:
            transformed = transformed_raw.permute(3, 1, 0, 2).reshape(
                self.n_fft // 2 + 1, x.shape[1], -1
            )
            real = transformed.real.contiguous()
            imag = transformed.imag.contiguous()
            if self.backend == "native":
                a, b, c = self._native_mix(real, imag)
            else:
                a = torch.bmm(self._wr, real)
                b = torch.bmm(self._wi, imag)
                c = torch.bmm(self._ws, real + imag)
            product = torch.complex(a - b, c - a - b)
        product = product.reshape(
            self.n_fft // 2 + 1, self.weight.shape[0], x.shape[0], blocks
        ).permute(2, 1, 3, 0)
        output = torch.fft.irfft(product, n=self.n_fft)[
            ..., self.effective_kernel - 1 :
        ]
        output = output.reshape(x.shape[0], self.weight.shape[0], -1)
        output = output[
            ..., self.effective_kernel - 1 : self.effective_kernel - 1 + output_length
        ]
        if self.bias is not None:
            output = output + self.bias[None, :, None]
        return output

    def _native_mix(self, real, imag):
        if self._native is None:
            self._native = load_native()
        frequencies, channels, rows = real.shape
        padded_rows = (rows + 7) // 8 * 8
        data = torch.cat((real, imag, real + imag), dim=0).transpose(1, 2)
        data = (
            F.pad(data, (0, 0, 0, padded_rows - rows))
            .reshape(3 * frequencies, padded_rows // 8, 8, channels)
            .transpose(2, 3)
            .contiguous()
        )
        outputs = self.weight.shape[0]
        product = real.new_empty(
            3 * frequencies, padded_rows // 8, outputs // 12, 8, 12
        )
        self._native.cpufast_fft_gemm(
            data.data_ptr(),
            self._packed.data_ptr(),
            product.data_ptr(),
            3 * frequencies,
            padded_rows,
            outputs,
            channels,
        )
        product = (
            product.permute(0, 1, 3, 2, 4)
            .reshape(3, frequencies, padded_rows, outputs)[:, :, :rows]
            .transpose(2, 3)
        )
        return product[0], product[1], product[2]


def load_native():
    import ctypes
    import hashlib
    import json
    import os
    import platform
    from pathlib import Path

    if platform.machine() != "aarch64" or platform.system() != "Linux":
        raise RuntimeError("Native spectral GEMM requires Linux ARM64")
    path = Path(os.environ.get("CPUFAST_FFT_LIBRARY", "/work/libcpufast_fft.so"))
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    root = Path(__file__).resolve().parent
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["library_sha256"]:
        raise RuntimeError("Spectral GEMM library checksum mismatch")
    if any(
        hashlib.sha256((root / name).read_bytes()).hexdigest() != value
        for name, value in manifest["sources"].items()
    ):
        raise RuntimeError("Spectral GEMM sources do not match build")
    library = ctypes.CDLL(str(path))
    if library.cpufast_fft_abi() != 1:
        raise RuntimeError("Spectral GEMM ABI mismatch")
    library.cpufast_fft_gemm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 4
    library.cpufast_fft_gemm.restype = None
    return library


def tuned_size(channels, kernel, dilation):
    """Geometry-only policy from the frozen-quota probes, bounded for 8GB RSS."""
    if channels >= 256:
        if kernel == 7 and dilation == 1:
            return 64
        return 256 if dilation >= 3 else 128
    if channels == 192:
        return 512 if kernel == 7 and dilation >= 3 else 256
    if channels == 96:
        return 1024 if kernel == 11 or dilation >= 3 else 256
    if channels == 48:
        return 1024 if dilation >= 3 else 128
    return 1024 if kernel == 11 and dilation >= 3 else 256


def install_fft(model, *, min_kernel=7, backend="torch", tuned=False):
    """Use measured long-kernel candidates only; leave phase upsampling intact."""
    from cpufast_optimizations import PackedConv1d, PhaseUpsample1d

    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, PhaseUpsample1d):
            continue
        if (
            isinstance(child, (torch.nn.Conv1d, PackedConv1d))
            and not getattr(child, "transposed", False)
            and (
                child.weight.shape[-1] >= min_kernel
                or (
                    tuned
                    and child.weight.shape[-1] == 3
                    and child.weight.shape[1] == 192
                )
            )
            and child.groups == 1
            and child.stride == (1,)
        ):
            size = (
                tuned_size(
                    child.weight.shape[1], child.weight.shape[-1], child.dilation[0]
                )
                if tuned
                else None
            )
            setattr(model, name, FFTConv1d(child, n_fft=size, backend=backend))
            count += 1
        elif not isinstance(child, FFTConv1d):
            count += install_fft(
                child, min_kernel=min_kernel, backend=backend, tuned=tuned
            )
    return count
