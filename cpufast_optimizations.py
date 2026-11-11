"""Optional CPUFast FP32 inference layouts; original parameters stay authoritative."""

import functools
import inspect
import threading

import torch
import torch.nn.functional as F


class PackedConv1d(torch.nn.Module):
    """Cache a 2-D inference representation without changing parameter storage."""

    def __init__(self, original, backend="channels_last"):
        super().__init__()
        if not isinstance(original, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            raise TypeError("Expected a one-dimensional convolution")
        if original.padding_mode != "zeros" or isinstance(original.padding, str):
            raise ValueError("Only explicit zero padding is supported")
        if backend not in ("channels_last", "xnnpack"):
            raise ValueError("Unknown FP32 convolution backend")
        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.transposed = isinstance(original, torch.nn.ConvTranspose1d)
        self.output_padding = original.output_padding if self.transposed else (0,)
        self.backend = backend
        self._key = self._packed = None
        self.train(original.training)

    def _fallback(self, x):
        if self.transposed:
            return F.conv_transpose1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, x):
        if (
            self.training
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cpu")
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or self.weight.device.type != "cpu"
            or self.weight.dtype != torch.float32
            or x.ndim != 3
        ):
            return self._fallback(x)
        parameters = [self.weight] + ([] if self.bias is None else [self.bias])
        key = (
            None
            if any(p.is_inference() for p in parameters)
            else (
                tuple(
                    (id(p), p._version, p.data_ptr(), p.dtype, p.device)
                    for p in parameters
                ),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.output_padding,
                self.backend,
            )
        )
        stride, padding, dilation = (
            [1, self.stride[0]],
            [0, self.padding[0]],
            [1, self.dilation[0]],
        )
        if self._packed is None or key is None or key != self._key:
            weight = self.weight.detach().unsqueeze(2)
            if self.backend == "channels_last":
                self._packed = weight.contiguous(memory_format=torch.channels_last)
            elif self.transposed:
                self._packed = torch.ops.prepacked.conv2d_transpose_clamp_prepack(
                    weight,
                    self.bias,
                    stride,
                    padding,
                    [0, self.output_padding[0]],
                    dilation,
                    self.groups,
                    None,
                    None,
                )
            else:
                self._packed = torch.ops.prepacked.conv2d_clamp_prepack(
                    weight,
                    self.bias,
                    stride,
                    padding,
                    dilation,
                    self.groups,
                    None,
                    None,
                )
            self._key = key
        value = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        if self.backend == "xnnpack":
            if self.transposed:
                return torch.ops.prepacked.conv2d_transpose_clamp_run(
                    value, self._packed
                ).squeeze(2)
            return torch.ops.prepacked.conv2d_clamp_run(value, self._packed).squeeze(2)
        if self.transposed:
            return F.conv_transpose2d(
                value,
                self._packed,
                self.bias,
                stride,
                padding,
                [0, self.output_padding[0]],
                self.groups,
                dilation,
            ).squeeze(2)
        return F.conv2d(
            value,
            self._packed,
            self.bias,
            stride,
            padding,
            dilation,
            self.groups,
        ).squeeze(2)


def install_convolutions(model, backend="channels_last"):
    """Replace execution modules, preserving parameter objects and state-dict keys."""
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            if hasattr(child, "weight_g") or hasattr(child, "weight_v"):
                raise ValueError(
                    "Remove runtime weight norm before packing convolutions"
                )
            setattr(model, name, PackedConv1d(child, backend))
            count += 1
        elif not isinstance(child, PackedConv1d):
            count += install_convolutions(child, backend)
    return count


def parameter_digest(model):
    """Hash values and names in canonical order, independent of execution layout."""
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def install_subpixel(model, *, pack_convolutions=True):
    """Exact upsampling geometry using unchanged FP32 phase-kernel caches."""
    count = 0
    for index, layer in enumerate(model.ups):
        if isinstance(layer, torch.nn.ConvTranspose1d):
            model.ups[index] = PhaseUpsample1d(layer)
            count += 1
    if pack_convolutions:
        count += install_convolutions(model, "channels_last")
    return count


class PhaseUpsample1d(PackedConv1d):
    """Phase convolution for stride-divisible kernels, including V2ProPlus 1/4 taps."""

    def __init__(self, original):
        if (
            not isinstance(original, torch.nn.ConvTranspose1d)
            or original.groups != 1
            or original.dilation != (1,)
            or original.output_padding != (0,)
            or original.kernel_size[0] % original.stride[0] != 0
        ):
            raise ValueError(
                "Expected ungrouped, stride-divisible transposed convolution"
            )
        super().__init__(original)
        self.taps = original.kernel_size[0] // original.stride[0]
        self._packed_bias = None

    def forward(self, x):
        if (
            self.training
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cpu")
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or self.weight.device.type != "cpu"
            or self.weight.dtype != torch.float32
        ):
            return self._fallback(x)
        parameters = [self.weight] + ([] if self.bias is None else [self.bias])
        key = (
            None
            if any(p.is_inference() for p in parameters)
            else tuple(
                (id(p), p._version, p.data_ptr(), p.dtype, p.device) for p in parameters
            )
        )
        stride = self.stride[0]
        if self._packed is None or key is None or key != self._key:
            c_in, c_out, _ = self.weight.shape
            self._packed = (
                self.weight.detach()
                .reshape(c_in, c_out, self.taps, stride)
                .permute(1, 3, 0, 2)
                .flip(-1)
                .reshape(c_out * stride, c_in, self.taps)
                .unsqueeze(2)
                .contiguous(memory_format=torch.channels_last)
            )
            self._packed_bias = (
                None
                if self.bias is None
                else self.bias.detach().repeat_interleave(stride)
            )
            self._key = key
        output = F.conv2d(
            x.unsqueeze(2).contiguous(memory_format=torch.channels_last),
            self._packed,
            self._packed_bias,
            padding=(0, self.taps - 1),
        ).squeeze(2)
        n, _, length = output.shape
        result = (
            output.reshape(n, self.weight.shape[1], stride, length)
            .permute(0, 1, 3, 2)
            .reshape(n, self.weight.shape[1], length * stride)
        )
        padding = self.padding[0]
        return result[..., padding : result.shape[-1] - padding]


class CPUFastRuntime:
    """Serialize a mutable CPUFast pipeline and schedule AR separately.

    The lock is shared by these wrappers because PyTorch intra-op thread count
    is process-wide. Keep other inference workloads in separate processes.
    """

    _lock = threading.RLock()

    def __init__(self, pipeline, *, ar_threads=1, acoustic_threads=2):
        if ar_threads not in (1, 2) or acoustic_threads not in (1, 2):
            raise ValueError("Expected one or two CPU threads per stage")
        decoder = pipeline.t2s_model.model
        if getattr(decoder, "_cpufast_runtime_installed", False):
            raise ValueError("Runtime scheduling has already been installed")
        self.pipeline = pipeline
        self.ar_threads = ar_threads
        self.acoustic_threads = acoustic_threads

        def wrap(original):
            if inspect.isgeneratorfunction(inspect.unwrap(original)):

                @functools.wraps(original)
                def generate(*args, **kwargs):
                    old = torch.get_num_threads()
                    torch.set_num_threads(ar_threads)
                    try:
                        yield from original(*args, **kwargs)
                    finally:
                        torch.set_num_threads(old)

                return generate

            @functools.wraps(original)
            def call(*args, **kwargs):
                old = torch.get_num_threads()
                torch.set_num_threads(ar_threads)
                try:
                    return original(*args, **kwargs)
                finally:
                    torch.set_num_threads(old)

            return call

        for name in (
            "infer_panel_batch_infer",
            "infer_panel_naive_batched",
            "infer_panel_naive",
        ):
            setattr(decoder, name, wrap(getattr(decoder, name)))
        decoder._cpufast_runtime_installed = True

    def run(self, inputs):
        with self._lock:
            old = torch.get_num_threads()
            torch.set_num_threads(self.acoustic_threads)
            try:
                yield from self.pipeline.run(inputs)
            finally:
                torch.set_num_threads(old)
