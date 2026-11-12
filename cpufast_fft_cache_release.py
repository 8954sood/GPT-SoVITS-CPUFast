"""Research opt-in: release unused FP32 spectral buffers after half inference.

Authoritative FP32 parameters stay unchanged. The base FFT implementation and
default runtime remain untouched. Logical released tensor bytes do not imply
an equal decrease in process RSS or a measured RTF improvement.
"""

from cpufast_fft import FFTConv1d


def install_half_cache_release(decoder):
    targets = [
        (name, module)
        for name, module in decoder.named_modules()
        if isinstance(module, FFTConv1d)
    ]
    if not targets or any(
        not module.fast_half
        or module.backend != "packed"
        or "half_cache_release_enabled" in module.__dict__
        for _, module in targets
    ):
        raise ValueError(
            "Cache release requires fresh, explicitly selected packed half-FFT modules"
        )
    audit = []
    for name, module in targets:
        original = module.forward
        module.half_cache_release_enabled = True

        def forward(
            x, *, activate=False, residual=None, layer=module, implementation=original
        ):
            # A previous successful half call may have released only its FP32
            # derived buffer. Rebuild from original parameters when half mode
            # or this memory option is turned off. Parameter versions are still
            # checked by the original FFT implementation on every forward.
            if (
                layer._packed is None
                and layer._key is not None
                and (
                    not layer.fast_half
                    or not layer.half_cache_release_enabled
                    or layer._packed_half is None
                )
            ):
                layer._key = None
            output = implementation(x, activate=activate, residual=residual)
            if (
                layer.half_cache_release_enabled
                and layer.fast_half
                and layer._packed_half is not None
            ):
                layer._packed = None
            return output

        module.forward = forward
        audit.append(
            {
                "module": name,
                "input_channels": int(module.weight.shape[1]),
                "output_channels": int(module.weight.shape[0]),
                "n_fft": module.n_fft,
                "authoritative_parameters": "Original FP32 objects unchanged",
                "release_scope": "Unused derived FP32 packed spectral tensor only",
            }
        )
    return audit


def live_spectral_cache_bytes(decoder):
    totals = {"fp32_derived_bytes": 0, "fp16_derived_bytes": 0}
    for module in decoder.modules():
        if isinstance(module, FFTConv1d):
            for name, key in (
                ("_packed", "fp32_derived_bytes"),
                ("_packed_half", "fp16_derived_bytes"),
            ):
                value = getattr(module, name, None)
                if value is not None:
                    totals[key] += value.numel() * value.element_size()
    return totals
