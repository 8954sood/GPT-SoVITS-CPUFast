"""Explicit unapproved FP32 FFT-size runtime candidate; authoritative weights unchanged."""

from cpufast_fft import FFTConv1d


def enable_fft128_cin240(decoder):
    """Change only six cold stage0 conv2 caches, before any inference/cache build."""
    selected = []
    expected_names = {f"resblocks.{branch}.convs2.{pair}" for branch in (0, 1) for pair in range(3)}
    for name, layer in decoder.named_modules():
        if isinstance(layer, FFTConv1d) and tuple(layer.weight.shape[:2]) == (384, 240):
            selected.append((name, layer))
    if {name for name, _ in selected} != expected_names:
        raise ValueError("Expected exactly the six original V stage0 Cin240/Cout384 conv2 modules")
    for name, layer in selected:
        if (layer.training or layer.weight.device.type != "cpu"
            or layer.n_fft != 256 or layer._key is not None or layer._packed is not None
            or layer.dilation != 1 or layer.weight.shape[-1] not in (7, 11)
            or layer.backend != "packed" or not layer.output_layout
            or not layer.stream_products or layer.fast_half or layer.half_stream_products
            or layer.weight.dtype.is_complex or str(layer.weight.dtype) != "torch.float32"):
            raise ValueError("Candidate requires cold canonical packed FP32 FFT modules: " + name)
    audit = []
    for name, layer in selected:
        before = (id(layer.weight), layer.weight._version, id(layer.bias))
        old_hop = layer.hop
        layer.n_fft = 128
        layer.hop = 128 - layer.effective_kernel + 1
        if before != (id(layer.weight), layer.weight._version, id(layer.bias)):
            raise RuntimeError("Authoritative parameter identity/version changed")
        channels_out, channels_in, _ = layer.weight.shape
        audit.append({"module": name, "weight_shape": list(layer.weight.shape),
            "n_fft_before": 256, "n_fft_after": 128,
            "hop_before": old_hop, "hop_after": layer.hop,
            "derived_spectral_bytes_before": 3*129*channels_out*channels_in*4,
            "derived_spectral_bytes_after": 3*65*channels_out*channels_in*4,
            "parameter_identity_version_unchanged": before == (id(layer.weight), layer.weight._version, id(layer.bias)),
            "arithmetic": "FP32 FFT, spectral products, and inverse FFT; transform length changes rounding"})
    return audit
