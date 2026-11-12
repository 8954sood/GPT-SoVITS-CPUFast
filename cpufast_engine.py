"""Validated, opt-in V2ProPlus runtime for the fixed CPU-throughput proxy."""

from cpufast_integrity import ar_digest
from cpufast_optimizations import CPUFastRuntime, install_subpixel, parameter_digest


def optimize(
    pipeline,
    *,
    native_attention=False,
    head_major=False,
    cook_phase=False,
    cook_resblock=0,
    fft_layout=False,
    packed_fft=False,
    fused_finish=False,
    tap_small=False,
    packed_small_48=False,
    n1_fft_sizes=False,
    ar_threads=1,
    acoustic_threads=1,
    fast_experimental=False,
):
    """Return a serial engine; every original learned coefficient is preserved.

    Instantiate/load the CPUFast TTS pipeline before calling this function.
    Set ar_threads and acoustic_threads to 1 or 2 for the target CPU.
    Recreate the engine after changing model checkpoints. The native library
    must be built for the installed Linux ARM64 PyTorch version.
    """
    if ar_threads not in (1, 2) or acoustic_threads not in (1, 2):
        raise ValueError("Expected one or two CPU threads per stage")
    if packed_small_48 and not tap_small:
        raise ValueError("packed_small_48 requires tap_small")
    if fast_experimental and (tap_small or cook_resblock):
        raise ValueError(
            "The fast profile cannot be combined with tap_small or cook_resblock"
        )
    existing = getattr(pipeline, "_cpufast_engine", None)
    if existing is not None:
        if (existing.ar_threads, existing.acoustic_threads) != (
            ar_threads,
            acoustic_threads,
        ):
            raise ValueError("Recreate the pipeline before changing thread settings")
        if existing.audit.get("tap_small", False) != tap_small:
            raise ValueError("Recreate the pipeline before changing tap settings")
        if existing.audit.get("packed_small_48", False) != packed_small_48:
            raise ValueError(
                "Recreate the pipeline before changing packed small-channel settings"
            )
        if existing.audit.get("n1_fft_sizes", False) != n1_fft_sizes:
            raise ValueError("Recreate the pipeline before changing FFT sizes")
        if existing.audit.get("fast_experimental", False) != fast_experimental:
            raise ValueError("Recreate the pipeline before changing fast settings")
        return existing
    if pipeline.configs.version != "v2ProPlus" or pipeline.configs.is_half:
        raise ValueError("This profile requires the FP32 V2ProPlus CPU pipeline")
    if str(pipeline.configs.device) != "cpu":
        raise ValueError("This profile is for CPU inference")
    if pipeline.vits_model.training or pipeline.t2s_model.model.training:
        raise ValueError("Load evaluation models before optimizing")
    if fast_experimental:
        import cpufast_fast_ops  # noqa: F401
        from cpufast_fast_ar import build

        head_major = cook_phase = fft_layout = packed_fft = fused_finish = True
    else:
        from cpufast_ar_cpp import build
    from cpufast_fft import install_fft
    from cpufast_predictor import NativePredictor
    from cpufast_winograd import install_winograd

    decoder = pipeline.t2s_model.model
    before = {
        "ar_tensors": ar_digest(decoder.t2s_transformer),
        "ar_parameters": parameter_digest(decoder),
        "vits_parameters": parameter_digest(pipeline.vits_model),
    }
    # build() fails closed if any coefficient cannot be represented exactly.
    decoder.t2s_transformer, audit = build(
        decoder.t2s_transformer,
        native_attention=native_attention,
        head_major=head_major,
    )
    decoder.ar_predict_layer = NativePredictor(decoder.ar_predict_layer)
    if cook_phase:
        from cpufast_cooktoom import CookToomPhase

        for index, layer in enumerate(pipeline.vits_model.dec.ups):
            if layer.kernel_size[0] // layer.stride[0] > 1:
                pipeline.vits_model.dec.ups[index] = CookToomPhase(
                    layer, m=6 if layer.stride[0] >= 8 else 4, native=True
                )
    phase = install_subpixel(pipeline.vits_model.dec)
    fft = install_fft(
        pipeline.vits_model.dec.resblocks,
        backend="packed" if packed_fft else "fused",
        n1_selected=n1_fft_sizes,
    )
    winograd = install_winograd(pipeline.vits_model.dec.resblocks, native=True)
    if fft_layout:
        from cpufast_fft import FFTConv1d

        for module in pipeline.vits_model.dec.modules():
            if isinstance(module, FFTConv1d):
                module.output_layout = True
    if cook_resblock:
        from cpufast_cooktoom import CookToomConv1d
        from cpufast_winograd import WinogradConv1d

        for block in pipeline.vits_model.dec.resblocks:
            for group in (block.convs1, block.convs2):
                for index, module in enumerate(group):
                    if isinstance(module, WinogradConv1d):
                        group[index] = CookToomConv1d(
                            module, m=cook_resblock, native=True
                        )
    if fused_finish:
        from cpufast_fused_blocks import install_fused_blocks

        install_fused_blocks(pipeline.vits_model.dec.resblocks)
        from cpufast_fft import FFTConv1d

        for module in pipeline.vits_model.dec.modules():
            if isinstance(module, FFTConv1d):
                module.fused_finish = True
                module.fast_half = fast_experimental
    taps = 0
    if tap_small:
        from cpufast_tap import install_small_taps

        taps = install_small_taps(
            pipeline.vits_model.dec.resblocks, packed_48=packed_small_48
        )
    from cpufast_fft import FFTConv1d
    from cpufast_tap import PackedSmallTap

    n1_fft_count = (
        sum(
            isinstance(m, FFTConv1d)
            and m.weight.shape[-1] == 11
            and m.dilation == 5
            and (
                (m.weight.shape[1] == 384 and m.n_fft == 256)
                or (m.weight.shape[1] == 96 and m.n_fft == 512)
            )
            for m in pipeline.vits_model.dec.modules()
        )
        if n1_fft_sizes
        else 0
    )
    if n1_fft_sizes and not n1_fft_count:
        raise ValueError("No selected FFT sizes matched the model")

    packed_small_count = sum(
        isinstance(m, PackedSmallTap) for m in pipeline.vits_model.dec.modules()
    )
    if packed_small_48 and not packed_small_count:
        raise ValueError("No 48-channel packed convolution matched the model")
    after = {
        "ar_tensors": ar_digest(decoder.t2s_transformer),
        "ar_parameters": parameter_digest(decoder),
        "vits_parameters": parameter_digest(pipeline.vits_model),
    }
    if before != after:
        raise RuntimeError("An authoritative model parameter changed")
    runtime = CPUFastRuntime(
        pipeline, ar_threads=ar_threads, acoustic_threads=acoustic_threads
    )
    runtime.audit = {
        "profile": "config/cpufast_proxy.json",
        "before": before,
        "after": after,
        "parameters_unchanged": True,
        "ar_coefficients": audit,
        "phase_and_layout_replacements": phase,
        "fft_convolutions": fft,
        "winograd_convolutions": winograd,
        "tap_convolutions": taps,
        "tap_small": tap_small,
        "packed_small_48": packed_small_48,
        "n1_fft_sizes": n1_fft_sizes,
        "n1_fft_size_count": n1_fft_count,
        "packed_small_48_count": packed_small_count,
        "fast_experimental": fast_experimental,
        "arithmetic": "W8A8 AR and FP16 FFT accumulation"
        if fast_experimental
        else "FP32; exact FP16 AR storage cache only",
    }
    pipeline._cpufast_engine = runtime
    return runtime
