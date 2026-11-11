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
):
    """Return a serial engine; every original learned coefficient is preserved.

    Instantiate/load the CPUFast TTS pipeline before calling this function.
    Recreate the engine after changing model checkpoints. The native library
    must be built for the installed Linux ARM64 PyTorch version.
    """
    existing = getattr(pipeline, "_cpufast_engine", None)
    if existing is not None:
        return existing
    if pipeline.configs.version != "v2ProPlus" or pipeline.configs.is_half:
        raise ValueError("This profile requires the FP32 V2ProPlus CPU pipeline")
    if str(pipeline.configs.device) != "cpu":
        raise ValueError("This profile is for CPU inference")
    if pipeline.vits_model.training or pipeline.t2s_model.model.training:
        raise ValueError("Load evaluation models before optimizing")
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
        pipeline.vits_model.dec.resblocks, backend="packed" if packed_fft else "fused"
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
    after = {
        "ar_tensors": ar_digest(decoder.t2s_transformer),
        "ar_parameters": parameter_digest(decoder),
        "vits_parameters": parameter_digest(pipeline.vits_model),
    }
    if before != after:
        raise RuntimeError("An authoritative model parameter changed")
    runtime = CPUFastRuntime(pipeline, ar_threads=1, acoustic_threads=1)
    runtime.audit = {
        "profile": "config/cpufast_proxy.json",
        "before": before,
        "after": after,
        "parameters_unchanged": True,
        "ar_coefficients": audit,
        "phase_and_layout_replacements": phase,
        "fft_convolutions": fft,
        "winograd_convolutions": winograd,
        "arithmetic": "FP32; exact FP16 AR storage cache only",
    }
    pipeline._cpufast_engine = runtime
    return runtime
