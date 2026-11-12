"""Opt-in execution equivalent to approved P, validated separately before N1."""

from cpufast_fft import FFTConv1d
from cpufast_fft_half_runtime import enable_fft_half_products


def enable_fft_half_stream(decoder):
    audit = enable_fft_half_products(decoder)
    selected = {row["module"] for row in audit}
    for name, module in decoder.named_modules():
        if name in selected:
            if not isinstance(module, FFTConv1d):
                raise RuntimeError("Half-stream target changed after validation")
            module.half_stream_products = True
    for row in audit:
        row.update(
            half_stream_products=True,
            intermediate_product_scope="Four-frequency thread-local FP32 scratch",
            full_frequency_products_materialized=False,
        )
    return audit
