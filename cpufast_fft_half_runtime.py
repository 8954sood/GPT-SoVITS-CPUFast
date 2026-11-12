"""Explicit approximate FFT-product candidate; never enabled by the engine default."""

from cpufast_fft import FFTConv1d


def enable_fft_half_products(decoder):
    """Change only large packed FFT products, preserving learned parameters.

    Input spectra and derived coefficient caches are converted to FP16, GEMM
    accumulates in FP16 and returns FP32 products. FFT transforms and convolution
    finishing stay FP32. Materialized products replace the FP32 streaming path.
    Validate every target before changing any flags.
    """
    modules = list(decoder.named_modules())
    if any(bool(getattr(module, "fast_half", False)) for _, module in modules):
        raise ValueError("FFT-product candidate requires initially FP32 acoustics")
    targets = []
    for name, module in modules:
        if not isinstance(module, FFTConv1d):
            continue
        cout, cin, kernel = module.weight.shape
        if min(cin, cout) <= 48:
            raise ValueError("Small convolutions must retain the approved Tap path")
        if module.backend != "packed" or not module.output_layout:
            raise ValueError("FFT-product candidate requires packed output layout")
        targets.append((name, module, int(cin), int(cout), int(kernel)))
    if not targets:
        raise ValueError("FFT-product candidate has no eligible large convolutions")
    audit = []
    for name, module, cin, cout, kernel in targets:
        audit.append(
            {
                "module": name,
                "input_channels": cin,
                "output_channels": cout,
                "kernel": kernel,
                "dilation": module.dilation,
                "n_fft": module.n_fft,
                "previous_stream_products": module.stream_products,
                "stream_products": False,
                "spectral_inputs_and_coefficients": "FP16 derived buffers",
                "product_accumulation": "FP16",
                "materialized_product_dtype": "FP32",
                "forward_inverse_fft_dtype": "FP32 / complex64",
            }
        )
        module.fast_half = True
        module.stream_products = False
    return audit
