"""Diagnostic module labels, installed exclusively for excluded diagnostic inference."""

from contextlib import ExitStack, contextmanager

import torch


@contextmanager
def module_range(module, label):
    original = module.forward
    overridden = "forward" in module.__dict__
    shapes = {}
    metadata = {
        "scope": "Extra profiled synthesis only; no instrumentation in timed samples",
        "range": label,
        "weight_shape": list(module.weight.shape),
        "weight_dtype": str(module.weight.dtype),
        "negative_eligibility_cache": getattr(module, "cache_ineligible", False),
    }

    def forward(x, *args, **kwargs):
        shape = tuple(x.shape)
        shapes[shape] = shapes.get(shape, 0) + 1
        with torch.profiler.record_function(metadata["range"]):
            return original(x, *args, **kwargs)

    module.forward = forward
    try:
        yield metadata
    finally:
        if overridden:
            module.forward = original
        else:
            del module.forward
        metadata["input_shapes"] = [
            {"shape": list(shape), "calls": count} for shape, count in shapes.items()
        ]


@contextmanager
def predictor_range(module):
    with module_range(module, "cpufast_predictor_projection") as metadata:
        yield metadata
    metadata["packed_half_classifier"] = getattr(module, "_packed", None) is not None


@contextmanager
def decoder_tap_ranges(decoder):
    from cpufast_tap import TapConv1d

    metadata = []
    with ExitStack() as stack:
        for name, module in decoder.named_modules():
            if isinstance(module, TapConv1d):
                channels, _, kernel = module.weight.shape
                label = f"cpufast_decoder_tap_C{channels}_K{kernel}_D{module.dilation}"
                row = stack.enter_context(module_range(module, label))
                row.update(
                    {
                        "module": name,
                        "dilation": module.dilation,
                        "padding": module.padding,
                    }
                )
                metadata.append(row)
        yield metadata
