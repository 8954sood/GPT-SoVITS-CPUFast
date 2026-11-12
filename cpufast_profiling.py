"""Classifier-only labels, installed exclusively for excluded diagnostic inference."""

from contextlib import contextmanager

import torch


@contextmanager
def predictor_range(module):
    original = module.forward
    overridden = "forward" in module.__dict__
    shapes = {}
    metadata = {
        "scope": "Extra profiled synthesis only; no instrumentation in timed samples",
        "range": "cpufast_predictor_projection",
        "weight_shape": list(module.weight.shape),
        "weight_dtype": str(module.weight.dtype),
        "negative_eligibility_cache": getattr(module, "cache_ineligible", False),
    }

    def forward(x):
        shape = tuple(x.shape)
        shapes[shape] = shapes.get(shape, 0) + 1
        with torch.profiler.record_function(metadata["range"]):
            return original(x)

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
        metadata["packed_half_classifier"] = (
            getattr(module, "_packed", None) is not None
        )
