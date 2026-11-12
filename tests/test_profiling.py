"""Diagnostic instrumentation leaves normal inference untouched and restores on error."""

import pytest
import torch

from cpufast_profiling import predictor_range


def test_profile_range_is_excluded_from_normal_inference_and_preserves_output():
    module = torch.nn.Linear(8, 16).eval()
    value = torch.randn(1, 1, 8)
    expected = module(value)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False
    ) as prof:
        module(value)
        with predictor_range(module) as metadata:
            actual = module(value)
        module(value)
    assert torch.equal(actual, expected)
    assert "forward" not in module.__dict__
    events = {e.key: e for e in prof.key_averages()}
    assert events[metadata["range"]].count == 1
    assert metadata["weight_shape"] == [16, 8]
    assert metadata["input_shapes"] == [{"shape": [1, 1, 8], "calls": 1}]


def test_profile_wrapper_restored_after_exception():
    module = torch.nn.Linear(8, 16).eval()
    with pytest.raises(RuntimeError):
        with predictor_range(module):
            raise RuntimeError("diagnostic failure")
    assert "forward" not in module.__dict__
