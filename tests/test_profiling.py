"""Diagnostic instrumentation leaves normal inference untouched and restores on error."""

import pytest
import torch

from cpufast_profiling import predictor_range, decoder_tap_ranges


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


def test_decoder_metadata_captures_fused_kwargs_and_actual_length():
    from cpufast_fft import FFTConv1d
    from cpufast_tap import TapConv1d

    layer = TapConv1d(
        FFTConv1d(torch.nn.Conv1d(24, 24, 7, padding=9, dilation=3).eval())
    ).eval()
    decoder = torch.nn.Sequential(layer)
    value = torch.randn(1, 24, 79)
    residual = torch.randn_like(value)
    with torch.no_grad():
        expected = layer(value, activate=True, residual=residual)
        with decoder_tap_ranges(decoder) as metadata:
            actual = layer(value, activate=True, residual=residual)
    assert torch.equal(actual, expected)
    assert "forward" not in layer.__dict__
    assert metadata[0]["dilation"] == 3
    assert metadata[0]["input_shapes"] == [{"shape": [1, 24, 79], "calls": 1}]
