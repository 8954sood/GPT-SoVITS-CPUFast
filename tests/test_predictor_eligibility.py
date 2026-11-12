"""Avoid repeated lossy-eligibility work while invalidating on parameter changes."""

from unittest.mock import patch

import torch
import torch.nn.functional as F

from cpufast_predictor import NativePredictor


def predictor():
    torch.manual_seed(28)
    model = NativePredictor(torch.nn.Linear(16, 24).eval(), cache_ineligible=True)
    x = torch.randn(1, 1, 16)
    return model, x


def test_general_fp32_retains_exact_linear_without_repeated_conversion_checks():
    model, x = predictor()
    expected = F.linear(x, model.weight, model.bias)
    equal = torch.equal
    with torch.no_grad(), patch("cpufast_predictor.torch.equal", wraps=equal) as check:
        outputs = [model(x) for _ in range(4)]
        assert check.call_count == 1
    assert all(torch.equal(y, expected) for y in outputs)


def test_version_bias_parameter_and_storage_changes_invalidate_eligibility():
    model, x = predictor()
    equal = torch.equal
    with torch.no_grad(), patch("cpufast_predictor.torch.equal", wraps=equal) as check:
        model(x)
        model.weight.add_(0.00123)
        first = model(x)
        model.bias.add_(0.013)
        second = model(x)
        model.weight = torch.nn.Parameter(model.weight.detach().clone() + 0.017)
        third = model(x)
        model.weight.data = model.weight.detach().clone() + 0.0013
        fourth = model(x)
        assert check.call_count == 5
    assert not torch.equal(first, second)
    assert not torch.equal(second, third)
    assert torch.equal(fourth, F.linear(x, model.weight, model.bias))


def test_changed_coefficients_can_become_native_eligible_and_back():
    model, x = predictor()
    calls = []

    def native(input, weight, bias):
        calls.append(True)
        restored = weight.transpose(1, 2).reshape(-1, weight.shape[1]).float()
        return F.linear(input, restored, bias)

    with (
        torch.no_grad(),
        patch.object(torch.ops.cpufast_cpp, "lossless_linear", native, create=True),
    ):
        model(x)
        assert not calls
        model.weight.copy_(model.weight.half().float())
        native_output = model(x)
        assert len(calls) == 1
        assert torch.equal(native_output, F.linear(x, model.weight, model.bias))
        model.weight.add_(0.000123)
        fallback_output = model(x)
        assert len(calls) == 1
        assert torch.equal(fallback_output, F.linear(x, model.weight, model.bias))
        model(x)
        assert len(calls) == 1


def test_disabled_option_retains_baseline_eligibility_checks():
    model, x = predictor()
    model.cache_ineligible = False
    equal = torch.equal
    with torch.no_grad(), patch("cpufast_predictor.torch.equal", wraps=equal) as check:
        for _ in range(3):
            model(x)
        assert check.call_count == 3
