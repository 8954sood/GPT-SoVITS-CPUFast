import pytest
import torch

from cpufast_optimizations import parameter_digest
from cpufast_winograd import WinogradConv1d


@pytest.mark.parametrize("dilation", [1, 3, 5])
@pytest.mark.parametrize("bias", [False, True])
def test_dilated_winograd_boundary_tiles_and_parameters(dilation, bias):
    torch.manual_seed(31)
    original = torch.nn.Conv1d(
        8, 12, 3, padding=dilation, dilation=dilation, bias=bias
    ).eval()
    before = parameter_digest(original)
    winograd = WinogradConv1d(original).eval()
    with torch.no_grad():
        for length in (1, 3, 7, 31, 129):
            x = torch.randn(2, 8, length)
            torch.testing.assert_close(winograd(x), original(x), rtol=5e-5, atol=5e-6)
    assert winograd.weight is original.weight
    assert parameter_digest(winograd) == parameter_digest(original) == before


def test_packed_small_winograd_geometry_cache_and_fallback():
    from cpufast_winograd import PackedSmallWinograd

    original = torch.nn.Conv1d(24, 24, 3, dilation=3, padding=3).eval()
    candidate = PackedSmallWinograd(original).eval()
    x, residual = torch.randn(1, 24, 64), torch.randn(1, 24, 64)
    with torch.no_grad():
        for _ in range(2):
            expected = torch.nn.functional.leaky_relu(original(x), 0.1) + residual
            torch.testing.assert_close(
                candidate(x, activate=True, residual=residual),
                expected,
                atol=3e-6,
                rtol=1e-5,
            )
            original.weight.add_(0.01)
    assert candidate.weight is original.weight
    candidate.train()
    candidate(x.requires_grad_()).sum().backward()
    assert x.grad is not None
