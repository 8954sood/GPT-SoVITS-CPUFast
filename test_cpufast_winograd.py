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
