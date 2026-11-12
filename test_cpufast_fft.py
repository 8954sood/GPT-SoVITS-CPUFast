import pytest
import torch

from cpufast_fft import FFTConv1d
from cpufast_optimizations import parameter_digest


@pytest.mark.parametrize(
    "kernel,dilation,n_fft", [(3, 1, 32), (7, 3, 64), (11, 5, 128)]
)
@pytest.mark.parametrize("bias", [False, True])
def test_overlap_save_matches_cross_correlation_and_preserves_parameters(
    kernel, dilation, n_fft, bias
):
    torch.manual_seed(13)
    original = torch.nn.Conv1d(
        8,
        12,
        kernel,
        padding=dilation * (kernel - 1) // 2,
        dilation=dilation,
        bias=bias,
    ).eval()
    before = parameter_digest(original)
    fft = FFTConv1d(original, n_fft).eval()
    assert fft.weight is original.weight and fft.bias is original.bias
    with torch.no_grad():
        for length in (1, 7, 129, 521):
            x = torch.randn(2, 8, length)
            torch.testing.assert_close(fft(x), original(x), rtol=3e-5, atol=3e-6)
    assert parameter_digest(fft) == before == parameter_digest(original)
    old = fft._wr
    with torch.no_grad():
        original.weight.add_(0.01)
        torch.testing.assert_close(fft(x), original(x), rtol=3e-5, atol=3e-6)
    assert fft._wr is not old


def test_fft_training_uses_original_differentiable_convolution():
    original = torch.nn.Conv1d(3, 5, 7, padding=3)
    fft = FFTConv1d(original, 32)
    x = torch.randn(1, 3, 17, requires_grad=True)
    fft(x).sum().backward()
    assert x.grad is not None and original.weight.grad is not None
    assert fft._wr is None


def test_n1_fft_selection_applies_in_nested_models():
    from cpufast_fft import FFTConv1d, install_fft

    model = torch.nn.Sequential(torch.nn.Sequential(
        torch.nn.Conv1d(384, 384, 11, padding=25, dilation=5),
        torch.nn.Conv1d(96, 96, 11, padding=25, dilation=5),
        torch.nn.Conv1d(192, 192, 11, padding=25, dilation=5),
    ))
    assert install_fft(model, n1_selected=True) == 3
    assert all(isinstance(m, FFTConv1d) for m in model[0])
    assert [m.n_fft for m in model[0]] == [256, 512, 256]
