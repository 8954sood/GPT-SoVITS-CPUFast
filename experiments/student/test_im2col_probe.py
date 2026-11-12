import torch
import torch.nn.functional as F

from experiments.rtf07.bench_im2col_tap import tap_im2col
from experiments.rtf07.bench_tap_tiles import tap_tiles


def test_im2col_tap_order_dilation_bias_and_ragged_tiles():
    torch.manual_seed(911)
    for channels, kernel, dilation, length in [(8, 7, 1, 73), (24, 11, 5, 97)]:
        weight = torch.randn(channels, channels, kernel)
        before = weight.clone()
        value = torch.randn(1, channels, length)
        padding = dilation * (kernel - 1) // 2
        padded = F.pad(value.squeeze(0), (padding, padding))
        packed = weight.permute(2, 0, 1).contiguous()
        for bias in (None, torch.randn(channels)):
            baseline = tap_tiles(packed, padded, bias, length, dilation)
            direct = F.conv1d(value, weight, bias, padding=padding, dilation=dilation)
            for tile in (0, 16, 32, 128):
                actual = tap_im2col(
                    weight.reshape(channels, -1),
                    padded,
                    bias,
                    length,
                    kernel,
                    dilation,
                    tile,
                )
                assert actual.shape == baseline.shape
                torch.testing.assert_close(actual, baseline, atol=2e-4, rtol=2e-5)
                torch.testing.assert_close(actual, direct, atol=2e-4, rtol=2e-5)
        assert torch.equal(weight, before)
