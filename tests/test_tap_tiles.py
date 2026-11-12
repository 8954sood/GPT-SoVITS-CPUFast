"""Tiling retains tap order, output layout, padding and authoritative coefficients."""

import torch
import torch.nn.functional as F

from experiments.rtf07.bench_tap_tiles import tap_tiles


def test_tiles_cover_ragged_lengths_dilation_and_bias_without_coefficient_mutation():
    torch.manual_seed(219)
    for channels, kernel, dilation, length in [
        (8, 7, 1, 71),
        (24, 11, 5, 91),
        (48, 7, 3, 83),
    ]:
        weight = torch.randn(channels, channels, kernel)
        before = weight.clone()
        packed = weight.permute(2, 0, 1).contiguous()
        value = torch.randn(1, channels, length)
        bias = torch.randn(channels)
        padding = dilation * (kernel - 1) // 2
        padded = F.pad(value.squeeze(0), (padding, padding))
        expected = tap_tiles(packed, padded, bias, length, dilation, 0)
        direct = F.conv1d(value, weight, bias, padding=padding, dilation=dilation)
        torch.testing.assert_close(expected, direct, rtol=2e-5, atol=1e-4)
        for tile in (16, 32, 128):
            actual = tap_tiles(packed, padded, bias, length, dilation, tile)
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-4)
        assert torch.equal(weight, before)
