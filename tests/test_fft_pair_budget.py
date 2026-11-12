"""Validate analytical budgets and isolated removal math, not perceptual quality."""

import torch
import torch.nn.functional as F

from experiments.rtf07.rank_fft_pair_budget import budget


def test_actual_two_pair_spectral_MACs_match_rectangular_geometry():
    first = budget(192, 144, 11, 1, 18880, 256)
    last = budget(192, 144, 11, 5, 18880, 256)
    assert first["blocks"] == [77, 77]
    assert last["blocks"] == [92, 77]
    assert first["hops"] == [246, 246] and last["hops"] == [206, 246]
    assert first["spectral_MAC_before"] + last["spectral_MAC_before"] == 4608036864
    assert first["spectral_MAC_saved"] + last["spectral_MAC_saved"] == 1152009216
    assert first["cache_bytes_saved"] + last["cache_bytes_saved"] == 57065472


def test_removed_contribution_equals_raw_slice_residual_delta():
    torch.manual_seed(2026)
    hidden = torch.randn(1, 8, 137)
    weight = torch.randn(8, 8, 11)
    bias = torch.randn(8)
    residual = torch.randn(1, 8, 137)
    keep, removed = [0, 1, 2, 4, 5, 7], [3, 6]
    for dilation in (1, 3, 5):
        padding = dilation * 5
        parent = (
            F.conv1d(hidden, weight, bias, padding=padding, dilation=dilation)
            + residual
        )
        candidate = (
            F.conv1d(
                hidden[:, keep],
                weight[:, keep],
                bias,
                padding=padding,
                dilation=dilation,
            )
            + residual
        )
        delta = F.conv1d(
            hidden[:, removed],
            weight[:, removed],
            None,
            padding=padding,
            dilation=dilation,
        )
        torch.testing.assert_close(parent - candidate, delta, rtol=2e-5, atol=2e-5)
