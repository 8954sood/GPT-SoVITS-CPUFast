"""Explicit FFT-size scope is validated atomically before a derived cache can change."""
import pytest
import torch

from cpufast_fft import FFTConv1d
from cpufast_fft128_cin240_runtime import enable_fft128_cin240


def decoder():
    root = torch.nn.Module()
    root.resblocks = torch.nn.ModuleList()
    for kernel in (7, 11):
        branch = torch.nn.Module()
        branch.convs2 = torch.nn.ModuleList()
        for _ in range(3):
            layer = FFTConv1d(torch.nn.Conv1d(240, 384, kernel, padding=kernel//2).eval(), backend="packed")
            layer.output_layout = layer.stream_products = True
            branch.convs2.append(layer)
        root.resblocks.append(branch)
    return root


def test_only_six_cold_fft_modules_change_without_parameter_edits():
    root = decoder()
    before = [(p, p._version) for p in root.parameters()]
    rows = enable_fft128_cin240(root)
    assert len(rows) == 6 and all(r["parameter_identity_version_unchanged"] for r in rows)
    assert sum(r["derived_spectral_bytes_before"]-r["derived_spectral_bytes_after"] for r in rows) == 424673280
    assert all(layer.n_fft == 128 and layer._key is None for branch in root.resblocks for layer in branch.convs2)
    assert all(old is new and version == new._version for (old, version), new in zip(before, root.parameters()))


def test_warmed_or_incomplete_selection_fails_before_any_change():
    root = decoder()
    root.resblocks[1].convs2[2]._key = ("already_warmed",)
    with pytest.raises(ValueError, match="cold canonical"):
        enable_fft128_cin240(root)
    assert all(layer.n_fft == 256 for branch in root.resblocks for layer in branch.convs2)
