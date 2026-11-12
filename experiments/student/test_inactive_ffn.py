import torch

from experiments.student.analyze_inactive_ffn import protected_maxima


def test_retained_row_maxima_keep_original_int8_grid_with_ties_and_zero_rows():
    weight = torch.tensor([[1., -1., .2, .0], [.0, .2, -3., 3.], [.0, .0, .0, .0]])
    keep = protected_maxima(weight)
    assert keep.tolist() == [0, 2]
    original_scale = (weight.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
    selected_scale = (weight[:, keep].abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
    assert torch.equal(original_scale, selected_scale)
    assert torch.equal((weight / original_scale).round()[:, keep], (weight[:, keep] / selected_scale).round())
