import copy

import torch

from experiments.student.channel_reconstruction import projection, reconstruct
from experiments.student.test_decoder_channels import teacher
from student_runtime import StudentGenerator


def test_correlated_omitted_channels_are_recovered_without_altering_retained_ones():
    torch.manual_seed(98)
    selected = torch.randn(6, 512)
    full = torch.cat([selected, torch.randn(6, 6) @ selected])
    covariance = full @ full.t() / full.shape[1]
    original = covariance.clone()
    fitted = projection(covariance, list(range(6)))
    assert torch.equal(fitted[:6], torch.eye(6))
    weight = torch.randn(4, 12)
    expected = weight @ full
    sliced = weight[:, :6] @ selected
    reconstructed = (weight @ fitted) @ selected
    assert (reconstructed - expected).square().mean() < (
        sliced - expected
    ).square().mean() * 1e-4
    assert torch.equal(covariance, original)
    assert torch.isfinite(projection(torch.zeros(8, 8), [0, 2, 4, 6])).all()


def test_transposed_convolution_recovers_correlated_inputs_and_preserves_parent():
    torch.manual_seed(42)
    parent = StudentGenerator(teacher(), [[0, 2], [1, 2]])
    original = parent.ups[0].weight.clone()
    candidate = copy.deepcopy(parent)
    candidate.select_channels([list(range(8)), list(range(4)), list(range(2))])
    retained = torch.randn(8, 256)
    full = torch.cat([retained, torch.randn(8, 8) @ retained])
    before = candidate.ups[0](retained.unsqueeze(0))
    expected = parent.ups[0](full.unsqueeze(0))[:, :4]
    reconstruct(parent, candidate, {"ups.0": full @ full.t() / full.shape[1]})
    actual = candidate.ups[0](retained.unsqueeze(0))
    assert (actual - expected).square().mean() < (
        before - expected
    ).square().mean() * 1e-4
    assert torch.equal(parent.ups[0].weight, original)
