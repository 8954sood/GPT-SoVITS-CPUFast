import copy

import pytest
import torch

from experiments.student.distill_decoder import generate_with_features
from student_runtime import StudentGenerator


def teacher():
    from module.models import Generator

    result = Generator(
        4, "1", [3, 5, 7], [[1, 3, 5]] * 3, [2, 2], 16, [4, 4], gin_channels=3
    ).eval()
    result.remove_weight_norm()
    return result


def test_full_channel_selection_is_exact_and_parent_immutable():
    source = teacher()
    before = {k: v.clone() for k, v in source.state_dict().items()}
    branches = [[0, 2], [1, 2]]
    base = StudentGenerator(copy.deepcopy(source), branches)
    selected = StudentGenerator(
        source, branches, channel_keep=[list(range(n)) for n in (16, 8, 4)]
    )
    x, g = torch.randn(1, 4, 13), torch.randn(1, 3, 1)
    assert torch.equal(base(x, g), selected(x, g))
    assert selected.conv_pre.weight is not source.conv_pre.weight
    assert selected.ups[0].weight is not source.ups[0].weight
    for name, value in source.state_dict().items():
        assert torch.equal(value, before[name])


def test_nonconsecutive_channels_transpose_axes_features_and_finite_gradients():
    source = teacher()
    keep = [[0, 2, 3, 7, 8, 10, 11, 15], [0, 2, 5, 7], [1, 3]]
    model = StudentGenerator(source, [[0, 2], [1, 2]], channel_keep=keep)
    expected = source.ups[0].weight[keep[0]][:, keep[1]]
    assert torch.equal(model.ups[0].weight, expected)
    assert torch.equal(model.conv_pre.weight, source.conv_pre.weight[keep[0]])
    assert torch.equal(model.conv_post.weight, source.conv_post.weight[:, keep[-1]])
    x, g = torch.randn(1, 4, 13), torch.randn(1, 3, 1)
    audio, features = generate_with_features(model, x, g)
    assert audio.shape == source(x, g).shape
    assert torch.equal(audio, model(x, g))
    assert [f.shape[1] for f in features] == [4, 2]
    audio.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(ValueError, match="only once"):
        model.select_channels(keep)


def test_channel_selection_rejects_invalid_ids_before_teacher_changes():
    source = teacher()
    before = source.conv_pre.weight.clone()
    with pytest.raises(ValueError):
        StudentGenerator(source, [[0, 2], [1]], channel_keep=[[0], [0]])
    with pytest.raises(ValueError):
        StudentGenerator(source, [[0, 2], [1]], channel_keep=[[0, 0], [0], [0]])
    assert torch.equal(source.conv_pre.weight, before)
