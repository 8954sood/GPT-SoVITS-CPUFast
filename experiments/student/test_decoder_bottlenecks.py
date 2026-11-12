import copy

import pytest
import torch

from experiments.student.decoder_boundaries import generate_with_boundaries
from experiments.student.test_decoder_channels import teacher
from student_runtime import StudentGenerator


def test_internal_mapping_keeps_outer_graph_parent_and_deploy_forward():
    source = teacher()
    before = {k: v.clone() for k, v in source.state_dict().items()}
    selected = [
        {"stage": 0, "original_branch": 2, "pair": 1, "keep": [0, 2, 3, 5, 6, 7]}
    ]
    model = StudentGenerator(source, [[0, 2], [1, 2]], bottlenecks=selected)
    block, original = model.resblocks[1], source.resblocks[2]
    assert block.convs1[1].weight.shape == (6, 8, 7)
    assert torch.equal(
        block.convs1[1].weight, original.convs1[1].weight[[0, 2, 3, 5, 6, 7]]
    )
    assert torch.equal(
        block.convs1[1].bias, original.convs1[1].bias[[0, 2, 3, 5, 6, 7]]
    )
    assert torch.equal(
        block.convs2[1].weight, original.convs2[1].weight[:, [0, 2, 3, 5, 6, 7]]
    )
    assert block.convs2[1].out_channels == block.convs1[1].in_channels == 8
    assert block.convs1[1].dilation == original.convs1[1].dilation
    assert block.convs1[1].padding == original.convs1[1].padding
    x, g = torch.randn(2, 4, 13), torch.randn(2, 3, 1)
    audio, features, pairs = generate_with_boundaries(model, x, g, selected)
    assert torch.equal(audio, model(x, g))
    assert audio.shape == source(x, g).shape
    assert pairs[(0, 2, 1)].shape == (2, 8, 26)
    assert [v.shape[1] for v in features] == [8, 4]
    (audio.square().mean() + pairs[(0, 2, 1)].square().mean()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    for name, value in source.state_dict().items():
        assert torch.equal(value, before[name])
    restored = StudentGenerator(
        copy.deepcopy(source), [[0, 2], [1, 2]], bottlenecks=selected
    )
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored(x, g), model(x, g))


def test_full_keep_identity_teacher_boundary_clone_and_atomic_validation():
    source = teacher()
    selected = [{"stage": 0, "original_branch": 1, "pair": 0, "keep": list(range(8))}]
    model = StudentGenerator(
        copy.deepcopy(source), [[0, 1, 2]] * 2, bottlenecks=selected
    )
    x, g = torch.randn(1, 4, 9), torch.randn(1, 3, 1)
    assert torch.equal(model(x, g), source(x, g))
    audio, _, boundaries = generate_with_boundaries(source, x, g, selected)
    assert torch.equal(audio, source(x, g))
    stable = boundaries[(0, 1, 0)].clone()
    source(x * 2, g)
    assert torch.equal(boundaries[(0, 1, 0)], stable)
    fresh = StudentGenerator(copy.deepcopy(source), [[0, 1, 2]] * 2)
    before = fresh.resblocks[1].convs1[0].weight.clone()
    with pytest.raises(ValueError):
        fresh.select_bottlenecks([selected[0], dict(selected[0], pair=9)])
    assert fresh.bottlenecks is None
    assert torch.equal(fresh.resblocks[1].convs1[0].weight, before)
    with pytest.raises(ValueError):
        fresh.select_bottlenecks([dict(selected[0], keep=[0, 0])])


def test_extend_approved_pairs_preserves_weights_identity_and_atomic_validation():
    source = teacher()
    original = copy.deepcopy(source.state_dict())
    first = {"stage": 0, "original_branch": 2, "pair": 0, "keep": [0, 2, 3, 5, 6, 7]}
    second = {"stage": 0, "original_branch": 2, "pair": 1, "keep": [0, 2, 4, 6]}
    model = StudentGenerator(
        copy.deepcopy(source), [[0, 2], [1, 2]], bottlenecks=[first]
    )
    # A recovered parent need not equal the original teacher's slice.
    with torch.no_grad():
        model.resblocks[1].convs2[0].weight.add_(0.01)
    parent_state = copy.deepcopy(model.state_dict())
    candidate = copy.deepcopy(model)
    candidate.select_bottlenecks([second])
    assert candidate.bottlenecks == (first, second)
    assert torch.equal(
        candidate.resblocks[1].convs2[0].weight,
        parent_state["resblocks.1.convs2.0.weight"],
    )
    assert candidate.resblocks[1].convs1[1].weight.shape[:2] == (4, 8)
    for key, value in model.state_dict().items():
        assert torch.equal(value, parent_state[key])
    restored = StudentGenerator(
        copy.deepcopy(source), [[0, 2], [1, 2]], bottlenecks=[first, second]
    )
    restored.load_state_dict(candidate.state_dict())
    x, g = torch.randn(1, 4, 11), torch.randn(1, 3, 1)
    assert torch.equal(candidate(x, g), restored(x, g))
    before = copy.deepcopy(candidate.state_dict())
    with pytest.raises(ValueError):
        candidate.select_bottlenecks([dict(second, pair=2), first])
    assert candidate.bottlenecks == (first, second)
    for key, value in candidate.state_dict().items():
        assert torch.equal(value, before[key])
    for key, value in source.state_dict().items():
        assert torch.equal(value, original[key])
