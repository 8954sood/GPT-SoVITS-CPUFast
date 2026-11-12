import pytest
import torch

from experiments.student.prepare_half_cache import round_matrices


def checkpoint():
    names = [
        "self_attn.in_proj_weight",
        "self_attn.out_proj.weight",
        "linear1.weight",
        "linear2.weight",
    ]
    weights = {f"model.h.layers.0.{name}": torch.randn(8, 8) for name in names}
    weights.update(
        {
            "model.ar_predict_layer.weight": torch.randn(16, 8),
            "model.h.layers.0.linear1.bias": torch.randn(8),
            "model.h.layers.0.norm1.weight": torch.randn(8),
            "model.ar_audio_embedding.weight": torch.randn(16, 8),
        }
    )
    return {"config": {"model": {"n_layer": 1}}, "weight": weights}


def test_half_copy_rounds_only_selected_matrices_and_keeps_parent_immutable():
    source = checkpoint()
    before = {k: v.clone() for k, v in source["weight"].items()}
    result, audit = round_matrices(source)
    selected = {row["name"] for row in audit}
    assert len(selected) == 5
    assert sum(row["changed_elements"] for row in audit) > 0
    for key, original in source["weight"].items():
        assert torch.equal(original, before[key])
        expected = original.half().float() if key in selected else original
        assert torch.equal(result["weight"][key], expected)
    assert not result["student_metadata"]["quality_approved"]


def test_half_copy_rejects_overflow_and_incomplete_matrix_set():
    source = checkpoint()
    source["weight"]["model.ar_predict_layer.weight"][0, 0] = 1e6
    with pytest.raises(ValueError, match="overflow"):
        round_matrices(source)
    del source["weight"]["model.ar_predict_layer.weight"]
    with pytest.raises(ValueError, match="Expected"):
        round_matrices(source)
