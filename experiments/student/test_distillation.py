import copy

import torch

from experiments.student.distill_ar import MAPPING, ARStudent, distill_loss


def small_teacher():
    width, ffn = 4, 8
    state = {"model.ar_predict_layer.weight": torch.randn(7, width)}
    shapes = {
        "qkv": (3 * width, width),
        "out": (width, width),
        "ff1": (ffn, width),
        "ff2": (width, ffn),
        "norm1": (width,),
        "norm2": (width,),
    }
    for i in range(3):
        for local, remote in MAPPING.items():
            shape = shapes[local]
            for field in ("weight", "bias"):
                key = (
                    f"model.h.layers.{i}.{remote}"
                    + ("_" if local == "qkv" else ".")
                    + field
                )
                dims = shape if field == "weight" else (shape[0],)
                state[key] = torch.randn(*dims)
    return {
        "config": {
            "model": {
                "n_layer": 3,
                "hidden_dim": width,
                "head": 2,
                "linear_units": ffn,
                "vocab_size": 7,
            }
        },
        "weight": state,
    }


def test_kl_identity_and_finite_gradients():
    teacher = torch.randn(8, 7)
    student = teacher.clone().requires_grad_()
    loss = distill_loss(student, teacher, 2.0)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    shifted = student + torch.randn_like(student) * 0.1
    distill_loss(shifted, teacher, 2.0).backward()
    assert torch.isfinite(student.grad).all()


def test_trainable_subset_checkpoint_roundtrip_and_teacher_is_immutable():
    original = small_teacher()
    before = copy.deepcopy(original)
    student = ARStudent(original, [0, 2])
    inputs = torch.randn(1, 5, 4)
    mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
    output = student(inputs, mask, 2)
    checkpoint = student.checkpoint(original)
    restored = ARStudent(checkpoint, [0, 1])
    torch.testing.assert_close(output, restored(inputs, mask, 2))
    assert checkpoint["config"]["model"]["n_layer"] == 2
    for key in original["weight"]:
        assert torch.equal(before["weight"][key], original["weight"][key])
