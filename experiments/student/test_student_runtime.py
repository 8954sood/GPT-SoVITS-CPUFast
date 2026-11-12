import hashlib
import tempfile
from pathlib import Path

import pytest
import torch

from student_runtime import StudentGenerator, checked_keep, export_ar_student


class TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_upsamples, self.num_kernels = 2, 3
        self.conv_pre = torch.nn.Conv1d(2, 2, 1)
        self.ups = torch.nn.ModuleList([torch.nn.Conv1d(2, 2, 1) for _ in range(2)])
        self.resblocks = torch.nn.ModuleList(
            [torch.nn.Conv1d(2, 2, 1) for _ in range(6)]
        )
        self.conv_post = torch.nn.Conv1d(2, 1, 1)

    def forward(self, x):
        x = self.conv_pre(x)
        for s, up in enumerate(self.ups):
            x = up(torch.nn.functional.leaky_relu(x, 0.1))
            x = sum(self.resblocks[s * 3 + j](x) for j in range(3)) / 3
        return torch.tanh(self.conv_post(torch.nn.functional.leaky_relu(x)))


def test_all_branches_reconstruct_teacher_and_nonuniform_stays_same_length():
    teacher = TinyTeacher().eval()
    x = torch.randn(1, 2, 13)
    student = StudentGenerator(teacher, [[0, 1, 2], [0, 1, 2]])
    torch.testing.assert_close(student(x), teacher(x))
    pruned = StudentGenerator(teacher, [[0, 2], [1]])
    assert len(pruned.resblocks) == 3
    assert pruned(x).shape == teacher(x).shape
    assert pruned.conv_pre.weight is teacher.conv_pre.weight
    assert len(teacher.resblocks) == 6


def test_invalid_ids_rejected():
    for keep in ([], [1, 1], [2, 1], [-1], [3]):
        with pytest.raises(ValueError):
            checked_keep(keep, 3, "test")


def test_ar_checkpoint_remap_is_loadable_and_preserves_teacher():
    with tempfile.TemporaryDirectory() as directory:
        source, output = Path(directory) / "teacher.pt", Path(directory) / "student.pt"
        torch.save(
            {
                "config": {"model": {"n_layer": 3}},
                "weight": {
                    "model.h.layers.0.linear1.weight": torch.ones(2, 2),
                    "model.h.layers.1.linear1.weight": torch.ones(2, 2) * 2,
                    "model.h.layers.2.linear1.weight": torch.ones(2, 2) * 3,
                    "model.ar_predict_layer.weight": torch.ones(4, 2),
                },
            },
            source,
        )
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        export_ar_student(source, output, [0, 2])
        saved = torch.load(output, weights_only=False)
        assert saved["config"]["model"]["n_layer"] == 2
        assert saved["weight"]["model.h.layers.1.linear1.weight"][0, 0] == 3
        assert "model.h.layers.2.linear1.weight" not in saved["weight"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        with pytest.raises(ValueError):
            export_ar_student(source, output, [0, 2])
