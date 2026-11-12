import copy

import torch

from experiments.student.prune_approved import copied_decoder
from experiments.student.test_student_runtime import TinyTeacher
from student_runtime import StudentGenerator


def test_incremental_decoder_remaps_original_branch_ids_and_keeps_parent_immutable():
    teacher = TinyTeacher().eval()
    parent = StudentGenerator(copy.deepcopy(teacher), [[0, 2], [0, 1, 2]])
    with torch.no_grad():
        parent.conv_pre.weight.add_(0.25)
        parent.resblocks[1].weight.add_(0.5)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    candidate = copied_decoder(teacher, parent, [[2], [0, 2]])
    assert torch.equal(candidate.conv_pre.weight, parent.conv_pre.weight)
    assert torch.equal(candidate.resblocks[0].weight, parent.resblocks[1].weight)
    assert torch.equal(candidate.resblocks[1].weight, parent.resblocks[2].weight)
    assert torch.equal(candidate.resblocks[2].weight, parent.resblocks[4].weight)
    assert candidate.conv_pre.weight is not parent.conv_pre.weight
    assert candidate(torch.randn(1, 2, 11)).shape == (1, 1, 11)
    for name, tensor in parent.state_dict().items():
        assert torch.equal(tensor, before[name])
