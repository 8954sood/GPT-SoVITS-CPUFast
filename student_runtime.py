"""Explicit research students; original checkpoint files are never rewritten."""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def checked_keep(keep, total, label):
    keep = tuple(int(i) for i in keep)
    if not keep or keep != tuple(sorted(set(keep))) or keep[0] < 0 or keep[-1] >= total:
        raise ValueError(
            f"{label} must be nonempty, unique, increasing IDs within 0..{total - 1}"
        )
    return keep


def ar_subset(transformer, keep):
    from AR.models.t2s_model import T2STransformer

    keep = checked_keep(keep, len(transformer.blocks), "AR keep")
    return T2STransformer(len(keep), [transformer.blocks[i] for i in keep])


class StudentGenerator(torch.nn.Module):
    """Preserve stage widths/rates; independently select branches at each stage."""

    def __init__(self, teacher, branches, pairs=None):
        super().__init__()
        if len(branches) != teacher.num_upsamples:
            raise ValueError("One branch selection is required per upsampling stage")
        self.num_upsamples = teacher.num_upsamples
        self.stage_branches = tuple(
            checked_keep(k, teacher.num_kernels, "branch keep") for k in branches
        )
        self.conv_pre, self.ups, self.conv_post = (
            teacher.conv_pre,
            teacher.ups,
            teacher.conv_post,
        )
        if hasattr(teacher, "cond"):
            self.cond = teacher.cond
        self.resblocks = torch.nn.ModuleList()
        self.stage_offsets = []
        for stage, keep in enumerate(self.stage_branches):
            self.stage_offsets.append(len(self.resblocks))
            for branch in keep:
                block = teacher.resblocks[stage * teacher.num_kernels + branch]
                if pairs is not None:
                    # Independent module graph: later changes do not mutate teacher lists.
                    block = copy.deepcopy(block)
                    indices = checked_keep(
                        pairs[stage][branch], len(block.convs1), "pair keep"
                    )
                    block.convs1 = torch.nn.ModuleList(
                        [block.convs1[i] for i in indices]
                    )
                    block.convs2 = torch.nn.ModuleList(
                        [block.convs2[i] for i in indices]
                    )
                self.resblocks.append(block)
        self.train(teacher.training)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for stage in range(self.num_upsamples):
            x = self.ups[stage](F.leaky_relu(x, 0.1))
            offset = self.stage_offsets[stage]
            count = len(self.stage_branches[stage])
            values = [self.resblocks[offset + j](x) for j in range(count)]
            merged = values[0]
            for value in values[1:]:
                merged = merged + value
            x = merged * (1.0 / count)
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


def export_ar_student(source, output, keep):
    """Materialize remapped original layers into a separate loadable checkpoint."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError(
            "Student checkpoint must be a fresh path distinct from teacher"
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    keep = checked_keep(keep, checkpoint["config"]["model"]["n_layer"], "AR keep")
    remap = {old: new for new, old in enumerate(keep)}
    weights = {}
    for name, value in checkpoint["weight"].items():
        match = re.match(r"(.*h\.layers\.)(\d+)(\..*)", name)
        if match:
            old = int(match[2])
            if old not in remap:
                continue
            name = f"{match[1]}{remap[old]}{match[3]}"
        weights[name] = value
    config = copy.deepcopy(checkpoint["config"])
    config["model"]["n_layer"] = len(keep)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "weight": weights,
        "config": config,
        "student_metadata": {
            "teacher_sha256": digest,
            "original_layer_ids": keep,
            "training": "untrained structured layer pruning; not quality approved",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(result, stream)
    return result["student_metadata"]
