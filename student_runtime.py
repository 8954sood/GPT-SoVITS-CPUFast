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

    def __init__(
        self, teacher, branches, pairs=None, channel_keep=None, bottlenecks=None
    ):
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
        self.channel_keep = None
        self.bottlenecks = None
        if channel_keep is not None:
            self.select_channels(channel_keep)
        if bottlenecks:
            if pairs is not None:
                raise ValueError(
                    "Pair removal and internal bottlenecks must be separate"
                )
            self.select_bottlenecks(bottlenecks)
        self.train(teacher.training)

    def select_bottlenecks(self, selections):
        """Reduce only paired internal channels; every residual boundary stays C."""
        if self.bottlenecks is not None or self.channel_keep is not None:
            raise ValueError("Bottleneck selection requires a fresh full-width decoder")
        checked, seen = [], set()
        for item in selections:
            stage, branch, pair = (
                int(item[k]) for k in ("stage", "original_branch", "pair")
            )
            key = stage, branch, pair
            if key in seen or not 0 <= stage < self.num_upsamples:
                raise ValueError("Unique valid bottleneck locations required")
            if branch not in self.stage_branches[stage]:
                raise ValueError("Bottleneck branch is not retained")
            index = self.stage_offsets[stage] + self.stage_branches[stage].index(branch)
            block = self.resblocks[index]
            if not 0 <= pair < len(block.convs1) or len(block.convs1) != len(
                block.convs2
            ):
                raise ValueError("Valid Conv pair required")
            c1, c2 = block.convs1[pair], block.convs2[pair]
            for conv in (c1, c2):
                if not isinstance(conv, torch.nn.Conv1d) or conv.groups != 1:
                    raise ValueError("Raw ungrouped Conv1d pair required")
                if any(
                    hasattr(conv, k)
                    for k in ("weight_g", "weight_v", "parametrizations")
                ):
                    raise ValueError(
                        "Remove weight normalization before bottleneck selection"
                    )
            if c1.in_channels != c2.out_channels or c1.out_channels != c2.in_channels:
                raise ValueError(
                    "Conv pair must preserve its residual channel boundary"
                )
            keep = checked_keep(
                item["keep"], c1.out_channels, "pair internal channel keep"
            )
            checked.append(
                (
                    index,
                    pair,
                    keep,
                    dict(
                        item,
                        stage=stage,
                        original_branch=branch,
                        pair=pair,
                        keep=list(keep),
                    ),
                )
            )
            seen.add(key)
        # Validate every request before changing the independent selected blocks.
        blocks = {
            index: copy.deepcopy(self.resblocks[index]) for index, _, _, _ in checked
        }
        for index, pair, keep, _ in checked:
            block = blocks[index]
            c1, c2 = block.convs1[pair], block.convs2[pair]
            ix = torch.tensor(keep, device=c1.weight.device)
            c1.weight = torch.nn.Parameter(
                c1.weight.detach().index_select(0, ix).clone(),
                requires_grad=c1.weight.requires_grad,
            )
            if c1.bias is not None:
                c1.bias = torch.nn.Parameter(
                    c1.bias.detach().index_select(0, ix).clone(),
                    requires_grad=c1.bias.requires_grad,
                )
            c2.weight = torch.nn.Parameter(
                c2.weight.detach().index_select(1, ix).clone(),
                requires_grad=c2.weight.requires_grad,
            )
            c1.out_channels = c2.in_channels = len(keep)
        for index, block in blocks.items():
            self.resblocks[index] = block
        self.bottlenecks = tuple(item for _, _, _, item in checked)

    def select_channels(self, keep):
        """Slice a fresh decoder copy; widths include stem and every upsample stage."""
        if self.channel_keep is not None:
            raise ValueError("Channel selection may be applied only once")
        if self.bottlenecks is not None:
            raise ValueError("Global width and bottleneck selection must be separate")
        widths = [self.conv_pre.out_channels] + [m.out_channels for m in self.ups]
        if len(keep) != len(widths):
            raise ValueError("Channel IDs required for stem and every stage")
        selected = tuple(
            checked_keep(ids, width, "decoder channel keep")
            for ids, width in zip(keep, widths)
        )

        def sliced(module, inputs, outputs):
            if not isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)) or (
                module.groups != 1
            ):
                raise ValueError("Raw ungrouped decoder convolutions required")
            if hasattr(module, "weight_g") or hasattr(module, "weight_v"):
                raise ValueError("Remove weight normalization before slicing")
            clone = copy.deepcopy(module)
            source = module.weight.detach()
            ix = torch.tensor(inputs, device=source.device)
            ox = torch.tensor(outputs, device=source.device)
            if isinstance(module, torch.nn.ConvTranspose1d):
                weight = source.index_select(0, ix).index_select(1, ox)
            else:
                weight = source.index_select(0, ox).index_select(1, ix)
            clone.weight = torch.nn.Parameter(
                weight.clone(), requires_grad=module.weight.requires_grad
            )
            if module.bias is not None:
                clone.bias = torch.nn.Parameter(
                    module.bias.detach().index_select(0, ox).clone(),
                    requires_grad=module.bias.requires_grad,
                )
            clone.in_channels, clone.out_channels = len(inputs), len(outputs)
            return clone

        self.conv_pre = sliced(
            self.conv_pre, tuple(range(self.conv_pre.in_channels)), selected[0]
        )
        if hasattr(self, "cond"):
            self.cond = sliced(
                self.cond, tuple(range(self.cond.in_channels)), selected[0]
            )
        self.ups = torch.nn.ModuleList(
            [sliced(up, selected[i], selected[i + 1]) for i, up in enumerate(self.ups)]
        )
        blocks = []
        for stage, branches in enumerate(self.stage_branches):
            indices = selected[stage + 1]
            for j in range(len(branches)):
                block = copy.deepcopy(self.resblocks[self.stage_offsets[stage] + j])
                for group in ("convs1", "convs2"):
                    setattr(
                        block,
                        group,
                        torch.nn.ModuleList(
                            [
                                sliced(conv, indices, indices)
                                for conv in getattr(block, group)
                            ]
                        ),
                    )
                blocks.append(block)
        self.resblocks = torch.nn.ModuleList(blocks)
        self.conv_post = sliced(
            self.conv_post, selected[-1], tuple(range(self.conv_post.out_channels))
        )
        self.channel_keep = selected

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
