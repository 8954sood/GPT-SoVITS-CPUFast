"""Same-width pair and stage supervision without matching bottleneck internals."""

import torch
from torch.nn import functional as F

from student_runtime import StudentGenerator


def generate_with_boundaries(model, x, g, locations):
    wanted = {
        (int(p["stage"]), int(p["original_branch"]), int(p["pair"])) for p in locations
    }
    captured, features = {}, []
    x = model.conv_pre(x) + model.cond(g)
    for stage, up in enumerate(model.ups):
        x = up(F.leaky_relu(x, 0.1))
        branches = (
            model.stage_branches[stage]
            if isinstance(model, StudentGenerator)
            else range(model.num_kernels)
        )
        values = []
        for position, branch in enumerate(branches):
            index = (
                model.stage_offsets[stage] + position
                if isinstance(model, StudentGenerator)
                else stage * model.num_kernels + branch
            )
            block = model.resblocks[index]
            if any(key[:2] == (stage, branch) for key in wanted):
                value = x
                for pair, (c1, c2) in enumerate(zip(block.convs1, block.convs2)):
                    value = c2(F.leaky_relu(c1(F.leaky_relu(value, 0.1)), 0.1)) + value
                    key = stage, branch, pair
                    if key in wanted:
                        captured[key] = value.clone()
            else:
                value = block(x)
            values.append(value)
        merged = values[0]
        for value in values[1:]:
            merged = merged + value
        x = merged * (1.0 / len(values))
        features.append(x.clone())
    if set(captured) != wanted:
        raise ValueError("Every requested residual pair boundary must exist")
    return torch.tanh(model.conv_post(F.leaky_relu(x))), features, captured
