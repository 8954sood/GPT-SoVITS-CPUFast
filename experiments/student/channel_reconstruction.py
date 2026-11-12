"""Calibrate omitted-channel contributions into retained decoder input weights.

All fitting uses training activations. The result has exactly the same thin
shape and inference work as simple slicing; original parent weights stay fixed.
"""

import torch

from student_runtime import checked_keep


def projection(covariance, keep, ridge_ratio=1e-3):
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Square channel covariance required")
    keep = checked_keep(keep, covariance.shape[0], "projection channels")
    if not torch.isfinite(covariance).all() or ridge_ratio <= 0:
        raise ValueError("Finite covariance and positive ridge required")
    cov = covariance.double()
    ix = torch.tensor(keep, device=cov.device)
    reduced = cov.index_select(0, ix).index_select(1, ix)
    ridge = reduced.diag().mean().clamp_min(1e-12) * ridge_ratio
    fitted = torch.linalg.solve(
        reduced + ridge * torch.eye(len(keep), device=cov.device, dtype=cov.dtype),
        cov.index_select(0, ix),
    ).t()
    # Preserve retained channels exactly; estimate only omitted contributions.
    fitted[ix] = torch.eye(len(keep), device=cov.device, dtype=cov.dtype)
    return fitted.to(covariance.dtype)


def module_channels(model, name):
    keep = model.channel_keep
    if name.startswith("ups."):
        stage = int(name.split(".")[1])
        return keep[stage], keep[stage + 1]
    if name.startswith("resblocks."):
        index = int(name.split(".")[1])
        stage = max(
            i for i, offset in enumerate(model.stage_offsets) if offset <= index
        )
        return keep[stage + 1], keep[stage + 1]
    if name == "conv_post":
        return keep[-1], tuple(range(model.conv_post.out_channels))
    raise ValueError("Only internal decoder input reductions are reconstructed")


def reconstruct(parent, candidate, covariances):
    modules = dict(candidate.named_modules())
    rows = []
    with torch.no_grad():
        for name, covariance in covariances.items():
            source = parent.get_submodule(name)
            target = modules[name]
            inputs, outputs = module_channels(candidate, name)
            mapping = projection(covariance, inputs)
            ox = torch.tensor(outputs, device=source.weight.device)
            if isinstance(source, torch.nn.ConvTranspose1d):
                original = source.weight.index_select(1, ox)
                weight = torch.einsum("ij,iok->jok", mapping, original)
            else:
                original = source.weight.index_select(0, ox)
                weight = torch.einsum("oik,ij->ojk", original, mapping)
            if weight.shape != target.weight.shape or not torch.isfinite(weight).all():
                raise ValueError("Invalid reconstructed decoder weight")
            target.weight.copy_(weight)
            rows.append(
                {
                    "module": name,
                    "input_channels": source.in_channels,
                    "retained_inputs": len(inputs),
                    "ridge_ratio": 1e-3,
                    "mapping_rms": float(mapping.square().mean().sqrt()),
                }
            )
    return rows
