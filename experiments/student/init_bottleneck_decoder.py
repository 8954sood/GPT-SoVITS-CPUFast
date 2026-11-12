"""Initialize selected residual-pair bottlenecks from immutable approved K.

Only existing JA/KO/EN training contexts rank channels. Outer channels, branches,
upsamplers, residual boundaries and the last C24 stage remain unchanged.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from cpufast_optimizations import parameter_digest
from experiments.student.channel_reconstruction import projection
from experiments.student.distill_decoder import decoder_from_teacher
from experiments.student.init_thin_decoder import sha
from student_runtime import StudentGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("approved", "s2", "manifest", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--branches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--inner", type=int, default=36)
    parser.add_argument("--contexts-per-language", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.pairs <= 3:
        parser.error("Fresh output and conservative 1..3 pair budget required")
    if args.contexts_per_language < 1 or args.inner < 1:
        parser.error("Positive context count and inner width required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    release = json.loads((args.approved / "manifest.json").read_text())
    if not release["quality_approved"]:
        raise ValueError("Approved parent required")
    for name, meta in release["files"].items():
        if sha(args.approved / name) != meta["sha256"]:
            raise ValueError("Approved parent changed")
    corpus = json.loads(args.manifest.read_text())
    if not corpus["teacher_unchanged"] or not corpus["collection_complete"]:
        raise ValueError("Immutable completed corpus required")
    teacher = decoder_from_teacher(args.s2, args.manifest.parent / "teacher_decoder.pt")
    spec = json.loads((args.approved / "student_spec.json").read_text())
    if spec.get("decoder_channel_keep"):
        raise ValueError("Internal pairs require unchanged approved outer widths")
    parent = StudentGenerator(
        copy.deepcopy(teacher),
        spec["decoder_branches"],
        bottlenecks=spec.get("decoder_bottlenecks"),
    )
    if not 0 <= args.stage < parent.num_upsamples - 1:
        raise ValueError("Keep the final C24 stage unchanged")
    if len(set(args.branches)) != len(args.branches) or any(
        b not in parent.stage_branches[args.stage] for b in args.branches
    ):
        raise ValueError("Unique retained branch IDs required")
    parent.load_state_dict(
        torch.load(args.approved / "decoder.pt", weights_only=True)["state_dict"]
    )
    parent.eval().requires_grad_(False)
    before = parameter_digest(parent)
    records = []
    for lang in ("ja", "ko", "en"):
        available = [
            r
            for r in corpus["records"]
            if r["split"] == "train" and r["language"] == lang
        ]
        if len(available) < args.contexts_per_language:
            raise ValueError("Insufficient existing training contexts")
        records.extend(available[: args.contexts_per_language])
    existing_locations = {
        (r["stage"], r["original_branch"], r["pair"])
        for r in (parent.bottlenecks or ())
    }
    candidates, hooks = [], []
    for branch in args.branches:
        block = parent.resblocks[
            parent.stage_offsets[args.stage]
            + parent.stage_branches[args.stage].index(branch)
        ]
        for pair, (c1, c2) in enumerate(zip(block.convs1, block.convs2)):
            if (args.stage, branch, pair) in existing_locations:
                continue
            width = c1.out_channels
            if (
                c1.in_channels != width
                or c2.in_channels != width
                or not args.inner < width
            ):
                raise ValueError(
                    "New internal pair must reduce a full-width residual pair"
                )
            row = {
                "stage": args.stage,
                "outer_width": width,
                "inner_width": args.inner,
                "original_branch": branch,
                "pair": pair,
                "kernel": c1.kernel_size[0],
                "dilation": c1.dilation[0],
                "covariance": torch.zeros(width, width),
                "activity": torch.zeros(width),
                "consumer": c2.weight.square().sum((0, 2)).sqrt(),
            }
            candidates.append(row)

            def capture(_, inputs, row=row, width=width):
                value = inputs[0].detach().transpose(0, 1).reshape(width, -1)
                row["activity"].add_(value.square().mean(1).sqrt() / len(records))
                positions = torch.linspace(
                    0, value.shape[1] - 1, min(512, value.shape[1])
                ).long()
                sample = value[:, positions]
                row["covariance"].add_(
                    (sample @ sample.t()) / (sample.shape[1] * len(records))
                )

            hooks.append(c2.register_forward_pre_hook(capture))
    with torch.no_grad():
        for record in records:
            context = torch.load(
                args.manifest.parent / record["decoder_context"], weights_only=True
            )
            parent(context["latent"], context["conditioning"])
    for hook in hooks:
        hook.remove()
    for row in candidates:
        scores = row["activity"] * row["consumer"]
        row["keep"] = sorted(torch.topk(scores, args.inner).indices.tolist())
        retained = scores[row["keep"]].sum()
        row["removed_score_fraction"] = float(
            1 - retained / scores.sum().clamp_min(1e-12)
        )
        row["selection_priority"] = row["removed_score_fraction"] / row["kernel"]
    if len(candidates) < args.pairs:
        raise ValueError("Insufficient unmodified candidate pairs")
    chosen = sorted(candidates, key=lambda r: r["selection_priority"])[: args.pairs]
    selections = [
        {
            k: r[k]
            for k in (
                "stage",
                "original_branch",
                "pair",
                "keep",
                "kernel",
                "dilation",
                "outer_width",
                "inner_width",
                "removed_score_fraction",
            )
        }
        for r in chosen
    ]
    candidate = copy.deepcopy(parent)
    candidate.select_bottlenecks(selections)
    # Fold predictable omitted post-activation channels into Conv2 weights.
    # This changes initialization only: no projection exists at inference.
    with torch.no_grad():
        for row in chosen:
            index = parent.stage_offsets[args.stage] + parent.stage_branches[
                args.stage
            ].index(row["original_branch"])
            original = parent.resblocks[index].convs2[row["pair"]]
            target = candidate.resblocks[index].convs2[row["pair"]]
            mapping = projection(row["covariance"], row["keep"])
            target.weight.copy_(torch.einsum("oik,ij->ojk", original.weight, mapping))
    assert parameter_digest(parent) == before
    spec.update(
        decoder_bottlenecks=list(candidate.bottlenecks),
        quality_approved=False,
        training=f"Selected stage{args.stage} internal pairs to{args.inner}; approved outer widths/branches/pairs preserved; CPU recovery required",
    )
    metadata = {
        "scope": "Conservative pair-only candidate; no listening approval or N1 speed claim",
        "parent_manifest_sha256": sha(args.approved / "manifest.json"),
        "parent_decoder_sha256": sha(args.approved / "decoder.pt"),
        "teacher_s2_sha256": sha(args.s2),
        "corpus_manifest_sha256": sha(args.manifest),
        "contexts": [
            {
                "path": r["decoder_context"],
                "sha256": sha(args.manifest.parent / r["decoder_context"]),
            }
            for r in records
        ],
        "initialization": "postactivation RMS × Conv2 consumer norm; rank requested new pairs by removed score fraction/kernel cost proxy; ridge1e-3 omitted-input reconstruction into Conv2 weights",
        "selected_pairs": selections,
        "preserved_parent_pairs": list(parent.bottlenecks or ()),
        "selection_stage": args.stage,
        "selection_inner_width": args.inner,
        "all_ranked_pairs": [
            {
                k: r[k]
                for k in (
                    "stage",
                    "original_branch",
                    "pair",
                    "kernel",
                    "removed_score_fraction",
                    "selection_priority",
                )
            }
            for r in candidates
        ],
        "original_parameters": sum(p.numel() for p in parent.parameters()),
        "candidate_parameters": sum(p.numel() for p in candidate.parameters()),
        "parent_unchanged": True,
    }
    args.output.mkdir(parents=True)
    torch.save(
        {
            "state_dict": candidate.state_dict(),
            "student_spec": spec,
            "quality_approved": False,
            "initialization": metadata,
        },
        args.output / "initial.pt",
    )
    (args.output / "student_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (args.output / "initialization.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
