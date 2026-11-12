"""Initialize one multi-pair student from training sensitivity and operation budget.

The budget is spectral MACs under the existing FFT-size heuristic, not RTF.
Independent pair errors do not predict the combined student's audible quality.
"""

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

import torch

from cpufast_optimizations import parameter_digest
from experiments.student.channel_reconstruction import projection
from experiments.student.distill_decoder import decoder_from_teacher
from experiments.student.init_thin_decoder import sha
from student_runtime import StudentGenerator


def actual_fft_budget(row):
    outer, inner = row["outer_channels"], row["inner_channels"]
    if not 0 < inner < outer or inner % 12 or len(row["keep"]) != inner:
        raise ValueError(
            "Actual channel IDs and native aligned internal width must agree"
        )
    if inner <= 48:
        raise ValueError("Avoid an unvalidated automatic switch to TapConv1d")
    length = row["budget"]["actual_n1_input_time"]
    kernel, dilation = row["kernel"], row["dilation"]
    sizes = [128 if outer >= 256 else 256, 128 if inner >= 256 else 256]
    after = 0
    for size, d in zip(sizes, [dilation, 1]):
        effective = d * (kernel - 1) + 1
        blocks = math.ceil((length + d * (kernel - 1)) / (size - effective + 1))
        after += 3 * (size // 2 + 1) * blocks * outer * inner
    before = row["baseline_spectral_MACs"]
    return {
        "before": before,
        "after": after,
        "saved": before - after,
        "expected_fft_sizes": sizes,
    }


def choose_plan(report, minimum_reduction, maximum_local_rmse):
    groups = {}
    for row in report["choices"]:
        key = row["stage"], row["original_branch"], row["pair"]
        groups.setdefault(key, [])
        if (
            row["relative_pair_output_RMSE"] <= maximum_local_rmse
            and row["inner_channels"] > 48
        ):
            groups[key].append(dict(row, execution_budget=actual_fft_budget(row)))
    if not groups or any(not rows for rows in groups.values()):
        raise ValueError(
            "Every measured pair needs a candidate inside the explicit local-error budget"
        )
    ordered = sorted(groups)
    baseline = sum(groups[key][0]["execution_budget"]["before"] for key in ordered)
    best = None
    for combination in itertools.product(*(groups[key] for key in ordered)):
        removed = sum(row["execution_budget"]["saved"] for row in combination)
        if removed < baseline * minimum_reduction:
            continue
        errors = [row["relative_pair_output_RMSE"] for row in combination]
        objective = max(errors), sum(e * e for e in errors), -removed
        if best is None or objective < best[0]:
            best = objective, combination, removed
    if best is None:
        raise ValueError(
            "No measured combination meets both operation and local-error budgets"
        )
    return list(best[1]), baseline, best[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("approved", "s2", "manifest", "analysis", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--minimum-reduction", type=float, default=0.40)
    parser.add_argument("--maximum-local-rmse", type=float, default=0.23)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 0 < args.minimum_reduction < 1
        or not 0 < args.maximum_local_rmse < 1
    ):
        parser.error(
            "Fresh output and explicit fractional operation/error budgets required"
        )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    release = json.loads((args.approved / "manifest.json").read_text())
    if not release["quality_approved"]:
        raise ValueError("Approved immutable parent required")
    for name, meta in release["files"].items():
        if sha(args.approved / name) != meta["sha256"]:
            raise ValueError("Approved parent changed")
    analysis = json.loads(args.analysis.read_text())
    if (
        not analysis.get("complete")
        or not analysis.get("parent_parameters_unchanged")
        or analysis.get("heldout_or_final_evaluation_used") is not False
        or analysis["approved_parent_decoder_sha256"]
        != sha(args.approved / "decoder.pt")
        or analysis["training_corpus_sha256"] != sha(args.manifest)
    ):
        raise ValueError(
            "Training-only analysis must match this exact parent and corpus"
        )
    chosen, baseline, removed = choose_plan(
        analysis, args.minimum_reduction, args.maximum_local_rmse
    )
    corpus = json.loads(args.manifest.read_text())
    if not corpus["teacher_unchanged"] or not corpus["collection_complete"]:
        raise ValueError("Immutable complete corpus required")
    spec = json.loads((args.approved / "student_spec.json").read_text())
    teacher = decoder_from_teacher(args.s2, args.manifest.parent / "teacher_decoder.pt")
    parent = (
        StudentGenerator(
            copy.deepcopy(teacher),
            spec["decoder_branches"],
            bottlenecks=spec.get("decoder_bottlenecks"),
        )
        .eval()
        .requires_grad_(False)
    )
    parent.load_state_dict(
        torch.load(args.approved / "decoder.pt", weights_only=True)["state_dict"]
    )
    before = parameter_digest(parent)
    records = [row for row in corpus["records"] if row["split"] == "train"]
    covariances, hooks = {}, []
    for row in chosen:
        stage, branch, pair = row["stage"], row["original_branch"], row["pair"]
        index = parent.stage_offsets[stage] + parent.stage_branches[stage].index(branch)
        conv = parent.resblocks[index].convs2[pair]
        width = row["outer_channels"]
        if conv.in_channels != width or conv.out_channels != width:
            raise ValueError("Only untouched full-width pairs may be initialized")
        key = stage, branch, pair
        covariances[key] = torch.zeros(width, width)

        def capture(_, inputs, key=key, width=width):
            value = inputs[0].detach().transpose(0, 1).reshape(width, -1)
            ix = torch.linspace(0, value.shape[1] - 1, min(512, value.shape[1])).long()
            value = value[:, ix]
            covariances[key].add_((value @ value.t()) / (value.shape[1] * len(records)))

        hooks.append(conv.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            for row in records:
                path = args.manifest.parent / row["decoder_context"]
                if sha(path) != row["decoder_context_sha256"]:
                    raise ValueError("Training context changed")
                context = torch.load(path, map_location="cpu", weights_only=True)
                parent(context["latent"], context["conditioning"])
    finally:
        for hook in hooks:
            hook.remove()
    selections = [
        {
            key: row[key]
            for key in [
                "stage",
                "original_branch",
                "pair",
                "keep",
                "kernel",
                "dilation",
                "inner_channels",
                "outer_channels",
            ]
        }
        for row in chosen
    ]
    student = copy.deepcopy(parent)
    student.select_bottlenecks(selections)
    with torch.no_grad():
        for row in chosen:
            stage, branch, pair = row["stage"], row["original_branch"], row["pair"]
            index = parent.stage_offsets[stage] + parent.stage_branches[stage].index(
                branch
            )
            source = parent.resblocks[index].convs2[pair].weight
            mapping = projection(covariances[stage, branch, pair], row["keep"])
            student.resblocks[index].convs2[pair].weight.copy_(
                torch.einsum("oik,ij->ojk", source, mapping)
            )
    if parameter_digest(parent) != before:
        raise RuntimeError("Approved parent changed")
    spec.update(
        decoder_bottlenecks=list(student.bottlenecks),
        quality_approved=False,
        training="Training-only multi-pair spectral operation-budget initialization; CPU recovery and listening required",
    )
    metadata = {
        "scope": __doc__,
        "analysis_sha256": sha(args.analysis),
        "parent_decoder_sha256": sha(args.approved / "decoder.pt"),
        "parent_manifest_sha256": sha(args.approved / "manifest.json"),
        "corpus_manifest_sha256": sha(args.manifest),
        "runtime_geometry_source_sha256": {
            name: sha(Path(name)) for name in ("cpufast_fft.py", "cpufast_tap.py")
        },
        "minimum_spectral_MAC_reduction": args.minimum_reduction,
        "maximum_independent_pair_RMSE": args.maximum_local_rmse,
        "baseline_measured_pair_spectral_MACs": baseline,
        "removed_spectral_MACs": removed,
        "fraction_spectral_MACs_removed": removed / baseline,
        "selection_objective": "Minimum worst independent local pair RMSE, then sum squared local pair RMSE, within the explicit operation budget; no combined/perceptual guarantee",
        "selected_pairs": selections,
        "per_pair_execution_budget": [
            dict(
                stage=r["stage"],
                branch=r["original_branch"],
                pair=r["pair"],
                **r["execution_budget"],
            )
            for r in chosen
        ],
        "covariance_training_contexts": len(records),
        "parent_unchanged": True,
        "original_parameters": sum(p.numel() for p in parent.parameters()),
        "candidate_parameters": sum(p.numel() for p in student.parameters()),
        "quality_approved": False,
        "actual_n1_measured": False,
    }
    args.output.mkdir(parents=True)
    torch.save(
        {
            "state_dict": student.state_dict(),
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
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "fraction_spectral_MACs_removed",
                    "removed_spectral_MACs",
                    "candidate_parameters",
                    "covariance_training_contexts",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
