"""Capture real approved-K block inputs for fixed-input kernel replay, not timing."""

import argparse
import copy
import json
from pathlib import Path

import torch

from cpufast_optimizations import parameter_digest
from experiments.student.distill_decoder import decoder_from_teacher
from experiments.student.init_thin_decoder import sha
from student_runtime import StudentGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("approved", "s2", "manifest", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    release = json.loads((args.approved / "manifest.json").read_text())
    for name, meta in release["files"].items():
        if sha(args.approved / name) != meta["sha256"]:
            raise ValueError("Approved parent changed")
    teacher = decoder_from_teacher(args.s2, args.manifest.parent / "teacher_decoder.pt")
    spec = json.loads((args.approved / "student_spec.json").read_text())
    parent = StudentGenerator(copy.deepcopy(teacher), spec["decoder_branches"]).eval()
    parent.load_state_dict(
        torch.load(args.approved / "decoder.pt", weights_only=True)["state_dict"]
    )
    parent.requires_grad_(False)
    before = parameter_digest(parent)
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("collection_complete") or not manifest.get("teacher_unchanged"):
        raise ValueError("Completed training corpus required")
    args.output.mkdir(parents=True)
    report = {
        "scope": "Actual K raw block activations on existing JA/KO/EN training contexts; host capture only, no speed measurement or N1 claim",
        "decoder_sha256": sha(args.approved / "decoder.pt"),
        "corpus_manifest_sha256": sha(args.manifest),
        "captures": [],
    }
    for language in ("ja", "ko", "en"):
        record = next(
            r
            for r in manifest["records"]
            if r["split"] == "train" and r["language"] == language
        )
        ctx_path = args.manifest.parent / record["decoder_context"]
        context = torch.load(ctx_path, weights_only=True)
        captured, hooks = {}, []
        for branch in (1, 2):
            block = parent.resblocks[
                parent.stage_offsets[3] + parent.stage_branches[3].index(branch)
            ]

            def pre(_, inputs, branch=branch, captured=captured):
                captured[branch] = inputs[0].detach().clone()

            hooks.append(block.register_forward_pre_hook(pre))
        with torch.no_grad():
            parent(context["latent"], context["conditioning"])
        for hook in hooks:
            hook.remove()
        for branch, value in captured.items():
            block = parent.resblocks[
                parent.stage_offsets[3] + parent.stage_branches[3].index(branch)
            ]
            name = f"{language}_stage3_branch{branch}.pt"
            torch.save(
                {
                    "input": value,
                    "state_dict": block.state_dict(),
                    "kernel": block.kernel_size,
                    "dilation": block.dilation,
                    "outer_channels": 48,
                    "stage": 3,
                    "original_branch": branch,
                    "context_sha256": sha(ctx_path),
                },
                args.output / name,
            )
            report["captures"].append(
                {
                    "file": name,
                    "sha256": sha(args.output / name),
                    "shape": list(value.shape),
                    "kernel": block.kernel_size,
                    "context_sha256": sha(ctx_path),
                }
            )
    if parameter_digest(parent) != before:
        raise RuntimeError("Approved decoder mutated while capturing")
    report["parent_parameters_unchanged"] = True
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
