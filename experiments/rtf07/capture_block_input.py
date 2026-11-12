"""Read-only raw approved decoder capture, stopped before the selected block.

Outputs include private parameters/activations and must never enter Git or a
public workflow artifact. This is correctness data, not performance evidence.
"""

import argparse
import json
from pathlib import Path

import torch

from cpufast_optimizations import parameter_digest
from experiments.rtf07.replay_rectangular_block import sha
from experiments.student.distill_decoder import decoder_from_teacher
from student_runtime import StudentGenerator


class Captured(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("approved", "s2", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--branch", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh private output required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    release = json.loads((args.approved / "manifest.json").read_text())
    for name, meta in release["files"].items():
        if sha(args.approved / name) != meta["sha256"]:
            raise ValueError("Approved decoder release changed")
    teacher = decoder_from_teacher(args.s2, args.manifest.parent / "teacher_decoder.pt")
    spec = json.loads((args.approved / "student_spec.json").read_text())
    parent = (
        StudentGenerator(
            teacher,
            spec["decoder_branches"],
            bottlenecks=spec.get("decoder_bottlenecks"),
        )
        .eval()
        .requires_grad_(False)
    )
    parent.load_state_dict(
        torch.load(args.approved / "decoder.pt", weights_only=True)["state_dict"],
        strict=True,
    )
    before = parameter_digest(parent)
    corpus = json.loads(args.manifest.read_text())
    if not corpus.get("collection_complete") or not corpus.get("teacher_unchanged"):
        raise ValueError("Immutable completed decoder corpus required")
    block = parent.resblocks[
        parent.stage_offsets[args.stage]
        + parent.stage_branches[args.stage].index(args.branch)
    ]
    args.output.mkdir(parents=True, mode=0o700)
    report = {
        "scope": "Host read-only real approved decoder activations; no speed or quality claim",
        "decoder_sha256": sha(args.approved / "decoder.pt"),
        "corpus_manifest_sha256": sha(args.manifest),
        "captures": [],
    }
    for language in ("ja", "ko", "en"):
        row = next(
            r
            for r in corpus["records"]
            if r["split"] == "train" and r["language"] == language
        )
        path = args.manifest.parent / row["decoder_context"]
        if sha(path) != row["decoder_context_sha256"]:
            raise RuntimeError("Context checksum mismatch")
        context = torch.load(path, weights_only=True)
        captured = []

        def stop(_, inputs, captured=captured):
            captured.append(inputs[0].detach().clone())
            raise Captured

        hook = block.register_forward_pre_hook(stop)
        try:
            with torch.no_grad():
                parent(context["latent"], context["conditioning"])
        except Captured:
            pass
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Selected real block was not captured exactly once")
        name = f"{language}_stage{args.stage}_branch{args.branch}.pt"
        torch.save(
            {
                "input": captured[0],
                "state_dict": block.state_dict(),
                "kernel": block.kernel_size,
                "dilation": block.dilation,
                "outer_channels": block.convs1[0].in_channels,
                "stage": args.stage,
                "original_branch": args.branch,
                "context_sha256": sha(path),
            },
            args.output / name,
        )
        (args.output / name).chmod(0o600)
        report["captures"].append(
            {
                "file": name,
                "sha256": sha(args.output / name),
                "shape": list(captured[0].shape),
                "kernel": block.kernel_size,
                "context_sha256": sha(path),
            }
        )
    if before != parameter_digest(parent):
        raise RuntimeError("Approved parent changed during capture")
    report["parent_parameters_unchanged"] = True
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
