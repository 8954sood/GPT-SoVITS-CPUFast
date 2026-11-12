"""Initialize one explicit decoder-branch deletion from a verified parent release.

This copies trained parent parameters, never writes original/parent checkpoints,
and does not claim quality or speed approval for the resulting student.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from experiments.student.distill_decoder import decoder_from_teacher
from experiments.student.prune_approved import copied_decoder, sha
from student_runtime import StudentGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("approved", "s2", "teacher-decoder", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--original-branch", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output required")
    torch.set_num_threads(2)
    manifest = json.loads((args.approved / "manifest.json").read_text())
    if manifest.get("quality_approved") is not True:
        parser.error("Listening-approved parent release required")
    for name, metadata in manifest["files"].items():
        if sha(args.approved / name) != metadata["sha256"]:
            raise ValueError(f"Parent release hash mismatch: {name}")
    spec = json.loads((args.approved / "student_spec.json").read_text())
    branches = copy.deepcopy(spec["decoder_branches"])
    if (
        not 0 <= args.stage < len(branches)
        or args.original_branch not in branches[args.stage]
        or len(branches[args.stage]) < 2
    ):
        parser.error("Deletion must leave at least one existing branch per stage")
    teacher = decoder_from_teacher(args.s2, args.teacher_decoder)
    parent = StudentGenerator(copy.deepcopy(teacher), branches)
    parent.load_state_dict(
        torch.load(args.approved / "decoder.pt", weights_only=True)["state_dict"]
    )
    branches[args.stage].remove(args.original_branch)
    candidate = copied_decoder(teacher, parent, branches)
    spec.update(
        decoder_branches=branches,
        quality_approved=False,
        training="Explicit incremental parent deletion; requires new posttraining/listening",
    )
    metadata = {
        "stage": args.stage,
        "original_branch_removed": args.original_branch,
        "parent_sha256": manifest["files"]["decoder.pt"]["sha256"],
        "teacher_s2_sha256": sha(args.s2),
        "teacher_decoder_sha256": sha(args.teacher_decoder),
        "quality_approved": False,
    }
    args.output.mkdir(parents=True)
    torch.save(
        {"state_dict": candidate.state_dict(), "student_spec": spec, **metadata},
        args.output / "initial.pt",
    )
    (args.output / "student_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (args.output / "initialization.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("initialized", branches, flush=True)


if __name__ == "__main__":
    main()
