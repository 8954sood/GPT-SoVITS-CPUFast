"""Prepare an unapproved final-C24 two-branch decoder from an approved V parent.

The excluded module audit identified K11 Tap GEMMs in the final stage as a
large cost. Only that final branch is removed; AR files, upsamplers, outer
widths and all retained decoder coefficients remain unchanged at initialization.
This is neither listening approval nor target-server performance evidence.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from cpufast_student_assets import load_manifest, sha256
from experiments.student.distill_decoder import decoder_from_teacher
from student_runtime import StudentGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("approved", "s2", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output required")
    torch.set_num_threads(4)
    release = load_manifest(args.approved / "manifest.json", args.approved_sha256)
    if sha256(args.s2) != release["original_s2_sha256"]:
        raise ValueError("Original S2 teacher checksum mismatch")
    corpus = json.loads(args.manifest.read_text())
    if not corpus.get("collection_complete") or not corpus.get("teacher_unchanged"):
        raise ValueError("Completed immutable teacher corpus required")
    spec = json.loads((args.approved / "student_spec.json").read_text())
    branches = spec["decoder_branches"]
    if len(branches) != 5 or branches[-1] != [0, 1, 2] or spec.get("decoder_channel_keep"):
        raise ValueError("Original outer widths and complete final three branches required")
    if any(row["stage"] == 4 for row in spec.get("decoder_bottlenecks", [])):
        raise ValueError("Final-stage bottlenecks are outside this isolated hypothesis")
    parent = torch.load(args.approved / "decoder.pt", map_location="cpu", weights_only=True)
    for key in ("decoder_branches", "decoder_bottlenecks", "decoder_channel_keep"):
        if parent["student_spec"].get(key) != spec.get(key):
            raise ValueError("Approved decoder geometry/spec mismatch")
    teacher = decoder_from_teacher(args.s2, args.manifest.parent / "teacher_decoder.pt")
    if teacher.ups[-1].out_channels != 24:
        raise ValueError("Final stage must have24 channels")
    selected_branches = copy.deepcopy(branches)
    selected_branches[-1] = [0, 1]
    candidate = StudentGenerator(copy.deepcopy(teacher), selected_branches,
                                 bottlenecks=spec.get("decoder_bottlenecks"))
    source = parent["state_dict"]
    state = candidate.state_dict()
    removed_index = sum(map(len, branches[:-1])) + 2
    omitted = set(source) - set(state)
    if not omitted or any(not key.startswith(f"resblocks.{removed_index}.") for key in omitted):
        raise ValueError("Only the final K11 residual block may be omitted")
    if not set(state).issubset(source) or any(state[k].shape != source[k].shape for k in state):
        raise ValueError("Retained decoder tensors must keep original shapes")
    candidate.load_state_dict({key: source[key] for key in state}, strict=True)
    if any(not torch.equal(value, source[key]) for key, value in candidate.state_dict().items()):
        raise RuntimeError("Initialization changed a retained coefficient")
    spec.update(decoder_branches=selected_branches, quality_approved=False,
                training="Unapproved C24 final K11 branch removal; last-stage CPU recovery/listening required")
    args.output.mkdir()
    torch.save({"state_dict": candidate.state_dict(), "student_spec": spec,
                "quality_approved": False, "training": __doc__}, args.output / "initial.pt")
    (args.output / "student_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    load_manifest(args.approved / "manifest.json", args.approved_sha256)
    report = {"scope": __doc__, "quality_approved": False, "actual_n1_measured": False,
              "approved_parent_manifest_sha256": args.approved_sha256,
              "parent_decoder_sha256": release["files"]["decoder.pt"]["sha256"],
              "initial_sha256": sha256(args.output / "initial.pt"),
              "spec_sha256": sha256(args.output / "student_spec.json"),
              "corpus_manifest_sha256": sha256(args.manifest),
              "teacher_decoder_sha256": sha256(args.manifest.parent / "teacher_decoder.pt"),
              "ar_file_not_modified": True, "parent_bundle_verified_unchanged": True,
              "retained_coefficients_exact": True, "removed_state_keys": sorted(omitted),
              "parent_branches": branches, "candidate_branches": selected_branches,
              "initial_parameters": sum(p.numel() for p in candidate.parameters())}
    (args.output / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Unapproved decoder branches", selected_branches, flush=True)


if __name__ == "__main__":
    main()
