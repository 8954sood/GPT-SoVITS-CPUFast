"""Report decoder errors separately by existing language and latent origin.

This is fixed-input CPU evaluation, not perceptual approval or an RTF benchmark.
No weights are trained, selected or rewritten here.
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import torch

from cpufast_optimizations import parameter_digest
from experiments.student.decoder_boundaries import generate_with_boundaries
from experiments.student.distill_decoder import decoder_from_teacher
from experiments.student.init_thin_decoder import sha
from experiments.student.voice_loss import VoiceLoss
from student_runtime import StudentGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "spec", "s2", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--teacher-student-manifest", type=Path)
    parser.add_argument("--teacher-student-sha256")
    args = parser.parse_args()
    if bool(args.teacher_student_manifest) != bool(args.teacher_student_sha256):
        parser.error("Approved student teacher manifest and expected SHA must be paired")
    if args.output.exists():
        parser.error("Fresh report path required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("collection_complete") or not manifest.get("teacher_unchanged"):
        raise ValueError("Completed immutable corpus required")
    spec = json.loads(args.spec.read_text())
    teacher = decoder_from_teacher(args.s2, args.manifest.parent / "teacher_decoder.pt")
    model = (
        StudentGenerator(
            copy.deepcopy(teacher),
            spec["decoder_branches"],
            bottlenecks=spec.get("decoder_bottlenecks"),
        )
        .eval()
        .requires_grad_(False)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if spec.get("decoder_channel_keep"):
        raise ValueError("Pair-only students with original outer widths required")
    for key in ("decoder_branches", "decoder_bottlenecks", "decoder_channel_keep"):
        if checkpoint["student_spec"].get(key) != spec.get(key):
            raise ValueError("Deployment checkpoint/spec channel identity mismatch")
    model.load_state_dict(checkpoint["state_dict"])
    teacher_release = None
    if args.teacher_student_manifest:
        from experiments.student.approved_decoder_teacher import approved_decoder_teacher

        teacher, teacher_release = approved_decoder_teacher(
            teacher, args.s2, args.teacher_student_manifest, args.teacher_student_sha256)
    before = parameter_digest(teacher), parameter_digest(model)
    criterion, rows, grouped = VoiceLoss(), [], defaultdict(list)
    locations = spec.get("decoder_bottlenecks", [])
    with torch.no_grad():
        for record in manifest["records"]:
            if record["split"] != "validation":
                continue
            path = args.manifest.parent / record["decoder_context"]
            if sha(path) != record["decoder_context_sha256"]:
                raise ValueError("Validation context changed")
            context = torch.load(path, map_location="cpu", weights_only=True)
            z, g = context["latent"], context["conditioning"]
            target, _, teacher_pairs = generate_with_boundaries(
                teacher, z, g, locations
            )
            audio, _, student_pairs = generate_with_boundaries(model, z, g, locations)
            mel, stft = criterion(audio, target)
            pair_error = {
                str(key): float(
                    (student_pairs[key] - value).abs().mean()
                    / value.abs().mean().clamp_min(1e-5)
                )
                for key, value in teacher_pairs.items()
            }
            row = {
                "id": record["id"],
                "language": record["language"],
                "latent_origin": record["latent_origin"],
                "log_mel": float(mel),
                "log_stft": float(stft),
                "pair_normalized_l1": pair_error,
            }
            rows.append(row)
            grouped[(row["language"], row["latent_origin"])].append(row)
    if not rows or before != (parameter_digest(teacher), parameter_digest(model)):
        raise RuntimeError("Nonempty evaluation and unchanged parameters required")
    if teacher_release:
        from cpufast_student_assets import load_manifest

        load_manifest(args.teacher_student_manifest, args.teacher_student_sha256)
    report = {
        "scope": __doc__,
        "checkpoint_sha256": sha(args.checkpoint),
        "spec_sha256": sha(args.spec),
        "manifest_sha256": sha(args.manifest),
        "teacher_s2_sha256": sha(args.s2),
        "teacher_and_student_unchanged": True,
        "quality_approved": False,
        "actual_n1_measured": False,
        "teacher_target": "approved_student" if teacher_release else "original_decoder",
        "teacher_student_manifest_sha256": args.teacher_student_sha256,
        "teacher_student_decoder_sha256": teacher_release["files"]["decoder.pt"]["sha256"] if teacher_release else None,
        "groups": [
            {
                "language": key[0],
                "latent_origin": key[1],
                "count": len(values),
                "log_mel": sum(v["log_mel"] for v in values) / len(values),
                "log_stft": sum(v["log_stft"] for v in values) / len(values),
            }
            for key, values in grouped.items()
        ],
        "cases": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
