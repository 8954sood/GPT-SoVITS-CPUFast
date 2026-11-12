"""Prepare a separate decoder width candidate from a verified approved parent.

Ranks channels by RMS activation times next-layer weight norm on three training
contexts only. Original widths/rates/weights remain unchanged. MAC and parameter
reductions are hypotheses for actual N1 testing, not measured speed successes.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from experiments.student.distill_decoder import (
    decoder_from_teacher,
    generate_with_features,
)
from student_runtime import StudentGenerator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("approved", "s2", "manifest", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if args.output.exists() or args.width <= 0 or args.width % 32:
        parser.error("Fresh output and positive width divisible by32 required")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    release = json.loads((args.approved / "manifest.json").read_text())
    if not release["quality_approved"]:
        raise ValueError("Approved parent release required")
    for name, meta in release["files"].items():
        if sha(args.approved / name) != meta["sha256"]:
            raise ValueError("Parent release changed")
    corpus = json.loads(args.manifest.read_text())
    if not corpus["teacher_unchanged"] or not corpus["collection_complete"]:
        raise ValueError("Completed immutable teacher corpus required")
    root = args.manifest.parent
    teacher = decoder_from_teacher(args.s2, root / "teacher_decoder.pt")
    spec = json.loads((args.approved / "student_spec.json").read_text())
    if spec.get("decoder_channel_keep") is not None:
        raise ValueError("This initializer requires the full-width parent")
    parent = StudentGenerator(copy.deepcopy(teacher), spec["decoder_branches"])
    parent.load_state_dict(
        torch.load(args.approved / "decoder.pt", weights_only=True)["state_dict"]
    )
    parent.eval().requires_grad_(False)
    widths = [parent.conv_pre.out_channels] + [up.out_channels for up in parent.ups]
    targets = [args.width // 2**i for i in range(len(widths))]
    if any(not 0 < new < old for new, old in zip(targets, widths)):
        raise ValueError("Every target width must be smaller than original")
    records = [
        next(
            r
            for r in corpus["records"]
            if r["split"] == "train" and r["language"] == lang
        )
        for lang in ("ja", "ko", "en")
    ]
    activity = [torch.zeros(width) for width in widths]
    context_hashes = []
    with torch.no_grad():
        for record in records:
            path = root / record["decoder_context"]
            ctx = torch.load(path, weights_only=True)
            z, g = ctx["latent"], ctx["conditioning"]
            stem = parent.conv_pre(z) + parent.cond(g)
            _, features = generate_with_features(parent, z, g)
            for scores, feature in zip(activity, [stem] + features):
                scores.add_(feature.square().mean((0, 2)).sqrt() / len(records))
            context_hashes.append(
                {"path": record["decoder_context"], "sha256": sha(path)}
            )
        consumers = [up.weight.square().sum((1, 2)).sqrt() for up in parent.ups]
        consumers.append(parent.conv_post.weight.square().sum((0, 2)).sqrt())
        indices = [
            sorted(torch.topk(a * w, n).indices.tolist())
            for a, w, n in zip(activity, consumers, targets)
        ]
    candidate = copy.deepcopy(parent)
    candidate.select_channels(indices)
    spec.update(
        decoder_channel_keep=indices,
        quality_approved=False,
        training="Decoder width candidate from approved K; posttraining/listening/N1 required",
    )
    metadata = {
        "scope": "Three training contexts, no validation selection or N1 performance claim",
        "parent_decoder_sha256": release["files"]["decoder.pt"]["sha256"],
        "parent_manifest_sha256": sha(args.approved / "manifest.json"),
        "teacher_s2_sha256": sha(args.s2),
        "corpus_manifest_sha256": sha(args.manifest),
        "training_contexts": context_hashes,
        "original_widths": widths,
        "candidate_widths": targets,
        "original_parameters": sum(p.numel() for p in parent.parameters()),
        "candidate_parameters": sum(p.numel() for p in candidate.parameters()),
        "rough_dense_width_squared_ratio": (args.width / widths[0]) ** 2,
        "quality_approved": False,
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
