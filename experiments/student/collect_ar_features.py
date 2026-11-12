"""Collect immutable original-teacher AR hidden targets without altering contexts."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.student.distill_ar import ARStudent, load_split


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["teacher", "manifest", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--anchors", nargs="+", type=int, default=[6, 13, 23])
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.anchors
        or len(set(args.anchors)) != len(args.anchors)
    ):
        parser.error("Fresh output and unique teacher anchors required")
    train, validation = load_split(args.manifest)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(
        args.teacher, map_location="cpu", weights_only=False, mmap=True
    )
    teacher = (
        ARStudent(checkpoint, list(range(checkpoint["config"]["model"]["n_layer"])))
        .eval()
        .requires_grad_(False)
    )
    args.output.mkdir(parents=True)
    manifest = {
        "teacher_sha256": sha(args.teacher),
        "context_manifest_sha256": sha(args.manifest),
        "anchors": args.anchors,
        "scope": "Original AR24, CPU FP32 semantic-position hidden states; immutable contexts",
        "files": {},
    }
    with torch.no_grad():
        for path in train + validation:
            ctx = torch.load(path, map_location="cpu", weights_only=True)
            logits, hidden = teacher.forward_with_features(
                ctx["inputs"], ctx["mask"], ctx["prediction_start"], args.anchors
            )
            error = float((logits - ctx["teacher_logits"]).abs().max())
            if error > 1e-3:
                raise RuntimeError("Original teacher reconstruction mismatch")
            output = args.output / path.name
            torch.save({"hidden": hidden}, output)
            manifest["files"][path.name] = {
                "file": output.name,
                "sha256": sha(output),
                "context_sha256": sha(path),
                "teacher_logits_max_abs": error,
            }
    if sha(args.teacher) != manifest["teacher_sha256"]:
        raise RuntimeError("Teacher file changed")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("collected", len(manifest["files"]), "contexts", args.anchors, flush=True)


if __name__ == "__main__":
    main()
