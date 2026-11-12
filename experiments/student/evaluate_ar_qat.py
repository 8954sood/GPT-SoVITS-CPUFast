"""Evaluate immutable AR checkpoints on original-teacher heldout contexts.

QAT is a training surrogate, not native bitwise inference or a voice-quality
rating. This script never selects layers, trains parameters or measures RTF.
"""

import argparse
import json
from pathlib import Path

import torch

from cpufast_optimizations import parameter_digest
from cpufast_student_assets import sha256
from experiments.student.distill_ar import ARStudent, distill_loss, load_split
from experiments.student.quant_aware_ar import QuantAwareARStudent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "checkpoint", "spec", "manifest", "feature-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runtime", choices=("fp32", "qat"), default="qat")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh report path required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    files_before = {name: sha256(getattr(args, name)) for name in ("teacher", "checkpoint", "spec", "manifest")}
    features = json.loads((args.feature_dir / "manifest.json").read_text())
    if (
        features["teacher_sha256"] != files_before["teacher"]
        or features["context_manifest_sha256"] != files_before["manifest"]
    ):
        raise ValueError("Original teacher and context feature provenance mismatch")
    _, validation = load_split(args.manifest)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec = json.loads(args.spec.read_text())
    if list(saved["student_metadata"]["original_layer_ids"]) != spec["ar_keep"]:
        raise ValueError("Checkpoint/spec original AR layer IDs mismatch")
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False, mmap=True)
    model_type = QuantAwareARStudent if args.runtime == "qat" else ARStudent
    model = model_type(teacher, spec["ar_keep"], ffn_keep=spec.get("ar_ffn_keep"))
    current = ARStudent(saved, list(range(len(spec["ar_keep"]))))
    model.load_state_dict(current.state_dict(), strict=True)
    del teacher, saved, current
    model.eval().requires_grad_(False)
    if not set(features["anchors"]).issubset(spec["ar_keep"]):
        raise ValueError("Original teacher feature anchors must be retained")
    before = parameter_digest(model)
    rows = []
    with torch.no_grad():
        for path in validation:
            entry = features["files"][path.name]
            feature_path = args.feature_dir / entry["file"]
            if sha256(path) != entry["context_sha256"] or sha256(feature_path) != entry["sha256"]:
                raise ValueError("Heldout context or feature checksum mismatch")
            ctx = torch.load(path, map_location="cpu", weights_only=True)
            target = torch.load(feature_path, map_location="cpu", weights_only=True)["hidden"]
            logits, hidden = model.forward_with_features(
                ctx["inputs"], ctx["mask"], ctx["prediction_start"], features["anchors"]
            )
            teacher_logits = ctx["teacher_logits"]
            if logits.shape != teacher_logits.shape or not torch.isfinite(logits).all():
                raise ValueError("Finite logits with the original teacher shape required")
            feature_error = torch.stack([
                (hidden[key] - value).square().mean() / value.square().mean().clamp_min(1e-3)
                for key, value in target.items()
            ]).mean()
            matches = int((logits.argmax(-1) == teacher_logits.argmax(-1)).sum())
            rows.append({"context": path.name, "language": ctx["language"],
                         "tokens": logits.shape[0], "top1_matches": matches,
                         "top1_agreement": matches / logits.shape[0],
                         "kl": float(distill_loss(logits, teacher_logits, 2.0)),
                         "feature_error": float(feature_error)})
    if not rows or parameter_digest(model) != before:
        raise RuntimeError("Nonempty evaluation with unchanged parameters required")
    for name, digest in files_before.items():
        if sha256(getattr(args, name)) != digest:
            raise RuntimeError("Evaluation changed an input file")
    groups = []
    for language in ("ja", "ko", "en"):
        cases = [row for row in rows if row["language"] == language]
        if not cases:
            raise ValueError("All three heldout language groups are required")
        groups.append({"language": language, "count": len(cases),
                       "kl": sum(row["kl"] for row in cases) / len(cases),
                       "feature_error": sum(row["feature_error"] for row in cases) / len(cases),
                       "top1_agreement": sum(row["top1_matches"] for row in cases) / sum(row["tokens"] for row in cases)})
    report = {"scope": __doc__, "runtime_surrogate": args.runtime,
              "ar_layers": len(spec["ar_keep"]), "ar_keep": spec["ar_keep"],
              "inputs_sha256": files_before,
              "feature_manifest_sha256": sha256(args.feature_dir / "manifest.json"),
              "parameters_unchanged": True, "input_files_unchanged": True,
              "quality_approved": False, "actual_n1_measured": False,
              "mean_kl": sum(row["kl"] for row in rows) / len(rows),
              "mean_feature_error": sum(row["feature_error"] for row in rows) / len(rows),
              "groups": groups, "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.runtime, len(spec["ar_keep"]), report["mean_kl"], flush=True)


if __name__ == "__main__":
    main()
