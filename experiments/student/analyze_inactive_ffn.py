"""Analyze inactive FFN neurons while protecting every FFN2 INT8 scale maximum.

Training-only QAT-surrogate observations are not a global sparsity guarantee,
native exactness, listening approval or target-server performance. No model is
exported. This tests a narrower hypothesis than the previously rejected S1536.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from cpufast_student_assets import load_manifest, sha256
from experiments.student.distill_ar import load_split
from experiments.student import quant_aware_ar as qat


def protected_maxima(weight):
    """One surviving maximum per output row preserves per-row INT8 scales."""
    return weight.abs().argmax(-1).unique().sort().values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("approved", "manifest", "feature-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-sha256", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh analysis path required")
    release = load_manifest(args.approved / "manifest.json", args.approved_sha256)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    saved = torch.load(args.approved / "ar.pt", map_location="cpu", weights_only=False)
    original_ids = json.loads((args.approved / "student_spec.json").read_text())["ar_keep"]
    if list(saved["student_metadata"]["original_layer_ids"]) != original_ids:
        raise ValueError("Approved original layer IDs must match")
    features = json.loads((args.feature_dir / "manifest.json").read_text())
    if (features["context_manifest_sha256"] != sha256(args.manifest)
            or features["teacher_sha256"] != saved["student_metadata"]["teacher_sha256"]):
        raise ValueError("Original context/teacher provenance mismatch")
    model = qat.QuantAwareARStudent(saved, range(len(original_ids))).eval().requires_grad_(False)
    train, validation = load_split(args.manifest)
    maxima = [torch.full((block.ff1.out_features,), -torch.inf) for block in model.blocks]
    layer_for = {id(block.ff1): index for index, block in enumerate(model.blocks)}
    original_linear = qat.decode_linear

    def observe(module, value, prefix):
        output = original_linear(module, value, prefix)
        index = layer_for.get(id(module))
        if index is not None:
            maxima[index] = torch.maximum(maxima[index], output.amax((0, 1)))
        return output

    inputs = []
    with torch.no_grad(), patch.object(qat, "decode_linear", observe):
        for path in train:
            if sha256(path) != features["files"][path.name]["context_sha256"]:
                raise ValueError("Original training context checksum mismatch")
            ctx = torch.load(path, map_location="cpu", weights_only=True)
            if ctx["split"] != "train" or path in validation:
                raise ValueError("Analysis cannot access validation contexts")
            logits = model(ctx["inputs"], ctx["mask"], ctx["prediction_start"])
            if not torch.isfinite(logits).all():
                raise ValueError("Finite training forward required")
            inputs.append({"name": path.name, "sha256": sha256(path), "language": ctx["language"]})
    rows = []
    for original, block, maximum in zip(original_ids, model.blocks, maxima):
        weight = qat.half_ste(block.ff2.weight).detach()
        protected = protected_maxima(weight)
        eligible = maximum < -0.05  # Positive margin avoids near-zero sign changes in this surrogate.
        eligible[protected] = False
        count = int(eligible.sum())
        remove_count = min(128, count // 64 * 64)
        removed = maximum.masked_fill(~eligible, torch.inf).argsort()[:remove_count]
        retained = torch.ones(len(maximum), dtype=torch.bool)
        retained[removed] = False
        keep = retained.nonzero().flatten()
        original_scales = weight.abs().amax(-1)
        selected_scales = weight[:, keep].abs().amax(-1)
        original_q = (weight / (original_scales[:, None] / 127).clamp_min(1e-12)).round().clamp(-127, 127)
        selected_q = (weight[:, keep] / (selected_scales[:, None] / 127).clamp_min(1e-12)).round().clamp(-127, 127)
        if not torch.equal(original_scales, selected_scales) or not torch.equal(original_q[:, keep], selected_q):
            raise RuntimeError("Proposed selection changed retained FFN2 quantization grid")
        rows.append({"original_layer_id": original, "ffn_width": len(maximum),
                     "all_training_rows_inactive": int((maximum <= 0).sum()),
                     "margin_inactive_not_scale_maximum": count,
                     "protected_maximum_columns": len(protected),
                     "proposed_remove": remove_count, "keep": keep.tolist(),
                     "removed_max_preactivation": float(maximum[removed].max()) if remove_count else None,
                     "ffn2_scale_and_retained_int8_coefficients_exact": True})
    load_manifest(args.approved / "manifest.json", args.approved_sha256)
    report = {"scope": __doc__, "approved_manifest_sha256": args.approved_sha256,
              "parent_ar_sha256": release["files"]["ar.pt"]["sha256"],
              "parent_bundle_verified_unchanged": True, "quality_approved": False,
              "actual_n1_measured": False, "model_exported": False,
              "training_contexts": inputs, "validation_used_for_selection": False,
              "context_manifest_sha256": sha256(args.manifest),
              "source_sha256": sha256(__file__), "layers": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print([(row["original_layer_id"], row["proposed_remove"]) for row in rows], flush=True)


if __name__ == "__main__":
    main()
