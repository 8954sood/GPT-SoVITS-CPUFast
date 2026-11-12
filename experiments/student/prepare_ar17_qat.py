"""Select one AR18 layer on training contexts for a separate AR17 QAT pilot.

This creates an unapproved initialization, not a speed or listening result.
Protected original teacher feature anchors are retained. Decoder files are not
read or changed. Quantization-aware scoring is a training surrogate only.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cpufast_student_assets import sha256
from experiments.student.distill_ar import ARStudent, distill_loss, load_split
from experiments.student.quant_aware_ar import (
    QuantAwareARStudent,
    block_forward,
    half_ste,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "spec", "manifest", "feature-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--contexts-per-language", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.contexts_per_language <= 10:
        parser.error("Fresh output and 1..10 training contexts per language required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    before = sha256(args.parent)
    saved = torch.load(args.parent, map_location="cpu", weights_only=False)
    spec = json.loads(args.spec.read_text())
    original_ids = list(saved["student_metadata"]["original_layer_ids"])
    if (
        original_ids != spec["ar_keep"]
        or len(original_ids) != 18
        or saved["config"]["model"]["n_layer"] != 18
        or saved["config"]["model"].get("linear_units_by_layer")
        or saved["student_metadata"].get("ffn_neuron_keep")
    ):
        raise ValueError("AR18 parent and matching original IDs without FFN pruning required")
    features = json.loads((args.feature_dir / "manifest.json").read_text())
    if (
        features["context_manifest_sha256"] != sha256(args.manifest)
        or features["teacher_sha256"] != saved["student_metadata"]["teacher_sha256"]
    ):
        raise ValueError("Original teacher/context feature provenance mismatch")
    train, validation = load_split(args.manifest)
    contexts = []
    counts = {language: 0 for language in ("ja", "ko", "en")}
    for path in train:
        if sha256(path) != features["files"][path.name]["context_sha256"]:
            raise ValueError("Training context checksum mismatch")
        ctx = torch.load(path, map_location="cpu", weights_only=True)
        language = ctx["language"]
        if language in counts and counts[language] < args.contexts_per_language:
            contexts.append((path, ctx))
            counts[language] += 1
    if any(n != args.contexts_per_language for n in counts.values()):
        raise ValueError("Insufficient balanced training contexts")
    if {p.resolve() for p, _ in contexts} & set(validation):
        raise ValueError("Selection may not access heldout contexts")
    model = QuantAwareARStudent(saved, list(range(18))).eval().requires_grad_(False)
    protected = {original_ids[0], original_ids[-1], *features["anchors"]}

    def score(removed):
        rows = []
        for path, ctx in contexts:
            value = ctx["inputs"]
            for index, block in enumerate(model.blocks):
                if index != removed:
                    value = block_forward(block, value, ctx["mask"], ctx["prediction_start"] + 1)
            logits = F.linear(value[:, ctx["prediction_start"]:], half_ste(model.predictor.weight))
            loss = distill_loss(logits.reshape(-1, logits.shape[-1]), ctx["teacher_logits"], 2.0)
            rows.append({"context": path.name, "language": ctx["language"], "kl": float(loss)})
        return {"mean_kl": sum(r["kl"] for r in rows) / len(rows), "cases": rows}

    with torch.no_grad():
        baseline = score(None)
        candidates = []
        for index, original in enumerate(original_ids):
            if original in protected:
                continue
            row = {"local_index": index, "original_id": original, **score(index)}
            candidates.append(row)
            print(original, row["mean_kl"], flush=True)
    selected = min(candidates, key=lambda row: row["mean_kl"])
    keep = [i for i in range(18) if i != selected["local_index"]]
    reduced = ARStudent(saved, keep).eval()
    exported = reduced.checkpoint(saved)
    exported["student_metadata"].update(
        original_layer_ids=[original_ids[i] for i in keep],
        teacher_sha256=features["teacher_sha256"],
        parent_sha256=before,
        training="Unapproved single-layer AR17 initialization, selected on training-only QAT surrogate",
        quality_approved=False,
    )
    # Round-trip layer mapping must preserve every selected tensor exactly.
    restored = ARStudent(exported, list(range(17)))
    if any(not torch.equal(value, restored.state_dict()[key]) for key, value in reduced.state_dict().items()):
        raise RuntimeError("Export changed selected AR tensors")
    if sha256(args.parent) != before:
        raise RuntimeError("Parent checkpoint changed")
    spec.update(ar_keep=[original_ids[i] for i in keep], quality_approved=False,
                training="AR17 training-only QAT selection; CPU recovery and listening required")
    args.output.mkdir(parents=True)
    torch.save(exported, args.output / "initial.pt")
    (args.output / "student_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    report = {"scope": __doc__, "quality_approved": False, "actual_n1_measured": False,
              "parent_sha256": before, "parent_unchanged": True,
              "context_manifest_sha256": sha256(args.manifest),
              "feature_manifest_sha256": sha256(args.feature_dir / "manifest.json"),
              "training_contexts": [{"name": p.name, "sha256": sha256(p)} for p, _ in contexts],
              "protected_original_ids": sorted(protected), "baseline": baseline,
              "candidates": candidates, "selected": selected,
              "export_tensors_exact": True, "initial_sha256": sha256(args.output / "initial.pt")}
    (args.output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
