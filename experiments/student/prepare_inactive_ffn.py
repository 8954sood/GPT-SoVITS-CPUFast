"""Export a separate, unapproved AR17 FFN slice from verified training analysis.

Decoder, depth, attention and all retained coefficients are preserved. FFN2
per-row INT8 grids are verified; native PCM equality and RTF are not inferred.
"""

import argparse
import json
from pathlib import Path

import torch

from cpufast_student_assets import load_manifest, sha256
from experiments.student.distill_ar import ARStudent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("approved", "analysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output required")
    torch.set_num_threads(2)
    release = load_manifest(args.approved / "manifest.json", args.approved_sha256)
    analysis = json.loads(args.analysis.read_text())
    if (analysis["approved_manifest_sha256"] != args.approved_sha256
            or analysis["parent_ar_sha256"] != release["files"]["ar.pt"]["sha256"]
            or analysis["validation_used_for_selection"] is not False):
        raise ValueError("Exact approved parent and training-only analysis required")
    parent = torch.load(args.approved / "ar.pt", map_location="cpu", weights_only=False)
    spec = json.loads((args.approved / "student_spec.json").read_text())
    original_ids = spec["ar_keep"]
    if list(parent["student_metadata"]["original_layer_ids"]) != original_ids or len(original_ids) != 17:
        raise ValueError("Matching AR17 parent and spec required")
    if [row["original_layer_id"] for row in analysis["layers"]] != original_ids:
        raise ValueError("Analysis must cover every original AR layer in order")
    local_keep = {}
    for index, row in enumerate(analysis["layers"]):
        if not row["proposed_remove"]:
            continue
        keep = row["keep"]
        if len(keep) not in (1920, 1984) or 2048 - len(keep) != row["proposed_remove"]:
            raise ValueError("Only the bounded64/128-neuron proposal is supported")
        weight = parent["weight"][f"model.h.layers.{index}.linear2.weight"].float()
        if not torch.equal(weight.abs().amax(-1), weight[:, keep].abs().amax(-1)):
            raise ValueError("FFN2 row maximum changed")
        scale = (weight.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        if not torch.equal((weight / scale).round()[:, keep], (weight[:, keep] / scale).round()):
            raise ValueError("Retained FFN2 integer coefficients changed")
        local_keep[index] = keep
    if not local_keep:
        raise ValueError("At least one verified FFN slice required")
    selected = ARStudent(parent, range(17), ffn_keep=local_keep).eval()
    exported = selected.checkpoint(parent)
    neurons = {str(original_ids[index]): keep for index, keep in local_keep.items()}
    exported["student_metadata"].update(
        original_layer_ids=original_ids, ffn_neuron_keep=neurons,
        teacher_sha256=parent["student_metadata"]["teacher_sha256"],
        parent_sha256=release["files"]["ar.pt"]["sha256"], quality_approved=False,
        training="Untrained sparse inactive-FFN slice; same AR17 depth and retained INT8 row grids",
    )
    restored = ARStudent(exported, range(17))
    if any(not torch.equal(value, restored.state_dict()[key]) for key, value in selected.state_dict().items()):
        raise RuntimeError("Export changed retained coefficients")
    spec.update(ar_ffn_keep=neurons, quality_approved=False,
                training="Unapproved AR17 inactive FFN slice; native free-generation listening required")
    args.output.mkdir()
    torch.save(exported, args.output / "initial.pt")
    (args.output / "student_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    load_manifest(args.approved / "manifest.json", args.approved_sha256)
    report = {"scope": __doc__, "quality_approved": False, "actual_n1_measured": False,
              "parent_manifest_sha256": args.approved_sha256, "analysis_sha256": sha256(args.analysis),
              "initial_sha256": sha256(args.output / "initial.pt"),
              "spec_sha256": sha256(args.output / "student_spec.json"),
              "original_layer_ids": original_ids,
              "ffn_widths": exported["config"]["model"]["linear_units_by_layer"],
              "selected_tensor_roundtrip_exact": True, "ffn2_quantization_grids_exact": True,
              "parent_bundle_verified_unchanged": True,
              "decoder_unchanged_sha256": release["files"]["decoder.pt"]["sha256"]}
    (args.output / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["ffn_widths"], flush=True)


if __name__ == "__main__":
    main()
