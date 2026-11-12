"""Combine existing original-AR and approved-student-AR decoder inputs safely."""

import argparse
import json
import os
from pathlib import Path

import torch

from experiments.student.init_thin_decoder import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("teacher", "student", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output required")
    sources = [args.teacher, args.student]
    manifests = [json.loads(p.read_text()) for p in sources]
    if any(
        not m.get("collection_complete") or not m.get("teacher_unchanged")
        for m in manifests
    ):
        raise ValueError("Two complete immutable collections required")
    if manifests[1].get("latent_origin") != "approved_L_free_generation":
        raise ValueError("B must use the approved L free-generation runtime")
    states = [
        torch.load(p.parent / "teacher_decoder.pt", weights_only=True)["state_dict"]
        for p in sources
    ]
    if states[0].keys() != states[1].keys() or any(
        not torch.equal(v, states[1][k]) for k, v in states[0].items()
    ):
        raise ValueError("Both collections must use the exact original decoder teacher")
    args.output.mkdir(parents=True)
    os.link(
        sources[0].parent / "teacher_decoder.pt", args.output / "teacher_decoder.pt"
    )
    report = {
        "scope": "Decoder-only A(original AR) + B(approved L AR) inputs; original teacher target recomputed on identical latent/conditioning",
        "ar_distillation_allowed": False,
        "collection_complete": False,
        "teacher_unchanged": True,
        "teacher_decoder_sha256": sha(args.output / "teacher_decoder.pt"),
        "source_manifest_sha256": [sha(p) for p in sources],
        "records": [],
    }
    splits = {"train": set(), "validation": set()}
    for label, source, manifest in zip(("A", "B"), sources, manifests):
        for record in manifest["records"]:
            original = source.parent / record["decoder_context"]
            context = torch.load(original, weights_only=True)
            split = record["split"]
            if context["split"] != split or context["language"] != record["language"]:
                raise ValueError("Context split/language mismatch")
            splits[split].add((context["language"], context["text"]))
            name = label + "_" + record["decoder_context"]
            os.link(original, args.output / name)
            report["records"].append(
                {
                    "id": label + "_" + record["id"],
                    "split": split,
                    "decoder_context": name,
                    "decoder_context_sha256": sha(original),
                    "language": record["language"],
                    "speaker": record["speaker"],
                    "latent_origin": "original_AR_teacher"
                    if label == "A"
                    else "approved_L_free_generation",
                }
            )
    if (
        not splits["train"]
        or not splits["validation"]
        or splits["train"] & splits["validation"]
    ):
        raise ValueError("Nonempty disjoint text splits required across both origins")
    report["collection_complete"] = True
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
