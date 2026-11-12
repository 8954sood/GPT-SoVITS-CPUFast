"""Select incremental AR18/decoder12 initializations from the accepted G weights.

Uses one training context per language for sensitivity selection, never validation
texts. MAC estimates rank candidates; they are not measured speed improvements.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import torch

from experiments.student.distill_ar import ARStudent, distill_loss
from experiments.student.distill_decoder import (
    decoder_from_teacher,
    generate_with_features,
)
from experiments.student.voice_loss import VoiceLoss
from student_runtime import StudentGenerator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copied_decoder(teacher, source, branches):
    candidate = StudentGenerator(copy.deepcopy(teacher), branches)
    remap = {}
    for stage, kept in enumerate(candidate.stage_branches):
        for j, original in enumerate(kept):
            new = candidate.stage_offsets[stage] + j
            old = source.stage_offsets[stage] + source.stage_branches[stage].index(
                original
            )
            remap[new] = old
    old_state = source.state_dict()
    state = {}
    for name in candidate.state_dict():
        match = re.match(r"resblocks\.(\d+)\.(.*)", name)
        old_name = f"resblocks.{remap[int(match[1])]}.{match[2]}" if match else name
        state[name] = old_state[old_name]
    candidate.load_state_dict(state, strict=True)
    return candidate.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("approved", "teacher", "s2", "manifest", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--max-parent-log-mel", type=float, default=0.55)
    a = p.parse_args()
    if a.output.exists():
        p.error("Fresh output required")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    approved = json.loads((a.approved / "manifest.json").read_text())
    assert approved["quality_approved"]
    for name, metadata in approved["files"].items():
        assert sha(a.approved / name) == metadata["sha256"], name
    manifest = json.loads(a.manifest.read_text())
    assert manifest["collection_complete"] and manifest["teacher_unchanged"]
    records = [
        next(
            r
            for r in manifest["records"]
            if r["split"] == "train" and r["language"] == lang
        )
        for lang in ("ja", "ko", "en")
    ]
    root = a.manifest.parent
    contexts = [torch.load(root / r["ar_context"], weights_only=True) for r in records]
    saved = torch.load(
        a.approved / "ar.pt", weights_only=False, map_location="cpu", mmap=True
    )
    full = ARStudent(saved, list(range(20))).eval()
    original_ids = list(saved["student_metadata"]["original_layer_ids"])
    keep = list(range(20))
    rounds = []
    with torch.no_grad():
        for _ in range(2):
            scores = []
            for remove in keep:
                if original_ids[remove] in (0, 23):
                    continue
                selected = [i for i in keep if i != remove]
                values = []
                for ctx in contexts:
                    x = ctx["inputs"]
                    for i in selected:
                        x = full.blocks[i](x, ctx["mask"])
                    logits = full.predictor(x[:, ctx["prediction_start"] :]).reshape(
                        -1, full.predictor.out_features
                    )
                    values.append(
                        float(distill_loss(logits, ctx["teacher_logits"], 2.0))
                    )
                scores.append(
                    {
                        "local_layer": remove,
                        "original_layer": original_ids[remove],
                        "kl": sum(values) / len(values),
                        "values": values,
                    }
                )
            selected = min(scores, key=lambda r: r["kl"])
            keep.remove(selected["local_layer"])
            rounds.append({"chosen": selected, "candidates": scores})
            print("AR remove", selected, flush=True)
    a.output.mkdir(parents=True)
    original = torch.load(a.teacher, map_location="cpu", weights_only=False, mmap=True)
    ar = ARStudent(saved, keep)
    exported = ar.checkpoint(original)
    exported["student_metadata"].update(
        {
            "original_layer_ids": [original_ids[i] for i in keep],
            "training": "Untrained incremental pruning of approved AR20; quality unapproved",
            "parent_sha256": sha(a.approved / "ar.pt"),
        }
    )
    torch.save(exported, a.output / "ar18_initial.pt")
    del full, ar, original, exported, saved

    teacher = decoder_from_teacher(a.s2, root / "teacher_decoder.pt")
    spec = json.loads((a.approved / "student_spec.json").read_text())
    generator = StudentGenerator(
        copy.deepcopy(teacher), spec["decoder_branches"]
    ).eval()
    generator.load_state_dict(
        torch.load(a.approved / "decoder.pt", weights_only=True, map_location="cpu")[
            "state_dict"
        ]
    )
    loss = VoiceLoss()
    examples = []
    with torch.no_grad():
        for record in records:
            ctx = torch.load(root / record["decoder_context"], weights_only=True)
            target, _ = generate_with_features(
                teacher, ctx["latent"], ctx["conditioning"]
            )
            baseline, features = generate_with_features(
                generator, ctx["latent"], ctx["conditioning"]
            )
            examples.append((ctx, target, baseline, features))
        baseline_loss = sum(float(loss(b, t)[0]) for _, t, b, _ in examples) / len(
            examples
        )
        candidates = []
        for stage in (0, 1):
            for branch in generator.stage_branches[stage]:
                branches = [list(x) for x in generator.stage_branches]
                branches[stage].remove(branch)
                model = copied_decoder(teacher, generator, branches)
                original_losses = []
                parent_losses = []
                macs = []
                idx = generator.stage_offsets[stage] + generator.stage_branches[
                    stage
                ].index(branch)
                parameters = sum(
                    p.numel()
                    for p in generator.resblocks[idx].parameters()
                    if p.ndim == 3
                )
                for ctx, target, baseline, features in examples:
                    output, _ = generate_with_features(
                        model, ctx["latent"], ctx["conditioning"]
                    )
                    original_losses.append(float(loss(output, target)[0]))
                    parent_losses.append(float(loss(output, baseline)[0]))
                    macs.append(parameters * features[stage].shape[-1])
                mean = sum(original_losses) / len(original_losses)
                cost = sum(macs) / len(macs)
                row = {
                    "stage": stage,
                    "original_branch": branch,
                    "branches": branches,
                    "log_mel_to_original": mean,
                    "log_mel_to_G": sum(parent_losses) / len(parent_losses),
                    "delta_to_original": mean - baseline_loss,
                    "removed_macs_estimate": cost,
                    "selection_score": max(mean - baseline_loss, 1e-8) / cost,
                }
                candidates.append(row)
                print("Decoder candidate", row, flush=True)
                del model
        eligible = [r for r in candidates if r["log_mel_to_G"] <= a.max_parent_log_mel]
        if not eligible:
            raise RuntimeError("No deletion satisfied the incremental quality budget")
        chosen = min(eligible, key=lambda r: r["selection_score"])
        model = copied_decoder(teacher, generator, chosen["branches"])
        spec.update(
            {
                "ar_keep": [original_ids[i] for i in keep],
                "decoder_branches": chosen["branches"],
                "quality_approved": False,
                "training": "Incremental untrained pruning from approved G; requires new posttraining/listening",
            }
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "student_spec": spec,
                "quality_approved": False,
                "parent_sha256": sha(a.approved / "decoder.pt"),
            },
            a.output / "decoder12_initial.pt",
        )
    (a.output / "student_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    report = {
        "scope": "Selection on3 training contexts, one per language; validation reserved for training selection; MACs are not N1 timings",
        "parent_manifest_sha256": sha(a.approved / "manifest.json"),
        "ar_greedy_rounds": rounds,
        "ar_keep": spec["ar_keep"],
        "decoder_parent_log_mel": baseline_loss,
        "decoder_candidates": candidates,
        "decoder_chosen": chosen,
        "max_parent_log_mel": a.max_parent_log_mel,
        "quality_approved": False,
    }
    (a.output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
