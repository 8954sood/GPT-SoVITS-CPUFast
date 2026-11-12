"""Create training-only semantic-history substitutions with original teacher targets.

This adds no new texts or speakers. It is not an on-policy rollout, a native
inference implementation, a quality approval or an RTF measurement. Heldout
contexts and approved models are never changed.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path

import torch

from cpufast_student_assets import sha256
from experiments.student.distill_ar import ARStudent, load_split


def substitute_history(ctx, embedding, alpha, generator, probability=0.1, top_k=4):
    """Recover actual token IDs exactly, then substitute only post-prefill rows."""
    x, mask, start = ctx["inputs"], ctx["mask"], ctx["prediction_start"]
    if x.ndim != 3 or x.shape[0] != 1 or not 0 < probability <= 0.25:
        raise ValueError("One CPU sequence and bounded substitution probability required")
    width = x.shape[-1]
    if width % 2 or embedding.shape[1] != width or not 2 <= top_k < embedding.shape[0]:
        raise ValueError("Compatible even-width embedding and top-k required")
    text = int((~mask[0, 0, 0]).sum())
    prefix = start + 1
    if not 0 < text <= start < x.shape[1] - 1:
        raise ValueError("Text/prompt prefix and generated history required")
    expected = torch.arange(x.shape[1])[None, :] > torch.arange(prefix, x.shape[1])[:, None]
    if not torch.equal(mask[0, 0, prefix:], expected):
        raise ValueError("Generated suffix must have the original causal mask")
    positions = torch.arange(prefix - text, x.shape[1] - text).float()[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2).float() * -(math.log(10000) / width))
    pe = torch.empty(x.shape[1] - prefix, width)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * frequencies), torch.cos(positions * frequencies)
    audio = x[0, prefix:] - alpha * pe
    distances = audio.square().sum(-1, keepdim=True) + embedding.square().sum(-1) - 2 * audio @ embedding.T
    original = distances.argmin(-1)
    reconstructed = embedding[original] + alpha * pe
    # Refuse to alter contexts if their coefficient/position convention differs.
    if not torch.equal(reconstructed, x[0, prefix:]):
        raise ValueError("Original teacher embedding/position reconstruction is not exact")
    logits = ctx["teacher_logits"]
    if logits.shape != (x.shape[1] - start, embedding.shape[0]):
        raise ValueError("Teacher prediction/history alignment mismatch")
    # Each previous prediction proposes the token consumed by the following row.
    # EOS is not a valid continuing-history input and is excluded explicitly.
    proposals = logits[:-1, :-1].topk(top_k, -1).indices
    draws = torch.rand(len(original), generator=generator)
    indices = (draws < probability).nonzero().flatten()
    if not len(indices):
        indices = draws.argmin().reshape(1)
    changed = x.clone()
    substitutions = []
    for index in indices.tolist():
        choices = proposals[index][proposals[index] != original[index]]
        rank = int(torch.randint(len(choices), (), generator=generator))
        token = int(choices[rank])
        changed[0, prefix + index] = embedding[token] + alpha * pe[index]
        substitutions.append({"row": prefix + index, "old_token": int(original[index]), "new_token": token})
    if not torch.equal(changed[:, :prefix], x[:, :prefix]):
        raise RuntimeError("Substitution changed the text/reference prompt")
    return changed, substitutions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "manifest", "feature-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1742)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    teacher_sha = sha256(args.teacher)
    source_sha = sha256(args.manifest)
    features = json.loads((args.feature_dir / "manifest.json").read_text())
    if features["teacher_sha256"] != teacher_sha or features["context_manifest_sha256"] != source_sha:
        raise ValueError("Original teacher/context feature provenance mismatch")
    train, validation = load_split(args.manifest)
    saved = torch.load(args.teacher, map_location="cpu", weights_only=False, mmap=True)
    model = ARStudent(saved, list(range(saved["config"]["model"]["n_layer"]))).eval().requires_grad_(False)
    embedding = saved["weight"]["model.ar_audio_embedding.word_embeddings.weight"].float()
    alpha = saved["weight"]["model.ar_audio_position.alpha"].float()
    generator = torch.Generator().manual_seed(args.seed)
    args.output.mkdir()
    feature_out = args.output / "features"
    feature_out.mkdir()
    target_features = copy.deepcopy(features)
    target_features["files"] = {}
    manifest = {"scope": __doc__, "teacher_unchanged": True, "train": [], "validation": [],
                "teacher_sha256": teacher_sha, "source_manifest_sha256": source_sha,
                "augmentation_seed": args.seed, "records": []}
    for path in train + validation:
        entry = features["files"][path.name]
        feature_path = args.feature_dir / entry["file"]
        if sha256(path) != entry["context_sha256"] or sha256(feature_path) != entry["sha256"]:
            raise ValueError("Original context/feature checksum mismatch")
        os.link(path, args.output / path.name)
        os.link(feature_path, feature_out / entry["file"])
        split = "train" if path in train else "validation"
        manifest[split].append(path.name)
        target_features["files"][path.name] = entry
        if split == "validation":
            continue
        ctx = torch.load(path, map_location="cpu", weights_only=True)
        altered, substitutions = substitute_history(ctx, embedding, alpha, generator)
        with torch.no_grad():
            baseline = model(ctx["inputs"], ctx["mask"], ctx["prediction_start"])
            error = float((baseline - ctx["teacher_logits"]).abs().max())
            if error > 0.0005:
                raise ValueError("Pure CPU teacher does not reproduce original context logits")
            logits, hidden = model.forward_with_features(altered, ctx["mask"], ctx["prediction_start"], features["anchors"])
        augmented = {**ctx, "inputs": altered, "teacher_logits": logits,
                     "augmentation": "Sparse original-teacher top-k history substitutions, not on-policy"}
        name = path.stem + "_history.pt"
        feature_name = path.stem + "_history_features.pt"
        torch.save(augmented, args.output / name)
        torch.save({"hidden": hidden}, feature_out / feature_name)
        target_features["files"][name] = {"file": feature_name, "sha256": sha256(feature_out / feature_name),
                                         "context_sha256": sha256(args.output / name)}
        manifest["train"].append(name)
        manifest["records"].append({"source": path.name, "source_sha256": sha256(path), "file": name,
                                    "language": ctx["language"], "substitutions": substitutions,
                                    "original_teacher_max_logit_error": error})
        print(name, len(substitutions), flush=True)
    if sha256(args.teacher) != teacher_sha or sha256(args.manifest) != source_sha:
        raise RuntimeError("Original teacher/corpus changed")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    target_features["context_manifest_sha256"] = sha256(args.output / "manifest.json")
    (feature_out / "manifest.json").write_text(json.dumps(target_features, indent=2) + "\n")
    load_split(args.output / "manifest.json")


if __name__ == "__main__":
    main()
