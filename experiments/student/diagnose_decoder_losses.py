"""Diagnose decoder reconstruction/feature gradient alignment without training."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from experiments.student.distill_decoder import (
    decoder_from_teacher,
    generate_with_features,
    spectral_loss,
)
from experiments.student.voice_loss import VoiceLoss
from student_runtime import StudentGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("s2", "manifest", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("Fresh output and positive CPU thread count required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    root = args.manifest.parent
    manifest = json.loads(args.manifest.read_text())
    source = args.checkpoint
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    spec = checkpoint["student_spec"]
    teacher = decoder_from_teacher(
        args.s2,
        root / "teacher_decoder.pt",
    )
    student = (
        StudentGenerator(
            copy.deepcopy(teacher),
            spec["decoder_branches"],
            channel_keep=spec["decoder_channel_keep"],
        )
        .eval()
        .requires_grad_(True)
    )
    student.load_state_dict(checkpoint["state_dict"])
    criterion = VoiceLoss()
    parameters = list(student.parameters())
    rows = []
    for language in ("ja", "ko", "en"):
        record = next(
            r
            for r in manifest["records"]
            if r["split"] == "train" and r["language"] == language
        )
        ctx = torch.load(
            root / record["decoder_context"], map_location="cpu", weights_only=True
        )
        with torch.no_grad():
            target, features = generate_with_features(
                teacher, ctx["latent"], ctx["conditioning"]
            )
            features = [
                x[:, list(student.channel_keep[i + 1]), :]
                for i, x in enumerate(features)
            ]
        audio, student_features = generate_with_features(
            student, ctx["latent"], ctx["conditioning"]
        )
        mel, stft = criterion(audio, target)
        reconstruction = mel + 0.25 * stft + 0.5 * spectral_loss(audio, target)
        feature = torch.stack(
            [
                (a - b).abs().mean() / b.abs().mean().clamp_min(1e-5)
                for a, b in zip(student_features, features)
            ]
        ).mean()
        first = torch.autograd.grad(reconstruction, parameters, retain_graph=True)
        second = torch.autograd.grad(feature, parameters, allow_unused=True)
        second = [
            torch.zeros_like(p) if g is None else g for p, g in zip(parameters, second)
        ]
        dot = sum((a.double() * b.double()).sum().item() for a, b in zip(first, second))
        norm1 = sum(a.double().square().sum().item() for a in first) ** 0.5
        norm2 = sum(a.double().square().sum().item() for a in second) ** 0.5
        rows.append(
            {
                "language": language,
                "context": record["decoder_context"],
                "reconstruction": float(reconstruction.detach()),
                "feature": float(feature.detach()),
                "gradient_cosine": dot / max(norm1 * norm2, 1e-30),
                "feature_gradient_relative_norm_at_weight_half": 0.5
                * norm2
                / max(norm1, 1e-30),
            }
        )
        print(rows[-1], flush=True)
        del first, second, audio, student_features, reconstruction, feature
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    args.output.write_text(
        json.dumps(
            {
                "scope": "First training context per JA/KO/EN; diagnostic only, no quality/N1 speed claim",
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "teacher_s2_sha256": hashlib.sha256(args.s2.read_bytes()).hexdigest(),
                "corpus_manifest_sha256": hashlib.sha256(
                    args.manifest.read_bytes()
                ).hexdigest(),
                "checkpoint_sha256": digest,
                "checkpoint_unchanged": True,
                "feature_weight": 0.5,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
