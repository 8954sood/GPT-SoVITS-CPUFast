"""Local CPU FP32 decoder student pilot with heldout latent/waveform contexts."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.student.decoder_boundaries import generate_with_boundaries
from experiments.student.voice_loss import VoiceLoss
from student_runtime import StudentGenerator


def decoder_from_teacher(s2, state):
    from module.models import Generator

    config = torch.load(s2, map_location="cpu", weights_only=False, mmap=True)[
        "config"
    ]["model"]
    teacher = Generator(
        config["inter_channels"],
        config["resblock"],
        config["resblock_kernel_sizes"],
        config["resblock_dilation_sizes"],
        config["upsample_rates"],
        config["upsample_initial_channel"],
        config["upsample_kernel_sizes"],
        gin_channels=config["gin_channels"],
    )
    teacher.remove_weight_norm()
    teacher.load_state_dict(
        torch.load(state, map_location="cpu", weights_only=True)["state_dict"]
    )
    return teacher.eval().requires_grad_(False)


def generate_with_features(model, x, g):
    x = model.conv_pre(x) + model.cond(g)
    features = []
    for stage, up in enumerate(model.ups):
        x = up(F.leaky_relu(x, 0.1))
        if isinstance(model, StudentGenerator):
            offset = model.stage_offsets[stage]
            count = len(model.stage_branches[stage])
        else:
            offset = stage * model.num_kernels
            count = model.num_kernels
        values = [model.resblocks[offset + j](x) for j in range(count)]
        x = sum(values) * (1.0 / count)
        features.append(x.clone())
    return torch.tanh(model.conv_post(F.leaky_relu(x))), features


def spectral_loss(output, target):
    losses = []
    for size in (256, 512, 1024):
        window = torch.hann_window(size)
        a = torch.stft(
            output.flatten(0, 1),
            size,
            hop_length=size // 4,
            window=window,
            return_complex=True,
        ).abs()
        b = torch.stft(
            target.flatten(0, 1),
            size,
            hop_length=size // 4,
            window=window,
            return_complex=True,
        ).abs()
        losses.append(
            (a - b).abs().mean() / b.abs().mean().clamp_min(1e-5)
            + F.l1_loss(torch.log(a + 1e-5), torch.log(b + 1e-5))
        )
    return torch.stack(losses).mean() + 0.1 * F.l1_loss(output, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("s2", "spec", "manifest", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--crop", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--log-mel", action="store_true")
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr-final-ratio", type=float, default=1.0)
    parser.add_argument("--train-scope", choices=("all", "last_stage"), default="all")
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.steps <= 1000
        or args.crop < 8
        or args.validate_every < 1
        or not 1 <= args.grad_accum <= 3
        or not 0 <= args.lr_final_ratio <= 1
        or args.lr <= 0
    ):
        parser.error("Fresh output and bounded pilot required")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("teacher_unchanged") is not True:
        raise ValueError("Completed integrity-checked teacher collection required")
    root = args.manifest.parent
    train = [r for r in manifest["records"] if r["split"] == "train"]
    validation = [r for r in manifest["records"] if r["split"] == "validation"]
    buckets = {}
    for record in train:
        buckets.setdefault(record["language"], []).append(record)
    train = [
        rows[i]
        for i in range(max(map(len, buckets.values()), default=0))
        for rows in buckets.values()
        if i < len(rows)
    ]
    if (
        not train
        or not validation
        or {r["decoder_context"] for r in train}
        & {r["decoder_context"] for r in validation}
    ):
        raise ValueError("Disjoint nonempty train/validation required")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    teacher = decoder_from_teacher(args.s2, root / "teacher_decoder.pt")
    spec = json.loads(args.spec.read_text())
    student = (
        StudentGenerator(
            copy.deepcopy(teacher),
            spec["decoder_branches"],
            channel_keep=spec.get("decoder_channel_keep"),
            bottlenecks=spec.get("decoder_bottlenecks"),
        )
        .eval()
        .requires_grad_(True)
    )
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resumed["student_spec"]["decoder_branches"] != spec["decoder_branches"]:
            raise ValueError("Resume branch selection mismatch")
        if resumed["student_spec"].get("decoder_channel_keep") != spec.get(
            "decoder_channel_keep"
        ):
            raise ValueError("Resume decoder channel selection mismatch")
        if resumed["student_spec"].get("decoder_bottlenecks") != spec.get(
            "decoder_bottlenecks"
        ):
            raise ValueError("Resume decoder bottleneck selection mismatch")
        student.load_state_dict(resumed["state_dict"], strict=True)
    if any(
        p.device.type != "cpu" or p.dtype != torch.float32 for p in student.parameters()
    ):
        raise RuntimeError("Student training requires CPU FP32 parameters")
    if any(p.requires_grad or p.device.type != "cpu" for p in teacher.parameters()):
        raise RuntimeError("Teacher must be frozen on CPU")
    from experiments.student.decoder_training_scope import configure_trainable

    trainable_names = configure_trainable(student, args.train_scope)
    frozen_before = {name: value.detach().clone() for name, value in student.named_parameters()
                     if not value.requires_grad}
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=0.0)
    best = float("inf")
    history = []
    started = time.perf_counter()
    criterion = VoiceLoss() if args.log_mel else None
    provenance = {
        "decoder_bottlenecks": spec.get("decoder_bottlenecks"),
        "pair_boundary_loss_weight": 0.25 if spec.get("decoder_bottlenecks") else 0.0,
        "teacher_s2_sha256": hashlib.sha256(args.s2.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "resume_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest()
        if args.resume
        else None,
        "full_context": args.full_context,
        "log_mel": args.log_mel,
        "sample_rate": 32000,
        "lr": args.lr,
        "device": "cpu",
        "dtype": "float32",
        "grad_accum": args.grad_accum,
        "lr_final_ratio": args.lr_final_ratio,
        "train_scope": args.train_scope,
        "trainable_parameter_names": trainable_names,
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
    }
    cache = {}

    def batch(record, random_crop):
        if (
            record.get("decoder_context_sha256")
            and hashlib.sha256(
                (root / record["decoder_context"]).read_bytes()
            ).hexdigest()
            != record["decoder_context_sha256"]
        ):
            raise ValueError("Decoder training context changed")
        ctx = torch.load(
            root / record["decoder_context"], map_location="cpu", weights_only=True
        )
        latent = ctx["latent"]
        n = latent.shape[-1]
        length = n if args.full_context else min(args.crop, n)
        start = random.randrange(n - length + 1) if random_crop else (n - length) // 2
        return latent[..., start : start + length], ctx["conditioning"]

    def target_batch(record, random_crop):
        key = record["decoder_context"]
        if args.full_context and key in cache:
            return cache[key]
        z, g = batch(record, random_crop)
        with torch.no_grad():
            if spec.get("decoder_bottlenecks"):
                target, features, pairs = generate_with_boundaries(
                    teacher, z, g, spec["decoder_bottlenecks"]
                )
            else:
                target, features = generate_with_features(teacher, z, g)
                pairs = {}
        result = z, g, target, features, pairs
        if args.full_context:
            cache[key] = result
        return result

    def reconstruction(audio, target):
        spectral = spectral_loss(audio, target)
        if criterion is None:
            zero = spectral.new_zeros(())
            return spectral, zero, zero, spectral
        mel, stft = criterion(audio, target)
        return mel + 0.25 * stft + 0.5 * spectral, mel, stft, spectral

    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            progress = (step - 1) / max(args.steps - 1, 1)
            lr = args.lr * (
                args.lr_final_ratio
                + (1 - args.lr_final_ratio) * (1 + math.cos(math.pi * progress)) / 2
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            for micro in range(args.grad_accum):
                z, g, target, teacher_features, teacher_pairs = target_batch(
                    train[((step - 1) * args.grad_accum + micro) % len(train)], True
                )
                if teacher_pairs:
                    audio, features, student_pairs = generate_with_boundaries(
                        student, z, g, spec["decoder_bottlenecks"]
                    )
                    pair_loss = torch.stack(
                        [
                            (student_pairs[k] - v).abs().mean()
                            / v.abs().mean().clamp_min(1e-5)
                            for k, v in teacher_pairs.items()
                        ]
                    ).mean()
                else:
                    audio, features = generate_with_features(student, z, g)
                    pair_loss = audio.new_zeros(())
                if student.channel_keep is not None:
                    teacher_features = [
                        feature[:, list(student.channel_keep[i + 1]), :]
                        for i, feature in enumerate(teacher_features)
                    ]
                feature_loss = torch.stack(
                    [
                        (a - b).abs().mean() / b.abs().mean().clamp_min(1e-5)
                        for a, b in zip(features, teacher_features)
                    ]
                ).mean()
                loss = (
                    reconstruction(audio, target)[0]
                    + 0.5 * feature_loss
                    + 0.25 * pair_loss
                ) / args.grad_accum
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite decoder loss")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        if step % args.validate_every == 0 or step == args.steps:
            values = []
            with torch.no_grad():
                for record in validation:
                    z, g, target, _, _ = target_batch(record, False)
                    audio, _ = generate_with_features(student, z, g)
                    values.append([float(v) for v in reconstruction(audio, target)])
            metric, mel, stft, spectral = torch.tensor(values).mean(0).tolist()
            history.append(
                {
                    "step": step,
                    "validation_objective": metric,
                    "validation_log_mel": mel,
                    "validation_log_stft": stft,
                    "validation_spectral": spectral,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if metric < best:
                best = metric
                temporary = args.output / "best.pt.tmp"
                torch.save(
                    {
                        "state_dict": student.state_dict(),
                        "student_spec": spec,
                        "quality_approved": False,
                        "training": "CPU FP32 V2ProPlus latent teacher; voice conditioning recorded by manifest",
                        "step": step,
                        "validation_objective": metric,
                        "validation_spectral": spectral,
                        "validation_log_mel": mel,
                        "provenance": provenance,
                    },
                    temporary,
                )
                os.replace(temporary, args.output / "best.pt")
            (args.output / "history.json").write_text(
                json.dumps(
                    {
                        "scope": "CPU recovery research; not listening approved or N1 performance",
                        "train_records": len(train),
                        "validation_records": len(validation),
                        "history": history,
                        "provenance": provenance,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(history[-1], flush=True)
    torch.save(
        {
            "state_dict": student.state_dict(),
            "student_spec": spec,
            "quality_approved": False,
            "provenance": provenance,
            "step": args.steps,
        },
        args.output / "latest.pt",
    )
    if any(not torch.equal(value, dict(student.named_parameters())[name])
           for name, value in frozen_before.items()):
        raise RuntimeError("Frozen decoder prefix parameters changed")
    if (
        hashlib.sha256(args.s2.read_bytes()).hexdigest()
        != provenance["teacher_s2_sha256"]
    ):
        raise RuntimeError("Original S2 teacher file changed")


if __name__ == "__main__":
    main()
