"""FP32 AR student distillation on explicitly split teacher contexts.

This trains only transformer layers and predictor on teacher-forced contexts.
Free-generation and heldout listening remain mandatory before adoption.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from student_runtime import checked_keep


class StudentBlock(torch.nn.Module):
    def __init__(self, width, heads, ffn):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.out = torch.nn.Linear(width, width)
        self.norm1 = torch.nn.LayerNorm(width)
        self.ff1 = torch.nn.Linear(width, ffn)
        self.ff2 = torch.nn.Linear(ffn, width)
        self.norm2 = torch.nn.LayerNorm(width)

    def forward(self, x, blocked):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [
            t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for t in (q, k, v)
        ]
        attention = F.scaled_dot_product_attention(q, k, v, ~blocked)
        attention = attention.transpose(1, 2).reshape(batch, length, width)
        x = self.norm1(x + self.out(attention))
        return self.norm2(x + self.ff2(F.relu(self.ff1(x))))


MAPPING = {
    "qkv": "self_attn.in_proj",
    "out": "self_attn.out_proj",
    "norm1": "norm1",
    "norm2": "norm2",
    "ff1": "linear1",
    "ff2": "linear2",
}


class ARStudent(torch.nn.Module):
    def __init__(self, checkpoint, keep):
        super().__init__()
        cfg = checkpoint["config"]["model"]
        width = cfg["hidden_dim"]
        keep = checked_keep(keep, cfg["n_layer"], "AR keep")
        self.keep = keep
        self.blocks = torch.nn.ModuleList(
            [StudentBlock(width, cfg["head"], cfg["linear_units"]) for _ in keep]
        )
        self.predictor = torch.nn.Linear(width, cfg["vocab_size"], bias=False)
        state = checkpoint["weight"]
        with torch.no_grad():
            for block, old in zip(self.blocks, keep):
                for local, remote in MAPPING.items():
                    module = getattr(block, local)
                    for field in ("weight", "bias"):
                        key = (
                            f"model.h.layers.{old}.{remote}"
                            + ("_" if local == "qkv" else ".")
                            + field
                        )
                        getattr(module, field).copy_(state[key].float())
            self.predictor.weight.copy_(state["model.ar_predict_layer.weight"].float())

    def forward(self, x, blocked, start):
        for block in self.blocks:
            x = block(x, blocked)
        return self.predictor(x[:, start:]).reshape(-1, self.predictor.out_features)

    def forward_with_features(self, x, blocked, start, anchors):
        captured = {}
        for original, block in zip(self.keep, self.blocks):
            x = block(x, blocked)
            if original in anchors:
                captured[str(original)] = x[:, start:].clone()
        if len(captured) != len(anchors):
            raise ValueError("Feature anchors must be retained original layers")
        return self.predictor(x[:, start:]).reshape(
            -1, self.predictor.out_features
        ), captured

    def checkpoint(self, teacher):
        cfg = copy.deepcopy(teacher["config"])
        cfg["model"]["n_layer"] = len(self.keep)
        weights = {
            k: v
            for k, v in teacher["weight"].items()
            if not k.startswith("model.h.layers.")
        }
        for new, block in enumerate(self.blocks):
            for local, remote in MAPPING.items():
                for field in ("weight", "bias"):
                    key = (
                        f"model.h.layers.{new}.{remote}"
                        + ("_" if local == "qkv" else ".")
                        + field
                    )
                    weights[key] = (
                        getattr(getattr(block, local), field).detach().cpu().clone()
                    )
        weights["model.ar_predict_layer.weight"] = (
            self.predictor.weight.detach().cpu().clone()
        )
        return {
            "config": cfg,
            "weight": weights,
            "student_metadata": {
                "original_layer_ids": self.keep,
                "training": "teacher-context FP32 distillation",
                "quality_approved": False,
            },
        }


def distill_loss(student, teacher, temperature):
    # Explicit token-normalized KL(teacher || student), stable in log space.
    log_t = F.log_softmax(teacher / temperature, -1)
    log_s = F.log_softmax(student / temperature, -1)
    return (log_t.exp() * (log_t - log_s)).sum(-1).mean() * temperature**2


def load_split(manifest_path):
    data = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    if data.get("teacher_unchanged") is not True:
        raise ValueError("Completed integrity-checked teacher collection required")
    train = [(root / p).resolve() for p in data["train"]]
    validation = [(root / p).resolve() for p in data["validation"]]
    if not train or not validation or set(train) & set(validation):
        raise ValueError(
            "Disjoint nonempty train and validation context paths required"
        )
    import hashlib

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if {digest(p) for p in train} & {digest(p) for p in validation}:
        raise ValueError("Duplicate content across train/validation")
    return train, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr-final-ratio", type=float, default=1.0)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--feature-weight", type=float, default=0.0)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.steps <= 1000
        or args.temperature <= 0
        or args.validate_every < 1
        or not 1 <= args.grad_accum <= 3
        or not 0 <= args.lr_final_ratio <= 1
        or args.lr <= 0
        or args.feature_weight < 0
        or bool(args.feature_dir) != bool(args.feature_weight)
    ):
        parser.error("Fresh output, 1–1000 steps and positive temperature required")
    train, validation = load_split(args.manifest)
    # Interleave languages so every three updates cover JA/KO/EN.
    buckets = {}
    for path in train:
        lang = torch.load(path, map_location="cpu", weights_only=True)["language"]
        buckets.setdefault(lang, []).append(path)
    train = [
        rows[i]
        for i in range(max(map(len, buckets.values())))
        for rows in buckets.values()
        if i < len(rows)
    ]
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    teacher = torch.load(
        args.teacher, map_location="cpu", weights_only=False, mmap=True
    )
    keep = json.loads(args.spec.read_text())["ar_keep"]
    model = ARStudent(teacher, keep)
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        if list(resumed["student_metadata"]["original_layer_ids"]) != keep:
            raise ValueError("Resume original-layer selection mismatch")
        model.load_state_dict(ARStudent(resumed, list(range(len(keep)))).state_dict())
    if any(
        p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()
    ):
        raise RuntimeError("Student training requires CPU FP32 parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    history = []
    best = float("inf")
    started = time.perf_counter()
    provenance = {
        "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "resume_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest()
        if args.resume
        else None,
        "lr": args.lr,
        "temperature": args.temperature,
        "device": "cpu",
        "dtype": "float32",
        "grad_accum": args.grad_accum,
        "lr_final_ratio": args.lr_final_ratio,
        "feature_weight": args.feature_weight,
    }
    feature_manifest = None
    if args.feature_dir:
        feature_manifest = json.loads((args.feature_dir / "manifest.json").read_text())
        if (
            feature_manifest["teacher_sha256"] != provenance["teacher_sha256"]
            or feature_manifest["context_manifest_sha256"]
            != provenance["manifest_sha256"]
        ):
            raise ValueError("Feature targets belong to another teacher/corpus")
        provenance["feature_manifest_sha256"] = hashlib.sha256(
            (args.feature_dir / "manifest.json").read_bytes()
        ).hexdigest()
        for path in train + validation:
            entry = feature_manifest["files"][path.name]
            if (
                hashlib.sha256(path.read_bytes()).hexdigest() != entry["context_sha256"]
                or hashlib.sha256(
                    (args.feature_dir / entry["file"]).read_bytes()
                ).hexdigest()
                != entry["sha256"]
            ):
                raise ValueError("Feature/context integrity mismatch")

    def prediction(path, ctx):
        if feature_manifest is None:
            return model(ctx["inputs"], ctx["mask"], ctx["prediction_start"]), ctx[
                "inputs"
            ].new_zeros(())
        targets = torch.load(
            args.feature_dir / feature_manifest["files"][path.name]["file"],
            map_location="cpu",
            weights_only=True,
        )["hidden"]
        logits, hidden = model.forward_with_features(
            ctx["inputs"],
            ctx["mask"],
            ctx["prediction_start"],
            feature_manifest["anchors"],
        )
        feature = torch.stack(
            [
                (hidden[k] - v).square().mean() / v.square().mean().clamp_min(1e-3)
                for k, v in targets.items()
            ]
        ).mean()
        return logits, feature

    for step in range(args.steps + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress = (step - 1) / max(args.steps - 1, 1)
            lr = args.lr * (
                args.lr_final_ratio
                + (1 - args.lr_final_ratio) * (1 + math.cos(math.pi * progress)) / 2
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            for micro in range(args.grad_accum):
                context_path = train[
                    ((step - 1) * args.grad_accum + micro) % len(train)
                ]
                ctx = torch.load(
                    context_path,
                    map_location="cpu",
                    weights_only=True,
                )
                logits, feature = prediction(context_path, ctx)
                loss = (
                    distill_loss(logits, ctx["teacher_logits"], args.temperature)
                    + args.feature_weight * feature
                ) / args.grad_accum
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % args.validate_every == 0 or step == args.steps:
            model.eval()
            values = []
            feature_values = []
            with torch.no_grad():
                for path in validation:
                    ctx = torch.load(path, map_location="cpu", weights_only=True)
                    logits, feature = prediction(path, ctx)
                    feature_values.append(float(feature))
                    values.append(
                        float(
                            distill_loss(
                                logits, ctx["teacher_logits"], args.temperature
                            )
                        )
                    )
            metric = sum(values) / len(values)
            feature_metric = sum(feature_values) / len(feature_values)
            selection_metric = metric + args.feature_weight * feature_metric
            history.append(
                {
                    "step": step,
                    "validation_kl": metric,
                    "validation_feature": feature_metric,
                    "validation_objective": selection_metric,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if selection_metric < best:
                best = selection_metric
                temporary = args.output / "best.pt.tmp"
                saved = model.checkpoint(teacher)
                saved["student_metadata"].update(
                    {
                        **provenance,
                        "step": step,
                        "validation_kl": metric,
                        "validation_feature": feature_metric,
                        "validation_objective": selection_metric,
                    }
                )
                torch.save(saved, temporary)
                os.replace(temporary, args.output / "best.pt")
            (args.output / "history.json").write_text(
                json.dumps(
                    {
                        "scope": "Teacher forcing only; no free-generation approval",
                        "train_contexts": len(train),
                        "validation_contexts": len(validation),
                        "history": history,
                        "provenance": provenance,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(history[-1], flush=True)
    latest = model.checkpoint(teacher)
    latest["student_metadata"].update({**provenance, "step": args.steps})
    torch.save(latest, args.output / "latest.pt")
    if (
        hashlib.sha256(args.teacher.read_bytes()).hexdigest()
        != provenance["teacher_sha256"]
    ):
        raise RuntimeError("Original teacher file changed")


if __name__ == "__main__":
    main()
