"""Load an immutable released decoder as a frozen CPU FP32 distillation target."""

import copy
import json

import torch

from cpufast_student_assets import load_manifest, sha256
from student_runtime import StudentGenerator


def approved_decoder_teacher(original, s2, manifest_path, expected_sha256):
    release = load_manifest(manifest_path, expected_sha256)
    if sha256(s2) != release["original_s2_sha256"]:
        raise ValueError("Approved teacher must use the same original S2")
    root = manifest_path.parent
    spec = json.loads((root / "student_spec.json").read_text())
    checkpoint = torch.load(root / "decoder.pt", map_location="cpu", weights_only=True)
    for key in ("decoder_branches", "decoder_channel_keep", "decoder_bottlenecks"):
        if checkpoint["student_spec"].get(key) != spec.get(key):
            raise ValueError("Approved decoder teacher checkpoint/spec mismatch")
    teacher = StudentGenerator(
        copy.deepcopy(original), spec["decoder_branches"],
        channel_keep=spec.get("decoder_channel_keep"),
        bottlenecks=spec.get("decoder_bottlenecks"),
    )
    teacher.load_state_dict(checkpoint["state_dict"], strict=True)
    teacher.eval().requires_grad_(False)
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in teacher.parameters()):
        raise ValueError("Approved decoder teacher requires CPU FP32 parameters")
    return teacher, release
