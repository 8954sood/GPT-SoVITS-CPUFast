"""Validate immutable approved student execution assets; never rewrite checkpoints."""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

FILES = {"ar.pt", "decoder.pt", "student_spec.json"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path, expected_sha256, *, check_files=True):
    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("A full manifest SHA256 is required")
    if sha256(path) != expected_sha256:
        raise ValueError("Approved student manifest checksum mismatch")
    data = json.loads(path.read_text())
    if data.get("quality_approved") is not True or set(data.get("files", {})) != FILES:
        raise ValueError("Approved immutable student bundle required")
    if data.get("ar_cache_storage", "fp32") not in ("fp32", "exact_fp16"):
        raise ValueError("Unsupported student coefficient cache storage")
    if data.get("ar_runtime", "fp32") not in ("fp32", "int8_native"):
        raise ValueError("Unsupported approved AR runtime")
    if data.get("ar_runtime") == "int8_native":
        record = data.get("runtime_implementation", {})
        source = PurePosixPath(record.get("path", ""))
        if (
            data.get("ar_cache_storage") != "exact_fp16"
            or source.is_absolute()
            or ".." in source.parts
            or source.parts[:2] != ("experiments", "student")
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))
        ):
            raise ValueError("Source-pinned approved INT8 runtime required")
    channels = data.get("decoder_channel_keep")
    if channels is not None:
        if not isinstance(channels, list) or len(channels) != 6:
            raise ValueError("Six original decoder channel selections required")
        for selected, width in zip(channels, [768, 384, 192, 96, 48, 24]):
            if (
                not isinstance(selected, list)
                or not selected
                or any(type(i) is not int or not 0 <= i < width for i in selected)
                or sorted(set(selected)) != selected
            ):
                raise ValueError("Invalid original decoder channel selection")
    bottlenecks = data.get("decoder_bottlenecks")
    if bottlenecks is not None:
        if not isinstance(bottlenecks, list) or not bottlenecks or channels is not None:
            raise ValueError(
                "Internal pair bottlenecks must preserve original outer channels"
            )
        seen = set()
        for entry in bottlenecks:
            if not isinstance(entry, dict):
                raise TypeError("Bottleneck entries must be mappings")
            stage, branch, pair = (
                entry.get(k) for k in ("stage", "original_branch", "pair")
            )
            if (
                any(type(i) is not int for i in (stage, branch, pair))
                or not 0 <= stage < 5
                or not 0 <= branch < 3
                or not 0 <= pair < 3
                or branch not in data["decoder_branches"][stage]
                or (stage, branch, pair) in seen
            ):
                raise ValueError("Invalid unique original bottleneck pair location")
            seen.add((stage, branch, pair))
            outer = [384, 192, 96, 48, 24][stage]
            selected = entry.get("keep")
            if (
                not isinstance(selected, list)
                or not 0 < len(selected) < outer
                or any(type(i) is not int or not 0 <= i < outer for i in selected)
                or sorted(set(selected)) != selected
                or entry.get("outer_channels") != outer
                or entry.get("inner_channels") != len(selected)
                or entry.get("kernel") != (3, 7, 11)[branch]
                or entry.get("dilation") != (1, 3, 5)[pair]
            ):
                raise ValueError(
                    "Bottleneck original IDs/inner/outer/kernel/dilation mismatch"
                )
    if data.get("seed") != 42 or data.get("prompt_language") not in ("ja", "ko", "en"):
        raise ValueError("Unsupported approved synthesis request")
    for key in ("reference_sha256", "original_s2_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", data.get(key, "")):
            raise ValueError("Missing original reference/teacher checksum")
    cases = data.get("listening_report", {}).get("cases", [])
    if {r.get("language") for r in cases} != {"ja", "ko", "en"} or len(cases) != 3:
        raise ValueError("Exactly one JA/KO/EN case is required")
    for row in cases:
        if (
            not row.get("text")
            or row.get("sample_rate") != 32000
            or row.get("audio_seconds", 0) <= 0
        ):
            raise ValueError("Invalid original-rate listening case")
    for name, record in data["files"].items():
        if (
            not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))
            or not 0 < record.get("bytes", 0) < 1_000_000_000
        ):
            raise ValueError("Invalid student file integrity record")
        if check_files:
            file = path.parent / name
            if (
                file.is_symlink()
                or not file.is_file()
                or file.stat().st_size != record["bytes"]
                or sha256(file) != record["sha256"]
            ):
                raise ValueError(f"Student execution asset integrity failure: {name}")
    return data


def inline_runtime_approval(data, manifest_sha, root):
    """Verify the source declaration of a whole-bundle listening approval."""
    record = data["runtime_implementation"]
    root = Path(root).resolve()
    source = (root / record["path"]).resolve()
    if (
        not source.is_relative_to(root)
        or not source.is_file()
        or sha256(source) != record["sha256"]
    ):
        raise ValueError("Approved bundle runtime implementation checksum mismatch")
    return {
        "model_variant": data["model_variant"],
        "quality_approved": data["quality_approved"],
        "scope": data["scope"],
        "approval_basis": "Immutable whole-bundle approval declares int8_native",
        "underlying_manifest_sha256": manifest_sha,
        "implementation": record["path"],
        "implementation_sha256": record["sha256"],
        "reference_sha256": data["reference_sha256"],
    }


def snapshot(path):
    path = Path(path)
    return {
        name: sha256(path.parent / name) for name in ["manifest.json", *sorted(FILES)]
    }
