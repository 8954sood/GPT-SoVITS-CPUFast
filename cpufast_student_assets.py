"""Validate immutable approved student execution assets; never rewrite checkpoints."""

import hashlib
import json
import re
from pathlib import Path


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


def snapshot(path):
    path = Path(path)
    return {
        name: sha256(path.parent / name) for name in ["manifest.json", *sorted(FILES)]
    }
