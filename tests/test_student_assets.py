"""Fail closed before loading a changed/unapproved private checkpoint bundle."""

import copy
import json
from pathlib import Path

import pytest

from cpufast_student_assets import load_manifest, sha256


def bundle(tmp_path):
    files = {}
    for name in ("ar.pt", "decoder.pt", "student_spec.json"):
        path = tmp_path / name
        path.write_bytes((name + "-synthetic-test-content").encode())
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    data = {
        "quality_approved": True,
        "seed": 42,
        "prompt_language": "ja",
        "reference_sha256": "a" * 64,
        "original_s2_sha256": "b" * 64,
        "files": files,
        "listening_report": {
            "cases": [
                {
                    "language": lang,
                    "text": lang + " test",
                    "sample_rate": 32000,
                    "audio_seconds": 1.0,
                }
                for lang in ("ja", "ko", "en")
            ]
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, data


def test_model_tampering_rejected_before_deserialization(tmp_path):
    manifest, _ = bundle(tmp_path)
    expected = sha256(manifest)
    assert load_manifest(manifest, expected)["quality_approved"]
    with (tmp_path / "ar.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        load_manifest(manifest, expected)


def test_unapproved_or_changed_manifest_rejected(tmp_path):
    manifest, data = bundle(tmp_path)
    expected = sha256(manifest)
    data["quality_approved"] = False
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(manifest, expected)
    with pytest.raises(ValueError, match="Approved"):
        load_manifest(manifest, sha256(manifest))


def test_checkpoint_symlink_and_extra_file_rejected(tmp_path):
    manifest, data = bundle(tmp_path)
    checkpoint = tmp_path / "ar.pt"
    external = tmp_path / "external.pt"
    checkpoint.rename(external)
    checkpoint.symlink_to(external)
    with pytest.raises(ValueError, match="integrity"):
        load_manifest(manifest, sha256(manifest))
    changed = copy.deepcopy(data)
    changed["files"]["../external.pt"] = data["files"]["ar.pt"]
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Approved"):
        load_manifest(manifest, sha256(manifest), check_files=False)


def test_only_explicit_lossless_cache_storage_allowed(tmp_path):
    manifest, data = bundle(tmp_path)
    for cache in ("fp32", "exact_fp16"):
        data["ar_cache_storage"] = cache
        manifest.write_text(json.dumps(data))
        assert load_manifest(manifest, sha256(manifest))["ar_cache_storage"] == cache
    data["ar_cache_storage"] = "fp16_arithmetic"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cache storage"):
        load_manifest(manifest, sha256(manifest))
