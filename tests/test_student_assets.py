"""Fail closed before loading a changed/unapproved private checkpoint bundle."""

import copy
import json

import pytest

from cpufast_student_assets import inline_runtime_approval, load_manifest, sha256


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


def test_declared_runtime_source_is_pinned_and_traversal_rejected(tmp_path):
    manifest, data = bundle(tmp_path)
    source = tmp_path / "experiments/student/listen_thin_decoder.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic approved implementation\n")
    data.update(
        ar_runtime="int8_native",
        ar_cache_storage="exact_fp16",
        model_variant="synthetic_width",
        scope="Synthetic unit case only",
        runtime_implementation={
            "path": "experiments/student/listen_thin_decoder.py",
            "sha256": sha256(source),
        },
    )
    manifest.write_text(json.dumps(data))
    verified = load_manifest(manifest, sha256(manifest))
    approval = inline_runtime_approval(verified, sha256(manifest), tmp_path)
    assert approval["underlying_manifest_sha256"] == sha256(manifest)
    source.write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="checksum"):
        inline_runtime_approval(verified, sha256(manifest), tmp_path)
    data["runtime_implementation"]["path"] = "experiments/student/../../outside.py"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Source-pinned"):
        load_manifest(manifest, sha256(manifest))


def test_optional_channel_selections_are_original_unique_sorted_ids(tmp_path):
    manifest, data = bundle(tmp_path)
    selected = [list(range(n)) for n in (640, 320, 160, 80, 40, 20)]
    data["decoder_channel_keep"] = selected
    manifest.write_text(json.dumps(data))
    assert load_manifest(manifest, sha256(manifest))["decoder_channel_keep"] == selected
    selected[-1].append(19)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="channel selection"):
        load_manifest(manifest, sha256(manifest))


def test_bottleneck_metadata_preserves_outer_geometry_and_unique_original_ids(tmp_path):
    manifest, data = bundle(tmp_path)
    data["decoder_branches"] = [[0, 1, 2]] * 5
    entry = {
        "stage": 3,
        "original_branch": 2,
        "pair": 1,
        "keep": list(range(36)),
        "inner_channels": 36,
        "outer_channels": 48,
        "kernel": 11,
        "dilation": 3,
    }
    data["decoder_bottlenecks"] = [entry]
    manifest.write_text(json.dumps(data))
    assert load_manifest(manifest, sha256(manifest))["decoder_bottlenecks"] == [entry]
    for changed in (
        {**entry, "outer_channels": 36},
        {**entry, "inner_channels": 35},
        {**entry, "kernel": 7},
        {**entry, "dilation": 5},
        {**entry, "keep": [0] * 36},
    ):
        data["decoder_bottlenecks"] = [changed]
        manifest.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="mismatch"):
            load_manifest(manifest, sha256(manifest))
    data["decoder_bottlenecks"] = [entry, entry]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        load_manifest(manifest, sha256(manifest))
