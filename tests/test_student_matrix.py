"""Exercise private model comparison boundaries without loading model weights."""

import json
import sys
from pathlib import Path

import pytest

from cpufast_student_assets import sha256
from scripts import run_student_matrix


def synthetic_bundle(root, variant, storage):
    root.mkdir()
    files = {}
    for name in ("ar.pt", "decoder.pt", "student_spec.json"):
        file = root / name
        file.write_text(variant + name)
        files[name] = {"bytes": file.stat().st_size, "sha256": sha256(file)}
    manifest = {
        "quality_approved": True,
        "model_variant": variant,
        "ar_cache_storage": storage,
        "ar_selected_additional_step": 25,
        "decoder_selected_additional_step": 175,
        "seed": 42,
        "prompt_text": "Same reference transcript",
        "prompt_language": "ja",
        "reference_sha256": "a" * 64,
        "original_s2_sha256": "b" * 64,
        "files": files,
        "listening_report": {
            "cases": [
                {
                    "language": lang,
                    "text": lang + " same test",
                    "sample_rate": 32000,
                    "audio_seconds": 1.0,
                    "pcm_sha256": "pcm",
                    "semantic": [{"tokens": 1, "sha256": "semantic"}],
                }
                for lang in ("ko", "ja", "en")
            ]
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize(
    "baseline_storage,candidate_storage", [("fp32", "fp32"), ("fp32", "exact_fp16")]
)
def test_comparison_uses_each_approved_bundle_and_matching_cache(
    monkeypatch, tmp_path, baseline_storage, candidate_storage
):
    baseline = synthetic_bundle(tmp_path / "baseline", "H2", baseline_storage)
    candidate = synthetic_bundle(tmp_path / "candidate", "H3", candidate_storage)
    output = tmp_path / "output"
    argv = [
        "matrix",
        "--manifest",
        str(candidate),
        "--manifest-sha256",
        sha256(candidate),
        "--baseline-manifest",
        str(baseline),
        "--baseline-manifest-sha256",
        sha256(baseline),
        "--output",
        str(output),
        "--prefetch",
        "0",
        "--predictor-cache",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def execute(command, check):
        assert check
        calls.append(command)

        def value(key):
            return command[command.index(key) + 1]

        manifest = json.loads(Path(value("--student-manifest")).read_text())
        file = Path(value("--output"))
        data = {
            "checkpoint_files_unchanged": True,
            "parameters_unchanged": True,
            "ar_parameters_unchanged": True,
            "student_files_unchanged": True,
            "model_variant": manifest["model_variant"],
            "median_rtf": 1.2,
            "median_seconds": 1.2,
            "cgroup": {
                "cpu.max": "200000 100000",
                "memory.max": "8000000000",
                "memory.swap.max": "0",
            },
            "runs": [
                {
                    "rtf": 1.2,
                    "cpu_seconds": 2.0,
                    "stages": {"ar_seconds": 0.5, "dec_seconds": 0.6},
                    "audio_seconds": 1.0,
                    "pcm_sha256": "pcm",
                    "semantic": [{"index": 1, "generated_sha256": "semantic"}],
                    "comparison": {"exact": True},
                }
            ],
        }
        file.write_text(json.dumps(data))
        Path(value("--report")).write_text(
            json.dumps({"exit_code": 0, "peak_sampled_tree_rss_bytes": 100})
        )

    monkeypatch.setattr(run_student_matrix.subprocess, "run", execute)
    run_student_matrix.main()
    has_control = candidate_storage == "exact_fp16"
    assert len(calls) == (4 if has_control else 3)
    for index, command in enumerate(calls):
        bundle = baseline if index == 0 else candidate
        storage = baseline_storage if index == 0 else candidate_storage
        if has_control and index == 1:
            storage = "fp32"
        assert command[command.index("--s1") + 1] == str(bundle.parent / "ar.pt")
        assert ("--ar-fp32-cache" in command) == (storage == "fp32")
        assert command[command.index("--ar-prefetch") + 1] == (
            "0" if storage == "fp32" else "256"
        )
        assert "--predictor-eligibility-cache" in command
        assert command[command.index("--samples") + 1] == (
            "1" if index == len(calls) - 1 else "5"
        )
        assert ("--student-fp32-control" in command) == (has_control and index == 1)
        assert ("--compare-pcm" in command) == (index >= 2)
    assert calls[-1][calls[-1].index("--compare-pcm") + 1] == str(
        output / "ko_comparison_candidate.wav"
    )
    summary = json.loads((output / "student_summary.json").read_text())
    assert summary["complete"] and summary["student_assets_unchanged"]
    assert [r["manifest_sha256"] for r in summary["cases"]] == [
        sha256(baseline),
        *([sha256(candidate)] * (2 if has_control else 1)),
    ]
    assert [r["model_variant"] for r in summary["cases"]] == [
        "H2",
        *(["H3"] * (2 if has_control else 1)),
    ]
