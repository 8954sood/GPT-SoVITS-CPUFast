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
        "decoder_branches": [[1, 2], [0, 1, 2], [0, 2], [1, 2], [0, 1, 2]],
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
    "baseline_storage,candidate_storage,decoder_only,ar_only",
    [
        ("fp32", "fp32", False, False),
        ("fp32", "exact_fp16", False, False),
        ("exact_fp16", "exact_fp16", True, False),
        ("exact_fp16", "exact_fp16", True, "decoder"),
        ("exact_fp16", "exact_fp16", False, True),
        ("exact_fp16", "exact_fp16", False, "confirm"),
        ("exact_fp16", "exact_fp16", False, "fft"),
        ("exact_fp16", "exact_fp16", False, "stream"),
        ("exact_fp16", "exact_fp16", True, "bottleneck"),
        ("exact_fp16", "exact_fp16", True, "bottleneckU"),
        ("exact_fp16", "exact_fp16", True, "bottleneckV"),
        ("exact_fp16", "exact_fp16", False, "packed"),
        ("exact_fp16", "exact_fp16", False, "int8packed"),
        ("exact_fp16", "exact_fp16", False, "outputtile4"),
        ("exact_fp16", "exact_fp16", False, "profile"),
    ],
)
def test_comparison_uses_each_approved_bundle_and_matching_cache(
    monkeypatch, tmp_path, baseline_storage, candidate_storage, decoder_only, ar_only
):
    baseline = synthetic_bundle(tmp_path / "baseline", "H2", baseline_storage)
    candidate = synthetic_bundle(tmp_path / "candidate", "H3", candidate_storage)
    if decoder_only:
        (candidate.parent / "ar.pt").write_bytes(
            (baseline.parent / "ar.pt").read_bytes()
        )
        data = json.loads(candidate.read_text())
        data["files"]["ar.pt"] = json.loads(baseline.read_text())["files"]["ar.pt"]
        if ar_only == "decoder":
            data["decoder_channel_keep"] = [
                list(range(n)) for n in (672, 336, 168, 84, 42, 21)
            ]
            data["ar_runtime"] = "int8_native"
            data["runtime_implementation"] = {
                "path": "experiments/student/synthetic.py",
                "sha256": "a" * 64,
            }
        if ar_only in ("bottleneck", "bottleneckU", "bottleneckV"):
            data["decoder_bottlenecks"] = [
                {
                    "stage": 3,
                    "original_branch": 2,
                    "pair": 1,
                    "keep": list(range(36)),
                    "inner_channels": 36,
                    "outer_channels": 48,
                    "kernel": 11,
                    "dilation": 3,
                }
            ]
        if ar_only in ("bottleneckU", "bottleneckV"):
            data["model_variant"] = ar_only[-1]
        candidate.write_text(json.dumps(data))
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
    ]
    if ar_only and ar_only not in ("decoder", "bottleneck", "bottleneckU", "bottleneckV"):
        for key in ("--baseline-manifest", "--baseline-manifest-sha256"):
            index = argv.index(key)
            del argv[index : index + 2]
        argv.append(
            "--fft-output-tiles-comparison"
            if ar_only == "outputtile4"
            else "--int8-prefill-comparison"
            if ar_only == "int8packed"
            else "--diagnostic-profile-only"
            if ar_only == "profile"
            else "--packed-prefill-comparison"
            if ar_only == "packed"
            else "--fft-half-stream-comparison"
            if ar_only == "stream"
            else "--fft-half-comparison"
            if ar_only == "fft"
            else "--ar-int8-comparison"
        )
        if ar_only == "confirm":
            argv.append("--ar-int8-confirm-only")
        if ar_only in ("int8packed", "outputtile4"):
            argv += ["--samples", "5"]
    if decoder_only:
        argv.append("--decoder-only")
    if ar_only in ("decoder", "bottleneck", "bottleneckU", "bottleneckV"):
        argv.append("--decoder-ar-int8")
    if ar_only in ("bottleneck", "bottleneckU", "bottleneckV"):
        argv.append("--alternating-decoder-comparison")
        if ar_only in ("bottleneckU", "bottleneckV"):
            argv += ["--decoder-candidate-label", ar_only[-1], "--samples", "5"]
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
            "approximate_ar_int8_only": "--ar-int8-only" in command,
            "int8_prefill_ffn": "--int8-prefill-ffn" in command,
            "int8_prefill_audit": [{"block": 0}]
            if "--int8-prefill-ffn" in command
            else [],
            "int8_prefill_library_manifest": {"source": "trueGEMM"}
            if "--int8-prefill-ffn" in command
            else None,
            "packed_prefill_ffn": "--packed-prefill-ffn" in command,
            "packed_prefill_audit": [{"block": 0}]
            if "--packed-prefill-ffn" in command
            else [],
            "approximate_fft_half_products": "--fft-half-products" in command,
            "approximate_fft_half_stream": "--fft-half-stream" in command,
            "fft_half_product_modules": [
                {
                    "module": "large",
                    "half_stream_products": "--fft-half-stream" in command,
                }
            ]
            if "--fft-half-products" in command
            else [],
            "fp16_acoustic_modules": 1 if "--fft-half-products" in command else 0,
            "experimental_options": {"fast_experimental": False},
            "fft_output_tiles": {"calls": 120} if "/repo/scripts/benchmark_fft_output_tiles.py" in command else {},
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
    if ar_only == "profile":
        assert len(calls) == 1
        command = calls[0]
        assert command[command.index("--warmups") + 1] == "3"
        assert command[command.index("--samples") + 1] == "1"
        assert "--ar-int8-only" in command and "--profile-output" in command
        assert "--packed-prefill-ffn" not in command
        assert "--fft-half-products" not in command
        summary = json.loads((output / "student_summary.json").read_text())
        assert summary["cases"] == [] and "profile_run" in summary
        assert summary["complete"] and summary["student_assets_unchanged"]
        return
    if ar_only in ("bottleneck", "bottleneckU", "bottleneckV", "packed", "int8packed", "outputtile4"):
        assert len(calls) == 11
        candidate_label = (
            "tile4"
            if ar_only == "outputtile4"
            else "W"
            if ar_only == "int8packed"
            else "packed"
            if ar_only == "packed"
            else ar_only[-1]
            if ar_only in ("bottleneckU", "bottleneckV")
            else "R"
        )
        order = [
            "L",
            candidate_label,
            candidate_label,
            "L",
            "L",
            candidate_label,
            candidate_label,
            "L",
            "L",
            candidate_label,
        ]
        for label, command in zip(order, calls):
            path = (
                baseline
                if label == "L" and ar_only in ("bottleneck", "bottleneckU", "bottleneckV")
                else candidate
            )
            assert command[command.index("--s1") + 1] == str(path.parent / "ar.pt")
            assert command[command.index("--samples") + 1] == "1"
            assert (
                "--ar-int8-only" in command
                and "--fft-half-products" not in command
                and "--fft-pad-channels" not in command
            )
            assert ("--packed-prefill-ffn" in command) == (label == "packed")
            assert ("--int8-prefill-ffn" in command) == (label == "W")
            assert ("/repo/scripts/benchmark_fft_output_tiles.py" in command) == (label == "tile4")
        if ar_only in ("bottleneckU", "bottleneckV", "int8packed", "outputtile4"):
            assert calls[-1][calls[-1].index("--warmups") + 1] == "3"
        summary = json.loads((output / "student_summary.json").read_text())
        assert [r["runtime"] for r in summary["comparison_order"]] == order
        assert [r["samples"] for r in summary["paired_aggregate"]] == [5, 5]
        assert summary["complete"] and summary["student_assets_unchanged"]
        return
    one_bundle_runtime = bool(ar_only) and ar_only != "decoder"
    has_control = candidate_storage == "exact_fp16" and not decoder_only and not ar_only
    confirm_only = ar_only == "confirm"
    assert len(calls) == (2 if confirm_only else 4 if has_control else 3)
    for index, command in enumerate(calls):
        bundle = baseline if index == 0 and not one_bundle_runtime else candidate
        storage = (
            baseline_storage
            if index == 0 and not one_bundle_runtime
            else candidate_storage
        )
        if has_control and index == 1:
            storage = "fp32"
        assert command[command.index("--s1") + 1] == str(bundle.parent / "ar.pt")
        assert ("--ar-fp32-cache" in command) == (storage == "fp32")
        assert ("--fft-pad-channels" in command) == (ar_only == "decoder" and index > 0)
        assert command[command.index("--ar-prefetch") + 1] == (
            "0" if storage == "fp32" else "256"
        )
        assert command[command.index("--samples") + 1] == (
            "1" if index == len(calls) - 1 else "5"
        )
        assert ("--student-fp32-control" in command) == (has_control and index == 1)
        assert ("--compare-pcm" in command) == (
            index >= 2
            or (confirm_only and index == 1)
            or (ar_only == "stream" and index == 1)
        )
        assert ("--ar-int8-only" in command) == bool(
            ar_only
            and (index > 0 or confirm_only or ar_only in ("decoder", "fft", "stream"))
        )
        assert ("--fft-half-products" in command) == (
            (ar_only == "fft" and index > 0) or ar_only == "stream"
        )
        assert ("--fft-half-stream" in command) == (ar_only == "stream" and index > 0)
    assert calls[-1][calls[-1].index("--compare-pcm") + 1] == str(
        output
        / (
            "ko_fft_product_candidate.wav"
            if ar_only in ("fft", "stream")
            else "ko_ar_runtime_candidate.wav"
            if one_bundle_runtime
            else "ko_comparison_candidate.wav"
        )
    )
    summary = json.loads((output / "student_summary.json").read_text())
    assert summary["complete"] and summary["student_assets_unchanged"]
    assert [r["manifest_sha256"] for r in summary["cases"]] == (
        [sha256(candidate)]
        if confirm_only
        else [
            sha256(candidate if one_bundle_runtime else baseline),
            *([sha256(candidate)] * (2 if has_control else 1)),
        ]
    )
    assert [r["model_variant"] for r in summary["cases"]] == (
        ["H3"]
        if confirm_only
        else [
            "H3" if one_bundle_runtime else "H2",
            *(["H3"] * (2 if has_control else 1)),
        ]
    )


def test_decoder_comparison_rejects_different_ar_before_inference(
    monkeypatch, tmp_path
):
    baseline = synthetic_bundle(tmp_path / "baseline", "H3", "exact_fp16")
    candidate = synthetic_bundle(tmp_path / "candidate", "K", "exact_fp16")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matrix",
            "--manifest",
            str(candidate),
            "--manifest-sha256",
            sha256(candidate),
            "--baseline-manifest",
            str(baseline),
            "--baseline-manifest-sha256",
            sha256(baseline),
            "--decoder-only",
            "--output",
            str(tmp_path / "out"),
        ],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Different AR model reached inference")

    monkeypatch.setattr(run_student_matrix.subprocess, "run", forbidden)
    with pytest.raises(SystemExit):
        run_student_matrix.main()


@pytest.mark.parametrize("corruption", [None, "decoder", "pcm", "blocks"])
def test_ar_depth_separates_approved_lengths_and_guards(monkeypatch, tmp_path, corruption):
    paths = [synthetic_bundle(tmp_path / "baseline", "V", "exact_fp16"),
             synthetic_bundle(tmp_path / "candidate", "AR17_QAT200_V400", "exact_fp16")]
    shared_decoder = (paths[0].parent / "decoder.pt").read_bytes()
    for index, path in enumerate(paths):
        (path.parent / "decoder.pt").write_bytes(shared_decoder)
        data = json.loads(path.read_text())
        data.update(ar_layers=18-index, ar_runtime="int8_native",
                    runtime_implementation={"path": "experiments/student/synthetic.py", "sha256": "a"*64})
        data["files"]["decoder.pt"] = {"bytes": len(shared_decoder), "sha256": sha256(path.parent / "decoder.pt")}
        data.pop("ar_selected_additional_step")  # New model training metadata is independent.
        for case in data["listening_report"]["cases"]:
            case.update(audio_seconds=1.0+index, pcm_sha256=f"pcm{index}",
                        semantic=[{"tokens": index+1, "sha256": f"semantic{index}"}])
        path.write_text(json.dumps(data))
    if corruption == "decoder":
        path = paths[1]
        (path.parent / "decoder.pt").write_text("different")
        data = json.loads(path.read_text())
        data["files"]["decoder.pt"] = {"bytes": 9, "sha256": sha256(path.parent / "decoder.pt")}
        path.write_text(json.dumps(data))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["matrix", "--ar-depth-comparison", "--warmups", "3", "--samples", "5",
        "--manifest", str(paths[1]), "--manifest-sha256", sha256(paths[1]),
        "--baseline-manifest", str(paths[0]), "--baseline-manifest-sha256", sha256(paths[0]),
        "--output", str(output)])
    calls = []
    def execute(command, check):
        assert check and corruption != "decoder"
        calls.append(command)
        value = lambda key: command[command.index(key)+1]
        manifest = json.loads(Path(value("--student-manifest")).read_text())
        index = 18-manifest["ar_layers"]
        assert value("--warmups") == "3" and value("--samples") == "1"
        assert "--ar-int8-only" in command and "--fft-half-products" not in command
        file = Path(value("--output"))
        data = {"checkpoint_files_unchanged": True, "parameters_unchanged": True,
            "ar_parameters_unchanged": True, "student_files_unchanged": True,
            "approximate_ar_int8_only": True, "experimental_options": {"fast_experimental": False},
            "fp16_acoustic_modules": 0, "model_variant": manifest["model_variant"],
            "median_rtf": 1.2, "median_seconds": 1.2*(1+index),
            "cgroup": {"cpu.max": "200000 100000", "memory.max": "8000000000", "memory.swap.max": "0"},
            "ar_coefficient_audit": {"matrices": [{"block": b} for b in range(manifest["ar_layers"]-(corruption == "blocks"))]},
            "runs": [{"rtf": 1.2, "cpu_seconds": 2., "audio_seconds": float(1+index),
                "stages": {"ar_seconds": .5, "dec_seconds": .6},
                "pcm_sha256": "bad" if corruption == "pcm" else f"pcm{index}",
                "semantic": [{"index": 1+index, "generated_sha256": f"semantic{index}"}],
                "comparison": {"exact": True}}]}
        file.write_text(json.dumps(data))
        file.with_suffix(".wav").write_bytes(b"wav")
        Path(value("--report")).write_text(json.dumps({"exit_code": 0, "peak_sampled_tree_rss_bytes": 100}))
        if "--profile-output" in command:
            Path(value("--profile-output")).write_text("{}")
    monkeypatch.setattr(run_student_matrix.subprocess, "run", execute)
    if corruption:
        with pytest.raises(SystemExit if corruption == "decoder" else RuntimeError):
            run_student_matrix.main()
        assert len(calls) == (0 if corruption == "decoder" else 1)
        return
    run_student_matrix.main()
    summary = json.loads((output / "student_summary.json").read_text())
    assert len(calls) == 12 and summary["complete"] and summary["student_assets_unchanged"]
    assert [r["runtime"] for r in summary["comparison_order"]] == [
        "AR18_V400", "AR17_QAT200_V400", "AR17_QAT200_V400", "AR18_V400",
        "AR18_V400", "AR17_QAT200_V400", "AR17_QAT200_V400", "AR18_V400", "AR18_V400", "AR17_QAT200_V400"]
    assert [r["audio_seconds"] for r in summary["paired_aggregate"]] == [1., 2.]
    assert [r["generated_tokens"] for r in summary["paired_aggregate"]] == [1, 2]
    assert [r["median_ar_seconds_per_generated_token"] for r in summary["paired_aggregate"]] == [.5, .25]
    assert len(summary["excluded_profiles"]) == 2
