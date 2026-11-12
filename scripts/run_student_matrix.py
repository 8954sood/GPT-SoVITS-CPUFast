"""Serial actual-N1 evaluation of one immutable approved student bundle."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from cpufast_student_assets import load_manifest, snapshot, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--prefetch", type=int, choices=[0, 64, 128, 256], default=0)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--baseline-manifest-sha256")
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--ar-int8-comparison", action="store_true")
    parser.add_argument("--ar-int8-confirm-only", action="store_true")
    args = parser.parse_args()
    if bool(args.baseline_manifest) != bool(args.baseline_manifest_sha256):
        parser.error("Comparison baseline manifest and SHA required together")
    if args.decoder_only and not args.baseline_manifest:
        parser.error("Decoder comparison requires an approved baseline bundle")
    if args.ar_int8_comparison and args.baseline_manifest:
        parser.error("AR runtime comparison uses one unchanged approved bundle")
    if args.ar_int8_confirm_only and not args.ar_int8_comparison:
        parser.error("Runtime confirmation requires AR-only runtime comparison mode")
    manifest = load_manifest(args.manifest, args.manifest_sha256)
    before = snapshot(args.manifest)
    cases = {r["language"]: r for r in manifest["listening_report"]["cases"]}
    baseline = None
    baseline_before = None
    if args.baseline_manifest:
        baseline = load_manifest(args.baseline_manifest, args.baseline_manifest_sha256)
        baseline_before = snapshot(args.baseline_manifest)
        baseline_ko = next(
            r for r in baseline["listening_report"]["cases"] if r["language"] == "ko"
        )
        if (
            any(
                baseline[k] != manifest[k]
                for k in (
                    "seed",
                    "prompt_text",
                    "prompt_language",
                    "reference_sha256",
                    "original_s2_sha256",
                )
            )
            or baseline_ko["text"] != cases["ko"]["text"]
        ):
            parser.error(
                "Model comparison must share the approved request and reference"
            )
        if args.decoder_only and (
            baseline["files"]["ar.pt"] != manifest["files"]["ar.pt"]
            or baseline.get("ar_cache_storage", "fp32")
            != manifest.get("ar_cache_storage", "fp32")
        ):
            parser.error(
                "Decoder comparison requires identical approved AR weights/cache"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    summary_file = args.output / "student_summary.json"
    if summary_file.exists():
        parser.error("Fresh output directory required")
    summary = {
        "scope": "Actual N1 original/student performance; user approval covers Korean listening only",
        "manifest_sha256": args.manifest_sha256,
        "ar_selected_additional_step": manifest["ar_selected_additional_step"],
        "decoder_selected_additional_step": manifest[
            "decoder_selected_additional_step"
        ],
        "prefetch_selection": "KO median RTF; retain 0 unless best hint improves at least 1%; provisional until independent verification",
        "cases": [],
    }

    def save():
        summary_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )

    def run(
        language,
        name,
        distance=0,
        combined=False,
        original=False,
        confirm=False,
        profile=False,
        compare=None,
        baseline_bundle=False,
        fp32_control=False,
        ar_int8_only=False,
    ):
        run_manifest = baseline if baseline_bundle else manifest
        run_path = args.baseline_manifest if baseline_bundle else args.manifest
        run_cases = {
            r["language"]: r for r in run_manifest["listening_report"]["cases"]
        }
        if (
            baseline is not None
            and run_manifest.get("ar_cache_storage", "fp32") == "exact_fp16"
            and not fp32_control
        ):
            # Use the verified original exact-half profile, not the FP32-cache hint.
            distance = 256
        file = args.output / f"{language}_{name}.json"
        command = [
            sys.executable,
            "/repo/benchmark_cpufast.py",
            "--engine",
            "--head-major",
            "--cook-phase",
            "--fft-layout",
            "--packed-fft",
            "--fused-finish",
            "--tap-small",
            "--ar-threads",
            "2",
            "--threads",
            "2",
            "--s1",
            "/project/models/s1v3.ckpt" if original else str(run_path.parent / "ar.pt"),
            "--s2",
            "/repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
            "--hubert",
            "/project/models/chinese-hubert-base",
            "--reference",
            "/reference.wav",
            "--prompt-text",
            run_manifest["prompt_text"],
            "--prompt-lang",
            run_manifest["prompt_language"],
            "--lang",
            language,
            "--text",
            run_cases[language]["text"],
            "--warmups",
            "1" if profile else str(args.warmups),
            "--samples",
            "1" if profile else ("5" if confirm else str(args.samples)),
            "--ar-prefetch",
            "256" if original else str(distance),
            "--output",
            str(file),
        ]
        if not original:
            command += [
                "--student-manifest",
                str(run_path),
                "--student-manifest-sha256",
                sha256(run_path),
            ]
            if run_manifest.get("ar_cache_storage", "fp32") == "fp32" or fp32_control:
                command += ["--ar-fp32-cache"]
            if fp32_control:
                command += ["--student-fp32-control"]
        if ar_int8_only:
            command += ["--ar-int8-only"]
        if combined:
            command += ["--fft-stream", "--parallel-transforms", "--inference-mode"]
        if compare is not None:
            command += ["--compare-pcm", str(compare)]
        if profile:
            command += [
                "--profile-output",
                str(args.output / "ko_student_profile_ops.json"),
                "--profile-no-shapes",
            ]
        guarded = [
            sys.executable,
            "/repo/resource_guard.py",
            "--report",
            str(file.with_name(file.stem + "_guard.json")),
            "--",
            *command,
        ]
        subprocess.run(guarded, check=True)
        data = json.loads(file.read_text())
        guard = json.loads(file.with_name(file.stem + "_guard.json").read_text())
        required = [
            "checkpoint_files_unchanged",
            "parameters_unchanged",
            "ar_parameters_unchanged",
        ]
        if not original:
            required.append("student_files_unchanged")
        if not all(data.get(k) for k in required) or guard["exit_code"] != 0:
            raise RuntimeError("Model/resource integrity failure")
        if ar_int8_only and (
            not data["approximate_ar_int8_only"]
            or data["fp16_acoustic_modules"] != 0
            or data["experimental_options"]["fast_experimental"]
        ):
            raise RuntimeError("Approximate AR runtime contaminated FP32 acoustics")
        if (
            data["cgroup"]["cpu.max"] != "200000 100000"
            or data["cgroup"]["memory.max"] != "8000000000"
            or data["cgroup"]["memory.swap.max"] != "0"
        ):
            raise RuntimeError("Actual N1 resource profile changed")
        if compare is not None:
            reference = json.loads(compare.with_suffix(".json").read_text())
            tokens_exact = all(
                r["semantic"] == reference["runs"][0]["semantic"]
                and r["audio_seconds"] == reference["runs"][0]["audio_seconds"]
                for r in data["runs"]
            )
        else:
            tokens_exact = None
        row = {
            "language": language,
            "name": name,
            "prefetch": None if ar_int8_only else (256 if original else distance),
            "approximate_ar_int8_only": ar_int8_only,
            "runtime_variant": "L_ar_int8_only"
            if ar_int8_only
            else data["model_variant"],
            "runtime_local_generated_tokens_match": all(
                r.get("runtime_local_generated_tokens_match", False)
                for r in data["runs"]
            )
            if ar_int8_only
            else None,
            "runtime_local_pcm_match": all(
                r.get("runtime_local_pcm_match", False) for r in data["runs"]
            )
            if ar_int8_only
            else None,
            "original": original,
            "combined": combined,
            "model_variant": data["model_variant"],
            "manifest_sha256": None if original else sha256(run_path),
            "ar_cache_storage": "int8_approximate"
            if ar_int8_only
            else "fp32"
            if fp32_control
            else (
                "exact_fp16"
                if original
                else run_manifest.get("ar_cache_storage", "fp32")
            ),
            "student_fp32_control": fp32_control,
            "median_rtf": data["median_rtf"],
            "rtfs": [r["rtf"] for r in data["runs"]],
            "median_seconds": data["median_seconds"],
            "median_cpu_seconds": statistics.median(
                r["cpu_seconds"] for r in data["runs"]
            ),
            "median_stages": {
                k: statistics.median(r["stages"][k] for r in data["runs"])
                for k in data["runs"][0]["stages"]
            },
            "audio_seconds": data["runs"][0]["audio_seconds"],
            "median_decoder_seconds_per_audio_second": statistics.median(
                r["stages"]["dec_seconds"] / r["audio_seconds"] for r in data["runs"]
            ),
            "peak_sampled_tree_rss_bytes": guard["peak_sampled_tree_rss_bytes"],
            "semantic": data["runs"][0]["semantic"],
            "candidate_tokens_exact_to_n1_student": tokens_exact,
            "candidate_pcm_exact_to_n1_student": None
            if compare is None
            else all(r["comparison"].get("exact", False) for r in data["runs"]),
            "local_approved_pcm_match": None
            if original
            else all(
                r["pcm_sha256"] == run_cases[language]["pcm_sha256"]
                for r in data["runs"]
            ),
            "local_approved_generated_tokens_match": None
            if original
            else all(
                [
                    {"tokens": r2["index"], "sha256": r2["generated_sha256"]}
                    for r2 in r["semantic"]
                ]
                == run_cases[language]["semantic"]
                for r in data["runs"]
            ),
        }
        if profile:
            summary["profile_run"] = row
        else:
            summary["cases"].append(row)
        save()
        print(
            f"Completed {language}/{name}: median RTF {row['median_rtf']:.4f}",
            flush=True,
        )
        return data, row

    if args.ar_int8_comparison:
        if manifest.get("ar_cache_storage", "fp32") != "exact_fp16":
            parser.error(
                "Approximate AR runtime requires an approved half-exact bundle"
            )
        summary["scope"] = (
            "Actual N1 separately listening-approved approximate AR-only runtime; unchanged approved checkpoint bytes and normal FP32 acoustics"
        )
        summary["prefetch_selection"] = (
            "Baseline verified half-cache256; candidate INT8 runtime has no half-cache prefetch"
        )
        if not args.ar_int8_confirm_only:
            run("ko", "ar_runtime_baseline", distance=256, combined=True, confirm=True)
        else:
            summary["scope"] = (
                "Independent actual N1 five-sample confirmation of separately approved L AR-only runtime"
            )
        run(
            "ko",
            "ar_runtime_candidate",
            distance=256,
            combined=True,
            confirm=True,
            ar_int8_only=True,
        )
        try:
            run(
                "ko",
                "ar_runtime_profile_run",
                distance=256,
                combined=True,
                profile=True,
                ar_int8_only=True,
                compare=args.output / "ko_ar_runtime_candidate.wav",
            )
        except subprocess.CalledProcessError as error:
            summary["profile_error"] = {"exit_code": error.returncode}
        summary["student_assets_unchanged"] = before == snapshot(args.manifest)
        summary["complete"] = True
        save()
        if not summary["student_assets_unchanged"]:
            raise RuntimeError("Private runtime comparison execution assets changed")
        return
    if baseline is not None:
        summary["scope"] = (
            "Actual N1 separately approved Korean student model comparison"
        )
        summary["prefetch_selection"] = (
            "Fixed FP32 cache distance; exact-half cache uses verified distance 256; no parameter search"
        )
        summary["selected_prefetch"] = args.prefetch
        summary["baseline_manifest_sha256"] = args.baseline_manifest_sha256
        baseline_data, _ = run(
            "ko",
            "comparison_baseline",
            distance=args.prefetch,
            combined=True,
            confirm=True,
            baseline_bundle=True,
        )
        candidate_compare = None
        if (
            manifest.get("ar_cache_storage", "fp32") == "exact_fp16"
            and not args.decoder_only
        ):
            run(
                "ko",
                "comparison_candidate_fp32",
                distance=args.prefetch,
                combined=True,
                confirm=True,
                fp32_control=True,
            )
            candidate_compare = args.output / "ko_comparison_candidate_fp32.wav"
        candidate_data, candidate_result = run(
            "ko",
            "comparison_candidate",
            distance=args.prefetch,
            combined=True,
            confirm=True,
            compare=candidate_compare,
        )
        if candidate_compare is not None and (
            not candidate_result["candidate_pcm_exact_to_n1_student"]
            or not candidate_result["candidate_tokens_exact_to_n1_student"]
        ):
            raise RuntimeError(
                "Same approved coefficients changed PCM/tokens between cache storage paths"
            )
        if args.decoder_only and not all(
            row["semantic"] == baseline_data["runs"][0]["semantic"]
            and row["audio_seconds"] == baseline_data["runs"][0]["audio_seconds"]
            for row in baseline_data["runs"] + candidate_data["runs"]
        ):
            raise RuntimeError(
                "Decoder-only model comparison changed semantic tokens/duration"
            )
        try:
            run(
                "ko",
                "comparison_profile_run",
                distance=args.prefetch,
                combined=True,
                profile=True,
                compare=args.output / "ko_comparison_candidate.wav",
            )
        except subprocess.CalledProcessError as error:
            summary["profile_error"] = {"exit_code": error.returncode}
        summary["student_assets_unchanged"] = before == snapshot(
            args.manifest
        ) and baseline_before == snapshot(args.baseline_manifest)
        summary["complete"] = True
        save()
        if not summary["student_assets_unchanged"]:
            raise RuntimeError("Private comparison execution assets changed")
        return
    run("ko", "original", original=True, combined=True)
    _, zero = run("ko", "student_p0")
    hints = [zero]
    for distance in (64, 128, 256):
        data, row = run(
            "ko",
            f"student_p{distance}",
            distance=distance,
            compare=args.output / "ko_student_p0.wav",
        )
        if (
            not row["candidate_pcm_exact_to_n1_student"]
            or not row["candidate_tokens_exact_to_n1_student"]
        ):
            raise RuntimeError(
                "Prefetch hint changed student PCM; inspect retained reports"
            )
        hints.append(row)
    winner = min(hints, key=lambda row: row["median_rtf"])
    selected = (
        winner["prefetch"] if winner["median_rtf"] < zero["median_rtf"] * 0.99 else 0
    )
    summary["selected_prefetch"] = selected
    save()
    _, combined = run(
        "ko",
        "student_combined",
        distance=selected,
        combined=True,
        confirm=True,
        profile=False,
        compare=args.output / "ko_student_p0.wav",
    )
    if (
        not combined["candidate_pcm_exact_to_n1_student"]
        or not combined["candidate_tokens_exact_to_n1_student"]
    ):
        raise RuntimeError(
            "Combined kernels changed student PCM; inspect retained reports"
        )
    for language in ("ja", "en"):
        run(language, "original", original=True, combined=True)
        run(language, "student_combined", distance=selected, combined=True)
    try:
        run(
            "ko",
            "student_profile_run",
            distance=selected,
            combined=True,
            profile=True,
            compare=args.output / "ko_student_p0.wav",
        )
    except subprocess.CalledProcessError as error:
        # Retain all completed RTF reports if the additional profiler exceeds its budget.
        summary["profile_error"] = {
            "exit_code": error.returncode,
            "scope": "Separate diagnostic; completed performance cases retained",
        }
    summary["student_assets_unchanged"] = before == snapshot(args.manifest)
    summary["complete"] = True
    save()
    if not summary["student_assets_unchanged"]:
        raise RuntimeError("Private student execution assets changed")


if __name__ == "__main__":
    main()
