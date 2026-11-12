"""Serial actual-N1 evaluation of one immutable approved student bundle."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from cpufast_student_assets import load_manifest, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, args.manifest_sha256)
    before = snapshot(args.manifest)
    cases = {r["language"]: r for r in manifest["listening_report"]["cases"]}
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
    ):
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
            "/project/models/s1v3.ckpt"
            if original
            else str(args.manifest.parent / "ar.pt"),
            "--s2",
            "/repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
            "--hubert",
            "/project/models/chinese-hubert-base",
            "--reference",
            "/reference.wav",
            "--prompt-text",
            manifest["prompt_text"],
            "--prompt-lang",
            manifest["prompt_language"],
            "--lang",
            language,
            "--text",
            cases[language]["text"],
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
                "--ar-fp32-cache",
                "--student-manifest",
                str(args.manifest),
                "--student-manifest-sha256",
                args.manifest_sha256,
            ]
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
            "prefetch": 256 if original else distance,
            "original": original,
            "combined": combined,
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
            "peak_sampled_tree_rss_bytes": guard["peak_sampled_tree_rss_bytes"],
            "semantic": data["runs"][0]["semantic"],
            "candidate_tokens_exact_to_n1_student": tokens_exact,
            "candidate_pcm_exact_to_n1_student": None
            if compare is None
            else all(r["comparison"].get("exact", False) for r in data["runs"]),
            "local_approved_pcm_match": None
            if original
            else all(
                r["pcm_sha256"] == cases[language]["pcm_sha256"] for r in data["runs"]
            ),
            "local_approved_generated_tokens_match": None
            if original
            else all(
                [
                    {"tokens": r2["index"], "sha256": r2["generated_sha256"]}
                    for r2 in r["semantic"]
                ]
                == cases[language]["semantic"]
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
