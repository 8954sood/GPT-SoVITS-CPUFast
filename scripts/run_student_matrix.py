"""Serial actual-N1 evaluation of one immutable approved student bundle."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from cpufast_student_assets import load_manifest, sha256, snapshot


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
    parser.add_argument("--head-v32-comparison", action="store_true")
    parser.add_argument("--ar-depth-comparison", action="store_true")
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--decoder-ar-int8", action="store_true")
    parser.add_argument("--ar-int8-comparison", action="store_true")
    parser.add_argument("--ar-int8-confirm-only", action="store_true")
    parser.add_argument("--fft-half-comparison", action="store_true")
    parser.add_argument("--fft-half-stream-comparison", action="store_true")
    parser.add_argument("--alternating-decoder-comparison", action="store_true")
    parser.add_argument("--packed-prefill-comparison", action="store_true")
    parser.add_argument("--int8-prefill-comparison", action="store_true")
    parser.add_argument("--diagnostic-profile-only", action="store_true")
    parser.add_argument("--fft-output-tiles-comparison", action="store_true")
    parser.add_argument("--decoder-candidate-label", choices=["R", "U", "V"], default="R")
    args = parser.parse_args()
    if args.ar_depth_comparison and (
        not args.baseline_manifest or args.warmups != 3 or args.samples != 5
        or any((args.decoder_only, args.decoder_ar_int8, args.ar_int8_comparison,
                args.ar_int8_confirm_only, args.fft_half_comparison,
                args.fft_half_stream_comparison, args.alternating_decoder_comparison,
                args.packed_prefill_comparison, args.int8_prefill_comparison,
                args.diagnostic_profile_only, args.fft_output_tiles_comparison, args.head_v32_comparison))
        or args.decoder_candidate_label != "R"
    ):
        parser.error("AR depth comparison requires isolated two approved bundles, 3warm5sample")
    prefill_comparison = args.packed_prefill_comparison or args.int8_prefill_comparison
    runtime_comparison = prefill_comparison or args.fft_output_tiles_comparison or args.head_v32_comparison
    if args.head_v32_comparison and (
        prefill_comparison or args.fft_output_tiles_comparison or args.diagnostic_profile_only
        or args.warmups != 3 or args.samples != 5
    ):
        parser.error("Head scheduling requires isolated3warm5sample comparison")
    if args.fft_output_tiles_comparison and (
        prefill_comparison or args.warmups != 3 or args.samples != 5
    ):
        parser.error("FFT output tiling requires isolated3warm5sample comparison")
    if args.int8_prefill_comparison and (
        args.packed_prefill_comparison or args.warmups != 3 or args.samples != 5
    ):
        parser.error("W requires one counterbalanced3warm5sample prefill comparison")
    if args.decoder_candidate_label in ("U", "V") and (
        not args.alternating_decoder_comparison
        or args.warmups != 3
        or args.samples != 5
    ):
        parser.error(
            "U/V requires counterbalanced decoder comparison with exactly3warm5samples"
        )
    if args.diagnostic_profile_only and args.warmups != 3:
        parser.error("Diagnostic profile uses exactly three warmups")
    if args.diagnostic_profile_only and (
        args.baseline_manifest
        or args.decoder_only
        or args.ar_int8_comparison
        or args.fft_half_comparison
        or args.fft_half_stream_comparison
        or args.packed_prefill_comparison
        or args.int8_prefill_comparison
        or args.fft_output_tiles_comparison
        or args.ar_int8_confirm_only
        or args.decoder_ar_int8
        or args.alternating_decoder_comparison
    ):
        parser.error("Diagnostic profile requires one unchanged approved L bundle")
    if bool(args.baseline_manifest) != bool(args.baseline_manifest_sha256):
        parser.error("Comparison baseline manifest and SHA required together")
    if args.decoder_only and not args.baseline_manifest:
        parser.error("Decoder comparison requires an approved baseline bundle")
    if (args.ar_int8_comparison) and args.baseline_manifest:
        parser.error("AR runtime comparison uses one unchanged approved bundle")
    if args.ar_int8_confirm_only and not args.ar_int8_comparison:
        parser.error("Runtime confirmation requires AR-only runtime comparison mode")
    if args.decoder_ar_int8 and not args.decoder_only:
        parser.error("Decoder AR INT8 mode requires a same-AR decoder-only comparison")
    if args.alternating_decoder_comparison and not (
        args.decoder_only and args.decoder_ar_int8 and args.baseline_manifest
    ):
        parser.error(
            "Alternating decoder comparison requires two same-AR approved INT8 bundles"
        )
    if (args.fft_half_comparison or args.fft_half_stream_comparison) and (
        args.baseline_manifest or args.ar_int8_comparison or args.decoder_only
    ):
        parser.error("FFT-product comparison uses one unchanged approved L bundle")
    if args.fft_half_comparison and args.fft_half_stream_comparison:
        parser.error("Select one FFT-product comparison")
    if runtime_comparison and (
        args.baseline_manifest
        or args.decoder_only
        or args.ar_int8_comparison
        or args.fft_half_comparison
        or args.fft_half_stream_comparison
    ):
        parser.error("Packed prefill comparison uses one unchanged approved L bundle")
    manifest = load_manifest(args.manifest, args.manifest_sha256)
    if args.decoder_candidate_label in ("U", "V") and (
        manifest.get("model_variant") != args.decoder_candidate_label
    ):
        parser.error("Decoder label must match the explicitly approved manifest")
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
        if args.alternating_decoder_comparison and (
            baseline["decoder_branches"] != manifest["decoder_branches"]
            or baseline.get("decoder_channel_keep") is not None
            or manifest.get("decoder_channel_keep") is not None
            or not manifest.get("decoder_bottlenecks")
        ):
            parser.error(
                "First bottleneck comparison must preserve original outer channels and branches"
            )
    if args.ar_depth_comparison:
        for key in ("decoder.pt",):
            if baseline["files"][key] != manifest["files"][key]:
                parser.error("AR depth comparison must preserve decoder bytes")
        for key in ("decoder_branches", "decoder_bottlenecks", "decoder_channel_keep",
                    "ar_runtime", "ar_cache_storage", "original_s1_sha256"):
            if baseline.get(key) != manifest.get(key):
                parser.error("AR depth comparison changed shared decoder/runtime/original: " + key)
        if (baseline.get("ar_layers") != 18 or manifest.get("ar_layers") != 17
            or manifest.get("ar_runtime") != "int8_native"
            or manifest.get("ar_cache_storage") != "exact_fp16"):
            parser.error("This approved depth comparison requires native AR18 versus AR17")
    args.output.mkdir(parents=True, exist_ok=True)
    summary_file = args.output / "student_summary.json"
    if summary_file.exists():
        parser.error("Fresh output directory required")
    summary = {
        "scope": "Actual N1 original/student performance; user approval covers Korean listening only",
        "manifest_sha256": args.manifest_sha256,
        "ar_selected_additional_step": manifest.get("ar_selected_additional_step"),
        "decoder_selected_additional_step": manifest[
            "decoder_selected_additional_step"
        ],
        "ar_training": manifest.get("ar_training"),
        "ar_selected_step": manifest.get("ar_selected_step"),
        "ar_recovery_updates": manifest.get("ar_recovery_updates"),
        "ar_control_parent_updates": manifest.get("ar_control_parent_updates"),
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
        fft_half_products=False,
        fft_half_stream=False,
        measured_samples=None,
        packed_prefill=False,
        int8_prefill=False,
        fft_output_tiles=False,
        head_v32=False,
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
            "1"
            if profile
            and not (
                args.diagnostic_profile_only
                or args.ar_depth_comparison
                or args.head_v32_comparison
                or args.decoder_candidate_label in ("U", "V")
                or args.int8_prefill_comparison
                or args.fft_output_tiles_comparison
            )
            else str(args.warmups),
            "--samples",
            "1"
            if profile
            else str(measured_samples)
            if measured_samples is not None
            else ("5" if confirm else str(args.samples)),
            "--ar-prefetch",
            "256" if original else str(distance),
            "--output",
            str(file),
        ]
        if fft_output_tiles:
            command[1] = "/repo/scripts/benchmark_fft_output_tiles.py"
            command += [
                "--fft-output-tile-library", "/repo/libcpufast_fft_output_tiles.so"
            ]
        if head_v32:
            command[1] = "/repo/scripts/benchmark_head_v32.py"
            command += ["--head-v32-library", "/repo/libcpufast_head_v32_runtime.so"]
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
        if not original and run_manifest.get("decoder_channel_keep") is not None:
            command += ["--fft-pad-channels"]
        if ar_int8_only:
            command += ["--ar-int8-only"]
        if fft_half_products:
            command += ["--fft-half-products"]
        if fft_half_stream:
            command += ["--fft-half-stream"]
        if packed_prefill:
            command += ["--packed-prefill-ffn"]
        if int8_prefill:
            command += ["--int8-prefill-ffn"]
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
            or data["experimental_options"]["fast_experimental"]
        ):
            raise RuntimeError("Approximate AR runtime contaminated FP32 acoustics")
        if fft_half_products:
            if (
                not data.get("approximate_fft_half_products")
                or not data.get("fft_half_product_modules")
                or data["fp16_acoustic_modules"]
                != len(data["fft_half_product_modules"])
            ):
                raise RuntimeError("FFT-product candidate scope audit failed")
        elif ar_int8_only and data["fp16_acoustic_modules"] != 0:
            raise RuntimeError("AR-only runtime changed FP32 acoustics")
        if fft_half_stream and (
            not data.get("approximate_fft_half_stream")
            or not all(
                row.get("half_stream_products")
                for row in data["fft_half_product_modules"]
            )
        ):
            raise RuntimeError("Half-stream candidate scope audit failed")
        if packed_prefill and (
            not data.get("packed_prefill_ffn") or not data.get("packed_prefill_audit")
        ):
            raise RuntimeError("Packed FP32 prefill candidate scope audit failed")
        if fft_output_tiles and not data.get("fft_output_tiles", {}).get("calls"):
            raise RuntimeError("Output-tiling candidate did not use the native probe")
        if head_v32 and not data.get("head_v32", {}).get("calls"):
            raise RuntimeError("Head scheduling candidate did not use the native probe")
        if int8_prefill and (
            not data.get("int8_prefill_ffn")
            or not data.get("int8_prefill_audit")
            or not data.get("int8_prefill_library_manifest")
        ):
            raise RuntimeError("W8A8 prefill candidate scope/library audit failed")
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
            "approximate_fft_half_products": fft_half_products,
            "approximate_fft_half_stream": fft_half_stream,
            "fft_half_product_modules": data.get("fft_half_product_modules", []),
            "packed_prefill_ffn": packed_prefill,
            "int8_prefill_ffn": int8_prefill,
            "runtime_variant": "W_L_with_multitoken_W8A8_prefill"
            if int8_prefill
            else "L_with_packed_FP32_prefill"
            if packed_prefill
            else "P_equivalent_half_stream"
            if fft_half_stream
            else "P_fft_half_products"
            if fft_half_products
            else "L_ar_int8_only"
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
            "fft_pad_channels": not original
            and run_manifest.get("decoder_channel_keep") is not None,
            "fft_padded_output_cache": data.get("fft_padded_output_cache", []),
            "cold_load_seconds": data.get("load_seconds"),
            "installation_seconds": data.get("installation_seconds"),
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
        if args.head_v32_comparison and not (
            row["local_approved_pcm_match"] and row["local_approved_generated_tokens_match"]
            and all(r["audio_seconds"] == run_cases[language]["audio_seconds"] for r in data["runs"])
        ):
            raise RuntimeError("Head scheduling comparison did not reproduce approved model PCM/tokens")
        if args.ar_depth_comparison:
            tokens = sum(int(r["index"]) for r in data["runs"][0]["semantic"])
            audit = data["ar_coefficient_audit"]["matrices"]
            blocks = sorted({int(r["block"]) for r in audit})
            if blocks != list(range(run_manifest["ar_layers"])):
                raise RuntimeError("Actual AR block audit does not match approved depth")
            if not (row["local_approved_pcm_match"] and row["local_approved_generated_tokens_match"]
                    and all(r["audio_seconds"] == run_cases[language]["audio_seconds"] for r in data["runs"])):
                raise RuntimeError("AR comparison did not reproduce its own approved voice")
            row.update({
                "runtime_variant": "unchanged_native_W8A8_FP32_acoustics",
                "generated_tokens": tokens,
                "actual_ar_blocks": blocks,
                "median_ar_seconds_per_generated_token": statistics.median(
                    r["stages"]["ar_seconds"] / sum(int(x["index"]) for x in r["semantic"])
                    for r in data["runs"]),
            })
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

    if args.diagnostic_profile_only:
        summary["scope"] = (
            "Excluded actual N1 unchanged approved L diagnostic only; three warmups, "
            "one unprofiled PCM reference and one separate operator profile; no speed decision"
        )
        summary["prefetch_selection"] = "No tuning or alternative arithmetic"
        run(
            "ko",
            "L_diagnostic_profile_run",
            distance=256,
            combined=True,
            ar_int8_only=True,
            profile=True,
        )
        summary["student_assets_unchanged"] = before == snapshot(args.manifest)
        summary["complete"] = True
        save()
        if not summary["student_assets_unchanged"]:
            raise RuntimeError("Private diagnostic assets changed")
        return
    if args.ar_depth_comparison:
        summary["scope"] = (
            "Actual N1 owner-approved AR18 V400 versus AR17 QAT200 same V400; "
            "fresh processes, 3 warmups excluded per process, 5 AB/BA pairs. "
            "Different approved token counts/audio lengths; RTF is not isolated layer compute gain."
        )
        summary["prefetch_selection"] = "No search; same native W8A8 AR and normal FP32 acoustics"
        summary["comparison_order"] = []
        samples = {"AR18_V400": [], "AR17_QAT200_V400": []}
        for index in range(5):
            order = list(samples) if index % 2 == 0 else list(reversed(samples))
            for label in order:
                data, row = run("ko", f"paired_{index}_{label}", distance=256,
                    combined=True, ar_int8_only=True, measured_samples=1,
                    baseline_bundle=label == "AR18_V400")
                observed = data["runs"][0]
                if samples[label] and any(
                    observed[k] != samples[label][0][0][k]
                    for k in ("semantic", "audio_seconds", "pcm_sha256")
                ):
                    raise RuntimeError("Same-model process changed approved tokens/duration/PCM")
                samples[label].append((observed, row))
                summary["comparison_order"].append({"round": index, "runtime": label, "name": row["name"]})
                save()
        summary["paired_aggregate"] = []
        for label, records in samples.items():
            summary["paired_aggregate"].append({
                "runtime": label, "samples": 5,
                "rtfs": [r[0]["rtf"] for r in records],
                "median_rtf": statistics.median(r[0]["rtf"] for r in records),
                "median_seconds": statistics.median(r[1]["median_seconds"] for r in records),
                "median_cpu_seconds": statistics.median(r[0]["cpu_seconds"] for r in records),
                "median_stages": {k: statistics.median(r[0]["stages"][k] for r in records)
                                  for k in records[0][0]["stages"]},
                "audio_seconds": records[0][0]["audio_seconds"],
                "generated_tokens": records[0][1]["generated_tokens"],
                "semantic": records[0][0]["semantic"],
                "pcm_sha256": records[0][0]["pcm_sha256"],
                "actual_ar_blocks": records[0][1]["actual_ar_blocks"],
                "median_ar_seconds_per_generated_token": statistics.median(
                    r[1]["median_ar_seconds_per_generated_token"] for r in records),
                "median_decoder_seconds_per_audio_second": statistics.median(
                    r[1]["median_decoder_seconds_per_audio_second"] for r in records),
                "peak_sampled_tree_rss_bytes": max(r[1]["peak_sampled_tree_rss_bytes"] for r in records),
            })
        summary["excluded_profiles"] = []
        for label in samples:
            _, row = run("ko", f"{label}_profile_run", distance=256,
                combined=True, ar_int8_only=True, profile=True,
                baseline_bundle=label == "AR18_V400",
                compare=args.output / f"ko_paired_4_{label}.wav")
            if not row["candidate_pcm_exact_to_n1_student"]:
                raise RuntimeError("Profile process changed approved same-model PCM")
            # Each model keeps a distinct operator profile file.
            (args.output / "ko_student_profile_ops.json").rename(
                args.output / f"ko_{label}_profile_ops.json")
            summary["excluded_profiles"].append(row)
        summary["student_assets_unchanged"] = (
            before == snapshot(args.manifest) and baseline_before == snapshot(args.baseline_manifest))
        summary["complete"] = True
        save()
        if not summary["student_assets_unchanged"]:
            raise RuntimeError("AR depth comparison changed private assets")
        return
    if args.alternating_decoder_comparison or runtime_comparison:
        candidate_label = (
            "head_v32"
            if args.head_v32_comparison
            else "tile4"
            if args.fft_output_tiles_comparison
            else "W"
            if args.int8_prefill_comparison
            else "packed"
            if args.packed_prefill_comparison
            else args.decoder_candidate_label
        )
        summary["scope"] = (
            f"Actual N1 approved L versus approved {candidate_label} pair bottlenecks only; counterbalanced AB/BA rounds; no new prefill/half-products"
        )
        summary["comparison_order"] = []
        if args.packed_prefill_comparison:
            summary["scope"] = (
                "Actual N1 Korean L versus source-pinned PCM-equivalent packed FP32 FFN prefill only, counterbalanced AB/BA; identical checkpoints and normal FP32 acoustics"
            )
        if args.int8_prefill_comparison:
            summary["scope"] = (
                "Actual N1 Korean L versus source-pinned PCM-equivalent W; genuine multirow W8A8 FFN prefill only, counterbalanced AB/BA; unchanged K weights/decode/FP32 acoustics"
            )
            summary["prefetch_selection"] = (
                "No tuning; fixed L decode/runtime, only prefill FFN arithmetic differs"
            )
        if args.fft_output_tiles_comparison:
            summary["scope"] = (
                "Actual N1 unchanged approved L versus exact-FP32 spectral output tiling4; "
                "counterbalanced fresh processes, no model/prefill/precision changes"
            )
            summary["prefetch_selection"] = "No tuning; unchanged approved L AR/runtime"
        if args.head_v32_comparison:
            summary["scope"] = (
                "Actual N1 unchanged approved model versus exact token-order head value "
                "scheduling; fresh counterbalanced processes, unchanged weights/AR arithmetic/FP32 acoustics"
            )
            summary["prefetch_selection"] = "No tuning; only dim32 weighted-value accumulation schedule changes"
            summary["baseline_model_variant"] = manifest["model_variant"]
            summary["runtime_label_note"] = "L denotes unchanged runtime here, not the earlier L decoder model"
        samples = {"L": [], candidate_label: []}
        last_candidate = None
        reference = None
        for index in range(5):
            for label in (
                ("L", candidate_label) if index % 2 == 0 else (candidate_label, "L")
            ):
                data, row = run(
                    "ko",
                    f"paired_{index}_{label}",
                    distance=256,
                    combined=True,
                    baseline_bundle=label == "L" and not runtime_comparison,
                    ar_int8_only=True,
                    measured_samples=1,
                    packed_prefill=label == "packed",
                    int8_prefill=label == "W",
                    fft_output_tiles=label == "tile4",
                    head_v32=label == "head_v32",
                )
                observed = data["runs"][0]
                if reference is None:
                    reference = observed
                if (
                    observed["semantic"] != reference["semantic"]
                    or observed["audio_seconds"] != reference["audio_seconds"]
                ):
                    raise RuntimeError(
                        "Bottleneck comparison changed identical-AR tokens/duration"
                    )
                samples[label].append((observed, row))
                if (
                    runtime_comparison
                    and observed["pcm_sha256"] != reference["pcm_sha256"]
                ):
                    raise RuntimeError(
                        "Runtime changed approved L PCM on actual N1; listening gate required"
                    )
                summary["comparison_order"].append(
                    {"round": index, "runtime": label, "name": row["name"]}
                )
                if label == candidate_label:
                    last_candidate = (
                        args.output / f"ko_paired_{index}_{candidate_label}.wav"
                    )
                save()
        summary["paired_aggregate"] = []
        for label, records in samples.items():
            summary["paired_aggregate"].append(
                {
                    "runtime": label,
                    "samples": 5,
                    "rtfs": [r[0]["rtf"] for r in records],
                    "median_rtf": statistics.median(r[0]["rtf"] for r in records),
                    "median_seconds": statistics.median(
                        r[1]["median_seconds"] for r in records
                    ),
                    "median_cpu_seconds": statistics.median(
                        r[0]["cpu_seconds"] for r in records
                    ),
                    "median_stages": {
                        key: statistics.median(r[0]["stages"][key] for r in records)
                        for key in records[0][0]["stages"]
                    },
                    "audio_seconds": reference["audio_seconds"],
                    "peak_sampled_tree_rss_bytes": max(
                        r[1]["peak_sampled_tree_rss_bytes"] for r in records
                    ),
                }
            )
        try:
            _, row = run(
                "ko",
                "head_v32_profile_run"
                if args.head_v32_comparison
                else "fft_output_tile4_profile_run"
                if args.fft_output_tiles_comparison
                else "int8_prefill_profile_run"
                if args.int8_prefill_comparison
                else "packed_prefill_profile_run"
                if args.packed_prefill_comparison
                else "bottleneck_profile_run",
                distance=256,
                combined=True,
                ar_int8_only=True,
                profile=True,
                compare=last_candidate,
                packed_prefill=args.packed_prefill_comparison,
                int8_prefill=args.int8_prefill_comparison,
                fft_output_tiles=args.fft_output_tiles_comparison,
                head_v32=args.head_v32_comparison,
            )
            if not row["candidate_pcm_exact_to_n1_student"]:
                raise RuntimeError("Profile process changed approved candidate PCM")
        except subprocess.CalledProcessError as error:
            summary["profile_error"] = {"exit_code": error.returncode}
        summary["student_assets_unchanged"] = before == snapshot(args.manifest) and (
            baseline is None or baseline_before == snapshot(args.baseline_manifest)
        )
        summary["complete"] = True
        save()
        if not summary["student_assets_unchanged"]:
            raise RuntimeError("Private bottleneck comparison assets changed")
        return
    if args.fft_half_comparison or args.fft_half_stream_comparison:
        summary["scope"] = (
            "Actual N1 L versus separately KO listening-approved P; identical K weights and W8A8/native FP32 AR; only large FFT spectral products approximate"
        )
        summary["prefetch_selection"] = (
            "No search; P disables FFT streaming and materializes FP32 products after FP16 spectral multiplication/accumulation"
        )
        if args.fft_half_stream_comparison:
            summary["scope"] = (
                "Actual N1 independent P baseline repeat versus locally PCM-identical half-stream; unchanged FP16 accumulation and approved Korean waveform"
            )
            summary["prefetch_selection"] = (
                "No retuning; same P arithmetic, four-frequency scratch instead of full-frequency products"
            )
        baseline_data, _ = run(
            "ko",
            "fft_product_baseline",
            distance=256,
            combined=True,
            confirm=True,
            ar_int8_only=True,
            fft_half_products=args.fft_half_stream_comparison,
        )
        candidate_data, _ = run(
            "ko",
            "fft_product_candidate",
            distance=256,
            combined=True,
            confirm=True,
            ar_int8_only=True,
            fft_half_products=True,
            fft_half_stream=args.fft_half_stream_comparison,
            compare=args.output / "ko_fft_product_baseline.wav"
            if args.fft_half_stream_comparison
            else None,
        )
        if not all(
            row["semantic"] == baseline_data["runs"][0]["semantic"]
            and row["audio_seconds"] == baseline_data["runs"][0]["audio_seconds"]
            for row in baseline_data["runs"] + candidate_data["runs"]
        ):
            raise RuntimeError(
                "FFT-product-only comparison changed semantic tokens/duration"
            )
        if args.fft_half_stream_comparison and not all(
            row["comparison"].get("exact") for row in candidate_data["runs"]
        ):
            raise RuntimeError("Half-stream changed approved P PCM on actual N1")
        try:
            run(
                "ko",
                "fft_product_profile_run",
                distance=256,
                combined=True,
                profile=True,
                ar_int8_only=True,
                fft_half_products=True,
                fft_half_stream=args.fft_half_stream_comparison,
                compare=args.output / "ko_fft_product_candidate.wav",
            )
        except subprocess.CalledProcessError as error:
            summary["profile_error"] = {"exit_code": error.returncode}
        summary["student_assets_unchanged"] = before == snapshot(args.manifest)
        summary["complete"] = True
        save()
        if not summary["student_assets_unchanged"]:
            raise RuntimeError("Private FFT-product runtime assets changed")
        return
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
            run(
                "ko",
                "ar_runtime_baseline",
                distance=256,
                combined=True,
                confirm=True,
            )
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
            ar_int8_only=args.decoder_ar_int8,
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
            ar_int8_only=args.decoder_ar_int8,
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
                ar_int8_only=args.decoder_ar_int8,
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
        _, row = run(
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
