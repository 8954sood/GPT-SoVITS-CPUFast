"""Measure CPUFast inference using unchanged checkpoints and a fixed cgroup quota.

Run inside Linux ARM64 Docker, through resource_guard.py. Set PYTHONPATH to
the CPUFast checkout, its GPT_SoVITS directory, and optional extra dependencies.
"""

import argparse
import functools
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from cpufast_optimizations import (
    CPUFastRuntime,
    install_convolutions,
    install_subpixel,
    parameter_digest,
)


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def cgroup():
    root = Path("/sys/fs/cgroup")
    return {
        key: (root / key).read_text().strip()
        for key in (
            "cpu.max",
            "cpuset.cpus.effective",
            "memory.max",
            "memory.swap.max",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument("--hubert", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--prompt-lang", choices=["ja", "ko", "en"])
    parser.add_argument("--student-manifest", type=Path)
    parser.add_argument("--student-manifest-sha256")
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", choices=["ja", "ko", "en"], default="ja")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--ar-threads", type=int, choices=[1, 2], default=1)
    parser.add_argument("--threads", type=int, choices=[1, 2])
    parser.add_argument(
        "--backend",
        choices=[
            "original",
            "channels_last",
            "xnnpack",
            "phase",
            "phase_channels_last",
            "phase_fft",
            "phase_fft_tuned",
            "phase_fft_winograd",
        ],
        default="original",
    )
    parser.add_argument("--compare-pcm", type=Path)
    parser.add_argument("--packed-fft", action="store_true")
    parser.add_argument("--fft-stream", action="store_true")
    parser.add_argument("--fft-pad-channels", action="store_true")
    parser.add_argument("--parallel-transforms", action="store_true")
    parser.add_argument("--fused-finish", action="store_true")
    parser.add_argument("--tap-small", action="store_true")
    parser.add_argument("--packed-small-48", action="store_true")
    parser.add_argument("--n1-fft-sizes", action="store_true")
    parser.add_argument("--ar-fp32-cache", action="store_true")
    parser.add_argument("--n1-small-direct", action="store_true")
    parser.add_argument(
        "--ar-prefetch", type=int, choices=(0, 64, 128, 256), default=None
    )
    parser.add_argument("--fast-experimental", action="store_true")
    parser.add_argument("--no-save-audio", action="store_true")
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Optional fresh JSON for a separate extra profiled inference (excluded from RTF)",
    )
    parser.add_argument("--profile-no-shapes", action="store_true")
    parser.add_argument("--head-major", action="store_true")
    parser.add_argument("--cook-phase", action="store_true")
    parser.add_argument("--cook-resblock", type=int, choices=[0, 6, 8], default=0)
    parser.add_argument("--fft-layout", action="store_true")
    parser.add_argument("--native-attention", action="store_true")
    parser.add_argument(
        "--engine", action="store_true", help="Use the optimized engine factory"
    )
    parser.add_argument("--native-predictor", action="store_true")
    parser.add_argument("--student-fp32-control", action="store_true")
    parser.add_argument("--ar-int8-only", action="store_true")
    parser.add_argument("--fft-half-products", action="store_true")
    parser.add_argument("--inference-mode", action="store_true")
    parser.add_argument("--fused-fft", action="store_true")
    parser.add_argument(
        "--ar-layout",
        choices=["original", "row", "lossless", "cpp"],
        default="original",
    )
    args = parser.parse_args()
    if args.prompt_lang is None:
        args.prompt_lang = args.lang
    if bool(args.student_manifest) != bool(args.student_manifest_sha256):
        parser.error("Student manifest and its trusted SHA256 are required together")
    if args.student_fp32_control and not args.student_manifest:
        parser.error("--student-fp32-control requires an approved exact-half bundle")
    if args.ar_int8_only and (
        not args.engine
        or not args.student_manifest
        or args.fast_experimental
        or args.ar_fp32_cache
    ):
        parser.error(
            "AR-only INT8 requires a separately approved half-exact student and normal FP32 acoustic engine"
        )
    if args.student_manifest and (not args.engine or args.fast_experimental):
        parser.error("Approved student requires the FP32 arithmetic engine")
    if args.fft_half_products and (
        not args.ar_int8_only
        or not args.packed_fft
        or not args.fft_layout
        or args.lang != "ko"
    ):
        parser.error(
            "FFT half-products requires the separately approved Korean L-derived packed FFT runtime"
        )
    student_manifest = None
    student_files_before = None
    ar_runtime_approval = None
    ar_runtime_listening_case = None
    fft_half_runtime_approval = None
    if args.student_manifest:
        from cpufast_student_assets import load_manifest, snapshot

        student_manifest = load_manifest(
            args.student_manifest, args.student_manifest_sha256
        )
        if args.student_fp32_control and (
            student_manifest.get("ar_cache_storage", "fp32") != "exact_fp16"
            or not args.ar_fp32_cache
        ):
            parser.error(
                "FP32 control must use an approved exact-half bundle with FP32 cache"
            )
        if not args.student_fp32_control and args.ar_fp32_cache != (
            student_manifest.get("ar_cache_storage", "fp32") == "fp32"
        ):
            parser.error("AR cache storage must match the approved student manifest")
        if args.s1.resolve() != (args.student_manifest.parent / "ar.pt").resolve():
            parser.error("AR checkpoint must come from the approved student bundle")
        if (
            file_digest(args.s2) != student_manifest["original_s2_sha256"]
            or file_digest(args.reference) != student_manifest["reference_sha256"]
            or args.prompt_text != student_manifest["prompt_text"]
            or args.prompt_lang != student_manifest["prompt_language"]
            or args.text
            not in [
                r["text"]
                for r in student_manifest["listening_report"]["cases"]
                if r["language"] == args.lang
            ]
        ):
            parser.error(
                "Student teacher/reference/request differs from approved manifest"
            )
        student_files_before = snapshot(args.student_manifest)
        if args.ar_int8_only:
            root = Path(__file__).resolve().parent
            if student_manifest.get("ar_runtime") == "int8_native":
                from cpufast_student_assets import inline_runtime_approval

                if args.lang != "ko":
                    parser.error(
                        "Whole-bundle runtime approval currently covers Korean only"
                    )
                ar_runtime_approval = inline_runtime_approval(
                    student_manifest, args.student_manifest_sha256, root
                )
                runtime_cases = student_manifest["listening_report"]["cases"]
            else:
                ar_runtime_approval = json.loads(
                    (root / "results/student_pilot/approved_l_runtime.json").read_text()
                )
                if (
                    ar_runtime_approval.get("quality_approved") is not True
                    or ar_runtime_approval["underlying_manifest_sha256"]
                    != args.student_manifest_sha256
                    or ar_runtime_approval["reference_sha256"]
                    != student_manifest["reference_sha256"]
                    or ar_runtime_approval["implementation_sha256"]
                    != file_digest(root / ar_runtime_approval["implementation"])
                ):
                    parser.error(
                        "AR-only runtime does not match its separate listening approval"
                    )
                runtime_cases = json.loads(
                    (root / ar_runtime_approval["listening_report"]).read_text()
                )["comparison"]["cases"]
            ar_runtime_listening_case = next(
                row for row in runtime_cases if row["language"] == args.lang
            )
        elif student_manifest.get("ar_runtime") == "int8_native":
            parser.error("Use the declared approved INT8 runtime for this model bundle")
        if args.fft_half_products:
            root = Path(__file__).resolve().parent
            fft_half_runtime_approval = json.loads(
                (root / "results/student_pilot/approved_p_runtime.json").read_text()
            )
            approval = fft_half_runtime_approval
            if (
                approval.get("quality_approved") is not True
                or approval.get("model_variant") != "P"
                or approval.get("seed") != 42
                or approval.get("underlying_manifest_sha256")
                != args.student_manifest_sha256
                or approval.get("reference_sha256")
                != student_manifest["reference_sha256"]
                or approval.get("implementation")
                != "experiments/student/listen_fft_half.py"
                or approval.get("implementation_sha256")
                != file_digest(root / approval["implementation"])
            ):
                parser.error(
                    "FFT-product runtime does not match its separate listening approval"
                )
            runtime_cases = json.loads(
                (root / approval["listening_report"]).read_text()
            )["comparison"]["cases"]
            ar_runtime_listening_case = next(
                row for row in runtime_cases if row["language"] == "ko"
            )
            if ar_runtime_listening_case["text"] != args.text:
                parser.error(
                    "FFT-product runtime text differs from its listening approval"
                )
    if args.threads is None:
        args.threads = 1 if args.engine else 2
    if args.ar_prefetch is None:
        args.ar_prefetch = (
            256
            if args.engine and not args.fast_experimental and not args.ar_fp32_cache
            else 0
        )
    if args.ar_prefetch and not args.engine:
        parser.error("--ar-prefetch requires --engine")
    if args.ar_fp32_cache and not args.engine:
        parser.error("--ar-fp32-cache requires --engine")
    if args.n1_small_direct and not args.engine:
        parser.error("--n1-small-direct requires --engine")
    if args.n1_fft_sizes and not args.engine:
        parser.error("--n1-fft-sizes requires --engine")
    if args.packed_small_48 and not args.tap_small:
        parser.error("--packed-small-48 requires --tap-small")
    if args.tap_small and not args.engine:
        parser.error("--tap-small requires --engine")
    if args.fast_experimental and not args.engine:
        parser.error("--fast-experimental requires --engine")
    if (
        args.output.exists()
        or (args.profile_output is not None and args.profile_output.exists())
        or not 1 <= args.samples <= 20
        or not 1 <= args.warmups <= 5
    ):
        parser.error("Use a fresh output, 1–20 samples and 1–5 warmups")
    if (
        args.profile_output is not None
        and args.profile_output.resolve() == args.output.resolve()
    ):
        parser.error("Profile and benchmark outputs must be different files")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from TTS_infer_pack.TTS import TTS, TTS_Config

    class NonChineseTTS(TTS):
        def init_bert_weights(self, base_path):
            # CPUFast uses zero BERT features for the three supported languages.
            pass

    weights_before = {str(p): file_digest(p) for p in (args.s1, args.s2)}
    start = time.perf_counter()
    pipeline = NonChineseTTS(
        TTS_Config(
            {
                "custom": {
                    "device": "cpu",
                    "is_half": False,
                    "version": "v2ProPlus",
                    "t2s_weights_path": str(args.s1),
                    "vits_weights_path": str(args.s2),
                    "cnhuhbert_base_path": str(args.hubert),
                    "bert_base_path": ".",
                }
            }
        )
    )
    student_load_start = time.perf_counter()
    if student_manifest is not None:
        from student_runtime import StudentGenerator

        spec = json.loads(
            (args.student_manifest.parent / "student_spec.json").read_text()
        )
        trained = torch.load(
            args.student_manifest.parent / "decoder.pt",
            map_location="cpu",
            weights_only=True,
        )
        if (
            spec["decoder_branches"] != student_manifest["decoder_branches"]
            or spec.get("decoder_channel_keep")
            != student_manifest.get("decoder_channel_keep")
            or trained["student_spec"].get("decoder_channel_keep")
            != spec.get("decoder_channel_keep")
            or len(spec["ar_keep"]) != student_manifest["ar_layers"]
            or len(set(spec["ar_keep"])) != len(spec["ar_keep"])
            or any(type(i) is not int or not 0 <= i < 24 for i in spec["ar_keep"])
            or trained["student_spec"]["decoder_branches"] != spec["decoder_branches"]
            or trained["provenance"]["teacher_s2_sha256"]
            != student_manifest["original_s2_sha256"]
            or len(pipeline.t2s_model.model.t2s_transformer.blocks)
            != student_manifest["ar_layers"]
        ):
            raise ValueError("Approved student architecture/teacher mismatch")
        channel_options = (
            {"channel_keep": spec["decoder_channel_keep"]}
            if spec.get("decoder_channel_keep") is not None
            else {}
        )
        pipeline.vits_model.dec = StudentGenerator(
            pipeline.vits_model.dec, spec["decoder_branches"], **channel_options
        ).eval()
        pipeline.vits_model.dec.load_state_dict(trained["state_dict"], strict=True)
        del trained
    student_load_seconds = time.perf_counter() - student_load_start
    load_seconds = time.perf_counter() - start
    from cpufast_integrity import ar_digest

    ar_before = ar_digest(pipeline.t2s_model.model.t2s_transformer)
    ar_module_before = parameter_digest(pipeline.t2s_model.model)
    if args.engine:
        from cpufast_engine import optimize

        original_ar = pipeline.t2s_model.model.t2s_transformer
        start_install = time.perf_counter()
        runtime = optimize(
            pipeline,
            native_attention=args.native_attention,
            head_major=args.head_major,
            cook_phase=args.cook_phase,
            cook_resblock=args.cook_resblock,
            fft_layout=args.fft_layout,
            packed_fft=args.packed_fft,
            fft_stream=args.fft_stream,
            fft_pad_channels=args.fft_pad_channels,
            parallel_transforms=args.parallel_transforms,
            fused_finish=args.fused_finish,
            tap_small=args.tap_small,
            packed_small_48=args.packed_small_48,
            n1_fft_sizes=args.n1_fft_sizes,
            n1_small_direct=args.n1_small_direct,
            ar_fp32_cache=args.ar_fp32_cache,
            ar_prefetch=args.ar_prefetch,
            ar_threads=args.ar_threads,
            acoustic_threads=args.threads,
            fast_experimental=args.fast_experimental,
        )
        if args.ar_int8_only:
            import cpufast_fast_ops
            from cpufast_fast_ar import build as build_int8_ar

            approximate_ar, details = build_int8_ar(
                original_ar,
                head_major=True,
            )
            pipeline.t2s_model.model.t2s_transformer = approximate_ar
            half_acoustics = sum(
                bool(getattr(module, "fast_half", False))
                for module in pipeline.vits_model.modules()
            )
            if half_acoustics or runtime.audit["fast_experimental"]:
                raise RuntimeError("AR-only approximate runtime changed FP32 acoustics")
            runtime.audit.update(
                ar_coefficients=details,
                arithmetic="Approximate W8A8 AR/native FP32 attention; unchanged normal FP32 acoustic runtime",
                fp16_acoustic_modules=half_acoustics,
            )
        if args.fft_half_products:
            from cpufast_fft_half_runtime import enable_fft_half_products

            product_modules = enable_fft_half_products(pipeline.vits_model.dec)
            half_acoustics = sum(
                bool(getattr(module, "fast_half", False))
                for module in pipeline.vits_model.modules()
            )
            if half_acoustics != len(product_modules):
                raise RuntimeError(
                    "FFT-product candidate changed unrelated acoustic modules"
                )
            runtime.audit.update(
                fft_half_product_modules=product_modules,
                fp16_acoustic_modules=half_acoustics,
                arithmetic="Approximate W8A8 AR/native FP32 attention; large FFT spectral inputs/derived weights and accumulation FP16, materialized FP32 products; transforms/Tap/Winograd FP32",
            )
        ar_install_seconds = 0.0
        packing_seconds = time.perf_counter() - start_install
        ar_audit = runtime.audit["ar_coefficients"]
        replaced = runtime.audit["phase_and_layout_replacements"]
        fft_replaced = runtime.audit["fft_convolutions"]
        winograd_replaced = runtime.audit["winograd_convolutions"]
        parameters_before = runtime.audit["before"]["vits_parameters"]
        args.backend = "phase_fft_winograd"
        args.ar_layout = "cpp"
        args.native_predictor = args.fused_fft = True
        if args.fast_experimental:
            args.head_major = args.cook_phase = args.fft_layout = True
            args.packed_fft = args.fused_finish = True
    else:
        ar_audit = None
        ar_install_start = time.perf_counter()
        if args.ar_layout == "row":
            from cpufast_ar_layout import build

            pipeline.t2s_model.model.t2s_transformer = build(
                pipeline.t2s_model.model.t2s_transformer
            )
        elif args.ar_layout == "lossless":
            from cpufast_ar_lossless import build

            pipeline.t2s_model.model.t2s_transformer, ar_audit = build(
                pipeline.t2s_model.model.t2s_transformer
            )
        elif args.ar_layout == "cpp":
            from cpufast_ar_cpp import build

            pipeline.t2s_model.model.t2s_transformer, ar_audit = build(
                pipeline.t2s_model.model.t2s_transformer
            )
        ar_install_seconds = time.perf_counter() - ar_install_start
        if args.native_predictor:
            if args.ar_layout != "cpp":
                parser.error("Native predictor requires the CPP AR operator")
            from cpufast_predictor import NativePredictor

            pipeline.t2s_model.model.ar_predict_layer = NativePredictor(
                pipeline.t2s_model.model.ar_predict_layer
            )
        parameters_before = parameter_digest(pipeline.vits_model)
        packing_start = time.perf_counter()
        fft_replaced = 0
        winograd_replaced = 0
        if args.backend == "original":
            replaced = 0
        elif args.backend in (
            "phase",
            "phase_channels_last",
            "phase_fft",
            "phase_fft_tuned",
            "phase_fft_winograd",
        ):
            replaced = install_subpixel(
                pipeline.vits_model.dec,
                pack_convolutions=args.backend != "phase",
            )
            if args.backend in ("phase_fft", "phase_fft_tuned", "phase_fft_winograd"):
                from cpufast_fft import install_fft

                fft_replaced = install_fft(
                    pipeline.vits_model.dec.resblocks,
                    tuned=args.backend == "phase_fft_tuned",
                    backend="fused" if args.fused_fft else "torch",
                )
            if args.backend == "phase_fft_winograd":
                if args.ar_layout != "cpp":
                    parser.error("Native Winograd requires the CPP runtime library")
                from cpufast_winograd import install_winograd

                winograd_replaced = install_winograd(
                    pipeline.vits_model.dec.resblocks, native=True
                )
        else:
            replaced = install_convolutions(pipeline.vits_model.dec, args.backend)
        assert parameter_digest(pipeline.vits_model) == parameters_before
        packing_seconds = time.perf_counter() - packing_start
        runtime = CPUFastRuntime(
            pipeline, ar_threads=args.ar_threads, acoustic_threads=args.threads
        )
    state = {"stages": {}, "semantic": []}
    decoder = pipeline.t2s_model.model
    original = decoder.infer_panel_batch_infer

    @functools.wraps(original)
    def infer(*a, **kw):
        begin = time.perf_counter()
        try:
            result = original(*a, **kw)
        finally:
            state["stages"]["ar_seconds"] = time.perf_counter() - begin
        tokens, indices = result
        state["semantic"] = [
            {
                "sha256": hashlib.sha256(
                    t.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "index": int(i),
                "generated_sha256": hashlib.sha256(
                    t[-int(i) :].detach().cpu().numpy().tobytes()
                ).hexdigest(),
            }
            for t, i in zip(tokens, indices)
        ]
        return result

    decoder.infer_panel_batch_infer = infer
    for name in ("enc_p", "flow", "dec"):
        module = getattr(pipeline.vits_model, name)

        def pre(module, inputs, name=name):
            state[name + "_start"] = time.perf_counter()

        def post(module, inputs, output, name=name):
            key = name + "_seconds"
            state["stages"][key] = (
                state["stages"].get(key, 0)
                + time.perf_counter()
                - state[name + "_start"]
            )

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)
    inputs = {
        "text": args.text,
        "text_lang": args.lang,
        "ref_audio_path": str(args.reference),
        "prompt_text": args.prompt_text,
        "prompt_lang": args.prompt_lang,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut0",
        "batch_size": 1,
        "seed": 42,
        "parallel_infer": True,
        "vits_parallel_infer": False,
        "fragment_interval": 0.3,
    }
    report = {
        "model_variant": student_manifest.get("model_variant", "approved_G_student")
        if student_manifest
        else "original",
        "student_manifest_sha256": args.student_manifest_sha256,
        "student_quality_scope": student_manifest["scope"]
        if student_manifest
        else None,
        "student_files_before": student_files_before,
        "student_fp32_control": args.student_fp32_control,
        "approximate_ar_int8_only": args.ar_int8_only,
        "ar_runtime_approval": ar_runtime_approval,
        "approximate_fft_half_products": args.fft_half_products,
        "fft_half_runtime_approval": fft_half_runtime_approval,
        "fft_half_product_modules": runtime.audit.get("fft_half_product_modules", [])
        if args.engine
        else [],
        "ar_runtime_library_manifest": json.loads(
            cpufast_fast_ops.path.with_suffix(
                cpufast_fast_ops.path.suffix + ".json"
            ).read_text()
        )
        if args.ar_int8_only
        else None,
        "fp16_acoustic_modules": runtime.audit.get("fp16_acoustic_modules", 0)
        if args.engine
        else None,
        "fft_padded_output_cache": runtime.audit.get("fft_padded_output_cache", [])
        if args.engine
        else [],
        "student_load_seconds": student_load_seconds,
        "experimental_options": {
            k: getattr(args, k)
            for k in (
                "head_major",
                "cook_phase",
                "cook_resblock",
                "fft_layout",
                "native_attention",
                "packed_fft",
                "fft_stream",
                "fft_pad_channels",
                "fft_half_products",
                "parallel_transforms",
                "fused_finish",
                "tap_small",
                "packed_small_48",
                "n1_fft_sizes",
                "n1_small_direct",
                "ar_fp32_cache",
                "ar_prefetch",
                "fast_experimental",
            )
        },
        "scope": "nonstreaming synthesis on the executing host; see cgroup and hardware",
        "cgroup": cgroup(),
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
        "inputs": inputs,
        "backend": args.backend,
        "ar_threads": args.ar_threads,
        "native_threading": {
            "configured_ar_threads": args.ar_threads,
            "configured_acoustic_threads": args.threads,
            "torch_interop_threads": torch.get_num_interop_threads(),
            "native_operator_pool": "ATen parallel_for uses the global Torch intra-op pool; legacy native_workers=1 audit field is not the actual thread cap",
        },
        "ar_layout": "cpp_int8_ar_only"
        if args.ar_int8_only
        else "cpp_int8_experimental"
        if args.fast_experimental
        else args.ar_layout,
        "ar_coefficient_audit": ar_audit,
        "n1_small_direct_count": runtime.audit.get("n1_small_direct_count", 0)
        if args.engine
        else 0,
        "n1_fft_size_count": runtime.audit.get("n1_fft_size_count", 0)
        if args.engine
        else 0,
        "packed_small_48_count": runtime.audit.get("packed_small_48_count", 0)
        if args.engine
        else 0,
        "native_predictor": args.native_predictor,
        "inference_mode": args.inference_mode,
        "fused_fft": args.fused_fft,
        "acoustic_threads": args.threads,
        "load_seconds": load_seconds,
        "installation_seconds": packing_seconds,
        "ar_installation_seconds": ar_install_seconds,
        "convolutions_replaced": replaced,
        "fft_convolutions": fft_replaced,
        "winograd_convolutions": winograd_replaced,
        "reference_sha256": file_digest(args.reference),
        "checkpoints_before": weights_before,
        "vits_parameters_before": parameters_before,
        "ar_parameters_before": ar_before,
        "ar_module_before": ar_module_before,
        "warmups": [],
        "runs": [],
        "frontend_note": "Chinese BERT not loaded; original full HuBERT used",
    }
    reference_pcm = reference_rate = None
    if args.compare_pcm:
        reference_pcm, reference_rate = sf.read(args.compare_pcm, dtype="int16")
    for index in range(args.warmups + args.samples):
        state["stages"] = {}
        start = time.perf_counter()
        cpu_start = time.process_time()
        if args.inference_mode:
            with torch.inference_mode():
                chunks = list(runtime.run(inputs))
        else:
            chunks = list(runtime.run(inputs))
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - cpu_start
        if len(chunks) != 1:
            raise RuntimeError("Expected one complete, nonstreaming audio result")
        rate, audio = chunks[0]
        assert np.isfinite(audio).all() and audio.dtype == np.int16
        row = {
            "seconds": elapsed,
            "cpu_seconds": cpu_elapsed,
            "rtf": elapsed / (len(audio) / rate),
            "audio_seconds": len(audio) / rate,
            "sample_rate": rate,
            "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            "stages": state["stages"].copy(),
            "semantic": state["semantic"],
        }
        if ar_runtime_listening_case is not None:
            row["runtime_local_pcm_match"] = (
                row["pcm_sha256"] == ar_runtime_listening_case["pcm_sha256"]
            )
            row["runtime_local_generated_tokens_match"] = [
                {"tokens": item["index"], "sha256": item["generated_sha256"]}
                for item in row["semantic"]
            ] == ar_runtime_listening_case["semantic"]
        if reference_pcm is not None:
            row["comparison"] = {
                "same_shape": audio.shape == reference_pcm.shape,
                "same_rate": rate == reference_rate,
            }
            if audio.shape == reference_pcm.shape and rate == reference_rate:
                delta = audio.astype(np.float64) - reference_pcm.astype(np.float64)
                mse = np.mean(delta**2)
                row["comparison"].update(
                    {
                        "max_pcm_delta": float(np.max(np.abs(delta))),
                        "changed_samples": int(np.count_nonzero(delta)),
                        "snr_db": None
                        if mse == 0
                        else float(
                            10
                            * np.log10(
                                np.mean(reference_pcm.astype(np.float64) ** 2) / mse
                            )
                        ),
                        "exact": bool(mse == 0),
                    }
                )
        is_warmup = index < args.warmups
        report["warmups" if is_warmup else "runs"].append(row)
        if index == args.warmups and not args.no_save_audio:
            sf.write(args.output.with_suffix(".wav"), audio, rate, subtype="PCM_16")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("CPUFAST_RESULT", json.dumps(row), flush=True)
    report["median_rtf"] = statistics.median(r["rtf"] for r in report["runs"])
    report["median_seconds"] = statistics.median(r["seconds"] for r in report["runs"])
    report["checkpoints_after"] = {str(p): file_digest(p) for p in (args.s1, args.s2)}
    report["vits_parameters_after"] = parameter_digest(pipeline.vits_model)
    report["checkpoint_files_unchanged"] = (
        report["checkpoints_before"] == report["checkpoints_after"]
    )
    report["parameters_unchanged"] = (
        parameters_before == report["vits_parameters_after"]
    )
    report["ar_parameters_after"] = ar_digest(pipeline.t2s_model.model.t2s_transformer)
    report["ar_module_after"] = parameter_digest(pipeline.t2s_model.model)
    report["ar_parameters_unchanged"] = (
        report["ar_parameters_after"] == ar_before
        and report["ar_module_after"] == ar_module_before
    )
    assert report["checkpoint_files_unchanged"] and report["parameters_unchanged"]
    assert report["ar_parameters_unchanged"]
    if student_manifest is not None:
        report["student_files_after"] = snapshot(args.student_manifest)
        report["student_files_unchanged"] = (
            report["student_files_after"] == student_files_before
        )
        assert report["student_files_unchanged"]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.profile_output is not None:
        from cpufast_profiling import predictor_range, decoder_tap_ranges

        # The additional profiled synthesis is explicitly excluded from the RTF runs.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=not args.profile_no_shapes,
        ) as prof:
            # Profile the same execution mode as the measured inference.
            with (
                predictor_range(
                    pipeline.t2s_model.model.ar_predict_layer
                ) as predictor_metadata,
                decoder_tap_ranges(pipeline.vits_model.dec) as tap_metadata,
                torch.inference_mode(args.inference_mode),
            ):
                profile_chunks = list(runtime.run(inputs))
        if len(profile_chunks) != 1:
            raise RuntimeError("Profile did not return one complete audio result")
        profile_rate, profile_audio = profile_chunks[0]
        profile_checks = {
            "sample_rate": profile_rate,
            "pcm_sha256": hashlib.sha256(profile_audio.tobytes()).hexdigest(),
            "semantic": state["semantic"],
            "audio_seconds": len(profile_audio) / profile_rate,
        }
        profile_checks["exact_to_unprofiled_same_process"] = all(
            profile_checks[key] == report["runs"][-1][key]
            for key in ("sample_rate", "pcm_sha256", "semantic", "audio_seconds")
        )
        if not profile_checks["exact_to_unprofiled_same_process"]:
            raise RuntimeError("Profiling instrumentation changed PCM/tokens/duration")
        rows = [
            {
                "name": event.key,
                "calls": event.count,
                "input_shapes": event.input_shapes,
                "self_cpu_us": event.self_cpu_time_total,
                "total_cpu_us": event.cpu_time_total,
            }
            for event in prof.key_averages(
                group_by_input_shape=not args.profile_no_shapes
            )
        ]
        args.profile_output.write_text(
            json.dumps(
                {
                    "note": "Extra profiled inference, excluded from RTF; CPU totals may overlap across threads and include profiler overhead",
                    "record_shapes": not args.profile_no_shapes,
                    "predictor": predictor_metadata,
                    "decoder_taps": tap_metadata,
                    "profile_output_checks": profile_checks,
                    "operators": sorted(
                        rows, key=lambda row: row["self_cpu_us"], reverse=True
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        report["profile_output"] = str(args.profile_output)
        if (
            student_manifest is not None
            and snapshot(args.student_manifest) != student_files_before
        ):
            raise RuntimeError("Student execution assets changed during profiling")
        if (
            parameter_digest(pipeline.vits_model) != parameters_before
            or ar_digest(pipeline.t2s_model.model.t2s_transformer) != ar_before
            or parameter_digest(pipeline.t2s_model.model) != ar_module_before
        ):
            raise RuntimeError("Model parameters changed during profiling")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
