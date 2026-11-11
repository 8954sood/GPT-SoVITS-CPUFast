"""Measure CPUFast inference using unchanged checkpoints and a fixed cgroup quota.

Run inside Linux ARM64 Docker, through resource_guard.py. Set PYTHONPATH to
the CPUFast checkout, its GPT_SoVITS directory, and optional extra dependencies.
"""

import argparse
import functools
import hashlib
import json
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
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", choices=["ja", "ko", "en"], default="ja")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--ar-threads", type=int, choices=[1, 2], default=1)
    parser.add_argument("--threads", type=int, choices=[1, 2], default=2)
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
    parser.add_argument("--engine", action="store_true", help="Use the optimized engine factory")
    parser.add_argument("--native-predictor", action="store_true")
    parser.add_argument("--inference-mode", action="store_true")
    parser.add_argument("--fused-fft", action="store_true")
    parser.add_argument(
        "--ar-layout",
        choices=["original", "row", "lossless", "cpp"],
        default="original",
    )
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.samples <= 20
        or not 1 <= args.warmups <= 5
    ):
        parser.error("Use a fresh output, 1–20 samples and 1–5 warmups")
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
    load_seconds = time.perf_counter() - start
    from cpufast_integrity import ar_digest

    ar_before = ar_digest(pipeline.t2s_model.model.t2s_transformer)
    ar_module_before = parameter_digest(pipeline.t2s_model.model)
    if args.engine:
        from cpufast_engine import optimize

        start_install = time.perf_counter()
        runtime = optimize(pipeline)
        ar_install_seconds = 0.0
        packing_seconds = time.perf_counter() - start_install
        ar_audit = runtime.audit["ar_coefficients"]
        replaced = runtime.audit["phase_and_layout_replacements"]
        fft_replaced = runtime.audit["fft_convolutions"]
        winograd_replaced = runtime.audit["winograd_convolutions"]
        parameters_before = runtime.audit["before"]["vits_parameters"]
        args.ar_threads = args.threads = 1
        args.backend = "phase_fft_winograd"
        args.ar_layout = "cpp"
        args.native_predictor = args.fused_fft = True
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
        "prompt_lang": args.lang,
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
        "scope": "fixed CPU-quota throughput proxy; not a real N1 measurement",
        "cgroup": cgroup(),
        "inputs": inputs,
        "backend": args.backend,
        "ar_threads": args.ar_threads,
        "ar_layout": args.ar_layout,
        "ar_coefficient_audit": ar_audit,
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
        if index == args.warmups:
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
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
