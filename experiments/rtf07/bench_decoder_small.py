"""Measure FP32 decoder kernels at small-channel target geometries.

Synthetic tensors avoid distributing reference audio and learned weights. This
is a kernel selection probe, not a synthesis RTF or a speech-quality test.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

import cpufast_cpp_ops  # noqa: F401; registers native FFT operators
from cpufast_fft import FFTConv1d
from cpufast_tap import TapConv1d
from cpufast_optimizations import PackedConv1d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.warmups <= 5
        or not 1 <= args.samples <= 10
    ):
        parser.error("Use a new output, 1–5 warmups and 1–10 samples")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    cases = [
        (24, 181760, 7, 1),
        (24, 181760, 11, 5),
        (48, 90880, 7, 1),
        (48, 90880, 11, 5),
    ]
    report = {
        "scope": "Synthetic decoder convolution probe; not end-to-end inference or RTF",
        "machine": platform.machine(),
        "torch": torch.__version__,
        "threads": args.threads,
        "cases": [],
    }
    with torch.no_grad():
        for channels, length, kernel, dilation in cases:
            conv = torch.nn.Conv1d(
                channels,
                channels,
                kernel,
                dilation=dilation,
                padding=(kernel - 1) * dilation // 2,
            ).eval()
            x = torch.randn(1, channels, length)
            reference = conv(x)
            for method in ("fft", "tap", "packed"):
                fft = FFTConv1d(conv, backend="packed").eval()
                fft.output_layout = fft.fused_finish = True
                module = fft if method == "fft" else (TapConv1d(fft).eval() if method == "tap" else PackedConv1d(conv).eval())
                times, cpu_times = [], []
                for _ in range(args.warmups + args.samples):
                    start, cpu_start = time.perf_counter(), time.process_time()
                    output = module(x)
                    times.append(time.perf_counter() - start)
                    cpu_times.append(time.process_time() - cpu_start)
                error = (output - reference).double()
                snr = float(
                    10
                    * torch.log10(
                        reference.double().square().mean() / error.square().mean()
                    )
                )
                row = {
                    "channels": channels,
                    "length": length,
                    "kernel": kernel,
                    "dilation": dilation,
                    "method": method,
                    "warmup_seconds": times[: args.warmups],
                    "seconds": times[args.warmups :],
                    "cpu_seconds": cpu_times[args.warmups :],
                    "median_seconds": statistics.median(times[args.warmups :]),
                    "max_abs_error": float(error.abs().max()),
                    "snr_db": snr if snr != float("inf") else None,
                }
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
