"""Compare FP32 overlap-save FFT sizes at target decoder geometries."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import cpufast_cpp_ops  # noqa: F401
from cpufast_fft import FFTConv1d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.warmups <= 5
        or not 1 <= args.samples <= 10
    ):
        parser.error("Use a new output, 1–5 warmups and 1–10 samples")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    report = {
        "scope": "Synthetic FP32 convolution FFT-size probe, not full synthesis",
        "cases": [],
    }
    with torch.no_grad():
        for channels, length in ((384, 11360), (192, 22720), (96, 45440)):
            for kernel, dilation in ((7, 1), (11, 5)):
                conv = torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel,
                    padding=(kernel - 1) * dilation // 2,
                    dilation=dilation,
                ).eval()
                x = torch.randn(1, channels, length)
                reference = conv(x)
                for size in (64, 128, 256, 512):
                    if size < (kernel - 1) * dilation + 1:
                        continue
                    module = FFTConv1d(conv, n_fft=size, backend="packed").eval()
                    module.output_layout = module.fused_finish = True
                    times = []
                    for _ in range(args.warmups + args.samples):
                        start = time.perf_counter()
                        output = module(x)
                        times.append(time.perf_counter() - start)
                    delta = (output - reference).double()
                    err = delta.square().mean()
                    snr = float(
                        10 * torch.log10(reference.double().square().mean() / err)
                    )
                    row = {
                        "channels": channels,
                        "length": length,
                        "kernel": kernel,
                        "dilation": dilation,
                        "n_fft": size,
                        "warmup_seconds": times[: args.warmups],
                        "seconds": times[args.warmups :],
                        "median_seconds": statistics.median(times[args.warmups :]),
                        "max_abs_error": float(delta.abs().max()),
                        "snr_db": snr if math.isfinite(snr) else None,
                    }
                    report["cases"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                    del module, output


if __name__ == "__main__":
    main()
