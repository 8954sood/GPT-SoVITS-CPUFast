"""Compare FP32 three-tap decoder kernels at target tensor geometries."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import cpufast_cpp_ops  # noqa: F401
from cpufast_optimizations import PackedConv1d
from cpufast_winograd import WinogradConv1d


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
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report = {
        "scope": "Synthetic FP32 three-tap convolution probe, not full synthesis",
        "cases": [],
    }
    with torch.no_grad():
        for channels, length in (
            (384, 2840),
            (192, 22720),
            (96, 45440),
            (48, 90880),
            (24, 181760),
        ):
            for dilation in (1, 3, 5):
                conv = torch.nn.Conv1d(
                    channels, channels, 3, padding=dilation, dilation=dilation
                ).eval()
                x = torch.randn(1, channels, length)
                reference = conv(x)
                for method in ("winograd", "packed"):
                    module = (
                        WinogradConv1d(conv, native=True)
                        if method == "winograd"
                        else PackedConv1d(conv)
                    ).eval()
                    times = []
                    for _ in range(args.warmups + args.samples):
                        start = time.perf_counter()
                        output = module(x)
                        times.append(time.perf_counter() - start)
                    delta = (output - reference).double()
                    snr = float(
                        10
                        * torch.log10(
                            reference.double().square().mean() / delta.square().mean()
                        )
                    )
                    row = {
                        "channels": channels,
                        "length": length,
                        "dilation": dilation,
                        "method": method,
                        "warmup_seconds": times[: args.warmups],
                        "seconds": times[args.warmups :],
                        "median_seconds": statistics.median(times[args.warmups :]),
                        "max_abs_error": float(delta.abs().max()),
                        "snr_db": snr if math.isfinite(snr) else None,
                    }
                    report["cases"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
