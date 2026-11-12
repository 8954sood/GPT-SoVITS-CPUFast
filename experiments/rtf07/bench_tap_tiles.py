"""Actual-N1 synthetic probe of temporal tiles for existing FP32 tap GEMM.

This is not an end-to-end TTS benchmark. No model file is loaded or rewritten.
BLAS shape changes can alter FP32 rounding; numerical differences are measured,
never assumed absent. Full synthesis and quality checks are required separately.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def tap_tiles(packed, padded, bias, length, dilation, tile=0):
    output = padded.new_empty((packed.shape[1], length))
    for left in range(0, length, tile or length):
        right = min(length, left + (tile or length))
        part = output[:, left:right]
        torch.mm(packed[0], padded[:, left:right], out=part)
        for tap in range(1, packed.shape[0]):
            start = left + tap * dilation
            torch.addmm(
                part, packed[tap], padded[:, start : start + right - left], out=part
            )
    if bias is not None:
        output.add_(bias[:, None])
    return output.unsqueeze(0)


def cgroup():
    root = Path("/sys/fs/cgroup")
    return {
        name: (root / name).read_text().strip()
        for name in ("cpu.max", "memory.max", "memory.swap.max")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.warmups <= 5
        or not 1 <= args.samples <= 20
    ):
        parser.error("Fresh output and bounded sample counts required")
    settings = cgroup()
    if platform.machine() != "aarch64" or settings != {
        "cpu.max": "200000 100000",
        "memory.max": "8000000000",
        "memory.swap.max": "0",
    }:
        parser.error("Actual 2CPU / 8GB / no-swap ARM target required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report = {
        "scope": "Synthetic actual-N1 tap tiling; not full TTS/PCM/quality approval",
        "cgroup": settings,
        "threads": 2,
        "warmups": args.warmups,
        "samples": args.samples,
        "seed": 412,
        "rows": [],
        "complete": False,
    }
    torch.manual_seed(412)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with torch.inference_mode():
        for channels, kernel, dilation, length in [
            (48, 7, 1, 90880),
            (48, 11, 5, 90880),
            (24, 7, 1, 181760),
            (24, 11, 5, 181760),
        ]:
            x = torch.randn(channels, length)
            padding = dilation * (kernel - 1) // 2
            padded = F.pad(x, (padding, padding))
            packed = torch.randn(kernel, channels, channels) / channels**0.5
            bias = torch.randn(channels)
            expected = tap_tiles(packed, padded, bias, length, dilation)
            before = packed.clone()
            tiles = [0, 1024, 4096, 8192]
            # Reverse order for alternating shapes to limit chronological bias.
            if channels == 24:
                tiles.reverse()
            for tile in tiles:
                warmups = []
                for _ in range(args.warmups):
                    started = time.perf_counter()
                    actual = tap_tiles(packed, padded, bias, length, dilation, tile)
                    warmups.append(time.perf_counter() - started)
                wall, cpu = [], []
                for _ in range(args.samples):
                    cpu_started, started = time.process_time(), time.perf_counter()
                    actual = tap_tiles(packed, padded, bias, length, dilation, tile)
                    wall.append(time.perf_counter() - started)
                    cpu.append(time.process_time() - cpu_started)
                error = actual - expected
                report["rows"].append(
                    {
                        "channels": channels,
                        "kernel": kernel,
                        "dilation": dilation,
                        "length": length,
                        "tile": tile,
                        "wall_seconds": wall,
                        "cpu_seconds": cpu,
                        "warmup_seconds": warmups,
                        "median_seconds": statistics.median(wall),
                        "median_cpu_seconds": statistics.median(cpu),
                        "exact": torch.equal(actual, expected),
                        "changed_values": int(torch.count_nonzero(error)),
                        "max_abs_error": float(error.abs().max()),
                        "rmse": float(error.square().mean().sqrt()),
                        "finite": bool(torch.isfinite(actual).all()),
                        "coefficients_unchanged": torch.equal(packed, before),
                    }
                )
                save()
                print(
                    f"tap C{channels} K{kernel} D{dilation} tile{tile}: {statistics.median(wall):.6f}s, exact={torch.equal(actual, expected)}",
                    flush=True,
                )
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
