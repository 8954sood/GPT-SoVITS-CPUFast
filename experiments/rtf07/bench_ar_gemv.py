"""Probe exact-coefficient AR GEMV tiling and threading without changing weights."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

import cpufast_cpp_ops  # noqa: F401; registers CPU operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.calls <= 1000
        or not 1 <= args.samples <= 20
    ):
        parser.error("Use a new output, 1–1000 calls and 1–20 samples")
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    report = {
        "scope": "Synthetic AR GEMV probe; not end-to-end synthesis or RTF",
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cases": [],
    }
    with torch.no_grad():
        for n, k in ((1536, 512), (512, 512), (2048, 512), (512, 2048), (1040, 512)):
            x, bias = torch.randn(1, 1, k), torch.randn(n)
            weights = torch.randn(n, k).half()
            reference = None
            for tile in (8, 16):
                packed = (
                    weights.reshape(n // tile, tile, k).transpose(1, 2).contiguous()
                )
                op = (
                    torch.ops.cpufast_cpp.lossless_linear
                    if tile == 8
                    else torch.ops.cpufast_cpp.lossless_linear16
                )
                for threads in (1, 2):
                    torch.set_num_threads(threads)
                    out = op(x, packed, bias)
                    if reference is None:
                        reference = out
                    if not torch.equal(out, reference):
                        raise RuntimeError(
                            "GEMV tile or thread count changed the FP32 output"
                        )
                    times, cpus = [], []
                    for _ in range(args.samples + 1):
                        start, cpu = time.perf_counter(), time.process_time()
                        for _ in range(args.calls):
                            out = op(x, packed, bias)
                        times.append((time.perf_counter() - start) / args.calls)
                        cpus.append((time.process_time() - cpu) / args.calls)
                    row = {
                        "n": n,
                        "k": k,
                        "tile": tile,
                        "threads": threads,
                        "warmup_seconds_per_call": times[0],
                        "seconds_per_call": times[1:],
                        "cpu_seconds_per_call": cpus[1:],
                        "median_seconds_per_call": statistics.median(times[1:]),
                        "exact_vs_baseline": True,
                    }
                    report["cases"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
