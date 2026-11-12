"""Measure exact AR GEMV prefetch locality with a 24-layer, model-sized weight set."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import cpufast_cpp_ops  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.tokens <= 100
        or not 1 <= args.samples <= 20
    ):
        parser.error("Use a new output, 1–100 tokens and 1–20 samples")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    shapes = ((1536, 512), (512, 512), (2048, 512), (512, 2048))
    weights = []
    for _ in range(24):
        for n, k in shapes:
            w = (
                torch.randn(n, k)
                .half()
                .reshape(n // 8, 8, k)
                .transpose(1, 2)
                .contiguous()
            )
            weights.append((torch.randn(1, 1, k), w, torch.randn(n)))
    total_bytes = sum(w.numel() * w.element_size() for _, w, _ in weights)
    op = torch.ops.cpufast_cpp.lossless_linear_prefetch_locality
    with torch.no_grad():
        for x, w, b in weights:
            baseline = op(x, w, b, 0)
            for distance in (1, 2, 3):
                if not torch.equal(baseline, op(x, w, b, distance)):
                    raise RuntimeError("Prefetch altered the output")
        rows = []
        for distance in (0, 1, 2, 3):
            timings = []
            for _ in range(args.samples + 1):
                start = time.perf_counter()
                for _ in range(args.tokens):
                    for x, w, b in weights:
                        op(x, w, b, distance)
                timings.append((time.perf_counter() - start) / args.tokens)
            rows.append(
                {
                    "locality": distance,
                    "warmup_seconds_per_token": timings[0],
                    "seconds_per_token": timings[1:],
                    "median_seconds_per_token": statistics.median(timings[1:]),
                    "bit_exact_all_layers": True,
                }
            )
            args.output.write_text(
                json.dumps(
                    {
                        "scope": "Synthetic 24-layer weight traversal; not end-to-end RTF",
                        "weight_bytes": total_bytes,
                        "cases": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
