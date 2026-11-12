"""Compare FP32 head-major attention kernels at AR cache lengths on target ARM."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import cpufast_cpp_ops  # noqa: F401


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
        parser.error("Use a fresh output and bounded iteration counts")
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    q = torch.randn(1, 1, 512)
    keys = torch.randn(1, 16, 768, 32)
    values = torch.randn_like(keys)
    rows = []
    with torch.no_grad():
        for length in (64, 128, 256, 512):
            for threads in (1, 2):
                torch.set_num_threads(threads)

                def standard():
                    return (
                        F.scaled_dot_product_attention(
                            q.view(1, 1, 16, 32).transpose(1, 2),
                            keys[:, :, :length],
                            values[:, :, :length],
                            None,
                        )
                        .transpose(1, 2)
                        .reshape_as(q)
                    )

                reference = standard()
                for label in ("torch", "native"):

                    def call():
                        return (
                            standard()
                            if label == "torch"
                            else torch.ops.cpufast_cpp.head_attn_fp32(
                                q, keys, values, length, 16, None
                            )
                        )

                    out = call()
                    torch.testing.assert_close(out, reference, atol=1e-6, rtol=1e-5)
                    seconds, cpus = [], []
                    for _ in range(args.samples + 1):
                        start, cpu = time.perf_counter(), time.process_time()
                        for _ in range(args.calls):
                            out = call()
                        seconds.append((time.perf_counter() - start) / args.calls)
                        cpus.append((time.process_time() - cpu) / args.calls)
                    row = {
                        "length": length,
                        "threads": threads,
                        "method": label,
                        "warmup_seconds_per_call": seconds[0],
                        "seconds_per_call": seconds[1:],
                        "cpu_seconds_per_call": cpus[1:],
                        "median_seconds_per_call": statistics.median(seconds[1:]),
                        "max_abs_error": float((out - reference).abs().max()),
                    }
                    rows.append(row)
                    args.output.write_text(
                        json.dumps(
                            {
                                "scope": "Synthetic attention probe; not end-to-end RTF",
                                "cases": rows,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
