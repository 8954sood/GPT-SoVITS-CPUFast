"""Exact-FP32 output tiling correctness and actual-host primitive timings.

This is a shape microbenchmark, not synthesis RTF. Default native code stays
unchanged. Only run one benchmark at a time on the target host.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch


def load_probe(path):
    import cpufast_cpp_ops  # noqa: F401

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if manifest["torch_version"] != torch.__version__:
        raise ValueError("Probe PyTorch version mismatch")
    for file, digest in manifest["sources"].items():
        if hashlib.sha256((root / file).read_bytes()).hexdigest() != digest:
            raise ValueError("Probe source changed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["library_sha256"]:
        raise ValueError("Probe DSO changed")
    torch.ops.load_library(str(path))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest = load_probe(args.library)
    torch.manual_seed(20261112)
    report = {
        "scope": "actual executing host primitive only; not synthesis RTF",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "probe_manifest": manifest,
        "groups_per_tile": 4,
        "correctness": [],
        "cases": [],
    }
    baseline = torch.ops.cpufast_cpp.fft_packed_stream
    candidate = torch.ops.cpufast_cpp.fft_packed_output_tiles
    # Batch/ragged row/frequency/output-chunk tails and 1/2 threads.
    for batch, blocks, k, groups, freq in (
        (1, 1, 24, 1, 5), (2, 7, 48, 5, 9), (1, 19, 192, 16, 129),
    ):
        x = torch.randn(3 * freq, (batch * blocks + 7) // 8, k, 8)
        w = torch.randn(3 * freq, groups, k, 12)
        old_x, old_w = x.clone(), w.clone()
        for threads in (1, 2):
            torch.set_num_threads(threads)
            for chunk in (1, min(4, groups), groups):
                expected = baseline(x, w, batch, blocks)
                actual = candidate(x, w, batch, blocks, chunk)
                if not torch.equal(actual, expected):
                    raise AssertionError("FP32 output tiling changed exact values")
                report["correctness"].append({
                    "batch": batch, "blocks": blocks, "k": k, "groups": groups,
                    "freq": freq, "threads": threads, "chunk": chunk,
                    "bitwise_equal": True,
                })
        assert torch.equal(x, old_x) and torch.equal(w, old_w)
    torch.set_num_threads(2)
    # Approved L stage shapes and proposed V rectangular execution shapes.
    # These synthetic values do not load, validate or approve a model.
    for label, k, outputs, blocks, freq in (
        ("L_C384_short", 384, 384, 25, 65),
        ("L_C192_long", 192, 192, 84, 129),
        ("L_C96_long", 96, 96, 184, 129),
        ("V_C384_to240", 384, 240, 25, 65),
        ("V_C240_to384", 240, 384, 10, 129),
        ("V_C192_to96", 192, 96, 84, 129),
    ):
        x = torch.randn(3 * freq, (blocks + 7) // 8, k, 8)
        w = torch.randn(3 * freq, outputs // 12, k, 12)
        functions = {
            "baseline": lambda: baseline(x, w, 1, blocks),
            "output_tile4": lambda: candidate(x, w, 1, blocks, 4),
        }
        assert torch.equal(functions["baseline"](), functions["output_tile4"]())
        for call in functions.values():
            for _ in range(3):
                call()
        times = {name: [] for name in functions}
        for turn in range(7):
            for name in (tuple(functions) if turn % 2 == 0 else tuple(reversed(functions))):
                started = time.perf_counter()
                functions[name]()
                times[name].append(time.perf_counter() - started)
        row = {"shape": label, "k": k, "outputs": outputs, "blocks": blocks,
               "freq": freq, "samples": times,
               "medians": {name: statistics.median(v) for name, v in times.items()}}
        row["candidate_over_baseline"] = row["medians"]["output_tile4"] / row["medians"]["baseline"]
        report["cases"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(label, row["medians"], row["candidate_over_baseline"], flush=True)


if __name__ == "__main__":
    main()
