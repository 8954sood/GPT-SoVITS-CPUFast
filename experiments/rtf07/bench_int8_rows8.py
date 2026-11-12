"""Row8 SDOT primitive equality and synthetic timing; no model/RTF claims."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.rtf07.bench_fft_output_tiles import load_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    import cpufast_fast_ops  # noqa: F401

    manifest = load_probe(args.library)
    baseline = torch.ops.cpufast_cpp.int8_linear
    candidate = torch.ops.cpufast_cpp.int8_linear_rows8
    torch.manual_seed(2026111208)
    report = {"scope": "Actual executing host hot primitive; not synthesis RTF",
              "probe_manifest": manifest, "correctness": [], "cases": []}
    # Keep quantization unchanged: cover zero, tiny/large finite magnitudes and
    # real AR dimensions under both thread settings. Integer sum/dequant order
    # must match bitwise, not merely satisfy a quantization tolerance.
    for n, k in ((16, 32), (1536, 512), (512, 512), (2048, 512), (512, 2048)):
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8)
        s = torch.rand(n) / 100
        bias = torch.randn(n)
        for factor in (0.0, 1e-10, 1.0, 1e10):
            x = torch.randn(1, 1, k) * factor
            originals = [t.clone() for t in (x, w, s, bias)]
            for threads in (1, 2):
                torch.set_num_threads(threads)
                expected = baseline(x, w, s, bias)
                actual = candidate(x, w, s, bias)
                assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (
                    n, k, factor, threads, float((actual - expected).abs().max())
                )
                report["correctness"].append({"n": n, "k": k, "factor": factor,
                                              "threads": threads, "bitwise_equal": True})
            assert all(torch.equal(a, b) for a, b in zip((x, w, s, bias), originals))
        for invalid in (float("nan"), float("inf"), -float("inf")):
            x = torch.randn(1, 1, k); x[0, 0, 0] = invalid
            for call in (baseline, candidate):
                try:
                    call(x, w, s, bias)
                except RuntimeError:
                    continue
                raise AssertionError("Nonfinite input reached SDOT")
        if n == 16:
            continue
        torch.set_num_threads(2)
        x = torch.randn(1, 1, k)
        functions = {"rows4": lambda: baseline(x, w, s, bias),
                     "rows8": lambda: candidate(x, w, s, bias)}
        for call in functions.values():
            for _ in range(20):
                call()
        samples = {name: [] for name in functions}
        for turn in range(7):
            for name in (tuple(functions) if turn % 2 == 0 else tuple(reversed(functions))):
                start = time.perf_counter()
                for _ in range(100):
                    functions[name]()
                samples[name].append((time.perf_counter() - start) / 100)
        row = {"n": n, "k": k, "samples": samples,
               "medians": {name: statistics.median(v) for name, v in samples.items()}}
        row["candidate_over_baseline"] = row["medians"]["rows8"] / row["medians"]["rows4"]
        report["cases"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(n, k, row["medians"], row["candidate_over_baseline"], flush=True)


if __name__ == "__main__":
    main()
