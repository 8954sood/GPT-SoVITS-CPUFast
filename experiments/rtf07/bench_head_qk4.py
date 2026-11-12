"""Exact per-token dimension-order four-token QK scheduling; primitive, not synthesis RTF."""

import argparse
import hashlib
import platform
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
    baseline = torch.ops.cpufast_cpp.head_attn
    candidate = torch.ops.cpufast_cpp.head_attn_qk4
    torch.manual_seed(20261112)
    root = Path(__file__).resolve().parents[2]
    baseline_paths = [root / "libcpufast_ar_torch.so", root / "libcpufast_fast.so"]
    baseline_paths += [p.with_suffix(p.suffix + ".json") for p in baseline_paths]
    baseline_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_paths}
    report = {"baseline_files_before": baseline_before,
              "baseline_fast_manifest": cpufast_fast_ops.manifest,
              "machine": platform.machine(), "torch": torch.__version__,
              "cgroup": {name: (Path("/sys/fs/cgroup") / name).read_text().strip()
                         for name in ("cpu.max", "memory.max", "memory.swap.max")},
              "scope": "Actual host primitive only; unchanged softmax, canonical value scheduling and per-token dimension FMA order",
              "probe_manifest": manifest, "correctness": [], "cases": []}
    for heads, dim, length, capacity in (
        (1, 32, 1, 5), (16, 32, 257, 1756), (16, 32, 374, 1756),
        (3, 16, 19, 31), (2, 64, 11, 17),
    ):
        q = torch.randn(1, 1, heads * dim)
        keys = torch.randn(1, heads, capacity, dim)
        values = torch.randn_like(keys)
        shared = torch.rand(length) < 0.25
        per_head = torch.rand(1, heads, 1, length) < 0.25
        per_head[:, 0] = True
        for mask_label, mask in (
            ("none", None), ("shared", shared), ("per_head_one_blocked", per_head),
            ("all_blocked", torch.ones(length, dtype=torch.bool)),
        ):
            for factor in (0.0, 1.0, 100.0):
                scaled = q * factor
                before = [x.clone() for x in (scaled, keys, values)]
                old_mask = mask.clone() if mask is not None else None
                for threads in (1, 2):
                    torch.set_num_threads(threads)
                    expected = baseline(scaled, keys, values, length, heads, mask)
                    actual = candidate(scaled, keys, values, length, heads, mask)
                    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (
                        heads, dim, length, mask_label, factor, threads,
                        float((actual - expected).abs().max()),
                    )
                    report["correctness"].append({
                        "heads": heads, "dim": dim, "length": length,
                        "capacity": capacity, "mask": mask_label,
                        "q_scale": factor, "threads": threads, "bitwise_equal": True,
                    })
                assert all(torch.equal(a, b) for a, b in zip((scaled, keys, values), before))
                assert mask is None or torch.equal(mask, old_mask)
    torch.set_num_threads(2)
    for length in (257, 315, 374, 768):
        heads, dim, capacity = 16, 32, 1756
        q = torch.randn(1, 1, heads * dim)
        keys = torch.randn(1, heads, capacity, dim)
        values = torch.randn_like(keys)
        mask = torch.zeros(1, heads, 1, length, dtype=torch.bool)
        functions = {
            "baseline": lambda: baseline(q, keys, values, length, heads, mask),
            "qk4": lambda: candidate(q, keys, values, length, heads, mask),
        }
        for call in functions.values():
            for _ in range(20):
                call()
        samples = {name: [] for name in functions}
        for turn in range(7):
            for name in (tuple(functions) if turn % 2 == 0 else tuple(reversed(functions))):
                started = time.perf_counter()
                for _ in range(100):
                    functions[name]()
                samples[name].append((time.perf_counter() - started) / 100)
        medians = {name: statistics.median(v) for name, v in samples.items()}
        row = {"length": length, "heads": heads, "dim": dim, "capacity": capacity,
               "hot_samples": samples, "hot_medians": medians,
               "hot_candidate_over_baseline": medians["qk4"] / medians["baseline"]}
        # One actual decoder's17 distinct cache tensors: expose a realistic
        # multi-layer active KV footprint rather than only one hot layer.
        pool = [(torch.randn_like(keys), torch.randn_like(values)) for _ in range(17)]
        rotating = {name: [] for name in functions}
        operators = {"baseline": baseline, "qk4": candidate}
        for turn in range(7):
            for name in (tuple(operators) if turn % 2 == 0 else tuple(reversed(operators))):
                started = time.perf_counter()
                for key, value in pool:
                    operators[name](q, key, value, length, heads, mask)
                rotating[name].append((time.perf_counter() - started) / len(pool))
        medians = {name: statistics.median(v) for name, v in rotating.items()}
        row["rotating_caches"] = {
            "layers": len(pool), "active_bytes": len(pool) * heads * length * dim * 4 * 2,
            "samples": rotating, "medians": medians,
            "candidate_over_baseline": medians["qk4"] / medians["baseline"],
            "note": "Synthetic KV-only layer traversal; actual AR also streams weights",
        }
        del pool
        report["cases"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        baseline_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_paths}
        assert baseline_before == baseline_after
        report["baseline_files_after"] = baseline_after
        report["baseline_files_unchanged"] = baseline_before == baseline_after
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(length, row["hot_candidate_over_baseline"],
              row["rotating_caches"]["candidate_over_baseline"], flush=True)


if __name__ == "__main__":
    main()
