"""Exact row4 packed-weight cache; actual model weights/synthetic inputs, not RTF."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.rtf07.bench_fft_output_tiles import load_probe


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--ar-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    import cpufast_fast_ops

    manifest = load_probe(args.library)
    assert manifest["flags"] == cpufast_fast_ops.manifest["flags"]
    baseline = torch.ops.cpufast_cpp.int8_linear
    candidate = torch.ops.cpufast_cpp.int8_linear_packed4
    root = Path(__file__).resolve().parents[2]
    native_paths = [root / "libcpufast_ar_torch.so", root / "libcpufast_fast.so"]
    native_paths += [p.with_suffix(p.suffix + ".json") for p in native_paths]
    before = {p.name: digest(p) for p in native_paths}
    assert digest(args.ar_checkpoint) == args.ar_sha256
    torch.manual_seed(2026111208)
    report = {"scope": "Actual N1 primitive only; actual approved AR weights with synthetic activations, not synthesis RTF",
        "probe_manifest": manifest, "baseline_fast_manifest": cpufast_fast_ops.manifest,
        "baseline_files_before": before, "ar_checkpoint_sha256": args.ar_sha256,
        "cgroup": {name: (Path("/sys/fs/cgroup") / name).read_text().strip()
                   for name in ("cpu.max", "memory.max", "memory.swap.max")},
        "correctness": [], "timings": []}
    for n, k in ((4, 16), (12, 32), (16, 32), (1536, 512), (512, 512), (2048, 512), (512, 2048)):
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8)
        scales, bias = torch.rand(n) / 100, torch.randn(n)
        for factor in (0.0, 1e-10, 1.0, 1e10):
            x = torch.randn(1, 1, k) * factor
            originals = [a.clone() for a in (x, w, scales, bias)]
            for threads in (1, 2):
                torch.set_num_threads(threads)
                expected, actual = baseline(x, w, scales, bias), candidate(x, w, scales, bias)
                assert torch.equal(expected.view(torch.int32), actual.view(torch.int32)), (n, k, factor, threads)
                report["correctness"].append({"n": n, "k": k, "factor": factor,
                    "threads": threads, "selected_rows": 4, "bitwise_equal": True})
            assert all(torch.equal(a, b) for a, b in zip((x, w, scales, bias), originals))
        for invalid in (float("nan"), float("inf"), -float("inf")):
            x = torch.randn(1, 1, k); x[..., 0] = invalid
            for op in (baseline, candidate):
                try:
                    op(x, w, scales, bias)
                except RuntimeError:
                    continue
                raise AssertionError("Nonfinite input accepted")
        x = torch.zeros(1, 1, k)
        signed = torch.zeros(n); signed[1::2] = -0.0
        for threads in (1, 2):
            torch.set_num_threads(threads)
            assert torch.equal(baseline(x, w, signed, signed).view(torch.int32),
                               candidate(x, w, signed, signed).view(torch.int32))
            report["correctness"].append({"n": n, "k": k, "signed_zero": True,
                "threads": threads, "bitwise_equal": True})
    # A stale pack after weight mutation would silently use the wrong model.
    w=torch.randint(-127,128,(4,16),dtype=torch.int8)
    x=torch.randn(1,1,16);scale=torch.ones(4);bias=torch.zeros(4)
    candidate(x,w,scale,bias)
    w[0,0] = 91
    assert torch.equal(baseline(x,w,scale,bias).view(torch.int32),candidate(x,w,scale,bias).view(torch.int32))
    with torch.inference_mode():
        wi=w.clone()
        candidate(x,wi,scale,bias)
        wi[0,0] = -91
        assert torch.equal(baseline(x,wi,scale,bias).view(torch.int32),candidate(x,wi,scale,bias).view(torch.int32))
    report["weight_mutation_and_inference_weight_no_stale_cache"] = True
    torch.set_num_threads(2)
    state = torch.load(args.ar_checkpoint, map_location="cpu", weights_only=False)["weight"]
    roles = (("qkv", "self_attn.in_proj"), ("out", "self_attn.out_proj"),
             ("ffn1", "linear1"), ("ffn2", "linear2"))
    pool = []
    for layer in range(17):
        for role, stem in roles:
            prefix = f"model.h.layers.{layer}.{stem}"
            key = prefix + ("_weight" if role == "qkv" else ".weight")
            bias_key = prefix + ("_bias" if role == "qkv" else ".bias")
            weight = state[key].float()
            scale = torch.clamp(weight.abs().amax(dim=1) / 127.0, min=1e-12)
            quant = torch.clamp(torch.round(weight / scale[:, None]), -127, 127).to(torch.int8).contiguous()
            bias = state[bias_key].float().contiguous()
            x = torch.randn(1, 1, weight.shape[1])
            assert torch.equal(baseline(x, quant, scale, bias).view(torch.int32),
                               candidate(x, quant, scale, bias).view(torch.int32))
            pool.append((role, x, quant, scale.contiguous(), bias))
    report["actual_weight_matrices"] = [{"role": r, "shape": list(w.shape)} for r, _, w, _, _ in pool]
    report["actual_weight_bytes"] = sum(w.numel() for _, _, w, _, _ in pool)
    del state
    for label, rows, repeats in (("hot_actual_FFN2", [pool[3]], 100),
                                 ("rotating17layers_FFN2_only", [r for r in pool if r[0] == "ffn2"], 5),
                                 ("rotating17layers_all4_GEMV", pool, 5)):
        functions = {"row4": baseline, "packed4": candidate}
        for op in functions.values():
            for _, x, w, scale, bias in rows:
                op(x, w, scale, bias)
        samples = {name: [] for name in functions}
        start4stats = list(torch.ops.cpufast_cpp.int8_packed4_stats())

        for turn in range(7):
            for name in (tuple(functions) if turn % 2 == 0 else tuple(reversed(functions))):
                started = time.perf_counter()
                for _ in range(repeats):
                    for _, x, w, scale, bias in rows:
                        functions[name](x, w, scale, bias)
                samples[name].append((time.perf_counter() - started) / (repeats * len(rows)))
        medians = {name: statistics.median(v) for name, v in samples.items()}
        report["timings"].append({"case": label, "samples_seconds_per_GEMV": samples,
            "medians": medians, "candidate_over_baseline": medians["packed4"] / medians["row4"],
            "packed_calls": torch.ops.cpufast_cpp.int8_packed4_stats()[0]-start4stats[0],
            "cache_stats_calls_hits_misses_entries_bytes": list(torch.ops.cpufast_cpp.int8_packed4_stats())})
    report["baseline_files_after"] = {p.name: digest(p) for p in native_paths}
    assert report["baseline_files_after"] == before and digest(args.ar_checkpoint) == args.ar_sha256
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print([(r["case"], r["candidate_over_baseline"]) for r in report["timings"]], flush=True)


if __name__ == "__main__":
    main()
