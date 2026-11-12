"""One-process opt-in FFN2-only row8 substitution for end-to-end PCM checks."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from experiments.rtf07.bench_fft_output_tiles import load_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--int8-ffn2-rows8-library", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    root = Path(__file__).resolve().parents[1]
    base_paths = [
        Path(os.environ.get("CPUFAST_AR_LIBRARY", root / "libcpufast_ar_torch.so")),
        Path(os.environ.get("CPUFAST_FAST_LIBRARY", root / "libcpufast_fast.so")),
    ]
    base_paths += [p.with_suffix(p.suffix + ".json") for p in base_paths]

    def baseline_hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in base_paths}

    baseline_before = baseline_hashes()
    import cpufast_fast_ops  # noqa: F401

    manifest = json.loads(
        args.int8_ffn2_rows8_library.with_suffix(".so.json").read_text()
    )
    if "-DCPUFAST_REPLACE_INT8_FFN2_ROWS8=1" not in manifest["flags"]:
        raise ValueError("Explicit disposable-process substitution DSO required")
    verified = load_probe(args.int8_ffn2_rows8_library)
    before = torch.ops.cpufast_cpp.int8_ffn2_rows8_calls()
    fallback_before = torch.ops.cpufast_cpp.int8_ffn2_fallback_calls()
    import benchmark_cpufast

    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        benchmark_cpufast.main()
    finally:
        sys.argv = previous
    calls = torch.ops.cpufast_cpp.int8_ffn2_rows8_calls() - before
    if calls == 0:
        raise RuntimeError("TorchScript did not call the FFN2-only row8 kernel")
    baseline_after = baseline_hashes()
    if baseline_before != baseline_after:
        raise RuntimeError("A baseline DSO or manifest file changed")
    report = json.loads(output.read_text())
    report["int8_ffn2_rows8"] = {
        "fallback_row4_calls": torch.ops.cpufast_cpp.int8_ffn2_fallback_calls() - fallback_before,
        "scope": "Only512x2048 FFN2 uses row8; other shapes retain original row4, quantization/dequantization unchanged",
        "calls": calls,
        "library_manifest": verified,
        "baseline_library_files_before": baseline_before,
        "baseline_library_files_after": baseline_after,
        "baseline_library_files_unchanged": baseline_before == baseline_after,
        "performance_adoption_claim": False,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
