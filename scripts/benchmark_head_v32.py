"""One-process opt-in head-value scheduling substitution for end-to-end PCM checks."""

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
    parser.add_argument("--head-v32-library", type=Path, required=True)
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
        args.head_v32_library.with_suffix(".so.json").read_text()
    )
    if "-DCPUFAST_REPLACE_HEAD_V32=1" not in manifest["flags"]:
        raise ValueError("Explicit disposable-process substitution DSO required")
    verified = load_probe(args.head_v32_library)
    before = torch.ops.cpufast_cpp.head_v32_calls()
    import benchmark_cpufast

    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        benchmark_cpufast.main()
    finally:
        sys.argv = previous
    calls = torch.ops.cpufast_cpp.head_v32_calls() - before
    if calls == 0:
        raise RuntimeError("TorchScript did not call the head-value scheduling kernel")
    baseline_after = baseline_hashes()
    if baseline_before != baseline_after:
        raise RuntimeError("A baseline DSO or manifest file changed")
    report = json.loads(output.read_text())
    report["head_v32"] = {
        "scope": "Dim32 weighted-value schedule only; unchanged QK/softmax and per-dimension token order",
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
