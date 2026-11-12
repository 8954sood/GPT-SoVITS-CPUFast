"""One-process opt-in quantization substitution for end-to-end PCM checks."""

import argparse
import json
import sys
from pathlib import Path

import torch

from experiments.rtf07.bench_fft_output_tiles import load_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--int8-quant-neon-library", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    import cpufast_fast_ops  # noqa: F401

    manifest = json.loads(
        args.int8_quant_neon_library.with_suffix(".so.json").read_text()
    )
    if "-DCPUFAST_REPLACE_INT8_QUANT_NEON=1" not in manifest["flags"]:
        raise ValueError("Explicit disposable-process substitution DSO required")
    verified = load_probe(args.int8_quant_neon_library)
    before = torch.ops.cpufast_cpp.int8_quant_neon_calls()
    import benchmark_cpufast

    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        benchmark_cpufast.main()
    finally:
        sys.argv = previous
    calls = torch.ops.cpufast_cpp.int8_quant_neon_calls() - before
    if calls == 0:
        raise RuntimeError("TorchScript did not call the NEON quantization kernel")
    report = json.loads(output.read_text())
    report["int8_quant_neon"] = {
        "scope": "Input quantization only; unchanged row4 SDOT/dequantization",
        "calls": calls,
        "library_manifest": verified,
        "baseline_library_files_unchanged": True,
        "performance_adoption_claim": False,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
