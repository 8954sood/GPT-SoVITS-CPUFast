"""Probe W8A8 with the same head-major KV cache and Torch SDPA attention.

Keeps the approved L helper byte-for-byte unchanged. Only the optional attention
implementation is injected; model parameters and normal FP32 acoustics remain.
"""

import argparse
import functools
import importlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault(
        "CPUFAST_FAST_LIBRARY", str(root / ".experiments/libcpufast_fast.so")
    )
    importlib.import_module("cpufast_fast_ops")
    import cpufast_fast_ar

    original = cpufast_fast_ar.build
    cpufast_fast_ar.build = functools.partial(original, torch_head_attention=True)
    try:
        from experiments.student.listen_ar_int8 import main as listen

        listen()
    finally:
        cpufast_fast_ar.build = original
        path = args.output / "comparison.json"
        if path.exists():
            report = json.loads(path.read_text())
            report.update(
                candidate="M: approved K weights with W8A8 AR and head-major Torch SDPA",
                ar_arithmetic="W8A8 with FP32 dequantization and Torch SDPA attention",
                quality_approved=False,
            )
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
