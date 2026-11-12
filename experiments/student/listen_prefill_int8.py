"""W listening: unchanged L weights/decode, approximate multirow W8A8 prompt FFNs."""

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefill-library",
        type=Path,
        default=root / ".experiments/libcpufast_prefill_int8_probe.so",
    )
    args, remaining = parser.parse_known_args()
    os.environ.setdefault(
        "CPUFAST_FAST_LIBRARY", str(root / ".experiments/libcpufast_fast.so")
    )
    import_module("cpufast_fast_ops")
    import cpufast_fast_ar
    from cpufast_prefill_int8_probe import build_prefill_ffn_proxy, load_library
    from experiments.student.listen_ar_int8 import main as listen

    load_library(args.prefill_library)
    original_build, previous = cpufast_fast_ar.build, sys.argv
    proxies = []

    def build(*build_args, **kwargs):
        runtime, details = original_build(*build_args, **kwargs)
        proxy = build_prefill_ffn_proxy(runtime)
        proxies.append(proxy)
        return proxy, dict(
            details,
            prefill_arithmetic="Experimental genuine multirow W8A8 FFN GEMM; existing decode delegated unchanged",
        )

    try:
        cpufast_fast_ar.build = build
        sys.argv = [previous[0], "--output", str(args.output), *remaining]
        listen()
    finally:
        cpufast_fast_ar.build = original_build
        sys.argv = previous
    if len(proxies) != 1 or not proxies[0].prefill_calls or proxies[0].busy:
        raise RuntimeError(
            "One scoped and restored prefill proxy must actually execute"
        )
    report_path = args.output / "comparison.json"
    report = json.loads(report_path.read_text())
    report.update(
        candidate="W: unchanged approved L model tensors and decode; W8A8 genuine multirow FFN prefill",
        prefill_calls=proxies[0].prefill_calls,
        prefill_audit=proxies[0].audit,
        prefill_arithmetic="Per-token activation INT8, per-output weight INT8, int32 SDOT, FP32 dequantization/bias; FFN1 fused ReLU",
        model_weights_changed=False,
        quality_approved=False,
        actual_n1_measured=False,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
