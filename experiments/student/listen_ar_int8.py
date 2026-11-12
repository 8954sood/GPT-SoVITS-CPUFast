"""Isolated listening probe: W8A8/native-attention AR, normal FP32 acoustics.

Reuses existing experimental AR kernels without enabling the mixed global fast
profile. Original model tensors/files remain authoritative and unchanged; the
derived AR runtime is approximate and requires listening and actual-N1 checks.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if "--ar-fp32-cache" in remaining:
        parser.error("This probe requires the approved half-representable AR copy")
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault(
        "CPUFAST_FAST_LIBRARY", str(root / ".experiments/libcpufast_fast.so")
    )
    import cpufast_engine

    original_optimize = cpufast_engine.optimize
    audit = {}

    def optimize_ar_only(pipeline, **options):
        original_ar = pipeline.t2s_model.model.t2s_transformer
        runtime = original_optimize(pipeline, **options)
        import cpufast_fast_ops  # noqa: F401
        from cpufast_fast_ar import build

        approximate, details = build(original_ar, head_major=True)
        pipeline.t2s_model.model.t2s_transformer = approximate
        half_acoustics = sum(
            bool(getattr(module, "fast_half", False))
            for module in pipeline.vits_model.modules()
        )
        if half_acoustics or runtime.audit["fast_experimental"]:
            raise RuntimeError("AR-only probe must preserve normal FP32 acoustics")
        audit.update(details)
        audit["fp16_acoustic_modules"] = half_acoustics
        runtime.audit.update(
            ar_coefficients=details,
            arithmetic="Approximate W8A8 AR/native FP32 attention; normal FP32 acoustics",
        )
        return runtime

    cpufast_engine.optimize = optimize_ar_only
    try:
        from experiments.student.synthesize_student import main as synthesize

        sys.argv = [sys.argv[0], "--output", str(args.output), *remaining]
        synthesize()
    finally:
        cpufast_engine.optimize = original_optimize
        report_path = args.output / "comparison.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report.update(
                candidate="L: approved K weights with isolated experimental AR runtime",
                ar_cache_storage="INT8 per-output-channel weights",
                ar_arithmetic="W8A8 with FP32 dequantization and native FP32 attention",
                acoustic_arithmetic="Normal FP32 engine, no FP16 FFT accumulation",
                approximate_inference=True,
                quality_approved=False,
                ar_runtime_audit=audit,
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
