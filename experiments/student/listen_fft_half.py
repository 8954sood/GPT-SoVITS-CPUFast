"""Explicit listening probe of FP16 FFT products on approved L model weights.

AR stays W8A8/native FP32 attention. Only remaining large FFT convolution
spectral products use FP16 coefficients, inputs and accumulation. Transform,
tap/Winograd and finishing operations retain their normal FP32 paths.
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
        parser.error("Approved half-representable AR checkpoint required")
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault(
        "CPUFAST_FAST_LIBRARY", str(root / ".experiments/libcpufast_fast.so")
    )
    import cpufast_engine

    original = cpufast_engine.optimize
    audit = {}

    def approximate(pipeline, **options):
        original_ar = pipeline.t2s_model.model.t2s_transformer
        runtime = original(pipeline, **options)
        import cpufast_fast_ops  # noqa: F401
        from cpufast_fast_ar import build
        from cpufast_fft_half_runtime import enable_fft_half_products

        ar, ar_audit = build(original_ar, head_major=True)
        pipeline.t2s_model.model.t2s_transformer = ar
        modified = enable_fft_half_products(pipeline.vits_model.dec)
        if not modified or runtime.audit["fast_experimental"]:
            raise RuntimeError("Isolated FFT product probe expected")
        audit.update(ar_runtime=ar_audit, fp16_fft_modules=modified)
        runtime.audit.update(
            arithmetic="W8A8 AR; FP16 FFT products; other acoustic operations FP32"
        )
        return runtime

    cpufast_engine.optimize = approximate
    try:
        from experiments.student.synthesize_student import main as synthesize

        sys.argv = [sys.argv[0], "--output", str(args.output), *remaining]
        synthesize()
    finally:
        cpufast_engine.optimize = original
        path = args.output / "comparison.json"
        if path.exists():
            report = json.loads(path.read_text())
            report.update(
                candidate="P: approved K weights, L AR execution, isolated FP16 FFT products",
                ar_cache_storage="INT8 derived weight cache",
                ar_arithmetic="W8A8 with native FP32 attention",
                acoustic_arithmetic="FP16 FFT spectral products and accumulation; other acoustic ops FP32",
                approximate_inference=True,
                quality_approved=False,
                precision_audit=audit,
            )
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
