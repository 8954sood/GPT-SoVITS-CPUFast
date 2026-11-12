"""Quality-only FFT128-on-Cin240 probe on approved AR17+original V400 weights."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if "--ar-fp32-cache" in remaining:
        parser.error("Keep approved exact-half-representable INT8 AR unchanged")
    import cpufast_engine
    from cpufast_fft128_cin240_runtime import enable_fft128_cin240

    original = cpufast_engine.optimize
    audit = {}
    def optimize(pipeline, **options):
        original_ar = pipeline.t2s_model.model.t2s_transformer
        options.update(fft_stream=True, parallel_transforms=True)
        runtime = original(pipeline, **options)
        import cpufast_fast_ops  # noqa: F401
        from cpufast_fast_ar import build
        ar, ar_audit = build(original_ar, head_major=True)
        pipeline.t2s_model.model.t2s_transformer = ar
        changed = enable_fft128_cin240(pipeline.vits_model.dec)
        if runtime.audit["fast_experimental"] or any(getattr(m, "fast_half", False) for m in pipeline.vits_model.modules()):
            raise RuntimeError("FFT128 probe must keep normal FP32 acoustic arithmetic")
        audit.update(ar_runtime=ar_audit, fft128_modules=changed,
                     baseline_fast_manifest=cpufast_fast_ops.manifest)
        runtime.audit["ar_coefficients"] = ar_audit
        return runtime
    cpufast_engine.optimize = optimize
    previous = sys.argv
    try:
        from experiments.student.synthesize_student import main as synthesize
        sys.argv = [previous[0], "--output", str(args.output), *remaining]
        synthesize()
    finally:
        sys.argv = previous
        cpufast_engine.optimize = original
        path = args.output / "comparison.json"
        if path.exists():
            report = json.loads(path.read_text())
            report.update(candidate="Unapproved runtime FFT128-on-Cin240; unchanged approved AR17 QAT200 and original V400 model weights",
                quality_approved=False, scope="Quality-only runtime numeric/listening candidate, not a performance benchmark",
                ar_arithmetic="Unchanged approved W8A8/native FP32 attention with canonical value schedule",
                acoustic_arithmetic="Normal FP32 except six FFT transform lengths256to128; no half products",
                runtime_audit=audit)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
