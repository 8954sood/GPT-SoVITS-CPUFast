"""Unapproved packed FP32 FFN-prefill voice candidate on unchanged L weights."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    import cpufast_engine
    from cpufast_prefill_probe import build_prefill_ffn_proxy, load_library

    load_library(os.environ["CPUFAST_PREFILL_LIBRARY"])
    original = cpufast_engine.optimize
    audit = {}

    def candidate(pipeline, **options):
        original_ar = pipeline.t2s_model.model.t2s_transformer
        runtime = original(pipeline, **options)
        import cpufast_fast_ops  # noqa: F401
        from cpufast_fast_ar import build

        ar, details = build(original_ar, head_major=True)
        proxy = build_prefill_ffn_proxy(ar)
        pipeline.t2s_model.model.t2s_transformer = proxy
        half = sum(
            bool(getattr(m, "fast_half", False)) for m in pipeline.vits_model.modules()
        )
        if half or runtime.audit["fast_experimental"]:
            raise RuntimeError("L acoustic FP32 required")
        audit.update(
            prefill_ffn=proxy.audit,
            fp16_acoustic_modules=half,
            decode="Unchanged L W8A8/native FP32 attention",
            prefill="FP32 packed FFN1-bias-ReLU/FFN2 only",
        )
        runtime.audit.update(
            ar_coefficients=details,
            arithmetic="Unapproved FP32 packed FFN prefill; unchanged L decode/acoustics",
        )
        return runtime

    cpufast_engine.optimize = candidate
    try:
        from experiments.student.synthesize_student import main as synthesize

        sys.argv = [sys.argv[0], "--output", str(args.output), *remaining]
        synthesize()
    finally:
        cpufast_engine.optimize = original
        output = args.output / "comparison.json"
        if output.exists():
            report = json.loads(output.read_text())
            report.update(
                candidate="Unapproved FP32 packed FFN prefill on approved L weights",
                quality_approved=False,
                approximate_inference=True,
                precision_audit=audit,
            )
            output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
