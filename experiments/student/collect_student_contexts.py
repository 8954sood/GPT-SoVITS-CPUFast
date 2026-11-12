"""Collect B latents with approved L AR, retaining the original V2 decoder teacher.

AR contexts/logits in this collection come from approved AR18, not original AR24;
they are explicitly excluded from AR distillation. Decoder training recomputes
original V2 teacher targets on exactly each captured latent/conditioning pair.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    approval = json.loads(
        (root / "results/student_pilot/approved_l_runtime.json").read_text()
    )
    if not approval["quality_approved"]:
        raise ValueError("Approved L AR required")
    release = json.loads((args.s1.parent / "manifest.json").read_text())
    if (
        hashlib.sha256(args.s1.read_bytes()).hexdigest()
        != release["files"]["ar.pt"]["sha256"]
    ):
        raise ValueError("Approved AR file changed")
    if (
        hashlib.sha256((args.s1.parent / "manifest.json").read_bytes()).hexdigest()
        != approval["underlying_manifest_sha256"]
    ):
        raise ValueError("L approval does not identify this immutable bundle")
    os.environ.setdefault(
        "CPUFAST_FAST_LIBRARY", str(root / ".experiments/libcpufast_fast.so")
    )
    import cpufast_engine

    original = cpufast_engine.optimize
    audit = {}

    def approved_ar(pipeline, **options):
        authoritative = pipeline.t2s_model.model.t2s_transformer
        runtime = original(pipeline, **options)
        import cpufast_fast_ops  # noqa: F401
        from cpufast_fast_ar import build

        approximate, details = build(authoritative, head_major=True)
        pipeline.t2s_model.model.t2s_transformer = approximate
        audit.update(details)
        return runtime

    cpufast_engine.optimize = approved_ar
    try:
        from experiments.student.collect_contexts import main as collect

        collect()
    finally:
        cpufast_engine.optimize = original
    path = args.output / "manifest.json"
    report = json.loads(path.read_text())
    report.update(
        scope="B: approved L AR free-generation latents; original V2ProPlus decoder targets on identical latent and conditioning; NOT an original AR teacher corpus",
        latent_origin="approved_L_free_generation",
        ar_distillation_allowed=False,
        approved_ar_sha256=hashlib.sha256(args.s1.read_bytes()).hexdigest(),
        ar_arithmetic="W8A8 decode/native FP32 head attention; existing FP32 prefill",
        ar_runtime_audit=audit,
    )
    for record in report["records"]:
        record["latent_origin"] = "approved_L_free_generation"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
