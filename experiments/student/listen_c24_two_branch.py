"""Quality-only probe for approved AR17 plus a separate final-C24 decoder candidate."""

import argparse
import json
import sys
from pathlib import Path

from cpufast_student_assets import load_manifest, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    for name in ("approved", "s1", "s2", "spec", "decoder-student", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-sha256", required=True)
    parser.add_argument("--variant", choices=("D",), default="D")
    args, remaining = parser.parse_known_args()
    release = load_manifest(args.approved / "manifest.json", args.approved_sha256)
    if sha256(args.s1) != release["files"]["ar.pt"]["sha256"]:
        raise ValueError("Listening must retain the approved AR file exactly")
    if sha256(args.s2) != release["original_s2_sha256"]:
        raise ValueError("Listening requires the same original S2 teacher")
    before = sha256(args.decoder_student), sha256(args.spec)
    parent = json.loads((args.approved / "student_spec.json").read_text())
    spec = json.loads(args.spec.read_text())
    if len(parent["ar_keep"]) != 17 or spec["ar_keep"] != parent["ar_keep"]:
        raise ValueError("Same approved AR17 layer identities required")
    expected = [list(row) for row in parent["decoder_branches"]]
    if len(expected) != 5 or expected[-1] != [0, 1, 2]:
        raise ValueError("Expected the approved three-branch final C24 stage")
    expected[-1] = [0, 1]
    if spec["decoder_branches"] != expected:
        raise ValueError("Only the final C24 K11 branch may be removed")
    for key in ("decoder_bottlenecks", "decoder_channel_keep"):
        if spec.get(key) != parent.get(key):
            raise ValueError("Other decoder geometry must retain the approved layout")
    import cpufast_engine
    from cpufast_fft import FFTConv1d
    original_optimize = cpufast_engine.optimize
    runtime_audit = {}
    def canonical_optimize(pipeline, **options):
        options.update(fft_stream=True, parallel_transforms=True)
        runtime = original_optimize(pipeline, **options)
        geometry = []
        for name, layer in pipeline.vits_model.dec.named_modules():
            if isinstance(layer, FFTConv1d):
                expected_fft = 128 if layer.weight.shape[1] >= 256 else 256
                if layer.n_fft != expected_fft or layer.fast_half or layer.half_stream_products:
                    raise RuntimeError("C24 quality must use canonical original FFT policy")
                geometry.append({"module": name, "weight_shape": list(layer.weight.shape),
                                 "n_fft": layer.n_fft, "dilation": layer.dilation,
                                 "stream_products": layer.stream_products,
                                 "parallel_transforms": layer.parallel_transforms})
        if not runtime.audit["fft_stream"] or not runtime.audit["parallel_transforms"]:
            raise RuntimeError("C24 quality must execute canonical streamed parallel FP32 FFT")
        runtime_audit.update(fft_stream=runtime.audit["fft_stream"],
                             parallel_transforms=runtime.audit["parallel_transforms"],
                             fft_geometry=geometry, fft128_cin240_substitution=False)
        return runtime
    cpufast_engine.optimize = canonical_optimize
    previous = sys.argv
    try:
        from experiments.student.listen_ar_int8 import main as listen

        sys.argv = [previous[0], "--output", str(args.output), "--s1", str(args.s1),
                    "--s2", str(args.s2),
                    "--spec", str(args.spec), "--decoder-student", str(args.decoder_student),
                    "--variant", "D", *remaining]
        listen()
    finally:
        sys.argv = previous
        cpufast_engine.optimize = original_optimize
    load_manifest(args.approved / "manifest.json", args.approved_sha256)
    if before != (sha256(args.decoder_student), sha256(args.spec)):
        raise RuntimeError("Candidate checkpoint or spec changed while generating audio")
    path = args.output / "comparison.json"
    report = json.loads(path.read_text())
    report.update(
        candidate="Approved AR17 with unapproved final-C24 two-branch decoder",
        approved_parent_manifest_sha256=args.approved_sha256,
        candidate_decoder_sha256=before[0],
        candidate_spec_sha256=before[1],
        candidate_checkpoint_and_spec_unchanged=True,
        original_ar_layer_ids=spec["ar_keep"],
        ar_file_equal_approved=True,
        parent_decoder_branches=parent["decoder_branches"],
        candidate_decoder_branches=spec["decoder_branches"],
        runtime_audit=runtime_audit,
        additional_branch_reduction=True,
        additional_ar_depth_reduction=False,
        quality_approved=False,
        actual_n1_benchmark=False,
        timing_scope="Quality-generation observations only; not an RTF benchmark",
    )
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
