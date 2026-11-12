"""Export a separate student copy with FP16-representable AR matrix weights.

Only four linear matrices per AR block and the predictor matrix are rounded.
Biases, normalization, embeddings, decoder and source checkpoint stay unchanged.
Native inference reads a half-sized cache and computes in FP32. This export is
lossy relative to the parent student and requires separate listening approval.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

MATRIX = re.compile(
    r"model\.h\.layers\.\d+\.(?:self_attn\.in_proj_weight|"
    r"self_attn\.out_proj\.weight|linear[12]\.weight)$"
)


def round_matrices(checkpoint):
    result = {**checkpoint, "weight": dict(checkpoint["weight"])}
    audit = []
    for name, value in checkpoint["weight"].items():
        if not (MATRIX.fullmatch(name) or name == "model.ar_predict_layer.weight"):
            continue
        rounded = value.half().float()
        if not torch.isfinite(rounded).all():
            raise ValueError(f"FP16 conversion overflow: {name}")
        error = rounded - value.float()
        result["weight"][name] = rounded
        audit.append(
            {
                "name": name,
                "elements": value.numel(),
                "changed_elements": int((rounded != value).sum()),
                "max_abs_error": float(error.abs().max()),
                "rms_error": float(error.square().mean().sqrt()),
            }
        )
    expected = 4 * checkpoint["config"]["model"]["n_layer"] + 1
    if len(audit) != expected:
        raise ValueError(f"Expected {expected} AR matrices, found {len(audit)}")
    result["student_metadata"] = {
        **checkpoint.get("student_metadata", {}),
        "quality_approved": False,
        "inference_conversion": "FP16-representable linear matrix copy; FP32 native arithmetic",
    }
    return result, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        parser.error("Fresh output required; source may not be overwritten")
    torch.set_num_threads(2)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = digest(args.source)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    converted, audit = round_matrices(checkpoint)
    converted["student_metadata"]["parent_sha256"] = before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    if digest(args.source) != before:
        raise RuntimeError("Source checkpoint changed")
    report = {
        "parent_sha256": before,
        "output_sha256": digest(args.output),
        "quality_approved": False,
        "lossy_conversion_relative_to_parent": True,
        "compute_dtype": "FP32",
        "native_ar_cache_dtype": "FP16",
        "unchanged": "All other checkpoint tensor values, including bias/norm/embedding",
        "matrices": audit,
    }
    args.output.with_suffix(".conversion.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("converted", len(audit), "matrices; original source unchanged", flush=True)


if __name__ == "__main__":
    main()
