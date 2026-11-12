"""Correctness-only replay of private real K activations, not a timing probe."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cpufast_fft import FFTConv1d
from cpufast_tap import TapConv1d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_layers(state, kernel, dilations):
    layers = []
    for index, dilation in enumerate(dilations):
        pair = []
        for name, d in (("convs1", dilation), ("convs2", 1)):
            weight, bias = (
                state[f"{name}.{index}.weight"],
                state[f"{name}.{index}.bias"],
            )
            conv = torch.nn.Conv1d(
                weight.shape[1],
                weight.shape[0],
                kernel,
                padding=d * (kernel - 1) // 2,
                dilation=d,
            ).eval()
            conv.weight.copy_(weight)
            conv.bias.copy_(bias)
            pair.append(conv)
        layers.append(pair)
    return layers


def execute(layers, value, native):
    for first, second in layers:
        if native:
            value = second(
                first(F.leaky_relu(value, 0.1), activate=True), residual=value
            )
        else:
            value = second(F.leaky_relu(first(F.leaky_relu(value, 0.1)), 0.1)) + value
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["tap", "fft"], default="tap")
    parser.add_argument("--n-fft", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.backend == "fft":
        import cpufast_cpp_ops  # noqa: F401
    if args.output.exists():
        parser.error("Fresh output required")
    manifest = json.loads((args.replay / "manifest.json").read_text())
    if manifest.get("parent_parameters_unchanged") is not True:
        parser.error("Immutable K capture audit required")
    trained = torch.load(args.candidate, map_location="cpu", weights_only=True)
    spec = trained["student_spec"]
    report = {
        "scope": "Local real-activation correctness only; no timing/performance/voice-quality claim",
        "native_backend": args.backend,
        "approved_model_sha256": manifest["decoder_sha256"],
        "capture_manifest_sha256": sha(args.replay / "manifest.json"),
        "candidate_checkpoint_sha256": sha(args.candidate),
        "candidate_quality_approved": False,
        "cases": [],
    }
    before_files = {
        row["file"]: sha(args.replay / row["file"]) for row in manifest["captures"]
    }
    with torch.no_grad():
        for row in manifest["captures"]:
            if before_files[row["file"]] != row["sha256"]:
                raise RuntimeError("Capture checksum changed")
            capture = torch.load(
                args.replay / row["file"], map_location="cpu", weights_only=True
            )
            value = capture["input"]
            original_input = value.clone()
            stage, branch = capture["stage"], capture["original_branch"]
            index = sum(len(r) for r in spec["decoder_branches"][:stage]) + spec[
                "decoder_branches"
            ][stage].index(branch)
            prefix = f"resblocks.{index}."
            candidate = {
                key[len(prefix) :]: tensor
                for key, tensor in trained["state_dict"].items()
                if key.startswith(prefix)
            }
            selected = {
                r["pair"]
                for r in spec["decoder_bottlenecks"]
                if r["stage"] == stage and r["original_branch"] == branch
            }
            for name, tensor in capture["state_dict"].items():
                changed = (
                    name.split(".")[0] in ("convs1", "convs2")
                    and int(name.split(".")[1]) in selected
                )
                if not changed and not torch.equal(tensor, candidate[name]):
                    raise RuntimeError("Unselected K coefficient changed")
            for kind, state in (
                ("approved_parent", capture["state_dict"]),
                ("unapproved_pair_init", candidate),
            ):
                raw = make_layers(state, capture["kernel"], capture["dilation"])
                native = []
                for pair in raw:
                    native_pair = []
                    for conv in pair:
                        layer = FFTConv1d(conv, backend="packed", n_fft=args.n_fft)
                        if args.backend == "tap":
                            layer = TapConv1d(layer)
                        else:
                            layer.output_layout = layer.fused_finish = True
                            layer.stream_products = layer.parallel_transforms = True
                        native_pair.append(layer)
                    native.append(native_pair)
                parameters = [p for pair in raw for c in pair for p in c.parameters()]
                snapshots = [p.clone() for p in parameters]
                expected = execute(raw, value, False)
                errors = []
                for threads in (1, 2):
                    torch.set_num_threads(threads)
                    actual = execute(native, value, True)
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                    errors.append(
                        {
                            "threads": threads,
                            "max_abs_error": float((actual - expected).abs().max()),
                            "finite": bool(torch.isfinite(actual).all()),
                        }
                    )
                unchanged = all(
                    torch.equal(a, b) for a, b in zip(parameters, snapshots)
                ) and torch.equal(value, original_input)
                if not unchanged:
                    raise RuntimeError("Replay inputs/parameters changed")
                report["cases"].append(
                    {
                        "capture": row["file"],
                        "kind": kind,
                        "input_shape": list(value.shape),
                        "output_shape": list(actual.shape),
                        "kernel": capture["kernel"],
                        "dilations": list(capture["dilation"]),
                        "pair_weight_shapes": [
                            [list(c.weight.shape) for c in pair] for pair in raw
                        ],
                        "checks": errors,
                        "parameters_and_inputs_unchanged": unchanged,
                    }
                )
    report["files_unchanged"] = before_files == {
        row["file"]: sha(args.replay / row["file"]) for row in manifest["captures"]
    }
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
