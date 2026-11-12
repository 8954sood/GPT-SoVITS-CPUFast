"""Training-only internal-pair sensitivity and actual-N1 spectral MAC budgets.

No student is initialized or trained. Every candidate masks a single Conv2
input-channel subset on immutable parent activations. These local proxies are
not sequential-model, perceptual, validation-set or measured-speed evidence.
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from cpufast_optimizations import parameter_digest
from experiments.rtf07.capture_block_input import Captured
from experiments.rtf07.replay_rectangular_block import sha
from experiments.student.distill_decoder import decoder_from_teacher
from student_runtime import StudentGenerator

TARGETS = ((0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def budget(channels, keep, kernel, dilation, time, n_fft):
    blocks = []
    for d in (dilation, 1):
        effective = d * (kernel - 1) + 1
        blocks.append(math.ceil((time + d * (kernel - 1)) / (n_fft - effective + 1)))
    full = 3 * (n_fft // 2 + 1) * sum(blocks) * channels * channels
    return {
        "actual_n1_input_time": time,
        "n_fft": n_fft,
        "blocks": blocks,
        "hops": [n_fft - dilation * (kernel - 1), n_fft - (kernel - 1)],
        "spectral_MAC_before": full,
        "spectral_MAC_after": full * keep // channels,
        "spectral_MAC_saved": full * (channels - keep) // channels,
        "cache_bytes_saved": 2
        * 3
        * (n_fft // 2 + 1)
        * channels
        * (channels - keep)
        * 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("approved", "s2", "corpus", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.875, 0.75])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh nonmodel report required")
    if len(set(args.fractions)) != len(args.fractions) or any(
        not 0 < fraction < 1 for fraction in args.fractions
    ):
        parser.error(
            "Distinct internal keep fractions strictly between zero and one required"
        )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    release = json.loads((args.approved / "manifest.json").read_text())
    if not release["quality_approved"]:
        raise ValueError("Approved immutable parent required")
    for name, info in release["files"].items():
        if sha(args.approved / name) != info["sha256"]:
            raise ValueError("Approved parent asset changed")
    corpus = json.loads(args.corpus.read_text())
    if not corpus["collection_complete"] or not corpus["teacher_unchanged"]:
        raise ValueError("Completed immutable training corpus required")
    selected = []
    for language in ("ja", "ko", "en"):
        for origin in ("A_", "B_"):
            selected.append(
                next(
                    r
                    for r in corpus["records"]
                    if r["split"] == "train"
                    and r["language"] == language
                    and r["id"].startswith(origin)
                )
            )
    spec = json.loads((args.approved / "student_spec.json").read_text())
    teacher = decoder_from_teacher(args.s2, args.corpus.parent / "teacher_decoder.pt")
    parent = (
        StudentGenerator(
            teacher,
            spec["decoder_branches"],
            bottlenecks=spec.get("decoder_bottlenecks"),
        )
        .eval()
        .requires_grad_(False)
    )
    parent.load_state_dict(
        torch.load(args.approved / "decoder.pt", weights_only=True)["state_dict"],
        strict=True,
    )
    before = parameter_digest(parent)
    profile = json.loads(args.profile.read_text())
    modules = dict(parent.named_modules())
    fft_meta = {r["module"]: r for r in profile["decoder_fft"]}
    events = {r["name"]: r for r in profile["operators"]}
    range_counts = {}
    for row in fft_meta.values():
        range_counts[row["range"]] = range_counts.get(row["range"], 0) + 1
    pairs = {}
    for stage, branch in TARGETS:
        index = parent.stage_offsets[stage] + parent.stage_branches[stage].index(branch)
        for pair in range(3):
            key = f"resblocks.{index}.convs2.{pair}"
            first = f"resblocks.{index}.convs1.{pair}"
            meta = fft_meta[first]
            channels = modules[key].in_channels
            if channels != modules[key].out_channels:
                raise ValueError("Only full-width measured FFT pairs selected")
            proxy = sum(
                events[fft_meta[name]["range"]]["total_cpu_us"]
                / 1e6
                / range_counts[fft_meta[name]["range"]]
                for name in (first, key)
            )
            pairs[key] = {
                "stage": stage,
                "original_branch": branch,
                "pair": pair,
                "channels": channels,
                "kernel": modules[key].kernel_size[0],
                "dilation": modules[first].dilation[0],
                "time": meta["input_shapes"][0]["shape"][-1],
                "n_fft": meta["n_fft"],
                "scope_proxy": proxy,
                "energy": torch.zeros(channels, dtype=torch.float64),
            }
    captured = {}
    block_inputs = {}
    hooks = []
    for stage, branch in TARGETS:
        index = parent.stage_offsets[stage] + parent.stage_branches[stage].index(branch)
        block_key = f"resblocks.{index}"

        def before_block(_, inputs, key=block_key):
            block_inputs[key] = inputs[0].detach().clone()

        hooks.append(modules[block_key].register_forward_pre_hook(before_block))
    for key in pairs:

        def capture(_, inputs, output, key=key):
            captured[key] = (inputs[0].detach().clone(), output.detach().clone())

        hooks.append(modules[key].register_forward_hook(capture))
    last = parent.stage_offsets[2] + parent.stage_branches[2].index(2)

    def stop(*_):
        raise Captured

    hooks.append(parent.resblocks[last].register_forward_hook(stop))
    report = {
        "scope": __doc__,
        "approved_parent_decoder_sha256": sha(args.approved / "decoder.pt"),
        "parent_spec_sha256": sha(args.approved / "student_spec.json"),
        "training_corpus_sha256": sha(args.corpus),
        "n1_profile_sha256": sha(args.profile),
        "threads": 2,
        "heldout_or_final_evaluation_used": False,
        "student_initialized": False,
        "training_performed": False,
        "quality_approved": False,
        "actual_n1_candidate_measured": False,
        "contexts": [],
        "choices": [],
    }
    try:
        # Store only private temporary activations; automatically remove on success/error.
        with (
            tempfile.TemporaryDirectory(
                prefix="pair_rank_", dir=args.output.parent
            ) as temporary,
            torch.no_grad(),
        ):
            temporary = Path(temporary)
            for row in selected:
                path = args.corpus.parent / row["decoder_context"]
                if sha(path) != row["decoder_context_sha256"]:
                    raise ValueError("Training context checksum changed")
                context = torch.load(path, map_location="cpu", weights_only=True)
                captured.clear()
                block_inputs.clear()
                try:
                    parent(context["latent"], context["conditioning"])
                except Captured:
                    pass
                if set(captured) != set(pairs):
                    raise RuntimeError("Required measured FFT activations missing")
                for key, (hidden, _) in captured.items():
                    # Equal weight per A/B-language context despite different audio lengths.
                    pairs[key]["energy"] += hidden.double().square().mean(dim=(0, 2))
                torch.save(
                    {"pairs": captured, "block_inputs": block_inputs},
                    temporary / (row["id"] + ".pt"),
                )
                report["contexts"].append(
                    {
                        k: row[k]
                        for k in (
                            "id",
                            "split",
                            "language",
                            "latent_origin",
                            "decoder_context_sha256",
                        )
                    }
                )
                print("Captured train", row["id"], flush=True)
            for key, info in pairs.items():
                conv = modules[key]
                score = (
                    info["energy"]
                    / len(selected)
                    * conv.weight.double().square().sum(dim=(0, 2))
                )
                ranking = torch.argsort(score, descending=True, stable=True)
                for fraction in args.fractions:
                    count = int(info["channels"] * fraction)
                    if count % 12:
                        raise ValueError("Candidate native output rank must divide12")
                    keep = sorted(ranking[:count].tolist())
                    removed = sorted(ranking[count:].tolist())
                    choice = {k: v for k, v in info.items() if k != "energy"}
                    choice.update(
                        module=key,
                        keep_fraction=fraction,
                        inner_channels=count,
                        parent_fft_size_may_require_explicit_override=(
                            info["channels"] >= 256 and count < 256
                        ),
                        conv2_may_switch_to_tap_backend=count <= 48,
                        keep=keep,
                        removed_score_fraction=float(
                            score[removed].sum() / score.sum()
                        ),
                        budget=budget(
                            info["channels"],
                            count,
                            info["kernel"],
                            info["dilation"],
                            info["time"],
                            info["n_fft"],
                        ),
                        linear_scope_saving_proxy_seconds=info["scope_proxy"]
                        * (1 - fraction),
                        contexts=[],
                    )
                    report["choices"].append(choice)
            # Read each private activation file once, not once per option.
            for row in selected:
                data = torch.load(temporary / (row["id"] + ".pt"), weights_only=True)
                for choice in report["choices"]:
                    key = choice["module"]
                    info = pairs[key]
                    conv = modules[key]
                    hidden, original = data["pairs"][key]
                    removed = sorted(set(range(info["channels"])) - set(choice["keep"]))
                    delta = F.conv1d(
                        hidden[:, removed].contiguous(),
                        conv.weight[:, removed].contiguous(),
                        None,
                        padding=conv.padding,
                        dilation=conv.dilation,
                    )
                    block_key = key.split(".convs2.")[0]
                    residual = data["block_inputs"][block_key]
                    for previous in range(info["pair"] + 1):
                        residual = (
                            residual
                            + data["pairs"][f"{block_key}.convs2.{previous}"][1]
                        )
                    error = delta.double().square().sum()
                    choice["contexts"].append(
                        {
                            "id": row["id"],
                            "language": row["language"],
                            "input_context_sha256": row["decoder_context_sha256"],
                            "conv2_relative_RMSE": math.sqrt(
                                float(
                                    error
                                    / original.double().square().sum().clamp_min(1e-30)
                                )
                            ),
                            "residual_pair_relative_RMSE": math.sqrt(
                                float(
                                    error
                                    / residual.double().square().sum().clamp_min(1e-30)
                                )
                            ),
                        }
                    )
                print("Evaluated train", row["id"], flush=True)
            for choice in report["choices"]:
                errors = [r["residual_pair_relative_RMSE"] for r in choice["contexts"]]
                choice["macro_residual_pair_relative_RMSE"] = sum(errors) / len(errors)
                choice["relative_pair_output_RMSE"] = choice[
                    "macro_residual_pair_relative_RMSE"
                ]
                choice["outer_channels"] = choice["channels"]
                choice["baseline_spectral_MACs"] = choice["budget"][
                    "spectral_MAC_before"
                ]
                choice["removed_spectral_MACs"] = choice["budget"]["spectral_MAC_saved"]
                choice["worst_residual_pair_relative_RMSE"] = max(errors)
                choice["proxy_RMSE_per_saved_GMAC"] = choice[
                    "relative_pair_output_RMSE"
                ] / (choice["removed_spectral_MACs"] / 1e9)
    finally:
        for hook in hooks:
            hook.remove()
    if parameter_digest(parent) != before:
        raise RuntimeError("Parent parameters changed")
    report["parent_parameters_unchanged"] = True
    report["choices"].sort(key=lambda row: row["proxy_RMSE_per_saved_GMAC"])
    for rank, row in enumerate(report["choices"], 1):
        row["rank_by_local_error_per_saved_spectral_MAC"] = rank
    report["limitations"] = [
        "Training-only ranking on six balanced A/B-language contexts; no heldout/perceptual proof.",
        "Single-pair raw-slice delta excludes ridge folding, recovery and downstream nonlinear effects.",
        "Combined-pair errors can interact; local RMSE and removed energy cannot be summed into full-voice quality.",
        "Spectral MAC budgets use actual N1 shapes; diagnostic scope estimates assume equal cost within grouped geometries and linear scaling, not measured speed.",
        "No model checkpoint or learned parameter is emitted.",
    ]
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
