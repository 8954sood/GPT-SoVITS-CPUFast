"""Actual V block/native FFT correctness on training inputs, without PT captures.

Reconstruct the existing V checkpoint for inference only. Capture one language's
five block inputs in memory, compare raw/fused native execution, then release
them. No latency, new student initialization, training or quality claim.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cpufast_fft import FFTConv1d, install_fft
from cpufast_optimizations import parameter_digest
from cpufast_tap import install_small_taps
from experiments.rtf07.capture_block_input import Captured
from experiments.rtf07.replay_rectangular_block import execute, make_layers, sha
from experiments.student.distill_decoder import decoder_from_teacher
from student_runtime import StudentGenerator

TARGETS = ((0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "parent", "corpus", "s2", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh small numerical JSON required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import cpufast_cpp_ops  # noqa: F401

    files = [
        args.candidate,
        args.parent / "decoder.pt",
        args.parent / "student_spec.json",
        args.corpus,
        args.corpus.parent / "teacher_decoder.pt",
    ]
    file_hashes = {str(path): sha(path) for path in files}
    candidate = torch.load(
        args.candidate, map_location="cpu", weights_only=True, mmap=True
    )
    spec = candidate["student_spec"]
    parent_state = torch.load(
        args.parent / "decoder.pt", map_location="cpu", weights_only=True, mmap=True
    )["state_dict"]
    teacher = decoder_from_teacher(args.s2, args.corpus.parent / "teacher_decoder.pt")
    model = (
        StudentGenerator(
            teacher, spec["decoder_branches"], bottlenecks=spec["decoder_bottlenecks"]
        )
        .eval()
        .requires_grad_(False)
    )
    del teacher
    model.load_state_dict(candidate["state_dict"], strict=True)
    before = parameter_digest(model)
    offsets = {
        target: model.stage_offsets[target[0]]
        + model.stage_branches[target[0]].index(target[1])
        for target in TARGETS
    }
    changed_prefixes = tuple(f"resblocks.{index}.convs" for index in offsets.values())
    for key, value in candidate["state_dict"].items():
        if not key.startswith(changed_prefixes) and not torch.equal(
            value, parent_state[key]
        ):
            raise RuntimeError(
                "Nonselected R coefficients or existing thin pairs changed"
            )
    corpus = json.loads(args.corpus.read_text())
    if not corpus["collection_complete"] or not corpus["teacher_unchanged"]:
        raise RuntimeError("Completed training context corpus required")
    report = {
        "scope": __doc__,
        "candidate_checkpoint_sha256": file_hashes[str(args.candidate)],
        "parent_decoder_sha256": file_hashes[str(args.parent / "decoder.pt")],
        "training_corpus_sha256": file_hashes[str(args.corpus)],
        "stored_activation_or_parameter_PT_files": 0,
        "nonselected_R_tensors_including_existing_thin_pairs_exact": True,
        "cases": [],
    }
    for language in ("ja", "ko", "en"):
        record = next(
            r
            for r in corpus["records"]
            if r["split"] == "train"
            and r["language"] == language
            and r["id"].startswith("A_")
        )
        path = args.corpus.parent / record["decoder_context"]
        if sha(path) != record["decoder_context_sha256"]:
            raise RuntimeError("Training context changed")
        context = torch.load(path, map_location="cpu", weights_only=True)
        captures, hooks = {}, []
        for target, index in offsets.items():

            def capture(_, inputs, target=target, captures=captures):
                captures[target] = inputs[0].detach().clone()

            hooks.append(model.resblocks[index].register_forward_pre_hook(capture))

        def stop(*_):
            raise Captured

        hooks.append(model.resblocks[offsets[(2, 2)]].register_forward_hook(stop))
        try:
            with torch.no_grad():
                model(context["latent"], context["conditioning"])
        except Captured:
            pass
        finally:
            for hook in hooks:
                hook.remove()
        if set(captures) != set(TARGETS):
            raise RuntimeError("Selected actual V block input missing")
        for target, value in captures.items():
            index = offsets[target]
            source = model.resblocks[index]
            with torch.no_grad():
                raw = make_layers(
                    source.state_dict(), source.kernel_size, source.dilation
                )
                expected = execute(raw, value, False)
                weights_before = [
                    (p, p.clone())
                    for pair in raw
                    for layer in pair
                    for p in layer.parameters()
                ]
                block = torch.nn.Module()
                block.convs1 = torch.nn.ModuleList([pair[0] for pair in raw])
                block.convs2 = torch.nn.ModuleList([pair[1] for pair in raw])
                fft_count = install_fft(block, backend="packed")
                tap_count = install_small_taps([block])
                if fft_count != 6 or tap_count != 0:
                    raise RuntimeError("V geometry changed intended FFT backend")
                native = list(zip(block.convs1, block.convs2))
                geometry = []
                for pair_index, pair in enumerate(native):
                    for side, layer in enumerate(pair):
                        if (
                            not isinstance(layer, FFTConv1d)
                            or layer.backend != "packed"
                        ):
                            raise RuntimeError("Expected native FFT, no Tap fallback")
                        expected_size = 128 if layer.weight.shape[1] >= 256 else 256
                        if layer.n_fft != expected_size:
                            raise RuntimeError("Engine FFT heuristic differs")
                        layer.output_layout = layer.fused_finish = True
                        layer.stream_products = layer.parallel_transforms = True
                        geometry.append(
                            {
                                "pair": pair_index,
                                "side": side + 1,
                                "weight_shape": list(layer.weight.shape),
                                "dilation": layer.dilation,
                                "n_fft": layer.n_fft,
                                "hop": layer.hop,
                                "backend": layer.backend,
                            }
                        )
                checks, caches = [], None
                for threads in (1, 2):
                    torch.set_num_threads(threads)
                    actual = execute(native, value, True)
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                    if actual.shape != value.shape or not torch.isfinite(actual).all():
                        raise RuntimeError("Outer residual shape/finiteness changed")
                    current_caches = [
                        (layer._key, layer._packed.data_ptr())
                        for pair in native
                        for layer in pair
                    ]
                    if caches is not None and current_caches != caches:
                        raise RuntimeError(
                            "Unchanged parameters unexpectedly repacked across threads"
                        )
                    caches = current_caches
                    checks.append(
                        {
                            "threads": threads,
                            "max_abs_error": float((actual - expected).abs().max()),
                            "finite": True,
                        }
                    )
                # Mutation is on disposable block copies only; source V stays unchanged.
                first = native[0][0]
                saved, key = first.weight[0, 0, 0].clone(), first._key
                first.weight[0, 0, 0].add_(0.03125)
                changed = first(F.leaky_relu(value, 0.1), activate=True)
                torch.testing.assert_close(
                    changed,
                    F.leaky_relu(raw[0][0](F.leaky_relu(value, 0.1)), 0.1),
                    rtol=2e-4,
                    atol=2e-4,
                )
                if first._key == key:
                    raise RuntimeError(
                        "Changed disposable coefficient did not invalidate cache"
                    )
                first.weight[0, 0, 0].copy_(saved)
                if not all(torch.equal(p, original) for p, original in weights_before):
                    raise RuntimeError("Disposable coefficient restoration failed")
                if any(
                    layer.weight is not raw[pair][side].weight
                    for pair, group in enumerate(native)
                    for side, layer in enumerate(group)
                ):
                    raise RuntimeError(
                        "Engine replaced authoritative parameter identity"
                    )
                report["cases"].append(
                    {
                        "language": language,
                        "training_context_id": record["id"],
                        "input_context_sha256": record["decoder_context_sha256"],
                        "stage": target[0],
                        "original_branch": target[1],
                        "input_shape": list(value.shape),
                        "output_shape": list(actual.shape),
                        "geometries": geometry,
                        "checks": checks,
                        "fft_modules": fft_count,
                        "tap_modules": tap_count,
                        "cache_reuse_and_mutation_invalidation": True,
                        "disposable_parameters_restored_exact": True,
                    }
                )
                del raw, native, block, expected, actual, first, changed, weights_before
            gc.collect()
        captures.clear()
        del context
        gc.collect()
        print("Completed train-only V geometry", language, flush=True)
    if parameter_digest(model) != before or file_hashes != {
        str(path): sha(path) for path in files
    }:
        raise RuntimeError("V/source checkpoint parameters or input files changed")
    report["source_model_and_files_unchanged"] = True
    report["complete"] = True
    report["quality_approved"] = False
    report["actual_N1_candidate_measured"] = False
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
