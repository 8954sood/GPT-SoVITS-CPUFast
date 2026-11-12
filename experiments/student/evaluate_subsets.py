"""Score explicit AR layer subsets on saved teacher-forced pilot contexts."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from experiments.student.analyze_teacher import compare_logits, logits_for
from student_runtime import ar_subset, export_ar_student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use fresh output")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from AR.models.t2s_lightning_module import Text2SemanticLightningModule

    saved = torch.load(args.s1, map_location="cpu", weights_only=False, mmap=True)
    model = Text2SemanticLightningModule(
        saved["config"],
        "student-pilot",
        is_train=False,
        build_t2s_transformer=False,
        build_h_module=False,
    )
    model.load_inference_only_state_dict(saved["weight"])
    model.eval()
    teacher = model.model
    stock = teacher.t2s_transformer
    report = json.loads((args.analysis / "analysis.json").read_text())
    scores = []
    for layer in range(len(stock.blocks)):
        scores.append(
            (
                layer,
                statistics.mean(
                    c["ar_deletions"][layer]["mean_token_kl"] for c in report["cases"]
                ),
            )
        )
    ranked = [
        i
        for i, _ in sorted(scores, key=lambda pair: pair[1])
        if i not in (0, len(stock.blocks) - 1)
    ]
    result = {
        "scope": "Same-context subset selection; not heldout quality or deployment RTF",
        "candidates": [],
    }
    with torch.no_grad():
        for layers in (20, 16, 12):
            removed = ranked[: len(stock.blocks) - layers]
            keep = [i for i in range(len(stock.blocks)) if i not in removed]
            candidate = ar_subset(stock, keep)
            rows = []
            for case in report["cases"]:
                ctx = torch.load(
                    args.analysis / f"{case['language']}_ar_context.pt",
                    weights_only=True,
                )
                logits = logits_for(
                    candidate,
                    ctx["inputs"],
                    ctx["mask"],
                    ctx["prediction_start"],
                    teacher.ar_predict_layer,
                )
                rows.append(
                    {
                        "language": case["language"],
                        **compare_logits(ctx["teacher_logits"], logits, teacher.EOS),
                    }
                )
            result["candidates"].append(
                {
                    "layers": layers,
                    "removed_layer_ids": removed,
                    "keep_layer_ids": keep,
                    "teacher_forcing": rows,
                }
            )
            print(layers, rows, flush=True)
    selected = result["candidates"][0]
    spec = {
        "version": 1,
        "training": "none",
        "quality_approved": False,
        "ar_keep": selected["keep_layer_ids"],
        "decoder_branches": [[0, 1, 2], [0, 1, 2], [0, 2], [1, 2], [0, 1, 2]],
        "selection_scope": "3 languages, one reference; no heldout generalization evidence",
    }
    (args.output / "student_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    export_ar_student(
        args.s1, args.output / "ar20_untrained.ckpt", selected["keep_layer_ids"]
    )
    (args.output / "subset_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
