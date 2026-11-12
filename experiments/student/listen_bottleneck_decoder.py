"""R listening: approved L AR, original outer decoder widths, selected internal pairs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args, _ = parser.parse_known_args()
    spec = json.loads(args.spec.read_text())
    if not spec.get("decoder_bottlenecks") or spec.get("decoder_channel_keep"):
        raise ValueError("R requires pair-only bottlenecks and original outer widths")
    from experiments.student.listen_ar_int8 import main as listen

    listen()
    path = args.output / "comparison.json"
    report = json.loads(path.read_text())
    report.update(
        candidate="R: approved L AR and normal FP32 acoustics, selected internal48→36→48 pairs",
        quality_approved=False,
        decoder_bottlenecks=spec["decoder_bottlenecks"],
        decoder_outer_widths=[768, 384, 192, 96, 48, 24],
        additional_branch_or_ar_depth_reduction=False,
    )
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
