"""Listen to a separate channel-pruned decoder using approved L AR execution.

The only extra engine option pads disposable FFT output caches with zeros and
narrows them back to real channels; trained decoder parameters are not padded.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    import cpufast_engine

    original = cpufast_engine.optimize
    audit = {}

    def padded(pipeline, **options):
        runtime = original(pipeline, fft_pad_channels=True, **options)
        audit.update(
            {
                k: runtime.audit[k]
                for k in ("fft_pad_channels", "fft_padded_output_cache")
            }
        )
        return runtime

    cpufast_engine.optimize = padded
    try:
        from experiments.student.listen_ar_int8 import main as listen

        listen()
    finally:
        cpufast_engine.optimize = original
        path = args.output / "comparison.json"
        if path.exists():
            report = json.loads(path.read_text())
            report.update(
                candidate="N: approved L AR execution with decoder640 channel candidate",
                quality_approved=False,
                fft_output_padding_audit=audit,
            )
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
