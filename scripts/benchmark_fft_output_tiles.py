"""Explicit FP32 spectral locality benchmark wrapper; baseline engine unchanged."""

import argparse
import json
import sys
from pathlib import Path

import torch

from experiments.rtf07.bench_fft_output_tiles import load_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--fft-output-tile-library", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    manifest = load_probe(args.fft_output_tile_library)
    original = torch.ops.cpufast_cpp.fft_packed_stream
    calls = 0
    shapes = set()

    def output_tile4(x, w, batch, blocks):
        nonlocal calls
        calls += 1
        shapes.add((tuple(x.shape), tuple(w.shape), batch, blocks))
        return torch.ops.cpufast_cpp.fft_packed_output_tiles(
            x, w, batch, blocks, min(4, w.shape[1])
        )

    import benchmark_cpufast

    previous = sys.argv
    torch.ops.cpufast_cpp.fft_packed_stream = output_tile4
    try:
        sys.argv = [previous[0], *remaining]
        benchmark_cpufast.main()
    finally:
        torch.ops.cpufast_cpp.fft_packed_stream = original
        sys.argv = previous
    if calls == 0:
        raise RuntimeError("No spectral output-tiling calls were observed")
    report = json.loads(output.read_text())
    report["fft_output_tiles"] = {
        "scope": "Same FP32 ACL8x12 K accumulation and complex order; output groups tiled4",
        "calls": calls,
        "shapes": sorted(shapes),
        "library_manifest": manifest,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
