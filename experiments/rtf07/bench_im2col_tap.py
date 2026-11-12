"""N1-only probe: replace repeated tap GEMMs with explicit im2col + one SGEMM.

Includes im2col materialization in timings; authoritative weight packing is done
once for both methods. Synthetic speed is not full TTS RTF or listening approval.
The different accumulation order is explicitly measured against current taps.
"""

import argparse
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.rtf07.bench_tap_tiles import cgroup, tap_tiles


def bulk_shapes(profile=None):
    # K's actual profile attributes most tap time to these four D1 geometries.
    targets = [(48, 7), (48, 11), (24, 7), (24, 11)]
    if profile is None:
        return [(c, k, 1, 76800 if c == 48 else 153600) for c, k in targets]
    rows = json.loads(Path(profile).read_text())["decoder_taps"]
    result = []
    for channels, kernel in targets:
        lengths = {
            shape["shape"][2]
            for row in rows
            if row["weight_shape"] == [channels, channels, kernel]
            and row["dilation"] == 1
            for shape in row["input_shapes"]
            if len(shape["shape"]) == 3 and shape["shape"][:2] == [1, channels]
        }
        if len(lengths) != 1:
            raise ValueError("One actual profiled D1 length per bulk geometry required")
        result.append((channels, kernel, 1, lengths.pop()))
    return result


def tap_im2col(flat, padded, bias, length, kernel, dilation, tile=0):
    output = padded.new_empty((flat.shape[0], length))
    spread = (kernel - 1) * dilation
    width = tile or length
    for left in range(0, length, width):
        right = min(length, left + width)
        columns = F.unfold(
            padded[:, left : right + spread].unsqueeze(0).unsqueeze(2),
            kernel_size=(1, kernel),
            dilation=(1, dilation),
        ).squeeze(0)
        torch.mm(flat, columns, out=output[:, left:right])
    if bias is not None:
        output.add_(bias[:, None])
    return output.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-shapes", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.warmups <= 5
        or not 1 <= args.samples <= 20
    ):
        parser.error("Fresh output and bounded samples required")
    settings = cgroup()
    if platform.machine() != "aarch64" or settings != {
        "cpu.max": "200000 100000",
        "memory.max": "8000000000",
        "memory.swap.max": "0",
    }:
        parser.error("Actual 2CPU/8GB/no-swap ARM target required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(932)
    report = {
        "scope": "Synthetic im2col+SGEMM probe, not TTS performance/quality",
        "im2col_materialization_included_in_timing": True,
        "cgroup": settings,
        "threads": 2,
        "seed": 932,
        "shape_source": str(args.profile_shapes)
        if args.profile_shapes
        else "Approved H3/K KO duration-derived shapes",
        "rows": [],
        "complete": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with torch.inference_mode():
        for channels, kernel, dilation, length in bulk_shapes(args.profile_shapes):
            weight = torch.randn(channels, channels, kernel) / channels**0.5
            before = weight.clone()
            packed = weight.permute(2, 0, 1).contiguous()
            flat = weight.reshape(channels, -1).contiguous()
            value = torch.randn(channels, length)
            padding = dilation * (kernel - 1) // 2
            padded = F.pad(value, (padding, padding))
            bias = torch.randn(channels)
            expected = tap_tiles(packed, padded, bias, length, dilation)
            tiles = [0, 1024, 4096, 8192]
            if channels == 24:
                tiles.reverse()
            for method, tile in [("current_taps", 0)] + [
                ("im2col", size) for size in tiles
            ]:
                run = (
                    partial(tap_tiles, packed, padded, bias, length, dilation)
                    if method == "current_taps"
                    else partial(
                        tap_im2col, flat, padded, bias, length, kernel, dilation, tile
                    )
                )

                warmups = []
                for _ in range(args.warmups):
                    start = time.perf_counter()
                    actual = run()
                    warmups.append(time.perf_counter() - start)
                wall, cpu = [], []
                for _ in range(args.samples):
                    start, cpu_start = time.perf_counter(), time.process_time()
                    actual = run()
                    wall.append(time.perf_counter() - start)
                    cpu.append(time.process_time() - cpu_start)
                error = actual - expected
                row = {
                    "channels": channels,
                    "kernel": kernel,
                    "dilation": dilation,
                    "length": length,
                    "method": method,
                    "tile": tile,
                    "median_seconds": statistics.median(wall),
                    "median_cpu_seconds": statistics.median(cpu),
                    "wall_seconds": wall,
                    "cpu_seconds": cpu,
                    "warmup_seconds": warmups,
                    "exact": torch.equal(actual, expected),
                    "max_abs_error": float(error.abs().max()),
                    "rmse": float(error.square().mean().sqrt()),
                    "changed_values": int(torch.count_nonzero(error)),
                    "finite": bool(torch.isfinite(actual).all()),
                    "coefficients_unchanged": torch.equal(weight, before),
                    "estimated_im2col_buffer_bytes": 0
                    if method == "current_taps"
                    else channels * kernel * min(tile or length, length) * 4,
                }
                report["rows"].append(row)
                save()
                print(
                    f"C{channels} K{kernel} D{dilation} {method}/{tile}: "
                    f"{row['median_seconds']:.6f}s max-error={row['max_abs_error']:.3g}",
                    flush=True,
                )
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
