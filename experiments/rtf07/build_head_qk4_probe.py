"""Standalone head-major four-token QK scheduling probe; default libraries untouched."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    "native/cpufast_head_qk4.cpp",
]


def hashes():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-runtime", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("Use the Linux ARM64 kernel-build image")
    # Match the approved fast DSO compiler policy, including default FMA
    # contraction in the existing dequantization expression.
    flags = [
        "-O3", "-std=c++17", "-march=armv8.2-a+fp16+dotprod",
        "-mtune=neoverse-n1", "-fopenmp", "-fPIC", "-shared",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
    ]
    before = hashes()
    if args.replace_runtime:
        flags += ["-DCPUFAST_REPLACE_HEAD_QK4=1"]
    command = ["c++", *flags]
    for path in include_paths():
        command += ["-I", path]
    command += [str(ROOT / SOURCES[0]), "-o", str(args.output)]
    for path in library_paths():
        command += ["-L", path, f"-Wl,-rpath,{path}"]
    command += ["-ltorch", "-ltorch_cpu", "-lc10"]
    subprocess.run(command, check=True)
    if hashes() != before:
        raise RuntimeError("Sources changed during the research build")
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(
            {
                "abi": 1,
                "torch_version": torch.__version__,
                "architecture": platform.machine(),
                "flags": flags,
                "sources": before,
                "library_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
