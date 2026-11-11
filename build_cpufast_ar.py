"""Build a PyTorch CPU operator using the same FP32 ARM kernel without callbacks."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

ROOT = Path(__file__).resolve().parent
SOURCES = [
    "native/cpufast_ar_torch.cpp",
    "native/cpufast_ar_gemv.cpp",
    "native/cpufast_winograd_torch.cpp",
    "native/cpufast_fft_torch.cpp",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("Use Linux ARM64 kernel-build image")
    flags = [
        "-O3",
        "-std=c++17",
        "-march=armv8-a",
        "-mtune=neoverse-n1",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-fPIC",
        "-shared",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
    ]
    command = ["c++", *flags]
    for path in include_paths():
        command.extend(["-I", path])
    command.extend(
        [
            str(ROOT / SOURCES[0]),
            str(ROOT / SOURCES[2]),
            str(ROOT / SOURCES[3]),
            "-o",
            str(args.output),
        ]
    )
    for path in library_paths():
        command.extend(["-L", path, f"-Wl,-rpath,{path}"])
    command.extend(["-ltorch", "-ltorch_cpu", "-lc10"])
    subprocess.run(command, check=True)
    manifest = {
        "abi": 1,
        "torch_version": torch.__version__,
        "system": platform.system(),
        "architecture": platform.machine(),
        "flags": flags,
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        "library_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
