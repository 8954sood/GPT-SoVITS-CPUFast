"""Build the opt-in experimental NEON dot-product / FP16 library on Linux ARM64."""

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
    "native/cpufast_fast_int8.cpp",
    "native/cpufast_fast_attention.cpp",
    "native/cpufast_fast_half.cpp",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("The experimental library requires Linux ARM64")
    flags = ["-O3", "-std=c++17", "-march=armv8.2-a+fp16+dotprod", "-mtune=neoverse-n1", "-fopenmp", "-fPIC", "-shared", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    before = {name: digest(ROOT / name) for name in SOURCES}
    command = ["c++", *flags]
    for path in include_paths():
        command.extend(["-I", path])
    command.extend(str(ROOT / name) for name in SOURCES)
    command.extend(["-o", str(args.output)])
    for path in library_paths():
        command.extend(["-L", path, f"-Wl,-rpath,{path}"])
    command.extend(["-ltorch", "-ltorch_cpu", "-lc10"])
    subprocess.run(command, check=True)
    if before != {name: digest(ROOT / name) for name in SOURCES}:
        raise RuntimeError("Sources changed during build; rebuild from a stable checkout")
    manifest = {"abi": 1, "torch_version": torch.__version__, "system": platform.system(), "architecture": platform.machine(), "flags": flags, "sources": before, "library_sha256": digest(args.output)}
    manifest["head_value_schedule"] = {
        "dim32": "eight_independent_vectors_preserve_token_FMA_order",
        "other_dims": "original_four_channel_token_loop",
        "QK_softmax_masks": "unchanged",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
