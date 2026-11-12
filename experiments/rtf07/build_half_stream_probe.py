"""Build a separate research library; do not change the production source list."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
build_cpufast_fast = importlib.import_module("build_cpufast_fast")


if __name__ == "__main__":
    build_cpufast_fast.SOURCES = [
        *build_cpufast_fast.SOURCES,
        "native/cpufast_fft_half_stream.cpp",
    ]
    build_cpufast_fast.main()
