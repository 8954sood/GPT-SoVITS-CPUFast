"""Load an explicitly enabled, source-checked experimental ARM library."""

import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import cpufast_cpp_ops  # Register the base namespace first.

ROOT = Path(__file__).resolve().parent
path = Path(os.environ.get("CPUFAST_FAST_LIBRARY", str(ROOT / "libcpufast_fast.so")))
if platform.system() != "Linux" or platform.machine() != "aarch64":
    raise RuntimeError("Experimental fast operators require Linux ARM64")
feature_sets = []
for line in Path("/proc/cpuinfo").read_text().splitlines():
    if line.lower().startswith("features"):
        feature_sets.append(set(line.split(":", 1)[1].split()))
features = set.intersection(*feature_sets) if feature_sets else set()
if not {"asimddp", "asimdhp"}.issubset(features):
    raise RuntimeError("Experimental fast operators require ARM dot product and FP16 SIMD")
manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
if manifest.get("abi") != 1 or manifest.get("torch_version") != torch.__version__:
    raise RuntimeError("Experimental library ABI or PyTorch version mismatch")
if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["library_sha256"]:
    raise RuntimeError("Experimental library checksum mismatch")
if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value for name, value in manifest["sources"].items()):
    raise RuntimeError("Experimental library sources do not match its build")
torch.ops.load_library(str(path))
