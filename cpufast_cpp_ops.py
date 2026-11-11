"""Load a source-checked, PyTorch-version-matched portable ARM CPU operator."""

import hashlib
import json
import os
import platform
from pathlib import Path

import torch

WORKERS = 1
ROOT = Path(__file__).resolve().parent
path = Path(os.environ.get("CPUFAST_AR_LIBRARY", str(ROOT / "libcpufast_ar_torch.so")))
manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
if platform.system() != "Linux" or platform.machine() != "aarch64":
    raise RuntimeError("Native AR operator requires Linux ARM64")
if manifest["torch_version"] != torch.__version__:
    raise RuntimeError("Native AR operator was built against another PyTorch version")
if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["library_sha256"]:
    raise RuntimeError("Native AR operator checksum mismatch")
if any(
    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
    for name, value in manifest["sources"].items()
):
    raise RuntimeError("Native AR operator sources do not match its build")
torch.ops.load_library(str(path))
