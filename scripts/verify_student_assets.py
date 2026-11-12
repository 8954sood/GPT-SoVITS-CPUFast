"""Verify private execution assets without printing paths, tokens or model data."""

import argparse
from pathlib import Path
from cpufast_student_assets import load_manifest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directory", type=Path)
p.add_argument("sha256")
a = p.parse_args()
load_manifest(a.directory / "manifest.json", a.sha256)
print("Approved private execution assets verified")
