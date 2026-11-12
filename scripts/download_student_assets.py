"""Download only an approved private HF bundle pinned to its full commit SHA."""

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

from cpufast_student_assets import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    repo = os.environ.get("HF_STUDENT_REPO", "")
    revision = os.environ.get("HF_STUDENT_REVISION", "")
    filename = os.environ.get("HF_STUDENT_MANIFEST_PATH", "manifest.json")
    expected = os.environ.get("HF_STUDENT_MANIFEST_SHA256", "")
    token = os.environ.get("HF_STUDENT_TOKEN", "")
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not token
    ):
        parser.error("Private repo, full immutable HF revision and token required")
    remote = PurePosixPath(filename)
    if (
        remote.is_absolute()
        or any(p in (".", "..") for p in remote.parts)
        or remote.name != "manifest.json"
    ):
        parser.error("Relative approved manifest path required")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    disable_progress_bars()
    # Signed storage URLs can contain temporary credentials; never log retries/URLs.
    logging.disable(logging.CRITICAL)

    def fetch(name, target):
        source = hf_hub_download(
            repo_id=repo,
            filename=str(name),
            revision=revision,
            token=token,
            cache_dir=args.output / ".hf-cache",
        )
        shutil.copyfile(source, target)
        target.chmod(0o600)

    fetch(remote, args.output / "manifest.json")
    data = load_manifest(args.output / "manifest.json", expected, check_files=False)
    if not args.manifest_only:
        for name in data["files"]:
            fetch(remote.parent / name, args.output / name)
        load_manifest(args.output / "manifest.json", expected)
    print("Approved private student manifest/assets verified", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        print(
            f"Private asset download failed: type={type(error).__name__}, status={status}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
