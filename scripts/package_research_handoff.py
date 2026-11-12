"""Package private, local-only continuation data without tokens or cached models.

Run from the research checkout after training has stopped. The archive preserves
repository-relative paths; its checksum index can be verified without extracting.
Published approved models are restored separately from immutable HF revisions.
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def stream_digest(source):
    result = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def digest(path):
    with path.open("rb") as source:
        return stream_digest(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        with tarfile.open(args.output, "r:gz") as archive:
            index = json.load(archive.extractfile("TRANSFER_INDEX.json"))
            for name, record in index["files"].items():
                source = archive.extractfile(name)
                if source is None or stream_digest(source) != record["sha256"]:
                    raise ValueError(f"Transfer checksum mismatch: {name}")
        print(f"Verified {len(index['files'])} private transfer files")
        return
    root = args.root.resolve()
    folders = [
        ".experiments/student/recovery_corpus36_retry",
        ".experiments/student/bottleneck_contexts72",
        ".experiments/student/ar_teacher_features",
        ".experiments/student/v5_voices_retry",
        ".experiments/student/listening_L",
        ".experiments/student/listening_V",
        ".experiments/student/listening_W",
    ]
    names = set()
    for folder in folders:
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        names.update(
            str(p.relative_to(root)) for p in directory.rglob("*") if p.is_file()
        )
    for name in [
        ".experiments/handoff/reference_segment.wav",
        ".experiments/student/budget40/best.pt",
        ".experiments/student/budget40/history.json",
        ".experiments/student/budget40_init/student_spec.json",
        ".experiments/student/budget40_init/initialization.json",
        ".experiments/student/budget40_host.json",
    ]:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
        names.add(name)
    index = {
        "scope": "Private local transfer; contains generated voice/context data. Do not publish to Git or upload automatically.",
        "restore_root": "GPT-SoVITS-CPUFast checkout",
        "models_restored_separately": "K/R/U immutable HF bundles; original S1/S2/HuBERT/SV; see docs/ENVIRONMENT_HANDOFF.md",
        "files": {
            name: {"bytes": (root / name).stat().st_size, "sha256": digest(root / name)}
            for name in sorted(names)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("xb") as destination,
        tarfile.open(fileobj=destination, mode="w:gz", compresslevel=1) as archive,
    ):
        for name in sorted(names):
            archive.add(root / name, arcname=name, recursive=False)
        encoded = (json.dumps(index, indent=2, ensure_ascii=False) + "\n").encode()
        info = tarfile.TarInfo("TRANSFER_INDEX.json")
        info.size = len(encoded)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(encoded))
    args.output.chmod(0o600)
    receipt = {
        "archive": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": digest(args.output),
        "files": len(names),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
