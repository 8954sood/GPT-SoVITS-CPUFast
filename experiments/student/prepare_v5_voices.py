"""Render existing original-V5 teacher mel trajectories as attributed voice data."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output directory required")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.v5_root))
    from ort_vocoder import load_torch_vocoder

    model = load_torch_vocoder(args.v5_root / "models").requires_grad_(False).eval()
    corpus = json.loads((args.corpus / "corpus.json").read_text())
    args.output.mkdir(parents=True)
    result = {
        "source": "Existing original V5 Turbo FP32 teacher trajectory + original V5 vocoder",
        "vocoder_sha256": sha(args.v5_root / "models/vocoder.pth"),
        "use": "Synthetic voice references for V2ProPlus teacher/student; V5 latent is not substituted into V2 decoder",
        "voices": [],
    }
    with torch.no_grad():
        for speaker in ("reference",):
            record = next(
                r
                for r in corpus["records"]
                if str(r.get("speaker")) == speaker
                and r["lang"] == "ja"
                and r["split"] == "train"
                and r.get("noise_variant", 0) == 0
            )
            path = args.corpus / record["file"]
            example = torch.load(path, weights_only=True, map_location="cpu")
            mel = (
                example["target"][..., example["prompt"].shape[-1] :] + 1
            ) / 2 * 14 - 12
            audio = model(mel)[0, 0].numpy()
            name = f"v5_{speaker}.wav"
            expected_seconds = record["teacher_timing"]["audio_seconds"]
            if abs(len(audio) / 48000 - expected_seconds) > 0.02:
                raise ValueError(
                    "Cached V5 trajectory is cropped; transcript alignment not guaranteed"
                )
            sf.write(args.output / name, audio, 48000)
            row = {
                "speaker": speaker,
                "path": name,
                "text": record["text"],
                "language": record["lang"],
                "seconds": len(audio) / 48000,
                "source_trajectory": record["file"],
                "trajectory_sha256": sha(path),
                "source_voice": "User-provided reference voice, V5-generated utterance; local use only",
                "waveform_sha256": sha(args.output / name),
            }
            result["voices"].append(row)
            print(row, flush=True)
    public_root = args.v5_root / "data/distill_voices"
    public = json.loads((public_root / "references.json").read_text())
    for speaker in ("2035", "2277"):
        record = next(r for r in public["records"] if r["speaker"] == speaker)
        source = public_root / record["file"]
        shutil.copyfile(source, args.output / source.name)
        result["voices"].append(
            {
                "speaker": speaker,
                "path": source.name,
                "text": record["text"],
                "language": "en",
                "seconds": record["seconds"],
                "source_voice": public["attribution"],
                "waveform_sha256": sha(source),
                "kind": "original recorded voice from existing V5 project data",
            }
        )
    result["excluded"] = (
        "Public-speaker cached V5 trajectories are cropped; not used as transcript-aligned references."
    )
    (args.output / "voices.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
