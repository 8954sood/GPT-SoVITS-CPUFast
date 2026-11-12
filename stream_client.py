"""Register/list/delete durable styles or consume streaming PCM into a finalized WAV."""

import argparse
import json
import time
import wave
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9880")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("audio", type=Path)
    register.add_argument("--name", required=True)
    register.add_argument("--prompt-text", required=True)
    register.add_argument("--lang", choices=["ja", "ko", "en"], default="ja")
    register.add_argument("--start", type=float, default=0)
    register.add_argument("--end", type=float)
    commands.add_parser("styles")
    delete = commands.add_parser("delete")
    delete.add_argument("style_id")
    tts = commands.add_parser("tts")
    tts.add_argument("style_id")
    tts.add_argument("text")
    tts.add_argument("--lang", choices=["ja", "ko", "en"], default="ja")
    tts.add_argument(
        "--latency-mode", choices=["quality", "balanced", "fast"], default="fast"
    )
    tts.add_argument("--seed", type=int, default=42)
    tts.add_argument("--output", type=Path, default=Path("results/streamed.wav"))
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=180) as client:
        if args.command == "styles":
            response = client.get("/v1/styles")
            response.raise_for_status()
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        elif args.command == "delete":
            response = client.delete(f"/v1/styles/{args.style_id}")
            response.raise_for_status()
            print(json.dumps({"deleted": args.style_id}, indent=2))
        elif args.command == "register":
            data = {
                "name": args.name,
                "prompt_text": args.prompt_text,
                "prompt_lang": args.lang,
                "start": str(args.start),
            }
            if args.end is not None:
                data["end"] = str(args.end)
            with args.audio.open("rb") as stream:
                response = client.post(
                    "/v1/styles",
                    files={"audio": (args.audio.name, stream, "audio/wav")},
                    data=data,
                )
            response.raise_for_status()
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        else:
            started = time.perf_counter()
            first = None
            count = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with client.stream(
                "POST",
                "/v1/tts/stream",
                json={
                    "style_id": args.style_id,
                    "text": args.text,
                    "lang": args.lang,
                    "seed": args.seed,
                    "latency_mode": args.latency_mode,
                },
            ) as response:
                response.raise_for_status()
                if response.headers.get("x-audio-format") != "s16le":
                    raise ValueError("Unexpected stream encoding")
                sample_rate = int(response.headers["x-audio-sample-rate"])
                if not 8000 <= sample_rate <= 192000 or response.headers.get("x-audio-channels") != "1":
                    raise ValueError("Unexpected stream sample rate or channels")
                with wave.open(str(args.output), "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(sample_rate)
                    pending = b""
                    for chunk in response.iter_bytes():
                        if chunk and first is None:
                            first = time.perf_counter() - started
                        pending += chunk
                        even = len(pending) // 2 * 2
                        out.writeframesraw(pending[:even])
                        count += even
                        pending = pending[even:]
                    if pending or not count:
                        raise ValueError("Incomplete or empty PCM stream")
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "first_audio_seconds": first,
                        "total_seconds": time.perf_counter() - started,
                        "sample_rate": sample_rate,
                        "audio_seconds": count / (sample_rate * 2),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
