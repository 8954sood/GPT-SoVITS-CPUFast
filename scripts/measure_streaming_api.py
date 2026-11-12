"""Measure actual HTTP PCM receipt without saving generated audio or replay caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import httpx


class PCMReceipt:
    """Track wire arrival and playback starvation for native mono PCM16."""

    def __init__(self, sample_rate=32000):
        if not 8000 <= sample_rate <= 192000:
            raise ValueError("Invalid PCM sample rate")
        self.sample_rate = sample_rate
        self.bytes_per_second = sample_rate * 2
        self.bytes = 0
        self.first_seconds = None
        self.last_seconds = None
        self.initial_buffer_seconds = 0.0
        self.hash = hashlib.sha256()
        self.arrivals = []

    def receive(self, chunk, seconds):
        if not chunk:
            return
        if seconds < 0 or (self.last_seconds is not None and seconds < self.last_seconds):
            raise ValueError("PCM arrival times must be nonnegative and ordered")
        previous_samples = self.bytes // 2
        self.bytes += len(chunk)
        self.hash.update(chunk)
        self.last_seconds = seconds
        if self.first_seconds is None and self.bytes >= 2:
            self.first_seconds = seconds
        if self.first_seconds is not None:
            late = seconds - self.first_seconds - previous_samples / self.sample_rate
            self.initial_buffer_seconds = max(self.initial_buffer_seconds, late)
        self.arrivals.append({"seconds": seconds, "received_bytes": self.bytes})

    def finish(self, complete_seconds):
        if not self.bytes or self.bytes % 2 or self.first_seconds is None:
            raise ValueError("Empty or incomplete PCM16 response")
        if complete_seconds < self.last_seconds:
            raise ValueError("Response completion precedes last PCM")
        duration = self.bytes / self.bytes_per_second
        return {
            "client_first_pcm_seconds": self.first_seconds,
            "client_last_pcm_seconds": self.last_seconds,
            "client_response_complete_seconds": complete_seconds,
            "audio_seconds": duration,
            "pcm_bytes": self.bytes,
            "sample_rate": self.sample_rate,
            "pcm_sha256": self.hash.hexdigest(),
            "client_response_rtf": complete_seconds / duration,
            "required_extra_initial_buffer_seconds": self.initial_buffer_seconds,
            "immediate_playback_underrun_seconds": self.initial_buffer_seconds,
            "arrivals": self.arrivals,
        }


def measure(client, request):
    started = time.perf_counter()
    with client.stream("POST", "/v1/tts/stream", json=request) as response:
        response.raise_for_status()
        expected = {"x-audio-channels": "1", "x-audio-format": "s16le"}
        if any(response.headers.get(k) != v for k, v in expected.items()):
            raise ValueError("Expected native mono PCM16 wire format")
        receipt = PCMReceipt(int(response.headers["x-audio-sample-rate"]))
        headers = {k: v for k, v in response.headers.items() if k.startswith("x-")}
        for chunk in response.iter_bytes():
            receipt.receive(chunk, time.perf_counter() - started)
    result = receipt.finish(time.perf_counter() - started)
    result["response_headers"] = headers
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("style_id")
    parser.add_argument("text")
    parser.add_argument("--url", default="http://127.0.0.1:9880")
    parser.add_argument("--lang", choices=("ko", "ja", "en"), default="ko")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-mode", choices=("fast", "quality", "balanced"), default="fast")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 0 or args.samples < 1 or not 0 <= args.seed < 2**32:
        parser.error("Nonnegative warmups, positive samples and uint32 seed required")
    request = dict(style_id=args.style_id, text=args.text, lang=args.lang, seed=args.seed,
                   latency_mode=args.latency_mode)
    trials = []
    with httpx.Client(base_url=args.url, timeout=180) as client:
        for index in range(args.warmups + args.samples):
            health = client.get("/health")
            health.raise_for_status()
            state = health.json()
            if not state["ready"] or state["busy"]:
                raise RuntimeError("Benchmark requires a ready, idle inference worker")
            row = measure(client, request)
            row.update(index=index, role="warmup" if index < args.warmups else "sample")
            trials.append(row)
    samples = [r for r in trials if r["role"] == "sample"]
    print(json.dumps({
        "scope": "Actual client request start through complete PCM response; initial latency included; audio never saved",
        "request": request, "trials": trials,
        "worst_client_response_rtf": max(r["client_response_rtf"] for r in samples),
        "all_samples_rtf_at_most_one": all(r["client_response_rtf"] <= 1 for r in samples),
        "client_timing_includes_network": True,
        "buffer_definition": "Additional playback start delay after first complete PCM sample needed to avoid starvation in the observed arrival trace; excludes device/decode latency",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
