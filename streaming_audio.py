"""V5-compatible sentence PCM transport, preserving the approved CPUFast model."""

import re

import numpy as np


def split_text(text, max_chars=120):
    # Keep punctuation with its sentence. A long unpunctuated span is bounded.
    sentences = re.findall(r"[^。！？!?\n]+[。！？!?\n]*|[。！？!?\n]+", text.strip())
    result = []
    for sentence in sentences:
        # English periods only end a sentence before whitespace/end (keeps decimals).
        for part in re.split(r"(?<=[.])\s+", sentence.strip()):
            while len(part) > max_chars:
                position = max(
                    part.rfind(mark, 0, max_chars)
                    for mark in (" ", "、", ",", "；", ";")
                )
                position = position + 1 if position >= max_chars // 3 else max_chars
                candidate = part[:position].strip()
                if re.search(r"\w", candidate):
                    result.append(candidate)
                part = part[position:].strip()
            if part and re.search(r"\w", part):
                result.append(part)
    return result


def pcm16(audio):
    if not np.isfinite(audio).all():
        raise RuntimeError("Non-finite generated audio")
    # Explicit little endian, mono signed PCM16 at 48kHz (not RFC audio/L16 big endian).
    return np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes()


def stream_segments(
    engine,
    reference,
    segments,
    lang,
    seed,
    latency_mode="fast",
    reference_frames=None,
    token_stream=False,
    chunk_tokens=64,
):
    if latency_mode not in ("quality", "balanced", "fast"):
        raise ValueError("Unknown latency_mode")
    if reference_frames is not None:
        raise ValueError("reference_frames is a V5 CFM option")
    if token_stream:
        raise ValueError("This approved CPUFast model supports sentence streaming")
    for index, text in enumerate(segments):
        audio, report = engine.synthesize_prepared(
            text, reference, lang=lang, seed=(seed + index) % 2**32
        )
        yield pcm16(audio), report
