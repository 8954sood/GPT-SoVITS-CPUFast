"""Durable zero-shot style references. Called only on the inference owner thread."""

import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import soundfile as sf
import torch

CACHE_VERSION = 2


def _file_digest(stream, algorithm):
    digest = hashlib.new(algorithm)
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest


class CapacityError(ValueError):
    pass


class DeletionError(OSError):
    """A filesystem operation failed; retry the same deletion after recovery."""


def tensor_bytes(reference, expected_keys=None):
    if not isinstance(reference, dict) or not reference or (
        expected_keys is not None and set(reference) != set(expected_keys)
    ) or not all(
        isinstance(v, torch.Tensor) for v in reference.values()
    ):
        raise ValueError("Invalid prepared-reference tensor keys")
    for value in reference.values():
        if (
            value.device.type != "cpu"
            or value.numel() == 0
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid prepared-reference tensor")
    return sum(v.numel() * v.element_size() for v in reference.values())


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class StyleStore:
    def __init__(
        self,
        root,
        engine,
        fingerprint,
        *,
        max_styles=32,
        max_bytes=64 * 1024**2,
        warmup=True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine, self.fingerprint = engine, fingerprint
        self.reference_keys = getattr(engine, "reference_keys", None)
        self.max_styles, self.max_bytes, self.warmup = max_styles, max_bytes, warmup
        self.styles = {}
        self.catalog = ()
        self.bytes = 0
        self.warmed_languages = set()
        self.loaded_from_disk = 0
        self.rebuilt_on_startup = 0

    def _publish(self):
        self.catalog = tuple(dict(metadata) for metadata, _ in self.styles.values())

    def _capacity(self, size):
        if len(self.styles) >= self.max_styles or self.bytes + size > self.max_bytes:
            raise CapacityError(
                "Style capacity exceeded; increase limits within the memory budget"
            )

    def _prepare(self, path, metadata):
        old = self.engine.reference_segment
        try:
            self.engine.reference_segment = (metadata["start"], metadata["end"])
            return self.engine.prepare_reference(
                path, metadata["prompt_text"], metadata["prompt_lang"]
            )
        finally:
            self.engine.reference_segment = old
            self.engine.clear_reference()

    def _warm(self, metadata, reference):
        language = metadata["prompt_lang"]
        if self.warmup and language not in self.warmed_languages:
            text = {"ja": "テスト。", "ko": "테스트입니다.", "en": "Hello."}[language]
            self.engine.synthesize_prepared(text, reference, lang=language, seed=42)
            self.warmed_languages.add(language)

    def _save_cache(self, directory, metadata, reference):
        temporary = directory / "prepared.tmp"
        with temporary.open("wb") as stream:
            torch.save(reference, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "prepared.pt")
        with (directory / "prepared.pt").open("rb") as stream:
            metadata["cache_sha256"] = _file_digest(stream, "sha256").hexdigest()
        metadata["fingerprint"] = self.fingerprint
        metadata["cache_version"] = CACHE_VERSION
        atomic_json(directory / "metadata.json", metadata)

    def load(self):
        """Preload before readiness; rebuild incompatible/corrupt caches with same ID."""
        # A renamed deletion is committed: never restore it as a live style.
        # Finish interrupted disk cleanup before accepting requests.
        for directory in self.root.glob(".deleted-*"):
            if re.fullmatch(r"\.deleted-[a-f0-9]{32}", directory.name):
                shutil.rmtree(directory)
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata["style_id"] != directory.name:
                raise ValueError("Style directory/ID mismatch")
            with (directory / "reference.wav").open("rb") as stream:
                if (
                    _file_digest(stream, "sha256").hexdigest()
                    != metadata["audio_sha256"]
                ):
                    raise ValueError(
                        f"Stored reference checksum mismatch: {directory.name}"
                    )
            try:
                if (
                    metadata["fingerprint"] != self.fingerprint
                    or metadata["cache_version"] != CACHE_VERSION
                ):
                    raise ValueError("Cache version changed")
                with (directory / "prepared.pt").open("rb") as stream:
                    if (
                        _file_digest(stream, "sha256").hexdigest()
                        != metadata["cache_sha256"]
                    ):
                        raise ValueError("Cache checksum changed")
                reference = torch.load(
                    directory / "prepared.pt", map_location="cpu", weights_only=True
                )
                size = tensor_bytes(reference, self.reference_keys)
                self.loaded_from_disk += 1
            except (
                ValueError,
                OSError,
                RuntimeError,
                EOFError,
                KeyError,
                pickle.UnpicklingError,
            ):
                reference = self._prepare(directory / "reference.wav", metadata)
                size = tensor_bytes(reference, self.reference_keys)
                self._save_cache(directory, metadata, reference)
                self.rebuilt_on_startup += 1
            self._capacity(size)
            self._warm(metadata, reference)
            metadata.update(ready=True, cache_bytes=size)
            self.styles[metadata["style_id"]] = (metadata, reference)
            self.bytes += size
        self._publish()

    def register(self, source, *, name, prompt_text, prompt_lang, start=0.0, end=None):
        info = sf.info(source)
        if (
            info.format not in ("WAV", "WAVEX")
            or not 8000 <= info.samplerate <= 192000
            or info.channels not in (1, 2)
        ):
            raise ValueError("Upload mono/stereo WAV at 8–192 kHz")
        if (
            info.duration > 60
            or start < 0
            or start >= info.duration
            or (end is not None and end > info.duration)
        ):
            raise ValueError("Invalid crop or source longer than 60 seconds")
        duration = (info.duration if end is None else end) - start
        if not 3 <= duration <= 10:
            raise ValueError("Selected reference must be 3–10 seconds")
        self._capacity(0)
        style_id = uuid.uuid4().hex
        metadata = {
            "style_id": style_id,
            "name": name,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "start": start,
            "end": end,
            "duration_seconds": duration,
            "created_at": time.time(),
        }

        with tempfile.TemporaryDirectory(
            prefix=".register-", dir=self.root
        ) as temporary:
            directory = Path(temporary)
            target = directory / "reference.wav"
            shutil.copyfile(source, target)
            with target.open("rb") as stream:
                metadata["audio_sha256"] = _file_digest(
                    stream, "sha256"
                ).hexdigest()
            reference = self._prepare(target, metadata)
            size = tensor_bytes(reference, self.reference_keys)
            self._capacity(size)
            self._warm(metadata, reference)
            metadata.update(ready=True, cache_bytes=size)
            self._save_cache(directory, metadata, reference)
            # Publish only complete registrations. Temporary directories are ignored on restart.
            os.replace(directory, self.root / style_id)
        self.styles[style_id] = (metadata, reference)
        self.bytes += size
        self._publish()
        return dict(metadata)

    def get(self, style_id):
        return self.styles[style_id][1]

    def delete(self, style_id):
        """Remove a style on the owner thread; retry interrupted disk cleanup.

        Rename before unlinking so a partial deletion cannot be restored at
        startup. Once renamed, drop all registry references and free capacity.
        A successful return means the tombstone and its files are gone too.
        """
        if not re.fullmatch(r"[a-f0-9]{32}", style_id):
            raise KeyError(style_id)
        entry = self.styles.get(style_id)
        directory = self.root / style_id
        tombstone = self.root / f".deleted-{style_id}"
        if entry is None and not tombstone.exists():
            raise KeyError(style_id)
        try:
            if entry is not None:
                if directory.exists():
                    os.replace(directory, tombstone)
                del self.styles[style_id]
                self.bytes -= entry[0]["cache_bytes"]
                self._publish()
                del entry
            if tombstone.exists():
                shutil.rmtree(tombstone)
        except OSError as error:
            raise DeletionError(
                "Style deletion could not finish; retry this DELETE request"
            ) from error
