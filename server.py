"""Single-owner ARM64 CPU streaming server with durable zero-shot styles."""

import argparse
import asyncio
import functools
import json
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from streaming_audio import split_text, stream_segments
from style_store import CapacityError, DeletionError, StyleStore

ROOT = Path(__file__).resolve().parent
APPROVED_MANIFEST_SHA256 = "696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865"

LOG = logging.getLogger("uvicorn.error.gsv")
UPLOAD_BYTES = 8 * 1024**2
END = object()


@dataclass
class Settings:
    data_dir: Path = ROOT / "data/styles"
    manifest: Path = ROOT / "models/student/manifest.json"
    manifest_sha256: str = APPROVED_MANIFEST_SHA256
    s2: Path = ROOT / "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
    hubert: Path = ROOT / "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    max_styles: int = 32
    cache_bytes: int = 64 * 1024**2
    warmup: bool = True
    ar_threads: int = 2
    threads: int = 2


class SpeechRequest(BaseModel):
    style_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    text: str = Field(min_length=1, max_length=2000)
    lang: Literal["ja", "ko", "en"] = "ja"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    latency_mode: Literal["quality", "balanced", "fast"] = "fast"
    max_segment_chars: int = Field(default=120, ge=20, le=240)
    reference_frames: int | None = Field(default=None, ge=16, le=500)
    streaming_mode: Literal["sentences", "tokens"] = "sentences"
    chunk_tokens: int = Field(default=48, ge=16, le=128)

    @field_validator("text")
    @classmethod
    def nonempty(cls, text):
        if not text.strip():
            raise ValueError("Text must not be blank")
        return text.strip()


class UploadLimit:
    """Bound multipart bodies before they can fill a spool file."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/v1/styles"
        ):
            return await self.app(scope, receive, send)
        limit = UPLOAD_BYTES + 64 * 1024
        headers = dict(scope["headers"])
        try:
            oversized = int(headers.get(b"content-length", b"0")) > limit
        except ValueError:
            oversized = True
        if oversized:
            from starlette.responses import JSONResponse

            return await JSONResponse(
                {"detail": "Upload exceeds 8 MiB plus form metadata"}, 413
            )(scope, receive, send)
        count = 0

        async def limited_receive():
            nonlocal count
            message = await receive()
            count += len(message.get("body", b""))
            if count > limit:
                raise HTTPException(413, "Upload exceeds 8 MiB plus form metadata")
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings=None, *, engine_factory=None, fingerprint=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gsv-inference")
        app.state.executor = executor
        app.state.busy = asyncio.Lock()
        loop = asyncio.get_running_loop()

        def initialize():
            if engine_factory is None:
                import torch

                from cpufast_server_engine import Engine

                torch.set_num_threads(2)
                torch.set_num_interop_threads(1)
                engine = Engine(settings)
            else:
                engine = engine_factory()
            identity = fingerprint or engine.fingerprint
            store = StyleStore(
                settings.data_dir,
                engine,
                identity,
                max_styles=settings.max_styles,
                max_bytes=settings.cache_bytes,
                warmup=settings.warmup,
            )
            store.load()
            return engine, store

        try:
            app.state.engine, app.state.store = await loop.run_in_executor(
                executor, initialize
            )
            yield
        finally:
            # Shutdown waits for owner work; never load a second model worker.
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)

    app = FastAPI(
        title="GPT-SoVITS CPUFast ARM64 streaming API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(UploadLimit)

    def acquire():
        # No await between checking and taking a free asyncio lock.
        if not getattr(app.state.engine, "ready", True):
            raise HTTPException(503, "Model worker needs restart after an inference failure")
        if app.state.busy.locked():
            raise HTTPException(
                429,
                "Model worker busy; retry after the active operation finishes",
                headers={"Retry-After": "1"},
            )
        return app.state.busy.acquire()

    def submit(function, *args, **kwargs):
        return app.state.executor.submit(functools.partial(function, *args, **kwargs))

    async def wait(future):
        # Cancelling a client cannot cancel an already-running native model call.
        return await asyncio.shield(asyncio.wrap_future(future))

    class Lease:
        def __init__(self, iterator=None):
            self.iterator = iterator
            self.closed = False
            self.pending = None
            self.loop = asyncio.get_running_loop()

        def submit(self, function, *args, **kwargs):
            self.pending = submit(function, *args, **kwargs)
            return self.pending

        async def close(self):
            if self.closed:
                return
            self.closed = True
            work_done = self.pending is None or self.pending.done()
            if self.iterator is None and work_done:
                app.state.busy.release()
                return
            # FIFO: an in-flight next()/registration completes before cleanup.
            # Keep admission closed until the real worker is finished.
            future = submit(
                self.iterator.close if self.iterator is not None else lambda: None
            )
            future.add_done_callback(
                lambda _future: self.loop.call_soon_threadsafe(app.state.busy.release)
            )
            if work_done:
                # Publish an idle worker before a successful response completes.
                # Cancellation during native work still returns immediately above.
                await wait(future)

    @app.get("/health")
    async def health():
        return {
            "ready": bool(getattr(app.state.engine, "ready", True)),
            "busy": app.state.busy.locked(),
            "styles": len(app.state.store.catalog),
            "cache_bytes": app.state.store.bytes,
            "preloaded_from_disk": app.state.store.loaded_from_disk,
            "rebuilt_on_startup": app.state.store.rebuilt_on_startup,
            "steps": None,
            "cfg": None,
            "acoustic_threads": settings.threads,
            "ar_threads": settings.ar_threads,
            "ar_dispatch": getattr(app.state.engine, "ar_dispatch", None),
            "streaming": "sentence_fragments",
            "default_latency_mode": "fast",
            "cfm_backend": None,
            "native_gelu": False,
            "cached_rotary": False,
            "arm_extreme": False,
            "arm_profile": None,
            "upstream_revision": None,
            "model_variant": "AR17_QAT200_C24_TWO_BRANCH_VTARGET100",
            "model_manifest_sha256": settings.manifest_sha256,
            "native_sample_rate": 32000,
            "output_sample_rate": app.state.engine.sample_rate,
            "latency_mode_effect": "fixed_approved_model",
        }

    @app.get("/v1/styles")
    async def styles():
        return {
            "styles": [dict(item) for item in app.state.store.catalog],
            "max_styles": settings.max_styles,
            "cache_limit_bytes": settings.cache_bytes,
        }

    @app.get("/v1/styles/{style_id}")
    async def style(style_id: str):
        for metadata in app.state.store.catalog:
            if metadata["style_id"] == style_id:
                return dict(metadata)
        raise HTTPException(404, "Unknown style_id")

    @app.delete("/v1/styles/{style_id}", status_code=204, response_class=Response)
    async def delete_style(style_id: str):
        await acquire()
        lease = Lease()
        try:
            await wait(lease.submit(app.state.store.delete, style_id))
            return Response(status_code=204)
        except KeyError as error:
            raise HTTPException(404, "Unknown style_id") from error
        except DeletionError as error:
            LOG.exception("Style deletion failed: %s", style_id)
            raise HTTPException(
                503, str(error), headers={"Retry-After": "1"}
            ) from error
        finally:
            await lease.close()

    @app.post("/v1/styles", status_code=201)
    async def register(
        audio: Annotated[
            UploadFile,
            File(description="3–10 second reference WAV, or crop a <=60 second source"),
        ],
        name: Annotated[str, Form(min_length=1, max_length=80)],
        prompt_text: Annotated[str, Form(min_length=1, max_length=1000)],
        prompt_lang: Annotated[Literal["ja", "ko", "en"], Form()],
        start: Annotated[float, Form(ge=0, allow_inf_nan=False)] = 0,
        end: Annotated[float | None, Form(gt=0, allow_inf_nan=False)] = None,
    ):
        if not name.strip() or not prompt_text.strip():
            raise HTTPException(422, "Name and reference transcript must not be blank")
        await acquire()
        lease = Lease()
        source = None
        try:
            # The body is already bounded by middleware; do not trust upload filenames.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stream:
                source = Path(stream.name)
                size = 0
                while chunk := await audio.read(65536):
                    size += len(chunk)
                    if size > UPLOAD_BYTES:
                        raise HTTPException(413, "Reference exceeds 8 MiB")
                    stream.write(chunk)
            path = source
            source = None  # Worker owns cleanup even if the HTTP task disconnects.

            def work():
                try:
                    return app.state.store.register(
                        path,
                        name=name.strip(),
                        prompt_text=prompt_text.strip(),
                        prompt_lang=prompt_lang,
                        start=start,
                        end=end,
                    )
                finally:
                    path.unlink(missing_ok=True)

            return await wait(lease.submit(work))
        except CapacityError as error:
            raise HTTPException(507, str(error)) from error
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error
        finally:
            if source is not None:
                source.unlink(missing_ok=True)
            await lease.close()
            await audio.close()

    @app.post(
        "/v1/tts/stream",
        response_class=StreamingResponse,
        responses={
            200: {
                "content": {"application/octet-stream": {}},
                "description": "Native32kHz mono PCM s16le",
            }
        },
    )
    async def speech(request: SpeechRequest):
        if request.streaming_mode == "tokens":
            raise HTTPException(422, "This approved CPUFast model supports sentence streaming")
        if request.reference_frames is not None:
            raise HTTPException(422, "reference_frames is a V5 CFM option; CPUFast keeps the full reference")
        segments = split_text(request.text, request.max_segment_chars)
        if not segments:
            raise HTTPException(422, "No speakable text")
        try:
            reference = app.state.store.get(request.style_id)
        except KeyError as error:
            raise HTTPException(404, "Unknown style_id") from error
        await acquire()
        iterator = stream_segments(
            app.state.engine,
            reference,
            segments,
            request.lang,
            request.seed,
            request.latency_mode,
            request.reference_frames,
            request.streaming_mode == "tokens",
            request.chunk_tokens,
        )
        lease = Lease(iterator)
        started = time.perf_counter()
        request_id = uuid.uuid4().hex

        def next_chunk():
            return next(iterator, END)

        try:
            # A real first PCM segment is prepared before committing HTTP 200.
            first = await wait(lease.submit(next_chunk))
            if first is END:
                raise ValueError("No audio produced")
        except BaseException as error:
            await lease.close()
            if isinstance(error, (ValueError, RuntimeError)):
                raise HTTPException(422, str(error)) from error
            raise
        first_seconds = time.perf_counter() - started

        async def body():
            item = first
            completed = 0
            segment_timings = []
            try:
                while item is not END:
                    pcm, _report = item
                    segment_timings.append(_report)
                    # Only one generated segment is retained; no unbounded producer queue.
                    for offset in range(0, len(pcm), 9600):
                        yield pcm[offset : offset + 9600]
                        await asyncio.sleep(0)
                    completed += 1
                    item = await wait(lease.submit(next_chunk))
            except Exception:
                LOG.exception(
                    "Streaming inference failed after HTTP headers: %s", request_id
                )
                raise
            finally:
                await lease.close()
                LOG.info(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "style_id": request.style_id,
                            "first_audio_seconds": first_seconds,
                            "latency_mode": request.latency_mode,
                            "first_segment_timing": first[1],
                            "segment_timings": segment_timings,
                            "completed_segments": completed,
                            "total_seconds": time.perf_counter() - started,
                        }
                    )
                )

        class OwnedStreamingResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    await lease.close()

        return OwnedStreamingResponse(
            body(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Sample-Rate": str(app.state.engine.sample_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Format": "s16le",
                "X-Style-ID": request.style_id,
                "X-Latency-Mode": request.latency_mode,
                "X-Latency-Mode-Effective": "fixed_approved_model",
                "X-First-Audio-Samples": str(len(first[0]) // 2),
                "X-Stream-Segments": str(len(segments)),
                "X-Request-ID": request_id,
                "X-Time-To-First-Audio-Ms": f"{first_seconds * 1000:.3f}",
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(lease.close),
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/styles")
    parser.add_argument("--manifest", type=Path, default=Settings.manifest)
    parser.add_argument("--manifest-sha256", default=APPROVED_MANIFEST_SHA256)
    parser.add_argument("--s2", type=Path, default=Settings.s2)
    parser.add_argument("--hubert", type=Path, default=Settings.hubert)
    parser.add_argument("--max-styles", type=int, default=32)
    parser.add_argument("--cache-mib", type=int, default=64)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ar-threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.max_styles < 1 or args.cache_mib < 1:
        parser.error("Style and cache limits must be positive")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "2"
    import uvicorn

    settings = Settings(
        data_dir=args.data_dir,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        s2=args.s2,
        hubert=args.hubert,
        max_styles=args.max_styles,
        cache_bytes=args.cache_mib * 1024**2,
        warmup=args.warmup,
        ar_threads=args.ar_threads,
        threads=args.threads,
    )
    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, workers=1, loop="asyncio"
    )


if __name__ == "__main__":
    main()
