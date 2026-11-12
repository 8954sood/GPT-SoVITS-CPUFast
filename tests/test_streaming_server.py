"""API persistence, cache validity, backpressure and disconnect regression tests."""

import asyncio
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from server import Settings, create_app
from streaming_audio import pcm16, split_text


class FakeEngine:
    reference_keys = frozenset({"tokens", "phones", "spec", "features", "ge", "mel"})
    def __init__(self):
        self.reference_segment = None
        self.preparations = 0
        self.syntheses = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False

    def clear_reference(self):
        pass

    def prepare_reference(self, path, transcript, lang):
        self.preparations += 1
        return {
            key: torch.ones(1, 4)
            for key in ("tokens", "phones", "spec", "features", "ge", "mel")
        }

    def synthesize_prepared(self, text, reference, *, lang, seed):
        self.syntheses.append((text, lang, seed))
        if self.block:
            self.entered.set()
            assert self.release.wait(5)
        return np.linspace(-0.5, 0.5, 480, dtype=np.float32), {"seed": seed}


def wav():
    out = io.BytesIO()
    sf.write(out, np.zeros(4 * 16000), 16000, format="WAV", subtype="PCM_16")
    return out.getvalue()


def upload(client, **kwargs):
    data = {
        "name": "neutral",
        "prompt_text": "テストです。",
        "prompt_lang": "ja",
        **kwargs,
    }
    return client.post(
        "/v1/styles",
        files={"audio": ("../../reference.wav", wav(), "audio/wav")},
        data=data,
    )


def application(tmp_path, engine, fingerprint="model-a", **options):
    return create_app(
        Settings(data_dir=tmp_path, warmup=False, **options),
        engine_factory=lambda: engine,
        fingerprint=fingerprint,
    )


def test_register_stream_and_restart_skips_reference_processing(tmp_path):
    first = FakeEngine()
    with TestClient(application(tmp_path, first)) as client:
        response = upload(client)
        assert response.status_code == 201, response.text
        metadata = response.json()
        style_id = metadata["style_id"]
        assert metadata["ready"]
        assert first.preparations == 1
        assert client.get("/v1/styles").json()["styles"][0]["style_id"] == style_id
        response = client.post(
            "/v1/tts/stream",
            json={"style_id": style_id, "text": "こんにちは。テストです。"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-audio-format"] == "s16le"
        assert response.headers["x-stream-segments"] == "2"
        assert len(response.content) == 480 * 2 * 2
        assert [entry[2] for entry in first.syntheses] == [42, 43]
        assert first.preparations == 1
    second = FakeEngine()
    with TestClient(application(tmp_path, second)) as client:
        assert second.preparations == 0
        assert client.get(f"/v1/styles/{style_id}").json()["style_id"] == style_id
        assert (
            client.post(
                "/v1/tts/stream", json={"style_id": style_id, "text": "テスト。"}
            ).status_code
            == 200
        )
        assert second.preparations == 0


@pytest.mark.parametrize("change", ["model", "corrupt"])
def test_incompatible_cache_rebuild_preserves_id(tmp_path, change):
    with TestClient(application(tmp_path, FakeEngine())) as client:
        style_id = upload(client).json()["style_id"]
    if change == "corrupt":
        (tmp_path / style_id / "prepared.pt").write_bytes(b"broken")
    engine = FakeEngine()
    with TestClient(
        application(
            tmp_path, engine, fingerprint="model-b" if change == "model" else "model-a"
        )
    ) as client:
        assert engine.preparations == 1
        assert client.get("/v1/styles").json()["styles"][0]["style_id"] == style_id
        assert (
            client.post(
                "/v1/tts/stream", json={"style_id": style_id, "text": "テスト。"}
            ).status_code
            == 200
        )


def test_capacity_and_invalid_upload_do_not_publish_partial_styles(tmp_path):
    with TestClient(application(tmp_path, FakeEngine(), max_styles=1)) as client:
        assert upload(client, end="2").status_code == 422
        assert client.get("/v1/styles").json()["styles"] == []
        assert upload(client).status_code == 201
        assert upload(client).status_code == 507
        assert len(client.get("/v1/styles").json()["styles"]) == 1
        assert len(list(tmp_path.iterdir())) == 1


def test_empty_text_unknown_style_and_body_limit(tmp_path):
    with TestClient(application(tmp_path, FakeEngine())) as client:
        assert (
            client.post(
                "/v1/tts/stream", json={"style_id": "a" * 32, "text": "  "}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/tts/stream", json={"style_id": "a" * 32, "text": "test"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/v1/styles", content=b"x", headers={"Content-Length": str(9 * 1024**2)}
            ).status_code
            == 413
        )
        assert client.get("/health").json()["busy"] is False


def test_busy_worker_rejects_concurrent_mutation_and_health_stays_responsive(tmp_path):
    engine = FakeEngine()
    with TestClient(application(tmp_path, engine)) as client:
        style_id = upload(client).json()["style_id"]
        engine.block = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                client.post,
                "/v1/tts/stream",
                json={"style_id": style_id, "text": "test"},
            )
            assert engine.entered.wait(3)
            assert client.get("/health").json()["busy"] is True
            busy = client.post(
                "/v1/tts/stream", json={"style_id": style_id, "text": "test"}
            )
            assert busy.status_code == 429 and busy.headers["retry-after"] == "1"
            assert upload(client).status_code == 429
            engine.release.set()
            assert pending.result(timeout=5).status_code == 200
        assert client.get("/health").json()["busy"] is False


def test_cancelled_prefetch_keeps_lease_until_native_call_finishes(tmp_path):
    async def run():
        engine = FakeEngine()
        app = application(tmp_path, engine)
        async with app.router.lifespan_context(app):
            # Run a real multipart registration through ASGI transport.
            import httpx

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/styles",
                    files={"audio": ("ref.wav", wav())},
                    data={"name": "voice", "prompt_text": "test", "prompt_lang": "en"},
                )
                style_id = response.json()["style_id"]
                for _ in range(100):
                    if not app.state.busy.locked():
                        break
                    await asyncio.sleep(0.001)
                engine.block = True
                task = asyncio.create_task(
                    client.post(
                        "/v1/tts/stream", json={"style_id": style_id, "text": "test"}
                    )
                )
                assert await asyncio.to_thread(engine.entered.wait, 3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert app.state.busy.locked()
                assert (
                    await client.post(
                        "/v1/tts/stream", json={"style_id": style_id, "text": "test"}
                    )
                ).status_code == 429
                engine.release.set()
                for _ in range(1000):
                    if not app.state.busy.locked():
                        break
                    await asyncio.sleep(0.001)
                assert not app.state.busy.locked()
                engine.block = False
                assert (
                    await client.post(
                        "/v1/tts/stream", json={"style_id": style_id, "text": "test"}
                    )
                ).status_code == 200

    asyncio.run(run())


def test_disconnect_after_headers_releases_owner(tmp_path):
    async def run():
        engine = FakeEngine()
        app = application(tmp_path, engine)
        async with app.router.lifespan_context(app):
            source = tmp_path / ".source.wav"
            source.write_bytes(wav())
            metadata = await asyncio.get_running_loop().run_in_executor(
                app.state.executor,
                lambda: app.state.store.register(
                    source, name="v", prompt_text="test", prompt_lang="en"
                ),
            )
            body = json.dumps(
                {"style_id": metadata["style_id"], "text": "hello"}
            ).encode()
            sent = False

            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body}
                await asyncio.sleep(100)

            async def send(message):
                if message["type"] == "http.response.start":
                    raise OSError("client disconnected")

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "method": "POST",
                "path": "/v1/tts/stream",
                "raw_path": b"/v1/tts/stream",
                "query_string": b"",
                "scheme": "http",
                "server": ("test", 80),
                "client": ("test", 1),
                "headers": [(b"content-type", b"application/json")],
            }
            with pytest.raises(ClientDisconnect):
                await app(scope, receive, send)
            for _ in range(1000):
                if not app.state.busy.locked():
                    break
                await asyncio.sleep(0.001)
            assert not app.state.busy.locked()

    asyncio.run(run())


def test_segmentation_pcm_and_seed_boundaries():
    assert split_text("こんにちは。テストです！\nHello. World!") == [
        "こんにちは。",
        "テストです！",
        "Hello.",
        "World!",
    ]
    assert all(len(x) <= 40 for x in split_text("あ" * 101, 40))
    assert split_text("！？") == []
    assert np.array_equal(
        np.frombuffer(pcm16(np.array([-2.0, 0.0, 2.0])), dtype="<i2"),
        [-32768, 0, 32767],
    )
    with pytest.raises(RuntimeError):
        pcm16(np.array([float("nan")]))


def test_multiple_styles_keep_separate_prepared_tensors(tmp_path):
    class DistinctEngine(FakeEngine):
        def prepare_reference(self, *args):
            ref = super().prepare_reference(*args)
            return {k: v * self.preparations for k, v in ref.items()}

        def synthesize_prepared(self, text, reference, *, lang, seed):
            return np.full(480, reference["features"][0, 0].item() / 10, np.float32), {}

    with TestClient(application(tmp_path, DistinctEngine())) as client:
        first = upload(client, name="voice-a").json()["style_id"]
        second = upload(client, name="voice-b").json()["style_id"]
        assert first != second

        def speak(style_id):
            response = client.post(
                "/v1/tts/stream", json={"style_id": style_id, "text": "test"}
            )
            assert response.status_code == 200
            return response.content

        a, b, again_a = speak(first), speak(second), speak(first)
        assert a == again_a and a != b
    with TestClient(application(tmp_path, DistinctEngine())) as client:
        assert speak(first) == a and speak(second) == b


def test_disconnect_during_second_segment_does_not_overlap_next_request(tmp_path):
    class BlockingSecondEngine(FakeEngine):
        def synthesize_prepared(self, *args, **kwargs):
            self.block = len(self.syntheses) == 1
            return super().synthesize_prepared(*args, **kwargs)

    async def run():
        engine = BlockingSecondEngine()
        app = application(tmp_path, engine)
        async with app.router.lifespan_context(app):
            source = tmp_path / ".source.wav"
            source.write_bytes(wav())
            metadata = await asyncio.get_running_loop().run_in_executor(
                app.state.executor,
                lambda: app.state.store.register(
                    source, name="v", prompt_text="test", prompt_lang="en"
                ),
            )
            payload = json.dumps(
                {"style_id": metadata["style_id"], "text": "first!second!"}
            ).encode()
            sent = False

            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": payload}
                assert await asyncio.to_thread(engine.entered.wait, 3)
                return {"type": "http.disconnect"}

            received = []

            async def send(message):
                received.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "method": "POST",
                "path": "/v1/tts/stream",
                "raw_path": b"/v1/tts/stream",
                "query_string": b"",
                "scheme": "http",
                "server": ("test", 80),
                "client": ("test", 1),
                "headers": [(b"content-type", b"application/json")],
            }
            await app(scope, receive, send)
            assert any(m.get("body") for m in received)
            assert app.state.busy.locked()
            engine.release.set()
            for _ in range(1000):
                if not app.state.busy.locked():
                    break
                await asyncio.sleep(0.001)
            assert not app.state.busy.locked()

    asyncio.run(run())


def test_latency_aliases_preserve_full_approved_reference(tmp_path):
    from server import SpeechRequest
    from streaming_audio import stream_segments

    assert SpeechRequest(style_id="a" * 32, text="hello").latency_mode == "fast"
    seen = []

    class ContextEngine(FakeEngine):
        def synthesize_prepared(self, text, reference, **kwargs):
            seen.append(reference)
            return super().synthesize_prepared(text, reference, **kwargs)

    engine = ContextEngine()
    with TestClient(application(tmp_path, engine)) as client:
        style_id = upload(client).json()["style_id"]
        original = client.app.state.store.get(style_id)
        bodies = []
        for mode in ("quality", "balanced", "fast"):
            deadline = time.monotonic() + 3
            while client.get("/health").json()["busy"]:
                assert time.monotonic() < deadline
                time.sleep(.001)
            response = client.post("/v1/tts/stream", json={
                "style_id": style_id, "text": "first!second!", "latency_mode": mode})
            assert response.status_code == 200
            assert response.headers["x-latency-mode"] == mode
            assert response.headers["x-latency-mode-effective"] == "fixed_approved_model"
            assert response.headers["x-audio-sample-rate"] == "48000"
            bodies.append(response.content)
        assert bodies[0] == bodies[1] == bodies[2]
        assert len(seen) == 6 and all(ref is original for ref in seen)
        assert client.get("/health").json()["latency_mode_effect"] == "fixed_approved_model"
    engine.syntheses.clear()
    list(stream_segments(engine, original, ["first", "second"], "ko", 2**32 - 1))
    assert [row[2] for row in engine.syntheses] == [2**32 - 1, 0]


def test_v5_reference_frame_option_is_rejected_before_model_work(tmp_path):
    with TestClient(application(tmp_path, FakeEngine())) as client:
        style_id = upload(client).json()["style_id"]
        response = client.post("/v1/tts/stream", json={
            "style_id": style_id, "text": "hello", "reference_frames": 128})
        assert response.status_code == 422
        assert "V5 CFM" in response.json()["detail"]
        assert not client.app.state.engine.syntheses


def test_delete_removes_files_cache_and_capacity_and_survives_restart(tmp_path):
    with TestClient(application(tmp_path, FakeEngine(), max_styles=2)) as client:
        first = upload(client, name="remove").json()
        second = upload(client, name="keep").json()
        style_id = first["style_id"]
        response = client.delete(f"/v1/styles/{style_id}")
        assert response.status_code == 204 and response.content == b""
        assert not (tmp_path / style_id).exists()
        assert not (tmp_path / f".deleted-{style_id}").exists()
        assert client.get(f"/v1/styles/{style_id}").status_code == 404
        assert client.delete(f"/v1/styles/{style_id}").status_code == 404
        assert (
            client.post(
                "/v1/tts/stream", json={"style_id": style_id, "text": "test"}
            ).status_code
            == 404
        )
        health = client.get("/health").json()
        assert health["styles"] == 1 and health["cache_bytes"] == second["cache_bytes"]
        assert (
            client.post(
                "/v1/tts/stream", json={"style_id": second["style_id"], "text": "test"}
            ).status_code
            == 200
        )
        # Deletion releases both style-count and prepared-tensor quota.
        assert upload(client, name="replacement").status_code == 201
    engine = FakeEngine()
    with TestClient(application(tmp_path, engine, max_styles=2)) as client:
        ids = [item["style_id"] for item in client.get("/v1/styles").json()["styles"]]
        assert style_id not in ids and second["style_id"] in ids
        assert engine.preparations == 0
        assert client.get(f"/v1/styles/{style_id}").status_code == 404
        schema = client.get("/openapi.json").json()
        assert (
            "content"
            not in schema["paths"]["/v1/styles/{style_id}"]["delete"]["responses"][
                "204"
            ]
        )


def test_delete_cannot_mutate_an_active_stream(tmp_path):
    engine = FakeEngine()
    with TestClient(application(tmp_path, engine)) as client:
        style_id = upload(client).json()["style_id"]
        engine.block = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                client.post,
                "/v1/tts/stream",
                json={"style_id": style_id, "text": "test"},
            )
            assert engine.entered.wait(3)
            response = client.delete(f"/v1/styles/{style_id}")
            assert (
                response.status_code == 429 and response.headers["retry-after"] == "1"
            )
            assert (tmp_path / style_id / "prepared.pt").exists()
            assert client.get(f"/v1/styles/{style_id}").status_code == 200
            engine.release.set()
            assert pending.result(timeout=5).status_code == 200
        assert client.delete(f"/v1/styles/{style_id}").status_code == 204


@pytest.mark.parametrize("retry_via_restart", [False, True])
def test_partial_delete_never_resurrects_and_cleanup_can_retry(
    tmp_path, monkeypatch, retry_via_restart
):
    import style_store

    original = style_store.shutil.rmtree

    def fail(path, *args, **kwargs):
        if str(path).split("/")[-1].startswith(".deleted-"):
            raise PermissionError("simulated disk cleanup error")
        return original(path, *args, **kwargs)

    with TestClient(application(tmp_path, FakeEngine())) as client:
        style_id = upload(client).json()["style_id"]
        monkeypatch.setattr(style_store.shutil, "rmtree", fail)
        response = client.delete(f"/v1/styles/{style_id}")
        assert response.status_code == 503
        assert client.get(f"/v1/styles/{style_id}").status_code == 404
        assert client.get("/health").json()["cache_bytes"] == 0
        assert (tmp_path / f".deleted-{style_id}").exists()
        monkeypatch.setattr(style_store.shutil, "rmtree", original)
        if not retry_via_restart:
            assert client.delete(f"/v1/styles/{style_id}").status_code == 204
    with TestClient(application(tmp_path, FakeEngine())) as client:
        assert not (tmp_path / f".deleted-{style_id}").exists()
        assert client.get("/v1/styles").json()["styles"] == []
        assert client.get(f"/v1/styles/{style_id}").status_code == 404


def test_failed_delete_rename_keeps_style_and_can_retry(tmp_path, monkeypatch):
    import style_store

    original = style_store.os.replace
    with TestClient(application(tmp_path, FakeEngine())) as client:
        style_id = upload(client).json()["style_id"]

        def fail(source, destination):
            if str(destination).endswith(f".deleted-{style_id}"):
                raise PermissionError("simulated rename error")
            return original(source, destination)

        monkeypatch.setattr(style_store.os, "replace", fail)
        assert client.delete(f"/v1/styles/{style_id}").status_code == 503
        assert client.get(f"/v1/styles/{style_id}").status_code == 200
        assert (tmp_path / style_id / "reference.wav").exists()
        monkeypatch.setattr(style_store.os, "replace", original)
        assert client.delete(f"/v1/styles/{style_id}").status_code == 204


def test_store_delete_rejects_path_components(tmp_path):
    from style_store import StyleStore

    store = StyleStore(tmp_path / "styles", FakeEngine(), "test", warmup=False)
    other = tmp_path / "keep"
    other.mkdir()
    (other / "reference.wav").write_bytes(b"keep")
    for style_id in ["../keep", ".", "", "A" * 32, "a" * 31]:
        with pytest.raises(KeyError):
            store.delete(style_id)
    assert (other / "reference.wav").read_bytes() == b"keep"


def test_delete_releases_prepared_tensor_byte_limit(tmp_path):
    with TestClient(application(tmp_path, FakeEngine(), cache_bytes=96)) as client:
        style_id = upload(client).json()["style_id"]
        assert upload(client).status_code == 507
        assert client.delete(f"/v1/styles/{style_id}").status_code == 204
        assert client.get("/health").json()["cache_bytes"] == 0
        assert upload(client).status_code == 201


def test_token_stream_requires_explicit_arm_server(tmp_path):
    with TestClient(application(tmp_path, FakeEngine())) as client:
        style = upload(client).json()["style_id"]
        response = client.post(
            "/v1/tts/stream",
            json={
                "style_id": style,
                "text": "テストです。",
                "lang": "ja",
                "streaming_mode": "tokens",
            },
        )
        assert response.status_code == 422
        assert not client.get("/health").json()["busy"]


def test_failed_model_is_not_ready_and_rejects_additional_work(tmp_path):
    engine = FakeEngine()
    with TestClient(application(tmp_path, engine)) as client:
        style_id = upload(client).json()["style_id"]
        engine.ready = False
        assert not client.get("/health").json()["ready"]
        response = client.post("/v1/tts/stream", json={"style_id": style_id, "text": "hello"})
        assert response.status_code == 503
        assert upload(client).status_code == 503
        assert not engine.syntheses


def test_hash_valid_cache_with_wrong_tensor_schema_is_rebuilt(tmp_path):
    import hashlib

    with TestClient(application(tmp_path, FakeEngine())) as client:
        style_id = upload(client).json()["style_id"]
    folder = tmp_path / style_id
    tensors = torch.load(folder / "prepared.pt", weights_only=True)
    tensors["unexpected"] = torch.ones(1)
    torch.save(tensors, folder / "prepared.pt")
    meta = json.loads((folder / "metadata.json").read_text())
    meta["cache_sha256"] = hashlib.sha256((folder / "prepared.pt").read_bytes()).hexdigest()
    (folder / "metadata.json").write_text(json.dumps(meta))
    engine = FakeEngine()
    with TestClient(application(tmp_path, engine)) as client:
        assert engine.preparations == 1
        assert client.get("/health").json()["rebuilt_on_startup"] == 1
        assert client.get(f"/v1/styles/{style_id}").status_code == 200
