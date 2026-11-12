import hashlib

import pytest

from scripts.measure_streaming_api import PCMReceipt


def test_wire_rtf_includes_first_latency_and_reports_starvation():
    wire = PCMReceipt()
    packet = b"\x01\x00" * 48000
    for seconds in (1.0, 2.5, 3.0):
        wire.receive(packet, seconds)
    report = wire.finish(3.2)
    assert report["audio_seconds"] == 3
    assert report["client_response_rtf"] == pytest.approx(3.2 / 3)
    assert report["client_first_pcm_seconds"] == 1
    assert report["required_extra_initial_buffer_seconds"] == .5
    assert report["pcm48k_sha256"] == hashlib.sha256(packet * 3).hexdigest()


def test_network_packet_boundaries_and_empty_chunks_do_not_change_duration():
    wire = PCMReceipt()
    wire.receive(b"", 0)
    wire.receive(b"\x01", .1)
    wire.receive(b"\x02\x03", .2)
    wire.receive(b"\x04", .2)
    report = wire.finish(.3)
    assert report["audio_seconds"] == 2 / 48000
    assert report["client_first_pcm_seconds"] == .2
    assert report["pcm_bytes"] == 4
    assert report["required_extra_initial_buffer_seconds"] == 0


def test_complete_response_requires_nonempty_even_pcm_and_ordered_times():
    wire = PCMReceipt()
    with pytest.raises(ValueError, match="Empty"):
        wire.finish(0)
    wire.receive(b"\x01", 1)
    with pytest.raises(ValueError, match="incomplete"):
        wire.finish(1)
    with pytest.raises(ValueError, match="ordered"):
        wire.receive(b"\x02", .5)
    wire.receive(b"\x02", 2)
    with pytest.raises(ValueError, match="precedes"):
        wire.finish(1)
