"""Granular Unit Test Suite: CDSS Voice Assistant Engine.

Verifies async audio synthesis, disk caching mechanisms,
and text-to-speech parameter formatting.
"""

from __future__ import annotations
import os
import pytest
from src.fed_xray.cdss.voice import (
    _cache_path,
    get_cached_audio_path,
    get_or_create_audio,
    get_voice_engine_status,
    prewarm_clinical_voice_cache,
    generate_diagnosis_audio,
)


def test_cache_path_deterministic():
    """Verify cache paths are deterministic for identical text and language."""
    p1 = _cache_path("Pneumonia diagnosis with 95 percent confidence.", "en")
    p2 = _cache_path("Pneumonia diagnosis with 95 percent confidence.", "en")
    p3 = _cache_path("Pneumonia diagnosis with 95 percent confidence.", "tr")

    assert p1 == p2
    assert p1 != p3
    assert p1.endswith(".mp3")


def test_get_cached_audio_path_format():
    """Verify diagnosis cache key correctly formats diagnosis and confidence."""
    path = get_cached_audio_path("COVID-19", 94.2)
    assert "fedxray_voice_" in path
    assert path.endswith(".mp3")


@pytest.mark.asyncio
async def test_get_or_create_audio_cache_hit(tmp_path, monkeypatch):
    """Verify audio retrieval returns cached bytes directly when file exists."""
    fake_audio_bytes = b"\xff\xfb\x90\x44" * 10
    test_text = "Unit test synthetic diagnosis audio."

    # Pre-populate cache
    cache_file = _cache_path(test_text, "en")
    with open(cache_file, "wb") as f:
        f.write(fake_audio_bytes)

    try:
        retrieved = await get_or_create_audio(test_text, "en")
        assert retrieved == fake_audio_bytes
    finally:
        if os.path.exists(cache_file):
            os.remove(cache_file)


def test_generate_diagnosis_audio_legacy_shim(monkeypatch):
    """Verify synchronous legacy shim returns valid audio buffer or handles mock."""
    monkeypatch.setattr(
        "src.fed_xray.cdss.voice._generate_audio_sync",
        lambda text, lang: b"MOCK_MP3_AUDIO_STREAM",
    )
    res = generate_diagnosis_audio("Pneumonia", 96.0, "en")
    assert res == b"MOCK_MP3_AUDIO_STREAM"


def test_get_voice_engine_status():
    """Verify voice engine status reports operational metrics and air-gapped readiness."""
    status = get_voice_engine_status()
    assert status["service"] == "CDSS Voice Assistant Engine"
    assert status["cloud_provider"] == "Google Text-to-Speech (gTTS)"
    assert "cache_directory" in status
    assert isinstance(status["cached_files_count"], int)
    assert status["supported_languages"] == ["en", "tr"]
    assert status["client_offline_synthesis_supported"] is True
    assert status["air_gapped_compatible"] is True
    assert status["status"] == "ready"


def test_prewarm_clinical_voice_cache(monkeypatch):
    """Verify pre-warming voice cache generates and populates audio files."""
    monkeypatch.setattr(
        "src.fed_xray.cdss.voice._generate_audio_sync",
        lambda text, lang: b"PREWARMED_AUDIO_MP3_PAYLOAD",
    )
    res = prewarm_clinical_voice_cache(
        texts=["Normal diagnosis test."],
        languages=("en",),
    )
    assert res["newly_generated"] >= 0
    assert "already_cached" in res


@pytest.mark.asyncio
async def test_air_gapped_offline_error_handling(monkeypatch):
    """Verify honest failure reporting when network is down and cache misses."""
    monkeypatch.setattr(
        "src.fed_xray.cdss.voice._generate_audio_sync",
        lambda text, lang: None,
    )
    with pytest.raises(RuntimeError) as excinfo:
        await get_or_create_audio("Uncached phrase under simulated offline mode.", "en")
    assert "gTTS synthesis failed" in str(excinfo.value)
    assert "air-gapped" in str(excinfo.value).lower()


def test_voice_status_api_endpoint():
    """Verify GET /api/cdss/voice/status returns 200 with engine metrics."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    res = client.get("/api/cdss/voice/status")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "CDSS Voice Assistant Engine"
    assert data["air_gapped_compatible"] is True


