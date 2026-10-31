"""
Fed-XRay Voice Engine (Backward-Compatibility Shim)
===================================================
Re-exports TTS audio utilities for synchronous/Streamlit callers.
"""

import os
from typing import Optional
from src.fed_xray.cdss.voice import (
    generate_diagnosis_audio,
    get_cached_audio_path,
)


def get_or_create_audio(
    diagnosis: str,
    confidence: float,
    use_cache: bool = True,
    language: str = "en",
) -> Optional[bytes]:
    """Get cached audio or generate new one synchronously for Streamlit/legacy callers."""
    cache_path = get_cached_audio_path(diagnosis, confidence)
    if use_cache and cache_path:
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return f.read()
        except OSError:
            pass

    audio_data = generate_diagnosis_audio(diagnosis, confidence, language=language)
    if audio_data and use_cache and cache_path:
        try:
            with open(cache_path, "wb") as f:
                f.write(audio_data)
        except OSError:
            pass
    return audio_data


__all__ = ["generate_diagnosis_audio", "get_cached_audio_path", "get_or_create_audio"]
