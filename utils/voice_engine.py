"""
Fed-XRay Voice Engine (Backward-Compatibility Shim)
===================================================
Re-exports TTS audio utilities from `src.fed_xray.cdss.voice`.
"""

from src.fed_xray.cdss.voice import (
    generate_diagnosis_audio,
    get_cached_audio_path,
    get_or_create_audio
)

__all__ = ["generate_diagnosis_audio", "get_cached_audio_path", "get_or_create_audio"]
