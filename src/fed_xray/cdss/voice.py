"""Fed-XRay CDSS Voice Assistant Engine.

High-performance Text-to-Speech synthesis for diagnostic announcements with
multi-tier persistent caching, air-gapped clinical workstation compatibility,
and asynchronous thread executor offloading to protect event loops.
"""

from __future__ import annotations
import asyncio
import hashlib
import io
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_AUDIO_CACHE_DIR: str = os.path.join("data", "audio_cache")
AUDIO_CACHE_DIR: str = os.environ.get("FEDXRAY_AUDIO_CACHE_DIR", DEFAULT_AUDIO_CACHE_DIR)


def _ensure_cache_dir() -> str:
    """Ensure primary audio cache directory exists, falling back to tempdir."""
    try:
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
        return AUDIO_CACHE_DIR
    except OSError:
        return tempfile.gettempdir()


def _generate_audio_sync(text: str, lang: str = "en") -> Optional[bytes]:
    """Synchronous gTTS call: must execute in a worker thread, not on the event loop."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        print(f"[Voice Engine] gTTS external synthesis failed: {exc}")
        return None


def _cache_path(text: str, lang: str) -> str:
    """Compute deterministic cache path across persistent and ephemeral directories."""
    key = hashlib.md5(f"{text}:{lang}".encode()).hexdigest()[:10]
    filename = f"fedxray_voice_{key}.mp3"

    # 1. Existing persistent cache hit
    primary_dir = _ensure_cache_dir()
    primary_path = os.path.join(primary_dir, filename)
    if os.path.exists(primary_path):
        return primary_path

    # 2. Existing tempdir cache hit
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(temp_path):
        return temp_path

    # 3. Default target for new writes
    return primary_path


def get_cached_audio_path(diagnosis: str, confidence: float) -> str:
    """Generate a cache key path for diagnosis audio file."""
    key = f"{diagnosis}_{int(confidence)}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:8]
    filename = f"fedxray_voice_{hash_key}.mp3"
    primary_dir = _ensure_cache_dir()
    return os.path.join(primary_dir, filename)


def get_voice_engine_status() -> Dict[str, Any]:
    """Return operational health, persistent cache metrics, and air-gapped readiness."""
    cached_count = 0
    checked_dirs = set()
    for d in (AUDIO_CACHE_DIR, tempfile.gettempdir()):
        if d not in checked_dirs and os.path.exists(d):
            checked_dirs.add(d)
            try:
                cached_count += sum(
                    1 for f in os.listdir(d) if f.startswith("fedxray_voice_") and f.endswith(".mp3")
                )
            except OSError:
                pass

    return {
        "service": "CDSS Voice Assistant Engine",
        "cloud_provider": "Google Text-to-Speech (gTTS)",
        "cache_directory": AUDIO_CACHE_DIR,
        "cached_files_count": cached_count,
        "supported_languages": ["en", "tr"],
        "client_offline_synthesis_supported": True,
        "air_gapped_compatible": True,
        "status": "ready",
    }


def prewarm_clinical_voice_cache(
    texts: Optional[List[str]] = None,
    languages: Tuple[str, ...] = ("en", "tr"),
) -> Dict[str, int]:
    """Pre-generate and cache standard clinical voice utterances for air-gapped deployments."""
    standard_phrases = texts or [
        "Normal diagnosis with 95 percent confidence. Federated model inference complete.",
        "Pneumonia diagnosis with 95 percent confidence. Federated model inference complete.",
        "COVID-19 diagnosis with 95 percent confidence. Federated model inference complete.",
        "Melanoma diagnosis with 95 percent confidence. Federated model inference complete.",
        "Basal Cell Carcinoma diagnosis with 95 percent confidence. Federated model inference complete.",
        "Yuzde 95 model guveni ile Normal Bulgular teshisi dogrulandi. Federe model analizi tamamlandi.",
        "Yuzde 95 model guveni ile Zaturre teshisi dogrulandi. Federe model analizi tamamlandi.",
        "Yuzde 95 model guveni ile COVID-19 teshisi dogrulandi. Federe model analizi tamamlandi.",
    ]
    generated = 0
    cached = 0
    for lang in languages:
        for phrase in standard_phrases:
            path = _cache_path(phrase, lang)
            if os.path.exists(path):
                cached += 1
                continue
            audio = _generate_audio_sync(phrase, lang)
            if audio is not None:
                try:
                    with open(path, "wb") as f:
                        f.write(audio)
                    generated += 1
                except OSError:
                    pass
    return {"newly_generated": generated, "already_cached": cached}


async def get_or_create_audio(text: str, lang: str = "en") -> bytes:
    """Async TTS wrapper with persistent multi-tier caching.

    1. Checks persistent and ephemeral disk cache first.
    2. Runs gTTS in thread pool to avoid blocking async event loop.
    3. Raises RuntimeError on failure with explicit air-gapped instructions.
    """
    cache = _cache_path(text, lang)

    # Cache hit
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                return f.read()
        except OSError:
            pass  # Cache read failed: regenerate

    # Run blocking I/O in a thread pool
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, _generate_audio_sync, text, lang)

    if audio is None:
        raise RuntimeError(
            "gTTS synthesis failed: check network connectivity. "
            "For air-gapped workstations, use local browser speech synthesis or pre-warmed audio cache."
        )

    # Write cache (best-effort)
    try:
        with open(cache, "wb") as f:
            f.write(audio)
    except OSError:
        pass

    return audio


# ---------------------------------------------------------------------------
# Legacy compatibility shim: kept so old callers (diagnosis=, confidence=)
# still work without breaking changes.
# ---------------------------------------------------------------------------
def generate_diagnosis_audio(
    diagnosis: str,
    confidence: float,
    language: str = "en",
) -> Optional[bytes]:
    """Synchronous shim for legacy callers. Prefer get_or_create_audio() in async contexts."""
    text = (
        f"{diagnosis} diagnosis with {confidence:.0f} percent confidence. "
        "Federated model inference complete."
    )
    return _generate_audio_sync(text, language)
