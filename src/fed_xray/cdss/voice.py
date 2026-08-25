"""
Fed-XRay Voice Assistant Engine
================================
Text-to-Speech for diagnosis announcements using gTTS.
Runs gTTS in a thread executor to avoid blocking the async event loop.
"""

import asyncio
import io
import os
import hashlib
import tempfile
from typing import Optional


def _generate_audio_sync(text: str, lang: str = "en") -> Optional[bytes]:
    """Synchronous gTTS call — must run in a thread, not on the event loop."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        print(f"[Voice Engine] gTTS failed: {exc}")
        return None


def _cache_path(text: str, lang: str) -> str:
    key = hashlib.md5(f"{text}:{lang}".encode()).hexdigest()[:10]
    return os.path.join(tempfile.gettempdir(), f"fedxray_voice_{key}.mp3")


async def get_or_create_audio(text: str, lang: str = "en") -> bytes:
    """
    Async TTS wrapper.
    - Checks disk cache first (survives server restarts within same Render instance).
    - Runs gTTS in a thread pool so the FastAPI event loop is never blocked.
    - Raises RuntimeError on failure so the caller can return 503.
    """
    cache = _cache_path(text, lang)

    # Cache hit
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                return f.read()
        except OSError:
            pass  # Cache read failed — regenerate

    # Run blocking I/O in a thread pool
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, _generate_audio_sync, text, lang)

    if audio is None:
        raise RuntimeError("gTTS synthesis failed — check network connectivity")

    # Write cache (best-effort)
    try:
        with open(cache, "wb") as f:
            f.write(audio)
    except OSError:
        pass

    return audio


# ---------------------------------------------------------------------------
# Legacy compatibility shim — kept so old callers (diagnosis=, confidence=)
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
