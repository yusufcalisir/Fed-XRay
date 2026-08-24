"""
Fed-XRay Voice Assistant Engine
================================
Text-to-Speech for diagnosis announcements using gTTS.
"""

import io
import tempfile
import os
import hashlib
from typing import Optional


def generate_diagnosis_audio(
    diagnosis: str,
    confidence: float,
    language: str = 'en'
) -> Optional[bytes]:
    """Generate audio announcement for clinical diagnosis."""
    try:
        from gtts import gTTS
        
        confidence_text = f"{confidence:.0f} percent" if confidence > 0 else "uncertain"
        
        if diagnosis == "Normal":
            message = f"Analysis complete. The scan appears normal with {confidence_text} confidence. No abnormalities detected."
        elif diagnosis == "Pneumonia":
            message = f"Analysis complete. High probability of Pneumonia. Confidence: {confidence_text}. Please review the highlighted regions."
        elif diagnosis == "COVID-19":
            message = f"Analysis complete. Potential COVID-19 indicators detected with {confidence_text} confidence. Recommend further testing."
        else:
            message = f"Analysis complete. Diagnosis: {diagnosis}. Confidence: {confidence_text}."
        
        tts = gTTS(text=message, lang=language, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
        
    except Exception as e:
        print(f"[Voice Engine Error] {e}")
        return None


def get_cached_audio_path(diagnosis: str, confidence: float) -> str:
    """Generate a cache key path for audio file."""
    key = f"{diagnosis}_{int(confidence)}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:8]
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, f"fedxray_voice_{hash_key}.mp3")


def get_or_create_audio(
    diagnosis: str,
    confidence: float,
    use_cache: bool = True
) -> Optional[bytes]:
    """Get cached audio or synthesize new audio stream."""
    if use_cache:
        cache_path = get_cached_audio_path(diagnosis, confidence)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except:
                pass
    
    audio_data = generate_diagnosis_audio(diagnosis, confidence)
    
    if audio_data and use_cache:
        try:
            cache_path = get_cached_audio_path(diagnosis, confidence)
            with open(cache_path, 'wb') as f:
                f.write(audio_data)
        except:
            pass
    
    return audio_data
