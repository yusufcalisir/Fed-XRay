"""
Fed-XRay: Voice Assistant Engine
================================
Text-to-Speech for diagnosis announcements using gTTS.
Provides audio feedback for clinical decision support.
"""

import io
import tempfile
import os
from typing import Optional
import hashlib


def generate_diagnosis_audio(
    diagnosis: str,
    confidence: float,
    language: str = 'en'
) -> Optional[bytes]:
    """
    Generate audio announcement for diagnosis.
    
    Args:
        diagnosis: Predicted class name (Normal, Pneumonia, COVID-19)
        confidence: Confidence percentage (0-100)
        language: Language code ('en' for English)
        
    Returns:
        Audio data as bytes (MP3 format), or None if gTTS fails
    """
    try:
        from gtts import gTTS
        
        # Create diagnosis message
        confidence_text = f"{confidence:.0f} percent" if confidence > 0 else "uncertain"
        
        if diagnosis == "Normal":
            message = f"Analysis complete. The scan appears normal with {confidence_text} confidence. No abnormalities detected."
        elif diagnosis == "Pneumonia":
            message = f"Analysis complete. High probability of Pneumonia. Confidence: {confidence_text}. Please review the highlighted regions."
        elif diagnosis == "COVID-19":
            message = f"Analysis complete. Potential COVID-19 indicators detected with {confidence_text} confidence. Recommend further testing."
        else:
            message = f"Analysis complete. Diagnosis: {diagnosis}. Confidence: {confidence_text}."
        
        # Generate speech
        tts = gTTS(text=message, lang=language, slow=False)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
        
    except Exception as e:
        print(f"[Voice Engine Error] {e}")
        return None


def get_cached_audio_path(diagnosis: str, confidence: float) -> str:
    """
    Generate a cache key path for audio file.
    
    Args:
        diagnosis: Predicted class name
        confidence: Confidence percentage
        
    Returns:
        Path to cached audio file
    """
    # Create hash from diagnosis and confidence
    key = f"{diagnosis}_{int(confidence)}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:8]
    
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, f"fedxray_voice_{hash_key}.mp3")


def get_or_create_audio(
    diagnosis: str,
    confidence: float,
    use_cache: bool = True
) -> Optional[bytes]:
    """
    Get cached audio or create new one.
    
    Args:
        diagnosis: Predicted class name
        confidence: Confidence percentage
        use_cache: Whether to use caching
        
    Returns:
        Audio data as bytes
    """
    if use_cache:
        cache_path = get_cached_audio_path(diagnosis, confidence)
        
        # Check if cached file exists
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except:
                pass
    
    # Generate new audio
    audio_data = generate_diagnosis_audio(diagnosis, confidence)
    
    # Cache it
    if audio_data and use_cache:
        try:
            cache_path = get_cached_audio_path(diagnosis, confidence)
            with open(cache_path, 'wb') as f:
                f.write(audio_data)
        except:
            pass
    
    return audio_data
