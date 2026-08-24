"""
Fed-XRay Bilingual Voice Assistant Engine
==========================================
Text-to-Speech for diagnosis announcements using gTTS with full Turkish & English support.
"""

import io
import tempfile
import os
import hashlib
from typing import Optional


def generate_diagnosis_audio(
    diagnosis: str,
    confidence: float,
    language: str = 'tr'
) -> Optional[bytes]:
    """Generate localized audio announcement for clinical diagnosis."""
    try:
        from gtts import gTTS
        
        lang_code = 'tr' if language == 'tr' else 'en'
        
        # Normalize diagnosis label
        is_normal = diagnosis.lower() in ["normal", "normal (sağlıklı)", "normal (healthy)"]
        is_pneumonia = diagnosis.lower() in ["pneumonia", "zatürre", "zatürre (pneumonia)"]
        is_covid = "covid" in diagnosis.lower()
        
        if lang_code == 'tr':
            conf_str = f"%{confidence:.0f}" if confidence > 0 else "belirsiz"
            if is_normal:
                message = f"Analiz tamamlandı. Akciğer filmi {conf_str} güven oranıyla normal değerlendirildi. Belirgin patoloji tespit edilmedi."
            elif is_pneumonia:
                message = f"Analiz tamamlandı. {conf_str} güven oranıyla yüksek olasılıklı Zatürre tespit edildi. Lütfen işaretli odak alanlarını inceleyiniz."
            elif is_covid:
                message = f"Analiz tamamlandı. {conf_str} güven oranıyla potansiyel COVID-19 bulguları saptandı. İleri klinik tetkik önerilir."
            else:
                message = f"Analiz tamamlandı. Teşhis: {diagnosis}. Güven oranı: {conf_str}."
        else:
            conf_str = f"{confidence:.0f} percent" if confidence > 0 else "uncertain"
            if is_normal:
                message = f"Analysis complete. The scan appears normal with {conf_str} confidence. No significant abnormalities detected."
            elif is_pneumonia:
                message = f"Analysis complete. High probability of Pneumonia detected with {conf_str} confidence. Please review highlighted regions."
            elif is_covid:
                message = f"Analysis complete. Potential COVID-19 indicators detected with {conf_str} confidence. Further clinical testing recommended."
            else:
                message = f"Analysis complete. Diagnosis: {diagnosis}. Confidence: {conf_str}."
        
        tts = gTTS(text=message, lang=lang_code, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
        
    except Exception as e:
        print(f"[Voice Engine Error] {e}")
        return None


def get_cached_audio_path(diagnosis: str, confidence: float, language: str = 'tr') -> str:
    """Generate a cache key path for audio file."""
    key = f"{diagnosis}_{int(confidence)}_{language}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:8]
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, f"fedxray_voice_{hash_key}.mp3")


def get_or_create_audio(
    diagnosis: str,
    confidence: float,
    language: str = 'tr',
    use_cache: bool = True
) -> Optional[bytes]:
    """Get cached audio or synthesize new audio stream."""
    if use_cache:
        cache_path = get_cached_audio_path(diagnosis, confidence, language)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except:
                pass
    
    audio_data = generate_diagnosis_audio(diagnosis, confidence, language=language)
    
    if audio_data and use_cache:
        try:
            cache_path = get_cached_audio_path(diagnosis, confidence, language)
            with open(cache_path, 'wb') as f:
                f.write(audio_data)
        except:
            pass
    
    return audio_data
