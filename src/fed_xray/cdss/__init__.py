"""
Fed-XRay Clinical Decision Support System (CDSS) Engines
"""

from .xai import GradCAM, create_overlay, get_explanation_text
from .similarity import HistoricalCaseBank, extract_embedding, LABEL_NAMES, LABEL_COLORS
from .voice import generate_diagnosis_audio, get_or_create_audio
from .report import generate_medical_report, get_diagnosis_explanation
from .i18n import get_text, get_all_texts
