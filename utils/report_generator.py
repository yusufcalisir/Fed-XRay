"""
Fed-XRay PDF Report Generator (Backward-Compatibility Shim)
===========================================================
Re-exports report generator and explanation functions from `src.fed_xray.cdss.report`.
"""

from src.fed_xray.cdss.report import (
    generate_medical_report,
    get_diagnosis_explanation
)

__all__ = ["generate_medical_report", "get_diagnosis_explanation"]
