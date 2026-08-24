"""
Fed-XRay Explainable AI Engine (Backward-Compatibility Shim)
============================================================
Re-exports Grad-CAM and heatmap overlay utilities from `src.fed_xray.cdss.xai`.
"""

from src.fed_xray.cdss.xai import (
    GradCAM,
    create_overlay,
    get_explanation_text
)

__all__ = ["GradCAM", "create_overlay", "get_explanation_text"]
