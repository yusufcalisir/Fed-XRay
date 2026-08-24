"""
Fed-XRay CNN Model (Backward-Compatibility Shim)
================================================
Re-exports model classes and helpers from `src.fed_xray.models.cnn`.
"""

from src.fed_xray.models.cnn import (
    XRayClassifier,
    create_model,
    count_parameters
)

__all__ = ["XRayClassifier", "create_model", "count_parameters"]
