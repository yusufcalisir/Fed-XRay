"""
Fed-XRay: Federated Medical Imaging & Oncology AI Platform
===========================================================
State-of-the-Art Privacy-Preserving, Personalized, and Multimodal
Federated Learning Systems for Medical, Radiological, and Cancer Imaging.
"""

from typing import Any

__version__ = "2.0.0"
__author__ = "Fed-XRay Engineering Consortium"

__all__ = [
    "XRayClassifier",
    "create_model",
    "count_parameters",
    "HospitalClient",
    "CentralServer",
    "TrainingMetrics",
    "EvaluationMetrics",
    "SecurityReport",
]


def __getattr__(name: str) -> Any:
    """Lazy-load package members to keep import-time memory under 150 MB."""
    if name in ("XRayClassifier", "create_model", "count_parameters"):
        from .models.cnn import XRayClassifier, create_model, count_parameters
        return locals()[name]
    if name == "HospitalClient":
        from .core.client import HospitalClient
        return HospitalClient
    if name == "CentralServer":
        from .core.server import CentralServer
        return CentralServer
    if name in ("TrainingMetrics", "EvaluationMetrics", "SecurityReport"):
        from .core.metrics import TrainingMetrics, EvaluationMetrics, SecurityReport
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

