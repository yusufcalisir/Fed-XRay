"""
Fed-XRay: Federated Medical Imaging & Oncology AI Platform
===========================================================
State-of-the-Art Privacy-Preserving, Personalized, and Multimodal
Federated Learning Systems for Medical, Radiological, and Cancer Imaging.
"""

__version__ = "2.0.0"
__author__ = "Fed-XRay Engineering Consortium"

from .models.cnn import XRayClassifier, create_model, count_parameters
from .core.client import HospitalClient
from .core.server import CentralServer
from .core.metrics import TrainingMetrics, EvaluationMetrics, SecurityReport
