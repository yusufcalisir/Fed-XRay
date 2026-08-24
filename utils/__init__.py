"""
Fed-XRay Backward Compatibility Re-Export Layer
================================================
Transparently re-exports modules and classes from the modular `src.fed_xray` package.
"""

from src.fed_xray.models.cnn import XRayClassifier, create_model, count_parameters
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.core.metrics import TrainingMetrics, EvaluationMetrics, SecurityReport
