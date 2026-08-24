"""
Fed-XRay Core Federated Learning Engine
"""

from .client import HospitalClient
from .server import CentralServer, run_federated_round
from .metrics import TrainingMetrics, EvaluationMetrics, SecurityReport
from .algorithms import (
    compute_fedprox_loss,
    compute_feddyn_loss,
    compute_moon_contrastive_loss,
    ScaffoldController
)
