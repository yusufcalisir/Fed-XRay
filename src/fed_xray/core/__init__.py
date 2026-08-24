"""
Fed-XRay Core Federated Learning Engine
"""

from .client import HospitalClient
from .server import CentralServer, run_federated_round
from .metrics import TrainingMetrics, EvaluationMetrics, SecurityReport
from .imbalance_losses import (
    DynamicAdaptiveFocalLoss,
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
    LDAMLoss,
    PrototypeRepelLoss
)
from .prototypes import (
    extract_features,
    compute_local_prototypes_and_dispersion,
    aggregate_prototypes_dispersion_weighted,
    compute_prototype_distance_loss
)
