"""
Fed-XRay FastAPI Request & Response Schemas (Pydantic v2)
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class CohortGenerateRequest(BaseModel):
    num_hospitals: int = Field(default=4, ge=2, le=10)
    samples_per_hospital: int = Field(default=200, ge=50, le=500)


class HospitalCohortInfo(BaseModel):
    hospital_id: int
    name: str
    num_samples: int
    distribution: List[float]
    counts: Dict[str, int]
    sample_images: List[List[List[float]]]
    sample_labels: List[int]


class CohortGenerateResponse(BaseModel):
    success: bool
    num_hospitals: int
    total_samples: int
    hospitals: List[HospitalCohortInfo]


class TrainFLRequest(BaseModel):
    num_rounds: int = Field(default=5, ge=1, le=20)
    local_epochs: int = Field(default=2, ge=1, le=5)
    learning_rate: float = Field(default=0.0001, ge=0.00001, le=0.01)
    privacy_noise: float = Field(default=0.01, ge=0.0, le=0.1)
    simulate_attack: bool = Field(default=False)
    activate_defense: bool = Field(default=True)
    algorithm: str = Field(default="FedAvg")
    loss_fn: str = Field(default="CE")


class RoundTelemetryUpdate(BaseModel):
    round_num: int
    total_rounds: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    precision: float
    recall: float
    f1_score: float
    threat_detected: bool
    blocked_nodes: List[int]
    status: str


class DiagnoseRequest(BaseModel):
    class_index: Optional[int] = None
    opacity: float = Field(default=0.55, ge=0.0, le=1.0)
    colormap: str = Field(default="Hot")


class DiagnoseResponse(BaseModel):
    predicted_class: int
    predicted_name: str
    true_class: int
    true_name: str
    confidence: float
    probabilities: List[float]
    findings: str
    raw_image: List[List[float]]
    heatmap: List[List[float]]


class RagTwinCase(BaseModel):
    case_id: str
    label_id: int
    label_name: str
    similarity: float
    history: str


class RagSimilarResponse(BaseModel):
    query_class: int
    matched_cases: List[RagTwinCase]
