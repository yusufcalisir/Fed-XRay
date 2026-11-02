"""Fed-XRay FastAPI Request & Response Schemas (Pydantic v2)."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class CohortGenerateRequest(BaseModel):
    num_hospitals: int = Field(default=4, ge=2, le=8)
    samples_per_hospital: int = Field(default=200, ge=50, le=500)
    scenario: str = Field(default="A", description="Imbalance Scenario: A (IID), B (Mild), C (Mod), D (Severe), E (Missing), F (Long-Tail), G (Combined)")
    dataset_name: str = Field(default="ISIC_2019", description="Medical Dataset: ISIC_2019, CRC_HISTO, MIMIC_CXR")
    manifest_name: Optional[str] = Field(
        default=None,
        description="CSV/TSV manifest relative to the server FED_XRAY_DATA_DIR. When omitted, a labelled synthetic phantom cohort is generated.",
    )
    images_dir_name: Optional[str] = Field(
        default=None, description="Image directory relative to FED_XRAY_DATA_DIR (defaults to the manifest directory)."
    )
    image_size: Optional[int] = Field(
        default=None, ge=28, le=512, description="Target image dimension (defaults to 224 for Strategy E disk manifests)."
    )


class HospitalCohortInfo(BaseModel):
    hospital_id: int
    name: str
    num_samples: int
    distribution: List[float]
    counts: Dict[str, int]
    sample_images: List[List[List[float]]]
    sample_labels: List[int]
    provenance: Optional[str] = None


class CohortGenerateResponse(BaseModel):
    success: bool
    num_hospitals: int
    total_samples: int
    data_source: str
    provenance: Optional[str] = None
    hospitals: List[HospitalCohortInfo]


class TrainFLRequest(BaseModel):
    num_rounds: int = Field(default=5, ge=1, le=20)
    local_epochs: int = Field(default=2, ge=1, le=5)
    learning_rate: float = Field(default=0.0001, ge=0.00001, le=0.01)
    privacy_noise: float = Field(default=0.01, ge=0.0, le=0.1)
    simulate_attack: bool = Field(default=False)
    activate_defense: bool = Field(default=True)
    algorithm: str = Field(default="FedAvg", description="FedAvg, FedProx, FedDyn, FedOpt, SCAFFOLD, MOON")
    loss_fn: str = Field(default="CE", description="CE, DAFL, BSM, LDAM, CB")
    model_type: str = Field(default="vit_tiny", description="vit_tiny, vit_small, uni, swin_unetr, cnn")
    peft_mode: Optional[str] = Field(default="ffa_lora", description="ffa_lora, fedsa_lora, lora, None")
    secure_aggregation: bool = Field(default=False, description="Enable SecAgg+ pairwise zero-sum random masking")
    use_ckks: bool = Field(default=False, description="Enable Leveled CKKS Threshold Homomorphic Encryption")


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
    model_type: str = "vit_tiny"
    peft_mode: str = "ffa_lora"
    privacy_epsilon: float = 0.0
    uplink_bytes_per_client: int = 0
    full_model_bytes: int = 0


class DiagnoseRequest(BaseModel):
    class_index: Optional[int] = None
    image_data: Optional[List[List[float]]] = None
    image_base64: Optional[str] = None
    opacity: float = Field(default=0.55, ge=0.0, le=1.0)
    colormap: str = Field(default="Hot")


class DiagnoseResponse(BaseModel):
    patient_id: str = "PAT_UNKNOWN"
    predicted_class: int
    predicted_name: str
    true_class: int
    true_name: str
    confidence: float
    probabilities: List[float]
    findings: str
    raw_image: List[List[float]]
    heatmap: Optional[List[List[float]]] = None
    heatmap_available: bool = True
    heatmap_error: Optional[str] = None
    image_source: str = "uploaded"
    provenance: Optional[str] = None
    anatomical_site: Optional[str] = None
    biopsy_confirmed: Optional[bool] = None


class RagTwinCase(BaseModel):
    case_id: str
    label_id: int
    label_name: str
    similarity: float
    history: str
    biopsy_confirmed: Optional[bool] = None
    therapeutic_outcome: Optional[str] = None
    anatomical_site: Optional[str] = None
    site_id: Optional[str] = None
    provenance: Optional[str] = None


class RagSimilarResponse(BaseModel):
    query_class: int
    matched_cases: List[RagTwinCase]
