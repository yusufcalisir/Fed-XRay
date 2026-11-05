"""Models package for Federated Medical Imaging."""

from src.fed_xray.models.cnn import (
    XRayClassifier,
    create_model,
    count_parameters,
)
from src.fed_xray.models.vit import (
    VisionTransformer,
    PatchEmbedding,
    MultiHeadSelfAttention,
    TransformerEncoderBlock,
    create_medical_vit,
)
from src.fed_xray.models.peft import (
    LoRALinear,
    FFALoRALinear,
    FedSALoRALinear,
    inject_lora_to_model,
    extract_peft_state_dict,
    load_peft_state_dict,
)
from src.fed_xray.models.foundation import (
    UNIFoundationBackbone,
    SwinUNETRRadiologyBackbone,
    ensure_foundation_checkpoints,
)

__all__ = [
    "XRayClassifier",
    "create_model",
    "count_parameters",
    "VisionTransformer",
    "PatchEmbedding",
    "MultiHeadSelfAttention",
    "TransformerEncoderBlock",
    "create_medical_vit",
    "LoRALinear",
    "FFALoRALinear",
    "FedSALoRALinear",
    "inject_lora_to_model",
    "extract_peft_state_dict",
    "load_peft_state_dict",
    "UNIFoundationBackbone",
    "SwinUNETRRadiologyBackbone",
    "ensure_foundation_checkpoints",
]
