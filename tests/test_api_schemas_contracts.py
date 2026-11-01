"""Granular Unit Test Suite: API Request & Response Schemas (Pydantic v2).

Verifies payload validation, field constraints, default values,
and cross-tier serialization contracts.
"""

from __future__ import annotations
import pytest
from pydantic import ValidationError
from api.schemas import (
    CohortGenerateRequest,
    TrainFLRequest,
    DiagnoseRequest,
    DiagnoseResponse,
)


def test_cohort_generate_request_valid():
    """Verify valid cohort generation request parses properly with defaults."""
    req = CohortGenerateRequest(num_hospitals=4, samples_per_hospital=200)
    assert req.num_hospitals == 4
    assert req.samples_per_hospital == 200
    assert req.dataset_name == "ISIC_2019"


def test_cohort_generate_request_out_of_bounds():
    """Verify validation error when hospital counts or sample sizes violate bounds."""
    with pytest.raises(ValidationError):
        CohortGenerateRequest(num_hospitals=1)  # ge=2

    with pytest.raises(ValidationError):
        CohortGenerateRequest(num_hospitals=10)  # le=8

    with pytest.raises(ValidationError):
        CohortGenerateRequest(samples_per_hospital=20)  # ge=50


def test_train_fl_request_constraints():
    """Verify federated training hyperparameters enforce strict bounds."""
    valid_req = TrainFLRequest(
        num_rounds=5,
        local_epochs=2,
        learning_rate=0.0001,
        model_type="vit_base",
        peft_mode="ffa_lora",
    )
    assert valid_req.num_rounds == 5
    assert valid_req.model_type == "vit_base"

    with pytest.raises(ValidationError):
        TrainFLRequest(num_rounds=0)  # ge=1

    with pytest.raises(ValidationError):
        TrainFLRequest(learning_rate=0.5)  # le=0.01


def test_diagnose_request_opacity_bounds():
    """Verify Grad-CAM overlay opacity is strictly bounded in [0.0, 1.0]."""
    req = DiagnoseRequest(opacity=0.75, colormap="Jet")
    assert req.opacity == 0.75
    assert req.colormap == "Jet"

    with pytest.raises(ValidationError):
        DiagnoseRequest(opacity=-0.1)

    with pytest.raises(ValidationError):
        DiagnoseRequest(opacity=1.5)


def test_diagnose_response_serialization():
    """Verify complete diagnostic verdict serialization."""
    resp = DiagnoseResponse(
        patient_id="PAT-TEST-99",
        predicted_class=1,
        predicted_name="Pneumonia",
        true_class=1,
        true_name="Pneumonia",
        confidence=96.4,
        probabilities=[0.02, 0.964, 0.016],
        findings="Alveolar consolidation in inferior lobe.",
        raw_image=[[0.1, 0.2], [0.3, 0.4]],
        heatmap=[[0.8, 0.9], [0.7, 0.6]],
        heatmap_available=True,
    )

    data = resp.model_dump()
    assert data["patient_id"] == "PAT-TEST-99"
    assert data["predicted_class"] == 1
    assert data["confidence"] == 96.4
    assert len(data["probabilities"]) == 3
