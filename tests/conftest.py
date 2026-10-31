"""Shared API fixtures: the CDSS endpoints require a loaded cohort and a trained model."""

import pytest
from fastapi.testclient import TestClient

from api.main import app, state


@pytest.fixture()
def trained_api() -> TestClient:
    """Return a client whose backend has a loaded cohort and a model trained for one federated round."""
    client = TestClient(app)
    if not state.model_trained:
        res = client.post(
            "/api/cohorts/generate",
            json={"num_hospitals": 4, "samples_per_hospital": 100, "scenario": "A", "dataset_name": "ISIC_2019"},
        )
        assert res.status_code == 200
        stream = client.get(
            "/api/fl/train-stream?num_rounds=1&local_epochs=1&learning_rate=0.001"
            "&algorithm=FedDyn&loss_fn=DAFL&model_type=vit_tiny&peft_mode=ffa_lora"
        )
        assert stream.status_code == 200
        assert state.model_trained
    return client


@pytest.fixture()
def diagnosed_api(trained_api: TestClient) -> TestClient:
    """Return a trained client that has already diagnosed one held-out scan."""
    res = trained_api.post("/api/cdss/diagnose", json={})
    assert res.status_code == 200
    return trained_api
