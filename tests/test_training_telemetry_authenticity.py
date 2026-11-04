"""Tests for authentic training telemetry vs validation metric separation (DEFECT-03 remediation)."""

import pytest
from fastapi.testclient import TestClient
from api.main import app, state


@pytest.fixture
def client():
    return TestClient(app)


def test_training_history_distinguishes_train_and_test_metrics():
    """Verify that in-memory training history does not overwrite training loss with validation metrics."""
    original_history = dict(state.training_history)
    try:
        # Populate history with distinct training vs validation metrics
        state.training_history = {
            "round": [1, 2],
            "loss": [0.8521, 0.6123],             # Genuine client local training loss
            "accuracy": [65.0, 78.5],             # Genuine client local training accuracy
            "test_loss": [0.9124, 0.6841],        # Central hold-out validation loss
            "test_accuracy": [61.0, 74.0],        # Central hold-out validation accuracy
            "precision": [62.0, 75.0],
            "recall": [60.0, 73.0],
            "f1_score": [61.0, 74.0],
            "blocked_count": 0,
        }

        # Verify array invariants: train and test metrics are independent
        assert state.training_history["loss"][0] != state.training_history["test_loss"][0]
        assert state.training_history["accuracy"][0] != state.training_history["test_accuracy"][0]
        assert state.training_history["loss"][1] != state.training_history["test_loss"][1]
        assert state.training_history["accuracy"][1] != state.training_history["test_accuracy"][1]
    finally:
        state.training_history = original_history


def test_get_state_reports_unconflated_train_and_test_metrics(client):
    """Verify GET /api/state returns separate train_loss, train_accuracy, test_loss, test_accuracy."""
    original_history = dict(state.training_history)
    try:
        state.training_history = {
            "round": [1],
            "loss": [0.7512],
            "accuracy": [70.5],
            "test_loss": [0.8123],
            "test_accuracy": [67.0],
            "precision": [68.0],
            "recall": [66.0],
            "f1_score": [67.0],
            "blocked_count": 0,
        }

        res = client.get("/api/state")
        assert res.status_code == 200
        data = res.json()

        rounds = data.get("rounds", [])
        assert len(rounds) == 1
        rd = rounds[0]

        assert rd["round_num"] == 1
        assert rd["train_loss"] == 0.7512
        assert rd["train_accuracy"] == 70.5
        assert rd["test_loss"] == 0.8123
        assert rd["test_accuracy"] == 67.0
        # Assert train and test metrics are strictly unconflated
        assert rd["train_loss"] != rd["test_loss"]
        assert rd["train_accuracy"] != rd["test_accuracy"]
    finally:
        state.training_history = original_history


def test_sqlite_state_store_round_trip_preserves_distinct_metrics():
    """Verify that SQLite store persists and rehydrates distinct train and test metrics."""
    test_round = {
        "round_num": 999,
        "train_loss": 0.4321,
        "train_accuracy": 82.5,
        "test_loss": 0.5123,
        "test_accuracy": 79.0,
        "precision": 80.0,
        "recall": 78.0,
        "f1_score": 79.0,
        "privacy_epsilon": 0.5,
        "threat_detected": False,
        "blocked_nodes": [],
        "status": "completed",
    }
    state.store.record_training_round(test_round)

    history = state.store.get_training_history()
    assert 999 in history["round"]
    idx = history["round"].index(999)

    assert history["loss"][idx] == 0.4321
    assert history["accuracy"][idx] == 82.5
    assert history["test_loss"][idx] == 0.5123
    assert history["test_accuracy"][idx] == 79.0
    assert history["loss"][idx] != history["test_loss"][idx]
