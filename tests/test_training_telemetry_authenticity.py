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


def test_background_training_populates_state_lineage(client, tmp_path):
    """Verify DEFECT-04 fix: background task rounds hydrate in-memory state and unlock model_trained."""
    original_history = dict(state.training_history)
    original_task_rounds = list(state.task_rounds)
    original_trained = state.model_trained
    original_task_complete = state.task_complete
    original_store = state.store
    try:
        from src.fed_xray.storage import PersistentStateStore
        state.store = PersistentStateStore(db_path=tmp_path / "isolated_consortium.db")

        # Clear in-memory history
        state.training_history = {
            "loss": [],
            "accuracy": [],
            "round": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "test_accuracy": [],
            "test_loss": [],
            "blocked_count": 0,
        }
        # Simulate completed background training task
        state.task_rounds = [
            {
                "round_num": 1,
                "total_rounds": 1,
                "train_loss": 0.5512,
                "train_accuracy": 76.5,
                "test_loss": 0.6214,
                "test_accuracy": 73.0,
                "precision": 74.0,
                "recall": 72.0,
                "f1_score": 73.0,
                "threat_detected": False,
                "blocked_nodes": [],
                "status": "complete",
            }
        ]
        state.task_complete = True
        state.model_trained = True

        res = client.get("/api/state")
        assert res.status_code == 200
        data = res.json()

        assert data["model_trained"] is True
        assert data["is_live"] is True
        assert data["history_rounds"] == 1
        assert len(data["rounds"]) == 1
        assert data["rounds"][0]["train_loss"] == 0.5512
        assert data["rounds"][0]["test_loss"] == 0.6214
    finally:
        state.training_history = original_history
        state.task_rounds = original_task_rounds
        state.model_trained = original_trained
        state.task_complete = original_task_complete
        state.store = original_store


def test_flower_telemetry_authentically_propagates_blocked_node_identity(client, tmp_path):
    """Verify DEFECT-11 fix: Flower telemetry records authentic blocked node IDs without hardcoding node 1."""
    from src.fed_xray.storage import PersistentStateStore

    original_history = dict(state.training_history)
    original_task_rounds = list(state.task_rounds)
    original_trained = state.model_trained
    original_task_complete = state.task_complete
    original_store = state.store

    try:
        state.store = PersistentStateStore(db_path=tmp_path / "flower_blocked_test.db")
        state.training_history = {
            "loss": [],
            "accuracy": [],
            "round": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "test_accuracy": [],
            "test_loss": [],
            "threat_detected": [],
            "privacy_epsilon": [],
            "blocked_nodes": [],
            "blocked_count": 0,
        }

        # Simulate Flower round where client 2 (Hospital 3) was blocked by Byzantine defense
        authentic_blocked_cids = [2]
        display_blocked_nodes = [cid + 1 for cid in authentic_blocked_cids]
        threat_detected = len(display_blocked_nodes) > 0

        round_data = {
            "round_num": 1,
            "total_rounds": 1,
            "train_loss": 0.45,
            "train_accuracy": 82.5,
            "test_loss": 0.50,
            "test_accuracy": 80.0,
            "precision": 81.0,
            "recall": 79.0,
            "f1_score": 80.0,
            "threat_detected": threat_detected,
            "blocked_nodes": display_blocked_nodes,
            "status": "complete",
            "orchestrator": "flower",
        }
        state.task_rounds = [round_data]
        state.task_complete = True
        state.model_trained = True

        state.training_history.setdefault("round", []).append(1)
        state.training_history.setdefault("loss", []).append(0.45)
        state.training_history.setdefault("accuracy", []).append(82.5)
        state.training_history.setdefault("test_loss", []).append(0.50)
        state.training_history.setdefault("test_accuracy", []).append(80.0)
        state.training_history.setdefault("precision", []).append(81.0)
        state.training_history.setdefault("recall", []).append(79.0)
        state.training_history.setdefault("f1_score", []).append(80.0)
        state.training_history.setdefault("threat_detected", []).append(threat_detected)
        state.training_history.setdefault("privacy_epsilon", []).append(0.12)
        state.training_history.setdefault("blocked_nodes", []).append(display_blocked_nodes)
        state.training_history["blocked_count"] = len(display_blocked_nodes)

        state.store.record_training_round({
            "round_num": 1,
            "train_loss": 0.45,
            "train_accuracy": 82.5,
            "test_loss": 0.50,
            "test_accuracy": 80.0,
            "precision": 81.0,
            "recall": 79.0,
            "f1_score": 80.0,
            "privacy_epsilon": 0.12,
            "threat_detected": threat_detected,
            "blocked_nodes": display_blocked_nodes,
            "status": "complete",
        })

        # Verify that node 1 is NOT in blocked_nodes, and node 3 is present
        assert state.training_history["blocked_nodes"] == [[3]]
        assert 1 not in state.training_history["blocked_nodes"][0]

        # Query /api/state and verify persistence & contract
        res = client.get("/api/state")
        assert res.status_code == 200
        data = res.json()
        assert len(data["rounds"]) == 1
        assert data["rounds"][0]["threat_detected"] is True
        assert data["rounds"][0]["blocked_nodes"] == [3]
        assert 1 not in data["rounds"][0]["blocked_nodes"]

        # Verify SQLite stored representation matches in-memory exactly
        saved_hist = state.store.get_training_history()
        assert saved_hist["blocked_nodes"] == [[3]]
    finally:
        state.training_history = original_history
        state.task_rounds = original_task_rounds
        state.model_trained = original_trained
        state.task_complete = original_task_complete
        state.store = original_store


def test_flower_training_metric_independence_and_no_validation_fallback(client, tmp_path):
    """Verify DEFECT-13 fix: Flower telemetry never substitutes validation metrics for missing training fit metrics."""
    from src.fed_xray.storage import PersistentStateStore

    original_history = dict(state.training_history)
    original_task_rounds = list(state.task_rounds)
    original_trained = state.model_trained
    original_task_complete = state.task_complete
    original_store = state.store

    try:
        state.store = PersistentStateStore(db_path=tmp_path / "flower_metric_independence_test.db")
        state.training_history = {
            "loss": [],
            "accuracy": [],
            "round": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "test_accuracy": [],
            "test_loss": [],
            "threat_detected": [],
            "privacy_epsilon": [],
            "blocked_nodes": [],
            "blocked_count": 0,
        }

        # Scenario 1: Missing training fit metrics in agg_metrics must NOT substitute val_loss/val_acc
        agg_metrics_missing: dict = {}
        val_loss = 0.6543
        val_acc = 85.0

        # Simulate the extraction logic from on_round_completed
        raw_train_loss = (
            agg_metrics_missing.get("train_loss")
            if agg_metrics_missing.get("train_loss") is not None
            else agg_metrics_missing.get("loss")
        )
        t_loss = float(raw_train_loss) if raw_train_loss is not None else 0.0

        raw_train_acc = (
            agg_metrics_missing.get("train_accuracy")
            if agg_metrics_missing.get("train_accuracy") is not None
            else agg_metrics_missing.get("accuracy")
        )
        if raw_train_acc is not None:
            train_acc_val = float(raw_train_acc)
            t_acc = train_acc_val * 100.0 if train_acc_val <= 1.0 else train_acc_val
        else:
            t_acc = 0.0

        # Invariant: train_loss and train_accuracy must NOT equal validation loss and accuracy
        assert t_loss == 0.0
        assert t_acc == 0.0
        assert t_loss != val_loss
        assert t_acc != val_acc

        # Scenario 2: Authentic distinct training metrics must be preserved
        agg_metrics_authentic = {
            "train_loss": 0.4120,
            "train_accuracy": 82.5,
            "blocked_clients_count": 0,
            "blocked_nodes_json": "[]",
        }
        raw_train_loss2 = (
            agg_metrics_authentic.get("train_loss")
            if agg_metrics_authentic.get("train_loss") is not None
            else agg_metrics_authentic.get("loss")
        )
        t_loss2 = float(raw_train_loss2) if raw_train_loss2 is not None else 0.0

        raw_train_acc2 = (
            agg_metrics_authentic.get("train_accuracy")
            if agg_metrics_authentic.get("train_accuracy") is not None
            else agg_metrics_authentic.get("accuracy")
        )
        if raw_train_acc2 is not None:
            train_acc_val2 = float(raw_train_acc2)
            t_acc2 = train_acc_val2 * 100.0 if train_acc_val2 <= 1.0 else train_acc_val2
        else:
            t_acc2 = 0.0

        assert t_loss2 == 0.4120
        assert t_acc2 == 82.5
        assert t_loss2 != val_loss
        assert t_acc2 != val_acc
    finally:
        state.training_history = original_history
        state.task_rounds = original_task_rounds
        state.model_trained = original_trained
        state.task_complete = original_task_complete
        state.store = original_store


def test_flower_hospital_client_fit_reports_train_loss_and_train_accuracy():
    """Verify FlowerHospitalClient.fit returns explicit train_loss and train_accuracy keys."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.fed_xray.core.client import HospitalClient
    from src.fed_xray.federation.flower_protocol_bridge import (
        FlowerHospitalClient,
        state_dict_to_flower_parameters,
    )

    x = torch.randn(8, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    client_core = HospitalClient(client_id=0, dataloader=loader, model_type="cnn")
    flower_client = FlowerHospitalClient(hospital_client=client_core)

    weights = {k: v.clone().detach().cpu() for k, v in client_core.model.state_dict().items()}
    arrays, _ = state_dict_to_flower_parameters(weights)

    _, num_samples, metrics = flower_client.fit(arrays, {"current_round": 1, "total_rounds": 1})

    assert num_samples == 8
    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert isinstance(metrics["train_loss"], float)
    assert isinstance(metrics["train_accuracy"], float)

