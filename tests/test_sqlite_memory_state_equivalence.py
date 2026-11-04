"""Verification suite for strict equivalence between in-memory telemetry and SQLite WAL storage.

Validates Finding DEFECT-09 remediation:
- In-memory training history and SQLite training_rounds table remain 100% consistent.
- No metric conflation across process restart boundaries.
- Precision and recall are not overwritten with accuracy during disk synchronization.
- Threat detection flags, blocked nodes, and privacy epsilons survive reboots faithfully.
- State reset clears both relational database tables and in-memory caches.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app, state, AppState
from src.fed_xray.storage.state_store import PersistentStateStore


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)


def test_in_memory_and_sqlite_history_strict_equivalence(tmp_path: Path):
    """Verify that records saved to SQLite reconstruct an identical schema and values to memory."""
    db_file = tmp_path / "test_equiv.db"
    store = PersistentStateStore(db_path=db_file)

    rounds_data = [
        {
            "round_num": 1,
            "train_loss": 0.9412,
            "train_accuracy": 68.50,
            "test_loss": 0.8123,
            "test_accuracy": 74.20,
            "precision": 72.10,
            "recall": 69.80,
            "f1_score": 70.93,
            "privacy_epsilon": 0.4215,
            "threat_detected": False,
            "blocked_nodes": [],
            "status": "completed",
        },
        {
            "round_num": 2,
            "train_loss": 0.5218,
            "train_accuracy": 82.40,
            "test_loss": 0.4891,
            "test_accuracy": 85.10,
            "precision": 83.50,
            "recall": 84.00,
            "f1_score": 83.75,
            "privacy_epsilon": 0.8430,
            "threat_detected": True,
            "blocked_nodes": [2],
            "status": "completed",
        },
    ]

    for r in rounds_data:
        store.record_training_round(r)

    history = store.get_training_history()

    # Exact key presence check
    expected_keys = {
        "round",
        "loss",
        "accuracy",
        "test_loss",
        "test_accuracy",
        "precision",
        "recall",
        "f1_score",
        "privacy_epsilon",
        "uplink_bytes",
        "threat_detected",
        "blocked_nodes",
        "blocked_count",
        "privacy_spent",
    }
    assert set(history.keys()) == expected_keys

    # Exact value equivalence
    assert history["round"] == [1, 2]
    assert history["loss"] == [0.9412, 0.5218]
    assert history["accuracy"] == [68.50, 82.40]
    assert history["test_loss"] == [0.8123, 0.4891]
    assert history["test_accuracy"] == [74.20, 85.10]
    assert history["precision"] == [72.10, 83.50]
    assert history["recall"] == [69.80, 84.00]
    assert history["f1_score"] == [70.93, 83.75]
    assert history["privacy_epsilon"] == [0.4215, 0.8430]
    assert history["uplink_bytes"] == [0, 0]
    assert history["threat_detected"] == [False, True]
    assert history["blocked_nodes"] == [[], [2]]
    assert history["blocked_count"] == 1
    assert history["privacy_spent"] == pytest.approx(0.8430, rel=1e-3)


def test_save_to_disk_preserves_unconflated_precision_and_recall(tmp_path: Path):
    """Verify that save_to_disk writes authentic precision and recall to SQLite, not accuracy."""
    db_file = tmp_path / "test_save_preservation.db"
    store = PersistentStateStore(db_path=db_file)

    s = AppState()
    s.store = store
    s.model_trained = True
    s.training_history = {
        "round": [1],
        "loss": [0.6543],
        "accuracy": [88.50],
        "test_loss": [0.6210],
        "test_accuracy": [89.10],
        "precision": [91.20],
        "recall": [86.40],
        "f1_score": [88.73],
        "privacy_epsilon": [0.35],
        "threat_detected": [False],
        "blocked_nodes": [[]],
        "blocked_count": 0,
        "privacy_spent": 0.35,
    }

    # Save to SQLite via save_to_disk
    s.save_to_disk(checkpoint_dir=str(tmp_path / "ckpts"))

    # Reconstruct from SQLite
    restored_hist = store.get_training_history()
    assert restored_hist["accuracy"] == [88.50]
    # Invariant: precision and recall MUST NOT have been overwritten with accuracy (88.50)
    assert restored_hist["precision"] == [91.20]
    assert restored_hist["recall"] == [86.40]
    assert restored_hist["loss"] == [0.6543]
    assert restored_hist["test_loss"] == [0.6210]


def test_api_state_reports_identical_rounds_before_and_after_restart(monkeypatch: pytest.MonkeyPatch, test_client: TestClient, tmp_path: Path):
    """Verify /api/state reports the exact same rounds whether reading memory or hydrating from SQLite."""
    db_file = tmp_path / "test_lifecycle.db"
    store = PersistentStateStore(db_path=db_file)

    monkeypatch.setattr(state, "store", store)
    monkeypatch.setattr(state, "hospital_cohort_info", [])
    monkeypatch.setattr(state, "server", None)

    # 1. Populate training history in memory and save
    state.training_history = {
        "round": [1, 2],
        "loss": [1.1250, 0.6540],
        "accuracy": [65.0, 84.5],
        "test_loss": [1.0500, 0.5890],
        "test_accuracy": [67.5, 86.0],
        "precision": [66.0, 85.0],
        "recall": [64.0, 84.0],
        "f1_score": [65.0, 84.5],
        "privacy_epsilon": [0.45, 0.90],
        "threat_detected": [False, True],
        "blocked_nodes": [[], [3]],
        "blocked_count": 1,
        "privacy_spent": 0.90,
    }
    state.model_trained = True
    state.save_to_disk(checkpoint_dir=str(tmp_path / "ckpts"))

    # 2. Query /api/state before restart (reading in-memory cache)
    res_before = test_client.get("/api/state")
    assert res_before.status_code == 200
    data_before = res_before.json()
    rounds_before = data_before["rounds"]
    assert len(rounds_before) == 2
    assert rounds_before[0]["train_loss"] == 1.1250
    assert rounds_before[0]["test_loss"] == 1.0500
    assert rounds_before[0]["threat_detected"] is False
    assert rounds_before[1]["train_loss"] == 0.6540
    assert rounds_before[1]["test_loss"] == 0.5890
    assert rounds_before[1]["threat_detected"] is True

    # 3. Simulate process restart by resetting in-memory history cache to empty
    monkeypatch.setattr(state, "training_history", {
        "loss": [],
        "accuracy": [],
        "round": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "test_accuracy": [],
        "test_loss": [],
        "privacy_epsilon": [],
        "threat_detected": [],
        "blocked_nodes": [],
        "blocked_count": 0,
        "privacy_spent": 0.0,
    })

    # 4. Query /api/state after restart (hydrates from SQLite store)
    res_after = test_client.get("/api/state")
    assert res_after.status_code == 200
    data_after = res_after.json()
    rounds_after = data_after["rounds"]

    # 5. Assert strict telemetry equality across server reboot
    assert rounds_before == rounds_after
    assert data_after["model_trained"] is True


def test_api_state_reset_clears_both_memory_and_sqlite(test_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that POST /api/state/reset cleanly purges both SQLite tables and memory."""
    db_file = tmp_path / "test_reset.db"
    store = PersistentStateStore(db_path=db_file)

    monkeypatch.setattr(state, "store", store)
    state.training_history = {
        "round": [1],
        "loss": [0.5],
        "accuracy": [80.0],
        "test_loss": [0.4],
        "test_accuracy": [85.0],
        "precision": [82.0],
        "recall": [78.0],
        "f1_score": [80.0],
        "privacy_epsilon": [0.5],
        "threat_detected": [False],
        "blocked_nodes": [[]],
        "blocked_count": 0,
        "privacy_spent": 0.5,
    }
    state.model_trained = True
    state.save_to_disk(checkpoint_dir=str(tmp_path / "ckpts"))

    # Reset
    res = test_client.post("/api/state/reset")
    assert res.status_code == 200

    # In-memory must be empty
    assert len(state.training_history.get("round", [])) == 0
    assert state.model_trained is False

    # SQLite must be empty
    sql_hist = store.get_training_history()
    assert len(sql_hist.get("round", [])) == 0
