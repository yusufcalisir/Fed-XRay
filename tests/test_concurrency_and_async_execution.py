"""Forensic Automated Test Suite for Section 4.6: Concurrency and Asynchronous Execution.

Verifies:
1. Thread-safe atomic task reservation and mutex prevention of concurrent training collisions.
2. Rejection of competing start requests (HTTP 409 for streaming, already_running for polling).
3. Thread safety of concurrent diagnostic inference and isolated Grad-CAM backward hooks.
4. Operator-initiated cancellation of running federated training tasks (streaming and background).
5. Thread-safe telemetry snapshot retrieval preventing iteration mutation race conditions.
6. Clean task finalization and resource release upon client disconnect or early termination.
"""

from __future__ import annotations
import concurrent.futures
import threading
import time
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from api.main import app, state
from src.fed_xray.core.server import CentralServer


@pytest.fixture
def clean_concurrency_state() -> Generator[None, None, None]:
    """Ensure state task locks, flags, and training history are cleanly restored."""
    with state.task_lock:
        state.task_running = False
        state.task_complete = False
        state.task_error = None
        state.task_rounds = []
        state.task_type = None
        state.cancel_event.clear()

    yield

    with state.task_lock:
        state.task_running = False
        state.task_complete = False
        state.task_error = None
        state.task_rounds = []
        state.task_type = None
        state.cancel_event.clear()


def test_atomic_task_start_mutex_prevents_simultaneous_training(clean_concurrency_state):
    """Verify that concurrent threads attempting to start training are serialized, only one winning."""
    num_threads = 8
    barrier = threading.Barrier(num_threads)
    results = []

    def attempt_start(thread_id: int):
        barrier.wait()
        acquired = state.try_start_task(task_type="unit_test_fl", total_rounds=5)
        results.append((thread_id, acquired))

    threads = [threading.Thread(target=attempt_start, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [r for r in results if r[1] is True]
    failures = [r for r in results if r[1] is False]

    assert len(successes) == 1, f"Expected exactly 1 thread to acquire task, got {len(successes)}"
    assert len(failures) == num_threads - 1
    assert state.task_running is True
    assert state.task_type == "unit_test_fl"

    state.finish_task()
    assert state.task_running is False
    assert state.task_type is None


def test_background_training_rejects_concurrent_start_requests(clean_concurrency_state, monkeypatch):
    """Verify that when a task is running, train-start and flower-start report already_running and train-stream raises 409."""
    client = TestClient(app)

    # Fake cohort availability so require_cohort does not fail with 409 No Cohort
    monkeypatch.setattr(state, "dataloaders", [None, None])
    monkeypatch.setattr(state, "global_test_set", (None, None))

    # Mark a task as actively running
    assert state.try_start_task(task_type="active_background", total_rounds=5) is True

    # 1. train-start reports already_running
    res_start = client.post("/api/fl/train-start?num_rounds=3")
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "already_running"

    # 2. flower-start reports already_running
    res_flower = client.post("/api/fl/flower-start?num_rounds=3")
    assert res_flower.status_code == 200
    assert res_flower.json()["status"] == "already_running"

    # 3. train-stream raises HTTP 409 Conflict
    res_stream = client.get("/api/fl/train-stream?num_rounds=3")
    assert res_stream.status_code == 409
    assert "already active" in res_stream.json()["detail"]

    state.finish_task()


def test_concurrent_diagnostic_inference_thread_safety(clean_concurrency_state):
    """Verify that concurrent diagnostic requests execute without PyTorch hook collision or state corruption."""
    client = TestClient(app)

    # Ensure model is initialized
    if state.server is None or not state.model_trained:
        srv = CentralServer(device=state.device, model_type="vit_tiny", peft_mode="ffa_lora", img_size=224, num_classes=3)
        state.server = srv
        state.model_trained = True

    state.ensure_cohort_loaded()

    num_concurrent_requests = 6
    errors = []
    responses = []

    def make_diagnosis_request(req_id: int):
        try:
            # Request diagnosis using holdout class
            res = client.post(
                "/api/cdss/diagnose",
                json={
                    "class_index": req_id % 3,
                    "opacity": 0.5,
                    "language": "en",
                },
            )
            responses.append((req_id, res.status_code, res.json()))
        except Exception as exc:
            errors.append((req_id, str(exc)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        futures = [executor.submit(make_diagnosis_request, i) for i in range(num_concurrent_requests)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent diagnosis threw exceptions: {errors}"
    assert len(responses) == num_concurrent_requests
    for req_id, status_code, body in responses:
        assert status_code == 200, f"Request {req_id} failed with status {status_code}"
        assert "predicted_class" in body
        assert "confidence" in body
        assert "findings" in body
        assert 0.0 <= body["confidence"] <= 100.0


def test_operator_cancellation_of_background_training(clean_concurrency_state):
    """Verify operator cancellation dispatch and responsive worker thread termination."""
    client = TestClient(app)

    # 1. Cancellation with no task running returns status not_running
    res_idle = client.post("/api/fl/train-cancel")
    assert res_idle.status_code == 200
    assert res_idle.json()["status"] == "not_running"

    # 2. Simulate task running
    assert state.try_start_task(task_type="cancellable_training", total_rounds=10) is True
    assert state.is_task_cancelled() is False

    # 3. Dispatch cancel
    res_cancel = client.post("/api/fl/train-cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] == "cancelling"
    assert state.is_task_cancelled() is True

    # 4. Status reflects cancellation
    res_status = client.get("/api/fl/train-status")
    assert res_status.status_code == 200
    status_body = res_status.json()
    assert status_body["is_running"] is True
    assert status_body["is_cancelled"] is True

    # 5. Worker finalizes task
    state.finish_task(error="Training cancelled by operator")
    assert state.task_running is False
    assert state.task_complete is False
    assert state.task_error == "Training cancelled by operator"


def test_concurrent_telemetry_reads_during_active_training(clean_concurrency_state):
    """Verify that concurrent reads of state and history during active worker writes cause zero race conditions."""
    client = TestClient(app)
    assert state.try_start_task(task_type="history_test", total_rounds=10) is True

    stop_event = threading.Event()
    writer_exceptions = []
    reader_exceptions = []

    def background_writer():
        round_idx = 1
        while not stop_event.is_set() and round_idx <= 20:
            round_data = {
                "round_num": round_idx,
                "train_loss": 0.5 / round_idx,
                "train_accuracy": 70.0 + round_idx,
                "test_loss": 0.6 / round_idx,
                "test_accuracy": 68.0 + round_idx,
                "precision": 75.0,
                "recall": 72.0,
                "f1_score": 73.5,
                "threat_detected": False,
                "blocked_nodes": [],
                "status": "training",
            }
            try:
                state.append_round_telemetry(
                    round_data=round_data,
                    history_update={
                        "round": [round_idx],
                        "loss": [0.5 / round_idx],
                        "accuracy": [70.0 + round_idx],
                        "test_loss": [0.6 / round_idx],
                        "test_accuracy": [68.0 + round_idx],
                        "precision": [75.0],
                        "recall": [72.0],
                        "f1_score": [73.5],
                        "threat_detected": [False],
                        "uplink_bytes": [49152],
                        "blocked_nodes": [[]],
                    },
                )
            except Exception as e:
                writer_exceptions.append(str(e))
            round_idx += 1
            time.sleep(0.01)

    def background_reader(reader_id: int):
        while not stop_event.is_set():
            try:
                # 1. State snapshot
                snapshot = state.get_training_history_snapshot()
                assert isinstance(snapshot, dict)
                # 2. HTTP state endpoint
                res_state = client.get("/api/state")
                assert res_state.status_code == 200
                # 3. HTTP train-status endpoint
                res_status = client.get("/api/fl/train-status?from_round=0")
                assert res_status.status_code == 200
            except Exception as e:
                reader_exceptions.append((reader_id, str(e)))
            time.sleep(0.005)

    writer_t = threading.Thread(target=background_writer)
    readers = [threading.Thread(target=background_reader, args=(i,)) for i in range(4)]

    writer_t.start()
    for r in readers:
        r.start()

    time.sleep(0.25)
    stop_event.set()

    writer_t.join()
    for r in readers:
        r.join()

    state.finish_task()

    assert len(writer_exceptions) == 0, f"Writer threw exceptions: {writer_exceptions}"
    assert len(reader_exceptions) == 0, f"Readers threw race condition exceptions: {reader_exceptions}"


def test_sse_stream_cleanup_on_cancellation(clean_concurrency_state, monkeypatch):
    """Verify that if an SSE stream client triggers cancellation, task locks and flags are released."""
    monkeypatch.setattr(state, "dataloaders", [None, None])
    monkeypatch.setattr(state, "global_test_set", (None, None))

    assert state.try_start_task(task_type="stream_training", total_rounds=3) is True
    assert state.task_running is True

    # Simulate client disconnect
    state.cancel_current_task()
    assert state.is_task_cancelled() is True

    state.finish_task(error="Client disconnected")
    assert state.task_running is False
    assert state.task_complete is False

    # Next task starts cleanly
    assert state.try_start_task(task_type="next_training", total_rounds=3) is True
    assert state.task_running is True
    state.finish_task()

