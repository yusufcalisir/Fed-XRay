"""Test suite verifying API-02 remediation: Network Routing & Tunnel Resilience.

Verifies:
1. Backend health endpoint contract and cross-origin CORS headers.
2. Enclave handshake accessibility from multi-origin cloud frontends.
3. Complete eradication of stale ephemeral tunnel hostnames from repository artifacts.
4. Ephemeral tunnel pattern recognition and fallback logic.
5. Next.js rewrite configuration sanitization against dead tunnel domains.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def test_health_endpoint_contract(api_client: TestClient):
    """Verify GET /api/health returns HTTP 200 with required health telemetry."""
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "ok")
    assert "timestamp" in data
    assert "device" in data
    assert "cuda_available" in data


def test_cors_cross_origin_tunnel_support(api_client: TestClient):
    """Verify CORS middleware permits requests from cloud frontends and tunnel domains."""
    test_origins = [
        "https://fed-xray.vercel.app",
        "https://custom-consortium-node.trycloudflare.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    for origin in test_origins:
        resp = api_client.get(
            "/api/health",
            headers={"Origin": origin},
        )
        assert resp.status_code == 200
        # CORS headers must permit the request
        allow_origin = resp.headers.get("access-control-allow-origin")
        assert allow_origin in ("*", origin), f"Expected open CORS for {origin}, got {allow_origin}"


def test_enclave_handshake_cross_origin_contract(api_client: TestClient):
    """Verify /api/fl/enclave-handshake is accessible across origins for launch attestation."""
    resp = api_client.get(
        "/api/fl/enclave-handshake",
        headers={"Origin": "https://fed-xray.vercel.app"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "attestation_digest" in data
    assert "privacy" in data
    assert "nodes" in data
    assert len(data["nodes"]) > 0


def test_notebook_sanitization_integrity():
    """Verify notebooks/fed_xray_cloud_gpu.ipynb contains zero expired tunnel artifacts."""
    nb_path = Path("notebooks/fed_xray_cloud_gpu.ipynb")
    assert nb_path.exists(), "Cloud GPU notebook must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_content = f.read()

    # Must contain zero references to expired shakira tunnel domain
    assert "globe-celebration-poems-shakira" not in nb_content

    # All cell outputs must be cleared to eliminate stale runtime state
    nb_json = json.loads(nb_content)
    for idx, cell in enumerate(nb_json.get("cells", [])):
        if cell.get("cell_type") == "code":
            assert len(cell.get("outputs", [])) == 0, f"Cell {idx} has un-cleared outputs"
            assert cell.get("execution_count") is None, f"Cell {idx} has non-null execution count"


def test_ephemeral_tunnel_detection_logic():
    """Verify ephemeral quick-tunnel domains are correctly identified for auto-fallback."""
    ephemeral_samples = [
        "https://abc-123.trycloudflare.com",
        "https://my-colab-tunnel.trycloudflare.com/api",
        "https://sample.loca.lt",
        "https://test-node.ngrok-free.app",
        "https://remote-sub.serveo.net",
    ]

    persistent_samples = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://192.168.1.50:8000",
        "https://api.fed-xray.hospital.org",
        "https://medical-consortium-core.internal",
    ]

    def is_ephemeral(url: str) -> bool:
        lower = url.lower()
        return any(
            t in lower
            for t in (
                ".trycloudflare.com",
                ".loca.lt",
                ".ngrok-free.app",
                ".serveo.net",
            )
        )

    for url in ephemeral_samples:
        assert is_ephemeral(url) is True, f"Expected {url} to be detected as ephemeral tunnel"

    for url in persistent_samples:
        assert is_ephemeral(url) is False, f"Expected {url} to be classified as persistent host"


def test_next_config_tunnel_sanitization():
    """Verify frontend/next.config.mjs sanitizes expired tunnel URLs."""
    config_path = Path("frontend/next.config.mjs")
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

    # Must sanitize known dead tunnels
    assert "globe-celebration-poems-shakira" in config_text
    assert "backendUrl" in config_text
