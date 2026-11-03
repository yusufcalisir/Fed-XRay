"""React Client Subtree Resilience and Error Boundaries Verification Suite.

Verifies Next.js app/error.tsx and app/global-error.tsx fault containment,
granular section error boundaries, and defensive array access across nullable payloads.
"""

from __future__ import annotations
from pathlib import Path


def test_nextjs_error_boundary_files_presence_and_contracts():
    """Verify Next.js App Router error.tsx and global-error.tsx exist with proper contracts."""
    error_path = Path("frontend/src/app/error.tsx")
    assert error_path.exists(), "frontend/src/app/error.tsx must exist"
    error_code = error_path.read_text(encoding="utf-8")

    assert '"use client"' in error_code
    assert "RootRouteError" in error_code or "export default function" in error_code
    assert "reset" in error_code
    assert "Clinical Interface Protection Active" in error_code
    assert "Recover View" in error_code

    global_error_path = Path("frontend/src/app/global-error.tsx")
    assert global_error_path.exists(), "frontend/src/app/global-error.tsx must exist"
    global_code = global_error_path.read_text(encoding="utf-8")

    assert '"use client"' in global_code
    assert "GlobalError" in global_code
    assert "<html" in global_code
    assert "<body" in global_code
    assert "Fed-XRay Root Runtime Protection Active" in global_code
    assert "Recover Application" in global_code


def test_clinical_error_boundary_bilingual_and_recovery_contracts():
    """Verify ClinicalErrorBoundary handles bilingual props and reset lifecycle safely."""
    boundary_path = Path("frontend/src/components/ClinicalErrorBoundary.tsx")
    assert boundary_path.exists(), "ClinicalErrorBoundary.tsx must exist"
    code = boundary_path.read_text(encoding="utf-8")

    assert "language" in code
    assert "toUpperCase()" in code
    assert "handleReset" in code
    assert "handleReload" in code
    assert "Klinik Arayüz Koruma Kalkanı Devrede" in code
    assert "Clinical Interface Protection Active" in code


def test_dashboard_and_landing_page_containment_architecture():
    """Verify dashboard and landing pages encapsulate subtrees with ClinicalErrorBoundary."""
    dashboard_path = Path("frontend/src/app/dashboard/page.tsx")
    assert dashboard_path.exists(), "dashboard/page.tsx must exist"
    dashboard_code = dashboard_path.read_text(encoding="utf-8")

    assert "ClinicalErrorBoundary" in dashboard_code
    assert "<ClinicalErrorBoundary" in dashboard_code
    # Verify defensive array totalSamples reduction
    assert "reduce((acc, h) => acc + (h?.num_samples || 0), 0)" in dashboard_code

    landing_path = Path("frontend/src/app/page.tsx")
    assert landing_path.exists(), "page.tsx must exist"
    landing_code = landing_path.read_text(encoding="utf-8")

    assert "ClinicalErrorBoundary" in landing_code
    assert "<ClinicalErrorBoundary" in landing_code


def test_defensive_telemetry_and_rag_array_handling():
    """Verify TelemetryCockpit and RagDigitalTwins handle null or undefined collections safely."""
    cockpit_path = Path("frontend/src/components/TelemetryCockpit.tsx")
    assert cockpit_path.exists(), "TelemetryCockpit.tsx must exist"
    cockpit_code = cockpit_path.read_text(encoding="utf-8")

    assert "safeHistory" in cockpit_code
    assert "Array.isArray(history)" in cockpit_code

    rag_path = Path("frontend/src/components/RagDigitalTwins.tsx")
    assert rag_path.exists(), "RagDigitalTwins.tsx must exist"
    rag_code = rag_path.read_text(encoding="utf-8")

    assert "!Array.isArray(twins)" in rag_code

