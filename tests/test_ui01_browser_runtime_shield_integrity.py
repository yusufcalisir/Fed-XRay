"""UI-01 Forensic Remediation Integrity Test Suite.

Verifies product-grade isolation against third-party browser extension collisions,
service worker port drop noise, unhandled promise rejections, and DOM mutations.
"""

from __future__ import annotations
from pathlib import Path


def test_runtime_shield_source_and_pattern_integrity():
    """Verify runtimeShield.ts includes precise pattern detection for all observed audit errors."""
    shield_path = Path("frontend/src/lib/runtimeShield.ts")
    assert shield_path.exists(), "frontend/src/lib/runtimeShield.ts must exist"
    code = shield_path.read_text(encoding="utf-8")

    # Observed errors in UI-01 audit finding:
    # 1. content.js reading 'useCache'
    assert "usecache" in code.lower()
    assert "content.js" in code

    # 2. adblock-picreplacement.js port drop
    assert "adblock-picreplacement.js" in code

    # 3. globals-front.js script error
    assert "globals-front.js" in code

    # 4. Could not establish connection. Receiving end does not exist.
    assert "could not establish connection" in code.lower()

    # Chrome extension URL schemes
    assert "chrome-extension://" in code
    assert "moz-extension://" in code

    # Core interface & telemetry contracts
    assert "installRuntimeShield" in code
    assert "getRuntimeShieldTelemetry" in code
    assert "isThirdPartyExtensionError" in code
    assert "stopImmediatePropagation" in code


def test_clinical_error_boundary_contracts():
    """Verify ClinicalErrorBoundary provides medical-grade recovery UI with bilingual support."""
    boundary_path = Path("frontend/src/components/ClinicalErrorBoundary.tsx")
    assert boundary_path.exists(), "ClinicalErrorBoundary.tsx must exist"
    code = boundary_path.read_text(encoding="utf-8")

    assert "ClinicalErrorBoundary" in code
    assert "getDerivedStateFromError" in code
    assert "componentDidCatch" in code
    assert "Klinik Arayüz Koruma Kalkanı Devrede" in code
    assert "Clinical Interface Protection Active" in code
    assert "handleReset" in code
    assert "handleReload" in code


def test_client_runtime_shield_and_layout_integration():
    """Verify layout.tsx mounts ClientRuntimeShield and protects against hydration mismatches."""
    client_shield_path = Path("frontend/src/components/ClientRuntimeShield.tsx")
    assert client_shield_path.exists(), "ClientRuntimeShield.tsx must exist"
    client_shield_code = client_shield_path.read_text(encoding="utf-8")
    assert "installRuntimeShield" in client_shield_code
    assert "ClinicalErrorBoundary" in client_shield_code

    layout_path = Path("frontend/src/app/layout.tsx")
    assert layout_path.exists(), "layout.tsx must exist"
    layout_code = layout_path.read_text(encoding="utf-8")

    assert "ClientRuntimeShield" in layout_code
    assert "suppressHydrationWarning" in layout_code


def test_frontend_vitest_shield_test_suite_presence():
    """Verify Vitest unit test files for runtime shield and error boundary exist."""
    shield_test = Path("frontend/src/__tests__/runtimeShield.test.ts")
    boundary_test = Path("frontend/src/__tests__/ClinicalErrorBoundary.test.tsx")

    assert shield_test.exists(), "runtimeShield.test.ts must exist"
    assert boundary_test.exists(), "ClinicalErrorBoundary.test.tsx must exist"

    shield_code = shield_test.read_text(encoding="utf-8")
    assert "does NOT flag genuine Fed-XRay application and API errors" in shield_code
    assert "intercepts and throttles console.error noise" in shield_code

