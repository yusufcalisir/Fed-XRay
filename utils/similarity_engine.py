"""
Fed-XRay Similarity Engine (Backward-Compatibility Shim)
=========================================================
Re-exports Case-Based Reasoning components from `src.fed_xray.cdss.similarity`.
"""

from src.fed_xray.cdss.similarity import (
    HistoricalCaseBank,
    extract_embedding,
    LABEL_NAMES,
    LABEL_COLORS
)

__all__ = ["HistoricalCaseBank", "extract_embedding", "LABEL_NAMES", "LABEL_COLORS"]
