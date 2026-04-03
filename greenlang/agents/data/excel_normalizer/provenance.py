# -*- coding: utf-8 -*-
"""
Provenance Tracking for Excel/CSV Normalizer Operations - AGENT-DATA-002

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"excel-normalizer"``.

Example:
    >>> from greenlang.agents.data.excel_normalizer.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("spreadsheet", "ss_001", "normalize", "abc123")
    >>> valid, chain = tracker.verify_chain("ss_001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel/CSV Normalizer
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for Excel/CSV Normalizer operations."""

    def __init__(self) -> None:
        """Initialize with excel-normalizer genesis hash."""
        super().__init__(agent_name="excel-normalizer")


__all__ = [
    "ProvenanceTracker",
]
