# -*- coding: utf-8 -*-
"""
Provenance Tracking for Spend Data Categorizer - AGENT-DATA-009

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"spend-categorizer"``.

Example:
    >>> from greenlang.agents.data.spend_categorizer.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("spend_record", "sp_001", "categorize", "abc123")
    >>> valid, chain = tracker.verify_chain("sp_001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for Spend Data Categorizer operations."""

    def __init__(self) -> None:
        """Initialize with spend-categorizer genesis hash."""
        super().__init__(agent_name="spend-categorizer")


__all__ = [
    "ProvenanceTracker",
]
