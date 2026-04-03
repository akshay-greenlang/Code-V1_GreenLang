# -*- coding: utf-8 -*-
"""
Provenance Tracking for Data Quality Profiler - AGENT-DATA-010

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"data-quality-profiler"``.

Example:
    >>> from greenlang.agents.data.data_quality_profiler.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("profile", "prof_001", "profile", "abc123")
    >>> valid, chain = tracker.verify_chain("prof_001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for Data Quality Profiler operations."""

    def __init__(self) -> None:
        """Initialize with data-quality-profiler genesis hash."""
        super().__init__(agent_name="data-quality-profiler")


__all__ = [
    "ProvenanceTracker",
]
