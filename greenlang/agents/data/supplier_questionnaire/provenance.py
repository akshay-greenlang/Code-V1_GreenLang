# -*- coding: utf-8 -*-
"""
Provenance Tracking for Supplier Questionnaire Processor - AGENT-DATA-008

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"supplier-questionnaire"``.

Example:
    >>> from greenlang.agents.data.supplier_questionnaire.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("questionnaire", "q_001", "process", "abc123")
    >>> valid, chain = tracker.verify_chain("q_001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for Supplier Questionnaire Processor operations."""

    def __init__(self) -> None:
        """Initialize with supplier-questionnaire genesis hash."""
        super().__init__(agent_name="supplier-questionnaire")


__all__ = [
    "ProvenanceTracker",
]
