# -*- coding: utf-8 -*-
"""
Provenance Tracking for ERP/Finance Connector Operations - AGENT-DATA-003

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"erp-connector"``.

Example:
    >>> from greenlang.agents.data.erp_connector.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("erp_record", "rec_001", "extract", "abc123")
    >>> valid, chain = tracker.verify_chain("rec_001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for ERP/Finance Connector operations."""

    def __init__(self) -> None:
        """Initialize with erp-connector genesis hash."""
        super().__init__(agent_name="erp-connector")


__all__ = [
    "ProvenanceTracker",
]
