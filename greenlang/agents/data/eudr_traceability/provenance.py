# -*- coding: utf-8 -*-
"""
Provenance Tracking for EUDR Traceability Connector - AGENT-DATA-004

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"eudr-traceability"``.

Example:
    >>> from greenlang.agents.data.eudr_traceability.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("plot", "PLOT-abc123", "register", "abc123")
    >>> valid, chain = tracker.verify_chain("PLOT-abc123")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for EUDR Traceability Connector operations."""

    AGENT_ID = "GL-DATA-EUDR-001"

    VALID_OPERATIONS = frozenset({
        "plot_registration",
        "custody_transfer",
        "dds_generation",
        "risk_assessment",
        "commodity_classification",
        "compliance_check",
        "eu_submission",
        "batch_split",
        "batch_merge",
        "transfer_verification",
        "compliance_update",
        "declaration_registration",
    })

    def __init__(self) -> None:
        """Initialize with eudr-traceability genesis hash."""
        super().__init__(agent_name="eudr-traceability")


__all__ = [
    "ProvenanceTracker",
]
