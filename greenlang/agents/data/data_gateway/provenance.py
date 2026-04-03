# -*- coding: utf-8 -*-
"""
Provenance Tracking for API Gateway Agent - AGENT-DATA-004 (GL-DATA-GW-001)

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"data-gateway"``.

Example:
    >>> from greenlang.agents.data.data_gateway.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("query", "QRY-abc123", "execute", "abc123")
    >>> valid, chain = tracker.verify_chain("QRY-abc123")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for API Gateway Agent operations."""

    AGENT_ID = "GL-DATA-GW-001"

    VALID_OPERATIONS = frozenset({
        "query_execution",
        "source_registration",
        "schema_translation",
        "cache_operation",
        "aggregation",
        "health_check",
        "template_creation",
    })

    def __init__(self) -> None:
        """Initialize with data-gateway genesis hash."""
        super().__init__(agent_name="data-gateway")


__all__ = [
    "ProvenanceTracker",
]
