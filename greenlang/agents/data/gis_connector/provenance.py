# -*- coding: utf-8 -*-
"""
Provenance Tracking for GIS/Mapping Connector - AGENT-DATA-006

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"gis-connector"``.

Example:
    >>> from greenlang.agents.data.gis_connector.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("layer", "LYR-abc123", "layer_operation", "abc123")
    >>> valid, chain = tracker.verify_chain("LYR-abc123")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for GIS/Mapping Connector operations."""

    AGENT_ID = "GL-DATA-GEO-001"

    VALID_OPERATIONS = frozenset({
        "format_parse",
        "crs_transform",
        "spatial_analysis",
        "land_cover_classify",
        "boundary_resolve",
        "geocoding",
        "layer_operation",
    })

    def __init__(self) -> None:
        """Initialize with gis-connector genesis hash."""
        super().__init__(agent_name="gis-connector")


__all__ = [
    "ProvenanceTracker",
]
