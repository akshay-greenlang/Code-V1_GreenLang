# -*- coding: utf-8 -*-
"""
Provenance Tracking for Deforestation Satellite Connector - AGENT-DATA-007

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"deforestation-satellite"``.

Example:
    >>> from greenlang.agents.data.deforestation_satellite.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record(
    ...     "satellite_acquisition", "SCN-abc123", "acquire", "abc123",
    ... )
    >>> valid, chain = tracker.verify_chain("SCN-abc123")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker

# Module-level constant preserved for backward compatibility
VALID_OPERATION_TYPES = frozenset({
    "satellite_acquisition",
    "change_detection",
    "alert_aggregation",
    "baseline_assessment",
    "classification",
    "compliance_report",
    "pipeline_execution",
})


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for Deforestation Satellite Connector operations."""

    def __init__(self) -> None:
        """Initialize with deforestation-satellite genesis hash."""
        super().__init__(agent_name="deforestation-satellite")

    def get_by_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all provenance entries for an entity (alias for get_chain).

        Args:
            entity_id: Entity ID to look up.

        Returns:
            List of provenance entries.
        """
        return self.get_chain(entity_id)


__all__ = [
    "ProvenanceTracker",
    "VALID_OPERATION_TYPES",
]
