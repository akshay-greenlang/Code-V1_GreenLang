# -*- coding: utf-8 -*-
"""
Provenance Tracking for PDF & Invoice Extractor Operations - AGENT-DATA-001

Thin shim that delegates to the shared ``greenlang.data_commons.provenance``
base class with a hardcoded agent name of ``"pdf-extractor"``.

Example:
    >>> from greenlang.agents.data.pdf_extractor.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> chain_hash = tracker.record("document", "doc_001", "ingest", "abc123")
    >>> valid, chain = tracker.verify_chain("doc_001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for PDF & Invoice Extractor operations."""

    def __init__(self) -> None:
        """Initialize with pdf-extractor genesis hash."""
        super().__init__(agent_name="pdf-extractor")


__all__ = [
    "ProvenanceTracker",
]
