# -*- coding: utf-8 -*-
"""
Citations Service Setup - AGENT-FOUND-005: Citations & Evidence

Provides ``configure_citations_service(app)`` which wires up the
Citations & Evidence SDK (registry, evidence, verification, provenance,
export/import) and mounts the REST API.

Also exposes ``get_citations_service(app)`` for programmatic access
and the ``CitationsService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.citations.setup import configure_citations_service
    >>> app = FastAPI()
    >>> configure_citations_service(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from greenlang.citations.config import CitationsConfig, get_config
from greenlang.citations.registry import CitationRegistry
from greenlang.citations.evidence import EvidenceManager
from greenlang.citations.verification import VerificationEngine
from greenlang.citations.provenance import ProvenanceTracker
from greenlang.citations.export_import import ExportImportManager
from greenlang.citations.metrics import PROMETHEUS_AVAILABLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# CitationsService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["CitationsService"] = None


class CitationsService:
    """Unified facade over the Citations & Evidence SDK.

    Manages citation registry, evidence packaging, verification,
    provenance tracking, and export/import through a single entry point.

    Attributes:
        registry: CitationRegistry instance.
        evidence_manager: EvidenceManager instance.
        verification_engine: VerificationEngine instance.
        provenance: ProvenanceTracker instance.
        export_import: ExportImportManager instance.
        config: CitationsConfig instance.

    Example:
        >>> service = CitationsService()
        >>> c = service.registry.create(
        ...     citation_type="emission_factor",
        ...     source_authority="defra",
        ...     metadata={"title": "DEFRA 2024"},
        ...     effective_date="2024-01-01",
        ... )
        >>> print(service.registry.count)
    """

    def __init__(
        self,
        config: Optional[CitationsConfig] = None,
    ) -> None:
        """Initialize the Citations Service facade.

        Args:
            config: Optional citations config. Uses global config if None.
        """
        self.config = config or get_config()
        self.provenance = ProvenanceTracker()
        self.registry = CitationRegistry(
            config=self.config,
            provenance=self.provenance,
        )
        self.evidence_manager = EvidenceManager(
            config=self.config,
            provenance=self.provenance,
        )
        self.verification_engine = VerificationEngine(
            registry=self.registry,
            config=self.config,
        )
        self.export_import = ExportImportManager(
            registry=self.registry,
            config=self.config,
        )
        self._started = False

        logger.info("CitationsService facade created")

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_registry(self) -> CitationRegistry:
        """Get the CitationRegistry instance.

        Returns:
            The CitationRegistry used by this service.
        """
        return self.registry

    def get_evidence_manager(self) -> EvidenceManager:
        """Get the EvidenceManager instance.

        Returns:
            The EvidenceManager used by this service.
        """
        return self.evidence_manager

    def get_verification_engine(self) -> VerificationEngine:
        """Get the VerificationEngine instance.

        Returns:
            The VerificationEngine used by this service.
        """
        return self.verification_engine

    def get_provenance(self) -> ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            The ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_export_import(self) -> ExportImportManager:
        """Get the ExportImportManager instance.

        Returns:
            The ExportImportManager used by this service.
        """
        return self.export_import

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get citations service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "citations_count": self.registry.count,
            "packages_count": self.evidence_manager.count,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the citations service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("CitationsService already started; skipping")
            return

        logger.info("CitationsService starting up...")
        self._started = True
        logger.info("CitationsService startup complete")

    def shutdown(self) -> None:
        """Shutdown the citations service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("CitationsService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> CitationsService:
    """Get or create the singleton CitationsService instance.

    Returns:
        The singleton CitationsService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = CitationsService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


def configure_citations_service(
    app: Any,
    config: Optional[CitationsConfig] = None,
) -> CitationsService:
    """Configure the Citations Service on a FastAPI application.

    Creates the CitationsService, stores it in app.state, mounts
    the citations API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional citations config.

    Returns:
        CitationsService instance.
    """
    global _singleton_instance

    service = CitationsService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.citations_service = service

    # Mount citations API router
    try:
        from greenlang.citations.api.router import router as citations_router
        if citations_router is not None:
            app.include_router(citations_router)
            logger.info("Citations service API router mounted")
    except ImportError:
        logger.warning("Citations router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("Citations service configured on app")
    return service


def get_citations_service(app: Any) -> CitationsService:
    """Get the CitationsService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        CitationsService instance.

    Raises:
        RuntimeError: If citations service not configured.
    """
    service = getattr(app.state, "citations_service", None)
    if service is None:
        raise RuntimeError(
            "Citations service not configured. "
            "Call configure_citations_service(app) first."
        )
    return service


def get_router() -> Any:
    """Get the citations API router.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.citations.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "CitationsService",
    "configure_citations_service",
    "get_citations_service",
    "get_router",
]
