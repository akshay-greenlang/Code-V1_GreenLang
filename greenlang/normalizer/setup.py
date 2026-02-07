# -*- coding: utf-8 -*-
"""
Normalizer Service Setup - AGENT-FOUND-003: Unit & Reference Normalizer

Provides ``configure_normalizer_service(app)`` which wires up the
Normalizer SDK (converter, resolver, dimensional analyzer, provenance)
and mounts the REST API.

Also exposes ``get_normalizer_service(app)`` for programmatic access
and the ``NormalizerService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.normalizer.setup import configure_normalizer_service
    >>> app = FastAPI()
    >>> configure_normalizer_service(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from greenlang.normalizer.config import NormalizerConfig, get_config
from greenlang.normalizer.converter import UnitConverter
from greenlang.normalizer.dimensional import DimensionalAnalyzer
from greenlang.normalizer.entity_resolver import EntityResolver
from greenlang.normalizer.models import (
    BatchConversionResult,
    ConversionResult,
    EntityMatch,
    EntityResolutionResult,
)
from greenlang.normalizer.provenance import ConversionProvenanceTracker
from greenlang.normalizer.metrics import PROMETHEUS_AVAILABLE

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
# NormalizerService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["NormalizerService"] = None


class NormalizerService:
    """Unified facade over the Normalizer SDK.

    Manages unit conversion, entity resolution, dimensional analysis,
    and provenance tracking through a single entry point.

    Attributes:
        converter: UnitConverter instance.
        resolver: EntityResolver instance.
        analyzer: DimensionalAnalyzer instance.
        provenance_tracker: ConversionProvenanceTracker instance.
        config: NormalizerConfig instance.

    Example:
        >>> service = NormalizerService()
        >>> result = service.convert(100, "kWh", "MWh")
        >>> print(result.converted_value)  # Decimal('0.1')
    """

    def __init__(
        self,
        config: Optional[NormalizerConfig] = None,
    ) -> None:
        """Initialize the Normalizer Service facade.

        Args:
            config: Optional normalizer config. Uses global config if None.
        """
        self.config = config or get_config()
        self.converter = UnitConverter(self.config)
        self.resolver = EntityResolver()
        self.analyzer = DimensionalAnalyzer()
        self.provenance_tracker = ConversionProvenanceTracker()
        self._started = False

        logger.info("NormalizerService facade created")

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(
        self,
        value: Any,
        from_unit: str,
        to_unit: str,
        precision: Optional[int] = None,
    ) -> ConversionResult:
        """Convert a value from one unit to another.

        Delegates to UnitConverter.convert() with provenance recording.

        Args:
            value: Numeric value to convert.
            from_unit: Source unit.
            to_unit: Target unit.
            precision: Decimal places override.

        Returns:
            ConversionResult with Decimal precision.
        """
        result = self.converter.convert(value, from_unit, to_unit, precision)

        # Record provenance
        self.provenance_tracker.record_conversion(
            input_data={"value": str(value), "from_unit": from_unit, "to_unit": to_unit},
            output_data={"converted_value": str(result.converted_value)},
            factors={"conversion_factor": str(result.conversion_factor)},
        )

        return result

    def batch_convert(
        self,
        items: List[Dict[str, Any]],
    ) -> BatchConversionResult:
        """Convert a batch of items.

        Delegates to UnitConverter.batch_convert().

        Args:
            items: List of conversion request dicts.

        Returns:
            BatchConversionResult with all results.
        """
        return self.converter.batch_convert(items)

    # ------------------------------------------------------------------
    # Entity Resolution
    # ------------------------------------------------------------------

    def resolve_fuel(self, name: str) -> EntityMatch:
        """Resolve a fuel name to canonical form.

        Args:
            name: Raw fuel name.

        Returns:
            EntityMatch with resolution result.
        """
        result = self.resolver.resolve_fuel(name)

        self.provenance_tracker.record_resolution(
            input_name=name,
            resolved_entity={
                "canonical_name": result.canonical_name,
                "resolved_id": result.resolved_id,
            },
        )

        return result

    def resolve_material(self, name: str) -> EntityMatch:
        """Resolve a material name to canonical form.

        Args:
            name: Raw material name.

        Returns:
            EntityMatch with resolution result.
        """
        result = self.resolver.resolve_material(name)

        self.provenance_tracker.record_resolution(
            input_name=name,
            resolved_entity={
                "canonical_name": result.canonical_name,
                "resolved_id": result.resolved_id,
            },
        )

        return result

    def resolve_process(self, name: str) -> EntityMatch:
        """Resolve a process name to canonical form.

        Args:
            name: Raw process name.

        Returns:
            EntityMatch with resolution result.
        """
        result = self.resolver.resolve_process(name)

        self.provenance_tracker.record_resolution(
            input_name=name,
            resolved_entity={
                "canonical_name": result.canonical_name,
                "resolved_id": result.resolved_id,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Accessor methods
    # ------------------------------------------------------------------

    def get_converter(self) -> UnitConverter:
        """Get the UnitConverter instance.

        Returns:
            The UnitConverter used by this service.
        """
        return self.converter

    def get_resolver(self) -> EntityResolver:
        """Get the EntityResolver instance.

        Returns:
            The EntityResolver used by this service.
        """
        return self.resolver

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get normalizer service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "provenance_operations": len(
                self.provenance_tracker.get_audit_trail()
            ),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the normalizer service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("NormalizerService already started; skipping")
            return

        logger.info("NormalizerService starting up...")
        self._started = True
        logger.info("NormalizerService startup complete")

    def shutdown(self) -> None:
        """Shutdown the normalizer service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("NormalizerService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> NormalizerService:
    """Get or create the singleton NormalizerService instance.

    Returns:
        The singleton NormalizerService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = NormalizerService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


def configure_normalizer_service(
    app: Any,
    config: Optional[NormalizerConfig] = None,
) -> NormalizerService:
    """Configure the Normalizer Service on a FastAPI application.

    Creates the NormalizerService, stores it in app.state, mounts
    the normalizer API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional normalizer config.

    Returns:
        NormalizerService instance.
    """
    global _singleton_instance

    service = NormalizerService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.normalizer_service = service

    # Mount normalizer API router
    try:
        from greenlang.normalizer.api.router import router as normalizer_router
        if normalizer_router is not None:
            app.include_router(normalizer_router)
            logger.info("Normalizer service API router mounted")
    except ImportError:
        logger.warning("Normalizer router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("Normalizer service configured on app")
    return service


def get_normalizer_service(app: Any) -> NormalizerService:
    """Get the NormalizerService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        NormalizerService instance.

    Raises:
        RuntimeError: If normalizer service not configured.
    """
    service = getattr(app.state, "normalizer_service", None)
    if service is None:
        raise RuntimeError(
            "Normalizer service not configured. "
            "Call configure_normalizer_service(app) first."
        )
    return service


__all__ = [
    "NormalizerService",
    "configure_normalizer_service",
    "get_normalizer_service",
]
