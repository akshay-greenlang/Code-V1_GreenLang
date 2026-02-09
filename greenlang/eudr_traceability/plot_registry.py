# -*- coding: utf-8 -*-
"""
Plot Registry Engine - AGENT-DATA-004: EUDR Traceability Connector

Manages registration, validation, and retrieval of production plots
for EUDR-regulated commodities. Each plot represents a geographically
defined area where commodities are produced, with geolocation data
per EUDR Article 9, deforestation-free declarations, and compliance
tracking.

Zero-Hallucination Guarantees:
    - All geolocation validation is deterministic (bounds checking)
    - Polygon requirement (>4 ha) enforced per EUDR Article 9
    - SHA-256 provenance hashes on all plot operations
    - Risk levels assessed via deterministic scoring

Example:
    >>> from greenlang.eudr_traceability.plot_registry import PlotRegistryEngine
    >>> engine = PlotRegistryEngine()
    >>> plot = engine.register_plot(request)
    >>> assert plot.plot_id is not None

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.eudr_traceability.models import (
    EUDRCommodity,
    GeolocationData,
    LandUseType,
    PlotRecord,
    RegisterPlotRequest,
    RiskLevel,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash.

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class PlotRegistryEngine:
    """Production plot registration and management engine.

    Manages the lifecycle of production plots including registration,
    geolocation validation, compliance tracking, and risk-level
    indexing for EUDR-regulated commodities.

    Attributes:
        _config: Configuration dictionary or object.
        _plots: In-memory plot storage keyed by plot_id.
        _idx_commodity: Index of plot IDs by commodity.
        _idx_country: Index of plot IDs by country code.
        _idx_risk: Index of plot IDs by risk level.
        _provenance: Provenance tracker instance.

    Example:
        >>> engine = PlotRegistryEngine()
        >>> plot = engine.register_plot(request)
        >>> retrieved = engine.get_plot(plot.plot_id)
        >>> assert retrieved is not None
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize PlotRegistryEngine.

        Args:
            config: Optional EUDRTraceabilityConfig or dict.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._provenance = provenance

        # In-memory storage
        self._plots: Dict[str, PlotRecord] = {}

        # Indexes for fast lookup
        self._idx_commodity: Dict[str, List[str]] = {}
        self._idx_country: Dict[str, List[str]] = {}
        self._idx_risk: Dict[str, List[str]] = {}

        logger.info("PlotRegistryEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_plot(self, request: RegisterPlotRequest) -> PlotRecord:
        """Register a new production plot with validation and risk assessment.

        Validates geolocation data per EUDR Article 9, assesses initial
        risk level, creates the plot record, and indexes it for fast
        retrieval by commodity, country, and risk level.

        Args:
            request: Plot registration request with geolocation data.

        Returns:
            PlotRecord with generated plot_id and provenance hash.

        Raises:
            ValueError: If geolocation validation fails.
        """
        start_time = time.monotonic()

        # Step 1: Validate geolocation
        self._validate_geolocation(request.geolocation)

        # Step 2: Generate plot ID
        plot_id = self._generate_plot_id()

        # Step 3: Assess initial risk level
        risk_level = self._assess_initial_risk(
            request.geolocation.country_code,
            request.commodity,
        )

        # Step 4: Create plot record using new model field names
        plot = PlotRecord(
            plot_id=plot_id,
            commodity=request.commodity,
            geolocation=request.geolocation,
            producer_id=request.producer_id,
            producer_name=request.producer_name,
            country_code=request.country_code,
            land_use_type=request.land_use_type,
            deforestation_free=True,
            legal_compliance=True,
            supporting_documents=list(request.supporting_documents),
            risk_level=risk_level,
            certification=request.certification,
        )

        # Step 5: Store and index
        self._plots[plot_id] = plot
        self._index_plot(plot)

        # Step 6: Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(plot)
            self._provenance.record(
                entity_type="plot",
                entity_id=plot_id,
                action="plot_registration",
                data_hash=data_hash,
            )

        # Step 7: Record metrics
        try:
            from greenlang.eudr_traceability.metrics import (
                record_plot_registered,
                update_active_plots,
            )
            record_plot_registered(
                request.commodity.value,
                request.geolocation.country_code,
            )
            update_active_plots(1)
        except ImportError:
            pass

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Registered plot %s: commodity=%s, country=%s, risk=%s (%.1f ms)",
            plot_id, request.commodity.value,
            request.geolocation.country_code,
            risk_level.value if risk_level else "unknown",
            elapsed_ms,
        )
        return plot

    def get_plot(self, plot_id: str) -> Optional[PlotRecord]:
        """Get a plot record by ID.

        Args:
            plot_id: Plot identifier.

        Returns:
            PlotRecord or None if not found.
        """
        return self._plots.get(plot_id)

    def list_plots(
        self,
        commodity: Optional[str] = None,
        country: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[PlotRecord]:
        """List plots with optional filtering.

        Args:
            commodity: Optional commodity filter.
            country: Optional country code filter.
            risk_level: Optional risk level filter.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of PlotRecord instances matching filters.
        """
        # Start with candidate set from indexes
        candidates = self._get_filtered_ids(commodity, country, risk_level)
        if candidates is None:
            # No filter applied; use all plots
            plots = list(self._plots.values())
        else:
            plots = [
                self._plots[pid] for pid in candidates
                if pid in self._plots
            ]

        return plots[offset:offset + limit]

    def update_compliance(
        self,
        plot_id: str,
        deforestation_free: Optional[bool] = None,
        legal_compliance: Optional[bool] = None,
        supporting_docs: Optional[List[str]] = None,
    ) -> Optional[PlotRecord]:
        """Update compliance declarations for a plot.

        Args:
            plot_id: Plot identifier.
            deforestation_free: Updated deforestation-free status.
            legal_compliance: Updated legal compliance status.
            supporting_docs: Additional supporting document references.

        Returns:
            Updated PlotRecord or None if plot not found.
        """
        plot = self._plots.get(plot_id)
        if plot is None:
            logger.warning("Cannot update compliance: plot %s not found", plot_id)
            return None

        if deforestation_free is not None:
            plot.deforestation_free = deforestation_free
        if legal_compliance is not None:
            plot.legal_compliance = legal_compliance
        if supporting_docs:
            plot.supporting_documents.extend(supporting_docs)

        plot.updated_at = _utcnow()

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(plot)
            self._provenance.record(
                entity_type="plot",
                entity_id=plot_id,
                action="compliance_update",
                data_hash=data_hash,
            )

        logger.info(
            "Updated compliance for plot %s: deforestation_free=%s, legal=%s",
            plot_id, deforestation_free, legal_compliance,
        )
        return plot

    def bulk_import_plots(
        self,
        plots: List[RegisterPlotRequest],
    ) -> Dict[str, Any]:
        """Bulk import multiple plots.

        Args:
            plots: List of plot registration requests.

        Returns:
            Dictionary with success count, failure count, and results.
        """
        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        success_count = 0
        failure_count = 0

        for i, request in enumerate(plots):
            try:
                plot = self.register_plot(request)
                results.append({
                    "index": i,
                    "plot_id": plot.plot_id,
                    "status": "success",
                })
                success_count += 1
            except Exception as exc:
                results.append({
                    "index": i,
                    "status": "failed",
                    "error": str(exc),
                })
                failure_count += 1
                logger.warning("Bulk import failed at index %d: %s", i, exc)

        elapsed = time.monotonic() - start_time

        try:
            from greenlang.eudr_traceability.metrics import record_batch_operation
            record_batch_operation("bulk_import_plots", elapsed)
        except ImportError:
            pass

        logger.info(
            "Bulk import completed: %d success, %d failed in %.2fs",
            success_count, failure_count, elapsed,
        )
        return {
            "total": len(plots),
            "success": success_count,
            "failed": failure_count,
            "results": results,
            "duration_seconds": round(elapsed, 3),
        }

    def get_plots_by_commodity(
        self,
        commodity: str,
    ) -> List[PlotRecord]:
        """Get all plots for a specific commodity.

        Args:
            commodity: EUDR commodity value.

        Returns:
            List of PlotRecord instances.
        """
        plot_ids = self._idx_commodity.get(commodity, [])
        return [self._plots[pid] for pid in plot_ids if pid in self._plots]

    def get_plots_by_country(
        self,
        country_code: str,
    ) -> List[PlotRecord]:
        """Get all plots for a specific country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            List of PlotRecord instances.
        """
        code = country_code.upper()
        plot_ids = self._idx_country.get(code, [])
        return [self._plots[pid] for pid in plot_ids if pid in self._plots]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_geolocation(self, geo: GeolocationData) -> None:
        """Validate geolocation data per EUDR Article 9 requirements.

        Checks coordinate bounds, country code format, and polygon
        requirement for plots larger than 4 hectares.

        Args:
            geo: Geolocation data to validate.

        Raises:
            ValueError: If geolocation data is invalid.
        """
        # Latitude bounds
        if not -90.0 <= geo.latitude <= 90.0:
            raise ValueError(
                f"Latitude {geo.latitude} out of range [-90, 90]"
            )

        # Longitude bounds
        if not -180.0 <= geo.longitude <= 180.0:
            raise ValueError(
                f"Longitude {geo.longitude} out of range [-180, 180]"
            )

        # Country code format
        if len(geo.country_code) != 2:
            raise ValueError(
                f"Country code must be 2 characters, got '{geo.country_code}'"
            )

        # Polygon requirement for plots > 4 hectares (EUDR Article 9)
        threshold = 4.0
        if hasattr(self._config, "require_polygon_above_hectares"):
            threshold = self._config.require_polygon_above_hectares

        if geo.plot_area_hectares > threshold:
            if geo.polygon_coordinates is None or len(geo.polygon_coordinates) < 3:
                raise ValueError(
                    f"Plots larger than {threshold} hectares require a polygon "
                    f"with at least 3 coordinate pairs (Article 9). "
                    f"Plot area: {geo.plot_area_hectares} ha"
                )

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _generate_plot_id(self) -> str:
        """Generate a unique plot identifier.

        Returns:
            Plot ID in format "PLOT-{hex12}".
        """
        return f"PLOT-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def _assess_initial_risk(
        self,
        country_code: str,
        commodity: EUDRCommodity,
    ) -> RiskLevel:
        """Assess initial risk level based on country and commodity.

        Uses deterministic scoring: high-risk countries and commodities
        known for deforestation association receive higher scores.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            commodity: EUDR commodity type.

        Returns:
            RiskLevel classification.
        """
        high_risk_countries = {
            "BR", "ID", "MY", "AR", "PY", "BO", "CO", "PE",
            "EC", "CG", "CD", "CM", "CI", "GH", "NG", "LA",
            "MM", "PG",
        }
        high_risk_commodities = {
            EUDRCommodity.CATTLE,
            EUDRCommodity.SOYA,
            EUDRCommodity.OIL_PALM,
        }

        score = 0.0
        if country_code.upper() in high_risk_countries:
            score += 50.0
        if commodity in high_risk_commodities:
            score += 30.0

        # Determine thresholds from config
        low_threshold = 30.0
        high_threshold = 70.0
        if hasattr(self._config, "low_risk_threshold"):
            low_threshold = self._config.low_risk_threshold
        if hasattr(self._config, "high_risk_threshold"):
            high_threshold = self._config.high_risk_threshold

        if score >= high_threshold:
            return RiskLevel.HIGH
        elif score <= low_threshold:
            return RiskLevel.LOW
        else:
            return RiskLevel.STANDARD

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_plot(self, plot: PlotRecord) -> None:
        """Add a plot to all indexes.

        Args:
            plot: PlotRecord to index.
        """
        # Commodity index
        commodity_key = plot.commodity.value
        if commodity_key not in self._idx_commodity:
            self._idx_commodity[commodity_key] = []
        self._idx_commodity[commodity_key].append(plot.plot_id)

        # Country index
        country_key = plot.geolocation.country_code.upper()
        if country_key not in self._idx_country:
            self._idx_country[country_key] = []
        self._idx_country[country_key].append(plot.plot_id)

        # Risk level index
        if plot.risk_level is not None:
            risk_key = plot.risk_level.value
            if risk_key not in self._idx_risk:
                self._idx_risk[risk_key] = []
            self._idx_risk[risk_key].append(plot.plot_id)

    def _get_filtered_ids(
        self,
        commodity: Optional[str],
        country: Optional[str],
        risk_level: Optional[str],
    ) -> Optional[List[str]]:
        """Get plot IDs matching all specified filters.

        Args:
            commodity: Optional commodity filter.
            country: Optional country filter.
            risk_level: Optional risk level filter.

        Returns:
            List of matching plot IDs, or None if no filters.
        """
        if commodity is None and country is None and risk_level is None:
            return None

        sets: List[set] = []
        if commodity is not None:
            sets.append(set(self._idx_commodity.get(commodity, [])))
        if country is not None:
            sets.append(set(self._idx_country.get(country.upper(), [])))
        if risk_level is not None:
            sets.append(set(self._idx_risk.get(risk_level, [])))

        if not sets:
            return None

        result = sets[0]
        for s in sets[1:]:
            result = result.intersection(s)

        return list(result)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def plot_count(self) -> int:
        """Return the total number of registered plots."""
        return len(self._plots)


__all__ = [
    "PlotRegistryEngine",
]
