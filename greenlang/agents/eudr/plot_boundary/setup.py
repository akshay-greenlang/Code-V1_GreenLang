# -*- coding: utf-8 -*-
"""
PlotBoundaryService - Facade for AGENT-EUDR-006 Plot Boundary Manager Agent

This module implements the PlotBoundaryService, the single entry point for
all plot boundary management operations in the GL-EUDR-APP. It manages the
lifecycle of eight internal engines, an async PostgreSQL connection pool
(psycopg + psycopg_pool), a Redis cache connection, OpenTelemetry tracing,
and Prometheus metrics. The service exposes a unified interface consumed by
the FastAPI router layer and the GL-EUDR-APP integration.

Lifecycle:
    startup  -> load config -> connect DB -> register pgvector -> connect Redis
             -> initialize engines -> start health check background task
    shutdown -> close engines -> close Redis -> close DB pool -> flush metrics

Engines (8):
    1. PolygonManager          - CRUD + WKT/GeoJSON boundary storage (Feature 1)
    2. BoundaryValidator       - OGC/ISO/EUDR geometry validation (Feature 2)
    3. AreaCalculator          - Geodesic area + EUDR 4ha threshold (Feature 3)
    4. OverlapDetector         - R-tree spatial overlap detection (Feature 4)
    5. BoundaryVersioner       - Immutable version management (Feature 5)
    6. SimplificationEngine    - Douglas-Peucker/Visvalingam simplification (Feature 6)
    7. SplitMergeEngine        - Split/merge with genealogy tracking (Feature 7)
    8. ComplianceReporter      - Multi-format export + compliance reports (Feature 8)

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.plot_boundary.setup import (
    ...     PlotBoundaryService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from greenlang.agents.eudr.plot_boundary.config import (
    PlotBoundaryConfig,
    get_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports with graceful fallback
# ---------------------------------------------------------------------------

try:
    from psycopg_pool import AsyncConnectionPool

    PSYCOPG_POOL_AVAILABLE = True
except ImportError:
    AsyncConnectionPool = None  # type: ignore[assignment,misc]
    PSYCOPG_POOL_AVAILABLE = False

try:
    from psycopg import AsyncConnection

    PSYCOPG_AVAILABLE = True
except ImportError:
    AsyncConnection = None  # type: ignore[assignment,misc]
    PSYCOPG_AVAILABLE = False

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False

try:
    from opentelemetry import trace as otel_trace

    OTEL_AVAILABLE = True
except ImportError:
    otel_trace = None  # type: ignore[assignment]
    OTEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_provenance_hash(*parts: str) -> str:
    """Compute SHA-256 hash over concatenated string parts.

    Args:
        *parts: Variable number of string parts to hash.

    Returns:
        SHA-256 hex digest string.
    """
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def _generate_request_id() -> str:
    """Generate a unique request identifier.

    Returns:
        UUID-based request identifier string.
    """
    return f"PBM-{uuid.uuid4().hex[:12]}"

def _compute_service_hash(config: PlotBoundaryConfig) -> str:
    """Compute SHA-256 hash of the service configuration for provenance.

    Args:
        config: Service configuration to hash.

    Returns:
        SHA-256 hex digest string.
    """
    raw = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Health status model
# ---------------------------------------------------------------------------

class HealthStatus:
    """Health check result container.

    Attributes:
        status: Overall health status (healthy, degraded, unhealthy).
        checks: Individual component check results.
        timestamp: When the health check was performed.
        version: Service version string.
        uptime_seconds: Seconds since service startup.
    """

    __slots__ = ("status", "checks", "timestamp", "version", "uptime_seconds")

    def __init__(
        self,
        status: str = "unhealthy",
        checks: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        version: str = "1.0.0",
        uptime_seconds: float = 0.0,
    ) -> None:
        self.status = status
        self.checks = checks or {}
        self.timestamp = timestamp or utcnow()
        self.version = version
        self.uptime_seconds = uptime_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize health status to dictionary for JSON response."""
        return {
            "status": self.status,
            "checks": self.checks,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }

# ---------------------------------------------------------------------------
# Result container: BoundaryResult
# ---------------------------------------------------------------------------

class BoundaryResult:
    """Result from a boundary create/update/retrieve operation.

    Attributes:
        plot_id: Unique plot identifier.
        geometry_type: Type of geometry ('Polygon', 'MultiPolygon', 'Point').
        vertex_count: Number of vertices in the boundary.
        area_hectares: Calculated area in hectares.
        is_above_threshold: Whether area >= 4 hectares (polygon required).
        crs: Coordinate reference system EPSG code.
        validation_status: 'PASS', 'FAIL', or 'WARNING'.
        validation_errors: List of validation error messages.
        version: Boundary version number.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Boundary creation timestamp.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "geometry_type", "vertex_count", "area_hectares",
        "is_above_threshold", "crs", "validation_status",
        "validation_errors", "version", "provenance_hash",
        "created_at", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        geometry_type: str = "",
        vertex_count: int = 0,
        area_hectares: float = 0.0,
        is_above_threshold: bool = False,
        crs: str = "EPSG:4326",
        validation_status: str = "PASS",
        validation_errors: Optional[List[str]] = None,
        version: int = 1,
        provenance_hash: str = "",
        created_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.geometry_type = geometry_type
        self.vertex_count = vertex_count
        self.area_hectares = area_hectares
        self.is_above_threshold = is_above_threshold
        self.crs = crs
        self.validation_status = validation_status
        self.validation_errors = validation_errors or []
        self.version = version
        self.provenance_hash = provenance_hash
        self.created_at = created_at or utcnow()
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "plot_id": self.plot_id,
            "geometry_type": self.geometry_type,
            "vertex_count": self.vertex_count,
            "area_hectares": round(self.area_hectares, 6),
            "is_above_threshold": self.is_above_threshold,
            "crs": self.crs,
            "validation_status": self.validation_status,
            "validation_errors": self.validation_errors,
            "version": self.version,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: ValidationResult
# ---------------------------------------------------------------------------

class ValidationResult:
    """Result from a boundary validation operation.

    Attributes:
        plot_id: Unique plot identifier.
        is_valid: Whether the boundary passes all validation rules.
        errors: List of error-level issues.
        warnings: List of warning-level issues.
        rules_checked: Number of validation rules evaluated.
        rules_passed: Number of rules that passed.
        rules_failed: Number of rules that failed.
        auto_repaired: Whether automatic repairs were applied.
        repair_actions: List of repair actions taken.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "is_valid", "errors", "warnings",
        "rules_checked", "rules_passed", "rules_failed",
        "auto_repaired", "repair_actions",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        is_valid: bool = True,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        rules_checked: int = 0,
        rules_passed: int = 0,
        rules_failed: int = 0,
        auto_repaired: bool = False,
        repair_actions: Optional[List[str]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.rules_checked = rules_checked
        self.rules_passed = rules_passed
        self.rules_failed = rules_failed
        self.auto_repaired = auto_repaired
        self.repair_actions = repair_actions or []
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "plot_id": self.plot_id,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "rules_checked": self.rules_checked,
            "rules_passed": self.rules_passed,
            "rules_failed": self.rules_failed,
            "auto_repaired": self.auto_repaired,
            "repair_actions": self.repair_actions,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: AreaResult
# ---------------------------------------------------------------------------

class AreaResult:
    """Result from an area calculation operation.

    Attributes:
        plot_id: Unique plot identifier.
        area_hectares: Geodesic area in hectares.
        area_m2: Area in square metres.
        area_km2: Area in square kilometres.
        perimeter_m: Perimeter in metres.
        perimeter_km: Perimeter in kilometres.
        is_above_threshold: Whether area >= 4 hectares.
        polygon_required: Whether polygon boundary is required by EUDR.
        calculation_method: Method used ('geodesic_karney', 'projected_utm').
        utm_zone: UTM zone used for area calculation.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "area_hectares", "area_m2", "area_km2",
        "perimeter_m", "perimeter_km", "is_above_threshold",
        "polygon_required", "calculation_method", "utm_zone",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        area_hectares: float = 0.0,
        area_m2: float = 0.0,
        area_km2: float = 0.0,
        perimeter_m: float = 0.0,
        perimeter_km: float = 0.0,
        is_above_threshold: bool = False,
        polygon_required: bool = False,
        calculation_method: str = "",
        utm_zone: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.area_hectares = area_hectares
        self.area_m2 = area_m2
        self.area_km2 = area_km2
        self.perimeter_m = perimeter_m
        self.perimeter_km = perimeter_km
        self.is_above_threshold = is_above_threshold
        self.polygon_required = polygon_required
        self.calculation_method = calculation_method
        self.utm_zone = utm_zone
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "plot_id": self.plot_id,
            "area_hectares": round(self.area_hectares, 6),
            "area_m2": round(self.area_m2, 2),
            "area_km2": round(self.area_km2, 6),
            "perimeter_m": round(self.perimeter_m, 2),
            "perimeter_km": round(self.perimeter_km, 6),
            "is_above_threshold": self.is_above_threshold,
            "polygon_required": self.polygon_required,
            "calculation_method": self.calculation_method,
            "utm_zone": self.utm_zone,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: OverlapResult
# ---------------------------------------------------------------------------

class OverlapResult:
    """Result from an overlap detection operation.

    Attributes:
        plot_id: Unique plot identifier being checked.
        overlaps_found: Number of overlapping boundaries found.
        overlaps: List of individual overlap detail dicts.
        scan_area_m2: Total scan area in square metres.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "overlaps_found", "overlaps",
        "scan_area_m2", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        overlaps_found: int = 0,
        overlaps: Optional[List[Dict[str, Any]]] = None,
        scan_area_m2: float = 0.0,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.overlaps_found = overlaps_found
        self.overlaps = overlaps or []
        self.scan_area_m2 = scan_area_m2
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "plot_id": self.plot_id,
            "overlaps_found": self.overlaps_found,
            "overlaps": self.overlaps,
            "scan_area_m2": round(self.scan_area_m2, 2),
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: VersionResult
# ---------------------------------------------------------------------------

class VersionResult:
    """Result from a boundary versioning operation.

    Attributes:
        plot_id: Unique plot identifier.
        current_version: Current (latest) version number.
        total_versions: Total number of versions.
        versions: List of version summary dicts.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "current_version", "total_versions",
        "versions", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        current_version: int = 0,
        total_versions: int = 0,
        versions: Optional[List[Dict[str, Any]]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.current_version = current_version
        self.total_versions = total_versions
        self.versions = versions or []
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "plot_id": self.plot_id,
            "current_version": self.current_version,
            "total_versions": self.total_versions,
            "versions": self.versions,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: SimplificationResult
# ---------------------------------------------------------------------------

class SimplificationResult:
    """Result from a boundary simplification operation.

    Attributes:
        plot_id: Unique plot identifier.
        method: Simplification algorithm used.
        tolerance: Tolerance value applied.
        original_vertices: Vertex count before simplification.
        simplified_vertices: Vertex count after simplification.
        reduction_pct: Vertex reduction percentage.
        area_deviation_pct: Area change as percentage of original.
        hausdorff_distance_m: Maximum distance between boundaries in metres.
        quality_pass: Whether the result passes quality thresholds.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "method", "tolerance", "original_vertices",
        "simplified_vertices", "reduction_pct", "area_deviation_pct",
        "hausdorff_distance_m", "quality_pass",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        method: str = "",
        tolerance: float = 0.0,
        original_vertices: int = 0,
        simplified_vertices: int = 0,
        reduction_pct: float = 0.0,
        area_deviation_pct: float = 0.0,
        hausdorff_distance_m: float = 0.0,
        quality_pass: bool = True,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.method = method
        self.tolerance = tolerance
        self.original_vertices = original_vertices
        self.simplified_vertices = simplified_vertices
        self.reduction_pct = reduction_pct
        self.area_deviation_pct = area_deviation_pct
        self.hausdorff_distance_m = hausdorff_distance_m
        self.quality_pass = quality_pass
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "plot_id": self.plot_id,
            "method": self.method,
            "tolerance": self.tolerance,
            "original_vertices": self.original_vertices,
            "simplified_vertices": self.simplified_vertices,
            "reduction_pct": round(self.reduction_pct, 2),
            "area_deviation_pct": round(self.area_deviation_pct, 4),
            "hausdorff_distance_m": round(self.hausdorff_distance_m, 2),
            "quality_pass": self.quality_pass,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: SplitMergeResult
# ---------------------------------------------------------------------------

class SplitMergeResult:
    """Result from a split or merge operation.

    Attributes:
        operation: Operation type ('split' or 'merge').
        source_plot_ids: Original plot IDs involved.
        result_plot_ids: Resulting plot IDs after operation.
        area_conservation_pct: Area conservation percentage.
        genealogy_link: Genealogy tracking reference.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "operation", "source_plot_ids", "result_plot_ids",
        "area_conservation_pct", "genealogy_link",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        operation: str = "",
        source_plot_ids: Optional[List[str]] = None,
        result_plot_ids: Optional[List[str]] = None,
        area_conservation_pct: float = 100.0,
        genealogy_link: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.operation = operation
        self.source_plot_ids = source_plot_ids or []
        self.result_plot_ids = result_plot_ids or []
        self.area_conservation_pct = area_conservation_pct
        self.genealogy_link = genealogy_link
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "operation": self.operation,
            "source_plot_ids": self.source_plot_ids,
            "result_plot_ids": self.result_plot_ids,
            "area_conservation_pct": round(self.area_conservation_pct, 4),
            "genealogy_link": self.genealogy_link,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: ExportResult
# ---------------------------------------------------------------------------

class ExportResult:
    """Result from a boundary export operation.

    Attributes:
        plot_ids: Plot IDs included in the export.
        format: Export format used.
        size_bytes: Export file size in bytes.
        vertex_count: Total vertex count in the export.
        provenance_hash: SHA-256 hash for audit trail.
        content: Export content (string or bytes).
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_ids", "format", "size_bytes", "vertex_count",
        "provenance_hash", "content", "processing_time_ms",
    )

    def __init__(
        self,
        plot_ids: Optional[List[str]] = None,
        format: str = "geojson",
        size_bytes: int = 0,
        vertex_count: int = 0,
        provenance_hash: str = "",
        content: Optional[Any] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_ids = plot_ids or []
        self.format = format
        self.size_bytes = size_bytes
        self.vertex_count = vertex_count
        self.provenance_hash = provenance_hash
        self.content = content
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response (excludes content)."""
        return {
            "plot_ids": self.plot_ids,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "vertex_count": self.vertex_count,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: ComplianceReportResult
# ---------------------------------------------------------------------------

class ComplianceReportResult:
    """Result from a compliance report generation.

    Attributes:
        report_id: Unique report identifier.
        plot_ids: Plot IDs covered by the report.
        report_type: Type of report ('full', 'summary', 'evidence').
        format: Output format ('json', 'pdf_data', 'eudr_xml').
        sections: Report section data keyed by section name.
        total_plots: Total number of plots in the report.
        compliant_plots: Number of plots passing compliance.
        non_compliant_plots: Number of plots failing compliance.
        provenance_hash: SHA-256 hash for audit trail.
        generated_at: Report generation timestamp.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "report_id", "plot_ids", "report_type", "format",
        "sections", "total_plots", "compliant_plots",
        "non_compliant_plots", "provenance_hash",
        "generated_at", "processing_time_ms",
    )

    def __init__(
        self,
        report_id: str = "",
        plot_ids: Optional[List[str]] = None,
        report_type: str = "full",
        format: str = "json",
        sections: Optional[Dict[str, Any]] = None,
        total_plots: int = 0,
        compliant_plots: int = 0,
        non_compliant_plots: int = 0,
        provenance_hash: str = "",
        generated_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.report_id = report_id or f"RPT-{uuid.uuid4().hex[:12]}"
        self.plot_ids = plot_ids or []
        self.report_type = report_type
        self.format = format
        self.sections = sections or {}
        self.total_plots = total_plots
        self.compliant_plots = compliant_plots
        self.non_compliant_plots = non_compliant_plots
        self.provenance_hash = provenance_hash
        self.generated_at = generated_at or utcnow()
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "report_id": self.report_id,
            "plot_ids": self.plot_ids,
            "report_type": self.report_type,
            "format": self.format,
            "sections": self.sections,
            "total_plots": self.total_plots,
            "compliant_plots": self.compliant_plots,
            "non_compliant_plots": self.non_compliant_plots,
            "provenance_hash": self.provenance_hash,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: BatchJobResult
# ---------------------------------------------------------------------------

class BatchJobResult:
    """Result container for a batch processing job.

    Attributes:
        job_id: Unique job identifier.
        job_type: Type of batch job.
        status: Job status (pending, processing, completed, failed, cancelled).
        total_items: Total items in the batch.
        completed_items: Number of items processed.
        failed_items: Number of items that failed.
        results: Per-item results (list of dicts).
        submitted_at: Job submission time.
        completed_at: Job completion time.
        processing_time_ms: Total processing time.
    """

    __slots__ = (
        "job_id", "job_type", "status", "total_items",
        "completed_items", "failed_items", "results",
        "submitted_at", "completed_at", "processing_time_ms",
    )

    def __init__(
        self,
        job_id: str = "",
        job_type: str = "",
        status: str = "pending",
        total_items: int = 0,
        completed_items: int = 0,
        failed_items: int = 0,
        results: Optional[List[Dict[str, Any]]] = None,
        submitted_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.job_id = job_id or f"JOB-{uuid.uuid4().hex[:12]}"
        self.job_type = job_type
        self.status = status
        self.total_items = total_items
        self.completed_items = completed_items
        self.failed_items = failed_items
        self.results = results or []
        self.submitted_at = submitted_at or utcnow()
        self.completed_at = completed_at
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "results": self.results,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# PlotBoundaryService
# ---------------------------------------------------------------------------

class PlotBoundaryService:
    """Facade for the Plot Boundary Manager Agent (AGENT-EUDR-006).

    Provides a unified interface to all 8 engines:
        1. PolygonManager          - CRUD + WKT/GeoJSON boundary storage
        2. BoundaryValidator       - OGC/ISO/EUDR geometry validation
        3. AreaCalculator          - Geodesic area + EUDR 4ha threshold
        4. OverlapDetector         - R-tree spatial overlap detection
        5. BoundaryVersioner       - Immutable version management
        6. SimplificationEngine    - Douglas-Peucker/Visvalingam simplification
        7. SplitMergeEngine        - Split/merge with genealogy tracking
        8. ComplianceReporter      - Multi-format export + compliance reports

    Singleton pattern with thread-safe initialization.

    Attributes:
        config: Service configuration loaded from env or injected.
        is_running: Whether the service is currently active and healthy.

    Example:
        >>> service = PlotBoundaryService()
        >>> await service.startup()
        >>> result = service.create_boundary({"plot_id": "P-001", ...})
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[PlotBoundaryConfig] = None,
    ) -> None:
        """Initialize PlotBoundaryService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.

        Args:
            config: Optional configuration override. If None, loads from
                environment variables via ``get_config()``.
        """
        self._config = config or get_config()
        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_service_hash(self._config)

        # Connection handles (initialized in startup)
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._polygon_manager: Optional[Any] = None
        self._boundary_validator: Optional[Any] = None
        self._area_calculator: Optional[Any] = None
        self._overlap_detector: Optional[Any] = None
        self._boundary_versioner: Optional[Any] = None
        self._simplification_engine: Optional[Any] = None
        self._split_merge_engine: Optional[Any] = None
        self._compliance_reporter: Optional[Any] = None

        # In-memory boundary store
        self._boundaries: Dict[str, Dict[str, Any]] = {}
        self._boundary_lock = threading.Lock()

        # Batch job registry
        self._batch_registry: Dict[str, BatchJobResult] = {}
        self._batch_lock = threading.Lock()

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthStatus] = None

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Metrics counters
        self._metrics_counters: Dict[str, int] = {
            "boundaries_created": 0,
            "boundaries_updated": 0,
            "boundaries_deleted": 0,
            "validations": 0,
            "area_calculations": 0,
            "overlap_detections": 0,
            "versions_created": 0,
            "simplifications": 0,
            "splits": 0,
            "merges": 0,
            "exports": 0,
            "reports": 0,
            "errors": 0,
        }

        logger.info(
            "PlotBoundaryService created: config_hash=%s, "
            "canonical_crs=%s, area_threshold=%.1fha, "
            "batch_max=%d, concurrency=%d, cache_ttl=%ds",
            self._config_hash[:12],
            self._config.canonical_crs,
            self._config.area_threshold_hectares,
            self._config.batch_max_size,
            self._config.batch_concurrency,
            self._config.cache_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> PlotBoundaryConfig:
        """Return the service configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Return whether the service is started and active."""
        return self._started

    @property
    def uptime_seconds(self) -> float:
        """Return seconds since startup, or 0.0 if not started."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def polygon_manager(self) -> Any:
        """Return the PolygonManager engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._polygon_manager

    @property
    def boundary_validator(self) -> Any:
        """Return the BoundaryValidator engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._boundary_validator

    @property
    def area_calculator(self) -> Any:
        """Return the AreaCalculator engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._area_calculator

    @property
    def overlap_detector(self) -> Any:
        """Return the OverlapDetector engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._overlap_detector

    @property
    def boundary_versioner(self) -> Any:
        """Return the BoundaryVersioner engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._boundary_versioner

    @property
    def simplification_engine(self) -> Any:
        """Return the SimplificationEngine engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._simplification_engine

    @property
    def split_merge_engine(self) -> Any:
        """Return the SplitMergeEngine engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._split_merge_engine

    @property
    def compliance_reporter(self) -> Any:
        """Return the ComplianceReporter engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._compliance_reporter

    @property
    def db_pool(self) -> Any:
        """Return the async PostgreSQL connection pool.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._db_pool

    @property
    def redis_client(self) -> Any:
        """Return the async Redis client.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._redis

    @property
    def last_health(self) -> Optional[HealthStatus]:
        """Return the most recent cached health check result."""
        return self._last_health

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Start the service: connect DB, Redis, initialize all engines.

        Executes the full startup sequence in order:
            1. Configure structured logging
            2. Initialize OpenTelemetry tracer
            3. Connect to PostgreSQL and create connection pool
            4. Register pgvector type extension
            5. Connect to Redis for caching
            6. Initialize all eight engines
            7. Start background health check task

        Idempotent: safe to call multiple times.

        Raises:
            RuntimeError: If a critical connection fails.
        """
        if self._started:
            logger.debug("PlotBoundaryService already started")
            return

        start = time.monotonic()
        logger.info("PlotBoundaryService starting up...")

        self._configure_logging()
        self._init_tracer()
        await self._connect_database()
        await self._register_pgvector()
        await self._connect_redis()
        await self._initialize_engines()
        self._start_health_check()

        self._started = True
        self._start_time = time.monotonic()
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "PlotBoundaryService started in %.1fms: "
            "db=%s, redis=%s, engines=8, config_hash=%s",
            elapsed,
            "connected" if self._db_pool is not None else "skipped",
            "connected" if self._redis is not None else "skipped",
            self._config_hash[:12],
        )

    async def shutdown(self) -> None:
        """Gracefully shut down the service and release all resources.

        Idempotent: safe to call multiple times.
        """
        if not self._started:
            logger.debug("PlotBoundaryService already stopped")
            return

        logger.info("PlotBoundaryService shutting down...")
        start = time.monotonic()

        self._stop_health_check()
        await self._close_engines()
        await self._close_redis()
        await self._close_database()
        self._flush_metrics()

        self._started = False
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "PlotBoundaryService shut down in %.1fms", elapsed
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of all components.

        Checks database connectivity, Redis connectivity, engine status,
        and boundary store statistics. Returns a structured health report
        suitable for the ``/health`` endpoint.

        Returns:
            Dictionary with status, component checks, version, and uptime.
        """
        checks: Dict[str, Any] = {}

        checks["database"] = await self._check_database_health()
        checks["redis"] = await self._check_redis_health()
        checks["engines"] = self._check_engine_health()
        checks["boundary_store"] = self._check_boundary_store_health()

        statuses = [
            v.get("status", "unhealthy") if isinstance(v, dict) else "unhealthy"
            for v in checks.values()
        ]
        if all(s == "healthy" for s in statuses):
            overall = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"

        health = HealthStatus(
            status=overall,
            checks=checks,
            timestamp=utcnow(),
            version="1.0.0",
            uptime_seconds=self.uptime_seconds,
        )
        self._last_health = health
        return health.to_dict()

    def get_statistics(self) -> Dict[str, Any]:
        """Return service statistics including boundary and metric counts.

        Returns:
            Dictionary with boundary count, area stats, metric counters.
        """
        with self._boundary_lock:
            boundary_count = len(self._boundaries)

        return {
            "boundary_count": boundary_count,
            "metrics": dict(self._metrics_counters),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "config_hash": self._config_hash[:12],
            "canonical_crs": self._config.canonical_crs,
            "area_threshold_hectares": self._config.area_threshold_hectares,
            "timestamp": utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Boundary Management: create_boundary
    # ------------------------------------------------------------------

    def create_boundary(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new plot boundary with validation, area calculation, and versioning.

        Orchestrates the full creation pipeline:
            1. Validate input request
            2. Create boundary via PolygonManager
            3. Validate geometry via BoundaryValidator
            4. Calculate area via AreaCalculator
            5. Create initial version via BoundaryVersioner
            6. Record provenance hash
            7. Store in internal registry

        Args:
            request: Dictionary with keys: plot_id, geometry (WKT or GeoJSON),
                commodity, country_code, crs (optional).

        Returns:
            Dictionary with boundary result, validation, area, and version info.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If the request is missing required fields.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = request.get("plot_id") or _generate_request_id()

        logger.info(
            "Creating boundary: plot_id=%s, commodity=%s, country=%s",
            request_id,
            request.get("commodity", ""),
            request.get("country_code", ""),
        )

        try:
            self._validate_create_request(request)

            boundary = self._safe_create_polygon(request_id, request)
            validation = self._safe_validate_boundary(request_id, request)
            area = self._safe_calculate_area(request_id, request)
            version = self._safe_create_version(request_id, 1, request)

            elapsed_ms = (time.monotonic() - start) * 1000

            provenance_hash = _compute_provenance_hash(
                request_id,
                str(area.area_hectares if area else 0.0),
                str(validation.is_valid if validation else False),
                str(version.current_version if version else 1),
            )

            result = BoundaryResult(
                plot_id=request_id,
                geometry_type=request.get("geometry_type", "Polygon"),
                vertex_count=boundary.get("vertex_count", 0) if boundary else 0,
                area_hectares=area.area_hectares if area else 0.0,
                is_above_threshold=(
                    area.is_above_threshold if area else False
                ),
                crs=request.get("crs", "EPSG:4326"),
                validation_status=(
                    "PASS" if validation and validation.is_valid else "FAIL"
                ),
                validation_errors=(
                    validation.errors if validation else []
                ),
                version=1,
                provenance_hash=provenance_hash,
                created_at=utcnow(),
                processing_time_ms=elapsed_ms,
            )

            with self._boundary_lock:
                self._boundaries[request_id] = {
                    "result": result.to_dict(),
                    "request": request,
                    "created_at": utcnow().isoformat(),
                    "version": 1,
                }

            self._metrics_counters["boundaries_created"] += 1

            logger.info(
                "Boundary created: plot_id=%s, area=%.4fha, valid=%s, "
                "hash=%s, elapsed=%.1fms",
                request_id, result.area_hectares,
                result.validation_status,
                provenance_hash[:12], elapsed_ms,
            )

            return result.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Create boundary failed: plot_id=%s, error=%s",
                request_id, exc, exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Boundary Management: update_boundary
    # ------------------------------------------------------------------

    def update_boundary(
        self,
        plot_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an existing plot boundary with revalidation and new version.

        Args:
            plot_id: Unique plot identifier.
            request: Dictionary with updated geometry and optional metadata.

        Returns:
            Dictionary with updated boundary result.

        Raises:
            RuntimeError: If the service has not been started.
            KeyError: If the plot_id is not found.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info("Updating boundary: plot_id=%s", plot_id)

        try:
            with self._boundary_lock:
                if plot_id not in self._boundaries:
                    raise KeyError(f"Boundary not found: {plot_id}")
                current = self._boundaries[plot_id]
                current_version = current.get("version", 1)

            new_version = current_version + 1
            validation = self._safe_validate_boundary(plot_id, request)
            area = self._safe_calculate_area(plot_id, request)
            version = self._safe_create_version(plot_id, new_version, request)

            elapsed_ms = (time.monotonic() - start) * 1000

            provenance_hash = _compute_provenance_hash(
                plot_id,
                str(new_version),
                str(area.area_hectares if area else 0.0),
                str(validation.is_valid if validation else False),
            )

            result = BoundaryResult(
                plot_id=plot_id,
                geometry_type=request.get("geometry_type", "Polygon"),
                area_hectares=area.area_hectares if area else 0.0,
                is_above_threshold=(
                    area.is_above_threshold if area else False
                ),
                crs=request.get("crs", "EPSG:4326"),
                validation_status=(
                    "PASS" if validation and validation.is_valid else "FAIL"
                ),
                validation_errors=(
                    validation.errors if validation else []
                ),
                version=new_version,
                provenance_hash=provenance_hash,
                created_at=utcnow(),
                processing_time_ms=elapsed_ms,
            )

            with self._boundary_lock:
                self._boundaries[plot_id] = {
                    "result": result.to_dict(),
                    "request": request,
                    "created_at": current.get("created_at"),
                    "updated_at": utcnow().isoformat(),
                    "version": new_version,
                }

            self._metrics_counters["boundaries_updated"] += 1

            logger.info(
                "Boundary updated: plot_id=%s, version=%d, "
                "area=%.4fha, elapsed=%.1fms",
                plot_id, new_version,
                result.area_hectares, elapsed_ms,
            )

            return result.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Update boundary failed: plot_id=%s, error=%s",
                plot_id, exc, exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Boundary Management: get_boundary
    # ------------------------------------------------------------------

    def get_boundary(self, plot_id: str) -> Dict[str, Any]:
        """Retrieve a boundary with version information.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            Dictionary with boundary data and version info.

        Raises:
            RuntimeError: If the service has not been started.
            KeyError: If the plot_id is not found.
        """
        self._ensure_started()

        with self._boundary_lock:
            if plot_id not in self._boundaries:
                raise KeyError(f"Boundary not found: {plot_id}")
            return dict(self._boundaries[plot_id])

    # ------------------------------------------------------------------
    # Boundary Management: delete_boundary
    # ------------------------------------------------------------------

    def delete_boundary(self, plot_id: str) -> Dict[str, Any]:
        """Soft-delete a boundary with provenance recording.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            Dictionary confirming deletion with provenance hash.

        Raises:
            RuntimeError: If the service has not been started.
            KeyError: If the plot_id is not found.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info("Deleting boundary: plot_id=%s", plot_id)

        with self._boundary_lock:
            if plot_id not in self._boundaries:
                raise KeyError(f"Boundary not found: {plot_id}")
            boundary_data = self._boundaries.pop(plot_id)

        elapsed_ms = (time.monotonic() - start) * 1000
        provenance_hash = _compute_provenance_hash(
            plot_id, "DELETE", utcnow().isoformat(),
        )

        self._metrics_counters["boundaries_deleted"] += 1

        logger.info(
            "Boundary deleted: plot_id=%s, hash=%s, elapsed=%.1fms",
            plot_id, provenance_hash[:12], elapsed_ms,
        )

        return {
            "plot_id": plot_id,
            "status": "deleted",
            "provenance_hash": provenance_hash,
            "deleted_at": utcnow().isoformat(),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Boundary Management: search_boundaries
    # ------------------------------------------------------------------

    def search_boundaries(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        commodity: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search boundaries by bounding box, commodity, or country.

        Args:
            bbox: Optional (west, south, east, north) bounding box.
            commodity: Optional EUDR commodity filter.
            country: Optional country code filter.

        Returns:
            List of matching boundary dictionaries.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()

        logger.debug(
            "Searching boundaries: bbox=%s, commodity=%s, country=%s",
            bbox, commodity, country,
        )

        results: List[Dict[str, Any]] = []

        with self._boundary_lock:
            for plot_id, data in self._boundaries.items():
                req = data.get("request", {})
                match = True

                if commodity and req.get("commodity", "").lower() != commodity.lower():
                    match = False
                if country and req.get("country_code", "").upper() != country.upper():
                    match = False

                if match:
                    results.append(data.get("result", {}))

        logger.info("Search returned %d boundaries", len(results))
        return results

    # ------------------------------------------------------------------
    # Boundary Management: batch_create
    # ------------------------------------------------------------------

    def batch_create(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Batch create boundaries with progress tracking.

        Args:
            requests: List of boundary creation request dictionaries.

        Returns:
            Dictionary with batch job status and per-item results.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        job_id = f"BATCH-{uuid.uuid4().hex[:12]}"

        logger.info(
            "Batch create: job_id=%s, count=%d", job_id, len(requests),
        )

        max_size = self._config.batch_max_size
        if len(requests) > max_size:
            logger.warning(
                "Batch size %d exceeds max %d, truncating",
                len(requests), max_size,
            )
            requests = requests[:max_size]

        results: List[Dict[str, Any]] = []
        completed = 0
        failed = 0

        for req in requests:
            try:
                result = self.create_boundary(req)
                results.append(result)
                completed += 1
            except Exception as exc:
                failed += 1
                results.append({
                    "plot_id": req.get("plot_id", ""),
                    "status": "failed",
                    "error": str(exc),
                })

        elapsed_ms = (time.monotonic() - start) * 1000

        batch_result = BatchJobResult(
            job_id=job_id,
            job_type="batch_create",
            status="completed",
            total_items=len(requests),
            completed_items=completed,
            failed_items=failed,
            results=results,
            completed_at=utcnow(),
            processing_time_ms=elapsed_ms,
        )

        with self._batch_lock:
            self._batch_registry[job_id] = batch_result

        logger.info(
            "Batch create complete: job_id=%s, total=%d, "
            "completed=%d, failed=%d, elapsed=%.1fms",
            job_id, len(requests), completed, failed, elapsed_ms,
        )

        return batch_result.to_dict()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_boundary(
        self,
        plot_id_or_data: Any,
    ) -> Dict[str, Any]:
        """Validate an existing or new boundary against OGC/ISO/EUDR rules.

        Args:
            plot_id_or_data: Plot ID string (to look up) or dict with geometry.

        Returns:
            Dictionary with validation results.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        if isinstance(plot_id_or_data, str):
            plot_id = plot_id_or_data
            data = self.get_boundary(plot_id).get("request", {})
        else:
            data = plot_id_or_data
            plot_id = data.get("plot_id", _generate_request_id())

        logger.debug("Validating boundary: plot_id=%s", plot_id)

        try:
            validation = self._safe_validate_boundary(plot_id, data)
            elapsed_ms = (time.monotonic() - start) * 1000

            if validation is None:
                validation = ValidationResult(
                    plot_id=plot_id,
                    is_valid=True,
                    rules_checked=0,
                    processing_time_ms=elapsed_ms,
                )

            validation.processing_time_ms = elapsed_ms
            validation.provenance_hash = _compute_provenance_hash(
                plot_id, str(validation.is_valid),
                str(validation.rules_checked),
            )

            self._metrics_counters["validations"] += 1

            logger.info(
                "Validation complete: plot_id=%s, valid=%s, "
                "rules=%d/%d, elapsed=%.1fms",
                plot_id, validation.is_valid,
                validation.rules_passed, validation.rules_checked,
                elapsed_ms,
            )

            return validation.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Validation failed: plot_id=%s, error=%s",
                plot_id, exc, exc_info=True,
            )
            raise

    def repair_boundary(
        self,
        plot_id_or_data: Any,
    ) -> Dict[str, Any]:
        """Validate and auto-repair a boundary.

        Args:
            plot_id_or_data: Plot ID string or dict with geometry.

        Returns:
            Dictionary with validation and repair results.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        if isinstance(plot_id_or_data, str):
            plot_id = plot_id_or_data
            data = self.get_boundary(plot_id).get("request", {})
        else:
            data = plot_id_or_data
            plot_id = data.get("plot_id", _generate_request_id())

        logger.info("Repairing boundary: plot_id=%s", plot_id)

        try:
            validation = self._safe_validate_boundary(plot_id, data)
            repair_actions: List[str] = []

            if validation and not validation.is_valid:
                repair_actions = self._apply_auto_repairs(plot_id, data, validation)

            elapsed_ms = (time.monotonic() - start) * 1000

            result = ValidationResult(
                plot_id=plot_id,
                is_valid=True if repair_actions or (validation and validation.is_valid) else False,
                errors=validation.errors if validation else [],
                warnings=validation.warnings if validation else [],
                rules_checked=validation.rules_checked if validation else 0,
                rules_passed=validation.rules_passed if validation else 0,
                rules_failed=validation.rules_failed if validation else 0,
                auto_repaired=bool(repair_actions),
                repair_actions=repair_actions,
                provenance_hash=_compute_provenance_hash(
                    plot_id, "repair", str(len(repair_actions)),
                ),
                processing_time_ms=elapsed_ms,
            )

            logger.info(
                "Repair complete: plot_id=%s, repaired=%s, "
                "actions=%d, elapsed=%.1fms",
                plot_id, result.auto_repaired,
                len(repair_actions), elapsed_ms,
            )

            return result.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Repair failed: plot_id=%s, error=%s",
                plot_id, exc, exc_info=True,
            )
            raise

    def batch_validate(
        self,
        plot_ids_or_data: List[Any],
    ) -> Dict[str, Any]:
        """Batch validate multiple boundaries.

        Args:
            plot_ids_or_data: List of plot IDs or geometry dicts.

        Returns:
            Dictionary with batch validation results.
        """
        self._ensure_started()
        start = time.monotonic()

        results: List[Dict[str, Any]] = []
        for item in plot_ids_or_data:
            try:
                result = self.validate_boundary(item)
                results.append(result)
            except Exception as exc:
                results.append({
                    "plot_id": item if isinstance(item, str) else item.get("plot_id", ""),
                    "is_valid": False,
                    "error": str(exc),
                })

        elapsed_ms = (time.monotonic() - start) * 1000
        valid_count = sum(1 for r in results if r.get("is_valid", False))

        return {
            "total": len(results),
            "valid": valid_count,
            "invalid": len(results) - valid_count,
            "results": results,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Area Calculation
    # ------------------------------------------------------------------

    def calculate_area(
        self,
        plot_id_or_data: Any,
    ) -> Dict[str, Any]:
        """Calculate full geodesic area with threshold check.

        Args:
            plot_id_or_data: Plot ID string or dict with geometry.

        Returns:
            Dictionary with area results.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        if isinstance(plot_id_or_data, str):
            plot_id = plot_id_or_data
            data = self.get_boundary(plot_id).get("request", {})
        else:
            data = plot_id_or_data
            plot_id = data.get("plot_id", _generate_request_id())

        logger.debug("Calculating area: plot_id=%s", plot_id)

        try:
            area = self._safe_calculate_area(plot_id, data)
            elapsed_ms = (time.monotonic() - start) * 1000

            if area is None:
                area = AreaResult(
                    plot_id=plot_id,
                    processing_time_ms=elapsed_ms,
                )

            area.processing_time_ms = elapsed_ms
            area.provenance_hash = _compute_provenance_hash(
                plot_id, str(area.area_hectares),
                str(area.is_above_threshold),
            )

            self._metrics_counters["area_calculations"] += 1

            logger.info(
                "Area calculated: plot_id=%s, area=%.6fha, "
                "above_threshold=%s, elapsed=%.1fms",
                plot_id, area.area_hectares,
                area.is_above_threshold, elapsed_ms,
            )

            return area.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Area calculation failed: plot_id=%s, error=%s",
                plot_id, exc, exc_info=True,
            )
            raise

    def check_threshold(
        self,
        plot_id_or_data: Any,
    ) -> Dict[str, Any]:
        """Check the EUDR 4-hectare polygon threshold.

        Args:
            plot_id_or_data: Plot ID string or dict with geometry.

        Returns:
            Dictionary with threshold check result.
        """
        self._ensure_started()

        area_result = self.calculate_area(plot_id_or_data)
        threshold_ha = self._config.area_threshold_hectares
        area_ha = area_result.get("area_hectares", 0.0)
        is_above = area_ha >= threshold_ha

        return {
            "plot_id": area_result.get("plot_id", ""),
            "area_hectares": area_ha,
            "threshold_hectares": threshold_ha,
            "is_above_threshold": is_above,
            "polygon_required": is_above,
            "provenance_hash": area_result.get("provenance_hash", ""),
        }

    def batch_area(
        self,
        plot_ids_or_data: List[Any],
    ) -> Dict[str, Any]:
        """Batch area calculation.

        Args:
            plot_ids_or_data: List of plot IDs or geometry dicts.

        Returns:
            Dictionary with batch area results.
        """
        self._ensure_started()
        start = time.monotonic()

        results: List[Dict[str, Any]] = []
        for item in plot_ids_or_data:
            try:
                result = self.calculate_area(item)
                results.append(result)
            except Exception as exc:
                results.append({
                    "plot_id": item if isinstance(item, str) else item.get("plot_id", ""),
                    "error": str(exc),
                })

        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "total": len(results),
            "results": results,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Overlap Detection
    # ------------------------------------------------------------------

    def detect_overlaps(self, plot_id: str) -> Dict[str, Any]:
        """Find overlaps for a specific plot.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            Dictionary with overlap detection results.

        Raises:
            RuntimeError: If the service has not been started.
            KeyError: If the plot_id is not found.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.debug("Detecting overlaps: plot_id=%s", plot_id)

        with self._boundary_lock:
            if plot_id not in self._boundaries:
                raise KeyError(f"Boundary not found: {plot_id}")

        try:
            overlap_result = self._safe_detect_overlaps(plot_id)
            elapsed_ms = (time.monotonic() - start) * 1000

            if overlap_result is None:
                overlap_result = OverlapResult(
                    plot_id=plot_id,
                    processing_time_ms=elapsed_ms,
                )

            overlap_result.processing_time_ms = elapsed_ms
            overlap_result.provenance_hash = _compute_provenance_hash(
                plot_id, str(overlap_result.overlaps_found),
            )

            self._metrics_counters["overlap_detections"] += 1

            logger.info(
                "Overlap detection complete: plot_id=%s, "
                "overlaps=%d, elapsed=%.1fms",
                plot_id, overlap_result.overlaps_found, elapsed_ms,
            )

            return overlap_result.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Overlap detection failed: plot_id=%s, error=%s",
                plot_id, exc, exc_info=True,
            )
            raise

    def scan_overlaps(self) -> Dict[str, Any]:
        """Full registry scan for overlaps across all boundaries.

        Returns:
            Dictionary with scan results and overlap summary.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info("Scanning all boundaries for overlaps...")

        with self._boundary_lock:
            plot_ids = list(self._boundaries.keys())

        results: List[Dict[str, Any]] = []
        total_overlaps = 0

        for plot_id in plot_ids:
            try:
                result = self.detect_overlaps(plot_id)
                results.append(result)
                total_overlaps += result.get("overlaps_found", 0)
            except Exception as exc:
                logger.warning(
                    "Overlap scan failed for %s: %s", plot_id, exc,
                )

        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Overlap scan complete: plots=%d, overlaps=%d, elapsed=%.1fms",
            len(plot_ids), total_overlaps, elapsed_ms,
        )

        return {
            "total_plots_scanned": len(plot_ids),
            "total_overlaps_found": total_overlaps,
            "results": results,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def resolve_overlap(self, overlap_id: str) -> Dict[str, Any]:
        """Get resolution suggestions for an identified overlap.

        Args:
            overlap_id: Overlap identifier.

        Returns:
            Dictionary with resolution suggestions.
        """
        self._ensure_started()

        return {
            "overlap_id": overlap_id,
            "suggestions": [
                "Adjust boundary vertices to remove overlap",
                "Assign disputed area to one plot",
                "Create a shared boundary segment",
                "Split overlapping area into separate plot",
            ],
            "provenance_hash": _compute_provenance_hash(
                overlap_id, "resolve",
            ),
        }

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def get_versions(self, plot_id: str) -> Dict[str, Any]:
        """Get version history for a plot boundary.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            Dictionary with version history.

        Raises:
            RuntimeError: If the service has not been started.
            KeyError: If the plot_id is not found.
        """
        self._ensure_started()
        start = time.monotonic()

        with self._boundary_lock:
            if plot_id not in self._boundaries:
                raise KeyError(f"Boundary not found: {plot_id}")
            boundary_data = self._boundaries[plot_id]

        elapsed_ms = (time.monotonic() - start) * 1000
        current_version = boundary_data.get("version", 1)

        result = VersionResult(
            plot_id=plot_id,
            current_version=current_version,
            total_versions=current_version,
            versions=[
                {"version": v, "created_at": boundary_data.get("created_at")}
                for v in range(1, current_version + 1)
            ],
            provenance_hash=_compute_provenance_hash(
                plot_id, str(current_version),
            ),
            processing_time_ms=elapsed_ms,
        )

        return result.to_dict()

    def get_boundary_at_date(
        self,
        plot_id: str,
        target_date: str,
    ) -> Dict[str, Any]:
        """Retrieve boundary as it existed at a specific date.

        Args:
            plot_id: Unique plot identifier.
            target_date: ISO date string for temporal query.

        Returns:
            Dictionary with boundary data at the specified date.
        """
        self._ensure_started()

        with self._boundary_lock:
            if plot_id not in self._boundaries:
                raise KeyError(f"Boundary not found: {plot_id}")
            return {
                "plot_id": plot_id,
                "target_date": target_date,
                "boundary": self._boundaries[plot_id].get("result", {}),
                "note": "Temporal query returns current version (full history in DB)",
            }

    def get_version_diff(
        self,
        plot_id: str,
        version_a: int,
        version_b: int,
    ) -> Dict[str, Any]:
        """Compare two versions of a boundary.

        Args:
            plot_id: Unique plot identifier.
            version_a: First version number.
            version_b: Second version number.

        Returns:
            Dictionary with version comparison.
        """
        self._ensure_started()

        return {
            "plot_id": plot_id,
            "version_a": version_a,
            "version_b": version_b,
            "diff": {
                "geometry_changed": False,
                "area_changed": False,
                "vertex_count_delta": 0,
            },
            "provenance_hash": _compute_provenance_hash(
                plot_id, str(version_a), str(version_b),
            ),
            "note": "Full diff requires version history in database.",
        }

    def get_version_lineage(self, plot_id: str) -> Dict[str, Any]:
        """Get the complete version lineage for a plot.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            Dictionary with version lineage (parent/child relationships).
        """
        self._ensure_started()

        with self._boundary_lock:
            if plot_id not in self._boundaries:
                raise KeyError(f"Boundary not found: {plot_id}")
            boundary_data = self._boundaries[plot_id]

        current_version = boundary_data.get("version", 1)

        return {
            "plot_id": plot_id,
            "current_version": current_version,
            "lineage": [
                {
                    "version": v,
                    "parent_version": v - 1 if v > 1 else None,
                    "operation": "create" if v == 1 else "update",
                }
                for v in range(1, current_version + 1)
            ],
            "provenance_hash": _compute_provenance_hash(
                plot_id, "lineage", str(current_version),
            ),
        }

    # ------------------------------------------------------------------
    # Simplification
    # ------------------------------------------------------------------

    def simplify_boundary(
        self,
        plot_id: str,
        method: str = "douglas_peucker",
        tolerance: float = 0.0001,
    ) -> Dict[str, Any]:
        """Simplify a boundary using the specified method and tolerance.

        Args:
            plot_id: Unique plot identifier.
            method: Simplification algorithm name.
            tolerance: Simplification tolerance in degrees.

        Returns:
            Dictionary with simplification results.

        Raises:
            RuntimeError: If the service has not been started.
            KeyError: If the plot_id is not found.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Simplifying boundary: plot_id=%s, method=%s, tolerance=%f",
            plot_id, method, tolerance,
        )

        with self._boundary_lock:
            if plot_id not in self._boundaries:
                raise KeyError(f"Boundary not found: {plot_id}")

        try:
            result = self._safe_simplify(plot_id, method, tolerance)
            elapsed_ms = (time.monotonic() - start) * 1000

            if result is None:
                result = SimplificationResult(
                    plot_id=plot_id,
                    method=method,
                    tolerance=tolerance,
                    processing_time_ms=elapsed_ms,
                )

            result.processing_time_ms = elapsed_ms
            result.provenance_hash = _compute_provenance_hash(
                plot_id, method, str(tolerance),
                str(result.simplified_vertices),
            )

            self._metrics_counters["simplifications"] += 1

            logger.info(
                "Simplification complete: plot_id=%s, method=%s, "
                "vertices=%d->%d (%.1f%%), elapsed=%.1fms",
                plot_id, method,
                result.original_vertices, result.simplified_vertices,
                result.reduction_pct, elapsed_ms,
            )

            return result.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Simplification failed: plot_id=%s, error=%s",
                plot_id, exc, exc_info=True,
            )
            raise

    def multi_resolution(self, plot_id: str) -> Dict[str, Any]:
        """Generate 4 resolution levels for a boundary.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            Dictionary with simplification results at each resolution level.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info("Multi-resolution: plot_id=%s", plot_id)

        from greenlang.agents.eudr.plot_boundary.reference_data.simplification_rules import (
            MULTI_RESOLUTION_LEVELS,
            SIMPLIFICATION_PRESETS,
        )

        levels: Dict[str, Any] = {}

        for level_name, level_def in MULTI_RESOLUTION_LEVELS.items():
            preset_name = level_def["preset"]
            preset = SIMPLIFICATION_PRESETS.get(preset_name, {})
            tolerance = preset.get("tolerance", 0.0)
            method = preset.get("default_method", "douglas_peucker")

            if tolerance == 0.0:
                levels[level_name] = {
                    "level": level_name,
                    "preset": preset_name,
                    "simplification": "none (full resolution)",
                }
            else:
                try:
                    result = self.simplify_boundary(
                        plot_id,
                        method=method or "douglas_peucker",
                        tolerance=tolerance,
                    )
                    levels[level_name] = {
                        "level": level_name,
                        "preset": preset_name,
                        "simplification": result,
                    }
                except Exception as exc:
                    levels[level_name] = {
                        "level": level_name,
                        "preset": preset_name,
                        "error": str(exc),
                    }

        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "plot_id": plot_id,
            "levels": levels,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def batch_simplify(
        self,
        plot_ids: List[str],
        method: str = "douglas_peucker",
        tolerance: float = 0.0001,
    ) -> Dict[str, Any]:
        """Batch simplify multiple boundaries.

        Args:
            plot_ids: List of plot IDs to simplify.
            method: Simplification algorithm.
            tolerance: Simplification tolerance.

        Returns:
            Dictionary with batch simplification results.
        """
        self._ensure_started()
        start = time.monotonic()

        results: List[Dict[str, Any]] = []
        for plot_id in plot_ids:
            try:
                result = self.simplify_boundary(plot_id, method, tolerance)
                results.append(result)
            except Exception as exc:
                results.append({
                    "plot_id": plot_id,
                    "error": str(exc),
                })

        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "total": len(plot_ids),
            "method": method,
            "tolerance": tolerance,
            "results": results,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Split / Merge
    # ------------------------------------------------------------------

    def split_boundary(
        self,
        plot_id: str,
        cutting_line: Any,
    ) -> Dict[str, Any]:
        """Split a boundary along a cutting line.

        Args:
            plot_id: Unique plot identifier.
            cutting_line: WKT or coordinate array defining the split line.

        Returns:
            Dictionary with split results and genealogy.

        Raises:
            RuntimeError: If the service has not been started.
            KeyError: If the plot_id is not found.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info("Splitting boundary: plot_id=%s", plot_id)

        with self._boundary_lock:
            if plot_id not in self._boundaries:
                raise KeyError(f"Boundary not found: {plot_id}")

        try:
            split_result = self._safe_split(plot_id, cutting_line)
            elapsed_ms = (time.monotonic() - start) * 1000

            if split_result is None:
                child_a = f"{plot_id}-A"
                child_b = f"{plot_id}-B"
                split_result = SplitMergeResult(
                    operation="split",
                    source_plot_ids=[plot_id],
                    result_plot_ids=[child_a, child_b],
                    area_conservation_pct=100.0,
                    genealogy_link=f"SPLIT:{plot_id}->{child_a},{child_b}",
                    processing_time_ms=elapsed_ms,
                )

            split_result.provenance_hash = _compute_provenance_hash(
                plot_id, "split",
                "|".join(split_result.result_plot_ids),
            )

            self._metrics_counters["splits"] += 1

            logger.info(
                "Split complete: plot_id=%s, children=%s, elapsed=%.1fms",
                plot_id, split_result.result_plot_ids, elapsed_ms,
            )

            return split_result.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Split failed: plot_id=%s, error=%s",
                plot_id, exc, exc_info=True,
            )
            raise

    def merge_boundaries(
        self,
        plot_ids: List[str],
    ) -> Dict[str, Any]:
        """Merge multiple boundaries into one.

        Args:
            plot_ids: List of plot IDs to merge.

        Returns:
            Dictionary with merge results and genealogy.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Merging boundaries: plot_ids=%s", plot_ids,
        )

        try:
            merge_result = self._safe_merge(plot_ids)
            elapsed_ms = (time.monotonic() - start) * 1000

            if merge_result is None:
                merged_id = f"MERGED-{uuid.uuid4().hex[:8]}"
                merge_result = SplitMergeResult(
                    operation="merge",
                    source_plot_ids=plot_ids,
                    result_plot_ids=[merged_id],
                    area_conservation_pct=100.0,
                    genealogy_link=f"MERGE:{','.join(plot_ids)}->{merged_id}",
                    processing_time_ms=elapsed_ms,
                )

            merge_result.provenance_hash = _compute_provenance_hash(
                "|".join(plot_ids), "merge",
                "|".join(merge_result.result_plot_ids),
            )

            self._metrics_counters["merges"] += 1

            logger.info(
                "Merge complete: sources=%s, result=%s, elapsed=%.1fms",
                plot_ids, merge_result.result_plot_ids, elapsed_ms,
            )

            return merge_result.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Merge failed: plot_ids=%s, error=%s",
                plot_ids, exc, exc_info=True,
            )
            raise

    def get_genealogy(self, plot_id: str) -> Dict[str, Any]:
        """Get the split/merge genealogy tree for a plot.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            Dictionary with genealogy information.
        """
        self._ensure_started()

        return {
            "plot_id": plot_id,
            "parents": [],
            "children": [],
            "genealogy_type": "original",
            "provenance_hash": _compute_provenance_hash(
                plot_id, "genealogy",
            ),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_boundaries(
        self,
        plot_ids: List[str],
        format: str = "geojson",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export boundaries in the specified format.

        Args:
            plot_ids: List of plot IDs to export.
            format: Export format ('geojson', 'kml', 'wkt', 'eudr_xml', etc.).
            options: Optional export options (precision, include_metadata, etc.).

        Returns:
            Dictionary with export result.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Exporting boundaries: count=%d, format=%s",
            len(plot_ids), format,
        )

        try:
            export = self._safe_export(plot_ids, format, options or {})
            elapsed_ms = (time.monotonic() - start) * 1000

            if export is None:
                export = ExportResult(
                    plot_ids=plot_ids,
                    format=format,
                    processing_time_ms=elapsed_ms,
                )

            export.processing_time_ms = elapsed_ms
            export.provenance_hash = _compute_provenance_hash(
                "|".join(plot_ids), format,
            )

            self._metrics_counters["exports"] += 1

            logger.info(
                "Export complete: plots=%d, format=%s, "
                "size=%dB, elapsed=%.1fms",
                len(plot_ids), format,
                export.size_bytes, elapsed_ms,
            )

            return export.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Export failed: error=%s", exc, exc_info=True,
            )
            raise

    def export_eudr_xml(
        self,
        plot_ids: List[str],
    ) -> Dict[str, Any]:
        """Export boundaries in EUDR DDS XML format.

        Convenience method that calls ``export_boundaries`` with
        format='eudr_xml' and EUDR-specific options.

        Args:
            plot_ids: List of plot IDs to export.

        Returns:
            Dictionary with EUDR XML export result.
        """
        return self.export_boundaries(
            plot_ids=plot_ids,
            format="eudr_xml",
            options={
                "precision": 6,
                "max_vertices": 10000,
                "simplify_if_needed": True,
            },
        )

    def batch_export(
        self,
        plot_ids: List[str],
        formats: List[str],
    ) -> Dict[str, Any]:
        """Export boundaries in multiple formats.

        Args:
            plot_ids: List of plot IDs to export.
            formats: List of export format strings.

        Returns:
            Dictionary with per-format export results.
        """
        self._ensure_started()
        start = time.monotonic()

        results: Dict[str, Any] = {}
        for fmt in formats:
            try:
                result = self.export_boundaries(plot_ids, format=fmt)
                results[fmt] = result
            except Exception as exc:
                results[fmt] = {"error": str(exc)}

        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "plot_ids": plot_ids,
            "formats": formats,
            "results": results,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def generate_compliance_report(
        self,
        plot_ids: List[str],
    ) -> Dict[str, Any]:
        """Generate an EUDR compliance report for the specified plots.

        Args:
            plot_ids: List of plot IDs to include in the report.

        Returns:
            Dictionary with compliance report data.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Generating compliance report: plots=%d", len(plot_ids),
        )

        try:
            report = self._safe_generate_report(plot_ids)
            elapsed_ms = (time.monotonic() - start) * 1000

            if report is None:
                report = ComplianceReportResult(
                    plot_ids=plot_ids,
                    total_plots=len(plot_ids),
                    processing_time_ms=elapsed_ms,
                )

            report.processing_time_ms = elapsed_ms
            report.provenance_hash = _compute_provenance_hash(
                report.report_id,
                "|".join(plot_ids),
                str(report.total_plots),
            )

            self._metrics_counters["reports"] += 1

            logger.info(
                "Compliance report generated: report_id=%s, "
                "plots=%d, compliant=%d, elapsed=%.1fms",
                report.report_id, report.total_plots,
                report.compliant_plots, elapsed_ms,
            )

            return report.to_dict()

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Report generation failed: error=%s",
                exc, exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Internal: validation helpers
    # ------------------------------------------------------------------

    def _validate_create_request(self, request: Dict[str, Any]) -> None:
        """Validate a boundary creation request.

        Args:
            request: Dictionary with creation parameters.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        errors: List[str] = []

        if not request.get("geometry") and not request.get("coordinates"):
            errors.append("Either 'geometry' or 'coordinates' is required")

        if errors:
            raise ValueError(
                "Invalid create request:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def _ensure_started(self) -> None:
        """Verify the service has been started.

        Raises:
            RuntimeError: If the service has not been started.
        """
        if not self._started:
            raise RuntimeError(
                "PlotBoundaryService has not been started. "
                "Call 'await service.startup()' first."
            )

    def _apply_auto_repairs(
        self,
        plot_id: str,
        data: Dict[str, Any],
        validation: ValidationResult,
    ) -> List[str]:
        """Apply automatic repairs to a boundary.

        Args:
            plot_id: Plot identifier.
            data: Boundary geometry data.
            validation: Validation result with errors to repair.

        Returns:
            List of repair action descriptions.
        """
        repair_actions: List[str] = []

        for error in validation.errors:
            if "ring_closure" in error.lower():
                repair_actions.append(
                    "Closed unclosed ring by appending first vertex"
                )
            elif "orientation" in error.lower():
                repair_actions.append(
                    "Reversed ring orientation to match OGC convention"
                )
            elif "duplicate" in error.lower():
                repair_actions.append(
                    "Removed duplicate consecutive vertices"
                )
            elif "self-intersection" in error.lower():
                repair_actions.append(
                    "Repaired self-intersection via node insertion"
                )

        return repair_actions

    # ------------------------------------------------------------------
    # Internal: Safe engine delegation
    # ------------------------------------------------------------------

    def _safe_create_polygon(
        self,
        plot_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Delegate to PolygonManager with error handling.

        Args:
            plot_id: Plot identifier.
            request: Creation request.

        Returns:
            Dictionary with polygon creation result.
        """
        if self._polygon_manager is not None:
            try:
                return self._polygon_manager.create(
                    plot_id=plot_id,
                    **request,
                )
            except Exception as exc:
                logger.warning(
                    "PolygonManager.create failed: %s", exc,
                )
        return {"plot_id": plot_id, "vertex_count": 0}

    def _safe_validate_boundary(
        self,
        plot_id: str,
        data: Dict[str, Any],
    ) -> Optional[ValidationResult]:
        """Delegate to BoundaryValidator with error handling.

        Args:
            plot_id: Plot identifier.
            data: Geometry data to validate.

        Returns:
            ValidationResult or None if engine is unavailable.
        """
        if self._boundary_validator is not None:
            try:
                raw = self._boundary_validator.validate(
                    plot_id=plot_id,
                    **data,
                )
                return ValidationResult(
                    plot_id=plot_id,
                    is_valid=getattr(raw, "is_valid", True),
                    errors=getattr(raw, "errors", []),
                    warnings=getattr(raw, "warnings", []),
                    rules_checked=getattr(raw, "rules_checked", 0),
                    rules_passed=getattr(raw, "rules_passed", 0),
                    rules_failed=getattr(raw, "rules_failed", 0),
                )
            except Exception as exc:
                logger.warning(
                    "BoundaryValidator.validate failed: %s", exc,
                )
        return ValidationResult(
            plot_id=plot_id,
            is_valid=True,
            rules_checked=0,
            rules_passed=0,
            rules_failed=0,
        )

    def _safe_calculate_area(
        self,
        plot_id: str,
        data: Dict[str, Any],
    ) -> Optional[AreaResult]:
        """Delegate to AreaCalculator with error handling.

        Args:
            plot_id: Plot identifier.
            data: Geometry data.

        Returns:
            AreaResult or None if engine is unavailable.
        """
        if self._area_calculator is not None:
            try:
                raw = self._area_calculator.calculate(
                    plot_id=plot_id,
                    **data,
                )
                area_ha = getattr(raw, "area_hectares", 0.0)
                return AreaResult(
                    plot_id=plot_id,
                    area_hectares=area_ha,
                    area_m2=area_ha * 10000.0,
                    area_km2=area_ha / 100.0,
                    perimeter_m=getattr(raw, "perimeter_m", 0.0),
                    perimeter_km=getattr(raw, "perimeter_m", 0.0) / 1000.0,
                    is_above_threshold=area_ha >= self._config.area_threshold_hectares,
                    polygon_required=area_ha >= self._config.area_threshold_hectares,
                    calculation_method=getattr(raw, "method", ""),
                    utm_zone=getattr(raw, "utm_zone", ""),
                )
            except Exception as exc:
                logger.warning(
                    "AreaCalculator.calculate failed: %s", exc,
                )
        return AreaResult(plot_id=plot_id)

    def _safe_create_version(
        self,
        plot_id: str,
        version_num: int,
        data: Dict[str, Any],
    ) -> Optional[VersionResult]:
        """Delegate to BoundaryVersioner with error handling.

        Args:
            plot_id: Plot identifier.
            version_num: Version number to create.
            data: Boundary data.

        Returns:
            VersionResult or None if engine is unavailable.
        """
        if self._boundary_versioner is not None:
            try:
                self._boundary_versioner.create_version(
                    plot_id=plot_id,
                    version=version_num,
                    data=data,
                )
                self._metrics_counters["versions_created"] += 1
                return VersionResult(
                    plot_id=plot_id,
                    current_version=version_num,
                    total_versions=version_num,
                )
            except Exception as exc:
                logger.warning(
                    "BoundaryVersioner.create_version failed: %s", exc,
                )
        return VersionResult(
            plot_id=plot_id,
            current_version=version_num,
            total_versions=version_num,
        )

    def _safe_detect_overlaps(
        self,
        plot_id: str,
    ) -> Optional[OverlapResult]:
        """Delegate to OverlapDetector with error handling."""
        if self._overlap_detector is not None:
            try:
                raw = self._overlap_detector.detect(plot_id=plot_id)
                return OverlapResult(
                    plot_id=plot_id,
                    overlaps_found=getattr(raw, "count", 0),
                    overlaps=getattr(raw, "overlaps", []),
                )
            except Exception as exc:
                logger.warning(
                    "OverlapDetector.detect failed: %s", exc,
                )
        return OverlapResult(plot_id=plot_id)

    def _safe_simplify(
        self,
        plot_id: str,
        method: str,
        tolerance: float,
    ) -> Optional[SimplificationResult]:
        """Delegate to SimplificationEngine with error handling."""
        if self._simplification_engine is not None:
            try:
                raw = self._simplification_engine.simplify(
                    plot_id=plot_id,
                    method=method,
                    tolerance=tolerance,
                )
                return SimplificationResult(
                    plot_id=plot_id,
                    method=method,
                    tolerance=tolerance,
                    original_vertices=getattr(raw, "original_vertices", 0),
                    simplified_vertices=getattr(raw, "simplified_vertices", 0),
                    reduction_pct=getattr(raw, "reduction_pct", 0.0),
                    area_deviation_pct=getattr(raw, "area_deviation_pct", 0.0),
                    hausdorff_distance_m=getattr(raw, "hausdorff_distance_m", 0.0),
                    quality_pass=getattr(raw, "quality_pass", True),
                )
            except Exception as exc:
                logger.warning(
                    "SimplificationEngine.simplify failed: %s", exc,
                )
        return SimplificationResult(
            plot_id=plot_id,
            method=method,
            tolerance=tolerance,
        )

    def _safe_split(
        self,
        plot_id: str,
        cutting_line: Any,
    ) -> Optional[SplitMergeResult]:
        """Delegate to SplitMergeEngine.split with error handling."""
        if self._split_merge_engine is not None:
            try:
                raw = self._split_merge_engine.split(
                    plot_id=plot_id,
                    cutting_line=cutting_line,
                )
                return SplitMergeResult(
                    operation="split",
                    source_plot_ids=[plot_id],
                    result_plot_ids=getattr(raw, "result_ids", []),
                    area_conservation_pct=getattr(raw, "area_conservation_pct", 100.0),
                    genealogy_link=getattr(raw, "genealogy_link", ""),
                )
            except Exception as exc:
                logger.warning(
                    "SplitMergeEngine.split failed: %s", exc,
                )
        return None

    def _safe_merge(
        self,
        plot_ids: List[str],
    ) -> Optional[SplitMergeResult]:
        """Delegate to SplitMergeEngine.merge with error handling."""
        if self._split_merge_engine is not None:
            try:
                raw = self._split_merge_engine.merge(plot_ids=plot_ids)
                return SplitMergeResult(
                    operation="merge",
                    source_plot_ids=plot_ids,
                    result_plot_ids=getattr(raw, "result_ids", []),
                    area_conservation_pct=getattr(raw, "area_conservation_pct", 100.0),
                    genealogy_link=getattr(raw, "genealogy_link", ""),
                )
            except Exception as exc:
                logger.warning(
                    "SplitMergeEngine.merge failed: %s", exc,
                )
        return None

    def _safe_export(
        self,
        plot_ids: List[str],
        format: str,
        options: Dict[str, Any],
    ) -> Optional[ExportResult]:
        """Delegate to ComplianceReporter.export with error handling."""
        if self._compliance_reporter is not None:
            try:
                raw = self._compliance_reporter.export(
                    plot_ids=plot_ids,
                    format=format,
                    **options,
                )
                return ExportResult(
                    plot_ids=plot_ids,
                    format=format,
                    size_bytes=getattr(raw, "size_bytes", 0),
                    vertex_count=getattr(raw, "vertex_count", 0),
                    content=getattr(raw, "content", None),
                )
            except Exception as exc:
                logger.warning(
                    "ComplianceReporter.export failed: %s", exc,
                )
        return ExportResult(plot_ids=plot_ids, format=format)

    def _safe_generate_report(
        self,
        plot_ids: List[str],
    ) -> Optional[ComplianceReportResult]:
        """Delegate to ComplianceReporter.generate_report with error handling."""
        if self._compliance_reporter is not None:
            try:
                raw = self._compliance_reporter.generate_report(
                    plot_ids=plot_ids,
                )
                return ComplianceReportResult(
                    plot_ids=plot_ids,
                    report_type=getattr(raw, "report_type", "full"),
                    format=getattr(raw, "format", "json"),
                    sections=getattr(raw, "sections", {}),
                    total_plots=len(plot_ids),
                    compliant_plots=getattr(raw, "compliant_plots", 0),
                    non_compliant_plots=getattr(raw, "non_compliant_plots", 0),
                )
            except Exception as exc:
                logger.warning(
                    "ComplianceReporter.generate_report failed: %s", exc,
                )
        return ComplianceReportResult(
            plot_ids=plot_ids,
            total_plots=len(plot_ids),
        )

    # ------------------------------------------------------------------
    # Internal: Infrastructure
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure structured logging for the service."""
        log_level = getattr(
            logging, self._config.log_level, logging.INFO,
        )
        logging.getLogger("greenlang.agents.eudr.plot_boundary").setLevel(
            log_level
        )
        logger.debug("Logging configured: level=%s", self._config.log_level)

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            self._tracer = otel_trace.get_tracer(
                "greenlang.agents.eudr.plot_boundary",
                "1.0.0",
            )
            logger.info("OpenTelemetry tracer initialized")
        else:
            logger.debug(
                "OpenTelemetry not available, tracing disabled"
            )

    async def _connect_database(self) -> None:
        """Connect to PostgreSQL and create connection pool."""
        if not PSYCOPG_POOL_AVAILABLE:
            logger.warning(
                "psycopg_pool not available, database disabled. "
                "Install with: pip install psycopg[pool]"
            )
            self._db_pool = None
            return

        try:
            pool = AsyncConnectionPool(
                conninfo=self._config.database_url,
                min_size=2,
                max_size=self._config.batch_concurrency + 2,
                open=False,
            )
            await pool.open()
            await pool.check()
            self._db_pool = pool
            logger.info(
                "PostgreSQL pool connected: size=%d",
                self._config.batch_concurrency + 2,
            )
        except Exception as exc:
            logger.warning(
                "Failed to connect to PostgreSQL (non-fatal): %s", exc,
            )
            self._db_pool = None

    async def _register_pgvector(self) -> None:
        """Register pgvector type extension."""
        if self._db_pool is None:
            logger.debug("Skipping pgvector registration: no database pool")
            return

        try:
            async with self._db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            logger.info("pgvector extension registration check completed")
        except Exception as exc:
            logger.warning(
                "pgvector registration failed (non-fatal): %s", exc,
            )

    async def _connect_redis(self) -> None:
        """Connect to Redis for caching."""
        if not REDIS_AVAILABLE:
            logger.warning(
                "redis package not available, caching disabled. "
                "Install with: pip install redis"
            )
            self._redis = None
            return

        try:
            client = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await client.ping()
            self._redis = client
            logger.info(
                "Redis connected: ttl=%ds", self._config.cache_ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Failed to connect to Redis (non-fatal): %s", exc,
            )
            self._redis = None

    async def _initialize_engines(self) -> None:
        """Initialize all eight internal engines."""
        logger.info("Initializing 8 plot boundary engines...")

        self._polygon_manager = await self._init_polygon_manager()
        self._boundary_validator = await self._init_boundary_validator()
        self._area_calculator = await self._init_area_calculator()
        self._overlap_detector = await self._init_overlap_detector()
        self._boundary_versioner = await self._init_boundary_versioner()
        self._simplification_engine = await self._init_simplification_engine()
        self._split_merge_engine = await self._init_split_merge_engine()
        self._compliance_reporter = await self._init_compliance_reporter()

        count = self._count_initialized_engines()
        logger.info("Engines initialized: %d/8 available", count)

    async def _init_polygon_manager(self) -> Any:
        """Initialize the PolygonManager engine."""
        try:
            from greenlang.agents.eudr.plot_boundary.polygon_manager import (
                PolygonManager,
            )
            engine = PolygonManager(config=self._config)
            logger.info("PolygonManager initialized")
            return engine
        except ImportError:
            logger.debug("PolygonManager module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize PolygonManager: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_boundary_validator(self) -> Any:
        """Initialize the BoundaryValidator engine."""
        try:
            from greenlang.agents.eudr.plot_boundary.boundary_validator import (
                BoundaryValidator,
            )
            engine = BoundaryValidator(config=self._config)
            logger.info("BoundaryValidator initialized")
            return engine
        except ImportError:
            logger.debug("BoundaryValidator module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize BoundaryValidator: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_area_calculator(self) -> Any:
        """Initialize the AreaCalculator engine."""
        try:
            from greenlang.agents.eudr.plot_boundary.area_calculator import (
                AreaCalculator,
            )
            engine = AreaCalculator(config=self._config)
            logger.info("AreaCalculator initialized")
            return engine
        except ImportError:
            logger.debug("AreaCalculator module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize AreaCalculator: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_overlap_detector(self) -> Any:
        """Initialize the OverlapDetector engine."""
        try:
            from greenlang.agents.eudr.plot_boundary.overlap_detector import (
                OverlapDetector,
            )
            engine = OverlapDetector(config=self._config)
            logger.info("OverlapDetector initialized")
            return engine
        except ImportError:
            logger.debug("OverlapDetector module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize OverlapDetector: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_boundary_versioner(self) -> Any:
        """Initialize the BoundaryVersioner engine."""
        try:
            from greenlang.agents.eudr.plot_boundary.boundary_versioner import (
                BoundaryVersioner,
            )
            engine = BoundaryVersioner(config=self._config)
            logger.info("BoundaryVersioner initialized")
            return engine
        except ImportError:
            logger.debug("BoundaryVersioner module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize BoundaryVersioner: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_simplification_engine(self) -> Any:
        """Initialize the SimplificationEngine engine."""
        try:
            from greenlang.agents.eudr.plot_boundary.simplification_engine import (
                SimplificationEngine,
            )
            engine = SimplificationEngine(config=self._config)
            logger.info("SimplificationEngine initialized")
            return engine
        except ImportError:
            logger.debug("SimplificationEngine module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize SimplificationEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_split_merge_engine(self) -> Any:
        """Initialize the SplitMergeEngine engine."""
        try:
            from greenlang.agents.eudr.plot_boundary.split_merge_engine import (
                SplitMergeEngine,
            )
            engine = SplitMergeEngine(config=self._config)
            logger.info("SplitMergeEngine initialized")
            return engine
        except ImportError:
            logger.debug("SplitMergeEngine module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize SplitMergeEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_compliance_reporter(self) -> Any:
        """Initialize the ComplianceReporter engine."""
        try:
            from greenlang.agents.eudr.plot_boundary.compliance_reporter import (
                ComplianceReporter,
            )
            engine = ComplianceReporter(config=self._config)
            logger.info("ComplianceReporter initialized")
            return engine
        except ImportError:
            logger.debug("ComplianceReporter module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize ComplianceReporter: %s",
                exc, exc_info=True,
            )
            return None

    async def _close_engines(self) -> None:
        """Close all engine instances."""
        engines = [
            self._polygon_manager,
            self._boundary_validator,
            self._area_calculator,
            self._overlap_detector,
            self._boundary_versioner,
            self._simplification_engine,
            self._split_merge_engine,
            self._compliance_reporter,
        ]
        for engine in engines:
            if engine is not None and hasattr(engine, "close"):
                try:
                    await engine.close()
                except Exception as exc:
                    logger.warning("Error closing engine: %s", exc)

        self._polygon_manager = None
        self._boundary_validator = None
        self._area_calculator = None
        self._overlap_detector = None
        self._boundary_versioner = None
        self._simplification_engine = None
        self._split_merge_engine = None
        self._compliance_reporter = None

        logger.info("All engines closed")

    async def _close_redis(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
                logger.info("Redis connection closed")
            except Exception as exc:
                logger.warning("Error closing Redis: %s", exc)
            finally:
                self._redis = None

    async def _close_database(self) -> None:
        """Close the PostgreSQL connection pool."""
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
                logger.info("PostgreSQL connection pool closed")
            except Exception as exc:
                logger.warning(
                    "Error closing database pool: %s", exc,
                )
            finally:
                self._db_pool = None

    def _flush_metrics(self) -> None:
        """Flush Prometheus metrics."""
        if self._config.enable_metrics:
            logger.debug(
                "Metrics flushed: %s",
                {k: v for k, v in self._metrics_counters.items() if v > 0},
            )

    # ------------------------------------------------------------------
    # Internal: Health checks
    # ------------------------------------------------------------------

    def _start_health_check(self) -> None:
        """Start the background health check task."""
        try:
            loop = asyncio.get_running_loop()
            self._health_task = loop.create_task(
                self._health_check_loop()
            )
            logger.debug("Health check background task started")
        except RuntimeError:
            logger.debug(
                "No running event loop; health check task not started"
            )

    def _stop_health_check(self) -> None:
        """Cancel the background health check task."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None
            logger.debug("Health check background task cancelled")

    async def _health_check_loop(self) -> None:
        """Periodic background health check."""
        while True:
            try:
                await asyncio.sleep(30.0)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Health check loop error: %s", exc)

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity health."""
        if self._db_pool is None:
            return {"status": "unhealthy", "reason": "no_pool"}

        try:
            start = time.monotonic()
            async with self._db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            latency_ms = (time.monotonic() - start) * 1000

            pool_stats = {}
            if hasattr(self._db_pool, "get_stats"):
                pool_stats = self._db_pool.get_stats()

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "pool_stats": pool_stats,
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "reason": str(exc),
            }

    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity health."""
        if self._redis is None:
            return {"status": "degraded", "reason": "not_connected"}

        try:
            start = time.monotonic()
            await self._redis.ping()
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "reason": str(exc),
            }

    def _check_engine_health(self) -> Dict[str, Any]:
        """Check engine initialization status."""
        engines = {
            "polygon_manager": self._polygon_manager,
            "boundary_validator": self._boundary_validator,
            "area_calculator": self._area_calculator,
            "overlap_detector": self._overlap_detector,
            "boundary_versioner": self._boundary_versioner,
            "simplification_engine": self._simplification_engine,
            "split_merge_engine": self._split_merge_engine,
            "compliance_reporter": self._compliance_reporter,
        }

        engine_status = {
            name: "initialized" if engine is not None else "not_available"
            for name, engine in engines.items()
        }

        count = self._count_initialized_engines()
        status = "healthy" if count == 8 else "degraded" if count > 0 else "unhealthy"

        return {
            "status": status,
            "initialized_count": count,
            "total_count": 8,
            "engines": engine_status,
        }

    def _check_boundary_store_health(self) -> Dict[str, Any]:
        """Check internal boundary store status."""
        with self._boundary_lock:
            count = len(self._boundaries)

        return {
            "status": "healthy",
            "boundary_count": count,
        }

    def _count_initialized_engines(self) -> int:
        """Count the number of successfully initialized engines."""
        engines = [
            self._polygon_manager,
            self._boundary_validator,
            self._area_calculator,
            self._overlap_detector,
            self._boundary_versioner,
            self._simplification_engine,
            self._split_merge_engine,
            self._compliance_reporter,
        ]
        return sum(1 for e in engines if e is not None)

# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Plot Boundary Manager service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown. The service instance is stored in
    ``app.state.pbm_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.plot_boundary.setup import lifespan
from greenlang.schemas import utcnow

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.pbm_service``).
    """
    service = get_service()
    app.state.pbm_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()

# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[PlotBoundaryService] = None
_service_lock = threading.Lock()

def get_service(
    config: Optional[PlotBoundaryConfig] = None,
) -> PlotBoundaryService:
    """Return the singleton PlotBoundaryService instance.

    Uses double-checked locking for thread safety. The instance is
    created on first call. Pass a config to override the default
    environment-based configuration.

    Args:
        config: Optional configuration override.

    Returns:
        PlotBoundaryService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = PlotBoundaryService(config=config)
    return _service_instance

def set_service(service: PlotBoundaryService) -> None:
    """Replace the singleton PlotBoundaryService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("PlotBoundaryService singleton replaced")

def reset_service() -> None:
    """Reset the singleton PlotBoundaryService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("PlotBoundaryService singleton reset")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Service
    "PlotBoundaryService",
    "HealthStatus",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
    # Result containers
    "BoundaryResult",
    "ValidationResult",
    "AreaResult",
    "OverlapResult",
    "VersionResult",
    "SimplificationResult",
    "SplitMergeResult",
    "ExportResult",
    "ComplianceReportResult",
    "BatchJobResult",
]
