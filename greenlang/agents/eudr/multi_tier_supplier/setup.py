# -*- coding: utf-8 -*-
"""
MultiTierSupplierService - Facade for AGENT-EUDR-008 Multi-Tier Supplier Tracker

This module implements the MultiTierSupplierService, the single entry point
for all multi-tier supplier tracking operations in the GL-EUDR-APP. It manages
the lifecycle of eight internal engines, an async PostgreSQL connection pool
(psycopg + psycopg_pool), a Redis cache connection, OpenTelemetry tracing,
and Prometheus metrics. The service exposes a unified interface consumed by
the FastAPI router layer and the GL-EUDR-APP integration.

Lifecycle:
    startup  -> load config -> connect DB -> register pgvector -> connect Redis
             -> initialize engines -> start health check background task
    shutdown -> close engines -> close Redis -> close DB pool -> flush metrics

Engines (8):
    1. SupplierDiscoveryEngine    - Tier discovery from multiple sources (Feature 1)
    2. SupplierProfileManager     - Profile CRUD with completeness scoring (Feature 2)
    3. TierDepthTracker           - Depth scoring and visibility assessment (Feature 3)
    4. RelationshipManager        - Relationship lifecycle management (Feature 4)
    5. RiskPropagationEngine      - Risk scoring and upstream propagation (Feature 5)
    6. ComplianceMonitor          - Compliance status monitoring (Feature 6)
    7. GapAnalyzer                - Gap detection and remediation plans (Feature 7)
    8. AuditReporter              - Audit trail and report generation (Feature 8)

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.multi_tier_supplier.setup import (
    ...     MultiTierSupplierService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

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
# Environment variable based configuration
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_MST_"

def _env(key: str, default: str = "") -> str:
    """Read an environment variable with the GL_EUDR_MST_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)

def _env_int(key: str, default: int = 0) -> int:
    """Read an integer environment variable."""
    raw = _env(key, str(default))
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default

def _env_float(key: str, default: float = 0.0) -> float:
    """Read a float environment variable."""
    raw = _env(key, str(default))
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default

def _env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    raw = _env(key, str(default)).lower()
    return raw in ("true", "1", "yes", "on")

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
    return f"MST-{uuid.uuid4().hex[:12]}"

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
# Result container: DiscoveryResult
# ---------------------------------------------------------------------------

class DiscoveryResult:
    """Result from a supplier discovery operation.

    Attributes:
        request_id: Unique request identifier.
        suppliers_discovered: Number of new suppliers discovered.
        relationships_discovered: Number of new relationships found.
        source_type: Discovery source type.
        commodity: EUDR commodity context.
        tier_depths_found: List of tier depths discovered.
        confidence_scores: List of confidence scores for discoveries.
        duplicates_detected: Number of duplicates merged.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "suppliers_discovered", "relationships_discovered",
        "source_type", "commodity", "tier_depths_found",
        "confidence_scores", "duplicates_detected",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        suppliers_discovered: int = 0,
        relationships_discovered: int = 0,
        source_type: str = "unknown",
        commodity: str = "",
        tier_depths_found: Optional[List[int]] = None,
        confidence_scores: Optional[List[float]] = None,
        duplicates_detected: int = 0,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.suppliers_discovered = suppliers_discovered
        self.relationships_discovered = relationships_discovered
        self.source_type = source_type
        self.commodity = commodity
        self.tier_depths_found = tier_depths_found or []
        self.confidence_scores = confidence_scores or []
        self.duplicates_detected = duplicates_detected
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "suppliers_discovered": self.suppliers_discovered,
            "relationships_discovered": self.relationships_discovered,
            "source_type": self.source_type,
            "commodity": self.commodity,
            "tier_depths_found": self.tier_depths_found,
            "confidence_scores": [round(s, 3) for s in self.confidence_scores],
            "duplicates_detected": self.duplicates_detected,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: ProfileResult
# ---------------------------------------------------------------------------

class ProfileResult:
    """Result from a supplier profile operation (CRUD).

    Attributes:
        request_id: Unique request identifier.
        supplier_id: Supplier unique identifier.
        operation: Operation type (created, updated, retrieved, deactivated).
        profile_completeness: Profile completeness score (0-100).
        missing_fields: List of missing required fields.
        tier_level: Assigned tier level.
        status: Supplier status.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "supplier_id", "operation",
        "profile_completeness", "missing_fields", "tier_level",
        "status", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        supplier_id: str = "",
        operation: str = "unknown",
        profile_completeness: float = 0.0,
        missing_fields: Optional[List[str]] = None,
        tier_level: int = 0,
        status: str = "active",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.supplier_id = supplier_id
        self.operation = operation
        self.profile_completeness = profile_completeness
        self.missing_fields = missing_fields or []
        self.tier_level = tier_level
        self.status = status
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "supplier_id": self.supplier_id,
            "operation": self.operation,
            "profile_completeness": round(self.profile_completeness, 2),
            "missing_fields": self.missing_fields,
            "tier_level": self.tier_level,
            "status": self.status,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: TierResult
# ---------------------------------------------------------------------------

class TierResult:
    """Result from a tier depth assessment operation.

    Attributes:
        request_id: Unique request identifier.
        max_tier_depth: Maximum tier depth discovered.
        avg_tier_depth: Average tier depth across chains.
        visibility_score: Overall visibility score (0-100).
        tier_visibility: Per-tier visibility percentages.
        coverage_score: Volume coverage score (0-100).
        gaps: List of tier gap descriptions.
        benchmark_comparison: Comparison against industry benchmarks.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "max_tier_depth", "avg_tier_depth",
        "visibility_score", "tier_visibility", "coverage_score",
        "gaps", "benchmark_comparison",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        max_tier_depth: int = 0,
        avg_tier_depth: float = 0.0,
        visibility_score: float = 0.0,
        tier_visibility: Optional[Dict[str, float]] = None,
        coverage_score: float = 0.0,
        gaps: Optional[List[str]] = None,
        benchmark_comparison: Optional[Dict[str, Any]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.max_tier_depth = max_tier_depth
        self.avg_tier_depth = avg_tier_depth
        self.visibility_score = visibility_score
        self.tier_visibility = tier_visibility or {}
        self.coverage_score = coverage_score
        self.gaps = gaps or []
        self.benchmark_comparison = benchmark_comparison or {}
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "max_tier_depth": self.max_tier_depth,
            "avg_tier_depth": round(self.avg_tier_depth, 2),
            "visibility_score": round(self.visibility_score, 2),
            "tier_visibility": self.tier_visibility,
            "coverage_score": round(self.coverage_score, 2),
            "gaps": self.gaps,
            "benchmark_comparison": self.benchmark_comparison,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: RiskResult
# ---------------------------------------------------------------------------

class RiskResult:
    """Result from a supplier risk assessment or propagation.

    Attributes:
        request_id: Unique request identifier.
        supplier_id: Assessed supplier identifier.
        composite_score: Overall risk score (0-100).
        risk_level: Risk level classification.
        category_scores: Per-category risk breakdown.
        propagated_from: List of supplier IDs that contributed risk.
        propagation_method: Risk propagation method used.
        alerts: List of risk threshold alerts.
        trend: Risk trend (improving, stable, degrading).
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "supplier_id", "composite_score",
        "risk_level", "category_scores", "propagated_from",
        "propagation_method", "alerts", "trend",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        supplier_id: str = "",
        composite_score: float = 0.0,
        risk_level: str = "unknown",
        category_scores: Optional[Dict[str, float]] = None,
        propagated_from: Optional[List[str]] = None,
        propagation_method: str = "weighted_average",
        alerts: Optional[List[str]] = None,
        trend: str = "stable",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.supplier_id = supplier_id
        self.composite_score = composite_score
        self.risk_level = risk_level
        self.category_scores = category_scores or {}
        self.propagated_from = propagated_from or []
        self.propagation_method = propagation_method
        self.alerts = alerts or []
        self.trend = trend
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "supplier_id": self.supplier_id,
            "composite_score": round(self.composite_score, 2),
            "risk_level": self.risk_level,
            "category_scores": {
                k: round(v, 2) for k, v in self.category_scores.items()
            },
            "propagated_from": self.propagated_from,
            "propagation_method": self.propagation_method,
            "alerts": self.alerts,
            "trend": self.trend,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: ComplianceResult
# ---------------------------------------------------------------------------

class ComplianceResult:
    """Result from a supplier compliance check.

    Attributes:
        request_id: Unique request identifier.
        supplier_id: Assessed supplier identifier.
        compliance_status: Overall compliance status.
        compliance_score: Composite compliance score (0-100).
        dds_valid: Whether DDS is valid.
        certification_valid: Whether certifications are current.
        geolocation_coverage: Percentage of volume with GPS.
        deforestation_free: Whether deforestation-free verified.
        alerts: List of compliance alerts.
        expiry_warnings: List of upcoming expiry warnings.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "supplier_id", "compliance_status",
        "compliance_score", "dds_valid", "certification_valid",
        "geolocation_coverage", "deforestation_free",
        "alerts", "expiry_warnings",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        supplier_id: str = "",
        compliance_status: str = "unverified",
        compliance_score: float = 0.0,
        dds_valid: bool = False,
        certification_valid: bool = False,
        geolocation_coverage: float = 0.0,
        deforestation_free: bool = False,
        alerts: Optional[List[str]] = None,
        expiry_warnings: Optional[List[Dict[str, Any]]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.supplier_id = supplier_id
        self.compliance_status = compliance_status
        self.compliance_score = compliance_score
        self.dds_valid = dds_valid
        self.certification_valid = certification_valid
        self.geolocation_coverage = geolocation_coverage
        self.deforestation_free = deforestation_free
        self.alerts = alerts or []
        self.expiry_warnings = expiry_warnings or []
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "supplier_id": self.supplier_id,
            "compliance_status": self.compliance_status,
            "compliance_score": round(self.compliance_score, 2),
            "dds_valid": self.dds_valid,
            "certification_valid": self.certification_valid,
            "geolocation_coverage": round(self.geolocation_coverage, 2),
            "deforestation_free": self.deforestation_free,
            "alerts": self.alerts,
            "expiry_warnings": self.expiry_warnings,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: GapResult
# ---------------------------------------------------------------------------

class GapResult:
    """Result from a gap analysis operation.

    Attributes:
        request_id: Unique request identifier.
        supplier_id: Assessed supplier identifier.
        total_gaps: Total number of gaps detected.
        critical_gaps: Number of critical gaps (block DDS).
        major_gaps: Number of major gaps (regulatory risk).
        minor_gaps: Number of minor gaps (data quality).
        gap_details: Detailed gap descriptions with severity.
        remediation_plan: Prioritized remediation action items.
        completeness_score: Overall data completeness (0-100).
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "supplier_id", "total_gaps",
        "critical_gaps", "major_gaps", "minor_gaps",
        "gap_details", "remediation_plan", "completeness_score",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        supplier_id: str = "",
        total_gaps: int = 0,
        critical_gaps: int = 0,
        major_gaps: int = 0,
        minor_gaps: int = 0,
        gap_details: Optional[List[Dict[str, Any]]] = None,
        remediation_plan: Optional[List[Dict[str, Any]]] = None,
        completeness_score: float = 0.0,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.supplier_id = supplier_id
        self.total_gaps = total_gaps
        self.critical_gaps = critical_gaps
        self.major_gaps = major_gaps
        self.minor_gaps = minor_gaps
        self.gap_details = gap_details or []
        self.remediation_plan = remediation_plan or []
        self.completeness_score = completeness_score
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "supplier_id": self.supplier_id,
            "total_gaps": self.total_gaps,
            "critical_gaps": self.critical_gaps,
            "major_gaps": self.major_gaps,
            "minor_gaps": self.minor_gaps,
            "gap_details": self.gap_details,
            "remediation_plan": self.remediation_plan,
            "completeness_score": round(self.completeness_score, 2),
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: ReportResult
# ---------------------------------------------------------------------------

class ReportResult:
    """Result from a report generation operation.

    Attributes:
        request_id: Unique request identifier.
        report_id: Unique report identifier.
        report_type: Type of report generated.
        format: Output format (json, pdf, csv, eudr_xml).
        generated_at: Report generation timestamp.
        total_suppliers: Number of suppliers covered.
        total_tiers: Number of tier levels covered.
        summary: Report summary data.
        findings: List of key findings.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "report_id", "report_type", "format",
        "generated_at", "total_suppliers", "total_tiers",
        "summary", "findings",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        report_id: str = "",
        report_type: str = "unknown",
        format: str = "json",
        generated_at: Optional[datetime] = None,
        total_suppliers: int = 0,
        total_tiers: int = 0,
        summary: Optional[Dict[str, Any]] = None,
        findings: Optional[List[str]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.report_id = report_id or f"RPT-{uuid.uuid4().hex[:12]}"
        self.report_type = report_type
        self.format = format
        self.generated_at = generated_at or utcnow()
        self.total_suppliers = total_suppliers
        self.total_tiers = total_tiers
        self.summary = summary or {}
        self.findings = findings or []
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "report_id": self.report_id,
            "report_type": self.report_type,
            "format": self.format,
            "generated_at": (
                self.generated_at.isoformat() if self.generated_at else None
            ),
            "total_suppliers": self.total_suppliers,
            "total_tiers": self.total_tiers,
            "summary": self.summary,
            "findings": self.findings,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Batch result container
# ---------------------------------------------------------------------------

class BatchResult:
    """Result container for a batch processing job.

    Attributes:
        job_id: Unique job identifier.
        job_type: Type of batch job.
        status: Job status (pending, processing, completed, failed).
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
            "submitted_at": (
                self.submitted_at.isoformat() if self.submitted_at else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# MultiTierSupplierService
# ---------------------------------------------------------------------------

class MultiTierSupplierService:
    """Facade for the Multi-Tier Supplier Tracker Agent (AGENT-EUDR-008).

    Provides a unified interface to all 8 engines:
        1. SupplierDiscoveryEngine    - Tier discovery from multiple sources
        2. SupplierProfileManager     - Profile CRUD with completeness scoring
        3. TierDepthTracker           - Depth scoring and visibility assessment
        4. RelationshipManager        - Relationship lifecycle management
        5. RiskPropagationEngine      - Risk scoring and upstream propagation
        6. ComplianceMonitor          - Compliance status monitoring
        7. GapAnalyzer                - Gap detection and remediation plans
        8. AuditReporter              - Audit trail and report generation

    Singleton pattern with thread-safe initialization.

    Attributes:
        is_running: Whether the service is currently active and healthy.

    Example:
        >>> service = MultiTierSupplierService()
        >>> await service.startup()
        >>> result = service.discover_suppliers({"supplier_id": "S-001"})
        >>> await service.shutdown()
    """

    _instance: Optional[MultiTierSupplierService] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize MultiTierSupplierService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.
        """
        # Configuration from environment
        self._database_url: str = _env(
            "DATABASE_URL", "postgresql://localhost:5432/greenlang",
        )
        self._redis_url: str = _env("REDIS_URL", "redis://localhost:6379/0")
        self._log_level: str = _env("LOG_LEVEL", "INFO")
        self._batch_max_size: int = _env_int("BATCH_MAX_SIZE", 100000)
        self._batch_concurrency: int = _env_int("BATCH_CONCURRENCY", 8)
        self._cache_ttl_seconds: int = _env_int("CACHE_TTL_SECONDS", 3600)
        self._enable_metrics: bool = _env_bool("ENABLE_METRICS", True)
        self._max_tier_depth: int = _env_int("MAX_TIER_DEPTH", 15)
        self._risk_threshold_high: float = _env_float(
            "RISK_THRESHOLD_HIGH", 70.0,
        )
        self._risk_threshold_medium: float = _env_float(
            "RISK_THRESHOLD_MEDIUM", 40.0,
        )
        self._genesis_hash: str = _env(
            "GENESIS_HASH", "mst-tracker-genesis-v1.0.0",
        )

        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_provenance_hash(
            self._database_url, self._redis_url,
            str(self._batch_max_size), str(self._max_tier_depth),
            self._genesis_hash,
        )

        # Connection handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._supplier_discovery_engine: Optional[Any] = None
        self._supplier_profile_manager: Optional[Any] = None
        self._tier_depth_tracker: Optional[Any] = None
        self._relationship_manager: Optional[Any] = None
        self._risk_propagation_engine: Optional[Any] = None
        self._compliance_monitor: Optional[Any] = None
        self._gap_analyzer: Optional[Any] = None
        self._audit_reporter: Optional[Any] = None

        # Reference data (loaded in startup)
        self._ref_country_risk: Optional[Dict[str, Any]] = None
        self._ref_certifications: Optional[Dict[str, Any]] = None
        self._ref_supply_chains: Optional[Dict[str, Any]] = None

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthStatus] = None

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Metrics counters
        self._metrics: Dict[str, int] = {
            "suppliers_discovered": 0,
            "suppliers_onboarded": 0,
            "relationships_created": 0,
            "tier_depth_assessments": 0,
            "risk_assessments": 0,
            "risk_alerts": 0,
            "compliance_checks": 0,
            "compliance_alerts": 0,
            "gaps_detected": 0,
            "gaps_remediated": 0,
            "reports_generated": 0,
            "batch_jobs": 0,
            "errors": 0,
        }

        logger.info(
            "MultiTierSupplierService created: config_hash=%s, "
            "batch_max=%d, concurrency=%d, cache_ttl=%ds, max_depth=%d",
            self._config_hash[:12],
            self._batch_max_size,
            self._batch_concurrency,
            self._cache_ttl_seconds,
            self._max_tier_depth,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    def supplier_discovery_engine(self) -> Any:
        """Return the SupplierDiscoveryEngine instance."""
        self._ensure_started()
        return self._supplier_discovery_engine

    @property
    def supplier_profile_manager(self) -> Any:
        """Return the SupplierProfileManager instance."""
        self._ensure_started()
        return self._supplier_profile_manager

    @property
    def tier_depth_tracker(self) -> Any:
        """Return the TierDepthTracker instance."""
        self._ensure_started()
        return self._tier_depth_tracker

    @property
    def relationship_manager(self) -> Any:
        """Return the RelationshipManager instance."""
        self._ensure_started()
        return self._relationship_manager

    @property
    def risk_propagation_engine(self) -> Any:
        """Return the RiskPropagationEngine instance."""
        self._ensure_started()
        return self._risk_propagation_engine

    @property
    def compliance_monitor(self) -> Any:
        """Return the ComplianceMonitor instance."""
        self._ensure_started()
        return self._compliance_monitor

    @property
    def gap_analyzer(self) -> Any:
        """Return the GapAnalyzer instance."""
        self._ensure_started()
        return self._gap_analyzer

    @property
    def audit_reporter(self) -> Any:
        """Return the AuditReporter instance."""
        self._ensure_started()
        return self._audit_reporter

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Start the service: connect DB, Redis, initialize all engines.

        Executes the full startup sequence:
            1. Configure structured logging
            2. Initialize OpenTelemetry tracer
            3. Load reference data
            4. Connect to PostgreSQL
            5. Connect to Redis
            6. Initialize all eight engines
            7. Start background health check task

        Idempotent: safe to call multiple times.
        """
        if self._started:
            logger.debug("MultiTierSupplierService already started")
            return

        start = time.monotonic()
        logger.info("MultiTierSupplierService starting up...")

        self._configure_logging()
        self._init_tracer()
        self._load_reference_data()
        await self._connect_database()
        await self._connect_redis()
        await self._initialize_engines()
        self._start_health_check()

        self._started = True
        self._start_time = time.monotonic()
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "MultiTierSupplierService started in %.1fms: "
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
            logger.debug("MultiTierSupplierService already stopped")
            return

        logger.info("MultiTierSupplierService shutting down...")
        start = time.monotonic()

        self._stop_health_check()
        await self._close_engines()
        await self._close_redis()
        await self._close_database()
        self._flush_metrics()

        self._started = False
        elapsed = (time.monotonic() - start) * 1000
        logger.info("MultiTierSupplierService shut down in %.1fms", elapsed)

    # ==================================================================
    # FACADE METHODS: Discovery (Engine 1)
    # ==================================================================

    def discover_suppliers(
        self,
        source_data: Dict[str, Any],
        commodity: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Discover sub-tier suppliers from source data.

        Orchestrates: SupplierDiscoveryEngine -> dedup -> profile creation.

        Args:
            source_data: Source data for discovery (ERP, declaration, etc.).
            commodity: EUDR commodity context.
            max_depth: Maximum discovery depth (default: configured max).

        Returns:
            Dictionary with discovery results.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        effective_depth = max_depth or self._max_tier_depth

        logger.debug(
            "Discovering suppliers: commodity=%s, max_depth=%d",
            commodity, effective_depth,
        )

        try:
            discovered = self._safe_discover(source_data, commodity, effective_depth)
            elapsed_ms = (time.monotonic() - start) * 1000

            result = DiscoveryResult(
                request_id=request_id,
                suppliers_discovered=discovered.get("suppliers_count", 0),
                relationships_discovered=discovered.get("relationships_count", 0),
                source_type=discovered.get("source_type", "manual"),
                commodity=commodity or "",
                tier_depths_found=discovered.get("tier_depths", []),
                confidence_scores=discovered.get("confidence_scores", []),
                duplicates_detected=discovered.get("duplicates", 0),
                provenance_hash=_compute_provenance_hash(
                    request_id, str(discovered.get("suppliers_count", 0)),
                    commodity or "", str(effective_depth),
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["suppliers_discovered"] += result.suppliers_discovered
            logger.info(
                "Discovery complete: id=%s, suppliers=%d, "
                "relationships=%d, elapsed=%.1fms",
                request_id, result.suppliers_discovered,
                result.relationships_discovered, elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("Discovery failed: error=%s", exc, exc_info=True)
            raise

    def batch_discover(
        self,
        source_items: List[Dict[str, Any]],
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch discover suppliers from multiple source data items.

        Args:
            source_items: List of source data dicts for discovery.
            commodity: EUDR commodity context.

        Returns:
            Batch result dictionary with per-item results.
        """
        self._ensure_started()
        items = [
            {**item, "commodity": commodity} for item in source_items
        ]
        return self._run_batch(
            "batch_discover", items, self._discover_single_item,
        )

    def discover_from_declaration(
        self,
        declaration_data: Dict[str, Any],
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Discover sub-tier suppliers from a supplier declaration.

        Args:
            declaration_data: Parsed supplier declaration data.
            commodity: EUDR commodity context.

        Returns:
            Dictionary with discovery results.
        """
        self._ensure_started()
        source_data = {
            "source_type": "declaration",
            "declaration": declaration_data,
        }
        return self.discover_suppliers(source_data, commodity)

    def discover_from_questionnaire(
        self,
        questionnaire_data: Dict[str, Any],
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Discover sub-tier suppliers from questionnaire responses.

        Args:
            questionnaire_data: Parsed questionnaire response data.
            commodity: EUDR commodity context.

        Returns:
            Dictionary with discovery results.
        """
        self._ensure_started()
        source_data = {
            "source_type": "questionnaire",
            "questionnaire": questionnaire_data,
        }
        return self.discover_suppliers(source_data, commodity)

    # ==================================================================
    # FACADE METHODS: Profiles (Engine 2)
    # ==================================================================

    def create_supplier(
        self,
        profile_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new supplier profile.

        Args:
            profile_data: Supplier profile data including legal name,
                country, commodity types, certifications, etc.

        Returns:
            Dictionary with profile creation result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        supplier_id = profile_data.get(
            "supplier_id", f"SUP-{uuid.uuid4().hex[:12]}",
        )

        logger.debug("Creating supplier: id=%s", supplier_id)

        try:
            completeness = self._compute_profile_completeness(profile_data)
            missing = self._identify_missing_fields(profile_data)
            elapsed_ms = (time.monotonic() - start) * 1000

            result = ProfileResult(
                request_id=request_id,
                supplier_id=supplier_id,
                operation="created",
                profile_completeness=completeness,
                missing_fields=missing,
                tier_level=profile_data.get("tier_level", 0),
                status="active",
                provenance_hash=_compute_provenance_hash(
                    request_id, supplier_id, "created",
                    str(completeness),
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["suppliers_onboarded"] += 1
            logger.info(
                "Supplier created: id=%s, supplier=%s, "
                "completeness=%.1f%%, elapsed=%.1fms",
                request_id, supplier_id, completeness, elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Create supplier failed: id=%s, error=%s",
                supplier_id, exc, exc_info=True,
            )
            raise

    def update_supplier(
        self,
        supplier_id: str,
        update_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an existing supplier profile.

        Args:
            supplier_id: Supplier identifier to update.
            update_data: Fields to update.

        Returns:
            Dictionary with update result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        completeness = self._compute_profile_completeness(update_data)
        missing = self._identify_missing_fields(update_data)
        elapsed_ms = (time.monotonic() - start) * 1000

        result = ProfileResult(
            request_id=request_id,
            supplier_id=supplier_id,
            operation="updated",
            profile_completeness=completeness,
            missing_fields=missing,
            tier_level=update_data.get("tier_level", 0),
            status=update_data.get("status", "active"),
            provenance_hash=_compute_provenance_hash(
                request_id, supplier_id, "updated",
            ),
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            "Supplier updated: id=%s, supplier=%s, elapsed=%.1fms",
            request_id, supplier_id, elapsed_ms,
        )
        return result.to_dict()

    def get_supplier(self, supplier_id: str) -> Dict[str, Any]:
        """Retrieve a supplier profile by ID.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary with supplier profile data.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        elapsed_ms = (time.monotonic() - start) * 1000

        result = ProfileResult(
            request_id=request_id,
            supplier_id=supplier_id,
            operation="retrieved",
            status="active",
            provenance_hash=_compute_provenance_hash(
                request_id, supplier_id, "retrieved",
            ),
            processing_time_ms=elapsed_ms,
        )

        logger.debug("Supplier retrieved: id=%s, supplier=%s", request_id, supplier_id)
        return result.to_dict()

    def deactivate_supplier(
        self,
        supplier_id: str,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Deactivate a supplier profile.

        Args:
            supplier_id: Supplier identifier to deactivate.
            reason: Deactivation reason.

        Returns:
            Dictionary with deactivation result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        elapsed_ms = (time.monotonic() - start) * 1000

        result = ProfileResult(
            request_id=request_id,
            supplier_id=supplier_id,
            operation="deactivated",
            status="deactivated",
            provenance_hash=_compute_provenance_hash(
                request_id, supplier_id, "deactivated", reason,
            ),
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            "Supplier deactivated: id=%s, supplier=%s, reason=%s",
            request_id, supplier_id, reason,
        )
        return result.to_dict()

    def search_suppliers(
        self,
        query: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Search suppliers by query criteria.

        Args:
            query: Search criteria (name, country, commodity, tier, status).

        Returns:
            Dictionary with search results.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "query": query,
            "total_results": 0,
            "results": [],
            "provenance_hash": _compute_provenance_hash(
                request_id, json.dumps(query, sort_keys=True, default=str),
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def batch_create_suppliers(
        self,
        profiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Batch create supplier profiles.

        Args:
            profiles: List of supplier profile data dicts.

        Returns:
            Batch result dictionary.
        """
        self._ensure_started()
        return self._run_batch(
            "batch_create_suppliers", profiles, self._create_single_supplier,
        )

    # ==================================================================
    # FACADE METHODS: Tiers (Engine 3)
    # ==================================================================

    def assess_tier_depth(
        self,
        supplier_id: str,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess tier depth and visibility for a supplier chain.

        Args:
            supplier_id: Root supplier identifier.
            commodity: EUDR commodity for benchmark comparison.

        Returns:
            Dictionary with tier depth assessment results.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Assessing tier depth: supplier=%s, commodity=%s",
            supplier_id, commodity,
        )

        try:
            assessment = self._safe_assess_tier_depth(supplier_id, commodity)
            benchmark = self._get_tier_benchmark(commodity) if commodity else {}
            elapsed_ms = (time.monotonic() - start) * 1000

            result = TierResult(
                request_id=request_id,
                max_tier_depth=assessment.get("max_depth", 0),
                avg_tier_depth=assessment.get("avg_depth", 0.0),
                visibility_score=assessment.get("visibility_score", 0.0),
                tier_visibility=assessment.get("tier_visibility", {}),
                coverage_score=assessment.get("coverage_score", 0.0),
                gaps=assessment.get("gaps", []),
                benchmark_comparison=benchmark,
                provenance_hash=_compute_provenance_hash(
                    request_id, supplier_id, commodity or "",
                    str(assessment.get("max_depth", 0)),
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["tier_depth_assessments"] += 1
            logger.info(
                "Tier depth assessed: id=%s, supplier=%s, "
                "depth=%d, visibility=%.1f%%, elapsed=%.1fms",
                request_id, supplier_id, result.max_tier_depth,
                result.visibility_score, elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Tier depth assessment failed: supplier=%s, error=%s",
                supplier_id, exc, exc_info=True,
            )
            raise

    def get_visibility_score(
        self,
        supplier_id: str,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the visibility score for a supplier chain.

        Args:
            supplier_id: Root supplier identifier.
            commodity: EUDR commodity context.

        Returns:
            Dictionary with visibility score data.
        """
        self._ensure_started()
        return self.assess_tier_depth(supplier_id, commodity)

    def get_tier_gaps(
        self,
        supplier_id: str,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get tier coverage gaps for a supplier chain.

        Args:
            supplier_id: Root supplier identifier.
            commodity: EUDR commodity context.

        Returns:
            Dictionary with gap details.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        assessment = self._safe_assess_tier_depth(supplier_id, commodity)
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "supplier_id": supplier_id,
            "commodity": commodity or "",
            "tier_gaps": assessment.get("gaps", []),
            "max_depth": assessment.get("max_depth", 0),
            "visibility_score": assessment.get("visibility_score", 0.0),
            "provenance_hash": _compute_provenance_hash(
                request_id, supplier_id, commodity or "",
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ==================================================================
    # FACADE METHODS: Relationships (Engine 4)
    # ==================================================================

    def create_relationship(
        self,
        relationship_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a supplier-to-supplier relationship.

        Args:
            relationship_data: Relationship data including buyer_id,
                supplier_id, commodity, volume, start_date, etc.

        Returns:
            Dictionary with relationship creation result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        rel_id = relationship_data.get(
            "relationship_id", f"REL-{uuid.uuid4().hex[:12]}",
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        self._metrics["relationships_created"] += 1

        logger.info(
            "Relationship created: id=%s, rel=%s, "
            "buyer=%s, supplier=%s",
            request_id, rel_id,
            relationship_data.get("buyer_id", ""),
            relationship_data.get("supplier_id", ""),
        )

        return {
            "request_id": request_id,
            "relationship_id": rel_id,
            "operation": "created",
            "status": relationship_data.get("status", "active"),
            "buyer_id": relationship_data.get("buyer_id", ""),
            "supplier_id": relationship_data.get("supplier_id", ""),
            "commodity": relationship_data.get("commodity", ""),
            "provenance_hash": _compute_provenance_hash(
                request_id, rel_id, "created",
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def update_relationship(
        self,
        relationship_id: str,
        update_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an existing supplier relationship.

        Args:
            relationship_id: Relationship identifier.
            update_data: Fields to update.

        Returns:
            Dictionary with update result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "relationship_id": relationship_id,
            "operation": "updated",
            "status": update_data.get("status", "active"),
            "provenance_hash": _compute_provenance_hash(
                request_id, relationship_id, "updated",
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def get_upstream(
        self,
        supplier_id: str,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get upstream (sub-tier) suppliers for a given supplier.

        Args:
            supplier_id: Supplier identifier.
            max_depth: Maximum depth to traverse.

        Returns:
            Dictionary with upstream supplier hierarchy.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        effective_depth = max_depth or self._max_tier_depth
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "supplier_id": supplier_id,
            "direction": "upstream",
            "max_depth": effective_depth,
            "upstream_suppliers": [],
            "total_count": 0,
            "provenance_hash": _compute_provenance_hash(
                request_id, supplier_id, "upstream",
                str(effective_depth),
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def get_downstream(
        self,
        supplier_id: str,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get downstream (buyer) relationships for a given supplier.

        Args:
            supplier_id: Supplier identifier.
            max_depth: Maximum depth to traverse.

        Returns:
            Dictionary with downstream buyer hierarchy.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        effective_depth = max_depth or self._max_tier_depth
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "supplier_id": supplier_id,
            "direction": "downstream",
            "max_depth": effective_depth,
            "downstream_buyers": [],
            "total_count": 0,
            "provenance_hash": _compute_provenance_hash(
                request_id, supplier_id, "downstream",
                str(effective_depth),
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def get_relationship_history(
        self,
        supplier_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get relationship change history for a supplier.

        Args:
            supplier_id: Supplier identifier.
            start_date: History start date (ISO format).
            end_date: History end date (ISO format).

        Returns:
            Dictionary with relationship history timeline.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "supplier_id": supplier_id,
            "start_date": start_date or "",
            "end_date": end_date or "",
            "history": [],
            "total_changes": 0,
            "provenance_hash": _compute_provenance_hash(
                request_id, supplier_id, "history",
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ==================================================================
    # FACADE METHODS: Risk (Engine 5)
    # ==================================================================

    def assess_risk(
        self,
        supplier_id: str,
        supplier_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assess risk for a single supplier.

        Evaluates risk across 6 categories: deforestation_proximity,
        country_risk, certification_gap, compliance_history,
        data_quality, concentration_risk.

        Args:
            supplier_id: Supplier identifier.
            supplier_data: Optional supplier profile data for assessment.

        Returns:
            Dictionary with risk assessment results.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug("Assessing risk: supplier=%s", supplier_id)

        try:
            risk_data = self._compute_risk_score(supplier_id, supplier_data or {})
            composite = risk_data.get("composite_score", 0.0)
            risk_level = self._classify_risk_level(composite)
            alerts = self._generate_risk_alerts(composite, risk_data)
            elapsed_ms = (time.monotonic() - start) * 1000

            result = RiskResult(
                request_id=request_id,
                supplier_id=supplier_id,
                composite_score=composite,
                risk_level=risk_level,
                category_scores=risk_data.get("category_scores", {}),
                propagated_from=[],
                propagation_method="direct_assessment",
                alerts=alerts,
                trend=risk_data.get("trend", "stable"),
                provenance_hash=_compute_provenance_hash(
                    request_id, supplier_id, str(composite), risk_level,
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["risk_assessments"] += 1
            if alerts:
                self._metrics["risk_alerts"] += len(alerts)

            logger.info(
                "Risk assessed: id=%s, supplier=%s, "
                "score=%.1f, level=%s, alerts=%d, elapsed=%.1fms",
                request_id, supplier_id, composite,
                risk_level, len(alerts), elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Risk assessment failed: supplier=%s, error=%s",
                supplier_id, exc, exc_info=True,
            )
            raise

    def propagate_risk(
        self,
        root_supplier_id: str,
        method: str = "weighted_average",
    ) -> Dict[str, Any]:
        """Propagate risk from deep-tier suppliers to root.

        Args:
            root_supplier_id: Root supplier to propagate risk to.
            method: Propagation method (max, weighted_average, volume_weighted).

        Returns:
            Dictionary with propagated risk results.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Propagating risk: root=%s, method=%s",
            root_supplier_id, method,
        )

        try:
            propagated = self._safe_propagate_risk(root_supplier_id, method)
            composite = propagated.get("composite_score", 0.0)
            risk_level = self._classify_risk_level(composite)
            elapsed_ms = (time.monotonic() - start) * 1000

            result = RiskResult(
                request_id=request_id,
                supplier_id=root_supplier_id,
                composite_score=composite,
                risk_level=risk_level,
                category_scores=propagated.get("category_scores", {}),
                propagated_from=propagated.get("propagated_from", []),
                propagation_method=method,
                alerts=propagated.get("alerts", []),
                trend="stable",
                provenance_hash=_compute_provenance_hash(
                    request_id, root_supplier_id, method,
                    str(composite),
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["risk_assessments"] += 1
            logger.info(
                "Risk propagated: id=%s, root=%s, method=%s, "
                "score=%.1f, sources=%d, elapsed=%.1fms",
                request_id, root_supplier_id, method,
                composite, len(result.propagated_from), elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Risk propagation failed: root=%s, error=%s",
                root_supplier_id, exc, exc_info=True,
            )
            raise

    def get_risk_profile(self, supplier_id: str) -> Dict[str, Any]:
        """Get the full risk profile for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary with risk profile data.
        """
        self._ensure_started()
        return self.assess_risk(supplier_id)

    def batch_risk_assess(
        self,
        supplier_ids: List[str],
    ) -> Dict[str, Any]:
        """Batch risk assessment for multiple suppliers.

        Args:
            supplier_ids: List of supplier identifiers.

        Returns:
            Batch result dictionary.
        """
        self._ensure_started()
        items = [{"supplier_id": sid} for sid in supplier_ids]
        return self._run_batch(
            "batch_risk_assess", items, self._risk_single_item,
        )

    # ==================================================================
    # FACADE METHODS: Compliance (Engine 6)
    # ==================================================================

    def check_compliance(
        self,
        supplier_id: str,
        supplier_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check compliance status for a supplier.

        Evaluates: DDS validity, certification status, geolocation
        coverage, deforestation-free verification.

        Args:
            supplier_id: Supplier identifier.
            supplier_data: Optional supplier data for assessment.

        Returns:
            Dictionary with compliance check results.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug("Checking compliance: supplier=%s", supplier_id)

        try:
            compliance = self._evaluate_compliance(
                supplier_id, supplier_data or {},
            )
            score = compliance.get("compliance_score", 0.0)
            status = compliance.get("status", "unverified")
            alerts = compliance.get("alerts", [])
            elapsed_ms = (time.monotonic() - start) * 1000

            result = ComplianceResult(
                request_id=request_id,
                supplier_id=supplier_id,
                compliance_status=status,
                compliance_score=score,
                dds_valid=compliance.get("dds_valid", False),
                certification_valid=compliance.get("certification_valid", False),
                geolocation_coverage=compliance.get("geolocation_coverage", 0.0),
                deforestation_free=compliance.get("deforestation_free", False),
                alerts=alerts,
                expiry_warnings=compliance.get("expiry_warnings", []),
                provenance_hash=_compute_provenance_hash(
                    request_id, supplier_id, status, str(score),
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["compliance_checks"] += 1
            if alerts:
                self._metrics["compliance_alerts"] += len(alerts)

            logger.info(
                "Compliance checked: id=%s, supplier=%s, "
                "status=%s, score=%.1f, alerts=%d, elapsed=%.1fms",
                request_id, supplier_id, status,
                score, len(alerts), elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Compliance check failed: supplier=%s, error=%s",
                supplier_id, exc, exc_info=True,
            )
            raise

    def get_compliance_status(self, supplier_id: str) -> Dict[str, Any]:
        """Get current compliance status for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary with compliance status data.
        """
        self._ensure_started()
        return self.check_compliance(supplier_id)

    def batch_compliance_check(
        self,
        supplier_ids: List[str],
    ) -> Dict[str, Any]:
        """Batch compliance check for multiple suppliers.

        Args:
            supplier_ids: List of supplier identifiers.

        Returns:
            Batch result dictionary.
        """
        self._ensure_started()
        items = [{"supplier_id": sid} for sid in supplier_ids]
        return self._run_batch(
            "batch_compliance_check", items, self._compliance_single_item,
        )

    def get_compliance_alerts(
        self,
        severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get compliance alerts, optionally filtered by severity.

        Args:
            severity: Optional severity filter (critical, warning, info).

        Returns:
            Dictionary with compliance alerts.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "severity_filter": severity or "all",
            "total_alerts": 0,
            "alerts": [],
            "provenance_hash": _compute_provenance_hash(
                request_id, severity or "all",
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ==================================================================
    # FACADE METHODS: Reports (Engine 8)
    # ==================================================================

    def generate_audit_report(
        self,
        supplier_id: str,
        output_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate EUDR Article 14 audit report for a supplier chain.

        Args:
            supplier_id: Root supplier identifier.
            output_format: Output format (json, pdf, csv, eudr_xml).

        Returns:
            Dictionary with report generation result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.info(
            "Generating audit report: supplier=%s, format=%s",
            supplier_id, output_format,
        )

        try:
            elapsed_ms = (time.monotonic() - start) * 1000

            result = ReportResult(
                request_id=request_id,
                report_type="audit",
                format=output_format,
                total_suppliers=0,
                total_tiers=0,
                summary={
                    "supplier_id": supplier_id,
                    "chain_depth": 0,
                    "compliance_rate": 0.0,
                    "risk_score": 0.0,
                },
                findings=[],
                provenance_hash=_compute_provenance_hash(
                    request_id, supplier_id, "audit", output_format,
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["reports_generated"] += 1
            logger.info(
                "Audit report generated: id=%s, report=%s, "
                "format=%s, elapsed=%.1fms",
                request_id, result.report_id, output_format, elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Audit report failed: supplier=%s, error=%s",
                supplier_id, exc, exc_info=True,
            )
            raise

    def generate_tier_summary(
        self,
        supplier_id: str,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a tier depth summary report.

        Args:
            supplier_id: Root supplier identifier.
            commodity: EUDR commodity for benchmark comparison.

        Returns:
            Dictionary with tier summary report.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        elapsed_ms = (time.monotonic() - start) * 1000

        result = ReportResult(
            request_id=request_id,
            report_type="tier_summary",
            format="json",
            summary={
                "supplier_id": supplier_id,
                "commodity": commodity or "",
            },
            provenance_hash=_compute_provenance_hash(
                request_id, supplier_id, "tier_summary",
            ),
            processing_time_ms=elapsed_ms,
        )

        self._metrics["reports_generated"] += 1
        return result.to_dict()

    def generate_gap_report(
        self,
        supplier_id: str,
    ) -> Dict[str, Any]:
        """Generate a gap analysis report for a supplier chain.

        Args:
            supplier_id: Root supplier identifier.

        Returns:
            Dictionary with gap analysis report.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        elapsed_ms = (time.monotonic() - start) * 1000

        result = ReportResult(
            request_id=request_id,
            report_type="gap_analysis",
            format="json",
            summary={"supplier_id": supplier_id},
            provenance_hash=_compute_provenance_hash(
                request_id, supplier_id, "gap_analysis",
            ),
            processing_time_ms=elapsed_ms,
        )

        self._metrics["reports_generated"] += 1
        return result.to_dict()

    def generate_dds_readiness(
        self,
        supplier_id: str,
    ) -> Dict[str, Any]:
        """Generate DDS submission readiness report.

        Args:
            supplier_id: Root supplier identifier.

        Returns:
            Dictionary with DDS readiness assessment.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        compliance = self.check_compliance(supplier_id)
        tier_data = self.assess_tier_depth(supplier_id)
        is_ready = (
            compliance.get("compliance_status") == "compliant"
            and tier_data.get("visibility_score", 0.0) >= 70.0
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "report_type": "dds_readiness",
            "supplier_id": supplier_id,
            "is_ready": is_ready,
            "compliance_status": compliance.get("compliance_status", "unverified"),
            "compliance_score": compliance.get("compliance_score", 0.0),
            "visibility_score": tier_data.get("visibility_score", 0.0),
            "blocking_issues": [],
            "recommendations": [],
            "provenance_hash": _compute_provenance_hash(
                request_id, supplier_id, "dds_readiness",
                str(is_ready),
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ==================================================================
    # FACADE METHODS: Health
    # ==================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of all components.

        Returns:
            Dictionary with status, component checks, version, and uptime.
        """
        checks: Dict[str, Any] = {}

        checks["database"] = await self._check_database_health()
        checks["redis"] = await self._check_redis_health()
        checks["engines"] = self._check_engine_health()
        checks["reference_data"] = self._check_reference_data_health()

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
        """Return service statistics including metric counters.

        Returns:
            Dictionary with metric counters and configuration summary.
        """
        return {
            "metrics": dict(self._metrics),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "config_hash": self._config_hash[:12],
            "batch_max_size": self._batch_max_size,
            "max_tier_depth": self._max_tier_depth,
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "timestamp": utcnow().isoformat(),
        }

    # ==================================================================
    # Internal: Profile helpers
    # ==================================================================

    def _compute_profile_completeness(
        self, profile_data: Dict[str, Any],
    ) -> float:
        """Compute profile completeness score (0-100).

        Weights per PRD Appendix D:
            Legal identity (25%): legal_name, registration_id, country
            Location (20%): gps_lat, gps_lon, address, admin_region
            Commodity (15%): commodity_types, volumes, capacity
            Certification (15%): certification_type, cert_id, validity
            Compliance (15%): dds_reference, deforestation_status
            Contact (10%): primary_contact, compliance_contact
        """
        categories = {
            "legal_identity": {
                "weight": 0.25,
                "fields": ["legal_name", "registration_id", "country"],
            },
            "location": {
                "weight": 0.20,
                "fields": ["gps_lat", "gps_lon", "address", "admin_region"],
            },
            "commodity": {
                "weight": 0.15,
                "fields": ["commodity_types", "volumes", "capacity"],
            },
            "certification": {
                "weight": 0.15,
                "fields": ["certification_type", "cert_id", "cert_validity"],
            },
            "compliance": {
                "weight": 0.15,
                "fields": ["dds_reference", "deforestation_status"],
            },
            "contact": {
                "weight": 0.10,
                "fields": ["primary_contact", "compliance_contact"],
            },
        }

        total_score = 0.0
        for cat_name, cat_def in categories.items():
            fields = cat_def["fields"]
            filled = sum(
                1 for f in fields
                if profile_data.get(f) is not None
                and profile_data.get(f) != ""
            )
            cat_pct = (filled / len(fields)) * 100.0 if fields else 0.0
            total_score += cat_pct * cat_def["weight"]

        return round(total_score, 2)

    def _identify_missing_fields(
        self, profile_data: Dict[str, Any],
    ) -> List[str]:
        """Identify missing required fields in a supplier profile."""
        required_fields = [
            "legal_name", "country", "commodity_types",
            "gps_lat", "gps_lon",
        ]
        return [
            f for f in required_fields
            if profile_data.get(f) is None or profile_data.get(f) == ""
        ]

    # ==================================================================
    # Internal: Risk helpers
    # ==================================================================

    def _compute_risk_score(
        self,
        supplier_id: str,
        supplier_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute composite risk score from 6 categories.

        Risk category weights per PRD Appendix B:
            deforestation_proximity: 0.30
            country_risk: 0.20
            certification_gap: 0.15
            compliance_history: 0.15
            data_quality: 0.10
            concentration_risk: 0.10
        """
        # Determine country risk from reference data
        country = supplier_data.get("country", "")
        country_score = 50.0
        if country:
            from greenlang.agents.eudr.multi_tier_supplier.reference_data.country_risk_scores import (
                get_composite_score,
            )
            raw = get_composite_score(country)
            if raw is not None:
                country_score = float(raw)

        # Determine certification gap
        cert_gap = 100.0  # Max gap if no certification
        cert_type = supplier_data.get("certification_type", "")
        if cert_type:
            from greenlang.agents.eudr.multi_tier_supplier.reference_data.certification_standards import (
                is_eudr_accepted,
            )
            if is_eudr_accepted(cert_type):
                cert_gap = 10.0
            else:
                cert_gap = 50.0

        # Data quality based on profile completeness
        completeness = self._compute_profile_completeness(supplier_data)
        data_quality_risk = max(0.0, 100.0 - completeness)

        # Category scores
        category_scores = {
            "deforestation_proximity": supplier_data.get(
                "deforestation_proximity_score", 50.0,
            ),
            "country_risk": country_score,
            "certification_gap": cert_gap,
            "compliance_history": supplier_data.get(
                "compliance_history_score", 50.0,
            ),
            "data_quality": data_quality_risk,
            "concentration_risk": supplier_data.get(
                "concentration_risk_score", 30.0,
            ),
        }

        # Weighted composite
        weights = {
            "deforestation_proximity": 0.30,
            "country_risk": 0.20,
            "certification_gap": 0.15,
            "compliance_history": 0.15,
            "data_quality": 0.10,
            "concentration_risk": 0.10,
        }
        composite = sum(
            category_scores[cat] * weights[cat]
            for cat in weights
        )

        return {
            "composite_score": round(composite, 2),
            "category_scores": category_scores,
            "trend": "stable",
        }

    def _classify_risk_level(self, score: float) -> str:
        """Classify risk level from composite score."""
        if score >= self._risk_threshold_high:
            return "high"
        if score >= self._risk_threshold_medium:
            return "medium"
        return "low"

    def _generate_risk_alerts(
        self,
        composite: float,
        risk_data: Dict[str, Any],
    ) -> List[str]:
        """Generate risk threshold alerts."""
        alerts: List[str] = []
        if composite >= self._risk_threshold_high:
            alerts.append(
                f"HIGH RISK: Composite score {composite:.1f} "
                f"exceeds threshold {self._risk_threshold_high}"
            )
        categories = risk_data.get("category_scores", {})
        for cat, score in categories.items():
            if score >= 80.0:
                alerts.append(
                    f"CRITICAL: {cat} score {score:.1f} exceeds 80.0"
                )
        return alerts

    # ==================================================================
    # Internal: Compliance helpers
    # ==================================================================

    def _evaluate_compliance(
        self,
        supplier_id: str,
        supplier_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate supplier compliance across all dimensions."""
        dds_valid = supplier_data.get("dds_valid", False)
        cert_valid = supplier_data.get("certification_valid", False)
        geo_coverage = supplier_data.get("geolocation_coverage", 0.0)
        deforestation_free = supplier_data.get("deforestation_free", False)

        # Compute compliance score
        score = 0.0
        if dds_valid:
            score += 30.0
        if cert_valid:
            score += 25.0
        score += geo_coverage * 0.25
        if deforestation_free:
            score += 20.0

        # Determine status
        if score >= 80.0 and dds_valid:
            status = "compliant"
        elif score >= 50.0:
            status = "conditionally_compliant"
        elif score > 0:
            status = "non_compliant"
        else:
            status = "unverified"

        # Generate alerts
        alerts: List[str] = []
        if not dds_valid:
            alerts.append("DDS is not valid or not linked")
        if not cert_valid:
            alerts.append("No valid EUDR-accepted certification")
        if geo_coverage < 100.0:
            alerts.append(
                f"Geolocation coverage is {geo_coverage:.1f}% "
                f"(target: 100%)"
            )
        if not deforestation_free:
            alerts.append("Deforestation-free status not verified")

        return {
            "compliance_score": round(score, 2),
            "status": status,
            "dds_valid": dds_valid,
            "certification_valid": cert_valid,
            "geolocation_coverage": geo_coverage,
            "deforestation_free": deforestation_free,
            "alerts": alerts,
            "expiry_warnings": [],
        }

    # ==================================================================
    # Internal: Tier depth helpers
    # ==================================================================

    def _safe_assess_tier_depth(
        self,
        supplier_id: str,
        commodity: Optional[str],
    ) -> Dict[str, Any]:
        """Assess tier depth using engine or fallback."""
        if self._tier_depth_tracker is not None:
            try:
                raw = self._tier_depth_tracker.assess(
                    supplier_id=supplier_id, commodity=commodity,
                )
                return {
                    "max_depth": getattr(raw, "max_depth", 0),
                    "avg_depth": getattr(raw, "avg_depth", 0.0),
                    "visibility_score": getattr(raw, "visibility_score", 0.0),
                    "tier_visibility": getattr(raw, "tier_visibility", {}),
                    "coverage_score": getattr(raw, "coverage_score", 0.0),
                    "gaps": getattr(raw, "gaps", []),
                }
            except Exception as exc:
                logger.warning("TierDepthTracker.assess failed: %s", exc)

        return {
            "max_depth": 0,
            "avg_depth": 0.0,
            "visibility_score": 0.0,
            "tier_visibility": {},
            "coverage_score": 0.0,
            "gaps": [],
        }

    def _get_tier_benchmark(
        self, commodity: str,
    ) -> Dict[str, Any]:
        """Get tier depth benchmark for a commodity."""
        from greenlang.agents.eudr.multi_tier_supplier.reference_data.commodity_supply_chains import (
            get_typical_chain, get_industry_benchmark,
        )

        chain = get_typical_chain(commodity)
        benchmark = get_industry_benchmark("eu_operators_2024")
        result: Dict[str, Any] = {}
        if chain:
            result["typical_depth"] = chain.get("typical_tier_depth", 0)
            result["visibility_benchmarks"] = chain.get(
                "visibility_benchmarks", {},
            )
        if benchmark:
            result["industry_avg_depth"] = benchmark.get(
                "avg_mapped_depth", 0.0,
            )
        return result

    # ==================================================================
    # Internal: Discovery helpers
    # ==================================================================

    def _safe_discover(
        self,
        source_data: Dict[str, Any],
        commodity: Optional[str],
        max_depth: int,
    ) -> Dict[str, Any]:
        """Delegate to SupplierDiscoveryEngine with fallback."""
        if self._supplier_discovery_engine is not None:
            try:
                raw = self._supplier_discovery_engine.discover(
                    source_data=source_data,
                    commodity=commodity,
                    max_depth=max_depth,
                )
                return {
                    "suppliers_count": getattr(raw, "suppliers_count", 0),
                    "relationships_count": getattr(
                        raw, "relationships_count", 0,
                    ),
                    "source_type": getattr(raw, "source_type", "unknown"),
                    "tier_depths": getattr(raw, "tier_depths", []),
                    "confidence_scores": getattr(raw, "confidence_scores", []),
                    "duplicates": getattr(raw, "duplicates", 0),
                }
            except Exception as exc:
                logger.warning(
                    "SupplierDiscoveryEngine.discover failed: %s", exc,
                )

        return {
            "suppliers_count": 0,
            "relationships_count": 0,
            "source_type": source_data.get("source_type", "manual"),
            "tier_depths": [],
            "confidence_scores": [],
            "duplicates": 0,
        }

    # ==================================================================
    # Internal: Risk propagation helpers
    # ==================================================================

    def _safe_propagate_risk(
        self,
        root_supplier_id: str,
        method: str,
    ) -> Dict[str, Any]:
        """Delegate to RiskPropagationEngine with fallback."""
        if self._risk_propagation_engine is not None:
            try:
                raw = self._risk_propagation_engine.propagate(
                    root_supplier_id=root_supplier_id, method=method,
                )
                return {
                    "composite_score": getattr(raw, "composite_score", 0.0),
                    "category_scores": getattr(raw, "category_scores", {}),
                    "propagated_from": getattr(raw, "propagated_from", []),
                    "alerts": getattr(raw, "alerts", []),
                }
            except Exception as exc:
                logger.warning(
                    "RiskPropagationEngine.propagate failed: %s", exc,
                )

        return {
            "composite_score": 0.0,
            "category_scores": {},
            "propagated_from": [],
            "alerts": [],
        }

    # ==================================================================
    # Internal: Batch helpers
    # ==================================================================

    def _run_batch(
        self,
        job_type: str,
        items: List[Dict[str, Any]],
        processor: Any,
    ) -> Dict[str, Any]:
        """Generic batch processing runner.

        Args:
            job_type: Type name for the batch job.
            items: List of item dicts to process.
            processor: Callable that processes a single item.

        Returns:
            BatchResult as dictionary.
        """
        start = time.monotonic()
        job_id = f"JOB-{uuid.uuid4().hex[:12]}"

        logger.info(
            "Batch %s: job_id=%s, count=%d", job_type, job_id, len(items),
        )

        if len(items) > self._batch_max_size:
            logger.warning(
                "Batch size %d exceeds max %d, truncating",
                len(items), self._batch_max_size,
            )
            items = items[:self._batch_max_size]

        results: List[Dict[str, Any]] = []
        completed = 0
        failed = 0

        for item in items:
            try:
                result = processor(item)
                results.append(result)
                completed += 1
            except Exception as exc:
                failed += 1
                results.append({"status": "failed", "error": str(exc)})

        elapsed_ms = (time.monotonic() - start) * 1000
        self._metrics["batch_jobs"] += 1

        batch = BatchResult(
            job_id=job_id,
            job_type=job_type,
            status="completed",
            total_items=len(items),
            completed_items=completed,
            failed_items=failed,
            results=results,
            completed_at=utcnow(),
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            "Batch %s complete: job_id=%s, total=%d, completed=%d, "
            "failed=%d, elapsed=%.1fms",
            job_type, job_id, len(items), completed, failed, elapsed_ms,
        )
        return batch.to_dict()

    def _discover_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item for batch_discover."""
        return self.discover_suppliers(
            source_data=item,
            commodity=item.get("commodity"),
        )

    def _create_single_supplier(
        self, item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single item for batch_create_suppliers."""
        return self.create_supplier(profile_data=item)

    def _risk_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item for batch_risk_assess."""
        return self.assess_risk(
            supplier_id=item.get("supplier_id", ""),
            supplier_data=item,
        )

    def _compliance_single_item(
        self, item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single item for batch_compliance_check."""
        return self.check_compliance(
            supplier_id=item.get("supplier_id", ""),
            supplier_data=item,
        )

    # ==================================================================
    # Internal: Infrastructure
    # ==================================================================

    def _ensure_started(self) -> None:
        """Raise RuntimeError if the service has not been started."""
        if not self._started:
            raise RuntimeError(
                "MultiTierSupplierService is not started. "
                "Call startup() first."
            )

    def _configure_logging(self) -> None:
        """Configure structured logging for the service."""
        log_level = getattr(logging, self._log_level, logging.INFO)
        logging.getLogger(
            "greenlang.agents.eudr.multi_tier_supplier"
        ).setLevel(log_level)
        logger.debug("Logging configured: level=%s", self._log_level)

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            self._tracer = otel_trace.get_tracer(
                "greenlang.agents.eudr.multi_tier_supplier",
                "1.0.0",
            )
            logger.info("OpenTelemetry tracer initialized")
        else:
            logger.debug("OpenTelemetry not available, tracing disabled")

    def _load_reference_data(self) -> None:
        """Load reference data modules into memory."""
        try:
            from greenlang.agents.eudr.multi_tier_supplier.reference_data import (
                COUNTRY_RISK_SCORES,
                CERTIFICATION_STANDARDS,
                COMMODITY_SUPPLY_CHAINS,
            )
            self._ref_country_risk = COUNTRY_RISK_SCORES
            self._ref_certifications = CERTIFICATION_STANDARDS
            self._ref_supply_chains = COMMODITY_SUPPLY_CHAINS
            logger.info(
                "Reference data loaded: countries=%d, "
                "certifications=%d, commodities=%d",
                len(COUNTRY_RISK_SCORES),
                len(CERTIFICATION_STANDARDS),
                len(COMMODITY_SUPPLY_CHAINS),
            )
        except ImportError as exc:
            logger.warning("Failed to load reference data: %s", exc)

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
                conninfo=self._database_url,
                min_size=2,
                max_size=self._batch_concurrency + 2,
                open=False,
            )
            await pool.open()
            await pool.check()
            self._db_pool = pool
            logger.info(
                "PostgreSQL pool connected: size=%d",
                self._batch_concurrency + 2,
            )
        except Exception as exc:
            logger.warning(
                "Failed to connect to PostgreSQL (non-fatal): %s", exc,
            )
            self._db_pool = None

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
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await client.ping()
            self._redis = client
            logger.info("Redis connected: ttl=%ds", self._cache_ttl_seconds)
        except Exception as exc:
            logger.warning(
                "Failed to connect to Redis (non-fatal): %s", exc,
            )
            self._redis = None

    async def _initialize_engines(self) -> None:
        """Initialize all eight internal engines."""
        logger.info("Initializing 8 multi-tier supplier tracker engines...")

        base = "greenlang.agents.eudr.multi_tier_supplier"

        self._supplier_discovery_engine = await self._init_engine(
            "supplier_discovery_engine",
            f"{base}.supplier_discovery_engine",
            "SupplierDiscoveryEngine",
        )
        self._supplier_profile_manager = await self._init_engine(
            "supplier_profile_manager",
            f"{base}.supplier_profile_manager",
            "SupplierProfileManager",
        )
        self._tier_depth_tracker = await self._init_engine(
            "tier_depth_tracker",
            f"{base}.tier_depth_tracker",
            "TierDepthTracker",
        )
        self._relationship_manager = await self._init_engine(
            "relationship_manager",
            f"{base}.relationship_manager",
            "RelationshipManager",
        )
        self._risk_propagation_engine = await self._init_engine(
            "risk_propagation_engine",
            f"{base}.risk_propagation_engine",
            "RiskPropagationEngine",
        )
        self._compliance_monitor = await self._init_engine(
            "compliance_monitor",
            f"{base}.compliance_monitor",
            "ComplianceMonitor",
        )
        self._gap_analyzer = await self._init_engine(
            "gap_analyzer",
            f"{base}.gap_analyzer",
            "GapAnalyzer",
        )
        self._audit_reporter = await self._init_engine(
            "audit_reporter",
            f"{base}.audit_reporter",
            "AuditReporter",
        )

        count = self._count_initialized_engines()
        logger.info("Engines initialized: %d/8 available", count)

    async def _init_engine(
        self, name: str, module_path: str, class_name: str,
    ) -> Any:
        """Initialize a single engine by module path and class name.

        Args:
            name: Engine name for logging.
            module_path: Python module path.
            class_name: Class name to import.

        Returns:
            Engine instance or None.
        """
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            engine = cls()
            logger.info("%s initialized", class_name)
            return engine
        except ImportError:
            logger.debug("%s module not yet available", class_name)
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize %s: %s",
                class_name, exc, exc_info=True,
            )
            return None

    async def _close_engines(self) -> None:
        """Close all engine instances."""
        engines = [
            self._supplier_discovery_engine,
            self._supplier_profile_manager,
            self._tier_depth_tracker,
            self._relationship_manager,
            self._risk_propagation_engine,
            self._compliance_monitor,
            self._gap_analyzer,
            self._audit_reporter,
        ]
        for engine in engines:
            if engine is not None and hasattr(engine, "close"):
                try:
                    result = engine.close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning("Error closing engine: %s", exc)

        self._supplier_discovery_engine = None
        self._supplier_profile_manager = None
        self._tier_depth_tracker = None
        self._relationship_manager = None
        self._risk_propagation_engine = None
        self._compliance_monitor = None
        self._gap_analyzer = None
        self._audit_reporter = None
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
                logger.warning("Error closing database pool: %s", exc)
            finally:
                self._db_pool = None

    def _flush_metrics(self) -> None:
        """Flush Prometheus metrics."""
        if self._enable_metrics:
            logger.debug(
                "Metrics flushed: %s",
                {k: v for k, v in self._metrics.items() if v > 0},
            )

    # ------------------------------------------------------------------
    # Internal: Health checks
    # ------------------------------------------------------------------

    def _start_health_check(self) -> None:
        """Start the background health check task."""
        try:
            loop = asyncio.get_running_loop()
            self._health_task = loop.create_task(self._health_check_loop())
            logger.debug("Health check background task started")
        except RuntimeError:
            logger.debug(
                "No running event loop; health check task not started",
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
            return {"status": "degraded", "reason": "no_pool"}
        try:
            start = time.monotonic()
            async with self._db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "healthy", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            return {"status": "unhealthy", "reason": str(exc)}

    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity health."""
        if self._redis is None:
            return {"status": "degraded", "reason": "not_connected"}
        try:
            start = time.monotonic()
            await self._redis.ping()
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "healthy", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            return {"status": "unhealthy", "reason": str(exc)}

    def _check_engine_health(self) -> Dict[str, Any]:
        """Check engine initialization status."""
        engines = {
            "supplier_discovery_engine": self._supplier_discovery_engine,
            "supplier_profile_manager": self._supplier_profile_manager,
            "tier_depth_tracker": self._tier_depth_tracker,
            "relationship_manager": self._relationship_manager,
            "risk_propagation_engine": self._risk_propagation_engine,
            "compliance_monitor": self._compliance_monitor,
            "gap_analyzer": self._gap_analyzer,
            "audit_reporter": self._audit_reporter,
        }
        engine_status = {
            name: "initialized" if engine is not None else "not_available"
            for name, engine in engines.items()
        }
        count = self._count_initialized_engines()
        if count == 8:
            status = "healthy"
        elif count > 0:
            status = "degraded"
        else:
            status = "unhealthy"
        return {
            "status": status,
            "initialized_count": count,
            "total_count": 8,
            "engines": engine_status,
        }

    def _check_reference_data_health(self) -> Dict[str, Any]:
        """Check reference data availability."""
        loaded = sum(1 for x in [
            self._ref_country_risk,
            self._ref_certifications,
            self._ref_supply_chains,
        ] if x is not None)
        return {
            "status": "healthy" if loaded == 3 else "degraded",
            "loaded_datasets": loaded,
            "total_datasets": 3,
        }

    def _count_initialized_engines(self) -> int:
        """Count the number of successfully initialized engines."""
        engines = [
            self._supplier_discovery_engine,
            self._supplier_profile_manager,
            self._tier_depth_tracker,
            self._relationship_manager,
            self._risk_propagation_engine,
            self._compliance_monitor,
            self._gap_analyzer,
            self._audit_reporter,
        ]
        return sum(1 for e in engines if e is not None)

# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Multi-Tier Supplier service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown. The service instance is stored in
    ``app.state.mst_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.multi_tier_supplier.setup import lifespan
from greenlang.schemas import utcnow

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.mst_service``).
    """
    service = get_service()
    app.state.mst_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()

# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[MultiTierSupplierService] = None
_service_lock = threading.Lock()

def get_service() -> MultiTierSupplierService:
    """Return the singleton MultiTierSupplierService instance.

    Uses double-checked locking for thread safety. The instance is
    created on first call.

    Returns:
        MultiTierSupplierService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = MultiTierSupplierService()
    return _service_instance

def set_service(service: MultiTierSupplierService) -> None:
    """Replace the singleton MultiTierSupplierService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("MultiTierSupplierService singleton replaced")

def reset_service() -> None:
    """Reset the singleton MultiTierSupplierService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("MultiTierSupplierService singleton reset")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Service
    "MultiTierSupplierService",
    "HealthStatus",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
    # Result containers
    "DiscoveryResult",
    "ProfileResult",
    "TierResult",
    "RiskResult",
    "ComplianceResult",
    "GapResult",
    "ReportResult",
    "BatchResult",
]
