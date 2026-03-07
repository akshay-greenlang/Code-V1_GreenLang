# -*- coding: utf-8 -*-
"""
GPSCoordinateValidatorService - Facade for AGENT-EUDR-007 GPS Coordinate Validator

This module implements the GPSCoordinateValidatorService, the single entry point
for all GPS coordinate validation operations in the GL-EUDR-APP. It manages the
lifecycle of eight internal engines, an async PostgreSQL connection pool
(psycopg + psycopg_pool), a Redis cache connection, OpenTelemetry tracing,
and Prometheus metrics. The service exposes a unified interface consumed by
the FastAPI router layer and the GL-EUDR-APP integration.

Lifecycle:
    startup  -> load config -> connect DB -> register pgvector -> connect Redis
             -> initialize engines -> start health check background task
    shutdown -> close engines -> close Redis -> close DB pool -> flush metrics

Engines (8):
    1. CoordinateParser          - Multi-format coordinate parsing (Feature 1)
    2. DatumTransformer          - Geodetic datum transformation (Feature 2)
    3. PrecisionAnalyzer         - Precision assessment & EUDR adequacy (Feature 3)
    4. FormatValidator           - Format validation & error detection (Feature 4)
    5. SpatialPlausibilityChecker - Land/ocean, country, commodity checks (Feature 5)
    6. ReverseGeocoder           - Coordinate-to-location lookup (Feature 6)
    7. AccuracyAssessor          - Overall accuracy scoring (Feature 7)
    8. ComplianceReporter        - EUDR compliance certification (Feature 8)

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.gps_coordinate_validator.setup import (
    ...     GPSCoordinateValidatorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
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

_ENV_PREFIX = "GL_EUDR_GPS_"


def _env(key: str, default: str = "") -> str:
    """Read an environment variable with the GL_EUDR_GPS_ prefix."""
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


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    return f"GPS-{uuid.uuid4().hex[:12]}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km using the Haversine formula.

    Args:
        lat1: First point latitude (degrees).
        lon1: First point longitude (degrees).
        lat2: Second point latitude (degrees).
        lon2: Second point longitude (degrees).

    Returns:
        Distance in kilometres.
    """
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return 6371.0 * c


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
        self.timestamp = timestamp or _utcnow()
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
# Result container: CoordinateResult
# ---------------------------------------------------------------------------


class CoordinateResult:
    """Result from a coordinate parsing or transformation operation.

    Attributes:
        request_id: Unique request identifier.
        lat: Parsed/transformed latitude in decimal degrees WGS84.
        lon: Parsed/transformed longitude in decimal degrees WGS84.
        original_lat: Original latitude before transformation.
        original_lon: Original longitude before transformation.
        input_format: Detected or declared input format.
        source_datum: Source geodetic datum.
        target_datum: Target geodetic datum (always WGS84 for output).
        precision_level: Precision classification.
        decimal_places: Number of significant decimal places.
        ground_resolution_m: Estimated ground resolution in metres.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "lat", "lon", "original_lat", "original_lon",
        "input_format", "source_datum", "target_datum",
        "precision_level", "decimal_places", "ground_resolution_m",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        lat: float = 0.0,
        lon: float = 0.0,
        original_lat: float = 0.0,
        original_lon: float = 0.0,
        input_format: str = "decimal_degrees",
        source_datum: str = "WGS84",
        target_datum: str = "WGS84",
        precision_level: str = "unknown",
        decimal_places: int = 0,
        ground_resolution_m: float = 0.0,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.lat = lat
        self.lon = lon
        self.original_lat = original_lat
        self.original_lon = original_lon
        self.input_format = input_format
        self.source_datum = source_datum
        self.target_datum = target_datum
        self.precision_level = precision_level
        self.decimal_places = decimal_places
        self.ground_resolution_m = ground_resolution_m
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "lat": round(self.lat, 8),
            "lon": round(self.lon, 8),
            "original_lat": round(self.original_lat, 8),
            "original_lon": round(self.original_lon, 8),
            "input_format": self.input_format,
            "source_datum": self.source_datum,
            "target_datum": self.target_datum,
            "precision_level": self.precision_level,
            "decimal_places": self.decimal_places,
            "ground_resolution_m": round(self.ground_resolution_m, 3),
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: ValidationResult
# ---------------------------------------------------------------------------


class ValidationResult:
    """Result from a coordinate validation operation.

    Attributes:
        request_id: Unique request identifier.
        is_valid: Whether the coordinate passes all validation checks.
        errors: List of error-level issues found.
        warnings: List of warning-level issues found.
        checks_performed: Number of validation checks evaluated.
        checks_passed: Number of checks that passed.
        checks_failed: Number of checks that failed.
        error_types: List of error type codes detected.
        suggested_corrections: Suggested corrections for detected errors.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "is_valid", "errors", "warnings",
        "checks_performed", "checks_passed", "checks_failed",
        "error_types", "suggested_corrections",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        is_valid: bool = True,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        checks_performed: int = 0,
        checks_passed: int = 0,
        checks_failed: int = 0,
        error_types: Optional[List[str]] = None,
        suggested_corrections: Optional[List[Dict[str, Any]]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.checks_performed = checks_performed
        self.checks_passed = checks_passed
        self.checks_failed = checks_failed
        self.error_types = error_types or []
        self.suggested_corrections = suggested_corrections or []
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks_performed": self.checks_performed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "error_types": self.error_types,
            "suggested_corrections": self.suggested_corrections,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: PlausibilityResult
# ---------------------------------------------------------------------------


class PlausibilityResult:
    """Result from a spatial plausibility check.

    Attributes:
        request_id: Unique request identifier.
        is_plausible: Whether the coordinate is spatially plausible.
        is_on_land: Whether the coordinate is on land.
        detected_country: ISO code of the detected country.
        declared_country_match: Whether declared country matches detected.
        commodity_plausible: Whether commodity is plausible at location.
        elevation_plausible: Whether elevation is plausible for commodity.
        is_urban: Whether the coordinate falls in a major city.
        findings: List of plausibility findings.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "is_plausible", "is_on_land",
        "detected_country", "declared_country_match",
        "commodity_plausible", "elevation_plausible", "is_urban",
        "findings", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        is_plausible: bool = True,
        is_on_land: bool = True,
        detected_country: str = "",
        declared_country_match: bool = True,
        commodity_plausible: bool = True,
        elevation_plausible: bool = True,
        is_urban: bool = False,
        findings: Optional[List[str]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.is_plausible = is_plausible
        self.is_on_land = is_on_land
        self.detected_country = detected_country
        self.declared_country_match = declared_country_match
        self.commodity_plausible = commodity_plausible
        self.elevation_plausible = elevation_plausible
        self.is_urban = is_urban
        self.findings = findings or []
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "is_plausible": self.is_plausible,
            "is_on_land": self.is_on_land,
            "detected_country": self.detected_country,
            "declared_country_match": self.declared_country_match,
            "commodity_plausible": self.commodity_plausible,
            "elevation_plausible": self.elevation_plausible,
            "is_urban": self.is_urban,
            "findings": self.findings,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: AccuracyResult
# ---------------------------------------------------------------------------


class AccuracyResult:
    """Result from an accuracy assessment operation.

    Attributes:
        request_id: Unique request identifier.
        overall_score: Overall accuracy score 0-100.
        grade: Letter grade (A/B/C/D/F).
        eudr_compliant: Whether the coordinate meets EUDR requirements.
        precision_score: Precision component score 0-100.
        validation_score: Validation component score 0-100.
        plausibility_score: Plausibility component score 0-100.
        source_type: GPS data source type.
        recommendations: List of improvement recommendations.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "overall_score", "grade", "eudr_compliant",
        "precision_score", "validation_score", "plausibility_score",
        "source_type", "recommendations",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        overall_score: float = 0.0,
        grade: str = "F",
        eudr_compliant: bool = False,
        precision_score: float = 0.0,
        validation_score: float = 0.0,
        plausibility_score: float = 0.0,
        source_type: str = "unknown",
        recommendations: Optional[List[str]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.overall_score = overall_score
        self.grade = grade
        self.eudr_compliant = eudr_compliant
        self.precision_score = precision_score
        self.validation_score = validation_score
        self.plausibility_score = plausibility_score
        self.source_type = source_type
        self.recommendations = recommendations or []
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "overall_score": round(self.overall_score, 2),
            "grade": self.grade,
            "eudr_compliant": self.eudr_compliant,
            "precision_score": round(self.precision_score, 2),
            "validation_score": round(self.validation_score, 2),
            "plausibility_score": round(self.plausibility_score, 2),
            "source_type": self.source_type,
            "recommendations": self.recommendations,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: GeocodeResult
# ---------------------------------------------------------------------------


class GeocodeResult:
    """Result from a reverse geocoding operation.

    Attributes:
        request_id: Unique request identifier.
        country_iso: ISO 3166-1 alpha-2 country code.
        country_name: Full country name.
        is_on_land: Whether the coordinate is on land.
        region: Detected administrative region or ocean name.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "country_iso", "country_name", "is_on_land",
        "region", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        country_iso: str = "",
        country_name: str = "",
        is_on_land: bool = True,
        region: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.country_iso = country_iso
        self.country_name = country_name
        self.is_on_land = is_on_land
        self.region = region
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "country_iso": self.country_iso,
            "country_name": self.country_name,
            "is_on_land": self.is_on_land,
            "region": self.region,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: ComplianceCertResult
# ---------------------------------------------------------------------------


class ComplianceCertResult:
    """Result from a compliance certification generation.

    Attributes:
        request_id: Unique request identifier.
        certificate_id: Unique certificate identifier.
        is_compliant: Whether the coordinate passes EUDR compliance.
        lat: Validated latitude.
        lon: Validated longitude.
        commodity: EUDR commodity.
        country_iso: Country ISO code.
        accuracy_grade: Accuracy letter grade.
        overall_score: Accuracy score 0-100.
        validation_passed: Whether format validation passed.
        plausibility_passed: Whether spatial plausibility passed.
        precision_adequate: Whether precision meets EUDR requirements.
        findings: List of findings for the certificate.
        generated_at: Certificate generation timestamp.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "certificate_id", "is_compliant",
        "lat", "lon", "commodity", "country_iso",
        "accuracy_grade", "overall_score",
        "validation_passed", "plausibility_passed", "precision_adequate",
        "findings", "generated_at",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        certificate_id: str = "",
        is_compliant: bool = False,
        lat: float = 0.0,
        lon: float = 0.0,
        commodity: str = "",
        country_iso: str = "",
        accuracy_grade: str = "F",
        overall_score: float = 0.0,
        validation_passed: bool = False,
        plausibility_passed: bool = False,
        precision_adequate: bool = False,
        findings: Optional[List[str]] = None,
        generated_at: Optional[datetime] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.certificate_id = certificate_id or f"CERT-{uuid.uuid4().hex[:12]}"
        self.is_compliant = is_compliant
        self.lat = lat
        self.lon = lon
        self.commodity = commodity
        self.country_iso = country_iso
        self.accuracy_grade = accuracy_grade
        self.overall_score = overall_score
        self.validation_passed = validation_passed
        self.plausibility_passed = plausibility_passed
        self.precision_adequate = precision_adequate
        self.findings = findings or []
        self.generated_at = generated_at or _utcnow()
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "certificate_id": self.certificate_id,
            "is_compliant": self.is_compliant,
            "lat": round(self.lat, 8),
            "lon": round(self.lon, 8),
            "commodity": self.commodity,
            "country_iso": self.country_iso,
            "accuracy_grade": self.accuracy_grade,
            "overall_score": round(self.overall_score, 2),
            "validation_passed": self.validation_passed,
            "plausibility_passed": self.plausibility_passed,
            "precision_adequate": self.precision_adequate,
            "findings": self.findings,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: BatchResult
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
        self.submitted_at = submitted_at or _utcnow()
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
# GPSCoordinateValidatorService
# ---------------------------------------------------------------------------


class GPSCoordinateValidatorService:
    """Facade for the GPS Coordinate Validator Agent (AGENT-EUDR-007).

    Provides a unified interface to all 8 engines:
        1. CoordinateParser           - Multi-format coordinate parsing
        2. DatumTransformer           - Geodetic datum transformation
        3. PrecisionAnalyzer          - Precision assessment & EUDR adequacy
        4. FormatValidator            - Format validation & error detection
        5. SpatialPlausibilityChecker - Land/ocean, country, commodity
        6. ReverseGeocoder            - Coordinate-to-location lookup
        7. AccuracyAssessor           - Overall accuracy scoring
        8. ComplianceReporter         - EUDR compliance certification

    Singleton pattern with thread-safe initialization.

    Attributes:
        is_running: Whether the service is currently active and healthy.

    Example:
        >>> service = GPSCoordinateValidatorService()
        >>> await service.startup()
        >>> result = service.validate_coordinate(5.603, -0.187, "GH", "cocoa")
        >>> await service.shutdown()
    """

    _instance: Optional[GPSCoordinateValidatorService] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize GPSCoordinateValidatorService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.
        """
        # Configuration from environment
        self._database_url: str = _env("DATABASE_URL", "postgresql://localhost:5432/greenlang")
        self._redis_url: str = _env("REDIS_URL", "redis://localhost:6379/0")
        self._log_level: str = _env("LOG_LEVEL", "INFO")
        self._batch_max_size: int = _env_int("BATCH_MAX_SIZE", 10000)
        self._batch_concurrency: int = _env_int("BATCH_CONCURRENCY", 4)
        self._cache_ttl_seconds: int = _env_int("CACHE_TTL_SECONDS", 3600)
        self._enable_metrics: bool = _env_bool("ENABLE_METRICS", True)
        self._null_island_radius_km: float = _env_float("NULL_ISLAND_RADIUS_KM", 1.0)
        self._duplicate_threshold_m: float = _env_float("DUPLICATE_THRESHOLD_M", 0.1)
        self._genesis_hash: str = _env("GENESIS_HASH", "gps-validator-genesis-v1.0.0")

        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_provenance_hash(
            self._database_url, self._redis_url,
            str(self._batch_max_size), self._genesis_hash,
        )

        # Connection handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._coordinate_parser: Optional[Any] = None
        self._datum_transformer: Optional[Any] = None
        self._precision_analyzer: Optional[Any] = None
        self._format_validator: Optional[Any] = None
        self._spatial_plausibility_checker: Optional[Any] = None
        self._reverse_geocoder: Optional[Any] = None
        self._accuracy_assessor: Optional[Any] = None
        self._compliance_reporter: Optional[Any] = None

        # Reference data (loaded in startup)
        self._ref_country_boundaries: Optional[Dict[str, Any]] = None
        self._ref_datum_parameters: Optional[Dict[str, Any]] = None
        self._ref_commodity_zones: Optional[Dict[str, Any]] = None

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthStatus] = None

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Metrics counters
        self._metrics: Dict[str, int] = {
            "coordinates_parsed": 0,
            "coordinates_validated": 0,
            "coordinates_transformed": 0,
            "plausibility_checks": 0,
            "accuracy_assessments": 0,
            "reverse_geocodes": 0,
            "compliance_certs": 0,
            "batch_jobs": 0,
            "errors": 0,
        }

        logger.info(
            "GPSCoordinateValidatorService created: config_hash=%s, "
            "batch_max=%d, concurrency=%d, cache_ttl=%ds",
            self._config_hash[:12],
            self._batch_max_size,
            self._batch_concurrency,
            self._cache_ttl_seconds,
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
    def coordinate_parser(self) -> Any:
        """Return the CoordinateParser engine instance."""
        self._ensure_started()
        return self._coordinate_parser

    @property
    def datum_transformer(self) -> Any:
        """Return the DatumTransformer engine instance."""
        self._ensure_started()
        return self._datum_transformer

    @property
    def precision_analyzer(self) -> Any:
        """Return the PrecisionAnalyzer engine instance."""
        self._ensure_started()
        return self._precision_analyzer

    @property
    def format_validator(self) -> Any:
        """Return the FormatValidator engine instance."""
        self._ensure_started()
        return self._format_validator

    @property
    def spatial_plausibility_checker(self) -> Any:
        """Return the SpatialPlausibilityChecker engine instance."""
        self._ensure_started()
        return self._spatial_plausibility_checker

    @property
    def reverse_geocoder(self) -> Any:
        """Return the ReverseGeocoder engine instance."""
        self._ensure_started()
        return self._reverse_geocoder

    @property
    def accuracy_assessor(self) -> Any:
        """Return the AccuracyAssessor engine instance."""
        self._ensure_started()
        return self._accuracy_assessor

    @property
    def compliance_reporter(self) -> Any:
        """Return the ComplianceReporter engine instance."""
        self._ensure_started()
        return self._compliance_reporter

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
            logger.debug("GPSCoordinateValidatorService already started")
            return

        start = time.monotonic()
        logger.info("GPSCoordinateValidatorService starting up...")

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
            "GPSCoordinateValidatorService started in %.1fms: "
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
            logger.debug("GPSCoordinateValidatorService already stopped")
            return

        logger.info("GPSCoordinateValidatorService shutting down...")
        start = time.monotonic()

        self._stop_health_check()
        await self._close_engines()
        await self._close_redis()
        await self._close_database()
        self._flush_metrics()

        self._started = False
        elapsed = (time.monotonic() - start) * 1000
        logger.info("GPSCoordinateValidatorService shut down in %.1fms", elapsed)

    # ==================================================================
    # FACADE METHODS: Parsing
    # ==================================================================

    def parse_coordinate(
        self,
        raw_input: str,
        format_hint: Optional[str] = None,
        datum_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse a raw coordinate string into normalized decimal degrees.

        Orchestrates: CoordinateParser -> DatumTransformer -> PrecisionAnalyzer.

        Args:
            raw_input: Raw coordinate string in any supported format.
            format_hint: Optional format hint (e.g. 'dms', 'utm').
            datum_hint: Optional datum hint (e.g. 'NAD27', 'ED50').

        Returns:
            Dictionary with parsed coordinate, format, datum, precision.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If the coordinate cannot be parsed.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug("Parsing coordinate: input=%s, hint=%s", raw_input[:50], format_hint)

        try:
            parsed = self._safe_parse(raw_input, format_hint, datum_hint)
            precision = self._safe_analyze_precision(parsed.get("lat", 0.0), parsed.get("lon", 0.0))
            elapsed_ms = (time.monotonic() - start) * 1000

            result = CoordinateResult(
                request_id=request_id,
                lat=parsed.get("lat", 0.0),
                lon=parsed.get("lon", 0.0),
                original_lat=parsed.get("original_lat", parsed.get("lat", 0.0)),
                original_lon=parsed.get("original_lon", parsed.get("lon", 0.0)),
                input_format=parsed.get("format", format_hint or "decimal_degrees"),
                source_datum=parsed.get("datum", datum_hint or "WGS84"),
                target_datum="WGS84",
                precision_level=precision.get("level", "unknown"),
                decimal_places=precision.get("decimal_places", 0),
                ground_resolution_m=precision.get("ground_resolution_m", 0.0),
                provenance_hash=_compute_provenance_hash(
                    request_id, raw_input, str(parsed.get("lat", 0.0)),
                    str(parsed.get("lon", 0.0)),
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["coordinates_parsed"] += 1
            logger.info(
                "Coordinate parsed: id=%s, lat=%.6f, lon=%.6f, "
                "format=%s, datum=%s, elapsed=%.1fms",
                request_id, result.lat, result.lon,
                result.input_format, result.source_datum, elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("Parse failed: input=%s, error=%s", raw_input[:50], exc, exc_info=True)
            raise

    def batch_parse(self, raw_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch parse multiple raw coordinate strings.

        Args:
            raw_inputs: List of dicts with 'raw_input', optional 'format_hint',
                optional 'datum_hint'.

        Returns:
            Batch result dictionary with per-item results.
        """
        self._ensure_started()
        return self._run_batch("batch_parse", raw_inputs, self._parse_single_item)

    def detect_format(self, raw_input: str) -> Dict[str, Any]:
        """Detect the coordinate format of a raw input string.

        Args:
            raw_input: Raw coordinate string.

        Returns:
            Dictionary with detected format, confidence, alternatives.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        detected = self._safe_detect_format(raw_input)
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "detected_format": detected.get("format", "unknown"),
            "confidence": detected.get("confidence", 0.0),
            "alternatives": detected.get("alternatives", []),
            "provenance_hash": _compute_provenance_hash(request_id, raw_input),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def normalize(self, raw_input: str) -> Dict[str, Any]:
        """Parse and normalize a coordinate to WGS84 decimal degrees.

        Combines parse + datum transformation to WGS84 DD.

        Args:
            raw_input: Raw coordinate string in any supported format.

        Returns:
            Dictionary with normalized WGS84 DD coordinate.
        """
        self._ensure_started()
        return self.parse_coordinate(raw_input)

    # ==================================================================
    # FACADE METHODS: Validation
    # ==================================================================

    def validate_coordinate(
        self,
        lat: float,
        lon: float,
        country_iso: Optional[str] = None,
        commodity: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a coordinate through the full pipeline.

        Orchestrates: Parse -> Transform -> Validate -> Precision ->
        Plausibility -> Score -> Return.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            country_iso: Expected ISO country code.
            commodity: EUDR commodity for plausibility check.
            source_type: GPS data source type.

        Returns:
            Dictionary with comprehensive validation results.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Validating coordinate: lat=%.6f, lon=%.6f, country=%s, commodity=%s",
            lat, lon, country_iso, commodity,
        )

        try:
            errors: List[str] = []
            warnings: List[str] = []
            error_types: List[str] = []
            checks_performed = 0
            checks_passed = 0
            checks_failed = 0

            # Range check
            range_result = self._check_range_internal(lat, lon)
            checks_performed += 1
            if range_result["is_valid"]:
                checks_passed += 1
            else:
                checks_failed += 1
                errors.extend(range_result.get("errors", []))
                error_types.extend(range_result.get("error_types", []))

            # NaN/Inf check
            nan_result = self._check_nan_inf(lat, lon)
            checks_performed += 1
            if nan_result["is_valid"]:
                checks_passed += 1
            else:
                checks_failed += 1
                errors.extend(nan_result.get("errors", []))
                error_types.extend(nan_result.get("error_types", []))

            # Null island check
            null_result = self._check_null_island(lat, lon)
            checks_performed += 1
            if null_result["is_valid"]:
                checks_passed += 1
            else:
                checks_failed += 1
                warnings.extend(null_result.get("warnings", []))
                error_types.extend(null_result.get("error_types", []))

            # Swap detection
            if country_iso:
                swap_result = self._detect_swap_internal(lat, lon, country_iso)
                checks_performed += 1
                if swap_result["is_valid"]:
                    checks_passed += 1
                else:
                    checks_failed += 1
                    warnings.extend(swap_result.get("warnings", []))
                    error_types.extend(swap_result.get("error_types", []))

            # Precision check
            precision = self._safe_analyze_precision(lat, lon)
            checks_performed += 1
            if precision.get("eudr_adequate", False):
                checks_passed += 1
            else:
                checks_failed += 1
                warnings.append(
                    f"Precision level '{precision.get('level', 'unknown')}' "
                    f"may not meet EUDR requirements"
                )

            # Plausibility checks (if country/commodity provided)
            if country_iso or commodity:
                plaus = self._check_plausibility_internal(lat, lon, commodity, country_iso)
                checks_performed += 1
                if plaus.get("is_plausible", True):
                    checks_passed += 1
                else:
                    checks_failed += 1
                    warnings.extend(plaus.get("findings", []))

            # Format validator engine delegation
            fmt_result = self._safe_format_validate(lat, lon)
            checks_performed += 1
            if fmt_result.get("is_valid", True):
                checks_passed += 1
            else:
                checks_failed += 1
                errors.extend(fmt_result.get("errors", []))

            is_valid = len(errors) == 0
            elapsed_ms = (time.monotonic() - start) * 1000

            result = ValidationResult(
                request_id=request_id,
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                error_types=error_types,
                provenance_hash=_compute_provenance_hash(
                    request_id, str(lat), str(lon),
                    str(is_valid), str(checks_performed),
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["coordinates_validated"] += 1
            logger.info(
                "Coordinate validated: id=%s, valid=%s, checks=%d/%d, "
                "errors=%d, warnings=%d, elapsed=%.1fms",
                request_id, is_valid, checks_passed, checks_performed,
                len(errors), len(warnings), elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Validation failed: lat=%.6f, lon=%.6f, error=%s",
                lat, lon, exc, exc_info=True,
            )
            raise

    def batch_validate(self, coordinates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch validate multiple coordinates.

        Args:
            coordinates: List of dicts with 'lat', 'lon', optional
                'country_iso', 'commodity', 'source_type'.

        Returns:
            Batch result dictionary with per-item results.
        """
        self._ensure_started()
        return self._run_batch("batch_validate", coordinates, self._validate_single_item)

    def check_range(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check whether lat/lon values are within WGS84 valid range.

        Args:
            lat: Latitude value.
            lon: Longitude value.

        Returns:
            Dictionary with range check result.
        """
        self._ensure_started()
        start = time.monotonic()
        result = self._check_range_internal(lat, lon)
        result["processing_time_ms"] = round((time.monotonic() - start) * 1000, 2)
        result["provenance_hash"] = _compute_provenance_hash(str(lat), str(lon), "range_check")
        return result

    def detect_swaps(self, lat: float, lon: float, country_iso: str) -> Dict[str, Any]:
        """Detect whether latitude and longitude may be swapped.

        Args:
            lat: Declared latitude.
            lon: Declared longitude.
            country_iso: Expected country ISO code for context.

        Returns:
            Dictionary with swap detection result.
        """
        self._ensure_started()
        start = time.monotonic()
        result = self._detect_swap_internal(lat, lon, country_iso)
        result["processing_time_ms"] = round((time.monotonic() - start) * 1000, 2)
        result["provenance_hash"] = _compute_provenance_hash(
            str(lat), str(lon), country_iso, "swap_detect",
        )
        return result

    def detect_duplicates(self, coordinates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect duplicate and near-duplicate coordinates in a list.

        Args:
            coordinates: List of dicts with 'lat' and 'lon' keys.

        Returns:
            Dictionary with duplicate detection results.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        duplicates: List[Dict[str, Any]] = []
        threshold_m = self._duplicate_threshold_m
        threshold_km = threshold_m / 1000.0

        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                lat_i = coordinates[i].get("lat", 0.0)
                lon_i = coordinates[i].get("lon", 0.0)
                lat_j = coordinates[j].get("lat", 0.0)
                lon_j = coordinates[j].get("lon", 0.0)

                if lat_i == lat_j and lon_i == lon_j:
                    duplicates.append({
                        "index_a": i, "index_b": j,
                        "type": "exact", "distance_m": 0.0,
                    })
                else:
                    dist = _haversine_km(lat_i, lon_i, lat_j, lon_j)
                    if dist <= threshold_km:
                        duplicates.append({
                            "index_a": i, "index_b": j,
                            "type": "near_duplicate",
                            "distance_m": round(dist * 1000.0, 3),
                        })

        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "request_id": request_id,
            "total_coordinates": len(coordinates),
            "duplicates_found": len(duplicates),
            "duplicates": duplicates,
            "threshold_m": threshold_m,
            "provenance_hash": _compute_provenance_hash(
                request_id, str(len(coordinates)), str(len(duplicates)),
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ==================================================================
    # FACADE METHODS: Plausibility
    # ==================================================================

    def check_plausibility(
        self,
        lat: float,
        lon: float,
        commodity: Optional[str] = None,
        country_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full spatial plausibility checks on a coordinate.

        Args:
            lat: Latitude in decimal degrees WGS84.
            lon: Longitude in decimal degrees WGS84.
            commodity: EUDR commodity for plausibility.
            country_iso: Declared country ISO code.

        Returns:
            Dictionary with plausibility results.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        plaus = self._check_plausibility_internal(lat, lon, commodity, country_iso)
        elapsed_ms = (time.monotonic() - start) * 1000

        result = PlausibilityResult(
            request_id=request_id,
            is_plausible=plaus.get("is_plausible", True),
            is_on_land=plaus.get("is_on_land", True),
            detected_country=plaus.get("detected_country", ""),
            declared_country_match=plaus.get("declared_country_match", True),
            commodity_plausible=plaus.get("commodity_plausible", True),
            elevation_plausible=plaus.get("elevation_plausible", True),
            is_urban=plaus.get("is_urban", False),
            findings=plaus.get("findings", []),
            provenance_hash=_compute_provenance_hash(
                request_id, str(lat), str(lon),
                commodity or "", country_iso or "",
            ),
            processing_time_ms=elapsed_ms,
        )

        self._metrics["plausibility_checks"] += 1
        logger.info(
            "Plausibility check: id=%s, plausible=%s, country=%s, elapsed=%.1fms",
            request_id, result.is_plausible, result.detected_country, elapsed_ms,
        )
        return result.to_dict()

    def check_land_ocean(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check whether a coordinate is on land or in ocean.

        Args:
            lat: Latitude in decimal degrees WGS84.
            lon: Longitude in decimal degrees WGS84.

        Returns:
            Dictionary with land/ocean classification.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.country_boundaries import (
            find_country, is_ocean,
        )

        country = find_country(lat, lon)
        on_ocean = is_ocean(lat, lon)
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "lat": round(lat, 8),
            "lon": round(lon, 8),
            "is_on_land": not on_ocean,
            "is_ocean": on_ocean,
            "detected_country": country or "",
            "provenance_hash": _compute_provenance_hash(
                request_id, str(lat), str(lon), "land_ocean",
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def check_country(
        self, lat: float, lon: float, declared_country: str,
    ) -> Dict[str, Any]:
        """Check whether a coordinate matches the declared country.

        Args:
            lat: Latitude in decimal degrees WGS84.
            lon: Longitude in decimal degrees WGS84.
            declared_country: Expected ISO 3166-1 alpha-2 country code.

        Returns:
            Dictionary with country match result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.country_boundaries import (
            find_country, get_country,
        )

        detected = find_country(lat, lon)
        matches = (detected is not None
                   and detected.upper() == declared_country.upper())
        country_data = get_country(declared_country)
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "declared_country": declared_country.upper(),
            "detected_country": detected or "",
            "match": matches,
            "declared_country_name": country_data["name"] if country_data else "",
            "provenance_hash": _compute_provenance_hash(
                request_id, str(lat), str(lon), declared_country,
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def check_commodity(
        self, lat: float, lon: float, commodity: str,
    ) -> Dict[str, Any]:
        """Check whether a commodity is plausible at the given location.

        Args:
            lat: Latitude in decimal degrees WGS84.
            lon: Longitude in decimal degrees WGS84.
            commodity: EUDR commodity key.

        Returns:
            Dictionary with commodity plausibility result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.commodity_zones import (
            is_commodity_plausible, is_urban, get_commodity_zones,
        )

        plausible = is_commodity_plausible(lat, lon, commodity)
        urban = is_urban(lat, lon)
        zone_data = get_commodity_zones(commodity)
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "commodity": commodity,
            "is_plausible": plausible,
            "is_urban": urban,
            "latitude_range": zone_data["extended_range"] if zone_data else [],
            "elevation_range_m": zone_data["elevation_range_m"] if zone_data else [],
            "provenance_hash": _compute_provenance_hash(
                request_id, str(lat), str(lon), commodity,
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def check_elevation(
        self,
        lat: float,
        lon: float,
        commodity: str,
        elevation: float,
    ) -> Dict[str, Any]:
        """Check whether elevation is plausible for a commodity.

        Args:
            lat: Latitude in decimal degrees WGS84.
            lon: Longitude in decimal degrees WGS84.
            commodity: EUDR commodity key.
            elevation: Elevation in metres above sea level.

        Returns:
            Dictionary with elevation plausibility result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.commodity_zones import (
            get_elevation_range,
        )

        elev_range = get_elevation_range(commodity)
        plausible = True
        if elev_range:
            plausible = elev_range[0] <= elevation <= elev_range[1]

        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "request_id": request_id,
            "commodity": commodity,
            "elevation_m": elevation,
            "elevation_range_m": list(elev_range) if elev_range else [],
            "is_plausible": plausible,
            "provenance_hash": _compute_provenance_hash(
                request_id, str(lat), str(lon), commodity, str(elevation),
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ==================================================================
    # FACADE METHODS: Assessment
    # ==================================================================

    def assess_accuracy(
        self,
        lat: float,
        lon: float,
        source_type: Optional[str] = None,
        commodity: Optional[str] = None,
        country_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full accuracy assessment pipeline.

        Orchestrates: validate + precision + plausibility + score.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            source_type: GPS data source type.
            commodity: EUDR commodity.
            country_iso: Country ISO code.

        Returns:
            Dictionary with accuracy assessment results.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Assessing accuracy: lat=%.6f, lon=%.6f, source=%s",
            lat, lon, source_type,
        )

        try:
            # Validation score
            validation = self.validate_coordinate(lat, lon, country_iso, commodity, source_type)
            val_score = 100.0 if validation.get("is_valid", False) else 0.0
            if validation.get("warnings"):
                val_score = max(0.0, val_score - len(validation["warnings"]) * 10.0)

            # Precision score
            precision = self._safe_analyze_precision(lat, lon)
            prec_score = self._compute_precision_score(precision)

            # Plausibility score
            plaus_score = 100.0
            if commodity or country_iso:
                plaus = self._check_plausibility_internal(lat, lon, commodity, country_iso)
                plaus_score = 100.0 if plaus.get("is_plausible", True) else 30.0
                if plaus.get("is_urban", False):
                    plaus_score = max(0.0, plaus_score - 20.0)

            # Overall score (weighted)
            overall = (val_score * 0.4 + prec_score * 0.35 + plaus_score * 0.25)
            grade = self._score_to_grade(overall)
            eudr_compliant = overall >= 70.0 and validation.get("is_valid", False)

            recommendations = self._generate_recommendations(
                val_score, prec_score, plaus_score, precision, source_type,
            )

            elapsed_ms = (time.monotonic() - start) * 1000

            result = AccuracyResult(
                request_id=request_id,
                overall_score=overall,
                grade=grade,
                eudr_compliant=eudr_compliant,
                precision_score=prec_score,
                validation_score=val_score,
                plausibility_score=plaus_score,
                source_type=source_type or "unknown",
                recommendations=recommendations,
                provenance_hash=_compute_provenance_hash(
                    request_id, str(lat), str(lon),
                    str(overall), grade,
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["accuracy_assessments"] += 1
            logger.info(
                "Accuracy assessed: id=%s, score=%.1f, grade=%s, "
                "eudr=%s, elapsed=%.1fms",
                request_id, overall, grade, eudr_compliant, elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Accuracy assessment failed: lat=%.6f, lon=%.6f, error=%s",
                lat, lon, exc, exc_info=True,
            )
            raise

    def batch_assess(self, coordinates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch accuracy assessment for multiple coordinates.

        Args:
            coordinates: List of dicts with 'lat', 'lon', optional fields.

        Returns:
            Batch result dictionary.
        """
        self._ensure_started()
        return self._run_batch("batch_assess", coordinates, self._assess_single_item)

    def assess_precision(self, lat: float, lon: float) -> Dict[str, Any]:
        """Assess the precision of a coordinate based on decimal places.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            Dictionary with precision assessment.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        precision = self._safe_analyze_precision(lat, lon)
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "request_id": request_id,
            "lat": round(lat, 8),
            "lon": round(lon, 8),
            "precision_level": precision.get("level", "unknown"),
            "decimal_places": precision.get("decimal_places", 0),
            "ground_resolution_m": round(precision.get("ground_resolution_m", 0.0), 3),
            "eudr_adequate": precision.get("eudr_adequate", False),
            "provenance_hash": _compute_provenance_hash(
                request_id, str(lat), str(lon), "precision",
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ==================================================================
    # FACADE METHODS: Reverse Geocoding
    # ==================================================================

    def reverse_geocode(self, lat: float, lon: float) -> Dict[str, Any]:
        """Reverse geocode a coordinate to country and region.

        Args:
            lat: Latitude in decimal degrees WGS84.
            lon: Longitude in decimal degrees WGS84.

        Returns:
            Dictionary with reverse geocode result.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.country_boundaries import (
            find_country, get_country, is_ocean,
        )

        country_iso = find_country(lat, lon)
        country_data = get_country(country_iso) if country_iso else None
        on_ocean = is_ocean(lat, lon)
        elapsed_ms = (time.monotonic() - start) * 1000

        result = GeocodeResult(
            request_id=request_id,
            country_iso=country_iso or "",
            country_name=country_data["name"] if country_data else "",
            is_on_land=not on_ocean,
            region="",
            provenance_hash=_compute_provenance_hash(
                request_id, str(lat), str(lon), country_iso or "ocean",
            ),
            processing_time_ms=elapsed_ms,
        )

        self._metrics["reverse_geocodes"] += 1
        logger.info(
            "Reverse geocode: id=%s, country=%s, land=%s, elapsed=%.1fms",
            request_id, result.country_iso, result.is_on_land, elapsed_ms,
        )
        return result.to_dict()

    def batch_reverse_geocode(self, coordinates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch reverse geocode multiple coordinates.

        Args:
            coordinates: List of dicts with 'lat' and 'lon'.

        Returns:
            Batch result dictionary.
        """
        self._ensure_started()
        return self._run_batch(
            "batch_reverse_geocode", coordinates, self._geocode_single_item,
        )

    def lookup_country(self, lat: float, lon: float) -> Dict[str, Any]:
        """Simple country lookup for a coordinate.

        Args:
            lat: Latitude in decimal degrees WGS84.
            lon: Longitude in decimal degrees WGS84.

        Returns:
            Dictionary with country ISO code and name.
        """
        self._ensure_started()
        return self.reverse_geocode(lat, lon)

    # ==================================================================
    # FACADE METHODS: Datum Transformation
    # ==================================================================

    def transform_datum(
        self,
        lat: float,
        lon: float,
        source_datum: str,
        target_datum: str = "WGS84",
    ) -> Dict[str, Any]:
        """Transform a coordinate from one datum to another.

        Args:
            lat: Latitude in source datum.
            lon: Longitude in source datum.
            source_datum: Source datum key (e.g. 'NAD27').
            target_datum: Target datum key (default 'WGS84').

        Returns:
            Dictionary with transformed coordinate.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Transforming datum: lat=%.6f, lon=%.6f, %s -> %s",
            lat, lon, source_datum, target_datum,
        )

        try:
            transformed = self._safe_datum_transform(lat, lon, source_datum, target_datum)
            elapsed_ms = (time.monotonic() - start) * 1000

            result = CoordinateResult(
                request_id=request_id,
                lat=transformed.get("lat", lat),
                lon=transformed.get("lon", lon),
                original_lat=lat,
                original_lon=lon,
                source_datum=source_datum,
                target_datum=target_datum,
                provenance_hash=_compute_provenance_hash(
                    request_id, str(lat), str(lon),
                    source_datum, target_datum,
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["coordinates_transformed"] += 1
            logger.info(
                "Datum transformed: id=%s, %s->%s, "
                "in=(%.6f,%.6f), out=(%.6f,%.6f), elapsed=%.1fms",
                request_id, source_datum, target_datum,
                lat, lon, result.lat, result.lon, elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Datum transform failed: %s->%s, error=%s",
                source_datum, target_datum, exc, exc_info=True,
            )
            raise

    def batch_transform(
        self,
        coordinates: List[Dict[str, Any]],
        source_datum: str,
    ) -> Dict[str, Any]:
        """Batch transform coordinates from a source datum to WGS84.

        Args:
            coordinates: List of dicts with 'lat' and 'lon'.
            source_datum: Source datum key for all coordinates.

        Returns:
            Batch result dictionary.
        """
        self._ensure_started()
        items = [
            {**c, "source_datum": source_datum, "target_datum": "WGS84"}
            for c in coordinates
        ]
        return self._run_batch("batch_transform", items, self._transform_single_item)

    def list_datums(self) -> Dict[str, Any]:
        """List all supported geodetic datums.

        Returns:
            Dictionary with list of datum keys and metadata.
        """
        self._ensure_started()

        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.datum_parameters import (
            DATUM_PARAMETERS, TOTAL_DATUMS,
        )

        datums = []
        for key, params in DATUM_PARAMETERS.items():
            datums.append({
                "key": key,
                "name": params["name"],
                "ellipsoid": params["ellipsoid"],
                "region": params["region"],
                "accuracy_m": params["accuracy_m"],
            })

        return {
            "total_datums": TOTAL_DATUMS,
            "datums": datums,
        }

    # ==================================================================
    # FACADE METHODS: Reporting
    # ==================================================================

    def generate_compliance_cert(
        self,
        lat: float,
        lon: float,
        commodity: str,
        country_iso: str,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an EUDR compliance certificate for a coordinate.

        Runs the full assessment pipeline and produces a compliance
        certificate with pass/fail determination.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            commodity: EUDR commodity.
            country_iso: Country ISO code.
            source_type: GPS data source type.

        Returns:
            Dictionary with compliance certificate.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.info(
            "Generating compliance cert: lat=%.6f, lon=%.6f, "
            "commodity=%s, country=%s",
            lat, lon, commodity, country_iso,
        )

        try:
            assessment = self.assess_accuracy(lat, lon, source_type, commodity, country_iso)
            validation = self.validate_coordinate(lat, lon, country_iso, commodity, source_type)
            plausibility = self.check_plausibility(lat, lon, commodity, country_iso)

            val_passed = validation.get("is_valid", False)
            plaus_passed = plausibility.get("is_plausible", True)
            precision = self._safe_analyze_precision(lat, lon)
            prec_adequate = precision.get("eudr_adequate", False)
            is_compliant = val_passed and plaus_passed and prec_adequate

            findings: List[str] = []
            if not val_passed:
                findings.append("Coordinate validation failed")
                findings.extend(validation.get("errors", []))
            if not plaus_passed:
                findings.append("Spatial plausibility check failed")
                findings.extend(plausibility.get("findings", []))
            if not prec_adequate:
                findings.append(
                    f"Precision level '{precision.get('level', 'unknown')}' "
                    f"is insufficient for EUDR compliance"
                )
            if is_compliant:
                findings.append("Coordinate meets EUDR compliance requirements")

            elapsed_ms = (time.monotonic() - start) * 1000

            result = ComplianceCertResult(
                request_id=request_id,
                is_compliant=is_compliant,
                lat=lat,
                lon=lon,
                commodity=commodity,
                country_iso=country_iso,
                accuracy_grade=assessment.get("grade", "F"),
                overall_score=assessment.get("overall_score", 0.0),
                validation_passed=val_passed,
                plausibility_passed=plaus_passed,
                precision_adequate=prec_adequate,
                findings=findings,
                provenance_hash=_compute_provenance_hash(
                    request_id, str(lat), str(lon),
                    commodity, country_iso, str(is_compliant),
                ),
                processing_time_ms=elapsed_ms,
            )

            self._metrics["compliance_certs"] += 1
            logger.info(
                "Compliance cert generated: id=%s, cert=%s, compliant=%s, "
                "grade=%s, elapsed=%.1fms",
                request_id, result.certificate_id, is_compliant,
                result.accuracy_grade, elapsed_ms,
            )
            return result.to_dict()

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "Compliance cert failed: lat=%.6f, lon=%.6f, error=%s",
                lat, lon, exc, exc_info=True,
            )
            raise

    def generate_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary report from batch results.

        Args:
            results: List of individual assessment/validation result dicts.

        Returns:
            Summary dictionary with statistics.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        total = len(results)
        valid_count = sum(1 for r in results if r.get("is_valid", r.get("eudr_compliant", False)))
        invalid_count = total - valid_count

        scores = [r.get("overall_score", 0.0) for r in results if "overall_score" in r]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "request_id": request_id,
            "total": total,
            "valid": valid_count,
            "invalid": invalid_count,
            "compliance_rate": round(valid_count / total * 100, 2) if total > 0 else 0.0,
            "average_score": round(avg_score, 2),
            "provenance_hash": _compute_provenance_hash(
                request_id, str(total), str(valid_count),
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def generate_remediation(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate remediation guidance for coordinate errors.

        Args:
            errors: List of error dicts from validation results.

        Returns:
            Dictionary with remediation steps for each error type.
        """
        self._ensure_started()
        request_id = _generate_request_id()

        remediation_map = {
            "out_of_range": "Verify coordinate values. Latitude must be -90 to 90, longitude -180 to 180.",
            "swapped": "Latitude and longitude appear swapped. Try reversing the values.",
            "sign_error": "Check hemisphere. Negative latitude = South, negative longitude = West.",
            "null_island": "Coordinate is at (0,0). This is likely a data entry error or missing value.",
            "nan_value": "Coordinate contains non-numeric value. Ensure valid numbers.",
            "inf_value": "Coordinate contains infinity. Check data source for overflow.",
            "duplicate": "Exact duplicate coordinate found. Verify each plot has a unique location.",
            "near_duplicate": "Near-duplicate within threshold. Verify distinct plot locations.",
            "boundary_value": "Coordinate is at an exact boundary value (e.g. whole degree). Verify precision.",
        }

        remediations: List[Dict[str, str]] = []
        for error in errors:
            error_type = error.get("type", error.get("error_type", "unknown"))
            remediations.append({
                "error_type": error_type,
                "guidance": remediation_map.get(error_type, "Review the coordinate data source."),
            })

        return {
            "request_id": request_id,
            "total_errors": len(errors),
            "remediations": remediations,
            "provenance_hash": _compute_provenance_hash(
                request_id, str(len(errors)),
            ),
        }

    def submission_readiness(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess batch readiness for EUDR submission.

        Args:
            results: List of compliance cert or assessment result dicts.

        Returns:
            Dictionary with readiness assessment.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        total = len(results)
        compliant = sum(
            1 for r in results
            if r.get("is_compliant", r.get("eudr_compliant", False))
        )
        non_compliant = total - compliant
        compliance_rate = (compliant / total * 100.0) if total > 0 else 0.0
        is_ready = compliance_rate >= 100.0

        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "request_id": request_id,
            "is_ready": is_ready,
            "total_coordinates": total,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "compliance_rate_pct": round(compliance_rate, 2),
            "blocking_issues": non_compliant,
            "recommendation": (
                "All coordinates meet EUDR requirements. Ready for submission."
                if is_ready
                else f"{non_compliant} coordinate(s) require remediation before submission."
            ),
            "provenance_hash": _compute_provenance_hash(
                request_id, str(total), str(compliant), str(is_ready),
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
            timestamp=_utcnow(),
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
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "timestamp": _utcnow().isoformat(),
        }

    # ==================================================================
    # Internal: Validation helpers
    # ==================================================================

    def _check_range_internal(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check whether lat/lon values are within WGS84 valid range."""
        errors: List[str] = []
        error_types: List[str] = []
        is_valid = True

        if not (-90.0 <= lat <= 90.0):
            is_valid = False
            errors.append(f"Latitude {lat} out of range [-90, 90]")
            error_types.append("out_of_range")
        if not (-180.0 <= lon <= 180.0):
            is_valid = False
            errors.append(f"Longitude {lon} out of range [-180, 180]")
            error_types.append("out_of_range")

        return {
            "is_valid": is_valid,
            "errors": errors,
            "error_types": error_types,
        }

    def _check_nan_inf(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check for NaN and infinity values."""
        errors: List[str] = []
        error_types: List[str] = []
        is_valid = True

        if math.isnan(lat) or math.isnan(lon):
            is_valid = False
            errors.append("Coordinate contains NaN value")
            error_types.append("nan_value")
        if math.isinf(lat) or math.isinf(lon):
            is_valid = False
            errors.append("Coordinate contains infinity value")
            error_types.append("inf_value")

        return {
            "is_valid": is_valid,
            "errors": errors,
            "error_types": error_types,
        }

    def _check_null_island(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check for null island (0,0) coordinates."""
        warnings: List[str] = []
        error_types: List[str] = []
        is_valid = True

        dist = _haversine_km(lat, lon, 0.0, 0.0)
        if dist <= self._null_island_radius_km:
            is_valid = False
            warnings.append(
                f"Coordinate ({lat}, {lon}) is within "
                f"{self._null_island_radius_km}km of Null Island (0,0)"
            )
            error_types.append("null_island")

        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "error_types": error_types,
        }

    def _detect_swap_internal(
        self, lat: float, lon: float, country_iso: str,
    ) -> Dict[str, Any]:
        """Detect whether lat/lon may be swapped by checking country bbox."""
        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.country_boundaries import (
            get_country,
        )

        warnings: List[str] = []
        error_types: List[str] = []
        is_valid = True

        country = get_country(country_iso)
        if country:
            bbox = country["bbox"]
            # Check if original is outside but swapped is inside
            original_in = (
                bbox["min_lat"] <= lat <= bbox["max_lat"]
                and bbox["min_lon"] <= lon <= bbox["max_lon"]
            )
            swapped_in = (
                bbox["min_lat"] <= lon <= bbox["max_lat"]
                and bbox["min_lon"] <= lat <= bbox["max_lon"]
            )
            if not original_in and swapped_in:
                is_valid = False
                warnings.append(
                    f"Latitude ({lat}) and longitude ({lon}) appear swapped "
                    f"for country {country_iso}"
                )
                error_types.append("swapped")

        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "error_types": error_types,
            "likely_swapped": not is_valid,
        }

    def _check_plausibility_internal(
        self,
        lat: float,
        lon: float,
        commodity: Optional[str],
        country_iso: Optional[str],
    ) -> Dict[str, Any]:
        """Internal plausibility check combining country, commodity, urban."""
        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.country_boundaries import (
            find_country, is_ocean,
        )
        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.commodity_zones import (
            is_commodity_plausible, is_urban as check_urban,
        )

        findings: List[str] = []
        is_plausible = True

        # Land/ocean
        on_ocean = is_ocean(lat, lon)
        if on_ocean:
            is_plausible = False
            findings.append("Coordinate is in ocean, not on land")

        # Country match
        detected = find_country(lat, lon)
        country_match = True
        if country_iso and detected:
            if detected.upper() != country_iso.upper():
                country_match = False
                findings.append(
                    f"Detected country {detected} does not match "
                    f"declared country {country_iso}"
                )

        # Commodity plausibility
        commodity_plaus = True
        if commodity:
            commodity_plaus = is_commodity_plausible(lat, lon, commodity)
            if not commodity_plaus:
                findings.append(
                    f"Commodity '{commodity}' is not plausible at "
                    f"({lat:.4f}, {lon:.4f})"
                )

        # Urban check
        urban = check_urban(lat, lon)
        if urban and commodity and commodity.lower() not in ("wood",):
            findings.append("Coordinate is in a major urban area")

        if findings and not on_ocean:
            # Only mark implausible if there are non-land findings
            is_plausible = country_match and commodity_plaus

        return {
            "is_plausible": is_plausible,
            "is_on_land": not on_ocean,
            "detected_country": detected or "",
            "declared_country_match": country_match,
            "commodity_plausible": commodity_plaus,
            "elevation_plausible": True,
            "is_urban": urban,
            "findings": findings,
        }

    # ==================================================================
    # Internal: Precision helpers
    # ==================================================================

    def _safe_analyze_precision(self, lat: float, lon: float) -> Dict[str, Any]:
        """Analyze coordinate precision based on decimal places.

        Returns dict with level, decimal_places, ground_resolution_m, eudr_adequate.
        """
        # Delegate to engine if available
        if self._precision_analyzer is not None:
            try:
                raw = self._precision_analyzer.analyze(lat=lat, lon=lon)
                return {
                    "level": getattr(raw, "level", "unknown"),
                    "decimal_places": getattr(raw, "decimal_places", 0),
                    "ground_resolution_m": getattr(raw, "ground_resolution_m", 0.0),
                    "eudr_adequate": getattr(raw, "eudr_adequate", False),
                }
            except Exception as exc:
                logger.warning("PrecisionAnalyzer.analyze failed: %s", exc)

        # Fallback: built-in precision estimation
        lat_str = f"{lat:.15g}"
        lon_str = f"{lon:.15g}"
        lat_dp = len(lat_str.split(".")[-1]) if "." in lat_str else 0
        lon_dp = len(lon_str.split(".")[-1]) if "." in lon_str else 0
        dp = min(lat_dp, lon_dp)

        # Ground resolution approximation at equator
        # 1 degree ~ 111,320m, each decimal place divides by 10
        resolution_m = 111320.0 / (10.0 ** dp) if dp > 0 else 111320.0

        if dp >= 7:
            level = "survey_grade"
        elif dp >= 5:
            level = "high"
        elif dp >= 3:
            level = "moderate"
        elif dp >= 1:
            level = "low"
        else:
            level = "inadequate"

        eudr_adequate = dp >= 4  # ~11m resolution or better

        return {
            "level": level,
            "decimal_places": dp,
            "ground_resolution_m": resolution_m,
            "eudr_adequate": eudr_adequate,
        }

    def _compute_precision_score(self, precision: Dict[str, Any]) -> float:
        """Convert precision data into a 0-100 score."""
        level = precision.get("level", "inadequate")
        score_map = {
            "survey_grade": 100.0,
            "high": 90.0,
            "moderate": 65.0,
            "low": 30.0,
            "inadequate": 10.0,
        }
        return score_map.get(level, 0.0)

    def _score_to_grade(self, score: float) -> str:
        """Convert a numeric score to a letter grade."""
        if score >= 90.0:
            return "A"
        if score >= 80.0:
            return "B"
        if score >= 70.0:
            return "C"
        if score >= 60.0:
            return "D"
        return "F"

    def _generate_recommendations(
        self,
        val_score: float,
        prec_score: float,
        plaus_score: float,
        precision: Dict[str, Any],
        source_type: Optional[str],
    ) -> List[str]:
        """Generate improvement recommendations based on scores."""
        recs: List[str] = []
        if val_score < 100.0:
            recs.append("Fix validation errors before submission")
        if prec_score < 65.0:
            recs.append(
                f"Increase coordinate precision (currently {precision.get('decimal_places', 0)} "
                f"decimal places). EUDR requires at least 4."
            )
        if plaus_score < 100.0:
            recs.append("Review coordinate location for plausibility issues")
        if source_type in ("manual_entry", "digitized_map"):
            recs.append("Consider using GPS or satellite-derived coordinates for higher accuracy")
        return recs

    # ==================================================================
    # Internal: Engine delegation (safe wrappers)
    # ==================================================================

    def _safe_parse(
        self,
        raw_input: str,
        format_hint: Optional[str],
        datum_hint: Optional[str],
    ) -> Dict[str, Any]:
        """Delegate to CoordinateParser with fallback."""
        if self._coordinate_parser is not None:
            try:
                raw = self._coordinate_parser.parse(
                    raw_input=raw_input,
                    format_hint=format_hint,
                    datum_hint=datum_hint,
                )
                return {
                    "lat": getattr(raw, "lat", 0.0),
                    "lon": getattr(raw, "lon", 0.0),
                    "original_lat": getattr(raw, "original_lat", 0.0),
                    "original_lon": getattr(raw, "original_lon", 0.0),
                    "format": getattr(raw, "format", "decimal_degrees"),
                    "datum": getattr(raw, "datum", "WGS84"),
                }
            except Exception as exc:
                logger.warning("CoordinateParser.parse failed: %s", exc)

        # Fallback: try to parse as decimal degrees
        return self._fallback_parse_dd(raw_input)

    def _fallback_parse_dd(self, raw_input: str) -> Dict[str, Any]:
        """Fallback parser for simple decimal degree strings."""
        parts = raw_input.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                return {
                    "lat": lat, "lon": lon,
                    "original_lat": lat, "original_lon": lon,
                    "format": "decimal_degrees", "datum": "WGS84",
                }
            except ValueError:
                pass
        raise ValueError(f"Cannot parse coordinate: {raw_input}")

    def _safe_detect_format(self, raw_input: str) -> Dict[str, Any]:
        """Delegate to CoordinateParser.detect_format with fallback."""
        if self._coordinate_parser is not None:
            try:
                raw = self._coordinate_parser.detect_format(raw_input=raw_input)
                return {
                    "format": getattr(raw, "format", "unknown"),
                    "confidence": getattr(raw, "confidence", 0.0),
                    "alternatives": getattr(raw, "alternatives", []),
                }
            except Exception as exc:
                logger.warning("CoordinateParser.detect_format failed: %s", exc)
        return {"format": "unknown", "confidence": 0.0, "alternatives": []}

    def _safe_format_validate(self, lat: float, lon: float) -> Dict[str, Any]:
        """Delegate to FormatValidator with fallback."""
        if self._format_validator is not None:
            try:
                raw = self._format_validator.validate(lat=lat, lon=lon)
                return {
                    "is_valid": getattr(raw, "is_valid", True),
                    "errors": getattr(raw, "errors", []),
                }
            except Exception as exc:
                logger.warning("FormatValidator.validate failed: %s", exc)
        return {"is_valid": True, "errors": []}

    def _safe_datum_transform(
        self,
        lat: float,
        lon: float,
        source_datum: str,
        target_datum: str,
    ) -> Dict[str, Any]:
        """Delegate to DatumTransformer with fallback Helmert 3-param."""
        if self._datum_transformer is not None:
            try:
                raw = self._datum_transformer.transform(
                    lat=lat, lon=lon,
                    source_datum=source_datum,
                    target_datum=target_datum,
                )
                return {
                    "lat": getattr(raw, "lat", lat),
                    "lon": getattr(raw, "lon", lon),
                }
            except Exception as exc:
                logger.warning("DatumTransformer.transform failed: %s", exc)

        # Fallback: apply 3-parameter Molodensky approximation
        return self._fallback_molodensky(lat, lon, source_datum, target_datum)

    def _fallback_molodensky(
        self,
        lat: float,
        lon: float,
        source_datum: str,
        target_datum: str,
    ) -> Dict[str, Any]:
        """Fallback Molodensky 3-parameter approximation."""
        from greenlang.agents.eudr.gps_coordinate_validator.reference_data.datum_parameters import (
            get_datum_params, get_ellipsoid_params,
        )

        source_params = get_datum_params(source_datum)
        if source_params is None or target_datum.upper() != "WGS84":
            return {"lat": lat, "lon": lon}

        to_wgs84 = source_params["to_wgs84"]
        dx = to_wgs84["dx"]
        dy = to_wgs84["dy"]
        dz = to_wgs84["dz"]

        if dx == 0 and dy == 0 and dz == 0:
            return {"lat": lat, "lon": lon}

        src_ellipsoid = get_ellipsoid_params(source_params["ellipsoid"])
        wgs84_ellipsoid = get_ellipsoid_params("WGS84")
        if not src_ellipsoid or not wgs84_ellipsoid:
            return {"lat": lat, "lon": lon}

        a = src_ellipsoid["semi_major_axis"]
        f = src_ellipsoid["flattening"]
        da = wgs84_ellipsoid["semi_major_axis"] - a
        df = wgs84_ellipsoid["flattening"] - f

        phi = math.radians(lat)
        lam = math.radians(lon)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)

        e2 = 2 * f - f * f
        rn = a / math.sqrt(1.0 - e2 * sin_phi * sin_phi)
        rm = a * (1.0 - e2) / ((1.0 - e2 * sin_phi * sin_phi) ** 1.5)

        dlat = ((-dx * sin_phi * cos_lam - dy * sin_phi * sin_lam + dz * cos_phi
                 + da * (rn * e2 * sin_phi * cos_phi) / a
                 + df * (rm / (1.0 - f) + rn * (1.0 - f)) * sin_phi * cos_phi)
                / (rm + 0.0))
        dlon = (-dx * sin_lam + dy * cos_lam) / ((rn + 0.0) * cos_phi) if abs(cos_phi) > 1e-10 else 0.0

        new_lat = lat + math.degrees(dlat / rm) if rm != 0 else lat
        new_lon = lon + math.degrees(dlon)

        return {"lat": new_lat, "lon": new_lon}

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

        logger.info("Batch %s: job_id=%s, count=%d", job_type, job_id, len(items))

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
            completed_at=_utcnow(),
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            "Batch %s complete: job_id=%s, total=%d, completed=%d, "
            "failed=%d, elapsed=%.1fms",
            job_type, job_id, len(items), completed, failed, elapsed_ms,
        )
        return batch.to_dict()

    def _parse_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item for batch_parse."""
        return self.parse_coordinate(
            raw_input=item.get("raw_input", ""),
            format_hint=item.get("format_hint"),
            datum_hint=item.get("datum_hint"),
        )

    def _validate_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item for batch_validate."""
        return self.validate_coordinate(
            lat=item.get("lat", 0.0),
            lon=item.get("lon", 0.0),
            country_iso=item.get("country_iso"),
            commodity=item.get("commodity"),
            source_type=item.get("source_type"),
        )

    def _assess_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item for batch_assess."""
        return self.assess_accuracy(
            lat=item.get("lat", 0.0),
            lon=item.get("lon", 0.0),
            source_type=item.get("source_type"),
            commodity=item.get("commodity"),
            country_iso=item.get("country_iso"),
        )

    def _geocode_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item for batch_reverse_geocode."""
        return self.reverse_geocode(
            lat=item.get("lat", 0.0),
            lon=item.get("lon", 0.0),
        )

    def _transform_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item for batch_transform."""
        return self.transform_datum(
            lat=item.get("lat", 0.0),
            lon=item.get("lon", 0.0),
            source_datum=item.get("source_datum", "WGS84"),
            target_datum=item.get("target_datum", "WGS84"),
        )

    # ==================================================================
    # Internal: Infrastructure
    # ==================================================================

    def _ensure_started(self) -> None:
        """Raise RuntimeError if the service has not been started."""
        if not self._started:
            raise RuntimeError(
                "GPSCoordinateValidatorService is not started. "
                "Call startup() first."
            )

    def _configure_logging(self) -> None:
        """Configure structured logging for the service."""
        log_level = getattr(logging, self._log_level, logging.INFO)
        logging.getLogger(
            "greenlang.agents.eudr.gps_coordinate_validator"
        ).setLevel(log_level)
        logger.debug("Logging configured: level=%s", self._log_level)

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            self._tracer = otel_trace.get_tracer(
                "greenlang.agents.eudr.gps_coordinate_validator",
                "1.0.0",
            )
            logger.info("OpenTelemetry tracer initialized")
        else:
            logger.debug("OpenTelemetry not available, tracing disabled")

    def _load_reference_data(self) -> None:
        """Load reference data modules into memory."""
        try:
            from greenlang.agents.eudr.gps_coordinate_validator.reference_data import (
                COUNTRY_BOUNDARIES, DATUM_PARAMETERS, COMMODITY_ZONES,
            )
            self._ref_country_boundaries = COUNTRY_BOUNDARIES
            self._ref_datum_parameters = DATUM_PARAMETERS
            self._ref_commodity_zones = COMMODITY_ZONES
            logger.info(
                "Reference data loaded: countries=%d, datums=%d, commodities=%d",
                len(COUNTRY_BOUNDARIES), len(DATUM_PARAMETERS), len(COMMODITY_ZONES),
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
            logger.warning("Failed to connect to PostgreSQL (non-fatal): %s", exc)
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
            logger.warning("Failed to connect to Redis (non-fatal): %s", exc)
            self._redis = None

    async def _initialize_engines(self) -> None:
        """Initialize all eight internal engines."""
        logger.info("Initializing 8 GPS coordinate validator engines...")

        self._coordinate_parser = await self._init_engine(
            "coordinate_parser",
            "greenlang.agents.eudr.gps_coordinate_validator.coordinate_parser",
            "CoordinateParser",
        )
        self._datum_transformer = await self._init_engine(
            "datum_transformer",
            "greenlang.agents.eudr.gps_coordinate_validator.datum_transformer",
            "DatumTransformer",
        )
        self._precision_analyzer = await self._init_engine(
            "precision_analyzer",
            "greenlang.agents.eudr.gps_coordinate_validator.precision_analyzer",
            "PrecisionAnalyzer",
        )
        self._format_validator = await self._init_engine(
            "format_validator",
            "greenlang.agents.eudr.gps_coordinate_validator.format_validator",
            "FormatValidator",
        )
        self._spatial_plausibility_checker = await self._init_engine(
            "spatial_plausibility_checker",
            "greenlang.agents.eudr.gps_coordinate_validator.spatial_plausibility_checker",
            "SpatialPlausibilityChecker",
        )
        self._reverse_geocoder = await self._init_engine(
            "reverse_geocoder",
            "greenlang.agents.eudr.gps_coordinate_validator.reverse_geocoder",
            "ReverseGeocoder",
        )
        self._accuracy_assessor = await self._init_engine(
            "accuracy_assessor",
            "greenlang.agents.eudr.gps_coordinate_validator.accuracy_assessor",
            "AccuracyAssessor",
        )
        self._compliance_reporter = await self._init_engine(
            "compliance_reporter",
            "greenlang.agents.eudr.gps_coordinate_validator.compliance_reporter",
            "ComplianceReporter",
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
                "Failed to initialize %s: %s", class_name, exc, exc_info=True,
            )
            return None

    async def _close_engines(self) -> None:
        """Close all engine instances."""
        engines = [
            self._coordinate_parser,
            self._datum_transformer,
            self._precision_analyzer,
            self._format_validator,
            self._spatial_plausibility_checker,
            self._reverse_geocoder,
            self._accuracy_assessor,
            self._compliance_reporter,
        ]
        for engine in engines:
            if engine is not None and hasattr(engine, "close"):
                try:
                    result = engine.close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning("Error closing engine: %s", exc)

        self._coordinate_parser = None
        self._datum_transformer = None
        self._precision_analyzer = None
        self._format_validator = None
        self._spatial_plausibility_checker = None
        self._reverse_geocoder = None
        self._accuracy_assessor = None
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
            logger.debug("No running event loop; health check task not started")

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
            "coordinate_parser": self._coordinate_parser,
            "datum_transformer": self._datum_transformer,
            "precision_analyzer": self._precision_analyzer,
            "format_validator": self._format_validator,
            "spatial_plausibility_checker": self._spatial_plausibility_checker,
            "reverse_geocoder": self._reverse_geocoder,
            "accuracy_assessor": self._accuracy_assessor,
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

    def _check_reference_data_health(self) -> Dict[str, Any]:
        """Check reference data availability."""
        loaded = sum(1 for x in [
            self._ref_country_boundaries,
            self._ref_datum_parameters,
            self._ref_commodity_zones,
        ] if x is not None)
        return {
            "status": "healthy" if loaded == 3 else "degraded",
            "loaded_datasets": loaded,
            "total_datasets": 3,
        }

    def _count_initialized_engines(self) -> int:
        """Count the number of successfully initialized engines."""
        engines = [
            self._coordinate_parser,
            self._datum_transformer,
            self._precision_analyzer,
            self._format_validator,
            self._spatial_plausibility_checker,
            self._reverse_geocoder,
            self._accuracy_assessor,
            self._compliance_reporter,
        ]
        return sum(1 for e in engines if e is not None)


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the GPS Coordinate Validator service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown. The service instance is stored in
    ``app.state.gps_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.gps_coordinate_validator.setup import lifespan

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.gps_service``).
    """
    service = get_service()
    app.state.gps_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[GPSCoordinateValidatorService] = None
_service_lock = threading.Lock()


def get_service() -> GPSCoordinateValidatorService:
    """Return the singleton GPSCoordinateValidatorService instance.

    Uses double-checked locking for thread safety. The instance is
    created on first call.

    Returns:
        GPSCoordinateValidatorService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = GPSCoordinateValidatorService()
    return _service_instance


def set_service(service: GPSCoordinateValidatorService) -> None:
    """Replace the singleton GPSCoordinateValidatorService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("GPSCoordinateValidatorService singleton replaced")


def reset_service() -> None:
    """Reset the singleton GPSCoordinateValidatorService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("GPSCoordinateValidatorService singleton reset")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Service
    "GPSCoordinateValidatorService",
    "HealthStatus",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
    # Result containers
    "CoordinateResult",
    "ValidationResult",
    "PlausibilityResult",
    "AccuracyResult",
    "GeocodeResult",
    "ComplianceCertResult",
    "BatchResult",
]
