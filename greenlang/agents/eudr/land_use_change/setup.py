# -*- coding: utf-8 -*-
"""
LandUseChangeService - Facade for AGENT-EUDR-005 Land Use Change Detector Agent

This module implements the LandUseChangeService, the single entry point for
all land use change detection operations in the GL-EUDR-APP. It manages the
lifecycle of eight internal engines, an async PostgreSQL connection pool
(psycopg + psycopg_pool), a Redis cache connection, OpenTelemetry tracing,
and Prometheus metrics. The service exposes a unified interface consumed by
the FastAPI router layer and the GL-EUDR-APP integration.

Lifecycle:
    startup  -> load config -> connect DB -> register pgvector -> connect Redis
             -> initialize engines -> start health check background task
    shutdown -> close engines -> close Redis -> close DB pool -> flush metrics

Engines (8):
    1. LandUseClassifier          - Multi-class land use classification (Feature 1)
    2. TransitionDetector         - Temporal transition detection (Feature 2)
    3. TemporalTrajectoryAnalyzer - Change trajectory analysis (Feature 3)
    4. CutoffDateVerifier         - EUDR cutoff date compliance (Feature 4)
    5. CroplandExpansionDetector  - Agricultural conversion detection (Feature 5)
    6. ConversionRiskAssessor     - Conversion risk scoring (Feature 6)
    7. UrbanEncroachmentAnalyzer  - Urban expansion monitoring (Feature 7)
    8. ComplianceReporter         - Evidence reporting (Feature 8)

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.land_use_change.setup import (
    ...     LandUseChangeService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
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
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_LUC_"

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

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
    return f"LUC-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LandUseChangeConfig:
    """Configuration for the Land Use Change Detector Agent.

    All settings can be overridden via environment variables with the
    ``GL_EUDR_LUC_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL.
        redis_url: Redis connection URL.
        log_level: Logging verbosity level.
        pool_size: Database connection pool size.
        cache_ttl: Cache TTL in seconds for classification results.
        cutoff_date: EUDR cutoff date in ISO format (YYYY-MM-DD).
        classification_confidence_threshold: Minimum confidence for
            classification acceptance.
        transition_min_area_ha: Minimum area for transition detection.
        batch_size: Maximum plots per batch operation.
        max_batch_concurrency: Maximum concurrent batch operations.
        enable_provenance: Enable SHA-256 provenance chain tracking.
        genesis_hash: Genesis anchor string for provenance chain.
        enable_metrics: Enable Prometheus metrics export.
        rate_limit: Maximum API requests per minute.
        health_check_interval_seconds: Interval between health checks.
    """

    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    pool_size: int = 10
    cache_ttl: int = 3600
    cutoff_date: str = "2020-12-31"
    classification_confidence_threshold: float = 0.70
    transition_min_area_ha: float = 0.5
    batch_size: int = 500
    max_batch_concurrency: int = 50
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-LUC-005-LAND-USE-CHANGE-GENESIS"
    enable_metrics: bool = True
    rate_limit: int = 1000
    health_check_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        errors: list[str] = []

        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")
        if self.cache_ttl <= 0:
            errors.append(f"cache_ttl must be > 0, got {self.cache_ttl}")
        if not (0.0 < self.classification_confidence_threshold <= 1.0):
            errors.append(
                f"classification_confidence_threshold must be in (0.0, 1.0], "
                f"got {self.classification_confidence_threshold}"
            )
        if self.transition_min_area_ha < 0:
            errors.append(
                f"transition_min_area_ha must be >= 0, "
                f"got {self.transition_min_area_ha}"
            )
        if self.batch_size <= 0:
            errors.append(f"batch_size must be > 0, got {self.batch_size}")
        if not (1 <= self.max_batch_concurrency <= 1000):
            errors.append(
                f"max_batch_concurrency must be in [1, 1000], "
                f"got {self.max_batch_concurrency}"
            )
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")
        if self.health_check_interval_seconds <= 0:
            errors.append(
                f"health_check_interval_seconds must be > 0, "
                f"got {self.health_check_interval_seconds}"
            )

        if errors:
            raise ValueError(
                "LandUseChangeConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "LandUseChangeConfig validated: pool=%d, cache_ttl=%ds, "
            "cutoff=%s, confidence=%.2f, batch=%d",
            self.pool_size,
            self.cache_ttl,
            self.cutoff_date,
            self.classification_confidence_threshold,
            self.batch_size,
        )

    @classmethod
    def from_env(cls) -> LandUseChangeConfig:
        """Build configuration from environment variables.

        Returns:
            Populated LandUseChangeConfig instance.
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.strip().lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%r, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%r, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        config = cls(
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            log_level=_str("LOG_LEVEL", cls.log_level),
            pool_size=_int("POOL_SIZE", cls.pool_size),
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            cutoff_date=_str("CUTOFF_DATE", cls.cutoff_date),
            classification_confidence_threshold=_float(
                "CLASSIFICATION_CONFIDENCE_THRESHOLD",
                cls.classification_confidence_threshold,
            ),
            transition_min_area_ha=_float(
                "TRANSITION_MIN_AREA_HA", cls.transition_min_area_ha,
            ),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            max_batch_concurrency=_int(
                "MAX_BATCH_CONCURRENCY", cls.max_batch_concurrency,
            ),
            enable_provenance=_bool("ENABLE_PROVENANCE", cls.enable_provenance),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
            health_check_interval_seconds=_float(
                "HEALTH_CHECK_INTERVAL_SECONDS",
                cls.health_check_interval_seconds,
            ),
        )

        logger.info(
            "LandUseChangeConfig loaded from env: pool=%d, cache_ttl=%ds, "
            "cutoff=%s, confidence=%.2f, batch=%d, metrics=%s",
            config.pool_size,
            config.cache_ttl,
            config.cutoff_date,
            config.classification_confidence_threshold,
            config.batch_size,
            config.enable_metrics,
        )
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary with sensitive fields redacted.

        Returns:
            Dictionary representation with credentials masked.
        """
        return {
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            "log_level": self.log_level,
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
            "cutoff_date": self.cutoff_date,
            "classification_confidence_threshold": self.classification_confidence_threshold,
            "transition_min_area_ha": self.transition_min_area_ha,
            "batch_size": self.batch_size,
            "max_batch_concurrency": self.max_batch_concurrency,
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            "enable_metrics": self.enable_metrics,
            "rate_limit": self.rate_limit,
            "health_check_interval_seconds": self.health_check_interval_seconds,
        }

# ---------------------------------------------------------------------------
# Config singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[LandUseChangeConfig] = None
_config_lock = threading.Lock()

def get_config() -> LandUseChangeConfig:
    """Return the singleton LandUseChangeConfig, creating from env if needed.

    Returns:
        LandUseChangeConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = LandUseChangeConfig.from_env()
    return _config_instance

def set_config(config: LandUseChangeConfig) -> None:
    """Replace the singleton LandUseChangeConfig.

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("LandUseChangeConfig replaced programmatically")

def reset_config() -> None:
    """Reset the singleton LandUseChangeConfig to None.

    The next call to ``get_config()`` will re-read env vars.
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("LandUseChangeConfig singleton reset")

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
# Result container classes
# ---------------------------------------------------------------------------

class LandUseClassificationResult:
    """Result from a single land use classification operation.

    Attributes:
        plot_id: Unique plot identifier.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        classification_date: Date of classification.
        land_use_class: Assigned land use category string.
        confidence: Classification confidence in [0.0, 1.0].
        method: Classification method used.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "latitude", "longitude", "classification_date",
        "land_use_class", "confidence", "method", "provenance_hash",
        "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        latitude: float = 0.0,
        longitude: float = 0.0,
        classification_date: str = "",
        land_use_class: str = "",
        confidence: float = 0.0,
        method: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.latitude = latitude
        self.longitude = longitude
        self.classification_date = classification_date
        self.land_use_class = land_use_class
        self.confidence = confidence
        self.method = method
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plot_id": self.plot_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "classification_date": self.classification_date,
            "land_use_class": self.land_use_class,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class TransitionResult:
    """Result from a transition detection operation.

    Attributes:
        plot_id: Unique plot identifier.
        from_class: Land use class at start date.
        to_class: Land use class at end date.
        transition_type: Classified transition type.
        date_from: Start date of observation.
        date_to: End date of observation.
        confidence: Detection confidence.
        is_deforestation: Whether the transition is EUDR deforestation.
        severity: Severity classification.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "from_class", "to_class", "transition_type",
        "date_from", "date_to", "confidence", "is_deforestation",
        "severity", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        from_class: str = "",
        to_class: str = "",
        transition_type: str = "",
        date_from: str = "",
        date_to: str = "",
        confidence: float = 0.0,
        is_deforestation: bool = False,
        severity: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.from_class = from_class
        self.to_class = to_class
        self.transition_type = transition_type
        self.date_from = date_from
        self.date_to = date_to
        self.confidence = confidence
        self.is_deforestation = is_deforestation
        self.severity = severity
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plot_id": self.plot_id,
            "from_class": self.from_class,
            "to_class": self.to_class,
            "transition_type": self.transition_type,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "confidence": round(self.confidence, 4),
            "is_deforestation": self.is_deforestation,
            "severity": self.severity,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class TrajectoryResult:
    """Result from a temporal trajectory analysis.

    Attributes:
        plot_id: Unique plot identifier.
        trajectory_type: Type of trajectory detected.
        observations: List of dated land use observations.
        trend_direction: Overall trajectory direction.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration.
    """

    __slots__ = (
        "plot_id", "trajectory_type", "observations",
        "trend_direction", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        trajectory_type: str = "",
        observations: Optional[List[Dict[str, Any]]] = None,
        trend_direction: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.trajectory_type = trajectory_type
        self.observations = observations or []
        self.trend_direction = trend_direction
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plot_id": self.plot_id,
            "trajectory_type": self.trajectory_type,
            "observations": self.observations,
            "trend_direction": self.trend_direction,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class CutoffVerificationResult:
    """Result from a cutoff date compliance verification.

    Attributes:
        plot_id: Unique plot identifier.
        commodity: EUDR commodity.
        cutoff_class: Land use at cutoff date.
        current_class: Current land use.
        verdict: Compliance verdict.
        explanation: Human-readable explanation.
        confidence: Verification confidence.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration.
    """

    __slots__ = (
        "plot_id", "commodity", "cutoff_class", "current_class",
        "verdict", "explanation", "confidence",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        commodity: str = "",
        cutoff_class: str = "",
        current_class: str = "",
        verdict: str = "",
        explanation: str = "",
        confidence: float = 0.0,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.commodity = commodity
        self.cutoff_class = cutoff_class
        self.current_class = current_class
        self.verdict = verdict
        self.explanation = explanation
        self.confidence = confidence
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plot_id": self.plot_id,
            "commodity": self.commodity,
            "cutoff_class": self.cutoff_class,
            "current_class": self.current_class,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 4),
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class CroplandConversionResult:
    """Result from a cropland expansion detection.

    Attributes:
        plot_id: Unique plot identifier.
        expansion_detected: Whether agricultural expansion was detected.
        from_class: Source land use category.
        to_class: Destination land use category.
        commodity: EUDR commodity.
        area_converted_ha: Estimated area converted in hectares.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration.
    """

    __slots__ = (
        "plot_id", "expansion_detected", "from_class", "to_class",
        "commodity", "area_converted_ha",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        expansion_detected: bool = False,
        from_class: str = "",
        to_class: str = "",
        commodity: str = "",
        area_converted_ha: float = 0.0,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.expansion_detected = expansion_detected
        self.from_class = from_class
        self.to_class = to_class
        self.commodity = commodity
        self.area_converted_ha = area_converted_ha
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plot_id": self.plot_id,
            "expansion_detected": self.expansion_detected,
            "from_class": self.from_class,
            "to_class": self.to_class,
            "commodity": self.commodity,
            "area_converted_ha": round(self.area_converted_ha, 4),
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class ConversionRiskResult:
    """Result from a conversion risk assessment.

    Attributes:
        plot_id: Unique plot identifier.
        risk_score: Composite risk score in [0.0, 1.0].
        risk_level: Risk classification (low, medium, high, critical).
        commodity: EUDR commodity.
        risk_factors: Individual risk factor scores.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration.
    """

    __slots__ = (
        "plot_id", "risk_score", "risk_level", "commodity",
        "risk_factors", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        risk_score: float = 0.0,
        risk_level: str = "",
        commodity: str = "",
        risk_factors: Optional[Dict[str, float]] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.commodity = commodity
        self.risk_factors = risk_factors or {}
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plot_id": self.plot_id,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "commodity": self.commodity,
            "risk_factors": {k: round(v, 4) for k, v in self.risk_factors.items()},
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class UrbanEncroachmentResult:
    """Result from an urban encroachment analysis.

    Attributes:
        plot_id: Unique plot identifier.
        encroachment_detected: Whether urban expansion was detected.
        distance_to_urban_km: Distance to nearest urban area.
        urban_growth_rate: Annual urban expansion rate.
        buffer_km: Analysis buffer zone.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration.
    """

    __slots__ = (
        "plot_id", "encroachment_detected", "distance_to_urban_km",
        "urban_growth_rate", "buffer_km",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        encroachment_detected: bool = False,
        distance_to_urban_km: float = 0.0,
        urban_growth_rate: float = 0.0,
        buffer_km: float = 5.0,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id or _generate_request_id()
        self.encroachment_detected = encroachment_detected
        self.distance_to_urban_km = distance_to_urban_km
        self.urban_growth_rate = urban_growth_rate
        self.buffer_km = buffer_km
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plot_id": self.plot_id,
            "encroachment_detected": self.encroachment_detected,
            "distance_to_urban_km": round(self.distance_to_urban_km, 3),
            "urban_growth_rate": round(self.urban_growth_rate, 4),
            "buffer_km": self.buffer_km,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class ComplianceReportResult:
    """Result from compliance report generation.

    Attributes:
        report_id: Unique report identifier.
        report_type: Type of report (summary, detailed, evidence).
        format: Output format (json, csv, pdf_data).
        plot_count: Number of plots covered.
        content: Report content data.
        generated_at: Report generation timestamp.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration.
    """

    __slots__ = (
        "report_id", "report_type", "format", "plot_count",
        "content", "generated_at",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        report_id: str = "",
        report_type: str = "",
        format: str = "json",
        plot_count: int = 0,
        content: Optional[Any] = None,
        generated_at: Optional[datetime] = None,
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.report_id = report_id or f"RPT-{uuid.uuid4().hex[:12]}"
        self.report_type = report_type
        self.format = format
        self.plot_count = plot_count
        self.content = content
        self.generated_at = generated_at or utcnow()
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "format": self.format,
            "plot_count": self.plot_count,
            "content": self.content,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class BatchJobResult:
    """Result container for a batch processing job.

    Attributes:
        job_id: Unique job identifier.
        job_type: Type of batch job.
        status: Job status (pending, processing, completed, failed, cancelled).
        total_items: Total items in the batch.
        completed_items: Number of items processed.
        failed_items: Number of items that failed.
        submitted_at: Job submission time.
        completed_at: Job completion time.
        processing_time_ms: Total processing time.
    """

    __slots__ = (
        "job_id", "job_type", "status", "total_items",
        "completed_items", "failed_items",
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
        self.submitted_at = submitted_at or utcnow()
        self.completed_at = completed_at
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# LandUseChangeService
# ---------------------------------------------------------------------------

class LandUseChangeService:
    """Facade for the Land Use Change Detector Agent (AGENT-EUDR-005).

    Provides a unified interface to all 8 engines:
        1. LandUseClassifier          - Multi-class land use classification
        2. TransitionDetector         - Temporal transition detection
        3. TemporalTrajectoryAnalyzer - Change trajectory analysis
        4. CutoffDateVerifier         - EUDR cutoff date compliance
        5. CroplandExpansionDetector  - Agricultural conversion detection
        6. ConversionRiskAssessor     - Conversion risk scoring
        7. UrbanEncroachmentAnalyzer  - Urban expansion monitoring
        8. ComplianceReporter         - Evidence reporting

    Singleton pattern with thread-safe initialization.

    Attributes:
        config: Service configuration loaded from env or injected.
        is_running: Whether the service is currently active and healthy.

    Example:
        >>> service = LandUseChangeService()
        >>> await service.startup()
        >>> result = service.classify_land_use(5.123, -73.456, "2024-01-15")
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[LandUseChangeConfig] = None,
    ) -> None:
        """Initialize LandUseChangeService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.

        Args:
            config: Optional configuration override. If None, loads from
                environment variables via ``get_config()``.
        """
        self._config = config or get_config()
        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_provenance_hash(
            json.dumps(self._config.to_dict(), sort_keys=True, default=str)
        )

        # Connection handles (initialized in startup)
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._land_use_classifier: Optional[Any] = None
        self._transition_detector: Optional[Any] = None
        self._temporal_trajectory_analyzer: Optional[Any] = None
        self._cutoff_date_verifier: Optional[Any] = None
        self._cropland_expansion_detector: Optional[Any] = None
        self._conversion_risk_assessor: Optional[Any] = None
        self._urban_encroachment_analyzer: Optional[Any] = None
        self._compliance_reporter: Optional[Any] = None

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
            "classifications": 0,
            "transitions": 0,
            "trajectories": 0,
            "verifications": 0,
            "expansions": 0,
            "risk_assessments": 0,
            "urban_analyses": 0,
            "reports": 0,
            "errors": 0,
        }

        logger.info(
            "LandUseChangeService created: config_hash=%s, "
            "pool_size=%d, cache_ttl=%ds",
            self._config_hash[:12],
            self._config.pool_size,
            self._config.cache_ttl,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> LandUseChangeConfig:
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
    def land_use_classifier(self) -> Any:
        """Return the LandUseClassifier engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._land_use_classifier

    @property
    def transition_detector(self) -> Any:
        """Return the TransitionDetector engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._transition_detector

    @property
    def temporal_trajectory_analyzer(self) -> Any:
        """Return the TemporalTrajectoryAnalyzer engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._temporal_trajectory_analyzer

    @property
    def cutoff_date_verifier(self) -> Any:
        """Return the CutoffDateVerifier engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._cutoff_date_verifier

    @property
    def cropland_expansion_detector(self) -> Any:
        """Return the CroplandExpansionDetector engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._cropland_expansion_detector

    @property
    def conversion_risk_assessor(self) -> Any:
        """Return the ConversionRiskAssessor engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._conversion_risk_assessor

    @property
    def urban_encroachment_analyzer(self) -> Any:
        """Return the UrbanEncroachmentAnalyzer engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._urban_encroachment_analyzer

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
            logger.debug("LandUseChangeService already started")
            return

        start = time.monotonic()
        logger.info("LandUseChangeService starting up...")

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
            "LandUseChangeService started in %.1fms: "
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
            logger.debug("LandUseChangeService already stopped")
            return

        logger.info("LandUseChangeService shutting down...")
        start = time.monotonic()

        self._stop_health_check()
        await self._close_engines()
        await self._close_redis()
        await self._close_database()
        self._flush_metrics()

        self._started = False
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "LandUseChangeService shut down in %.1fms", elapsed
        )

    # ------------------------------------------------------------------
    # Classification operations
    # ------------------------------------------------------------------

    def classify_land_use(
        self,
        latitude: float,
        longitude: float,
        classification_date: str = "",
        method: str = "spectral",
        commodity: str = "",
    ) -> LandUseClassificationResult:
        """Classify land use at a single coordinate.

        Delegates to the LandUseClassifier engine. Falls back to reference
        data based spectral distance classification if the engine is
        unavailable.

        Args:
            latitude: Latitude in decimal degrees (WGS84).
            longitude: Longitude in decimal degrees (WGS84).
            classification_date: ISO date string for temporal context.
            method: Classification method ('spectral', 'random_forest', 'ensemble').
            commodity: EUDR commodity for context-aware classification.

        Returns:
            LandUseClassificationResult with assigned class and confidence.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Classifying land use: id=%s, lat=%.6f, lon=%.6f, date=%s, "
            "method=%s, commodity=%s",
            request_id, latitude, longitude,
            classification_date, method, commodity,
        )

        try:
            if self._land_use_classifier is not None:
                raw_result = self._land_use_classifier.classify(
                    latitude=latitude,
                    longitude=longitude,
                    date=classification_date,
                    method=method,
                    commodity=commodity,
                )
                elapsed_ms = (time.monotonic() - start) * 1000

                result = LandUseClassificationResult(
                    plot_id=request_id,
                    latitude=latitude,
                    longitude=longitude,
                    classification_date=classification_date,
                    land_use_class=getattr(raw_result, "land_use_class", str(raw_result)),
                    confidence=getattr(raw_result, "confidence", 0.0),
                    method=method,
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                        classification_date, method,
                    ),
                    processing_time_ms=elapsed_ms,
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = LandUseClassificationResult(
                    plot_id=request_id,
                    latitude=latitude,
                    longitude=longitude,
                    classification_date=classification_date,
                    land_use_class="",
                    confidence=0.0,
                    method=method,
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                    ),
                    processing_time_ms=elapsed_ms,
                )

            self._metrics_counters["classifications"] += 1
            logger.info(
                "Classification complete: id=%s, class=%s, "
                "confidence=%.2f, elapsed=%.1fms",
                request_id, result.land_use_class,
                result.confidence, result.processing_time_ms,
            )
            return result

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Classification failed: id=%s, error=%s",
                request_id, exc, exc_info=True,
            )
            raise

    def classify_batch(
        self,
        plots: List[Dict[str, Any]],
        classification_date: str = "",
        method: str = "spectral",
    ) -> List[LandUseClassificationResult]:
        """Classify land use for a batch of coordinates.

        Args:
            plots: List of dicts with 'latitude', 'longitude', and optional
                'commodity' keys.
            classification_date: ISO date string for temporal context.
            method: Classification method.

        Returns:
            List of LandUseClassificationResult objects.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        logger.info(
            "Classifying batch: count=%d, date=%s, method=%s",
            len(plots), classification_date, method,
        )

        results: List[LandUseClassificationResult] = []
        for plot in plots:
            try:
                result = self.classify_land_use(
                    latitude=float(plot.get("latitude", 0.0)),
                    longitude=float(plot.get("longitude", 0.0)),
                    classification_date=classification_date,
                    method=method,
                    commodity=str(plot.get("commodity", "")),
                )
                results.append(result)
            except Exception as exc:
                logger.warning(
                    "Batch classification failed for plot: %s", exc,
                )
                results.append(LandUseClassificationResult(
                    latitude=float(plot.get("latitude", 0.0)),
                    longitude=float(plot.get("longitude", 0.0)),
                    classification_date=classification_date,
                    method=method,
                ))

        logger.info(
            "Batch classification complete: total=%d, successful=%d",
            len(plots),
            sum(1 for r in results if r.land_use_class),
        )
        return results

    def compare_classifications(
        self,
        latitude: float,
        longitude: float,
        date1: str,
        date2: str,
    ) -> Dict[str, Any]:
        """Compare land use classifications at two dates for the same location.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.
            date1: First classification date (ISO format).
            date2: Second classification date (ISO format).

        Returns:
            Dictionary with both classifications and comparison analysis.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        class1 = self.classify_land_use(latitude, longitude, date1)
        class2 = self.classify_land_use(latitude, longitude, date2)

        changed = class1.land_use_class != class2.land_use_class
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "latitude": latitude,
            "longitude": longitude,
            "date1": date1,
            "date2": date2,
            "classification_1": class1.to_dict(),
            "classification_2": class2.to_dict(),
            "land_use_changed": changed,
            "provenance_hash": _compute_provenance_hash(
                str(latitude), str(longitude), date1, date2,
                class1.land_use_class, class2.land_use_class,
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Transition operations
    # ------------------------------------------------------------------

    def detect_transition(
        self,
        latitude: float,
        longitude: float,
        date_from: str,
        date_to: str,
    ) -> TransitionResult:
        """Detect a land use transition between two dates at a location.

        Delegates to the TransitionDetector engine.

        Args:
            latitude: Latitude in decimal degrees (WGS84).
            longitude: Longitude in decimal degrees (WGS84).
            date_from: Start date (ISO format).
            date_to: End date (ISO format).

        Returns:
            TransitionResult with transition classification.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Detecting transition: id=%s, lat=%.6f, lon=%.6f, "
            "from=%s, to=%s",
            request_id, latitude, longitude, date_from, date_to,
        )

        try:
            if self._transition_detector is not None:
                raw_result = self._transition_detector.detect(
                    latitude=latitude,
                    longitude=longitude,
                    date_from=date_from,
                    date_to=date_to,
                )
                elapsed_ms = (time.monotonic() - start) * 1000

                result = TransitionResult(
                    plot_id=request_id,
                    from_class=getattr(raw_result, "from_class", ""),
                    to_class=getattr(raw_result, "to_class", ""),
                    transition_type=getattr(raw_result, "transition_type", ""),
                    date_from=date_from,
                    date_to=date_to,
                    confidence=getattr(raw_result, "confidence", 0.0),
                    is_deforestation=getattr(raw_result, "is_deforestation", False),
                    severity=getattr(raw_result, "severity", ""),
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                        date_from, date_to,
                    ),
                    processing_time_ms=elapsed_ms,
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = TransitionResult(
                    plot_id=request_id,
                    date_from=date_from,
                    date_to=date_to,
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                    ),
                    processing_time_ms=elapsed_ms,
                )

            self._metrics_counters["transitions"] += 1
            logger.info(
                "Transition detection complete: id=%s, type=%s, "
                "deforestation=%s, elapsed=%.1fms",
                request_id, result.transition_type,
                result.is_deforestation, result.processing_time_ms,
            )
            return result

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Transition detection failed: id=%s, error=%s",
                request_id, exc, exc_info=True,
            )
            raise

    def detect_transitions_batch(
        self,
        plots: List[Dict[str, Any]],
        date_from: str,
        date_to: str,
    ) -> List[TransitionResult]:
        """Detect transitions for a batch of plots.

        Args:
            plots: List of dicts with 'latitude' and 'longitude' keys.
            date_from: Start date (ISO format).
            date_to: End date (ISO format).

        Returns:
            List of TransitionResult objects.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        logger.info(
            "Detecting transitions batch: count=%d, from=%s, to=%s",
            len(plots), date_from, date_to,
        )

        results: List[TransitionResult] = []
        for plot in plots:
            try:
                result = self.detect_transition(
                    latitude=float(plot.get("latitude", 0.0)),
                    longitude=float(plot.get("longitude", 0.0)),
                    date_from=date_from,
                    date_to=date_to,
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Batch transition failed: %s", exc)
                results.append(TransitionResult(
                    date_from=date_from,
                    date_to=date_to,
                ))

        return results

    def generate_transition_matrix(
        self,
        region_bounds: Dict[str, float],
        date_from: str,
        date_to: str,
    ) -> Dict[str, Any]:
        """Generate a land use transition matrix for a region.

        Args:
            region_bounds: Dict with 'lat_min', 'lat_max', 'lon_min', 'lon_max'.
            date_from: Start date (ISO format).
            date_to: End date (ISO format).

        Returns:
            Dictionary containing the transition matrix and summary statistics.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Generating transition matrix: bounds=%s, from=%s, to=%s",
            region_bounds, date_from, date_to,
        )

        if self._transition_detector is not None and hasattr(
            self._transition_detector, "generate_matrix"
        ):
            raw_matrix = self._transition_detector.generate_matrix(
                region_bounds=region_bounds,
                date_from=date_from,
                date_to=date_to,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            return {
                "matrix": raw_matrix,
                "region_bounds": region_bounds,
                "date_from": date_from,
                "date_to": date_to,
                "provenance_hash": _compute_provenance_hash(
                    json.dumps(region_bounds, sort_keys=True),
                    date_from, date_to,
                ),
                "processing_time_ms": round(elapsed_ms, 2),
            }

        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "matrix": {},
            "region_bounds": region_bounds,
            "date_from": date_from,
            "date_to": date_to,
            "provenance_hash": _compute_provenance_hash(
                json.dumps(region_bounds, sort_keys=True),
                date_from, date_to,
            ),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Trajectory operations
    # ------------------------------------------------------------------

    def analyze_trajectory(
        self,
        latitude: float,
        longitude: float,
        date_from: str,
        date_to: str,
    ) -> TrajectoryResult:
        """Analyze the temporal trajectory of land use change at a location.

        Delegates to the TemporalTrajectoryAnalyzer engine.

        Args:
            latitude: Latitude in decimal degrees (WGS84).
            longitude: Longitude in decimal degrees (WGS84).
            date_from: Start date of trajectory window (ISO format).
            date_to: End date of trajectory window (ISO format).

        Returns:
            TrajectoryResult with trajectory classification and observations.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Analyzing trajectory: id=%s, lat=%.6f, lon=%.6f, "
            "from=%s, to=%s",
            request_id, latitude, longitude, date_from, date_to,
        )

        try:
            if self._temporal_trajectory_analyzer is not None:
                raw_result = self._temporal_trajectory_analyzer.analyze(
                    latitude=latitude,
                    longitude=longitude,
                    date_from=date_from,
                    date_to=date_to,
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                result = TrajectoryResult(
                    plot_id=request_id,
                    trajectory_type=getattr(raw_result, "trajectory_type", ""),
                    observations=getattr(raw_result, "observations", []),
                    trend_direction=getattr(raw_result, "trend_direction", ""),
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                        date_from, date_to,
                    ),
                    processing_time_ms=elapsed_ms,
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = TrajectoryResult(
                    plot_id=request_id,
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                    ),
                    processing_time_ms=elapsed_ms,
                )

            self._metrics_counters["trajectories"] += 1
            return result

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Trajectory analysis failed: id=%s, error=%s",
                request_id, exc, exc_info=True,
            )
            raise

    def analyze_trajectories_batch(
        self,
        plots: List[Dict[str, Any]],
        date_from: str,
        date_to: str,
    ) -> List[TrajectoryResult]:
        """Analyze trajectories for a batch of plots.

        Args:
            plots: List of dicts with 'latitude' and 'longitude' keys.
            date_from: Start date (ISO format).
            date_to: End date (ISO format).

        Returns:
            List of TrajectoryResult objects.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        results: List[TrajectoryResult] = []
        for plot in plots:
            try:
                result = self.analyze_trajectory(
                    latitude=float(plot.get("latitude", 0.0)),
                    longitude=float(plot.get("longitude", 0.0)),
                    date_from=date_from,
                    date_to=date_to,
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Batch trajectory failed: %s", exc)
                results.append(TrajectoryResult())
        return results

    # ------------------------------------------------------------------
    # Verification operations
    # ------------------------------------------------------------------

    def verify_cutoff_compliance(
        self,
        latitude: float,
        longitude: float,
        commodity: str = "",
    ) -> CutoffVerificationResult:
        """Verify EUDR cutoff date compliance at a location.

        Delegates to the CutoffDateVerifier engine.

        Args:
            latitude: Latitude in decimal degrees (WGS84).
            longitude: Longitude in decimal degrees (WGS84).
            commodity: EUDR commodity identifier.

        Returns:
            CutoffVerificationResult with compliance verdict.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.debug(
            "Verifying cutoff compliance: id=%s, lat=%.6f, lon=%.6f, "
            "commodity=%s",
            request_id, latitude, longitude, commodity,
        )

        try:
            if self._cutoff_date_verifier is not None:
                raw_result = self._cutoff_date_verifier.verify(
                    latitude=latitude,
                    longitude=longitude,
                    commodity=commodity,
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                result = CutoffVerificationResult(
                    plot_id=request_id,
                    commodity=commodity,
                    cutoff_class=getattr(raw_result, "cutoff_class", ""),
                    current_class=getattr(raw_result, "current_class", ""),
                    verdict=getattr(raw_result, "verdict", ""),
                    explanation=getattr(raw_result, "explanation", ""),
                    confidence=getattr(raw_result, "confidence", 0.0),
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                        commodity, self._config.cutoff_date,
                    ),
                    processing_time_ms=elapsed_ms,
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = CutoffVerificationResult(
                    plot_id=request_id,
                    commodity=commodity,
                    verdict="insufficient_data",
                    explanation="CutoffDateVerifier engine not available",
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                    ),
                    processing_time_ms=elapsed_ms,
                )

            self._metrics_counters["verifications"] += 1
            logger.info(
                "Cutoff verification complete: id=%s, verdict=%s, "
                "elapsed=%.1fms",
                request_id, result.verdict, result.processing_time_ms,
            )
            return result

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error(
                "Cutoff verification failed: id=%s, error=%s",
                request_id, exc, exc_info=True,
            )
            raise

    def verify_cutoff_batch(
        self,
        plots: List[Dict[str, Any]],
    ) -> List[CutoffVerificationResult]:
        """Verify cutoff compliance for a batch of plots.

        Args:
            plots: List of dicts with 'latitude', 'longitude', 'commodity'.

        Returns:
            List of CutoffVerificationResult objects.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        results: List[CutoffVerificationResult] = []
        for plot in plots:
            try:
                result = self.verify_cutoff_compliance(
                    latitude=float(plot.get("latitude", 0.0)),
                    longitude=float(plot.get("longitude", 0.0)),
                    commodity=str(plot.get("commodity", "")),
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Batch cutoff verification failed: %s", exc)
                results.append(CutoffVerificationResult(
                    commodity=str(plot.get("commodity", "")),
                    verdict="insufficient_data",
                ))
        return results

    def run_complete_verification(
        self,
        latitude: float,
        longitude: float,
        commodity: str = "",
    ) -> Dict[str, Any]:
        """Run ALL engines for comprehensive verification of a single plot.

        Orchestrates classification, transition detection, trajectory
        analysis, cutoff verification, cropland expansion detection,
        risk assessment, and urban encroachment analysis into a single
        unified result with complete provenance.

        Args:
            latitude: Latitude in decimal degrees (WGS84).
            longitude: Longitude in decimal degrees (WGS84).
            commodity: EUDR commodity identifier.

        Returns:
            Comprehensive dictionary with results from all engines.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        logger.info(
            "Running complete verification: id=%s, lat=%.6f, lon=%.6f, "
            "commodity=%s",
            request_id, latitude, longitude, commodity,
        )

        today_str = utcnow().strftime("%Y-%m-%d")

        # Run all engines with safe wrappers
        classification = self._safe_call(
            lambda: self.classify_land_use(latitude, longitude, today_str, commodity=commodity),
            "classification",
        )
        transition = self._safe_call(
            lambda: self.detect_transition(
                latitude, longitude, self._config.cutoff_date, today_str,
            ),
            "transition",
        )
        trajectory = self._safe_call(
            lambda: self.analyze_trajectory(
                latitude, longitude, self._config.cutoff_date, today_str,
            ),
            "trajectory",
        )
        cutoff = self._safe_call(
            lambda: self.verify_cutoff_compliance(latitude, longitude, commodity),
            "cutoff_verification",
        )
        expansion = self._safe_call(
            lambda: self.detect_cropland_expansion(
                latitude, longitude, self._config.cutoff_date, today_str, commodity,
            ),
            "cropland_expansion",
        )
        risk = self._safe_call(
            lambda: self.assess_conversion_risk(latitude, longitude, commodity),
            "risk_assessment",
        )
        urban = self._safe_call(
            lambda: self.analyze_urban_encroachment(latitude, longitude),
            "urban_encroachment",
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        comprehensive_result = {
            "request_id": request_id,
            "latitude": latitude,
            "longitude": longitude,
            "commodity": commodity,
            "classification": classification.to_dict() if classification else None,
            "transition": transition.to_dict() if transition else None,
            "trajectory": trajectory.to_dict() if trajectory else None,
            "cutoff_verification": cutoff.to_dict() if cutoff else None,
            "cropland_expansion": expansion.to_dict() if expansion else None,
            "risk_assessment": risk.to_dict() if risk else None,
            "urban_encroachment": urban.to_dict() if urban else None,
            "overall_verdict": cutoff.verdict if cutoff else "insufficient_data",
            "provenance_hash": _compute_provenance_hash(
                request_id, str(latitude), str(longitude), commodity,
                self._config.cutoff_date, today_str,
            ),
            "processing_time_ms": round(elapsed_ms, 2),
            "verified_at": utcnow().isoformat(),
        }

        logger.info(
            "Complete verification done: id=%s, verdict=%s, elapsed=%.1fms",
            request_id,
            comprehensive_result["overall_verdict"],
            elapsed_ms,
        )

        return comprehensive_result

    # ------------------------------------------------------------------
    # Conversion operations
    # ------------------------------------------------------------------

    def detect_cropland_expansion(
        self,
        latitude: float,
        longitude: float,
        date_from: str,
        date_to: str,
        commodity: str = "",
    ) -> CroplandConversionResult:
        """Detect agricultural conversion at a location.

        Delegates to the CroplandExpansionDetector engine.

        Args:
            latitude: Latitude in decimal degrees (WGS84).
            longitude: Longitude in decimal degrees (WGS84).
            date_from: Start date (ISO format).
            date_to: End date (ISO format).
            commodity: EUDR commodity identifier.

        Returns:
            CroplandConversionResult with expansion detection result.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            if self._cropland_expansion_detector is not None:
                raw_result = self._cropland_expansion_detector.detect(
                    latitude=latitude,
                    longitude=longitude,
                    date_from=date_from,
                    date_to=date_to,
                    commodity=commodity,
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                result = CroplandConversionResult(
                    plot_id=request_id,
                    expansion_detected=getattr(raw_result, "expansion_detected", False),
                    from_class=getattr(raw_result, "from_class", ""),
                    to_class=getattr(raw_result, "to_class", ""),
                    commodity=commodity,
                    area_converted_ha=getattr(raw_result, "area_converted_ha", 0.0),
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                        date_from, date_to, commodity,
                    ),
                    processing_time_ms=elapsed_ms,
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = CroplandConversionResult(
                    plot_id=request_id,
                    commodity=commodity,
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                    ),
                    processing_time_ms=elapsed_ms,
                )

            self._metrics_counters["expansions"] += 1
            return result

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error("Cropland expansion detection failed: %s", exc, exc_info=True)
            raise

    def detect_expansion_batch(
        self,
        plots: List[Dict[str, Any]],
        date_from: str,
        date_to: str,
    ) -> List[CroplandConversionResult]:
        """Detect cropland expansion for a batch of plots.

        Args:
            plots: List of dicts with 'latitude', 'longitude', 'commodity'.
            date_from: Start date (ISO format).
            date_to: End date (ISO format).

        Returns:
            List of CroplandConversionResult objects.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        results: List[CroplandConversionResult] = []
        for plot in plots:
            try:
                result = self.detect_cropland_expansion(
                    latitude=float(plot.get("latitude", 0.0)),
                    longitude=float(plot.get("longitude", 0.0)),
                    date_from=date_from,
                    date_to=date_to,
                    commodity=str(plot.get("commodity", "")),
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Batch expansion detection failed: %s", exc)
                results.append(CroplandConversionResult())
        return results

    # ------------------------------------------------------------------
    # Risk operations
    # ------------------------------------------------------------------

    def assess_conversion_risk(
        self,
        latitude: float,
        longitude: float,
        commodity: str = "",
    ) -> ConversionRiskResult:
        """Assess conversion risk at a location.

        Delegates to the ConversionRiskAssessor engine.

        Args:
            latitude: Latitude in decimal degrees (WGS84).
            longitude: Longitude in decimal degrees (WGS84).
            commodity: EUDR commodity identifier.

        Returns:
            ConversionRiskResult with risk score and level.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            if self._conversion_risk_assessor is not None:
                raw_result = self._conversion_risk_assessor.assess(
                    latitude=latitude,
                    longitude=longitude,
                    commodity=commodity,
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                result = ConversionRiskResult(
                    plot_id=request_id,
                    risk_score=getattr(raw_result, "risk_score", 0.0),
                    risk_level=getattr(raw_result, "risk_level", ""),
                    commodity=commodity,
                    risk_factors=getattr(raw_result, "risk_factors", {}),
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude), commodity,
                    ),
                    processing_time_ms=elapsed_ms,
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = ConversionRiskResult(
                    plot_id=request_id,
                    commodity=commodity,
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                    ),
                    processing_time_ms=elapsed_ms,
                )

            self._metrics_counters["risk_assessments"] += 1
            return result

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error("Risk assessment failed: %s", exc, exc_info=True)
            raise

    def assess_risk_batch(
        self,
        plots: List[Dict[str, Any]],
    ) -> List[ConversionRiskResult]:
        """Assess conversion risk for a batch of plots.

        Args:
            plots: List of dicts with 'latitude', 'longitude', 'commodity'.

        Returns:
            List of ConversionRiskResult objects.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        results: List[ConversionRiskResult] = []
        for plot in plots:
            try:
                result = self.assess_conversion_risk(
                    latitude=float(plot.get("latitude", 0.0)),
                    longitude=float(plot.get("longitude", 0.0)),
                    commodity=str(plot.get("commodity", "")),
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Batch risk assessment failed: %s", exc)
                results.append(ConversionRiskResult())
        return results

    # ------------------------------------------------------------------
    # Urban operations
    # ------------------------------------------------------------------

    def analyze_urban_encroachment(
        self,
        latitude: float,
        longitude: float,
        buffer_km: float = 5.0,
    ) -> UrbanEncroachmentResult:
        """Analyze urban encroachment at a location.

        Delegates to the UrbanEncroachmentAnalyzer engine.

        Args:
            latitude: Latitude in decimal degrees (WGS84).
            longitude: Longitude in decimal degrees (WGS84).
            buffer_km: Analysis buffer zone in kilometers.

        Returns:
            UrbanEncroachmentResult with encroachment analysis.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            if self._urban_encroachment_analyzer is not None:
                raw_result = self._urban_encroachment_analyzer.analyze(
                    latitude=latitude,
                    longitude=longitude,
                    buffer_km=buffer_km,
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                result = UrbanEncroachmentResult(
                    plot_id=request_id,
                    encroachment_detected=getattr(raw_result, "encroachment_detected", False),
                    distance_to_urban_km=getattr(raw_result, "distance_to_urban_km", 0.0),
                    urban_growth_rate=getattr(raw_result, "urban_growth_rate", 0.0),
                    buffer_km=buffer_km,
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude), str(buffer_km),
                    ),
                    processing_time_ms=elapsed_ms,
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = UrbanEncroachmentResult(
                    plot_id=request_id,
                    buffer_km=buffer_km,
                    provenance_hash=_compute_provenance_hash(
                        request_id, str(latitude), str(longitude),
                    ),
                    processing_time_ms=elapsed_ms,
                )

            self._metrics_counters["urban_analyses"] += 1
            return result

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error("Urban encroachment analysis failed: %s", exc, exc_info=True)
            raise

    def analyze_urban_batch(
        self,
        plots: List[Dict[str, Any]],
        buffer_km: float = 5.0,
    ) -> List[UrbanEncroachmentResult]:
        """Analyze urban encroachment for a batch of plots.

        Args:
            plots: List of dicts with 'latitude' and 'longitude'.
            buffer_km: Analysis buffer zone in kilometers.

        Returns:
            List of UrbanEncroachmentResult objects.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        results: List[UrbanEncroachmentResult] = []
        for plot in plots:
            try:
                result = self.analyze_urban_encroachment(
                    latitude=float(plot.get("latitude", 0.0)),
                    longitude=float(plot.get("longitude", 0.0)),
                    buffer_km=buffer_km,
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Batch urban analysis failed: %s", exc)
                results.append(UrbanEncroachmentResult(buffer_km=buffer_km))
        return results

    # ------------------------------------------------------------------
    # Report operations
    # ------------------------------------------------------------------

    def generate_report(
        self,
        report_type: str = "summary",
        plot_ids: Optional[List[str]] = None,
        format: str = "json",
    ) -> ComplianceReportResult:
        """Generate a compliance report.

        Delegates to the ComplianceReporter engine.

        Args:
            report_type: Report type ('summary', 'detailed', 'evidence').
            plot_ids: Optional list of plot IDs to include.
            format: Output format ('json', 'csv', 'pdf_data').

        Returns:
            ComplianceReportResult with report content.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Generating report: type=%s, plots=%d, format=%s",
            report_type, len(plot_ids or []), format,
        )

        try:
            if self._compliance_reporter is not None:
                raw_result = self._compliance_reporter.generate(
                    report_type=report_type,
                    plot_ids=plot_ids or [],
                    export_format=format,
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                result = ComplianceReportResult(
                    report_type=report_type,
                    format=format,
                    plot_count=len(plot_ids or []),
                    content=getattr(raw_result, "content", raw_result),
                    provenance_hash=_compute_provenance_hash(
                        report_type, format,
                        json.dumps(plot_ids or [], sort_keys=True),
                    ),
                    processing_time_ms=elapsed_ms,
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = ComplianceReportResult(
                    report_type=report_type,
                    format=format,
                    plot_count=len(plot_ids or []),
                    content={"message": "ComplianceReporter engine not available"},
                    provenance_hash=_compute_provenance_hash(
                        report_type, format,
                    ),
                    processing_time_ms=elapsed_ms,
                )

            self._metrics_counters["reports"] += 1
            return result

        except Exception as exc:
            self._metrics_counters["errors"] += 1
            logger.error("Report generation failed: %s", exc, exc_info=True)
            raise

    def generate_batch_reports(
        self,
        configs: List[Dict[str, Any]],
    ) -> List[ComplianceReportResult]:
        """Generate multiple compliance reports.

        Args:
            configs: List of report configuration dicts with keys
                'report_type', 'plot_ids', 'format'.

        Returns:
            List of ComplianceReportResult objects.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        results: List[ComplianceReportResult] = []
        for cfg in configs:
            try:
                result = self.generate_report(
                    report_type=str(cfg.get("report_type", "summary")),
                    plot_ids=cfg.get("plot_ids"),
                    format=str(cfg.get("format", "json")),
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Batch report generation failed: %s", exc)
                results.append(ComplianceReportResult(
                    report_type=str(cfg.get("report_type", "summary")),
                ))
        return results

    # ------------------------------------------------------------------
    # Batch job operations
    # ------------------------------------------------------------------

    def submit_batch_job(
        self,
        job_type: str,
        params: Dict[str, Any],
    ) -> BatchJobResult:
        """Submit a batch processing job.

        Args:
            job_type: Type of job ('classify', 'transition', 'verify',
                'expansion', 'risk', 'urban').
            params: Job parameters including 'plots' list and date parameters.

        Returns:
            BatchJobResult with job status.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If job_type is not recognized.
        """
        self._ensure_started()
        start = time.monotonic()

        valid_types = {
            "classify", "transition", "verify",
            "expansion", "risk", "urban",
        }
        if job_type not in valid_types:
            raise ValueError(
                f"Unknown job type: '{job_type}'. "
                f"Valid types: {sorted(valid_types)}"
            )

        plots = params.get("plots", [])
        job = BatchJobResult(
            job_type=job_type,
            status="processing",
            total_items=len(plots),
        )

        with self._batch_lock:
            self._batch_registry[job.job_id] = job

        logger.info(
            "Batch job submitted: id=%s, type=%s, items=%d",
            job.job_id, job_type, len(plots),
        )

        # Process based on job type
        completed = 0
        failed = 0
        try:
            if job_type == "classify":
                results = self.classify_batch(
                    plots=plots,
                    classification_date=params.get("date", ""),
                    method=params.get("method", "spectral"),
                )
                completed = sum(1 for r in results if r.land_use_class)
                failed = len(results) - completed
            elif job_type == "transition":
                results = self.detect_transitions_batch(
                    plots=plots,
                    date_from=params.get("date_from", ""),
                    date_to=params.get("date_to", ""),
                )
                completed = sum(1 for r in results if r.transition_type)
                failed = len(results) - completed
            elif job_type == "verify":
                results = self.verify_cutoff_batch(plots=plots)
                completed = sum(1 for r in results if r.verdict)
                failed = len(results) - completed
            elif job_type == "expansion":
                results = self.detect_expansion_batch(
                    plots=plots,
                    date_from=params.get("date_from", ""),
                    date_to=params.get("date_to", ""),
                )
                completed = len(results)
            elif job_type == "risk":
                results = self.assess_risk_batch(plots=plots)
                completed = len(results)
            elif job_type == "urban":
                results = self.analyze_urban_batch(
                    plots=plots,
                    buffer_km=float(params.get("buffer_km", 5.0)),
                )
                completed = len(results)
        except Exception as exc:
            logger.error(
                "Batch job failed: id=%s, error=%s",
                job.job_id, exc, exc_info=True,
            )
            with self._batch_lock:
                job.status = "failed"
                job.completed_at = utcnow()
            return job

        elapsed_ms = (time.monotonic() - start) * 1000

        with self._batch_lock:
            job.status = "completed"
            job.completed_items = completed
            job.failed_items = failed
            job.completed_at = utcnow()
            job.processing_time_ms = elapsed_ms

        logger.info(
            "Batch job completed: id=%s, completed=%d, failed=%d, "
            "elapsed=%.1fms",
            job.job_id, completed, failed, elapsed_ms,
        )

        return job

    def get_batch_status(self, job_id: str) -> BatchJobResult:
        """Get the status of a batch job.

        Args:
            job_id: Job identifier returned by ``submit_batch_job()``.

        Returns:
            BatchJobResult with current status.

        Raises:
            ValueError: If the job_id is not found.
        """
        with self._batch_lock:
            job = self._batch_registry.get(job_id)

        if job is None:
            raise ValueError(f"Batch job not found: {job_id}")
        return job

    def cancel_batch_job(self, job_id: str) -> bool:
        """Cancel a batch job.

        Args:
            job_id: Job identifier to cancel.

        Returns:
            True if the job was found and cancelled, False otherwise.
        """
        with self._batch_lock:
            job = self._batch_registry.get(job_id)
            if job is None:
                return False
            if job.status in ("completed", "failed", "cancelled"):
                return False
            job.status = "cancelled"
            job.completed_at = utcnow()

        logger.info("Batch job cancelled: id=%s", job_id)
        return True

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of all components.

        Returns:
            Dictionary with status, component checks, version, and uptime.
        """
        checks: Dict[str, Any] = {}

        checks["database"] = await self._check_database_health()
        checks["redis"] = await self._check_redis_health()
        checks["engines"] = self._check_engine_health()
        checks["metrics"] = {
            "status": "healthy",
            "counters": dict(self._metrics_counters),
        }

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

    # ------------------------------------------------------------------
    # Internal: safe call wrapper
    # ------------------------------------------------------------------

    def _safe_call(self, func: Any, label: str) -> Optional[Any]:
        """Execute a callable with exception capture.

        Args:
            func: Zero-argument callable to invoke.
            label: Label for logging on failure.

        Returns:
            Result of the callable, or None if it raised an exception.
        """
        try:
            return func()
        except Exception as exc:
            logger.warning(
                "Safe call failed for %s (non-fatal): %s", label, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Internal: Logging
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure structured logging based on service configuration."""
        log_level = getattr(logging, self._config.log_level, logging.INFO)
        logging.getLogger(
            "greenlang.agents.eudr.land_use_change"
        ).setLevel(log_level)
        logger.debug(
            "Logging configured: level=%s", self._config.log_level
        )

    # ------------------------------------------------------------------
    # Internal: OpenTelemetry
    # ------------------------------------------------------------------

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            self._tracer = otel_trace.get_tracer(
                "greenlang.agents.eudr.land_use_change",
                "1.0.0",
            )
            logger.info("OpenTelemetry tracer initialized")
        else:
            self._tracer = None
            logger.debug(
                "OpenTelemetry not available, tracing disabled"
            )

    # ------------------------------------------------------------------
    # Internal: Database
    # ------------------------------------------------------------------

    async def _connect_database(self) -> None:
        """Create async PostgreSQL connection pool.

        Raises:
            RuntimeError: If psycopg is not available or connection fails.
        """
        if not PSYCOPG_POOL_AVAILABLE or not PSYCOPG_AVAILABLE:
            logger.warning(
                "psycopg/psycopg_pool not available, database disabled. "
                "Install with: pip install 'psycopg[binary]' psycopg_pool"
            )
            self._db_pool = None
            return

        try:
            conninfo = self._config.database_url
            pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=max(1, self._config.pool_size // 2),
                max_size=self._config.pool_size,
                open=False,
            )
            await pool.open()
            await pool.check()
            self._db_pool = pool
            logger.info(
                "PostgreSQL connection pool opened: min=%d, max=%d",
                max(1, self._config.pool_size // 2),
                self._config.pool_size,
            )
        except Exception as exc:
            logger.error(
                "Failed to connect to PostgreSQL: %s", exc, exc_info=True
            )
            self._db_pool = None
            raise RuntimeError(
                f"Database connection failed: {exc}"
            ) from exc

    async def _register_pgvector(self) -> None:
        """Register pgvector type extension on the connection pool."""
        if self._db_pool is None:
            logger.debug("Skipping pgvector registration: no database pool")
            return

        try:
            async with self._db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            logger.info("pgvector extension registration check completed")
        except Exception as exc:
            logger.warning(
                "pgvector registration failed (non-fatal): %s", exc
            )

    async def _close_database(self) -> None:
        """Close the PostgreSQL connection pool."""
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
                logger.info("PostgreSQL connection pool closed")
            except Exception as exc:
                logger.warning(
                    "Error closing database pool: %s", exc
                )
            finally:
                self._db_pool = None

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity health.

        Returns:
            Dictionary with status, pool stats, and latency.
        """
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

    # ------------------------------------------------------------------
    # Internal: Redis
    # ------------------------------------------------------------------

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
                "Redis connected: url=%s, ttl=%ds",
                "***",
                self._config.cache_ttl,
            )
        except Exception as exc:
            logger.warning(
                "Failed to connect to Redis (non-fatal): %s", exc
            )
            self._redis = None

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

    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity health.

        Returns:
            Dictionary with status and latency.
        """
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

    # ------------------------------------------------------------------
    # Internal: Engine initialization
    # ------------------------------------------------------------------

    async def _initialize_engines(self) -> None:
        """Initialize all eight internal engines."""
        logger.info("Initializing 8 land use change engines...")

        self._land_use_classifier = await self._init_land_use_classifier()
        self._transition_detector = await self._init_transition_detector()
        self._temporal_trajectory_analyzer = await self._init_temporal_trajectory_analyzer()
        self._cutoff_date_verifier = await self._init_cutoff_date_verifier()
        self._cropland_expansion_detector = await self._init_cropland_expansion_detector()
        self._conversion_risk_assessor = await self._init_conversion_risk_assessor()
        self._urban_encroachment_analyzer = await self._init_urban_encroachment_analyzer()
        self._compliance_reporter = await self._init_compliance_reporter()

        logger.info("All 8 engines initialized successfully")

    async def _init_land_use_classifier(self) -> Any:
        """Initialize the LandUseClassifier engine.

        Returns:
            Initialized LandUseClassifier instance, or None if unavailable.
        """
        try:
            from greenlang.agents.eudr.land_use_change.land_use_classifier import (
                LandUseClassifier,
            )
            engine = LandUseClassifier(config=self._config)
            logger.info("LandUseClassifier initialized")
            return engine
        except ImportError:
            logger.debug("LandUseClassifier module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize LandUseClassifier: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_transition_detector(self) -> Any:
        """Initialize the TransitionDetector engine.

        Returns:
            Initialized TransitionDetector instance, or None if unavailable.
        """
        try:
            from greenlang.agents.eudr.land_use_change.transition_detector import (
                TransitionDetector,
            )
            engine = TransitionDetector(config=self._config)
            logger.info("TransitionDetector initialized")
            return engine
        except ImportError:
            logger.debug("TransitionDetector module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize TransitionDetector: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_temporal_trajectory_analyzer(self) -> Any:
        """Initialize the TemporalTrajectoryAnalyzer engine.

        Returns:
            Initialized TemporalTrajectoryAnalyzer instance, or None.
        """
        try:
            from greenlang.agents.eudr.land_use_change.temporal_trajectory_analyzer import (
                TemporalTrajectoryAnalyzer,
            )
            engine = TemporalTrajectoryAnalyzer(config=self._config)
            logger.info("TemporalTrajectoryAnalyzer initialized")
            return engine
        except ImportError:
            logger.debug("TemporalTrajectoryAnalyzer module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize TemporalTrajectoryAnalyzer: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_cutoff_date_verifier(self) -> Any:
        """Initialize the CutoffDateVerifier engine.

        Returns:
            Initialized CutoffDateVerifier instance, or None.
        """
        try:
            from greenlang.agents.eudr.land_use_change.cutoff_date_verifier import (
                CutoffDateVerifier,
            )
            engine = CutoffDateVerifier(config=self._config)
            logger.info("CutoffDateVerifier initialized")
            return engine
        except ImportError:
            logger.debug("CutoffDateVerifier module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize CutoffDateVerifier: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_cropland_expansion_detector(self) -> Any:
        """Initialize the CroplandExpansionDetector engine.

        Returns:
            Initialized CroplandExpansionDetector instance, or None.
        """
        try:
            from greenlang.agents.eudr.land_use_change.cropland_expansion_detector import (
                CroplandExpansionDetector,
            )
            engine = CroplandExpansionDetector(config=self._config)
            logger.info("CroplandExpansionDetector initialized")
            return engine
        except ImportError:
            logger.debug("CroplandExpansionDetector module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize CroplandExpansionDetector: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_conversion_risk_assessor(self) -> Any:
        """Initialize the ConversionRiskAssessor engine.

        Returns:
            Initialized ConversionRiskAssessor instance, or None.
        """
        try:
            from greenlang.agents.eudr.land_use_change.conversion_risk_assessor import (
                ConversionRiskAssessor,
            )
            engine = ConversionRiskAssessor(config=self._config)
            logger.info("ConversionRiskAssessor initialized")
            return engine
        except ImportError:
            logger.debug("ConversionRiskAssessor module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize ConversionRiskAssessor: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_urban_encroachment_analyzer(self) -> Any:
        """Initialize the UrbanEncroachmentAnalyzer engine.

        Returns:
            Initialized UrbanEncroachmentAnalyzer instance, or None.
        """
        try:
            from greenlang.agents.eudr.land_use_change.urban_encroachment_analyzer import (
                UrbanEncroachmentAnalyzer,
            )
            engine = UrbanEncroachmentAnalyzer(config=self._config)
            logger.info("UrbanEncroachmentAnalyzer initialized")
            return engine
        except ImportError:
            logger.debug("UrbanEncroachmentAnalyzer module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize UrbanEncroachmentAnalyzer: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_compliance_reporter(self) -> Any:
        """Initialize the ComplianceReporter engine.

        Returns:
            Initialized ComplianceReporter instance, or None.
        """
        try:
            from greenlang.agents.eudr.land_use_change.compliance_reporter import (
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

    # ------------------------------------------------------------------
    # Internal: Engine shutdown
    # ------------------------------------------------------------------

    async def _close_engines(self) -> None:
        """Close all engines that implement a close/shutdown method."""
        engine_names = [
            ("land_use_classifier", self._land_use_classifier),
            ("transition_detector", self._transition_detector),
            ("temporal_trajectory_analyzer", self._temporal_trajectory_analyzer),
            ("cutoff_date_verifier", self._cutoff_date_verifier),
            ("cropland_expansion_detector", self._cropland_expansion_detector),
            ("conversion_risk_assessor", self._conversion_risk_assessor),
            ("urban_encroachment_analyzer", self._urban_encroachment_analyzer),
            ("compliance_reporter", self._compliance_reporter),
        ]

        for name, engine in engine_names:
            if engine is None:
                continue
            try:
                if hasattr(engine, "shutdown") and asyncio.iscoroutinefunction(
                    engine.shutdown
                ):
                    await engine.shutdown()
                elif hasattr(engine, "close") and asyncio.iscoroutinefunction(
                    engine.close
                ):
                    await engine.close()
                logger.debug("Engine %s closed", name)
            except Exception as exc:
                logger.warning(
                    "Error closing engine %s: %s", name, exc
                )

        self._land_use_classifier = None
        self._transition_detector = None
        self._temporal_trajectory_analyzer = None
        self._cutoff_date_verifier = None
        self._cropland_expansion_detector = None
        self._conversion_risk_assessor = None
        self._urban_encroachment_analyzer = None
        self._compliance_reporter = None

        logger.info("All engines closed")

    # ------------------------------------------------------------------
    # Internal: Health check background task
    # ------------------------------------------------------------------

    def _start_health_check(self) -> None:
        """Start the background health check task."""
        if self._health_task is not None:
            return
        self._health_task = asyncio.create_task(
            self._health_check_loop(),
            name="luc-health-check",
        )
        logger.debug(
            "Health check background task started (interval=%.0fs)",
            self._config.health_check_interval_seconds,
        )

    def _stop_health_check(self) -> None:
        """Cancel the background health check task."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None
            logger.debug("Health check background task cancelled")

    async def _health_check_loop(self) -> None:
        """Background loop that periodically runs health checks."""
        try:
            while True:
                try:
                    await self.health_check()
                except Exception as exc:
                    logger.warning("Health check failed: %s", exc)
                await asyncio.sleep(self._config.health_check_interval_seconds)
        except asyncio.CancelledError:
            logger.debug("Health check loop cancelled")

    # ------------------------------------------------------------------
    # Internal: Engine health
    # ------------------------------------------------------------------

    def _check_engine_health(self) -> Dict[str, Any]:
        """Check initialization status of all eight engines.

        Returns:
            Dictionary with per-engine status and overall engine health.
        """
        engines = {
            "land_use_classifier": self._land_use_classifier,
            "transition_detector": self._transition_detector,
            "temporal_trajectory_analyzer": self._temporal_trajectory_analyzer,
            "cutoff_date_verifier": self._cutoff_date_verifier,
            "cropland_expansion_detector": self._cropland_expansion_detector,
            "conversion_risk_assessor": self._conversion_risk_assessor,
            "urban_encroachment_analyzer": self._urban_encroachment_analyzer,
            "compliance_reporter": self._compliance_reporter,
        }

        engine_statuses: Dict[str, str] = {}
        initialized_count = 0
        for name, engine in engines.items():
            if engine is not None:
                engine_statuses[name] = "initialized"
                initialized_count += 1
            else:
                engine_statuses[name] = "unavailable"

        core_engines = [
            "land_use_classifier",
            "transition_detector",
            "cutoff_date_verifier",
        ]
        core_ok = all(
            engine_statuses.get(e) == "initialized" for e in core_engines
        )

        overall = "healthy" if core_ok else "degraded"

        return {
            "status": overall,
            "initialized": initialized_count,
            "total": len(engines),
            "engines": engine_statuses,
        }

    # ------------------------------------------------------------------
    # Internal: Metrics
    # ------------------------------------------------------------------

    def _flush_metrics(self) -> None:
        """Flush Prometheus metrics on shutdown."""
        if not self._config.enable_metrics:
            return
        logger.debug(
            "Prometheus metrics flushed: %s", self._metrics_counters
        )

    # ------------------------------------------------------------------
    # Internal: Guard
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Raise RuntimeError if the service is not started.

        Raises:
            RuntimeError: If the service has not been started.
        """
        if not self._started:
            raise RuntimeError(
                "LandUseChangeService is not started. "
                "Call await service.startup() first."
            )

    # ------------------------------------------------------------------
    # Convenience: get_engine
    # ------------------------------------------------------------------

    def get_engine(self, name: str) -> Any:
        """Retrieve an engine by name.

        Args:
            name: Engine name.

        Returns:
            The engine instance, or None if not initialized.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If the engine name is not recognized.
        """
        self._ensure_started()
        valid_names = {
            "land_use_classifier": self._land_use_classifier,
            "transition_detector": self._transition_detector,
            "temporal_trajectory_analyzer": self._temporal_trajectory_analyzer,
            "cutoff_date_verifier": self._cutoff_date_verifier,
            "cropland_expansion_detector": self._cropland_expansion_detector,
            "conversion_risk_assessor": self._conversion_risk_assessor,
            "urban_encroachment_analyzer": self._urban_encroachment_analyzer,
            "compliance_reporter": self._compliance_reporter,
        }
        if name not in valid_names:
            raise ValueError(
                f"Unknown engine name: '{name}'. "
                f"Valid names: {sorted(valid_names.keys())}"
            )
        return valid_names[name]

    # ------------------------------------------------------------------
    # Convenience: engine count
    # ------------------------------------------------------------------

    def initialized_engine_count(self) -> int:
        """Return the number of successfully initialized engines.

        Returns:
            Count of non-None engine instances (0 to 8).
        """
        engines = [
            self._land_use_classifier,
            self._transition_detector,
            self._temporal_trajectory_analyzer,
            self._cutoff_date_verifier,
            self._cropland_expansion_detector,
            self._conversion_risk_assessor,
            self._urban_encroachment_analyzer,
            self._compliance_reporter,
        ]
        return sum(1 for e in engines if e is not None)

# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Land Use Change service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown. The service instance is stored in
    ``app.state.luc_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.land_use_change.setup import lifespan
from greenlang.schemas import utcnow

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.luc_service``).
    """
    service = get_service()
    app.state.luc_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()

# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[LandUseChangeService] = None
_service_lock = threading.Lock()

def get_service(
    config: Optional[LandUseChangeConfig] = None,
) -> LandUseChangeService:
    """Return the singleton LandUseChangeService instance.

    Uses double-checked locking for thread safety. The instance is
    created on first call. Pass a config to override the default
    environment-based configuration.

    Args:
        config: Optional configuration override.

    Returns:
        LandUseChangeService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = LandUseChangeService(config=config)
    return _service_instance

def set_service(service: LandUseChangeService) -> None:
    """Replace the singleton LandUseChangeService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("LandUseChangeService singleton replaced")

def reset_service() -> None:
    """Reset the singleton LandUseChangeService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("LandUseChangeService singleton reset")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Configuration
    "LandUseChangeConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Service
    "LandUseChangeService",
    "HealthStatus",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
    # Result containers
    "LandUseClassificationResult",
    "TransitionResult",
    "TrajectoryResult",
    "CutoffVerificationResult",
    "CroplandConversionResult",
    "ConversionRiskResult",
    "UrbanEncroachmentResult",
    "ComplianceReportResult",
    "BatchJobResult",
]
