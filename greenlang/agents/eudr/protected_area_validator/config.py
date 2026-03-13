# -*- coding: utf-8 -*-
"""
Protected Area Validator Configuration - AGENT-EUDR-022

Centralized configuration for the Protected Area Validator Agent covering:
- Database and cache connection settings (PostgreSQL/PostGIS, Redis) with
  configurable pool sizes, timeouts, and key prefixes using ``gl_eudr_pav_``
- WDPA integration: data source URLs, update schedule, staleness thresholds,
  spatial indexing (GIST) and version tracking for 270,000+ protected areas
- Spatial analysis: overlap detection tolerances, buffer zone defaults (1/5/
  10/25/50 km), ST_Intersects precision, SRID 4326 (WGS84)
- Risk scoring weights: IUCN category score (0.50), designation level (0.20),
  management effectiveness gap (0.15), country enforcement gap (0.15) with
  overlap type multipliers (INSIDE=1.0, PARTIAL=0.8, BOUNDARY=0.6, BUFFER=0.3)
- Buffer zone monitoring: default radii per IUCN category, national regulation
  overrides (Brazil 10 km, Indonesia 5 km, Colombia per Decree 2372/2010)
- Alert generation: proximity thresholds for high-risk protected areas,
  enhanced due diligence SLA deadlines (CRITICAL=14d, SEVERE=30d)
- Compliance tracking: SLA deadlines for investigation (14d), remediation
  plan (30d), execution (90d), verification (30d); 5-year retention
- Reporting: 8 report types, 5 formats (PDF/JSON/HTML/CSV/XLSX), 5 languages
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)

All settings can be overridden via environment variables with the
``GL_EUDR_PAV_`` prefix (e.g. ``GL_EUDR_PAV_DATABASE_URL``,
``GL_EUDR_PAV_DEFAULT_BUFFER_RADIUS_KM``).

Example:
    >>> from greenlang.agents.eudr.protected_area_validator.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_buffer_radius_km, cfg.wdpa_staleness_days)
    10 60

    >>> from greenlang.agents.eudr.protected_area_validator.config import (
    ...     set_config, reset_config, ProtectedAreaValidatorConfig,
    ... )
    >>> set_config(ProtectedAreaValidatorConfig(default_buffer_radius_km=Decimal("25")))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 Protected Area Validator (GL-EUDR-PAV-022)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_PAV_"

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_CHAIN_ALGORITHMS = frozenset({"sha256", "sha384", "sha512"})
_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "csv", "xlsx"})
_VALID_REPORT_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})
_VALID_MARKET_RESTRICTION_THRESHOLDS = frozenset({"CRITICAL", "HIGH", "MEDIUM", "LOW"})

# ---------------------------------------------------------------------------
# Default risk scoring weights (PRD Section 6.1 Feature 2)
# ---------------------------------------------------------------------------

_DEFAULT_RISK_WEIGHTS: Dict[str, Decimal] = {
    "iucn_category": Decimal("0.50"),
    "designation_level": Decimal("0.20"),
    "management_gap": Decimal("0.15"),
    "enforcement_gap": Decimal("0.15"),
}

# ---------------------------------------------------------------------------
# Default proximity risk weights (PRD Section 6.1 Feature 5)
# ---------------------------------------------------------------------------

_DEFAULT_PROXIMITY_WEIGHTS: Dict[str, Decimal] = {
    "distance": Decimal("0.40"),
    "iucn_category": Decimal("0.25"),
    "designation_level": Decimal("0.15"),
    "multi_designation": Decimal("0.10"),
    "deforestation_trend": Decimal("0.10"),
}

# ---------------------------------------------------------------------------
# Default buffer radii (km)
# ---------------------------------------------------------------------------

_DEFAULT_BUFFER_RADII: List[int] = [1, 5, 10, 25, 50]

# ---------------------------------------------------------------------------
# Default SLA deadlines (days)
# ---------------------------------------------------------------------------

_DEFAULT_SLA_INVESTIGATION_DAYS: int = 14
_DEFAULT_SLA_REMEDIATION_PLAN_DAYS: int = 30
_DEFAULT_SLA_REMEDIATION_EXEC_DAYS: int = 90
_DEFAULT_SLA_VERIFICATION_DAYS: int = 30

# ---------------------------------------------------------------------------
# Default enhanced DD SLA deadlines (days)
# ---------------------------------------------------------------------------

_DEFAULT_EDD_CRITICAL_DAYS: int = 14
_DEFAULT_EDD_SEVERE_DAYS: int = 30


@dataclass
class ProtectedAreaValidatorConfig:
    """Configuration for the Protected Area Validator Agent (AGENT-EUDR-022).

    This dataclass encapsulates all configuration settings for WDPA
    integration, spatial overlap detection, buffer zone monitoring, IUCN
    category risk scoring, designation validation, proximity alerting,
    compliance tracking, and compliance reporting. All settings have
    sensible defaults aligned with EUDR requirements and can be overridden
    via environment variables with the GL_EUDR_PAV_ prefix.

    Attributes:
        database_url: PostgreSQL/PostGIS connection URL.
        pool_size: Connection pool size.
        pool_timeout_s: Connection pool timeout seconds.
        pool_recycle_s: Connection pool recycle seconds.
        redis_url: Redis connection URL.
        redis_ttl_s: Redis cache TTL seconds.
        redis_key_prefix: Redis key prefix.
        log_level: Logging level.
        wdpa_staleness_days: Max days before WDPA data considered stale.
        default_buffer_radius_km: Default buffer radius for monitoring.
        min_buffer_km: Minimum allowed buffer radius.
        max_buffer_km: Maximum allowed buffer radius.
        buffer_resolution_points: Points per buffer geometry circle.
        overlap_precision_m: Spatial overlap precision in meters.
        batch_max_size: Maximum batch processing size.
        batch_concurrency: Batch processing concurrency.
        batch_timeout_s: Batch processing timeout seconds.
        retention_years: Data retention years per EUDR Article 31.
        enable_provenance: Enable provenance tracking.
        genesis_hash: Genesis hash anchor for provenance chain.
        chain_algorithm: Hash algorithm for provenance chain.
        enable_metrics: Enable Prometheus metrics collection.
        metrics_prefix: Prometheus metrics prefix.
    """

    # -----------------------------------------------------------------------
    # Database settings
    # -----------------------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    pool_size: int = 20
    pool_timeout_s: int = 30
    pool_recycle_s: int = 3600

    # -----------------------------------------------------------------------
    # Redis settings
    # -----------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_s: int = 3600
    redis_key_prefix: str = "gl:eudr:pav:"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # WDPA data management
    # -----------------------------------------------------------------------
    wdpa_staleness_days: int = 60
    wdpa_auto_refresh: bool = True
    wdpa_min_confidence: str = "low"

    # -----------------------------------------------------------------------
    # Spatial analysis settings
    # -----------------------------------------------------------------------
    overlap_precision_m: int = 1
    spatial_srid: int = 4326
    default_buffer_radius_km: Decimal = Decimal("10")
    min_buffer_km: Decimal = Decimal("1")
    max_buffer_km: Decimal = Decimal("50")
    buffer_resolution_points: int = 64
    buffer_radii: List[int] = field(
        default_factory=lambda: [1, 5, 10, 25, 50]
    )

    # -----------------------------------------------------------------------
    # Risk scoring weights (must sum to 1.0)
    # -----------------------------------------------------------------------
    risk_weight_iucn_category: Decimal = Decimal("0.50")
    risk_weight_designation_level: Decimal = Decimal("0.20")
    risk_weight_management_gap: Decimal = Decimal("0.15")
    risk_weight_enforcement_gap: Decimal = Decimal("0.15")

    # -----------------------------------------------------------------------
    # Overlap type multipliers
    # -----------------------------------------------------------------------
    multiplier_inside: Decimal = Decimal("1.0")
    multiplier_partial: Decimal = Decimal("0.8")
    multiplier_boundary: Decimal = Decimal("0.6")
    multiplier_buffer: Decimal = Decimal("0.3")

    # -----------------------------------------------------------------------
    # Proximity risk weights (must sum to 1.0)
    # -----------------------------------------------------------------------
    proximity_weight_distance: Decimal = Decimal("0.40")
    proximity_weight_iucn: Decimal = Decimal("0.25")
    proximity_weight_designation: Decimal = Decimal("0.15")
    proximity_weight_multi_designation: Decimal = Decimal("0.10")
    proximity_weight_deforestation: Decimal = Decimal("0.10")

    # -----------------------------------------------------------------------
    # Alert thresholds
    # -----------------------------------------------------------------------
    alert_high_risk_radius_km: Decimal = Decimal("25")
    edd_critical_sla_days: int = _DEFAULT_EDD_CRITICAL_DAYS
    edd_severe_sla_days: int = _DEFAULT_EDD_SEVERE_DAYS

    # -----------------------------------------------------------------------
    # Compliance SLA deadlines (days)
    # -----------------------------------------------------------------------
    sla_investigation_days: int = _DEFAULT_SLA_INVESTIGATION_DAYS
    sla_remediation_plan_days: int = _DEFAULT_SLA_REMEDIATION_PLAN_DAYS
    sla_remediation_exec_days: int = _DEFAULT_SLA_REMEDIATION_EXEC_DAYS
    sla_verification_days: int = _DEFAULT_SLA_VERIFICATION_DAYS

    # -----------------------------------------------------------------------
    # Compliance settings
    # -----------------------------------------------------------------------
    market_restriction_threshold: str = "HIGH"

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "csv", "xlsx"]
    )
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )

    # -----------------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------------
    batch_max_size: int = 1000
    batch_concurrency: int = 8
    batch_timeout_s: int = 600

    # -----------------------------------------------------------------------
    # Data retention
    # -----------------------------------------------------------------------
    retention_years: int = 5

    # -----------------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-PAV-022-PROTECTED-AREA-VALIDATOR-GENESIS"
    chain_algorithm: str = "sha256"

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_pav_"

    # -----------------------------------------------------------------------
    # Rate limiting (requests per minute)
    # -----------------------------------------------------------------------
    rate_limit_anonymous: int = 10
    rate_limit_basic: int = 60
    rate_limit_standard: int = 300
    rate_limit_premium: int = 1000
    rate_limit_admin: int = 10000

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_log_level()
        self._validate_chain_algorithm()
        self._validate_spatial_buffer()
        self._validate_risk_weights()
        self._validate_proximity_weights()
        self._validate_compliance()
        self._validate_output_formats()
        self._validate_languages()
        self._validate_positive_integers()

        logger.info(
            f"ProtectedAreaValidatorConfig initialized: "
            f"default_buffer_km={self.default_buffer_radius_km}, "
            f"wdpa_staleness={self.wdpa_staleness_days}d, "
            f"pool_size={self.pool_size}"
        )

    def _validate_log_level(self) -> None:
        """Validate log_level is a recognized Python logging level."""
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level: {self.log_level}. "
                f"Must be one of {_VALID_LOG_LEVELS}"
            )

    def _validate_chain_algorithm(self) -> None:
        """Validate chain_algorithm is a supported hash algorithm."""
        if self.chain_algorithm not in _VALID_CHAIN_ALGORITHMS:
            raise ValueError(
                f"Invalid chain_algorithm: {self.chain_algorithm}. "
                f"Must be one of {_VALID_CHAIN_ALGORITHMS}"
            )

    def _validate_spatial_buffer(self) -> None:
        """Validate spatial buffer parameters."""
        if not (
            Decimal("0")
            < self.min_buffer_km
            <= self.default_buffer_radius_km
            <= self.max_buffer_km
        ):
            raise ValueError(
                "Buffer radii must satisfy 0 < min <= default <= max. "
                f"Got min={self.min_buffer_km}, "
                f"default={self.default_buffer_radius_km}, "
                f"max={self.max_buffer_km}"
            )
        if self.buffer_resolution_points < 4:
            raise ValueError(
                f"buffer_resolution_points must be >= 4, "
                f"got {self.buffer_resolution_points}"
            )

    def _validate_risk_weights(self) -> None:
        """Validate risk scoring weights are positive and sum to 1.0."""
        weights = [
            ("risk_weight_iucn_category", self.risk_weight_iucn_category),
            ("risk_weight_designation_level", self.risk_weight_designation_level),
            ("risk_weight_management_gap", self.risk_weight_management_gap),
            ("risk_weight_enforcement_gap", self.risk_weight_enforcement_gap),
        ]
        for name, w in weights:
            if w <= Decimal("0") or w > Decimal("1"):
                raise ValueError(
                    f"{name} must be between 0 (exclusive) and 1 (inclusive), "
                    f"got {w}"
                )
        total = sum(w for _, w in weights)
        if abs(total - Decimal("1")) > Decimal("0.001"):
            raise ValueError(
                f"Risk weights must sum to 1.0, got {total}"
            )

    def _validate_proximity_weights(self) -> None:
        """Validate proximity risk weights are positive and sum to 1.0."""
        weights = [
            ("proximity_weight_distance", self.proximity_weight_distance),
            ("proximity_weight_iucn", self.proximity_weight_iucn),
            ("proximity_weight_designation", self.proximity_weight_designation),
            ("proximity_weight_multi_designation", self.proximity_weight_multi_designation),
            ("proximity_weight_deforestation", self.proximity_weight_deforestation),
        ]
        for name, w in weights:
            if w < Decimal("0") or w > Decimal("1"):
                raise ValueError(
                    f"{name} must be between 0 and 1, got {w}"
                )
        total = sum(w for _, w in weights)
        if abs(total - Decimal("1")) > Decimal("0.001"):
            raise ValueError(
                f"Proximity weights must sum to 1.0, got {total}"
            )

    def _validate_compliance(self) -> None:
        """Validate compliance configuration parameters."""
        if self.market_restriction_threshold not in _VALID_MARKET_RESTRICTION_THRESHOLDS:
            raise ValueError(
                f"Invalid market_restriction_threshold: "
                f"{self.market_restriction_threshold}. "
                f"Must be one of {_VALID_MARKET_RESTRICTION_THRESHOLDS}"
            )

    def _validate_output_formats(self) -> None:
        """Validate output formats against allowed set."""
        for fmt in self.output_formats:
            if fmt not in _VALID_OUTPUT_FORMATS:
                raise ValueError(
                    f"Invalid output format: {fmt}. "
                    f"Must be one of {_VALID_OUTPUT_FORMATS}"
                )

    def _validate_languages(self) -> None:
        """Validate language settings against allowed set."""
        if self.default_language not in _VALID_REPORT_LANGUAGES:
            raise ValueError(
                f"Invalid default_language: {self.default_language}. "
                f"Must be one of {_VALID_REPORT_LANGUAGES}"
            )
        for lang in self.supported_languages:
            if lang not in _VALID_REPORT_LANGUAGES:
                raise ValueError(
                    f"Invalid supported language: {lang}. "
                    f"Must be one of {_VALID_REPORT_LANGUAGES}"
                )

    def _validate_positive_integers(self) -> None:
        """Validate fields that must be positive integers."""
        checks = [
            ("pool_size", self.pool_size),
            ("batch_max_size", self.batch_max_size),
            ("batch_concurrency", self.batch_concurrency),
            ("retention_years", self.retention_years),
            ("wdpa_staleness_days", self.wdpa_staleness_days),
        ]
        for name, val in checks:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

    @classmethod
    def from_env(cls) -> "ProtectedAreaValidatorConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_PAV_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            ProtectedAreaValidatorConfig instance with env overrides.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_PAV_DEFAULT_BUFFER_RADIUS_KM"] = "25"
            >>> cfg = ProtectedAreaValidatorConfig.from_env()
            >>> assert cfg.default_buffer_radius_km == Decimal("25")
        """
        kwargs: Dict[str, Any] = {}

        # Database
        if val := os.getenv(f"{_ENV_PREFIX}DATABASE_URL"):
            kwargs["database_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}POOL_SIZE"):
            kwargs["pool_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POOL_TIMEOUT_S"):
            kwargs["pool_timeout_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POOL_RECYCLE_S"):
            kwargs["pool_recycle_s"] = int(val)

        # Redis
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_URL"):
            kwargs["redis_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_TTL_S"):
            kwargs["redis_ttl_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_KEY_PREFIX"):
            kwargs["redis_key_prefix"] = val

        # Logging
        if val := os.getenv(f"{_ENV_PREFIX}LOG_LEVEL"):
            kwargs["log_level"] = val.upper()

        # WDPA
        if val := os.getenv(f"{_ENV_PREFIX}WDPA_STALENESS_DAYS"):
            kwargs["wdpa_staleness_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}WDPA_AUTO_REFRESH"):
            kwargs["wdpa_auto_refresh"] = val.lower() in ("true", "1", "yes")

        # Spatial
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_BUFFER_RADIUS_KM"):
            kwargs["default_buffer_radius_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MIN_BUFFER_KM"):
            kwargs["min_buffer_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_BUFFER_KM"):
            kwargs["max_buffer_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}BUFFER_RESOLUTION_POINTS"):
            kwargs["buffer_resolution_points"] = int(val)

        # Risk weights
        if val := os.getenv(f"{_ENV_PREFIX}RISK_WEIGHT_IUCN_CATEGORY"):
            kwargs["risk_weight_iucn_category"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}RISK_WEIGHT_DESIGNATION_LEVEL"):
            kwargs["risk_weight_designation_level"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}RISK_WEIGHT_MANAGEMENT_GAP"):
            kwargs["risk_weight_management_gap"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}RISK_WEIGHT_ENFORCEMENT_GAP"):
            kwargs["risk_weight_enforcement_gap"] = Decimal(val)

        # Overlap multipliers
        if val := os.getenv(f"{_ENV_PREFIX}MULTIPLIER_INSIDE"):
            kwargs["multiplier_inside"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MULTIPLIER_PARTIAL"):
            kwargs["multiplier_partial"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MULTIPLIER_BOUNDARY"):
            kwargs["multiplier_boundary"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MULTIPLIER_BUFFER"):
            kwargs["multiplier_buffer"] = Decimal(val)

        # Alert
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_HIGH_RISK_RADIUS_KM"):
            kwargs["alert_high_risk_radius_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}EDD_CRITICAL_SLA_DAYS"):
            kwargs["edd_critical_sla_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}EDD_SEVERE_SLA_DAYS"):
            kwargs["edd_severe_sla_days"] = int(val)

        # Compliance SLAs
        if val := os.getenv(f"{_ENV_PREFIX}SLA_INVESTIGATION_DAYS"):
            kwargs["sla_investigation_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SLA_REMEDIATION_PLAN_DAYS"):
            kwargs["sla_remediation_plan_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SLA_REMEDIATION_EXEC_DAYS"):
            kwargs["sla_remediation_exec_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SLA_VERIFICATION_DAYS"):
            kwargs["sla_verification_days"] = int(val)

        # Compliance
        if val := os.getenv(f"{_ENV_PREFIX}MARKET_RESTRICTION_THRESHOLD"):
            kwargs["market_restriction_threshold"] = val.upper()

        # Reporting
        if val := os.getenv(f"{_ENV_PREFIX}OUTPUT_FORMATS"):
            kwargs["output_formats"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_LANGUAGE"):
            kwargs["default_language"] = val
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_LANGUAGES"):
            kwargs["supported_languages"] = [x.strip() for x in val.split(",")]

        # Batch
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_MAX_SIZE"):
            kwargs["batch_max_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_CONCURRENCY"):
            kwargs["batch_concurrency"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_TIMEOUT_S"):
            kwargs["batch_timeout_s"] = int(val)

        # Retention
        if val := os.getenv(f"{_ENV_PREFIX}RETENTION_YEARS"):
            kwargs["retention_years"] = int(val)

        # Provenance
        if val := os.getenv(f"{_ENV_PREFIX}ENABLE_PROVENANCE"):
            kwargs["enable_provenance"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}GENESIS_HASH"):
            kwargs["genesis_hash"] = val
        if val := os.getenv(f"{_ENV_PREFIX}CHAIN_ALGORITHM"):
            kwargs["chain_algorithm"] = val

        # Metrics
        if val := os.getenv(f"{_ENV_PREFIX}ENABLE_METRICS"):
            kwargs["enable_metrics"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}METRICS_PREFIX"):
            kwargs["metrics_prefix"] = val

        # Rate limiting
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_ANONYMOUS"):
            kwargs["rate_limit_anonymous"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_BASIC"):
            kwargs["rate_limit_basic"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_STANDARD"):
            kwargs["rate_limit_standard"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_PREMIUM"):
            kwargs["rate_limit_premium"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_ADMIN"):
            kwargs["rate_limit_admin"] = int(val)

        return cls(**kwargs)

    def to_dict(self, redact_secrets: bool = True) -> Dict[str, Any]:
        """Export configuration as a dictionary.

        Args:
            redact_secrets: If True, redact sensitive fields.

        Returns:
            Dictionary representation of configuration.
        """
        data: Dict[str, Any] = {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "redis_url": self.redis_url,
            "redis_key_prefix": self.redis_key_prefix,
            "log_level": self.log_level,
            "wdpa_staleness_days": self.wdpa_staleness_days,
            "default_buffer_radius_km": str(self.default_buffer_radius_km),
            "min_buffer_km": str(self.min_buffer_km),
            "max_buffer_km": str(self.max_buffer_km),
            "buffer_resolution_points": self.buffer_resolution_points,
            "risk_weight_iucn_category": str(self.risk_weight_iucn_category),
            "risk_weight_designation_level": str(self.risk_weight_designation_level),
            "risk_weight_management_gap": str(self.risk_weight_management_gap),
            "risk_weight_enforcement_gap": str(self.risk_weight_enforcement_gap),
            "batch_max_size": self.batch_max_size,
            "retention_years": self.retention_years,
            "enable_provenance": self.enable_provenance,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
        }
        if redact_secrets:
            if "://" in data["database_url"]:
                data["database_url"] = "REDACTED"
            if "://" in data["redis_url"]:
                data["redis_url"] = "REDACTED"
        return data


# ---------------------------------------------------------------------------
# Thread-safe singleton pattern (double-checked locking)
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[ProtectedAreaValidatorConfig] = None


def get_config() -> ProtectedAreaValidatorConfig:
    """Get the global ProtectedAreaValidatorConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance.

    Returns:
        ProtectedAreaValidatorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = ProtectedAreaValidatorConfig.from_env()
    return _global_config


def set_config(config: ProtectedAreaValidatorConfig) -> None:
    """Set the global ProtectedAreaValidatorConfig singleton instance.

    Args:
        config: ProtectedAreaValidatorConfig instance to set as global.
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global ProtectedAreaValidatorConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.
    """
    global _global_config
    with _config_lock:
        _global_config = None
