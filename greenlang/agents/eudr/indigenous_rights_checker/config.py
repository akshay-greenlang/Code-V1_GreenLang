# -*- coding: utf-8 -*-
"""
Indigenous Rights Checker Configuration - AGENT-EUDR-021

Centralized configuration for the Indigenous Rights Checker Agent covering:
- Database and cache connection settings (PostgreSQL + PostGIS, Redis)
- Territory database settings: data sources (LandMark, RAISG, FUNAI,
  BPN/AMAN, ACHPR, national registries), freshness thresholds, buffer zones
- FPIC verification: scoring weights for 10-element checklist, temporal
  compliance minimum lead time (90 days), coercion detection thresholds,
  country-specific rule sets, validity period (5 years), renewal lead times
- Overlap detection: buffer zones (inner=5km, outer=25km), batch size
  (10,000 plots), risk scoring weights (overlap_type=0.40,
  territory_legal_status=0.20, community_population=0.10,
  conflict_history=0.15, country_rights_framework=0.15)
- Community consultation: SLA timelines per stage, grievance SLA
  (acknowledge=5d, investigate=30d, resolve=90d)
- Violation monitoring: source list (10+), severity scoring weights,
  deduplication window (7 days), alert thresholds
- FPIC workflow: stage SLA timelines, escalation levels, validity period
- Compliance reporting: output formats, languages, report retention
- Provenance, metrics, rate limiting, batch processing settings

All settings can be overridden via environment variables with the
``GL_EUDR_IRC_`` prefix (e.g. ``GL_EUDR_IRC_DATABASE_URL``,
``GL_EUDR_IRC_FPIC_VALIDITY_YEARS``).

Example:
    >>> from greenlang.agents.eudr.indigenous_rights_checker.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.fpic_validity_years, cfg.inner_buffer_km)
    5 5.0

    >>> from greenlang.agents.eudr.indigenous_rights_checker.config import (
    ...     set_config, reset_config, IndigenousRightsCheckerConfig,
    ... )
    >>> set_config(IndigenousRightsCheckerConfig(inner_buffer_km=10.0))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_IRC_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid chain hash algorithms
# ---------------------------------------------------------------------------

_VALID_CHAIN_ALGORITHMS = frozenset({"sha256", "sha384", "sha512"})

# ---------------------------------------------------------------------------
# Valid output formats
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "csv", "xlsx"})

# ---------------------------------------------------------------------------
# Valid report languages
# ---------------------------------------------------------------------------

_VALID_REPORT_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})

# ---------------------------------------------------------------------------
# Territory data sources
# ---------------------------------------------------------------------------

_VALID_TERRITORY_SOURCES = frozenset({
    "landmark",
    "raisg",
    "funai",
    "bpn_aman",
    "achpr",
    "national_registry",
})

# ---------------------------------------------------------------------------
# Violation monitoring sources
# ---------------------------------------------------------------------------

_VALID_VIOLATION_SOURCES = frozenset({
    "iwgia",
    "cultural_survival",
    "forest_peoples_programme",
    "amnesty_international",
    "human_rights_watch",
    "national_human_rights_commission",
    "iachr",
    "achpr",
    "ohchr",
    "judicial_database",
    "media_monitoring",
})

# ---------------------------------------------------------------------------
# Default FPIC scoring weights (10 elements, sum = 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_FPIC_WEIGHTS: Dict[str, float] = {
    "community_identification": 0.10,
    "information_disclosure": 0.15,
    "prior_timing": 0.10,
    "consultation_process": 0.15,
    "community_representation": 0.10,
    "consent_record": 0.15,
    "absence_of_coercion": 0.10,
    "agreement_documentation": 0.05,
    "benefit_sharing": 0.05,
    "monitoring_provisions": 0.05,
}

# ---------------------------------------------------------------------------
# Default overlap risk scoring weights (sum = 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_OVERLAP_RISK_WEIGHTS: Dict[str, float] = {
    "overlap_type": 0.40,
    "territory_legal_status": 0.20,
    "community_population": 0.10,
    "conflict_history": 0.15,
    "country_rights_framework": 0.15,
}

# ---------------------------------------------------------------------------
# Default violation severity weights (sum = 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_VIOLATION_SEVERITY_WEIGHTS: Dict[str, float] = {
    "violation_type": 0.30,
    "spatial_proximity": 0.25,
    "community_population": 0.15,
    "legal_framework_gap": 0.15,
    "media_coverage": 0.15,
}

# ---------------------------------------------------------------------------
# Default FPIC workflow SLA timelines (days per stage)
# ---------------------------------------------------------------------------

_DEFAULT_WORKFLOW_SLA_DAYS: Dict[str, int] = {
    "identification": 14,
    "information_disclosure": 30,
    "consultation": 60,
    "consent_decision": 30,
    "agreement": 30,
    "implementation": 365,
    "monitoring": 365,
}

# ---------------------------------------------------------------------------
# Default grievance SLA timelines (days)
# ---------------------------------------------------------------------------

_DEFAULT_GRIEVANCE_SLA_DAYS: Dict[str, int] = {
    "acknowledge": 5,
    "investigate": 30,
    "resolve": 90,
}

# ---------------------------------------------------------------------------
# FPIC workflow escalation thresholds (days overdue)
# ---------------------------------------------------------------------------

_DEFAULT_ESCALATION_THRESHOLDS: Dict[str, int] = {
    "level_1": 7,
    "level_2": 14,
    "level_3": 30,
}

# ---------------------------------------------------------------------------
# FPIC template languages
# ---------------------------------------------------------------------------

_VALID_FPIC_TEMPLATE_LANGUAGES = frozenset({
    "en", "fr", "de", "es", "pt", "id", "qu", "sw",
})


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class IndigenousRightsCheckerConfig:
    """Configuration for the Indigenous Rights Checker Agent (AGENT-EUDR-021).

    This dataclass encapsulates all configuration settings for indigenous
    territory database management, FPIC verification, land rights overlap
    detection, community consultation tracking, rights violation monitoring,
    indigenous community registry, FPIC workflow management, and compliance
    reporting. All settings have sensible defaults aligned with EUDR
    requirements and can be overridden via environment variables with the
    GL_EUDR_IRC_ prefix.

    Attributes:
        database_url: PostgreSQL connection URL (PostGIS-enabled).
        pool_size: Connection pool size.
        pool_timeout_s: Connection pool timeout seconds.
        pool_recycle_s: Connection pool recycle seconds.
        redis_url: Redis connection URL.
        redis_ttl_s: Redis cache TTL seconds.
        redis_key_prefix: Redis key prefix.
        log_level: Logging level.
        inner_buffer_km: Inner proximity buffer in kilometers (default 5km).
        outer_buffer_km: Outer proximity buffer in kilometers (default 25km).
        buffer_polygon_points: Points in buffer polygon approximation.
        territory_staleness_months: Max age before territory data is stale.
        fpic_weights: FPIC scoring element weights (10 elements, sum=1.0).
        fpic_validity_years: FPIC consent validity period in years.
        fpic_renewal_lead_days: Days before expiry to trigger renewal alerts.
        fpic_min_lead_time_days: Minimum days consent must precede production.
        fpic_coercion_min_days: Minimum days between disclosure and consent.
        overlap_risk_weights: Overlap risk scoring factor weights.
        violation_severity_weights: Violation severity scoring weights.
        violation_dedup_window_days: Deduplication window for violations.
        workflow_sla_days: SLA timelines per FPIC workflow stage (days).
        grievance_sla_days: SLA timelines for grievance stages (days).
        escalation_thresholds_days: Overdue days triggering escalation levels.
        batch_max_size: Maximum batch size for overlap screening.
        batch_concurrency: Concurrent workers for batch processing.
        batch_timeout_s: Batch operation timeout seconds.
        retention_years: Data retention years (EUDR Article 31 = 5).
        output_formats: Supported report output formats.
        default_language: Default report language.
        supported_languages: Supported report languages.
        enable_provenance: Enable SHA-256 provenance tracking.
        genesis_hash: Genesis hash anchor for provenance chain.
        chain_algorithm: Hash algorithm for provenance chain.
        enable_metrics: Enable Prometheus metrics.
        metrics_prefix: Prometheus metrics prefix.
        rate_limit_anonymous: Anonymous rate limit (requests/minute).
        rate_limit_basic: Basic rate limit (requests/minute).
        rate_limit_standard: Standard rate limit (requests/minute).
        rate_limit_premium: Premium rate limit (requests/minute).
        rate_limit_admin: Admin rate limit (requests/minute).
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
    redis_key_prefix: str = "gl:eudr:irc:"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # Territory database settings
    # -----------------------------------------------------------------------
    territory_staleness_months: int = 12
    point_buffer_radius_km: float = 10.0

    # -----------------------------------------------------------------------
    # Overlap detection settings (PostGIS buffer zones)
    # -----------------------------------------------------------------------
    inner_buffer_km: float = 5.0
    outer_buffer_km: float = 25.0
    buffer_polygon_points: int = 64

    # -----------------------------------------------------------------------
    # FPIC verification scoring weights (10 elements, sum=1.0)
    # -----------------------------------------------------------------------
    fpic_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_FPIC_WEIGHTS)
    )

    # -----------------------------------------------------------------------
    # FPIC temporal settings
    # -----------------------------------------------------------------------
    fpic_validity_years: int = 5
    fpic_renewal_lead_days: List[int] = field(
        default_factory=lambda: [180, 90, 30]
    )
    fpic_min_lead_time_days: int = 90
    fpic_coercion_min_days: int = 30

    # -----------------------------------------------------------------------
    # Overlap risk scoring weights (sum=1.0)
    # -----------------------------------------------------------------------
    overlap_risk_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_OVERLAP_RISK_WEIGHTS)
    )

    # -----------------------------------------------------------------------
    # Violation severity scoring weights (sum=1.0)
    # -----------------------------------------------------------------------
    violation_severity_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_VIOLATION_SEVERITY_WEIGHTS)
    )
    violation_dedup_window_days: int = 7

    # -----------------------------------------------------------------------
    # FPIC workflow SLA timelines (days per stage)
    # -----------------------------------------------------------------------
    workflow_sla_days: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_WORKFLOW_SLA_DAYS)
    )

    # -----------------------------------------------------------------------
    # Grievance SLA timelines (days)
    # -----------------------------------------------------------------------
    grievance_sla_days: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_GRIEVANCE_SLA_DAYS)
    )

    # -----------------------------------------------------------------------
    # Escalation thresholds (days overdue)
    # -----------------------------------------------------------------------
    escalation_thresholds_days: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_ESCALATION_THRESHOLDS)
    )

    # -----------------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------------
    batch_max_size: int = 10000
    batch_concurrency: int = 8
    batch_timeout_s: int = 300

    # -----------------------------------------------------------------------
    # Data retention (EUDR Article 31 requires 5 years)
    # -----------------------------------------------------------------------
    retention_years: int = 5

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "csv"]
    )
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )

    # -----------------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-IRC-021-INDIGENOUS-RIGHTS-CHECKER-GENESIS"
    chain_algorithm: str = "sha256"

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_irc_"

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
        self._validate_buffer_zones()
        self._validate_fpic_weights()
        self._validate_overlap_risk_weights()
        self._validate_violation_severity_weights()
        self._validate_temporal_settings()
        self._validate_sla_settings()
        self._validate_positive_integers()
        self._validate_output_formats()
        self._validate_languages()

        logger.info(
            "IndigenousRightsCheckerConfig initialized: "
            f"inner_buffer={self.inner_buffer_km}km, "
            f"outer_buffer={self.outer_buffer_km}km, "
            f"fpic_validity={self.fpic_validity_years}yr, "
            f"batch_max={self.batch_max_size}"
        )

    # -----------------------------------------------------------------------
    # Validation methods
    # -----------------------------------------------------------------------

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

    def _validate_buffer_zones(self) -> None:
        """Validate buffer zone settings are positive and ordered."""
        if self.inner_buffer_km <= 0:
            raise ValueError(
                f"inner_buffer_km must be > 0, got {self.inner_buffer_km}"
            )
        if self.outer_buffer_km <= 0:
            raise ValueError(
                f"outer_buffer_km must be > 0, got {self.outer_buffer_km}"
            )
        if self.inner_buffer_km >= self.outer_buffer_km:
            raise ValueError(
                f"inner_buffer_km ({self.inner_buffer_km}) must be < "
                f"outer_buffer_km ({self.outer_buffer_km})"
            )
        if self.buffer_polygon_points < 8:
            raise ValueError(
                f"buffer_polygon_points must be >= 8, "
                f"got {self.buffer_polygon_points}"
            )

    def _validate_fpic_weights(self) -> None:
        """Validate FPIC scoring weights sum to 1.0 and all are positive."""
        required_elements = set(_DEFAULT_FPIC_WEIGHTS.keys())
        for elem in required_elements:
            if elem not in self.fpic_weights:
                raise ValueError(
                    f"Missing FPIC weight element: {elem}. "
                    f"Required: {required_elements}"
                )
        for elem, weight in self.fpic_weights.items():
            if elem not in required_elements:
                raise ValueError(
                    f"Invalid FPIC element: {elem}. "
                    f"Must be one of {required_elements}"
                )
            if weight < 0.0 or weight > 1.0:
                raise ValueError(
                    f"FPIC weight for {elem} must be between 0.0 and 1.0, "
                    f"got {weight}"
                )
        total = sum(self.fpic_weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"FPIC weights must sum to 1.0, got {total:.4f}"
            )

    def _validate_overlap_risk_weights(self) -> None:
        """Validate overlap risk scoring weights sum to 1.0."""
        required_factors = set(_DEFAULT_OVERLAP_RISK_WEIGHTS.keys())
        for factor in required_factors:
            if factor not in self.overlap_risk_weights:
                raise ValueError(
                    f"Missing overlap risk factor: {factor}. "
                    f"Required: {required_factors}"
                )
        for factor, weight in self.overlap_risk_weights.items():
            if weight < 0.0 or weight > 1.0:
                raise ValueError(
                    f"Overlap risk weight for {factor} must be between "
                    f"0.0 and 1.0, got {weight}"
                )
        total = sum(self.overlap_risk_weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Overlap risk weights must sum to 1.0, got {total:.4f}"
            )

    def _validate_violation_severity_weights(self) -> None:
        """Validate violation severity weights sum to 1.0."""
        required_factors = set(_DEFAULT_VIOLATION_SEVERITY_WEIGHTS.keys())
        for factor in required_factors:
            if factor not in self.violation_severity_weights:
                raise ValueError(
                    f"Missing violation severity factor: {factor}. "
                    f"Required: {required_factors}"
                )
        for factor, weight in self.violation_severity_weights.items():
            if weight < 0.0 or weight > 1.0:
                raise ValueError(
                    f"Violation severity weight for {factor} must be "
                    f"between 0.0 and 1.0, got {weight}"
                )
        total = sum(self.violation_severity_weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Violation severity weights must sum to 1.0, "
                f"got {total:.4f}"
            )

    def _validate_temporal_settings(self) -> None:
        """Validate FPIC temporal compliance settings."""
        if self.fpic_validity_years < 1:
            raise ValueError(
                f"fpic_validity_years must be >= 1, "
                f"got {self.fpic_validity_years}"
            )
        if self.fpic_min_lead_time_days < 1:
            raise ValueError(
                f"fpic_min_lead_time_days must be >= 1, "
                f"got {self.fpic_min_lead_time_days}"
            )
        if self.fpic_coercion_min_days < 1:
            raise ValueError(
                f"fpic_coercion_min_days must be >= 1, "
                f"got {self.fpic_coercion_min_days}"
            )
        for lead_day in self.fpic_renewal_lead_days:
            if lead_day < 1:
                raise ValueError(
                    f"fpic_renewal_lead_days entries must be >= 1, "
                    f"got {lead_day}"
                )

    def _validate_sla_settings(self) -> None:
        """Validate SLA timeline settings."""
        for stage, days in self.workflow_sla_days.items():
            if days < 1:
                raise ValueError(
                    f"workflow_sla_days[{stage}] must be >= 1, got {days}"
                )
        for stage, days in self.grievance_sla_days.items():
            if days < 1:
                raise ValueError(
                    f"grievance_sla_days[{stage}] must be >= 1, got {days}"
                )

    def _validate_positive_integers(self) -> None:
        """Validate fields that must be positive integers."""
        checks = [
            ("pool_size", self.pool_size),
            ("batch_max_size", self.batch_max_size),
            ("batch_concurrency", self.batch_concurrency),
            ("retention_years", self.retention_years),
            ("territory_staleness_months", self.territory_staleness_months),
        ]
        for name, val in checks:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

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

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "IndigenousRightsCheckerConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_IRC_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            IndigenousRightsCheckerConfig instance with env overrides.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_IRC_INNER_BUFFER_KM"] = "10.0"
            >>> cfg = IndigenousRightsCheckerConfig.from_env()
            >>> assert cfg.inner_buffer_km == 10.0
        """
        kwargs: Dict[str, Any] = {}

        # Database settings
        if val := os.getenv(f"{_ENV_PREFIX}DATABASE_URL"):
            kwargs["database_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}POOL_SIZE"):
            kwargs["pool_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POOL_TIMEOUT_S"):
            kwargs["pool_timeout_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POOL_RECYCLE_S"):
            kwargs["pool_recycle_s"] = int(val)

        # Redis settings
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_URL"):
            kwargs["redis_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_TTL_S"):
            kwargs["redis_ttl_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_KEY_PREFIX"):
            kwargs["redis_key_prefix"] = val

        # Logging
        if val := os.getenv(f"{_ENV_PREFIX}LOG_LEVEL"):
            kwargs["log_level"] = val.upper()

        # Territory settings
        if val := os.getenv(f"{_ENV_PREFIX}TERRITORY_STALENESS_MONTHS"):
            kwargs["territory_staleness_months"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POINT_BUFFER_RADIUS_KM"):
            kwargs["point_buffer_radius_km"] = float(val)

        # Buffer zone settings
        if val := os.getenv(f"{_ENV_PREFIX}INNER_BUFFER_KM"):
            kwargs["inner_buffer_km"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}OUTER_BUFFER_KM"):
            kwargs["outer_buffer_km"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}BUFFER_POLYGON_POINTS"):
            kwargs["buffer_polygon_points"] = int(val)

        # FPIC temporal settings
        if val := os.getenv(f"{_ENV_PREFIX}FPIC_VALIDITY_YEARS"):
            kwargs["fpic_validity_years"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}FPIC_MIN_LEAD_TIME_DAYS"):
            kwargs["fpic_min_lead_time_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}FPIC_COERCION_MIN_DAYS"):
            kwargs["fpic_coercion_min_days"] = int(val)

        # Violation settings
        if val := os.getenv(f"{_ENV_PREFIX}VIOLATION_DEDUP_WINDOW_DAYS"):
            kwargs["violation_dedup_window_days"] = int(val)

        # Batch processing
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_MAX_SIZE"):
            kwargs["batch_max_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_CONCURRENCY"):
            kwargs["batch_concurrency"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_TIMEOUT_S"):
            kwargs["batch_timeout_s"] = int(val)

        # Data retention
        if val := os.getenv(f"{_ENV_PREFIX}RETENTION_YEARS"):
            kwargs["retention_years"] = int(val)

        # Reporting
        if val := os.getenv(f"{_ENV_PREFIX}OUTPUT_FORMATS"):
            kwargs["output_formats"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_LANGUAGE"):
            kwargs["default_language"] = val
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_LANGUAGES"):
            kwargs["supported_languages"] = [
                x.strip() for x in val.split(",")
            ]

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

        Example:
            >>> cfg = IndigenousRightsCheckerConfig()
            >>> d = cfg.to_dict(redact_secrets=True)
            >>> assert "REDACTED" in d["database_url"]
        """
        data: Dict[str, Any] = {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "pool_timeout_s": self.pool_timeout_s,
            "pool_recycle_s": self.pool_recycle_s,
            "redis_url": self.redis_url,
            "redis_ttl_s": self.redis_ttl_s,
            "redis_key_prefix": self.redis_key_prefix,
            "log_level": self.log_level,
            "territory_staleness_months": self.territory_staleness_months,
            "point_buffer_radius_km": self.point_buffer_radius_km,
            "inner_buffer_km": self.inner_buffer_km,
            "outer_buffer_km": self.outer_buffer_km,
            "buffer_polygon_points": self.buffer_polygon_points,
            "fpic_weights": self.fpic_weights,
            "fpic_validity_years": self.fpic_validity_years,
            "fpic_renewal_lead_days": self.fpic_renewal_lead_days,
            "fpic_min_lead_time_days": self.fpic_min_lead_time_days,
            "fpic_coercion_min_days": self.fpic_coercion_min_days,
            "overlap_risk_weights": self.overlap_risk_weights,
            "violation_severity_weights": self.violation_severity_weights,
            "violation_dedup_window_days": self.violation_dedup_window_days,
            "workflow_sla_days": self.workflow_sla_days,
            "grievance_sla_days": self.grievance_sla_days,
            "escalation_thresholds_days": self.escalation_thresholds_days,
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            "retention_years": self.retention_years,
            "output_formats": self.output_formats,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            "chain_algorithm": self.chain_algorithm,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "rate_limit_anonymous": self.rate_limit_anonymous,
            "rate_limit_basic": self.rate_limit_basic,
            "rate_limit_standard": self.rate_limit_standard,
            "rate_limit_premium": self.rate_limit_premium,
            "rate_limit_admin": self.rate_limit_admin,
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
_global_config: Optional[IndigenousRightsCheckerConfig] = None


def get_config() -> IndigenousRightsCheckerConfig:
    """Get the global IndigenousRightsCheckerConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance.

    Returns:
        IndigenousRightsCheckerConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> assert cfg.inner_buffer_km == 5.0
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = IndigenousRightsCheckerConfig.from_env()
    return _global_config


def set_config(config: IndigenousRightsCheckerConfig) -> None:
    """Set the global IndigenousRightsCheckerConfig singleton instance.

    Used for testing and programmatic configuration override.

    Args:
        config: IndigenousRightsCheckerConfig instance to set as global.

    Example:
        >>> set_config(IndigenousRightsCheckerConfig(inner_buffer_km=10.0))
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global IndigenousRightsCheckerConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_config()
        >>> # Next get_config() call will re-initialize from environment
    """
    global _global_config
    with _config_lock:
        _global_config = None
