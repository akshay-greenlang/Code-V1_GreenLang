# -*- coding: utf-8 -*-
"""
Third-Party Audit Manager Configuration - AGENT-EUDR-024

Centralized configuration for the Third-Party Audit Manager Agent covering:
- Database and cache connection settings (PostgreSQL, Redis) with configurable
  pool sizes, timeouts, and key prefixes using ``gl_eudr_tam_`` namespace
- Audit planning settings: risk weights for composite priority formula
  (country_risk 0.25, supplier_risk 0.25, nc_history 0.20, certification
  gap 0.15, deforestation_alert 0.15), frequency tiers (HIGH=quarterly,
  STANDARD=semi-annual, LOW=annual), recency multiplier cap (2.0)
- Auditor registry: qualification expiry warnings (60 days), rotation
  requirements, performance thresholds, CPD compliance tracking
- Audit execution: checklist version management, evidence file size limits
  (100 MB per doc, 5 GB per audit), sampling plan ISO 19011 Annex A
- Non-conformance: severity SLA deadlines (critical 30d, major 90d,
  minor 365d), root cause analysis frameworks (5-Whys, Ishikawa)
- CAR management: 9-phase lifecycle, escalation stages (4 levels),
  SLA warning thresholds (75%, 90%), overdue grace periods
- Certification schemes: FSC/PEFC/RSPO/RA/ISCC integration, certificate
  sync interval (24h), scheme coverage matrix
- Report generation: ISO 19011:2018 compliance, 5 formats (PDF/JSON/HTML/
  XLSX/XML), 5 languages (EN/FR/DE/ES/PT), report retention (5 years)
- Competent authority: 27 EU Member State profiles, response SLA defaults
  (30 days standard, 5 days urgent), inspection coordination
- Analytics: dashboard refresh intervals, KPI thresholds, trend windows

All settings can be overridden via environment variables with the
``GL_EUDR_TAM_`` prefix (e.g. ``GL_EUDR_TAM_DATABASE_URL``,
``GL_EUDR_TAM_CRITICAL_SLA_DAYS``).

Example:
    >>> from greenlang.agents.eudr.third_party_audit_manager.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.critical_sla_days, cfg.country_risk_weight)
    30 0.25

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
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

_ENV_PREFIX = "GL_EUDR_TAM_"

# ---------------------------------------------------------------------------
# Valid constants
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

_VALID_CHAIN_ALGORITHMS = frozenset({"sha256", "sha384", "sha512"})

_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "xlsx", "xml"})

_VALID_REPORT_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})

_VALID_AUDIT_SCOPES = frozenset({"full", "targeted", "surveillance", "unscheduled"})

_VALID_AUDIT_MODALITIES = frozenset({"on_site", "remote", "hybrid", "unannounced"})

_VALID_NC_SEVERITIES = frozenset({"critical", "major", "minor", "observation"})

_VALID_FREQUENCY_TIERS = frozenset({"HIGH", "STANDARD", "LOW"})

_VALID_CERTIFICATION_SCHEMES = frozenset({
    "fsc", "pefc", "rspo", "rainforest_alliance", "iscc",
})

_VALID_MARKET_RESTRICTION_THRESHOLDS = frozenset({
    "CRITICAL", "HIGH", "MEDIUM", "LOW",
})

# ---------------------------------------------------------------------------
# Default risk weights for composite audit priority formula (sum = 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_COUNTRY_RISK_WEIGHT = Decimal("0.25")
_DEFAULT_SUPPLIER_RISK_WEIGHT = Decimal("0.25")
_DEFAULT_NC_HISTORY_WEIGHT = Decimal("0.20")
_DEFAULT_CERTIFICATION_GAP_WEIGHT = Decimal("0.15")
_DEFAULT_DEFORESTATION_ALERT_WEIGHT = Decimal("0.15")

# ---------------------------------------------------------------------------
# Default NC severity scoring weights (for risk impact score)
# ---------------------------------------------------------------------------

_DEFAULT_CRITICAL_NC_WEIGHT: int = 30
_DEFAULT_MAJOR_NC_WEIGHT: int = 15
_DEFAULT_MINOR_NC_WEIGHT: int = 5

# ---------------------------------------------------------------------------
# Default SLA deadlines (days)
# ---------------------------------------------------------------------------

_DEFAULT_CRITICAL_SLA_DAYS: int = 30
_DEFAULT_MAJOR_SLA_DAYS: int = 90
_DEFAULT_MINOR_SLA_DAYS: int = 365

# ---------------------------------------------------------------------------
# Default CAR SLA sub-deadlines (days)
# ---------------------------------------------------------------------------

_DEFAULT_CRITICAL_ACKNOWLEDGE_DAYS: int = 3
_DEFAULT_CRITICAL_RCA_DAYS: int = 7
_DEFAULT_CRITICAL_CAP_DAYS: int = 14

_DEFAULT_MAJOR_ACKNOWLEDGE_DAYS: int = 7
_DEFAULT_MAJOR_RCA_DAYS: int = 14
_DEFAULT_MAJOR_CAP_DAYS: int = 30

_DEFAULT_MINOR_ACKNOWLEDGE_DAYS: int = 14
_DEFAULT_MINOR_RCA_DAYS: int = 30
_DEFAULT_MINOR_CAP_DAYS: int = 60

# ---------------------------------------------------------------------------
# Default escalation thresholds
# ---------------------------------------------------------------------------

_DEFAULT_ESCALATION_STAGE1_PCT = Decimal("0.75")
_DEFAULT_ESCALATION_STAGE2_PCT = Decimal("0.90")
_DEFAULT_MAX_ESCALATION_LEVELS: int = 4

# ---------------------------------------------------------------------------
# Default frequency tier thresholds
# ---------------------------------------------------------------------------

_DEFAULT_HIGH_THRESHOLD = Decimal("70")
_DEFAULT_STANDARD_THRESHOLD = Decimal("40")
_DEFAULT_RECENCY_CAP = Decimal("2.0")

# ---------------------------------------------------------------------------
# Default evidence limits
# ---------------------------------------------------------------------------

_DEFAULT_MAX_EVIDENCE_SIZE_MB: int = 100
_DEFAULT_MAX_EVIDENCE_PACKAGE_GB: int = 5

# ---------------------------------------------------------------------------
# Default certification sync interval
# ---------------------------------------------------------------------------

_DEFAULT_CERT_SYNC_INTERVAL_HOURS: int = 24
_DEFAULT_ACCREDITATION_EXPIRY_WARNING_DAYS: int = 60

# ---------------------------------------------------------------------------
# Default competent authority SLA
# ---------------------------------------------------------------------------

_DEFAULT_AUTHORITY_RESPONSE_DAYS: int = 30
_DEFAULT_AUTHORITY_URGENT_RESPONSE_DAYS: int = 5

# ---------------------------------------------------------------------------
# Default report generation
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_GENERATION_TIMEOUT_S: int = 30
_DEFAULT_REPORT_RETENTION_YEARS: int = 5

# ---------------------------------------------------------------------------
# Default analytics
# ---------------------------------------------------------------------------

_DEFAULT_ANALYTICS_REFRESH_MINUTES: int = 60
_DEFAULT_DASHBOARD_LOAD_TIMEOUT_S: int = 3


@dataclass
class ThirdPartyAuditManagerConfig:
    """Configuration for the Third-Party Audit Manager Agent (AGENT-EUDR-024).

    This dataclass encapsulates all configuration settings for audit planning
    and scheduling, auditor qualification tracking, audit execution monitoring,
    non-conformance classification, CAR lifecycle management, certification
    scheme integration, ISO 19011 report generation, competent authority
    liaison, and audit analytics. All settings have sensible defaults aligned
    with EUDR requirements and ISO 19011:2018 / ISO/IEC 17065:2012 /
    ISO/IEC 17021-1:2015 standards.

    All settings can be overridden via environment variables with the
    ``GL_EUDR_TAM_`` prefix.
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
    redis_key_prefix: str = "gl:eudr:tam:"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # Audit planning: risk weights (must sum to 1.0)
    # -----------------------------------------------------------------------
    country_risk_weight: Decimal = _DEFAULT_COUNTRY_RISK_WEIGHT
    supplier_risk_weight: Decimal = _DEFAULT_SUPPLIER_RISK_WEIGHT
    nc_history_weight: Decimal = _DEFAULT_NC_HISTORY_WEIGHT
    certification_gap_weight: Decimal = _DEFAULT_CERTIFICATION_GAP_WEIGHT
    deforestation_alert_weight: Decimal = _DEFAULT_DEFORESTATION_ALERT_WEIGHT

    # -----------------------------------------------------------------------
    # Audit planning: frequency tier thresholds
    # -----------------------------------------------------------------------
    high_risk_threshold: Decimal = _DEFAULT_HIGH_THRESHOLD
    standard_risk_threshold: Decimal = _DEFAULT_STANDARD_THRESHOLD
    recency_multiplier_cap: Decimal = _DEFAULT_RECENCY_CAP

    # -----------------------------------------------------------------------
    # Audit planning: NC severity weights for NC_History_Score
    # -----------------------------------------------------------------------
    critical_nc_weight: int = _DEFAULT_CRITICAL_NC_WEIGHT
    major_nc_weight: int = _DEFAULT_MAJOR_NC_WEIGHT
    minor_nc_weight: int = _DEFAULT_MINOR_NC_WEIGHT

    # -----------------------------------------------------------------------
    # NC SLA deadlines (days)
    # -----------------------------------------------------------------------
    critical_sla_days: int = _DEFAULT_CRITICAL_SLA_DAYS
    major_sla_days: int = _DEFAULT_MAJOR_SLA_DAYS
    minor_sla_days: int = _DEFAULT_MINOR_SLA_DAYS

    # -----------------------------------------------------------------------
    # CAR sub-deadlines (days from CAR issuance)
    # -----------------------------------------------------------------------
    critical_acknowledge_days: int = _DEFAULT_CRITICAL_ACKNOWLEDGE_DAYS
    critical_rca_days: int = _DEFAULT_CRITICAL_RCA_DAYS
    critical_cap_days: int = _DEFAULT_CRITICAL_CAP_DAYS

    major_acknowledge_days: int = _DEFAULT_MAJOR_ACKNOWLEDGE_DAYS
    major_rca_days: int = _DEFAULT_MAJOR_RCA_DAYS
    major_cap_days: int = _DEFAULT_MAJOR_CAP_DAYS

    minor_acknowledge_days: int = _DEFAULT_MINOR_ACKNOWLEDGE_DAYS
    minor_rca_days: int = _DEFAULT_MINOR_RCA_DAYS
    minor_cap_days: int = _DEFAULT_MINOR_CAP_DAYS

    # -----------------------------------------------------------------------
    # Escalation settings
    # -----------------------------------------------------------------------
    escalation_stage1_pct: Decimal = _DEFAULT_ESCALATION_STAGE1_PCT
    escalation_stage2_pct: Decimal = _DEFAULT_ESCALATION_STAGE2_PCT
    max_escalation_levels: int = _DEFAULT_MAX_ESCALATION_LEVELS

    # -----------------------------------------------------------------------
    # Evidence limits
    # -----------------------------------------------------------------------
    max_evidence_size_mb: int = _DEFAULT_MAX_EVIDENCE_SIZE_MB
    max_evidence_package_gb: int = _DEFAULT_MAX_EVIDENCE_PACKAGE_GB

    # -----------------------------------------------------------------------
    # Certification scheme settings
    # -----------------------------------------------------------------------
    cert_sync_interval_hours: int = _DEFAULT_CERT_SYNC_INTERVAL_HOURS
    accreditation_expiry_warning_days: int = _DEFAULT_ACCREDITATION_EXPIRY_WARNING_DAYS
    enabled_schemes: List[str] = field(
        default_factory=lambda: [
            "fsc", "pefc", "rspo", "rainforest_alliance", "iscc",
        ]
    )

    # -----------------------------------------------------------------------
    # Auditor registry settings
    # -----------------------------------------------------------------------
    auditor_rotation_years: int = 3
    auditor_pool_match_timeout_ms: int = 500
    cpd_hours_minimum: int = 40

    # -----------------------------------------------------------------------
    # Competent authority settings
    # -----------------------------------------------------------------------
    authority_response_days: int = _DEFAULT_AUTHORITY_RESPONSE_DAYS
    authority_urgent_response_days: int = _DEFAULT_AUTHORITY_URGENT_RESPONSE_DAYS

    # -----------------------------------------------------------------------
    # Report generation settings
    # -----------------------------------------------------------------------
    report_generation_timeout_s: int = _DEFAULT_REPORT_GENERATION_TIMEOUT_S
    report_retention_years: int = _DEFAULT_REPORT_RETENTION_YEARS
    default_report_format: str = "pdf"
    default_report_language: str = "en"
    supported_report_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "xlsx", "xml"]
    )
    supported_report_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )

    # -----------------------------------------------------------------------
    # Analytics settings
    # -----------------------------------------------------------------------
    analytics_refresh_minutes: int = _DEFAULT_ANALYTICS_REFRESH_MINUTES
    dashboard_load_timeout_s: int = _DEFAULT_DASHBOARD_LOAD_TIMEOUT_S

    # -----------------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------------
    batch_max_size: int = 1000
    batch_concurrency: int = 8
    batch_timeout_s: int = 600

    # -----------------------------------------------------------------------
    # Data retention (EUDR Article 31)
    # -----------------------------------------------------------------------
    retention_years: int = 5

    # -----------------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-TAM-024-THIRD-PARTY-AUDIT-MANAGER-GENESIS"
    chain_algorithm: str = "sha256"

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_tam_"

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
        self._validate_risk_weights()
        self._validate_frequency_thresholds()
        self._validate_sla_deadlines()
        self._validate_car_sub_deadlines()
        self._validate_escalation()
        self._validate_evidence_limits()
        self._validate_schemes()
        self._validate_report_settings()
        self._validate_authority_settings()
        self._validate_positive_integers()

        logger.info(
            "ThirdPartyAuditManagerConfig initialized: "
            f"critical_sla={self.critical_sla_days}d, "
            f"major_sla={self.major_sla_days}d, "
            f"minor_sla={self.minor_sla_days}d, "
            f"schemes={len(self.enabled_schemes)}"
        )

    def _validate_log_level(self) -> None:
        """Validate log_level is a recognized Python logging level."""
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level: {self.log_level}. "
                f"Must be one of {sorted(_VALID_LOG_LEVELS)}"
            )

    def _validate_chain_algorithm(self) -> None:
        """Validate chain_algorithm is a supported hash algorithm."""
        if self.chain_algorithm not in _VALID_CHAIN_ALGORITHMS:
            raise ValueError(
                f"Invalid chain_algorithm: {self.chain_algorithm}. "
                f"Must be one of {sorted(_VALID_CHAIN_ALGORITHMS)}"
            )

    def _validate_risk_weights(self) -> None:
        """Validate risk weights are positive and sum to 1.0."""
        weights = [
            ("country_risk_weight", self.country_risk_weight),
            ("supplier_risk_weight", self.supplier_risk_weight),
            ("nc_history_weight", self.nc_history_weight),
            ("certification_gap_weight", self.certification_gap_weight),
            ("deforestation_alert_weight", self.deforestation_alert_weight),
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

    def _validate_frequency_thresholds(self) -> None:
        """Validate frequency tier thresholds are ordered."""
        if not (
            Decimal("0")
            < self.standard_risk_threshold
            < self.high_risk_threshold
            <= Decimal("100")
        ):
            raise ValueError(
                "Frequency thresholds must satisfy "
                "0 < standard < high <= 100. "
                f"Got standard={self.standard_risk_threshold}, "
                f"high={self.high_risk_threshold}"
            )
        if self.recency_multiplier_cap < Decimal("1"):
            raise ValueError(
                f"recency_multiplier_cap must be >= 1.0, "
                f"got {self.recency_multiplier_cap}"
            )

    def _validate_sla_deadlines(self) -> None:
        """Validate SLA deadlines are in ascending severity order."""
        if not (
            0 < self.critical_sla_days
            < self.major_sla_days
            < self.minor_sla_days
        ):
            raise ValueError(
                "SLA deadlines must satisfy 0 < critical < major < minor. "
                f"Got critical={self.critical_sla_days}, "
                f"major={self.major_sla_days}, "
                f"minor={self.minor_sla_days}"
            )

    def _validate_car_sub_deadlines(self) -> None:
        """Validate CAR sub-deadlines are within their SLA."""
        if self.critical_acknowledge_days >= self.critical_sla_days:
            raise ValueError(
                f"critical_acknowledge_days ({self.critical_acknowledge_days}) "
                f"must be < critical_sla_days ({self.critical_sla_days})"
            )
        if self.major_acknowledge_days >= self.major_sla_days:
            raise ValueError(
                f"major_acknowledge_days ({self.major_acknowledge_days}) "
                f"must be < major_sla_days ({self.major_sla_days})"
            )

    def _validate_escalation(self) -> None:
        """Validate escalation settings."""
        if not (
            Decimal("0")
            < self.escalation_stage1_pct
            < self.escalation_stage2_pct
            <= Decimal("1")
        ):
            raise ValueError(
                "Escalation percentages must satisfy "
                "0 < stage1 < stage2 <= 1.0. "
                f"Got stage1={self.escalation_stage1_pct}, "
                f"stage2={self.escalation_stage2_pct}"
            )
        if self.max_escalation_levels < 1:
            raise ValueError(
                f"max_escalation_levels must be >= 1, "
                f"got {self.max_escalation_levels}"
            )

    def _validate_evidence_limits(self) -> None:
        """Validate evidence size limits."""
        if self.max_evidence_size_mb < 1:
            raise ValueError(
                f"max_evidence_size_mb must be >= 1, "
                f"got {self.max_evidence_size_mb}"
            )
        if self.max_evidence_package_gb < 1:
            raise ValueError(
                f"max_evidence_package_gb must be >= 1, "
                f"got {self.max_evidence_package_gb}"
            )

    def _validate_schemes(self) -> None:
        """Validate certification scheme configuration."""
        for scheme in self.enabled_schemes:
            if scheme not in _VALID_CERTIFICATION_SCHEMES:
                raise ValueError(
                    f"Invalid certification scheme: {scheme}. "
                    f"Must be one of {sorted(_VALID_CERTIFICATION_SCHEMES)}"
                )

    def _validate_report_settings(self) -> None:
        """Validate report generation settings."""
        if self.default_report_format not in _VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid default_report_format: {self.default_report_format}. "
                f"Must be one of {sorted(_VALID_OUTPUT_FORMATS)}"
            )
        if self.default_report_language not in _VALID_REPORT_LANGUAGES:
            raise ValueError(
                f"Invalid default_report_language: {self.default_report_language}. "
                f"Must be one of {sorted(_VALID_REPORT_LANGUAGES)}"
            )
        for fmt in self.supported_report_formats:
            if fmt not in _VALID_OUTPUT_FORMATS:
                raise ValueError(
                    f"Invalid report format: {fmt}. "
                    f"Must be one of {sorted(_VALID_OUTPUT_FORMATS)}"
                )
        for lang in self.supported_report_languages:
            if lang not in _VALID_REPORT_LANGUAGES:
                raise ValueError(
                    f"Invalid report language: {lang}. "
                    f"Must be one of {sorted(_VALID_REPORT_LANGUAGES)}"
                )

    def _validate_authority_settings(self) -> None:
        """Validate competent authority settings."""
        if self.authority_response_days < 1:
            raise ValueError(
                f"authority_response_days must be >= 1, "
                f"got {self.authority_response_days}"
            )
        if self.authority_urgent_response_days < 1:
            raise ValueError(
                f"authority_urgent_response_days must be >= 1, "
                f"got {self.authority_urgent_response_days}"
            )
        if self.authority_urgent_response_days >= self.authority_response_days:
            raise ValueError(
                f"authority_urgent_response_days "
                f"({self.authority_urgent_response_days}) must be < "
                f"authority_response_days ({self.authority_response_days})"
            )

    def _validate_positive_integers(self) -> None:
        """Validate fields that must be positive integers."""
        checks = [
            ("pool_size", self.pool_size),
            ("batch_max_size", self.batch_max_size),
            ("batch_concurrency", self.batch_concurrency),
            ("retention_years", self.retention_years),
            ("report_generation_timeout_s", self.report_generation_timeout_s),
            ("report_retention_years", self.report_retention_years),
            ("analytics_refresh_minutes", self.analytics_refresh_minutes),
            ("dashboard_load_timeout_s", self.dashboard_load_timeout_s),
        ]
        for name, val in checks:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

    @classmethod
    def from_env(cls) -> "ThirdPartyAuditManagerConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_TAM_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            ThirdPartyAuditManagerConfig instance with env overrides applied.
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

        # Risk weights
        if val := os.getenv(f"{_ENV_PREFIX}COUNTRY_RISK_WEIGHT"):
            kwargs["country_risk_weight"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}SUPPLIER_RISK_WEIGHT"):
            kwargs["supplier_risk_weight"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}NC_HISTORY_WEIGHT"):
            kwargs["nc_history_weight"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}CERTIFICATION_GAP_WEIGHT"):
            kwargs["certification_gap_weight"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}DEFORESTATION_ALERT_WEIGHT"):
            kwargs["deforestation_alert_weight"] = Decimal(val)

        # Frequency thresholds
        if val := os.getenv(f"{_ENV_PREFIX}HIGH_RISK_THRESHOLD"):
            kwargs["high_risk_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}STANDARD_RISK_THRESHOLD"):
            kwargs["standard_risk_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}RECENCY_MULTIPLIER_CAP"):
            kwargs["recency_multiplier_cap"] = Decimal(val)

        # NC weights
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_NC_WEIGHT"):
            kwargs["critical_nc_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAJOR_NC_WEIGHT"):
            kwargs["major_nc_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MINOR_NC_WEIGHT"):
            kwargs["minor_nc_weight"] = int(val)

        # SLA deadlines
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_SLA_DAYS"):
            kwargs["critical_sla_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAJOR_SLA_DAYS"):
            kwargs["major_sla_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MINOR_SLA_DAYS"):
            kwargs["minor_sla_days"] = int(val)

        # CAR sub-deadlines
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_ACKNOWLEDGE_DAYS"):
            kwargs["critical_acknowledge_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_RCA_DAYS"):
            kwargs["critical_rca_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_CAP_DAYS"):
            kwargs["critical_cap_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAJOR_ACKNOWLEDGE_DAYS"):
            kwargs["major_acknowledge_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAJOR_RCA_DAYS"):
            kwargs["major_rca_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAJOR_CAP_DAYS"):
            kwargs["major_cap_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MINOR_ACKNOWLEDGE_DAYS"):
            kwargs["minor_acknowledge_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MINOR_RCA_DAYS"):
            kwargs["minor_rca_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MINOR_CAP_DAYS"):
            kwargs["minor_cap_days"] = int(val)

        # Escalation
        if val := os.getenv(f"{_ENV_PREFIX}ESCALATION_STAGE1_PCT"):
            kwargs["escalation_stage1_pct"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}ESCALATION_STAGE2_PCT"):
            kwargs["escalation_stage2_pct"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_ESCALATION_LEVELS"):
            kwargs["max_escalation_levels"] = int(val)

        # Evidence limits
        if val := os.getenv(f"{_ENV_PREFIX}MAX_EVIDENCE_SIZE_MB"):
            kwargs["max_evidence_size_mb"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_EVIDENCE_PACKAGE_GB"):
            kwargs["max_evidence_package_gb"] = int(val)

        # Certification schemes
        if val := os.getenv(f"{_ENV_PREFIX}CERT_SYNC_INTERVAL_HOURS"):
            kwargs["cert_sync_interval_hours"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ACCREDITATION_EXPIRY_WARNING_DAYS"):
            kwargs["accreditation_expiry_warning_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ENABLED_SCHEMES"):
            kwargs["enabled_schemes"] = [x.strip() for x in val.split(",")]

        # Auditor registry
        if val := os.getenv(f"{_ENV_PREFIX}AUDITOR_ROTATION_YEARS"):
            kwargs["auditor_rotation_years"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}AUDITOR_POOL_MATCH_TIMEOUT_MS"):
            kwargs["auditor_pool_match_timeout_ms"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CPD_HOURS_MINIMUM"):
            kwargs["cpd_hours_minimum"] = int(val)

        # Competent authority
        if val := os.getenv(f"{_ENV_PREFIX}AUTHORITY_RESPONSE_DAYS"):
            kwargs["authority_response_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}AUTHORITY_URGENT_RESPONSE_DAYS"):
            kwargs["authority_urgent_response_days"] = int(val)

        # Report generation
        if val := os.getenv(f"{_ENV_PREFIX}REPORT_GENERATION_TIMEOUT_S"):
            kwargs["report_generation_timeout_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}REPORT_RETENTION_YEARS"):
            kwargs["report_retention_years"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_REPORT_FORMAT"):
            kwargs["default_report_format"] = val.lower()
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_REPORT_LANGUAGE"):
            kwargs["default_report_language"] = val.lower()
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_REPORT_FORMATS"):
            kwargs["supported_report_formats"] = [
                x.strip() for x in val.split(",")
            ]
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_REPORT_LANGUAGES"):
            kwargs["supported_report_languages"] = [
                x.strip() for x in val.split(",")
            ]

        # Analytics
        if val := os.getenv(f"{_ENV_PREFIX}ANALYTICS_REFRESH_MINUTES"):
            kwargs["analytics_refresh_minutes"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DASHBOARD_LOAD_TIMEOUT_S"):
            kwargs["dashboard_load_timeout_s"] = int(val)

        # Batch processing
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
            "pool_timeout_s": self.pool_timeout_s,
            "pool_recycle_s": self.pool_recycle_s,
            "redis_url": self.redis_url,
            "redis_ttl_s": self.redis_ttl_s,
            "redis_key_prefix": self.redis_key_prefix,
            "log_level": self.log_level,
            "country_risk_weight": str(self.country_risk_weight),
            "supplier_risk_weight": str(self.supplier_risk_weight),
            "nc_history_weight": str(self.nc_history_weight),
            "certification_gap_weight": str(self.certification_gap_weight),
            "deforestation_alert_weight": str(self.deforestation_alert_weight),
            "high_risk_threshold": str(self.high_risk_threshold),
            "standard_risk_threshold": str(self.standard_risk_threshold),
            "recency_multiplier_cap": str(self.recency_multiplier_cap),
            "critical_sla_days": self.critical_sla_days,
            "major_sla_days": self.major_sla_days,
            "minor_sla_days": self.minor_sla_days,
            "escalation_stage1_pct": str(self.escalation_stage1_pct),
            "escalation_stage2_pct": str(self.escalation_stage2_pct),
            "max_escalation_levels": self.max_escalation_levels,
            "enabled_schemes": self.enabled_schemes,
            "retention_years": self.retention_years,
            "enable_provenance": self.enable_provenance,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
        }

        if redact_secrets:
            if "://" in data.get("database_url", ""):
                data["database_url"] = "REDACTED"
            if "://" in data.get("redis_url", ""):
                data["redis_url"] = "REDACTED"

        return data


# ---------------------------------------------------------------------------
# Thread-safe singleton pattern (double-checked locking)
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[ThirdPartyAuditManagerConfig] = None


def get_config() -> ThirdPartyAuditManagerConfig:
    """Get the global ThirdPartyAuditManagerConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.

    Returns:
        ThirdPartyAuditManagerConfig singleton instance.
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = ThirdPartyAuditManagerConfig.from_env()
    return _global_config


def set_config(config: ThirdPartyAuditManagerConfig) -> None:
    """Set the global ThirdPartyAuditManagerConfig singleton instance.

    Args:
        config: ThirdPartyAuditManagerConfig instance to set as global.
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global ThirdPartyAuditManagerConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.
    """
    global _global_config
    with _config_lock:
        _global_config = None
