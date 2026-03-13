# -*- coding: utf-8 -*-
"""
Multi-Tier Supplier Tracker Configuration - AGENT-EUDR-008

Centralized configuration for the Multi-Tier Supplier Tracker Agent covering:
- Supplier discovery: max tier depth, confidence thresholds, deduplication
- Supplier profiles: completeness weights per field category
- Risk assessment: category weights, propagation methods, thresholds
- Compliance monitoring: status thresholds, alert timing
- Tier visibility: minimum visibility scores, benchmark defaults
- Batch processing: size limits, concurrency
- Data retention: EUDR Article 31 five-year retention
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_MST_`` prefix (e.g. ``GL_EUDR_MST_DATABASE_URL``,
``GL_EUDR_MST_MAX_TIER_DEPTH``).

Environment Variable Reference (GL_EUDR_MST_ prefix):
    GL_EUDR_MST_DATABASE_URL                   - PostgreSQL connection URL
    GL_EUDR_MST_REDIS_URL                      - Redis connection URL
    GL_EUDR_MST_LOG_LEVEL                      - Logging level
    GL_EUDR_MST_MAX_TIER_DEPTH                 - Maximum supplier tier depth
    GL_EUDR_MST_DISCOVERY_CONFIDENCE_THRESHOLD - Min confidence for discovery
    GL_EUDR_MST_DEDUPLICATION_THRESHOLD        - Dedup similarity threshold
    GL_EUDR_MST_PROFILE_WEIGHT_LEGAL_IDENTITY  - Profile weight: legal identity
    GL_EUDR_MST_PROFILE_WEIGHT_LOCATION        - Profile weight: location
    GL_EUDR_MST_PROFILE_WEIGHT_COMMODITY       - Profile weight: commodity
    GL_EUDR_MST_PROFILE_WEIGHT_CERTIFICATION   - Profile weight: certification
    GL_EUDR_MST_PROFILE_WEIGHT_COMPLIANCE      - Profile weight: compliance
    GL_EUDR_MST_PROFILE_WEIGHT_CONTACT         - Profile weight: contact
    GL_EUDR_MST_RISK_WEIGHT_DEFORESTATION_PROXIMITY - Risk weight: deforestation
    GL_EUDR_MST_RISK_WEIGHT_COUNTRY_RISK       - Risk weight: country risk
    GL_EUDR_MST_RISK_WEIGHT_CERTIFICATION_GAP  - Risk weight: cert gap
    GL_EUDR_MST_RISK_WEIGHT_COMPLIANCE_HISTORY - Risk weight: compliance hist
    GL_EUDR_MST_RISK_WEIGHT_DATA_QUALITY       - Risk weight: data quality
    GL_EUDR_MST_RISK_WEIGHT_CONCENTRATION_RISK - Risk weight: concentration
    GL_EUDR_MST_DEFAULT_RISK_PROPAGATION_METHOD - Default propagation method
    GL_EUDR_MST_COMPLIANT_THRESHOLD            - Compliant score threshold
    GL_EUDR_MST_CONDITIONALLY_COMPLIANT_THRESHOLD - Conditional threshold
    GL_EUDR_MST_DDS_EXPIRY_WARNING_DAYS        - DDS expiry warn days (CSV)
    GL_EUDR_MST_CERT_EXPIRY_WARNING_DAYS       - Cert expiry warn days (CSV)
    GL_EUDR_MST_MIN_VISIBILITY_SCORE           - Min tier visibility score
    GL_EUDR_MST_INDUSTRY_AVG_TIER_DEPTH        - Industry avg tier depth
    GL_EUDR_MST_BATCH_MAX_SIZE                 - Maximum batch size
    GL_EUDR_MST_BATCH_CONCURRENCY              - Batch concurrency
    GL_EUDR_MST_RETENTION_YEARS                - Data retention (years)
    GL_EUDR_MST_ENABLE_PROVENANCE              - Enable provenance tracking
    GL_EUDR_MST_GENESIS_HASH                   - Genesis hash anchor
    GL_EUDR_MST_ENABLE_METRICS                 - Enable Prometheus metrics
    GL_EUDR_MST_POOL_SIZE                      - Database pool size
    GL_EUDR_MST_RATE_LIMIT                     - Max requests per minute
    GL_EUDR_MST_RELATIONSHIP_STRENGTH_VOLUME_WEIGHT   - Rel strength: volume
    GL_EUDR_MST_RELATIONSHIP_STRENGTH_FREQUENCY_WEIGHT - Rel strength: freq
    GL_EUDR_MST_RELATIONSHIP_STRENGTH_DURATION_WEIGHT  - Rel strength: duration
    GL_EUDR_MST_RISK_ALERT_THRESHOLD           - Risk alert threshold
    GL_EUDR_MST_RISK_CRITICAL_THRESHOLD        - Risk critical threshold
    GL_EUDR_MST_GAP_AUTO_QUESTIONNAIRE_ENABLED - Enable auto questionnaire
    GL_EUDR_MST_REPORT_GENERATION_TIMEOUT_S    - Report generation timeout (s)

Example:
    >>> from greenlang.agents.eudr.multi_tier_supplier.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_tier_depth)
    15

    >>> # Override for testing
    >>> from greenlang.agents.eudr.multi_tier_supplier.config import (
    ...     set_config, reset_config, MultiTierSupplierConfig,
    ... )
    >>> set_config(MultiTierSupplierConfig(max_tier_depth=10))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
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

_ENV_PREFIX = "GL_EUDR_MST_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid risk propagation methods
# ---------------------------------------------------------------------------

_VALID_RISK_PROPAGATION_METHODS = frozenset(
    {"max", "weighted_average", "volume_weighted"}
)

# ---------------------------------------------------------------------------
# Default profile completeness weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE_COMPLETENESS_WEIGHTS: Dict[str, float] = {
    "legal_identity": 0.25,
    "location": 0.20,
    "commodity": 0.15,
    "certification": 0.15,
    "compliance": 0.15,
    "contact": 0.10,
}

# ---------------------------------------------------------------------------
# Default risk category weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_RISK_CATEGORY_WEIGHTS: Dict[str, float] = {
    "deforestation_proximity": 0.30,
    "country_risk": 0.20,
    "certification_gap": 0.15,
    "compliance_history": 0.15,
    "data_quality": 0.10,
    "concentration_risk": 0.10,
}

# ---------------------------------------------------------------------------
# Default relationship strength weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_RELATIONSHIP_STRENGTH_WEIGHTS: Dict[str, float] = {
    "volume": 0.40,
    "frequency": 0.35,
    "duration": 0.25,
}

# ---------------------------------------------------------------------------
# Default DDS and certification expiry warning days
# ---------------------------------------------------------------------------

_DEFAULT_DDS_EXPIRY_WARNING_DAYS: List[int] = [30, 14, 7]
_DEFAULT_CERT_EXPIRY_WARNING_DAYS: List[int] = [90, 30, 14]

# ---------------------------------------------------------------------------
# Default report formats
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_FORMATS: List[str] = ["json", "pdf", "csv", "eudr_xml"]

# ---------------------------------------------------------------------------
# Default commodity types (7 EUDR commodities)
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]


# ---------------------------------------------------------------------------
# MultiTierSupplierConfig
# ---------------------------------------------------------------------------


@dataclass
class MultiTierSupplierConfig:
    """Complete configuration for the EUDR Multi-Tier Supplier Tracker Agent.

    Attributes are grouped by concern: connections, logging, discovery,
    profile completeness, risk assessment, risk propagation, compliance
    monitoring, alerting, tier visibility, relationship strength, batch
    processing, report settings, data retention, provenance tracking,
    metrics, and performance tuning.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_MST_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching.
        log_level: Logging verbosity level.
        max_tier_depth: Maximum supplier tier depth for recursive discovery.
        discovery_confidence_threshold: Minimum confidence for accepting
            a discovered supplier relationship.
        deduplication_threshold: Similarity threshold for supplier dedup.
        profile_weight_legal_identity: Weight for legal identity fields.
        profile_weight_location: Weight for location fields.
        profile_weight_commodity: Weight for commodity fields.
        profile_weight_certification: Weight for certification fields.
        profile_weight_compliance: Weight for compliance fields.
        profile_weight_contact: Weight for contact fields.
        risk_weight_deforestation_proximity: Weight for deforestation risk.
        risk_weight_country_risk: Weight for country-level risk.
        risk_weight_certification_gap: Weight for certification gap risk.
        risk_weight_compliance_history: Weight for compliance history risk.
        risk_weight_data_quality: Weight for data quality risk.
        risk_weight_concentration_risk: Weight for concentration risk.
        default_risk_propagation_method: Default risk propagation method.
        compliant_threshold: Min score for compliant status (0-100).
        conditionally_compliant_threshold: Min score for conditional status.
        dds_expiry_warning_days: DDS expiry warning thresholds in days.
        cert_expiry_warning_days: Certification expiry warning thresholds.
        min_visibility_score: Min tier visibility score (0.0-1.0).
        industry_avg_tier_depth: Industry average tier depth benchmark.
        relationship_strength_volume_weight: Rel strength volume weight.
        relationship_strength_frequency_weight: Rel strength freq weight.
        relationship_strength_duration_weight: Rel strength duration weight.
        risk_alert_threshold: Risk score triggering alerts (0-100).
        risk_critical_threshold: Risk score triggering critical alerts.
        gap_auto_questionnaire_enabled: Auto-generate gap questionnaires.
        batch_max_size: Maximum records in a single batch.
        batch_concurrency: Maximum concurrent batch workers.
        report_formats: Supported report output formats.
        report_generation_timeout_s: Report generation timeout in seconds.
        retention_years: Data retention in years per EUDR Article 31.
        eudr_commodities: List of EUDR-regulated commodity types.
        enable_provenance: Enable SHA-256 provenance chain tracking.
        genesis_hash: Genesis anchor string for provenance chain.
        enable_metrics: Enable Prometheus metrics export.
        pool_size: Database connection pool size.
        rate_limit: Maximum API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Discovery settings --------------------------------------------------
    max_tier_depth: int = 15
    discovery_confidence_threshold: float = 0.70
    deduplication_threshold: float = 0.85

    # -- Profile completeness weights ----------------------------------------
    profile_weight_legal_identity: float = 0.25
    profile_weight_location: float = 0.20
    profile_weight_commodity: float = 0.15
    profile_weight_certification: float = 0.15
    profile_weight_compliance: float = 0.15
    profile_weight_contact: float = 0.10

    # -- Risk category weights -----------------------------------------------
    risk_weight_deforestation_proximity: float = 0.30
    risk_weight_country_risk: float = 0.20
    risk_weight_certification_gap: float = 0.15
    risk_weight_compliance_history: float = 0.15
    risk_weight_data_quality: float = 0.10
    risk_weight_concentration_risk: float = 0.10

    # -- Risk propagation ----------------------------------------------------
    default_risk_propagation_method: str = "max"

    # -- Compliance thresholds -----------------------------------------------
    compliant_threshold: float = 90.0
    conditionally_compliant_threshold: float = 70.0

    # -- Alert thresholds ----------------------------------------------------
    dds_expiry_warning_days: List[int] = field(
        default_factory=lambda: list(_DEFAULT_DDS_EXPIRY_WARNING_DAYS)
    )
    cert_expiry_warning_days: List[int] = field(
        default_factory=lambda: list(_DEFAULT_CERT_EXPIRY_WARNING_DAYS)
    )
    risk_alert_threshold: float = 70.0
    risk_critical_threshold: float = 90.0

    # -- Tier visibility -----------------------------------------------------
    min_visibility_score: float = 0.60
    industry_avg_tier_depth: float = 3.5

    # -- Relationship strength weights ---------------------------------------
    relationship_strength_volume_weight: float = 0.40
    relationship_strength_frequency_weight: float = 0.35
    relationship_strength_duration_weight: float = 0.25

    # -- Gap analysis --------------------------------------------------------
    gap_auto_questionnaire_enabled: bool = True

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 100_000
    batch_concurrency: int = 8

    # -- Report settings -----------------------------------------------------
    report_formats: List[str] = field(
        default_factory=lambda: list(_DEFAULT_REPORT_FORMATS)
    )
    report_generation_timeout_s: int = 30

    # -- Data retention (EUDR Article 31) ------------------------------------
    retention_years: int = 5

    # -- EUDR commodities ----------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-MST-008-MULTI-TIER-SUPPLIER-TRACKER-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10
    rate_limit: int = 1000

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, weight sum validation, threshold ordering,
        and normalization. Collects all errors before raising a single
        ValueError with all violations listed.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint.
        """
        errors: list[str] = []

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- Discovery -------------------------------------------------------
        if not (1 <= self.max_tier_depth <= 50):
            errors.append(
                f"max_tier_depth must be in [1, 50], "
                f"got {self.max_tier_depth}"
            )

        if not (0.0 < self.discovery_confidence_threshold <= 1.0):
            errors.append(
                f"discovery_confidence_threshold must be in (0, 1], "
                f"got {self.discovery_confidence_threshold}"
            )

        if not (0.0 < self.deduplication_threshold <= 1.0):
            errors.append(
                f"deduplication_threshold must be in (0, 1], "
                f"got {self.deduplication_threshold}"
            )

        # -- Profile completeness weights ------------------------------------
        profile_weights = [
            ("profile_weight_legal_identity",
             self.profile_weight_legal_identity),
            ("profile_weight_location",
             self.profile_weight_location),
            ("profile_weight_commodity",
             self.profile_weight_commodity),
            ("profile_weight_certification",
             self.profile_weight_certification),
            ("profile_weight_compliance",
             self.profile_weight_compliance),
            ("profile_weight_contact",
             self.profile_weight_contact),
        ]
        for name, value in profile_weights:
            if not (0.0 <= value <= 1.0):
                errors.append(
                    f"{name} must be in [0.0, 1.0], got {value}"
                )
        profile_weight_sum = sum(v for _, v in profile_weights)
        if abs(profile_weight_sum - 1.0) > 0.001:
            errors.append(
                f"profile completeness weights must sum to 1.0, "
                f"got {profile_weight_sum:.4f}"
            )

        # -- Risk category weights -------------------------------------------
        risk_weights = [
            ("risk_weight_deforestation_proximity",
             self.risk_weight_deforestation_proximity),
            ("risk_weight_country_risk",
             self.risk_weight_country_risk),
            ("risk_weight_certification_gap",
             self.risk_weight_certification_gap),
            ("risk_weight_compliance_history",
             self.risk_weight_compliance_history),
            ("risk_weight_data_quality",
             self.risk_weight_data_quality),
            ("risk_weight_concentration_risk",
             self.risk_weight_concentration_risk),
        ]
        for name, value in risk_weights:
            if not (0.0 <= value <= 1.0):
                errors.append(
                    f"{name} must be in [0.0, 1.0], got {value}"
                )
        risk_weight_sum = sum(v for _, v in risk_weights)
        if abs(risk_weight_sum - 1.0) > 0.001:
            errors.append(
                f"risk category weights must sum to 1.0, "
                f"got {risk_weight_sum:.4f}"
            )

        # -- Relationship strength weights -----------------------------------
        rel_weights = [
            ("relationship_strength_volume_weight",
             self.relationship_strength_volume_weight),
            ("relationship_strength_frequency_weight",
             self.relationship_strength_frequency_weight),
            ("relationship_strength_duration_weight",
             self.relationship_strength_duration_weight),
        ]
        for name, value in rel_weights:
            if not (0.0 <= value <= 1.0):
                errors.append(
                    f"{name} must be in [0.0, 1.0], got {value}"
                )
        rel_weight_sum = sum(v for _, v in rel_weights)
        if abs(rel_weight_sum - 1.0) > 0.001:
            errors.append(
                f"relationship strength weights must sum to 1.0, "
                f"got {rel_weight_sum:.4f}"
            )

        # -- Risk propagation method -----------------------------------------
        if self.default_risk_propagation_method not in (
            _VALID_RISK_PROPAGATION_METHODS
        ):
            errors.append(
                f"default_risk_propagation_method must be one of "
                f"{sorted(_VALID_RISK_PROPAGATION_METHODS)}, "
                f"got '{self.default_risk_propagation_method}'"
            )

        # -- Compliance thresholds -------------------------------------------
        if not (0.0 <= self.compliant_threshold <= 100.0):
            errors.append(
                f"compliant_threshold must be in [0, 100], "
                f"got {self.compliant_threshold}"
            )
        if not (0.0 <= self.conditionally_compliant_threshold <= 100.0):
            errors.append(
                f"conditionally_compliant_threshold must be in [0, 100], "
                f"got {self.conditionally_compliant_threshold}"
            )
        if self.conditionally_compliant_threshold >= self.compliant_threshold:
            errors.append(
                f"conditionally_compliant_threshold "
                f"({self.conditionally_compliant_threshold}) "
                f"must be < compliant_threshold "
                f"({self.compliant_threshold})"
            )

        # -- Risk alert thresholds -------------------------------------------
        if not (0.0 <= self.risk_alert_threshold <= 100.0):
            errors.append(
                f"risk_alert_threshold must be in [0, 100], "
                f"got {self.risk_alert_threshold}"
            )
        if not (0.0 <= self.risk_critical_threshold <= 100.0):
            errors.append(
                f"risk_critical_threshold must be in [0, 100], "
                f"got {self.risk_critical_threshold}"
            )
        if self.risk_alert_threshold >= self.risk_critical_threshold:
            errors.append(
                f"risk_alert_threshold ({self.risk_alert_threshold}) "
                f"must be < risk_critical_threshold "
                f"({self.risk_critical_threshold})"
            )

        # -- DDS expiry warning days -----------------------------------------
        if not self.dds_expiry_warning_days:
            errors.append("dds_expiry_warning_days must not be empty")
        else:
            for d in self.dds_expiry_warning_days:
                if d < 1:
                    errors.append(
                        f"dds_expiry_warning_days values must be >= 1, "
                        f"got {d}"
                    )

        # -- Cert expiry warning days ----------------------------------------
        if not self.cert_expiry_warning_days:
            errors.append("cert_expiry_warning_days must not be empty")
        else:
            for d in self.cert_expiry_warning_days:
                if d < 1:
                    errors.append(
                        f"cert_expiry_warning_days values must be >= 1, "
                        f"got {d}"
                    )

        # -- Tier visibility -------------------------------------------------
        if not (0.0 <= self.min_visibility_score <= 1.0):
            errors.append(
                f"min_visibility_score must be in [0.0, 1.0], "
                f"got {self.min_visibility_score}"
            )

        if self.industry_avg_tier_depth <= 0:
            errors.append(
                f"industry_avg_tier_depth must be > 0, "
                f"got {self.industry_avg_tier_depth}"
            )

        # -- Batch processing ------------------------------------------------
        if self.batch_max_size < 1:
            errors.append(
                f"batch_max_size must be >= 1, got {self.batch_max_size}"
            )
        if not (1 <= self.batch_concurrency <= 256):
            errors.append(
                f"batch_concurrency must be in [1, 256], "
                f"got {self.batch_concurrency}"
            )

        # -- Report settings -------------------------------------------------
        if not self.report_formats:
            errors.append("report_formats must not be empty")

        if self.report_generation_timeout_s < 1:
            errors.append(
                f"report_generation_timeout_s must be >= 1, "
                f"got {self.report_generation_timeout_s}"
            )

        # -- Data retention --------------------------------------------------
        if self.retention_years < 1:
            errors.append(
                f"retention_years must be >= 1, "
                f"got {self.retention_years}"
            )

        # -- EUDR commodities ------------------------------------------------
        if not self.eudr_commodities:
            errors.append("eudr_commodities must not be empty")

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance tuning ----------------------------------------------
        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")

        if errors:
            raise ValueError(
                "MultiTierSupplierConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "MultiTierSupplierConfig validated successfully: "
            "max_tier_depth=%d, discovery_conf=%.2f, dedup=%.2f, "
            "propagation=%s, compliant>=%.0f, conditional>=%.0f, "
            "batch_max=%d, concurrency=%d, retention=%dy, "
            "provenance=%s, metrics=%s",
            self.max_tier_depth,
            self.discovery_confidence_threshold,
            self.deduplication_threshold,
            self.default_risk_propagation_method,
            self.compliant_threshold,
            self.conditionally_compliant_threshold,
            self.batch_max_size,
            self.batch_concurrency,
            self.retention_years,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> MultiTierSupplierConfig:
        """Build a MultiTierSupplierConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_MST_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        List[int] values accept comma-separated integers.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated MultiTierSupplierConfig instance, validated
            via ``__post_init__``.
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

        def _int_list(
            name: str, default: List[int],
        ) -> List[int]:
            val = _env(name)
            if val is None:
                return list(default)
            try:
                return [int(x.strip()) for x in val.split(",") if x.strip()]
            except ValueError:
                logger.warning(
                    "Invalid int list for %s%s=%r, using default %s",
                    prefix, name, val, default,
                )
                return list(default)

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Discovery
            max_tier_depth=_int(
                "MAX_TIER_DEPTH", cls.max_tier_depth,
            ),
            discovery_confidence_threshold=_float(
                "DISCOVERY_CONFIDENCE_THRESHOLD",
                cls.discovery_confidence_threshold,
            ),
            deduplication_threshold=_float(
                "DEDUPLICATION_THRESHOLD",
                cls.deduplication_threshold,
            ),
            # Profile completeness weights
            profile_weight_legal_identity=_float(
                "PROFILE_WEIGHT_LEGAL_IDENTITY",
                cls.profile_weight_legal_identity,
            ),
            profile_weight_location=_float(
                "PROFILE_WEIGHT_LOCATION",
                cls.profile_weight_location,
            ),
            profile_weight_commodity=_float(
                "PROFILE_WEIGHT_COMMODITY",
                cls.profile_weight_commodity,
            ),
            profile_weight_certification=_float(
                "PROFILE_WEIGHT_CERTIFICATION",
                cls.profile_weight_certification,
            ),
            profile_weight_compliance=_float(
                "PROFILE_WEIGHT_COMPLIANCE",
                cls.profile_weight_compliance,
            ),
            profile_weight_contact=_float(
                "PROFILE_WEIGHT_CONTACT",
                cls.profile_weight_contact,
            ),
            # Risk category weights
            risk_weight_deforestation_proximity=_float(
                "RISK_WEIGHT_DEFORESTATION_PROXIMITY",
                cls.risk_weight_deforestation_proximity,
            ),
            risk_weight_country_risk=_float(
                "RISK_WEIGHT_COUNTRY_RISK",
                cls.risk_weight_country_risk,
            ),
            risk_weight_certification_gap=_float(
                "RISK_WEIGHT_CERTIFICATION_GAP",
                cls.risk_weight_certification_gap,
            ),
            risk_weight_compliance_history=_float(
                "RISK_WEIGHT_COMPLIANCE_HISTORY",
                cls.risk_weight_compliance_history,
            ),
            risk_weight_data_quality=_float(
                "RISK_WEIGHT_DATA_QUALITY",
                cls.risk_weight_data_quality,
            ),
            risk_weight_concentration_risk=_float(
                "RISK_WEIGHT_CONCENTRATION_RISK",
                cls.risk_weight_concentration_risk,
            ),
            # Risk propagation
            default_risk_propagation_method=_str(
                "DEFAULT_RISK_PROPAGATION_METHOD",
                cls.default_risk_propagation_method,
            ),
            # Compliance thresholds
            compliant_threshold=_float(
                "COMPLIANT_THRESHOLD",
                cls.compliant_threshold,
            ),
            conditionally_compliant_threshold=_float(
                "CONDITIONALLY_COMPLIANT_THRESHOLD",
                cls.conditionally_compliant_threshold,
            ),
            # Alert thresholds
            dds_expiry_warning_days=_int_list(
                "DDS_EXPIRY_WARNING_DAYS",
                _DEFAULT_DDS_EXPIRY_WARNING_DAYS,
            ),
            cert_expiry_warning_days=_int_list(
                "CERT_EXPIRY_WARNING_DAYS",
                _DEFAULT_CERT_EXPIRY_WARNING_DAYS,
            ),
            risk_alert_threshold=_float(
                "RISK_ALERT_THRESHOLD",
                cls.risk_alert_threshold,
            ),
            risk_critical_threshold=_float(
                "RISK_CRITICAL_THRESHOLD",
                cls.risk_critical_threshold,
            ),
            # Tier visibility
            min_visibility_score=_float(
                "MIN_VISIBILITY_SCORE",
                cls.min_visibility_score,
            ),
            industry_avg_tier_depth=_float(
                "INDUSTRY_AVG_TIER_DEPTH",
                cls.industry_avg_tier_depth,
            ),
            # Relationship strength weights
            relationship_strength_volume_weight=_float(
                "RELATIONSHIP_STRENGTH_VOLUME_WEIGHT",
                cls.relationship_strength_volume_weight,
            ),
            relationship_strength_frequency_weight=_float(
                "RELATIONSHIP_STRENGTH_FREQUENCY_WEIGHT",
                cls.relationship_strength_frequency_weight,
            ),
            relationship_strength_duration_weight=_float(
                "RELATIONSHIP_STRENGTH_DURATION_WEIGHT",
                cls.relationship_strength_duration_weight,
            ),
            # Gap analysis
            gap_auto_questionnaire_enabled=_bool(
                "GAP_AUTO_QUESTIONNAIRE_ENABLED",
                cls.gap_auto_questionnaire_enabled,
            ),
            # Batch processing
            batch_max_size=_int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            batch_concurrency=_int(
                "BATCH_CONCURRENCY", cls.batch_concurrency,
            ),
            # Report settings
            report_generation_timeout_s=_int(
                "REPORT_GENERATION_TIMEOUT_S",
                cls.report_generation_timeout_s,
            ),
            # Data retention
            retention_years=_int(
                "RETENTION_YEARS", cls.retention_years,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "MultiTierSupplierConfig loaded: "
            "max_tier_depth=%d, discovery_conf=%.2f, dedup=%.2f, "
            "profile_weights=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f], "
            "risk_weights=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f], "
            "propagation=%s, "
            "compliant>=%.0f, conditional>=%.0f, "
            "risk_alert=%.0f, risk_critical=%.0f, "
            "dds_expiry=%s, cert_expiry=%s, "
            "min_visibility=%.2f, industry_avg=%.1f, "
            "batch_max=%d, concurrency=%d, "
            "retention=%dy, provenance=%s, "
            "pool=%d, rate_limit=%d/min, metrics=%s",
            config.max_tier_depth,
            config.discovery_confidence_threshold,
            config.deduplication_threshold,
            config.profile_weight_legal_identity,
            config.profile_weight_location,
            config.profile_weight_commodity,
            config.profile_weight_certification,
            config.profile_weight_compliance,
            config.profile_weight_contact,
            config.risk_weight_deforestation_proximity,
            config.risk_weight_country_risk,
            config.risk_weight_certification_gap,
            config.risk_weight_compliance_history,
            config.risk_weight_data_quality,
            config.risk_weight_concentration_risk,
            config.default_risk_propagation_method,
            config.compliant_threshold,
            config.conditionally_compliant_threshold,
            config.risk_alert_threshold,
            config.risk_critical_threshold,
            config.dds_expiry_warning_days,
            config.cert_expiry_warning_days,
            config.min_visibility_score,
            config.industry_avg_tier_depth,
            config.batch_max_size,
            config.batch_concurrency,
            config.retention_years,
            config.enable_provenance,
            config.pool_size,
            config.rate_limit,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def profile_completeness_weights(self) -> Dict[str, float]:
        """Return profile completeness weights as a dictionary.

        Returns:
            Dictionary with keys: legal_identity, location, commodity,
            certification, compliance, contact.
        """
        return {
            "legal_identity": self.profile_weight_legal_identity,
            "location": self.profile_weight_location,
            "commodity": self.profile_weight_commodity,
            "certification": self.profile_weight_certification,
            "compliance": self.profile_weight_compliance,
            "contact": self.profile_weight_contact,
        }

    @property
    def risk_category_weights(self) -> Dict[str, float]:
        """Return risk category weights as a dictionary.

        Returns:
            Dictionary with keys: deforestation_proximity, country_risk,
            certification_gap, compliance_history, data_quality,
            concentration_risk.
        """
        return {
            "deforestation_proximity": (
                self.risk_weight_deforestation_proximity
            ),
            "country_risk": self.risk_weight_country_risk,
            "certification_gap": self.risk_weight_certification_gap,
            "compliance_history": self.risk_weight_compliance_history,
            "data_quality": self.risk_weight_data_quality,
            "concentration_risk": self.risk_weight_concentration_risk,
        }

    @property
    def relationship_strength_weights(self) -> Dict[str, float]:
        """Return relationship strength weights as a dictionary.

        Returns:
            Dictionary with keys: volume, frequency, duration.
        """
        return {
            "volume": self.relationship_strength_volume_weight,
            "frequency": self.relationship_strength_frequency_weight,
            "duration": self.relationship_strength_duration_weight,
        }

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Re-run post-init validation and return True if valid.

        Returns:
            True if configuration passes all validation checks.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        self.__post_init__()
        return True

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain Python dictionary.

        Sensitive connection strings (database_url, redis_url) are
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # Connections (redacted)
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # Logging
            "log_level": self.log_level,
            # Discovery
            "max_tier_depth": self.max_tier_depth,
            "discovery_confidence_threshold": (
                self.discovery_confidence_threshold
            ),
            "deduplication_threshold": self.deduplication_threshold,
            # Profile completeness weights
            "profile_weight_legal_identity": (
                self.profile_weight_legal_identity
            ),
            "profile_weight_location": self.profile_weight_location,
            "profile_weight_commodity": self.profile_weight_commodity,
            "profile_weight_certification": (
                self.profile_weight_certification
            ),
            "profile_weight_compliance": self.profile_weight_compliance,
            "profile_weight_contact": self.profile_weight_contact,
            # Risk category weights
            "risk_weight_deforestation_proximity": (
                self.risk_weight_deforestation_proximity
            ),
            "risk_weight_country_risk": self.risk_weight_country_risk,
            "risk_weight_certification_gap": (
                self.risk_weight_certification_gap
            ),
            "risk_weight_compliance_history": (
                self.risk_weight_compliance_history
            ),
            "risk_weight_data_quality": self.risk_weight_data_quality,
            "risk_weight_concentration_risk": (
                self.risk_weight_concentration_risk
            ),
            # Risk propagation
            "default_risk_propagation_method": (
                self.default_risk_propagation_method
            ),
            # Compliance thresholds
            "compliant_threshold": self.compliant_threshold,
            "conditionally_compliant_threshold": (
                self.conditionally_compliant_threshold
            ),
            # Alert thresholds
            "dds_expiry_warning_days": list(self.dds_expiry_warning_days),
            "cert_expiry_warning_days": list(self.cert_expiry_warning_days),
            "risk_alert_threshold": self.risk_alert_threshold,
            "risk_critical_threshold": self.risk_critical_threshold,
            # Tier visibility
            "min_visibility_score": self.min_visibility_score,
            "industry_avg_tier_depth": self.industry_avg_tier_depth,
            # Relationship strength weights
            "relationship_strength_volume_weight": (
                self.relationship_strength_volume_weight
            ),
            "relationship_strength_frequency_weight": (
                self.relationship_strength_frequency_weight
            ),
            "relationship_strength_duration_weight": (
                self.relationship_strength_duration_weight
            ),
            # Gap analysis
            "gap_auto_questionnaire_enabled": (
                self.gap_auto_questionnaire_enabled
            ),
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            # Report settings
            "report_formats": list(self.report_formats),
            "report_generation_timeout_s": self.report_generation_timeout_s,
            # Data retention
            "retention_years": self.retention_years,
            # EUDR commodities
            "eudr_commodities": list(self.eudr_commodities),
            # Provenance
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # Metrics
            "enable_metrics": self.enable_metrics,
            # Performance tuning
            "pool_size": self.pool_size,
            "rate_limit": self.rate_limit,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"MultiTierSupplierConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[MultiTierSupplierConfig] = None
_config_lock = threading.Lock()


def get_config() -> MultiTierSupplierConfig:
    """Return the singleton MultiTierSupplierConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_MST_*`` environment variables.

    Returns:
        MultiTierSupplierConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.max_tier_depth
        15
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = MultiTierSupplierConfig.from_env()
    return _config_instance


def set_config(config: MultiTierSupplierConfig) -> None:
    """Replace the singleton MultiTierSupplierConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New MultiTierSupplierConfig to install.

    Example:
        >>> cfg = MultiTierSupplierConfig(max_tier_depth=10)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "MultiTierSupplierConfig replaced programmatically: "
        "max_tier_depth=%d, propagation=%s, batch_max=%d",
        config.max_tier_depth,
        config.default_risk_propagation_method,
        config.batch_max_size,
    )


def reset_config() -> None:
    """Reset the singleton MultiTierSupplierConfig to None.

    The next call to get_config() will re-read GL_EUDR_MST_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("MultiTierSupplierConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "MultiTierSupplierConfig",
    "get_config",
    "set_config",
    "reset_config",
]
