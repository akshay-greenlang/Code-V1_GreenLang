# -*- coding: utf-8 -*-
"""
Segregation Verifier Configuration - AGENT-EUDR-010

Centralized configuration for the Segregation Verifier Agent covering:
- Storage verification: barrier types, zone separation, adjacent risk scoring
- Transport verification: cleaning verification, cargo history, dedication bonus
- Processing verification: changeover time, flush volume, first-run flagging
- Contamination detection: temporal/spatial proximity, auto-downgrade rules
- Labeling: required fields, color code maps, label type requirements
- Facility assessment: weighted scoring (layout/protocols/history/labeling/docs)
- SCP verification: reverification intervals, risk classification thresholds
- Batch processing: size limits, concurrency, timeout
- Data retention: EUDR Article 31 five-year retention (1825 days)
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_SGV_`` prefix (e.g. ``GL_EUDR_SGV_DATABASE_URL``,
``GL_EUDR_SGV_MIN_ZONE_SEPARATION_METERS``).

Environment Variable Reference (GL_EUDR_SGV_ prefix):
    GL_EUDR_SGV_DATABASE_URL                    - PostgreSQL connection URL
    GL_EUDR_SGV_REDIS_URL                       - Redis connection URL
    GL_EUDR_SGV_LOG_LEVEL                       - Logging level
    GL_EUDR_SGV_MIN_ZONE_SEPARATION_METERS      - Min zone separation (m)
    GL_EUDR_SGV_MAX_ADJACENT_RISK_SCORE          - Max adjacent risk score
    GL_EUDR_SGV_CLEANING_VERIFICATION_REQUIRED   - Require cleaning verification
    GL_EUDR_SGV_MAX_PREVIOUS_CARGOES_TRACKED     - Max previous cargoes tracked
    GL_EUDR_SGV_DEDICATED_VEHICLE_BONUS_SCORE    - Dedicated vehicle bonus
    GL_EUDR_SGV_MIN_CHANGEOVER_TIME_MINUTES      - Min changeover time (min)
    GL_EUDR_SGV_FLUSH_VOLUME_THRESHOLD           - Flush volume threshold (L)
    GL_EUDR_SGV_FIRST_RUN_AFTER_CHANGEOVER_FLAG  - First-run flagging enabled
    GL_EUDR_SGV_TEMPORAL_PROXIMITY_HOURS         - Temporal proximity (hours)
    GL_EUDR_SGV_SPATIAL_PROXIMITY_METERS         - Spatial proximity (meters)
    GL_EUDR_SGV_CONTAMINATION_AUTO_DOWNGRADE     - Auto-downgrade on contamination
    GL_EUDR_SGV_ASSESSMENT_LAYOUT_WEIGHT         - Assessment layout weight
    GL_EUDR_SGV_ASSESSMENT_PROTOCOL_WEIGHT       - Assessment protocol weight
    GL_EUDR_SGV_ASSESSMENT_HISTORY_WEIGHT        - Assessment history weight
    GL_EUDR_SGV_ASSESSMENT_LABELING_WEIGHT       - Assessment labeling weight
    GL_EUDR_SGV_ASSESSMENT_DOCUMENTATION_WEIGHT  - Assessment documentation weight
    GL_EUDR_SGV_MIN_REASSESSMENT_SCORE           - Min reassessment score
    GL_EUDR_SGV_REVERIFICATION_INTERVAL_DAYS     - Reverification interval (days)
    GL_EUDR_SGV_RISK_THRESHOLD_LOW               - Risk threshold: low
    GL_EUDR_SGV_RISK_THRESHOLD_MEDIUM            - Risk threshold: medium
    GL_EUDR_SGV_RISK_THRESHOLD_HIGH              - Risk threshold: high
    GL_EUDR_SGV_BATCH_MAX_SIZE                   - Maximum batch size
    GL_EUDR_SGV_BATCH_CONCURRENCY                - Batch concurrency
    GL_EUDR_SGV_BATCH_TIMEOUT_S                  - Batch timeout seconds
    GL_EUDR_SGV_RETENTION_YEARS                  - Data retention years
    GL_EUDR_SGV_REPORT_DEFAULT_FORMAT            - Default report format
    GL_EUDR_SGV_REPORT_RETENTION_DAYS            - Report retention days
    GL_EUDR_SGV_ENABLE_PROVENANCE                - Enable provenance tracking
    GL_EUDR_SGV_GENESIS_HASH                     - Genesis hash anchor
    GL_EUDR_SGV_ENABLE_METRICS                   - Enable Prometheus metrics
    GL_EUDR_SGV_POOL_SIZE                        - Database pool size
    GL_EUDR_SGV_RATE_LIMIT                       - Max requests per minute

Example:
    >>> from greenlang.agents.eudr.segregation_verifier.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.min_zone_separation_meters, cfg.temporal_proximity_hours)
    5.0 4.0

    >>> # Override for testing
    >>> from greenlang.agents.eudr.segregation_verifier.config import (
    ...     set_config, reset_config, SegregationVerifierConfig,
    ... )
    >>> set_config(SegregationVerifierConfig(min_zone_separation_meters=10.0))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier (GL-EUDR-SGV-010)
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

_ENV_PREFIX = "GL_EUDR_SGV_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid report formats
# ---------------------------------------------------------------------------

_VALID_REPORT_FORMATS = frozenset({"json", "pdf", "csv", "eudr_xml"})

# ---------------------------------------------------------------------------
# Default barrier types for storage zone separation
# ---------------------------------------------------------------------------

_DEFAULT_BARRIER_TYPES: List[str] = [
    "concrete_wall",
    "steel_partition",
    "wire_mesh_fence",
    "floor_marking",
    "plastic_curtain",
    "separate_room",
    "locked_cage",
    "sealed_container",
]

# ---------------------------------------------------------------------------
# Default required label fields
# ---------------------------------------------------------------------------

_DEFAULT_REQUIRED_LABEL_FIELDS: List[str] = [
    "compliance_status",
    "batch_id",
    "commodity",
    "date_applied",
    "operator_id",
]

# ---------------------------------------------------------------------------
# Default color code map for segregation zones
# ---------------------------------------------------------------------------

_DEFAULT_COLOR_CODE_MAP: Dict[str, str] = {
    "green": "compliant",
    "red": "non_compliant",
    "yellow": "pending",
    "blue": "buffer",
}

# ---------------------------------------------------------------------------
# Default EUDR commodities
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ---------------------------------------------------------------------------
# Default report formats
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_FORMATS: List[str] = ["json", "pdf", "csv", "eudr_xml"]


# ---------------------------------------------------------------------------
# SegregationVerifierConfig
# ---------------------------------------------------------------------------


@dataclass
class SegregationVerifierConfig:
    """Complete configuration for the EUDR Segregation Verifier Agent.

    Attributes are grouped by concern: connections, logging, storage
    verification, transport verification, processing verification,
    contamination detection, labeling, facility assessment, SCP
    verification, batch processing, reporting, data retention,
    provenance tracking, metrics, and performance tuning.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_SGV_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage
            of segregation control points, events, and assessments.
        redis_url: Redis connection URL for caching SCP lookups and
            distributed locks.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        default_barrier_types: Recognized barrier types for storage
            zone separation verification.
        min_zone_separation_meters: Minimum physical distance in meters
            between compliant and non-compliant storage zones.
        max_adjacent_risk_score: Maximum acceptable risk score for
            zones adjacent to non-compliant material.
        cleaning_verification_required: Whether cleaning must be
            verified before loading compliant cargo onto a vehicle.
        max_previous_cargoes_tracked: Number of previous cargoes
            tracked in transport vehicle history.
        dedicated_vehicle_bonus_score: Bonus score (0-100) awarded to
            vehicles dedicated exclusively to compliant cargo.
        min_changeover_time_minutes: Minimum changeover time in minutes
            between non-compliant and compliant processing runs.
        flush_volume_threshold: Minimum flush volume in liters required
            during changeover for liquid processing lines.
        first_run_after_changeover_flag: Whether to flag the first
            production run after a changeover for enhanced monitoring.
        temporal_proximity_hours: Maximum time window in hours for
            detecting temporal proximity contamination risks.
        spatial_proximity_meters: Maximum distance in meters for
            detecting spatial proximity contamination risks.
        contamination_auto_downgrade: Whether to automatically downgrade
            affected batch compliance status on contamination detection.
        required_label_fields: List of mandatory fields on compliance
            labels and zone signs.
        color_code_map: Mapping of color codes to compliance statuses
            for visual zone identification.
        assessment_layout_weight: Weight for layout score in facility
            assessment (0.0-1.0, sum of all weights must equal 1.0).
        assessment_protocol_weight: Weight for protocol score in
            facility assessment.
        assessment_history_weight: Weight for history score in facility
            assessment.
        assessment_labeling_weight: Weight for labeling score in
            facility assessment.
        assessment_documentation_weight: Weight for documentation score
            in facility assessment.
        min_reassessment_score: Minimum overall score (0-100) required
            to pass facility reassessment.
        reverification_interval_days: Number of days between mandatory
            SCP reverification inspections.
        risk_threshold_low: Score threshold below which risk is low.
        risk_threshold_medium: Score threshold below which risk is medium.
        risk_threshold_high: Score threshold below which risk is high.
        batch_max_size: Maximum number of records in a single batch
            processing job.
        batch_concurrency: Maximum concurrent batch processing workers.
        batch_timeout_s: Timeout in seconds for a single batch job.
        report_default_format: Default output format for reports.
        report_retention_days: Number of days to retain generated
            reports (1825 = 5 years per EUDR Article 14).
        report_formats: Supported report output formats.
        retention_years: Data retention in years per EUDR Article 31.
        eudr_commodities: List of EUDR-regulated commodity types.
        enable_provenance: Enable SHA-256 provenance chain tracking
            for all segregation verification operations.
        genesis_hash: Genesis anchor string for the provenance chain,
            unique to the Segregation Verifier agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_sgv_`` prefix.
        pool_size: PostgreSQL connection pool size.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Storage verification ------------------------------------------------
    default_barrier_types: List[str] = field(
        default_factory=lambda: list(_DEFAULT_BARRIER_TYPES)
    )
    min_zone_separation_meters: float = 5.0
    max_adjacent_risk_score: float = 30.0

    # -- Transport verification ----------------------------------------------
    cleaning_verification_required: bool = True
    max_previous_cargoes_tracked: int = 5
    dedicated_vehicle_bonus_score: float = 20.0

    # -- Processing verification ---------------------------------------------
    min_changeover_time_minutes: int = 60
    flush_volume_threshold: float = 50.0
    first_run_after_changeover_flag: bool = True

    # -- Contamination detection ---------------------------------------------
    temporal_proximity_hours: float = 4.0
    spatial_proximity_meters: float = 5.0
    contamination_auto_downgrade: bool = True

    # -- Labeling ------------------------------------------------------------
    required_label_fields: List[str] = field(
        default_factory=lambda: list(_DEFAULT_REQUIRED_LABEL_FIELDS)
    )
    color_code_map: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_COLOR_CODE_MAP)
    )

    # -- Facility assessment weights (must sum to 1.0) -----------------------
    assessment_layout_weight: float = 0.30
    assessment_protocol_weight: float = 0.25
    assessment_history_weight: float = 0.20
    assessment_labeling_weight: float = 0.15
    assessment_documentation_weight: float = 0.10

    # -- Facility assessment thresholds --------------------------------------
    min_reassessment_score: float = 60.0

    # -- SCP verification ----------------------------------------------------
    reverification_interval_days: int = 90
    risk_threshold_low: float = 80.0
    risk_threshold_medium: float = 60.0
    risk_threshold_high: float = 40.0

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 100_000
    batch_concurrency: int = 8
    batch_timeout_s: int = 300

    # -- Reporting -----------------------------------------------------------
    report_default_format: str = "json"
    report_retention_days: int = 1825
    report_formats: List[str] = field(
        default_factory=lambda: list(_DEFAULT_REPORT_FORMATS)
    )

    # -- Data retention (EUDR Article 31) ------------------------------------
    retention_years: int = 5

    # -- EUDR commodities ----------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-SGV-010-SEGREGATION-VERIFIER-GENESIS"

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

        # -- Storage verification -------------------------------------------
        if self.min_zone_separation_meters < 0.0:
            errors.append(
                f"min_zone_separation_meters must be >= 0, "
                f"got {self.min_zone_separation_meters}"
            )
        if self.min_zone_separation_meters > 1000.0:
            errors.append(
                f"min_zone_separation_meters must be <= 1000, "
                f"got {self.min_zone_separation_meters}"
            )

        if not (0.0 <= self.max_adjacent_risk_score <= 100.0):
            errors.append(
                f"max_adjacent_risk_score must be in [0, 100], "
                f"got {self.max_adjacent_risk_score}"
            )

        # -- Transport verification -----------------------------------------
        if not (1 <= self.max_previous_cargoes_tracked <= 100):
            errors.append(
                f"max_previous_cargoes_tracked must be in [1, 100], "
                f"got {self.max_previous_cargoes_tracked}"
            )

        if not (0.0 <= self.dedicated_vehicle_bonus_score <= 100.0):
            errors.append(
                f"dedicated_vehicle_bonus_score must be in [0, 100], "
                f"got {self.dedicated_vehicle_bonus_score}"
            )

        # -- Processing verification ----------------------------------------
        if self.min_changeover_time_minutes < 0:
            errors.append(
                f"min_changeover_time_minutes must be >= 0, "
                f"got {self.min_changeover_time_minutes}"
            )
        if self.min_changeover_time_minutes > 1440:
            errors.append(
                f"min_changeover_time_minutes must be <= 1440 (24h), "
                f"got {self.min_changeover_time_minutes}"
            )

        if self.flush_volume_threshold < 0.0:
            errors.append(
                f"flush_volume_threshold must be >= 0, "
                f"got {self.flush_volume_threshold}"
            )

        # -- Contamination detection ----------------------------------------
        if self.temporal_proximity_hours <= 0.0:
            errors.append(
                f"temporal_proximity_hours must be > 0, "
                f"got {self.temporal_proximity_hours}"
            )
        if self.temporal_proximity_hours > 720.0:
            errors.append(
                f"temporal_proximity_hours must be <= 720 (30 days), "
                f"got {self.temporal_proximity_hours}"
            )

        if self.spatial_proximity_meters < 0.0:
            errors.append(
                f"spatial_proximity_meters must be >= 0, "
                f"got {self.spatial_proximity_meters}"
            )
        if self.spatial_proximity_meters > 10000.0:
            errors.append(
                f"spatial_proximity_meters must be <= 10000, "
                f"got {self.spatial_proximity_meters}"
            )

        # -- Assessment weights (must sum to 1.0) ---------------------------
        weights = [
            self.assessment_layout_weight,
            self.assessment_protocol_weight,
            self.assessment_history_weight,
            self.assessment_labeling_weight,
            self.assessment_documentation_weight,
        ]
        for i, w in enumerate(weights):
            if not (0.0 <= w <= 1.0):
                weight_names = [
                    "layout", "protocol", "history",
                    "labeling", "documentation",
                ]
                errors.append(
                    f"assessment_{weight_names[i]}_weight must be in "
                    f"[0.0, 1.0], got {w}"
                )

        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(
                f"Assessment weights must sum to 1.0, "
                f"got {weight_sum:.4f}"
            )

        # -- Assessment thresholds ------------------------------------------
        if not (0.0 <= self.min_reassessment_score <= 100.0):
            errors.append(
                f"min_reassessment_score must be in [0, 100], "
                f"got {self.min_reassessment_score}"
            )

        # -- SCP verification -----------------------------------------------
        if not (1 <= self.reverification_interval_days <= 365):
            errors.append(
                f"reverification_interval_days must be in [1, 365], "
                f"got {self.reverification_interval_days}"
            )

        # Risk threshold ordering: low > medium > high
        if not (0.0 <= self.risk_threshold_high <= 100.0):
            errors.append(
                f"risk_threshold_high must be in [0, 100], "
                f"got {self.risk_threshold_high}"
            )
        if not (0.0 <= self.risk_threshold_medium <= 100.0):
            errors.append(
                f"risk_threshold_medium must be in [0, 100], "
                f"got {self.risk_threshold_medium}"
            )
        if not (0.0 <= self.risk_threshold_low <= 100.0):
            errors.append(
                f"risk_threshold_low must be in [0, 100], "
                f"got {self.risk_threshold_low}"
            )
        if self.risk_threshold_high >= self.risk_threshold_medium:
            errors.append(
                f"risk_threshold_high ({self.risk_threshold_high}) "
                f"must be < risk_threshold_medium "
                f"({self.risk_threshold_medium})"
            )
        if self.risk_threshold_medium >= self.risk_threshold_low:
            errors.append(
                f"risk_threshold_medium ({self.risk_threshold_medium}) "
                f"must be < risk_threshold_low "
                f"({self.risk_threshold_low})"
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

        if self.batch_timeout_s < 1:
            errors.append(
                f"batch_timeout_s must be >= 1, got {self.batch_timeout_s}"
            )

        # -- Report settings -------------------------------------------------
        if self.report_default_format not in _VALID_REPORT_FORMATS:
            errors.append(
                f"report_default_format must be one of "
                f"{sorted(_VALID_REPORT_FORMATS)}, "
                f"got '{self.report_default_format}'"
            )

        if self.report_retention_days < 1:
            errors.append(
                f"report_retention_days must be >= 1, "
                f"got {self.report_retention_days}"
            )

        if not self.report_formats:
            errors.append("report_formats must not be empty")

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
                "SegregationVerifierConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "SegregationVerifierConfig validated successfully: "
            "zone_sep=%.1fm, temporal=%.1fh, spatial=%.1fm, "
            "changeover=%dmin, reverification=%dd, "
            "assessment_weights=[%.2f,%.2f,%.2f,%.2f,%.2f], "
            "batch_max=%d, concurrency=%d, retention=%dy, "
            "provenance=%s, metrics=%s",
            self.min_zone_separation_meters,
            self.temporal_proximity_hours,
            self.spatial_proximity_meters,
            self.min_changeover_time_minutes,
            self.reverification_interval_days,
            self.assessment_layout_weight,
            self.assessment_protocol_weight,
            self.assessment_history_weight,
            self.assessment_labeling_weight,
            self.assessment_documentation_weight,
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
    def from_env(cls) -> SegregationVerifierConfig:
        """Build a SegregationVerifierConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_SGV_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated SegregationVerifierConfig instance, validated via
            ``__post_init__``.
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
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Storage verification
            min_zone_separation_meters=_float(
                "MIN_ZONE_SEPARATION_METERS",
                cls.min_zone_separation_meters,
            ),
            max_adjacent_risk_score=_float(
                "MAX_ADJACENT_RISK_SCORE",
                cls.max_adjacent_risk_score,
            ),
            # Transport verification
            cleaning_verification_required=_bool(
                "CLEANING_VERIFICATION_REQUIRED",
                cls.cleaning_verification_required,
            ),
            max_previous_cargoes_tracked=_int(
                "MAX_PREVIOUS_CARGOES_TRACKED",
                cls.max_previous_cargoes_tracked,
            ),
            dedicated_vehicle_bonus_score=_float(
                "DEDICATED_VEHICLE_BONUS_SCORE",
                cls.dedicated_vehicle_bonus_score,
            ),
            # Processing verification
            min_changeover_time_minutes=_int(
                "MIN_CHANGEOVER_TIME_MINUTES",
                cls.min_changeover_time_minutes,
            ),
            flush_volume_threshold=_float(
                "FLUSH_VOLUME_THRESHOLD",
                cls.flush_volume_threshold,
            ),
            first_run_after_changeover_flag=_bool(
                "FIRST_RUN_AFTER_CHANGEOVER_FLAG",
                cls.first_run_after_changeover_flag,
            ),
            # Contamination detection
            temporal_proximity_hours=_float(
                "TEMPORAL_PROXIMITY_HOURS",
                cls.temporal_proximity_hours,
            ),
            spatial_proximity_meters=_float(
                "SPATIAL_PROXIMITY_METERS",
                cls.spatial_proximity_meters,
            ),
            contamination_auto_downgrade=_bool(
                "CONTAMINATION_AUTO_DOWNGRADE",
                cls.contamination_auto_downgrade,
            ),
            # Assessment weights
            assessment_layout_weight=_float(
                "ASSESSMENT_LAYOUT_WEIGHT",
                cls.assessment_layout_weight,
            ),
            assessment_protocol_weight=_float(
                "ASSESSMENT_PROTOCOL_WEIGHT",
                cls.assessment_protocol_weight,
            ),
            assessment_history_weight=_float(
                "ASSESSMENT_HISTORY_WEIGHT",
                cls.assessment_history_weight,
            ),
            assessment_labeling_weight=_float(
                "ASSESSMENT_LABELING_WEIGHT",
                cls.assessment_labeling_weight,
            ),
            assessment_documentation_weight=_float(
                "ASSESSMENT_DOCUMENTATION_WEIGHT",
                cls.assessment_documentation_weight,
            ),
            min_reassessment_score=_float(
                "MIN_REASSESSMENT_SCORE",
                cls.min_reassessment_score,
            ),
            # SCP verification
            reverification_interval_days=_int(
                "REVERIFICATION_INTERVAL_DAYS",
                cls.reverification_interval_days,
            ),
            risk_threshold_low=_float(
                "RISK_THRESHOLD_LOW",
                cls.risk_threshold_low,
            ),
            risk_threshold_medium=_float(
                "RISK_THRESHOLD_MEDIUM",
                cls.risk_threshold_medium,
            ),
            risk_threshold_high=_float(
                "RISK_THRESHOLD_HIGH",
                cls.risk_threshold_high,
            ),
            # Batch processing
            batch_max_size=_int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            batch_concurrency=_int(
                "BATCH_CONCURRENCY", cls.batch_concurrency,
            ),
            batch_timeout_s=_int(
                "BATCH_TIMEOUT_S", cls.batch_timeout_s,
            ),
            # Reporting
            report_default_format=_str(
                "REPORT_DEFAULT_FORMAT", cls.report_default_format,
            ),
            report_retention_days=_int(
                "REPORT_RETENTION_DAYS", cls.report_retention_days,
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
            "SegregationVerifierConfig loaded: "
            "zone_sep=%.1fm, adjacent_risk=%.1f, "
            "cleaning_req=%s, cargoes_tracked=%d, "
            "dedicated_bonus=%.1f, changeover=%dmin, "
            "flush_threshold=%.1fL, first_run_flag=%s, "
            "temporal=%.1fh, spatial=%.1fm, auto_downgrade=%s, "
            "weights=[%.2f,%.2f,%.2f,%.2f,%.2f], "
            "min_reassess=%.1f, reverify=%dd, "
            "risk=[%.1f,%.1f,%.1f], "
            "batch_max=%d, concurrency=%d, timeout=%ds, "
            "retention=%dy, report_format=%s, "
            "report_retention=%dd, "
            "provenance=%s, pool=%d, rate_limit=%d/min, "
            "metrics=%s",
            config.min_zone_separation_meters,
            config.max_adjacent_risk_score,
            config.cleaning_verification_required,
            config.max_previous_cargoes_tracked,
            config.dedicated_vehicle_bonus_score,
            config.min_changeover_time_minutes,
            config.flush_volume_threshold,
            config.first_run_after_changeover_flag,
            config.temporal_proximity_hours,
            config.spatial_proximity_meters,
            config.contamination_auto_downgrade,
            config.assessment_layout_weight,
            config.assessment_protocol_weight,
            config.assessment_history_weight,
            config.assessment_labeling_weight,
            config.assessment_documentation_weight,
            config.min_reassessment_score,
            config.reverification_interval_days,
            config.risk_threshold_low,
            config.risk_threshold_medium,
            config.risk_threshold_high,
            config.batch_max_size,
            config.batch_concurrency,
            config.batch_timeout_s,
            config.retention_years,
            config.report_default_format,
            config.report_retention_days,
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
    def storage_settings(self) -> Dict[str, Any]:
        """Return storage verification settings as a dictionary.

        Returns:
            Dictionary with keys: min_zone_separation_meters,
            max_adjacent_risk_score, default_barrier_types.
        """
        return {
            "min_zone_separation_meters": self.min_zone_separation_meters,
            "max_adjacent_risk_score": self.max_adjacent_risk_score,
            "default_barrier_types": list(self.default_barrier_types),
        }

    @property
    def transport_settings(self) -> Dict[str, Any]:
        """Return transport verification settings as a dictionary.

        Returns:
            Dictionary with keys: cleaning_verification_required,
            max_previous_cargoes_tracked, dedicated_vehicle_bonus_score.
        """
        return {
            "cleaning_verification_required": (
                self.cleaning_verification_required
            ),
            "max_previous_cargoes_tracked": (
                self.max_previous_cargoes_tracked
            ),
            "dedicated_vehicle_bonus_score": (
                self.dedicated_vehicle_bonus_score
            ),
        }

    @property
    def processing_settings(self) -> Dict[str, Any]:
        """Return processing verification settings as a dictionary.

        Returns:
            Dictionary with keys: min_changeover_time_minutes,
            flush_volume_threshold, first_run_after_changeover_flag.
        """
        return {
            "min_changeover_time_minutes": self.min_changeover_time_minutes,
            "flush_volume_threshold": self.flush_volume_threshold,
            "first_run_after_changeover_flag": (
                self.first_run_after_changeover_flag
            ),
        }

    @property
    def contamination_settings(self) -> Dict[str, Any]:
        """Return contamination detection settings as a dictionary.

        Returns:
            Dictionary with keys: temporal_proximity_hours,
            spatial_proximity_meters, contamination_auto_downgrade.
        """
        return {
            "temporal_proximity_hours": self.temporal_proximity_hours,
            "spatial_proximity_meters": self.spatial_proximity_meters,
            "contamination_auto_downgrade": self.contamination_auto_downgrade,
        }

    @property
    def assessment_weights(self) -> Dict[str, float]:
        """Return facility assessment weights as a dictionary.

        Returns:
            Dictionary with keys: layout, protocol, history,
            labeling, documentation.
        """
        return {
            "layout": self.assessment_layout_weight,
            "protocol": self.assessment_protocol_weight,
            "history": self.assessment_history_weight,
            "labeling": self.assessment_labeling_weight,
            "documentation": self.assessment_documentation_weight,
        }

    @property
    def risk_thresholds(self) -> Dict[str, float]:
        """Return risk classification thresholds as a dictionary.

        Returns:
            Dictionary with keys: low, medium, high.
        """
        return {
            "low": self.risk_threshold_low,
            "medium": self.risk_threshold_medium,
            "high": self.risk_threshold_high,
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
            # Storage verification
            "default_barrier_types": list(self.default_barrier_types),
            "min_zone_separation_meters": self.min_zone_separation_meters,
            "max_adjacent_risk_score": self.max_adjacent_risk_score,
            # Transport verification
            "cleaning_verification_required": (
                self.cleaning_verification_required
            ),
            "max_previous_cargoes_tracked": (
                self.max_previous_cargoes_tracked
            ),
            "dedicated_vehicle_bonus_score": (
                self.dedicated_vehicle_bonus_score
            ),
            # Processing verification
            "min_changeover_time_minutes": self.min_changeover_time_minutes,
            "flush_volume_threshold": self.flush_volume_threshold,
            "first_run_after_changeover_flag": (
                self.first_run_after_changeover_flag
            ),
            # Contamination detection
            "temporal_proximity_hours": self.temporal_proximity_hours,
            "spatial_proximity_meters": self.spatial_proximity_meters,
            "contamination_auto_downgrade": self.contamination_auto_downgrade,
            # Labeling
            "required_label_fields": list(self.required_label_fields),
            "color_code_map": dict(self.color_code_map),
            # Assessment weights
            "assessment_layout_weight": self.assessment_layout_weight,
            "assessment_protocol_weight": self.assessment_protocol_weight,
            "assessment_history_weight": self.assessment_history_weight,
            "assessment_labeling_weight": self.assessment_labeling_weight,
            "assessment_documentation_weight": (
                self.assessment_documentation_weight
            ),
            "min_reassessment_score": self.min_reassessment_score,
            # SCP verification
            "reverification_interval_days": (
                self.reverification_interval_days
            ),
            "risk_threshold_low": self.risk_threshold_low,
            "risk_threshold_medium": self.risk_threshold_medium,
            "risk_threshold_high": self.risk_threshold_high,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Reporting
            "report_default_format": self.report_default_format,
            "report_retention_days": self.report_retention_days,
            "report_formats": list(self.report_formats),
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
        return f"SegregationVerifierConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[SegregationVerifierConfig] = None
_config_lock = threading.Lock()


def get_config() -> SegregationVerifierConfig:
    """Return the singleton SegregationVerifierConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_SGV_*`` environment variables.

    Returns:
        SegregationVerifierConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.min_zone_separation_meters
        5.0
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = SegregationVerifierConfig.from_env()
    return _config_instance


def set_config(config: SegregationVerifierConfig) -> None:
    """Replace the singleton SegregationVerifierConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New SegregationVerifierConfig to install.

    Example:
        >>> cfg = SegregationVerifierConfig(min_zone_separation_meters=10.0)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "SegregationVerifierConfig replaced programmatically: "
        "zone_sep=%.1fm, temporal=%.1fh, "
        "reverification=%dd, batch_max=%d",
        config.min_zone_separation_meters,
        config.temporal_proximity_hours,
        config.reverification_interval_days,
        config.batch_max_size,
    )


def reset_config() -> None:
    """Reset the singleton SegregationVerifierConfig to None.

    The next call to get_config() will re-read GL_EUDR_SGV_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("SegregationVerifierConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "SegregationVerifierConfig",
    "get_config",
    "set_config",
    "reset_config",
]
