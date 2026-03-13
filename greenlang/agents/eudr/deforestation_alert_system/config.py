# -*- coding: utf-8 -*-
"""
Deforestation Alert System Configuration - AGENT-EUDR-020

Centralized configuration for the Deforestation Alert System Agent covering:
- Database and cache connection settings (PostgreSQL, Redis) with configurable
  pool sizes, timeouts, and key prefixes using ``gl_eudr_das_`` namespace
- Satellite source enablement and parameters: Sentinel-2 (10m resolution,
  5-day revisit, 20% max cloud cover), Landsat 8/9 (30m resolution, 8-day
  revisit), GLAD weekly alerts, Hansen Global Forest Change annual data,
  RADD radar alerts with per-source enable/disable toggles
- Change detection thresholds: NDVI change threshold (-0.15 indicating
  significant vegetation loss), EVI change threshold (-0.12), minimum
  clearing area (0.5 ha for detection sensitivity), maximum cloud cover
  percentage (20%), temporal analysis window (30 days), and confidence
  threshold (0.75 minimum for alert generation)
- Alert generation parameters: batch size (1000), real-time streaming
  enablement, deduplication window (72 hours to prevent duplicate alerts
  for the same event), daily alert cap (10000), and retention period
  (5 years per EUDR Article 31)
- Severity classification: area thresholds (critical >=50 ha, high >=10 ha,
  medium >=1 ha), proximity thresholds (critical <=1 km, high <=5 km,
  medium <=25 km from supply chain plots), multipliers for protected area
  overlay (1.5x) and post-cutoff events (2.0x), and weighted scoring
  across five dimensions: area (0.25), deforestation rate (0.20),
  proximity to plots (0.25), protected area status (0.15), and
  post-cutoff timing (0.15)
- Spatial buffer monitoring: default radius (10 km), minimum (1 km),
  maximum (50 km), and resolution (64 points for buffer geometry)
- EUDR cutoff date verification: cutoff date (2020-12-31 per Article 2(1)),
  pre-cutoff grace period (90 days), minimum temporal evidence sources (2),
  and cutoff confidence threshold (0.85)
- Historical baseline settings: reference period 2018-2020, minimum 3
  baseline samples, canopy cover threshold at 10%
- Workflow management: auto-triage enablement, SLA deadlines (triage 4h,
  investigation 48h, resolution 168h), escalation up to 3 levels
- Compliance settings: auto impact assessment, market restriction at
  HIGH severity, remediation plan required
- Rate limiting across 5 tiers: anonymous (10/min), basic (60/min),
  standard (300/min), premium (1000/min), admin (10000/min)

All settings can be overridden via environment variables with the
``GL_EUDR_DAS_`` prefix (e.g. ``GL_EUDR_DAS_DATABASE_URL``,
``GL_EUDR_DAS_NDVI_CHANGE_THRESHOLD``).

Environment Variable Reference (GL_EUDR_DAS_ prefix):
    GL_EUDR_DAS_DATABASE_URL                    - PostgreSQL connection URL
    GL_EUDR_DAS_REDIS_URL                       - Redis connection URL
    GL_EUDR_DAS_LOG_LEVEL                       - Logging level
    GL_EUDR_DAS_POOL_SIZE                       - Database pool size
    GL_EUDR_DAS_POOL_TIMEOUT_S                  - Pool timeout seconds
    GL_EUDR_DAS_POOL_RECYCLE_S                  - Pool recycle seconds
    GL_EUDR_DAS_REDIS_TTL_S                     - Redis cache TTL seconds
    GL_EUDR_DAS_REDIS_KEY_PREFIX                - Redis key prefix
    GL_EUDR_DAS_SENTINEL2_ENABLED               - Enable Sentinel-2 source
    GL_EUDR_DAS_LANDSAT_ENABLED                 - Enable Landsat source
    GL_EUDR_DAS_GLAD_ENABLED                    - Enable GLAD alerts
    GL_EUDR_DAS_HANSEN_GFC_ENABLED              - Enable Hansen GFC
    GL_EUDR_DAS_RADD_ENABLED                    - Enable RADD alerts
    GL_EUDR_DAS_SENTINEL2_RESOLUTION_M          - Sentinel-2 resolution (meters)
    GL_EUDR_DAS_LANDSAT_RESOLUTION_M            - Landsat resolution (meters)
    GL_EUDR_DAS_SENTINEL2_REVISIT_DAYS          - Sentinel-2 revisit period
    GL_EUDR_DAS_LANDSAT_REVISIT_DAYS            - Landsat revisit period
    GL_EUDR_DAS_CLOUD_COVER_MAX_PCT             - Max cloud cover percentage
    GL_EUDR_DAS_NDVI_CHANGE_THRESHOLD           - NDVI change threshold (negative)
    GL_EUDR_DAS_EVI_CHANGE_THRESHOLD            - EVI change threshold (negative)
    GL_EUDR_DAS_MIN_CLEARING_AREA_HA            - Min clearing area hectares
    GL_EUDR_DAS_MAX_CLOUD_COVER_PCT             - Max cloud cover for detection
    GL_EUDR_DAS_TEMPORAL_WINDOW_DAYS            - Temporal analysis window days
    GL_EUDR_DAS_CONFIDENCE_THRESHOLD            - Min confidence for alerts
    GL_EUDR_DAS_ALERT_BATCH_SIZE                - Alert batch processing size
    GL_EUDR_DAS_REAL_TIME_ENABLED               - Enable real-time alerting
    GL_EUDR_DAS_DEDUP_WINDOW_HOURS              - Deduplication window hours
    GL_EUDR_DAS_MAX_ALERTS_PER_DAY              - Max alerts per day
    GL_EUDR_DAS_ALERT_RETENTION_DAYS            - Alert retention days
    GL_EUDR_DAS_CRITICAL_AREA_THRESHOLD_HA      - Critical severity area (ha)
    GL_EUDR_DAS_HIGH_AREA_THRESHOLD_HA          - High severity area (ha)
    GL_EUDR_DAS_MEDIUM_AREA_THRESHOLD_HA        - Medium severity area (ha)
    GL_EUDR_DAS_PROXIMITY_CRITICAL_KM           - Critical proximity (km)
    GL_EUDR_DAS_PROXIMITY_HIGH_KM               - High proximity (km)
    GL_EUDR_DAS_PROXIMITY_MEDIUM_KM             - Medium proximity (km)
    GL_EUDR_DAS_PROTECTED_AREA_MULTIPLIER       - Protected area score multiplier
    GL_EUDR_DAS_POST_CUTOFF_MULTIPLIER          - Post-cutoff score multiplier
    GL_EUDR_DAS_AREA_WEIGHT                     - Severity area weight
    GL_EUDR_DAS_RATE_WEIGHT                     - Severity rate weight
    GL_EUDR_DAS_PROXIMITY_WEIGHT                - Severity proximity weight
    GL_EUDR_DAS_PROTECTED_WEIGHT                - Severity protected area weight
    GL_EUDR_DAS_TIMING_WEIGHT                   - Severity timing weight
    GL_EUDR_DAS_DEFAULT_BUFFER_RADIUS_KM        - Default buffer radius (km)
    GL_EUDR_DAS_MIN_BUFFER_KM                   - Minimum buffer radius (km)
    GL_EUDR_DAS_MAX_BUFFER_KM                   - Maximum buffer radius (km)
    GL_EUDR_DAS_BUFFER_RESOLUTION_POINTS        - Buffer geometry resolution
    GL_EUDR_DAS_CUTOFF_DATE                     - EUDR cutoff date (YYYY-MM-DD)
    GL_EUDR_DAS_PRE_CUTOFF_GRACE_DAYS           - Pre-cutoff grace period days
    GL_EUDR_DAS_TEMPORAL_EVIDENCE_SOURCES_MIN   - Min temporal evidence sources
    GL_EUDR_DAS_CUTOFF_CONFIDENCE_THRESHOLD     - Cutoff verification confidence
    GL_EUDR_DAS_BASELINE_START_YEAR             - Historical baseline start year
    GL_EUDR_DAS_BASELINE_END_YEAR               - Historical baseline end year
    GL_EUDR_DAS_MIN_BASELINE_SAMPLES            - Min baseline sample count
    GL_EUDR_DAS_CANOPY_COVER_THRESHOLD_PCT      - Canopy cover threshold pct
    GL_EUDR_DAS_AUTO_TRIAGE_ENABLED             - Enable auto-triage
    GL_EUDR_DAS_SLA_TRIAGE_HOURS                - SLA triage deadline hours
    GL_EUDR_DAS_SLA_INVESTIGATION_HOURS         - SLA investigation hours
    GL_EUDR_DAS_SLA_RESOLUTION_HOURS            - SLA resolution hours
    GL_EUDR_DAS_ESCALATION_ENABLED              - Enable escalation
    GL_EUDR_DAS_MAX_ESCALATION_LEVELS           - Max escalation levels
    GL_EUDR_DAS_IMPACT_ASSESSMENT_AUTO          - Auto impact assessment
    GL_EUDR_DAS_MARKET_RESTRICTION_THRESHOLD    - Market restriction severity
    GL_EUDR_DAS_REMEDIATION_PLAN_REQUIRED       - Require remediation plans
    GL_EUDR_DAS_BATCH_MAX_SIZE                  - Batch processing max size
    GL_EUDR_DAS_BATCH_CONCURRENCY               - Batch concurrency workers
    GL_EUDR_DAS_BATCH_TIMEOUT_S                 - Batch timeout seconds
    GL_EUDR_DAS_RETENTION_YEARS                 - Data retention years
    GL_EUDR_DAS_ENABLE_PROVENANCE               - Enable provenance tracking
    GL_EUDR_DAS_GENESIS_HASH                    - Genesis hash anchor
    GL_EUDR_DAS_CHAIN_ALGORITHM                 - Hash algorithm
    GL_EUDR_DAS_ENABLE_METRICS                  - Enable Prometheus metrics
    GL_EUDR_DAS_METRICS_PREFIX                  - Prometheus metrics prefix
    GL_EUDR_DAS_RATE_LIMIT_ANONYMOUS            - Rate limit anonymous tier
    GL_EUDR_DAS_RATE_LIMIT_BASIC                - Rate limit basic tier
    GL_EUDR_DAS_RATE_LIMIT_STANDARD             - Rate limit standard tier
    GL_EUDR_DAS_RATE_LIMIT_PREMIUM              - Rate limit premium tier
    GL_EUDR_DAS_RATE_LIMIT_ADMIN                - Rate limit admin tier

Example:
    >>> from greenlang.agents.eudr.deforestation_alert_system.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.ndvi_change_threshold, cfg.cutoff_date)
    -0.15 2020-12-31

    >>> # Override for testing
    >>> from greenlang.agents.eudr.deforestation_alert_system.config import (
    ...     set_config, reset_config, DeforestationAlertSystemConfig,
    ... )
    >>> set_config(DeforestationAlertSystemConfig(ndvi_change_threshold=-0.20))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
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

_ENV_PREFIX = "GL_EUDR_DAS_"

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
# Valid market restriction thresholds (severity levels)
# ---------------------------------------------------------------------------

_VALID_MARKET_RESTRICTION_THRESHOLDS = frozenset(
    {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
)

# ---------------------------------------------------------------------------
# Valid output formats
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "excel", "csv", "geojson"})

# ---------------------------------------------------------------------------
# Valid report languages
# ---------------------------------------------------------------------------

_VALID_REPORT_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})

# ---------------------------------------------------------------------------
# Supported satellite sources
# ---------------------------------------------------------------------------

_SATELLITE_SOURCES = frozenset({
    "sentinel2",
    "landsat",
    "glad",
    "hansen_gfc",
    "radd",
    "planet",
    "custom",
})

# ---------------------------------------------------------------------------
# Severity weight dimension keys
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT_KEYS = frozenset({
    "area",
    "rate",
    "proximity",
    "protected",
    "timing",
})

# ---------------------------------------------------------------------------
# Default severity weights (sum = 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_SEVERITY_WEIGHTS: Dict[str, Decimal] = {
    "area": Decimal("0.25"),
    "rate": Decimal("0.20"),
    "proximity": Decimal("0.25"),
    "protected": Decimal("0.15"),
    "timing": Decimal("0.15"),
}

# ---------------------------------------------------------------------------
# Default satellite resolutions (meters)
# ---------------------------------------------------------------------------

_DEFAULT_SENTINEL2_RESOLUTION_M: int = 10
_DEFAULT_LANDSAT_RESOLUTION_M: int = 30

# ---------------------------------------------------------------------------
# Default revisit periods (days)
# ---------------------------------------------------------------------------

_DEFAULT_SENTINEL2_REVISIT_DAYS: int = 5
_DEFAULT_LANDSAT_REVISIT_DAYS: int = 8

# ---------------------------------------------------------------------------
# Default change detection thresholds
# ---------------------------------------------------------------------------

_DEFAULT_NDVI_CHANGE_THRESHOLD = Decimal("-0.15")
_DEFAULT_EVI_CHANGE_THRESHOLD = Decimal("-0.12")
_DEFAULT_MIN_CLEARING_AREA_HA = Decimal("0.5")
_DEFAULT_MAX_CLOUD_COVER_PCT: int = 20
_DEFAULT_TEMPORAL_WINDOW_DAYS: int = 30
_DEFAULT_CONFIDENCE_THRESHOLD = Decimal("0.75")

# ---------------------------------------------------------------------------
# Default severity area thresholds (hectares)
# ---------------------------------------------------------------------------

_DEFAULT_CRITICAL_AREA_HA = Decimal("50")
_DEFAULT_HIGH_AREA_HA = Decimal("10")
_DEFAULT_MEDIUM_AREA_HA = Decimal("1")

# ---------------------------------------------------------------------------
# Default proximity thresholds (kilometers)
# ---------------------------------------------------------------------------

_DEFAULT_PROXIMITY_CRITICAL_KM = Decimal("1")
_DEFAULT_PROXIMITY_HIGH_KM = Decimal("5")
_DEFAULT_PROXIMITY_MEDIUM_KM = Decimal("25")

# ---------------------------------------------------------------------------
# Default multipliers
# ---------------------------------------------------------------------------

_DEFAULT_PROTECTED_AREA_MULTIPLIER = Decimal("1.5")
_DEFAULT_POST_CUTOFF_MULTIPLIER = Decimal("2.0")

# ---------------------------------------------------------------------------
# Default buffer parameters
# ---------------------------------------------------------------------------

_DEFAULT_BUFFER_RADIUS_KM = Decimal("10")
_DEFAULT_MIN_BUFFER_KM = Decimal("1")
_DEFAULT_MAX_BUFFER_KM = Decimal("50")
_DEFAULT_BUFFER_RESOLUTION: int = 64

# ---------------------------------------------------------------------------
# Default cutoff date settings
# ---------------------------------------------------------------------------

_DEFAULT_CUTOFF_DATE = "2020-12-31"
_DEFAULT_PRE_CUTOFF_GRACE_DAYS: int = 90
_DEFAULT_TEMPORAL_EVIDENCE_MIN: int = 2
_DEFAULT_CUTOFF_CONFIDENCE = Decimal("0.85")

# ---------------------------------------------------------------------------
# Default baseline settings
# ---------------------------------------------------------------------------

_DEFAULT_BASELINE_START_YEAR: int = 2018
_DEFAULT_BASELINE_END_YEAR: int = 2020
_DEFAULT_MIN_BASELINE_SAMPLES: int = 3
_DEFAULT_CANOPY_COVER_THRESHOLD = Decimal("10")

# ---------------------------------------------------------------------------
# Default workflow SLA settings (hours)
# ---------------------------------------------------------------------------

_DEFAULT_SLA_TRIAGE_HOURS: int = 4
_DEFAULT_SLA_INVESTIGATION_HOURS: int = 48
_DEFAULT_SLA_RESOLUTION_HOURS: int = 168
_DEFAULT_MAX_ESCALATION_LEVELS: int = 3


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DeforestationAlertSystemConfig:
    """Configuration for the Deforestation Alert System Agent (AGENT-EUDR-020).

    This dataclass encapsulates all configuration settings for satellite
    change detection, alert generation, severity classification, spatial
    buffer monitoring, EUDR cutoff date verification, historical baseline
    comparison, alert workflow management, and compliance impact assessment.
    All settings have sensible defaults aligned with EUDR requirements and
    can be overridden via environment variables with the GL_EUDR_DAS_ prefix.

    Attributes:
        # Database settings
        database_url: PostgreSQL connection URL
        pool_size: Connection pool size
        pool_timeout_s: Connection pool timeout seconds
        pool_recycle_s: Connection pool recycle seconds

        # Redis settings
        redis_url: Redis connection URL
        redis_ttl_s: Redis cache TTL seconds
        redis_key_prefix: Redis key prefix

        # Logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        # Satellite source settings
        sentinel2_enabled: Enable Sentinel-2 data source
        landsat_enabled: Enable Landsat 8/9 data source
        glad_enabled: Enable GLAD weekly alerts
        hansen_gfc_enabled: Enable Hansen Global Forest Change
        radd_enabled: Enable RADD radar alerts
        sentinel2_resolution_m: Sentinel-2 spatial resolution (meters)
        landsat_resolution_m: Landsat spatial resolution (meters)
        sentinel2_revisit_days: Sentinel-2 revisit period (days)
        landsat_revisit_days: Landsat revisit period (days)
        cloud_cover_max_pct: Maximum cloud cover for satellite imagery (%)

        # Change detection thresholds
        ndvi_change_threshold: NDVI change threshold (negative = vegetation loss)
        evi_change_threshold: EVI change threshold (negative = vegetation loss)
        min_clearing_area_ha: Minimum clearing area for detection (hectares)
        max_cloud_cover_pct: Maximum cloud cover for detection (%)
        temporal_window_days: Temporal analysis window (days)
        confidence_threshold: Minimum confidence for alert generation (0-1)

        # Alert generation settings
        alert_batch_size: Batch processing size for alert generation
        real_time_enabled: Enable real-time alert streaming
        dedup_window_hours: Deduplication window to prevent duplicate alerts
        max_alerts_per_day: Maximum alerts generated per day
        alert_retention_days: Alert data retention period (days)

        # Severity classification thresholds
        critical_area_threshold_ha: Area threshold for CRITICAL severity
        high_area_threshold_ha: Area threshold for HIGH severity
        medium_area_threshold_ha: Area threshold for MEDIUM severity
        proximity_critical_km: Distance threshold for CRITICAL proximity
        proximity_high_km: Distance threshold for HIGH proximity
        proximity_medium_km: Distance threshold for MEDIUM proximity
        protected_area_multiplier: Score multiplier for protected areas
        post_cutoff_multiplier: Score multiplier for post-cutoff events

        # Severity weight settings (must sum to 1.0)
        area_weight: Weight for area component in severity scoring
        rate_weight: Weight for deforestation rate component
        proximity_weight: Weight for proximity component
        protected_weight: Weight for protected area component
        timing_weight: Weight for post-cutoff timing component

        # Spatial buffer settings
        default_buffer_radius_km: Default monitoring buffer radius (km)
        min_buffer_km: Minimum allowed buffer radius (km)
        max_buffer_km: Maximum allowed buffer radius (km)
        buffer_resolution_points: Points per buffer geometry circle

        # EUDR cutoff date settings
        cutoff_date: EUDR deforestation cutoff date (YYYY-MM-DD)
        pre_cutoff_grace_days: Grace period before cutoff (days)
        temporal_evidence_sources_min: Minimum evidence sources for cutoff
        cutoff_confidence_threshold: Confidence threshold for cutoff verification

        # Historical baseline settings
        baseline_start_year: Start year for baseline reference period
        baseline_end_year: End year for baseline reference period
        min_baseline_samples: Minimum number of baseline observations
        canopy_cover_threshold_pct: Minimum canopy cover to classify as forest

        # Workflow settings
        auto_triage_enabled: Enable automatic alert triage
        sla_triage_hours: SLA deadline for triage (hours)
        sla_investigation_hours: SLA deadline for investigation (hours)
        sla_resolution_hours: SLA deadline for resolution (hours)
        escalation_enabled: Enable automatic escalation
        max_escalation_levels: Maximum escalation levels

        # Compliance settings
        impact_assessment_auto: Enable automatic impact assessment
        market_restriction_threshold: Severity triggering market restriction
        remediation_plan_required: Require remediation plan for incidents

        # Reporting
        output_formats: Report output formats
        default_language: Default report language
        supported_languages: Supported report languages

        # Batch processing
        batch_max_size: Batch processing max size
        batch_concurrency: Batch concurrency workers
        batch_timeout_s: Batch timeout seconds

        # Data retention
        retention_years: Data retention years per EUDR Article 31

        # Provenance
        enable_provenance: Enable provenance tracking
        genesis_hash: Genesis hash anchor for provenance chain
        chain_algorithm: Hash algorithm for provenance chain

        # Metrics
        enable_metrics: Enable Prometheus metrics collection
        metrics_prefix: Prometheus metrics prefix

        # Rate limiting
        rate_limit_anonymous: Rate limit anonymous tier (requests/minute)
        rate_limit_basic: Rate limit basic tier (requests/minute)
        rate_limit_standard: Rate limit standard tier (requests/minute)
        rate_limit_premium: Rate limit premium tier (requests/minute)
        rate_limit_admin: Rate limit admin tier (requests/minute)
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
    redis_key_prefix: str = "gl:eudr:das:"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # Satellite source settings
    # -----------------------------------------------------------------------
    sentinel2_enabled: bool = True
    landsat_enabled: bool = True
    glad_enabled: bool = True
    hansen_gfc_enabled: bool = True
    radd_enabled: bool = True
    sentinel2_resolution_m: int = _DEFAULT_SENTINEL2_RESOLUTION_M
    landsat_resolution_m: int = _DEFAULT_LANDSAT_RESOLUTION_M
    sentinel2_revisit_days: int = _DEFAULT_SENTINEL2_REVISIT_DAYS
    landsat_revisit_days: int = _DEFAULT_LANDSAT_REVISIT_DAYS
    cloud_cover_max_pct: int = _DEFAULT_MAX_CLOUD_COVER_PCT

    # -----------------------------------------------------------------------
    # Change detection thresholds
    # -----------------------------------------------------------------------
    ndvi_change_threshold: Decimal = _DEFAULT_NDVI_CHANGE_THRESHOLD
    evi_change_threshold: Decimal = _DEFAULT_EVI_CHANGE_THRESHOLD
    min_clearing_area_ha: Decimal = _DEFAULT_MIN_CLEARING_AREA_HA
    max_cloud_cover_pct: int = _DEFAULT_MAX_CLOUD_COVER_PCT
    temporal_window_days: int = _DEFAULT_TEMPORAL_WINDOW_DAYS
    confidence_threshold: Decimal = _DEFAULT_CONFIDENCE_THRESHOLD

    # -----------------------------------------------------------------------
    # Alert generation settings
    # -----------------------------------------------------------------------
    alert_batch_size: int = 1000
    real_time_enabled: bool = True
    dedup_window_hours: int = 72
    max_alerts_per_day: int = 10000
    alert_retention_days: int = 365 * 5  # 5 years per EUDR Article 31

    # -----------------------------------------------------------------------
    # Severity classification thresholds
    # -----------------------------------------------------------------------
    critical_area_threshold_ha: Decimal = _DEFAULT_CRITICAL_AREA_HA
    high_area_threshold_ha: Decimal = _DEFAULT_HIGH_AREA_HA
    medium_area_threshold_ha: Decimal = _DEFAULT_MEDIUM_AREA_HA
    proximity_critical_km: Decimal = _DEFAULT_PROXIMITY_CRITICAL_KM
    proximity_high_km: Decimal = _DEFAULT_PROXIMITY_HIGH_KM
    proximity_medium_km: Decimal = _DEFAULT_PROXIMITY_MEDIUM_KM
    protected_area_multiplier: Decimal = _DEFAULT_PROTECTED_AREA_MULTIPLIER
    post_cutoff_multiplier: Decimal = _DEFAULT_POST_CUTOFF_MULTIPLIER

    # -----------------------------------------------------------------------
    # Severity weight settings (must sum to 1.0)
    # -----------------------------------------------------------------------
    area_weight: Decimal = _DEFAULT_SEVERITY_WEIGHTS["area"]
    rate_weight: Decimal = _DEFAULT_SEVERITY_WEIGHTS["rate"]
    proximity_weight: Decimal = _DEFAULT_SEVERITY_WEIGHTS["proximity"]
    protected_weight: Decimal = _DEFAULT_SEVERITY_WEIGHTS["protected"]
    timing_weight: Decimal = _DEFAULT_SEVERITY_WEIGHTS["timing"]

    # -----------------------------------------------------------------------
    # Spatial buffer settings
    # -----------------------------------------------------------------------
    default_buffer_radius_km: Decimal = _DEFAULT_BUFFER_RADIUS_KM
    min_buffer_km: Decimal = _DEFAULT_MIN_BUFFER_KM
    max_buffer_km: Decimal = _DEFAULT_MAX_BUFFER_KM
    buffer_resolution_points: int = _DEFAULT_BUFFER_RESOLUTION

    # -----------------------------------------------------------------------
    # EUDR cutoff date settings
    # -----------------------------------------------------------------------
    cutoff_date: str = _DEFAULT_CUTOFF_DATE
    pre_cutoff_grace_days: int = _DEFAULT_PRE_CUTOFF_GRACE_DAYS
    temporal_evidence_sources_min: int = _DEFAULT_TEMPORAL_EVIDENCE_MIN
    cutoff_confidence_threshold: Decimal = _DEFAULT_CUTOFF_CONFIDENCE

    # -----------------------------------------------------------------------
    # Historical baseline settings
    # -----------------------------------------------------------------------
    baseline_start_year: int = _DEFAULT_BASELINE_START_YEAR
    baseline_end_year: int = _DEFAULT_BASELINE_END_YEAR
    min_baseline_samples: int = _DEFAULT_MIN_BASELINE_SAMPLES
    canopy_cover_threshold_pct: Decimal = _DEFAULT_CANOPY_COVER_THRESHOLD

    # -----------------------------------------------------------------------
    # Workflow settings
    # -----------------------------------------------------------------------
    auto_triage_enabled: bool = True
    sla_triage_hours: int = _DEFAULT_SLA_TRIAGE_HOURS
    sla_investigation_hours: int = _DEFAULT_SLA_INVESTIGATION_HOURS
    sla_resolution_hours: int = _DEFAULT_SLA_RESOLUTION_HOURS
    escalation_enabled: bool = True
    max_escalation_levels: int = _DEFAULT_MAX_ESCALATION_LEVELS

    # -----------------------------------------------------------------------
    # Compliance settings
    # -----------------------------------------------------------------------
    impact_assessment_auto: bool = True
    market_restriction_threshold: str = "HIGH"
    remediation_plan_required: bool = True

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "geojson"]
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
    genesis_hash: str = "GL-EUDR-DAS-020-DEFORESTATION-ALERT-SYSTEM-GENESIS"
    chain_algorithm: str = "sha256"

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_das_"

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
        self._validate_satellite_resolutions()
        self._validate_change_detection()
        self._validate_alert_generation()
        self._validate_severity_thresholds()
        self._validate_severity_weights()
        self._validate_spatial_buffer()
        self._validate_cutoff_date()
        self._validate_baseline()
        self._validate_workflow()
        self._validate_compliance()
        self._validate_positive_integers()
        self._validate_output_formats()
        self._validate_languages()

        logger.info(
            f"DeforestationAlertSystemConfig initialized: "
            f"ndvi_threshold={self.ndvi_change_threshold}, "
            f"cutoff_date={self.cutoff_date}, "
            f"default_buffer_km={self.default_buffer_radius_km}, "
            f"sla_triage_h={self.sla_triage_hours}"
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

    def _validate_satellite_resolutions(self) -> None:
        """Validate satellite resolution and revisit parameters."""
        if self.sentinel2_resolution_m < 1:
            raise ValueError(
                f"sentinel2_resolution_m must be >= 1, "
                f"got {self.sentinel2_resolution_m}"
            )
        if self.landsat_resolution_m < 1:
            raise ValueError(
                f"landsat_resolution_m must be >= 1, "
                f"got {self.landsat_resolution_m}"
            )
        if self.sentinel2_revisit_days < 1:
            raise ValueError(
                f"sentinel2_revisit_days must be >= 1, "
                f"got {self.sentinel2_revisit_days}"
            )
        if self.landsat_revisit_days < 1:
            raise ValueError(
                f"landsat_revisit_days must be >= 1, "
                f"got {self.landsat_revisit_days}"
            )
        if not 0 <= self.cloud_cover_max_pct <= 100:
            raise ValueError(
                f"cloud_cover_max_pct must be between 0 and 100, "
                f"got {self.cloud_cover_max_pct}"
            )

    def _validate_change_detection(self) -> None:
        """Validate change detection threshold parameters."""
        if self.ndvi_change_threshold >= Decimal("0"):
            raise ValueError(
                f"ndvi_change_threshold must be negative (vegetation loss), "
                f"got {self.ndvi_change_threshold}"
            )
        if self.ndvi_change_threshold < Decimal("-1"):
            raise ValueError(
                f"ndvi_change_threshold must be >= -1.0, "
                f"got {self.ndvi_change_threshold}"
            )
        if self.evi_change_threshold >= Decimal("0"):
            raise ValueError(
                f"evi_change_threshold must be negative (vegetation loss), "
                f"got {self.evi_change_threshold}"
            )
        if self.evi_change_threshold < Decimal("-1"):
            raise ValueError(
                f"evi_change_threshold must be >= -1.0, "
                f"got {self.evi_change_threshold}"
            )
        if self.min_clearing_area_ha <= Decimal("0"):
            raise ValueError(
                f"min_clearing_area_ha must be > 0, "
                f"got {self.min_clearing_area_ha}"
            )
        if not 0 <= self.max_cloud_cover_pct <= 100:
            raise ValueError(
                f"max_cloud_cover_pct must be between 0 and 100, "
                f"got {self.max_cloud_cover_pct}"
            )
        if self.temporal_window_days < 1:
            raise ValueError(
                f"temporal_window_days must be >= 1, "
                f"got {self.temporal_window_days}"
            )
        if not Decimal("0") < self.confidence_threshold <= Decimal("1"):
            raise ValueError(
                f"confidence_threshold must be between 0 (exclusive) and 1 "
                f"(inclusive), got {self.confidence_threshold}"
            )

    def _validate_alert_generation(self) -> None:
        """Validate alert generation parameters."""
        if self.alert_batch_size < 1:
            raise ValueError(
                f"alert_batch_size must be >= 1, got {self.alert_batch_size}"
            )
        if self.dedup_window_hours < 0:
            raise ValueError(
                f"dedup_window_hours must be >= 0, "
                f"got {self.dedup_window_hours}"
            )
        if self.max_alerts_per_day < 1:
            raise ValueError(
                f"max_alerts_per_day must be >= 1, "
                f"got {self.max_alerts_per_day}"
            )
        if self.alert_retention_days < 1:
            raise ValueError(
                f"alert_retention_days must be >= 1, "
                f"got {self.alert_retention_days}"
            )

    def _validate_severity_thresholds(self) -> None:
        """Validate severity classification thresholds are ordered."""
        # Area thresholds must be in descending order
        if not (
            self.critical_area_threshold_ha
            > self.high_area_threshold_ha
            > self.medium_area_threshold_ha
            > Decimal("0")
        ):
            raise ValueError(
                "Severity area thresholds must be in descending order: "
                "critical > high > medium > 0. "
                f"Got critical={self.critical_area_threshold_ha}, "
                f"high={self.high_area_threshold_ha}, "
                f"medium={self.medium_area_threshold_ha}"
            )
        # Proximity thresholds must be in ascending order (closer = more severe)
        if not (
            Decimal("0")
            < self.proximity_critical_km
            < self.proximity_high_km
            < self.proximity_medium_km
        ):
            raise ValueError(
                "Severity proximity thresholds must be in ascending order: "
                "0 < critical < high < medium. "
                f"Got critical={self.proximity_critical_km}, "
                f"high={self.proximity_high_km}, "
                f"medium={self.proximity_medium_km}"
            )
        # Multipliers must be >= 1.0
        if self.protected_area_multiplier < Decimal("1"):
            raise ValueError(
                f"protected_area_multiplier must be >= 1.0, "
                f"got {self.protected_area_multiplier}"
            )
        if self.post_cutoff_multiplier < Decimal("1"):
            raise ValueError(
                f"post_cutoff_multiplier must be >= 1.0, "
                f"got {self.post_cutoff_multiplier}"
            )

    def _validate_severity_weights(self) -> None:
        """Validate severity weights are positive and sum to 1.0."""
        weights = [
            ("area_weight", self.area_weight),
            ("rate_weight", self.rate_weight),
            ("proximity_weight", self.proximity_weight),
            ("protected_weight", self.protected_weight),
            ("timing_weight", self.timing_weight),
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
                f"Severity weights must sum to 1.0, got {total}"
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

    def _validate_cutoff_date(self) -> None:
        """Validate EUDR cutoff date format and related parameters."""
        # Validate date format
        from datetime import date as date_type
        try:
            date_type.fromisoformat(self.cutoff_date)
        except ValueError:
            raise ValueError(
                f"cutoff_date must be in YYYY-MM-DD format, "
                f"got {self.cutoff_date}"
            )
        if self.pre_cutoff_grace_days < 0:
            raise ValueError(
                f"pre_cutoff_grace_days must be >= 0, "
                f"got {self.pre_cutoff_grace_days}"
            )
        if self.temporal_evidence_sources_min < 1:
            raise ValueError(
                f"temporal_evidence_sources_min must be >= 1, "
                f"got {self.temporal_evidence_sources_min}"
            )
        if not (
            Decimal("0") < self.cutoff_confidence_threshold <= Decimal("1")
        ):
            raise ValueError(
                f"cutoff_confidence_threshold must be between 0 (exclusive) "
                f"and 1 (inclusive), got {self.cutoff_confidence_threshold}"
            )

    def _validate_baseline(self) -> None:
        """Validate historical baseline parameters."""
        if self.baseline_start_year < 2000:
            raise ValueError(
                f"baseline_start_year must be >= 2000, "
                f"got {self.baseline_start_year}"
            )
        if self.baseline_end_year <= self.baseline_start_year:
            raise ValueError(
                f"baseline_end_year ({self.baseline_end_year}) must be > "
                f"baseline_start_year ({self.baseline_start_year})"
            )
        if self.min_baseline_samples < 1:
            raise ValueError(
                f"min_baseline_samples must be >= 1, "
                f"got {self.min_baseline_samples}"
            )
        if not (
            Decimal("0")
            <= self.canopy_cover_threshold_pct
            <= Decimal("100")
        ):
            raise ValueError(
                f"canopy_cover_threshold_pct must be between 0 and 100, "
                f"got {self.canopy_cover_threshold_pct}"
            )

    def _validate_workflow(self) -> None:
        """Validate workflow SLA and escalation parameters."""
        if self.sla_triage_hours < 1:
            raise ValueError(
                f"sla_triage_hours must be >= 1, "
                f"got {self.sla_triage_hours}"
            )
        if self.sla_investigation_hours <= self.sla_triage_hours:
            raise ValueError(
                f"sla_investigation_hours ({self.sla_investigation_hours}) "
                f"must be > sla_triage_hours ({self.sla_triage_hours})"
            )
        if self.sla_resolution_hours <= self.sla_investigation_hours:
            raise ValueError(
                f"sla_resolution_hours ({self.sla_resolution_hours}) "
                f"must be > sla_investigation_hours "
                f"({self.sla_investigation_hours})"
            )
        if self.max_escalation_levels < 1:
            raise ValueError(
                f"max_escalation_levels must be >= 1, "
                f"got {self.max_escalation_levels}"
            )

    def _validate_compliance(self) -> None:
        """Validate compliance configuration parameters."""
        if self.market_restriction_threshold not in _VALID_MARKET_RESTRICTION_THRESHOLDS:
            raise ValueError(
                f"Invalid market_restriction_threshold: "
                f"{self.market_restriction_threshold}. "
                f"Must be one of {_VALID_MARKET_RESTRICTION_THRESHOLDS}"
            )

    def _validate_positive_integers(self) -> None:
        """Validate fields that must be positive integers."""
        checks = [
            ("pool_size", self.pool_size),
            ("batch_max_size", self.batch_max_size),
            ("batch_concurrency", self.batch_concurrency),
            ("retention_years", self.retention_years),
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

    @classmethod
    def from_env(cls) -> "DeforestationAlertSystemConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_DAS_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            DeforestationAlertSystemConfig instance with env overrides applied.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_DAS_NDVI_CHANGE_THRESHOLD"] = "-0.20"
            >>> cfg = DeforestationAlertSystemConfig.from_env()
            >>> assert cfg.ndvi_change_threshold == Decimal("-0.20")
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

        # Satellite source settings
        if val := os.getenv(f"{_ENV_PREFIX}SENTINEL2_ENABLED"):
            kwargs["sentinel2_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}LANDSAT_ENABLED"):
            kwargs["landsat_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}GLAD_ENABLED"):
            kwargs["glad_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}HANSEN_GFC_ENABLED"):
            kwargs["hansen_gfc_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}RADD_ENABLED"):
            kwargs["radd_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}SENTINEL2_RESOLUTION_M"):
            kwargs["sentinel2_resolution_m"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}LANDSAT_RESOLUTION_M"):
            kwargs["landsat_resolution_m"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SENTINEL2_REVISIT_DAYS"):
            kwargs["sentinel2_revisit_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}LANDSAT_REVISIT_DAYS"):
            kwargs["landsat_revisit_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CLOUD_COVER_MAX_PCT"):
            kwargs["cloud_cover_max_pct"] = int(val)

        # Change detection thresholds
        if val := os.getenv(f"{_ENV_PREFIX}NDVI_CHANGE_THRESHOLD"):
            kwargs["ndvi_change_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}EVI_CHANGE_THRESHOLD"):
            kwargs["evi_change_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MIN_CLEARING_AREA_HA"):
            kwargs["min_clearing_area_ha"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_CLOUD_COVER_PCT"):
            kwargs["max_cloud_cover_pct"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TEMPORAL_WINDOW_DAYS"):
            kwargs["temporal_window_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CONFIDENCE_THRESHOLD"):
            kwargs["confidence_threshold"] = Decimal(val)

        # Alert generation settings
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_BATCH_SIZE"):
            kwargs["alert_batch_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}REAL_TIME_ENABLED"):
            kwargs["real_time_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}DEDUP_WINDOW_HOURS"):
            kwargs["dedup_window_hours"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_ALERTS_PER_DAY"):
            kwargs["max_alerts_per_day"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_RETENTION_DAYS"):
            kwargs["alert_retention_days"] = int(val)

        # Severity classification thresholds
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_AREA_THRESHOLD_HA"):
            kwargs["critical_area_threshold_ha"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}HIGH_AREA_THRESHOLD_HA"):
            kwargs["high_area_threshold_ha"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MEDIUM_AREA_THRESHOLD_HA"):
            kwargs["medium_area_threshold_ha"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}PROXIMITY_CRITICAL_KM"):
            kwargs["proximity_critical_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}PROXIMITY_HIGH_KM"):
            kwargs["proximity_high_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}PROXIMITY_MEDIUM_KM"):
            kwargs["proximity_medium_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}PROTECTED_AREA_MULTIPLIER"):
            kwargs["protected_area_multiplier"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}POST_CUTOFF_MULTIPLIER"):
            kwargs["post_cutoff_multiplier"] = Decimal(val)

        # Severity weights
        if val := os.getenv(f"{_ENV_PREFIX}AREA_WEIGHT"):
            kwargs["area_weight"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_WEIGHT"):
            kwargs["rate_weight"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}PROXIMITY_WEIGHT"):
            kwargs["proximity_weight"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}PROTECTED_WEIGHT"):
            kwargs["protected_weight"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}TIMING_WEIGHT"):
            kwargs["timing_weight"] = Decimal(val)

        # Spatial buffer settings
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_BUFFER_RADIUS_KM"):
            kwargs["default_buffer_radius_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MIN_BUFFER_KM"):
            kwargs["min_buffer_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_BUFFER_KM"):
            kwargs["max_buffer_km"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}BUFFER_RESOLUTION_POINTS"):
            kwargs["buffer_resolution_points"] = int(val)

        # EUDR cutoff date settings
        if val := os.getenv(f"{_ENV_PREFIX}CUTOFF_DATE"):
            kwargs["cutoff_date"] = val
        if val := os.getenv(f"{_ENV_PREFIX}PRE_CUTOFF_GRACE_DAYS"):
            kwargs["pre_cutoff_grace_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TEMPORAL_EVIDENCE_SOURCES_MIN"):
            kwargs["temporal_evidence_sources_min"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CUTOFF_CONFIDENCE_THRESHOLD"):
            kwargs["cutoff_confidence_threshold"] = Decimal(val)

        # Historical baseline settings
        if val := os.getenv(f"{_ENV_PREFIX}BASELINE_START_YEAR"):
            kwargs["baseline_start_year"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BASELINE_END_YEAR"):
            kwargs["baseline_end_year"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MIN_BASELINE_SAMPLES"):
            kwargs["min_baseline_samples"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CANOPY_COVER_THRESHOLD_PCT"):
            kwargs["canopy_cover_threshold_pct"] = Decimal(val)

        # Workflow settings
        if val := os.getenv(f"{_ENV_PREFIX}AUTO_TRIAGE_ENABLED"):
            kwargs["auto_triage_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}SLA_TRIAGE_HOURS"):
            kwargs["sla_triage_hours"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SLA_INVESTIGATION_HOURS"):
            kwargs["sla_investigation_hours"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SLA_RESOLUTION_HOURS"):
            kwargs["sla_resolution_hours"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ESCALATION_ENABLED"):
            kwargs["escalation_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}MAX_ESCALATION_LEVELS"):
            kwargs["max_escalation_levels"] = int(val)

        # Compliance settings
        if val := os.getenv(f"{_ENV_PREFIX}IMPACT_ASSESSMENT_AUTO"):
            kwargs["impact_assessment_auto"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}MARKET_RESTRICTION_THRESHOLD"):
            kwargs["market_restriction_threshold"] = val.upper()
        if val := os.getenv(f"{_ENV_PREFIX}REMEDIATION_PLAN_REQUIRED"):
            kwargs["remediation_plan_required"] = val.lower() in ("true", "1", "yes")

        # Reporting
        if val := os.getenv(f"{_ENV_PREFIX}OUTPUT_FORMATS"):
            kwargs["output_formats"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_LANGUAGE"):
            kwargs["default_language"] = val
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_LANGUAGES"):
            kwargs["supported_languages"] = [x.strip() for x in val.split(",")]

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
            redact_secrets: If True, redact sensitive fields like database_url.

        Returns:
            Dictionary representation of configuration.

        Example:
            >>> cfg = DeforestationAlertSystemConfig()
            >>> d = cfg.to_dict(redact_secrets=True)
            >>> assert "database_url" in d
            >>> assert "REDACTED" in d["database_url"]
        """
        data: Dict[str, Any] = {
            # Database
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "pool_timeout_s": self.pool_timeout_s,
            "pool_recycle_s": self.pool_recycle_s,
            # Redis
            "redis_url": self.redis_url,
            "redis_ttl_s": self.redis_ttl_s,
            "redis_key_prefix": self.redis_key_prefix,
            # Logging
            "log_level": self.log_level,
            # Satellite sources
            "sentinel2_enabled": self.sentinel2_enabled,
            "landsat_enabled": self.landsat_enabled,
            "glad_enabled": self.glad_enabled,
            "hansen_gfc_enabled": self.hansen_gfc_enabled,
            "radd_enabled": self.radd_enabled,
            "sentinel2_resolution_m": self.sentinel2_resolution_m,
            "landsat_resolution_m": self.landsat_resolution_m,
            "sentinel2_revisit_days": self.sentinel2_revisit_days,
            "landsat_revisit_days": self.landsat_revisit_days,
            "cloud_cover_max_pct": self.cloud_cover_max_pct,
            # Change detection
            "ndvi_change_threshold": str(self.ndvi_change_threshold),
            "evi_change_threshold": str(self.evi_change_threshold),
            "min_clearing_area_ha": str(self.min_clearing_area_ha),
            "max_cloud_cover_pct": self.max_cloud_cover_pct,
            "temporal_window_days": self.temporal_window_days,
            "confidence_threshold": str(self.confidence_threshold),
            # Alert generation
            "alert_batch_size": self.alert_batch_size,
            "real_time_enabled": self.real_time_enabled,
            "dedup_window_hours": self.dedup_window_hours,
            "max_alerts_per_day": self.max_alerts_per_day,
            "alert_retention_days": self.alert_retention_days,
            # Severity thresholds
            "critical_area_threshold_ha": str(self.critical_area_threshold_ha),
            "high_area_threshold_ha": str(self.high_area_threshold_ha),
            "medium_area_threshold_ha": str(self.medium_area_threshold_ha),
            "proximity_critical_km": str(self.proximity_critical_km),
            "proximity_high_km": str(self.proximity_high_km),
            "proximity_medium_km": str(self.proximity_medium_km),
            "protected_area_multiplier": str(self.protected_area_multiplier),
            "post_cutoff_multiplier": str(self.post_cutoff_multiplier),
            # Severity weights
            "area_weight": str(self.area_weight),
            "rate_weight": str(self.rate_weight),
            "proximity_weight": str(self.proximity_weight),
            "protected_weight": str(self.protected_weight),
            "timing_weight": str(self.timing_weight),
            # Spatial buffer
            "default_buffer_radius_km": str(self.default_buffer_radius_km),
            "min_buffer_km": str(self.min_buffer_km),
            "max_buffer_km": str(self.max_buffer_km),
            "buffer_resolution_points": self.buffer_resolution_points,
            # Cutoff date
            "cutoff_date": self.cutoff_date,
            "pre_cutoff_grace_days": self.pre_cutoff_grace_days,
            "temporal_evidence_sources_min": self.temporal_evidence_sources_min,
            "cutoff_confidence_threshold": str(self.cutoff_confidence_threshold),
            # Baseline
            "baseline_start_year": self.baseline_start_year,
            "baseline_end_year": self.baseline_end_year,
            "min_baseline_samples": self.min_baseline_samples,
            "canopy_cover_threshold_pct": str(self.canopy_cover_threshold_pct),
            # Workflow
            "auto_triage_enabled": self.auto_triage_enabled,
            "sla_triage_hours": self.sla_triage_hours,
            "sla_investigation_hours": self.sla_investigation_hours,
            "sla_resolution_hours": self.sla_resolution_hours,
            "escalation_enabled": self.escalation_enabled,
            "max_escalation_levels": self.max_escalation_levels,
            # Compliance
            "impact_assessment_auto": self.impact_assessment_auto,
            "market_restriction_threshold": self.market_restriction_threshold,
            "remediation_plan_required": self.remediation_plan_required,
            # Reporting
            "output_formats": self.output_formats,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            # Batch
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Retention
            "retention_years": self.retention_years,
            # Provenance
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            "chain_algorithm": self.chain_algorithm,
            # Metrics
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            # Rate limiting
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
_global_config: Optional[DeforestationAlertSystemConfig] = None


def get_config() -> DeforestationAlertSystemConfig:
    """Get the global DeforestationAlertSystemConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance. Uses double-checked locking
    to minimize contention after initialization.

    Returns:
        DeforestationAlertSystemConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> assert cfg.ndvi_change_threshold == Decimal("-0.15")
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2  # Same instance
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = DeforestationAlertSystemConfig.from_env()
    return _global_config


def set_config(config: DeforestationAlertSystemConfig) -> None:
    """Set the global DeforestationAlertSystemConfig singleton instance.

    Used for testing and programmatic configuration override.

    Args:
        config: DeforestationAlertSystemConfig instance to set as global.

    Example:
        >>> from greenlang.agents.eudr.deforestation_alert_system.config import (
        ...     set_config, DeforestationAlertSystemConfig,
        ... )
        >>> test_cfg = DeforestationAlertSystemConfig(
        ...     ndvi_change_threshold=Decimal("-0.20"),
        ... )
        >>> set_config(test_cfg)
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global DeforestationAlertSystemConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_config()
        >>> # Next get_config() call will re-initialize from environment
    """
    global _global_config
    with _config_lock:
        _global_config = None
