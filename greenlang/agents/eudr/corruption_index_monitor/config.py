# -*- coding: utf-8 -*-
"""
Corruption Index Monitor Configuration - AGENT-EUDR-019

Centralized configuration for the Corruption Index Monitor Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- CPI score ranges and risk thresholds: 0-100 scale (0=most corrupt,
  100=cleanest), high_risk_threshold=30, moderate_threshold=50, low_risk
  threshold=70, with configurable country classification boundaries
- WGI 6 dimensions: voice_accountability, political_stability,
  government_effectiveness, regulatory_quality, rule_of_law,
  control_of_corruption on -2.5 to +2.5 scale with configurable risk
  threshold at -0.5 and composite weighting across dimensions
- Bribery risk sectors: forestry (0.25), customs (0.20), agriculture
  (0.20), mining (0.15), extraction (0.10), judiciary (0.10) with
  sector-specific weights summing to 1.0 and configurable risk multipliers
- Trend analysis parameters: min_years=5 (minimum data points for valid
  trend), trajectory_window=10 (years for trajectory computation),
  prediction_horizon=3 (years forward), min_r_squared=0.3 (goodness of
  fit threshold for reliable trend), trend_reversal_sensitivity=0.15
- Correlation analysis: min_data_points=10 (for statistical validity),
  significance_level=0.05 (p-value threshold), min_correlation=0.3
  (minimum Pearson r for reportable correlation)
- Alert thresholds: cpi_change_alert=5 (points change triggering alert),
  wgi_change_alert=0.3 (units change triggering alert),
  trend_reversal_alert=True (enable trend reversal detection),
  alert_cooldown_hours=24, max_alerts_per_country_per_day=10
- EUDR Article 29 country classification thresholds: low_risk (CPI>=60
  AND WGI>=0.5), high_risk (CPI<=30 OR WGI<=-0.5), standard_risk
  (everything in between) with configurable override capability
- Due diligence level mapping: simplified (low_risk countries),
  standard (standard_risk countries), enhanced (high_risk countries)
  with configurable cost ranges and audit frequency recommendations
- Institutional quality dimension weights: judicial_independence=0.30,
  regulatory_enforcement=0.25, forest_governance=0.25,
  law_enforcement_capacity=0.20
- Batch processing, provenance, metrics, rate limiting settings

All settings can be overridden via environment variables with the
``GL_EUDR_CIM_`` prefix (e.g. ``GL_EUDR_CIM_DATABASE_URL``,
``GL_EUDR_CIM_CPI_HIGH_RISK_THRESHOLD``).

Environment Variable Reference (GL_EUDR_CIM_ prefix):
    GL_EUDR_CIM_DATABASE_URL                    - PostgreSQL connection URL
    GL_EUDR_CIM_REDIS_URL                       - Redis connection URL
    GL_EUDR_CIM_LOG_LEVEL                       - Logging level
    GL_EUDR_CIM_POOL_SIZE                       - Database pool size
    GL_EUDR_CIM_POOL_TIMEOUT_S                  - Pool timeout seconds
    GL_EUDR_CIM_POOL_RECYCLE_S                  - Pool recycle seconds
    GL_EUDR_CIM_CPI_HIGH_RISK_THRESHOLD         - CPI high risk threshold (0-100)
    GL_EUDR_CIM_CPI_MODERATE_THRESHOLD          - CPI moderate risk threshold (0-100)
    GL_EUDR_CIM_CPI_LOW_RISK_THRESHOLD          - CPI low risk threshold (0-100)
    GL_EUDR_CIM_WGI_RISK_THRESHOLD              - WGI risk threshold (-2.5 to +2.5)
    GL_EUDR_CIM_WGI_LOW_RISK_THRESHOLD          - WGI low risk threshold (-2.5 to +2.5)
    GL_EUDR_CIM_WGI_VA_WEIGHT                   - WGI voice_accountability weight
    GL_EUDR_CIM_WGI_PS_WEIGHT                   - WGI political_stability weight
    GL_EUDR_CIM_WGI_GE_WEIGHT                   - WGI government_effectiveness weight
    GL_EUDR_CIM_WGI_RQ_WEIGHT                   - WGI regulatory_quality weight
    GL_EUDR_CIM_WGI_RL_WEIGHT                   - WGI rule_of_law weight
    GL_EUDR_CIM_WGI_CC_WEIGHT                   - WGI control_of_corruption weight
    GL_EUDR_CIM_BRIBERY_FORESTRY_WEIGHT         - Bribery forestry sector weight
    GL_EUDR_CIM_BRIBERY_CUSTOMS_WEIGHT          - Bribery customs sector weight
    GL_EUDR_CIM_BRIBERY_AGRICULTURE_WEIGHT      - Bribery agriculture sector weight
    GL_EUDR_CIM_BRIBERY_MINING_WEIGHT           - Bribery mining sector weight
    GL_EUDR_CIM_BRIBERY_EXTRACTION_WEIGHT       - Bribery extraction sector weight
    GL_EUDR_CIM_BRIBERY_JUDICIARY_WEIGHT        - Bribery judiciary sector weight
    GL_EUDR_CIM_BRIBERY_RISK_MULTIPLIER         - Global bribery risk multiplier
    GL_EUDR_CIM_IQ_JUDICIAL_WEIGHT              - Institutional quality judicial independence weight
    GL_EUDR_CIM_IQ_REGULATORY_WEIGHT            - Institutional quality regulatory enforcement weight
    GL_EUDR_CIM_IQ_FOREST_GOV_WEIGHT            - Institutional quality forest governance weight
    GL_EUDR_CIM_IQ_LAW_ENFORCEMENT_WEIGHT       - Institutional quality law enforcement weight
    GL_EUDR_CIM_TREND_MIN_YEARS                 - Minimum years for valid trend analysis
    GL_EUDR_CIM_TREND_TRAJECTORY_WINDOW         - Trajectory window in years
    GL_EUDR_CIM_TREND_PREDICTION_HORIZON        - Prediction horizon in years
    GL_EUDR_CIM_TREND_MIN_R_SQUARED             - Minimum R-squared for reliable trend
    GL_EUDR_CIM_TREND_REVERSAL_SENSITIVITY      - Trend reversal detection sensitivity
    GL_EUDR_CIM_CORRELATION_MIN_DATA_POINTS     - Minimum data points for correlation
    GL_EUDR_CIM_CORRELATION_SIGNIFICANCE_LEVEL  - P-value significance level
    GL_EUDR_CIM_CORRELATION_MIN_COEFFICIENT     - Minimum correlation coefficient to report
    GL_EUDR_CIM_ALERT_CPI_CHANGE_THRESHOLD      - CPI change threshold for alert (points)
    GL_EUDR_CIM_ALERT_WGI_CHANGE_THRESHOLD      - WGI change threshold for alert (units)
    GL_EUDR_CIM_ALERT_TREND_REVERSAL            - Enable trend reversal alerts (bool)
    GL_EUDR_CIM_ALERT_COOLDOWN_HOURS            - Alert cooldown period in hours
    GL_EUDR_CIM_ALERT_MAX_PER_COUNTRY_PER_DAY   - Max alerts per country per day
    GL_EUDR_CIM_ART29_LOW_RISK_CPI              - Article 29 low risk CPI threshold
    GL_EUDR_CIM_ART29_LOW_RISK_WGI              - Article 29 low risk WGI threshold
    GL_EUDR_CIM_ART29_HIGH_RISK_CPI             - Article 29 high risk CPI threshold
    GL_EUDR_CIM_ART29_HIGH_RISK_WGI             - Article 29 high risk WGI threshold
    GL_EUDR_CIM_DD_SIMPLIFIED_COST_EUR          - Simplified DD cost estimate (EUR)
    GL_EUDR_CIM_DD_STANDARD_COST_EUR            - Standard DD cost estimate (EUR)
    GL_EUDR_CIM_DD_ENHANCED_COST_EUR            - Enhanced DD cost estimate (EUR)
    GL_EUDR_CIM_DD_SIMPLIFIED_AUDIT_MONTHS      - Simplified DD audit interval (months)
    GL_EUDR_CIM_DD_STANDARD_AUDIT_MONTHS        - Standard DD audit interval (months)
    GL_EUDR_CIM_DD_ENHANCED_AUDIT_MONTHS        - Enhanced DD audit interval (months)
    GL_EUDR_CIM_REDIS_TTL_S                     - Redis cache TTL seconds
    GL_EUDR_CIM_REDIS_KEY_PREFIX                - Redis key prefix
    GL_EUDR_CIM_BATCH_MAX_SIZE                  - Batch processing max size
    GL_EUDR_CIM_BATCH_CONCURRENCY               - Batch concurrency workers
    GL_EUDR_CIM_BATCH_TIMEOUT_S                 - Batch timeout seconds
    GL_EUDR_CIM_RETENTION_YEARS                 - Data retention years
    GL_EUDR_CIM_ENABLE_PROVENANCE               - Enable provenance tracking
    GL_EUDR_CIM_GENESIS_HASH                    - Genesis hash anchor
    GL_EUDR_CIM_CHAIN_ALGORITHM                 - Provenance chain hash algorithm
    GL_EUDR_CIM_ENABLE_METRICS                  - Enable Prometheus metrics
    GL_EUDR_CIM_METRICS_PREFIX                  - Prometheus metrics prefix
    GL_EUDR_CIM_RATE_LIMIT_ANONYMOUS            - Rate limit anonymous tier
    GL_EUDR_CIM_RATE_LIMIT_BASIC                - Rate limit basic tier
    GL_EUDR_CIM_RATE_LIMIT_STANDARD             - Rate limit standard tier
    GL_EUDR_CIM_RATE_LIMIT_PREMIUM              - Rate limit premium tier
    GL_EUDR_CIM_RATE_LIMIT_ADMIN                - Rate limit admin tier

Example:
    >>> from greenlang.agents.eudr.corruption_index_monitor.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.cpi_high_risk_threshold, cfg.wgi_risk_threshold)
    30 -0.5

    >>> # Override for testing
    >>> from greenlang.agents.eudr.corruption_index_monitor.config import (
    ...     set_config, reset_config, CorruptionIndexMonitorConfig,
    ... )
    >>> set_config(CorruptionIndexMonitorConfig(cpi_high_risk_threshold=25))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
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

_ENV_PREFIX = "GL_EUDR_CIM_"

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
# Valid DD levels
# ---------------------------------------------------------------------------

_VALID_DD_LEVELS = frozenset({"simplified", "standard", "enhanced"})

# ---------------------------------------------------------------------------
# Valid output formats
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "excel", "csv"})

# ---------------------------------------------------------------------------
# Valid report languages
# ---------------------------------------------------------------------------

_VALID_REPORT_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})

# ---------------------------------------------------------------------------
# WGI dimension identifiers
# ---------------------------------------------------------------------------

_WGI_DIMENSIONS = frozenset({
    "voice_accountability",
    "political_stability",
    "government_effectiveness",
    "regulatory_quality",
    "rule_of_law",
    "control_of_corruption",
})

# ---------------------------------------------------------------------------
# Bribery risk sectors
# ---------------------------------------------------------------------------

_BRIBERY_SECTORS = frozenset({
    "forestry",
    "customs",
    "agriculture",
    "mining",
    "extraction",
    "judiciary",
})

# ---------------------------------------------------------------------------
# Default WGI dimension weights (sum = 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_WGI_WEIGHTS: Dict[str, float] = {
    "voice_accountability": 0.10,
    "political_stability": 0.10,
    "government_effectiveness": 0.20,
    "regulatory_quality": 0.15,
    "rule_of_law": 0.20,
    "control_of_corruption": 0.25,
}

# ---------------------------------------------------------------------------
# Default bribery sector weights (sum = 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_BRIBERY_SECTOR_WEIGHTS: Dict[str, float] = {
    "forestry": 0.25,
    "customs": 0.20,
    "agriculture": 0.20,
    "mining": 0.15,
    "extraction": 0.10,
    "judiciary": 0.10,
}

# ---------------------------------------------------------------------------
# Default institutional quality dimension weights (sum = 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_IQ_WEIGHTS: Dict[str, float] = {
    "judicial_independence": 0.30,
    "regulatory_enforcement": 0.25,
    "forest_governance": 0.25,
    "law_enforcement_capacity": 0.20,
}

# ---------------------------------------------------------------------------
# Default risk level thresholds for CPI
# These define boundary values on the 0-100 CPI scale:
#   <=30 = high risk, 31-50 = moderate risk, 51-70 = low-moderate, >70 = low
# ---------------------------------------------------------------------------

_DEFAULT_CPI_HIGH_RISK = 30
_DEFAULT_CPI_MODERATE = 50
_DEFAULT_CPI_LOW_RISK = 70

# ---------------------------------------------------------------------------
# Default WGI risk thresholds on the -2.5 to +2.5 scale
# ---------------------------------------------------------------------------

_DEFAULT_WGI_RISK_THRESHOLD = -0.5
_DEFAULT_WGI_LOW_RISK_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Default EUDR Article 29 classification thresholds
# ---------------------------------------------------------------------------

_DEFAULT_ART29_LOW_RISK_CPI = 60
_DEFAULT_ART29_LOW_RISK_WGI = 0.5
_DEFAULT_ART29_HIGH_RISK_CPI = 30
_DEFAULT_ART29_HIGH_RISK_WGI = -0.5

# ---------------------------------------------------------------------------
# Default DD cost estimates (EUR) and audit intervals (months)
# ---------------------------------------------------------------------------

_DEFAULT_DD_COSTS: Dict[str, int] = {
    "simplified": 500,
    "standard": 3000,
    "enhanced": 15000,
}

_DEFAULT_DD_AUDIT_MONTHS: Dict[str, int] = {
    "simplified": 24,
    "standard": 12,
    "enhanced": 6,
}

# ---------------------------------------------------------------------------
# Supported regions for CPI analysis
# ---------------------------------------------------------------------------

_SUPPORTED_REGIONS: List[str] = [
    "africa",
    "americas",
    "asia_pacific",
    "eastern_europe_central_asia",
    "eu_western_europe",
    "middle_east_north_africa",
    "sub_saharan_africa",
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class CorruptionIndexMonitorConfig:
    """Configuration for the Corruption Index Monitor Agent (AGENT-EUDR-019).

    This dataclass encapsulates all configuration settings for CPI monitoring,
    WGI analysis, bribery risk assessment, institutional quality scoring,
    trend analysis, deforestation-corruption correlation, alert generation,
    and compliance impact assessment. All settings have sensible defaults
    aligned with EUDR requirements and can be overridden via environment
    variables with the GL_EUDR_CIM_ prefix.

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

        # CPI thresholds (0-100 scale, 0=most corrupt, 100=cleanest)
        cpi_high_risk_threshold: CPI score at or below which = high risk
        cpi_moderate_threshold: CPI score at or below which = moderate risk
        cpi_low_risk_threshold: CPI score above which = low risk

        # WGI thresholds (-2.5 to +2.5 scale)
        wgi_risk_threshold: WGI composite at or below which = high risk
        wgi_low_risk_threshold: WGI composite above which = low risk
        wgi_dimension_weights: Per-dimension weights for composite score

        # Bribery risk settings
        bribery_sector_weights: Per-sector weights for bribery risk scoring
        bribery_risk_multiplier: Global multiplier for bribery risk scores

        # Institutional quality weights
        iq_judicial_weight: Judicial independence weight
        iq_regulatory_weight: Regulatory enforcement weight
        iq_forest_gov_weight: Forest governance weight
        iq_law_enforcement_weight: Law enforcement capacity weight

        # Trend analysis settings
        trend_min_years: Minimum years of data for valid trend
        trend_trajectory_window: Years window for trajectory computation
        trend_prediction_horizon: Years forward for prediction
        trend_min_r_squared: Minimum R-squared for reliable trend
        trend_reversal_sensitivity: Sensitivity for reversal detection

        # Correlation analysis settings
        correlation_min_data_points: Minimum data points for correlation
        correlation_significance_level: P-value significance level
        correlation_min_coefficient: Minimum Pearson r to report

        # Alert settings
        alert_cpi_change_threshold: CPI change triggering alert (points)
        alert_wgi_change_threshold: WGI change triggering alert (units)
        alert_trend_reversal: Enable trend reversal alerts
        alert_cooldown_hours: Cooldown between duplicate alerts (hours)
        alert_max_per_country_per_day: Max alerts per country per day

        # EUDR Article 29 classification settings
        art29_low_risk_cpi: CPI threshold for low risk classification
        art29_low_risk_wgi: WGI threshold for low risk classification
        art29_high_risk_cpi: CPI threshold for high risk classification
        art29_high_risk_wgi: WGI threshold for high risk classification

        # Due diligence settings
        dd_simplified_cost_eur: Simplified DD cost estimate (EUR)
        dd_standard_cost_eur: Standard DD cost estimate (EUR)
        dd_enhanced_cost_eur: Enhanced DD cost estimate (EUR)
        dd_simplified_audit_months: Simplified DD audit interval (months)
        dd_standard_audit_months: Standard DD audit interval (months)
        dd_enhanced_audit_months: Enhanced DD audit interval (months)

        # Reporting
        output_formats: Report output formats
        default_language: Default report language
        supported_languages: Supported report languages

        # Batch processing
        batch_max_size: Batch processing max size
        batch_concurrency: Batch concurrency workers
        batch_timeout_s: Batch timeout seconds

        # Data retention
        retention_years: Data retention years

        # Provenance
        enable_provenance: Enable provenance tracking
        genesis_hash: Genesis hash anchor
        chain_algorithm: Provenance chain hash algorithm

        # Metrics
        enable_metrics: Enable Prometheus metrics
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
    redis_key_prefix: str = "gl:eudr:cim:"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # CPI thresholds (0-100 scale, 0=most corrupt, 100=cleanest)
    # -----------------------------------------------------------------------
    cpi_high_risk_threshold: int = _DEFAULT_CPI_HIGH_RISK
    cpi_moderate_threshold: int = _DEFAULT_CPI_MODERATE
    cpi_low_risk_threshold: int = _DEFAULT_CPI_LOW_RISK

    # -----------------------------------------------------------------------
    # WGI thresholds (-2.5 to +2.5 scale)
    # -----------------------------------------------------------------------
    wgi_risk_threshold: float = _DEFAULT_WGI_RISK_THRESHOLD
    wgi_low_risk_threshold: float = _DEFAULT_WGI_LOW_RISK_THRESHOLD
    wgi_dimension_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_WGI_WEIGHTS)
    )

    # -----------------------------------------------------------------------
    # Bribery risk settings
    # -----------------------------------------------------------------------
    bribery_sector_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_BRIBERY_SECTOR_WEIGHTS)
    )
    bribery_risk_multiplier: float = 1.0

    # -----------------------------------------------------------------------
    # Institutional quality dimension weights
    # -----------------------------------------------------------------------
    iq_judicial_weight: float = 0.30
    iq_regulatory_weight: float = 0.25
    iq_forest_gov_weight: float = 0.25
    iq_law_enforcement_weight: float = 0.20

    # -----------------------------------------------------------------------
    # Trend analysis settings
    # -----------------------------------------------------------------------
    trend_min_years: int = 5
    trend_trajectory_window: int = 10
    trend_prediction_horizon: int = 3
    trend_min_r_squared: float = 0.3
    trend_reversal_sensitivity: float = 0.15

    # -----------------------------------------------------------------------
    # Correlation analysis settings
    # -----------------------------------------------------------------------
    correlation_min_data_points: int = 10
    correlation_significance_level: float = 0.05
    correlation_min_coefficient: float = 0.3

    # -----------------------------------------------------------------------
    # Alert settings
    # -----------------------------------------------------------------------
    alert_cpi_change_threshold: int = 5
    alert_wgi_change_threshold: float = 0.3
    alert_trend_reversal: bool = True
    alert_cooldown_hours: int = 24
    alert_max_per_country_per_day: int = 10

    # -----------------------------------------------------------------------
    # EUDR Article 29 classification thresholds
    # -----------------------------------------------------------------------
    art29_low_risk_cpi: int = _DEFAULT_ART29_LOW_RISK_CPI
    art29_low_risk_wgi: float = _DEFAULT_ART29_LOW_RISK_WGI
    art29_high_risk_cpi: int = _DEFAULT_ART29_HIGH_RISK_CPI
    art29_high_risk_wgi: float = _DEFAULT_ART29_HIGH_RISK_WGI

    # -----------------------------------------------------------------------
    # Due diligence settings
    # -----------------------------------------------------------------------
    dd_simplified_cost_eur: int = _DEFAULT_DD_COSTS["simplified"]
    dd_standard_cost_eur: int = _DEFAULT_DD_COSTS["standard"]
    dd_enhanced_cost_eur: int = _DEFAULT_DD_COSTS["enhanced"]
    dd_simplified_audit_months: int = _DEFAULT_DD_AUDIT_MONTHS["simplified"]
    dd_standard_audit_months: int = _DEFAULT_DD_AUDIT_MONTHS["standard"]
    dd_enhanced_audit_months: int = _DEFAULT_DD_AUDIT_MONTHS["enhanced"]

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "excel"]
    )
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )

    # -----------------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------------
    batch_max_size: int = 500
    batch_concurrency: int = 4
    batch_timeout_s: int = 300

    # -----------------------------------------------------------------------
    # Data retention
    # -----------------------------------------------------------------------
    retention_years: int = 5

    # -----------------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-CIM-019-CORRUPTION-INDEX-MONITOR-GENESIS"
    chain_algorithm: str = "sha256"

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_cim_"

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
        self._validate_cpi_thresholds()
        self._validate_wgi_thresholds()
        self._validate_wgi_weights()
        self._validate_bribery_weights()
        self._validate_iq_weights()
        self._validate_trend_settings()
        self._validate_correlation_settings()
        self._validate_alert_settings()
        self._validate_art29_thresholds()
        self._validate_dd_settings()
        self._validate_positive_integers()
        self._validate_output_formats()
        self._validate_languages()

        logger.info(
            f"CorruptionIndexMonitorConfig initialized: "
            f"cpi_high_risk={self.cpi_high_risk_threshold}, "
            f"wgi_risk={self.wgi_risk_threshold}, "
            f"trend_min_years={self.trend_min_years}, "
            f"alert_cpi_change={self.alert_cpi_change_threshold}"
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

    def _validate_cpi_thresholds(self) -> None:
        """Validate CPI thresholds are in ascending order within 0-100."""
        if not (
            0 <= self.cpi_high_risk_threshold
            < self.cpi_moderate_threshold
            < self.cpi_low_risk_threshold
            <= 100
        ):
            raise ValueError(
                "CPI thresholds must be in ascending order: "
                "0 <= high_risk < moderate < low_risk <= 100. "
                f"Got high_risk={self.cpi_high_risk_threshold}, "
                f"moderate={self.cpi_moderate_threshold}, "
                f"low_risk={self.cpi_low_risk_threshold}"
            )

    def _validate_wgi_thresholds(self) -> None:
        """Validate WGI thresholds are within -2.5 to +2.5 and ordered."""
        if not -2.5 <= self.wgi_risk_threshold <= 2.5:
            raise ValueError(
                f"wgi_risk_threshold must be between -2.5 and +2.5, "
                f"got {self.wgi_risk_threshold}"
            )
        if not -2.5 <= self.wgi_low_risk_threshold <= 2.5:
            raise ValueError(
                f"wgi_low_risk_threshold must be between -2.5 and +2.5, "
                f"got {self.wgi_low_risk_threshold}"
            )
        if self.wgi_risk_threshold >= self.wgi_low_risk_threshold:
            raise ValueError(
                "wgi_risk_threshold must be less than wgi_low_risk_threshold: "
                f"got risk={self.wgi_risk_threshold}, "
                f"low_risk={self.wgi_low_risk_threshold}"
            )

    def _validate_wgi_weights(self) -> None:
        """Validate WGI dimension weights sum to 1.0 and all are positive."""
        for dim in _WGI_DIMENSIONS:
            if dim not in self.wgi_dimension_weights:
                raise ValueError(
                    f"Missing WGI dimension weight: {dim}. "
                    f"Required dimensions: {_WGI_DIMENSIONS}"
                )
        for dim, weight in self.wgi_dimension_weights.items():
            if dim not in _WGI_DIMENSIONS:
                raise ValueError(
                    f"Invalid WGI dimension: {dim}. "
                    f"Must be one of {_WGI_DIMENSIONS}"
                )
            if weight <= 0.0 or weight > 1.0:
                raise ValueError(
                    f"WGI weight for {dim} must be between 0.0 (exclusive) "
                    f"and 1.0 (inclusive), got {weight}"
                )
        total = sum(self.wgi_dimension_weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"WGI dimension weights must sum to 1.0, got {total:.4f}"
            )

    def _validate_bribery_weights(self) -> None:
        """Validate bribery sector weights sum to 1.0 and all are positive."""
        for sector in _BRIBERY_SECTORS:
            if sector not in self.bribery_sector_weights:
                raise ValueError(
                    f"Missing bribery sector weight: {sector}. "
                    f"Required sectors: {_BRIBERY_SECTORS}"
                )
        for sector, weight in self.bribery_sector_weights.items():
            if sector not in _BRIBERY_SECTORS:
                raise ValueError(
                    f"Invalid bribery sector: {sector}. "
                    f"Must be one of {_BRIBERY_SECTORS}"
                )
            if weight < 0.0 or weight > 1.0:
                raise ValueError(
                    f"Bribery weight for {sector} must be between 0.0 and "
                    f"1.0, got {weight}"
                )
        total = sum(self.bribery_sector_weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Bribery sector weights must sum to 1.0, got {total:.4f}"
            )
        if self.bribery_risk_multiplier <= 0.0:
            raise ValueError(
                f"bribery_risk_multiplier must be > 0, "
                f"got {self.bribery_risk_multiplier}"
            )

    def _validate_iq_weights(self) -> None:
        """Validate institutional quality weights sum to 1.0."""
        weights = [
            self.iq_judicial_weight,
            self.iq_regulatory_weight,
            self.iq_forest_gov_weight,
            self.iq_law_enforcement_weight,
        ]
        for w in weights:
            if w < 0.0 or w > 1.0:
                raise ValueError(
                    f"Institutional quality weights must be between 0.0 and "
                    f"1.0, got {w}"
                )
        total = sum(weights)
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Institutional quality weights must sum to 1.0, "
                f"got {total:.4f}"
            )

    def _validate_trend_settings(self) -> None:
        """Validate trend analysis settings are within acceptable ranges."""
        if self.trend_min_years < 2:
            raise ValueError(
                f"trend_min_years must be >= 2, got {self.trend_min_years}"
            )
        if self.trend_trajectory_window < self.trend_min_years:
            raise ValueError(
                f"trend_trajectory_window ({self.trend_trajectory_window}) "
                f"must be >= trend_min_years ({self.trend_min_years})"
            )
        if self.trend_prediction_horizon < 1:
            raise ValueError(
                f"trend_prediction_horizon must be >= 1, "
                f"got {self.trend_prediction_horizon}"
            )
        if not 0.0 <= self.trend_min_r_squared <= 1.0:
            raise ValueError(
                f"trend_min_r_squared must be between 0.0 and 1.0, "
                f"got {self.trend_min_r_squared}"
            )
        if not 0.0 < self.trend_reversal_sensitivity <= 1.0:
            raise ValueError(
                f"trend_reversal_sensitivity must be between 0.0 (exclusive) "
                f"and 1.0, got {self.trend_reversal_sensitivity}"
            )

    def _validate_correlation_settings(self) -> None:
        """Validate correlation analysis settings."""
        if self.correlation_min_data_points < 3:
            raise ValueError(
                f"correlation_min_data_points must be >= 3, "
                f"got {self.correlation_min_data_points}"
            )
        if not 0.0 < self.correlation_significance_level < 1.0:
            raise ValueError(
                f"correlation_significance_level must be between 0.0 and 1.0 "
                f"(exclusive), got {self.correlation_significance_level}"
            )
        if not 0.0 <= self.correlation_min_coefficient <= 1.0:
            raise ValueError(
                f"correlation_min_coefficient must be between 0.0 and 1.0, "
                f"got {self.correlation_min_coefficient}"
            )

    def _validate_alert_settings(self) -> None:
        """Validate alert threshold settings."""
        if self.alert_cpi_change_threshold < 1:
            raise ValueError(
                f"alert_cpi_change_threshold must be >= 1, "
                f"got {self.alert_cpi_change_threshold}"
            )
        if self.alert_wgi_change_threshold <= 0.0:
            raise ValueError(
                f"alert_wgi_change_threshold must be > 0, "
                f"got {self.alert_wgi_change_threshold}"
            )
        if self.alert_cooldown_hours < 0:
            raise ValueError(
                f"alert_cooldown_hours must be >= 0, "
                f"got {self.alert_cooldown_hours}"
            )
        if self.alert_max_per_country_per_day < 1:
            raise ValueError(
                f"alert_max_per_country_per_day must be >= 1, "
                f"got {self.alert_max_per_country_per_day}"
            )

    def _validate_art29_thresholds(self) -> None:
        """Validate EUDR Article 29 classification thresholds."""
        if not 0 <= self.art29_high_risk_cpi <= 100:
            raise ValueError(
                f"art29_high_risk_cpi must be between 0 and 100, "
                f"got {self.art29_high_risk_cpi}"
            )
        if not 0 <= self.art29_low_risk_cpi <= 100:
            raise ValueError(
                f"art29_low_risk_cpi must be between 0 and 100, "
                f"got {self.art29_low_risk_cpi}"
            )
        if self.art29_high_risk_cpi >= self.art29_low_risk_cpi:
            raise ValueError(
                f"art29_high_risk_cpi ({self.art29_high_risk_cpi}) must be "
                f"less than art29_low_risk_cpi ({self.art29_low_risk_cpi})"
            )
        if not -2.5 <= self.art29_high_risk_wgi <= 2.5:
            raise ValueError(
                f"art29_high_risk_wgi must be between -2.5 and +2.5, "
                f"got {self.art29_high_risk_wgi}"
            )
        if not -2.5 <= self.art29_low_risk_wgi <= 2.5:
            raise ValueError(
                f"art29_low_risk_wgi must be between -2.5 and +2.5, "
                f"got {self.art29_low_risk_wgi}"
            )
        if self.art29_high_risk_wgi >= self.art29_low_risk_wgi:
            raise ValueError(
                f"art29_high_risk_wgi ({self.art29_high_risk_wgi}) must be "
                f"less than art29_low_risk_wgi ({self.art29_low_risk_wgi})"
            )

    def _validate_dd_settings(self) -> None:
        """Validate due diligence cost and audit interval settings."""
        if self.dd_simplified_cost_eur < 0:
            raise ValueError(
                f"dd_simplified_cost_eur must be >= 0, "
                f"got {self.dd_simplified_cost_eur}"
            )
        if self.dd_standard_cost_eur < self.dd_simplified_cost_eur:
            raise ValueError(
                f"dd_standard_cost_eur ({self.dd_standard_cost_eur}) must be "
                f">= dd_simplified_cost_eur ({self.dd_simplified_cost_eur})"
            )
        if self.dd_enhanced_cost_eur < self.dd_standard_cost_eur:
            raise ValueError(
                f"dd_enhanced_cost_eur ({self.dd_enhanced_cost_eur}) must be "
                f">= dd_standard_cost_eur ({self.dd_standard_cost_eur})"
            )
        for name, val in [
            ("dd_simplified_audit_months", self.dd_simplified_audit_months),
            ("dd_standard_audit_months", self.dd_standard_audit_months),
            ("dd_enhanced_audit_months", self.dd_enhanced_audit_months),
        ]:
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
        # Enhanced should be more frequent (lower months) than simplified
        if self.dd_enhanced_audit_months > self.dd_simplified_audit_months:
            raise ValueError(
                f"dd_enhanced_audit_months ({self.dd_enhanced_audit_months}) "
                f"should be <= dd_simplified_audit_months "
                f"({self.dd_simplified_audit_months}) "
                f"(enhanced DD requires more frequent audits)"
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
    def from_env(cls) -> "CorruptionIndexMonitorConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_CIM_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            CorruptionIndexMonitorConfig instance with env overrides applied.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_CIM_CPI_HIGH_RISK_THRESHOLD"] = "25"
            >>> cfg = CorruptionIndexMonitorConfig.from_env()
            >>> assert cfg.cpi_high_risk_threshold == 25
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

        # CPI thresholds
        if val := os.getenv(f"{_ENV_PREFIX}CPI_HIGH_RISK_THRESHOLD"):
            kwargs["cpi_high_risk_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CPI_MODERATE_THRESHOLD"):
            kwargs["cpi_moderate_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CPI_LOW_RISK_THRESHOLD"):
            kwargs["cpi_low_risk_threshold"] = int(val)

        # WGI thresholds
        if val := os.getenv(f"{_ENV_PREFIX}WGI_RISK_THRESHOLD"):
            kwargs["wgi_risk_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}WGI_LOW_RISK_THRESHOLD"):
            kwargs["wgi_low_risk_threshold"] = float(val)

        # WGI dimension weights
        wgi_weights = dict(_DEFAULT_WGI_WEIGHTS)
        wgi_env_map = {
            "WGI_VA_WEIGHT": "voice_accountability",
            "WGI_PS_WEIGHT": "political_stability",
            "WGI_GE_WEIGHT": "government_effectiveness",
            "WGI_RQ_WEIGHT": "regulatory_quality",
            "WGI_RL_WEIGHT": "rule_of_law",
            "WGI_CC_WEIGHT": "control_of_corruption",
        }
        wgi_overridden = False
        for env_key, dim_key in wgi_env_map.items():
            if val := os.getenv(f"{_ENV_PREFIX}{env_key}"):
                wgi_weights[dim_key] = float(val)
                wgi_overridden = True
        if wgi_overridden:
            kwargs["wgi_dimension_weights"] = wgi_weights

        # Bribery sector weights
        bribery_weights = dict(_DEFAULT_BRIBERY_SECTOR_WEIGHTS)
        bribery_env_map = {
            "BRIBERY_FORESTRY_WEIGHT": "forestry",
            "BRIBERY_CUSTOMS_WEIGHT": "customs",
            "BRIBERY_AGRICULTURE_WEIGHT": "agriculture",
            "BRIBERY_MINING_WEIGHT": "mining",
            "BRIBERY_EXTRACTION_WEIGHT": "extraction",
            "BRIBERY_JUDICIARY_WEIGHT": "judiciary",
        }
        bribery_overridden = False
        for env_key, sector_key in bribery_env_map.items():
            if val := os.getenv(f"{_ENV_PREFIX}{env_key}"):
                bribery_weights[sector_key] = float(val)
                bribery_overridden = True
        if bribery_overridden:
            kwargs["bribery_sector_weights"] = bribery_weights
        if val := os.getenv(f"{_ENV_PREFIX}BRIBERY_RISK_MULTIPLIER"):
            kwargs["bribery_risk_multiplier"] = float(val)

        # Institutional quality weights
        if val := os.getenv(f"{_ENV_PREFIX}IQ_JUDICIAL_WEIGHT"):
            kwargs["iq_judicial_weight"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}IQ_REGULATORY_WEIGHT"):
            kwargs["iq_regulatory_weight"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}IQ_FOREST_GOV_WEIGHT"):
            kwargs["iq_forest_gov_weight"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}IQ_LAW_ENFORCEMENT_WEIGHT"):
            kwargs["iq_law_enforcement_weight"] = float(val)

        # Trend analysis settings
        if val := os.getenv(f"{_ENV_PREFIX}TREND_MIN_YEARS"):
            kwargs["trend_min_years"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TREND_TRAJECTORY_WINDOW"):
            kwargs["trend_trajectory_window"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TREND_PREDICTION_HORIZON"):
            kwargs["trend_prediction_horizon"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TREND_MIN_R_SQUARED"):
            kwargs["trend_min_r_squared"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}TREND_REVERSAL_SENSITIVITY"):
            kwargs["trend_reversal_sensitivity"] = float(val)

        # Correlation analysis settings
        if val := os.getenv(f"{_ENV_PREFIX}CORRELATION_MIN_DATA_POINTS"):
            kwargs["correlation_min_data_points"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CORRELATION_SIGNIFICANCE_LEVEL"):
            kwargs["correlation_significance_level"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}CORRELATION_MIN_COEFFICIENT"):
            kwargs["correlation_min_coefficient"] = float(val)

        # Alert settings
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_CPI_CHANGE_THRESHOLD"):
            kwargs["alert_cpi_change_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_WGI_CHANGE_THRESHOLD"):
            kwargs["alert_wgi_change_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_TREND_REVERSAL"):
            kwargs["alert_trend_reversal"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_COOLDOWN_HOURS"):
            kwargs["alert_cooldown_hours"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_MAX_PER_COUNTRY_PER_DAY"):
            kwargs["alert_max_per_country_per_day"] = int(val)

        # EUDR Article 29 classification thresholds
        if val := os.getenv(f"{_ENV_PREFIX}ART29_LOW_RISK_CPI"):
            kwargs["art29_low_risk_cpi"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ART29_LOW_RISK_WGI"):
            kwargs["art29_low_risk_wgi"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}ART29_HIGH_RISK_CPI"):
            kwargs["art29_high_risk_cpi"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ART29_HIGH_RISK_WGI"):
            kwargs["art29_high_risk_wgi"] = float(val)

        # Due diligence settings
        if val := os.getenv(f"{_ENV_PREFIX}DD_SIMPLIFIED_COST_EUR"):
            kwargs["dd_simplified_cost_eur"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DD_STANDARD_COST_EUR"):
            kwargs["dd_standard_cost_eur"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DD_ENHANCED_COST_EUR"):
            kwargs["dd_enhanced_cost_eur"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DD_SIMPLIFIED_AUDIT_MONTHS"):
            kwargs["dd_simplified_audit_months"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DD_STANDARD_AUDIT_MONTHS"):
            kwargs["dd_standard_audit_months"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DD_ENHANCED_AUDIT_MONTHS"):
            kwargs["dd_enhanced_audit_months"] = int(val)

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
            >>> cfg = CorruptionIndexMonitorConfig()
            >>> d = cfg.to_dict(redact_secrets=True)
            >>> assert "database_url" in d
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
            "cpi_high_risk_threshold": self.cpi_high_risk_threshold,
            "cpi_moderate_threshold": self.cpi_moderate_threshold,
            "cpi_low_risk_threshold": self.cpi_low_risk_threshold,
            "wgi_risk_threshold": self.wgi_risk_threshold,
            "wgi_low_risk_threshold": self.wgi_low_risk_threshold,
            "wgi_dimension_weights": self.wgi_dimension_weights,
            "bribery_sector_weights": self.bribery_sector_weights,
            "bribery_risk_multiplier": self.bribery_risk_multiplier,
            "iq_judicial_weight": self.iq_judicial_weight,
            "iq_regulatory_weight": self.iq_regulatory_weight,
            "iq_forest_gov_weight": self.iq_forest_gov_weight,
            "iq_law_enforcement_weight": self.iq_law_enforcement_weight,
            "trend_min_years": self.trend_min_years,
            "trend_trajectory_window": self.trend_trajectory_window,
            "trend_prediction_horizon": self.trend_prediction_horizon,
            "trend_min_r_squared": self.trend_min_r_squared,
            "trend_reversal_sensitivity": self.trend_reversal_sensitivity,
            "correlation_min_data_points": self.correlation_min_data_points,
            "correlation_significance_level": self.correlation_significance_level,
            "correlation_min_coefficient": self.correlation_min_coefficient,
            "alert_cpi_change_threshold": self.alert_cpi_change_threshold,
            "alert_wgi_change_threshold": self.alert_wgi_change_threshold,
            "alert_trend_reversal": self.alert_trend_reversal,
            "alert_cooldown_hours": self.alert_cooldown_hours,
            "alert_max_per_country_per_day": self.alert_max_per_country_per_day,
            "art29_low_risk_cpi": self.art29_low_risk_cpi,
            "art29_low_risk_wgi": self.art29_low_risk_wgi,
            "art29_high_risk_cpi": self.art29_high_risk_cpi,
            "art29_high_risk_wgi": self.art29_high_risk_wgi,
            "dd_simplified_cost_eur": self.dd_simplified_cost_eur,
            "dd_standard_cost_eur": self.dd_standard_cost_eur,
            "dd_enhanced_cost_eur": self.dd_enhanced_cost_eur,
            "dd_simplified_audit_months": self.dd_simplified_audit_months,
            "dd_standard_audit_months": self.dd_standard_audit_months,
            "dd_enhanced_audit_months": self.dd_enhanced_audit_months,
            "output_formats": self.output_formats,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            "retention_years": self.retention_years,
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
_global_config: Optional[CorruptionIndexMonitorConfig] = None


def get_config() -> CorruptionIndexMonitorConfig:
    """Get the global CorruptionIndexMonitorConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance. Uses double-checked locking
    to minimize contention after initialization.

    Returns:
        CorruptionIndexMonitorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> assert cfg.cpi_high_risk_threshold == 30
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2  # Same instance
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = CorruptionIndexMonitorConfig.from_env()
    return _global_config


def set_config(config: CorruptionIndexMonitorConfig) -> None:
    """Set the global CorruptionIndexMonitorConfig singleton instance.

    Used for testing and programmatic configuration override.

    Args:
        config: CorruptionIndexMonitorConfig instance to set as global.

    Example:
        >>> from greenlang.agents.eudr.corruption_index_monitor.config import (
        ...     set_config, CorruptionIndexMonitorConfig,
        ... )
        >>> test_cfg = CorruptionIndexMonitorConfig(cpi_high_risk_threshold=25)
        >>> set_config(test_cfg)
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global CorruptionIndexMonitorConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_config()
        >>> # Next get_config() call will re-initialize from environment
    """
    global _global_config
    with _config_lock:
        _global_config = None
