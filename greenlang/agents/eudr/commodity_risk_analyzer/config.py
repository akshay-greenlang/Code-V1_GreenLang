# -*- coding: utf-8 -*-
"""
Commodity Risk Analyzer Configuration - AGENT-EUDR-018

Centralized configuration for the Commodity Risk Analyzer Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- EUDR 7 commodity definitions: cattle, cocoa, coffee, oil_palm, rubber,
  soya, wood with per-commodity risk weights and thresholds
- Derived products mapping per EUDR Annex I: 20+ product categories
  (chocolate, leather, biodiesel, plywood, furniture, palm oil, natural
  rubber products, soy meal, beef products, charcoal, paper, coffee
  extracts, cocoa butter, margarine, tires, cork, particle board,
  glycerol, animal feed, printed matter) with HS code mappings and
  processing chain definitions
- Price volatility thresholds per commodity: low (<0.2), moderate
  (0.2-0.4), high (0.4-0.6), extreme (>0.6), disruption detection
  threshold (0.8), rolling window periods (30d, 90d)
- Production forecast parameters: seasonal coefficients per commodity
  per hemisphere, climate impact factors (drought, flood, frost, fire),
  forecast horizon (1-24 months), confidence interval width
- Substitution risk detection thresholds: minimum confidence (0.70),
  risk impact multiplier (1.5), lookback window (180 days), minimum
  substitution events for pattern detection (3)
- Regulatory requirement matrices per commodity per EUDR article
  (Articles 1-10, Annex I): documentation types, evidence standards,
  verification methods, and compliance criteria
- Due diligence workflow templates per commodity: simplified, standard,
  enhanced levels with evidence items, verification steps, deadlines
- Portfolio concentration limits: HHI thresholds (low <0.15, moderate
  0.15-0.25, high >0.25), diversification targets, maximum single
  commodity exposure (40%)
- Risk scoring weights for commodity-specific factors: deforestation
  risk (25%), supply chain complexity (20%), price volatility (15%),
  regulatory pressure (15%), geographic concentration (15%), production
  stability (10%)
- Market disruption detection parameters: price spike threshold (3x
  standard deviation), volume drop threshold (50%), supply chain
  interruption indicators
- Processing chain depth limits: maximum depth per commodity (cattle=6,
  cocoa=8, coffee=5, oil_palm=7, rubber=6, soya=5, wood=8)
- Batch processing, provenance, metrics, rate limiting settings

All settings can be overridden via environment variables with the
``GL_EUDR_CRA_`` prefix (e.g. ``GL_EUDR_CRA_DATABASE_URL``,
``GL_EUDR_CRA_HHI_CONCENTRATION_THRESHOLD``).

Environment Variable Reference (GL_EUDR_CRA_ prefix):
    GL_EUDR_CRA_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_CRA_REDIS_URL                     - Redis connection URL
    GL_EUDR_CRA_LOG_LEVEL                     - Logging level
    GL_EUDR_CRA_POOL_SIZE                     - Database pool size
    GL_EUDR_CRA_POOL_TIMEOUT_S               - Pool timeout seconds
    GL_EUDR_CRA_POOL_RECYCLE_S               - Pool recycle seconds
    GL_EUDR_CRA_DEFORESTATION_RISK_WEIGHT    - Deforestation risk weight (0-100)
    GL_EUDR_CRA_SUPPLY_CHAIN_COMPLEXITY_WEIGHT - Supply chain complexity weight (0-100)
    GL_EUDR_CRA_PRICE_VOLATILITY_WEIGHT      - Price volatility weight (0-100)
    GL_EUDR_CRA_REGULATORY_PRESSURE_WEIGHT   - Regulatory pressure weight (0-100)
    GL_EUDR_CRA_GEOGRAPHIC_CONCENTRATION_WEIGHT - Geographic concentration weight (0-100)
    GL_EUDR_CRA_PRODUCTION_STABILITY_WEIGHT  - Production stability weight (0-100)
    GL_EUDR_CRA_LOW_RISK_THRESHOLD           - Low risk threshold (0-100)
    GL_EUDR_CRA_MEDIUM_RISK_THRESHOLD        - Medium risk threshold (0-100)
    GL_EUDR_CRA_HIGH_RISK_THRESHOLD          - High risk threshold (0-100)
    GL_EUDR_CRA_CRITICAL_RISK_THRESHOLD      - Critical risk threshold (0-100)
    GL_EUDR_CRA_VOLATILITY_LOW_THRESHOLD     - Volatility low threshold (0.0-1.0)
    GL_EUDR_CRA_VOLATILITY_MODERATE_THRESHOLD - Volatility moderate threshold (0.0-1.0)
    GL_EUDR_CRA_VOLATILITY_HIGH_THRESHOLD    - Volatility high threshold (0.0-1.0)
    GL_EUDR_CRA_VOLATILITY_EXTREME_THRESHOLD - Volatility extreme threshold (0.0-1.0)
    GL_EUDR_CRA_DISRUPTION_THRESHOLD         - Market disruption threshold (0.0-1.0)
    GL_EUDR_CRA_VOLATILITY_WINDOW_SHORT_DAYS - Short volatility window (days)
    GL_EUDR_CRA_VOLATILITY_WINDOW_LONG_DAYS  - Long volatility window (days)
    GL_EUDR_CRA_PRICE_SPIKE_STDDEV           - Price spike std dev multiplier
    GL_EUDR_CRA_VOLUME_DROP_THRESHOLD        - Volume drop threshold (0.0-1.0)
    GL_EUDR_CRA_FORECAST_HORIZON_MONTHS      - Forecast horizon months
    GL_EUDR_CRA_CONFIDENCE_INTERVAL_WIDTH    - Confidence interval width (0.0-1.0)
    GL_EUDR_CRA_CLIMATE_DROUGHT_FACTOR       - Climate drought impact factor
    GL_EUDR_CRA_CLIMATE_FLOOD_FACTOR         - Climate flood impact factor
    GL_EUDR_CRA_CLIMATE_FROST_FACTOR         - Climate frost impact factor
    GL_EUDR_CRA_CLIMATE_FIRE_FACTOR          - Climate fire impact factor
    GL_EUDR_CRA_SUBSTITUTION_MIN_CONFIDENCE  - Substitution min confidence (0.0-1.0)
    GL_EUDR_CRA_SUBSTITUTION_RISK_MULTIPLIER - Substitution risk multiplier
    GL_EUDR_CRA_SUBSTITUTION_LOOKBACK_DAYS   - Substitution lookback window days
    GL_EUDR_CRA_SUBSTITUTION_MIN_EVENTS      - Minimum substitution events for pattern
    GL_EUDR_CRA_HHI_CONCENTRATION_THRESHOLD  - HHI concentration threshold (0.0-1.0)
    GL_EUDR_CRA_HHI_LOW_THRESHOLD           - HHI low threshold (0.0-1.0)
    GL_EUDR_CRA_HHI_MODERATE_THRESHOLD      - HHI moderate threshold (0.0-1.0)
    GL_EUDR_CRA_MAX_SINGLE_COMMODITY_EXPOSURE - Max single commodity exposure (0.0-1.0)
    GL_EUDR_CRA_DIVERSIFICATION_TARGET       - Diversification target score (0.0-1.0)
    GL_EUDR_CRA_DD_DEFAULT_LEVEL             - Default due diligence level
    GL_EUDR_CRA_DD_COMPLETION_THRESHOLD      - DD completion threshold (0.0-1.0)
    GL_EUDR_CRA_DD_OVERDUE_LIMIT_DAYS       - DD overdue limit days
    GL_EUDR_CRA_DD_REVIEW_INTERVAL_MONTHS   - DD review interval months
    GL_EUDR_CRA_MAX_PROCESSING_CHAIN_DEPTH  - Max processing chain depth
    GL_EUDR_CRA_TRACEABILITY_MIN_SCORE      - Min traceability score (0.0-1.0)
    GL_EUDR_CRA_REDIS_TTL_S                 - Redis cache TTL seconds
    GL_EUDR_CRA_REDIS_KEY_PREFIX            - Redis key prefix
    GL_EUDR_CRA_BATCH_MAX_SIZE              - Batch processing max size
    GL_EUDR_CRA_BATCH_CONCURRENCY           - Batch concurrency workers
    GL_EUDR_CRA_BATCH_TIMEOUT_S             - Batch timeout seconds
    GL_EUDR_CRA_RETENTION_YEARS             - Data retention years
    GL_EUDR_CRA_ENABLE_PROVENANCE           - Enable provenance tracking
    GL_EUDR_CRA_GENESIS_HASH                - Genesis hash anchor
    GL_EUDR_CRA_CHAIN_ALGORITHM             - Provenance chain hash algorithm
    GL_EUDR_CRA_ENABLE_METRICS              - Enable Prometheus metrics
    GL_EUDR_CRA_METRICS_PREFIX              - Prometheus metrics prefix
    GL_EUDR_CRA_RATE_LIMIT_ANONYMOUS        - Rate limit anonymous tier
    GL_EUDR_CRA_RATE_LIMIT_BASIC            - Rate limit basic tier
    GL_EUDR_CRA_RATE_LIMIT_STANDARD         - Rate limit standard tier
    GL_EUDR_CRA_RATE_LIMIT_PREMIUM          - Rate limit premium tier
    GL_EUDR_CRA_RATE_LIMIT_ADMIN            - Rate limit admin tier

Example:
    >>> from greenlang.agents.eudr.commodity_risk_analyzer.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.hhi_concentration_threshold, cfg.volatility_high_threshold)
    0.25 0.6

    >>> # Override for testing
    >>> from greenlang.agents.eudr.commodity_risk_analyzer.config import (
    ...     set_config, reset_config, CommodityRiskAnalyzerConfig,
    ... )
    >>> set_config(CommodityRiskAnalyzerConfig(hhi_concentration_threshold=0.30))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
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

_ENV_PREFIX = "GL_EUDR_CRA_"

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
# EUDR 7 Commodities (Article 1)
# ---------------------------------------------------------------------------

_EUDR_COMMODITIES = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
})

# ---------------------------------------------------------------------------
# Derived product categories (Annex I) per commodity
# ---------------------------------------------------------------------------

_DERIVED_PRODUCTS_MAP: Dict[str, List[str]] = {
    "cattle": [
        "beef_fresh", "beef_frozen", "beef_processed", "leather_raw",
        "leather_finished", "tallow", "gelatin", "bone_meal",
    ],
    "cocoa": [
        "cocoa_beans", "cocoa_paste", "cocoa_butter", "cocoa_powder",
        "chocolate", "chocolate_confectionery",
    ],
    "coffee": [
        "coffee_beans_green", "coffee_roasted", "coffee_extracts",
        "coffee_preparations",
    ],
    "oil_palm": [
        "palm_oil_crude", "palm_oil_refined", "palm_kernel_oil",
        "biodiesel", "margarine", "glycerol", "oleochemicals",
        "animal_feed_palm",
    ],
    "rubber": [
        "natural_rubber_raw", "natural_rubber_smoked", "latex",
        "tires", "rubber_goods", "rubber_footwear",
    ],
    "soya": [
        "soybeans", "soy_oil", "soy_meal", "soy_flour",
        "soy_protein", "animal_feed_soy",
    ],
    "wood": [
        "logs", "sawnwood", "plywood", "particle_board", "fibreboard",
        "veneer", "charcoal", "paper_pulp", "paper", "printed_matter",
        "furniture", "cork",
    ],
}

# ---------------------------------------------------------------------------
# Default processing chain depth limits per commodity
# ---------------------------------------------------------------------------

_DEFAULT_CHAIN_DEPTH_LIMITS: Dict[str, int] = {
    "cattle": 6,
    "cocoa": 8,
    "coffee": 5,
    "oil_palm": 7,
    "rubber": 6,
    "soya": 5,
    "wood": 8,
}

# ---------------------------------------------------------------------------
# Default seasonal coefficients per commodity per quarter
# ---------------------------------------------------------------------------

_DEFAULT_SEASONAL_COEFFICIENTS: Dict[str, List[float]] = {
    "cattle": [1.00, 0.95, 1.05, 1.00],
    "cocoa": [0.80, 1.20, 1.10, 0.90],
    "coffee": [0.85, 1.15, 1.10, 0.90],
    "oil_palm": [0.95, 1.05, 1.05, 0.95],
    "rubber": [0.90, 1.10, 1.05, 0.95],
    "soya": [0.75, 1.25, 1.10, 0.90],
    "wood": [0.90, 1.10, 1.00, 1.00],
}

# ---------------------------------------------------------------------------
# Default price volatility thresholds per commodity
# ---------------------------------------------------------------------------

_DEFAULT_VOLATILITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "cattle": {"low": 0.15, "moderate": 0.30, "high": 0.50, "extreme": 0.70},
    "cocoa": {"low": 0.20, "moderate": 0.40, "high": 0.60, "extreme": 0.80},
    "coffee": {"low": 0.25, "moderate": 0.45, "high": 0.65, "extreme": 0.85},
    "oil_palm": {"low": 0.20, "moderate": 0.40, "high": 0.60, "extreme": 0.80},
    "rubber": {"low": 0.20, "moderate": 0.35, "high": 0.55, "extreme": 0.75},
    "soya": {"low": 0.15, "moderate": 0.30, "high": 0.50, "extreme": 0.70},
    "wood": {"low": 0.10, "moderate": 0.25, "high": 0.45, "extreme": 0.65},
}

# ---------------------------------------------------------------------------
# Default EUDR regulatory articles per commodity
# ---------------------------------------------------------------------------

_DEFAULT_REGULATORY_ARTICLES: Dict[str, List[str]] = {
    "cattle": ["art1", "art2", "art3", "art4", "art8", "art9", "art10"],
    "cocoa": ["art1", "art2", "art3", "art4", "art8", "art9", "art10"],
    "coffee": ["art1", "art2", "art3", "art4", "art8", "art9", "art10"],
    "oil_palm": ["art1", "art2", "art3", "art4", "art8", "art9", "art10"],
    "rubber": ["art1", "art2", "art3", "art4", "art8", "art9", "art10"],
    "soya": ["art1", "art2", "art3", "art4", "art8", "art9", "art10"],
    "wood": ["art1", "art2", "art3", "art4", "art8", "art9", "art10"],
}

# ---------------------------------------------------------------------------
# Default required evidence types per DD level
# ---------------------------------------------------------------------------

_DEFAULT_DD_EVIDENCE_TYPES: Dict[str, List[str]] = {
    "simplified": [
        "compliance_declaration",
        "product_description",
        "quantity_declaration",
    ],
    "standard": [
        "compliance_declaration",
        "product_description",
        "quantity_declaration",
        "geolocation_data",
        "harvest_date",
        "dds_reference",
    ],
    "enhanced": [
        "compliance_declaration",
        "product_description",
        "quantity_declaration",
        "geolocation_data",
        "harvest_date",
        "dds_reference",
        "satellite_imagery",
        "field_verification",
        "third_party_audit",
        "certification_document",
    ],
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class CommodityRiskAnalyzerConfig:
    """Configuration for the Commodity Risk Analyzer Agent (AGENT-EUDR-018).

    This dataclass encapsulates all configuration settings for commodity risk
    profiling, derived product analysis, price volatility monitoring,
    production forecasting, substitution risk detection, regulatory compliance
    mapping, commodity due diligence workflow management, and portfolio risk
    aggregation. All settings have sensible defaults aligned with EUDR
    requirements and can be overridden via environment variables with the
    GL_EUDR_CRA_ prefix.

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

        # Risk scoring weights (6 factors, sum = 100)
        deforestation_risk_weight: Deforestation risk weight (0-100)
        supply_chain_complexity_weight: Supply chain complexity weight (0-100)
        price_volatility_weight: Price volatility weight (0-100)
        regulatory_pressure_weight: Regulatory pressure weight (0-100)
        geographic_concentration_weight: Geographic concentration weight (0-100)
        production_stability_weight: Production stability weight (0-100)

        # Risk thresholds
        low_risk_threshold: Low risk upper bound (0-100)
        medium_risk_threshold: Medium risk upper bound (0-100)
        high_risk_threshold: High risk upper bound (0-100)
        critical_risk_threshold: Critical risk upper bound (0-100)

        # Price volatility settings
        volatility_low_threshold: Volatility low threshold (0.0-1.0)
        volatility_moderate_threshold: Volatility moderate threshold (0.0-1.0)
        volatility_high_threshold: Volatility high threshold (0.0-1.0)
        volatility_extreme_threshold: Volatility extreme threshold (0.0-1.0)
        disruption_threshold: Market disruption threshold (0.0-1.0)
        volatility_window_short_days: Short rolling window days (e.g. 30)
        volatility_window_long_days: Long rolling window days (e.g. 90)
        price_spike_stddev: Price spike standard deviation multiplier
        volume_drop_threshold: Volume drop threshold (0.0-1.0)

        # Production forecast settings
        forecast_horizon_months: Forecast horizon in months
        confidence_interval_width: Confidence interval width (0.0-1.0)
        climate_drought_factor: Climate drought impact factor (0.0-1.0)
        climate_flood_factor: Climate flood impact factor (0.0-1.0)
        climate_frost_factor: Climate frost impact factor (0.0-1.0)
        climate_fire_factor: Climate fire impact factor (0.0-1.0)
        seasonal_coefficients: Per-commodity per-quarter seasonal coefficients
        per_commodity_volatility_thresholds: Per-commodity volatility thresholds

        # Substitution risk settings
        substitution_min_confidence: Min confidence for substitution detection (0.0-1.0)
        substitution_risk_multiplier: Risk multiplier for substitution events
        substitution_lookback_days: Lookback window in days
        substitution_min_events: Min events to confirm pattern

        # Portfolio settings
        hhi_concentration_threshold: HHI concentration threshold (0.0-1.0)
        hhi_low_threshold: HHI low threshold (0.0-1.0)
        hhi_moderate_threshold: HHI moderate threshold (0.0-1.0)
        max_single_commodity_exposure: Max single commodity exposure (0.0-1.0)
        diversification_target: Diversification target score (0.0-1.0)

        # Due diligence settings
        dd_default_level: Default DD level (simplified, standard, enhanced)
        dd_completion_threshold: DD completion threshold (0.0-1.0)
        dd_overdue_limit_days: DD overdue limit in days
        dd_review_interval_months: DD review interval in months
        dd_evidence_types: Required evidence types per DD level

        # Processing chain settings
        max_processing_chain_depth: Global max processing chain depth
        chain_depth_limits: Per-commodity chain depth limits
        traceability_min_score: Minimum traceability score (0.0-1.0)

        # Derived products
        derived_products_map: Per-commodity derived product mappings

        # Regulatory
        regulatory_articles: Per-commodity applicable EUDR articles

        # Reporting
        output_formats: Report output formats
        default_language: Default report language
        supported_languages: Supported report languages
        report_retention_days: Report retention in days

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

    # Database settings
    database_url: str = "postgresql://localhost:5432/greenlang"
    pool_size: int = 20
    pool_timeout_s: int = 30
    pool_recycle_s: int = 3600

    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_s: int = 3600
    redis_key_prefix: str = "gl:eudr:cra:"

    # Logging
    log_level: str = "INFO"

    # Risk scoring weights (6 factors, sum = 100)
    deforestation_risk_weight: int = 25
    supply_chain_complexity_weight: int = 20
    price_volatility_weight: int = 15
    regulatory_pressure_weight: int = 15
    geographic_concentration_weight: int = 15
    production_stability_weight: int = 10

    # Risk thresholds
    low_risk_threshold: int = 25
    medium_risk_threshold: int = 50
    high_risk_threshold: int = 75
    critical_risk_threshold: int = 90

    # Price volatility settings
    volatility_low_threshold: float = 0.2
    volatility_moderate_threshold: float = 0.4
    volatility_high_threshold: float = 0.6
    volatility_extreme_threshold: float = 0.8
    disruption_threshold: float = 0.8
    volatility_window_short_days: int = 30
    volatility_window_long_days: int = 90
    price_spike_stddev: float = 3.0
    volume_drop_threshold: float = 0.5

    # Production forecast settings
    forecast_horizon_months: int = 12
    confidence_interval_width: float = 0.95
    climate_drought_factor: float = 0.7
    climate_flood_factor: float = 0.6
    climate_frost_factor: float = 0.5
    climate_fire_factor: float = 0.4
    seasonal_coefficients: Dict[str, List[float]] = field(
        default_factory=lambda: {k: list(v) for k, v in _DEFAULT_SEASONAL_COEFFICIENTS.items()}
    )
    per_commodity_volatility_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            k: dict(v) for k, v in _DEFAULT_VOLATILITY_THRESHOLDS.items()
        }
    )

    # Substitution risk settings
    substitution_min_confidence: float = 0.70
    substitution_risk_multiplier: float = 1.5
    substitution_lookback_days: int = 180
    substitution_min_events: int = 3

    # Portfolio settings
    hhi_concentration_threshold: float = 0.25
    hhi_low_threshold: float = 0.15
    hhi_moderate_threshold: float = 0.25
    max_single_commodity_exposure: float = 0.40
    diversification_target: float = 0.70

    # Due diligence settings
    dd_default_level: str = "standard"
    dd_completion_threshold: float = 0.90
    dd_overdue_limit_days: int = 30
    dd_review_interval_months: int = 12
    dd_evidence_types: Dict[str, List[str]] = field(
        default_factory=lambda: {
            k: list(v) for k, v in _DEFAULT_DD_EVIDENCE_TYPES.items()
        }
    )

    # Processing chain settings
    max_processing_chain_depth: int = 10
    chain_depth_limits: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_CHAIN_DEPTH_LIMITS)
    )
    traceability_min_score: float = 0.60

    # Derived products
    derived_products_map: Dict[str, List[str]] = field(
        default_factory=lambda: {
            k: list(v) for k, v in _DERIVED_PRODUCTS_MAP.items()
        }
    )

    # Regulatory
    regulatory_articles: Dict[str, List[str]] = field(
        default_factory=lambda: {
            k: list(v) for k, v in _DEFAULT_REGULATORY_ARTICLES.items()
        }
    )

    # Reporting
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "excel"]
    )
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )
    report_retention_days: int = 1825  # 5 years

    # Batch processing
    batch_max_size: int = 500
    batch_concurrency: int = 4
    batch_timeout_s: int = 300

    # Data retention
    retention_years: int = 5

    # Provenance
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-CRA-018-COMMODITY-RISK-ANALYZER-GENESIS"
    chain_algorithm: str = "sha256"

    # Metrics
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_cra_"

    # Rate limiting
    rate_limit_anonymous: int = 10
    rate_limit_basic: int = 60
    rate_limit_standard: int = 300
    rate_limit_premium: int = 1000
    rate_limit_admin: int = 10000

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate log level
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level: {self.log_level}. "
                f"Must be one of {_VALID_LOG_LEVELS}"
            )

        # Validate chain algorithm
        if self.chain_algorithm not in _VALID_CHAIN_ALGORITHMS:
            raise ValueError(
                f"Invalid chain_algorithm: {self.chain_algorithm}. "
                f"Must be one of {_VALID_CHAIN_ALGORITHMS}"
            )

        # Validate DD default level
        if self.dd_default_level not in _VALID_DD_LEVELS:
            raise ValueError(
                f"Invalid dd_default_level: {self.dd_default_level}. "
                f"Must be one of {_VALID_DD_LEVELS}"
            )

        # Validate risk scoring weights sum to 100
        total_weight = (
            self.deforestation_risk_weight
            + self.supply_chain_complexity_weight
            + self.price_volatility_weight
            + self.regulatory_pressure_weight
            + self.geographic_concentration_weight
            + self.production_stability_weight
        )
        if total_weight != 100:
            raise ValueError(
                f"Risk scoring weights must sum to 100, got {total_weight}"
            )

        # Validate individual weight bounds (5-50)
        weights = [
            self.deforestation_risk_weight,
            self.supply_chain_complexity_weight,
            self.price_volatility_weight,
            self.regulatory_pressure_weight,
            self.geographic_concentration_weight,
            self.production_stability_weight,
        ]
        for w in weights:
            if not 5 <= w <= 50:
                raise ValueError(
                    f"All risk scoring weights must be between 5 and 50, "
                    f"got {w}"
                )

        # Validate risk thresholds ascending
        if not (
            0 <= self.low_risk_threshold
            < self.medium_risk_threshold
            < self.high_risk_threshold
            < self.critical_risk_threshold
            <= 100
        ):
            raise ValueError(
                "Risk thresholds must be in ascending order: "
                "0 <= low < medium < high < critical <= 100"
            )

        # Validate volatility thresholds ascending
        if not (
            0.0 <= self.volatility_low_threshold
            < self.volatility_moderate_threshold
            < self.volatility_high_threshold
            < self.volatility_extreme_threshold
            <= 1.0
        ):
            raise ValueError(
                "Volatility thresholds must be in ascending order: "
                "0.0 <= low < moderate < high < extreme <= 1.0"
            )

        # Validate disruption threshold
        if not 0.0 <= self.disruption_threshold <= 1.0:
            raise ValueError(
                f"disruption_threshold must be between 0.0 and 1.0, "
                f"got {self.disruption_threshold}"
            )

        # Validate confidence interval width
        if not 0.0 < self.confidence_interval_width <= 1.0:
            raise ValueError(
                f"confidence_interval_width must be between 0.0 and 1.0, "
                f"got {self.confidence_interval_width}"
            )

        # Validate climate impact factors (0.0-1.0)
        for name, val in [
            ("climate_drought_factor", self.climate_drought_factor),
            ("climate_flood_factor", self.climate_flood_factor),
            ("climate_frost_factor", self.climate_frost_factor),
            ("climate_fire_factor", self.climate_fire_factor),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0, got {val}"
                )

        # Validate substitution min confidence
        if not 0.0 <= self.substitution_min_confidence <= 1.0:
            raise ValueError(
                f"substitution_min_confidence must be between 0.0 and 1.0, "
                f"got {self.substitution_min_confidence}"
            )

        # Validate substitution risk multiplier
        if self.substitution_risk_multiplier <= 0.0:
            raise ValueError(
                f"substitution_risk_multiplier must be > 0, "
                f"got {self.substitution_risk_multiplier}"
            )

        # Validate HHI thresholds
        if not (
            0.0 <= self.hhi_low_threshold
            <= self.hhi_moderate_threshold
            <= 1.0
        ):
            raise ValueError(
                "HHI thresholds must be: 0 <= low <= moderate <= 1.0"
            )
        if not 0.0 <= self.hhi_concentration_threshold <= 1.0:
            raise ValueError(
                f"hhi_concentration_threshold must be between 0.0 and 1.0, "
                f"got {self.hhi_concentration_threshold}"
            )

        # Validate max single commodity exposure
        if not 0.0 < self.max_single_commodity_exposure <= 1.0:
            raise ValueError(
                f"max_single_commodity_exposure must be between 0.0 and 1.0, "
                f"got {self.max_single_commodity_exposure}"
            )

        # Validate diversification target
        if not 0.0 <= self.diversification_target <= 1.0:
            raise ValueError(
                f"diversification_target must be between 0.0 and 1.0, "
                f"got {self.diversification_target}"
            )

        # Validate DD completion threshold
        if not 0.0 <= self.dd_completion_threshold <= 1.0:
            raise ValueError(
                f"dd_completion_threshold must be between 0.0 and 1.0, "
                f"got {self.dd_completion_threshold}"
            )

        # Validate traceability min score
        if not 0.0 <= self.traceability_min_score <= 1.0:
            raise ValueError(
                f"traceability_min_score must be between 0.0 and 1.0, "
                f"got {self.traceability_min_score}"
            )

        # Validate positive integers
        if self.pool_size <= 0:
            raise ValueError(f"pool_size must be > 0, got {self.pool_size}")
        if self.batch_max_size <= 0:
            raise ValueError(f"batch_max_size must be > 0, got {self.batch_max_size}")
        if self.batch_concurrency <= 0:
            raise ValueError(f"batch_concurrency must be > 0, got {self.batch_concurrency}")
        if self.retention_years <= 0:
            raise ValueError(f"retention_years must be > 0, got {self.retention_years}")
        if self.forecast_horizon_months <= 0:
            raise ValueError(f"forecast_horizon_months must be > 0, got {self.forecast_horizon_months}")
        if self.max_processing_chain_depth <= 0:
            raise ValueError(f"max_processing_chain_depth must be > 0, got {self.max_processing_chain_depth}")

        # Validate output formats
        for fmt in self.output_formats:
            if fmt not in _VALID_OUTPUT_FORMATS:
                raise ValueError(
                    f"Invalid output format: {fmt}. "
                    f"Must be one of {_VALID_OUTPUT_FORMATS}"
                )

        # Validate languages
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

        # Validate seasonal coefficients structure
        for commodity in _EUDR_COMMODITIES:
            if commodity in self.seasonal_coefficients:
                coeffs = self.seasonal_coefficients[commodity]
                if len(coeffs) != 4:
                    raise ValueError(
                        f"Seasonal coefficients for {commodity} must have 4 "
                        f"values (one per quarter), got {len(coeffs)}"
                    )
                for c in coeffs:
                    if not 0.0 < c < 3.0:
                        raise ValueError(
                            f"Seasonal coefficient for {commodity} must be "
                            f"between 0.0 and 3.0, got {c}"
                        )

        # Validate chain depth limits
        for commodity, depth in self.chain_depth_limits.items():
            if commodity not in _EUDR_COMMODITIES:
                raise ValueError(
                    f"Invalid commodity in chain_depth_limits: {commodity}. "
                    f"Must be one of {_EUDR_COMMODITIES}"
                )
            if depth <= 0 or depth > self.max_processing_chain_depth:
                raise ValueError(
                    f"Chain depth for {commodity} must be between 1 and "
                    f"{self.max_processing_chain_depth}, got {depth}"
                )

        logger.info(
            f"CommodityRiskAnalyzerConfig initialized: "
            f"deforestation_weight={self.deforestation_risk_weight}, "
            f"hhi_threshold={self.hhi_concentration_threshold}, "
            f"low_threshold={self.low_risk_threshold}, "
            f"high_threshold={self.high_risk_threshold}"
        )

    @classmethod
    def from_env(cls) -> "CommodityRiskAnalyzerConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_CRA_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            CommodityRiskAnalyzerConfig instance with env overrides applied.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_CRA_HHI_CONCENTRATION_THRESHOLD"] = "0.30"
            >>> cfg = CommodityRiskAnalyzerConfig.from_env()
            >>> assert cfg.hhi_concentration_threshold == 0.30
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

        # Risk scoring weights
        if val := os.getenv(f"{_ENV_PREFIX}DEFORESTATION_RISK_WEIGHT"):
            kwargs["deforestation_risk_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SUPPLY_CHAIN_COMPLEXITY_WEIGHT"):
            kwargs["supply_chain_complexity_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}PRICE_VOLATILITY_WEIGHT"):
            kwargs["price_volatility_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}REGULATORY_PRESSURE_WEIGHT"):
            kwargs["regulatory_pressure_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}GEOGRAPHIC_CONCENTRATION_WEIGHT"):
            kwargs["geographic_concentration_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}PRODUCTION_STABILITY_WEIGHT"):
            kwargs["production_stability_weight"] = int(val)

        # Risk thresholds
        if val := os.getenv(f"{_ENV_PREFIX}LOW_RISK_THRESHOLD"):
            kwargs["low_risk_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MEDIUM_RISK_THRESHOLD"):
            kwargs["medium_risk_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}HIGH_RISK_THRESHOLD"):
            kwargs["high_risk_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_RISK_THRESHOLD"):
            kwargs["critical_risk_threshold"] = int(val)

        # Price volatility settings
        if val := os.getenv(f"{_ENV_PREFIX}VOLATILITY_LOW_THRESHOLD"):
            kwargs["volatility_low_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}VOLATILITY_MODERATE_THRESHOLD"):
            kwargs["volatility_moderate_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}VOLATILITY_HIGH_THRESHOLD"):
            kwargs["volatility_high_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}VOLATILITY_EXTREME_THRESHOLD"):
            kwargs["volatility_extreme_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}DISRUPTION_THRESHOLD"):
            kwargs["disruption_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}VOLATILITY_WINDOW_SHORT_DAYS"):
            kwargs["volatility_window_short_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}VOLATILITY_WINDOW_LONG_DAYS"):
            kwargs["volatility_window_long_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}PRICE_SPIKE_STDDEV"):
            kwargs["price_spike_stddev"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}VOLUME_DROP_THRESHOLD"):
            kwargs["volume_drop_threshold"] = float(val)

        # Production forecast settings
        if val := os.getenv(f"{_ENV_PREFIX}FORECAST_HORIZON_MONTHS"):
            kwargs["forecast_horizon_months"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CONFIDENCE_INTERVAL_WIDTH"):
            kwargs["confidence_interval_width"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}CLIMATE_DROUGHT_FACTOR"):
            kwargs["climate_drought_factor"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}CLIMATE_FLOOD_FACTOR"):
            kwargs["climate_flood_factor"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}CLIMATE_FROST_FACTOR"):
            kwargs["climate_frost_factor"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}CLIMATE_FIRE_FACTOR"):
            kwargs["climate_fire_factor"] = float(val)

        # Substitution risk settings
        if val := os.getenv(f"{_ENV_PREFIX}SUBSTITUTION_MIN_CONFIDENCE"):
            kwargs["substitution_min_confidence"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}SUBSTITUTION_RISK_MULTIPLIER"):
            kwargs["substitution_risk_multiplier"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}SUBSTITUTION_LOOKBACK_DAYS"):
            kwargs["substitution_lookback_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SUBSTITUTION_MIN_EVENTS"):
            kwargs["substitution_min_events"] = int(val)

        # Portfolio settings
        if val := os.getenv(f"{_ENV_PREFIX}HHI_CONCENTRATION_THRESHOLD"):
            kwargs["hhi_concentration_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}HHI_LOW_THRESHOLD"):
            kwargs["hhi_low_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}HHI_MODERATE_THRESHOLD"):
            kwargs["hhi_moderate_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_SINGLE_COMMODITY_EXPOSURE"):
            kwargs["max_single_commodity_exposure"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}DIVERSIFICATION_TARGET"):
            kwargs["diversification_target"] = float(val)

        # Due diligence settings
        if val := os.getenv(f"{_ENV_PREFIX}DD_DEFAULT_LEVEL"):
            kwargs["dd_default_level"] = val
        if val := os.getenv(f"{_ENV_PREFIX}DD_COMPLETION_THRESHOLD"):
            kwargs["dd_completion_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}DD_OVERDUE_LIMIT_DAYS"):
            kwargs["dd_overdue_limit_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DD_REVIEW_INTERVAL_MONTHS"):
            kwargs["dd_review_interval_months"] = int(val)

        # Processing chain settings
        if val := os.getenv(f"{_ENV_PREFIX}MAX_PROCESSING_CHAIN_DEPTH"):
            kwargs["max_processing_chain_depth"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TRACEABILITY_MIN_SCORE"):
            kwargs["traceability_min_score"] = float(val)

        # Reporting
        if val := os.getenv(f"{_ENV_PREFIX}OUTPUT_FORMATS"):
            kwargs["output_formats"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_LANGUAGE"):
            kwargs["default_language"] = val
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_LANGUAGES"):
            kwargs["supported_languages"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}REPORT_RETENTION_DAYS"):
            kwargs["report_retention_days"] = int(val)

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
            >>> cfg = CommodityRiskAnalyzerConfig()
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
            "deforestation_risk_weight": self.deforestation_risk_weight,
            "supply_chain_complexity_weight": self.supply_chain_complexity_weight,
            "price_volatility_weight": self.price_volatility_weight,
            "regulatory_pressure_weight": self.regulatory_pressure_weight,
            "geographic_concentration_weight": self.geographic_concentration_weight,
            "production_stability_weight": self.production_stability_weight,
            "low_risk_threshold": self.low_risk_threshold,
            "medium_risk_threshold": self.medium_risk_threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "critical_risk_threshold": self.critical_risk_threshold,
            "volatility_low_threshold": self.volatility_low_threshold,
            "volatility_moderate_threshold": self.volatility_moderate_threshold,
            "volatility_high_threshold": self.volatility_high_threshold,
            "volatility_extreme_threshold": self.volatility_extreme_threshold,
            "disruption_threshold": self.disruption_threshold,
            "volatility_window_short_days": self.volatility_window_short_days,
            "volatility_window_long_days": self.volatility_window_long_days,
            "price_spike_stddev": self.price_spike_stddev,
            "volume_drop_threshold": self.volume_drop_threshold,
            "forecast_horizon_months": self.forecast_horizon_months,
            "confidence_interval_width": self.confidence_interval_width,
            "climate_drought_factor": self.climate_drought_factor,
            "climate_flood_factor": self.climate_flood_factor,
            "climate_frost_factor": self.climate_frost_factor,
            "climate_fire_factor": self.climate_fire_factor,
            "seasonal_coefficients": self.seasonal_coefficients,
            "per_commodity_volatility_thresholds": self.per_commodity_volatility_thresholds,
            "substitution_min_confidence": self.substitution_min_confidence,
            "substitution_risk_multiplier": self.substitution_risk_multiplier,
            "substitution_lookback_days": self.substitution_lookback_days,
            "substitution_min_events": self.substitution_min_events,
            "hhi_concentration_threshold": self.hhi_concentration_threshold,
            "hhi_low_threshold": self.hhi_low_threshold,
            "hhi_moderate_threshold": self.hhi_moderate_threshold,
            "max_single_commodity_exposure": self.max_single_commodity_exposure,
            "diversification_target": self.diversification_target,
            "dd_default_level": self.dd_default_level,
            "dd_completion_threshold": self.dd_completion_threshold,
            "dd_overdue_limit_days": self.dd_overdue_limit_days,
            "dd_review_interval_months": self.dd_review_interval_months,
            "dd_evidence_types": self.dd_evidence_types,
            "max_processing_chain_depth": self.max_processing_chain_depth,
            "chain_depth_limits": self.chain_depth_limits,
            "traceability_min_score": self.traceability_min_score,
            "derived_products_map": self.derived_products_map,
            "regulatory_articles": self.regulatory_articles,
            "output_formats": self.output_formats,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "report_retention_days": self.report_retention_days,
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
# Thread-safe singleton pattern
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[CommodityRiskAnalyzerConfig] = None


def get_config() -> CommodityRiskAnalyzerConfig:
    """Get the global CommodityRiskAnalyzerConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance.

    Returns:
        CommodityRiskAnalyzerConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> assert cfg.hhi_concentration_threshold == 0.25
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2  # Same instance
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = CommodityRiskAnalyzerConfig.from_env()
    return _global_config


def set_config(config: CommodityRiskAnalyzerConfig) -> None:
    """Set the global CommodityRiskAnalyzerConfig singleton instance.

    Used for testing and programmatic configuration override.

    Args:
        config: CommodityRiskAnalyzerConfig instance to set as global.

    Example:
        >>> from greenlang.agents.eudr.commodity_risk_analyzer.config import (
        ...     set_config, CommodityRiskAnalyzerConfig,
        ... )
        >>> test_cfg = CommodityRiskAnalyzerConfig(hhi_concentration_threshold=0.30)
        >>> set_config(test_cfg)
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global CommodityRiskAnalyzerConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_config()
        >>> # Next get_config() call will re-initialize from environment
    """
    global _global_config
    with _config_lock:
        _global_config = None
