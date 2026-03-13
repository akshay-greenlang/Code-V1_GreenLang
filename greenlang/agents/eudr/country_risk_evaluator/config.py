# -*- coding: utf-8 -*-
"""
Country Risk Evaluator Configuration - AGENT-EUDR-016

Centralized configuration for the Country Risk Evaluator Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Country risk scoring: default factor weights (deforestation_rate=30,
  governance=20, enforcement=15, corruption=15, forest_law=10, trend=10),
  score normalization mode, confidence thresholds, data freshness limits,
  EC benchmark override, classification thresholds (low 0-30, standard
  31-65, high 66-100), custom weight bounds (5-50 per factor)
- Commodity analysis: 7 EUDR commodities, correlation threshold,
  seasonal analysis toggle, certification weight, production volume
  weight, supply chain complexity scoring
- Hotspot detection: alert threshold, DBSCAN clustering min points
  and radius km, fire alert correlation toggle, protected area buffer
  km, min deforestation rate alert, indigenous territory buffer km,
  trend analysis window years
- Governance: WGI weight, CPI weight, forest governance weight,
  enforcement weight, data source list (WGI, CPI, FAO, ITTO),
  legal framework scoring toggle, judicial scoring toggle
- Due diligence: simplified threshold (30), standard threshold (60),
  enhanced always above 60, certification credit max (30), audit
  frequency multiplier, cost model ranges (simplified/standard/enhanced),
  time-to-compliance estimation toggle
- Trade flows: min trade volume tonnes, re-export risk threshold,
  transshipment risk countries list, HS code depth, concentration
  risk toggle (HHI), sanction overlay enabled, FTA impact toggle
- Reports: output formats (pdf, json, html), default language (en),
  supported languages (en, fr, de, es, pt), report retention days
  (1825 = 5 years), template directory, max report size MB
- Regulatory updates: monitoring interval hours, EC benchmarking URL,
  grace period days, reminder periods, enforcement tracking toggle
- Database: PostgreSQL connection settings
- Redis: caching settings with TTL and key prefix
- Batch processing: max concurrent, timeout, batch size
- Provenance: enabled toggle, chain algorithm (sha256), genesis hash
- Metrics: Prometheus enabled toggle, prefix gl_eudr_cre_
- Rate limiting: 5 tiers (anonymous, basic, standard, premium, admin)

All settings can be overridden via environment variables with the
``GL_EUDR_CRE_`` prefix (e.g. ``GL_EUDR_CRE_DATABASE_URL``,
``GL_EUDR_CRE_DEFORESTATION_WEIGHT``).

Environment Variable Reference (GL_EUDR_CRE_ prefix):
    GL_EUDR_CRE_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_CRE_REDIS_URL                     - Redis connection URL
    GL_EUDR_CRE_LOG_LEVEL                     - Logging level
    GL_EUDR_CRE_POOL_SIZE                     - Database pool size
    GL_EUDR_CRE_DEFORESTATION_WEIGHT          - Deforestation rate weight (0-100)
    GL_EUDR_CRE_GOVERNANCE_WEIGHT             - Governance index weight (0-100)
    GL_EUDR_CRE_ENFORCEMENT_WEIGHT            - Enforcement score weight (0-100)
    GL_EUDR_CRE_CORRUPTION_WEIGHT             - Corruption index weight (0-100)
    GL_EUDR_CRE_FOREST_LAW_WEIGHT             - Forest law compliance weight (0-100)
    GL_EUDR_CRE_TREND_WEIGHT                  - Historical trend weight (0-100)
    GL_EUDR_CRE_SCORE_NORMALIZATION           - Score normalization mode
    GL_EUDR_CRE_CONFIDENCE_THRESHOLD          - Confidence threshold (0.0-1.0)
    GL_EUDR_CRE_DATA_FRESHNESS_MAX_DAYS       - Max data freshness in days
    GL_EUDR_CRE_EC_BENCHMARK_OVERRIDE         - Enable EC benchmark override
    GL_EUDR_CRE_LOW_RISK_THRESHOLD            - Low risk upper bound (0-100)
    GL_EUDR_CRE_HIGH_RISK_THRESHOLD           - High risk lower bound (0-100)
    GL_EUDR_CRE_MIN_FACTOR_WEIGHT             - Min weight per factor (0-100)
    GL_EUDR_CRE_MAX_FACTOR_WEIGHT             - Max weight per factor (0-100)
    GL_EUDR_CRE_CORRELATION_THRESHOLD         - Commodity correlation threshold
    GL_EUDR_CRE_ENABLE_SEASONAL_ANALYSIS      - Enable seasonal risk analysis
    GL_EUDR_CRE_CERTIFICATION_WEIGHT          - Certification scheme weight
    GL_EUDR_CRE_PRODUCTION_VOLUME_WEIGHT      - Production volume weight
    GL_EUDR_CRE_SUPPLY_CHAIN_COMPLEXITY_MAX   - Max supply chain complexity score
    GL_EUDR_CRE_ALERT_THRESHOLD               - Hotspot alert threshold (0-100)
    GL_EUDR_CRE_CLUSTERING_MIN_POINTS         - DBSCAN min cluster points
    GL_EUDR_CRE_CLUSTERING_RADIUS_KM          - DBSCAN clustering radius km
    GL_EUDR_CRE_ENABLE_FIRE_CORRELATION       - Enable fire-deforestation correlation
    GL_EUDR_CRE_PROTECTED_AREA_BUFFER_KM      - Protected area buffer km
    GL_EUDR_CRE_MIN_DEFORESTATION_RATE_ALERT  - Min deforestation rate for alert
    GL_EUDR_CRE_INDIGENOUS_TERRITORY_BUFFER_KM - Indigenous territory buffer km
    GL_EUDR_CRE_TREND_WINDOW_YEARS            - Trend analysis window years
    GL_EUDR_CRE_WGI_WEIGHT                    - World Bank WGI weight
    GL_EUDR_CRE_CPI_WEIGHT                    - Transparency Intl CPI weight
    GL_EUDR_CRE_FOREST_GOVERNANCE_WEIGHT      - Forest governance framework weight
    GL_EUDR_CRE_GOV_ENFORCEMENT_WEIGHT        - Governance enforcement weight
    GL_EUDR_CRE_ENABLE_LEGAL_FRAMEWORK_SCORING - Enable legal framework scoring
    GL_EUDR_CRE_ENABLE_JUDICIAL_SCORING       - Enable judicial scoring
    GL_EUDR_CRE_SIMPLIFIED_THRESHOLD          - Simplified DD upper bound (0-100)
    GL_EUDR_CRE_ENHANCED_THRESHOLD            - Enhanced DD lower bound (0-100)
    GL_EUDR_CRE_CERTIFICATION_CREDIT_MAX      - Max certification risk credit
    GL_EUDR_CRE_AUDIT_FREQUENCY_MULTIPLIER    - Audit frequency multiplier
    GL_EUDR_CRE_SIMPLIFIED_COST_MIN_EUR       - Simplified DD min cost EUR
    GL_EUDR_CRE_SIMPLIFIED_COST_MAX_EUR       - Simplified DD max cost EUR
    GL_EUDR_CRE_STANDARD_COST_MIN_EUR         - Standard DD min cost EUR
    GL_EUDR_CRE_STANDARD_COST_MAX_EUR         - Standard DD max cost EUR
    GL_EUDR_CRE_ENHANCED_COST_MIN_EUR         - Enhanced DD min cost EUR
    GL_EUDR_CRE_ENHANCED_COST_MAX_EUR         - Enhanced DD max cost EUR
    GL_EUDR_CRE_ENABLE_TIME_TO_COMPLIANCE     - Enable time-to-compliance estimation
    GL_EUDR_CRE_MIN_TRADE_VOLUME_TONNES       - Min trade volume tonnes
    GL_EUDR_CRE_RE_EXPORT_RISK_THRESHOLD      - Re-export risk threshold (0.0-1.0)
    GL_EUDR_CRE_HS_CODE_DEPTH                 - HS code matching depth (6 or 8)
    GL_EUDR_CRE_ENABLE_CONCENTRATION_RISK     - Enable HHI concentration risk
    GL_EUDR_CRE_ENABLE_SANCTION_OVERLAY       - Enable sanction overlay
    GL_EUDR_CRE_ENABLE_FTA_IMPACT             - Enable FTA impact analysis
    GL_EUDR_CRE_DEFAULT_LANGUAGE              - Default report language
    GL_EUDR_CRE_REPORT_RETENTION_DAYS         - Report retention in days
    GL_EUDR_CRE_TEMPLATE_DIR                  - Report template directory
    GL_EUDR_CRE_MAX_REPORT_SIZE_MB            - Max report file size MB
    GL_EUDR_CRE_MONITORING_INTERVAL_HOURS     - EC monitoring interval hours
    GL_EUDR_CRE_EC_BENCHMARKING_URL           - EC benchmarking URL
    GL_EUDR_CRE_GRACE_PERIOD_DAYS             - Regulatory grace period days
    GL_EUDR_CRE_ENABLE_ENFORCEMENT_TRACKING   - Enable enforcement tracking
    GL_EUDR_CRE_REDIS_TTL_S                   - Redis cache TTL seconds
    GL_EUDR_CRE_REDIS_KEY_PREFIX              - Redis key prefix
    GL_EUDR_CRE_BATCH_MAX_SIZE                - Batch processing max size
    GL_EUDR_CRE_BATCH_CONCURRENCY             - Batch concurrency workers
    GL_EUDR_CRE_BATCH_TIMEOUT_S               - Batch timeout seconds
    GL_EUDR_CRE_RETENTION_YEARS               - Data retention years
    GL_EUDR_CRE_ENABLE_PROVENANCE             - Enable provenance tracking
    GL_EUDR_CRE_GENESIS_HASH                  - Genesis hash anchor
    GL_EUDR_CRE_CHAIN_ALGORITHM               - Provenance chain hash algorithm
    GL_EUDR_CRE_ENABLE_METRICS                - Enable Prometheus metrics
    GL_EUDR_CRE_METRICS_PREFIX                - Prometheus metrics prefix
    GL_EUDR_CRE_RATE_LIMIT_ANONYMOUS          - Rate limit anonymous tier
    GL_EUDR_CRE_RATE_LIMIT_BASIC              - Rate limit basic tier
    GL_EUDR_CRE_RATE_LIMIT_STANDARD           - Rate limit standard tier
    GL_EUDR_CRE_RATE_LIMIT_PREMIUM            - Rate limit premium tier
    GL_EUDR_CRE_RATE_LIMIT_ADMIN              - Rate limit admin tier

Example:
    >>> from greenlang.agents.eudr.country_risk_evaluator.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.deforestation_weight, cfg.low_risk_threshold)
    30 30

    >>> # Override for testing
    >>> from greenlang.agents.eudr.country_risk_evaluator.config import (
    ...     set_config, reset_config, CountryRiskEvaluatorConfig,
    ... )
    >>> set_config(CountryRiskEvaluatorConfig(deforestation_weight=40))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
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

_ENV_PREFIX = "GL_EUDR_CRE_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid score normalization modes
# ---------------------------------------------------------------------------

_VALID_NORMALIZATION_MODES = frozenset({
    "minmax",
    "percentile",
    "zscore",
    "raw",
})

# ---------------------------------------------------------------------------
# Valid report output formats
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "csv", "excel"})

# ---------------------------------------------------------------------------
# Valid report languages
# ---------------------------------------------------------------------------

_VALID_REPORT_LANGUAGES = frozenset({
    "en", "fr", "de", "es", "pt",
})

# ---------------------------------------------------------------------------
# Valid chain hash algorithms
# ---------------------------------------------------------------------------

_VALID_CHAIN_ALGORITHMS = frozenset({"sha256", "sha384", "sha512"})

# ---------------------------------------------------------------------------
# Valid HS code depths
# ---------------------------------------------------------------------------

_VALID_HS_CODE_DEPTHS = frozenset({6, 8})

# ---------------------------------------------------------------------------
# Default EUDR commodities (EU 2023/1115 Article 1)
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ---------------------------------------------------------------------------
# Default governance data sources
# ---------------------------------------------------------------------------

_DEFAULT_GOVERNANCE_DATA_SOURCES: List[str] = [
    "wgi",   # World Bank Worldwide Governance Indicators
    "cpi",   # Transparency International Corruption Perceptions Index
    "fao",   # FAO Forest Resources Assessment & governance frameworks
    "itto",  # International Tropical Timber Organization
]

# ---------------------------------------------------------------------------
# Default transshipment risk countries
# ---------------------------------------------------------------------------

_DEFAULT_TRANSSHIPMENT_RISK_COUNTRIES: List[str] = [
    "SG",  # Singapore
    "AE",  # United Arab Emirates
    "HK",  # Hong Kong
    "NL",  # Netherlands
    "BE",  # Belgium
    "MY",  # Malaysia (for palm oil re-export)
    "VN",  # Vietnam (for wood/rubber re-export)
    "TH",  # Thailand (for rubber re-export)
    "TR",  # Turkey
    "IN",  # India
]

# ---------------------------------------------------------------------------
# Default report output formats
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_FORMATS: List[str] = ["pdf", "json", "html"]

# ---------------------------------------------------------------------------
# Default supported report languages
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORTED_LANGUAGES: List[str] = [
    "en", "fr", "de", "es", "pt",
]

# ---------------------------------------------------------------------------
# Default reminder periods for regulatory changes (days before deadline)
# ---------------------------------------------------------------------------

_DEFAULT_REMINDER_PERIODS: List[int] = [90, 60, 30, 7]


# ---------------------------------------------------------------------------
# CountryRiskEvaluatorConfig
# ---------------------------------------------------------------------------


@dataclass
class CountryRiskEvaluatorConfig:
    """Complete configuration for the EUDR Country Risk Evaluator Agent.

    Attributes are grouped by concern: connections, logging, country risk
    scoring, commodity analysis, hotspot detection, governance indices,
    due diligence classification, trade flow analysis, report generation,
    regulatory update tracking, database, Redis caching, batch processing,
    data retention, provenance tracking, metrics, and rate limiting.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_CRE_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            risk assessments, hotspot detections, and audit logs.
        redis_url: Redis connection URL for risk score caching, session
            management, and rate limiting.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        pool_size: PostgreSQL connection pool size.
        deforestation_weight: Weight for deforestation rate factor in
            composite score (0-100, default 30). Part of 6-factor
            weighted composite per EUDR Article 29(2)(a).
        governance_weight: Weight for governance index factor (0-100,
            default 20). Per EUDR Article 29(3).
        enforcement_weight: Weight for enforcement score factor (0-100,
            default 15). Per EUDR Article 29(3).
        corruption_weight: Weight for corruption perception index
            factor (0-100, default 15). Per EUDR Article 29(3).
        forest_law_weight: Weight for forest law compliance factor
            (0-100, default 10). Per EUDR Article 29(3).
        trend_weight: Weight for historical trend factor (0-100,
            default 10). 5-year rolling window.
        score_normalization: Normalization mode for risk scores.
            Options: minmax, percentile, zscore, raw.
        confidence_threshold: Minimum confidence level (0.0-1.0) for
            a risk score to be considered actionable.
        data_freshness_max_days: Maximum age in days for source data
            before confidence is downgraded. < 180 = high,
            180-365 = medium, > 365 = low.
        ec_benchmark_override: When True, EC-published benchmark
            classifications override agent-computed classifications.
        low_risk_threshold: Upper bound for low-risk classification
            (0-100). Scores <= this value are LOW.
        high_risk_threshold: Lower bound for high-risk classification
            (0-100). Scores > this value are HIGH. Everything in
            between is STANDARD.
        min_factor_weight: Minimum allowed weight per factor when
            operators customize weights (0-100, default 5).
        max_factor_weight: Maximum allowed weight per factor (0-100,
            default 50).
        correlation_threshold: Minimum correlation coefficient (0.0-1.0)
            for commodity-deforestation correlation to be flagged.
        enable_seasonal_analysis: Enable seasonal risk variation
            analysis for commodity-country pairs.
        certification_weight: Weight given to certification scheme
            effectiveness in commodity risk scoring (0.0-1.0).
        production_volume_weight: Weight given to production volume
            in commodity risk scoring (0.0-1.0).
        supply_chain_complexity_max: Maximum score for supply chain
            complexity (1-10 scale).
        alert_threshold: Risk score threshold (0-100) above which a
            deforestation hotspot triggers an alert.
        clustering_min_points: Minimum number of deforestation alerts
            required to form a spatial cluster (DBSCAN min_samples).
        clustering_radius_km: Maximum distance in km between alerts
            for DBSCAN clustering (epsilon parameter).
        enable_fire_correlation: Enable fire-deforestation correlation
            analysis using FIRMS/VIIRS data.
        protected_area_buffer_km: Buffer distance in km around
            protected areas for proximity scoring.
        min_deforestation_rate_alert: Minimum annual deforestation
            rate (percentage) to trigger alert.
        indigenous_territory_buffer_km: Buffer distance in km around
            indigenous territories for overlap detection.
        trend_window_years: Number of years for historical trend
            analysis rolling window.
        wgi_weight: Weight for World Bank WGI indicators within
            governance composite (0.0-1.0).
        cpi_weight: Weight for Transparency International CPI within
            governance composite (0.0-1.0).
        forest_governance_weight: Weight for FAO/ITTO forest
            governance framework within governance composite (0.0-1.0).
        gov_enforcement_weight: Weight for enforcement effectiveness
            within governance composite (0.0-1.0).
        governance_data_sources: List of enabled governance data
            sources (wgi, cpi, fao, itto).
        enable_legal_framework_scoring: Enable legal framework
            strength scoring (5-criteria assessment).
        enable_judicial_scoring: Enable judicial independence and
            rule of law scoring.
        simplified_threshold: Upper bound for simplified due diligence
            classification (0-100, default 30). Aligned with Art. 13.
        enhanced_threshold: Lower bound for enhanced due diligence
            classification (0-100, default 60). Above this -> enhanced.
        certification_credit_max: Maximum certification-based risk
            mitigation credit in points (0-30).
        audit_frequency_multiplier: Multiplier applied to base audit
            frequency per risk level.
        simplified_cost_min_eur: Minimum cost in EUR for simplified
            due diligence per shipment.
        simplified_cost_max_eur: Maximum cost in EUR for simplified
            due diligence per shipment.
        standard_cost_min_eur: Minimum cost in EUR for standard
            due diligence per shipment.
        standard_cost_max_eur: Maximum cost in EUR for standard
            due diligence per shipment.
        enhanced_cost_min_eur: Minimum cost in EUR for enhanced
            due diligence per shipment.
        enhanced_cost_max_eur: Maximum cost in EUR for enhanced
            due diligence per shipment.
        enable_time_to_compliance: Enable time-to-compliance
            estimation per due diligence level.
        min_trade_volume_tonnes: Minimum trade volume in tonnes for
            a bilateral flow to be tracked.
        re_export_risk_threshold: Threshold (0.0-1.0) for flagging
            re-export risk (export/production ratio).
        transshipment_risk_countries: List of ISO-2 country codes
            known for commodity transshipment.
        hs_code_depth: HS code matching depth (6 = international
            HS, 8 = EU CN codes).
        enable_concentration_risk: Enable Herfindahl-Hirschman
            Index (HHI) concentration risk calculation.
        enable_sanction_overlay: Enable EU sanction and embargo
            overlay on trade flows.
        enable_fta_impact: Enable free trade agreement impact
            analysis on commodity flows.
        output_formats: List of supported report output formats.
        default_language: Default language for report generation.
        supported_languages: List of supported languages for
            multi-language report generation.
        report_retention_days: Number of days to retain generated
            reports (1825 = 5 years per EUDR Article 31).
        template_dir: Directory path for report Jinja2 templates.
        max_report_size_mb: Maximum report file size in megabytes.
        monitoring_interval_hours: Hours between EC benchmarking
            portal polling checks (default: 24).
        ec_benchmarking_url: URL for EC EUDR benchmarking portal.
        grace_period_days: Default grace period in days for
            regulatory changes.
        reminder_periods: List of days-before-deadline for
            compliance reminder notifications.
        enable_enforcement_tracking: Enable enforcement action
            tracking per country.
        redis_ttl_s: Redis cache TTL in seconds for risk scores.
        redis_key_prefix: Redis key prefix for namespacing.
        batch_max_size: Maximum number of records in a single
            batch processing job.
        batch_concurrency: Maximum concurrent batch processing
            workers.
        batch_timeout_s: Timeout in seconds for a single batch
            job.
        retention_years: Data retention in years per EUDR
            Article 31 (default 5).
        eudr_commodities: List of EUDR-regulated commodity types.
        enable_provenance: Enable SHA-256 provenance chain
            tracking for all risk assessment operations.
        genesis_hash: Genesis anchor string for the provenance
            chain, unique to the Country Risk Evaluator agent.
        chain_algorithm: Hash algorithm for provenance chain
            (sha256, sha384, sha512).
        enable_metrics: Enable Prometheus metrics export under
            the ``gl_eudr_cre_`` prefix.
        metrics_prefix: Prometheus metrics name prefix.
        rate_limit_anonymous: Max requests per minute for
            anonymous tier.
        rate_limit_basic: Max requests per minute for basic tier.
        rate_limit_standard: Max requests per minute for
            standard tier.
        rate_limit_premium: Max requests per minute for
            premium tier.
        rate_limit_admin: Max requests per minute for admin tier.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10

    # -- Country risk scoring settings ---------------------------------------
    # Factor weights (must sum to 100)
    deforestation_weight: int = 30
    governance_weight: int = 20
    enforcement_weight: int = 15
    corruption_weight: int = 15
    forest_law_weight: int = 10
    trend_weight: int = 10
    # Scoring parameters
    score_normalization: str = "minmax"
    confidence_threshold: float = 0.6
    data_freshness_max_days: int = 365
    ec_benchmark_override: bool = True
    # Classification thresholds
    low_risk_threshold: int = 30
    high_risk_threshold: int = 65
    # Custom weight bounds
    min_factor_weight: int = 5
    max_factor_weight: int = 50

    # -- Commodity analysis settings -----------------------------------------
    correlation_threshold: float = 0.5
    enable_seasonal_analysis: bool = True
    certification_weight: float = 0.3
    production_volume_weight: float = 0.2
    supply_chain_complexity_max: int = 10

    # -- Hotspot detection settings ------------------------------------------
    alert_threshold: int = 70
    clustering_min_points: int = 10
    clustering_radius_km: float = 5.0
    enable_fire_correlation: bool = True
    protected_area_buffer_km: float = 10.0
    min_deforestation_rate_alert: float = 0.5
    indigenous_territory_buffer_km: float = 10.0
    trend_window_years: int = 5

    # -- Governance settings -------------------------------------------------
    wgi_weight: float = 0.30
    cpi_weight: float = 0.30
    forest_governance_weight: float = 0.25
    gov_enforcement_weight: float = 0.15
    governance_data_sources: List[str] = field(
        default_factory=lambda: list(_DEFAULT_GOVERNANCE_DATA_SOURCES)
    )
    enable_legal_framework_scoring: bool = True
    enable_judicial_scoring: bool = True

    # -- Due diligence classification settings -------------------------------
    simplified_threshold: int = 30
    enhanced_threshold: int = 60
    certification_credit_max: int = 30
    audit_frequency_multiplier: float = 1.0
    # Cost model ranges per DD level (EUR per shipment)
    simplified_cost_min_eur: int = 200
    simplified_cost_max_eur: int = 500
    standard_cost_min_eur: int = 1000
    standard_cost_max_eur: int = 3000
    enhanced_cost_min_eur: int = 5000
    enhanced_cost_max_eur: int = 15000
    enable_time_to_compliance: bool = True

    # -- Trade flow settings -------------------------------------------------
    min_trade_volume_tonnes: float = 10.0
    re_export_risk_threshold: float = 0.7
    transshipment_risk_countries: List[str] = field(
        default_factory=lambda: list(_DEFAULT_TRANSSHIPMENT_RISK_COUNTRIES)
    )
    hs_code_depth: int = 6
    enable_concentration_risk: bool = True
    enable_sanction_overlay: bool = True
    enable_fta_impact: bool = True

    # -- Report generation settings ------------------------------------------
    output_formats: List[str] = field(
        default_factory=lambda: list(_DEFAULT_OUTPUT_FORMATS)
    )
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: list(_DEFAULT_SUPPORTED_LANGUAGES)
    )
    report_retention_days: int = 1825
    template_dir: str = "/data/greenlang/cre/templates"
    max_report_size_mb: int = 50

    # -- Regulatory update settings ------------------------------------------
    monitoring_interval_hours: int = 24
    ec_benchmarking_url: str = (
        "https://environment.ec.europa.eu/topics/"
        "forests/deforestation/regulation/benchmarking_en"
    )
    grace_period_days: int = 90
    reminder_periods: List[int] = field(
        default_factory=lambda: list(_DEFAULT_REMINDER_PERIODS)
    )
    enable_enforcement_tracking: bool = True

    # -- Redis caching settings ----------------------------------------------
    redis_ttl_s: int = 3600
    redis_key_prefix: str = "gl_eudr_cre"

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 500
    batch_concurrency: int = 8
    batch_timeout_s: int = 600

    # -- Data retention (EUDR Article 31) ------------------------------------
    retention_years: int = 5

    # -- EUDR commodities ----------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-CRE-016-COUNTRY-RISK-EVALUATOR-GENESIS"
    chain_algorithm: str = "sha256"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_cre_"

    # -- Rate limiting (5 tiers) ---------------------------------------------
    rate_limit_anonymous: int = 10
    rate_limit_basic: int = 50
    rate_limit_standard: int = 200
    rate_limit_premium: int = 1000
    rate_limit_admin: int = 5000

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, threshold ordering validation, weight-sum
        validation, and normalization. Collects all errors before
        raising a single ValueError with all violations listed.

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

        # -- Performance tuning ----------------------------------------------
        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")

        # -- Country risk scoring: factor weights ----------------------------
        weights = [
            ("deforestation_weight", self.deforestation_weight),
            ("governance_weight", self.governance_weight),
            ("enforcement_weight", self.enforcement_weight),
            ("corruption_weight", self.corruption_weight),
            ("forest_law_weight", self.forest_law_weight),
            ("trend_weight", self.trend_weight),
        ]
        for name, val in weights:
            if not (0 <= val <= 100):
                errors.append(
                    f"{name} must be in [0, 100], got {val}"
                )

        weight_sum = sum(v for _, v in weights)
        if weight_sum != 100:
            errors.append(
                f"Factor weights must sum to 100, "
                f"got {weight_sum} "
                f"(deforestation={self.deforestation_weight}, "
                f"governance={self.governance_weight}, "
                f"enforcement={self.enforcement_weight}, "
                f"corruption={self.corruption_weight}, "
                f"forest_law={self.forest_law_weight}, "
                f"trend={self.trend_weight})"
            )

        # -- Score normalization ---------------------------------------------
        normalised_norm = self.score_normalization.lower().strip()
        if normalised_norm not in _VALID_NORMALIZATION_MODES:
            errors.append(
                f"score_normalization must be one of "
                f"{sorted(_VALID_NORMALIZATION_MODES)}, "
                f"got '{self.score_normalization}'"
            )
        else:
            self.score_normalization = normalised_norm

        # -- Confidence threshold --------------------------------------------
        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append(
                f"confidence_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_threshold}"
            )

        # -- Data freshness --------------------------------------------------
        if self.data_freshness_max_days < 1:
            errors.append(
                f"data_freshness_max_days must be >= 1, "
                f"got {self.data_freshness_max_days}"
            )

        # -- Classification thresholds ---------------------------------------
        if not (0 <= self.low_risk_threshold <= 100):
            errors.append(
                f"low_risk_threshold must be in [0, 100], "
                f"got {self.low_risk_threshold}"
            )
        if not (0 <= self.high_risk_threshold <= 100):
            errors.append(
                f"high_risk_threshold must be in [0, 100], "
                f"got {self.high_risk_threshold}"
            )
        if self.low_risk_threshold >= self.high_risk_threshold:
            errors.append(
                f"low_risk_threshold ({self.low_risk_threshold}) must be < "
                f"high_risk_threshold ({self.high_risk_threshold})"
            )

        # -- Custom weight bounds --------------------------------------------
        if not (0 <= self.min_factor_weight <= 100):
            errors.append(
                f"min_factor_weight must be in [0, 100], "
                f"got {self.min_factor_weight}"
            )
        if not (0 <= self.max_factor_weight <= 100):
            errors.append(
                f"max_factor_weight must be in [0, 100], "
                f"got {self.max_factor_weight}"
            )
        if self.min_factor_weight >= self.max_factor_weight:
            errors.append(
                f"min_factor_weight ({self.min_factor_weight}) must be < "
                f"max_factor_weight ({self.max_factor_weight})"
            )

        # -- Commodity analysis settings -------------------------------------
        if not (0.0 <= self.correlation_threshold <= 1.0):
            errors.append(
                f"correlation_threshold must be in [0.0, 1.0], "
                f"got {self.correlation_threshold}"
            )
        if not (0.0 <= self.certification_weight <= 1.0):
            errors.append(
                f"certification_weight must be in [0.0, 1.0], "
                f"got {self.certification_weight}"
            )
        if not (0.0 <= self.production_volume_weight <= 1.0):
            errors.append(
                f"production_volume_weight must be in [0.0, 1.0], "
                f"got {self.production_volume_weight}"
            )
        if not (1 <= self.supply_chain_complexity_max <= 20):
            errors.append(
                f"supply_chain_complexity_max must be in [1, 20], "
                f"got {self.supply_chain_complexity_max}"
            )

        # -- Hotspot detection settings --------------------------------------
        if not (0 <= self.alert_threshold <= 100):
            errors.append(
                f"alert_threshold must be in [0, 100], "
                f"got {self.alert_threshold}"
            )
        if self.clustering_min_points < 1:
            errors.append(
                f"clustering_min_points must be >= 1, "
                f"got {self.clustering_min_points}"
            )
        if self.clustering_min_points > 1000:
            errors.append(
                f"clustering_min_points must be <= 1000, "
                f"got {self.clustering_min_points}"
            )
        if self.clustering_radius_km <= 0:
            errors.append(
                f"clustering_radius_km must be > 0, "
                f"got {self.clustering_radius_km}"
            )
        if self.clustering_radius_km > 500:
            errors.append(
                f"clustering_radius_km must be <= 500, "
                f"got {self.clustering_radius_km}"
            )
        if self.protected_area_buffer_km < 0:
            errors.append(
                f"protected_area_buffer_km must be >= 0, "
                f"got {self.protected_area_buffer_km}"
            )
        if self.protected_area_buffer_km > 100:
            errors.append(
                f"protected_area_buffer_km must be <= 100, "
                f"got {self.protected_area_buffer_km}"
            )
        if self.min_deforestation_rate_alert < 0:
            errors.append(
                f"min_deforestation_rate_alert must be >= 0, "
                f"got {self.min_deforestation_rate_alert}"
            )
        if self.indigenous_territory_buffer_km < 0:
            errors.append(
                f"indigenous_territory_buffer_km must be >= 0, "
                f"got {self.indigenous_territory_buffer_km}"
            )
        if not (1 <= self.trend_window_years <= 30):
            errors.append(
                f"trend_window_years must be in [1, 30], "
                f"got {self.trend_window_years}"
            )

        # -- Governance settings ---------------------------------------------
        gov_weights = [
            ("wgi_weight", self.wgi_weight),
            ("cpi_weight", self.cpi_weight),
            ("forest_governance_weight", self.forest_governance_weight),
            ("gov_enforcement_weight", self.gov_enforcement_weight),
        ]
        for name, val in gov_weights:
            if not (0.0 <= val <= 1.0):
                errors.append(
                    f"{name} must be in [0.0, 1.0], got {val}"
                )

        gov_weight_sum = sum(v for _, v in gov_weights)
        if abs(gov_weight_sum - 1.0) > 0.01:
            errors.append(
                f"Governance weights must sum to 1.0, "
                f"got {gov_weight_sum:.4f} "
                f"(wgi={self.wgi_weight}, cpi={self.cpi_weight}, "
                f"forest_gov={self.forest_governance_weight}, "
                f"enforcement={self.gov_enforcement_weight})"
            )

        if not self.governance_data_sources:
            errors.append("governance_data_sources must not be empty")

        # -- Due diligence settings ------------------------------------------
        if not (0 <= self.simplified_threshold <= 100):
            errors.append(
                f"simplified_threshold must be in [0, 100], "
                f"got {self.simplified_threshold}"
            )
        if not (0 <= self.enhanced_threshold <= 100):
            errors.append(
                f"enhanced_threshold must be in [0, 100], "
                f"got {self.enhanced_threshold}"
            )
        if self.simplified_threshold >= self.enhanced_threshold:
            errors.append(
                f"simplified_threshold ({self.simplified_threshold}) "
                f"must be < enhanced_threshold ({self.enhanced_threshold})"
            )
        if not (0 <= self.certification_credit_max <= 50):
            errors.append(
                f"certification_credit_max must be in [0, 50], "
                f"got {self.certification_credit_max}"
            )
        if self.audit_frequency_multiplier <= 0:
            errors.append(
                f"audit_frequency_multiplier must be > 0, "
                f"got {self.audit_frequency_multiplier}"
            )
        # Cost model validation
        if self.simplified_cost_min_eur < 0:
            errors.append(
                f"simplified_cost_min_eur must be >= 0, "
                f"got {self.simplified_cost_min_eur}"
            )
        if self.simplified_cost_max_eur < self.simplified_cost_min_eur:
            errors.append(
                f"simplified_cost_max_eur ({self.simplified_cost_max_eur}) "
                f"must be >= simplified_cost_min_eur "
                f"({self.simplified_cost_min_eur})"
            )
        if self.standard_cost_min_eur < 0:
            errors.append(
                f"standard_cost_min_eur must be >= 0, "
                f"got {self.standard_cost_min_eur}"
            )
        if self.standard_cost_max_eur < self.standard_cost_min_eur:
            errors.append(
                f"standard_cost_max_eur ({self.standard_cost_max_eur}) "
                f"must be >= standard_cost_min_eur "
                f"({self.standard_cost_min_eur})"
            )
        if self.enhanced_cost_min_eur < 0:
            errors.append(
                f"enhanced_cost_min_eur must be >= 0, "
                f"got {self.enhanced_cost_min_eur}"
            )
        if self.enhanced_cost_max_eur < self.enhanced_cost_min_eur:
            errors.append(
                f"enhanced_cost_max_eur ({self.enhanced_cost_max_eur}) "
                f"must be >= enhanced_cost_min_eur "
                f"({self.enhanced_cost_min_eur})"
            )

        # -- Trade flow settings ---------------------------------------------
        if self.min_trade_volume_tonnes < 0:
            errors.append(
                f"min_trade_volume_tonnes must be >= 0, "
                f"got {self.min_trade_volume_tonnes}"
            )
        if not (0.0 <= self.re_export_risk_threshold <= 1.0):
            errors.append(
                f"re_export_risk_threshold must be in [0.0, 1.0], "
                f"got {self.re_export_risk_threshold}"
            )
        if not self.transshipment_risk_countries:
            errors.append("transshipment_risk_countries must not be empty")
        if self.hs_code_depth not in _VALID_HS_CODE_DEPTHS:
            errors.append(
                f"hs_code_depth must be one of "
                f"{sorted(_VALID_HS_CODE_DEPTHS)}, "
                f"got {self.hs_code_depth}"
            )

        # -- Report settings -------------------------------------------------
        if not self.output_formats:
            errors.append("output_formats must not be empty")
        for fmt in self.output_formats:
            if fmt not in _VALID_OUTPUT_FORMATS:
                errors.append(
                    f"output_formats contains invalid format '{fmt}'; "
                    f"must be one of {sorted(_VALID_OUTPUT_FORMATS)}"
                )

        normalised_lang = self.default_language.lower().strip()
        if normalised_lang not in _VALID_REPORT_LANGUAGES:
            errors.append(
                f"default_language must be one of "
                f"{sorted(_VALID_REPORT_LANGUAGES)}, "
                f"got '{self.default_language}'"
            )
        else:
            self.default_language = normalised_lang

        if not self.supported_languages:
            errors.append("supported_languages must not be empty")

        if self.report_retention_days < 1:
            errors.append(
                f"report_retention_days must be >= 1, "
                f"got {self.report_retention_days}"
            )

        if self.max_report_size_mb < 1:
            errors.append(
                f"max_report_size_mb must be >= 1, "
                f"got {self.max_report_size_mb}"
            )

        # -- Regulatory update settings --------------------------------------
        if self.monitoring_interval_hours < 1:
            errors.append(
                f"monitoring_interval_hours must be >= 1, "
                f"got {self.monitoring_interval_hours}"
            )
        if self.monitoring_interval_hours > 168:
            errors.append(
                f"monitoring_interval_hours must be <= 168 (1 week), "
                f"got {self.monitoring_interval_hours}"
            )
        if not self.ec_benchmarking_url:
            errors.append("ec_benchmarking_url must not be empty")
        if self.grace_period_days < 0:
            errors.append(
                f"grace_period_days must be >= 0, "
                f"got {self.grace_period_days}"
            )
        if not self.reminder_periods:
            errors.append("reminder_periods must not be empty")

        # -- Redis caching settings ------------------------------------------
        if self.redis_ttl_s < 1:
            errors.append(
                f"redis_ttl_s must be >= 1, got {self.redis_ttl_s}"
            )
        if not self.redis_key_prefix:
            errors.append("redis_key_prefix must not be empty")

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
                f"batch_timeout_s must be >= 1, "
                f"got {self.batch_timeout_s}"
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

        normalised_chain = self.chain_algorithm.lower().strip()
        if normalised_chain not in _VALID_CHAIN_ALGORITHMS:
            errors.append(
                f"chain_algorithm must be one of "
                f"{sorted(_VALID_CHAIN_ALGORITHMS)}, "
                f"got '{self.chain_algorithm}'"
            )
        else:
            self.chain_algorithm = normalised_chain

        # -- Metrics ---------------------------------------------------------
        if not self.metrics_prefix:
            errors.append("metrics_prefix must not be empty")

        # -- Rate limiting ---------------------------------------------------
        rate_limits = [
            ("rate_limit_anonymous", self.rate_limit_anonymous),
            ("rate_limit_basic", self.rate_limit_basic),
            ("rate_limit_standard", self.rate_limit_standard),
            ("rate_limit_premium", self.rate_limit_premium),
            ("rate_limit_admin", self.rate_limit_admin),
        ]
        for name, val in rate_limits:
            if val <= 0:
                errors.append(f"{name} must be > 0, got {val}")

        # Verify rate limit tier ordering
        if (
            self.rate_limit_anonymous > 0
            and self.rate_limit_basic > 0
            and self.rate_limit_standard > 0
            and self.rate_limit_premium > 0
            and self.rate_limit_admin > 0
        ):
            if not (
                self.rate_limit_anonymous
                <= self.rate_limit_basic
                <= self.rate_limit_standard
                <= self.rate_limit_premium
                <= self.rate_limit_admin
            ):
                errors.append(
                    "Rate limit tiers must be in non-decreasing order: "
                    f"anonymous={self.rate_limit_anonymous}, "
                    f"basic={self.rate_limit_basic}, "
                    f"standard={self.rate_limit_standard}, "
                    f"premium={self.rate_limit_premium}, "
                    f"admin={self.rate_limit_admin}"
                )

        if errors:
            raise ValueError(
                "CountryRiskEvaluatorConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "CountryRiskEvaluatorConfig validated successfully: "
            "weights=(%d/%d/%d/%d/%d/%d), norm=%s, "
            "thresholds=low=%d/high=%d, "
            "confidence=%.2f, freshness=%dd, "
            "ec_override=%s, "
            "hotspot_alert=%d, cluster=%d/%.1fkm, "
            "gov_wts=(%.2f/%.2f/%.2f/%.2f), "
            "dd_thresholds=simplified=%d/enhanced=%d, "
            "cert_credit=%d, "
            "trade_vol=%.1ft, re_export=%.2f, hs=%d, "
            "reports=%s, lang=%s, retention=%dd, "
            "monitoring=%dh, grace=%dd, "
            "batch=%d/%d/%ds, retention=%dy, "
            "provenance=%s, metrics=%s",
            self.deforestation_weight,
            self.governance_weight,
            self.enforcement_weight,
            self.corruption_weight,
            self.forest_law_weight,
            self.trend_weight,
            self.score_normalization,
            self.low_risk_threshold,
            self.high_risk_threshold,
            self.confidence_threshold,
            self.data_freshness_max_days,
            self.ec_benchmark_override,
            self.alert_threshold,
            self.clustering_min_points,
            self.clustering_radius_km,
            self.wgi_weight,
            self.cpi_weight,
            self.forest_governance_weight,
            self.gov_enforcement_weight,
            self.simplified_threshold,
            self.enhanced_threshold,
            self.certification_credit_max,
            self.min_trade_volume_tonnes,
            self.re_export_risk_threshold,
            self.hs_code_depth,
            self.output_formats,
            self.default_language,
            self.report_retention_days,
            self.monitoring_interval_hours,
            self.grace_period_days,
            self.batch_max_size,
            self.batch_concurrency,
            self.batch_timeout_s,
            self.retention_years,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> CountryRiskEvaluatorConfig:
        """Build a CountryRiskEvaluatorConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_CRE_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated CountryRiskEvaluatorConfig instance, validated via
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
                    "Invalid float for %s%s=%r, using default %s",
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
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            # Country risk scoring
            deforestation_weight=_int(
                "DEFORESTATION_WEIGHT", cls.deforestation_weight,
            ),
            governance_weight=_int(
                "GOVERNANCE_WEIGHT", cls.governance_weight,
            ),
            enforcement_weight=_int(
                "ENFORCEMENT_WEIGHT", cls.enforcement_weight,
            ),
            corruption_weight=_int(
                "CORRUPTION_WEIGHT", cls.corruption_weight,
            ),
            forest_law_weight=_int(
                "FOREST_LAW_WEIGHT", cls.forest_law_weight,
            ),
            trend_weight=_int(
                "TREND_WEIGHT", cls.trend_weight,
            ),
            score_normalization=_str(
                "SCORE_NORMALIZATION", cls.score_normalization,
            ),
            confidence_threshold=_float(
                "CONFIDENCE_THRESHOLD", cls.confidence_threshold,
            ),
            data_freshness_max_days=_int(
                "DATA_FRESHNESS_MAX_DAYS",
                cls.data_freshness_max_days,
            ),
            ec_benchmark_override=_bool(
                "EC_BENCHMARK_OVERRIDE",
                cls.ec_benchmark_override,
            ),
            low_risk_threshold=_int(
                "LOW_RISK_THRESHOLD", cls.low_risk_threshold,
            ),
            high_risk_threshold=_int(
                "HIGH_RISK_THRESHOLD", cls.high_risk_threshold,
            ),
            min_factor_weight=_int(
                "MIN_FACTOR_WEIGHT", cls.min_factor_weight,
            ),
            max_factor_weight=_int(
                "MAX_FACTOR_WEIGHT", cls.max_factor_weight,
            ),
            # Commodity analysis
            correlation_threshold=_float(
                "CORRELATION_THRESHOLD",
                cls.correlation_threshold,
            ),
            enable_seasonal_analysis=_bool(
                "ENABLE_SEASONAL_ANALYSIS",
                cls.enable_seasonal_analysis,
            ),
            certification_weight=_float(
                "CERTIFICATION_WEIGHT", cls.certification_weight,
            ),
            production_volume_weight=_float(
                "PRODUCTION_VOLUME_WEIGHT",
                cls.production_volume_weight,
            ),
            supply_chain_complexity_max=_int(
                "SUPPLY_CHAIN_COMPLEXITY_MAX",
                cls.supply_chain_complexity_max,
            ),
            # Hotspot detection
            alert_threshold=_int(
                "ALERT_THRESHOLD", cls.alert_threshold,
            ),
            clustering_min_points=_int(
                "CLUSTERING_MIN_POINTS", cls.clustering_min_points,
            ),
            clustering_radius_km=_float(
                "CLUSTERING_RADIUS_KM", cls.clustering_radius_km,
            ),
            enable_fire_correlation=_bool(
                "ENABLE_FIRE_CORRELATION",
                cls.enable_fire_correlation,
            ),
            protected_area_buffer_km=_float(
                "PROTECTED_AREA_BUFFER_KM",
                cls.protected_area_buffer_km,
            ),
            min_deforestation_rate_alert=_float(
                "MIN_DEFORESTATION_RATE_ALERT",
                cls.min_deforestation_rate_alert,
            ),
            indigenous_territory_buffer_km=_float(
                "INDIGENOUS_TERRITORY_BUFFER_KM",
                cls.indigenous_territory_buffer_km,
            ),
            trend_window_years=_int(
                "TREND_WINDOW_YEARS", cls.trend_window_years,
            ),
            # Governance
            wgi_weight=_float("WGI_WEIGHT", cls.wgi_weight),
            cpi_weight=_float("CPI_WEIGHT", cls.cpi_weight),
            forest_governance_weight=_float(
                "FOREST_GOVERNANCE_WEIGHT",
                cls.forest_governance_weight,
            ),
            gov_enforcement_weight=_float(
                "GOV_ENFORCEMENT_WEIGHT",
                cls.gov_enforcement_weight,
            ),
            enable_legal_framework_scoring=_bool(
                "ENABLE_LEGAL_FRAMEWORK_SCORING",
                cls.enable_legal_framework_scoring,
            ),
            enable_judicial_scoring=_bool(
                "ENABLE_JUDICIAL_SCORING",
                cls.enable_judicial_scoring,
            ),
            # Due diligence
            simplified_threshold=_int(
                "SIMPLIFIED_THRESHOLD", cls.simplified_threshold,
            ),
            enhanced_threshold=_int(
                "ENHANCED_THRESHOLD", cls.enhanced_threshold,
            ),
            certification_credit_max=_int(
                "CERTIFICATION_CREDIT_MAX",
                cls.certification_credit_max,
            ),
            audit_frequency_multiplier=_float(
                "AUDIT_FREQUENCY_MULTIPLIER",
                cls.audit_frequency_multiplier,
            ),
            simplified_cost_min_eur=_int(
                "SIMPLIFIED_COST_MIN_EUR",
                cls.simplified_cost_min_eur,
            ),
            simplified_cost_max_eur=_int(
                "SIMPLIFIED_COST_MAX_EUR",
                cls.simplified_cost_max_eur,
            ),
            standard_cost_min_eur=_int(
                "STANDARD_COST_MIN_EUR",
                cls.standard_cost_min_eur,
            ),
            standard_cost_max_eur=_int(
                "STANDARD_COST_MAX_EUR",
                cls.standard_cost_max_eur,
            ),
            enhanced_cost_min_eur=_int(
                "ENHANCED_COST_MIN_EUR",
                cls.enhanced_cost_min_eur,
            ),
            enhanced_cost_max_eur=_int(
                "ENHANCED_COST_MAX_EUR",
                cls.enhanced_cost_max_eur,
            ),
            enable_time_to_compliance=_bool(
                "ENABLE_TIME_TO_COMPLIANCE",
                cls.enable_time_to_compliance,
            ),
            # Trade flows
            min_trade_volume_tonnes=_float(
                "MIN_TRADE_VOLUME_TONNES",
                cls.min_trade_volume_tonnes,
            ),
            re_export_risk_threshold=_float(
                "RE_EXPORT_RISK_THRESHOLD",
                cls.re_export_risk_threshold,
            ),
            hs_code_depth=_int(
                "HS_CODE_DEPTH", cls.hs_code_depth,
            ),
            enable_concentration_risk=_bool(
                "ENABLE_CONCENTRATION_RISK",
                cls.enable_concentration_risk,
            ),
            enable_sanction_overlay=_bool(
                "ENABLE_SANCTION_OVERLAY",
                cls.enable_sanction_overlay,
            ),
            enable_fta_impact=_bool(
                "ENABLE_FTA_IMPACT", cls.enable_fta_impact,
            ),
            # Reports
            default_language=_str(
                "DEFAULT_LANGUAGE", cls.default_language,
            ),
            report_retention_days=_int(
                "REPORT_RETENTION_DAYS",
                cls.report_retention_days,
            ),
            template_dir=_str(
                "TEMPLATE_DIR", cls.template_dir,
            ),
            max_report_size_mb=_int(
                "MAX_REPORT_SIZE_MB", cls.max_report_size_mb,
            ),
            # Regulatory updates
            monitoring_interval_hours=_int(
                "MONITORING_INTERVAL_HOURS",
                cls.monitoring_interval_hours,
            ),
            ec_benchmarking_url=_str(
                "EC_BENCHMARKING_URL",
                cls.ec_benchmarking_url,
            ),
            grace_period_days=_int(
                "GRACE_PERIOD_DAYS", cls.grace_period_days,
            ),
            enable_enforcement_tracking=_bool(
                "ENABLE_ENFORCEMENT_TRACKING",
                cls.enable_enforcement_tracking,
            ),
            # Redis caching
            redis_ttl_s=_int("REDIS_TTL_S", cls.redis_ttl_s),
            redis_key_prefix=_str(
                "REDIS_KEY_PREFIX", cls.redis_key_prefix,
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
            # Data retention
            retention_years=_int(
                "RETENTION_YEARS", cls.retention_years,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            chain_algorithm=_str(
                "CHAIN_ALGORITHM", cls.chain_algorithm,
            ),
            # Metrics
            enable_metrics=_bool(
                "ENABLE_METRICS", cls.enable_metrics,
            ),
            metrics_prefix=_str(
                "METRICS_PREFIX", cls.metrics_prefix,
            ),
            # Rate limiting
            rate_limit_anonymous=_int(
                "RATE_LIMIT_ANONYMOUS",
                cls.rate_limit_anonymous,
            ),
            rate_limit_basic=_int(
                "RATE_LIMIT_BASIC", cls.rate_limit_basic,
            ),
            rate_limit_standard=_int(
                "RATE_LIMIT_STANDARD", cls.rate_limit_standard,
            ),
            rate_limit_premium=_int(
                "RATE_LIMIT_PREMIUM", cls.rate_limit_premium,
            ),
            rate_limit_admin=_int(
                "RATE_LIMIT_ADMIN", cls.rate_limit_admin,
            ),
        )

        logger.info(
            "CountryRiskEvaluatorConfig loaded: "
            "weights=(%d/%d/%d/%d/%d/%d), norm=%s, "
            "thresholds=low=%d/high=%d, "
            "confidence=%.2f, freshness=%dd, "
            "ec_override=%s, "
            "hotspot_alert=%d, cluster=%d/%.1fkm, fire=%s, "
            "gov_wts=(%.2f/%.2f/%.2f/%.2f), "
            "dd=(simplified=%d/enhanced=%d), cert_credit=%d, "
            "trade_vol=%.1ft, re_export=%.2f, hs=%d, "
            "reports=%s, lang=%s, retention=%dd, "
            "monitoring=%dh, grace=%dd, "
            "batch=%d/%d/%ds, retention=%dy, pool=%d, "
            "provenance=%s, metrics=%s",
            config.deforestation_weight,
            config.governance_weight,
            config.enforcement_weight,
            config.corruption_weight,
            config.forest_law_weight,
            config.trend_weight,
            config.score_normalization,
            config.low_risk_threshold,
            config.high_risk_threshold,
            config.confidence_threshold,
            config.data_freshness_max_days,
            config.ec_benchmark_override,
            config.alert_threshold,
            config.clustering_min_points,
            config.clustering_radius_km,
            config.enable_fire_correlation,
            config.wgi_weight,
            config.cpi_weight,
            config.forest_governance_weight,
            config.gov_enforcement_weight,
            config.simplified_threshold,
            config.enhanced_threshold,
            config.certification_credit_max,
            config.min_trade_volume_tonnes,
            config.re_export_risk_threshold,
            config.hs_code_depth,
            config.output_formats,
            config.default_language,
            config.report_retention_days,
            config.monitoring_interval_hours,
            config.grace_period_days,
            config.batch_max_size,
            config.batch_concurrency,
            config.batch_timeout_s,
            config.retention_years,
            config.pool_size,
            config.enable_provenance,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def country_risk_scoring_settings(self) -> Dict[str, Any]:
        """Return country risk scoring settings as a dictionary.

        Returns:
            Dictionary with all country risk scoring configuration keys.
        """
        return {
            "deforestation_weight": self.deforestation_weight,
            "governance_weight": self.governance_weight,
            "enforcement_weight": self.enforcement_weight,
            "corruption_weight": self.corruption_weight,
            "forest_law_weight": self.forest_law_weight,
            "trend_weight": self.trend_weight,
            "score_normalization": self.score_normalization,
            "confidence_threshold": self.confidence_threshold,
            "data_freshness_max_days": self.data_freshness_max_days,
            "ec_benchmark_override": self.ec_benchmark_override,
            "low_risk_threshold": self.low_risk_threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "min_factor_weight": self.min_factor_weight,
            "max_factor_weight": self.max_factor_weight,
        }

    @property
    def commodity_analysis_settings(self) -> Dict[str, Any]:
        """Return commodity analysis settings as a dictionary.

        Returns:
            Dictionary with commodity analysis configuration keys.
        """
        return {
            "correlation_threshold": self.correlation_threshold,
            "enable_seasonal_analysis": self.enable_seasonal_analysis,
            "certification_weight": self.certification_weight,
            "production_volume_weight": self.production_volume_weight,
            "supply_chain_complexity_max": (
                self.supply_chain_complexity_max
            ),
        }

    @property
    def hotspot_detection_settings(self) -> Dict[str, Any]:
        """Return hotspot detection settings as a dictionary.

        Returns:
            Dictionary with hotspot detection configuration keys.
        """
        return {
            "alert_threshold": self.alert_threshold,
            "clustering_min_points": self.clustering_min_points,
            "clustering_radius_km": self.clustering_radius_km,
            "enable_fire_correlation": self.enable_fire_correlation,
            "protected_area_buffer_km": self.protected_area_buffer_km,
            "min_deforestation_rate_alert": (
                self.min_deforestation_rate_alert
            ),
            "indigenous_territory_buffer_km": (
                self.indigenous_territory_buffer_km
            ),
            "trend_window_years": self.trend_window_years,
        }

    @property
    def governance_settings(self) -> Dict[str, Any]:
        """Return governance index settings as a dictionary.

        Returns:
            Dictionary with governance index configuration keys.
        """
        return {
            "wgi_weight": self.wgi_weight,
            "cpi_weight": self.cpi_weight,
            "forest_governance_weight": self.forest_governance_weight,
            "gov_enforcement_weight": self.gov_enforcement_weight,
            "governance_data_sources": list(
                self.governance_data_sources
            ),
            "enable_legal_framework_scoring": (
                self.enable_legal_framework_scoring
            ),
            "enable_judicial_scoring": self.enable_judicial_scoring,
        }

    @property
    def due_diligence_settings(self) -> Dict[str, Any]:
        """Return due diligence classification settings as a dictionary.

        Returns:
            Dictionary with due diligence configuration keys.
        """
        return {
            "simplified_threshold": self.simplified_threshold,
            "enhanced_threshold": self.enhanced_threshold,
            "certification_credit_max": self.certification_credit_max,
            "audit_frequency_multiplier": (
                self.audit_frequency_multiplier
            ),
            "simplified_cost_min_eur": self.simplified_cost_min_eur,
            "simplified_cost_max_eur": self.simplified_cost_max_eur,
            "standard_cost_min_eur": self.standard_cost_min_eur,
            "standard_cost_max_eur": self.standard_cost_max_eur,
            "enhanced_cost_min_eur": self.enhanced_cost_min_eur,
            "enhanced_cost_max_eur": self.enhanced_cost_max_eur,
            "enable_time_to_compliance": (
                self.enable_time_to_compliance
            ),
        }

    @property
    def trade_flow_settings(self) -> Dict[str, Any]:
        """Return trade flow analysis settings as a dictionary.

        Returns:
            Dictionary with trade flow configuration keys.
        """
        return {
            "min_trade_volume_tonnes": self.min_trade_volume_tonnes,
            "re_export_risk_threshold": self.re_export_risk_threshold,
            "transshipment_risk_countries": list(
                self.transshipment_risk_countries
            ),
            "hs_code_depth": self.hs_code_depth,
            "enable_concentration_risk": (
                self.enable_concentration_risk
            ),
            "enable_sanction_overlay": self.enable_sanction_overlay,
            "enable_fta_impact": self.enable_fta_impact,
        }

    @property
    def report_settings(self) -> Dict[str, Any]:
        """Return report generation settings as a dictionary.

        Returns:
            Dictionary with report generation configuration keys.
        """
        return {
            "output_formats": list(self.output_formats),
            "default_language": self.default_language,
            "supported_languages": list(self.supported_languages),
            "report_retention_days": self.report_retention_days,
            "template_dir": self.template_dir,
            "max_report_size_mb": self.max_report_size_mb,
        }

    @property
    def regulatory_update_settings(self) -> Dict[str, Any]:
        """Return regulatory update tracking settings as a dictionary.

        Returns:
            Dictionary with regulatory update configuration keys.
        """
        return {
            "monitoring_interval_hours": self.monitoring_interval_hours,
            "ec_benchmarking_url": self.ec_benchmarking_url,
            "grace_period_days": self.grace_period_days,
            "reminder_periods": list(self.reminder_periods),
            "enable_enforcement_tracking": (
                self.enable_enforcement_tracking
            ),
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
            # Performance tuning
            "pool_size": self.pool_size,
            # Country risk scoring
            "deforestation_weight": self.deforestation_weight,
            "governance_weight": self.governance_weight,
            "enforcement_weight": self.enforcement_weight,
            "corruption_weight": self.corruption_weight,
            "forest_law_weight": self.forest_law_weight,
            "trend_weight": self.trend_weight,
            "score_normalization": self.score_normalization,
            "confidence_threshold": self.confidence_threshold,
            "data_freshness_max_days": self.data_freshness_max_days,
            "ec_benchmark_override": self.ec_benchmark_override,
            "low_risk_threshold": self.low_risk_threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "min_factor_weight": self.min_factor_weight,
            "max_factor_weight": self.max_factor_weight,
            # Commodity analysis
            "correlation_threshold": self.correlation_threshold,
            "enable_seasonal_analysis": self.enable_seasonal_analysis,
            "certification_weight": self.certification_weight,
            "production_volume_weight": self.production_volume_weight,
            "supply_chain_complexity_max": (
                self.supply_chain_complexity_max
            ),
            # Hotspot detection
            "alert_threshold": self.alert_threshold,
            "clustering_min_points": self.clustering_min_points,
            "clustering_radius_km": self.clustering_radius_km,
            "enable_fire_correlation": self.enable_fire_correlation,
            "protected_area_buffer_km": self.protected_area_buffer_km,
            "min_deforestation_rate_alert": (
                self.min_deforestation_rate_alert
            ),
            "indigenous_territory_buffer_km": (
                self.indigenous_territory_buffer_km
            ),
            "trend_window_years": self.trend_window_years,
            # Governance
            "wgi_weight": self.wgi_weight,
            "cpi_weight": self.cpi_weight,
            "forest_governance_weight": self.forest_governance_weight,
            "gov_enforcement_weight": self.gov_enforcement_weight,
            "governance_data_sources": list(
                self.governance_data_sources
            ),
            "enable_legal_framework_scoring": (
                self.enable_legal_framework_scoring
            ),
            "enable_judicial_scoring": self.enable_judicial_scoring,
            # Due diligence
            "simplified_threshold": self.simplified_threshold,
            "enhanced_threshold": self.enhanced_threshold,
            "certification_credit_max": self.certification_credit_max,
            "audit_frequency_multiplier": (
                self.audit_frequency_multiplier
            ),
            "simplified_cost_min_eur": self.simplified_cost_min_eur,
            "simplified_cost_max_eur": self.simplified_cost_max_eur,
            "standard_cost_min_eur": self.standard_cost_min_eur,
            "standard_cost_max_eur": self.standard_cost_max_eur,
            "enhanced_cost_min_eur": self.enhanced_cost_min_eur,
            "enhanced_cost_max_eur": self.enhanced_cost_max_eur,
            "enable_time_to_compliance": (
                self.enable_time_to_compliance
            ),
            # Trade flows
            "min_trade_volume_tonnes": self.min_trade_volume_tonnes,
            "re_export_risk_threshold": self.re_export_risk_threshold,
            "transshipment_risk_countries": list(
                self.transshipment_risk_countries
            ),
            "hs_code_depth": self.hs_code_depth,
            "enable_concentration_risk": (
                self.enable_concentration_risk
            ),
            "enable_sanction_overlay": self.enable_sanction_overlay,
            "enable_fta_impact": self.enable_fta_impact,
            # Reports
            "output_formats": list(self.output_formats),
            "default_language": self.default_language,
            "supported_languages": list(self.supported_languages),
            "report_retention_days": self.report_retention_days,
            "template_dir": self.template_dir,
            "max_report_size_mb": self.max_report_size_mb,
            # Regulatory updates
            "monitoring_interval_hours": self.monitoring_interval_hours,
            "ec_benchmarking_url": self.ec_benchmarking_url,
            "grace_period_days": self.grace_period_days,
            "reminder_periods": list(self.reminder_periods),
            "enable_enforcement_tracking": (
                self.enable_enforcement_tracking
            ),
            # Redis caching
            "redis_ttl_s": self.redis_ttl_s,
            "redis_key_prefix": self.redis_key_prefix,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Data retention
            "retention_years": self.retention_years,
            # EUDR commodities
            "eudr_commodities": list(self.eudr_commodities),
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

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"CountryRiskEvaluatorConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[CountryRiskEvaluatorConfig] = None
_config_lock = threading.Lock()


def get_config() -> CountryRiskEvaluatorConfig:
    """Return the singleton CountryRiskEvaluatorConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_CRE_*`` environment variables.

    Returns:
        CountryRiskEvaluatorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.deforestation_weight
        30
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = CountryRiskEvaluatorConfig.from_env()
    return _config_instance


def set_config(config: CountryRiskEvaluatorConfig) -> None:
    """Replace the singleton CountryRiskEvaluatorConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New CountryRiskEvaluatorConfig to install.

    Example:
        >>> cfg = CountryRiskEvaluatorConfig(deforestation_weight=40,
        ...     governance_weight=20, enforcement_weight=10,
        ...     corruption_weight=10, forest_law_weight=10,
        ...     trend_weight=10)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "CountryRiskEvaluatorConfig replaced programmatically: "
        "weights=(%d/%d/%d/%d/%d/%d), "
        "thresholds=low=%d/high=%d",
        config.deforestation_weight,
        config.governance_weight,
        config.enforcement_weight,
        config.corruption_weight,
        config.forest_law_weight,
        config.trend_weight,
        config.low_risk_threshold,
        config.high_risk_threshold,
    )


def reset_config() -> None:
    """Reset the singleton CountryRiskEvaluatorConfig to None.

    The next call to get_config() will re-read GL_EUDR_CRE_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("CountryRiskEvaluatorConfig singleton reset")
