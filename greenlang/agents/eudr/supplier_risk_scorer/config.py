# -*- coding: utf-8 -*-
"""
Supplier Risk Scorer Configuration - AGENT-EUDR-017

Centralized configuration for the Supplier Risk Scorer Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Supplier risk scoring: 8 default factor weights (geographic_sourcing=20,
  compliance_history=15, documentation_quality=15, certification_status=15,
  traceability_completeness=10, financial_stability=10,
  environmental_performance=10, social_compliance=5), risk thresholds
  (low=25, medium=50, high=75, critical=90), confidence thresholds,
  score normalization mode, trend analysis window, aggregation method
- Due diligence: tracking periods, non-conformance categories (minor,
  major, critical), escalation thresholds, audit intervals, corrective
  action deadlines, status tracking, completion criteria, overdue limits
- Documentation: required EUDR document types (geolocation, DDS reference,
  product description, quantity declaration, harvest date, compliance
  declaration, certificate, trade license, phytosanitary), completeness
  thresholds, expiry warning periods, quality scoring criteria, format
  support (PDF, Excel, JSON, XML, Image), validation rules, gap detection
- Certification: supported schemes (FSC, PEFC, RSPO, Rainforest Alliance,
  UTZ, Organic, Fair Trade, ISCC), expiry buffer days, verification
  endpoints, chain-of-custody validation, multi-site support, scope
  verification, status tracking (valid, expired, suspended, revoked)
- Geographic sourcing: integration with AGENT-EUDR-016 (country risk),
  concentration thresholds (HHI), proximity buffers, high-risk zone
  detection, multi-origin aggregation, deforestation overlay, protected
  area detection, indigenous territory overlap
- Network analysis: max depth, risk propagation decay factor, graph
  analysis params, sub-supplier evaluation, intermediary tracking,
  tier mapping, relationship strength scoring, circular dependency
  detection, aggregate risk calculation
- Monitoring: check intervals (daily, weekly, biweekly, monthly,
  quarterly), alert thresholds by severity (info, warning, high,
  critical), watchlist limits, portfolio aggregation params, behavior
  change detection, trend analysis, escalation rules
- Reporting: formats (pdf, json, html, excel), languages (en, fr, de,
  es, pt), retention days, templates, max file size, DDS package
  generation, audit package assembly, executive summary format
- Database: PostgreSQL connection settings
- Redis: caching settings with TTL and key prefix
- Batch processing: max concurrent, timeout, batch size
- Provenance: enabled toggle, chain algorithm (sha256), genesis hash
- Metrics: Prometheus enabled toggle, prefix gl_eudr_srs_
- Rate limiting: 5 tiers (anonymous, basic, standard, premium, admin)

All settings can be overridden via environment variables with the
``GL_EUDR_SRS_`` prefix (e.g. ``GL_EUDR_SRS_DATABASE_URL``,
``GL_EUDR_SRS_GEOGRAPHIC_SOURCING_WEIGHT``).

Environment Variable Reference (GL_EUDR_SRS_ prefix):
    GL_EUDR_SRS_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_SRS_REDIS_URL                     - Redis connection URL
    GL_EUDR_SRS_LOG_LEVEL                     - Logging level
    GL_EUDR_SRS_POOL_SIZE                     - Database pool size
    GL_EUDR_SRS_GEOGRAPHIC_SOURCING_WEIGHT    - Geographic sourcing weight (0-100)
    GL_EUDR_SRS_COMPLIANCE_HISTORY_WEIGHT     - Compliance history weight (0-100)
    GL_EUDR_SRS_DOCUMENTATION_QUALITY_WEIGHT  - Documentation quality weight (0-100)
    GL_EUDR_SRS_CERTIFICATION_STATUS_WEIGHT   - Certification status weight (0-100)
    GL_EUDR_SRS_TRACEABILITY_COMPLETENESS_WEIGHT - Traceability completeness weight (0-100)
    GL_EUDR_SRS_FINANCIAL_STABILITY_WEIGHT    - Financial stability weight (0-100)
    GL_EUDR_SRS_ENVIRONMENTAL_PERFORMANCE_WEIGHT - Environmental performance weight (0-100)
    GL_EUDR_SRS_SOCIAL_COMPLIANCE_WEIGHT      - Social compliance weight (0-100)
    GL_EUDR_SRS_LOW_RISK_THRESHOLD            - Low risk threshold (0-100)
    GL_EUDR_SRS_MEDIUM_RISK_THRESHOLD         - Medium risk threshold (0-100)
    GL_EUDR_SRS_HIGH_RISK_THRESHOLD           - High risk threshold (0-100)
    GL_EUDR_SRS_CRITICAL_RISK_THRESHOLD       - Critical risk threshold (0-100)
    GL_EUDR_SRS_CONFIDENCE_THRESHOLD          - Confidence threshold (0.0-1.0)
    GL_EUDR_SRS_SCORE_NORMALIZATION           - Score normalization mode
    GL_EUDR_SRS_TREND_WINDOW_MONTHS           - Trend analysis window months
    GL_EUDR_SRS_AGGREGATION_METHOD            - Risk aggregation method
    GL_EUDR_SRS_DD_TRACKING_PERIOD_MONTHS     - DD tracking period months
    GL_EUDR_SRS_MINOR_NC_THRESHOLD            - Minor non-conformance threshold
    GL_EUDR_SRS_MAJOR_NC_THRESHOLD            - Major non-conformance threshold
    GL_EUDR_SRS_CRITICAL_NC_THRESHOLD         - Critical non-conformance threshold
    GL_EUDR_SRS_AUDIT_INTERVAL_MONTHS         - Audit interval months
    GL_EUDR_SRS_CORRECTIVE_ACTION_DEADLINE_DAYS - Corrective action deadline days
    GL_EUDR_SRS_DD_OVERDUE_LIMIT_DAYS         - DD overdue limit days
    GL_EUDR_SRS_REQUIRED_DOCUMENTS            - Required document types (comma-separated)
    GL_EUDR_SRS_COMPLETENESS_THRESHOLD        - Documentation completeness threshold (0.0-1.0)
    GL_EUDR_SRS_EXPIRY_WARNING_DAYS           - Document expiry warning days
    GL_EUDR_SRS_QUALITY_SCORING_ENABLED       - Enable quality scoring
    GL_EUDR_SRS_GAP_DETECTION_ENABLED         - Enable gap detection
    GL_EUDR_SRS_SUPPORTED_CERT_SCHEMES        - Supported certification schemes (comma-separated)
    GL_EUDR_SRS_CERT_EXPIRY_BUFFER_DAYS       - Certification expiry buffer days
    GL_EUDR_SRS_CHAIN_OF_CUSTODY_REQUIRED     - Require chain-of-custody validation
    GL_EUDR_SRS_MULTI_SITE_ENABLED            - Enable multi-site certification
    GL_EUDR_SRS_CONCENTRATION_THRESHOLD       - Geographic concentration threshold (HHI)
    GL_EUDR_SRS_PROXIMITY_BUFFER_KM           - Proximity buffer km
    GL_EUDR_SRS_HIGH_RISK_ZONE_ENABLED        - Enable high-risk zone detection
    GL_EUDR_SRS_DEFORESTATION_OVERLAY_ENABLED - Enable deforestation overlay
    GL_EUDR_SRS_PROTECTED_AREA_DETECTION      - Enable protected area detection
    GL_EUDR_SRS_INDIGENOUS_TERRITORY_OVERLAP  - Enable indigenous territory overlap
    GL_EUDR_SRS_NETWORK_MAX_DEPTH             - Network analysis max depth
    GL_EUDR_SRS_RISK_PROPAGATION_DECAY        - Risk propagation decay factor (0.0-1.0)
    GL_EUDR_SRS_SUB_SUPPLIER_EVALUATION       - Enable sub-supplier evaluation
    GL_EUDR_SRS_INTERMEDIARY_TRACKING         - Enable intermediary tracking
    GL_EUDR_SRS_CIRCULAR_DEPENDENCY_DETECTION - Enable circular dependency detection
    GL_EUDR_SRS_MONITORING_DEFAULT_FREQUENCY  - Default monitoring frequency
    GL_EUDR_SRS_ALERT_INFO_THRESHOLD          - Alert info severity threshold (0-100)
    GL_EUDR_SRS_ALERT_WARNING_THRESHOLD       - Alert warning severity threshold (0-100)
    GL_EUDR_SRS_ALERT_HIGH_THRESHOLD          - Alert high severity threshold (0-100)
    GL_EUDR_SRS_ALERT_CRITICAL_THRESHOLD      - Alert critical severity threshold (0-100)
    GL_EUDR_SRS_WATCHLIST_MAX_SIZE            - Watchlist max size
    GL_EUDR_SRS_BEHAVIOR_CHANGE_DETECTION     - Enable behavior change detection
    GL_EUDR_SRS_DEFAULT_LANGUAGE              - Default report language
    GL_EUDR_SRS_REPORT_RETENTION_DAYS         - Report retention in days
    GL_EUDR_SRS_TEMPLATE_DIR                  - Report template directory
    GL_EUDR_SRS_MAX_REPORT_SIZE_MB            - Max report file size MB
    GL_EUDR_SRS_DDS_PACKAGE_GENERATION        - Enable DDS package generation
    GL_EUDR_SRS_AUDIT_PACKAGE_ASSEMBLY        - Enable audit package assembly
    GL_EUDR_SRS_REDIS_TTL_S                   - Redis cache TTL seconds
    GL_EUDR_SRS_REDIS_KEY_PREFIX              - Redis key prefix
    GL_EUDR_SRS_BATCH_MAX_SIZE                - Batch processing max size
    GL_EUDR_SRS_BATCH_CONCURRENCY             - Batch concurrency workers
    GL_EUDR_SRS_BATCH_TIMEOUT_S               - Batch timeout seconds
    GL_EUDR_SRS_RETENTION_YEARS               - Data retention years
    GL_EUDR_SRS_ENABLE_PROVENANCE             - Enable provenance tracking
    GL_EUDR_SRS_GENESIS_HASH                  - Genesis hash anchor
    GL_EUDR_SRS_CHAIN_ALGORITHM               - Provenance chain hash algorithm
    GL_EUDR_SRS_ENABLE_METRICS                - Enable Prometheus metrics
    GL_EUDR_SRS_METRICS_PREFIX                - Prometheus metrics prefix
    GL_EUDR_SRS_RATE_LIMIT_ANONYMOUS          - Rate limit anonymous tier
    GL_EUDR_SRS_RATE_LIMIT_BASIC              - Rate limit basic tier
    GL_EUDR_SRS_RATE_LIMIT_STANDARD           - Rate limit standard tier
    GL_EUDR_SRS_RATE_LIMIT_PREMIUM            - Rate limit premium tier
    GL_EUDR_SRS_RATE_LIMIT_ADMIN              - Rate limit admin tier

Example:
    >>> from greenlang.agents.eudr.supplier_risk_scorer.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.geographic_sourcing_weight, cfg.low_risk_threshold)
    20 25

    >>> # Override for testing
    >>> from greenlang.agents.eudr.supplier_risk_scorer.config import (
    ...     set_config, reset_config, SupplierRiskScorerConfig,
    ... )
    >>> set_config(SupplierRiskScorerConfig(geographic_sourcing_weight=25))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
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

_ENV_PREFIX = "GL_EUDR_SRS_"

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
# Valid aggregation methods
# ---------------------------------------------------------------------------

_VALID_AGGREGATION_METHODS = frozenset({
    "weighted_average",
    "maximum",
    "minimum",
    "median",
})

# ---------------------------------------------------------------------------
# Valid monitoring frequencies
# ---------------------------------------------------------------------------

_VALID_MONITORING_FREQUENCIES = frozenset({
    "daily",
    "weekly",
    "biweekly",
    "monthly",
    "quarterly",
})

# ---------------------------------------------------------------------------
# Valid report output formats
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "excel", "csv"})

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
# Valid document types
# ---------------------------------------------------------------------------

_VALID_DOCUMENT_TYPES = frozenset({
    "geolocation",
    "dds_reference",
    "product_description",
    "quantity_declaration",
    "harvest_date",
    "compliance_declaration",
    "certificate",
    "trade_license",
    "phytosanitary",
})

# ---------------------------------------------------------------------------
# Valid certification schemes
# ---------------------------------------------------------------------------

_VALID_CERTIFICATION_SCHEMES = frozenset({
    "FSC", "PEFC", "RSPO", "RAINFOREST_ALLIANCE", "UTZ",
    "ORGANIC", "FAIR_TRADE", "ISCC",
})

# ---------------------------------------------------------------------------
# Valid document formats
# ---------------------------------------------------------------------------

_VALID_DOCUMENT_FORMATS = frozenset({
    "pdf", "excel", "json", "xml", "image",
})

# ---------------------------------------------------------------------------
# Default required documents
# ---------------------------------------------------------------------------

_DEFAULT_REQUIRED_DOCUMENTS: List[str] = [
    "geolocation",
    "dds_reference",
    "product_description",
    "quantity_declaration",
    "harvest_date",
    "compliance_declaration",
]

# ---------------------------------------------------------------------------
# Default supported certification schemes
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORTED_SCHEMES: List[str] = [
    "FSC", "PEFC", "RSPO", "RAINFOREST_ALLIANCE", "UTZ",
    "ORGANIC", "FAIR_TRADE", "ISCC",
]

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SupplierRiskScorerConfig:
    """Configuration for the Supplier Risk Scorer Agent (AGENT-EUDR-017).

    This dataclass encapsulates all configuration settings for supplier risk
    scoring, due diligence tracking, documentation analysis, certification
    validation, geographic sourcing analysis, network analysis, monitoring,
    and reporting. All settings have sensible defaults aligned with EUDR
    requirements and can be overridden via environment variables with the
    GL_EUDR_SRS_ prefix.

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

        # Supplier Risk Scoring (8 factors)
        geographic_sourcing_weight: Geographic sourcing risk weight (0-100)
        compliance_history_weight: Compliance history weight (0-100)
        documentation_quality_weight: Documentation quality weight (0-100)
        certification_status_weight: Certification status weight (0-100)
        traceability_completeness_weight: Traceability completeness weight (0-100)
        financial_stability_weight: Financial stability weight (0-100)
        environmental_performance_weight: Environmental performance weight (0-100)
        social_compliance_weight: Social compliance weight (0-100)
        low_risk_threshold: Low risk threshold (0-100)
        medium_risk_threshold: Medium risk threshold (0-100)
        high_risk_threshold: High risk threshold (0-100)
        critical_risk_threshold: Critical risk threshold (0-100)
        confidence_threshold: Confidence threshold (0.0-1.0)
        score_normalization: Score normalization mode
        trend_window_months: Trend analysis window months
        aggregation_method: Risk aggregation method
        min_factor_weight: Minimum weight per factor (0-100)
        max_factor_weight: Maximum weight per factor (0-100)

        # Due Diligence settings
        dd_tracking_period_months: DD tracking period months
        minor_nc_threshold: Minor non-conformance threshold
        major_nc_threshold: Major non-conformance threshold
        critical_nc_threshold: Critical non-conformance threshold
        audit_interval_months: Audit interval months
        corrective_action_deadline_days: Corrective action deadline days
        dd_overdue_limit_days: DD overdue limit days
        enable_status_tracking: Enable DD status tracking
        enable_completion_criteria: Enable DD completion criteria validation

        # Documentation settings
        required_documents: Required document types list
        completeness_threshold: Documentation completeness threshold (0.0-1.0)
        expiry_warning_days: Document expiry warning days
        quality_scoring_enabled: Enable quality scoring
        gap_detection_enabled: Enable gap detection
        supported_document_formats: Supported document formats
        max_document_size_mb: Max document file size MB
        validation_rules_enabled: Enable validation rules

        # Certification settings
        supported_cert_schemes: Supported certification schemes
        cert_expiry_buffer_days: Certification expiry buffer days
        chain_of_custody_required: Require chain-of-custody validation
        multi_site_enabled: Enable multi-site certification
        scope_verification_enabled: Enable scope verification
        cert_verification_endpoint: Certification verification endpoint URL

        # Geographic Sourcing settings
        concentration_threshold: Geographic concentration threshold (HHI)
        proximity_buffer_km: Proximity buffer km
        high_risk_zone_enabled: Enable high-risk zone detection
        deforestation_overlay_enabled: Enable deforestation overlay
        protected_area_detection: Enable protected area detection
        indigenous_territory_overlap: Enable indigenous territory overlap
        multi_origin_aggregation: Enable multi-origin aggregation
        country_risk_integration_enabled: Enable country risk integration (EUDR-016)

        # Network Analysis settings
        network_max_depth: Network analysis max depth
        risk_propagation_decay: Risk propagation decay factor (0.0-1.0)
        sub_supplier_evaluation: Enable sub-supplier evaluation
        intermediary_tracking: Enable intermediary tracking
        tier_mapping_enabled: Enable tier mapping
        relationship_strength_scoring: Enable relationship strength scoring
        circular_dependency_detection: Enable circular dependency detection
        aggregate_risk_calculation: Enable aggregate risk calculation

        # Monitoring settings
        monitoring_default_frequency: Default monitoring frequency
        alert_info_threshold: Alert info severity threshold (0-100)
        alert_warning_threshold: Alert warning severity threshold (0-100)
        alert_high_threshold: Alert high severity threshold (0-100)
        alert_critical_threshold: Alert critical severity threshold (0-100)
        watchlist_max_size: Watchlist max size
        behavior_change_detection: Enable behavior change detection
        trend_analysis_enabled: Enable trend analysis
        escalation_rules_enabled: Enable escalation rules
        portfolio_aggregation_enabled: Enable portfolio aggregation

        # Reporting settings
        output_formats: Report output formats
        default_language: Default report language
        supported_languages: Supported report languages
        report_retention_days: Report retention in days
        template_dir: Report template directory
        max_report_size_mb: Max report file size MB
        dds_package_generation: Enable DDS package generation
        audit_package_assembly: Enable audit package assembly
        executive_summary_format: Executive summary format

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
    redis_key_prefix: str = "gl:eudr:srs:"

    # Logging
    log_level: str = "INFO"

    # Supplier Risk Scoring (8 factors)
    geographic_sourcing_weight: int = 20
    compliance_history_weight: int = 15
    documentation_quality_weight: int = 15
    certification_status_weight: int = 15
    traceability_completeness_weight: int = 10
    financial_stability_weight: int = 10
    environmental_performance_weight: int = 10
    social_compliance_weight: int = 5
    low_risk_threshold: int = 25
    medium_risk_threshold: int = 50
    high_risk_threshold: int = 75
    critical_risk_threshold: int = 90
    confidence_threshold: float = 0.70
    score_normalization: str = "minmax"
    trend_window_months: int = 12
    aggregation_method: str = "weighted_average"
    min_factor_weight: int = 5
    max_factor_weight: int = 50

    # Due Diligence settings
    dd_tracking_period_months: int = 12
    minor_nc_threshold: int = 3
    major_nc_threshold: int = 2
    critical_nc_threshold: int = 1
    audit_interval_months: int = 12
    corrective_action_deadline_days: int = 90
    dd_overdue_limit_days: int = 30
    enable_status_tracking: bool = True
    enable_completion_criteria: bool = True

    # Documentation settings
    required_documents: List[str] = field(
        default_factory=lambda: _DEFAULT_REQUIRED_DOCUMENTS.copy()
    )
    completeness_threshold: float = 0.80
    expiry_warning_days: int = 90
    quality_scoring_enabled: bool = True
    gap_detection_enabled: bool = True
    supported_document_formats: List[str] = field(
        default_factory=lambda: ["pdf", "excel", "json", "xml", "image"]
    )
    max_document_size_mb: int = 25
    validation_rules_enabled: bool = True

    # Certification settings
    supported_cert_schemes: List[str] = field(
        default_factory=lambda: _DEFAULT_SUPPORTED_SCHEMES.copy()
    )
    cert_expiry_buffer_days: int = 90
    chain_of_custody_required: bool = True
    multi_site_enabled: bool = True
    scope_verification_enabled: bool = True
    cert_verification_endpoint: str = ""

    # Geographic Sourcing settings
    concentration_threshold: float = 0.25
    proximity_buffer_km: float = 10.0
    high_risk_zone_enabled: bool = True
    deforestation_overlay_enabled: bool = True
    protected_area_detection: bool = True
    indigenous_territory_overlap: bool = True
    multi_origin_aggregation: bool = True
    country_risk_integration_enabled: bool = True

    # Network Analysis settings
    network_max_depth: int = 3
    risk_propagation_decay: float = 0.80
    sub_supplier_evaluation: bool = True
    intermediary_tracking: bool = True
    tier_mapping_enabled: bool = True
    relationship_strength_scoring: bool = True
    circular_dependency_detection: bool = True
    aggregate_risk_calculation: bool = True

    # Monitoring settings
    monitoring_default_frequency: str = "monthly"
    alert_info_threshold: int = 25
    alert_warning_threshold: int = 50
    alert_high_threshold: int = 75
    alert_critical_threshold: int = 90
    watchlist_max_size: int = 1000
    behavior_change_detection: bool = True
    trend_analysis_enabled: bool = True
    escalation_rules_enabled: bool = True
    portfolio_aggregation_enabled: bool = True

    # Reporting settings
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "excel"]
    )
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )
    report_retention_days: int = 1825  # 5 years
    template_dir: str = "templates/eudr/supplier_risk_scorer"
    max_report_size_mb: int = 50
    dds_package_generation: bool = True
    audit_package_assembly: bool = True
    executive_summary_format: str = "pdf"

    # Batch processing
    batch_max_size: int = 500
    batch_concurrency: int = 4
    batch_timeout_s: int = 300

    # Data retention
    retention_years: int = 5

    # Provenance
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-SRS-017-SUPPLIER-RISK-SCORER-GENESIS"
    chain_algorithm: str = "sha256"

    # Metrics
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_srs_"

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

        # Validate score normalization
        if self.score_normalization not in _VALID_NORMALIZATION_MODES:
            raise ValueError(
                f"Invalid score_normalization: {self.score_normalization}. "
                f"Must be one of {_VALID_NORMALIZATION_MODES}"
            )

        # Validate aggregation method
        if self.aggregation_method not in _VALID_AGGREGATION_METHODS:
            raise ValueError(
                f"Invalid aggregation_method: {self.aggregation_method}. "
                f"Must be one of {_VALID_AGGREGATION_METHODS}"
            )

        # Validate monitoring frequency
        if self.monitoring_default_frequency not in _VALID_MONITORING_FREQUENCIES:
            raise ValueError(
                f"Invalid monitoring_default_frequency: {self.monitoring_default_frequency}. "
                f"Must be one of {_VALID_MONITORING_FREQUENCIES}"
            )

        # Validate chain algorithm
        if self.chain_algorithm not in _VALID_CHAIN_ALGORITHMS:
            raise ValueError(
                f"Invalid chain_algorithm: {self.chain_algorithm}. "
                f"Must be one of {_VALID_CHAIN_ALGORITHMS}"
            )

        # Validate weights sum to 100
        total_weight = (
            self.geographic_sourcing_weight
            + self.compliance_history_weight
            + self.documentation_quality_weight
            + self.certification_status_weight
            + self.traceability_completeness_weight
            + self.financial_stability_weight
            + self.environmental_performance_weight
            + self.social_compliance_weight
        )
        if total_weight != 100:
            raise ValueError(
                f"Factor weights must sum to 100, got {total_weight}"
            )

        # Validate weight bounds
        weights = [
            self.geographic_sourcing_weight,
            self.compliance_history_weight,
            self.documentation_quality_weight,
            self.certification_status_weight,
            self.traceability_completeness_weight,
            self.financial_stability_weight,
            self.environmental_performance_weight,
            self.social_compliance_weight,
        ]
        for w in weights:
            if not self.min_factor_weight <= w <= self.max_factor_weight:
                raise ValueError(
                    f"All factor weights must be between {self.min_factor_weight} "
                    f"and {self.max_factor_weight}"
                )

        # Validate risk thresholds
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

        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        # Validate completeness threshold
        if not 0.0 <= self.completeness_threshold <= 1.0:
            raise ValueError(
                f"completeness_threshold must be between 0.0 and 1.0, "
                f"got {self.completeness_threshold}"
            )

        # Validate concentration threshold
        if not 0.0 <= self.concentration_threshold <= 1.0:
            raise ValueError(
                f"concentration_threshold must be between 0.0 and 1.0, "
                f"got {self.concentration_threshold}"
            )

        # Validate risk propagation decay
        if not 0.0 <= self.risk_propagation_decay <= 1.0:
            raise ValueError(
                f"risk_propagation_decay must be between 0.0 and 1.0, "
                f"got {self.risk_propagation_decay}"
            )

        # Validate alert thresholds
        if not (
            0 <= self.alert_info_threshold
            < self.alert_warning_threshold
            < self.alert_high_threshold
            < self.alert_critical_threshold
            <= 100
        ):
            raise ValueError(
                "Alert thresholds must be in ascending order: "
                "0 <= info < warning < high < critical <= 100"
            )

        # Validate required documents
        for doc in self.required_documents:
            if doc not in _VALID_DOCUMENT_TYPES:
                raise ValueError(
                    f"Invalid required document type: {doc}. "
                    f"Must be one of {_VALID_DOCUMENT_TYPES}"
                )

        # Validate supported certification schemes
        for scheme in self.supported_cert_schemes:
            if scheme not in _VALID_CERTIFICATION_SCHEMES:
                raise ValueError(
                    f"Invalid certification scheme: {scheme}. "
                    f"Must be one of {_VALID_CERTIFICATION_SCHEMES}"
                )

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

        # Validate positive integers
        if self.pool_size <= 0:
            raise ValueError(f"pool_size must be > 0, got {self.pool_size}")
        if self.batch_max_size <= 0:
            raise ValueError(f"batch_max_size must be > 0, got {self.batch_max_size}")
        if self.batch_concurrency <= 0:
            raise ValueError(f"batch_concurrency must be > 0, got {self.batch_concurrency}")
        if self.retention_years <= 0:
            raise ValueError(f"retention_years must be > 0, got {self.retention_years}")

        logger.info(
            f"SupplierRiskScorerConfig initialized: "
            f"geo_weight={self.geographic_sourcing_weight}, "
            f"compliance_weight={self.compliance_history_weight}, "
            f"low_threshold={self.low_risk_threshold}, "
            f"high_threshold={self.high_risk_threshold}"
        )

    @classmethod
    def from_env(cls) -> "SupplierRiskScorerConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_SRS_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            SupplierRiskScorerConfig instance with env overrides applied.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_SRS_GEOGRAPHIC_SOURCING_WEIGHT"] = "25"
            >>> cfg = SupplierRiskScorerConfig.from_env()
            >>> assert cfg.geographic_sourcing_weight == 25
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

        # Supplier Risk Scoring
        if val := os.getenv(f"{_ENV_PREFIX}GEOGRAPHIC_SOURCING_WEIGHT"):
            kwargs["geographic_sourcing_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPLIANCE_HISTORY_WEIGHT"):
            kwargs["compliance_history_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DOCUMENTATION_QUALITY_WEIGHT"):
            kwargs["documentation_quality_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CERTIFICATION_STATUS_WEIGHT"):
            kwargs["certification_status_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TRACEABILITY_COMPLETENESS_WEIGHT"):
            kwargs["traceability_completeness_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}FINANCIAL_STABILITY_WEIGHT"):
            kwargs["financial_stability_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ENVIRONMENTAL_PERFORMANCE_WEIGHT"):
            kwargs["environmental_performance_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SOCIAL_COMPLIANCE_WEIGHT"):
            kwargs["social_compliance_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}LOW_RISK_THRESHOLD"):
            kwargs["low_risk_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MEDIUM_RISK_THRESHOLD"):
            kwargs["medium_risk_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}HIGH_RISK_THRESHOLD"):
            kwargs["high_risk_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_RISK_THRESHOLD"):
            kwargs["critical_risk_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CONFIDENCE_THRESHOLD"):
            kwargs["confidence_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}SCORE_NORMALIZATION"):
            kwargs["score_normalization"] = val
        if val := os.getenv(f"{_ENV_PREFIX}TREND_WINDOW_MONTHS"):
            kwargs["trend_window_months"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}AGGREGATION_METHOD"):
            kwargs["aggregation_method"] = val
        if val := os.getenv(f"{_ENV_PREFIX}MIN_FACTOR_WEIGHT"):
            kwargs["min_factor_weight"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_FACTOR_WEIGHT"):
            kwargs["max_factor_weight"] = int(val)

        # Due Diligence
        if val := os.getenv(f"{_ENV_PREFIX}DD_TRACKING_PERIOD_MONTHS"):
            kwargs["dd_tracking_period_months"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MINOR_NC_THRESHOLD"):
            kwargs["minor_nc_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAJOR_NC_THRESHOLD"):
            kwargs["major_nc_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CRITICAL_NC_THRESHOLD"):
            kwargs["critical_nc_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}AUDIT_INTERVAL_MONTHS"):
            kwargs["audit_interval_months"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CORRECTIVE_ACTION_DEADLINE_DAYS"):
            kwargs["corrective_action_deadline_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DD_OVERDUE_LIMIT_DAYS"):
            kwargs["dd_overdue_limit_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ENABLE_STATUS_TRACKING"):
            kwargs["enable_status_tracking"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}ENABLE_COMPLETION_CRITERIA"):
            kwargs["enable_completion_criteria"] = val.lower() in ("true", "1", "yes")

        # Documentation
        if val := os.getenv(f"{_ENV_PREFIX}REQUIRED_DOCUMENTS"):
            kwargs["required_documents"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}COMPLETENESS_THRESHOLD"):
            kwargs["completeness_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}EXPIRY_WARNING_DAYS"):
            kwargs["expiry_warning_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}QUALITY_SCORING_ENABLED"):
            kwargs["quality_scoring_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}GAP_DETECTION_ENABLED"):
            kwargs["gap_detection_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}MAX_DOCUMENT_SIZE_MB"):
            kwargs["max_document_size_mb"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}VALIDATION_RULES_ENABLED"):
            kwargs["validation_rules_enabled"] = val.lower() in ("true", "1", "yes")

        # Certification
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_CERT_SCHEMES"):
            kwargs["supported_cert_schemes"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}CERT_EXPIRY_BUFFER_DAYS"):
            kwargs["cert_expiry_buffer_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CHAIN_OF_CUSTODY_REQUIRED"):
            kwargs["chain_of_custody_required"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}MULTI_SITE_ENABLED"):
            kwargs["multi_site_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}SCOPE_VERIFICATION_ENABLED"):
            kwargs["scope_verification_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}CERT_VERIFICATION_ENDPOINT"):
            kwargs["cert_verification_endpoint"] = val

        # Geographic Sourcing
        if val := os.getenv(f"{_ENV_PREFIX}CONCENTRATION_THRESHOLD"):
            kwargs["concentration_threshold"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}PROXIMITY_BUFFER_KM"):
            kwargs["proximity_buffer_km"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}HIGH_RISK_ZONE_ENABLED"):
            kwargs["high_risk_zone_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}DEFORESTATION_OVERLAY_ENABLED"):
            kwargs["deforestation_overlay_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}PROTECTED_AREA_DETECTION"):
            kwargs["protected_area_detection"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}INDIGENOUS_TERRITORY_OVERLAP"):
            kwargs["indigenous_territory_overlap"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}MULTI_ORIGIN_AGGREGATION"):
            kwargs["multi_origin_aggregation"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}COUNTRY_RISK_INTEGRATION_ENABLED"):
            kwargs["country_risk_integration_enabled"] = val.lower() in ("true", "1", "yes")

        # Network Analysis
        if val := os.getenv(f"{_ENV_PREFIX}NETWORK_MAX_DEPTH"):
            kwargs["network_max_depth"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RISK_PROPAGATION_DECAY"):
            kwargs["risk_propagation_decay"] = float(val)
        if val := os.getenv(f"{_ENV_PREFIX}SUB_SUPPLIER_EVALUATION"):
            kwargs["sub_supplier_evaluation"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}INTERMEDIARY_TRACKING"):
            kwargs["intermediary_tracking"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}TIER_MAPPING_ENABLED"):
            kwargs["tier_mapping_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}RELATIONSHIP_STRENGTH_SCORING"):
            kwargs["relationship_strength_scoring"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}CIRCULAR_DEPENDENCY_DETECTION"):
            kwargs["circular_dependency_detection"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}AGGREGATE_RISK_CALCULATION"):
            kwargs["aggregate_risk_calculation"] = val.lower() in ("true", "1", "yes")

        # Monitoring
        if val := os.getenv(f"{_ENV_PREFIX}MONITORING_DEFAULT_FREQUENCY"):
            kwargs["monitoring_default_frequency"] = val
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_INFO_THRESHOLD"):
            kwargs["alert_info_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_WARNING_THRESHOLD"):
            kwargs["alert_warning_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_HIGH_THRESHOLD"):
            kwargs["alert_high_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_CRITICAL_THRESHOLD"):
            kwargs["alert_critical_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}WATCHLIST_MAX_SIZE"):
            kwargs["watchlist_max_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BEHAVIOR_CHANGE_DETECTION"):
            kwargs["behavior_change_detection"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}TREND_ANALYSIS_ENABLED"):
            kwargs["trend_analysis_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}ESCALATION_RULES_ENABLED"):
            kwargs["escalation_rules_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}PORTFOLIO_AGGREGATION_ENABLED"):
            kwargs["portfolio_aggregation_enabled"] = val.lower() in ("true", "1", "yes")

        # Reporting
        if val := os.getenv(f"{_ENV_PREFIX}OUTPUT_FORMATS"):
            kwargs["output_formats"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_LANGUAGE"):
            kwargs["default_language"] = val
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_LANGUAGES"):
            kwargs["supported_languages"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}REPORT_RETENTION_DAYS"):
            kwargs["report_retention_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TEMPLATE_DIR"):
            kwargs["template_dir"] = val
        if val := os.getenv(f"{_ENV_PREFIX}MAX_REPORT_SIZE_MB"):
            kwargs["max_report_size_mb"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}DDS_PACKAGE_GENERATION"):
            kwargs["dds_package_generation"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}AUDIT_PACKAGE_ASSEMBLY"):
            kwargs["audit_package_assembly"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}EXECUTIVE_SUMMARY_FORMAT"):
            kwargs["executive_summary_format"] = val

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
            >>> cfg = SupplierRiskScorerConfig()
            >>> d = cfg.to_dict(redact_secrets=True)
            >>> assert "database_url" in d
            >>> assert "REDACTED" in d["database_url"]
        """
        data = {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "pool_timeout_s": self.pool_timeout_s,
            "pool_recycle_s": self.pool_recycle_s,
            "redis_url": self.redis_url,
            "redis_ttl_s": self.redis_ttl_s,
            "redis_key_prefix": self.redis_key_prefix,
            "log_level": self.log_level,
            "geographic_sourcing_weight": self.geographic_sourcing_weight,
            "compliance_history_weight": self.compliance_history_weight,
            "documentation_quality_weight": self.documentation_quality_weight,
            "certification_status_weight": self.certification_status_weight,
            "traceability_completeness_weight": self.traceability_completeness_weight,
            "financial_stability_weight": self.financial_stability_weight,
            "environmental_performance_weight": self.environmental_performance_weight,
            "social_compliance_weight": self.social_compliance_weight,
            "low_risk_threshold": self.low_risk_threshold,
            "medium_risk_threshold": self.medium_risk_threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "critical_risk_threshold": self.critical_risk_threshold,
            "confidence_threshold": self.confidence_threshold,
            "score_normalization": self.score_normalization,
            "trend_window_months": self.trend_window_months,
            "aggregation_method": self.aggregation_method,
            "min_factor_weight": self.min_factor_weight,
            "max_factor_weight": self.max_factor_weight,
            "dd_tracking_period_months": self.dd_tracking_period_months,
            "minor_nc_threshold": self.minor_nc_threshold,
            "major_nc_threshold": self.major_nc_threshold,
            "critical_nc_threshold": self.critical_nc_threshold,
            "audit_interval_months": self.audit_interval_months,
            "corrective_action_deadline_days": self.corrective_action_deadline_days,
            "dd_overdue_limit_days": self.dd_overdue_limit_days,
            "enable_status_tracking": self.enable_status_tracking,
            "enable_completion_criteria": self.enable_completion_criteria,
            "required_documents": self.required_documents,
            "completeness_threshold": self.completeness_threshold,
            "expiry_warning_days": self.expiry_warning_days,
            "quality_scoring_enabled": self.quality_scoring_enabled,
            "gap_detection_enabled": self.gap_detection_enabled,
            "supported_document_formats": self.supported_document_formats,
            "max_document_size_mb": self.max_document_size_mb,
            "validation_rules_enabled": self.validation_rules_enabled,
            "supported_cert_schemes": self.supported_cert_schemes,
            "cert_expiry_buffer_days": self.cert_expiry_buffer_days,
            "chain_of_custody_required": self.chain_of_custody_required,
            "multi_site_enabled": self.multi_site_enabled,
            "scope_verification_enabled": self.scope_verification_enabled,
            "cert_verification_endpoint": self.cert_verification_endpoint,
            "concentration_threshold": self.concentration_threshold,
            "proximity_buffer_km": self.proximity_buffer_km,
            "high_risk_zone_enabled": self.high_risk_zone_enabled,
            "deforestation_overlay_enabled": self.deforestation_overlay_enabled,
            "protected_area_detection": self.protected_area_detection,
            "indigenous_territory_overlap": self.indigenous_territory_overlap,
            "multi_origin_aggregation": self.multi_origin_aggregation,
            "country_risk_integration_enabled": self.country_risk_integration_enabled,
            "network_max_depth": self.network_max_depth,
            "risk_propagation_decay": self.risk_propagation_decay,
            "sub_supplier_evaluation": self.sub_supplier_evaluation,
            "intermediary_tracking": self.intermediary_tracking,
            "tier_mapping_enabled": self.tier_mapping_enabled,
            "relationship_strength_scoring": self.relationship_strength_scoring,
            "circular_dependency_detection": self.circular_dependency_detection,
            "aggregate_risk_calculation": self.aggregate_risk_calculation,
            "monitoring_default_frequency": self.monitoring_default_frequency,
            "alert_info_threshold": self.alert_info_threshold,
            "alert_warning_threshold": self.alert_warning_threshold,
            "alert_high_threshold": self.alert_high_threshold,
            "alert_critical_threshold": self.alert_critical_threshold,
            "watchlist_max_size": self.watchlist_max_size,
            "behavior_change_detection": self.behavior_change_detection,
            "trend_analysis_enabled": self.trend_analysis_enabled,
            "escalation_rules_enabled": self.escalation_rules_enabled,
            "portfolio_aggregation_enabled": self.portfolio_aggregation_enabled,
            "output_formats": self.output_formats,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "report_retention_days": self.report_retention_days,
            "template_dir": self.template_dir,
            "max_report_size_mb": self.max_report_size_mb,
            "dds_package_generation": self.dds_package_generation,
            "audit_package_assembly": self.audit_package_assembly,
            "executive_summary_format": self.executive_summary_format,
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
            if data["cert_verification_endpoint"]:
                data["cert_verification_endpoint"] = "REDACTED"

        return data


# ---------------------------------------------------------------------------
# Thread-safe singleton pattern
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[SupplierRiskScorerConfig] = None


def get_config() -> SupplierRiskScorerConfig:
    """Get the global SupplierRiskScorerConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance.

    Returns:
        SupplierRiskScorerConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> assert cfg.geographic_sourcing_weight == 20
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2  # Same instance
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = SupplierRiskScorerConfig.from_env()
    return _global_config


def set_config(config: SupplierRiskScorerConfig) -> None:
    """Set the global SupplierRiskScorerConfig singleton instance.

    Used for testing and programmatic configuration override.

    Args:
        config: SupplierRiskScorerConfig instance to set as global.

    Example:
        >>> from greenlang.agents.eudr.supplier_risk_scorer.config import (
        ...     set_config, SupplierRiskScorerConfig,
        ... )
        >>> test_cfg = SupplierRiskScorerConfig(geographic_sourcing_weight=25)
        >>> set_config(test_cfg)
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global SupplierRiskScorerConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_config()
        >>> # Next get_config() call will re-initialize from environment
    """
    global _global_config
    with _config_lock:
        _global_config = None
