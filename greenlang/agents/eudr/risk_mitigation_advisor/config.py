# -*- coding: utf-8 -*-
"""
Risk Mitigation Advisor Configuration - AGENT-EUDR-025

Centralized configuration for the Risk Mitigation Advisor Agent covering:
- Database and cache connection settings (PostgreSQL, Redis) with configurable
  pool sizes, timeouts, and key prefixes using ``gl_eudr_rma_`` namespace
- ML model settings: XGBoost/LightGBM for strategy recommendation with
  configurable confidence thresholds, feature engineering parameters,
  model versioning, and deterministic fallback mode
- Strategy selection parameters: top-k recommendations (default 5), minimum
  confidence (0.7), SHAP explainability enablement, deterministic mode toggle
- Remediation plan settings: plan templates (8 types), SMART milestone
  generation, 4 plan phases (Preparation, Implementation, Verification,
  Monitoring), versioning, and approval workflows
- Capacity building: 4 tiers (Awareness, Basic, Advanced, Leadership),
  7 commodity-specific curricula, 22 modules per commodity, competency
  assessment thresholds
- Measure library: 500+ measures across 8 risk categories, PostgreSQL
  full-text search with tsvector ranking, faceted filtering, version control
- Effectiveness tracking: baseline snapshot intervals, ROI calculation
  parameters, statistical significance testing (paired t-test, p < 0.05),
  Decimal arithmetic for zero-hallucination calculations
- Continuous monitoring: event processing SLA, trigger detection thresholds,
  6 trigger event types, 5 adjustment types, alert fatigue prevention,
  configurable escalation chains
- Cost-benefit optimization: linear programming solver (scipy.optimize),
  budget constraint types, RICE prioritization framework, Pareto frontier
  generation, multi-scenario analysis support
- Stakeholder collaboration: 6 stakeholder roles, RBAC access matrix,
  threaded communication, document sharing, supplier self-service portal
- Reporting: 5 output formats (PDF, JSON, HTML, XLSX, XML), 5 languages
  (EN, FR, DE, ES, PT), 7 report types, scheduling intervals, 5-year
  archive per EUDR Article 31
- Provenance: SHA-256 chain hashing, genesis hash anchor, configurable
  hash algorithm (sha256/sha384/sha512)
- Rate limiting across 5 tiers: anonymous (10/min), basic (60/min),
  standard (300/min), premium (1000/min), admin (10000/min)

All settings can be overridden via environment variables with the
``GL_EUDR_RMA_`` prefix (e.g. ``GL_EUDR_RMA_DATABASE_URL``,
``GL_EUDR_RMA_ML_CONFIDENCE_THRESHOLD``).

Environment Variable Reference (GL_EUDR_RMA_ prefix):
    GL_EUDR_RMA_DATABASE_URL                    - PostgreSQL connection URL
    GL_EUDR_RMA_REDIS_URL                       - Redis connection URL
    GL_EUDR_RMA_LOG_LEVEL                       - Logging level
    GL_EUDR_RMA_POOL_SIZE                       - Database pool size
    GL_EUDR_RMA_POOL_TIMEOUT_S                  - Pool timeout seconds
    GL_EUDR_RMA_POOL_RECYCLE_S                  - Pool recycle seconds
    GL_EUDR_RMA_REDIS_TTL_S                     - Redis cache TTL seconds
    GL_EUDR_RMA_REDIS_KEY_PREFIX                - Redis key prefix
    GL_EUDR_RMA_ML_MODEL_TYPE                   - ML model type (xgboost/lightgbm)
    GL_EUDR_RMA_ML_CONFIDENCE_THRESHOLD         - ML confidence threshold
    GL_EUDR_RMA_DETERMINISTIC_MODE              - Enable deterministic mode
    GL_EUDR_RMA_TOP_K_STRATEGIES                - Top-K strategy recommendations
    GL_EUDR_RMA_SHAP_ENABLED                    - Enable SHAP explainability
    GL_EUDR_RMA_PLAN_PHASES_COUNT               - Number of plan phases
    GL_EUDR_RMA_CAPACITY_TIERS                  - Number of capacity tiers
    GL_EUDR_RMA_MODULES_PER_COMMODITY           - Modules per commodity
    GL_EUDR_RMA_MEASURE_LIBRARY_SIZE            - Minimum measure library size
    GL_EUDR_RMA_EFFECTIVENESS_INTERVAL_DAYS     - Effectiveness measurement interval
    GL_EUDR_RMA_SIGNIFICANCE_LEVEL              - Statistical significance level
    GL_EUDR_RMA_MONITORING_INTERVAL_S           - Monitoring interval seconds
    GL_EUDR_RMA_TRIGGER_RISK_SPIKE_PCT          - Trigger: risk spike threshold pct
    GL_EUDR_RMA_ALERT_FATIGUE_WINDOW_H          - Alert fatigue window hours
    GL_EUDR_RMA_OPTIMIZATION_TIMEOUT_S          - LP optimization timeout seconds
    GL_EUDR_RMA_MAX_SUPPLIERS_OPTIMIZE          - Max suppliers in optimization
    GL_EUDR_RMA_STAKEHOLDER_ROLES_COUNT         - Number of stakeholder roles
    GL_EUDR_RMA_OUTPUT_FORMATS                  - Comma-separated output formats
    GL_EUDR_RMA_DEFAULT_LANGUAGE                - Default report language
    GL_EUDR_RMA_SUPPORTED_LANGUAGES             - Comma-separated languages
    GL_EUDR_RMA_BATCH_MAX_SIZE                  - Batch processing max size
    GL_EUDR_RMA_BATCH_CONCURRENCY               - Batch concurrency workers
    GL_EUDR_RMA_BATCH_TIMEOUT_S                 - Batch timeout seconds
    GL_EUDR_RMA_RETENTION_YEARS                 - Data retention years
    GL_EUDR_RMA_ENABLE_PROVENANCE               - Enable provenance tracking
    GL_EUDR_RMA_GENESIS_HASH                    - Genesis hash anchor
    GL_EUDR_RMA_CHAIN_ALGORITHM                 - Hash algorithm
    GL_EUDR_RMA_ENABLE_METRICS                  - Enable Prometheus metrics
    GL_EUDR_RMA_METRICS_PREFIX                  - Prometheus metrics prefix
    GL_EUDR_RMA_RATE_LIMIT_ANONYMOUS            - Rate limit anonymous tier
    GL_EUDR_RMA_RATE_LIMIT_BASIC                - Rate limit basic tier
    GL_EUDR_RMA_RATE_LIMIT_STANDARD             - Rate limit standard tier
    GL_EUDR_RMA_RATE_LIMIT_PREMIUM              - Rate limit premium tier
    GL_EUDR_RMA_RATE_LIMIT_ADMIN                - Rate limit admin tier
    GL_EUDR_RMA_ESCALATION_CHAIN_24H            - Escalation chain 24h target
    GL_EUDR_RMA_ESCALATION_CHAIN_48H            - Escalation chain 48h target
    GL_EUDR_RMA_ESCALATION_CHAIN_72H            - Escalation chain 72h target
    GL_EUDR_RMA_BUDGET_DEFAULT_EUR              - Default mitigation budget EUR
    GL_EUDR_RMA_ROI_PENALTY_EXPOSURE_EUR        - Penalty exposure EUR for ROI

Example:
    >>> from greenlang.agents.eudr.risk_mitigation_advisor.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.ml_confidence_threshold, cfg.top_k_strategies)
    0.7 5

    >>> # Override for testing
    >>> from greenlang.agents.eudr.risk_mitigation_advisor.config import (
    ...     set_config, reset_config, RiskMitigationAdvisorConfig,
    ... )
    >>> set_config(RiskMitigationAdvisorConfig(deterministic_mode=True))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
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

_ENV_PREFIX = "GL_EUDR_RMA_"

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
# Valid ML model types
# ---------------------------------------------------------------------------

_VALID_ML_MODEL_TYPES = frozenset({"xgboost", "lightgbm", "rule_based"})

# ---------------------------------------------------------------------------
# Valid output formats
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "xlsx", "xml", "csv"})

# ---------------------------------------------------------------------------
# Valid report languages
# ---------------------------------------------------------------------------

_VALID_REPORT_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})

# ---------------------------------------------------------------------------
# Valid plan statuses
# ---------------------------------------------------------------------------

_VALID_PLAN_STATUSES = frozenset({
    "draft", "active", "on_track", "at_risk",
    "delayed", "completed", "suspended", "abandoned",
})

# ---------------------------------------------------------------------------
# Risk category constants
# ---------------------------------------------------------------------------

RISK_CATEGORIES = frozenset({
    "country", "supplier", "commodity", "corruption",
    "deforestation", "indigenous_rights", "protected_areas",
    "legal_compliance",
})

# ---------------------------------------------------------------------------
# EUDR regulated commodities
# ---------------------------------------------------------------------------

EUDR_COMMODITIES = frozenset({
    "cattle", "cocoa", "coffee", "palm_oil",
    "rubber", "soya", "wood",
})

# ---------------------------------------------------------------------------
# ISO 31000 treatment types
# ---------------------------------------------------------------------------

ISO_31000_TREATMENT_TYPES = frozenset({
    "avoid", "reduce", "share", "retain",
})

# ---------------------------------------------------------------------------
# Default ML parameters
# ---------------------------------------------------------------------------

_DEFAULT_ML_CONFIDENCE_THRESHOLD = Decimal("0.7")
_DEFAULT_TOP_K_STRATEGIES: int = 5
_DEFAULT_ML_MODEL_TYPE: str = "xgboost"

# ---------------------------------------------------------------------------
# Default effectiveness parameters
# ---------------------------------------------------------------------------

_DEFAULT_EFFECTIVENESS_INTERVAL_DAYS: int = 30
_DEFAULT_SIGNIFICANCE_LEVEL = Decimal("0.05")
_DEFAULT_ROI_PENALTY_EXPOSURE = Decimal("1000000")

# ---------------------------------------------------------------------------
# Default monitoring parameters
# ---------------------------------------------------------------------------

_DEFAULT_MONITORING_INTERVAL_S: int = 300
_DEFAULT_TRIGGER_RISK_SPIKE_PCT = Decimal("20")
_DEFAULT_ALERT_FATIGUE_WINDOW_H: int = 24
_DEFAULT_ADAPTIVE_RESPONSE_SLA_H: int = 48

# ---------------------------------------------------------------------------
# Default optimization parameters
# ---------------------------------------------------------------------------

_DEFAULT_OPTIMIZATION_TIMEOUT_S: int = 30
_DEFAULT_MAX_SUPPLIERS_OPTIMIZE: int = 500
_DEFAULT_BUDGET_EUR = Decimal("100000")

# ---------------------------------------------------------------------------
# Default capacity building parameters
# ---------------------------------------------------------------------------

_DEFAULT_CAPACITY_TIERS: int = 4
_DEFAULT_MODULES_PER_COMMODITY: int = 22

# ---------------------------------------------------------------------------
# Escalation chain defaults
# ---------------------------------------------------------------------------

_DEFAULT_ESCALATION_24H: str = "team_lead"
_DEFAULT_ESCALATION_48H: str = "director"
_DEFAULT_ESCALATION_72H: str = "executive"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RiskMitigationAdvisorConfig:
    """Configuration for the Risk Mitigation Advisor Agent (AGENT-EUDR-025).

    This dataclass encapsulates all configuration settings for ML-powered
    strategy recommendation, remediation plan design, supplier capacity
    building, mitigation measure library management, effectiveness tracking,
    continuous monitoring and adaptive management, cost-benefit optimization,
    stakeholder collaboration, and mitigation reporting. All settings have
    sensible defaults aligned with EUDR requirements and can be overridden
    via environment variables with the GL_EUDR_RMA_ prefix.

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

        # ML model settings
        ml_model_type: ML model type (xgboost, lightgbm, rule_based)
        ml_confidence_threshold: Min confidence for ML recommendations
        deterministic_mode: Force deterministic rule-based mode
        top_k_strategies: Number of top strategies to recommend
        shap_enabled: Enable SHAP explainability values
        model_version: Current ML model version string

        # Strategy selection settings
        min_risk_score_for_mitigation: Minimum risk score triggering mitigation
        composite_weight_country: Weight for country risk in composite
        composite_weight_supplier: Weight for supplier risk in composite
        composite_weight_commodity: Weight for commodity risk in composite
        composite_weight_corruption: Weight for corruption risk in composite
        composite_weight_deforestation: Weight for deforestation risk in composite
        composite_weight_indigenous: Weight for indigenous rights in composite
        composite_weight_protected: Weight for protected areas in composite
        composite_weight_legal: Weight for legal compliance in composite

        # Remediation plan settings
        plan_phases_count: Number of plan phases
        plan_max_duration_weeks: Maximum plan duration in weeks
        smart_milestones_enabled: Enable SMART milestone auto-generation
        plan_versioning_enabled: Enable plan versioning
        plan_approval_required: Require approval for plan activation

        # Capacity building settings
        capacity_tiers: Number of capacity building tiers
        modules_per_commodity: Training modules per commodity
        competency_pass_threshold: Minimum score to pass competency gate

        # Measure library settings
        measure_library_min_size: Minimum measures in library
        fulltext_search_enabled: Enable PostgreSQL full-text search
        measure_update_cycle_days: Measure update cycle in days

        # Effectiveness tracking settings
        effectiveness_interval_days: Measurement interval in days
        significance_level: Statistical significance level (p-value)
        roi_penalty_exposure_eur: Penalty exposure EUR for ROI calculation
        underperformance_threshold_pct: Threshold for underperformance flagging
        baseline_snapshot_enabled: Enable automatic baseline snapshots

        # Continuous monitoring settings
        monitoring_interval_s: Monitoring check interval seconds
        trigger_risk_spike_pct: Risk spike percentage triggering adjustment
        alert_fatigue_window_h: Alert fatigue prevention window hours
        adaptive_response_sla_h: SLA for adaptive response generation
        escalation_24h_target: 24h escalation target role
        escalation_48h_target: 48h escalation target role
        escalation_72h_target: 72h escalation target role

        # Cost-benefit optimization settings
        optimization_timeout_s: LP solver timeout seconds
        max_suppliers_optimize: Maximum suppliers in optimization batch
        default_budget_eur: Default mitigation budget EUR
        pareto_points: Number of points on Pareto frontier
        rice_weights_enabled: Enable RICE framework weighting

        # Stakeholder collaboration settings
        stakeholder_roles_count: Number of stakeholder roles
        supplier_portal_enabled: Enable supplier self-service portal
        ngo_workspace_enabled: Enable NGO partnership workspace
        communication_threads_enabled: Enable threaded communication
        document_sharing_enabled: Enable document sharing with versioning

        # Reporting settings
        output_formats: Report output formats
        default_language: Default report language
        supported_languages: Supported report languages
        report_retention_years: Report archive retention years
        report_scheduling_enabled: Enable scheduled report generation

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

        # Rate limiting (requests per minute)
        rate_limit_anonymous: Rate limit anonymous tier
        rate_limit_basic: Rate limit basic tier
        rate_limit_standard: Rate limit standard tier
        rate_limit_premium: Rate limit premium tier
        rate_limit_admin: Rate limit admin tier
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
    redis_key_prefix: str = "gl:eudr:rma:"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # ML model settings
    # -----------------------------------------------------------------------
    ml_model_type: str = _DEFAULT_ML_MODEL_TYPE
    ml_confidence_threshold: Decimal = _DEFAULT_ML_CONFIDENCE_THRESHOLD
    deterministic_mode: bool = False
    top_k_strategies: int = _DEFAULT_TOP_K_STRATEGIES
    shap_enabled: bool = True
    model_version: str = "1.0.0"

    # -----------------------------------------------------------------------
    # Strategy selection settings
    # -----------------------------------------------------------------------
    min_risk_score_for_mitigation: Decimal = Decimal("30")
    composite_weight_country: Decimal = Decimal("0.15")
    composite_weight_supplier: Decimal = Decimal("0.15")
    composite_weight_commodity: Decimal = Decimal("0.10")
    composite_weight_corruption: Decimal = Decimal("0.10")
    composite_weight_deforestation: Decimal = Decimal("0.20")
    composite_weight_indigenous: Decimal = Decimal("0.10")
    composite_weight_protected: Decimal = Decimal("0.10")
    composite_weight_legal: Decimal = Decimal("0.10")

    # -----------------------------------------------------------------------
    # Remediation plan settings
    # -----------------------------------------------------------------------
    plan_phases_count: int = 4
    plan_max_duration_weeks: int = 52
    smart_milestones_enabled: bool = True
    plan_versioning_enabled: bool = True
    plan_approval_required: bool = True

    # -----------------------------------------------------------------------
    # Capacity building settings
    # -----------------------------------------------------------------------
    capacity_tiers: int = _DEFAULT_CAPACITY_TIERS
    modules_per_commodity: int = _DEFAULT_MODULES_PER_COMMODITY
    competency_pass_threshold: Decimal = Decimal("70")

    # -----------------------------------------------------------------------
    # Measure library settings
    # -----------------------------------------------------------------------
    measure_library_min_size: int = 500
    fulltext_search_enabled: bool = True
    measure_update_cycle_days: int = 90

    # -----------------------------------------------------------------------
    # Effectiveness tracking settings
    # -----------------------------------------------------------------------
    effectiveness_interval_days: int = _DEFAULT_EFFECTIVENESS_INTERVAL_DAYS
    significance_level: Decimal = _DEFAULT_SIGNIFICANCE_LEVEL
    roi_penalty_exposure_eur: Decimal = _DEFAULT_ROI_PENALTY_EXPOSURE
    underperformance_threshold_pct: Decimal = Decimal("50")
    baseline_snapshot_enabled: bool = True

    # -----------------------------------------------------------------------
    # Continuous monitoring settings
    # -----------------------------------------------------------------------
    monitoring_interval_s: int = _DEFAULT_MONITORING_INTERVAL_S
    trigger_risk_spike_pct: Decimal = _DEFAULT_TRIGGER_RISK_SPIKE_PCT
    alert_fatigue_window_h: int = _DEFAULT_ALERT_FATIGUE_WINDOW_H
    adaptive_response_sla_h: int = _DEFAULT_ADAPTIVE_RESPONSE_SLA_H
    escalation_24h_target: str = _DEFAULT_ESCALATION_24H
    escalation_48h_target: str = _DEFAULT_ESCALATION_48H
    escalation_72h_target: str = _DEFAULT_ESCALATION_72H

    # -----------------------------------------------------------------------
    # Cost-benefit optimization settings
    # -----------------------------------------------------------------------
    optimization_timeout_s: int = _DEFAULT_OPTIMIZATION_TIMEOUT_S
    max_suppliers_optimize: int = _DEFAULT_MAX_SUPPLIERS_OPTIMIZE
    default_budget_eur: Decimal = _DEFAULT_BUDGET_EUR
    pareto_points: int = 20
    rice_weights_enabled: bool = True

    # -----------------------------------------------------------------------
    # Stakeholder collaboration settings
    # -----------------------------------------------------------------------
    stakeholder_roles_count: int = 6
    supplier_portal_enabled: bool = True
    ngo_workspace_enabled: bool = True
    communication_threads_enabled: bool = True
    document_sharing_enabled: bool = True

    # -----------------------------------------------------------------------
    # Reporting settings
    # -----------------------------------------------------------------------
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "xlsx", "xml"]
    )
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )
    report_retention_years: int = 5
    report_scheduling_enabled: bool = True

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
    genesis_hash: str = "GL-EUDR-RMA-025-RISK-MITIGATION-ADVISOR-GENESIS"
    chain_algorithm: str = "sha256"

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_rma_"

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
        self._validate_ml_settings()
        self._validate_composite_weights()
        self._validate_plan_settings()
        self._validate_capacity_settings()
        self._validate_effectiveness_settings()
        self._validate_monitoring_settings()
        self._validate_optimization_settings()
        self._validate_positive_integers()
        self._validate_output_formats()
        self._validate_languages()

        logger.info(
            f"RiskMitigationAdvisorConfig initialized: "
            f"ml_model={self.ml_model_type}, "
            f"confidence={self.ml_confidence_threshold}, "
            f"deterministic={self.deterministic_mode}, "
            f"top_k={self.top_k_strategies}"
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

    def _validate_ml_settings(self) -> None:
        """Validate ML model configuration parameters."""
        if self.ml_model_type not in _VALID_ML_MODEL_TYPES:
            raise ValueError(
                f"Invalid ml_model_type: {self.ml_model_type}. "
                f"Must be one of {_VALID_ML_MODEL_TYPES}"
            )
        if not (Decimal("0") < self.ml_confidence_threshold <= Decimal("1")):
            raise ValueError(
                f"ml_confidence_threshold must be between 0 (exclusive) and "
                f"1 (inclusive), got {self.ml_confidence_threshold}"
            )
        if self.top_k_strategies < 1 or self.top_k_strategies > 20:
            raise ValueError(
                f"top_k_strategies must be between 1 and 20, "
                f"got {self.top_k_strategies}"
            )

    def _validate_composite_weights(self) -> None:
        """Validate risk composite weights sum to 1.0."""
        weights = [
            ("country", self.composite_weight_country),
            ("supplier", self.composite_weight_supplier),
            ("commodity", self.composite_weight_commodity),
            ("corruption", self.composite_weight_corruption),
            ("deforestation", self.composite_weight_deforestation),
            ("indigenous", self.composite_weight_indigenous),
            ("protected", self.composite_weight_protected),
            ("legal", self.composite_weight_legal),
        ]
        for name, w in weights:
            if w < Decimal("0") or w > Decimal("1"):
                raise ValueError(
                    f"composite_weight_{name} must be between 0 and 1, got {w}"
                )
        total = sum(w for _, w in weights)
        if abs(total - Decimal("1")) > Decimal("0.001"):
            raise ValueError(
                f"Composite risk weights must sum to 1.0, got {total}"
            )

    def _validate_plan_settings(self) -> None:
        """Validate remediation plan parameters."""
        if self.plan_phases_count < 1 or self.plan_phases_count > 10:
            raise ValueError(
                f"plan_phases_count must be between 1 and 10, "
                f"got {self.plan_phases_count}"
            )
        if self.plan_max_duration_weeks < 1:
            raise ValueError(
                f"plan_max_duration_weeks must be >= 1, "
                f"got {self.plan_max_duration_weeks}"
            )

    def _validate_capacity_settings(self) -> None:
        """Validate capacity building parameters."""
        if self.capacity_tiers < 1 or self.capacity_tiers > 10:
            raise ValueError(
                f"capacity_tiers must be between 1 and 10, "
                f"got {self.capacity_tiers}"
            )
        if self.modules_per_commodity < 1:
            raise ValueError(
                f"modules_per_commodity must be >= 1, "
                f"got {self.modules_per_commodity}"
            )
        if not (Decimal("0") <= self.competency_pass_threshold <= Decimal("100")):
            raise ValueError(
                f"competency_pass_threshold must be between 0 and 100, "
                f"got {self.competency_pass_threshold}"
            )

    def _validate_effectiveness_settings(self) -> None:
        """Validate effectiveness tracking parameters."""
        if self.effectiveness_interval_days < 1:
            raise ValueError(
                f"effectiveness_interval_days must be >= 1, "
                f"got {self.effectiveness_interval_days}"
            )
        if not (Decimal("0") < self.significance_level < Decimal("1")):
            raise ValueError(
                f"significance_level must be between 0 and 1 (exclusive), "
                f"got {self.significance_level}"
            )
        if self.roi_penalty_exposure_eur <= Decimal("0"):
            raise ValueError(
                f"roi_penalty_exposure_eur must be > 0, "
                f"got {self.roi_penalty_exposure_eur}"
            )
        if not (Decimal("0") < self.underperformance_threshold_pct <= Decimal("100")):
            raise ValueError(
                f"underperformance_threshold_pct must be between 0 and 100, "
                f"got {self.underperformance_threshold_pct}"
            )

    def _validate_monitoring_settings(self) -> None:
        """Validate continuous monitoring parameters."""
        if self.monitoring_interval_s < 1:
            raise ValueError(
                f"monitoring_interval_s must be >= 1, "
                f"got {self.monitoring_interval_s}"
            )
        if self.trigger_risk_spike_pct <= Decimal("0"):
            raise ValueError(
                f"trigger_risk_spike_pct must be > 0, "
                f"got {self.trigger_risk_spike_pct}"
            )
        if self.alert_fatigue_window_h < 1:
            raise ValueError(
                f"alert_fatigue_window_h must be >= 1, "
                f"got {self.alert_fatigue_window_h}"
            )
        if self.adaptive_response_sla_h < 1:
            raise ValueError(
                f"adaptive_response_sla_h must be >= 1, "
                f"got {self.adaptive_response_sla_h}"
            )

    def _validate_optimization_settings(self) -> None:
        """Validate cost-benefit optimization parameters."""
        if self.optimization_timeout_s < 1:
            raise ValueError(
                f"optimization_timeout_s must be >= 1, "
                f"got {self.optimization_timeout_s}"
            )
        if self.max_suppliers_optimize < 1:
            raise ValueError(
                f"max_suppliers_optimize must be >= 1, "
                f"got {self.max_suppliers_optimize}"
            )
        if self.default_budget_eur <= Decimal("0"):
            raise ValueError(
                f"default_budget_eur must be > 0, "
                f"got {self.default_budget_eur}"
            )
        if self.pareto_points < 2:
            raise ValueError(
                f"pareto_points must be >= 2, got {self.pareto_points}"
            )

    def _validate_positive_integers(self) -> None:
        """Validate fields that must be positive integers."""
        checks = [
            ("pool_size", self.pool_size),
            ("batch_max_size", self.batch_max_size),
            ("batch_concurrency", self.batch_concurrency),
            ("retention_years", self.retention_years),
            ("report_retention_years", self.report_retention_years),
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
    def from_env(cls) -> "RiskMitigationAdvisorConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_RMA_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            RiskMitigationAdvisorConfig instance with env overrides applied.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_RMA_ML_CONFIDENCE_THRESHOLD"] = "0.8"
            >>> cfg = RiskMitigationAdvisorConfig.from_env()
            >>> assert cfg.ml_confidence_threshold == Decimal("0.8")
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

        # ML model settings
        if val := os.getenv(f"{_ENV_PREFIX}ML_MODEL_TYPE"):
            kwargs["ml_model_type"] = val.lower()
        if val := os.getenv(f"{_ENV_PREFIX}ML_CONFIDENCE_THRESHOLD"):
            kwargs["ml_confidence_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}DETERMINISTIC_MODE"):
            kwargs["deterministic_mode"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}TOP_K_STRATEGIES"):
            kwargs["top_k_strategies"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SHAP_ENABLED"):
            kwargs["shap_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}MODEL_VERSION"):
            kwargs["model_version"] = val

        # Strategy selection weights
        if val := os.getenv(f"{_ENV_PREFIX}MIN_RISK_SCORE_FOR_MITIGATION"):
            kwargs["min_risk_score_for_mitigation"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPOSITE_WEIGHT_COUNTRY"):
            kwargs["composite_weight_country"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPOSITE_WEIGHT_SUPPLIER"):
            kwargs["composite_weight_supplier"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPOSITE_WEIGHT_COMMODITY"):
            kwargs["composite_weight_commodity"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPOSITE_WEIGHT_CORRUPTION"):
            kwargs["composite_weight_corruption"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPOSITE_WEIGHT_DEFORESTATION"):
            kwargs["composite_weight_deforestation"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPOSITE_WEIGHT_INDIGENOUS"):
            kwargs["composite_weight_indigenous"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPOSITE_WEIGHT_PROTECTED"):
            kwargs["composite_weight_protected"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPOSITE_WEIGHT_LEGAL"):
            kwargs["composite_weight_legal"] = Decimal(val)

        # Plan settings
        if val := os.getenv(f"{_ENV_PREFIX}PLAN_PHASES_COUNT"):
            kwargs["plan_phases_count"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}PLAN_MAX_DURATION_WEEKS"):
            kwargs["plan_max_duration_weeks"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SMART_MILESTONES_ENABLED"):
            kwargs["smart_milestones_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}PLAN_VERSIONING_ENABLED"):
            kwargs["plan_versioning_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}PLAN_APPROVAL_REQUIRED"):
            kwargs["plan_approval_required"] = val.lower() in ("true", "1", "yes")

        # Capacity building settings
        if val := os.getenv(f"{_ENV_PREFIX}CAPACITY_TIERS"):
            kwargs["capacity_tiers"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MODULES_PER_COMMODITY"):
            kwargs["modules_per_commodity"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}COMPETENCY_PASS_THRESHOLD"):
            kwargs["competency_pass_threshold"] = Decimal(val)

        # Measure library settings
        if val := os.getenv(f"{_ENV_PREFIX}MEASURE_LIBRARY_MIN_SIZE"):
            kwargs["measure_library_min_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}FULLTEXT_SEARCH_ENABLED"):
            kwargs["fulltext_search_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}MEASURE_UPDATE_CYCLE_DAYS"):
            kwargs["measure_update_cycle_days"] = int(val)

        # Effectiveness settings
        if val := os.getenv(f"{_ENV_PREFIX}EFFECTIVENESS_INTERVAL_DAYS"):
            kwargs["effectiveness_interval_days"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SIGNIFICANCE_LEVEL"):
            kwargs["significance_level"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}ROI_PENALTY_EXPOSURE_EUR"):
            kwargs["roi_penalty_exposure_eur"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}UNDERPERFORMANCE_THRESHOLD_PCT"):
            kwargs["underperformance_threshold_pct"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}BASELINE_SNAPSHOT_ENABLED"):
            kwargs["baseline_snapshot_enabled"] = val.lower() in ("true", "1", "yes")

        # Monitoring settings
        if val := os.getenv(f"{_ENV_PREFIX}MONITORING_INTERVAL_S"):
            kwargs["monitoring_interval_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}TRIGGER_RISK_SPIKE_PCT"):
            kwargs["trigger_risk_spike_pct"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}ALERT_FATIGUE_WINDOW_H"):
            kwargs["alert_fatigue_window_h"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ADAPTIVE_RESPONSE_SLA_H"):
            kwargs["adaptive_response_sla_h"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}ESCALATION_CHAIN_24H"):
            kwargs["escalation_24h_target"] = val
        if val := os.getenv(f"{_ENV_PREFIX}ESCALATION_CHAIN_48H"):
            kwargs["escalation_48h_target"] = val
        if val := os.getenv(f"{_ENV_PREFIX}ESCALATION_CHAIN_72H"):
            kwargs["escalation_72h_target"] = val

        # Optimization settings
        if val := os.getenv(f"{_ENV_PREFIX}OPTIMIZATION_TIMEOUT_S"):
            kwargs["optimization_timeout_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_SUPPLIERS_OPTIMIZE"):
            kwargs["max_suppliers_optimize"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BUDGET_DEFAULT_EUR"):
            kwargs["default_budget_eur"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}PARETO_POINTS"):
            kwargs["pareto_points"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RICE_WEIGHTS_ENABLED"):
            kwargs["rice_weights_enabled"] = val.lower() in ("true", "1", "yes")

        # Stakeholder settings
        if val := os.getenv(f"{_ENV_PREFIX}STAKEHOLDER_ROLES_COUNT"):
            kwargs["stakeholder_roles_count"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}SUPPLIER_PORTAL_ENABLED"):
            kwargs["supplier_portal_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}NGO_WORKSPACE_ENABLED"):
            kwargs["ngo_workspace_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}COMMUNICATION_THREADS_ENABLED"):
            kwargs["communication_threads_enabled"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}DOCUMENT_SHARING_ENABLED"):
            kwargs["document_sharing_enabled"] = val.lower() in ("true", "1", "yes")

        # Reporting
        if val := os.getenv(f"{_ENV_PREFIX}OUTPUT_FORMATS"):
            kwargs["output_formats"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_LANGUAGE"):
            kwargs["default_language"] = val
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_LANGUAGES"):
            kwargs["supported_languages"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}REPORT_RETENTION_YEARS"):
            kwargs["report_retention_years"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}REPORT_SCHEDULING_ENABLED"):
            kwargs["report_scheduling_enabled"] = val.lower() in ("true", "1", "yes")

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
            >>> cfg = RiskMitigationAdvisorConfig()
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
            # ML settings
            "ml_model_type": self.ml_model_type,
            "ml_confidence_threshold": str(self.ml_confidence_threshold),
            "deterministic_mode": self.deterministic_mode,
            "top_k_strategies": self.top_k_strategies,
            "shap_enabled": self.shap_enabled,
            "model_version": self.model_version,
            # Strategy selection weights
            "min_risk_score_for_mitigation": str(self.min_risk_score_for_mitigation),
            "composite_weight_country": str(self.composite_weight_country),
            "composite_weight_supplier": str(self.composite_weight_supplier),
            "composite_weight_commodity": str(self.composite_weight_commodity),
            "composite_weight_corruption": str(self.composite_weight_corruption),
            "composite_weight_deforestation": str(self.composite_weight_deforestation),
            "composite_weight_indigenous": str(self.composite_weight_indigenous),
            "composite_weight_protected": str(self.composite_weight_protected),
            "composite_weight_legal": str(self.composite_weight_legal),
            # Plan settings
            "plan_phases_count": self.plan_phases_count,
            "plan_max_duration_weeks": self.plan_max_duration_weeks,
            "smart_milestones_enabled": self.smart_milestones_enabled,
            "plan_versioning_enabled": self.plan_versioning_enabled,
            "plan_approval_required": self.plan_approval_required,
            # Capacity building
            "capacity_tiers": self.capacity_tiers,
            "modules_per_commodity": self.modules_per_commodity,
            "competency_pass_threshold": str(self.competency_pass_threshold),
            # Measure library
            "measure_library_min_size": self.measure_library_min_size,
            "fulltext_search_enabled": self.fulltext_search_enabled,
            "measure_update_cycle_days": self.measure_update_cycle_days,
            # Effectiveness tracking
            "effectiveness_interval_days": self.effectiveness_interval_days,
            "significance_level": str(self.significance_level),
            "roi_penalty_exposure_eur": str(self.roi_penalty_exposure_eur),
            "underperformance_threshold_pct": str(self.underperformance_threshold_pct),
            "baseline_snapshot_enabled": self.baseline_snapshot_enabled,
            # Monitoring
            "monitoring_interval_s": self.monitoring_interval_s,
            "trigger_risk_spike_pct": str(self.trigger_risk_spike_pct),
            "alert_fatigue_window_h": self.alert_fatigue_window_h,
            "adaptive_response_sla_h": self.adaptive_response_sla_h,
            "escalation_24h_target": self.escalation_24h_target,
            "escalation_48h_target": self.escalation_48h_target,
            "escalation_72h_target": self.escalation_72h_target,
            # Optimization
            "optimization_timeout_s": self.optimization_timeout_s,
            "max_suppliers_optimize": self.max_suppliers_optimize,
            "default_budget_eur": str(self.default_budget_eur),
            "pareto_points": self.pareto_points,
            "rice_weights_enabled": self.rice_weights_enabled,
            # Stakeholder
            "stakeholder_roles_count": self.stakeholder_roles_count,
            "supplier_portal_enabled": self.supplier_portal_enabled,
            "ngo_workspace_enabled": self.ngo_workspace_enabled,
            "communication_threads_enabled": self.communication_threads_enabled,
            "document_sharing_enabled": self.document_sharing_enabled,
            # Reporting
            "output_formats": self.output_formats,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "report_retention_years": self.report_retention_years,
            "report_scheduling_enabled": self.report_scheduling_enabled,
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
_global_config: Optional[RiskMitigationAdvisorConfig] = None


def get_config() -> RiskMitigationAdvisorConfig:
    """Get the global RiskMitigationAdvisorConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance. Uses double-checked locking
    to minimize contention after initialization.

    Returns:
        RiskMitigationAdvisorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> assert cfg.ml_confidence_threshold == Decimal("0.7")
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2  # Same instance
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = RiskMitigationAdvisorConfig.from_env()
    return _global_config


def set_config(config: RiskMitigationAdvisorConfig) -> None:
    """Set the global RiskMitigationAdvisorConfig singleton instance.

    Used for testing and programmatic configuration override.

    Args:
        config: RiskMitigationAdvisorConfig instance to set as global.

    Example:
        >>> from greenlang.agents.eudr.risk_mitigation_advisor.config import (
        ...     set_config, RiskMitigationAdvisorConfig,
        ... )
        >>> test_cfg = RiskMitigationAdvisorConfig(deterministic_mode=True)
        >>> set_config(test_cfg)
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global RiskMitigationAdvisorConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_config()
        >>> # Next get_config() call will re-initialize from environment
    """
    global _global_config
    with _config_lock:
        _global_config = None
