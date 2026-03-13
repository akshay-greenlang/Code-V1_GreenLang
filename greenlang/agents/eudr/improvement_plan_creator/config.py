# -*- coding: utf-8 -*-
"""
Improvement Plan Creator Configuration - AGENT-EUDR-035

Centralized configuration for the Improvement Plan Creator covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Finding aggregation: source agent limits, deduplication thresholds
- Gap analysis: severity thresholds, compliance framework mappings
- Action generation: SMART criteria, action type templates, max actions
- Root cause analysis: 5-Whys depth, fishbone category limits
- Prioritization: Eisenhower matrix weights, risk-based scoring factors
- Progress tracking: milestone intervals, effectiveness review periods
- Stakeholder coordination: RACI assignment rules, notification channels
- Upstream agent URLs: EUDR-028 to EUDR-034 (risk/mitigation agents)
- Report generation settings: format, evidence, provenance
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_ipc_

All settings overridable via environment variables with ``GL_EUDR_IPC_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 Improvement Plan Creator (GL-EUDR-IPC-035)
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 12, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV_PREFIX = "GL_EUDR_IPC_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_IPC_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_IPC_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_IPC_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_IPC_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_IPC_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class ImprovementPlanCreatorConfig:
    """Centralized configuration for AGENT-EUDR-035.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_IPC_ environment variables.
    """

    # -- 1. Database ------------------------------------------------------------
    db_host: str = field(
        default_factory=lambda: _env("DB_HOST", "localhost")
    )
    db_port: int = field(
        default_factory=lambda: _env_int("DB_PORT", 5432)
    )
    db_name: str = field(
        default_factory=lambda: _env("DB_NAME", "greenlang")
    )
    db_user: str = field(
        default_factory=lambda: _env("DB_USER", "gl")
    )
    db_password: str = field(
        default_factory=lambda: _env("DB_PASSWORD", "gl")
    )
    db_pool_min: int = field(
        default_factory=lambda: _env_int("DB_POOL_MIN", 2)
    )
    db_pool_max: int = field(
        default_factory=lambda: _env_int("DB_POOL_MAX", 10)
    )

    # -- 2. Redis ---------------------------------------------------------------
    redis_host: str = field(
        default_factory=lambda: _env("REDIS_HOST", "localhost")
    )
    redis_port: int = field(
        default_factory=lambda: _env_int("REDIS_PORT", 6379)
    )
    redis_db: int = field(
        default_factory=lambda: _env_int("REDIS_DB", 0)
    )
    redis_password: str = field(
        default_factory=lambda: _env("REDIS_PASSWORD", "")
    )
    cache_ttl: int = field(
        default_factory=lambda: _env_int("CACHE_TTL", 3600)
    )

    # -- 3. Finding Aggregation -------------------------------------------------
    max_source_agents: int = field(
        default_factory=lambda: _env_int("MAX_SOURCE_AGENTS", 20)
    )
    deduplication_similarity_threshold: Decimal = field(
        default_factory=lambda: _env_decimal(
            "DEDUP_SIMILARITY_THRESHOLD", "0.85"
        )
    )
    finding_staleness_days: int = field(
        default_factory=lambda: _env_int("FINDING_STALENESS_DAYS", 90)
    )
    max_findings_per_aggregation: int = field(
        default_factory=lambda: _env_int("MAX_FINDINGS_PER_AGGREGATION", 500)
    )
    finding_confidence_threshold: Decimal = field(
        default_factory=lambda: _env_decimal(
            "FINDING_CONFIDENCE_THRESHOLD", "0.60"
        )
    )

    # -- 4. Gap Analysis --------------------------------------------------------
    gap_severity_critical_threshold: Decimal = field(
        default_factory=lambda: _env_decimal(
            "GAP_SEVERITY_CRITICAL", "0.80"
        )
    )
    gap_severity_high_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("GAP_SEVERITY_HIGH", "0.60")
    )
    gap_severity_medium_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("GAP_SEVERITY_MEDIUM", "0.40")
    )
    gap_severity_low_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("GAP_SEVERITY_LOW", "0.20")
    )
    max_gaps_per_analysis: int = field(
        default_factory=lambda: _env_int("MAX_GAPS_PER_ANALYSIS", 200)
    )
    compliance_framework_version: str = field(
        default_factory=lambda: _env(
            "COMPLIANCE_FRAMEWORK_VERSION", "EU-2023-1115-v1"
        )
    )

    # -- 5. Action Generation ---------------------------------------------------
    max_actions_per_plan: int = field(
        default_factory=lambda: _env_int("MAX_ACTIONS_PER_PLAN", 50)
    )
    default_action_deadline_days: int = field(
        default_factory=lambda: _env_int("DEFAULT_ACTION_DEADLINE_DAYS", 30)
    )
    smart_validation_enabled: bool = field(
        default_factory=lambda: _env_bool("SMART_VALIDATION_ENABLED", True)
    )
    action_cost_estimation_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "ACTION_COST_ESTIMATION_ENABLED", True
        )
    )
    min_actions_per_critical_gap: int = field(
        default_factory=lambda: _env_int("MIN_ACTIONS_PER_CRITICAL_GAP", 2)
    )

    # -- 6. Root Cause Analysis -------------------------------------------------
    five_whys_max_depth: int = field(
        default_factory=lambda: _env_int("FIVE_WHYS_MAX_DEPTH", 5)
    )
    fishbone_max_categories: int = field(
        default_factory=lambda: _env_int("FISHBONE_MAX_CATEGORIES", 8)
    )
    fishbone_max_causes_per_category: int = field(
        default_factory=lambda: _env_int(
            "FISHBONE_MAX_CAUSES_PER_CAT", 10
        )
    )
    root_cause_confidence_threshold: Decimal = field(
        default_factory=lambda: _env_decimal(
            "ROOT_CAUSE_CONFIDENCE_THRESHOLD", "0.70"
        )
    )

    # -- 7. Prioritization Engine -----------------------------------------------
    eisenhower_urgency_weight: Decimal = field(
        default_factory=lambda: _env_decimal(
            "EISENHOWER_URGENCY_WEIGHT", "0.50"
        )
    )
    eisenhower_importance_weight: Decimal = field(
        default_factory=lambda: _env_decimal(
            "EISENHOWER_IMPORTANCE_WEIGHT", "0.50"
        )
    )
    risk_score_weight: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_SCORE_WEIGHT", "0.30")
    )
    compliance_impact_weight: Decimal = field(
        default_factory=lambda: _env_decimal(
            "COMPLIANCE_IMPACT_WEIGHT", "0.25"
        )
    )
    resource_efficiency_weight: Decimal = field(
        default_factory=lambda: _env_decimal(
            "RESOURCE_EFFICIENCY_WEIGHT", "0.20"
        )
    )
    stakeholder_impact_weight: Decimal = field(
        default_factory=lambda: _env_decimal(
            "STAKEHOLDER_IMPACT_WEIGHT", "0.15"
        )
    )
    time_sensitivity_weight: Decimal = field(
        default_factory=lambda: _env_decimal(
            "TIME_SENSITIVITY_WEIGHT", "0.10"
        )
    )

    # -- 8. Progress Tracking ---------------------------------------------------
    milestone_check_interval_days: int = field(
        default_factory=lambda: _env_int("MILESTONE_CHECK_INTERVAL_DAYS", 7)
    )
    effectiveness_review_period_days: int = field(
        default_factory=lambda: _env_int(
            "EFFECTIVENESS_REVIEW_PERIOD_DAYS", 30
        )
    )
    overdue_alert_threshold_days: int = field(
        default_factory=lambda: _env_int("OVERDUE_ALERT_THRESHOLD_DAYS", 7)
    )
    max_extensions_per_action: int = field(
        default_factory=lambda: _env_int("MAX_EXTENSIONS_PER_ACTION", 3)
    )
    auto_escalation_enabled: bool = field(
        default_factory=lambda: _env_bool("AUTO_ESCALATION_ENABLED", True)
    )
    escalation_threshold_days: int = field(
        default_factory=lambda: _env_int("ESCALATION_THRESHOLD_DAYS", 14)
    )

    # -- 9. Stakeholder Coordination -------------------------------------------
    max_stakeholders_per_action: int = field(
        default_factory=lambda: _env_int("MAX_STAKEHOLDERS_PER_ACTION", 10)
    )
    notification_channels_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "NOTIFICATION_CHANNELS_ENABLED", True
        )
    )
    raci_validation_enabled: bool = field(
        default_factory=lambda: _env_bool("RACI_VALIDATION_ENABLED", True)
    )
    default_notification_channel: str = field(
        default_factory=lambda: _env(
            "DEFAULT_NOTIFICATION_CHANNEL", "email"
        )
    )

    # -- 10. Upstream Agent URLs ------------------------------------------------
    risk_assessment_url: str = field(
        default_factory=lambda: _env(
            "RISK_ASSESSMENT_URL",
            "http://eudr-risk-assessment:8028/api/v1/eudr/risk-assessment",
        )
    )
    mitigation_measure_url: str = field(
        default_factory=lambda: _env(
            "MITIGATION_MEASURE_URL",
            "http://eudr-mitigation-measure:8029/api/v1/eudr/mitigation-measure-designer",
        )
    )
    documentation_generator_url: str = field(
        default_factory=lambda: _env(
            "DOCUMENTATION_GENERATOR_URL",
            "http://eudr-documentation-generator:8030/api/v1/eudr/documentation-generator",
        )
    )
    due_diligence_orchestrator_url: str = field(
        default_factory=lambda: _env(
            "DUE_DILIGENCE_ORCHESTRATOR_URL",
            "http://eudr-due-diligence:8026/api/v1/eudr/due-diligence",
        )
    )
    country_risk_url: str = field(
        default_factory=lambda: _env(
            "COUNTRY_RISK_URL",
            "http://eudr-country-risk:8016/api/v1/eudr/country-risk",
        )
    )
    supplier_risk_url: str = field(
        default_factory=lambda: _env(
            "SUPPLIER_RISK_URL",
            "http://eudr-supplier-risk:8017/api/v1/eudr/supplier-risk",
        )
    )

    # -- 11. Report Generation --------------------------------------------------
    report_format: str = field(
        default_factory=lambda: _env("REPORT_FORMAT", "json")
    )
    include_evidence_summary: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_EVIDENCE_SUMMARY", True)
    )
    include_provenance: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_PROVENANCE", True)
    )
    include_gantt_data: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_GANTT_DATA", True)
    )

    # -- 12. Rate Limiting ------------------------------------------------------
    rate_limit_anonymous: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ANONYMOUS", 10)
    )
    rate_limit_basic: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_BASIC", 30)
    )
    rate_limit_standard: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_STANDARD", 100)
    )
    rate_limit_premium: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PREMIUM", 500)
    )
    rate_limit_admin: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ADMIN", 2000)
    )

    # -- 13. Circuit Breaker ----------------------------------------------------
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5)
    )
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _env_int("CB_RESET_TIMEOUT", 60)
    )
    circuit_breaker_half_open_max: int = field(
        default_factory=lambda: _env_int("CB_HALF_OPEN_MAX", 3)
    )

    # -- 14. Batch Processing ---------------------------------------------------
    max_concurrent: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT", 10)
    )
    batch_timeout_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_TIMEOUT", 300)
    )

    # -- 15. Provenance ---------------------------------------------------------
    provenance_enabled: bool = field(
        default_factory=lambda: _env_bool("PROVENANCE_ENABLED", True)
    )
    provenance_algorithm: str = "sha256"
    provenance_chain_enabled: bool = field(
        default_factory=lambda: _env_bool("PROVENANCE_CHAIN_ENABLED", True)
    )
    provenance_genesis_hash: str = (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )

    # -- 16. Metrics ------------------------------------------------------------
    metrics_enabled: bool = field(
        default_factory=lambda: _env_bool("METRICS_ENABLED", True)
    )
    metrics_prefix: str = "gl_eudr_ipc_"

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate gap severity ordering
        if not (
            self.gap_severity_low_threshold
            < self.gap_severity_medium_threshold
            < self.gap_severity_high_threshold
            < self.gap_severity_critical_threshold
        ):
            logger.warning(
                "Gap severity thresholds are not in ascending order: "
                "low=%s, medium=%s, high=%s, critical=%s",
                self.gap_severity_low_threshold,
                self.gap_severity_medium_threshold,
                self.gap_severity_high_threshold,
                self.gap_severity_critical_threshold,
            )

        # Validate prioritization weights sum to 1.0
        total_weight = (
            self.risk_score_weight
            + self.compliance_impact_weight
            + self.resource_efficiency_weight
            + self.stakeholder_impact_weight
            + self.time_sensitivity_weight
        )
        if total_weight != Decimal("1.00"):
            logger.warning(
                "Prioritization weights do not sum to 1.00: total=%s "
                "(risk=%s, compliance=%s, resource=%s, stakeholder=%s, "
                "time=%s)",
                total_weight,
                self.risk_score_weight,
                self.compliance_impact_weight,
                self.resource_efficiency_weight,
                self.stakeholder_impact_weight,
                self.time_sensitivity_weight,
            )

        # Validate Eisenhower weights sum to 1.0
        eisenhower_total = (
            self.eisenhower_urgency_weight
            + self.eisenhower_importance_weight
        )
        if eisenhower_total != Decimal("1.00"):
            logger.warning(
                "Eisenhower weights do not sum to 1.00: total=%s "
                "(urgency=%s, importance=%s)",
                eisenhower_total,
                self.eisenhower_urgency_weight,
                self.eisenhower_importance_weight,
            )

        # Validate pool sizing
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min,
                self.db_pool_max,
            )

        # Validate 5-Whys depth
        if self.five_whys_max_depth < 1 or self.five_whys_max_depth > 10:
            logger.warning(
                "5-Whys max depth %d is outside recommended range [1, 10].",
                self.five_whys_max_depth,
            )

        logger.info(
            "ImprovementPlanCreatorConfig initialized: "
            "gap_thresholds (LOW<%s, MED<%s, HIGH<%s, CRIT<%s), "
            "max_actions_per_plan=%d, 5whys_depth=%d, "
            "smart_validation=%s, auto_escalation=%s, "
            "raci_validation=%s",
            self.gap_severity_low_threshold,
            self.gap_severity_medium_threshold,
            self.gap_severity_high_threshold,
            self.gap_severity_critical_threshold,
            self.max_actions_per_plan,
            self.five_whys_max_depth,
            self.smart_validation_enabled,
            self.auto_escalation_enabled,
            self.raci_validation_enabled,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[ImprovementPlanCreatorConfig] = None
_config_lock = threading.Lock()


def get_config() -> ImprovementPlanCreatorConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        ImprovementPlanCreatorConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ImprovementPlanCreatorConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
