# -*- coding: utf-8 -*-
"""
Annual Review Scheduler Agent Configuration - AGENT-EUDR-034

Centralized configuration for the Annual Review Scheduler Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Review cycle management: scheduling intervals, grace periods, auto-creation
- Deadline tracking: regulatory deadlines, submission windows, escalation thresholds
- Checklist generation: template loading, commodity-specific customization
- Entity coordination: cascade depth, dependency resolution, parallel limits
- Year-over-year comparison: significance thresholds, scoring weights
- Calendar management: event retention, iCal sync, timezone defaults
- Notification engine: channels, escalation tiers, retry policies
- Rate limiting, circuit breaker, batch processing
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_ars_

All settings overridable via environment variables with ``GL_EUDR_ARS_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 Annual Review Scheduler Agent (GL-EUDR-ARS-034)
Regulation: EU 2023/1115 (EUDR) Articles 8, 10, 11, 12, 14, 29, 31
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

_ENV_PREFIX = "GL_EUDR_ARS_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_ARS_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_ARS_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_ARS_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_ARS_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_ARS_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class AnnualReviewSchedulerConfig:
    """Centralized configuration for AGENT-EUDR-034.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_ARS_ environment variables.
    """

    # -- 1. Database ------------------------------------------------------------
    db_host: str = field(default_factory=lambda: _env("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: _env_int("DB_PORT", 5432))
    db_name: str = field(default_factory=lambda: _env("DB_NAME", "greenlang"))
    db_user: str = field(default_factory=lambda: _env("DB_USER", "gl"))
    db_password: str = field(default_factory=lambda: _env("DB_PASSWORD", "gl"))
    db_pool_min: int = field(default_factory=lambda: _env_int("DB_POOL_MIN", 2))
    db_pool_max: int = field(default_factory=lambda: _env_int("DB_POOL_MAX", 10))

    # -- 2. Redis ---------------------------------------------------------------
    redis_host: str = field(default_factory=lambda: _env("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))
    redis_db: int = field(default_factory=lambda: _env_int("REDIS_DB", 0))
    redis_password: str = field(default_factory=lambda: _env("REDIS_PASSWORD", ""))
    cache_ttl: int = field(default_factory=lambda: _env_int("CACHE_TTL", 3600))

    # -- 3. Review Cycle Management Settings ------------------------------------
    review_cycle_auto_create: bool = field(
        default_factory=lambda: _env_bool("CYCLE_AUTO_CREATE", True)
    )
    review_cycle_creation_lead_days: int = field(
        default_factory=lambda: _env_int("CYCLE_CREATION_LEAD_DAYS", 90)
    )
    review_cycle_duration_days: int = field(
        default_factory=lambda: _env_int("REVIEW_CYCLE_DURATION_DAYS",
                                          _env_int("CYCLE_DURATION_DAYS", 120))
    )
    review_cycle_grace_period_days: int = field(
        default_factory=lambda: _env_int("CYCLE_GRACE_PERIOD_DAYS", 14)
    )
    review_cycle_max_active: int = field(
        default_factory=lambda: _env_int("CYCLE_MAX_ACTIVE", 5)
    )
    review_cycle_default_month: int = field(
        default_factory=lambda: _env_int("CYCLE_DEFAULT_MONTH", 3)
    )
    review_cycle_default_day: int = field(
        default_factory=lambda: _env_int("CYCLE_DEFAULT_DAY", 31)
    )
    review_cycle_task_batch_size: int = field(
        default_factory=lambda: _env_int("CYCLE_TASK_BATCH_SIZE", 50)
    )
    auto_schedule_enabled: bool = field(
        default_factory=lambda: _env_bool("AUTO_SCHEDULE_ENABLED", True)
    )

    # -- 3b. Phase Duration Settings -------------------------------------------
    preparation_phase_days: int = field(
        default_factory=lambda: _env_int("PREPARATION_PHASE_DAYS", 14)
    )
    data_collection_phase_days: int = field(
        default_factory=lambda: _env_int("DATA_COLLECTION_PHASE_DAYS", 30)
    )
    analysis_phase_days: int = field(
        default_factory=lambda: _env_int("ANALYSIS_PHASE_DAYS", 21)
    )
    review_meeting_phase_days: int = field(
        default_factory=lambda: _env_int("REVIEW_MEETING_PHASE_DAYS", 7)
    )
    remediation_phase_days: int = field(
        default_factory=lambda: _env_int("REMEDIATION_PHASE_DAYS", 30)
    )
    sign_off_phase_days: int = field(
        default_factory=lambda: _env_int("SIGN_OFF_PHASE_DAYS", 7)
    )

    # -- 3c. Compliance & Risk Targets -----------------------------------------
    compliance_rate_target: Decimal = field(
        default_factory=lambda: _env_decimal("COMPLIANCE_RATE_TARGET", "95.00")
    )
    risk_score_improvement_target: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_SCORE_IMPROVEMENT_TARGET", "10.00")
    )
    max_review_cycles_per_operator: int = field(
        default_factory=lambda: _env_int("MAX_REVIEW_CYCLES_PER_OPERATOR", 5)
    )

    # -- 4. Deadline Tracking Settings ------------------------------------------
    deadline_warning_days: int = field(
        default_factory=lambda: _env_int("DEADLINE_WARNING_DAYS", 7)
    )
    deadline_critical_days: int = field(
        default_factory=lambda: _env_int("DEADLINE_CRITICAL_DAYS", 3)
    )
    deadline_overdue_escalation_hours: int = field(
        default_factory=lambda: _env_int("DEADLINE_OVERDUE_ESCALATION_HOURS", 24)
    )
    submission_window_days: int = field(
        default_factory=lambda: _env_int("SUBMISSION_WINDOW_DAYS", 30)
    )
    regulatory_submission_buffer_days: int = field(
        default_factory=lambda: _env_int("REG_SUBMISSION_BUFFER_DAYS", 5)
    )
    deadline_check_interval_minutes: int = field(
        default_factory=lambda: _env_int("DEADLINE_CHECK_INTERVAL_MINUTES", 60)
    )

    # -- 5. Checklist Generation Settings ---------------------------------------
    checklist_template_version: str = field(
        default_factory=lambda: _env("CHECKLIST_TEMPLATE_VERSION", "2026.1")
    )
    checklist_auto_generate: bool = field(
        default_factory=lambda: _env_bool("CHECKLIST_AUTO_GENERATE", True)
    )
    checklist_max_items: int = field(
        default_factory=lambda: _env_int("CHECKLIST_MAX_ITEMS", 200)
    )
    checklist_commodity_templates_enabled: bool = field(
        default_factory=lambda: _env_bool("CHECKLIST_COMMODITY_TEMPLATES", True)
    )
    checklist_completion_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("CHECKLIST_COMPLETION_THRESHOLD", "100.0")
    )
    checklist_mandatory_item_weight: Decimal = field(
        default_factory=lambda: _env_decimal("CHECKLIST_MANDATORY_WEIGHT", "2.0")
    )

    # -- 6. Entity Coordination Settings ----------------------------------------
    entity_cascade_max_depth: int = field(
        default_factory=lambda: _env_int("ENTITY_CASCADE_MAX_DEPTH", 5)
    )
    entity_parallel_review_limit: int = field(
        default_factory=lambda: _env_int("ENTITY_PARALLEL_LIMIT", 20)
    )
    entity_dependency_timeout_hours: int = field(
        default_factory=lambda: _env_int("ENTITY_DEP_TIMEOUT_HOURS", 72)
    )
    entity_auto_cascade: bool = field(
        default_factory=lambda: _env_bool("ENTITY_AUTO_CASCADE", True)
    )
    entity_aggregation_batch_size: int = field(
        default_factory=lambda: _env_int("ENTITY_AGGREGATION_BATCH", 100)
    )

    # -- 7. Year-over-Year Comparison Settings ----------------------------------
    yoy_significance_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("YOY_SIGNIFICANCE_THRESHOLD", "5.0")
    )
    yoy_critical_change_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("YOY_CRITICAL_CHANGE_THRESHOLD", "25.0")
    )
    yoy_max_comparison_years: int = field(
        default_factory=lambda: _env_int("YOY_MAX_COMPARISON_YEARS", 5)
    )
    year_comparison_lookback: int = field(
        default_factory=lambda: _env_int("YEAR_COMPARISON_LOOKBACK", 3)
    )
    yoy_weight_supplier_count: Decimal = field(
        default_factory=lambda: _env_decimal("YOY_WEIGHT_SUPPLIER_COUNT", "0.20")
    )
    yoy_weight_risk_score: Decimal = field(
        default_factory=lambda: _env_decimal("YOY_WEIGHT_RISK_SCORE", "0.30")
    )
    yoy_weight_compliance_score: Decimal = field(
        default_factory=lambda: _env_decimal("YOY_WEIGHT_COMPLIANCE", "0.30")
    )
    yoy_weight_deforestation: Decimal = field(
        default_factory=lambda: _env_decimal("YOY_WEIGHT_DEFORESTATION", "0.20")
    )

    # -- 8. Calendar Management Settings ----------------------------------------
    calendar_event_retention_years: int = field(
        default_factory=lambda: _env_int("CALENDAR_RETENTION_YEARS", 5)
    )
    calendar_default_timezone: str = field(
        default_factory=lambda: _env("CALENDAR_DEFAULT_TZ", "UTC")
    )
    calendar_ical_sync_enabled: bool = field(
        default_factory=lambda: _env_bool("CALENDAR_ICAL_SYNC", True)
    )
    calendar_sync_enabled: bool = field(
        default_factory=lambda: _env_bool("CALENDAR_SYNC_ENABLED", True)
    )
    calendar_max_events_per_query: int = field(
        default_factory=lambda: _env_int("CALENDAR_MAX_EVENTS", 500)
    )
    calendar_reminder_days: List[int] = field(
        default_factory=lambda: [30, 14, 7, 3, 1]
    )

    # -- 9. Notification Engine Settings ----------------------------------------
    notification_channels: List[str] = field(
        default_factory=lambda: ["email", "webhook", "dashboard", "sms"]
    )
    notification_retry_max: int = field(
        default_factory=lambda: _env_int("NOTIFICATION_RETRY_MAX",
                                          _env_int("NOTIF_RETRY_MAX", 3))
    )
    notification_retry_delay_seconds: int = field(
        default_factory=lambda: _env_int("NOTIF_RETRY_DELAY", 300)
    )
    notification_batch_size: int = field(
        default_factory=lambda: _env_int("NOTIF_BATCH_SIZE", 100)
    )
    notification_escalation_tiers: List[str] = field(
        default_factory=lambda: ["reviewer", "manager", "director", "compliance_officer"]
    )
    notification_escalation_interval_hours: int = field(
        default_factory=lambda: _env_int("NOTIF_ESCALATION_INTERVAL_HOURS", 24)
    )
    notification_quiet_hours_start: int = field(
        default_factory=lambda: _env_int("NOTIF_QUIET_START", 22)
    )
    notification_quiet_hours_end: int = field(
        default_factory=lambda: _env_int("NOTIF_QUIET_END", 7)
    )

    # -- 10. Upstream Agent URLs ------------------------------------------------
    continuous_monitoring_url: str = field(
        default_factory=lambda: _env(
            "CONTINUOUS_MONITORING_URL",
            "http://eudr-continuous-monitoring:8033/api/v1/eudr/continuous-monitoring",
        )
    )
    due_diligence_url: str = field(
        default_factory=lambda: _env(
            "DUE_DILIGENCE_URL",
            "http://eudr-due-diligence:8026/api/v1/eudr/due-diligence-orchestrator",
        )
    )
    risk_assessment_url: str = field(
        default_factory=lambda: _env(
            "RISK_ASSESSMENT_URL",
            "http://eudr-risk-assessment:8028/api/v1/eudr/risk-assessment-engine",
        )
    )
    legal_compliance_url: str = field(
        default_factory=lambda: _env(
            "LEGAL_COMPLIANCE_URL",
            "http://eudr-legal-compliance:8023/api/v1/eudr/legal-compliance-verifier",
        )
    )
    documentation_generator_url: str = field(
        default_factory=lambda: _env(
            "DOCUMENTATION_GENERATOR_URL",
            "http://eudr-documentation-generator:8030/api/v1/eudr/documentation-generator",
        )
    )

    # -- 11. Rate Limiting ------------------------------------------------------
    rate_limit_anonymous: int = field(default_factory=lambda: _env_int("RATE_LIMIT_ANONYMOUS", 10))
    rate_limit_basic: int = field(default_factory=lambda: _env_int("RATE_LIMIT_BASIC", 30))
    rate_limit_standard: int = field(default_factory=lambda: _env_int("RATE_LIMIT_STANDARD", 100))
    rate_limit_premium: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PREMIUM", 500))
    rate_limit_admin: int = field(default_factory=lambda: _env_int("RATE_LIMIT_ADMIN", 2000))

    # -- 12. Circuit Breaker ----------------------------------------------------
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5)
    )
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _env_int("CB_RESET_TIMEOUT", 60)
    )
    circuit_breaker_half_open_max: int = field(
        default_factory=lambda: _env_int("CB_HALF_OPEN_MAX", 3)
    )

    # -- 13. Batch Processing ---------------------------------------------------
    max_concurrent: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT", 10))
    batch_timeout_seconds: int = field(default_factory=lambda: _env_int("BATCH_TIMEOUT", 300))

    # -- 14. Provenance ---------------------------------------------------------
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

    # -- 15. Metrics ------------------------------------------------------------
    metrics_enabled: bool = field(
        default_factory=lambda: _env_bool("METRICS_ENABLED", True)
    )
    metrics_prefix: str = "gl_eudr_ars_"

    # -- 16. Retention ----------------------------------------------------------
    retention_years: int = field(
        default_factory=lambda: _env_int("RETENTION_YEARS", 5)
    )

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min, self.db_pool_max,
            )

        # Validate YoY comparison weights sum to 1.0
        total_yoy_weight = (
            self.yoy_weight_supplier_count
            + self.yoy_weight_risk_score
            + self.yoy_weight_compliance_score
            + self.yoy_weight_deforestation
        )
        if abs(total_yoy_weight - Decimal("1.0")) > Decimal("0.01"):
            logger.warning(
                "Year-over-year comparison weights sum to %s, expected 1.0.",
                total_yoy_weight,
            )

        if self.retention_years < 5:
            logger.warning(
                "Retention years %d is below EUDR minimum of 5 years "
                "(Article 31).", self.retention_years,
            )

        if self.deadline_critical_days >= self.deadline_warning_days:
            logger.warning(
                "Deadline critical threshold (%d days) should be less "
                "than warning threshold (%d days).",
                self.deadline_critical_days,
                self.deadline_warning_days,
            )

        if self.review_cycle_grace_period_days >= self.review_cycle_duration_days:
            logger.warning(
                "Review cycle grace period (%d days) should be less "
                "than cycle duration (%d days).",
                self.review_cycle_grace_period_days,
                self.review_cycle_duration_days,
            )

        if self.review_cycle_default_month < 1 or self.review_cycle_default_month > 12:
            logger.warning(
                "Review cycle default month %d is invalid, defaulting to March.",
                self.review_cycle_default_month,
            )
            object.__setattr__(self, "review_cycle_default_month", 3)

        # Validate total phase duration doesn't exceed cycle duration
        total_phase_days = (
            self.preparation_phase_days
            + self.data_collection_phase_days
            + self.analysis_phase_days
            + self.review_meeting_phase_days
            + self.remediation_phase_days
            + self.sign_off_phase_days
        )
        if total_phase_days > self.review_cycle_duration_days:
            logger.warning(
                "Total phase duration (%d days) exceeds cycle duration (%d days).",
                total_phase_days,
                self.review_cycle_duration_days,
            )

        # Validate compliance rate target
        if self.compliance_rate_target > Decimal("100.0"):
            logger.warning(
                "Compliance rate target %s exceeds 100 percent.",
                self.compliance_rate_target,
            )

        logger.info(
            "AnnualReviewSchedulerConfig initialized: "
            "cycle_lead=%dd, cycle_duration=%dd, "
            "deadline_warn=%dd, deadline_crit=%dd, "
            "yoy_max_years=%d, notif_channels=%d",
            self.review_cycle_creation_lead_days,
            self.review_cycle_duration_days,
            self.deadline_warning_days,
            self.deadline_critical_days,
            self.yoy_max_comparison_years,
            len(self.notification_channels),
        )

    def get_deadline_severity(self, days_remaining: int) -> str:
        """Classify deadline urgency by days remaining.

        Args:
            days_remaining: Days until the deadline.

        Returns:
            Severity level string.
        """
        if days_remaining < 0:
            return "overdue"
        elif days_remaining <= self.deadline_critical_days:
            return "critical"
        elif days_remaining <= self.deadline_warning_days:
            return "warning"
        return "normal"

    def get_yoy_comparison_weights(self) -> Dict[str, Decimal]:
        """Get year-over-year comparison weights as a dictionary.

        Returns:
            Dictionary mapping dimension names to their weights.
        """
        return {
            "supplier_count": self.yoy_weight_supplier_count,
            "risk_score": self.yoy_weight_risk_score,
            "compliance_score": self.yoy_weight_compliance_score,
            "deforestation": self.yoy_weight_deforestation,
        }

    def get_change_significance(self, change_percent: Decimal) -> str:
        """Classify the significance of a year-over-year change.

        Args:
            change_percent: Absolute percentage change.

        Returns:
            Significance level string.
        """
        abs_change = abs(change_percent)
        if abs_change >= self.yoy_critical_change_threshold:
            return "critical"
        elif abs_change >= self.yoy_significance_threshold:
            return "significant"
        return "minor"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[AnnualReviewSchedulerConfig] = None
_config_lock = threading.Lock()


def get_config() -> AnnualReviewSchedulerConfig:
    """Return the thread-safe singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AnnualReviewSchedulerConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
