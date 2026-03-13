# -*- coding: utf-8 -*-
"""
Mitigation Measure Designer Configuration - AGENT-EUDR-029

Centralized configuration for the Mitigation Measure Designer covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Measure template management: version, limits, refresh intervals
- Risk thresholds: NEGLIGIBLE/LOW/STANDARD/HIGH/CRITICAL boundaries
  with mitigation target score for strategy design
- Effectiveness estimation: conservative/moderate/optimistic factors
  with min threshold and max cap for risk reduction projections
- Implementation tracking: deadlines, overdue alerts, extensions,
  evidence requirements
- Verification settings: cooldown, minimum data points, timeout
- Workflow configuration: approvals, auto-close, duration limits,
  escalation thresholds
- Upstream agent URLs: EUDR-016 to EUDR-020, EUDR-028 (risk assessment)
- Report generation settings: format, evidence, provenance
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_mmd_

All settings overridable via environment variables with ``GL_EUDR_MMD_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 Mitigation Measure Designer (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 29, 31
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

_ENV_PREFIX = "GL_EUDR_MMD_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_MMD_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_MMD_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_MMD_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_MMD_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_MMD_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class MitigationMeasureDesignerConfig:
    """Centralized configuration for AGENT-EUDR-029.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_MMD_ environment variables.
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

    # -- 3. Measure Templates ---------------------------------------------------
    default_template_version: str = field(
        default_factory=lambda: _env("DEFAULT_TEMPLATE_VERSION", "1.0")
    )
    max_measures_per_strategy: int = field(
        default_factory=lambda: _env_int("MAX_MEASURES_PER_STRATEGY", 20)
    )
    template_refresh_interval: int = field(
        default_factory=lambda: _env_int("TEMPLATE_REFRESH_INTERVAL", 86400)
    )

    # -- 4. Risk Thresholds -----------------------------------------------------
    # Score ranges: [0, negligible_max) -> NEGLIGIBLE, etc.
    negligible_max: Decimal = field(
        default_factory=lambda: _env_decimal("NEGLIGIBLE_MAX", "15")
    )
    low_max: Decimal = field(
        default_factory=lambda: _env_decimal("LOW_MAX", "30")
    )
    standard_max: Decimal = field(
        default_factory=lambda: _env_decimal("STANDARD_MAX", "60")
    )
    high_max: Decimal = field(
        default_factory=lambda: _env_decimal("HIGH_MAX", "80")
    )
    target_risk_level: str = field(
        default_factory=lambda: _env("TARGET_RISK_LEVEL", "low")
    )
    mitigation_target_score: Decimal = field(
        default_factory=lambda: _env_decimal("MITIGATION_TARGET_SCORE", "30")
    )

    # -- 5. Effectiveness Estimation -------------------------------------------
    conservative_factor: Decimal = field(
        default_factory=lambda: _env_decimal("CONSERVATIVE_FACTOR", "0.70")
    )
    moderate_factor: Decimal = field(
        default_factory=lambda: _env_decimal("MODERATE_FACTOR", "1.00")
    )
    optimistic_factor: Decimal = field(
        default_factory=lambda: _env_decimal("OPTIMISTIC_FACTOR", "1.30")
    )
    min_effectiveness_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("MIN_EFFECTIVENESS_THRESHOLD", "5")
    )
    max_effectiveness_cap: Decimal = field(
        default_factory=lambda: _env_decimal("MAX_EFFECTIVENESS_CAP", "80")
    )

    # -- 6. Implementation Tracking --------------------------------------------
    default_deadline_days: int = field(
        default_factory=lambda: _env_int("DEFAULT_DEADLINE_DAYS", 30)
    )
    overdue_alert_threshold_days: int = field(
        default_factory=lambda: _env_int("OVERDUE_ALERT_THRESHOLD_DAYS", 7)
    )
    max_extensions: int = field(
        default_factory=lambda: _env_int("MAX_EXTENSIONS", 3)
    )
    evidence_required: bool = field(
        default_factory=lambda: _env_bool("EVIDENCE_REQUIRED", True)
    )

    # -- 7. Verification -------------------------------------------------------
    verification_cooldown_days: int = field(
        default_factory=lambda: _env_int("VERIFICATION_COOLDOWN_DAYS", 7)
    )
    min_data_points_for_verification: int = field(
        default_factory=lambda: _env_int("MIN_DATA_POINTS_FOR_VERIFICATION", 3)
    )
    re_evaluation_timeout_seconds: int = field(
        default_factory=lambda: _env_int("RE_EVALUATION_TIMEOUT_SECONDS", 30)
    )

    # -- 8. Workflow ------------------------------------------------------------
    approval_required: bool = field(
        default_factory=lambda: _env_bool("APPROVAL_REQUIRED", True)
    )
    auto_close_on_negligible: bool = field(
        default_factory=lambda: _env_bool("AUTO_CLOSE_ON_NEGLIGIBLE", True)
    )
    max_workflow_duration_days: int = field(
        default_factory=lambda: _env_int("MAX_WORKFLOW_DURATION_DAYS", 90)
    )
    escalation_threshold_days: int = field(
        default_factory=lambda: _env_int("ESCALATION_THRESHOLD_DAYS", 60)
    )

    # -- 9. Upstream Agent URLs -------------------------------------------------
    risk_assessment_url: str = field(
        default_factory=lambda: _env(
            "RISK_ASSESSMENT_URL",
            "http://eudr-risk-assessment:8028/api/v1/eudr/risk-assessment",
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
    commodity_risk_url: str = field(
        default_factory=lambda: _env(
            "COMMODITY_RISK_URL",
            "http://eudr-commodity-risk:8018/api/v1/eudr/commodity-risk",
        )
    )
    corruption_index_url: str = field(
        default_factory=lambda: _env(
            "CORRUPTION_INDEX_URL",
            "http://eudr-corruption-index:8019/api/v1/eudr/corruption-index",
        )
    )
    deforestation_alert_url: str = field(
        default_factory=lambda: _env(
            "DEFORESTATION_ALERT_URL",
            "http://eudr-deforestation-alert:8020/api/v1/eudr/deforestation-alert",
        )
    )

    # -- 10. Report Generation --------------------------------------------------
    report_format: str = field(
        default_factory=lambda: _env("REPORT_FORMAT", "json")
    )
    include_evidence_summary: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_EVIDENCE_SUMMARY", True)
    )
    include_provenance: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_PROVENANCE", True)
    )

    # -- 11. Rate Limiting ------------------------------------------------------
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
    max_concurrent: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT", 10)
    )
    batch_timeout_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_TIMEOUT", 300)
    )

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
    metrics_prefix: str = "gl_eudr_mmd_"

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate threshold ordering
        if not (
            self.negligible_max
            < self.low_max
            < self.standard_max
            < self.high_max
        ):
            logger.warning(
                "Risk thresholds are not in ascending order: "
                "negligible_max=%s, low_max=%s, standard_max=%s, high_max=%s",
                self.negligible_max,
                self.low_max,
                self.standard_max,
                self.high_max,
            )

        # Validate effectiveness factors ordering
        if not (
            self.conservative_factor
            <= self.moderate_factor
            <= self.optimistic_factor
        ):
            logger.warning(
                "Effectiveness factors are not in ascending order: "
                "conservative=%s, moderate=%s, optimistic=%s",
                self.conservative_factor,
                self.moderate_factor,
                self.optimistic_factor,
            )

        # Validate mitigation target is within low range
        if self.mitigation_target_score > self.low_max:
            logger.warning(
                "Mitigation target score %s exceeds LOW threshold %s. "
                "Strategies may not achieve target risk level.",
                self.mitigation_target_score,
                self.low_max,
            )

        # Validate effectiveness bounds
        if self.min_effectiveness_threshold >= self.max_effectiveness_cap:
            logger.warning(
                "Min effectiveness threshold %s >= max cap %s. "
                "No valid effectiveness range exists.",
                self.min_effectiveness_threshold,
                self.max_effectiveness_cap,
            )

        # Validate pool sizing
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min,
                self.db_pool_max,
            )

        logger.info(
            "MitigationMeasureDesignerConfig initialized: "
            "thresholds (NEGLIGIBLE<%s, LOW<%s, STANDARD<%s, HIGH<%s), "
            "mitigation_target=%s, "
            "effectiveness_factors=[%s, %s, %s], "
            "effectiveness_range=[%s, %s], "
            "approval_required=%s, auto_close_on_negligible=%s, "
            "max_measures_per_strategy=%d",
            self.negligible_max,
            self.low_max,
            self.standard_max,
            self.high_max,
            self.mitigation_target_score,
            self.conservative_factor,
            self.moderate_factor,
            self.optimistic_factor,
            self.min_effectiveness_threshold,
            self.max_effectiveness_cap,
            self.approval_required,
            self.auto_close_on_negligible,
            self.max_measures_per_strategy,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[MitigationMeasureDesignerConfig] = None
_config_lock = threading.Lock()


def get_config() -> MitigationMeasureDesignerConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        MitigationMeasureDesignerConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = MitigationMeasureDesignerConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
