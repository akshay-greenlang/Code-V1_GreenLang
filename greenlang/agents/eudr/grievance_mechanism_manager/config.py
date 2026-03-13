# -*- coding: utf-8 -*-
"""
Grievance Mechanism Manager Configuration - AGENT-EUDR-032

Centralized configuration for the Grievance Mechanism Manager covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Analytics settings: analysis windows, clustering thresholds
- Root cause analysis: max depth, min confidence, methods
- Mediation settings: max sessions, stage SLAs, mediator types
- Remediation settings: verification requirements, satisfaction thresholds
- Risk scoring: weight factors, scoring windows, level thresholds
- Collective grievance: min stakeholders, negotiation timeouts
- Regulatory reporting: report types, retention years
- Rate limiting, circuit breaker, batch processing
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_gmm_

All settings overridable via environment variables with ``GL_EUDR_GMM_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-032 Grievance Mechanism Manager (GL-EUDR-GMM-032)
Regulation: EU 2023/1115 (EUDR); CSDDD Article 8; UNGP Principle 31
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

_ENV_PREFIX = "GL_EUDR_GMM_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_GMM_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_GMM_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_GMM_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_GMM_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_GMM_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class GrievanceMechanismManagerConfig:
    """Centralized configuration for AGENT-EUDR-032.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_GMM_ environment variables.
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

    # -- 3. Analytics Settings --------------------------------------------------
    analytics_default_window_days: int = field(
        default_factory=lambda: _env_int("ANALYTICS_WINDOW_DAYS", 90)
    )
    analytics_min_grievances_for_pattern: int = field(
        default_factory=lambda: _env_int("ANALYTICS_MIN_GRIEVANCES", 3)
    )
    analytics_clustering_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("ANALYTICS_CLUSTER_THRESHOLD", "0.70")
    )
    analytics_trend_window_days: int = field(
        default_factory=lambda: _env_int("ANALYTICS_TREND_WINDOW_DAYS", 30)
    )

    # -- 4. Root Cause Analysis Settings ----------------------------------------
    root_cause_max_depth: int = field(
        default_factory=lambda: _env_int("RC_MAX_DEPTH", 5)
    )
    root_cause_min_confidence: Decimal = field(
        default_factory=lambda: _env_decimal("RC_MIN_CONFIDENCE", "50")
    )
    root_cause_default_method: str = field(
        default_factory=lambda: _env("RC_DEFAULT_METHOD", "five_whys")
    )

    # -- 5. Mediation Settings --------------------------------------------------
    mediation_max_sessions: int = field(
        default_factory=lambda: _env_int("MEDIATION_MAX_SESSIONS", 20)
    )
    mediation_default_session_minutes: int = field(
        default_factory=lambda: _env_int("MEDIATION_SESSION_MINUTES", 120)
    )
    mediation_stage_sla_preparation_days: int = field(
        default_factory=lambda: _env_int("MEDIATION_SLA_PREP_DAYS", 7)
    )
    mediation_stage_sla_dialogue_days: int = field(
        default_factory=lambda: _env_int("MEDIATION_SLA_DIALOGUE_DAYS", 14)
    )
    mediation_stage_sla_negotiation_days: int = field(
        default_factory=lambda: _env_int("MEDIATION_SLA_NEGOTIATION_DAYS", 21)
    )
    mediation_stage_sla_settlement_days: int = field(
        default_factory=lambda: _env_int("MEDIATION_SLA_SETTLEMENT_DAYS", 14)
    )
    mediation_stage_sla_implementation_days: int = field(
        default_factory=lambda: _env_int("MEDIATION_SLA_IMPL_DAYS", 30)
    )

    # -- 6. Remediation Settings ------------------------------------------------
    remediation_verification_required: bool = field(
        default_factory=lambda: _env_bool("REMEDIATION_VERIFY_REQUIRED", True)
    )
    remediation_min_satisfaction: Decimal = field(
        default_factory=lambda: _env_decimal("REMEDIATION_MIN_SATISFACTION", "3.0")
    )
    remediation_effectiveness_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("REMEDIATION_EFFECTIVENESS_MIN", "60")
    )

    # -- 7. Risk Scoring Settings -----------------------------------------------
    risk_scoring_window_days: int = field(
        default_factory=lambda: _env_int("RISK_WINDOW_DAYS", 180)
    )
    risk_weight_frequency: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_WEIGHT_FREQUENCY", "0.30")
    )
    risk_weight_severity: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_WEIGHT_SEVERITY", "0.25")
    )
    risk_weight_resolution: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_WEIGHT_RESOLUTION", "0.20")
    )
    risk_weight_escalation: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_WEIGHT_ESCALATION", "0.15")
    )
    risk_weight_unresolved: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_WEIGHT_UNRESOLVED", "0.10")
    )
    risk_level_negligible_max: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_NEGLIGIBLE_MAX", "15")
    )
    risk_level_low_max: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_LOW_MAX", "30")
    )
    risk_level_moderate_max: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_MODERATE_MAX", "60")
    )
    risk_level_high_max: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_HIGH_MAX", "80")
    )

    # -- 8. Collective Grievance Settings ---------------------------------------
    collective_min_stakeholders: int = field(
        default_factory=lambda: _env_int("COLLECTIVE_MIN_STAKEHOLDERS", 3)
    )
    collective_negotiation_timeout_days: int = field(
        default_factory=lambda: _env_int("COLLECTIVE_NEGOTIATION_TIMEOUT", 60)
    )

    # -- 9. Regulatory Reporting Settings ---------------------------------------
    retention_years: int = field(
        default_factory=lambda: _env_int("RETENTION_YEARS", 5)
    )
    report_formats: List[str] = field(
        default_factory=lambda: ["json", "pdf", "html", "xlsx"]
    )
    default_report_format: str = field(
        default_factory=lambda: _env("DEFAULT_REPORT_FORMAT", "json")
    )

    # -- 10. Upstream Agent URLs ------------------------------------------------
    stakeholder_engagement_url: str = field(
        default_factory=lambda: _env(
            "STAKEHOLDER_ENGAGEMENT_URL",
            "http://eudr-stakeholder-engagement:8031/api/v1/eudr/stakeholder-engagement",
        )
    )
    supply_chain_url: str = field(
        default_factory=lambda: _env(
            "SUPPLY_CHAIN_URL",
            "http://eudr-supply-chain:8001/api/v1/eudr/supply-chain",
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
    metrics_prefix: str = "gl_eudr_gmm_"

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min, self.db_pool_max,
            )

        if self.root_cause_max_depth < 1 or self.root_cause_max_depth > 10:
            logger.warning(
                "Root cause max depth %d is outside [1, 10].",
                self.root_cause_max_depth,
            )

        total_risk_weight = (
            self.risk_weight_frequency + self.risk_weight_severity
            + self.risk_weight_resolution + self.risk_weight_escalation
            + self.risk_weight_unresolved
        )
        if abs(total_risk_weight - Decimal("1.0")) > Decimal("0.01"):
            logger.warning(
                "Risk scoring weights sum to %s, expected 1.0.",
                total_risk_weight,
            )

        if self.retention_years < 5:
            logger.warning(
                "Retention years %d is below EUDR minimum of 5 years "
                "(Article 31).", self.retention_years,
            )

        if self.mediation_max_sessions < 1:
            logger.warning(
                "Mediation max sessions %d must be at least 1.",
                self.mediation_max_sessions,
            )

        logger.info(
            "GrievanceMechanismManagerConfig initialized: "
            "analytics_window=%dd, rc_max_depth=%d, "
            "mediation_max_sessions=%d, risk_window=%dd, "
            "collective_min=%d, retention_years=%d",
            self.analytics_default_window_days,
            self.root_cause_max_depth,
            self.mediation_max_sessions,
            self.risk_scoring_window_days,
            self.collective_min_stakeholders,
            self.retention_years,
        )

    def get_risk_level(self, score: Decimal) -> str:
        """Classify a numeric risk score into a risk level.

        Args:
            score: Numeric risk score (0-100).

        Returns:
            Risk level string.
        """
        if score <= self.risk_level_negligible_max:
            return "negligible"
        elif score <= self.risk_level_low_max:
            return "low"
        elif score <= self.risk_level_moderate_max:
            return "moderate"
        elif score <= self.risk_level_high_max:
            return "high"
        return "critical"

    def get_mediation_stage_sla(self, stage: str) -> int:
        """Get mediation stage SLA in days.

        Args:
            stage: Mediation stage name.

        Returns:
            SLA in days for the given stage.
        """
        sla_map: Dict[str, int] = {
            "preparation": self.mediation_stage_sla_preparation_days,
            "dialogue": self.mediation_stage_sla_dialogue_days,
            "negotiation": self.mediation_stage_sla_negotiation_days,
            "settlement": self.mediation_stage_sla_settlement_days,
            "implementation": self.mediation_stage_sla_implementation_days,
        }
        return sla_map.get(stage, 14)

    def get_risk_weights(self) -> Dict[str, Decimal]:
        """Get risk scoring weights as a dictionary.

        Returns:
            Dictionary mapping factor names to their weights.
        """
        return {
            "frequency": self.risk_weight_frequency,
            "severity": self.risk_weight_severity,
            "resolution": self.risk_weight_resolution,
            "escalation": self.risk_weight_escalation,
            "unresolved": self.risk_weight_unresolved,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[GrievanceMechanismManagerConfig] = None
_config_lock = threading.Lock()


def get_config() -> GrievanceMechanismManagerConfig:
    """Return the thread-safe singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = GrievanceMechanismManagerConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
