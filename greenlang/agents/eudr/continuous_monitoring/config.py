# -*- coding: utf-8 -*-
"""
Continuous Monitoring Agent Configuration - AGENT-EUDR-033

Centralized configuration for the Continuous Monitoring Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Supply chain monitoring: scan intervals, certification expiry thresholds
- Deforestation monitoring: alert correlation windows, impact thresholds
- Compliance checking: article freshness, risk assessment validity
- Change detection: sensitivity, impact scoring weights
- Risk score monitoring: degradation thresholds, trend analysis windows
- Data freshness validation: staleness thresholds, refresh scheduling
- Regulatory tracking: update sources, notification channels
- Rate limiting, circuit breaker, batch processing
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_cm_

All settings overridable via environment variables with ``GL_EUDR_CM_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 Continuous Monitoring Agent (GL-EUDR-CM-033)
Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 10, 11, 12, 14, 29, 31
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

_ENV_PREFIX = "GL_EUDR_CM_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_CM_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_CM_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_CM_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_CM_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_CM_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class ContinuousMonitoringConfig:
    """Centralized configuration for AGENT-EUDR-033.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_CM_ environment variables.
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

    # -- 3. Supply Chain Monitoring Settings ------------------------------------
    supply_chain_scan_interval_minutes: int = field(
        default_factory=lambda: _env_int("SC_SCAN_INTERVAL_MINUTES", 60)
    )
    certification_expiry_warning_days: int = field(
        default_factory=lambda: _env_int("CERT_EXPIRY_WARNING_DAYS", 30)
    )
    certification_expiry_critical_days: int = field(
        default_factory=lambda: _env_int("CERT_EXPIRY_CRITICAL_DAYS", 7)
    )
    geolocation_drift_threshold_km: Decimal = field(
        default_factory=lambda: _env_decimal("GEO_DRIFT_THRESHOLD_KM", "0.5")
    )
    supplier_status_change_lookback_days: int = field(
        default_factory=lambda: _env_int("SUPPLIER_CHANGE_LOOKBACK_DAYS", 90)
    )
    max_suppliers_per_scan: int = field(
        default_factory=lambda: _env_int("MAX_SUPPLIERS_PER_SCAN", 500)
    )

    # -- 4. Deforestation Monitoring Settings -----------------------------------
    deforestation_alert_correlation_window_hours: int = field(
        default_factory=lambda: _env_int("DEFOREST_CORR_WINDOW_HOURS", 72)
    )
    deforestation_impact_high_threshold_hectares: Decimal = field(
        default_factory=lambda: _env_decimal("DEFOREST_HIGH_THRESHOLD_HA", "10.0")
    )
    deforestation_impact_critical_threshold_hectares: Decimal = field(
        default_factory=lambda: _env_decimal("DEFOREST_CRITICAL_THRESHOLD_HA", "50.0")
    )
    deforestation_check_interval_minutes: int = field(
        default_factory=lambda: _env_int("DEFOREST_CHECK_INTERVAL_MINUTES", 30)
    )
    investigation_auto_trigger_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("INVEST_AUTO_TRIGGER_THRESHOLD", "0.75")
    )

    # -- 5. Compliance Checking Settings ----------------------------------------
    compliance_audit_interval_days: int = field(
        default_factory=lambda: _env_int("COMPLIANCE_AUDIT_INTERVAL_DAYS", 30)
    )
    article_8_freshness_max_days: int = field(
        default_factory=lambda: _env_int("ARTICLE_8_FRESHNESS_DAYS", 365)
    )
    risk_assessment_validity_days: int = field(
        default_factory=lambda: _env_int("RISK_ASSESSMENT_VALIDITY_DAYS", 180)
    )
    due_diligence_statement_max_age_days: int = field(
        default_factory=lambda: _env_int("DDS_MAX_AGE_DAYS", 365)
    )
    compliance_pass_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("COMPLIANCE_PASS_THRESHOLD", "80.0")
    )

    # -- 6. Change Detection Settings -------------------------------------------
    change_detection_sensitivity: Decimal = field(
        default_factory=lambda: _env_decimal("CHANGE_SENSITIVITY", "0.10")
    )
    change_impact_weight_compliance: Decimal = field(
        default_factory=lambda: _env_decimal("CHANGE_WEIGHT_COMPLIANCE", "0.35")
    )
    change_impact_weight_risk: Decimal = field(
        default_factory=lambda: _env_decimal("CHANGE_WEIGHT_RISK", "0.30")
    )
    change_impact_weight_supply_chain: Decimal = field(
        default_factory=lambda: _env_decimal("CHANGE_WEIGHT_SUPPLY_CHAIN", "0.20")
    )
    change_impact_weight_regulatory: Decimal = field(
        default_factory=lambda: _env_decimal("CHANGE_WEIGHT_REGULATORY", "0.15")
    )
    change_lookback_days: int = field(
        default_factory=lambda: _env_int("CHANGE_LOOKBACK_DAYS", 30)
    )

    # -- 7. Risk Score Monitoring Settings --------------------------------------
    risk_degradation_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_DEGRAD_THRESHOLD", "10.0")
    )
    risk_trend_window_days: int = field(
        default_factory=lambda: _env_int("RISK_TREND_WINDOW_DAYS", 90)
    )
    risk_correlation_min_confidence: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_CORR_MIN_CONFIDENCE", "0.60")
    )
    risk_score_alert_high: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_ALERT_HIGH", "70.0")
    )
    risk_score_alert_critical: Decimal = field(
        default_factory=lambda: _env_decimal("RISK_ALERT_CRITICAL", "85.0")
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

    # -- 8. Data Freshness Settings ---------------------------------------------
    data_freshness_check_interval_minutes: int = field(
        default_factory=lambda: _env_int("FRESHNESS_CHECK_INTERVAL_MINUTES", 60)
    )
    data_stale_warning_hours: int = field(
        default_factory=lambda: _env_int("DATA_STALE_WARNING_HOURS", 24)
    )
    data_stale_critical_hours: int = field(
        default_factory=lambda: _env_int("DATA_STALE_CRITICAL_HOURS", 72)
    )
    data_refresh_batch_size: int = field(
        default_factory=lambda: _env_int("DATA_REFRESH_BATCH_SIZE", 100)
    )
    data_freshness_target_percent: Decimal = field(
        default_factory=lambda: _env_decimal("DATA_FRESHNESS_TARGET_PCT", "95.0")
    )

    # -- 9. Regulatory Tracking Settings ----------------------------------------
    regulatory_check_interval_hours: int = field(
        default_factory=lambda: _env_int("REG_CHECK_INTERVAL_HOURS", 24)
    )
    regulatory_sources: List[str] = field(
        default_factory=lambda: [
            "eur-lex", "eu-commission", "national-authorities", "fao",
        ]
    )
    regulatory_notification_channels: List[str] = field(
        default_factory=lambda: ["email", "webhook", "dashboard"]
    )
    regulatory_impact_assessment_auto: bool = field(
        default_factory=lambda: _env_bool("REG_IMPACT_AUTO", True)
    )

    # -- 10. Upstream Agent URLs ------------------------------------------------
    deforestation_alert_url: str = field(
        default_factory=lambda: _env(
            "DEFORESTATION_ALERT_URL",
            "http://eudr-deforestation-alert:8020/api/v1/eudr/deforestation-alert-system",
        )
    )
    supply_chain_url: str = field(
        default_factory=lambda: _env(
            "SUPPLY_CHAIN_URL",
            "http://eudr-supply-chain:8001/api/v1/eudr/supply-chain",
        )
    )
    risk_assessment_url: str = field(
        default_factory=lambda: _env(
            "RISK_ASSESSMENT_URL",
            "http://eudr-risk-assessment:8028/api/v1/eudr/risk-assessment-engine",
        )
    )
    due_diligence_url: str = field(
        default_factory=lambda: _env(
            "DUE_DILIGENCE_URL",
            "http://eudr-due-diligence:8026/api/v1/eudr/due-diligence-orchestrator",
        )
    )
    legal_compliance_url: str = field(
        default_factory=lambda: _env(
            "LEGAL_COMPLIANCE_URL",
            "http://eudr-legal-compliance:8023/api/v1/eudr/legal-compliance-verifier",
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
    metrics_prefix: str = "gl_eudr_cm_"

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

        # Validate change detection impact weights sum to 1.0
        total_change_weight = (
            self.change_impact_weight_compliance
            + self.change_impact_weight_risk
            + self.change_impact_weight_supply_chain
            + self.change_impact_weight_regulatory
        )
        if abs(total_change_weight - Decimal("1.0")) > Decimal("0.01"):
            logger.warning(
                "Change detection impact weights sum to %s, expected 1.0.",
                total_change_weight,
            )

        if self.retention_years < 5:
            logger.warning(
                "Retention years %d is below EUDR minimum of 5 years "
                "(Article 31).", self.retention_years,
            )

        if self.certification_expiry_critical_days >= self.certification_expiry_warning_days:
            logger.warning(
                "Certification critical threshold (%d days) should be less "
                "than warning threshold (%d days).",
                self.certification_expiry_critical_days,
                self.certification_expiry_warning_days,
            )

        if self.data_stale_warning_hours >= self.data_stale_critical_hours:
            logger.warning(
                "Data stale warning (%d hours) should be less "
                "than critical (%d hours).",
                self.data_stale_warning_hours,
                self.data_stale_critical_hours,
            )

        logger.info(
            "ContinuousMonitoringConfig initialized: "
            "sc_scan_interval=%dm, cert_warn=%dd, "
            "compliance_audit=%dd, risk_trend=%dd, "
            "freshness_interval=%dm, reg_interval=%dh",
            self.supply_chain_scan_interval_minutes,
            self.certification_expiry_warning_days,
            self.compliance_audit_interval_days,
            self.risk_trend_window_days,
            self.data_freshness_check_interval_minutes,
            self.regulatory_check_interval_hours,
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

    def get_change_impact_weights(self) -> Dict[str, Decimal]:
        """Get change detection impact weights as a dictionary.

        Returns:
            Dictionary mapping factor names to their weights.
        """
        return {
            "compliance": self.change_impact_weight_compliance,
            "risk": self.change_impact_weight_risk,
            "supply_chain": self.change_impact_weight_supply_chain,
            "regulatory": self.change_impact_weight_regulatory,
        }

    def get_deforestation_severity(self, hectares: Decimal) -> str:
        """Classify deforestation impact by area affected.

        Args:
            hectares: Area of deforestation in hectares.

        Returns:
            Severity level string.
        """
        if hectares >= self.deforestation_impact_critical_threshold_hectares:
            return "critical"
        elif hectares >= self.deforestation_impact_high_threshold_hectares:
            return "high"
        elif hectares > Decimal("0"):
            return "moderate"
        return "negligible"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[ContinuousMonitoringConfig] = None
_config_lock = threading.Lock()


def get_config() -> ContinuousMonitoringConfig:
    """Return the thread-safe singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ContinuousMonitoringConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
