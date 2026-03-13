# -*- coding: utf-8 -*-
"""
Stakeholder Engagement Configuration - AGENT-EUDR-031

Centralized configuration for the Stakeholder Engagement Tool covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Stakeholder settings: max per supply chain, auto-discovery
- FPIC settings: deliberation days, stage SLA defaults
- Grievance settings: acknowledgement SLA, resolution SLA by severity,
  supported languages
- Consultation settings: offline sync, evidence storage
- Communication settings: supported channels, template library, retries
- Engagement scoring: dimension weights, minimum acceptable score
- Compliance settings: retention years, report formats
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_set_

All settings overridable via environment variables with ``GL_EUDR_SET_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 Stakeholder Engagement Tool (GL-EUDR-SET-031)
Regulation: EU 2023/1115 (EUDR) Articles 2, 4, 8, 9, 10, 11, 12, 29, 31
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

_ENV_PREFIX = "GL_EUDR_SET_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_SET_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_SET_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_SET_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_SET_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_SET_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class StakeholderEngagementConfig:
    """Centralized configuration for AGENT-EUDR-031.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_SET_ environment variables.
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

    # -- 3. Stakeholder Settings ------------------------------------------------
    max_stakeholders_per_operator: int = field(
        default_factory=lambda: _env_int("MAX_STAKEHOLDERS_PER_OPERATOR", 500)
    )
    discovery_radius_km: int = field(
        default_factory=lambda: _env_int("DISCOVERY_RADIUS_KM", 50)
    )
    enable_auto_discovery: bool = field(
        default_factory=lambda: _env_bool("ENABLE_AUTO_DISCOVERY", True)
    )
    rights_classification_enabled: bool = field(
        default_factory=lambda: _env_bool("RIGHTS_CLASSIFICATION_ENABLED", True)
    )
    stakeholder_dedup_threshold: Decimal = field(
        default_factory=lambda: _env_decimal(
            "STAKEHOLDER_DEDUP_THRESHOLD", "0.85"
        )
    )

    # -- 4. FPIC Settings -------------------------------------------------------
    fpic_notification_period_days: int = field(
        default_factory=lambda: _env_int("FPIC_NOTIFICATION_PERIOD_DAYS", 30)
    )
    fpic_deliberation_period_days: int = field(
        default_factory=lambda: _env_int("FPIC_DELIBERATION_PERIOD_DAYS", 90)
    )
    fpic_min_consultation_sessions: int = field(
        default_factory=lambda: _env_int("FPIC_MIN_CONSULTATION_SESSIONS", 3)
    )
    fpic_independent_facilitator_required: bool = field(
        default_factory=lambda: _env_bool("FPIC_INDEPENDENT_FACILITATOR_REQUIRED", True)
    )
    fpic_min_attendance_percentage: Decimal = field(
        default_factory=lambda: _env_decimal("FPIC_MIN_ATTENDANCE_PERCENTAGE", "60")
    )
    fpic_consent_validity_years: int = field(
        default_factory=lambda: _env_int("FPIC_CONSENT_VALIDITY_YEARS", 5)
    )
    fpic_require_evidence_per_stage: bool = field(
        default_factory=lambda: _env_bool("FPIC_REQUIRE_EVIDENCE", True)
    )

    # -- 5. Grievance Settings --------------------------------------------------
    grievance_acknowledgement_sla_hours: int = field(
        default_factory=lambda: _env_int("GRIEVANCE_ACK_SLA_HOURS", 48)
    )
    grievance_sla_critical_hours: int = field(
        default_factory=lambda: _env_int("GRIEVANCE_SLA_CRITICAL_HOURS", 24)
    )
    grievance_sla_high_hours: int = field(
        default_factory=lambda: _env_int("GRIEVANCE_SLA_HIGH_HOURS", 72)
    )
    grievance_sla_standard_days: int = field(
        default_factory=lambda: _env_int("GRIEVANCE_SLA_STANDARD_DAYS", 14)
    )
    grievance_sla_minor_days: int = field(
        default_factory=lambda: _env_int("GRIEVANCE_SLA_MINOR_DAYS", 30)
    )
    grievance_appeal_window_days: int = field(
        default_factory=lambda: _env_int("GRIEVANCE_APPEAL_WINDOW_DAYS", 30)
    )
    grievance_satisfaction_survey: bool = field(
        default_factory=lambda: _env_bool("GRIEVANCE_SATISFACTION_SURVEY", True)
    )
    grievance_supported_languages: List[str] = field(
        default_factory=lambda: [
            "en", "fr", "de", "es", "pt", "id", "sw",
            "ar", "zh", "hi", "ms", "th",
        ]
    )
    grievance_anonymous_reporting: bool = field(
        default_factory=lambda: _env_bool("GRIEVANCE_ANONYMOUS", True)
    )

    # -- 6. Consultation Settings -----------------------------------------------
    consultation_min_notice_days: int = field(
        default_factory=lambda: _env_int("CONSULTATION_MIN_NOTICE_DAYS", 14)
    )
    consultation_quorum_percentage: Decimal = field(
        default_factory=lambda: _env_decimal("CONSULTATION_QUORUM_PERCENTAGE", "50")
    )
    consultation_require_minutes: bool = field(
        default_factory=lambda: _env_bool("CONSULTATION_REQUIRE_MINUTES", True)
    )
    consultation_require_attendance: bool = field(
        default_factory=lambda: _env_bool("CONSULTATION_REQUIRE_ATTENDANCE", True)
    )
    consultation_offline_sync_enabled: bool = field(
        default_factory=lambda: _env_bool("CONSULTATION_OFFLINE_SYNC", True)
    )
    consultation_evidence_storage_path: str = field(
        default_factory=lambda: _env(
            "CONSULTATION_EVIDENCE_PATH",
            "/data/eudr/stakeholder/consultation_evidence",
        )
    )
    consultation_immutable_after_finalize: bool = field(
        default_factory=lambda: _env_bool(
            "CONSULTATION_IMMUTABLE_FINALIZED", True
        )
    )

    # -- 7. Communication Settings ----------------------------------------------
    communication_max_batch_size: int = field(
        default_factory=lambda: _env_int("COMMUNICATION_MAX_BATCH_SIZE", 100)
    )
    communication_retry_attempts: int = field(
        default_factory=lambda: _env_int("COMMUNICATION_RETRY_ATTEMPTS", 3)
    )
    communication_retry_delay_seconds: int = field(
        default_factory=lambda: _env_int("COMMUNICATION_RETRY_DELAY_SECONDS", 60)
    )
    communication_default_language: str = field(
        default_factory=lambda: _env("COMMUNICATION_DEFAULT_LANGUAGE", "en")
    )
    communication_supported_channels: List[str] = field(
        default_factory=lambda: [
            "email", "sms", "whatsapp", "portal", "print", "radio",
        ]
    )
    communication_template_library_path: str = field(
        default_factory=lambda: _env(
            "COMMUNICATION_TEMPLATE_PATH",
            "/data/eudr/stakeholder/templates",
        )
    )

    # -- 8. Engagement Scoring --------------------------------------------------
    assessment_min_score: Decimal = field(
        default_factory=lambda: _env_decimal("ASSESSMENT_MIN_SCORE", "0")
    )
    assessment_max_score: Decimal = field(
        default_factory=lambda: _env_decimal("ASSESSMENT_MAX_SCORE", "100")
    )
    assessment_passing_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("ASSESSMENT_PASSING_THRESHOLD", "60")
    )
    assessment_high_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("ASSESSMENT_HIGH_THRESHOLD", "80")
    )
    engagement_weight_cultural_appropriateness: Decimal = field(
        default_factory=lambda: _env_decimal(
            "ENGAGEMENT_WEIGHT_CULTURAL", "0.20"
        )
    )
    engagement_weight_language_accessibility: Decimal = field(
        default_factory=lambda: _env_decimal(
            "ENGAGEMENT_WEIGHT_LANGUAGE", "0.15"
        )
    )
    engagement_weight_deliberation_time: Decimal = field(
        default_factory=lambda: _env_decimal(
            "ENGAGEMENT_WEIGHT_DELIBERATION", "0.15"
        )
    )
    engagement_weight_inclusiveness: Decimal = field(
        default_factory=lambda: _env_decimal(
            "ENGAGEMENT_WEIGHT_INCLUSIVENESS", "0.20"
        )
    )
    engagement_weight_genuineness: Decimal = field(
        default_factory=lambda: _env_decimal(
            "ENGAGEMENT_WEIGHT_GENUINENESS", "0.15"
        )
    )
    engagement_weight_decision_respect: Decimal = field(
        default_factory=lambda: _env_decimal(
            "ENGAGEMENT_WEIGHT_DECISION_RESPECT", "0.15"
        )
    )
    engagement_min_acceptable_score: int = field(
        default_factory=lambda: _env_int("ENGAGEMENT_MIN_SCORE", 60)
    )

    # -- 9. Compliance Settings -------------------------------------------------
    report_format: str = field(
        default_factory=lambda: _env("REPORT_FORMAT", "json")
    )
    report_retention_years: int = field(
        default_factory=lambda: _env_int("REPORT_RETENTION_YEARS", 5)
    )
    include_provenance: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_PROVENANCE", True)
    )
    retention_years: int = field(
        default_factory=lambda: _env_int("RETENTION_YEARS", 5)
    )
    report_formats: List[str] = field(
        default_factory=lambda: ["json", "pdf", "html", "xlsx"]
    )
    default_report_format: str = field(
        default_factory=lambda: _env("DEFAULT_REPORT_FORMAT", "json")
    )
    include_regulatory_refs: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_REGULATORY_REFS", True)
    )

    # -- 10. Upstream Agent URLs ------------------------------------------------
    supply_chain_url: str = field(
        default_factory=lambda: _env(
            "SUPPLY_CHAIN_URL",
            "http://eudr-supply-chain:8001/api/v1/eudr/supply-chain",
        )
    )
    geolocation_url: str = field(
        default_factory=lambda: _env(
            "GEOLOCATION_URL",
            "http://eudr-geolocation:8002/api/v1/eudr/geolocation",
        )
    )
    indigenous_rights_url: str = field(
        default_factory=lambda: _env(
            "INDIGENOUS_RIGHTS_URL",
            "http://eudr-indigenous-rights:8021/api/v1/eudr/indigenous-rights",
        )
    )
    documentation_generator_url: str = field(
        default_factory=lambda: _env(
            "DOCUMENTATION_GENERATOR_URL",
            "http://eudr-documentation:8030/api/v1/eudr/documentation-generator",
        )
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
    metrics_prefix: str = field(
        default_factory=lambda: _env("METRICS_PREFIX", "gl_eudr_set_")
    )

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate FPIC deliberation days
        if self.fpic_notification_period_days < 14:
            logger.warning(
                "FPIC notification period days %d is below recommended minimum of 14.",
                self.fpic_notification_period_days,
            )

        if self.fpic_deliberation_period_days < 30:
            logger.warning(
                "FPIC deliberation period days %d is below recommended minimum of 30.",
                self.fpic_deliberation_period_days,
            )

        # Validate FPIC attendance percentage
        if not (0 <= self.fpic_min_attendance_percentage <= 100):
            logger.warning(
                "FPIC min attendance percentage %s is outside [0, 100].",
                self.fpic_min_attendance_percentage,
            )

        # Validate grievance SLA ordering
        if self.grievance_sla_critical_hours < 1:
            logger.warning(
                "Critical SLA %d hours is below minimum of 1 hour.",
                self.grievance_sla_critical_hours,
            )

        # Validate assessment thresholds
        if self.assessment_passing_threshold > self.assessment_max_score:
            logger.warning(
                "Assessment passing threshold %s exceeds max score %s.",
                self.assessment_passing_threshold,
                self.assessment_max_score,
            )

        if self.assessment_passing_threshold < self.assessment_min_score:
            logger.warning(
                "Assessment passing threshold %s is below min score %s.",
                self.assessment_passing_threshold,
                self.assessment_min_score,
            )

        # Validate consultation quorum percentage
        if not (0 <= self.consultation_quorum_percentage <= 100):
            logger.warning(
                "Consultation quorum percentage %s is outside [0, 100].",
                self.consultation_quorum_percentage,
            )

        # Validate discovery radius
        if self.discovery_radius_km < 0:
            logger.warning(
                "Discovery radius %d km is negative.",
                self.discovery_radius_km,
            )

        # Validate max stakeholders
        if self.max_stakeholders_per_operator < 1:
            logger.warning(
                "Max stakeholders per operator %d is below minimum of 1.",
                self.max_stakeholders_per_operator,
            )

        # Validate engagement dimension weights sum to 1.0
        total_weight = (
            self.engagement_weight_cultural_appropriateness
            + self.engagement_weight_language_accessibility
            + self.engagement_weight_deliberation_time
            + self.engagement_weight_inclusiveness
            + self.engagement_weight_genuineness
            + self.engagement_weight_decision_respect
        )
        if abs(total_weight - Decimal("1.0")) > Decimal("0.01"):
            logger.warning(
                "Engagement dimension weights sum to %s, expected 1.0.",
                total_weight,
            )

        # Validate pool sizing
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min,
                self.db_pool_max,
            )

        # Validate retention years
        if self.report_retention_years < 5:
            logger.warning(
                "Report retention years %d is below EUDR minimum of 5 years "
                "(Article 31).",
                self.report_retention_years,
            )

        # Validate report format
        if self.report_format not in ["json", "pdf", "html", "xlsx", "xml", "csv"]:
            logger.warning(
                "Report format %s is not in standard formats.",
                self.report_format,
            )

        # Validate engagement minimum score range
        if not (0 <= self.engagement_min_acceptable_score <= 100):
            logger.warning(
                "Engagement min acceptable score %d is outside [0, 100].",
                self.engagement_min_acceptable_score,
            )

        logger.info(
            "StakeholderEngagementConfig initialized: "
            "max_stakeholders=%d, auto_discovery=%s, "
            "fpic_deliberation=%d days, "
            "grievance_ack_sla=%dh, grievance_languages=%d, "
            "engagement_min_score=%d, retention_years=%d",
            self.max_stakeholders_per_operator,
            self.enable_auto_discovery,
            self.fpic_deliberation_period_days,
            self.grievance_acknowledgement_sla_hours,
            len(self.grievance_supported_languages),
            self.engagement_min_acceptable_score,
            self.report_retention_years,
        )

    def get_grievance_resolution_sla(self, severity: str) -> int:
        """Get grievance resolution SLA in days by severity.

        Args:
            severity: Grievance severity level.

        Returns:
            Resolution SLA in days.
        """
        sla_map: Dict[str, int] = {
            "critical": self.grievance_sla_critical_hours // 24 or 1,
            "high": self.grievance_sla_high_hours // 24 or 3,
            "standard": self.grievance_sla_standard_days,
            "minor": self.grievance_sla_minor_days,
        }
        return sla_map.get(severity.lower(), self.grievance_sla_standard_days)

    def get_engagement_dimension_weights(self) -> Dict[str, Decimal]:
        """Get engagement scoring dimension weights as a dictionary.

        Returns:
            Dictionary mapping dimension names to their weights.
        """
        return {
            "cultural_appropriateness": self.engagement_weight_cultural_appropriateness,
            "language_accessibility": self.engagement_weight_language_accessibility,
            "deliberation_time": self.engagement_weight_deliberation_time,
            "inclusiveness": self.engagement_weight_inclusiveness,
            "genuineness": self.engagement_weight_genuineness,
            "decision_respect": self.engagement_weight_decision_respect,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[StakeholderEngagementConfig] = None
_config_lock = threading.Lock()


def get_config() -> StakeholderEngagementConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        StakeholderEngagementConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = StakeholderEngagementConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
