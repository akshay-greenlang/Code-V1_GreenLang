# -*- coding: utf-8 -*-
"""
Authority Communication Manager Configuration - AGENT-EUDR-040

Centralized configuration for the Authority Communication Manager covering:
- Database and cache connection settings (PostgreSQL, Redis for message queue)
- 27 EU member state authority configurations with contact endpoints
- Multi-language support (24 official EU languages)
- Email/API notification channel settings
- Response deadline defaults (urgent: 24h, normal: 5 days, routine: 15 days)
- Template repository paths for multi-language correspondence
- Encryption settings for sensitive documents (AES-256)
- Deadline tracking and automated reminder configuration
- GDPR compliance settings (right to erasure, data minimization)
- Upstream agent URLs for EUDR-026 to EUDR-030 integration
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_acm_
- Performance tuning: connection pooling, cache TTL, timeouts

All settings overridable via environment variables with ``GL_EUDR_ACM_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-040 Authority Communication Manager (GL-EUDR-ACM-040)
Regulation: EU 2023/1115 (EUDR) Articles 15, 16, 17, 19, 31
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

_ENV_PREFIX = "GL_EUDR_ACM_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_ACM_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_ACM_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_ACM_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_ACM_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_ACM_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# EU Member State Authority Registry
# ---------------------------------------------------------------------------

EU_MEMBER_STATES: Dict[str, Dict[str, str]] = {
    "AT": {"name": "Austria", "authority": "Austrian Federal Ministry of Agriculture", "language": "de", "endpoint": "https://eudr-portal.at/api/v1"},
    "BE": {"name": "Belgium", "authority": "Belgian Federal Public Service Health", "language": "nl", "endpoint": "https://eudr-portal.be/api/v1"},
    "BG": {"name": "Bulgaria", "authority": "Bulgarian Executive Forest Agency", "language": "bg", "endpoint": "https://eudr-portal.bg/api/v1"},
    "HR": {"name": "Croatia", "authority": "Croatian Ministry of Agriculture", "language": "hr", "endpoint": "https://eudr-portal.hr/api/v1"},
    "CY": {"name": "Cyprus", "authority": "Cyprus Department of Forests", "language": "el", "endpoint": "https://eudr-portal.cy/api/v1"},
    "CZ": {"name": "Czech Republic", "authority": "Czech Environmental Inspectorate", "language": "cs", "endpoint": "https://eudr-portal.cz/api/v1"},
    "DK": {"name": "Denmark", "authority": "Danish Environmental Protection Agency", "language": "da", "endpoint": "https://eudr-portal.dk/api/v1"},
    "EE": {"name": "Estonia", "authority": "Estonian Environment Agency", "language": "et", "endpoint": "https://eudr-portal.ee/api/v1"},
    "FI": {"name": "Finland", "authority": "Finnish Food Authority", "language": "fi", "endpoint": "https://eudr-portal.fi/api/v1"},
    "FR": {"name": "France", "authority": "French Ministry of Ecological Transition", "language": "fr", "endpoint": "https://eudr-portal.fr/api/v1"},
    "DE": {"name": "Germany", "authority": "German Federal Agency for Nature Conservation", "language": "de", "endpoint": "https://eudr-portal.de/api/v1"},
    "GR": {"name": "Greece", "authority": "Greek Ministry of Environment", "language": "el", "endpoint": "https://eudr-portal.gr/api/v1"},
    "HU": {"name": "Hungary", "authority": "Hungarian National Food Chain Safety Office", "language": "hu", "endpoint": "https://eudr-portal.hu/api/v1"},
    "IE": {"name": "Ireland", "authority": "Irish Department of Agriculture", "language": "en", "endpoint": "https://eudr-portal.ie/api/v1"},
    "IT": {"name": "Italy", "authority": "Italian Ministry of Agriculture", "language": "it", "endpoint": "https://eudr-portal.it/api/v1"},
    "LV": {"name": "Latvia", "authority": "Latvian State Forest Service", "language": "lv", "endpoint": "https://eudr-portal.lv/api/v1"},
    "LT": {"name": "Lithuania", "authority": "Lithuanian State Forest Service", "language": "lt", "endpoint": "https://eudr-portal.lt/api/v1"},
    "LU": {"name": "Luxembourg", "authority": "Luxembourg Nature and Forest Administration", "language": "fr", "endpoint": "https://eudr-portal.lu/api/v1"},
    "MT": {"name": "Malta", "authority": "Malta Environment and Resources Authority", "language": "mt", "endpoint": "https://eudr-portal.mt/api/v1"},
    "NL": {"name": "Netherlands", "authority": "Netherlands Food and Consumer Product Safety Authority", "language": "nl", "endpoint": "https://eudr-portal.nl/api/v1"},
    "PL": {"name": "Poland", "authority": "Polish Inspection of Environmental Protection", "language": "pl", "endpoint": "https://eudr-portal.pl/api/v1"},
    "PT": {"name": "Portugal", "authority": "Portuguese Institute for Nature Conservation", "language": "pt", "endpoint": "https://eudr-portal.pt/api/v1"},
    "RO": {"name": "Romania", "authority": "Romanian National Environmental Guard", "language": "ro", "endpoint": "https://eudr-portal.ro/api/v1"},
    "SK": {"name": "Slovakia", "authority": "Slovak Inspection of Environment", "language": "sk", "endpoint": "https://eudr-portal.sk/api/v1"},
    "SI": {"name": "Slovenia", "authority": "Slovenian Inspectorate for Environment", "language": "sl", "endpoint": "https://eudr-portal.si/api/v1"},
    "ES": {"name": "Spain", "authority": "Spanish Ministry for Ecological Transition", "language": "es", "endpoint": "https://eudr-portal.es/api/v1"},
    "SE": {"name": "Sweden", "authority": "Swedish Environmental Protection Agency", "language": "sv", "endpoint": "https://eudr-portal.se/api/v1"},
}

# 24 official EU languages
EU_LANGUAGES: List[str] = [
    "bg", "cs", "da", "de", "el", "en", "es", "et",
    "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv",
    "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
]


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class AuthorityCommunicationManagerConfig:
    """Centralized configuration for AGENT-EUDR-040.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_ACM_ environment variables.
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

    # -- 2. Redis (Message Queue) -----------------------------------------------
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
    message_queue_prefix: str = field(
        default_factory=lambda: _env("MQ_PREFIX", "gl_eudr_acm_mq")
    )

    # -- 3. Response Deadlines --------------------------------------------------
    deadline_urgent_hours: int = field(
        default_factory=lambda: _env_int("DEADLINE_URGENT_HOURS", 24)
    )
    deadline_normal_days: int = field(
        default_factory=lambda: _env_int("DEADLINE_NORMAL_DAYS", 5)
    )
    deadline_routine_days: int = field(
        default_factory=lambda: _env_int("DEADLINE_ROUTINE_DAYS", 15)
    )
    deadline_appeal_days: int = field(
        default_factory=lambda: _env_int("DEADLINE_APPEAL_DAYS", 30)
    )
    reminder_before_deadline_hours: int = field(
        default_factory=lambda: _env_int("REMINDER_BEFORE_DEADLINE_HOURS", 48)
    )
    escalation_after_deadline_hours: int = field(
        default_factory=lambda: _env_int("ESCALATION_AFTER_DEADLINE_HOURS", 24)
    )

    # -- 4. Multi-Language Support -----------------------------------------------
    default_language: str = field(
        default_factory=lambda: _env("DEFAULT_LANGUAGE", "en")
    )
    supported_languages: List[str] = field(
        default_factory=lambda: EU_LANGUAGES.copy()
    )
    template_base_path: str = field(
        default_factory=lambda: _env(
            "TEMPLATE_BASE_PATH",
            "/data/eudr/acm/templates"
        )
    )
    template_fallback_language: str = field(
        default_factory=lambda: _env("TEMPLATE_FALLBACK_LANGUAGE", "en")
    )

    # -- 5. Notification Channels ------------------------------------------------
    email_smtp_host: str = field(
        default_factory=lambda: _env("SMTP_HOST", "smtp.greenlang.io")
    )
    email_smtp_port: int = field(
        default_factory=lambda: _env_int("SMTP_PORT", 587)
    )
    email_smtp_user: str = field(
        default_factory=lambda: _env("SMTP_USER", "")
    )
    email_smtp_password: str = field(
        default_factory=lambda: _env("SMTP_PASSWORD", "")
    )
    email_from_address: str = field(
        default_factory=lambda: _env(
            "EMAIL_FROM", "eudr-compliance@greenlang.io"
        )
    )
    email_reply_to: str = field(
        default_factory=lambda: _env(
            "EMAIL_REPLY_TO", "eudr-support@greenlang.io"
        )
    )
    api_notification_enabled: bool = field(
        default_factory=lambda: _env_bool("API_NOTIFICATION_ENABLED", True)
    )
    portal_notification_enabled: bool = field(
        default_factory=lambda: _env_bool("PORTAL_NOTIFICATION_ENABLED", True)
    )

    # -- 6. Encryption (AES-256) -------------------------------------------------
    encryption_enabled: bool = field(
        default_factory=lambda: _env_bool("ENCRYPTION_ENABLED", True)
    )
    encryption_algorithm: str = field(
        default_factory=lambda: _env("ENCRYPTION_ALGORITHM", "AES-256-GCM")
    )
    encryption_key_id: str = field(
        default_factory=lambda: _env("ENCRYPTION_KEY_ID", "eudr-acm-doc-key-v1")
    )
    encryption_key_rotation_days: int = field(
        default_factory=lambda: _env_int("ENCRYPTION_KEY_ROTATION_DAYS", 90)
    )

    # -- 7. Penalty Ranges -------------------------------------------------------
    penalty_min_amount: Decimal = field(
        default_factory=lambda: _env_decimal("PENALTY_MIN_AMOUNT", "1000")
    )
    penalty_max_amount: Decimal = field(
        default_factory=lambda: _env_decimal("PENALTY_MAX_AMOUNT", "10000000")
    )
    penalty_currency: str = field(
        default_factory=lambda: _env("PENALTY_CURRENCY", "EUR")
    )

    # -- 8. Inspection Settings --------------------------------------------------
    inspection_notice_days: int = field(
        default_factory=lambda: _env_int("INSPECTION_NOTICE_DAYS", 5)
    )
    inspection_max_duration_hours: int = field(
        default_factory=lambda: _env_int("INSPECTION_MAX_DURATION_HOURS", 48)
    )
    inspection_follow_up_days: int = field(
        default_factory=lambda: _env_int("INSPECTION_FOLLOW_UP_DAYS", 14)
    )

    # -- 9. Appeal Settings ------------------------------------------------------
    appeal_window_days: int = field(
        default_factory=lambda: _env_int("APPEAL_WINDOW_DAYS", 60)
    )
    appeal_max_extensions: int = field(
        default_factory=lambda: _env_int("APPEAL_MAX_EXTENSIONS", 2)
    )
    appeal_extension_days: int = field(
        default_factory=lambda: _env_int("APPEAL_EXTENSION_DAYS", 30)
    )

    # -- 10. GDPR Compliance -----------------------------------------------------
    gdpr_data_retention_days: int = field(
        default_factory=lambda: _env_int("GDPR_DATA_RETENTION_DAYS", 1825)
    )
    gdpr_erasure_enabled: bool = field(
        default_factory=lambda: _env_bool("GDPR_ERASURE_ENABLED", True)
    )
    gdpr_minimization_enabled: bool = field(
        default_factory=lambda: _env_bool("GDPR_MINIMIZATION_ENABLED", True)
    )
    gdpr_audit_log_retention_days: int = field(
        default_factory=lambda: _env_int("GDPR_AUDIT_LOG_RETENTION_DAYS", 3650)
    )

    # -- 11. Upstream Agent URLs -------------------------------------------------
    due_diligence_orchestrator_url: str = field(
        default_factory=lambda: _env(
            "DUE_DILIGENCE_ORCHESTRATOR_URL",
            "http://eudr-due-diligence:8026/api/v1/eudr/due-diligence",
        )
    )
    information_gathering_url: str = field(
        default_factory=lambda: _env(
            "INFORMATION_GATHERING_URL",
            "http://eudr-info-gathering:8027/api/v1/eudr/information-gathering",
        )
    )
    risk_assessment_url: str = field(
        default_factory=lambda: _env(
            "RISK_ASSESSMENT_URL",
            "http://eudr-risk-assessment:8028/api/v1/eudr/risk-assessment",
        )
    )
    mitigation_designer_url: str = field(
        default_factory=lambda: _env(
            "MITIGATION_DESIGNER_URL",
            "http://eudr-mitigation:8029/api/v1/eudr/mitigation-measure-designer",
        )
    )
    documentation_generator_url: str = field(
        default_factory=lambda: _env(
            "DOCUMENTATION_GENERATOR_URL",
            "http://eudr-documentation:8030/api/v1/eudr/documentation-generator",
        )
    )

    # -- 12. Rate Limiting -------------------------------------------------------
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

    # -- 13. Circuit Breaker -----------------------------------------------------
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5)
    )
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _env_int("CB_RESET_TIMEOUT", 60)
    )
    circuit_breaker_half_open_max: int = field(
        default_factory=lambda: _env_int("CB_HALF_OPEN_MAX", 3)
    )

    # -- 14. Batch Processing ----------------------------------------------------
    max_concurrent: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT", 10)
    )
    batch_timeout_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_TIMEOUT", 300)
    )

    # -- 15. Provenance ----------------------------------------------------------
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

    # -- 16. Metrics -------------------------------------------------------------
    metrics_enabled: bool = field(
        default_factory=lambda: _env_bool("METRICS_ENABLED", True)
    )
    metrics_prefix: str = "gl_eudr_acm_"

    # -- Logging -----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate deadline ordering
        if self.deadline_urgent_hours >= self.deadline_normal_days * 24:
            logger.warning(
                "Urgent deadline (%dh) is not shorter than normal deadline (%dd). "
                "Urgent communications may not be prioritized correctly.",
                self.deadline_urgent_hours,
                self.deadline_normal_days,
            )

        # Validate reminder is before deadline
        if self.reminder_before_deadline_hours <= 0:
            logger.warning(
                "Reminder before deadline hours must be positive: %d",
                self.reminder_before_deadline_hours,
            )

        # Validate pool sizing
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min,
                self.db_pool_max,
            )

        # Validate penalty range
        if self.penalty_min_amount >= self.penalty_max_amount:
            logger.warning(
                "Penalty min %s >= max %s. Range is invalid.",
                self.penalty_min_amount,
                self.penalty_max_amount,
            )

        # Validate default language is supported
        if self.default_language not in self.supported_languages:
            logger.warning(
                "Default language '%s' not in supported languages list.",
                self.default_language,
            )

        # Validate GDPR retention
        if self.gdpr_data_retention_days < 365:
            logger.warning(
                "GDPR data retention %d days may be insufficient for "
                "EUDR Article 31 record-keeping (5 years minimum).",
                self.gdpr_data_retention_days,
            )

        logger.info(
            "AuthorityCommunicationManagerConfig initialized: "
            "deadlines (urgent=%dh, normal=%dd, routine=%dd), "
            "languages=%d, member_states=27, "
            "encryption=%s, gdpr_retention=%dd, "
            "penalty_range=[%s, %s] %s",
            self.deadline_urgent_hours,
            self.deadline_normal_days,
            self.deadline_routine_days,
            len(self.supported_languages),
            self.encryption_algorithm if self.encryption_enabled else "disabled",
            self.gdpr_data_retention_days,
            self.penalty_min_amount,
            self.penalty_max_amount,
            self.penalty_currency,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[AuthorityCommunicationManagerConfig] = None
_config_lock = threading.Lock()


def get_config() -> AuthorityCommunicationManagerConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        AuthorityCommunicationManagerConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AuthorityCommunicationManagerConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
