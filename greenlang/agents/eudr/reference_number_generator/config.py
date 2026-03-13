# -*- coding: utf-8 -*-
"""
Reference Number Generator Configuration - AGENT-EUDR-038

Centralized configuration for the Reference Number Generator covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Reference number format templates for all 27 EU member states
- Sequence ranges: start, end, rollover strategy
- Checksum algorithm selection (Luhn, ISO 7064, CRC-16)
- Batch size limits and generation timeout
- Expiration periods (default 12 months)
- Performance tuning: generation rate limit, concurrent requests
- Distributed locking via Redis (TTL, retry, backoff)
- Collision detection parameters (max retries, bloom filter)
- Lifecycle management: active, used, expired, revoked, transferred
- Upstream agent URLs (documentation generator, due diligence)
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_rng_

All settings overridable via environment variables with ``GL_EUDR_RNG_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-038 Reference Number Generator (GL-EUDR-RNG-038)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 33
Status: Production Ready
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENV_PREFIX = "GL_EUDR_RNG_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_RNG_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_RNG_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_RNG_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_RNG_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# EU Member State Code to Country Name Mapping (27 states)
# ---------------------------------------------------------------------------

EU_MEMBER_STATES: Dict[str, str] = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LV": "Latvia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
}


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class ReferenceNumberGeneratorConfig:
    """Centralized configuration for AGENT-EUDR-038.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_RNG_ environment variables.
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
        default_factory=lambda: _env_int("DB_POOL_MAX", 20)
    )

    # -- 2. Redis (Distributed Locks & Cache) -----------------------------------
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
    redis_lock_ttl_seconds: int = field(
        default_factory=lambda: _env_int("REDIS_LOCK_TTL", 10)
    )
    redis_lock_retry_count: int = field(
        default_factory=lambda: _env_int("REDIS_LOCK_RETRY_COUNT", 5)
    )
    redis_lock_retry_delay_ms: int = field(
        default_factory=lambda: _env_int("REDIS_LOCK_RETRY_DELAY_MS", 100)
    )

    # -- 3. Reference Number Format ---------------------------------------------
    format_version: str = field(
        default_factory=lambda: _env("FORMAT_VERSION", "1.0")
    )
    reference_prefix: str = field(
        default_factory=lambda: _env("REFERENCE_PREFIX", "EUDR")
    )
    default_member_state: str = field(
        default_factory=lambda: _env("DEFAULT_MEMBER_STATE", "DE")
    )
    separator: str = field(
        default_factory=lambda: _env("SEPARATOR", "-")
    )
    # Format: {PREFIX}-{MS}-{YEAR}-{OPERATOR_CODE}-{SEQUENCE}-{CHECKSUM}
    # Example: EUDR-DE-2026-OP12345-000001-7
    format_template: str = field(
        default_factory=lambda: _env(
            "FORMAT_TEMPLATE",
            "{prefix}{sep}{ms}{sep}{year}{sep}{operator}{sep}{sequence}{sep}{checksum}",
        )
    )
    sequence_digits: int = field(
        default_factory=lambda: _env_int("SEQUENCE_DIGITS", 6)
    )
    operator_code_max_length: int = field(
        default_factory=lambda: _env_int("OPERATOR_CODE_MAX_LENGTH", 10)
    )

    # -- 4. Sequence Ranges -----------------------------------------------------
    sequence_start: int = field(
        default_factory=lambda: _env_int("SEQUENCE_START", 1)
    )
    sequence_end: int = field(
        default_factory=lambda: _env_int("SEQUENCE_END", 999999)
    )
    sequence_overflow_strategy: str = field(
        default_factory=lambda: _env("SEQUENCE_OVERFLOW_STRATEGY", "extend")
    )
    sequence_rollover_year: bool = field(
        default_factory=lambda: _env_bool("SEQUENCE_ROLLOVER_YEAR", True)
    )

    # -- 5. Checksum Algorithm --------------------------------------------------
    checksum_algorithm: str = field(
        default_factory=lambda: _env("CHECKSUM_ALGORITHM", "luhn")
    )
    checksum_length: int = field(
        default_factory=lambda: _env_int("CHECKSUM_LENGTH", 1)
    )

    # -- 6. Batch Processing ----------------------------------------------------
    max_batch_size: int = field(
        default_factory=lambda: _env_int("MAX_BATCH_SIZE", 10000)
    )
    batch_chunk_size: int = field(
        default_factory=lambda: _env_int("BATCH_CHUNK_SIZE", 500)
    )
    batch_timeout_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_TIMEOUT", 300)
    )
    max_concurrent_batches: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_BATCHES", 5)
    )

    # -- 7. Generation Timeout & Performance ------------------------------------
    generation_timeout_seconds: int = field(
        default_factory=lambda: _env_int("GENERATION_TIMEOUT", 5)
    )
    generation_rate_limit_per_second: int = field(
        default_factory=lambda: _env_int("GENERATION_RATE_LIMIT", 10000)
    )
    max_concurrent_requests: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_REQUESTS", 100)
    )
    enable_bloom_filter: bool = field(
        default_factory=lambda: _env_bool("ENABLE_BLOOM_FILTER", True)
    )
    bloom_filter_capacity: int = field(
        default_factory=lambda: _env_int("BLOOM_FILTER_CAPACITY", 10000000)
    )
    bloom_filter_error_rate: float = field(
        default_factory=lambda: _env_float("BLOOM_FILTER_ERROR_RATE", 0.001)
    )

    # -- 8. Expiration Periods --------------------------------------------------
    default_expiration_months: int = field(
        default_factory=lambda: _env_int("DEFAULT_EXPIRATION_MONTHS", 12)
    )
    max_expiration_months: int = field(
        default_factory=lambda: _env_int("MAX_EXPIRATION_MONTHS", 60)
    )
    expiration_warning_days: int = field(
        default_factory=lambda: _env_int("EXPIRATION_WARNING_DAYS", 30)
    )

    # -- 9. Collision Detection -------------------------------------------------
    collision_max_retries: int = field(
        default_factory=lambda: _env_int("COLLISION_MAX_RETRIES", 10)
    )
    collision_backoff_base_ms: int = field(
        default_factory=lambda: _env_int("COLLISION_BACKOFF_BASE_MS", 5)
    )
    collision_backoff_max_ms: int = field(
        default_factory=lambda: _env_int("COLLISION_BACKOFF_MAX_MS", 500)
    )

    # -- 10. Lifecycle Management -----------------------------------------------
    enable_auto_expiration: bool = field(
        default_factory=lambda: _env_bool("ENABLE_AUTO_EXPIRATION", True)
    )
    retention_years: int = field(
        default_factory=lambda: _env_int("RETENTION_YEARS", 5)
    )
    allow_transfer: bool = field(
        default_factory=lambda: _env_bool("ALLOW_TRANSFER", True)
    )
    require_revocation_reason: bool = field(
        default_factory=lambda: _env_bool("REQUIRE_REVOCATION_REASON", True)
    )

    # -- 11. Idempotency -------------------------------------------------------
    enable_idempotency: bool = field(
        default_factory=lambda: _env_bool("ENABLE_IDEMPOTENCY", True)
    )
    idempotency_key_ttl_seconds: int = field(
        default_factory=lambda: _env_int("IDEMPOTENCY_KEY_TTL", 86400)
    )

    # -- 12. Upstream Agent URLs ------------------------------------------------
    documentation_generator_url: str = field(
        default_factory=lambda: _env(
            "DOCUMENTATION_GENERATOR_URL",
            "http://eudr-doc-generator:8030/api/v1/eudr/documentation-generator",
        )
    )
    due_diligence_url: str = field(
        default_factory=lambda: _env(
            "DUE_DILIGENCE_URL",
            "http://eudr-due-diligence:8026/api/v1/eudr/due-diligence",
        )
    )
    eu_information_system_url: str = field(
        default_factory=lambda: _env(
            "EU_INFORMATION_SYSTEM_URL",
            "https://eudr-is.europa.eu/api/v1",
        )
    )

    # -- 13. Rate Limiting ------------------------------------------------------
    rate_limit_anonymous: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ANONYMOUS", 10)
    )
    rate_limit_basic: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_BASIC", 50)
    )
    rate_limit_standard: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_STANDARD", 200)
    )
    rate_limit_premium: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PREMIUM", 1000)
    )
    rate_limit_admin: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ADMIN", 5000)
    )

    # -- 14. Circuit Breaker ----------------------------------------------------
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5)
    )
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _env_int("CB_RESET_TIMEOUT", 60)
    )
    circuit_breaker_half_open_max: int = field(
        default_factory=lambda: _env_int("CB_HALF_OPEN_MAX", 3)
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
    metrics_prefix: str = "gl_eudr_rng_"

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate member state
        if self.default_member_state not in EU_MEMBER_STATES:
            logger.warning(
                "Default member state '%s' is not a valid EU member state.",
                self.default_member_state,
            )

        # Validate sequence range
        if self.sequence_start < 0:
            logger.warning(
                "Sequence start %d must be non-negative.",
                self.sequence_start,
            )
        if self.sequence_end <= self.sequence_start:
            logger.warning(
                "Sequence end %d must be greater than start %d.",
                self.sequence_end,
                self.sequence_start,
            )

        # Validate sequence digits
        max_from_digits = (10 ** self.sequence_digits) - 1
        if self.sequence_end > max_from_digits:
            logger.warning(
                "Sequence end %d exceeds max for %d digits (%d).",
                self.sequence_end,
                self.sequence_digits,
                max_from_digits,
            )

        # Validate checksum algorithm
        valid_algorithms = ("luhn", "iso7064", "crc16", "modulo97")
        if self.checksum_algorithm.lower() not in valid_algorithms:
            logger.warning(
                "Checksum algorithm '%s' is not one of %s.",
                self.checksum_algorithm,
                valid_algorithms,
            )

        # Validate overflow strategy
        valid_strategies = ("extend", "reject", "rollover")
        if self.sequence_overflow_strategy.lower() not in valid_strategies:
            logger.warning(
                "Overflow strategy '%s' is not one of %s.",
                self.sequence_overflow_strategy,
                valid_strategies,
            )

        # Validate batch size
        if self.max_batch_size < 1:
            logger.warning(
                "Max batch size %d must be at least 1.",
                self.max_batch_size,
            )

        # Validate pool sizing
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min,
                self.db_pool_max,
            )

        # Validate retention years
        if self.retention_years < 5:
            logger.warning(
                "Retention years %d is below EUDR minimum of 5 years "
                "(Article 31).",
                self.retention_years,
            )

        # Validate expiration
        if self.default_expiration_months < 1:
            logger.warning(
                "Default expiration months %d must be at least 1.",
                self.default_expiration_months,
            )
        if self.default_expiration_months > self.max_expiration_months:
            logger.warning(
                "Default expiration %d exceeds max expiration %d months.",
                self.default_expiration_months,
                self.max_expiration_months,
            )

        logger.info(
            "ReferenceNumberGeneratorConfig initialized: "
            "prefix=%s, format_version=%s, default_ms=%s, "
            "seq_range=[%d, %d], seq_digits=%d, "
            "checksum=%s, batch_max=%d, "
            "expiration_months=%d, retention_years=%d, "
            "rate_limit_std=%d/s, concurrent=%d",
            self.reference_prefix,
            self.format_version,
            self.default_member_state,
            self.sequence_start,
            self.sequence_end,
            self.sequence_digits,
            self.checksum_algorithm,
            self.max_batch_size,
            self.default_expiration_months,
            self.retention_years,
            self.rate_limit_standard,
            self.max_concurrent_requests,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[ReferenceNumberGeneratorConfig] = None
_config_lock = threading.Lock()


def get_config() -> ReferenceNumberGeneratorConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        ReferenceNumberGeneratorConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ReferenceNumberGeneratorConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
