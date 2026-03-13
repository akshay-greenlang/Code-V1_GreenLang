# -*- coding: utf-8 -*-
"""
EU Information System Interface Configuration - AGENT-EUDR-036

Centralized configuration for the EU Information System Interface covering:
- Database and cache connection settings (PostgreSQL, Redis)
- EU Information System API: base URL, version, auth credentials, timeouts
- DDS submission: max size, retry policy, validation strictness
- Operator registration: EORI format, registration expiry, renewal
- Geolocation formatting: coordinate precision, CRS system, polygon limits
- Package assembly: max attachments, compression, document ordering
- Status tracking: polling intervals, TTL for status cache, lifecycle states
- API client: connection pooling, TLS settings, rate limiting
- Audit recording: Article 31 requirements, retention period, detail level
- Upstream agent URLs: EUDR-035, EUDR-030, EUDR-026 references
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_euis_

All settings overridable via environment variables with ``GL_EUDR_EUIS_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-036 EU Information System Interface (GL-EUDR-EUIS-036)
Regulation: EU 2023/1115 (EUDR) Articles 4, 12, 13, 14, 31, 33
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

_ENV_PREFIX = "GL_EUDR_EUIS_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_EUIS_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_EUIS_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_EUIS_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_EUIS_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_EUIS_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class EUInformationSystemInterfaceConfig:
    """Centralized configuration for AGENT-EUDR-036.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_EUIS_ environment variables.
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

    # -- 3. EU Information System API -------------------------------------------
    eu_api_base_url: str = field(
        default_factory=lambda: _env(
            "EU_API_BASE_URL",
            "https://eudr-is.europa.eu/api/v1",
        )
    )
    eu_api_version: str = field(
        default_factory=lambda: _env("EU_API_VERSION", "v1")
    )
    eu_api_client_id: str = field(
        default_factory=lambda: _env("EU_API_CLIENT_ID", "")
    )
    eu_api_client_secret: str = field(
        default_factory=lambda: _env("EU_API_CLIENT_SECRET", "")
    )
    eu_api_certificate_path: str = field(
        default_factory=lambda: _env("EU_API_CERTIFICATE_PATH", "")
    )
    eu_api_key_path: str = field(
        default_factory=lambda: _env("EU_API_KEY_PATH", "")
    )
    eu_api_timeout_seconds: int = field(
        default_factory=lambda: _env_int("EU_API_TIMEOUT_SECONDS", 60)
    )
    eu_api_connect_timeout_seconds: int = field(
        default_factory=lambda: _env_int("EU_API_CONNECT_TIMEOUT", 15)
    )
    eu_api_max_retries: int = field(
        default_factory=lambda: _env_int("EU_API_MAX_RETRIES", 3)
    )
    eu_api_retry_backoff_factor: float = field(
        default_factory=lambda: _env_float("EU_API_RETRY_BACKOFF", 1.5)
    )

    # -- 4. DDS Submission Settings ---------------------------------------------
    dds_max_size_bytes: int = field(
        default_factory=lambda: _env_int("DDS_MAX_SIZE_BYTES", 52428800)
    )
    dds_validation_strict: bool = field(
        default_factory=lambda: _env_bool("DDS_VALIDATION_STRICT", True)
    )
    dds_auto_submit: bool = field(
        default_factory=lambda: _env_bool("DDS_AUTO_SUBMIT", False)
    )
    dds_draft_ttl_hours: int = field(
        default_factory=lambda: _env_int("DDS_DRAFT_TTL_HOURS", 168)
    )
    dds_max_commodities_per_statement: int = field(
        default_factory=lambda: _env_int("DDS_MAX_COMMODITIES", 50)
    )
    dds_required_fields: str = field(
        default_factory=lambda: _env(
            "DDS_REQUIRED_FIELDS",
            "operator_id,commodity,geolocation,risk_assessment,volume",
        )
    )

    # -- 5. Operator Registration -----------------------------------------------
    eori_format_pattern: str = field(
        default_factory=lambda: _env(
            "EORI_FORMAT_PATTERN",
            r"^[A-Z]{2}[0-9A-Z]{1,15}$",
        )
    )
    registration_expiry_days: int = field(
        default_factory=lambda: _env_int("REGISTRATION_EXPIRY_DAYS", 365)
    )
    registration_renewal_notice_days: int = field(
        default_factory=lambda: _env_int("REGISTRATION_RENEWAL_NOTICE", 30)
    )
    max_operators_per_account: int = field(
        default_factory=lambda: _env_int("MAX_OPERATORS_PER_ACCOUNT", 100)
    )

    # -- 6. Geolocation Formatting ----------------------------------------------
    coordinate_precision: int = field(
        default_factory=lambda: _env_int("COORDINATE_PRECISION", 6)
    )
    coordinate_reference_system: str = field(
        default_factory=lambda: _env("CRS", "EPSG:4326")
    )
    max_polygon_vertices: int = field(
        default_factory=lambda: _env_int("MAX_POLYGON_VERTICES", 500)
    )
    geolocation_area_threshold_ha: Decimal = field(
        default_factory=lambda: _env_decimal("AREA_THRESHOLD_HA", "4.0")
    )
    polygon_simplification_tolerance: Decimal = field(
        default_factory=lambda: _env_decimal(
            "POLYGON_SIMPLIFY_TOLERANCE", "0.0001"
        )
    )

    # -- 7. Package Assembly ----------------------------------------------------
    max_attachments_per_package: int = field(
        default_factory=lambda: _env_int("MAX_ATTACHMENTS", 100)
    )
    max_attachment_size_bytes: int = field(
        default_factory=lambda: _env_int("MAX_ATTACHMENT_SIZE", 10485760)
    )
    compress_packages: bool = field(
        default_factory=lambda: _env_bool("COMPRESS_PACKAGES", True)
    )
    package_format: str = field(
        default_factory=lambda: _env("PACKAGE_FORMAT", "json")
    )
    include_provenance_in_package: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_PROVENANCE", True)
    )

    # -- 8. Status Tracking -----------------------------------------------------
    status_poll_interval_seconds: int = field(
        default_factory=lambda: _env_int("STATUS_POLL_INTERVAL", 300)
    )
    status_cache_ttl_seconds: int = field(
        default_factory=lambda: _env_int("STATUS_CACHE_TTL", 60)
    )
    max_poll_attempts: int = field(
        default_factory=lambda: _env_int("MAX_POLL_ATTEMPTS", 288)
    )
    submission_timeout_hours: int = field(
        default_factory=lambda: _env_int("SUBMISSION_TIMEOUT_HOURS", 72)
    )

    # -- 9. API Client Connection Pool ------------------------------------------
    api_pool_max_connections: int = field(
        default_factory=lambda: _env_int("API_POOL_MAX_CONNECTIONS", 20)
    )
    api_pool_max_keepalive: int = field(
        default_factory=lambda: _env_int("API_POOL_MAX_KEEPALIVE", 10)
    )
    api_tls_verify: bool = field(
        default_factory=lambda: _env_bool("API_TLS_VERIFY", True)
    )
    api_tls_min_version: str = field(
        default_factory=lambda: _env("API_TLS_MIN_VERSION", "TLSv1.3")
    )

    # -- 10. Audit Recording (Article 31) ---------------------------------------
    audit_retention_years: int = field(
        default_factory=lambda: _env_int("AUDIT_RETENTION_YEARS", 5)
    )
    audit_detail_level: str = field(
        default_factory=lambda: _env("AUDIT_DETAIL_LEVEL", "full")
    )
    audit_include_request_body: bool = field(
        default_factory=lambda: _env_bool("AUDIT_INCLUDE_REQUEST_BODY", True)
    )
    audit_include_response_body: bool = field(
        default_factory=lambda: _env_bool("AUDIT_INCLUDE_RESPONSE_BODY", True)
    )

    # -- 11. Upstream Agent URLs ------------------------------------------------
    improvement_plan_url: str = field(
        default_factory=lambda: _env(
            "IMPROVEMENT_PLAN_URL",
            "http://eudr-improvement-plan:8035/api/v1/eudr/improvement-plan",
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
    geolocation_verification_url: str = field(
        default_factory=lambda: _env(
            "GEOLOCATION_VERIFICATION_URL",
            "http://eudr-geolocation:8002/api/v1/eudr/geolocation",
        )
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
    metrics_prefix: str = "gl_eudr_euis_"

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate pool sizing
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min,
                self.db_pool_max,
            )

        # Validate coordinate precision
        if self.coordinate_precision < 1 or self.coordinate_precision > 15:
            logger.warning(
                "Coordinate precision %d is outside recommended range [1, 15].",
                self.coordinate_precision,
            )

        # Validate area threshold
        if self.geolocation_area_threshold_ha <= Decimal("0"):
            logger.warning(
                "Geolocation area threshold %s must be positive.",
                self.geolocation_area_threshold_ha,
            )

        # Validate API timeouts
        if self.eu_api_connect_timeout_seconds >= self.eu_api_timeout_seconds:
            logger.warning(
                "Connect timeout %ds >= total timeout %ds. "
                "This may cause premature connection failures.",
                self.eu_api_connect_timeout_seconds,
                self.eu_api_timeout_seconds,
            )

        # Validate audit retention
        if self.audit_retention_years < 5:
            logger.warning(
                "Audit retention %d years is below EUDR Article 31 "
                "minimum of 5 years.",
                self.audit_retention_years,
            )

        # Validate DDS size limit
        if self.dds_max_size_bytes < 1048576:
            logger.warning(
                "DDS max size %d bytes is very small. "
                "Consider increasing for production use.",
                self.dds_max_size_bytes,
            )

        logger.info(
            "EUInformationSystemInterfaceConfig initialized: "
            "eu_api=%s, api_version=%s, "
            "dds_strict=%s, auto_submit=%s, "
            "coord_precision=%d, crs=%s, "
            "audit_retention=%d years, "
            "max_attachments=%d, compress=%s, "
            "status_poll=%ds, max_poll=%d",
            self.eu_api_base_url,
            self.eu_api_version,
            self.dds_validation_strict,
            self.dds_auto_submit,
            self.coordinate_precision,
            self.coordinate_reference_system,
            self.audit_retention_years,
            self.max_attachments_per_package,
            self.compress_packages,
            self.status_poll_interval_seconds,
            self.max_poll_attempts,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[EUInformationSystemInterfaceConfig] = None
_config_lock = threading.Lock()


def get_config() -> EUInformationSystemInterfaceConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        EUInformationSystemInterfaceConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = EUInformationSystemInterfaceConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
