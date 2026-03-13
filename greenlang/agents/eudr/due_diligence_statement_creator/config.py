# -*- coding: utf-8 -*-
"""
Due Diligence Statement Creator Configuration - AGENT-EUDR-037

Centralized configuration for the Due Diligence Statement Creator covering:
- Database and cache connection settings (PostgreSQL, Redis)
- DDS template settings: structure, mandatory fields, section ordering
- Multi-language support: 24 EU official languages
- Document size limits: max attachment size, max package size
- Signature requirements: digital signature algorithm, timestamp format
- Validation rules: Article 4 mandatory fields, geolocation precision
- Performance tuning: statement generation timeout, batch size
- Upstream agent URLs: EUDR-001 to 025, EUDR-036
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_ddsc_

All settings overridable via environment variables with ``GL_EUDR_DDSC_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-037 Due Diligence Statement Creator (GL-EUDR-DDSC-037)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 14, 31, 33
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

_ENV_PREFIX = "GL_EUDR_DDSC_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_DDSC_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_DDSC_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_DDSC_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_DDSC_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_DDSC_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class DDSCreatorConfig:
    """Centralized configuration for AGENT-EUDR-037.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_DDSC_ environment variables.
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

    # -- 3. DDS Template Settings -----------------------------------------------
    dds_template_version: str = field(
        default_factory=lambda: _env("DDS_TEMPLATE_VERSION", "1.0")
    )
    dds_max_commodities_per_statement: int = field(
        default_factory=lambda: _env_int("DDS_MAX_COMMODITIES", 50)
    )
    dds_max_plots_per_commodity: int = field(
        default_factory=lambda: _env_int("DDS_MAX_PLOTS", 10000)
    )
    dds_mandatory_sections: List[str] = field(
        default_factory=lambda: [
            "operator_information",
            "commodity_description",
            "country_of_production",
            "geolocation_of_plots",
            "quantity_and_volume",
            "supplier_information",
            "risk_assessment_summary",
            "compliance_declaration",
            "digital_signature",
        ]
    )
    dds_reference_number_prefix: str = field(
        default_factory=lambda: _env("DDS_REF_PREFIX", "GL-DDS")
    )

    # -- 4. Multi-Language Support (24 EU Official Languages) -------------------
    supported_languages: List[str] = field(
        default_factory=lambda: [
            "bg", "cs", "da", "de", "el", "en", "es", "et",
            "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv",
            "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
        ]
    )
    default_language: str = field(
        default_factory=lambda: _env("DEFAULT_LANGUAGE", "en")
    )
    translation_cache_ttl: int = field(
        default_factory=lambda: _env_int("TRANSLATION_CACHE_TTL", 86400)
    )

    # -- 5. Document Size Limits ------------------------------------------------
    max_attachment_size_mb: int = field(
        default_factory=lambda: _env_int("MAX_ATTACHMENT_SIZE_MB", 25)
    )
    max_package_size_mb: int = field(
        default_factory=lambda: _env_int("MAX_PACKAGE_SIZE_MB", 500)
    )
    max_attachments_per_statement: int = field(
        default_factory=lambda: _env_int("MAX_ATTACHMENTS", 100)
    )
    allowed_attachment_types: List[str] = field(
        default_factory=lambda: [
            "pdf", "jpg", "jpeg", "png", "tiff", "geojson",
            "kml", "gpx", "xlsx", "csv", "xml",
        ]
    )

    # -- 6. Signature Requirements ----------------------------------------------
    signature_algorithm: str = field(
        default_factory=lambda: _env("SIGNATURE_ALGORITHM", "RSA-SHA256")
    )
    signature_timestamp_format: str = field(
        default_factory=lambda: _env(
            "SIGNATURE_TIMESTAMP_FORMAT", "ISO-8601"
        )
    )
    signature_validity_days: int = field(
        default_factory=lambda: _env_int("SIGNATURE_VALIDITY_DAYS", 365)
    )
    require_qualified_signature: bool = field(
        default_factory=lambda: _env_bool("REQUIRE_QUALIFIED_SIGNATURE", True)
    )

    # -- 7. Validation Rules ----------------------------------------------------
    geolocation_precision_digits: int = field(
        default_factory=lambda: _env_int("GEOLOCATION_PRECISION", 6)
    )
    geolocation_max_polygon_vertices: int = field(
        default_factory=lambda: _env_int("MAX_POLYGON_VERTICES", 5000)
    )
    article4_mandatory_field_count: int = field(
        default_factory=lambda: _env_int("ARTICLE4_MANDATORY_FIELDS", 14)
    )
    quantity_tolerance_percent: Decimal = field(
        default_factory=lambda: _env_decimal(
            "QUANTITY_TOLERANCE_PERCENT", "0.5"
        )
    )
    min_risk_assessment_score: Decimal = field(
        default_factory=lambda: _env_decimal(
            "MIN_RISK_ASSESSMENT_SCORE", "0.0"
        )
    )
    max_risk_level_for_auto_submit: str = field(
        default_factory=lambda: _env("MAX_RISK_AUTO_SUBMIT", "low")
    )

    # -- 8. Performance Tuning --------------------------------------------------
    statement_generation_timeout_seconds: int = field(
        default_factory=lambda: _env_int("GENERATION_TIMEOUT", 120)
    )
    batch_size: int = field(
        default_factory=lambda: _env_int("BATCH_SIZE", 50)
    )
    parallel_engine_calls: int = field(
        default_factory=lambda: _env_int("PARALLEL_ENGINE_CALLS", 5)
    )
    geolocation_formatting_timeout_seconds: int = field(
        default_factory=lambda: _env_int("GEOLOCATION_TIMEOUT", 30)
    )
    risk_integration_timeout_seconds: int = field(
        default_factory=lambda: _env_int("RISK_INTEGRATION_TIMEOUT", 60)
    )
    supply_chain_compilation_timeout_seconds: int = field(
        default_factory=lambda: _env_int("SUPPLY_CHAIN_TIMEOUT", 60)
    )

    # -- 9. Compliance Settings -------------------------------------------------
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
    deforestation_cutoff_date: str = field(
        default_factory=lambda: _env(
            "DEFORESTATION_CUTOFF", "2020-12-31"
        )
    )

    # -- 10. Upstream Agent URLs ------------------------------------------------
    supply_chain_mapper_url: str = field(
        default_factory=lambda: _env(
            "SUPPLY_CHAIN_MAPPER_URL",
            "http://eudr-supply-chain:8001/api/v1/eudr/supply-chain",
        )
    )
    geolocation_verification_url: str = field(
        default_factory=lambda: _env(
            "GEOLOCATION_VERIFICATION_URL",
            "http://eudr-geolocation:8002/api/v1/eudr/geolocation",
        )
    )
    country_risk_evaluator_url: str = field(
        default_factory=lambda: _env(
            "COUNTRY_RISK_EVALUATOR_URL",
            "http://eudr-country-risk:8016/api/v1/eudr/country-risk",
        )
    )
    supplier_risk_scorer_url: str = field(
        default_factory=lambda: _env(
            "SUPPLIER_RISK_SCORER_URL",
            "http://eudr-supplier-risk:8017/api/v1/eudr/supplier-risk",
        )
    )
    eu_information_system_url: str = field(
        default_factory=lambda: _env(
            "EU_IS_URL",
            "http://eudr-eu-is:8036/api/v1/eudr/eu-information-system",
        )
    )
    document_authentication_url: str = field(
        default_factory=lambda: _env(
            "DOCUMENT_AUTH_URL",
            "http://eudr-doc-auth:8012/api/v1/eudr/document-authentication",
        )
    )
    blockchain_integration_url: str = field(
        default_factory=lambda: _env(
            "BLOCKCHAIN_URL",
            "http://eudr-blockchain:8013/api/v1/eudr/blockchain",
        )
    )
    risk_assessment_engine_url: str = field(
        default_factory=lambda: _env(
            "RISK_ASSESSMENT_URL",
            "http://eudr-risk-assessment:8028/api/v1/eudr/risk-assessment",
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
    metrics_prefix: str = "gl_eudr_ddsc_"

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

        # Validate retention years per EUDR Article 31
        if self.retention_years < 5:
            logger.warning(
                "Retention years %d is below EUDR minimum of 5 years "
                "(Article 31).",
                self.retention_years,
            )

        # Validate geolocation precision
        if self.geolocation_precision_digits < 4:
            logger.warning(
                "Geolocation precision %d digits may be insufficient "
                "for Article 9 compliance (recommend >= 6).",
                self.geolocation_precision_digits,
            )

        # Validate document size limits
        if self.max_attachment_size_mb > self.max_package_size_mb:
            logger.warning(
                "Max attachment size %dMB exceeds max package size %dMB.",
                self.max_attachment_size_mb,
                self.max_package_size_mb,
            )

        # Validate default language is in supported languages
        if self.default_language not in self.supported_languages:
            logger.warning(
                "Default language '%s' not in supported languages list.",
                self.default_language,
            )

        # Validate signature validity
        if self.signature_validity_days < 365:
            logger.warning(
                "Signature validity %d days may be insufficient "
                "for annual DDS renewals.",
                self.signature_validity_days,
            )

        # Validate generation timeout
        if self.statement_generation_timeout_seconds < 30:
            logger.warning(
                "Statement generation timeout %ds may be too short "
                "for complex supply chains.",
                self.statement_generation_timeout_seconds,
            )

        logger.info(
            "DDSCreatorConfig initialized: "
            "template_version=%s, languages=%d, "
            "max_commodities=%d, max_plots=%d, "
            "precision=%d digits, retention=%d years, "
            "generation_timeout=%ds, batch_size=%d",
            self.dds_template_version,
            len(self.supported_languages),
            self.dds_max_commodities_per_statement,
            self.dds_max_plots_per_commodity,
            self.geolocation_precision_digits,
            self.retention_years,
            self.statement_generation_timeout_seconds,
            self.batch_size,
        )

    def get_rate_limit(self, tier: str) -> int:
        """Get rate limit for a given tier.

        Args:
            tier: Rate limit tier name.

        Returns:
            Rate limit value (requests per minute).
        """
        tier_map: Dict[str, int] = {
            "anonymous": self.rate_limit_anonymous,
            "basic": self.rate_limit_basic,
            "standard": self.rate_limit_standard,
            "premium": self.rate_limit_premium,
            "admin": self.rate_limit_admin,
        }
        return tier_map.get(tier.lower(), self.rate_limit_standard)

    def get_upstream_urls(self) -> Dict[str, str]:
        """Get all upstream agent URLs as a dictionary.

        Returns:
            Dictionary mapping agent names to their URLs.
        """
        return {
            "supply_chain_mapper": self.supply_chain_mapper_url,
            "geolocation_verification": self.geolocation_verification_url,
            "country_risk_evaluator": self.country_risk_evaluator_url,
            "supplier_risk_scorer": self.supplier_risk_scorer_url,
            "eu_information_system": self.eu_information_system_url,
            "document_authentication": self.document_authentication_url,
            "blockchain_integration": self.blockchain_integration_url,
            "risk_assessment_engine": self.risk_assessment_engine_url,
            "documentation_generator": self.documentation_generator_url,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[DDSCreatorConfig] = None
_config_lock = threading.Lock()


def get_config() -> DDSCreatorConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        DDSCreatorConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DDSCreatorConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
