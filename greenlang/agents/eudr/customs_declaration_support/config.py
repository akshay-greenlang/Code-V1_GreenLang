# -*- coding: utf-8 -*-
"""
Customs Declaration Support Configuration - AGENT-EUDR-039

Centralized configuration for the Customs Declaration Support agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- CN code database path and HS code database path
- Supported customs systems (NCTS, AIS, ICS2)
- Tariff calculation settings and official EU tariff rates
- Port of entry configurations
- Declaration form templates (SAD format)
- Compliance thresholds for EUDR Article 4(2)
- Currency conversion settings (EUR, USD, GBP, JPY)
- Performance tuning (declaration generation timeout, submission retry)
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_cds_

All settings overridable via environment variables with ``GL_EUDR_CDS_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-039 Customs Declaration Support (GL-EUDR-CDS-039)
Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 6, 12, 31; EU UCC 952/2013
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

_ENV_PREFIX = "GL_EUDR_CDS_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_CDS_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_CDS_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_CDS_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_CDS_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_CDS_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class CustomsDeclarationSupportConfig:
    """Centralized configuration for AGENT-EUDR-039.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_CDS_ environment variables.
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

    # -- 3. CN Code Database Settings -------------------------------------------
    cn_code_database_path: str = field(
        default_factory=lambda: _env(
            "CN_CODE_DB_PATH",
            "/data/eudr/customs/cn_codes.json",
        )
    )
    cn_code_cache_ttl: int = field(
        default_factory=lambda: _env_int("CN_CODE_CACHE_TTL", 86400)
    )
    cn_code_update_url: str = field(
        default_factory=lambda: _env(
            "CN_CODE_UPDATE_URL",
            "https://ec.europa.eu/taxation_customs/dds2/taric/measures.jsp",
        )
    )

    # -- 4. HS Code Database Settings -------------------------------------------
    hs_code_database_path: str = field(
        default_factory=lambda: _env(
            "HS_CODE_DB_PATH",
            "/data/eudr/customs/hs_codes.json",
        )
    )
    hs_code_wco_version: str = field(
        default_factory=lambda: _env("HS_CODE_WCO_VERSION", "2022")
    )

    # -- 5. Supported Customs Systems -------------------------------------------
    supported_customs_systems: List[str] = field(
        default_factory=lambda: [
            "NCTS",   # New Computerised Transit System
            "AIS",    # Automated Import System
            "ICS2",   # Import Control System 2
        ]
    )
    ncts_api_url: str = field(
        default_factory=lambda: _env(
            "NCTS_API_URL",
            "https://customs.ec.europa.eu/ncts/api/v1",
        )
    )
    ais_api_url: str = field(
        default_factory=lambda: _env(
            "AIS_API_URL",
            "https://customs.ec.europa.eu/ais/api/v1",
        )
    )
    ics2_api_url: str = field(
        default_factory=lambda: _env(
            "ICS2_API_URL",
            "https://customs.ec.europa.eu/ics2/api/v1",
        )
    )
    customs_api_timeout_seconds: int = field(
        default_factory=lambda: _env_int("CUSTOMS_API_TIMEOUT", 30)
    )
    customs_api_key: str = field(
        default_factory=lambda: _env("CUSTOMS_API_KEY", "")
    )

    # -- 5b. Customs System Specific Settings -----------------------------------
    ncts_timeout_seconds: int = field(
        default_factory=lambda: _env_int("NCTS_TIMEOUT", 60)
    )
    ais_timeout_seconds: int = field(
        default_factory=lambda: _env_int("AIS_TIMEOUT", 60)
    )
    ncts_retry_count: int = field(
        default_factory=lambda: _env_int("NCTS_RETRY_COUNT", 3)
    )
    ais_retry_count: int = field(
        default_factory=lambda: _env_int("AIS_RETRY_COUNT", 3)
    )
    customs_submission_timeout_seconds: int = field(
        default_factory=lambda: _env_int("CUSTOMS_SUBMISSION_TIMEOUT", 120)
    )
    ncts_message_format: str = field(
        default_factory=lambda: _env("NCTS_MESSAGE_FORMAT", "xml")
    )
    ais_message_format: str = field(
        default_factory=lambda: _env("AIS_MESSAGE_FORMAT", "xml")
    )
    customs_auth_method: str = field(
        default_factory=lambda: _env("CUSTOMS_AUTH_METHOD", "certificate")
    )
    sad_form_enabled: bool = field(
        default_factory=lambda: _env_bool("SAD_FORM_ENABLED", True)
    )
    mrn_generation_enabled: bool = field(
        default_factory=lambda: _env_bool("MRN_GENERATION_ENABLED", True)
    )

    # -- 6. Tariff Calculation Settings -----------------------------------------
    default_currency: str = field(
        default_factory=lambda: _env("DEFAULT_CURRENCY", "EUR")
    )
    default_tariff_currency: str = field(
        default_factory=lambda: _env("DEFAULT_TARIFF_CURRENCY", "EUR")
    )
    default_vat_rate: Decimal = field(
        default_factory=lambda: _env_decimal("DEFAULT_VAT_RATE", "21.0")
    )
    tariff_precision_digits: int = field(
        default_factory=lambda: _env_int("TARIFF_PRECISION_DIGITS", 4)
    )
    tariff_rounding_mode: str = field(
        default_factory=lambda: _env("TARIFF_ROUNDING_MODE", "ROUND_HALF_UP")
    )
    preferential_tariff_enabled: bool = field(
        default_factory=lambda: _env_bool("PREFERENTIAL_TARIFF_ENABLED", True)
    )
    anti_dumping_check_enabled: bool = field(
        default_factory=lambda: _env_bool("ANTI_DUMPING_CHECK_ENABLED", True)
    )
    tariff_database_version: str = field(
        default_factory=lambda: _env("TARIFF_DATABASE_VERSION", "2024.1")
    )
    currency_conversion_precision: int = field(
        default_factory=lambda: _env_int("CURRENCY_CONVERSION_PRECISION", 4)
    )
    tariff_calculation_timeout_seconds: int = field(
        default_factory=lambda: _env_int("TARIFF_CALCULATION_TIMEOUT", 30)
    )
    cn_code_format_digits: int = field(
        default_factory=lambda: _env_int("CN_CODE_FORMAT_DIGITS", 8)
    )
    hs_code_format_digits: int = field(
        default_factory=lambda: _env_int("HS_CODE_FORMAT_DIGITS", 6)
    )

    # -- 7. Port of Entry Configurations ----------------------------------------
    mrn_country_code: str = field(
        default_factory=lambda: _env("MRN_COUNTRY_CODE", "NL")
    )
    mrn_year_digits: int = 2
    customs_office_code: str = field(
        default_factory=lambda: _env("CUSTOMS_OFFICE_CODE", "NLRTM001")
    )
    default_port_of_entry: str = field(
        default_factory=lambda: _env("DEFAULT_PORT_OF_ENTRY", "NLRTM")
    )
    supported_port_types: List[str] = field(
        default_factory=lambda: ["sea", "air", "land", "rail", "inland_waterway"]
    )
    eu_member_state_codes: List[str] = field(
        default_factory=lambda: [
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        ]
    )

    # -- 8. Declaration Form Templates (SAD) ------------------------------------
    sad_template_version: str = field(
        default_factory=lambda: _env("SAD_TEMPLATE_VERSION", "2024.1")
    )
    sad_form_version: str = field(
        default_factory=lambda: _env("SAD_FORM_VERSION", "1.0")
    )
    sad_form_type_import: str = "IM"
    sad_form_type_export: str = "EX"
    sad_form_type_transit: str = "TR"
    max_items_per_declaration: int = field(
        default_factory=lambda: _env_int("MAX_ITEMS_PER_DECLARATION", 99)
    )
    max_line_items_per_declaration: int = field(
        default_factory=lambda: _env_int("MAX_LINE_ITEMS_PER_DECLARATION", 999)
    )
    declaration_template_path: str = field(
        default_factory=lambda: _env(
            "DECLARATION_TEMPLATE_PATH",
            "/data/eudr/customs/templates",
        )
    )

    # -- 9. Compliance Thresholds -----------------------------------------------
    dds_reference_required: bool = field(
        default_factory=lambda: _env_bool("DDS_REFERENCE_REQUIRED", True)
    )
    eudr_article_4_compliance_check: bool = field(
        default_factory=lambda: _env_bool("EUDR_ARTICLE_4_CHECK", True)
    )
    minimum_due_diligence_score: Decimal = field(
        default_factory=lambda: _env_decimal(
            "MIN_DUE_DILIGENCE_SCORE", "0.70"
        )
    )
    deforestation_cutoff_date: str = field(
        default_factory=lambda: _env(
            "DEFORESTATION_CUTOFF_DATE", "2020-12-31"
        )
    )
    compliance_check_timeout_seconds: int = field(
        default_factory=lambda: _env_int("COMPLIANCE_CHECK_TIMEOUT", 60)
    )
    max_compliance_retries: int = field(
        default_factory=lambda: _env_int("MAX_COMPLIANCE_RETRIES", 3)
    )

    # -- 10. Currency Conversion ------------------------------------------------
    supported_currencies: List[str] = field(
        default_factory=lambda: ["EUR", "USD", "GBP", "JPY", "CHF", "BRL", "IDR"]
    )
    ecb_exchange_rate_url: str = field(
        default_factory=lambda: _env(
            "ECB_EXCHANGE_RATE_URL",
            "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
        )
    )
    exchange_rate_cache_ttl: int = field(
        default_factory=lambda: _env_int("EXCHANGE_RATE_CACHE_TTL", 3600)
    )
    exchange_rate_fallback_enabled: bool = field(
        default_factory=lambda: _env_bool("EXCHANGE_RATE_FALLBACK", True)
    )

    # -- 11. Performance Tuning -------------------------------------------------
    declaration_generation_timeout: int = field(
        default_factory=lambda: _env_int("DECLARATION_GEN_TIMEOUT", 120)
    )
    submission_retry_max: int = field(
        default_factory=lambda: _env_int("SUBMISSION_RETRY_MAX", 3)
    )
    submission_retry_delay_seconds: int = field(
        default_factory=lambda: _env_int("SUBMISSION_RETRY_DELAY", 30)
    )
    submission_retry_backoff_factor: float = field(
        default_factory=lambda: _env_float("SUBMISSION_RETRY_BACKOFF", 2.0)
    )

    # -- 12. Upstream Agent URLs ------------------------------------------------
    supply_chain_url: str = field(
        default_factory=lambda: _env(
            "SUPPLY_CHAIN_URL",
            "http://eudr-supply-chain:8001/api/v1/eudr/supply-chain",
        )
    )
    due_diligence_url: str = field(
        default_factory=lambda: _env(
            "DUE_DILIGENCE_URL",
            "http://eudr-due-diligence:8026/api/v1/eudr/due-diligence",
        )
    )
    dds_creator_url: str = field(
        default_factory=lambda: _env(
            "DDS_CREATOR_URL",
            "http://eudr-dds-creator:8032/api/v1/eudr/dds-creator",
        )
    )
    reference_number_url: str = field(
        default_factory=lambda: _env(
            "REFERENCE_NUMBER_URL",
            "http://eudr-ref-number:8034/api/v1/eudr/reference-number",
        )
    )
    eu_info_system_url: str = field(
        default_factory=lambda: _env(
            "EU_INFO_SYSTEM_URL",
            "http://eudr-eu-is:8035/api/v1/eudr/eu-information-system",
        )
    )

    # -- 13. Retention and Compliance -------------------------------------------
    retention_years: int = field(
        default_factory=lambda: _env_int("RETENTION_YEARS", 5)
    )
    declaration_retention_years: int = field(
        default_factory=lambda: _env_int("DECLARATION_RETENTION_YEARS", 5)
    )
    report_formats: List[str] = field(
        default_factory=lambda: ["json", "pdf", "xml", "xlsx"]
    )
    default_report_format: str = field(
        default_factory=lambda: _env("DEFAULT_REPORT_FORMAT", "json")
    )

    # -- 13b. EUDR Compliance --------------------------------------------------
    eudr_dds_validation_enabled: bool = field(
        default_factory=lambda: _env_bool("EUDR_DDS_VALIDATION_ENABLED", True)
    )
    origin_cross_check_enabled: bool = field(
        default_factory=lambda: _env_bool("ORIGIN_CROSS_CHECK_ENABLED", True)
    )

    # -- 13c. Parallel Processing -----------------------------------------------
    parallel_engine_calls: int = field(
        default_factory=lambda: _env_int("PARALLEL_ENGINE_CALLS", 5)
    )

    # -- 14. Rate Limiting ------------------------------------------------------
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

    # -- 15. Circuit Breaker ----------------------------------------------------
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5)
    )
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _env_int("CB_RESET_TIMEOUT", 60)
    )
    circuit_breaker_half_open_max: int = field(
        default_factory=lambda: _env_int("CB_HALF_OPEN_MAX", 3)
    )

    # -- 16. Batch Processing ---------------------------------------------------
    batch_size: int = field(
        default_factory=lambda: _env_int("BATCH_SIZE", 50)
    )
    max_concurrent: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT", 10)
    )
    batch_timeout_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_TIMEOUT", 300)
    )

    # -- 17. Provenance ---------------------------------------------------------
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

    # -- 18. Metrics ------------------------------------------------------------
    metrics_enabled: bool = field(
        default_factory=lambda: _env_bool("METRICS_ENABLED", True)
    )
    metrics_prefix: str = "gl_eudr_cds_"

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Sync RETENTION_YEARS env override to declaration_retention_years
        _rv = _env("RETENTION_YEARS")
        if _rv is not None:
            try:
                self.declaration_retention_years = int(_rv)
            except (ValueError, TypeError):
                pass

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
        if self.declaration_retention_years < 5:
            logger.warning(
                "Declaration retention years %d is below EUDR minimum of 5 years "
                "(Article 31).",
                self.declaration_retention_years,
            )

        # Validate tariff precision
        if self.tariff_precision_digits < 2 or self.tariff_precision_digits > 8:
            logger.warning(
                "Tariff precision %d digits is outside recommended range [2, 8].",
                self.tariff_precision_digits,
            )

        # Validate minimum due diligence score range
        if not (Decimal("0") <= self.minimum_due_diligence_score <= Decimal("1")):
            logger.warning(
                "Minimum due diligence score %s is outside [0, 1].",
                self.minimum_due_diligence_score,
            )

        # Validate max items per declaration (SAD limit is 99)
        if self.max_items_per_declaration > 99:
            logger.warning(
                "Max items per declaration %d exceeds SAD form limit of 99.",
                self.max_items_per_declaration,
            )

        # Validate submission retry settings
        if self.submission_retry_max < 0:
            logger.warning(
                "Submission retry max %d is negative.",
                self.submission_retry_max,
            )

        # Validate NCTS timeout
        if self.ncts_timeout_seconds < 10:
            logger.warning(
                "NCTS timeout %d seconds is below recommended minimum of 10.",
                self.ncts_timeout_seconds,
            )

        # Validate AIS timeout
        if self.ais_timeout_seconds < 10:
            logger.warning(
                "AIS timeout %d seconds is below recommended minimum of 10.",
                self.ais_timeout_seconds,
            )

        logger.info(
            "CustomsDeclarationSupportConfig initialized: "
            "customs_systems=%s, default_port=%s, "
            "currencies=%d, dds_required=%s, "
            "max_items=%d, retention=%d years",
            ",".join(self.supported_customs_systems),
            self.default_port_of_entry,
            len(self.supported_currencies),
            self.dds_reference_required,
            self.max_items_per_declaration,
            self.retention_years,
        )

    def get_customs_system_url(self, system: str) -> str:
        """Get the API URL for a customs system.

        Args:
            system: Customs system identifier (NCTS, AIS, ICS2).

        Returns:
            API URL for the specified customs system.
        """
        url_map: Dict[str, str] = {
            "NCTS": self.ncts_api_url,
            "AIS": self.ais_api_url,
            "ICS2": self.ics2_api_url,
        }
        return url_map.get(system.upper(), self.ais_api_url)

    def get_incoterms_adjustments(self) -> Dict[str, List[str]]:
        """Get Incoterms adjustment rules for CIF/FOB calculation.

        Returns:
            Dictionary mapping Incoterms to required adjustment components.
        """
        return {
            "EXW": ["freight", "insurance", "loading"],
            "FCA": ["freight", "insurance"],
            "FAS": ["freight", "insurance"],
            "FOB": ["freight", "insurance"],
            "CFR": ["insurance"],
            "CIF": [],
            "CPT": ["insurance"],
            "CIP": [],
            "DAP": [],
            "DPU": [],
            "DDP": [],
        }

    # -- Property Aliases for Backward Compatibility ----------------------------
    @property
    def ncts_endpoint(self) -> str:
        """Alias for ncts_api_url."""
        return self.ncts_api_url

    @property
    def ais_endpoint(self) -> str:
        """Alias for ais_api_url."""
        return self.ais_api_url

    def get_rate_limit(self, tier: str) -> int:
        """Get rate limit for a given tier.

        Args:
            tier: Rate limit tier (anonymous/basic/standard/premium/admin).

        Returns:
            Requests per minute for the tier.
        """
        tier_map = {
            "anonymous": self.rate_limit_anonymous,
            "basic": self.rate_limit_basic,
            "standard": self.rate_limit_standard,
            "premium": self.rate_limit_premium,
            "admin": self.rate_limit_admin,
        }
        return tier_map.get(tier.lower(), self.rate_limit_standard)

    def get_upstream_urls(self) -> Dict[str, str]:
        """Get all upstream agent URLs.

        Returns:
            Dictionary mapping agent name to URL.
        """
        return {
            "supply_chain_mapper": self.supply_chain_url,
            "due_diligence": self.due_diligence_url,
            "dds_creator": self.dds_creator_url,
            "reference_number": self.reference_number_url,
            "eu_information_system": self.eu_info_system_url,
            "country_risk_evaluator": f"http://eudr-country-risk:8016/api/v1/eudr/country-risk",
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[CustomsDeclarationSupportConfig] = None
_config_lock = threading.Lock()


def get_config() -> CustomsDeclarationSupportConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        CustomsDeclarationSupportConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = CustomsDeclarationSupportConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None

# Alias for backward compatibility with tests
CustomsDeclarationConfig = CustomsDeclarationSupportConfig
