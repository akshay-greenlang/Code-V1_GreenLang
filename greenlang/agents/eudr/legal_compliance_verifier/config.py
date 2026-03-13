# -*- coding: utf-8 -*-
"""
Legal Compliance Verifier Configuration - AGENT-EUDR-023

Centralized configuration for the Legal Compliance Verifier Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Compliance thresholds: COMPLIANT (>=80), PARTIALLY_COMPLIANT (50-79),
  NON_COMPLIANT (<50) with configurable boundary values
- Red flag severity thresholds: CRITICAL (>=75), HIGH (>=50), MODERATE (>=25),
  LOW (<25) with configurable weights and multipliers
- Document verification settings: expiry warning days (30/60/90),
  verification pipeline timeouts, issuing authority validation
- Certification scheme validation: scheme-specific regex patterns,
  EUDR equivalence mapping, CoC model validation
- Batch processing: max_size=1000, concurrency=10, timeout=120s
- External API integration: FAO LEX, ECOLEX, FSC, RSPO, PEFC, ISCC
  with per-API rate limiting and timeout configuration
- Report generation: 8 types, 5 formats (PDF/JSON/HTML/XBRL/XML),
  5 languages (EN/FR/DE/ES/PT)
- Provenance: SHA-256 chain hashing for EUDR Article 31 compliance
- Rate limiting: per-tenant configurable request limits

All settings can be overridden via environment variables with the
``GL_EUDR_LCV_`` prefix (e.g. ``GL_EUDR_LCV_DATABASE_URL``,
``GL_EUDR_LCV_COMPLIANT_THRESHOLD``).

Example:
    >>> from greenlang.agents.eudr.legal_compliance_verifier.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.compliant_threshold, cfg.partial_threshold)
    80 50

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_LCV_"

# ---------------------------------------------------------------------------
# Valid constants
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_CHAIN_ALGORITHMS = frozenset({"sha256", "sha384", "sha512"})
_VALID_REPORT_FORMATS = frozenset({"pdf", "json", "html", "xbrl", "xml"})
_VALID_REPORT_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})
_VALID_REPORT_TYPES = frozenset({
    "full_assessment", "category_specific", "supplier_scorecard",
    "red_flag_summary", "document_status", "certification_validity",
    "country_framework", "dds_annex",
})

# ---------------------------------------------------------------------------
# Legislation categories per EUDR Article 2(40)
# ---------------------------------------------------------------------------

_LEGISLATION_CATEGORIES = frozenset({
    "land_use_rights", "environmental_protection", "forest_related_rules",
    "third_party_rights", "labour_rights", "tax_and_royalty",
    "trade_and_customs", "anti_corruption",
})

# ---------------------------------------------------------------------------
# EUDR commodities per Article 1
# ---------------------------------------------------------------------------

_EUDR_COMMODITIES = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
})

# ---------------------------------------------------------------------------
# Default compliance scoring thresholds
# ---------------------------------------------------------------------------

_DEFAULT_COMPLIANT_THRESHOLD = 80
_DEFAULT_PARTIAL_THRESHOLD = 50
_DEFAULT_RED_FLAG_CRITICAL = 75
_DEFAULT_RED_FLAG_HIGH = 50
_DEFAULT_RED_FLAG_MODERATE = 25

# ---------------------------------------------------------------------------
# Default document verification weights
# ---------------------------------------------------------------------------

_DEFAULT_DOC_WEIGHTS: Dict[str, float] = {
    "documents_present": 0.40,
    "document_validity": 0.30,
    "scope_alignment": 0.20,
    "authenticity": 0.10,
}

# ---------------------------------------------------------------------------
# Default document expiry warning days
# ---------------------------------------------------------------------------

_DEFAULT_EXPIRY_WARNING_DAYS = [90, 60, 30]


@dataclass
class LegalComplianceVerifierConfig:
    """Configuration for the Legal Compliance Verifier Agent (AGENT-EUDR-023).

    This dataclass encapsulates all configuration settings for legal framework
    management, document verification, certification validation, red flag
    detection, country compliance checking, third-party audit integration,
    and compliance reporting. All settings have sensible defaults aligned with
    EUDR Article 2(40) requirements and can be overridden via environment
    variables with the GL_EUDR_LCV_ prefix.

    Attributes:
        database_url: PostgreSQL connection URL.
        pool_size: Connection pool size (default 10).
        pool_timeout_s: Connection pool timeout seconds.
        pool_recycle_s: Connection pool recycle seconds.
        redis_url: Redis connection URL.
        redis_ttl_s: Redis cache TTL seconds (default 86400 = 24h).
        redis_key_prefix: Redis key prefix.
        log_level: Logging level.
        compliant_threshold: Score >= this = COMPLIANT (default 80).
        partial_threshold: Score >= this = PARTIALLY_COMPLIANT (default 50).
        red_flag_critical_threshold: Red flag score >= this = CRITICAL.
        red_flag_high_threshold: Red flag score >= this = HIGH.
        red_flag_moderate_threshold: Red flag score >= this = MODERATE.
        document_expiry_warning_days: Days before expiry to trigger warning.
        doc_verification_weights: Weights for document verification scoring.
        batch_max_size: Maximum batch assessment size.
        batch_concurrency: Batch concurrency workers.
        batch_timeout_s: Batch timeout seconds.
        retention_years: EUDR Article 31 data retention (default 5 years).
        enable_provenance: Enable SHA-256 provenance tracking.
        genesis_hash: Genesis hash anchor for provenance chain.
        chain_algorithm: Hash algorithm for provenance chain.
        enable_metrics: Enable Prometheus metrics.
        metrics_prefix: Prometheus metrics prefix.

    Example:
        >>> cfg = LegalComplianceVerifierConfig()
        >>> assert cfg.compliant_threshold == 80
        >>> assert cfg.partial_threshold == 50
    """

    # -----------------------------------------------------------------------
    # Database settings
    # -----------------------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    pool_size: int = 10
    pool_timeout_s: int = 30
    pool_recycle_s: int = 3600

    # -----------------------------------------------------------------------
    # Redis settings
    # -----------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_s: int = 86400
    redis_key_prefix: str = "gl:eudr:lcv:"

    # -----------------------------------------------------------------------
    # S3 settings
    # -----------------------------------------------------------------------
    s3_bucket: str = "gl-eudr-lcv-documents"
    s3_region: str = "eu-west-1"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # Compliance thresholds
    # -----------------------------------------------------------------------
    compliant_threshold: int = _DEFAULT_COMPLIANT_THRESHOLD
    partial_threshold: int = _DEFAULT_PARTIAL_THRESHOLD

    # -----------------------------------------------------------------------
    # Red flag thresholds
    # -----------------------------------------------------------------------
    red_flag_critical_threshold: int = _DEFAULT_RED_FLAG_CRITICAL
    red_flag_high_threshold: int = _DEFAULT_RED_FLAG_HIGH
    red_flag_moderate_threshold: int = _DEFAULT_RED_FLAG_MODERATE

    # -----------------------------------------------------------------------
    # Document verification settings
    # -----------------------------------------------------------------------
    document_expiry_warning_days: List[int] = field(
        default_factory=lambda: list(_DEFAULT_EXPIRY_WARNING_DAYS)
    )
    doc_verification_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_DOC_WEIGHTS)
    )

    # -----------------------------------------------------------------------
    # External API settings
    # -----------------------------------------------------------------------
    faolex_api_url: str = "https://faolex.fao.org/api/v1"
    ecolex_api_url: str = "https://www.ecolex.org/api/v1"
    fsc_api_url: str = "https://info.fsc.org/api/v1"
    rspo_api_url: str = "https://rspo.org/api/v1"
    pefc_api_url: str = "https://pefc.org/api/v1"
    iscc_api_url: str = "https://www.iscc-system.org/api/v1"
    external_api_timeout_s: int = 30

    # -----------------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------------
    batch_max_size: int = 1000
    batch_concurrency: int = 10
    batch_timeout_s: int = 120

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------
    default_report_format: str = "pdf"
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )

    # -----------------------------------------------------------------------
    # Data retention (EUDR Article 31 = 5 years)
    # -----------------------------------------------------------------------
    retention_years: int = 5

    # -----------------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-LCV-023-LEGAL-COMPLIANCE-VERIFIER-GENESIS"
    chain_algorithm: str = "sha256"

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_lcv_"

    # -----------------------------------------------------------------------
    # Rate limiting (requests per minute per tenant)
    # -----------------------------------------------------------------------
    rate_limit_default: int = 100
    rate_limit_batch: int = 10
    rate_limit_admin: int = 20

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_log_level()
        self._validate_chain_algorithm()
        self._validate_compliance_thresholds()
        self._validate_red_flag_thresholds()
        self._validate_doc_weights()
        self._validate_positive_integers()
        self._validate_languages()

        logger.info(
            f"LegalComplianceVerifierConfig initialized: "
            f"compliant={self.compliant_threshold}, "
            f"partial={self.partial_threshold}, "
            f"rf_critical={self.red_flag_critical_threshold}"
        )

    def _validate_log_level(self) -> None:
        """Validate log_level is a recognized Python logging level."""
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level: {self.log_level}. "
                f"Must be one of {_VALID_LOG_LEVELS}"
            )

    def _validate_chain_algorithm(self) -> None:
        """Validate chain_algorithm is a supported hash algorithm."""
        if self.chain_algorithm not in _VALID_CHAIN_ALGORITHMS:
            raise ValueError(
                f"Invalid chain_algorithm: {self.chain_algorithm}. "
                f"Must be one of {_VALID_CHAIN_ALGORITHMS}"
            )

    def _validate_compliance_thresholds(self) -> None:
        """Validate compliance thresholds are ordered and within 0-100."""
        if not (
            0 <= self.partial_threshold
            < self.compliant_threshold
            <= 100
        ):
            raise ValueError(
                "Compliance thresholds must satisfy: "
                "0 <= partial < compliant <= 100. "
                f"Got partial={self.partial_threshold}, "
                f"compliant={self.compliant_threshold}"
            )

    def _validate_red_flag_thresholds(self) -> None:
        """Validate red flag severity thresholds are ordered."""
        if not (
            0 <= self.red_flag_moderate_threshold
            < self.red_flag_high_threshold
            < self.red_flag_critical_threshold
            <= 100
        ):
            raise ValueError(
                "Red flag thresholds must satisfy: "
                "0 <= moderate < high < critical <= 100. "
                f"Got moderate={self.red_flag_moderate_threshold}, "
                f"high={self.red_flag_high_threshold}, "
                f"critical={self.red_flag_critical_threshold}"
            )

    def _validate_doc_weights(self) -> None:
        """Validate document verification weights sum to 1.0."""
        required_keys = {"documents_present", "document_validity",
                         "scope_alignment", "authenticity"}
        for key in required_keys:
            if key not in self.doc_verification_weights:
                raise ValueError(
                    f"Missing doc_verification_weights key: {key}"
                )
        total = sum(self.doc_verification_weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"doc_verification_weights must sum to 1.0, got {total:.4f}"
            )

    def _validate_positive_integers(self) -> None:
        """Validate fields that must be positive integers."""
        checks = [
            ("pool_size", self.pool_size),
            ("batch_max_size", self.batch_max_size),
            ("batch_concurrency", self.batch_concurrency),
            ("retention_years", self.retention_years),
        ]
        for name, val in checks:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

    def _validate_languages(self) -> None:
        """Validate language settings against allowed set."""
        if self.default_language not in _VALID_REPORT_LANGUAGES:
            raise ValueError(
                f"Invalid default_language: {self.default_language}. "
                f"Must be one of {_VALID_REPORT_LANGUAGES}"
            )
        for lang in self.supported_languages:
            if lang not in _VALID_REPORT_LANGUAGES:
                raise ValueError(
                    f"Invalid supported language: {lang}. "
                    f"Must be one of {_VALID_REPORT_LANGUAGES}"
                )

    @classmethod
    def from_env(cls) -> "LegalComplianceVerifierConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_LCV_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            LegalComplianceVerifierConfig instance with env overrides applied.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_LCV_COMPLIANT_THRESHOLD"] = "85"
            >>> cfg = LegalComplianceVerifierConfig.from_env()
            >>> assert cfg.compliant_threshold == 85
        """
        kwargs: Dict[str, Any] = {}

        # Database
        if val := os.getenv(f"{_ENV_PREFIX}DATABASE_URL"):
            kwargs["database_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}POOL_SIZE"):
            kwargs["pool_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POOL_TIMEOUT_S"):
            kwargs["pool_timeout_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POOL_RECYCLE_S"):
            kwargs["pool_recycle_s"] = int(val)

        # Redis
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_URL"):
            kwargs["redis_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_TTL_S"):
            kwargs["redis_ttl_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_KEY_PREFIX"):
            kwargs["redis_key_prefix"] = val

        # S3
        if val := os.getenv(f"{_ENV_PREFIX}S3_BUCKET"):
            kwargs["s3_bucket"] = val
        if val := os.getenv(f"{_ENV_PREFIX}S3_REGION"):
            kwargs["s3_region"] = val

        # Logging
        if val := os.getenv(f"{_ENV_PREFIX}LOG_LEVEL"):
            kwargs["log_level"] = val.upper()

        # Compliance thresholds
        if val := os.getenv(f"{_ENV_PREFIX}COMPLIANT_THRESHOLD"):
            kwargs["compliant_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}PARTIAL_THRESHOLD"):
            kwargs["partial_threshold"] = int(val)

        # Red flag thresholds
        if val := os.getenv(f"{_ENV_PREFIX}RED_FLAG_CRITICAL_THRESHOLD"):
            kwargs["red_flag_critical_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RED_FLAG_HIGH_THRESHOLD"):
            kwargs["red_flag_high_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RED_FLAG_MODERATE_THRESHOLD"):
            kwargs["red_flag_moderate_threshold"] = int(val)

        # Document verification
        if val := os.getenv(f"{_ENV_PREFIX}DOCUMENT_EXPIRY_WARNING_DAYS"):
            kwargs["document_expiry_warning_days"] = [
                int(x.strip()) for x in val.split(",")
            ]

        # External APIs
        if val := os.getenv(f"{_ENV_PREFIX}FAOLEX_API_URL"):
            kwargs["faolex_api_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}ECOLEX_API_URL"):
            kwargs["ecolex_api_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}FSC_API_URL"):
            kwargs["fsc_api_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}RSPO_API_URL"):
            kwargs["rspo_api_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}EXTERNAL_API_TIMEOUT_S"):
            kwargs["external_api_timeout_s"] = int(val)

        # Batch processing
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_MAX_SIZE"):
            kwargs["batch_max_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_CONCURRENCY"):
            kwargs["batch_concurrency"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_TIMEOUT_S"):
            kwargs["batch_timeout_s"] = int(val)

        # Reports
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_REPORT_FORMAT"):
            kwargs["default_report_format"] = val
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_LANGUAGE"):
            kwargs["default_language"] = val

        # Data retention
        if val := os.getenv(f"{_ENV_PREFIX}RETENTION_YEARS"):
            kwargs["retention_years"] = int(val)

        # Provenance
        if val := os.getenv(f"{_ENV_PREFIX}ENABLE_PROVENANCE"):
            kwargs["enable_provenance"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}GENESIS_HASH"):
            kwargs["genesis_hash"] = val
        if val := os.getenv(f"{_ENV_PREFIX}CHAIN_ALGORITHM"):
            kwargs["chain_algorithm"] = val

        # Metrics
        if val := os.getenv(f"{_ENV_PREFIX}ENABLE_METRICS"):
            kwargs["enable_metrics"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}METRICS_PREFIX"):
            kwargs["metrics_prefix"] = val

        # Rate limiting
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_DEFAULT"):
            kwargs["rate_limit_default"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_BATCH"):
            kwargs["rate_limit_batch"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_ADMIN"):
            kwargs["rate_limit_admin"] = int(val)

        return cls(**kwargs)

    def to_dict(self, redact_secrets: bool = True) -> Dict[str, Any]:
        """Export configuration as a dictionary.

        Args:
            redact_secrets: If True, redact sensitive fields like database_url.

        Returns:
            Dictionary representation of configuration.

        Example:
            >>> cfg = LegalComplianceVerifierConfig()
            >>> d = cfg.to_dict(redact_secrets=True)
            >>> assert "REDACTED" in d["database_url"]
        """
        data: Dict[str, Any] = {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "pool_timeout_s": self.pool_timeout_s,
            "pool_recycle_s": self.pool_recycle_s,
            "redis_url": self.redis_url,
            "redis_ttl_s": self.redis_ttl_s,
            "redis_key_prefix": self.redis_key_prefix,
            "s3_bucket": self.s3_bucket,
            "s3_region": self.s3_region,
            "log_level": self.log_level,
            "compliant_threshold": self.compliant_threshold,
            "partial_threshold": self.partial_threshold,
            "red_flag_critical_threshold": self.red_flag_critical_threshold,
            "red_flag_high_threshold": self.red_flag_high_threshold,
            "red_flag_moderate_threshold": self.red_flag_moderate_threshold,
            "document_expiry_warning_days": self.document_expiry_warning_days,
            "doc_verification_weights": self.doc_verification_weights,
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            "retention_years": self.retention_years,
            "enable_provenance": self.enable_provenance,
            "chain_algorithm": self.chain_algorithm,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "rate_limit_default": self.rate_limit_default,
            "rate_limit_batch": self.rate_limit_batch,
            "rate_limit_admin": self.rate_limit_admin,
        }

        if redact_secrets:
            if "://" in data["database_url"]:
                data["database_url"] = "REDACTED"
            if "://" in data["redis_url"]:
                data["redis_url"] = "REDACTED"

        return data


# ---------------------------------------------------------------------------
# Thread-safe singleton pattern (double-checked locking)
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[LegalComplianceVerifierConfig] = None


def get_config() -> LegalComplianceVerifierConfig:
    """Get the global LegalComplianceVerifierConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance.

    Returns:
        LegalComplianceVerifierConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> assert cfg.compliant_threshold == 80
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = LegalComplianceVerifierConfig.from_env()
    return _global_config


def set_config(config: LegalComplianceVerifierConfig) -> None:
    """Set the global LegalComplianceVerifierConfig singleton instance.

    Used for testing and programmatic configuration override.

    Args:
        config: LegalComplianceVerifierConfig instance to set as global.

    Example:
        >>> set_config(LegalComplianceVerifierConfig(compliant_threshold=85))
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global LegalComplianceVerifierConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_config()
    """
    global _global_config
    with _config_lock:
        _global_config = None
