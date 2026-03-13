# -*- coding: utf-8 -*-
"""
Document Authentication Configuration - AGENT-EUDR-012

Centralized configuration for the Document Authentication Agent covering:
- Document classification: confidence thresholds, batch sizing, supported
  document types for EUDR supply chain traceability
- Signature verification: timeout, timestamp requirement, self-signed
  certificate handling per eIDAS Regulation (EU) No 910/2014
- Hash integrity: SHA-256 primary / SHA-512 secondary algorithms, registry
  TTL (five-year retention per EUDR Article 14)
- Certificate chain validation: OCSP stapling, CRL refresh scheduling,
  minimum key sizes (RSA 2048, ECDSA 256), optional CT log checking
- Metadata extraction: creation date tolerance, author matching enforcement,
  empty metadata flagging for anomaly detection
- Fraud pattern detection: quantity tolerance, date tolerance, velocity
  threshold, round-number bias detection, 15 fraud rule types
- Cross-reference verification: FSC/RSPO/ISCC API rate limits, caching,
  timeout configuration for registry lookups
- Report generation: JSON/PDF/CSV/EUDR_XML formats, evidence packages,
  five-year report retention per EUDR Article 14
- Batch processing: size limits, concurrency, timeout
- Data retention: EUDR Article 14 five-year retention
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_DAV_`` prefix (e.g. ``GL_EUDR_DAV_DATABASE_URL``,
``GL_EUDR_DAV_MIN_CONFIDENCE_HIGH``).

Environment Variable Reference (GL_EUDR_DAV_ prefix):
    GL_EUDR_DAV_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_DAV_REDIS_URL                     - Redis connection URL
    GL_EUDR_DAV_LOG_LEVEL                     - Logging level
    GL_EUDR_DAV_POOL_SIZE                     - Database pool size
    GL_EUDR_DAV_MIN_CONFIDENCE_HIGH           - Min confidence for HIGH
    GL_EUDR_DAV_MIN_CONFIDENCE_MEDIUM         - Min confidence for MEDIUM
    GL_EUDR_DAV_MAX_BATCH_SIZE                - Max classification batch size
    GL_EUDR_DAV_SIGNATURE_TIMEOUT_S           - Signature verification timeout
    GL_EUDR_DAV_REQUIRE_TIMESTAMP             - Require signed timestamps
    GL_EUDR_DAV_ACCEPT_SELF_SIGNED            - Accept self-signed certs
    GL_EUDR_DAV_HASH_ALGORITHM                - Primary hash algorithm
    GL_EUDR_DAV_SECONDARY_HASH                - Secondary hash algorithm
    GL_EUDR_DAV_REGISTRY_TTL_DAYS             - Hash registry TTL in days
    GL_EUDR_DAV_OCSP_ENABLED                  - Enable OCSP checking
    GL_EUDR_DAV_CRL_REFRESH_HOURS             - CRL refresh interval hours
    GL_EUDR_DAV_MIN_KEY_SIZE_RSA              - Minimum RSA key size bits
    GL_EUDR_DAV_MIN_KEY_SIZE_ECDSA            - Minimum ECDSA key size bits
    GL_EUDR_DAV_CERT_TRANSPARENCY_ENABLED     - Enable CT log checking
    GL_EUDR_DAV_CREATION_DATE_TOLERANCE_DAYS  - Creation date tolerance
    GL_EUDR_DAV_REQUIRE_AUTHOR_MATCH          - Require author match
    GL_EUDR_DAV_FLAG_EMPTY_METADATA           - Flag empty metadata
    GL_EUDR_DAV_QUANTITY_TOLERANCE_PERCENT     - Quantity tolerance %
    GL_EUDR_DAV_DATE_TOLERANCE_DAYS           - Date tolerance days
    GL_EUDR_DAV_VELOCITY_THRESHOLD_PER_DAY    - Velocity threshold per day
    GL_EUDR_DAV_ROUND_NUMBER_THRESHOLD_PERCENT - Round number threshold %
    GL_EUDR_DAV_FRAUD_RULES_ENABLED           - Enable fraud detection rules
    GL_EUDR_DAV_CACHE_TTL_HOURS               - Cross-ref cache TTL hours
    GL_EUDR_DAV_FSC_API_RATE_LIMIT            - FSC API rate limit/min
    GL_EUDR_DAV_RSPO_API_RATE_LIMIT           - RSPO API rate limit/min
    GL_EUDR_DAV_ISCC_API_RATE_LIMIT           - ISCC API rate limit/min
    GL_EUDR_DAV_CROSSREF_TIMEOUT_S            - Cross-ref timeout seconds
    GL_EUDR_DAV_DEFAULT_FORMAT                - Default report format
    GL_EUDR_DAV_RETENTION_DAYS                - Report retention days
    GL_EUDR_DAV_EVIDENCE_PACKAGE_ENABLED      - Enable evidence packages
    GL_EUDR_DAV_BATCH_MAX_SIZE                - Batch processing max size
    GL_EUDR_DAV_BATCH_CONCURRENCY             - Batch concurrency workers
    GL_EUDR_DAV_BATCH_TIMEOUT_S               - Batch timeout seconds
    GL_EUDR_DAV_RETENTION_YEARS               - Data retention years
    GL_EUDR_DAV_ENABLE_PROVENANCE             - Enable provenance tracking
    GL_EUDR_DAV_GENESIS_HASH                  - Genesis hash anchor
    GL_EUDR_DAV_ENABLE_METRICS                - Enable Prometheus metrics
    GL_EUDR_DAV_RATE_LIMIT                    - Max requests per minute

Example:
    >>> from greenlang.agents.eudr.document_authentication.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.hash_algorithm, cfg.min_confidence_high)
    sha256 0.95

    >>> # Override for testing
    >>> from greenlang.agents.eudr.document_authentication.config import (
    ...     set_config, reset_config, DocumentAuthenticationConfig,
    ... )
    >>> set_config(DocumentAuthenticationConfig(hash_algorithm="sha512"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication (GL-EUDR-DAV-012)
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

_ENV_PREFIX = "GL_EUDR_DAV_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid hash algorithms
# ---------------------------------------------------------------------------

_VALID_HASH_ALGORITHMS = frozenset(
    {"sha256", "sha512", "hmac_sha256"}
)

# ---------------------------------------------------------------------------
# Valid report formats
# ---------------------------------------------------------------------------

_VALID_REPORT_FORMATS = frozenset(
    {"json", "csv", "pdf", "eudr_xml"}
)

# ---------------------------------------------------------------------------
# Valid signature standards
# ---------------------------------------------------------------------------

_VALID_SIGNATURE_STANDARDS = frozenset(
    {"cades", "pades", "xades", "jades", "qes", "pgp", "pkcs7"}
)

# ---------------------------------------------------------------------------
# Default document types for EUDR supply chain
# ---------------------------------------------------------------------------

_DEFAULT_DOCUMENT_TYPES: List[str] = [
    "coo", "pc", "bol", "cde", "cdi",
    "rspo_cert", "fsc_cert", "iscc_cert", "ft_cert", "utz_cert",
    "ltr", "ltd", "fmp", "fc", "wqc",
    "dds_draft", "ssd", "ic", "tc", "wr",
]

# ---------------------------------------------------------------------------
# Default fraud pattern types
# ---------------------------------------------------------------------------

_DEFAULT_FRAUD_PATTERN_TYPES: List[str] = [
    "duplicate_reuse",
    "quantity_tampering",
    "date_manipulation",
    "expired_cert",
    "serial_anomaly",
    "issuer_mismatch",
    "template_forgery",
    "cross_doc_inconsistency",
    "geo_impossibility",
    "velocity_anomaly",
    "modification_anomaly",
    "round_number_bias",
    "copy_paste",
    "missing_required",
    "scope_mismatch",
]

# ---------------------------------------------------------------------------
# Default trusted certificate authorities for EUDR document signing
# ---------------------------------------------------------------------------

_DEFAULT_TRUSTED_CAS: List[str] = [
    "DigiCert Global Root G2",
    "GlobalSign Root CA - R3",
    "Entrust Root Certification Authority - G4",
    "QuoVadis Root CA 2 G3",
    "SwissSign Gold CA - G2",
    "TUEV Sued eID Root CA 1",
    "D-TRUST Root Class 3 CA 2 2009",
    "Bundesdruckerei GmbH",
]

# ---------------------------------------------------------------------------
# Default registry rate limits (requests per minute)
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY_RATE_LIMITS: Dict[str, int] = {
    "fsc": 100,
    "rspo": 60,
    "iscc": 30,
    "fairtrade": 30,
    "utz_ra": 30,
    "ippc": 20,
    "national_customs": 10,
}

# ---------------------------------------------------------------------------
# Default metadata fields expected in EUDR documents
# ---------------------------------------------------------------------------

_DEFAULT_REQUIRED_METADATA_FIELDS: List[str] = [
    "title",
    "author",
    "creation_date",
    "producer",
]

# ---------------------------------------------------------------------------
# Default fraud severity weights (used for composite risk scoring)
# ---------------------------------------------------------------------------

_DEFAULT_FRAUD_SEVERITY_WEIGHTS: Dict[str, float] = {
    "low": 1.0,
    "medium": 3.0,
    "high": 7.0,
    "critical": 10.0,
}

# ---------------------------------------------------------------------------
# Default EUDR commodities
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]


# ---------------------------------------------------------------------------
# DocumentAuthenticationConfig
# ---------------------------------------------------------------------------


@dataclass
class DocumentAuthenticationConfig:
    """Complete configuration for the EUDR Document Authentication Agent.

    Attributes are grouped by concern: connections, logging, classification,
    signature verification, hash integrity, certificate chain validation,
    metadata extraction, fraud detection, cross-reference verification,
    report generation, batch processing, data retention, provenance
    tracking, metrics, and performance tuning.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_DAV_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            document records, hash registry, fraud alerts, and reports.
        redis_url: Redis connection URL for cross-reference caching,
            rate limiting counters, and duplicate detection bloom filters.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        pool_size: PostgreSQL connection pool size.
        min_confidence_high: Minimum classification confidence score
            (0.0-1.0) to assign HIGH confidence. Documents below this
            threshold but above min_confidence_medium get MEDIUM.
        min_confidence_medium: Minimum classification confidence score
            (0.0-1.0) to assign MEDIUM confidence. Documents below this
            threshold get LOW confidence.
        max_batch_size: Maximum number of documents in a single
            classification batch request.
        signature_timeout_s: Timeout in seconds for signature
            verification operations including OCSP/CRL checks.
        require_timestamp: Whether signed timestamps are required
            for signature validation per eIDAS.
        accept_self_signed: Whether to accept self-signed certificates
            during signature verification. Should be False in production.
        hash_algorithm: Primary hash algorithm for document integrity
            validation. Supports sha256, sha512, hmac_sha256.
        secondary_hash: Secondary hash algorithm for dual-hash
            verification. Provides defense-in-depth.
        registry_ttl_days: Hash registry TTL in days. Defaults to
            1825 (5 years) per EUDR Article 14 retention.
        ocsp_enabled: Whether to enable OCSP (Online Certificate
            Status Protocol) checking during chain validation.
        crl_refresh_hours: Hours between CRL (Certificate Revocation
            List) refresh cycles.
        min_key_size_rsa: Minimum RSA public key size in bits.
            NIST recommends 2048+ for document signing.
        min_key_size_ecdsa: Minimum ECDSA curve size in bits.
            NIST P-256 (256 bits) is the minimum recommended.
        cert_transparency_enabled: Whether to verify certificate
            transparency (CT) logs for issued certificates.
        creation_date_tolerance_days: Maximum acceptable difference
            in days between document creation date and upload date.
        require_author_match: Whether the document author metadata
            must match the submitting operator identity.
        flag_empty_metadata: Whether to flag documents with missing
            or empty metadata fields as suspicious.
        quantity_tolerance_percent: Acceptable percentage tolerance
            for quantity values when cross-referencing documents.
        date_tolerance_days: Acceptable tolerance in days for date
            fields when cross-referencing documents.
        velocity_threshold_per_day: Maximum number of documents from
            a single issuer per day before triggering velocity alert.
        round_number_threshold_percent: Percentage of values that are
            round numbers above which round-number bias is flagged.
        fraud_rules_enabled: Master switch to enable/disable fraud
            pattern detection rules.
        cache_ttl_hours: TTL in hours for cross-reference verification
            cache entries.
        fsc_api_rate_limit: FSC registry API rate limit (requests/min).
        rspo_api_rate_limit: RSPO registry API rate limit (requests/min).
        iscc_api_rate_limit: ISCC registry API rate limit (requests/min).
        crossref_timeout_s: Timeout in seconds for individual
            cross-reference API calls.
        default_format: Default output format for generated reports.
        retention_days: Number of days to retain generated reports.
        evidence_package_enabled: Whether to generate evidence packages
            (bundled PDF + metadata + hashes) for compliance submission.
        batch_max_size: Maximum number of documents in a single batch
            verification job.
        batch_concurrency: Maximum concurrent batch verification workers.
        batch_timeout_s: Timeout in seconds for a single batch job.
        retention_years: Data retention in years per EUDR Article 14.
        eudr_commodities: List of EUDR-regulated commodity types.
        document_types: Supported EUDR document type identifiers.
        fraud_pattern_types: Active fraud pattern detection types.
        trusted_cas: List of trusted certificate authority names.
        registry_rate_limits: Per-registry API rate limits.
        required_metadata_fields: Metadata fields required for
            document completeness validation.
        fraud_severity_weights: Severity level weights for composite
            fraud risk scoring.
        enable_provenance: Enable SHA-256 provenance chain tracking
            for all document authentication operations.
        genesis_hash: Genesis anchor string for the provenance chain,
            unique to the Document Authentication agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_dav_`` prefix.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10

    # -- Classification settings ---------------------------------------------
    min_confidence_high: float = 0.95
    min_confidence_medium: float = 0.70
    max_batch_size: int = 500

    # -- Signature verification ----------------------------------------------
    signature_timeout_s: int = 30
    require_timestamp: bool = True
    accept_self_signed: bool = False

    # -- Hash integrity ------------------------------------------------------
    hash_algorithm: str = "sha256"
    secondary_hash: str = "sha512"
    registry_ttl_days: int = 1825

    # -- Certificate chain validation ----------------------------------------
    ocsp_enabled: bool = True
    crl_refresh_hours: int = 24
    min_key_size_rsa: int = 2048
    min_key_size_ecdsa: int = 256
    cert_transparency_enabled: bool = False

    # -- Metadata extraction -------------------------------------------------
    creation_date_tolerance_days: int = 30
    require_author_match: bool = True
    flag_empty_metadata: bool = True

    # -- Fraud detection -----------------------------------------------------
    quantity_tolerance_percent: float = 5.0
    date_tolerance_days: int = 30
    velocity_threshold_per_day: int = 10
    round_number_threshold_percent: float = 80.0
    fraud_rules_enabled: bool = True

    # -- Cross-reference verification ----------------------------------------
    cache_ttl_hours: int = 24
    fsc_api_rate_limit: int = 100
    rspo_api_rate_limit: int = 60
    iscc_api_rate_limit: int = 30
    crossref_timeout_s: int = 30

    # -- Report generation ---------------------------------------------------
    default_format: str = "json"
    retention_days: int = 1825
    evidence_package_enabled: bool = True

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 500
    batch_concurrency: int = 4
    batch_timeout_s: int = 300

    # -- Data retention (EUDR Article 14) ------------------------------------
    retention_years: int = 5

    # -- EUDR commodities ----------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Document types ------------------------------------------------------
    document_types: List[str] = field(
        default_factory=lambda: list(_DEFAULT_DOCUMENT_TYPES)
    )

    # -- Fraud pattern types -------------------------------------------------
    fraud_pattern_types: List[str] = field(
        default_factory=lambda: list(_DEFAULT_FRAUD_PATTERN_TYPES)
    )

    # -- Trusted certificate authorities -------------------------------------
    trusted_cas: List[str] = field(
        default_factory=lambda: list(_DEFAULT_TRUSTED_CAS)
    )

    # -- Registry rate limits ------------------------------------------------
    registry_rate_limits: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_REGISTRY_RATE_LIMITS)
    )

    # -- Required metadata fields --------------------------------------------
    required_metadata_fields: List[str] = field(
        default_factory=lambda: list(_DEFAULT_REQUIRED_METADATA_FIELDS)
    )

    # -- Fraud severity weights ----------------------------------------------
    fraud_severity_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_FRAUD_SEVERITY_WEIGHTS)
    )

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-DAV-012-DOCUMENT-AUTHENTICATION-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Rate limiting -------------------------------------------------------
    rate_limit: int = 300

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, threshold ordering validation, and
        normalization. Collects all errors before raising a single
        ValueError with all violations listed.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint.
        """
        errors: list[str] = []

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- Classification settings -----------------------------------------
        if not (0.0 < self.min_confidence_high <= 1.0):
            errors.append(
                f"min_confidence_high must be in (0.0, 1.0], "
                f"got {self.min_confidence_high}"
            )

        if not (0.0 < self.min_confidence_medium <= 1.0):
            errors.append(
                f"min_confidence_medium must be in (0.0, 1.0], "
                f"got {self.min_confidence_medium}"
            )

        if self.min_confidence_medium >= self.min_confidence_high:
            errors.append(
                f"min_confidence_medium ({self.min_confidence_medium}) "
                f"must be < min_confidence_high ({self.min_confidence_high})"
            )

        if self.max_batch_size < 1:
            errors.append(
                f"max_batch_size must be >= 1, got {self.max_batch_size}"
            )
        if self.max_batch_size > 10000:
            errors.append(
                f"max_batch_size must be <= 10000, got {self.max_batch_size}"
            )

        # -- Signature verification ------------------------------------------
        if self.signature_timeout_s < 1:
            errors.append(
                f"signature_timeout_s must be >= 1, "
                f"got {self.signature_timeout_s}"
            )
        if self.signature_timeout_s > 300:
            errors.append(
                f"signature_timeout_s must be <= 300, "
                f"got {self.signature_timeout_s}"
            )

        # -- Hash integrity --------------------------------------------------
        normalised_hash = self.hash_algorithm.lower().strip()
        if normalised_hash not in _VALID_HASH_ALGORITHMS:
            errors.append(
                f"hash_algorithm must be one of "
                f"{sorted(_VALID_HASH_ALGORITHMS)}, "
                f"got '{self.hash_algorithm}'"
            )
        else:
            self.hash_algorithm = normalised_hash

        normalised_secondary = self.secondary_hash.lower().strip()
        if normalised_secondary not in _VALID_HASH_ALGORITHMS:
            errors.append(
                f"secondary_hash must be one of "
                f"{sorted(_VALID_HASH_ALGORITHMS)}, "
                f"got '{self.secondary_hash}'"
            )
        else:
            self.secondary_hash = normalised_secondary

        if self.registry_ttl_days < 1:
            errors.append(
                f"registry_ttl_days must be >= 1, "
                f"got {self.registry_ttl_days}"
            )

        # -- Certificate chain validation ------------------------------------
        if self.crl_refresh_hours < 1:
            errors.append(
                f"crl_refresh_hours must be >= 1, "
                f"got {self.crl_refresh_hours}"
            )
        if self.crl_refresh_hours > 168:
            errors.append(
                f"crl_refresh_hours must be <= 168 (7 days), "
                f"got {self.crl_refresh_hours}"
            )

        if self.min_key_size_rsa < 1024:
            errors.append(
                f"min_key_size_rsa must be >= 1024, "
                f"got {self.min_key_size_rsa}"
            )
        if self.min_key_size_rsa > 8192:
            errors.append(
                f"min_key_size_rsa must be <= 8192, "
                f"got {self.min_key_size_rsa}"
            )

        if self.min_key_size_ecdsa < 224:
            errors.append(
                f"min_key_size_ecdsa must be >= 224, "
                f"got {self.min_key_size_ecdsa}"
            )
        if self.min_key_size_ecdsa > 521:
            errors.append(
                f"min_key_size_ecdsa must be <= 521, "
                f"got {self.min_key_size_ecdsa}"
            )

        # -- Metadata extraction ---------------------------------------------
        if self.creation_date_tolerance_days < 0:
            errors.append(
                f"creation_date_tolerance_days must be >= 0, "
                f"got {self.creation_date_tolerance_days}"
            )
        if self.creation_date_tolerance_days > 365:
            errors.append(
                f"creation_date_tolerance_days must be <= 365, "
                f"got {self.creation_date_tolerance_days}"
            )

        # -- Fraud detection -------------------------------------------------
        if self.quantity_tolerance_percent < 0.0:
            errors.append(
                f"quantity_tolerance_percent must be >= 0.0, "
                f"got {self.quantity_tolerance_percent}"
            )
        if self.quantity_tolerance_percent > 100.0:
            errors.append(
                f"quantity_tolerance_percent must be <= 100.0, "
                f"got {self.quantity_tolerance_percent}"
            )

        if self.date_tolerance_days < 0:
            errors.append(
                f"date_tolerance_days must be >= 0, "
                f"got {self.date_tolerance_days}"
            )
        if self.date_tolerance_days > 365:
            errors.append(
                f"date_tolerance_days must be <= 365, "
                f"got {self.date_tolerance_days}"
            )

        if self.velocity_threshold_per_day < 1:
            errors.append(
                f"velocity_threshold_per_day must be >= 1, "
                f"got {self.velocity_threshold_per_day}"
            )
        if self.velocity_threshold_per_day > 10000:
            errors.append(
                f"velocity_threshold_per_day must be <= 10000, "
                f"got {self.velocity_threshold_per_day}"
            )

        if not (0.0 < self.round_number_threshold_percent <= 100.0):
            errors.append(
                f"round_number_threshold_percent must be in (0.0, 100.0], "
                f"got {self.round_number_threshold_percent}"
            )

        # -- Cross-reference verification ------------------------------------
        if self.cache_ttl_hours < 1:
            errors.append(
                f"cache_ttl_hours must be >= 1, got {self.cache_ttl_hours}"
            )
        if self.cache_ttl_hours > 720:
            errors.append(
                f"cache_ttl_hours must be <= 720 (30 days), "
                f"got {self.cache_ttl_hours}"
            )

        if self.fsc_api_rate_limit < 1:
            errors.append(
                f"fsc_api_rate_limit must be >= 1, "
                f"got {self.fsc_api_rate_limit}"
            )

        if self.rspo_api_rate_limit < 1:
            errors.append(
                f"rspo_api_rate_limit must be >= 1, "
                f"got {self.rspo_api_rate_limit}"
            )

        if self.iscc_api_rate_limit < 1:
            errors.append(
                f"iscc_api_rate_limit must be >= 1, "
                f"got {self.iscc_api_rate_limit}"
            )

        if self.crossref_timeout_s < 1:
            errors.append(
                f"crossref_timeout_s must be >= 1, "
                f"got {self.crossref_timeout_s}"
            )
        if self.crossref_timeout_s > 300:
            errors.append(
                f"crossref_timeout_s must be <= 300, "
                f"got {self.crossref_timeout_s}"
            )

        # -- Report generation -----------------------------------------------
        normalised_format = self.default_format.lower().strip()
        if normalised_format not in _VALID_REPORT_FORMATS:
            errors.append(
                f"default_format must be one of "
                f"{sorted(_VALID_REPORT_FORMATS)}, "
                f"got '{self.default_format}'"
            )
        else:
            self.default_format = normalised_format

        if self.retention_days < 1:
            errors.append(
                f"retention_days must be >= 1, "
                f"got {self.retention_days}"
            )

        # -- Batch processing ------------------------------------------------
        if self.batch_max_size < 1:
            errors.append(
                f"batch_max_size must be >= 1, got {self.batch_max_size}"
            )

        if not (1 <= self.batch_concurrency <= 256):
            errors.append(
                f"batch_concurrency must be in [1, 256], "
                f"got {self.batch_concurrency}"
            )

        if self.batch_timeout_s < 1:
            errors.append(
                f"batch_timeout_s must be >= 1, got {self.batch_timeout_s}"
            )

        # -- Data retention --------------------------------------------------
        if self.retention_years < 1:
            errors.append(
                f"retention_years must be >= 1, "
                f"got {self.retention_years}"
            )

        # -- EUDR commodities ------------------------------------------------
        if not self.eudr_commodities:
            errors.append("eudr_commodities must not be empty")

        # -- Document types --------------------------------------------------
        if not self.document_types:
            errors.append("document_types must not be empty")

        # -- Fraud pattern types ---------------------------------------------
        if self.fraud_rules_enabled and not self.fraud_pattern_types:
            errors.append(
                "fraud_pattern_types must not be empty when "
                "fraud_rules_enabled is True"
            )

        # -- Fraud severity weights ------------------------------------------
        for severity, weight in self.fraud_severity_weights.items():
            if weight < 0.0:
                errors.append(
                    f"fraud_severity_weights['{severity}'] must be >= 0.0, "
                    f"got {weight}"
                )

        # -- Registry rate limits --------------------------------------------
        for registry, limit in self.registry_rate_limits.items():
            if limit < 1:
                errors.append(
                    f"registry_rate_limits['{registry}'] must be >= 1, "
                    f"got {limit}"
                )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance tuning ----------------------------------------------
        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")

        if errors:
            raise ValueError(
                "DocumentAuthenticationConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "DocumentAuthenticationConfig validated successfully: "
            "hash=%s, secondary=%s, confidence_high=%.2f, "
            "confidence_medium=%.2f, sig_timeout=%ds, "
            "require_timestamp=%s, accept_self_signed=%s, "
            "ocsp=%s, crl_refresh=%dh, rsa_min=%d, ecdsa_min=%d, "
            "ct_log=%s, fraud_enabled=%s, velocity=%d/day, "
            "batch_max=%d, concurrency=%d, retention=%dy, "
            "provenance=%s, metrics=%s",
            self.hash_algorithm,
            self.secondary_hash,
            self.min_confidence_high,
            self.min_confidence_medium,
            self.signature_timeout_s,
            self.require_timestamp,
            self.accept_self_signed,
            self.ocsp_enabled,
            self.crl_refresh_hours,
            self.min_key_size_rsa,
            self.min_key_size_ecdsa,
            self.cert_transparency_enabled,
            self.fraud_rules_enabled,
            self.velocity_threshold_per_day,
            self.batch_max_size,
            self.batch_concurrency,
            self.retention_years,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DocumentAuthenticationConfig:
        """Build a DocumentAuthenticationConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_DAV_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated DocumentAuthenticationConfig instance, validated via
            ``__post_init__``.
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.strip().lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%r, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%r, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            # Classification
            min_confidence_high=_float(
                "MIN_CONFIDENCE_HIGH", cls.min_confidence_high,
            ),
            min_confidence_medium=_float(
                "MIN_CONFIDENCE_MEDIUM", cls.min_confidence_medium,
            ),
            max_batch_size=_int(
                "MAX_BATCH_SIZE", cls.max_batch_size,
            ),
            # Signature verification
            signature_timeout_s=_int(
                "SIGNATURE_TIMEOUT_S", cls.signature_timeout_s,
            ),
            require_timestamp=_bool(
                "REQUIRE_TIMESTAMP", cls.require_timestamp,
            ),
            accept_self_signed=_bool(
                "ACCEPT_SELF_SIGNED", cls.accept_self_signed,
            ),
            # Hash integrity
            hash_algorithm=_str(
                "HASH_ALGORITHM", cls.hash_algorithm,
            ),
            secondary_hash=_str(
                "SECONDARY_HASH", cls.secondary_hash,
            ),
            registry_ttl_days=_int(
                "REGISTRY_TTL_DAYS", cls.registry_ttl_days,
            ),
            # Certificate chain validation
            ocsp_enabled=_bool(
                "OCSP_ENABLED", cls.ocsp_enabled,
            ),
            crl_refresh_hours=_int(
                "CRL_REFRESH_HOURS", cls.crl_refresh_hours,
            ),
            min_key_size_rsa=_int(
                "MIN_KEY_SIZE_RSA", cls.min_key_size_rsa,
            ),
            min_key_size_ecdsa=_int(
                "MIN_KEY_SIZE_ECDSA", cls.min_key_size_ecdsa,
            ),
            cert_transparency_enabled=_bool(
                "CERT_TRANSPARENCY_ENABLED",
                cls.cert_transparency_enabled,
            ),
            # Metadata extraction
            creation_date_tolerance_days=_int(
                "CREATION_DATE_TOLERANCE_DAYS",
                cls.creation_date_tolerance_days,
            ),
            require_author_match=_bool(
                "REQUIRE_AUTHOR_MATCH", cls.require_author_match,
            ),
            flag_empty_metadata=_bool(
                "FLAG_EMPTY_METADATA", cls.flag_empty_metadata,
            ),
            # Fraud detection
            quantity_tolerance_percent=_float(
                "QUANTITY_TOLERANCE_PERCENT",
                cls.quantity_tolerance_percent,
            ),
            date_tolerance_days=_int(
                "DATE_TOLERANCE_DAYS", cls.date_tolerance_days,
            ),
            velocity_threshold_per_day=_int(
                "VELOCITY_THRESHOLD_PER_DAY",
                cls.velocity_threshold_per_day,
            ),
            round_number_threshold_percent=_float(
                "ROUND_NUMBER_THRESHOLD_PERCENT",
                cls.round_number_threshold_percent,
            ),
            fraud_rules_enabled=_bool(
                "FRAUD_RULES_ENABLED", cls.fraud_rules_enabled,
            ),
            # Cross-reference verification
            cache_ttl_hours=_int(
                "CACHE_TTL_HOURS", cls.cache_ttl_hours,
            ),
            fsc_api_rate_limit=_int(
                "FSC_API_RATE_LIMIT", cls.fsc_api_rate_limit,
            ),
            rspo_api_rate_limit=_int(
                "RSPO_API_RATE_LIMIT", cls.rspo_api_rate_limit,
            ),
            iscc_api_rate_limit=_int(
                "ISCC_API_RATE_LIMIT", cls.iscc_api_rate_limit,
            ),
            crossref_timeout_s=_int(
                "CROSSREF_TIMEOUT_S", cls.crossref_timeout_s,
            ),
            # Report generation
            default_format=_str(
                "DEFAULT_FORMAT", cls.default_format,
            ),
            retention_days=_int(
                "RETENTION_DAYS", cls.retention_days,
            ),
            evidence_package_enabled=_bool(
                "EVIDENCE_PACKAGE_ENABLED",
                cls.evidence_package_enabled,
            ),
            # Batch processing
            batch_max_size=_int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            batch_concurrency=_int(
                "BATCH_CONCURRENCY", cls.batch_concurrency,
            ),
            batch_timeout_s=_int(
                "BATCH_TIMEOUT_S", cls.batch_timeout_s,
            ),
            # Data retention
            retention_years=_int(
                "RETENTION_YEARS", cls.retention_years,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            # Rate limiting
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "DocumentAuthenticationConfig loaded: "
            "hash=%s, secondary=%s, "
            "confidence_high=%.2f, confidence_medium=%.2f, "
            "sig_timeout=%ds, require_timestamp=%s, "
            "accept_self_signed=%s, "
            "ocsp=%s, crl_refresh=%dh, "
            "rsa_min=%d, ecdsa_min=%d, ct_log=%s, "
            "create_date_tol=%dd, author_match=%s, "
            "qty_tol=%.1f%%, date_tol=%dd, velocity=%d/day, "
            "round_num=%.1f%%, fraud_enabled=%s, "
            "cache_ttl=%dh, fsc_rate=%d, rspo_rate=%d, iscc_rate=%d, "
            "crossref_timeout=%ds, "
            "format=%s, evidence_pkg=%s, "
            "batch_max=%d, concurrency=%d, timeout=%ds, "
            "retention=%dy, report_retention=%dd, "
            "provenance=%s, pool=%d, rate_limit=%d/min, "
            "metrics=%s",
            config.hash_algorithm,
            config.secondary_hash,
            config.min_confidence_high,
            config.min_confidence_medium,
            config.signature_timeout_s,
            config.require_timestamp,
            config.accept_self_signed,
            config.ocsp_enabled,
            config.crl_refresh_hours,
            config.min_key_size_rsa,
            config.min_key_size_ecdsa,
            config.cert_transparency_enabled,
            config.creation_date_tolerance_days,
            config.require_author_match,
            config.quantity_tolerance_percent,
            config.date_tolerance_days,
            config.velocity_threshold_per_day,
            config.round_number_threshold_percent,
            config.fraud_rules_enabled,
            config.cache_ttl_hours,
            config.fsc_api_rate_limit,
            config.rspo_api_rate_limit,
            config.iscc_api_rate_limit,
            config.crossref_timeout_s,
            config.default_format,
            config.evidence_package_enabled,
            config.batch_max_size,
            config.batch_concurrency,
            config.batch_timeout_s,
            config.retention_years,
            config.retention_days,
            config.enable_provenance,
            config.pool_size,
            config.rate_limit,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def classification_settings(self) -> Dict[str, Any]:
        """Return classification confidence settings as a dictionary.

        Returns:
            Dictionary with keys: min_confidence_high,
            min_confidence_medium, max_batch_size.
        """
        return {
            "min_confidence_high": self.min_confidence_high,
            "min_confidence_medium": self.min_confidence_medium,
            "max_batch_size": self.max_batch_size,
        }

    @property
    def signature_settings(self) -> Dict[str, Any]:
        """Return signature verification settings as a dictionary.

        Returns:
            Dictionary with keys: timeout_s, require_timestamp,
            accept_self_signed.
        """
        return {
            "timeout_s": self.signature_timeout_s,
            "require_timestamp": self.require_timestamp,
            "accept_self_signed": self.accept_self_signed,
        }

    @property
    def hash_settings(self) -> Dict[str, Any]:
        """Return hash integrity settings as a dictionary.

        Returns:
            Dictionary with keys: algorithm, secondary, registry_ttl_days.
        """
        return {
            "algorithm": self.hash_algorithm,
            "secondary": self.secondary_hash,
            "registry_ttl_days": self.registry_ttl_days,
        }

    @property
    def certificate_settings(self) -> Dict[str, Any]:
        """Return certificate chain validation settings.

        Returns:
            Dictionary with keys: ocsp_enabled, crl_refresh_hours,
            min_key_size_rsa, min_key_size_ecdsa,
            cert_transparency_enabled.
        """
        return {
            "ocsp_enabled": self.ocsp_enabled,
            "crl_refresh_hours": self.crl_refresh_hours,
            "min_key_size_rsa": self.min_key_size_rsa,
            "min_key_size_ecdsa": self.min_key_size_ecdsa,
            "cert_transparency_enabled": self.cert_transparency_enabled,
        }

    @property
    def metadata_settings(self) -> Dict[str, Any]:
        """Return metadata extraction settings.

        Returns:
            Dictionary with keys: creation_date_tolerance_days,
            require_author_match, flag_empty_metadata.
        """
        return {
            "creation_date_tolerance_days": (
                self.creation_date_tolerance_days
            ),
            "require_author_match": self.require_author_match,
            "flag_empty_metadata": self.flag_empty_metadata,
        }

    @property
    def fraud_settings(self) -> Dict[str, Any]:
        """Return fraud detection settings.

        Returns:
            Dictionary with keys: quantity_tolerance_percent,
            date_tolerance_days, velocity_threshold_per_day,
            round_number_threshold_percent, fraud_rules_enabled.
        """
        return {
            "quantity_tolerance_percent": self.quantity_tolerance_percent,
            "date_tolerance_days": self.date_tolerance_days,
            "velocity_threshold_per_day": self.velocity_threshold_per_day,
            "round_number_threshold_percent": (
                self.round_number_threshold_percent
            ),
            "fraud_rules_enabled": self.fraud_rules_enabled,
        }

    @property
    def crossref_settings(self) -> Dict[str, Any]:
        """Return cross-reference verification settings.

        Returns:
            Dictionary with keys: cache_ttl_hours, fsc_api_rate_limit,
            rspo_api_rate_limit, iscc_api_rate_limit, crossref_timeout_s.
        """
        return {
            "cache_ttl_hours": self.cache_ttl_hours,
            "fsc_api_rate_limit": self.fsc_api_rate_limit,
            "rspo_api_rate_limit": self.rspo_api_rate_limit,
            "iscc_api_rate_limit": self.iscc_api_rate_limit,
            "crossref_timeout_s": self.crossref_timeout_s,
        }

    def get_registry_rate_limit(self, registry: str) -> int:
        """Return the rate limit for a given registry type.

        Args:
            registry: Registry identifier (fsc, rspo, iscc, fairtrade,
                utz_ra, ippc, national_customs).

        Returns:
            Requests per minute for the specified registry.
        """
        registry_lower = registry.lower().strip()
        return self.registry_rate_limits.get(registry_lower, 10)

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Re-run post-init validation and return True if valid.

        Returns:
            True if configuration passes all validation checks.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        self.__post_init__()
        return True

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain Python dictionary.

        Sensitive connection strings (database_url, redis_url) are
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # Connections (redacted)
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # Logging
            "log_level": self.log_level,
            # Performance tuning
            "pool_size": self.pool_size,
            # Classification
            "min_confidence_high": self.min_confidence_high,
            "min_confidence_medium": self.min_confidence_medium,
            "max_batch_size": self.max_batch_size,
            # Signature verification
            "signature_timeout_s": self.signature_timeout_s,
            "require_timestamp": self.require_timestamp,
            "accept_self_signed": self.accept_self_signed,
            # Hash integrity
            "hash_algorithm": self.hash_algorithm,
            "secondary_hash": self.secondary_hash,
            "registry_ttl_days": self.registry_ttl_days,
            # Certificate chain validation
            "ocsp_enabled": self.ocsp_enabled,
            "crl_refresh_hours": self.crl_refresh_hours,
            "min_key_size_rsa": self.min_key_size_rsa,
            "min_key_size_ecdsa": self.min_key_size_ecdsa,
            "cert_transparency_enabled": self.cert_transparency_enabled,
            # Metadata extraction
            "creation_date_tolerance_days": (
                self.creation_date_tolerance_days
            ),
            "require_author_match": self.require_author_match,
            "flag_empty_metadata": self.flag_empty_metadata,
            # Fraud detection
            "quantity_tolerance_percent": self.quantity_tolerance_percent,
            "date_tolerance_days": self.date_tolerance_days,
            "velocity_threshold_per_day": self.velocity_threshold_per_day,
            "round_number_threshold_percent": (
                self.round_number_threshold_percent
            ),
            "fraud_rules_enabled": self.fraud_rules_enabled,
            # Cross-reference verification
            "cache_ttl_hours": self.cache_ttl_hours,
            "fsc_api_rate_limit": self.fsc_api_rate_limit,
            "rspo_api_rate_limit": self.rspo_api_rate_limit,
            "iscc_api_rate_limit": self.iscc_api_rate_limit,
            "crossref_timeout_s": self.crossref_timeout_s,
            # Report generation
            "default_format": self.default_format,
            "retention_days": self.retention_days,
            "evidence_package_enabled": self.evidence_package_enabled,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Data retention
            "retention_years": self.retention_years,
            # EUDR commodities
            "eudr_commodities": list(self.eudr_commodities),
            # Document types (count only to keep output concise)
            "document_types_count": len(self.document_types),
            # Fraud pattern types (count only)
            "fraud_pattern_types_count": len(self.fraud_pattern_types),
            # Trusted CAs (count only)
            "trusted_cas_count": len(self.trusted_cas),
            # Registry rate limits
            "registry_rate_limits": dict(self.registry_rate_limits),
            # Provenance
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # Metrics
            "enable_metrics": self.enable_metrics,
            # Rate limiting
            "rate_limit": self.rate_limit,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"DocumentAuthenticationConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[DocumentAuthenticationConfig] = None
_config_lock = threading.Lock()


def get_config() -> DocumentAuthenticationConfig:
    """Return the singleton DocumentAuthenticationConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_DAV_*`` environment variables.

    Returns:
        DocumentAuthenticationConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.hash_algorithm
        'sha256'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DocumentAuthenticationConfig.from_env()
    return _config_instance


def set_config(config: DocumentAuthenticationConfig) -> None:
    """Replace the singleton DocumentAuthenticationConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New DocumentAuthenticationConfig to install.

    Example:
        >>> cfg = DocumentAuthenticationConfig(hash_algorithm="sha512")
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "DocumentAuthenticationConfig replaced programmatically: "
        "hash=%s, confidence_high=%.2f, "
        "batch_max=%d",
        config.hash_algorithm,
        config.min_confidence_high,
        config.batch_max_size,
    )


def reset_config() -> None:
    """Reset the singleton DocumentAuthenticationConfig to None.

    The next call to get_config() will re-read GL_EUDR_DAV_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("DocumentAuthenticationConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "DocumentAuthenticationConfig",
    "get_config",
    "set_config",
    "reset_config",
]
