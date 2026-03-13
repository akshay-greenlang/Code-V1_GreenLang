# -*- coding: utf-8 -*-
"""
QR Code Generator Configuration - AGENT-EUDR-014

Centralized configuration for the QR Code Generator Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- QR Generation: version selection (auto/v1-v40), error correction level
  (L/M/Q/H), module size, quiet zone, output format (PNG/SVG/PDF/ZPL/EPS),
  DPI, logo embedding, ISO/IEC 18004 quality grade target
- Payload: content type (full_traceability/compact_verification/consumer_summary/
  batch_identifier/blockchain_anchor), max payload bytes (2953 for QR v40-H),
  zlib compression, AES encryption, payload schema versioning
- Label: template selection (product/shipping/pallet/container/consumer),
  font settings, EUDR compliance status colour coding, print bleed
- Batch Codes: prefix format with operator/commodity/year tokens, check
  digit algorithm (Luhn/ISO 7064/CRC-8), zero-padding, sequence start
- Verification URLs: base URL, short URL integration, HMAC-signed
  verification token with configurable TTL and truncation length
- Anti-Counterfeiting: HMAC secret key rotation, digital watermark
  toggle, scan velocity throttling, geo-fence enforcement
- Bulk Generation: max job size (100,000), worker count, timeout,
  output packaging (ZIP), post-generation validation
- Lifecycle: code TTL (5 years per EUDR Article 14), scan logging,
  reprint limits
- Batch processing: size limits, concurrency, timeout
- Data retention: EUDR Article 14 five-year retention
- Provenance tracking (genesis hash, SHA-256 chain hashing)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_QRG_`` prefix (e.g. ``GL_EUDR_QRG_DATABASE_URL``,
``GL_EUDR_QRG_DEFAULT_ERROR_CORRECTION``).

Environment Variable Reference (GL_EUDR_QRG_ prefix):
    GL_EUDR_QRG_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_QRG_REDIS_URL                     - Redis connection URL
    GL_EUDR_QRG_LOG_LEVEL                     - Logging level
    GL_EUDR_QRG_POOL_SIZE                     - Database pool size
    GL_EUDR_QRG_DEFAULT_VERSION               - QR code version (auto or 1-40)
    GL_EUDR_QRG_DEFAULT_ERROR_CORRECTION      - Error correction (L/M/Q/H)
    GL_EUDR_QRG_DEFAULT_MODULE_SIZE           - Module pixel size
    GL_EUDR_QRG_DEFAULT_QUIET_ZONE            - Quiet zone modules
    GL_EUDR_QRG_DEFAULT_OUTPUT_FORMAT         - Output format (png/svg/pdf/zpl/eps)
    GL_EUDR_QRG_DEFAULT_DPI                   - Output DPI
    GL_EUDR_QRG_ENABLE_LOGO_EMBEDDING         - Enable centre logo embedding
    GL_EUDR_QRG_QUALITY_GRADE_TARGET          - ISO 18004 grade (A/B/C/D)
    GL_EUDR_QRG_DEFAULT_CONTENT_TYPE          - Payload content type
    GL_EUDR_QRG_MAX_PAYLOAD_BYTES             - Max payload bytes
    GL_EUDR_QRG_ENABLE_COMPRESSION            - Enable zlib compression
    GL_EUDR_QRG_COMPRESSION_THRESHOLD_BYTES   - Compression threshold
    GL_EUDR_QRG_ENABLE_ENCRYPTION             - Enable AES payload encryption
    GL_EUDR_QRG_PAYLOAD_VERSION               - Payload schema version
    GL_EUDR_QRG_DEFAULT_TEMPLATE              - Label template name
    GL_EUDR_QRG_DEFAULT_FONT                  - Label font family
    GL_EUDR_QRG_DEFAULT_FONT_SIZE             - Label font size (pt)
    GL_EUDR_QRG_COMPLIANT_COLOR_HEX           - Hex colour for compliant
    GL_EUDR_QRG_PENDING_COLOR_HEX             - Hex colour for pending
    GL_EUDR_QRG_NON_COMPLIANT_COLOR_HEX       - Hex colour for non-compliant
    GL_EUDR_QRG_BLEED_MM                      - Print bleed in mm
    GL_EUDR_QRG_DEFAULT_PREFIX_FORMAT         - Batch code prefix template
    GL_EUDR_QRG_CHECK_DIGIT_ALGORITHM         - Check digit algorithm
    GL_EUDR_QRG_CODE_PADDING                  - Zero-pad width
    GL_EUDR_QRG_START_SEQUENCE                - Starting sequence number
    GL_EUDR_QRG_BASE_VERIFICATION_URL         - Verification base URL
    GL_EUDR_QRG_SHORT_URL_ENABLED             - Enable short URL service
    GL_EUDR_QRG_SHORT_URL_SERVICE             - Short URL service endpoint
    GL_EUDR_QRG_VERIFICATION_TOKEN_TTL_YEARS  - Token TTL in years
    GL_EUDR_QRG_HMAC_TRUNCATION_LENGTH        - HMAC truncation length
    GL_EUDR_QRG_HMAC_SECRET_KEY               - HMAC signing secret key
    GL_EUDR_QRG_KEY_ROTATION_DAYS             - Key rotation interval days
    GL_EUDR_QRG_ENABLE_DIGITAL_WATERMARK      - Enable digital watermark
    GL_EUDR_QRG_SCAN_VELOCITY_THRESHOLD       - Max scans per minute
    GL_EUDR_QRG_GEO_FENCE_ENABLED             - Enable geo-fence checks
    GL_EUDR_QRG_BULK_MAX_SIZE                 - Max codes per bulk job
    GL_EUDR_QRG_BULK_WORKERS                  - Bulk generation workers
    GL_EUDR_QRG_BULK_TIMEOUT_S                - Bulk job timeout seconds
    GL_EUDR_QRG_BULK_OUTPUT_FORMAT            - Bulk output format (zip)
    GL_EUDR_QRG_ENABLE_OUTPUT_VALIDATION      - Validate after generation
    GL_EUDR_QRG_DEFAULT_TTL_YEARS             - Code TTL in years
    GL_EUDR_QRG_SCAN_LOGGING_ENABLED          - Enable scan event logging
    GL_EUDR_QRG_MAX_REPRINTS                  - Max reprint count
    GL_EUDR_QRG_BATCH_MAX_SIZE                - Batch processing max size
    GL_EUDR_QRG_BATCH_CONCURRENCY             - Batch concurrency workers
    GL_EUDR_QRG_BATCH_TIMEOUT_S               - Batch timeout seconds
    GL_EUDR_QRG_RETENTION_YEARS               - Data retention years
    GL_EUDR_QRG_ENABLE_PROVENANCE             - Enable provenance tracking
    GL_EUDR_QRG_GENESIS_HASH                  - Genesis hash anchor
    GL_EUDR_QRG_ENABLE_METRICS                - Enable Prometheus metrics
    GL_EUDR_QRG_RATE_LIMIT                    - Max requests per minute

Example:
    >>> from greenlang.agents.eudr.qr_code_generator.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_error_correction, cfg.default_dpi)
    M 300

    >>> # Override for testing
    >>> from greenlang.agents.eudr.qr_code_generator.config import (
    ...     set_config, reset_config, QRCodeGeneratorConfig,
    ... )
    >>> set_config(QRCodeGeneratorConfig(default_error_correction="H"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
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

_ENV_PREFIX = "GL_EUDR_QRG_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid QR code versions
# ---------------------------------------------------------------------------

_VALID_QR_VERSIONS = frozenset(
    {"auto"} | {str(i) for i in range(1, 41)}
)

# ---------------------------------------------------------------------------
# Valid error correction levels (ISO/IEC 18004)
# ---------------------------------------------------------------------------

_VALID_ERROR_CORRECTIONS = frozenset({"L", "M", "Q", "H"})

# ---------------------------------------------------------------------------
# Valid output formats
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = frozenset({"png", "svg", "pdf", "zpl", "eps"})

# ---------------------------------------------------------------------------
# Valid content types for QR payload
# ---------------------------------------------------------------------------

_VALID_CONTENT_TYPES = frozenset({
    "full_traceability",
    "compact_verification",
    "consumer_summary",
    "batch_identifier",
    "blockchain_anchor",
})

# ---------------------------------------------------------------------------
# Valid symbology types
# ---------------------------------------------------------------------------

_VALID_SYMBOLOGY_TYPES = frozenset({
    "qr_code",
    "micro_qr",
    "data_matrix",
    "gs1_digital_link",
})

# ---------------------------------------------------------------------------
# Valid label templates
# ---------------------------------------------------------------------------

_VALID_LABEL_TEMPLATES = frozenset({
    "product_label",
    "shipping_label",
    "pallet_label",
    "container_label",
    "consumer_label",
})

# ---------------------------------------------------------------------------
# Valid check digit algorithms
# ---------------------------------------------------------------------------

_VALID_CHECK_DIGIT_ALGORITHMS = frozenset({
    "luhn",
    "iso7064_mod11_10",
    "crc8",
})

# ---------------------------------------------------------------------------
# Valid quality grades (ISO/IEC 18004 / 15416)
# ---------------------------------------------------------------------------

_VALID_QUALITY_GRADES = frozenset({"A", "B", "C", "D"})

# ---------------------------------------------------------------------------
# Valid DPI levels
# ---------------------------------------------------------------------------

_VALID_DPI_VALUES = frozenset({72, 150, 300, 600})

# ---------------------------------------------------------------------------
# Valid bulk output formats
# ---------------------------------------------------------------------------

_VALID_BULK_OUTPUT_FORMATS = frozenset({"zip", "tar_gz"})

# ---------------------------------------------------------------------------
# Valid payload encodings
# ---------------------------------------------------------------------------

_VALID_PAYLOAD_ENCODINGS = frozenset({"utf8", "base64", "zlib_base64"})

# ---------------------------------------------------------------------------
# Default EUDR commodities
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ---------------------------------------------------------------------------
# Valid hex colour pattern
# ---------------------------------------------------------------------------


def _is_valid_hex_colour(value: str) -> bool:
    """Check if a string is a valid 6-digit hex colour with leading #."""
    if len(value) != 7 or value[0] != "#":
        return False
    try:
        int(value[1:], 16)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# QRCodeGeneratorConfig
# ---------------------------------------------------------------------------


@dataclass
class QRCodeGeneratorConfig:
    """Complete configuration for the EUDR QR Code Generator Agent.

    Attributes are grouped by concern: connections, logging, QR generation,
    payload, label rendering, batch codes, verification URLs,
    anti-counterfeiting, bulk generation, lifecycle, batch processing,
    data retention, provenance tracking, metrics, and rate limiting.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_QRG_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            QR code records, labels, batch codes, and audit logs.
        redis_url: Redis connection URL for scan event deduplication,
            rate limiting counters, and verification token caching.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        pool_size: PostgreSQL connection pool size.
        default_version: QR code version selection. ``auto`` for automatic
            version selection based on payload size, or ``1``-``40`` for
            a fixed version (ISO/IEC 18004).
        default_error_correction: Error correction level per ISO/IEC 18004.
            L (7%), M (15%), Q (25%), H (30%) recovery capability.
        default_module_size: Pixel width/height of each QR module in the
            rendered image.
        default_quiet_zone: Number of module-width quiet zone border around
            the QR code (ISO/IEC 18004 requires minimum 4).
        default_output_format: Default output image format for generated
            QR codes (png, svg, pdf, zpl, eps).
        default_dpi: Output resolution in dots per inch for raster formats.
        enable_logo_embedding: Whether to embed a centre logo (reduces
            effective error correction capacity).
        quality_grade_target: Target ISO/IEC 15416 print quality grade
            (A/B/C/D). Grade B is the minimum for reliable scanning.
        default_content_type: Default payload content type for QR codes.
        max_payload_bytes: Maximum payload size in bytes. QR v40-H supports
            up to 2953 bytes of binary data.
        enable_compression: Enable zlib compression for payloads exceeding
            the compression threshold.
        compression_threshold_bytes: Payload size in bytes above which
            zlib compression is applied.
        enable_encryption: Enable AES-256-GCM payload encryption for
            sensitive supply chain data.
        payload_version: Payload schema version string for forward
            compatibility of encoded data.
        default_template: Default label rendering template.
        default_font: Font family for label text rendering.
        default_font_size: Font size in points for label text.
        compliant_color_hex: Hex colour code for EUDR-compliant status
            indicators on labels.
        pending_color_hex: Hex colour code for pending/under-review status
            indicators on labels.
        non_compliant_color_hex: Hex colour code for non-compliant status
            indicators on labels.
        bleed_mm: Print bleed margin in millimetres for label templates.
        default_prefix_format: Template string for batch code prefixes
            with ``{operator}``, ``{commodity}``, ``{year}`` tokens.
        check_digit_algorithm: Algorithm for computing batch code check
            digits (luhn, iso7064_mod11_10, crc8).
        code_padding: Zero-pad width for the numeric portion of batch
            codes (e.g. 5 produces ``00001``).
        start_sequence: Starting sequence number for batch code generation.
        base_verification_url: Base URL for verification landing pages.
        short_url_enabled: Whether to generate short URLs via an external
            shortener service.
        short_url_service: Endpoint URL for the short URL generation service.
        verification_token_ttl_years: Number of years before a verification
            token expires (aligned with EUDR Article 14).
        hmac_truncation_length: Number of hex characters to retain from
            the HMAC signature in verification tokens.
        hmac_secret_key: Secret key for HMAC-SHA256 signing of verification
            tokens. Must be set in production via env var.
        key_rotation_days: Interval in days for rotating the HMAC secret key.
        enable_digital_watermark: Whether to embed a digital watermark in
            generated QR code images for anti-counterfeiting.
        scan_velocity_threshold: Maximum number of scans per minute for a
            single code before triggering a counterfeit alert.
        geo_fence_enabled: Whether to enforce geographic boundary checks
            on scan events.
        bulk_max_size: Maximum number of QR codes in a single bulk
            generation job.
        bulk_workers: Number of concurrent worker threads for bulk
            generation.
        bulk_timeout_s: Timeout in seconds for a bulk generation job.
        bulk_output_format: Packaging format for bulk output (zip, tar_gz).
        enable_output_validation: Whether to validate (decode/scan) each
            generated QR code after rendering.
        default_ttl_years: Default time-to-live in years for generated
            QR codes before automatic expiry.
        scan_logging_enabled: Whether to log every scan event for audit
            and analytics.
        max_reprints: Maximum number of times a QR code label may be
            reprinted before requiring re-authorization.
        batch_max_size: Maximum number of records in a single batch
            processing job.
        batch_concurrency: Maximum concurrent batch processing workers.
        batch_timeout_s: Timeout in seconds for a single batch job.
        retention_years: Data retention in years per EUDR Article 14.
        eudr_commodities: List of EUDR-regulated commodity types.
        enable_provenance: Enable SHA-256 provenance chain tracking
            for all QR code generation operations.
        genesis_hash: Genesis anchor string for the provenance chain,
            unique to the QR Code Generator agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_qrg_`` prefix.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10

    # -- QR generation settings ----------------------------------------------
    default_version: str = "auto"
    default_error_correction: str = "M"
    default_module_size: int = 10
    default_quiet_zone: int = 4
    default_output_format: str = "png"
    default_dpi: int = 300
    enable_logo_embedding: bool = False
    quality_grade_target: str = "B"

    # -- Payload settings ----------------------------------------------------
    default_content_type: str = "compact_verification"
    max_payload_bytes: int = 2953
    enable_compression: bool = True
    compression_threshold_bytes: int = 500
    enable_encryption: bool = False
    payload_version: str = "1.0"

    # -- Label settings ------------------------------------------------------
    default_template: str = "product_label"
    default_font: str = "DejaVuSans"
    default_font_size: int = 12
    compliant_color_hex: str = "#2E7D32"
    pending_color_hex: str = "#F57F17"
    non_compliant_color_hex: str = "#C62828"
    bleed_mm: int = 3

    # -- Batch code settings -------------------------------------------------
    default_prefix_format: str = "{operator}-{commodity}-{year}"
    check_digit_algorithm: str = "luhn"
    code_padding: int = 5
    start_sequence: int = 1

    # -- Verification URL settings -------------------------------------------
    base_verification_url: str = "https://verify.greenlang.eu"
    short_url_enabled: bool = False
    short_url_service: str = ""
    verification_token_ttl_years: int = 5
    hmac_truncation_length: int = 8

    # -- Anti-counterfeiting settings ----------------------------------------
    hmac_secret_key: str = ""
    key_rotation_days: int = 90
    enable_digital_watermark: bool = False
    scan_velocity_threshold: int = 100
    geo_fence_enabled: bool = False

    # -- Bulk generation settings --------------------------------------------
    bulk_max_size: int = 100000
    bulk_workers: int = 4
    bulk_timeout_s: int = 3600
    bulk_output_format: str = "zip"
    enable_output_validation: bool = True

    # -- Lifecycle settings --------------------------------------------------
    default_ttl_years: int = 5
    scan_logging_enabled: bool = True
    max_reprints: int = 3

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 500
    batch_concurrency: int = 4
    batch_timeout_s: int = 600

    # -- Data retention (EUDR Article 14) ------------------------------------
    retention_years: int = 5

    # -- EUDR commodities ----------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-QRG-014-QR-CODE-GENERATOR-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Rate limiting -------------------------------------------------------
    rate_limit: int = 200

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

        # -- Performance tuning ----------------------------------------------
        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")

        # -- QR generation settings ------------------------------------------
        normalised_version = self.default_version.lower().strip()
        if normalised_version not in _VALID_QR_VERSIONS:
            errors.append(
                f"default_version must be 'auto' or 1-40, "
                f"got '{self.default_version}'"
            )
        else:
            self.default_version = normalised_version

        normalised_ec = self.default_error_correction.upper().strip()
        if normalised_ec not in _VALID_ERROR_CORRECTIONS:
            errors.append(
                f"default_error_correction must be one of "
                f"{sorted(_VALID_ERROR_CORRECTIONS)}, "
                f"got '{self.default_error_correction}'"
            )
        else:
            self.default_error_correction = normalised_ec

        if self.default_module_size < 1:
            errors.append(
                f"default_module_size must be >= 1, "
                f"got {self.default_module_size}"
            )
        if self.default_module_size > 100:
            errors.append(
                f"default_module_size must be <= 100, "
                f"got {self.default_module_size}"
            )

        if self.default_quiet_zone < 0:
            errors.append(
                f"default_quiet_zone must be >= 0, "
                f"got {self.default_quiet_zone}"
            )
        if self.default_quiet_zone > 20:
            errors.append(
                f"default_quiet_zone must be <= 20, "
                f"got {self.default_quiet_zone}"
            )

        normalised_format = self.default_output_format.lower().strip()
        if normalised_format not in _VALID_OUTPUT_FORMATS:
            errors.append(
                f"default_output_format must be one of "
                f"{sorted(_VALID_OUTPUT_FORMATS)}, "
                f"got '{self.default_output_format}'"
            )
        else:
            self.default_output_format = normalised_format

        if self.default_dpi < 72:
            errors.append(
                f"default_dpi must be >= 72, got {self.default_dpi}"
            )
        if self.default_dpi > 1200:
            errors.append(
                f"default_dpi must be <= 1200, got {self.default_dpi}"
            )

        normalised_grade = self.quality_grade_target.upper().strip()
        if normalised_grade not in _VALID_QUALITY_GRADES:
            errors.append(
                f"quality_grade_target must be one of "
                f"{sorted(_VALID_QUALITY_GRADES)}, "
                f"got '{self.quality_grade_target}'"
            )
        else:
            self.quality_grade_target = normalised_grade

        # -- Payload settings ------------------------------------------------
        normalised_content = self.default_content_type.lower().strip()
        if normalised_content not in _VALID_CONTENT_TYPES:
            errors.append(
                f"default_content_type must be one of "
                f"{sorted(_VALID_CONTENT_TYPES)}, "
                f"got '{self.default_content_type}'"
            )
        else:
            self.default_content_type = normalised_content

        if self.max_payload_bytes < 1:
            errors.append(
                f"max_payload_bytes must be >= 1, "
                f"got {self.max_payload_bytes}"
            )
        if self.max_payload_bytes > 4296:
            errors.append(
                f"max_payload_bytes must be <= 4296 (QR v40-L max), "
                f"got {self.max_payload_bytes}"
            )

        if self.compression_threshold_bytes < 0:
            errors.append(
                f"compression_threshold_bytes must be >= 0, "
                f"got {self.compression_threshold_bytes}"
            )
        if self.compression_threshold_bytes > self.max_payload_bytes:
            errors.append(
                f"compression_threshold_bytes must be <= "
                f"max_payload_bytes ({self.max_payload_bytes}), "
                f"got {self.compression_threshold_bytes}"
            )

        if not self.payload_version:
            errors.append("payload_version must not be empty")

        # -- Label settings --------------------------------------------------
        normalised_template = self.default_template.lower().strip()
        if normalised_template not in _VALID_LABEL_TEMPLATES:
            errors.append(
                f"default_template must be one of "
                f"{sorted(_VALID_LABEL_TEMPLATES)}, "
                f"got '{self.default_template}'"
            )
        else:
            self.default_template = normalised_template

        if not self.default_font:
            errors.append("default_font must not be empty")

        if self.default_font_size < 4:
            errors.append(
                f"default_font_size must be >= 4, "
                f"got {self.default_font_size}"
            )
        if self.default_font_size > 72:
            errors.append(
                f"default_font_size must be <= 72, "
                f"got {self.default_font_size}"
            )

        if not _is_valid_hex_colour(self.compliant_color_hex):
            errors.append(
                f"compliant_color_hex must be a valid #RRGGBB hex "
                f"colour, got '{self.compliant_color_hex}'"
            )

        if not _is_valid_hex_colour(self.pending_color_hex):
            errors.append(
                f"pending_color_hex must be a valid #RRGGBB hex "
                f"colour, got '{self.pending_color_hex}'"
            )

        if not _is_valid_hex_colour(self.non_compliant_color_hex):
            errors.append(
                f"non_compliant_color_hex must be a valid #RRGGBB hex "
                f"colour, got '{self.non_compliant_color_hex}'"
            )

        if self.bleed_mm < 0:
            errors.append(
                f"bleed_mm must be >= 0, got {self.bleed_mm}"
            )
        if self.bleed_mm > 20:
            errors.append(
                f"bleed_mm must be <= 20, got {self.bleed_mm}"
            )

        # -- Batch code settings ---------------------------------------------
        if not self.default_prefix_format:
            errors.append("default_prefix_format must not be empty")

        normalised_check = self.check_digit_algorithm.lower().strip()
        if normalised_check not in _VALID_CHECK_DIGIT_ALGORITHMS:
            errors.append(
                f"check_digit_algorithm must be one of "
                f"{sorted(_VALID_CHECK_DIGIT_ALGORITHMS)}, "
                f"got '{self.check_digit_algorithm}'"
            )
        else:
            self.check_digit_algorithm = normalised_check

        if self.code_padding < 1:
            errors.append(
                f"code_padding must be >= 1, got {self.code_padding}"
            )
        if self.code_padding > 20:
            errors.append(
                f"code_padding must be <= 20, got {self.code_padding}"
            )

        if self.start_sequence < 0:
            errors.append(
                f"start_sequence must be >= 0, "
                f"got {self.start_sequence}"
            )

        # -- Verification URL settings ---------------------------------------
        if not self.base_verification_url:
            errors.append("base_verification_url must not be empty")

        if self.short_url_enabled and not self.short_url_service:
            errors.append(
                "short_url_service must be set when "
                "short_url_enabled is True"
            )

        if self.verification_token_ttl_years < 1:
            errors.append(
                f"verification_token_ttl_years must be >= 1, "
                f"got {self.verification_token_ttl_years}"
            )
        if self.verification_token_ttl_years > 25:
            errors.append(
                f"verification_token_ttl_years must be <= 25, "
                f"got {self.verification_token_ttl_years}"
            )

        if self.hmac_truncation_length < 4:
            errors.append(
                f"hmac_truncation_length must be >= 4, "
                f"got {self.hmac_truncation_length}"
            )
        if self.hmac_truncation_length > 64:
            errors.append(
                f"hmac_truncation_length must be <= 64, "
                f"got {self.hmac_truncation_length}"
            )

        # -- Anti-counterfeiting settings ------------------------------------
        if self.key_rotation_days < 1:
            errors.append(
                f"key_rotation_days must be >= 1, "
                f"got {self.key_rotation_days}"
            )
        if self.key_rotation_days > 3650:
            errors.append(
                f"key_rotation_days must be <= 3650 (10 years), "
                f"got {self.key_rotation_days}"
            )

        if self.scan_velocity_threshold < 1:
            errors.append(
                f"scan_velocity_threshold must be >= 1, "
                f"got {self.scan_velocity_threshold}"
            )
        if self.scan_velocity_threshold > 100000:
            errors.append(
                f"scan_velocity_threshold must be <= 100000, "
                f"got {self.scan_velocity_threshold}"
            )

        # -- Bulk generation settings ----------------------------------------
        if self.bulk_max_size < 1:
            errors.append(
                f"bulk_max_size must be >= 1, "
                f"got {self.bulk_max_size}"
            )
        if self.bulk_max_size > 1000000:
            errors.append(
                f"bulk_max_size must be <= 1000000, "
                f"got {self.bulk_max_size}"
            )

        if self.bulk_workers < 1:
            errors.append(
                f"bulk_workers must be >= 1, "
                f"got {self.bulk_workers}"
            )
        if self.bulk_workers > 64:
            errors.append(
                f"bulk_workers must be <= 64, "
                f"got {self.bulk_workers}"
            )

        if self.bulk_timeout_s < 1:
            errors.append(
                f"bulk_timeout_s must be >= 1, "
                f"got {self.bulk_timeout_s}"
            )
        if self.bulk_timeout_s > 86400:
            errors.append(
                f"bulk_timeout_s must be <= 86400 (24h), "
                f"got {self.bulk_timeout_s}"
            )

        normalised_bulk_fmt = self.bulk_output_format.lower().strip()
        if normalised_bulk_fmt not in _VALID_BULK_OUTPUT_FORMATS:
            errors.append(
                f"bulk_output_format must be one of "
                f"{sorted(_VALID_BULK_OUTPUT_FORMATS)}, "
                f"got '{self.bulk_output_format}'"
            )
        else:
            self.bulk_output_format = normalised_bulk_fmt

        # -- Lifecycle settings ----------------------------------------------
        if self.default_ttl_years < 1:
            errors.append(
                f"default_ttl_years must be >= 1, "
                f"got {self.default_ttl_years}"
            )
        if self.default_ttl_years > 25:
            errors.append(
                f"default_ttl_years must be <= 25, "
                f"got {self.default_ttl_years}"
            )

        if self.max_reprints < 0:
            errors.append(
                f"max_reprints must be >= 0, "
                f"got {self.max_reprints}"
            )
        if self.max_reprints > 100:
            errors.append(
                f"max_reprints must be <= 100, "
                f"got {self.max_reprints}"
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

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Rate limiting ---------------------------------------------------
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")

        if errors:
            raise ValueError(
                "QRCodeGeneratorConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "QRCodeGeneratorConfig validated successfully: "
            "version=%s, ec=%s, module=%d, quiet=%d, "
            "format=%s, dpi=%d, logo=%s, grade=%s, "
            "content=%s, max_payload=%d, compress=%s, "
            "threshold=%d, encrypt=%s, payload_ver=%s, "
            "template=%s, font=%s/%dpt, "
            "prefix=%s, check=%s, pad=%d, start=%d, "
            "verify_url=%s, short=%s, ttl=%dy, hmac_len=%d, "
            "rotation=%dd, watermark=%s, velocity=%d, geo=%s, "
            "bulk_max=%d, workers=%d, bulk_timeout=%ds, "
            "ttl_years=%d, scan_log=%s, reprints=%d, "
            "batch_max=%d, concurrency=%d, retention=%dy, "
            "provenance=%s, metrics=%s",
            self.default_version,
            self.default_error_correction,
            self.default_module_size,
            self.default_quiet_zone,
            self.default_output_format,
            self.default_dpi,
            self.enable_logo_embedding,
            self.quality_grade_target,
            self.default_content_type,
            self.max_payload_bytes,
            self.enable_compression,
            self.compression_threshold_bytes,
            self.enable_encryption,
            self.payload_version,
            self.default_template,
            self.default_font,
            self.default_font_size,
            self.default_prefix_format,
            self.check_digit_algorithm,
            self.code_padding,
            self.start_sequence,
            self.base_verification_url,
            self.short_url_enabled,
            self.verification_token_ttl_years,
            self.hmac_truncation_length,
            self.key_rotation_days,
            self.enable_digital_watermark,
            self.scan_velocity_threshold,
            self.geo_fence_enabled,
            self.bulk_max_size,
            self.bulk_workers,
            self.bulk_timeout_s,
            self.default_ttl_years,
            self.scan_logging_enabled,
            self.max_reprints,
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
    def from_env(cls) -> QRCodeGeneratorConfig:
        """Build a QRCodeGeneratorConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_QRG_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated QRCodeGeneratorConfig instance, validated via
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
            # QR generation
            default_version=_str(
                "DEFAULT_VERSION", cls.default_version,
            ),
            default_error_correction=_str(
                "DEFAULT_ERROR_CORRECTION",
                cls.default_error_correction,
            ),
            default_module_size=_int(
                "DEFAULT_MODULE_SIZE", cls.default_module_size,
            ),
            default_quiet_zone=_int(
                "DEFAULT_QUIET_ZONE", cls.default_quiet_zone,
            ),
            default_output_format=_str(
                "DEFAULT_OUTPUT_FORMAT", cls.default_output_format,
            ),
            default_dpi=_int("DEFAULT_DPI", cls.default_dpi),
            enable_logo_embedding=_bool(
                "ENABLE_LOGO_EMBEDDING", cls.enable_logo_embedding,
            ),
            quality_grade_target=_str(
                "QUALITY_GRADE_TARGET", cls.quality_grade_target,
            ),
            # Payload
            default_content_type=_str(
                "DEFAULT_CONTENT_TYPE", cls.default_content_type,
            ),
            max_payload_bytes=_int(
                "MAX_PAYLOAD_BYTES", cls.max_payload_bytes,
            ),
            enable_compression=_bool(
                "ENABLE_COMPRESSION", cls.enable_compression,
            ),
            compression_threshold_bytes=_int(
                "COMPRESSION_THRESHOLD_BYTES",
                cls.compression_threshold_bytes,
            ),
            enable_encryption=_bool(
                "ENABLE_ENCRYPTION", cls.enable_encryption,
            ),
            payload_version=_str(
                "PAYLOAD_VERSION", cls.payload_version,
            ),
            # Label
            default_template=_str(
                "DEFAULT_TEMPLATE", cls.default_template,
            ),
            default_font=_str("DEFAULT_FONT", cls.default_font),
            default_font_size=_int(
                "DEFAULT_FONT_SIZE", cls.default_font_size,
            ),
            compliant_color_hex=_str(
                "COMPLIANT_COLOR_HEX", cls.compliant_color_hex,
            ),
            pending_color_hex=_str(
                "PENDING_COLOR_HEX", cls.pending_color_hex,
            ),
            non_compliant_color_hex=_str(
                "NON_COMPLIANT_COLOR_HEX",
                cls.non_compliant_color_hex,
            ),
            bleed_mm=_int("BLEED_MM", cls.bleed_mm),
            # Batch codes
            default_prefix_format=_str(
                "DEFAULT_PREFIX_FORMAT", cls.default_prefix_format,
            ),
            check_digit_algorithm=_str(
                "CHECK_DIGIT_ALGORITHM", cls.check_digit_algorithm,
            ),
            code_padding=_int("CODE_PADDING", cls.code_padding),
            start_sequence=_int(
                "START_SEQUENCE", cls.start_sequence,
            ),
            # Verification URLs
            base_verification_url=_str(
                "BASE_VERIFICATION_URL", cls.base_verification_url,
            ),
            short_url_enabled=_bool(
                "SHORT_URL_ENABLED", cls.short_url_enabled,
            ),
            short_url_service=_str(
                "SHORT_URL_SERVICE", cls.short_url_service,
            ),
            verification_token_ttl_years=_int(
                "VERIFICATION_TOKEN_TTL_YEARS",
                cls.verification_token_ttl_years,
            ),
            hmac_truncation_length=_int(
                "HMAC_TRUNCATION_LENGTH",
                cls.hmac_truncation_length,
            ),
            # Anti-counterfeiting
            hmac_secret_key=_str(
                "HMAC_SECRET_KEY", cls.hmac_secret_key,
            ),
            key_rotation_days=_int(
                "KEY_ROTATION_DAYS", cls.key_rotation_days,
            ),
            enable_digital_watermark=_bool(
                "ENABLE_DIGITAL_WATERMARK",
                cls.enable_digital_watermark,
            ),
            scan_velocity_threshold=_int(
                "SCAN_VELOCITY_THRESHOLD",
                cls.scan_velocity_threshold,
            ),
            geo_fence_enabled=_bool(
                "GEO_FENCE_ENABLED", cls.geo_fence_enabled,
            ),
            # Bulk generation
            bulk_max_size=_int(
                "BULK_MAX_SIZE", cls.bulk_max_size,
            ),
            bulk_workers=_int("BULK_WORKERS", cls.bulk_workers),
            bulk_timeout_s=_int(
                "BULK_TIMEOUT_S", cls.bulk_timeout_s,
            ),
            bulk_output_format=_str(
                "BULK_OUTPUT_FORMAT", cls.bulk_output_format,
            ),
            enable_output_validation=_bool(
                "ENABLE_OUTPUT_VALIDATION",
                cls.enable_output_validation,
            ),
            # Lifecycle
            default_ttl_years=_int(
                "DEFAULT_TTL_YEARS", cls.default_ttl_years,
            ),
            scan_logging_enabled=_bool(
                "SCAN_LOGGING_ENABLED", cls.scan_logging_enabled,
            ),
            max_reprints=_int("MAX_REPRINTS", cls.max_reprints),
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
            "QRCodeGeneratorConfig loaded: "
            "version=%s, ec=%s, module=%d, quiet=%d, "
            "format=%s, dpi=%d, logo=%s, grade=%s, "
            "content=%s, max_payload=%d, compress=%s, "
            "threshold=%d, encrypt=%s, "
            "template=%s, font=%s/%dpt, "
            "prefix=%s, check=%s, pad=%d, "
            "verify_url=%s, short=%s, ttl=%dy, "
            "hmac_len=%d, rotation=%dd, "
            "watermark=%s, velocity=%d, geo=%s, "
            "bulk_max=%d, workers=%d, bulk_timeout=%ds, "
            "output_fmt=%s, validate=%s, "
            "ttl_years=%d, scan_log=%s, reprints=%d, "
            "batch_max=%d, concurrency=%d, timeout=%ds, "
            "retention=%dy, "
            "provenance=%s, pool=%d, rate_limit=%d/min, "
            "metrics=%s",
            config.default_version,
            config.default_error_correction,
            config.default_module_size,
            config.default_quiet_zone,
            config.default_output_format,
            config.default_dpi,
            config.enable_logo_embedding,
            config.quality_grade_target,
            config.default_content_type,
            config.max_payload_bytes,
            config.enable_compression,
            config.compression_threshold_bytes,
            config.enable_encryption,
            config.default_template,
            config.default_font,
            config.default_font_size,
            config.default_prefix_format,
            config.check_digit_algorithm,
            config.code_padding,
            config.base_verification_url,
            config.short_url_enabled,
            config.verification_token_ttl_years,
            config.hmac_truncation_length,
            config.key_rotation_days,
            config.enable_digital_watermark,
            config.scan_velocity_threshold,
            config.geo_fence_enabled,
            config.bulk_max_size,
            config.bulk_workers,
            config.bulk_timeout_s,
            config.bulk_output_format,
            config.enable_output_validation,
            config.default_ttl_years,
            config.scan_logging_enabled,
            config.max_reprints,
            config.batch_max_size,
            config.batch_concurrency,
            config.batch_timeout_s,
            config.retention_years,
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
    def qr_generation_settings(self) -> Dict[str, Any]:
        """Return QR generation settings as a dictionary.

        Returns:
            Dictionary with keys: default_version, default_error_correction,
            default_module_size, default_quiet_zone, default_output_format,
            default_dpi, enable_logo_embedding, quality_grade_target.
        """
        return {
            "default_version": self.default_version,
            "default_error_correction": self.default_error_correction,
            "default_module_size": self.default_module_size,
            "default_quiet_zone": self.default_quiet_zone,
            "default_output_format": self.default_output_format,
            "default_dpi": self.default_dpi,
            "enable_logo_embedding": self.enable_logo_embedding,
            "quality_grade_target": self.quality_grade_target,
        }

    @property
    def payload_settings(self) -> Dict[str, Any]:
        """Return payload settings as a dictionary.

        Returns:
            Dictionary with keys: default_content_type, max_payload_bytes,
            enable_compression, compression_threshold_bytes,
            enable_encryption, payload_version.
        """
        return {
            "default_content_type": self.default_content_type,
            "max_payload_bytes": self.max_payload_bytes,
            "enable_compression": self.enable_compression,
            "compression_threshold_bytes": (
                self.compression_threshold_bytes
            ),
            "enable_encryption": self.enable_encryption,
            "payload_version": self.payload_version,
        }

    @property
    def label_settings(self) -> Dict[str, Any]:
        """Return label rendering settings as a dictionary.

        Returns:
            Dictionary with keys: default_template, default_font,
            default_font_size, compliant_color_hex, pending_color_hex,
            non_compliant_color_hex, bleed_mm.
        """
        return {
            "default_template": self.default_template,
            "default_font": self.default_font,
            "default_font_size": self.default_font_size,
            "compliant_color_hex": self.compliant_color_hex,
            "pending_color_hex": self.pending_color_hex,
            "non_compliant_color_hex": self.non_compliant_color_hex,
            "bleed_mm": self.bleed_mm,
        }

    @property
    def batch_code_settings(self) -> Dict[str, Any]:
        """Return batch code settings as a dictionary.

        Returns:
            Dictionary with keys: default_prefix_format,
            check_digit_algorithm, code_padding, start_sequence.
        """
        return {
            "default_prefix_format": self.default_prefix_format,
            "check_digit_algorithm": self.check_digit_algorithm,
            "code_padding": self.code_padding,
            "start_sequence": self.start_sequence,
        }

    @property
    def verification_url_settings(self) -> Dict[str, Any]:
        """Return verification URL settings as a dictionary.

        Returns:
            Dictionary with keys: base_verification_url, short_url_enabled,
            short_url_service, verification_token_ttl_years,
            hmac_truncation_length.
        """
        return {
            "base_verification_url": self.base_verification_url,
            "short_url_enabled": self.short_url_enabled,
            "short_url_service": self.short_url_service,
            "verification_token_ttl_years": (
                self.verification_token_ttl_years
            ),
            "hmac_truncation_length": self.hmac_truncation_length,
        }

    @property
    def anti_counterfeiting_settings(self) -> Dict[str, Any]:
        """Return anti-counterfeiting settings as a dictionary.

        Returns:
            Dictionary with sensitive key redacted.
        """
        return {
            "hmac_secret_key": "***" if self.hmac_secret_key else "",
            "key_rotation_days": self.key_rotation_days,
            "enable_digital_watermark": self.enable_digital_watermark,
            "scan_velocity_threshold": self.scan_velocity_threshold,
            "geo_fence_enabled": self.geo_fence_enabled,
        }

    @property
    def bulk_generation_settings(self) -> Dict[str, Any]:
        """Return bulk generation settings as a dictionary.

        Returns:
            Dictionary with keys: bulk_max_size, bulk_workers,
            bulk_timeout_s, bulk_output_format, enable_output_validation.
        """
        return {
            "bulk_max_size": self.bulk_max_size,
            "bulk_workers": self.bulk_workers,
            "bulk_timeout_s": self.bulk_timeout_s,
            "bulk_output_format": self.bulk_output_format,
            "enable_output_validation": self.enable_output_validation,
        }

    @property
    def lifecycle_settings(self) -> Dict[str, Any]:
        """Return lifecycle settings as a dictionary.

        Returns:
            Dictionary with keys: default_ttl_years,
            scan_logging_enabled, max_reprints.
        """
        return {
            "default_ttl_years": self.default_ttl_years,
            "scan_logging_enabled": self.scan_logging_enabled,
            "max_reprints": self.max_reprints,
        }

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

        Sensitive connection strings (database_url, redis_url) and
        HMAC secret keys are redacted to prevent accidental credential
        leakage in logs, exception tracebacks, and monitoring dashboards.

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
            # QR generation
            "default_version": self.default_version,
            "default_error_correction": self.default_error_correction,
            "default_module_size": self.default_module_size,
            "default_quiet_zone": self.default_quiet_zone,
            "default_output_format": self.default_output_format,
            "default_dpi": self.default_dpi,
            "enable_logo_embedding": self.enable_logo_embedding,
            "quality_grade_target": self.quality_grade_target,
            # Payload
            "default_content_type": self.default_content_type,
            "max_payload_bytes": self.max_payload_bytes,
            "enable_compression": self.enable_compression,
            "compression_threshold_bytes": (
                self.compression_threshold_bytes
            ),
            "enable_encryption": self.enable_encryption,
            "payload_version": self.payload_version,
            # Label
            "default_template": self.default_template,
            "default_font": self.default_font,
            "default_font_size": self.default_font_size,
            "compliant_color_hex": self.compliant_color_hex,
            "pending_color_hex": self.pending_color_hex,
            "non_compliant_color_hex": self.non_compliant_color_hex,
            "bleed_mm": self.bleed_mm,
            # Batch codes
            "default_prefix_format": self.default_prefix_format,
            "check_digit_algorithm": self.check_digit_algorithm,
            "code_padding": self.code_padding,
            "start_sequence": self.start_sequence,
            # Verification URLs
            "base_verification_url": self.base_verification_url,
            "short_url_enabled": self.short_url_enabled,
            "short_url_service": (
                "***" if self.short_url_service else ""
            ),
            "verification_token_ttl_years": (
                self.verification_token_ttl_years
            ),
            "hmac_truncation_length": self.hmac_truncation_length,
            # Anti-counterfeiting (secrets redacted)
            "hmac_secret_key": "***" if self.hmac_secret_key else "",
            "key_rotation_days": self.key_rotation_days,
            "enable_digital_watermark": self.enable_digital_watermark,
            "scan_velocity_threshold": self.scan_velocity_threshold,
            "geo_fence_enabled": self.geo_fence_enabled,
            # Bulk generation
            "bulk_max_size": self.bulk_max_size,
            "bulk_workers": self.bulk_workers,
            "bulk_timeout_s": self.bulk_timeout_s,
            "bulk_output_format": self.bulk_output_format,
            "enable_output_validation": self.enable_output_validation,
            # Lifecycle
            "default_ttl_years": self.default_ttl_years,
            "scan_logging_enabled": self.scan_logging_enabled,
            "max_reprints": self.max_reprints,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Data retention
            "retention_years": self.retention_years,
            # EUDR commodities
            "eudr_commodities": list(self.eudr_commodities),
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
        return f"QRCodeGeneratorConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[QRCodeGeneratorConfig] = None
_config_lock = threading.Lock()


def get_config() -> QRCodeGeneratorConfig:
    """Return the singleton QRCodeGeneratorConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_QRG_*`` environment variables.

    Returns:
        QRCodeGeneratorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_error_correction
        'M'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = QRCodeGeneratorConfig.from_env()
    return _config_instance


def set_config(config: QRCodeGeneratorConfig) -> None:
    """Replace the singleton QRCodeGeneratorConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New QRCodeGeneratorConfig to install.

    Example:
        >>> cfg = QRCodeGeneratorConfig(default_error_correction="H")
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "QRCodeGeneratorConfig replaced programmatically: "
        "version=%s, ec=%s, format=%s",
        config.default_version,
        config.default_error_correction,
        config.default_output_format,
    )


def reset_config() -> None:
    """Reset the singleton QRCodeGeneratorConfig to None.

    The next call to get_config() will re-read GL_EUDR_QRG_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("QRCodeGeneratorConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "QRCodeGeneratorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
