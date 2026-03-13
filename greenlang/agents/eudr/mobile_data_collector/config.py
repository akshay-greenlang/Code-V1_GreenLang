# -*- coding: utf-8 -*-
"""
Mobile Data Collector Configuration - AGENT-EUDR-015

Centralized configuration for the Mobile Data Collector Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Offline form engine: max form size, local storage path, queue batch size,
  validation strictness, max fields per form, supported form types, form
  submission timeout, draft expiry days
- GPS capture: min accuracy meters, default CRS (WGS84/EPSG:4326), polygon
  min/max vertices, altitude capture, HDOP threshold, satellite count
  threshold, augmentation detection, coordinate decimal places, bounds
  checking, capture timeout
- Photo evidence: max photo size MB, compression quality levels,
  EXIF extraction, hash algorithm (SHA-256), supported formats
  (JPEG/PNG/HEIC), max photos per form, min resolution width/height,
  min file size bytes, timestamp deviation threshold seconds
- Sync: sync interval seconds, max retry count, retry backoff multiplier,
  delta compression enabled, bandwidth limit KB/s, conflict resolution
  strategy (server_wins/client_wins/manual), max upload size per sync MB,
  idempotency enabled, sync timeout, priority ordering
- Form templates: max templates, max fields per form, supported languages,
  conditional logic depth, template versioning, inheritance enabled,
  meta-validation enabled
- Digital signatures: algorithm (ECDSA P-256), timestamp binding,
  signature expiry days, revocation window hours, multi-signature
  enabled, visual signature capture
- Data packages: max package size MB, compression format (gzip),
  Merkle tree enabled, package TTL years (5 per EUDR Art 14),
  compression level, incremental build enabled, supported export
  formats (zip/tar_gz/json_ld)
- Device fleet: max devices, heartbeat interval seconds, offline
  threshold minutes, storage warning threshold percent, low battery
  threshold percent, agent version enforcement, decommission enabled
- Batch processing: batch size, concurrency, timeout
- Data retention: EUDR 5-year retention
- Provenance: genesis hash, SHA-256 chain
- Metrics: export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_MDC_`` prefix (e.g. ``GL_EUDR_MDC_DATABASE_URL``,
``GL_EUDR_MDC_MIN_ACCURACY_METERS``).

Environment Variable Reference (GL_EUDR_MDC_ prefix):
    GL_EUDR_MDC_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_MDC_REDIS_URL                     - Redis connection URL
    GL_EUDR_MDC_LOG_LEVEL                     - Logging level
    GL_EUDR_MDC_POOL_SIZE                     - Database pool size
    GL_EUDR_MDC_MAX_FORM_SIZE_KB              - Max form payload size in KB
    GL_EUDR_MDC_LOCAL_STORAGE_PATH            - Local SQLite storage path
    GL_EUDR_MDC_QUEUE_BATCH_SIZE              - Sync queue batch size
    GL_EUDR_MDC_VALIDATION_STRICTNESS         - Validation strictness (strict/lenient)
    GL_EUDR_MDC_MAX_FIELDS_PER_FORM           - Max fields per form template
    GL_EUDR_MDC_FORM_SUBMISSION_TIMEOUT_S     - Form submission timeout seconds
    GL_EUDR_MDC_DRAFT_EXPIRY_DAYS             - Draft form expiry in days
    GL_EUDR_MDC_MIN_ACCURACY_METERS           - Min GPS accuracy in meters
    GL_EUDR_MDC_DEFAULT_CRS                   - Default coordinate reference system
    GL_EUDR_MDC_DEFAULT_SRID                  - Default SRID (4326 for WGS84)
    GL_EUDR_MDC_POLYGON_MIN_VERTICES          - Min polygon vertices
    GL_EUDR_MDC_POLYGON_MAX_VERTICES          - Max polygon vertices
    GL_EUDR_MDC_ENABLE_ALTITUDE_CAPTURE       - Enable altitude capture
    GL_EUDR_MDC_HDOP_THRESHOLD                - Max HDOP threshold
    GL_EUDR_MDC_SATELLITE_COUNT_THRESHOLD     - Min satellite count
    GL_EUDR_MDC_COORDINATE_DECIMAL_PLACES     - Coordinate decimal places
    GL_EUDR_MDC_ENABLE_BOUNDS_CHECKING        - Enable geographic bounds checking
    GL_EUDR_MDC_GPS_CAPTURE_TIMEOUT_S         - GPS capture timeout seconds
    GL_EUDR_MDC_POLYGON_ACCURACY_METERS       - Max polygon vertex accuracy
    GL_EUDR_MDC_MIN_PLOT_AREA_HA              - Min plausible plot area hectares
    GL_EUDR_MDC_MAX_PHOTO_SIZE_MB             - Max photo size in MB
    GL_EUDR_MDC_COMPRESSION_QUALITY_HIGH      - High quality compression percent
    GL_EUDR_MDC_COMPRESSION_QUALITY_MEDIUM    - Medium quality compression percent
    GL_EUDR_MDC_COMPRESSION_QUALITY_LOW       - Low quality compression percent
    GL_EUDR_MDC_ENABLE_EXIF_EXTRACTION        - Enable EXIF metadata extraction
    GL_EUDR_MDC_PHOTO_HASH_ALGORITHM          - Photo integrity hash algorithm
    GL_EUDR_MDC_MAX_PHOTOS_PER_FORM           - Max photos per form
    GL_EUDR_MDC_MIN_PHOTO_WIDTH               - Min photo width pixels
    GL_EUDR_MDC_MIN_PHOTO_HEIGHT              - Min photo height pixels
    GL_EUDR_MDC_MIN_PHOTO_FILE_SIZE_BYTES     - Min photo file size bytes
    GL_EUDR_MDC_TIMESTAMP_DEVIATION_THRESHOLD_S - Photo timestamp deviation threshold
    GL_EUDR_MDC_SYNC_INTERVAL_S               - Sync check interval seconds
    GL_EUDR_MDC_MAX_RETRY_COUNT               - Max sync retry count
    GL_EUDR_MDC_RETRY_BACKOFF_MULTIPLIER      - Retry backoff multiplier
    GL_EUDR_MDC_ENABLE_DELTA_COMPRESSION      - Enable delta sync compression
    GL_EUDR_MDC_BANDWIDTH_LIMIT_KB_S          - Bandwidth limit KB/s (0=unlimited)
    GL_EUDR_MDC_CONFLICT_RESOLUTION_STRATEGY  - Conflict resolution strategy
    GL_EUDR_MDC_MAX_UPLOAD_SIZE_PER_SYNC_MB   - Max upload size per sync session MB
    GL_EUDR_MDC_ENABLE_IDEMPOTENCY            - Enable idempotency keys
    GL_EUDR_MDC_SYNC_TIMEOUT_S                - Sync session timeout seconds
    GL_EUDR_MDC_MAX_TEMPLATES                 - Max form templates
    GL_EUDR_MDC_CONDITIONAL_LOGIC_DEPTH       - Max conditional logic nesting depth
    GL_EUDR_MDC_ENABLE_TEMPLATE_VERSIONING    - Enable template versioning
    GL_EUDR_MDC_ENABLE_TEMPLATE_INHERITANCE   - Enable template inheritance
    GL_EUDR_MDC_ENABLE_META_VALIDATION        - Enable template meta-validation
    GL_EUDR_MDC_SIGNATURE_ALGORITHM           - Digital signature algorithm
    GL_EUDR_MDC_ENABLE_TIMESTAMP_BINDING      - Enable timestamp binding
    GL_EUDR_MDC_SIGNATURE_EXPIRY_DAYS         - Signature expiry in days
    GL_EUDR_MDC_REVOCATION_WINDOW_HOURS       - Signature revocation window hours
    GL_EUDR_MDC_ENABLE_MULTI_SIGNATURE        - Enable multi-signature workflows
    GL_EUDR_MDC_ENABLE_VISUAL_SIGNATURE       - Enable visual signature capture
    GL_EUDR_MDC_MAX_PACKAGE_SIZE_MB           - Max data package size MB
    GL_EUDR_MDC_PACKAGE_COMPRESSION_FORMAT    - Package compression format
    GL_EUDR_MDC_ENABLE_MERKLE_TREE            - Enable Merkle tree integrity
    GL_EUDR_MDC_PACKAGE_TTL_YEARS             - Package retention years
    GL_EUDR_MDC_PACKAGE_COMPRESSION_LEVEL     - Gzip compression level (1-9)
    GL_EUDR_MDC_ENABLE_INCREMENTAL_BUILD      - Enable incremental package building
    GL_EUDR_MDC_MAX_DEVICES                   - Max fleet devices
    GL_EUDR_MDC_HEARTBEAT_INTERVAL_S          - Device heartbeat interval seconds
    GL_EUDR_MDC_OFFLINE_THRESHOLD_MINUTES     - Offline detection threshold minutes
    GL_EUDR_MDC_STORAGE_WARNING_THRESHOLD_PCT - Storage warning threshold percent
    GL_EUDR_MDC_LOW_BATTERY_THRESHOLD_PCT     - Low battery threshold percent
    GL_EUDR_MDC_ENABLE_VERSION_ENFORCEMENT    - Enable agent version enforcement
    GL_EUDR_MDC_ENABLE_DECOMMISSION           - Enable device decommissioning
    GL_EUDR_MDC_BATCH_MAX_SIZE                - Batch processing max size
    GL_EUDR_MDC_BATCH_CONCURRENCY             - Batch concurrency workers
    GL_EUDR_MDC_BATCH_TIMEOUT_S               - Batch timeout seconds
    GL_EUDR_MDC_RETENTION_YEARS               - Data retention years
    GL_EUDR_MDC_ENABLE_PROVENANCE             - Enable provenance tracking
    GL_EUDR_MDC_GENESIS_HASH                  - Genesis hash anchor
    GL_EUDR_MDC_ENABLE_METRICS                - Enable Prometheus metrics
    GL_EUDR_MDC_RATE_LIMIT                    - Max requests per minute

Example:
    >>> from greenlang.agents.eudr.mobile_data_collector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.min_accuracy_meters, cfg.hdop_threshold)
    3.0 2.0

    >>> # Override for testing
    >>> from greenlang.agents.eudr.mobile_data_collector.config import (
    ...     set_config, reset_config, MobileDataCollectorConfig,
    ... )
    >>> set_config(MobileDataCollectorConfig(min_accuracy_meters=5.0))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
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

_ENV_PREFIX = "GL_EUDR_MDC_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid validation strictness levels
# ---------------------------------------------------------------------------

_VALID_VALIDATION_STRICTNESS = frozenset({"strict", "lenient"})

# ---------------------------------------------------------------------------
# Valid coordinate reference systems
# ---------------------------------------------------------------------------

_VALID_CRS = frozenset({"WGS84", "EPSG:4326"})

# ---------------------------------------------------------------------------
# Valid photo hash algorithms
# ---------------------------------------------------------------------------

_VALID_HASH_ALGORITHMS = frozenset({"sha256", "sha384", "sha512"})

# ---------------------------------------------------------------------------
# Valid supported photo formats
# ---------------------------------------------------------------------------

_VALID_PHOTO_FORMATS = frozenset({"jpeg", "png", "heic"})

# ---------------------------------------------------------------------------
# Valid conflict resolution strategies
# ---------------------------------------------------------------------------

_VALID_CONFLICT_STRATEGIES = frozenset({
    "server_wins",
    "client_wins",
    "manual",
})

# ---------------------------------------------------------------------------
# Valid signature algorithms
# ---------------------------------------------------------------------------

_VALID_SIGNATURE_ALGORITHMS = frozenset({"ecdsa_p256", "ecdsa_p384"})

# ---------------------------------------------------------------------------
# Valid package compression formats
# ---------------------------------------------------------------------------

_VALID_COMPRESSION_FORMATS = frozenset({"gzip", "zstd", "lz4"})

# ---------------------------------------------------------------------------
# Valid package export formats
# ---------------------------------------------------------------------------

_VALID_EXPORT_FORMATS = frozenset({"zip", "tar_gz", "json_ld"})

# ---------------------------------------------------------------------------
# Default EUDR commodities
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ---------------------------------------------------------------------------
# Default supported languages (24 EU + 20 local)
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORTED_LANGUAGES: List[str] = [
    # 24 EU official languages
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi",
    "fr", "de", "el", "hu", "ga", "it", "lv", "lt",
    "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv",
    # 20 local languages
    "sw", "tw", "fo", "yo", "ha", "am", "ln", "rw",
    "mg", "wo", "pt-BR", "es-419", "qu", "gn", "fr-GF",
    "id", "ms", "th", "vi", "km",
]

# ---------------------------------------------------------------------------
# Default supported form types per EUDR
# ---------------------------------------------------------------------------

_DEFAULT_FORM_TYPES: List[str] = [
    "producer_registration",
    "plot_survey",
    "harvest_log",
    "custody_transfer",
    "quality_inspection",
    "smallholder_declaration",
]


# ---------------------------------------------------------------------------
# MobileDataCollectorConfig
# ---------------------------------------------------------------------------


@dataclass
class MobileDataCollectorConfig:
    """Complete configuration for the EUDR Mobile Data Collector Agent.

    Attributes are grouped by concern: connections, logging, offline form
    engine, GPS capture, photo evidence, sync, form templates, digital
    signatures, data packages, device fleet, batch processing, data
    retention, provenance tracking, metrics, and rate limiting.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_MDC_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            form submissions, GPS captures, photos, and audit logs.
        redis_url: Redis connection URL for sync queue management,
            device heartbeat tracking, and conflict resolution caching.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        pool_size: PostgreSQL connection pool size.
        max_form_size_kb: Maximum form submission payload size in
            kilobytes. Default 1024 KB (1 MB).
        local_storage_path: Path for local SQLite storage on device.
        queue_batch_size: Number of items to batch in sync queue.
        validation_strictness: Validation mode (strict rejects all
            errors; lenient warns on non-critical issues).
        max_fields_per_form: Maximum number of fields allowed per
            form template definition.
        form_submission_timeout_s: Server-side form submission
            processing timeout in seconds.
        draft_expiry_days: Days before draft forms expire and are
            flagged for cleanup.
        min_accuracy_meters: Minimum required GPS horizontal accuracy
            in meters for point captures (default 3.0m CEP with
            WAAS/EGNOS augmentation).
        default_crs: Default coordinate reference system identifier.
        default_srid: Default Spatial Reference Identifier (4326 for
            WGS84 datum).
        polygon_min_vertices: Minimum vertices for a valid polygon
            trace (3 per geometry rules).
        polygon_max_vertices: Maximum vertices for a polygon trace
            (10000 for detailed boundaries).
        enable_altitude_capture: Enable altitude recording via
            barometric sensor or GPS-derived altitude.
        hdop_threshold: Maximum acceptable HDOP (Horizontal Dilution
            of Precision) for GPS captures.
        satellite_count_threshold: Minimum visible satellite count
            for acceptable GPS fix quality.
        coordinate_decimal_places: Number of decimal places for
            coordinate precision (6 = ~0.11m resolution).
        enable_bounds_checking: Enable geographic bounding box
            validation against country/commodity bounds.
        gps_capture_timeout_s: Timeout in seconds for GPS fix
            acquisition before reporting failure.
        polygon_accuracy_meters: Maximum acceptable accuracy per
            polygon vertex (default 5.0m).
        min_plot_area_ha: Minimum plausible plot area in hectares
            (default 0.01 ha).
        max_photo_size_mb: Maximum photo file size in megabytes.
        compression_quality_high: JPEG compression quality for
            high setting (0-100).
        compression_quality_medium: JPEG compression quality for
            medium setting (0-100).
        compression_quality_low: JPEG compression quality for
            low setting (0-100).
        enable_exif_extraction: Enable EXIF metadata extraction
            from captured photos.
        photo_hash_algorithm: Hash algorithm for photo integrity
            hashing (sha256, sha384, sha512).
        supported_photo_formats: List of accepted photo formats.
        max_photos_per_form: Maximum photos attachable to a single
            form submission.
        min_photo_width: Minimum photo width in pixels.
        min_photo_height: Minimum photo height in pixels.
        min_photo_file_size_bytes: Minimum photo file size to
            reject blank/corrupted captures.
        timestamp_deviation_threshold_s: Maximum acceptable
            deviation between photo EXIF timestamp and device
            system time in seconds.
        sync_interval_s: Interval in seconds between automatic
            sync checks.
        max_retry_count: Maximum retry attempts before marking
            a sync item as permanently_failed.
        retry_backoff_multiplier: Multiplier for exponential
            backoff between sync retries.
        enable_delta_compression: Enable delta sync (only changed
            fields transmitted).
        bandwidth_limit_kb_s: Upload bandwidth limit in KB/s
            (0 = unlimited).
        conflict_resolution_strategy: Default conflict resolution
            strategy for sync conflicts.
        max_upload_size_per_sync_mb: Maximum upload payload per
            sync session in megabytes.
        enable_idempotency: Enable UUID-based idempotency keys
            for exactly-once delivery.
        sync_timeout_s: Sync session timeout in seconds.
        max_templates: Maximum number of form templates.
        supported_languages: List of supported language codes
            for form template rendering.
        conditional_logic_depth: Maximum nesting depth for
            conditional logic in form templates.
        enable_template_versioning: Enable semantic versioning
            for form templates.
        enable_template_inheritance: Enable template inheritance
            for operator-specific extensions.
        enable_meta_validation: Enable JSON schema meta-validation
            of template definitions.
        signature_algorithm: Digital signature algorithm for
            custody transfer and declarations.
        enable_timestamp_binding: Include timestamp in signature
            payload for tamper detection.
        signature_expiry_days: Number of days before a digital
            signature expires.
        revocation_window_hours: Hours within which a signatory
            can revoke their signature.
        enable_multi_signature: Enable multi-signature workflows
            for custody transfers.
        enable_visual_signature: Enable handwritten SVG touch-path
            signature capture.
        max_package_size_mb: Maximum data package size in
            megabytes.
        package_compression_format: Compression algorithm for
            data packages.
        enable_merkle_tree: Enable SHA-256 Merkle tree for
            package integrity verification.
        package_ttl_years: Data package retention in years per
            EUDR Article 14.
        package_compression_level: Gzip compression level (1-9,
            default 6).
        enable_incremental_build: Enable incremental package
            building across multiple sessions.
        supported_export_formats: List of supported package
            export formats.
        max_devices: Maximum number of devices in the fleet.
        heartbeat_interval_s: Device heartbeat telemetry interval
            in seconds.
        offline_threshold_minutes: Minutes since last sync before
            a device is flagged as offline.
        storage_warning_threshold_pct: Storage used percentage
            that triggers a warning.
        low_battery_threshold_pct: Battery percentage below which
            a low-battery alert is generated.
        enable_version_enforcement: Flag devices running outdated
            agent versions.
        enable_decommission: Allow device decommissioning.
        batch_max_size: Maximum number of records in a single
            batch processing job.
        batch_concurrency: Maximum concurrent batch processing
            workers.
        batch_timeout_s: Timeout in seconds for a single batch
            job.
        retention_years: Data retention in years per EUDR
            Article 14.
        eudr_commodities: List of EUDR-regulated commodity types.
        form_types: List of supported EUDR form types.
        enable_provenance: Enable SHA-256 provenance chain
            tracking for all operations.
        genesis_hash: Genesis anchor string for the provenance
            chain, unique to the Mobile Data Collector agent.
        enable_metrics: Enable Prometheus metrics export under
            the ``gl_eudr_mdc_`` prefix.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10

    # -- Offline form engine settings ----------------------------------------
    max_form_size_kb: int = 1024
    local_storage_path: str = "/data/greenlang/mdc/local"
    queue_batch_size: int = 50
    validation_strictness: str = "strict"
    max_fields_per_form: int = 200
    form_submission_timeout_s: int = 30
    draft_expiry_days: int = 90

    # -- GPS capture settings ------------------------------------------------
    min_accuracy_meters: float = 3.0
    default_crs: str = "WGS84"
    default_srid: int = 4326
    polygon_min_vertices: int = 3
    polygon_max_vertices: int = 10000
    enable_altitude_capture: bool = True
    hdop_threshold: float = 2.0
    satellite_count_threshold: int = 6
    coordinate_decimal_places: int = 6
    enable_bounds_checking: bool = True
    gps_capture_timeout_s: int = 120
    polygon_accuracy_meters: float = 5.0
    min_plot_area_ha: float = 0.01

    # -- Photo evidence settings ---------------------------------------------
    max_photo_size_mb: int = 20
    compression_quality_high: int = 90
    compression_quality_medium: int = 75
    compression_quality_low: int = 50
    enable_exif_extraction: bool = True
    photo_hash_algorithm: str = "sha256"
    supported_photo_formats: List[str] = field(
        default_factory=lambda: ["jpeg", "png", "heic"]
    )
    max_photos_per_form: int = 20
    min_photo_width: int = 1280
    min_photo_height: int = 960
    min_photo_file_size_bytes: int = 102400
    timestamp_deviation_threshold_s: int = 60

    # -- Sync settings -------------------------------------------------------
    sync_interval_s: int = 60
    max_retry_count: int = 20
    retry_backoff_multiplier: float = 2.0
    enable_delta_compression: bool = True
    bandwidth_limit_kb_s: int = 0
    conflict_resolution_strategy: str = "server_wins"
    max_upload_size_per_sync_mb: int = 50
    enable_idempotency: bool = True
    sync_timeout_s: int = 300

    # -- Form template settings ----------------------------------------------
    max_templates: int = 100
    supported_languages: List[str] = field(
        default_factory=lambda: list(_DEFAULT_SUPPORTED_LANGUAGES)
    )
    conditional_logic_depth: int = 5
    enable_template_versioning: bool = True
    enable_template_inheritance: bool = True
    enable_meta_validation: bool = True

    # -- Digital signature settings ------------------------------------------
    signature_algorithm: str = "ecdsa_p256"
    enable_timestamp_binding: bool = True
    signature_expiry_days: int = 1825
    revocation_window_hours: int = 24
    enable_multi_signature: bool = True
    enable_visual_signature: bool = True

    # -- Data package settings -----------------------------------------------
    max_package_size_mb: int = 500
    package_compression_format: str = "gzip"
    enable_merkle_tree: bool = True
    package_ttl_years: int = 5
    package_compression_level: int = 6
    enable_incremental_build: bool = True
    supported_export_formats: List[str] = field(
        default_factory=lambda: ["zip", "tar_gz", "json_ld"]
    )

    # -- Device fleet settings -----------------------------------------------
    max_devices: int = 5000
    heartbeat_interval_s: int = 300
    offline_threshold_minutes: int = 2880
    storage_warning_threshold_pct: int = 80
    low_battery_threshold_pct: int = 20
    enable_version_enforcement: bool = True
    enable_decommission: bool = True

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

    # -- EUDR form types -----------------------------------------------------
    form_types: List[str] = field(
        default_factory=lambda: list(_DEFAULT_FORM_TYPES)
    )

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-MDC-015-MOBILE-DATA-COLLECTOR-GENESIS"

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

        # -- Offline form engine settings ------------------------------------
        if self.max_form_size_kb < 1:
            errors.append(
                f"max_form_size_kb must be >= 1, "
                f"got {self.max_form_size_kb}"
            )
        if self.max_form_size_kb > 10240:
            errors.append(
                f"max_form_size_kb must be <= 10240 (10 MB), "
                f"got {self.max_form_size_kb}"
            )

        if not self.local_storage_path:
            errors.append("local_storage_path must not be empty")

        if self.queue_batch_size < 1:
            errors.append(
                f"queue_batch_size must be >= 1, "
                f"got {self.queue_batch_size}"
            )
        if self.queue_batch_size > 1000:
            errors.append(
                f"queue_batch_size must be <= 1000, "
                f"got {self.queue_batch_size}"
            )

        normalised_strictness = self.validation_strictness.lower().strip()
        if normalised_strictness not in _VALID_VALIDATION_STRICTNESS:
            errors.append(
                f"validation_strictness must be one of "
                f"{sorted(_VALID_VALIDATION_STRICTNESS)}, "
                f"got '{self.validation_strictness}'"
            )
        else:
            self.validation_strictness = normalised_strictness

        if self.max_fields_per_form < 1:
            errors.append(
                f"max_fields_per_form must be >= 1, "
                f"got {self.max_fields_per_form}"
            )
        if self.max_fields_per_form > 1000:
            errors.append(
                f"max_fields_per_form must be <= 1000, "
                f"got {self.max_fields_per_form}"
            )

        if self.form_submission_timeout_s < 1:
            errors.append(
                f"form_submission_timeout_s must be >= 1, "
                f"got {self.form_submission_timeout_s}"
            )

        if self.draft_expiry_days < 1:
            errors.append(
                f"draft_expiry_days must be >= 1, "
                f"got {self.draft_expiry_days}"
            )

        # -- GPS capture settings --------------------------------------------
        if self.min_accuracy_meters <= 0:
            errors.append(
                f"min_accuracy_meters must be > 0, "
                f"got {self.min_accuracy_meters}"
            )
        if self.min_accuracy_meters > 100:
            errors.append(
                f"min_accuracy_meters must be <= 100, "
                f"got {self.min_accuracy_meters}"
            )

        normalised_crs = self.default_crs.upper().strip()
        if normalised_crs not in _VALID_CRS:
            errors.append(
                f"default_crs must be one of {sorted(_VALID_CRS)}, "
                f"got '{self.default_crs}'"
            )
        else:
            self.default_crs = normalised_crs

        if self.default_srid not in (4326, 3857):
            errors.append(
                f"default_srid must be 4326 or 3857, "
                f"got {self.default_srid}"
            )

        if self.polygon_min_vertices < 3:
            errors.append(
                f"polygon_min_vertices must be >= 3, "
                f"got {self.polygon_min_vertices}"
            )

        if self.polygon_max_vertices < self.polygon_min_vertices:
            errors.append(
                f"polygon_max_vertices must be >= polygon_min_vertices "
                f"({self.polygon_min_vertices}), "
                f"got {self.polygon_max_vertices}"
            )
        if self.polygon_max_vertices > 100000:
            errors.append(
                f"polygon_max_vertices must be <= 100000, "
                f"got {self.polygon_max_vertices}"
            )

        if self.hdop_threshold <= 0:
            errors.append(
                f"hdop_threshold must be > 0, "
                f"got {self.hdop_threshold}"
            )
        if self.hdop_threshold > 50:
            errors.append(
                f"hdop_threshold must be <= 50, "
                f"got {self.hdop_threshold}"
            )

        if self.satellite_count_threshold < 1:
            errors.append(
                f"satellite_count_threshold must be >= 1, "
                f"got {self.satellite_count_threshold}"
            )
        if self.satellite_count_threshold > 50:
            errors.append(
                f"satellite_count_threshold must be <= 50, "
                f"got {self.satellite_count_threshold}"
            )

        if self.coordinate_decimal_places < 1:
            errors.append(
                f"coordinate_decimal_places must be >= 1, "
                f"got {self.coordinate_decimal_places}"
            )
        if self.coordinate_decimal_places > 15:
            errors.append(
                f"coordinate_decimal_places must be <= 15, "
                f"got {self.coordinate_decimal_places}"
            )

        if self.gps_capture_timeout_s < 1:
            errors.append(
                f"gps_capture_timeout_s must be >= 1, "
                f"got {self.gps_capture_timeout_s}"
            )

        if self.polygon_accuracy_meters <= 0:
            errors.append(
                f"polygon_accuracy_meters must be > 0, "
                f"got {self.polygon_accuracy_meters}"
            )

        if self.min_plot_area_ha <= 0:
            errors.append(
                f"min_plot_area_ha must be > 0, "
                f"got {self.min_plot_area_ha}"
            )

        # -- Photo evidence settings -----------------------------------------
        if self.max_photo_size_mb < 1:
            errors.append(
                f"max_photo_size_mb must be >= 1, "
                f"got {self.max_photo_size_mb}"
            )
        if self.max_photo_size_mb > 100:
            errors.append(
                f"max_photo_size_mb must be <= 100, "
                f"got {self.max_photo_size_mb}"
            )

        if not (1 <= self.compression_quality_high <= 100):
            errors.append(
                f"compression_quality_high must be in [1, 100], "
                f"got {self.compression_quality_high}"
            )
        if not (1 <= self.compression_quality_medium <= 100):
            errors.append(
                f"compression_quality_medium must be in [1, 100], "
                f"got {self.compression_quality_medium}"
            )
        if not (1 <= self.compression_quality_low <= 100):
            errors.append(
                f"compression_quality_low must be in [1, 100], "
                f"got {self.compression_quality_low}"
            )

        if self.compression_quality_high <= self.compression_quality_medium:
            errors.append(
                f"compression_quality_high ({self.compression_quality_high}) "
                f"must be > compression_quality_medium "
                f"({self.compression_quality_medium})"
            )
        if self.compression_quality_medium <= self.compression_quality_low:
            errors.append(
                f"compression_quality_medium ({self.compression_quality_medium}) "
                f"must be > compression_quality_low "
                f"({self.compression_quality_low})"
            )

        normalised_hash = self.photo_hash_algorithm.lower().strip()
        if normalised_hash not in _VALID_HASH_ALGORITHMS:
            errors.append(
                f"photo_hash_algorithm must be one of "
                f"{sorted(_VALID_HASH_ALGORITHMS)}, "
                f"got '{self.photo_hash_algorithm}'"
            )
        else:
            self.photo_hash_algorithm = normalised_hash

        if not self.supported_photo_formats:
            errors.append("supported_photo_formats must not be empty")

        if self.max_photos_per_form < 1:
            errors.append(
                f"max_photos_per_form must be >= 1, "
                f"got {self.max_photos_per_form}"
            )
        if self.max_photos_per_form > 100:
            errors.append(
                f"max_photos_per_form must be <= 100, "
                f"got {self.max_photos_per_form}"
            )

        if self.min_photo_width < 100:
            errors.append(
                f"min_photo_width must be >= 100, "
                f"got {self.min_photo_width}"
            )

        if self.min_photo_height < 100:
            errors.append(
                f"min_photo_height must be >= 100, "
                f"got {self.min_photo_height}"
            )

        if self.min_photo_file_size_bytes < 0:
            errors.append(
                f"min_photo_file_size_bytes must be >= 0, "
                f"got {self.min_photo_file_size_bytes}"
            )

        if self.timestamp_deviation_threshold_s < 1:
            errors.append(
                f"timestamp_deviation_threshold_s must be >= 1, "
                f"got {self.timestamp_deviation_threshold_s}"
            )

        # -- Sync settings ---------------------------------------------------
        if self.sync_interval_s < 1:
            errors.append(
                f"sync_interval_s must be >= 1, "
                f"got {self.sync_interval_s}"
            )

        if self.max_retry_count < 0:
            errors.append(
                f"max_retry_count must be >= 0, "
                f"got {self.max_retry_count}"
            )
        if self.max_retry_count > 100:
            errors.append(
                f"max_retry_count must be <= 100, "
                f"got {self.max_retry_count}"
            )

        if self.retry_backoff_multiplier < 1.0:
            errors.append(
                f"retry_backoff_multiplier must be >= 1.0, "
                f"got {self.retry_backoff_multiplier}"
            )
        if self.retry_backoff_multiplier > 10.0:
            errors.append(
                f"retry_backoff_multiplier must be <= 10.0, "
                f"got {self.retry_backoff_multiplier}"
            )

        if self.bandwidth_limit_kb_s < 0:
            errors.append(
                f"bandwidth_limit_kb_s must be >= 0, "
                f"got {self.bandwidth_limit_kb_s}"
            )

        normalised_conflict = (
            self.conflict_resolution_strategy.lower().strip()
        )
        if normalised_conflict not in _VALID_CONFLICT_STRATEGIES:
            errors.append(
                f"conflict_resolution_strategy must be one of "
                f"{sorted(_VALID_CONFLICT_STRATEGIES)}, "
                f"got '{self.conflict_resolution_strategy}'"
            )
        else:
            self.conflict_resolution_strategy = normalised_conflict

        if self.max_upload_size_per_sync_mb < 1:
            errors.append(
                f"max_upload_size_per_sync_mb must be >= 1, "
                f"got {self.max_upload_size_per_sync_mb}"
            )

        if self.sync_timeout_s < 1:
            errors.append(
                f"sync_timeout_s must be >= 1, "
                f"got {self.sync_timeout_s}"
            )

        # -- Form template settings ------------------------------------------
        if self.max_templates < 1:
            errors.append(
                f"max_templates must be >= 1, "
                f"got {self.max_templates}"
            )

        if not self.supported_languages:
            errors.append("supported_languages must not be empty")

        if self.conditional_logic_depth < 0:
            errors.append(
                f"conditional_logic_depth must be >= 0, "
                f"got {self.conditional_logic_depth}"
            )
        if self.conditional_logic_depth > 20:
            errors.append(
                f"conditional_logic_depth must be <= 20, "
                f"got {self.conditional_logic_depth}"
            )

        # -- Digital signature settings --------------------------------------
        normalised_sig = self.signature_algorithm.lower().strip()
        if normalised_sig not in _VALID_SIGNATURE_ALGORITHMS:
            errors.append(
                f"signature_algorithm must be one of "
                f"{sorted(_VALID_SIGNATURE_ALGORITHMS)}, "
                f"got '{self.signature_algorithm}'"
            )
        else:
            self.signature_algorithm = normalised_sig

        if self.signature_expiry_days < 1:
            errors.append(
                f"signature_expiry_days must be >= 1, "
                f"got {self.signature_expiry_days}"
            )

        if self.revocation_window_hours < 0:
            errors.append(
                f"revocation_window_hours must be >= 0, "
                f"got {self.revocation_window_hours}"
            )
        if self.revocation_window_hours > 720:
            errors.append(
                f"revocation_window_hours must be <= 720 (30 days), "
                f"got {self.revocation_window_hours}"
            )

        # -- Data package settings -------------------------------------------
        if self.max_package_size_mb < 1:
            errors.append(
                f"max_package_size_mb must be >= 1, "
                f"got {self.max_package_size_mb}"
            )
        if self.max_package_size_mb > 10240:
            errors.append(
                f"max_package_size_mb must be <= 10240 (10 GB), "
                f"got {self.max_package_size_mb}"
            )

        normalised_comp = self.package_compression_format.lower().strip()
        if normalised_comp not in _VALID_COMPRESSION_FORMATS:
            errors.append(
                f"package_compression_format must be one of "
                f"{sorted(_VALID_COMPRESSION_FORMATS)}, "
                f"got '{self.package_compression_format}'"
            )
        else:
            self.package_compression_format = normalised_comp

        if self.package_ttl_years < 1:
            errors.append(
                f"package_ttl_years must be >= 1, "
                f"got {self.package_ttl_years}"
            )
        if self.package_ttl_years > 25:
            errors.append(
                f"package_ttl_years must be <= 25, "
                f"got {self.package_ttl_years}"
            )

        if not (1 <= self.package_compression_level <= 9):
            errors.append(
                f"package_compression_level must be in [1, 9], "
                f"got {self.package_compression_level}"
            )

        if not self.supported_export_formats:
            errors.append("supported_export_formats must not be empty")

        # -- Device fleet settings -------------------------------------------
        if self.max_devices < 1:
            errors.append(
                f"max_devices must be >= 1, "
                f"got {self.max_devices}"
            )
        if self.max_devices > 100000:
            errors.append(
                f"max_devices must be <= 100000, "
                f"got {self.max_devices}"
            )

        if self.heartbeat_interval_s < 1:
            errors.append(
                f"heartbeat_interval_s must be >= 1, "
                f"got {self.heartbeat_interval_s}"
            )

        if self.offline_threshold_minutes < 1:
            errors.append(
                f"offline_threshold_minutes must be >= 1, "
                f"got {self.offline_threshold_minutes}"
            )

        if not (1 <= self.storage_warning_threshold_pct <= 99):
            errors.append(
                f"storage_warning_threshold_pct must be in [1, 99], "
                f"got {self.storage_warning_threshold_pct}"
            )

        if not (1 <= self.low_battery_threshold_pct <= 99):
            errors.append(
                f"low_battery_threshold_pct must be in [1, 99], "
                f"got {self.low_battery_threshold_pct}"
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

        # -- EUDR form types -------------------------------------------------
        if not self.form_types:
            errors.append("form_types must not be empty")

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Rate limiting ---------------------------------------------------
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")

        if errors:
            raise ValueError(
                "MobileDataCollectorConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "MobileDataCollectorConfig validated successfully: "
            "accuracy=%sm, hdop=%s, sats=%d, crs=%s, "
            "polygon_verts=[%d,%d], altitude=%s, "
            "photo_max=%dMB, quality=%d/%d/%d, "
            "sync_interval=%ds, retries=%d, backoff=%.1f, "
            "conflict=%s, delta=%s, "
            "templates=%d, languages=%d, logic_depth=%d, "
            "sig_algo=%s, sig_expiry=%dd, revoke=%dh, "
            "pkg_max=%dMB, compression=%s/%d, merkle=%s, "
            "pkg_ttl=%dy, "
            "devices=%d, heartbeat=%ds, offline=%dm, "
            "batch_max=%d, concurrency=%d, retention=%dy, "
            "provenance=%s, metrics=%s",
            self.min_accuracy_meters,
            self.hdop_threshold,
            self.satellite_count_threshold,
            self.default_crs,
            self.polygon_min_vertices,
            self.polygon_max_vertices,
            self.enable_altitude_capture,
            self.max_photo_size_mb,
            self.compression_quality_high,
            self.compression_quality_medium,
            self.compression_quality_low,
            self.sync_interval_s,
            self.max_retry_count,
            self.retry_backoff_multiplier,
            self.conflict_resolution_strategy,
            self.enable_delta_compression,
            self.max_templates,
            len(self.supported_languages),
            self.conditional_logic_depth,
            self.signature_algorithm,
            self.signature_expiry_days,
            self.revocation_window_hours,
            self.max_package_size_mb,
            self.package_compression_format,
            self.package_compression_level,
            self.enable_merkle_tree,
            self.package_ttl_years,
            self.max_devices,
            self.heartbeat_interval_s,
            self.offline_threshold_minutes,
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
    def from_env(cls) -> MobileDataCollectorConfig:
        """Build a MobileDataCollectorConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_MDC_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated MobileDataCollectorConfig instance, validated via
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
                    "Invalid float for %s%s=%r, using default %s",
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
            # Offline form engine
            max_form_size_kb=_int(
                "MAX_FORM_SIZE_KB", cls.max_form_size_kb,
            ),
            local_storage_path=_str(
                "LOCAL_STORAGE_PATH", cls.local_storage_path,
            ),
            queue_batch_size=_int(
                "QUEUE_BATCH_SIZE", cls.queue_batch_size,
            ),
            validation_strictness=_str(
                "VALIDATION_STRICTNESS", cls.validation_strictness,
            ),
            max_fields_per_form=_int(
                "MAX_FIELDS_PER_FORM", cls.max_fields_per_form,
            ),
            form_submission_timeout_s=_int(
                "FORM_SUBMISSION_TIMEOUT_S",
                cls.form_submission_timeout_s,
            ),
            draft_expiry_days=_int(
                "DRAFT_EXPIRY_DAYS", cls.draft_expiry_days,
            ),
            # GPS capture
            min_accuracy_meters=_float(
                "MIN_ACCURACY_METERS", cls.min_accuracy_meters,
            ),
            default_crs=_str("DEFAULT_CRS", cls.default_crs),
            default_srid=_int("DEFAULT_SRID", cls.default_srid),
            polygon_min_vertices=_int(
                "POLYGON_MIN_VERTICES", cls.polygon_min_vertices,
            ),
            polygon_max_vertices=_int(
                "POLYGON_MAX_VERTICES", cls.polygon_max_vertices,
            ),
            enable_altitude_capture=_bool(
                "ENABLE_ALTITUDE_CAPTURE",
                cls.enable_altitude_capture,
            ),
            hdop_threshold=_float(
                "HDOP_THRESHOLD", cls.hdop_threshold,
            ),
            satellite_count_threshold=_int(
                "SATELLITE_COUNT_THRESHOLD",
                cls.satellite_count_threshold,
            ),
            coordinate_decimal_places=_int(
                "COORDINATE_DECIMAL_PLACES",
                cls.coordinate_decimal_places,
            ),
            enable_bounds_checking=_bool(
                "ENABLE_BOUNDS_CHECKING",
                cls.enable_bounds_checking,
            ),
            gps_capture_timeout_s=_int(
                "GPS_CAPTURE_TIMEOUT_S",
                cls.gps_capture_timeout_s,
            ),
            polygon_accuracy_meters=_float(
                "POLYGON_ACCURACY_METERS",
                cls.polygon_accuracy_meters,
            ),
            min_plot_area_ha=_float(
                "MIN_PLOT_AREA_HA", cls.min_plot_area_ha,
            ),
            # Photo evidence
            max_photo_size_mb=_int(
                "MAX_PHOTO_SIZE_MB", cls.max_photo_size_mb,
            ),
            compression_quality_high=_int(
                "COMPRESSION_QUALITY_HIGH",
                cls.compression_quality_high,
            ),
            compression_quality_medium=_int(
                "COMPRESSION_QUALITY_MEDIUM",
                cls.compression_quality_medium,
            ),
            compression_quality_low=_int(
                "COMPRESSION_QUALITY_LOW",
                cls.compression_quality_low,
            ),
            enable_exif_extraction=_bool(
                "ENABLE_EXIF_EXTRACTION",
                cls.enable_exif_extraction,
            ),
            photo_hash_algorithm=_str(
                "PHOTO_HASH_ALGORITHM",
                cls.photo_hash_algorithm,
            ),
            max_photos_per_form=_int(
                "MAX_PHOTOS_PER_FORM", cls.max_photos_per_form,
            ),
            min_photo_width=_int(
                "MIN_PHOTO_WIDTH", cls.min_photo_width,
            ),
            min_photo_height=_int(
                "MIN_PHOTO_HEIGHT", cls.min_photo_height,
            ),
            min_photo_file_size_bytes=_int(
                "MIN_PHOTO_FILE_SIZE_BYTES",
                cls.min_photo_file_size_bytes,
            ),
            timestamp_deviation_threshold_s=_int(
                "TIMESTAMP_DEVIATION_THRESHOLD_S",
                cls.timestamp_deviation_threshold_s,
            ),
            # Sync
            sync_interval_s=_int(
                "SYNC_INTERVAL_S", cls.sync_interval_s,
            ),
            max_retry_count=_int(
                "MAX_RETRY_COUNT", cls.max_retry_count,
            ),
            retry_backoff_multiplier=_float(
                "RETRY_BACKOFF_MULTIPLIER",
                cls.retry_backoff_multiplier,
            ),
            enable_delta_compression=_bool(
                "ENABLE_DELTA_COMPRESSION",
                cls.enable_delta_compression,
            ),
            bandwidth_limit_kb_s=_int(
                "BANDWIDTH_LIMIT_KB_S",
                cls.bandwidth_limit_kb_s,
            ),
            conflict_resolution_strategy=_str(
                "CONFLICT_RESOLUTION_STRATEGY",
                cls.conflict_resolution_strategy,
            ),
            max_upload_size_per_sync_mb=_int(
                "MAX_UPLOAD_SIZE_PER_SYNC_MB",
                cls.max_upload_size_per_sync_mb,
            ),
            enable_idempotency=_bool(
                "ENABLE_IDEMPOTENCY", cls.enable_idempotency,
            ),
            sync_timeout_s=_int(
                "SYNC_TIMEOUT_S", cls.sync_timeout_s,
            ),
            # Form templates
            max_templates=_int(
                "MAX_TEMPLATES", cls.max_templates,
            ),
            conditional_logic_depth=_int(
                "CONDITIONAL_LOGIC_DEPTH",
                cls.conditional_logic_depth,
            ),
            enable_template_versioning=_bool(
                "ENABLE_TEMPLATE_VERSIONING",
                cls.enable_template_versioning,
            ),
            enable_template_inheritance=_bool(
                "ENABLE_TEMPLATE_INHERITANCE",
                cls.enable_template_inheritance,
            ),
            enable_meta_validation=_bool(
                "ENABLE_META_VALIDATION",
                cls.enable_meta_validation,
            ),
            # Digital signatures
            signature_algorithm=_str(
                "SIGNATURE_ALGORITHM", cls.signature_algorithm,
            ),
            enable_timestamp_binding=_bool(
                "ENABLE_TIMESTAMP_BINDING",
                cls.enable_timestamp_binding,
            ),
            signature_expiry_days=_int(
                "SIGNATURE_EXPIRY_DAYS",
                cls.signature_expiry_days,
            ),
            revocation_window_hours=_int(
                "REVOCATION_WINDOW_HOURS",
                cls.revocation_window_hours,
            ),
            enable_multi_signature=_bool(
                "ENABLE_MULTI_SIGNATURE",
                cls.enable_multi_signature,
            ),
            enable_visual_signature=_bool(
                "ENABLE_VISUAL_SIGNATURE",
                cls.enable_visual_signature,
            ),
            # Data packages
            max_package_size_mb=_int(
                "MAX_PACKAGE_SIZE_MB", cls.max_package_size_mb,
            ),
            package_compression_format=_str(
                "PACKAGE_COMPRESSION_FORMAT",
                cls.package_compression_format,
            ),
            enable_merkle_tree=_bool(
                "ENABLE_MERKLE_TREE", cls.enable_merkle_tree,
            ),
            package_ttl_years=_int(
                "PACKAGE_TTL_YEARS", cls.package_ttl_years,
            ),
            package_compression_level=_int(
                "PACKAGE_COMPRESSION_LEVEL",
                cls.package_compression_level,
            ),
            enable_incremental_build=_bool(
                "ENABLE_INCREMENTAL_BUILD",
                cls.enable_incremental_build,
            ),
            # Device fleet
            max_devices=_int("MAX_DEVICES", cls.max_devices),
            heartbeat_interval_s=_int(
                "HEARTBEAT_INTERVAL_S", cls.heartbeat_interval_s,
            ),
            offline_threshold_minutes=_int(
                "OFFLINE_THRESHOLD_MINUTES",
                cls.offline_threshold_minutes,
            ),
            storage_warning_threshold_pct=_int(
                "STORAGE_WARNING_THRESHOLD_PCT",
                cls.storage_warning_threshold_pct,
            ),
            low_battery_threshold_pct=_int(
                "LOW_BATTERY_THRESHOLD_PCT",
                cls.low_battery_threshold_pct,
            ),
            enable_version_enforcement=_bool(
                "ENABLE_VERSION_ENFORCEMENT",
                cls.enable_version_enforcement,
            ),
            enable_decommission=_bool(
                "ENABLE_DECOMMISSION", cls.enable_decommission,
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
            "MobileDataCollectorConfig loaded: "
            "accuracy=%sm, hdop=%s, sats=%d, crs=%s, "
            "polygon=[%d,%d], altitude=%s, bounds=%s, "
            "photo_max=%dMB, quality=%d/%d/%d, exif=%s, "
            "sync=%ds, retries=%d, backoff=%.1f, "
            "conflict=%s, delta=%s, bandwidth=%d, "
            "templates=%d, langs=%d, logic=%d, "
            "sig=%s, sig_expiry=%dd, revoke=%dh, "
            "multi_sig=%s, visual_sig=%s, "
            "pkg=%dMB, compress=%s/%d, merkle=%s, "
            "pkg_ttl=%dy, incremental=%s, "
            "devices=%d, heartbeat=%ds, offline=%dm, "
            "storage_warn=%d%%, battery_warn=%d%%, "
            "batch=%d, concurrency=%d, timeout=%ds, "
            "retention=%dy, pool=%d, rate=%d/min, "
            "provenance=%s, metrics=%s",
            config.min_accuracy_meters,
            config.hdop_threshold,
            config.satellite_count_threshold,
            config.default_crs,
            config.polygon_min_vertices,
            config.polygon_max_vertices,
            config.enable_altitude_capture,
            config.enable_bounds_checking,
            config.max_photo_size_mb,
            config.compression_quality_high,
            config.compression_quality_medium,
            config.compression_quality_low,
            config.enable_exif_extraction,
            config.sync_interval_s,
            config.max_retry_count,
            config.retry_backoff_multiplier,
            config.conflict_resolution_strategy,
            config.enable_delta_compression,
            config.bandwidth_limit_kb_s,
            config.max_templates,
            len(config.supported_languages),
            config.conditional_logic_depth,
            config.signature_algorithm,
            config.signature_expiry_days,
            config.revocation_window_hours,
            config.enable_multi_signature,
            config.enable_visual_signature,
            config.max_package_size_mb,
            config.package_compression_format,
            config.package_compression_level,
            config.enable_merkle_tree,
            config.package_ttl_years,
            config.enable_incremental_build,
            config.max_devices,
            config.heartbeat_interval_s,
            config.offline_threshold_minutes,
            config.storage_warning_threshold_pct,
            config.low_battery_threshold_pct,
            config.batch_max_size,
            config.batch_concurrency,
            config.batch_timeout_s,
            config.retention_years,
            config.pool_size,
            config.rate_limit,
            config.enable_provenance,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def offline_form_settings(self) -> Dict[str, Any]:
        """Return offline form engine settings as a dictionary.

        Returns:
            Dictionary with offline form configuration keys.
        """
        return {
            "max_form_size_kb": self.max_form_size_kb,
            "local_storage_path": self.local_storage_path,
            "queue_batch_size": self.queue_batch_size,
            "validation_strictness": self.validation_strictness,
            "max_fields_per_form": self.max_fields_per_form,
            "form_submission_timeout_s": self.form_submission_timeout_s,
            "draft_expiry_days": self.draft_expiry_days,
        }

    @property
    def gps_capture_settings(self) -> Dict[str, Any]:
        """Return GPS capture settings as a dictionary.

        Returns:
            Dictionary with GPS capture configuration keys.
        """
        return {
            "min_accuracy_meters": self.min_accuracy_meters,
            "default_crs": self.default_crs,
            "default_srid": self.default_srid,
            "polygon_min_vertices": self.polygon_min_vertices,
            "polygon_max_vertices": self.polygon_max_vertices,
            "enable_altitude_capture": self.enable_altitude_capture,
            "hdop_threshold": self.hdop_threshold,
            "satellite_count_threshold": self.satellite_count_threshold,
            "coordinate_decimal_places": self.coordinate_decimal_places,
            "enable_bounds_checking": self.enable_bounds_checking,
            "gps_capture_timeout_s": self.gps_capture_timeout_s,
            "polygon_accuracy_meters": self.polygon_accuracy_meters,
            "min_plot_area_ha": self.min_plot_area_ha,
        }

    @property
    def photo_evidence_settings(self) -> Dict[str, Any]:
        """Return photo evidence settings as a dictionary.

        Returns:
            Dictionary with photo evidence configuration keys.
        """
        return {
            "max_photo_size_mb": self.max_photo_size_mb,
            "compression_quality_high": self.compression_quality_high,
            "compression_quality_medium": self.compression_quality_medium,
            "compression_quality_low": self.compression_quality_low,
            "enable_exif_extraction": self.enable_exif_extraction,
            "photo_hash_algorithm": self.photo_hash_algorithm,
            "supported_photo_formats": list(self.supported_photo_formats),
            "max_photos_per_form": self.max_photos_per_form,
            "min_photo_width": self.min_photo_width,
            "min_photo_height": self.min_photo_height,
            "min_photo_file_size_bytes": self.min_photo_file_size_bytes,
            "timestamp_deviation_threshold_s": (
                self.timestamp_deviation_threshold_s
            ),
        }

    @property
    def sync_settings(self) -> Dict[str, Any]:
        """Return sync settings as a dictionary.

        Returns:
            Dictionary with sync configuration keys.
        """
        return {
            "sync_interval_s": self.sync_interval_s,
            "max_retry_count": self.max_retry_count,
            "retry_backoff_multiplier": self.retry_backoff_multiplier,
            "enable_delta_compression": self.enable_delta_compression,
            "bandwidth_limit_kb_s": self.bandwidth_limit_kb_s,
            "conflict_resolution_strategy": (
                self.conflict_resolution_strategy
            ),
            "max_upload_size_per_sync_mb": (
                self.max_upload_size_per_sync_mb
            ),
            "enable_idempotency": self.enable_idempotency,
            "sync_timeout_s": self.sync_timeout_s,
        }

    @property
    def form_template_settings(self) -> Dict[str, Any]:
        """Return form template settings as a dictionary.

        Returns:
            Dictionary with form template configuration keys.
        """
        return {
            "max_templates": self.max_templates,
            "supported_languages": list(self.supported_languages),
            "conditional_logic_depth": self.conditional_logic_depth,
            "enable_template_versioning": self.enable_template_versioning,
            "enable_template_inheritance": (
                self.enable_template_inheritance
            ),
            "enable_meta_validation": self.enable_meta_validation,
        }

    @property
    def digital_signature_settings(self) -> Dict[str, Any]:
        """Return digital signature settings as a dictionary.

        Returns:
            Dictionary with digital signature configuration keys.
        """
        return {
            "signature_algorithm": self.signature_algorithm,
            "enable_timestamp_binding": self.enable_timestamp_binding,
            "signature_expiry_days": self.signature_expiry_days,
            "revocation_window_hours": self.revocation_window_hours,
            "enable_multi_signature": self.enable_multi_signature,
            "enable_visual_signature": self.enable_visual_signature,
        }

    @property
    def data_package_settings(self) -> Dict[str, Any]:
        """Return data package settings as a dictionary.

        Returns:
            Dictionary with data package configuration keys.
        """
        return {
            "max_package_size_mb": self.max_package_size_mb,
            "package_compression_format": (
                self.package_compression_format
            ),
            "enable_merkle_tree": self.enable_merkle_tree,
            "package_ttl_years": self.package_ttl_years,
            "package_compression_level": self.package_compression_level,
            "enable_incremental_build": self.enable_incremental_build,
            "supported_export_formats": list(
                self.supported_export_formats
            ),
        }

    @property
    def device_fleet_settings(self) -> Dict[str, Any]:
        """Return device fleet settings as a dictionary.

        Returns:
            Dictionary with device fleet configuration keys.
        """
        return {
            "max_devices": self.max_devices,
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "offline_threshold_minutes": self.offline_threshold_minutes,
            "storage_warning_threshold_pct": (
                self.storage_warning_threshold_pct
            ),
            "low_battery_threshold_pct": self.low_battery_threshold_pct,
            "enable_version_enforcement": (
                self.enable_version_enforcement
            ),
            "enable_decommission": self.enable_decommission,
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
            # Offline form engine
            "max_form_size_kb": self.max_form_size_kb,
            "local_storage_path": self.local_storage_path,
            "queue_batch_size": self.queue_batch_size,
            "validation_strictness": self.validation_strictness,
            "max_fields_per_form": self.max_fields_per_form,
            "form_submission_timeout_s": self.form_submission_timeout_s,
            "draft_expiry_days": self.draft_expiry_days,
            # GPS capture
            "min_accuracy_meters": self.min_accuracy_meters,
            "default_crs": self.default_crs,
            "default_srid": self.default_srid,
            "polygon_min_vertices": self.polygon_min_vertices,
            "polygon_max_vertices": self.polygon_max_vertices,
            "enable_altitude_capture": self.enable_altitude_capture,
            "hdop_threshold": self.hdop_threshold,
            "satellite_count_threshold": self.satellite_count_threshold,
            "coordinate_decimal_places": self.coordinate_decimal_places,
            "enable_bounds_checking": self.enable_bounds_checking,
            "gps_capture_timeout_s": self.gps_capture_timeout_s,
            "polygon_accuracy_meters": self.polygon_accuracy_meters,
            "min_plot_area_ha": self.min_plot_area_ha,
            # Photo evidence
            "max_photo_size_mb": self.max_photo_size_mb,
            "compression_quality_high": self.compression_quality_high,
            "compression_quality_medium": self.compression_quality_medium,
            "compression_quality_low": self.compression_quality_low,
            "enable_exif_extraction": self.enable_exif_extraction,
            "photo_hash_algorithm": self.photo_hash_algorithm,
            "supported_photo_formats": list(self.supported_photo_formats),
            "max_photos_per_form": self.max_photos_per_form,
            "min_photo_width": self.min_photo_width,
            "min_photo_height": self.min_photo_height,
            "min_photo_file_size_bytes": self.min_photo_file_size_bytes,
            "timestamp_deviation_threshold_s": (
                self.timestamp_deviation_threshold_s
            ),
            # Sync
            "sync_interval_s": self.sync_interval_s,
            "max_retry_count": self.max_retry_count,
            "retry_backoff_multiplier": self.retry_backoff_multiplier,
            "enable_delta_compression": self.enable_delta_compression,
            "bandwidth_limit_kb_s": self.bandwidth_limit_kb_s,
            "conflict_resolution_strategy": (
                self.conflict_resolution_strategy
            ),
            "max_upload_size_per_sync_mb": (
                self.max_upload_size_per_sync_mb
            ),
            "enable_idempotency": self.enable_idempotency,
            "sync_timeout_s": self.sync_timeout_s,
            # Form templates
            "max_templates": self.max_templates,
            "supported_languages": list(self.supported_languages),
            "conditional_logic_depth": self.conditional_logic_depth,
            "enable_template_versioning": self.enable_template_versioning,
            "enable_template_inheritance": (
                self.enable_template_inheritance
            ),
            "enable_meta_validation": self.enable_meta_validation,
            # Digital signatures
            "signature_algorithm": self.signature_algorithm,
            "enable_timestamp_binding": self.enable_timestamp_binding,
            "signature_expiry_days": self.signature_expiry_days,
            "revocation_window_hours": self.revocation_window_hours,
            "enable_multi_signature": self.enable_multi_signature,
            "enable_visual_signature": self.enable_visual_signature,
            # Data packages
            "max_package_size_mb": self.max_package_size_mb,
            "package_compression_format": (
                self.package_compression_format
            ),
            "enable_merkle_tree": self.enable_merkle_tree,
            "package_ttl_years": self.package_ttl_years,
            "package_compression_level": self.package_compression_level,
            "enable_incremental_build": self.enable_incremental_build,
            "supported_export_formats": list(
                self.supported_export_formats
            ),
            # Device fleet
            "max_devices": self.max_devices,
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "offline_threshold_minutes": self.offline_threshold_minutes,
            "storage_warning_threshold_pct": (
                self.storage_warning_threshold_pct
            ),
            "low_battery_threshold_pct": self.low_battery_threshold_pct,
            "enable_version_enforcement": (
                self.enable_version_enforcement
            ),
            "enable_decommission": self.enable_decommission,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Data retention
            "retention_years": self.retention_years,
            # EUDR commodities
            "eudr_commodities": list(self.eudr_commodities),
            # EUDR form types
            "form_types": list(self.form_types),
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
        return f"MobileDataCollectorConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[MobileDataCollectorConfig] = None
_config_lock = threading.Lock()


def get_config() -> MobileDataCollectorConfig:
    """Return the singleton MobileDataCollectorConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_MDC_*`` environment variables.

    Returns:
        MobileDataCollectorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.min_accuracy_meters
        3.0
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = MobileDataCollectorConfig.from_env()
    return _config_instance


def set_config(config: MobileDataCollectorConfig) -> None:
    """Replace the singleton MobileDataCollectorConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New MobileDataCollectorConfig to install.

    Example:
        >>> cfg = MobileDataCollectorConfig(min_accuracy_meters=5.0)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "MobileDataCollectorConfig replaced programmatically: "
        "accuracy=%sm, hdop=%s, conflict=%s",
        config.min_accuracy_meters,
        config.hdop_threshold,
        config.conflict_resolution_strategy,
    )


def reset_config() -> None:
    """Reset the singleton MobileDataCollectorConfig to None.

    The next call to get_config() will re-read GL_EUDR_MDC_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("MobileDataCollectorConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "MobileDataCollectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
