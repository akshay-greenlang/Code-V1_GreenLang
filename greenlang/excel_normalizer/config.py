# -*- coding: utf-8 -*-
"""
Excel & CSV Normalizer Service Configuration - AGENT-DATA-002: Excel Normalizer

Centralized configuration for the Excel & CSV Normalizer SDK covering:
- Parser settings (file size, row/sheet/column limits)
- CSV settings (encoding, delimiter, detection)
- Column mapping strategy and thresholds
- Data type detection options
- Quality scoring weights and thresholds
- Batch processing defaults (max files, worker count)
- Connection pool sizing
- Processing timeouts

All settings can be overridden via environment variables with the
``GL_EXCEL_NORMALIZER_`` prefix (e.g. ``GL_EXCEL_NORMALIZER_MAX_FILE_SIZE_MB``).

Example:
    >>> from greenlang.excel_normalizer.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_file_size_mb, cfg.default_delimiter)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel & CSV Normalizer
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EXCEL_NORMALIZER_"


# ---------------------------------------------------------------------------
# ExcelNormalizerConfig
# ---------------------------------------------------------------------------


@dataclass
class ExcelNormalizerConfig:
    """Complete configuration for the GreenLang Excel & CSV Normalizer SDK.

    Attributes are grouped by concern: connections, parser settings,
    CSV settings, column mapping, type detection, quality scoring,
    batch processing, timeouts, pool sizing, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_EXCEL_NORMALIZER_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for file storage.
        max_file_size_mb: Maximum file size in megabytes for ingestion.
        max_rows_per_sheet: Maximum number of rows per sheet to process.
        max_sheets_per_workbook: Maximum number of sheets per workbook.
        max_columns: Maximum number of columns per sheet.
        default_encoding: Default character encoding for CSV files.
        default_delimiter: Default column delimiter for CSV files.
        enable_encoding_detection: Whether to auto-detect file encoding.
        sample_rows_for_detection: Number of rows sampled for delimiter detection.
        default_mapping_strategy: Default column mapping strategy.
        fuzzy_match_threshold: Minimum score for fuzzy column matching.
        enable_synonym_matching: Whether to use synonym-based column matching.
        sample_rows_for_type_detection: Rows sampled for data type detection.
        date_formats: Comma-separated date format identifiers to attempt.
        enable_currency_detection: Whether to detect currency values.
        min_quality_score: Minimum acceptable overall quality score.
        completeness_weight: Weight of completeness in quality score.
        accuracy_weight: Weight of accuracy in quality score.
        consistency_weight: Weight of consistency in quality score.
        batch_max_files: Maximum files allowed in a single batch job.
        batch_worker_count: Number of parallel workers for batch processing.
        processing_timeout_seconds: Timeout in seconds for a single file.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the Excel normalizer service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- Parser settings -----------------------------------------------------
    max_file_size_mb: int = 50
    max_rows_per_sheet: int = 1_000_000
    max_sheets_per_workbook: int = 50
    max_columns: int = 500

    # -- CSV settings --------------------------------------------------------
    default_encoding: str = "utf-8"
    default_delimiter: str = ","
    enable_encoding_detection: bool = True
    sample_rows_for_detection: int = 100

    # -- Column mapping ------------------------------------------------------
    default_mapping_strategy: str = "fuzzy"
    fuzzy_match_threshold: float = 0.75
    enable_synonym_matching: bool = True

    # -- Type detection ------------------------------------------------------
    sample_rows_for_type_detection: int = 1000
    date_formats: str = "ISO,US,EU"
    enable_currency_detection: bool = True

    # -- Quality scoring -----------------------------------------------------
    min_quality_score: float = 0.5
    completeness_weight: float = 0.4
    accuracy_weight: float = 0.35
    consistency_weight: float = 0.25

    # -- Batch processing ----------------------------------------------------
    batch_max_files: int = 100
    batch_worker_count: int = 4

    # -- Timeouts ------------------------------------------------------------
    processing_timeout_seconds: int = 300

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ExcelNormalizerConfig:
        """Build an ExcelNormalizerConfig from environment variables.

        Every field can be overridden via ``GL_EXCEL_NORMALIZER_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated ExcelNormalizerConfig instance.
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%s, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%s, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            max_file_size_mb=_int(
                "MAX_FILE_SIZE_MB", cls.max_file_size_mb,
            ),
            max_rows_per_sheet=_int(
                "MAX_ROWS_PER_SHEET", cls.max_rows_per_sheet,
            ),
            max_sheets_per_workbook=_int(
                "MAX_SHEETS_PER_WORKBOOK", cls.max_sheets_per_workbook,
            ),
            max_columns=_int("MAX_COLUMNS", cls.max_columns),
            default_encoding=_str(
                "DEFAULT_ENCODING", cls.default_encoding,
            ),
            default_delimiter=_str(
                "DEFAULT_DELIMITER", cls.default_delimiter,
            ),
            enable_encoding_detection=_bool(
                "ENABLE_ENCODING_DETECTION",
                cls.enable_encoding_detection,
            ),
            sample_rows_for_detection=_int(
                "SAMPLE_ROWS_FOR_DETECTION",
                cls.sample_rows_for_detection,
            ),
            default_mapping_strategy=_str(
                "DEFAULT_MAPPING_STRATEGY",
                cls.default_mapping_strategy,
            ),
            fuzzy_match_threshold=_float(
                "FUZZY_MATCH_THRESHOLD",
                cls.fuzzy_match_threshold,
            ),
            enable_synonym_matching=_bool(
                "ENABLE_SYNONYM_MATCHING",
                cls.enable_synonym_matching,
            ),
            sample_rows_for_type_detection=_int(
                "SAMPLE_ROWS_FOR_TYPE_DETECTION",
                cls.sample_rows_for_type_detection,
            ),
            date_formats=_str("DATE_FORMATS", cls.date_formats),
            enable_currency_detection=_bool(
                "ENABLE_CURRENCY_DETECTION",
                cls.enable_currency_detection,
            ),
            min_quality_score=_float(
                "MIN_QUALITY_SCORE", cls.min_quality_score,
            ),
            completeness_weight=_float(
                "COMPLETENESS_WEIGHT", cls.completeness_weight,
            ),
            accuracy_weight=_float(
                "ACCURACY_WEIGHT", cls.accuracy_weight,
            ),
            consistency_weight=_float(
                "CONSISTENCY_WEIGHT", cls.consistency_weight,
            ),
            batch_max_files=_int(
                "BATCH_MAX_FILES", cls.batch_max_files,
            ),
            batch_worker_count=_int(
                "BATCH_WORKER_COUNT", cls.batch_worker_count,
            ),
            processing_timeout_seconds=_int(
                "PROCESSING_TIMEOUT_SECONDS",
                cls.processing_timeout_seconds,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )

        logger.info(
            "ExcelNormalizerConfig loaded: max_file_size_mb=%d, "
            "max_rows_per_sheet=%d, max_sheets=%d, max_columns=%d, "
            "encoding=%s, delimiter=%r, mapping_strategy=%s, "
            "fuzzy_threshold=%.2f, min_quality=%.2f, batch_max=%d, "
            "workers=%d",
            config.max_file_size_mb,
            config.max_rows_per_sheet,
            config.max_sheets_per_workbook,
            config.max_columns,
            config.default_encoding,
            config.default_delimiter,
            config.default_mapping_strategy,
            config.fuzzy_match_threshold,
            config.min_quality_score,
            config.batch_max_files,
            config.batch_worker_count,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ExcelNormalizerConfig] = None
_config_lock = threading.Lock()


def get_config() -> ExcelNormalizerConfig:
    """Return the singleton ExcelNormalizerConfig, creating from env if needed.

    Returns:
        ExcelNormalizerConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ExcelNormalizerConfig.from_env()
    return _config_instance


def set_config(config: ExcelNormalizerConfig) -> None:
    """Replace the singleton ExcelNormalizerConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("ExcelNormalizerConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "ExcelNormalizerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
