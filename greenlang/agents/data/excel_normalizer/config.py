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
    >>> from greenlang.agents.data.excel_normalizer.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_file_size_mb, cfg.default_delimiter)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel & CSV Normalizer
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from greenlang.data_commons.config_base import (
    BaseDataConfig,
    EnvReader,
    create_config_singleton,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EXCEL_NORMALIZER_"


# ---------------------------------------------------------------------------
# ExcelNormalizerConfig
# ---------------------------------------------------------------------------


@dataclass
class ExcelNormalizerConfig(BaseDataConfig):
    """Configuration for the GreenLang Excel & CSV Normalizer SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only Excel/CSV-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_EXCEL_NORMALIZER_`` prefix.

    Attributes:
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
        processing_timeout_seconds: Timeout in seconds for a single file.
    """

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

    # -- Batch processing (Excel-specific alias) ------------------------------
    batch_max_files: int = 100

    # -- Timeouts ------------------------------------------------------------
    processing_timeout_seconds: int = 300

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ExcelNormalizerConfig:
        """Build an ExcelNormalizerConfig from environment variables.

        Every field can be overridden via ``GL_EXCEL_NORMALIZER_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated ExcelNormalizerConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # Parser settings
            max_file_size_mb=env.int(
                "MAX_FILE_SIZE_MB", cls.max_file_size_mb,
            ),
            max_rows_per_sheet=env.int(
                "MAX_ROWS_PER_SHEET", cls.max_rows_per_sheet,
            ),
            max_sheets_per_workbook=env.int(
                "MAX_SHEETS_PER_WORKBOOK", cls.max_sheets_per_workbook,
            ),
            max_columns=env.int("MAX_COLUMNS", cls.max_columns),
            # CSV settings
            default_encoding=env.str(
                "DEFAULT_ENCODING", cls.default_encoding,
            ),
            default_delimiter=env.str(
                "DEFAULT_DELIMITER", cls.default_delimiter,
            ),
            enable_encoding_detection=env.bool(
                "ENABLE_ENCODING_DETECTION",
                cls.enable_encoding_detection,
            ),
            sample_rows_for_detection=env.int(
                "SAMPLE_ROWS_FOR_DETECTION",
                cls.sample_rows_for_detection,
            ),
            # Column mapping
            default_mapping_strategy=env.str(
                "DEFAULT_MAPPING_STRATEGY",
                cls.default_mapping_strategy,
            ),
            fuzzy_match_threshold=env.float(
                "FUZZY_MATCH_THRESHOLD",
                cls.fuzzy_match_threshold,
            ),
            enable_synonym_matching=env.bool(
                "ENABLE_SYNONYM_MATCHING",
                cls.enable_synonym_matching,
            ),
            # Type detection
            sample_rows_for_type_detection=env.int(
                "SAMPLE_ROWS_FOR_TYPE_DETECTION",
                cls.sample_rows_for_type_detection,
            ),
            date_formats=env.str("DATE_FORMATS", cls.date_formats),
            enable_currency_detection=env.bool(
                "ENABLE_CURRENCY_DETECTION",
                cls.enable_currency_detection,
            ),
            # Quality scoring
            min_quality_score=env.float(
                "MIN_QUALITY_SCORE", cls.min_quality_score,
            ),
            completeness_weight=env.float(
                "COMPLETENESS_WEIGHT", cls.completeness_weight,
            ),
            accuracy_weight=env.float(
                "ACCURACY_WEIGHT", cls.accuracy_weight,
            ),
            consistency_weight=env.float(
                "CONSISTENCY_WEIGHT", cls.consistency_weight,
            ),
            # Batch (Excel-specific alias)
            batch_max_files=env.int(
                "BATCH_MAX_FILES", cls.batch_max_files,
            ),
            # Timeouts
            processing_timeout_seconds=env.int(
                "PROCESSING_TIMEOUT_SECONDS",
                cls.processing_timeout_seconds,
            ),
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

get_config, set_config, reset_config = create_config_singleton(
    ExcelNormalizerConfig, _ENV_PREFIX,
)

__all__ = [
    "ExcelNormalizerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
