# -*- coding: utf-8 -*-
"""
PDF & Invoice Extractor Service Configuration - AGENT-DATA-001: PDF Extractor

Centralized configuration for the PDF & Invoice Extractor SDK covering:
- OCR engine selection and paths (Tesseract, AWS Textract, Azure Vision, Google)
- Extraction defaults (confidence threshold, page limits, file size)
- Document classification toggles
- Line item and cross-field validation toggles
- Batch processing defaults (max documents, worker count)
- Supported file formats
- Connection pool sizing

All settings can be overridden via environment variables with the
``GL_PDF_EXTRACTOR_`` prefix (e.g. ``GL_PDF_EXTRACTOR_DEFAULT_OCR_ENGINE``).

Example:
    >>> from greenlang.pdf_extractor.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_ocr_engine, cfg.default_confidence_threshold)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
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

_ENV_PREFIX = "GL_PDF_EXTRACTOR_"


# ---------------------------------------------------------------------------
# PDFExtractorConfig
# ---------------------------------------------------------------------------


@dataclass
class PDFExtractorConfig:
    """Complete configuration for the GreenLang PDF & Invoice Extractor SDK.

    Attributes are grouped by concern: connections, OCR engines,
    extraction defaults, feature toggles, batch processing, format support,
    pool sizing, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_PDF_EXTRACTOR_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for document storage.
        default_ocr_engine: Default OCR engine to use for extraction.
        tesseract_path: Filesystem path to the Tesseract binary.
        aws_textract_region: AWS region for Textract API calls.
        azure_vision_endpoint: Azure Computer Vision endpoint URL.
        google_vision_credentials_path: Path to Google Vision credentials JSON.
        default_confidence_threshold: Minimum confidence for accepted extractions.
        max_pages_per_document: Maximum number of pages to process per document.
        max_file_size_mb: Maximum file size in megabytes for ingestion.
        enable_line_item_extraction: Whether to extract line items from invoices.
        enable_cross_field_validation: Whether to run cross-field validation rules.
        enable_document_classification: Whether to auto-classify document types.
        batch_max_documents: Maximum documents allowed in a single batch job.
        batch_worker_count: Number of parallel workers for batch processing.
        supported_formats: Comma-separated list of supported file formats.
        extraction_timeout_seconds: Timeout in seconds for a single extraction.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the PDF extractor service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- OCR engines ---------------------------------------------------------
    default_ocr_engine: str = "tesseract"
    tesseract_path: str = "/usr/bin/tesseract"
    aws_textract_region: str = "us-east-1"
    azure_vision_endpoint: str = ""
    google_vision_credentials_path: str = ""

    # -- Extraction defaults -------------------------------------------------
    default_confidence_threshold: float = 0.7
    max_pages_per_document: int = 100
    max_file_size_mb: int = 50

    # -- Feature toggles -----------------------------------------------------
    enable_line_item_extraction: bool = True
    enable_cross_field_validation: bool = True
    enable_document_classification: bool = True

    # -- Batch processing ----------------------------------------------------
    batch_max_documents: int = 100
    batch_worker_count: int = 4

    # -- Format support ------------------------------------------------------
    supported_formats: str = "pdf,png,jpg,jpeg,tiff,bmp"

    # -- Timeouts ------------------------------------------------------------
    extraction_timeout_seconds: int = 120

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> PDFExtractorConfig:
        """Build a PDFExtractorConfig from environment variables.

        Every field can be overridden via ``GL_PDF_EXTRACTOR_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated PDFExtractorConfig instance.
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
            default_ocr_engine=_str(
                "DEFAULT_OCR_ENGINE", cls.default_ocr_engine,
            ),
            tesseract_path=_str("TESSERACT_PATH", cls.tesseract_path),
            aws_textract_region=_str(
                "AWS_TEXTRACT_REGION", cls.aws_textract_region,
            ),
            azure_vision_endpoint=_str(
                "AZURE_VISION_ENDPOINT", cls.azure_vision_endpoint,
            ),
            google_vision_credentials_path=_str(
                "GOOGLE_VISION_CREDENTIALS_PATH",
                cls.google_vision_credentials_path,
            ),
            default_confidence_threshold=_float(
                "DEFAULT_CONFIDENCE_THRESHOLD",
                cls.default_confidence_threshold,
            ),
            max_pages_per_document=_int(
                "MAX_PAGES_PER_DOCUMENT", cls.max_pages_per_document,
            ),
            max_file_size_mb=_int(
                "MAX_FILE_SIZE_MB", cls.max_file_size_mb,
            ),
            enable_line_item_extraction=_bool(
                "ENABLE_LINE_ITEM_EXTRACTION",
                cls.enable_line_item_extraction,
            ),
            enable_cross_field_validation=_bool(
                "ENABLE_CROSS_FIELD_VALIDATION",
                cls.enable_cross_field_validation,
            ),
            enable_document_classification=_bool(
                "ENABLE_DOCUMENT_CLASSIFICATION",
                cls.enable_document_classification,
            ),
            batch_max_documents=_int(
                "BATCH_MAX_DOCUMENTS", cls.batch_max_documents,
            ),
            batch_worker_count=_int(
                "BATCH_WORKER_COUNT", cls.batch_worker_count,
            ),
            supported_formats=_str(
                "SUPPORTED_FORMATS", cls.supported_formats,
            ),
            extraction_timeout_seconds=_int(
                "EXTRACTION_TIMEOUT_SECONDS",
                cls.extraction_timeout_seconds,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )

        logger.info(
            "PDFExtractorConfig loaded: ocr_engine=%s, confidence=%.2f, "
            "max_pages=%d, max_file_size_mb=%d, line_items=%s, "
            "cross_validation=%s, classification=%s, batch_max=%d, "
            "workers=%d, formats=%s",
            config.default_ocr_engine,
            config.default_confidence_threshold,
            config.max_pages_per_document,
            config.max_file_size_mb,
            config.enable_line_item_extraction,
            config.enable_cross_field_validation,
            config.enable_document_classification,
            config.batch_max_documents,
            config.batch_worker_count,
            config.supported_formats,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[PDFExtractorConfig] = None
_config_lock = threading.Lock()


def get_config() -> PDFExtractorConfig:
    """Return the singleton PDFExtractorConfig, creating from env if needed.

    Returns:
        PDFExtractorConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = PDFExtractorConfig.from_env()
    return _config_instance


def set_config(config: PDFExtractorConfig) -> None:
    """Replace the singleton PDFExtractorConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("PDFExtractorConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "PDFExtractorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
