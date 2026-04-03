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
    >>> from greenlang.agents.data.pdf_extractor.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_ocr_engine, cfg.default_confidence_threshold)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
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

_ENV_PREFIX = "GL_PDF_EXTRACTOR_"


# ---------------------------------------------------------------------------
# PDFExtractorConfig
# ---------------------------------------------------------------------------


@dataclass
class PDFExtractorConfig(BaseDataConfig):
    """Configuration for the GreenLang PDF & Invoice Extractor SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only PDF-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_PDF_EXTRACTOR_`` prefix.

    Attributes:
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
        supported_formats: Comma-separated list of supported file formats.
        extraction_timeout_seconds: Timeout in seconds for a single extraction.
    """

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

    # -- Batch processing (PDF-specific alias) --------------------------------
    batch_max_documents: int = 100

    # -- Format support ------------------------------------------------------
    supported_formats: str = "pdf,png,jpg,jpeg,tiff,bmp"

    # -- Timeouts ------------------------------------------------------------
    extraction_timeout_seconds: int = 120

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> PDFExtractorConfig:
        """Build a PDFExtractorConfig from environment variables.

        Every field can be overridden via ``GL_PDF_EXTRACTOR_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated PDFExtractorConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # OCR engines
            default_ocr_engine=env.str(
                "DEFAULT_OCR_ENGINE", cls.default_ocr_engine,
            ),
            tesseract_path=env.str("TESSERACT_PATH", cls.tesseract_path),
            aws_textract_region=env.str(
                "AWS_TEXTRACT_REGION", cls.aws_textract_region,
            ),
            azure_vision_endpoint=env.str(
                "AZURE_VISION_ENDPOINT", cls.azure_vision_endpoint,
            ),
            google_vision_credentials_path=env.str(
                "GOOGLE_VISION_CREDENTIALS_PATH",
                cls.google_vision_credentials_path,
            ),
            # Extraction defaults
            default_confidence_threshold=env.float(
                "DEFAULT_CONFIDENCE_THRESHOLD",
                cls.default_confidence_threshold,
            ),
            max_pages_per_document=env.int(
                "MAX_PAGES_PER_DOCUMENT", cls.max_pages_per_document,
            ),
            max_file_size_mb=env.int(
                "MAX_FILE_SIZE_MB", cls.max_file_size_mb,
            ),
            # Feature toggles
            enable_line_item_extraction=env.bool(
                "ENABLE_LINE_ITEM_EXTRACTION",
                cls.enable_line_item_extraction,
            ),
            enable_cross_field_validation=env.bool(
                "ENABLE_CROSS_FIELD_VALIDATION",
                cls.enable_cross_field_validation,
            ),
            enable_document_classification=env.bool(
                "ENABLE_DOCUMENT_CLASSIFICATION",
                cls.enable_document_classification,
            ),
            # Batch (PDF-specific alias reads both keys)
            batch_max_documents=env.int(
                "BATCH_MAX_DOCUMENTS", cls.batch_max_documents,
            ),
            # Formats & timeouts
            supported_formats=env.str(
                "SUPPORTED_FORMATS", cls.supported_formats,
            ),
            extraction_timeout_seconds=env.int(
                "EXTRACTION_TIMEOUT_SECONDS",
                cls.extraction_timeout_seconds,
            ),
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

get_config, set_config, reset_config = create_config_singleton(
    PDFExtractorConfig, _ENV_PREFIX,
)

__all__ = [
    "PDFExtractorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
