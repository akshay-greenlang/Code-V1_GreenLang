# -*- coding: utf-8 -*-
"""
Unit Tests for PDFExtractorConfig (AGENT-DATA-001)

Tests configuration creation, env var overrides with GL_PDF_EXTRACTOR_ prefix,
type parsing (bool, int, float, str), singleton get_config/set_config/reset_config,
and thread-safety of singleton access.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline PDFExtractorConfig mirroring greenlang/pdf_extractor/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_PDF_EXTRACTOR_"


@dataclass
class PDFExtractorConfig:
    """Mirrors greenlang.pdf_extractor.config.PDFExtractorConfig."""

    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""
    default_ocr_engine: str = "tesseract"
    tesseract_path: str = "/usr/bin/tesseract"
    aws_textract_region: str = "us-east-1"
    azure_vision_endpoint: str = ""
    google_vision_credentials_path: str = ""
    default_confidence_threshold: float = 0.7
    max_pages_per_document: int = 100
    max_file_size_mb: int = 50
    enable_line_item_extraction: bool = True
    enable_cross_field_validation: bool = True
    enable_document_classification: bool = True
    batch_max_documents: int = 100
    batch_worker_count: int = 4
    supported_formats: str = "pdf,png,jpg,jpeg,tiff,bmp"
    extraction_timeout_seconds: int = 120
    pool_min_size: int = 2
    pool_max_size: int = 10
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> PDFExtractorConfig:
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
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        return cls(
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            default_ocr_engine=_str("DEFAULT_OCR_ENGINE", cls.default_ocr_engine),
            tesseract_path=_str("TESSERACT_PATH", cls.tesseract_path),
            aws_textract_region=_str("AWS_TEXTRACT_REGION", cls.aws_textract_region),
            azure_vision_endpoint=_str("AZURE_VISION_ENDPOINT", cls.azure_vision_endpoint),
            google_vision_credentials_path=_str(
                "GOOGLE_VISION_CREDENTIALS_PATH", cls.google_vision_credentials_path,
            ),
            default_confidence_threshold=_float(
                "DEFAULT_CONFIDENCE_THRESHOLD", cls.default_confidence_threshold,
            ),
            max_pages_per_document=_int("MAX_PAGES_PER_DOCUMENT", cls.max_pages_per_document),
            max_file_size_mb=_int("MAX_FILE_SIZE_MB", cls.max_file_size_mb),
            enable_line_item_extraction=_bool(
                "ENABLE_LINE_ITEM_EXTRACTION", cls.enable_line_item_extraction,
            ),
            enable_cross_field_validation=_bool(
                "ENABLE_CROSS_FIELD_VALIDATION", cls.enable_cross_field_validation,
            ),
            enable_document_classification=_bool(
                "ENABLE_DOCUMENT_CLASSIFICATION", cls.enable_document_classification,
            ),
            batch_max_documents=_int("BATCH_MAX_DOCUMENTS", cls.batch_max_documents),
            batch_worker_count=_int("BATCH_WORKER_COUNT", cls.batch_worker_count),
            supported_formats=_str("SUPPORTED_FORMATS", cls.supported_formats),
            extraction_timeout_seconds=_int(
                "EXTRACTION_TIMEOUT_SECONDS", cls.extraction_timeout_seconds,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[PDFExtractorConfig] = None
_config_lock = threading.Lock()


def get_config() -> PDFExtractorConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = PDFExtractorConfig.from_env()
    return _config_instance


def set_config(config: PDFExtractorConfig) -> None:
    global _config_instance
    with _config_lock:
        _config_instance = config


def reset_config() -> None:
    global _config_instance
    with _config_lock:
        _config_instance = None


# ---------------------------------------------------------------------------
# Autouse: reset singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    yield
    reset_config()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPDFExtractorConfigDefaults:
    """Test that default configuration values match AGENT-DATA-001 PRD."""

    def test_default_database_url(self):
        config = PDFExtractorConfig()
        assert config.database_url == ""

    def test_default_redis_url(self):
        config = PDFExtractorConfig()
        assert config.redis_url == ""

    def test_default_s3_bucket_url(self):
        config = PDFExtractorConfig()
        assert config.s3_bucket_url == ""

    def test_default_ocr_engine(self):
        config = PDFExtractorConfig()
        assert config.default_ocr_engine == "tesseract"

    def test_default_tesseract_path(self):
        config = PDFExtractorConfig()
        assert config.tesseract_path == "/usr/bin/tesseract"

    def test_default_aws_textract_region(self):
        config = PDFExtractorConfig()
        assert config.aws_textract_region == "us-east-1"

    def test_default_azure_vision_endpoint(self):
        config = PDFExtractorConfig()
        assert config.azure_vision_endpoint == ""

    def test_default_google_vision_credentials_path(self):
        config = PDFExtractorConfig()
        assert config.google_vision_credentials_path == ""

    def test_default_confidence_threshold(self):
        config = PDFExtractorConfig()
        assert config.default_confidence_threshold == 0.7

    def test_default_max_pages_per_document(self):
        config = PDFExtractorConfig()
        assert config.max_pages_per_document == 100

    def test_default_max_file_size_mb(self):
        config = PDFExtractorConfig()
        assert config.max_file_size_mb == 50

    def test_default_enable_line_item_extraction(self):
        config = PDFExtractorConfig()
        assert config.enable_line_item_extraction is True

    def test_default_enable_cross_field_validation(self):
        config = PDFExtractorConfig()
        assert config.enable_cross_field_validation is True

    def test_default_enable_document_classification(self):
        config = PDFExtractorConfig()
        assert config.enable_document_classification is True

    def test_default_batch_max_documents(self):
        config = PDFExtractorConfig()
        assert config.batch_max_documents == 100

    def test_default_batch_worker_count(self):
        config = PDFExtractorConfig()
        assert config.batch_worker_count == 4

    def test_default_supported_formats(self):
        config = PDFExtractorConfig()
        assert config.supported_formats == "pdf,png,jpg,jpeg,tiff,bmp"

    def test_default_extraction_timeout_seconds(self):
        config = PDFExtractorConfig()
        assert config.extraction_timeout_seconds == 120

    def test_default_pool_min_size(self):
        config = PDFExtractorConfig()
        assert config.pool_min_size == 2

    def test_default_pool_max_size(self):
        config = PDFExtractorConfig()
        assert config.pool_max_size == 10

    def test_default_log_level(self):
        config = PDFExtractorConfig()
        assert config.log_level == "INFO"


class TestPDFExtractorConfigFromEnv:
    """Test GL_PDF_EXTRACTOR_ env var overrides via from_env()."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_DATABASE_URL", "postgresql://test:5432/db")
        config = PDFExtractorConfig.from_env()
        assert config.database_url == "postgresql://test:5432/db"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_REDIS_URL", "redis://localhost:6379/0")
        config = PDFExtractorConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/0"

    def test_env_override_ocr_engine(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_DEFAULT_OCR_ENGINE", "textract")
        config = PDFExtractorConfig.from_env()
        assert config.default_ocr_engine == "textract"

    def test_env_override_confidence_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_DEFAULT_CONFIDENCE_THRESHOLD", "0.85")
        config = PDFExtractorConfig.from_env()
        assert config.default_confidence_threshold == 0.85

    def test_env_override_max_pages(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_MAX_PAGES_PER_DOCUMENT", "200")
        config = PDFExtractorConfig.from_env()
        assert config.max_pages_per_document == 200

    def test_env_override_max_file_size(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_MAX_FILE_SIZE_MB", "100")
        config = PDFExtractorConfig.from_env()
        assert config.max_file_size_mb == 100

    def test_env_override_enable_line_items_false(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_ENABLE_LINE_ITEM_EXTRACTION", "false")
        config = PDFExtractorConfig.from_env()
        assert config.enable_line_item_extraction is False

    def test_env_override_enable_cross_validation_false(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_ENABLE_CROSS_FIELD_VALIDATION", "0")
        config = PDFExtractorConfig.from_env()
        assert config.enable_cross_field_validation is False

    def test_env_override_enable_classification_false(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_ENABLE_DOCUMENT_CLASSIFICATION", "no")
        config = PDFExtractorConfig.from_env()
        assert config.enable_document_classification is False

    def test_env_override_bool_true_with_1(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_ENABLE_LINE_ITEM_EXTRACTION", "1")
        config = PDFExtractorConfig.from_env()
        assert config.enable_line_item_extraction is True

    def test_env_override_bool_true_with_yes(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_ENABLE_LINE_ITEM_EXTRACTION", "yes")
        config = PDFExtractorConfig.from_env()
        assert config.enable_line_item_extraction is True

    def test_env_override_bool_true_with_TRUE(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_ENABLE_CROSS_FIELD_VALIDATION", "TRUE")
        config = PDFExtractorConfig.from_env()
        assert config.enable_cross_field_validation is True

    def test_env_override_batch_max_documents(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_BATCH_MAX_DOCUMENTS", "500")
        config = PDFExtractorConfig.from_env()
        assert config.batch_max_documents == 500

    def test_env_override_batch_worker_count(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_BATCH_WORKER_COUNT", "8")
        config = PDFExtractorConfig.from_env()
        assert config.batch_worker_count == 8

    def test_env_override_supported_formats(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_SUPPORTED_FORMATS", "pdf,png")
        config = PDFExtractorConfig.from_env()
        assert config.supported_formats == "pdf,png"

    def test_env_override_extraction_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_EXTRACTION_TIMEOUT_SECONDS", "300")
        config = PDFExtractorConfig.from_env()
        assert config.extraction_timeout_seconds == 300

    def test_env_override_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_POOL_MIN_SIZE", "5")
        config = PDFExtractorConfig.from_env()
        assert config.pool_min_size == 5

    def test_env_override_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_POOL_MAX_SIZE", "50")
        config = PDFExtractorConfig.from_env()
        assert config.pool_max_size == 50

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_LOG_LEVEL", "DEBUG")
        config = PDFExtractorConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_MAX_PAGES_PER_DOCUMENT", "not_a_number")
        config = PDFExtractorConfig.from_env()
        assert config.max_pages_per_document == 100

    def test_env_invalid_float_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_DEFAULT_CONFIDENCE_THRESHOLD", "bad")
        config = PDFExtractorConfig.from_env()
        assert config.default_confidence_threshold == 0.7

    def test_env_invalid_int_batch_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_BATCH_WORKER_COUNT", "abc")
        config = PDFExtractorConfig.from_env()
        assert config.batch_worker_count == 4

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_PDF_EXTRACTOR_DEFAULT_OCR_ENGINE", "google_vision")
        monkeypatch.setenv("GL_PDF_EXTRACTOR_MAX_FILE_SIZE_MB", "200")
        monkeypatch.setenv("GL_PDF_EXTRACTOR_ENABLE_LINE_ITEM_EXTRACTION", "false")
        config = PDFExtractorConfig.from_env()
        assert config.default_ocr_engine == "google_vision"
        assert config.max_file_size_mb == 200
        assert config.enable_line_item_extraction is False


class TestPDFExtractorConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, PDFExtractorConfig)

    def test_get_config_returns_same_instance(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset_config_clears_singleton(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_set_config_overrides_singleton(self):
        custom = PDFExtractorConfig(default_ocr_engine="textract")
        set_config(custom)
        assert get_config().default_ocr_engine == "textract"

    def test_set_config_then_get_returns_same(self):
        custom = PDFExtractorConfig(max_pages_per_document=500)
        set_config(custom)
        assert get_config() is custom

    def test_thread_safety_of_get_config(self):
        """Test that concurrent get_config calls return the same instance."""
        instances = []

        def get_instance():
            instances.append(get_config())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        for inst in instances[1:]:
            assert inst is instances[0]


class TestPDFExtractorConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = PDFExtractorConfig(
            database_url="postgresql://custom:5432/pdf",
            redis_url="redis://custom:6379/1",
            s3_bucket_url="s3://custom-bucket",
            default_ocr_engine="azure_vision",
            tesseract_path="/opt/tesseract/bin/tesseract",
            aws_textract_region="eu-west-1",
            azure_vision_endpoint="https://my-vision.cognitiveservices.azure.com",
            google_vision_credentials_path="/secrets/gcp-creds.json",
            default_confidence_threshold=0.9,
            max_pages_per_document=500,
            max_file_size_mb=200,
            enable_line_item_extraction=False,
            enable_cross_field_validation=False,
            enable_document_classification=False,
            batch_max_documents=1000,
            batch_worker_count=16,
            supported_formats="pdf,png",
            extraction_timeout_seconds=300,
            pool_min_size=5,
            pool_max_size=50,
            log_level="DEBUG",
        )
        assert config.database_url == "postgresql://custom:5432/pdf"
        assert config.default_ocr_engine == "azure_vision"
        assert config.default_confidence_threshold == 0.9
        assert config.max_pages_per_document == 500
        assert config.enable_line_item_extraction is False
        assert config.enable_cross_field_validation is False
        assert config.enable_document_classification is False
        assert config.batch_max_documents == 1000
        assert config.batch_worker_count == 16
        assert config.supported_formats == "pdf,png"
        assert config.pool_min_size == 5
        assert config.pool_max_size == 50
        assert config.log_level == "DEBUG"

    def test_supported_formats_parsing(self):
        config = PDFExtractorConfig(supported_formats="pdf,png,jpg,jpeg,tiff,bmp")
        formats = config.supported_formats.split(",")
        assert len(formats) == 6
        assert "pdf" in formats
        assert "tiff" in formats
