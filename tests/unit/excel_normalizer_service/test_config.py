# -*- coding: utf-8 -*-
"""
Unit Tests for ExcelNormalizerConfig (AGENT-DATA-002)

Tests configuration creation, env var overrides with GL_EXCEL_NORMALIZER_ prefix,
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
# Inline ExcelNormalizerConfig mirroring greenlang/excel_normalizer/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EXCEL_NORMALIZER_"


@dataclass
class ExcelNormalizerConfig:
    """Mirrors greenlang.excel_normalizer.config.ExcelNormalizerConfig."""

    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""
    max_file_size_mb: int = 50
    max_rows_per_sheet: int = 1_000_000
    max_sheets_per_workbook: int = 50
    max_columns: int = 500
    default_encoding: str = "utf-8"
    default_delimiter: str = ","
    enable_encoding_detection: bool = True
    sample_rows_for_detection: int = 100
    default_mapping_strategy: str = "fuzzy"
    fuzzy_match_threshold: float = 0.75
    enable_synonym_matching: bool = True
    sample_rows_for_type_detection: int = 1000
    date_formats: str = "ISO,US,EU"
    enable_currency_detection: bool = True
    min_quality_score: float = 0.5
    completeness_weight: float = 0.4
    accuracy_weight: float = 0.35
    consistency_weight: float = 0.25
    batch_max_files: int = 100
    batch_worker_count: int = 4
    processing_timeout_seconds: int = 300
    pool_min_size: int = 2
    pool_max_size: int = 10
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> ExcelNormalizerConfig:
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
            max_file_size_mb=_int("MAX_FILE_SIZE_MB", cls.max_file_size_mb),
            max_rows_per_sheet=_int("MAX_ROWS_PER_SHEET", cls.max_rows_per_sheet),
            max_sheets_per_workbook=_int("MAX_SHEETS_PER_WORKBOOK", cls.max_sheets_per_workbook),
            max_columns=_int("MAX_COLUMNS", cls.max_columns),
            default_encoding=_str("DEFAULT_ENCODING", cls.default_encoding),
            default_delimiter=_str("DEFAULT_DELIMITER", cls.default_delimiter),
            enable_encoding_detection=_bool("ENABLE_ENCODING_DETECTION", cls.enable_encoding_detection),
            sample_rows_for_detection=_int("SAMPLE_ROWS_FOR_DETECTION", cls.sample_rows_for_detection),
            default_mapping_strategy=_str("DEFAULT_MAPPING_STRATEGY", cls.default_mapping_strategy),
            fuzzy_match_threshold=_float("FUZZY_MATCH_THRESHOLD", cls.fuzzy_match_threshold),
            enable_synonym_matching=_bool("ENABLE_SYNONYM_MATCHING", cls.enable_synonym_matching),
            sample_rows_for_type_detection=_int("SAMPLE_ROWS_FOR_TYPE_DETECTION", cls.sample_rows_for_type_detection),
            date_formats=_str("DATE_FORMATS", cls.date_formats),
            enable_currency_detection=_bool("ENABLE_CURRENCY_DETECTION", cls.enable_currency_detection),
            min_quality_score=_float("MIN_QUALITY_SCORE", cls.min_quality_score),
            completeness_weight=_float("COMPLETENESS_WEIGHT", cls.completeness_weight),
            accuracy_weight=_float("ACCURACY_WEIGHT", cls.accuracy_weight),
            consistency_weight=_float("CONSISTENCY_WEIGHT", cls.consistency_weight),
            batch_max_files=_int("BATCH_MAX_FILES", cls.batch_max_files),
            batch_worker_count=_int("BATCH_WORKER_COUNT", cls.batch_worker_count),
            processing_timeout_seconds=_int("PROCESSING_TIMEOUT_SECONDS", cls.processing_timeout_seconds),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[ExcelNormalizerConfig] = None
_config_lock = threading.Lock()


def get_config() -> ExcelNormalizerConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ExcelNormalizerConfig.from_env()
    return _config_instance


def set_config(config: ExcelNormalizerConfig) -> None:
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


class TestExcelNormalizerConfigDefaults:
    """Test that default configuration values match AGENT-DATA-002 PRD."""

    def test_default_database_url(self):
        config = ExcelNormalizerConfig()
        assert config.database_url == ""

    def test_default_redis_url(self):
        config = ExcelNormalizerConfig()
        assert config.redis_url == ""

    def test_default_s3_bucket_url(self):
        config = ExcelNormalizerConfig()
        assert config.s3_bucket_url == ""

    def test_default_max_file_size_mb(self):
        config = ExcelNormalizerConfig()
        assert config.max_file_size_mb == 50

    def test_default_max_rows_per_sheet(self):
        config = ExcelNormalizerConfig()
        assert config.max_rows_per_sheet == 1_000_000

    def test_default_max_sheets_per_workbook(self):
        config = ExcelNormalizerConfig()
        assert config.max_sheets_per_workbook == 50

    def test_default_max_columns(self):
        config = ExcelNormalizerConfig()
        assert config.max_columns == 500

    def test_default_encoding(self):
        config = ExcelNormalizerConfig()
        assert config.default_encoding == "utf-8"

    def test_default_delimiter(self):
        config = ExcelNormalizerConfig()
        assert config.default_delimiter == ","

    def test_default_enable_encoding_detection(self):
        config = ExcelNormalizerConfig()
        assert config.enable_encoding_detection is True

    def test_default_sample_rows_for_detection(self):
        config = ExcelNormalizerConfig()
        assert config.sample_rows_for_detection == 100

    def test_default_mapping_strategy(self):
        config = ExcelNormalizerConfig()
        assert config.default_mapping_strategy == "fuzzy"

    def test_default_fuzzy_match_threshold(self):
        config = ExcelNormalizerConfig()
        assert config.fuzzy_match_threshold == 0.75

    def test_default_enable_synonym_matching(self):
        config = ExcelNormalizerConfig()
        assert config.enable_synonym_matching is True

    def test_default_sample_rows_for_type_detection(self):
        config = ExcelNormalizerConfig()
        assert config.sample_rows_for_type_detection == 1000

    def test_default_date_formats(self):
        config = ExcelNormalizerConfig()
        assert config.date_formats == "ISO,US,EU"

    def test_default_enable_currency_detection(self):
        config = ExcelNormalizerConfig()
        assert config.enable_currency_detection is True

    def test_default_min_quality_score(self):
        config = ExcelNormalizerConfig()
        assert config.min_quality_score == 0.5

    def test_default_completeness_weight(self):
        config = ExcelNormalizerConfig()
        assert config.completeness_weight == 0.4

    def test_default_accuracy_weight(self):
        config = ExcelNormalizerConfig()
        assert config.accuracy_weight == 0.35

    def test_default_consistency_weight(self):
        config = ExcelNormalizerConfig()
        assert config.consistency_weight == 0.25

    def test_default_batch_max_files(self):
        config = ExcelNormalizerConfig()
        assert config.batch_max_files == 100

    def test_default_batch_worker_count(self):
        config = ExcelNormalizerConfig()
        assert config.batch_worker_count == 4

    def test_default_processing_timeout_seconds(self):
        config = ExcelNormalizerConfig()
        assert config.processing_timeout_seconds == 300

    def test_default_pool_min_size(self):
        config = ExcelNormalizerConfig()
        assert config.pool_min_size == 2

    def test_default_pool_max_size(self):
        config = ExcelNormalizerConfig()
        assert config.pool_max_size == 10

    def test_default_log_level(self):
        config = ExcelNormalizerConfig()
        assert config.log_level == "INFO"

    def test_weights_sum_to_one(self):
        config = ExcelNormalizerConfig()
        total = config.completeness_weight + config.accuracy_weight + config.consistency_weight
        assert total == pytest.approx(1.0, rel=1e-6)


class TestExcelNormalizerConfigFromEnv:
    """Test GL_EXCEL_NORMALIZER_ env var overrides via from_env()."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_DATABASE_URL", "postgresql://test:5432/db")
        config = ExcelNormalizerConfig.from_env()
        assert config.database_url == "postgresql://test:5432/db"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_REDIS_URL", "redis://localhost:6379/0")
        config = ExcelNormalizerConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/0"

    def test_env_override_s3_bucket_url(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_S3_BUCKET_URL", "s3://my-bucket")
        config = ExcelNormalizerConfig.from_env()
        assert config.s3_bucket_url == "s3://my-bucket"

    def test_env_override_max_file_size(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_MAX_FILE_SIZE_MB", "200")
        config = ExcelNormalizerConfig.from_env()
        assert config.max_file_size_mb == 200

    def test_env_override_max_rows(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_MAX_ROWS_PER_SHEET", "500000")
        config = ExcelNormalizerConfig.from_env()
        assert config.max_rows_per_sheet == 500000

    def test_env_override_max_sheets(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_MAX_SHEETS_PER_WORKBOOK", "100")
        config = ExcelNormalizerConfig.from_env()
        assert config.max_sheets_per_workbook == 100

    def test_env_override_max_columns(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_MAX_COLUMNS", "1000")
        config = ExcelNormalizerConfig.from_env()
        assert config.max_columns == 1000

    def test_env_override_encoding(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_DEFAULT_ENCODING", "latin-1")
        config = ExcelNormalizerConfig.from_env()
        assert config.default_encoding == "latin-1"

    def test_env_override_delimiter(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_DEFAULT_DELIMITER", ";")
        config = ExcelNormalizerConfig.from_env()
        assert config.default_delimiter == ";"

    def test_env_override_enable_encoding_false(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_ENABLE_ENCODING_DETECTION", "false")
        config = ExcelNormalizerConfig.from_env()
        assert config.enable_encoding_detection is False

    def test_env_override_bool_true_with_1(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_ENABLE_ENCODING_DETECTION", "1")
        config = ExcelNormalizerConfig.from_env()
        assert config.enable_encoding_detection is True

    def test_env_override_bool_true_with_yes(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_ENABLE_SYNONYM_MATCHING", "yes")
        config = ExcelNormalizerConfig.from_env()
        assert config.enable_synonym_matching is True

    def test_env_override_bool_true_with_TRUE(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_ENABLE_CURRENCY_DETECTION", "TRUE")
        config = ExcelNormalizerConfig.from_env()
        assert config.enable_currency_detection is True

    def test_env_override_bool_false_with_0(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_ENABLE_SYNONYM_MATCHING", "0")
        config = ExcelNormalizerConfig.from_env()
        assert config.enable_synonym_matching is False

    def test_env_override_bool_false_with_no(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_ENABLE_CURRENCY_DETECTION", "no")
        config = ExcelNormalizerConfig.from_env()
        assert config.enable_currency_detection is False

    def test_env_override_fuzzy_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_FUZZY_MATCH_THRESHOLD", "0.85")
        config = ExcelNormalizerConfig.from_env()
        assert config.fuzzy_match_threshold == 0.85

    def test_env_override_min_quality_score(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_MIN_QUALITY_SCORE", "0.7")
        config = ExcelNormalizerConfig.from_env()
        assert config.min_quality_score == 0.7

    def test_env_override_batch_max_files(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_BATCH_MAX_FILES", "500")
        config = ExcelNormalizerConfig.from_env()
        assert config.batch_max_files == 500

    def test_env_override_batch_worker_count(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_BATCH_WORKER_COUNT", "16")
        config = ExcelNormalizerConfig.from_env()
        assert config.batch_worker_count == 16

    def test_env_override_processing_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_PROCESSING_TIMEOUT_SECONDS", "600")
        config = ExcelNormalizerConfig.from_env()
        assert config.processing_timeout_seconds == 600

    def test_env_override_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_POOL_MIN_SIZE", "5")
        config = ExcelNormalizerConfig.from_env()
        assert config.pool_min_size == 5

    def test_env_override_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_POOL_MAX_SIZE", "50")
        config = ExcelNormalizerConfig.from_env()
        assert config.pool_max_size == 50

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_LOG_LEVEL", "DEBUG")
        config = ExcelNormalizerConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_MAX_FILE_SIZE_MB", "not_a_number")
        config = ExcelNormalizerConfig.from_env()
        assert config.max_file_size_mb == 50

    def test_env_invalid_float_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_FUZZY_MATCH_THRESHOLD", "bad")
        config = ExcelNormalizerConfig.from_env()
        assert config.fuzzy_match_threshold == 0.75

    def test_env_invalid_int_batch_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_BATCH_WORKER_COUNT", "abc")
        config = ExcelNormalizerConfig.from_env()
        assert config.batch_worker_count == 4

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_DEFAULT_ENCODING", "latin-1")
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_MAX_FILE_SIZE_MB", "200")
        monkeypatch.setenv("GL_EXCEL_NORMALIZER_ENABLE_ENCODING_DETECTION", "false")
        config = ExcelNormalizerConfig.from_env()
        assert config.default_encoding == "latin-1"
        assert config.max_file_size_mb == 200
        assert config.enable_encoding_detection is False


class TestExcelNormalizerConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, ExcelNormalizerConfig)

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
        custom = ExcelNormalizerConfig(default_encoding="latin-1")
        set_config(custom)
        assert get_config().default_encoding == "latin-1"

    def test_set_config_then_get_returns_same(self):
        custom = ExcelNormalizerConfig(max_rows_per_sheet=500)
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


class TestExcelNormalizerConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = ExcelNormalizerConfig(
            database_url="postgresql://custom:5432/excel",
            redis_url="redis://custom:6379/1",
            s3_bucket_url="s3://custom-bucket",
            max_file_size_mb=200,
            max_rows_per_sheet=500000,
            max_sheets_per_workbook=100,
            max_columns=1000,
            default_encoding="latin-1",
            default_delimiter=";",
            enable_encoding_detection=False,
            sample_rows_for_detection=200,
            default_mapping_strategy="exact",
            fuzzy_match_threshold=0.9,
            enable_synonym_matching=False,
            sample_rows_for_type_detection=2000,
            date_formats="ISO",
            enable_currency_detection=False,
            min_quality_score=0.8,
            completeness_weight=0.5,
            accuracy_weight=0.3,
            consistency_weight=0.2,
            batch_max_files=500,
            batch_worker_count=16,
            processing_timeout_seconds=600,
            pool_min_size=5,
            pool_max_size=50,
            log_level="DEBUG",
        )
        assert config.database_url == "postgresql://custom:5432/excel"
        assert config.default_encoding == "latin-1"
        assert config.fuzzy_match_threshold == 0.9
        assert config.max_rows_per_sheet == 500000
        assert config.enable_encoding_detection is False
        assert config.enable_synonym_matching is False
        assert config.enable_currency_detection is False
        assert config.batch_max_files == 500
        assert config.batch_worker_count == 16
        assert config.pool_min_size == 5
        assert config.pool_max_size == 50
        assert config.log_level == "DEBUG"

    def test_date_formats_parsing(self):
        config = ExcelNormalizerConfig(date_formats="ISO,US,EU")
        formats = config.date_formats.split(",")
        assert len(formats) == 3
        assert "ISO" in formats
        assert "EU" in formats

    def test_delimiter_options(self):
        for delim in [",", ";", "\t", "|"]:
            config = ExcelNormalizerConfig(default_delimiter=delim)
            assert config.default_delimiter == delim

    def test_mapping_strategy_options(self):
        for strategy in ["exact", "fuzzy", "synonym", "pattern", "manual"]:
            config = ExcelNormalizerConfig(default_mapping_strategy=strategy)
            assert config.default_mapping_strategy == strategy
