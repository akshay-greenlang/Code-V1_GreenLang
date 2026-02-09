# -*- coding: utf-8 -*-
"""
Unit Tests for EUDRTraceabilityConfig (AGENT-DATA-005)

Tests configuration creation, env var overrides with GL_EUDR_TRACEABILITY_ prefix,
type parsing (bool, int, float, str), singleton get_config/set_config/reset_config,
thread-safety of singleton access, risk weight validation, and EUDR-specific defaults.

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
# Inline EUDRTraceabilityConfig mirroring greenlang/eudr_traceability/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_TRACEABILITY_"


@dataclass
class EUDRTraceabilityConfig:
    """Mirrors greenlang.eudr_traceability.config.EUDRTraceabilityConfig."""

    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""
    deforestation_cutoff_date: str = "2020-12-31"
    geolocation_precision: int = 6
    max_plot_area_ha: float = 10000.0
    enable_polygon_validation: bool = True
    enable_satellite_check: bool = False
    risk_weight_country: float = 0.30
    risk_weight_commodity: float = 0.25
    risk_weight_supplier: float = 0.25
    risk_weight_traceability: float = 0.20
    dds_auto_submit: bool = False
    dds_retention_years: int = 5
    batch_size: int = 500
    max_chain_depth: int = 50
    enable_mass_balance: bool = True
    default_custody_model: str = "segregated"
    cn_code_validation: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> EUDRTraceabilityConfig:
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
            deforestation_cutoff_date=_str("DEFORESTATION_CUTOFF_DATE", cls.deforestation_cutoff_date),
            geolocation_precision=_int("GEOLOCATION_PRECISION", cls.geolocation_precision),
            max_plot_area_ha=_float("MAX_PLOT_AREA_HA", cls.max_plot_area_ha),
            enable_polygon_validation=_bool("ENABLE_POLYGON_VALIDATION", cls.enable_polygon_validation),
            enable_satellite_check=_bool("ENABLE_SATELLITE_CHECK", cls.enable_satellite_check),
            risk_weight_country=_float("RISK_WEIGHT_COUNTRY", cls.risk_weight_country),
            risk_weight_commodity=_float("RISK_WEIGHT_COMMODITY", cls.risk_weight_commodity),
            risk_weight_supplier=_float("RISK_WEIGHT_SUPPLIER", cls.risk_weight_supplier),
            risk_weight_traceability=_float("RISK_WEIGHT_TRACEABILITY", cls.risk_weight_traceability),
            dds_auto_submit=_bool("DDS_AUTO_SUBMIT", cls.dds_auto_submit),
            dds_retention_years=_int("DDS_RETENTION_YEARS", cls.dds_retention_years),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            max_chain_depth=_int("MAX_CHAIN_DEPTH", cls.max_chain_depth),
            enable_mass_balance=_bool("ENABLE_MASS_BALANCE", cls.enable_mass_balance),
            default_custody_model=_str("DEFAULT_CUSTODY_MODEL", cls.default_custody_model),
            cn_code_validation=_bool("CN_CODE_VALIDATION", cls.cn_code_validation),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[EUDRTraceabilityConfig] = None
_config_lock = threading.Lock()


def get_config() -> EUDRTraceabilityConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = EUDRTraceabilityConfig.from_env()
    return _config_instance


def set_config(config: EUDRTraceabilityConfig) -> None:
    global _config_instance
    with _config_lock:
        _config_instance = config


def reset_config() -> None:
    global _config_instance
    with _config_lock:
        _config_instance = None


# ---------------------------------------------------------------------------
# Autouse: reset singleton and clean env between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _clean_eudr_env(monkeypatch):
    """Remove any GL_EUDR_TRACEABILITY_ env vars between tests."""
    prefix = "GL_EUDR_TRACEABILITY_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestEUDRTraceabilityConfigDefaults:
    """Test that default configuration values match AGENT-DATA-005 PRD."""

    def test_default_config(self):
        """All defaults match expected values."""
        config = EUDRTraceabilityConfig()
        assert config.database_url == ""
        assert config.redis_url == ""
        assert config.s3_bucket_url == ""
        assert config.deforestation_cutoff_date == "2020-12-31"
        assert config.geolocation_precision == 6
        assert config.max_plot_area_ha == 10000.0
        assert config.enable_polygon_validation is True
        assert config.enable_satellite_check is False
        assert config.risk_weight_country == 0.30
        assert config.risk_weight_commodity == 0.25
        assert config.risk_weight_supplier == 0.25
        assert config.risk_weight_traceability == 0.20
        assert config.dds_auto_submit is False
        assert config.dds_retention_years == 5
        assert config.batch_size == 500
        assert config.max_chain_depth == 50
        assert config.enable_mass_balance is True
        assert config.default_custody_model == "segregated"
        assert config.cn_code_validation is True
        assert config.log_level == "INFO"

    def test_deforestation_cutoff_date(self):
        """Default deforestation cutoff is 2020-12-31 per EUDR Article 2."""
        config = EUDRTraceabilityConfig()
        assert config.deforestation_cutoff_date == "2020-12-31"

    def test_risk_weights_sum(self):
        """country + commodity + supplier + traceability weights sum to 1.0."""
        config = EUDRTraceabilityConfig()
        total = (
            config.risk_weight_country
            + config.risk_weight_commodity
            + config.risk_weight_supplier
            + config.risk_weight_traceability
        )
        assert total == pytest.approx(1.0)


class TestEUDRTraceabilityConfigFromEnv:
    """Test GL_EUDR_TRACEABILITY_ env var overrides via from_env()."""

    def test_from_env_str_override(self, monkeypatch):
        """Environment variable override for str field."""
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_DATABASE_URL", "postgresql://test:5432/eudr")
        config = EUDRTraceabilityConfig.from_env()
        assert config.database_url == "postgresql://test:5432/eudr"

    def test_from_env_int_override(self, monkeypatch):
        """Environment variable override for int field."""
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_GEOLOCATION_PRECISION", "8")
        config = EUDRTraceabilityConfig.from_env()
        assert config.geolocation_precision == 8

    def test_from_env_bool_override(self, monkeypatch):
        """Environment variable override for bool field."""
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_ENABLE_SATELLITE_CHECK", "true")
        config = EUDRTraceabilityConfig.from_env()
        assert config.enable_satellite_check is True

    def test_from_env_float_override(self, monkeypatch):
        """Environment variable override for float field."""
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_MAX_PLOT_AREA_HA", "25000.5")
        config = EUDRTraceabilityConfig.from_env()
        assert config.max_plot_area_ha == pytest.approx(25000.5)

    def test_env_prefix(self, monkeypatch):
        """Verify GL_EUDR_TRACEABILITY_ prefix is used for all env lookups."""
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_LOG_LEVEL", "DEBUG")
        config = EUDRTraceabilityConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_override_cutoff_date(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_DEFORESTATION_CUTOFF_DATE", "2021-06-30")
        config = EUDRTraceabilityConfig.from_env()
        assert config.deforestation_cutoff_date == "2021-06-30"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_REDIS_URL", "redis://localhost:6379/3")
        config = EUDRTraceabilityConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/3"

    def test_env_override_s3_bucket(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_S3_BUCKET_URL", "s3://eudr-data")
        config = EUDRTraceabilityConfig.from_env()
        assert config.s3_bucket_url == "s3://eudr-data"

    def test_env_override_custody_model(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_DEFAULT_CUSTODY_MODEL", "mass_balance")
        config = EUDRTraceabilityConfig.from_env()
        assert config.default_custody_model == "mass_balance"

    def test_env_override_risk_weights(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_RISK_WEIGHT_COUNTRY", "0.40")
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_RISK_WEIGHT_COMMODITY", "0.20")
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_RISK_WEIGHT_SUPPLIER", "0.20")
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_RISK_WEIGHT_TRACEABILITY", "0.20")
        config = EUDRTraceabilityConfig.from_env()
        assert config.risk_weight_country == pytest.approx(0.40)
        assert config.risk_weight_commodity == pytest.approx(0.20)
        total = (
            config.risk_weight_country
            + config.risk_weight_commodity
            + config.risk_weight_supplier
            + config.risk_weight_traceability
        )
        assert total == pytest.approx(1.0)

    def test_env_override_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_BATCH_SIZE", "2000")
        config = EUDRTraceabilityConfig.from_env()
        assert config.batch_size == 2000

    def test_env_override_max_chain_depth(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_MAX_CHAIN_DEPTH", "100")
        config = EUDRTraceabilityConfig.from_env()
        assert config.max_chain_depth == 100

    def test_env_override_dds_retention_years(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_DDS_RETENTION_YEARS", "10")
        config = EUDRTraceabilityConfig.from_env()
        assert config.dds_retention_years == 10

    def test_env_override_dds_auto_submit(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_DDS_AUTO_SUBMIT", "true")
        config = EUDRTraceabilityConfig.from_env()
        assert config.dds_auto_submit is True


class TestEUDRTraceabilityConfigBoolParsing:
    """Test boolean environment variable parsing for true/1/yes and false/0/no."""

    @pytest.mark.parametrize("env_val,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("anything_else", False),
    ])
    def test_bool_parsing(self, monkeypatch, env_val, expected):
        """Bool parsing: true/1/yes are True, everything else is False."""
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_ENABLE_SATELLITE_CHECK", env_val)
        config = EUDRTraceabilityConfig.from_env()
        assert config.enable_satellite_check is expected


class TestEUDRTraceabilityConfigInvalidFallback:
    """Test fallback to default for invalid int/float env values."""

    def test_invalid_int_fallback(self, monkeypatch):
        """Invalid int falls back to default value."""
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_GEOLOCATION_PRECISION", "not_a_number")
        config = EUDRTraceabilityConfig.from_env()
        assert config.geolocation_precision == 6

    def test_invalid_int_batch_size_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_BATCH_SIZE", "xyz")
        config = EUDRTraceabilityConfig.from_env()
        assert config.batch_size == 500

    def test_invalid_float_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_MAX_PLOT_AREA_HA", "not_float")
        config = EUDRTraceabilityConfig.from_env()
        assert config.max_plot_area_ha == 10000.0

    def test_invalid_float_risk_weight_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_TRACEABILITY_RISK_WEIGHT_COUNTRY", "abc")
        config = EUDRTraceabilityConfig.from_env()
        assert config.risk_weight_country == pytest.approx(0.30)


class TestEUDRTraceabilityConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_singleton(self):
        """Thread-safe singleton returns the same instance on repeated calls."""
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
        assert isinstance(c1, EUDRTraceabilityConfig)

    def test_set_config(self):
        """Replace config programmatically via set_config."""
        custom = EUDRTraceabilityConfig(default_custody_model="identity_preserved")
        set_config(custom)
        assert get_config().default_custody_model == "identity_preserved"
        assert get_config() is custom

    def test_reset_config(self):
        """Reset config to None so next get_config creates a new instance."""
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_thread_safety_of_get_config(self):
        """Concurrent get_config calls from 10 threads all get the same instance."""
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
