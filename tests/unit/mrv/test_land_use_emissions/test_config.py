# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-006 Land Use Emissions Agent Configuration.

Tests LandUseConfig default values, environment variable overrides,
thread-safe singleton, reset_config, validation of ranges, and all
configuration fields.

Target: 40 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.land_use_emissions.config import (
    LandUseConfig,
    get_config,
    set_config,
    reset_config,
    _LandUseConfigHolder,
)


# ===========================================================================
# Default Value Tests
# ===========================================================================


class TestLandUseConfigDefaults:
    """Tests for LandUseConfig default values."""

    def test_enabled_default_true(self):
        """Default enabled is True."""
        cfg = LandUseConfig()
        assert cfg.enabled is True

    def test_default_gwp_source(self):
        """Default GWP source is AR6."""
        cfg = LandUseConfig()
        assert cfg.default_gwp_source == "AR6"

    def test_default_tier(self):
        """Default calculation tier is 1."""
        cfg = LandUseConfig()
        assert cfg.default_tier == 1

    def test_default_method(self):
        """Default method is STOCK_DIFFERENCE."""
        cfg = LandUseConfig()
        assert cfg.default_method == "STOCK_DIFFERENCE"

    def test_default_emission_factor_source(self):
        """Default emission factor source is IPCC_2006."""
        cfg = LandUseConfig()
        assert cfg.default_emission_factor_source == "IPCC_2006"

    def test_default_decimal_precision(self):
        """Default decimal precision is 8."""
        cfg = LandUseConfig()
        assert cfg.decimal_precision == 8

    def test_default_transition_years(self):
        """Default transition years is 20."""
        cfg = LandUseConfig()
        assert cfg.default_transition_years == 20

    def test_default_carbon_fraction(self):
        """Default carbon fraction is 0.47."""
        cfg = LandUseConfig()
        assert cfg.carbon_fraction == 0.47

    def test_default_soc_depth(self):
        """Default SOC depth is 30 cm."""
        cfg = LandUseConfig()
        assert cfg.soc_depth_cm == 30

    def test_default_soc_method(self):
        """Default SOC method is tier1."""
        cfg = LandUseConfig()
        assert cfg.default_soc_method == "tier1"

    def test_default_max_batch_size(self):
        """Default max batch size is 10000."""
        cfg = LandUseConfig()
        assert cfg.max_batch_size == 10_000

    def test_default_monte_carlo_iterations(self):
        """Default Monte Carlo iterations is 5000."""
        cfg = LandUseConfig()
        assert cfg.monte_carlo_iterations == 5_000

    def test_default_monte_carlo_seed(self):
        """Default Monte Carlo seed is 42."""
        cfg = LandUseConfig()
        assert cfg.monte_carlo_seed == 42

    def test_default_confidence_levels(self):
        """Default confidence levels is '90,95,99'."""
        cfg = LandUseConfig()
        assert cfg.confidence_levels == "90,95,99"

    def test_feature_toggles_all_true(self):
        """All feature toggles default to True."""
        cfg = LandUseConfig()
        assert cfg.enable_soc_assessment is True
        assert cfg.enable_peatland is True
        assert cfg.enable_fire_emissions is True
        assert cfg.enable_n2o_soil is True
        assert cfg.enable_compliance_checking is True
        assert cfg.enable_uncertainty is True
        assert cfg.enable_provenance is True
        assert cfg.enable_metrics is True

    def test_default_api_prefix(self):
        """Default API prefix is /api/v1/land-use-emissions."""
        cfg = LandUseConfig()
        assert cfg.api_prefix == "/api/v1/land-use-emissions"

    def test_default_worker_threads(self):
        """Default worker threads is 4."""
        cfg = LandUseConfig()
        assert cfg.worker_threads == 4

    def test_default_genesis_hash(self):
        """Default genesis hash is the expected anchor string."""
        cfg = LandUseConfig()
        assert cfg.genesis_hash == "GL-MRV-X-006-LAND-USE-EMISSIONS-GENESIS"


# ===========================================================================
# Environment Variable Override Tests
# ===========================================================================


class TestLandUseConfigEnvOverrides:
    """Tests for environment variable overrides with GL_LAND_USE_ prefix."""

    def test_override_default_method(self, clean_env):
        """GL_LAND_USE_DEFAULT_METHOD overrides default_method."""
        os.environ["GL_LAND_USE_DEFAULT_METHOD"] = "GAIN_LOSS"
        cfg = LandUseConfig.from_env()
        assert cfg.default_method == "GAIN_LOSS"

    def test_override_default_tier(self, clean_env):
        """GL_LAND_USE_DEFAULT_TIER overrides default_tier."""
        os.environ["GL_LAND_USE_DEFAULT_TIER"] = "2"
        cfg = LandUseConfig.from_env()
        assert cfg.default_tier == 2

    def test_override_gwp_source(self, clean_env):
        """GL_LAND_USE_DEFAULT_GWP_SOURCE overrides default_gwp_source."""
        os.environ["GL_LAND_USE_DEFAULT_GWP_SOURCE"] = "AR5"
        cfg = LandUseConfig.from_env()
        assert cfg.default_gwp_source == "AR5"

    def test_override_enabled_false(self, clean_env):
        """GL_LAND_USE_ENABLED=false disables the agent."""
        os.environ["GL_LAND_USE_ENABLED"] = "false"
        cfg = LandUseConfig.from_env()
        assert cfg.enabled is False

    def test_override_decimal_precision(self, clean_env):
        """GL_LAND_USE_DECIMAL_PRECISION overrides decimal_precision."""
        os.environ["GL_LAND_USE_DECIMAL_PRECISION"] = "12"
        cfg = LandUseConfig.from_env()
        assert cfg.decimal_precision == 12

    def test_override_max_batch_size(self, clean_env):
        """GL_LAND_USE_MAX_BATCH_SIZE overrides max_batch_size."""
        os.environ["GL_LAND_USE_MAX_BATCH_SIZE"] = "5000"
        cfg = LandUseConfig.from_env()
        assert cfg.max_batch_size == 5000

    def test_invalid_integer_falls_back_to_default(self, clean_env):
        """Invalid integer value falls back to default with warning."""
        os.environ["GL_LAND_USE_MAX_BATCH_SIZE"] = "not_a_number"
        cfg = LandUseConfig.from_env()
        assert cfg.max_batch_size == 10_000

    def test_boolean_accepts_yes(self, clean_env):
        """Boolean env vars accept 'yes' as True."""
        os.environ["GL_LAND_USE_ENABLE_METRICS"] = "yes"
        cfg = LandUseConfig.from_env()
        assert cfg.enable_metrics is True

    def test_boolean_accepts_1(self, clean_env):
        """Boolean env vars accept '1' as True."""
        os.environ["GL_LAND_USE_ENABLE_PROVENANCE"] = "1"
        cfg = LandUseConfig.from_env()
        assert cfg.enable_provenance is True


# ===========================================================================
# Validation Tests
# ===========================================================================


class TestLandUseConfigValidation:
    """Tests for LandUseConfig post-init validation."""

    def test_invalid_gwp_source_raises(self):
        """Invalid GWP source raises ValueError."""
        with pytest.raises(ValueError, match="default_gwp_source"):
            LandUseConfig(default_gwp_source="INVALID")

    def test_invalid_tier_raises(self):
        """Invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="default_tier"):
            LandUseConfig(default_tier=99)

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="default_method"):
            LandUseConfig(default_method="INVALID_METHOD")

    def test_negative_decimal_precision_raises(self):
        """Negative decimal precision raises ValueError."""
        with pytest.raises(ValueError, match="decimal_precision"):
            LandUseConfig(decimal_precision=-1)

    def test_decimal_precision_too_high_raises(self):
        """Decimal precision > 20 raises ValueError."""
        with pytest.raises(ValueError, match="decimal_precision"):
            LandUseConfig(decimal_precision=21)

    def test_zero_transition_years_raises(self):
        """Zero transition years raises ValueError."""
        with pytest.raises(ValueError, match="default_transition_years"):
            LandUseConfig(default_transition_years=0)

    def test_transition_years_above_100_raises(self):
        """Transition years > 100 raises ValueError."""
        with pytest.raises(ValueError, match="default_transition_years"):
            LandUseConfig(default_transition_years=101)

    def test_carbon_fraction_zero_raises(self):
        """Carbon fraction of 0 raises ValueError."""
        with pytest.raises(ValueError, match="carbon_fraction"):
            LandUseConfig(carbon_fraction=0.0)

    def test_carbon_fraction_above_one_raises(self):
        """Carbon fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="carbon_fraction"):
            LandUseConfig(carbon_fraction=1.5)

    def test_negative_monte_carlo_iterations_raises(self):
        """Negative Monte Carlo iterations raises ValueError."""
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            LandUseConfig(monte_carlo_iterations=-1)

    def test_empty_genesis_hash_raises(self):
        """Empty genesis hash raises ValueError."""
        with pytest.raises(ValueError, match="genesis_hash"):
            LandUseConfig(genesis_hash="")

    def test_invalid_log_level_raises(self):
        """Invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="log_level"):
            LandUseConfig(log_level="VERBOSE")

    def test_invalid_confidence_levels_raises(self):
        """Non-numeric confidence levels raise ValueError."""
        with pytest.raises(ValueError, match="confidence_levels"):
            LandUseConfig(confidence_levels="abc,def")

    def test_confidence_level_out_of_range_raises(self):
        """Confidence level >= 100 raises ValueError."""
        with pytest.raises(ValueError, match="confidence level"):
            LandUseConfig(confidence_levels="50,100")

    def test_api_default_page_exceeds_max_raises(self):
        """Default page size exceeding max raises ValueError."""
        with pytest.raises(ValueError, match="api_default_page_size"):
            LandUseConfig(api_default_page_size=200, api_max_page_size=100)

    def test_worker_threads_zero_raises(self):
        """Zero worker threads raises ValueError."""
        with pytest.raises(ValueError, match="worker_threads"):
            LandUseConfig(worker_threads=0)

    def test_worker_threads_above_64_raises(self):
        """Worker threads > 64 raises ValueError."""
        with pytest.raises(ValueError, match="worker_threads"):
            LandUseConfig(worker_threads=65)


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestSingleton:
    """Tests for thread-safe singleton access."""

    def test_get_config_returns_instance(self, clean_env):
        """get_config returns a LandUseConfig instance."""
        cfg = get_config()
        assert isinstance(cfg, LandUseConfig)

    def test_get_config_returns_same_instance(self, clean_env):
        """get_config returns the same instance on repeated calls."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self, clean_env):
        """set_config replaces the singleton."""
        custom = LandUseConfig(default_method="GAIN_LOSS")
        set_config(custom)
        assert get_config().default_method == "GAIN_LOSS"

    def test_reset_config_clears_singleton(self, clean_env):
        """reset_config clears the singleton so next get creates fresh."""
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_thread_safe_get_config(self, clean_env):
        """get_config is thread-safe under concurrent access."""
        results = []
        errors = []

        def worker():
            try:
                cfg = get_config()
                results.append(id(cfg))
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(set(results)) == 1


# ===========================================================================
# Serialization Tests
# ===========================================================================


class TestSerialization:
    """Tests for LandUseConfig serialization helpers."""

    def test_to_dict_redacts_database_url(self):
        """to_dict redacts database_url when it has a value."""
        cfg = LandUseConfig(database_url="postgresql://user:pass@host/db")
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_empty_database_url(self):
        """to_dict shows empty string when database_url is empty."""
        cfg = LandUseConfig()
        d = cfg.to_dict()
        assert d["database_url"] == ""

    def test_repr_is_credential_safe(self):
        """repr does not leak connection strings."""
        cfg = LandUseConfig(database_url="postgresql://secret@host/db")
        text = repr(cfg)
        assert "secret" not in text
        assert "***" in text

    def test_to_dict_has_all_fields(self):
        """to_dict includes all configuration fields."""
        cfg = LandUseConfig()
        d = cfg.to_dict()
        assert "default_gwp_source" in d
        assert "default_tier" in d
        assert "default_method" in d
        assert "enable_provenance" in d
        assert "genesis_hash" in d
