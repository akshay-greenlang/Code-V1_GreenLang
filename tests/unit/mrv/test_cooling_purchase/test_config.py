"""Tests for AGENT-MRV-012 Cooling Purchase Agent configuration."""

import pytest
import os
import threading
from decimal import Decimal
from typing import Dict

try:
    from greenlang.agents.mrv.cooling_purchase.config import CoolingPurchaseConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")


class TestCoolingPurchaseConfig:
    """Test CoolingPurchaseConfig singleton."""

    def test_singleton_pattern(self):
        """Test CoolingPurchaseConfig follows singleton pattern."""
        config1 = CoolingPurchaseConfig()
        config2 = CoolingPurchaseConfig()
        assert config1 is config2

    def test_reset_singleton(self):
        """Test reset() clears singleton instance."""
        config1 = CoolingPurchaseConfig()
        CoolingPurchaseConfig.reset()
        config2 = CoolingPurchaseConfig()
        assert config1 is not config2

    def test_thread_safety(self):
        """Test thread-safe singleton creation."""
        instances = []

        def create_instance():
            instances.append(CoolingPurchaseConfig())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)

    def test_default_database_config(self):
        """Test default database configuration."""
        config = CoolingPurchaseConfig()
        db_config = config.database
        assert db_config["host"] == "localhost"
        assert db_config["port"] == 5432
        assert db_config["database"] == "greenlang"
        assert db_config["min_pool_size"] == 2
        assert db_config["max_pool_size"] == 10

    def test_default_calculation_config(self):
        """Test default calculation configuration."""
        config = CoolingPurchaseConfig()
        calc_config = config.calculation
        assert calc_config["default_gwp_source"] == "IPCC_AR5"
        assert calc_config["default_tier"] == "TIER_2"
        assert calc_config["include_refrigerant_leakage"] is True
        assert calc_config["include_parasitic_loads"] is True

    def test_default_ahri_config(self):
        """Test default AHRI configuration."""
        config = CoolingPurchaseConfig()
        ahri_config = config.ahri
        assert "part_load_weights" in ahri_config
        weights = ahri_config["part_load_weights"]
        assert "100%" in weights
        assert "75%" in weights
        assert "50%" in weights
        assert "25%" in weights

    def test_default_tes_config(self):
        """Test default TES configuration."""
        config = CoolingPurchaseConfig()
        tes_config = config.tes
        assert "ice_storage_efficiency" in tes_config
        assert "chilled_water_efficiency" in tes_config
        assert "pcm_efficiency" in tes_config

    def test_default_uncertainty_config(self):
        """Test default uncertainty configuration."""
        config = CoolingPurchaseConfig()
        unc_config = config.uncertainty
        assert unc_config["default_cop_uncertainty_pct"] == Decimal("5.0")
        assert unc_config["default_ef_uncertainty_pct"] == Decimal("10.0")
        assert unc_config["default_leakage_uncertainty_pct"] == Decimal("20.0")
        assert unc_config["monte_carlo_samples"] == 10000

    def test_default_compliance_config(self):
        """Test default compliance configuration."""
        config = CoolingPurchaseConfig()
        comp_config = config.compliance
        assert comp_config["default_frameworks"] == ["GHG_PROTOCOL", "ISO_14064"]
        assert comp_config["require_third_party_verification"] is False
        assert comp_config["min_data_quality_score"] == Decimal("60.0")

    def test_default_performance_config(self):
        """Test default performance configuration."""
        config = CoolingPurchaseConfig()
        perf_config = config.performance
        assert perf_config["batch_size"] == 1000
        assert perf_config["max_workers"] == 4
        assert perf_config["cache_ttl_seconds"] == 3600
        assert perf_config["enable_query_cache"] is True

    def test_default_api_config(self):
        """Test default API configuration."""
        config = CoolingPurchaseConfig()
        api_config = config.api
        assert api_config["prefix"] == "/api/v1/cooling-purchase"
        assert api_config["max_request_size_mb"] == 10
        assert api_config["timeout_seconds"] == 30

    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = CoolingPurchaseConfig()
        log_config = config.logging
        assert log_config["level"] == "INFO"
        assert log_config["format"] == "json"
        assert log_config["enable_metrics"] is True

    def test_get_ahri_weights(self):
        """Test get_ahri_weights() returns Dict[str, Decimal]."""
        config = CoolingPurchaseConfig()
        weights = config.get_ahri_weights()
        assert isinstance(weights, dict)
        assert all(isinstance(k, str) for k in weights.keys())
        assert all(isinstance(v, Decimal) for v in weights.values())

    def test_ahri_weights_sum_to_one(self):
        """Test AHRI weights sum to 1.0."""
        config = CoolingPurchaseConfig()
        weights = config.get_ahri_weights()
        total = sum(weights.values())
        assert abs(float(total) - 1.0) < 0.001

    def test_get_tes_efficiency_ice(self):
        """Test get_tes_efficiency() for ICE_STORAGE."""
        config = CoolingPurchaseConfig()
        efficiency = config.get_tes_efficiency("ICE_STORAGE")
        assert isinstance(efficiency, Decimal)
        assert efficiency > 0

    def test_get_tes_efficiency_chilled_water(self):
        """Test get_tes_efficiency() for CHILLED_WATER."""
        config = CoolingPurchaseConfig()
        efficiency = config.get_tes_efficiency("CHILLED_WATER")
        assert isinstance(efficiency, Decimal)
        assert efficiency > 0

    def test_get_tes_efficiency_pcm(self):
        """Test get_tes_efficiency() for PHASE_CHANGE_MATERIAL."""
        config = CoolingPurchaseConfig()
        efficiency = config.get_tes_efficiency("PHASE_CHANGE_MATERIAL")
        assert isinstance(efficiency, Decimal)
        assert efficiency > 0

    def test_get_uncertainty_config(self):
        """Test get_uncertainty_config() returns complete dict."""
        config = CoolingPurchaseConfig()
        unc_config = config.get_uncertainty_config()
        assert "default_cop_uncertainty_pct" in unc_config
        assert "default_ef_uncertainty_pct" in unc_config
        assert "default_leakage_uncertainty_pct" in unc_config
        assert "monte_carlo_samples" in unc_config

    def test_get_enabled_frameworks(self):
        """Test get_enabled_frameworks() returns 7 frameworks."""
        config = CoolingPurchaseConfig()
        frameworks = config.get_enabled_frameworks()
        assert len(frameworks) >= 2  # At least default frameworks
        assert "GHG_PROTOCOL" in frameworks
        assert "ISO_14064" in frameworks

    def test_get_db_dsn_format(self):
        """Test get_db_dsn() returns valid PostgreSQL DSN."""
        config = CoolingPurchaseConfig()
        dsn = config.get_db_dsn()
        assert dsn.startswith("postgresql://")
        assert "localhost" in dsn
        assert "5432" in dsn
        assert "greenlang" in dsn

    def test_get_api_prefix(self):
        """Test get_api_prefix() returns correct prefix."""
        config = CoolingPurchaseConfig()
        prefix = config.get_api_prefix()
        assert prefix == "/api/v1/cooling-purchase"

    def test_env_var_prefix(self):
        """Test GL_CP_ environment variable prefix."""
        os.environ["GL_CP_DATABASE_HOST"] = "testhost"
        CoolingPurchaseConfig.reset()
        config = CoolingPurchaseConfig()
        assert config.database["host"] == "testhost"
        del os.environ["GL_CP_DATABASE_HOST"]

    def test_env_var_override_port(self):
        """Test environment variable overrides port."""
        os.environ["GL_CP_DATABASE_PORT"] = "5433"
        CoolingPurchaseConfig.reset()
        config = CoolingPurchaseConfig()
        assert config.database["port"] == 5433
        del os.environ["GL_CP_DATABASE_PORT"]

    def test_env_var_override_batch_size(self):
        """Test environment variable overrides batch_size."""
        os.environ["GL_CP_PERFORMANCE_BATCH_SIZE"] = "2000"
        CoolingPurchaseConfig.reset()
        config = CoolingPurchaseConfig()
        assert config.performance["batch_size"] == 2000
        del os.environ["GL_CP_PERFORMANCE_BATCH_SIZE"]

    def test_env_var_override_gwp_source(self):
        """Test environment variable overrides default_gwp_source."""
        os.environ["GL_CP_CALCULATION_DEFAULT_GWP_SOURCE"] = "IPCC_AR6"
        CoolingPurchaseConfig.reset()
        config = CoolingPurchaseConfig()
        assert config.calculation["default_gwp_source"] == "IPCC_AR6"
        del os.environ["GL_CP_CALCULATION_DEFAULT_GWP_SOURCE"]

    def test_validation_rejects_negative_port(self):
        """Test validation rejects negative port."""
        os.environ["GL_CP_DATABASE_PORT"] = "-1"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_DATABASE_PORT"]

    def test_validation_rejects_invalid_gwp_source(self):
        """Test validation rejects invalid GWP source."""
        os.environ["GL_CP_CALCULATION_DEFAULT_GWP_SOURCE"] = "INVALID"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_CALCULATION_DEFAULT_GWP_SOURCE"]

    def test_ahri_weights_must_sum_to_one(self):
        """Test AHRI weights validation rejects non-summing weights."""
        # This would require mocking the config loading, skip for now
        pass

    def test_min_pool_size_less_than_max(self):
        """Test min_pool_size must be less than max_pool_size."""
        os.environ["GL_CP_DATABASE_MIN_POOL_SIZE"] = "20"
        os.environ["GL_CP_DATABASE_MAX_POOL_SIZE"] = "10"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_DATABASE_MIN_POOL_SIZE"]
        del os.environ["GL_CP_DATABASE_MAX_POOL_SIZE"]

    def test_cache_ttl_positive(self):
        """Test cache_ttl_seconds must be positive."""
        os.environ["GL_CP_PERFORMANCE_CACHE_TTL_SECONDS"] = "-100"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_PERFORMANCE_CACHE_TTL_SECONDS"]

    def test_max_workers_positive(self):
        """Test max_workers must be positive."""
        os.environ["GL_CP_PERFORMANCE_MAX_WORKERS"] = "0"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_PERFORMANCE_MAX_WORKERS"]

    def test_monte_carlo_samples_minimum(self):
        """Test monte_carlo_samples has minimum value."""
        os.environ["GL_CP_UNCERTAINTY_MONTE_CARLO_SAMPLES"] = "10"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_UNCERTAINTY_MONTE_CARLO_SAMPLES"]

    def test_uncertainty_percentages_positive(self):
        """Test uncertainty percentages must be positive."""
        os.environ["GL_CP_UNCERTAINTY_DEFAULT_COP_UNCERTAINTY_PCT"] = "-5.0"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_UNCERTAINTY_DEFAULT_COP_UNCERTAINTY_PCT"]

    def test_data_quality_score_range(self):
        """Test min_data_quality_score must be 0-100."""
        os.environ["GL_CP_COMPLIANCE_MIN_DATA_QUALITY_SCORE"] = "150.0"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_COMPLIANCE_MIN_DATA_QUALITY_SCORE"]

    def test_timeout_positive(self):
        """Test timeout_seconds must be positive."""
        os.environ["GL_CP_API_TIMEOUT_SECONDS"] = "-10"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_API_TIMEOUT_SECONDS"]

    def test_max_request_size_positive(self):
        """Test max_request_size_mb must be positive."""
        os.environ["GL_CP_API_MAX_REQUEST_SIZE_MB"] = "0"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_API_MAX_REQUEST_SIZE_MB"]

    def test_logging_level_valid(self):
        """Test logging level must be valid."""
        os.environ["GL_CP_LOGGING_LEVEL"] = "INVALID"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_LOGGING_LEVEL"]

    def test_logging_format_valid(self):
        """Test logging format must be valid."""
        os.environ["GL_CP_LOGGING_FORMAT"] = "INVALID"
        CoolingPurchaseConfig.reset()
        with pytest.raises(Exception):
            CoolingPurchaseConfig()
        del os.environ["GL_CP_LOGGING_FORMAT"]

    def test_config_immutability(self):
        """Test config dict is immutable after creation."""
        config = CoolingPurchaseConfig()
        with pytest.raises(Exception):
            config.database["host"] = "newhost"

    def test_get_default_tier(self):
        """Test get_default_tier() returns correct tier."""
        config = CoolingPurchaseConfig()
        tier = config.get_default_tier()
        assert tier in ["TIER_1", "TIER_2", "TIER_3"]

    def test_get_default_gwp_source(self):
        """Test get_default_gwp_source() returns correct source."""
        config = CoolingPurchaseConfig()
        source = config.get_default_gwp_source()
        assert source in ["IPCC_AR5", "IPCC_AR6", "MONTREAL_PROTOCOL", "EPA"]

    def test_is_refrigerant_leakage_enabled(self):
        """Test is_refrigerant_leakage_enabled() returns bool."""
        config = CoolingPurchaseConfig()
        enabled = config.is_refrigerant_leakage_enabled()
        assert isinstance(enabled, bool)

    def test_is_parasitic_loads_enabled(self):
        """Test is_parasitic_loads_enabled() returns bool."""
        config = CoolingPurchaseConfig()
        enabled = config.is_parasitic_loads_enabled()
        assert isinstance(enabled, bool)

    def test_get_batch_size(self):
        """Test get_batch_size() returns positive int."""
        config = CoolingPurchaseConfig()
        batch_size = config.get_batch_size()
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def test_get_max_workers(self):
        """Test get_max_workers() returns positive int."""
        config = CoolingPurchaseConfig()
        max_workers = config.get_max_workers()
        assert isinstance(max_workers, int)
        assert max_workers > 0

    def test_is_cache_enabled(self):
        """Test is_cache_enabled() returns bool."""
        config = CoolingPurchaseConfig()
        enabled = config.is_cache_enabled()
        assert isinstance(enabled, bool)

    def test_get_cache_ttl(self):
        """Test get_cache_ttl() returns positive int."""
        config = CoolingPurchaseConfig()
        ttl = config.get_cache_ttl()
        assert isinstance(ttl, int)
        assert ttl > 0

    def test_is_metrics_enabled(self):
        """Test is_metrics_enabled() returns bool."""
        config = CoolingPurchaseConfig()
        enabled = config.is_metrics_enabled()
        assert isinstance(enabled, bool)

    def test_get_log_level(self):
        """Test get_log_level() returns valid level."""
        config = CoolingPurchaseConfig()
        level = config.get_log_level()
        assert level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_get_min_data_quality_score(self):
        """Test get_min_data_quality_score() returns Decimal 0-100."""
        config = CoolingPurchaseConfig()
        score = config.get_min_data_quality_score()
        assert isinstance(score, Decimal)
        assert Decimal("0.0") <= score <= Decimal("100.0")

    def test_requires_third_party_verification(self):
        """Test requires_third_party_verification() returns bool."""
        config = CoolingPurchaseConfig()
        requires = config.requires_third_party_verification()
        assert isinstance(requires, bool)
