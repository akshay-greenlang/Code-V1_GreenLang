"""
Test suite for AGENT-MRV-014 configuration.

Tests the PurchasedGoodsServicesConfig singleton and environment variable handling.
"""

import pytest
import os
from decimal import Decimal

from greenlang.agents.mrv.purchased_goods_services.config import PurchasedGoodsServicesConfig


class TestPurchasedGoodsServicesConfig:
    """Test PurchasedGoodsServicesConfig singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        PurchasedGoodsServicesConfig.reset()

    def teardown_method(self):
        """Clean up environment variables after each test."""
        # Remove all GL_PGS_ prefixed env vars
        for key in list(os.environ.keys()):
            if key.startswith("GL_PGS_"):
                del os.environ[key]
        PurchasedGoodsServicesConfig.reset()

    def test_singleton_pattern(self):
        """Test config follows singleton pattern."""
        config1 = PurchasedGoodsServicesConfig()
        config2 = PurchasedGoodsServicesConfig()
        assert config1 is config2

    def test_database_section_defaults(self):
        """Test database section has correct defaults."""
        config = PurchasedGoodsServicesConfig()
        db = config.database

        assert db["host"] == "localhost"
        assert db["port"] == 5432
        assert db["database"] == "greenlang"
        assert db["pool_size"] == 20
        assert db["max_overflow"] == 40
        assert db["pool_timeout"] == 30

    def test_calculation_section_defaults(self):
        """Test calculation section has correct defaults."""
        config = PurchasedGoodsServicesConfig()
        calc = config.calculation

        assert calc["default_eeio_database"] == "EXIOBASE"
        assert calc["default_physical_ef_source"] == "IPCC_2006"
        assert calc["default_currency"] == "USD"
        assert calc["hybrid_threshold_high"] == Decimal("80.0")
        assert calc["hybrid_threshold_medium"] == Decimal("50.0")
        assert calc["monte_carlo_iterations"] == 10000
        assert calc["uncertainty_quantile_95"] == Decimal("1.96")

    def test_dqi_section_defaults(self):
        """Test DQI section has correct defaults."""
        config = PurchasedGoodsServicesConfig()
        dqi = config.dqi

        assert dqi["technological_weight"] == Decimal("0.25")
        assert dqi["temporal_weight"] == Decimal("0.20")
        assert dqi["geographical_weight"] == Decimal("0.20")
        assert dqi["completeness_weight"] == Decimal("0.20")
        assert dqi["reliability_weight"] == Decimal("0.15")
        # Weights should sum to 1.0
        total_weight = sum(Decimal(str(v)) for v in dqi.values() if isinstance(v, (int, float, Decimal)))
        assert total_weight == Decimal("1.0")

    def test_validation_section_defaults(self):
        """Test validation section has correct defaults."""
        config = PurchasedGoodsServicesConfig()
        val = config.validation

        assert val["max_emission_factor"] == Decimal("100.0")
        assert val["min_emission_factor"] == Decimal("0.0")
        assert val["max_spend_amount"] == Decimal("1000000000.0")
        assert val["min_data_quality_score"] == Decimal("1.0")
        assert val["require_supplier_verification"] is False

    def test_performance_section_defaults(self):
        """Test performance section has correct defaults."""
        config = PurchasedGoodsServicesConfig()
        perf = config.performance

        assert perf["batch_size"] == 1000
        assert perf["max_workers"] == 4
        assert perf["cache_ttl"] == 3600
        assert perf["enable_caching"] is True
        assert perf["async_processing"] is True

    def test_logging_section_defaults(self):
        """Test logging section has correct defaults."""
        config = PurchasedGoodsServicesConfig()
        log = config.logging

        assert log["level"] == "INFO"
        assert log["format"] == "json"
        assert log["enable_provenance_logging"] is True

    def test_compliance_section_defaults(self):
        """Test compliance section has correct defaults."""
        config = PurchasedGoodsServicesConfig()
        comp = config.compliance

        assert comp["default_frameworks"] == ["GHG_PROTOCOL", "ISO_14064_1"]
        assert comp["require_uncertainty_analysis"] is False
        assert comp["require_biogenic_separation"] is False

    def test_environment_variable_override_database(self):
        """Test environment variables override database config."""
        os.environ["GL_PGS_DATABASE_HOST"] = "prod-db.example.com"
        os.environ["GL_PGS_DATABASE_PORT"] = "5433"
        os.environ["GL_PGS_DATABASE_POOL_SIZE"] = "50"

        PurchasedGoodsServicesConfig.reset()
        config = PurchasedGoodsServicesConfig()

        assert config.database["host"] == "prod-db.example.com"
        assert config.database["port"] == 5433
        assert config.database["pool_size"] == 50

    def test_environment_variable_override_calculation(self):
        """Test environment variables override calculation config."""
        os.environ["GL_PGS_CALCULATION_DEFAULT_EEIO_DATABASE"] = "USEEIO"
        os.environ["GL_PGS_CALCULATION_MONTE_CARLO_ITERATIONS"] = "50000"

        PurchasedGoodsServicesConfig.reset()
        config = PurchasedGoodsServicesConfig()

        assert config.calculation["default_eeio_database"] == "USEEIO"
        assert config.calculation["monte_carlo_iterations"] == 50000

    def test_environment_variable_override_dqi(self):
        """Test environment variables override DQI weights."""
        os.environ["GL_PGS_DQI_TECHNOLOGICAL_WEIGHT"] = "0.30"
        os.environ["GL_PGS_DQI_TEMPORAL_WEIGHT"] = "0.25"

        PurchasedGoodsServicesConfig.reset()
        config = PurchasedGoodsServicesConfig()

        assert config.dqi["technological_weight"] == Decimal("0.30")
        assert config.dqi["temporal_weight"] == Decimal("0.25")

    def test_environment_variable_override_validation(self):
        """Test environment variables override validation config."""
        os.environ["GL_PGS_VALIDATION_MAX_EMISSION_FACTOR"] = "200.0"
        os.environ["GL_PGS_VALIDATION_REQUIRE_SUPPLIER_VERIFICATION"] = "true"

        PurchasedGoodsServicesConfig.reset()
        config = PurchasedGoodsServicesConfig()

        assert config.validation["max_emission_factor"] == Decimal("200.0")
        assert config.validation["require_supplier_verification"] is True

    def test_environment_variable_override_performance(self):
        """Test environment variables override performance config."""
        os.environ["GL_PGS_PERFORMANCE_BATCH_SIZE"] = "2000"
        os.environ["GL_PGS_PERFORMANCE_MAX_WORKERS"] = "8"
        os.environ["GL_PGS_PERFORMANCE_ENABLE_CACHING"] = "false"

        PurchasedGoodsServicesConfig.reset()
        config = PurchasedGoodsServicesConfig()

        assert config.performance["batch_size"] == 2000
        assert config.performance["max_workers"] == 8
        assert config.performance["enable_caching"] is False

    def test_environment_variable_boolean_parsing(self):
        """Test boolean environment variables are parsed correctly."""
        os.environ["GL_PGS_PERFORMANCE_ASYNC_PROCESSING"] = "false"
        os.environ["GL_PGS_LOGGING_ENABLE_PROVENANCE_LOGGING"] = "true"

        PurchasedGoodsServicesConfig.reset()
        config = PurchasedGoodsServicesConfig()

        assert config.performance["async_processing"] is False
        assert config.logging["enable_provenance_logging"] is True

    def test_reset_clears_singleton(self):
        """Test reset() clears the singleton instance."""
        config1 = PurchasedGoodsServicesConfig()
        config1_id = id(config1)

        PurchasedGoodsServicesConfig.reset()

        config2 = PurchasedGoodsServicesConfig()
        config2_id = id(config2)

        # After reset, should get a new instance
        assert config1_id != config2_id

    def test_section_accessor_database(self):
        """Test database section accessor."""
        config = PurchasedGoodsServicesConfig()
        db = config.database

        assert isinstance(db, dict)
        assert "host" in db
        assert "port" in db
        assert "database" in db

    def test_section_accessor_calculation(self):
        """Test calculation section accessor."""
        config = PurchasedGoodsServicesConfig()
        calc = config.calculation

        assert isinstance(calc, dict)
        assert "default_eeio_database" in calc
        assert "monte_carlo_iterations" in calc

    def test_section_accessor_dqi(self):
        """Test DQI section accessor."""
        config = PurchasedGoodsServicesConfig()
        dqi = config.dqi

        assert isinstance(dqi, dict)
        assert "technological_weight" in dqi
        assert "temporal_weight" in dqi

    def test_section_accessor_validation(self):
        """Test validation section accessor."""
        config = PurchasedGoodsServicesConfig()
        val = config.validation

        assert isinstance(val, dict)
        assert "max_emission_factor" in val
        assert "min_emission_factor" in val

    def test_section_accessor_performance(self):
        """Test performance section accessor."""
        config = PurchasedGoodsServicesConfig()
        perf = config.performance

        assert isinstance(perf, dict)
        assert "batch_size" in perf
        assert "max_workers" in perf

    def test_section_accessor_logging(self):
        """Test logging section accessor."""
        config = PurchasedGoodsServicesConfig()
        log = config.logging

        assert isinstance(log, dict)
        assert "level" in log
        assert "format" in log

    def test_section_accessor_compliance(self):
        """Test compliance section accessor."""
        config = PurchasedGoodsServicesConfig()
        comp = config.compliance

        assert isinstance(comp, dict)
        assert "default_frameworks" in comp
        assert isinstance(comp["default_frameworks"], list)
