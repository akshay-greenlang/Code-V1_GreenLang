# -*- coding: utf-8 -*-
"""
Test suite for employee_commuting.config - AGENT-MRV-020.

Tests configuration management for the Employee Commuting Agent
(GL-MRV-S3-007) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for all 15 config sections: GeneralConfig,
  DatabaseConfig, RedisConfig, CommuteModeConfig, TeleworkConfig,
  SurveyConfig, WorkingDaysConfig, SpendConfig, ComplianceConfig,
  EFSourceConfig, UncertaintyConfig, CacheConfig, APIConfig,
  ProvenanceConfig, MetricsConfig
- GL_EC_ environment variable loading (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety (concurrent get_config calls)
- Validation (invalid values raise ValueError)
- to_dict / from_dict round-trip
- Frozen dataclass immutability
- reset_config functionality
- EmployeeCommutingConfig master class and validate_all
- Cross-section validation rules

Author: GL-TestEngineer
Date: February 2026
"""

import os
import threading
from decimal import Decimal
from typing import Any, Dict
import pytest

from greenlang.agents.mrv.employee_commuting.config import (
    GeneralConfig,
    DatabaseConfig,
    RedisConfig,
    CommuteModeConfig,
    TeleworkConfig,
    SurveyConfig,
    WorkingDaysConfig,
    SpendConfig,
    ComplianceConfig,
    EFSourceConfig,
    UncertaintyConfig,
    CacheConfig,
    APIConfig,
    ProvenanceConfig,
    MetricsConfig,
    EmployeeCommutingConfig,
    get_config,
    set_config,
    reset_config,
    validate_config,
)


# ==============================================================================
# GENERAL CONFIGURATION TESTS
# ==============================================================================


class TestGeneralConfig:
    """Tests for GeneralConfig dataclass."""

    def test_defaults(self):
        """Test default general config values."""
        config = GeneralConfig()
        assert config.enabled is True
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.agent_id == "GL-MRV-S3-007"
        assert config.version == "1.0.0"
        assert config.table_prefix == "gl_ec_"
        assert config.max_retries == 3
        assert config.timeout == 300

    def test_agent_id(self):
        """Test default agent_id is GL-MRV-S3-007."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-007"

    def test_version_semver(self):
        """Test version follows SemVer format."""
        config = GeneralConfig()
        parts = config.version.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_table_prefix_ends_with_underscore(self):
        """Test table_prefix ends with underscore."""
        config = GeneralConfig()
        assert config.table_prefix.endswith("_")

    def test_frozen(self):
        """Test GeneralConfig is frozen (immutable)."""
        config = GeneralConfig()
        with pytest.raises(AttributeError):
            config.enabled = False

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = GeneralConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_log_level(self):
        """Test validate() rejects invalid log_level."""
        config = GeneralConfig(log_level="TRACE")
        with pytest.raises(ValueError, match="Invalid log_level"):
            config.validate()

    def test_validate_empty_agent_id(self):
        """Test validate() rejects empty agent_id."""
        config = GeneralConfig(agent_id="")
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            config.validate()

    def test_validate_invalid_version(self):
        """Test validate() rejects invalid version format."""
        config = GeneralConfig(version="1.0")
        with pytest.raises(ValueError, match="Invalid version"):
            config.validate()

    def test_validate_table_prefix_no_underscore(self):
        """Test validate() rejects table_prefix without trailing underscore."""
        config = GeneralConfig(table_prefix="gl_ec")
        with pytest.raises(ValueError, match="table_prefix must end with"):
            config.validate()

    def test_validate_max_retries_negative(self):
        """Test validate() rejects negative max_retries."""
        config = GeneralConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries"):
            config.validate()

    def test_validate_timeout_zero(self):
        """Test validate() rejects zero timeout."""
        config = GeneralConfig(timeout=0)
        with pytest.raises(ValueError, match="timeout"):
            config.validate()

    def test_to_dict(self):
        """Test to_dict() returns correct dictionary."""
        config = GeneralConfig()
        d = config.to_dict()
        assert d["agent_id"] == "GL-MRV-S3-007"
        assert d["version"] == "1.0.0"
        assert d["enabled"] is True

    def test_from_dict_roundtrip(self):
        """Test from_dict(to_dict()) round-trip preserves values."""
        config = GeneralConfig()
        d = config.to_dict()
        restored = GeneralConfig.from_dict(d)
        assert restored.agent_id == config.agent_id
        assert restored.version == config.version

    def test_from_env_defaults(self):
        """Test from_env() loads defaults when no env vars set."""
        config = GeneralConfig.from_env()
        assert config.agent_id == "GL-MRV-S3-007"

    def test_from_env_override(self, monkeypatch):
        """Test from_env() reads GL_EC_ environment variables."""
        monkeypatch.setenv("GL_EC_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GL_EC_MAX_RETRIES", "5")
        config = GeneralConfig.from_env()
        assert config.log_level == "DEBUG"
        assert config.max_retries == 5


# ==============================================================================
# DATABASE CONFIGURATION TESTS
# ==============================================================================


class TestDatabaseConfig:
    """Tests for DatabaseConfig dataclass."""

    def test_defaults(self):
        """Test default database config values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "greenlang"
        assert config.user == "greenlang"
        assert config.password == ""
        assert config.pool_min == 2
        assert config.pool_max == 10
        assert config.ssl is False
        assert config.timeout == 30
        assert config.schema == "employee_commuting_service"

    def test_frozen(self):
        """Test DatabaseConfig is frozen."""
        config = DatabaseConfig()
        with pytest.raises(AttributeError):
            config.host = "newhost"

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = DatabaseConfig()
        config.validate()

    def test_validate_empty_host(self):
        """Test validate() rejects empty host."""
        config = DatabaseConfig(host="")
        with pytest.raises(ValueError, match="host cannot be empty"):
            config.validate()

    def test_validate_invalid_port(self):
        """Test validate() rejects port out of range."""
        config = DatabaseConfig(port=0)
        with pytest.raises(ValueError, match="port"):
            config.validate()

    def test_validate_pool_min_exceeds_max(self):
        """Test validate() rejects pool_min > pool_max."""
        config = DatabaseConfig(pool_min=20, pool_max=5)
        with pytest.raises(ValueError, match="pool_min must be <= pool_max"):
            config.validate()

    def test_get_connection_url_no_ssl(self):
        """Test connection URL without SSL."""
        config = DatabaseConfig()
        url = config.get_connection_url()
        assert url.startswith("postgresql://")
        assert "sslmode" not in url

    def test_get_connection_url_with_ssl(self):
        """Test connection URL with SSL."""
        config = DatabaseConfig(ssl=True)
        url = config.get_connection_url()
        assert "sslmode=require" in url

    def test_to_dict_redacts_password(self):
        """Test to_dict() redacts password."""
        config = DatabaseConfig(password="secret123")
        d = config.to_dict()
        assert d["password"] == "***REDACTED***"

    def test_from_env_override(self, monkeypatch):
        """Test from_env() reads GL_EC_DB_ environment variables."""
        monkeypatch.setenv("GL_EC_DB_HOST", "db.example.com")
        monkeypatch.setenv("GL_EC_DB_PORT", "5433")
        config = DatabaseConfig.from_env()
        assert config.host == "db.example.com"
        assert config.port == 5433


# ==============================================================================
# REDIS CONFIGURATION TESTS
# ==============================================================================


class TestRedisConfig:
    """Tests for RedisConfig dataclass."""

    def test_defaults(self):
        """Test default Redis config values."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password == ""
        assert config.ssl is False
        assert config.ttl_seconds == 3600
        assert config.max_connections == 20
        assert config.prefix == "gl_ec:"

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = RedisConfig()
        config.validate()

    def test_validate_prefix_no_colon(self):
        """Test validate() rejects prefix without trailing colon."""
        config = RedisConfig(prefix="gl_ec")
        with pytest.raises(ValueError, match="prefix must end with ':'"):
            config.validate()

    def test_validate_invalid_db(self):
        """Test validate() rejects db number out of range."""
        config = RedisConfig(db=16)
        with pytest.raises(ValueError, match="db must be between"):
            config.validate()

    def test_get_connection_url_no_ssl(self):
        """Test Redis URL without SSL."""
        config = RedisConfig()
        url = config.get_connection_url()
        assert url.startswith("redis://")

    def test_get_connection_url_with_ssl(self):
        """Test Redis URL with SSL uses rediss:// scheme."""
        config = RedisConfig(ssl=True)
        url = config.get_connection_url()
        assert url.startswith("rediss://")

    def test_from_env_override(self, monkeypatch):
        """Test from_env() reads GL_EC_REDIS_ environment variables."""
        monkeypatch.setenv("GL_EC_REDIS_PORT", "6380")
        config = RedisConfig.from_env()
        assert config.port == 6380


# ==============================================================================
# COMMUTE MODE CONFIGURATION TESTS
# ==============================================================================


class TestCommuteModeConfig:
    """Tests for CommuteModeConfig dataclass."""

    def test_defaults(self):
        """Test default commute mode config values."""
        config = CommuteModeConfig()
        assert config.default_vehicle_type == "AVERAGE_CAR"
        assert config.default_fuel_type == "GASOLINE"
        assert config.max_distance_km == 500.0
        assert config.max_occupancy == 15
        assert config.include_wtt is True
        assert config.default_occupancy_sov == Decimal("1.0")
        assert config.default_occupancy_carpool == Decimal("2.5")
        assert config.default_occupancy_vanpool == Decimal("7.0")
        assert config.default_occupancy_bus == Decimal("30.0")
        assert config.default_gwp_source == "AR5"
        assert config.round_trip_factor == Decimal("2.0")
        assert config.e_bicycle_ef_fraction == Decimal("0.05")

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = CommuteModeConfig()
        config.validate()

    def test_validate_invalid_vehicle_type(self):
        """Test validate() rejects invalid vehicle type."""
        config = CommuteModeConfig(default_vehicle_type="INVALID")
        with pytest.raises(ValueError, match="Invalid default_vehicle_type"):
            config.validate()

    def test_validate_invalid_fuel_type(self):
        """Test validate() rejects invalid fuel type."""
        config = CommuteModeConfig(default_fuel_type="INVALID")
        with pytest.raises(ValueError, match="Invalid default_fuel_type"):
            config.validate()

    def test_validate_max_distance_zero(self):
        """Test validate() rejects zero max_distance_km."""
        config = CommuteModeConfig(max_distance_km=0)
        with pytest.raises(ValueError, match="max_distance_km"):
            config.validate()

    def test_validate_max_distance_over_2000(self):
        """Test validate() rejects max_distance_km over 2000."""
        config = CommuteModeConfig(max_distance_km=2001)
        with pytest.raises(ValueError, match="max_distance_km"):
            config.validate()

    def test_validate_sov_occupancy_out_of_range(self):
        """Test validate() rejects SOV occupancy below 1.0."""
        config = CommuteModeConfig(default_occupancy_sov=Decimal("0.5"))
        with pytest.raises(ValueError, match="default_occupancy_sov"):
            config.validate()

    def test_validate_invalid_gwp_source(self):
        """Test validate() rejects invalid GWP source."""
        config = CommuteModeConfig(default_gwp_source="AR2")
        with pytest.raises(ValueError, match="Invalid default_gwp_source"):
            config.validate()

    def test_frozen(self):
        """Test CommuteModeConfig is frozen."""
        config = CommuteModeConfig()
        with pytest.raises(AttributeError):
            config.max_distance_km = 1000.0

    def test_to_dict_from_dict_roundtrip(self):
        """Test to_dict/from_dict round-trip preserves values."""
        config = CommuteModeConfig()
        d = config.to_dict()
        restored = CommuteModeConfig.from_dict(d)
        assert restored.default_vehicle_type == config.default_vehicle_type
        assert restored.default_occupancy_carpool == config.default_occupancy_carpool

    def test_from_env_override(self, monkeypatch):
        """Test from_env() reads GL_EC_ environment variables."""
        monkeypatch.setenv("GL_EC_MAX_DISTANCE_KM", "750.0")
        monkeypatch.setenv("GL_EC_INCLUDE_WTT", "false")
        config = CommuteModeConfig.from_env()
        assert config.max_distance_km == 750.0
        assert config.include_wtt is False


# ==============================================================================
# TELEWORK CONFIGURATION TESTS
# ==============================================================================


class TestTeleworkConfig:
    """Tests for TeleworkConfig dataclass."""

    def test_defaults(self):
        """Test default telework config values."""
        config = TeleworkConfig()
        assert config.enabled is True
        assert config.default_daily_kwh == Decimal("4.0")
        assert config.laptop_kwh == Decimal("0.3")
        assert config.monitor_kwh == Decimal("0.1")
        assert config.heating_kwh == Decimal("3.5")
        assert config.cooling_kwh == Decimal("1.5")
        assert config.lighting_kwh == Decimal("0.2")
        assert config.internet_kwh == Decimal("0.1")
        assert config.seasonal_adjustment == "FULL_SEASONAL"
        assert config.default_grid_ef_source == "IEA"
        assert config.include_cooling is True
        assert config.summer_heating_fraction == Decimal("0.0")
        assert config.winter_cooling_fraction == Decimal("0.0")

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = TeleworkConfig()
        config.validate()

    def test_validate_negative_daily_kwh(self):
        """Test validate() rejects negative daily kWh."""
        config = TeleworkConfig(default_daily_kwh=Decimal("-1"))
        with pytest.raises(ValueError, match="default_daily_kwh must be >= 0"):
            config.validate()

    def test_validate_excessive_daily_kwh(self):
        """Test validate() rejects daily kWh exceeding 50."""
        config = TeleworkConfig(default_daily_kwh=Decimal("51"))
        with pytest.raises(ValueError, match="default_daily_kwh must be <= 50"):
            config.validate()

    def test_validate_invalid_seasonal_adjustment(self):
        """Test validate() rejects invalid seasonal adjustment."""
        config = TeleworkConfig(seasonal_adjustment="INVALID")
        with pytest.raises(ValueError, match="Invalid seasonal_adjustment"):
            config.validate()

    def test_validate_invalid_grid_ef_source(self):
        """Test validate() rejects invalid grid EF source."""
        config = TeleworkConfig(default_grid_ef_source="INVALID")
        with pytest.raises(ValueError, match="Invalid default_grid_ef_source"):
            config.validate()

    def test_validate_summer_heating_fraction_out_of_range(self):
        """Test validate() rejects summer_heating_fraction > 1.0."""
        config = TeleworkConfig(summer_heating_fraction=Decimal("1.5"))
        with pytest.raises(ValueError, match="summer_heating_fraction"):
            config.validate()

    def test_frozen(self):
        """Test TeleworkConfig is frozen."""
        config = TeleworkConfig()
        with pytest.raises(AttributeError):
            config.enabled = False

    def test_from_env_override(self, monkeypatch):
        """Test from_env() reads GL_EC_TELEWORK_ environment variables."""
        monkeypatch.setenv("GL_EC_TELEWORK_DAILY_KWH", "5.5")
        monkeypatch.setenv("GL_EC_TELEWORK_ENABLED", "false")
        config = TeleworkConfig.from_env()
        assert config.default_daily_kwh == Decimal("5.5")
        assert config.enabled is False


# ==============================================================================
# SURVEY CONFIGURATION TESTS
# ==============================================================================


class TestSurveyConfig:
    """Tests for SurveyConfig dataclass."""

    def test_defaults(self):
        """Test default survey config values."""
        config = SurveyConfig()
        assert config.min_response_rate == Decimal("0.1")
        assert config.min_sample_size == 30
        assert config.confidence_level == Decimal("0.95")
        assert config.z_score == Decimal("1.96")
        assert config.max_extrapolation_factor == Decimal("20.0")
        assert config.default_survey_method == "RANDOM_SAMPLE"
        assert config.margin_of_error == Decimal("0.05")
        assert config.response_weighting is True
        assert config.outlier_removal is True
        assert config.outlier_z_threshold == Decimal("3.0")
        assert config.allow_partial_responses is True
        assert config.min_completeness_rate == Decimal("0.7")

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = SurveyConfig()
        config.validate()

    def test_validate_min_response_rate_too_low(self):
        """Test validate() rejects response rate below 0.01."""
        config = SurveyConfig(min_response_rate=Decimal("0.001"))
        with pytest.raises(ValueError, match="min_response_rate"):
            config.validate()

    def test_validate_invalid_survey_method(self):
        """Test validate() rejects invalid survey method."""
        config = SurveyConfig(default_survey_method="INVALID")
        with pytest.raises(ValueError, match="Invalid default_survey_method"):
            config.validate()

    def test_validate_z_score_mismatch(self):
        """Test validate() rejects z_score not matching confidence_level."""
        config = SurveyConfig(
            confidence_level=Decimal("0.95"),
            z_score=Decimal("2.576"),
        )
        with pytest.raises(ValueError, match="z_score.*does not correspond"):
            config.validate()

    def test_validate_margin_of_error_out_of_range(self):
        """Test validate() rejects margin_of_error out of range."""
        config = SurveyConfig(margin_of_error=Decimal("0.6"))
        with pytest.raises(ValueError, match="margin_of_error"):
            config.validate()

    def test_frozen(self):
        """Test SurveyConfig is frozen."""
        config = SurveyConfig()
        with pytest.raises(AttributeError):
            config.min_sample_size = 50

    def test_from_env_override(self, monkeypatch):
        """Test from_env() reads GL_EC_SURVEY_ environment variables."""
        monkeypatch.setenv("GL_EC_SURVEY_MIN_SAMPLE_SIZE", "50")
        config = SurveyConfig.from_env()
        assert config.min_sample_size == 50


# ==============================================================================
# WORKING DAYS CONFIGURATION TESTS
# ==============================================================================


class TestWorkingDaysConfig:
    """Tests for WorkingDaysConfig dataclass."""

    def test_defaults(self):
        """Test default working days config values."""
        config = WorkingDaysConfig()
        assert config.default_region == "GLOBAL"
        assert config.default_working_days == 230
        assert config.include_holidays is True
        assert config.include_pto is True
        assert config.include_sick is True
        assert config.default_holidays == 10
        assert config.default_pto_days == 15
        assert config.default_sick_days == 5
        assert config.min_working_days == 50
        assert config.max_working_days == 365
        assert config.part_time_adjustment is True
        assert config.default_part_time_fraction == Decimal("0.5")

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = WorkingDaysConfig()
        config.validate()

    def test_validate_invalid_region(self):
        """Test validate() rejects invalid region."""
        config = WorkingDaysConfig(default_region="INVALID")
        with pytest.raises(ValueError, match="Invalid default_region"):
            config.validate()

    def test_validate_min_exceeds_max(self):
        """Test validate() rejects min_working_days > max_working_days."""
        config = WorkingDaysConfig(min_working_days=300, max_working_days=200)
        with pytest.raises(ValueError, match="min_working_days must be <= max_working_days"):
            config.validate()

    def test_validate_default_days_below_min(self):
        """Test validate() rejects default_working_days below min."""
        config = WorkingDaysConfig(
            default_working_days=40,
            min_working_days=50,
        )
        with pytest.raises(ValueError, match="default_working_days.*must be >= min_working_days"):
            config.validate()

    def test_get_effective_working_days(self):
        """Test get_effective_working_days() calculation."""
        config = WorkingDaysConfig()
        effective = config.get_effective_working_days()
        # 230 - 10 - 15 - 5 = 200
        assert effective == 200

    def test_get_effective_working_days_no_deductions(self):
        """Test get_effective_working_days() without deductions."""
        config = WorkingDaysConfig(
            include_holidays=False,
            include_pto=False,
            include_sick=False,
        )
        assert config.get_effective_working_days() == 230

    def test_frozen(self):
        """Test WorkingDaysConfig is frozen."""
        config = WorkingDaysConfig()
        with pytest.raises(AttributeError):
            config.default_working_days = 250


# ==============================================================================
# SPEND CONFIGURATION TESTS
# ==============================================================================


class TestSpendConfig:
    """Tests for SpendConfig dataclass."""

    def test_defaults(self):
        """Test default spend config values."""
        config = SpendConfig()
        assert config.default_currency == "USD"
        assert config.base_year == 2021
        assert config.eeio_source == "USEEIO"
        assert config.margin_removal_rate == Decimal("0.15")
        assert config.enable_cpi_deflation is True
        assert config.enable_currency_conversion is True
        assert config.transport_sector_code == "485000"
        assert config.purchaser_price_adjustment is True

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = SpendConfig()
        config.validate()

    def test_validate_invalid_currency(self):
        """Test validate() rejects invalid currency."""
        config = SpendConfig(default_currency="XYZ")
        with pytest.raises(ValueError, match="Invalid default_currency"):
            config.validate()

    def test_validate_invalid_eeio_source(self):
        """Test validate() rejects invalid EEIO source."""
        config = SpendConfig(eeio_source="INVALID")
        with pytest.raises(ValueError, match="Invalid eeio_source"):
            config.validate()

    def test_validate_margin_out_of_range(self):
        """Test validate() rejects margin_removal_rate > 0.5."""
        config = SpendConfig(margin_removal_rate=Decimal("0.6"))
        with pytest.raises(ValueError, match="margin_removal_rate"):
            config.validate()

    def test_validate_base_year_too_old(self):
        """Test validate() rejects base_year before 2000."""
        config = SpendConfig(base_year=1999)
        with pytest.raises(ValueError, match="base_year"):
            config.validate()

    def test_frozen(self):
        """Test SpendConfig is frozen."""
        config = SpendConfig()
        with pytest.raises(AttributeError):
            config.base_year = 2022


# ==============================================================================
# COMPLIANCE CONFIGURATION TESTS
# ==============================================================================


class TestComplianceConfig:
    """Tests for ComplianceConfig dataclass."""

    def test_defaults(self):
        """Test default compliance config values."""
        config = ComplianceConfig()
        assert "GHG_PROTOCOL_SCOPE3" in config.compliance_frameworks
        assert config.strict_mode is False
        assert config.telework_disclosure_required is True
        assert config.mode_share_required is True
        assert config.double_counting_check is True
        assert config.boundary_enforcement is True
        assert config.data_quality_required is True
        assert config.minimum_dqi_score == Decimal("2.0")

    def test_get_frameworks(self):
        """Test get_frameworks() returns list of 7 frameworks."""
        config = ComplianceConfig()
        frameworks = config.get_frameworks()
        assert len(frameworks) == 7
        assert "GHG_PROTOCOL_SCOPE3" in frameworks
        assert "ISO_14064" in frameworks
        assert "CSRD_ESRS_E1" in frameworks
        assert "CDP" in frameworks
        assert "SBTI" in frameworks
        assert "GRI_305" in frameworks
        assert "EPA_CCL" in frameworks

    def test_has_framework_true(self):
        """Test has_framework() returns True for enabled framework."""
        config = ComplianceConfig()
        assert config.has_framework("GHG_PROTOCOL_SCOPE3") is True

    def test_has_framework_false(self):
        """Test has_framework() returns False for non-enabled framework."""
        config = ComplianceConfig(compliance_frameworks="GHG_PROTOCOL_SCOPE3")
        assert config.has_framework("CDP") is False

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = ComplianceConfig()
        config.validate()

    def test_validate_empty_frameworks(self):
        """Test validate() rejects empty frameworks string."""
        config = ComplianceConfig(compliance_frameworks="")
        with pytest.raises(ValueError, match="At least one compliance framework"):
            config.validate()

    def test_validate_invalid_framework(self):
        """Test validate() rejects invalid framework name."""
        config = ComplianceConfig(compliance_frameworks="INVALID_FRAMEWORK")
        with pytest.raises(ValueError, match="Invalid framework"):
            config.validate()

    def test_validate_dqi_score_out_of_range(self):
        """Test validate() rejects minimum_dqi_score out of range."""
        config = ComplianceConfig(minimum_dqi_score=Decimal("6.0"))
        with pytest.raises(ValueError, match="minimum_dqi_score"):
            config.validate()

    def test_frozen(self):
        """Test ComplianceConfig is frozen."""
        config = ComplianceConfig()
        with pytest.raises(AttributeError):
            config.strict_mode = True


# ==============================================================================
# EF SOURCE CONFIGURATION TESTS
# ==============================================================================


class TestEFSourceConfig:
    """Tests for EFSourceConfig dataclass."""

    def test_defaults(self):
        """Test default EF source config values."""
        config = EFSourceConfig()
        assert config.hierarchy == "EMPLOYEE,DEFRA,EPA,IEA,CENSUS,EEIO"
        assert config.default_source == "DEFRA"
        assert config.allow_custom is True
        assert config.custom_ef_path is None
        assert config.cache_ef_lookups is True
        assert config.ef_year == 2024
        assert config.wtt_source == "DEFRA"
        assert config.fallback_enabled is True

    def test_get_hierarchy(self):
        """Test get_hierarchy() returns ordered list."""
        config = EFSourceConfig()
        hierarchy = config.get_hierarchy()
        assert hierarchy == ["EMPLOYEE", "DEFRA", "EPA", "IEA", "CENSUS", "EEIO"]

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = EFSourceConfig()
        config.validate()

    def test_validate_invalid_source_in_hierarchy(self):
        """Test validate() rejects invalid source in hierarchy."""
        config = EFSourceConfig(hierarchy="EMPLOYEE,INVALID")
        with pytest.raises(ValueError, match="Invalid EF source"):
            config.validate()

    def test_validate_custom_default_without_path(self):
        """Test validate() rejects CUSTOM default_source without path."""
        config = EFSourceConfig(default_source="CUSTOM", custom_ef_path=None)
        with pytest.raises(ValueError, match="custom_ef_path must be set"):
            config.validate()

    def test_validate_ef_year_out_of_range(self):
        """Test validate() rejects ef_year out of range."""
        config = EFSourceConfig(ef_year=1999)
        with pytest.raises(ValueError, match="ef_year must be between"):
            config.validate()

    def test_validate_invalid_wtt_source(self):
        """Test validate() rejects invalid wtt_source."""
        config = EFSourceConfig(wtt_source="INVALID")
        with pytest.raises(ValueError, match="Invalid wtt_source"):
            config.validate()


# ==============================================================================
# UNCERTAINTY CONFIGURATION TESTS
# ==============================================================================


class TestUncertaintyConfig:
    """Tests for UncertaintyConfig dataclass."""

    def test_defaults(self):
        """Test default uncertainty config values."""
        config = UncertaintyConfig()
        assert config.method == "MONTE_CARLO"
        assert config.iterations == 10000
        assert config.confidence_level == Decimal("0.95")
        assert config.seed == 42
        assert config.include_ef_uncertainty is True
        assert config.include_activity_uncertainty is True
        assert config.include_survey_uncertainty is True
        assert config.distribution_type == "LOGNORMAL"

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = UncertaintyConfig()
        config.validate()

    def test_validate_invalid_method(self):
        """Test validate() rejects invalid method."""
        config = UncertaintyConfig(method="INVALID")
        with pytest.raises(ValueError, match="Invalid method"):
            config.validate()

    def test_validate_iterations_too_low(self):
        """Test validate() rejects iterations below 100."""
        config = UncertaintyConfig(iterations=50)
        with pytest.raises(ValueError, match="iterations must be between"):
            config.validate()

    def test_validate_invalid_distribution(self):
        """Test validate() rejects invalid distribution type."""
        config = UncertaintyConfig(distribution_type="INVALID")
        with pytest.raises(ValueError, match="Invalid distribution_type"):
            config.validate()

    def test_validate_negative_seed(self):
        """Test validate() rejects negative seed."""
        config = UncertaintyConfig(seed=-1)
        with pytest.raises(ValueError, match="seed must be >= 0"):
            config.validate()


# ==============================================================================
# CACHE CONFIGURATION TESTS
# ==============================================================================


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_defaults(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.ttl == 3600
        assert config.max_size == 10000
        assert config.warm_on_startup is True
        assert config.cache_ef_lookups is True
        assert config.cache_calculations is True
        assert config.cache_survey_data is True
        assert config.eviction_policy == "LRU"

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = CacheConfig()
        config.validate()

    def test_validate_ttl_zero(self):
        """Test validate() rejects zero TTL."""
        config = CacheConfig(ttl=0)
        with pytest.raises(ValueError, match="ttl must be between"):
            config.validate()

    def test_validate_invalid_eviction_policy(self):
        """Test validate() rejects invalid eviction policy."""
        config = CacheConfig(eviction_policy="RANDOM")
        with pytest.raises(ValueError, match="Invalid eviction_policy"):
            config.validate()


# ==============================================================================
# API CONFIGURATION TESTS
# ==============================================================================


class TestAPIConfig:
    """Tests for APIConfig dataclass."""

    def test_defaults(self):
        """Test default API config values."""
        config = APIConfig()
        assert config.prefix == "/api/v1/employee-commuting"
        assert config.page_size == 50
        assert config.max_page_size == 500
        assert config.rate_limit == 100
        assert config.timeout == 300
        assert config.enable_bulk is True
        assert config.max_bulk_size == 1000
        assert config.cors_enabled is True
        assert config.cors_origins == "*"

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = APIConfig()
        config.validate()

    def test_validate_empty_prefix(self):
        """Test validate() rejects empty prefix."""
        config = APIConfig(prefix="")
        with pytest.raises(ValueError, match="prefix cannot be empty"):
            config.validate()

    def test_validate_prefix_no_slash(self):
        """Test validate() rejects prefix not starting with /."""
        config = APIConfig(prefix="api/v1")
        with pytest.raises(ValueError, match="prefix must start with"):
            config.validate()

    def test_validate_page_size_exceeds_max(self):
        """Test validate() rejects page_size > max_page_size."""
        config = APIConfig(page_size=600, max_page_size=500)
        with pytest.raises(ValueError, match="page_size must be <= max_page_size"):
            config.validate()

    def test_get_cors_origins(self):
        """Test get_cors_origins() parses origins string."""
        config = APIConfig(cors_origins="http://localhost,https://app.example.com")
        origins = config.get_cors_origins()
        assert len(origins) == 2
        assert "http://localhost" in origins


# ==============================================================================
# PROVENANCE CONFIGURATION TESTS
# ==============================================================================


class TestProvenanceConfig:
    """Tests for ProvenanceConfig dataclass."""

    def test_defaults(self):
        """Test default provenance config values."""
        config = ProvenanceConfig()
        assert config.enabled is True
        assert config.hash_algorithm == "sha256"
        assert config.chain_validation is True
        assert config.store_intermediates is True
        assert config.include_config_hash is True
        assert config.include_ef_hash is True
        assert config.include_survey_hash is True
        assert config.retention_days == 365

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = ProvenanceConfig()
        config.validate()

    def test_validate_invalid_algorithm(self):
        """Test validate() rejects invalid hash algorithm."""
        config = ProvenanceConfig(hash_algorithm="md5")
        with pytest.raises(ValueError, match="Invalid hash_algorithm"):
            config.validate()

    def test_validate_retention_days_out_of_range(self):
        """Test validate() rejects retention_days out of range."""
        config = ProvenanceConfig(retention_days=0)
        with pytest.raises(ValueError, match="retention_days must be between"):
            config.validate()


# ==============================================================================
# METRICS CONFIGURATION TESTS
# ==============================================================================


class TestMetricsConfig:
    """Tests for MetricsConfig dataclass."""

    def test_defaults(self):
        """Test default metrics config values."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.prefix == "gl_ec_"
        assert config.collect_histograms is True
        assert config.collect_per_mode is True
        assert config.collect_per_framework is True
        assert config.collection_interval == 60
        assert config.include_survey_metrics is True

    def test_validate_passes_for_defaults(self):
        """Test validate() passes for default values."""
        config = MetricsConfig()
        config.validate()

    def test_validate_prefix_no_underscore(self):
        """Test validate() rejects prefix without trailing underscore."""
        config = MetricsConfig(prefix="gl_ec")
        with pytest.raises(ValueError, match="prefix must end with"):
            config.validate()

    def test_get_buckets(self):
        """Test get_buckets() parses histogram buckets string."""
        config = MetricsConfig()
        buckets = config.get_buckets()
        assert len(buckets) == 9
        assert buckets[0] == 0.01
        assert buckets[-1] == 10.0

    def test_validate_unsorted_buckets(self):
        """Test validate() rejects unsorted histogram buckets."""
        config = MetricsConfig(histogram_buckets="10.0,0.5,1.0")
        with pytest.raises(ValueError, match="strictly ascending"):
            config.validate()


# ==============================================================================
# MASTER CONFIGURATION TESTS
# ==============================================================================


class TestEmployeeCommutingConfig:
    """Tests for EmployeeCommutingConfig master dataclass."""

    def test_defaults_create_all_15_sections(self):
        """Test default config creates all 15 sections."""
        config = EmployeeCommutingConfig()
        assert isinstance(config.general, GeneralConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.commute_mode, CommuteModeConfig)
        assert isinstance(config.telework, TeleworkConfig)
        assert isinstance(config.survey, SurveyConfig)
        assert isinstance(config.working_days, WorkingDaysConfig)
        assert isinstance(config.spend, SpendConfig)
        assert isinstance(config.compliance, ComplianceConfig)
        assert isinstance(config.ef_source, EFSourceConfig)
        assert isinstance(config.uncertainty, UncertaintyConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.provenance, ProvenanceConfig)
        assert isinstance(config.metrics, MetricsConfig)

    def test_validate_all_passes_for_defaults(self):
        """Test validate_all() passes for default configuration."""
        config = EmployeeCommutingConfig()
        config.validate_all()

    def test_validate_all_cross_section_metrics_prefix(self):
        """Test validate_all() enforces metrics.prefix == general.table_prefix."""
        config = EmployeeCommutingConfig(
            metrics=MetricsConfig(prefix="wrong_prefix_"),
        )
        with pytest.raises(ValueError, match="metrics.prefix.*must match.*general.table_prefix"):
            config.validate_all()

    def test_validate_all_cross_section_cache_ttl(self):
        """Test validate_all() enforces cache.ttl <= redis.ttl_seconds."""
        config = EmployeeCommutingConfig(
            cache=CacheConfig(ttl=7200),
            redis=RedisConfig(ttl_seconds=3600),
        )
        with pytest.raises(ValueError, match="cache.ttl.*should not exceed.*redis.ttl_seconds"):
            config.validate_all()

    def test_validate_all_cross_section_ef_year(self):
        """Test validate_all() enforces ef_source.ef_year >= spend.base_year."""
        config = EmployeeCommutingConfig(
            ef_source=EFSourceConfig(ef_year=2020),
            spend=SpendConfig(base_year=2021),
        )
        with pytest.raises(ValueError, match="ef_source.ef_year.*should not be earlier"):
            config.validate_all()

    def test_to_dict_has_all_15_sections(self):
        """Test to_dict() includes all 15 section keys."""
        config = EmployeeCommutingConfig()
        d = config.to_dict()
        expected_keys = {
            "general", "database", "redis", "commute_mode", "telework",
            "survey", "working_days", "spend", "compliance", "ef_source",
            "uncertainty", "cache", "api", "provenance", "metrics",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_roundtrip(self):
        """Test from_dict(to_dict()) round-trip preserves values."""
        config = EmployeeCommutingConfig()
        d = config.to_dict()
        restored = EmployeeCommutingConfig.from_dict(d)
        assert restored.general.agent_id == config.general.agent_id
        assert restored.commute_mode.max_distance_km == config.commute_mode.max_distance_km

    def test_from_env_creates_valid_config(self):
        """Test from_env() creates a valid config from environment."""
        config = EmployeeCommutingConfig.from_env()
        assert config.general.agent_id == "GL-MRV-S3-007"

    def test_frozen(self):
        """Test EmployeeCommutingConfig is frozen."""
        config = EmployeeCommutingConfig()
        with pytest.raises(AttributeError):
            config.general = GeneralConfig(agent_id="changed")


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Tests for thread-safe singleton configuration pattern."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_returns_instance(self):
        """Test get_config() returns an EmployeeCommutingConfig instance."""
        config = get_config()
        assert isinstance(config, EmployeeCommutingConfig)

    def test_get_config_singleton(self):
        """Test get_config() returns same instance on repeated calls."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config_clears_singleton(self):
        """Test reset_config() forces reload on next get_config()."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        # After reset, a new instance is created (may or may not be `is`)
        # but both should be valid
        assert isinstance(config2, EmployeeCommutingConfig)
        assert config2.general.agent_id == "GL-MRV-S3-007"

    def test_set_config_replaces_singleton(self):
        """Test set_config() replaces the singleton instance."""
        custom = EmployeeCommutingConfig(
            general=GeneralConfig(log_level="DEBUG"),
        )
        set_config(custom)
        config = get_config()
        assert config.general.log_level == "DEBUG"

    def test_thread_safety(self):
        """Test get_config() is thread-safe under concurrent access."""
        results = []
        errors = []

        def worker():
            try:
                config = get_config()
                results.append(config)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All threads should get the same singleton
        for r in results[1:]:
            assert r is results[0]


# ==============================================================================
# ENVIRONMENT VARIABLE OVERRIDE TESTS
# ==============================================================================


class TestEnvironmentVariableOverrides:
    """Tests for GL_EC_ environment variable overrides."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_general_env_override(self, monkeypatch):
        """Test GL_EC_ENABLED env var override."""
        monkeypatch.setenv("GL_EC_ENABLED", "false")
        config = GeneralConfig.from_env()
        assert config.enabled is False

    def test_database_env_override(self, monkeypatch):
        """Test GL_EC_DB_* env var overrides."""
        monkeypatch.setenv("GL_EC_DB_HOST", "production-db.example.com")
        monkeypatch.setenv("GL_EC_DB_SSL", "true")
        config = DatabaseConfig.from_env()
        assert config.host == "production-db.example.com"
        assert config.ssl is True

    def test_telework_env_override(self, monkeypatch):
        """Test GL_EC_TELEWORK_* env var overrides."""
        monkeypatch.setenv("GL_EC_TELEWORK_HEATING_KWH", "5.0")
        config = TeleworkConfig.from_env()
        assert config.heating_kwh == Decimal("5.0")

    def test_commute_mode_env_override(self, monkeypatch):
        """Test GL_EC_DEFAULT_VEHICLE_TYPE env var override."""
        monkeypatch.setenv("GL_EC_DEFAULT_VEHICLE_TYPE", "HYBRID")
        config = CommuteModeConfig.from_env()
        assert config.default_vehicle_type == "HYBRID"

    def test_compliance_env_override(self, monkeypatch):
        """Test GL_EC_COMPLIANCE_STRICT_MODE env var override."""
        monkeypatch.setenv("GL_EC_COMPLIANCE_STRICT_MODE", "true")
        config = ComplianceConfig.from_env()
        assert config.strict_mode is True

    def test_uncertainty_env_override(self, monkeypatch):
        """Test GL_EC_UNCERTAINTY_ITERATIONS env var override."""
        monkeypatch.setenv("GL_EC_UNCERTAINTY_ITERATIONS", "50000")
        config = UncertaintyConfig.from_env()
        assert config.iterations == 50000


# ==============================================================================
# VALIDATE CONFIG UTILITY TESTS
# ==============================================================================


class TestValidateConfigUtility:
    """Tests for validate_config() utility function."""

    def test_valid_config_returns_empty_errors(self):
        """Test validate_config() returns empty list for valid config."""
        config = EmployeeCommutingConfig()
        errors = validate_config(config)
        assert errors == []

    def test_invalid_config_collects_errors(self):
        """Test validate_config() collects all errors without raising."""
        config = EmployeeCommutingConfig(
            general=GeneralConfig(log_level="TRACE"),
            metrics=MetricsConfig(prefix="wrong_"),
        )
        errors = validate_config(config)
        assert len(errors) > 0

    def test_cross_section_error_detected(self):
        """Test validate_config() detects cross-section violations."""
        config = EmployeeCommutingConfig(
            cache=CacheConfig(ttl=7200),
            redis=RedisConfig(ttl_seconds=3600),
        )
        errors = validate_config(config)
        cross_errors = [e for e in errors if "cross-section" in e]
        assert len(cross_errors) > 0
