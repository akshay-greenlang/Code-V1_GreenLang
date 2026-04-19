# -*- coding: utf-8 -*-
"""
GL-010 Configuration Tests
==========================

Unit tests for GL-010 configuration module.
Tests all configuration schemas, validation, and factory functions.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_010_emissions_guardian.config import (
    GL010Config,
    EmissionsMonitoringConfig,
    CEMSIntegrationConfig,
    CEMSAnalyzerConfig,
    RATAConfig,
    CarbonTradingConfig,
    LDARConfig,
    ReportingConfig,
    ExplainabilityConfig,
    ProvenanceConfig,
    PermitLimitsConfig,
    AlertThresholdsConfig,
    SHAPConfig,
    LIMEConfig,
    PollutantType,
    FuelType,
    MonitoringMethod,
    RegulatoryProgram,
    AlertSeverity,
    ExplainabilityMethod,
    create_default_config,
    create_cems_config,
)


class TestPermitLimitsConfig:
    """Tests for permit limits configuration."""

    def test_default_permit_limits(self):
        """Test default permit limits."""
        limits = PermitLimitsConfig()
        assert limits.opacity_pct == 20.0
        assert limits.nox_lb_hr is None
        assert limits.co2_lb_hr is None

    def test_custom_permit_limits(self):
        """Test custom permit limits."""
        limits = PermitLimitsConfig(
            nox_lb_hr=25.0,
            co2_lb_hr=50000.0,
            so2_lb_hr=10.0,
            co_lb_hr=100.0,
        )
        assert limits.nox_lb_hr == 25.0
        assert limits.co2_lb_hr == 50000.0

    def test_permit_limits_bounds(self):
        """Test permit limits bounds validation."""
        with pytest.raises(ValidationError):
            PermitLimitsConfig(nox_lb_hr=-10.0)

        with pytest.raises(ValidationError):
            PermitLimitsConfig(opacity_pct=150.0)


class TestAlertThresholdsConfig:
    """Tests for alert thresholds configuration."""

    def test_default_thresholds(self, default_alert_thresholds):
        """Test default threshold values."""
        t = default_alert_thresholds
        assert t.warning_threshold_pct == 80.0
        assert t.alarm_threshold_pct == 90.0
        assert t.critical_threshold_pct == 95.0

    def test_threshold_ordering(self, default_alert_thresholds):
        """Test that thresholds are properly ordered."""
        t = default_alert_thresholds
        assert t.warning_threshold_pct < t.alarm_threshold_pct < t.critical_threshold_pct

    def test_threshold_bounds(self):
        """Test threshold bounds validation."""
        with pytest.raises(ValidationError):
            AlertThresholdsConfig(warning_threshold_pct=40.0)

        with pytest.raises(ValidationError):
            AlertThresholdsConfig(critical_threshold_pct=120.0)


class TestEmissionsMonitoringConfig:
    """Tests for emissions monitoring configuration."""

    def test_required_source_id(self):
        """Test that source_id is required."""
        with pytest.raises(ValidationError):
            EmissionsMonitoringConfig()

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = EmissionsMonitoringConfig(source_id="STACK-001")
        assert config.source_id == "STACK-001"
        assert config.monitoring_method == MonitoringMethod.CEMS
        assert config.primary_fuel == FuelType.NATURAL_GAS

    def test_full_config(self, default_monitoring_config):
        """Test full monitoring configuration."""
        config = default_monitoring_config
        assert config.source_id == "STACK-001"
        assert config.source_name == "Main Boiler Stack"
        assert len(config.monitored_pollutants) == 4

    def test_default_pollutants(self):
        """Test default monitored pollutants."""
        config = EmissionsMonitoringConfig(source_id="TEST")
        assert PollutantType.CO2 in config.monitored_pollutants
        assert PollutantType.NOX in config.monitored_pollutants

    def test_sampling_interval_bounds(self):
        """Test sampling interval bounds."""
        with pytest.raises(ValidationError):
            EmissionsMonitoringConfig(source_id="TEST", sampling_interval_s=0)

        with pytest.raises(ValidationError):
            EmissionsMonitoringConfig(source_id="TEST", sampling_interval_s=5000)

    def test_data_availability_bounds(self):
        """Test data availability bounds."""
        with pytest.raises(ValidationError):
            EmissionsMonitoringConfig(source_id="TEST", min_data_availability_pct=50.0)


class TestCEMSAnalyzerConfig:
    """Tests for CEMS analyzer configuration."""

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            CEMSAnalyzerConfig()

    def test_valid_analyzer(self, default_cems_analyzer_nox):
        """Test valid analyzer configuration."""
        config = default_cems_analyzer_nox
        assert config.analyzer_id == "NOX-01"
        assert config.pollutant == PollutantType.NOX
        assert config.span_value == 500.0

    def test_response_time_bounds(self):
        """Test response time bounds."""
        with pytest.raises(ValidationError):
            CEMSAnalyzerConfig(
                analyzer_id="TEST",
                pollutant=PollutantType.NOX,
                span_value=100,
                measurement_range_high=100,
                cal_gas_high_ppm=80,
                response_time_s=0,
            )


class TestCEMSIntegrationConfig:
    """Tests for CEMS integration configuration."""

    def test_default_config(self):
        """Test default CEMS configuration."""
        config = CEMSIntegrationConfig()
        assert config.enabled is False
        assert config.unit_id == "CEMS-001"
        assert config.daily_calibration_enabled is True

    def test_enabled_config(self, default_cems_config):
        """Test enabled CEMS configuration."""
        config = default_cems_config
        assert config.enabled is True
        assert len(config.analyzers) == 2

    def test_calibration_drift_bounds(self):
        """Test calibration drift bounds."""
        with pytest.raises(ValidationError):
            CEMSIntegrationConfig(calibration_drift_limit_pct=15.0)

    def test_rata_frequency_bounds(self):
        """Test RATA frequency bounds."""
        with pytest.raises(ValidationError):
            CEMSIntegrationConfig(rata_frequency_quarters=5)


class TestRATAConfig:
    """Tests for RATA configuration."""

    def test_default_config(self, default_rata_config):
        """Test default RATA configuration."""
        config = default_rata_config
        assert config.enabled is True
        assert config.base_frequency == "quarterly"
        assert config.min_test_runs == 9

    def test_min_test_runs_bounds(self):
        """Test minimum test runs bounds."""
        with pytest.raises(ValidationError):
            RATAConfig(min_test_runs=5)

    def test_relative_accuracy_bounds(self):
        """Test relative accuracy limit bounds."""
        with pytest.raises(ValidationError):
            RATAConfig(relative_accuracy_limit_pct=3.0)


class TestCarbonTradingConfig:
    """Tests for carbon trading configuration."""

    def test_default_config(self):
        """Test default trading configuration."""
        config = CarbonTradingConfig()
        assert config.enabled is False
        assert config.primary_market == "voluntary"

    def test_enabled_config(self, default_trading_config):
        """Test enabled trading configuration."""
        config = default_trading_config
        assert config.enabled is True
        assert config.entity_id == "ENTITY-001"
        assert config.minimum_offset_quality_score == 60.0

    def test_offset_usage_bounds(self):
        """Test offset usage limit bounds."""
        with pytest.raises(ValidationError):
            CarbonTradingConfig(offset_usage_limit_pct=30.0)


class TestLDARConfig:
    """Tests for LDAR configuration."""

    def test_default_config(self):
        """Test default LDAR configuration."""
        config = LDARConfig()
        assert config.enabled is False
        assert config.valve_leak_threshold_ppm == 500

    def test_repair_timeline_bounds(self):
        """Test repair timeline bounds."""
        with pytest.raises(ValidationError):
            LDARConfig(first_attempt_repair_days=20)


class TestReportingConfig:
    """Tests for reporting configuration."""

    def test_default_config(self, default_reporting_config):
        """Test default reporting configuration."""
        config = default_reporting_config
        assert config.enabled is True
        assert config.facility_name == "Test Facility"
        assert config.part98_reporting_enabled is True

    def test_retention_bounds(self):
        """Test report retention bounds."""
        with pytest.raises(ValidationError):
            ReportingConfig(report_retention_years=2)


class TestExplainabilityConfig:
    """Tests for explainability configuration."""

    def test_default_config(self, default_explainability_config):
        """Test default explainability configuration."""
        config = default_explainability_config
        assert config.enabled is True
        assert config.shap.enabled is True
        assert config.lime.enabled is True

    def test_shap_config(self):
        """Test SHAP configuration."""
        shap = SHAPConfig(n_samples=200, background_samples=100)
        assert shap.n_samples == 200
        assert shap.method == "kernel"

    def test_lime_config(self):
        """Test LIME configuration."""
        lime = LIMEConfig(num_features=15)
        assert lime.num_features == 15
        assert lime.discretize_continuous is True


class TestProvenanceConfig:
    """Tests for provenance configuration."""

    def test_default_config(self, default_provenance_config):
        """Test default provenance configuration."""
        config = default_provenance_config
        assert config.enabled is True
        assert config.hash_algorithm == "sha256"
        assert config.hash_inputs is True
        assert config.hash_outputs is True

    def test_retention_bounds(self):
        """Test retention bounds."""
        with pytest.raises(ValidationError):
            ProvenanceConfig(retention_days=10)


class TestGL010Config:
    """Tests for complete GL-010 configuration."""

    def test_required_monitoring(self):
        """Test that monitoring config is required."""
        with pytest.raises(ValidationError):
            GL010Config()

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = GL010Config(
            monitoring=EmissionsMonitoringConfig(source_id="STACK-001"),
        )
        assert config.agent_id == "GL-010"
        assert config.agent_name == "EmissionsGuardian"

    def test_full_config(self, gl010_config):
        """Test complete configuration."""
        config = gl010_config
        assert config.agent_id == "GL-010-TEST"
        assert config.monitoring.source_id == "STACK-001"
        assert config.cems.enabled is True
        assert config.rata.enabled is True
        assert config.trading.enabled is True
        assert config.explainability.enabled is True

    def test_default_subconfigs(self):
        """Test default sub-configurations."""
        config = GL010Config(
            monitoring=EmissionsMonitoringConfig(source_id="TEST"),
        )
        assert config.cems is not None
        assert config.rata is not None
        assert config.trading is not None
        assert config.reporting is not None
        assert config.explainability is not None
        assert config.provenance is not None


class TestFactoryFunctions:
    """Tests for configuration factory functions."""

    def test_create_default_config(self):
        """Test default configuration factory."""
        config = create_default_config(
            source_id="BOILER-001",
            source_name="Main Boiler",
            nox_limit_lb_hr=25.0,
        )
        assert config.monitoring.source_id == "BOILER-001"
        assert config.monitoring.source_name == "Main Boiler"
        assert config.monitoring.permit_limits.nox_lb_hr == 25.0

    def test_create_default_config_with_fuel(self):
        """Test default configuration with custom fuel."""
        config = create_default_config(
            source_id="TURBINE-001",
            primary_fuel=FuelType.DISTILLATE_OIL,
        )
        assert config.monitoring.primary_fuel == FuelType.DISTILLATE_OIL

    def test_create_cems_config(self):
        """Test CEMS configuration factory."""
        config = create_cems_config(
            source_id="STACK-001",
            unit_id="CEMS-001",
            nox_span=500.0,
            co_span=500.0,
            o2_span=25.0,
        )
        assert config.monitoring.source_id == "STACK-001"
        assert config.cems.enabled is True
        assert config.cems.unit_id == "CEMS-001"
        assert len(config.cems.analyzers) == 3


class TestEnums:
    """Tests for configuration enums."""

    def test_pollutant_type_values(self):
        """Test pollutant type enum values."""
        assert PollutantType.CO2.value == "co2"
        assert PollutantType.NOX.value == "nox"
        assert PollutantType.SO2.value == "so2"

    def test_fuel_type_values(self):
        """Test fuel type enum values."""
        assert FuelType.NATURAL_GAS.value == "natural_gas"
        assert FuelType.DISTILLATE_OIL.value == "distillate_oil"
        assert FuelType.HYDROGEN.value == "hydrogen"

    def test_monitoring_method_values(self):
        """Test monitoring method enum values."""
        assert MonitoringMethod.CEMS.value == "cems"
        assert MonitoringMethod.FUEL_ANALYSIS.value == "fuel_analysis"
        assert MonitoringMethod.EMISSION_FACTOR.value == "emission_factor"

    def test_regulatory_program_values(self):
        """Test regulatory program enum values."""
        assert RegulatoryProgram.EPA_PART_98.value == "epa_part_98"
        assert RegulatoryProgram.EPA_PART_75.value == "epa_part_75"
        assert RegulatoryProgram.TITLE_V.value == "title_v"

    def test_alert_severity_values(self):
        """Test alert severity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_explainability_method_values(self):
        """Test explainability method enum values."""
        assert ExplainabilityMethod.SHAP.value == "shap"
        assert ExplainabilityMethod.LIME.value == "lime"
