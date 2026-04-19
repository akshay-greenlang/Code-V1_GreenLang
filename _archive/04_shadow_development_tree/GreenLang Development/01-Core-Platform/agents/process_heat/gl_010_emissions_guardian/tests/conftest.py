# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian Test Fixtures
======================================

Pytest fixtures for GL-010 test suite.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, date, timedelta
from typing import List

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
)

from greenlang.agents.process_heat.gl_010_emissions_guardian.schemas import (
    EmissionsReading,
    EmissionsAggregate,
    CEMSDataPoint,
    CEMSHourlyRecord,
    CalculationInput,
    CalculationResult,
    ComplianceAssessment,
    EmissionsAlert,
    DataQuality,
    ValidityStatus,
    CalculationMethod,
    ComplianceResult,
    TimeResolution,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def default_permit_limits():
    """Default permit limits configuration."""
    return PermitLimitsConfig(
        nox_lb_hr=25.0,
        co2_lb_hr=50000.0,
        so2_lb_hr=10.0,
        co_lb_hr=100.0,
        opacity_pct=20.0,
    )


@pytest.fixture
def default_alert_thresholds():
    """Default alert thresholds configuration."""
    return AlertThresholdsConfig(
        warning_threshold_pct=80.0,
        alarm_threshold_pct=90.0,
        critical_threshold_pct=95.0,
        rate_of_change_warning_pct_min=5.0,
        exceedance_prediction_window_hr=4.0,
        exceedance_probability_threshold=0.7,
    )


@pytest.fixture
def default_monitoring_config(default_permit_limits, default_alert_thresholds):
    """Default emissions monitoring configuration."""
    return EmissionsMonitoringConfig(
        source_id="STACK-001",
        source_name="Main Boiler Stack",
        monitoring_method=MonitoringMethod.CEMS,
        primary_fuel=FuelType.NATURAL_GAS,
        monitored_pollutants=[
            PollutantType.CO2,
            PollutantType.NOX,
            PollutantType.CO,
            PollutantType.O2,
        ],
        permit_limits=default_permit_limits,
        alert_thresholds=default_alert_thresholds,
        sampling_interval_s=60,
        averaging_period_min=60,
        min_data_availability_pct=90.0,
        applicable_programs=[RegulatoryProgram.TITLE_V, RegulatoryProgram.EPA_PART_98],
    )


@pytest.fixture
def default_cems_analyzer_nox():
    """Default NOx CEMS analyzer configuration."""
    return CEMSAnalyzerConfig(
        analyzer_id="NOX-01",
        pollutant=PollutantType.NOX,
        analyzer_type="chemiluminescence",
        span_value=500.0,
        measurement_range_high=500.0,
        unit="ppm",
        response_time_s=30.0,
        cal_gas_high_ppm=400.0,
    )


@pytest.fixture
def default_cems_analyzer_o2():
    """Default O2 CEMS analyzer configuration."""
    return CEMSAnalyzerConfig(
        analyzer_id="O2-01",
        pollutant=PollutantType.O2,
        analyzer_type="paramagnetic",
        span_value=25.0,
        measurement_range_high=25.0,
        unit="%",
        response_time_s=20.0,
        cal_gas_high_ppm=20.0,
    )


@pytest.fixture
def default_cems_config(default_cems_analyzer_nox, default_cems_analyzer_o2):
    """Default CEMS integration configuration."""
    return CEMSIntegrationConfig(
        enabled=True,
        unit_id="CEMS-001",
        unit_name="Main Boiler CEMS",
        analyzers=[default_cems_analyzer_nox, default_cems_analyzer_o2],
        daily_calibration_enabled=True,
        daily_calibration_time="00:00",
        calibration_drift_limit_pct=2.5,
        rata_frequency_quarters=4,
        rata_relative_accuracy_limit=10.0,
    )


@pytest.fixture
def default_rata_config():
    """Default RATA configuration."""
    return RATAConfig(
        enabled=True,
        base_frequency="quarterly",
        auto_schedule_enabled=True,
        advance_notice_days=30,
        min_test_runs=9,
        relative_accuracy_limit_pct=10.0,
        bias_adjustment_threshold_pct=5.0,
        auto_apply_bias_adjustment=True,
    )


@pytest.fixture
def default_trading_config():
    """Default carbon trading configuration."""
    return CarbonTradingConfig(
        enabled=True,
        primary_market="voluntary",
        entity_id="ENTITY-001",
        minimum_offset_quality_score=60.0,
        price_alert_enabled=True,
        price_alert_threshold_pct=10.0,
        target_surplus_pct=10.0,
    )


@pytest.fixture
def default_reporting_config():
    """Default reporting configuration."""
    return ReportingConfig(
        enabled=True,
        facility_name="Test Facility",
        part98_reporting_enabled=True,
        part98_subparts=["C"],
        title_v_reporting_enabled=True,
        emission_inventory_enabled=True,
        deadline_reminder_days=30,
    )


@pytest.fixture
def default_explainability_config():
    """Default explainability configuration."""
    return ExplainabilityConfig(
        enabled=True,
        shap=SHAPConfig(
            enabled=True,
            method="kernel",
            n_samples=100,
            background_samples=50,
        ),
        lime=LIMEConfig(
            enabled=True,
            num_features=10,
            num_samples=5000,
        ),
        auto_explain_exceedances=True,
        auto_explain_anomalies=True,
        feature_importance_enabled=True,
        top_features_to_report=10,
    )


@pytest.fixture
def default_provenance_config():
    """Default provenance configuration."""
    return ProvenanceConfig(
        enabled=True,
        hash_algorithm="sha256",
        hash_inputs=True,
        hash_outputs=True,
        track_data_sources=True,
        track_calculation_steps=True,
        include_timestamps=True,
        retention_days=365,
    )


@pytest.fixture
def gl010_config(
    default_monitoring_config,
    default_cems_config,
    default_rata_config,
    default_trading_config,
    default_reporting_config,
    default_explainability_config,
    default_provenance_config,
):
    """Complete GL-010 configuration."""
    return GL010Config(
        monitoring=default_monitoring_config,
        cems=default_cems_config,
        rata=default_rata_config,
        trading=default_trading_config,
        reporting=default_reporting_config,
        explainability=default_explainability_config,
        provenance=default_provenance_config,
        agent_id="GL-010-TEST",
        agent_name="EmissionsGuardian-Test",
        data_retention_days=365,
    )


# =============================================================================
# DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_emissions_reading():
    """Sample emissions reading."""
    return EmissionsReading(
        source_id="STACK-001",
        timestamp=datetime.now(timezone.utc),
        pollutant="NOx",
        value=15.5,
        unit="lb/hr",
        data_quality=DataQuality.MEASURED,
        validity_status=ValidityStatus.VALID,
        load_pct=85.0,
        stack_temperature_f=350.0,
        o2_pct=3.5,
    )


@pytest.fixture
def sample_cems_data_point():
    """Sample CEMS data point."""
    return CEMSDataPoint(
        unit_id="CEMS-001",
        timestamp=datetime.now(timezone.utc),
        parameter="NOx",
        value=125.5,
        unit="ppm",
        analyzer_id="NOX-01",
        validity_status=ValidityStatus.VALID,
        quality_assured=True,
        in_calibration=True,
        range_high=500.0,
        span_value=500.0,
    )


@pytest.fixture
def sample_cems_hourly():
    """Sample CEMS hourly record."""
    return CEMSHourlyRecord(
        unit_id="CEMS-001",
        operating_date=date.today(),
        operating_hour=14,
        nox_ppm=125.5,
        co2_pct=8.5,
        o2_pct=3.5,
        nox_rate_lb_mmbtu=0.15,
        heat_input_mmbtu=150.5,
        nox_mass_lb=22.6,
        op_time=1.0,
        percent_available=100.0,
        daily_calibration_status=ValidityStatus.VALID,
    )


@pytest.fixture
def sample_calculation_input():
    """Sample calculation input."""
    return CalculationInput(
        calculation_type="ghg_emissions",
        source_id="BOILER-001",
        fuel_type="natural_gas",
        fuel_consumption=1500.0,
        fuel_unit="MMBtu",
        operating_hours=24.0,
        method=CalculationMethod.DEFAULT_EF,
    )


@pytest.fixture
def sample_calculation_result():
    """Sample calculation result."""
    return CalculationResult(
        calculation_id="CALC-2024-001",
        source_id="BOILER-001",
        pollutant="CO2",
        value=79590.0,
        unit="kg",
        method=CalculationMethod.DEFAULT_EF,
        method_reference="40 CFR 98.33(a)(3)",
        emission_factor_value=53.06,
        emission_factor_unit="kg/MMBtu",
        emission_factor_source="40 CFR Part 98 Table C-1",
        uncertainty_pct=5.0,
        fuel_consumption=1500.0,
        fuel_unit="MMBtu",
        operating_hours=24.0,
    )


@pytest.fixture
def sample_compliance_assessment():
    """Sample compliance assessment."""
    return ComplianceAssessment(
        assessment_id="CA-2024-001",
        source_id="STACK-001",
        pollutant="NOx",
        period_start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        period_end=datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
        measured_value=22.5,
        limit_value=25.0,
        unit="lb/hr",
        result=ComplianceResult.COMPLIANT,
        margin_pct=10.0,
    )


@pytest.fixture
def sample_emissions_alert():
    """Sample emissions alert."""
    return EmissionsAlert(
        alert_id="ALERT-2024-001",
        source_id="STACK-001",
        alert_type="approaching_limit",
        severity="warning",
        pollutant="NOx",
        current_value=22.5,
        threshold_value=25.0,
        unit="lb/hr",
        message="NOx at 90% of permit limit",
        status="active",
    )


@pytest.fixture
def sample_emissions_aggregate():
    """Sample emissions aggregate."""
    return EmissionsAggregate(
        source_id="STACK-001",
        pollutant="CO2",
        period_start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        period_end=datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
        resolution=TimeResolution.MONTHLY,
        total_mass=1500.0,
        total_mass_unit="tons",
        avg_rate=50.0,
        max_rate=75.0,
        min_rate=25.0,
        rate_unit="tons/day",
        reading_count=720,
        valid_reading_count=700,
        data_availability_pct=97.2,
        operating_hours=720.0,
        calculation_method=CalculationMethod.CEMS,
    )
