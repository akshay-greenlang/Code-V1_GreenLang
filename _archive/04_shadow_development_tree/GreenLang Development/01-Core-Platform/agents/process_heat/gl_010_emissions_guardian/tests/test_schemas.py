# -*- coding: utf-8 -*-
"""
GL-010 Schema Tests
===================

Unit tests for GL-010 data schemas module.
Tests all Pydantic models for emissions data, CEMS, calculations, and compliance.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, date, timedelta
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_010_emissions_guardian.schemas import (
    BaseEmissionsSchema,
    MeasurementValue,
    EmissionsReading,
    EmissionsAggregate,
    CEMSDataPoint,
    CEMSHourlyRecord,
    CEMSQuarterlySummary,
    CalculationInput,
    CalculationResult,
    EmissionFactor,
    PermitLimit,
    ComplianceAssessment,
    ExceedanceEvent,
    EmissionsAlert,
    SourceEmissionsSummary,
    FacilityEmissionsSummary,
    DataQuality,
    ValidityStatus,
    CalculationMethod,
    ComplianceResult,
    TimeResolution,
    SourceCategory,
)


class TestBaseEmissionsSchema:
    """Tests for base emissions schema."""

    def test_provenance_hash_calculation(self, sample_emissions_reading):
        """Test provenance hash calculation."""
        hash1 = sample_emissions_reading.calculate_provenance_hash()
        assert len(hash1) == 64  # SHA-256 hex digest length
        assert isinstance(hash1, str)

    def test_update_provenance(self, sample_emissions_reading):
        """Test provenance update."""
        sample_emissions_reading.update_provenance()
        assert sample_emissions_reading.provenance_hash is not None
        assert sample_emissions_reading.updated_at is not None


class TestMeasurementValue:
    """Tests for measurement value schema."""

    def test_basic_measurement(self):
        """Test basic measurement creation."""
        m = MeasurementValue(value=25.5, unit="lb/hr")
        assert m.value == 25.5
        assert m.unit == "lb/hr"
        assert m.quality == DataQuality.MEASURED

    def test_measurement_with_uncertainty(self):
        """Test measurement with uncertainty."""
        m = MeasurementValue(
            value=100.0,
            unit="ppm",
            uncertainty=5.0,
            uncertainty_pct=5.0,
        )
        assert m.uncertainty == 5.0
        assert m.uncertainty_pct == 5.0

    def test_measurement_with_bounds(self):
        """Test measurement with confidence bounds."""
        m = MeasurementValue(
            value=100.0,
            unit="ppm",
            lower_bound=95.0,
            upper_bound=105.0,
        )
        assert m.lower_bound == 95.0
        assert m.upper_bound == 105.0

    def test_unit_conversion(self):
        """Test unit conversion method."""
        m = MeasurementValue(value=1.0, unit="kg")
        converted = m.to_unit("lb", 2.205)
        assert abs(converted.value - 2.205) < 0.001
        assert converted.unit == "lb"

    def test_nan_validation(self):
        """Test NaN value rejection."""
        with pytest.raises(ValidationError):
            MeasurementValue(value=float('nan'), unit="ppm")


class TestEmissionsReading:
    """Tests for emissions reading schema."""

    def test_valid_reading(self, sample_emissions_reading):
        """Test valid emissions reading."""
        reading = sample_emissions_reading
        assert reading.source_id == "STACK-001"
        assert reading.pollutant == "NOX"  # Normalized to uppercase
        assert reading.value == 15.5

    def test_pollutant_normalization(self):
        """Test pollutant name normalization."""
        reading = EmissionsReading(
            source_id="TEST",
            pollutant="nox",  # lowercase
            value=10.0,
            unit="lb/hr",
        )
        assert reading.pollutant == "NOX"  # Normalized to uppercase

    def test_validity_status(self, sample_emissions_reading):
        """Test validity status."""
        assert sample_emissions_reading.validity_status == ValidityStatus.VALID
        assert sample_emissions_reading.data_quality == DataQuality.MEASURED

    def test_operating_conditions(self, sample_emissions_reading):
        """Test operating conditions fields."""
        reading = sample_emissions_reading
        assert reading.load_pct == 85.0
        assert reading.stack_temperature_f == 350.0
        assert reading.o2_pct == 3.5

    def test_o2_bounds(self):
        """Test O2 percentage bounds."""
        with pytest.raises(ValidationError):
            EmissionsReading(
                source_id="TEST",
                pollutant="CO2",
                value=100.0,
                unit="tons/hr",
                o2_pct=25.0,  # Over 21%
            )


class TestEmissionsAggregate:
    """Tests for emissions aggregate schema."""

    def test_valid_aggregate(self, sample_emissions_aggregate):
        """Test valid emissions aggregate."""
        agg = sample_emissions_aggregate
        assert agg.source_id == "STACK-001"
        assert agg.pollutant == "CO2"
        assert agg.total_mass == 1500.0
        assert agg.resolution == TimeResolution.MONTHLY

    def test_period_validation(self):
        """Test period start/end validation."""
        with pytest.raises(ValidationError):
            EmissionsAggregate(
                source_id="TEST",
                pollutant="CO2",
                period_start=datetime(2024, 7, 1, tzinfo=timezone.utc),
                period_end=datetime(2024, 6, 1, tzinfo=timezone.utc),  # Before start
                resolution=TimeResolution.MONTHLY,
                total_mass=100.0,
            )

    def test_data_availability(self, sample_emissions_aggregate):
        """Test data availability calculation."""
        agg = sample_emissions_aggregate
        assert agg.data_availability_pct == 97.2
        assert agg.valid_reading_count == 700


class TestCEMSDataPoint:
    """Tests for CEMS data point schema."""

    def test_valid_data_point(self, sample_cems_data_point):
        """Test valid CEMS data point."""
        dp = sample_cems_data_point
        assert dp.unit_id == "CEMS-001"
        assert dp.parameter == "NOx"
        assert dp.value == 125.5

    def test_calibration_status(self, sample_cems_data_point):
        """Test calibration status fields."""
        dp = sample_cems_data_point
        assert dp.in_calibration is True
        assert dp.quality_assured is True

    def test_data_flags(self):
        """Test data quality flags."""
        dp = CEMSDataPoint(
            unit_id="CEMS-001",
            timestamp=datetime.now(timezone.utc),
            parameter="NOx",
            value=550.0,
            unit="ppm",
            range_high=500.0,
            out_of_range=True,
        )
        assert dp.out_of_range is True


class TestCEMSHourlyRecord:
    """Tests for CEMS hourly record schema."""

    def test_valid_hourly(self, sample_cems_hourly):
        """Test valid hourly record."""
        hourly = sample_cems_hourly
        assert hourly.unit_id == "CEMS-001"
        assert hourly.operating_hour == 14
        assert hourly.nox_ppm == 125.5

    def test_hour_bounds(self):
        """Test operating hour bounds."""
        with pytest.raises(ValidationError):
            CEMSHourlyRecord(
                unit_id="TEST",
                operating_date=date.today(),
                operating_hour=25,  # Invalid hour
            )

    def test_op_time_bounds(self):
        """Test operating time bounds."""
        with pytest.raises(ValidationError):
            CEMSHourlyRecord(
                unit_id="TEST",
                operating_date=date.today(),
                operating_hour=10,
                op_time=1.5,  # Over 1.0
            )


class TestCEMSQuarterlySummary:
    """Tests for CEMS quarterly summary schema."""

    def test_valid_summary(self):
        """Test valid quarterly summary."""
        summary = CEMSQuarterlySummary(
            unit_id="CEMS-001",
            year=2024,
            quarter=2,
            nox_tons=50.0,
            so2_tons=10.0,
            co2_tons=5000.0,
            total_heat_input_mmbtu=100000.0,
            operating_hours=2000.0,
        )
        assert summary.year == 2024
        assert summary.quarter == 2
        assert summary.nox_tons == 50.0

    def test_quarter_bounds(self):
        """Test quarter bounds."""
        with pytest.raises(ValidationError):
            CEMSQuarterlySummary(
                unit_id="TEST",
                year=2024,
                quarter=5,  # Invalid quarter
            )


class TestCalculationInput:
    """Tests for calculation input schema."""

    def test_valid_input(self, sample_calculation_input):
        """Test valid calculation input."""
        input_data = sample_calculation_input
        assert input_data.calculation_type == "ghg_emissions"
        assert input_data.fuel_type == "natural_gas"
        assert input_data.fuel_consumption == 1500.0

    def test_fuel_analysis_input(self):
        """Test fuel analysis calculation input."""
        input_data = CalculationInput(
            calculation_type="ghg_emissions",
            source_id="BOILER-001",
            fuel_type="natural_gas",
            fuel_consumption=1000.0,
            fuel_carbon_content=75.0,
            fuel_hhv=23875.0,
            method=CalculationMethod.FUEL_ANALYSIS,
        )
        assert input_data.fuel_carbon_content == 75.0
        assert input_data.method == CalculationMethod.FUEL_ANALYSIS


class TestCalculationResult:
    """Tests for calculation result schema."""

    def test_valid_result(self, sample_calculation_result):
        """Test valid calculation result."""
        result = sample_calculation_result
        assert result.calculation_id == "CALC-2024-001"
        assert result.pollutant == "CO2"
        assert result.value == 79590.0

    def test_method_reference(self, sample_calculation_result):
        """Test method reference documentation."""
        result = sample_calculation_result
        assert result.method_reference == "40 CFR 98.33(a)(3)"
        assert result.emission_factor_source == "40 CFR Part 98 Table C-1"

    def test_uncertainty_analysis(self, sample_calculation_result):
        """Test uncertainty fields."""
        result = sample_calculation_result
        assert result.uncertainty_pct == 5.0


class TestEmissionFactor:
    """Tests for emission factor schema."""

    def test_valid_factor(self):
        """Test valid emission factor."""
        ef = EmissionFactor(
            factor_id="EF-CO2-NG-001",
            pollutant="CO2",
            fuel_type="natural_gas",
            value=53.06,
            unit="kg/MMBtu",
            source="40 CFR Part 98 Table C-1",
        )
        assert ef.value == 53.06
        assert ef.source == "40 CFR Part 98 Table C-1"

    def test_positive_value(self):
        """Test positive value requirement."""
        with pytest.raises(ValidationError):
            EmissionFactor(
                factor_id="TEST",
                pollutant="CO2",
                fuel_type="natural_gas",
                value=-10.0,  # Negative
                unit="kg/MMBtu",
                source="test",
            )


class TestPermitLimit:
    """Tests for permit limit schema."""

    def test_valid_limit(self):
        """Test valid permit limit."""
        limit = PermitLimit(
            limit_id="NOX-LIMIT-001",
            pollutant="NOx",
            limit_value=25.0,
            unit="lb/hr",
            averaging_period_hr=1.0,
            limit_type="short_term",
        )
        assert limit.limit_value == 25.0
        assert limit.averaging_period_hr == 1.0


class TestComplianceAssessment:
    """Tests for compliance assessment schema."""

    def test_valid_assessment(self, sample_compliance_assessment):
        """Test valid compliance assessment."""
        assessment = sample_compliance_assessment
        assert assessment.assessment_id == "CA-2024-001"
        assert assessment.result == ComplianceResult.COMPLIANT
        assert assessment.margin_pct == 10.0

    def test_exceedance_assessment(self):
        """Test exceedance assessment."""
        assessment = ComplianceAssessment(
            assessment_id="CA-2024-002",
            source_id="STACK-001",
            pollutant="NOx",
            period_start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 6, 30, tzinfo=timezone.utc),
            measured_value=30.0,
            limit_value=25.0,
            unit="lb/hr",
            result=ComplianceResult.EXCEEDANCE,
            exceedance_pct=20.0,
            corrective_action_required=True,
        )
        assert assessment.result == ComplianceResult.EXCEEDANCE
        assert assessment.corrective_action_required is True


class TestExceedanceEvent:
    """Tests for exceedance event schema."""

    def test_valid_exceedance(self):
        """Test valid exceedance event."""
        event = ExceedanceEvent(
            event_id="EXC-2024-001",
            source_id="STACK-001",
            pollutant="NOx",
            exceedance_start=datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc),
            exceedance_end=datetime(2024, 6, 15, 15, 45, tzinfo=timezone.utc),
            max_value=28.5,
            limit_value=25.0,
            unit="lb/hr",
            max_exceedance_pct=14.0,
        )
        assert event.max_value == 28.5
        assert event.max_exceedance_pct == 14.0


class TestEmissionsAlert:
    """Tests for emissions alert schema."""

    def test_valid_alert(self, sample_emissions_alert):
        """Test valid emissions alert."""
        alert = sample_emissions_alert
        assert alert.alert_id == "ALERT-2024-001"
        assert alert.alert_type == "approaching_limit"
        assert alert.severity == "warning"

    def test_predictive_alert(self):
        """Test predictive alert."""
        alert = EmissionsAlert(
            alert_id="ALERT-2024-002",
            source_id="STACK-001",
            alert_type="predicted_exceedance",
            severity="warning",
            message="Exceedance predicted in 2 hours",
            predicted_exceedance=True,
            predicted_exceedance_time=datetime.now(timezone.utc) + timedelta(hours=2),
            prediction_confidence=0.85,
        )
        assert alert.predicted_exceedance is True
        assert alert.prediction_confidence == 0.85


class TestSourceEmissionsSummary:
    """Tests for source emissions summary schema."""

    def test_valid_summary(self):
        """Test valid source summary."""
        summary = SourceEmissionsSummary(
            source_id="STACK-001",
            source_category=SourceCategory.COMBUSTION,
            period_type=TimeResolution.MONTHLY,
            period_start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 6, 30, tzinfo=timezone.utc),
            co2_tons=1500.0,
            nox_tons=2.5,
            co2e_tons=1505.0,
            data_availability_pct=97.0,
        )
        assert summary.co2_tons == 1500.0
        assert summary.source_category == SourceCategory.COMBUSTION


class TestFacilityEmissionsSummary:
    """Tests for facility emissions summary schema."""

    def test_valid_summary(self):
        """Test valid facility summary."""
        summary = FacilityEmissionsSummary(
            facility_id="FACILITY-001",
            facility_name="Test Facility",
            period_type=TimeResolution.ANNUAL,
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            total_co2_tons=50000.0,
            total_co2e_tons=51000.0,
            total_co2e_mtco2e=51.0,
            source_count=5,
        )
        assert summary.total_co2e_mtco2e == 51.0
        assert summary.source_count == 5

    def test_part98_threshold(self):
        """Test Part 98 threshold evaluation."""
        summary = FacilityEmissionsSummary(
            facility_id="FACILITY-001",
            period_type=TimeResolution.ANNUAL,
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            total_co2e_mtco2e=30000.0,
            exceeds_part98_threshold=True,
        )
        assert summary.exceeds_part98_threshold is True


class TestEnums:
    """Tests for schema enums."""

    def test_data_quality_values(self):
        """Test data quality enum values."""
        assert DataQuality.MEASURED.value == "measured"
        assert DataQuality.SUBSTITUTE.value == "substitute"
        assert DataQuality.DEFAULT.value == "default"

    def test_validity_status_values(self):
        """Test validity status enum values."""
        assert ValidityStatus.VALID.value == "valid"
        assert ValidityStatus.INVALID.value == "invalid"
        assert ValidityStatus.CALIBRATION.value == "calibration"

    def test_calculation_method_values(self):
        """Test calculation method enum values."""
        assert CalculationMethod.CEMS.value == "cems"
        assert CalculationMethod.FUEL_ANALYSIS.value == "fuel_analysis"
        assert CalculationMethod.F_FACTOR.value == "f_factor"

    def test_compliance_result_values(self):
        """Test compliance result enum values."""
        assert ComplianceResult.COMPLIANT.value == "compliant"
        assert ComplianceResult.EXCEEDANCE.value == "exceedance"
        assert ComplianceResult.DEVIATION.value == "deviation"

    def test_time_resolution_values(self):
        """Test time resolution enum values."""
        assert TimeResolution.HOURLY.value == "hourly"
        assert TimeResolution.DAILY.value == "daily"
        assert TimeResolution.ANNUAL.value == "annual"
