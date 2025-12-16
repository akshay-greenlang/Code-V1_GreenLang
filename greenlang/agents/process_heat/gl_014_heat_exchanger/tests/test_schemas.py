# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Schema Tests

Comprehensive tests for all Pydantic schema classes including validation,
serialization, and edge cases.

Coverage Target: 90%+
"""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError as PydanticValidationError
import json

from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    HeatExchangerInput,
    HeatExchangerOperatingData,
    HeatExchangerOutput,
    StreamConditions,
    ProcessMeasurement,
    TubeInspectionData,
    CleaningRecord,
    ThermalPerformanceResult,
    FoulingAnalysisResult,
    HydraulicAnalysisResult,
    TubeIntegrityResult,
    CleaningRecommendation,
    EconomicAnalysisResult,
    Alert,
    ASMEPTC125Result,
    HealthStatus,
    TrendDirection,
    OperatingMode,
    TestCompliance,
)

from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    AlertSeverity,
    CleaningMethod,
)


class TestStreamConditions:
    """Tests for StreamConditions schema."""

    def test_create_stream_conditions(self):
        """Test creating stream conditions with all fields."""
        stream = StreamConditions(
            temperature_c=100.0,
            pressure_barg=5.0,
            mass_flow_kg_s=10.0,
            density_kg_m3=998.0,
            viscosity_cp=1.0,
            specific_heat_kj_kgk=4.18,
            thermal_conductivity_w_mk=0.62,
        )
        assert stream.temperature_c == 100.0
        assert stream.pressure_barg == 5.0
        assert stream.mass_flow_kg_s == 10.0

    def test_stream_conditions_minimal(self):
        """Test stream conditions with minimal required fields."""
        stream = StreamConditions(
            temperature_c=50.0,
            pressure_barg=3.0,
            mass_flow_kg_s=5.0,
        )
        assert stream.density_kg_m3 is None
        assert stream.viscosity_cp is None

    def test_stream_conditions_temperature_validation(self):
        """Test temperature must be above absolute zero (reasonable limit)."""
        # Valid temperatures
        stream = StreamConditions(
            temperature_c=-40.0,  # Cryogenic
            pressure_barg=1.0,
            mass_flow_kg_s=1.0,
        )
        assert stream.temperature_c == -40.0

        # Too cold (below -273.15C)
        with pytest.raises(PydanticValidationError):
            StreamConditions(
                temperature_c=-300.0,
                pressure_barg=1.0,
                mass_flow_kg_s=1.0,
            )

    def test_stream_conditions_positive_flow(self):
        """Test mass flow must be positive."""
        with pytest.raises(PydanticValidationError):
            StreamConditions(
                temperature_c=50.0,
                pressure_barg=1.0,
                mass_flow_kg_s=-5.0,
            )


class TestProcessMeasurement:
    """Tests for ProcessMeasurement schema."""

    def test_create_measurement(self):
        """Test creating process measurement."""
        measurement = ProcessMeasurement(
            tag="TI-1001",
            value=150.5,
            unit="C",
            timestamp=datetime.now(timezone.utc),
            quality="good",
        )
        assert measurement.tag == "TI-1001"
        assert measurement.value == 150.5

    def test_measurement_quality_validation(self):
        """Test measurement quality must be valid."""
        # Valid qualities
        for quality in ["good", "bad", "uncertain"]:
            m = ProcessMeasurement(
                tag="TI-1001",
                value=100.0,
                unit="C",
                quality=quality,
            )
            assert m.quality == quality


class TestOperatingData:
    """Tests for HeatExchangerOperatingData schema."""

    def test_create_operating_data(self, stream_conditions_hot, stream_conditions_cold):
        """Test creating operating data."""
        op_data = HeatExchangerOperatingData(
            timestamp=datetime.now(timezone.utc),
            shell_inlet=stream_conditions_hot,
            shell_outlet=StreamConditions(
                temperature_c=100.0,
                pressure_barg=4.8,
                mass_flow_kg_s=10.0,
            ),
            tube_inlet=stream_conditions_cold,
            tube_outlet=StreamConditions(
                temperature_c=60.0,
                pressure_barg=2.8,
                mass_flow_kg_s=15.0,
            ),
            operating_mode=OperatingMode.NORMAL,
            load_percent=100.0,
        )
        assert op_data.shell_inlet.temperature_c == 150.0
        assert op_data.operating_mode == OperatingMode.NORMAL

    def test_operating_data_with_pressure_drops(self, stream_conditions_hot, stream_conditions_cold):
        """Test operating data with measured pressure drops."""
        op_data = HeatExchangerOperatingData(
            timestamp=datetime.now(timezone.utc),
            shell_inlet=stream_conditions_hot,
            shell_outlet=StreamConditions(
                temperature_c=100.0, pressure_barg=4.5, mass_flow_kg_s=10.0
            ),
            shell_pressure_drop_bar=0.5,
            tube_inlet=stream_conditions_cold,
            tube_outlet=StreamConditions(
                temperature_c=60.0, pressure_barg=2.7, mass_flow_kg_s=15.0
            ),
            tube_pressure_drop_bar=0.3,
        )
        assert op_data.shell_pressure_drop_bar == 0.5
        assert op_data.tube_pressure_drop_bar == 0.3

    @pytest.mark.parametrize("mode", list(OperatingMode))
    def test_all_operating_modes(self, mode, stream_conditions_hot, stream_conditions_cold):
        """Test all operating modes are valid."""
        op_data = HeatExchangerOperatingData(
            timestamp=datetime.now(timezone.utc),
            shell_inlet=stream_conditions_hot,
            shell_outlet=stream_conditions_cold,
            tube_inlet=stream_conditions_cold,
            tube_outlet=stream_conditions_hot,
            operating_mode=mode,
        )
        assert op_data.operating_mode == mode


class TestTubeInspectionData:
    """Tests for TubeInspectionData schema."""

    def test_create_inspection_data(self):
        """Test creating tube inspection data."""
        inspection = TubeInspectionData(
            inspection_date=datetime.now(timezone.utc),
            inspection_method="eddy_current",
            total_tubes=100,
            tubes_inspected=100,
            tubes_with_defects=5,
            tubes_plugged=2,
        )
        assert inspection.total_tubes == 100
        assert inspection.tubes_with_defects == 5
        assert inspection.defect_rate == 0.05

    def test_defect_rate_calculation(self):
        """Test defect rate property calculation."""
        inspection = TubeInspectionData(
            inspection_date=datetime.now(timezone.utc),
            inspection_method="eddy_current",
            total_tubes=200,
            tubes_inspected=200,
            tubes_with_defects=10,
            tubes_plugged=5,
        )
        expected_rate = 10 / 200
        assert inspection.defect_rate == pytest.approx(expected_rate)

    def test_inspection_with_wall_loss_summary(self):
        """Test inspection data with wall loss summary."""
        inspection = TubeInspectionData(
            inspection_date=datetime.now(timezone.utc),
            inspection_method="eddy_current",
            total_tubes=100,
            tubes_inspected=100,
            tubes_with_defects=10,
            tubes_plugged=2,
            wall_loss_summary={
                "<20%": 85,
                "20-40%": 10,
                "40-60%": 3,
                "60-80%": 2,
                ">80%": 0,
            },
        )
        assert inspection.wall_loss_summary["<20%"] == 85

    def test_inspection_with_recommendations(self):
        """Test inspection data with plugging recommendations."""
        inspection = TubeInspectionData(
            inspection_date=datetime.now(timezone.utc),
            inspection_method="eddy_current",
            total_tubes=100,
            tubes_inspected=100,
            tubes_with_defects=8,
            tubes_plugged=3,
            tubes_recommended_for_plugging=[45, 67, 89],
            retube_recommended=False,
        )
        assert len(inspection.tubes_recommended_for_plugging) == 3
        assert 45 in inspection.tubes_recommended_for_plugging


class TestCleaningRecord:
    """Tests for CleaningRecord schema."""

    def test_create_cleaning_record(self):
        """Test creating cleaning record."""
        record = CleaningRecord(
            cleaning_date=datetime.now(timezone.utc),
            cleaning_method=CleaningMethod.HIGH_PRESSURE_WATER,
            duration_hours=8.0,
            cost_usd=5500.0,
            u_before_cleaning=380.0,
            u_after_cleaning=480.0,
            effectiveness_before=0.65,
            effectiveness_after=0.82,
        )
        assert record.cleaning_method == CleaningMethod.HIGH_PRESSURE_WATER
        assert record.u_after_cleaning > record.u_before_cleaning

    def test_cleaning_record_u_recovery(self):
        """Test U value recovery calculation."""
        record = CleaningRecord(
            cleaning_date=datetime.now(timezone.utc),
            cleaning_method=CleaningMethod.CHEMICAL,
            duration_hours=12.0,
            cost_usd=8000.0,
            u_before_cleaning=350.0,
            u_after_cleaning=490.0,
            effectiveness_before=0.60,
            effectiveness_after=0.85,
        )
        recovery = (record.u_after_cleaning - record.u_before_cleaning) / record.u_before_cleaning
        assert recovery == pytest.approx(0.4, rel=0.01)


class TestThermalPerformanceResult:
    """Tests for ThermalPerformanceResult schema."""

    def test_create_thermal_result(self):
        """Test creating thermal performance result."""
        result = ThermalPerformanceResult(
            actual_duty_kw=950.0,
            design_duty_kw=1000.0,
            duty_ratio=0.95,
            lmtd_c=35.0,
            lmtd_correction_factor=0.92,
            corrected_lmtd_c=32.2,
            approach_temperature_c=10.0,
            u_clean_w_m2k=500.0,
            u_actual_w_m2k=420.0,
            u_design_w_m2k=450.0,
            u_degradation_percent=16.0,
            ntu=2.5,
            heat_capacity_ratio=0.8,
            thermal_effectiveness=0.78,
            design_effectiveness=0.85,
            effectiveness_ratio=0.918,
            calculated_fouling_m2kw=0.00038,
            effectiveness_trend=TrendDirection.STABLE,
        )
        assert result.thermal_effectiveness == 0.78
        assert result.u_degradation_percent == 16.0

    def test_thermal_result_effectiveness_bounds(self):
        """Test effectiveness must be 0-1."""
        with pytest.raises(PydanticValidationError):
            ThermalPerformanceResult(
                actual_duty_kw=1000.0,
                design_duty_kw=1000.0,
                duty_ratio=1.0,
                lmtd_c=30.0,
                lmtd_correction_factor=1.0,
                corrected_lmtd_c=30.0,
                approach_temperature_c=5.0,
                u_clean_w_m2k=500.0,
                u_actual_w_m2k=500.0,
                u_design_w_m2k=500.0,
                u_degradation_percent=0.0,
                ntu=2.0,
                heat_capacity_ratio=0.5,
                thermal_effectiveness=1.5,  # Invalid: > 1
                design_effectiveness=0.85,
                effectiveness_ratio=1.0,
                calculated_fouling_m2kw=0.0,
                effectiveness_trend=TrendDirection.STABLE,
            )


class TestFoulingAnalysisResult:
    """Tests for FoulingAnalysisResult schema."""

    def test_create_fouling_result(self):
        """Test creating fouling analysis result."""
        result = FoulingAnalysisResult(
            shell_side_fouling_m2kw=0.00020,
            tube_side_fouling_m2kw=0.00018,
            total_fouling_m2kw=0.00038,
            design_fouling_m2kw=0.00034,
            fouling_ratio=1.12,
            fouling_rate_m2kw_per_day=0.0000015,
            fouling_trend=TrendDirection.DEGRADING,
            days_to_cleaning_threshold=45,
            ml_predicted_fouling_30d=0.00042,
            ml_prediction_confidence=0.85,
            asymptotic_fouling_m2kw=0.0005,
            current_fouling_factor=1.12,
        )
        assert result.total_fouling_m2kw == 0.00038
        assert result.fouling_ratio > 1.0


class TestHydraulicAnalysisResult:
    """Tests for HydraulicAnalysisResult schema."""

    def test_create_hydraulic_result(self):
        """Test creating hydraulic analysis result."""
        result = HydraulicAnalysisResult(
            shell_pressure_drop_bar=0.35,
            shell_dp_design_bar=1.0,
            shell_dp_measured_bar=0.38,
            shell_dp_ratio=0.35,
            shell_velocity_m_s=1.2,
            tube_pressure_drop_bar=0.45,
            tube_dp_design_bar=1.0,
            tube_dp_measured_bar=0.48,
            tube_dp_ratio=0.45,
            tube_velocity_m_s=1.8,
            shell_dp_fouling_contribution_bar=0.05,
            tube_dp_fouling_contribution_bar=0.08,
            shell_reynolds=25000.0,
            tube_reynolds=35000.0,
            shell_dp_alarm=False,
            tube_dp_alarm=False,
        )
        assert result.shell_pressure_drop_bar == 0.35
        assert result.tube_velocity_m_s == 1.8
        assert not result.shell_dp_alarm


class TestTubeIntegrityResult:
    """Tests for TubeIntegrityResult schema."""

    def test_create_tube_integrity_result(self):
        """Test creating tube integrity result."""
        result = TubeIntegrityResult(
            current_wall_thickness_mm=1.8,
            minimum_required_thickness_mm=1.25,
            thickness_margin_mm=0.55,
            wall_loss_percent=15.0,
            tubes_plugged=5,
            plugging_rate_percent=5.0,
            tubes_at_risk=3,
            estimated_remaining_life_years=8.5,
            remaining_life_confidence=0.82,
            predicted_failures_1yr=1,
            predicted_failures_5yr=5,
            weibull_beta=2.5,
            weibull_eta_years=20.0,
            retube_recommended=False,
            next_inspection_date=datetime.now(timezone.utc) + timedelta(days=180),
            inspection_urgency=AlertSeverity.INFO,
            failure_modes=[
                {"mode": "tube_leak", "probability": 0.05, "severity": "low"}
            ],
        )
        assert result.estimated_remaining_life_years == 8.5
        assert not result.retube_recommended

    def test_tube_integrity_retube_recommendation(self):
        """Test tube integrity with retube recommendation."""
        result = TubeIntegrityResult(
            current_wall_thickness_mm=1.3,
            minimum_required_thickness_mm=1.25,
            thickness_margin_mm=0.05,
            wall_loss_percent=38.0,
            tubes_plugged=12,
            plugging_rate_percent=12.0,
            tubes_at_risk=8,
            estimated_remaining_life_years=1.5,
            remaining_life_confidence=0.75,
            predicted_failures_1yr=5,
            predicted_failures_5yr=15,
            retube_recommended=True,
            next_inspection_date=datetime.now(timezone.utc) + timedelta(days=30),
            inspection_urgency=AlertSeverity.CRITICAL,
            failure_modes=[],
        )
        assert result.retube_recommended
        assert result.plugging_rate_percent > 10


class TestCleaningRecommendation:
    """Tests for CleaningRecommendation schema."""

    def test_create_cleaning_recommendation(self):
        """Test creating cleaning recommendation."""
        rec = CleaningRecommendation(
            recommended=True,
            recommended_method=CleaningMethod.HIGH_PRESSURE_WATER,
            alternative_methods=[CleaningMethod.CHEMICAL],
            urgency=AlertSeverity.WARNING,
            days_until_recommended=15,
            optimal_cleaning_date=datetime.now(timezone.utc) + timedelta(days=15),
            estimated_cleaning_cost_usd=5500.0,
            estimated_downtime_hours=8.0,
            expected_u_recovery_percent=90.0,
            expected_effectiveness_after=0.82,
            cleaning_roi_percent=250.0,
            reasoning="Fouling approaching threshold",
        )
        assert rec.recommended
        assert rec.recommended_method == CleaningMethod.HIGH_PRESSURE_WATER
        assert rec.cleaning_roi_percent > 0


class TestEconomicAnalysisResult:
    """Tests for EconomicAnalysisResult schema."""

    def test_create_economic_result(self):
        """Test creating economic analysis result."""
        result = EconomicAnalysisResult(
            energy_loss_kw=50.0,
            energy_cost_usd_per_day=120.0,
            energy_cost_usd_per_month=3600.0,
            energy_cost_usd_per_year=43800.0,
            cleaning_roi_percent=280.0,
            payback_period_days=14,
            optimal_cleaning_frequency_days=120,
            annual_cleaning_cost_usd=18000.0,
            remaining_value_usd=350000.0,
            replacement_timing_years=8.0,
            replace_vs_maintain_npv_usd=-150000.0,
            annual_tco_usd=65000.0,
            lifecycle_cost_usd=520000.0,
            optimization_savings_usd_per_year=25000.0,
            optimization_recommendations=[
                "Adjust cleaning schedule to 90 days",
                "Consider online cleaning system",
            ],
        )
        assert result.cleaning_roi_percent > 0
        assert result.replace_vs_maintain_npv_usd < 0  # Maintain is better


class TestAlert:
    """Tests for Alert schema."""

    def test_create_alert(self):
        """Test creating alert."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            category="THERMAL",
            message="Effectiveness below 70%",
            parameter="thermal_effectiveness",
            current_value=0.68,
            threshold_value=0.70,
        )
        assert alert.severity == AlertSeverity.WARNING
        assert alert.category == "THERMAL"

    @pytest.mark.parametrize("severity", list(AlertSeverity))
    def test_all_alert_severities(self, severity):
        """Test all alert severities."""
        alert = Alert(
            severity=severity,
            category="TEST",
            message="Test alert",
            parameter="test",
        )
        assert alert.severity == severity


class TestHealthStatusEnum:
    """Tests for HealthStatus enumeration."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.EXCELLENT.value == "excellent"
        assert HealthStatus.GOOD.value == "good"
        assert HealthStatus.FAIR.value == "fair"
        assert HealthStatus.POOR.value == "poor"
        assert HealthStatus.CRITICAL.value == "critical"

    def test_health_status_ordering(self):
        """Test health status ordering for comparison."""
        # Verify all statuses exist
        statuses = list(HealthStatus)
        assert len(statuses) == 5


class TestTrendDirectionEnum:
    """Tests for TrendDirection enumeration."""

    def test_trend_direction_values(self):
        """Test trend direction enum values."""
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.DEGRADING.value == "degrading"
        assert TrendDirection.RAPID_DEGRADATION.value == "rapid_degradation"


class TestHeatExchangerOutput:
    """Tests for HeatExchangerOutput schema."""

    def test_create_minimal_output(self):
        """Test creating minimal output with required fields."""
        output = HeatExchangerOutput(
            request_id="req-001",
            exchanger_id="E-1001",
            timestamp=datetime.now(timezone.utc),
            status="success",
            processing_time_ms=150.0,
            health_status=HealthStatus.GOOD,
            health_score=85.0,
            health_trend=TrendDirection.STABLE,
            thermal_performance=ThermalPerformanceResult(
                actual_duty_kw=950.0,
                design_duty_kw=1000.0,
                duty_ratio=0.95,
                lmtd_c=30.0,
                lmtd_correction_factor=1.0,
                corrected_lmtd_c=30.0,
                approach_temperature_c=10.0,
                u_clean_w_m2k=500.0,
                u_actual_w_m2k=450.0,
                u_design_w_m2k=450.0,
                u_degradation_percent=10.0,
                ntu=2.0,
                heat_capacity_ratio=0.7,
                thermal_effectiveness=0.78,
                design_effectiveness=0.85,
                effectiveness_ratio=0.92,
                calculated_fouling_m2kw=0.0002,
                effectiveness_trend=TrendDirection.STABLE,
            ),
            fouling_analysis=FoulingAnalysisResult(
                shell_side_fouling_m2kw=0.0001,
                tube_side_fouling_m2kw=0.0001,
                total_fouling_m2kw=0.0002,
                design_fouling_m2kw=0.00034,
                fouling_ratio=0.59,
                fouling_rate_m2kw_per_day=0.000001,
                fouling_trend=TrendDirection.STABLE,
                current_fouling_factor=0.59,
            ),
            active_alerts=[],
            alert_count_by_severity={"critical": 0, "alarm": 0, "warning": 0, "info": 0},
            kpis={"effectiveness": 78.0, "u_degradation": 10.0},
            analysis_methods=["e-NTU", "LMTD"],
            data_quality_score=0.95,
            model_versions={"thermal": "1.0.0"},
        )
        assert output.status == "success"
        assert output.health_score == 85.0

    def test_output_json_serialization(self):
        """Test output can be serialized to JSON."""
        output = HeatExchangerOutput(
            request_id="req-001",
            exchanger_id="E-1001",
            timestamp=datetime.now(timezone.utc),
            status="success",
            processing_time_ms=150.0,
            health_status=HealthStatus.GOOD,
            health_score=85.0,
            health_trend=TrendDirection.STABLE,
            thermal_performance=ThermalPerformanceResult(
                actual_duty_kw=950.0,
                design_duty_kw=1000.0,
                duty_ratio=0.95,
                lmtd_c=30.0,
                lmtd_correction_factor=1.0,
                corrected_lmtd_c=30.0,
                approach_temperature_c=10.0,
                u_clean_w_m2k=500.0,
                u_actual_w_m2k=450.0,
                u_design_w_m2k=450.0,
                u_degradation_percent=10.0,
                ntu=2.0,
                heat_capacity_ratio=0.7,
                thermal_effectiveness=0.78,
                design_effectiveness=0.85,
                effectiveness_ratio=0.92,
                calculated_fouling_m2kw=0.0002,
                effectiveness_trend=TrendDirection.STABLE,
            ),
            fouling_analysis=FoulingAnalysisResult(
                shell_side_fouling_m2kw=0.0001,
                tube_side_fouling_m2kw=0.0001,
                total_fouling_m2kw=0.0002,
                design_fouling_m2kw=0.00034,
                fouling_ratio=0.59,
                fouling_rate_m2kw_per_day=0.000001,
                fouling_trend=TrendDirection.STABLE,
                current_fouling_factor=0.59,
            ),
            active_alerts=[],
            alert_count_by_severity={},
            kpis={},
            analysis_methods=[],
            data_quality_score=0.9,
            model_versions={},
        )
        json_str = output.json()
        assert "E-1001" in json_str
        assert "success" in json_str

    def test_output_provenance_hash(self):
        """Test output with provenance hash."""
        output = HeatExchangerOutput(
            request_id="req-001",
            exchanger_id="E-1001",
            timestamp=datetime.now(timezone.utc),
            status="success",
            processing_time_ms=150.0,
            health_status=HealthStatus.GOOD,
            health_score=85.0,
            health_trend=TrendDirection.STABLE,
            thermal_performance=ThermalPerformanceResult(
                actual_duty_kw=950.0,
                design_duty_kw=1000.0,
                duty_ratio=0.95,
                lmtd_c=30.0,
                lmtd_correction_factor=1.0,
                corrected_lmtd_c=30.0,
                approach_temperature_c=10.0,
                u_clean_w_m2k=500.0,
                u_actual_w_m2k=450.0,
                u_design_w_m2k=450.0,
                u_degradation_percent=10.0,
                ntu=2.0,
                heat_capacity_ratio=0.7,
                thermal_effectiveness=0.78,
                design_effectiveness=0.85,
                effectiveness_ratio=0.92,
                calculated_fouling_m2kw=0.0002,
                effectiveness_trend=TrendDirection.STABLE,
            ),
            fouling_analysis=FoulingAnalysisResult(
                shell_side_fouling_m2kw=0.0001,
                tube_side_fouling_m2kw=0.0001,
                total_fouling_m2kw=0.0002,
                design_fouling_m2kw=0.00034,
                fouling_ratio=0.59,
                fouling_rate_m2kw_per_day=0.000001,
                fouling_trend=TrendDirection.STABLE,
                current_fouling_factor=0.59,
            ),
            active_alerts=[],
            alert_count_by_severity={},
            kpis={},
            analysis_methods=[],
            data_quality_score=0.9,
            model_versions={},
            provenance_hash="a" * 64,
        )
        assert output.provenance_hash is not None
        assert len(output.provenance_hash) == 64


class TestHeatExchangerInput:
    """Tests for HeatExchangerInput schema."""

    def test_create_heat_exchanger_input(self, operating_data):
        """Test creating heat exchanger input."""
        input_data = HeatExchangerInput(
            exchanger_id="E-1001",
            operating_data=operating_data,
            time_since_last_cleaning_days=90.0,
            running_hours=40000.0,
        )
        assert input_data.exchanger_id == "E-1001"
        assert input_data.running_hours == 40000.0

    def test_input_with_history(self, operating_data, tube_inspection_data, cleaning_record):
        """Test input with historical data."""
        input_data = HeatExchangerInput(
            exchanger_id="E-1001",
            operating_data=operating_data,
            operating_history=[operating_data],
            inspection_data=tube_inspection_data,
            cleaning_history=[cleaning_record],
            time_since_last_cleaning_days=90.0,
        )
        assert len(input_data.operating_history) == 1
        assert len(input_data.cleaning_history) == 1
        assert input_data.inspection_data is not None

    def test_hot_side_temperature_property(self, operating_data):
        """Test hot side inlet temperature property."""
        input_data = HeatExchangerInput(
            exchanger_id="E-1001",
            operating_data=operating_data,
        )
        # Shell side is hotter (150C vs 30C)
        assert input_data.hot_side_inlet_temp_c == 150.0

    def test_cold_side_temperature_property(self, operating_data):
        """Test cold side inlet temperature property."""
        input_data = HeatExchangerInput(
            exchanger_id="E-1001",
            operating_data=operating_data,
        )
        # Tube side is colder (30C vs 150C)
        assert input_data.cold_side_inlet_temp_c == 30.0
