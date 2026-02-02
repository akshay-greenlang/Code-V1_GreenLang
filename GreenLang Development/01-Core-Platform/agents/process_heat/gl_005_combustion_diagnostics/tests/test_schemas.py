# -*- coding: utf-8 -*-
"""
GL-005 Schema Tests
===================

Unit tests for GL-005 data schemas module.
Tests all Pydantic models for input/output validation.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    FuelCategory,
    AnomalyType,
    MaintenancePriority,
    ComplianceFramework,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    # Enums
    CQIRating,
    AnomalySeverity,
    TrendDirection,
    AnalysisStatus,
    # Input schemas
    FlueGasReading,
    CombustionOperatingData,
    DiagnosticsInput,
    # Output schemas - CQI
    CQIComponentScore,
    CQIResult,
    # Output schemas - Anomaly
    AnomalyEvent,
    AnomalyDetectionResult,
    # Output schemas - Fuel
    FuelProperties,
    FuelCharacterizationResult,
    # Output schemas - Maintenance
    MaintenanceRecommendation,
    FoulingAssessment,
    BurnerWearAssessment,
    MaintenanceAdvisoryResult,
    CMMSWorkOrder,
    # Output schemas - Compliance
    ComplianceStatus,
    ComplianceReportResult,
    # Complete output
    DiagnosticsOutput,
)


class TestSchemaEnums:
    """Tests for schema enums."""

    def test_cqi_rating_values(self):
        """Test CQI rating enum values."""
        assert CQIRating.EXCELLENT.value == "excellent"
        assert CQIRating.GOOD.value == "good"
        assert CQIRating.ACCEPTABLE.value == "acceptable"
        assert CQIRating.POOR.value == "poor"
        assert CQIRating.CRITICAL.value == "critical"

    def test_anomaly_severity_values(self):
        """Test anomaly severity enum values."""
        assert AnomalySeverity.INFO.value == "info"
        assert AnomalySeverity.WARNING.value == "warning"
        assert AnomalySeverity.ALARM.value == "alarm"
        assert AnomalySeverity.CRITICAL.value == "critical"

    def test_trend_direction_values(self):
        """Test trend direction enum values."""
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.DEGRADING.value == "degrading"
        assert TrendDirection.UNKNOWN.value == "unknown"

    def test_analysis_status_values(self):
        """Test analysis status enum values."""
        assert AnalysisStatus.SUCCESS.value == "success"
        assert AnalysisStatus.PARTIAL.value == "partial"
        assert AnalysisStatus.FAILED.value == "failed"
        assert AnalysisStatus.INSUFFICIENT_DATA.value == "insufficient_data"


class TestFlueGasReading:
    """Tests for FlueGasReading schema."""

    def test_valid_optimal_reading(self, optimal_flue_gas_reading):
        """Test valid optimal flue gas reading."""
        reading = optimal_flue_gas_reading
        assert reading.oxygen_pct == 3.0
        assert reading.co2_pct == 10.5
        assert reading.co_ppm == 25.0
        assert reading.nox_ppm == 40.0

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            FlueGasReading()  # Missing required fields

    def test_minimal_valid_reading(self):
        """Test minimal valid reading."""
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.0,
            co_ppm=30.0,
            flue_gas_temp_c=180.0,
        )
        assert reading.oxygen_pct == 3.0
        # Check defaults
        assert reading.nox_ppm == 0.0
        assert reading.sensor_status == "ok"
        assert reading.data_quality_flag == "good"

    def test_oxygen_bounds(self):
        """Test oxygen percentage bounds."""
        # O2 must be between 0 and 21
        with pytest.raises(ValidationError):
            FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                oxygen_pct=-1.0,
                co2_pct=10.0,
                co_ppm=30.0,
                flue_gas_temp_c=180.0,
            )

        with pytest.raises(ValidationError):
            FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                oxygen_pct=25.0,
                co2_pct=10.0,
                co_ppm=30.0,
                flue_gas_temp_c=180.0,
            )

    def test_oxygen_safety_validation(self):
        """Test oxygen safety level validation."""
        # O2 below 0.5% should raise error
        with pytest.raises(ValidationError):
            FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                oxygen_pct=0.3,
                co2_pct=12.0,
                co_ppm=500.0,
                flue_gas_temp_c=180.0,
            )

    def test_co_bounds(self):
        """Test CO concentration bounds."""
        # CO must be non-negative
        with pytest.raises(ValidationError):
            FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                oxygen_pct=3.0,
                co2_pct=10.0,
                co_ppm=-10.0,
                flue_gas_temp_c=180.0,
            )

    def test_temperature_bounds(self):
        """Test flue gas temperature bounds."""
        # Temperature must be reasonable
        with pytest.raises(ValidationError):
            FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                oxygen_pct=3.0,
                co2_pct=10.0,
                co_ppm=30.0,
                flue_gas_temp_c=30.0,  # Too low
            )

    def test_optional_fields(self):
        """Test optional fields."""
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.0,
            co_ppm=30.0,
            flue_gas_temp_c=180.0,
            so2_ppm=5.0,
            combustibles_pct=0.1,
            moisture_pct=8.0,
        )
        assert reading.so2_ppm == 5.0
        assert reading.combustibles_pct == 0.1
        assert reading.moisture_pct == 8.0


class TestCombustionOperatingData:
    """Tests for CombustionOperatingData schema."""

    def test_valid_operating_data(self, normal_operating_data):
        """Test valid operating data."""
        data = normal_operating_data
        assert data.firing_rate_pct == 75.0
        assert data.fuel_type == FuelCategory.NATURAL_GAS
        assert data.burner_status == "modulating"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            CombustionOperatingData()

    def test_firing_rate_bounds(self):
        """Test firing rate bounds."""
        with pytest.raises(ValidationError):
            CombustionOperatingData(
                timestamp=datetime.now(timezone.utc),
                firing_rate_pct=-10.0,
                fuel_flow_rate=100.0,
            )

        # Allow slightly over 100% for transients
        data = CombustionOperatingData(
            timestamp=datetime.now(timezone.utc),
            firing_rate_pct=105.0,
            fuel_flow_rate=100.0,
        )
        assert data.firing_rate_pct == 105.0

    def test_fuel_flow_bounds(self):
        """Test fuel flow rate bounds."""
        with pytest.raises(ValidationError):
            CombustionOperatingData(
                timestamp=datetime.now(timezone.utc),
                firing_rate_pct=75.0,
                fuel_flow_rate=-50.0,
            )


class TestDiagnosticsInput:
    """Tests for DiagnosticsInput schema."""

    def test_valid_input(self, diagnostics_input):
        """Test valid diagnostics input."""
        input_data = diagnostics_input
        assert input_data.equipment_id == "BLR-TEST-001"
        assert input_data.request_id == "REQ-TEST-001"
        assert input_data.run_cqi_analysis is True

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DiagnosticsInput()

    def test_analysis_flags(self):
        """Test analysis flag defaults."""
        input_data = DiagnosticsInput(
            equipment_id="BLR-001",
            request_id="REQ-001",
            flue_gas=FlueGasReading(
                timestamp=datetime.now(timezone.utc),
                oxygen_pct=3.0,
                co2_pct=10.0,
                co_ppm=30.0,
                flue_gas_temp_c=180.0,
            ),
            operating_data=CombustionOperatingData(
                timestamp=datetime.now(timezone.utc),
                firing_rate_pct=75.0,
                fuel_flow_rate=100.0,
            ),
        )
        # Default analysis flags should be True
        assert input_data.run_cqi_analysis is True
        assert input_data.run_anomaly_detection is True


class TestCQIComponentScore:
    """Tests for CQIComponentScore schema."""

    def test_valid_component_score(self):
        """Test valid component score."""
        score = CQIComponentScore(
            component="oxygen",
            raw_value=3.0,
            normalized_score=95.0,
            weight=0.25,
            weighted_score=23.75,
            status="optimal",
        )
        assert score.component == "oxygen"
        assert score.normalized_score == 95.0

    def test_score_bounds(self):
        """Test normalized score bounds."""
        with pytest.raises(ValidationError):
            CQIComponentScore(
                component="oxygen",
                raw_value=3.0,
                normalized_score=105.0,  # Over 100
                weight=0.25,
                weighted_score=26.25,
                status="optimal",
            )


class TestCQIResult:
    """Tests for CQIResult schema."""

    def test_valid_cqi_result(self):
        """Test valid CQI result."""
        result = CQIResult(
            cqi_score=85.5,
            cqi_rating=CQIRating.GOOD,
            components=[
                CQIComponentScore(
                    component="oxygen",
                    raw_value=3.0,
                    normalized_score=95.0,
                    weight=0.25,
                    weighted_score=23.75,
                    status="optimal",
                ),
            ],
            co_corrected_ppm=35.0,
            nox_corrected_ppm=50.0,
            o2_reference_pct=3.0,
            excess_air_pct=15.0,
            combustion_efficiency_pct=88.5,
            calculation_timestamp=datetime.now(timezone.utc),
            provenance_hash="a" * 64,
        )
        assert result.cqi_score == 85.5
        assert result.cqi_rating == CQIRating.GOOD

    def test_cqi_score_bounds(self):
        """Test CQI score bounds."""
        with pytest.raises(ValidationError):
            CQIResult(
                cqi_score=105.0,  # Over 100
                cqi_rating=CQIRating.EXCELLENT,
                components=[],
                co_corrected_ppm=35.0,
                nox_corrected_ppm=50.0,
                o2_reference_pct=3.0,
                excess_air_pct=15.0,
                combustion_efficiency_pct=88.5,
                calculation_timestamp=datetime.now(timezone.utc),
                provenance_hash="a" * 64,
            )


class TestAnomalyEvent:
    """Tests for AnomalyEvent schema."""

    def test_valid_anomaly_event(self):
        """Test valid anomaly event."""
        event = AnomalyEvent(
            anomaly_id="ANO-001",
            timestamp=datetime.now(timezone.utc),
            anomaly_type=AnomalyType.HIGH_CO,
            severity=AnomalySeverity.ALARM,
            detection_method="rule_based",
            confidence=0.95,
            observed_value=500.0,
            expected_value=50.0,
            deviation_pct=900.0,
            affected_parameter="co_ppm",
            potential_causes=["Insufficient combustion air", "Burner fouling"],
            recommended_actions=["Increase air flow", "Inspect burner"],
        )
        assert event.anomaly_type == AnomalyType.HIGH_CO
        assert event.severity == AnomalySeverity.ALARM

    def test_confidence_bounds(self):
        """Test confidence bounds."""
        with pytest.raises(ValidationError):
            AnomalyEvent(
                anomaly_id="ANO-001",
                timestamp=datetime.now(timezone.utc),
                anomaly_type=AnomalyType.HIGH_CO,
                severity=AnomalySeverity.ALARM,
                detection_method="rule_based",
                confidence=1.5,  # Over 1.0
                observed_value=500.0,
                expected_value=50.0,
                deviation_pct=900.0,
                affected_parameter="co_ppm",
            )


class TestAnomalyDetectionResult:
    """Tests for AnomalyDetectionResult schema."""

    def test_valid_result_no_anomalies(self):
        """Test valid result with no anomalies."""
        result = AnomalyDetectionResult(
            status=AnalysisStatus.SUCCESS,
            anomaly_detected=False,
            total_anomalies=0,
            analysis_timestamp=datetime.now(timezone.utc),
            provenance_hash="b" * 64,
        )
        assert result.anomaly_detected is False
        assert result.spc_in_control is True

    def test_valid_result_with_anomalies(self):
        """Test valid result with anomalies."""
        result = AnomalyDetectionResult(
            status=AnalysisStatus.SUCCESS,
            anomaly_detected=True,
            total_anomalies=2,
            critical_count=1,
            alarm_count=1,
            spc_in_control=False,
            spc_violations=["Rule 1: Point beyond 3sigma"],
            analysis_timestamp=datetime.now(timezone.utc),
            provenance_hash="b" * 64,
        )
        assert result.total_anomalies == 2
        assert result.spc_in_control is False


class TestFuelProperties:
    """Tests for FuelProperties schema."""

    def test_valid_fuel_properties(self):
        """Test valid fuel properties."""
        props = FuelProperties(
            fuel_category=FuelCategory.NATURAL_GAS,
            confidence=0.95,
            carbon_content_pct=75.0,
            hydrogen_content_pct=24.0,
            oxygen_content_pct=0.0,
            nitrogen_content_pct=1.0,
            sulfur_content_pct=0.0,
            hhv_mj_kg=55.5,
            lhv_mj_kg=50.0,
            stoich_air_fuel_ratio=17.2,
            theoretical_co2_pct=11.8,
            co2_emission_factor_kg_mj=0.0561,
        )
        assert props.fuel_category == FuelCategory.NATURAL_GAS
        assert props.hhv_mj_kg == 55.5


class TestFoulingAssessment:
    """Tests for FoulingAssessment schema."""

    def test_no_fouling(self):
        """Test no fouling assessment."""
        assessment = FoulingAssessment(
            fouling_detected=False,
            fouling_severity="none",
        )
        assert assessment.fouling_detected is False
        assert assessment.efficiency_loss_pct == 0.0

    def test_severe_fouling(self):
        """Test severe fouling assessment."""
        assessment = FoulingAssessment(
            fouling_detected=True,
            fouling_severity="severe",
            efficiency_loss_pct=12.0,
            stack_temp_increase_c=50.0,
            delta_t_degradation_pct=15.0,
            days_until_cleaning_recommended=3,
            assessment_confidence=0.85,
        )
        assert assessment.fouling_severity == "severe"
        assert assessment.efficiency_loss_pct == 12.0


class TestBurnerWearAssessment:
    """Tests for BurnerWearAssessment schema."""

    def test_normal_wear(self):
        """Test normal wear assessment."""
        assessment = BurnerWearAssessment(
            wear_detected=False,
            wear_level="normal",
            operating_hours=5000.0,
            expected_life_remaining_pct=75.0,
        )
        assert assessment.wear_detected is False

    def test_replacement_needed(self):
        """Test replacement needed assessment."""
        assessment = BurnerWearAssessment(
            wear_detected=True,
            wear_level="replacement_needed",
            operating_hours=19500.0,
            expected_life_remaining_pct=2.5,
            co_trend_slope=0.15,
            flame_stability_score=0.85,
            estimated_remaining_life_hours=500.0,
            replacement_recommended_by=datetime.now(timezone.utc) + timedelta(days=30),
            assessment_confidence=0.9,
        )
        assert assessment.wear_level == "replacement_needed"


class TestMaintenanceRecommendation:
    """Tests for MaintenanceRecommendation schema."""

    def test_valid_recommendation(self):
        """Test valid maintenance recommendation."""
        rec = MaintenanceRecommendation(
            recommendation_id="REC-001",
            timestamp=datetime.now(timezone.utc),
            maintenance_type="cleaning",
            priority=MaintenancePriority.HIGH,
            component="Heat Transfer Surfaces",
            title="Boiler tube cleaning required",
            description="Fouling detected causing 8% efficiency loss",
            justification="Cost savings from improved efficiency",
            recommended_by_date=datetime.now(timezone.utc) + timedelta(days=7),
            estimated_duration_hours=8.0,
            risk_if_deferred="high",
            potential_consequences=["Further efficiency loss", "Tube overheating"],
        )
        assert rec.priority == MaintenancePriority.HIGH
        assert rec.maintenance_type == "cleaning"


class TestCMMSWorkOrder:
    """Tests for CMMSWorkOrder schema."""

    def test_valid_work_order(self):
        """Test valid CMMS work order."""
        wo = CMMSWorkOrder(
            work_order_id="WO-GL005-20240115-ABC123",
            created_timestamp=datetime.now(timezone.utc),
            equipment_id="BLR-001",
            equipment_name="Boiler 1",
            location="Building A",
            work_type="PM",
            priority=MaintenancePriority.MEDIUM,
            title="Heat exchanger cleaning",
            description="Scheduled cleaning due to fouling detection",
            estimated_hours=8.0,
            source_agent="GL-005",
            source_analysis_id="ANL-001",
            provenance_hash="c" * 64,
        )
        assert wo.work_type == "PM"
        assert wo.status == "pending_approval"


class TestDiagnosticsOutput:
    """Tests for complete DiagnosticsOutput schema."""

    def test_minimal_output(self):
        """Test minimal diagnostics output."""
        output = DiagnosticsOutput(
            request_id="REQ-001",
            equipment_id="BLR-001",
            status=AnalysisStatus.SUCCESS,
            processing_time_ms=125.5,
            input_timestamp=datetime.now(timezone.utc),
            output_timestamp=datetime.now(timezone.utc),
            provenance_hash="d" * 64,
        )
        assert output.status == AnalysisStatus.SUCCESS
        assert output.cqi is None  # Optional

    def test_full_output(self):
        """Test full diagnostics output with all results."""
        now = datetime.now(timezone.utc)

        output = DiagnosticsOutput(
            request_id="REQ-001",
            equipment_id="BLR-001",
            agent_id="GL005-TEST",
            agent_version="1.0.0",
            status=AnalysisStatus.SUCCESS,
            processing_time_ms=250.0,
            cqi=CQIResult(
                cqi_score=85.0,
                cqi_rating=CQIRating.GOOD,
                components=[],
                co_corrected_ppm=35.0,
                nox_corrected_ppm=50.0,
                o2_reference_pct=3.0,
                excess_air_pct=15.0,
                combustion_efficiency_pct=88.0,
                calculation_timestamp=now,
                provenance_hash="e" * 64,
            ),
            anomaly_detection=AnomalyDetectionResult(
                status=AnalysisStatus.SUCCESS,
                anomaly_detected=False,
                analysis_timestamp=now,
                provenance_hash="f" * 64,
            ),
            alerts=[{"type": "info", "message": "All systems normal"}],
            recommendations=["Continue normal operation"],
            input_timestamp=now,
            output_timestamp=now,
            provenance_hash="g" * 64,
            audit_trail=[{"step": "cqi", "status": "success"}],
        )
        assert output.cqi.cqi_score == 85.0
        assert output.anomaly_detection.anomaly_detected is False


class TestSchemaJSONSerialization:
    """Tests for JSON serialization of schemas."""

    def test_flue_gas_reading_json(self, optimal_flue_gas_reading):
        """Test FlueGasReading JSON serialization."""
        json_str = optimal_flue_gas_reading.json()
        assert "oxygen_pct" in json_str
        assert "co_ppm" in json_str

    def test_diagnostics_output_json(self):
        """Test DiagnosticsOutput JSON serialization."""
        output = DiagnosticsOutput(
            request_id="REQ-001",
            equipment_id="BLR-001",
            status=AnalysisStatus.SUCCESS,
            processing_time_ms=100.0,
            input_timestamp=datetime.now(timezone.utc),
            output_timestamp=datetime.now(timezone.utc),
            provenance_hash="h" * 64,
        )
        json_dict = output.dict()
        assert json_dict["status"] == "success"
        assert json_dict["equipment_id"] == "BLR-001"
