# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Schema Tests

Unit tests for schemas.py module including all Pydantic models,
validation rules, and data serialization.

Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from pydantic import ValidationError
import json

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    TrapStatus,
    DiagnosisConfidence,
    TrendDirection,
    MaintenancePriority,
    SurveyStatus,
    SensorReading,
    UltrasonicReading,
    TemperatureReading,
    VisualInspectionReading,
    TrapInfo,
    TrapDiagnosticInput,
    TrapDiagnosticOutput,
    TrapCondition,
    TrapHealthScore,
    SteamLossEstimate,
    MaintenanceRecommendation,
    FailureModeProbability,
    CondensateLoadInput,
    CondensateLoadOutput,
    CondensateCalculation,
    TrapSurveyInput,
    SurveyRouteOutput,
    RouteStop,
    TrapStatusSummary,
    EconomicAnalysisOutput,
)


class TestTrapStatusEnum:
    """Tests for TrapStatus enumeration."""

    def test_all_statuses_defined(self):
        """Verify all expected trap statuses are defined."""
        expected = ["good", "failed_open", "failed_closed", "leaking", "cold", "flooded", "unknown"]
        actual = [s.value for s in TrapStatus]

        for status in expected:
            assert status in actual, f"Missing status: {status}"

    def test_status_string_values(self):
        """Verify statuses are string enums."""
        assert TrapStatus.GOOD.value == "good"
        assert TrapStatus.FAILED_OPEN.value == "failed_open"
        assert TrapStatus.FAILED_CLOSED.value == "failed_closed"

    def test_status_from_string(self):
        """Test creating status from string."""
        status = TrapStatus("failed_open")
        assert status == TrapStatus.FAILED_OPEN


class TestDiagnosisConfidenceEnum:
    """Tests for DiagnosisConfidence enumeration."""

    def test_confidence_levels(self):
        """Verify all confidence levels defined."""
        expected = ["high", "medium", "low", "uncertain"]
        actual = [c.value for c in DiagnosisConfidence]

        for level in expected:
            assert level in actual


class TestMaintenancePriorityEnum:
    """Tests for MaintenancePriority enumeration."""

    def test_priority_levels(self):
        """Verify all priority levels defined."""
        expected = ["urgent", "high", "medium", "low", "routine"]
        actual = [p.value for p in MaintenancePriority]

        for priority in expected:
            assert priority in actual


class TestUltrasonicReading:
    """Tests for UltrasonicReading schema."""

    def test_valid_creation(self):
        """Test creating valid ultrasonic reading."""
        reading = UltrasonicReading(
            sensor_id="SENSOR-001",
            timestamp=datetime.now(timezone.utc),
            quality_score=0.95,
            decibel_level_db=55.0,
            frequency_khz=38.0,
        )

        assert reading.sensor_id == "SENSOR-001"
        assert reading.decibel_level_db == 55.0
        assert reading.frequency_khz == 38.0

    def test_optional_fields(self):
        """Test optional fields have defaults."""
        reading = UltrasonicReading(
            sensor_id="SENSOR-001",
            timestamp=datetime.now(timezone.utc),
            decibel_level_db=55.0,
        )

        assert reading.quality_score == 1.0  # Default
        assert reading.cycling_detected is False
        assert reading.continuous_flow_detected is False

    def test_cycling_detection(self):
        """Test cycling detection field."""
        reading = UltrasonicReading(
            sensor_id="SENSOR-001",
            timestamp=datetime.now(timezone.utc),
            decibel_level_db=55.0,
            cycling_detected=True,
            cycle_period_s=10.5,
        )

        assert reading.cycling_detected is True
        assert reading.cycle_period_s == 10.5

    def test_json_serialization(self):
        """Test JSON serialization."""
        reading = UltrasonicReading(
            sensor_id="SENSOR-001",
            timestamp=datetime.now(timezone.utc),
            decibel_level_db=55.0,
        )

        json_str = reading.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["sensor_id"] == "SENSOR-001"
        assert parsed["decibel_level_db"] == 55.0

    def test_quality_score_range(self):
        """Test quality score validation."""
        # Valid score
        reading = UltrasonicReading(
            sensor_id="SENSOR-001",
            timestamp=datetime.now(timezone.utc),
            decibel_level_db=55.0,
            quality_score=0.5,
        )
        assert reading.quality_score == 0.5

    def test_invalid_quality_score(self):
        """Test invalid quality score is rejected."""
        with pytest.raises(ValidationError):
            UltrasonicReading(
                sensor_id="SENSOR-001",
                timestamp=datetime.now(timezone.utc),
                decibel_level_db=55.0,
                quality_score=1.5,  # Invalid: > 1.0
            )


class TestTemperatureReading:
    """Tests for TemperatureReading schema."""

    def test_valid_creation(self):
        """Test creating valid temperature reading."""
        reading = TemperatureReading(
            sensor_id="SENSOR-002",
            timestamp=datetime.now(timezone.utc),
            inlet_temp_f=366.0,
            outlet_temp_f=340.0,
        )

        assert reading.inlet_temp_f == 366.0
        assert reading.outlet_temp_f == 340.0

    def test_delta_t_calculation(self):
        """Test delta T is calculated or provided."""
        reading = TemperatureReading(
            sensor_id="SENSOR-002",
            timestamp=datetime.now(timezone.utc),
            inlet_temp_f=366.0,
            outlet_temp_f=340.0,
            delta_t_f=26.0,
        )

        assert reading.delta_t_f == 26.0

    def test_ambient_temperature(self):
        """Test ambient temperature field."""
        reading = TemperatureReading(
            sensor_id="SENSOR-002",
            timestamp=datetime.now(timezone.utc),
            inlet_temp_f=366.0,
            outlet_temp_f=340.0,
            ambient_temp_f=72.0,
        )

        assert reading.ambient_temp_f == 72.0


class TestVisualInspectionReading:
    """Tests for VisualInspectionReading schema."""

    def test_valid_creation(self):
        """Test creating valid visual inspection."""
        inspection = VisualInspectionReading(
            inspector_id="INSPECTOR-01",
            timestamp=datetime.now(timezone.utc),
            visible_steam_discharge=False,
            condensate_visible=True,
            trap_cycling_observed=True,
        )

        assert inspection.inspector_id == "INSPECTOR-01"
        assert inspection.visible_steam_discharge is False
        assert inspection.trap_cycling_observed is True

    def test_steam_discharge_detection(self):
        """Test steam discharge detection flags failed open."""
        inspection = VisualInspectionReading(
            inspector_id="INSPECTOR-01",
            timestamp=datetime.now(timezone.utc),
            visible_steam_discharge=True,
            condensate_visible=False,
        )

        assert inspection.visible_steam_discharge is True

    def test_condition_fields(self):
        """Test condition assessment fields."""
        inspection = VisualInspectionReading(
            inspector_id="INSPECTOR-01",
            timestamp=datetime.now(timezone.utc),
            visible_steam_discharge=False,
            trap_body_condition="corroded",
            insulation_condition="damaged",
            leaks_detected=True,
        )

        assert inspection.trap_body_condition == "corroded"
        assert inspection.leaks_detected is True


class TestTrapInfo:
    """Tests for TrapInfo schema."""

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        trap = TrapInfo(
            trap_id="ST-0001",
            trap_type="float_thermostatic",
        )

        assert trap.trap_id == "ST-0001"
        assert trap.trap_type == "float_thermostatic"

    def test_full_creation(self, sample_trap_info):
        """Test creating with all fields."""
        trap = sample_trap_info

        assert trap.trap_id is not None
        assert trap.trap_type == "float_thermostatic"
        assert trap.pressure_rating_psig == 150.0
        assert trap.application == "drip_leg"

    def test_manufacturer_info(self):
        """Test manufacturer information."""
        trap = TrapInfo(
            trap_id="ST-0001",
            trap_type="inverted_bucket",
            manufacturer="Armstrong",
            model="811",
        )

        assert trap.manufacturer == "Armstrong"
        assert trap.model == "811"

    def test_location_fields(self):
        """Test location-related fields."""
        trap = TrapInfo(
            trap_id="ST-0001",
            trap_type="thermodynamic",
            location="Building A, Level 2, Rack 5",
            area_code="AREA-01",
            gps_coordinates=(40.7128, -74.0060),
        )

        assert trap.location == "Building A, Level 2, Rack 5"
        assert trap.area_code == "AREA-01"
        assert trap.gps_coordinates == (40.7128, -74.0060)


class TestTrapCondition:
    """Tests for TrapCondition schema."""

    def test_good_condition(self):
        """Test good trap condition."""
        condition = TrapCondition(
            status=TrapStatus.GOOD,
            confidence=DiagnosisConfidence.HIGH,
            confidence_score=0.92,
            evidence=["Normal cycling detected", "Temperature differential appropriate"],
        )

        assert condition.status == TrapStatus.GOOD
        assert condition.confidence_score == 0.92
        assert len(condition.evidence) == 2

    def test_failed_condition(self):
        """Test failed open condition."""
        condition = TrapCondition(
            status=TrapStatus.FAILED_OPEN,
            confidence=DiagnosisConfidence.HIGH,
            confidence_score=0.88,
            failed_open_probability=0.88,
            evidence=["Continuous ultrasonic flow detected"],
        )

        assert condition.status == TrapStatus.FAILED_OPEN
        assert condition.failed_open_probability == 0.88

    def test_assessment_fields(self):
        """Test individual assessment fields."""
        condition = TrapCondition(
            status=TrapStatus.LEAKING,
            confidence=DiagnosisConfidence.MEDIUM,
            confidence_score=0.72,
            ultrasonic_assessment="leaking",
            temperature_assessment="good",
            visual_assessment=None,
            inconsistencies=["Methods disagree: ultrasonic vs temperature"],
        )

        assert condition.ultrasonic_assessment == "leaking"
        assert condition.temperature_assessment == "good"
        assert len(condition.inconsistencies) == 1


class TestTrapHealthScore:
    """Tests for TrapHealthScore schema."""

    def test_excellent_score(self):
        """Test excellent health score."""
        health = TrapHealthScore(
            overall_score=95.0,
            category="excellent",
            thermal_efficiency_score=96.0,
            mechanical_condition_score=94.0,
            operational_score=95.0,
        )

        assert health.overall_score == 95.0
        assert health.category == "excellent"

    def test_critical_score(self):
        """Test critical health score."""
        health = TrapHealthScore(
            overall_score=15.0,
            category="critical",
            thermal_efficiency_score=10.0,
            mechanical_condition_score=20.0,
            operational_score=15.0,
        )

        assert health.overall_score == 15.0
        assert health.category == "critical"

    def test_trend_direction(self):
        """Test trend direction tracking."""
        health = TrapHealthScore(
            overall_score=70.0,
            category="good",
            trend=TrendDirection.DECLINING,
        )

        assert health.trend == TrendDirection.DECLINING


class TestSteamLossEstimate:
    """Tests for SteamLossEstimate schema."""

    def test_minimal_creation(self):
        """Test creating with no losses (good trap)."""
        loss = SteamLossEstimate()

        assert loss.steam_loss_lb_hr == 0.0
        assert loss.cost_per_year_usd == 0.0

    def test_full_loss_calculation(self):
        """Test full loss estimate."""
        loss = SteamLossEstimate(
            steam_loss_lb_hr=26.5,
            steam_loss_lb_year=232140.0,
            energy_loss_mmbtu_hr=0.0283,
            energy_loss_mmbtu_year=247.9,
            cost_per_hour_usd=0.35,
            cost_per_year_usd=3098.75,
            co2_emissions_lb_hr=3.31,
            co2_emissions_tons_year=14.5,
            calculation_method="napier_orifice",
            orifice_diameter_in=0.1875,
        )

        assert loss.steam_loss_lb_hr == 26.5
        assert loss.cost_per_year_usd == 3098.75
        assert loss.co2_emissions_tons_year == 14.5


class TestMaintenanceRecommendation:
    """Tests for MaintenanceRecommendation schema."""

    def test_urgent_recommendation(self):
        """Test urgent priority recommendation."""
        rec = MaintenanceRecommendation(
            priority=MaintenancePriority.URGENT,
            action="Replace steam trap",
            description="Trap has failed open and is passing live steam",
            deadline_hours=24.0,
            reason="Failed open - continuous steam loss",
        )

        assert rec.priority == MaintenancePriority.URGENT
        assert rec.deadline_hours == 24.0

    def test_with_cost_estimate(self):
        """Test recommendation with cost estimate."""
        rec = MaintenanceRecommendation(
            priority=MaintenancePriority.HIGH,
            action="Repair steam trap",
            description="Clear blockage and replace internals",
            deadline_hours=48.0,
            estimated_duration_hours=2.0,
            estimated_cost_usd=450.0,
            parts_required=["Repair kit", "Gaskets", "Screen"],
            reason="Failed closed - condensate backup",
        )

        assert rec.estimated_cost_usd == 450.0
        assert len(rec.parts_required) == 3

    def test_with_savings_estimate(self):
        """Test recommendation with savings estimate."""
        rec = MaintenanceRecommendation(
            priority=MaintenancePriority.MEDIUM,
            action="Inspect and repair",
            description="Address minor steam leak",
            deadline_hours=168.0,
            potential_savings_usd=2500.0,
            reason="Steam leakage detected",
        )

        assert rec.potential_savings_usd == 2500.0


class TestFailureModeProbability:
    """Tests for FailureModeProbability schema."""

    def test_high_probability_failure(self):
        """Test high probability failure mode."""
        prob = FailureModeProbability(
            failure_mode="failed_open",
            probability=0.88,
            confidence=DiagnosisConfidence.HIGH,
            indicators=["Continuous ultrasonic", "Low delta T"],
            contradictors=[],
        )

        assert prob.probability == 0.88
        assert len(prob.indicators) == 2

    def test_uncertain_failure(self):
        """Test uncertain failure mode."""
        prob = FailureModeProbability(
            failure_mode="leaking",
            probability=0.45,
            confidence=DiagnosisConfidence.LOW,
            indicators=["Elevated ultrasonic"],
            contradictors=["Normal temperature differential"],
        )

        assert prob.probability == 0.45
        assert len(prob.contradictors) == 1


class TestTrapDiagnosticInput:
    """Tests for TrapDiagnosticInput schema."""

    def test_minimal_input(self, sample_trap_info):
        """Test minimal diagnostic input."""
        input_data = TrapDiagnosticInput(
            trap_info=sample_trap_info,
            steam_pressure_psig=150.0,
        )

        assert input_data.trap_info == sample_trap_info
        assert input_data.steam_pressure_psig == 150.0

    def test_full_input(self, sample_diagnostic_input_good):
        """Test full diagnostic input."""
        input_data = sample_diagnostic_input_good

        assert input_data.trap_info is not None
        assert len(input_data.ultrasonic_readings) > 0
        assert len(input_data.temperature_readings) > 0

    def test_request_id_generation(self, sample_trap_info):
        """Test request ID is generated if not provided."""
        input_data = TrapDiagnosticInput(
            trap_info=sample_trap_info,
            steam_pressure_psig=150.0,
        )

        assert input_data.request_id is not None
        assert len(input_data.request_id) > 0


class TestTrapDiagnosticOutput:
    """Tests for TrapDiagnosticOutput schema."""

    def test_valid_output_creation(self):
        """Test creating valid diagnostic output."""
        output = TrapDiagnosticOutput(
            request_id="REQ-12345",
            trap_id="ST-0001",
            timestamp=datetime.now(timezone.utc),
            status="success",
            processing_time_ms=15.5,
            condition=TrapCondition(
                status=TrapStatus.GOOD,
                confidence=DiagnosisConfidence.HIGH,
                confidence_score=0.92,
            ),
            health_score=TrapHealthScore(
                overall_score=95.0,
                category="excellent",
            ),
            steam_loss=SteamLossEstimate(),
            provenance_hash="a" * 64,
        )

        assert output.status == "success"
        assert output.condition.status == TrapStatus.GOOD

    def test_compliance_fields(self):
        """Test compliance-related fields."""
        output = TrapDiagnosticOutput(
            request_id="REQ-12345",
            trap_id="ST-0001",
            timestamp=datetime.now(timezone.utc),
            status="success",
            processing_time_ms=15.5,
            condition=TrapCondition(
                status=TrapStatus.GOOD,
                confidence=DiagnosisConfidence.HIGH,
                confidence_score=0.92,
            ),
            health_score=TrapHealthScore(
                overall_score=95.0,
                category="excellent",
            ),
            steam_loss=SteamLossEstimate(),
            asme_b16_34_compliant=True,
            pressure_rating_adequate=True,
            provenance_hash="a" * 64,
        )

        assert output.asme_b16_34_compliant is True
        assert output.pressure_rating_adequate is True


class TestCondensateLoadInput:
    """Tests for CondensateLoadInput schema."""

    def test_drip_leg_input(self, sample_condensate_load_input):
        """Test drip leg condensate load input."""
        input_data = sample_condensate_load_input

        assert input_data.application == "drip_leg"
        assert input_data.steam_pressure_psig == 150.0
        assert input_data.pipe_diameter_in == 4.0
        assert input_data.pipe_length_ft == 100.0

    def test_heat_exchanger_input(self):
        """Test heat exchanger condensate load input."""
        input_data = CondensateLoadInput(
            application="heat_exchanger",
            steam_pressure_psig=100.0,
            heat_load_btu_hr=500000.0,
        )

        assert input_data.application == "heat_exchanger"
        assert input_data.heat_load_btu_hr == 500000.0


class TestCondensateLoadOutput:
    """Tests for CondensateLoadOutput schema."""

    def test_load_output_creation(self):
        """Test condensate load output creation."""
        output = CondensateLoadOutput(
            request_id="REQ-12345",
            timestamp=datetime.now(timezone.utc),
            status="success",
            application="drip_leg",
            operating_condensate_lb_hr=25.0,
            startup_condensate_lb_hr=75.0,
            safety_allowance_pct=200.0,
            total_condensate_lb_hr=75.0,
            recommended_trap_capacity_lb_hr=225.0,
            provenance_hash="a" * 64,
        )

        assert output.operating_condensate_lb_hr == 25.0
        assert output.startup_condensate_lb_hr == 75.0
        assert output.recommended_trap_capacity_lb_hr == 225.0


class TestTrapSurveyInput:
    """Tests for TrapSurveyInput schema."""

    def test_survey_input_creation(self, sample_trap_survey_input):
        """Test survey input creation."""
        input_data = sample_trap_survey_input

        assert input_data.plant_id == "TEST-PLANT-001"
        assert len(input_data.trap_ids) == 25
        assert input_data.max_traps_per_route == 50


class TestSurveyRouteOutput:
    """Tests for SurveyRouteOutput schema."""

    def test_route_output_creation(self):
        """Test survey route output creation."""
        stops = [
            RouteStop(
                sequence=1,
                trap_id="ST-0001",
                area_code="AREA-01",
                estimated_time_minutes=5.0,
                priority="normal",
            ),
            RouteStop(
                sequence=2,
                trap_id="ST-0002",
                area_code="AREA-01",
                estimated_time_minutes=5.0,
                priority="high",
            ),
        ]

        output = SurveyRouteOutput(
            request_id="REQ-12345",
            total_routes=1,
            total_traps=2,
            total_distance_ft=150.0,
            total_time_hours=0.17,
            routes=[stops],
            optimization_method="nearest_neighbor",
            coverage_by_area={"AREA-01": 2},
            provenance_hash="a" * 64,
        )

        assert output.total_traps == 2
        assert len(output.routes[0]) == 2


class TestTrapStatusSummary:
    """Tests for TrapStatusSummary schema."""

    def test_summary_creation(self):
        """Test status summary creation."""
        summary = TrapStatusSummary(
            plant_id="PLANT-001",
            total_traps=100,
            traps_good=85,
            traps_failed_open=5,
            traps_failed_closed=3,
            traps_leaking=4,
            traps_unknown=3,
            overall_failure_rate_pct=12.0,
            failed_open_rate_pct=5.0,
            survey_status=SurveyStatus.COMPLETED,
            traps_surveyed_this_cycle=100,
            priority_repairs_count=8,
        )

        assert summary.total_traps == 100
        assert summary.traps_good == 85
        assert summary.overall_failure_rate_pct == 12.0


class TestEconomicAnalysisOutput:
    """Tests for EconomicAnalysisOutput schema."""

    def test_analysis_output_creation(self):
        """Test economic analysis output creation."""
        output = EconomicAnalysisOutput(
            request_id="REQ-12345",
            total_traps_analyzed=50,
            traps_failed=6,
            failure_rate_pct=12.0,
            total_steam_loss_lb_hr=150.0,
            total_steam_loss_lb_year=1314000.0,
            total_annual_loss_usd=45000.0,
            repair_cost_usd=4500.0,
            net_annual_savings_usd=40500.0,
            simple_payback_months=1.2,
            roi_pct=900.0,
            npv_5year_usd=150000.0,
            total_co2_reduction_tons_year=65.0,
            provenance_hash="a" * 64,
        )

        assert output.failure_rate_pct == 12.0
        assert output.simple_payback_months == 1.2


class TestSchemaValidation:
    """Tests for schema validation rules."""

    def test_quality_score_bounds(self):
        """Test quality score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            UltrasonicReading(
                sensor_id="SENSOR-001",
                timestamp=datetime.now(timezone.utc),
                decibel_level_db=55.0,
                quality_score=-0.5,
            )

    def test_confidence_score_bounds(self):
        """Test confidence score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            TrapCondition(
                status=TrapStatus.GOOD,
                confidence=DiagnosisConfidence.HIGH,
                confidence_score=1.5,  # Invalid
            )

    def test_probability_bounds(self):
        """Test probability must be between 0 and 1."""
        with pytest.raises(ValidationError):
            FailureModeProbability(
                failure_mode="failed_open",
                probability=1.2,  # Invalid
                confidence=DiagnosisConfidence.HIGH,
            )


class TestSchemaSerialization:
    """Tests for schema JSON serialization."""

    def test_diagnostic_output_json(self):
        """Test diagnostic output serializes to JSON."""
        output = TrapDiagnosticOutput(
            request_id="REQ-12345",
            trap_id="ST-0001",
            timestamp=datetime.now(timezone.utc),
            status="success",
            processing_time_ms=15.5,
            condition=TrapCondition(
                status=TrapStatus.GOOD,
                confidence=DiagnosisConfidence.HIGH,
                confidence_score=0.92,
            ),
            health_score=TrapHealthScore(
                overall_score=95.0,
                category="excellent",
            ),
            steam_loss=SteamLossEstimate(),
            provenance_hash="a" * 64,
        )

        json_str = output.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["request_id"] == "REQ-12345"
        assert parsed["trap_id"] == "ST-0001"
        assert parsed["condition"]["status"] == "good"

    def test_round_trip_serialization(self, sample_diagnostic_input_good):
        """Test input serializes and deserializes correctly."""
        json_str = sample_diagnostic_input_good.model_dump_json()
        parsed = TrapDiagnosticInput.model_validate_json(json_str)

        assert parsed.trap_info.trap_id == sample_diagnostic_input_good.trap_info.trap_id
        assert parsed.steam_pressure_psig == sample_diagnostic_input_good.steam_pressure_psig
