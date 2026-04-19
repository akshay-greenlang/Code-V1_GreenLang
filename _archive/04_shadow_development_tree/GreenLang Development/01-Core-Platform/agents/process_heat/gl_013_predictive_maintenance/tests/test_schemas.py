# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Schema Tests

Tests for input/output schemas and data models.
Validates Pydantic models, serialization, and data integrity.

Coverage Target: 85%+
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any
import json

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    FailureMode,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    CurrentReading,
    DiagnosisConfidence,
    FailurePrediction,
    HealthStatus,
    MaintenanceRecommendation,
    MCSAResult,
    OilAnalysisReading,
    OilAnalysisResult,
    PredictiveMaintenanceInput,
    PredictiveMaintenanceOutput,
    TemperatureReading,
    ThermalImage,
    ThermographyResult,
    TrendDirection,
    VibrationAnalysisResult,
    VibrationReading,
    WeibullAnalysisResult,
    WorkOrderPriority,
    WorkOrderRequest,
    WorkOrderType,
)


class TestHealthStatusEnum:
    """Tests for HealthStatus enumeration."""

    def test_all_statuses_exist(self):
        """Verify all health statuses are defined."""
        expected = {"healthy", "degraded", "warning", "critical", "failed"}
        actual = {s.value for s in HealthStatus}
        assert actual == expected

    def test_status_ordering_by_severity(self):
        """Test statuses can be conceptually ordered by severity."""
        severity_order = [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.WARNING,
            HealthStatus.CRITICAL,
            HealthStatus.FAILED,
        ]
        # All unique
        assert len(set(severity_order)) == 5


class TestTrendDirectionEnum:
    """Tests for TrendDirection enumeration."""

    def test_all_directions_exist(self):
        """Verify all trend directions are defined."""
        expected = {"stable", "increasing", "decreasing", "erratic"}
        actual = {t.value for t in TrendDirection}
        assert actual == expected


class TestDiagnosisConfidenceEnum:
    """Tests for DiagnosisConfidence enumeration."""

    def test_all_confidence_levels_exist(self):
        """Verify all confidence levels are defined."""
        expected = {"high", "medium", "low", "uncertain"}
        actual = {c.value for c in DiagnosisConfidence}
        assert actual == expected


class TestVibrationReading:
    """Tests for VibrationReading model."""

    def test_valid_reading(self, vibration_reading_healthy):
        """Test valid vibration reading."""
        reading = vibration_reading_healthy

        assert reading.sensor_id == "ACCEL-001"
        assert reading.location == "DE"
        assert reading.velocity_rms_mm_s == 2.0
        assert reading.acceleration_rms_g == 0.5

    def test_velocity_non_negative(self):
        """Test velocity must be non-negative."""
        with pytest.raises(ValueError):
            VibrationReading(
                sensor_id="ACCEL-001",
                location="DE",
                velocity_rms_mm_s=-1.0,
                acceleration_rms_g=0.5,
                operating_speed_rpm=1800.0,
            )

    def test_speed_positive(self):
        """Test operating speed must be positive."""
        with pytest.raises(ValueError):
            VibrationReading(
                sensor_id="ACCEL-001",
                location="DE",
                velocity_rms_mm_s=2.0,
                acceleration_rms_g=0.5,
                operating_speed_rpm=0.0,
            )

    def test_optional_spectrum(self, vibration_reading_with_spectrum):
        """Test optional spectrum data."""
        reading = vibration_reading_with_spectrum

        assert reading.spectrum is not None
        assert len(reading.spectrum) > 0
        assert reading.frequency_resolution_hz == 1.0

    def test_timestamp_default(self):
        """Test timestamp defaults to current time."""
        reading = VibrationReading(
            sensor_id="ACCEL-001",
            location="DE",
            velocity_rms_mm_s=2.0,
            acceleration_rms_g=0.5,
            operating_speed_rpm=1800.0,
        )

        assert reading.timestamp is not None
        assert isinstance(reading.timestamp, datetime)


class TestOilAnalysisReading:
    """Tests for OilAnalysisReading model."""

    def test_valid_reading(self, oil_reading_healthy):
        """Test valid oil analysis reading."""
        reading = oil_reading_healthy

        assert reading.sample_id == "OIL-001"
        assert reading.viscosity_40c_cst == 46.0
        assert reading.tan_mg_koh_g == 0.5

    def test_viscosity_positive(self):
        """Test viscosity must be positive."""
        with pytest.raises(ValueError):
            OilAnalysisReading(
                sample_id="OIL-001",
                sample_point="Sump",
                viscosity_40c_cst=-10.0,
                tan_mg_koh_g=0.5,
            )

    def test_metals_non_negative(self):
        """Test metal contents must be non-negative."""
        # Valid with zero
        reading = OilAnalysisReading(
            sample_id="OIL-001",
            sample_point="Sump",
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=0.5,
            iron_ppm=0.0,
            copper_ppm=0.0,
        )
        assert reading.iron_ppm == 0.0

    def test_iso_4406_format(self, oil_reading_healthy):
        """Test ISO 4406 particle count format."""
        reading = oil_reading_healthy

        assert reading.particle_count_iso_4406 == "16/14/11"
        # Should be in X/Y/Z format
        parts = reading.particle_count_iso_4406.split("/")
        assert len(parts) == 3


class TestTemperatureReading:
    """Tests for TemperatureReading model."""

    def test_valid_reading(self, temperature_reading_healthy):
        """Test valid temperature reading."""
        reading = temperature_reading_healthy

        assert reading.sensor_id == "TEMP-001"
        assert reading.temperature_c == 55.0
        assert reading.ambient_c == 25.0

    def test_delta_calculation(self, temperature_reading_healthy):
        """Test delta temperature."""
        reading = temperature_reading_healthy

        assert reading.delta_c == 30.0  # 55 - 25


class TestThermalImage:
    """Tests for ThermalImage model."""

    def test_valid_image(self, thermal_image_healthy):
        """Test valid thermal image."""
        image = thermal_image_healthy

        assert image.image_id == "THERM-001"
        assert image.max_temperature_c == 55.0
        assert image.min_temperature_c == 35.0

    def test_hot_spots_list(self, thermal_image_warning):
        """Test hot spots list."""
        image = thermal_image_warning

        assert len(image.hot_spots) == 1
        assert image.hot_spots[0]["temperature_c"] == 85.0

    def test_emissivity_bounds(self):
        """Test emissivity bounds."""
        # Valid
        image = ThermalImage(
            image_id="THERM-001",
            camera_id="CAM-001",
            min_temperature_c=30.0,
            max_temperature_c=50.0,
            avg_temperature_c=40.0,
            emissivity=0.95,
        )
        assert image.emissivity == 0.95

        # Invalid - too low
        with pytest.raises(ValueError):
            ThermalImage(
                image_id="THERM-001",
                camera_id="CAM-001",
                min_temperature_c=30.0,
                max_temperature_c=50.0,
                avg_temperature_c=40.0,
                emissivity=0.05,
            )


class TestCurrentReading:
    """Tests for CurrentReading model."""

    def test_valid_reading(self, current_reading_healthy):
        """Test valid current reading."""
        reading = current_reading_healthy

        assert reading.sensor_id == "MCSA-001"
        assert reading.phase_a_rms_a == 100.0
        assert reading.line_frequency_hz == 60.0

    def test_currents_non_negative(self):
        """Test phase currents must be non-negative."""
        with pytest.raises(ValueError):
            CurrentReading(
                sensor_id="MCSA-001",
                phase_a_rms_a=-10.0,
                phase_b_rms_a=100.0,
                phase_c_rms_a=100.0,
            )


class TestWeibullAnalysisResult:
    """Tests for WeibullAnalysisResult model."""

    def test_valid_result(self):
        """Test valid Weibull analysis result."""
        result = WeibullAnalysisResult(
            beta=2.5,
            eta_hours=50000.0,
            gamma_hours=0.0,
            rul_p10_hours=15000.0,
            rul_p50_hours=25000.0,
            rul_p90_hours=40000.0,
            current_age_hours=25000.0,
            current_failure_probability=0.25,
            conditional_failure_probability_30d=0.05,
            failure_mode_interpretation="Wear-out pattern",
        )

        assert result.beta == 2.5
        assert result.eta_hours == 50000.0
        assert result.rul_p50_hours == 25000.0

    def test_beta_positive(self):
        """Test beta must be positive."""
        with pytest.raises(ValueError):
            WeibullAnalysisResult(
                beta=-1.0,
                eta_hours=50000.0,
                rul_p10_hours=15000.0,
                rul_p50_hours=25000.0,
                rul_p90_hours=40000.0,
                current_age_hours=25000.0,
                current_failure_probability=0.25,
                conditional_failure_probability_30d=0.05,
                failure_mode_interpretation="Invalid",
            )

    def test_probability_bounds(self):
        """Test probability must be in [0, 1]."""
        with pytest.raises(ValueError):
            WeibullAnalysisResult(
                beta=2.5,
                eta_hours=50000.0,
                rul_p10_hours=15000.0,
                rul_p50_hours=25000.0,
                rul_p90_hours=40000.0,
                current_age_hours=25000.0,
                current_failure_probability=1.5,  # Invalid
                conditional_failure_probability_30d=0.05,
                failure_mode_interpretation="Invalid",
            )


class TestVibrationAnalysisResult:
    """Tests for VibrationAnalysisResult model."""

    def test_valid_result(self):
        """Test valid vibration analysis result."""
        result = VibrationAnalysisResult(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            overall_velocity_mm_s=2.5,
            overall_acceleration_g=0.8,
            iso_zone=AlertSeverity.GOOD,
            dominant_frequency_hz=30.0,
            dominant_amplitude=1.5,
        )

        assert result.overall_velocity_mm_s == 2.5
        assert result.iso_zone == AlertSeverity.GOOD

    def test_fault_indicators(self):
        """Test fault indicator flags."""
        result = VibrationAnalysisResult(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            overall_velocity_mm_s=5.0,
            overall_acceleration_g=2.0,
            iso_zone=AlertSeverity.UNSATISFACTORY,
            dominant_frequency_hz=30.0,
            dominant_amplitude=3.0,
            bearing_defect_detected=True,
            bearing_defect_type="BPFO",
            imbalance_detected=False,
        )

        assert result.bearing_defect_detected is True
        assert result.bearing_defect_type == "BPFO"


class TestOilAnalysisResult:
    """Tests for OilAnalysisResult model."""

    def test_valid_result(self):
        """Test valid oil analysis result."""
        result = OilAnalysisResult(
            sample_id="OIL-001",
            timestamp=datetime.now(timezone.utc),
            oil_condition=HealthStatus.HEALTHY,
            viscosity_status="normal",
            viscosity_change_pct=2.0,
            tan_status=AlertSeverity.GOOD,
            water_status=AlertSeverity.GOOD,
            particle_status=AlertSeverity.ACCEPTABLE,
        )

        assert result.oil_condition == HealthStatus.HEALTHY
        assert result.viscosity_status == "normal"


class TestFailurePrediction:
    """Tests for FailurePrediction model."""

    def test_valid_prediction(self, failure_prediction_sample):
        """Test valid failure prediction."""
        pred = failure_prediction_sample

        assert pred.failure_mode == FailureMode.BEARING_WEAR
        assert pred.probability == 0.65
        assert pred.confidence == 0.85

    def test_probability_bounds(self):
        """Test probability must be in [0, 1]."""
        with pytest.raises(ValueError):
            FailurePrediction(
                failure_mode=FailureMode.BEARING_WEAR,
                probability=1.5,  # Invalid
                confidence=0.85,
                model_id="test",
                model_version="1.0.0",
            )

    def test_feature_importance(self, failure_prediction_sample):
        """Test feature importance dictionary."""
        pred = failure_prediction_sample

        assert "velocity_rms_normalized" in pred.feature_importance
        assert len(pred.top_contributing_features) == 3


class TestMaintenanceRecommendation:
    """Tests for MaintenanceRecommendation model."""

    def test_valid_recommendation(self, maintenance_recommendation_sample):
        """Test valid maintenance recommendation."""
        rec = maintenance_recommendation_sample

        assert rec.failure_mode == FailureMode.BEARING_WEAR
        assert rec.priority == WorkOrderPriority.HIGH
        assert rec.action_type == "inspection"

    def test_recommendation_id_generated(self):
        """Test recommendation ID is auto-generated."""
        rec = MaintenanceRecommendation(
            failure_mode=FailureMode.IMBALANCE,
            priority=WorkOrderPriority.MEDIUM,
            action_type="balance",
            description="Perform field balancing",
        )

        assert rec.recommendation_id is not None
        assert len(rec.recommendation_id) > 0


class TestWorkOrderRequest:
    """Tests for WorkOrderRequest model."""

    def test_valid_work_order(self):
        """Test valid work order request."""
        wo = WorkOrderRequest(
            equipment_id="PUMP-001",
            equipment_tag="P-1001A",
            order_type=WorkOrderType.CORRECTIVE,
            priority=WorkOrderPriority.HIGH,
            title="PdM: Bearing Wear - PUMP-001",
            description="Bearing wear detected. Schedule repair.",
            source_analysis_id="REQ-001",
        )

        assert wo.equipment_id == "PUMP-001"
        assert wo.order_type == WorkOrderType.CORRECTIVE

    def test_work_order_id_generated(self):
        """Test work order ID is auto-generated."""
        wo = WorkOrderRequest(
            equipment_id="PUMP-001",
            order_type=WorkOrderType.PREVENTIVE,
            priority=WorkOrderPriority.MEDIUM,
            title="Test Work Order",
            description="Test description",
            source_analysis_id="REQ-001",
        )

        assert wo.work_order_id is not None


class TestPredictiveMaintenanceInput:
    """Tests for PredictiveMaintenanceInput model."""

    def test_valid_input(self, predictive_maintenance_input_healthy):
        """Test valid predictive maintenance input."""
        input_data = predictive_maintenance_input_healthy

        assert input_data.equipment_id == "PUMP-001"
        assert len(input_data.vibration_readings) == 1
        assert input_data.oil_analysis is not None

    def test_request_id_generated(self):
        """Test request ID is auto-generated."""
        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
        )

        assert input_data.request_id is not None

    def test_empty_sensor_lists(self):
        """Test input with empty sensor lists."""
        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            vibration_readings=[],
            temperature_readings=[],
            thermal_images=[],
            current_readings=[],
        )

        assert len(input_data.vibration_readings) == 0

    def test_load_percent_bounds(self):
        """Test load percent bounds."""
        # Valid
        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            load_percent=100.0,
        )
        assert input_data.load_percent == 100.0

        # Above 100% allowed (overload)
        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            load_percent=120.0,
        )
        assert input_data.load_percent == 120.0


class TestPredictiveMaintenanceOutput:
    """Tests for PredictiveMaintenanceOutput model."""

    def test_valid_output(self):
        """Test valid predictive maintenance output."""
        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            health_status=HealthStatus.HEALTHY,
            health_score=92.5,
        )

        assert output.equipment_id == "PUMP-001"
        assert output.health_status == HealthStatus.HEALTHY
        assert output.health_score == 92.5

    def test_health_score_bounds(self):
        """Test health score must be in [0, 100]."""
        # Valid bounds
        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            health_status=HealthStatus.HEALTHY,
            health_score=0.0,
        )
        assert output.health_score == 0.0

        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            health_status=HealthStatus.HEALTHY,
            health_score=100.0,
        )
        assert output.health_score == 100.0

        # Invalid
        with pytest.raises(ValueError):
            PredictiveMaintenanceOutput(
                request_id="REQ-001",
                equipment_id="PUMP-001",
                health_status=HealthStatus.HEALTHY,
                health_score=105.0,
            )

    def test_default_lists(self):
        """Test default empty lists."""
        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            health_status=HealthStatus.HEALTHY,
            health_score=90.0,
        )

        assert output.vibration_analysis == []
        assert output.failure_predictions == []
        assert output.recommendations == []
        assert output.work_orders == []

    def test_json_serialization(self):
        """Test output can be serialized to JSON."""
        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            health_status=HealthStatus.HEALTHY,
            health_score=90.0,
            timestamp=datetime.now(timezone.utc),
        )

        json_str = output.json()
        assert "PUMP-001" in json_str
        assert "healthy" in json_str

        # Can be parsed back
        parsed = json.loads(json_str)
        assert parsed["equipment_id"] == "PUMP-001"


class TestSchemaIntegration:
    """Integration tests for schema models."""

    def test_full_output_with_all_results(self):
        """Test output with all result types populated."""
        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            timestamp=datetime.now(timezone.utc),
            status="success",
            processing_time_ms=250.5,
            health_status=HealthStatus.WARNING,
            health_score=55.0,
            health_trend=TrendDirection.DECREASING,
            vibration_analysis=[
                VibrationAnalysisResult(
                    sensor_id="ACCEL-001",
                    timestamp=datetime.now(timezone.utc),
                    overall_velocity_mm_s=6.5,
                    overall_acceleration_g=2.0,
                    iso_zone=AlertSeverity.UNSATISFACTORY,
                    dominant_frequency_hz=30.0,
                    dominant_amplitude=3.0,
                ),
            ],
            failure_predictions=[
                FailurePrediction(
                    failure_mode=FailureMode.BEARING_WEAR,
                    probability=0.45,
                    confidence=0.80,
                    model_id="test",
                    model_version="1.0.0",
                ),
            ],
            highest_risk_failure_mode=FailureMode.BEARING_WEAR,
            overall_failure_probability_30d=0.35,
            rul_hours=2500.0,
            active_alerts=[
                {"type": "VIBRATION", "severity": "warning"},
            ],
            alert_count_by_severity={"warning": 1, "critical": 0},
            kpis={"health_score": 55.0, "max_velocity_mm_s": 6.5},
            analysis_methods=["FFT", "Weibull"],
            data_quality_score=0.85,
        )

        assert output.health_status == HealthStatus.WARNING
        assert len(output.vibration_analysis) == 1
        assert len(output.failure_predictions) == 1
        assert output.highest_risk_failure_mode == FailureMode.BEARING_WEAR
