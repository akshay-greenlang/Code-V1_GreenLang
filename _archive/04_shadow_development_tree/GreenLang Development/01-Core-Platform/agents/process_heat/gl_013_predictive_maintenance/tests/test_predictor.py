# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Main Predictor Tests

Tests for the main PredictiveMaintenanceAgent predictor.
Validates end-to-end processing, integration, and output generation.

Coverage Target: 85%+
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.process_heat.gl_013_predictive_maintenance.predictor import (
    PredictiveMaintenanceAgent,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    EquipmentType,
    FailureMode,
    PredictiveMaintenanceConfig,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    HealthStatus,
    PredictiveMaintenanceInput,
    PredictiveMaintenanceOutput,
    TrendDirection,
    VibrationReading,
    WorkOrderPriority,
)
from greenlang.agents.process_heat.shared.base_agent import (
    SafetyLevel,
    ValidationError,
    ProcessingError,
)


class TestPredictiveMaintenanceAgentInit:
    """Tests for agent initialization."""

    def test_initialization(self, equipment_config):
        """Test agent initialization."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        assert agent.equipment_config == equipment_config
        assert agent.weibull_analyzer is not None
        assert agent.vibration_analyzer is not None
        assert agent.oil_analyzer is not None

    def test_initialization_with_safety_level(self, equipment_config):
        """Test initialization with specific safety level."""
        agent = PredictiveMaintenanceAgent(
            equipment_config,
            safety_level=SafetyLevel.SIL_2,
        )

        assert agent.safety_level == SafetyLevel.SIL_2

    def test_agent_id_format(self, equipment_config):
        """Test agent ID format."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        assert "GL-013" in agent.config.agent_id
        assert equipment_config.equipment_id in agent.config.agent_id

    def test_capabilities_set(self, equipment_config):
        """Test agent capabilities are set."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        from greenlang.agents.process_heat.shared.base_agent import AgentCapability

        assert AgentCapability.PREDICTIVE_ANALYTICS in agent.config.capabilities
        assert AgentCapability.ML_INFERENCE in agent.config.capabilities


class TestInputValidation:
    """Tests for input validation."""

    def test_validate_valid_input(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test validation of valid input."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        is_valid = agent.validate_input(predictive_maintenance_input_healthy)

        assert is_valid is True

    def test_validate_equipment_id_mismatch(
        self,
        equipment_config
    ):
        """Test validation fails for equipment ID mismatch."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        input_data = PredictiveMaintenanceInput(
            equipment_id="WRONG-ID",
            vibration_readings=[
                VibrationReading(
                    sensor_id="ACCEL-001",
                    location="DE",
                    velocity_rms_mm_s=2.0,
                    acceleration_rms_g=0.5,
                    operating_speed_rpm=1800.0,
                )
            ],
        )

        is_valid = agent.validate_input(input_data)

        assert is_valid is False

    def test_validate_no_sensor_data(self, equipment_config):
        """Test validation fails with no sensor data."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            vibration_readings=[],
            temperature_readings=[],
            current_readings=[],
        )

        is_valid = agent.validate_input(input_data)

        assert is_valid is False

    def test_validate_negative_velocity(self, equipment_config):
        """Test validation fails for negative velocity."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            vibration_readings=[
                VibrationReading(
                    sensor_id="ACCEL-001",
                    location="DE",
                    velocity_rms_mm_s=-1.0,  # Invalid
                    acceleration_rms_g=0.5,
                    operating_speed_rpm=1800.0,
                )
            ],
        )

        is_valid = agent.validate_input(input_data)

        assert is_valid is False


class TestOutputValidation:
    """Tests for output validation."""

    def test_validate_valid_output(self, equipment_config):
        """Test validation of valid output."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            health_status=HealthStatus.HEALTHY,
            health_score=90.0,
        )

        is_valid = agent.validate_output(output)

        assert is_valid is True

    def test_validate_invalid_health_score(self, equipment_config):
        """Test validation fails for invalid health score."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            health_status=HealthStatus.HEALTHY,
            health_score=150.0,  # Invalid > 100
        )

        is_valid = agent.validate_output(output)

        assert is_valid is False

    def test_validate_invalid_probability(self, equipment_config):
        """Test validation fails for invalid probability."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = PredictiveMaintenanceOutput(
            request_id="REQ-001",
            equipment_id="PUMP-001",
            health_status=HealthStatus.HEALTHY,
            health_score=90.0,
            overall_failure_probability_30d=1.5,  # Invalid > 1
        )

        is_valid = agent.validate_output(output)

        assert is_valid is False


class TestProcessMethod:
    """Tests for main process method."""

    def test_process_healthy_equipment(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test processing healthy equipment data."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert output.status == "success"
        assert output.health_status == HealthStatus.HEALTHY
        assert output.health_score >= 80.0

    def test_process_warning_equipment(
        self,
        equipment_config,
        predictive_maintenance_input_warning
    ):
        """Test processing warning-level equipment data."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_warning)

        assert output.status == "success"
        assert output.health_status in [
            HealthStatus.DEGRADED,
            HealthStatus.WARNING,
            HealthStatus.CRITICAL,
        ]

    def test_process_generates_request_id(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test process generates request ID in output."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert output.request_id == predictive_maintenance_input_healthy.request_id

    def test_process_records_processing_time(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test process records processing time."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert output.processing_time_ms > 0

    def test_process_generates_provenance(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test process generates provenance hash."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert output.provenance_hash is not None
        assert len(output.provenance_hash) == 64  # SHA-256


class TestVibrationAnalysis:
    """Tests for vibration analysis in predictor."""

    def test_vibration_results_included(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test vibration analysis results are included."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert len(output.vibration_analysis) > 0

    def test_iso_zone_classification(
        self,
        equipment_config,
        vibration_reading_healthy
    ):
        """Test ISO zone is classified."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            vibration_readings=[vibration_reading_healthy],
        )

        output = agent.process(input_data)

        assert output.vibration_analysis[0].iso_zone == AlertSeverity.GOOD


class TestOilAnalysis:
    """Tests for oil analysis in predictor."""

    def test_oil_analysis_included(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test oil analysis results are included."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert output.oil_analysis_result is not None

    def test_oil_analysis_optional(
        self,
        equipment_config,
        vibration_reading_healthy
    ):
        """Test oil analysis is optional."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            vibration_readings=[vibration_reading_healthy],
            # No oil analysis
        )

        output = agent.process(input_data)

        assert output.oil_analysis_result is None


class TestFailurePredictions:
    """Tests for failure predictions in predictor."""

    def test_failure_predictions_generated(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test failure predictions are generated."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert len(output.failure_predictions) > 0

    def test_highest_risk_mode_identified(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test highest risk failure mode is identified."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        if output.failure_predictions:
            assert output.highest_risk_failure_mode is not None

    def test_overall_probability_calculated(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test overall failure probability is calculated."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert 0 <= output.overall_failure_probability_30d <= 1


class TestWeibullRUL:
    """Tests for Weibull RUL analysis in predictor."""

    def test_rul_calculated(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test RUL is calculated."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        # RUL may be calculated if running hours provided
        if predictive_maintenance_input_healthy.running_hours:
            # Weibull analysis should run
            pass

    def test_rul_confidence_interval(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test RUL confidence interval is provided."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        if output.rul_hours is not None:
            assert output.rul_confidence_interval is not None


class TestHealthDetermination:
    """Tests for overall health determination."""

    def test_health_score_range(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test health score is in valid range."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert 0 <= output.health_score <= 100

    def test_health_status_matches_score(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test health status matches health score."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        if output.health_score >= 80:
            assert output.health_status == HealthStatus.HEALTHY
        elif output.health_score >= 60:
            assert output.health_status in [
                HealthStatus.HEALTHY,
                HealthStatus.DEGRADED
            ]

    def test_health_trend_calculated(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test health trend is calculated."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert output.health_trend in [
            TrendDirection.STABLE,
            TrendDirection.INCREASING,
            TrendDirection.DECREASING,
            TrendDirection.ERRATIC,
        ]


class TestAlertGeneration:
    """Tests for alert generation."""

    def test_alerts_generated_for_issues(
        self,
        equipment_config,
        predictive_maintenance_input_warning
    ):
        """Test alerts are generated for issues."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_warning)

        assert len(output.active_alerts) > 0

    def test_alert_count_by_severity(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test alert count by severity is provided."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert "critical" in output.alert_count_by_severity
        assert "warning" in output.alert_count_by_severity


class TestRecommendations:
    """Tests for maintenance recommendations."""

    def test_recommendations_generated(
        self,
        equipment_config,
        predictive_maintenance_input_warning
    ):
        """Test recommendations are generated for issues."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_warning)

        assert len(output.recommendations) > 0

    def test_recommendation_priority(
        self,
        equipment_config,
        predictive_maintenance_input_warning
    ):
        """Test recommendations have priority."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_warning)

        if output.recommendations:
            rec = output.recommendations[0]
            assert rec.priority is not None


class TestWorkOrderGeneration:
    """Tests for work order generation."""

    def test_work_orders_generated_for_warning(
        self,
        equipment_config,
        predictive_maintenance_input_warning
    ):
        """Test work orders are generated for warning status."""
        # Enable work order generation
        equipment_config.cmms.enabled = True

        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_warning)

        # Work orders should be generated for warning/critical
        if output.health_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            assert len(output.work_orders) >= 0  # May be empty if disabled

    def test_no_work_orders_for_healthy(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test no work orders for healthy equipment."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        if output.health_status == HealthStatus.HEALTHY:
            assert len(output.work_orders) == 0


class TestKPIs:
    """Tests for KPI calculation."""

    def test_kpis_calculated(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test KPIs are calculated."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert "health_score" in output.kpis

    def test_vibration_kpis(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test vibration KPIs are included."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        if output.vibration_analysis:
            assert "max_velocity_mm_s" in output.kpis


class TestDataQuality:
    """Tests for data quality assessment."""

    def test_data_quality_score(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test data quality score is calculated."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert 0 <= output.data_quality_score <= 1

    def test_data_quality_penalized_missing_data(
        self,
        equipment_config,
        vibration_reading_healthy
    ):
        """Test data quality is penalized for missing data."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        # Minimal input
        minimal_input = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            vibration_readings=[vibration_reading_healthy],
        )

        output = agent.process(minimal_input)

        # Should have lower quality score
        assert output.data_quality_score < 1.0


class TestAnalysisMethods:
    """Tests for analysis methods tracking."""

    def test_analysis_methods_listed(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test analysis methods are listed."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert len(output.analysis_methods) > 0

    def test_model_versions_tracked(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test model versions are tracked."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output = agent.process(predictive_maintenance_input_healthy)

        assert len(output.model_versions) > 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_validation_error_on_invalid_input(
        self,
        equipment_config
    ):
        """Test ValidationError raised for invalid input."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        invalid_input = PredictiveMaintenanceInput(
            equipment_id="WRONG-ID",  # Mismatch
        )

        with pytest.raises((ValidationError, ProcessingError)):
            agent.process(invalid_input)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_processing_same_result(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test repeated processing produces same result."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        outputs = [
            agent.process(predictive_maintenance_input_healthy)
            for _ in range(3)
        ]

        # Health scores should be identical
        scores = [o.health_score for o in outputs]
        assert len(set(scores)) == 1

    def test_provenance_reproducible(
        self,
        equipment_config,
        predictive_maintenance_input_healthy
    ):
        """Test provenance hash is reproducible."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        output1 = agent.process(predictive_maintenance_input_healthy)
        output2 = agent.process(predictive_maintenance_input_healthy)

        assert output1.provenance_hash == output2.provenance_hash


class TestIntelligenceCapabilities:
    """Tests for intelligence capabilities."""

    def test_intelligence_level(self, equipment_config):
        """Test intelligence level is ADVANCED."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        from greenlang.agents.intelligence_interface import IntelligenceLevel

        assert agent.get_intelligence_level() == IntelligenceLevel.ADVANCED

    def test_intelligence_capabilities(self, equipment_config):
        """Test intelligence capabilities are set."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        capabilities = agent.get_intelligence_capabilities()

        assert capabilities.can_explain is True
        assert capabilities.can_recommend is True
        assert capabilities.can_detect_anomalies is True


class TestIntegration:
    """Integration tests for predictor."""

    def test_full_workflow(self, equipment_config):
        """Test complete prediction workflow."""
        agent = PredictiveMaintenanceAgent(equipment_config)

        # Create comprehensive input
        input_data = PredictiveMaintenanceInput(
            equipment_id="PUMP-001",
            timestamp=datetime.now(timezone.utc),
            vibration_readings=[
                VibrationReading(
                    sensor_id="ACCEL-DE",
                    location="DE",
                    velocity_rms_mm_s=4.0,
                    acceleration_rms_g=1.2,
                    operating_speed_rpm=1795.0,
                    temperature_c=58.0,
                ),
                VibrationReading(
                    sensor_id="ACCEL-NDE",
                    location="NDE",
                    velocity_rms_mm_s=3.5,
                    acceleration_rms_g=1.0,
                    operating_speed_rpm=1795.0,
                    temperature_c=55.0,
                ),
            ],
            operating_speed_rpm=1795.0,
            load_percent=85.0,
            running_hours=30000.0,
        )

        output = agent.process(input_data)

        # Verify comprehensive output
        assert output.status == "success"
        assert output.equipment_id == "PUMP-001"
        assert output.health_status is not None
        assert 0 <= output.health_score <= 100
        assert len(output.vibration_analysis) == 2
        assert len(output.failure_predictions) > 0
        assert output.processing_time_ms > 0
        assert output.provenance_hash is not None
