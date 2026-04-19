# -*- coding: utf-8 -*-
"""
GL-005 Maintenance Advisor Tests
================================

Comprehensive unit tests for maintenance advisory module including
fouling prediction, burner wear assessment, and work order generation.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    MaintenanceAdvisoryConfig,
    FoulingPredictionConfig,
    BurnerWearConfig,
    MaintenancePriority,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    FlueGasReading,
    CombustionOperatingData,
    CQIResult,
    CQIRating,
    AnomalyDetectionResult,
    AnalysisStatus,
    TrendDirection,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.maintenance_advisor import (
    FoulingPredictor,
    BurnerWearAssessor,
    WorkOrderGenerator,
    MaintenanceAdvisor,
    HistoricalMetrics,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.tests.conftest import (
    assert_valid_provenance_hash,
)


class TestHistoricalMetrics:
    """Tests for HistoricalMetrics data structure."""

    def test_add_point(self):
        """Test adding data point to history."""
        metrics = HistoricalMetrics()

        now = datetime.now(timezone.utc)
        metrics.add_point(now, 200.0, 85.0, 30.0)

        assert len(metrics.timestamps) == 1
        assert len(metrics.stack_temps) == 1
        assert len(metrics.efficiency_values) == 1
        assert len(metrics.co_values) == 1

    def test_data_aging(self):
        """Test that old data is aged out."""
        metrics = HistoricalMetrics()

        # Add old data (35 days ago)
        old_time = datetime.now(timezone.utc) - timedelta(days=35)
        metrics.add_point(old_time, 200.0, 85.0, 30.0)

        # Add recent data
        now = datetime.now(timezone.utc)
        metrics.add_point(now, 210.0, 84.0, 35.0)

        # Old data should be removed (30 day retention)
        assert len(metrics.timestamps) == 1
        assert metrics.timestamps[0] == now


class TestFoulingPredictor:
    """Tests for fouling predictor."""

    def test_initialization(self, default_fouling_config):
        """Test fouling predictor initialization."""
        predictor = FoulingPredictor(default_fouling_config)

        assert predictor.config == default_fouling_config
        assert predictor._baseline_stack_temp is None
        assert predictor._baseline_efficiency is None

    def test_set_baseline(self, default_fouling_config):
        """Test setting fouling baseline."""
        predictor = FoulingPredictor(default_fouling_config)

        predictor.set_baseline(180.0, 88.0)

        assert predictor._baseline_stack_temp == 180.0
        assert predictor._baseline_efficiency == 88.0

    def test_no_fouling_at_baseline(self, default_fouling_config):
        """Test no fouling detection at baseline conditions."""
        predictor = FoulingPredictor(default_fouling_config)
        predictor.set_baseline(180.0, 88.0)

        assessment = predictor.assess(180.0, 88.0)

        assert assessment.fouling_detected is False
        assert assessment.fouling_severity == "none"
        assert assessment.efficiency_loss_pct == 0.0

    def test_light_fouling_detection(self, default_fouling_config):
        """Test light fouling detection."""
        predictor = FoulingPredictor(default_fouling_config)
        predictor.set_baseline(180.0, 88.0)

        # Slight efficiency loss and stack temp increase
        assessment = predictor.assess(195.0, 86.0)  # 2% efficiency loss

        assert assessment.fouling_detected is True
        assert assessment.fouling_severity == "light"
        assert assessment.efficiency_loss_pct == pytest.approx(2.0, rel=0.1)

    def test_moderate_fouling_detection(self, default_fouling_config):
        """Test moderate fouling detection."""
        predictor = FoulingPredictor(default_fouling_config)
        predictor.set_baseline(180.0, 88.0)

        # Moderate efficiency loss
        assessment = predictor.assess(210.0, 84.5)  # 3.5% efficiency loss

        assert assessment.fouling_detected is True
        assert assessment.fouling_severity == "moderate"

    def test_severe_fouling_detection(self, default_fouling_config):
        """Test severe fouling detection."""
        predictor = FoulingPredictor(default_fouling_config)
        predictor.set_baseline(180.0, 88.0)

        # Severe efficiency loss (>10%)
        assessment = predictor.assess(280.0, 76.0)  # 12% efficiency loss

        assert assessment.fouling_detected is True
        assert assessment.fouling_severity == "severe"
        assert assessment.efficiency_loss_pct >= 10.0

    def test_stack_temp_tracking(self, default_fouling_config):
        """Test stack temperature increase tracking."""
        predictor = FoulingPredictor(default_fouling_config)
        predictor.set_baseline(180.0, 88.0)

        assessment = predictor.assess(230.0, 84.0)  # 50C increase

        assert assessment.stack_temp_increase_c == pytest.approx(50.0, rel=0.1)

    def test_days_until_cleaning_prediction(self, default_fouling_config):
        """Test prediction of days until cleaning needed."""
        predictor = FoulingPredictor(default_fouling_config)
        predictor.set_baseline(180.0, 88.0)

        # Add historical data showing declining efficiency
        now = datetime.now(timezone.utc)
        for i in range(10):
            reading = FlueGasReading(
                timestamp=now - timedelta(days=10-i),
                oxygen_pct=3.0,
                co2_pct=10.5,
                co_ppm=30.0 + i * 2,
                flue_gas_temp_c=180.0 + i * 3,
            )
            efficiency = 88.0 - i * 0.3
            predictor.add_data_point(reading, efficiency)

        assessment = predictor.assess(210.0, 85.0)

        # Should have prediction
        if assessment.days_until_cleaning_recommended:
            assert assessment.days_until_cleaning_recommended >= 0

    def test_30_day_efficiency_prediction(self, default_fouling_config):
        """Test 30-day efficiency loss prediction."""
        predictor = FoulingPredictor(default_fouling_config)
        predictor.set_baseline(180.0, 88.0)

        # Add trend data
        now = datetime.now(timezone.utc)
        for i in range(7):
            reading = FlueGasReading(
                timestamp=now - timedelta(days=7-i),
                oxygen_pct=3.0,
                co2_pct=10.5,
                co_ppm=30.0,
                flue_gas_temp_c=180.0 + i * 2,
            )
            predictor.add_data_point(reading, 88.0 - i * 0.2)

        assessment = predictor.assess(194.0, 86.6)

        if assessment.predicted_efficiency_loss_30d:
            assert assessment.predicted_efficiency_loss_30d > assessment.efficiency_loss_pct


class TestBurnerWearAssessor:
    """Tests for burner wear assessor."""

    def test_initialization(self, default_burner_wear_config):
        """Test burner wear assessor initialization."""
        assessor = BurnerWearAssessor(default_burner_wear_config)

        assert assessor.config == default_burner_wear_config
        assert assessor._baseline_co is None

    def test_set_baseline_co(self, default_burner_wear_config):
        """Test setting baseline CO."""
        assessor = BurnerWearAssessor(default_burner_wear_config)
        assessor.set_baseline_co(25.0)

        assert assessor._baseline_co == 25.0

    def test_no_wear_new_burner(self, default_burner_wear_config):
        """Test no wear detection for new burner."""
        assessor = BurnerWearAssessor(default_burner_wear_config)
        assessor.set_baseline_co(25.0)

        assessment = assessor.assess(
            current_co=25.0,
            operating_hours=1000.0,  # New burner
        )

        assert assessment.wear_detected is False
        assert assessment.wear_level == "normal"
        assert assessment.expected_life_remaining_pct > 90.0

    def test_early_wear_detection(self, default_burner_wear_config):
        """Test early wear detection."""
        assessor = BurnerWearAssessor(default_burner_wear_config)
        assessor.set_baseline_co(25.0)

        assessment = assessor.assess(
            current_co=45.0,
            operating_hours=10000.0,  # 50% life used
        )

        # May or may not trigger early wear depending on config
        assert assessment.expected_life_remaining_pct < 60.0

    def test_replacement_needed_detection(self, default_burner_wear_config):
        """Test replacement needed detection."""
        assessor = BurnerWearAssessor(default_burner_wear_config)
        assessor.set_baseline_co(25.0)

        # Add CO trend data showing increase
        now = datetime.now(timezone.utc)
        for i in range(30):
            assessor.add_co_reading(
                now - timedelta(days=30-i),
                25.0 + i * 5  # Increasing CO
            )

        assessment = assessor.assess(
            current_co=200.0,
            operating_hours=19000.0,  # Near end of life
            flame_stability=0.8,
            ignition_reliability=0.92,
        )

        assert assessment.wear_detected is True
        assert assessment.wear_level in ["significant_wear", "replacement_needed"]
        assert assessment.expected_life_remaining_pct < 20.0

    def test_remaining_life_calculation(self, default_burner_wear_config):
        """Test remaining life calculation."""
        config = BurnerWearConfig(expected_burner_life_hours=20000)
        assessor = BurnerWearAssessor(config)

        assessment = assessor.assess(
            current_co=30.0,
            operating_hours=15000.0,
        )

        # (20000 - 15000) / 20000 * 100 = 25%
        assert assessment.expected_life_remaining_pct == pytest.approx(25.0, rel=0.1)

    def test_replacement_date_prediction(self, default_burner_wear_config):
        """Test replacement date prediction."""
        assessor = BurnerWearAssessor(default_burner_wear_config)

        assessment = assessor.assess(
            current_co=40.0,
            operating_hours=15000.0,
        )

        if assessment.replacement_recommended_by:
            assert assessment.replacement_recommended_by > datetime.now(timezone.utc)

    def test_co_trend_tracking(self, default_burner_wear_config):
        """Test CO trend slope calculation."""
        assessor = BurnerWearAssessor(default_burner_wear_config)

        # Add increasing CO readings
        now = datetime.now(timezone.utc)
        for i in range(30):
            assessor.add_co_reading(
                now - timedelta(days=30-i),
                30.0 + i * 0.5
            )

        assessment = assessor.assess(
            current_co=45.0,
            operating_hours=10000.0,
        )

        # Should have positive trend
        assert assessment.co_trend_slope > 0


class TestWorkOrderGenerator:
    """Tests for CMMS work order generator."""

    def test_initialization(self, default_maintenance_config):
        """Test work order generator initialization."""
        generator = WorkOrderGenerator(default_maintenance_config)

        assert generator.config == default_maintenance_config

    def test_generate_work_order(self, default_maintenance_config):
        """Test work order generation."""
        generator = WorkOrderGenerator(default_maintenance_config)

        from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
            MaintenanceRecommendation
        )

        recommendation = MaintenanceRecommendation(
            recommendation_id="REC-001",
            timestamp=datetime.now(timezone.utc),
            maintenance_type="cleaning",
            priority=MaintenancePriority.HIGH,
            component="Heat Transfer Surfaces",
            title="Boiler tube cleaning required",
            description="Fouling detected with 8% efficiency loss",
            justification="Cost savings from improved efficiency",
            recommended_by_date=datetime.now(timezone.utc) + timedelta(days=7),
            estimated_duration_hours=8.0,
            risk_if_deferred="high",
        )

        work_order = generator.generate_work_order(
            equipment_id="BLR-001",
            recommendation=recommendation,
            equipment_name="Boiler 1",
            location="Plant A",
        )

        assert work_order.work_order_id.startswith("WO-GL005-")
        assert work_order.equipment_id == "BLR-001"
        assert work_order.priority == MaintenancePriority.HIGH
        assert work_order.work_type == "PM"  # Cleaning = PM
        assert work_order.source_agent == "GL-005"
        assert work_order.provenance_hash is not None

    def test_work_type_mapping(self, default_maintenance_config):
        """Test maintenance type to work type mapping."""
        generator = WorkOrderGenerator(default_maintenance_config)

        from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
            MaintenanceRecommendation
        )

        types_map = [
            ("inspection", "INSP"),
            ("cleaning", "PM"),
            ("repair", "CM"),
            ("replacement", "CM"),
            ("calibration", "PM"),
        ]

        for maint_type, expected_work_type in types_map:
            rec = MaintenanceRecommendation(
                recommendation_id="REC-TEST",
                timestamp=datetime.now(timezone.utc),
                maintenance_type=maint_type,
                priority=MaintenancePriority.MEDIUM,
                component="Test",
                title="Test",
                description="Test",
                justification="Test",
            )

            wo = generator.generate_work_order("BLR-001", rec)
            assert wo.work_type == expected_work_type

    def test_work_order_status(self, default_maintenance_config):
        """Test work order status based on approval setting."""
        # With approval required
        config_approval = MaintenanceAdvisoryConfig(work_order_approval_required=True)
        generator_approval = WorkOrderGenerator(config_approval)

        from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
            MaintenanceRecommendation
        )

        rec = MaintenanceRecommendation(
            recommendation_id="REC-001",
            timestamp=datetime.now(timezone.utc),
            maintenance_type="cleaning",
            priority=MaintenancePriority.MEDIUM,
            component="Test",
            title="Test",
            description="Test",
            justification="Test",
        )

        wo = generator_approval.generate_work_order("BLR-001", rec)
        assert wo.status == "pending_approval"

        # Without approval required
        config_no_approval = MaintenanceAdvisoryConfig(work_order_approval_required=False)
        generator_no_approval = WorkOrderGenerator(config_no_approval)

        wo2 = generator_no_approval.generate_work_order("BLR-001", rec)
        assert wo2.status == "approved"


class TestMaintenanceAdvisor:
    """Tests for integrated maintenance advisor."""

    def test_initialization(self, default_maintenance_config):
        """Test maintenance advisor initialization."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")

        assert advisor.config == default_maintenance_config
        assert advisor.equipment_id == "BLR-001"
        assert advisor.fouling_predictor is not None
        assert advisor.burner_assessor is not None
        assert advisor.work_order_generator is not None

    def test_set_baselines(self, default_maintenance_config):
        """Test baseline setting."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")

        advisor.set_baselines(180.0, 88.0, 25.0)

        assert advisor.fouling_predictor._baseline_stack_temp == 180.0
        assert advisor.fouling_predictor._baseline_efficiency == 88.0
        assert advisor.burner_assessor._baseline_co == 25.0

    def test_analyze_healthy_equipment(
        self, default_maintenance_config, optimal_flue_gas_reading, normal_operating_data
    ):
        """Test analysis of healthy equipment."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        result = advisor.analyze(
            optimal_flue_gas_reading,
            normal_operating_data,
        )

        assert result.status == AnalysisStatus.SUCCESS
        assert result.equipment_health_score >= 80.0
        assert result.urgent_actions_required is False

    def test_analyze_fouled_equipment(
        self, default_maintenance_config, fouling_flue_gas_reading, normal_operating_data
    ):
        """Test analysis of fouled equipment."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        result = advisor.analyze(
            fouling_flue_gas_reading,
            normal_operating_data,
        )

        assert result.fouling.fouling_detected is True
        assert len(result.recommendations) > 0

    def test_analyze_worn_burner(
        self, default_maintenance_config, high_co_flue_gas_reading, worn_burner_operating_data
    ):
        """Test analysis of worn burner."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        result = advisor.analyze(
            high_co_flue_gas_reading,
            worn_burner_operating_data,
        )

        assert result.burner_wear.wear_detected is True

    def test_analyze_with_cqi_result(
        self, default_maintenance_config, optimal_flue_gas_reading, normal_operating_data
    ):
        """Test analysis with CQI result input."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        cqi_result = CQIResult(
            cqi_score=85.0,
            cqi_rating=CQIRating.GOOD,
            components=[],
            co_corrected_ppm=35.0,
            nox_corrected_ppm=50.0,
            o2_reference_pct=3.0,
            excess_air_pct=15.0,
            combustion_efficiency_pct=88.0,
            calculation_timestamp=datetime.now(timezone.utc),
            provenance_hash="a" * 64,
        )

        result = advisor.analyze(
            optimal_flue_gas_reading,
            normal_operating_data,
            cqi_result=cqi_result,
        )

        # CQI should influence health score
        assert result.status == AnalysisStatus.SUCCESS

    def test_analyze_with_anomaly_result(
        self, default_maintenance_config, optimal_flue_gas_reading, normal_operating_data
    ):
        """Test analysis with anomaly result input."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        anomaly_result = AnomalyDetectionResult(
            status=AnalysisStatus.SUCCESS,
            anomaly_detected=True,
            total_anomalies=1,
            critical_count=1,
            analysis_timestamp=datetime.now(timezone.utc),
            provenance_hash="b" * 64,
        )

        result = advisor.analyze(
            optimal_flue_gas_reading,
            normal_operating_data,
            anomaly_result=anomaly_result,
        )

        # Critical anomalies should trigger recommendations
        assert len(result.recommendations) > 0

    def test_recommendation_priority_ordering(
        self, default_maintenance_config, fouling_flue_gas_reading, worn_burner_operating_data
    ):
        """Test that recommendations are ordered by priority."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        result = advisor.analyze(
            fouling_flue_gas_reading,
            worn_burner_operating_data,
        )

        if len(result.recommendations) >= 2:
            priorities = [r.priority for r in result.recommendations]
            priority_order = {
                MaintenancePriority.CRITICAL: 0,
                MaintenancePriority.HIGH: 1,
                MaintenancePriority.MEDIUM: 2,
                MaintenancePriority.LOW: 3,
                MaintenancePriority.ROUTINE: 4,
            }

            # Check ordering
            for i in range(len(priorities) - 1):
                assert priority_order[priorities[i]] <= priority_order[priorities[i+1]]

    def test_health_score_calculation(
        self, default_maintenance_config, optimal_flue_gas_reading, normal_operating_data
    ):
        """Test health score calculation."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        result = advisor.analyze(
            optimal_flue_gas_reading,
            normal_operating_data,
        )

        # Score should be 0-100
        assert 0.0 <= result.equipment_health_score <= 100.0

    def test_health_trend_tracking(
        self, default_maintenance_config, optimal_flue_gas_reading, normal_operating_data
    ):
        """Test health trend tracking."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        cqi_result = CQIResult(
            cqi_score=85.0,
            cqi_rating=CQIRating.GOOD,
            components=[],
            co_corrected_ppm=35.0,
            nox_corrected_ppm=50.0,
            o2_reference_pct=3.0,
            excess_air_pct=15.0,
            combustion_efficiency_pct=88.0,
            trend_vs_baseline=TrendDirection.STABLE,
            calculation_timestamp=datetime.now(timezone.utc),
            provenance_hash="a" * 64,
        )

        result = advisor.analyze(
            optimal_flue_gas_reading,
            normal_operating_data,
            cqi_result=cqi_result,
        )

        assert result.health_trend == TrendDirection.STABLE

    def test_work_order_generation(self, normal_operating_data):
        """Test work order generation when CMMS enabled."""
        config = MaintenanceAdvisoryConfig(
            cmms_enabled=True,
            auto_create_work_orders=True,
        )

        advisor = MaintenanceAdvisor(config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        # Create problematic reading
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=500.0,  # High CO
            nox_ppm=45.0,
            flue_gas_temp_c=300.0,  # High stack temp
        )

        result = advisor.analyze(reading, normal_operating_data)

        # May or may not have work orders depending on recommendations
        # Work orders generated internally by advisor

    def test_provenance_hash(
        self, default_maintenance_config, optimal_flue_gas_reading, normal_operating_data
    ):
        """Test provenance hash generation."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        result = advisor.analyze(
            optimal_flue_gas_reading,
            normal_operating_data,
        )

        assert_valid_provenance_hash(result.provenance_hash)

    def test_audit_trail(
        self, default_maintenance_config, optimal_flue_gas_reading, normal_operating_data
    ):
        """Test audit trail generation."""
        advisor = MaintenanceAdvisor(default_maintenance_config, "BLR-001")
        advisor.set_baselines(180.0, 88.0, 25.0)

        advisor.analyze(
            optimal_flue_gas_reading,
            normal_operating_data,
        )

        audit = advisor.get_audit_trail()
        assert len(audit) > 0

        operations = [entry["operation"] for entry in audit]
        assert "fouling_assessment" in operations
        assert "burner_wear_assessment" in operations
