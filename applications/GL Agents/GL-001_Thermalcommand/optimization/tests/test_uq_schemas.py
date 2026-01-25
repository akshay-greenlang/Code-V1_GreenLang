"""
Tests for UQ Schemas

Tests Pydantic models for uncertainty quantification including:
    - PredictionInterval validation and properties
    - Scenario and ScenarioSet validation
    - CalibrationMetrics computation
    - ProvenanceRecord SHA-256 hashing
    - UncertaintyBand alignment validation

Author: GreenLang Process Heat Team
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from ..uq_schemas import (
    ProvenanceRecord,
    PredictionInterval,
    QuantileSet,
    QuantileValue,
    Scenario,
    ScenarioSet,
    ScenarioVariable,
    CalibrationMetrics,
    CalibrationStatus,
    UncertaintySource,
    UncertaintySourceType,
    DistributionType,
    RobustSolution,
    RobustConstraint,
    ConstraintType,
    OptimizationObjective,
    UncertaintyBand,
    FanChartData,
    RiskAssessment,
    ReliabilityDiagram,
    ReliabilityDiagramPoint,
    ScenarioComparison,
)


class TestProvenanceRecord:
    """Tests for ProvenanceRecord SHA-256 provenance tracking."""

    def test_compute_hash_deterministic(self):
        """Hash computation should be deterministic."""
        data = {"a": 1, "b": "test", "c": [1, 2, 3]}

        hash1 = ProvenanceRecord.compute_hash(data)
        hash2 = ProvenanceRecord.compute_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_compute_hash_key_order_invariant(self):
        """Hash should be same regardless of key order."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}

        assert ProvenanceRecord.compute_hash(data1) == ProvenanceRecord.compute_hash(data2)

    def test_create_provenance_record(self):
        """Create provenance record with hashes."""
        inputs = {"x": 10, "y": 20}
        outputs = {"result": 30}

        record = ProvenanceRecord.create(
            calculation_type="test_calculation",
            inputs=inputs,
            outputs=outputs,
            computation_time_ms=5.0
        )

        assert record.calculation_type == "test_calculation"
        assert record.input_hash is not None
        assert record.output_hash is not None
        assert record.combined_hash is not None
        assert record.computation_time_ms == 5.0
        assert record.inputs == inputs
        assert record.outputs == outputs

    def test_provenance_hash_changes_with_inputs(self):
        """Different inputs should produce different hashes."""
        record1 = ProvenanceRecord.create(
            calculation_type="test",
            inputs={"x": 1},
            outputs={"y": 2}
        )
        record2 = ProvenanceRecord.create(
            calculation_type="test",
            inputs={"x": 2},
            outputs={"y": 2}
        )

        assert record1.input_hash != record2.input_hash
        assert record1.combined_hash != record2.combined_hash


class TestPredictionInterval:
    """Tests for PredictionInterval schema."""

    def test_valid_prediction_interval(self):
        """Valid prediction interval should be created."""
        interval = PredictionInterval(
            point_estimate=Decimal("100.0"),
            lower_bound=Decimal("90.0"),
            upper_bound=Decimal("110.0"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC",
            horizon_minutes=60
        )

        assert interval.point_estimate == Decimal("100.0")
        assert interval.lower_bound == Decimal("90.0")
        assert interval.upper_bound == Decimal("110.0")
        assert interval.confidence_level == Decimal("0.90")

    def test_prediction_interval_width(self):
        """Interval width should be computed correctly."""
        interval = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("80"),
            upper_bound=Decimal("120"),
            confidence_level=Decimal("0.90"),
            variable_name="test",
            unit="units"
        )

        assert interval.interval_width == Decimal("40")

    def test_prediction_interval_relative_width(self):
        """Relative width should be computed correctly."""
        interval = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("80"),
            upper_bound=Decimal("120"),
            confidence_level=Decimal("0.90"),
            variable_name="test",
            unit="units"
        )

        assert interval.relative_width == Decimal("0.4")

    def test_prediction_interval_contains(self):
        """Contains method should work correctly."""
        interval = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("80"),
            upper_bound=Decimal("120"),
            confidence_level=Decimal("0.90"),
            variable_name="test",
            unit="units"
        )

        assert interval.contains(Decimal("100")) is True
        assert interval.contains(Decimal("80")) is True
        assert interval.contains(Decimal("120")) is True
        assert interval.contains(Decimal("79")) is False
        assert interval.contains(Decimal("121")) is False

    def test_prediction_interval_invalid_bounds(self):
        """Invalid bounds should raise validation error."""
        with pytest.raises(ValueError):
            PredictionInterval(
                point_estimate=Decimal("100"),
                lower_bound=Decimal("110"),  # Invalid: lower > point
                upper_bound=Decimal("120"),
                confidence_level=Decimal("0.90"),
                variable_name="test",
                unit="units"
            )

    def test_prediction_interval_coercion(self):
        """Float values should be coerced to Decimal."""
        interval = PredictionInterval(
            point_estimate=100.0,  # float
            lower_bound=90.0,
            upper_bound=110.0,
            confidence_level=0.90,
            variable_name="test",
            unit="units"
        )

        assert isinstance(interval.point_estimate, Decimal)
        assert isinstance(interval.confidence_level, Decimal)


class TestQuantileSet:
    """Tests for QuantileSet schema."""

    def test_create_quantile_set(self):
        """Valid quantile set should be created."""
        quantiles = [
            QuantileValue(probability=Decimal("0.10"), value=Decimal("80")),
            QuantileValue(probability=Decimal("0.50"), value=Decimal("100")),
            QuantileValue(probability=Decimal("0.90"), value=Decimal("120"))
        ]

        qset = QuantileSet(
            quantiles=quantiles,
            variable_name="temperature",
            unit="degC"
        )

        assert len(qset.quantiles) == 3
        assert qset.p10 == Decimal("80")
        assert qset.p50 == Decimal("100")
        assert qset.p90 == Decimal("120")

    def test_quantile_set_get_interval(self):
        """Get interval between quantiles."""
        quantiles = [
            QuantileValue(probability=Decimal("0.10"), value=Decimal("80")),
            QuantileValue(probability=Decimal("0.90"), value=Decimal("120"))
        ]

        qset = QuantileSet(
            quantiles=quantiles,
            variable_name="test",
            unit="units"
        )

        lower, upper = qset.get_interval(Decimal("0.10"), Decimal("0.90"))
        assert lower == Decimal("80")
        assert upper == Decimal("120")


class TestScenario:
    """Tests for Scenario and ScenarioSet schemas."""

    def test_create_scenario(self):
        """Valid scenario should be created."""
        now = datetime.utcnow()
        variables = [
            ScenarioVariable(name="temperature", value=Decimal("25.0"), unit="degC"),
            ScenarioVariable(name="demand", value=Decimal("100.0"), unit="MW")
        ]

        scenario = Scenario(
            name="Base Case",
            probability=Decimal("0.5"),
            variables=variables,
            horizon_start=now,
            horizon_end=now + timedelta(hours=24),
            is_base_case=True
        )

        assert scenario.name == "Base Case"
        assert scenario.probability == Decimal("0.5")
        assert len(scenario.variables) == 2
        assert scenario.is_base_case is True

    def test_scenario_get_variable(self):
        """Get variable by name should work."""
        now = datetime.utcnow()
        variables = [
            ScenarioVariable(name="temperature", value=Decimal("25.0"), unit="degC"),
            ScenarioVariable(name="demand", value=Decimal("100.0"), unit="MW")
        ]

        scenario = Scenario(
            name="Test",
            variables=variables,
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )

        temp_var = scenario.get_variable("temperature")
        assert temp_var is not None
        assert temp_var.value == Decimal("25.0")

        missing_var = scenario.get_variable("nonexistent")
        assert missing_var is None

    def test_scenario_get_value(self):
        """Get value by name should work."""
        now = datetime.utcnow()
        variables = [
            ScenarioVariable(name="temperature", value=Decimal("25.0"), unit="degC")
        ]

        scenario = Scenario(
            name="Test",
            variables=variables,
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )

        assert scenario.get_value("temperature") == Decimal("25.0")
        assert scenario.get_value("nonexistent") is None

    def test_scenario_set_probability_validation(self):
        """Scenario set probabilities must sum to 1."""
        now = datetime.utcnow()

        scenario1 = Scenario(
            name="S1",
            probability=Decimal("0.5"),
            variables=[ScenarioVariable(name="x", value=Decimal("1"), unit="u")],
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )
        scenario2 = Scenario(
            name="S2",
            probability=Decimal("0.5"),
            variables=[ScenarioVariable(name="x", value=Decimal("2"), unit="u")],
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )

        # Valid: probabilities sum to 1
        scenario_set = ScenarioSet(
            name="Test Set",
            scenarios=[scenario1, scenario2]
        )
        assert scenario_set.num_scenarios == 2

    def test_scenario_set_invalid_probabilities(self):
        """Invalid probabilities should raise error."""
        now = datetime.utcnow()

        scenario1 = Scenario(
            name="S1",
            probability=Decimal("0.6"),
            variables=[ScenarioVariable(name="x", value=Decimal("1"), unit="u")],
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )
        scenario2 = Scenario(
            name="S2",
            probability=Decimal("0.6"),  # Total = 1.2 > 1
            variables=[ScenarioVariable(name="x", value=Decimal("2"), unit="u")],
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )

        with pytest.raises(ValueError):
            ScenarioSet(
                name="Invalid Set",
                scenarios=[scenario1, scenario2]
            )


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics schema."""

    def test_create_calibration_metrics(self):
        """Valid calibration metrics should be created."""
        now = datetime.utcnow()

        metrics = CalibrationMetrics(
            model_name="test_model",
            variable_name="temperature",
            evaluation_period_start=now - timedelta(days=7),
            evaluation_period_end=now,
            num_predictions=100,
            picp=Decimal("0.88"),
            target_coverage=Decimal("0.90"),
            mpiw=Decimal("10.5"),
            nmpiw=Decimal("0.15")
        )

        assert metrics.model_name == "test_model"
        assert metrics.picp == Decimal("0.88")
        assert metrics.calibration_error == Decimal("0.02")

    def test_calibration_status_well_calibrated(self):
        """Well calibrated status should be set correctly."""
        now = datetime.utcnow()

        metrics = CalibrationMetrics(
            model_name="test",
            variable_name="temp",
            evaluation_period_start=now,
            evaluation_period_end=now,
            num_predictions=100,
            picp=Decimal("0.89"),  # Within 5% of target
            target_coverage=Decimal("0.90"),
            mpiw=Decimal("10"),
            nmpiw=Decimal("0.1")
        )

        assert metrics.status == CalibrationStatus.WELL_CALIBRATED

    def test_calibration_status_over_confident(self):
        """Over confident status should be set correctly."""
        now = datetime.utcnow()

        metrics = CalibrationMetrics(
            model_name="test",
            variable_name="temp",
            evaluation_period_start=now,
            evaluation_period_end=now,
            num_predictions=100,
            picp=Decimal("0.80"),  # Below target - 5%
            target_coverage=Decimal("0.90"),
            mpiw=Decimal("10"),
            nmpiw=Decimal("0.1")
        )

        assert metrics.status == CalibrationStatus.OVER_CONFIDENT

    def test_calibration_status_under_confident(self):
        """Under confident status should be set correctly."""
        now = datetime.utcnow()

        metrics = CalibrationMetrics(
            model_name="test",
            variable_name="temp",
            evaluation_period_start=now,
            evaluation_period_end=now,
            num_predictions=100,
            picp=Decimal("0.98"),  # Above target + 5%
            target_coverage=Decimal("0.90"),
            mpiw=Decimal("10"),
            nmpiw=Decimal("0.1")
        )

        assert metrics.status == CalibrationStatus.UNDER_CONFIDENT


class TestRobustSolution:
    """Tests for RobustSolution schema."""

    def test_create_robust_solution(self):
        """Valid robust solution should be created."""
        solution = RobustSolution(
            objective_value=Decimal("1000.50"),
            objective_type=OptimizationObjective.MIN_EXPECTED_COST,
            decision_variables={"x": Decimal("10"), "y": Decimal("20")},
            scenario_set_id=uuid4(),
            feasibility_rate=Decimal("1.0"),
            solver_status="optimal"
        )

        assert solution.objective_value == Decimal("1000.50")
        assert solution.objective_type == OptimizationObjective.MIN_EXPECTED_COST
        assert solution.feasibility_rate == Decimal("1.0")
        assert len(solution.decision_variables) == 2


class TestRobustConstraint:
    """Tests for RobustConstraint schema."""

    def test_create_hard_constraint(self):
        """Hard constraint should be created correctly."""
        constraint = RobustConstraint(
            name="capacity_limit",
            constraint_type=ConstraintType.HARD,
            expression="demand <= capacity",
            bound=Decimal("100"),
            bound_type="<="
        )

        assert constraint.name == "capacity_limit"
        assert constraint.constraint_type == ConstraintType.HARD
        assert constraint.reliability == Decimal("0.95")  # Default

    def test_create_chance_constraint(self):
        """Chance constraint should be created correctly."""
        constraint = RobustConstraint(
            name="reserve_margin",
            constraint_type=ConstraintType.CHANCE,
            expression="reserve >= min_reserve",
            bound=Decimal("10"),
            bound_type=">=",
            reliability=Decimal("0.99")
        )

        assert constraint.constraint_type == ConstraintType.CHANCE
        assert constraint.reliability == Decimal("0.99")


class TestUncertaintySource:
    """Tests for UncertaintySource schema."""

    def test_create_normal_source(self):
        """Normal distribution source should be created."""
        source = UncertaintySource(
            name="temperature_forecast",
            source_type=UncertaintySourceType.WEATHER,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("25.0"), "std": Decimal("2.0")},
            unit="degC"
        )

        assert source.name == "temperature_forecast"
        assert source.source_type == UncertaintySourceType.WEATHER
        assert source.distribution == DistributionType.NORMAL
        assert source.parameters["mean"] == Decimal("25.0")

    def test_create_lognormal_source(self):
        """Lognormal distribution source should be created."""
        source = UncertaintySource(
            name="electricity_price",
            source_type=UncertaintySourceType.PRICE,
            distribution=DistributionType.LOGNORMAL,
            parameters={"mean": Decimal("0.10"), "sigma": Decimal("0.20")},
            unit="USD/kWh"
        )

        assert source.distribution == DistributionType.LOGNORMAL


class TestReliabilityDiagram:
    """Tests for ReliabilityDiagram schema."""

    def test_create_reliability_diagram(self):
        """Valid reliability diagram should be created."""
        points = [
            ReliabilityDiagramPoint(
                predicted_probability=Decimal("0.1"),
                observed_frequency=Decimal("0.12"),
                num_samples=50
            ),
            ReliabilityDiagramPoint(
                predicted_probability=Decimal("0.5"),
                observed_frequency=Decimal("0.48"),
                num_samples=100
            ),
            ReliabilityDiagramPoint(
                predicted_probability=Decimal("0.9"),
                observed_frequency=Decimal("0.85"),
                num_samples=50
            )
        ]

        diagram = ReliabilityDiagram(
            model_name="test_model",
            variable_name="temperature",
            points=points,
            num_bins=3,
            total_samples=200,
            expected_calibration_error=Decimal("0.03"),
            maximum_calibration_error=Decimal("0.05")
        )

        assert len(diagram.points) == 3
        assert diagram.total_samples == 200
        assert diagram.expected_calibration_error == Decimal("0.03")


class TestRiskAssessment:
    """Tests for RiskAssessment schema."""

    def test_create_risk_assessment(self):
        """Valid risk assessment should be created."""
        assessment = RiskAssessment(
            constraint_name="capacity_limit",
            current_value=Decimal("85"),
            constraint_bound=Decimal("100"),
            headroom=Decimal("15"),
            headroom_percent=Decimal("15.0"),
            probability_of_binding=Decimal("0.25"),
            probability_of_violation=Decimal("0.05")
        )

        assert assessment.constraint_name == "capacity_limit"
        assert assessment.risk_level == "medium"  # Based on prob_binding

    def test_risk_level_critical(self):
        """Critical risk level should be computed."""
        assessment = RiskAssessment(
            constraint_name="test",
            current_value=Decimal("95"),
            constraint_bound=Decimal("100"),
            headroom=Decimal("5"),
            headroom_percent=Decimal("5.0"),
            probability_of_binding=Decimal("0.80"),
            probability_of_violation=Decimal("0.15")  # > 10%
        )

        assert assessment.risk_level == "critical"


class TestFanChartData:
    """Tests for FanChartData schema."""

    def test_create_fan_chart_data(self):
        """Valid fan chart data should be created."""
        now = datetime.utcnow()
        timestamps = [now + timedelta(hours=i) for i in range(5)]

        fan_chart = FanChartData(
            variable_name="temperature",
            unit="degC",
            timestamps=timestamps,
            central_values=[Decimal("25"), Decimal("26"), Decimal("27"), Decimal("26"), Decimal("25")],
            bands={
                "90%": [
                    (Decimal("22"), Decimal("28")),
                    (Decimal("22"), Decimal("30")),
                    (Decimal("23"), Decimal("31")),
                    (Decimal("22"), Decimal("30")),
                    (Decimal("21"), Decimal("29"))
                ]
            }
        )

        assert len(fan_chart.timestamps) == 5
        assert len(fan_chart.central_values) == 5
        assert "90%" in fan_chart.bands

    def test_fan_chart_alignment_validation(self):
        """Misaligned data should raise error."""
        now = datetime.utcnow()
        timestamps = [now + timedelta(hours=i) for i in range(5)]

        with pytest.raises(ValueError):
            FanChartData(
                variable_name="test",
                unit="units",
                timestamps=timestamps,
                central_values=[Decimal("1"), Decimal("2"), Decimal("3")],  # Wrong length
                bands={}
            )
