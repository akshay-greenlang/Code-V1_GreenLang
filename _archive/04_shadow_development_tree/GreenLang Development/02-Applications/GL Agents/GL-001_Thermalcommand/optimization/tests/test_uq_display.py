"""
Tests for UQ Display

Tests for uncertainty visualization data generation including:
    - Summary display generation
    - Fan chart data generation
    - Risk assessment
    - Scenario comparison

All tests verify determinism and correctness.

Author: GreenLang Process Heat Team
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ..uq_display import (
    UQDisplayEngine,
    FanChartGenerator,
    RiskAssessmentEngine,
    ScenarioComparisonEngine,
)
from ..uq_schemas import (
    PredictionInterval,
    QuantileSet,
    QuantileValue,
    Scenario,
    ScenarioSet,
    ScenarioVariable,
    UncertaintySource,
    UncertaintySourceType,
    DistributionType,
    RobustSolution,
    OptimizationObjective,
)
from ..uncertainty_models import UncertaintyModelEngine


def create_test_scenario_set() -> ScenarioSet:
    """Create a simple scenario set for testing."""
    now = datetime.utcnow()

    scenarios = [
        Scenario(
            name="Low",
            probability=Decimal("0.3"),
            variables=[
                ScenarioVariable(name="cost", value=Decimal("800"), unit="USD"),
                ScenarioVariable(name="emissions", value=Decimal("50"), unit="tCO2e")
            ],
            horizon_start=now,
            horizon_end=now + timedelta(hours=24)
        ),
        Scenario(
            name="Base",
            probability=Decimal("0.5"),
            variables=[
                ScenarioVariable(name="cost", value=Decimal("1000"), unit="USD"),
                ScenarioVariable(name="emissions", value=Decimal("60"), unit="tCO2e")
            ],
            horizon_start=now,
            horizon_end=now + timedelta(hours=24),
            is_base_case=True
        ),
        Scenario(
            name="High",
            probability=Decimal("0.2"),
            variables=[
                ScenarioVariable(name="cost", value=Decimal("1200"), unit="USD"),
                ScenarioVariable(name="emissions", value=Decimal("70"), unit="tCO2e")
            ],
            horizon_start=now,
            horizon_end=now + timedelta(hours=24),
            is_worst_case=True
        )
    ]

    return ScenarioSet(name="Test Scenarios", scenarios=scenarios)


class TestUQDisplayEngine:
    """Tests for UQDisplayEngine."""

    def test_generate_summary_display(self):
        """Should generate summary display correctly."""
        engine = UQDisplayEngine()

        prediction = PredictionInterval(
            point_estimate=Decimal("100.5"),
            lower_bound=Decimal("90.0"),
            upper_bound=Decimal("111.0"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC",
            horizon_minutes=60,
            source_model="test_model"
        )

        summary = engine.generate_summary_display(prediction)

        assert summary["variable_name"] == "temperature"
        assert summary["unit"] == "degC"
        assert summary["expected_value"] == "100.5"
        assert summary["p10"] == "90.0"
        assert summary["p90"] == "111.0"
        assert summary["confidence_level"] == "0.90"
        assert "display" in summary
        assert "expected" in summary["display"]
        assert "range" in summary["display"]

    def test_generate_multi_quantile_display(self):
        """Should generate multi-quantile display."""
        engine = UQDisplayEngine()

        quantiles = [
            QuantileValue(probability=Decimal("0.10"), value=Decimal("80")),
            QuantileValue(probability=Decimal("0.50"), value=Decimal("100")),
            QuantileValue(probability=Decimal("0.90"), value=Decimal("120"))
        ]

        qset = QuantileSet(
            quantiles=quantiles,
            variable_name="demand",
            unit="MW"
        )

        display = engine.generate_multi_quantile_display(qset)

        assert display["variable_name"] == "demand"
        assert len(display["quantiles"]) == 3
        assert "highlighted" in display
        assert "p10" in display["highlighted"]
        assert "p50" in display["highlighted"]
        assert "p90" in display["highlighted"]

    def test_generate_scenario_display(self):
        """Should generate scenario display correctly."""
        engine = UQDisplayEngine()

        scenario_set = create_test_scenario_set()

        display = engine.generate_scenario_display(
            scenario_set=scenario_set,
            metric_name="cost"
        )

        assert display["metric_name"] == "cost"
        assert display["unit"] == "USD"
        assert display["num_scenarios"] == 3
        assert "statistics" in display
        assert "expected_value" in display["statistics"]
        assert "minimum" in display["statistics"]
        assert "maximum" in display["statistics"]
        assert len(display["scenarios"]) == 3

    def test_generate_scenario_display_with_expected_value(self):
        """Expected value should be probability-weighted."""
        engine = UQDisplayEngine()

        scenario_set = create_test_scenario_set()

        display = engine.generate_scenario_display(
            scenario_set=scenario_set,
            metric_name="cost"
        )

        # Expected = 0.3 * 800 + 0.5 * 1000 + 0.2 * 1200 = 240 + 500 + 240 = 980
        expected = Decimal("980")
        actual = Decimal(display["statistics"]["expected_value"])

        assert abs(actual - expected) < Decimal("1")

    def test_generate_solution_display(self):
        """Should generate solution display correctly."""
        engine = UQDisplayEngine()

        from uuid import uuid4
        solution = RobustSolution(
            objective_value=Decimal("1500.50"),
            objective_type=OptimizationObjective.MIN_EXPECTED_COST,
            decision_variables={
                "production": Decimal("100"),
                "reserve": Decimal("20")
            },
            scenario_set_id=uuid4(),
            feasibility_rate=Decimal("0.95"),
            expected_objective=Decimal("1450"),
            worst_case_objective=Decimal("1800"),
            solver_status="optimal",
            solve_time_ms=15.5
        )

        scenario_set = create_test_scenario_set()

        display = engine.generate_solution_display(solution, scenario_set)

        assert display["objective_type"] == "min_expected_cost"
        assert display["objective_value"] == "1500.50"
        assert display["solver_status"] == "optimal"
        assert display["feasibility"]["percent"] == "95%"
        assert len(display["decisions"]) == 2

    def test_format_value_handles_magnitudes(self):
        """Should format values appropriately for display."""
        engine = UQDisplayEngine()

        # Test large value
        prediction_large = PredictionInterval(
            point_estimate=Decimal("1500000"),
            lower_bound=Decimal("1400000"),
            upper_bound=Decimal("1600000"),
            confidence_level=Decimal("0.90"),
            variable_name="cost",
            unit="USD"
        )

        summary_large = engine.generate_summary_display(prediction_large)
        assert "M" in summary_large["display"]["expected"]  # Millions

        # Test small value
        prediction_small = PredictionInterval(
            point_estimate=Decimal("0.0015"),
            lower_bound=Decimal("0.001"),
            upper_bound=Decimal("0.002"),
            confidence_level=Decimal("0.90"),
            variable_name="efficiency",
            unit=""
        )

        summary_small = engine.generate_summary_display(prediction_small)
        # Should handle small values


class TestFanChartGenerator:
    """Tests for FanChartGenerator."""

    def test_generate_fan_chart(self):
        """Should generate fan chart data."""
        generator = FanChartGenerator()

        now = datetime.utcnow()
        timestamps = [now + timedelta(hours=i) for i in range(5)]
        forecasts = [
            Decimal("100"), Decimal("102"), Decimal("105"),
            Decimal("103"), Decimal("100")
        ]

        source = UncertaintySource(
            name="temperature",
            source_type=UncertaintySourceType.WEATHER,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("5")},
            unit="degC"
        )

        fan_chart = generator.generate(
            point_forecasts=forecasts,
            timestamps=timestamps,
            uncertainty_source=source
        )

        assert fan_chart.variable_name == "temperature"
        assert fan_chart.unit == "degC"
        assert len(fan_chart.timestamps) == 5
        assert len(fan_chart.central_values) == 5
        assert "90%" in fan_chart.bands
        assert len(fan_chart.bands["90%"]) == 5

    def test_fan_chart_bands_ordered(self):
        """Confidence bands should be properly ordered."""
        generator = FanChartGenerator()

        now = datetime.utcnow()
        timestamps = [now + timedelta(hours=i) for i in range(3)]
        forecasts = [Decimal("100"), Decimal("100"), Decimal("100")]

        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MODEL,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        fan_chart = generator.generate(
            point_forecasts=forecasts,
            timestamps=timestamps,
            uncertainty_source=source,
            confidence_levels=[Decimal("0.50"), Decimal("0.90"), Decimal("0.95")]
        )

        # Wider confidence should have wider bands
        for i in range(3):
            lower_50, upper_50 = fan_chart.bands["50%"][i]
            lower_90, upper_90 = fan_chart.bands["90%"][i]
            lower_95, upper_95 = fan_chart.bands["95%"][i]

            width_50 = upper_50 - lower_50
            width_90 = upper_90 - lower_90
            width_95 = upper_95 - lower_95

            assert width_50 <= width_90 <= width_95

    def test_generate_from_scenarios(self):
        """Should generate fan chart from scenarios."""
        generator = FanChartGenerator()

        scenario_set = create_test_scenario_set()
        now = datetime.utcnow()
        timestamps = [now + timedelta(hours=i) for i in range(3)]

        fan_chart = generator.generate_from_scenarios(
            scenario_set=scenario_set,
            variable_name="cost",
            timestamps=timestamps
        )

        assert fan_chart.variable_name == "cost"
        assert fan_chart.unit == "USD"
        assert len(fan_chart.timestamps) == 3
        assert len(fan_chart.central_values) == 3

    def test_to_highcharts_format(self):
        """Should convert to Highcharts format."""
        generator = FanChartGenerator()

        now = datetime.utcnow()
        timestamps = [now + timedelta(hours=i) for i in range(3)]
        forecasts = [Decimal("100"), Decimal("100"), Decimal("100")]

        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MODEL,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        fan_chart = generator.generate(forecasts, timestamps, source)

        highcharts_data = generator.to_chart_format(fan_chart, "highcharts")

        assert "chart" in highcharts_data
        assert "series" in highcharts_data
        assert highcharts_data["chart"]["type"] == "line"

    def test_to_plotly_format(self):
        """Should convert to Plotly format."""
        generator = FanChartGenerator()

        now = datetime.utcnow()
        timestamps = [now + timedelta(hours=i) for i in range(3)]
        forecasts = [Decimal("100"), Decimal("100"), Decimal("100")]

        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MODEL,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        fan_chart = generator.generate(forecasts, timestamps, source)

        plotly_data = generator.to_chart_format(fan_chart, "plotly")

        assert "data" in plotly_data
        assert "layout" in plotly_data


class TestRiskAssessmentEngine:
    """Tests for RiskAssessmentEngine."""

    def test_assess_constraint_risk_low(self):
        """Should assess low risk correctly."""
        engine = RiskAssessmentEngine()

        assessment = engine.assess_constraint_risk(
            constraint_name="capacity",
            current_value=Decimal("50"),
            constraint_bound=Decimal("100"),
            bound_type="<=",
            uncertainty_std=Decimal("5"),
            horizon_minutes=60
        )

        assert assessment.constraint_name == "capacity"
        assert assessment.headroom == Decimal("50")  # 100 - 50
        assert assessment.probability_of_binding < Decimal("0.2")
        assert assessment.risk_level == "low"

    def test_assess_constraint_risk_high(self):
        """Should assess high risk correctly."""
        engine = RiskAssessmentEngine()

        assessment = engine.assess_constraint_risk(
            constraint_name="capacity",
            current_value=Decimal("95"),
            constraint_bound=Decimal("100"),
            bound_type="<=",
            uncertainty_std=Decimal("10"),
            horizon_minutes=60
        )

        assert assessment.headroom == Decimal("5")
        assert assessment.probability_of_binding > Decimal("0.3")
        assert assessment.risk_level in ["high", "critical"]

    def test_assess_constraint_risk_critical(self):
        """Should assess critical risk correctly."""
        engine = RiskAssessmentEngine()

        assessment = engine.assess_constraint_risk(
            constraint_name="capacity",
            current_value=Decimal("99"),
            constraint_bound=Decimal("100"),
            bound_type="<=",
            uncertainty_std=Decimal("10"),
            horizon_minutes=60
        )

        assert assessment.probability_of_violation > Decimal("0.1")
        assert assessment.risk_level == "critical"
        assert assessment.recommended_action is not None

    def test_assess_constraint_risk_ge_bound(self):
        """Should handle >= bound type."""
        engine = RiskAssessmentEngine()

        assessment = engine.assess_constraint_risk(
            constraint_name="reserve",
            current_value=Decimal("15"),
            constraint_bound=Decimal("10"),
            bound_type=">=",
            uncertainty_std=Decimal("2"),
            horizon_minutes=60
        )

        # Headroom = current - bound for >= constraint
        assert assessment.headroom == Decimal("5")
        assert assessment.risk_level == "low"

    def test_assess_multiple_constraints(self):
        """Should assess multiple constraints and sort by risk."""
        engine = RiskAssessmentEngine()

        constraints = [
            {
                "name": "low_risk",
                "current_value": 50,
                "bound": 100,
                "bound_type": "<=",
                "uncertainty_std": 5
            },
            {
                "name": "high_risk",
                "current_value": 95,
                "bound": 100,
                "bound_type": "<=",
                "uncertainty_std": 10
            },
            {
                "name": "medium_risk",
                "current_value": 80,
                "bound": 100,
                "bound_type": "<=",
                "uncertainty_std": 10
            }
        ]

        assessments = engine.assess_multiple_constraints(constraints)

        # Should be sorted by risk level (highest first)
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for i in range(len(assessments) - 1):
            assert risk_order[assessments[i].risk_level] <= risk_order[assessments[i + 1].risk_level]

    def test_generate_risk_dashboard_data(self):
        """Should generate dashboard data."""
        engine = RiskAssessmentEngine()

        constraints = [
            {
                "name": "c1",
                "current_value": 50,
                "bound": 100,
                "bound_type": "<=",
                "uncertainty_std": 5
            },
            {
                "name": "c2",
                "current_value": 90,
                "bound": 100,
                "bound_type": "<=",
                "uncertainty_std": 10
            }
        ]

        assessments = engine.assess_multiple_constraints(constraints)
        dashboard = engine.generate_risk_dashboard_data(assessments)

        assert "summary" in dashboard
        assert "by_risk_level" in dashboard
        assert "top_risks" in dashboard
        assert dashboard["summary"]["total_constraints"] == 2

    def test_risk_assessment_has_provenance(self):
        """Risk assessment should have provenance."""
        engine = RiskAssessmentEngine()

        assessment = engine.assess_constraint_risk(
            constraint_name="test",
            current_value=Decimal("50"),
            constraint_bound=Decimal("100"),
            bound_type="<=",
            uncertainty_std=Decimal("5")
        )

        assert assessment.provenance is not None
        assert assessment.provenance.calculation_type == "constraint_risk_assessment"


class TestScenarioComparisonEngine:
    """Tests for ScenarioComparisonEngine."""

    def test_compare_scenarios(self):
        """Should compare scenarios correctly."""
        engine = ScenarioComparisonEngine()

        scenario_set = create_test_scenario_set()

        comparison = engine.compare_scenarios(
            scenario_set=scenario_set,
            metric_name="cost"
        )

        assert comparison.metric_name == "cost"
        assert comparison.unit == "USD"
        assert comparison.min_value == Decimal("800")
        assert comparison.max_value == Decimal("1200")
        assert comparison.range_value == Decimal("400")

    def test_compare_scenarios_expected_value(self):
        """Expected value should be probability-weighted."""
        engine = ScenarioComparisonEngine()

        scenario_set = create_test_scenario_set()

        comparison = engine.compare_scenarios(
            scenario_set=scenario_set,
            metric_name="cost"
        )

        # Expected = 0.3 * 800 + 0.5 * 1000 + 0.2 * 1200 = 980
        expected = Decimal("980")
        assert abs(comparison.expected_value - expected) < Decimal("1")

    def test_compare_scenarios_with_regret(self):
        """Should compute regret values."""
        engine = ScenarioComparisonEngine()

        scenario_set = create_test_scenario_set()

        comparison = engine.compare_scenarios(
            scenario_set=scenario_set,
            metric_name="cost",
            compute_regret=True
        )

        assert comparison.regret_by_scenario is not None
        assert comparison.max_regret is not None

        # Regret for minimum scenario should be 0
        assert comparison.regret_by_scenario["Low"] == Decimal("0")

        # Regret for max scenario
        assert comparison.regret_by_scenario["High"] == Decimal("400")  # 1200 - 800

    def test_generate_comparison_table(self):
        """Should generate comparison table."""
        engine = ScenarioComparisonEngine()

        scenario_set = create_test_scenario_set()

        table = engine.generate_comparison_table(
            scenario_set=scenario_set,
            metric_names=["cost", "emissions"]
        )

        assert "scenarios" in table
        assert "statistics" in table
        assert len(table["scenarios"]) == 3
        assert table["num_scenarios"] == 3
        assert table["num_metrics"] == 2

    def test_comparison_has_provenance(self):
        """Comparison should have provenance."""
        engine = ScenarioComparisonEngine()

        scenario_set = create_test_scenario_set()

        comparison = engine.compare_scenarios(
            scenario_set=scenario_set,
            metric_name="cost"
        )

        assert comparison.provenance is not None
        assert comparison.provenance.calculation_type == "scenario_comparison"

    def test_compare_scenarios_missing_metric(self):
        """Should raise error for missing metric."""
        engine = ScenarioComparisonEngine()

        scenario_set = create_test_scenario_set()

        with pytest.raises(ValueError):
            engine.compare_scenarios(
                scenario_set=scenario_set,
                metric_name="nonexistent"
            )
