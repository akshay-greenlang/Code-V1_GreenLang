# -*- coding: utf-8 -*-
"""
Tests for Report Generator module.

Validates:
- Per-decision explanation reports
- Batch explanation summaries
- Dashboard-ready data formats
- JSON export
- Audit record generation
"""

import pytest
import json
import numpy as np
from datetime import datetime

from explainability.report_generator import (
    ReportGenerator,
    ReportConfig,
    ReportMetadata,
    generate_quick_report,
    export_report_json,
)
from explainability.explanation_schemas import (
    ExplanationReport,
    SHAPExplanation,
    LIMEExplanation,
    DecisionExplanation,
    Counterfactual,
    FeatureContribution,
    UncertaintyRange,
    ConfidenceBounds,
    BatchExplanationSummary,
    DashboardExplanationData,
    PredictionType,
    ConfidenceLevel,
)


@pytest.fixture
def sample_feature_contributions():
    """Create sample feature contributions."""
    return [
        FeatureContribution(
            feature_name="temperature_inlet_c",
            feature_value=350.0,
            contribution=25.5,
            contribution_percentage=35.0,
            direction="positive",
            unit="celsius"
        ),
        FeatureContribution(
            feature_name="flow_rate_kg_s",
            feature_value=12.5,
            contribution=18.2,
            contribution_percentage=25.0,
            direction="positive",
            unit="kg/s"
        ),
        FeatureContribution(
            feature_name="ambient_temp_c",
            feature_value=25.0,
            contribution=-15.0,
            contribution_percentage=20.0,
            direction="negative",
            unit="celsius"
        ),
        FeatureContribution(
            feature_name="load_factor",
            feature_value=0.85,
            contribution=10.3,
            contribution_percentage=14.0,
            direction="positive"
        ),
        FeatureContribution(
            feature_name="pressure_bar",
            feature_value=15.0,
            contribution=-4.5,
            contribution_percentage=6.0,
            direction="negative",
            unit="bar"
        )
    ]


@pytest.fixture
def sample_shap_explanation(sample_feature_contributions):
    """Create sample SHAP explanation."""
    return SHAPExplanation(
        explanation_id="shap_001",
        prediction_type=PredictionType.DEMAND_FORECAST,
        base_value=50.0,
        prediction_value=84.5,
        feature_contributions=sample_feature_contributions,
        consistency_check=0.001,
        explainer_type="tree",
        timestamp=datetime.utcnow(),
        computation_time_ms=150.0,
        random_seed=42
    )


@pytest.fixture
def sample_lime_explanation(sample_feature_contributions):
    """Create sample LIME explanation."""
    return LIMEExplanation(
        explanation_id="lime_001",
        prediction_type=PredictionType.DEMAND_FORECAST,
        prediction_value=84.5,
        feature_contributions=sample_feature_contributions,
        local_model_r2=0.92,
        local_model_intercept=48.0,
        neighborhood_size=5000,
        kernel_width=0.75,
        timestamp=datetime.utcnow(),
        computation_time_ms=250.0,
        random_seed=42
    )


@pytest.fixture
def sample_uncertainty():
    """Create sample uncertainty range."""
    return UncertaintyRange(
        point_estimate=84.5,
        standard_error=3.2,
        confidence_interval=ConfidenceBounds(
            lower_bound=78.1,
            upper_bound=90.9,
            confidence_level=0.95,
            method="bootstrap"
        ),
        prediction_variance=10.24,
        epistemic_uncertainty=1.9,
        aleatoric_uncertainty=1.3
    )


@pytest.fixture
def sample_counterfactual():
    """Create sample counterfactual."""
    return Counterfactual(
        counterfactual_id="cf_001",
        original_prediction=84.5,
        target_prediction=100.0,
        feature_changes={
            "temperature_inlet_c": {"from": 350.0, "to": 400.0, "change": 50.0},
            "load_factor": {"from": 0.85, "to": 0.95, "change": 0.1}
        },
        feasibility_score=0.85,
        sparsity=2,
        distance=51.0,
        validity=True,
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_explanation_report(
    sample_shap_explanation,
    sample_lime_explanation,
    sample_uncertainty,
    sample_counterfactual,
    sample_feature_contributions
):
    """Create sample explanation report."""
    return ExplanationReport(
        report_id="report_001",
        prediction_type=PredictionType.DEMAND_FORECAST,
        model_name="DemandForecaster",
        model_version="1.0.0",
        input_features={
            "temperature_inlet_c": 350.0,
            "flow_rate_kg_s": 12.5,
            "ambient_temp_c": 25.0,
            "load_factor": 0.85,
            "pressure_bar": 15.0
        },
        prediction_value=84.5,
        uncertainty=sample_uncertainty,
        confidence_level=ConfidenceLevel.HIGH,
        shap_explanation=sample_shap_explanation,
        lime_explanation=sample_lime_explanation,
        counterfactuals=[sample_counterfactual],
        top_features=sample_feature_contributions,
        narrative_summary="The 24-hour demand forecast is 84.5 MW.",
        timestamp=datetime.utcnow(),
        computation_time_ms=500.0,
        provenance_hash="abc123def456",
        deterministic=True
    )


class TestReportConfig:
    """Tests for report configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()
        assert config.include_raw_data is True
        assert config.include_visualizations is True
        assert config.max_features_display == 10
        assert config.decimal_precision == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            max_features_display=5,
            decimal_precision=2,
            include_raw_data=False
        )
        assert config.max_features_display == 5
        assert config.decimal_precision == 2
        assert config.include_raw_data is False


@pytest.mark.reports
class TestReportGenerator:
    """Tests for report generator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = ReportGenerator()
        assert generator.config is not None

    def test_initialization_with_config(self):
        """Test generator initialization with custom config."""
        config = ReportConfig(decimal_precision=2)
        generator = ReportGenerator(config=config)
        assert generator.config.decimal_precision == 2

    def test_generate_decision_report(self, sample_explanation_report):
        """Test generating decision report."""
        generator = ReportGenerator()
        report = generator.generate_decision_report(sample_explanation_report)

        assert "report_type" in report
        assert report["report_type"] == "decision_explanation"
        assert "metadata" in report
        assert "summary" in report
        assert "prediction" in report
        assert "explanations" in report
        assert "uncertainty" in report
        assert "top_features" in report

    def test_report_metadata(self, sample_explanation_report):
        """Test report metadata generation."""
        generator = ReportGenerator()
        report = generator.generate_decision_report(sample_explanation_report)

        metadata = report["metadata"]
        assert "report_id" in metadata
        assert "report_type" in metadata
        assert "generation_timestamp" in metadata
        assert "generator" in metadata
        assert metadata["deterministic"] is True

    def test_report_summary(self, sample_explanation_report):
        """Test report summary generation."""
        generator = ReportGenerator()
        report = generator.generate_decision_report(sample_explanation_report)

        summary = report["summary"]
        assert "prediction_type" in summary
        assert "prediction_value" in summary
        assert "confidence" in summary
        assert "top_driver" in summary
        assert "narrative" in summary

    def test_report_explanations_section(self, sample_explanation_report):
        """Test explanations section in report."""
        generator = ReportGenerator()
        report = generator.generate_decision_report(sample_explanation_report)

        explanations = report["explanations"]
        assert "shap" in explanations
        assert "lime" in explanations
        assert "base_value" in explanations["shap"]
        assert "local_model_r2" in explanations["lime"]

    def test_report_uncertainty_section(self, sample_explanation_report):
        """Test uncertainty section in report."""
        generator = ReportGenerator()
        report = generator.generate_decision_report(sample_explanation_report)

        uncertainty = report["uncertainty"]
        assert "point_estimate" in uncertainty
        assert "standard_error" in uncertainty
        assert "confidence_interval" in uncertainty
        assert uncertainty["confidence_interval"]["lower"] <= uncertainty["point_estimate"]
        assert uncertainty["confidence_interval"]["upper"] >= uncertainty["point_estimate"]

    def test_report_counterfactuals_section(self, sample_explanation_report):
        """Test counterfactuals section in report."""
        generator = ReportGenerator()
        report = generator.generate_decision_report(sample_explanation_report)

        assert "counterfactuals" in report
        assert len(report["counterfactuals"]) > 0
        cf = report["counterfactuals"][0]
        assert "target_prediction" in cf
        assert "feature_changes" in cf
        assert "feasibility" in cf

    def test_report_with_additional_context(self, sample_explanation_report):
        """Test report with additional context."""
        generator = ReportGenerator()
        context = {
            "facility_id": "PLANT-001",
            "timestamp": datetime.utcnow().isoformat()
        }
        report = generator.generate_decision_report(
            sample_explanation_report,
            additional_context=context
        )

        assert "context" in report
        assert report["context"]["facility_id"] == "PLANT-001"

    def test_number_formatting(self, sample_explanation_report):
        """Test number formatting in report."""
        config = ReportConfig(decimal_precision=2)
        generator = ReportGenerator(config=config)
        report = generator.generate_decision_report(sample_explanation_report)

        # Check that numbers are formatted with correct precision
        prediction = report["prediction"]["value"]
        assert isinstance(prediction, float)
        # Value should be rounded to 2 decimal places
        assert prediction == round(prediction, 2)


@pytest.mark.reports
class TestBatchReport:
    """Tests for batch report generation."""

    @pytest.fixture
    def sample_batch_summary(self):
        """Create sample batch summary."""
        return BatchExplanationSummary(
            summary_id="batch_001",
            prediction_type=PredictionType.DEMAND_FORECAST,
            batch_size=100,
            global_feature_importance={
                "temperature_inlet_c": 0.35,
                "flow_rate_kg_s": 0.25,
                "ambient_temp_c": 0.20,
                "load_factor": 0.14,
                "pressure_bar": 0.06
            },
            feature_importance_std={
                "temperature_inlet_c": 0.05,
                "flow_rate_kg_s": 0.04,
                "ambient_temp_c": 0.03,
                "load_factor": 0.02,
                "pressure_bar": 0.01
            },
            mean_prediction=85.0,
            std_prediction=12.5,
            mean_confidence=0.92,
            mean_shap_consistency=0.002,
            mean_lime_r2=0.88,
            timestamp=datetime.utcnow(),
            computation_time_ms=5000.0
        )

    def test_generate_batch_report(self, sample_batch_summary):
        """Test generating batch report."""
        generator = ReportGenerator()
        report = generator.generate_batch_report(sample_batch_summary)

        assert report["report_type"] == "batch_explanation_summary"
        assert "batch_statistics" in report
        assert "global_feature_importance" in report
        assert "quality_metrics" in report

    def test_batch_statistics(self, sample_batch_summary):
        """Test batch statistics in report."""
        generator = ReportGenerator()
        report = generator.generate_batch_report(sample_batch_summary)

        stats = report["batch_statistics"]
        assert stats["batch_size"] == 100
        assert "mean_prediction" in stats
        assert "std_prediction" in stats

    def test_batch_feature_importance(self, sample_batch_summary):
        """Test feature importance in batch report."""
        generator = ReportGenerator()
        report = generator.generate_batch_report(sample_batch_summary)

        importance = report["global_feature_importance"]
        assert isinstance(importance, list)
        assert len(importance) > 0
        assert "feature" in importance[0]
        assert "importance" in importance[0]
        assert "std" in importance[0]


@pytest.mark.reports
class TestDashboardData:
    """Tests for dashboard data generation."""

    def test_generate_dashboard_data(self, sample_explanation_report):
        """Test generating dashboard data."""
        generator = ReportGenerator()
        dashboard_data = generator.generate_dashboard_data(sample_explanation_report)

        assert isinstance(dashboard_data, DashboardExplanationData)
        assert len(dashboard_data.waterfall_data) > 0
        assert len(dashboard_data.force_plot_data) > 0
        assert len(dashboard_data.feature_importance_chart) > 0

    def test_waterfall_data_structure(self, sample_explanation_report):
        """Test waterfall data structure."""
        generator = ReportGenerator()
        dashboard_data = generator.generate_dashboard_data(sample_explanation_report)

        waterfall = dashboard_data.waterfall_data
        # Should have base value, features, and prediction
        assert waterfall[0]["name"] == "Base Value"
        assert waterfall[-1]["name"] == "Prediction"

        for item in waterfall:
            assert "name" in item
            assert "value" in item
            assert "cumulative" in item
            assert "type" in item

    def test_force_plot_data_structure(self, sample_explanation_report):
        """Test force plot data structure."""
        generator = ReportGenerator()
        dashboard_data = generator.generate_dashboard_data(sample_explanation_report)

        force_plot = dashboard_data.force_plot_data
        assert "base_value" in force_plot
        assert "prediction" in force_plot
        assert "positive_features" in force_plot
        assert "negative_features" in force_plot
        assert "positive_sum" in force_plot
        assert "negative_sum" in force_plot

    def test_feature_importance_chart_structure(self, sample_explanation_report):
        """Test feature importance chart structure."""
        generator = ReportGenerator()
        dashboard_data = generator.generate_dashboard_data(sample_explanation_report)

        chart = dashboard_data.feature_importance_chart
        for item in chart:
            assert "feature" in item
            assert "importance" in item
            assert "direction" in item

    def test_counterfactual_chart(self, sample_explanation_report):
        """Test counterfactual chart data."""
        generator = ReportGenerator()
        dashboard_data = generator.generate_dashboard_data(sample_explanation_report)

        if dashboard_data.counterfactual_chart:
            for cf in dashboard_data.counterfactual_chart:
                assert "id" in cf
                assert "sparsity" in cf
                assert "feasibility" in cf


@pytest.mark.reports
class TestOptimizationReport:
    """Tests for optimization report generation."""

    @pytest.fixture
    def sample_decision_explanation(self):
        """Create sample decision explanation."""
        return DecisionExplanation(
            decision_id="opt_001",
            objective_value=12500.0,
            binding_constraints=["max_boiler_1", "total_demand"],
            shadow_prices={
                "total_demand": 25.0,
                "max_boiler_1": 5.0,
                "min_efficiency": -10.0
            },
            reduced_costs={
                "boiler_1_output": 0.0,
                "boiler_2_output": 2.0,
                "heat_pump_output": -1.0
            },
            sensitivity_analysis={
                "total_demand": {
                    "shadow_price": 25.0,
                    "allowable_increase": 10.0,
                    "allowable_decrease": 5.0
                }
            },
            alternative_solutions=[
                {
                    "objective_value": 12600.0,
                    "gap_from_optimal": 0.008,
                    "changed_variable": "heat_pump_output"
                }
            ],
            optimality_gap=0.001,
            timestamp=datetime.utcnow()
        )

    def test_generate_optimization_report(self, sample_decision_explanation):
        """Test generating optimization report."""
        generator = ReportGenerator()
        report = generator.generate_optimization_report(sample_decision_explanation)

        assert report["report_type"] == "optimization_decision"
        assert "objective" in report
        assert "binding_constraints" in report
        assert "shadow_prices" in report
        assert "reduced_costs" in report
        assert "sensitivity_analysis" in report

    def test_optimization_objective(self, sample_decision_explanation):
        """Test objective section in optimization report."""
        generator = ReportGenerator()
        report = generator.generate_optimization_report(sample_decision_explanation)

        objective = report["objective"]
        assert "value" in objective
        assert "optimality_gap" in objective

    def test_optimization_alternatives(self, sample_decision_explanation):
        """Test alternatives section in optimization report."""
        generator = ReportGenerator()
        report = generator.generate_optimization_report(sample_decision_explanation)

        assert "alternatives" in report
        assert len(report["alternatives"]) > 0


@pytest.mark.reports
class TestJSONExport:
    """Tests for JSON export functionality."""

    def test_export_to_json(self, sample_explanation_report):
        """Test exporting report to JSON."""
        generator = ReportGenerator()
        report = generator.generate_decision_report(sample_explanation_report)
        json_str = generator.export_to_json(report)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "report_type" in parsed

    def test_export_preserves_structure(self, sample_explanation_report):
        """Test that JSON export preserves structure."""
        generator = ReportGenerator()
        report = generator.generate_decision_report(sample_explanation_report)
        json_str = generator.export_to_json(report)
        parsed = json.loads(json_str)

        # Check key sections exist
        assert parsed["report_type"] == report["report_type"]
        assert "prediction" in parsed
        assert "uncertainty" in parsed


@pytest.mark.reports
class TestAuditRecord:
    """Tests for audit record generation."""

    def test_generate_audit_record(self, sample_explanation_report):
        """Test generating audit record."""
        generator = ReportGenerator()
        audit = generator.generate_audit_record(sample_explanation_report)

        assert audit["audit_type"] == "ml_explanation"
        assert "report_id" in audit
        assert "timestamp" in audit
        assert "model_info" in audit
        assert "prediction" in audit
        assert "determinism" in audit
        assert "input_hash" in audit

    def test_audit_determinism_info(self, sample_explanation_report):
        """Test determinism info in audit record."""
        generator = ReportGenerator()
        audit = generator.generate_audit_record(sample_explanation_report)

        determinism = audit["determinism"]
        assert determinism["is_deterministic"] is True
        assert "provenance_hash" in determinism


@pytest.mark.reports
class TestReportHistory:
    """Tests for report history tracking."""

    def test_track_reports(self, sample_explanation_report):
        """Test that reports are tracked."""
        generator = ReportGenerator()

        # Generate multiple reports
        for _ in range(5):
            generator.generate_decision_report(sample_explanation_report)

        history = generator.get_report_history()
        assert len(history) == 5

    def test_filter_history_by_type(self, sample_explanation_report):
        """Test filtering history by report type."""
        generator = ReportGenerator()

        # Generate decision reports
        for _ in range(3):
            generator.generate_decision_report(sample_explanation_report)

        history = generator.get_report_history(report_type="decision_explanation")
        assert len(history) == 3

    def test_history_limit(self, sample_explanation_report):
        """Test history limit parameter."""
        generator = ReportGenerator()

        # Generate multiple reports
        for _ in range(10):
            generator.generate_decision_report(sample_explanation_report)

        history = generator.get_report_history(limit=5)
        assert len(history) == 5


@pytest.mark.reports
class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_quick_report(self, sample_explanation_report):
        """Test quick report generation."""
        report = generate_quick_report(sample_explanation_report)

        assert "report_type" in report
        assert "prediction" in report

    def test_export_report_json_function(self, sample_explanation_report):
        """Test export function."""
        json_str = export_report_json(sample_explanation_report)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
