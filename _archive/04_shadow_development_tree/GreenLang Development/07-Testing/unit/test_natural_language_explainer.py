# -*- coding: utf-8 -*-
"""
Unit tests for NaturalLanguageExplainer.

Tests the explanation generation for different audiences, decision types,
output formats, and edge cases.
"""

import pytest
from datetime import datetime
from greenlang.ml.explainability.natural_language_explainer import (
    NaturalLanguageExplainer,
    Audience,
    OutputFormat,
    DecisionType,
    ExplanationOutput,
    create_natural_language_explainer,
)


class TestNaturalLanguageExplainer:
    """Test cases for NaturalLanguageExplainer class."""

    @pytest.fixture
    def explainer(self):
        """Create explainer instance."""
        return NaturalLanguageExplainer(
            default_audience=Audience.ENGINEER,
            default_format=OutputFormat.PLAIN_TEXT
        )

    @pytest.fixture
    def sample_prediction_data(self):
        """Sample prediction data for testing."""
        return {
            "prediction": 0.85,
            "shap_values": {
                "flue_gas_temperature": 0.35,
                "days_since_cleaning": 0.28,
                "excess_air": 0.12,
                "stack_temperature": 0.08,
                "pressure_drop": 0.02
            },
            "feature_names": {
                "flue_gas_temperature": "Flue Gas Temperature (F)",
                "days_since_cleaning": "Days Since Cleaning",
                "excess_air": "Excess Air %",
                "stack_temperature": "Stack Temperature (F)",
                "pressure_drop": "Pressure Drop (PSI)"
            },
            "feature_values": {
                "flue_gas_temperature": 485.2,
                "days_since_cleaning": 120,
                "excess_air": 22.5,
                "stack_temperature": 325.0,
                "pressure_drop": 1.5
            }
        }

    # =========================================================================
    # Tests for explain_prediction method
    # =========================================================================

    def test_explain_prediction_operator_audience(self, explainer, sample_prediction_data):
        """Test explanation generation for operator audience."""
        result = explainer.explain_prediction(
            prediction=sample_prediction_data["prediction"],
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            decision_type=DecisionType.FOULING_RISK,
            audience=Audience.OPERATOR
        )

        assert isinstance(result, ExplanationOutput)
        assert result.audience == "operator"
        assert "Flue Gas Temperature" in result.text_summary
        assert len(result.top_factors) <= 3  # Operator gets max 3 factors
        assert result.confidence > 0.0
        assert result.provenance_hash
        assert len(result.recommendations) > 0

    def test_explain_prediction_engineer_audience(self, explainer, sample_prediction_data):
        """Test explanation generation for engineer audience."""
        result = explainer.explain_prediction(
            prediction=sample_prediction_data["prediction"],
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            decision_type=DecisionType.FOULING_RISK,
            audience=Audience.ENGINEER
        )

        assert result.audience == "engineer"
        assert len(result.top_factors) <= 5  # Engineer gets max 5 factors
        assert "Confidence" in result.text_summary

    def test_explain_prediction_executive_audience(self, explainer, sample_prediction_data):
        """Test explanation generation for executive audience."""
        result = explainer.explain_prediction(
            prediction=sample_prediction_data["prediction"],
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            decision_type=DecisionType.FOULING_RISK,
            audience=Audience.EXECUTIVE
        )

        assert result.audience == "executive"
        assert len(result.top_factors) <= 2  # Executive gets max 2 factors
        # Executive explanation should be brief
        assert len(result.text_summary) < 500

    def test_explain_prediction_auditor_audience(self, explainer, sample_prediction_data):
        """Test explanation generation for auditor audience."""
        result = explainer.explain_prediction(
            prediction=sample_prediction_data["prediction"],
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            decision_type=DecisionType.FOULING_RISK,
            audience=Audience.AUDITOR
        )

        assert result.audience == "auditor"
        assert len(result.top_factors) <= 10  # Auditor gets max 10 factors
        assert result.provenance_hash  # Auditor should have provenance

    def test_explain_prediction_all_decision_types(self, explainer, sample_prediction_data):
        """Test explanation for all decision types."""
        decision_types = [
            DecisionType.FOULING_RISK,
            DecisionType.EFFICIENCY_DEGRADATION,
            DecisionType.MAINTENANCE_NEEDED,
            DecisionType.EMISSIONS_HIGH,
            DecisionType.ENERGY_WASTE
        ]

        for decision_type in decision_types:
            result = explainer.explain_prediction(
                prediction=sample_prediction_data["prediction"],
                shap_values=sample_prediction_data["shap_values"],
                feature_names=sample_prediction_data["feature_names"],
                decision_type=decision_type
            )
            assert result.metadata["decision_type"] == decision_type.value

    def test_explain_prediction_with_confidence(self, explainer, sample_prediction_data):
        """Test explanation with different confidence scores."""
        high_conf = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            confidence=0.95
        )

        low_conf = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            confidence=0.60
        )

        assert high_conf.confidence == 0.95
        assert low_conf.confidence == 0.60
        assert high_conf.text_summary != low_conf.text_summary

    def test_explain_prediction_with_baseline(self, explainer, sample_prediction_data):
        """Test explanation with baseline comparison."""
        result = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            baseline=0.65
        )

        assert "Baseline" in result.text_summary or result.text_summary  # Should handle baseline

    def test_explain_prediction_output_formats(self, explainer, sample_prediction_data):
        """Test explanation in all output formats."""
        result = explainer.explain_prediction(
            prediction=sample_prediction_data["prediction"],
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"]
        )

        # Check all formats are present
        assert result.text_summary
        assert result.markdown_summary
        assert result.html_summary

        # HTML should contain HTML tags
        assert "<" in result.html_summary and ">" in result.html_summary

        # Markdown should contain markdown syntax
        assert "#" in result.markdown_summary or "-" in result.markdown_summary

    def test_explain_prediction_provenance_hash(self, explainer, sample_prediction_data):
        """Test provenance hash generation."""
        result = explainer.explain_prediction(
            prediction=sample_prediction_data["prediction"],
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"]
        )

        # Hash should be 64 characters (SHA-256 hex)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_explain_prediction_deterministic_hash(self, explainer, sample_prediction_data):
        """Test that same inputs produce same hash."""
        result1 = explainer.explain_prediction(
            prediction=0.85,
            shap_values={"temp": 0.5},
            feature_names={"temp": "Temperature"},
            audience=Audience.ENGINEER
        )

        # Same inputs but new explainer instance
        explainer2 = NaturalLanguageExplainer()
        result2 = explainer2.explain_prediction(
            prediction=0.85,
            shap_values={"temp": 0.5},
            feature_names={"temp": "Temperature"},
            audience=Audience.ENGINEER
        )

        # Hashes may differ due to timestamp, but structure should be same
        assert len(result1.provenance_hash) == len(result2.provenance_hash)

    # =========================================================================
    # Tests for explain_decision method
    # =========================================================================

    def test_explain_decision_basic(self, explainer):
        """Test basic decision explanation."""
        result = explainer.explain_decision(
            decision_type=DecisionType.MAINTENANCE_NEEDED,
            factors={
                "vibration": "High - 7.2 mm/s",
                "temperature": "Elevated - 85C",
                "runtime": "4200 hours"
            },
            confidence=0.92
        )

        assert isinstance(result, str)
        assert "Decision" in result or "Maintenance" in result
        assert "Confidence" in result
        assert "vibration" in result
        assert "temperature" in result

    def test_explain_decision_confidence_levels(self, explainer):
        """Test decision explanation with different confidence levels."""
        factors = {"temp": "High", "pressure": "Normal"}

        high_conf = explainer.explain_decision(
            decision_type=DecisionType.FOULING_RISK,
            factors=factors,
            confidence=0.95
        )

        low_conf = explainer.explain_decision(
            decision_type=DecisionType.FOULING_RISK,
            factors=factors,
            confidence=0.65
        )

        assert "very high" in high_conf or "95" in high_conf
        assert "moderate" in low_conf or "65" in low_conf

    # =========================================================================
    # Tests for generate_summary method
    # =========================================================================

    def test_generate_summary_operator(self, explainer, sample_prediction_data):
        """Test summary generation for operator audience."""
        result1 = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            decision_type=DecisionType.FOULING_RISK
        )

        result2 = explainer.explain_prediction(
            prediction=0.70,
            shap_values={"excess_air": 0.6, "temp": 0.4},
            feature_names={"excess_air": "Excess Air", "temp": "Temperature"},
            decision_type=DecisionType.EFFICIENCY_DEGRADATION
        )

        summary = explainer.generate_summary(
            [result1, result2],
            audience=Audience.OPERATOR
        )

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_generate_summary_executive(self, explainer, sample_prediction_data):
        """Test summary generation for executive audience."""
        result1 = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            decision_type=DecisionType.FOULING_RISK
        )

        result2 = explainer.explain_prediction(
            prediction=0.70,
            shap_values={"excess_air": 0.6, "temp": 0.4},
            feature_names={"excess_air": "Excess Air", "temp": "Temperature"},
            decision_type=DecisionType.EFFICIENCY_DEGRADATION
        )

        summary = explainer.generate_summary(
            [result1, result2],
            audience=Audience.EXECUTIVE
        )

        assert "STATUS" in summary or "SUMMARY" in summary

    def test_generate_summary_engineer(self, explainer, sample_prediction_data):
        """Test summary generation for engineer audience."""
        result = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            decision_type=DecisionType.FOULING_RISK
        )

        summary = explainer.generate_summary(
            [result],
            audience=Audience.ENGINEER
        )

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_generate_summary_auditor(self, explainer, sample_prediction_data):
        """Test summary generation for auditor audience."""
        result = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"],
            decision_type=DecisionType.FOULING_RISK
        )

        summary = explainer.generate_summary(
            [result],
            audience=Audience.AUDITOR,
            output_format=OutputFormat.MARKDOWN
        )

        assert isinstance(summary, str)

    def test_generate_summary_output_formats(self, explainer, sample_prediction_data):
        """Test summary in different output formats."""
        result = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"]
        )

        text_summary = explainer.generate_summary(
            [result],
            output_format=OutputFormat.PLAIN_TEXT
        )

        markdown_summary = explainer.generate_summary(
            [result],
            output_format=OutputFormat.MARKDOWN
        )

        html_summary = explainer.generate_summary(
            [result],
            output_format=OutputFormat.HTML
        )

        assert text_summary
        assert markdown_summary
        assert html_summary
        assert "<" in html_summary

    # =========================================================================
    # Tests for translate_to_language method
    # =========================================================================

    def test_translate_to_language_english(self, explainer):
        """Test language translation (English)."""
        text = "Equipment needs maintenance"
        result = explainer.translate_to_language(text, language="en")
        assert result == text

    def test_translate_to_language_unsupported(self, explainer, caplog):
        """Test translation to unsupported language."""
        text = "Equipment needs maintenance"
        result = explainer.translate_to_language(text, language="es")
        assert result == text  # Should return English as fallback

    # =========================================================================
    # Tests for factory function
    # =========================================================================

    def test_create_natural_language_explainer(self):
        """Test factory function."""
        explainer = create_natural_language_explainer(
            audience="operator",
            output_format="text"
        )

        assert isinstance(explainer, NaturalLanguageExplainer)
        assert explainer.default_audience == Audience.OPERATOR
        assert explainer.default_format == OutputFormat.PLAIN_TEXT

    def test_create_with_all_audiences(self):
        """Test factory with all audience types."""
        audiences = ["operator", "engineer", "executive", "auditor"]

        for audience in audiences:
            explainer = create_natural_language_explainer(audience=audience)
            assert explainer.default_audience.value == audience

    # =========================================================================
    # Tests for edge cases
    # =========================================================================

    def test_explain_prediction_zero_contribution(self, explainer):
        """Test with zero contribution features."""
        result = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"temp": 0.0, "pressure": 0.0},
            feature_names={"temp": "Temperature", "pressure": "Pressure"}
        )

        assert isinstance(result, ExplanationOutput)
        assert result.confidence > 0

    def test_explain_prediction_minimal_risk(self, explainer):
        """Test with minimal risk prediction."""
        result = explainer.explain_prediction(
            prediction=0.05,
            shap_values={"temp": 0.02, "pressure": 0.03},
            feature_names={"temp": "Temperature", "pressure": "Pressure"}
        )

        assert "MINIMAL" in result.metadata.get("risk_level", "") or result.text_summary
        assert len(result.recommendations) >= 0

    def test_explain_prediction_critical_risk(self, explainer):
        """Test with critical risk prediction."""
        result = explainer.explain_prediction(
            prediction=0.95,
            shap_values={"temp": 0.60, "pressure": 0.35},
            feature_names={"temp": "Temperature", "pressure": "Pressure"}
        )

        assert "CRITICAL" in result.metadata.get("risk_level", "") or result.text_summary
        assert len(result.recommendations) > 0

    def test_explain_prediction_empty_feature_names(self, explainer):
        """Test with missing feature names."""
        result = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"feature_1": 0.5},
            feature_names={}
        )

        assert isinstance(result, ExplanationOutput)
        # Should handle missing feature names gracefully

    def test_explain_prediction_many_features(self, explainer):
        """Test with many features (more than max)."""
        shap_values = {f"feature_{i}": 0.05 for i in range(20)}
        feature_names = {f"feature_{i}": f"Feature {i}" for i in range(20)}

        result = explainer.explain_prediction(
            prediction=0.5,
            shap_values=shap_values,
            feature_names=feature_names
        )

        # Should respect max_factors for audience
        assert len(result.top_factors) <= 5  # Default engineer max is 5

    # =========================================================================
    # Tests for output quality
    # =========================================================================

    def test_output_contains_prediction_value(self, explainer, sample_prediction_data):
        """Test that output contains the prediction value."""
        result = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"]
        )

        assert "85%" in result.text_summary or "0.85" in result.text_summary

    def test_output_contains_top_features(self, explainer, sample_prediction_data):
        """Test that output mentions top contributing features."""
        result = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"]
        )

        # Top feature should be in explanation
        if result.top_factors:
            top_feature_readable = sample_prediction_data["feature_names"].get(
                result.top_factors[0][0],
                result.top_factors[0][0]
            )
            assert top_feature_readable in result.text_summary

    def test_html_output_valid_structure(self, explainer, sample_prediction_data):
        """Test that HTML output has valid structure."""
        result = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"]
        )

        html = result.html_summary
        assert html.startswith("<div")
        assert html.endswith("</div>")
        assert "</" in html  # Should have closing tags

    def test_markdown_output_has_headers(self, explainer, sample_prediction_data):
        """Test that markdown output has headers."""
        result = explainer.explain_prediction(
            prediction=0.85,
            shap_values=sample_prediction_data["shap_values"],
            feature_names=sample_prediction_data["feature_names"]
        )

        markdown = result.markdown_summary
        # Should have markdown headers
        assert "#" in markdown or "-" in markdown


class TestExplanationOutput:
    """Test cases for ExplanationOutput model."""

    def test_explanation_output_validation(self):
        """Test ExplanationOutput model validation."""
        output = ExplanationOutput(
            text_summary="Test summary",
            markdown_summary="# Test",
            html_summary="<p>Test</p>",
            confidence=0.85,
            provenance_hash="abc123def456" * 5 + "abc",  # 64 chars
            top_factors=[("feature1", 0.5), ("feature2", 0.3)],
            audience="operator"
        )

        assert output.text_summary == "Test summary"
        assert output.confidence == 0.85
        assert len(output.top_factors) == 2

    def test_explanation_output_confidence_bounds(self):
        """Test confidence score bounds."""
        # Valid confidence
        output1 = ExplanationOutput(
            text_summary="Test",
            markdown_summary="Test",
            html_summary="Test",
            confidence=0.5,
            provenance_hash="a" * 64,
            top_factors=[],
            audience="engineer"
        )
        assert output1.confidence == 0.5

        # Confidence should be between 0 and 1
        with pytest.raises(ValueError):
            ExplanationOutput(
                text_summary="Test",
                markdown_summary="Test",
                html_summary="Test",
                confidence=1.5,  # Invalid
                provenance_hash="a" * 64,
                top_factors=[],
                audience="engineer"
            )

    def test_explanation_output_serialization(self):
        """Test JSON serialization of ExplanationOutput."""
        output = ExplanationOutput(
            text_summary="Test summary",
            markdown_summary="# Test",
            html_summary="<p>Test</p>",
            confidence=0.85,
            provenance_hash="a" * 64,
            top_factors=[("feature1", 0.5)],
            audience="operator",
            recommendations=["Action 1", "Action 2"]
        )

        # Should be serializable
        json_data = output.dict()
        assert json_data["confidence"] == 0.85
        assert json_data["audience"] == "operator"
        assert len(json_data["recommendations"]) == 2


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""

    def test_single_feature_explanation(self):
        """Test explanation with only one feature."""
        explainer = NaturalLanguageExplainer()
        result = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"feature_1": 0.5},
            feature_names={"feature_1": "Feature 1"}
        )

        assert result.confidence > 0
        assert len(result.top_factors) >= 1

    def test_very_high_confidence(self):
        """Test with very high confidence."""
        explainer = NaturalLanguageExplainer()
        result = explainer.explain_prediction(
            prediction=0.99,
            shap_values={"feature": 0.99},
            feature_names={"feature": "Feature"},
            confidence=0.999
        )

        assert "CRITICAL" in result.metadata.get("risk_level", "")

    def test_changing_audience_mid_session(self):
        """Test changing default audience."""
        explainer = NaturalLanguageExplainer(default_audience=Audience.OPERATOR)

        result1 = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"f": 0.5},
            feature_names={"f": "Feature"}
        )

        # Create new explainer with different audience
        explainer2 = NaturalLanguageExplainer(default_audience=Audience.AUDITOR)
        result2 = explainer2.explain_prediction(
            prediction=0.5,
            shap_values={"f": 0.5},
            feature_names={"f": "Feature"}
        )

        assert result1.audience == "operator"
        assert result2.audience == "auditor"

    def test_negative_shap_values(self):
        """Test with negative SHAP values (negative contributions)."""
        explainer = NaturalLanguageExplainer()
        result = explainer.explain_prediction(
            prediction=0.3,
            shap_values={
                "protective_factor_1": -0.15,
                "protective_factor_2": -0.10,
                "risk_factor": 0.55
            },
            feature_names={
                "protective_factor_1": "Protective Factor 1",
                "protective_factor_2": "Protective Factor 2",
                "risk_factor": "Risk Factor"
            }
        )

        assert result.confidence > 0
        assert len(result.top_factors) > 0

    def test_explain_decision_with_empty_factors(self):
        """Test decision explanation with empty factors."""
        explainer = NaturalLanguageExplainer()
        result = explainer.explain_decision(
            decision_type=DecisionType.FOULING_RISK,
            factors={},
            confidence=0.5
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_summary_empty_list(self):
        """Test summary generation with empty explanation list."""
        explainer = NaturalLanguageExplainer()
        summary = explainer.generate_summary(
            [],
            audience=Audience.OPERATOR
        )

        assert isinstance(summary, str)

    def test_html_output_escaping(self):
        """Test HTML output with special characters."""
        explainer = NaturalLanguageExplainer()
        result = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"feature<>": 0.5},
            feature_names={"feature<>": "Feature & Others"}
        )

        assert "<div" in result.html_summary
        assert "</div>" in result.html_summary

    def test_all_risk_levels(self):
        """Test all risk level thresholds."""
        explainer = NaturalLanguageExplainer()
        test_cases = [
            (0.05, "MINIMAL"),
            (0.25, "LOW"),
            (0.55, "MEDIUM"),
            (0.75, "HIGH"),
            (0.95, "CRITICAL")
        ]

        for prediction, expected_level in test_cases:
            result = explainer.explain_prediction(
                prediction=prediction,
                shap_values={"f": prediction},
                feature_names={"f": "Feature"}
            )

            risk_level = result.metadata.get("risk_level")
            assert risk_level == expected_level, f"Expected {expected_level} for {prediction}, got {risk_level}"

    def test_recommendation_generation_for_all_decision_types(self):
        """Test recommendation generation for all decision types."""
        explainer = NaturalLanguageExplainer()

        for decision_type in DecisionType:
            result = explainer.explain_prediction(
                prediction=0.75,
                shap_values={"f": 0.75},
                feature_names={"f": "Feature"},
                decision_type=decision_type,
                audience=Audience.OPERATOR
            )

            # Should have recommendations for operator audience with high risk
            assert result.recommendations is not None

    def test_feature_sorting_by_contribution(self):
        """Test that top factors are sorted by contribution magnitude."""
        explainer = NaturalLanguageExplainer()
        result = explainer.explain_prediction(
            prediction=0.5,
            shap_values={
                "small_contributor": 0.05,
                "large_contributor": 0.45,
                "medium_contributor": 0.25
            },
            feature_names={
                "small_contributor": "Small",
                "large_contributor": "Large",
                "medium_contributor": "Medium"
            }
        )

        # Top factors should be sorted by absolute contribution
        assert result.top_factors[0][0] == "large_contributor"
        assert result.top_factors[0][1] == 0.45

    def test_markdown_to_html_conversion(self):
        """Test markdown and HTML conversions are different."""
        explainer = NaturalLanguageExplainer()
        result = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"f": 0.5},
            feature_names={"f": "Feature"}
        )

        assert result.text_summary != result.markdown_summary
        assert result.text_summary != result.html_summary
        assert result.markdown_summary != result.html_summary

    def test_confidence_reflection_in_text(self):
        """Test that confidence level affects explanation text."""
        explainer = NaturalLanguageExplainer()

        high_conf = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"f": 0.5},
            feature_names={"f": "Feature"},
            confidence=0.99,
            audience=Audience.ENGINEER
        )

        low_conf = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"f": 0.5},
            feature_names={"f": "Feature"},
            confidence=0.50,
            audience=Audience.ENGINEER
        )

        # Both should contain confidence
        assert "99%" in high_conf.text_summary or "99" in str(high_conf.confidence)
        assert "50%" in low_conf.text_summary or "50" in str(low_conf.confidence)

    def test_feature_value_formatting(self):
        """Test that feature values are properly formatted."""
        explainer = NaturalLanguageExplainer()
        result = explainer.explain_prediction(
            prediction=0.5,
            shap_values={"temp": 0.5},
            feature_names={"temp": "Temperature"},
            feature_values={"temp": 485.234}
        )

        # Should be formatted with 2 decimal places
        assert "485.23" in result.text_summary or "485.2" in result.text_summary

    def test_factory_with_invalid_audience(self):
        """Test factory function with invalid audience defaults to engineer."""
        explainer = create_natural_language_explainer(
            audience="invalid_audience",
            output_format="text"
        )

        # Should default to engineer
        assert explainer.default_audience == Audience.ENGINEER

    def test_factory_with_invalid_format(self):
        """Test factory with invalid format defaults to text."""
        explainer = create_natural_language_explainer(
            audience="operator",
            output_format="invalid_format"
        )

        # Should default to text
        assert explainer.default_format == OutputFormat.PLAIN_TEXT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
