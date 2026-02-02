# -*- coding: utf-8 -*-
"""
Unit tests for DiagnosticExplainer.

Tests SHAP-compatible explanations, counterfactuals, and evidence chains.

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
from datetime import datetime, timezone
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from explainability.diagnostic_explainer import (
    DiagnosticExplainer,
    ExplainerConfig,
    ExplanationResult,
    FeatureContribution,
    SummaryLevel,
)


@dataclass
class MockClassificationResult:
    """Mock classification result for testing."""
    trap_id: str
    condition: str
    confidence: float
    feature_importance: dict = None
    acoustic_score: float = 0.5
    thermal_score: float = 0.5
    context_score: float = 0.5

    def __post_init__(self):
        if self.feature_importance is None:
            self.feature_importance = {
                "acoustic_amplitude": 0.35,
                "thermal_differential": 0.30,
                "frequency_pattern": 0.20,
                "pressure": 0.10,
                "age": 0.05,
            }


class TestDiagnosticExplainer:
    """Tests for DiagnosticExplainer class."""

    @pytest.fixture
    def explainer(self):
        """Create default explainer."""
        return DiagnosticExplainer()

    @pytest.fixture
    def healthy_result(self):
        """Create mock healthy classification."""
        return MockClassificationResult(
            trap_id="ST-001",
            condition="healthy",
            confidence=0.92,
        )

    @pytest.fixture
    def failed_result(self):
        """Create mock failed classification."""
        return MockClassificationResult(
            trap_id="ST-002",
            condition="failed",
            confidence=0.88,
            feature_importance={
                "acoustic_amplitude": 0.45,
                "thermal_differential": 0.35,
                "frequency_pattern": 0.15,
                "pressure": 0.05,
            },
        )

    def test_explainer_initialization(self, explainer):
        """Test explainer initializes correctly."""
        assert explainer is not None
        assert explainer.config is not None

    def test_explain_healthy_trap(self, explainer, healthy_result):
        """Test explanation for healthy trap."""
        explanation = explainer.explain(healthy_result)

        assert explanation is not None
        assert explanation.trap_id == "ST-001"
        assert explanation.condition == "healthy"

    def test_explain_failed_trap(self, explainer, failed_result):
        """Test explanation for failed trap."""
        explanation = explainer.explain(failed_result)

        assert explanation is not None
        assert explanation.trap_id == "ST-002"
        assert explanation.condition == "failed"

    def test_feature_contributions(self, explainer, failed_result):
        """Test feature contributions are calculated."""
        explanation = explainer.explain(failed_result)

        assert explanation.feature_contributions is not None
        assert len(explanation.feature_contributions) > 0

        # Check contribution structure
        for contrib in explanation.feature_contributions:
            assert hasattr(contrib, 'feature_name')
            assert hasattr(contrib, 'contribution')

    def test_feature_contributions_sum(self, explainer, failed_result):
        """Test feature contributions sum to approximately 1."""
        explanation = explainer.explain(failed_result)

        total = sum(abs(c.contribution) for c in explanation.feature_contributions)
        # Allow some tolerance for normalization
        assert 0.9 <= total <= 1.1

    def test_counterfactual_explanation(self, explainer, failed_result):
        """Test counterfactual explanations."""
        explanation = explainer.explain(failed_result)

        assert explanation.counterfactuals is not None
        if len(explanation.counterfactuals) > 0:
            cf = explanation.counterfactuals[0]
            assert hasattr(cf, 'target_condition')
            assert hasattr(cf, 'changes_required')

    def test_evidence_chain(self, explainer, failed_result):
        """Test evidence chain is provided."""
        explanation = explainer.explain(failed_result)

        assert explanation.evidence_chain is not None
        assert len(explanation.evidence_chain) > 0

    def test_summary_technical(self, explainer, failed_result):
        """Test technical summary generation."""
        explanation = explainer.explain(failed_result)
        summary = explanation.get_summary(SummaryLevel.TECHNICAL)

        assert summary is not None
        assert len(summary) > 0

    def test_summary_operator(self, explainer, failed_result):
        """Test operator summary generation."""
        explanation = explainer.explain(failed_result)
        summary = explanation.get_summary(SummaryLevel.OPERATOR)

        assert summary is not None
        # Operator summary should be more accessible
        assert "ST-002" in summary or "failed" in summary.lower()

    def test_summary_executive(self, explainer, failed_result):
        """Test executive summary generation."""
        explanation = explainer.explain(failed_result)
        summary = explanation.get_summary(SummaryLevel.EXECUTIVE)

        assert summary is not None
        # Executive summary should be concise
        assert len(summary) < 500

    def test_explanation_to_dict(self, explainer, failed_result):
        """Test explanation serialization."""
        explanation = explainer.explain(failed_result)
        exp_dict = explanation.to_dict()

        assert "trap_id" in exp_dict
        assert "condition" in exp_dict
        assert "feature_contributions" in exp_dict

    def test_provenance_hash(self, explainer, failed_result):
        """Test provenance hash is included."""
        explanation = explainer.explain(failed_result)

        assert explanation.provenance_hash is not None
        assert len(explanation.provenance_hash) == 16

    def test_deterministic_explanation(self, explainer, healthy_result):
        """Test that same input produces same output."""
        exp1 = explainer.explain(healthy_result)
        exp2 = explainer.explain(healthy_result)

        # Feature contributions should be identical
        assert len(exp1.feature_contributions) == len(exp2.feature_contributions)
        for c1, c2 in zip(exp1.feature_contributions, exp2.feature_contributions):
            assert c1.feature_name == c2.feature_name
            assert abs(c1.contribution - c2.contribution) < 0.001

    def test_custom_config(self):
        """Test explainer with custom configuration."""
        config = ExplainerConfig(
            include_counterfactuals=False,
            max_features=3,
        )
        explainer = DiagnosticExplainer(config)

        assert explainer.config.include_counterfactuals == False
        assert explainer.config.max_features == 3

    def test_confidence_in_explanation(self, explainer, healthy_result):
        """Test confidence is included in explanation."""
        explanation = explainer.explain(healthy_result)

        assert explanation.confidence is not None
        assert 0.0 <= explanation.confidence <= 1.0


class TestFeatureContribution:
    """Tests for FeatureContribution class."""

    def test_contribution_creation(self):
        """Test feature contribution creation."""
        contrib = FeatureContribution(
            feature_name="acoustic_amplitude",
            contribution=0.35,
            feature_value=85.0,
            baseline_value=50.0,
            description="High acoustic amplitude indicates steam leakage",
        )

        assert contrib.feature_name == "acoustic_amplitude"
        assert contrib.contribution == 0.35
        assert contrib.feature_value == 85.0

    def test_contribution_direction(self):
        """Test contribution direction interpretation."""
        positive_contrib = FeatureContribution(
            feature_name="acoustic",
            contribution=0.3,
        )

        negative_contrib = FeatureContribution(
            feature_name="thermal",
            contribution=-0.2,
        )

        assert positive_contrib.contribution > 0
        assert negative_contrib.contribution < 0


class TestExplainerConfig:
    """Tests for ExplainerConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExplainerConfig()

        assert config.include_counterfactuals == True
        assert config.max_features > 0
        assert config.include_evidence_chain == True

    def test_config_validation(self):
        """Test configuration validation."""
        config = ExplainerConfig(max_features=5)

        assert config.max_features == 5
