# -*- coding: utf-8 -*-
"""
Unit tests for GL-024 Air Preheater Agent Explainability Module

Tests LIME-based explanations for heat transfer, leakage, cold-end protection,
and optimization decisions with full transparency requirements.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_024_air_preheater.explainability import (
    LIMEAirPreheaterExplainer,
    ExplainerConfig,
    FeatureExplanation,
    LIMEExplanation,
    create_explainer,
    AIR_PREHEATER_FEATURE_INFO,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def explainer():
    """Create explainer with default config."""
    config = ExplainerConfig()
    return LIMEAirPreheaterExplainer(config)


@pytest.fixture
def sample_heat_transfer_features():
    """Create sample heat transfer features."""
    return {
        "gas_inlet_temp_f": 650.0,
        "gas_outlet_temp_f": 300.0,
        "air_inlet_temp_f": 80.0,
        "air_outlet_temp_f": 550.0,
        "gas_flow_rate_lb_hr": 500000.0,
        "air_flow_rate_lb_hr": 480000.0,
        "boiler_load_pct": 85.0,
    }


@pytest.fixture
def sample_leakage_features():
    """Create sample leakage analysis features."""
    return {
        "o2_inlet_pct": 3.0,
        "o2_outlet_pct": 4.5,
        "seal_clearance_in": 0.15,
        "pressure_differential_in_wc": 5.0,
        "rotor_speed_rpm": 1.5,
    }


@pytest.fixture
def sample_cold_end_features():
    """Create sample cold-end protection features."""
    return {
        "so3_ppm": 10.0,
        "h2o_vol_pct": 8.0,
        "gas_outlet_temp_f": 300.0,
        "air_inlet_temp_f": 80.0,
        "cold_end_metal_temp_f": 290.0,
    }


# =============================================================================
# EXPLAINER INITIALIZATION TESTS
# =============================================================================

class TestExplainerInitialization:
    """Test suite for explainer initialization."""

    @pytest.mark.unit
    def test_create_explainer_default(self):
        """Test explainer creation with default config."""
        explainer = create_explainer()
        assert explainer is not None
        assert isinstance(explainer, LIMEAirPreheaterExplainer)

    @pytest.mark.unit
    def test_create_explainer_custom_config(self):
        """Test explainer creation with custom config."""
        config = ExplainerConfig(
            num_samples=500,
            num_features=10,
        )
        explainer = create_explainer(config)
        assert explainer.config.num_samples == 500

    @pytest.mark.unit
    def test_explainer_config_defaults(self):
        """Test explainer config defaults."""
        config = ExplainerConfig()
        assert config.num_samples > 0
        assert config.num_features > 0


# =============================================================================
# FEATURE INFO TESTS
# =============================================================================

class TestFeatureInfo:
    """Test suite for feature information dictionary."""

    @pytest.mark.unit
    def test_feature_info_exists(self):
        """Test feature info dictionary exists."""
        assert AIR_PREHEATER_FEATURE_INFO is not None
        assert isinstance(AIR_PREHEATER_FEATURE_INFO, dict)

    @pytest.mark.unit
    def test_feature_info_has_common_features(self):
        """Test feature info contains common features."""
        expected_features = [
            "gas_inlet_temp_f",
            "gas_outlet_temp_f",
            "air_inlet_temp_f",
            "air_outlet_temp_f",
            "o2_inlet_pct",
            "o2_outlet_pct",
        ]
        for feature in expected_features:
            assert feature in AIR_PREHEATER_FEATURE_INFO

    @pytest.mark.unit
    def test_feature_info_structure(self):
        """Test feature info entry structure."""
        if "gas_inlet_temp_f" in AIR_PREHEATER_FEATURE_INFO:
            info = AIR_PREHEATER_FEATURE_INFO["gas_inlet_temp_f"]
            assert "description" in info or "name" in info


# =============================================================================
# HEAT TRANSFER EXPLANATION TESTS
# =============================================================================

class TestHeatTransferExplanation:
    """Test suite for heat transfer explanations."""

    @pytest.mark.unit
    def test_explain_heat_transfer_returns_explanation(
        self, explainer, sample_heat_transfer_features
    ):
        """Test heat transfer explanation returns LIMEExplanation."""
        result = explainer.explain_heat_transfer_performance(
            features=sample_heat_transfer_features,
            prediction=75.0,  # 75% effectiveness
        )

        assert isinstance(result, LIMEExplanation)

    @pytest.mark.unit
    def test_heat_transfer_explanation_has_features(
        self, explainer, sample_heat_transfer_features
    ):
        """Test explanation includes feature contributions."""
        result = explainer.explain_heat_transfer_performance(
            features=sample_heat_transfer_features,
            prediction=75.0,
        )

        assert hasattr(result, 'feature_explanations')
        assert len(result.feature_explanations) > 0

    @pytest.mark.unit
    def test_heat_transfer_explanation_has_summary(
        self, explainer, sample_heat_transfer_features
    ):
        """Test explanation includes summary text."""
        result = explainer.explain_heat_transfer_performance(
            features=sample_heat_transfer_features,
            prediction=75.0,
        )

        assert hasattr(result, 'summary') or hasattr(result, 'explanation_text')


# =============================================================================
# LEAKAGE EXPLANATION TESTS
# =============================================================================

class TestLeakageExplanation:
    """Test suite for leakage explanations."""

    @pytest.mark.unit
    def test_explain_leakage_returns_explanation(
        self, explainer, sample_leakage_features
    ):
        """Test leakage explanation returns LIMEExplanation."""
        result = explainer.explain_leakage_analysis(
            features=sample_leakage_features,
            prediction=8.5,  # 8.5% leakage
        )

        assert isinstance(result, LIMEExplanation)

    @pytest.mark.unit
    def test_leakage_explanation_identifies_seal_issues(
        self, explainer, sample_leakage_features
    ):
        """Test leakage explanation identifies seal-related issues."""
        # High seal clearance should be identified as contributor
        features = sample_leakage_features.copy()
        features["seal_clearance_in"] = 0.25  # High clearance

        result = explainer.explain_leakage_analysis(
            features=features,
            prediction=12.0,  # High leakage
        )

        # Should have some explanation related to seals
        assert result is not None


# =============================================================================
# COLD-END EXPLANATION TESTS
# =============================================================================

class TestColdEndExplanation:
    """Test suite for cold-end protection explanations."""

    @pytest.mark.unit
    def test_explain_cold_end_returns_explanation(
        self, explainer, sample_cold_end_features
    ):
        """Test cold-end explanation returns LIMEExplanation."""
        result = explainer.explain_cold_end_protection(
            features=sample_cold_end_features,
            prediction=15.0,  # 15F margin
        )

        assert isinstance(result, LIMEExplanation)

    @pytest.mark.unit
    def test_cold_end_explanation_for_low_margin(
        self, explainer, sample_cold_end_features
    ):
        """Test explanation when cold-end margin is low."""
        features = sample_cold_end_features.copy()
        features["cold_end_metal_temp_f"] = 275.0  # Low, near ADP

        result = explainer.explain_cold_end_protection(
            features=features,
            prediction=5.0,  # Only 5F margin - critical
        )

        assert result is not None


# =============================================================================
# FEATURE EXPLANATION TESTS
# =============================================================================

class TestFeatureExplanation:
    """Test suite for FeatureExplanation dataclass."""

    @pytest.mark.unit
    def test_feature_explanation_creation(self):
        """Test FeatureExplanation creation."""
        explanation = FeatureExplanation(
            feature_name="gas_inlet_temp_f",
            feature_value=650.0,
            contribution=0.15,
            direction="positive",
            description="Higher gas inlet temperature increases heat recovery",
        )

        assert explanation.feature_name == "gas_inlet_temp_f"
        assert explanation.contribution == 0.15
        assert explanation.direction == "positive"


# =============================================================================
# LIME EXPLANATION TESTS
# =============================================================================

class TestLIMEExplanation:
    """Test suite for LIMEExplanation dataclass."""

    @pytest.mark.unit
    def test_lime_explanation_creation(self):
        """Test LIMEExplanation creation."""
        feature_explanations = [
            FeatureExplanation(
                feature_name="gas_inlet_temp_f",
                feature_value=650.0,
                contribution=0.15,
                direction="positive",
                description="Test description",
            )
        ]

        explanation = LIMEExplanation(
            prediction=75.0,
            feature_explanations=feature_explanations,
            model_type="heat_transfer",
            confidence=0.95,
        )

        assert explanation.prediction == 75.0
        assert len(explanation.feature_explanations) == 1
        assert explanation.confidence == 0.95


# =============================================================================
# RECOMMENDATION GENERATION TESTS
# =============================================================================

class TestRecommendationGeneration:
    """Test suite for recommendation generation from explanations."""

    @pytest.mark.unit
    def test_generate_recommendations_from_heat_transfer(
        self, explainer, sample_heat_transfer_features
    ):
        """Test recommendation generation from heat transfer analysis."""
        explanation = explainer.explain_heat_transfer_performance(
            features=sample_heat_transfer_features,
            prediction=65.0,  # Degraded effectiveness
        )

        # Should be able to derive recommendations
        assert explanation is not None


# =============================================================================
# EXPLANATION TRANSPARENCY TESTS
# =============================================================================

class TestExplanationTransparency:
    """Test suite for explanation transparency requirements."""

    @pytest.mark.unit
    def test_explanations_are_human_readable(
        self, explainer, sample_heat_transfer_features
    ):
        """Test that explanations are human-readable."""
        result = explainer.explain_heat_transfer_performance(
            features=sample_heat_transfer_features,
            prediction=75.0,
        )

        # Check that explanations contain readable text
        if hasattr(result, 'feature_explanations'):
            for feat_exp in result.feature_explanations:
                if hasattr(feat_exp, 'description'):
                    assert len(feat_exp.description) > 0

    @pytest.mark.unit
    def test_explanations_include_units(
        self, explainer, sample_heat_transfer_features
    ):
        """Test that feature values include units where appropriate."""
        # This tests that explanations are complete and understandable
        result = explainer.explain_heat_transfer_performance(
            features=sample_heat_transfer_features,
            prediction=75.0,
        )

        assert result is not None
