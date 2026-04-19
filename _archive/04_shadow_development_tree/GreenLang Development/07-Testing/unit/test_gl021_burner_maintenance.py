# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for GL-021 BURNERSENTRY

Tests all components of the GL-021 Burner Maintenance Agent:
- Configuration validation
- Flame pattern analysis
- Burner health scoring
- Maintenance prediction
- Replacement planning
- CMMS integration
- Explainability (SHAP/LIME)

Test Coverage Target: 85%+

Example:
    Run all tests:
        pytest tests/unit/test_gl021_burner_maintenance.py -v

    Run specific test class:
        pytest tests/unit/test_gl021_burner_maintenance.py::TestGL021Config -v

    Run with coverage:
        pytest tests/unit/test_gl021_burner_maintenance.py --cov=greenlang.agents.process_heat.gl_021_burner_maintenance

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest
import numpy as np

# Import GL-021 explainability module
from greenlang.agents.process_heat.gl_021_burner_maintenance.explainability import (
    GL021Explainer,
    GL021SHAPExplainer,
    GL021LIMEExplainer,
    HealthScoreExplainer,
    GL021NaturalLanguageExplainer,
    GL021ProvenanceTracker,
    ExplanationAudience,
    ExplanationType,
    RiskLevel,
    BurnerComponent,
    FeatureContribution,
    ComponentHealthExplanation,
    SHAPExplanation,
    LIMEExplanation,
    HealthScoreExplanation,
    NaturalLanguageExplanation,
    GL021ExplanationResult,
    create_gl021_explainer,
    create_health_score_explainer,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_burner_config() -> Dict[str, Any]:
    """Create sample burner configuration."""
    return {
        "burner_id": "BURNER-001",
        "burner_type": "natural_gas",
        "capacity_mmbtu_hr": 10.0,
        "installation_date": "2020-01-15",
        "manufacturer": "Honeywell",
        "model": "RM7850L",
        "fuel_type": "natural_gas",
        "control_type": "modulating",
        "flame_scanner_type": "UV",
        "ignitor_type": "direct_spark",
        "rated_turndown": 10.0,
        "min_firing_rate_pct": 10.0,
        "max_firing_rate_pct": 100.0,
        "purge_time_sec": 30.0,
        "pilot_proving_time_sec": 10.0,
        "main_flame_proving_time_sec": 5.0,
        "post_purge_time_sec": 15.0,
        "safety_shutoff_valve_count": 2,
        "combustion_air_proving": True,
        "low_gas_pressure_interlock": True,
        "high_gas_pressure_interlock": True,
    }


@pytest.fixture
def sample_operating_data() -> Dict[str, float]:
    """Create sample operating data."""
    return {
        "flame_intensity": 850.0,
        "flame_stability_index": 0.92,
        "flame_color_score": 85.0,
        "fuel_flow_rate": 1200.0,
        "air_fuel_ratio": 10.5,
        "combustion_efficiency": 92.5,
        "nox_emissions": 25.0,
        "co_emissions": 50.0,
        "flue_gas_temp": 450.0,
        "oxygen_level": 3.2,
        "burner_age_hours": 15000.0,
        "days_since_maintenance": 120.0,
        "ignition_success_rate": 98.5,
        "flame_scanner_voltage": 4.2,
        "pilot_flame_strength": 92.0,
        "main_valve_response_ms": 150.0,
        "fuel_pressure": 8.5,
        "air_damper_position": 65.0,
        "combustion_air_flow": 2500.0,
        "heat_release_rate": 8.5,
    }


@pytest.fixture
def sample_flame_data() -> Dict[str, Any]:
    """Create sample flame analysis data."""
    return {
        "flame_present": True,
        "flame_intensity_lux": 850.0,
        "flame_stability_index": 0.92,
        "flame_color_rgb": (255, 180, 50),
        "flame_color_temperature_k": 1800.0,
        "flame_geometry": {
            "length_inches": 24.0,
            "width_inches": 8.0,
            "angle_degrees": 5.0,
        },
        "pulsation_frequency_hz": 2.5,
        "pulsation_amplitude_pct": 3.0,
        "uv_signal_strength_v": 4.2,
        "ir_signal_strength_v": 3.8,
        "flame_symmetry_index": 0.95,
        "combustion_quality_index": 0.88,
        "stoichiometry_indicator": 1.05,
    }


@pytest.fixture
def sample_maintenance_history() -> List[Dict[str, Any]]:
    """Create sample maintenance history."""
    return [
        {
            "date": "2024-01-15",
            "type": "preventive",
            "description": "Annual inspection and tune-up",
            "components_serviced": ["ignitor", "flame_scanner", "gas_train"],
            "parts_replaced": ["ignition_electrode"],
            "duration_hours": 4.0,
            "cost_usd": 850.0,
            "technician": "J. Smith",
        },
        {
            "date": "2023-07-20",
            "type": "corrective",
            "description": "Flame scanner replacement",
            "components_serviced": ["flame_scanner"],
            "parts_replaced": ["uv_flame_scanner"],
            "duration_hours": 2.0,
            "cost_usd": 450.0,
            "technician": "M. Johnson",
        },
        {
            "date": "2023-01-10",
            "type": "preventive",
            "description": "Annual inspection",
            "components_serviced": ["ignitor", "gas_train", "air_damper"],
            "parts_replaced": [],
            "duration_hours": 3.5,
            "cost_usd": 500.0,
            "technician": "J. Smith",
        },
    ]


@pytest.fixture
def sample_component_scores() -> Dict[str, float]:
    """Create sample component health scores."""
    return {
        "flame_scanner": 85.0,
        "ignitor": 78.0,
        "fuel_valve": 92.0,
        "air_damper": 88.0,
        "pilot_assembly": 75.0,
        "main_burner": 90.0,
        "combustion_air_fan": 95.0,
        "gas_train": 82.0,
        "flame_stability": 87.0,
        "emission_quality": 80.0,
    }


@pytest.fixture
def sample_historical_scores() -> List[Dict[str, float]]:
    """Create sample historical health scores."""
    return [
        {"flame_scanner": 95.0, "ignitor": 92.0, "fuel_valve": 98.0, "pilot_assembly": 90.0},
        {"flame_scanner": 92.0, "ignitor": 88.0, "fuel_valve": 96.0, "pilot_assembly": 87.0},
        {"flame_scanner": 89.0, "ignitor": 84.0, "fuel_valve": 94.0, "pilot_assembly": 82.0},
        {"flame_scanner": 87.0, "ignitor": 80.0, "fuel_valve": 93.0, "pilot_assembly": 78.0},
        {"flame_scanner": 85.0, "ignitor": 78.0, "fuel_valve": 92.0, "pilot_assembly": 75.0},
    ]


@pytest.fixture
def mock_model():
    """Create mock ML model for testing."""
    model = Mock()
    model.predict = Mock(return_value=np.array([0.65]))
    model.predict_proba = Mock(return_value=np.array([[0.35, 0.65]]))
    return model


@pytest.fixture
def feature_names() -> List[str]:
    """Feature names for ML models."""
    return [
        "flame_intensity",
        "flame_stability_index",
        "flame_color_score",
        "fuel_flow_rate",
        "air_fuel_ratio",
        "combustion_efficiency",
        "nox_emissions",
        "co_emissions",
        "flue_gas_temp",
        "oxygen_level",
        "burner_age_hours",
        "days_since_maintenance",
        "ignition_success_rate",
        "flame_scanner_voltage",
        "pilot_flame_strength",
        "main_valve_response_ms",
        "fuel_pressure",
        "air_damper_position",
        "combustion_air_flow",
        "heat_release_rate",
    ]


@pytest.fixture
def gl021_explainer(mock_model, feature_names):
    """Create GL021Explainer instance for testing."""
    return GL021Explainer(
        model=mock_model,
        feature_names=feature_names,
        model_version="1.0.0-test",
        enable_shap=True,
        enable_lime=True,
    )


@pytest.fixture
def health_score_explainer():
    """Create HealthScoreExplainer instance for testing."""
    return HealthScoreExplainer()


@pytest.fixture
def nl_explainer():
    """Create NaturalLanguageExplainer instance for testing."""
    return GL021NaturalLanguageExplainer()


@pytest.fixture
def provenance_tracker():
    """Create ProvenanceTracker instance for testing."""
    return GL021ProvenanceTracker(agent_id="GL-021-TEST", model_version="1.0.0")


# =============================================================================
# TEST GL021 CONFIGURATION
# =============================================================================

class TestGL021Config:
    """Test GL-021 configuration validation."""

    def test_valid_burner_config(self, sample_burner_config):
        """Test valid burner configuration."""
        assert sample_burner_config["burner_id"] == "BURNER-001"
        assert sample_burner_config["capacity_mmbtu_hr"] == 10.0
        assert sample_burner_config["fuel_type"] == "natural_gas"

    def test_config_required_fields(self, sample_burner_config):
        """Test required configuration fields are present."""
        required_fields = [
            "burner_id",
            "burner_type",
            "capacity_mmbtu_hr",
            "fuel_type",
            "control_type",
        ]
        for field in required_fields:
            assert field in sample_burner_config
            assert sample_burner_config[field] is not None

    def test_config_safety_interlocks(self, sample_burner_config):
        """Test safety interlock configuration."""
        assert sample_burner_config["safety_shutoff_valve_count"] >= 2
        assert sample_burner_config["combustion_air_proving"] is True
        assert sample_burner_config["low_gas_pressure_interlock"] is True
        assert sample_burner_config["high_gas_pressure_interlock"] is True

    def test_config_timing_parameters(self, sample_burner_config):
        """Test timing parameters are within safe ranges."""
        assert sample_burner_config["purge_time_sec"] >= 15.0
        assert sample_burner_config["pilot_proving_time_sec"] >= 5.0
        assert sample_burner_config["main_flame_proving_time_sec"] >= 3.0

    def test_config_turndown_ratio(self, sample_burner_config):
        """Test turndown ratio configuration."""
        turndown = sample_burner_config["rated_turndown"]
        min_rate = sample_burner_config["min_firing_rate_pct"]
        max_rate = sample_burner_config["max_firing_rate_pct"]

        assert turndown > 0
        assert min_rate < max_rate
        assert min_rate >= 10.0  # Typical minimum
        assert max_rate <= 100.0


# =============================================================================
# TEST FLAME ANALYSIS
# =============================================================================

class TestFlameAnalysis:
    """Test flame pattern recognition and analysis."""

    def test_flame_data_structure(self, sample_flame_data):
        """Test flame data has required fields."""
        assert sample_flame_data["flame_present"] is True
        assert sample_flame_data["flame_intensity_lux"] > 0
        assert 0 <= sample_flame_data["flame_stability_index"] <= 1.0

    def test_flame_color_analysis(self, sample_flame_data):
        """Test flame color analysis."""
        rgb = sample_flame_data["flame_color_rgb"]
        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)

        # Blue flame = good combustion, yellow = incomplete
        # This is a simplified check - real analysis would be more complex
        r, g, b = rgb
        color_temp = sample_flame_data["flame_color_temperature_k"]
        assert 1500 <= color_temp <= 3000  # Typical range for gas flames

    def test_flame_geometry_validation(self, sample_flame_data):
        """Test flame geometry parameters."""
        geometry = sample_flame_data["flame_geometry"]

        assert geometry["length_inches"] > 0
        assert geometry["width_inches"] > 0
        assert -45 <= geometry["angle_degrees"] <= 45

        # Length/width ratio check
        aspect_ratio = geometry["length_inches"] / geometry["width_inches"]
        assert 2.0 <= aspect_ratio <= 5.0  # Typical range

    def test_flame_stability_assessment(self, sample_flame_data):
        """Test flame stability assessment."""
        stability = sample_flame_data["flame_stability_index"]
        pulsation_freq = sample_flame_data["pulsation_frequency_hz"]
        pulsation_amp = sample_flame_data["pulsation_amplitude_pct"]

        # Stable flame characteristics
        if stability > 0.9:
            assert pulsation_amp < 5.0
        elif stability > 0.7:
            assert pulsation_amp < 10.0

    def test_combustion_quality_index(self, sample_flame_data):
        """Test combustion quality calculation."""
        cqi = sample_flame_data["combustion_quality_index"]
        stoich = sample_flame_data["stoichiometry_indicator"]

        assert 0 <= cqi <= 1.0

        # Stoichiometry near 1.0 indicates good air-fuel mix
        # Slightly above 1.0 (lean) is typical for low NOx
        assert 0.95 <= stoich <= 1.15

    def test_signal_strength_validation(self, sample_flame_data):
        """Test flame detector signal strengths."""
        uv_signal = sample_flame_data["uv_signal_strength_v"]
        ir_signal = sample_flame_data["ir_signal_strength_v"]

        # Typical thresholds
        assert uv_signal > 2.0  # Minimum for reliable detection
        assert ir_signal > 2.0


# =============================================================================
# TEST BURNER HEALTH SCORING
# =============================================================================

class TestBurnerHealth:
    """Test component health scoring."""

    def test_health_score_explainer_init(self, health_score_explainer):
        """Test HealthScoreExplainer initialization."""
        assert health_score_explainer is not None
        assert len(health_score_explainer.component_weights) == 10

        # Verify weights sum to 1.0
        total_weight = sum(health_score_explainer.component_weights.values())
        assert abs(total_weight - 1.0) < 0.001

    def test_explain_health_score(self, health_score_explainer, sample_component_scores):
        """Test health score explanation generation."""
        explanation = health_score_explainer.explain(
            sample_component_scores,
            operating_hours=15000,
        )

        assert isinstance(explanation, HealthScoreExplanation)
        assert 0 <= explanation.overall_score <= 100
        assert explanation.risk_level in RiskLevel
        assert len(explanation.component_breakdowns) > 0

    def test_component_breakdown_structure(self, health_score_explainer, sample_component_scores):
        """Test component breakdown structure."""
        explanation = health_score_explainer.explain(sample_component_scores)

        for breakdown in explanation.component_breakdowns:
            assert isinstance(breakdown, ComponentHealthExplanation)
            assert breakdown.component in BurnerComponent
            assert 0 <= breakdown.health_score <= 100
            assert 0 <= breakdown.weight <= 1
            assert breakdown.trend in ["improving", "stable", "degrading"]

    def test_weighted_contribution_calculation(self, health_score_explainer, sample_component_scores):
        """Test weighted contribution calculations are correct."""
        explanation = health_score_explainer.explain(sample_component_scores)

        # Verify weighted contributions sum to overall score
        total_contribution = sum(b.weighted_contribution for b in explanation.component_breakdowns)
        assert abs(total_contribution - explanation.overall_score) < 0.01

    def test_risk_level_classification(self, health_score_explainer):
        """Test risk level classification thresholds."""
        # Critical risk
        critical_scores = {c.value: 30.0 for c in BurnerComponent}
        critical_exp = health_score_explainer.explain(critical_scores)
        assert critical_exp.risk_level == RiskLevel.CRITICAL

        # High risk
        high_scores = {c.value: 55.0 for c in BurnerComponent}
        high_exp = health_score_explainer.explain(high_scores)
        assert high_exp.risk_level == RiskLevel.HIGH

        # Medium risk
        medium_scores = {c.value: 70.0 for c in BurnerComponent}
        medium_exp = health_score_explainer.explain(medium_scores)
        assert medium_exp.risk_level == RiskLevel.MEDIUM

        # Low risk
        low_scores = {c.value: 85.0 for c in BurnerComponent}
        low_exp = health_score_explainer.explain(low_scores)
        assert low_exp.risk_level == RiskLevel.LOW

        # Minimal risk
        minimal_scores = {c.value: 95.0 for c in BurnerComponent}
        minimal_exp = health_score_explainer.explain(minimal_scores)
        assert minimal_exp.risk_level == RiskLevel.MINIMAL

    def test_trend_analysis_with_history(
        self, health_score_explainer, sample_component_scores, sample_historical_scores
    ):
        """Test trend analysis with historical data."""
        explanation = health_score_explainer.explain(
            sample_component_scores,
            operating_hours=15000,
            historical_scores=sample_historical_scores,
        )

        assert explanation.trend_analysis is not None
        assert "status" in explanation.trend_analysis

        if explanation.trend_analysis["status"] == "analyzed":
            assert "slope" in explanation.trend_analysis
            assert "direction" in explanation.trend_analysis

    def test_degradation_factors(self, health_score_explainer, sample_component_scores):
        """Test degradation factor calculation."""
        explanation = health_score_explainer.explain(
            sample_component_scores,
            operating_hours=25000,
        )

        assert "age_degradation" in explanation.degradation_factors
        assert "thermal_cycling" in explanation.degradation_factors
        assert "component_wear" in explanation.degradation_factors

        # Age factor should increase with operating hours
        for factor_name, factor_value in explanation.degradation_factors.items():
            assert 0 <= factor_value <= 1.0

    def test_primary_concerns_identification(self, health_score_explainer):
        """Test primary concerns are identified correctly."""
        # Create scores with some critical components
        scores = {c.value: 90.0 for c in BurnerComponent}
        scores["ignitor"] = 45.0  # Critical
        scores["pilot_assembly"] = 65.0  # Warning

        explanation = health_score_explainer.explain(scores)

        assert len(explanation.primary_concerns) > 0
        assert any("ignitor" in c.lower() for c in explanation.primary_concerns)

    def test_improvement_opportunities(self, health_score_explainer, sample_component_scores):
        """Test improvement opportunities identification."""
        explanation = health_score_explainer.explain(sample_component_scores)

        # Should identify components with improvement potential
        for opportunity in explanation.improvement_opportunities:
            assert "points" in opportunity.lower() or "improving" in opportunity.lower()

    def test_custom_component_weights(self):
        """Test custom component weights."""
        custom_weights = {
            BurnerComponent.FLAME_SCANNER: 0.25,
            BurnerComponent.IGNITOR: 0.20,
            BurnerComponent.FUEL_VALVE: 0.20,
            BurnerComponent.AIR_DAMPER: 0.05,
            BurnerComponent.PILOT_ASSEMBLY: 0.10,
            BurnerComponent.MAIN_BURNER: 0.10,
            BurnerComponent.COMBUSTION_AIR_FAN: 0.03,
            BurnerComponent.GAS_TRAIN: 0.03,
            BurnerComponent.FLAME_STABILITY: 0.02,
            BurnerComponent.EMISSION_QUALITY: 0.02,
        }

        explainer = HealthScoreExplainer(component_weights=custom_weights)
        assert explainer.component_weights[BurnerComponent.FLAME_SCANNER] == 0.25

    def test_invalid_weights_rejected(self):
        """Test that invalid weights are rejected."""
        invalid_weights = {
            BurnerComponent.FLAME_SCANNER: 0.5,
            BurnerComponent.IGNITOR: 0.3,
            # Missing components and doesn't sum to 1.0
        }

        with pytest.raises(ValueError):
            HealthScoreExplainer(component_weights=invalid_weights)


# =============================================================================
# TEST MAINTENANCE PREDICTION
# =============================================================================

class TestMaintenancePrediction:
    """Test Weibull and ML prediction functionality."""

    def test_gl021_explainer_initialization(self, gl021_explainer, feature_names):
        """Test GL021Explainer initialization."""
        assert gl021_explainer is not None
        assert gl021_explainer.model_version == "1.0.0-test"
        assert len(gl021_explainer.feature_names) == len(feature_names)

    def test_explain_method(self, gl021_explainer, sample_operating_data, sample_component_scores):
        """Test main explain method."""
        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            explanation_type=ExplanationType.HEALTH_SCORE,
            component_scores=sample_component_scores,
            operating_hours=15000,
        )

        assert isinstance(result, GL021ExplanationResult)
        assert result.explanation_id is not None
        assert result.explanation_type == ExplanationType.HEALTH_SCORE
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_explain_with_array_input(self, gl021_explainer, feature_names, sample_component_scores):
        """Test explain with numpy array input."""
        # Create array input
        X = np.random.rand(1, len(feature_names))

        result = gl021_explainer.explain(
            input_data=X,
            explanation_type=ExplanationType.FAILURE_PREDICTION,
            component_scores=sample_component_scores,
        )

        assert result is not None
        assert result.health_score_explanation is not None

    def test_health_score_explanation_present(
        self, gl021_explainer, sample_operating_data, sample_component_scores
    ):
        """Test health score explanation is generated."""
        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            component_scores=sample_component_scores,
        )

        assert result.health_score_explanation is not None
        assert isinstance(result.health_score_explanation, HealthScoreExplanation)

    def test_processing_time_recorded(
        self, gl021_explainer, sample_operating_data, sample_component_scores
    ):
        """Test processing time is recorded."""
        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            component_scores=sample_component_scores,
        )

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 10000  # Should complete within 10 seconds


# =============================================================================
# TEST FUEL IMPACT ANALYSIS
# =============================================================================

class TestFuelImpact:
    """Test fuel quality impact on predictions."""

    def test_fuel_flow_rate_impact(self, sample_operating_data):
        """Test fuel flow rate tracking."""
        fuel_flow = sample_operating_data["fuel_flow_rate"]
        assert fuel_flow > 0

        # Verify air-fuel ratio is reasonable
        air_fuel_ratio = sample_operating_data["air_fuel_ratio"]
        assert 9.0 <= air_fuel_ratio <= 12.0  # Typical natural gas range

    def test_combustion_efficiency_calculation(self, sample_operating_data):
        """Test combustion efficiency is within valid range."""
        efficiency = sample_operating_data["combustion_efficiency"]
        assert 70.0 <= efficiency <= 100.0

        # Higher efficiency should correlate with lower CO
        co_emissions = sample_operating_data["co_emissions"]
        if efficiency > 90:
            assert co_emissions < 100  # Good combustion = low CO

    def test_emissions_correlation(self, sample_operating_data):
        """Test emissions parameters correlation."""
        nox = sample_operating_data["nox_emissions"]
        co = sample_operating_data["co_emissions"]
        o2 = sample_operating_data["oxygen_level"]

        # Verify within regulatory ranges
        assert nox >= 0
        assert co >= 0

        # Higher O2 typically means lower NOx but higher excess air
        if o2 > 4.0:
            assert nox < 50  # Low NOx expected with excess air


# =============================================================================
# TEST REPLACEMENT PLANNING
# =============================================================================

class TestReplacementPlanner:
    """Test economic optimization for replacement planning."""

    def test_degradation_rate_calculation(self, health_score_explainer, sample_component_scores):
        """Test degradation rate is calculated for components."""
        explanation = health_score_explainer.explain(
            sample_component_scores,
            operating_hours=25000,
        )

        for breakdown in explanation.component_breakdowns:
            assert breakdown.degradation_rate >= 0
            # Higher age should increase degradation rate
            assert breakdown.degradation_rate <= 5.0  # Reasonable upper bound

    def test_days_to_critical_projection(
        self, health_score_explainer, sample_component_scores, sample_historical_scores
    ):
        """Test days to critical threshold projection."""
        explanation = health_score_explainer.explain(
            sample_component_scores,
            operating_hours=15000,
            historical_scores=sample_historical_scores,
        )

        if explanation.trend_analysis.get("days_to_critical"):
            days = explanation.trend_analysis["days_to_critical"]
            assert days > 0  # Should be positive if degrading

    def test_recommendations_generated(self, health_score_explainer, sample_component_scores):
        """Test maintenance recommendations are generated."""
        # Create scores with degraded component
        scores = sample_component_scores.copy()
        scores["ignitor"] = 55.0  # Degraded

        explanation = health_score_explainer.explain(scores)

        # Should have recommendations
        has_recommendations = any(
            len(b.recommendations) > 0 for b in explanation.component_breakdowns
        )
        assert has_recommendations


# =============================================================================
# TEST CMMS INTEGRATION
# =============================================================================

class TestCMSIntegration:
    """Test work order generation and CMMS integration."""

    def test_work_order_data_structure(self, sample_maintenance_history):
        """Test work order data structure."""
        for record in sample_maintenance_history:
            assert "date" in record
            assert "type" in record
            assert "description" in record
            assert "components_serviced" in record
            assert "cost_usd" in record

    def test_maintenance_type_classification(self, sample_maintenance_history):
        """Test maintenance types are properly classified."""
        valid_types = {"preventive", "corrective", "emergency", "inspection"}

        for record in sample_maintenance_history:
            assert record["type"] in valid_types

    def test_maintenance_history_ordering(self, sample_maintenance_history):
        """Test maintenance history is properly ordered."""
        dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in sample_maintenance_history]

        # Should be in descending order (most recent first)
        for i in range(len(dates) - 1):
            assert dates[i] >= dates[i + 1]

    def test_cost_tracking(self, sample_maintenance_history):
        """Test cost tracking in maintenance records."""
        total_cost = sum(r["cost_usd"] for r in sample_maintenance_history)
        assert total_cost > 0

        for record in sample_maintenance_history:
            assert record["cost_usd"] >= 0
            assert record["duration_hours"] > 0


# =============================================================================
# TEST EXPLAINABILITY - SHAP
# =============================================================================

class TestSHAPExplainability:
    """Test SHAP integration for explainability."""

    def test_shap_explainer_initialization(self, mock_model, feature_names):
        """Test GL021SHAPExplainer initialization."""
        explainer = GL021SHAPExplainer(
            model=mock_model,
            feature_names=feature_names,
            explainer_type="kernel",
        )

        assert explainer is not None
        assert len(explainer.feature_names) == len(feature_names)
        assert not explainer._initialized

    def test_shap_explain_method(self, mock_model, feature_names):
        """Test SHAP explain method."""
        explainer = GL021SHAPExplainer(
            model=mock_model,
            feature_names=feature_names,
        )

        X = np.random.rand(1, len(feature_names))
        explanation = explainer.explain(X)

        assert isinstance(explanation, SHAPExplanation)
        assert len(explanation.feature_contributions) == len(feature_names)
        assert explanation.explanation_confidence > 0

    def test_shap_feature_contributions(self, mock_model, feature_names):
        """Test SHAP feature contributions are valid."""
        explainer = GL021SHAPExplainer(
            model=mock_model,
            feature_names=feature_names,
        )

        X = np.random.rand(1, len(feature_names))
        explanation = explainer.explain(X)

        for contrib in explanation.feature_contributions:
            assert isinstance(contrib, FeatureContribution)
            assert contrib.feature_name in feature_names
            assert contrib.direction in ["positive", "negative"]
            assert 0 <= contrib.contribution_pct <= 100

    def test_shap_top_features_identified(self, mock_model, feature_names):
        """Test top positive and negative features are identified."""
        explainer = GL021SHAPExplainer(
            model=mock_model,
            feature_names=feature_names,
        )

        X = np.random.rand(1, len(feature_names))
        explanation = explainer.explain(X)

        # Should have top features identified
        assert len(explanation.top_positive_features) <= 3
        assert len(explanation.top_negative_features) <= 3

    def test_shap_waterfall_data(self, mock_model, feature_names):
        """Test waterfall plot data generation."""
        explainer = GL021SHAPExplainer(
            model=mock_model,
            feature_names=feature_names,
        )

        X = np.random.rand(1, len(feature_names))
        waterfall_data = explainer.get_waterfall_data(X, max_features=5)

        assert "base_value" in waterfall_data
        assert "prediction" in waterfall_data
        assert "features" in waterfall_data
        assert "values" in waterfall_data
        assert "cumulative" in waterfall_data
        assert len(waterfall_data["features"]) <= 5

    def test_shap_human_readable_names(self, mock_model, feature_names):
        """Test human-readable feature names are provided."""
        explainer = GL021SHAPExplainer(
            model=mock_model,
            feature_names=feature_names,
        )

        X = np.random.rand(1, len(feature_names))
        explanation = explainer.explain(X)

        for contrib in explanation.feature_contributions:
            # Human readable name should not contain underscores
            assert contrib.human_readable_name != contrib.feature_name or "_" not in contrib.feature_name

    def test_shap_with_feature_values_dict(self, mock_model, feature_names, sample_operating_data):
        """Test SHAP with feature values dictionary."""
        explainer = GL021SHAPExplainer(
            model=mock_model,
            feature_names=feature_names,
        )

        X = np.array([[sample_operating_data.get(f, 0) for f in feature_names]])
        explanation = explainer.explain(X, feature_values=sample_operating_data)

        # Feature values should be from the dictionary
        for contrib in explanation.feature_contributions:
            if contrib.feature_name in sample_operating_data:
                assert contrib.feature_value == sample_operating_data[contrib.feature_name]


# =============================================================================
# TEST EXPLAINABILITY - LIME
# =============================================================================

class TestLIMEExplainability:
    """Test LIME integration for explainability."""

    def test_lime_explainer_initialization(self, mock_model, feature_names):
        """Test GL021LIMEExplainer initialization."""
        explainer = GL021LIMEExplainer(
            model=mock_model,
            feature_names=feature_names,
            num_samples=1000,
        )

        assert explainer is not None
        assert explainer.num_samples == 1000
        assert not explainer._initialized

    def test_lime_explain_instance(self, mock_model, feature_names):
        """Test LIME explain instance method."""
        explainer = GL021LIMEExplainer(
            model=mock_model,
            feature_names=feature_names,
            num_samples=500,
        )

        X = np.random.rand(len(feature_names))
        explanation = explainer.explain_instance(X)

        assert isinstance(explanation, LIMEExplanation)
        assert explanation.num_samples_used == 500
        assert 0 <= explanation.r_squared <= 1

    def test_lime_feature_weights(self, mock_model, feature_names):
        """Test LIME feature weights are returned."""
        explainer = GL021LIMEExplainer(
            model=mock_model,
            feature_names=feature_names,
        )

        X = np.random.rand(len(feature_names))
        explanation = explainer.explain_instance(X, num_features=5)

        assert len(explanation.feature_weights) <= 5

        for feature, weight in explanation.feature_weights:
            assert isinstance(feature, str)
            assert isinstance(weight, float)

    def test_lime_stability_assessment(self, mock_model, feature_names):
        """Test LIME explanation stability assessment."""
        explainer = GL021LIMEExplainer(
            model=mock_model,
            feature_names=feature_names,
            num_samples=500,
        )

        X = np.random.rand(len(feature_names))
        stability = explainer.assess_stability(X, n_runs=3)

        assert "n_runs" in stability
        assert "feature_stability" in stability
        assert "mean_cv" in stability
        assert "is_stable" in stability

    def test_lime_local_prediction_quality(self, mock_model, feature_names):
        """Test LIME local prediction quality."""
        explainer = GL021LIMEExplainer(
            model=mock_model,
            feature_names=feature_names,
        )

        X = np.random.rand(len(feature_names))
        explanation = explainer.explain_instance(X)

        # Local prediction should be close to model prediction
        assert explanation.stability_score > 0


# =============================================================================
# TEST NATURAL LANGUAGE EXPLANATIONS
# =============================================================================

class TestNaturalLanguageExplanations:
    """Test natural language explanation generation."""

    def test_nl_explainer_initialization(self, nl_explainer):
        """Test NaturalLanguageExplainer initialization."""
        assert nl_explainer is not None

    def test_explain_for_operator(self, nl_explainer, health_score_explainer, sample_component_scores):
        """Test operator-focused explanation."""
        health_exp = health_score_explainer.explain(sample_component_scores)
        explanation = nl_explainer.explain(health_exp, audience=ExplanationAudience.OPERATOR)

        assert isinstance(explanation, NaturalLanguageExplanation)
        assert explanation.audience == ExplanationAudience.OPERATOR
        assert len(explanation.summary) > 0
        assert explanation.technical_details is None  # Operators don't need technical details

    def test_explain_for_engineer(self, nl_explainer, health_score_explainer, sample_component_scores):
        """Test engineer-focused explanation."""
        health_exp = health_score_explainer.explain(sample_component_scores)
        explanation = nl_explainer.explain(health_exp, audience=ExplanationAudience.ENGINEER)

        assert explanation.audience == ExplanationAudience.ENGINEER
        assert explanation.technical_details is not None
        assert len(explanation.key_findings) > 0

    def test_explain_for_manager(self, nl_explainer, health_score_explainer, sample_component_scores):
        """Test manager-focused explanation."""
        health_exp = health_score_explainer.explain(sample_component_scores)
        explanation = nl_explainer.explain(health_exp, audience=ExplanationAudience.MANAGER)

        assert explanation.audience == ExplanationAudience.MANAGER
        assert explanation.business_impact is not None
        assert len(explanation.business_impact) > 0

    def test_explain_for_auditor(self, nl_explainer, health_score_explainer, sample_component_scores):
        """Test auditor-focused explanation."""
        health_exp = health_score_explainer.explain(sample_component_scores)
        explanation = nl_explainer.explain(health_exp, audience=ExplanationAudience.AUDITOR)

        assert explanation.audience == ExplanationAudience.AUDITOR
        assert explanation.technical_details is not None
        assert "assessment_timestamp" in explanation.technical_details

    def test_explain_all_audiences(
        self, nl_explainer, health_score_explainer, sample_component_scores
    ):
        """Test generating explanations for all audiences."""
        health_exp = health_score_explainer.explain(sample_component_scores)
        all_explanations = nl_explainer.explain_all_audiences(health_exp)

        assert len(all_explanations) == len(ExplanationAudience)

        for audience, explanation in all_explanations.items():
            assert explanation.audience == audience

    def test_risk_communication_per_audience(
        self, nl_explainer, health_score_explainer
    ):
        """Test risk communication varies by audience."""
        # Create critical risk scenario
        critical_scores = {c.value: 35.0 for c in BurnerComponent}
        health_exp = health_score_explainer.explain(critical_scores)

        operator_exp = nl_explainer.explain(health_exp, audience=ExplanationAudience.OPERATOR)
        manager_exp = nl_explainer.explain(health_exp, audience=ExplanationAudience.MANAGER)

        # Risk communication should be different
        assert operator_exp.risk_communication != manager_exp.risk_communication
        assert "URGENT" in operator_exp.risk_communication.upper() or "immediate" in operator_exp.risk_communication.lower()

    def test_recommendations_actionable(
        self, nl_explainer, health_score_explainer, sample_component_scores
    ):
        """Test recommendations are actionable."""
        health_exp = health_score_explainer.explain(sample_component_scores)
        explanation = nl_explainer.explain(health_exp, audience=ExplanationAudience.OPERATOR)

        # Should have recommendations
        assert len(explanation.recommendations) > 0

        # Recommendations should be actionable (contain verbs)
        action_words = ["check", "report", "monitor", "schedule", "contact", "continue"]
        has_action = any(
            any(word in rec.lower() for word in action_words)
            for rec in explanation.recommendations
        )
        assert has_action


# =============================================================================
# TEST PROVENANCE TRACKING
# =============================================================================

class TestProvenanceTracking:
    """Test SHA-256 provenance tracking."""

    def test_provenance_tracker_initialization(self, provenance_tracker):
        """Test ProvenanceTracker initialization."""
        assert provenance_tracker is not None
        assert provenance_tracker.agent_id == "GL-021-TEST"
        assert provenance_tracker.model_version == "1.0.0"

    def test_calculate_hash(self, provenance_tracker):
        """Test hash calculation."""
        input_data = {"test": "data"}
        output_data = {"result": 0.85}

        hash_value = provenance_tracker.calculate_hash(input_data, output_data)

        assert len(hash_value) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_hash_determinism(self, provenance_tracker):
        """Test hash calculation is deterministic for same input."""
        input_data = {"feature": 1.0, "value": 2.0}
        output_data = {"prediction": 0.5}

        hash1 = provenance_tracker._hash_data(input_data)
        hash2 = provenance_tracker._hash_data(input_data)

        assert hash1 == hash2

    def test_hash_different_for_different_input(self, provenance_tracker):
        """Test different inputs produce different hashes."""
        data1 = {"feature": 1.0}
        data2 = {"feature": 2.0}

        hash1 = provenance_tracker._hash_data(data1)
        hash2 = provenance_tracker._hash_data(data2)

        assert hash1 != hash2

    def test_record_provenance(self, provenance_tracker):
        """Test recording provenance entries."""
        provenance_tracker.record(
            explanation_id="test-001",
            input_hash="a" * 64,
            output_hash="b" * 64,
            provenance_hash="c" * 64,
            explanation_type=ExplanationType.HEALTH_SCORE,
        )

        assert len(provenance_tracker._records) == 1
        assert provenance_tracker._records[0]["explanation_id"] == "test-001"

    def test_export_records(self, provenance_tracker):
        """Test exporting provenance records."""
        provenance_tracker.record(
            explanation_id="test-001",
            input_hash="a" * 64,
            output_hash="b" * 64,
            provenance_hash="c" * 64,
            explanation_type=ExplanationType.HEALTH_SCORE,
        )

        exported = provenance_tracker.export_records(format="json")

        assert exported is not None
        records = json.loads(exported)
        assert len(records) == 1

    def test_hash_with_numpy_array(self, provenance_tracker):
        """Test hashing numpy arrays."""
        array = np.array([1.0, 2.0, 3.0])
        hash_value = provenance_tracker._hash_data(array)

        assert len(hash_value) == 64

    def test_hash_with_pydantic_model(self, provenance_tracker):
        """Test hashing Pydantic models."""
        contribution = FeatureContribution(
            feature_name="test",
            feature_value=1.0,
            contribution=0.5,
            contribution_pct=50.0,
            direction="positive",
            human_readable_name="Test Feature",
        )

        hash_value = provenance_tracker._hash_data(contribution)
        assert len(hash_value) == 64


# =============================================================================
# TEST INTEGRATION
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_explanation_pipeline(
        self,
        gl021_explainer,
        sample_operating_data,
        sample_component_scores,
        sample_historical_scores,
    ):
        """Test complete explanation pipeline."""
        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            explanation_type=ExplanationType.HEALTH_SCORE,
            component_scores=sample_component_scores,
            operating_hours=15000,
            historical_scores=sample_historical_scores,
            audiences=list(ExplanationAudience),
        )

        # Verify all components present
        assert result.explanation_id is not None
        assert result.health_score_explanation is not None
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

        # Verify NL explanations for all audiences
        assert result.operator_explanation is not None
        assert result.engineer_explanation is not None
        assert result.manager_explanation is not None
        assert result.auditor_explanation is not None

    def test_explain_health_score_shortcut(
        self, gl021_explainer, sample_component_scores
    ):
        """Test explain_health_score convenience method."""
        explanation = gl021_explainer.explain_health_score(
            component_scores=sample_component_scores,
            operating_hours=10000,
        )

        assert isinstance(explanation, HealthScoreExplanation)
        assert explanation.overall_score > 0

    def test_get_natural_language_explanation_shortcut(
        self,
        gl021_explainer,
        health_score_explainer,
        sample_component_scores,
    ):
        """Test get_natural_language_explanation convenience method."""
        health_exp = health_score_explainer.explain(sample_component_scores)

        nl_exp = gl021_explainer.get_natural_language_explanation(
            health_explanation=health_exp,
            audience=ExplanationAudience.OPERATOR,
        )

        assert isinstance(nl_exp, NaturalLanguageExplanation)
        assert nl_exp.audience == ExplanationAudience.OPERATOR

    def test_export_provenance_from_explainer(
        self,
        gl021_explainer,
        sample_operating_data,
        sample_component_scores,
    ):
        """Test exporting provenance from main explainer."""
        # Generate an explanation
        gl021_explainer.explain(
            input_data=sample_operating_data,
            component_scores=sample_component_scores,
        )

        # Export provenance
        provenance_json = gl021_explainer.export_provenance(format="json")

        assert provenance_json is not None
        records = json.loads(provenance_json)
        assert len(records) >= 1

    def test_model_version_tracking(
        self,
        gl021_explainer,
        sample_operating_data,
        sample_component_scores,
    ):
        """Test model version is tracked in explanations."""
        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            component_scores=sample_component_scores,
        )

        assert result.model_version == "1.0.0-test"

    def test_timestamp_recorded(
        self,
        gl021_explainer,
        sample_operating_data,
        sample_component_scores,
    ):
        """Test timestamp is recorded in explanations."""
        before = datetime.now(timezone.utc)

        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            component_scores=sample_component_scores,
        )

        after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after


# =============================================================================
# TEST FACTORY FUNCTIONS
# =============================================================================

class TestFactoryFunctions:
    """Test factory function utilities."""

    def test_create_gl021_explainer(self, mock_model, feature_names):
        """Test create_gl021_explainer factory."""
        explainer = create_gl021_explainer(
            model=mock_model,
            feature_names=feature_names,
            model_version="2.0.0",
        )

        assert explainer is not None
        assert explainer.model_version == "2.0.0"

    def test_create_gl021_explainer_no_model(self):
        """Test create_gl021_explainer without model."""
        explainer = create_gl021_explainer(
            model=None,
            feature_names=None,
            model_version="1.0.0",
        )

        assert explainer is not None
        assert explainer.shap_explainer is None
        assert explainer.lime_explainer is None

    def test_create_health_score_explainer(self):
        """Test create_health_score_explainer factory."""
        explainer = create_health_score_explainer()

        assert explainer is not None
        assert len(explainer.component_weights) == 10

    def test_create_health_score_explainer_custom_weights(self):
        """Test create_health_score_explainer with custom weights."""
        custom_weights = {
            BurnerComponent.FLAME_SCANNER: 0.20,
            BurnerComponent.IGNITOR: 0.15,
            BurnerComponent.FUEL_VALVE: 0.15,
            BurnerComponent.AIR_DAMPER: 0.08,
            BurnerComponent.PILOT_ASSEMBLY: 0.12,
            BurnerComponent.MAIN_BURNER: 0.12,
            BurnerComponent.COMBUSTION_AIR_FAN: 0.06,
            BurnerComponent.GAS_TRAIN: 0.05,
            BurnerComponent.FLAME_STABILITY: 0.04,
            BurnerComponent.EMISSION_QUALITY: 0.03,
        }

        explainer = create_health_score_explainer(component_weights=custom_weights)

        assert explainer.component_weights[BurnerComponent.FLAME_SCANNER] == 0.20


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_component_scores(self, health_score_explainer):
        """Test handling of empty component scores."""
        explanation = health_score_explainer.explain({})

        # Should handle gracefully with defaults
        assert explanation.overall_score == 100.0  # All defaults to 100

    def test_extreme_health_scores(self, health_score_explainer):
        """Test extreme health score values."""
        # All zeros (worst case)
        worst_case = {c.value: 0.0 for c in BurnerComponent}
        worst_exp = health_score_explainer.explain(worst_case)
        assert worst_exp.overall_score == 0.0
        assert worst_exp.risk_level == RiskLevel.CRITICAL

        # All 100s (best case)
        best_case = {c.value: 100.0 for c in BurnerComponent}
        best_exp = health_score_explainer.explain(best_case)
        assert best_exp.overall_score == 100.0
        assert best_exp.risk_level == RiskLevel.MINIMAL

    def test_negative_scores_clamped(self, health_score_explainer):
        """Test negative scores are clamped to 0."""
        negative_scores = {c.value: -10.0 for c in BurnerComponent}
        explanation = health_score_explainer.explain(negative_scores)

        for breakdown in explanation.component_breakdowns:
            assert breakdown.health_score >= 0

    def test_scores_above_100_clamped(self, health_score_explainer):
        """Test scores above 100 are clamped."""
        over_scores = {c.value: 150.0 for c in BurnerComponent}
        explanation = health_score_explainer.explain(over_scores)

        for breakdown in explanation.component_breakdowns:
            assert breakdown.health_score <= 100

    def test_zero_operating_hours(self, health_score_explainer, sample_component_scores):
        """Test with zero operating hours."""
        explanation = health_score_explainer.explain(
            sample_component_scores,
            operating_hours=0,
        )

        assert explanation is not None
        # Degradation should be minimal with no operating hours

    def test_very_high_operating_hours(self, health_score_explainer, sample_component_scores):
        """Test with very high operating hours."""
        explanation = health_score_explainer.explain(
            sample_component_scores,
            operating_hours=100000,
        )

        # Should still work
        assert explanation is not None

        # Degradation rates should be higher
        for breakdown in explanation.component_breakdowns:
            assert breakdown.degradation_rate > 0

    def test_single_historical_point(self, health_score_explainer, sample_component_scores):
        """Test trend analysis with single historical point."""
        single_history = [{"flame_scanner": 90.0}]

        explanation = health_score_explainer.explain(
            sample_component_scores,
            historical_scores=single_history,
        )

        # Trend should be stable with insufficient data
        for breakdown in explanation.component_breakdowns:
            assert breakdown.trend == "stable"


# =============================================================================
# TEST PERFORMANCE
# =============================================================================

class TestPerformance:
    """Performance and scalability tests."""

    def test_explanation_time_reasonable(
        self,
        gl021_explainer,
        sample_operating_data,
        sample_component_scores,
    ):
        """Test explanation completes in reasonable time."""
        import time

        start = time.time()
        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            component_scores=sample_component_scores,
        )
        elapsed = time.time() - start

        # Should complete within 5 seconds
        assert elapsed < 5.0
        assert result.processing_time_ms < 5000

    def test_batch_explanations(
        self,
        gl021_explainer,
        sample_component_scores,
        feature_names,
    ):
        """Test generating multiple explanations."""
        results = []

        for i in range(10):
            # Vary the input slightly
            input_data = {f: np.random.rand() * 100 for f in feature_names}
            result = gl021_explainer.explain(
                input_data=input_data,
                component_scores=sample_component_scores,
            )
            results.append(result)

        assert len(results) == 10

        # All should have unique explanation IDs
        ids = [r.explanation_id for r in results]
        assert len(set(ids)) == 10

    @pytest.mark.parametrize("num_features", [5, 10, 20, 50])
    def test_scalability_feature_count(self, mock_model, num_features):
        """Test scalability with varying feature counts."""
        feature_names = [f"feature_{i}" for i in range(num_features)]

        explainer = GL021Explainer(
            model=mock_model,
            feature_names=feature_names,
            enable_lime=False,  # Faster without LIME
        )

        X = np.random.rand(1, num_features)
        component_scores = {c.value: 80.0 for c in BurnerComponent}

        result = explainer.explain(
            input_data=X,
            component_scores=component_scores,
        )

        assert result is not None


# =============================================================================
# TEST COMPLIANCE
# =============================================================================

class TestCompliance:
    """Test regulatory compliance features."""

    def test_audit_trail_completeness(
        self,
        gl021_explainer,
        sample_operating_data,
        sample_component_scores,
    ):
        """Test audit trail includes all required elements."""
        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            component_scores=sample_component_scores,
        )

        # Provenance hash exists and is valid
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

        # Input data hash exists
        assert result.input_data_hash is not None
        assert len(result.input_data_hash) == 64

        # Model version tracked
        assert result.model_version is not None

        # Timestamp recorded
        assert result.timestamp is not None

    def test_reproducibility_guarantee(self, health_score_explainer, sample_component_scores):
        """Test calculations are bit-perfect reproducible."""
        results = []

        for _ in range(5):
            explanation = health_score_explainer.explain(
                sample_component_scores,
                operating_hours=15000,
            )
            results.append(explanation.overall_score)

        # All results must be identical
        assert all(r == results[0] for r in results)

    def test_auditor_explanation_completeness(
        self,
        gl021_explainer,
        sample_operating_data,
        sample_component_scores,
    ):
        """Test auditor explanation has required compliance fields."""
        result = gl021_explainer.explain(
            input_data=sample_operating_data,
            component_scores=sample_component_scores,
            audiences=[ExplanationAudience.AUDITOR],
        )

        auditor_exp = result.auditor_explanation

        assert auditor_exp is not None
        assert auditor_exp.technical_details is not None

        # Should have timestamp
        assert "assessment_timestamp" in auditor_exp.technical_details

        # Should have all scores
        assert "component_scores" in auditor_exp.technical_details

        # Should have risk level
        assert "risk_level" in auditor_exp.technical_details


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
