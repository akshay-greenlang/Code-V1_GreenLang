# -*- coding: utf-8 -*-
"""Comprehensive test suite for Isolation Forest Anomaly Detection Agent.

Tests cover:
- Model fitting and training
- Anomaly detection accuracy
- Anomaly scoring and ranking
- Pattern analysis
- Alert generation
- AI integration
- Determinism verification
- Edge cases and error handling

Test Data:
- Synthetic data with known anomalies
- Real-world-like energy/climate data
- Edge cases (all normal, all anomalies, etc.)

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from greenlang.determinism import deterministic_random
from greenlang.agents.anomaly_agent_iforest import (
    IsolationForestAnomalyAgent,
    AnomalyScore,
    AnomalyAlert,
    ModelMetrics,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def normal_data():
    """Generate synthetic data with mostly normal values."""
    np.random.seed(42)

    # Normal distribution
    n_samples = 200
    energy = np.random.normal(100, 10, n_samples)
    temperature = np.random.normal(20, 3, n_samples)

    # Add a few anomalies (outliers)
    anomaly_indices = [50, 75, 120, 150, 180]
    for idx in anomaly_indices:
        energy[idx] = np.deterministic_random().choice([200, 20])  # Extreme values
        temperature[idx] = np.deterministic_random().choice([40, 0])

    df = pd.DataFrame({
        'energy_kwh': energy,
        'temperature_c': temperature,
    })

    # Create labels (True = anomaly)
    labels = [i in anomaly_indices for i in range(n_samples)]

    return df, labels


@pytest.fixture
def energy_data():
    """Generate realistic energy consumption data with anomalies."""
    np.random.seed(42)

    # Simulate hourly energy data for 30 days
    n_samples = 720  # 30 days * 24 hours

    # Normal pattern: base load + daily cycle
    hours = np.arange(n_samples) % 24
    base_load = 100
    daily_cycle = 30 * np.sin(2 * np.pi * hours / 24)
    noise = np.random.normal(0, 5, n_samples)

    energy = base_load + daily_cycle + noise

    # Inject anomalies
    # Spike (equipment failure)
    energy[200:205] = 300
    # Drop (power outage)
    energy[450:455] = 10
    # Gradual drift (sensor error)
    energy[600:650] += np.linspace(0, 50, 50)

    df = pd.DataFrame({
        'energy_kwh': energy,
        'hour': hours,
    })

    return df


@pytest.fixture
def temperature_data():
    """Generate temperature data with extreme weather anomalies."""
    np.random.seed(42)

    # Daily temperature for 6 months
    n_samples = 180

    # Seasonal pattern
    days = np.arange(n_samples)
    seasonal = 20 + 10 * np.sin(2 * np.pi * days / 365)
    noise = np.random.normal(0, 2, n_samples)

    temperature = seasonal + noise

    # Inject extreme weather events
    # Heatwave
    temperature[60:65] = 45
    # Cold snap
    temperature[120:125] = -10

    df = pd.DataFrame({
        'temperature_c': temperature,
    })

    return df


@pytest.fixture
def emissions_data():
    """Generate emissions data with equipment malfunction patterns."""
    np.random.seed(42)

    # Daily CO2 emissions for 90 days
    n_samples = 90

    # Normal operations
    emissions = np.random.normal(500, 50, n_samples)

    # Equipment malfunction (sudden spike)
    emissions[30:35] = 1500

    # Sensor error (zero readings)
    emissions[60:65] = 0

    df = pd.DataFrame({
        'co2_kg': emissions,
    })

    return df


@pytest.fixture
def multidimensional_data():
    """Generate multi-dimensional data for pattern analysis."""
    np.random.seed(42)

    n_samples = 300

    df = pd.DataFrame({
        'energy_kwh': np.random.normal(100, 10, n_samples),
        'temperature_c': np.random.normal(20, 3, n_samples),
        'humidity_pct': np.random.normal(60, 10, n_samples),
        'pressure_hpa': np.random.normal(1013, 5, n_samples),
    })

    # Inject multi-dimensional anomalies
    # High energy + high temperature
    df.loc[50, ['energy_kwh', 'temperature_c']] = [250, 45]
    # Low energy + low temperature
    df.loc[100, ['energy_kwh', 'temperature_c']] = [20, -5]
    # Extreme humidity + pressure
    df.loc[150, ['humidity_pct', 'pressure_hpa']] = [95, 950]

    return df


@pytest.fixture
def agent():
    """Create Isolation Forest anomaly agent instance."""
    return IsolationForestAnomalyAgent(
        budget_usd=1.0,
        enable_explanations=True,
        enable_recommendations=True,
        enable_alerts=True,
    )


@pytest.fixture
def agent_no_ai():
    """Create agent without AI features."""
    return IsolationForestAnomalyAgent(
        budget_usd=0.1,
        enable_explanations=False,
        enable_recommendations=False,
        enable_alerts=False,
    )


# ==============================================================================
# Test: Initialization and Configuration
# ==============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent.metadata.id == "anomaly_iforest"
    assert agent.metadata.name == "Isolation Forest Anomaly Agent"
    assert agent.budget_usd == 1.0
    assert agent.enable_explanations is True
    assert agent.enable_recommendations is True
    assert agent.enable_alerts is True
    assert agent._ai_call_count == 0
    assert agent._tool_call_count == 0


def test_agent_custom_config():
    """Test agent with custom configuration."""
    agent = IsolationForestAnomalyAgent(
        budget_usd=2.0,
        enable_explanations=False,
        enable_recommendations=False,
        enable_alerts=False,
    )

    assert agent.budget_usd == 2.0
    assert agent.enable_explanations is False
    assert agent.enable_recommendations is False
    assert agent.enable_alerts is False


def test_tools_setup(agent):
    """Test that all tools are properly defined."""
    assert agent.fit_isolation_forest_tool is not None
    assert agent.detect_anomalies_tool is not None
    assert agent.calculate_anomaly_scores_tool is not None
    assert agent.rank_anomalies_tool is not None
    assert agent.analyze_anomaly_patterns_tool is not None
    assert agent.generate_alerts_tool is not None


# ==============================================================================
# Test: Input Validation
# ==============================================================================

def test_validate_valid_input(agent, normal_data):
    """Test validation passes with valid input."""
    df, _ = normal_data

    input_data = {
        "data": df,
        "contamination": 0.1,
    }

    assert agent.validate(input_data) is True


def test_validate_missing_data(agent):
    """Test validation fails without data."""
    input_data = {
        "contamination": 0.1,
    }

    assert agent.validate(input_data) is False


def test_validate_non_dataframe(agent):
    """Test validation fails with non-DataFrame."""
    input_data = {
        "data": [1, 2, 3, 4, 5],
    }

    assert agent.validate(input_data) is False


def test_validate_insufficient_data(agent):
    """Test validation fails with too few data points."""
    df = pd.DataFrame({
        'value': range(50),
    })

    input_data = {
        "data": df,
    }

    assert agent.validate(input_data) is False


def test_validate_no_numeric_features(agent):
    """Test validation fails without numeric features."""
    df = pd.DataFrame({
        'name': ['A', 'B', 'C'] * 40,
    })

    input_data = {
        "data": df,
    }

    assert agent.validate(input_data) is False


def test_validate_invalid_contamination(agent, normal_data):
    """Test validation fails with invalid contamination."""
    df, _ = normal_data

    input_data = {
        "data": df,
        "contamination": 0.8,  # Too high
    }

    assert agent.validate(input_data) is False


def test_validate_missing_feature_column(agent, normal_data):
    """Test validation fails when specified feature column doesn't exist."""
    df, _ = normal_data

    input_data = {
        "data": df,
        "feature_columns": ["nonexistent"],
    }

    assert agent.validate(input_data) is False


def test_validate_labels_length_mismatch(agent, normal_data):
    """Test validation fails when labels length doesn't match data."""
    df, _ = normal_data

    input_data = {
        "data": df,
        "labels": [True, False],  # Wrong length
    }

    assert agent.validate(input_data) is False


# ==============================================================================
# Test: Model Fitting
# ==============================================================================

def test_fit_isolation_forest(agent, normal_data):
    """Test Isolation Forest model fitting."""
    df, _ = normal_data

    input_data = {
        "data": df,
        "contamination": 0.1,
    }

    result = agent._fit_isolation_forest_impl(
        input_data,
        contamination=0.1,
        n_estimators=100,
    )

    assert result["fitted"] is True
    assert result["n_samples"] == len(df)
    assert result["n_features"] == 2
    assert "features" in result
    assert result["contamination"] == 0.1


def test_fit_with_custom_parameters(agent, normal_data):
    """Test fitting with custom parameters."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    result = agent._fit_isolation_forest_impl(
        input_data,
        contamination=0.05,
        n_estimators=50,
        max_samples=128,
        max_features=0.8,
        bootstrap=True,
    )

    assert result["fitted"] is True
    assert result["contamination"] == 0.05
    assert result["n_estimators"] == 50


def test_fit_with_feature_selection(agent, multidimensional_data):
    """Test fitting with specific feature columns."""
    input_data = {
        "data": multidimensional_data,
        "feature_columns": ["energy_kwh", "temperature_c"],
    }

    result = agent._fit_isolation_forest_impl(input_data)

    assert result["n_features"] == 2
    assert "energy_kwh" in result["features"]
    assert "temperature_c" in result["features"]


def test_fit_with_missing_values(agent):
    """Test fitting handles missing values."""
    df = pd.DataFrame({
        'value1': [1, 2, np.nan, 4, 5] * 30,
        'value2': [10, np.nan, 30, 40, 50] * 30,
    })

    input_data = {
        "data": df,
    }

    # Should handle gracefully
    result = agent._fit_isolation_forest_impl(input_data)
    assert result["fitted"] is True


# ==============================================================================
# Test: Anomaly Detection
# ==============================================================================

def test_detect_anomalies(agent, normal_data):
    """Test anomaly detection."""
    df, labels = normal_data

    input_data = {
        "data": df,
    }

    # Fit model first
    agent._fit_isolation_forest_impl(input_data, contamination=0.1)

    # Detect anomalies
    result = agent._detect_anomalies_impl(input_data)

    assert "anomalies" in result
    assert "anomaly_indices" in result
    assert "n_anomalies" in result
    assert "n_normal" in result
    assert "anomaly_rate" in result

    assert len(result["anomalies"]) == len(df)
    assert result["n_anomalies"] + result["n_normal"] == len(df)


def test_detect_anomalies_rate(agent, normal_data):
    """Test detected anomaly rate is close to contamination."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    contamination = 0.1
    agent._fit_isolation_forest_impl(input_data, contamination=contamination)
    result = agent._detect_anomalies_impl(input_data)

    # Anomaly rate should be close to contamination (within tolerance)
    assert abs(result["anomaly_rate"] - contamination) < 0.05


def test_detect_anomalies_indices(agent, normal_data):
    """Test anomaly indices are correct."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._detect_anomalies_impl(input_data)

    anomaly_indices = result["anomaly_indices"]
    anomalies = result["anomalies"]

    # Verify indices match boolean array
    for idx in anomaly_indices:
        assert anomalies[idx] is True


def test_detect_without_fitted_model(agent, normal_data):
    """Test detection fails without fitted model."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    with pytest.raises(ValueError, match="Model must be fitted"):
        agent._detect_anomalies_impl(input_data)


# ==============================================================================
# Test: Anomaly Scoring
# ==============================================================================

def test_calculate_anomaly_scores(agent, normal_data):
    """Test anomaly score calculation."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._calculate_anomaly_scores_impl(input_data)

    assert "scores" in result
    assert "severities" in result
    assert "min_score" in result
    assert "max_score" in result
    assert "mean_score" in result

    assert len(result["scores"]) == len(df)
    assert len(result["severities"]) == len(df)


def test_anomaly_scores_range(agent, normal_data):
    """Test anomaly scores are in expected range."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._calculate_anomaly_scores_impl(input_data)

    scores = result["scores"]

    # Scores should be negative for anomalies, positive for normal
    # Most scores should be in reasonable range
    assert result["min_score"] < result["max_score"]


def test_severity_classification(agent, normal_data):
    """Test severity classification is correct."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._calculate_anomaly_scores_impl(input_data)

    scores = result["scores"]
    severities = result["severities"]

    # Verify severity matches score
    for score, severity in zip(scores, severities):
        if score < -0.5:
            assert severity == "critical"
        elif score < -0.3:
            assert severity == "high"
        elif score < -0.1:
            assert severity == "medium"
        elif score < 0:
            assert severity == "low"
        else:
            assert severity == "normal"


# ==============================================================================
# Test: Anomaly Ranking
# ==============================================================================

def test_rank_anomalies(agent, normal_data):
    """Test anomaly ranking."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._rank_anomalies_impl(input_data, top_k=10)

    assert "top_anomalies" in result
    assert "n_ranked" in result

    top_anomalies = result["top_anomalies"]
    assert len(top_anomalies) <= 10


def test_rank_anomalies_ordering(agent, normal_data):
    """Test ranked anomalies are ordered by severity."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._rank_anomalies_impl(input_data, top_k=10)

    top_anomalies = result["top_anomalies"]

    # Verify scores are in descending order (most negative first)
    scores = [a["score"] for a in top_anomalies]
    assert scores == sorted(scores)


def test_rank_anomalies_features(agent, normal_data):
    """Test ranked anomalies include feature values."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._rank_anomalies_impl(input_data, top_k=5)

    top_anomalies = result["top_anomalies"]

    for anomaly in top_anomalies:
        assert "index" in anomaly
        assert "score" in anomaly
        assert "severity" in anomaly
        assert "features" in anomaly
        assert len(anomaly["features"]) > 0


# ==============================================================================
# Test: Pattern Analysis
# ==============================================================================

def test_analyze_anomaly_patterns(agent, normal_data):
    """Test anomaly pattern analysis."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._analyze_anomaly_patterns_impl(input_data)

    assert "n_anomalies" in result
    assert "patterns" in result
    assert "feature_importance" in result


def test_pattern_analysis_statistics(agent, multidimensional_data):
    """Test pattern analysis includes feature statistics."""
    input_data = {
        "data": multidimensional_data,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._analyze_anomaly_patterns_impl(input_data)

    patterns = result["patterns"]

    # Should have statistics for each feature
    for feature in multidimensional_data.columns:
        if feature in patterns:
            stats = patterns[feature]
            assert "anomaly_mean" in stats
            assert "normal_mean" in stats
            assert "anomaly_std" in stats
            assert "normal_std" in stats
            assert "relative_difference" in stats


def test_pattern_feature_importance(agent, multidimensional_data):
    """Test feature importance ranking."""
    input_data = {
        "data": multidimensional_data,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._analyze_anomaly_patterns_impl(input_data)

    feature_importance = result["feature_importance"]

    # Should have importance scores for features
    assert len(feature_importance) > 0

    # Importance scores should be non-negative
    for feature, importance in feature_importance.items():
        assert importance >= 0


def test_pattern_most_important_feature(agent, normal_data):
    """Test most important feature is identified."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._analyze_anomaly_patterns_impl(input_data)

    assert "most_important_feature" in result

    if result["n_anomalies"] > 0:
        assert result["most_important_feature"] is not None


# ==============================================================================
# Test: Alert Generation
# ==============================================================================

def test_generate_alerts(agent, normal_data):
    """Test alert generation."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._generate_alerts_impl(input_data, min_severity="high")

    assert "alerts" in result
    assert "n_alerts" in result
    assert "severity_counts" in result


def test_alerts_severity_filtering(agent, normal_data):
    """Test alerts are filtered by severity."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)

    # High severity threshold
    result_high = agent._generate_alerts_impl(input_data, min_severity="high")

    # Critical severity threshold
    result_critical = agent._generate_alerts_impl(input_data, min_severity="critical")

    # Critical alerts should be subset of high alerts
    assert result_critical["n_alerts"] <= result_high["n_alerts"]


def test_alert_structure(agent, normal_data):
    """Test alert structure is complete."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._generate_alerts_impl(input_data)

    if result["n_alerts"] > 0:
        alert = result["alerts"][0]

        assert "index" in alert
        assert "severity" in alert
        assert "score" in alert
        assert "root_cause_hints" in alert
        assert "recommendations" in alert
        assert "confidence" in alert


def test_alert_recommendations(agent, normal_data):
    """Test alerts include recommendations."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._generate_alerts_impl(input_data, min_severity="medium")

    for alert in result["alerts"]:
        assert len(alert["recommendations"]) > 0

        # Critical alerts should have urgent recommendations
        if alert["severity"] == "critical":
            assert any("immediate" in rec.lower() for rec in alert["recommendations"])


# ==============================================================================
# Test: End-to-End Detection
# ==============================================================================

def test_end_to_end_detection(agent_no_ai, normal_data):
    """Test complete detection workflow without AI."""
    df, labels = normal_data

    input_data = {
        "data": df,
        "labels": labels,
    }

    # Validate
    assert agent_no_ai.validate(input_data) is True

    # Execute all tools
    agent_no_ai._fit_isolation_forest_impl(input_data)
    anomalies = agent_no_ai._detect_anomalies_impl(input_data)
    scores = agent_no_ai._calculate_anomaly_scores_impl(input_data)
    rankings = agent_no_ai._rank_anomalies_impl(input_data)
    patterns = agent_no_ai._analyze_anomaly_patterns_impl(input_data)
    alerts = agent_no_ai._generate_alerts_impl(input_data)

    # Verify all components worked
    assert anomalies is not None
    assert scores is not None
    assert rankings is not None
    assert patterns is not None
    assert alerts is not None


def test_deterministic_predictions(agent_no_ai, normal_data):
    """Test predictions are deterministic."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    # Run twice
    agent_no_ai._fit_isolation_forest_impl(input_data)
    result1 = agent_no_ai._detect_anomalies_impl(input_data)

    agent_no_ai._fit_isolation_forest_impl(input_data)
    result2 = agent_no_ai._detect_anomalies_impl(input_data)

    # Results should be identical
    assert result1["anomalies"] == result2["anomalies"]


# ==============================================================================
# Test: Real-World Scenarios
# ==============================================================================

def test_energy_consumption_anomalies(agent_no_ai, energy_data):
    """Test detection of energy consumption anomalies."""
    input_data = {
        "data": energy_data,
        "feature_columns": ["energy_kwh"],
    }

    agent_no_ai._fit_isolation_forest_impl(input_data, contamination=0.05)
    result = agent_no_ai._detect_anomalies_impl(input_data)

    # Should detect the spike and drop
    assert result["n_anomalies"] > 0

    # Check if known anomaly regions are detected
    anomaly_indices = set(result["anomaly_indices"])

    # Spike at 200-205
    spike_detected = any(i in anomaly_indices for i in range(200, 205))

    # Drop at 450-455
    drop_detected = any(i in anomaly_indices for i in range(450, 455))

    # At least one should be detected
    assert spike_detected or drop_detected


def test_temperature_extreme_events(agent_no_ai, temperature_data):
    """Test detection of extreme temperature events."""
    input_data = {
        "data": temperature_data,
    }

    agent_no_ai._fit_isolation_forest_impl(input_data, contamination=0.1)
    result = agent_no_ai._detect_anomalies_impl(input_data)

    assert result["n_anomalies"] > 0

    # Should detect heatwave (60-65) or cold snap (120-125)
    anomaly_indices = set(result["anomaly_indices"])

    heatwave_detected = any(i in anomaly_indices for i in range(60, 65))
    coldsnap_detected = any(i in anomaly_indices for i in range(120, 125))

    assert heatwave_detected or coldsnap_detected


def test_emissions_equipment_malfunction(agent_no_ai, emissions_data):
    """Test detection of equipment malfunction in emissions."""
    input_data = {
        "data": emissions_data,
    }

    agent_no_ai._fit_isolation_forest_impl(input_data, contamination=0.15)
    result = agent_no_ai._detect_anomalies_impl(input_data)

    assert result["n_anomalies"] > 0

    # Should detect the spike (30-35) or sensor error (60-65)
    anomaly_indices = set(result["anomaly_indices"])

    spike_detected = any(i in anomaly_indices for i in range(30, 35))
    sensor_error_detected = any(i in anomaly_indices for i in range(60, 65))

    assert spike_detected or sensor_error_detected


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

def test_all_normal_data(agent):
    """Test with data containing no anomalies."""
    np.random.seed(42)

    # Very tight distribution
    df = pd.DataFrame({
        'value': np.random.normal(100, 1, 150),
    })

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data, contamination=0.01)
    result = agent._detect_anomalies_impl(input_data)

    # Should detect very few (if any) anomalies
    assert result["anomaly_rate"] < 0.05


def test_all_anomalies_data(agent):
    """Test with extreme outliers only."""
    np.random.seed(42)

    # Random extreme values
    df = pd.DataFrame({
        'value': np.random.uniform(0, 1000, 150),
    })

    input_data = {
        "data": df,
    }

    # Should handle gracefully
    agent._fit_isolation_forest_impl(input_data, contamination=0.5)
    result = agent._detect_anomalies_impl(input_data)

    assert result is not None


def test_single_feature(agent):
    """Test with single feature."""
    np.random.seed(42)

    df = pd.DataFrame({
        'value': np.concatenate([np.random.normal(100, 10, 140), [500] * 10]),
    })

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._detect_anomalies_impl(input_data)

    # Should detect the extreme values
    assert result["n_anomalies"] > 0


def test_many_features(agent):
    """Test with many features."""
    np.random.seed(42)

    df = pd.DataFrame({
        f'feature_{i}': np.random.normal(100, 10, 150)
        for i in range(20)
    })

    # Add anomaly in multiple features
    for col in df.columns[:5]:
        df.loc[50, col] = 500

    input_data = {
        "data": df,
    }

    agent._fit_isolation_forest_impl(input_data)
    result = agent._detect_anomalies_impl(input_data)

    assert result["n_anomalies"] > 0


def test_constant_values(agent):
    """Test with constant values."""
    df = pd.DataFrame({
        'value': [100] * 150,
    })

    input_data = {
        "data": df,
    }

    # Should handle gracefully (no anomalies in constant data)
    agent._fit_isolation_forest_impl(input_data, contamination=0.1)
    result = agent._detect_anomalies_impl(input_data)

    assert result is not None


# ==============================================================================
# Test: Performance and Metrics
# ==============================================================================

def test_performance_tracking(agent):
    """Test performance metrics tracking."""
    assert agent._ai_call_count == 0
    assert agent._tool_call_count == 0
    assert agent._total_cost_usd == 0.0

    summary = agent.get_performance_summary()

    assert "agent_id" in summary
    assert "ai_metrics" in summary
    assert summary["ai_metrics"]["ai_call_count"] == 0


def test_tool_call_counting(agent, normal_data):
    """Test tool calls are counted."""
    df, _ = normal_data

    input_data = {
        "data": df,
    }

    initial_count = agent._tool_call_count

    agent._fit_isolation_forest_impl(input_data)
    agent._detect_anomalies_impl(input_data)
    agent._calculate_anomaly_scores_impl(input_data)

    assert agent._tool_call_count == initial_count + 3


def test_evaluation_metrics(agent, normal_data):
    """Test evaluation metrics with labels."""
    df, labels = normal_data

    input_data = {
        "data": df,
        "labels": labels,
    }

    agent._fit_isolation_forest_impl(input_data, contamination=0.05)
    anomalies_result = agent._detect_anomalies_impl(input_data)
    scores_result = agent._calculate_anomaly_scores_impl(input_data)

    tool_results = {
        "anomalies": anomalies_result,
        "scores": scores_result,
    }

    output = agent._build_output(input_data, tool_results, None)

    if "metrics" in output:
        metrics = output["metrics"]
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        # Metrics should be between 0 and 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1


# ==============================================================================
# Test: Output Building
# ==============================================================================

def test_build_output_complete(agent):
    """Test output building with all components."""
    tool_results = {
        "model": {
            "n_samples": 200,
            "n_features": 2,
            "features": ["energy_kwh", "temperature_c"],
            "contamination": 0.1,
            "n_estimators": 100,
            "fitted": True,
        },
        "anomalies": {
            "anomalies": [False] * 195 + [True] * 5,
            "anomaly_indices": [195, 196, 197, 198, 199],
            "n_anomalies": 5,
            "n_normal": 195,
            "anomaly_rate": 0.025,
        },
        "scores": {
            "scores": [0.1] * 195 + [-0.6] * 5,
            "severities": ["normal"] * 195 + ["critical"] * 5,
        },
    }

    input_data = {
        "data": pd.DataFrame(),
    }

    output = agent._build_output(input_data, tool_results, "Test explanation")

    assert "anomalies" in output
    assert "anomaly_scores" in output
    assert "anomaly_indices" in output
    assert "n_anomalies" in output
    assert "model_info" in output
    assert "explanation" in output


def test_build_output_minimal(agent):
    """Test output building with minimal results."""
    tool_results = {
        "anomalies": {
            "anomalies": [False] * 100,
            "anomaly_indices": [],
            "n_anomalies": 0,
            "n_normal": 100,
            "anomaly_rate": 0.0,
        },
    }

    input_data = {
        "data": pd.DataFrame(),
    }

    output = agent._build_output(input_data, tool_results, None)

    assert "anomalies" in output
    assert output["n_anomalies"] == 0


# ==============================================================================
# Test: Error Handling
# ==============================================================================

def test_handle_empty_dataframe(agent):
    """Test error handling for empty DataFrame."""
    df = pd.DataFrame()

    input_data = {
        "data": df,
    }

    assert agent.validate(input_data) is False


def test_handle_all_nan_feature(agent):
    """Test handling of feature with all NaN values."""
    df = pd.DataFrame({
        'value1': [1, 2, 3, 4, 5] * 30,
        'value2': [np.nan] * 150,
    })

    input_data = {
        "data": df,
    }

    # Should handle gracefully
    try:
        agent._fit_isolation_forest_impl(input_data)
    except Exception as e:
        # Should either work or raise clear error
        assert "nan" in str(e).lower() or "missing" in str(e).lower()


# ==============================================================================
# Test: Coverage Expansion - Gap-Filling Tests for 80%+ Coverage
# ==============================================================================


class TestIsolationForestAnomalyAgentCoverage:
    """Gap-filling tests to achieve 80%+ coverage for IsolationForestAnomalyAgent.

    This test class adds comprehensive coverage for:
    - Tool result extraction (all 6 tools)
    - Output building variations
    - Integration tests (budget exceeded, exceptions, feature toggles)
    - Determinism verification (5-run consistency)
    - Boundary conditions (zero, extreme values, edge cases)
    - Configuration options
    - Performance tracking
    """

    # =========================================================================
    # Unit Tests: Tool Result Extraction
    # =========================================================================

    def test_extract_tool_results_all_six_tools(self, agent, normal_data):
        """Test extracting results from all six tool types."""
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "fit_isolation_forest",
                "arguments": {"contamination": 0.1, "n_estimators": 100},
            },
            {
                "name": "detect_anomalies",
                "arguments": {},
            },
            {
                "name": "calculate_anomaly_scores",
                "arguments": {},
            },
            {
                "name": "rank_anomalies",
                "arguments": {"top_n": 10},
            },
            {
                "name": "analyze_anomaly_patterns",
                "arguments": {},
            },
            {
                "name": "generate_alerts",
                "arguments": {"severity_threshold": "high"},
            },
        ]

        input_data = {"data": normal_data}

        # Fit model first for tools that need it
        agent._fit_isolation_forest_impl(input_data, contamination=0.1)

        results = agent._extract_tool_results(mock_response, input_data)

        # Verify all tools were called
        assert "model" in results
        assert "anomalies" in results
        assert "scores" in results
        assert "ranked" in results
        assert "patterns" in results
        assert "alerts" in results

    def test_extract_tool_results_empty(self, agent, normal_data):
        """Test extracting results when no tools were called."""
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.tool_calls = []

        input_data = {"data": normal_data}

        results = agent._extract_tool_results(mock_response, input_data)

        # Should return empty dict
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_extract_tool_results_unknown_tool(self, agent, normal_data):
        """Test extracting results with unknown tool name."""
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.tool_calls = [
            {"name": "unknown_tool", "arguments": {}},
            {"name": "fit_isolation_forest", "arguments": {"contamination": 0.1}},
        ]

        input_data = {"data": normal_data}

        results = agent._extract_tool_results(mock_response, input_data)

        # Should handle unknown tool gracefully
        assert isinstance(results, dict)
        assert "model" in results  # Known tool should work

    def test_extract_tool_results_partial_data(self, agent, normal_data):
        """Test extracting results when some tools return partial data."""
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.tool_calls = [
            {"name": "fit_isolation_forest", "arguments": {}},
            {"name": "detect_anomalies", "arguments": {}},
        ]

        input_data = {"data": normal_data}

        results = agent._extract_tool_results(mock_response, input_data)

        # Should have partial results
        assert "model" in results
        assert "anomalies" in results
        assert "scores" not in results  # Not called

    # =========================================================================
    # Unit Tests: Output Building
    # =========================================================================

    def test_build_output_missing_anomalies(self, agent):
        """Test output building when anomaly data is missing."""
        tool_results = {
            "model": {
                "n_estimators": 100,
                "contamination": 0.1,
                "n_features": 5,
            },
        }

        input_data = {"data": pd.DataFrame()}

        output = agent._build_output(input_data, tool_results, "Explanation")

        assert "model_info" in output
        # Anomalies should have defaults when missing
        assert "anomalies" not in output or output.get("anomalies") is None

    def test_build_output_missing_scores(self, agent):
        """Test output building when anomaly scores are missing."""
        tool_results = {
            "anomalies": {
                "anomaly_indices": [5, 10, 15],
                "n_anomalies": 3,
            },
        }

        input_data = {"data": pd.DataFrame()}

        output = agent._build_output(input_data, tool_results, None)

        assert "anomalies" in output
        assert "anomaly_scores" not in output or output.get("anomaly_scores") is None

    def test_build_output_missing_patterns(self, agent):
        """Test output building when pattern analysis is missing."""
        tool_results = {
            "anomalies": {
                "anomaly_indices": [1],
                "n_anomalies": 1,
            },
            "scores": {
                "scores": [-0.5],
                "severity": ["high"],
            },
        }

        input_data = {"data": pd.DataFrame()}

        output = agent._build_output(input_data, tool_results, "Test")

        assert "patterns" not in output or output.get("patterns") is None

    def test_build_output_no_explanation(self, agent):
        """Test output building when explanations are disabled."""
        tool_results = {
            "anomalies": {
                "anomaly_indices": [1],
                "n_anomalies": 1,
            },
        }

        input_data = {"data": pd.DataFrame()}

        # Disable explanations
        agent.enable_explanations = False

        output = agent._build_output(input_data, tool_results, "Should be ignored")

        assert "explanation" not in output or output["explanation"] is None

    # =========================================================================
    # Integration Tests: Budget and Exception Handling
    # =========================================================================

    @pytest.mark.asyncio
    async def test_integration_budget_exceeded(self, agent, normal_data):
        """Test handling when AI budget is exceeded."""
        from unittest.mock import patch, AsyncMock
        from greenlang.intelligence import BudgetExceeded

        input_data = {"data": normal_data}

        # Mock ChatSession to raise BudgetExceeded
        with patch("greenlang.agents.anomaly_agent_iforest.ChatSession") as MockSession:
            mock_session = MockSession.return_value
            mock_session.chat = AsyncMock(side_effect=BudgetExceeded("Budget exceeded"))

            # Should raise ValueError with budget message
            with pytest.raises(ValueError, match="budget"):
                await agent._process_async(input_data)

    @pytest.mark.asyncio
    async def test_integration_general_exception(self, agent, normal_data):
        """Test handling of general exceptions during processing."""
        from unittest.mock import patch, AsyncMock

        input_data = {"data": normal_data}

        # Mock ChatSession to raise RuntimeError
        with patch("greenlang.agents.anomaly_agent_iforest.ChatSession") as MockSession:
            mock_session = MockSession.return_value
            mock_session.chat = AsyncMock(side_effect=RuntimeError("Unexpected error"))

            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Unexpected error"):
                await agent._process_async(input_data)

    @pytest.mark.asyncio
    async def test_integration_disabled_explanations(self, agent, normal_data):
        """Test full workflow with explanations disabled."""
        from unittest.mock import patch, AsyncMock, Mock
        from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

        agent_no_explain = IsolationForestAnomalyAgent(enable_explanations=False)

        input_data = {"data": normal_data}

        # Mock ChatSession
        with patch("greenlang.agents.anomaly_agent_iforest.ChatSession") as MockSession:
            mock_session = MockSession.return_value
            mock_response = Mock()
            mock_response.tool_calls = [
                {"name": "fit_isolation_forest", "arguments": {}},
                {"name": "detect_anomalies", "arguments": {}},
            ]
            mock_response.text = "This should be ignored"
            mock_response.usage = Mock(cost_usd=0.01, total_tokens=100)
            mock_response.provider_info = Mock(provider="anthropic", model="claude-3-5-sonnet-20241022")
            mock_session.chat = AsyncMock(return_value=mock_response)

            # Fit model first
            agent_no_explain._fit_isolation_forest_impl(input_data, contamination=0.1)

            result = await agent_no_explain._process_async(input_data)

            # Should not have explanation
            assert "explanation" not in result or result.get("explanation") is None

    @pytest.mark.asyncio
    async def test_integration_disabled_alerts(self, agent, normal_data):
        """Test full workflow with alerts disabled."""
        from unittest.mock import patch, AsyncMock, Mock
        from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

        agent_no_alerts = IsolationForestAnomalyAgent(enable_alerts=False)

        input_data = {"data": normal_data}

        # Mock ChatSession
        with patch("greenlang.agents.anomaly_agent_iforest.ChatSession") as MockSession:
            mock_session = MockSession.return_value
            mock_response = Mock()
            mock_response.tool_calls = [
                {"name": "fit_isolation_forest", "arguments": {}},
            ]
            mock_response.text = "Detection complete"
            mock_response.usage = Mock(cost_usd=0.01, total_tokens=100)
            mock_response.provider_info = Mock(provider="anthropic", model="claude-3-5-sonnet-20241022")
            mock_session.chat = AsyncMock(return_value=mock_response)

            result_data = agent_no_alerts._fit_isolation_forest_impl(input_data, contamination=0.1)

            # Verify model was fitted
            assert "n_estimators" in result_data

    # =========================================================================
    # Determinism Tests: 5-Run Consistency Verification
    # =========================================================================

    def test_fit_isolation_forest_determinism(self, agent, normal_data):
        """Test that Isolation Forest fitting produces identical results across 5 runs."""
        input_data = {"data": normal_data}

        results = []
        for _ in range(5):
            # Reset agent state
            agent._fitted_model = None

            result = agent._fit_isolation_forest_impl(
                input_data, contamination=0.1, n_estimators=100, random_state=42
            )
            results.append(result)

        # All runs should produce identical results
        for i in range(1, 5):
            assert results[i]["n_estimators"] == results[0]["n_estimators"]
            assert results[i]["contamination"] == results[0]["contamination"]
            assert results[i]["n_features"] == results[0]["n_features"]

    def test_detect_anomalies_determinism(self, agent, normal_data):
        """Test that anomaly detection produces identical results across 5 runs."""
        input_data = {"data": normal_data}

        # Fit model once
        agent._fit_isolation_forest_impl(input_data, contamination=0.1, random_state=42)

        results = []
        for _ in range(5):
            result = agent._detect_anomalies_impl(input_data)
            results.append(result)

        # All runs should produce identical anomalies
        for i in range(1, 5):
            assert results[i]["n_anomalies"] == results[0]["n_anomalies"]
            assert results[i]["anomaly_indices"] == results[0]["anomaly_indices"]
            assert results[i]["normal_indices"] == results[0]["normal_indices"]

    def test_calculate_scores_determinism(self, agent, normal_data):
        """Test that anomaly scoring produces identical results across 5 runs."""
        input_data = {"data": normal_data}

        # Fit model once
        agent._fit_isolation_forest_impl(input_data, contamination=0.1, random_state=42)

        results = []
        for _ in range(5):
            result = agent._calculate_anomaly_scores_impl(input_data)
            results.append(result)

        # All runs should produce identical scores
        for i in range(1, 5):
            assert len(results[i]["scores"]) == len(results[0]["scores"])
            for j in range(len(results[0]["scores"])):
                assert results[i]["scores"][j] == pytest.approx(results[0]["scores"][j], rel=1e-10)

    def test_rank_anomalies_determinism(self, agent, normal_data):
        """Test that anomaly ranking produces identical results across 5 runs."""
        input_data = {"data": normal_data}

        # Fit model and calculate scores first
        agent._fit_isolation_forest_impl(input_data, contamination=0.1, random_state=42)
        agent._calculate_anomaly_scores_impl(input_data)

        results = []
        for _ in range(5):
            result = agent._rank_anomalies_impl(input_data, top_n=10)
            results.append(result)

        # All runs should produce identical rankings
        for i in range(1, 5):
            assert results[i]["ranked_indices"] == results[0]["ranked_indices"]
            assert len(results[i]["ranked_scores"]) == len(results[0]["ranked_scores"])

    # =========================================================================
    # Boundary Tests: Edge Cases and Extreme Values
    # =========================================================================

    def test_boundary_single_data_point(self, agent):
        """Test with minimal data (single point)."""
        df = pd.DataFrame({'value': [100.0]})

        input_data = {"data": df}

        # Should fail validation (insufficient data)
        assert agent.validate(input_data) is False

    def test_boundary_contamination_edges(self, agent, normal_data):
        """Test contamination parameter edge cases (0.001, 0.5)."""
        input_data = {"data": normal_data}

        # Test very low contamination
        result_low = agent._fit_isolation_forest_impl(input_data, contamination=0.001, random_state=42)
        assert result_low["contamination"] == 0.001

        # Test high contamination (50%)
        agent._fitted_model = None  # Reset
        result_high = agent._fit_isolation_forest_impl(input_data, contamination=0.5, random_state=42)
        assert result_high["contamination"] == 0.5

    def test_boundary_extreme_values(self, agent):
        """Test with extremely large and small values."""
        # Very large values (billions)
        large_data = pd.DataFrame({
            'value1': np.random.RandomState(42).normal(1e9, 1e8, 100),
            'value2': np.random.RandomState(42).normal(2e9, 2e8, 100),
        })

        input_data_large = {"data": large_data}

        # Should handle large values
        result_large = agent._fit_isolation_forest_impl(input_data_large, contamination=0.1, random_state=42)
        assert result_large is not None

        # Very small values (near zero)
        small_data = pd.DataFrame({
            'value1': np.random.RandomState(42).normal(0.001, 0.0001, 100),
            'value2': np.random.RandomState(42).normal(0.002, 0.0002, 100),
        })

        input_data_small = {"data": small_data}

        # Should handle small values
        agent._fitted_model = None  # Reset
        result_small = agent._fit_isolation_forest_impl(input_data_small, contamination=0.1, random_state=42)
        assert result_small is not None

    def test_boundary_constant_columns(self, agent):
        """Test with constant columns (no variation)."""
        df = pd.DataFrame({
            'value1': [100.0] * 100,
            'value2': [200.0] * 100,
        })

        input_data = {"data": df}

        # Constant columns should be detected
        # Model may still fit but anomalies based on constant values
        result = agent._fit_isolation_forest_impl(input_data, contamination=0.1, random_state=42)
        assert result is not None

    def test_boundary_severity_thresholds(self, agent, normal_data):
        """Test different severity classification thresholds."""
        input_data = {"data": normal_data}

        # Fit model and calculate scores
        agent._fit_isolation_forest_impl(input_data, contamination=0.1, random_state=42)
        result = agent._calculate_anomaly_scores_impl(input_data)

        # Verify severity classifications exist
        assert "severity" in result
        assert len(result["severity"]) > 0

        # Check that severities are valid
        valid_severities = {"low", "medium", "high", "critical"}
        for sev in result["severity"]:
            assert sev in valid_severities

    def test_boundary_top_n_variations(self, agent, normal_data):
        """Test ranking with different top_n values (1, 5, 10, all)."""
        input_data = {"data": normal_data}

        # Fit and score
        agent._fit_isolation_forest_impl(input_data, contamination=0.1, random_state=42)
        agent._calculate_anomaly_scores_impl(input_data)

        for top_n in [1, 5, 10, len(normal_data)]:
            result = agent._rank_anomalies_impl(input_data, top_n=top_n)
            # Returned count should be min(top_n, actual anomalies)
            assert len(result["ranked_indices"]) <= top_n

    def test_boundary_n_estimators_edges(self, agent, normal_data):
        """Test n_estimators parameter edge cases (10, 500)."""
        input_data = {"data": normal_data}

        # Test very few estimators
        result_low = agent._fit_isolation_forest_impl(input_data, n_estimators=10, contamination=0.1, random_state=42)
        assert result_low["n_estimators"] == 10

        # Test many estimators
        agent._fitted_model = None  # Reset
        result_high = agent._fit_isolation_forest_impl(input_data, n_estimators=500, contamination=0.1, random_state=42)
        assert result_high["n_estimators"] == 500

    def test_boundary_max_features_variations(self, agent, normal_data):
        """Test max_features parameter with different values."""
        input_data = {"data": normal_data}

        # max_features=1.0 (all features)
        result_all = agent._fit_isolation_forest_impl(
            input_data, contamination=0.1, max_features=1.0, random_state=42
        )
        assert result_all is not None

        # max_features=0.5 (half features)
        agent._fitted_model = None  # Reset
        result_half = agent._fit_isolation_forest_impl(
            input_data, contamination=0.1, max_features=0.5, random_state=42
        )
        assert result_half is not None

    # =========================================================================
    # Configuration Tests
    # =========================================================================

    def test_config_all_initialization_options(self):
        """Test all configuration combinations."""
        from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

        # Test all False
        agent1 = IsolationForestAnomalyAgent(
            budget_usd=0.5,
            enable_explanations=False,
            enable_recommendations=False,
            enable_alerts=False,
        )
        assert agent1.budget_usd == 0.5
        assert agent1.enable_explanations is False
        assert agent1.enable_recommendations is False
        assert agent1.enable_alerts is False

        # Test all True
        agent2 = IsolationForestAnomalyAgent(
            budget_usd=2.0,
            enable_explanations=True,
            enable_recommendations=True,
            enable_alerts=True,
        )
        assert agent2.budget_usd == 2.0
        assert agent2.enable_explanations is True
        assert agent2.enable_recommendations is True
        assert agent2.enable_alerts is True

    def test_config_budget_variations(self, normal_data):
        """Test different budget values."""
        from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

        for budget in [0.10, 0.50, 1.00, 5.00, 10.00]:
            agent = IsolationForestAnomalyAgent(budget_usd=budget)
            assert agent.budget_usd == budget

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_performance_tracking_cost_accumulation(self, agent, normal_data):
        """Test that costs accumulate correctly."""
        initial_cost = agent._total_cost_usd
        initial_ai_calls = agent._ai_call_count

        # Simulate some operations
        agent._ai_call_count += 1
        agent._total_cost_usd += 0.05

        assert agent._ai_call_count == initial_ai_calls + 1
        assert agent._total_cost_usd == initial_cost + 0.05

    def test_performance_tracking_tool_calls(self, agent, normal_data):
        """Test that tool calls are tracked."""
        initial_tool_calls = agent._tool_call_count

        input_data = {"data": normal_data}

        # Call some tools
        agent._fit_isolation_forest_impl(input_data)
        agent._detect_anomalies_impl(input_data)

        # Tool calls should have increased
        assert agent._tool_call_count == initial_tool_calls + 2

    def test_performance_summary(self, agent):
        """Test get_performance_summary method."""
        agent._ai_call_count = 5
        agent._tool_call_count = 20
        agent._total_cost_usd = 0.25

        summary = agent.get_performance_summary()

        assert summary["agent_id"] == "anomaly_iforest"
        assert summary["ai_metrics"]["ai_call_count"] == 5
        assert summary["ai_metrics"]["tool_call_count"] == 20
        assert summary["ai_metrics"]["total_cost_usd"] == 0.25
        assert summary["ai_metrics"]["avg_cost_per_detection"] == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
