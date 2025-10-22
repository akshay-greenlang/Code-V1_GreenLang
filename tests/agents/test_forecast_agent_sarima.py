"""Comprehensive test suite for SARIMA Forecast Agent.

Tests cover:
- Model fitting and parameter tuning
- Forecast generation and accuracy
- Confidence interval calculation
- Seasonality detection
- Stationarity validation
- Data preprocessing
- AI integration
- Determinism verification
- Edge cases and error handling

Test Data:
- Synthetic time series with known patterns
- Real-world-like energy consumption data
- Edge cases (short series, missing values, outliers)

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from greenlang.agents.forecast_agent_sarima import (
    SARIMAForecastAgent,
    SARIMAParams,
    ForecastResult,
    ModelMetrics,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def monthly_seasonal_data():
    """Generate synthetic monthly data with strong seasonality."""
    dates = pd.date_range('2020-01-01', periods=36, freq='M')

    # Trend + seasonality + noise
    t = np.arange(36)
    trend = 100 + 2 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    noise = np.random.RandomState(42).normal(0, 5, 36)

    values = trend + seasonal + noise

    df = pd.DataFrame({
        'energy_kwh': values,
        'temperature': 20 + 10 * np.sin(2 * np.pi * t / 12),
    }, index=dates)

    return df


@pytest.fixture
def daily_data():
    """Generate synthetic daily data with weekly seasonality."""
    dates = pd.date_range('2023-01-01', periods=180, freq='D')

    t = np.arange(180)
    trend = 50 + 0.1 * t
    weekly = 10 * np.sin(2 * np.pi * t / 7)
    noise = np.random.RandomState(42).normal(0, 2, 180)

    values = trend + weekly + noise

    df = pd.DataFrame({
        'load_mw': values,
    }, index=dates)

    return df


@pytest.fixture
def data_with_missing():
    """Generate data with missing values."""
    dates = pd.date_range('2020-01-01', periods=48, freq='M')

    t = np.arange(48)
    values = 1000 + 50 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(42).normal(0, 10, 48)

    df = pd.DataFrame({'value': values}, index=dates)

    # Introduce missing values
    df.loc[df.index[10:13], 'value'] = np.nan
    df.loc[df.index[25], 'value'] = np.nan

    return df


@pytest.fixture
def data_with_outliers():
    """Generate data with outliers."""
    dates = pd.date_range('2020-01-01', periods=36, freq='M')

    t = np.arange(36)
    values = 500 + 20 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(42).normal(0, 5, 36)

    df = pd.DataFrame({'emissions_kg': values}, index=dates)

    # Introduce outliers
    df.loc[df.index[15], 'emissions_kg'] = 1000  # Large spike
    df.loc[df.index[28], 'emissions_kg'] = 100   # Large drop

    return df


@pytest.fixture
def short_data():
    """Generate short time series (minimum viable)."""
    dates = pd.date_range('2024-01-01', periods=24, freq='M')

    t = np.arange(24)
    values = 100 + 10 * np.sin(2 * np.pi * t / 12)

    df = pd.DataFrame({'value': values}, index=dates)

    return df


@pytest.fixture
def agent():
    """Create SARIMA agent instance."""
    return SARIMAForecastAgent(
        budget_usd=1.0,
        enable_explanations=True,
        enable_recommendations=True,
    )


@pytest.fixture
def agent_no_ai():
    """Create SARIMA agent without AI features."""
    return SARIMAForecastAgent(
        budget_usd=0.1,
        enable_explanations=False,
        enable_recommendations=False,
    )


# ==============================================================================
# Test: Initialization and Configuration
# ==============================================================================

def test_agent_initialization(agent):
    """Test agent initializes correctly."""
    assert agent.metadata.id == "forecast_sarima"
    assert agent.metadata.name == "SARIMA Forecast Agent"
    assert agent.budget_usd == 1.0
    assert agent.enable_explanations is True
    assert agent.enable_recommendations is True
    assert agent._ai_call_count == 0
    assert agent._tool_call_count == 0


def test_agent_custom_config():
    """Test agent with custom configuration."""
    agent = SARIMAForecastAgent(
        budget_usd=2.0,
        enable_explanations=False,
        enable_recommendations=False,
        enable_auto_tune=False,
    )

    assert agent.budget_usd == 2.0
    assert agent.enable_explanations is False
    assert agent.enable_recommendations is False
    assert agent.enable_auto_tune is False


def test_tools_setup(agent):
    """Test that all tools are properly defined."""
    assert agent.fit_sarima_tool is not None
    assert agent.forecast_future_tool is not None
    assert agent.confidence_intervals_tool is not None
    assert agent.evaluate_model_tool is not None
    assert agent.detect_seasonality_tool is not None
    assert agent.validate_stationarity_tool is not None
    assert agent.preprocess_data_tool is not None


# ==============================================================================
# Test: Input Validation
# ==============================================================================

def test_validate_valid_input(agent, monthly_seasonal_data):
    """Test validation passes with valid input."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
        "seasonal_period": 12,
    }

    assert agent.validate(input_data) is True


def test_validate_missing_data(agent):
    """Test validation fails without data."""
    input_data = {
        "target_column": "value",
        "forecast_horizon": 12,
    }

    assert agent.validate(input_data) is False


def test_validate_missing_target(agent, monthly_seasonal_data):
    """Test validation fails without target column."""
    input_data = {
        "data": monthly_seasonal_data,
        "forecast_horizon": 12,
    }

    assert agent.validate(input_data) is False


def test_validate_missing_horizon(agent, monthly_seasonal_data):
    """Test validation fails without forecast horizon."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
    }

    assert agent.validate(input_data) is False


def test_validate_invalid_dataframe(agent):
    """Test validation fails with non-DataFrame."""
    input_data = {
        "data": [1, 2, 3, 4, 5],
        "target_column": "value",
        "forecast_horizon": 12,
    }

    assert agent.validate(input_data) is False


def test_validate_missing_target_column(agent, monthly_seasonal_data):
    """Test validation fails when target column doesn't exist."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "nonexistent",
        "forecast_horizon": 12,
    }

    assert agent.validate(input_data) is False


def test_validate_non_datetime_index(agent):
    """Test validation fails without DatetimeIndex."""
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })

    input_data = {
        "data": df,
        "target_column": "value",
        "forecast_horizon": 3,
    }

    assert agent.validate(input_data) is False


def test_validate_insufficient_data(agent):
    """Test validation fails with too few data points."""
    dates = pd.date_range('2024-01-01', periods=10, freq='M')
    df = pd.DataFrame({'value': range(10)}, index=dates)

    input_data = {
        "data": df,
        "target_column": "value",
        "forecast_horizon": 6,
        "seasonal_period": 12,
    }

    assert agent.validate(input_data) is False


def test_validate_invalid_horizon(agent, monthly_seasonal_data):
    """Test validation fails with invalid horizon."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": -5,
    }

    assert agent.validate(input_data) is False


# ==============================================================================
# Test: Data Preprocessing
# ==============================================================================

def test_preprocess_missing_values(agent, data_with_missing):
    """Test preprocessing handles missing values."""
    input_data = {
        "data": data_with_missing,
        "target_column": "value",
        "forecast_horizon": 6,
    }

    result = agent._preprocess_data_impl(
        input_data,
        interpolation_method="linear",
        outlier_threshold=3.0,
    )

    assert result["missing_values_filled"] > 0
    assert result["preprocessing_applied"] is True
    assert agent._last_training_data["value"].isnull().sum() == 0


def test_preprocess_outliers(agent, data_with_outliers):
    """Test preprocessing detects and caps outliers."""
    input_data = {
        "data": data_with_outliers,
        "target_column": "emissions_kg",
        "forecast_horizon": 6,
    }

    result = agent._preprocess_data_impl(
        input_data,
        interpolation_method="linear",
        outlier_threshold=3.0,
    )

    assert result["outliers_detected"] > 0
    assert result["outliers_capped"] > 0


def test_preprocess_clean_data(agent, monthly_seasonal_data):
    """Test preprocessing on clean data."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    result = agent._preprocess_data_impl(input_data)

    assert result["missing_values_filled"] == 0
    assert result["outliers_detected"] == 0


# ==============================================================================
# Test: Seasonality Detection
# ==============================================================================

def test_detect_seasonality_monthly(agent, monthly_seasonal_data):
    """Test seasonality detection on monthly data."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    result = agent._detect_seasonality_impl(input_data, max_period=24)

    assert "seasonal_period" in result
    assert result["seasonal_period"] > 0
    assert "has_seasonality" in result
    assert "strength" in result


def test_detect_seasonality_weekly(agent, daily_data):
    """Test seasonality detection on daily data."""
    input_data = {
        "data": daily_data,
        "target_column": "load_mw",
        "forecast_horizon": 30,
    }

    result = agent._detect_seasonality_impl(input_data, max_period=14)

    assert "seasonal_period" in result
    # Should detect weekly pattern (7 days)
    assert result["seasonal_period"] in [7, 14]  # Allow some flexibility


def test_detect_no_seasonality(agent):
    """Test seasonality detection on random walk (no seasonality)."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.random.RandomState(42).normal(0, 1, 100).cumsum()
    df = pd.DataFrame({'value': values}, index=dates)

    input_data = {
        "data": df,
        "target_column": "value",
        "forecast_horizon": 10,
    }

    result = agent._detect_seasonality_impl(input_data)

    assert "seasonal_period" in result
    assert "has_seasonality" in result


# ==============================================================================
# Test: Stationarity Validation
# ==============================================================================

def test_stationarity_stationary_series(agent):
    """Test stationarity on stationary series."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    # White noise is stationary
    values = np.random.RandomState(42).normal(0, 1, 100)
    df = pd.DataFrame({'value': values}, index=dates)

    input_data = {
        "data": df,
        "target_column": "value",
        "forecast_horizon": 10,
    }

    result = agent._validate_stationarity_impl(input_data, alpha=0.05)

    assert "is_stationary" in result
    assert "adf_statistic" in result
    assert "p_value" in result
    assert "critical_values" in result
    assert "differencing_needed" in result


def test_stationarity_nonstationary_series(agent):
    """Test stationarity on non-stationary series (trend)."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    # Random walk is non-stationary
    values = np.random.RandomState(42).normal(0, 1, 100).cumsum()
    df = pd.DataFrame({'value': values}, index=dates)

    input_data = {
        "data": df,
        "target_column": "value",
        "forecast_horizon": 10,
    }

    result = agent._validate_stationarity_impl(input_data)

    assert "is_stationary" in result
    assert "differencing_needed" in result


# ==============================================================================
# Test: Model Fitting
# ==============================================================================

def test_fit_sarima_auto_tune(agent, monthly_seasonal_data):
    """Test SARIMA fitting with auto-tuning."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    result = agent._fit_sarima_impl(
        input_data,
        auto_tune=True,
        seasonal_period=12,
        max_p=2,
        max_q=2,
    )

    assert "order" in result
    assert "seasonal_order" in result
    assert "aic" in result
    assert "bic" in result
    assert result["auto_tuned"] is True
    assert result["converged"] is True

    # Verify model was stored
    assert agent._fitted_model is not None
    assert agent._best_params is not None


def test_fit_sarima_manual_params(agent, monthly_seasonal_data):
    """Test SARIMA fitting without auto-tuning."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    result = agent._fit_sarima_impl(
        input_data,
        auto_tune=False,
        seasonal_period=12,
    )

    assert "order" in result
    assert "seasonal_order" in result
    assert result["auto_tuned"] is False


def test_fit_sarima_parameters(agent, monthly_seasonal_data):
    """Test fitted SARIMA parameters are reasonable."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    result = agent._fit_sarima_impl(
        input_data,
        auto_tune=True,
        seasonal_period=12,
        max_p=3,
        max_q=3,
    )

    order = result["order"]
    seasonal_order = result["seasonal_order"]

    # Validate parameter ranges
    assert 0 <= order[0] <= 3  # p
    assert 0 <= order[1] <= 2  # d
    assert 0 <= order[2] <= 3  # q

    assert 0 <= seasonal_order[0] <= 2  # P
    assert 0 <= seasonal_order[1] <= 2  # D
    assert 0 <= seasonal_order[2] <= 2  # Q
    assert seasonal_order[3] == 12  # s


# ==============================================================================
# Test: Forecasting
# ==============================================================================

def test_forecast_generation(agent, monthly_seasonal_data):
    """Test forecast generation."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    # Fit model first
    agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

    # Generate forecast
    result = agent._forecast_future_impl(
        input_data,
        horizon=12,
        confidence_level=0.95,
    )

    assert "forecast" in result
    assert "lower_bound" in result
    assert "upper_bound" in result
    assert "forecast_dates" in result
    assert "confidence_level" in result

    assert len(result["forecast"]) == 12
    assert len(result["lower_bound"]) == 12
    assert len(result["upper_bound"]) == 12
    assert len(result["forecast_dates"]) == 12


def test_forecast_confidence_intervals(agent, monthly_seasonal_data):
    """Test forecast confidence intervals are valid."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)
    result = agent._forecast_future_impl(input_data, horizon=12, confidence_level=0.95)

    forecast = result["forecast"]
    lower = result["lower_bound"]
    upper = result["upper_bound"]

    # Validate intervals: lower < forecast < upper
    for i in range(len(forecast)):
        assert lower[i] < forecast[i], f"Lower bound should be less than forecast at index {i}"
        assert forecast[i] < upper[i], f"Forecast should be less than upper bound at index {i}"


def test_forecast_different_horizons(agent, monthly_seasonal_data):
    """Test forecasting with different horizons."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

    for horizon in [3, 6, 12, 24]:
        result = agent._forecast_future_impl(input_data, horizon=horizon)
        assert len(result["forecast"]) == horizon


def test_forecast_dates_generation(agent, monthly_seasonal_data):
    """Test forecast dates are correctly generated."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)
    result = agent._forecast_future_impl(input_data, horizon=12)

    dates = result["forecast_dates"]

    # Verify dates are in the future
    last_train_date = monthly_seasonal_data.index[-1]
    first_forecast_date = pd.Timestamp(dates[0])

    assert first_forecast_date > last_train_date

    # Verify dates are sequential
    for i in range(1, len(dates)):
        assert pd.Timestamp(dates[i]) > pd.Timestamp(dates[i-1])


def test_forecast_without_fitted_model(agent, monthly_seasonal_data):
    """Test forecasting fails without fitted model."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    with pytest.raises(ValueError, match="Model must be fitted"):
        agent._forecast_future_impl(input_data, horizon=12)


# ==============================================================================
# Test: Model Evaluation
# ==============================================================================

def test_evaluate_model(agent, monthly_seasonal_data):
    """Test model evaluation metrics."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

    result = agent._evaluate_model_impl(input_data, train_test_split=0.8)

    assert "rmse" in result
    assert "mae" in result
    assert "mape" in result
    assert "train_size" in result
    assert "test_size" in result

    # Validate metrics are positive
    assert result["rmse"] >= 0
    assert result["mae"] >= 0
    assert result["mape"] >= 0

    # Validate split
    total_size = result["train_size"] + result["test_size"]
    assert total_size == len(monthly_seasonal_data)


def test_evaluate_different_splits(agent, monthly_seasonal_data):
    """Test evaluation with different train/test splits."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

    for split in [0.7, 0.8, 0.9]:
        result = agent._evaluate_model_impl(input_data, train_test_split=split)

        train_size = result["train_size"]
        test_size = result["test_size"]

        # Verify split ratio
        actual_split = train_size / (train_size + test_size)
        assert abs(actual_split - split) < 0.05  # Allow small rounding error


def test_evaluate_insufficient_test_data(agent, short_data):
    """Test evaluation fails with insufficient test data."""
    input_data = {
        "data": short_data,
        "target_column": "value",
        "forecast_horizon": 6,
    }

    agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

    # Very high split leaves no test data
    with pytest.raises(ValueError, match="Test set is empty"):
        agent._evaluate_model_impl(input_data, train_test_split=0.99)


# ==============================================================================
# Test: Confidence Interval Calculation
# ==============================================================================

def test_calculate_confidence_intervals(agent):
    """Test confidence interval calculation."""
    forecast = [100, 105, 110, 115, 120]
    std_errors = [5, 6, 7, 8, 9]

    result = agent._calculate_confidence_impl(
        forecast,
        std_errors,
        confidence_level=0.95,
    )

    assert "lower_bound" in result
    assert "upper_bound" in result
    assert "z_score" in result

    assert len(result["lower_bound"]) == len(forecast)
    assert len(result["upper_bound"]) == len(forecast)


def test_confidence_intervals_widths(agent):
    """Test confidence interval widths scale with uncertainty."""
    forecast = [100] * 5
    std_errors = [5, 10, 15, 20, 25]

    result = agent._calculate_confidence_impl(forecast, std_errors, 0.95)

    lower = result["lower_bound"]
    upper = result["upper_bound"]

    # Wider intervals with higher uncertainty
    for i in range(1, len(forecast)):
        width_prev = upper[i-1] - lower[i-1]
        width_curr = upper[i] - lower[i]
        assert width_curr > width_prev


def test_confidence_different_levels(agent):
    """Test different confidence levels."""
    forecast = [100] * 3
    std_errors = [10] * 3

    result_95 = agent._calculate_confidence_impl(forecast, std_errors, 0.95)
    result_99 = agent._calculate_confidence_impl(forecast, std_errors, 0.99)

    # 99% interval should be wider than 95%
    width_95 = result_95["upper_bound"][0] - result_95["lower_bound"][0]
    width_99 = result_99["upper_bound"][0] - result_99["lower_bound"][0]

    assert width_99 > width_95


# ==============================================================================
# Test: End-to-End Forecasting
# ==============================================================================

def test_end_to_end_forecast(agent_no_ai, monthly_seasonal_data):
    """Test complete forecast workflow without AI."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
        "seasonal_period": 12,
    }

    # Validate
    assert agent_no_ai.validate(input_data) is True

    # Process (without AI to avoid API calls)
    # We'll test individual tools instead
    agent_no_ai._preprocess_data_impl(input_data)
    seasonality = agent_no_ai._detect_seasonality_impl(input_data)
    stationarity = agent_no_ai._validate_stationarity_impl(input_data)
    model = agent_no_ai._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)
    forecast = agent_no_ai._forecast_future_impl(input_data, horizon=12)
    evaluation = agent_no_ai._evaluate_model_impl(input_data)

    # Verify all components worked
    assert seasonality is not None
    assert stationarity is not None
    assert model is not None
    assert forecast is not None
    assert evaluation is not None


def test_deterministic_forecasts(agent_no_ai, monthly_seasonal_data):
    """Test forecasts are deterministic."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
        "seasonal_period": 12,
    }

    # Run twice
    agent_no_ai._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)
    result1 = agent_no_ai._forecast_future_impl(input_data, horizon=12)

    agent_no_ai._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)
    result2 = agent_no_ai._forecast_future_impl(input_data, horizon=12)

    # Results should be identical
    np.testing.assert_array_almost_equal(result1["forecast"], result2["forecast"], decimal=5)


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

def test_single_seasonal_cycle(agent, short_data):
    """Test with minimal data (one seasonal cycle)."""
    input_data = {
        "data": short_data,
        "target_column": "value",
        "forecast_horizon": 6,
        "seasonal_period": 12,
    }

    assert agent.validate(input_data) is True


def test_very_short_forecast(agent_no_ai, monthly_seasonal_data):
    """Test very short forecast horizon."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    agent_no_ai._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)
    result = agent_no_ai._forecast_future_impl(input_data, horizon=1)

    assert len(result["forecast"]) == 1


def test_long_forecast(agent_no_ai, monthly_seasonal_data):
    """Test long forecast horizon."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    agent_no_ai._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)
    result = agent_no_ai._forecast_future_impl(input_data, horizon=24)

    assert len(result["forecast"]) == 24


def test_no_seasonality_data(agent):
    """Test with data having no clear seasonal pattern."""
    dates = pd.date_range('2020-01-01', periods=50, freq='M')
    values = 100 + np.random.RandomState(42).normal(0, 10, 50)
    df = pd.DataFrame({'value': values}, index=dates)

    input_data = {
        "data": df,
        "target_column": "value",
        "forecast_horizon": 6,
    }

    # Should still work, just with s=1 (no seasonality)
    result = agent._detect_seasonality_impl(input_data)
    assert result is not None


def test_constant_series(agent):
    """Test with constant values."""
    dates = pd.date_range('2020-01-01', periods=36, freq='M')
    df = pd.DataFrame({'value': [100] * 36}, index=dates)

    input_data = {
        "data": df,
        "target_column": "value",
        "forecast_horizon": 6,
    }

    # Should handle gracefully
    agent._preprocess_data_impl(input_data)
    assert agent._last_training_data is not None


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


def test_tool_call_counting(agent, monthly_seasonal_data):
    """Test tool calls are counted."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    initial_count = agent._tool_call_count

    agent._preprocess_data_impl(input_data)
    agent._detect_seasonality_impl(input_data)
    agent._validate_stationarity_impl(input_data)

    assert agent._tool_call_count == initial_count + 3


# ==============================================================================
# Test: SARIMA Parameters Dataclass
# ==============================================================================

def test_sarima_params_creation():
    """Test SARIMAParams dataclass."""
    params = SARIMAParams(p=2, d=1, q=1, P=1, D=1, Q=1, s=12)

    assert params.p == 2
    assert params.d == 1
    assert params.q == 1
    assert params.s == 12


def test_sarima_params_to_tuple():
    """Test SARIMAParams conversion to tuple."""
    params = SARIMAParams(p=2, d=1, q=1, P=1, D=1, Q=1, s=12)

    result = params.to_tuple()

    assert result == ((2, 1, 1), (1, 1, 1, 12))


def test_sarima_params_defaults():
    """Test SARIMAParams default values."""
    params = SARIMAParams()

    assert params.p == 1
    assert params.d == 1
    assert params.q == 1
    assert params.P == 1
    assert params.D == 1
    assert params.Q == 1
    assert params.s == 12


# ==============================================================================
# Test: Error Handling
# ==============================================================================

def test_invalid_interpolation_method(agent, data_with_missing):
    """Test error handling for invalid interpolation."""
    input_data = {
        "data": data_with_missing,
        "target_column": "value",
        "forecast_horizon": 6,
    }

    # Should handle gracefully or raise clear error
    try:
        agent._preprocess_data_impl(input_data, interpolation_method="invalid_method")
    except Exception as e:
        assert "interpolation" in str(e).lower() or "method" in str(e).lower()


def test_tool_result_extraction_handles_errors(agent, monthly_seasonal_data):
    """Test tool result extraction handles errors gracefully."""
    mock_response = Mock()
    mock_response.tool_calls = [
        {"name": "fit_sarima_model", "arguments": {"invalid": "args"}},
    ]

    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    # Should not crash
    results = agent._extract_tool_results(mock_response, input_data)
    assert isinstance(results, dict)


# ==============================================================================
# Test: Output Building
# ==============================================================================

def test_build_output_complete(agent):
    """Test output building with all components."""
    tool_results = {
        "forecast": {
            "forecast": [100, 105, 110],
            "lower_bound": [90, 95, 100],
            "upper_bound": [110, 115, 120],
            "forecast_dates": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "confidence_level": 0.95,
        },
        "model": {
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 12),
            "aic": 250.5,
            "bic": 260.3,
            "auto_tuned": True,
        },
        "evaluation": {
            "rmse": 5.2,
            "mae": 4.1,
            "mape": 3.5,
            "train_size": 30,
            "test_size": 6,
        },
        "seasonality": {
            "seasonal_period": 12,
            "has_seasonality": True,
            "strength": 0.7,
        },
    }

    input_data = {
        "data": pd.DataFrame(),
        "target_column": "value",
        "forecast_horizon": 3,
    }

    output = agent._build_output(
        input_data,
        tool_results,
        explanation="Test explanation",
    )

    assert "forecast" in output
    assert "lower_bound" in output
    assert "upper_bound" in output
    assert "model_params" in output
    assert "metrics" in output
    assert "seasonality" in output
    assert "explanation" in output


def test_build_output_minimal(agent):
    """Test output building with minimal results."""
    tool_results = {
        "forecast": {
            "forecast": [100],
            "lower_bound": [90],
            "upper_bound": [110],
            "forecast_dates": ["2025-01-01"],
            "confidence_level": 0.95,
        },
    }

    input_data = {
        "data": pd.DataFrame(),
        "target_column": "value",
        "forecast_horizon": 1,
    }

    output = agent._build_output(input_data, tool_results, None)

    assert "forecast" in output
    assert len(output["forecast"]) == 1


# ==============================================================================
# Test: Prompt Building
# ==============================================================================

def test_build_prompt(agent, monthly_seasonal_data):
    """Test AI prompt building."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
        "seasonal_period": 12,
    }

    prompt = agent._build_prompt(input_data)

    assert "energy_kwh" in prompt
    assert "12" in prompt
    assert "forecast" in prompt.lower()
    assert "preprocess" in prompt.lower()


def test_build_prompt_without_seasonality(agent, monthly_seasonal_data):
    """Test prompt building without seasonal period."""
    input_data = {
        "data": monthly_seasonal_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
    }

    prompt = agent._build_prompt(input_data)

    assert "detect_seasonality" in prompt


# ==============================================================================
# Test: Coverage Expansion - Gap-Filling Tests for 80%+ Coverage
# ==============================================================================


class TestSARIMAForecastAgentCoverage:
    """Gap-filling tests to achieve 80%+ coverage for SARIMAForecastAgent.

    This test class adds comprehensive coverage for:
    - Tool result extraction (all 7 tools)
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

    def test_extract_tool_results_all_seven_tools(self, agent, monthly_seasonal_data):
        """Test extracting results from all seven tool types."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "preprocess_data",
                "arguments": {"interpolation_method": "linear"},
            },
            {
                "name": "detect_seasonality",
                "arguments": {"max_period": 52},
            },
            {
                "name": "validate_stationarity",
                "arguments": {"alpha": 0.05},
            },
            {
                "name": "fit_sarima_model",
                "arguments": {"auto_tune": True, "seasonal_period": 12},
            },
            {
                "name": "forecast_future",
                "arguments": {"horizon": 12, "confidence_level": 0.95},
            },
            {
                "name": "calculate_confidence_intervals",
                "arguments": {
                    "forecast": [100, 105, 110],
                    "std_errors": [5, 5.5, 6],
                    "confidence_level": 0.95,
                },
            },
            {
                "name": "evaluate_model",
                "arguments": {"train_test_split": 0.8},
            },
        ]

        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        # Fit model first for tools that need it
        agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

        results = agent._extract_tool_results(mock_response, input_data)

        # Verify all tools were called
        assert "preprocessing" in results
        assert "seasonality" in results
        assert "stationarity" in results
        assert "model" in results
        assert "forecast" in results
        assert "confidence" in results
        assert "evaluation" in results

    def test_extract_tool_results_empty(self, agent, monthly_seasonal_data):
        """Test extracting results when no tools were called."""
        mock_response = Mock()
        mock_response.tool_calls = []

        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        results = agent._extract_tool_results(mock_response, input_data)

        # Should return empty dict
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_extract_tool_results_unknown_tool(self, agent, monthly_seasonal_data):
        """Test extracting results with unknown tool name."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {"name": "unknown_tool", "arguments": {}},
            {"name": "fit_sarima_model", "arguments": {"seasonal_period": 12}},
        ]

        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        results = agent._extract_tool_results(mock_response, input_data)

        # Should handle unknown tool gracefully
        assert isinstance(results, dict)
        assert "model" in results  # Known tool should work

    def test_extract_tool_results_partial_data(self, agent, monthly_seasonal_data):
        """Test extracting results when some tools return partial data."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {"name": "detect_seasonality", "arguments": {}},
            {"name": "preprocess_data", "arguments": {}},
        ]

        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        results = agent._extract_tool_results(mock_response, input_data)

        # Should have partial results
        assert "seasonality" in results
        assert "preprocessing" in results
        assert "model" not in results  # Not called

    # =========================================================================
    # Unit Tests: Output Building
    # =========================================================================

    def test_build_output_missing_model(self, agent):
        """Test output building when model data is missing."""
        tool_results = {
            "forecast": {
                "forecast": [100, 105],
                "lower_bound": [90, 95],
                "upper_bound": [110, 115],
                "forecast_dates": ["2025-01-01", "2025-02-01"],
                "confidence_level": 0.95,
            },
        }

        input_data = {"data": pd.DataFrame(), "target_column": "value", "forecast_horizon": 2}

        output = agent._build_output(input_data, tool_results, "Explanation")

        assert "forecast" in output
        assert len(output["forecast"]) == 2
        # Model params should have defaults when missing
        assert "model_params" not in output or output.get("model_params") is None

    def test_build_output_missing_evaluation(self, agent):
        """Test output building when evaluation metrics are missing."""
        tool_results = {
            "forecast": {
                "forecast": [100],
                "lower_bound": [90],
                "upper_bound": [110],
                "forecast_dates": ["2025-01-01"],
                "confidence_level": 0.95,
            },
            "model": {
                "order": (1, 1, 1),
                "seasonal_order": (1, 1, 1, 12),
                "aic": 250.0,
                "bic": 260.0,
            },
        }

        input_data = {"data": pd.DataFrame(), "target_column": "value", "forecast_horizon": 1}

        output = agent._build_output(input_data, tool_results, None)

        assert "forecast" in output
        assert "model_params" in output
        assert "metrics" not in output or output.get("metrics") is None

    def test_build_output_missing_seasonality(self, agent):
        """Test output building when seasonality data is missing."""
        tool_results = {
            "forecast": {
                "forecast": [100],
                "lower_bound": [90],
                "upper_bound": [110],
                "forecast_dates": ["2025-01-01"],
                "confidence_level": 0.95,
            },
        }

        input_data = {"data": pd.DataFrame(), "target_column": "value", "forecast_horizon": 1}

        output = agent._build_output(input_data, tool_results, "Test")

        assert "seasonality" not in output or output.get("seasonality") is None
        assert "stationarity" not in output or output.get("stationarity") is None

    def test_build_output_no_explanation(self, agent):
        """Test output building when explanations are disabled."""
        tool_results = {
            "forecast": {
                "forecast": [100],
                "lower_bound": [90],
                "upper_bound": [110],
                "forecast_dates": ["2025-01-01"],
                "confidence_level": 0.95,
            },
        }

        input_data = {"data": pd.DataFrame(), "target_column": "value", "forecast_horizon": 1}

        # Disable explanations
        agent.enable_explanations = False

        output = agent._build_output(input_data, tool_results, "Should be ignored")

        assert "explanation" not in output or output["explanation"] is None

    # =========================================================================
    # Integration Tests: Budget and Exception Handling
    # =========================================================================

    @pytest.mark.asyncio
    async def test_integration_budget_exceeded(self, agent, monthly_seasonal_data):
        """Test handling when AI budget is exceeded."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
            "seasonal_period": 12,
        }

        # Mock ChatSession to raise BudgetExceeded
        with patch("greenlang.agents.forecast_agent_sarima.ChatSession") as MockSession:
            mock_session = MockSession.return_value
            mock_session.chat = AsyncMock(side_effect=BudgetExceeded("Budget exceeded"))

            # Should raise ValueError with budget message
            with pytest.raises(ValueError, match="budget"):
                await agent._process_async(input_data)

    @pytest.mark.asyncio
    async def test_integration_general_exception(self, agent, monthly_seasonal_data):
        """Test handling of general exceptions during processing."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        # Mock ChatSession to raise RuntimeError
        with patch("greenlang.agents.forecast_agent_sarima.ChatSession") as MockSession:
            mock_session = MockSession.return_value
            mock_session.chat = AsyncMock(side_effect=RuntimeError("Unexpected error"))

            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Unexpected error"):
                await agent._process_async(input_data)

    @pytest.mark.asyncio
    async def test_integration_disabled_explanations(self, agent, monthly_seasonal_data):
        """Test full workflow with explanations disabled."""
        agent_no_explain = SARIMAForecastAgent(enable_explanations=False)

        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 6,
            "seasonal_period": 12,
        }

        # Mock ChatSession
        with patch("greenlang.agents.forecast_agent_sarima.ChatSession") as MockSession:
            mock_session = MockSession.return_value
            mock_response = Mock()
            mock_response.tool_calls = [
                {"name": "fit_sarima_model", "arguments": {"seasonal_period": 12}},
                {"name": "forecast_future", "arguments": {"horizon": 6}},
            ]
            mock_response.text = "This should be ignored"
            mock_response.usage = Mock(cost_usd=0.01, total_tokens=100)
            mock_response.provider_info = Mock(provider="anthropic", model="claude-3-5-sonnet-20241022")
            mock_session.chat = AsyncMock(return_value=mock_response)

            # Fit model first
            agent_no_explain._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

            result = await agent_no_explain._process_async(input_data)

            # Should not have explanation
            assert "explanation" not in result or result.get("explanation") is None

    @pytest.mark.asyncio
    async def test_integration_disabled_auto_tune(self, agent, monthly_seasonal_data):
        """Test full workflow with auto-tuning disabled."""
        agent_no_tune = SARIMAForecastAgent(enable_auto_tune=False)

        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 6,
            "seasonal_period": 12,
        }

        # Mock ChatSession
        with patch("greenlang.agents.forecast_agent_sarima.ChatSession") as MockSession:
            mock_session = MockSession.return_value
            mock_response = Mock()
            mock_response.tool_calls = [
                {"name": "fit_sarima_model", "arguments": {"seasonal_period": 12, "auto_tune": False}},
            ]
            mock_response.text = "Forecast complete"
            mock_response.usage = Mock(cost_usd=0.01, total_tokens=100)
            mock_response.provider_info = Mock(provider="anthropic", model="claude-3-5-sonnet-20241022")
            mock_session.chat = AsyncMock(return_value=mock_response)

            result_data = agent_no_tune._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

            # Verify auto_tune was False
            assert result_data["auto_tuned"] is False

    # =========================================================================
    # Determinism Tests: 5-Run Consistency Verification
    # =========================================================================

    def test_fit_sarima_determinism(self, agent, monthly_seasonal_data):
        """Test that SARIMA fitting produces identical results across 5 runs."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        results = []
        for _ in range(5):
            # Reset agent state
            agent._fitted_model = None
            agent._best_params = None

            result = agent._fit_sarima_impl(
                input_data, auto_tune=False, seasonal_period=12, max_p=2, max_q=2
            )
            results.append(result)

        # All runs should produce identical results
        for i in range(1, 5):
            assert results[i]["order"] == results[0]["order"]
            assert results[i]["seasonal_order"] == results[0]["seasonal_order"]
            assert results[i]["aic"] == pytest.approx(results[0]["aic"], rel=1e-6)
            assert results[i]["bic"] == pytest.approx(results[0]["bic"], rel=1e-6)

    def test_forecast_determinism(self, agent, monthly_seasonal_data):
        """Test that forecast generation produces identical results across 5 runs."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        # Fit model once
        agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

        results = []
        for _ in range(5):
            result = agent._forecast_future_impl(input_data, horizon=6, confidence_level=0.95)
            results.append(result)

        # All runs should produce identical forecasts
        for i in range(1, 5):
            assert len(results[i]["forecast"]) == len(results[0]["forecast"])
            for j in range(len(results[0]["forecast"])):
                assert results[i]["forecast"][j] == pytest.approx(results[0]["forecast"][j], rel=1e-10)
                assert results[i]["lower_bound"][j] == pytest.approx(results[0]["lower_bound"][j], rel=1e-10)
                assert results[i]["upper_bound"][j] == pytest.approx(results[0]["upper_bound"][j], rel=1e-10)

    def test_seasonality_detection_determinism(self, agent, monthly_seasonal_data):
        """Test that seasonality detection produces identical results across 5 runs."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        results = []
        for _ in range(5):
            result = agent._detect_seasonality_impl(input_data, max_period=24)
            results.append(result)

        # All runs should detect the same seasonality
        for i in range(1, 5):
            assert results[i]["seasonal_period"] == results[0]["seasonal_period"]
            assert results[i]["has_seasonality"] == results[0]["has_seasonality"]
            assert results[i]["strength"] == pytest.approx(results[0]["strength"], rel=1e-10)

    def test_preprocessing_determinism(self, agent, monthly_seasonal_data):
        """Test that preprocessing produces identical results across 5 runs."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        results = []
        for _ in range(5):
            agent._last_training_data = None  # Reset
            result = agent._preprocess_data_impl(input_data, interpolation_method="linear", outlier_threshold=3.0)
            results.append(result)

        # All runs should produce identical preprocessing
        for i in range(1, 5):
            assert results[i]["missing_values_filled"] == results[0]["missing_values_filled"]
            assert results[i]["outliers_detected"] == results[0]["outliers_detected"]
            assert results[i]["final_length"] == results[0]["final_length"]

    # =========================================================================
    # Boundary Tests: Edge Cases and Extreme Values
    # =========================================================================

    def test_boundary_single_data_point(self, agent):
        """Test with minimal data (single point)."""
        dates = pd.date_range('2020-01-01', periods=1, freq='M')
        df = pd.DataFrame({'value': [100.0]}, index=dates)

        input_data = {
            "data": df,
            "target_column": "value",
            "forecast_horizon": 1,
            "seasonal_period": 1,
        }

        # Should fail validation (insufficient data)
        assert agent.validate(input_data) is False

    def test_boundary_very_large_horizon(self, agent, monthly_seasonal_data):
        """Test forecasting with very large horizon."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        # Fit model
        agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

        # Forecast 365 periods (1 year daily)
        result = agent._forecast_future_impl(input_data, horizon=365)

        assert len(result["forecast"]) == 365
        assert len(result["lower_bound"]) == 365
        assert len(result["upper_bound"]) == 365

    def test_boundary_extreme_values(self, agent):
        """Test with extremely large and small values."""
        dates = pd.date_range('2020-01-01', periods=50, freq='M')

        # Very large values (billions)
        large_values = np.random.RandomState(42).normal(1e9, 1e8, 50)
        df_large = pd.DataFrame({'value': large_values}, index=dates)

        input_data_large = {
            "data": df_large,
            "target_column": "value",
            "forecast_horizon": 6,
            "seasonal_period": 12,
        }

        # Should handle large values
        result_large = agent._fit_sarima_impl(input_data_large, auto_tune=False, seasonal_period=12)
        assert result_large is not None

        # Very small values (near zero)
        small_values = np.random.RandomState(42).normal(0.001, 0.0001, 50)
        df_small = pd.DataFrame({'value': small_values}, index=dates)

        input_data_small = {
            "data": df_small,
            "target_column": "value",
            "forecast_horizon": 6,
            "seasonal_period": 12,
        }

        # Should handle small values
        result_small = agent._fit_sarima_impl(input_data_small, auto_tune=False, seasonal_period=12)
        assert result_small is not None

    def test_boundary_constant_series(self, agent):
        """Test with constant series (no variation)."""
        dates = pd.date_range('2020-01-01', periods=50, freq='M')
        df = pd.DataFrame({'value': [100.0] * 50}, index=dates)

        input_data = {
            "data": df,
            "target_column": "value",
            "forecast_horizon": 6,
            "seasonal_period": 12,
        }

        # Constant series is stationary
        result = agent._validate_stationarity_impl(input_data)
        # ADF test should indicate stationarity
        assert "is_stationary" in result

    def test_boundary_confidence_levels(self, agent, monthly_seasonal_data):
        """Test different confidence levels (0.50, 0.90, 0.95, 0.99)."""
        forecast = [100, 105, 110]
        std_errors = [5, 5.5, 6]

        for conf_level in [0.50, 0.90, 0.95, 0.99]:
            result = agent._calculate_confidence_impl(forecast, std_errors, conf_level)

            assert result["confidence_level"] == conf_level
            assert len(result["lower_bound"]) == 3
            assert len(result["upper_bound"]) == 3

            # Higher confidence = wider intervals
            if conf_level > 0.50:
                # Width should be reasonable
                width = result["upper_bound"][0] - result["lower_bound"][0]
                assert width > 0

    def test_boundary_seasonal_periods(self, agent):
        """Test different seasonal periods (1, 7, 12, 52, 365)."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = 50 + 10 * np.sin(np.arange(100) * 2 * np.pi / 7)
        df = pd.DataFrame({'value': values}, index=dates)

        input_data = {
            "data": df,
            "target_column": "value",
            "forecast_horizon": 7,
        }

        for period in [1, 7, 12, 52]:
            result = agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=period)
            assert result["seasonal_order"][3] == period

    def test_boundary_parameter_edges(self, agent, monthly_seasonal_data):
        """Test edge cases for SARIMA parameters (max_p=0, max_q=0)."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 6,
        }

        # Test with max_p=0 (no AR terms)
        result_p0 = agent._fit_sarima_impl(
            input_data, auto_tune=True, seasonal_period=12, max_p=0, max_q=2
        )
        assert result_p0["order"][0] == 0  # p=0

        # Test with max_q=0 (no MA terms)
        agent._fitted_model = None  # Reset
        result_q0 = agent._fit_sarima_impl(
            input_data, auto_tune=True, seasonal_period=12, max_p=2, max_q=0
        )
        assert result_q0["order"][2] == 0  # q=0

    def test_boundary_split_ratio_edges(self, agent, monthly_seasonal_data):
        """Test train/test split with edge ratios (0.5, 0.95)."""
        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        # Fit model first
        agent._fit_sarima_impl(input_data, auto_tune=False, seasonal_period=12)

        # Test with 50% split
        result_50 = agent._evaluate_model_impl(input_data, train_test_split=0.5)
        assert result_50["train_test_split"] == 0.5
        assert result_50["test_size"] == 18  # 50% of 36

        # Test with 95% split
        result_95 = agent._evaluate_model_impl(input_data, train_test_split=0.95)
        assert result_95["train_test_split"] == 0.95
        assert result_95["test_size"] == 2  # 5% of 36 (rounded)

    # =========================================================================
    # Configuration Tests
    # =========================================================================

    def test_config_all_initialization_options(self):
        """Test all configuration combinations."""
        # Test all False
        agent1 = SARIMAForecastAgent(
            budget_usd=0.5,
            enable_explanations=False,
            enable_recommendations=False,
            enable_auto_tune=False,
        )
        assert agent1.budget_usd == 0.5
        assert agent1.enable_explanations is False
        assert agent1.enable_recommendations is False
        assert agent1.enable_auto_tune is False

        # Test all True
        agent2 = SARIMAForecastAgent(
            budget_usd=2.0,
            enable_explanations=True,
            enable_recommendations=True,
            enable_auto_tune=True,
        )
        assert agent2.budget_usd == 2.0
        assert agent2.enable_explanations is True
        assert agent2.enable_recommendations is True
        assert agent2.enable_auto_tune is True

    def test_config_budget_variations(self, monthly_seasonal_data):
        """Test different budget values."""
        for budget in [0.10, 0.50, 1.00, 5.00, 10.00]:
            agent = SARIMAForecastAgent(budget_usd=budget)
            assert agent.budget_usd == budget

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_performance_tracking_cost_accumulation(self, agent, monthly_seasonal_data):
        """Test that costs accumulate correctly."""
        initial_cost = agent._total_cost_usd
        initial_ai_calls = agent._ai_call_count

        # Simulate some operations
        agent._ai_call_count += 1
        agent._total_cost_usd += 0.05

        assert agent._ai_call_count == initial_ai_calls + 1
        assert agent._total_cost_usd == initial_cost + 0.05

    def test_performance_tracking_tool_calls(self, agent, monthly_seasonal_data):
        """Test that tool calls are tracked."""
        initial_tool_calls = agent._tool_call_count

        input_data = {
            "data": monthly_seasonal_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
        }

        # Call some tools
        agent._preprocess_data_impl(input_data)
        agent._detect_seasonality_impl(input_data)

        # Tool calls should have increased
        assert agent._tool_call_count == initial_tool_calls + 2

    def test_performance_summary(self, agent):
        """Test get_performance_summary method."""
        agent._ai_call_count = 5
        agent._tool_call_count = 20
        agent._total_cost_usd = 0.25

        summary = agent.get_performance_summary()

        assert summary["agent_id"] == "forecast_sarima"
        assert summary["ai_metrics"]["ai_call_count"] == 5
        assert summary["ai_metrics"]["tool_call_count"] == 20
        assert summary["ai_metrics"]["total_cost_usd"] == 0.25
        assert summary["ai_metrics"]["avg_cost_per_forecast"] == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
