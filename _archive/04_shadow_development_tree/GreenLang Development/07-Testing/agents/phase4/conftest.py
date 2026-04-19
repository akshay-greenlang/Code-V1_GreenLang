# -*- coding: utf-8 -*-
"""
Pytest fixtures for Phase 4 agent tests.

Provides mocks for:
- RAGEngine
- ChatSession
- Tool execution
- Common test data for InsightAgent pattern
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timedelta


@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for testing."""
    engine = Mock()

    # Mock query result with relevant chunks
    mock_result = Mock()
    mock_result.chunks = [
        Mock(text="Historical anomaly case: sensor drift caused by calibration issue"),
        Mock(text="Forecasting pattern: strong seasonality with 12-month period"),
        Mock(text="Benchmark insight: top performers achieve 20% reduction through efficiency"),
        Mock(text="Report best practice: structure narrative by framework requirements")
    ]
    mock_result.relevance_scores = [0.92, 0.88, 0.85, 0.81]
    mock_result.search_time_ms = 120

    engine.query = AsyncMock(return_value=mock_result)

    return engine


@pytest.fixture
def mock_chat_session():
    """Mock ChatSession for testing."""
    session = Mock()

    # Mock chat response for insight generation
    mock_response = Mock()
    mock_response.text = """
    Based on the deterministic analysis, here are the key insights:

    ## Root Cause Analysis
    The detected anomalies correlate strongly with:
    1. Sensor calibration drift (85% confidence)
    2. Extreme weather conditions (72% confidence)
    3. Recent equipment maintenance (68% confidence)

    ## Recommendations
    1. Immediate sensor recalibration required
    2. Review maintenance schedules
    3. Implement weather-based adjustment protocols

    ## Evidence Summary
    Multiple data sources confirm sensor degradation as primary factor.
    Historical patterns show similar anomalies post-maintenance events.
    """
    mock_response.tool_calls = []
    mock_response.provider_info = {"model": "gpt-4", "provider": "openai"}
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 1800
    mock_response.usage.cost_usd = 0.08

    session.chat = AsyncMock(return_value=mock_response)

    return session


@pytest.fixture
def sample_anomaly_data():
    """Sample data for anomaly investigation testing."""
    np.random.seed(42)
    n_samples = 500

    # Normal data
    energy = np.random.normal(100, 10, n_samples)
    temp = np.random.normal(20, 2, n_samples)

    # Inject anomalies
    anomaly_indices = [50, 150, 300, 420]
    energy[anomaly_indices] = [500, 450, 480, 510]
    temp[anomaly_indices] = [35, 38, 36, 40]

    df = pd.DataFrame({
        "energy_kwh": energy,
        "temperature_c": temp,
        "humidity_pct": np.random.normal(60, 10, n_samples),
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H")
    })

    return df


@pytest.fixture
def sample_forecast_data():
    """Sample data for forecast explanation testing."""
    np.random.seed(42)
    n_samples = 48  # 4 years monthly

    dates = pd.date_range("2021-01-01", periods=n_samples, freq="M")
    trend = np.linspace(100, 150, n_samples)
    seasonal = 20 * np.sin(np.arange(n_samples) * 2 * np.pi / 12)
    noise = np.random.normal(0, 5, n_samples)

    df = pd.DataFrame({
        "energy_kwh": trend + seasonal + noise,
        "temperature_c": 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 12)
    }, index=dates)

    return df


@pytest.fixture
def sample_benchmark_context():
    """Sample context for benchmark testing."""
    return {
        "building_type": "commercial_office",
        "total_emissions_kg": 250000,
        "building_area": 5000,
        "period_months": 12,
        "region": "California",
        "industry": "Technology",
        "building_age": 15
    }


@pytest.fixture
def sample_report_context():
    """Sample context for report narrative testing."""
    return {
        "framework": "TCFD",
        "carbon_data": {
            "total_co2e_tons": 45.5,
            "total_co2e_kg": 45500,
            "emissions_breakdown": [
                {"source": "Electricity", "co2e_tons": 25.0, "percentage": 54.9},
                {"source": "Natural Gas", "co2e_tons": 15.0, "percentage": 33.0},
                {"source": "Transportation", "co2e_tons": 5.5, "percentage": 12.1}
            ],
            "carbon_intensity": {
                "kg_per_sqft": 9.1
            }
        },
        "building_info": {
            "type": "commercial_office",
            "area": 5000,
            "location": "California"
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
    }


@pytest.fixture
def assert_insight_agent_result():
    """Helper to assert InsightAgent result structure."""
    def _assert(calculation_result: Dict[str, Any], explanation: str = None):
        """Assert that result has expected InsightAgent structure."""
        # Check calculation result structure
        assert isinstance(calculation_result, dict)
        assert "calculation_trace" in calculation_result
        assert isinstance(calculation_result["calculation_trace"], list)

        # Calculation trace should have steps
        assert len(calculation_result["calculation_trace"]) > 0

        # If explanation provided, check it
        if explanation:
            assert isinstance(explanation, str)
            assert len(explanation) > 100  # Should be substantial

    return _assert


@pytest.fixture
def mock_tool_responses():
    """Mock tool execution responses."""
    return {
        "maintenance_log_tool": {
            "status": "success",
            "events_found": 2,
            "events": [
                {
                    "timestamp": "2025-11-05T14:30:00Z",
                    "type": "calibration",
                    "description": "Sensor recalibration performed"
                }
            ],
            "correlation": "High - maintenance 24h before anomaly",
            "confidence": 0.75
        },
        "sensor_diagnostic_tool": {
            "status": "success",
            "sensors_checked": 3,
            "diagnostics": [
                {
                    "sensor_id": "TEMP_01",
                    "health_status": "degraded",
                    "calibration_status": "overdue",
                    "drift_detected": True,
                    "recommendation": "Immediate recalibration required"
                }
            ],
            "overall_health": "degraded",
            "confidence": 0.80
        },
        "weather_data_tool": {
            "status": "success",
            "location": "California",
            "conditions": [
                {
                    "timestamp": "2025-11-05T12:00:00Z",
                    "temperature_c": 35.2,
                    "humidity_pct": 85,
                    "conditions": "Severe heatwave"
                }
            ],
            "extreme_events": ["Heatwave (>35C)"],
            "correlation": "Strong - anomalies during extreme heat",
            "confidence": 0.85
        },
        "historical_trend_tool": {
            "status": "success",
            "trend_type": "linear",
            "overall_direction": "increasing",
            "trend_strength": 15.2,
            "trend_change_pct": 15.2,
            "confidence": 0.75
        },
        "seasonality_tool": {
            "status": "success",
            "has_seasonality": True,
            "seasonal_period": 12,
            "seasonal_strength": 0.65,
            "confidence": 0.82
        },
        "event_correlation_tool": {
            "status": "success",
            "events_detected": 3,
            "correlated_events": [
                {
                    "event": "Heatwave (July 2025)",
                    "type": "weather",
                    "correlation_strength": 0.87,
                    "impact_magnitude": "+15% demand spike"
                }
            ],
            "confidence": 0.80
        },
        "data_visualization_tool": {
            "status": "success",
            "chart_recommendations": [
                {
                    "chart_type": "pie",
                    "purpose": "Show emission source breakdown",
                    "priority": "high"
                }
            ],
            "recommended_chart_count": 3
        },
        "stakeholder_preference_tool": {
            "status": "success",
            "stakeholder_level": "executive",
            "preferences": {
                "language_style": "Strategic, high-level",
                "technical_depth": "Minimal",
                "tone": "Action-oriented"
            }
        }
    }


@pytest.fixture
def assert_temperature_compliance():
    """Helper to assert temperature compliance."""
    def _assert(mock_session: Mock, expected_temp: float = 0.6):
        """Check that all chat calls used correct temperature."""
        for call in mock_session.chat.call_args_list:
            temp = call.kwargs.get("temperature")
            assert temp is not None, "Temperature not set"
            assert temp == expected_temp, f"Expected temp {expected_temp}, got {temp}"

    return _assert


@pytest.fixture
def assert_rag_collections():
    """Helper to assert RAG collections were queried."""
    def _assert(mock_rag: Mock, expected_collections: List[str]):
        """Check that expected RAG collections were queried."""
        assert mock_rag.query.called, "RAG query not called"

        call_args = mock_rag.query.call_args
        collections = call_args.kwargs.get("collections", [])

        assert len(collections) > 0, "No collections specified"

        for expected in expected_collections:
            assert expected in collections, f"Missing collection: {expected}"

    return _assert


@pytest.fixture
def assert_deterministic_calculation():
    """Helper to verify deterministic calculation."""
    def _assert(agent, inputs: Dict[str, Any]):
        """Run calculation twice and verify same results."""
        result1 = agent.calculate(inputs)
        result2 = agent.calculate(inputs)

        # Compare key numeric fields
        for key in result1.keys():
            if isinstance(result1[key], (int, float)):
                assert result1[key] == result2[key], f"Non-deterministic: {key} differs"
            elif isinstance(result1[key], list) and len(result1[key]) > 0:
                if isinstance(result1[key][0], (int, float)):
                    assert result1[key] == result2[key], f"Non-deterministic: {key} differs"

        return True

    return _assert
