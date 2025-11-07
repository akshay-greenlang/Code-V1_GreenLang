"""
Pytest fixtures for Phase 3 agent tests.

Provides mocks for:
- RAGEngine
- ChatSession
- Tool execution
- Common test data
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, MagicMock


@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for testing."""
    engine = Mock()

    # Mock query result
    mock_result = Mock()
    mock_result.chunks = [
        Mock(text="Case study: 30% emissions reduction achieved"),
        Mock(text="Technology spec: 95% efficiency available"),
        Mock(text="Best practice: Implement in phases")
    ]
    mock_result.relevance_scores = [0.95, 0.88, 0.82]
    mock_result.search_time_ms = 150

    engine.query = AsyncMock(return_value=mock_result)

    return engine


@pytest.fixture
def mock_chat_session():
    """Mock ChatSession for testing."""
    session = Mock()

    # Mock chat response
    mock_response = Mock()
    mock_response.text = """
    Based on the analysis, I recommend a High-Efficiency Condensing Boiler
    with the following specifications:
    - Capacity: 50 MMBtu/hr
    - Efficiency: 95%
    - CAPEX: $1,200,000
    - Payback: 4.2 years
    - Emissions reduction: 20%
    """
    mock_response.tool_calls = []  # No tools called by default
    mock_response.provider_info = {"model": "gpt-4", "provider": "openai"}
    mock_response.usage = {
        "total_tokens": 2500,
        "total_cost": 0.15
    }

    session.chat = AsyncMock(return_value=mock_response)

    return session


@pytest.fixture
def sample_facility_context():
    """Sample facility context for testing."""
    return {
        "facility_id": "TEST-001",
        "industry_type": "Food & Beverage",
        "fuel_consumption": {"natural_gas": 50000},
        "electricity_consumption_kwh": 15000000,
        "grid_region": "CAISO",
        "capital_budget_usd": 10000000,
        "target_reduction_percent": 50,
        "target_year": 2030,
        "facility_area_sqm": 50000,
        "available_land_sqm": 10000
    }


@pytest.fixture
def sample_boiler_context():
    """Sample boiler context for testing."""
    return {
        "current_boiler_type": "firetube",
        "current_fuel": "natural_gas",
        "current_efficiency": 78.5,
        "rated_capacity_mmbtu_hr": 50,
        "annual_operating_hours": 6000,
        "steam_pressure_psi": 150,
        "facility_type": "food_processing",
        "region": "US_Northeast",
        "budget_usd": 1500000
    }


@pytest.fixture
def sample_heat_pump_context():
    """Sample heat pump context for testing."""
    return {
        "process_heat_requirement_kw": 500,
        "supply_temperature_c": 80,
        "return_temperature_c": 60,
        "heat_source_type": "waste_heat",
        "heat_source_temp_c": 40,
        "annual_operating_hours": 7000,
        "electricity_cost_per_kwh": 0.12,
        "grid_region": "CAISO",
        "facility_type": "food_processing",
        "budget_usd": 800000
    }


@pytest.fixture
def sample_whr_context():
    """Sample waste heat recovery context for testing."""
    return {
        "waste_heat_sources": [
            {"source": "flue_gas", "temp_c": 180, "flow_rate_kg_s": 5.0},
            {"source": "cooling_water", "temp_c": 60, "flow_rate_kg_s": 10.0}
        ],
        "heat_sinks": [
            {"sink": "process_water", "temp_c": 40, "demand_kw": 300}
        ],
        "facility_type": "chemical_plant",
        "region": "US_Midwest",
        "budget_usd": 500000
    }


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry with sample tool implementations."""
    return {
        "aggregate_ghg_inventory": AsyncMock(return_value={
            "total_emissions_kg_co2e": 5000000,
            "scope1_kg_co2e": 3000000,
            "scope2_kg_co2e": 2000000
        }),
        "technology_database_tool": AsyncMock(return_value={
            "technologies": [
                {"name": "Solar Thermal", "capex": 2500000, "efficiency": 0.85}
            ]
        }),
        "financial_analysis_tool": AsyncMock(return_value={
            "npv_usd": 3500000,
            "irr_percent": 12.5,
            "payback_years": 6.2
        })
    }


@pytest.fixture
def assert_reasoning_agent_result():
    """Helper to assert ReasoningAgent result structure."""
    def _assert(result: Dict[str, Any]):
        """Assert that result has expected ReasoningAgent structure."""
        assert "success" in result
        assert result["success"] == True

        assert "reasoning_trace" in result
        trace = result["reasoning_trace"]

        assert "rag_context" in trace
        assert "chunks_retrieved" in trace["rag_context"]
        assert "collections_searched" in trace["rag_context"]

        assert "tool_execution" in trace
        assert "total_tools_called" in trace["tool_execution"]

        assert "orchestration_iterations" in trace
        assert "temperature" in trace
        assert trace["temperature"] == 0.7

        assert "pattern" in trace
        assert trace["pattern"] == "ReasoningAgent"

        assert "version" in trace
        assert trace["version"] == "3.0.0"

        assert "metadata" in result
        assert "model" in result["metadata"]
        assert "tokens_used" in result["metadata"]

    return _assert
