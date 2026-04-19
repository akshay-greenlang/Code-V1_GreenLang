"""
GL-006 WasteHeatRecovery Agent - Test Suite

Comprehensive test suite for the Waste Heat Recovery Analyzer agent.
Targets 85%+ code coverage with unit, integration, and performance tests.

Test Modules:
    - test_analyzer.py: Main analyzer tests
    - test_pinch_analysis.py: Pinch analysis algorithm tests
    - test_hen_synthesis.py: Heat exchanger network synthesis tests
    - test_exergy_analysis.py: Exergy (second law) analysis tests
    - test_economic_optimizer.py: Economic analysis tests (NPV, IRR, payback)

Coverage Targets:
    - Unit tests: 85%+ line coverage
    - Branch coverage: 80%+
    - Integration tests for all module interactions

Test Categories (pytest markers):
    - unit: Unit tests (fast, isolated)
    - integration: Integration tests (slower, real dependencies)
    - performance: Performance benchmarks
    - compliance: Regulatory compliance tests
    - slow: Long-running tests

Example Usage:
    # Run all tests
    pytest greenlang/agents/process_heat/gl_006_waste_heat_recovery/tests/

    # Run only unit tests
    pytest -m unit greenlang/agents/process_heat/gl_006_waste_heat_recovery/tests/

    # Run with coverage
    pytest --cov=greenlang.agents.process_heat.gl_006_waste_heat_recovery --cov-report=html

    # Run specific test file
    pytest greenlang/agents/process_heat/gl_006_waste_heat_recovery/tests/test_pinch_analysis.py
"""

import pytest
from typing import Any, Dict, List


# =============================================================================
# SHARED TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_hot_streams():
    """Create sample hot streams for testing."""
    from greenlang.agents.process_heat.gl_006_waste_heat_recovery.pinch_analysis import (
        HeatStream, StreamType
    )

    return [
        HeatStream(
            name="H1",
            stream_type=StreamType.HOT,
            supply_temp_f=300.0,
            target_temp_f=150.0,
            mcp=10.0,
        ),
        HeatStream(
            name="H2",
            stream_type=StreamType.HOT,
            supply_temp_f=250.0,
            target_temp_f=100.0,
            mcp=15.0,
        ),
    ]


@pytest.fixture
def sample_cold_streams():
    """Create sample cold streams for testing."""
    from greenlang.agents.process_heat.gl_006_waste_heat_recovery.pinch_analysis import (
        HeatStream, StreamType
    )

    return [
        HeatStream(
            name="C1",
            stream_type=StreamType.COLD,
            supply_temp_f=80.0,
            target_temp_f=200.0,
            mcp=12.0,
        ),
        HeatStream(
            name="C2",
            stream_type=StreamType.COLD,
            supply_temp_f=120.0,
            target_temp_f=280.0,
            mcp=8.0,
        ),
    ]


@pytest.fixture
def sample_streams(sample_hot_streams, sample_cold_streams):
    """Combine hot and cold streams."""
    return sample_hot_streams + sample_cold_streams


@pytest.fixture
def sample_waste_heat_source():
    """Create sample waste heat source."""
    from greenlang.agents.process_heat.gl_006_waste_heat_recovery.analyzer import (
        WasteHeatSource
    )

    return WasteHeatSource(
        source_id="WHS-001",
        source_type="exhaust_gas",
        temperature_f=450.0,
        flow_rate=10000.0,
        flow_unit="lb/hr",
        specific_heat=0.25,
        availability_pct=95.0,
        operating_hours_yr=8000,
        min_discharge_temp_f=250.0,
    )


@pytest.fixture
def sample_heat_sink():
    """Create sample heat sink."""
    from greenlang.agents.process_heat.gl_006_waste_heat_recovery.analyzer import (
        WasteHeatSink
    )

    return WasteHeatSink(
        sink_id="HS-001",
        sink_type="process_heating",
        required_temperature_f=350.0,
        inlet_temperature_f=100.0,
        flow_rate=5000.0,
        flow_unit="lb/hr",
        specific_heat=0.5,
        current_energy_source="natural_gas",
        current_cost_per_mmbtu=8.0,
    )


@pytest.fixture
def sample_exergy_streams():
    """Create sample exergy streams for testing."""
    from greenlang.agents.process_heat.gl_006_waste_heat_recovery.exergy_analysis import (
        ExergyStream
    )

    return [
        ExergyStream(
            name="Exhaust_In",
            temp_f=800.0,
            pressure_psia=15.0,
            mass_flow_lb_hr=10000.0,
            specific_heat_btu_lb_f=0.25,
            is_inlet=True,
        ),
        ExergyStream(
            name="Exhaust_Out",
            temp_f=300.0,
            pressure_psia=14.7,
            mass_flow_lb_hr=10000.0,
            specific_heat_btu_lb_f=0.25,
            is_inlet=False,
        ),
    ]


@pytest.fixture
def sample_process_component():
    """Create sample process component."""
    from greenlang.agents.process_heat.gl_006_waste_heat_recovery.exergy_analysis import (
        ProcessComponent, ComponentType
    )

    return ProcessComponent(
        name="Economizer",
        component_type=ComponentType.HEAT_EXCHANGER,
        inlet_streams=["Exhaust_In"],
        outlet_streams=["Exhaust_Out"],
        heat_transfer_btu_hr=-500000,
        heat_transfer_temp_f=550.0,
        capital_cost_usd=75000.0,
    )


@pytest.fixture
def sample_project():
    """Create sample economic project."""
    from greenlang.agents.process_heat.gl_006_waste_heat_recovery.economic_optimizer import (
        WasteHeatProject, EnergyMetrics
    )

    return WasteHeatProject(
        name="Economizer Installation",
        capital_cost_usd=150000.0,
        annual_operating_cost_usd=5000.0,
        annual_energy_savings_usd=45000.0,
        project_life_years=20,
        energy_metrics=EnergyMetrics(
            annual_energy_savings_mmbtu=5000.0,
            fuel_type="natural_gas",
            co2_reduction_tons_yr=265.0,
        ),
    )


# =============================================================================
# TEST UTILITIES
# =============================================================================

def assert_within_tolerance(actual: float, expected: float, tolerance: float = 0.01):
    """Assert value is within relative tolerance."""
    if expected == 0:
        assert abs(actual) < tolerance
    else:
        assert abs((actual - expected) / expected) < tolerance, \
            f"Expected {expected}, got {actual} (tolerance: {tolerance*100}%)"


def assert_provenance_hash(hash_value: str):
    """Assert provenance hash is valid SHA-256."""
    assert isinstance(hash_value, str)
    assert len(hash_value) == 64
    assert all(c in '0123456789abcdef' for c in hash_value)


def create_test_streams_for_pinch(
    num_hot: int = 2,
    num_cold: int = 2,
    temp_range: tuple = (100, 400),
    mcp_range: tuple = (5, 20),
) -> List[Any]:
    """Create test streams with specified parameters."""
    import random
    from greenlang.agents.process_heat.gl_006_waste_heat_recovery.pinch_analysis import (
        HeatStream, StreamType
    )

    streams = []

    for i in range(num_hot):
        supply_temp = random.uniform(temp_range[0] + 100, temp_range[1])
        target_temp = random.uniform(temp_range[0], supply_temp - 50)
        mcp = random.uniform(mcp_range[0], mcp_range[1])

        streams.append(HeatStream(
            name=f"H{i+1}",
            stream_type=StreamType.HOT,
            supply_temp_f=supply_temp,
            target_temp_f=target_temp,
            mcp=mcp,
        ))

    for i in range(num_cold):
        supply_temp = random.uniform(temp_range[0], temp_range[1] - 100)
        target_temp = random.uniform(supply_temp + 50, temp_range[1])
        mcp = random.uniform(mcp_range[0], mcp_range[1])

        streams.append(HeatStream(
            name=f"C{i+1}",
            stream_type=StreamType.COLD,
            supply_temp_f=supply_temp,
            target_temp_f=target_temp,
            mcp=mcp,
        ))

    return streams


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance benchmarks")
    config.addinivalue_line("markers", "compliance: Regulatory compliance tests")
    config.addinivalue_line("markers", "slow: Long-running tests")
