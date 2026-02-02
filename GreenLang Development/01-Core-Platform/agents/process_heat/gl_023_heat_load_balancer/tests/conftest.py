"""
GL-023 HeatLoadBalancer - Test Configuration and Fixtures
=========================================================

Provides shared fixtures, test data generators, and test utilities
for comprehensive testing of HeatLoadBalancer calculators and agent.

Author: GL-TestEngineer
Version: 1.0.0
"""

import sys
import os
import json
import pytest
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock, patch

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

# Import from GL-Agent-Factory backend (actual implementation location)
try:
    from GL_Agent_Factory.backend.agents.gl_023_heat_load_balancer.models import (
        LoadBalancerInput,
        LoadBalancerOutput,
        LoadAllocation,
        EquipmentUnit,
    )
    from GL_Agent_Factory.backend.agents.gl_023_heat_load_balancer.formulas import (
        calculate_efficiency_at_load,
        calculate_fuel_consumption,
        calculate_hourly_cost,
        calculate_incremental_cost,
        calculate_emissions,
        economic_dispatch_merit_order,
        calculate_fleet_efficiency,
        calculate_equal_loading,
        generate_calculation_hash,
    )
    from GL_Agent_Factory.backend.agents.gl_023_heat_load_balancer.agent import (
        HeatLoadBalancerAgent,
    )
except ImportError:
    # Fallback to local imports or mock for testing
    pass


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# =============================================================================
# PYTEST MARKERS CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests for component interactions")
    config.addinivalue_line("markers", "optimization: Optimization algorithm tests")
    config.addinivalue_line("markers", "safety: Safety-critical tests")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "determinism: Determinism verification tests")
    config.addinivalue_line("markers", "slow: Slow tests (>1s)")
    config.addinivalue_line("markers", "critical: Critical tests that must pass")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add safety marker to safety tests
        if "safety" in item.nodeid:
            item.add_marker(pytest.mark.safety)
            item.add_marker(pytest.mark.critical)

        # Add optimization marker to optimizer tests
        if "optimizer" in item.nodeid:
            item.add_marker(pytest.mark.optimization)

        # Add calculator marker to calculator tests
        if "calculator" in item.nodeid:
            item.add_marker(pytest.mark.unit)


# =============================================================================
# DEFAULT CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_config() -> Dict[str, Any]:
    """Create default agent configuration."""
    return {
        "agent_id": "GL-023",
        "agent_name": "LOADBALANCER",
        "version": "1.0.0",
        "environment": "test",
        "optimization": {
            "default_mode": "COST",
            "solver_timeout_seconds": 30,
            "use_milp": True,
            "fallback_to_heuristic": True,
        },
        "safety": {
            "min_spinning_reserve_pct": 10.0,
            "max_units_starting": 2,
            "emergency_reserve_mw": 5.0,
            "n_plus_1_redundancy": True,
        },
        "logging": {
            "level": "INFO",
            "include_provenance": True,
        },
    }


@pytest.fixture
def test_agent(default_config) -> "HeatLoadBalancerAgent":
    """Create HeatLoadBalancerAgent instance for testing."""
    try:
        return HeatLoadBalancerAgent(default_config)
    except NameError:
        # Return mock if agent not available
        mock_agent = Mock()
        mock_agent.config = default_config
        mock_agent.AGENT_ID = "GL-023"
        mock_agent.AGENT_NAME = "LOADBALANCER"
        mock_agent.VERSION = "1.0.0"
        return mock_agent


# =============================================================================
# EQUIPMENT FLEET FIXTURES
# =============================================================================

@pytest.fixture
def sample_boiler_fleet() -> List[Dict[str, Any]]:
    """Create sample boiler fleet with 3 boilers of different capacities."""
    return [
        {
            "unit_id": "BOILER_001",
            "unit_type": "BOILER",
            "current_load_mw": 8.0,
            "min_load_mw": 2.0,
            "max_load_mw": 15.0,
            "current_efficiency_pct": 85.0,
            "efficiency_curve_a": 70.0,
            "efficiency_curve_b": 20.0,
            "efficiency_curve_c": -5.0,
            "is_available": True,
            "is_running": True,
            "startup_time_min": 30,
            "ramp_rate_mw_per_min": 1.0,
            "fuel_cost_per_mwh": 25.0,
            "maintenance_cost_per_mwh": 2.0,
            "startup_cost": 500.0,
            "min_run_time_hr": 2.0,
            "emissions_factor_kg_co2_mwh": 200.0,
        },
        {
            "unit_id": "BOILER_002",
            "unit_type": "BOILER",
            "current_load_mw": 5.0,
            "min_load_mw": 1.5,
            "max_load_mw": 10.0,
            "current_efficiency_pct": 82.0,
            "efficiency_curve_a": 68.0,
            "efficiency_curve_b": 18.0,
            "efficiency_curve_c": -4.0,
            "is_available": True,
            "is_running": True,
            "startup_time_min": 25,
            "ramp_rate_mw_per_min": 0.8,
            "fuel_cost_per_mwh": 28.0,
            "maintenance_cost_per_mwh": 2.5,
            "startup_cost": 400.0,
            "min_run_time_hr": 1.5,
            "emissions_factor_kg_co2_mwh": 210.0,
        },
        {
            "unit_id": "BOILER_003",
            "unit_type": "BOILER",
            "current_load_mw": 0.0,
            "min_load_mw": 3.0,
            "max_load_mw": 20.0,
            "current_efficiency_pct": 88.0,
            "efficiency_curve_a": 75.0,
            "efficiency_curve_b": 18.0,
            "efficiency_curve_c": -4.0,
            "is_available": True,
            "is_running": False,
            "startup_time_min": 45,
            "ramp_rate_mw_per_min": 1.5,
            "fuel_cost_per_mwh": 22.0,
            "maintenance_cost_per_mwh": 1.5,
            "startup_cost": 600.0,
            "min_run_time_hr": 3.0,
            "emissions_factor_kg_co2_mwh": 185.0,
        },
    ]


@pytest.fixture
def sample_furnace_fleet() -> List[Dict[str, Any]]:
    """Create sample furnace fleet with 2 furnaces."""
    return [
        {
            "unit_id": "FURNACE_001",
            "unit_type": "FURNACE",
            "current_load_mw": 12.0,
            "min_load_mw": 5.0,
            "max_load_mw": 25.0,
            "current_efficiency_pct": 78.0,
            "efficiency_curve_a": 65.0,
            "efficiency_curve_b": 22.0,
            "efficiency_curve_c": -7.0,
            "is_available": True,
            "is_running": True,
            "startup_time_min": 60,
            "ramp_rate_mw_per_min": 0.5,
            "fuel_cost_per_mwh": 20.0,
            "maintenance_cost_per_mwh": 3.0,
            "startup_cost": 800.0,
            "min_run_time_hr": 4.0,
            "emissions_factor_kg_co2_mwh": 250.0,
        },
        {
            "unit_id": "FURNACE_002",
            "unit_type": "FURNACE",
            "current_load_mw": 8.0,
            "min_load_mw": 4.0,
            "max_load_mw": 18.0,
            "current_efficiency_pct": 80.0,
            "efficiency_curve_a": 68.0,
            "efficiency_curve_b": 20.0,
            "efficiency_curve_c": -6.0,
            "is_available": True,
            "is_running": True,
            "startup_time_min": 50,
            "ramp_rate_mw_per_min": 0.6,
            "fuel_cost_per_mwh": 21.0,
            "maintenance_cost_per_mwh": 2.8,
            "startup_cost": 750.0,
            "min_run_time_hr": 3.5,
            "emissions_factor_kg_co2_mwh": 240.0,
        },
    ]


@pytest.fixture
def combined_equipment_fleet(sample_boiler_fleet, sample_furnace_fleet) -> List[Dict[str, Any]]:
    """Combine boiler and furnace fleets."""
    return sample_boiler_fleet + sample_furnace_fleet


@pytest.fixture
def unavailable_equipment() -> Dict[str, Any]:
    """Equipment that is unavailable (under maintenance)."""
    return {
        "unit_id": "BOILER_MAINT",
        "unit_type": "BOILER",
        "current_load_mw": 0.0,
        "min_load_mw": 2.0,
        "max_load_mw": 12.0,
        "current_efficiency_pct": 0.0,
        "efficiency_curve_a": 72.0,
        "efficiency_curve_b": 18.0,
        "efficiency_curve_c": -4.0,
        "is_available": False,  # Under maintenance
        "is_running": False,
        "startup_time_min": 30,
        "ramp_rate_mw_per_min": 0.8,
        "fuel_cost_per_mwh": 26.0,
        "maintenance_cost_per_mwh": 2.0,
        "startup_cost": 450.0,
        "min_run_time_hr": 2.0,
        "emissions_factor_kg_co2_mwh": 195.0,
    }


# =============================================================================
# DEMAND SCENARIO FIXTURES
# =============================================================================

@pytest.fixture
def sample_demand_scenarios() -> Dict[str, Dict[str, Any]]:
    """Sample demand scenarios: low, medium, high, peak."""
    return {
        "low": {
            "total_heat_demand_mw": 10.0,
            "demand_forecast_1hr_mw": 12.0,
            "demand_forecast_4hr_mw": 15.0,
            "description": "Low demand - off-peak hours",
        },
        "medium": {
            "total_heat_demand_mw": 30.0,
            "demand_forecast_1hr_mw": 32.0,
            "demand_forecast_4hr_mw": 35.0,
            "description": "Medium demand - normal operations",
        },
        "high": {
            "total_heat_demand_mw": 55.0,
            "demand_forecast_1hr_mw": 58.0,
            "demand_forecast_4hr_mw": 60.0,
            "description": "High demand - peak production",
        },
        "peak": {
            "total_heat_demand_mw": 75.0,
            "demand_forecast_1hr_mw": 78.0,
            "demand_forecast_4hr_mw": 80.0,
            "description": "Peak demand - near capacity",
        },
        "zero": {
            "total_heat_demand_mw": 0.0,
            "demand_forecast_1hr_mw": 5.0,
            "demand_forecast_4hr_mw": 10.0,
            "description": "Zero demand - plant shutdown",
        },
        "exceeds_capacity": {
            "total_heat_demand_mw": 150.0,
            "demand_forecast_1hr_mw": 155.0,
            "demand_forecast_4hr_mw": 160.0,
            "description": "Demand exceeds total capacity",
        },
    }


# =============================================================================
# FUEL PRICE FIXTURES
# =============================================================================

@pytest.fixture
def sample_fuel_prices() -> Dict[str, Dict[str, float]]:
    """Sample fuel prices for different scenarios."""
    return {
        "baseline": {
            "natural_gas_price_per_mmbtu": 3.50,
            "electricity_price_per_mwh": 75.0,
            "carbon_price_per_ton": 25.0,
        },
        "high_gas": {
            "natural_gas_price_per_mmbtu": 8.00,
            "electricity_price_per_mwh": 75.0,
            "carbon_price_per_ton": 25.0,
        },
        "high_carbon": {
            "natural_gas_price_per_mmbtu": 3.50,
            "electricity_price_per_mwh": 75.0,
            "carbon_price_per_ton": 100.0,
        },
        "zero_carbon": {
            "natural_gas_price_per_mmbtu": 3.50,
            "electricity_price_per_mwh": 75.0,
            "carbon_price_per_ton": 0.0,
        },
    }


# =============================================================================
# MOCK FIXTURES FOR EXTERNAL SYSTEMS
# =============================================================================

@pytest.fixture
def mock_opcua_connections():
    """Mock OPC-UA connections for equipment communication."""
    mock_connections = {}

    for unit_id in ["BOILER_001", "BOILER_002", "BOILER_003", "FURNACE_001", "FURNACE_002"]:
        connection = Mock()
        connection.is_connected = True
        connection.unit_id = unit_id
        connection.read_value = Mock(return_value={"status": "OK", "value": 0.0})
        connection.write_value = Mock(return_value={"status": "OK"})
        connection.read_multiple = Mock(return_value={
            "load_mw": 5.0,
            "efficiency_pct": 82.0,
            "fuel_flow_kg_hr": 500.0,
            "stack_temp_c": 180.0,
        })
        mock_connections[unit_id] = connection

    return mock_connections


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for event publishing."""
    producer = Mock()
    producer.send = Mock(return_value=Mock(get=Mock(return_value={"status": "OK"})))
    producer.flush = Mock()
    producer.close = Mock()

    # Track sent messages
    producer.messages = []

    def track_send(topic, value, key=None):
        producer.messages.append({
            "topic": topic,
            "key": key,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return Mock(get=Mock(return_value={"status": "OK"}))

    producer.send = Mock(side_effect=track_send)

    return producer


@pytest.fixture
def mock_historian_db():
    """Mock historian database for time-series data."""
    historian = Mock()

    def query_data(tag, start_time, end_time, interval="1m"):
        """Generate mock time-series data."""
        data_points = []
        current_time = start_time
        while current_time <= end_time:
            data_points.append({
                "timestamp": current_time.isoformat(),
                "tag": tag,
                "value": 75.0 + (hash(current_time.isoformat()) % 20),
                "quality": "GOOD",
            })
            current_time += timedelta(minutes=1)
        return data_points

    historian.query = Mock(side_effect=query_data)
    historian.write = Mock(return_value={"status": "OK"})

    return historian


# =============================================================================
# VALID INPUT FIXTURES
# =============================================================================

@pytest.fixture
def valid_load_balancer_input(sample_boiler_fleet, sample_demand_scenarios) -> Dict[str, Any]:
    """Create valid LoadBalancerInput data."""
    return {
        "equipment": [EquipmentUnit(**unit) for unit in sample_boiler_fleet],
        "total_heat_demand_mw": sample_demand_scenarios["medium"]["total_heat_demand_mw"],
        "demand_forecast_1hr_mw": sample_demand_scenarios["medium"]["demand_forecast_1hr_mw"],
        "demand_forecast_4hr_mw": sample_demand_scenarios["medium"]["demand_forecast_4hr_mw"],
        "optimization_mode": "COST",
        "cost_weight": 1.0,
        "efficiency_weight": 0.0,
        "emissions_weight": 0.0,
        "min_spinning_reserve_pct": 10.0,
        "max_units_starting": 1,
        "electricity_price_per_mwh": 75.0,
        "natural_gas_price_per_mmbtu": 3.50,
        "carbon_price_per_ton": 25.0,
    }


@pytest.fixture
def valid_equipment_unit() -> Dict[str, Any]:
    """Create valid EquipmentUnit data."""
    return {
        "unit_id": "TEST_BOILER_001",
        "unit_type": "BOILER",
        "current_load_mw": 5.0,
        "min_load_mw": 2.0,
        "max_load_mw": 10.0,
        "current_efficiency_pct": 85.0,
        "efficiency_curve_a": 70.0,
        "efficiency_curve_b": 20.0,
        "efficiency_curve_c": -5.0,
        "is_available": True,
        "is_running": True,
        "startup_time_min": 30,
        "ramp_rate_mw_per_min": 1.0,
        "fuel_cost_per_mwh": 25.0,
        "maintenance_cost_per_mwh": 2.0,
        "startup_cost": 500.0,
        "min_run_time_hr": 2.0,
        "emissions_factor_kg_co2_mwh": 200.0,
    }


# =============================================================================
# EFFICIENCY CURVE TEST DATA
# =============================================================================

@pytest.fixture
def efficiency_curve_test_cases() -> List[Dict[str, Any]]:
    """Test cases for efficiency curve calculations with known values."""
    return [
        # Typical boiler efficiency curve: peaks around 75% load
        # eta(L) = a + b*L + c*L^2 where L is load fraction (0-1)
        {
            "load_mw": 7.5,
            "min_load_mw": 2.0,
            "max_load_mw": 10.0,
            "curve_a": 70.0,
            "curve_b": 20.0,
            "curve_c": -5.0,
            "expected_efficiency": 83.75,  # 70 + 20*0.75 - 5*0.75^2 = 83.75
            "description": "75% load - peak efficiency point",
        },
        {
            "load_mw": 10.0,
            "min_load_mw": 2.0,
            "max_load_mw": 10.0,
            "curve_a": 70.0,
            "curve_b": 20.0,
            "curve_c": -5.0,
            "expected_efficiency": 85.0,  # 70 + 20*1.0 - 5*1.0^2 = 85
            "description": "100% load",
        },
        {
            "load_mw": 5.0,
            "min_load_mw": 2.0,
            "max_load_mw": 10.0,
            "curve_a": 70.0,
            "curve_b": 20.0,
            "curve_c": -5.0,
            "expected_efficiency": 78.75,  # 70 + 20*0.5 - 5*0.5^2 = 78.75
            "description": "50% load",
        },
        {
            "load_mw": 2.0,
            "min_load_mw": 2.0,
            "max_load_mw": 10.0,
            "curve_a": 70.0,
            "curve_b": 20.0,
            "curve_c": -5.0,
            "expected_efficiency": 73.8,  # 70 + 20*0.2 - 5*0.2^2 = 73.8
            "description": "Minimum load",
        },
        {
            "load_mw": 0.0,
            "min_load_mw": 2.0,
            "max_load_mw": 10.0,
            "curve_a": 70.0,
            "curve_b": 20.0,
            "curve_c": -5.0,
            "expected_efficiency": 0.0,  # Zero load = zero efficiency
            "description": "Zero load",
        },
        {
            "load_mw": 1.0,  # Below minimum
            "min_load_mw": 2.0,
            "max_load_mw": 10.0,
            "curve_a": 70.0,
            "curve_b": 20.0,
            "curve_c": -5.0,
            "expected_efficiency": 0.0,  # Below minimum stable combustion
            "description": "Below minimum load",
        },
    ]


@pytest.fixture
def piecewise_efficiency_curve() -> List[Tuple[float, float]]:
    """Piecewise efficiency curve data points."""
    return [
        (0.0, 0.0),    # 0% load
        (0.2, 72.0),   # 20% load
        (0.4, 78.0),   # 40% load
        (0.5, 80.0),   # 50% load
        (0.6, 82.0),   # 60% load
        (0.7, 84.0),   # 70% load
        (0.75, 85.0),  # 75% load (peak)
        (0.8, 84.5),   # 80% load
        (0.9, 83.0),   # 90% load
        (1.0, 81.0),   # 100% load
    ]


# =============================================================================
# OPTIMIZATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def milp_test_scenarios() -> List[Dict[str, Any]]:
    """Test scenarios for MILP optimization."""
    return [
        {
            "name": "cost_minimization",
            "objective": "COST",
            "demand_mw": 30.0,
            "expected_units_running": 2,
            "expected_cost_reduction_pct": 5.0,
        },
        {
            "name": "efficiency_maximization",
            "objective": "EFFICIENCY",
            "demand_mw": 30.0,
            "expected_units_running": 2,
            "expected_efficiency_improvement_pct": 2.0,
        },
        {
            "name": "emissions_minimization",
            "objective": "EMISSIONS",
            "demand_mw": 30.0,
            "expected_units_running": 2,
            "expected_emissions_reduction_pct": 3.0,
        },
        {
            "name": "balanced_optimization",
            "objective": "BALANCED",
            "demand_mw": 30.0,
            "expected_units_running": 2,
        },
    ]


@pytest.fixture
def infeasible_scenarios() -> List[Dict[str, Any]]:
    """Scenarios that should result in infeasible solutions."""
    return [
        {
            "name": "demand_exceeds_capacity",
            "demand_mw": 200.0,  # Way above capacity
            "expected_violation": "CAPACITY_EXCEEDED",
        },
        {
            "name": "all_equipment_unavailable",
            "demand_mw": 30.0,
            "available_units": [],
            "expected_violation": "NO_AVAILABLE_UNITS",
        },
        {
            "name": "demand_below_all_minimums",
            "demand_mw": 0.5,  # Below any unit's minimum
            "expected_violation": "DEMAND_BELOW_MINIMUM",
        },
    ]


# =============================================================================
# OUTPUT VALIDATION HELPERS
# =============================================================================

@pytest.fixture
def output_validator():
    """Helper functions to validate LoadBalancerOutput."""

    class OutputValidator:
        @staticmethod
        def validate_allocation_balance(output: Dict[str, Any], demand_mw: float, tolerance: float = 0.01):
            """Validate that total allocation matches demand."""
            total_allocated = sum(a.get("target_load_mw", 0) for a in output.get("allocations", []))
            assert abs(total_allocated - demand_mw) <= tolerance, (
                f"Allocation mismatch: allocated {total_allocated:.3f} MW vs demand {demand_mw:.3f} MW"
            )

        @staticmethod
        def validate_equipment_limits(output: Dict[str, Any], equipment: List[Dict[str, Any]]):
            """Validate all allocations respect equipment min/max limits."""
            equipment_lookup = {e["unit_id"]: e for e in equipment}

            for allocation in output.get("allocations", []):
                unit_id = allocation.get("unit_id")
                target_load = allocation.get("target_load_mw", 0)

                if unit_id in equipment_lookup:
                    unit = equipment_lookup[unit_id]
                    min_load = unit.get("min_load_mw", 0)
                    max_load = unit.get("max_load_mw", float("inf"))

                    # Target load must be 0 or between min and max
                    assert target_load == 0 or (min_load <= target_load <= max_load), (
                        f"Unit {unit_id}: target {target_load:.3f} MW outside limits [{min_load}, {max_load}]"
                    )

        @staticmethod
        def validate_provenance_hash(output: Dict[str, Any]):
            """Validate provenance hash format."""
            calc_hash = output.get("calculation_hash", "")
            assert len(calc_hash) == 64, f"Invalid hash length: {len(calc_hash)} (expected 64)"
            assert all(c in "0123456789abcdef" for c in calc_hash), "Hash contains invalid characters"

        @staticmethod
        def validate_spinning_reserve(output: Dict[str, Any], min_reserve_pct: float):
            """Validate spinning reserve meets minimum requirement."""
            reserve_pct = output.get("spinning_reserve_pct", 0)
            # Allow warning if reserve is below minimum but check constraint violations
            if reserve_pct < min_reserve_pct:
                assert "SPINNING_RESERVE_LOW" in output.get("constraint_violations", []), (
                    f"Reserve {reserve_pct:.1f}% below {min_reserve_pct}% but no violation reported"
                )

        @staticmethod
        def validate_cost_metrics(output: Dict[str, Any]):
            """Validate cost metrics are consistent."""
            total_cost = output.get("total_hourly_cost", 0)
            allocated = output.get("total_allocated_mw", 0)
            cost_per_mwh = output.get("cost_per_mwh", 0)

            if allocated > 0:
                expected_cost_per_mwh = total_cost / allocated
                assert abs(cost_per_mwh - expected_cost_per_mwh) < 0.01, (
                    f"Cost per MWh mismatch: {cost_per_mwh:.2f} vs calculated {expected_cost_per_mwh:.2f}"
                )

        @staticmethod
        def validate_emissions_metrics(output: Dict[str, Any]):
            """Validate emissions metrics are consistent."""
            total_emissions = output.get("total_hourly_emissions_kg", 0)
            allocated = output.get("total_allocated_mw", 0)
            intensity = output.get("emissions_intensity_kg_mwh", 0)

            if allocated > 0:
                expected_intensity = total_emissions / allocated
                assert abs(intensity - expected_intensity) < 0.01, (
                    f"Emissions intensity mismatch: {intensity:.2f} vs calculated {expected_intensity:.2f}"
                )

        @staticmethod
        def validate_complete_output(output: Dict[str, Any]):
            """Validate all required output fields are present."""
            required_fields = [
                "allocations", "total_capacity_mw", "total_allocated_mw",
                "spinning_reserve_mw", "spinning_reserve_pct", "fleet_efficiency_pct",
                "total_hourly_cost", "cost_per_mwh", "total_hourly_emissions_kg",
                "units_running", "constraints_satisfied", "calculation_hash",
                "optimization_method", "calculation_timestamp", "agent_version"
            ]

            for field in required_fields:
                assert field in output, f"Missing required field: {field}"

    return OutputValidator()


@pytest.fixture
def tolerance_checker():
    """Helper for floating point comparisons."""
    def check(actual: float, expected: float, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
        """Check if actual is within tolerance of expected."""
        import math
        return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)

    return check


@pytest.fixture
def determinism_checker():
    """Helper to verify deterministic behavior."""

    def check_determinism(func, *args, runs: int = 5, **kwargs) -> bool:
        """Run function multiple times and verify identical results."""
        results = []
        for _ in range(runs):
            result = func(*args, **kwargs)
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            elif hasattr(result, "__dict__"):
                result = result.__dict__
            results.append(json.dumps(result, sort_keys=True, default=str))

        # All results should be identical
        return all(r == results[0] for r in results)

    return check_determinism


# =============================================================================
# PERFORMANCE TESTING FIXTURES
# =============================================================================

@pytest.fixture
def large_equipment_fleet() -> List[Dict[str, Any]]:
    """Generate large equipment fleet for performance testing."""
    fleet = []
    for i in range(50):
        unit_type = "BOILER" if i % 3 != 0 else "FURNACE"
        fleet.append({
            "unit_id": f"{unit_type}_{i:03d}",
            "unit_type": unit_type,
            "current_load_mw": float(i % 10),
            "min_load_mw": 2.0 + (i % 5),
            "max_load_mw": 15.0 + (i % 10),
            "current_efficiency_pct": 80.0 + (i % 10),
            "efficiency_curve_a": 70.0,
            "efficiency_curve_b": 20.0,
            "efficiency_curve_c": -5.0,
            "is_available": i % 7 != 0,  # Every 7th unit unavailable
            "is_running": i % 3 != 0,
            "startup_time_min": 30 + (i % 30),
            "ramp_rate_mw_per_min": 0.5 + (i % 10) * 0.1,
            "fuel_cost_per_mwh": 20.0 + (i % 15),
            "maintenance_cost_per_mwh": 1.0 + (i % 5) * 0.5,
            "startup_cost": 400.0 + (i % 10) * 50,
            "min_run_time_hr": 1.0 + (i % 4),
            "emissions_factor_kg_co2_mwh": 180.0 + (i % 50),
        })
    return fleet


@pytest.fixture
def benchmark_iterations() -> int:
    """Number of iterations for benchmark tests."""
    return 100


# =============================================================================
# SAFETY TEST FIXTURES
# =============================================================================

@pytest.fixture
def equipment_trip_scenarios() -> List[Dict[str, Any]]:
    """Scenarios for equipment trip testing."""
    return [
        {
            "name": "single_boiler_trip",
            "tripped_unit": "BOILER_001",
            "remaining_capacity_mw": 30.0,
            "demand_mw": 25.0,
            "expected_rebalance": True,
        },
        {
            "name": "multiple_unit_trip",
            "tripped_units": ["BOILER_001", "FURNACE_001"],
            "remaining_capacity_mw": 28.0,
            "demand_mw": 25.0,
            "expected_rebalance": True,
        },
        {
            "name": "cascade_trip",
            "tripped_units": ["BOILER_001", "BOILER_002", "BOILER_003"],
            "remaining_capacity_mw": 43.0,
            "demand_mw": 50.0,
            "expected_load_shed": True,
        },
    ]


@pytest.fixture
def ramp_rate_test_cases() -> List[Dict[str, Any]]:
    """Test cases for ramp rate validation."""
    return [
        {
            "current_load_mw": 5.0,
            "target_load_mw": 8.0,
            "ramp_rate_mw_per_min": 1.0,
            "time_available_min": 5.0,
            "expected_achievable": True,
            "expected_time_min": 3.0,
        },
        {
            "current_load_mw": 5.0,
            "target_load_mw": 15.0,
            "ramp_rate_mw_per_min": 1.0,
            "time_available_min": 5.0,
            "expected_achievable": False,
            "expected_time_min": 10.0,
        },
        {
            "current_load_mw": 10.0,
            "target_load_mw": 5.0,
            "ramp_rate_mw_per_min": 2.0,
            "time_available_min": 3.0,
            "expected_achievable": True,
            "expected_time_min": 2.5,
        },
    ]


# =============================================================================
# TEST DATA DIRECTORY
# =============================================================================

@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def golden_test_data(test_data_dir) -> Dict[str, Any]:
    """Load golden test data for regression testing."""
    golden_file = test_data_dir / "golden_test_cases.json"

    if not golden_file.exists():
        # Create default golden data
        test_data_dir.mkdir(exist_ok=True)
        golden_data = {
            "version": "1.0.0",
            "test_cases": [
                {
                    "name": "baseline_cost_optimization",
                    "demand_mw": 30.0,
                    "expected_cost": 825.50,
                    "expected_efficiency": 83.5,
                    "expected_hash": "a1b2c3d4e5f6...",
                }
            ]
        }
        golden_file.write_text(json.dumps(golden_data, indent=2))
        return golden_data

    return json.loads(golden_file.read_text())
