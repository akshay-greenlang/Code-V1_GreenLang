# -*- coding: utf-8 -*-
"""
GL-007 FurnacePerformanceMonitor Test Configuration and Fixtures

Provides shared test fixtures, configurations, and utilities for all GL-007 tests.
Includes mock data, agent instances, and integration test helpers.

Version: 1.0.0
Date: 2025-11-21
"""

import pytest
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from greenlang.determinism import DeterministicClock


# ============================================================================
# PYTEST CONFIGURATION HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark tests based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


# ============================================================================
# CORE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        "environment": "test",
        "log_level": "DEBUG",
        "enable_monitoring": False,
        "enable_tracing": False,
        "timeout_seconds": 30,
        "max_retries": 3,
    }


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return {
        "agent_id": "GL-007",
        "name": "FurnacePerformanceMonitor",
        "version": "1.0.0",
        "environment": "test",
        "log_level": "DEBUG",
        "enable_provenance": True,
        "enable_monitoring": False,
    }


@pytest.fixture
def mock_agent(agent_config):
    """Create mock GL-007 agent instance."""
    agent = Mock()
    agent.config = agent_config
    agent.agent_id = "GL-007"
    agent.name = "FurnacePerformanceMonitor"
    agent.version = "1.0.0"
    return agent


# ============================================================================
# FURNACE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_furnace_data():
    """Sample furnace operating data."""
    return {
        "furnace_id": "FURNACE-001",
        "furnace_type": "process_heater",
        "timestamp": DeterministicClock.now().isoformat(),
        "fuel_input_mw": 25.5,
        "fuel_flow_kg_hr": 1850.0,
        "fuel_type": "natural_gas",
        "heating_value_mj_kg": 50.0,
        "flue_gas_temperature_c": 185.0,
        "ambient_temperature_c": 20.0,
        "stack_o2_percent": 3.5,
        "process_fluid_inlet_temp_c": 120.0,
        "process_fluid_outlet_temp_c": 350.0,
        "process_fluid_flow_kg_hr": 45000.0,
        "excess_air_percent": 15.0,
        "draft_pressure_pa": -25.0,
        "production_rate_ton_hr": 18.5,
    }


@pytest.fixture
def sample_thermal_efficiency_input():
    """Sample input for thermal efficiency calculation."""
    return {
        "fuel_input_mw": 25.5,
        "fuel_type": "natural_gas",
        "heating_value_mj_kg": 50.0,
        "flue_gas_temperature_c": 185.0,
        "flue_gas_flow_kg_hr": 28000.0,
        "ambient_temperature_c": 20.0,
        "stack_o2_percent": 3.5,
        "heat_absorbed_mw": 20.8,
        "radiation_losses_percent": 2.5,
        "convection_losses_percent": 1.2,
        "unaccounted_losses_percent": 1.8,
    }


@pytest.fixture
def sample_fuel_consumption_data():
    """Sample fuel consumption data for analysis."""
    base_time = DeterministicClock.now()
    consumption_data = []

    for i in range(24):  # 24 hours of data
        consumption_data.append({
            "timestamp": (base_time - timedelta(hours=23-i)).isoformat(),
            "fuel_type": "natural_gas",
            "consumption_rate_kg_hr": 1850.0 + np.random.normal(0, 50),
            "consumption_rate_nm3_hr": 2450.0 + np.random.normal(0, 65),
            "heating_value_mj_kg": 50.0,
            "production_rate": 18.5 + np.random.normal(0, 0.5),
            "furnace_load_percent": 85.0 + np.random.normal(0, 5),
        })

    return {
        "consumption_data": consumption_data,
        "baseline_performance": {
            "expected_sec_gj_ton": 4.8,
            "design_sec_gj_ton": 4.5,
            "best_achieved_sec_gj_ton": 4.3,
            "variability_factor": 0.05,
        },
        "cost_parameters": {
            "fuel_cost_usd_per_gj": 8.5,
            "carbon_price_usd_per_ton_co2": 50.0,
            "emission_factor_kg_co2_per_gj": 56.1,
        }
    }


@pytest.fixture
def sample_equipment_inventory():
    """Sample equipment inventory for maintenance prediction."""
    return [
        {
            "equipment_id": "REFRACTORY-001",
            "equipment_type": "refractory",
            "installation_date": "2020-01-15",
            "last_maintenance_date": "2024-06-01",
            "design_life_years": 5.0,
            "criticality": "critical",
        },
        {
            "equipment_id": "BURNER-001",
            "equipment_type": "burner",
            "installation_date": "2021-03-10",
            "last_maintenance_date": "2025-01-10",
            "design_life_years": 10.0,
            "criticality": "high",
        },
        {
            "equipment_id": "TUBE-BANK-A",
            "equipment_type": "tube",
            "installation_date": "2018-05-20",
            "last_maintenance_date": "2024-11-01",
            "design_life_years": 15.0,
            "criticality": "critical",
        },
    ]


@pytest.fixture
def sample_condition_monitoring_data():
    """Sample condition monitoring data."""
    base_time = DeterministicClock.now()

    return [
        {
            "equipment_id": "REFRACTORY-001",
            "timestamp": base_time.isoformat(),
            "temperature_readings": [1250.0, 1245.0, 1255.0, 1248.0],
            "vibration_readings": [],
            "thermal_imaging_data": {
                "max_temp_c": 1265.0,
                "min_temp_c": 1235.0,
                "hotspot_count": 2,
            },
            "performance_metrics": {
                "thermal_conductivity_w_m_k": 1.2,
                "heat_loss_kw": 125.0,
            },
            "alarm_history": [],
        },
        {
            "equipment_id": "BURNER-001",
            "timestamp": base_time.isoformat(),
            "temperature_readings": [1580.0, 1575.0, 1585.0],
            "vibration_readings": [0.5, 0.52, 0.48, 0.51],
            "thermal_imaging_data": {},
            "performance_metrics": {
                "nox_ppm": 45.0,
                "co_ppm": 15.0,
                "flame_stability_score": 0.92,
            },
            "alarm_history": [],
        },
    ]


@pytest.fixture
def sample_operating_history():
    """Sample operating history."""
    return {
        "operating_hours": 42000,
        "thermal_cycles": 850,
        "temperature_excursions": 12,
        "emergency_shutdowns": 2,
        "load_factor_average": 0.85,
    }


@pytest.fixture
def sample_historical_baseline():
    """Sample historical baseline for anomaly detection."""
    return {
        "mean_values": {
            "efficiency": 81.5,
            "stack_temp_c": 185.0,
            "o2_percent": 3.5,
            "nox_ppm": 45.0,
            "co_ppm": 18.0,
        },
        "standard_deviations": {
            "efficiency": 1.2,
            "stack_temp_c": 8.5,
            "o2_percent": 0.5,
            "nox_ppm": 5.0,
            "co_ppm": 3.0,
        },
        "control_limits_upper": {
            "efficiency": 84.0,
            "stack_temp_c": 210.0,
            "o2_percent": 5.0,
            "nox_ppm": 60.0,
            "co_ppm": 30.0,
        },
        "control_limits_lower": {
            "efficiency": 79.0,
            "stack_temp_c": 160.0,
            "o2_percent": 2.0,
            "nox_ppm": 30.0,
            "co_ppm": 8.0,
        },
        "correlation_matrix": [
            [1.0, -0.65, 0.42, -0.38, 0.28],
            [-0.65, 1.0, -0.55, 0.48, -0.32],
            [0.42, -0.55, 1.0, -0.25, 0.18],
            [-0.38, 0.48, -0.25, 1.0, -0.42],
            [0.28, -0.32, 0.18, -0.42, 1.0],
        ],
    }


@pytest.fixture
def sample_multi_furnace_data():
    """Sample data for multi-furnace fleet optimization."""
    return {
        "furnaces": [
            {
                "furnace_id": "FURNACE-001",
                "current_load_mw": 20.5,
                "max_capacity_mw": 30.0,
                "efficiency_at_current_load": 81.5,
                "fuel_cost_usd_per_mwh": 35.0,
                "availability": True,
                "maintenance_due_days": 45,
            },
            {
                "furnace_id": "FURNACE-002",
                "current_load_mw": 25.0,
                "max_capacity_mw": 35.0,
                "efficiency_at_current_load": 83.2,
                "fuel_cost_usd_per_mwh": 33.0,
                "availability": True,
                "maintenance_due_days": 120,
            },
            {
                "furnace_id": "FURNACE-003",
                "current_load_mw": 15.0,
                "max_capacity_mw": 25.0,
                "efficiency_at_current_load": 78.5,
                "fuel_cost_usd_per_mwh": 38.0,
                "availability": True,
                "maintenance_due_days": 10,
            },
        ],
        "total_heat_demand_mw": 60.0,
        "optimization_objective": "minimize_cost",
        "constraints": {
            "min_load_per_furnace_mw": 10.0,
            "max_load_change_rate_mw_per_min": 0.5,
            "reserve_capacity_percent": 10.0,
        }
    }


# ============================================================================
# EMISSION FACTORS AND REFERENCE DATA
# ============================================================================

@pytest.fixture
def emission_factors_database():
    """Emission factors for various fuels."""
    return {
        ("natural_gas", "US", "stationary_combustion"): 56.1,  # kg CO2e per GJ
        ("diesel", "US", "stationary_combustion"): 74.1,
        ("coal", "US", "stationary_combustion"): 94.6,
        ("fuel_oil", "US", "stationary_combustion"): 77.4,
        ("biomass", "US", "stationary_combustion"): 0.0,  # Carbon neutral
        ("hydrogen", "US", "stationary_combustion"): 0.0,  # Zero emissions
    }


@pytest.fixture
def fuel_properties_database():
    """Physical and chemical properties of various fuels."""
    return {
        "natural_gas": {
            "heating_value_mj_kg": 50.0,
            "heating_value_mj_nm3": 37.5,
            "density_kg_m3": 0.75,
            "stoichiometric_air_ratio": 17.2,
            "adiabatic_flame_temp_c": 1950,
        },
        "diesel": {
            "heating_value_mj_kg": 45.6,
            "heating_value_mj_l": 38.6,
            "density_kg_m3": 850.0,
            "stoichiometric_air_ratio": 14.5,
            "adiabatic_flame_temp_c": 2050,
        },
        "coal": {
            "heating_value_mj_kg": 25.0,
            "density_kg_m3": 1400.0,
            "stoichiometric_air_ratio": 10.5,
            "adiabatic_flame_temp_c": 1850,
        },
        "hydrogen": {
            "heating_value_mj_kg": 120.0,
            "heating_value_mj_nm3": 10.8,
            "density_kg_m3": 0.09,
            "stoichiometric_air_ratio": 34.3,
            "adiabatic_flame_temp_c": 2200,
        },
    }


# ============================================================================
# MOCK EXTERNAL SYSTEMS
# ============================================================================

@pytest.fixture
def mock_dcs_client():
    """Mock DCS (Distributed Control System) client."""
    mock_client = Mock()
    mock_client.connect = Mock(return_value=True)
    mock_client.disconnect = Mock(return_value=True)
    mock_client.read_tag = Mock(return_value={"value": 100.0, "timestamp": DeterministicClock.now()})
    mock_client.write_tag = Mock(return_value=True)
    mock_client.read_multiple_tags = Mock(return_value={
        "FT-101": 1850.0,
        "TT-102": 185.0,
        "PT-103": -25.0,
        "AT-104": 3.5,
    })
    return mock_client


@pytest.fixture
def mock_cems_client():
    """Mock CEMS (Continuous Emissions Monitoring System) client."""
    mock_client = Mock()
    mock_client.connect = Mock(return_value=True)
    mock_client.get_emissions_data = Mock(return_value={
        "nox_ppm": 45.0,
        "co_ppm": 18.0,
        "co2_percent": 8.5,
        "so2_ppm": 5.0,
        "o2_percent": 3.5,
        "opacity_percent": 5.0,
        "timestamp": DeterministicClock.now(),
    })
    return mock_client


@pytest.fixture
def mock_cmms_client():
    """Mock CMMS (Computerized Maintenance Management System) client."""
    mock_client = Mock()
    mock_client.get_maintenance_history = Mock(return_value=[
        {
            "date": "2025-01-10",
            "equipment_id": "BURNER-001",
            "maintenance_type": "preventive",
            "findings": "Burner nozzle cleaned, minor wear observed",
            "cost_usd": 2500.0,
        }
    ])
    mock_client.create_work_order = Mock(return_value={"work_order_id": "WO-12345"})
    return mock_client


@pytest.fixture
def mock_erp_client():
    """Mock ERP (Enterprise Resource Planning) client."""
    mock_client = Mock()
    mock_client.get_fuel_pricing = Mock(return_value={
        "natural_gas": 8.5,  # USD per GJ
        "diesel": 12.0,
        "coal": 4.5,
    })
    mock_client.get_production_schedule = Mock(return_value=[
        {
            "furnace_id": "FURNACE-001",
            "start_time": DeterministicClock.now(),
            "end_time": DeterministicClock.now() + timedelta(hours=8),
            "production_target_ton": 150.0,
        }
    ])
    return mock_client


# ============================================================================
# VALIDATION AND ASSERTION HELPERS
# ============================================================================

@pytest.fixture
def assert_thermal_efficiency_valid():
    """Helper to validate thermal efficiency calculation results."""
    def _validate(result):
        assert "thermal_efficiency_percent" in result
        assert "heat_input_mw" in result
        assert "heat_output_mw" in result
        assert "losses_breakdown" in result

        # Efficiency should be between 0 and 100%
        assert 0 <= result["thermal_efficiency_percent"] <= 100

        # Heat output should not exceed heat input
        assert result["heat_output_mw"] <= result["heat_input_mw"]

        # Losses should sum to (100 - efficiency)
        total_losses = sum(result["losses_breakdown"].values())
        expected_losses = 100.0 - result["thermal_efficiency_percent"]
        assert abs(total_losses - expected_losses) < 0.5  # ±0.5% tolerance

        # Provenance hash should be present
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64  # SHA-256

    return _validate


@pytest.fixture
def assert_fuel_consumption_valid():
    """Helper to validate fuel consumption analysis results."""
    def _validate(result):
        assert "consumption_summary" in result
        assert "deviation_analysis" in result
        assert "cost_impact" in result

        # All numeric values should be non-negative
        assert result["consumption_summary"]["total_fuel_consumed_kg"] >= 0
        assert result["consumption_summary"]["fuel_cost_usd"] >= 0

    return _validate


@pytest.fixture
def assert_provenance_deterministic():
    """Helper to validate provenance hash is deterministic."""
    def _validate(result1, result2):
        # Same inputs should produce same provenance hash
        assert result1["provenance_hash"] == result2["provenance_hash"]

    return _validate


# ============================================================================
# PERFORMANCE BENCHMARKING FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_target_latency():
    """Target latency for various operations (milliseconds)."""
    return {
        "thermal_efficiency_calculation": 50.0,
        "fuel_consumption_analysis": 100.0,
        "maintenance_prediction": 200.0,
        "anomaly_detection": 80.0,
        "multi_furnace_optimization": 3000.0,
    }


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_furnace_timeseries(
        furnace_id: str,
        duration_hours: int = 24,
        interval_minutes: int = 5,
        add_anomalies: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate realistic furnace time series data."""
        data = []
        base_time = DeterministicClock.now()
        num_points = (duration_hours * 60) // interval_minutes

        for i in range(num_points):
            timestamp = base_time - timedelta(minutes=interval_minutes * (num_points - i - 1))

            # Base values with realistic variations
            data_point = {
                "timestamp": timestamp.isoformat(),
                "furnace_id": furnace_id,
                "fuel_flow_kg_hr": 1850.0 + np.random.normal(0, 50),
                "flue_gas_temp_c": 185.0 + np.random.normal(0, 8),
                "stack_o2_percent": 3.5 + np.random.normal(0, 0.3),
                "efficiency_percent": 81.5 + np.random.normal(0, 1.0),
                "production_rate_ton_hr": 18.5 + np.random.normal(0, 0.5),
            }

            # Add anomalies if requested
            if add_anomalies and i in [100, 200, 300]:
                data_point["flue_gas_temp_c"] += 30.0  # Temperature spike
                data_point["efficiency_percent"] -= 5.0  # Efficiency drop

            data.append(data_point)

        return data


@pytest.fixture
def test_data_generator():
    """Provide TestDataGenerator instance."""
    return TestDataGenerator()


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Cleanup test artifacts after each test."""
    yield
    # Cleanup code runs after test
    # Could delete temporary files, close connections, etc.
    pass
