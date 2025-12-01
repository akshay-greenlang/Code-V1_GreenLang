# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Test Suite - Shared Pytest Fixtures.

This module provides comprehensive test fixtures for the FuelManagementOrchestrator
test suite, including sample fuel specifications, market data, inventory fixtures,
mock ERP/market connectors, and test configurations.

Coverage Target: 85%+
Standards Compliance:
- ISO 6976:2016 - Natural gas calorific value
- ISO 17225 - Solid biofuels specifications
- ASTM D4809 - Heat of combustion
- GHG Protocol - Emissions calculations

Author: GreenLang Industrial Optimization Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add parent directories to path for imports
TEST_DIR = Path(__file__).parent
AGENT_DIR = TEST_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR.parent.parent))

# Import agent modules
try:
    from config import (
        FuelManagementConfig,
        FuelSpecification,
        FuelInventory,
        MarketPriceData,
        BlendingConstraints,
        EmissionLimits,
        OptimizationParameters,
        IntegrationSettings,
        FuelCategory,
        FuelState,
        EmissionStandard,
        create_default_config,
    )
    from fuel_management_orchestrator import (
        FuelManagementOrchestrator,
    )
    from calculators.multi_fuel_optimizer import (
        MultiFuelOptimizer,
        MultiFuelOptimizationInput,
    )
    from calculators.cost_optimization_calculator import (
        CostOptimizationCalculator,
        CostOptimizationInput,
    )
    from calculators.fuel_blending_calculator import (
        FuelBlendingCalculator,
        BlendingInput,
    )
    from calculators.carbon_footprint_calculator import (
        CarbonFootprintCalculator,
        CarbonFootprintInput,
    )
    from calculators.provenance_tracker import (
        ProvenanceTracker,
    )
except ImportError as e:
    print(f"Warning: Import error during fixture setup: {e}")


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "determinism: Determinism tests")
    config.addinivalue_line("markers", "compliance: Regulatory compliance tests")
    config.addinivalue_line("markers", "golden: Golden test cases")
    config.addinivalue_line("markers", "concurrency: Thread safety tests")
    config.addinivalue_line("markers", "edge_case: Edge case tests")
    config.addinivalue_line("markers", "error_path: Error handling tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "asyncio: Async tests")


# =============================================================================
# EVENT LOOP FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def base_config() -> FuelManagementConfig:
    """Create base configuration for testing."""
    return create_default_config()


@pytest.fixture
def optimization_params() -> OptimizationParameters:
    """Create optimization parameters."""
    return OptimizationParameters(
        primary_objective="balanced",
        cost_weight=0.4,
        emissions_weight=0.3,
        efficiency_weight=0.2,
        reliability_weight=0.1,
        optimization_algorithm="linear_programming",
        max_iterations=1000,
        convergence_tolerance=0.0001,
        time_limit_seconds=60,
    )


# =============================================================================
# FUEL SPECIFICATION FIXTURES
# =============================================================================

@pytest.fixture
def natural_gas_spec() -> FuelSpecification:
    """Create natural gas specification."""
    return FuelSpecification(
        fuel_id="NG-001",
        fuel_name="Natural Gas",
        fuel_type="natural_gas",
        category=FuelCategory.FOSSIL,
        state=FuelState.GAS,
        gross_calorific_value_mj_kg=55.5,
        net_calorific_value_mj_kg=50.0,
        density_kg_m3=0.75,
        carbon_content_percent=75.0,
        hydrogen_content_percent=25.0,
        emission_factor_co2_kg_gj=56.1,
        emission_factor_nox_g_gj=50.0,
        emission_factor_sox_g_gj=0.3,
        is_renewable=False,
    )


@pytest.fixture
def coal_spec() -> FuelSpecification:
    """Create bituminous coal specification."""
    return FuelSpecification(
        fuel_id="COAL-001",
        fuel_name="Bituminous Coal",
        fuel_type="coal",
        category=FuelCategory.FOSSIL,
        state=FuelState.SOLID,
        gross_calorific_value_mj_kg=28.0,
        net_calorific_value_mj_kg=25.0,
        density_kg_m3=1350.0,
        bulk_density_kg_m3=800.0,
        carbon_content_percent=60.0,
        hydrogen_content_percent=4.0,
        oxygen_content_percent=8.0,
        nitrogen_content_percent=1.0,
        sulfur_content_percent=2.0,
        moisture_content_percent=8.0,
        ash_content_percent=10.0,
        emission_factor_co2_kg_gj=94.6,
        emission_factor_nox_g_gj=250.0,
        emission_factor_sox_g_gj=500.0,
        emission_factor_pm_g_gj=50.0,
        is_renewable=False,
    )


@pytest.fixture
def biomass_spec() -> FuelSpecification:
    """Create biomass wood pellets specification."""
    return FuelSpecification(
        fuel_id="BIO-001",
        fuel_name="Wood Pellets",
        fuel_type="biomass",
        category=FuelCategory.RENEWABLE,
        state=FuelState.SOLID,
        gross_calorific_value_mj_kg=19.0,
        net_calorific_value_mj_kg=17.5,
        density_kg_m3=1200.0,
        bulk_density_kg_m3=650.0,
        carbon_content_percent=50.0,
        hydrogen_content_percent=6.0,
        oxygen_content_percent=43.0,
        nitrogen_content_percent=0.3,
        sulfur_content_percent=0.02,
        moisture_content_percent=8.0,
        ash_content_percent=0.5,
        emission_factor_co2_kg_gj=0.0,  # Biogenic
        emission_factor_nox_g_gj=120.0,
        emission_factor_sox_g_gj=10.0,
        emission_factor_pm_g_gj=30.0,
        is_renewable=True,
        biogenic_carbon_percent=100.0,
        certification="ENplus A1",
    )


@pytest.fixture
def hydrogen_spec() -> FuelSpecification:
    """Create green hydrogen specification."""
    return FuelSpecification(
        fuel_id="H2-001",
        fuel_name="Green Hydrogen",
        fuel_type="hydrogen",
        category=FuelCategory.RENEWABLE,
        state=FuelState.GAS,
        gross_calorific_value_mj_kg=142.0,
        net_calorific_value_mj_kg=120.0,
        density_kg_m3=0.09,
        carbon_content_percent=0.0,
        hydrogen_content_percent=100.0,
        emission_factor_co2_kg_gj=0.0,
        emission_factor_nox_g_gj=10.0,
        emission_factor_sox_g_gj=0.0,
        is_renewable=True,
        flash_point_c=-253,
        auto_ignition_temp_c=500,
        explosive_limits_lower_percent=4.0,
        explosive_limits_upper_percent=75.0,
    )


@pytest.fixture
def fuel_oil_spec() -> FuelSpecification:
    """Create fuel oil No. 2 specification."""
    return FuelSpecification(
        fuel_id="OIL-002",
        fuel_name="Fuel Oil No. 2",
        fuel_type="fuel_oil",
        category=FuelCategory.FOSSIL,
        state=FuelState.LIQUID,
        gross_calorific_value_mj_kg=45.5,
        net_calorific_value_mj_kg=42.7,
        density_kg_m3=850.0,
        carbon_content_percent=87.0,
        hydrogen_content_percent=12.5,
        oxygen_content_percent=0.1,
        nitrogen_content_percent=0.01,
        sulfur_content_percent=0.5,
        ash_content_percent=0.01,
        moisture_content_percent=0.1,
        emission_factor_co2_kg_gj=77.4,
        emission_factor_nox_g_gj=150.0,
        emission_factor_sox_g_gj=350.0,
        emission_factor_pm_g_gj=20.0,
        is_renewable=False,
        flash_point_c=52,
    )


@pytest.fixture
def fuel_properties() -> Dict[str, Dict[str, Any]]:
    """Create fuel properties dictionary for testing."""
    return {
        'natural_gas': {
            'heating_value_mj_kg': 50.0,
            'emission_factor_co2_kg_gj': 56.1,
            'emission_factor_nox_g_gj': 50,
            'renewable': False,
        },
        'coal': {
            'heating_value_mj_kg': 25.0,
            'emission_factor_co2_kg_gj': 94.6,
            'emission_factor_nox_g_gj': 250,
            'renewable': False,
        },
        'biomass': {
            'heating_value_mj_kg': 18.0,
            'emission_factor_co2_kg_gj': 0.0,
            'emission_factor_nox_g_gj': 150,
            'renewable': True,
        },
        'hydrogen': {
            'heating_value_mj_kg': 120.0,
            'emission_factor_co2_kg_gj': 0.0,
            'emission_factor_nox_g_gj': 10,
            'renewable': True,
        },
    }


# =============================================================================
# MARKET PRICE FIXTURES
# =============================================================================

@pytest.fixture
def market_prices() -> Dict[str, float]:
    """Create standard market prices for testing."""
    return {
        'natural_gas': 0.045,  # USD/kg
        'coal': 0.035,
        'biomass': 0.08,
        'hydrogen': 0.25,
        'fuel_oil': 0.055,
    }


@pytest.fixture
def volatile_market_prices() -> Dict[str, float]:
    """Create volatile market prices (extreme scenario)."""
    return {
        'natural_gas': 0.450,  # 10x spike
        'coal': 0.035,
        'biomass': 0.08,
    }


@pytest.fixture
def market_price_data_natural_gas() -> MarketPriceData:
    """Create market price data for natural gas."""
    return MarketPriceData(
        fuel_id="NG-001",
        price_source="NYMEX",
        current_price=0.045,
        price_unit="USD/kg",
        currency="USD",
        price_low_24h=0.043,
        price_high_24h=0.048,
        price_avg_30d=0.046,
        volatility_percent=8.5,
        delivery_premium=0.002,
        minimum_order_quantity=10000,
    )


# =============================================================================
# INVENTORY FIXTURES
# =============================================================================

@pytest.fixture
def natural_gas_inventory() -> FuelInventory:
    """Create natural gas inventory."""
    return FuelInventory(
        fuel_id="NG-001",
        site_id="SITE-001",
        storage_unit_id="NG-TANK-01",
        current_quantity=50000,
        quantity_unit="kg",
        storage_capacity=100000,
        minimum_level=10000,
        safety_stock=15000,
        reorder_point=20000,
        reorder_quantity=40000,
        lead_time_days=3,
        average_cost_per_unit=0.045,
        last_purchase_price=0.046,
    )


@pytest.fixture
def fuel_inventories() -> Dict[str, float]:
    """Create fuel inventory dictionary."""
    return {
        'natural_gas': 100000,
        'coal': 100000,
        'biomass': 100000,
        'hydrogen': 50000,
    }


# =============================================================================
# BLENDING CONSTRAINT FIXTURES
# =============================================================================

@pytest.fixture
def blending_constraints() -> BlendingConstraints:
    """Create fuel blending constraints."""
    return BlendingConstraints(
        blend_id="BLEND-001",
        blend_name="Coal-Biomass Blend",
        fuel_limits={
            "coal": {"min": 0.4, "max": 0.7},
            "biomass": {"min": 0.3, "max": 0.6},
        },
        min_heating_value_mj_kg=20.0,
        max_moisture_percent=15.0,
        max_ash_percent=12.0,
        max_sulfur_percent=1.5,
        incompatible_fuels=[],
        max_blend_components=3,
        min_component_percent=10.0,
    )


# =============================================================================
# EMISSION LIMIT FIXTURES
# =============================================================================

@pytest.fixture
def epa_emission_limits() -> EmissionLimits:
    """Create EPA emission limits."""
    return EmissionLimits(
        limit_id="EPA-NSPS-001",
        standard=EmissionStandard.EPA_NSPS,
        jurisdiction="US",
        nox_limit_mg_nm3=150,
        sox_limit_mg_nm3=200,
        pm_limit_mg_nm3=30,
        co_limit_mg_nm3=100,
        reference_oxygen_percent=3.0,
    )


@pytest.fixture
def eu_ied_emission_limits() -> EmissionLimits:
    """Create EU IED emission limits."""
    return EmissionLimits(
        limit_id="EU-IED-001",
        standard=EmissionStandard.EU_IED,
        jurisdiction="EU",
        nox_limit_mg_nm3=100,
        sox_limit_mg_nm3=150,
        pm_limit_mg_nm3=10,
        co_limit_mg_nm3=100,
        reference_oxygen_percent=6.0,
    )


# =============================================================================
# COMPONENT FIXTURES
# =============================================================================

@pytest.fixture
def orchestrator(base_config) -> FuelManagementOrchestrator:
    """Create FuelManagementOrchestrator instance."""
    return FuelManagementOrchestrator(base_config)


@pytest.fixture
def multi_fuel_optimizer() -> MultiFuelOptimizer:
    """Create MultiFuelOptimizer instance."""
    return MultiFuelOptimizer()


@pytest.fixture
def cost_calculator() -> CostOptimizationCalculator:
    """Create CostOptimizationCalculator instance."""
    return CostOptimizationCalculator()


@pytest.fixture
def blending_calculator() -> FuelBlendingCalculator:
    """Create FuelBlendingCalculator instance."""
    return FuelBlendingCalculator()


@pytest.fixture
def carbon_calculator() -> CarbonFootprintCalculator:
    """Create CarbonFootprintCalculator instance."""
    return CarbonFootprintCalculator()


@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create ProvenanceTracker instance."""
    return ProvenanceTracker()


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_erp_connector():
    """Create mock ERP connector."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    mock.is_connected.return_value = True
    mock.get_fuel_prices.return_value = {
        'natural_gas': 0.045,
        'coal': 0.035,
        'biomass': 0.08,
    }
    mock.get_inventory_levels.return_value = {
        'natural_gas': 100000,
        'coal': 100000,
        'biomass': 100000,
    }
    mock.create_purchase_order.return_value = {
        'po_id': 'PO-12345',
        'status': 'created',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    return mock


@pytest.fixture
def mock_market_data_connector():
    """Create mock market data connector."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.get_current_price.return_value = 0.045
    mock.get_price_history.return_value = [
        {'timestamp': '2025-01-01T00:00:00Z', 'price': 0.043},
        {'timestamp': '2025-01-01T12:00:00Z', 'price': 0.045},
        {'timestamp': '2025-01-02T00:00:00Z', 'price': 0.046},
    ]
    mock.get_volatility.return_value = 8.5
    mock.is_data_stale.return_value = False
    return mock


@pytest.fixture
def mock_storage_connector():
    """Create mock fuel storage connector."""
    mock = MagicMock()
    mock.get_current_level.return_value = 50000
    mock.get_capacity.return_value = 100000
    mock.get_temperature.return_value = 15.0
    mock.get_pressure.return_value = 101.325
    mock.is_within_safety_limits.return_value = True
    return mock


@pytest.fixture
def mock_emissions_monitoring_connector():
    """Create mock emissions monitoring connector."""
    mock = MagicMock()
    mock.get_current_emissions.return_value = {
        'nox_mg_nm3': 120.0,
        'sox_mg_nm3': 100.0,
        'pm_mg_nm3': 8.0,
        'co2_percent': 10.5,
    }
    mock.check_compliance.return_value = True
    return mock


# =============================================================================
# GOLDEN TEST CASE FIXTURES
# =============================================================================

@pytest.fixture
def golden_test_cases() -> List[Dict[str, Any]]:
    """
    Golden test cases with known input-output pairs.

    These values must remain constant across all test runs.
    """
    return [
        {
            'name': 'single_fuel_natural_gas',
            'input': {
                'energy_demand_mw': 100,
                'available_fuels': ['natural_gas'],
                'fuel_properties': {
                    'natural_gas': {
                        'heating_value_mj_kg': 50.0,
                        'emission_factor_co2_kg_gj': 56.1,
                    }
                },
                'market_prices': {'natural_gas': 0.045},
            },
            'expected_output': {
                'optimal_fuel_mix': {'natural_gas': 1.0},
                'total_fuel_consumption_kg_tolerance': 7200.0,  # ±1%
                'cost_tolerance': 324.0,  # ±1%
            },
        },
        {
            'name': 'dual_fuel_coal_biomass',
            'input': {
                'energy_demand_mw': 100,
                'available_fuels': ['coal', 'biomass'],
                'fuel_properties': {
                    'coal': {
                        'heating_value_mj_kg': 25.0,
                        'emission_factor_co2_kg_gj': 94.6,
                    },
                    'biomass': {
                        'heating_value_mj_kg': 18.0,
                        'emission_factor_co2_kg_gj': 0.0,
                    },
                },
                'market_prices': {'coal': 0.035, 'biomass': 0.08},
            },
            'expected_output': {
                'blend_contains_both_fuels': True,
                'biomass_percentage_min': 0.2,  # At least 20% biomass
            },
        },
    ]


# =============================================================================
# CONCURRENCY TEST FIXTURES
# =============================================================================

@pytest.fixture
def thread_pool():
    """Create thread pool for concurrency tests."""
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=50)
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def concurrency_barrier():
    """Create barrier for synchronized concurrent access."""
    return threading.Barrier(10)


# =============================================================================
# EDGE CASE FIXTURES
# =============================================================================

@pytest.fixture
def zero_inventory():
    """Create zero inventory scenario."""
    return {
        'natural_gas': 0,
        'coal': 0,
        'biomass': 0,
    }


@pytest.fixture
def negative_price_scenario():
    """Create negative price scenario (subsidized fuel)."""
    return {
        'biomass': -0.01,  # Subsidized
        'coal': 0.035,
    }


# =============================================================================
# COMPLIANCE TEST FIXTURES
# =============================================================================

@pytest.fixture
def iso_6976_test_cases() -> List[Dict[str, Any]]:
    """ISO 6976 calorific value test cases."""
    return [
        {
            'name': 'methane_pure',
            'composition': {'CH4': 100.0},
            'expected_gcv_mj_m3': 37.8,
            'tolerance': 0.5,
        },
        {
            'name': 'natural_gas_typical',
            'composition': {
                'CH4': 95.0,
                'C2H6': 2.5,
                'C3H8': 0.5,
                'N2': 2.0,
            },
            'expected_gcv_mj_m3': 38.5,
            'tolerance': 1.0,
        },
    ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_provenance_hash(data: Dict[str, Any]) -> str:
    """Generate SHA-256 hash for provenance verification."""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


def assert_within_tolerance(actual: float, expected: float, tolerance: float, message: str = ""):
    """Assert value is within tolerance of expected."""
    assert abs(actual - expected) <= tolerance, (
        f"{message} Expected {expected} +/- {tolerance}, got {actual}"
    )


def assert_deterministic(results: List[Any], message: str = ""):
    """Assert all results are identical (deterministic)."""
    if len(results) < 2:
        return
    first = results[0]
    for i, result in enumerate(results[1:], 2):
        assert result == first, f"{message} Result {i} differs from result 1"


# =============================================================================
# TEMPORARY DIRECTORY FIXTURE
# =============================================================================

@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_directory, base_config) -> Path:
    """Create temporary configuration file."""
    config_file = temp_directory / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(base_config.model_dump(), f, default=str)
    return config_file


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Cleanup logic if needed


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_session():
    """Clean up after test session."""
    yield
    # Session cleanup logic if needed
