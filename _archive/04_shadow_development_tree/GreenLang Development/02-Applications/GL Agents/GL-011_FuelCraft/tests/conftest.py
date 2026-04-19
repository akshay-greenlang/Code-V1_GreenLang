# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft Test Configuration

Shared pytest fixtures for comprehensive testing of the FuelCraft
fuel management optimization agent.

Provides:
- Sample fuel data (natural gas, diesel, HFO, biomass)
- Sample contracts (take-or-pay, spot)
- Sample inventory (tanks, levels)
- Mock Kafka producer/consumer
- Optimization configuration fixtures
- Deterministic test data generators

Author: GL-TestEngineer
Date: 2025-01-01
Version: 1.0.0
"""

import sys
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib
import json
import uuid

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fuel Data Fixtures
# =============================================================================

@pytest.fixture
def sample_natural_gas_properties() -> Dict[str, Any]:
    """Natural gas fuel properties for testing."""
    return {
        "fuel_id": "NG-001",
        "fuel_type": "natural_gas",
        "lhv_mj_kg": Decimal("50.00"),
        "hhv_mj_kg": Decimal("55.50"),
        "density_kg_m3": Decimal("0.717"),
        "sulfur_wt_pct": Decimal("0.0"),
        "ash_wt_pct": Decimal("0.0"),
        "water_vol_pct": Decimal("0.0"),
        "viscosity_50c_cst": Decimal("0.1"),
        "flash_point_c": Decimal("-188.0"),
        "vapor_pressure_kpa": Decimal("101.325"),
        "carbon_intensity_kg_co2e_mj": Decimal("0.0561"),
        "price_per_mj": Decimal("0.0035"),
    }


@pytest.fixture
def sample_diesel_properties() -> Dict[str, Any]:
    """Diesel fuel properties for testing."""
    return {
        "fuel_id": "DIESEL-001",
        "fuel_type": "diesel",
        "lhv_mj_kg": Decimal("43.00"),
        "hhv_mj_kg": Decimal("45.80"),
        "density_kg_m3": Decimal("840.0"),
        "sulfur_wt_pct": Decimal("0.05"),
        "ash_wt_pct": Decimal("0.01"),
        "water_vol_pct": Decimal("0.02"),
        "viscosity_50c_cst": Decimal("3.5"),
        "flash_point_c": Decimal("65.0"),
        "vapor_pressure_kpa": Decimal("0.5"),
        "carbon_intensity_kg_co2e_mj": Decimal("0.0741"),
        "price_per_mj": Decimal("0.0045"),
    }


@pytest.fixture
def sample_hfo_properties() -> Dict[str, Any]:
    """Heavy Fuel Oil properties for testing."""
    return {
        "fuel_id": "HFO-001",
        "fuel_type": "heavy_fuel_oil",
        "lhv_mj_kg": Decimal("40.00"),
        "hhv_mj_kg": Decimal("42.50"),
        "density_kg_m3": Decimal("990.0"),
        "sulfur_wt_pct": Decimal("3.5"),
        "ash_wt_pct": Decimal("0.10"),
        "water_vol_pct": Decimal("0.50"),
        "viscosity_50c_cst": Decimal("380.0"),
        "flash_point_c": Decimal("70.0"),
        "vapor_pressure_kpa": Decimal("0.1"),
        "carbon_intensity_kg_co2e_mj": Decimal("0.0771"),
        "price_per_mj": Decimal("0.0028"),
    }


@pytest.fixture
def sample_biomass_properties() -> Dict[str, Any]:
    """Biomass fuel properties for testing."""
    return {
        "fuel_id": "BIOMASS-001",
        "fuel_type": "biomass_wood",
        "lhv_mj_kg": Decimal("15.50"),
        "hhv_mj_kg": Decimal("17.00"),
        "density_kg_m3": Decimal("400.0"),
        "sulfur_wt_pct": Decimal("0.02"),
        "ash_wt_pct": Decimal("1.50"),
        "water_vol_pct": Decimal("15.0"),
        "viscosity_50c_cst": Decimal("0.0"),
        "flash_point_c": Decimal("250.0"),
        "vapor_pressure_kpa": Decimal("0.0"),
        "carbon_intensity_kg_co2e_mj": Decimal("0.0150"),
        "price_per_mj": Decimal("0.0032"),
    }


@pytest.fixture
def sample_fuel_data(
    sample_natural_gas_properties,
    sample_diesel_properties,
    sample_hfo_properties,
    sample_biomass_properties
) -> List[Dict[str, Any]]:
    """Complete set of sample fuel data for testing."""
    return [
        sample_natural_gas_properties,
        sample_diesel_properties,
        sample_hfo_properties,
        sample_biomass_properties,
    ]


# =============================================================================
# Blend Calculator Fixtures
# =============================================================================

@pytest.fixture
def blend_component_natural_gas(sample_natural_gas_properties):
    """Create BlendComponent for natural gas."""
    from calculators.blend_calculator import BlendComponent

    props = sample_natural_gas_properties
    return BlendComponent(
        component_id=props["fuel_id"],
        fuel_type=props["fuel_type"],
        mass_kg=Decimal("1000"),
        lhv_mj_kg=props["lhv_mj_kg"],
        hhv_mj_kg=props["hhv_mj_kg"],
        density_kg_m3=props["density_kg_m3"],
        sulfur_wt_pct=props["sulfur_wt_pct"],
        ash_wt_pct=props["ash_wt_pct"],
        water_vol_pct=props["water_vol_pct"],
        viscosity_50c_cst=props["viscosity_50c_cst"],
        flash_point_c=props["flash_point_c"],
        vapor_pressure_kpa=props["vapor_pressure_kpa"],
        carbon_intensity_kg_co2e_mj=props["carbon_intensity_kg_co2e_mj"],
    )


@pytest.fixture
def blend_component_diesel(sample_diesel_properties):
    """Create BlendComponent for diesel."""
    from calculators.blend_calculator import BlendComponent

    props = sample_diesel_properties
    return BlendComponent(
        component_id=props["fuel_id"],
        fuel_type=props["fuel_type"],
        mass_kg=Decimal("1000"),
        lhv_mj_kg=props["lhv_mj_kg"],
        hhv_mj_kg=props["hhv_mj_kg"],
        density_kg_m3=props["density_kg_m3"],
        sulfur_wt_pct=props["sulfur_wt_pct"],
        ash_wt_pct=props["ash_wt_pct"],
        water_vol_pct=props["water_vol_pct"],
        viscosity_50c_cst=props["viscosity_50c_cst"],
        flash_point_c=props["flash_point_c"],
        vapor_pressure_kpa=props["vapor_pressure_kpa"],
        carbon_intensity_kg_co2e_mj=props["carbon_intensity_kg_co2e_mj"],
    )


@pytest.fixture
def blend_component_hfo(sample_hfo_properties):
    """Create BlendComponent for heavy fuel oil."""
    from calculators.blend_calculator import BlendComponent

    props = sample_hfo_properties
    return BlendComponent(
        component_id=props["fuel_id"],
        fuel_type=props["fuel_type"],
        mass_kg=Decimal("1000"),
        lhv_mj_kg=props["lhv_mj_kg"],
        hhv_mj_kg=props["hhv_mj_kg"],
        density_kg_m3=props["density_kg_m3"],
        sulfur_wt_pct=props["sulfur_wt_pct"],
        ash_wt_pct=props["ash_wt_pct"],
        water_vol_pct=props["water_vol_pct"],
        viscosity_50c_cst=props["viscosity_50c_cst"],
        flash_point_c=props["flash_point_c"],
        vapor_pressure_kpa=props["vapor_pressure_kpa"],
        carbon_intensity_kg_co2e_mj=props["carbon_intensity_kg_co2e_mj"],
    )


@pytest.fixture
def blend_calculator():
    """Create BlendCalculator instance."""
    from calculators.blend_calculator import BlendCalculator
    return BlendCalculator()


# =============================================================================
# Contract Fixtures
# =============================================================================

@pytest.fixture
def sample_take_or_pay_contract() -> Dict[str, Any]:
    """Sample take-or-pay contract for testing."""
    now = datetime.now(timezone.utc)
    return {
        "contract_id": "CONTRACT-TOP-001",
        "contract_type": "take_or_pay",
        "fuel_id": "NG-001",
        "supplier_id": "SUPPLIER-001",
        "supplier_name": "Natural Gas Co.",
        "min_quantity_mj": Decimal("50000000"),
        "max_quantity_mj": Decimal("200000000"),
        "price_per_mj": Decimal("0.0035"),
        "penalty_per_mj_shortfall": Decimal("0.0010"),
        "start_date": now - timedelta(days=30),
        "end_date": now + timedelta(days=335),
        "start_period": 1,
        "end_period": 12,
    }


@pytest.fixture
def sample_spot_contract() -> Dict[str, Any]:
    """Sample spot purchase contract for testing."""
    now = datetime.now(timezone.utc)
    return {
        "contract_id": "CONTRACT-SPOT-001",
        "contract_type": "spot",
        "fuel_id": "DIESEL-001",
        "supplier_id": "SUPPLIER-002",
        "supplier_name": "Diesel Distributors",
        "min_quantity_mj": Decimal("0"),
        "max_quantity_mj": Decimal("100000000"),
        "price_per_mj": Decimal("0.0048"),
        "penalty_per_mj_shortfall": Decimal("0"),
        "start_date": now,
        "end_date": now + timedelta(days=30),
        "start_period": 1,
        "end_period": 12,
    }


@pytest.fixture
def sample_contracts(sample_take_or_pay_contract, sample_spot_contract) -> List[Dict[str, Any]]:
    """Complete set of sample contracts."""
    return [sample_take_or_pay_contract, sample_spot_contract]


# =============================================================================
# Inventory Fixtures
# =============================================================================

@pytest.fixture
def sample_tank_ng() -> Dict[str, Any]:
    """Natural gas storage tank data."""
    return {
        "tank_id": "TANK-NG-001",
        "fuel_type": "natural_gas",
        "capacity_mj": Decimal("500000000"),
        "min_level_mj": Decimal("25000000"),
        "max_level_mj": Decimal("475000000"),
        "initial_level_mj": Decimal("250000000"),
        "loss_rate_per_period": Decimal("0.001"),
        "compatible_fuels": ["NG-001"],
    }


@pytest.fixture
def sample_tank_liquid() -> Dict[str, Any]:
    """Liquid fuel storage tank data."""
    return {
        "tank_id": "TANK-LIQUID-001",
        "fuel_type": "mixed_liquid",
        "capacity_mj": Decimal("200000000"),
        "min_level_mj": Decimal("10000000"),
        "max_level_mj": Decimal("190000000"),
        "initial_level_mj": Decimal("100000000"),
        "loss_rate_per_period": Decimal("0.0005"),
        "compatible_fuels": ["DIESEL-001", "HFO-001"],
    }


@pytest.fixture
def sample_inventory(sample_tank_ng, sample_tank_liquid) -> List[Dict[str, Any]]:
    """Complete inventory data."""
    return [sample_tank_ng, sample_tank_liquid]


# =============================================================================
# Optimization Configuration Fixtures
# =============================================================================

@pytest.fixture
def optimization_config() -> Dict[str, Any]:
    """Optimization solver configuration."""
    return {
        "solver_type": "highs",
        "time_limit_seconds": 300.0,
        "mip_gap": 0.01,
        "threads": 1,
        "random_seed": 42,
        "presolve": True,
        "verbose": False,
        "deterministic": True,
    }


@pytest.fixture
def model_config() -> Dict[str, Any]:
    """Fuel optimization model configuration."""
    return {
        "model_name": "FuelProcurement_Test",
        "objective_type": "minimize",
        "time_periods": 12,
        "carbon_price_per_kg_co2e": Decimal("0.050"),
        "include_logistics": True,
        "logistics_cost_per_mj": Decimal("0.0001"),
        "big_m": Decimal("1e9"),
        "tolerance": Decimal("1e-6"),
    }


@pytest.fixture
def sample_demands() -> List[Dict[str, Any]]:
    """Sample energy demands for optimization."""
    base_demand = Decimal("50000000")  # 50 TJ base demand
    demands = []

    # Create 12 periods with seasonal variation
    seasonal_factors = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    for period, factor in enumerate(seasonal_factors, start=1):
        demands.append({
            "period": period,
            "demand_mj": base_demand * Decimal(str(factor)),
            "max_sulfur_pct": Decimal("0.50"),
            "max_carbon_intensity": Decimal("0.0700"),
            "min_flash_point_c": Decimal("60.0"),
        })

    return demands


# =============================================================================
# Carbon Calculator Fixtures
# =============================================================================

@pytest.fixture
def carbon_calculator():
    """Create CarbonCalculator instance."""
    from calculators.carbon_calculator import CarbonCalculator
    return CarbonCalculator()


@pytest.fixture
def sample_emission_factors() -> Dict[str, Decimal]:
    """Sample emission factors for testing."""
    return {
        "diesel_ttw": Decimal("0.0741"),
        "diesel_wtt": Decimal("0.0161"),
        "diesel_wtw": Decimal("0.0902"),
        "natural_gas_ttw": Decimal("0.0561"),
        "natural_gas_wtt": Decimal("0.0183"),
        "natural_gas_wtw": Decimal("0.0744"),
        "hfo_ttw": Decimal("0.0771"),
        "hfo_wtt": Decimal("0.0124"),
        "hfo_wtw": Decimal("0.0895"),
    }


# =============================================================================
# Heating Value Calculator Fixtures
# =============================================================================

@pytest.fixture
def heating_value_calculator():
    """Create HeatingValueCalculator instance."""
    from calculators.heating_value_calculator import HeatingValueCalculator
    return HeatingValueCalculator()


# =============================================================================
# Unit Converter Fixtures
# =============================================================================

@pytest.fixture
def unit_converter():
    """Create UnitConverter instance."""
    from calculators.unit_converter import UnitConverter
    return UnitConverter()


# =============================================================================
# Circuit Breaker Fixtures
# =============================================================================

@pytest.fixture
def circuit_breaker_config():
    """Circuit breaker configuration for testing."""
    from safety.circuit_breaker import CircuitBreakerConfig, SILLevel, FailureMode, RecoveryStrategy

    return CircuitBreakerConfig(
        circuit_id="test_circuit",
        sil_level=SILLevel.SIL_2,
        failure_mode=FailureMode.FAIL_CLOSED,
        recovery_strategy=RecoveryStrategy.AUTOMATIC,
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=60.0,
        initial_backoff_seconds=10.0,
        max_backoff_seconds=300.0,
        backoff_multiplier=2.0,
    )


@pytest.fixture
def circuit_breaker(circuit_breaker_config):
    """Create CircuitBreaker instance."""
    from safety.circuit_breaker import CircuitBreaker
    return CircuitBreaker(circuit_breaker_config)


# =============================================================================
# Kafka Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing."""
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock(return_value=None)
    producer._started = True
    producer._messages_sent = 0
    producer._messages_failed = 0
    producer._bytes_sent = 0
    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer for testing."""
    consumer = AsyncMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.getmany = AsyncMock(return_value={})
    consumer.commit = AsyncMock()
    return consumer


@pytest.fixture
def kafka_producer_config():
    """Kafka producer configuration for testing."""
    return {
        "bootstrap_servers": "localhost:9092",
        "security_protocol": "PLAINTEXT",
        "acks": "all",
        "retries": 3,
        "batch_size": 16384,
        "linger_ms": 5,
        "compression_type": "gzip",
        "enable_idempotence": True,
    }


# =============================================================================
# ERP Connector Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_erp_connector():
    """Mock ERP connector for testing."""
    connector = AsyncMock()
    connector.connect = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock()
    connector.is_connected = True

    # Mock contract retrieval
    connector.get_contracts = AsyncMock(return_value=[])
    connector.create_order = AsyncMock()
    connector.get_delivery_schedule = AsyncMock(return_value=[])

    return connector


@pytest.fixture
def erp_config():
    """ERP connector configuration for testing."""
    return {
        "system_type": "generic_rest",
        "base_url": "https://erp.test.local/api/v1",
        "auth_type": "api_key",
        "company_code": "TEST001",
        "timeout_seconds": 30,
        "max_retries": 3,
    }


# =============================================================================
# Provenance and Audit Fixtures
# =============================================================================

@pytest.fixture
def provenance_tracker():
    """Create ProvenanceTracker instance."""
    from audit.provenance import ProvenanceTracker
    run_id = f"RUN-TEST-{uuid.uuid4().hex[:8].upper()}"
    return ProvenanceTracker(run_id=run_id)


@pytest.fixture
def run_bundle_builder():
    """Create RunBundleBuilder instance."""
    from audit.run_bundle import RunBundleBuilder
    run_id = f"RUN-TEST-{uuid.uuid4().hex[:8].upper()}"
    return RunBundleBuilder(
        run_id=run_id,
        agent_version="1.0.0",
        environment="test",
    )


# =============================================================================
# Model Builder Fixtures
# =============================================================================

@pytest.fixture
def fuel_data_objects(sample_fuel_data):
    """Create FuelData objects for optimization."""
    from optimization.model_builder import FuelData

    fuel_objects = []
    for fuel in sample_fuel_data:
        fuel_objects.append(FuelData(
            fuel_id=fuel["fuel_id"],
            fuel_type=fuel["fuel_type"],
            lhv_mj_kg=fuel["lhv_mj_kg"],
            density_kg_m3=fuel["density_kg_m3"],
            price_per_mj=fuel["price_per_mj"],
            carbon_intensity_kg_co2e_mj=fuel["carbon_intensity_kg_co2e_mj"],
            sulfur_wt_pct=fuel["sulfur_wt_pct"],
            ash_wt_pct=fuel["ash_wt_pct"],
            viscosity_cst=fuel.get("viscosity_50c_cst", Decimal("1.0")),
            flash_point_c=fuel["flash_point_c"],
            min_order_mj=Decimal("1000000"),
            max_order_mj=Decimal("100000000"),
        ))

    return fuel_objects


@pytest.fixture
def tank_data_objects(sample_inventory):
    """Create TankData objects for optimization."""
    from optimization.model_builder import TankData

    tank_objects = []
    for tank in sample_inventory:
        tank_objects.append(TankData(
            tank_id=tank["tank_id"],
            capacity_mj=tank["capacity_mj"],
            min_level_mj=tank["min_level_mj"],
            max_level_mj=tank["max_level_mj"],
            initial_level_mj=tank["initial_level_mj"],
            loss_rate_per_period=tank["loss_rate_per_period"],
            compatible_fuels=tank["compatible_fuels"],
        ))

    return tank_objects


@pytest.fixture
def demand_data_objects(sample_demands):
    """Create DemandData objects for optimization."""
    from optimization.model_builder import DemandData

    demand_objects = []
    for demand in sample_demands:
        demand_objects.append(DemandData(
            period=demand["period"],
            demand_mj=demand["demand_mj"],
            max_sulfur_pct=demand["max_sulfur_pct"],
            max_carbon_intensity=demand.get("max_carbon_intensity"),
            min_flash_point_c=demand["min_flash_point_c"],
        ))

    return demand_objects


@pytest.fixture
def contract_data_objects(sample_contracts):
    """Create ContractData objects for optimization."""
    from optimization.model_builder import ContractData

    contract_objects = []
    for contract in sample_contracts:
        contract_objects.append(ContractData(
            contract_id=contract["contract_id"],
            fuel_id=contract["fuel_id"],
            min_quantity_mj=contract["min_quantity_mj"],
            max_quantity_mj=contract["max_quantity_mj"],
            price_per_mj=contract["price_per_mj"],
            penalty_per_mj_shortfall=contract["penalty_per_mj_shortfall"],
            start_period=contract["start_period"],
            end_period=contract["end_period"],
        ))

    return contract_objects


# =============================================================================
# Test Data Generation Helpers
# =============================================================================

@pytest.fixture
def deterministic_seed():
    """Fixed seed for deterministic test data generation."""
    return 42


@pytest.fixture
def test_timestamp():
    """Fixed timestamp for deterministic testing."""
    return datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def generate_test_run_id():
    """Generate a deterministic test run ID."""
    def _generate(seed: int = 42) -> str:
        hash_input = f"test_run_{seed}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"RUN-TEST-{hash_value.upper()}"
    return _generate


# =============================================================================
# Performance Fixtures
# =============================================================================

@pytest.fixture
def performance_threshold_ms():
    """Maximum allowed execution time in milliseconds."""
    return 5.0  # 5ms target for calculations


@pytest.fixture
def throughput_target_records_per_sec():
    """Target throughput for batch operations."""
    return 1000  # 1000 records/second


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_circuit_breaker_registry():
    """Clean up circuit breaker registry after each test."""
    yield
    # Reset singleton registry if needed
    try:
        from safety.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()
        registry._breakers = {}
    except Exception:
        pass


# =============================================================================
# Skip Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "golden: mark test as golden value test"
    )
    config.addinivalue_line(
        "markers", "property: mark test as property-based test"
    )
    config.addinivalue_line(
        "markers", "safety: mark test as safety-critical test"
    )
