"""
GL-011 FUELCRAFT Test Fixtures

Shared pytest fixtures for all GL-011 test modules.
Provides consistent test data and mock objects.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch
import uuid

# Import module components
from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    FuelOptimizationConfig,
    FuelPricingConfig,
    BlendingConfig,
    SwitchingConfig,
    InventoryConfig,
    CostOptimizationConfig,
    EquipmentConfig,
    FuelType,
    PriceSource,
    OptimizationMode,
    SwitchingMode,
    AlertLevel,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelProperties,
    FuelPrice,
    BlendRecommendation,
    SwitchingRecommendation,
    InventoryStatus,
    CostAnalysis,
    FuelOptimizationInput,
    FuelOptimizationOutput,
    OptimizationResult,
    BlendStatus,
    InventoryAlertType,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.heating_value import (
    HeatingValueCalculator,
    HeatingValueInput,
    HeatingValueResult,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_pricing import (
    FuelPricingService,
    PriceQuote,
    PriceCache,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_blending import (
    FuelBlendingOptimizer,
    BlendInput,
    BlendOutput,
    BlendConstraints,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_switching import (
    FuelSwitchingController,
    SwitchingInput,
    SwitchingOutput,
    SwitchingState,
    TriggerType,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.inventory import (
    InventoryManager,
    TankConfig,
    TankStatus,
    LevelStatus,
    ConsumptionTracker,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.cost_optimization import (
    CostOptimizer,
    TotalCostInput,
    TotalCostOutput,
    CostBreakdown,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def fuel_pricing_config():
    """Create test fuel pricing configuration."""
    return FuelPricingConfig(
        primary_source=PriceSource.HENRY_HUB,
        secondary_source=PriceSource.REGIONAL_HUB,
        update_interval_minutes=15,
        price_history_days=90,
        forecast_horizon_days=30,
        basis_differential_enabled=True,
        transport_cost_enabled=True,
        taxes_included=True,
        currency="USD",
    )


@pytest.fixture
def blending_config():
    """Create test blending configuration."""
    return BlendingConfig(
        enabled=True,
        max_fuels_in_blend=3,
        min_wobbe_index=1300.0,
        max_wobbe_index=1400.0,
        wobbe_tolerance_pct=5.0,
        min_hhv_btu_scf=900.0,
        max_hhv_btu_scf=1200.0,
        min_primary_fuel_pct=50.0,
        max_blend_change_rate_pct_min=5.0,
        optimization_interval_minutes=60,
    )


@pytest.fixture
def switching_config():
    """Create test switching configuration."""
    return SwitchingConfig(
        mode=SwitchingMode.SEMI_AUTOMATIC,
        price_differential_trigger_pct=15.0,
        min_savings_usd_hr=100.0,
        payback_period_hours=24.0,
        min_run_time_hours=4.0,
        max_switches_per_day=4,
        switch_lockout_minutes=30,
        transition_duration_minutes=15,
        ramp_rate_pct_min=5.0,
        safety_interlock_enabled=True,
        require_purge=True,
        purge_duration_seconds=60,
        operator_confirmation_required=True,
        confirmation_timeout_minutes=15,
    )


@pytest.fixture
def inventory_config():
    """Create test inventory configuration."""
    return InventoryConfig(
        enabled=True,
        tanks={
            "TANK-001": {
                "fuel_type": "natural_gas",
                "capacity_gal": 10000,
                "usable_capacity_pct": 95.0,
                "heel_volume_gal": 100,
                "reorder_point_pct": 30.0,
                "low_level_pct": 25.0,
                "critical_level_pct": 15.0,
                "high_level_pct": 95.0,
            },
        },
        reorder_point_pct=30.0,
        safety_stock_days=3.0,
        economic_order_quantity_enabled=True,
        low_level_alert_pct=25.0,
        critical_level_pct=15.0,
        high_level_alert_pct=95.0,
        lead_time_days=2.0,
        delivery_window_hours=4,
        preferred_delivery_days=[1, 2, 3, 4, 5],
        forecast_horizon_days=14,
    )


@pytest.fixture
def cost_optimization_config():
    """Create test cost optimization configuration."""
    return CostOptimizationConfig(
        enabled=True,
        mode=OptimizationMode.MINIMUM_COST,
        include_fuel_cost=True,
        include_transport_cost=True,
        include_storage_cost=True,
        include_emissions_cost=True,
        include_maintenance_cost=True,
        include_efficiency_impact=True,
        carbon_price_usd_ton=50.0,
        carbon_price_escalation_pct=5.0,
        cost_weight=0.6,
        emissions_weight=0.3,
        reliability_weight=0.1,
        optimization_horizon_hours=168,
        rolling_window_hours=24,
    )


@pytest.fixture
def equipment_config():
    """Create test equipment configuration."""
    return EquipmentConfig(
        equipment_id="BOILER-001",
        name="Main Process Boiler",
        supported_fuels=[FuelType.NATURAL_GAS, FuelType.NO2_FUEL_OIL],
        primary_fuel=FuelType.NATURAL_GAS,
        dual_fuel_capable=True,
        design_capacity_mmbtu_hr=50.0,
        min_load_pct=25.0,
        max_load_pct=110.0,
        design_efficiency_pct=82.0,
        fuel_switch_time_minutes=15,
        requires_shutdown_for_switch=False,
    )


@pytest.fixture
def fuel_optimization_config(
    fuel_pricing_config,
    blending_config,
    switching_config,
    inventory_config,
    cost_optimization_config,
    equipment_config,
):
    """Create complete fuel optimization configuration."""
    return FuelOptimizationConfig(
        facility_id="TEST-FACILITY-001",
        name="Test Facility",
        primary_fuel=FuelType.NATURAL_GAS,
        available_fuels=[FuelType.NATURAL_GAS, FuelType.NO2_FUEL_OIL, FuelType.LPG_PROPANE],
        pricing=fuel_pricing_config,
        blending=blending_config,
        switching=switching_config,
        inventory=inventory_config,
        cost_optimization=cost_optimization_config,
        equipment=[equipment_config],
        agent_id="GL-011",
        agent_version="1.0.0",
        safety_level=2,
    )


# =============================================================================
# FUEL PROPERTIES FIXTURES
# =============================================================================

@pytest.fixture
def natural_gas_properties():
    """Create natural gas properties."""
    return FuelProperties(
        fuel_type="natural_gas",
        fuel_name="Natural Gas",
        hhv_btu_scf=1020.0,
        lhv_btu_scf=918.0,
        specific_gravity=0.60,
        wobbe_index=1317.0,
        co2_kg_mmbtu=53.06,
        methane_pct=95.0,
        ethane_pct=2.5,
        propane_pct=0.5,
        nitrogen_pct=1.5,
        co2_pct=0.5,
    )


@pytest.fixture
def fuel_oil_properties():
    """Create #2 fuel oil properties."""
    return FuelProperties(
        fuel_type="no2_fuel_oil",
        fuel_name="#2 Fuel Oil",
        hhv_btu_lb=19580.0,
        lhv_btu_lb=18410.0,
        density_lb_gal=7.21,
        co2_kg_mmbtu=73.16,
    )


@pytest.fixture
def propane_properties():
    """Create propane properties."""
    return FuelProperties(
        fuel_type="lpg_propane",
        fuel_name="Propane",
        hhv_btu_scf=2516.0,
        lhv_btu_scf=2315.0,
        specific_gravity=1.52,
        wobbe_index=2040.0,
        co2_kg_mmbtu=62.87,
    )


@pytest.fixture
def hydrogen_properties():
    """Create hydrogen properties."""
    return FuelProperties(
        fuel_type="hydrogen",
        fuel_name="Hydrogen",
        hhv_btu_scf=324.0,
        lhv_btu_scf=274.0,
        specific_gravity=0.07,
        wobbe_index=1226.0,
        co2_kg_mmbtu=0.0,
        hydrogen_pct=100.0,
    )


@pytest.fixture
def biogas_properties():
    """Create biogas properties."""
    return FuelProperties(
        fuel_type="biogas",
        fuel_name="Biogas",
        hhv_btu_scf=600.0,
        lhv_btu_scf=540.0,
        specific_gravity=0.90,
        wobbe_index=632.0,
        co2_kg_mmbtu=0.0,
        methane_pct=60.0,
        co2_pct=38.0,
        nitrogen_pct=1.5,
    )


@pytest.fixture
def all_fuel_properties(
    natural_gas_properties,
    fuel_oil_properties,
    propane_properties,
    hydrogen_properties,
    biogas_properties,
):
    """Create dictionary of all fuel properties."""
    return {
        "natural_gas": natural_gas_properties,
        "no2_fuel_oil": fuel_oil_properties,
        "lpg_propane": propane_properties,
        "hydrogen": hydrogen_properties,
        "biogas": biogas_properties,
    }


# =============================================================================
# PRICE FIXTURES
# =============================================================================

@pytest.fixture
def natural_gas_price():
    """Create natural gas price fixture."""
    now = datetime.now(timezone.utc)
    return FuelPrice(
        fuel_type="natural_gas",
        price=3.50,
        unit="USD/MMBTU",
        source="henry_hub",
        timestamp=now,
        effective_until=now + timedelta(hours=1),
        commodity_price=3.00,
        transport_cost=0.15,
        basis_differential=0.20,
        taxes=0.15,
        confidence=0.95,
    )


@pytest.fixture
def fuel_oil_price():
    """Create fuel oil price fixture."""
    now = datetime.now(timezone.utc)
    return FuelPrice(
        fuel_type="no2_fuel_oil",
        price=15.00,
        unit="USD/MMBTU",
        source="spot",
        timestamp=now,
        effective_until=now + timedelta(hours=1),
        commodity_price=13.50,
        transport_cost=0.50,
        basis_differential=0.0,
        taxes=1.00,
        confidence=0.90,
    )


@pytest.fixture
def propane_price():
    """Create propane price fixture."""
    now = datetime.now(timezone.utc)
    return FuelPrice(
        fuel_type="lpg_propane",
        price=8.50,
        unit="USD/MMBTU",
        source="spot",
        timestamp=now,
        effective_until=now + timedelta(hours=1),
        commodity_price=7.50,
        transport_cost=0.35,
        basis_differential=0.0,
        taxes=0.65,
        confidence=0.90,
    )


@pytest.fixture
def all_fuel_prices(natural_gas_price, fuel_oil_price, propane_price):
    """Create dictionary of all fuel prices."""
    return {
        "natural_gas": natural_gas_price,
        "no2_fuel_oil": fuel_oil_price,
        "lpg_propane": propane_price,
    }


# =============================================================================
# SERVICE FIXTURES
# =============================================================================

@pytest.fixture
def heating_value_calculator():
    """Create heating value calculator instance."""
    return HeatingValueCalculator(reference_temp_f=60.0)


@pytest.fixture
def fuel_pricing_service(fuel_pricing_config):
    """Create fuel pricing service instance."""
    return FuelPricingService(
        config=fuel_pricing_config,
        carbon_price_usd_ton=50.0,
    )


@pytest.fixture
def fuel_blending_optimizer(blending_config, heating_value_calculator):
    """Create fuel blending optimizer instance."""
    return FuelBlendingOptimizer(
        config=blending_config,
        heating_value_calculator=heating_value_calculator,
    )


@pytest.fixture
def fuel_switching_controller(switching_config):
    """Create fuel switching controller instance."""
    return FuelSwitchingController(config=switching_config)


@pytest.fixture
def inventory_manager(inventory_config):
    """Create inventory manager instance."""
    return InventoryManager(config=inventory_config)


@pytest.fixture
def cost_optimizer(cost_optimization_config):
    """Create cost optimizer instance."""
    return CostOptimizer(config=cost_optimization_config)


# =============================================================================
# INPUT DATA FIXTURES
# =============================================================================

@pytest.fixture
def heating_value_input_natural_gas():
    """Create heating value input for natural gas."""
    return HeatingValueInput(
        fuel_type="natural_gas",
        methane_pct=95.0,
        ethane_pct=2.5,
        propane_pct=0.5,
        nitrogen_pct=1.5,
        co2_pct=0.5,
        temperature_f=60.0,
        pressure_psia=14.696,
    )


@pytest.fixture
def heating_value_input_biogas():
    """Create heating value input for biogas."""
    return HeatingValueInput(
        fuel_type="biogas",
        methane_pct=60.0,
        co2_pct=38.0,
        nitrogen_pct=1.5,
        h2s_pct=0.5,
        temperature_f=60.0,
        pressure_psia=14.696,
    )


@pytest.fixture
def blend_input(all_fuel_prices, all_fuel_properties):
    """Create blend optimization input."""
    return BlendInput(
        available_fuels=["natural_gas", "lpg_propane"],
        fuel_properties={
            "natural_gas": all_fuel_properties["natural_gas"],
            "lpg_propane": all_fuel_properties["lpg_propane"],
        },
        fuel_prices={
            "natural_gas": all_fuel_prices["natural_gas"],
            "lpg_propane": all_fuel_prices["lpg_propane"],
        },
        current_blend=None,
        required_heat_input_mmbtu_hr=50.0,
    )


@pytest.fixture
def switching_input(all_fuel_prices):
    """Create switching evaluation input."""
    return SwitchingInput(
        current_fuel="natural_gas",
        current_cost_usd_mmbtu=3.50,
        current_heat_input_mmbtu_hr=50.0,
        available_fuels=["natural_gas", "no2_fuel_oil", "lpg_propane"],
        fuel_prices=all_fuel_prices,
        equipment_id="BOILER-001",
        equipment_online=True,
        current_load_pct=80.0,
        time_on_current_fuel_hours=8.0,
        switches_today=1,
        safety_interlocks={
            "flame_proven": True,
            "fuel_pressure_ok": True,
            "no_active_alarms": True,
        },
    )


@pytest.fixture
def total_cost_input(all_fuel_prices, all_fuel_properties):
    """Create total cost optimization input."""
    return TotalCostInput(
        fuel_options=["natural_gas", "no2_fuel_oil", "lpg_propane"],
        fuel_prices=all_fuel_prices,
        fuel_properties=all_fuel_properties,
        heat_demand_mmbtu_hr=50.0,
        operating_hours_year=8000.0,
        current_fuel="natural_gas",
        equipment_efficiency={
            "natural_gas": 82.0,
            "no2_fuel_oil": 80.0,
            "lpg_propane": 81.0,
        },
        base_maintenance_cost_usd_year=50000.0,
        carbon_price_usd_ton=50.0,
    )


@pytest.fixture
def fuel_optimization_input(all_fuel_prices, all_fuel_properties):
    """Create fuel optimization input."""
    return FuelOptimizationInput(
        facility_id="TEST-FACILITY-001",
        current_fuel="natural_gas",
        current_fuel_flow_rate=50000.0,
        current_heat_input_mmbtu_hr=50.0,
        current_load_pct=80.0,
        current_blend=None,
        fuel_prices=all_fuel_prices,
        fuel_properties=all_fuel_properties,
        forecast_horizon_hours=24,
    )


# =============================================================================
# TANK FIXTURES
# =============================================================================

@pytest.fixture
def tank_config():
    """Create tank configuration."""
    return TankConfig(
        tank_id="TANK-001",
        name="Main Fuel Tank",
        fuel_type="natural_gas",
        capacity_gal=10000,
        usable_capacity_pct=95.0,
        heel_volume_gal=100,
        reorder_point_pct=30.0,
        low_level_pct=25.0,
        critical_level_pct=15.0,
        high_level_pct=95.0,
    )


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_pricing_api():
    """Create mock pricing API responses."""
    return {
        "natural_gas": {
            "price": 3.00,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "no2_fuel_oil": {
            "price": 13.50,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


@pytest.fixture
def mock_weather_data():
    """Create mock weather data for demand forecasting."""
    return {
        "temperature_f": 45.0,
        "heating_degree_days": 20.0,
        "forecast": [
            {"date": "2025-01-13", "high_f": 50, "low_f": 35},
            {"date": "2025-01-14", "high_f": 48, "low_f": 33},
        ],
    }


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def sample_request_id():
    """Generate sample request ID."""
    return str(uuid.uuid4())


@pytest.fixture
def current_timestamp():
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


# =============================================================================
# PARAMETRIZE DATA
# =============================================================================

# Fuel type test data
FUEL_TYPES = [
    ("natural_gas", 53.06),
    ("no2_fuel_oil", 73.16),
    ("no6_fuel_oil", 75.10),
    ("lpg_propane", 62.87),
    ("coal_bituminous", 93.28),
    ("biomass_wood", 0.0),
    ("biogas", 0.0),
    ("hydrogen", 0.0),
]

# Gas composition test data
GAS_COMPOSITIONS = [
    # (methane%, ethane%, propane%, nitrogen%, co2%, expected_hhv_btu_scf)
    (95.0, 2.5, 0.5, 1.5, 0.5, 1010.0),  # Pipeline quality
    (98.0, 1.0, 0.2, 0.5, 0.3, 1005.0),  # High methane
    (90.0, 5.0, 2.0, 2.0, 1.0, 1020.0),  # Rich gas
    (85.0, 7.0, 4.0, 2.5, 1.5, 1040.0),  # Very rich gas
]

# Price scenario test data
PRICE_SCENARIOS = [
    # (ng_price, oil_price, propane_price, expected_optimal)
    (3.00, 15.00, 8.00, "natural_gas"),
    (6.00, 12.00, 7.00, "lpg_propane"),
    (8.00, 10.00, 12.00, "no2_fuel_oil"),
]
