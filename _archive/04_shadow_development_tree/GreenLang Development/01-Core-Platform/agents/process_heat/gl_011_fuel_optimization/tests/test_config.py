"""
GL-011 FUELCRAFT - Configuration Tests

Unit tests for all configuration classes in config.py.
Validates default values, constraints, and validation logic.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

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


class TestFuelType:
    """Tests for FuelType enum."""

    def test_all_fuel_types_defined(self):
        """Test all expected fuel types are defined."""
        expected = {
            "NATURAL_GAS", "NO2_FUEL_OIL", "NO6_FUEL_OIL",
            "LPG_PROPANE", "LPG_BUTANE", "COAL_BITUMINOUS",
            "COAL_SUB_BITUMINOUS", "COAL_ANTHRACITE",
            "BIOMASS_WOOD", "BIOMASS_PELLETS", "BIOGAS",
            "HYDROGEN", "RNG", "DUAL_FUEL",
        }
        actual = {f.name for f in FuelType}
        assert expected == actual

    def test_fuel_type_values(self):
        """Test fuel type enum values."""
        assert FuelType.NATURAL_GAS.value == "natural_gas"
        assert FuelType.NO2_FUEL_OIL.value == "no2_fuel_oil"
        assert FuelType.HYDROGEN.value == "hydrogen"


class TestPriceSource:
    """Tests for PriceSource enum."""

    def test_all_price_sources_defined(self):
        """Test all expected price sources are defined."""
        expected = {
            "HENRY_HUB", "BRENT", "WTI", "REGIONAL_HUB",
            "API2", "API4", "SPOT", "CONTRACT", "MANUAL",
        }
        actual = {s.name for s in PriceSource}
        assert expected == actual


class TestOptimizationMode:
    """Tests for OptimizationMode enum."""

    def test_optimization_modes(self):
        """Test optimization mode values."""
        assert OptimizationMode.MINIMUM_COST.value == "minimum_cost"
        assert OptimizationMode.MINIMUM_EMISSIONS.value == "minimum_emissions"
        assert OptimizationMode.BALANCED.value == "balanced"
        assert OptimizationMode.RELIABILITY.value == "reliability"


class TestFuelPricingConfig:
    """Tests for FuelPricingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FuelPricingConfig()

        assert config.primary_source == PriceSource.HENRY_HUB
        assert config.secondary_source is None
        assert config.update_interval_minutes == 15
        assert config.price_history_days == 365
        assert config.forecast_horizon_days == 30
        assert config.basis_differential_enabled is True
        assert config.transport_cost_enabled is True
        assert config.taxes_included is True
        assert config.currency == "USD"

    def test_update_interval_constraints(self):
        """Test update interval validation."""
        # Valid values
        config = FuelPricingConfig(update_interval_minutes=1)
        assert config.update_interval_minutes == 1

        config = FuelPricingConfig(update_interval_minutes=1440)
        assert config.update_interval_minutes == 1440

        # Invalid values
        with pytest.raises(PydanticValidationError):
            FuelPricingConfig(update_interval_minutes=0)

        with pytest.raises(PydanticValidationError):
            FuelPricingConfig(update_interval_minutes=1441)

    def test_price_history_constraints(self):
        """Test price history days validation."""
        config = FuelPricingConfig(price_history_days=30)
        assert config.price_history_days == 30

        with pytest.raises(PydanticValidationError):
            FuelPricingConfig(price_history_days=29)

        with pytest.raises(PydanticValidationError):
            FuelPricingConfig(price_history_days=3651)

    def test_custom_api_endpoint(self):
        """Test custom API endpoint configuration."""
        config = FuelPricingConfig(
            api_endpoint="https://api.example.com/prices",
            api_key_env_var="MY_API_KEY",
        )
        assert config.api_endpoint == "https://api.example.com/prices"
        assert config.api_key_env_var == "MY_API_KEY"


class TestBlendingConfig:
    """Tests for BlendingConfig."""

    def test_default_values(self):
        """Test default blending configuration."""
        config = BlendingConfig()

        assert config.enabled is True
        assert config.max_fuels_in_blend == 3
        assert config.min_wobbe_index == 1300.0
        assert config.max_wobbe_index == 1400.0
        assert config.wobbe_tolerance_pct == 5.0
        assert config.min_primary_fuel_pct == 50.0
        assert config.max_blend_change_rate_pct_min == 5.0

    def test_wobbe_index_validation(self):
        """Test Wobbe Index range validation."""
        # Valid range
        config = BlendingConfig(min_wobbe_index=1250.0, max_wobbe_index=1350.0)
        assert config.min_wobbe_index == 1250.0
        assert config.max_wobbe_index == 1350.0

        # Invalid: max less than min
        with pytest.raises(PydanticValidationError):
            BlendingConfig(min_wobbe_index=1400.0, max_wobbe_index=1300.0)

    def test_max_fuels_constraints(self):
        """Test max fuels in blend constraints."""
        config = BlendingConfig(max_fuels_in_blend=2)
        assert config.max_fuels_in_blend == 2

        config = BlendingConfig(max_fuels_in_blend=5)
        assert config.max_fuels_in_blend == 5

        with pytest.raises(PydanticValidationError):
            BlendingConfig(max_fuels_in_blend=1)

        with pytest.raises(PydanticValidationError):
            BlendingConfig(max_fuels_in_blend=6)

    def test_blend_change_rate_constraints(self):
        """Test blend change rate constraints."""
        config = BlendingConfig(max_blend_change_rate_pct_min=1.0)
        assert config.max_blend_change_rate_pct_min == 1.0

        with pytest.raises(PydanticValidationError):
            BlendingConfig(max_blend_change_rate_pct_min=0.5)


class TestSwitchingConfig:
    """Tests for SwitchingConfig."""

    def test_default_values(self):
        """Test default switching configuration."""
        config = SwitchingConfig()

        assert config.mode == SwitchingMode.SEMI_AUTOMATIC
        assert config.price_differential_trigger_pct == 15.0
        assert config.min_savings_usd_hr == 100.0
        assert config.payback_period_hours == 24.0
        assert config.min_run_time_hours == 4.0
        assert config.max_switches_per_day == 4
        assert config.safety_interlock_enabled is True
        assert config.require_purge is True

    def test_switching_modes(self):
        """Test all switching modes."""
        for mode in SwitchingMode:
            config = SwitchingConfig(mode=mode)
            assert config.mode == mode

    def test_price_differential_constraints(self):
        """Test price differential trigger constraints."""
        config = SwitchingConfig(price_differential_trigger_pct=5.0)
        assert config.price_differential_trigger_pct == 5.0

        config = SwitchingConfig(price_differential_trigger_pct=50.0)
        assert config.price_differential_trigger_pct == 50.0

        with pytest.raises(PydanticValidationError):
            SwitchingConfig(price_differential_trigger_pct=4.0)

    def test_max_switches_constraints(self):
        """Test max switches per day constraints."""
        config = SwitchingConfig(max_switches_per_day=1)
        assert config.max_switches_per_day == 1

        with pytest.raises(PydanticValidationError):
            SwitchingConfig(max_switches_per_day=0)

        with pytest.raises(PydanticValidationError):
            SwitchingConfig(max_switches_per_day=13)


class TestInventoryConfig:
    """Tests for InventoryConfig."""

    def test_default_values(self):
        """Test default inventory configuration."""
        config = InventoryConfig()

        assert config.enabled is True
        assert config.reorder_point_pct == 30.0
        assert config.safety_stock_days == 3.0
        assert config.economic_order_quantity_enabled is True
        assert config.low_level_alert_pct == 25.0
        assert config.critical_level_pct == 15.0
        assert config.high_level_alert_pct == 95.0
        assert config.lead_time_days == 2.0

    def test_critical_level_validation(self):
        """Test critical level must be less than low level."""
        # Valid configuration
        config = InventoryConfig(low_level_alert_pct=30.0, critical_level_pct=15.0)
        assert config.critical_level_pct == 15.0

        # Invalid: critical >= low
        with pytest.raises(PydanticValidationError):
            InventoryConfig(low_level_alert_pct=20.0, critical_level_pct=20.0)

        with pytest.raises(PydanticValidationError):
            InventoryConfig(low_level_alert_pct=15.0, critical_level_pct=20.0)

    def test_preferred_delivery_days(self):
        """Test preferred delivery days configuration."""
        config = InventoryConfig(preferred_delivery_days=[1, 2, 3, 4, 5])
        assert config.preferred_delivery_days == [1, 2, 3, 4, 5]

        config = InventoryConfig(preferred_delivery_days=[1, 3, 5])
        assert config.preferred_delivery_days == [1, 3, 5]

    def test_tank_configuration(self):
        """Test tank configuration within inventory config."""
        tanks = {
            "TANK-001": {
                "fuel_type": "natural_gas",
                "capacity_gal": 10000,
            },
            "TANK-002": {
                "fuel_type": "fuel_oil",
                "capacity_gal": 5000,
            },
        }
        config = InventoryConfig(tanks=tanks)
        assert len(config.tanks) == 2
        assert "TANK-001" in config.tanks


class TestCostOptimizationConfig:
    """Tests for CostOptimizationConfig."""

    def test_default_values(self):
        """Test default cost optimization configuration."""
        config = CostOptimizationConfig()

        assert config.enabled is True
        assert config.mode == OptimizationMode.MINIMUM_COST
        assert config.include_fuel_cost is True
        assert config.include_emissions_cost is True
        assert config.carbon_price_usd_ton == 50.0
        assert config.cost_weight == 0.6
        assert config.emissions_weight == 0.3
        assert config.reliability_weight == 0.1

    def test_weight_validation(self):
        """Test that weights must sum to 1.0."""
        # Valid weights
        config = CostOptimizationConfig(
            cost_weight=0.5,
            emissions_weight=0.3,
            reliability_weight=0.2,
        )
        assert config.cost_weight + config.emissions_weight + config.reliability_weight == pytest.approx(1.0)

        # Invalid weights (don't sum to 1.0)
        with pytest.raises(PydanticValidationError):
            CostOptimizationConfig(
                cost_weight=0.5,
                emissions_weight=0.3,
                reliability_weight=0.3,
            )

    def test_carbon_price_constraints(self):
        """Test carbon price constraints."""
        config = CostOptimizationConfig(carbon_price_usd_ton=0.0)
        assert config.carbon_price_usd_ton == 0.0

        config = CostOptimizationConfig(carbon_price_usd_ton=500.0)
        assert config.carbon_price_usd_ton == 500.0

        with pytest.raises(PydanticValidationError):
            CostOptimizationConfig(carbon_price_usd_ton=-1.0)

        with pytest.raises(PydanticValidationError):
            CostOptimizationConfig(carbon_price_usd_ton=501.0)

    def test_optimization_modes(self):
        """Test all optimization modes are configurable."""
        for mode in OptimizationMode:
            config = CostOptimizationConfig(mode=mode)
            assert config.mode == mode


class TestEquipmentConfig:
    """Tests for EquipmentConfig."""

    def test_required_equipment_id(self):
        """Test equipment_id is required."""
        config = EquipmentConfig(equipment_id="BOILER-001")
        assert config.equipment_id == "BOILER-001"

        with pytest.raises(PydanticValidationError):
            EquipmentConfig()

    def test_default_values(self):
        """Test default equipment configuration."""
        config = EquipmentConfig(equipment_id="BOILER-001")

        assert config.supported_fuels == [FuelType.NATURAL_GAS]
        assert config.primary_fuel == FuelType.NATURAL_GAS
        assert config.dual_fuel_capable is False
        assert config.design_capacity_mmbtu_hr == 50.0
        assert config.min_load_pct == 25.0
        assert config.max_load_pct == 110.0
        assert config.design_efficiency_pct == 82.0

    def test_supported_fuels(self):
        """Test supported fuels configuration."""
        config = EquipmentConfig(
            equipment_id="BOILER-001",
            supported_fuels=[FuelType.NATURAL_GAS, FuelType.NO2_FUEL_OIL],
            primary_fuel=FuelType.NATURAL_GAS,
            dual_fuel_capable=True,
        )
        assert len(config.supported_fuels) == 2
        assert FuelType.NATURAL_GAS in config.supported_fuels
        assert config.dual_fuel_capable is True

    def test_capacity_constraints(self):
        """Test capacity constraints."""
        config = EquipmentConfig(
            equipment_id="BOILER-001",
            design_capacity_mmbtu_hr=100.0,
        )
        assert config.design_capacity_mmbtu_hr == 100.0

        with pytest.raises(PydanticValidationError):
            EquipmentConfig(
                equipment_id="BOILER-001",
                design_capacity_mmbtu_hr=0,
            )


class TestFuelOptimizationConfig:
    """Tests for FuelOptimizationConfig (main config)."""

    def test_required_facility_id(self):
        """Test facility_id is required."""
        config = FuelOptimizationConfig(facility_id="PLANT-001")
        assert config.facility_id == "PLANT-001"

        with pytest.raises(PydanticValidationError):
            FuelOptimizationConfig()

    def test_default_name_from_facility_id(self):
        """Test default name is set from facility_id."""
        config = FuelOptimizationConfig(facility_id="PLANT-001")
        assert config.name == "Facility PLANT-001"

        config = FuelOptimizationConfig(facility_id="PLANT-002", name="My Facility")
        assert config.name == "My Facility"

    def test_available_fuels_includes_primary(self):
        """Test primary fuel is added to available fuels."""
        config = FuelOptimizationConfig(
            facility_id="PLANT-001",
            primary_fuel=FuelType.NATURAL_GAS,
            available_fuels=[FuelType.NO2_FUEL_OIL],
        )
        assert FuelType.NATURAL_GAS in config.available_fuels
        assert FuelType.NO2_FUEL_OIL in config.available_fuels

    def test_sub_configurations(self):
        """Test sub-configurations are created with defaults."""
        config = FuelOptimizationConfig(facility_id="PLANT-001")

        assert config.pricing is not None
        assert isinstance(config.pricing, FuelPricingConfig)

        assert config.blending is not None
        assert isinstance(config.blending, BlendingConfig)

        assert config.switching is not None
        assert isinstance(config.switching, SwitchingConfig)

        assert config.inventory is not None
        assert isinstance(config.inventory, InventoryConfig)

        assert config.cost_optimization is not None
        assert isinstance(config.cost_optimization, CostOptimizationConfig)

    def test_safety_level_constraints(self):
        """Test safety level constraints (1-3)."""
        config = FuelOptimizationConfig(facility_id="PLANT-001", safety_level=1)
        assert config.safety_level == 1

        config = FuelOptimizationConfig(facility_id="PLANT-001", safety_level=3)
        assert config.safety_level == 3

        with pytest.raises(PydanticValidationError):
            FuelOptimizationConfig(facility_id="PLANT-001", safety_level=0)

        with pytest.raises(PydanticValidationError):
            FuelOptimizationConfig(facility_id="PLANT-001", safety_level=4)

    def test_complete_configuration(self, fuel_optimization_config):
        """Test complete configuration fixture."""
        assert fuel_optimization_config.facility_id == "TEST-FACILITY-001"
        assert fuel_optimization_config.primary_fuel == FuelType.NATURAL_GAS
        assert len(fuel_optimization_config.available_fuels) >= 3
        assert len(fuel_optimization_config.equipment) == 1


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_levels(self):
        """Test alert level values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"
