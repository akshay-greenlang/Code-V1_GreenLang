"""
GL-011 FUELCRAFT - Configuration Module

Configuration schemas for fuel optimization including pricing sources,
blending constraints, switching triggers, and inventory management.

This module defines all configuration parameters for the FuelOptimizationAgent
with comprehensive validation and sensible defaults for industrial applications.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field, validator


class FuelType(Enum):
    """Supported fuel types for optimization."""
    NATURAL_GAS = "natural_gas"
    NO2_FUEL_OIL = "no2_fuel_oil"
    NO6_FUEL_OIL = "no6_fuel_oil"
    LPG_PROPANE = "lpg_propane"
    LPG_BUTANE = "lpg_butane"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    COAL_ANTHRACITE = "coal_anthracite"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLETS = "biomass_pellets"
    BIOGAS = "biogas"
    HYDROGEN = "hydrogen"
    RNG = "rng"  # Renewable Natural Gas
    DUAL_FUEL = "dual_fuel"


class PriceSource(Enum):
    """Fuel price data sources."""
    HENRY_HUB = "henry_hub"  # Natural gas benchmark
    BRENT = "brent"  # Crude oil benchmark
    WTI = "wti"  # West Texas Intermediate
    REGIONAL_HUB = "regional_hub"  # Regional gas trading hubs
    API2 = "api2"  # Coal benchmark (Rotterdam)
    API4 = "api4"  # Coal benchmark (South Africa)
    SPOT = "spot"  # Spot market
    CONTRACT = "contract"  # Fixed contract price
    MANUAL = "manual"  # Manual entry


class OptimizationMode(Enum):
    """Optimization objective modes."""
    MINIMUM_COST = "minimum_cost"
    MINIMUM_EMISSIONS = "minimum_emissions"
    BALANCED = "balanced"
    RELIABILITY = "reliability"
    CUSTOM = "custom"


class SwitchingMode(Enum):
    """Fuel switching modes."""
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"  # Requires operator confirmation
    MANUAL = "manual"
    DISABLED = "disabled"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class FuelPricingConfig(BaseModel):
    """Fuel pricing configuration."""

    primary_source: PriceSource = Field(
        default=PriceSource.HENRY_HUB,
        description="Primary price data source"
    )
    secondary_source: Optional[PriceSource] = Field(
        default=None,
        description="Secondary/backup price source"
    )
    update_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=1440,
        description="Price update interval in minutes"
    )
    price_history_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Days of price history to maintain"
    )
    forecast_horizon_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Price forecast horizon in days"
    )

    # API endpoints (configurable)
    api_endpoint: Optional[str] = Field(
        default=None,
        description="Custom API endpoint for price data"
    )
    api_key_env_var: str = Field(
        default="FUEL_PRICE_API_KEY",
        description="Environment variable name for API key"
    )

    # Price adjustments
    basis_differential_enabled: bool = Field(
        default=True,
        description="Enable regional basis differential adjustments"
    )
    transport_cost_enabled: bool = Field(
        default=True,
        description="Include transportation costs"
    )
    taxes_included: bool = Field(
        default=True,
        description="Include applicable taxes"
    )

    # Currency
    currency: str = Field(
        default="USD",
        description="Price currency"
    )

    class Config:
        use_enum_values = True


class BlendingConfig(BaseModel):
    """Fuel blending configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable fuel blending optimization"
    )
    max_fuels_in_blend: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Maximum number of fuels in a blend"
    )

    # Wobbe Index constraints
    min_wobbe_index: float = Field(
        default=1300.0,
        ge=1000.0,
        le=1500.0,
        description="Minimum Wobbe Index (BTU/SCF)"
    )
    max_wobbe_index: float = Field(
        default=1400.0,
        ge=1100.0,
        le=1600.0,
        description="Maximum Wobbe Index (BTU/SCF)"
    )
    wobbe_tolerance_pct: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Wobbe Index tolerance percentage"
    )

    # Heating value constraints
    min_hhv_btu_scf: Optional[float] = Field(
        default=900.0,
        description="Minimum HHV (BTU/SCF)"
    )
    max_hhv_btu_scf: Optional[float] = Field(
        default=1200.0,
        description="Maximum HHV (BTU/SCF)"
    )

    # Emission constraints
    max_co2_kg_mmbtu: Optional[float] = Field(
        default=None,
        description="Maximum CO2 emissions (kg/MMBTU)"
    )
    max_nox_lb_mmbtu: Optional[float] = Field(
        default=None,
        description="Maximum NOx emissions (lb/MMBTU)"
    )

    # Blend ratio constraints
    min_primary_fuel_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum primary fuel percentage"
    )
    max_blend_change_rate_pct_min: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Maximum blend ratio change rate (%/min)"
    )

    # Optimization
    optimization_interval_minutes: int = Field(
        default=60,
        ge=15,
        le=1440,
        description="Blend optimization interval"
    )

    @validator("max_wobbe_index")
    def validate_wobbe_range(cls, v, values):
        """Ensure max Wobbe is greater than min."""
        if "min_wobbe_index" in values and v <= values["min_wobbe_index"]:
            raise ValueError("max_wobbe_index must be greater than min_wobbe_index")
        return v

    class Config:
        use_enum_values = True


class SwitchingConfig(BaseModel):
    """Fuel switching configuration."""

    mode: SwitchingMode = Field(
        default=SwitchingMode.SEMI_AUTOMATIC,
        description="Fuel switching mode"
    )

    # Economic triggers
    price_differential_trigger_pct: float = Field(
        default=15.0,
        ge=5.0,
        le=50.0,
        description="Price differential to trigger switching (%)"
    )
    min_savings_usd_hr: float = Field(
        default=100.0,
        ge=0.0,
        description="Minimum hourly savings to trigger switch"
    )
    payback_period_hours: float = Field(
        default=24.0,
        ge=1.0,
        le=168.0,
        description="Required payback period for switch"
    )

    # Timing constraints
    min_run_time_hours: float = Field(
        default=4.0,
        ge=1.0,
        le=24.0,
        description="Minimum run time on a fuel before switching"
    )
    max_switches_per_day: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Maximum fuel switches per day"
    )
    switch_lockout_minutes: int = Field(
        default=30,
        ge=15,
        le=240,
        description="Lockout time after a switch"
    )

    # Transition settings
    transition_duration_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Expected transition duration"
    )
    ramp_rate_pct_min: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Fuel ramp rate during transition (%/min)"
    )

    # Safety interlocks
    safety_interlock_enabled: bool = Field(
        default=True,
        description="Enable safety interlocks during switching"
    )
    require_purge: bool = Field(
        default=True,
        description="Require purge cycle during switch"
    )
    purge_duration_seconds: int = Field(
        default=60,
        ge=30,
        le=300,
        description="Purge duration in seconds"
    )

    # Operator interaction
    operator_confirmation_required: bool = Field(
        default=True,
        description="Require operator confirmation for switches"
    )
    confirmation_timeout_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Timeout for operator confirmation"
    )

    class Config:
        use_enum_values = True


class InventoryConfig(BaseModel):
    """Fuel inventory management configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable inventory management"
    )

    # Tank configuration
    tanks: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Tank configurations by ID"
    )

    # Reorder settings
    reorder_point_pct: float = Field(
        default=30.0,
        ge=10.0,
        le=50.0,
        description="Reorder point as percentage of capacity"
    )
    safety_stock_days: float = Field(
        default=3.0,
        ge=1.0,
        le=14.0,
        description="Safety stock in days of consumption"
    )
    economic_order_quantity_enabled: bool = Field(
        default=True,
        description="Enable EOQ calculation"
    )

    # Alert thresholds
    low_level_alert_pct: float = Field(
        default=25.0,
        ge=10.0,
        le=40.0,
        description="Low level alert threshold (%)"
    )
    critical_level_pct: float = Field(
        default=15.0,
        ge=5.0,
        le=25.0,
        description="Critical level threshold (%)"
    )
    high_level_alert_pct: float = Field(
        default=95.0,
        ge=85.0,
        le=99.0,
        description="High level alert threshold (%)"
    )

    # Delivery scheduling
    lead_time_days: float = Field(
        default=2.0,
        ge=0.5,
        le=14.0,
        description="Standard delivery lead time"
    )
    delivery_window_hours: int = Field(
        default=4,
        ge=2,
        le=12,
        description="Delivery window duration"
    )
    preferred_delivery_days: List[int] = Field(
        default=[1, 2, 3, 4, 5],  # Monday-Friday
        description="Preferred delivery days (1=Mon, 7=Sun)"
    )

    # Consumption forecasting
    forecast_horizon_days: int = Field(
        default=14,
        ge=7,
        le=90,
        description="Consumption forecast horizon"
    )
    use_weather_forecast: bool = Field(
        default=True,
        description="Use weather data for demand prediction"
    )

    @validator("critical_level_pct")
    def validate_critical_level(cls, v, values):
        """Ensure critical level is less than low level."""
        if "low_level_alert_pct" in values and v >= values["low_level_alert_pct"]:
            raise ValueError("critical_level_pct must be less than low_level_alert_pct")
        return v

    class Config:
        use_enum_values = True


class CostOptimizationConfig(BaseModel):
    """Total cost of ownership optimization configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable cost optimization"
    )
    mode: OptimizationMode = Field(
        default=OptimizationMode.MINIMUM_COST,
        description="Optimization objective"
    )

    # Cost components
    include_fuel_cost: bool = Field(default=True, description="Include fuel purchase cost")
    include_transport_cost: bool = Field(default=True, description="Include transport cost")
    include_storage_cost: bool = Field(default=True, description="Include storage cost")
    include_emissions_cost: bool = Field(default=True, description="Include carbon cost")
    include_maintenance_cost: bool = Field(default=True, description="Include maintenance impact")
    include_efficiency_impact: bool = Field(default=True, description="Account for efficiency differences")

    # Carbon pricing
    carbon_price_usd_ton: float = Field(
        default=50.0,
        ge=0.0,
        le=500.0,
        description="Carbon price ($/ton CO2)"
    )
    carbon_price_escalation_pct: float = Field(
        default=5.0,
        ge=0.0,
        le=20.0,
        description="Annual carbon price escalation (%)"
    )

    # Weighting factors for multi-objective optimization
    cost_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Cost weight in objective function"
    )
    emissions_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Emissions weight in objective function"
    )
    reliability_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Reliability weight in objective function"
    )

    # Optimization horizon
    optimization_horizon_hours: int = Field(
        default=168,  # 1 week
        ge=24,
        le=8760,
        description="Optimization horizon in hours"
    )
    rolling_window_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Rolling optimization window"
    )

    @validator("emissions_weight")
    def validate_weights(cls, v, values):
        """Ensure weights sum to approximately 1.0."""
        if "cost_weight" in values and "reliability_weight" in values:
            total = values["cost_weight"] + v + values["reliability_weight"]
            if abs(total - 1.0) > 0.01:
                raise ValueError("cost_weight + emissions_weight + reliability_weight must equal 1.0")
        return v

    class Config:
        use_enum_values = True


class EquipmentConfig(BaseModel):
    """Equipment capability configuration."""

    equipment_id: str = Field(..., description="Equipment identifier")
    name: str = Field(default="", description="Equipment name")

    # Fuel capabilities
    supported_fuels: List[FuelType] = Field(
        default=[FuelType.NATURAL_GAS],
        description="Fuels this equipment can use"
    )
    primary_fuel: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary/design fuel"
    )
    dual_fuel_capable: bool = Field(
        default=False,
        description="Can fire two fuels simultaneously"
    )

    # Capacity
    design_capacity_mmbtu_hr: float = Field(
        default=50.0,
        gt=0,
        description="Design heat input capacity (MMBTU/hr)"
    )
    min_load_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Minimum load percentage"
    )
    max_load_pct: float = Field(
        default=110.0,
        ge=100,
        le=120,
        description="Maximum load percentage"
    )

    # Efficiency
    design_efficiency_pct: float = Field(
        default=82.0,
        ge=50,
        le=100,
        description="Design efficiency on primary fuel (%)"
    )
    efficiency_by_fuel: Dict[str, float] = Field(
        default_factory=dict,
        description="Efficiency adjustment by fuel type"
    )

    # Switching constraints
    fuel_switch_time_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Time to complete fuel switch"
    )
    requires_shutdown_for_switch: bool = Field(
        default=False,
        description="Requires shutdown to switch fuels"
    )

    class Config:
        use_enum_values = True


class FuelOptimizationConfig(BaseModel):
    """
    Complete GL-011 FUELCRAFT agent configuration.

    This configuration defines all parameters for the Fuel Optimization Agent
    including pricing, blending, switching, inventory, and cost optimization.
    """

    # Identity
    facility_id: str = Field(..., description="Facility identifier")
    name: str = Field(default="", description="Facility name")

    # Primary settings
    primary_fuel: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )
    available_fuels: List[FuelType] = Field(
        default=[FuelType.NATURAL_GAS],
        description="All available fuel types"
    )

    # Sub-configurations
    pricing: FuelPricingConfig = Field(
        default_factory=FuelPricingConfig,
        description="Pricing configuration"
    )
    blending: BlendingConfig = Field(
        default_factory=BlendingConfig,
        description="Blending configuration"
    )
    switching: SwitchingConfig = Field(
        default_factory=SwitchingConfig,
        description="Switching configuration"
    )
    inventory: InventoryConfig = Field(
        default_factory=InventoryConfig,
        description="Inventory configuration"
    )
    cost_optimization: CostOptimizationConfig = Field(
        default_factory=CostOptimizationConfig,
        description="Cost optimization configuration"
    )

    # Equipment
    equipment: List[EquipmentConfig] = Field(
        default_factory=list,
        description="Equipment configurations"
    )

    # Agent settings
    agent_id: str = Field(
        default="GL-011",
        description="Agent identifier"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    # Safety
    safety_level: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Safety Integrity Level"
    )
    emergency_fuel: Optional[FuelType] = Field(
        default=None,
        description="Emergency backup fuel"
    )

    # Logging and metrics
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable detailed audit logging"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )

    # External integrations
    weather_api_enabled: bool = Field(
        default=True,
        description="Enable weather API for demand prediction"
    )
    weather_api_key_env_var: str = Field(
        default="WEATHER_API_KEY",
        description="Environment variable for weather API key"
    )
    cmms_integration_enabled: bool = Field(
        default=False,
        description="Enable CMMS integration for equipment constraints"
    )

    class Config:
        use_enum_values = True

    @validator("name", always=True)
    def set_default_name(cls, v, values):
        """Set default name from facility_id."""
        if not v and "facility_id" in values:
            return f"Facility {values['facility_id']}"
        return v

    @validator("available_fuels")
    def validate_available_fuels(cls, v, values):
        """Ensure primary fuel is in available fuels."""
        if "primary_fuel" in values:
            primary = values["primary_fuel"]
            if primary not in v:
                v.append(primary)
        return v
