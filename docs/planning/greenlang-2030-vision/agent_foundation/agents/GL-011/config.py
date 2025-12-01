# -*- coding: utf-8 -*-
"""
Configuration module for FuelManagementOrchestrator agent (GL-011 FUELCRAFT).

This module defines the configuration models and settings for the
FuelManagementOrchestrator agent, including fuel specifications,
inventory management, market pricing, blending constraints, emission limits,
and optimization parameters.

Standards Compliance:
- ISO 6976:2016 - Natural gas calorific value
- ISO 17225 - Solid biofuels specifications
- ASTM D4809 - Heat of combustion
- GHG Protocol - Emissions calculations
- Pydantic V2 for validation

Author: GreenLang Industrial Optimization Team
Date: December 2025
Agent ID: GL-011
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from pathlib import Path
from enum import Enum

try:
    from greenlang.determinism import DeterministicClock
except ImportError:
    # Fallback for standalone testing
    class DeterministicClock:
        @staticmethod
        def now():
            return datetime.utcnow()


class FuelCategory(str, Enum):
    """Fuel category classification."""
    FOSSIL = "fossil"
    RENEWABLE = "renewable"
    NUCLEAR = "nuclear"
    HYBRID = "hybrid"


class FuelState(str, Enum):
    """Physical state of fuel."""
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"


class EmissionStandard(str, Enum):
    """Emission regulatory standards."""
    EPA_NSPS = "epa_nsps"
    EU_IED = "eu_ied"
    EU_MCP = "eu_mcp"
    CHINA_GB = "china_gb"
    INDIA_CPCB = "india_cpcb"
    CUSTOM = "custom"


class FuelSpecification(BaseModel):
    """
    Technical specifications for a fuel type.

    This model captures all physical and chemical properties needed
    for fuel optimization calculations per ISO standards.
    """

    fuel_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique fuel identifier"
    )
    fuel_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable fuel name"
    )
    fuel_type: str = Field(
        ...,
        description="Fuel type (coal, natural_gas, biomass, etc.)"
    )
    category: FuelCategory = Field(
        ...,
        description="Fuel category (fossil, renewable, etc.)"
    )
    state: FuelState = Field(
        ...,
        description="Physical state (solid, liquid, gas)"
    )

    # Calorific values (ISO 6976, ASTM D4809)
    gross_calorific_value_mj_kg: float = Field(
        ...,
        ge=0,
        le=150,
        description="Gross (Higher) Heating Value in MJ/kg"
    )
    net_calorific_value_mj_kg: float = Field(
        ...,
        ge=0,
        le=140,
        description="Net (Lower) Heating Value in MJ/kg"
    )
    calorific_value_unit: str = Field(
        default="MJ/kg",
        description="Unit for calorific value"
    )

    # Density
    density_kg_m3: float = Field(
        ...,
        gt=0,
        le=5000,
        description="Density in kg/m3"
    )
    bulk_density_kg_m3: Optional[float] = Field(
        None,
        gt=0,
        le=2000,
        description="Bulk density for solid fuels"
    )

    # Composition (ultimate analysis)
    carbon_content_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Carbon content percentage"
    )
    hydrogen_content_percent: float = Field(
        ...,
        ge=0,
        le=25,
        description="Hydrogen content percentage"
    )
    oxygen_content_percent: float = Field(
        default=0,
        ge=0,
        le=60,
        description="Oxygen content percentage"
    )
    nitrogen_content_percent: float = Field(
        default=0,
        ge=0,
        le=5,
        description="Nitrogen content percentage"
    )
    sulfur_content_percent: float = Field(
        default=0,
        ge=0,
        le=10,
        description="Sulfur content percentage"
    )

    # Proximate analysis (for solid fuels)
    moisture_content_percent: float = Field(
        default=0,
        ge=0,
        le=80,
        description="Moisture content percentage"
    )
    ash_content_percent: float = Field(
        default=0,
        ge=0,
        le=50,
        description="Ash content percentage"
    )
    volatile_matter_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Volatile matter percentage"
    )
    fixed_carbon_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Fixed carbon percentage"
    )

    # Emission factors (GHG Protocol)
    emission_factor_co2_kg_gj: float = Field(
        ...,
        ge=0,
        le=200,
        description="CO2 emission factor in kg/GJ"
    )
    emission_factor_ch4_g_gj: float = Field(
        default=0,
        ge=0,
        le=100,
        description="CH4 emission factor in g/GJ"
    )
    emission_factor_n2o_g_gj: float = Field(
        default=0,
        ge=0,
        le=10,
        description="N2O emission factor in g/GJ"
    )
    emission_factor_nox_g_gj: float = Field(
        default=0,
        ge=0,
        le=1000,
        description="NOx emission factor in g/GJ"
    )
    emission_factor_sox_g_gj: float = Field(
        default=0,
        ge=0,
        le=2000,
        description="SOx emission factor in g/GJ"
    )
    emission_factor_pm_g_gj: float = Field(
        default=0,
        ge=0,
        le=500,
        description="Particulate matter emission factor in g/GJ"
    )

    # Renewable classification
    is_renewable: bool = Field(
        default=False,
        description="Whether fuel is renewable"
    )
    biogenic_carbon_percent: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Biogenic carbon percentage (carbon neutral)"
    )

    # Storage requirements
    storage_temperature_min_c: Optional[float] = Field(
        None,
        ge=-50,
        le=50,
        description="Minimum storage temperature"
    )
    storage_temperature_max_c: Optional[float] = Field(
        None,
        ge=-20,
        le=60,
        description="Maximum storage temperature"
    )
    shelf_life_days: Optional[int] = Field(
        None,
        ge=1,
        le=3650,
        description="Shelf life in days"
    )

    # Safety parameters
    flash_point_c: Optional[float] = Field(
        None,
        ge=-50,
        le=500,
        description="Flash point temperature"
    )
    auto_ignition_temp_c: Optional[float] = Field(
        None,
        ge=100,
        le=700,
        description="Auto-ignition temperature"
    )
    explosive_limits_lower_percent: Optional[float] = Field(
        None,
        ge=0,
        le=20,
        description="Lower explosive limit"
    )
    explosive_limits_upper_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Upper explosive limit"
    )

    # Source and certification
    source_region: Optional[str] = Field(
        None,
        description="Geographic source region"
    )
    certification: Optional[str] = Field(
        None,
        description="Certification (e.g., FSC, ISCC)"
    )

    @field_validator('net_calorific_value_mj_kg')
    @classmethod
    def validate_ncv_vs_gcv(cls, v, info):
        """Ensure NCV is less than or equal to GCV."""
        if 'gross_calorific_value_mj_kg' in info.data:
            gcv = info.data['gross_calorific_value_mj_kg']
            if v > gcv:
                raise ValueError('Net calorific value cannot exceed gross calorific value')
        return v

    @model_validator(mode='after')
    def validate_composition(self):
        """Validate fuel composition totals."""
        total = (
            self.carbon_content_percent +
            self.hydrogen_content_percent +
            self.oxygen_content_percent +
            self.nitrogen_content_percent +
            self.sulfur_content_percent +
            self.moisture_content_percent +
            self.ash_content_percent
        )
        if total > 101:  # Allow 1% tolerance
            raise ValueError(f'Fuel composition total ({total}%) exceeds 100%')
        return self


class FuelInventory(BaseModel):
    """
    Fuel inventory management model.

    Tracks current inventory levels, safety stocks, and reorder points
    for each fuel type at a facility.
    """

    fuel_id: str = Field(
        ...,
        description="Reference to fuel specification"
    )
    site_id: str = Field(
        ...,
        description="Facility site identifier"
    )
    storage_unit_id: str = Field(
        ...,
        description="Storage unit identifier"
    )

    # Current inventory
    current_quantity: float = Field(
        ...,
        ge=0,
        description="Current inventory quantity"
    )
    quantity_unit: str = Field(
        default="kg",
        description="Unit for quantity (kg, m3, tonnes)"
    )

    # Capacity limits
    storage_capacity: float = Field(
        ...,
        gt=0,
        description="Maximum storage capacity"
    )
    minimum_level: float = Field(
        default=0,
        ge=0,
        description="Minimum operational level"
    )

    # Safety and reorder
    safety_stock: float = Field(
        default=0,
        ge=0,
        description="Safety stock level"
    )
    reorder_point: float = Field(
        default=0,
        ge=0,
        description="Reorder trigger point"
    )
    reorder_quantity: float = Field(
        default=0,
        ge=0,
        description="Standard reorder quantity"
    )
    lead_time_days: int = Field(
        default=7,
        ge=0,
        le=180,
        description="Delivery lead time in days"
    )

    # Quality tracking
    quality_grade: Optional[str] = Field(
        None,
        description="Current quality grade"
    )
    receipt_date: Optional[datetime] = Field(
        None,
        description="Date of last receipt"
    )
    expiry_date: Optional[datetime] = Field(
        None,
        description="Expiry date if applicable"
    )

    # Cost tracking
    average_cost_per_unit: float = Field(
        default=0,
        ge=0,
        description="Weighted average cost per unit"
    )
    last_purchase_price: float = Field(
        default=0,
        ge=0,
        description="Most recent purchase price"
    )

    @field_validator('current_quantity')
    @classmethod
    def validate_within_capacity(cls, v, info):
        """Ensure current quantity doesn't exceed capacity."""
        if 'storage_capacity' in info.data:
            if v > info.data['storage_capacity']:
                raise ValueError('Current quantity exceeds storage capacity')
        return v

    @field_validator('reorder_point')
    @classmethod
    def validate_reorder_above_minimum(cls, v, info):
        """Ensure reorder point is above minimum level."""
        if 'minimum_level' in info.data:
            if v < info.data['minimum_level']:
                raise ValueError('Reorder point should be above minimum level')
        return v

    def days_of_supply(self, consumption_rate_per_day: float) -> float:
        """Calculate remaining days of supply."""
        if consumption_rate_per_day <= 0:
            return float('inf')
        return self.current_quantity / consumption_rate_per_day

    def needs_reorder(self) -> bool:
        """Check if reorder is needed."""
        return self.current_quantity <= self.reorder_point


class MarketPriceData(BaseModel):
    """
    Fuel market price data model.

    Captures current and historical pricing for fuel procurement
    optimization and forecasting.
    """

    fuel_id: str = Field(
        ...,
        description="Reference to fuel specification"
    )
    price_source: str = Field(
        ...,
        description="Price data source (exchange, supplier, index)"
    )

    # Current pricing
    current_price: float = Field(
        ...,
        ge=0,
        description="Current spot price"
    )
    price_unit: str = Field(
        default="USD/kg",
        description="Price unit"
    )
    currency: str = Field(
        default="USD",
        description="Currency code"
    )
    price_timestamp: datetime = Field(
        default_factory=DeterministicClock.now,
        description="Price timestamp"
    )

    # Price range
    price_low_24h: Optional[float] = Field(
        None,
        ge=0,
        description="24-hour low price"
    )
    price_high_24h: Optional[float] = Field(
        None,
        ge=0,
        description="24-hour high price"
    )
    price_avg_30d: Optional[float] = Field(
        None,
        ge=0,
        description="30-day average price"
    )

    # Volatility
    volatility_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Price volatility percentage"
    )

    # Contract pricing
    contract_price: Optional[float] = Field(
        None,
        ge=0,
        description="Contract/fixed price if applicable"
    )
    contract_volume: Optional[float] = Field(
        None,
        ge=0,
        description="Contract volume"
    )
    contract_expiry: Optional[datetime] = Field(
        None,
        description="Contract expiry date"
    )

    # Delivery
    delivery_premium: float = Field(
        default=0,
        ge=0,
        description="Delivery premium per unit"
    )
    minimum_order_quantity: float = Field(
        default=0,
        ge=0,
        description="Minimum order quantity"
    )

    @field_validator('price_high_24h')
    @classmethod
    def validate_high_above_low(cls, v, info):
        """Ensure high price is above low price."""
        if v is not None and 'price_low_24h' in info.data:
            low = info.data['price_low_24h']
            if low is not None and v < low:
                raise ValueError('24h high price must be >= 24h low price')
        return v


class BlendingConstraints(BaseModel):
    """
    Fuel blending constraints and requirements.

    Defines operational limits for fuel blending to ensure
    safe and efficient operation.
    """

    blend_id: str = Field(
        ...,
        description="Unique blend configuration identifier"
    )
    blend_name: str = Field(
        ...,
        description="Human-readable blend name"
    )

    # Blend composition limits
    fuel_limits: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Min/max percentages by fuel type"
    )
    # Format: {"natural_gas": {"min": 0.2, "max": 0.8}, ...}

    # Quality constraints
    min_heating_value_mj_kg: float = Field(
        default=15.0,
        ge=5,
        le=60,
        description="Minimum blend heating value"
    )
    max_moisture_percent: float = Field(
        default=20.0,
        ge=0,
        le=50,
        description="Maximum blend moisture content"
    )
    max_ash_percent: float = Field(
        default=15.0,
        ge=0,
        le=30,
        description="Maximum blend ash content"
    )
    max_sulfur_percent: float = Field(
        default=2.0,
        ge=0,
        le=5,
        description="Maximum blend sulfur content"
    )

    # Compatibility matrix
    incompatible_fuels: List[List[str]] = Field(
        default_factory=list,
        description="List of incompatible fuel pairs"
    )
    # Format: [["fuel_a", "fuel_b"], ...]

    # Operational constraints
    max_blend_components: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of fuels in blend"
    )
    min_component_percent: float = Field(
        default=5.0,
        ge=0,
        le=50,
        description="Minimum percentage for any component"
    )

    # Equipment compatibility
    compatible_burners: List[str] = Field(
        default_factory=list,
        description="List of compatible burner types"
    )
    requires_preprocessing: bool = Field(
        default=False,
        description="Whether blend requires preprocessing"
    )

    @field_validator('fuel_limits')
    @classmethod
    def validate_fuel_limits(cls, v):
        """Validate fuel limit structure."""
        for fuel, limits in v.items():
            if 'min' not in limits or 'max' not in limits:
                raise ValueError(f"Fuel {fuel} must have 'min' and 'max' limits")
            if limits['min'] > limits['max']:
                raise ValueError(f"Fuel {fuel} min cannot exceed max")
            if limits['max'] > 1.0:
                raise ValueError(f"Fuel {fuel} max cannot exceed 100%")
        return v


class EmissionLimits(BaseModel):
    """
    Regulatory emission limits for compliance monitoring.

    Captures pollutant limits per applicable regulatory framework.
    """

    limit_id: str = Field(
        ...,
        description="Unique limit configuration identifier"
    )
    standard: EmissionStandard = Field(
        ...,
        description="Regulatory standard"
    )
    jurisdiction: str = Field(
        ...,
        description="Applicable jurisdiction"
    )

    # CO2 limits
    co2_limit_kg_mwh: Optional[float] = Field(
        None,
        ge=0,
        le=2000,
        description="CO2 intensity limit"
    )
    co2_annual_cap_tonnes: Optional[float] = Field(
        None,
        ge=0,
        description="Annual CO2 cap"
    )

    # Criteria pollutants
    nox_limit_mg_nm3: float = Field(
        default=200,
        ge=0,
        le=2000,
        description="NOx limit in mg/Nm3"
    )
    nox_limit_g_gj: Optional[float] = Field(
        None,
        ge=0,
        le=500,
        description="NOx limit in g/GJ"
    )
    sox_limit_mg_nm3: float = Field(
        default=200,
        ge=0,
        le=2000,
        description="SOx limit in mg/Nm3"
    )
    sox_limit_g_gj: Optional[float] = Field(
        None,
        ge=0,
        le=1000,
        description="SOx limit in g/GJ"
    )
    pm_limit_mg_nm3: float = Field(
        default=30,
        ge=0,
        le=500,
        description="Particulate matter limit in mg/Nm3"
    )
    co_limit_mg_nm3: float = Field(
        default=100,
        ge=0,
        le=1000,
        description="CO limit in mg/Nm3"
    )

    # Reference conditions
    reference_oxygen_percent: float = Field(
        default=6.0,
        ge=0,
        le=21,
        description="Reference O2 for limit correction"
    )
    reference_temperature_c: float = Field(
        default=0,
        description="Reference temperature for normalization"
    )
    reference_pressure_kpa: float = Field(
        default=101.325,
        description="Reference pressure for normalization"
    )

    # Compliance tracking
    compliance_period: str = Field(
        default="continuous",
        description="Compliance averaging period"
    )
    effective_date: datetime = Field(
        default_factory=DeterministicClock.now,
        description="Limit effective date"
    )
    expiry_date: Optional[datetime] = Field(
        None,
        description="Limit expiry date if applicable"
    )

    # Penalties
    penalty_rate_per_kg: Optional[float] = Field(
        None,
        ge=0,
        description="Penalty rate for exceedance"
    )


class OptimizationParameters(BaseModel):
    """
    Parameters for fuel optimization algorithms.

    Controls optimization behavior, weights, and convergence criteria.
    """

    # Optimization objectives
    primary_objective: str = Field(
        default="balanced",
        description="Primary optimization objective"
    )
    secondary_objectives: List[str] = Field(
        default_factory=list,
        description="Secondary optimization objectives"
    )

    # Objective weights (must sum to 1.0)
    cost_weight: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Weight for cost minimization"
    )
    emissions_weight: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Weight for emissions minimization"
    )
    efficiency_weight: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Weight for efficiency maximization"
    )
    reliability_weight: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Weight for supply reliability"
    )

    # Algorithm parameters
    optimization_algorithm: str = Field(
        default="linear_programming",
        description="Optimization algorithm to use"
    )
    max_iterations: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum optimization iterations"
    )
    convergence_tolerance: float = Field(
        default=0.0001,
        ge=0,
        le=0.1,
        description="Convergence tolerance"
    )
    time_limit_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Maximum optimization time"
    )

    # Solution parameters
    solution_pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of alternative solutions to generate"
    )
    diversity_threshold: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Minimum diversity between solutions"
    )

    # Scenario parameters
    num_scenarios: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of scenarios for stochastic optimization"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.999,
        description="Confidence level for robust solutions"
    )

    @model_validator(mode='after')
    def validate_weights_sum(self):
        """Ensure weights sum to 1.0."""
        total = (
            self.cost_weight +
            self.emissions_weight +
            self.efficiency_weight +
            self.reliability_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f'Weights must sum to 1.0, got {total}')
        return self


class IntegrationSettings(BaseModel):
    """
    Settings for external system integration.

    Configures connections to ERP, market data, storage systems,
    and emissions monitoring platforms.
    """

    # ERP integration
    erp_enabled: bool = Field(
        default=True,
        description="Enable ERP integration"
    )
    erp_system: Optional[str] = Field(
        None,
        description="ERP system type (SAP, Oracle, etc.)"
    )
    erp_endpoint: Optional[str] = Field(
        None,
        description="ERP API endpoint"
    )

    # Market data integration
    market_data_enabled: bool = Field(
        default=True,
        description="Enable market data feed"
    )
    market_data_sources: List[str] = Field(
        default_factory=lambda: ["ICE", "NYMEX"],
        description="Market data sources"
    )
    price_update_interval_seconds: int = Field(
        default=300,
        ge=60,
        le=86400,
        description="Price update interval"
    )

    # Storage system integration
    storage_system_enabled: bool = Field(
        default=True,
        description="Enable storage system integration"
    )
    storage_protocol: str = Field(
        default="MODBUS",
        description="Storage system protocol"
    )

    # Emissions monitoring integration
    emissions_monitoring_enabled: bool = Field(
        default=True,
        description="Enable emissions monitoring"
    )
    emissions_monitoring_endpoint: Optional[str] = Field(
        None,
        description="CEMS endpoint"
    )

    # SCADA integration
    scada_enabled: bool = Field(
        default=True,
        description="Enable SCADA integration"
    )
    scada_endpoint: Optional[str] = Field(
        None,
        description="SCADA endpoint"
    )
    scada_polling_interval_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description="SCADA polling interval"
    )


class FuelManagementConfig(BaseModel):
    """
    Main configuration for FuelManagementOrchestrator agent (GL-011).

    Comprehensive configuration including all fuel specifications,
    inventory settings, market integration, and optimization parameters.

    SECURITY & COMPLIANCE:
    - Zero hardcoded credentials policy enforced
    - Deterministic mode required for regulatory compliance
    - TLS encryption mandatory for production
    - Provenance tracking required for audit trails
    - ISO 6976/ASTM D4809 fuel validation enforced
    """

    # Agent identification
    agent_id: str = Field(
        default="GL-011",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="FuelManagementOrchestrator",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production)"
    )

    # Fuel specifications (support multiple fuels)
    fuel_specifications: List[FuelSpecification] = Field(
        default_factory=list,
        description="Fuel specifications for all available fuels"
    )

    # Inventory configuration
    inventories: List[FuelInventory] = Field(
        default_factory=list,
        description="Fuel inventory configurations"
    )

    # Market data
    market_prices: List[MarketPriceData] = Field(
        default_factory=list,
        description="Market price data"
    )

    # Blending configuration
    blending_constraints: List[BlendingConstraints] = Field(
        default_factory=list,
        description="Blending constraint configurations"
    )

    # Emission limits
    emission_limits: List[EmissionLimits] = Field(
        default_factory=list,
        description="Emission limit configurations"
    )

    # Optimization parameters
    optimization: OptimizationParameters = Field(
        default_factory=OptimizationParameters,
        description="Optimization parameters"
    )

    # Integration settings
    integration: IntegrationSettings = Field(
        default_factory=IntegrationSettings,
        description="Integration settings"
    )

    # Performance settings
    enable_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    enable_learning: bool = Field(
        default=True,
        description="Enable learning from operations"
    )
    calculation_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Calculation timeout"
    )
    cache_ttl_seconds: int = Field(
        default=120,
        ge=10,
        le=3600,
        description="Cache time-to-live"
    )

    # Economic parameters
    carbon_price_usd_per_tonne: float = Field(
        default=50.0,
        ge=0,
        le=500,
        description="Carbon price for optimization"
    )
    discount_rate_percent: float = Field(
        default=8.0,
        ge=0,
        le=25,
        description="Discount rate for NPV calculations"
    )

    # Site information
    site_id: str = Field(
        default="SITE-001",
        description="Site identifier"
    )
    plant_id: str = Field(
        default="PLANT-001",
        description="Plant identifier"
    )
    plant_capacity_mw: float = Field(
        default=100.0,
        gt=0,
        description="Plant capacity in MW"
    )

    # Safety parameters
    enable_safety_limits: bool = Field(
        default=True,
        description="Enable safety limit enforcement"
    )
    safety_margin_percent: float = Field(
        default=5.0,
        ge=0,
        le=20,
        description="Safety margin for limits"
    )

    # ========================================================================
    # DETERMINISTIC SETTINGS (REGULATORY COMPLIANCE)
    # ========================================================================

    deterministic_mode: bool = Field(
        default=True,
        description="Enable deterministic mode (required for compliance)"
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=0.0,
        description="LLM temperature (must be 0.0 for determinism)"
    )

    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    # ========================================================================
    # SECURITY SETTINGS
    # ========================================================================

    zero_secrets: bool = Field(
        default=True,
        description="Enforce zero hardcoded secrets policy"
    )

    tls_enabled: bool = Field(
        default=True,
        description="Enable TLS 1.3 for API connections"
    )

    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )

    # ========================================================================
    # CALCULATION PARAMETERS
    # ========================================================================

    decimal_precision: int = Field(
        default=10,
        ge=6,
        le=20,
        description="Decimal precision for financial calculations"
    )

    supported_fuels: List[str] = Field(
        default_factory=lambda: [
            'natural_gas', 'coal', 'biomass', 'fuel_oil',
            'diesel', 'hydrogen', 'propane', 'syngas'
        ],
        description="ISO 6976/ASTM D4809 compliant fuel types"
    )

    # ========================================================================
    # OPERATIONAL SETTINGS
    # ========================================================================

    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (must be False in production)"
    )

    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'fuel_shortage': 0.15,
            'cost_overrun': 0.10,
            'emissions_violation': 0.05,
            'integration_failure': 0.0
        },
        description="Alert thresholds for operational monitoring"
    )

    # ========================================================================
    # VALIDATORS - COMPLIANCE ENFORCEMENT
    # ========================================================================

    @field_validator('fuel_specifications')
    @classmethod
    def validate_unique_fuel_ids(cls, v):
        """Ensure fuel IDs are unique."""
        fuel_ids = [f.fuel_id for f in v]
        if len(fuel_ids) != len(set(fuel_ids)):
            raise ValueError('Fuel IDs must be unique')
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Ensure temperature is 0.0 for deterministic operation."""
        if v != 0.0:
            raise ValueError(
                f"COMPLIANCE VIOLATION: temperature must be 0.0 for deterministic fuel optimization calculations. "
                f"Got: {v}. This ensures reproducible results for regulatory compliance."
            )
        return v

    @field_validator('seed')
    @classmethod
    def validate_seed(cls, v):
        """Ensure seed is 42 for bit-perfect reproducibility."""
        if v != 42:
            raise ValueError(
                f"COMPLIANCE VIOLATION: seed must be 42 for deterministic calculations. "
                f"Got: {v}. This ensures bit-perfect reproducibility across runs."
            )
        return v

    @field_validator('deterministic_mode')
    @classmethod
    def validate_deterministic(cls, v):
        """Ensure deterministic mode is enabled."""
        if not v:
            raise ValueError(
                "COMPLIANCE VIOLATION: deterministic_mode must be True for regulatory compliance. "
                "All fuel cost and emissions calculations must be reproducible for audit trails."
            )
        return v

    @field_validator('zero_secrets')
    @classmethod
    def validate_zero_secrets(cls, v):
        """Ensure zero_secrets policy is enabled."""
        if not v:
            raise ValueError(
                "SECURITY VIOLATION: zero_secrets must be True. "
                "API keys and credentials must never be in config.py. Use environment variables or secrets manager."
            )
        return v

    @field_validator('tls_enabled')
    @classmethod
    def validate_tls(cls, v):
        """Ensure TLS is enabled."""
        if not v:
            raise ValueError(
                "SECURITY VIOLATION: tls_enabled must be True for production deployments. "
                "All API connections must use TLS 1.3 for data protection."
            )
        return v

    @field_validator('supported_fuels')
    @classmethod
    def validate_fuels(cls, v):
        """Validate fuel types against ISO/ASTM standards."""
        allowed_fuels = {
            'natural_gas', 'coal', 'biomass', 'fuel_oil',
            'diesel', 'hydrogen', 'propane', 'syngas'
        }
        invalid = set(v) - allowed_fuels
        if invalid:
            raise ValueError(
                f"COMPLIANCE VIOLATION: Invalid fuel types: {invalid}. "
                f"Only ISO 6976 / ASTM D4809 compliant fuels are allowed: {allowed_fuels}"
            )
        return v

    @field_validator('decimal_precision')
    @classmethod
    def validate_precision(cls, v):
        """Validate decimal precision for financial calculations."""
        if v < 10:
            raise ValueError(
                "COMPLIANCE VIOLATION: decimal_precision must be >= 10 for financial calculations. "
                f"Got: {v}. Required for accurate cost optimization to 0.0000000001 precision."
            )
        return v

    @field_validator('enable_provenance')
    @classmethod
    def validate_provenance(cls, v):
        """Ensure provenance tracking is enabled."""
        if not v:
            raise ValueError(
                "COMPLIANCE VIOLATION: enable_provenance must be True. "
                "SHA-256 audit trails are required for all optimization decisions."
            )
        return v

    @field_validator('alert_thresholds')
    @classmethod
    def validate_thresholds(cls, v):
        """Validate required alert thresholds are configured."""
        required_alerts = {
            'fuel_shortage', 'cost_overrun',
            'emissions_violation', 'integration_failure'
        }
        missing = required_alerts - set(v.keys())
        if missing:
            raise ValueError(
                f"COMPLIANCE VIOLATION: Missing required alert thresholds: {missing}. "
                "These alerts are mandatory for safe fuel management operations."
            )

        # Validate threshold ranges
        for alert_type, threshold in v.items():
            if threshold < 0 or threshold > 1:
                raise ValueError(
                    f"COMPLIANCE VIOLATION: Alert threshold '{alert_type}' must be between 0 and 1. "
                    f"Got: {threshold}"
                )

        return v

    @model_validator(mode='after')
    def validate_environment_consistency(self):
        """Validate configuration consistency across environments."""
        if self.environment == 'production':
            # Production environment checks
            if not self.tls_enabled:
                raise ValueError(
                    "SECURITY VIOLATION: TLS required in production environment. "
                    "Set tls_enabled=True for production deployments."
                )

            if not self.deterministic_mode:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Deterministic mode required in production environment. "
                    "Set deterministic_mode=True for regulatory compliance."
                )

            if self.debug_mode:
                raise ValueError(
                    "SECURITY VIOLATION: debug_mode must be False in production environment. "
                    "Debug mode exposes sensitive operational data."
                )

            if not self.enable_provenance:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Provenance tracking required in production. "
                    "Set enable_provenance=True for audit trails."
                )

            if not self.enable_audit_logging:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Audit logging required in production. "
                    "Set enable_audit_logging=True for compliance."
                )

        # Validate calculation timeout is reasonable
        if self.calculation_timeout_seconds > 60:
            raise ValueError(
                "PERFORMANCE VIOLATION: calculation_timeout_seconds should not exceed 60 seconds. "
                f"Got: {self.calculation_timeout_seconds}. Long timeouts indicate inefficient calculations."
            )

        # Validate optimization weights if using balanced objective
        if hasattr(self.optimization, 'primary_objective'):
            if self.optimization.primary_objective == 'balanced':
                total_weight = (
                    self.optimization.cost_weight +
                    self.optimization.emissions_weight +
                    self.optimization.efficiency_weight +
                    self.optimization.reliability_weight
                )
                if abs(total_weight - 1.0) > 0.001:
                    raise ValueError(
                        f"COMPLIANCE VIOLATION: Optimization weights must sum to 1.0 for balanced objective. "
                        f"Got total: {total_weight}"
                    )

        return self

    # ========================================================================
    # RUNTIME ASSERTION HELPERS
    # ========================================================================

    def assert_compliance_ready(self) -> None:
        """
        Assert configuration is ready for compliance/production use.

        Raises:
            AssertionError: If any compliance requirements are not met

        Example:
            >>> config = FuelManagementConfig()
            >>> config.assert_compliance_ready()  # Raises if not compliant
        """
        assert self.deterministic_mode, "Deterministic mode required for compliance"
        assert self.temperature == 0.0, "Temperature must be 0.0 for determinism"
        assert self.seed == 42, "Seed must be 42 for reproducibility"
        assert self.zero_secrets, "Zero secrets policy must be enforced"
        assert self.enable_provenance, "Provenance tracking required"
        assert self.tls_enabled, "TLS encryption required"

        if self.environment == 'production':
            assert not self.debug_mode, "Debug mode not allowed in production"
            assert self.enable_audit_logging, "Audit logging required in production"

        # Validate alert thresholds configured
        required_alerts = {
            'fuel_shortage', 'cost_overrun',
            'emissions_violation', 'integration_failure'
        }
        assert required_alerts.issubset(set(self.alert_thresholds.keys())), \
            f"Missing required alerts: {required_alerts - set(self.alert_thresholds.keys())}"

    def assert_security_ready(self) -> None:
        """
        Assert configuration meets security requirements.

        Raises:
            AssertionError: If any security requirements are not met
        """
        assert self.zero_secrets, "Zero secrets policy must be enforced"
        assert self.tls_enabled, "TLS encryption must be enabled"

        # Check integration endpoints don't contain credentials
        if self.integration.erp_endpoint:
            from urllib.parse import urlparse
            parsed = urlparse(self.integration.erp_endpoint)
            assert not parsed.username and not parsed.password, \
                "ERP endpoint must not contain embedded credentials"

        if self.integration.emissions_monitoring_endpoint:
            from urllib.parse import urlparse
            parsed = urlparse(self.integration.emissions_monitoring_endpoint)
            assert not parsed.username and not parsed.password, \
                "CEMS endpoint must not contain embedded credentials"

        if self.integration.scada_endpoint:
            from urllib.parse import urlparse
            parsed = urlparse(self.integration.scada_endpoint)
            assert not parsed.username and not parsed.password, \
                "SCADA endpoint must not contain embedded credentials"

    def assert_determinism_ready(self) -> None:
        """
        Assert configuration ensures deterministic calculations.

        Raises:
            AssertionError: If any determinism requirements are not met
        """
        assert self.deterministic_mode, "Deterministic mode must be enabled"
        assert self.temperature == 0.0, "Temperature must be 0.0"
        assert self.seed == 42, "Seed must be 42"
        assert self.decimal_precision >= 10, "Decimal precision must be >= 10"

        # Validate optimization timeout is deterministic-friendly
        assert self.optimization.time_limit_seconds <= 300, \
            "Optimization timeout should be <= 300 seconds for deterministic results"


# ============================================================================
# DEFAULT CONFIGURATION FACTORY
# ============================================================================

def create_default_config() -> FuelManagementConfig:
    """
    Create default configuration for testing and demonstration.

    Returns:
        FuelManagementConfig with default values for common fuels.
    """
    # Default fuel specifications
    natural_gas = FuelSpecification(
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
        is_renewable=False
    )

    coal = FuelSpecification(
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
        is_renewable=False
    )

    biomass = FuelSpecification(
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
        certification="ENplus A1"
    )

    hydrogen = FuelSpecification(
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
        explosive_limits_upper_percent=75.0
    )

    # Default emission limits (EU IED)
    emission_limit = EmissionLimits(
        limit_id="EU-IED-001",
        standard=EmissionStandard.EU_IED,
        jurisdiction="EU",
        nox_limit_mg_nm3=150,
        sox_limit_mg_nm3=150,
        pm_limit_mg_nm3=10,
        co_limit_mg_nm3=100,
        reference_oxygen_percent=6.0
    )

    return FuelManagementConfig(
        fuel_specifications=[natural_gas, coal, biomass, hydrogen],
        emission_limits=[emission_limit],
        site_id="DEMO-SITE-001",
        plant_id="DEMO-PLANT-001",
        plant_capacity_mw=100.0,
        carbon_price_usd_per_tonne=50.0
    )
