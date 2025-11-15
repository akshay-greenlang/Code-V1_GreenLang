"""
Configuration module for BoilerEfficiencyOptimizer agent (GL-002).

This module defines the configuration models and settings for the
BoilerEfficiencyOptimizer agent, including boiler specifications,
operational constraints, optimization parameters, and integration settings.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from pathlib import Path


class BoilerSpecification(BaseModel):
    """Boiler technical specifications."""

    boiler_id: str = Field(..., min_length=1, max_length=50, description="Unique boiler identifier")
    manufacturer: str = Field(..., min_length=1, max_length=100, description="Boiler manufacturer")
    model: str = Field(..., min_length=1, max_length=100, description="Boiler model")
    type: str = Field(..., description="Boiler type (fire-tube, water-tube, etc.)")

    # Capacity specifications
    max_steam_capacity_kg_hr: float = Field(..., ge=1000, le=1000000, description="Maximum steam generation capacity")
    min_steam_capacity_kg_hr: float = Field(..., ge=100, le=500000, description="Minimum steam generation capacity")
    design_pressure_bar: float = Field(..., gt=0, le=300, description="Design pressure in bar")
    design_temperature_c: float = Field(..., ge=100, le=600, description="Design temperature in Celsius")

    # Fuel specifications
    primary_fuel_type: str = Field(..., description="Primary fuel type (natural_gas, coal, oil, biomass)")
    secondary_fuel_type: Optional[str] = Field(None, description="Secondary fuel type if dual-fuel")
    fuel_heating_value_mj_kg: float = Field(..., gt=0, le=100, description="Fuel heating value in MJ/kg")

    # Efficiency specifications
    design_efficiency_percent: float = Field(..., ge=50, le=100, description="Design thermal efficiency")
    actual_efficiency_percent: float = Field(..., ge=30, le=100, description="Current actual efficiency")

    # Physical dimensions
    heating_surface_area_m2: float = Field(..., gt=0, le=100000, description="Total heating surface area")
    furnace_volume_m3: float = Field(..., gt=0, le=10000, description="Furnace volume")

    # Age and maintenance
    commissioning_date: datetime = Field(..., description="Commissioning date")
    last_major_overhaul: Optional[datetime] = Field(None, description="Last major overhaul date")
    operating_hours: int = Field(..., ge=0, le=1000000, description="Total operating hours")

    @validator('max_steam_capacity_kg_hr')
    def validate_max_steam_capacity(cls, v: float, values: Dict) -> float:
        """Ensure max steam capacity >= min steam capacity."""
        if 'min_steam_capacity_kg_hr' in values and v < values['min_steam_capacity_kg_hr']:
            raise ValueError('max_steam_capacity_kg_hr must be >= min_steam_capacity_kg_hr')
        return v

    @validator('design_temperature_c')
    def validate_design_temperature(cls, v: float) -> float:
        """Validate design temperature."""
        if not (100 <= v <= 600):
            raise ValueError('Design temperature must be between 100 and 600 Celsius')
        return v

    @validator('commissioning_date')
    def validate_commissioning_date(cls, v: datetime) -> datetime:
        """Validate commissioning date is not in future."""
        if v > datetime.now():
            raise ValueError('Commissioning date cannot be in the future')
        return v

    @validator('actual_efficiency_percent')
    def validate_actual_efficiency(cls, v: float, values: Dict) -> float:
        """Validate actual efficiency is not greater than design efficiency."""
        if 'design_efficiency_percent' in values and v > values['design_efficiency_percent']:
            raise ValueError('Actual efficiency cannot exceed design efficiency')
        return v


class OperationalConstraints(BaseModel):
    """Operational constraints for boiler optimization."""

    # Safety constraints
    max_pressure_bar: float = Field(..., gt=0, description="Maximum operating pressure")
    min_pressure_bar: float = Field(..., ge=0, description="Minimum operating pressure")
    max_temperature_c: float = Field(..., gt=100, description="Maximum operating temperature")
    min_temperature_c: float = Field(..., ge=0, description="Minimum operating temperature")

    # Combustion constraints
    min_excess_air_percent: float = Field(5.0, ge=0, le=100, description="Minimum excess air")
    max_excess_air_percent: float = Field(25.0, ge=0, le=100, description="Maximum excess air")
    min_o2_percent: float = Field(2.0, ge=0, le=21, description="Minimum O2 in flue gas")
    max_co_ppm: float = Field(100.0, ge=0, le=1000, description="Maximum CO in flue gas")

    # Load constraints
    min_load_percent: float = Field(20.0, ge=0, le=100, description="Minimum operating load")
    max_load_percent: float = Field(100.0, ge=0, le=200, description="Maximum operating load")
    max_load_change_rate_percent_min: float = Field(5.0, ge=0, le=100, description="Max load change rate %/min")

    # Steam quality constraints
    min_steam_quality: float = Field(0.95, ge=0.0, le=1.0, description="Minimum steam quality")
    max_tds_ppm: float = Field(3500, ge=0, le=10000, description="Maximum total dissolved solids")
    max_moisture_percent: float = Field(0.5, ge=0, le=100, description="Maximum moisture in steam")

    @validator('min_pressure_bar', 'max_pressure_bar')
    def validate_pressure_range(cls, v: float) -> float:
        """Validate pressure is in reasonable operating range."""
        if not (0 < v <= 250):
            raise ValueError('Pressure must be between 0 and 250 bar')
        return v

    @validator('min_temperature_c', 'max_temperature_c')
    def validate_temperature_range(cls, v: float) -> float:
        """Validate temperature is in reasonable operating range."""
        if not (-50 <= v <= 600):
            raise ValueError('Temperature must be between -50 and 600 Celsius')
        return v

    @validator('max_pressure_bar')
    def validate_max_min_pressure(cls, v: float, values: Dict) -> float:
        """Ensure max pressure >= min pressure."""
        if 'min_pressure_bar' in values and v < values['min_pressure_bar']:
            raise ValueError('max_pressure_bar must be >= min_pressure_bar')
        return v

    @validator('max_temperature_c')
    def validate_max_min_temperature(cls, v: float, values: Dict) -> float:
        """Ensure max temperature >= min temperature."""
        if 'min_temperature_c' in values and v < values['min_temperature_c']:
            raise ValueError('max_temperature_c must be >= min_temperature_c')
        return v

    @validator('max_excess_air_percent')
    def validate_excess_air_range(cls, v: float, values: Dict) -> float:
        """Ensure max excess air >= min excess air."""
        if 'min_excess_air_percent' in values and v < values['min_excess_air_percent']:
            raise ValueError('max_excess_air_percent must be >= min_excess_air_percent')
        return v

    @validator('max_load_percent')
    def validate_load_range(cls, v: float, values: Dict) -> float:
        """Ensure max load >= min load."""
        if 'min_load_percent' in values and v < values['min_load_percent']:
            raise ValueError('max_load_percent must be >= min_load_percent')
        return v


class EmissionLimits(BaseModel):
    """Regulatory emission limits."""

    nox_limit_ppm: float = Field(..., ge=0, le=1000, description="NOx emission limit in ppm")
    nox_limit_mg_nm3: Optional[float] = Field(None, ge=0, le=5000, description="NOx limit in mg/Nm³")
    co_limit_ppm: float = Field(..., ge=0, le=1000, description="CO emission limit in ppm")
    co_limit_mg_nm3: Optional[float] = Field(None, ge=0, le=5000, description="CO limit in mg/Nm³")
    so2_limit_ppm: Optional[float] = Field(None, ge=0, le=500, description="SO2 emission limit if applicable")
    pm_limit_mg_nm3: Optional[float] = Field(None, ge=0, le=500, description="Particulate matter limit")

    # CO2 targets
    co2_intensity_target_kg_mwh: Optional[float] = Field(None, ge=0, le=1000, description="CO2 intensity target")
    co2_reduction_target_percent: Optional[float] = Field(None, ge=0, le=100, description="CO2 reduction target")

    # Regulatory framework
    regulation_standard: str = Field(..., min_length=1, max_length=100, description="Applicable regulation (EPA, EU-MCP, etc.)")
    compliance_deadline: Optional[datetime] = Field(None, description="Compliance deadline")

    @validator('nox_limit_ppm', 'co_limit_ppm')
    def validate_emission_limits(cls, v: float) -> float:
        """Validate emission limits are positive."""
        if v < 0:
            raise ValueError('Emission limits must be non-negative')
        return v

    @validator('co2_reduction_target_percent')
    def validate_co2_reduction(cls, v: Optional[float]) -> Optional[float]:
        """Validate CO2 reduction target is between 0 and 100 percent."""
        if v is not None and not (0 <= v <= 100):
            raise ValueError('CO2 reduction target must be between 0 and 100 percent')
        return v

    @validator('compliance_deadline')
    def validate_compliance_deadline(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Validate compliance deadline is in the future."""
        if v is not None and v < datetime.now():
            raise ValueError('Compliance deadline cannot be in the past')
        return v


class OptimizationParameters(BaseModel):
    """Parameters for optimization algorithms."""

    # Optimization targets
    primary_objective: str = Field(
        "efficiency",
        description="Primary optimization objective (efficiency, emissions, cost)"
    )
    secondary_objectives: List[str] = Field(
        default_factory=list,
        description="Secondary optimization objectives"
    )

    # Algorithm parameters
    optimization_interval_seconds: int = Field(60, ge=10, description="Optimization cycle interval")
    prediction_horizon_minutes: int = Field(30, ge=5, description="Prediction horizon for optimization")
    control_horizon_minutes: int = Field(10, ge=5, description="Control horizon for adjustments")

    # Convergence parameters
    convergence_tolerance: float = Field(0.001, description="Convergence tolerance")
    max_iterations: int = Field(100, description="Maximum iterations for optimization")

    # Weights for multi-objective optimization
    efficiency_weight: float = Field(0.4, ge=0, le=1, description="Weight for efficiency objective")
    emissions_weight: float = Field(0.3, ge=0, le=1, description="Weight for emissions objective")
    cost_weight: float = Field(0.3, ge=0, le=1, description="Weight for cost objective")

    @validator('efficiency_weight')
    def validate_weights(cls, v, values):
        """Ensure weights sum to 1.0."""
        if 'emissions_weight' in values and 'cost_weight' in values:
            total = v + values['emissions_weight'] + values['cost_weight']
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


class IntegrationSettings(BaseModel):
    """Settings for system integration."""

    # SCADA integration
    scada_enabled: bool = Field(True, description="Enable SCADA integration")
    scada_endpoint: Optional[str] = Field(None, description="SCADA system endpoint")
    scada_polling_interval_seconds: int = Field(5, description="SCADA polling interval")
    scada_tags: List[str] = Field(default_factory=list, description="SCADA tags to monitor")

    # DCS integration
    dcs_enabled: bool = Field(True, description="Enable DCS integration")
    dcs_endpoint: Optional[str] = Field(None, description="DCS system endpoint")
    dcs_write_enabled: bool = Field(False, description="Enable DCS write-back")
    dcs_control_tags: List[str] = Field(default_factory=list, description="DCS control tags")

    # Historian integration
    historian_enabled: bool = Field(True, description="Enable historian integration")
    historian_endpoint: Optional[str] = Field(None, description="Historian endpoint")
    historian_retention_days: int = Field(365, description="Data retention period")

    # Alert integration
    alert_enabled: bool = Field(True, description="Enable alerting")
    alert_channels: List[str] = Field(
        default_factory=lambda: ["email", "sms"],
        description="Alert channels"
    )
    alert_recipients: List[str] = Field(default_factory=list, description="Alert recipients")


class BoilerConfiguration(BaseModel):
    """Complete boiler configuration."""

    specification: BoilerSpecification = Field(..., description="Boiler specifications")
    constraints: OperationalConstraints = Field(..., description="Operational constraints")
    emission_limits: EmissionLimits = Field(..., description="Emission limits")
    optimization: OptimizationParameters = Field(..., description="Optimization parameters")
    integration: IntegrationSettings = Field(..., description="Integration settings")

    # Additional configuration
    site_id: str = Field(..., description="Site identifier")
    plant_id: str = Field(..., description="Plant identifier")
    unit_id: str = Field(..., description="Unit identifier")

    # Operational modes
    enabled_modes: List[str] = Field(
        default_factory=lambda: ["normal", "high_efficiency", "low_load"],
        description="Enabled operational modes"
    )

    # Performance baselines
    baseline_efficiency_percent: float = Field(..., description="Baseline efficiency for comparison")
    baseline_fuel_consumption_kg_hr: float = Field(..., description="Baseline fuel consumption")
    baseline_emissions_kg_hr: float = Field(..., description="Baseline CO2 emissions")


class BoilerEfficiencyConfig(BaseModel):
    """Main configuration for BoilerEfficiencyOptimizer agent."""

    # Agent identification
    agent_id: str = Field("GL-002", description="Agent identifier")
    agent_name: str = Field("BoilerEfficiencyOptimizer", description="Agent name")
    version: str = Field("1.0.0", description="Agent version")

    # Boiler configurations (support multiple boilers)
    boilers: List[BoilerConfiguration] = Field(..., description="Boiler configurations")
    primary_boiler_id: str = Field(..., description="Primary boiler ID for optimization")

    # Performance settings
    enable_monitoring: bool = Field(True, description="Enable performance monitoring")
    enable_learning: bool = Field(True, description="Enable learning from operations")
    enable_predictive: bool = Field(False, description="Enable predictive optimization")

    # Calculation settings
    calculation_timeout_seconds: int = Field(30, description="Calculation timeout")
    cache_ttl_seconds: int = Field(60, description="Cache time-to-live")

    # Economic parameters
    fuel_cost_usd_per_kg: float = Field(..., description="Fuel cost $/kg")
    steam_value_usd_per_ton: float = Field(..., description="Steam value $/ton")
    carbon_credit_usd_per_ton: float = Field(0.0, description="Carbon credit value $/ton CO2")
    efficiency_value_usd_per_percent: float = Field(1000.0, description="Value per % efficiency gain")

    # Safety parameters
    enable_safety_limits: bool = Field(True, description="Enable safety limit enforcement")
    safety_margin_percent: float = Field(5.0, description="Safety margin for limits")
    emergency_shutdown_enabled: bool = Field(True, description="Enable emergency shutdown")

    # Reporting settings
    report_interval_minutes: int = Field(60, description="Reporting interval")
    report_recipients: List[str] = Field(default_factory=list, description="Report recipients")

    # Compliance settings
    compliance_monitoring_enabled: bool = Field(True, description="Enable compliance monitoring")
    compliance_report_frequency: str = Field("daily", description="Compliance report frequency")

    # Optimization boundaries
    max_steam_capacity_kg_hr: float = Field(..., description="Maximum steam capacity")
    min_acceptable_efficiency: float = Field(70.0, description="Minimum acceptable efficiency")

    # Emission regulations
    emission_regulations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Applicable emission regulations"
    )

    @validator('primary_boiler_id')
    def validate_primary_boiler(cls, v, values):
        """Ensure primary boiler ID exists in boilers list."""
        if 'boilers' in values:
            boiler_ids = [b.specification.boiler_id for b in values['boilers']]
            if v not in boiler_ids:
                raise ValueError(f"Primary boiler ID {v} not found in boilers list")
        return v


# Default configuration factory
def create_default_config() -> BoilerEfficiencyConfig:
    """Create default configuration for testing."""

    boiler_spec = BoilerSpecification(
        boiler_id="BOILER-001",
        manufacturer="Cleaver-Brooks",
        model="CB-8000",
        type="water-tube",
        max_steam_capacity_kg_hr=50000,
        min_steam_capacity_kg_hr=10000,
        design_pressure_bar=40,
        design_temperature_c=450,
        primary_fuel_type="natural_gas",
        fuel_heating_value_mj_kg=50,
        design_efficiency_percent=85,
        actual_efficiency_percent=80,
        heating_surface_area_m2=500,
        furnace_volume_m3=100,
        commissioning_date=datetime(2020, 1, 1),
        operating_hours=20000
    )

    constraints = OperationalConstraints(
        max_pressure_bar=42,
        min_pressure_bar=5,
        max_temperature_c=480,
        min_temperature_c=150
    )

    emission_limits = EmissionLimits(
        nox_limit_ppm=30,
        nox_limit_mg_nm3=65,
        co_limit_ppm=50,
        co_limit_mg_nm3=100,
        regulation_standard="EPA-NSPS"
    )

    optimization = OptimizationParameters(
        primary_objective="efficiency",
        secondary_objectives=["emissions", "cost"]
    )

    integration = IntegrationSettings(
        scada_enabled=True,
        dcs_enabled=True,
        historian_enabled=True,
        alert_enabled=True
    )

    boiler_config = BoilerConfiguration(
        specification=boiler_spec,
        constraints=constraints,
        emission_limits=emission_limits,
        optimization=optimization,
        integration=integration,
        site_id="SITE-001",
        plant_id="PLANT-001",
        unit_id="UNIT-001",
        baseline_efficiency_percent=75,
        baseline_fuel_consumption_kg_hr=2000,
        baseline_emissions_kg_hr=4000
    )

    return BoilerEfficiencyConfig(
        boilers=[boiler_config],
        primary_boiler_id="BOILER-001",
        fuel_cost_usd_per_kg=0.05,
        steam_value_usd_per_ton=30,
        max_steam_capacity_kg_hr=50000
    )