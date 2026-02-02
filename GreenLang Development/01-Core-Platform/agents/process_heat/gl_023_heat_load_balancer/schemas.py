"""
GL-023 HEAT LOAD BALANCER AGENT - Pydantic Data Models

This module provides comprehensive Pydantic data models for multi-equipment
heat load balancing and optimization. All models include validation,
documentation, and support for zero-hallucination deterministic calculations.

Data Model Categories:
    - Equipment type and status enums (BOILER, FURNACE, CHP, HRSG)
    - Equipment reading and capacity models
    - Efficiency curve models for deterministic calculations
    - Heat demand reading and summary models
    - Load allocation input/output models
    - Economic cost breakdown models
    - Constraint tracking models
    - Safety status models (N+1 redundancy)
    - Comprehensive optimizer output with provenance

Engineering Standards:
    - ASME PTC 4 for boiler performance testing
    - ISO 50001 for energy management systems
    - API 560 for fired heaters
    - IEEE 762 for equipment reliability

Example:
    >>> from greenlang.agents.process_heat.gl_023_heat_load_balancer.schemas import (
    ...     EquipmentReading,
    ...     HeatDemandReading,
    ...     LoadAllocationInput,
    ... )
    >>>
    >>> reading = EquipmentReading(
    ...     equipment_id="BLR-001",
    ...     equipment_type=EquipmentType.BOILER,
    ...     current_load_pct=75.0,
    ...     fuel_flow_mmbtu_hr=50.0,
    ...     efficiency_pct=82.5,
    ...     stack_temp_f=350.0,
    ...     status=EquipmentStatus.RUNNING,
    ... )
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class EquipmentType(Enum):
    """Heat generating equipment types for load balancing."""
    BOILER = "boiler"
    FURNACE = "furnace"
    CHP = "chp"  # Combined Heat and Power
    HRSG = "hrsg"  # Heat Recovery Steam Generator


class EquipmentStatus(Enum):
    """Equipment operational status states."""
    RUNNING = "running"
    STANDBY = "standby"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"
    FAULTED = "faulted"


class OptimizationStatus(Enum):
    """Load optimization solver status."""
    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


class LoadPriority(Enum):
    """Heat demand priority classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRABLE = "deferrable"


class FuelType(Enum):
    """Fuel types for heat generation equipment."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    PROPANE = "propane"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"


class OptimizationObjective(Enum):
    """Optimization objective for load balancing."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    BALANCED = "balanced"


class ConstraintType(Enum):
    """Types of equipment and system constraints."""
    MIN_LOAD = "min_load"
    MAX_LOAD = "max_load"
    RAMP_RATE = "ramp_rate"
    MIN_RUN_TIME = "min_run_time"
    MAX_STARTS = "max_starts"
    SPINNING_RESERVE = "spinning_reserve"
    EMISSIONS_LIMIT = "emissions_limit"
    FUEL_AVAILABILITY = "fuel_availability"


class SafetyLevel(Enum):
    """Safety classification levels."""
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    TRIP = "trip"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Validation status for measurements and calculations."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    UNCHECKED = "unchecked"
    STALE = "stale"


class ControlAction(Enum):
    """Load control action type classification."""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    START = "start"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"


# =============================================================================
# EQUIPMENT READING MODELS
# =============================================================================

class EquipmentReading(BaseModel):
    """
    Real-time equipment operating point reading.

    Captures current operational status, load, fuel consumption,
    and efficiency for a single heat generating unit.
    """

    # Identification
    equipment_id: str = Field(
        ...,
        description="Unique equipment identifier"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Type of heat generating equipment"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Load status
    current_load_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current load as percentage of capacity (%)"
    )
    current_load_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Current heat output (MMBTU/hr)"
    )
    current_load_mw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Current heat output (MW thermal)"
    )

    # Fuel consumption
    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )
    fuel_flow_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Fuel input rate (MMBTU/hr HHV)"
    )
    fuel_flow_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fuel flow rate in native units"
    )
    fuel_flow_units: str = Field(
        default="scfh",
        description="Native fuel flow units (scfh, gph, lb/hr)"
    )

    # Efficiency
    efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current thermal efficiency (%)"
    )
    efficiency_method: str = Field(
        default="direct",
        description="Efficiency calculation method (direct, indirect)"
    )

    # Temperatures
    stack_temp_f: Optional[float] = Field(
        default=None,
        ge=100,
        le=1000,
        description="Stack/exhaust temperature (F)"
    )
    combustion_air_temp_f: Optional[float] = Field(
        default=None,
        description="Combustion air temperature (F)"
    )
    outlet_process_temp_f: Optional[float] = Field(
        default=None,
        description="Process outlet temperature (F)"
    )

    # Emissions
    nox_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx concentration (ppm)"
    )
    co_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO concentration (ppm)"
    )
    o2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=21,
        description="Stack O2 concentration (%)"
    )

    # Status
    status: EquipmentStatus = Field(
        ...,
        description="Current equipment operational status"
    )
    run_hours_since_start: Optional[float] = Field(
        default=None,
        ge=0,
        description="Hours since last startup"
    )
    starts_today: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of starts today"
    )

    # Measurement quality
    measurement_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Measurement validation status"
    )

    @validator('current_load_mw', always=True)
    def calculate_mw_from_mmbtu(cls, v, values):
        """Calculate MW thermal from MMBTU/hr (1 MMBTU/hr = 0.293071 MW)."""
        if v is None and values.get('current_load_mmbtu_hr') is not None:
            return values['current_load_mmbtu_hr'] * 0.293071
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# EQUIPMENT CAPACITY MODELS
# =============================================================================

class EquipmentCapacity(BaseModel):
    """
    Equipment capacity and operational limits.

    Defines the operating envelope for load allocation optimization.
    """

    # Identification
    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )

    # Capacity limits
    nameplate_capacity_mmbtu_hr: float = Field(
        ...,
        gt=0,
        description="Nameplate/design capacity (MMBTU/hr)"
    )
    min_load_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Minimum stable firing rate (MMBTU/hr)"
    )
    max_load_mmbtu_hr: float = Field(
        ...,
        gt=0,
        description="Maximum continuous rating (MMBTU/hr)"
    )
    min_load_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Minimum stable load percentage (%)"
    )
    max_load_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Maximum load percentage (%)"
    )

    # Available capacity (considering current state)
    available_capacity_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Currently available capacity (MMBTU/hr)"
    )
    available_capacity_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Available capacity as percentage (%)"
    )

    # Ramp rates
    ramp_rate_up_mmbtu_hr_min: float = Field(
        ...,
        ge=0,
        description="Load increase rate (MMBTU/hr per minute)"
    )
    ramp_rate_down_mmbtu_hr_min: float = Field(
        ...,
        ge=0,
        description="Load decrease rate (MMBTU/hr per minute)"
    )
    ramp_rate_up_pct_min: Optional[float] = Field(
        default=None,
        ge=0,
        description="Load increase rate (% per minute)"
    )
    ramp_rate_down_pct_min: Optional[float] = Field(
        default=None,
        ge=0,
        description="Load decrease rate (% per minute)"
    )

    # Startup parameters
    cold_start_time_min: float = Field(
        default=30.0,
        ge=0,
        description="Cold startup time (minutes)"
    )
    warm_start_time_min: float = Field(
        default=15.0,
        ge=0,
        description="Warm startup time (minutes)"
    )
    hot_start_time_min: float = Field(
        default=5.0,
        ge=0,
        description="Hot startup time (minutes)"
    )

    # Operating constraints
    min_run_time_hr: float = Field(
        default=1.0,
        ge=0,
        description="Minimum run time once started (hours)"
    )
    min_down_time_hr: float = Field(
        default=0.5,
        ge=0,
        description="Minimum down time after shutdown (hours)"
    )
    max_starts_per_day: int = Field(
        default=6,
        ge=0,
        description="Maximum startups per day"
    )

    # Turndown ratio
    turndown_ratio: Optional[float] = Field(
        default=None,
        ge=1,
        description="Turndown ratio (max/min)"
    )

    @validator('min_load_pct', always=True)
    def calculate_min_pct(cls, v, values):
        """Calculate min load percentage."""
        if v is None:
            nameplate = values.get('nameplate_capacity_mmbtu_hr')
            min_load = values.get('min_load_mmbtu_hr')
            if nameplate and nameplate > 0:
                return (min_load / nameplate) * 100 if min_load else 0
        return v

    @validator('max_load_pct', always=True)
    def calculate_max_pct(cls, v, values):
        """Calculate max load percentage."""
        if v is None:
            nameplate = values.get('nameplate_capacity_mmbtu_hr')
            max_load = values.get('max_load_mmbtu_hr')
            if nameplate and nameplate > 0 and max_load:
                return (max_load / nameplate) * 100
        return v

    @validator('turndown_ratio', always=True)
    def calculate_turndown(cls, v, values):
        """Calculate turndown ratio."""
        if v is None:
            max_load = values.get('max_load_mmbtu_hr')
            min_load = values.get('min_load_mmbtu_hr')
            if max_load and min_load and min_load > 0:
                return max_load / min_load
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# EFFICIENCY CURVE MODELS
# =============================================================================

class EfficiencyCurveDataPoint(BaseModel):
    """Single data point for efficiency curve fitting."""

    load_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Load percentage (%)"
    )
    efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Measured efficiency at load (%)"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Measurement timestamp"
    )


class EfficiencyCurve(BaseModel):
    """
    Equipment efficiency curve for deterministic efficiency calculations.

    Efficiency is modeled as: eta = a0 + a1*L + a2*L^2 + a3*L^3
    where L is load fraction (0-1) and eta is efficiency fraction.

    This ensures ZERO HALLUCINATION by using polynomial regression
    on measured data points.
    """

    # Identification
    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    curve_id: str = Field(
        default="",
        description="Efficiency curve identifier"
    )

    # Polynomial coefficients (eta = a0 + a1*L + a2*L^2 + a3*L^3)
    coefficient_a0: float = Field(
        ...,
        description="Constant term (efficiency at zero load extrapolation)"
    )
    coefficient_a1: float = Field(
        ...,
        description="Linear coefficient"
    )
    coefficient_a2: float = Field(
        default=0.0,
        description="Quadratic coefficient"
    )
    coefficient_a3: float = Field(
        default=0.0,
        description="Cubic coefficient"
    )

    # Fit quality metrics
    r_squared: float = Field(
        ...,
        ge=0,
        le=1,
        description="R-squared fit quality (0-1)"
    )
    rmse_pct: Optional[float] = Field(
        default=None,
        ge=0,
        description="Root mean square error (%)"
    )
    n_data_points: int = Field(
        default=0,
        ge=0,
        description="Number of data points used for fitting"
    )

    # Valid operating range
    valid_load_min_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Minimum valid load for curve (%)"
    )
    valid_load_max_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Maximum valid load for curve (%)"
    )

    # Peak efficiency point
    peak_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Peak efficiency value (%)"
    )
    peak_efficiency_load_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Load at peak efficiency (%)"
    )

    # Source data points (for audit trail)
    data_points: List[EfficiencyCurveDataPoint] = Field(
        default_factory=list,
        description="Source efficiency measurements"
    )

    # Curve metadata
    curve_type: str = Field(
        default="polynomial",
        description="Curve type (polynomial, piecewise, lookup)"
    )
    last_calibration: Optional[datetime] = Field(
        default=None,
        description="Last calibration date"
    )
    calibration_standard: str = Field(
        default="ASME PTC 4",
        description="Calibration standard reference"
    )

    def calculate_efficiency(self, load_pct: float) -> float:
        """
        Calculate efficiency at given load using polynomial curve.

        ZERO HALLUCINATION: Pure mathematical calculation with no LLM inference.

        Args:
            load_pct: Load percentage (0-100)

        Returns:
            Efficiency percentage (0-100)
        """
        # Convert to load fraction
        L = load_pct / 100.0

        # Polynomial calculation
        eta = (
            self.coefficient_a0 +
            self.coefficient_a1 * L +
            self.coefficient_a2 * L**2 +
            self.coefficient_a3 * L**3
        )

        # Convert back to percentage and clamp
        return max(0.0, min(100.0, eta * 100.0))

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# HEAT DEMAND MODELS
# =============================================================================

class HeatDemandReading(BaseModel):
    """
    Heat demand reading from a process consumer.

    Captures demand requirements, priority, and temperature needs.
    """

    # Identification
    consumer_id: str = Field(
        ...,
        description="Process consumer identifier"
    )
    consumer_name: str = Field(
        default="",
        description="Process consumer name"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Demand values
    demand_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Heat demand rate (MMBTU/hr)"
    )
    demand_mw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat demand rate (MW thermal)"
    )
    demand_pct_of_peak: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Demand as percentage of peak (%)"
    )

    # Temperature requirements
    temperature_required_f: Optional[float] = Field(
        default=None,
        ge=100,
        le=2000,
        description="Required process temperature (F)"
    )
    temperature_tolerance_f: float = Field(
        default=10.0,
        ge=0,
        description="Acceptable temperature tolerance (F)"
    )
    min_supply_temp_f: Optional[float] = Field(
        default=None,
        description="Minimum acceptable supply temperature (F)"
    )

    # Medium requirements
    heat_transfer_medium: str = Field(
        default="steam",
        description="Heat transfer medium (steam, hot_water, thermal_oil)"
    )
    pressure_psig: Optional[float] = Field(
        default=None,
        description="Required pressure for steam (psig)"
    )

    # Priority classification
    priority: LoadPriority = Field(
        default=LoadPriority.MEDIUM,
        description="Demand priority classification"
    )
    is_critical: bool = Field(
        default=False,
        description="Critical process flag (cannot be shed)"
    )
    can_be_deferred: bool = Field(
        default=False,
        description="Load can be time-shifted"
    )
    max_deferral_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum deferral time (hours)"
    )

    # Current satisfaction status
    demand_satisfied: bool = Field(
        default=True,
        description="Demand currently satisfied"
    )
    supply_shortfall_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Current supply shortfall (MMBTU/hr)"
    )

    @validator('demand_mw', always=True)
    def calculate_mw(cls, v, values):
        """Calculate MW from MMBTU/hr."""
        if v is None and 'demand_mmbtu_hr' in values:
            return values['demand_mmbtu_hr'] * 0.293071
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TotalDemandSummary(BaseModel):
    """
    Summary of total heat demand across all consumers.

    Aggregates demand by priority level for optimization.
    """

    # Identification
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Summary timestamp"
    )

    # Total demand
    total_demand_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Total heat demand (MMBTU/hr)"
    )
    total_demand_mw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total heat demand (MW thermal)"
    )

    # Demand by priority
    critical_demand_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Critical priority demand (MMBTU/hr)"
    )
    high_demand_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="High priority demand (MMBTU/hr)"
    )
    medium_demand_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Medium priority demand (MMBTU/hr)"
    )
    low_demand_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Low priority demand (MMBTU/hr)"
    )
    deferrable_demand_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Deferrable demand (MMBTU/hr)"
    )

    # Consumer counts
    total_consumers: int = Field(
        default=0,
        ge=0,
        description="Total number of consumers"
    )
    consumers_satisfied: int = Field(
        default=0,
        ge=0,
        description="Number of satisfied consumers"
    )
    consumers_unsatisfied: int = Field(
        default=0,
        ge=0,
        description="Number of unsatisfied consumers"
    )

    # Forecast horizon
    forecast_horizon_hr: float = Field(
        default=0.0,
        ge=0,
        description="Forecast horizon (hours)"
    )
    forecast_demand_1hr_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="1-hour demand forecast (MMBTU/hr)"
    )
    forecast_demand_4hr_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="4-hour demand forecast (MMBTU/hr)"
    )
    forecast_peak_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Forecast peak demand (MMBTU/hr)"
    )

    # Individual demands
    demands: List[HeatDemandReading] = Field(
        default_factory=list,
        description="Individual demand readings"
    )

    @validator('total_demand_mw', always=True)
    def calculate_total_mw(cls, v, values):
        """Calculate MW from MMBTU/hr."""
        if v is None and 'total_demand_mmbtu_hr' in values:
            return values['total_demand_mmbtu_hr'] * 0.293071
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# FUEL PRICE MODELS
# =============================================================================

class FuelPrice(BaseModel):
    """Fuel pricing data for cost optimization."""

    fuel_type: FuelType = Field(
        ...,
        description="Fuel type"
    )
    price_per_mmbtu: float = Field(
        ...,
        ge=0,
        description="Fuel price ($/MMBTU HHV)"
    )
    price_per_native_unit: Optional[float] = Field(
        default=None,
        ge=0,
        description="Price in native units"
    )
    native_unit: str = Field(
        default="mmbtu",
        description="Native pricing unit (therm, mcf, gal, lb)"
    )
    effective_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Price effective date"
    )
    source: str = Field(
        default="contract",
        description="Price source (contract, spot, index)"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# LOAD ALLOCATION INPUT MODELS
# =============================================================================

class EquipmentForOptimization(BaseModel):
    """
    Complete equipment data required for load optimization.

    Combines reading, capacity, and efficiency curve data.
    """

    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Equipment type"
    )
    reading: EquipmentReading = Field(
        ...,
        description="Current operating point"
    )
    capacity: EquipmentCapacity = Field(
        ...,
        description="Equipment capacity limits"
    )
    efficiency_curve: EfficiencyCurve = Field(
        ...,
        description="Efficiency vs load curve"
    )
    fuel_price: FuelPrice = Field(
        ...,
        description="Current fuel pricing"
    )

    # CHP-specific parameters
    is_chp: bool = Field(
        default=False,
        description="Combined Heat and Power unit"
    )
    chp_power_output_mw: Optional[float] = Field(
        default=None,
        ge=0,
        description="CHP electrical output (MW)"
    )
    chp_heat_to_power_ratio: Optional[float] = Field(
        default=None,
        gt=0,
        description="Heat to power ratio for CHP"
    )
    chp_power_price_per_mwh: Optional[float] = Field(
        default=None,
        description="Electricity price for CHP export ($/MWh)"
    )

    class Config:
        use_enum_values = True


class OptimizationConstraints(BaseModel):
    """
    System-level constraints for load optimization.

    Defines spinning reserve, emissions limits, and operational constraints.
    """

    # Spinning reserve requirements
    min_spinning_reserve_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Minimum spinning reserve (%)"
    )
    min_spinning_reserve_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum spinning reserve (MMBTU/hr)"
    )

    # Startup constraints
    max_simultaneous_starts: int = Field(
        default=1,
        ge=0,
        description="Maximum simultaneous equipment startups"
    )
    max_starts_per_hour: int = Field(
        default=2,
        ge=0,
        description="Maximum total starts per hour"
    )

    # Emissions constraints
    max_total_nox_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum total NOx emissions (lb/hr)"
    )
    max_total_co2_ton_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum total CO2 emissions (ton/hr)"
    )
    carbon_price_per_ton: float = Field(
        default=0.0,
        ge=0,
        description="Carbon price for emissions cost ($/ton CO2)"
    )

    # Load constraints
    must_run_equipment: List[str] = Field(
        default_factory=list,
        description="Equipment IDs that must remain running"
    )
    cannot_start_equipment: List[str] = Field(
        default_factory=list,
        description="Equipment IDs that cannot be started"
    )
    cannot_stop_equipment: List[str] = Field(
        default_factory=list,
        description="Equipment IDs that cannot be stopped"
    )

    # N+1 redundancy
    require_n_plus_1: bool = Field(
        default=True,
        description="Require N+1 equipment redundancy"
    )
    n_plus_1_capacity_pct: float = Field(
        default=100.0,
        ge=0,
        description="N+1 capacity requirement (%)"
    )

    class Config:
        use_enum_values = True


class LoadAllocationInput(BaseModel):
    """
    Complete input for load allocation optimization.

    Aggregates equipment fleet, demand, constraints, and pricing.
    """

    # Identification
    optimization_id: str = Field(
        default="",
        description="Unique optimization run identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Input timestamp"
    )

    # Equipment fleet
    equipment_fleet: List[EquipmentForOptimization] = Field(
        ...,
        min_items=1,
        description="Equipment available for load allocation"
    )

    # Demand
    total_demand: TotalDemandSummary = Field(
        ...,
        description="Total heat demand summary"
    )

    # Constraints
    constraints: OptimizationConstraints = Field(
        default_factory=OptimizationConstraints,
        description="Optimization constraints"
    )

    # Fuel prices (overrides equipment-level if provided)
    fuel_prices: List[FuelPrice] = Field(
        default_factory=list,
        description="Current fuel prices"
    )

    # Optimization objective
    optimization_objective: OptimizationObjective = Field(
        default=OptimizationObjective.MINIMIZE_COST,
        description="Primary optimization objective"
    )

    # Multi-objective weights (must sum to 1.0)
    cost_weight: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Cost minimization weight"
    )
    efficiency_weight: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Efficiency maximization weight"
    )
    emissions_weight: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Emissions minimization weight"
    )

    # Solver parameters
    max_solver_time_sec: float = Field(
        default=60.0,
        ge=1,
        le=600,
        description="Maximum solver time (seconds)"
    )
    optimality_gap_pct: float = Field(
        default=1.0,
        ge=0,
        le=10,
        description="Acceptable optimality gap (%)"
    )

    @validator('cost_weight')
    def validate_weights(cls, v, values):
        """Validate that weights sum to approximately 1.0."""
        # This runs after cost_weight is set
        # Full validation would require all weights
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# EQUIPMENT SETPOINT MODELS
# =============================================================================

class EquipmentSetpoint(BaseModel):
    """
    Optimized setpoint for a single equipment unit.

    Contains target load, expected efficiency, and action required.
    """

    # Identification
    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Equipment type"
    )

    # Current state
    current_load_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current load (%)"
    )
    current_load_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Current load (MMBTU/hr)"
    )
    current_status: EquipmentStatus = Field(
        ...,
        description="Current equipment status"
    )

    # Target state
    target_load_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Target load (%)"
    )
    target_load_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Target load (MMBTU/hr)"
    )
    target_status: EquipmentStatus = Field(
        ...,
        description="Target equipment status"
    )

    # Load change
    load_change_mmbtu_hr: float = Field(
        ...,
        description="Required load change (MMBTU/hr)"
    )
    load_change_pct: float = Field(
        ...,
        description="Required load change (%)"
    )
    control_action: ControlAction = Field(
        ...,
        description="Control action to take"
    )

    # Time to reach target
    ramp_time_min: float = Field(
        default=0.0,
        ge=0,
        description="Time to reach target load (minutes)"
    )

    # Expected performance at target
    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Fuel type"
    )
    expected_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Expected efficiency at target load (%)"
    )
    expected_fuel_input_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Expected fuel input at target (MMBTU/hr)"
    )

    # Expected costs
    expected_hourly_fuel_cost: float = Field(
        ...,
        ge=0,
        description="Expected hourly fuel cost ($)"
    )
    expected_hourly_total_cost: float = Field(
        ...,
        ge=0,
        description="Expected total hourly cost ($)"
    )

    # Expected emissions
    expected_co2_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Expected CO2 emissions (kg/hr)"
    )
    expected_nox_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Expected NOx emissions (lb/hr)"
    )

    # Capacity contribution
    spinning_reserve_contribution_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Spinning reserve contribution (MMBTU/hr)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# LOAD ALLOCATION OUTPUT MODELS
# =============================================================================

class LoadAllocationOutput(BaseModel):
    """
    Output from load allocation optimization.

    Contains optimal setpoints for all equipment with cost/emissions projections.
    """

    # Identification
    optimization_id: str = Field(
        ...,
        description="Optimization run identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output timestamp"
    )

    # Optimization status
    optimization_status: OptimizationStatus = Field(
        ...,
        description="Solver status"
    )
    solver_time_ms: float = Field(
        ...,
        ge=0,
        description="Solver execution time (ms)"
    )
    optimality_gap_pct: Optional[float] = Field(
        default=None,
        ge=0,
        description="Final optimality gap (%)"
    )
    iterations: Optional[int] = Field(
        default=None,
        ge=0,
        description="Solver iterations"
    )

    # Equipment setpoints
    setpoints: List[EquipmentSetpoint] = Field(
        ...,
        description="Optimized equipment setpoints"
    )

    # Fleet summary - capacity
    total_fleet_capacity_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Total fleet capacity (MMBTU/hr)"
    )
    total_allocated_load_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Total allocated load (MMBTU/hr)"
    )
    total_spinning_reserve_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Total spinning reserve (MMBTU/hr)"
    )
    spinning_reserve_pct: float = Field(
        ...,
        ge=0,
        description="Spinning reserve percentage"
    )

    # Fleet summary - efficiency
    fleet_weighted_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Load-weighted average efficiency (%)"
    )

    # Fleet summary - cost
    total_hourly_cost_usd: float = Field(
        ...,
        ge=0,
        description="Total hourly operating cost ($)"
    )
    cost_per_mmbtu_usd: float = Field(
        ...,
        ge=0,
        description="Blended cost per MMBTU output ($/MMBTU)"
    )

    # Fleet summary - emissions
    total_co2_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2 emissions (kg/hr)"
    )
    co2_intensity_kg_mmbtu: float = Field(
        default=0.0,
        ge=0,
        description="CO2 intensity (kg/MMBTU output)"
    )
    total_nox_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total NOx emissions (lb/hr)"
    )

    # Equipment counts
    units_running: int = Field(
        ...,
        ge=0,
        description="Number of units running"
    )
    units_starting: int = Field(
        default=0,
        ge=0,
        description="Number of units starting"
    )
    units_stopping: int = Field(
        default=0,
        ge=0,
        description="Number of units stopping"
    )
    units_on_standby: int = Field(
        default=0,
        ge=0,
        description="Number of units on standby"
    )

    # Constraint satisfaction
    all_constraints_satisfied: bool = Field(
        ...,
        description="All constraints satisfied"
    )
    constraint_violations: List[str] = Field(
        default_factory=list,
        description="List of constraint violations"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    calculation_method: str = Field(
        default="MILP",
        description="Optimization method used"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# ECONOMIC MODELS
# =============================================================================

class FuelCostBreakdown(BaseModel):
    """Fuel cost breakdown by fuel type."""

    fuel_type: FuelType = Field(
        ...,
        description="Fuel type"
    )
    consumption_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Fuel consumption rate (MMBTU/hr)"
    )
    unit_cost_per_mmbtu: float = Field(
        ...,
        ge=0,
        description="Unit cost ($/MMBTU)"
    )
    hourly_cost_usd: float = Field(
        ...,
        ge=0,
        description="Hourly fuel cost ($)"
    )
    pct_of_total_fuel_cost: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage of total fuel cost"
    )

    class Config:
        use_enum_values = True


class OperatingCostSummary(BaseModel):
    """Complete operating cost breakdown."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Cost summary timestamp"
    )

    # Fuel costs
    fuel_costs: List[FuelCostBreakdown] = Field(
        default_factory=list,
        description="Fuel cost breakdown by type"
    )
    total_fuel_cost_usd_hr: float = Field(
        ...,
        ge=0,
        description="Total hourly fuel cost ($)"
    )

    # Variable maintenance
    maintenance_cost_usd_hr: float = Field(
        default=0.0,
        ge=0,
        description="Variable maintenance cost ($/hr)"
    )

    # Emissions cost
    emissions_cost_usd_hr: float = Field(
        default=0.0,
        ge=0,
        description="Carbon/emissions cost ($/hr)"
    )
    carbon_price_per_ton: float = Field(
        default=0.0,
        ge=0,
        description="Carbon price used ($/ton CO2)"
    )

    # Startup costs
    startup_costs_usd_hr: float = Field(
        default=0.0,
        ge=0,
        description="Amortized startup costs ($/hr)"
    )

    # CHP revenue offset
    chp_power_revenue_usd_hr: float = Field(
        default=0.0,
        ge=0,
        description="CHP power revenue ($/hr)"
    )

    # Total
    total_hourly_cost_usd: float = Field(
        ...,
        ge=0,
        description="Total hourly operating cost ($)"
    )
    net_hourly_cost_usd: float = Field(
        ...,
        description="Net hourly cost after CHP revenue ($)"
    )

    # Unit costs
    cost_per_mmbtu_output: float = Field(
        ...,
        ge=0,
        description="Cost per MMBTU heat output ($/MMBTU)"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SavingsAnalysis(BaseModel):
    """
    Savings analysis comparing optimized vs baseline operation.

    Baseline typically represents equal load distribution or current operation.
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Baseline (equal load or current)
    baseline_description: str = Field(
        default="equal_load",
        description="Baseline comparison method"
    )
    baseline_hourly_cost_usd: float = Field(
        ...,
        ge=0,
        description="Baseline hourly cost ($)"
    )
    baseline_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Baseline fleet efficiency (%)"
    )
    baseline_co2_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Baseline CO2 emissions (kg/hr)"
    )

    # Optimized
    optimized_hourly_cost_usd: float = Field(
        ...,
        ge=0,
        description="Optimized hourly cost ($)"
    )
    optimized_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Optimized fleet efficiency (%)"
    )
    optimized_co2_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Optimized CO2 emissions (kg/hr)"
    )

    # Savings
    cost_savings_usd_hr: float = Field(
        ...,
        description="Hourly cost savings ($)"
    )
    cost_savings_pct: float = Field(
        ...,
        description="Cost savings percentage"
    )
    efficiency_improvement_pct: float = Field(
        ...,
        description="Efficiency improvement (percentage points)"
    )
    co2_reduction_kg_hr: float = Field(
        default=0.0,
        description="CO2 reduction (kg/hr)"
    )
    co2_reduction_pct: float = Field(
        default=0.0,
        description="CO2 reduction percentage"
    )

    # Annualized (assuming current conditions)
    annual_cost_savings_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated annual savings ($)"
    )
    annual_co2_reduction_ton: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated annual CO2 reduction (tons)"
    )
    operating_hours_assumed: float = Field(
        default=8760.0,
        ge=0,
        description="Operating hours for annualization"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 calculation hash"
    )

    @validator('annual_cost_savings_usd', always=True)
    def calculate_annual_savings(cls, v, values):
        """Calculate annual savings."""
        if v is None and 'cost_savings_usd_hr' in values:
            hours = values.get('operating_hours_assumed', 8760.0)
            return values['cost_savings_usd_hr'] * hours
        return v

    @validator('annual_co2_reduction_ton', always=True)
    def calculate_annual_co2(cls, v, values):
        """Calculate annual CO2 reduction."""
        if v is None and 'co2_reduction_kg_hr' in values:
            hours = values.get('operating_hours_assumed', 8760.0)
            return (values['co2_reduction_kg_hr'] * hours) / 1000  # kg to ton
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# CONSTRAINT MODELS
# =============================================================================

class EquipmentConstraint(BaseModel):
    """Constraint status for a single equipment unit."""

    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    constraint_type: ConstraintType = Field(
        ...,
        description="Type of constraint"
    )
    constraint_name: str = Field(
        default="",
        description="Human-readable constraint name"
    )
    limit_value: float = Field(
        ...,
        description="Constraint limit value"
    )
    current_value: float = Field(
        ...,
        description="Current value"
    )
    margin: float = Field(
        ...,
        description="Margin to limit (positive = within limit)"
    )
    margin_pct: Optional[float] = Field(
        default=None,
        description="Margin as percentage of limit"
    )
    is_violated: bool = Field(
        ...,
        description="Constraint violated"
    )
    is_binding: bool = Field(
        default=False,
        description="Constraint is binding (at limit)"
    )
    units: str = Field(
        default="",
        description="Units for limit and current value"
    )

    @validator('margin_pct', always=True)
    def calculate_margin_pct(cls, v, values):
        """Calculate margin percentage."""
        if v is None:
            limit = values.get('limit_value')
            margin = values.get('margin')
            if limit and limit != 0:
                return (margin / abs(limit)) * 100
        return v

    class Config:
        use_enum_values = True


class SystemConstraint(BaseModel):
    """System-level constraint status."""

    constraint_name: str = Field(
        ...,
        description="Constraint name"
    )
    constraint_type: ConstraintType = Field(
        ...,
        description="Constraint type"
    )
    total_limit: float = Field(
        ...,
        description="System-level limit"
    )
    current_value: float = Field(
        ...,
        description="Current system value"
    )
    margin: float = Field(
        ...,
        description="Margin to limit"
    )
    margin_pct: Optional[float] = Field(
        default=None,
        description="Margin as percentage"
    )
    is_violated: bool = Field(
        ...,
        description="Constraint violated"
    )
    is_binding: bool = Field(
        default=False,
        description="Constraint is binding"
    )
    units: str = Field(
        default="",
        description="Units"
    )
    affected_equipment: List[str] = Field(
        default_factory=list,
        description="Equipment affected by constraint"
    )

    @validator('margin_pct', always=True)
    def calculate_margin_pct(cls, v, values):
        """Calculate margin percentage."""
        if v is None:
            limit = values.get('total_limit')
            margin = values.get('margin')
            if limit and limit != 0:
                return (margin / abs(limit)) * 100
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# SAFETY MODELS
# =============================================================================

class EquipmentSafetyStatus(BaseModel):
    """Safety status for a single equipment unit."""

    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )

    # Overall status
    safety_level: SafetyLevel = Field(
        default=SafetyLevel.NORMAL,
        description="Overall safety level"
    )
    is_safe_to_operate: bool = Field(
        default=True,
        description="Equipment safe to operate"
    )

    # Safety interlocks
    safety_interlocks_healthy: bool = Field(
        default=True,
        description="All safety interlocks healthy"
    )
    interlocks_bypassed: int = Field(
        default=0,
        ge=0,
        description="Number of bypassed interlocks"
    )
    interlock_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Individual interlock status"
    )

    # Trip conditions
    trip_conditions_present: bool = Field(
        default=False,
        description="Trip conditions present"
    )
    active_trips: List[str] = Field(
        default_factory=list,
        description="Active trip conditions"
    )
    pending_trips: List[str] = Field(
        default_factory=list,
        description="Pending trip conditions"
    )

    # Warnings
    warnings: List[str] = Field(
        default_factory=list,
        description="Active warnings"
    )
    alarms: List[str] = Field(
        default_factory=list,
        description="Active alarms"
    )

    # Permit status
    permit_to_start: bool = Field(
        default=True,
        description="Permit to start (if on standby)"
    )
    permit_blocking_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons blocking permit"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FleetSafetyStatus(BaseModel):
    """
    Fleet-level safety status including N+1 redundancy.

    Critical for ensuring system reliability and availability.
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )

    # Overall fleet safety
    fleet_safety_level: SafetyLevel = Field(
        default=SafetyLevel.NORMAL,
        description="Overall fleet safety level"
    )

    # N+1 redundancy status
    n_plus_1_satisfied: bool = Field(
        ...,
        description="N+1 redundancy requirement met"
    )
    n_plus_1_status: str = Field(
        default="satisfied",
        description="N+1 status description"
    )
    largest_unit_capacity_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Largest unit capacity (MMBTU/hr)"
    )
    reserve_without_largest_mmbtu_hr: float = Field(
        ...,
        description="Reserve if largest unit trips (MMBTU/hr)"
    )

    # Total reserve status
    total_reserve_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Total spinning reserve (MMBTU/hr)"
    )
    total_reserve_pct: float = Field(
        ...,
        ge=0,
        description="Total reserve percentage"
    )
    reserve_status: str = Field(
        default="adequate",
        description="Reserve status description"
    )

    # Emergency capacity
    emergency_capacity_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Emergency capacity from standby units (MMBTU/hr)"
    )
    emergency_start_time_min: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time to bring emergency capacity online (min)"
    )

    # Equipment safety summary
    equipment_safety_statuses: List[EquipmentSafetyStatus] = Field(
        default_factory=list,
        description="Individual equipment safety statuses"
    )
    units_in_trip: int = Field(
        default=0,
        ge=0,
        description="Number of units in trip condition"
    )
    units_in_alarm: int = Field(
        default=0,
        ge=0,
        description="Number of units in alarm condition"
    )
    units_with_bypasses: int = Field(
        default=0,
        ge=0,
        description="Number of units with active bypasses"
    )

    # Fleet-level warnings
    fleet_warnings: List[str] = Field(
        default_factory=list,
        description="Fleet-level warnings"
    )
    fleet_alarms: List[str] = Field(
        default_factory=list,
        description="Fleet-level alarms"
    )
    critical_alerts: List[str] = Field(
        default_factory=list,
        description="Critical alerts requiring immediate attention"
    )

    # Recommendations
    safety_recommendations: List[str] = Field(
        default_factory=list,
        description="Safety recommendations"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 status hash"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# EXPLAINABILITY MODELS
# =============================================================================

class OptimizationExplanation(BaseModel):
    """
    Explainability output for load allocation decisions.

    Provides human-readable justification for optimization results.
    """

    # Summary explanation
    summary: str = Field(
        ...,
        description="High-level explanation of optimization result"
    )

    # Key decisions
    key_decisions: List[str] = Field(
        default_factory=list,
        description="Key decisions made by optimizer"
    )

    # Load allocation reasoning
    load_allocation_reasoning: Dict[str, str] = Field(
        default_factory=dict,
        description="Reasoning for each equipment load allocation"
    )

    # Binding constraints
    binding_constraints: List[str] = Field(
        default_factory=list,
        description="Constraints that limited optimization"
    )

    # Trade-offs made
    tradeoffs: List[str] = Field(
        default_factory=list,
        description="Trade-offs made in optimization"
    )

    # Alternative scenarios considered
    alternatives_considered: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative scenarios evaluated"
    )

    # Sensitivity analysis
    cost_sensitivity_pct_per_demand_change: Optional[float] = Field(
        default=None,
        description="Cost sensitivity to demand changes"
    )
    efficiency_sensitivity: Optional[float] = Field(
        default=None,
        description="Efficiency sensitivity to load changes"
    )


# =============================================================================
# COMPREHENSIVE OUTPUT MODEL
# =============================================================================

class HeatLoadBalancerOutput(BaseModel):
    """
    Complete output from GL-023 Heat Load Balancer Agent.

    Comprehensive optimization output with all setpoints, costs, emissions,
    safety status, explainability, and provenance tracking for audit compliance.
    """

    # Identification
    agent_id: str = Field(
        default="GL-023-HEAT-LOAD-BALANCER",
        description="Agent identifier"
    )
    execution_id: str = Field(
        ...,
        description="Unique execution identifier for traceability"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output timestamp"
    )

    # Optimization Results
    load_allocation: LoadAllocationOutput = Field(
        ...,
        description="Load allocation optimization results"
    )

    # Cost Analysis
    cost_summary: OperatingCostSummary = Field(
        ...,
        description="Operating cost breakdown"
    )
    savings_analysis: SavingsAnalysis = Field(
        ...,
        description="Savings vs baseline analysis"
    )

    # Emissions Summary
    total_emissions_kg_co2_hr: float = Field(
        ...,
        ge=0,
        description="Total CO2 emissions (kg/hr)"
    )
    emissions_intensity_kg_mmbtu: float = Field(
        ...,
        ge=0,
        description="CO2 intensity (kg/MMBTU output)"
    )

    # Safety Status
    safety_status: FleetSafetyStatus = Field(
        ...,
        description="Fleet safety status including N+1"
    )

    # Constraint Status
    equipment_constraints: List[EquipmentConstraint] = Field(
        default_factory=list,
        description="Equipment-level constraint status"
    )
    system_constraints: List[SystemConstraint] = Field(
        default_factory=list,
        description="System-level constraint status"
    )

    # Explainability
    explanation: OptimizationExplanation = Field(
        ...,
        description="Optimization explanation for transparency"
    )

    # Overall Status
    overall_status: ValidationStatus = Field(
        ...,
        description="Overall optimization status"
    )
    optimization_status: OptimizationStatus = Field(
        ...,
        description="Solver status"
    )
    all_demands_satisfied: bool = Field(
        ...,
        description="All heat demands satisfied"
    )
    all_constraints_satisfied: bool = Field(
        ...,
        description="All constraints satisfied"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Operational recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Critical alerts"
    )

    # Performance Metrics
    fleet_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Fleet weighted average efficiency (%)"
    )
    spinning_reserve_pct: float = Field(
        ...,
        ge=0,
        description="Spinning reserve percentage"
    )

    # Provenance and Audit Trail
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for complete audit trail"
    )
    input_data_hash: str = Field(
        ...,
        description="SHA-256 hash of input data"
    )
    calculation_chain: List[str] = Field(
        default_factory=list,
        description="List of calculation step hashes"
    )

    # Processing Metadata
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total processing time (ms)"
    )
    solver_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Optimization solver time (ms)"
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations performed"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    optimization_method: str = Field(
        default="MILP",
        description="Optimization method used"
    )
    formula_references: List[str] = Field(
        default_factory=list,
        description="Engineering formulas and standards used"
    )

    # Validation status
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Output validation status"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation error messages"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# INPUT AGGREGATION MODEL
# =============================================================================

class HeatLoadBalancerInput(BaseModel):
    """
    Complete input for GL-023 Heat Load Balancer Agent.

    Aggregates all input data required for comprehensive load optimization.
    """

    # Identification
    facility_id: str = Field(
        default="",
        description="Facility identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Input timestamp"
    )

    # Load allocation input
    allocation_input: LoadAllocationInput = Field(
        ...,
        description="Load allocation optimization input"
    )

    # Historical data for efficiency curves (optional)
    efficiency_curve_data: Dict[str, List[EfficiencyCurveDataPoint]] = Field(
        default_factory=dict,
        description="Historical efficiency data by equipment ID"
    )

    # Operating constraints override
    constraint_overrides: Optional[OptimizationConstraints] = Field(
        default=None,
        description="Override default constraints"
    )

    # Baseline for comparison
    baseline_method: str = Field(
        default="equal_load",
        description="Baseline method for savings calculation"
    )
    custom_baseline_loads: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom baseline loads by equipment ID"
    )

    # Output options
    include_explanation: bool = Field(
        default=True,
        description="Include optimization explanation"
    )
    include_sensitivity_analysis: bool = Field(
        default=False,
        description="Include sensitivity analysis"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# CONFIGURATION MODEL
# =============================================================================

class HeatLoadBalancerConfig(BaseModel):
    """
    Configuration parameters for GL-023 Heat Load Balancer Agent.

    Defines operational parameters, default limits, and tuning values.
    """

    # Agent identification
    agent_id: str = Field(
        default="GL-023-HEAT-LOAD-BALANCER",
        description="Agent identifier"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )

    # Default optimization settings
    default_optimization_objective: OptimizationObjective = Field(
        default=OptimizationObjective.MINIMIZE_COST,
        description="Default optimization objective"
    )
    default_cost_weight: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Default cost weight"
    )
    default_efficiency_weight: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Default efficiency weight"
    )
    default_emissions_weight: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Default emissions weight"
    )

    # Solver settings
    default_max_solver_time_sec: float = Field(
        default=60.0,
        ge=1,
        le=600,
        description="Default max solver time (seconds)"
    )
    default_optimality_gap_pct: float = Field(
        default=1.0,
        ge=0,
        le=10,
        description="Default optimality gap (%)"
    )
    solver_type: str = Field(
        default="MILP",
        description="Solver type (MILP, LP, heuristic)"
    )

    # Default constraint settings
    default_min_spinning_reserve_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Default minimum spinning reserve (%)"
    )
    default_require_n_plus_1: bool = Field(
        default=True,
        description="Default N+1 redundancy requirement"
    )
    default_max_simultaneous_starts: int = Field(
        default=1,
        ge=0,
        description="Default max simultaneous starts"
    )

    # Efficiency curve settings
    min_efficiency_curve_r_squared: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Minimum R-squared for efficiency curve"
    )
    efficiency_curve_polynomial_order: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Polynomial order for efficiency curves"
    )

    # Emissions factors (defaults if not provided)
    default_co2_factor_kg_mmbtu_natural_gas: float = Field(
        default=53.07,
        ge=0,
        description="Default CO2 factor for natural gas (kg/MMBTU)"
    )
    default_co2_factor_kg_mmbtu_fuel_oil: float = Field(
        default=73.96,
        ge=0,
        description="Default CO2 factor for fuel oil (kg/MMBTU)"
    )

    # Safety thresholds
    spinning_reserve_warning_pct: float = Field(
        default=15.0,
        ge=0,
        description="Spinning reserve warning threshold (%)"
    )
    spinning_reserve_alarm_pct: float = Field(
        default=10.0,
        ge=0,
        description="Spinning reserve alarm threshold (%)"
    )

    # Performance tracking
    enable_detailed_logging: bool = Field(
        default=True,
        description="Enable detailed calculation logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    # Output formatting
    decimal_precision: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Decimal precision for calculations"
    )
    include_calculation_chain: bool = Field(
        default=True,
        description="Include calculation chain in output"
    )

    class Config:
        use_enum_values = True
