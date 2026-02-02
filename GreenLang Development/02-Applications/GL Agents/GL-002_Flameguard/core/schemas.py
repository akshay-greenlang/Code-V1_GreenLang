"""
GL-002 FLAMEGUARD BoilerEfficiencyOptimizer - Schema Definitions

This module defines all Pydantic models for inputs, outputs, process data,
optimization results, and status reporting for the FLAMEGUARD agent.

All schemas support zero-hallucination principles with deterministic
calculations, SHA-256 provenance tracking, and regulatory compliance.
"""

from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from pydantic import BaseModel, Field, validator

from .config import (
    OperatingState,
    FuelType,
    BoilerType,
    CombustionMode,
    OptimizationObjective,
)


# =============================================================================
# ENUMS
# =============================================================================

class OptimizationStatus(Enum):
    """Optimization execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class CalculationType(Enum):
    """Types of calculations performed."""
    EFFICIENCY = "efficiency"
    EMISSIONS = "emissions"
    HEAT_BALANCE = "heat_balance"
    COMBUSTION = "combustion"
    FUEL_BLEND = "fuel_blend"
    COST = "cost"
    O2_TRIM = "o2_trim"
    EXCESS_AIR = "excess_air"


class SeverityLevel(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlarmState(Enum):
    """Alarm states."""
    NORMAL = "normal"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"
    SHELVED = "shelved"


class TripType(Enum):
    """Safety trip types."""
    HIGH_STEAM_PRESSURE = "high_steam_pressure"
    LOW_WATER_LEVEL = "low_water_level"
    HIGH_WATER_LEVEL = "high_water_level"
    FLAME_FAILURE = "flame_failure"
    LOW_FUEL_PRESSURE = "low_fuel_pressure"
    HIGH_FUEL_PRESSURE = "high_fuel_pressure"
    LOW_AIR_PRESSURE = "low_air_pressure"
    HIGH_FLUE_GAS_TEMP = "high_flue_gas_temp"
    MANUAL_TRIP = "manual_trip"
    COMMUNICATION_FAILURE = "communication_failure"


# =============================================================================
# PROCESS DATA SCHEMAS
# =============================================================================

class BoilerProcessData(BaseModel):
    """Real-time boiler process data from SCADA/DCS."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )
    boiler_id: str = Field(..., description="Boiler identifier")

    # Operating State
    operating_state: OperatingState = Field(
        default=OperatingState.MODULATING,
        description="Current operating state"
    )
    load_percent: float = Field(
        ...,
        ge=0.0,
        le=110.0,
        description="Current load as percentage of rated capacity"
    )

    # Steam Production
    steam_flow_klb_hr: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Steam flow rate (klb/hr)"
    )
    steam_pressure_psig: float = Field(
        ...,
        ge=0.0,
        le=5000.0,
        description="Steam header pressure (psig)"
    )
    steam_temperature_f: float = Field(
        ...,
        ge=200.0,
        le=1200.0,
        description="Steam temperature (°F)"
    )

    # Water Side
    feedwater_flow_klb_hr: float = Field(
        default=0.0,
        ge=0.0,
        le=10000.0,
        description="Feedwater flow rate (klb/hr)"
    )
    feedwater_temperature_f: float = Field(
        default=227.0,
        ge=60.0,
        le=500.0,
        description="Feedwater temperature (°F)"
    )
    drum_level_inches: float = Field(
        default=0.0,
        ge=-20.0,
        le=20.0,
        description="Drum level (inches from setpoint)"
    )
    blowdown_flow_klb_hr: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Blowdown flow rate (klb/hr)"
    )

    # Fuel Side
    fuel_flow_rate: float = Field(
        ...,
        ge=0.0,
        description="Fuel flow rate (units depend on fuel type)"
    )
    fuel_flow_unit: str = Field(
        default="scfh",
        description="Fuel flow unit (scfh, lb/hr, gal/hr)"
    )
    fuel_pressure_psig: float = Field(
        default=15.0,
        ge=0.0,
        le=500.0,
        description="Fuel supply pressure (psig)"
    )
    fuel_temperature_f: float = Field(
        default=70.0,
        ge=32.0,
        le=500.0,
        description="Fuel temperature (°F)"
    )

    # Combustion Air
    combustion_air_flow_scfm: float = Field(
        default=0.0,
        ge=0.0,
        le=500000.0,
        description="Combustion air flow (SCFM)"
    )
    combustion_air_temperature_f: float = Field(
        default=80.0,
        ge=0.0,
        le=600.0,
        description="Combustion air temperature (°F)"
    )
    combustion_air_humidity_percent: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Combustion air relative humidity (%)"
    )
    forced_draft_fan_speed_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="FD fan speed (%)"
    )

    # Flue Gas
    flue_gas_temperature_f: float = Field(
        ...,
        ge=100.0,
        le=1500.0,
        description="Stack/economizer outlet temperature (°F)"
    )
    flue_gas_o2_percent: float = Field(
        ...,
        ge=0.0,
        le=21.0,
        description="Flue gas O2 concentration (%)"
    )
    flue_gas_co_ppm: float = Field(
        default=0.0,
        ge=0.0,
        le=5000.0,
        description="Flue gas CO concentration (ppm)"
    )
    flue_gas_co2_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        description="Flue gas CO2 concentration (%)"
    )
    flue_gas_nox_ppm: float = Field(
        default=0.0,
        ge=0.0,
        le=1000.0,
        description="Flue gas NOx concentration (ppm)"
    )

    # Control System
    firing_rate_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Firing rate demand (%)"
    )
    air_damper_position_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Air damper position (%)"
    )
    fuel_valve_position_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Fuel control valve position (%)"
    )
    o2_trim_output_percent: float = Field(
        default=0.0,
        ge=-20.0,
        le=20.0,
        description="O2 trim controller output (%)"
    )

    # Flame Status
    flame_signal_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Flame scanner signal (%)"
    )
    flame_status: bool = Field(
        default=True,
        description="Flame present"
    )

    # Ambient Conditions
    ambient_temperature_f: float = Field(
        default=77.0,
        ge=-40.0,
        le=130.0,
        description="Ambient temperature (°F)"
    )
    barometric_pressure_psia: float = Field(
        default=14.696,
        ge=10.0,
        le=20.0,
        description="Barometric pressure (psia)"
    )

    # Data Quality
    data_quality: str = Field(
        default="good",
        description="Data quality indicator (good, suspect, bad)"
    )

    class Config:
        use_enum_values = True


class CombustionAnalysis(BaseModel):
    """Combustion analysis results."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Oxygen Analysis
    measured_o2_percent: float = Field(..., ge=0.0, le=21.0)
    target_o2_percent: float = Field(..., ge=0.0, le=15.0)
    o2_deviation_percent: float = Field(..., ge=-10.0, le=10.0)

    # Excess Air
    excess_air_percent: float = Field(..., ge=0.0, le=500.0)
    stoichiometric_air_lb_lb_fuel: float = Field(..., ge=1.0, le=50.0)
    actual_air_lb_lb_fuel: float = Field(..., ge=1.0, le=100.0)

    # CO Analysis
    measured_co_ppm: float = Field(default=0.0, ge=0.0, le=5000.0)
    co_limit_ppm: float = Field(default=400.0, ge=0.0, le=2000.0)
    co_breakthrough_detected: bool = Field(default=False)

    # Air-Fuel Ratio
    air_fuel_ratio_actual: float = Field(..., ge=1.0, le=100.0)
    air_fuel_ratio_stoichiometric: float = Field(..., ge=1.0, le=50.0)
    lambda_value: float = Field(..., ge=0.5, le=5.0)

    # Combustion Quality
    combustion_efficiency_percent: float = Field(..., ge=80.0, le=100.0)
    combustion_quality_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="0-100 quality score"
    )

    # Recommendations
    recommended_o2_adjustment_percent: float = Field(
        default=0.0,
        ge=-5.0,
        le=5.0
    )
    recommended_air_adjustment_percent: float = Field(
        default=0.0,
        ge=-20.0,
        le=20.0
    )


class EfficiencyCalculation(BaseModel):
    """Boiler efficiency calculation per ASME PTC 4.1."""

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )
    calculation_method: str = Field(
        default="indirect_losses",
        description="Method: direct or indirect_losses"
    )

    # Overall Efficiency
    efficiency_percent: float = Field(
        ...,
        ge=50.0,
        le=100.0,
        description="Overall boiler efficiency (%)"
    )
    efficiency_hhv_basis: float = Field(
        ...,
        ge=50.0,
        le=100.0,
        description="Efficiency on HHV basis (%)"
    )
    efficiency_lhv_basis: float = Field(
        ...,
        ge=50.0,
        le=100.0,
        description="Efficiency on LHV basis (%)"
    )

    # Heat Input
    fuel_input_mmbtu_hr: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Fuel heat input (MMBTU/hr)"
    )
    credit_heat_mmbtu_hr: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Heat credits (sensible heat in air/fuel)"
    )

    # Heat Output
    steam_output_mmbtu_hr: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Steam heat output (MMBTU/hr)"
    )

    # Losses (per ASME PTC 4.1)
    dry_flue_gas_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description="Dry flue gas sensible heat loss (%)"
    )
    moisture_in_fuel_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Moisture in fuel loss (%)"
    )
    hydrogen_combustion_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=15.0,
        description="H2 combustion moisture loss (%)"
    )
    moisture_in_air_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Moisture in combustion air loss (%)"
    )
    unburned_carbon_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Unburned carbon/ash loss (%)"
    )
    co_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="CO incomplete combustion loss (%)"
    )
    radiation_convection_loss_percent: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Surface radiation/convection loss (%)"
    )
    blowdown_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Blowdown heat loss (%)"
    )
    other_losses_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Other miscellaneous losses (%)"
    )
    total_losses_percent: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Total heat losses (%)"
    )

    # Derived Values
    heat_rate_btu_kwh: Optional[float] = Field(
        default=None,
        ge=3000.0,
        le=20000.0,
        description="Heat rate if cogeneration (BTU/kWh)"
    )
    fuel_utilization_percent: float = Field(
        ...,
        ge=50.0,
        le=100.0,
        description="Overall fuel utilization (%)"
    )

    # Uncertainty
    efficiency_uncertainty_percent: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Calculated efficiency uncertainty (%)"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of inputs for reproducibility"
    )

    class Config:
        use_enum_values = True


class EmissionsCalculation(BaseModel):
    """Emissions calculation results."""

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # CO2 Emissions
    co2_lb_hr: float = Field(
        ...,
        ge=0.0,
        le=1000000.0,
        description="CO2 mass emission rate (lb/hr)"
    )
    co2_lb_mmbtu: float = Field(
        ...,
        ge=0.0,
        le=300.0,
        description="CO2 emission factor (lb/MMBTU)"
    )
    co2_kg_mwh: float = Field(
        default=0.0,
        ge=0.0,
        le=1500.0,
        description="CO2 if cogeneration (kg/MWh)"
    )

    # NOx Emissions
    nox_lb_hr: float = Field(
        default=0.0,
        ge=0.0,
        le=1000.0,
        description="NOx mass emission rate (lb/hr)"
    )
    nox_lb_mmbtu: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="NOx emission factor (lb/MMBTU)"
    )
    nox_ppm_corrected_3pct_o2: float = Field(
        default=0.0,
        ge=0.0,
        le=500.0,
        description="NOx corrected to 3% O2 (ppm)"
    )

    # CO Emissions
    co_lb_hr: float = Field(
        default=0.0,
        ge=0.0,
        le=500.0,
        description="CO mass emission rate (lb/hr)"
    )
    co_lb_mmbtu: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="CO emission factor (lb/MMBTU)"
    )

    # SO2 Emissions
    so2_lb_hr: float = Field(
        default=0.0,
        ge=0.0,
        le=10000.0,
        description="SO2 mass emission rate (lb/hr)"
    )
    so2_lb_mmbtu: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="SO2 emission factor (lb/MMBTU)"
    )

    # Particulates
    pm_lb_hr: float = Field(
        default=0.0,
        ge=0.0,
        le=1000.0,
        description="Particulate matter (lb/hr)"
    )
    pm_lb_mmbtu: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="PM emission factor (lb/MMBTU)"
    )

    # Compliance Status
    nox_compliant: bool = Field(default=True)
    co_compliant: bool = Field(default=True)
    so2_compliant: bool = Field(default=True)
    pm_compliant: bool = Field(default=True)
    overall_compliant: bool = Field(default=True)

    # GHG Protocol
    ghg_scope: int = Field(default=1, ge=1, le=3)
    co2e_metric_tons_hr: float = Field(
        default=0.0,
        ge=0.0,
        le=1000.0,
        description="CO2 equivalent (metric tons/hr)"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(default=None)


class FuelBlendOptimization(BaseModel):
    """Fuel blending optimization results."""

    optimization_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Current Blend
    primary_fuel_fraction: float = Field(..., ge=0.0, le=1.0)
    secondary_fuel_fractions: Dict[str, float] = Field(
        default_factory=dict,
        description="Secondary fuel fractions by fuel ID"
    )

    # Blended Properties
    blended_hhv_btu_lb: float = Field(..., ge=1000.0, le=60000.0)
    blended_carbon_content_percent: float = Field(..., ge=0.0, le=100.0)
    blended_sulfur_content_percent: float = Field(default=0.0, ge=0.0, le=10.0)
    blended_stoich_air_fuel_ratio: float = Field(..., ge=1.0, le=50.0)

    # Optimization Results
    optimal_blend: Dict[str, float] = Field(
        default_factory=dict,
        description="Optimal fuel fractions"
    )
    projected_efficiency_gain_percent: float = Field(
        default=0.0,
        ge=-5.0,
        le=10.0
    )
    projected_cost_savings_hr: float = Field(
        default=0.0,
        description="Projected cost savings ($/hr)"
    )
    projected_emissions_reduction_percent: float = Field(
        default=0.0,
        ge=-50.0,
        le=50.0
    )

    # Constraints
    blend_feasible: bool = Field(default=True)
    constraint_violations: List[str] = Field(default_factory=list)


# =============================================================================
# OPTIMIZATION SCHEMAS
# =============================================================================

class OptimizationRequest(BaseModel):
    """Request for boiler optimization."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    boiler_id: str = Field(..., description="Target boiler ID")

    objective: OptimizationObjective = Field(
        default=OptimizationObjective.COMBINED,
        description="Optimization objective"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optimization constraints"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters"
    )

    # Operating Constraints
    min_load_percent: float = Field(default=25.0, ge=10.0, le=50.0)
    max_load_percent: float = Field(default=100.0, ge=50.0, le=110.0)
    target_steam_pressure_psig: Optional[float] = Field(default=None)
    max_emissions_multiplier: float = Field(default=1.0, ge=0.5, le=2.0)

    # Mode
    advisory_only: bool = Field(
        default=True,
        description="True = recommend only, False = auto-implement"
    )

    class Config:
        use_enum_values = True


class OptimizationResult(BaseModel):
    """Result from boiler optimization."""

    request_id: str = Field(..., description="Original request ID")
    optimization_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: OptimizationStatus = Field(
        default=OptimizationStatus.COMPLETED
    )

    # Optimization Objective
    objective_value: float = Field(
        ...,
        description="Objective function value (higher = better)"
    )
    objective_type: OptimizationObjective = Field(
        default=OptimizationObjective.COMBINED
    )

    # Current vs Optimal Comparison
    current_efficiency_percent: float = Field(..., ge=50.0, le=100.0)
    optimal_efficiency_percent: float = Field(..., ge=50.0, le=100.0)
    efficiency_improvement_percent: float = Field(..., ge=-5.0, le=15.0)

    current_emissions_kg_hr: float = Field(default=0.0, ge=0.0)
    optimal_emissions_kg_hr: float = Field(default=0.0, ge=0.0)
    emissions_reduction_percent: float = Field(default=0.0, ge=-50.0, le=50.0)

    current_cost_hr: float = Field(default=0.0, ge=0.0)
    optimal_cost_hr: float = Field(default=0.0, ge=0.0)
    cost_savings_hr: float = Field(default=0.0)

    # Recommended Setpoints
    recommended_setpoints: Dict[str, float] = Field(
        default_factory=dict,
        description="Recommended control setpoints"
    )
    setpoint_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of setpoint changes"
    )

    # Confidence and Uncertainty
    confidence_percent: float = Field(
        default=95.0,
        ge=50.0,
        le=100.0,
        description="Optimization confidence"
    )
    efficiency_uncertainty_bounds: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="95% confidence interval for efficiency"
    )

    # Execution Details
    optimization_time_ms: float = Field(default=0.0, ge=0.0)
    iterations: int = Field(default=0, ge=0)
    solver_used: str = Field(default="ensemble")
    convergence_achieved: bool = Field(default=True)

    # Explanation
    explanation: str = Field(
        default="",
        description="Human-readable optimization explanation"
    )
    key_factors: List[str] = Field(
        default_factory=list,
        description="Key factors influencing optimization"
    )

    # Implementation
    can_auto_implement: bool = Field(default=False)
    implementation_risk: str = Field(default="low")
    operator_approval_required: bool = Field(default=True)

    # Provenance
    provenance_hash: Optional[str] = Field(default=None)
    calculation_trace: List[str] = Field(
        default_factory=list,
        description="Calculation steps for audit"
    )

    class Config:
        use_enum_values = True


class SetpointRecommendation(BaseModel):
    """Individual setpoint recommendation."""

    tag_name: str = Field(..., description="Control tag name")
    current_value: float = Field(..., description="Current setpoint value")
    recommended_value: float = Field(..., description="Recommended value")
    change_amount: float = Field(..., description="Change amount")
    change_percent: float = Field(..., description="Change as percentage")
    unit: str = Field(default="", description="Engineering unit")
    confidence: float = Field(default=95.0, ge=50.0, le=100.0)
    rationale: str = Field(default="", description="Reason for change")
    expected_impact: str = Field(
        default="",
        description="Expected impact description"
    )
    risk_level: str = Field(default="low", description="Risk: low/medium/high")
    requires_approval: bool = Field(default=True)


# =============================================================================
# SAFETY SCHEMAS
# =============================================================================

class SafetyStatus(BaseModel):
    """Boiler safety system status."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    boiler_id: str = Field(...)

    # Overall Status
    safety_state: str = Field(
        default="normal",
        description="normal, warning, alarm, trip"
    )
    bms_status: str = Field(
        default="ready",
        description="BMS status: ready, firing, purge, lockout"
    )

    # Flame Status
    flame_detected: bool = Field(default=True)
    flame_signal_strength: float = Field(default=100.0, ge=0.0, le=100.0)
    flame_scanner_healthy: bool = Field(default=True)

    # Interlocks
    all_permissives_satisfied: bool = Field(default=True)
    active_interlocks: List[str] = Field(default_factory=list)
    bypassed_interlocks: List[str] = Field(default_factory=list)

    # Trips
    trip_active: bool = Field(default=False)
    trip_type: Optional[TripType] = Field(default=None)
    trip_timestamp: Optional[datetime] = Field(default=None)
    trip_description: Optional[str] = Field(default=None)

    # Alarms
    active_alarm_count: int = Field(default=0, ge=0)
    critical_alarm_count: int = Field(default=0, ge=0)
    acknowledged_alarm_count: int = Field(default=0, ge=0)

    # Safety System Health
    safety_plc_healthy: bool = Field(default=True)
    redundancy_status: str = Field(default="normal")
    last_proof_test: Optional[datetime] = Field(default=None)
    days_until_proof_test: Optional[int] = Field(default=None)

    class Config:
        use_enum_values = True


class SafetyEvent(BaseModel):
    """Safety-related event."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    boiler_id: str = Field(...)
    event_type: str = Field(..., description="Event type identifier")
    severity: SeverityLevel = Field(default=SeverityLevel.WARNING)
    description: str = Field(..., description="Event description")

    # Process Context
    source_tag: Optional[str] = Field(default=None)
    measured_value: Optional[float] = Field(default=None)
    threshold_value: Optional[float] = Field(default=None)
    unit: Optional[str] = Field(default=None)

    # Response
    automatic_response: Optional[str] = Field(default=None)
    operator_action_required: bool = Field(default=False)
    escalation_required: bool = Field(default=False)

    # Acknowledgment
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = Field(default=None)
    acknowledged_at: Optional[datetime] = Field(default=None)

    class Config:
        use_enum_values = True


# =============================================================================
# STATUS SCHEMAS
# =============================================================================

class BoilerStatus(BaseModel):
    """Comprehensive boiler status."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    boiler_id: str = Field(...)
    boiler_name: str = Field(default="")

    # Operating Status
    operating_state: OperatingState = Field(
        default=OperatingState.MODULATING
    )
    load_percent: float = Field(default=0.0, ge=0.0, le=110.0)
    hours_since_startup: float = Field(default=0.0, ge=0.0)

    # Performance
    current_efficiency_percent: float = Field(
        default=82.0, ge=50.0, le=100.0
    )
    efficiency_vs_design_percent: float = Field(
        default=0.0, ge=-20.0, le=10.0
    )
    efficiency_trend: str = Field(
        default="stable",
        description="improving, stable, declining"
    )

    # Emissions
    emissions_compliant: bool = Field(default=True)
    emissions_margin_percent: float = Field(default=50.0, ge=-100.0, le=100.0)

    # Safety
    safety_status: SafetyStatus = Field(default_factory=SafetyStatus)

    # Optimization
    optimization_active: bool = Field(default=False)
    last_optimization: Optional[datetime] = Field(default=None)
    optimization_mode: str = Field(default="advisory")
    potential_savings_hr: float = Field(default=0.0)

    # Equipment Health
    equipment_health_score: float = Field(
        default=100.0, ge=0.0, le=100.0
    )
    maintenance_due: bool = Field(default=False)
    next_maintenance: Optional[datetime] = Field(default=None)

    # Integration
    scada_connected: bool = Field(default=True)
    data_quality: str = Field(default="good")
    last_data_update: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Config:
        use_enum_values = True


class AgentStatus(BaseModel):
    """GL-002 FLAMEGUARD agent status."""

    agent_id: str = Field(...)
    agent_name: str = Field(default="FLAMEGUARD")
    agent_version: str = Field(default="1.0.0")
    agent_type: str = Field(default="GL-002")

    # Health
    status: str = Field(default="running")
    health: str = Field(default="healthy")
    uptime_seconds: float = Field(default=0.0, ge=0.0)
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Managed Boilers
    managed_boilers: List[str] = Field(default_factory=list)
    boiler_statuses: Dict[str, BoilerStatus] = Field(default_factory=dict)

    # Performance
    optimizations_performed: int = Field(default=0, ge=0)
    optimizations_successful: int = Field(default=0, ge=0)
    total_efficiency_improvement_percent: float = Field(default=0.0)
    total_emissions_reduction_kg: float = Field(default=0.0)
    total_cost_savings_usd: float = Field(default=0.0)

    # Metrics
    avg_optimization_time_ms: float = Field(default=0.0, ge=0.0)
    calculations_per_minute: float = Field(default=0.0, ge=0.0)
    api_requests_per_minute: float = Field(default=0.0, ge=0.0)

    # Resources
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)
    model_cache_size_mb: float = Field(default=0.0, ge=0.0)


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class FlameguardEvent(BaseModel):
    """Event emitted by FLAMEGUARD agent."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    event_type: str = Field(..., description="Event type")
    source: str = Field(default="GL-002")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    severity: SeverityLevel = Field(default=SeverityLevel.INFO)
    boiler_id: Optional[str] = Field(default=None)
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = Field(default=None)

    class Config:
        use_enum_values = True


class CalculationEvent(BaseModel):
    """Calculation completion event for audit trail."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    calculation_type: CalculationType = Field(...)
    boiler_id: str = Field(...)

    # Inputs
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    input_hash: str = Field(..., description="SHA-256 of inputs")

    # Outputs
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    output_hash: str = Field(..., description="SHA-256 of outputs")

    # Provenance
    formula_id: str = Field(..., description="Formula/method identifier")
    formula_version: str = Field(default="1.0.0")
    deterministic: bool = Field(default=True)
    reproducible: bool = Field(default=True)

    # Performance
    calculation_time_ms: float = Field(default=0.0, ge=0.0)

    class Config:
        use_enum_values = True


# =============================================================================
# API RESPONSE SCHEMAS
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = Field(...)
    message: str = Field(default="")
    data: Optional[Any] = Field(default=None)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    request_id: Optional[str] = Field(default=None)
    processing_time_ms: float = Field(default=0.0, ge=0.0)


class HealthCheckResponse(BaseModel):
    """Health check API response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    uptime_seconds: float = Field(default=0.0)
    checks: Dict[str, str] = Field(
        default_factory=dict,
        description="Component health checks"
    )


# Update forward references
BoilerStatus.update_forward_refs()
AgentStatus.update_forward_refs()
