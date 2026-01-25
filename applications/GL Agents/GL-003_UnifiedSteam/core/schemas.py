"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Schema Definitions

This module defines all Pydantic models for inputs, outputs, process data,
thermodynamic properties, optimization results, and status reporting for
the UNIFIEDSTEAM agent.

All schemas support zero-hallucination principles with deterministic
calculations, SHA-256 provenance tracking, and regulatory compliance.

Standards Compliance:
    - IAPWS-IF97 (International Association for Properties of Water and Steam)
    - ASME PTC 19.11 (Steam and Water in Industrial Systems)
    - ISO 50001 (Energy Management Systems)
    - GHG Protocol (Scope 1 emissions reporting)
"""

from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from .config import (
    OperatingState,
    SteamQuality,
    OptimizationType,
    DeploymentMode,
    TrapFailureMode,
    MaintenancePriority,
    ConfidenceLevel,
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
    AWAITING_APPROVAL = "awaiting_approval"


class CalculationType(Enum):
    """Types of calculations performed."""
    THERMODYNAMIC = "thermodynamic"
    ENTHALPY_BALANCE = "enthalpy_balance"
    DESUPERHEATER = "desuperheater"
    CONDENSATE_RECOVERY = "condensate_recovery"
    TRAP_DIAGNOSTICS = "trap_diagnostics"
    CAUSAL_ANALYSIS = "causal_analysis"
    UNCERTAINTY = "uncertainty"


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


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# PROCESS DATA SCHEMAS
# =============================================================================

class WaterChemistry(BaseModel):
    """Boiler water chemistry data."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )
    ph: float = Field(
        default=9.0,
        ge=0.0,
        le=14.0,
        description="Water pH"
    )
    conductivity_us_cm: float = Field(
        default=500.0,
        ge=0.0,
        le=50000.0,
        description="Conductivity (uS/cm)"
    )
    tds_ppm: float = Field(
        default=300.0,
        ge=0.0,
        le=50000.0,
        description="Total dissolved solids (ppm)"
    )
    silica_ppm: float = Field(
        default=5.0,
        ge=0.0,
        le=1000.0,
        description="Silica concentration (ppm)"
    )
    phosphate_ppm: float = Field(
        default=10.0,
        ge=0.0,
        le=500.0,
        description="Phosphate concentration (ppm)"
    )
    hardness_ppm: float = Field(
        default=0.0,
        ge=0.0,
        le=1000.0,
        description="Water hardness (ppm as CaCO3)"
    )
    dissolved_oxygen_ppb: float = Field(
        default=7.0,
        ge=0.0,
        le=10000.0,
        description="Dissolved oxygen (ppb)"
    )
    iron_ppm: float = Field(
        default=0.1,
        ge=0.0,
        le=100.0,
        description="Iron concentration (ppm)"
    )
    copper_ppm: float = Field(
        default=0.01,
        ge=0.0,
        le=10.0,
        description="Copper concentration (ppm)"
    )


class TrapAcousticsData(BaseModel):
    """Acoustic data from steam trap monitoring."""

    trap_id: str = Field(..., description="Steam trap identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )
    temperature_c: float = Field(
        default=150.0,
        ge=0.0,
        le=400.0,
        description="Trap surface temperature (C)"
    )
    acoustic_level_db: float = Field(
        default=60.0,
        ge=0.0,
        le=140.0,
        description="Acoustic level (dB)"
    )
    dominant_frequency_hz: float = Field(
        default=1000.0,
        ge=0.0,
        le=100000.0,
        description="Dominant frequency (Hz)"
    )
    spectral_signature: Optional[List[float]] = Field(
        default=None,
        description="FFT spectral signature"
    )
    cycle_time_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=3600.0,
        description="Trap cycle time (seconds)"
    )
    subcooling_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=100.0,
        description="Condensate subcooling (C)"
    )


class CondenserData(BaseModel):
    """Condenser/vacuum system data."""

    condenser_id: str = Field(default="COND-001", description="Condenser identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )
    vacuum_kpa_abs: float = Field(
        default=10.0,
        ge=0.1,
        le=101.325,
        description="Condenser vacuum pressure (kPa absolute)"
    )
    hotwell_temperature_c: float = Field(
        default=45.0,
        ge=0.0,
        le=100.0,
        description="Hotwell temperature (C)"
    )
    cooling_water_inlet_temp_c: float = Field(
        default=25.0,
        ge=0.0,
        le=60.0,
        description="Cooling water inlet temperature (C)"
    )
    cooling_water_outlet_temp_c: float = Field(
        default=35.0,
        ge=0.0,
        le=70.0,
        description="Cooling water outlet temperature (C)"
    )
    cooling_water_flow_m3_s: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Cooling water flow rate (m3/s)"
    )
    steam_flow_to_condenser_kg_s: float = Field(
        default=10.0,
        ge=0.0,
        le=500.0,
        description="Steam flow to condenser (kg/s)"
    )
    air_ingress_detected: bool = Field(
        default=False,
        description="Air in-leakage detected"
    )
    tube_fouling_factor: float = Field(
        default=0.0001,
        ge=0.0,
        le=0.01,
        description="Tube fouling factor (m2-K/W)"
    )


class SteamProcessData(BaseModel):
    """Real-time steam system process data from SCADA/DCS."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )
    system_id: str = Field(..., description="Steam system identifier")

    # Operating State
    operating_state: OperatingState = Field(
        default=OperatingState.NORMAL,
        description="Current operating state"
    )

    # Steam Header - Primary Measurements
    header_pressure_kpa: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Steam header pressure (kPa gauge)"
    )
    header_temperature_c: float = Field(
        ...,
        ge=0.0,
        le=700.0,
        description="Steam header temperature (C)"
    )
    steam_flow_kg_s: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Steam flow rate (kg/s)"
    )

    # Steam Quality
    steam_quality: SteamQuality = Field(
        default=SteamQuality.SUPERHEATED,
        description="Steam quality classification"
    )
    dryness_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Dryness fraction (x) for wet steam, 1.0 for superheated"
    )
    superheat_c: float = Field(
        default=50.0,
        ge=0.0,
        le=300.0,
        description="Degrees of superheat above saturation (C)"
    )

    # Condensate Return
    condensate_return_flow_kg_s: float = Field(
        default=0.0,
        ge=0.0,
        le=500.0,
        description="Condensate return flow rate (kg/s)"
    )
    condensate_return_temp_c: float = Field(
        default=80.0,
        ge=0.0,
        le=200.0,
        description="Condensate return temperature (C)"
    )
    condensate_return_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ratio of condensate returned vs steam produced"
    )

    # Water Chemistry
    water_chemistry: Optional[WaterChemistry] = Field(
        default=None,
        description="Water chemistry data"
    )

    # Steam Trap Acoustics
    trap_acoustics: List[TrapAcousticsData] = Field(
        default_factory=list,
        description="Steam trap acoustic monitoring data"
    )

    # Condenser/Vacuum System
    condenser_data: Optional[CondenserData] = Field(
        default=None,
        description="Condenser vacuum system data"
    )

    # Desuperheater
    desuperheater_inlet_temp_c: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=700.0,
        description="Desuperheater inlet temperature (C)"
    )
    desuperheater_outlet_temp_c: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=700.0,
        description="Desuperheater outlet temperature (C)"
    )
    desuperheater_spray_flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Desuperheater spray water flow (kg/s)"
    )
    desuperheater_spray_temp_c: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Spray water temperature (C)"
    )

    # Feedwater
    feedwater_flow_kg_s: float = Field(
        default=0.0,
        ge=0.0,
        le=500.0,
        description="Feedwater flow rate (kg/s)"
    )
    feedwater_temp_c: float = Field(
        default=105.0,
        ge=0.0,
        le=300.0,
        description="Feedwater temperature (C)"
    )
    feedwater_pressure_kpa: float = Field(
        default=5000.0,
        ge=0.0,
        le=50000.0,
        description="Feedwater pressure (kPa)"
    )

    # Blowdown
    blowdown_flow_kg_s: float = Field(
        default=0.5,
        ge=0.0,
        le=50.0,
        description="Continuous blowdown flow rate (kg/s)"
    )
    blowdown_temp_c: float = Field(
        default=180.0,
        ge=0.0,
        le=300.0,
        description="Blowdown temperature (C)"
    )

    # Ambient Conditions
    ambient_temperature_c: float = Field(
        default=25.0,
        ge=-50.0,
        le=60.0,
        description="Ambient temperature (C)"
    )
    ambient_pressure_kpa: float = Field(
        default=101.325,
        ge=80.0,
        le=110.0,
        description="Ambient pressure (kPa absolute)"
    )

    # Data Quality
    data_quality: str = Field(
        default="good",
        description="Data quality indicator (good, suspect, bad)"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# THERMODYNAMIC PROPERTY SCHEMAS
# =============================================================================

class SteamProperties(BaseModel):
    """Steam thermodynamic properties per IAPWS-IF97."""

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Input conditions
    pressure_kpa: float = Field(
        ...,
        ge=0.0,
        le=100000.0,
        description="Pressure (kPa)"
    )
    temperature_c: float = Field(
        ...,
        ge=-273.15,
        le=2000.0,
        description="Temperature (C)"
    )

    # Primary thermodynamic properties
    enthalpy_kj_kg: float = Field(
        ...,
        description="Specific enthalpy h (kJ/kg)"
    )
    entropy_kj_kg_k: float = Field(
        ...,
        description="Specific entropy s (kJ/kg-K)"
    )
    specific_volume_m3_kg: float = Field(
        ...,
        ge=0.0,
        description="Specific volume v (m3/kg)"
    )
    density_kg_m3: float = Field(
        ...,
        ge=0.0,
        description="Density rho (kg/m3)"
    )

    # Saturation properties
    saturation_temperature_c: float = Field(
        ...,
        description="Saturation temperature at given pressure Tsat (C)"
    )
    saturation_pressure_kpa: float = Field(
        ...,
        description="Saturation pressure at given temperature Psat (kPa)"
    )

    # Phase region
    dryness_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Dryness fraction x (0-1), 1.0 if superheated"
    )
    superheat_degree_c: float = Field(
        default=0.0,
        ge=0.0,
        description="Degrees of superheat (C)"
    )
    steam_quality: SteamQuality = Field(
        default=SteamQuality.SUPERHEATED,
        description="Steam quality classification"
    )
    iapws_region: int = Field(
        default=2,
        ge=1,
        le=5,
        description="IAPWS-IF97 region (1-5)"
    )

    # Additional properties
    internal_energy_kj_kg: float = Field(
        default=0.0,
        description="Specific internal energy u (kJ/kg)"
    )
    cp_kj_kg_k: float = Field(
        default=0.0,
        ge=0.0,
        description="Specific heat at constant pressure Cp (kJ/kg-K)"
    )
    cv_kj_kg_k: float = Field(
        default=0.0,
        ge=0.0,
        description="Specific heat at constant volume Cv (kJ/kg-K)"
    )
    speed_of_sound_m_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Speed of sound (m/s)"
    )
    viscosity_pa_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Dynamic viscosity (Pa-s)"
    )
    thermal_conductivity_w_m_k: float = Field(
        default=0.0,
        ge=0.0,
        description="Thermal conductivity (W/m-K)"
    )

    # Provenance
    calculation_method: str = Field(
        default="IAPWS-IF97",
        description="Calculation method/standard"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of inputs for reproducibility"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# ENTHALPY BALANCE SCHEMAS
# =============================================================================

class HeatLossBreakdown(BaseModel):
    """Breakdown of heat losses in steam system."""

    radiation_loss_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Radiation heat loss (kW)"
    )
    radiation_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Radiation loss as % of input"
    )
    convection_loss_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Convection heat loss (kW)"
    )
    convection_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Convection loss as % of input"
    )
    flash_steam_loss_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Flash steam loss (kW)"
    )
    flash_steam_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Flash loss as % of input"
    )
    condensate_loss_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Unrecovered condensate loss (kW)"
    )
    condensate_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Condensate loss as % of input"
    )
    blowdown_loss_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Blowdown heat loss (kW)"
    )
    blowdown_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Blowdown loss as % of input"
    )
    trap_leakage_loss_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Steam trap leakage loss (kW)"
    )
    trap_leakage_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Trap leakage as % of input"
    )
    other_losses_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Other miscellaneous losses (kW)"
    )
    total_losses_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Total heat losses (kW)"
    )
    total_losses_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Total losses as % of input"
    )


class EnthalpyBalanceResult(BaseModel):
    """Enthalpy/energy balance calculation result."""

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )
    system_id: str = Field(..., description="Steam system identifier")

    # Mass Balance
    steam_input_kg_s: float = Field(
        ...,
        ge=0.0,
        description="Steam mass input (kg/s)"
    )
    condensate_output_kg_s: float = Field(
        ...,
        ge=0.0,
        description="Condensate output (kg/s)"
    )
    mass_balance_error_percent: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        description="Mass balance error (%)"
    )
    mass_balance_valid: bool = Field(
        default=True,
        description="Mass balance within tolerance"
    )

    # Energy Balance
    energy_input_kw: float = Field(
        ...,
        ge=0.0,
        description="Total energy input (kW)"
    )
    energy_output_kw: float = Field(
        ...,
        ge=0.0,
        description="Useful energy output (kW)"
    )
    energy_balance_error_percent: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        description="Energy balance error (%)"
    )
    energy_balance_valid: bool = Field(
        default=True,
        description="Energy balance within tolerance"
    )

    # Losses
    losses: HeatLossBreakdown = Field(
        default_factory=HeatLossBreakdown,
        description="Breakdown of heat losses"
    )

    # Heat Rate / Efficiency
    heat_rate_kj_kg: float = Field(
        default=0.0,
        ge=0.0,
        description="Heat rate (kJ/kg steam)"
    )
    system_efficiency_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall system efficiency (%)"
    )

    # KPIs
    specific_steam_consumption_kg_kwh: float = Field(
        default=0.0,
        ge=0.0,
        description="Specific steam consumption (kg/kWh)"
    )
    condensate_return_efficiency_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Condensate return efficiency (%)"
    )
    co2_emissions_kg_hr: float = Field(
        default=0.0,
        ge=0.0,
        description="CO2 emissions from losses (kg/hr)"
    )
    cost_of_losses_usd_hr: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost of energy losses ($/hr)"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing duration (ms)"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# DESUPERHEATER OPTIMIZATION SCHEMAS
# =============================================================================

class SprayWaterSetpoint(BaseModel):
    """Spray water setpoint recommendation."""

    flow_kg_s: float = Field(
        ...,
        ge=0.0,
        description="Spray water flow rate (kg/s)"
    )
    temperature_c: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Spray water temperature (C)"
    )
    pressure_kpa: float = Field(
        ...,
        ge=0.0,
        description="Spray water pressure (kPa)"
    )
    valve_position_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Control valve position (%)"
    )


class DesuperheaterRecommendation(BaseModel):
    """Desuperheater optimization recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique recommendation ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recommendation timestamp"
    )

    # Current conditions
    current_inlet_temp_c: float = Field(
        ...,
        description="Current inlet temperature (C)"
    )
    current_outlet_temp_c: float = Field(
        ...,
        description="Current outlet temperature (C)"
    )
    current_spray_flow_kg_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Current spray flow (kg/s)"
    )

    # Recommended setpoints
    spray_water_setpoint: SprayWaterSetpoint = Field(
        ...,
        description="Recommended spray water setpoint"
    )
    target_outlet_temp_c: float = Field(
        ...,
        ge=0.0,
        le=700.0,
        description="Target outlet temperature (C)"
    )
    target_superheat_c: float = Field(
        ...,
        ge=0.0,
        le=300.0,
        description="Target superheat degree (C)"
    )

    # Constraints
    min_outlet_temp_c: float = Field(
        ...,
        description="Minimum allowable outlet temperature (C)"
    )
    max_spray_flow_kg_s: float = Field(
        ...,
        description="Maximum spray water flow (kg/s)"
    )
    saturation_margin_c: float = Field(
        default=10.0,
        ge=5.0,
        description="Margin above saturation (C)"
    )

    # Risk Assessment
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="Risk level of recommendation"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Identified risk factors"
    )
    water_droplet_carryover_risk: bool = Field(
        default=False,
        description="Risk of water droplet carryover"
    )
    thermal_shock_risk: bool = Field(
        default=False,
        description="Risk of thermal shock"
    )

    # Expected benefits
    expected_energy_savings_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Expected energy savings (kW)"
    )
    expected_cost_savings_usd_hr: float = Field(
        default=0.0,
        description="Expected cost savings ($/hr)"
    )

    # Confidence
    confidence_percent: float = Field(
        default=95.0,
        ge=50.0,
        le=100.0,
        description="Recommendation confidence (%)"
    )

    # Implementation
    requires_operator_approval: bool = Field(
        default=True,
        description="Requires operator approval"
    )
    auto_implement_eligible: bool = Field(
        default=False,
        description="Eligible for auto-implementation"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# CONDENSATE RECOVERY SCHEMAS
# =============================================================================

class FlashLossAnalysis(BaseModel):
    """Flash steam loss analysis."""

    high_pressure_source_kpa: float = Field(
        ...,
        description="High pressure condensate source (kPa)"
    )
    low_pressure_receiver_kpa: float = Field(
        ...,
        description="Low pressure receiver (kPa)"
    )
    condensate_flow_kg_s: float = Field(
        ...,
        ge=0.0,
        description="Condensate flow rate (kg/s)"
    )
    flash_steam_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percent of condensate flashing to steam"
    )
    flash_steam_flow_kg_s: float = Field(
        ...,
        ge=0.0,
        description="Flash steam flow rate (kg/s)"
    )
    energy_in_flash_kw: float = Field(
        ...,
        ge=0.0,
        description="Energy in flash steam (kW)"
    )
    recoverable_energy_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Recoverable energy if flash captured (kW)"
    )
    flash_recovery_recommendation: str = Field(
        default="",
        description="Recommendation for flash steam recovery"
    )


class CondensateRecoveryResult(BaseModel):
    """Condensate recovery optimization result."""

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )
    system_id: str = Field(..., description="Steam system identifier")

    # Current State
    current_return_ratio_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current condensate return ratio (%)"
    )
    current_return_flow_kg_s: float = Field(
        ...,
        ge=0.0,
        description="Current return flow rate (kg/s)"
    )
    current_return_temp_c: float = Field(
        ...,
        ge=0.0,
        description="Current return temperature (C)"
    )

    # Flash Loss Analysis
    flash_losses: List[FlashLossAnalysis] = Field(
        default_factory=list,
        description="Flash steam loss analysis by source"
    )
    total_flash_loss_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Total flash steam loss (kW)"
    )
    total_flash_loss_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Flash loss as % of condensate energy"
    )

    # Recovery Recommendations
    target_return_ratio_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Target condensate return ratio (%)"
    )
    improvement_potential_percent: float = Field(
        default=0.0,
        ge=0.0,
        description="Potential improvement (%)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Specific recommendations"
    )
    priority_actions: List[str] = Field(
        default_factory=list,
        description="Priority actions"
    )

    # ROI Analysis
    estimated_energy_savings_kw: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated energy savings (kW)"
    )
    estimated_annual_savings_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated annual savings ($)"
    )
    estimated_implementation_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Implementation cost ($)"
    )
    simple_payback_months: float = Field(
        default=0.0,
        ge=0.0,
        description="Simple payback period (months)"
    )
    roi_percent: float = Field(
        default=0.0,
        description="Return on investment (%)"
    )

    # Environmental Impact
    co2_reduction_kg_year: float = Field(
        default=0.0,
        ge=0.0,
        description="CO2 reduction potential (kg/year)"
    )
    water_savings_m3_year: float = Field(
        default=0.0,
        ge=0.0,
        description="Water savings potential (m3/year)"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# TRAP DIAGNOSTICS SCHEMAS
# =============================================================================

class TrapHealthAssessment(BaseModel):
    """Individual steam trap health assessment."""

    trap_id: str = Field(..., description="Steam trap identifier")
    location: str = Field(default="", description="Physical location")
    trap_type: str = Field(
        default="thermodynamic",
        description="Trap type (thermodynamic, float, inverted_bucket, thermostatic)"
    )
    size_mm: float = Field(
        default=25.0,
        ge=10.0,
        le=200.0,
        description="Trap size (mm)"
    )
    design_capacity_kg_hr: float = Field(
        default=100.0,
        ge=0.0,
        description="Design condensate capacity (kg/hr)"
    )

    # Current Condition
    status: TrapFailureMode = Field(
        default=TrapFailureMode.HEALTHY,
        description="Current failure mode status"
    )
    failure_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of failure (0-1)"
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.HIGH,
        description="Confidence in assessment"
    )

    # Diagnostic Evidence
    temperature_c: float = Field(
        default=0.0,
        description="Current surface temperature (C)"
    )
    expected_temperature_c: float = Field(
        default=0.0,
        description="Expected temperature for healthy trap (C)"
    )
    temperature_deviation_c: float = Field(
        default=0.0,
        description="Temperature deviation (C)"
    )
    acoustic_signature_match: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Acoustic signature match to healthy baseline"
    )
    cycle_time_normal: bool = Field(
        default=True,
        description="Cycle time within normal range"
    )

    # Loss Estimation
    estimated_loss_rate_kg_hr: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated steam loss rate (kg/hr)"
    )
    estimated_loss_cost_usd_hr: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated loss cost ($/hr)"
    )
    annual_loss_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual loss if not repaired ($)"
    )

    # Maintenance
    maintenance_priority: MaintenancePriority = Field(
        default=MaintenancePriority.ROUTINE,
        description="Maintenance priority"
    )
    recommended_action: str = Field(
        default="Continue monitoring",
        description="Recommended action"
    )
    days_since_last_inspection: Optional[int] = Field(
        default=None,
        description="Days since last inspection"
    )
    days_until_next_inspection: Optional[int] = Field(
        default=None,
        description="Days until next inspection"
    )

    model_config = ConfigDict(use_enum_values=True)


class TrapDiagnosticsResult(BaseModel):
    """Complete trap diagnostics result."""

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )
    system_id: str = Field(..., description="Steam system identifier")

    # Summary Statistics
    total_traps: int = Field(
        default=0,
        ge=0,
        description="Total number of traps assessed"
    )
    healthy_traps: int = Field(
        default=0,
        ge=0,
        description="Number of healthy traps"
    )
    failed_traps: int = Field(
        default=0,
        ge=0,
        description="Number of failed traps"
    )
    at_risk_traps: int = Field(
        default=0,
        ge=0,
        description="Number of traps at risk"
    )
    failure_rate_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall failure rate (%)"
    )

    # Individual Assessments
    trap_assessments: List[TrapHealthAssessment] = Field(
        default_factory=list,
        description="Individual trap assessments"
    )

    # Aggregate Losses
    total_estimated_loss_kg_hr: float = Field(
        default=0.0,
        ge=0.0,
        description="Total estimated steam loss (kg/hr)"
    )
    total_annual_loss_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total annual loss ($)"
    )
    co2_from_losses_kg_year: float = Field(
        default=0.0,
        ge=0.0,
        description="CO2 emissions from trap losses (kg/year)"
    )

    # Priority Maintenance List
    critical_traps: List[str] = Field(
        default_factory=list,
        description="Trap IDs requiring immediate attention"
    )
    high_priority_traps: List[str] = Field(
        default_factory=list,
        description="Trap IDs for high priority maintenance"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing duration (ms)"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# CAUSAL ANALYSIS SCHEMAS
# =============================================================================

class CausalFactor(BaseModel):
    """Individual causal factor in root cause analysis."""

    factor_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Factor identifier"
    )
    name: str = Field(..., description="Factor name")
    description: str = Field(default="", description="Factor description")
    category: str = Field(
        default="process",
        description="Category (process, equipment, environmental, operational)"
    )

    # Causal strength
    causal_strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Causal strength (0-1)"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in causal relationship"
    )
    rank: int = Field(
        default=1,
        ge=1,
        description="Rank among causal factors"
    )

    # Evidence
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence"
    )
    data_points: Dict[str, Any] = Field(
        default_factory=dict,
        description="Related data points"
    )

    # Relationship
    upstream_factors: List[str] = Field(
        default_factory=list,
        description="Upstream causal factors"
    )
    downstream_effects: List[str] = Field(
        default_factory=list,
        description="Downstream effects"
    )


class Counterfactual(BaseModel):
    """Counterfactual scenario analysis."""

    scenario_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Scenario identifier"
    )
    description: str = Field(..., description="Scenario description")
    variable_changed: str = Field(..., description="Variable that is changed")
    original_value: float = Field(..., description="Original value")
    counterfactual_value: float = Field(..., description="Counterfactual value")
    unit: str = Field(default="", description="Unit of measurement")

    # Predicted outcome
    predicted_outcome: str = Field(..., description="Predicted outcome")
    predicted_improvement_percent: float = Field(
        default=0.0,
        description="Predicted improvement (%)"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in prediction"
    )


class InterventionRecommendation(BaseModel):
    """Recommended intervention based on causal analysis."""

    intervention_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Intervention identifier"
    )
    title: str = Field(..., description="Intervention title")
    description: str = Field(..., description="Detailed description")
    target_factor: str = Field(..., description="Target causal factor")

    # Expected impact
    expected_improvement_percent: float = Field(
        default=0.0,
        ge=0.0,
        description="Expected improvement (%)"
    )
    expected_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Implementation cost ($)"
    )
    expected_savings_usd_year: float = Field(
        default=0.0,
        ge=0.0,
        description="Expected annual savings ($)"
    )
    payback_months: float = Field(
        default=0.0,
        ge=0.0,
        description="Payback period (months)"
    )

    # Priority
    priority: MaintenancePriority = Field(
        default=MaintenancePriority.MEDIUM,
        description="Implementation priority"
    )
    ease_of_implementation: str = Field(
        default="medium",
        description="Ease: easy, medium, difficult"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="Implementation risk"
    )

    model_config = ConfigDict(use_enum_values=True)


class CausalAnalysisResult(BaseModel):
    """Complete causal analysis result."""

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique analysis ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    system_id: str = Field(..., description="Steam system identifier")

    # Problem Statement
    problem_description: str = Field(
        ...,
        description="Description of the problem analyzed"
    )
    problem_metric: str = Field(
        default="efficiency",
        description="Primary metric affected"
    )
    problem_severity: SeverityLevel = Field(
        default=SeverityLevel.WARNING,
        description="Problem severity"
    )

    # Root Causes (ranked)
    root_causes: List[CausalFactor] = Field(
        default_factory=list,
        description="Root causes ranked by causal strength"
    )
    primary_root_cause: Optional[CausalFactor] = Field(
        default=None,
        description="Primary root cause"
    )

    # Evidence Summary
    evidence_summary: List[str] = Field(
        default_factory=list,
        description="Summary of supporting evidence"
    )
    data_quality_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Data quality score for analysis"
    )

    # Counterfactuals
    counterfactuals: List[Counterfactual] = Field(
        default_factory=list,
        description="Counterfactual scenarios"
    )

    # Interventions
    interventions: List[InterventionRecommendation] = Field(
        default_factory=list,
        description="Recommended interventions"
    )
    priority_intervention: Optional[InterventionRecommendation] = Field(
        default=None,
        description="Highest priority intervention"
    )

    # Analysis Metadata
    analysis_method: str = Field(
        default="bayesian_network",
        description="Analysis method used"
    )
    confidence_overall: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall confidence in analysis"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing duration (ms)"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# EXPLAINABILITY SCHEMAS
# =============================================================================

class FeatureContribution(BaseModel):
    """Feature contribution to model output."""

    feature_name: str = Field(..., description="Feature name")
    feature_value: float = Field(..., description="Feature value")
    contribution: float = Field(
        ...,
        description="Contribution to output (SHAP value or similar)"
    )
    contribution_percent: float = Field(
        default=0.0,
        description="Contribution as percentage"
    )
    direction: str = Field(
        default="positive",
        description="positive or negative contribution"
    )


class PhysicsTrace(BaseModel):
    """Physics-based calculation trace for explainability."""

    step_number: int = Field(..., ge=1, description="Step number")
    calculation_name: str = Field(..., description="Calculation name")
    formula: str = Field(..., description="Formula/equation used")
    formula_reference: str = Field(
        default="",
        description="Reference (e.g., IAPWS-IF97 Eq. 1)"
    )
    inputs: Dict[str, float] = Field(
        default_factory=dict,
        description="Input values"
    )
    output: float = Field(..., description="Output value")
    output_unit: str = Field(default="", description="Output unit")
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made"
    )


class ModelTrace(BaseModel):
    """ML model prediction trace for explainability."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(default="1.0.0", description="Model version")
    model_type: str = Field(
        default="ensemble",
        description="Model type (ensemble, neural_network, etc.)"
    )
    prediction: float = Field(..., description="Model prediction")
    prediction_unit: str = Field(default="", description="Prediction unit")

    # Feature contributions
    feature_contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Feature contributions (SHAP/LIME)"
    )
    top_features: List[str] = Field(
        default_factory=list,
        description="Top contributing features"
    )

    # Explanation method
    explanation_method: str = Field(
        default="SHAP",
        description="Explanation method (SHAP, LIME)"
    )
    base_value: float = Field(
        default=0.0,
        description="Base/expected value"
    )


class ExplainabilityPayload(BaseModel):
    """Complete explainability payload for optimization results."""

    explanation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique explanation ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Explanation timestamp"
    )

    # Physics-based trace
    physics_trace: List[PhysicsTrace] = Field(
        default_factory=list,
        description="Physics calculation trace"
    )
    physics_summary: str = Field(
        default="",
        description="Human-readable physics summary"
    )

    # ML model trace
    model_trace: Optional[ModelTrace] = Field(
        default=None,
        description="ML model prediction trace"
    )
    model_summary: str = Field(
        default="",
        description="Human-readable model summary"
    )

    # Confidence and uncertainty
    confidence_level: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Overall confidence level"
    )
    confidence_justification: str = Field(
        default="",
        description="Justification for confidence level"
    )

    # Uncertainty bounds
    uncertainty_quantified: bool = Field(
        default=True,
        description="Whether uncertainty was quantified"
    )
    uncertainty_method: str = Field(
        default="monte_carlo",
        description="Uncertainty quantification method"
    )

    # Natural language explanation
    natural_language_explanation: str = Field(
        default="",
        description="Plain English explanation"
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights"
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Important caveats/limitations"
    )


# =============================================================================
# UNCERTAINTY SCHEMAS
# =============================================================================

class ConfidenceInterval(BaseModel):
    """Statistical confidence interval."""

    confidence_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description="Confidence level (e.g., 0.95 for 95%)"
    )
    lower_bound: float = Field(..., description="Lower bound")
    upper_bound: float = Field(..., description="Upper bound")
    width: float = Field(default=0.0, description="Interval width")


class UncertaintyBounds(BaseModel):
    """Uncertainty bounds for a calculated value."""

    parameter_name: str = Field(..., description="Parameter name")
    mean_value: float = Field(..., description="Mean/expected value")
    unit: str = Field(default="", description="Unit of measurement")

    # 95% confidence interval (default)
    lower_95: float = Field(..., description="Lower 95% confidence bound")
    upper_95: float = Field(..., description="Upper 95% confidence bound")
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level"
    )

    # Additional percentiles
    lower_99: Optional[float] = Field(
        default=None,
        description="Lower 99% confidence bound"
    )
    upper_99: Optional[float] = Field(
        default=None,
        description="Upper 99% confidence bound"
    )

    # Distribution information
    std_deviation: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation"
    )
    coefficient_of_variation: float = Field(
        default=0.0,
        ge=0.0,
        description="Coefficient of variation (%)"
    )
    distribution_type: str = Field(
        default="normal",
        description="Assumed distribution type"
    )

    # Uncertainty sources
    measurement_uncertainty_contribution: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Contribution from measurement uncertainty"
    )
    model_uncertainty_contribution: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Contribution from model uncertainty"
    )


# =============================================================================
# COMBINED OPTIMIZATION RESULT
# =============================================================================

class OptimizationResult(BaseModel):
    """Complete optimization result with all components."""

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )
    system_id: str = Field(..., description="Steam system identifier")

    # Optimization Type
    optimization_type: OptimizationType = Field(
        default=OptimizationType.COMBINED,
        description="Type of optimization performed"
    )
    status: OptimizationStatus = Field(
        default=OptimizationStatus.COMPLETED,
        description="Optimization status"
    )

    # Input conditions summary
    input_process_data_hash: str = Field(
        default="",
        description="SHA-256 hash of input process data"
    )
    operating_state: OperatingState = Field(
        default=OperatingState.NORMAL,
        description="Operating state at time of optimization"
    )

    # Results by type
    enthalpy_balance: Optional[EnthalpyBalanceResult] = Field(
        default=None,
        description="Enthalpy balance result"
    )
    desuperheater_recommendation: Optional[DesuperheaterRecommendation] = Field(
        default=None,
        description="Desuperheater optimization recommendation"
    )
    condensate_recovery: Optional[CondensateRecoveryResult] = Field(
        default=None,
        description="Condensate recovery optimization result"
    )
    trap_diagnostics: Optional[TrapDiagnosticsResult] = Field(
        default=None,
        description="Steam trap diagnostics result"
    )
    causal_analysis: Optional[CausalAnalysisResult] = Field(
        default=None,
        description="Causal analysis result"
    )

    # Uncertainty Bounds
    uncertainty_bounds: List[UncertaintyBounds] = Field(
        default_factory=list,
        description="Uncertainty bounds for key parameters"
    )

    # Explainability
    explainability: Optional[ExplainabilityPayload] = Field(
        default=None,
        description="Explainability payload"
    )

    # Summary metrics
    efficiency_current_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Current system efficiency (%)"
    )
    efficiency_potential_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Potential system efficiency (%)"
    )
    efficiency_improvement_percent: float = Field(
        default=0.0,
        description="Potential improvement (%)"
    )
    annual_savings_potential_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual savings potential ($)"
    )
    co2_reduction_potential_kg_year: float = Field(
        default=0.0,
        ge=0.0,
        description="CO2 reduction potential (kg/year)"
    )

    # Execution details
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time (ms)"
    )
    optimizer_iterations: int = Field(
        default=0,
        ge=0,
        description="Optimizer iterations"
    )
    convergence_achieved: bool = Field(
        default=True,
        description="Optimization converged"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for complete audit trail"
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        description="Calculation steps for audit"
    )

    model_config = ConfigDict(use_enum_values=True)


class SystemOptimizationSummary(BaseModel):
    """High-level summary of system optimization status."""

    system_id: str = Field(..., description="Steam system identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Summary timestamp"
    )

    # Overall Status
    overall_health_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Overall system health score (0-100)"
    )
    optimization_status: str = Field(
        default="optimal",
        description="Status: optimal, suboptimal, degraded, critical"
    )

    # Key Metrics
    current_efficiency_percent: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Current efficiency (%)"
    )
    condensate_return_percent: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Condensate return rate (%)"
    )
    trap_failure_rate_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Trap failure rate (%)"
    )

    # Active Issues
    active_issues: List[str] = Field(
        default_factory=list,
        description="Active issues requiring attention"
    )
    pending_recommendations: int = Field(
        default=0,
        ge=0,
        description="Number of pending recommendations"
    )

    # Savings Opportunity
    total_annual_savings_opportunity_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total annual savings opportunity ($)"
    )


# =============================================================================
# STATUS SCHEMAS
# =============================================================================

class SteamSystemStatus(BaseModel):
    """Current steam system status."""

    system_id: str = Field(..., description="Steam system identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )

    # Operating Status
    operating_state: OperatingState = Field(
        default=OperatingState.NORMAL,
        description="Current operating state"
    )
    hours_since_startup: float = Field(
        default=0.0,
        ge=0.0,
        description="Hours since last startup"
    )

    # Current Readings
    header_pressure_kpa: float = Field(
        default=0.0,
        ge=0.0,
        description="Current header pressure (kPa)"
    )
    header_temperature_c: float = Field(
        default=0.0,
        description="Current header temperature (C)"
    )
    steam_flow_kg_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Current steam flow (kg/s)"
    )

    # Performance
    current_efficiency_percent: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Current efficiency (%)"
    )
    efficiency_trend: str = Field(
        default="stable",
        description="Trend: improving, stable, declining"
    )

    # Optimization
    optimization_active: bool = Field(
        default=False,
        description="Optimization currently running"
    )
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.ADVISORY,
        description="Current deployment mode"
    )
    last_optimization: Optional[datetime] = Field(
        default=None,
        description="Last optimization timestamp"
    )
    pending_recommendations: int = Field(
        default=0,
        ge=0,
        description="Pending recommendations"
    )

    # Alarms
    active_alarm_count: int = Field(
        default=0,
        ge=0,
        description="Active alarm count"
    )
    critical_alarm_count: int = Field(
        default=0,
        ge=0,
        description="Critical alarm count"
    )

    # Integration
    scada_connected: bool = Field(
        default=True,
        description="SCADA connection status"
    )
    data_quality: str = Field(
        default="good",
        description="Data quality: good, suspect, bad"
    )
    last_data_update: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last data update"
    )

    model_config = ConfigDict(use_enum_values=True)


class AgentStatus(BaseModel):
    """GL-003 UNIFIEDSTEAM agent status."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(default="UNIFIEDSTEAM", description="Agent name")
    agent_version: str = Field(default="1.0.0", description="Agent version")
    agent_type: str = Field(default="GL-003", description="Agent type")

    # Health
    status: str = Field(default="running", description="Agent status")
    health: str = Field(default="healthy", description="Health: healthy, degraded, unhealthy")
    uptime_seconds: float = Field(default=0.0, ge=0.0, description="Uptime")
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last heartbeat"
    )

    # Managed Systems
    managed_systems: List[str] = Field(
        default_factory=list,
        description="Managed steam system IDs"
    )
    system_statuses: Dict[str, SteamSystemStatus] = Field(
        default_factory=dict,
        description="Status per system"
    )

    # Performance
    optimizations_performed: int = Field(default=0, ge=0)
    optimizations_successful: int = Field(default=0, ge=0)
    total_efficiency_improvement_percent: float = Field(default=0.0)
    total_cost_savings_usd: float = Field(default=0.0, ge=0.0)
    total_co2_reduction_kg: float = Field(default=0.0, ge=0.0)

    # Resources
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class SteamSystemEvent(BaseModel):
    """Event emitted by UNIFIEDSTEAM agent."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event identifier"
    )
    event_type: str = Field(..., description="Event type")
    source: str = Field(default="GL-003", description="Event source")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.INFO,
        description="Event severity"
    )
    system_id: Optional[str] = Field(
        default=None,
        description="Related steam system ID"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID"
    )

    model_config = ConfigDict(use_enum_values=True)


class OptimizationEvent(BaseModel):
    """Optimization-related event."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    system_id: str = Field(..., description="Steam system ID")
    optimization_type: OptimizationType = Field(
        ...,
        description="Optimization type"
    )
    status: OptimizationStatus = Field(
        ...,
        description="Optimization status"
    )

    # Results summary
    efficiency_improvement_percent: float = Field(
        default=0.0,
        description="Efficiency improvement (%)"
    )
    cost_savings_usd_hr: float = Field(
        default=0.0,
        description="Cost savings ($/hr)"
    )

    # Recommendations
    setpoint_changes_count: int = Field(
        default=0,
        ge=0,
        description="Number of setpoint changes"
    )
    requires_approval: bool = Field(
        default=True,
        description="Requires operator approval"
    )
    auto_implemented: bool = Field(
        default=False,
        description="Was auto-implemented"
    )

    model_config = ConfigDict(use_enum_values=True)


class AlarmEvent(BaseModel):
    """Alarm event from steam system."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    system_id: str = Field(..., description="Steam system ID")
    alarm_type: str = Field(..., description="Alarm type")
    severity: SeverityLevel = Field(..., description="Alarm severity")
    state: AlarmState = Field(
        default=AlarmState.ACTIVE,
        description="Alarm state"
    )

    # Alarm details
    description: str = Field(..., description="Alarm description")
    source_tag: Optional[str] = Field(
        default=None,
        description="Source sensor/tag"
    )
    measured_value: Optional[float] = Field(
        default=None,
        description="Measured value"
    )
    threshold_value: Optional[float] = Field(
        default=None,
        description="Threshold value"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement"
    )

    # Response
    automatic_response: Optional[str] = Field(
        default=None,
        description="Automatic response taken"
    )
    operator_action_required: bool = Field(
        default=False,
        description="Operator action required"
    )

    # Acknowledgment
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = Field(default=None)
    acknowledged_at: Optional[datetime] = Field(default=None)

    model_config = ConfigDict(use_enum_values=True)


# Update forward references (Pydantic v2 syntax)
SteamSystemStatus.model_rebuild()
AgentStatus.model_rebuild()
OptimizationResult.model_rebuild()
