# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY - Schema Definitions

This module defines all Pydantic models for inputs, outputs, analysis results,
and status reporting for the Burner Maintenance Predictor Agent.

All models include comprehensive validation, documentation, and support for
provenance tracking through SHA-256 hashes.

Input Models:
    - BurnerOperatingData: Operating hours, cycles, fuel consumption
    - FlameCharacteristics: Temperature, shape, stability, color index
    - FuelQualityData: HHV, contaminants, moisture, ash content
    - MaintenanceHistoryRecord: Past maintenance records, failure modes
    - BurnerSensorData: Real-time sensor readings

Output Models:
    - BurnerHealthScore: 0-100 score with component breakdown
    - MaintenancePrediction: Predicted maintenance needs with confidence
    - ReplacementSchedule: Optimal replacement timing with economics
    - WorkOrder: CMMS work order details
    - GL021Result: Comprehensive analysis result with provenance hash

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.schemas import (
    ...     BurnerOperatingData,
    ...     FlameCharacteristics,
    ...     GL021Result,
    ... )
    >>> operating_data = BurnerOperatingData(
    ...     burner_id="BNR-001",
    ...     operating_hours=15000.0,
    ...     start_stop_cycles=450,
    ... )

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS - SCHEMA-SPECIFIC ENUMERATIONS
# =============================================================================

class MaintenanceType(str, Enum):
    """
    Types of burner maintenance activities.

    Categorized by scope and impact on operations:
    - INSPECTION: Visual and instrument-based checks
    - CLEANING: Removal of deposits and fouling
    - REPAIR: Corrective actions for identified issues
    - REPLACEMENT: Component or full burner replacement
    - CALIBRATION: Instrumentation and control calibration
    - OVERHAUL: Major comprehensive maintenance
    """
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    CALIBRATION = "calibration"
    OVERHAUL = "overhaul"
    TUNING = "tuning"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"


class FailureMode(str, Enum):
    """
    Burner failure modes tracked by BURNERSENTRY.

    Based on API 535, NFPA 85, and industry failure statistics.
    Categories include flame, fuel system, air system, and controls.
    """
    # Flame-related failures
    FLAME_INSTABILITY = "flame_instability"
    FLAME_LOSS = "flame_loss"
    FLAME_IMPINGEMENT = "flame_impingement"
    DELAYED_IGNITION = "delayed_ignition"
    FLASHBACK = "flashback"
    FLAME_ROLLOUT = "flame_rollout"
    PULSATION = "pulsation"

    # Nozzle/tip failures
    NOZZLE_DEGRADATION = "nozzle_degradation"
    NOZZLE_PLUGGING = "nozzle_plugging"
    NOZZLE_EROSION = "nozzle_erosion"
    TIP_BURNBACK = "tip_burnback"
    DRIPPING = "dripping"  # Oil burners

    # Refractory failures
    REFRACTORY_DAMAGE = "refractory_damage"
    REFRACTORY_SPALLING = "refractory_spalling"
    REFRACTORY_EROSION = "refractory_erosion"
    HOT_SPOTS = "hot_spots"

    # Ignition system
    IGNITOR_FAILURE = "ignitor_failure"
    PILOT_FAILURE = "pilot_failure"
    SPARK_ROD_FOULING = "spark_rod_fouling"

    # Flame safeguard
    FLAME_SCANNER_FAILURE = "flame_scanner_failure"
    FALSE_FLAME_SIGNAL = "false_flame_signal"
    UV_SENSOR_DEGRADATION = "uv_sensor_degradation"

    # Fuel system
    FUEL_VALVE_LEAK = "fuel_valve_leak"
    FUEL_VALVE_STUCK = "fuel_valve_stuck"
    FUEL_PRESSURE_LOSS = "fuel_pressure_loss"
    FUEL_FILTER_PLUGGING = "fuel_filter_plugging"
    REGULATOR_FAILURE = "regulator_failure"

    # Air system
    AIR_DAMPER_FAILURE = "air_damper_failure"
    COMBUSTION_AIR_LOSS = "combustion_air_loss"
    AIR_FUEL_RATIO_DRIFT = "air_fuel_ratio_drift"
    FGR_SYSTEM_FAILURE = "fgr_system_failure"

    # Controls
    CONTROL_SYSTEM_FAILURE = "control_system_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    SENSOR_FAILURE = "sensor_failure"
    BMS_LOCKOUT = "bms_lockout"

    # Other
    EXTERNAL_DAMAGE = "external_damage"
    CORROSION = "corrosion"
    UNKNOWN = "unknown"


class PredictionConfidence(str, Enum):
    """
    Confidence levels for maintenance predictions.

    Based on data quality, model performance, and uncertainty quantification.
    """
    VERY_HIGH = "very_high"  # >95% confidence
    HIGH = "high"  # 85-95% confidence
    MEDIUM = "medium"  # 70-85% confidence
    LOW = "low"  # 50-70% confidence
    UNCERTAIN = "uncertain"  # <50% confidence


class MaintenancePriority(str, Enum):
    """Work order priority levels."""
    EMERGENCY = "emergency"  # Immediate action required
    CRITICAL = "critical"  # Within 24 hours
    HIGH = "high"  # Within 48 hours
    MEDIUM = "medium"  # Within 1 week
    LOW = "low"  # Within 2 weeks
    ROUTINE = "routine"  # Next scheduled outage


class TaskStatus(str, Enum):
    """Status of maintenance tasks."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"
    OVERDUE = "overdue"


class ComponentType(str, Enum):
    """Burner component types for health tracking."""
    NOZZLE = "nozzle"
    REFRACTORY = "refractory"
    IGNITOR = "ignitor"
    PILOT = "pilot"
    FLAME_SCANNER = "flame_scanner"
    FUEL_VALVE = "fuel_valve"
    AIR_DAMPER = "air_damper"
    ACTUATOR = "actuator"
    SPARK_ROD = "spark_rod"
    DIFFUSER = "diffuser"
    SWIRLER = "swirler"
    BURNER_TILE = "burner_tile"
    FUEL_FILTER = "fuel_filter"
    REGULATOR = "regulator"
    BLOWER = "blower"
    FGR_VALVE = "fgr_valve"


class DegradationMode(str, Enum):
    """Component degradation modes for lifecycle modeling."""
    WEAR = "wear"  # Gradual mechanical wear
    FATIGUE = "fatigue"  # Cyclic stress fatigue
    CORROSION = "corrosion"  # Chemical/thermal corrosion
    EROSION = "erosion"  # Abrasive/particle erosion
    FOULING = "fouling"  # Deposit accumulation
    THERMAL_SHOCK = "thermal_shock"  # Rapid temp changes
    CREEP = "creep"  # High temp deformation
    OXIDATION = "oxidation"  # High temp oxidation
    RANDOM = "random"  # Random failures


class FlameStatus(str, Enum):
    """Flame condition status."""
    OPTIMAL = "optimal"
    STABLE = "stable"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"
    CRITICAL = "critical"
    EXTINGUISHED = "extinguished"


class HealthTrend(str, Enum):
    """Health score trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    RAPID_DEGRADATION = "rapid_degradation"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    ADVISORY = "advisory"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


# =============================================================================
# INPUT SCHEMAS - OPERATING DATA
# =============================================================================

class BurnerOperatingData(BaseModel):
    """
    Burner operating data for maintenance prediction.

    Captures runtime hours, cycles, fuel consumption, and firing rate
    information used for degradation modeling and RUL estimation.

    Attributes:
        burner_id: Unique burner identifier
        operating_hours: Total accumulated operating hours
        start_stop_cycles: Number of start/stop cycles
        fuel_consumption_rate_mmbtu_hr: Current fuel consumption rate
        firing_rate_percent: Current firing rate as percentage of capacity

    Example:
        >>> data = BurnerOperatingData(
        ...     burner_id="BNR-001",
        ...     operating_hours=15000.0,
        ...     start_stop_cycles=450,
        ...     fuel_consumption_rate_mmbtu_hr=45.0,
        ...     firing_rate_percent=90.0,
        ... )
    """

    burner_id: str = Field(
        ...,
        description="Unique burner identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Runtime accumulation
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    operating_hours_since_overhaul: Optional[float] = Field(
        default=None,
        ge=0,
        description="Hours since last major overhaul"
    )
    start_stop_cycles: int = Field(
        default=0,
        ge=0,
        description="Total start/stop cycle count"
    )
    cycles_since_overhaul: Optional[int] = Field(
        default=None,
        ge=0,
        description="Cycles since last major overhaul"
    )
    hot_starts: int = Field(
        default=0,
        ge=0,
        description="Number of hot starts (relight within 1 hour)"
    )
    cold_starts: int = Field(
        default=0,
        ge=0,
        description="Number of cold starts"
    )

    # Current operating conditions
    fuel_consumption_rate_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Current fuel consumption rate (MMBtu/hr)"
    )
    firing_rate_percent: float = Field(
        default=0.0,
        ge=0,
        le=110,  # Allow slight overfiring
        description="Current firing rate (% of capacity)"
    )
    is_operating: bool = Field(
        default=False,
        description="Burner currently operating"
    )
    operating_mode: str = Field(
        default="auto",
        description="Operating mode (auto, manual, low_fire, high_fire)"
    )

    # Load profile
    avg_firing_rate_24h_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="24-hour average firing rate (%)"
    )
    avg_firing_rate_7d_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="7-day average firing rate (%)"
    )
    time_at_high_fire_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of time at high fire"
    )
    time_at_low_fire_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of time at low fire"
    )

    # Combustion parameters
    excess_air_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Current excess air percentage"
    )
    stack_oxygen_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=21,
        description="Stack O2 percentage"
    )
    stack_temperature_c: Optional[float] = Field(
        default=None,
        description="Stack temperature (C)"
    )
    combustion_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=50,
        le=100,
        description="Current combustion efficiency (%)"
    )

    # Safety events
    flame_failures_30d: int = Field(
        default=0,
        ge=0,
        description="Flame failures in last 30 days"
    )
    safety_shutdowns_30d: int = Field(
        default=0,
        ge=0,
        description="Safety shutdowns in last 30 days"
    )
    failed_ignition_attempts_30d: int = Field(
        default=0,
        ge=0,
        description="Failed ignition attempts in last 30 days"
    )

    @validator("firing_rate_percent")
    def validate_firing_rate(cls, v, values):
        """Warn if firing rate exceeds 100%."""
        if v > 100:
            # Log warning but allow (burners can temporarily overfire)
            pass
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FlameCharacteristics(BaseModel):
    """
    Flame characteristic measurements for quality analysis.

    Captures flame temperature, stability, shape, and color metrics
    used to assess combustion quality and predict maintenance needs.

    Attributes:
        flame_temperature_c: Measured flame temperature
        flame_stability_index: Stability index (0-1, 1=perfectly stable)
        flame_shape_score: Shape conformance score (0-1)
        color_index: Color index based on blackbody radiation (0-1)

    Example:
        >>> flame = FlameCharacteristics(
        ...     burner_id="BNR-001",
        ...     flame_temperature_c=1650.0,
        ...     flame_stability_index=0.92,
        ...     flame_shape_score=0.88,
        ...     color_index=0.85,
        ... )
    """

    burner_id: str = Field(
        ...,
        description="Burner identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )

    # Temperature measurements
    flame_temperature_c: float = Field(
        ...,
        gt=500,
        lt=2500,
        description="Measured flame temperature (C)"
    )
    flame_temperature_variance_c: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flame temperature variance (C)"
    )
    peak_flame_temp_c: Optional[float] = Field(
        default=None,
        gt=0,
        description="Peak flame temperature (C)"
    )

    # Stability metrics (0-1 scale, 1 = optimal)
    flame_stability_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Flame stability index (0-1)"
    )
    flame_flicker_frequency_hz: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flame flicker frequency (Hz)"
    )
    flame_oscillation_amplitude_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Flame oscillation amplitude (%)"
    )

    # Shape metrics (0-1 scale, 1 = ideal shape)
    flame_shape_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Flame shape conformance score (0-1)"
    )
    flame_length_mm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Measured flame length (mm)"
    )
    flame_diameter_mm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Measured flame diameter (mm)"
    )
    flame_angle_deg: Optional[float] = Field(
        default=None,
        ge=0,
        le=90,
        description="Flame angle from burner axis (degrees)"
    )
    symmetry_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Flame symmetry score (0-1)"
    )

    # Color metrics (based on blackbody radiation)
    color_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Flame color index (0-1)"
    )
    color_temperature_k: Optional[float] = Field(
        default=None,
        gt=0,
        description="Color temperature (Kelvin)"
    )
    luminosity_index: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Flame luminosity index"
    )

    # Flame detection signals
    uv_signal_strength_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="UV detector signal strength (%)"
    )
    ir_signal_strength_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="IR detector signal strength (%)"
    )
    flame_rod_current_ua: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flame rod ionization current (uA)"
    )

    # Impingement detection
    impingement_detected: bool = Field(
        default=False,
        description="Flame impingement detected"
    )
    impingement_location: Optional[str] = Field(
        default=None,
        description="Impingement location if detected"
    )

    # Derived status
    flame_status: FlameStatus = Field(
        default=FlameStatus.STABLE,
        description="Overall flame status"
    )

    @root_validator
    def derive_flame_status(cls, values):
        """Derive flame status from metrics."""
        stability = values.get("flame_stability_index", 1.0)
        shape = values.get("flame_shape_score", 1.0)
        color = values.get("color_index", 1.0)

        avg_score = (stability + shape + color) / 3

        if avg_score >= 0.95:
            values["flame_status"] = FlameStatus.OPTIMAL
        elif avg_score >= 0.85:
            values["flame_status"] = FlameStatus.STABLE
        elif avg_score >= 0.75:
            values["flame_status"] = FlameStatus.MARGINAL
        elif avg_score >= 0.60:
            values["flame_status"] = FlameStatus.UNSTABLE
        else:
            values["flame_status"] = FlameStatus.CRITICAL

        return values


class FuelQualityData(BaseModel):
    """
    Fuel quality analysis data.

    Captures fuel composition and quality metrics that impact burner
    performance and component degradation.

    Attributes:
        higher_heating_value_mj_m3: Fuel HHV for gas (MJ/m3)
        methane_pct: Methane content for natural gas (%)
        h2s_ppm: Hydrogen sulfide content (ppm)
        moisture_pct: Moisture content (%)

    Example:
        >>> fuel = FuelQualityData(
        ...     fuel_id="NG-001",
        ...     higher_heating_value_mj_m3=38.5,
        ...     methane_pct=94.5,
        ...     h2s_ppm=2.5,
        ...     moisture_pct=0.1,
        ... )
    """

    fuel_id: str = Field(
        ...,
        description="Fuel sample/source identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    fuel_type: str = Field(
        default="natural_gas",
        description="Fuel type"
    )

    # Heating value
    higher_heating_value_mj_m3: Optional[float] = Field(
        default=None,
        gt=0,
        description="Higher Heating Value for gas (MJ/m3)"
    )
    higher_heating_value_btu_scf: Optional[float] = Field(
        default=None,
        gt=0,
        description="Higher Heating Value for gas (BTU/SCF)"
    )
    lower_heating_value_mj_m3: Optional[float] = Field(
        default=None,
        gt=0,
        description="Lower Heating Value for gas (MJ/m3)"
    )
    higher_heating_value_mj_kg: Optional[float] = Field(
        default=None,
        gt=0,
        description="Higher Heating Value for liquid (MJ/kg)"
    )

    # Gas composition (natural gas)
    methane_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Methane content (%)"
    )
    ethane_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Ethane content (%)"
    )
    propane_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Propane content (%)"
    )
    butane_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Butane content (%)"
    )
    pentane_plus_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Pentane+ content (%)"
    )
    nitrogen_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=50,
        description="Nitrogen content (%)"
    )
    co2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=20,
        description="CO2 content (%)"
    )
    oxygen_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=5,
        description="Oxygen content (%)"
    )
    hydrogen_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Hydrogen content (%)"
    )

    # Contaminants
    h2s_ppm: float = Field(
        default=0.0,
        ge=0,
        description="Hydrogen sulfide content (ppm)"
    )
    total_sulfur_ppm: float = Field(
        default=0.0,
        ge=0,
        description="Total sulfur content (ppm)"
    )
    mercaptan_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Mercaptan content (ppm)"
    )
    moisture_pct: float = Field(
        default=0.0,
        ge=0,
        le=10,
        description="Moisture content (%)"
    )
    water_dew_point_c: Optional[float] = Field(
        default=None,
        description="Water dew point (C)"
    )
    hydrocarbon_dew_point_c: Optional[float] = Field(
        default=None,
        description="Hydrocarbon dew point (C)"
    )

    # Liquid fuel properties (fuel oil)
    viscosity_cst_40c: Optional[float] = Field(
        default=None,
        gt=0,
        description="Kinematic viscosity at 40C (cSt)"
    )
    viscosity_cst_100c: Optional[float] = Field(
        default=None,
        gt=0,
        description="Kinematic viscosity at 100C (cSt)"
    )
    density_kg_m3: Optional[float] = Field(
        default=None,
        gt=0,
        description="Density (kg/m3)"
    )
    api_gravity: Optional[float] = Field(
        default=None,
        description="API gravity"
    )
    ash_content_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=5,
        description="Ash content (%)"
    )
    carbon_residue_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=20,
        description="Carbon residue (%)"
    )
    vanadium_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Vanadium content (ppm)"
    )
    sodium_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Sodium content (ppm)"
    )
    nickel_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Nickel content (ppm)"
    )
    pour_point_c: Optional[float] = Field(
        default=None,
        description="Pour point (C)"
    )
    flash_point_c: Optional[float] = Field(
        default=None,
        gt=0,
        description="Flash point (C)"
    )

    # Derived indices
    wobbe_index: Optional[float] = Field(
        default=None,
        gt=0,
        description="Wobbe Index (MJ/m3)"
    )
    methane_number: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Methane number (knock resistance)"
    )
    specific_gravity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Specific gravity"
    )

    # Quality assessment
    quality_grade: str = Field(
        default="standard",
        description="Quality grade (premium, standard, marginal, poor)"
    )
    within_specification: bool = Field(
        default=True,
        description="Fuel within specification limits"
    )
    specification_deviations: List[str] = Field(
        default_factory=list,
        description="List of specification deviations"
    )

    @root_validator
    def calculate_wobbe_index(cls, values):
        """Calculate Wobbe Index if HHV and SG available."""
        hhv = values.get("higher_heating_value_mj_m3")
        sg = values.get("specific_gravity")

        if hhv and sg and sg > 0 and values.get("wobbe_index") is None:
            values["wobbe_index"] = hhv / (sg ** 0.5)

        return values


class MaintenanceHistoryRecord(BaseModel):
    """
    Historical maintenance record for pattern analysis.

    Captures past maintenance activities, findings, and outcomes
    used to improve prediction accuracy.

    Attributes:
        maintenance_date: Date of maintenance activity
        maintenance_type: Type of maintenance performed
        failure_mode: Failure mode if corrective maintenance
        findings: Maintenance findings and observations
        downtime_hours: Equipment downtime duration

    Example:
        >>> record = MaintenanceHistoryRecord(
        ...     burner_id="BNR-001",
        ...     maintenance_date=datetime(2024, 6, 15),
        ...     maintenance_type=MaintenanceType.REPAIR,
        ...     failure_mode=FailureMode.NOZZLE_DEGRADATION,
        ...     components_replaced=["nozzle"],
        ... )
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier"
    )
    burner_id: str = Field(
        ...,
        description="Burner identifier"
    )
    maintenance_date: datetime = Field(
        ...,
        description="Date of maintenance"
    )
    maintenance_type: MaintenanceType = Field(
        ...,
        description="Type of maintenance"
    )

    # Work details
    work_order_number: Optional[str] = Field(
        default=None,
        description="CMMS work order number"
    )
    work_description: str = Field(
        default="",
        description="Description of work performed"
    )
    failure_mode: Optional[FailureMode] = Field(
        default=None,
        description="Failure mode (if corrective)"
    )
    failure_cause: Optional[str] = Field(
        default=None,
        description="Root cause of failure"
    )

    # Components
    components_inspected: List[str] = Field(
        default_factory=list,
        description="Components inspected"
    )
    components_repaired: List[str] = Field(
        default_factory=list,
        description="Components repaired"
    )
    components_replaced: List[str] = Field(
        default_factory=list,
        description="Components replaced"
    )
    parts_used: List[str] = Field(
        default_factory=list,
        description="Parts/materials used"
    )

    # Findings
    findings: str = Field(
        default="",
        description="Maintenance findings"
    )
    condition_before: str = Field(
        default="",
        description="Condition before maintenance"
    )
    condition_after: str = Field(
        default="",
        description="Condition after maintenance"
    )

    # Metrics
    operating_hours_at_maintenance: float = Field(
        default=0.0,
        ge=0,
        description="Operating hours at time of maintenance"
    )
    cycles_at_maintenance: int = Field(
        default=0,
        ge=0,
        description="Start/stop cycles at maintenance"
    )
    downtime_hours: float = Field(
        default=0.0,
        ge=0,
        description="Equipment downtime (hours)"
    )
    labor_hours: float = Field(
        default=0.0,
        ge=0,
        description="Labor hours for maintenance"
    )
    total_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total maintenance cost ($)"
    )

    # Quality
    was_unplanned: bool = Field(
        default=False,
        description="Unplanned/emergency maintenance"
    )
    caused_production_loss: bool = Field(
        default=False,
        description="Caused production loss"
    )
    warranty_claim: bool = Field(
        default=False,
        description="Warranty claim filed"
    )
    technician_id: Optional[str] = Field(
        default=None,
        description="Technician identifier"
    )

    class Config:
        use_enum_values = True


class BurnerSensorReading(BaseModel):
    """Individual sensor reading from burner monitoring system."""

    sensor_id: str = Field(..., description="Sensor identifier")
    sensor_type: str = Field(..., description="Sensor type")
    value: float = Field(..., description="Sensor value")
    unit: str = Field(..., description="Engineering unit")
    quality: str = Field(default="good", description="Data quality flag")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class BurnerSensorData(BaseModel):
    """
    Real-time sensor readings from burner monitoring systems.

    Aggregates all sensor data for comprehensive burner monitoring
    including temperature, pressure, flow, and emissions.

    Attributes:
        burner_id: Burner identifier
        readings: List of individual sensor readings
        fuel_gas_pressure_psig: Fuel gas pressure
        fuel_gas_temperature_c: Fuel gas temperature
        combustion_air_pressure_inwc: Combustion air pressure

    Example:
        >>> sensors = BurnerSensorData(
        ...     burner_id="BNR-001",
        ...     fuel_gas_pressure_psig=8.5,
        ...     combustion_air_pressure_inwc=3.2,
        ...     stack_oxygen_pct=3.5,
        ... )
    """

    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Individual readings (optional for flexibility)
    readings: List[BurnerSensorReading] = Field(
        default_factory=list,
        description="Raw sensor readings"
    )

    # Fuel system
    fuel_gas_pressure_psig: Optional[float] = Field(
        default=None,
        description="Fuel gas pressure (psig)"
    )
    fuel_gas_temperature_c: Optional[float] = Field(
        default=None,
        description="Fuel gas temperature (C)"
    )
    fuel_oil_pressure_psig: Optional[float] = Field(
        default=None,
        description="Fuel oil pressure (psig)"
    )
    fuel_oil_temperature_c: Optional[float] = Field(
        default=None,
        description="Fuel oil temperature (C)"
    )
    fuel_flow_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fuel flow rate (native units)"
    )
    fuel_valve_position_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Fuel valve position (%)"
    )

    # Combustion air
    combustion_air_pressure_inwc: Optional[float] = Field(
        default=None,
        description="Combustion air pressure (in. W.C.)"
    )
    combustion_air_temperature_c: Optional[float] = Field(
        default=None,
        description="Combustion air temperature (C)"
    )
    air_damper_position_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Air damper position (%)"
    )
    fgr_valve_position_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="FGR valve position (%)"
    )
    blower_speed_rpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Combustion air blower speed (RPM)"
    )

    # Firebox/furnace
    firebox_pressure_inwc: Optional[float] = Field(
        default=None,
        description="Firebox pressure (in. W.C.)"
    )
    firebox_temperature_c: Optional[float] = Field(
        default=None,
        description="Firebox temperature (C)"
    )
    radiant_section_temp_c: Optional[float] = Field(
        default=None,
        description="Radiant section temperature (C)"
    )
    convection_section_temp_c: Optional[float] = Field(
        default=None,
        description="Convection section temperature (C)"
    )

    # Stack/flue gas
    stack_temperature_c: Optional[float] = Field(
        default=None,
        description="Stack temperature (C)"
    )
    stack_oxygen_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=21,
        description="Stack O2 (%)"
    )
    stack_co_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Stack CO (ppm)"
    )
    stack_nox_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Stack NOx (ppm)"
    )
    stack_co2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=25,
        description="Stack CO2 (%)"
    )

    # Flame safeguard
    flame_signal_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Flame signal strength (%)"
    )
    uv_signal_strength: Optional[float] = Field(
        default=None,
        ge=0,
        description="UV flame detector signal"
    )
    ir_signal_strength: Optional[float] = Field(
        default=None,
        ge=0,
        description="IR flame detector signal"
    )
    flame_rod_current_ua: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flame rod ionization current (uA)"
    )

    # BMS status
    bms_status: str = Field(
        default="normal",
        description="BMS status (normal, alarm, lockout)"
    )
    active_interlocks: List[str] = Field(
        default_factory=list,
        description="List of active safety interlocks"
    )

    # Data quality
    data_quality_score: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Overall data quality score"
    )
    sensors_in_alarm: List[str] = Field(
        default_factory=list,
        description="Sensors currently in alarm"
    )


# =============================================================================
# INPUT AGGREGATION
# =============================================================================

class GL021Input(BaseModel):
    """
    Complete input data for GL-021 BURNERSENTRY analysis.

    Aggregates all input data types for comprehensive maintenance
    prediction analysis.

    Attributes:
        burner_id: Burner identifier
        operating_data: Operating hours and cycles
        flame_characteristics: Flame quality measurements
        fuel_quality: Fuel quality data
        sensor_data: Real-time sensor readings
        maintenance_history: Historical maintenance records

    Example:
        >>> input_data = GL021Input(
        ...     burner_id="BNR-001",
        ...     operating_data=operating_data,
        ...     flame_characteristics=flame_data,
        ...     fuel_quality=fuel_data,
        ... )
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    burner_id: str = Field(
        ...,
        description="Burner identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )

    # Core inputs
    operating_data: BurnerOperatingData = Field(
        ...,
        description="Operating data"
    )
    flame_characteristics: Optional[FlameCharacteristics] = Field(
        default=None,
        description="Flame characteristics"
    )
    fuel_quality: Optional[FuelQualityData] = Field(
        default=None,
        description="Fuel quality data"
    )
    sensor_data: Optional[BurnerSensorData] = Field(
        default=None,
        description="Real-time sensor data"
    )

    # Historical data
    maintenance_history: List[MaintenanceHistoryRecord] = Field(
        default_factory=list,
        description="Maintenance history records"
    )

    # Analysis options
    include_economic_analysis: bool = Field(
        default=True,
        description="Include replacement economics"
    )
    include_ml_predictions: bool = Field(
        default=True,
        description="Include ML failure predictions"
    )
    generate_work_orders: bool = Field(
        default=False,
        description="Generate CMMS work orders"
    )

    @root_validator
    def validate_burner_id_consistency(cls, values):
        """Ensure burner_id is consistent across inputs."""
        burner_id = values.get("burner_id")
        operating_data = values.get("operating_data")

        if operating_data and operating_data.burner_id != burner_id:
            raise ValueError(
                f"Burner ID mismatch: input={burner_id}, "
                f"operating_data={operating_data.burner_id}"
            )

        return values


# =============================================================================
# OUTPUT SCHEMAS - HEALTH ASSESSMENT
# =============================================================================

class BurnerComponentHealth(BaseModel):
    """Health assessment for individual burner components."""

    component: ComponentType = Field(..., description="Component type")
    component_id: str = Field(default="", description="Component identifier")
    health_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Component health score (0-100)"
    )
    condition: str = Field(
        default="good",
        description="Condition assessment (excellent, good, fair, poor, critical)"
    )
    degradation_mode: Optional[DegradationMode] = Field(
        default=None,
        description="Primary degradation mode"
    )
    degradation_rate_pct_per_1000h: Optional[float] = Field(
        default=None,
        ge=0,
        description="Degradation rate (% per 1000 hours)"
    )
    rul_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated remaining useful life (hours)"
    )
    rul_confidence: PredictionConfidence = Field(
        default=PredictionConfidence.MEDIUM,
        description="RUL prediction confidence"
    )
    last_replacement_date: Optional[datetime] = Field(
        default=None,
        description="Last replacement date"
    )
    hours_since_replacement: Optional[float] = Field(
        default=None,
        ge=0,
        description="Operating hours since replacement"
    )
    failure_probability_30d: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="30-day failure probability"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended maintenance action"
    )
    notes: str = Field(default="", description="Additional notes")


class BurnerHealthScore(BaseModel):
    """
    Overall burner health assessment with component breakdown.

    Provides a 0-100 health score with detailed component-level
    analysis and trend tracking.

    Attributes:
        overall_score: Overall health score (0-100)
        health_status: Health status classification
        component_scores: Individual component health scores
        trend: Health trend direction

    Example:
        >>> health = BurnerHealthScore(
        ...     burner_id="BNR-001",
        ...     overall_score=85.0,
        ...     health_status="good",
        ...     trend=HealthTrend.STABLE,
        ... )
    """

    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Assessment timestamp"
    )

    # Overall assessment
    overall_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall health score (0-100)"
    )
    health_status: str = Field(
        ...,
        description="Health status (excellent, good, fair, poor, critical)"
    )
    trend: HealthTrend = Field(
        default=HealthTrend.STABLE,
        description="Health trend direction"
    )
    trend_rate_pct_per_week: Optional[float] = Field(
        default=None,
        description="Health score change rate (% per week)"
    )

    # Component breakdown
    component_scores: List[BurnerComponentHealth] = Field(
        default_factory=list,
        description="Component health assessments"
    )
    limiting_component: Optional[ComponentType] = Field(
        default=None,
        description="Component limiting overall health"
    )

    # Score breakdown by category
    flame_quality_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Flame quality contribution to score"
    )
    fuel_system_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Fuel system health score"
    )
    air_system_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Air system health score"
    )
    controls_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Controls and instrumentation score"
    )
    safety_system_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Safety system health score"
    )

    # Risk assessment
    overall_risk_level: str = Field(
        default="low",
        description="Overall risk level (low, medium, high, critical)"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Active risk factors"
    )

    # Comparison
    score_percentile: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Score percentile vs fleet"
    )
    score_vs_baseline_pct: Optional[float] = Field(
        default=None,
        description="Score vs baseline (%)"
    )

    @validator("health_status", pre=True, always=True)
    def derive_health_status(cls, v, values):
        """Derive health status from score if not provided."""
        if v:
            return v
        score = values.get("overall_score", 100)
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "critical"


# =============================================================================
# OUTPUT SCHEMAS - PREDICTIONS
# =============================================================================

class FailurePredictionItem(BaseModel):
    """Individual failure mode prediction."""

    failure_mode: FailureMode = Field(..., description="Predicted failure mode")
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Failure probability"
    )
    confidence: PredictionConfidence = Field(
        default=PredictionConfidence.MEDIUM,
        description="Prediction confidence"
    )
    time_to_failure_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated time to failure (hours)"
    )
    time_to_failure_lower_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Lower bound (P10)"
    )
    time_to_failure_upper_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Upper bound (P90)"
    )
    affected_components: List[ComponentType] = Field(
        default_factory=list,
        description="Components at risk"
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to prediction"
    )
    recommended_action: str = Field(
        default="",
        description="Recommended preventive action"
    )
    model_id: str = Field(default="", description="Model identifier")


class MaintenancePrediction(BaseModel):
    """
    Comprehensive maintenance prediction output.

    Provides failure predictions, RUL estimates, and maintenance
    recommendations based on analysis.

    Attributes:
        rul_hours: Estimated remaining useful life
        failure_predictions: List of failure mode predictions
        next_maintenance_type: Recommended next maintenance type
        next_maintenance_date: Recommended maintenance date

    Example:
        >>> prediction = MaintenancePrediction(
        ...     burner_id="BNR-001",
        ...     rul_hours=5000.0,
        ...     rul_confidence=PredictionConfidence.HIGH,
        ...     overall_failure_probability_30d=0.05,
        ... )
    """

    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Prediction timestamp"
    )

    # RUL estimates
    rul_hours: float = Field(
        ...,
        ge=0,
        description="Estimated RUL (hours)"
    )
    rul_p10_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="RUL at P10 (optimistic)"
    )
    rul_p50_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="RUL at P50 (median)"
    )
    rul_p90_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="RUL at P90 (conservative)"
    )
    rul_confidence: PredictionConfidence = Field(
        default=PredictionConfidence.MEDIUM,
        description="RUL prediction confidence"
    )

    # Failure predictions
    failure_predictions: List[FailurePredictionItem] = Field(
        default_factory=list,
        description="Failure mode predictions"
    )
    highest_risk_failure_mode: Optional[FailureMode] = Field(
        default=None,
        description="Highest risk failure mode"
    )
    overall_failure_probability_30d: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="30-day failure probability"
    )
    overall_failure_probability_90d: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="90-day failure probability"
    )

    # Maintenance recommendations
    next_maintenance_type: MaintenanceType = Field(
        default=MaintenanceType.INSPECTION,
        description="Recommended next maintenance type"
    )
    next_maintenance_date: Optional[date] = Field(
        default=None,
        description="Recommended next maintenance date"
    )
    next_maintenance_priority: MaintenancePriority = Field(
        default=MaintenancePriority.ROUTINE,
        description="Next maintenance priority"
    )
    maintenance_window_start: Optional[date] = Field(
        default=None,
        description="Start of optimal maintenance window"
    )
    maintenance_window_end: Optional[date] = Field(
        default=None,
        description="End of optimal maintenance window"
    )

    # Analysis metadata
    analysis_methods: List[str] = Field(
        default_factory=list,
        description="Analysis methods used"
    )
    data_quality_score: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Input data quality score"
    )
    model_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Model versions used"
    )


# =============================================================================
# OUTPUT SCHEMAS - MAINTENANCE SCHEDULING
# =============================================================================

class MaintenanceTask(BaseModel):
    """Individual scheduled maintenance task."""

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Task identifier"
    )
    task_name: str = Field(..., description="Task name")
    task_type: MaintenanceType = Field(..., description="Task type")
    priority: MaintenancePriority = Field(
        default=MaintenancePriority.ROUTINE,
        description="Task priority"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Task status"
    )
    due_date: date = Field(..., description="Due date")
    due_operating_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Due at operating hours"
    )
    components: List[ComponentType] = Field(
        default_factory=list,
        description="Components involved"
    )
    description: str = Field(default="", description="Task description")
    estimated_duration_hours: float = Field(
        default=4.0,
        ge=0,
        description="Estimated duration (hours)"
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated cost ($)"
    )
    parts_required: List[str] = Field(
        default_factory=list,
        description="Parts required"
    )
    skills_required: List[str] = Field(
        default_factory=list,
        description="Skills required"
    )
    safety_requirements: List[str] = Field(
        default_factory=list,
        description="Safety requirements"
    )
    procedure_reference: Optional[str] = Field(
        default=None,
        description="Reference procedure document"
    )


class ReplacementCandidate(BaseModel):
    """Candidate component for replacement planning."""

    component: ComponentType = Field(..., description="Component type")
    component_id: str = Field(default="", description="Component ID")
    current_health_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current health (%)"
    )
    rul_hours: float = Field(
        ...,
        ge=0,
        description="Estimated RUL (hours)"
    )
    optimal_replacement_date: date = Field(
        ...,
        description="Optimal replacement date"
    )
    replacement_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Replacement cost ($)"
    )
    failure_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Cost if run to failure ($)"
    )
    cost_savings_usd: float = Field(
        default=0.0,
        description="Savings from planned replacement ($)"
    )
    replacement_urgency: str = Field(
        default="routine",
        description="Urgency level"
    )
    lead_time_days: int = Field(
        default=14,
        ge=0,
        description="Parts lead time (days)"
    )


class ReplacementSchedule(BaseModel):
    """
    Burner replacement planning output.

    Provides optimal replacement timing with economic analysis
    and lifecycle cost optimization.

    Attributes:
        recommended_action: Recommended replacement action
        optimal_replacement_date: Economically optimal date
        total_lifecycle_cost_usd: Total lifecycle cost
        npv_savings_usd: NPV savings from optimal timing

    Example:
        >>> schedule = ReplacementSchedule(
        ...     burner_id="BNR-001",
        ...     recommended_action="schedule_overhaul",
        ...     optimal_replacement_date=date(2025, 6, 1),
        ...     total_lifecycle_cost_usd=75000.0,
        ... )
    """

    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Overall recommendation
    recommended_action: str = Field(
        ...,
        description="Recommended action (continue_operation, schedule_maintenance, "
                    "schedule_overhaul, plan_replacement, immediate_replacement)"
    )
    recommendation_rationale: str = Field(
        default="",
        description="Rationale for recommendation"
    )

    # Timing recommendations
    optimal_replacement_date: Optional[date] = Field(
        default=None,
        description="Optimal replacement date"
    )
    latest_safe_operation_date: Optional[date] = Field(
        default=None,
        description="Latest date for safe operation"
    )
    maintenance_window: Optional[Tuple[date, date]] = Field(
        default=None,
        description="Recommended maintenance window"
    )

    # Component replacements
    replacement_candidates: List[ReplacementCandidate] = Field(
        default_factory=list,
        description="Components recommended for replacement"
    )

    # Economic analysis
    total_lifecycle_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total lifecycle cost ($)"
    )
    annual_operating_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual operating cost ($)"
    )
    npv_savings_usd: float = Field(
        default=0.0,
        description="NPV savings from optimal timing ($)"
    )
    roi_pct: Optional[float] = Field(
        default=None,
        description="Return on investment (%)"
    )
    payback_months: Optional[float] = Field(
        default=None,
        ge=0,
        description="Payback period (months)"
    )

    # Risk assessment
    risk_of_deferral: str = Field(
        default="low",
        description="Risk of deferring replacement"
    )
    failure_cost_if_deferred_usd: float = Field(
        default=0.0,
        ge=0,
        description="Expected failure cost if deferred ($)"
    )

    # Lead time management
    procurement_start_date: Optional[date] = Field(
        default=None,
        description="Start procurement by this date"
    )
    critical_spares_available: bool = Field(
        default=True,
        description="Critical spares in inventory"
    )
    spares_to_order: List[str] = Field(
        default_factory=list,
        description="Spares to order"
    )


# =============================================================================
# OUTPUT SCHEMAS - WORK ORDERS
# =============================================================================

class WorkOrder(BaseModel):
    """
    CMMS work order for maintenance activities.

    Formatted for integration with SAP PM, IBM Maximo, and other
    CMMS systems.

    Attributes:
        work_order_number: Work order number
        title: Work order title
        priority: Work order priority
        due_date: Required completion date
        estimated_cost_usd: Estimated total cost

    Example:
        >>> wo = WorkOrder(
        ...     burner_id="BNR-001",
        ...     title="Replace burner nozzle - degradation detected",
        ...     priority=MaintenancePriority.HIGH,
        ...     due_date=date(2024, 12, 15),
        ... )
    """

    work_order_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Internal work order ID"
    )
    work_order_number: Optional[str] = Field(
        default=None,
        description="CMMS work order number"
    )
    burner_id: str = Field(..., description="Burner identifier")
    equipment_tag: str = Field(default="", description="Plant equipment tag")
    created_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    # Work order details
    title: str = Field(
        ...,
        max_length=100,
        description="Work order title"
    )
    description: str = Field(
        ...,
        max_length=2000,
        description="Detailed description"
    )
    work_type: MaintenanceType = Field(..., description="Work type")
    priority: MaintenancePriority = Field(
        default=MaintenancePriority.MEDIUM,
        description="Priority"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Status"
    )

    # Scheduling
    due_date: date = Field(..., description="Due date")
    required_by_datetime: Optional[datetime] = Field(
        default=None,
        description="Required completion datetime"
    )
    scheduled_start: Optional[datetime] = Field(
        default=None,
        description="Scheduled start"
    )
    scheduled_end: Optional[datetime] = Field(
        default=None,
        description="Scheduled end"
    )

    # Scope
    failure_modes: List[FailureMode] = Field(
        default_factory=list,
        description="Related failure modes"
    )
    components: List[ComponentType] = Field(
        default_factory=list,
        description="Components to address"
    )
    tasks: List[str] = Field(
        default_factory=list,
        description="Task list"
    )

    # Resources
    parts_required: List[str] = Field(
        default_factory=list,
        description="Parts required"
    )
    tools_required: List[str] = Field(
        default_factory=list,
        description="Tools required"
    )
    skills_required: List[str] = Field(
        default_factory=list,
        description="Skills required"
    )
    estimated_labor_hours: float = Field(
        default=4.0,
        ge=0,
        description="Estimated labor hours"
    )
    crew_size: int = Field(
        default=2,
        ge=1,
        description="Required crew size"
    )

    # Costs
    estimated_labor_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated labor cost ($)"
    )
    estimated_parts_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated parts cost ($)"
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total estimated cost ($)"
    )

    # Safety
    safety_requirements: List[str] = Field(
        default_factory=list,
        description="Safety requirements"
    )
    permits_required: List[str] = Field(
        default_factory=list,
        description="Permits required"
    )
    lockout_tagout_required: bool = Field(
        default=True,
        description="LOTO required"
    )
    hot_work_permit_required: bool = Field(
        default=False,
        description="Hot work permit required"
    )
    confined_space_entry: bool = Field(
        default=False,
        description="Confined space entry required"
    )

    # References
    procedure_references: List[str] = Field(
        default_factory=list,
        description="Procedure document references"
    )
    drawing_references: List[str] = Field(
        default_factory=list,
        description="Drawing references"
    )

    # CMMS integration
    plant_code: str = Field(default="", description="Plant code")
    work_center: str = Field(default="", description="Work center")
    cost_center: str = Field(default="", description="Cost center")

    # Provenance
    source_analysis_id: str = Field(
        default="",
        description="Source analysis request ID"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# OUTPUT SCHEMAS - ANALYSIS RESULTS
# =============================================================================

class FlameAnalysisResult(BaseModel):
    """Detailed flame analysis result."""

    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    flame_status: FlameStatus = Field(..., description="Overall flame status")
    quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Flame quality score"
    )

    # Individual metrics
    temperature_status: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Temperature status"
    )
    stability_status: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Stability status"
    )
    shape_status: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Shape status"
    )
    color_status: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Color status"
    )

    # Anomalies detected
    anomalies_detected: List[str] = Field(
        default_factory=list,
        description="Detected anomalies"
    )
    impingement_risk: bool = Field(
        default=False,
        description="Impingement risk identified"
    )
    incomplete_combustion_indicators: List[str] = Field(
        default_factory=list,
        description="Incomplete combustion indicators"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )


class FuelQualityAnalysisResult(BaseModel):
    """Fuel quality impact analysis result."""

    fuel_id: str = Field(..., description="Fuel identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    quality_grade: str = Field(
        default="standard",
        description="Quality grade"
    )
    within_specification: bool = Field(
        default=True,
        description="Within specification"
    )

    # Impact assessment
    nozzle_degradation_impact_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Impact on nozzle life (%)"
    )
    refractory_degradation_impact_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Impact on refractory life (%)"
    )
    efficiency_impact_pct: float = Field(
        default=0.0,
        description="Impact on efficiency (%)"
    )
    emissions_impact_pct: float = Field(
        default=0.0,
        description="Impact on emissions (%)"
    )

    # Out-of-spec parameters
    out_of_spec_parameters: List[str] = Field(
        default_factory=list,
        description="Out-of-spec parameters"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )


class WeibullRULResult(BaseModel):
    """Weibull analysis result for RUL estimation."""

    component: str = Field(default="overall", description="Component analyzed")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Weibull parameters
    beta: float = Field(..., gt=0, description="Shape parameter")
    eta_hours: float = Field(..., gt=0, description="Scale parameter (hours)")
    gamma_hours: float = Field(default=0.0, ge=0, description="Location parameter")

    # RUL estimates
    current_age_hours: float = Field(..., ge=0, description="Current age (hours)")
    rul_p10_hours: float = Field(..., description="RUL at P10")
    rul_p50_hours: float = Field(..., description="RUL at P50 (median)")
    rul_p90_hours: float = Field(..., description="RUL at P90")

    # Failure probabilities
    current_failure_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Current cumulative failure probability"
    )
    failure_probability_30d: float = Field(
        ...,
        ge=0,
        le=1,
        description="30-day failure probability"
    )
    failure_probability_90d: float = Field(
        ...,
        ge=0,
        le=1,
        description="90-day failure probability"
    )

    # Interpretation
    failure_mode_interpretation: str = Field(
        default="",
        description="Beta interpretation (early_life, random, wear_out)"
    )
    confidence_level: float = Field(
        default=0.90,
        ge=0.50,
        le=0.99,
        description="Confidence level"
    )

    # Fit quality
    r_squared: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="R-squared of fit"
    )
    n_data_points: int = Field(default=0, ge=0, description="Data points used")

    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash"
    )


class EconomicAnalysisResult(BaseModel):
    """Economic analysis for replacement planning."""

    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    analysis_horizon_months: int = Field(
        default=24,
        ge=6,
        description="Analysis horizon"
    )

    # Cost breakdown
    replacement_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Replacement cost"
    )
    planned_maintenance_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Planned maintenance cost"
    )
    unplanned_failure_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Expected unplanned failure cost"
    )
    downtime_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Downtime cost"
    )
    efficiency_loss_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Efficiency loss cost"
    )
    total_cost_of_ownership_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total cost of ownership"
    )

    # NPV analysis
    npv_replace_now_usd: float = Field(
        default=0.0,
        description="NPV if replace now"
    )
    npv_replace_optimal_usd: float = Field(
        default=0.0,
        description="NPV if replace at optimal time"
    )
    npv_run_to_failure_usd: float = Field(
        default=0.0,
        description="NPV if run to failure"
    )
    optimal_strategy: str = Field(
        default="",
        description="Optimal strategy"
    )
    savings_vs_rtf_usd: float = Field(
        default=0.0,
        description="Savings vs run-to-failure"
    )

    # Assumptions
    discount_rate_pct: float = Field(default=10.0, description="Discount rate")
    fuel_cost_usd_mmbtu: float = Field(default=5.0, description="Fuel cost")
    downtime_cost_usd_hr: float = Field(
        default=5000.0,
        description="Downtime cost per hour"
    )


# =============================================================================
# MAIN OUTPUT SCHEMA
# =============================================================================

class GL021Result(BaseModel):
    """
    Comprehensive GL-021 BURNERSENTRY analysis result.

    Aggregates all analysis outputs including health assessment,
    maintenance predictions, replacement schedule, and work orders.

    Attributes:
        burner_id: Burner identifier
        health_score: Overall burner health score
        maintenance_prediction: Maintenance predictions
        replacement_schedule: Replacement planning results
        work_orders: Generated work orders
        provenance_hash: SHA-256 provenance hash

    Example:
        >>> result = GL021Result(
        ...     request_id="req-001",
        ...     burner_id="BNR-001",
        ...     health_score=health_score,
        ...     maintenance_prediction=prediction,
        ... )
        >>> print(f"Health: {result.health_score.overall_score}/100")
        >>> print(f"RUL: {result.maintenance_prediction.rul_hours} hours")
    """

    request_id: str = Field(..., description="Request identifier")
    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )
    status: str = Field(
        default="success",
        description="Analysis status"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )

    # Core results
    health_score: BurnerHealthScore = Field(
        ...,
        description="Health assessment"
    )
    maintenance_prediction: MaintenancePrediction = Field(
        ...,
        description="Maintenance predictions"
    )

    # Optional detailed results
    flame_analysis: Optional[FlameAnalysisResult] = Field(
        default=None,
        description="Flame analysis result"
    )
    fuel_quality_analysis: Optional[FuelQualityAnalysisResult] = Field(
        default=None,
        description="Fuel quality analysis"
    )
    weibull_analysis: Optional[WeibullRULResult] = Field(
        default=None,
        description="Weibull RUL analysis"
    )
    economic_analysis: Optional[EconomicAnalysisResult] = Field(
        default=None,
        description="Economic analysis"
    )
    replacement_schedule: Optional[ReplacementSchedule] = Field(
        default=None,
        description="Replacement schedule"
    )

    # Generated outputs
    maintenance_schedule: List[MaintenanceTask] = Field(
        default_factory=list,
        description="Scheduled maintenance tasks"
    )
    work_orders: List[WorkOrder] = Field(
        default_factory=list,
        description="Generated work orders"
    )

    # Alerts and recommendations
    active_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Key recommendations"
    )

    # Summary metrics
    next_maintenance_date: Optional[date] = Field(
        default=None,
        description="Next maintenance date"
    )
    rul_hours: float = Field(
        default=0.0,
        ge=0,
        description="Estimated RUL (hours)"
    )
    failure_probability_30d: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="30-day failure probability"
    )

    # Metadata
    analysis_methods: List[str] = Field(
        default_factory=list,
        description="Analysis methods used"
    )
    data_quality_score: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Data quality score"
    )
    model_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Model versions"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Analysis warnings"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }


# =============================================================================
# UPDATE FORWARD REFERENCES
# =============================================================================

GL021Input.update_forward_refs()
GL021Result.update_forward_refs()
BurnerHealthScore.update_forward_refs()
MaintenancePrediction.update_forward_refs()
ReplacementSchedule.update_forward_refs()
