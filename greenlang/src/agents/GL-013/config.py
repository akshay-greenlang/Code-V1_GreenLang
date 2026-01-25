# -*- coding: utf-8 -*-
"""
Configuration module for PredictMaint agent (GL-013 PREDICTMAINT).

This module defines the configuration models and settings for the
PredictMaint agent, including equipment specifications, sensor configurations,
failure mode analysis, maintenance scheduling, CMMS integration, machine
learning model parameters, and alert management.

The PredictMaint agent provides predictive maintenance capabilities by:
- Monitoring equipment health through vibration analysis (ISO 10816)
- Tracking temperature, pressure, and operating parameters
- Predicting remaining useful life (RUL) using ML models
- Scheduling maintenance based on condition monitoring
- Integrating with CMMS systems (SAP PM, Maximo, etc.)
- Managing spare parts inventory optimization

Standards Compliance:
- ISO 10816 - Mechanical Vibration Evaluation
- ISO 13373 - Condition Monitoring and Diagnostics
- ISO 13374 - Data Processing, Communication and Presentation
- ISO 13379 - Prognostics and Health Management
- ISO 13381 - Remaining Useful Life Estimation
- ISO 17359 - Condition Monitoring General Guidelines
- ISO 55000 - Asset Management
- Pydantic V2 for validation

Author: GreenLang Industrial Optimization Team
Date: December 2025
Agent ID: GL-013
Agent Name: PredictMaint
Domain: Predictive Maintenance
Type: Analyzer/Predictor
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# ENUMERATIONS
# ============================================================================

class EquipmentType(str, Enum):
    """Equipment type classification for predictive maintenance."""
    PUMP = "pump"
    MOTOR = "motor"
    COMPRESSOR = "compressor"
    FAN = "fan"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    CONVEYOR = "conveyor"
    TURBINE = "turbine"
    HEAT_EXCHANGER = "heat_exchanger"
    VALVE = "valve"
    GENERATOR = "generator"
    BLOWER = "blower"
    CENTRIFUGE = "centrifuge"
    AGITATOR = "agitator"
    CRUSHER = "crusher"


class FailureMode(str, Enum):
    """Failure mode classification per FMEA standards."""
    BEARING_WEAR = "bearing_wear"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    LOOSENESS = "looseness"
    LUBRICATION_FAILURE = "lubrication_failure"
    ELECTRICAL_FAULT = "electrical_fault"
    THERMAL_DEGRADATION = "thermal_degradation"
    CAVITATION = "cavitation"
    CORROSION = "corrosion"
    FATIGUE = "fatigue"
    SEAL_FAILURE = "seal_failure"
    GEAR_MESH_FAULT = "gear_mesh_fault"
    BELT_WEAR = "belt_wear"
    COUPLING_FAULT = "coupling_fault"
    RESONANCE = "resonance"
    OIL_CONTAMINATION = "oil_contamination"
    OVERLOAD = "overload"
    STARVATION = "starvation"


class MaintenanceStrategy(str, Enum):
    """Maintenance strategy classification per ISO 55000."""
    REACTIVE = "reactive"  # Run-to-failure
    PREVENTIVE = "preventive"  # Time-based maintenance
    PREDICTIVE = "predictive"  # Condition-based maintenance
    PRESCRIPTIVE = "prescriptive"  # AI-optimized maintenance
    RELIABILITY_CENTERED = "reliability_centered"  # RCM approach


class HealthStatus(str, Enum):
    """Equipment health status classification."""
    HEALTHY = "healthy"  # Normal operation, no anomalies
    MONITORED = "monitored"  # Watching closely, minor deviations
    DEGRADED = "degraded"  # Showing wear, maintenance recommended
    AT_RISK = "at_risk"  # High failure probability, plan maintenance
    CRITICAL = "critical"  # Imminent failure, immediate action required
    FAILED = "failed"  # Out of service, repair required


class AlertSeverity(str, Enum):
    """Alert severity levels per IEC 62682."""
    INFO = "info"  # Informational only
    WARNING = "warning"  # Attention needed
    HIGH = "high"  # Significant issue
    CRITICAL = "critical"  # Urgent action required
    EMERGENCY = "emergency"  # Immediate shutdown required


class SensorType(str, Enum):
    """Sensor type classification for condition monitoring."""
    ACCELEROMETER = "accelerometer"
    VELOCITY_SENSOR = "velocity_sensor"
    DISPLACEMENT_PROBE = "displacement_probe"
    RTD = "rtd"
    THERMOCOUPLE = "thermocouple"
    INFRARED = "infrared"
    PRESSURE_TRANSMITTER = "pressure_transmitter"
    FLOW_METER = "flow_meter"
    CURRENT_TRANSFORMER = "current_transformer"
    ULTRASONIC = "ultrasonic"
    OIL_PARTICLE_COUNTER = "oil_particle_counter"
    MOISTURE_SENSOR = "moisture_sensor"


class VibrationUnit(str, Enum):
    """Vibration measurement units per ISO 10816."""
    MM_S = "mm/s"  # Velocity in mm/s RMS
    IN_S = "in/s"  # Velocity in in/s peak
    G = "g"  # Acceleration in g RMS
    M_S2 = "m/s2"  # Acceleration in m/s^2
    MICRON = "micron"  # Displacement in microns peak-to-peak
    MIL = "mil"  # Displacement in mils peak-to-peak


class MachineClass(str, Enum):
    """Machine classification per ISO 10816-3."""
    CLASS_I = "class_i"  # Small machines up to 15 kW
    CLASS_II = "class_ii"  # Medium machines 15-75 kW without special foundations
    CLASS_III = "class_iii"  # Large machines on rigid foundations
    CLASS_IV = "class_iv"  # Large machines on flexible foundations


class MountingType(str, Enum):
    """Machine mounting type classification."""
    RIGID = "rigid"  # Rigid foundation/base
    FLEXIBLE = "flexible"  # Flexible foundation/isolators
    SOFT_FOOT = "soft_foot"  # Soft foot condition
    SUSPENDED = "suspended"  # Suspended/hanging mount


class CMMSType(str, Enum):
    """CMMS system type for integration."""
    SAP_PM = "sap_pm"
    MAXIMO = "maximo"
    INFOR_EAM = "infor_eam"
    ORACLE_EAM = "oracle_eam"
    FIIX = "fiix"
    UPTIMEKS = "uptimeks"
    MAINTENANCE_CONNECTION = "maintenance_connection"
    EMAINT = "emaint"
    CUSTOM = "custom"


class WorkOrderPriority(str, Enum):
    """Work order priority classification."""
    EMERGENCY = "emergency"  # Immediate action required
    URGENT = "urgent"  # Within 24 hours
    HIGH = "high"  # Within 1 week
    MEDIUM = "medium"  # Within 1 month
    LOW = "low"  # Scheduled maintenance window
    PLANNED = "planned"  # Next scheduled outage


class MLModelType(str, Enum):
    """Machine learning model type for anomaly detection."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    LSTM = "lstm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


class DataQualityLevel(str, Enum):
    """Data quality classification."""
    EXCELLENT = "excellent"  # >99% completeness, <1% outliers
    GOOD = "good"  # >95% completeness, <5% outliers
    ACCEPTABLE = "acceptable"  # >90% completeness, <10% outliers
    POOR = "poor"  # <90% completeness or >10% outliers
    INSUFFICIENT = "insufficient"  # Cannot use for predictions


# ============================================================================
# ISO 10816 VIBRATION LIMITS
# ============================================================================

class ISO10816VibrationLimits(BaseModel):
    """
    Vibration severity limits per ISO 10816-3.

    ISO 10816-3 defines vibration severity zones:
    - Zone A: Good condition (new/recently overhauled)
    - Zone B: Acceptable for unrestricted long-term operation
    - Zone C: Restricted operation, monitoring required
    - Zone D: Unacceptable, damage may occur

    Values are in mm/s RMS (velocity).

    Attributes:
        machine_class: ISO machine classification
        zone_a_limit: Upper limit for Zone A (mm/s RMS)
        zone_b_limit: Upper limit for Zone B (mm/s RMS)
        zone_c_limit: Upper limit for Zone C (mm/s RMS)
        zone_d_above: Values above this are Zone D (mm/s RMS)
    """

    machine_class: MachineClass = Field(
        default=MachineClass.CLASS_II,
        description="ISO 10816-3 machine classification"
    )

    zone_a_limit: float = Field(
        default=2.8,
        ge=0.1,
        le=10.0,
        description="Upper limit for Zone A (Good) in mm/s RMS"
    )
    zone_b_limit: float = Field(
        default=7.1,
        ge=0.5,
        le=20.0,
        description="Upper limit for Zone B (Acceptable) in mm/s RMS"
    )
    zone_c_limit: float = Field(
        default=11.2,
        ge=1.0,
        le=30.0,
        description="Upper limit for Zone C (Restricted) in mm/s RMS"
    )
    zone_d_above: float = Field(
        default=11.2,
        ge=1.0,
        le=50.0,
        description="Values above this are Zone D (Unacceptable) in mm/s RMS"
    )

    @model_validator(mode='after')
    def validate_zone_sequence(self):
        """Ensure zones are in ascending order."""
        if not (self.zone_a_limit < self.zone_b_limit <= self.zone_c_limit):
            raise ValueError(
                "Zone limits must be in ascending order: A < B <= C"
            )
        if self.zone_d_above < self.zone_c_limit:
            raise ValueError(
                "Zone D threshold must be >= Zone C limit"
            )
        return self


# Standard ISO 10816-3 limits by machine class
ISO_10816_STANDARDS: Dict[MachineClass, Dict[str, float]] = {
    MachineClass.CLASS_I: {
        "zone_a_limit": 0.71,
        "zone_b_limit": 1.8,
        "zone_c_limit": 4.5,
        "zone_d_above": 4.5
    },
    MachineClass.CLASS_II: {
        "zone_a_limit": 1.12,
        "zone_b_limit": 2.8,
        "zone_c_limit": 7.1,
        "zone_d_above": 7.1
    },
    MachineClass.CLASS_III: {
        "zone_a_limit": 1.8,
        "zone_b_limit": 4.5,
        "zone_c_limit": 11.2,
        "zone_d_above": 11.2
    },
    MachineClass.CLASS_IV: {
        "zone_a_limit": 2.8,
        "zone_b_limit": 7.1,
        "zone_c_limit": 18.0,
        "zone_d_above": 18.0
    }
}


def get_iso_10816_limits(machine_class: MachineClass) -> ISO10816VibrationLimits:
    """
    Get standard ISO 10816-3 vibration limits for a machine class.

    Args:
        machine_class: ISO machine classification

    Returns:
        ISO10816VibrationLimits with standard values
    """
    standards = ISO_10816_STANDARDS.get(machine_class, ISO_10816_STANDARDS[MachineClass.CLASS_II])
    return ISO10816VibrationLimits(
        machine_class=machine_class,
        **standards
    )


# ============================================================================
# WEIBULL DISTRIBUTION PARAMETERS
# ============================================================================

class WeibullParameters(BaseModel):
    """
    Weibull distribution parameters for reliability analysis.

    The Weibull distribution is commonly used in reliability engineering
    to model time-to-failure data. Parameters are typically determined
    from historical failure data or manufacturer specifications.

    Weibull CDF: F(t) = 1 - exp(-(t/eta)^beta)

    Attributes:
        beta: Shape parameter (beta < 1: infant mortality,
              beta = 1: random, beta > 1: wear-out)
        eta: Scale parameter (characteristic life in hours)
        gamma: Location parameter (minimum life, typically 0)
        mtbf_hours: Mean Time Between Failures
        b10_life_hours: B10 life (10% failure probability)
    """

    equipment_type: EquipmentType = Field(
        ...,
        description="Equipment type for these parameters"
    )
    failure_mode: FailureMode = Field(
        ...,
        description="Failure mode being modeled"
    )

    beta: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Weibull shape parameter (beta)"
    )
    eta: float = Field(
        default=50000.0,
        ge=100.0,
        le=1000000.0,
        description="Weibull scale parameter (eta) in hours"
    )
    gamma: float = Field(
        default=0.0,
        ge=0.0,
        le=100000.0,
        description="Weibull location parameter (gamma) in hours"
    )

    mtbf_hours: float = Field(
        default=40000.0,
        ge=100.0,
        le=1000000.0,
        description="Mean Time Between Failures in hours"
    )
    b10_life_hours: float = Field(
        default=20000.0,
        ge=100.0,
        le=500000.0,
        description="B10 life (10% failure probability) in hours"
    )

    confidence_level: float = Field(
        default=0.90,
        ge=0.5,
        le=0.99,
        description="Statistical confidence level for parameters"
    )

    data_source: str = Field(
        default="manufacturer",
        description="Source of reliability data (manufacturer, field_data, industry_average)"
    )

    @field_validator('data_source')
    @classmethod
    def validate_data_source(cls, v):
        """Validate data source type."""
        valid_sources = {"manufacturer", "field_data", "industry_average", "simulation"}
        if v not in valid_sources:
            raise ValueError(f"Invalid data source. Must be one of: {valid_sources}")
        return v


# Standard Weibull parameters by equipment type and failure mode
WEIBULL_DEFAULTS: Dict[Tuple[EquipmentType, FailureMode], Dict[str, float]] = {
    (EquipmentType.PUMP, FailureMode.BEARING_WEAR): {
        "beta": 2.5, "eta": 45000, "mtbf_hours": 40000, "b10_life_hours": 18000
    },
    (EquipmentType.PUMP, FailureMode.SEAL_FAILURE): {
        "beta": 1.8, "eta": 25000, "mtbf_hours": 22000, "b10_life_hours": 8000
    },
    (EquipmentType.PUMP, FailureMode.CAVITATION): {
        "beta": 3.0, "eta": 35000, "mtbf_hours": 31000, "b10_life_hours": 16000
    },
    (EquipmentType.MOTOR, FailureMode.BEARING_WEAR): {
        "beta": 2.8, "eta": 60000, "mtbf_hours": 53000, "b10_life_hours": 28000
    },
    (EquipmentType.MOTOR, FailureMode.ELECTRICAL_FAULT): {
        "beta": 1.5, "eta": 80000, "mtbf_hours": 72000, "b10_life_hours": 22000
    },
    (EquipmentType.MOTOR, FailureMode.THERMAL_DEGRADATION): {
        "beta": 2.2, "eta": 55000, "mtbf_hours": 49000, "b10_life_hours": 22000
    },
    (EquipmentType.GEARBOX, FailureMode.GEAR_MESH_FAULT): {
        "beta": 3.5, "eta": 70000, "mtbf_hours": 63000, "b10_life_hours": 38000
    },
    (EquipmentType.GEARBOX, FailureMode.BEARING_WEAR): {
        "beta": 2.6, "eta": 50000, "mtbf_hours": 44000, "b10_life_hours": 22000
    },
    (EquipmentType.GEARBOX, FailureMode.OIL_CONTAMINATION): {
        "beta": 1.4, "eta": 30000, "mtbf_hours": 27000, "b10_life_hours": 7000
    },
    (EquipmentType.COMPRESSOR, FailureMode.BEARING_WEAR): {
        "beta": 2.4, "eta": 40000, "mtbf_hours": 35000, "b10_life_hours": 16000
    },
    (EquipmentType.COMPRESSOR, FailureMode.SEAL_FAILURE): {
        "beta": 2.0, "eta": 20000, "mtbf_hours": 17700, "b10_life_hours": 6500
    },
    (EquipmentType.FAN, FailureMode.IMBALANCE): {
        "beta": 2.2, "eta": 55000, "mtbf_hours": 49000, "b10_life_hours": 22000
    },
    (EquipmentType.FAN, FailureMode.BEARING_WEAR): {
        "beta": 2.7, "eta": 65000, "mtbf_hours": 58000, "b10_life_hours": 30000
    },
    (EquipmentType.TURBINE, FailureMode.BEARING_WEAR): {
        "beta": 3.2, "eta": 100000, "mtbf_hours": 89000, "b10_life_hours": 52000
    },
    (EquipmentType.TURBINE, FailureMode.FATIGUE): {
        "beta": 4.0, "eta": 120000, "mtbf_hours": 109000, "b10_life_hours": 72000
    },
    (EquipmentType.CONVEYOR, FailureMode.BELT_WEAR): {
        "beta": 2.0, "eta": 15000, "mtbf_hours": 13300, "b10_life_hours": 4900
    },
    (EquipmentType.CONVEYOR, FailureMode.BEARING_WEAR): {
        "beta": 2.3, "eta": 35000, "mtbf_hours": 31000, "b10_life_hours": 13500
    },
}


def get_weibull_parameters(
    equipment_type: EquipmentType,
    failure_mode: FailureMode
) -> WeibullParameters:
    """
    Get default Weibull parameters for equipment type and failure mode.

    Args:
        equipment_type: Type of equipment
        failure_mode: Failure mode being analyzed

    Returns:
        WeibullParameters with industry-standard defaults
    """
    key = (equipment_type, failure_mode)
    if key in WEIBULL_DEFAULTS:
        params = WEIBULL_DEFAULTS[key]
        return WeibullParameters(
            equipment_type=equipment_type,
            failure_mode=failure_mode,
            **params
        )
    # Return generic defaults if not found
    return WeibullParameters(
        equipment_type=equipment_type,
        failure_mode=failure_mode,
        beta=2.0,
        eta=50000,
        mtbf_hours=44000,
        b10_life_hours=16000
    )


# ============================================================================
# EQUIPMENT CONFIGURATION
# ============================================================================

class EquipmentConfiguration(BaseModel):
    """
    Equipment specification and nameplate data.

    Defines the physical characteristics and operational parameters
    of equipment being monitored for predictive maintenance.

    Attributes:
        equipment_id: Unique equipment identifier
        equipment_tag: P&ID/asset tag
        equipment_name: Human-readable name
        equipment_type: Equipment classification
        manufacturer: Equipment manufacturer
        model_number: Model/part number
        serial_number: Serial number
        rated_power_kw: Rated power in kilowatts
        rated_speed_rpm: Rated speed in RPM
        installation_date: Date of installation
    """

    model_config = ConfigDict(protected_namespaces=())

    equipment_id: str = Field(
        default="EQ-001",
        min_length=1,
        max_length=50,
        description="Unique equipment identifier"
    )
    equipment_tag: str = Field(
        default="P-101A",
        min_length=1,
        max_length=50,
        description="P&ID/asset tag"
    )
    equipment_name: str = Field(
        default="Feed Water Pump A",
        min_length=1,
        max_length=100,
        description="Human-readable equipment name"
    )
    equipment_type: EquipmentType = Field(
        default=EquipmentType.PUMP,
        description="Equipment type classification"
    )

    # Manufacturer information
    manufacturer: str = Field(
        default="Industrial Pumps Inc.",
        min_length=1,
        max_length=100,
        description="Equipment manufacturer"
    )
    model_number: str = Field(
        default="IPX-5000",
        min_length=1,
        max_length=50,
        description="Model/part number"
    )
    serial_number: str = Field(
        default="SN-2024-001",
        min_length=1,
        max_length=50,
        description="Serial number"
    )

    # Rated specifications
    rated_power_kw: float = Field(
        default=75.0,
        ge=0.1,
        le=100000.0,
        description="Rated power in kilowatts"
    )
    rated_speed_rpm: float = Field(
        default=1800.0,
        ge=1.0,
        le=100000.0,
        description="Rated speed in RPM"
    )
    rated_voltage_v: float = Field(
        default=480.0,
        ge=1.0,
        le=100000.0,
        description="Rated voltage in volts"
    )
    rated_current_a: float = Field(
        default=95.0,
        ge=0.1,
        le=10000.0,
        description="Rated current in amperes"
    )

    # Physical specifications
    weight_kg: float = Field(
        default=500.0,
        ge=1.0,
        le=1000000.0,
        description="Equipment weight in kilograms"
    )

    # ISO classification
    machine_class: MachineClass = Field(
        default=MachineClass.CLASS_II,
        description="ISO 10816-3 machine classification"
    )
    mounting_type: MountingType = Field(
        default=MountingType.RIGID,
        description="Machine mounting type"
    )

    # Installation information
    installation_date: Optional[str] = Field(
        default=None,
        description="Installation date (ISO 8601 format)"
    )
    commissioning_date: Optional[str] = Field(
        default=None,
        description="Commissioning date (ISO 8601 format)"
    )
    last_major_overhaul: Optional[str] = Field(
        default=None,
        description="Date of last major overhaul (ISO 8601 format)"
    )

    # Operating hours tracking
    total_operating_hours: float = Field(
        default=0.0,
        ge=0.0,
        le=1000000.0,
        description="Total operating hours since installation"
    )
    hours_since_overhaul: float = Field(
        default=0.0,
        ge=0.0,
        le=500000.0,
        description="Operating hours since last major overhaul"
    )

    # Location information
    plant_id: str = Field(
        default="PLANT-001",
        description="Plant identifier"
    )
    area_id: str = Field(
        default="AREA-01",
        description="Plant area identifier"
    )
    location_description: str = Field(
        default="Main Process Building, Level 1",
        description="Physical location description"
    )

    # Criticality classification
    criticality_rating: str = Field(
        default="A",
        description="Criticality rating (A=Critical, B=Important, C=General)"
    )
    safety_critical: bool = Field(
        default=False,
        description="Safety-critical equipment flag"
    )
    environmental_critical: bool = Field(
        default=False,
        description="Environmental-critical equipment flag"
    )

    # Maintenance strategy
    maintenance_strategy: MaintenanceStrategy = Field(
        default=MaintenanceStrategy.PREDICTIVE,
        description="Assigned maintenance strategy"
    )

    # Vibration limits
    vibration_limits: ISO10816VibrationLimits = Field(
        default_factory=lambda: get_iso_10816_limits(MachineClass.CLASS_II),
        description="Vibration severity limits per ISO 10816"
    )

    @field_validator('criticality_rating')
    @classmethod
    def validate_criticality(cls, v):
        """Validate criticality rating."""
        valid_ratings = {"A", "B", "C", "D"}
        if v not in valid_ratings:
            raise ValueError(f"Invalid criticality rating. Must be one of: {valid_ratings}")
        return v

    @model_validator(mode='after')
    def validate_equipment_consistency(self):
        """Validate equipment configuration consistency."""
        # Ensure machine class matches vibration limits
        if self.vibration_limits.machine_class != self.machine_class:
            # Update vibration limits to match machine class
            self.vibration_limits = get_iso_10816_limits(self.machine_class)

        # Safety-critical equipment should be criticality A
        if self.safety_critical and self.criticality_rating not in {"A", "B"}:
            raise ValueError(
                "Safety-critical equipment must have criticality rating A or B"
            )

        return self


# ============================================================================
# VIBRATION SENSOR CONFIGURATION
# ============================================================================

class VibrationSensorConfiguration(BaseModel):
    """
    Vibration sensor configuration per ISO 10816 and ISO 13373.

    Defines accelerometer and velocity sensor settings for vibration
    monitoring on rotating machinery.

    Measurement Points per ISO 10816:
    - Bearing housings (radial and axial)
    - Drive end (DE) and non-drive end (NDE)
    - Horizontal, vertical, and axial orientations

    Attributes:
        sensor_id: Unique sensor identifier
        sensor_tag: P&ID instrument tag
        sensor_type: Type of vibration sensor
        measurement_unit: Vibration measurement unit
        frequency_range_hz: Frequency response range
        sensitivity: Sensor sensitivity
        mounting_location: Physical mounting location
    """

    model_config = ConfigDict(protected_namespaces=())

    sensor_id: str = Field(
        default="VS-001",
        min_length=1,
        max_length=50,
        description="Unique sensor identifier"
    )
    sensor_tag: str = Field(
        default="VT-P101A-DE-H",
        min_length=1,
        max_length=50,
        description="P&ID instrument tag"
    )
    sensor_name: str = Field(
        default="P-101A Drive End Horizontal Vibration",
        min_length=1,
        max_length=100,
        description="Human-readable sensor name"
    )
    sensor_type: SensorType = Field(
        default=SensorType.ACCELEROMETER,
        description="Type of vibration sensor"
    )

    # Sensor specifications
    manufacturer: str = Field(
        default="IMI Sensors",
        description="Sensor manufacturer"
    )
    model_number: str = Field(
        default="603C01",
        description="Sensor model number"
    )
    serial_number: Optional[str] = Field(
        default=None,
        description="Sensor serial number"
    )

    # Measurement parameters
    measurement_unit: VibrationUnit = Field(
        default=VibrationUnit.MM_S,
        description="Vibration measurement unit"
    )
    frequency_range_min_hz: float = Field(
        default=2.0,
        ge=0.1,
        le=100.0,
        description="Minimum frequency response in Hz"
    )
    frequency_range_max_hz: float = Field(
        default=10000.0,
        ge=100.0,
        le=50000.0,
        description="Maximum frequency response in Hz"
    )
    sensitivity_mv_per_g: float = Field(
        default=100.0,
        ge=1.0,
        le=1000.0,
        description="Sensor sensitivity in mV/g"
    )
    measurement_range_g: float = Field(
        default=50.0,
        ge=1.0,
        le=500.0,
        description="Measurement range in g peak"
    )

    # Mounting configuration
    mounting_location: str = Field(
        default="drive_end_horizontal",
        description="Physical mounting location (drive_end_horizontal, drive_end_vertical, "
                    "drive_end_axial, non_drive_end_horizontal, non_drive_end_vertical)"
    )
    mounting_type: str = Field(
        default="stud_mount",
        description="Mounting type (stud_mount, adhesive, magnetic, probe)"
    )
    orientation: str = Field(
        default="horizontal",
        description="Sensor orientation (horizontal, vertical, axial)"
    )

    # Signal conditioning
    iepe_powered: bool = Field(
        default=True,
        description="IEPE (ICP) powered sensor"
    )
    bias_voltage_v: float = Field(
        default=24.0,
        ge=10.0,
        le=30.0,
        description="IEPE bias voltage in volts"
    )
    output_signal: str = Field(
        default="4-20mA",
        description="Output signal type"
    )

    # Data acquisition parameters
    sampling_rate_hz: float = Field(
        default=25600.0,
        ge=256.0,
        le=102400.0,
        description="Data acquisition sampling rate in Hz"
    )
    sample_duration_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Sample duration per measurement in seconds"
    )
    averaging_samples: int = Field(
        default=4,
        ge=1,
        le=100,
        description="Number of samples to average"
    )
    window_type: str = Field(
        default="hanning",
        description="FFT window type (hanning, hamming, flat_top, rectangular)"
    )
    fft_lines: int = Field(
        default=6400,
        ge=400,
        le=25600,
        description="Number of FFT lines"
    )

    # Alarm thresholds (per ISO 10816)
    alarm_threshold_zone_b: float = Field(
        default=4.5,
        ge=0.1,
        le=50.0,
        description="Alert threshold (Zone B limit) in mm/s"
    )
    alarm_threshold_zone_c: float = Field(
        default=7.1,
        ge=0.5,
        le=100.0,
        description="Alarm threshold (Zone C limit) in mm/s"
    )
    trip_threshold_zone_d: float = Field(
        default=11.2,
        ge=1.0,
        le=150.0,
        description="Trip threshold (Zone D limit) in mm/s"
    )

    # Calibration
    calibration_interval_days: int = Field(
        default=365,
        ge=90,
        le=730,
        description="Calibration interval in days"
    )
    last_calibration_date: Optional[str] = Field(
        default=None,
        description="Last calibration date (ISO 8601 format)"
    )
    calibration_certificate: Optional[str] = Field(
        default=None,
        description="Calibration certificate reference"
    )

    @field_validator('mounting_location')
    @classmethod
    def validate_mounting_location(cls, v):
        """Validate mounting location."""
        valid_locations = {
            "drive_end_horizontal", "drive_end_vertical", "drive_end_axial",
            "non_drive_end_horizontal", "non_drive_end_vertical", "non_drive_end_axial",
            "gearbox_input", "gearbox_output", "coupling", "pedestal"
        }
        if v not in valid_locations:
            raise ValueError(f"Invalid mounting location. Must be one of: {valid_locations}")
        return v

    @field_validator('window_type')
    @classmethod
    def validate_window_type(cls, v):
        """Validate FFT window type."""
        valid_windows = {"hanning", "hamming", "flat_top", "rectangular", "blackman", "kaiser"}
        if v not in valid_windows:
            raise ValueError(f"Invalid window type. Must be one of: {valid_windows}")
        return v

    @model_validator(mode='after')
    def validate_threshold_sequence(self):
        """Ensure thresholds are in ascending order."""
        if not (self.alarm_threshold_zone_b < self.alarm_threshold_zone_c < self.trip_threshold_zone_d):
            raise ValueError(
                "Alarm thresholds must be in ascending order: Zone B < Zone C < Zone D"
            )
        return self


# ============================================================================
# TEMPERATURE SENSOR CONFIGURATION
# ============================================================================

class TemperatureSensorConfiguration(BaseModel):
    """
    Temperature sensor configuration for bearing and winding monitoring.

    RTD and thermocouple settings for thermal monitoring of equipment
    to detect bearing overheating, winding hot spots, and thermal degradation.

    Attributes:
        sensor_id: Unique sensor identifier
        sensor_tag: P&ID instrument tag
        sensor_type: RTD or thermocouple type
        measurement_range: Temperature measurement range
        alarm_thresholds: Temperature alarm setpoints
    """

    sensor_id: str = Field(
        default="TS-001",
        min_length=1,
        max_length=50,
        description="Unique sensor identifier"
    )
    sensor_tag: str = Field(
        default="TE-P101A-DE-BRG",
        min_length=1,
        max_length=50,
        description="P&ID instrument tag"
    )
    sensor_name: str = Field(
        default="P-101A Drive End Bearing Temperature",
        min_length=1,
        max_length=100,
        description="Human-readable sensor name"
    )
    sensor_type: SensorType = Field(
        default=SensorType.RTD,
        description="Temperature sensor type (RTD, thermocouple, infrared)"
    )

    # Sensor specifications
    rtd_type: str = Field(
        default="PT100",
        description="RTD type (PT100, PT1000)"
    )
    thermocouple_type: Optional[str] = Field(
        default=None,
        description="Thermocouple type (J, K, T, E, N)"
    )
    wire_configuration: str = Field(
        default="3-wire",
        description="Wire configuration (2-wire, 3-wire, 4-wire)"
    )

    # Measurement parameters
    measurement_range_min_c: float = Field(
        default=-50.0,
        ge=-200.0,
        le=0.0,
        description="Minimum temperature measurement range in Celsius"
    )
    measurement_range_max_c: float = Field(
        default=250.0,
        ge=50.0,
        le=1200.0,
        description="Maximum temperature measurement range in Celsius"
    )
    accuracy_c: float = Field(
        default=0.3,
        ge=0.01,
        le=5.0,
        description="Measurement accuracy in Celsius"
    )
    accuracy_class: str = Field(
        default="Class_A",
        description="IEC 60751 accuracy class (Class_A, Class_B, Class_AA)"
    )

    # Mounting location
    measurement_point: str = Field(
        default="bearing",
        description="Temperature measurement point (bearing, winding, casing, ambient)"
    )
    mounting_location: str = Field(
        default="drive_end_bearing",
        description="Physical mounting location"
    )

    # Signal output
    output_signal: str = Field(
        default="4-20mA",
        description="Output signal type"
    )
    transmitter_integrated: bool = Field(
        default=True,
        description="Integrated transmitter flag"
    )

    # Sampling configuration
    sampling_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Sampling interval in seconds"
    )
    filter_time_constant: float = Field(
        default=5.0,
        ge=0.0,
        le=60.0,
        description="Signal filter time constant in seconds"
    )

    # Alarm thresholds
    alarm_low_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=100.0,
        description="Low temperature alarm threshold"
    )
    alarm_high_c: float = Field(
        default=80.0,
        ge=30.0,
        le=200.0,
        description="High temperature alarm threshold"
    )
    alarm_high_high_c: float = Field(
        default=95.0,
        ge=40.0,
        le=250.0,
        description="High-high temperature alarm threshold"
    )
    trip_high_c: float = Field(
        default=110.0,
        ge=50.0,
        le=300.0,
        description="High temperature trip threshold"
    )

    # Rate of change monitoring
    rate_of_change_enabled: bool = Field(
        default=True,
        description="Enable rate of change monitoring"
    )
    max_rate_of_change_c_per_min: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Maximum acceptable rate of change in C/min"
    )

    # Calibration
    calibration_interval_days: int = Field(
        default=365,
        ge=90,
        le=730,
        description="Calibration interval in days"
    )
    last_calibration_date: Optional[str] = Field(
        default=None,
        description="Last calibration date (ISO 8601 format)"
    )

    @field_validator('rtd_type')
    @classmethod
    def validate_rtd_type(cls, v):
        """Validate RTD type."""
        valid_types = {"PT100", "PT1000", "PT500", "NI120", "CU10"}
        if v not in valid_types:
            raise ValueError(f"Invalid RTD type. Must be one of: {valid_types}")
        return v

    @field_validator('accuracy_class')
    @classmethod
    def validate_accuracy_class(cls, v):
        """Validate accuracy class."""
        valid_classes = {"Class_A", "Class_B", "Class_AA", "Class_C", "1/3_DIN"}
        if v not in valid_classes:
            raise ValueError(f"Invalid accuracy class. Must be one of: {valid_classes}")
        return v

    @model_validator(mode='after')
    def validate_temperature_thresholds(self):
        """Validate temperature threshold sequence."""
        if not (self.alarm_high_c < self.alarm_high_high_c < self.trip_high_c):
            raise ValueError(
                "Temperature thresholds must be in ascending order: high < high_high < trip"
            )
        return self


# ============================================================================
# PRESSURE SENSOR CONFIGURATION
# ============================================================================

class PressureSensorConfiguration(BaseModel):
    """
    Pressure sensor configuration for process monitoring.

    Pressure transmitter settings for monitoring suction/discharge
    pressure, differential pressure, and seal pressure.

    Attributes:
        sensor_id: Unique sensor identifier
        sensor_tag: P&ID instrument tag
        measurement_type: Gauge, absolute, or differential
        measurement_range: Pressure measurement range
    """

    sensor_id: str = Field(
        default="PS-001",
        min_length=1,
        max_length=50,
        description="Unique sensor identifier"
    )
    sensor_tag: str = Field(
        default="PT-P101A-DISCH",
        min_length=1,
        max_length=50,
        description="P&ID instrument tag"
    )
    sensor_name: str = Field(
        default="P-101A Discharge Pressure",
        min_length=1,
        max_length=100,
        description="Human-readable sensor name"
    )
    sensor_type: SensorType = Field(
        default=SensorType.PRESSURE_TRANSMITTER,
        description="Pressure sensor type"
    )

    # Measurement type
    measurement_type: str = Field(
        default="gauge",
        description="Measurement type (gauge, absolute, differential)"
    )

    # Measurement parameters
    measurement_range_min: float = Field(
        default=0.0,
        ge=-100.0,
        le=1000.0,
        description="Minimum pressure measurement range"
    )
    measurement_range_max: float = Field(
        default=20.0,
        ge=0.1,
        le=10000.0,
        description="Maximum pressure measurement range"
    )
    measurement_unit: str = Field(
        default="bar",
        description="Pressure measurement unit (bar, psi, kPa, MPa)"
    )

    # Accuracy
    accuracy_percent: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Measurement accuracy as percentage of span"
    )
    repeatability_percent: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Repeatability as percentage of span"
    )

    # Signal output
    output_signal: str = Field(
        default="4-20mA",
        description="Output signal type"
    )
    communication_protocol: str = Field(
        default="HART",
        description="Digital communication protocol"
    )

    # Sampling
    sampling_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Sampling interval in seconds"
    )

    # Alarm thresholds
    alarm_low: Optional[float] = Field(
        default=None,
        description="Low pressure alarm threshold"
    )
    alarm_low_low: Optional[float] = Field(
        default=None,
        description="Low-low pressure alarm threshold"
    )
    alarm_high: Optional[float] = Field(
        default=None,
        description="High pressure alarm threshold"
    )
    alarm_high_high: Optional[float] = Field(
        default=None,
        description="High-high pressure alarm threshold"
    )

    # Calibration
    calibration_interval_days: int = Field(
        default=365,
        ge=90,
        le=730,
        description="Calibration interval in days"
    )

    @field_validator('measurement_type')
    @classmethod
    def validate_measurement_type(cls, v):
        """Validate pressure measurement type."""
        valid_types = {"gauge", "absolute", "differential", "vacuum"}
        if v not in valid_types:
            raise ValueError(f"Invalid measurement type. Must be one of: {valid_types}")
        return v

    @field_validator('measurement_unit')
    @classmethod
    def validate_measurement_unit(cls, v):
        """Validate pressure measurement unit."""
        valid_units = {"bar", "psi", "kPa", "MPa", "mbar", "inH2O", "mmHg", "Pa"}
        if v not in valid_units:
            raise ValueError(f"Invalid measurement unit. Must be one of: {valid_units}")
        return v


# ============================================================================
# OPERATING HOURS CONFIGURATION
# ============================================================================

class OperatingHoursConfiguration(BaseModel):
    """
    Operating hours tracking and runtime configuration.

    Tracks equipment runtime for maintenance scheduling and
    reliability analysis.

    Attributes:
        total_hours: Total operating hours
        hours_since_maintenance: Hours since last maintenance
        operating_hours_source: Source of hours data
    """

    equipment_id: str = Field(
        default="EQ-001",
        description="Equipment identifier"
    )

    # Operating hours tracking
    total_operating_hours: float = Field(
        default=0.0,
        ge=0.0,
        le=1000000.0,
        description="Total operating hours since installation"
    )
    hours_since_major_overhaul: float = Field(
        default=0.0,
        ge=0.0,
        le=500000.0,
        description="Hours since last major overhaul"
    )
    hours_since_minor_maintenance: float = Field(
        default=0.0,
        ge=0.0,
        le=100000.0,
        description="Hours since last minor maintenance"
    )
    hours_since_lubrication: float = Field(
        default=0.0,
        ge=0.0,
        le=50000.0,
        description="Hours since last lubrication"
    )

    # Operating cycles
    start_stop_cycles: int = Field(
        default=0,
        ge=0,
        le=10000000,
        description="Total start/stop cycles"
    )
    cycles_since_maintenance: int = Field(
        default=0,
        ge=0,
        le=1000000,
        description="Start/stop cycles since last maintenance"
    )

    # Data source
    operating_hours_source: str = Field(
        default="dcs",
        description="Source of operating hours data (dcs, plc, manual, meter)"
    )

    # Tracking parameters
    hour_meter_tag: Optional[str] = Field(
        default=None,
        description="Hour meter instrument tag"
    )
    running_status_tag: Optional[str] = Field(
        default=None,
        description="Running status tag for automatic tracking"
    )

    # Thresholds for maintenance triggers
    major_overhaul_interval_hours: float = Field(
        default=50000.0,
        ge=1000.0,
        le=200000.0,
        description="Interval for major overhaul in hours"
    )
    minor_maintenance_interval_hours: float = Field(
        default=8760.0,  # 1 year
        ge=100.0,
        le=50000.0,
        description="Interval for minor maintenance in hours"
    )
    lubrication_interval_hours: float = Field(
        default=2000.0,
        ge=100.0,
        le=10000.0,
        description="Interval for lubrication in hours"
    )

    @field_validator('operating_hours_source')
    @classmethod
    def validate_hours_source(cls, v):
        """Validate operating hours source."""
        valid_sources = {"dcs", "plc", "manual", "meter", "scada", "historian"}
        if v not in valid_sources:
            raise ValueError(f"Invalid hours source. Must be one of: {valid_sources}")
        return v


# ============================================================================
# FAILURE MODE CONFIGURATION
# ============================================================================

class FailureModeConfiguration(BaseModel):
    """
    Failure mode and effects analysis (FMEA) configuration.

    Defines failure modes, their detectability, and associated
    parameters for predictive maintenance monitoring.

    Attributes:
        failure_mode: Type of failure
        equipment_type: Applicable equipment type
        severity: Failure severity rating (1-10)
        occurrence: Probability of occurrence (1-10)
        detection: Detection difficulty (1-10)
        rpn: Risk Priority Number (S x O x D)
    """

    failure_mode_id: str = Field(
        default="FM-001",
        min_length=1,
        max_length=50,
        description="Failure mode identifier"
    )
    failure_mode: FailureMode = Field(
        default=FailureMode.BEARING_WEAR,
        description="Type of failure mode"
    )
    equipment_type: EquipmentType = Field(
        default=EquipmentType.PUMP,
        description="Applicable equipment type"
    )

    # Failure mode description
    description: str = Field(
        default="Bearing wear due to normal operational fatigue",
        min_length=1,
        max_length=500,
        description="Detailed failure mode description"
    )
    failure_effect: str = Field(
        default="Increased vibration, elevated temperature, eventual seizure",
        min_length=1,
        max_length=500,
        description="Effect of failure on equipment/process"
    )
    failure_cause: str = Field(
        default="Normal wear, inadequate lubrication, contamination, overload",
        min_length=1,
        max_length=500,
        description="Root cause of failure mode"
    )

    # FMEA ratings (1-10 scale)
    severity: int = Field(
        default=7,
        ge=1,
        le=10,
        description="Severity rating (1=minor, 10=catastrophic)"
    )
    occurrence: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Occurrence rating (1=rare, 10=very frequent)"
    )
    detection: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Detection difficulty (1=easy, 10=impossible)"
    )

    # Detection methods
    detection_methods: List[str] = Field(
        default_factory=lambda: ["vibration_analysis", "temperature_monitoring"],
        description="Available detection methods"
    )
    primary_indicator: str = Field(
        default="vibration",
        description="Primary condition indicator (vibration, temperature, pressure, current)"
    )
    secondary_indicators: List[str] = Field(
        default_factory=lambda: ["temperature", "noise"],
        description="Secondary condition indicators"
    )

    # Weibull parameters for this failure mode
    weibull_parameters: Optional[WeibullParameters] = Field(
        default=None,
        description="Weibull distribution parameters for reliability analysis"
    )

    # P-F interval (Potential-to-Functional failure)
    pf_interval_hours: float = Field(
        default=500.0,
        ge=1.0,
        le=10000.0,
        description="P-F interval in hours (time from detectable to functional failure)"
    )
    warning_lead_time_hours: float = Field(
        default=168.0,  # 1 week
        ge=1.0,
        le=5000.0,
        description="Desired warning lead time for maintenance planning"
    )

    # Monitoring frequency based on P-F interval
    recommended_monitoring_interval_hours: float = Field(
        default=24.0,
        ge=0.5,
        le=720.0,
        description="Recommended monitoring interval (typically PF/3)"
    )

    @model_validator(mode='after')
    def calculate_rpn(self):
        """Calculate Risk Priority Number."""
        # RPN is calculated but not stored as a field to avoid mismatch
        # It can be accessed via property
        return self

    @property
    def rpn(self) -> int:
        """Calculate Risk Priority Number (RPN = S x O x D)."""
        return self.severity * self.occurrence * self.detection


# ============================================================================
# MAINTENANCE SCHEDULE CONFIGURATION
# ============================================================================

class MaintenanceTaskConfiguration(BaseModel):
    """Configuration for a single maintenance task."""

    task_id: str = Field(
        default="MT-001",
        description="Maintenance task identifier"
    )
    task_name: str = Field(
        default="Bearing inspection and lubrication",
        description="Task name"
    )
    task_type: str = Field(
        default="inspection",
        description="Task type (inspection, lubrication, replacement, repair, overhaul)"
    )

    # Scheduling parameters
    interval_hours: float = Field(
        default=2000.0,
        ge=1.0,
        le=100000.0,
        description="Task interval in operating hours"
    )
    interval_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=3650,
        description="Calendar interval in days (alternative to hours)"
    )
    condition_based: bool = Field(
        default=True,
        description="Condition-based trigger enabled"
    )

    # Duration and resources
    estimated_duration_hours: float = Field(
        default=2.0,
        ge=0.1,
        le=168.0,
        description="Estimated task duration in hours"
    )
    required_skills: List[str] = Field(
        default_factory=lambda: ["mechanical_technician"],
        description="Required skill sets"
    )
    required_parts: List[str] = Field(
        default_factory=list,
        description="Required spare parts"
    )

    # Shutdown requirements
    requires_shutdown: bool = Field(
        default=False,
        description="Requires equipment shutdown"
    )
    can_defer: bool = Field(
        default=True,
        description="Task can be deferred if needed"
    )
    max_deferral_days: int = Field(
        default=30,
        ge=0,
        le=365,
        description="Maximum deferral period in days"
    )

    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v):
        """Validate task type."""
        valid_types = {
            "inspection", "lubrication", "replacement", "repair",
            "overhaul", "calibration", "testing", "cleaning", "alignment"
        }
        if v not in valid_types:
            raise ValueError(f"Invalid task type. Must be one of: {valid_types}")
        return v


class MaintenanceScheduleConfiguration(BaseModel):
    """
    Maintenance schedule configuration.

    Defines scheduled maintenance tasks, intervals, and
    planning parameters for equipment.

    Attributes:
        equipment_id: Equipment identifier
        maintenance_strategy: Overall maintenance strategy
        scheduled_tasks: List of scheduled maintenance tasks
    """

    schedule_id: str = Field(
        default="MS-001",
        min_length=1,
        max_length=50,
        description="Maintenance schedule identifier"
    )
    equipment_id: str = Field(
        default="EQ-001",
        description="Equipment identifier"
    )
    schedule_name: str = Field(
        default="Standard Pump Maintenance Schedule",
        description="Schedule name"
    )

    # Maintenance strategy
    maintenance_strategy: MaintenanceStrategy = Field(
        default=MaintenanceStrategy.PREDICTIVE,
        description="Primary maintenance strategy"
    )

    # Scheduled tasks
    scheduled_tasks: List[MaintenanceTaskConfiguration] = Field(
        default_factory=lambda: [
            MaintenanceTaskConfiguration(
                task_id="MT-001",
                task_name="Vibration inspection",
                task_type="inspection",
                interval_hours=720,  # Monthly
                condition_based=True,
                requires_shutdown=False
            ),
            MaintenanceTaskConfiguration(
                task_id="MT-002",
                task_name="Bearing lubrication",
                task_type="lubrication",
                interval_hours=2000,
                condition_based=True,
                requires_shutdown=False
            ),
            MaintenanceTaskConfiguration(
                task_id="MT-003",
                task_name="Alignment check",
                task_type="inspection",
                interval_hours=8760,  # Annual
                condition_based=False,
                requires_shutdown=True
            ),
        ],
        description="Scheduled maintenance tasks"
    )

    # Planning parameters
    planning_horizon_days: int = Field(
        default=365,
        ge=30,
        le=1825,
        description="Maintenance planning horizon in days"
    )
    scheduling_buffer_percent: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        description="Scheduling buffer as percentage of interval"
    )

    # Work order generation
    auto_generate_work_orders: bool = Field(
        default=True,
        description="Automatically generate work orders"
    )
    work_order_lead_time_days: int = Field(
        default=14,
        ge=1,
        le=90,
        description="Lead time for work order generation"
    )

    # Shutdown optimization
    combine_shutdown_tasks: bool = Field(
        default=True,
        description="Combine tasks requiring shutdown"
    )
    preferred_maintenance_window: Optional[str] = Field(
        default=None,
        description="Preferred maintenance window (e.g., 'weekend', 'night_shift')"
    )


# ============================================================================
# SPARE PARTS CONFIGURATION
# ============================================================================

class SparePartConfiguration(BaseModel):
    """Configuration for a single spare part."""

    part_id: str = Field(
        default="SP-001",
        description="Spare part identifier"
    )
    part_number: str = Field(
        default="BRG-6205-2RS",
        description="Part number"
    )
    part_name: str = Field(
        default="Deep Groove Ball Bearing 6205-2RS",
        description="Part name/description"
    )

    # Inventory parameters
    minimum_stock_quantity: int = Field(
        default=2,
        ge=0,
        le=1000,
        description="Minimum stock quantity"
    )
    reorder_point: int = Field(
        default=2,
        ge=0,
        le=500,
        description="Reorder trigger point"
    )
    reorder_quantity: int = Field(
        default=4,
        ge=1,
        le=1000,
        description="Quantity to reorder"
    )
    current_stock: int = Field(
        default=4,
        ge=0,
        le=10000,
        description="Current stock quantity"
    )

    # Lead time
    lead_time_days: int = Field(
        default=14,
        ge=1,
        le=365,
        description="Procurement lead time in days"
    )

    # Cost
    unit_cost: float = Field(
        default=50.0,
        ge=0.01,
        le=1000000.0,
        description="Unit cost"
    )
    currency: str = Field(
        default="USD",
        description="Currency code"
    )

    # Criticality
    criticality: str = Field(
        default="high",
        description="Part criticality (critical, high, medium, low)"
    )

    # Storage
    storage_location: str = Field(
        default="STORE-01-A1",
        description="Storage location code"
    )
    shelf_life_days: Optional[int] = Field(
        default=None,
        ge=30,
        le=3650,
        description="Shelf life in days (if applicable)"
    )

    @field_validator('criticality')
    @classmethod
    def validate_criticality(cls, v):
        """Validate part criticality."""
        valid_levels = {"critical", "high", "medium", "low"}
        if v not in valid_levels:
            raise ValueError(f"Invalid criticality. Must be one of: {valid_levels}")
        return v


class SparePartsConfiguration(BaseModel):
    """
    Spare parts inventory configuration.

    Manages spare parts inventory for predictive maintenance
    and ensures parts availability for planned maintenance.

    Attributes:
        equipment_id: Equipment identifier
        spare_parts: List of spare parts configurations
    """

    inventory_id: str = Field(
        default="INV-001",
        min_length=1,
        max_length=50,
        description="Inventory configuration identifier"
    )
    equipment_id: str = Field(
        default="EQ-001",
        description="Equipment identifier"
    )

    # Spare parts list
    spare_parts: List[SparePartConfiguration] = Field(
        default_factory=lambda: [
            SparePartConfiguration(
                part_id="SP-001",
                part_number="BRG-6205-2RS",
                part_name="Drive End Bearing",
                minimum_stock_quantity=2,
                criticality="high"
            ),
            SparePartConfiguration(
                part_id="SP-002",
                part_number="SEAL-MECH-50",
                part_name="Mechanical Seal Assembly",
                minimum_stock_quantity=1,
                criticality="critical"
            ),
        ],
        description="List of spare parts"
    )

    # Inventory management
    auto_reorder_enabled: bool = Field(
        default=True,
        description="Enable automatic reorder generation"
    )
    safety_stock_days: int = Field(
        default=30,
        ge=0,
        le=365,
        description="Safety stock in days of lead time"
    )

    # Cost optimization
    total_inventory_value_limit: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100000000.0,
        description="Maximum inventory value"
    )


# ============================================================================
# CMS INTEGRATION CONFIGURATION
# ============================================================================

class CMSIntegrationConfiguration(BaseModel):
    """
    Condition Monitoring System (CMS) integration configuration.

    Settings for integrating with external condition monitoring
    systems and data acquisition platforms.

    Attributes:
        cms_type: Type of CMS system
        connection_settings: Connection parameters
        data_mapping: Tag mapping configuration
    """

    integration_id: str = Field(
        default="CMS-INT-001",
        min_length=1,
        max_length=50,
        description="Integration identifier"
    )
    integration_name: str = Field(
        default="Primary CMS Integration",
        description="Integration name"
    )
    cms_type: str = Field(
        default="bently_nevada",
        description="CMS system type"
    )
    enabled: bool = Field(
        default=True,
        description="Integration enabled flag"
    )

    # Connection settings (credentials via environment variables)
    host: str = Field(
        default="cms.example.com",
        description="CMS server hostname"
    )
    port: int = Field(
        default=443,
        ge=1,
        le=65535,
        description="Connection port"
    )
    use_tls: bool = Field(
        default=True,
        description="Use TLS encryption"
    )

    # Authentication (credentials from environment)
    auth_method: str = Field(
        default="api_key",
        description="Authentication method (api_key, oauth2, certificate)"
    )
    api_key_env_var: str = Field(
        default="CMS_API_KEY",
        description="Environment variable for API key"
    )

    # Data synchronization
    sync_interval_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Data synchronization interval"
    )
    batch_size: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Batch size for data retrieval"
    )

    # Data mapping
    tag_prefix: str = Field(
        default="CMS.",
        description="Tag prefix for CMS data"
    )
    timestamp_format: str = Field(
        default="ISO8601",
        description="Timestamp format"
    )

    # Error handling
    retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts"
    )
    retry_delay_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Delay between retries"
    )

    @field_validator('cms_type')
    @classmethod
    def validate_cms_type(cls, v):
        """Validate CMS type."""
        valid_types = {
            "bently_nevada", "emerson_ams", "skf_observer", "pruftechnik",
            "vibconnect", "gea", "fluke", "ge_bently", "rockwell", "custom"
        }
        if v not in valid_types:
            raise ValueError(f"Invalid CMS type. Must be one of: {valid_types}")
        return v

    @field_validator('auth_method')
    @classmethod
    def validate_auth_method(cls, v):
        """Validate authentication method."""
        valid_methods = {"api_key", "oauth2", "certificate", "basic", "none"}
        if v not in valid_methods:
            raise ValueError(f"Invalid auth method. Must be one of: {valid_methods}")
        return v


# ============================================================================
# CMMS INTEGRATION CONFIGURATION
# ============================================================================

class CMMSIntegrationConfiguration(BaseModel):
    """
    Computerized Maintenance Management System (CMMS) integration.

    Configuration for integrating with enterprise CMMS systems
    like SAP PM, Maximo, Infor EAM for work order management.

    Attributes:
        cmms_type: Type of CMMS system
        connection_settings: Connection parameters
        work_order_settings: Work order generation settings
    """

    integration_id: str = Field(
        default="CMMS-INT-001",
        min_length=1,
        max_length=50,
        description="Integration identifier"
    )
    integration_name: str = Field(
        default="SAP PM Integration",
        description="Integration name"
    )
    cmms_type: CMMSType = Field(
        default=CMMSType.SAP_PM,
        description="CMMS system type"
    )
    enabled: bool = Field(
        default=True,
        description="Integration enabled flag"
    )

    # Connection settings
    api_endpoint: str = Field(
        default="https://sap.example.com/api/pm",
        description="CMMS API endpoint URL"
    )
    use_tls: bool = Field(
        default=True,
        description="Use TLS encryption"
    )

    # Authentication (credentials from environment)
    auth_method: str = Field(
        default="oauth2",
        description="Authentication method"
    )
    client_id_env_var: str = Field(
        default="CMMS_CLIENT_ID",
        description="Environment variable for client ID"
    )
    client_secret_env_var: str = Field(
        default="CMMS_CLIENT_SECRET",
        description="Environment variable for client secret"
    )

    # Work order settings
    auto_create_work_orders: bool = Field(
        default=True,
        description="Automatically create work orders"
    )
    default_work_order_priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Default work order priority"
    )
    work_order_type: str = Field(
        default="PM01",
        description="Work order type code"
    )
    maintenance_plant: str = Field(
        default="1000",
        description="Maintenance plant code"
    )
    planning_group: str = Field(
        default="001",
        description="Planning group code"
    )

    # Priority mapping
    priority_mapping: Dict[AlertSeverity, WorkOrderPriority] = Field(
        default_factory=lambda: {
            AlertSeverity.EMERGENCY: WorkOrderPriority.EMERGENCY,
            AlertSeverity.CRITICAL: WorkOrderPriority.URGENT,
            AlertSeverity.HIGH: WorkOrderPriority.HIGH,
            AlertSeverity.WARNING: WorkOrderPriority.MEDIUM,
            AlertSeverity.INFO: WorkOrderPriority.LOW
        },
        description="Alert severity to work order priority mapping"
    )

    # Notification settings
    create_notifications: bool = Field(
        default=True,
        description="Create maintenance notifications"
    )
    notification_type: str = Field(
        default="M2",
        description="Notification type code"
    )

    # Data synchronization
    sync_interval_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Sync interval in seconds"
    )
    sync_equipment_master: bool = Field(
        default=True,
        description="Synchronize equipment master data"
    )

    # Error handling
    retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts"
    )

    @field_validator('auth_method')
    @classmethod
    def validate_auth_method(cls, v):
        """Validate authentication method."""
        valid_methods = {"oauth2", "basic", "api_key", "certificate", "saml"}
        if v not in valid_methods:
            raise ValueError(f"Invalid auth method. Must be one of: {valid_methods}")
        return v


# ============================================================================
# ALERT CONFIGURATION
# ============================================================================

class AlertThresholdConfiguration(BaseModel):
    """Configuration for a single alert threshold."""

    alert_id: str = Field(
        default="ALT-001",
        description="Alert identifier"
    )
    alert_name: str = Field(
        default="High Vibration Alert",
        description="Alert name"
    )
    parameter: str = Field(
        default="vibration_velocity",
        description="Monitored parameter"
    )

    # Threshold settings
    severity: AlertSeverity = Field(
        default=AlertSeverity.WARNING,
        description="Alert severity"
    )
    threshold_value: float = Field(
        default=4.5,
        description="Threshold value"
    )
    threshold_type: str = Field(
        default="high",
        description="Threshold type (high, low, high_high, low_low, deviation, rate_of_change)"
    )
    threshold_unit: str = Field(
        default="mm/s",
        description="Threshold unit"
    )

    # Alert behavior
    deadband: float = Field(
        default=0.5,
        ge=0.0,
        description="Alert deadband for return-to-normal"
    )
    delay_seconds: float = Field(
        default=10.0,
        ge=0.0,
        le=3600.0,
        description="Delay before alert activation"
    )
    auto_acknowledge: bool = Field(
        default=False,
        description="Auto-acknowledge on return to normal"
    )

    # Escalation
    escalation_enabled: bool = Field(
        default=True,
        description="Enable escalation"
    )
    escalation_delay_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Delay before escalation"
    )
    escalate_to_severity: Optional[AlertSeverity] = Field(
        default=AlertSeverity.HIGH,
        description="Escalate to this severity level"
    )

    @field_validator('threshold_type')
    @classmethod
    def validate_threshold_type(cls, v):
        """Validate threshold type."""
        valid_types = {"high", "low", "high_high", "low_low", "deviation", "rate_of_change", "band"}
        if v not in valid_types:
            raise ValueError(f"Invalid threshold type. Must be one of: {valid_types}")
        return v


class AlertConfiguration(BaseModel):
    """
    Alert and notification configuration.

    Defines alert thresholds, escalation rules, and notification
    settings for predictive maintenance alerts.

    Attributes:
        alert_thresholds: List of alert threshold configurations
        notification_settings: Notification delivery settings
        escalation_rules: Alert escalation rules
    """

    config_id: str = Field(
        default="ALERT-CFG-001",
        min_length=1,
        max_length=50,
        description="Alert configuration identifier"
    )
    config_name: str = Field(
        default="Standard Alert Configuration",
        description="Configuration name"
    )

    # Alert thresholds
    alert_thresholds: List[AlertThresholdConfiguration] = Field(
        default_factory=lambda: [
            AlertThresholdConfiguration(
                alert_id="ALT-VIB-WARN",
                alert_name="Vibration Warning",
                parameter="vibration_velocity",
                severity=AlertSeverity.WARNING,
                threshold_value=4.5,
                threshold_type="high"
            ),
            AlertThresholdConfiguration(
                alert_id="ALT-VIB-HIGH",
                alert_name="Vibration High Alarm",
                parameter="vibration_velocity",
                severity=AlertSeverity.HIGH,
                threshold_value=7.1,
                threshold_type="high"
            ),
            AlertThresholdConfiguration(
                alert_id="ALT-VIB-CRIT",
                alert_name="Vibration Critical",
                parameter="vibration_velocity",
                severity=AlertSeverity.CRITICAL,
                threshold_value=11.2,
                threshold_type="high"
            ),
            AlertThresholdConfiguration(
                alert_id="ALT-TEMP-HIGH",
                alert_name="Bearing Temperature High",
                parameter="bearing_temperature",
                severity=AlertSeverity.WARNING,
                threshold_value=80.0,
                threshold_type="high",
                threshold_unit="C"
            ),
            AlertThresholdConfiguration(
                alert_id="ALT-TEMP-CRIT",
                alert_name="Bearing Temperature Critical",
                parameter="bearing_temperature",
                severity=AlertSeverity.CRITICAL,
                threshold_value=95.0,
                threshold_type="high",
                threshold_unit="C"
            ),
        ],
        description="Alert threshold configurations"
    )

    # Notification settings
    email_notifications_enabled: bool = Field(
        default=True,
        description="Enable email notifications"
    )
    sms_notifications_enabled: bool = Field(
        default=False,
        description="Enable SMS notifications"
    )
    push_notifications_enabled: bool = Field(
        default=True,
        description="Enable push notifications"
    )

    # Notification recipients by severity
    notification_recipients: Dict[AlertSeverity, List[str]] = Field(
        default_factory=lambda: {
            AlertSeverity.EMERGENCY: ["maintenance_manager", "plant_manager", "on_call"],
            AlertSeverity.CRITICAL: ["maintenance_manager", "reliability_engineer", "on_call"],
            AlertSeverity.HIGH: ["reliability_engineer", "maintenance_planner"],
            AlertSeverity.WARNING: ["reliability_engineer"],
            AlertSeverity.INFO: []
        },
        description="Notification recipients by severity"
    )

    # Alert suppression
    suppression_enabled: bool = Field(
        default=True,
        description="Enable alert suppression during known events"
    )
    suppression_window_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Alert suppression window in minutes"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable notification rate limiting"
    )
    rate_limit_per_hour: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum notifications per hour per alert type"
    )

    # Acknowledgment settings
    require_acknowledgment: bool = Field(
        default=True,
        description="Require manual acknowledgment"
    )
    acknowledgment_timeout_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Timeout for acknowledgment before escalation"
    )


# ============================================================================
# ML MODEL CONFIGURATION
# ============================================================================

class MLModelConfiguration(BaseModel):
    """
    Machine learning model configuration for anomaly detection
    and remaining useful life prediction.

    Defines ML model parameters, training settings, and
    performance thresholds for predictive maintenance.

    Attributes:
        model_type: Type of ML model
        model_parameters: Model-specific parameters
        training_settings: Model training configuration
        performance_thresholds: Model performance requirements
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(
        default="ML-001",
        min_length=1,
        max_length=50,
        description="Model identifier"
    )
    model_name: str = Field(
        default="Anomaly Detection Model",
        description="Model name"
    )
    model_type: MLModelType = Field(
        default=MLModelType.ISOLATION_FOREST,
        description="Type of ML model"
    )
    model_purpose: str = Field(
        default="anomaly_detection",
        description="Model purpose (anomaly_detection, rul_prediction, classification)"
    )

    # Model parameters (Isolation Forest defaults)
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of estimators/trees"
    )
    contamination: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Expected proportion of anomalies"
    )
    max_features: float = Field(
        default=1.0,
        ge=0.1,
        le=1.0,
        description="Maximum features per estimator"
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )

    # Feature configuration
    feature_columns: List[str] = Field(
        default_factory=lambda: [
            "vibration_velocity_de_h",
            "vibration_velocity_de_v",
            "vibration_velocity_nde_h",
            "bearing_temp_de",
            "bearing_temp_nde",
            "motor_current",
            "operating_speed"
        ],
        description="Feature columns for model input"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column for supervised learning"
    )

    # Training settings
    training_data_window_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Training data window in days"
    )
    min_training_samples: int = Field(
        default=1000,
        ge=100,
        le=1000000,
        description="Minimum samples for training"
    )
    validation_split: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
        description="Validation data split ratio"
    )
    retraining_interval_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Model retraining interval in days"
    )

    # Performance thresholds
    min_precision: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Minimum precision threshold"
    )
    min_recall: float = Field(
        default=0.80,
        ge=0.5,
        le=1.0,
        description="Minimum recall threshold"
    )
    min_f1_score: float = Field(
        default=0.82,
        ge=0.5,
        le=1.0,
        description="Minimum F1 score threshold"
    )
    max_false_positive_rate: float = Field(
        default=0.15,
        ge=0.01,
        le=0.5,
        description="Maximum false positive rate"
    )

    # Anomaly detection settings
    anomaly_threshold: float = Field(
        default=-0.5,
        ge=-1.0,
        le=0.0,
        description="Anomaly score threshold (Isolation Forest)"
    )
    consecutive_anomalies_for_alert: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive anomalies before alert"
    )

    # Model versioning
    model_version: str = Field(
        default="1.0.0",
        description="Model version"
    )
    last_trained_date: Optional[str] = Field(
        default=None,
        description="Last training date (ISO 8601)"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to saved model file"
    )

    @field_validator('model_purpose')
    @classmethod
    def validate_model_purpose(cls, v):
        """Validate model purpose."""
        valid_purposes = {
            "anomaly_detection", "rul_prediction", "classification",
            "regression", "clustering", "health_index"
        }
        if v not in valid_purposes:
            raise ValueError(f"Invalid model purpose. Must be one of: {valid_purposes}")
        return v


# ============================================================================
# DATA QUALITY CONFIGURATION
# ============================================================================

class DataQualityConfiguration(BaseModel):
    """
    Data quality monitoring configuration.

    Defines data quality requirements and validation rules
    for predictive maintenance data.

    Attributes:
        completeness_threshold: Minimum data completeness
        outlier_threshold: Maximum acceptable outlier percentage
        staleness_threshold: Maximum data age
    """

    config_id: str = Field(
        default="DQ-001",
        description="Data quality configuration identifier"
    )

    # Completeness requirements
    min_completeness_percent: float = Field(
        default=95.0,
        ge=80.0,
        le=100.0,
        description="Minimum data completeness percentage"
    )

    # Outlier detection
    max_outlier_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=20.0,
        description="Maximum acceptable outlier percentage"
    )
    outlier_detection_method: str = Field(
        default="iqr",
        description="Outlier detection method (iqr, zscore, isolation_forest)"
    )
    zscore_threshold: float = Field(
        default=3.0,
        ge=2.0,
        le=5.0,
        description="Z-score threshold for outlier detection"
    )
    iqr_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="IQR multiplier for outlier detection"
    )

    # Data staleness
    max_data_age_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Maximum acceptable data age in seconds"
    )

    # Validation rules
    range_validation_enabled: bool = Field(
        default=True,
        description="Enable range validation"
    )
    rate_of_change_validation_enabled: bool = Field(
        default=True,
        description="Enable rate of change validation"
    )

    # Quality level thresholds
    excellent_threshold: float = Field(
        default=99.0,
        ge=95.0,
        le=100.0,
        description="Threshold for excellent quality level"
    )
    good_threshold: float = Field(
        default=95.0,
        ge=90.0,
        le=99.0,
        description="Threshold for good quality level"
    )
    acceptable_threshold: float = Field(
        default=90.0,
        ge=80.0,
        le=95.0,
        description="Threshold for acceptable quality level"
    )

    @field_validator('outlier_detection_method')
    @classmethod
    def validate_outlier_method(cls, v):
        """Validate outlier detection method."""
        valid_methods = {"iqr", "zscore", "isolation_forest", "mad", "dbscan"}
        if v not in valid_methods:
            raise ValueError(f"Invalid outlier method. Must be one of: {valid_methods}")
        return v


# ============================================================================
# MAIN AGENT CONFIGURATION
# ============================================================================

class PredictiveMaintenanceConfig(BaseModel):
    """
    Main configuration for PredictMaint agent (GL-013 PREDICTMAINT).

    Comprehensive configuration including equipment specifications,
    sensor configurations, failure mode analysis, maintenance scheduling,
    ML model parameters, and system integrations.

    The PredictMaint agent is responsible for:
    - Equipment health monitoring via condition sensors
    - Anomaly detection using machine learning
    - Remaining useful life (RUL) prediction
    - Maintenance scheduling optimization
    - Work order generation via CMMS integration
    - Spare parts inventory optimization

    SECURITY & COMPLIANCE:
    - Zero hardcoded credentials policy enforced
    - Deterministic mode required for reproducible predictions
    - TLS encryption mandatory for integrations
    - Provenance tracking required for audit trails
    - ISO 10816/13373/55000 compliance validation

    Attributes:
        agent_id: Unique agent identifier (GL-013)
        agent_name: Agent display name (PredictMaint)
        version: Agent software version
        equipment_configs: Equipment configurations
        sensor_configs: Sensor configurations
        ml_model_configs: ML model configurations
    """

    # ========================================================================
    # AGENT IDENTIFICATION
    # ========================================================================

    agent_id: str = Field(
        default="GL-013",
        min_length=1,
        max_length=20,
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="PredictMaint",
        min_length=1,
        max_length=100,
        description="Agent display name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent software version"
    )
    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production)"
    )
    domain: str = Field(
        default="Predictive Maintenance",
        description="Agent domain classification"
    )
    agent_type: str = Field(
        default="Analyzer",
        description="Agent type classification"
    )

    # ========================================================================
    # PERFORMANCE SETTINGS
    # ========================================================================

    calculation_timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Maximum time allowed for calculations"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on failure"
    )
    retry_delay_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Delay between retry attempts"
    )
    enable_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Cache time-to-live in seconds"
    )
    max_concurrent_analyses: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent analysis threads"
    )

    # ========================================================================
    # EQUIPMENT CONFIGURATIONS
    # ========================================================================

    equipment_configs: List[EquipmentConfiguration] = Field(
        default_factory=lambda: [EquipmentConfiguration()],
        description="Equipment configurations"
    )

    # ========================================================================
    # SENSOR CONFIGURATIONS
    # ========================================================================

    vibration_sensors: List[VibrationSensorConfiguration] = Field(
        default_factory=lambda: [
            VibrationSensorConfiguration(
                sensor_id="VS-001",
                sensor_tag="VT-P101A-DE-H",
                sensor_name="P-101A DE Horizontal",
                mounting_location="drive_end_horizontal"
            ),
            VibrationSensorConfiguration(
                sensor_id="VS-002",
                sensor_tag="VT-P101A-DE-V",
                sensor_name="P-101A DE Vertical",
                mounting_location="drive_end_vertical"
            ),
            VibrationSensorConfiguration(
                sensor_id="VS-003",
                sensor_tag="VT-P101A-NDE-H",
                sensor_name="P-101A NDE Horizontal",
                mounting_location="non_drive_end_horizontal"
            ),
        ],
        description="Vibration sensor configurations"
    )

    temperature_sensors: List[TemperatureSensorConfiguration] = Field(
        default_factory=lambda: [
            TemperatureSensorConfiguration(
                sensor_id="TS-001",
                sensor_tag="TE-P101A-DE-BRG",
                sensor_name="P-101A DE Bearing Temperature",
                measurement_point="bearing",
                mounting_location="drive_end_bearing"
            ),
            TemperatureSensorConfiguration(
                sensor_id="TS-002",
                sensor_tag="TE-P101A-NDE-BRG",
                sensor_name="P-101A NDE Bearing Temperature",
                measurement_point="bearing",
                mounting_location="non_drive_end_bearing"
            ),
        ],
        description="Temperature sensor configurations"
    )

    pressure_sensors: List[PressureSensorConfiguration] = Field(
        default_factory=lambda: [PressureSensorConfiguration()],
        description="Pressure sensor configurations"
    )

    # ========================================================================
    # OPERATING HOURS CONFIGURATIONS
    # ========================================================================

    operating_hours_configs: List[OperatingHoursConfiguration] = Field(
        default_factory=lambda: [OperatingHoursConfiguration()],
        description="Operating hours tracking configurations"
    )

    # ========================================================================
    # FAILURE MODE CONFIGURATIONS
    # ========================================================================

    failure_mode_configs: List[FailureModeConfiguration] = Field(
        default_factory=lambda: [
            FailureModeConfiguration(
                failure_mode_id="FM-001",
                failure_mode=FailureMode.BEARING_WEAR,
                equipment_type=EquipmentType.PUMP,
                severity=7,
                occurrence=5,
                detection=4
            ),
            FailureModeConfiguration(
                failure_mode_id="FM-002",
                failure_mode=FailureMode.MISALIGNMENT,
                equipment_type=EquipmentType.PUMP,
                severity=6,
                occurrence=4,
                detection=3
            ),
            FailureModeConfiguration(
                failure_mode_id="FM-003",
                failure_mode=FailureMode.IMBALANCE,
                equipment_type=EquipmentType.PUMP,
                severity=5,
                occurrence=3,
                detection=3
            ),
        ],
        description="Failure mode configurations"
    )

    # ========================================================================
    # MAINTENANCE SCHEDULE CONFIGURATIONS
    # ========================================================================

    maintenance_schedules: List[MaintenanceScheduleConfiguration] = Field(
        default_factory=lambda: [MaintenanceScheduleConfiguration()],
        description="Maintenance schedule configurations"
    )

    # ========================================================================
    # SPARE PARTS CONFIGURATIONS
    # ========================================================================

    spare_parts_configs: List[SparePartsConfiguration] = Field(
        default_factory=lambda: [SparePartsConfiguration()],
        description="Spare parts inventory configurations"
    )

    # ========================================================================
    # INTEGRATION CONFIGURATIONS
    # ========================================================================

    cms_integration: Optional[CMSIntegrationConfiguration] = Field(
        default_factory=CMSIntegrationConfiguration,
        description="Condition Monitoring System integration"
    )

    cmms_integration: Optional[CMMSIntegrationConfiguration] = Field(
        default_factory=CMMSIntegrationConfiguration,
        description="CMMS integration configuration"
    )

    # ========================================================================
    # ALERT CONFIGURATION
    # ========================================================================

    alert_config: AlertConfiguration = Field(
        default_factory=AlertConfiguration,
        description="Alert and notification configuration"
    )

    # ========================================================================
    # ML MODEL CONFIGURATIONS
    # ========================================================================

    ml_model_configs: List[MLModelConfiguration] = Field(
        default_factory=lambda: [
            MLModelConfiguration(
                model_id="ML-AD-001",
                model_name="Vibration Anomaly Detector",
                model_type=MLModelType.ISOLATION_FOREST,
                model_purpose="anomaly_detection"
            ),
            MLModelConfiguration(
                model_id="ML-RUL-001",
                model_name="RUL Predictor",
                model_type=MLModelType.GRADIENT_BOOSTING,
                model_purpose="rul_prediction"
            ),
        ],
        description="Machine learning model configurations"
    )

    # ========================================================================
    # DATA QUALITY CONFIGURATION
    # ========================================================================

    data_quality_config: DataQualityConfiguration = Field(
        default_factory=DataQualityConfiguration,
        description="Data quality monitoring configuration"
    )

    # ========================================================================
    # HISTORIAN INTEGRATION
    # ========================================================================

    historian_enabled: bool = Field(
        default=True,
        description="Enable process historian integration"
    )
    historian_logging_interval_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Historian logging interval"
    )
    historian_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Data retention period in days"
    )

    # ========================================================================
    # DETERMINISTIC SETTINGS (REGULATORY COMPLIANCE)
    # ========================================================================

    deterministic_mode: bool = Field(
        default=True,
        description="Enable deterministic mode (required for reproducible predictions)"
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
        description="Enable TLS 1.3 for integrations"
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
        description="Decimal precision for calculations"
    )

    # ========================================================================
    # OPERATIONAL SETTINGS
    # ========================================================================

    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (must be False in production)"
    )

    health_check_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Health check interval in seconds"
    )

    # ========================================================================
    # SITE INFORMATION
    # ========================================================================

    site_id: str = Field(
        default="SITE-001",
        description="Site identifier"
    )
    plant_id: str = Field(
        default="PLANT-001",
        description="Plant identifier"
    )

    # ========================================================================
    # VALIDATORS - COMPLIANCE ENFORCEMENT
    # ========================================================================

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Ensure temperature is 0.0 for deterministic operation."""
        if v != 0.0:
            raise ValueError(
                f"COMPLIANCE VIOLATION: temperature must be 0.0 for deterministic "
                f"predictive maintenance calculations. Got: {v}. This ensures "
                f"reproducible predictions for regulatory compliance."
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
                "COMPLIANCE VIOLATION: deterministic_mode must be True for "
                "predictive maintenance. All predictions must be reproducible "
                "for audit trails per ISO 55000."
            )
        return v

    @field_validator('zero_secrets')
    @classmethod
    def validate_zero_secrets(cls, v):
        """Ensure zero_secrets policy is enabled."""
        if not v:
            raise ValueError(
                "SECURITY VIOLATION: zero_secrets must be True. API keys and "
                "credentials must never be in config.py. Use environment variables "
                "or secrets manager."
            )
        return v

    @field_validator('tls_enabled')
    @classmethod
    def validate_tls(cls, v):
        """Ensure TLS is enabled."""
        if not v:
            raise ValueError(
                "SECURITY VIOLATION: tls_enabled must be True for production "
                "deployments. All integrations must use TLS 1.3."
            )
        return v

    @field_validator('enable_provenance')
    @classmethod
    def validate_provenance(cls, v):
        """Ensure provenance tracking is enabled."""
        if not v:
            raise ValueError(
                "COMPLIANCE VIOLATION: enable_provenance must be True. SHA-256 "
                "audit trails are required for all predictions per ISO 55000."
            )
        return v

    @field_validator('decimal_precision')
    @classmethod
    def validate_precision(cls, v):
        """Validate decimal precision for calculations."""
        if v < 10:
            raise ValueError(
                "COMPLIANCE VIOLATION: decimal_precision must be >= 10 for "
                f"predictive maintenance calculations. Got: {v}."
            )
        return v

    @model_validator(mode='after')
    def validate_environment_consistency(self):
        """Validate configuration consistency across environments."""
        if self.environment == 'production':
            # Production environment checks
            if not self.tls_enabled:
                raise ValueError(
                    "SECURITY VIOLATION: TLS required in production environment."
                )

            if not self.deterministic_mode:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Deterministic mode required in production."
                )

            if self.debug_mode:
                raise ValueError(
                    "SECURITY VIOLATION: debug_mode must be False in production."
                )

            if not self.enable_provenance:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Provenance tracking required in production."
                )

            if not self.enable_audit_logging:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Audit logging required in production."
                )

        # Validate calculation timeout is reasonable
        if self.calculation_timeout_seconds > 300:
            raise ValueError(
                "PERFORMANCE VIOLATION: calculation_timeout_seconds should not exceed "
                f"300 seconds for predictive maintenance. Got: {self.calculation_timeout_seconds}."
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

    def assert_security_ready(self) -> None:
        """
        Assert configuration meets security requirements.

        Raises:
            AssertionError: If any security requirements are not met
        """
        assert self.zero_secrets, "Zero secrets policy must be enforced"
        assert self.tls_enabled, "TLS encryption must be enabled"

    def assert_determinism_ready(self) -> None:
        """
        Assert configuration ensures deterministic predictions.

        Raises:
            AssertionError: If any determinism requirements are not met
        """
        assert self.deterministic_mode, "Deterministic mode must be enabled"
        assert self.temperature == 0.0, "Temperature must be 0.0"
        assert self.seed == 42, "Seed must be 42"
        assert self.decimal_precision >= 10, "Decimal precision must be >= 10"

    def calculate_config_hash(self) -> str:
        """
        Calculate SHA-256 hash of configuration for provenance tracking.

        Returns:
            str: SHA-256 hash of configuration JSON
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode()).hexdigest()

    def get_equipment_by_id(self, equipment_id: str) -> Optional[EquipmentConfiguration]:
        """
        Get equipment configuration by ID.

        Args:
            equipment_id: Equipment identifier

        Returns:
            EquipmentConfiguration or None if not found
        """
        for equipment in self.equipment_configs:
            if equipment.equipment_id == equipment_id:
                return equipment
        return None

    def get_sensors_for_equipment(
        self,
        equipment_id: str
    ) -> Dict[str, List[BaseModel]]:
        """
        Get all sensors associated with an equipment.

        Args:
            equipment_id: Equipment identifier

        Returns:
            Dictionary of sensor lists by type
        """
        # Note: In a real implementation, sensors would be linked to equipment
        # This is a placeholder showing the pattern
        return {
            "vibration": self.vibration_sensors,
            "temperature": self.temperature_sensors,
            "pressure": self.pressure_sensors
        }

    def get_failure_modes_for_equipment(
        self,
        equipment_type: EquipmentType
    ) -> List[FailureModeConfiguration]:
        """
        Get failure modes applicable to an equipment type.

        Args:
            equipment_type: Equipment type

        Returns:
            List of applicable failure mode configurations
        """
        return [
            fm for fm in self.failure_mode_configs
            if fm.equipment_type == equipment_type
        ]


# ============================================================================
# DEFAULT CONFIGURATION FACTORY
# ============================================================================

def create_default_config() -> PredictiveMaintenanceConfig:
    """
    Create default configuration for testing and demonstration.

    Returns:
        PredictiveMaintenanceConfig with default values for
        standard rotating equipment monitoring.
    """
    # Create default equipment configuration
    equipment = EquipmentConfiguration(
        equipment_id="EQ-P101A",
        equipment_tag="P-101A",
        equipment_name="Feed Water Pump A",
        equipment_type=EquipmentType.PUMP,
        manufacturer="Flowserve",
        model_number="3x4-10",
        rated_power_kw=75.0,
        rated_speed_rpm=1800.0,
        machine_class=MachineClass.CLASS_II,
        criticality_rating="A",
        maintenance_strategy=MaintenanceStrategy.PREDICTIVE
    )

    # Create vibration sensors per ISO 10816
    vibration_sensors = [
        VibrationSensorConfiguration(
            sensor_id="VS-P101A-DE-H",
            sensor_tag="VT-P101A-DE-H",
            sensor_name="P-101A Drive End Horizontal",
            mounting_location="drive_end_horizontal",
            orientation="horizontal",
            alarm_threshold_zone_b=2.8,
            alarm_threshold_zone_c=4.5,
            trip_threshold_zone_d=7.1
        ),
        VibrationSensorConfiguration(
            sensor_id="VS-P101A-DE-V",
            sensor_tag="VT-P101A-DE-V",
            sensor_name="P-101A Drive End Vertical",
            mounting_location="drive_end_vertical",
            orientation="vertical",
            alarm_threshold_zone_b=2.8,
            alarm_threshold_zone_c=4.5,
            trip_threshold_zone_d=7.1
        ),
        VibrationSensorConfiguration(
            sensor_id="VS-P101A-DE-A",
            sensor_tag="VT-P101A-DE-A",
            sensor_name="P-101A Drive End Axial",
            mounting_location="drive_end_axial",
            orientation="axial",
            alarm_threshold_zone_b=2.8,
            alarm_threshold_zone_c=4.5,
            trip_threshold_zone_d=7.1
        ),
        VibrationSensorConfiguration(
            sensor_id="VS-P101A-NDE-H",
            sensor_tag="VT-P101A-NDE-H",
            sensor_name="P-101A Non-Drive End Horizontal",
            mounting_location="non_drive_end_horizontal",
            orientation="horizontal",
            alarm_threshold_zone_b=2.8,
            alarm_threshold_zone_c=4.5,
            trip_threshold_zone_d=7.1
        ),
        VibrationSensorConfiguration(
            sensor_id="VS-P101A-NDE-V",
            sensor_tag="VT-P101A-NDE-V",
            sensor_name="P-101A Non-Drive End Vertical",
            mounting_location="non_drive_end_vertical",
            orientation="vertical",
            alarm_threshold_zone_b=2.8,
            alarm_threshold_zone_c=4.5,
            trip_threshold_zone_d=7.1
        ),
    ]

    # Create temperature sensors
    temperature_sensors = [
        TemperatureSensorConfiguration(
            sensor_id="TS-P101A-DE-BRG",
            sensor_tag="TE-P101A-DE-BRG",
            sensor_name="P-101A Drive End Bearing Temperature",
            measurement_point="bearing",
            mounting_location="drive_end_bearing",
            alarm_high_c=75.0,
            alarm_high_high_c=85.0,
            trip_high_c=95.0
        ),
        TemperatureSensorConfiguration(
            sensor_id="TS-P101A-NDE-BRG",
            sensor_tag="TE-P101A-NDE-BRG",
            sensor_name="P-101A Non-Drive End Bearing Temperature",
            measurement_point="bearing",
            mounting_location="non_drive_end_bearing",
            alarm_high_c=75.0,
            alarm_high_high_c=85.0,
            trip_high_c=95.0
        ),
        TemperatureSensorConfiguration(
            sensor_id="TS-P101A-MTR-WDG",
            sensor_tag="TE-P101A-MTR-WDG",
            sensor_name="P-101A Motor Winding Temperature",
            measurement_point="winding",
            mounting_location="motor_winding",
            alarm_high_c=120.0,
            alarm_high_high_c=130.0,
            trip_high_c=140.0
        ),
    ]

    # Create failure mode configurations
    failure_modes = [
        FailureModeConfiguration(
            failure_mode_id="FM-P101A-BRG",
            failure_mode=FailureMode.BEARING_WEAR,
            equipment_type=EquipmentType.PUMP,
            description="Bearing wear due to fatigue, contamination, or inadequate lubrication",
            failure_effect="Increased vibration, elevated temperature, eventual seizure",
            severity=7,
            occurrence=5,
            detection=3,
            detection_methods=["vibration_analysis", "temperature_monitoring", "oil_analysis"],
            pf_interval_hours=500,
            weibull_parameters=get_weibull_parameters(EquipmentType.PUMP, FailureMode.BEARING_WEAR)
        ),
        FailureModeConfiguration(
            failure_mode_id="FM-P101A-MISAL",
            failure_mode=FailureMode.MISALIGNMENT,
            equipment_type=EquipmentType.PUMP,
            description="Shaft misalignment between pump and motor",
            failure_effect="High 2x vibration, coupling wear, bearing damage",
            severity=6,
            occurrence=4,
            detection=2,
            detection_methods=["vibration_analysis", "laser_alignment"],
            pf_interval_hours=1000
        ),
        FailureModeConfiguration(
            failure_mode_id="FM-P101A-SEAL",
            failure_mode=FailureMode.SEAL_FAILURE,
            equipment_type=EquipmentType.PUMP,
            description="Mechanical seal failure",
            failure_effect="Leakage, environmental release, pump damage",
            severity=8,
            occurrence=4,
            detection=4,
            detection_methods=["visual_inspection", "leak_detection"],
            pf_interval_hours=200,
            weibull_parameters=get_weibull_parameters(EquipmentType.PUMP, FailureMode.SEAL_FAILURE)
        ),
    ]

    # Create ML model configurations
    ml_models = [
        MLModelConfiguration(
            model_id="ML-AD-P101A",
            model_name="P-101A Anomaly Detector",
            model_type=MLModelType.ISOLATION_FOREST,
            model_purpose="anomaly_detection",
            feature_columns=[
                "vibration_de_h", "vibration_de_v", "vibration_de_a",
                "vibration_nde_h", "vibration_nde_v",
                "temp_de_brg", "temp_nde_brg", "temp_winding",
                "motor_current", "discharge_pressure"
            ],
            n_estimators=100,
            contamination=0.05,
            min_precision=0.90,
            min_recall=0.85
        ),
        MLModelConfiguration(
            model_id="ML-RUL-P101A",
            model_name="P-101A RUL Predictor",
            model_type=MLModelType.GRADIENT_BOOSTING,
            model_purpose="rul_prediction",
            feature_columns=[
                "vibration_trend", "temperature_trend",
                "operating_hours", "start_stop_cycles",
                "load_factor", "speed_deviation"
            ],
            target_column="remaining_useful_life",
            min_precision=0.85,
            min_recall=0.80
        ),
    ]

    return PredictiveMaintenanceConfig(
        agent_id="GL-013",
        agent_name="PredictMaint",
        version="1.0.0",
        environment="development",
        equipment_configs=[equipment],
        vibration_sensors=vibration_sensors,
        temperature_sensors=temperature_sensors,
        failure_mode_configs=failure_modes,
        ml_model_configs=ml_models,
        site_id="DEMO-SITE-001",
        plant_id="DEMO-PLANT-001"
    )


def create_turbine_config() -> PredictiveMaintenanceConfig:
    """
    Create configuration for turbine monitoring application.

    Returns:
        PredictiveMaintenanceConfig configured for steam/gas turbine
        predictive maintenance.
    """
    config = create_default_config()

    # Update for turbine application
    turbine = EquipmentConfiguration(
        equipment_id="EQ-TG-001",
        equipment_tag="TG-001",
        equipment_name="Steam Turbine Generator",
        equipment_type=EquipmentType.TURBINE,
        manufacturer="GE Power",
        model_number="D11",
        rated_power_kw=50000.0,
        rated_speed_rpm=3600.0,
        machine_class=MachineClass.CLASS_IV,  # Large machine, flexible foundation
        criticality_rating="A",
        safety_critical=True,
        maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
        vibration_limits=get_iso_10816_limits(MachineClass.CLASS_IV)
    )

    config.equipment_configs = [turbine]

    # Update vibration thresholds for Class IV machines
    for sensor in config.vibration_sensors:
        sensor.alarm_threshold_zone_b = 7.1
        sensor.alarm_threshold_zone_c = 11.2
        sensor.trip_threshold_zone_d = 18.0

    return config


def create_compressor_config() -> PredictiveMaintenanceConfig:
    """
    Create configuration for compressor monitoring application.

    Returns:
        PredictiveMaintenanceConfig configured for centrifugal/reciprocating
        compressor predictive maintenance.
    """
    config = create_default_config()

    # Update for compressor application
    compressor = EquipmentConfiguration(
        equipment_id="EQ-C-101",
        equipment_tag="C-101",
        equipment_name="Process Gas Compressor",
        equipment_type=EquipmentType.COMPRESSOR,
        manufacturer="Atlas Copco",
        model_number="ZH-350",
        rated_power_kw=350.0,
        rated_speed_rpm=8000.0,
        machine_class=MachineClass.CLASS_III,
        criticality_rating="A",
        safety_critical=True,
        maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
        vibration_limits=get_iso_10816_limits(MachineClass.CLASS_III)
    )

    config.equipment_configs = [compressor]

    # Add compressor-specific failure modes
    config.failure_mode_configs.extend([
        FailureModeConfiguration(
            failure_mode_id="FM-C101-SURGE",
            failure_mode=FailureMode.STARVATION,  # Surge is similar to starvation
            equipment_type=EquipmentType.COMPRESSOR,
            description="Compressor surge due to low flow conditions",
            failure_effect="Rapid pressure/flow oscillations, mechanical damage",
            severity=9,
            occurrence=3,
            detection=2,
            detection_methods=["vibration_analysis", "surge_detection"],
            pf_interval_hours=1  # Very fast progression
        ),
    ])

    return config


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enumerations
    "EquipmentType",
    "FailureMode",
    "MaintenanceStrategy",
    "HealthStatus",
    "AlertSeverity",
    "SensorType",
    "VibrationUnit",
    "MachineClass",
    "MountingType",
    "CMMSType",
    "WorkOrderPriority",
    "MLModelType",
    "DataQualityLevel",
    # ISO 10816 Standards
    "ISO10816VibrationLimits",
    "ISO_10816_STANDARDS",
    "get_iso_10816_limits",
    # Weibull Parameters
    "WeibullParameters",
    "WEIBULL_DEFAULTS",
    "get_weibull_parameters",
    # Equipment Configuration
    "EquipmentConfiguration",
    # Sensor Configurations
    "VibrationSensorConfiguration",
    "TemperatureSensorConfiguration",
    "PressureSensorConfiguration",
    # Operating Hours
    "OperatingHoursConfiguration",
    # Failure Mode Configuration
    "FailureModeConfiguration",
    # Maintenance Configuration
    "MaintenanceTaskConfiguration",
    "MaintenanceScheduleConfiguration",
    # Spare Parts Configuration
    "SparePartConfiguration",
    "SparePartsConfiguration",
    # Integration Configurations
    "CMSIntegrationConfiguration",
    "CMMSIntegrationConfiguration",
    # Alert Configuration
    "AlertThresholdConfiguration",
    "AlertConfiguration",
    # ML Model Configuration
    "MLModelConfiguration",
    # Data Quality Configuration
    "DataQualityConfiguration",
    # Main Configuration
    "PredictiveMaintenanceConfig",
    # Factory Functions
    "create_default_config",
    "create_turbine_config",
    "create_compressor_config",
]
