# -*- coding: utf-8 -*-
"""
GL-024 Air Preheater Agent - Comprehensive Data Schemas Module

This module defines Pydantic v2 data models for industrial air preheater
monitoring, analysis, and optimization. Supports all major air preheater types
including regenerative (Ljungstrom), recuperative (tubular/plate), and heat pipe
configurations.

Air preheaters are critical heat recovery equipment in boiler systems that
recover waste heat from flue gas to preheat combustion air, improving overall
boiler efficiency by 2-10%. Proper monitoring prevents costly issues like
cold-end corrosion, excessive leakage, and fouling.

Standards Compliance:
    - ASME PTC 4.3 Air Heater Test Code (Performance calculations)
    - ASME PTC 4.1 Steam Generating Units (Efficiency impact)
    - API 560 Fired Heaters for General Refinery Service
    - NACE SP0198 Control of Corrosion Under Thermal Insulation
    - EPA Method 19 F-Factor calculations (Leakage impact on O2)

Data Model Categories:
    - Air preheater type and status enums
    - Gas-side input models (temperature, flow, composition)
    - Air-side input models (temperature, flow, humidity)
    - Operating parameters (pressure drops, rotational speed, seals)
    - Performance baseline models
    - Heat transfer analysis output (effectiveness, NTU, LMTD, X-ratio)
    - Leakage analysis output (O2 rise method, direct measurement)
    - Cold-end protection output (acid dew point, corrosion risk)
    - Fouling analysis output (cleanliness factor, pressure drop ratio)
    - Optimization recommendations with energy savings quantification

Engineering Context:
    Regenerative (Ljungstrom) preheaters use rotating heat storage elements
    (baskets) that alternately absorb heat from flue gas and release it to
    combustion air. They achieve high effectiveness (70-85%) but are subject
    to air-to-gas leakage through seals.

    Recuperative preheaters (tubular/plate) use fixed heat exchange surfaces.
    They have lower effectiveness (50-70%) but near-zero leakage and simpler
    maintenance.

    Heat pipe preheaters use sealed tubes with working fluid for heat transfer.
    They offer excellent cold-end corrosion resistance and moderate effectiveness.

Example:
    >>> from greenlang.agents.process_heat.gl_024_air_preheater.schemas import (
    ...     AirPreheaterInput,
    ...     AirPreheaterOutput,
    ...     PreheaterType,
    ... )
    >>> input_data = AirPreheaterInput(
    ...     preheater_id="APH-001",
    ...     preheater_type=PreheaterType.REGENERATIVE,
    ...     gas_inlet_temp_f=650.0,
    ...     gas_outlet_temp_f=300.0,
    ...     air_inlet_temp_f=80.0,
    ...     air_outlet_temp_f=550.0,
    ...     gas_flow_lb_hr=500000.0,
    ...     air_flow_lb_hr=480000.0,
    ... )

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ENUMS - AIR PREHEATER CLASSIFICATION
# =============================================================================


class PreheaterType(str, Enum):
    """
    Air preheater type classification.

    Different preheater types have distinct operating characteristics,
    failure modes, and optimization strategies.

    Attributes:
        REGENERATIVE: Ljungstrom-type rotating regenerative air heater.
            Heat storage elements (baskets) rotate between gas and air sides.
            Typical effectiveness: 70-85%. Subject to seal leakage.
        RECUPERATIVE_TUBULAR: Fixed tubular heat exchanger design.
            Gas flows over tubes, air flows through tubes (or vice versa).
            Typical effectiveness: 50-65%. Near-zero leakage.
        RECUPERATIVE_PLATE: Fixed plate heat exchanger design.
            Alternating gas and air channels with corrugated plates.
            Typical effectiveness: 60-70%. Compact design.
        HEAT_PIPE: Sealed heat pipe technology.
            Working fluid evaporates/condenses for heat transfer.
            Excellent cold-end corrosion resistance. Typical effectiveness: 55-70%.
    """
    REGENERATIVE = "regenerative"
    RECUPERATIVE_TUBULAR = "recuperative_tubular"
    RECUPERATIVE_PLATE = "recuperative_plate"
    HEAT_PIPE = "heat_pipe"
    # Additional aliases for compatibility with config.py
    LJUNGSTROM = "ljungstrom"
    ROTHEMUHLE = "rothemuhle"
    TUBULAR = "tubular"
    PLATE = "plate"


# Backward compatibility alias
AirPreheaterType = PreheaterType


class PreheaterStatus(str, Enum):
    """Air preheater operating status."""
    OFFLINE = "offline"
    STANDBY = "standby"
    WARMING = "warming"
    NORMAL = "normal"
    DEGRADED = "degraded"
    BYPASS = "bypass"
    ALARM = "alarm"
    TRIP = "trip"


class RotorStatus(str, Enum):
    """Regenerative preheater rotor status (Ljungstrom type only)."""
    STOPPED = "stopped"
    STARTING = "starting"
    NORMAL_SPEED = "normal_speed"
    SLOW_SPEED = "slow_speed"
    CREEP_MODE = "creep_mode"
    STOPPING = "stopping"
    FAULTED = "faulted"


class FoulingSeverity(str, Enum):
    """
    Fouling severity classification for air preheater.

    Fouling in air preheaters occurs from:
    - Fly ash deposition (gas side)
    - Ammonium bisulfate (ABS) deposition (SCR applications)
    - Moisture condensation and corrosion products
    """
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class CorrosionRiskLevel(str, Enum):
    """
    Cold-end corrosion risk classification.

    Corrosion occurs when metal temperatures fall below the acid dew point,
    causing sulfuric acid condensation and metal wastage.
    """
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class LeakageDirection(str, Enum):
    """
    Leakage direction classification for regenerative preheaters.

    Air-to-gas leakage increases O2 in flue gas, affecting emissions monitoring.
    Gas-to-air leakage introduces combustion products into combustion air.
    """
    AIR_TO_GAS = "air_to_gas"
    GAS_TO_AIR = "gas_to_air"
    COMBINED = "combined"


class CleaningMethod(str, Enum):
    """Air preheater cleaning methods."""
    SOOT_BLOWING = "soot_blowing"
    WATER_WASHING = "water_washing"
    STEAM_CLEANING = "steam_cleaning"
    HIGH_PRESSURE_WATER = "high_pressure_water"
    DRY_ICE_BLASTING = "dry_ice_blasting"
    MANUAL = "manual"


class AcidDewPointMethod(str, Enum):
    """
    Acid dew point calculation methods.

    Different correlations exist for predicting sulfuric acid dew point
    based on SO3 concentration and moisture content in flue gas.
    """
    VERHOFF_BANCHERO = "verhoff_banchero"
    OKKES = "okkes"
    ZARENezhad = "zarenezhad"
    MUELLER = "mueller"
    PIERCE = "pierce"
    MEASURED = "measured"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Data validation status."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    STALE = "stale"
    UNCHECKED = "unchecked"


# =============================================================================
# BASE SCHEMAS
# =============================================================================


class BasePreheaterSchema(BaseModel):
    """
    Base schema for all air preheater data models.

    Provides common fields for timestamps, identifiers, and provenance
    tracking required across all air preheater data types.
    """

    id: Optional[str] = Field(
        default=None,
        description="Unique record identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp (UTC)"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp (UTC)"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    def calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 provenance hash for this record.

        Returns:
            SHA-256 hex digest of record content
        """
        data = self.model_dump(exclude={"provenance_hash", "updated_at"})
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def update_provenance(self) -> None:
        """Update provenance hash and timestamp."""
        self.updated_at = datetime.now(timezone.utc)
        self.provenance_hash = self.calculate_provenance_hash()

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
        },
        "validate_assignment": True,
        "extra": "forbid",
    }


# =============================================================================
# INPUT SCHEMAS - GAS SIDE
# =============================================================================


class GasComposition(BaseModel):
    """
    Flue gas composition for acid dew point and heat transfer calculations.

    Accurate gas composition is critical for predicting acid dew point
    and assessing cold-end corrosion risk. SO3 is the primary driver
    of sulfuric acid formation.

    Attributes:
        o2_pct: Oxygen concentration at preheater inlet (dry basis).
            Typical range: 2-6% for coal, 2-4% for gas firing.
        co2_pct: Carbon dioxide concentration (dry basis).
            Used for excess air calculations.
        so2_ppm: Sulfur dioxide concentration.
            Primary source of sulfuric acid formation.
        so3_ppm: Sulfur trioxide concentration.
            Direct contributor to acid dew point. SO3 is 10-100x more
            corrosive than SO2. Typical: 1-5% of total SOx.
        moisture_pct: Water vapor content (wet basis).
            Affects acid dew point and heat transfer.
        n2_pct: Nitrogen concentration (balance gas).
    """

    o2_pct: float = Field(
        ...,
        ge=0.0,
        le=21.0,
        description="Oxygen concentration (% dry basis)"
    )
    co2_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=25.0,
        description="CO2 concentration (% dry basis)"
    )
    so2_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10000.0,
        description="SO2 concentration (ppm dry basis)"
    )
    so3_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="SO3 concentration (ppm). Critical for acid dew point."
    )
    moisture_pct: float = Field(
        default=8.0,
        ge=0.0,
        le=30.0,
        description="Moisture content (% wet basis). Typical: 6-12% for coal, 15-20% for gas."
    )
    n2_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=90.0,
        description="Nitrogen concentration (% dry basis)"
    )
    hcl_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Hydrogen chloride concentration (ppm). Relevant for waste/biomass fuels."
    )

    @field_validator('so3_ppm')
    @classmethod
    def validate_so3(cls, v: Optional[float], info) -> Optional[float]:
        """Warn if SO3 seems high relative to typical SO2/SO3 ratios."""
        # SO3 is typically 1-5% of SO2
        return v


class GasSideInput(BaseModel):
    """
    Gas-side operating data for air preheater analysis.

    The gas side (flue gas) is the heat source. Gas enters hot and exits
    cooler after transferring heat to combustion air.

    Attributes:
        inlet_temp_f: Flue gas inlet temperature (F).
            Typical range: 550-750F for utility boilers.
        outlet_temp_f: Flue gas outlet temperature (F).
            Critical for efficiency and acid dew point. Typical: 250-350F.
        flow_rate_lb_hr: Mass flow rate (lb/hr).
            Based on fuel firing rate and excess air.
        composition: Flue gas composition for detailed analysis.
        inlet_pressure_in_wc: Gas inlet pressure (in. WC gauge).
            Negative values indicate draft (suction).
        outlet_pressure_in_wc: Gas outlet pressure (in. WC gauge).
        pressure_drop_in_wc: Measured pressure drop across preheater.
    """

    inlet_temp_f: float = Field(
        ...,
        ge=200.0,
        le=1200.0,
        description="Flue gas inlet temperature (F). Typical: 550-750F."
    )
    outlet_temp_f: float = Field(
        ...,
        ge=100.0,
        le=800.0,
        description="Flue gas outlet temperature (F). Typical: 250-350F."
    )
    flow_rate_lb_hr: float = Field(
        ...,
        gt=0.0,
        description="Flue gas mass flow rate (lb/hr)"
    )
    composition: Optional[GasComposition] = Field(
        default=None,
        description="Flue gas composition for acid dew point calculations"
    )
    inlet_pressure_in_wc: float = Field(
        default=0.0,
        ge=-50.0,
        le=50.0,
        description="Gas inlet pressure (in. WC gauge). Negative = draft."
    )
    outlet_pressure_in_wc: float = Field(
        default=0.0,
        ge=-50.0,
        le=50.0,
        description="Gas outlet pressure (in. WC gauge)"
    )
    pressure_drop_in_wc: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Gas-side pressure drop (in. WC). Indicator of fouling."
    )
    specific_heat_btu_lb_f: float = Field(
        default=0.25,
        ge=0.20,
        le=0.35,
        description="Gas specific heat (BTU/lb-F). Typical: 0.24-0.26."
    )

    @field_validator('pressure_drop_in_wc', mode='before')
    @classmethod
    def calculate_pressure_drop(cls, v, info):
        """Calculate pressure drop if not provided."""
        if v is None:
            values = info.data
            inlet = values.get('inlet_pressure_in_wc', 0)
            outlet = values.get('outlet_pressure_in_wc', 0)
            return abs(inlet - outlet)
        return v

    @model_validator(mode='after')
    def validate_temperatures(self):
        """Validate gas temperatures are physically reasonable."""
        if self.outlet_temp_f >= self.inlet_temp_f:
            raise ValueError(
                f"Gas outlet temp ({self.outlet_temp_f}F) must be less than "
                f"inlet temp ({self.inlet_temp_f}F). Heat flows from hot gas to cold air."
            )
        return self


# =============================================================================
# INPUT SCHEMAS - AIR SIDE
# =============================================================================


class AirSideInput(BaseModel):
    """
    Air-side operating data for air preheater analysis.

    The air side (combustion air) is the heat sink. Air enters cold and
    exits hot after absorbing heat from flue gas.

    Attributes:
        inlet_temp_f: Combustion air inlet temperature (F).
            Typically ambient temperature or slightly preheated.
        outlet_temp_f: Preheated air outlet temperature (F).
            Target based on boiler design. Typical: 500-700F.
        flow_rate_lb_hr: Mass flow rate (lb/hr).
            Should match theoretical air plus excess air.
        relative_humidity_pct: Inlet air relative humidity.
            Affects moisture pickup in air stream.
        inlet_pressure_in_wc: Air inlet pressure (in. WC gauge).
        outlet_pressure_in_wc: Air outlet pressure (in. WC gauge).
        pressure_drop_in_wc: Measured pressure drop across preheater.
    """

    inlet_temp_f: float = Field(
        ...,
        ge=-40.0,
        le=300.0,
        description="Combustion air inlet temperature (F). Typically ambient."
    )
    outlet_temp_f: float = Field(
        ...,
        ge=50.0,
        le=900.0,
        description="Preheated air outlet temperature (F). Typical: 500-700F."
    )
    flow_rate_lb_hr: float = Field(
        ...,
        gt=0.0,
        description="Combustion air mass flow rate (lb/hr)"
    )
    relative_humidity_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Inlet air relative humidity (%)"
    )
    inlet_pressure_in_wc: float = Field(
        default=0.0,
        ge=-20.0,
        le=100.0,
        description="Air inlet pressure (in. WC gauge)"
    )
    outlet_pressure_in_wc: float = Field(
        default=0.0,
        ge=-20.0,
        le=100.0,
        description="Air outlet pressure (in. WC gauge)"
    )
    pressure_drop_in_wc: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Air-side pressure drop (in. WC). Indicator of fouling."
    )
    specific_heat_btu_lb_f: float = Field(
        default=0.24,
        ge=0.22,
        le=0.28,
        description="Air specific heat (BTU/lb-F). Typical: 0.24."
    )

    @field_validator('pressure_drop_in_wc', mode='before')
    @classmethod
    def calculate_pressure_drop(cls, v, info):
        """Calculate pressure drop if not provided."""
        if v is None:
            values = info.data
            inlet = values.get('inlet_pressure_in_wc', 0)
            outlet = values.get('outlet_pressure_in_wc', 0)
            return abs(inlet - outlet)
        return v

    @model_validator(mode='after')
    def validate_temperatures(self):
        """Validate air temperatures are physically reasonable."""
        if self.outlet_temp_f <= self.inlet_temp_f:
            raise ValueError(
                f"Air outlet temp ({self.outlet_temp_f}F) must be greater than "
                f"inlet temp ({self.inlet_temp_f}F). Air absorbs heat from gas."
            )
        return self


# =============================================================================
# INPUT SCHEMAS - OPERATING DATA
# =============================================================================


class RegenerativeOperatingData(BaseModel):
    """
    Operating data specific to regenerative (Ljungstrom) air preheaters.

    Regenerative preheaters have rotating heat storage elements and
    require monitoring of rotational speed and seal clearances.

    Attributes:
        rotor_speed_rpm: Current rotor rotational speed (RPM).
            Typical range: 1-3 RPM. Slower = more heat transfer but more leakage.
        design_speed_rpm: Design rotor speed for reference.
        rotor_status: Current rotor operational status.
        radial_seal_clearance_in: Radial seal gap (inches).
            Larger clearance = more leakage.
        axial_seal_clearance_in: Axial seal gap (inches).
        circumferential_seal_clearance_in: Circumferential seal gap.
        sector_plate_position: Hot/cold end sector plate angles.
        rotor_drive_motor_amps: Motor current draw.
            Rising current may indicate bearing issues.
    """

    rotor_speed_rpm: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Rotor speed (RPM). Typical: 1-3 RPM."
    )
    design_speed_rpm: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Design rotor speed (RPM)"
    )
    rotor_status: RotorStatus = Field(
        default=RotorStatus.NORMAL_SPEED,
        description="Current rotor operational status"
    )
    radial_seal_clearance_in: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Radial seal clearance (inches). Design: 0.1-0.25 in."
    )
    axial_seal_clearance_in: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Axial seal clearance (inches)"
    )
    circumferential_seal_clearance_in: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Circumferential seal clearance (inches)"
    )
    sector_plate_angle_hot_deg: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=360.0,
        description="Hot-end sector plate angle (degrees)"
    )
    sector_plate_angle_cold_deg: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=360.0,
        description="Cold-end sector plate angle (degrees)"
    )
    rotor_drive_motor_amps: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Rotor drive motor current (Amps)"
    )
    rotor_drive_motor_amps_design: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Design motor current (Amps)"
    )
    bearing_temperature_hot_end_f: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=400.0,
        description="Hot-end bearing temperature (F)"
    )
    bearing_temperature_cold_end_f: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=300.0,
        description="Cold-end bearing temperature (F)"
    )

    @field_validator('rotor_speed_rpm')
    @classmethod
    def validate_speed(cls, v: float) -> float:
        """Validate rotor speed is reasonable."""
        if v > 0 and v < 0.1:
            raise ValueError(
                f"Rotor speed {v} RPM is unusually slow. "
                "Check for drive issues or confirm creep mode."
            )
        return v


class SootBlowerStatus(BaseModel):
    """
    Soot blower status for air preheater cleaning systems.

    Air preheaters require periodic cleaning to remove ash deposits
    and maintain heat transfer performance.
    """

    soot_blower_available: bool = Field(
        default=True,
        description="Soot blowing system available"
    )
    soot_blower_in_operation: bool = Field(
        default=False,
        description="Soot blower currently operating"
    )
    last_soot_blow_timestamp: Optional[datetime] = Field(
        default=None,
        description="Last soot blow completion time"
    )
    hours_since_last_blow: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Hours since last soot blow"
    )
    soot_blow_steam_pressure_psig: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Soot blowing steam pressure (psig)"
    )
    soot_blow_frequency_hr: float = Field(
        default=8.0,
        ge=1.0,
        le=168.0,
        description="Normal soot blow interval (hours)"
    )


class PerformanceBaseline(BaseModel):
    """
    Design and baseline performance parameters for air preheater.

    Baseline data is essential for detecting performance degradation
    and calculating efficiency losses due to fouling and leakage.

    Attributes:
        design_gas_inlet_temp_f: Design gas inlet temperature.
        design_gas_outlet_temp_f: Design gas outlet temperature.
        design_air_inlet_temp_f: Design air inlet temperature.
        design_air_outlet_temp_f: Design air outlet temperature.
        design_effectiveness: Design heat transfer effectiveness (0-1).
        design_gas_dp_in_wc: Design gas-side pressure drop.
        design_air_dp_in_wc: Design air-side pressure drop.
        design_leakage_pct: Design air leakage (regenerative only).
        acceptance_test_date: Date of last performance test.
    """

    design_gas_inlet_temp_f: float = Field(
        ...,
        ge=200.0,
        le=1200.0,
        description="Design gas inlet temperature (F)"
    )
    design_gas_outlet_temp_f: float = Field(
        ...,
        ge=100.0,
        le=800.0,
        description="Design gas outlet temperature (F)"
    )
    design_air_inlet_temp_f: float = Field(
        ...,
        ge=-40.0,
        le=200.0,
        description="Design air inlet temperature (F)"
    )
    design_air_outlet_temp_f: float = Field(
        ...,
        ge=200.0,
        le=900.0,
        description="Design air outlet temperature (F)"
    )
    design_effectiveness: float = Field(
        ...,
        ge=0.3,
        le=0.95,
        description="Design heat transfer effectiveness (0-1). Typical: 0.65-0.85."
    )
    design_ntu: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=10.0,
        description="Design Number of Transfer Units"
    )
    design_gas_flow_lb_hr: float = Field(
        ...,
        gt=0.0,
        description="Design gas flow rate (lb/hr)"
    )
    design_air_flow_lb_hr: float = Field(
        ...,
        gt=0.0,
        description="Design air flow rate (lb/hr)"
    )
    design_gas_dp_in_wc: float = Field(
        ...,
        ge=0.0,
        le=15.0,
        description="Design gas-side pressure drop (in. WC)"
    )
    design_air_dp_in_wc: float = Field(
        ...,
        ge=0.0,
        le=15.0,
        description="Design air-side pressure drop (in. WC)"
    )
    design_leakage_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=25.0,
        description="Design air leakage (%). Regenerative only. Typical: 5-10%."
    )
    design_heat_duty_mmbtu_hr: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Design heat duty (MMBTU/hr)"
    )
    design_x_ratio: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=2.0,
        description="Design X-ratio (gas Cp*m / air Cp*m)"
    )
    heat_transfer_surface_ft2: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Total heat transfer surface area (ft2)"
    )
    acceptance_test_date: Optional[datetime] = Field(
        default=None,
        description="Date of acceptance performance test"
    )
    clean_ua_btu_hr_f: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Clean UA value (BTU/hr-F)"
    )


# =============================================================================
# COMPREHENSIVE INPUT MODEL
# =============================================================================


class AirPreheaterInput(BasePreheaterSchema):
    """
    Complete input data for GL-024 Air Preheater Agent analysis.

    Aggregates all gas-side, air-side, and operating data required
    for comprehensive air preheater performance analysis.

    Example:
        >>> input_data = AirPreheaterInput(
        ...     preheater_id="APH-001",
        ...     preheater_type=PreheaterType.REGENERATIVE,
        ...     gas_inlet_temp_f=650.0,
        ...     gas_outlet_temp_f=300.0,
        ...     air_inlet_temp_f=80.0,
        ...     air_outlet_temp_f=550.0,
        ...     gas_flow_lb_hr=500000.0,
        ...     air_flow_lb_hr=480000.0,
        ...     boiler_load_pct=85.0,
        ... )
    """

    # Identification
    preheater_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Air preheater identifier"
    )
    preheater_type: PreheaterType = Field(
        ...,
        description="Air preheater type (regenerative, recuperative, heat_pipe)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp (UTC)"
    )
    operating_status: PreheaterStatus = Field(
        default=PreheaterStatus.NORMAL,
        description="Current operating status"
    )

    # Operating conditions
    boiler_load_pct: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Current boiler load (%)"
    )

    # Gas side - simplified direct inputs
    gas_inlet_temp_f: float = Field(
        ...,
        ge=200.0,
        le=1200.0,
        description="Flue gas inlet temperature (F)"
    )
    gas_outlet_temp_f: float = Field(
        ...,
        ge=100.0,
        le=800.0,
        description="Flue gas outlet temperature (F)"
    )
    gas_flow_lb_hr: float = Field(
        ...,
        gt=0.0,
        description="Flue gas mass flow rate (lb/hr)"
    )
    gas_dp_in_wc: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Gas-side pressure drop (in. WC)"
    )

    # Air side - simplified direct inputs
    air_inlet_temp_f: float = Field(
        ...,
        ge=-40.0,
        le=300.0,
        description="Combustion air inlet temperature (F)"
    )
    air_outlet_temp_f: float = Field(
        ...,
        ge=50.0,
        le=900.0,
        description="Preheated air outlet temperature (F)"
    )
    air_flow_lb_hr: float = Field(
        ...,
        gt=0.0,
        description="Combustion air mass flow rate (lb/hr)"
    )
    air_dp_in_wc: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Air-side pressure drop (in. WC)"
    )

    # Detailed gas-side data (optional)
    gas_side: Optional[GasSideInput] = Field(
        default=None,
        description="Detailed gas-side operating data"
    )

    # Detailed air-side data (optional)
    air_side: Optional[AirSideInput] = Field(
        default=None,
        description="Detailed air-side operating data"
    )

    # Gas composition for acid dew point
    gas_composition: Optional[GasComposition] = Field(
        default=None,
        description="Flue gas composition for detailed analysis"
    )

    # Regenerative-specific data
    regenerative_data: Optional[RegenerativeOperatingData] = Field(
        default=None,
        description="Operating data for regenerative (Ljungstrom) preheaters"
    )

    # Leakage measurement (if available)
    measured_leakage_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=50.0,
        description="Measured air leakage (%). From O2 rise or direct measurement."
    )
    o2_inlet_gas_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=21.0,
        description="O2 at preheater gas inlet (%) for leakage calculation"
    )
    o2_outlet_gas_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=21.0,
        description="O2 at preheater gas outlet (%) for leakage calculation"
    )

    # Cold-end temperatures
    cold_end_avg_metal_temp_f: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=500.0,
        description="Average cold-end metal temperature (F)"
    )
    cold_end_min_metal_temp_f: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=500.0,
        description="Minimum cold-end metal temperature (F)"
    )
    cold_end_metal_temps_f: Optional[List[float]] = Field(
        default=None,
        description="Multiple cold-end metal temperature readings (F)"
    )

    # Soot blower status
    soot_blower: Optional[SootBlowerStatus] = Field(
        default=None,
        description="Soot blower status and history"
    )

    # Baseline data
    baseline: Optional[PerformanceBaseline] = Field(
        default=None,
        description="Design and baseline performance parameters"
    )

    # Fuel data (for acid dew point)
    fuel_type: Optional[str] = Field(
        default=None,
        description="Fuel type (coal, natural_gas, oil, biomass)"
    )
    fuel_sulfur_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Fuel sulfur content (%). For coal: 0.5-4%, oil: 0.5-3%."
    )

    # Ambient conditions
    ambient_temp_f: float = Field(
        default=70.0,
        ge=-40.0,
        le=130.0,
        description="Ambient temperature (F)"
    )
    barometric_pressure_in_hg: float = Field(
        default=29.92,
        ge=28.0,
        le=32.0,
        description="Barometric pressure (in. Hg)"
    )

    @model_validator(mode='after')
    def validate_temperatures_and_type(self):
        """Validate temperature relationships and type-specific data."""
        # Validate gas temperature drop
        if self.gas_outlet_temp_f >= self.gas_inlet_temp_f:
            raise ValueError(
                f"Gas outlet temp ({self.gas_outlet_temp_f}F) must be less than "
                f"inlet temp ({self.gas_inlet_temp_f}F)"
            )

        # Validate air temperature rise
        if self.air_outlet_temp_f <= self.air_inlet_temp_f:
            raise ValueError(
                f"Air outlet temp ({self.air_outlet_temp_f}F) must be greater than "
                f"inlet temp ({self.air_inlet_temp_f}F)"
            )

        # Validate regenerative preheater has rotor data
        if self.preheater_type == PreheaterType.REGENERATIVE:
            if self.regenerative_data is None:
                # Allow but with warning - regenerative data is highly recommended
                pass

        return self


# =============================================================================
# OUTPUT SCHEMAS - HEAT TRANSFER ANALYSIS
# =============================================================================


class HeatTransferAnalysis(BasePreheaterSchema):
    """
    Heat transfer performance analysis results.

    Calculates key heat transfer metrics per ASME PTC 4.3 methodology
    including effectiveness, NTU, LMTD, and X-ratio.

    Attributes:
        effectiveness: Actual heat transfer effectiveness (0-1).
            epsilon = Q_actual / Q_max = (T_gas_in - T_gas_out) / (T_gas_in - T_air_in)
        design_effectiveness: Design effectiveness for comparison.
        effectiveness_ratio: Actual/Design effectiveness ratio.
        ntu: Number of Transfer Units.
            NTU = UA / C_min. Higher NTU = more heat transfer.
        design_ntu: Design NTU for comparison.
        heat_duty_mmbtu_hr: Actual heat transfer rate (MMBTU/hr).
            Q = m_gas * Cp_gas * (T_gas_in - T_gas_out)
        design_heat_duty_mmbtu_hr: Design heat duty.
        lmtd_f: Log Mean Temperature Difference (F).
            LMTD = (dT1 - dT2) / ln(dT1/dT2)
        x_ratio: Capacity ratio (gas-side / air-side heat capacity rate).
            X = (m_gas * Cp_gas) / (m_air * Cp_air)
    """

    # Effectiveness analysis
    effectiveness: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Actual heat transfer effectiveness. >1.0 indicates measurement error."
    )
    design_effectiveness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Design effectiveness"
    )
    effectiveness_ratio: Optional[float] = Field(
        default=None,
        description="Actual/Design effectiveness ratio. <0.9 indicates degradation."
    )
    effectiveness_deviation_pct: Optional[float] = Field(
        default=None,
        description="Effectiveness deviation from design (%)"
    )

    # NTU analysis
    ntu: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Number of Transfer Units (NTU)"
    )
    design_ntu: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Design NTU"
    )
    ntu_ratio: Optional[float] = Field(
        default=None,
        description="Actual/Design NTU ratio"
    )

    # Heat duty
    heat_duty_mmbtu_hr: float = Field(
        ...,
        ge=0.0,
        description="Actual heat duty (MMBTU/hr)"
    )
    design_heat_duty_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Design heat duty (MMBTU/hr)"
    )
    heat_duty_deficit_mmbtu_hr: Optional[float] = Field(
        default=None,
        description="Heat duty shortfall (MMBTU/hr)"
    )
    heat_duty_deficit_pct: Optional[float] = Field(
        default=None,
        description="Heat duty shortfall (%)"
    )

    # LMTD analysis
    lmtd_f: float = Field(
        ...,
        ge=0.0,
        description="Log Mean Temperature Difference (F)"
    )
    approach_temp_hot_end_f: float = Field(
        ...,
        description="Hot-end approach: T_gas_in - T_air_out (F)"
    )
    approach_temp_cold_end_f: float = Field(
        ...,
        description="Cold-end approach: T_gas_out - T_air_in (F)"
    )

    # X-ratio analysis (critical for regenerative preheaters)
    x_ratio: float = Field(
        ...,
        ge=0.3,
        le=3.0,
        description="Capacity ratio X = (m*Cp)_gas / (m*Cp)_air"
    )
    design_x_ratio: Optional[float] = Field(
        default=None,
        ge=0.3,
        le=3.0,
        description="Design X-ratio"
    )
    x_ratio_deviation_pct: Optional[float] = Field(
        default=None,
        description="X-ratio deviation from design (%)"
    )

    # UA value analysis
    current_ua_btu_hr_f: float = Field(
        ...,
        ge=0.0,
        description="Current overall heat transfer coefficient times area (BTU/hr-F)"
    )
    design_ua_btu_hr_f: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Design UA value (BTU/hr-F)"
    )
    clean_ua_btu_hr_f: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Clean condition UA value (BTU/hr-F)"
    )
    ua_degradation_pct: Optional[float] = Field(
        default=None,
        description="UA degradation from clean condition (%)"
    )

    # Temperature drops/rises
    gas_temp_drop_f: float = Field(
        ...,
        ge=0.0,
        description="Gas temperature drop (F)"
    )
    air_temp_rise_f: float = Field(
        ...,
        ge=0.0,
        description="Air temperature rise (F)"
    )
    gas_temp_drop_design_f: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Design gas temperature drop (F)"
    )
    air_temp_rise_design_f: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Design air temperature rise (F)"
    )

    # Performance status
    performance_status: str = Field(
        default="normal",
        description="Performance status (normal, degraded, critical)"
    )
    performance_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Overall performance score (0-100)"
    )

    # Calculation provenance
    calculation_method: str = Field(
        default="ASME_PTC_4.3",
        description="Heat transfer calculation method"
    )
    formula_reference: str = Field(
        default="ASME PTC 4.3-2017 Air Heater Test Code",
        description="Standard reference"
    )


# =============================================================================
# OUTPUT SCHEMAS - LEAKAGE ANALYSIS
# =============================================================================


class LeakageAnalysis(BasePreheaterSchema):
    """
    Air leakage analysis for regenerative air preheaters.

    Leakage in regenerative preheaters occurs through seal gaps as the
    rotor rotates between gas and air sides. Higher air pressure causes
    air-to-gas leakage, which:
    - Reduces effective combustion air (efficiency loss)
    - Increases apparent O2 in flue gas (affects emissions monitoring)
    - Increases fan power consumption

    Attributes:
        air_to_gas_leakage_pct: Air leaking into gas side (%).
            Calculated from O2 rise across preheater.
        gas_to_air_leakage_pct: Gas leaking into air side (%).
            Less common, occurs at low differential pressure.
        total_leakage_pct: Combined leakage effect (%).
        leakage_method: Method used to calculate leakage.
        o2_rise_method: Detailed O2 rise calculation results.
    """

    # Leakage percentages
    air_to_gas_leakage_pct: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Air-to-gas leakage (%). From O2 rise method."
    )
    gas_to_air_leakage_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Gas-to-air leakage (%). Typically small."
    )
    total_leakage_pct: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Total effective leakage (%)"
    )

    # Design comparison
    design_leakage_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=25.0,
        description="Design leakage (%). Typical: 5-10%."
    )
    excess_leakage_pct: Optional[float] = Field(
        default=None,
        description="Leakage above design (%)"
    )

    # O2 rise method results
    o2_inlet_gas_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=21.0,
        description="O2 at gas inlet (% dry)"
    )
    o2_outlet_gas_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=21.0,
        description="O2 at gas outlet (% dry)"
    )
    o2_rise_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="O2 rise across preheater (%)"
    )

    # Seal analysis
    primary_leakage_source: Optional[str] = Field(
        default=None,
        description="Primary source of leakage (radial_seals, axial_seals, sector_plates)"
    )
    radial_seal_leakage_contribution_pct: Optional[float] = Field(
        default=None,
        description="Leakage contribution from radial seals (%)"
    )
    axial_seal_leakage_contribution_pct: Optional[float] = Field(
        default=None,
        description="Leakage contribution from axial seals (%)"
    )
    circumferential_seal_leakage_contribution_pct: Optional[float] = Field(
        default=None,
        description="Leakage contribution from circumferential seals (%)"
    )

    # Impact assessment
    efficiency_loss_due_to_leakage_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Boiler efficiency loss due to excess leakage (%)"
    )
    fan_power_increase_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=50.0,
        description="Increased fan power due to leakage (%)"
    )
    annual_energy_loss_mmbtu: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annual energy loss due to excess leakage (MMBTU)"
    )
    annual_cost_impact_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annual cost impact of excess leakage ($)"
    )

    # Leakage severity
    leakage_severity: FoulingSeverity = Field(
        default=FoulingSeverity.NONE,
        description="Leakage severity classification"
    )

    # Corrective actions
    seal_adjustment_recommended: bool = Field(
        default=False,
        description="Seal adjustment recommended"
    )
    seal_replacement_recommended: bool = Field(
        default=False,
        description="Seal replacement recommended"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Primary recommended corrective action"
    )

    # Calculation provenance
    calculation_method: str = Field(
        default="O2_RISE_METHOD",
        description="Leakage calculation method"
    )
    formula_reference: str = Field(
        default="ASME PTC 4.3 Section 6.2 Air Leakage Test",
        description="Standard reference"
    )


# =============================================================================
# OUTPUT SCHEMAS - COLD END PROTECTION
# =============================================================================


class ColdEndProtection(BasePreheaterSchema):
    """
    Cold-end corrosion protection analysis.

    Cold-end corrosion occurs when metal temperatures fall below the
    acid dew point of the flue gas, causing sulfuric acid condensation.
    This analysis calculates acid dew points using multiple methods
    and assesses corrosion risk.

    Acid Dew Point Correlations:
    - Verhoff-Banchero (1974): Most widely used for coal combustion
    - Okkes (1987): Good for high-sulfur fuels
    - ZareNezhad (2009): Modern correlation with improved accuracy
    - Mueller: European standard correlation
    - Pierce: Simplified correlation

    Attributes:
        acid_dew_point_f: Calculated sulfuric acid dew point (F).
        water_dew_point_f: Water dew point (F).
        min_metal_temp_f: Minimum measured cold-end metal temperature.
        margin_above_adp_f: Temperature margin above acid dew point.
        corrosion_risk: Assessed corrosion risk level.
    """

    # Acid dew point calculations
    acid_dew_point_f: float = Field(
        ...,
        ge=150.0,
        le=400.0,
        description="Calculated sulfuric acid dew point (F). Typical: 250-320F."
    )
    acid_dew_point_verhoff_banchero_f: Optional[float] = Field(
        default=None,
        ge=150.0,
        le=400.0,
        description="Acid dew point by Verhoff-Banchero method (F)"
    )
    acid_dew_point_okkes_f: Optional[float] = Field(
        default=None,
        ge=150.0,
        le=400.0,
        description="Acid dew point by Okkes method (F)"
    )
    acid_dew_point_zarenezhad_f: Optional[float] = Field(
        default=None,
        ge=150.0,
        le=400.0,
        description="Acid dew point by ZareNezhad method (F)"
    )
    primary_method_used: AcidDewPointMethod = Field(
        default=AcidDewPointMethod.VERHOFF_BANCHERO,
        description="Primary acid dew point calculation method"
    )

    # Water dew point
    water_dew_point_f: float = Field(
        ...,
        ge=80.0,
        le=200.0,
        description="Water dew point (F). Typically 115-145F."
    )

    # Metal temperatures
    min_metal_temp_f: float = Field(
        ...,
        ge=50.0,
        le=500.0,
        description="Minimum cold-end metal temperature (F)"
    )
    avg_metal_temp_f: float = Field(
        ...,
        ge=50.0,
        le=500.0,
        description="Average cold-end metal temperature (F)"
    )
    min_recommended_metal_temp_f: float = Field(
        ...,
        ge=100.0,
        le=400.0,
        description="Minimum recommended metal temperature (F)"
    )

    # Temperature margins
    margin_above_adp_f: float = Field(
        ...,
        description="Temperature margin above acid dew point (F). Target: 20-50F."
    )
    margin_above_wdp_f: float = Field(
        ...,
        description="Temperature margin above water dew point (F)"
    )
    margin_is_adequate: bool = Field(
        ...,
        description="Temperature margin meets recommended minimum"
    )
    recommended_margin_f: float = Field(
        default=30.0,
        ge=10.0,
        le=100.0,
        description="Recommended margin above acid dew point (F)"
    )

    # Corrosion risk assessment
    corrosion_risk: CorrosionRiskLevel = Field(
        ...,
        description="Cold-end corrosion risk level"
    )
    below_acid_dew_point: bool = Field(
        ...,
        description="Metal temperature below acid dew point"
    )
    hours_below_adp: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Hours metal temperature has been below acid dew point"
    )

    # SO3 analysis
    so3_concentration_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Estimated SO3 concentration (ppm)"
    )
    so3_calculation_method: Optional[str] = Field(
        default=None,
        description="SO3 estimation method (measured, 1%_of_SO2, correlation)"
    )

    # Moisture analysis
    flue_gas_moisture_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=30.0,
        description="Flue gas moisture content (%)"
    )

    # Corrosion impact
    estimated_corrosion_rate_mpy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Estimated corrosion rate (mils per year)"
    )
    remaining_element_life_years: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated remaining heating element life (years)"
    )

    # Corrective actions
    action_required: bool = Field(
        default=False,
        description="Corrective action required"
    )
    increase_air_inlet_temp_recommended: bool = Field(
        default=False,
        description="Recommend increasing air inlet temperature"
    )
    recommended_air_inlet_temp_f: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=300.0,
        description="Recommended air inlet temperature (F)"
    )
    steam_coil_preheat_recommended: bool = Field(
        default=False,
        description="Recommend steam coil air preheater activation"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Primary recommended corrective action"
    )

    # Calculation provenance
    calculation_method: str = Field(
        default="VERHOFF_BANCHERO",
        description="Primary acid dew point calculation method"
    )
    formula_reference: str = Field(
        default="Verhoff & Banchero, Chemical Engineering Progress, 1974",
        description="Formula reference"
    )


# =============================================================================
# OUTPUT SCHEMAS - FOULING ANALYSIS
# =============================================================================


class FoulingAnalysis(BasePreheaterSchema):
    """
    Air preheater fouling analysis results.

    Fouling in air preheaters reduces heat transfer and increases
    pressure drop. Sources include:
    - Fly ash deposition (coal firing)
    - Ammonium bisulfate (ABS) from SCR systems
    - Corrosion products
    - Unburned carbon

    Attributes:
        cleanliness_factor: Ratio of current to clean UA (0-1).
        gas_dp_ratio: Actual/Design gas-side pressure drop ratio.
        air_dp_ratio: Actual/Air-side pressure drop ratio.
        fouling_severity: Overall fouling severity classification.
        cleaning_effectiveness: Improvement from last cleaning.
    """

    # Cleanliness factor (primary fouling indicator)
    cleanliness_factor: float = Field(
        ...,
        ge=0.0,
        le=1.2,
        description="Cleanliness factor = UA_actual / UA_clean. Target: >0.85."
    )
    design_cleanliness_factor: float = Field(
        default=1.0,
        ge=0.8,
        le=1.0,
        description="Design/clean cleanliness factor"
    )
    cleanliness_deviation_pct: float = Field(
        default=0.0,
        description="Deviation from clean condition (%)"
    )

    # Pressure drop analysis - Gas side
    gas_dp_actual_in_wc: float = Field(
        ...,
        ge=0.0,
        description="Actual gas-side pressure drop (in. WC)"
    )
    gas_dp_design_in_wc: float = Field(
        ...,
        ge=0.0,
        description="Design gas-side pressure drop (in. WC)"
    )
    gas_dp_corrected_in_wc: float = Field(
        ...,
        ge=0.0,
        description="Flow-corrected gas-side pressure drop (in. WC)"
    )
    gas_dp_ratio: float = Field(
        ...,
        ge=0.5,
        le=5.0,
        description="Gas DP ratio (corrected/design). >1.5 indicates significant fouling."
    )

    # Pressure drop analysis - Air side
    air_dp_actual_in_wc: float = Field(
        ...,
        ge=0.0,
        description="Actual air-side pressure drop (in. WC)"
    )
    air_dp_design_in_wc: float = Field(
        ...,
        ge=0.0,
        description="Design air-side pressure drop (in. WC)"
    )
    air_dp_corrected_in_wc: float = Field(
        ...,
        ge=0.0,
        description="Flow-corrected air-side pressure drop (in. WC)"
    )
    air_dp_ratio: float = Field(
        ...,
        ge=0.5,
        le=5.0,
        description="Air DP ratio (corrected/design)"
    )

    # Combined fouling assessment
    fouling_severity: FoulingSeverity = Field(
        ...,
        description="Overall fouling severity"
    )
    fouling_type: Optional[str] = Field(
        default=None,
        description="Probable fouling type (fly_ash, ABS, corrosion_product)"
    )
    fouling_trend: str = Field(
        default="stable",
        description="Fouling trend (improving, stable, degrading)"
    )

    # Heat transfer degradation
    ua_degradation_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Heat transfer degradation from fouling (%)"
    )
    efficiency_loss_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Boiler efficiency loss due to fouling (%)"
    )

    # Estimated fouling layer
    estimated_fouling_thickness_in: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Estimated average fouling layer thickness (inches)"
    )
    fouling_resistance_hr_ft2_f_btu: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Calculated fouling resistance (hr-ft2-F/BTU)"
    )

    # Cleaning status
    cleaning_recommended: bool = Field(
        default=False,
        description="Cleaning recommended"
    )
    cleaning_urgency: str = Field(
        default="not_required",
        description="Cleaning urgency (not_required, monitor, recommended, required, urgent)"
    )
    recommended_cleaning_method: Optional[CleaningMethod] = Field(
        default=None,
        description="Recommended cleaning method"
    )
    hours_since_last_cleaning: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Hours since last cleaning"
    )
    estimated_hours_to_cleaning: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated hours until cleaning required"
    )

    # Cleaning effectiveness (if recent cleaning)
    pre_cleaning_cleanliness_factor: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cleanliness factor before last cleaning"
    )
    post_cleaning_cleanliness_factor: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.2,
        description="Cleanliness factor after last cleaning"
    )
    cleaning_effectiveness_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Cleaning effectiveness (% recovery)"
    )

    # Economic impact
    annual_efficiency_loss_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annual cost of efficiency loss due to fouling ($)"
    )
    annual_fan_power_increase_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annual cost of increased fan power ($)"
    )

    # Calculation provenance
    calculation_method: str = Field(
        default="ASME_PTC_4.3",
        description="Fouling calculation method"
    )


# =============================================================================
# OUTPUT SCHEMAS - OPTIMIZATION RESULTS
# =============================================================================


class OptimizationRecommendation(BaseModel):
    """Single optimization recommendation with quantified impact."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique recommendation identifier"
    )
    category: str = Field(
        ...,
        description="Category (heat_transfer, leakage, corrosion, fouling, operation)"
    )
    priority: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Recommendation priority"
    )
    title: str = Field(
        ...,
        description="Brief recommendation title"
    )
    description: str = Field(
        ...,
        description="Detailed recommendation description"
    )

    # Current vs target
    current_value: Optional[float] = Field(
        default=None,
        description="Current parameter value"
    )
    target_value: Optional[float] = Field(
        default=None,
        description="Target parameter value"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Parameter unit"
    )

    # Quantified benefits
    efficiency_improvement_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Expected efficiency improvement (%)"
    )
    energy_savings_mmbtu_yr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annual energy savings (MMBTU/yr)"
    )
    cost_savings_usd_yr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annual cost savings ($/yr)"
    )
    co2_reduction_tons_yr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Annual CO2 reduction (tons/yr)"
    )

    # Implementation
    implementation_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated implementation cost ($)"
    )
    simple_payback_months: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Simple payback period (months)"
    )
    implementation_difficulty: str = Field(
        default="low",
        description="Implementation difficulty (low, medium, high)"
    )
    requires_outage: bool = Field(
        default=False,
        description="Requires unit outage to implement"
    )

    model_config = {"use_enum_values": True}


# Backward compatibility alias
Recommendation = OptimizationRecommendation


class OptimizationResult(BasePreheaterSchema):
    """
    Comprehensive optimization analysis and recommendations.

    Provides optimal setpoints, energy savings potential, and
    prioritized recommendations for air preheater operation.
    """

    # Optimal setpoints
    optimal_air_inlet_temp_f: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=300.0,
        description="Optimal air inlet temperature (F)"
    )
    optimal_gas_outlet_temp_f: Optional[float] = Field(
        default=None,
        ge=200.0,
        le=500.0,
        description="Optimal gas outlet temperature (F)"
    )
    optimal_rotor_speed_rpm: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=5.0,
        description="Optimal rotor speed for regenerative preheaters (RPM)"
    )

    # Current vs optimal performance
    current_effectiveness: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Current heat transfer effectiveness"
    )
    achievable_effectiveness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Achievable effectiveness with optimization"
    )
    effectiveness_improvement_potential_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="Potential effectiveness improvement (%)"
    )

    # Energy savings
    current_heat_recovery_mmbtu_hr: float = Field(
        ...,
        ge=0.0,
        description="Current heat recovery rate (MMBTU/hr)"
    )
    potential_heat_recovery_mmbtu_hr: float = Field(
        ...,
        ge=0.0,
        description="Potential heat recovery with optimization (MMBTU/hr)"
    )
    additional_heat_recovery_mmbtu_hr: float = Field(
        default=0.0,
        ge=0.0,
        description="Additional recoverable heat (MMBTU/hr)"
    )
    annual_energy_savings_mmbtu: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual energy savings potential (MMBTU/yr)"
    )

    # Cost savings
    fuel_cost_per_mmbtu_usd: float = Field(
        default=5.0,
        ge=0.0,
        description="Fuel cost assumption ($/MMBTU)"
    )
    annual_cost_savings_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual cost savings potential ($/yr)"
    )
    operating_hours_per_year: float = Field(
        default=8000.0,
        ge=0.0,
        le=8760.0,
        description="Operating hours assumption (hr/yr)"
    )

    # Emissions reduction
    co2_emission_factor_tons_mmbtu: float = Field(
        default=0.0531,
        ge=0.0,
        description="CO2 emission factor (tons/MMBTU)"
    )
    annual_co2_reduction_tons: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual CO2 reduction potential (tons/yr)"
    )

    # Efficiency impact
    boiler_efficiency_improvement_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Boiler efficiency improvement potential (%)"
    )

    # Prioritized recommendations
    recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list,
        description="Prioritized optimization recommendations"
    )
    total_recommendations: int = Field(
        default=0,
        ge=0,
        description="Total number of recommendations"
    )
    critical_recommendations: int = Field(
        default=0,
        ge=0,
        description="Number of critical recommendations"
    )

    # Overall optimization score
    optimization_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Overall optimization score (0-100). 100 = fully optimized."
    )
    optimization_status: str = Field(
        default="optimal",
        description="Optimization status (optimal, near_optimal, suboptimal, poor)"
    )


# =============================================================================
# OUTPUT SCHEMAS - ALERTS
# =============================================================================


class PreheaterAlert(BaseModel):
    """Air preheater monitoring alert."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Alert identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp"
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity"
    )
    category: str = Field(
        ...,
        description="Alert category (heat_transfer, leakage, corrosion, fouling, mechanical)"
    )
    title: str = Field(
        ...,
        description="Alert title"
    )
    description: str = Field(
        ...,
        description="Alert description"
    )
    value: Optional[float] = Field(
        default=None,
        description="Triggering value"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Alert threshold"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Value unit"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended response action"
    )
    acknowledged: bool = Field(
        default=False,
        description="Alert acknowledged"
    )
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="Acknowledging user"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="Acknowledgment timestamp"
    )

    model_config = {"use_enum_values": True}


# Backward compatibility alias
Alert = PreheaterAlert


# =============================================================================
# COMPREHENSIVE OUTPUT MODEL
# =============================================================================


class AirPreheaterOutput(BasePreheaterSchema):
    """
    Complete output from GL-024 Air Preheater Agent analysis.

    Provides comprehensive analysis results including heat transfer
    performance, leakage assessment, cold-end protection, fouling
    analysis, optimization recommendations, and alerts.

    Example:
        >>> output = agent.analyze(input_data)
        >>> print(f"Effectiveness: {output.heat_transfer.effectiveness:.2%}")
        >>> print(f"Leakage: {output.leakage.air_to_gas_leakage_pct:.1f}%")
        >>> print(f"Corrosion Risk: {output.cold_end.corrosion_risk}")
        >>> for rec in output.optimization.recommendations:
        ...     print(f"- {rec.title}: ${rec.cost_savings_usd_yr:.0f}/yr")
    """

    # Identification
    agent_id: str = Field(
        default="GL-024-AIR-PREHEATER",
        description="Agent identifier"
    )
    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique execution identifier"
    )
    preheater_id: str = Field(
        ...,
        description="Air preheater identifier"
    )
    preheater_type: PreheaterType = Field(
        ...,
        description="Air preheater type"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output timestamp"
    )

    # Operating conditions summary
    boiler_load_pct: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Boiler load at time of analysis (%)"
    )
    operating_status: PreheaterStatus = Field(
        default=PreheaterStatus.NORMAL,
        description="Preheater operating status"
    )

    # Analysis results
    heat_transfer: HeatTransferAnalysis = Field(
        ...,
        description="Heat transfer performance analysis"
    )
    leakage: Optional[LeakageAnalysis] = Field(
        default=None,
        description="Leakage analysis (regenerative preheaters only)"
    )
    cold_end: ColdEndProtection = Field(
        ...,
        description="Cold-end corrosion protection analysis"
    )
    fouling: FoulingAnalysis = Field(
        ...,
        description="Fouling analysis"
    )
    optimization: OptimizationResult = Field(
        ...,
        description="Optimization analysis and recommendations"
    )

    # Key performance indicators
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Alerts
    alerts: List[PreheaterAlert] = Field(
        default_factory=list,
        description="Active alerts"
    )
    alert_count: int = Field(
        default=0,
        ge=0,
        description="Total alert count"
    )
    critical_alert_count: int = Field(
        default=0,
        ge=0,
        description="Critical alert count"
    )

    # Overall status
    overall_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Overall analysis status"
    )
    overall_health_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Overall preheater health score (0-100)"
    )
    primary_issue: Optional[str] = Field(
        default=None,
        description="Primary issue requiring attention"
    )

    # Processing metadata
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Analysis processing time (ms)"
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
    formula_references: List[str] = Field(
        default_factory=list,
        description="Engineering standards and formulas used"
    )

    # Provenance
    input_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of input data"
    )
    calculation_chain: List[str] = Field(
        default_factory=list,
        description="Calculation step hashes for audit trail"
    )

    # Intelligence outputs (for STANDARD level)
    explanation: Optional[str] = Field(
        default=None,
        description="LLM-generated natural language explanation"
    )
    intelligent_recommendations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="LLM-enhanced intelligent recommendations"
    )

    # Validation
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation error messages"
    )
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="Validation warning messages"
    )

    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        },
    }


# =============================================================================
# CONFIGURATION MODEL
# =============================================================================


class AirPreheaterConfig(BaseModel):
    """
    Configuration parameters for GL-024 Air Preheater Agent.

    Defines operational thresholds, calculation methods, and
    default values for air preheater analysis.
    """

    # Agent identification
    agent_id: str = Field(
        default="GL-024-AIR-PREHEATER",
        description="Agent identifier"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )

    # Acid dew point settings
    default_acid_dew_point_method: AcidDewPointMethod = Field(
        default=AcidDewPointMethod.VERHOFF_BANCHERO,
        description="Default acid dew point calculation method"
    )
    minimum_margin_above_adp_f: float = Field(
        default=25.0,
        ge=10.0,
        le=100.0,
        description="Minimum recommended margin above acid dew point (F)"
    )
    default_so3_fraction_of_sox: float = Field(
        default=0.02,
        ge=0.005,
        le=0.10,
        description="Default SO3 as fraction of total SOx (0.01-0.05 typical)"
    )

    # Leakage thresholds
    leakage_warning_pct: float = Field(
        default=12.0,
        ge=5.0,
        le=30.0,
        description="Leakage warning threshold (%)"
    )
    leakage_alarm_pct: float = Field(
        default=18.0,
        ge=10.0,
        le=40.0,
        description="Leakage alarm threshold (%)"
    )
    leakage_critical_pct: float = Field(
        default=25.0,
        ge=15.0,
        le=50.0,
        description="Leakage critical threshold (%)"
    )

    # Fouling thresholds
    cleanliness_factor_warning: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Cleanliness factor warning threshold"
    )
    cleanliness_factor_alarm: float = Field(
        default=0.75,
        ge=0.4,
        le=0.95,
        description="Cleanliness factor alarm threshold"
    )
    dp_ratio_warning: float = Field(
        default=1.3,
        ge=1.1,
        le=2.0,
        description="Pressure drop ratio warning threshold"
    )
    dp_ratio_alarm: float = Field(
        default=1.6,
        ge=1.3,
        le=3.0,
        description="Pressure drop ratio alarm threshold"
    )

    # Effectiveness thresholds
    effectiveness_ratio_warning: float = Field(
        default=0.90,
        ge=0.7,
        le=1.0,
        description="Effectiveness ratio warning threshold"
    )
    effectiveness_ratio_alarm: float = Field(
        default=0.80,
        ge=0.5,
        le=0.95,
        description="Effectiveness ratio alarm threshold"
    )

    # Economic parameters
    default_fuel_cost_per_mmbtu_usd: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="Default fuel cost ($/MMBTU)"
    )
    default_electricity_cost_per_kwh_usd: float = Field(
        default=0.08,
        ge=0.0,
        le=0.50,
        description="Default electricity cost ($/kWh)"
    )
    default_operating_hours_per_year: float = Field(
        default=8000.0,
        ge=0.0,
        le=8760.0,
        description="Default operating hours (hr/yr)"
    )
    default_co2_emission_factor_tons_mmbtu: float = Field(
        default=0.0531,
        ge=0.0,
        le=0.15,
        description="Default CO2 emission factor (tons/MMBTU)"
    )

    # Default physical properties
    default_gas_specific_heat_btu_lb_f: float = Field(
        default=0.25,
        ge=0.20,
        le=0.35,
        description="Default flue gas specific heat (BTU/lb-F)"
    )
    default_air_specific_heat_btu_lb_f: float = Field(
        default=0.24,
        ge=0.22,
        le=0.28,
        description="Default air specific heat (BTU/lb-F)"
    )

    # Provenance tracking
    enable_provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )
    enable_calculation_chain: bool = Field(
        default=True,
        description="Enable calculation chain logging"
    )

    # Calculation precision
    decimal_precision: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Decimal precision for calculations"
    )

    model_config = {"use_enum_values": True}


# =============================================================================
# EXPORT ALL SCHEMAS
# =============================================================================


__all__ = [
    # Enums
    "PreheaterType",
    "AirPreheaterType",  # Backward compatibility alias
    "PreheaterStatus",
    "RotorStatus",
    "FoulingSeverity",
    "CorrosionRiskLevel",
    "LeakageDirection",
    "CleaningMethod",
    "AcidDewPointMethod",
    "AlertSeverity",
    "ValidationStatus",
    # Base schemas
    "BasePreheaterSchema",
    # Input schemas
    "GasComposition",
    "GasSideInput",
    "AirSideInput",
    "RegenerativeOperatingData",
    "SootBlowerStatus",
    "PerformanceBaseline",
    "AirPreheaterInput",
    # Output schemas
    "HeatTransferAnalysis",
    "LeakageAnalysis",
    "ColdEndProtection",
    "FoulingAnalysis",
    "OptimizationRecommendation",
    "Recommendation",  # Backward compatibility alias
    "OptimizationResult",
    "PreheaterAlert",
    "Alert",  # Backward compatibility alias
    "AirPreheaterOutput",
    # Configuration
    "AirPreheaterConfig",
]
