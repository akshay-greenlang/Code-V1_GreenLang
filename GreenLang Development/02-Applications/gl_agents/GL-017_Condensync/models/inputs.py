# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Input Models

Comprehensive input models for condenser diagnostic and optimization.
These models capture all sensor inputs, configuration data, climate conditions,
and operating constraints required for deterministic condenser analysis.

Standards Reference:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- EPRI Condenser Performance Guidelines

Zero-Hallucination Guarantee:
All input models use strict validation with Pydantic.
Physical limits are enforced at the model boundary.
No inferred or interpolated values without explicit source.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .domain import (
    AlertLevel,
    CleaningMethod,
    FailureMode,
    OperatingMode,
    TubeEndConnection,
    TubeMaterial,
    TubeSupport,
    VacuumControlMode,
    WaterSource,
)


# ============================================================================
# SENSOR INPUT MODELS
# ============================================================================

class CondenserDiagnosticInput(BaseModel):
    """
    Primary input model for condenser diagnostic analysis.

    Captures all real-time sensor readings required for performance
    calculation, fouling assessment, and optimization recommendations.

    All temperature values in Celsius, pressures in kPa, flows in kg/s.

    Example:
        >>> input_data = CondenserDiagnosticInput(
        ...     condenser_id="COND-001",
        ...     timestamp=datetime.now(timezone.utc),
        ...     cw_inlet_temp_c=Decimal("25.0"),
        ...     cw_outlet_temp_c=Decimal("35.0"),
        ...     cw_flow_rate_kg_s=Decimal("15000.0"),
        ...     condenser_pressure_kpa_abs=Decimal("5.0"),
        ...     steam_flow_rate_kg_s=Decimal("150.0"),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    # Identification
    condenser_id: str = Field(
        ...,
        description="Unique condenser identifier",
        min_length=1,
        max_length=50,
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp (UTC)",
    )
    unit_id: Optional[str] = Field(
        None,
        description="Power unit identifier",
    )

    # Cooling Water Temperatures (Celsius)
    cw_inlet_temp_c: Decimal = Field(
        ...,
        description="Cooling water inlet temperature (Celsius)",
        ge=Decimal("-5.0"),
        le=Decimal("50.0"),
    )
    cw_outlet_temp_c: Decimal = Field(
        ...,
        description="Cooling water outlet temperature (Celsius)",
        ge=Decimal("0.0"),
        le=Decimal("60.0"),
    )
    cw_inlet_temp_pass_a_c: Optional[Decimal] = Field(
        None,
        description="CW inlet temperature, Pass A (multi-pass condenser)",
    )
    cw_outlet_temp_pass_a_c: Optional[Decimal] = Field(
        None,
        description="CW outlet temperature, Pass A (multi-pass condenser)",
    )
    cw_inlet_temp_pass_b_c: Optional[Decimal] = Field(
        None,
        description="CW inlet temperature, Pass B (multi-pass condenser)",
    )
    cw_outlet_temp_pass_b_c: Optional[Decimal] = Field(
        None,
        description="CW outlet temperature, Pass B (multi-pass condenser)",
    )

    # Condenser Vacuum / Pressure
    condenser_pressure_kpa_abs: Decimal = Field(
        ...,
        description="Condenser absolute pressure (kPa)",
        ge=Decimal("1.0"),
        le=Decimal("25.0"),
    )
    condenser_pressure_secondary_kpa: Optional[Decimal] = Field(
        None,
        description="Secondary pressure sensor reading (kPa abs)",
    )
    hotwell_pressure_kpa_abs: Optional[Decimal] = Field(
        None,
        description="Hotwell pressure (kPa abs)",
    )

    # Cooling Water Flow
    cw_flow_rate_kg_s: Decimal = Field(
        ...,
        description="Total cooling water mass flow rate (kg/s)",
        ge=Decimal("0.0"),
        le=Decimal("100000.0"),
    )
    cw_flow_rate_pass_a_kg_s: Optional[Decimal] = Field(
        None,
        description="CW flow rate, Pass A (kg/s)",
    )
    cw_flow_rate_pass_b_kg_s: Optional[Decimal] = Field(
        None,
        description="CW flow rate, Pass B (kg/s)",
    )
    cw_pump_discharge_pressure_kpa: Optional[Decimal] = Field(
        None,
        description="CW pump discharge pressure (kPa)",
    )
    cw_velocity_m_s: Optional[Decimal] = Field(
        None,
        description="CW tube-side velocity (m/s)",
        ge=Decimal("0.5"),
        le=Decimal("5.0"),
    )

    # Steam Side
    steam_flow_rate_kg_s: Decimal = Field(
        ...,
        description="Exhaust steam flow rate to condenser (kg/s)",
        ge=Decimal("0.0"),
        le=Decimal("2000.0"),
    )
    steam_enthalpy_kj_kg: Optional[Decimal] = Field(
        None,
        description="Exhaust steam enthalpy (kJ/kg)",
        ge=Decimal("2000.0"),
        le=Decimal("3000.0"),
    )
    condensate_temp_c: Optional[Decimal] = Field(
        None,
        description="Condensate temperature (Celsius)",
    )
    condensate_flow_rate_kg_s: Optional[Decimal] = Field(
        None,
        description="Condensate flow rate (kg/s)",
    )
    hotwell_level_percent: Optional[Decimal] = Field(
        None,
        description="Hotwell level (percent)",
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
    )

    # Air Removal System
    air_inleakage_scfm: Optional[Decimal] = Field(
        None,
        description="Air in-leakage rate (SCFM)",
        ge=Decimal("0.0"),
    )
    air_ejector_steam_flow_kg_s: Optional[Decimal] = Field(
        None,
        description="Air ejector motive steam flow (kg/s)",
    )
    dissolved_oxygen_ppb: Optional[Decimal] = Field(
        None,
        description="Condensate dissolved oxygen (ppb)",
        ge=Decimal("0.0"),
    )
    subcooling_c: Optional[Decimal] = Field(
        None,
        description="Condensate subcooling (Celsius)",
    )

    # Tube Cleaning System Status
    online_cleaning_active: bool = Field(
        False,
        description="Online tube cleaning system active",
    )
    last_cleaning_timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp of last cleaning cycle",
    )
    tubes_plugged_count: int = Field(
        0,
        description="Number of plugged tubes",
        ge=0,
    )

    # Operating Context
    operating_mode: OperatingMode = Field(
        OperatingMode.NORMAL,
        description="Current operating mode",
    )
    unit_load_mw: Optional[Decimal] = Field(
        None,
        description="Unit gross load (MW)",
        ge=Decimal("0.0"),
    )
    turbine_exhaust_pressure_kpa: Optional[Decimal] = Field(
        None,
        description="LP turbine exhaust pressure (kPa abs)",
    )

    # Data Quality Flags
    sensor_quality_flags: Dict[str, str] = Field(
        default_factory=dict,
        description="Sensor quality flags (sensor_name: quality_status)",
    )
    data_source: str = Field(
        "OPC-UA",
        description="Data source system (OPC-UA, PI, etc.)",
    )

    @field_validator("cw_outlet_temp_c")
    @classmethod
    def validate_cw_outlet_temp(cls, v: Decimal, info) -> Decimal:
        """Validate CW outlet temp is greater than inlet."""
        inlet_temp = info.data.get("cw_inlet_temp_c")
        if inlet_temp is not None and v < inlet_temp:
            raise ValueError(
                f"CW outlet temp ({v}) must be >= inlet temp ({inlet_temp})"
            )
        return v

    @model_validator(mode="after")
    def validate_temperatures(self) -> "CondenserDiagnosticInput":
        """Cross-validate temperature readings."""
        # CW rise should be reasonable (0.5 to 20 C typically)
        cw_rise = self.cw_outlet_temp_c - self.cw_inlet_temp_c
        if cw_rise > Decimal("25.0"):
            raise ValueError(
                f"CW temperature rise ({cw_rise} C) exceeds physical limit"
            )
        return self

    @property
    def cw_temperature_rise_c(self) -> Decimal:
        """Calculate CW temperature rise."""
        return self.cw_outlet_temp_c - self.cw_inlet_temp_c

    @property
    def heat_duty_kw(self) -> Decimal:
        """Calculate approximate heat duty (kW)."""
        cp = Decimal("4.186")  # kJ/kg-K for water
        return self.cw_flow_rate_kg_s * cp * self.cw_temperature_rise_c

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class CondenserConfiguration(BaseModel):
    """
    Static condenser geometry and design data.

    Captures physical configuration that does not change during operation.
    Used for performance calculations and HEI curve generation.

    Example:
        >>> config = CondenserConfiguration(
        ...     condenser_id="COND-001",
        ...     manufacturer="GE",
        ...     design_heat_duty_kw=Decimal("500000.0"),
        ...     tube_material=TubeMaterial.TITANIUM_GRADE_2,
        ...     total_tube_count=18000,
        ...     tube_od_mm=Decimal("25.4"),
        ...     tube_length_m=Decimal("12.0"),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    # Identification
    condenser_id: str = Field(
        ...,
        description="Unique condenser identifier",
    )
    manufacturer: str = Field(
        "",
        description="Condenser manufacturer",
    )
    model: str = Field(
        "",
        description="Condenser model number",
    )
    year_installed: Optional[int] = Field(
        None,
        description="Year of installation",
        ge=1950,
        le=2030,
    )

    # Design Performance
    design_heat_duty_kw: Decimal = Field(
        ...,
        description="Design heat duty (kW)",
        ge=Decimal("1000.0"),
    )
    design_pressure_kpa_abs: Decimal = Field(
        Decimal("5.0"),
        description="Design condenser pressure (kPa abs)",
        ge=Decimal("1.0"),
        le=Decimal("25.0"),
    )
    design_cw_inlet_temp_c: Decimal = Field(
        Decimal("21.1"),
        description="Design CW inlet temperature (Celsius)",
    )
    design_cw_flow_rate_kg_s: Decimal = Field(
        ...,
        description="Design CW flow rate (kg/s)",
        ge=Decimal("100.0"),
    )
    design_cw_velocity_m_s: Decimal = Field(
        Decimal("2.1"),
        description="Design CW tube velocity (m/s)",
        ge=Decimal("0.5"),
        le=Decimal("3.5"),
    )
    design_steam_flow_kg_s: Decimal = Field(
        ...,
        description="Design steam flow rate (kg/s)",
        ge=Decimal("1.0"),
    )
    design_cleanliness_factor: Decimal = Field(
        Decimal("0.85"),
        description="Design cleanliness factor",
        ge=Decimal("0.5"),
        le=Decimal("1.0"),
    )

    # Tube Geometry
    tube_material: TubeMaterial = Field(
        TubeMaterial.TITANIUM_GRADE_2,
        description="Tube material type",
    )
    total_tube_count: int = Field(
        ...,
        description="Total number of tubes",
        ge=100,
    )
    active_tube_count: Optional[int] = Field(
        None,
        description="Number of active (unplugged) tubes",
    )
    tube_od_mm: Decimal = Field(
        Decimal("25.4"),
        description="Tube outer diameter (mm)",
        ge=Decimal("10.0"),
        le=Decimal("50.0"),
    )
    tube_wall_thickness_mm: Decimal = Field(
        Decimal("0.889"),
        description="Tube wall thickness (mm)",
        ge=Decimal("0.3"),
        le=Decimal("3.0"),
    )
    tube_length_m: Decimal = Field(
        ...,
        description="Effective tube length (m)",
        ge=Decimal("3.0"),
        le=Decimal("25.0"),
    )
    tube_pitch_mm: Decimal = Field(
        Decimal("31.75"),
        description="Tube pitch (center-to-center) (mm)",
    )
    tube_pattern: str = Field(
        "triangular",
        description="Tube layout pattern (triangular, square, rotated)",
    )
    tube_support: TubeSupport = Field(
        TubeSupport.CARBON_STEEL,
        description="Tube support plate material",
    )
    tube_end_connection: TubeEndConnection = Field(
        TubeEndConnection.ROLLED,
        description="Tube-to-tubesheet connection type",
    )

    # Shell Configuration
    number_of_shells: int = Field(
        1,
        description="Number of condenser shells",
        ge=1,
        le=4,
    )
    number_of_passes: int = Field(
        2,
        description="Number of CW passes",
        ge=1,
        le=4,
    )
    shell_id_m: Optional[Decimal] = Field(
        None,
        description="Shell inside diameter (m)",
    )
    shell_length_m: Optional[Decimal] = Field(
        None,
        description="Shell length (m)",
    )

    # Waterbox Configuration
    waterbox_material: str = Field(
        "carbon_steel",
        description="Waterbox material",
    )
    waterbox_coating: Optional[str] = Field(
        None,
        description="Waterbox coating type",
    )
    inlet_waterbox_volume_m3: Optional[Decimal] = Field(
        None,
        description="Inlet waterbox volume (m3)",
    )
    outlet_waterbox_volume_m3: Optional[Decimal] = Field(
        None,
        description="Outlet waterbox volume (m3)",
    )

    # Air Removal System
    air_removal_type: str = Field(
        "steam_jet_ejector",
        description="Air removal system type",
    )
    air_removal_capacity_scfm: Decimal = Field(
        Decimal("50.0"),
        description="Air removal capacity (SCFM)",
        ge=Decimal("1.0"),
    )

    # Tube Cleaning System
    online_cleaning_type: Optional[CleaningMethod] = Field(
        None,
        description="Online tube cleaning system type",
    )
    online_cleaning_installed: bool = Field(
        False,
        description="Online cleaning system installed",
    )

    # Cooling Water Source
    water_source: WaterSource = Field(
        WaterSource.COOLING_TOWER_MECHANICAL,
        description="Cooling water source type",
    )
    water_chloride_ppm_design: Optional[Decimal] = Field(
        None,
        description="Design CW chloride concentration (ppm)",
    )

    @property
    def tube_id_mm(self) -> Decimal:
        """Calculate tube inside diameter."""
        return self.tube_od_mm - (Decimal("2") * self.tube_wall_thickness_mm)

    @property
    def total_tube_surface_area_m2(self) -> Decimal:
        """Calculate total tube surface area (based on OD)."""
        import math
        pi = Decimal(str(math.pi))
        od_m = self.tube_od_mm / Decimal("1000")
        return pi * od_m * self.tube_length_m * Decimal(self.total_tube_count)

    @property
    def active_tubes(self) -> int:
        """Get number of active tubes."""
        return self.active_tube_count or self.total_tube_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")


# ============================================================================
# CLIMATE INPUT MODEL
# ============================================================================

class ClimateInput(BaseModel):
    """
    Ambient climate and environmental conditions.

    Captures conditions that affect condenser and cooling system performance,
    particularly for cooling tower or air-cooled condenser applications.

    Example:
        >>> climate = ClimateInput(
        ...     timestamp=datetime.now(timezone.utc),
        ...     dry_bulb_temp_c=Decimal("30.0"),
        ...     wet_bulb_temp_c=Decimal("22.0"),
        ...     relative_humidity_percent=Decimal("55.0"),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # Timestamp
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp (UTC)",
    )
    location_id: Optional[str] = Field(
        None,
        description="Weather station or location identifier",
    )

    # Temperature
    dry_bulb_temp_c: Decimal = Field(
        ...,
        description="Dry-bulb ambient temperature (Celsius)",
        ge=Decimal("-50.0"),
        le=Decimal("60.0"),
    )
    wet_bulb_temp_c: Decimal = Field(
        ...,
        description="Wet-bulb temperature (Celsius)",
        ge=Decimal("-50.0"),
        le=Decimal("50.0"),
    )
    dew_point_temp_c: Optional[Decimal] = Field(
        None,
        description="Dew point temperature (Celsius)",
    )

    # Humidity
    relative_humidity_percent: Decimal = Field(
        ...,
        description="Relative humidity (percent)",
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
    )
    specific_humidity_g_kg: Optional[Decimal] = Field(
        None,
        description="Specific humidity (g/kg dry air)",
    )

    # Atmospheric
    barometric_pressure_kpa: Decimal = Field(
        Decimal("101.325"),
        description="Barometric pressure (kPa)",
        ge=Decimal("80.0"),
        le=Decimal("110.0"),
    )
    altitude_m: Optional[Decimal] = Field(
        None,
        description="Site altitude (m above sea level)",
    )

    # Wind (for cooling towers)
    wind_speed_m_s: Optional[Decimal] = Field(
        None,
        description="Wind speed (m/s)",
        ge=Decimal("0.0"),
    )
    wind_direction_deg: Optional[Decimal] = Field(
        None,
        description="Wind direction (degrees from north)",
        ge=Decimal("0.0"),
        le=Decimal("360.0"),
    )

    # Solar
    solar_irradiance_w_m2: Optional[Decimal] = Field(
        None,
        description="Solar irradiance (W/m2)",
        ge=Decimal("0.0"),
    )

    # Water temperatures (if available)
    river_water_temp_c: Optional[Decimal] = Field(
        None,
        description="River/lake intake water temperature (Celsius)",
    )
    ocean_water_temp_c: Optional[Decimal] = Field(
        None,
        description="Ocean intake water temperature (Celsius)",
    )
    cooling_tower_basin_temp_c: Optional[Decimal] = Field(
        None,
        description="Cooling tower basin temperature (Celsius)",
    )

    @field_validator("wet_bulb_temp_c")
    @classmethod
    def validate_wet_bulb(cls, v: Decimal, info) -> Decimal:
        """Validate wet-bulb <= dry-bulb."""
        dry_bulb = info.data.get("dry_bulb_temp_c")
        if dry_bulb is not None and v > dry_bulb:
            raise ValueError(
                f"Wet-bulb temp ({v}) cannot exceed dry-bulb temp ({dry_bulb})"
            )
        return v

    @property
    def approach_to_wet_bulb_c(self) -> Optional[Decimal]:
        """Calculate CW approach to wet-bulb (for cooling towers)."""
        if self.cooling_tower_basin_temp_c is not None:
            return self.cooling_tower_basin_temp_c - self.wet_bulb_temp_c
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")


# ============================================================================
# OPERATING CONSTRAINTS MODEL
# ============================================================================

class OperatingConstraints(BaseModel):
    """
    Safety limits and equipment operating constraints.

    Defines operational boundaries for condenser optimization.
    These constraints must not be violated by any optimization recommendation.

    Example:
        >>> constraints = OperatingConstraints(
        ...     min_vacuum_kpa_abs=Decimal("3.0"),
        ...     max_vacuum_kpa_abs=Decimal("10.0"),
        ...     max_cw_outlet_temp_c=Decimal("45.0"),
        ...     max_tube_velocity_m_s=Decimal("3.0"),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # Vacuum Limits
    min_vacuum_kpa_abs: Decimal = Field(
        Decimal("2.5"),
        description="Minimum allowable condenser pressure (kPa abs)",
        ge=Decimal("1.0"),
        le=Decimal("15.0"),
    )
    max_vacuum_kpa_abs: Decimal = Field(
        Decimal("15.0"),
        description="Maximum allowable condenser pressure (kPa abs)",
        ge=Decimal("3.0"),
        le=Decimal("25.0"),
    )
    vacuum_alarm_low_kpa: Optional[Decimal] = Field(
        None,
        description="Low vacuum alarm setpoint (kPa abs)",
    )
    vacuum_trip_low_kpa: Optional[Decimal] = Field(
        None,
        description="Low vacuum trip setpoint (kPa abs)",
    )
    vacuum_alarm_high_kpa: Optional[Decimal] = Field(
        None,
        description="High vacuum (low pressure) alarm (kPa abs)",
    )

    # Temperature Limits
    max_cw_outlet_temp_c: Decimal = Field(
        Decimal("45.0"),
        description="Maximum CW outlet temperature (Celsius)",
        ge=Decimal("30.0"),
        le=Decimal("60.0"),
    )
    max_condensate_temp_c: Optional[Decimal] = Field(
        None,
        description="Maximum condensate temperature (Celsius)",
    )
    max_cw_temperature_rise_c: Decimal = Field(
        Decimal("15.0"),
        description="Maximum CW temperature rise (Celsius)",
        ge=Decimal("5.0"),
        le=Decimal("25.0"),
    )

    # Flow Limits
    min_cw_flow_rate_kg_s: Decimal = Field(
        Decimal("0.0"),
        description="Minimum CW flow rate (kg/s)",
        ge=Decimal("0.0"),
    )
    max_cw_flow_rate_kg_s: Decimal = Field(
        Decimal("100000.0"),
        description="Maximum CW flow rate (kg/s)",
    )
    min_tube_velocity_m_s: Decimal = Field(
        Decimal("0.9"),
        description="Minimum tube velocity for fouling control (m/s)",
        ge=Decimal("0.3"),
    )
    max_tube_velocity_m_s: Decimal = Field(
        Decimal("3.0"),
        description="Maximum tube velocity for erosion limit (m/s)",
        le=Decimal("5.0"),
    )

    # Air In-leakage Limits
    max_air_inleakage_scfm: Decimal = Field(
        Decimal("10.0"),
        description="Maximum acceptable air in-leakage (SCFM)",
        ge=Decimal("1.0"),
    )
    max_dissolved_oxygen_ppb: Decimal = Field(
        Decimal("10.0"),
        description="Maximum dissolved oxygen in condensate (ppb)",
        ge=Decimal("1.0"),
    )

    # Hotwell Limits
    min_hotwell_level_percent: Decimal = Field(
        Decimal("30.0"),
        description="Minimum hotwell level (percent)",
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
    )
    max_hotwell_level_percent: Decimal = Field(
        Decimal("80.0"),
        description="Maximum hotwell level (percent)",
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
    )

    # Operational Constraints
    max_cw_pump_starts_per_hour: int = Field(
        2,
        description="Maximum CW pump starts per hour",
        ge=1,
    )
    min_time_between_cleanings_hours: Decimal = Field(
        Decimal("4.0"),
        description="Minimum time between online cleaning cycles (hours)",
        ge=Decimal("0.0"),
    )

    # Environmental Constraints
    max_thermal_discharge_delta_c: Optional[Decimal] = Field(
        None,
        description="Maximum thermal discharge temperature rise (Celsius)",
    )
    environmental_permit_limit_c: Optional[Decimal] = Field(
        None,
        description="Environmental permit discharge limit (Celsius)",
    )

    # Tube Plugging Constraints
    max_tube_plugging_percent: Decimal = Field(
        Decimal("10.0"),
        description="Maximum allowable tube plugging (percent)",
        ge=Decimal("0.0"),
        le=Decimal("30.0"),
    )

    @model_validator(mode="after")
    def validate_vacuum_limits(self) -> "OperatingConstraints":
        """Ensure vacuum limits are consistent."""
        if self.min_vacuum_kpa_abs >= self.max_vacuum_kpa_abs:
            raise ValueError(
                "min_vacuum_kpa_abs must be less than max_vacuum_kpa_abs"
            )
        return self

    def is_within_limits(
        self,
        vacuum_kpa: Decimal,
        cw_outlet_c: Decimal,
        cw_rise_c: Decimal,
    ) -> bool:
        """Check if operating conditions are within constraints."""
        return (
            self.min_vacuum_kpa_abs <= vacuum_kpa <= self.max_vacuum_kpa_abs
            and cw_outlet_c <= self.max_cw_outlet_temp_c
            and cw_rise_c <= self.max_cw_temperature_rise_c
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")


# ============================================================================
# HISTORICAL DATA INPUT
# ============================================================================

class HistoricalDataInput(BaseModel):
    """
    Historical performance data for trend analysis.

    Used for fouling trend prediction and performance degradation analysis.

    Example:
        >>> history = HistoricalDataInput(
        ...     condenser_id="COND-001",
        ...     start_time=datetime(2025, 1, 1),
        ...     end_time=datetime(2025, 12, 31),
        ...     cleanliness_factor_readings=[...],
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    condenser_id: str = Field(
        ...,
        description="Condenser identifier",
    )
    start_time: datetime = Field(
        ...,
        description="Start of historical period",
    )
    end_time: datetime = Field(
        ...,
        description="End of historical period",
    )

    # Time-series data (simplified as lists)
    cleanliness_factor_readings: List[Tuple[datetime, Decimal]] = Field(
        default_factory=list,
        description="Historical CF readings [(timestamp, value), ...]",
    )
    vacuum_readings: List[Tuple[datetime, Decimal]] = Field(
        default_factory=list,
        description="Historical vacuum readings [(timestamp, kPa), ...]",
    )
    cw_inlet_temp_readings: List[Tuple[datetime, Decimal]] = Field(
        default_factory=list,
        description="Historical CW inlet temps [(timestamp, C), ...]",
    )

    # Event history
    cleaning_events: List[Tuple[datetime, str]] = Field(
        default_factory=list,
        description="Cleaning events [(timestamp, method), ...]",
    )
    outage_events: List[Tuple[datetime, datetime, str]] = Field(
        default_factory=list,
        description="Outage events [(start, end, reason), ...]",
    )

    @property
    def duration_days(self) -> float:
        """Calculate duration of historical period in days."""
        return (self.end_time - self.start_time).total_seconds() / 86400


# ============================================================================
# OPTIMIZATION REQUEST INPUT
# ============================================================================

class OptimizationRequest(BaseModel):
    """
    Request model for vacuum optimization analysis.

    Specifies optimization objectives, constraints, and time horizon.

    Example:
        >>> request = OptimizationRequest(
        ...     condenser_id="COND-001",
        ...     objective="minimize_heat_rate",
        ...     time_horizon_hours=24,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    condenser_id: str = Field(
        ...,
        description="Condenser identifier",
    )
    objective: str = Field(
        "minimize_heat_rate",
        description="Optimization objective (minimize_heat_rate, maximize_output, minimize_cost)",
    )
    time_horizon_hours: int = Field(
        24,
        description="Optimization time horizon (hours)",
        ge=1,
        le=168,
    )

    # Economic parameters
    power_price_usd_mwh: Optional[Decimal] = Field(
        None,
        description="Electricity price ($/MWh)",
    )
    cw_pump_power_cost_usd_mwh: Optional[Decimal] = Field(
        None,
        description="CW pump power cost ($/MWh)",
    )
    unit_heat_rate_btu_kwh: Optional[Decimal] = Field(
        None,
        description="Unit heat rate (BTU/kWh)",
    )
    fuel_cost_usd_mmbtu: Optional[Decimal] = Field(
        None,
        description="Fuel cost ($/MMBtu)",
    )

    # Load forecast
    load_forecast_mw: Optional[List[Decimal]] = Field(
        None,
        description="Hourly load forecast (MW)",
    )
    ambient_temp_forecast_c: Optional[List[Decimal]] = Field(
        None,
        description="Hourly ambient temperature forecast (Celsius)",
    )

    # Constraints
    allow_cw_pump_cycling: bool = Field(
        True,
        description="Allow CW pump starts/stops for optimization",
    )
    allow_cleaning_recommendation: bool = Field(
        True,
        description="Include cleaning in optimization",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "CondenserDiagnosticInput",
    "CondenserConfiguration",
    "ClimateInput",
    "OperatingConstraints",
    "HistoricalDataInput",
    "OptimizationRequest",
]
