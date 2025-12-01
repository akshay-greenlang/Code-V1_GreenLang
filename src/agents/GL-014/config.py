# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Heat Exchanger Optimizer - Configuration Management

This module provides comprehensive configuration for the Heat Exchanger Optimizer
agent, including all enumerations, Pydantic models for inputs/outputs, validation
rules, and settings management.

Agent ID: GL-014
Codename: EXCHANGER-PRO
Version: 1.0.0
Category: Heat Exchangers
Type: Optimizer

The configuration follows GreenLang's zero-hallucination principle with
deterministic thermal calculations and complete data validation.

Example:
    >>> from config import Settings, HeatExchangerConfig
    >>> from config import TemperatureData, PressureData, FlowData
    >>>
    >>> settings = Settings()
    >>> config = HeatExchangerConfig()
    >>> temp_data = TemperatureData(
    ...     hot_inlet_temp_c=150.0,
    ...     hot_outlet_temp_c=90.0,
    ...     cold_inlet_temp_c=30.0,
    ...     cold_outlet_temp_c=70.0
    ... )

Author: GreenLang AI Agent Factory
License: Apache-2.0
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure module logger
logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class ExchangerType(str, Enum):
    """
    Heat exchanger type classification.

    Defines the physical configuration of the heat exchanger, which determines
    the applicable thermal calculation methods and fouling correlations.
    """

    SHELL_AND_TUBE = "shell_and_tube"
    """Traditional shell-and-tube heat exchanger per TEMA standards."""

    PLATE = "plate"
    """Plate heat exchanger with corrugated plates."""

    AIR_COOLED = "air_cooled"
    """Air-cooled heat exchanger (fin-fan) per API 661."""

    SPIRAL = "spiral"
    """Spiral heat exchanger for high-fouling services."""

    DOUBLE_PIPE = "double_pipe"
    """Double-pipe (hairpin) heat exchanger."""

    FINNED_TUBE = "finned_tube"
    """Finned tube heat exchanger for gas-liquid service."""

    PLATE_FIN = "plate_fin"
    """Plate-fin heat exchanger for cryogenic service."""

    PRINTED_CIRCUIT = "printed_circuit"
    """Printed circuit heat exchanger (PCHE) for compact applications."""


class FoulingMechanism(str, Enum):
    """
    Fouling mechanism classification.

    Identifies the primary fouling mechanism to enable targeted cleaning
    recommendations and accurate fouling progression predictions.
    """

    PARTICULATE = "particulate"
    """Deposition of suspended solids from process streams."""

    CRYSTALLIZATION = "crystallization"
    """Precipitation of dissolved salts (scaling)."""

    BIOLOGICAL = "biological"
    """Growth of biological organisms (biofouling)."""

    CORROSION = "corrosion"
    """Formation of corrosion products on heat transfer surfaces."""

    CHEMICAL_REACTION = "chemical_reaction"
    """Polymerization or coking from chemical reactions."""

    SOLIDIFICATION = "solidification"
    """Freezing or wax deposition from cooling fluids."""

    COMPOSITE = "composite"
    """Multiple fouling mechanisms occurring simultaneously."""


class CleaningMethod(str, Enum):
    """
    Heat exchanger cleaning method classification.

    Defines available cleaning methods with their typical effectiveness,
    cost, and downtime characteristics.
    """

    CHEMICAL = "chemical"
    """Chemical cleaning using appropriate solvents/acids."""

    MECHANICAL = "mechanical"
    """Mechanical cleaning (brushing, scraping, pigging)."""

    HYDROBLAST = "hydroblast"
    """High-pressure water jet cleaning."""

    OFFLINE_CHEMICAL = "offline_chemical"
    """Offline circulation chemical cleaning."""

    ONLINE_CHEMICAL = "online_chemical"
    """Online chemical injection without shutdown."""

    ULTRASONIC = "ultrasonic"
    """Ultrasonic cleaning for precision applications."""

    THERMAL = "thermal"
    """Thermal shock or bake-out cleaning."""


class CleaningSide(str, Enum):
    """
    Heat exchanger side designation for cleaning.

    Specifies which side of the exchanger requires cleaning based on
    fouling analysis.
    """

    SHELL = "shell"
    """Shell side (outside of tubes)."""

    TUBE = "tube"
    """Tube side (inside of tubes)."""

    BOTH = "both"
    """Both shell and tube sides."""


class FluidType(str, Enum):
    """
    Process fluid type classification.

    Categorizes process fluids for property lookup and fouling
    tendency estimation.
    """

    WATER = "water"
    """Process water, cooling water, boiler feedwater."""

    STEAM = "steam"
    """Saturated or superheated steam."""

    THERMAL_OIL = "thermal_oil"
    """Heat transfer oils (Therminol, Dowtherm)."""

    CRUDE_OIL = "crude_oil"
    """Crude oil and crude oil derivatives."""

    NAPHTHA = "naphtha"
    """Light naphtha, heavy naphtha."""

    KEROSENE = "kerosene"
    """Kerosene, jet fuel."""

    DIESEL = "diesel"
    """Diesel fuel, gas oil."""

    GAS_OIL = "gas_oil"
    """Vacuum gas oil, atmospheric gas oil."""

    FUEL_OIL = "fuel_oil"
    """Heavy fuel oil, residual fuel."""

    NATURAL_GAS = "natural_gas"
    """Natural gas, process gas."""

    AIR = "air"
    """Process air, instrument air."""

    NITROGEN = "nitrogen"
    """Nitrogen gas."""

    HYDROGEN = "hydrogen"
    """Hydrogen gas."""

    AMMONIA = "ammonia"
    """Ammonia (liquid or vapor)."""

    REFRIGERANT = "refrigerant"
    """Refrigerants (R-134a, R-410A, etc.)."""

    BRINE = "brine"
    """Brine solutions, salt water."""

    GLYCOL_SOLUTION = "glycol_solution"
    """Ethylene glycol, propylene glycol solutions."""

    CHEMICAL = "chemical"
    """Generic chemical process stream."""


class FluidPhase(str, Enum):
    """
    Fluid phase classification.

    Defines the thermodynamic phase of the process fluid for appropriate
    property calculations.
    """

    LIQUID = "liquid"
    """Single-phase liquid."""

    GAS = "gas"
    """Single-phase gas/vapor."""

    TWO_PHASE = "two_phase"
    """Two-phase (liquid-vapor) mixture."""

    SUPERCRITICAL = "supercritical"
    """Supercritical fluid above critical point."""


class FlowArrangement(str, Enum):
    """
    Heat exchanger flow arrangement classification.

    Defines the relative flow direction of hot and cold streams,
    affecting LMTD correction factor calculations.
    """

    COUNTERFLOW = "counterflow"
    """Streams flow in opposite directions (maximum efficiency)."""

    PARALLEL_FLOW = "parallel_flow"
    """Streams flow in same direction (co-current)."""

    CROSSFLOW = "crossflow"
    """Streams flow perpendicular to each other."""

    MULTIPASS = "multipass"
    """Multiple tube passes with shell-side flow."""


class ShellType(str, Enum):
    """
    TEMA shell type designation.

    Standard TEMA shell classifications for shell-and-tube exchangers.
    """

    E = "E"
    """One-pass shell - most common design."""

    F = "F"
    """Two-pass shell with longitudinal baffle."""

    G = "G"
    """Split flow shell."""

    H = "H"
    """Double split flow shell."""

    J = "J"
    """Divided flow shell."""

    K = "K"
    """Kettle type reboiler."""

    X = "X"
    """Cross flow shell."""


class TubeLayout(str, Enum):
    """
    Tube bundle layout pattern classification.

    Defines the arrangement of tubes in the tube sheet, affecting
    heat transfer and pressure drop calculations.
    """

    TRIANGULAR_30 = "triangular_30"
    """30-degree triangular pitch - best heat transfer."""

    ROTATED_TRIANGULAR_60 = "rotated_triangular_60"
    """60-degree rotated triangular pitch."""

    SQUARE_90 = "square_90"
    """90-degree square pitch - easier cleaning."""

    ROTATED_SQUARE_45 = "rotated_square_45"
    """45-degree rotated square pitch."""


class MaterialType(str, Enum):
    """
    Tube and shell material classification.

    Defines materials of construction affecting thermal conductivity
    and corrosion resistance.
    """

    CARBON_STEEL = "carbon_steel"
    """Carbon steel (A179, A214)."""

    STAINLESS_304 = "stainless_304"
    """304 stainless steel."""

    STAINLESS_316 = "stainless_316"
    """316 stainless steel."""

    STAINLESS_316L = "stainless_316L"
    """316L low-carbon stainless steel."""

    DUPLEX_2205 = "duplex_2205"
    """Duplex stainless steel 2205."""

    MONEL_400 = "monel_400"
    """Monel 400 (Ni-Cu alloy)."""

    INCONEL_625 = "inconel_625"
    """Inconel 625 (Ni-Cr-Mo alloy)."""

    HASTELLOY_C276 = "hastelloy_c276"
    """Hastelloy C-276."""

    TITANIUM_GRADE_2 = "titanium_grade_2"
    """Commercially pure titanium Grade 2."""

    COPPER = "copper"
    """Copper."""

    ADMIRALTY_BRASS = "admiralty_brass"
    """Admiralty brass (Cu-Zn-Sn)."""

    CUPRONICKEL_90_10 = "cupronickel_90_10"
    """90-10 Cupronickel."""

    CUPRONICKEL_70_30 = "cupronickel_70_30"
    """70-30 Cupronickel."""


class PerformanceStatus(str, Enum):
    """
    Heat exchanger performance status classification.

    Categorizes overall performance based on U-value ratio and
    thermal effectiveness.
    """

    OPTIMAL = "optimal"
    """U-ratio > 0.95 - Excellent performance."""

    GOOD = "good"
    """U-ratio 0.85-0.95 - Acceptable performance."""

    DEGRADED = "degraded"
    """U-ratio 0.70-0.85 - Noticeable degradation."""

    POOR = "poor"
    """U-ratio 0.55-0.70 - Significant degradation."""

    CRITICAL = "critical"
    """U-ratio <= 0.55 - Critical degradation."""


class FoulingState(str, Enum):
    """
    Fouling severity classification.

    Categorizes fouling state based on total fouling factor.
    """

    CLEAN = "clean"
    """Rf < 0.0001 m2-K/W - Clean condition."""

    LIGHT = "light"
    """0.0001 <= Rf < 0.0003 m2-K/W - Light fouling."""

    MODERATE = "moderate"
    """0.0003 <= Rf < 0.0005 m2-K/W - Moderate fouling."""

    HEAVY = "heavy"
    """0.0005 <= Rf < 0.001 m2-K/W - Heavy fouling."""

    SEVERE = "severe"
    """Rf >= 0.001 m2-K/W - Severe fouling."""


class MaintenanceUrgency(str, Enum):
    """
    Maintenance urgency classification.

    Defines urgency levels for cleaning recommendations.
    """

    ROUTINE = "routine"
    """Can be scheduled for next turnaround."""

    PLANNED = "planned"
    """Should be scheduled within 60 days."""

    URGENT = "urgent"
    """Should be cleaned within 14 days."""

    CRITICAL = "critical"
    """Immediate cleaning required."""


class FoulingTrend(str, Enum):
    """
    Fouling rate trend classification.

    Describes the trajectory of fouling rate changes.
    """

    STABLE = "stable"
    """Fouling rate change < 5%."""

    INCREASING = "increasing"
    """Fouling rate increasing 5-20%."""

    ACCELERATING = "accelerating"
    """Fouling rate increasing > 20%."""

    DECELERATING = "decelerating"
    """Fouling rate decreasing."""


class RecommendationPriority(str, Enum):
    """
    Recommendation priority classification.

    Defines priority levels for optimization recommendations.
    """

    CRITICAL = "critical"
    """Must be addressed immediately."""

    HIGH = "high"
    """Should be addressed within 7 days."""

    MEDIUM = "medium"
    """Should be addressed within 30 days."""

    LOW = "low"
    """Can be addressed during next opportunity."""


class RecommendationCategory(str, Enum):
    """
    Recommendation category classification.

    Categorizes recommendations by type of action required.
    """

    CLEANING = "cleaning"
    """Cleaning-related recommendation."""

    OPERATIONAL = "operational"
    """Operating parameter adjustment."""

    DESIGN = "design"
    """Design modification recommendation."""

    MONITORING = "monitoring"
    """Enhanced monitoring recommendation."""


# ==============================================================================
# INPUT MODELS
# ==============================================================================


class TemperatureData(BaseModel):
    """
    Temperature measurement data for heat exchanger analysis.

    Contains inlet and outlet temperatures for both hot and cold streams,
    which are essential for heat duty and LMTD calculations.

    Attributes:
        hot_inlet_temp_c: Hot stream inlet temperature in Celsius.
        hot_outlet_temp_c: Hot stream outlet temperature in Celsius.
        cold_inlet_temp_c: Cold stream inlet temperature in Celsius.
        cold_outlet_temp_c: Cold stream outlet temperature in Celsius.
        ambient_temp_c: Ambient temperature for heat loss calculations.
        measurement_timestamp: Timestamp of temperature measurement.

    Example:
        >>> temp_data = TemperatureData(
        ...     hot_inlet_temp_c=150.0,
        ...     hot_outlet_temp_c=90.0,
        ...     cold_inlet_temp_c=30.0,
        ...     cold_outlet_temp_c=70.0
        ... )
        >>> temp_data.hot_side_delta_t
        60.0
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    hot_inlet_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=500.0,
        description="Hot stream inlet temperature (Celsius)",
        examples=[150.0, 200.0, 300.0],
    )

    hot_outlet_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=500.0,
        description="Hot stream outlet temperature (Celsius)",
        examples=[90.0, 120.0, 180.0],
    )

    cold_inlet_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=500.0,
        description="Cold stream inlet temperature (Celsius)",
        examples=[30.0, 40.0, 50.0],
    )

    cold_outlet_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=500.0,
        description="Cold stream outlet temperature (Celsius)",
        examples=[70.0, 100.0, 150.0],
    )

    ambient_temp_c: float = Field(
        default=25.0,
        ge=-40.0,
        le=60.0,
        description="Ambient temperature (Celsius)",
    )

    measurement_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of temperature measurement (ISO8601)",
    )

    @field_validator("hot_outlet_temp_c")
    @classmethod
    def validate_hot_outlet(cls, v: float, info) -> float:
        """Validate that hot outlet is less than hot inlet (cooling)."""
        if "hot_inlet_temp_c" in info.data:
            hot_inlet = info.data["hot_inlet_temp_c"]
            if v >= hot_inlet:
                raise ValueError(
                    f"Hot outlet temp ({v}) must be less than hot inlet temp ({hot_inlet})"
                )
        return v

    @field_validator("cold_outlet_temp_c")
    @classmethod
    def validate_cold_outlet(cls, v: float, info) -> float:
        """Validate that cold outlet is greater than cold inlet (heating)."""
        if "cold_inlet_temp_c" in info.data:
            cold_inlet = info.data["cold_inlet_temp_c"]
            if v <= cold_inlet:
                raise ValueError(
                    f"Cold outlet temp ({v}) must be greater than cold inlet temp ({cold_inlet})"
                )
        return v

    @model_validator(mode="after")
    def validate_temperature_cross(self) -> "TemperatureData":
        """Validate no temperature cross for counterflow operation."""
        # Check for impossible temperature cross
        if self.hot_outlet_temp_c < self.cold_inlet_temp_c:
            # This is valid for counterflow
            pass
        if self.cold_outlet_temp_c > self.hot_inlet_temp_c:
            raise ValueError(
                f"Temperature cross detected: cold outlet ({self.cold_outlet_temp_c}) "
                f"cannot exceed hot inlet ({self.hot_inlet_temp_c})"
            )
        return self

    @computed_field
    @property
    def hot_side_delta_t(self) -> float:
        """Calculate hot side temperature change (K or C)."""
        return self.hot_inlet_temp_c - self.hot_outlet_temp_c

    @computed_field
    @property
    def cold_side_delta_t(self) -> float:
        """Calculate cold side temperature change (K or C)."""
        return self.cold_outlet_temp_c - self.cold_inlet_temp_c

    @computed_field
    @property
    def approach_temperature(self) -> float:
        """Calculate temperature approach (hot outlet - cold inlet)."""
        return self.hot_outlet_temp_c - self.cold_inlet_temp_c


class PressureData(BaseModel):
    """
    Pressure measurement data for heat exchanger analysis.

    Contains inlet and outlet pressures for shell and tube sides,
    used for pressure drop monitoring and fouling detection.

    Attributes:
        shell_inlet_pressure_kpa: Shell side inlet pressure in kPa.
        shell_outlet_pressure_kpa: Shell side outlet pressure in kPa.
        tube_inlet_pressure_kpa: Tube side inlet pressure in kPa.
        tube_outlet_pressure_kpa: Tube side outlet pressure in kPa.
        design_shell_pressure_drop_kpa: Design shell side pressure drop.
        design_tube_pressure_drop_kpa: Design tube side pressure drop.

    Example:
        >>> pressure_data = PressureData(
        ...     shell_inlet_pressure_kpa=500.0,
        ...     shell_outlet_pressure_kpa=480.0,
        ...     tube_inlet_pressure_kpa=600.0,
        ...     tube_outlet_pressure_kpa=570.0
        ... )
        >>> pressure_data.shell_pressure_drop_kpa
        20.0
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    shell_inlet_pressure_kpa: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Shell side inlet pressure (kPa)",
    )

    shell_outlet_pressure_kpa: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Shell side outlet pressure (kPa)",
    )

    tube_inlet_pressure_kpa: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Tube side inlet pressure (kPa)",
    )

    tube_outlet_pressure_kpa: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Tube side outlet pressure (kPa)",
    )

    design_shell_pressure_drop_kpa: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Design shell side pressure drop (kPa)",
    )

    design_tube_pressure_drop_kpa: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Design tube side pressure drop (kPa)",
    )

    measurement_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of pressure measurement (ISO8601)",
    )

    @field_validator("shell_outlet_pressure_kpa")
    @classmethod
    def validate_shell_outlet(cls, v: float, info) -> float:
        """Validate shell outlet pressure is less than inlet."""
        if "shell_inlet_pressure_kpa" in info.data:
            inlet = info.data["shell_inlet_pressure_kpa"]
            if v > inlet:
                raise ValueError(
                    f"Shell outlet pressure ({v}) cannot exceed inlet pressure ({inlet})"
                )
        return v

    @field_validator("tube_outlet_pressure_kpa")
    @classmethod
    def validate_tube_outlet(cls, v: float, info) -> float:
        """Validate tube outlet pressure is less than inlet."""
        if "tube_inlet_pressure_kpa" in info.data:
            inlet = info.data["tube_inlet_pressure_kpa"]
            if v > inlet:
                raise ValueError(
                    f"Tube outlet pressure ({v}) cannot exceed inlet pressure ({inlet})"
                )
        return v

    @computed_field
    @property
    def shell_pressure_drop_kpa(self) -> float:
        """Calculate shell side pressure drop (kPa)."""
        return self.shell_inlet_pressure_kpa - self.shell_outlet_pressure_kpa

    @computed_field
    @property
    def tube_pressure_drop_kpa(self) -> float:
        """Calculate tube side pressure drop (kPa)."""
        return self.tube_inlet_pressure_kpa - self.tube_outlet_pressure_kpa

    @computed_field
    @property
    def shell_dp_ratio(self) -> Optional[float]:
        """Calculate ratio of actual to design shell pressure drop."""
        if self.design_shell_pressure_drop_kpa and self.design_shell_pressure_drop_kpa > 0:
            return self.shell_pressure_drop_kpa / self.design_shell_pressure_drop_kpa
        return None

    @computed_field
    @property
    def tube_dp_ratio(self) -> Optional[float]:
        """Calculate ratio of actual to design tube pressure drop."""
        if self.design_tube_pressure_drop_kpa and self.design_tube_pressure_drop_kpa > 0:
            return self.tube_pressure_drop_kpa / self.design_tube_pressure_drop_kpa
        return None


class FlowData(BaseModel):
    """
    Flow rate data for heat exchanger analysis.

    Contains mass and volumetric flow rates for hot and cold streams,
    essential for heat duty calculations.

    Attributes:
        hot_mass_flow_kg_s: Hot stream mass flow rate in kg/s.
        cold_mass_flow_kg_s: Cold stream mass flow rate in kg/s.
        hot_volumetric_flow_m3_h: Hot stream volumetric flow rate in m3/h.
        cold_volumetric_flow_m3_h: Cold stream volumetric flow rate in m3/h.

    Example:
        >>> flow_data = FlowData(
        ...     hot_mass_flow_kg_s=10.0,
        ...     cold_mass_flow_kg_s=15.0
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    hot_mass_flow_kg_s: float = Field(
        ...,
        gt=0.0,
        le=10000.0,
        description="Hot stream mass flow rate (kg/s)",
    )

    cold_mass_flow_kg_s: float = Field(
        ...,
        gt=0.0,
        le=10000.0,
        description="Cold stream mass flow rate (kg/s)",
    )

    hot_volumetric_flow_m3_h: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Hot stream volumetric flow rate (m3/h)",
    )

    cold_volumetric_flow_m3_h: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Cold stream volumetric flow rate (m3/h)",
    )

    hot_velocity_m_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=30.0,
        description="Hot stream velocity in exchanger (m/s)",
    )

    cold_velocity_m_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=30.0,
        description="Cold stream velocity in exchanger (m/s)",
    )

    measurement_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of flow measurement (ISO8601)",
    )


class FluidPropertySet(BaseModel):
    """
    Fluid property set for a single process stream.

    Contains thermal and physical properties required for heat
    transfer calculations.

    Attributes:
        name: Fluid name or description.
        fluid_type: Type of fluid (water, oil, gas, etc.).
        phase: Fluid phase (liquid, gas, two-phase).
        cp_j_kgk: Specific heat capacity in J/kg-K.
        density_kg_m3: Density in kg/m3.
        viscosity_pa_s: Dynamic viscosity in Pa-s.
        thermal_conductivity_w_mk: Thermal conductivity in W/m-K.
        prandtl_number: Prandtl number (optional, calculated if not provided).
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    name: str = Field(
        ...,
        max_length=100,
        description="Fluid name or description",
    )

    fluid_type: FluidType = Field(
        ...,
        description="Type of process fluid",
    )

    phase: FluidPhase = Field(
        default=FluidPhase.LIQUID,
        description="Fluid phase",
    )

    cp_j_kgk: float = Field(
        ...,
        gt=0.0,
        le=50000.0,
        description="Specific heat capacity (J/kg-K)",
    )

    density_kg_m3: float = Field(
        ...,
        gt=0.0,
        le=15000.0,
        description="Density (kg/m3)",
    )

    viscosity_pa_s: float = Field(
        ...,
        gt=0.0,
        le=1000.0,
        description="Dynamic viscosity (Pa-s)",
    )

    thermal_conductivity_w_mk: float = Field(
        ...,
        gt=0.0,
        le=500.0,
        description="Thermal conductivity (W/m-K)",
    )

    prandtl_number: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Prandtl number",
    )

    @computed_field
    @property
    def calculated_prandtl(self) -> float:
        """Calculate Prandtl number if not provided."""
        if self.prandtl_number is not None:
            return self.prandtl_number
        # Pr = (Cp * mu) / k
        return (self.cp_j_kgk * self.viscosity_pa_s) / self.thermal_conductivity_w_mk

    @computed_field
    @property
    def kinematic_viscosity_m2_s(self) -> float:
        """Calculate kinematic viscosity (m2/s)."""
        return self.viscosity_pa_s / self.density_kg_m3


class FluidProperties(BaseModel):
    """
    Combined fluid properties for hot and cold streams.

    Attributes:
        hot_fluid: Hot stream fluid properties.
        cold_fluid: Cold stream fluid properties.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    hot_fluid: FluidPropertySet = Field(
        ...,
        description="Hot stream fluid properties",
    )

    cold_fluid: FluidPropertySet = Field(
        ...,
        description="Cold stream fluid properties",
    )


class ExchangerParameters(BaseModel):
    """
    Heat exchanger design and physical parameters.

    Contains all design specifications required for thermal performance
    calculations, fouling analysis, and cleaning schedule optimization.

    Attributes:
        exchanger_id: Unique heat exchanger identifier.
        exchanger_name: Descriptive name for the exchanger.
        exchanger_type: Type of heat exchanger.
        flow_arrangement: Flow arrangement (counter, parallel, cross).
        design_heat_duty_kw: Design heat duty in kW.
        design_u_w_m2k: Design overall heat transfer coefficient.
        clean_u_w_m2k: Clean overall heat transfer coefficient.
        heat_transfer_area_m2: Total heat transfer area in m2.

    Example:
        >>> params = ExchangerParameters(
        ...     exchanger_id="HX-101",
        ...     exchanger_type=ExchangerType.SHELL_AND_TUBE,
        ...     flow_arrangement=FlowArrangement.COUNTERFLOW,
        ...     design_heat_duty_kw=5000.0,
        ...     design_u_w_m2k=500.0,
        ...     heat_transfer_area_m2=200.0
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    # Core Identification
    exchanger_id: str = Field(
        ...,
        max_length=50,
        pattern=r"^[A-Za-z0-9\-_]+$",
        description="Unique heat exchanger identifier",
    )

    exchanger_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Heat exchanger name/description",
    )

    # Type and Configuration
    exchanger_type: ExchangerType = Field(
        ...,
        description="Type of heat exchanger",
    )

    flow_arrangement: FlowArrangement = Field(
        default=FlowArrangement.COUNTERFLOW,
        description="Flow arrangement",
    )

    # Design Performance
    design_heat_duty_kw: float = Field(
        ...,
        gt=0.0,
        le=1000000.0,
        description="Design heat duty (kW)",
    )

    design_u_w_m2k: float = Field(
        ...,
        gt=0.0,
        le=10000.0,
        description="Design overall heat transfer coefficient (W/m2-K)",
    )

    clean_u_w_m2k: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=10000.0,
        description="Clean overall heat transfer coefficient (W/m2-K)",
    )

    heat_transfer_area_m2: float = Field(
        ...,
        gt=0.0,
        le=100000.0,
        description="Total heat transfer area (m2)",
    )

    # Shell and Tube Specific Parameters
    shell_type: Optional[ShellType] = Field(
        default=None,
        description="TEMA shell type designation",
    )

    tube_layout: Optional[TubeLayout] = Field(
        default=None,
        description="Tube layout pattern",
    )

    number_of_tubes: Optional[int] = Field(
        default=None,
        ge=1,
        le=50000,
        description="Total number of tubes",
    )

    tube_length_m: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=30.0,
        description="Tube length (m)",
    )

    tube_od_mm: Optional[float] = Field(
        default=None,
        ge=5.0,
        le=100.0,
        description="Tube outer diameter (mm)",
    )

    tube_id_mm: Optional[float] = Field(
        default=None,
        ge=3.0,
        le=95.0,
        description="Tube inner diameter (mm)",
    )

    tube_pitch_mm: Optional[float] = Field(
        default=None,
        ge=6.0,
        le=150.0,
        description="Tube pitch (mm)",
    )

    tube_material: Optional[MaterialType] = Field(
        default=None,
        description="Tube material",
    )

    number_of_passes: Optional[int] = Field(
        default=None,
        ge=1,
        le=16,
        description="Number of tube passes",
    )

    baffle_spacing_mm: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=5000.0,
        description="Baffle spacing (mm)",
    )

    baffle_cut_percent: Optional[float] = Field(
        default=25.0,
        ge=15.0,
        le=45.0,
        description="Baffle cut percentage",
    )

    shell_diameter_mm: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=5000.0,
        description="Shell inner diameter (mm)",
    )

    # Design Fouling Factors
    design_fouling_shell_m2kw: float = Field(
        default=0.0002,
        ge=0.0,
        le=0.01,
        description="Design fouling factor for shell side (m2-K/W)",
    )

    design_fouling_tube_m2kw: float = Field(
        default=0.0002,
        ge=0.0,
        le=0.01,
        description="Design fouling factor for tube side (m2-K/W)",
    )

    @field_validator("tube_id_mm")
    @classmethod
    def validate_tube_id(cls, v: Optional[float], info) -> Optional[float]:
        """Validate tube ID is less than OD."""
        if v is not None and "tube_od_mm" in info.data:
            tube_od = info.data.get("tube_od_mm")
            if tube_od is not None and v >= tube_od:
                raise ValueError(
                    f"Tube ID ({v}) must be less than tube OD ({tube_od})"
                )
        return v

    @computed_field
    @property
    def design_fouling_total_m2kw(self) -> float:
        """Calculate total design fouling factor."""
        return self.design_fouling_shell_m2kw + self.design_fouling_tube_m2kw

    @computed_field
    @property
    def tube_wall_thickness_mm(self) -> Optional[float]:
        """Calculate tube wall thickness (mm)."""
        if self.tube_od_mm is not None and self.tube_id_mm is not None:
            return (self.tube_od_mm - self.tube_id_mm) / 2.0
        return None


class CleaningRecord(BaseModel):
    """
    Historical cleaning record.

    Attributes:
        date: Date of cleaning.
        method: Cleaning method used.
        side: Side that was cleaned.
        cost: Total cleaning cost.
        downtime_hours: Duration of downtime.
        pre_cleaning_fouling: Fouling factor before cleaning.
        post_cleaning_fouling: Fouling factor after cleaning.
        effectiveness: Cleaning effectiveness percentage.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    date: date = Field(..., description="Date of cleaning")
    method: CleaningMethod = Field(..., description="Cleaning method used")
    side: CleaningSide = Field(..., description="Side that was cleaned")
    cost: float = Field(ge=0.0, description="Total cleaning cost")
    downtime_hours: float = Field(ge=0.0, description="Duration of downtime")
    pre_cleaning_fouling: Optional[float] = Field(
        default=None, ge=0.0, description="Fouling factor before cleaning"
    )
    post_cleaning_fouling: Optional[float] = Field(
        default=None, ge=0.0, description="Fouling factor after cleaning"
    )
    effectiveness: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Cleaning effectiveness (%)"
    )


class FoulingDataPoint(BaseModel):
    """
    Historical fouling measurement data point.

    Attributes:
        timestamp: Measurement timestamp.
        shell_fouling_factor: Shell side fouling factor.
        tube_fouling_factor: Tube side fouling factor.
        total_fouling_factor: Total fouling factor.
        u_value: Overall heat transfer coefficient.
        heat_duty: Heat duty at measurement time.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    timestamp: datetime = Field(..., description="Measurement timestamp")
    shell_fouling_factor: Optional[float] = Field(
        default=None, ge=0.0, description="Shell side fouling factor"
    )
    tube_fouling_factor: Optional[float] = Field(
        default=None, ge=0.0, description="Tube side fouling factor"
    )
    total_fouling_factor: Optional[float] = Field(
        default=None, ge=0.0, description="Total fouling factor"
    )
    u_value: Optional[float] = Field(
        default=None, gt=0.0, description="Overall HTC"
    )
    heat_duty: Optional[float] = Field(
        default=None, gt=0.0, description="Heat duty (kW)"
    )


class OperatingHistory(BaseModel):
    """
    Historical operating and cleaning data.

    Contains historical information used for fouling trend analysis
    and cleaning schedule optimization.

    Attributes:
        commissioning_date: Date exchanger was commissioned.
        last_cleaning_date: Date of last cleaning.
        last_cleaning_method: Method used for last cleaning.
        last_cleaning_side: Side cleaned during last cleaning.
        days_since_cleaning: Days since last cleaning.
        cleaning_effectiveness_percent: Effectiveness of last cleaning.
        historical_fouling_data: Array of historical fouling measurements.
        cleaning_history: Array of historical cleaning records.
        average_operating_hours_per_day: Average daily operating hours.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    commissioning_date: Optional[date] = Field(
        default=None,
        description="Date exchanger was commissioned",
    )

    last_cleaning_date: Optional[date] = Field(
        default=None,
        description="Date of last cleaning",
    )

    last_cleaning_method: Optional[CleaningMethod] = Field(
        default=None,
        description="Method used for last cleaning",
    )

    last_cleaning_side: Optional[CleaningSide] = Field(
        default=None,
        description="Side cleaned during last cleaning",
    )

    days_since_cleaning: Optional[int] = Field(
        default=None,
        ge=0,
        description="Days since last cleaning",
    )

    cleaning_effectiveness_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Effectiveness of last cleaning (%)",
    )

    historical_fouling_data: Optional[List[FoulingDataPoint]] = Field(
        default=None,
        description="Array of historical fouling measurements",
    )

    cleaning_history: Optional[List[CleaningRecord]] = Field(
        default=None,
        description="Array of historical cleaning records",
    )

    average_operating_hours_per_day: float = Field(
        default=24.0,
        ge=0.0,
        le=24.0,
        description="Average daily operating hours",
    )


class HeatExchangerInput(BaseModel):
    """
    Complete input data model for HeatExchangerOptimizerAgent.

    Aggregates all input data required for comprehensive heat exchanger
    analysis including temperatures, pressures, flows, exchanger parameters,
    fluid properties, and operating history.

    Attributes:
        temperature_data: Inlet/outlet temperatures for hot and cold streams.
        pressure_data: Pressure drops across shell and tube sides.
        flow_data: Mass and volumetric flow rates.
        exchanger_parameters: Heat exchanger design specifications.
        fluid_properties: Thermal and physical properties of process fluids.
        operating_history: Historical operating and cleaning data (optional).
        analysis_timestamp: Timestamp for this analysis request.

    Example:
        >>> input_data = HeatExchangerInput(
        ...     temperature_data=temp_data,
        ...     pressure_data=pressure_data,
        ...     flow_data=flow_data,
        ...     exchanger_parameters=params,
        ...     fluid_properties=fluid_props
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    temperature_data: TemperatureData = Field(
        ...,
        description="Inlet/outlet temperatures for hot and cold streams",
    )

    pressure_data: PressureData = Field(
        ...,
        description="Pressure drops across shell and tube sides",
    )

    flow_data: FlowData = Field(
        ...,
        description="Mass and volumetric flow rates",
    )

    exchanger_parameters: ExchangerParameters = Field(
        ...,
        description="Heat exchanger design specifications",
    )

    fluid_properties: FluidProperties = Field(
        ...,
        description="Thermal and physical properties of process fluids",
    )

    operating_history: Optional[OperatingHistory] = Field(
        default=None,
        description="Historical operating and cleaning data",
    )

    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp for this analysis request",
    )

    def calculate_input_hash(self) -> str:
        """
        Calculate SHA-256 hash of input data for provenance tracking.

        Returns:
            str: SHA-256 hash of serialized input data.
        """
        # Serialize input data to JSON
        input_json = self.model_dump_json(exclude={"analysis_timestamp"})
        # Calculate and return SHA-256 hash
        return hashlib.sha256(input_json.encode()).hexdigest()


# ==============================================================================
# OUTPUT MODELS
# ==============================================================================


class CleaningMethodComparison(BaseModel):
    """
    Comparison of a single cleaning method option.

    Attributes:
        method: Cleaning method.
        cost: Estimated cost.
        downtime_hours: Estimated downtime.
        effectiveness: Expected effectiveness percentage.
        payback_days: Estimated payback period in days.
        recommended: Whether this method is recommended.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    method: CleaningMethod = Field(..., description="Cleaning method")
    cost: float = Field(ge=0.0, description="Estimated cost (USD)")
    downtime_hours: float = Field(ge=0.0, description="Estimated downtime (hours)")
    effectiveness: float = Field(ge=0.0, le=100.0, description="Expected effectiveness (%)")
    payback_days: int = Field(ge=0, description="Payback period (days)")
    recommended: bool = Field(default=False, description="Whether this method is recommended")


class OptimalCleaningWindow(BaseModel):
    """
    Optimal cleaning window specification.

    Attributes:
        start_date: Start of optimal window.
        end_date: End of optimal window.
        reason: Explanation for window selection.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    start_date: date = Field(..., description="Start of optimal window")
    end_date: date = Field(..., description="End of optimal window")
    reason: str = Field(..., description="Explanation for window selection")


class CleaningSchedule(BaseModel):
    """
    Optimized cleaning schedule output.

    Contains recommended cleaning timing, method, and cost-benefit analysis.

    Attributes:
        exchanger_id: Heat exchanger identifier.
        analysis_timestamp: Timestamp of analysis.
        recommended_cleaning_date: Recommended date for cleaning.
        recommended_cleaning_method: Recommended cleaning method.
        recommended_cleaning_side: Recommended side to clean.
        urgency_level: Cleaning urgency classification.
        days_until_recommended: Days until recommended cleaning.
        cost_benefit_ratio: Ratio of savings to cleaning cost.
        estimated_cleaning_cost: Estimated total cleaning cost.
        estimated_downtime_hours: Estimated downtime for cleaning.
        estimated_downtime_cost: Estimated cost of downtime.
        energy_savings_if_cleaned: Power savings if cleaned now.
        monthly_energy_savings: Monthly energy savings if cleaned.
        payback_period_days: Days to recover cleaning cost.
        optimal_cleaning_window: Optimal window for cleaning.
        cleaning_method_comparison: Comparison of cleaning methods.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")

    recommended_cleaning_date: date = Field(
        ..., description="Recommended date for cleaning"
    )
    recommended_cleaning_method: CleaningMethod = Field(
        ..., description="Recommended cleaning method"
    )
    recommended_cleaning_side: CleaningSide = Field(
        ..., description="Recommended side to clean"
    )
    urgency_level: MaintenanceUrgency = Field(
        ..., description="Cleaning urgency classification"
    )
    days_until_recommended: int = Field(
        ge=0, description="Days until recommended cleaning"
    )

    cost_benefit_ratio: float = Field(
        ge=0.0, description="Ratio of savings to cleaning cost"
    )
    estimated_cleaning_cost: float = Field(
        ge=0.0, description="Estimated total cleaning cost (USD)"
    )
    estimated_downtime_hours: float = Field(
        ge=0.0, description="Estimated downtime for cleaning (hours)"
    )
    estimated_downtime_cost: float = Field(
        ge=0.0, description="Estimated cost of downtime (USD)"
    )
    energy_savings_if_cleaned: float = Field(
        ge=0.0, description="Power savings if cleaned now (kW)"
    )
    monthly_energy_savings: float = Field(
        ge=0.0, description="Monthly energy savings if cleaned (kWh)"
    )
    payback_period_days: int = Field(
        ge=0, description="Days to recover cleaning cost"
    )

    optimal_cleaning_window: Optional[OptimalCleaningWindow] = Field(
        default=None, description="Optimal window for cleaning"
    )
    cleaning_method_comparison: Optional[List[CleaningMethodComparison]] = Field(
        default=None, description="Comparison of cleaning methods"
    )

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class PerformanceMetrics(BaseModel):
    """
    Current and projected performance metrics output.

    Contains comprehensive thermal performance data including heat duty,
    U-value, LMTD, effectiveness, and pressure drop metrics.

    Attributes:
        exchanger_id: Heat exchanger identifier.
        analysis_timestamp: Timestamp of analysis.
        current_heat_duty_kw: Current actual heat duty.
        design_heat_duty_kw: Design heat duty.
        heat_duty_ratio: Ratio of current to design heat duty.
        heat_duty_deficit_kw: Heat duty shortfall.
        current_u_w_m2k: Current overall HTC.
        design_u_w_m2k: Design overall HTC.
        clean_u_w_m2k: Clean condition HTC.
        u_ratio: Ratio of current to design U-value.
        u_degradation_percent: Percentage U-value degradation.
        lmtd_k: Log Mean Temperature Difference.
        lmtd_correction_factor: LMTD correction factor (F).
        corrected_lmtd_k: Corrected LMTD.
        effectiveness: Heat exchanger effectiveness (0-1).
        ntu: Number of Transfer Units.
        capacity_ratio: Heat capacity ratio (Cmin/Cmax).
        approach_temperature_k: Temperature approach.
        shell_pressure_drop_kpa: Shell side pressure drop.
        tube_pressure_drop_kpa: Tube side pressure drop.
        performance_status: Overall performance classification.
        performance_score: Composite performance score (0-100).
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")

    # Heat Duty
    current_heat_duty_kw: float = Field(ge=0.0, description="Current actual heat duty (kW)")
    design_heat_duty_kw: float = Field(gt=0.0, description="Design heat duty (kW)")
    heat_duty_ratio: float = Field(ge=0.0, le=1.5, description="Ratio of current to design")
    heat_duty_deficit_kw: float = Field(ge=0.0, description="Heat duty shortfall (kW)")

    # Overall Heat Transfer Coefficient
    current_u_w_m2k: float = Field(gt=0.0, description="Current overall HTC (W/m2-K)")
    design_u_w_m2k: float = Field(gt=0.0, description="Design overall HTC (W/m2-K)")
    clean_u_w_m2k: Optional[float] = Field(
        default=None, gt=0.0, description="Clean condition HTC (W/m2-K)"
    )
    u_ratio: float = Field(ge=0.0, le=1.5, description="Ratio of current to design U")
    u_degradation_percent: float = Field(
        ge=0.0, le=100.0, description="U-value degradation (%)"
    )

    # LMTD
    lmtd_k: float = Field(gt=0.0, description="Log Mean Temperature Difference (K)")
    lmtd_correction_factor: float = Field(
        ge=0.0, le=1.0, description="LMTD correction factor (F)"
    )
    corrected_lmtd_k: float = Field(gt=0.0, description="Corrected LMTD (K)")

    # Effectiveness-NTU
    effectiveness: float = Field(
        ge=0.0, le=1.0, description="Heat exchanger effectiveness (0-1)"
    )
    ntu: float = Field(ge=0.0, description="Number of Transfer Units")
    capacity_ratio: float = Field(
        ge=0.0, le=1.0, description="Heat capacity ratio (Cmin/Cmax)"
    )

    # Temperature Metrics
    approach_temperature_k: float = Field(ge=0.0, description="Temperature approach (K)")
    hot_side_duty_kw: float = Field(ge=0.0, description="Hot side heat duty (kW)")
    cold_side_duty_kw: float = Field(ge=0.0, description="Cold side heat duty (kW)")
    duty_imbalance_percent: float = Field(
        ge=0.0, le=100.0, description="Duty imbalance (%)"
    )

    # Pressure Drop
    shell_pressure_drop_kpa: float = Field(
        ge=0.0, description="Shell side pressure drop (kPa)"
    )
    tube_pressure_drop_kpa: float = Field(
        ge=0.0, description="Tube side pressure drop (kPa)"
    )
    shell_dp_ratio: Optional[float] = Field(
        default=None, ge=0.0, description="Shell DP to design ratio"
    )
    tube_dp_ratio: Optional[float] = Field(
        default=None, ge=0.0, description="Tube DP to design ratio"
    )

    # Status
    performance_status: PerformanceStatus = Field(
        ..., description="Overall performance classification"
    )
    performance_score: float = Field(
        ge=0.0, le=100.0, description="Composite performance score (0-100)"
    )

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class FoulingMechanismDetail(BaseModel):
    """
    Detail of an identified fouling mechanism.

    Attributes:
        mechanism: Fouling mechanism type.
        probability: Probability of this mechanism (0-1).
        severity: Severity classification.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    mechanism: FoulingMechanism = Field(..., description="Fouling mechanism type")
    probability: float = Field(ge=0.0, le=1.0, description="Probability (0-1)")
    severity: str = Field(..., description="Severity classification")


class FoulingForecastPoint(BaseModel):
    """
    Single point in fouling forecast.

    Attributes:
        days_from_now: Days from analysis date.
        predicted_fouling_factor: Predicted fouling factor.
        predicted_u_value: Predicted U-value.
        predicted_efficiency: Predicted efficiency.
        confidence_interval_lower: Lower bound of confidence interval.
        confidence_interval_upper: Upper bound of confidence interval.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    days_from_now: int = Field(ge=0, description="Days from analysis date")
    predicted_fouling_factor: float = Field(ge=0.0, description="Predicted Rf")
    predicted_u_value: float = Field(gt=0.0, description="Predicted U-value")
    predicted_efficiency: float = Field(
        ge=0.0, le=100.0, description="Predicted efficiency (%)"
    )
    confidence_interval_lower: float = Field(ge=0.0, description="Lower CI bound")
    confidence_interval_upper: float = Field(ge=0.0, description="Upper CI bound")


class FoulingAnalysis(BaseModel):
    """
    Fouling state, predictions, and trends output.

    Contains comprehensive fouling analysis including current state,
    mechanism identification, and progression predictions.

    Attributes:
        exchanger_id: Heat exchanger identifier.
        analysis_timestamp: Timestamp of analysis.
        shell_side_fouling_factor: Shell side fouling factor (m2-K/W).
        tube_side_fouling_factor: Tube side fouling factor (m2-K/W).
        total_fouling_factor: Total fouling factor (m2-K/W).
        fouling_state: Fouling severity classification.
        fouling_percentage: Fouling as percentage of design allowance.
        primary_fouling_mechanism: Primary mechanism identified.
        fouling_mechanisms: All identified mechanisms.
        fouling_rate_m2kw_per_day: Current fouling rate.
        asymptotic_fouling_factor: Predicted asymptotic Rf.
        time_constant_days: Fouling time constant.
        predicted_days_to_threshold: Days until cleaning threshold.
        fouling_trend: Fouling trend direction.
        fouling_forecast: Projected fouling over time.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")

    # Current Fouling State
    shell_side_fouling_factor: float = Field(
        ge=0.0, description="Shell side fouling factor (m2-K/W)"
    )
    tube_side_fouling_factor: float = Field(
        ge=0.0, description="Tube side fouling factor (m2-K/W)"
    )
    total_fouling_factor: float = Field(
        ge=0.0, description="Total fouling factor (m2-K/W)"
    )
    fouling_state: FoulingState = Field(..., description="Fouling severity classification")
    fouling_percentage: float = Field(
        ge=0.0, description="Fouling as % of design allowance"
    )

    # Fouling Mechanisms
    primary_fouling_mechanism: FoulingMechanism = Field(
        ..., description="Primary mechanism identified"
    )
    fouling_mechanisms: Optional[List[FoulingMechanismDetail]] = Field(
        default=None, description="All identified mechanisms"
    )

    # Fouling Prediction
    fouling_rate_m2kw_per_day: float = Field(
        ge=0.0, description="Current fouling rate (m2-K/W per day)"
    )
    asymptotic_fouling_factor: float = Field(
        ge=0.0, description="Predicted asymptotic Rf (m2-K/W)"
    )
    time_constant_days: float = Field(gt=0.0, description="Fouling time constant (days)")
    predicted_days_to_threshold: int = Field(
        ge=0, description="Days until cleaning threshold"
    )
    fouling_trend: FoulingTrend = Field(..., description="Fouling trend direction")

    # Fouling Forecast
    fouling_forecast: Optional[List[FoulingForecastPoint]] = Field(
        default=None, description="Projected fouling over time"
    )

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class Recommendation(BaseModel):
    """
    Single optimization recommendation.

    Attributes:
        priority: Recommendation priority.
        category: Recommendation category.
        action: Recommended action description.
        expected_improvement_percent: Expected improvement.
        estimated_cost: Estimated implementation cost.
        payback_period_days: Estimated payback period.
        implementation_complexity: Implementation complexity.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    priority: RecommendationPriority = Field(..., description="Priority level")
    category: RecommendationCategory = Field(..., description="Category")
    action: str = Field(..., description="Recommended action")
    expected_improvement_percent: float = Field(
        ge=0.0, le=100.0, description="Expected improvement (%)"
    )
    estimated_cost: float = Field(ge=0.0, description="Estimated cost (USD)")
    payback_period_days: int = Field(ge=0, description="Payback period (days)")
    implementation_complexity: str = Field(..., description="Implementation complexity")


class EfficiencyReport(BaseModel):
    """
    Thermal efficiency and optimization recommendations output.

    Contains efficiency metrics, energy losses, and actionable
    recommendations for performance improvement.

    Attributes:
        exchanger_id: Heat exchanger identifier.
        analysis_timestamp: Timestamp of analysis.
        thermal_efficiency_percent: Current thermal efficiency.
        design_efficiency_percent: Design thermal efficiency.
        clean_efficiency_percent: Clean condition efficiency.
        efficiency_loss_percent: Efficiency loss due to fouling.
        energy_loss_kw: Current power loss due to inefficiency.
        daily_energy_loss_kwh: Daily energy loss.
        monthly_energy_loss_mwh: Monthly energy loss.
        annual_energy_loss_mwh: Annual energy loss projection.
        exergy_efficiency_percent: Exergy (2nd law) efficiency.
        exergy_destruction_kw: Rate of exergy destruction.
        recommendations: Optimization recommendations.
        efficiency_improvement_potential: Potential efficiency gain.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")

    # Efficiency Metrics
    thermal_efficiency_percent: float = Field(
        ge=0.0, le=100.0, description="Current thermal efficiency (%)"
    )
    design_efficiency_percent: float = Field(
        ge=0.0, le=100.0, description="Design thermal efficiency (%)"
    )
    clean_efficiency_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Clean efficiency (%)"
    )
    efficiency_loss_percent: float = Field(
        ge=0.0, le=100.0, description="Efficiency loss due to fouling (%)"
    )

    # Energy Losses
    energy_loss_kw: float = Field(ge=0.0, description="Current power loss (kW)")
    daily_energy_loss_kwh: float = Field(ge=0.0, description="Daily energy loss (kWh)")
    monthly_energy_loss_mwh: float = Field(ge=0.0, description="Monthly energy loss (MWh)")
    annual_energy_loss_mwh: float = Field(ge=0.0, description="Annual energy loss (MWh)")

    # Exergy Analysis
    exergy_efficiency_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Exergy efficiency (%)"
    )
    exergy_destruction_kw: Optional[float] = Field(
        default=None, ge=0.0, description="Exergy destruction rate (kW)"
    )

    # Recommendations
    recommendations: Optional[List[Recommendation]] = Field(
        default=None, description="Optimization recommendations"
    )
    efficiency_improvement_potential: float = Field(
        ge=0.0, le=100.0, description="Potential efficiency gain (%)"
    )

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class CostBreakdown(BaseModel):
    """
    Breakdown of fouling-related costs.

    Attributes:
        energy_loss: Annual energy loss cost.
        cleaning_costs: Annual cleaning costs.
        downtime_costs: Annual downtime costs.
        maintenance_overhead: Annual maintenance overhead.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    energy_loss: float = Field(ge=0.0, description="Annual energy loss cost (USD)")
    cleaning_costs: float = Field(ge=0.0, description="Annual cleaning costs (USD)")
    downtime_costs: float = Field(ge=0.0, description="Annual downtime costs (USD)")
    maintenance_overhead: float = Field(ge=0.0, description="Annual overhead (USD)")


class EconomicImpact(BaseModel):
    """
    Cost analysis and ROI projections output.

    Contains comprehensive economic analysis of fouling impact and
    cleaning investment returns.

    Attributes:
        exchanger_id: Heat exchanger identifier.
        analysis_timestamp: Timestamp of analysis.
        currency: Currency for all monetary values.
        hourly_energy_cost_loss: Hourly energy cost loss.
        daily_energy_cost_loss: Daily energy cost loss.
        monthly_energy_cost_loss: Monthly energy cost loss.
        annual_energy_cost_loss: Annual energy cost loss.
        cleaning_cost_estimate: Estimated cleaning cost.
        downtime_cost_estimate: Estimated downtime cost.
        total_cleaning_investment: Total cleaning investment.
        post_cleaning_savings_monthly: Monthly savings after cleaning.
        cleaning_roi_percent: Return on cleaning investment.
        simple_payback_days: Simple payback period.
        optimal_cleaning_frequency_days: Optimal cleaning frequency.
        annual_optimal_cleanings: Optimal cleanings per year.
        annual_cleaning_cost_optimal: Annual cost at optimal frequency.
        total_cost_of_fouling_annual: Total annual cost of fouling.
        total_cost_breakdown: Breakdown of fouling costs.
        potential_annual_savings: Potential annual savings.
        npv_cleaning_now: NPV of cleaning now.
        npv_delayed_cleaning: NPV of delayed cleaning.
        optimal_decision: Optimal economic decision.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")
    currency: str = Field(default="USD", description="Currency for monetary values")

    # Energy Cost Losses
    hourly_energy_cost_loss: float = Field(
        ge=0.0, description="Hourly energy cost loss (USD/hour)"
    )
    daily_energy_cost_loss: float = Field(
        ge=0.0, description="Daily energy cost loss (USD/day)"
    )
    monthly_energy_cost_loss: float = Field(
        ge=0.0, description="Monthly energy cost loss (USD/month)"
    )
    annual_energy_cost_loss: float = Field(
        ge=0.0, description="Annual energy cost loss (USD/year)"
    )

    # Cleaning ROI
    cleaning_cost_estimate: float = Field(
        ge=0.0, description="Estimated cleaning cost (USD)"
    )
    downtime_cost_estimate: float = Field(
        ge=0.0, description="Estimated downtime cost (USD)"
    )
    total_cleaning_investment: float = Field(
        ge=0.0, description="Total cleaning investment (USD)"
    )
    post_cleaning_savings_monthly: float = Field(
        ge=0.0, description="Monthly savings after cleaning (USD/month)"
    )
    cleaning_roi_percent: float = Field(
        description="Return on cleaning investment (%)"
    )
    simple_payback_days: int = Field(ge=0, description="Simple payback period (days)")

    # Optimal Frequency
    optimal_cleaning_frequency_days: int = Field(
        ge=0, description="Optimal cleaning frequency (days)"
    )
    annual_optimal_cleanings: int = Field(
        ge=0, description="Optimal cleanings per year"
    )
    annual_cleaning_cost_optimal: float = Field(
        ge=0.0, description="Annual cleaning cost at optimal frequency (USD/year)"
    )

    # Total Cost Analysis
    total_cost_of_fouling_annual: float = Field(
        ge=0.0, description="Total annual cost of fouling (USD/year)"
    )
    total_cost_breakdown: Optional[CostBreakdown] = Field(
        default=None, description="Breakdown of fouling costs"
    )
    potential_annual_savings: float = Field(
        ge=0.0, description="Potential annual savings (USD/year)"
    )

    # NPV Analysis
    npv_cleaning_now: Optional[float] = Field(
        default=None, description="NPV of cleaning now (USD)"
    )
    npv_delayed_cleaning: Optional[float] = Field(
        default=None, description="NPV of delayed cleaning (USD)"
    )
    optimal_decision: str = Field(..., description="Optimal economic decision")

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class HeatExchangerOutput(BaseModel):
    """
    Complete output data model for HeatExchangerOptimizerAgent.

    Aggregates all output data from comprehensive heat exchanger analysis.

    Attributes:
        cleaning_schedule: Optimized cleaning schedule.
        performance_metrics: Current and projected performance metrics.
        fouling_analysis: Fouling state, predictions, and trends.
        efficiency_report: Thermal efficiency and recommendations.
        economic_impact: Cost analysis and ROI projections.
        processing_time_ms: Processing duration in milliseconds.
        validation_status: PASS or FAIL.
        master_provenance_hash: Combined SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    cleaning_schedule: CleaningSchedule = Field(
        ..., description="Optimized cleaning schedule"
    )
    performance_metrics: PerformanceMetrics = Field(
        ..., description="Current and projected performance metrics"
    )
    fouling_analysis: FoulingAnalysis = Field(
        ..., description="Fouling state, predictions, and trends"
    )
    efficiency_report: EfficiencyReport = Field(
        ..., description="Thermal efficiency and recommendations"
    )
    economic_impact: EconomicImpact = Field(
        ..., description="Cost analysis and ROI projections"
    )

    processing_time_ms: float = Field(
        ge=0.0, description="Processing duration (milliseconds)"
    )
    validation_status: str = Field(
        ..., pattern="^(PASS|FAIL)$", description="Validation status"
    )
    master_provenance_hash: str = Field(
        ..., description="Combined SHA-256 hash for audit trail"
    )


# ==============================================================================
# VALIDATION MODELS
# ==============================================================================


class ValidationError(BaseModel):
    """
    Single validation error.

    Attributes:
        field: Field name with error.
        error: Error description.
        value: Invalid value (if applicable).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    field: str = Field(..., description="Field name with error")
    error: str = Field(..., description="Error description")
    value: Optional[Any] = Field(default=None, description="Invalid value")


class ValidationResult(BaseModel):
    """
    Validation result for input or output data.

    Attributes:
        is_valid: Whether validation passed.
        errors: List of validation errors.
        warnings: List of validation warnings.
        validated_at: Timestamp of validation.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[ValidationError] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of validation warnings"
    )
    validated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of validation"
    )


# ==============================================================================
# CONFIGURATION CLASSES
# ==============================================================================


class FoulingThresholds(BaseModel):
    """
    Fouling factor thresholds for classification.

    Attributes:
        light: Threshold for light fouling (m2-K/W).
        moderate: Threshold for moderate fouling (m2-K/W).
        heavy: Threshold for heavy fouling (m2-K/W).
        severe: Threshold for severe fouling (m2-K/W).
        critical: Threshold for critical fouling (m2-K/W).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    light: float = Field(default=0.0001, ge=0.0, description="Light fouling threshold")
    moderate: float = Field(default=0.0003, ge=0.0, description="Moderate fouling threshold")
    heavy: float = Field(default=0.0005, ge=0.0, description="Heavy fouling threshold")
    severe: float = Field(default=0.001, ge=0.0, description="Severe fouling threshold")
    critical: float = Field(default=0.002, ge=0.0, description="Critical fouling threshold")


class PerformanceThresholds(BaseModel):
    """
    Performance thresholds for classification.

    Attributes:
        optimal: U-ratio threshold for optimal performance.
        good: U-ratio threshold for good performance.
        degraded: U-ratio threshold for degraded performance.
        poor: U-ratio threshold for poor performance.
        critical: U-ratio threshold for critical performance.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    optimal: float = Field(default=0.95, ge=0.0, le=1.0, description="Optimal threshold")
    good: float = Field(default=0.85, ge=0.0, le=1.0, description="Good threshold")
    degraded: float = Field(default=0.70, ge=0.0, le=1.0, description="Degraded threshold")
    poor: float = Field(default=0.55, ge=0.0, le=1.0, description="Poor threshold")
    critical: float = Field(default=0.40, ge=0.0, le=1.0, description="Critical threshold")


class CleaningCosts(BaseModel):
    """
    Default cleaning cost estimates.

    Attributes:
        chemical: Chemical cleaning cost (USD).
        mechanical: Mechanical cleaning cost (USD).
        hydroblast: Hydroblast cleaning cost (USD).
        offline_chemical: Offline chemical cleaning cost (USD).
        online_chemical: Online chemical cleaning cost (USD).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    chemical: float = Field(default=15000.0, ge=0.0, description="Chemical cleaning cost")
    mechanical: float = Field(default=25000.0, ge=0.0, description="Mechanical cleaning cost")
    hydroblast: float = Field(default=35000.0, ge=0.0, description="Hydroblast cleaning cost")
    offline_chemical: float = Field(default=20000.0, ge=0.0, description="Offline chemical cost")
    online_chemical: float = Field(default=8000.0, ge=0.0, description="Online chemical cost")


class CleaningDowntimes(BaseModel):
    """
    Default cleaning downtime estimates.

    Attributes:
        chemical: Chemical cleaning downtime (hours).
        mechanical: Mechanical cleaning downtime (hours).
        hydroblast: Hydroblast cleaning downtime (hours).
        offline_chemical: Offline chemical cleaning downtime (hours).
        online_chemical: Online chemical cleaning downtime (hours).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    chemical: float = Field(default=24.0, ge=0.0, description="Chemical cleaning downtime")
    mechanical: float = Field(default=48.0, ge=0.0, description="Mechanical cleaning downtime")
    hydroblast: float = Field(default=36.0, ge=0.0, description="Hydroblast cleaning downtime")
    offline_chemical: float = Field(default=36.0, ge=0.0, description="Offline chemical downtime")
    online_chemical: float = Field(default=0.0, ge=0.0, description="Online chemical downtime")


class CleaningEffectiveness(BaseModel):
    """
    Default cleaning effectiveness estimates.

    Attributes:
        chemical: Chemical cleaning effectiveness (%).
        mechanical: Mechanical cleaning effectiveness (%).
        hydroblast: Hydroblast cleaning effectiveness (%).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    chemical: float = Field(default=85.0, ge=0.0, le=100.0, description="Chemical effectiveness")
    mechanical: float = Field(default=95.0, ge=0.0, le=100.0, description="Mechanical effectiveness")
    hydroblast: float = Field(default=90.0, ge=0.0, le=100.0, description="Hydroblast effectiveness")


class EconomicParameters(BaseModel):
    """
    Economic parameters for cost calculations.

    Attributes:
        electricity_cost_per_kwh: Electricity cost (USD/kWh).
        steam_cost_per_kg: Steam cost (USD/kg).
        cooling_water_cost_per_m3: Cooling water cost (USD/m3).
        downtime_cost_per_hour: Production downtime cost (USD/hour).
        cleaning_labor_cost_per_hour: Cleaning labor rate (USD/hour).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    electricity_cost_per_kwh: float = Field(
        default=0.10, ge=0.0, description="Electricity cost (USD/kWh)"
    )
    steam_cost_per_kg: float = Field(
        default=0.03, ge=0.0, description="Steam cost (USD/kg)"
    )
    cooling_water_cost_per_m3: float = Field(
        default=0.50, ge=0.0, description="Cooling water cost (USD/m3)"
    )
    downtime_cost_per_hour: float = Field(
        default=5000.0, ge=0.0, description="Downtime cost (USD/hour)"
    )
    cleaning_labor_cost_per_hour: float = Field(
        default=150.0, ge=0.0, description="Cleaning labor rate (USD/hour)"
    )


class FoulingModelParameters(BaseModel):
    """
    Fouling model parameters for prediction.

    Attributes:
        default_time_constant_days: Default fouling time constant (days).
        max_fouling_factor: Maximum asymptotic fouling factor (m2-K/W).
        fouling_rate_acceleration_threshold: Threshold for acceleration detection.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    default_time_constant_days: float = Field(
        default=180.0, gt=0.0, description="Default time constant (days)"
    )
    max_fouling_factor: float = Field(
        default=0.002, gt=0.0, description="Maximum fouling factor (m2-K/W)"
    )
    fouling_rate_acceleration_threshold: float = Field(
        default=0.00001, gt=0.0, description="Acceleration threshold"
    )


class SafetyLimits(BaseModel):
    """
    Safety limits for operation.

    Attributes:
        max_pressure_drop_ratio: Maximum allowable pressure drop ratio.
        max_temperature_approach_k: Maximum temperature approach (K).
        min_flow_ratio: Minimum flow ratio to design.
        max_tube_velocity_m_s: Maximum tube velocity (m/s).
        min_tube_velocity_m_s: Minimum tube velocity (m/s).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    max_pressure_drop_ratio: float = Field(
        default=2.0, gt=1.0, description="Maximum pressure drop ratio"
    )
    max_temperature_approach_k: float = Field(
        default=50.0, gt=0.0, description="Maximum temperature approach (K)"
    )
    min_flow_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum flow ratio"
    )
    max_tube_velocity_m_s: float = Field(
        default=3.0, gt=0.0, description="Maximum tube velocity (m/s)"
    )
    min_tube_velocity_m_s: float = Field(
        default=0.5, gt=0.0, description="Minimum tube velocity (m/s)"
    )


class HeatExchangerConfig(BaseModel):
    """
    Comprehensive heat exchanger optimizer configuration.

    Contains all configurable parameters for the Heat Exchanger Optimizer
    agent including thresholds, economic parameters, and operational limits.

    Attributes:
        fouling_thresholds: Fouling factor thresholds.
        performance_thresholds: Performance classification thresholds.
        cleaning_costs: Default cleaning costs by method.
        cleaning_downtimes: Default cleaning downtimes by method.
        cleaning_effectiveness: Default cleaning effectiveness by method.
        economic_parameters: Economic parameters for cost calculations.
        fouling_model: Fouling model parameters.
        safety_limits: Safety limits for operation.
        cache_ttl_seconds: Cache time-to-live in seconds.
        max_batch_size: Maximum batch processing size.
        max_retries: Maximum retry attempts.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    fouling_thresholds: FoulingThresholds = Field(
        default_factory=FoulingThresholds,
        description="Fouling factor thresholds",
    )

    performance_thresholds: PerformanceThresholds = Field(
        default_factory=PerformanceThresholds,
        description="Performance classification thresholds",
    )

    cleaning_costs: CleaningCosts = Field(
        default_factory=CleaningCosts,
        description="Default cleaning costs by method",
    )

    cleaning_downtimes: CleaningDowntimes = Field(
        default_factory=CleaningDowntimes,
        description="Default cleaning downtimes by method",
    )

    cleaning_effectiveness: CleaningEffectiveness = Field(
        default_factory=CleaningEffectiveness,
        description="Default cleaning effectiveness by method",
    )

    economic_parameters: EconomicParameters = Field(
        default_factory=EconomicParameters,
        description="Economic parameters for cost calculations",
    )

    fouling_model: FoulingModelParameters = Field(
        default_factory=FoulingModelParameters,
        description="Fouling model parameters",
    )

    safety_limits: SafetyLimits = Field(
        default_factory=SafetyLimits,
        description="Safety limits for operation",
    )

    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="Cache time-to-live in seconds",
    )

    max_batch_size: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum batch processing size",
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )


# ==============================================================================
# SETTINGS (Environment-based Configuration)
# ==============================================================================


class Settings(BaseSettings):
    """
    Application configuration settings for GL-014 EXCHANGER-PRO.

    Loads settings from environment variables and .env file, with
    validation and sensible defaults for all parameters.

    Attributes:
        GREENLANG_ENV: Environment (development, staging, production).
        APP_NAME: Application name.
        APP_VERSION: Application version.
        LOG_LEVEL: Logging level.
        DEBUG: Debug mode flag.
        DATABASE_URL: PostgreSQL connection string.
        REDIS_URL: Redis connection string.
        And many more...

    Example:
        >>> settings = Settings()
        >>> print(settings.APP_NAME)
        'GL-014-HeatExchangerOptimizer'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Application Configuration
    # ==========================================================================
    GREENLANG_ENV: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )
    APP_NAME: str = Field(
        default="GL-014-HeatExchangerOptimizer",
        description="Application name",
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode",
    )

    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/greenlang",
        description="PostgreSQL connection string",
    )
    DB_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Database connection pool size",
    )
    DB_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum overflow connections",
    )
    DB_POOL_TIMEOUT: int = Field(
        default=30,
        ge=1,
        description="Pool timeout in seconds",
    )

    # ==========================================================================
    # Cache Configuration
    # ==========================================================================
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )
    REDIS_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Redis connection pool size",
    )
    REDIS_TIMEOUT: int = Field(
        default=5,
        ge=1,
        description="Redis timeout in seconds",
    )
    CACHE_TTL: int = Field(
        default=300,
        ge=0,
        description="Cache TTL in seconds",
    )

    # ==========================================================================
    # AI Model Configuration
    # ==========================================================================
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key (for advanced analytics)",
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key (for advanced analytics)",
    )

    # ==========================================================================
    # Security Configuration
    # ==========================================================================
    JWT_SECRET: str = Field(
        default="change-this-secret-key-in-production",
        description="JWT secret key",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT algorithm",
    )
    JWT_EXPIRATION_HOURS: int = Field(
        default=24,
        ge=1,
        description="JWT expiration in hours",
    )
    API_KEY: Optional[str] = Field(
        default=None,
        description="API key for authentication",
    )

    # ==========================================================================
    # Monitoring & Observability
    # ==========================================================================
    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    PROMETHEUS_PORT: int = Field(
        default=8001,
        ge=1024,
        le=65535,
        description="Prometheus metrics port",
    )
    TRACING_ENABLED: bool = Field(
        default=True,
        description="Enable distributed tracing",
    )
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(
        default="http://localhost:4317",
        description="OTLP exporter endpoint",
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format: json or text",
    )

    # ==========================================================================
    # Process Historian Configuration
    # ==========================================================================
    HISTORIAN_TYPE: str = Field(
        default="opc_ua",
        description="Historian type: osisoft_pi, honeywell_phd, aspentech_ip21, opc_ua",
    )
    HISTORIAN_HOST: str = Field(
        default="localhost",
        description="Historian host",
    )
    HISTORIAN_PORT: int = Field(
        default=4840,
        ge=1,
        le=65535,
        description="Historian port",
    )
    HISTORIAN_PROTOCOL: str = Field(
        default="opc_ua",
        description="Historian protocol",
    )

    # ==========================================================================
    # DCS/SCADA Configuration
    # ==========================================================================
    DCS_HOST: str = Field(
        default="localhost",
        description="DCS/SCADA host",
    )
    DCS_PORT: int = Field(
        default=502,
        ge=1,
        le=65535,
        description="DCS/SCADA port",
    )
    DCS_PROTOCOL: str = Field(
        default="modbus_tcp",
        description="DCS protocol: modbus_tcp, opc_ua",
    )
    SCADA_OPC_UA_ENDPOINT: str = Field(
        default="opc.tcp://localhost:4840",
        description="OPC UA endpoint",
    )

    # ==========================================================================
    # MQTT Configuration
    # ==========================================================================
    MQTT_BROKER_URL: str = Field(
        default="mqtt://localhost:1883",
        description="MQTT broker URL",
    )
    MQTT_TOPIC_PREFIX: str = Field(
        default="greenlang/gl-014",
        description="MQTT topic prefix",
    )

    # ==========================================================================
    # Economic Parameters
    # ==========================================================================
    ELECTRICITY_COST_PER_KWH: float = Field(
        default=0.10,
        ge=0.0,
        description="Electricity cost (USD/kWh)",
    )
    STEAM_COST_PER_KG: float = Field(
        default=0.03,
        ge=0.0,
        description="Steam cost (USD/kg)",
    )
    COOLING_WATER_COST_PER_M3: float = Field(
        default=0.50,
        ge=0.0,
        description="Cooling water cost (USD/m3)",
    )
    DOWNTIME_COST_PER_HOUR: float = Field(
        default=5000.0,
        ge=0.0,
        description="Production downtime cost (USD/hour)",
    )

    # ==========================================================================
    # Fouling Thresholds
    # ==========================================================================
    FOULING_THRESHOLD_LIGHT: float = Field(
        default=0.0001,
        ge=0.0,
        description="Light fouling threshold (m2-K/W)",
    )
    FOULING_THRESHOLD_MODERATE: float = Field(
        default=0.0003,
        ge=0.0,
        description="Moderate fouling threshold (m2-K/W)",
    )
    FOULING_THRESHOLD_HEAVY: float = Field(
        default=0.0005,
        ge=0.0,
        description="Heavy fouling threshold (m2-K/W)",
    )
    FOULING_THRESHOLD_SEVERE: float = Field(
        default=0.001,
        ge=0.0,
        description="Severe fouling threshold (m2-K/W)",
    )

    # ==========================================================================
    # Performance Thresholds
    # ==========================================================================
    EFFICIENCY_THRESHOLD_OPTIMAL: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Optimal efficiency threshold (%)",
    )
    EFFICIENCY_THRESHOLD_GOOD: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Good efficiency threshold (%)",
    )
    EFFICIENCY_THRESHOLD_DEGRADED: float = Field(
        default=75.0,
        ge=0.0,
        le=100.0,
        description="Degraded efficiency threshold (%)",
    )
    EFFICIENCY_THRESHOLD_POOR: float = Field(
        default=65.0,
        ge=0.0,
        le=100.0,
        description="Poor efficiency threshold (%)",
    )

    # ==========================================================================
    # Safety Limits
    # ==========================================================================
    MAX_PRESSURE_DROP_RATIO: float = Field(
        default=2.0,
        gt=1.0,
        description="Maximum allowable pressure drop ratio",
    )
    MAX_TEMPERATURE_APPROACH_K: float = Field(
        default=50.0,
        gt=0.0,
        description="Maximum temperature approach (K)",
    )
    MAX_TUBE_VELOCITY_M_S: float = Field(
        default=3.0,
        gt=0.0,
        description="Maximum tube velocity (m/s)",
    )
    MIN_TUBE_VELOCITY_M_S: float = Field(
        default=0.5,
        gt=0.0,
        description="Minimum tube velocity (m/s)",
    )

    # ==========================================================================
    # Analysis Configuration
    # ==========================================================================
    ANALYSIS_INTERVAL_SECONDS: int = Field(
        default=300,
        ge=10,
        description="Time between analysis cycles (seconds)",
    )
    DATA_COLLECTION_INTERVAL_SECONDS: int = Field(
        default=10,
        ge=1,
        description="Sensor data collection interval (seconds)",
    )
    FORECAST_HORIZON_DAYS: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Fouling forecast horizon (days)",
    )
    FOULING_MODEL_TIME_CONSTANT_DAYS: float = Field(
        default=180.0,
        gt=0.0,
        description="Default fouling time constant (days)",
    )

    # ==========================================================================
    # Batch Processing
    # ==========================================================================
    BATCH_PROCESSING_ENABLED: bool = Field(
        default=True,
        description="Enable batch processing",
    )
    MAX_BATCH_SIZE: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum batch size",
    )
    PARALLEL_WORKERS: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of parallel workers",
    )

    # ==========================================================================
    # Performance & Scaling
    # ==========================================================================
    WORKER_COUNT: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker processes",
    )
    MAX_CONNECTIONS: int = Field(
        default=1000,
        ge=1,
        description="Maximum concurrent connections",
    )
    TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
        description="Request timeout in seconds",
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=100,
        ge=1,
        description="API rate limit per minute",
    )

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    ENABLE_PROFILING: bool = Field(
        default=False,
        description="Enable performance profiling",
    )
    ENABLE_ADVANCED_ANALYTICS: bool = Field(
        default=False,
        description="Enable ML-based analytics",
    )
    ENABLE_CMMS_INTEGRATION: bool = Field(
        default=False,
        description="Enable CMMS integration",
    )
    ENABLE_HISTORIAN_INTEGRATION: bool = Field(
        default=False,
        description="Enable process historian integration",
    )

    # ==========================================================================
    # Deployment Information
    # ==========================================================================
    POD_NAME: Optional[str] = Field(
        default=None,
        description="Kubernetes pod name",
    )
    POD_NAMESPACE: Optional[str] = Field(
        default=None,
        description="Kubernetes namespace",
    )
    NODE_NAME: Optional[str] = Field(
        default=None,
        description="Kubernetes node name",
    )

    @field_validator("GREENLANG_ENV")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level setting."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format setting."""
        valid_formats = ["json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of {valid_formats}")
        return v.lower()

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.GREENLANG_ENV == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.GREENLANG_ENV == "development"

    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.GREENLANG_ENV == "staging"

    def get_config(self) -> HeatExchangerConfig:
        """
        Create HeatExchangerConfig from settings.

        Returns:
            HeatExchangerConfig: Configuration object populated from settings.
        """
        return HeatExchangerConfig(
            fouling_thresholds=FoulingThresholds(
                light=self.FOULING_THRESHOLD_LIGHT,
                moderate=self.FOULING_THRESHOLD_MODERATE,
                heavy=self.FOULING_THRESHOLD_HEAVY,
                severe=self.FOULING_THRESHOLD_SEVERE,
            ),
            economic_parameters=EconomicParameters(
                electricity_cost_per_kwh=self.ELECTRICITY_COST_PER_KWH,
                steam_cost_per_kg=self.STEAM_COST_PER_KG,
                cooling_water_cost_per_m3=self.COOLING_WATER_COST_PER_M3,
                downtime_cost_per_hour=self.DOWNTIME_COST_PER_HOUR,
            ),
            fouling_model=FoulingModelParameters(
                default_time_constant_days=self.FOULING_MODEL_TIME_CONSTANT_DAYS,
            ),
            safety_limits=SafetyLimits(
                max_pressure_drop_ratio=self.MAX_PRESSURE_DROP_RATIO,
                max_temperature_approach_k=self.MAX_TEMPERATURE_APPROACH_K,
                max_tube_velocity_m_s=self.MAX_TUBE_VELOCITY_M_S,
                min_tube_velocity_m_s=self.MIN_TUBE_VELOCITY_M_S,
            ),
            cache_ttl_seconds=self.CACHE_TTL,
            max_batch_size=self.MAX_BATCH_SIZE,
            max_retries=3,
        )


# Global settings instance
settings = Settings()
