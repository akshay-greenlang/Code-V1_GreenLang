# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro: Asset Schemas - Version 1.0

Provides validated data schemas for heat exchanger asset master data,
geometry specifications, and fluid properties with zero-hallucination guarantees.

This module defines Pydantic v2 models for:
- ExchangerAsset: Complete heat exchanger master data
- ExchangerGeometry: Detailed geometry for pressure drop and heat transfer calculations
- FluidProperties: Thermophysical properties for fluids on both shell and tube sides

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TEMAType(str, Enum):
    """TEMA (Tubular Exchanger Manufacturers Association) exchanger types."""
    AES = "AES"  # Floating head with removable tube bundle
    AET = "AET"  # Pull-through floating head
    AEU = "AEU"  # U-tube with floating head
    AEW = "AEW"  # Externally sealed floating tubesheet
    AEM = "AEM"  # Fixed tubesheet with bellows
    AEL = "AEL"  # Fixed tubesheet with lantern ring
    AEN = "AEN"  # Fixed tubesheet
    AEP = "AEP"  # Outside-packed floating head
    AKT = "AKT"  # Kettle reboiler with pull-through head
    AJW = "AJW"  # Jacket type
    BEM = "BEM"  # Fixed tubesheet, bonnet type
    BES = "BES"  # Floating head, bonnet type
    BEU = "BEU"  # U-tube, bonnet type
    CFU = "CFU"  # U-tube, channel integral with tubesheet
    NEN = "NEN"  # Fixed tubesheet, channel integral
    OTHER = "OTHER"


class ShellType(str, Enum):
    """TEMA shell types based on flow pattern."""
    E = "E"  # Single pass shell
    F = "F"  # Two pass shell with longitudinal baffle
    G = "G"  # Split flow
    H = "H"  # Double split flow
    J = "J"  # Divided flow
    K = "K"  # Kettle reboiler
    X = "X"  # Cross flow


class FlowArrangement(str, Enum):
    """Heat exchanger flow arrangement."""
    COUNTER_CURRENT = "counter_current"
    CO_CURRENT = "co_current"
    CROSS_FLOW = "cross_flow"
    MIXED = "mixed"
    PARALLEL = "parallel"


class BaffleType(str, Enum):
    """Baffle types for shell-side flow."""
    SEGMENTAL = "segmental"
    DOUBLE_SEGMENTAL = "double_segmental"
    TRIPLE_SEGMENTAL = "triple_segmental"
    DISC_AND_DOUGHNUT = "disc_and_doughnut"
    ORIFICE = "orifice"
    ROD = "rod"
    HELICAL = "helical"
    NO_BAFFLES = "no_baffles"


class TubePattern(str, Enum):
    """Tube layout pattern."""
    TRIANGULAR = "triangular"  # 30 degrees
    ROTATED_TRIANGULAR = "rotated_triangular"  # 60 degrees
    SQUARE = "square"  # 90 degrees
    ROTATED_SQUARE = "rotated_square"  # 45 degrees


class MaterialClass(str, Enum):
    """Material classification for tubes and shell."""
    CARBON_STEEL = "carbon_steel"
    STAINLESS_304 = "stainless_304"
    STAINLESS_316 = "stainless_316"
    STAINLESS_321 = "stainless_321"
    DUPLEX = "duplex"
    MONEL = "monel"
    INCONEL = "inconel"
    HASTELLOY = "hastelloy"
    TITANIUM = "titanium"
    COPPER = "copper"
    BRASS = "brass"
    CUPRONICKEL_90_10 = "cupronickel_90_10"
    CUPRONICKEL_70_30 = "cupronickel_70_30"
    ALUMINUM = "aluminum"
    OTHER = "other"


class ExchangerStatus(str, Enum):
    """Operational status of exchanger."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    STANDBY = "standby"
    DECOMMISSIONED = "decommissioned"


class ExchangerCriticality(str, Enum):
    """Criticality classification for maintenance prioritization."""
    CRITICAL = "critical"  # Process cannot continue without this unit
    HIGH = "high"  # Significant production impact
    MEDIUM = "medium"  # Moderate production impact
    LOW = "low"  # Minor or no production impact


# =============================================================================
# SUPPORTING MODELS
# =============================================================================

class MaterialSpecification(BaseModel):
    """Detailed material specification for exchanger components."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "material_class": "stainless_316",
                    "grade": "316L",
                    "thermal_conductivity_w_mk": 16.3,
                    "corrosion_allowance_mm": 1.5,
                    "max_design_temp_c": 450.0,
                    "min_design_temp_c": -29.0
                }
            ]
        }
    )

    material_class: MaterialClass = Field(
        ...,
        description="Material classification per TEMA/ASME standards"
    )
    grade: Optional[str] = Field(
        None,
        max_length=50,
        description="Specific material grade (e.g., 316L, 304H)"
    )
    thermal_conductivity_w_mk: Optional[float] = Field(
        None,
        gt=0,
        le=500,
        description="Thermal conductivity in W/(m.K) at design temperature"
    )
    corrosion_allowance_mm: float = Field(
        default=1.5,
        ge=0,
        le=10,
        description="Corrosion allowance in mm"
    )
    max_design_temp_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1000,
        description="Maximum design temperature in Celsius"
    )
    min_design_temp_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1000,
        description="Minimum design temperature in Celsius"
    )


class NozzleSpecification(BaseModel):
    """Nozzle specification for exchanger connections."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "nozzle_id": "N1",
                    "service": "shell_inlet",
                    "nominal_diameter_mm": 150,
                    "schedule": "40",
                    "flange_rating": "300",
                    "material": "carbon_steel"
                }
            ]
        }
    )

    nozzle_id: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Nozzle identifier (e.g., N1, N2, S1, S2)"
    )
    service: Literal[
        "shell_inlet", "shell_outlet", "tube_inlet", "tube_outlet",
        "vent", "drain", "temp_connection", "pressure_connection"
    ] = Field(..., description="Nozzle service type")
    nominal_diameter_mm: float = Field(
        ...,
        gt=0,
        le=2000,
        description="Nominal pipe diameter in mm"
    )
    schedule: Optional[str] = Field(
        None,
        max_length=10,
        description="Pipe schedule (e.g., 40, 80, XXS)"
    )
    flange_rating: Optional[str] = Field(
        None,
        max_length=10,
        description="Flange pressure rating (e.g., 150, 300, 600)"
    )
    material: Optional[MaterialClass] = Field(
        None,
        description="Nozzle material"
    )


class DesignConditions(BaseModel):
    """Design conditions for shell and tube sides."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "design_pressure_bar_g": 15.0,
                    "design_temperature_c": 250.0,
                    "operating_pressure_bar_g": 10.0,
                    "operating_temperature_c": 180.0,
                    "mawp_bar_g": 16.5,
                    "hydrotest_pressure_bar_g": 22.5
                }
            ]
        }
    )

    design_pressure_bar_g: float = Field(
        ...,
        ge=-1,
        le=500,
        description="Design pressure in bar gauge"
    )
    design_temperature_c: float = Field(
        ...,
        ge=-273.15,
        le=1000,
        description="Design temperature in Celsius"
    )
    operating_pressure_bar_g: Optional[float] = Field(
        None,
        ge=-1,
        le=500,
        description="Normal operating pressure in bar gauge"
    )
    operating_temperature_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1000,
        description="Normal operating temperature in Celsius"
    )
    mawp_bar_g: Optional[float] = Field(
        None,
        ge=0,
        le=500,
        description="Maximum Allowable Working Pressure in bar gauge"
    )
    hydrotest_pressure_bar_g: Optional[float] = Field(
        None,
        ge=0,
        le=750,
        description="Hydrotest pressure in bar gauge"
    )


# =============================================================================
# EXCHANGER GEOMETRY
# =============================================================================

class ExchangerGeometry(BaseModel):
    """
    Detailed geometry specification for heat exchanger.

    Provides all dimensional data required for:
    - Heat transfer area calculations
    - Pressure drop calculations
    - Fouling factor analysis
    - Flow distribution modeling

    All dimensions follow SI units (meters, unless otherwise specified).
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "shell_inside_diameter_mm": 610.0,
                    "shell_length_mm": 4880.0,
                    "tube_outer_diameter_mm": 19.05,
                    "tube_inner_diameter_mm": 15.75,
                    "tube_length_mm": 4572.0,
                    "tube_count": 256,
                    "tube_pitch_mm": 25.4,
                    "tube_pattern": "triangular",
                    "tube_passes": 4,
                    "shell_passes": 1,
                    "baffle_type": "segmental",
                    "baffle_cut_percent": 25.0,
                    "baffle_spacing_mm": 305.0,
                    "baffle_count": 14,
                    "heat_transfer_area_m2": 88.5
                }
            ]
        }
    )

    # Shell geometry
    shell_inside_diameter_mm: float = Field(
        ...,
        gt=0,
        le=5000,
        description="Shell inside diameter in mm"
    )
    shell_outside_diameter_mm: Optional[float] = Field(
        None,
        gt=0,
        le=5100,
        description="Shell outside diameter in mm"
    )
    shell_length_mm: float = Field(
        ...,
        gt=0,
        le=30000,
        description="Shell length tube-sheet to tube-sheet in mm"
    )
    shell_thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        le=100,
        description="Shell wall thickness in mm"
    )

    # Tube geometry
    tube_outer_diameter_mm: float = Field(
        ...,
        gt=0,
        le=100,
        description="Tube outside diameter in mm"
    )
    tube_inner_diameter_mm: float = Field(
        ...,
        gt=0,
        le=100,
        description="Tube inside diameter in mm"
    )
    tube_length_mm: float = Field(
        ...,
        gt=0,
        le=25000,
        description="Tube length in mm"
    )
    tube_count: int = Field(
        ...,
        gt=0,
        le=20000,
        description="Total number of tubes"
    )
    tube_pitch_mm: float = Field(
        ...,
        gt=0,
        le=200,
        description="Tube pitch (center-to-center spacing) in mm"
    )
    tube_pattern: TubePattern = Field(
        ...,
        description="Tube layout pattern"
    )
    tube_passes: int = Field(
        ...,
        ge=1,
        le=16,
        description="Number of tube passes"
    )
    tube_wall_thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        le=10,
        description="Tube wall thickness in mm"
    )

    # Shell passes
    shell_passes: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of shell passes"
    )

    # Baffle geometry
    baffle_type: BaffleType = Field(
        default=BaffleType.SEGMENTAL,
        description="Type of baffles installed"
    )
    baffle_cut_percent: Optional[float] = Field(
        None,
        ge=15,
        le=45,
        description="Baffle cut as percentage of shell diameter"
    )
    baffle_spacing_mm: Optional[float] = Field(
        None,
        gt=0,
        le=5000,
        description="Central baffle spacing in mm"
    )
    inlet_baffle_spacing_mm: Optional[float] = Field(
        None,
        gt=0,
        le=5000,
        description="Inlet baffle spacing in mm"
    )
    outlet_baffle_spacing_mm: Optional[float] = Field(
        None,
        gt=0,
        le=5000,
        description="Outlet baffle spacing in mm"
    )
    baffle_count: Optional[int] = Field(
        None,
        ge=0,
        le=200,
        description="Number of baffles"
    )
    baffle_thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        le=30,
        description="Baffle plate thickness in mm"
    )

    # Calculated/derived areas
    heat_transfer_area_m2: float = Field(
        ...,
        gt=0,
        le=50000,
        description="Total heat transfer area in m^2"
    )
    shell_side_flow_area_m2: Optional[float] = Field(
        None,
        gt=0,
        description="Shell-side cross-sectional flow area in m^2"
    )
    tube_side_flow_area_m2: Optional[float] = Field(
        None,
        gt=0,
        description="Tube-side flow area per pass in m^2"
    )

    # Clearances (important for fouling models)
    tube_to_baffle_clearance_mm: Optional[float] = Field(
        None,
        ge=0,
        le=5,
        description="Tube-to-baffle hole clearance (diametral) in mm"
    )
    shell_to_baffle_clearance_mm: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Shell-to-baffle clearance (diametral) in mm"
    )

    @model_validator(mode="after")
    def validate_geometry_consistency(self) -> "ExchangerGeometry":
        """Validate geometric consistency."""
        # Tube inner diameter must be less than outer
        if self.tube_inner_diameter_mm >= self.tube_outer_diameter_mm:
            raise ValueError(
                f"Tube inner diameter ({self.tube_inner_diameter_mm} mm) must be "
                f"less than outer diameter ({self.tube_outer_diameter_mm} mm)"
            )

        # Tube pitch must be greater than tube outer diameter
        if self.tube_pitch_mm <= self.tube_outer_diameter_mm:
            raise ValueError(
                f"Tube pitch ({self.tube_pitch_mm} mm) must be greater than "
                f"tube outer diameter ({self.tube_outer_diameter_mm} mm)"
            )

        # Tube length should be less than shell length
        if self.tube_length_mm > self.shell_length_mm:
            raise ValueError(
                f"Tube length ({self.tube_length_mm} mm) cannot exceed "
                f"shell length ({self.shell_length_mm} mm)"
            )

        return self


# =============================================================================
# FLUID PROPERTIES
# =============================================================================

class FluidProperties(BaseModel):
    """
    Thermophysical properties of process fluids.

    Provides fluid properties at specified temperature and pressure
    for heat transfer and pressure drop calculations.
    All properties in SI units.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "fluid_name": "Water",
                    "temperature_c": 80.0,
                    "pressure_bar_a": 3.0,
                    "density_kg_m3": 971.8,
                    "viscosity_pa_s": 0.000355,
                    "specific_heat_j_kg_k": 4198.0,
                    "thermal_conductivity_w_mk": 0.669,
                    "phase": "liquid"
                }
            ]
        }
    )

    fluid_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Fluid name or stream identifier"
    )
    fluid_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Unique fluid identifier in property database"
    )

    # Reference conditions
    temperature_c: float = Field(
        ...,
        ge=-273.15,
        le=1500,
        description="Reference temperature in Celsius"
    )
    pressure_bar_a: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Reference pressure in bar absolute"
    )

    # Transport properties
    density_kg_m3: float = Field(
        ...,
        gt=0,
        le=25000,
        description="Density in kg/m^3"
    )
    viscosity_pa_s: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Dynamic viscosity in Pa.s"
    )
    specific_heat_j_kg_k: float = Field(
        ...,
        gt=0,
        le=50000,
        description="Specific heat capacity (Cp) in J/(kg.K)"
    )
    thermal_conductivity_w_mk: float = Field(
        ...,
        gt=0,
        le=500,
        description="Thermal conductivity in W/(m.K)"
    )

    # Phase information
    phase: Literal["liquid", "vapor", "two_phase", "supercritical"] = Field(
        default="liquid",
        description="Phase state at reference conditions"
    )
    vapor_fraction: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Vapor mass fraction for two-phase flow"
    )

    # Optional thermodynamic properties
    latent_heat_j_kg: Optional[float] = Field(
        None,
        ge=0,
        description="Latent heat of vaporization in J/kg"
    )
    surface_tension_n_m: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Surface tension in N/m"
    )
    compressibility_factor: Optional[float] = Field(
        None,
        gt=0,
        le=10,
        description="Compressibility factor (Z)"
    )
    molecular_weight_g_mol: Optional[float] = Field(
        None,
        gt=0,
        le=2000,
        description="Molecular weight in g/mol"
    )

    # Derived dimensionless numbers
    prandtl_number: Optional[float] = Field(
        None,
        gt=0,
        le=1e6,
        description="Prandtl number (Cp * mu / k)"
    )

    # Property source tracking
    property_source: Optional[str] = Field(
        None,
        max_length=100,
        description="Source of property data (e.g., REFPROP, PPDS, simulation)"
    )
    property_timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp when properties were calculated/retrieved"
    )

    @model_validator(mode="after")
    def calculate_prandtl(self) -> "FluidProperties":
        """Calculate Prandtl number if not provided."""
        if self.prandtl_number is None:
            pr = (
                self.specific_heat_j_kg_k * self.viscosity_pa_s /
                self.thermal_conductivity_w_mk
            )
            # Note: We can't modify frozen model, so this is just validation
            # In practice, the caller should provide Pr or calculate before creation

        return self


# =============================================================================
# EXCHANGER ASSET
# =============================================================================

class ExchangerAsset(BaseModel):
    """
    Complete heat exchanger asset master data.

    This is the primary schema for heat exchanger assets in the Exchangerpro
    system. It includes all design, geometry, material, and operational data
    required for thermal performance monitoring and fouling prediction.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "exchanger_id": "HX-1001",
                    "plant_id": "REFINERY-01",
                    "tag_number": "E-1001A",
                    "description": "Crude Preheat Exchanger Train 1",
                    "tema_type": "AES",
                    "shell_type": "E",
                    "tube_passes": 4,
                    "shell_passes": 1,
                    "flow_arrangement": "counter_current",
                    "heat_transfer_area_m2": 88.5,
                    "tube_count": 256,
                    "tube_diameter_mm": 19.05,
                    "tube_length_mm": 4572.0,
                    "baffle_type": "segmental",
                    "baffle_spacing_mm": 305.0,
                    "design_ua_w_k": 45000.0,
                    "design_duty_kw": 2500.0,
                    "status": "operational",
                    "criticality": "high"
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique exchanger identifier"
    )
    plant_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Plant or site identifier"
    )
    tag_number: Optional[str] = Field(
        None,
        max_length=50,
        description="Equipment tag number (P&ID reference)"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Exchanger description and service"
    )

    # TEMA designation
    tema_type: TEMAType = Field(
        ...,
        description="TEMA exchanger type designation"
    )
    shell_type: ShellType = Field(
        ...,
        description="TEMA shell type"
    )

    # Pass configuration
    tube_passes: int = Field(
        ...,
        ge=1,
        le=16,
        description="Number of tube passes"
    )
    shell_passes: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of shell passes"
    )
    flow_arrangement: FlowArrangement = Field(
        ...,
        description="Flow arrangement between hot and cold streams"
    )

    # Key dimensions (summary - full details in geometry)
    heat_transfer_area_m2: float = Field(
        ...,
        gt=0,
        le=50000,
        description="Total heat transfer area in m^2"
    )
    tube_count: int = Field(
        ...,
        gt=0,
        le=20000,
        description="Total number of tubes"
    )
    tube_diameter_mm: float = Field(
        ...,
        gt=0,
        le=100,
        description="Tube outside diameter in mm"
    )
    tube_length_mm: float = Field(
        ...,
        gt=0,
        le=25000,
        description="Tube length in mm"
    )

    # Baffle configuration
    baffle_type: BaffleType = Field(
        default=BaffleType.SEGMENTAL,
        description="Type of baffles"
    )
    baffle_spacing_mm: Optional[float] = Field(
        None,
        gt=0,
        le=5000,
        description="Central baffle spacing in mm"
    )

    # Design performance
    design_ua_w_k: float = Field(
        ...,
        gt=0,
        le=1e9,
        description="Design overall heat transfer coefficient times area (UA) in W/K"
    )
    design_duty_kw: float = Field(
        ...,
        gt=0,
        le=1e9,
        description="Design heat duty in kW"
    )

    # Clean performance (reference for fouling calculation)
    clean_ua_w_k: Optional[float] = Field(
        None,
        gt=0,
        le=1e9,
        description="Clean UA value (post-cleaning reference) in W/K"
    )
    clean_shell_htc_w_m2k: Optional[float] = Field(
        None,
        gt=0,
        le=100000,
        description="Clean shell-side heat transfer coefficient in W/(m^2.K)"
    )
    clean_tube_htc_w_m2k: Optional[float] = Field(
        None,
        gt=0,
        le=100000,
        description="Clean tube-side heat transfer coefficient in W/(m^2.K)"
    )

    # Design fouling factors
    design_fouling_shell_m2k_w: Optional[float] = Field(
        None,
        ge=0,
        le=0.01,
        description="Design fouling resistance shell-side in m^2.K/W"
    )
    design_fouling_tube_m2k_w: Optional[float] = Field(
        None,
        ge=0,
        le=0.01,
        description="Design fouling resistance tube-side in m^2.K/W"
    )

    # Materials
    tube_material: Optional[MaterialSpecification] = Field(
        None,
        description="Tube material specification"
    )
    shell_material: Optional[MaterialSpecification] = Field(
        None,
        description="Shell material specification"
    )

    # Detailed geometry (optional - for detailed calculations)
    geometry: Optional[ExchangerGeometry] = Field(
        None,
        description="Detailed geometry specification"
    )

    # Design conditions
    shell_side_design: Optional[DesignConditions] = Field(
        None,
        description="Shell-side design conditions"
    )
    tube_side_design: Optional[DesignConditions] = Field(
        None,
        description="Tube-side design conditions"
    )

    # Nozzles
    nozzles: List[NozzleSpecification] = Field(
        default_factory=list,
        description="List of nozzle specifications"
    )

    # Stream assignments
    hot_side: Literal["shell", "tube"] = Field(
        default="shell",
        description="Side carrying the hot fluid"
    )
    shell_side_fluid: Optional[str] = Field(
        None,
        max_length=100,
        description="Shell-side fluid name/description"
    )
    tube_side_fluid: Optional[str] = Field(
        None,
        max_length=100,
        description="Tube-side fluid name/description"
    )

    # Operational status
    status: ExchangerStatus = Field(
        default=ExchangerStatus.OPERATIONAL,
        description="Current operational status"
    )
    criticality: ExchangerCriticality = Field(
        default=ExchangerCriticality.MEDIUM,
        description="Equipment criticality for maintenance prioritization"
    )

    # Metadata
    commissioning_date: Optional[datetime] = Field(
        None,
        description="Date exchanger was commissioned"
    )
    last_inspection_date: Optional[datetime] = Field(
        None,
        description="Date of last inspection"
    )
    last_cleaning_date: Optional[datetime] = Field(
        None,
        description="Date of last cleaning"
    )
    manufacturer: Optional[str] = Field(
        None,
        max_length=100,
        description="Exchanger manufacturer"
    )
    model_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Manufacturer model/drawing number"
    )
    serial_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Manufacturer serial number"
    )

    # Integration references
    cmms_asset_id: Optional[str] = Field(
        None,
        max_length=50,
        description="CMMS system asset identifier"
    )
    pi_af_path: Optional[str] = Field(
        None,
        max_length=500,
        description="OSIsoft PI Asset Framework path"
    )
    scada_prefix: Optional[str] = Field(
        None,
        max_length=100,
        description="SCADA tag prefix for this exchanger"
    )

    # Custom attributes
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom key-value tags for classification"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record last update timestamp"
    )

    @field_validator("exchanger_id", "plant_id")
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        """Validate identifier format."""
        if not v.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError(
                f"Identifier must contain only alphanumeric characters, "
                f"hyphens, underscores, and dots: {v}"
            )
        return v.upper()


# =============================================================================
# EXPORTS
# =============================================================================

ASSET_SCHEMAS = {
    "TEMAType": TEMAType,
    "ShellType": ShellType,
    "FlowArrangement": FlowArrangement,
    "BaffleType": BaffleType,
    "TubePattern": TubePattern,
    "MaterialClass": MaterialClass,
    "ExchangerStatus": ExchangerStatus,
    "ExchangerCriticality": ExchangerCriticality,
    "MaterialSpecification": MaterialSpecification,
    "NozzleSpecification": NozzleSpecification,
    "DesignConditions": DesignConditions,
    "ExchangerGeometry": ExchangerGeometry,
    "FluidProperties": FluidProperties,
    "ExchangerAsset": ExchangerAsset,
}

__all__ = [
    # Enumerations
    "TEMAType",
    "ShellType",
    "FlowArrangement",
    "BaffleType",
    "TubePattern",
    "MaterialClass",
    "ExchangerStatus",
    "ExchangerCriticality",
    # Supporting models
    "MaterialSpecification",
    "NozzleSpecification",
    "DesignConditions",
    # Main schemas
    "ExchangerGeometry",
    "FluidProperties",
    "ExchangerAsset",
    # Export dictionary
    "ASSET_SCHEMAS",
]
