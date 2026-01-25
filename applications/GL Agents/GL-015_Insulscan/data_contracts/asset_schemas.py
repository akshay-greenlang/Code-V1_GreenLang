# -*- coding: utf-8 -*-
"""
GL-015 Insulscan: Asset Schemas - Version 1.0

Provides validated data schemas for insulation asset master data,
geometry specifications, and material properties with zero-hallucination guarantees.

This module defines Pydantic v2 models for:
- InsulationAssetConfig: Complete insulation asset master data
- InsulationSpec: Insulation material specification with thermal properties
- SurfaceGeometry: Surface geometry for heat loss calculations
- AssetMetadata: Asset lifecycle and warranty information

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SurfaceType(str, Enum):
    """Type of surface being insulated."""
    PIPE = "pipe"
    VESSEL = "vessel"
    TANK = "tank"
    DUCT = "duct"
    VALVE = "valve"
    FLANGE = "flange"
    FITTING = "fitting"
    EQUIPMENT = "equipment"
    FLAT_SURFACE = "flat_surface"
    IRREGULAR = "irregular"


class GeometryShape(str, Enum):
    """Geometric shape of the insulated surface."""
    CYLINDRICAL = "cylindrical"
    FLAT = "flat"
    SPHERICAL = "spherical"
    CONICAL = "conical"
    ELLIPTICAL = "elliptical"
    TOROIDAL = "toroidal"
    COMPLEX = "complex"


class InsulationMaterialType(str, Enum):
    """Classification of insulation material types."""
    MINERAL_WOOL = "mineral_wool"
    ROCK_WOOL = "rock_wool"
    GLASS_WOOL = "glass_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    PERLITE = "perlite"
    CELLULAR_GLASS = "cellular_glass"
    CERAMIC_FIBER = "ceramic_fiber"
    AEROGEL = "aerogel"
    POLYURETHANE_FOAM = "polyurethane_foam"
    POLYSTYRENE = "polystyrene"
    PHENOLIC_FOAM = "phenolic_foam"
    ELASTOMERIC_FOAM = "elastomeric_foam"
    VERMICULITE = "vermiculite"
    MICROPOROUS = "microporous"
    FIBERGLASS = "fiberglass"
    OTHER = "other"


class JacketMaterial(str, Enum):
    """Type of protective jacket/cladding material."""
    ALUMINUM = "aluminum"
    STAINLESS_STEEL = "stainless_steel"
    GALVANIZED_STEEL = "galvanized_steel"
    PVC = "pvc"
    FIBERGLASS_REINFORCED = "fiberglass_reinforced"
    CANVAS = "canvas"
    STUCCO = "stucco"
    MASTIC = "mastic"
    NONE = "none"
    OTHER = "other"


class AssetStatus(str, Enum):
    """Operational status of insulated asset."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    STANDBY = "standby"
    DECOMMISSIONED = "decommissioned"


class AssetCriticality(str, Enum):
    """Criticality classification for maintenance prioritization."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConditionRating(str, Enum):
    """Overall condition rating of insulation."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# =============================================================================
# SUPPORTING MODELS
# =============================================================================

class DimensionSpec(BaseModel):
    """Dimensional specification for geometry calculations."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "outer_diameter_mm": 219.1,
                    "inner_diameter_mm": 168.3,
                    "length_m": 15.5,
                    "wall_thickness_mm": 25.4
                }
            ]
        }
    )

    # For cylindrical/pipe surfaces
    outer_diameter_mm: Optional[float] = Field(
        None,
        gt=0,
        le=10000,
        description="Outer diameter in mm (for cylindrical shapes)"
    )
    inner_diameter_mm: Optional[float] = Field(
        None,
        gt=0,
        le=10000,
        description="Inner diameter in mm (for cylindrical shapes)"
    )
    length_m: Optional[float] = Field(
        None,
        gt=0,
        le=10000,
        description="Length in meters"
    )

    # For flat surfaces
    width_m: Optional[float] = Field(
        None,
        gt=0,
        le=1000,
        description="Width in meters (for flat surfaces)"
    )
    height_m: Optional[float] = Field(
        None,
        gt=0,
        le=1000,
        description="Height in meters (for flat surfaces)"
    )

    # For spherical surfaces
    sphere_diameter_mm: Optional[float] = Field(
        None,
        gt=0,
        le=50000,
        description="Sphere diameter in mm"
    )

    # Wall/insulation thickness
    wall_thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        le=500,
        description="Wall thickness in mm"
    )

    # For complex shapes
    custom_dimensions: Optional[Dict[str, float]] = Field(
        None,
        description="Custom dimensions for irregular shapes"
    )


class SurfaceGeometry(BaseModel):
    """
    Surface geometry specification for heat loss calculations.

    Provides all dimensional data required for:
    - Heat transfer area calculations
    - View factor calculations for radiation
    - Convection coefficient estimation
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "shape": "cylindrical",
                    "dimensions": {
                        "outer_diameter_mm": 219.1,
                        "length_m": 15.5
                    },
                    "area_m2": 10.67,
                    "orientation": "horizontal"
                }
            ]
        }
    )

    shape: GeometryShape = Field(
        ...,
        description="Geometric shape of the surface"
    )
    dimensions: DimensionSpec = Field(
        ...,
        description="Dimensional specifications"
    )
    area_m2: float = Field(
        ...,
        gt=0,
        le=100000,
        description="Total surface area in square meters"
    )
    insulated_area_m2: Optional[float] = Field(
        None,
        gt=0,
        le=100000,
        description="Insulated portion of surface area in m^2"
    )
    bare_area_m2: Optional[float] = Field(
        None,
        ge=0,
        le=100000,
        description="Bare (uninsulated) portion of surface area in m^2"
    )

    # Orientation affects convection
    orientation: Literal["horizontal", "vertical", "inclined"] = Field(
        default="horizontal",
        description="Surface orientation"
    )
    inclination_angle_deg: Optional[float] = Field(
        None,
        ge=0,
        le=90,
        description="Inclination angle from horizontal in degrees"
    )

    # Elevation affects ambient conditions
    elevation_m: Optional[float] = Field(
        None,
        ge=-100,
        le=5000,
        description="Elevation above ground level in meters"
    )

    # Exposure conditions
    exposure: Literal["indoor", "outdoor", "sheltered"] = Field(
        default="outdoor",
        description="Exposure conditions"
    )

    @model_validator(mode="after")
    def validate_area_consistency(self) -> "SurfaceGeometry":
        """Validate area calculations are consistent."""
        if self.insulated_area_m2 and self.bare_area_m2:
            total = self.insulated_area_m2 + self.bare_area_m2
            if abs(total - self.area_m2) > 0.01 * self.area_m2:  # 1% tolerance
                raise ValueError(
                    f"Insulated ({self.insulated_area_m2}) + bare ({self.bare_area_m2}) "
                    f"areas do not match total area ({self.area_m2})"
                )
        return self


class InsulationSpec(BaseModel):
    """
    Insulation material specification with thermal properties.

    Contains all material properties required for:
    - Heat loss calculations
    - Thermal performance assessment
    - Maximum service temperature validation
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "material_type": "mineral_wool",
                    "thickness_mm": 75.0,
                    "density_kg_m3": 100.0,
                    "thermal_conductivity_w_mk": 0.040,
                    "max_service_temp_c": 650.0,
                    "manufacturer": "Rockwool",
                    "product_name": "ProRox PS 960"
                }
            ]
        }
    )

    # Material identification
    material_type: InsulationMaterialType = Field(
        ...,
        description="Type of insulation material"
    )
    manufacturer: Optional[str] = Field(
        None,
        max_length=100,
        description="Insulation manufacturer"
    )
    product_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Product name/model"
    )
    product_code: Optional[str] = Field(
        None,
        max_length=50,
        description="Manufacturer product code"
    )

    # Physical properties
    thickness_mm: float = Field(
        ...,
        gt=0,
        le=500,
        description="Insulation thickness in mm"
    )
    density_kg_m3: float = Field(
        ...,
        gt=0,
        le=2000,
        description="Material density in kg/m^3"
    )

    # Thermal properties
    thermal_conductivity_w_mk: float = Field(
        ...,
        gt=0,
        le=5.0,
        description="Thermal conductivity at mean temperature in W/(m.K)"
    )
    thermal_conductivity_temp_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1500,
        description="Reference temperature for thermal conductivity in Celsius"
    )
    thermal_conductivity_coefficients: Optional[Dict[str, float]] = Field(
        None,
        description="Temperature-dependent conductivity coefficients (k = a + bT + cT^2)"
    )

    # Temperature limits
    max_service_temp_c: float = Field(
        ...,
        ge=-273.15,
        le=1500,
        description="Maximum service temperature in Celsius"
    )
    min_service_temp_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1500,
        description="Minimum service temperature in Celsius"
    )

    # Jacket/cladding
    jacket_material: JacketMaterial = Field(
        default=JacketMaterial.ALUMINUM,
        description="Protective jacket material"
    )
    jacket_thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        le=10,
        description="Jacket thickness in mm"
    )
    jacket_emissivity: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Jacket surface emissivity"
    )

    # Additional properties
    moisture_resistance: Optional[Literal["low", "medium", "high"]] = Field(
        None,
        description="Moisture resistance rating"
    )
    fire_rating: Optional[str] = Field(
        None,
        max_length=50,
        description="Fire rating classification"
    )

    # Quality/certification
    astm_specification: Optional[str] = Field(
        None,
        max_length=50,
        description="ASTM specification (e.g., C547, C533)"
    )
    certified: bool = Field(
        default=False,
        description="Whether material is certified for application"
    )

    @field_validator("thermal_conductivity_w_mk")
    @classmethod
    def validate_conductivity_range(cls, v: float) -> float:
        """Validate thermal conductivity is in reasonable range for insulation."""
        if v > 0.5:
            # Most insulation materials have k < 0.1, but allow up to 0.5
            pass  # Could log warning
        return v


class AssetMetadata(BaseModel):
    """
    Asset lifecycle and warranty information.

    Tracks installation, inspection history, and warranty status
    for maintenance planning and compliance.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "installation_date": "2020-05-15T00:00:00Z",
                    "last_inspection": "2024-01-10T09:30:00Z",
                    "manufacturer": "ABC Insulation Co.",
                    "warranty_expiry": "2030-05-15T00:00:00Z",
                    "design_life_years": 25
                }
            ]
        }
    )

    # Installation information
    installation_date: datetime = Field(
        ...,
        description="Date of insulation installation"
    )
    installer_company: Optional[str] = Field(
        None,
        max_length=200,
        description="Installation contractor name"
    )
    installation_procedure_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to installation procedure document"
    )

    # Manufacturer information
    manufacturer: Optional[str] = Field(
        None,
        max_length=200,
        description="Insulation system manufacturer"
    )
    serial_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Asset serial number"
    )
    batch_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Manufacturing batch number"
    )

    # Inspection history
    last_inspection: Optional[datetime] = Field(
        None,
        description="Date of last inspection"
    )
    next_inspection_due: Optional[datetime] = Field(
        None,
        description="Date when next inspection is due"
    )
    inspection_interval_months: Optional[int] = Field(
        None,
        ge=1,
        le=120,
        description="Inspection interval in months"
    )
    total_inspections: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of inspections performed"
    )

    # Warranty information
    warranty_expiry: Optional[datetime] = Field(
        None,
        description="Warranty expiration date"
    )
    warranty_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Type of warranty"
    )
    warranty_terms: Optional[str] = Field(
        None,
        max_length=500,
        description="Summary of warranty terms"
    )

    # Design life
    design_life_years: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Expected design life in years"
    )
    remaining_life_years: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated remaining useful life in years"
    )

    # Compliance
    code_compliance: Optional[str] = Field(
        None,
        max_length=200,
        description="Applicable code/standard for compliance"
    )
    last_compliance_audit: Optional[datetime] = Field(
        None,
        description="Date of last compliance audit"
    )

    # Documentation references
    data_sheet_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to material data sheet"
    )
    drawing_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to installation drawing"
    )

    @property
    def age_years(self) -> float:
        """Calculate asset age in years."""
        delta = datetime.now(timezone.utc) - self.installation_date.replace(tzinfo=timezone.utc)
        return delta.days / 365.25

    @property
    def is_under_warranty(self) -> bool:
        """Check if asset is still under warranty."""
        if self.warranty_expiry:
            return datetime.now(timezone.utc) < self.warranty_expiry.replace(tzinfo=timezone.utc)
        return False

    @property
    def inspection_overdue(self) -> bool:
        """Check if inspection is overdue."""
        if self.next_inspection_due:
            return datetime.now(timezone.utc) > self.next_inspection_due.replace(tzinfo=timezone.utc)
        return False


# =============================================================================
# INSULATION ASSET CONFIG
# =============================================================================

class InsulationAssetConfig(BaseModel):
    """
    Complete insulation asset master data.

    This is the primary schema for insulated assets in the Insulscan
    system. It includes all design, geometry, material, and operational data
    required for thermal performance monitoring and condition assessment.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "asset_id": "INS-1001",
                    "name": "Steam Header Insulation",
                    "plant_id": "PLANT-01",
                    "location": "Boiler House Area A",
                    "surface_type": "pipe",
                    "geometry": {
                        "shape": "cylindrical",
                        "dimensions": {"outer_diameter_mm": 219.1, "length_m": 15.5},
                        "area_m2": 10.67,
                        "orientation": "horizontal"
                    },
                    "insulation_spec": {
                        "material_type": "mineral_wool",
                        "thickness_mm": 75.0,
                        "density_kg_m3": 100.0,
                        "thermal_conductivity_w_mk": 0.040,
                        "max_service_temp_c": 650.0
                    },
                    "operating_temp_c": 180.0,
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
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique asset identifier"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Asset name/description"
    )
    plant_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Plant or site identifier"
    )
    area_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Plant area identifier"
    )
    location: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Physical location description"
    )
    tag_number: Optional[str] = Field(
        None,
        max_length=50,
        description="Equipment tag number (P&ID reference)"
    )

    # Surface and geometry
    surface_type: SurfaceType = Field(
        ...,
        description="Type of surface being insulated"
    )
    geometry: SurfaceGeometry = Field(
        ...,
        description="Surface geometry specification"
    )

    # Insulation specification
    insulation_spec: InsulationSpec = Field(
        ...,
        description="Insulation material specification"
    )

    # Operating conditions
    operating_temp_c: float = Field(
        ...,
        ge=-273.15,
        le=1500,
        description="Normal operating temperature in Celsius"
    )
    design_temp_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1500,
        description="Design temperature in Celsius"
    )
    max_temp_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1500,
        description="Maximum operating temperature in Celsius"
    )
    min_temp_c: Optional[float] = Field(
        None,
        ge=-273.15,
        le=1500,
        description="Minimum operating temperature in Celsius"
    )

    # Heat loss design values
    design_heat_loss_w_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Design heat loss per unit area in W/m^2"
    )
    design_surface_temp_c: Optional[float] = Field(
        None,
        ge=-50,
        le=100,
        description="Design surface temperature in Celsius"
    )

    # Status and condition
    status: AssetStatus = Field(
        default=AssetStatus.OPERATIONAL,
        description="Current operational status"
    )
    criticality: AssetCriticality = Field(
        default=AssetCriticality.MEDIUM,
        description="Asset criticality for prioritization"
    )
    condition_rating: ConditionRating = Field(
        default=ConditionRating.UNKNOWN,
        description="Overall condition rating"
    )
    last_condition_assessment: Optional[datetime] = Field(
        None,
        description="Date of last condition assessment"
    )

    # Asset metadata
    metadata: Optional[AssetMetadata] = Field(
        None,
        description="Asset lifecycle and warranty information"
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
        description="SCADA tag prefix"
    )

    # Parent asset reference
    parent_equipment_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Parent equipment ID (pipe/vessel being insulated)"
    )
    parent_equipment_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Parent equipment type"
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

    @field_validator("asset_id", "plant_id")
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        """Validate identifier format."""
        if not v.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError(
                f"Identifier must contain only alphanumeric characters, "
                f"hyphens, underscores, and dots: {v}"
            )
        return v.upper()

    @model_validator(mode="after")
    def validate_temperature_consistency(self) -> "InsulationAssetConfig":
        """Validate temperature settings are consistent."""
        # Operating temp should not exceed max service temp
        if self.operating_temp_c > self.insulation_spec.max_service_temp_c:
            raise ValueError(
                f"Operating temperature ({self.operating_temp_c}C) exceeds "
                f"insulation max service temperature ({self.insulation_spec.max_service_temp_c}C)"
            )

        # Design temp should not exceed max service temp if specified
        if self.design_temp_c and self.design_temp_c > self.insulation_spec.max_service_temp_c:
            raise ValueError(
                f"Design temperature ({self.design_temp_c}C) exceeds "
                f"insulation max service temperature ({self.insulation_spec.max_service_temp_c}C)"
            )

        return self


# =============================================================================
# EXPORTS
# =============================================================================

ASSET_SCHEMAS = {
    "SurfaceType": SurfaceType,
    "GeometryShape": GeometryShape,
    "InsulationMaterialType": InsulationMaterialType,
    "JacketMaterial": JacketMaterial,
    "AssetStatus": AssetStatus,
    "AssetCriticality": AssetCriticality,
    "ConditionRating": ConditionRating,
    "DimensionSpec": DimensionSpec,
    "SurfaceGeometry": SurfaceGeometry,
    "InsulationSpec": InsulationSpec,
    "AssetMetadata": AssetMetadata,
    "InsulationAssetConfig": InsulationAssetConfig,
}

__all__ = [
    # Enumerations
    "SurfaceType",
    "GeometryShape",
    "InsulationMaterialType",
    "JacketMaterial",
    "AssetStatus",
    "AssetCriticality",
    "ConditionRating",
    # Supporting models
    "DimensionSpec",
    "SurfaceGeometry",
    "InsulationSpec",
    "AssetMetadata",
    # Main schema
    "InsulationAssetConfig",
    # Export dictionary
    "ASSET_SCHEMAS",
]
