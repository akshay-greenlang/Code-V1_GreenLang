# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Schema Definitions

Pydantic v2 models for all inputs, outputs, thermal measurements,
insulation assessments, heat loss calculations, and repair recommendations
for the INSULSCAN agent.

All schemas support zero-hallucination principles with deterministic
calculations, SHA-256 provenance tracking, and regulatory compliance.

Models:
    - InsulationAsset: Physical insulation asset definition
    - ThermalMeasurement: Temperature and thermal camera readings
    - HotSpotDetection: Detected thermal anomalies
    - InsulationCondition: Condition assessment results
    - HeatLossResult: Heat loss calculation outputs
    - RepairRecommendation: Maintenance recommendations
    - AnalysisResult: Complete analysis results with provenance

Author: GreenLang GL-015 INSULSCAN
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field

from .config import (
    InsulationType,
    SurfaceType,
    HotSpotSeverity,
    ConditionSeverity,
    RepairPriority,
    RepairType,
    DataQuality,
    DEFAULT_THERMAL_CONDUCTIVITY,
    DEFAULT_EMISSIVITY,
)


# =============================================================================
# ENUMS
# =============================================================================


class AnalysisStatus(str, Enum):
    """Analysis execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


class MeasurementType(str, Enum):
    """Type of thermal measurement."""
    SPOT = "spot"
    LINE_SCAN = "line_scan"
    AREA_SCAN = "area_scan"
    THERMAL_IMAGE = "thermal_image"
    CONTACT_PROBE = "contact_probe"


class DamageType(str, Enum):
    """Types of insulation damage."""
    NONE = "none"
    MOISTURE_INGRESS = "moisture_ingress"
    PHYSICAL_DAMAGE = "physical_damage"
    THERMAL_DEGRADATION = "thermal_degradation"
    CHEMICAL_ATTACK = "chemical_attack"
    UV_DEGRADATION = "uv_degradation"
    COMPRESSION = "compression"
    MISSING_SECTION = "missing_section"
    POOR_INSTALLATION = "poor_installation"
    JACKETING_FAILURE = "jacketing_failure"


# =============================================================================
# ASSET SCHEMAS
# =============================================================================


class AssetLocation(BaseModel):
    """
    Physical location of an insulation asset.

    Captures plant location hierarchy and geographic coordinates.
    """

    plant_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Plant or facility identifier"
    )
    area: str = Field(
        default="",
        max_length=100,
        description="Plant area or unit"
    )
    building: str = Field(
        default="",
        max_length=100,
        description="Building or structure"
    )
    level: str = Field(
        default="",
        max_length=50,
        description="Floor or elevation level"
    )
    equipment_tag: str = Field(
        default="",
        max_length=50,
        description="Associated equipment tag"
    )
    gps_latitude: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="GPS latitude"
    )
    gps_longitude: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="GPS longitude"
    )
    elevation_m: Optional[float] = Field(
        default=None,
        ge=-500.0,
        le=10000.0,
        description="Elevation above sea level (m)"
    )


class InsulationAsset(BaseModel):
    """
    Physical insulation asset definition.

    Defines the physical properties and configuration of an
    insulated surface or equipment.

    Example:
        >>> asset = InsulationAsset(
        ...     asset_id="INS-001",
        ...     surface_type=SurfaceType.PIPE,
        ...     insulation_type=InsulationType.MINERAL_WOOL,
        ...     thickness_mm=75.0,
        ...     operating_temp_c=200.0,
        ...     ambient_temp_c=25.0,
        ...     surface_area_m2=15.5
        ... )
    """

    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique asset identifier"
    )
    asset_name: str = Field(
        default="",
        max_length=200,
        description="Human-readable asset name"
    )

    # Surface and insulation type
    surface_type: SurfaceType = Field(
        ...,
        description="Type of surface (pipe, vessel, tank, etc.)"
    )
    insulation_type: InsulationType = Field(
        ...,
        description="Type of insulation material"
    )

    # Dimensions
    thickness_mm: float = Field(
        ...,
        ge=10.0,
        le=500.0,
        description="Insulation thickness (mm)"
    )
    surface_area_m2: float = Field(
        ...,
        ge=0.01,
        le=10000.0,
        description="Total insulated surface area (m2)"
    )

    # For cylindrical surfaces (pipes, vessels)
    outer_diameter_mm: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=10000.0,
        description="Outer diameter of pipe/vessel (mm)"
    )
    length_m: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=1000.0,
        description="Length of insulated section (m)"
    )

    # Operating conditions
    operating_temp_c: float = Field(
        ...,
        ge=-200.0,
        le=1000.0,
        description="Operating/process temperature (C)"
    )
    ambient_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=60.0,
        description="Ambient temperature (C)"
    )
    design_temp_c: Optional[float] = Field(
        default=None,
        ge=-200.0,
        le=1000.0,
        description="Design temperature (C)"
    )

    # Material properties
    thermal_conductivity_w_mk: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=1.0,
        description="Insulation thermal conductivity (W/m-K)"
    )
    surface_emissivity: float = Field(
        default=0.90,
        ge=0.01,
        le=1.0,
        description="Surface emissivity for radiation heat loss"
    )
    jacketing_type: str = Field(
        default="aluminum",
        description="Type of protective jacketing"
    )

    # Location
    location: Optional[AssetLocation] = Field(
        default=None,
        description="Physical location"
    )

    # Metadata
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Installation date"
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Last inspection date"
    )
    service_description: str = Field(
        default="",
        description="Service description (e.g., 'Steam header 150 psig')"
    )

    @computed_field
    @property
    def effective_thermal_conductivity(self) -> float:
        """Get effective thermal conductivity (W/m-K)."""
        if self.thermal_conductivity_w_mk is not None:
            return self.thermal_conductivity_w_mk
        return DEFAULT_THERMAL_CONDUCTIVITY.get(
            self.insulation_type.value, 0.04
        )

    @computed_field
    @property
    def temperature_differential_c(self) -> float:
        """Temperature difference between operating and ambient (C)."""
        return abs(self.operating_temp_c - self.ambient_temp_c)

    @computed_field
    @property
    def is_hot_service(self) -> bool:
        """True if operating temperature is above ambient (hot service)."""
        return self.operating_temp_c > self.ambient_temp_c

    @computed_field
    @property
    def asset_age_years(self) -> Optional[float]:
        """Calculate asset age in years if installation date is known."""
        if self.installation_date is None:
            return None
        delta = datetime.now(timezone.utc) - self.installation_date
        return delta.days / 365.25

    @model_validator(mode="after")
    def validate_asset(self) -> "InsulationAsset":
        """Validate asset configuration."""
        # For pipes, outer diameter should be specified
        if self.surface_type == SurfaceType.PIPE:
            if self.outer_diameter_mm is None:
                # Default to a reasonable pipe size if not specified
                pass
        return self

    model_config = {"use_enum_values": True}


# =============================================================================
# THERMAL MEASUREMENT SCHEMAS
# =============================================================================


class ThermalMeasurement(BaseModel):
    """
    Thermal measurement data from inspections.

    Captures temperature readings from thermal cameras, spot
    measurements, or contact probes.

    Example:
        >>> measurement = ThermalMeasurement(
        ...     asset_id="INS-001",
        ...     timestamp=datetime.now(timezone.utc),
        ...     surface_temp_c=45.0,
        ...     ambient_temp_c=25.0,
        ...     emissivity=0.90,
        ...     data_quality=DataQuality.GOOD
        ... )
    """

    measurement_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique measurement identifier"
    )
    asset_id: str = Field(
        ...,
        description="Asset being measured"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp (UTC)"
    )

    # Temperature readings
    surface_temp_c: float = Field(
        ...,
        ge=-100.0,
        le=1000.0,
        description="Measured surface temperature (C)"
    )
    ambient_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=60.0,
        description="Ambient temperature at measurement time (C)"
    )
    min_temp_c: Optional[float] = Field(
        default=None,
        ge=-100.0,
        le=1000.0,
        description="Minimum temperature in measurement area (C)"
    )
    max_temp_c: Optional[float] = Field(
        default=None,
        ge=-100.0,
        le=1000.0,
        description="Maximum temperature in measurement area (C)"
    )
    avg_temp_c: Optional[float] = Field(
        default=None,
        ge=-100.0,
        le=1000.0,
        description="Average temperature in measurement area (C)"
    )

    # Measurement parameters
    emissivity: float = Field(
        default=0.90,
        ge=0.01,
        le=1.0,
        description="Emissivity setting used for measurement"
    )
    reflected_temp_c: float = Field(
        default=25.0,
        ge=-50.0,
        le=100.0,
        description="Reflected apparent temperature (C)"
    )
    distance_m: float = Field(
        default=2.0,
        ge=0.1,
        le=100.0,
        description="Distance from camera/probe to surface (m)"
    )

    # Environmental conditions
    wind_speed_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Wind speed at measurement location (m/s)"
    )
    relative_humidity_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Relative humidity (%)"
    )

    # Measurement metadata
    measurement_type: MeasurementType = Field(
        default=MeasurementType.SPOT,
        description="Type of measurement"
    )
    data_quality: DataQuality = Field(
        default=DataQuality.GOOD,
        description="Data quality flag"
    )
    source_system: str = Field(
        default="thermal_camera",
        description="Data source system"
    )
    operator_id: str = Field(
        default="",
        description="Operator/inspector ID"
    )

    # Image reference
    thermal_image_path: Optional[str] = Field(
        default=None,
        description="Path to thermal image file"
    )

    @computed_field
    @property
    def temperature_delta_c(self) -> float:
        """Temperature difference between surface and ambient (C)."""
        return self.surface_temp_c - self.ambient_temp_c

    @computed_field
    @property
    def temperature_range_c(self) -> Optional[float]:
        """Temperature range in measurement area (C)."""
        if self.max_temp_c is not None and self.min_temp_c is not None:
            return self.max_temp_c - self.min_temp_c
        return None

    model_config = {"use_enum_values": True}


# =============================================================================
# HOT SPOT DETECTION SCHEMAS
# =============================================================================


class HotSpotDetection(BaseModel):
    """
    Detected thermal anomaly (hot spot).

    Represents a localized area of elevated temperature indicating
    potential insulation damage or degradation.

    Example:
        >>> hot_spot = HotSpotDetection(
        ...     location="Pipe section 3, near elbow",
        ...     severity=HotSpotSeverity.WARNING,
        ...     temp_differential_c=15.5,
        ...     area_m2=0.25
        ... )
    """

    detection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique detection identifier"
    )
    asset_id: str = Field(
        default="",
        description="Associated asset ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp"
    )

    # Location
    location: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Description of hot spot location"
    )
    x_position_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="X position on asset (m)"
    )
    y_position_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Y position on asset (m)"
    )

    # Thermal characteristics
    severity: HotSpotSeverity = Field(
        ...,
        description="Hot spot severity classification"
    )
    temp_differential_c: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Temperature above expected surface temp (C)"
    )
    peak_temp_c: float = Field(
        default=0.0,
        ge=-100.0,
        le=1000.0,
        description="Peak temperature in hot spot (C)"
    )
    expected_temp_c: float = Field(
        default=0.0,
        ge=-100.0,
        le=1000.0,
        description="Expected surface temperature (C)"
    )

    # Size
    area_m2: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Affected area (m2)"
    )
    diameter_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10000.0,
        description="Approximate diameter for circular hot spots (mm)"
    )

    # Likely cause
    probable_cause: DamageType = Field(
        default=DamageType.NONE,
        description="Probable cause of hot spot"
    )
    cause_confidence_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Confidence in cause determination"
    )

    # Personnel safety
    personnel_safety_risk: bool = Field(
        default=False,
        description="True if temperature exceeds personnel safety limits"
    )

    # Heat loss impact
    estimated_heat_loss_w: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated additional heat loss from hot spot (W)"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )

    @computed_field
    @property
    def is_critical(self) -> bool:
        """True if hot spot requires immediate attention."""
        return self.severity in [HotSpotSeverity.CRITICAL, HotSpotSeverity.EMERGENCY]

    model_config = {"use_enum_values": True}


# =============================================================================
# INSULATION CONDITION SCHEMAS
# =============================================================================


class InsulationCondition(BaseModel):
    """
    Insulation condition assessment results.

    Provides a comprehensive condition score and degradation
    analysis for an insulation asset.

    Example:
        >>> condition = InsulationCondition(
        ...     asset_id="INS-001",
        ...     condition_score=75,
        ...     degradation_percent=15.0,
        ...     remaining_life_years=12.5,
        ...     severity=ConditionSeverity.GOOD
        ... )
    """

    assessment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique assessment identifier"
    )
    asset_id: str = Field(
        ...,
        description="Asset identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Assessment timestamp"
    )

    # Condition metrics
    condition_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall condition score (0-100)"
    )
    degradation_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Degradation from as-new condition (%)"
    )
    remaining_life_years: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Estimated remaining service life (years)"
    )

    # Severity classification
    severity: ConditionSeverity = Field(
        ...,
        description="Condition severity classification"
    )

    # Thermal efficiency
    thermal_efficiency_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Thermal performance vs. as-new (%)"
    )
    effective_thickness_mm: float = Field(
        default=0.0,
        ge=0.0,
        le=500.0,
        description="Effective insulation thickness accounting for degradation"
    )

    # Damage assessment
    damage_types: List[DamageType] = Field(
        default_factory=list,
        description="Types of damage observed"
    )
    damage_extent_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Extent of damage as percentage of surface"
    )

    # Hot spots summary
    hot_spot_count: int = Field(
        default=0,
        ge=0,
        description="Number of hot spots detected"
    )
    critical_hot_spots: int = Field(
        default=0,
        ge=0,
        description="Number of critical hot spots"
    )

    # Contributing factors
    factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Contributing factors to condition (factor -> score)"
    )

    # Trend analysis
    trend_direction: str = Field(
        default="stable",
        description="Condition trend (improving, stable, degrading)"
    )
    degradation_rate_percent_year: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Annual degradation rate (%)"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )
    assessment_method: str = Field(
        default="thermal_analysis",
        description="Assessment method used"
    )

    @classmethod
    def calculate(
        cls,
        asset_id: str,
        thermal_efficiency: float,
        hot_spot_count: int,
        critical_hot_spots: int,
        damage_types: List[DamageType],
        asset_age_years: Optional[float] = None,
        typical_lifetime_years: float = 20.0,
    ) -> "InsulationCondition":
        """
        Calculate insulation condition from inspection data.

        ZERO-HALLUCINATION: Uses deterministic scoring formulas.

        Args:
            asset_id: Asset identifier
            thermal_efficiency: Thermal efficiency vs. as-new (0-100%)
            hot_spot_count: Number of detected hot spots
            critical_hot_spots: Number of critical hot spots
            damage_types: List of observed damage types
            asset_age_years: Asset age in years
            typical_lifetime_years: Expected lifetime

        Returns:
            InsulationCondition with calculated metrics
        """
        # Base score from thermal efficiency
        base_score = thermal_efficiency

        # Deductions for hot spots
        hot_spot_deduction = min(hot_spot_count * 3, 20)
        critical_deduction = min(critical_hot_spots * 10, 30)

        # Deductions for damage types
        damage_deduction = min(len(damage_types) * 5, 25)

        # Calculate condition score
        condition_score = max(0, int(
            base_score - hot_spot_deduction - critical_deduction - damage_deduction
        ))

        # Calculate degradation
        degradation_percent = 100.0 - thermal_efficiency

        # Estimate remaining life
        if asset_age_years is not None and asset_age_years > 0:
            # Use age-based degradation rate
            if degradation_percent > 0:
                degradation_rate = degradation_percent / asset_age_years
                remaining_capacity = 100.0 - degradation_percent
                remaining_life = remaining_capacity / max(degradation_rate, 0.5)
            else:
                remaining_life = typical_lifetime_years - asset_age_years
        else:
            # Estimate from condition score
            remaining_life = (condition_score / 100.0) * typical_lifetime_years

        remaining_life = max(0.0, min(remaining_life, typical_lifetime_years))

        # Determine severity
        if condition_score >= 90:
            severity = ConditionSeverity.EXCELLENT
        elif condition_score >= 75:
            severity = ConditionSeverity.GOOD
        elif condition_score >= 60:
            severity = ConditionSeverity.FAIR
        elif condition_score >= 40:
            severity = ConditionSeverity.POOR
        elif condition_score >= 20:
            severity = ConditionSeverity.CRITICAL
        else:
            severity = ConditionSeverity.FAILED

        # Compute provenance hash
        hash_input = {
            "asset_id": asset_id,
            "thermal_efficiency": round(thermal_efficiency, 2),
            "hot_spot_count": hot_spot_count,
            "condition_score": condition_score,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()[:16]

        return cls(
            asset_id=asset_id,
            condition_score=condition_score,
            degradation_percent=degradation_percent,
            remaining_life_years=remaining_life,
            severity=severity,
            thermal_efficiency_percent=thermal_efficiency,
            damage_types=damage_types,
            hot_spot_count=hot_spot_count,
            critical_hot_spots=critical_hot_spots,
            degradation_rate_percent_year=degradation_percent / max(asset_age_years or 1, 1),
            provenance_hash=provenance_hash,
        )

    model_config = {"use_enum_values": True}


# =============================================================================
# HEAT LOSS CALCULATION SCHEMAS
# =============================================================================


class HeatLossResult(BaseModel):
    """
    Heat loss calculation results.

    Contains deterministic heat loss calculations based on
    thermodynamic formulas and economic impact analysis.

    Example:
        >>> heat_loss = HeatLossResult(
        ...     asset_id="INS-001",
        ...     heat_loss_w=1500.0,
        ...     heat_loss_per_area_w_m2=96.8,
        ...     energy_cost_usd_year=1314.0,
        ...     co2_emissions_kg_year=5256.0
        ... )
    """

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation identifier"
    )
    asset_id: str = Field(
        ...,
        description="Asset identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Heat loss values
    heat_loss_w: float = Field(
        ...,
        ge=0.0,
        description="Total heat loss (W)"
    )
    heat_loss_per_area_w_m2: float = Field(
        ...,
        ge=0.0,
        description="Heat loss per unit area (W/m2)"
    )
    heat_loss_per_length_w_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Heat loss per unit length for pipes (W/m)"
    )

    # Component breakdown
    convective_loss_w: float = Field(
        default=0.0,
        ge=0.0,
        description="Convective heat loss component (W)"
    )
    radiative_loss_w: float = Field(
        default=0.0,
        ge=0.0,
        description="Radiative heat loss component (W)"
    )

    # Surface temperature
    calculated_surface_temp_c: float = Field(
        default=0.0,
        ge=-100.0,
        le=1000.0,
        description="Calculated outer surface temperature (C)"
    )
    measured_surface_temp_c: Optional[float] = Field(
        default=None,
        ge=-100.0,
        le=1000.0,
        description="Measured surface temperature for comparison (C)"
    )

    # Economic impact (annual)
    energy_cost_usd_year: float = Field(
        ...,
        ge=0.0,
        description="Annual energy cost from heat loss (USD)"
    )
    co2_emissions_kg_year: float = Field(
        ...,
        ge=0.0,
        description="Annual CO2 emissions from heat loss (kg)"
    )

    # Comparison to ideal
    excess_heat_loss_w: float = Field(
        default=0.0,
        ge=0.0,
        description="Excess heat loss vs. ideal condition (W)"
    )
    excess_cost_usd_year: float = Field(
        default=0.0,
        ge=0.0,
        description="Excess annual cost vs. ideal (USD)"
    )

    # Calculation metadata
    formula_version: str = Field(
        default="HEAT_LOSS_v1.0",
        description="Calculation formula version"
    )
    operating_hours_year: int = Field(
        default=8760,
        ge=1000,
        le=8760,
        description="Operating hours assumed for annual calculations"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )

    @classmethod
    def calculate(
        cls,
        asset: "InsulationAsset",
        surface_temp_c: Optional[float] = None,
        wind_speed_ms: float = 0.0,
        energy_cost_usd_per_kwh: float = 0.10,
        co2_factor_kg_per_kwh: float = 0.4,
        operating_hours: int = 8760,
    ) -> "HeatLossResult":
        """
        Calculate heat loss using deterministic thermodynamic formulas.

        ZERO-HALLUCINATION: Uses only physics-based heat transfer equations.

        Heat Transfer Equations:
        - Conduction: Q = k * A * (T1 - T2) / thickness
        - Convection: Q = h * A * (Ts - Ta)
        - Radiation: Q = epsilon * sigma * A * (Ts^4 - Ta^4)

        Args:
            asset: Insulation asset definition
            surface_temp_c: Measured surface temp (if None, calculates it)
            wind_speed_ms: Wind speed for convection adjustment
            energy_cost_usd_per_kwh: Energy cost per kWh
            co2_factor_kg_per_kwh: CO2 emission factor
            operating_hours: Annual operating hours

        Returns:
            HeatLossResult with calculated values
        """
        # Constants
        STEFAN_BOLTZMANN = 5.67e-8  # W/m2-K4

        # Get properties
        k = asset.effective_thermal_conductivity  # W/m-K
        A = asset.surface_area_m2  # m2
        t = asset.thickness_mm / 1000.0  # Convert to m
        T_op = asset.operating_temp_c + 273.15  # K
        T_amb = asset.ambient_temp_c + 273.15  # K
        epsilon = asset.surface_emissivity

        # Calculate or use measured surface temperature
        if surface_temp_c is not None:
            T_surf = surface_temp_c + 273.15  # K
        else:
            # Estimate surface temperature from thermal resistance
            # Simplified: assumes steady state conduction through insulation
            # R_insulation = thickness / (k * A)
            # For flat surface approximation
            delta_T = asset.operating_temp_c - asset.ambient_temp_c
            # Surface temp is between operating and ambient
            # Higher insulation -> surface temp closer to ambient
            thermal_resistance = t / k  # m2-K/W
            # Estimate convection coefficient
            h_conv = 10.0 + 5.0 * wind_speed_ms  # W/m2-K (simplified)
            surface_resistance = 1.0 / h_conv
            total_resistance = thermal_resistance + surface_resistance

            # Surface temperature
            T_surf_c = asset.ambient_temp_c + delta_T * (surface_resistance / total_resistance)
            T_surf = T_surf_c + 273.15  # K

        T_surf_c = T_surf - 273.15  # C

        # Convection heat loss
        # h = 10 + 5*v for wind effect (simplified correlation)
        h_conv = 10.0 + 5.0 * wind_speed_ms  # W/m2-K
        Q_conv = h_conv * A * (T_surf - T_amb)  # W

        # Radiation heat loss
        Q_rad = epsilon * STEFAN_BOLTZMANN * A * (T_surf**4 - T_amb**4)  # W

        # Total heat loss
        Q_total = Q_conv + Q_rad  # W

        # Heat loss per unit area
        Q_per_area = Q_total / A if A > 0 else 0.0  # W/m2

        # Heat loss per length (for pipes)
        Q_per_length = None
        if asset.length_m is not None and asset.length_m > 0:
            Q_per_length = Q_total / asset.length_m

        # Annual energy cost
        # Q_total in W -> kWh/year = Q_total / 1000 * hours
        energy_kwh_year = (Q_total / 1000.0) * operating_hours
        energy_cost = energy_kwh_year * energy_cost_usd_per_kwh

        # CO2 emissions
        co2_emissions = energy_kwh_year * co2_factor_kg_per_kwh

        # Compute provenance hash
        hash_input = {
            "asset_id": asset.asset_id,
            "heat_loss_w": round(Q_total, 2),
            "surface_temp_c": round(T_surf_c, 2),
            "formula": "HEAT_LOSS_v1.0",
        }
        provenance_hash = hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()[:16]

        return cls(
            asset_id=asset.asset_id,
            heat_loss_w=Q_total,
            heat_loss_per_area_w_m2=Q_per_area,
            heat_loss_per_length_w_m=Q_per_length,
            convective_loss_w=max(Q_conv, 0),
            radiative_loss_w=max(Q_rad, 0),
            calculated_surface_temp_c=T_surf_c,
            measured_surface_temp_c=surface_temp_c,
            energy_cost_usd_year=energy_cost,
            co2_emissions_kg_year=co2_emissions,
            operating_hours_year=operating_hours,
            provenance_hash=provenance_hash,
        )


# =============================================================================
# REPAIR RECOMMENDATION SCHEMAS
# =============================================================================


class RepairRecommendation(BaseModel):
    """
    Repair/maintenance recommendation for insulation assets.

    Provides actionable recommendations with cost-benefit analysis
    and payback calculations.

    Example:
        >>> recommendation = RepairRecommendation(
        ...     asset_id="INS-001",
        ...     priority=RepairPriority.MEDIUM,
        ...     repair_type=RepairType.SECTION_REPLACEMENT,
        ...     estimated_cost_usd=2500.0,
        ...     payback_years=1.8,
        ...     energy_savings_usd_year=1400.0
        ... )
    """

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique recommendation identifier"
    )
    asset_id: str = Field(
        ...,
        description="Asset identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recommendation generation timestamp"
    )

    # Recommendation
    priority: RepairPriority = Field(
        ...,
        description="Repair priority level"
    )
    repair_type: RepairType = Field(
        ...,
        description="Recommended repair type"
    )
    action_description: str = Field(
        default="",
        max_length=1000,
        description="Detailed action description"
    )

    # Cost estimates
    estimated_cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Estimated repair cost (USD)"
    )
    labor_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated labor hours"
    )
    material_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated material cost (USD)"
    )

    # Benefits
    energy_savings_usd_year: float = Field(
        ...,
        ge=0.0,
        description="Annual energy savings after repair (USD)"
    )
    co2_reduction_kg_year: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual CO2 reduction after repair (kg)"
    )
    heat_loss_reduction_w: float = Field(
        default=0.0,
        ge=0.0,
        description="Heat loss reduction after repair (W)"
    )

    # Payback analysis
    payback_years: float = Field(
        ...,
        ge=0.0,
        description="Simple payback period (years)"
    )
    roi_percent: float = Field(
        default=0.0,
        description="Return on investment (%)"
    )
    npv_usd: float = Field(
        default=0.0,
        description="Net present value over 10 years (USD)"
    )

    # Safety considerations
    safety_improvement: bool = Field(
        default=False,
        description="True if repair improves personnel safety"
    )
    safety_risk_current: str = Field(
        default="",
        description="Current safety risk description"
    )

    # Current condition reference
    current_condition: Optional[InsulationCondition] = Field(
        default=None,
        description="Current condition assessment"
    )
    current_heat_loss: Optional[HeatLossResult] = Field(
        default=None,
        description="Current heat loss calculation"
    )

    # Reasoning
    reasoning: List[str] = Field(
        default_factory=list,
        description="Reasoning for recommendation"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Risk factors if repair delayed"
    )

    # Confidence and provenance
    confidence_percent: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Confidence in recommendation"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )

    @classmethod
    def generate(
        cls,
        asset: InsulationAsset,
        condition: InsulationCondition,
        heat_loss: HeatLossResult,
        labor_rate: float = 75.0,
        material_cost_per_m2: float = 50.0,
    ) -> "RepairRecommendation":
        """
        Generate repair recommendation based on condition and heat loss.

        ZERO-HALLUCINATION: Uses deterministic rules and calculations.

        Args:
            asset: Insulation asset
            condition: Condition assessment
            heat_loss: Heat loss calculation
            labor_rate: Labor rate (USD/hour)
            material_cost_per_m2: Material cost per m2

        Returns:
            RepairRecommendation with cost-benefit analysis
        """
        # Determine priority based on condition severity
        priority_map = {
            ConditionSeverity.EXCELLENT: RepairPriority.NONE,
            ConditionSeverity.GOOD: RepairPriority.LOW,
            ConditionSeverity.FAIR: RepairPriority.MEDIUM,
            ConditionSeverity.POOR: RepairPriority.HIGH,
            ConditionSeverity.CRITICAL: RepairPriority.URGENT,
            ConditionSeverity.FAILED: RepairPriority.IMMEDIATE,
        }
        priority = priority_map.get(condition.severity, RepairPriority.MEDIUM)

        # Determine repair type based on condition
        if condition.condition_score >= 90:
            repair_type = RepairType.NO_ACTION
            action = "No action required. Continue routine monitoring."
        elif condition.condition_score >= 75:
            repair_type = RepairType.MONITORING
            action = "Increase monitoring frequency. Schedule inspection in 6 months."
        elif condition.condition_score >= 60:
            repair_type = RepairType.MINOR_PATCH
            action = "Repair localized damage areas. Apply weatherproofing sealant."
        elif condition.condition_score >= 40:
            repair_type = RepairType.SECTION_REPLACEMENT
            action = "Replace damaged insulation sections. Inspect jacketing integrity."
        else:
            repair_type = RepairType.FULL_REPLACEMENT
            action = "Complete insulation system replacement recommended."

        # Estimate costs
        damage_area = asset.surface_area_m2 * (condition.damage_extent_percent / 100.0)
        if repair_type == RepairType.FULL_REPLACEMENT:
            repair_area = asset.surface_area_m2
        elif repair_type == RepairType.SECTION_REPLACEMENT:
            repair_area = max(damage_area, asset.surface_area_m2 * 0.25)
        else:
            repair_area = damage_area

        material_cost = repair_area * material_cost_per_m2

        # Labor hours estimation (varies by repair type)
        labor_factor = {
            RepairType.NO_ACTION: 0,
            RepairType.MONITORING: 1,
            RepairType.MINOR_PATCH: 2,
            RepairType.SECTION_REPLACEMENT: 4,
            RepairType.FULL_REPLACEMENT: 8,
            RepairType.UPGRADE_THICKNESS: 6,
            RepairType.UPGRADE_MATERIAL: 8,
            RepairType.WEATHERPROOFING: 2,
        }
        labor_hours = repair_area * labor_factor.get(repair_type, 4)
        labor_cost = labor_hours * labor_rate
        total_cost = material_cost + labor_cost

        # Calculate potential savings
        # Assume repair restores to 95% efficiency
        current_efficiency = condition.thermal_efficiency_percent / 100.0
        restored_efficiency = 0.95
        efficiency_improvement = restored_efficiency - current_efficiency

        if efficiency_improvement > 0:
            current_annual_cost = heat_loss.energy_cost_usd_year
            # Simplified: savings proportional to efficiency improvement
            annual_savings = current_annual_cost * (efficiency_improvement / (1 - current_efficiency + 0.001))
            heat_loss_reduction = heat_loss.heat_loss_w * efficiency_improvement
        else:
            annual_savings = 0.0
            heat_loss_reduction = 0.0

        # Payback calculation
        if annual_savings > 0 and total_cost > 0:
            payback = total_cost / annual_savings
        else:
            payback = float('inf')

        # ROI calculation (10-year horizon)
        lifetime_savings = annual_savings * 10
        if total_cost > 0:
            roi = ((lifetime_savings - total_cost) / total_cost) * 100
        else:
            roi = 0.0

        # CO2 reduction
        co2_factor = 0.4  # kg CO2 per kWh
        energy_kwh_savings = annual_savings / 0.10  # Assume $0.10/kWh
        co2_reduction = energy_kwh_savings * co2_factor

        # Safety check
        safety_temp_c = 60.0
        if heat_loss.calculated_surface_temp_c > safety_temp_c:
            safety_improvement = True
            safety_risk = f"Surface temperature {heat_loss.calculated_surface_temp_c:.1f}C exceeds safety limit"
        else:
            safety_improvement = False
            safety_risk = ""

        # Build reasoning
        reasoning = []
        if condition.severity in [ConditionSeverity.POOR, ConditionSeverity.CRITICAL, ConditionSeverity.FAILED]:
            reasoning.append(f"Condition score {condition.condition_score} indicates significant degradation")
        if condition.hot_spot_count > 0:
            reasoning.append(f"{condition.hot_spot_count} hot spots detected")
        if condition.critical_hot_spots > 0:
            reasoning.append(f"{condition.critical_hot_spots} critical hot spots require immediate attention")
        if heat_loss.excess_heat_loss_w > 0:
            reasoning.append(f"Excess heat loss of {heat_loss.excess_heat_loss_w:.0f}W above optimal")
        if payback < 3.0:
            reasoning.append(f"Attractive payback period of {payback:.1f} years")

        # Risk factors
        risk_factors = []
        if condition.remaining_life_years < 3:
            risk_factors.append("Less than 3 years remaining service life")
        if safety_improvement:
            risk_factors.append("Personnel burn risk from elevated surface temperature")
        if condition.severity == ConditionSeverity.CRITICAL:
            risk_factors.append("Risk of complete insulation failure")

        # Provenance hash
        hash_input = {
            "asset_id": asset.asset_id,
            "priority": priority.value,
            "repair_type": repair_type.value,
            "estimated_cost": round(total_cost, 2),
            "payback": round(payback, 2) if payback != float('inf') else "inf",
        }
        provenance_hash = hashlib.sha256(
            json.dumps(hash_input, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        return cls(
            asset_id=asset.asset_id,
            priority=priority,
            repair_type=repair_type,
            action_description=action,
            estimated_cost_usd=total_cost,
            labor_hours=labor_hours,
            material_cost_usd=material_cost,
            energy_savings_usd_year=annual_savings,
            co2_reduction_kg_year=co2_reduction,
            heat_loss_reduction_w=heat_loss_reduction,
            payback_years=min(payback, 100.0),
            roi_percent=roi,
            safety_improvement=safety_improvement,
            safety_risk_current=safety_risk,
            current_condition=condition,
            current_heat_loss=heat_loss,
            reasoning=reasoning,
            risk_factors=risk_factors,
            confidence_percent=85.0,
            provenance_hash=provenance_hash,
        )

    model_config = {"use_enum_values": True}


# =============================================================================
# ANALYSIS RESULT SCHEMAS
# =============================================================================


class AnalysisResult(BaseModel):
    """
    Complete analysis result for an insulation asset.

    Contains all metrics combined with full provenance tracking
    for audit compliance.

    Example:
        >>> result = AnalysisResult(
        ...     asset_id="INS-001",
        ...     condition=condition,
        ...     heat_loss=heat_loss,
        ...     hot_spots=hot_spots,
        ...     recommendation=recommendation
        ... )
    """

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique analysis identifier"
    )
    asset_id: str = Field(
        ...,
        description="Asset identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.COMPLETED,
        description="Analysis status"
    )

    # Input references
    asset: Optional[InsulationAsset] = Field(
        default=None,
        description="Asset definition"
    )
    measurements: List[ThermalMeasurement] = Field(
        default_factory=list,
        description="Thermal measurements used"
    )

    # Results
    condition: Optional[InsulationCondition] = Field(
        default=None,
        description="Condition assessment"
    )
    heat_loss: Optional[HeatLossResult] = Field(
        default=None,
        description="Heat loss calculation"
    )
    hot_spots: List[HotSpotDetection] = Field(
        default_factory=list,
        description="Detected hot spots"
    )
    recommendation: Optional[RepairRecommendation] = Field(
        default=None,
        description="Repair recommendation"
    )

    # Summary metrics
    overall_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Overall asset health score"
    )
    risk_level: str = Field(
        default="low",
        description="Risk level (low, medium, high, critical)"
    )
    action_required: bool = Field(
        default=False,
        description="True if action is required"
    )

    # Execution metadata
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Analysis warnings"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Analysis errors"
    )

    # Provenance
    input_hash: str = Field(
        default="",
        description="SHA-256 hash of inputs"
    )
    output_hash: str = Field(
        default="",
        description="SHA-256 hash of outputs"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    @computed_field
    @property
    def has_critical_issues(self) -> bool:
        """True if any critical issues were found."""
        if self.condition and self.condition.critical_hot_spots > 0:
            return True
        if self.condition and self.condition.severity in [
            ConditionSeverity.CRITICAL, ConditionSeverity.FAILED
        ]:
            return True
        return False

    model_config = {"use_enum_values": True}


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "AnalysisStatus",
    "MeasurementType",
    "DamageType",
    # Location schemas
    "AssetLocation",
    # Asset schemas
    "InsulationAsset",
    # Measurement schemas
    "ThermalMeasurement",
    # Hot spot schemas
    "HotSpotDetection",
    # Condition schemas
    "InsulationCondition",
    # Heat loss schemas
    "HeatLossResult",
    # Recommendation schemas
    "RepairRecommendation",
    # Result schemas
    "AnalysisResult",
]
