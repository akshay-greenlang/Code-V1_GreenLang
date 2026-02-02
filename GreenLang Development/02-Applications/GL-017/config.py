# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC Configuration Models.

This module provides comprehensive Pydantic configuration models for the
Condenser Optimization Agent, including condenser configurations, cooling
water systems, vacuum systems, tube configurations, and performance targets.

All models include validators to ensure compliance with HEI (Heat Exchange
Institute) standards and industry best practices.

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class CondenserType(str, Enum):
    """Condenser type classification."""

    SURFACE = "surface"
    DIRECT_CONTACT = "direct_contact"
    AIR_COOLED = "air_cooled"
    HYBRID = "hybrid"


class CoolingSystemType(str, Enum):
    """Cooling system type classification."""

    ONCE_THROUGH = "once_through"
    OPEN_RECIRCULATING = "open_recirculating"
    CLOSED_RECIRCULATING = "closed_recirculating"
    DRY_COOLING = "dry_cooling"
    HYBRID = "hybrid"


class TubePattern(str, Enum):
    """Tube bundle pattern classification."""

    SINGLE_PASS = "single_pass"
    TWO_PASS = "two_pass"
    MULTI_PASS = "multi_pass"
    DIVIDED_WATERBOX = "divided_waterbox"


class CleaningMethod(str, Enum):
    """Tube cleaning method classification."""

    MECHANICAL_BRUSH = "mechanical_brush"
    SPONGE_BALL = "sponge_ball"
    HYDRO_BLAST = "hydro_blast"
    CHEMICAL = "chemical"
    BALL_RECIRCULATING = "ball_recirculating"


class FoulingType(str, Enum):
    """Fouling type classification."""

    BIOLOGICAL = "biological"
    MINERAL_SCALE = "mineral_scale"
    SILT_DEBRIS = "silt_debris"
    CORROSION_PRODUCTS = "corrosion_products"
    OIL_GREASE = "oil_grease"
    MIXED = "mixed"


class TubeMaterial(str, Enum):
    """Tube material classification."""

    ADMIRALTY_BRASS = "admiralty_brass"
    ALUMINUM_BRASS = "aluminum_brass"
    CUPRONICKEL_90_10 = "cupronickel_90_10"
    CUPRONICKEL_70_30 = "cupronickel_70_30"
    STAINLESS_304 = "stainless_304"
    STAINLESS_316 = "stainless_316"
    TITANIUM = "titanium"
    DUPLEX_SS = "duplex_ss"


class VacuumPumpType(str, Enum):
    """Vacuum pump type classification."""

    STEAM_JET_EJECTOR = "steam_jet_ejector"
    LIQUID_RING = "liquid_ring"
    ROTARY_VANE = "rotary_vane"
    HYBRID_EJECTOR_LRVP = "hybrid_ejector_lrvp"


class AnalyzerType(str, Enum):
    """Analyzer type classification for water quality monitoring."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    CONDUCTIVITY = "conductivity"
    PH = "ph"
    DISSOLVED_OXYGEN = "dissolved_oxygen"
    TURBIDITY = "turbidity"
    CHLORINE = "chlorine"


# ============================================================================
# CONDENSER CONFIGURATION
# ============================================================================


class CondenserConfiguration(BaseModel):
    """
    Condenser configuration and design parameters.

    Defines the condenser characteristics that influence performance
    optimization and efficiency calculations.
    """

    condenser_id: str = Field(..., description="Unique condenser identifier")
    condenser_type: CondenserType = Field(
        ..., description="Condenser type classification"
    )
    design_heat_duty_mmbtu_hr: float = Field(
        ..., gt=0, description="Design heat duty in MMBtu/hr"
    )
    design_steam_flow_lb_hr: float = Field(
        ..., gt=0, description="Design steam flow in lb/hr"
    )
    design_vacuum_in_hg_abs: float = Field(
        ..., gt=0, lt=30, description="Design vacuum in inches Hg absolute"
    )
    design_ttd_f: float = Field(
        ..., gt=0, le=30, description="Design terminal temperature difference in F"
    )
    design_u_value_btu_hr_sqft_f: float = Field(
        ..., gt=0, description="Design overall heat transfer coefficient"
    )
    surface_area_sqft: float = Field(
        ..., gt=0, description="Total heat transfer surface area in sq ft"
    )

    # Tube configuration
    tube_od_inch: float = Field(
        default=0.875, gt=0, description="Tube outer diameter in inches"
    )
    tube_wall_thickness_inch: float = Field(
        default=0.035, gt=0, description="Tube wall thickness in inches"
    )
    tube_length_ft: float = Field(
        default=40.0, gt=0, description="Effective tube length in feet"
    )
    number_of_tubes: int = Field(
        default=10000, gt=0, description="Total number of tubes"
    )
    tube_material: TubeMaterial = Field(
        default=TubeMaterial.STAINLESS_316, description="Tube material"
    )
    tube_pattern: TubePattern = Field(
        default=TubePattern.TWO_PASS, description="Tube bundle pattern"
    )

    # Waterbox configuration
    number_of_waterboxes: int = Field(
        default=2, ge=1, le=8, description="Number of waterboxes"
    )
    waterbox_volume_ft3: float = Field(
        default=500.0, gt=0, description="Waterbox volume in cubic feet"
    )

    # Operating limits
    max_vacuum_in_hg_abs: float = Field(
        default=29.0, gt=0, lt=30, description="Maximum achievable vacuum"
    )
    min_vacuum_in_hg_abs: float = Field(
        default=0.5, gt=0, lt=10, description="Minimum allowable vacuum"
    )
    max_cooling_water_temp_f: float = Field(
        default=95.0, gt=32, description="Maximum cooling water inlet temperature"
    )
    max_cooling_water_flow_gpm: float = Field(
        default=200000.0, gt=0, description="Maximum cooling water flow in GPM"
    )
    min_cooling_water_flow_gpm: float = Field(
        default=50000.0, gt=0, description="Minimum cooling water flow in GPM"
    )

    # Metadata
    location: Optional[str] = Field(None, description="Condenser physical location")
    commissioning_date: Optional[datetime] = Field(
        None, description="Commissioning date"
    )
    last_inspection_date: Optional[datetime] = Field(
        None, description="Last inspection date"
    )
    last_cleaning_date: Optional[datetime] = Field(
        None, description="Last tube cleaning date"
    )

    @field_validator("design_vacuum_in_hg_abs")
    @classmethod
    def validate_vacuum(cls, v: float) -> float:
        """Validate vacuum is within typical operating range."""
        if v < 0.5 or v > 5.0:
            logger.warning(
                f"Vacuum {v} in Hg abs is outside typical range (0.5-5.0)"
            )
        return v

    @model_validator(mode="after")
    def validate_flow_limits(self) -> "CondenserConfiguration":
        """Validate flow limits are consistent."""
        if self.min_cooling_water_flow_gpm >= self.max_cooling_water_flow_gpm:
            raise ValueError(
                "min_cooling_water_flow_gpm must be less than max_cooling_water_flow_gpm"
            )
        return self

    @classmethod
    def create_typical_utility_condenser(
        cls,
        condenser_id: str,
        turbine_capacity_mw: float,
    ) -> "CondenserConfiguration":
        """
        Factory method to create typical utility condenser configuration.

        Args:
            condenser_id: Unique identifier
            turbine_capacity_mw: Steam turbine capacity in MW

        Returns:
            CondenserConfiguration for typical utility application
        """
        # Typical utility condenser sizing ratios
        heat_duty = turbine_capacity_mw * 3.412 * 0.6  # ~60% of output as heat
        steam_flow = turbine_capacity_mw * 10000  # ~10,000 lb/hr per MW
        surface_area = heat_duty * 1000  # ~1000 sq ft per MMBtu/hr

        return cls(
            condenser_id=condenser_id,
            condenser_type=CondenserType.SURFACE,
            design_heat_duty_mmbtu_hr=heat_duty,
            design_steam_flow_lb_hr=steam_flow,
            design_vacuum_in_hg_abs=1.5,
            design_ttd_f=7.0,
            design_u_value_btu_hr_sqft_f=500.0,
            surface_area_sqft=surface_area,
            tube_od_inch=0.875,
            tube_wall_thickness_inch=0.035,
            tube_length_ft=45.0,
            number_of_tubes=int(surface_area / 10),
            tube_material=TubeMaterial.STAINLESS_316,
            tube_pattern=TubePattern.TWO_PASS,
        )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "condenser_id": "COND-001",
                "condenser_type": "surface",
                "design_heat_duty_mmbtu_hr": 2000,
                "design_steam_flow_lb_hr": 500000,
                "design_vacuum_in_hg_abs": 1.5,
                "design_ttd_f": 7.0,
                "design_u_value_btu_hr_sqft_f": 500,
                "surface_area_sqft": 250000,
            }
        }


# ============================================================================
# COOLING WATER CONFIGURATION
# ============================================================================


class CoolingWaterConfig(BaseModel):
    """
    Cooling water system configuration.

    Defines cooling water source, treatment, and flow parameters.
    """

    system_id: str = Field(..., description="Cooling water system identifier")
    system_type: CoolingSystemType = Field(
        ..., description="Cooling system type"
    )

    # Design parameters
    design_flow_gpm: float = Field(
        ..., gt=0, description="Design cooling water flow in GPM"
    )
    design_inlet_temp_f: float = Field(
        ..., gt=32, description="Design inlet temperature in F"
    )
    design_outlet_temp_f: float = Field(
        ..., gt=32, description="Design outlet temperature in F"
    )
    design_pressure_psi: float = Field(
        default=30.0, gt=0, description="Design water pressure in psi"
    )

    # Pumping configuration
    number_of_pumps: int = Field(
        default=3, ge=1, le=10, description="Number of circulating water pumps"
    )
    pump_capacity_gpm: float = Field(
        default=100000.0, gt=0, description="Each pump capacity in GPM"
    )
    pump_head_ft: float = Field(
        default=60.0, gt=0, description="Pump head in feet"
    )
    pump_efficiency_pct: float = Field(
        default=85.0, gt=0, le=100, description="Pump efficiency percentage"
    )
    vfd_enabled: bool = Field(
        default=False, description="Variable frequency drive enabled"
    )

    # Water source
    water_source: str = Field(
        default="cooling_tower", description="Water source (river, lake, tower, etc.)"
    )
    makeup_water_source: Optional[str] = Field(
        None, description="Makeup water source for recirculating systems"
    )
    blowdown_rate_pct: Optional[float] = Field(
        None, ge=0, le=20, description="Blowdown rate for recirculating systems"
    )

    # Operating constraints
    min_flow_gpm: float = Field(
        default=50000.0, gt=0, description="Minimum allowable flow in GPM"
    )
    max_flow_gpm: float = Field(
        default=250000.0, gt=0, description="Maximum allowable flow in GPM"
    )
    max_inlet_temp_f: float = Field(
        default=95.0, gt=32, description="Maximum inlet temperature in F"
    )
    min_inlet_temp_f: float = Field(
        default=40.0, gt=32, description="Minimum inlet temperature in F"
    )

    @field_validator("design_outlet_temp_f")
    @classmethod
    def validate_temp_rise(cls, v: float, info) -> float:
        """Validate temperature rise is reasonable."""
        if "design_inlet_temp_f" in info.data:
            temp_rise = v - info.data["design_inlet_temp_f"]
            if temp_rise < 5 or temp_rise > 40:
                logger.warning(
                    f"Temperature rise {temp_rise}F is outside typical range (5-40F)"
                )
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "system_id": "CW-001",
                "system_type": "open_recirculating",
                "design_flow_gpm": 150000,
                "design_inlet_temp_f": 75,
                "design_outlet_temp_f": 95,
                "number_of_pumps": 3,
            }
        }


# ============================================================================
# VACUUM SYSTEM CONFIGURATION
# ============================================================================


class VacuumSystemConfig(BaseModel):
    """
    Vacuum system configuration.

    Defines vacuum equipment and air removal capacity.
    """

    system_id: str = Field(..., description="Vacuum system identifier")
    primary_pump_type: VacuumPumpType = Field(
        ..., description="Primary vacuum pump type"
    )

    # Vacuum pump configuration
    number_of_stages: int = Field(
        default=2, ge=1, le=4, description="Number of ejector/pump stages"
    )
    design_air_removal_scfm: float = Field(
        ..., gt=0, description="Design air removal capacity in SCFM"
    )
    design_suction_pressure_in_hg_abs: float = Field(
        ..., gt=0, lt=5, description="Design suction pressure in Hg abs"
    )
    motive_steam_pressure_psig: Optional[float] = Field(
        None, gt=0, description="Motive steam pressure for ejectors"
    )
    motive_steam_flow_lb_hr: Optional[float] = Field(
        None, gt=0, description="Motive steam flow for ejectors"
    )

    # Inter-condenser configuration (for multi-stage systems)
    intercondenser_type: Optional[str] = Field(
        None, description="Inter-condenser type (surface, direct_contact)"
    )
    intercondenser_cooling_water_gpm: Optional[float] = Field(
        None, gt=0, description="Inter-condenser cooling water flow"
    )

    # Backup/standby equipment
    backup_pump_type: Optional[VacuumPumpType] = Field(
        None, description="Backup vacuum pump type"
    )
    backup_pump_capacity_scfm: Optional[float] = Field(
        None, gt=0, description="Backup pump capacity"
    )

    # Air leakage monitoring
    air_leakage_design_scfm: float = Field(
        default=5.0, gt=0, description="Design air leakage rate in SCFM"
    )
    air_leakage_alarm_scfm: float = Field(
        default=15.0, gt=0, description="Air leakage alarm threshold in SCFM"
    )

    # Operating constraints
    min_vacuum_in_hg_abs: float = Field(
        default=0.5, gt=0, description="Minimum achievable vacuum"
    )
    max_vacuum_in_hg_abs: float = Field(
        default=29.0, gt=0, lt=30, description="Maximum achievable vacuum"
    )

    @field_validator("air_leakage_alarm_scfm")
    @classmethod
    def validate_alarm_threshold(cls, v: float, info) -> float:
        """Validate alarm threshold is greater than design rate."""
        if "air_leakage_design_scfm" in info.data:
            if v <= info.data["air_leakage_design_scfm"]:
                raise ValueError(
                    "air_leakage_alarm_scfm must be greater than air_leakage_design_scfm"
                )
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "system_id": "VAC-001",
                "primary_pump_type": "steam_jet_ejector",
                "number_of_stages": 2,
                "design_air_removal_scfm": 100,
                "design_suction_pressure_in_hg_abs": 1.0,
                "motive_steam_pressure_psig": 150,
            }
        }


# ============================================================================
# TUBE CONFIGURATION
# ============================================================================


class TubeConfiguration(BaseModel):
    """
    Detailed tube configuration for heat transfer calculations.

    Provides tube geometry and thermal properties.
    """

    tube_id: str = Field(..., description="Tube configuration identifier")
    tube_material: TubeMaterial = Field(..., description="Tube material")

    # Geometry
    outer_diameter_inch: float = Field(
        ..., gt=0, le=2, description="Outer diameter in inches"
    )
    wall_thickness_inch: float = Field(
        ..., gt=0, description="Wall thickness in inches"
    )
    length_ft: float = Field(
        ..., gt=0, description="Tube length in feet"
    )

    # Thermal properties (material-dependent)
    thermal_conductivity_btu_hr_ft_f: float = Field(
        ..., gt=0, description="Thermal conductivity in BTU/hr-ft-F"
    )

    # Surface enhancement
    enhanced_surface: bool = Field(
        default=False, description="Enhanced surface tubes"
    )
    enhancement_factor: float = Field(
        default=1.0, ge=1.0, le=3.0, description="Surface enhancement factor"
    )

    # Fouling characteristics
    design_fouling_factor_hr_sqft_f_btu: float = Field(
        default=0.0005, gt=0, description="Design fouling factor"
    )
    current_fouling_factor_hr_sqft_f_btu: float = Field(
        default=0.0005, gt=0, description="Current estimated fouling factor"
    )
    max_fouling_factor_hr_sqft_f_btu: float = Field(
        default=0.002, gt=0, description="Maximum fouling factor before cleaning"
    )

    @property
    def inner_diameter_inch(self) -> float:
        """Calculate inner diameter."""
        return self.outer_diameter_inch - 2 * self.wall_thickness_inch

    @classmethod
    def from_material(
        cls,
        tube_id: str,
        material: TubeMaterial,
        od_inch: float = 0.875,
        wall_inch: float = 0.035,
        length_ft: float = 40.0,
    ) -> "TubeConfiguration":
        """
        Factory method to create tube configuration from material.

        Args:
            tube_id: Tube configuration identifier
            material: Tube material
            od_inch: Outer diameter in inches
            wall_inch: Wall thickness in inches
            length_ft: Tube length in feet

        Returns:
            TubeConfiguration with material-appropriate thermal conductivity
        """
        # Thermal conductivity values by material (BTU/hr-ft-F)
        conductivity_map = {
            TubeMaterial.ADMIRALTY_BRASS: 64.0,
            TubeMaterial.ALUMINUM_BRASS: 58.0,
            TubeMaterial.CUPRONICKEL_90_10: 26.0,
            TubeMaterial.CUPRONICKEL_70_30: 17.0,
            TubeMaterial.STAINLESS_304: 9.4,
            TubeMaterial.STAINLESS_316: 9.4,
            TubeMaterial.TITANIUM: 12.0,
            TubeMaterial.DUPLEX_SS: 9.0,
        }

        return cls(
            tube_id=tube_id,
            tube_material=material,
            outer_diameter_inch=od_inch,
            wall_thickness_inch=wall_inch,
            length_ft=length_ft,
            thermal_conductivity_btu_hr_ft_f=conductivity_map.get(material, 10.0),
        )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "tube_id": "TUBE-001",
                "tube_material": "stainless_316",
                "outer_diameter_inch": 0.875,
                "wall_thickness_inch": 0.035,
                "length_ft": 40,
                "thermal_conductivity_btu_hr_ft_f": 9.4,
            }
        }


# ============================================================================
# WATER QUALITY LIMITS
# ============================================================================


class WaterQualityLimits(BaseModel):
    """
    Cooling water quality limits for condenser operation.

    Defines acceptable ranges for water chemistry parameters
    that affect fouling and corrosion.
    """

    limit_id: str = Field(..., description="Limit set identifier")
    description: str = Field(
        default="Standard cooling water quality limits",
        description="Limit set description"
    )

    # Temperature limits
    max_inlet_temp_f: float = Field(
        default=95.0, gt=32, description="Maximum inlet temperature"
    )
    max_outlet_temp_f: float = Field(
        default=115.0, gt=32, description="Maximum outlet temperature"
    )
    max_temp_rise_f: float = Field(
        default=25.0, gt=0, description="Maximum temperature rise"
    )

    # Chemical limits
    ph_min: float = Field(
        default=6.5, ge=0, le=14, description="Minimum pH"
    )
    ph_max: float = Field(
        default=9.0, ge=0, le=14, description="Maximum pH"
    )
    total_dissolved_solids_max_ppm: float = Field(
        default=2000, gt=0, description="Maximum TDS in ppm"
    )
    calcium_hardness_max_ppm: float = Field(
        default=500, gt=0, description="Maximum calcium hardness in ppm"
    )
    total_alkalinity_max_ppm: float = Field(
        default=500, gt=0, description="Maximum alkalinity in ppm"
    )
    silica_max_ppm: float = Field(
        default=150, gt=0, description="Maximum silica in ppm"
    )
    chloride_max_ppm: float = Field(
        default=250, gt=0, description="Maximum chloride in ppm"
    )
    sulfate_max_ppm: float = Field(
        default=250, gt=0, description="Maximum sulfate in ppm"
    )

    # Biological parameters
    total_bacteria_max_cfu_ml: float = Field(
        default=10000, gt=0, description="Maximum bacteria count CFU/mL"
    )
    legionella_max_cfu_l: float = Field(
        default=1000, gt=0, description="Maximum Legionella CFU/L"
    )

    # Corrosion indicators
    chlorine_residual_min_ppm: float = Field(
        default=0.2, ge=0, description="Minimum free chlorine residual"
    )
    chlorine_residual_max_ppm: float = Field(
        default=1.0, ge=0, description="Maximum free chlorine residual"
    )
    langelier_saturation_index_min: float = Field(
        default=-1.0, description="Minimum LSI (negative = corrosive)"
    )
    langelier_saturation_index_max: float = Field(
        default=1.0, description="Maximum LSI (positive = scaling)"
    )

    @field_validator("ph_max")
    @classmethod
    def validate_ph_range(cls, v: float, info) -> float:
        """Validate pH max is greater than pH min."""
        if "ph_min" in info.data and v <= info.data["ph_min"]:
            raise ValueError("ph_max must be greater than ph_min")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "limit_id": "WQ-001",
                "ph_min": 6.5,
                "ph_max": 9.0,
                "total_dissolved_solids_max_ppm": 2000,
                "calcium_hardness_max_ppm": 500,
            }
        }


# ============================================================================
# PERFORMANCE TARGETS
# ============================================================================


class PerformanceTargets(BaseModel):
    """
    Condenser performance targets for optimization.

    Defines target values for key performance indicators.
    """

    target_id: str = Field(..., description="Target set identifier")
    description: str = Field(
        default="Standard performance targets",
        description="Target set description"
    )

    # Heat transfer performance
    target_cleanliness_factor_pct: float = Field(
        default=85.0, gt=0, le=100, description="Target cleanliness factor"
    )
    min_cleanliness_factor_pct: float = Field(
        default=70.0, gt=0, le=100, description="Minimum acceptable cleanliness factor"
    )
    target_u_value_ratio: float = Field(
        default=0.90, gt=0, le=1.0, description="Target U-value as ratio of design"
    )

    # Vacuum performance
    target_ttd_f: float = Field(
        default=7.0, gt=0, le=20, description="Target terminal temperature difference"
    )
    max_ttd_f: float = Field(
        default=15.0, gt=0, le=30, description="Maximum acceptable TTD"
    )
    target_vacuum_deviation_in_hg: float = Field(
        default=0.5, gt=0, description="Target vacuum deviation from design"
    )
    max_vacuum_deviation_in_hg: float = Field(
        default=1.5, gt=0, description="Maximum vacuum deviation before alarm"
    )

    # Air removal
    target_air_inleakage_scfm: float = Field(
        default=5.0, gt=0, description="Target air inleakage rate"
    )
    max_air_inleakage_scfm: float = Field(
        default=15.0, gt=0, description="Maximum air inleakage before alarm"
    )

    # Efficiency targets
    target_heat_rate_penalty_btu_kwh: float = Field(
        default=0, ge=0, description="Target heat rate penalty"
    )
    max_heat_rate_penalty_btu_kwh: float = Field(
        default=100, ge=0, description="Maximum heat rate penalty"
    )
    target_auxiliary_power_reduction_pct: float = Field(
        default=5.0, ge=0, description="Target auxiliary power reduction"
    )

    # Availability
    target_availability_pct: float = Field(
        default=99.0, gt=0, le=100, description="Target availability percentage"
    )
    min_availability_pct: float = Field(
        default=95.0, gt=0, le=100, description="Minimum acceptable availability"
    )

    @field_validator("min_cleanliness_factor_pct")
    @classmethod
    def validate_cleanliness_range(cls, v: float, info) -> float:
        """Validate minimum cleanliness is less than target."""
        if "target_cleanliness_factor_pct" in info.data:
            if v >= info.data["target_cleanliness_factor_pct"]:
                raise ValueError(
                    "min_cleanliness_factor_pct must be less than target"
                )
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "target_id": "PERF-001",
                "target_cleanliness_factor_pct": 85,
                "target_ttd_f": 7.0,
                "target_air_inleakage_scfm": 5.0,
            }
        }


# ============================================================================
# ALERT THRESHOLDS
# ============================================================================


class AlertThresholds(BaseModel):
    """
    Alert and alarm thresholds for condenser monitoring.

    Defines warning and critical thresholds for all monitored parameters.
    """

    threshold_id: str = Field(..., description="Threshold set identifier")
    description: str = Field(
        default="Standard alert thresholds",
        description="Threshold set description"
    )

    # Vacuum alerts (in Hg deviation from expected)
    vacuum_warning_deviation_in_hg: float = Field(
        default=0.5, gt=0, description="Vacuum warning threshold"
    )
    vacuum_critical_deviation_in_hg: float = Field(
        default=1.0, gt=0, description="Vacuum critical threshold"
    )

    # Temperature alerts
    ttd_warning_f: float = Field(
        default=12.0, gt=0, description="TTD warning threshold"
    )
    ttd_critical_f: float = Field(
        default=18.0, gt=0, description="TTD critical threshold"
    )
    cooling_water_temp_warning_f: float = Field(
        default=90.0, gt=32, description="Cooling water temperature warning"
    )
    cooling_water_temp_critical_f: float = Field(
        default=95.0, gt=32, description="Cooling water temperature critical"
    )

    # Air inleakage alerts
    air_inleakage_warning_scfm: float = Field(
        default=10.0, gt=0, description="Air inleakage warning threshold"
    )
    air_inleakage_critical_scfm: float = Field(
        default=20.0, gt=0, description="Air inleakage critical threshold"
    )

    # Fouling alerts (cleanliness factor percentage)
    fouling_warning_pct: float = Field(
        default=75.0, gt=0, le=100, description="Fouling warning (cleanliness factor)"
    )
    fouling_critical_pct: float = Field(
        default=60.0, gt=0, le=100, description="Fouling critical (cleanliness factor)"
    )

    # Heat rate penalty alerts
    heat_rate_penalty_warning_btu_kwh: float = Field(
        default=50, ge=0, description="Heat rate penalty warning"
    )
    heat_rate_penalty_critical_btu_kwh: float = Field(
        default=100, ge=0, description="Heat rate penalty critical"
    )

    # Flow alerts (percentage of design)
    low_flow_warning_pct: float = Field(
        default=80.0, gt=0, le=100, description="Low flow warning percentage"
    )
    low_flow_critical_pct: float = Field(
        default=60.0, gt=0, le=100, description="Low flow critical percentage"
    )

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "AlertThresholds":
        """Validate warning thresholds are less severe than critical."""
        if self.vacuum_warning_deviation_in_hg >= self.vacuum_critical_deviation_in_hg:
            raise ValueError(
                "vacuum_warning must be less than vacuum_critical"
            )
        if self.air_inleakage_warning_scfm >= self.air_inleakage_critical_scfm:
            raise ValueError(
                "air_inleakage_warning must be less than critical"
            )
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "threshold_id": "ALERT-001",
                "vacuum_warning_deviation_in_hg": 0.5,
                "vacuum_critical_deviation_in_hg": 1.0,
                "ttd_warning_f": 12.0,
                "ttd_critical_f": 18.0,
            }
        }


# ============================================================================
# SCADA INTEGRATION
# ============================================================================


class AnalyzerConfiguration(BaseModel):
    """Analyzer configuration for SCADA integration."""

    analyzer_id: str = Field(..., description="Unique analyzer identifier")
    analyzer_type: AnalyzerType = Field(..., description="Analyzer type")
    measurement_parameter: str = Field(..., description="Measured parameter")
    measurement_units: str = Field(..., description="Measurement units")
    scada_tag: str = Field(..., description="SCADA tag name")
    measurement_range_min: float = Field(..., description="Measurement range minimum")
    measurement_range_max: float = Field(..., description="Measurement range maximum")
    accuracy_pct: float = Field(
        default=1.0, gt=0, le=10, description="Analyzer accuracy in %"
    )
    sampling_interval_seconds: int = Field(
        default=10, gt=0, description="Sampling interval in seconds"
    )
    location: str = Field(..., description="Analyzer installation location")
    is_online: bool = Field(default=True, description="Analyzer online status")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "analyzer_id": "TEMP-CW-IN",
                "analyzer_type": "temperature",
                "measurement_parameter": "Cooling Water Inlet Temperature",
                "measurement_units": "F",
                "scada_tag": "COND.CW.INLET_TEMP",
                "measurement_range_min": 32.0,
                "measurement_range_max": 150.0,
                "location": "CW Inlet Header",
            }
        }


class SCADAIntegration(BaseModel):
    """SCADA system integration configuration."""

    scada_system_name: str = Field(..., description="SCADA system name")
    protocol: str = Field(
        default="OPC-UA", description="Communication protocol"
    )
    server_address: str = Field(..., description="SCADA server address")
    server_port: int = Field(default=4840, gt=0, le=65535, description="Server port")
    polling_interval_seconds: int = Field(
        default=5, gt=0, description="Polling interval in seconds"
    )
    timeout_seconds: int = Field(
        default=30, gt=0, description="Communication timeout"
    )
    authentication_required: bool = Field(
        default=True, description="Authentication required"
    )
    enable_ssl: bool = Field(default=True, description="Enable SSL/TLS")

    # Analyzers
    analyzers: List[AnalyzerConfiguration] = Field(
        default_factory=list, description="Analyzer configurations"
    )

    # SCADA tags for control
    vacuum_control_tag: Optional[str] = Field(
        None, description="Vacuum control setpoint tag"
    )
    cw_flow_control_tag: Optional[str] = Field(
        None, description="Cooling water flow control tag"
    )
    pump_control_tags: List[str] = Field(
        default_factory=list, description="Circulating water pump control tags"
    )

    def get_analyzer(self, analyzer_id: str) -> Optional[AnalyzerConfiguration]:
        """Get analyzer by ID."""
        for analyzer in self.analyzers:
            if analyzer.analyzer_id == analyzer_id:
                return analyzer
        return None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "scada_system_name": "Plant SCADA",
                "protocol": "OPC-UA",
                "server_address": "192.168.1.100",
                "server_port": 4840,
            }
        }


# ============================================================================
# COOLING TOWER INTEGRATION
# ============================================================================


class CoolingTowerIntegration(BaseModel):
    """
    Cooling tower integration configuration.

    Defines interface with cooling tower control systems.
    """

    tower_id: str = Field(..., description="Cooling tower identifier")
    enabled: bool = Field(default=True, description="Integration enabled")

    # Tower specifications
    number_of_cells: int = Field(
        default=4, ge=1, description="Number of tower cells"
    )
    design_wet_bulb_f: float = Field(
        default=78.0, gt=32, description="Design wet bulb temperature"
    )
    design_approach_f: float = Field(
        default=10.0, gt=0, description="Design approach temperature"
    )
    design_range_f: float = Field(
        default=20.0, gt=0, description="Design range (delta T)"
    )

    # Fan configuration
    fan_type: str = Field(default="induced_draft", description="Fan type")
    fans_per_cell: int = Field(default=1, ge=1, description="Fans per cell")
    fan_power_hp: float = Field(default=200.0, gt=0, description="Fan power in HP")
    vfd_enabled: bool = Field(default=True, description="VFD enabled on fans")

    # Control interface
    scada_integration: bool = Field(
        default=True, description="SCADA integration enabled"
    )
    fan_control_tags: List[str] = Field(
        default_factory=list, description="Fan control SCADA tags"
    )
    basin_temp_tag: Optional[str] = Field(
        None, description="Basin temperature SCADA tag"
    )
    wet_bulb_tag: Optional[str] = Field(
        None, description="Wet bulb temperature SCADA tag"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "tower_id": "CT-001",
                "number_of_cells": 4,
                "design_wet_bulb_f": 78,
                "design_approach_f": 10,
                "design_range_f": 20,
            }
        }


# ============================================================================
# TURBINE COORDINATION
# ============================================================================


class TurbineCoordination(BaseModel):
    """
    Turbine coordination configuration.

    Defines interface for turbine backpressure coordination.
    """

    turbine_id: str = Field(..., description="Turbine identifier")
    enabled: bool = Field(default=True, description="Coordination enabled")

    # Turbine specifications
    rated_output_mw: float = Field(
        ..., gt=0, description="Rated output in MW"
    )
    exhaust_annulus_area_sqft: float = Field(
        ..., gt=0, description="Exhaust annulus area in sq ft"
    )
    design_backpressure_in_hg_abs: float = Field(
        ..., gt=0, lt=10, description="Design backpressure"
    )

    # Heat rate sensitivity
    heat_rate_curve_coefficients: List[float] = Field(
        default_factory=lambda: [1.0, 0.0, 0.0],
        description="Heat rate correction polynomial coefficients"
    )
    backpressure_limit_in_hg_abs: float = Field(
        default=5.0, gt=0, description="Maximum backpressure limit"
    )

    # Control interface
    backpressure_tag: Optional[str] = Field(
        None, description="Backpressure SCADA tag"
    )
    exhaust_temp_tag: Optional[str] = Field(
        None, description="Exhaust temperature SCADA tag"
    )
    load_tag: Optional[str] = Field(
        None, description="Turbine load SCADA tag"
    )

    def calculate_heat_rate_correction(self, actual_bp: float) -> float:
        """
        Calculate heat rate correction for actual backpressure.

        Args:
            actual_bp: Actual backpressure in Hg abs

        Returns:
            Heat rate correction in BTU/kWh
        """
        delta_bp = actual_bp - self.design_backpressure_in_hg_abs
        correction = 0.0
        for i, coeff in enumerate(self.heat_rate_curve_coefficients):
            correction += coeff * (delta_bp ** i)
        return correction

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "turbine_id": "ST-001",
                "rated_output_mw": 500,
                "exhaust_annulus_area_sqft": 150,
                "design_backpressure_in_hg_abs": 1.5,
            }
        }


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================


class AgentConfiguration(BaseModel):
    """
    Complete configuration for GL-017 CONDENSYNC Agent.

    This is the master configuration model that brings together all
    subsystem configurations for the Condenser Optimization Agent.
    """

    # Agent identification
    agent_id: str = Field(default="GL-017", description="Agent identifier")
    agent_name: str = Field(
        default="CONDENSYNC", description="Agent display name"
    )
    version: str = Field(default="1.0.0", description="Agent version")

    # Condenser configuration
    condensers: List[CondenserConfiguration] = Field(
        ..., min_length=1, description="List of condenser configurations"
    )

    # Cooling water system
    cooling_water_config: CoolingWaterConfig = Field(
        ..., description="Cooling water system configuration"
    )

    # Vacuum system
    vacuum_system_config: VacuumSystemConfig = Field(
        ..., description="Vacuum system configuration"
    )

    # Tube configuration
    tube_configuration: Optional[TubeConfiguration] = Field(
        None, description="Tube configuration for heat transfer calculations"
    )

    # Water quality limits
    water_quality_limits: WaterQualityLimits = Field(
        default_factory=lambda: WaterQualityLimits(limit_id="DEFAULT"),
        description="Cooling water quality limits"
    )

    # Performance targets
    performance_targets: PerformanceTargets = Field(
        default_factory=lambda: PerformanceTargets(target_id="DEFAULT"),
        description="Performance targets"
    )

    # Alert thresholds
    alert_thresholds: AlertThresholds = Field(
        default_factory=lambda: AlertThresholds(threshold_id="DEFAULT"),
        description="Alert thresholds"
    )

    # Integration configurations
    scada_integration: SCADAIntegration = Field(
        ..., description="SCADA system integration"
    )
    cooling_tower_integration: Optional[CoolingTowerIntegration] = Field(
        None, description="Cooling tower integration"
    )
    turbine_coordination: Optional[TurbineCoordination] = Field(
        None, description="Turbine coordination"
    )

    # Agent operational settings
    monitoring_interval_seconds: int = Field(
        default=30, gt=0, description="Monitoring cycle interval"
    )
    optimization_interval_seconds: int = Field(
        default=300, gt=0, description="Optimization cycle interval"
    )
    alert_enabled: bool = Field(
        default=True, description="Enable alerting"
    )
    auto_optimization_enabled: bool = Field(
        default=False, description="Enable automatic optimization control"
    )

    # Logging and data retention
    log_level: str = Field(default="INFO", description="Logging level")
    data_retention_days: int = Field(
        default=365, gt=0, description="Data retention period"
    )
    enable_provenance_tracking: bool = Field(
        default=True, description="Enable data provenance tracking"
    )

    def get_condenser(self, condenser_id: str) -> Optional[CondenserConfiguration]:
        """Get condenser configuration by ID."""
        for condenser in self.condensers:
            if condenser.condenser_id == condenser_id:
                return condenser
        return None

    @model_validator(mode="after")
    def validate_configurations(self) -> "AgentConfiguration":
        """Validate configuration consistency."""
        # Validate cooling water flow is within condenser limits
        for condenser in self.condensers:
            if self.cooling_water_config.design_flow_gpm > condenser.max_cooling_water_flow_gpm:
                logger.warning(
                    f"Cooling water design flow exceeds condenser {condenser.condenser_id} maximum"
                )
        return self

    @classmethod
    def create_default_configuration(
        cls,
        condenser_id: str = "COND-001",
        turbine_capacity_mw: float = 500.0,
    ) -> "AgentConfiguration":
        """
        Factory method to create default configuration.

        Args:
            condenser_id: Condenser identifier
            turbine_capacity_mw: Turbine capacity in MW

        Returns:
            Default AgentConfiguration
        """
        condenser = CondenserConfiguration.create_typical_utility_condenser(
            condenser_id, turbine_capacity_mw
        )

        return cls(
            condensers=[condenser],
            cooling_water_config=CoolingWaterConfig(
                system_id="CW-001",
                system_type=CoolingSystemType.OPEN_RECIRCULATING,
                design_flow_gpm=turbine_capacity_mw * 300,
                design_inlet_temp_f=75.0,
                design_outlet_temp_f=95.0,
            ),
            vacuum_system_config=VacuumSystemConfig(
                system_id="VAC-001",
                primary_pump_type=VacuumPumpType.STEAM_JET_EJECTOR,
                design_air_removal_scfm=100.0,
                design_suction_pressure_in_hg_abs=1.0,
                motive_steam_pressure_psig=150.0,
            ),
            scada_integration=SCADAIntegration(
                scada_system_name="Plant SCADA",
                server_address="localhost",
            ),
        )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "agent_id": "GL-017",
                "agent_name": "CONDENSYNC",
                "version": "1.0.0",
                "monitoring_interval_seconds": 30,
                "optimization_interval_seconds": 300,
            }
        }
