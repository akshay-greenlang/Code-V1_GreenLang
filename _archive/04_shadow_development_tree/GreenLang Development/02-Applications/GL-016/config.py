# -*- coding: utf-8 -*-
"""
GL-016 WATERGUARD Configuration Models.

This module provides comprehensive Pydantic configuration models for the
Boiler Water Treatment Agent, including boiler configurations, water quality
limits, chemical inventory, SCADA integration, and agent settings.

All models include validators to ensure compliance with ASME/ABMA guidelines
and industry best practices.

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class BoilerType(str, Enum):
    """Boiler type classification."""

    FIRETUBE = "firetube"
    WATERTUBE = "watertube"
    ONCE_THROUGH = "once_through"
    WASTE_HEAT = "waste_heat"


class WaterSourceType(str, Enum):
    """Water source classification."""

    MUNICIPAL = "municipal"
    WELL = "well"
    SURFACE = "surface"
    RECYCLED = "recycled"
    DEMINERALIZED = "demineralized"


class TreatmentProgramType(str, Enum):
    """Water treatment program classification."""

    PHOSPHATE = "phosphate"
    ALL_VOLATILE = "all_volatile"
    COORDINATED_PHOSPHATE = "coordinated_phosphate"
    OXYGENATED = "oxygenated"


class ChemicalType(str, Enum):
    """Chemical classification for water treatment."""

    PHOSPHATE = "phosphate"
    OXYGEN_SCAVENGER = "oxygen_scavenger"
    AMINE = "amine"
    BIOCIDE = "biocide"
    POLYMER = "polymer"
    CAUSTIC = "caustic"
    ACID = "acid"


class AnalyzerType(str, Enum):
    """Water analyzer type classification."""

    PH = "ph"
    CONDUCTIVITY = "conductivity"
    DISSOLVED_OXYGEN = "dissolved_oxygen"
    SILICA = "silica"
    HARDNESS = "hardness"
    PHOSPHATE = "phosphate"
    MULTIPARAMETER = "multiparameter"


class DosingSystemType(str, Enum):
    """Chemical dosing system type."""

    METERING_PUMP = "metering_pump"
    PROPORTIONAL_FEEDER = "proportional_feeder"
    INJECTION_QUILL = "injection_quill"
    SHOT_FEEDER = "shot_feeder"


# ============================================================================
# BOILER CONFIGURATION
# ============================================================================


class BoilerConfiguration(BaseModel):
    """
    Boiler configuration and operating parameters.

    Defines the boiler characteristics that influence water treatment
    requirements and strategies.
    """

    boiler_id: str = Field(..., description="Unique boiler identifier")
    boiler_type: BoilerType = Field(..., description="Boiler type classification")
    operating_pressure_psig: float = Field(
        ..., gt=0, le=3000, description="Operating pressure in psig"
    )
    operating_temperature_f: float = Field(
        ..., gt=212, le=700, description="Operating temperature in °F"
    )
    steam_capacity_lb_hr: float = Field(
        ..., gt=0, description="Steam generation capacity in lb/hr"
    )
    makeup_water_rate_gpm: float = Field(
        ..., gt=0, description="Makeup water feed rate in GPM"
    )
    condensate_return_pct: float = Field(
        default=80.0, ge=0, le=100, description="Condensate return percentage"
    )
    water_source: WaterSourceType = Field(
        ..., description="Primary water source type"
    )
    treatment_program: TreatmentProgramType = Field(
        ..., description="Water treatment program type"
    )
    blowdown_type: str = Field(
        default="continuous", description="Blowdown type (continuous/intermittent)"
    )
    design_cycles_of_concentration: float = Field(
        default=10.0, gt=1, le=100, description="Design cycles of concentration"
    )

    # Boiler geometry
    boiler_volume_gallons: Optional[float] = Field(
        None, gt=0, description="Total boiler water volume in gallons"
    )
    heating_surface_sqft: Optional[float] = Field(
        None, gt=0, description="Heating surface area in sq ft"
    )

    # Operating constraints
    max_tds_ppm: float = Field(
        default=3500, gt=0, description="Maximum allowable TDS in ppm"
    )
    max_silica_ppm: float = Field(
        default=150, gt=0, description="Maximum allowable silica in ppm"
    )
    target_ph: float = Field(
        default=10.5, ge=7.0, le=12.5, description="Target pH value"
    )

    # Metadata
    location: Optional[str] = Field(None, description="Boiler physical location")
    commissioning_date: Optional[datetime] = Field(
        None, description="Boiler commissioning date"
    )
    last_inspection_date: Optional[datetime] = Field(
        None, description="Last inspection date"
    )

    @field_validator("operating_pressure_psig")
    @classmethod
    def validate_pressure(cls, v: float, info) -> float:
        """Validate operating pressure is within safe limits."""
        if v > 2400:
            logger.warning(
                f"High pressure boiler (>{v} psig) requires stringent water quality"
            )
        return v

    @model_validator(mode="after")
    def validate_pressure_temperature_correlation(self) -> "BoilerConfiguration":
        """Validate pressure and temperature are correlated."""
        # Approximate saturation temperature at given pressure
        # Using simplified formula: T ≈ 212 + 0.17 * P (for P < 1000 psig)
        if self.operating_pressure_psig < 1000:
            expected_temp = 212 + 0.17 * self.operating_pressure_psig
            if abs(self.operating_temperature_f - expected_temp) > 50:
                logger.warning(
                    f"Temperature {self.operating_temperature_f}°F may not match "
                    f"pressure {self.operating_pressure_psig} psig"
                )
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "boiler_id": "BOILER-001",
                "boiler_type": "watertube",
                "operating_pressure_psig": 150,
                "operating_temperature_f": 366,
                "steam_capacity_lb_hr": 50000,
                "makeup_water_rate_gpm": 10,
                "condensate_return_pct": 85,
                "water_source": "demineralized",
                "treatment_program": "phosphate",
                "design_cycles_of_concentration": 15,
            }
        }


# ============================================================================
# WATER QUALITY LIMITS
# ============================================================================


class WaterQualityLimits(BaseModel):
    """
    Water quality limits based on ASME and ABMA guidelines.

    These limits are pressure-dependent and vary by boiler type and
    treatment program.
    """

    # Based on boiler pressure
    pressure_range_psig: tuple[float, float] = Field(
        ..., description="Pressure range for these limits (min, max)"
    )

    # pH limits
    ph_min: float = Field(default=10.5, ge=7.0, le=12.0, description="Minimum pH")
    ph_max: float = Field(default=11.5, ge=7.0, le=13.0, description="Maximum pH")

    # Conductivity limits (µS/cm)
    specific_conductance_max_us_cm: Optional[float] = Field(
        None, gt=0, description="Maximum specific conductance in µS/cm"
    )

    # Dissolved solids
    total_dissolved_solids_max_ppm: float = Field(
        default=3500, gt=0, description="Maximum TDS in ppm"
    )

    # Silica limits
    silica_max_ppm: float = Field(
        default=150, gt=0, description="Maximum silica in ppm"
    )

    # Hardness limits
    total_hardness_max_ppm: float = Field(
        default=0.3, ge=0, description="Maximum total hardness as CaCO3 in ppm"
    )

    # Alkalinity limits
    total_alkalinity_min_ppm: float = Field(
        default=200, ge=0, description="Minimum total alkalinity as CaCO3 in ppm"
    )
    total_alkalinity_max_ppm: float = Field(
        default=700, ge=0, description="Maximum total alkalinity as CaCO3 in ppm"
    )

    # Dissolved oxygen
    dissolved_oxygen_max_ppb: float = Field(
        default=7, ge=0, description="Maximum dissolved oxygen in ppb"
    )

    # Iron and copper
    iron_max_ppm: float = Field(
        default=0.1, ge=0, description="Maximum iron in ppm"
    )
    copper_max_ppm: float = Field(
        default=0.05, ge=0, description="Maximum copper in ppm"
    )

    # Chlorides and sulfates
    chloride_max_ppm: Optional[float] = Field(
        None, ge=0, description="Maximum chloride in ppm"
    )
    sulfate_max_ppm: Optional[float] = Field(
        None, ge=0, description="Maximum sulfate in ppm"
    )

    # Phosphate (for phosphate programs)
    phosphate_min_ppm: Optional[float] = Field(
        None, ge=0, description="Minimum phosphate as PO4 in ppm"
    )
    phosphate_max_ppm: Optional[float] = Field(
        None, ge=0, description="Maximum phosphate as PO4 in ppm"
    )

    # Suspended solids
    suspended_solids_max_ppm: Optional[float] = Field(
        None, ge=0, description="Maximum suspended solids in ppm"
    )

    @field_validator("ph_max")
    @classmethod
    def validate_ph_range(cls, v: float, info) -> float:
        """Validate pH max is greater than pH min."""
        if "ph_min" in info.data and v <= info.data["ph_min"]:
            raise ValueError("ph_max must be greater than ph_min")
        return v

    @field_validator("total_alkalinity_max_ppm")
    @classmethod
    def validate_alkalinity_range(cls, v: float, info) -> float:
        """Validate alkalinity max is greater than alkalinity min."""
        if "total_alkalinity_min_ppm" in info.data and v <= info.data["total_alkalinity_min_ppm"]:
            raise ValueError("total_alkalinity_max_ppm must be greater than total_alkalinity_min_ppm")
        return v

    @classmethod
    def from_pressure(cls, pressure_psig: float, treatment_program: TreatmentProgramType) -> "WaterQualityLimits":
        """
        Create water quality limits based on boiler pressure and treatment program.

        This factory method implements ASME and ABMA guidelines for different
        pressure ranges and treatment programs.

        Args:
            pressure_psig: Operating pressure in psig
            treatment_program: Water treatment program type

        Returns:
            WaterQualityLimits configured for the specified conditions
        """
        # Pressure ranges based on ASME standards
        if pressure_psig <= 300:
            return cls(
                pressure_range_psig=(0, 300),
                ph_min=10.5,
                ph_max=11.5,
                total_dissolved_solids_max_ppm=3500,
                silica_max_ppm=150,
                total_hardness_max_ppm=0.3,
                total_alkalinity_min_ppm=200,
                total_alkalinity_max_ppm=700,
                dissolved_oxygen_max_ppb=7,
                iron_max_ppm=0.1,
                copper_max_ppm=0.05,
                phosphate_min_ppm=20 if treatment_program == TreatmentProgramType.PHOSPHATE else None,
                phosphate_max_ppm=60 if treatment_program == TreatmentProgramType.PHOSPHATE else None,
            )
        elif pressure_psig <= 600:
            return cls(
                pressure_range_psig=(301, 600),
                ph_min=10.5,
                ph_max=11.5,
                total_dissolved_solids_max_ppm=3000,
                silica_max_ppm=90,
                total_hardness_max_ppm=0.2,
                total_alkalinity_min_ppm=200,
                total_alkalinity_max_ppm=600,
                dissolved_oxygen_max_ppb=7,
                iron_max_ppm=0.05,
                copper_max_ppm=0.03,
                phosphate_min_ppm=15 if treatment_program == TreatmentProgramType.PHOSPHATE else None,
                phosphate_max_ppm=50 if treatment_program == TreatmentProgramType.PHOSPHATE else None,
            )
        elif pressure_psig <= 1000:
            return cls(
                pressure_range_psig=(601, 1000),
                ph_min=10.0,
                ph_max=11.0,
                total_dissolved_solids_max_ppm=2500,
                silica_max_ppm=40,
                total_hardness_max_ppm=0.1,
                total_alkalinity_min_ppm=150,
                total_alkalinity_max_ppm=500,
                dissolved_oxygen_max_ppb=5,
                iron_max_ppm=0.03,
                copper_max_ppm=0.02,
                phosphate_min_ppm=10 if treatment_program == TreatmentProgramType.COORDINATED_PHOSPHATE else None,
                phosphate_max_ppm=30 if treatment_program == TreatmentProgramType.COORDINATED_PHOSPHATE else None,
            )
        else:  # > 1000 psig
            return cls(
                pressure_range_psig=(1001, 3000),
                ph_min=9.5,
                ph_max=10.0,
                total_dissolved_solids_max_ppm=1500,
                silica_max_ppm=20,
                total_hardness_max_ppm=0.05,
                total_alkalinity_min_ppm=100,
                total_alkalinity_max_ppm=400,
                dissolved_oxygen_max_ppb=5,
                iron_max_ppm=0.02,
                copper_max_ppm=0.01,
                phosphate_min_ppm=None,  # All-volatile treatment at high pressure
                phosphate_max_ppm=None,
            )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "pressure_range_psig": (0, 300),
                "ph_min": 10.5,
                "ph_max": 11.5,
                "total_dissolved_solids_max_ppm": 3500,
                "silica_max_ppm": 150,
                "total_hardness_max_ppm": 0.3,
            }
        }


# ============================================================================
# CHEMICAL INVENTORY
# ============================================================================


class ChemicalSpecification(BaseModel):
    """Chemical specification and inventory data."""

    chemical_id: str = Field(..., description="Unique chemical identifier")
    chemical_name: str = Field(..., description="Chemical name")
    chemical_type: ChemicalType = Field(..., description="Chemical type classification")
    concentration_pct: float = Field(
        ..., gt=0, le=100, description="Chemical concentration percentage"
    )
    density_lb_gal: float = Field(
        ..., gt=0, description="Chemical density in lb/gal"
    )
    current_inventory_gallons: float = Field(
        default=0.0, ge=0, description="Current inventory in gallons"
    )
    min_inventory_gallons: float = Field(
        default=50.0, ge=0, description="Minimum inventory threshold in gallons"
    )
    max_inventory_gallons: float = Field(
        default=500.0, ge=0, description="Maximum inventory capacity in gallons"
    )
    unit_cost_usd_gal: Optional[float] = Field(
        None, ge=0, description="Unit cost in USD per gallon"
    )
    supplier: Optional[str] = Field(None, description="Chemical supplier")
    msds_number: Optional[str] = Field(None, description="MSDS/SDS number")
    expiration_date: Optional[datetime] = Field(
        None, description="Chemical expiration date"
    )

    @field_validator("current_inventory_gallons")
    @classmethod
    def validate_inventory_within_limits(cls, v: float, info) -> float:
        """Validate current inventory is within min/max limits."""
        if "max_inventory_gallons" in info.data and v > info.data["max_inventory_gallons"]:
            raise ValueError("current_inventory_gallons cannot exceed max_inventory_gallons")
        return v

    def days_of_supply(self, usage_rate_gal_day: float) -> float:
        """
        Calculate days of supply remaining.

        Args:
            usage_rate_gal_day: Average daily usage rate in gallons/day

        Returns:
            Days of supply remaining
        """
        if usage_rate_gal_day <= 0:
            return float("inf")
        return self.current_inventory_gallons / usage_rate_gal_day

    def needs_reorder(self) -> bool:
        """Check if chemical needs reordering."""
        return self.current_inventory_gallons <= self.min_inventory_gallons

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "chemical_id": "CHEM-001",
                "chemical_name": "Trisodium Phosphate",
                "chemical_type": "phosphate",
                "concentration_pct": 30,
                "density_lb_gal": 10.2,
                "current_inventory_gallons": 150,
                "min_inventory_gallons": 50,
                "max_inventory_gallons": 500,
            }
        }


class ChemicalInventory(BaseModel):
    """Collection of chemical specifications and inventory."""

    chemicals: List[ChemicalSpecification] = Field(
        default_factory=list, description="List of chemicals in inventory"
    )

    def get_chemical(self, chemical_id: str) -> Optional[ChemicalSpecification]:
        """Get chemical by ID."""
        for chem in self.chemicals:
            if chem.chemical_id == chemical_id:
                return chem
        return None

    def get_chemicals_by_type(self, chemical_type: ChemicalType) -> List[ChemicalSpecification]:
        """Get all chemicals of a specific type."""
        return [chem for chem in self.chemicals if chem.chemical_type == chemical_type]

    def get_low_inventory_chemicals(self) -> List[ChemicalSpecification]:
        """Get all chemicals below minimum inventory."""
        return [chem for chem in self.chemicals if chem.needs_reorder()]


# ============================================================================
# SCADA INTEGRATION
# ============================================================================


class WaterAnalyzerConfiguration(BaseModel):
    """Water analyzer configuration and connectivity."""

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
        default=60, gt=0, description="Sampling interval in seconds"
    )
    calibration_interval_days: int = Field(
        default=30, gt=0, description="Calibration interval in days"
    )
    last_calibration_date: Optional[datetime] = Field(
        None, description="Last calibration date"
    )
    location: str = Field(..., description="Analyzer installation location")
    is_online: bool = Field(default=True, description="Analyzer online status")

    def is_calibration_due(self) -> bool:
        """Check if calibration is due."""
        if self.last_calibration_date is None:
            return True
        days_since_calibration = (datetime.now() - self.last_calibration_date).days
        return days_since_calibration >= self.calibration_interval_days

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "analyzer_id": "ANALYZER-PH-001",
                "analyzer_type": "ph",
                "measurement_parameter": "pH",
                "measurement_units": "pH",
                "scada_tag": "BOILER.FEEDWATER.PH",
                "measurement_range_min": 0.0,
                "measurement_range_max": 14.0,
                "location": "Feedwater line",
            }
        }


class ChemicalDosingSystemConfiguration(BaseModel):
    """Chemical dosing system configuration."""

    dosing_system_id: str = Field(..., description="Unique dosing system identifier")
    dosing_system_type: DosingSystemType = Field(..., description="Dosing system type")
    chemical_id: str = Field(..., description="Chemical being dosed")
    scada_control_tag: str = Field(..., description="SCADA control tag")
    scada_feedback_tag: str = Field(..., description="SCADA feedback tag")
    max_dosing_rate_gph: float = Field(
        ..., gt=0, description="Maximum dosing rate in GPH"
    )
    min_dosing_rate_gph: float = Field(
        default=0.0, ge=0, description="Minimum dosing rate in GPH"
    )
    current_dosing_rate_gph: float = Field(
        default=0.0, ge=0, description="Current dosing rate in GPH"
    )
    injection_point: str = Field(..., description="Chemical injection point")
    control_mode: str = Field(
        default="automatic", description="Control mode (automatic/manual)"
    )
    is_online: bool = Field(default=True, description="Dosing system online status")

    @field_validator("current_dosing_rate_gph")
    @classmethod
    def validate_dosing_rate(cls, v: float, info) -> float:
        """Validate current dosing rate is within limits."""
        if "max_dosing_rate_gph" in info.data and v > info.data["max_dosing_rate_gph"]:
            raise ValueError("current_dosing_rate_gph cannot exceed max_dosing_rate_gph")
        if "min_dosing_rate_gph" in info.data and v < info.data["min_dosing_rate_gph"]:
            raise ValueError("current_dosing_rate_gph cannot be less than min_dosing_rate_gph")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "dosing_system_id": "DOSER-PHOSPHATE-001",
                "dosing_system_type": "metering_pump",
                "chemical_id": "CHEM-001",
                "scada_control_tag": "BOILER.CHEMICAL.PHOSPHATE.SETPOINT",
                "scada_feedback_tag": "BOILER.CHEMICAL.PHOSPHATE.ACTUAL",
                "max_dosing_rate_gph": 5.0,
                "injection_point": "Feedwater line",
            }
        }


class SCADAIntegration(BaseModel):
    """SCADA system integration configuration."""

    scada_system_name: str = Field(..., description="SCADA system name")
    protocol: str = Field(
        default="OPC-UA", description="Communication protocol (OPC-UA, Modbus, etc.)"
    )
    server_address: str = Field(..., description="SCADA server address")
    server_port: int = Field(default=4840, gt=0, le=65535, description="Server port")
    polling_interval_seconds: int = Field(
        default=5, gt=0, description="Polling interval in seconds"
    )
    timeout_seconds: int = Field(
        default=30, gt=0, description="Communication timeout in seconds"
    )
    authentication_required: bool = Field(
        default=True, description="Authentication required flag"
    )
    username: Optional[str] = Field(None, description="SCADA username")
    enable_ssl: bool = Field(default=True, description="Enable SSL/TLS encryption")

    # Device configurations
    water_analyzers: List[WaterAnalyzerConfiguration] = Field(
        default_factory=list, description="Water analyzer configurations"
    )
    dosing_systems: List[ChemicalDosingSystemConfiguration] = Field(
        default_factory=list, description="Chemical dosing system configurations"
    )

    def get_analyzer(self, analyzer_id: str) -> Optional[WaterAnalyzerConfiguration]:
        """Get water analyzer by ID."""
        for analyzer in self.water_analyzers:
            if analyzer.analyzer_id == analyzer_id:
                return analyzer
        return None

    def get_dosing_system(self, dosing_system_id: str) -> Optional[ChemicalDosingSystemConfiguration]:
        """Get dosing system by ID."""
        for system in self.dosing_systems:
            if system.dosing_system_id == dosing_system_id:
                return system
        return None

    def get_calibration_due_analyzers(self) -> List[WaterAnalyzerConfiguration]:
        """Get all analyzers with calibration due."""
        return [analyzer for analyzer in self.water_analyzers if analyzer.is_calibration_due()]

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "scada_system_name": "Plant SCADA",
                "protocol": "OPC-UA",
                "server_address": "192.168.1.100",
                "server_port": 4840,
                "authentication_required": True,
            }
        }


# ============================================================================
# ERP INTEGRATION
# ============================================================================


class ERPIntegration(BaseModel):
    """ERP system integration configuration."""

    erp_system_name: str = Field(..., description="ERP system name (SAP, Oracle, etc.)")
    api_endpoint: str = Field(..., description="ERP API endpoint URL")
    api_version: str = Field(default="v1", description="API version")
    authentication_type: str = Field(
        default="oauth2", description="Authentication type (oauth2, api_key, etc.)"
    )
    enable_chemical_ordering: bool = Field(
        default=True, description="Enable automatic chemical ordering"
    )
    enable_cost_tracking: bool = Field(
        default=True, description="Enable chemical cost tracking"
    )
    enable_maintenance_scheduling: bool = Field(
        default=True, description="Enable maintenance work order creation"
    )
    auto_reorder_threshold_days: int = Field(
        default=14, gt=0, description="Days of supply to trigger auto reorder"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "erp_system_name": "SAP",
                "api_endpoint": "https://erp.company.com/api",
                "authentication_type": "oauth2",
                "enable_chemical_ordering": True,
            }
        }


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================


class AgentConfiguration(BaseModel):
    """
    Complete configuration for GL-016 WATERGUARD Agent.

    This is the master configuration model that brings together all
    subsystem configurations.
    """

    # Agent identification
    agent_id: str = Field(default="GL-016", description="Agent identifier")
    agent_name: str = Field(
        default="WATERGUARD", description="Agent display name"
    )
    version: str = Field(default="1.0.0", description="Agent version")

    # Boiler configuration
    boilers: List[BoilerConfiguration] = Field(
        ..., min_length=1, description="List of boiler configurations"
    )

    # Water quality limits
    water_quality_limits: Dict[str, WaterQualityLimits] = Field(
        default_factory=dict, description="Water quality limits by boiler ID"
    )

    # Chemical inventory
    chemical_inventory: ChemicalInventory = Field(
        default_factory=ChemicalInventory, description="Chemical inventory"
    )

    # Integration configurations
    scada_integration: SCADAIntegration = Field(
        ..., description="SCADA system integration"
    )
    erp_integration: Optional[ERPIntegration] = Field(
        None, description="ERP system integration"
    )

    # Agent operational settings
    monitoring_interval_seconds: int = Field(
        default=60, gt=0, description="Monitoring cycle interval in seconds"
    )
    alert_enabled: bool = Field(
        default=True, description="Enable alerting"
    )
    auto_dosing_enabled: bool = Field(
        default=False, description="Enable automatic chemical dosing"
    )

    # Thresholds and limits
    scale_risk_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Scale risk alert threshold (0-1)"
    )
    corrosion_risk_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Corrosion risk alert threshold (0-1)"
    )

    # Logging and data retention
    log_level: str = Field(default="INFO", description="Logging level")
    data_retention_days: int = Field(
        default=365, gt=0, description="Data retention period in days"
    )
    enable_provenance_tracking: bool = Field(
        default=True, description="Enable data provenance tracking"
    )

    @model_validator(mode="after")
    def validate_boiler_configurations(self) -> "AgentConfiguration":
        """Validate boiler configurations and create water quality limits."""
        # Create water quality limits for each boiler if not specified
        for boiler in self.boilers:
            if boiler.boiler_id not in self.water_quality_limits:
                self.water_quality_limits[boiler.boiler_id] = WaterQualityLimits.from_pressure(
                    boiler.operating_pressure_psig,
                    boiler.treatment_program
                )
        return self

    def get_boiler(self, boiler_id: str) -> Optional[BoilerConfiguration]:
        """Get boiler configuration by ID."""
        for boiler in self.boilers:
            if boiler.boiler_id == boiler_id:
                return boiler
        return None

    def get_water_quality_limits(self, boiler_id: str) -> Optional[WaterQualityLimits]:
        """Get water quality limits for a specific boiler."""
        return self.water_quality_limits.get(boiler_id)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "agent_id": "GL-016",
                "agent_name": "WATERGUARD",
                "version": "1.0.0",
                "monitoring_interval_seconds": 60,
                "auto_dosing_enabled": False,
                "scale_risk_threshold": 0.7,
                "corrosion_risk_threshold": 0.7,
            }
        }
