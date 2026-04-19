# -*- coding: utf-8 -*-
"""
GreenLang Buildings MRV Base Agent
===================================

Base class for all building sector MRV (Measurement, Reporting, Verification) agents.
Provides common functionality for building emissions calculation, CBAM compliance,
energy benchmarking, and provenance tracking.

Design Principles:
    - Zero-hallucination: All calculations are deterministic
    - Building-sector specific: Commercial, residential, industrial buildings
    - Multi-standard support: ASHRAE, LEED, BREEAM, Energy Star
    - Auditable: SHA-256 provenance tracking for complete audit trails

Standards References:
    - ASHRAE Standard 90.1 - Energy Standard for Buildings
    - Energy Star Portfolio Manager Technical Reference
    - GHG Protocol Scope 1, 2, and 3 Guidance
    - EU EPBD (Energy Performance of Buildings Directive)

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic, Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

InputT = TypeVar("InputT", bound="BuildingMRVInput")
OutputT = TypeVar("OutputT", bound="BuildingMRVOutput")


# =============================================================================
# ENUMS
# =============================================================================

class BuildingType(str, Enum):
    """Building type classification."""
    COMMERCIAL_OFFICE = "commercial_office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    EDUCATION = "education"
    DATA_CENTER = "data_center"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    RESIDENTIAL_SINGLE = "residential_single"
    RESIDENTIAL_MULTI = "residential_multi"
    MIXED_USE = "mixed_use"
    LABORATORY = "laboratory"
    RESTAURANT = "restaurant"
    SUPERMARKET = "supermarket"


class ClimateZone(str, Enum):
    """ASHRAE Climate Zone classification."""
    ZONE_1A = "1A"  # Very Hot - Humid
    ZONE_1B = "1B"  # Very Hot - Dry
    ZONE_2A = "2A"  # Hot - Humid
    ZONE_2B = "2B"  # Hot - Dry
    ZONE_3A = "3A"  # Warm - Humid
    ZONE_3B = "3B"  # Warm - Dry
    ZONE_3C = "3C"  # Warm - Marine
    ZONE_4A = "4A"  # Mixed - Humid
    ZONE_4B = "4B"  # Mixed - Dry
    ZONE_4C = "4C"  # Mixed - Marine
    ZONE_5A = "5A"  # Cool - Humid
    ZONE_5B = "5B"  # Cool - Dry
    ZONE_5C = "5C"  # Cool - Marine
    ZONE_6A = "6A"  # Cold - Humid
    ZONE_6B = "6B"  # Cold - Dry
    ZONE_7 = "7"    # Very Cold
    ZONE_8 = "8"    # Subarctic


class EnergySource(str, Enum):
    """Building energy source types."""
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    PROPANE = "propane"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    SOLAR_PV = "solar_pv"
    SOLAR_THERMAL = "solar_thermal"
    BIOMASS = "biomass"
    GEOTHERMAL = "geothermal"


class EndUseCategory(str, Enum):
    """Building energy end-use categories."""
    HVAC = "hvac"
    LIGHTING = "lighting"
    PLUG_LOADS = "plug_loads"
    HOT_WATER = "hot_water"
    REFRIGERATION = "refrigeration"
    COOKING = "cooking"
    ELEVATORS = "elevators"
    DATA_CENTER_IT = "data_center_it"
    PROCESS_LOADS = "process_loads"
    EXTERIOR_LIGHTING = "exterior_lighting"
    OTHER = "other"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes for buildings."""
    SCOPE_1 = "scope_1"  # On-site combustion
    SCOPE_2 = "scope_2"  # Purchased electricity/heat
    SCOPE_3 = "scope_3"  # Embodied carbon, tenant activities


class VerificationStatus(str, Enum):
    """MRV verification status."""
    UNVERIFIED = "unverified"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    REJECTED = "rejected"


class DataQuality(str, Enum):
    """Data quality level per GHG Protocol."""
    METERED = "metered"  # Direct meter readings
    UTILITY_BILLS = "utility_bills"  # Utility bill data
    ESTIMATED = "estimated"  # Engineering estimates
    BENCHMARKED = "benchmarked"  # Statistical benchmarks


class CertificationStandard(str, Enum):
    """Building certification standards."""
    ENERGY_STAR = "energy_star"
    LEED = "leed"
    BREEAM = "breeam"
    WELL = "well"
    PASSIVE_HOUSE = "passive_house"
    NET_ZERO_CARBON = "net_zero_carbon"
    NABERS = "nabers"
    GREEN_MARK = "green_mark"


# =============================================================================
# DATA MODELS
# =============================================================================

class EnergyConsumption(BaseModel):
    """Energy consumption data for a specific source."""
    source: EnergySource
    consumption: Decimal = Field(..., ge=0, description="Consumption value")
    unit: str = Field(..., description="Unit (kWh, therms, gallons, etc.)")
    end_use: Optional[EndUseCategory] = None
    metered: bool = Field(default=False, description="Whether data is metered")
    billing_period_start: Optional[str] = None
    billing_period_end: Optional[str] = None


class EmissionFactor(BaseModel):
    """Emission factor with provenance."""
    factor_id: str = Field(..., description="Unique factor identifier")
    value: Decimal = Field(..., description="Factor value")
    unit: str = Field(..., description="Unit (e.g., kgCO2e/kWh)")
    source: str = Field(..., description="Authoritative source")
    region: str = Field(default="global", description="Geographic applicability")
    valid_from: str = Field(..., description="Validity start date (ISO)")
    valid_to: Optional[str] = Field(None, description="Validity end date (ISO)")
    uncertainty_percent: Optional[float] = Field(None, ge=0, le=100)


class CalculationStep(BaseModel):
    """Individual calculation step for audit trail."""
    step_number: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    formula: str = Field(..., description="Mathematical formula used")
    inputs: Dict[str, str] = Field(default_factory=dict)
    output_value: Decimal
    output_unit: str
    source: str = Field(default="", description="Source reference")


class BuildingMetadata(BaseModel):
    """Building metadata for context."""
    building_id: str
    building_name: Optional[str] = None
    building_type: BuildingType
    gross_floor_area_sqm: Decimal = Field(..., gt=0, le=Decimal("10000000"))
    conditioned_area_sqm: Optional[Decimal] = Field(None, gt=0)
    year_built: Optional[int] = Field(None, ge=1800, le=2100)
    year_renovated: Optional[int] = Field(None, ge=1800, le=2100)
    num_floors: Optional[int] = Field(None, ge=1, le=200)
    occupancy_hours_per_week: Optional[Decimal] = Field(None, ge=0, le=168)
    num_occupants: Optional[int] = Field(None, ge=0)
    climate_zone: Optional[ClimateZone] = None
    postal_code: Optional[str] = None
    country_code: str = Field(default="US")


class EnergyUseIntensity(BaseModel):
    """Energy Use Intensity (EUI) metrics."""
    site_eui_kwh_per_sqm: Decimal
    source_eui_kwh_per_sqm: Decimal
    weather_normalized_eui_kwh_per_sqm: Optional[Decimal] = None
    benchmark_median_eui: Optional[Decimal] = None
    percentile_rank: Optional[int] = Field(None, ge=0, le=100)


class CarbonIntensity(BaseModel):
    """Carbon intensity metrics."""
    total_carbon_intensity_kgco2e_per_sqm: Decimal
    operational_carbon_kgco2e_per_sqm: Decimal
    scope1_carbon_intensity_kgco2e_per_sqm: Decimal
    scope2_carbon_intensity_kgco2e_per_sqm: Decimal


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class BuildingMRVInput(BaseModel):
    """Base input model for building MRV agents."""

    # Building identification
    building_id: str = Field(..., description="Unique building identifier")
    reporting_period: str = Field(..., description="Reporting period (e.g., 2024)")
    building_metadata: BuildingMetadata

    # Energy consumption data
    energy_consumption: List[EnergyConsumption] = Field(default_factory=list)

    # Optional grid emission factor override
    grid_emission_factor_kgco2e_per_kwh: Optional[Decimal] = Field(
        None, ge=0, le=Decimal("2.0")
    )

    # Data quality
    data_quality: DataQuality = Field(default=DataQuality.UTILITY_BILLS)

    class Config:
        """Pydantic config."""
        json_encoders = {Decimal: str}


class BuildingMRVOutput(BaseModel):
    """Base output model for building MRV agents."""

    # Identification
    calculation_id: str
    agent_id: str
    agent_version: str
    timestamp: str

    # Building summary
    building_id: str
    building_type: BuildingType
    reporting_period: str
    gross_floor_area_sqm: Decimal

    # Energy metrics
    total_energy_kwh: Decimal = Field(default=Decimal("0"))
    energy_by_source: Dict[str, Decimal] = Field(default_factory=dict)
    energy_by_end_use: Dict[str, Decimal] = Field(default_factory=dict)
    eui_metrics: Optional[EnergyUseIntensity] = None

    # Emissions by scope
    scope_1_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    scope_2_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    scope_3_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    total_emissions_kgco2e: Decimal = Field(default=Decimal("0"))

    # Carbon intensity
    carbon_intensity: Optional[CarbonIntensity] = None

    # Audit trail
    calculation_steps: List[CalculationStep] = Field(default_factory=list)
    emission_factors_used: List[EmissionFactor] = Field(default_factory=list)

    # Provenance
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    provenance_hash: str = Field(default="")

    # Quality
    data_quality: DataQuality = Field(default=DataQuality.UTILITY_BILLS)
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED
    )

    # Validation
    is_valid: bool = Field(default=True)
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic config."""
        json_encoders = {Decimal: str}


# =============================================================================
# BASE AGENT
# =============================================================================

class BuildingMRVBaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for building sector MRV agents.

    ZERO-HALLUCINATION GUARANTEE:
        - All calculations use deterministic formulas
        - Emission factors are database/config lookups only
        - Same inputs always produce identical outputs
        - Complete SHA-256 provenance tracking

    Attributes:
        AGENT_ID: Unique agent identifier (e.g., "GL-MRV-BLD-001")
        AGENT_VERSION: Semantic version string
        BUILDING_CATEGORY: Building category (commercial, residential, industrial)

    Example:
        >>> class CommercialBuildingMRV(BuildingMRVBaseAgent):
        ...     AGENT_ID = "GL-MRV-BLD-001"
        ...     BUILDING_CATEGORY = "commercial"
        ...
        ...     def calculate_emissions(self, input_data):
        ...         # Implementation
        ...         pass
    """

    # Class attributes - override in subclasses
    AGENT_ID: str = "GL-MRV-BLD-BASE"
    AGENT_VERSION: str = "1.0.0"
    BUILDING_CATEGORY: str = "building"

    # Precision settings
    PRECISION_ENERGY: int = 2
    PRECISION_EMISSIONS: int = 4
    PRECISION_INTENSITY: int = 4

    def __init__(self):
        """Initialize the building MRV agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._emission_factors: Dict[str, EmissionFactor] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize agent resources. Override in subclasses."""
        self._load_emission_factors()

    @abstractmethod
    def _load_emission_factors(self) -> None:
        """Load sector-specific emission factors. Must be implemented."""
        pass

    @abstractmethod
    def calculate_emissions(self, input_data: InputT) -> OutputT:
        """
        Calculate emissions for the building.

        Args:
            input_data: Validated input data

        Returns:
            Complete output with emissions and provenance
        """
        pass

    def process(self, input_data: InputT) -> OutputT:
        """
        Main processing method with full lifecycle management.

        Args:
            input_data: Input data for calculation

        Returns:
            Complete MRV output with provenance
        """
        start_time = datetime.now(timezone.utc)

        try:
            self.logger.info(
                f"{self.AGENT_ID} processing: building={input_data.building_id}, "
                f"period={input_data.reporting_period}"
            )

            # Calculate emissions
            output = self.calculate_emissions(input_data)

            # Calculate provenance hashes
            output.input_hash = self._calculate_hash(input_data.model_dump())
            output.output_hash = self._calculate_hash({
                "total_emissions": str(output.total_emissions_kgco2e),
                "total_energy": str(output.total_energy_kwh)
            })
            output.provenance_hash = self._calculate_provenance_hash(
                output.input_hash,
                output.output_hash,
                output.calculation_steps,
                output.emission_factors_used
            )

            # Log completion
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.info(
                f"{self.AGENT_ID} completed in {duration_ms:.2f}ms: "
                f"emissions={output.total_emissions_kgco2e} kgCO2e"
            )

            return output

        except Exception as e:
            self.logger.error(f"{self.AGENT_ID} failed: {str(e)}", exc_info=True)
            raise

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Any) -> Decimal:
        """Convert value to Decimal for precise calculations."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _round_energy(self, value: Decimal) -> Decimal:
        """Round energy values to standard precision."""
        quantize_str = "0." + "0" * self.PRECISION_ENERGY
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _round_emissions(self, value: Decimal) -> Decimal:
        """Round emission values to regulatory precision."""
        quantize_str = "0." + "0" * self.PRECISION_EMISSIONS
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _round_intensity(self, value: Decimal) -> Decimal:
        """Round intensity values to regulatory precision."""
        quantize_str = "0." + "0" * self.PRECISION_INTENSITY
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _generate_calculation_id(self, building_id: str, period: str) -> str:
        """Generate unique calculation ID."""
        data = f"{self.AGENT_ID}:{building_id}:{period}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance."""
        def convert(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj

        converted = convert(data)
        json_str = json.dumps(converted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _calculate_provenance_hash(
        self,
        input_hash: str,
        output_hash: str,
        steps: List[CalculationStep],
        factors: List[EmissionFactor]
    ) -> str:
        """Calculate comprehensive provenance hash."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.AGENT_VERSION,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "steps_count": len(steps),
            "factors_count": len(factors)
        }
        return self._calculate_hash(provenance_data)

    def _convert_to_kwh(
        self,
        value: Decimal,
        unit: str,
        source: EnergySource
    ) -> Decimal:
        """Convert energy value to kWh equivalent."""
        conversion_factors = {
            "kwh": Decimal("1"),
            "mwh": Decimal("1000"),
            "therm": Decimal("29.3001"),
            "mmbtu": Decimal("293.071"),
            "gj": Decimal("277.778"),
            "therms": Decimal("29.3001"),
            "gallons_propane": Decimal("27.0"),
            "gallons_fuel_oil": Decimal("40.6"),
            "ccf": Decimal("29.3001"),  # 100 cubic feet natural gas
            "mcf": Decimal("293.01"),   # 1000 cubic feet natural gas
        }

        unit_lower = unit.lower()
        if unit_lower not in conversion_factors:
            raise ValueError(f"Unknown unit: {unit}")

        return value * conversion_factors[unit_lower]

    def _get_emission_factor(self, factor_id: str) -> EmissionFactor:
        """Get emission factor by ID - deterministic lookup."""
        if factor_id not in self._emission_factors:
            raise KeyError(f"Emission factor not found: {factor_id}")
        return self._emission_factors[factor_id]

    def _calculate_scope1_emissions(
        self,
        energy_consumption: List[EnergyConsumption]
    ) -> Decimal:
        """Calculate Scope 1 emissions from on-site combustion."""
        scope1_sources = {
            EnergySource.NATURAL_GAS,
            EnergySource.FUEL_OIL,
            EnergySource.PROPANE,
            EnergySource.BIOMASS
        }

        total_emissions = Decimal("0")

        for consumption in energy_consumption:
            if consumption.source in scope1_sources:
                ef = self._get_scope1_factor(consumption.source, consumption.unit)
                emissions = consumption.consumption * ef.value
                total_emissions += emissions

        return self._round_emissions(total_emissions)

    def _get_scope1_factor(
        self,
        source: EnergySource,
        unit: str
    ) -> EmissionFactor:
        """Get Scope 1 emission factor for fuel."""
        factor_id = f"scope1_{source.value}_{unit.lower()}"
        return self._get_emission_factor(factor_id)

    def _calculate_eui(
        self,
        total_energy_kwh: Decimal,
        floor_area_sqm: Decimal,
        source_to_site_ratio: Decimal = Decimal("1.0")
    ) -> EnergyUseIntensity:
        """Calculate Energy Use Intensity metrics."""
        site_eui = total_energy_kwh / floor_area_sqm if floor_area_sqm > 0 else Decimal("0")
        source_eui = site_eui * source_to_site_ratio

        return EnergyUseIntensity(
            site_eui_kwh_per_sqm=self._round_intensity(site_eui),
            source_eui_kwh_per_sqm=self._round_intensity(source_eui)
        )

    def _calculate_carbon_intensity(
        self,
        scope1: Decimal,
        scope2: Decimal,
        floor_area_sqm: Decimal
    ) -> CarbonIntensity:
        """Calculate carbon intensity metrics."""
        total_operational = scope1 + scope2

        scope1_intensity = scope1 / floor_area_sqm if floor_area_sqm > 0 else Decimal("0")
        scope2_intensity = scope2 / floor_area_sqm if floor_area_sqm > 0 else Decimal("0")
        total_intensity = total_operational / floor_area_sqm if floor_area_sqm > 0 else Decimal("0")

        return CarbonIntensity(
            total_carbon_intensity_kgco2e_per_sqm=self._round_intensity(total_intensity),
            operational_carbon_kgco2e_per_sqm=self._round_intensity(total_intensity),
            scope1_carbon_intensity_kgco2e_per_sqm=self._round_intensity(scope1_intensity),
            scope2_carbon_intensity_kgco2e_per_sqm=self._round_intensity(scope2_intensity)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.AGENT_ID}, version={self.AGENT_VERSION})"


# =============================================================================
# COMMON EMISSION FACTORS FOR BUILDINGS
# =============================================================================

# Natural gas emission factors
NATURAL_GAS_EF_KGCO2E_PER_THERM = Decimal("5.302")  # EPA
NATURAL_GAS_EF_KGCO2E_PER_KWH = Decimal("0.181")

# Fuel oil emission factors
FUEL_OIL_EF_KGCO2E_PER_GALLON = Decimal("10.21")  # EPA - No. 2 fuel oil
FUEL_OIL_EF_KGCO2E_PER_KWH = Decimal("0.251")

# Propane emission factors
PROPANE_EF_KGCO2E_PER_GALLON = Decimal("5.72")  # EPA
PROPANE_EF_KGCO2E_PER_KWH = Decimal("0.212")

# Grid emission factors by region (kgCO2e/kWh)
GRID_EF_BY_REGION_KGCO2E_PER_KWH = {
    "us_average": Decimal("0.379"),
    "us_northeast": Decimal("0.255"),
    "us_southeast": Decimal("0.402"),
    "us_midwest": Decimal("0.477"),
    "us_southwest": Decimal("0.358"),
    "us_west": Decimal("0.282"),
    "us_california": Decimal("0.206"),
    "eu_average": Decimal("0.251"),
    "uk": Decimal("0.207"),
    "germany": Decimal("0.350"),
    "france": Decimal("0.052"),
    "china": Decimal("0.555"),
    "india": Decimal("0.708"),
    "japan": Decimal("0.457"),
    "australia": Decimal("0.656"),
    "canada": Decimal("0.120"),
    "world_average": Decimal("0.436"),
}

# Source-to-site energy ratios
SOURCE_TO_SITE_RATIO = {
    EnergySource.ELECTRICITY: Decimal("2.80"),  # US average
    EnergySource.NATURAL_GAS: Decimal("1.05"),
    EnergySource.FUEL_OIL: Decimal("1.01"),
    EnergySource.PROPANE: Decimal("1.01"),
    EnergySource.DISTRICT_HEATING: Decimal("1.20"),
    EnergySource.DISTRICT_COOLING: Decimal("1.04"),
}

# Building type benchmark EUI (kWh/sqm/year) - US Energy Star medians
BENCHMARK_EUI_BY_TYPE = {
    BuildingType.COMMERCIAL_OFFICE: Decimal("178"),
    BuildingType.RETAIL: Decimal("206"),
    BuildingType.HOTEL: Decimal("295"),
    BuildingType.HOSPITAL: Decimal("498"),
    BuildingType.EDUCATION: Decimal("167"),
    BuildingType.DATA_CENTER: Decimal("1500"),
    BuildingType.WAREHOUSE: Decimal("73"),
    BuildingType.INDUSTRIAL: Decimal("221"),
    BuildingType.RESIDENTIAL_SINGLE: Decimal("130"),
    BuildingType.RESIDENTIAL_MULTI: Decimal("156"),
    BuildingType.MIXED_USE: Decimal("200"),
    BuildingType.LABORATORY: Decimal("450"),
    BuildingType.RESTAURANT: Decimal("556"),
    BuildingType.SUPERMARKET: Decimal("495"),
}
