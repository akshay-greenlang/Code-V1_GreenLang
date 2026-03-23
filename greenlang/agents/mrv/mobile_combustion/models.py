# -*- coding: utf-8 -*-
"""
Mobile Combustion Agent Data Models - AGENT-MRV-003

Pydantic v2 data models for the Mobile Combustion Agent SDK covering
GHG Protocol Scope 1 mobile combustion calculations including:
- On-road vehicles (passenger cars, light/medium/heavy-duty trucks, buses,
  motorcycles, vans) with gasoline, diesel, hybrid, and PHEV variants
- Off-road equipment (construction, agricultural, industrial, mining, forklifts)
- Marine vessels (inland, coastal, ocean)
- Aviation sources (corporate jets, helicopters, turboprops)
- Rail (diesel locomotives)
- Fuel-based, distance-based, and spend-based calculation methods
- Biofuel blends (E10, E85, B5, B20, B100, SAF) with biogenic CO2 separation
- Tier 1/2/3 calculation methodologies per GHG Protocol Chapter 4
- Fleet-level vehicle registration and trip tracking
- Monte Carlo uncertainty quantification
- Multi-framework regulatory compliance mapping
- SHA-256 provenance chain for complete audit trails

Enumerations (15):
    - VehicleCategory, VehicleType, FuelType, EmissionGas, CalculationMethod,
      CalculationTier, EmissionFactorSource, GWPSource, DistanceUnit,
      FuelEconomyUnit, EmissionControlTechnology, VehicleStatus,
      TripStatus, ComplianceStatus, ReportingPeriod, UnitType

Data Models (14):
    - VehicleTypeInfo, FuelTypeInfo, EmissionFactorRecord, VehicleRegistration,
      TripRecord, CalculationInput, CalculationResult, BatchCalculationInput,
      FleetAggregation, UncertaintyResult, ComplianceCheckResult,
      MobileCombustionInput, MobileCombustionOutput, AuditEntry

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of calculations in a single batch request.
MAX_CALCULATIONS_PER_BATCH: int = 10_000

#: Maximum number of gas emission entries per calculation result.
MAX_GASES_PER_RESULT: int = 10

#: Maximum number of trace steps in a single calculation.
MAX_TRACE_STEPS: int = 200

#: Maximum number of vehicles in a single fleet registration batch.
MAX_VEHICLES_PER_REGISTRATION: int = 10_000

#: Maximum number of trips in a single trip batch submission.
MAX_TRIPS_PER_BATCH: int = 5_000

#: Standard CO2, CH4, N2O GWP values by Assessment Report edition.
GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR4": {"CO2": 1.0, "CH4": 25.0, "N2O": 298.0},
    "AR5": {"CO2": 1.0, "CH4": 28.0, "N2O": 265.0},
    "AR6": {"CO2": 1.0, "CH4": 27.3, "N2O": 273.0},
    "AR6_20YR": {"CO2": 1.0, "CH4": 81.2, "N2O": 273.0},
}

#: Biofuel blend fractions for biogenic CO2 separation.
#: Maps fuel type value to the fraction of biofuel content (0.0-1.0).
BIOFUEL_FRACTIONS: Dict[str, float] = {
    "ETHANOL_E10": 0.10,
    "ETHANOL_E85": 0.85,
    "BIODIESEL_B5": 0.05,
    "BIODIESEL_B20": 0.20,
    "BIODIESEL_B100": 1.00,
    "SUSTAINABLE_AVIATION_FUEL": 0.50,
}


# =============================================================================
# Enumerations (16)
# =============================================================================


class VehicleCategory(str, Enum):
    """Broad classification of mobile combustion sources by transport mode.

    ON_ROAD: Highway vehicles (passenger cars, trucks, buses, motorcycles,
        vans) operating on public roads.
    OFF_ROAD: Non-highway equipment (construction, agricultural, industrial,
        mining, forklifts) operating off public roads.
    MARINE: Waterborne vessels (inland, coastal, ocean) burning marine
        fuels for propulsion.
    AVIATION: Aircraft (corporate jets, helicopters, turboprops) burning
        aviation fuels.
    RAIL: Rail vehicles (diesel locomotives) for freight and passenger
        transport.
    """

    ON_ROAD = "ON_ROAD"
    OFF_ROAD = "OFF_ROAD"
    MARINE = "MARINE"
    AVIATION = "AVIATION"
    RAIL = "RAIL"


class VehicleType(str, Enum):
    """Specific vehicle type identifiers for mobile combustion sources.

    Covers all major on-road, off-road, marine, aviation, and rail
    vehicle types encountered in Scope 1 mobile combustion inventories.
    Each vehicle type has associated default emission factors, fuel
    economy values, and emission control technology assumptions.

    Naming follows EPA, GHG Protocol Chapter 4, and IPCC 2006 GL
    conventions for cross-framework compatibility.
    """

    # On-road: Passenger cars
    PASSENGER_CAR_GASOLINE = "PASSENGER_CAR_GASOLINE"
    PASSENGER_CAR_DIESEL = "PASSENGER_CAR_DIESEL"
    PASSENGER_CAR_HYBRID = "PASSENGER_CAR_HYBRID"
    PASSENGER_CAR_PHEV = "PASSENGER_CAR_PHEV"
    # On-road: Light-duty trucks
    LIGHT_DUTY_TRUCK_GASOLINE = "LIGHT_DUTY_TRUCK_GASOLINE"
    LIGHT_DUTY_TRUCK_DIESEL = "LIGHT_DUTY_TRUCK_DIESEL"
    # On-road: Medium-duty trucks
    MEDIUM_DUTY_TRUCK_GASOLINE = "MEDIUM_DUTY_TRUCK_GASOLINE"
    MEDIUM_DUTY_TRUCK_DIESEL = "MEDIUM_DUTY_TRUCK_DIESEL"
    # On-road: Heavy-duty trucks
    HEAVY_DUTY_TRUCK = "HEAVY_DUTY_TRUCK"
    # On-road: Buses
    BUS_DIESEL = "BUS_DIESEL"
    BUS_CNG = "BUS_CNG"
    # On-road: Other
    MOTORCYCLE = "MOTORCYCLE"
    VAN_LCV = "VAN_LCV"
    # Off-road equipment
    CONSTRUCTION_EQUIPMENT = "CONSTRUCTION_EQUIPMENT"
    AGRICULTURAL_EQUIPMENT = "AGRICULTURAL_EQUIPMENT"
    INDUSTRIAL_EQUIPMENT = "INDUSTRIAL_EQUIPMENT"
    MINING_EQUIPMENT = "MINING_EQUIPMENT"
    FORKLIFT = "FORKLIFT"
    # Marine
    INLAND_VESSEL = "INLAND_VESSEL"
    COASTAL_VESSEL = "COASTAL_VESSEL"
    OCEAN_VESSEL = "OCEAN_VESSEL"
    # Aviation
    CORPORATE_JET = "CORPORATE_JET"
    HELICOPTER = "HELICOPTER"
    TURBOPROP = "TURBOPROP"
    # Rail
    DIESEL_LOCOMOTIVE = "DIESEL_LOCOMOTIVE"


class FuelType(str, Enum):
    """Fuel type identifiers for mobile combustion sources.

    Covers all major fossil fuels, biofuel blends, and alternative fuels
    used in mobile combustion. Each fuel type has associated emission
    factors, density, heating values, and biofuel fraction data.

    Biofuel blends (E10, E85, B5, B20, B100, SAF) have their biogenic
    CO2 component tracked separately per GHG Protocol Chapter 4 guidance.
    """

    GASOLINE = "GASOLINE"
    DIESEL = "DIESEL"
    BIODIESEL_B5 = "BIODIESEL_B5"
    BIODIESEL_B20 = "BIODIESEL_B20"
    BIODIESEL_B100 = "BIODIESEL_B100"
    ETHANOL_E10 = "ETHANOL_E10"
    ETHANOL_E85 = "ETHANOL_E85"
    CNG = "CNG"
    LNG = "LNG"
    LPG = "LPG"
    PROPANE = "PROPANE"
    JET_FUEL_A = "JET_FUEL_A"
    AVGAS = "AVGAS"
    MARINE_DIESEL_OIL = "MARINE_DIESEL_OIL"
    HEAVY_FUEL_OIL = "HEAVY_FUEL_OIL"
    SUSTAINABLE_AVIATION_FUEL = "SUSTAINABLE_AVIATION_FUEL"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in mobile combustion calculations.

    CO2: Carbon dioxide - primary combustion product from fuel oxidation.
    CH4: Methane - incomplete combustion by-product, depends on engine
        type and emission control technology.
    N2O: Nitrous oxide - combustion by-product, depends on catalyst type
        and operating temperature.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"


class CalculationMethod(str, Enum):
    """Calculation methodology for mobile combustion emissions.

    FUEL_BASED: Emissions calculated from actual fuel consumption data.
        Most accurate approach. Uses fuel quantity x emission factor.
    DISTANCE_BASED: Emissions estimated from distance travelled and
        vehicle fuel economy. Uses distance / fuel economy x emission
        factor. Appropriate when fuel purchase data is unavailable.
    SPEND_BASED: Emissions estimated from fuel expenditure using
        spend-based emission factors. Least accurate; used as fallback
        when neither fuel nor distance data is available.
    """

    FUEL_BASED = "FUEL_BASED"
    DISTANCE_BASED = "DISTANCE_BASED"
    SPEND_BASED = "SPEND_BASED"


class CalculationTier(str, Enum):
    """GHG Protocol / IPCC calculation methodology tier level.

    TIER_1: Default emission factors by fuel type. Uses published average
        factors. Fuel-based CO2 calculations only.
    TIER_2: Vehicle-type-specific emission factors for CH4 and N2O.
        Uses more granular factors reflecting vehicle technology and
        emission control systems.
    TIER_3: Model-year-specific and technology-specific factors.
        Highest accuracy using manufacturer data and MOVES/COPERT
        model outputs.
    """

    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class EmissionFactorSource(str, Enum):
    """Source authority for emission factor values.

    EPA: US Environmental Protection Agency (MOVES, AP-42, 40 CFR Part 98).
    IPCC: IPCC 2006 Guidelines for National GHG Inventories, Chapter 3.
    DEFRA: UK Department for Environment, Food and Rural Affairs
        (GHG Conversion Factors).
    EU_ETS: European Union Emissions Trading System factors.
    CUSTOM: Organization-specific or fleet-measured factors.
    """

    EPA = "EPA"
    IPCC = "IPCC"
    DEFRA = "DEFRA"
    EU_ETS = "EU_ETS"
    CUSTOM = "CUSTOM"


class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for GWP conversion factors.

    AR4: Fourth Assessment Report (2007). GWP-100: CH4=25, N2O=298.
    AR5: Fifth Assessment Report (2014). GWP-100: CH4=28, N2O=265.
    AR6: Sixth Assessment Report (2021). GWP-100: CH4=27.3, N2O=273.
    AR6_20YR: Sixth Assessment Report 20-year GWP. GWP-20: CH4=81.2, N2O=273.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class DistanceUnit(str, Enum):
    """Distance measurement units for mobile combustion calculations.

    KM: Kilometres. SI standard unit for distance.
    MILES: Statute miles. US and UK convention.
    NAUTICAL_MILES: Nautical miles. Used for marine and aviation distances.
    """

    KM = "KM"
    MILES = "MILES"
    NAUTICAL_MILES = "NAUTICAL_MILES"


class FuelEconomyUnit(str, Enum):
    """Fuel economy measurement units for distance-based calculations.

    L_PER_100KM: Litres per 100 kilometres. European convention.
    MPG_US: Miles per US gallon. US convention.
    MPG_UK: Miles per imperial gallon. UK convention.
    KM_PER_L: Kilometres per litre. Asian convention.
    """

    L_PER_100KM = "L_PER_100KM"
    MPG_US = "MPG_US"
    MPG_UK = "MPG_UK"
    KM_PER_L = "KM_PER_L"


class EmissionControlTechnology(str, Enum):
    """Vehicle emission control technology classifications.

    Determines applicable CH4 and N2O emission factors for Tier 2/3
    calculations. Vehicles with advanced catalytic converters have
    different CH4/N2O profiles than uncontrolled or older technology.

    UNCONTROLLED: No emission control devices. Pre-regulation vehicles.
    OXIDATION_CATALYST: Two-way oxidation catalyst (diesel vehicles).
    THREE_WAY_CATALYST: Three-way catalytic converter (gasoline vehicles).
    ADVANCED_CATALYST: Advanced multi-stage emission control system.
    EURO_1 through EURO_6: European emission standards stages.
    TIER_1_EPA through TIER_4_EPA: US EPA emission standards tiers.
    """

    UNCONTROLLED = "UNCONTROLLED"
    OXIDATION_CATALYST = "OXIDATION_CATALYST"
    THREE_WAY_CATALYST = "THREE_WAY_CATALYST"
    ADVANCED_CATALYST = "ADVANCED_CATALYST"
    EURO_1 = "EURO_1"
    EURO_2 = "EURO_2"
    EURO_3 = "EURO_3"
    EURO_4 = "EURO_4"
    EURO_5 = "EURO_5"
    EURO_6 = "EURO_6"
    TIER_1_EPA = "TIER_1_EPA"
    TIER_2_EPA = "TIER_2_EPA"
    TIER_3_EPA = "TIER_3_EPA"
    TIER_4_EPA = "TIER_4_EPA"


class VehicleStatus(str, Enum):
    """Operational status of a registered fleet vehicle.

    ACTIVE: Vehicle is in active service and generating emissions.
    INACTIVE: Vehicle is temporarily out of service (e.g. seasonal).
    DISPOSED: Vehicle has been permanently removed from the fleet.
    MAINTENANCE: Vehicle is undergoing maintenance or repair.
    """

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DISPOSED = "DISPOSED"
    MAINTENANCE = "MAINTENANCE"


class TripStatus(str, Enum):
    """Status of a vehicle trip record.

    PLANNED: Trip has been scheduled but not yet started.
    IN_PROGRESS: Trip is currently underway.
    COMPLETED: Trip has been completed with final distance/fuel recorded.
    CANCELLED: Trip was cancelled before completion.
    """

    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class ComplianceStatus(str, Enum):
    """Compliance status for regulatory framework checks.

    COMPLIANT: All applicable requirements are fully satisfied.
    NON_COMPLIANT: One or more requirements are not satisfied.
    NEEDS_REVIEW: Requirements need manual review for determination.
    EXEMPT: Entity is exempt from the applicable requirements.
    """

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    EXEMPT = "EXEMPT"


class ReportingPeriod(str, Enum):
    """Temporal granularity for emission reporting aggregation.

    MONTHLY: Calendar month aggregation.
    QUARTERLY: Calendar quarter (Q1-Q4) aggregation.
    ANNUAL: Full calendar or fiscal year aggregation.
    """

    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


class UnitType(str, Enum):
    """Physical units for fuel quantity, distance, and energy measurement.

    Covers volume, mass, energy, and distance units commonly used in
    mobile combustion fuel records and trip data. The calculation engine
    normalizes all inputs to standard units before applying emission
    factors.
    """

    # Volume units
    LITERS = "LITERS"
    GALLONS = "GALLONS"
    CUBIC_METERS = "CUBIC_METERS"
    # Mass units
    KG = "KG"
    TONNES = "TONNES"
    # Energy units
    KWH = "KWH"
    GJ = "GJ"


# =============================================================================
# Data Models (14)
# =============================================================================


class VehicleTypeInfo(BaseModel):
    """Reference data for a specific vehicle type.

    Provides default values for emission factor source, fuel type, and
    vehicle category that are applied when processing calculation inputs
    that do not specify these fields explicitly.

    Attributes:
        vehicle_type: Vehicle type identifier.
        category: Broad vehicle classification (ON_ROAD, OFF_ROAD, etc.).
        description: Human-readable description of the vehicle type.
        default_fuel_type: Default fuel type for this vehicle.
        default_ef_source: Default emission factor source authority.
        typical_fuel_economy: Typical fuel economy value for distance-based
            calculations when no specific value is provided.
        fuel_economy_unit: Unit for the typical fuel economy value.
    """

    vehicle_type: VehicleType = Field(
        ...,
        description="Vehicle type identifier",
    )
    category: VehicleCategory = Field(
        ...,
        description="Broad vehicle classification",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable description of the vehicle type",
    )
    default_fuel_type: FuelType = Field(
        ...,
        description="Default fuel type for this vehicle",
    )
    default_ef_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="Default emission factor source authority",
    )
    typical_fuel_economy: Optional[float] = Field(
        default=None,
        gt=0,
        description="Typical fuel economy value for distance-based calculations",
    )
    fuel_economy_unit: FuelEconomyUnit = Field(
        default=FuelEconomyUnit.L_PER_100KM,
        description="Unit for the typical fuel economy value",
    )


class FuelTypeInfo(BaseModel):
    """Physical and chemical properties of a mobile combustion fuel type.

    Defines the density, heating value, biofuel fraction, and default
    emission factor source for a fuel type used in mobile combustion
    calculations.

    Attributes:
        fuel_type: Fuel type identifier.
        description: Human-readable description of the fuel.
        density_kg_per_l: Fuel density for volume-to-mass conversion (kg/L).
        heating_value_gj_per_l: Heating value per litre (GJ/L).
        heating_value_gj_per_kg: Heating value per kilogram (GJ/kg).
        biofuel_fraction: Fraction of biofuel content (0.0-1.0). Zero for
            pure fossil fuels, non-zero for biofuel blends.
        is_biofuel_blend: Whether this fuel contains any biofuel component.
        carbon_content_kg_per_gj: Carbon content per unit energy (kg C/GJ).
        default_ef_source: Default emission factor source authority.
    """

    fuel_type: FuelType = Field(
        ...,
        description="Fuel type identifier",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable description of the fuel",
    )
    density_kg_per_l: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fuel density (kg/L) for volume-to-mass conversion",
    )
    heating_value_gj_per_l: Optional[float] = Field(
        default=None,
        gt=0,
        description="Heating value per litre (GJ/L)",
    )
    heating_value_gj_per_kg: Optional[float] = Field(
        default=None,
        gt=0,
        description="Heating value per kilogram (GJ/kg)",
    )
    biofuel_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of biofuel content (0.0-1.0)",
    )
    is_biofuel_blend: bool = Field(
        default=False,
        description="Whether this fuel contains any biofuel component",
    )
    carbon_content_kg_per_gj: Optional[float] = Field(
        default=None,
        gt=0,
        description="Carbon content per unit energy (kg C/GJ)",
    )
    default_ef_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="Default emission factor source authority",
    )

    @field_validator("is_biofuel_blend")
    @classmethod
    def biofuel_flag_consistency(cls, v: bool, info: Any) -> bool:
        """Ensure is_biofuel_blend is consistent with biofuel_fraction."""
        fraction = info.data.get("biofuel_fraction", 0.0)
        if fraction > 0.0 and not v:
            return True  # Auto-correct: fraction > 0 means it is a blend
        return v


class EmissionFactorRecord(BaseModel):
    """A single emission factor record for a vehicle-fuel-gas combination.

    Emission factors define the mass of GHG released per unit of fuel
    consumed or per unit of distance travelled. Each record is scoped to
    a specific vehicle type, fuel type, greenhouse gas, source authority,
    and model year range.

    Attributes:
        factor_id: Unique identifier for this emission factor record.
        vehicle_type: Vehicle type this factor applies to.
        fuel_type: Fuel type this factor applies to.
        gas: Greenhouse gas species this factor quantifies.
        value: Emission factor numeric value (mass GHG per unit).
        unit: Unit of measurement for the factor (e.g. kg CO2/litre,
            g CH4/km, g N2O/mile).
        source: Authority that published this emission factor.
        tier: GHG Protocol calculation tier this factor is appropriate for.
        emission_control: Emission control technology assumed for this factor.
        model_year_start: Earliest model year this factor applies to.
        model_year_end: Latest model year this factor applies to.
        geography: ISO 3166 country/region code or GLOBAL.
        effective_date: Date from which this factor is valid.
        expiry_date: Date after which this factor is superseded.
        reference: Bibliographic reference or document ID for audit trails.
        notes: Optional human-readable notes about applicability.
    """

    factor_id: str = Field(
        default_factory=lambda: f"ef_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this emission factor record",
    )
    vehicle_type: Optional[VehicleType] = Field(
        default=None,
        description="Vehicle type this factor applies to (None for fuel-only factors)",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Fuel type this factor applies to",
    )
    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species this factor quantifies",
    )
    value: float = Field(
        ...,
        gt=0,
        description="Emission factor numeric value (mass GHG per unit)",
    )
    unit: str = Field(
        ...,
        min_length=1,
        description="Unit of measurement (e.g. kg CO2/litre, g CH4/km)",
    )
    source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="Authority that published this emission factor",
    )
    tier: CalculationTier = Field(
        default=CalculationTier.TIER_1,
        description="Calculation tier this factor is appropriate for",
    )
    emission_control: Optional[EmissionControlTechnology] = Field(
        default=None,
        description="Emission control technology assumed for this factor",
    )
    model_year_start: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Earliest model year this factor applies to",
    )
    model_year_end: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Latest model year this factor applies to",
    )
    geography: str = Field(
        default="GLOBAL",
        description="ISO 3166 country/region code or GLOBAL",
    )
    effective_date: Optional[datetime] = Field(
        default=None,
        description="Date from which this factor is valid",
    )
    expiry_date: Optional[datetime] = Field(
        default=None,
        description="Date after which this factor is superseded",
    )
    reference: Optional[str] = Field(
        default=None,
        description="Bibliographic reference or document ID",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Human-readable notes about applicability",
    )

    @field_validator("model_year_end")
    @classmethod
    def model_year_range_valid(
        cls, v: Optional[int], info: Any
    ) -> Optional[int]:
        """Validate that model_year_end >= model_year_start when both set."""
        if v is not None and info.data.get("model_year_start") is not None:
            if v < info.data["model_year_start"]:
                raise ValueError(
                    "model_year_end must be >= model_year_start"
                )
        return v

    @field_validator("expiry_date")
    @classmethod
    def expiry_after_effective(
        cls, v: Optional[datetime], info: Any
    ) -> Optional[datetime]:
        """Validate that expiry_date is after effective_date when both set."""
        if v is not None and info.data.get("effective_date") is not None:
            if v <= info.data["effective_date"]:
                raise ValueError(
                    "expiry_date must be after effective_date"
                )
        return v


class VehicleRegistration(BaseModel):
    """Registration record for a fleet vehicle.

    Captures vehicle identification, type classification, fuel type,
    emission control technology, and organizational assignment for
    fleet-level emission tracking and aggregation.

    Attributes:
        vehicle_id: Unique identifier for this vehicle in the fleet.
        vin: Vehicle Identification Number (17 characters for on-road).
        make: Vehicle manufacturer name.
        model: Vehicle model name.
        model_year: Vehicle model year.
        vehicle_type: Vehicle type classification.
        fuel_type: Primary fuel type used by this vehicle.
        emission_control: Installed emission control technology.
        department: Organizational department that operates this vehicle.
        fleet_id: Fleet grouping identifier.
        status: Current operational status of the vehicle.
        odometer_km: Current odometer reading in kilometres.
        registration_date: Date when the vehicle was registered in the fleet.
        disposal_date: Date when the vehicle was disposed of (if applicable).
        notes: Optional notes about the vehicle.
    """

    vehicle_id: str = Field(
        default_factory=lambda: f"veh_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this vehicle in the fleet",
    )
    vin: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=17,
        description="Vehicle Identification Number",
    )
    make: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Vehicle manufacturer name",
    )
    model: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Vehicle model name",
    )
    model_year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Vehicle model year",
    )
    vehicle_type: VehicleType = Field(
        ...,
        description="Vehicle type classification",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Primary fuel type used by this vehicle",
    )
    emission_control: EmissionControlTechnology = Field(
        default=EmissionControlTechnology.THREE_WAY_CATALYST,
        description="Installed emission control technology",
    )
    department: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Organizational department that operates this vehicle",
    )
    fleet_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Fleet grouping identifier",
    )
    status: VehicleStatus = Field(
        default=VehicleStatus.ACTIVE,
        description="Current operational status of the vehicle",
    )
    odometer_km: Optional[float] = Field(
        default=None,
        ge=0,
        description="Current odometer reading in kilometres",
    )
    registration_date: datetime = Field(
        default_factory=_utcnow,
        description="Date when the vehicle was registered in the fleet",
    )
    disposal_date: Optional[datetime] = Field(
        default=None,
        description="Date when the vehicle was disposed of",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional notes about the vehicle",
    )

    @field_validator("disposal_date")
    @classmethod
    def disposal_after_registration(
        cls, v: Optional[datetime], info: Any
    ) -> Optional[datetime]:
        """Validate that disposal_date is after registration_date when set."""
        if v is not None and info.data.get("registration_date") is not None:
            if v <= info.data["registration_date"]:
                raise ValueError(
                    "disposal_date must be after registration_date"
                )
        return v


class TripRecord(BaseModel):
    """Record of a single vehicle trip for emission tracking.

    Captures trip details including vehicle identification, distance
    travelled, fuel consumed, route information, and timestamps.
    Supports both fuel-based and distance-based emission calculations.

    Attributes:
        trip_id: Unique identifier for this trip.
        vehicle_id: Reference to the registered vehicle that made this trip.
        distance_value: Distance travelled during the trip.
        distance_unit: Unit of the distance measurement.
        fuel_consumed_liters: Actual fuel consumed during the trip (litres).
        fuel_type: Fuel type used during this trip.
        start_time: UTC timestamp when the trip started.
        end_time: UTC timestamp when the trip ended.
        start_location: Starting location description or coordinates.
        end_location: Ending location description or coordinates.
        route_description: Optional description of the route taken.
        purpose: Trip purpose (business, commute, delivery, etc.).
        status: Current status of the trip record.
        driver_id: Optional identifier for the driver.
        cargo_weight_kg: Weight of cargo carried during the trip (kg).
        passengers: Number of passengers during the trip.
    """

    trip_id: str = Field(
        default_factory=lambda: f"trip_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this trip",
    )
    vehicle_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the registered vehicle",
    )
    distance_value: Optional[float] = Field(
        default=None,
        gt=0,
        description="Distance travelled during the trip",
    )
    distance_unit: DistanceUnit = Field(
        default=DistanceUnit.KM,
        description="Unit of the distance measurement",
    )
    fuel_consumed_liters: Optional[float] = Field(
        default=None,
        gt=0,
        description="Actual fuel consumed during the trip (litres)",
    )
    fuel_type: Optional[FuelType] = Field(
        default=None,
        description="Fuel type used during this trip",
    )
    start_time: datetime = Field(
        ...,
        description="UTC timestamp when the trip started",
    )
    end_time: datetime = Field(
        ...,
        description="UTC timestamp when the trip ended",
    )
    start_location: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Starting location description or coordinates",
    )
    end_location: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Ending location description or coordinates",
    )
    route_description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional description of the route taken",
    )
    purpose: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Trip purpose (business, commute, delivery, etc.)",
    )
    status: TripStatus = Field(
        default=TripStatus.COMPLETED,
        description="Current status of the trip record",
    )
    driver_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional identifier for the driver",
    )
    cargo_weight_kg: Optional[float] = Field(
        default=None,
        ge=0,
        description="Weight of cargo carried during the trip (kg)",
    )
    passengers: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of passengers during the trip",
    )

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v: datetime, info: Any) -> datetime:
        """Validate that end_time is after start_time."""
        start_time = info.data.get("start_time")
        if start_time is not None and v <= start_time:
            raise ValueError("end_time must be after start_time")
        return v


class CalculationInput(BaseModel):
    """Input data for a single mobile combustion emission calculation.

    Represents one fuel consumption or distance record for a vehicle,
    optionally linked to a specific trip and fleet. The calculation
    engine uses this input together with emission factors and fuel
    properties to compute GHG emissions.

    Attributes:
        vehicle_type: Type of vehicle that consumed the fuel.
        fuel_type: Type of fuel consumed.
        quantity: Amount of fuel consumed or distance travelled.
        unit: Measurement unit for the quantity field.
        calculation_method: Methodology for calculating emissions.
        fuel_economy_value: Fuel economy for distance-based calculations.
        fuel_economy_unit: Unit for the fuel economy value.
        vehicle_id: Optional link to a registered fleet vehicle.
        trip_id: Optional link to a specific trip record.
        fleet_id: Optional fleet identifier for aggregation.
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        emission_control: Emission control technology on the vehicle.
        model_year: Vehicle model year for Tier 2/3 factor selection.
        geography: ISO 3166 country/region code for regional factors.
        tier: Optional override of the default calculation tier.
        gwp_source: Optional override of the default GWP source.
        custom_emission_factor_co2: Custom CO2 emission factor override.
        custom_emission_factor_ch4: Custom CH4 emission factor override.
        custom_emission_factor_n2o: Custom N2O emission factor override.
    """

    vehicle_type: VehicleType = Field(
        ...,
        description="Type of vehicle that consumed the fuel",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Type of fuel consumed",
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Amount of fuel consumed or distance travelled",
    )
    unit: UnitType = Field(
        ...,
        description="Measurement unit for the quantity field",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.FUEL_BASED,
        description="Methodology for calculating emissions",
    )
    fuel_economy_value: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fuel economy for distance-based calculations",
    )
    fuel_economy_unit: FuelEconomyUnit = Field(
        default=FuelEconomyUnit.L_PER_100KM,
        description="Unit for the fuel economy value",
    )
    vehicle_id: Optional[str] = Field(
        default=None,
        description="Optional link to a registered fleet vehicle",
    )
    trip_id: Optional[str] = Field(
        default=None,
        description="Optional link to a specific trip record",
    )
    fleet_id: Optional[str] = Field(
        default=None,
        description="Optional fleet identifier for aggregation",
    )
    period_start: datetime = Field(
        ...,
        description="Start of the reporting period",
    )
    period_end: datetime = Field(
        ...,
        description="End of the reporting period",
    )
    emission_control: Optional[EmissionControlTechnology] = Field(
        default=None,
        description="Emission control technology on the vehicle",
    )
    model_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Vehicle model year for Tier 2/3 factor selection",
    )
    geography: Optional[str] = Field(
        default=None,
        description="ISO 3166 country/region code for regional factors",
    )
    tier: Optional[CalculationTier] = Field(
        default=None,
        description="Optional override of the default calculation tier",
    )
    gwp_source: Optional[GWPSource] = Field(
        default=None,
        description="Optional override of the default GWP source",
    )
    custom_emission_factor_co2: Optional[float] = Field(
        default=None,
        gt=0,
        description="Custom CO2 emission factor override (kg CO2/unit)",
    )
    custom_emission_factor_ch4: Optional[float] = Field(
        default=None,
        gt=0,
        description="Custom CH4 emission factor override (g CH4/unit)",
    )
    custom_emission_factor_n2o: Optional[float] = Field(
        default=None,
        gt=0,
        description="Custom N2O emission factor override (g N2O/unit)",
    )

    @field_validator("period_end")
    @classmethod
    def period_end_after_start(cls, v: datetime, info: Any) -> datetime:
        """Validate that period_end is after period_start."""
        period_start = info.data.get("period_start")
        if period_start is not None and v <= period_start:
            raise ValueError("period_end must be after period_start")
        return v


class GasEmission(BaseModel):
    """Emission result for a single greenhouse gas from a mobile combustion event.

    Captures the calculated emissions in both native mass units and
    CO2-equivalent, along with the emission factor and GWP used for
    full traceability.

    Attributes:
        gas: Greenhouse gas species (CO2, CH4, N2O).
        emissions_kg: Calculated emissions in kilograms of the specific gas.
        emissions_tco2e: Calculated emissions in tonnes of CO2-equivalent.
        emission_factor_value: Numeric value of the emission factor applied.
        emission_factor_unit: Unit of the emission factor applied.
        emission_factor_source: Source authority for the emission factor.
        gwp_applied: Global Warming Potential multiplier applied for
            CO2e conversion.
    """

    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species (CO2, CH4, N2O)",
    )
    emissions_kg: float = Field(
        ...,
        ge=0,
        description="Emissions in kilograms of the specific gas",
    )
    emissions_tco2e: float = Field(
        ...,
        ge=0,
        description="Emissions in tonnes of CO2-equivalent",
    )
    emission_factor_value: float = Field(
        ...,
        gt=0,
        description="Numeric value of the emission factor applied",
    )
    emission_factor_unit: str = Field(
        ...,
        min_length=1,
        description="Unit of the emission factor applied",
    )
    emission_factor_source: str = Field(
        ...,
        min_length=1,
        description="Source authority for the emission factor",
    )
    gwp_applied: float = Field(
        ...,
        gt=0,
        description="GWP multiplier applied for CO2e conversion",
    )


class CalculationResult(BaseModel):
    """Complete result of a single mobile combustion emission calculation.

    Contains all calculated emissions by gas, total CO2e, biogenic CO2
    (if applicable), the methodology parameters used, and a SHA-256
    provenance hash for audit trail integrity.

    Attributes:
        calculation_id: Unique identifier for this calculation result.
        vehicle_type: Vehicle type that generated emissions.
        fuel_type: Fuel type that was consumed.
        calculation_method: Methodology used for this calculation.
        fuel_quantity_liters: Fuel consumed in litres (if fuel-based).
        distance_km: Distance travelled in kilometres (if distance-based).
        fuel_economy_used: Fuel economy value applied (if distance-based).
        tier_used: Calculation tier applied.
        emissions_by_gas: Itemized emissions for each greenhouse gas.
        total_co2e_kg: Total CO2-equivalent emissions in kilograms.
        total_co2e_tonnes: Total CO2-equivalent emissions in metric tonnes.
        biogenic_co2_kg: Biogenic CO2 emissions in kilograms.
        biogenic_co2_tonnes: Biogenic CO2 emissions in metric tonnes.
        fossil_co2_kg: Fossil-origin CO2 emissions in kilograms.
        fossil_co2_tonnes: Fossil-origin CO2 emissions in metric tonnes.
        gwp_source_used: GWP source edition applied.
        provenance_hash: SHA-256 hash for audit trail integrity.
        calculation_trace: Ordered list of human-readable calculation steps.
        timestamp: UTC timestamp when the calculation was performed.
        vehicle_id: Vehicle identifier (if from fleet).
        trip_id: Trip identifier (if from trip record).
        fleet_id: Fleet identifier (if from fleet).
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
    """

    calculation_id: str = Field(
        default_factory=lambda: f"calc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this calculation result",
    )
    vehicle_type: VehicleType = Field(
        ...,
        description="Vehicle type that generated emissions",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Fuel type that was consumed",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Methodology used for this calculation",
    )
    fuel_quantity_liters: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fuel consumed in litres (if fuel-based)",
    )
    distance_km: Optional[float] = Field(
        default=None,
        ge=0,
        description="Distance travelled in kilometres (if distance-based)",
    )
    fuel_economy_used: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fuel economy value applied (if distance-based)",
    )
    tier_used: CalculationTier = Field(
        ...,
        description="Calculation tier applied",
    )
    emissions_by_gas: List[GasEmission] = Field(
        default_factory=list,
        max_length=MAX_GASES_PER_RESULT,
        description="Itemized emissions for each greenhouse gas",
    )
    total_co2e_kg: float = Field(
        ...,
        ge=0,
        description="Total CO2-equivalent emissions in kilograms",
    )
    total_co2e_tonnes: float = Field(
        ...,
        ge=0,
        description="Total CO2-equivalent emissions in metric tonnes",
    )
    biogenic_co2_kg: float = Field(
        default=0.0,
        ge=0,
        description="Biogenic CO2 emissions in kilograms",
    )
    biogenic_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Biogenic CO2 emissions in metric tonnes",
    )
    fossil_co2_kg: float = Field(
        default=0.0,
        ge=0,
        description="Fossil-origin CO2 emissions in kilograms",
    )
    fossil_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Fossil-origin CO2 emissions in metric tonnes",
    )
    gwp_source_used: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source edition applied",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail integrity",
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        max_length=MAX_TRACE_STEPS,
        description="Ordered list of human-readable calculation steps",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the calculation was performed",
    )
    vehicle_id: Optional[str] = Field(
        default=None,
        description="Vehicle identifier (if from fleet)",
    )
    trip_id: Optional[str] = Field(
        default=None,
        description="Trip identifier (if from trip record)",
    )
    fleet_id: Optional[str] = Field(
        default=None,
        description="Fleet identifier (if from fleet)",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the reporting period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the reporting period",
    )


class BatchCalculationInput(BaseModel):
    """Request model for batch mobile combustion calculations.

    Groups multiple calculation inputs for processing as a single
    batch, sharing common parameters like GWP source and biogenic
    tracking preference.

    Attributes:
        calculations: List of individual calculation inputs to process.
        gwp_source: IPCC Assessment Report for GWP values.
        include_biogenic: Whether to track biogenic CO2 separately.
        organization_id: Organization identifier for aggregation.
        reporting_period: Temporal granularity for the batch.
    """

    calculations: List[CalculationInput] = Field(
        ...,
        min_length=1,
        max_length=MAX_CALCULATIONS_PER_BATCH,
        description="List of individual calculation inputs to process",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    include_biogenic: bool = Field(
        default=True,
        description="Whether to track biogenic CO2 separately",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization identifier for aggregation",
    )
    reporting_period: Optional[ReportingPeriod] = Field(
        default=None,
        description="Temporal granularity for the batch",
    )


class BatchCalculationResponse(BaseModel):
    """Response model for a batch mobile combustion calculation.

    Aggregates individual calculation results with batch-level totals,
    emissions breakdown by vehicle type and fuel type, and processing
    metadata.

    Attributes:
        success: Whether all calculations in the batch succeeded.
        results: List of individual calculation results.
        total_co2e_tonnes: Batch total CO2-equivalent in metric tonnes.
        total_co2_tonnes: Batch total CO2 in metric tonnes.
        total_ch4_tco2e: Batch total CH4 in metric tonnes CO2e.
        total_n2o_tco2e: Batch total N2O in metric tonnes CO2e.
        total_biogenic_co2_tonnes: Batch total biogenic CO2 in tonnes.
        emissions_by_vehicle_type: Emissions by vehicle type (tCO2e).
        emissions_by_fuel_type: Emissions by fuel type (tCO2e).
        calculation_count: Number of successful calculations.
        failed_count: Number of failed calculations.
        processing_time_ms: Total batch processing wall-clock time in ms.
        provenance_hash: SHA-256 hash covering the entire batch result.
        gwp_source: GWP source used for this batch.
    """

    success: bool = Field(
        ...,
        description="Whether all calculations in the batch succeeded",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    total_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch total CO2-equivalent in metric tonnes",
    )
    total_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch total CO2 in metric tonnes",
    )
    total_ch4_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Batch total CH4 in metric tonnes CO2e",
    )
    total_n2o_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Batch total N2O in metric tonnes CO2e",
    )
    total_biogenic_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch total biogenic CO2 in metric tonnes",
    )
    emissions_by_vehicle_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions aggregated by vehicle type (tCO2e)",
    )
    emissions_by_fuel_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions aggregated by fuel type (tCO2e)",
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of successful calculations",
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of failed calculations",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total batch processing wall-clock time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash covering the entire batch result",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source used for this batch",
    )


class FleetAggregation(BaseModel):
    """Fleet-level emission aggregation across all vehicles and trips.

    Rolls up individual calculation results into a fleet total broken
    down by gas, vehicle type, and fuel type. Includes intensity metrics
    for benchmarking and reporting.

    Attributes:
        fleet_id: Unique fleet identifier.
        organization_id: Parent organization identifier.
        reporting_period_type: Temporal granularity of the aggregation.
        period_start: Start of the aggregation period.
        period_end: End of the aggregation period.
        total_co2e_tonnes: Fleet total CO2-equivalent (metric tonnes).
        total_co2_tonnes: Fleet total CO2 (metric tonnes).
        total_ch4_tco2e: Fleet total CH4 in CO2e (metric tonnes).
        total_n2o_tco2e: Fleet total N2O in CO2e (metric tonnes).
        biogenic_co2_tonnes: Biogenic CO2 total (metric tonnes).
        emissions_by_vehicle_type: Breakdown by vehicle type (tCO2e).
        emissions_by_fuel_type: Breakdown by fuel type (tCO2e).
        total_distance_km: Total fleet distance in kilometres.
        total_fuel_liters: Total fleet fuel consumption in litres.
        emission_intensity_per_km: Emissions per kilometre (kg CO2e/km).
        emission_intensity_per_liter: Emissions per litre (kg CO2e/L).
        vehicle_count: Number of active vehicles in the fleet.
        trip_count: Number of trips in this aggregation.
        calculation_count: Number of calculations in this aggregation.
        provenance_hash: SHA-256 hash for audit trail integrity.
    """

    fleet_id: str = Field(
        ...,
        min_length=1,
        description="Unique fleet identifier",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Parent organization identifier",
    )
    reporting_period_type: ReportingPeriod = Field(
        ...,
        description="Temporal granularity of the aggregation",
    )
    period_start: datetime = Field(
        ...,
        description="Start of the aggregation period",
    )
    period_end: datetime = Field(
        ...,
        description="End of the aggregation period",
    )
    total_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Fleet total CO2-equivalent (metric tonnes)",
    )
    total_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Fleet total CO2 (metric tonnes)",
    )
    total_ch4_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Fleet total CH4 in CO2e (metric tonnes)",
    )
    total_n2o_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Fleet total N2O in CO2e (metric tonnes)",
    )
    biogenic_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Biogenic CO2 total (metric tonnes)",
    )
    emissions_by_vehicle_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown by vehicle type (tCO2e)",
    )
    emissions_by_fuel_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown by fuel type (tCO2e)",
    )
    total_distance_km: float = Field(
        default=0.0,
        ge=0,
        description="Total fleet distance in kilometres",
    )
    total_fuel_liters: float = Field(
        default=0.0,
        ge=0,
        description="Total fleet fuel consumption in litres",
    )
    emission_intensity_per_km: Optional[float] = Field(
        default=None,
        ge=0,
        description="Emissions per kilometre (kg CO2e/km)",
    )
    emission_intensity_per_liter: Optional[float] = Field(
        default=None,
        ge=0,
        description="Emissions per litre (kg CO2e/L)",
    )
    vehicle_count: int = Field(
        default=0,
        ge=0,
        description="Number of active vehicles in the fleet",
    )
    trip_count: int = Field(
        default=0,
        ge=0,
        description="Number of trips in this aggregation",
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations in this aggregation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail integrity",
    )

    @field_validator("period_end")
    @classmethod
    def aggregation_end_after_start(
        cls, v: datetime, info: Any
    ) -> datetime:
        """Validate that period_end is after period_start."""
        period_start = info.data.get("period_start")
        if period_start is not None and v <= period_start:
            raise ValueError("period_end must be after period_start")
        return v


class UncertaintyResult(BaseModel):
    """Monte Carlo uncertainty quantification result for a mobile combustion calculation.

    Provides statistical characterization of emission estimate uncertainty
    including mean, standard deviation, confidence intervals at multiple
    levels, and contribution analysis showing which input parameters
    drive the most uncertainty.

    Attributes:
        mean_co2e: Mean CO2-equivalent emission estimate (tonnes).
        std_dev: Standard deviation of the CO2e estimate (tonnes).
        coefficient_of_variation: CV = std_dev / mean (dimensionless).
        confidence_interval_90: 90% confidence interval (lower, upper).
        confidence_interval_95: 95% confidence interval (lower, upper).
        confidence_interval_99: 99% confidence interval (lower, upper).
        iterations: Number of Monte Carlo iterations performed.
        data_quality_score: Overall data quality indicator (1-5 scale).
        tier: Calculation tier used for this uncertainty analysis.
        contributions: Parameter contribution to total variance.
        method: Uncertainty quantification method used.
    """

    mean_co2e: float = Field(
        ...,
        ge=0,
        description="Mean CO2-equivalent emission estimate (tonnes)",
    )
    std_dev: float = Field(
        ...,
        ge=0,
        description="Standard deviation of the CO2e estimate (tonnes)",
    )
    coefficient_of_variation: float = Field(
        ...,
        ge=0,
        description="CV = std_dev / mean (dimensionless)",
    )
    confidence_interval_90: Optional[Tuple[float, float]] = Field(
        default=None,
        description="90% confidence interval (lower, upper) in tonnes CO2e",
    )
    confidence_interval_95: Optional[Tuple[float, float]] = Field(
        default=None,
        description="95% confidence interval (lower, upper) in tonnes CO2e",
    )
    confidence_interval_99: Optional[Tuple[float, float]] = Field(
        default=None,
        description="99% confidence interval (lower, upper) in tonnes CO2e",
    )
    iterations: int = Field(
        ...,
        gt=0,
        description="Number of Monte Carlo iterations performed",
    )
    data_quality_score: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Data quality indicator (1-5 scale)",
    )
    tier: CalculationTier = Field(
        ...,
        description="Calculation tier used for this uncertainty analysis",
    )
    contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameter contribution to total variance (name -> fraction)",
    )
    method: str = Field(
        default="monte_carlo",
        description="Uncertainty quantification method used",
    )


class ComplianceCheckResult(BaseModel):
    """Result of a regulatory compliance check for mobile combustion emissions.

    Tracks how the Mobile Combustion Agent meets specific requirements
    from each supported regulatory framework, providing auditable
    evidence of compliance status.

    Attributes:
        framework: Regulatory framework checked.
        status: Overall compliance status.
        findings: List of specific compliance findings.
        recommendations: List of recommended actions for improvement.
        requirements_met: Number of requirements fully satisfied.
        requirements_total: Total number of applicable requirements.
        evidence_references: List of evidence references (calculation IDs,
            audit entries, etc.).
        checked_at: UTC timestamp when the check was performed.
        checked_by: Agent or user that performed the check.
    """

    framework: str = Field(
        ...,
        min_length=1,
        description="Regulatory framework checked",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="List of specific compliance findings",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of recommended actions for improvement",
    )
    requirements_met: int = Field(
        default=0,
        ge=0,
        description="Number of requirements fully satisfied",
    )
    requirements_total: int = Field(
        default=0,
        ge=0,
        description="Total number of applicable requirements",
    )
    evidence_references: List[str] = Field(
        default_factory=list,
        description="List of evidence references",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the check was performed",
    )
    checked_by: str = Field(
        default="GL-MRV-SCOPE1-003",
        description="Agent or user that performed the check",
    )


class MobileCombustionInput(BaseModel):
    """Overall input model for the Mobile Combustion Agent pipeline.

    Wraps one or more calculation inputs with shared configuration
    parameters for a complete mobile combustion emission calculation
    pipeline run.

    Attributes:
        inputs: List of individual calculation inputs.
        gwp_source: IPCC Assessment Report for GWP values.
        calculation_method: Default calculation methodology.
        include_biogenic: Whether to track biogenic CO2 separately.
        include_uncertainty: Whether to include uncertainty analysis.
        include_compliance: Whether to include compliance checks.
        regulatory_frameworks: Frameworks to check compliance against.
        organization_id: Organization identifier.
        fleet_id: Fleet identifier for fleet-level aggregation.
        reporting_period: Temporal granularity for reporting.
    """

    inputs: List[CalculationInput] = Field(
        ...,
        min_length=1,
        max_length=MAX_CALCULATIONS_PER_BATCH,
        description="List of individual calculation inputs",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.FUEL_BASED,
        description="Default calculation methodology",
    )
    include_biogenic: bool = Field(
        default=True,
        description="Whether to track biogenic CO2 separately",
    )
    include_uncertainty: bool = Field(
        default=True,
        description="Whether to include uncertainty analysis",
    )
    include_compliance: bool = Field(
        default=True,
        description="Whether to include compliance checks",
    )
    regulatory_frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL"],
        description="Frameworks to check compliance against",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization identifier",
    )
    fleet_id: Optional[str] = Field(
        default=None,
        description="Fleet identifier for fleet-level aggregation",
    )
    reporting_period: Optional[ReportingPeriod] = Field(
        default=None,
        description="Temporal granularity for reporting",
    )


class MobileCombustionOutput(BaseModel):
    """Overall output model for the Mobile Combustion Agent pipeline.

    Contains the complete results of a mobile combustion emission
    calculation pipeline including individual results, batch totals,
    fleet aggregation, uncertainty analysis, compliance checks, and
    provenance tracking.

    Attributes:
        success: Whether the pipeline completed successfully.
        results: List of individual calculation results.
        batch_response: Batch-level aggregation of results.
        fleet_aggregation: Fleet-level emission aggregation.
        uncertainty: Uncertainty analysis result.
        compliance_checks: List of compliance check results.
        provenance_hash: SHA-256 hash for the complete pipeline output.
        processing_time_ms: Total pipeline processing time in milliseconds.
        agent_id: Agent identifier that produced this output.
        agent_version: Agent version string.
        timestamp: UTC timestamp when the output was generated.
        errors: List of error messages for any failed calculations.
    """

    success: bool = Field(
        ...,
        description="Whether the pipeline completed successfully",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    batch_response: Optional[BatchCalculationResponse] = Field(
        default=None,
        description="Batch-level aggregation of results",
    )
    fleet_aggregation: Optional[FleetAggregation] = Field(
        default=None,
        description="Fleet-level emission aggregation",
    )
    uncertainty: Optional[UncertaintyResult] = Field(
        default=None,
        description="Uncertainty analysis result",
    )
    compliance_checks: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="List of compliance check results",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for the complete pipeline output",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total pipeline processing time in milliseconds",
    )
    agent_id: str = Field(
        default="GL-MRV-SCOPE1-003",
        description="Agent identifier that produced this output",
    )
    agent_version: str = Field(
        default=VERSION,
        description="Agent version string",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the output was generated",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages for any failed calculations",
    )


class AuditEntry(BaseModel):
    """A single step in the mobile combustion calculation audit trail.

    Records the input, output, and methodology reference for one
    discrete step in a mobile combustion emission calculation. The
    ordered collection of AuditEntry records forms a complete,
    reproducible calculation trace.

    Attributes:
        entry_id: Unique identifier for this audit entry.
        calculation_id: Parent calculation this entry belongs to.
        step_number: Ordinal position of this step in the calculation.
        step_name: Human-readable name of the calculation step.
        input_hash: SHA-256 hash of the input data for this step.
        output_hash: SHA-256 hash of the output data for this step.
        input_data: Input values consumed by this step.
        output_data: Output values produced by this step.
        emission_factor_used: Emission factor applied in this step.
        methodology_reference: Regulatory or methodological citation.
        timestamp: UTC timestamp when this step was executed.
        provenance_hash: SHA-256 hash for this audit entry.
    """

    entry_id: str = Field(
        default_factory=lambda: f"audit_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this audit entry",
    )
    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Parent calculation this entry belongs to",
    )
    step_number: int = Field(
        ...,
        ge=0,
        description="Ordinal position of this step in the calculation",
    )
    step_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name of the calculation step",
    )
    input_hash: str = Field(
        default="",
        description="SHA-256 hash of the input data for this step",
    )
    output_hash: str = Field(
        default="",
        description="SHA-256 hash of the output data for this step",
    )
    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input values consumed by this step",
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output values produced by this step",
    )
    emission_factor_used: Optional[float] = Field(
        default=None,
        description="Emission factor applied in this step (if applicable)",
    )
    methodology_reference: Optional[str] = Field(
        default=None,
        description="Regulatory or methodological citation for this step",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when this step was executed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for this audit entry",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_VEHICLES_PER_REGISTRATION",
    "MAX_TRIPS_PER_BATCH",
    "GWP_VALUES",
    "BIOFUEL_FRACTIONS",
    # Enums
    "VehicleCategory",
    "VehicleType",
    "FuelType",
    "EmissionGas",
    "CalculationMethod",
    "CalculationTier",
    "EmissionFactorSource",
    "GWPSource",
    "DistanceUnit",
    "FuelEconomyUnit",
    "EmissionControlTechnology",
    "VehicleStatus",
    "TripStatus",
    "ComplianceStatus",
    "ReportingPeriod",
    "UnitType",
    # Data models
    "VehicleTypeInfo",
    "FuelTypeInfo",
    "EmissionFactorRecord",
    "VehicleRegistration",
    "TripRecord",
    "CalculationInput",
    "GasEmission",
    "CalculationResult",
    "BatchCalculationInput",
    "BatchCalculationResponse",
    "FleetAggregation",
    "UncertaintyResult",
    "ComplianceCheckResult",
    "MobileCombustionInput",
    "MobileCombustionOutput",
    "AuditEntry",
]
