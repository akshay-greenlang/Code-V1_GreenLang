# -*- coding: utf-8 -*-
"""
Stationary Combustion Agent Data Models - AGENT-MRV-001

Pydantic v2 data models for the Stationary Combustion Agent SDK covering
GHG Protocol Scope 1 stationary combustion calculations including:
- Multi-fuel combustion (gaseous, liquid, solid, biomass)
- Tier 1/2/3 calculation methodologies
- Equipment-level emission tracking (boilers, furnaces, turbines, kilns)
- Biogenic CO2 separation per GHG Protocol guidance
- Monte Carlo uncertainty quantification
- Full regulatory framework mapping (GHG Protocol, ISO 14064, CSRD, EPA, EU ETS)
- SHA-256 provenance chain for complete audit trails

Enumerations (13):
    - FuelCategory, FuelType, EmissionGas, GWPSource, EFSource,
      CalculationTier, EquipmentType, HeatingValueBasis, ControlApproach,
      CalculationStatus, ReportingPeriod, RegulatoryFramework, UnitType

Data Models (12):
    - EmissionFactor, FuelProperties, EquipmentProfile, CombustionInput,
      GasEmission, CalculationResult, BatchCalculationRequest,
      BatchCalculationResponse, UncertaintyResult, FacilityAggregation,
      AuditEntry, ComplianceMapping

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
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

#: Maximum number of efficiency curve coefficients for equipment profiles.
MAX_EFFICIENCY_COEFFICIENTS: int = 10

#: Default oxidation factor for complete combustion.
DEFAULT_OXIDATION_FACTOR: float = 1.0

#: Standard CO2, CH4, N2O GWP values by Assessment Report.
GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR4": {"CO2": 1.0, "CH4": 25.0, "N2O": 298.0},
    "AR5": {"CO2": 1.0, "CH4": 28.0, "N2O": 265.0},
    "AR6": {"CO2": 1.0, "CH4": 27.3, "N2O": 273.0},
}


# =============================================================================
# Enumerations (13)
# =============================================================================


class FuelCategory(str, Enum):
    """Broad classification of combustion fuels by physical phase.

    Used to group fuel types for reporting aggregation and to determine
    applicable default emission factors and heating value ranges.

    GASEOUS: Natural gas, propane, LPG, biogas, and other gaseous fuels.
    LIQUID: Diesel, gasoline, fuel oils, kerosene, and liquid biofuels.
    SOLID: Coal varieties, petroleum coke, wood, and solid waste fuels.
    BIOMASS: Fuels derived from recent biological material (wood, biogas,
        biomass solids/liquids). Biogenic CO2 tracked separately per
        GHG Protocol.
    """

    GASEOUS = "gaseous"
    LIQUID = "liquid"
    SOLID = "solid"
    BIOMASS = "biomass"


class FuelType(str, Enum):
    """Specific fuel type identifiers for stationary combustion sources.

    Covers all major fossil fuels and biomass/waste fuels encountered
    in Scope 1 stationary combustion inventories. Each fuel type has
    associated default emission factors, heating values, and carbon
    content in the emission factor database.

    Naming follows EPA 40 CFR Part 98 Table C-1 and IPCC 2006 GL
    conventions for cross-framework compatibility.
    """

    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    LPG = "lpg"
    PROPANE = "propane"
    KEROSENE = "kerosene"
    JET_FUEL = "jet_fuel"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_ANTHRACITE = "coal_anthracite"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    COAL_LIGNITE = "coal_lignite"
    PETROLEUM_COKE = "petroleum_coke"
    WOOD = "wood"
    BIOMASS_SOLID = "biomass_solid"
    BIOMASS_LIQUID = "biomass_liquid"
    BIOGAS = "biogas"
    LANDFILL_GAS = "landfill_gas"
    COKE_OVEN_GAS = "coke_oven_gas"
    BLAST_FURNACE_GAS = "blast_furnace_gas"
    PEAT = "peat"
    WASTE_OIL = "waste_oil"
    MSW = "msw"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in stationary combustion calculations.

    CO2: Carbon dioxide - primary combustion product.
    CH4: Methane - incomplete combustion by-product.
    N2O: Nitrous oxide - combustion by-product dependent on temperature.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"


class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for GWP conversion factors.

    AR4: Fourth Assessment Report (2007). GWP-100: CH4=25, N2O=298.
    AR5: Fifth Assessment Report (2014). GWP-100: CH4=28, N2O=265.
    AR6: Sixth Assessment Report (2021). GWP-100: CH4=27.3, N2O=273.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class EFSource(str, Enum):
    """Source authority for emission factor values.

    EPA: US Environmental Protection Agency (40 CFR Part 98, AP-42).
    IPCC: IPCC 2006 Guidelines for National GHG Inventories.
    DEFRA: UK Department for Environment, Food and Rural Affairs.
    EU_ETS: European Union Emissions Trading System factors.
    CUSTOM: Organization-specific or facility-measured factors.
    """

    EPA = "EPA"
    IPCC = "IPCC"
    DEFRA = "DEFRA"
    EU_ETS = "EU_ETS"
    CUSTOM = "CUSTOM"


class CalculationTier(str, Enum):
    """GHG Protocol / IPCC calculation methodology tier level.

    TIER_1: Default emission factors by fuel type. Simplest approach
        using published average factors. Suitable when fuel-specific
        data is unavailable.
    TIER_2: Country-specific or fuel-grade-specific emission factors.
        Uses more granular factors reflecting regional fuel composition.
    TIER_3: Facility-level measurements (CEMS, fuel analysis, stack
        testing). Highest accuracy using direct measurement data.
    """

    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class EquipmentType(str, Enum):
    """Classification of stationary combustion equipment.

    Equipment type determines applicable efficiency curves, emission
    factor adjustments, and maintenance-related correction factors
    for Tier 2 and Tier 3 calculations.
    """

    BOILER_FIRE_TUBE = "boiler_fire_tube"
    BOILER_WATER_TUBE = "boiler_water_tube"
    FURNACE = "furnace"
    PROCESS_HEATER = "process_heater"
    GAS_TURBINE_SIMPLE = "gas_turbine_simple"
    GAS_TURBINE_COMBINED = "gas_turbine_combined"
    RECIPROCATING_ENGINE = "reciprocating_engine"
    KILN = "kiln"
    OVEN = "oven"
    DRYER = "dryer"
    FLARE = "flare"
    INCINERATOR = "incinerator"
    THERMAL_OXIDIZER = "thermal_oxidizer"


class HeatingValueBasis(str, Enum):
    """Basis for fuel heating value used in energy content calculations.

    HHV: Higher Heating Value (gross calorific value). Includes latent
        heat of water vapor formed during combustion. Used by EPA and
        GHG Protocol North American convention.
    NCV: Net Calorific Value (lower heating value, LHV). Excludes latent
        heat of water vapor. Used by IPCC and European convention.
    """

    HHV = "HHV"
    NCV = "NCV"


class ControlApproach(str, Enum):
    """Organizational boundary approach for emission ownership.

    OPERATIONAL: Organization reports 100% of emissions from operations
        over which it has operational control.
    FINANCIAL: Organization reports 100% of emissions from operations
        over which it has financial control.
    EQUITY_SHARE: Organization reports emissions proportional to its
        equity share in each operation.
    """

    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    EQUITY_SHARE = "equity_share"


class CalculationStatus(str, Enum):
    """Status of a combustion emission calculation.

    PENDING: Calculation queued but not yet started.
    RUNNING: Calculation in progress.
    COMPLETED: Calculation finished successfully.
    FAILED: Calculation terminated with an error.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportingPeriod(str, Enum):
    """Temporal granularity for emission reporting aggregation.

    MONTHLY: Calendar month aggregation.
    QUARTERLY: Calendar quarter (Q1-Q4) aggregation.
    ANNUAL: Full calendar or fiscal year aggregation.
    """

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class RegulatoryFramework(str, Enum):
    """Regulatory framework governing calculation methodology and reporting.

    GHG_PROTOCOL: WRI/WBCSD Corporate GHG Protocol.
    ISO_14064: ISO 14064-1 Organizational Level GHG Quantification.
    CSRD_ESRS_E1: EU Corporate Sustainability Reporting Directive,
        European Sustainability Reporting Standard E1 (Climate Change).
    EPA_40CFR98: US EPA Mandatory Greenhouse Gas Reporting Rule.
    UK_SECR: UK Streamlined Energy and Carbon Reporting.
    EU_ETS: European Union Emissions Trading System.
    """

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS_E1 = "csrd_esrs_e1"
    EPA_40CFR98 = "epa_40cfr98"
    UK_SECR = "uk_secr"
    EU_ETS = "eu_ets"


class UnitType(str, Enum):
    """Physical units for fuel quantity measurement and energy content.

    Covers volume, mass, and energy units commonly used in stationary
    combustion fuel records. The calculation engine normalizes all inputs
    to a common energy basis (GJ) before applying emission factors.
    """

    # Volume units
    LITERS = "liters"
    GALLONS = "gallons"
    CUBIC_METERS = "cubic_meters"
    CUBIC_FEET = "cubic_feet"
    BARRELS = "barrels"
    # Mass units
    KG = "kg"
    TONNES = "tonnes"
    LBS = "lbs"
    SHORT_TONS = "short_tons"
    # Energy units
    KWH = "kwh"
    MWH = "mwh"
    GJ = "gj"
    MMBTU = "mmbtu"
    THERMS = "therms"
    # Gas-specific volume units
    MCF = "mcf"
    SCF = "scf"


# =============================================================================
# Data Models (12)
# =============================================================================


class EmissionFactor(BaseModel):
    """A single emission factor record for a specific fuel-gas combination.

    Emission factors define the mass of GHG released per unit of fuel
    consumed or energy produced. Each record is scoped to a specific
    fuel type, greenhouse gas, EF source authority, tier level, and
    geographic jurisdiction.

    Attributes:
        factor_id: Unique identifier for this emission factor record.
        fuel_type: Fuel type this factor applies to.
        gas: Greenhouse gas species this factor quantifies.
        value: Emission factor numeric value (mass GHG per fuel/energy unit).
        unit: Unit of measurement for the factor (e.g. kg CO2/GJ,
            g CH4/mmBtu).
        source: Authority that published this emission factor.
        tier: GHG Protocol calculation tier this factor is appropriate for.
        geography: ISO 3166 country/region code or "GLOBAL" for
            geographically unscoped factors.
        effective_date: Date from which this factor is valid.
        expiry_date: Date after which this factor is superseded.
        reference: Bibliographic reference or document ID for audit trails.
        notes: Optional human-readable notes about applicability or
            limitations.
    """

    factor_id: str = Field(
        default_factory=lambda: f"ef_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this emission factor record",
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
        description="Unit of measurement (e.g. kg CO2/GJ)",
    )
    source: EFSource = Field(
        default=EFSource.EPA,
        description="Authority that published this emission factor",
    )
    tier: CalculationTier = Field(
        default=CalculationTier.TIER_1,
        description="Calculation tier this factor is appropriate for",
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


class FuelProperties(BaseModel):
    """Physical and chemical properties of a fuel type.

    Defines the heating values, density, carbon content, and oxidation
    characteristics needed for GHG emission calculations. Each fuel type
    has both HHV and NCV heating values to support different framework
    conventions.

    Attributes:
        fuel_type: Fuel type these properties describe.
        category: Broad fuel classification (gaseous, liquid, solid, biomass).
        hhv: Higher Heating Value (gross calorific value).
        hhv_unit: Unit for HHV (e.g. GJ/tonne, mmBtu/bbl).
        ncv: Net Calorific Value (lower heating value).
        ncv_unit: Unit for NCV (e.g. GJ/tonne, mmBtu/bbl).
        density: Fuel density for volume-to-mass conversion (kg/L or kg/m3).
        carbon_content: Mass fraction of carbon in the fuel (0.0-1.0).
        oxidation_factor: Fraction of carbon oxidized during combustion.
        is_biogenic: Whether emissions from this fuel are classified as
            biogenic (reported separately per GHG Protocol).
        ipcc_code: IPCC fuel code from 2006 Guidelines for cross-reference.
    """

    fuel_type: FuelType = Field(
        ...,
        description="Fuel type these properties describe",
    )
    category: FuelCategory = Field(
        ...,
        description="Broad fuel classification",
    )
    hhv: float = Field(
        ...,
        gt=0,
        description="Higher Heating Value (gross calorific value)",
    )
    hhv_unit: str = Field(
        ...,
        min_length=1,
        description="Unit for HHV (e.g. GJ/tonne, mmBtu/bbl)",
    )
    ncv: float = Field(
        ...,
        gt=0,
        description="Net Calorific Value (lower heating value)",
    )
    ncv_unit: str = Field(
        ...,
        min_length=1,
        description="Unit for NCV (e.g. GJ/tonne, mmBtu/bbl)",
    )
    density: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fuel density for volume-to-mass conversion (kg/L or kg/m3)",
    )
    carbon_content: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Mass fraction of carbon in the fuel (0.0-1.0)",
    )
    oxidation_factor: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Fraction of carbon oxidized during combustion (0.0-1.0)",
    )
    is_biogenic: bool = Field(
        default=False,
        description="Whether emissions are classified as biogenic",
    )
    ipcc_code: Optional[str] = Field(
        default=None,
        description="IPCC 2006 GL fuel code for cross-reference",
    )


class EquipmentProfile(BaseModel):
    """Operational profile for a stationary combustion equipment unit.

    Equipment profiles enable Tier 2/3 calculations by incorporating
    equipment-specific efficiency curves, age degradation, maintenance
    status, and load factor data into the emission calculation.

    Attributes:
        equipment_id: Unique identifier for this equipment unit.
        equipment_type: Classification of the combustion equipment.
        name: Human-readable name or asset tag for the equipment.
        rated_capacity_mmbtu_hr: Nameplate rated capacity in mmBtu/hr.
        efficiency_curve_coefficients: Polynomial coefficients for the
            efficiency curve as a function of load factor. Ordered from
            constant term to highest degree.
        age_years: Age of the equipment in years since installation.
        maintenance_status: Current maintenance status (good, fair, poor).
        load_factor_range: Tuple of (min, max) operational load factor
            as fractions of rated capacity (0.0-1.0).
    """

    equipment_id: str = Field(
        default_factory=lambda: f"eq_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this equipment unit",
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Classification of the combustion equipment",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name or asset tag",
    )
    rated_capacity_mmbtu_hr: Optional[float] = Field(
        default=None,
        gt=0,
        description="Nameplate rated capacity in mmBtu/hr",
    )
    efficiency_curve_coefficients: List[float] = Field(
        default_factory=list,
        max_length=MAX_EFFICIENCY_COEFFICIENTS,
        description="Polynomial efficiency curve coefficients (constant to highest degree)",
    )
    age_years: Optional[int] = Field(
        default=None,
        ge=0,
        description="Age of equipment in years since installation",
    )
    maintenance_status: Optional[str] = Field(
        default=None,
        description="Current maintenance status (good, fair, poor)",
    )
    load_factor_range: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Min/max operational load factor as fractions (0.0-1.0)",
    )

    @field_validator("load_factor_range")
    @classmethod
    def validate_load_factor_range(
        cls, v: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Validate load factor range bounds are within [0.0, 1.0] and ordered."""
        if v is not None:
            low, high = v
            if not (0.0 <= low <= 1.0):
                raise ValueError(
                    f"Load factor minimum must be in [0.0, 1.0], got {low}"
                )
            if not (0.0 <= high <= 1.0):
                raise ValueError(
                    f"Load factor maximum must be in [0.0, 1.0], got {high}"
                )
            if low > high:
                raise ValueError(
                    f"Load factor minimum ({low}) must be <= maximum ({high})"
                )
        return v

    @field_validator("maintenance_status")
    @classmethod
    def validate_maintenance_status(
        cls, v: Optional[str]
    ) -> Optional[str]:
        """Normalize and validate maintenance status to allowed values."""
        if v is not None:
            normalised = v.strip().lower()
            if normalised not in ("good", "fair", "poor"):
                raise ValueError(
                    f"maintenance_status must be good, fair, or poor, "
                    f"got '{v}'"
                )
            return normalised
        return v


class CombustionInput(BaseModel):
    """Input data for a single stationary combustion emission calculation.

    Represents one fuel consumption record for a specific time period,
    optionally linked to a specific piece of equipment and facility.
    The calculation engine uses this input together with emission factors
    and fuel properties to compute GHG emissions.

    Attributes:
        fuel_type: Type of fuel consumed.
        quantity: Amount of fuel consumed (must be > 0).
        unit: Measurement unit for the fuel quantity.
        equipment_id: Optional link to a specific equipment profile for
            Tier 2/3 calculations.
        facility_id: Optional facility identifier for aggregation.
        source_id: Optional source identifier for granular tracking.
        period_start: Start of the consumption reporting period.
        period_end: End of the consumption reporting period.
        heating_value_basis: Whether the fuel heating value is HHV or NCV.
        custom_heating_value: Optional override for the default heating
            value (in GJ per native unit).
        custom_emission_factor: Optional override emission factor
            (in kg CO2e per GJ).
        custom_oxidation_factor: Optional override oxidation factor (0-1).
        tier: Optional override of the default calculation tier.
        geography: Optional ISO 3166 code for region-specific factors.
    """

    fuel_type: FuelType = Field(
        ...,
        description="Type of fuel consumed",
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Amount of fuel consumed (must be > 0)",
    )
    unit: UnitType = Field(
        ...,
        description="Measurement unit for the fuel quantity",
    )
    equipment_id: Optional[str] = Field(
        default=None,
        description="Optional link to a specific equipment profile",
    )
    facility_id: Optional[str] = Field(
        default=None,
        description="Optional facility identifier for aggregation",
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Optional source identifier for granular tracking",
    )
    period_start: datetime = Field(
        ...,
        description="Start of the consumption reporting period",
    )
    period_end: datetime = Field(
        ...,
        description="End of the consumption reporting period",
    )
    heating_value_basis: HeatingValueBasis = Field(
        default=HeatingValueBasis.HHV,
        description="Whether the fuel heating value is HHV or NCV",
    )
    custom_heating_value: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional override heating value (GJ per native unit)",
    )
    custom_emission_factor: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional override emission factor (kg CO2e per GJ)",
    )
    custom_oxidation_factor: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional override oxidation factor (0.0-1.0)",
    )
    tier: Optional[CalculationTier] = Field(
        default=None,
        description="Optional override of the default calculation tier",
    )
    geography: Optional[str] = Field(
        default=None,
        description="Optional ISO 3166 code for region-specific factors",
    )

    @field_validator("period_end")
    @classmethod
    def period_end_after_start(
        cls, v: datetime, info: Any
    ) -> datetime:
        """Validate that period_end is after period_start."""
        period_start = info.data.get("period_start")
        if period_start is not None and v <= period_start:
            raise ValueError(
                "period_end must be after period_start"
            )
        return v


class GasEmission(BaseModel):
    """Emission result for a single greenhouse gas from a combustion event.

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
    """Complete result of a single stationary combustion emission calculation.

    Contains all calculated emissions by gas, total CO2e, biogenic CO2
    (if applicable), the methodology parameters used, and a SHA-256
    provenance hash for audit trail integrity.

    Attributes:
        calculation_id: Unique identifier for this calculation result.
        fuel_type: Fuel type that was combusted.
        equipment_type: Equipment type used (if equipment-level calc).
        fuel_quantity: Original fuel quantity consumed.
        fuel_unit: Unit of the fuel quantity.
        energy_gj: Fuel energy content in gigajoules.
        heating_value_used: Heating value applied (GJ per native unit).
        heating_value_basis: Whether HHV or NCV was used.
        oxidation_factor_used: Oxidation factor applied to the calculation.
        tier_used: Calculation tier applied (TIER_1, TIER_2, TIER_3).
        emissions_by_gas: Itemized emissions for each greenhouse gas.
        total_co2e_kg: Total CO2-equivalent emissions in kilograms.
        total_co2e_tonnes: Total CO2-equivalent emissions in metric tonnes.
        biogenic_co2_kg: Biogenic CO2 emissions in kilograms (reported
            separately per GHG Protocol guidance).
        biogenic_co2_tonnes: Biogenic CO2 emissions in metric tonnes.
        regulatory_framework: Framework governing this calculation.
        provenance_hash: SHA-256 hash for audit trail integrity.
        calculation_trace: Ordered list of human-readable calculation steps.
        timestamp: UTC timestamp when the calculation was performed.
        facility_id: Facility identifier (if provided in input).
        source_id: Source identifier (if provided in input).
        period_start: Start of the consumption reporting period.
        period_end: End of the consumption reporting period.
    """

    calculation_id: str = Field(
        default_factory=lambda: f"calc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this calculation result",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Fuel type that was combusted",
    )
    equipment_type: Optional[EquipmentType] = Field(
        default=None,
        description="Equipment type used (if equipment-level calc)",
    )
    fuel_quantity: float = Field(
        ...,
        gt=0,
        description="Original fuel quantity consumed",
    )
    fuel_unit: UnitType = Field(
        ...,
        description="Unit of the fuel quantity",
    )
    energy_gj: float = Field(
        ...,
        ge=0,
        description="Fuel energy content in gigajoules",
    )
    heating_value_used: float = Field(
        ...,
        gt=0,
        description="Heating value applied (GJ per native unit)",
    )
    heating_value_basis: HeatingValueBasis = Field(
        ...,
        description="Whether HHV or NCV was used",
    )
    oxidation_factor_used: float = Field(
        ...,
        ge=0,
        le=1,
        description="Oxidation factor applied to the calculation",
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
    regulatory_framework: Optional[RegulatoryFramework] = Field(
        default=None,
        description="Framework governing this calculation",
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
    facility_id: Optional[str] = Field(
        default=None,
        description="Facility identifier (if provided in input)",
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Source identifier (if provided in input)",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the consumption reporting period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the consumption reporting period",
    )


class BatchCalculationRequest(BaseModel):
    """Request model for batch stationary combustion calculations.

    Groups multiple combustion inputs for processing as a single
    batch, sharing common parameters like GWP source, biogenic
    tracking preference, and organizational context.

    Attributes:
        calculations: List of individual combustion inputs to process.
        gwp_source: IPCC Assessment Report for GWP values (AR4/AR5/AR6).
        include_biogenic: Whether to include biogenic CO2 in tracking.
        control_approach: Organizational boundary approach.
        organization_id: Organization identifier for aggregation.
        reporting_period: Temporal granularity for the batch.
    """

    calculations: List[CombustionInput] = Field(
        ...,
        min_length=1,
        max_length=MAX_CALCULATIONS_PER_BATCH,
        description="List of individual combustion inputs to process",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    include_biogenic: bool = Field(
        default=True,
        description="Whether to include biogenic CO2 in tracking",
    )
    control_approach: ControlApproach = Field(
        default=ControlApproach.OPERATIONAL,
        description="Organizational boundary approach",
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
    """Response model for a batch stationary combustion calculation.

    Aggregates individual calculation results with batch-level totals,
    emissions breakdown by fuel type, and processing metadata.

    Attributes:
        success: Whether all calculations in the batch succeeded.
        results: List of individual calculation results.
        total_co2e_tonnes: Batch total CO2-equivalent in metric tonnes.
        total_co2_tonnes: Batch total CO2 in metric tonnes.
        total_ch4_tonnes: Batch total CH4 in metric tonnes (CO2e).
        total_n2o_tonnes: Batch total N2O in metric tonnes (CO2e).
        total_biogenic_co2_tonnes: Batch total biogenic CO2 in tonnes.
        emissions_by_fuel: Emissions aggregated by fuel type (tCO2e).
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
    total_ch4_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch total CH4 in metric tonnes (CO2e)",
    )
    total_n2o_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch total N2O in metric tonnes (CO2e)",
    )
    total_biogenic_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch total biogenic CO2 in metric tonnes",
    )
    emissions_by_fuel: Dict[str, float] = Field(
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


class UncertaintyResult(BaseModel):
    """Monte Carlo uncertainty quantification result for an emission calculation.

    Provides statistical characterization of emission estimate uncertainty
    including mean, standard deviation, confidence intervals at multiple
    levels, and contribution analysis showing which input parameters
    drive the most uncertainty.

    Attributes:
        mean_co2e: Mean CO2-equivalent emission estimate (tonnes).
        std_dev: Standard deviation of the CO2e estimate (tonnes).
        coefficient_of_variation: CV = std_dev / mean (dimensionless).
        confidence_intervals: Confidence intervals keyed by level string
            (e.g. "90" -> (lower, upper) in tonnes CO2e).
        iterations: Number of Monte Carlo iterations performed.
        data_quality_score: Overall data quality indicator (1-5 scale,
            per GHG Protocol uncertainty guidance).
        tier: Calculation tier used for this uncertainty analysis.
        contributions: Parameter contribution to total variance, keyed
            by parameter name (e.g. "emission_factor" -> 0.45 meaning
            45% of variance).
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
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Confidence intervals keyed by level (e.g. '90' -> (lower, upper))",
    )
    iterations: int = Field(
        ...,
        gt=0,
        description="Number of Monte Carlo iterations performed",
    )
    data_quality_score: Optional[float] = Field(
        default=None,
        ge=1,
        le=5,
        description="Data quality indicator (1-5 scale, GHG Protocol guidance)",
    )
    tier: CalculationTier = Field(
        ...,
        description="Calculation tier used for this uncertainty analysis",
    )
    contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameter contribution to total variance (name -> fraction)",
    )


class FacilityAggregation(BaseModel):
    """Facility-level emission aggregation across all combustion sources.

    Rolls up individual calculation results into a facility total
    broken down by gas, including biogenic CO2 separation. Used for
    organizational GHG inventory reporting at the facility level.

    Attributes:
        facility_id: Unique facility identifier.
        organization_id: Parent organization identifier.
        control_approach: Boundary approach used for this aggregation.
        reporting_period_type: Temporal granularity of the aggregation.
        period_start: Start of the aggregation period.
        period_end: End of the aggregation period.
        total_co2e_tonnes: Facility total CO2-equivalent (metric tonnes).
        total_co2_tonnes: Facility total CO2 (metric tonnes).
        total_ch4_tonnes: Facility total CH4 in CO2e (metric tonnes).
        total_n2o_tonnes: Facility total N2O in CO2e (metric tonnes).
        biogenic_co2_tonnes: Biogenic CO2 total (metric tonnes, reported
            separately).
        calculation_count: Number of calculations in this aggregation.
        equipment_count: Number of distinct equipment units included.
        fuel_types_used: List of distinct fuel types in this aggregation.
        provenance_hash: SHA-256 hash for audit trail integrity.
    """

    facility_id: str = Field(
        ...,
        min_length=1,
        description="Unique facility identifier",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Parent organization identifier",
    )
    control_approach: ControlApproach = Field(
        default=ControlApproach.OPERATIONAL,
        description="Boundary approach used for this aggregation",
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
        description="Facility total CO2-equivalent (metric tonnes)",
    )
    total_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Facility total CO2 (metric tonnes)",
    )
    total_ch4_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Facility total CH4 in CO2e (metric tonnes)",
    )
    total_n2o_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Facility total N2O in CO2e (metric tonnes)",
    )
    biogenic_co2_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Biogenic CO2 total (metric tonnes, reported separately)",
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations in this aggregation",
    )
    equipment_count: int = Field(
        default=0,
        ge=0,
        description="Number of distinct equipment units included",
    )
    fuel_types_used: List[str] = Field(
        default_factory=list,
        description="List of distinct fuel types in this aggregation",
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
            raise ValueError(
                "period_end must be after period_start"
            )
        return v


class AuditEntry(BaseModel):
    """A single step in the calculation audit trail.

    Records the input, output, and methodology reference for one
    discrete step in a combustion emission calculation. The ordered
    collection of AuditEntry records forms a complete, reproducible
    calculation trace.

    Attributes:
        entry_id: Unique identifier for this audit entry.
        calculation_id: Parent calculation this entry belongs to.
        step_number: Ordinal position of this step in the calculation.
        step_name: Human-readable name of the calculation step.
        input_data: Input values consumed by this step.
        output_data: Output values produced by this step.
        emission_factor_used: Emission factor applied in this step
            (if applicable).
        methodology_reference: Regulatory or methodological citation
            for this step (e.g. "GHG Protocol Ch. 3, Eq. 3.1").
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


class ComplianceMapping(BaseModel):
    """Mapping of a regulatory requirement to how the agent satisfies it.

    Tracks how the Stationary Combustion Agent meets specific
    requirements from each supported regulatory framework, providing
    auditable evidence of compliance.

    Attributes:
        framework: Regulatory framework this mapping addresses.
        requirement_id: Identifier of the specific requirement within
            the framework (e.g. "GHG-3.1", "ESRS-E1-DR-E1-6").
        requirement_description: Human-readable description of what
            the requirement mandates.
        how_met: Explanation of how the agent satisfies the requirement.
        evidence_reference: Pointer to evidence (calculation ID, audit
            entry, configuration parameter, etc.).
        status: Current compliance status (met, partially_met, not_met,
            not_applicable).
    """

    framework: RegulatoryFramework = Field(
        ...,
        description="Regulatory framework this mapping addresses",
    )
    requirement_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the specific requirement",
    )
    requirement_description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of the requirement",
    )
    how_met: str = Field(
        ...,
        min_length=1,
        description="Explanation of how the agent satisfies the requirement",
    )
    evidence_reference: Optional[str] = Field(
        default=None,
        description="Pointer to evidence (calculation ID, audit entry, etc.)",
    )
    status: str = Field(
        default="met",
        description="Compliance status (met, partially_met, not_met, not_applicable)",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Normalize and validate compliance status."""
        normalised = v.strip().lower()
        valid_statuses = {"met", "partially_met", "not_met", "not_applicable"}
        if normalised not in valid_statuses:
            raise ValueError(
                f"status must be one of {sorted(valid_statuses)}, "
                f"got '{v}'"
            )
        return normalised


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_EFFICIENCY_COEFFICIENTS",
    "DEFAULT_OXIDATION_FACTOR",
    "GWP_VALUES",
    # Enums
    "FuelCategory",
    "FuelType",
    "EmissionGas",
    "GWPSource",
    "EFSource",
    "CalculationTier",
    "EquipmentType",
    "HeatingValueBasis",
    "ControlApproach",
    "CalculationStatus",
    "ReportingPeriod",
    "RegulatoryFramework",
    "UnitType",
    # Data models
    "EmissionFactor",
    "FuelProperties",
    "EquipmentProfile",
    "CombustionInput",
    "GasEmission",
    "CalculationResult",
    "BatchCalculationRequest",
    "BatchCalculationResponse",
    "UncertaintyResult",
    "FacilityAggregation",
    "AuditEntry",
    "ComplianceMapping",
]
