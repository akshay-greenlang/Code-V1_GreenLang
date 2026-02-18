# -*- coding: utf-8 -*-
"""
Process Emissions Agent Data Models - AGENT-MRV-004

Pydantic v2 data models for the Process Emissions Agent SDK covering
GHG Protocol Scope 1 non-combustion industrial process emissions including:
- 25 industrial process types across 6 categories (mineral, chemical,
  metal, electronics, pulp & paper, other)
- 8 greenhouse gas species (CO2, CH4, N2O, CF4, C2F6, SF6, NF3, HFC)
  with process-specific gas profiles
- 4 calculation methodologies (emission factor, mass balance,
  stoichiometric, direct measurement)
- Tier 1/2/3 IPCC calculation tiers
- Carbonate decomposition factors for mineral process emissions
- GWP values from AR4, AR5, AR6, and AR6-20yr timeframes
- Production route tracking for iron/steel and aluminum smelting
- Abatement tracking (catalytic reduction, carbon capture, scrubbing, etc.)
- Multi-framework compliance checking (GHG Protocol, ISO 14064, CSRD,
  EPA, UK SECR, EU ETS)

Enumerations (16):
    - ProcessCategory, ProcessType, EmissionGas, CalculationMethod,
      CalculationTier, EmissionFactorSource, GWPSource, MaterialType,
      AbatementType, ProcessUnitType, ProcessMode, ComplianceStatus,
      ReportingPeriod, UnitType, ProductionRoute, CarbonateType

Data Models (17):
    - ProcessTypeInfo, RawMaterialInfo, EmissionFactorRecord,
      ProcessUnitRecord, MaterialInputRecord, CalculationRequest,
      GasEmissionResult, CalculationResult, CalculationDetailResult,
      AbatementRecord, ComplianceCheckResult, BatchCalculationRequest,
      BatchCalculationResult, UncertaintyRequest, UncertaintyResult,
      AggregationRequest, AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
MAX_GASES_PER_RESULT: int = 20

#: Maximum number of trace steps in a single calculation.
MAX_TRACE_STEPS: int = 200

#: Maximum number of material inputs per single calculation.
MAX_MATERIAL_INPUTS_PER_CALC: int = 50


# =============================================================================
# Enumerations (16)
# =============================================================================


class ProcessCategory(str, Enum):
    """Broad classification of industrial processes by sector.

    Used to group process types for reporting aggregation and to determine
    applicable default emission factors and regulatory requirements.

    MINERAL: Cement, lime, glass, ceramics, soda ash, and other mineral
        processes involving carbonate decomposition.
    CHEMICAL: Ammonia, nitric acid, adipic acid, carbide, petrochemical,
        hydrogen, phosphoric acid, and titanium dioxide production.
    METAL: Iron & steel, aluminum smelting, ferroalloy, lead, zinc,
        magnesium, and copper production.
    ELECTRONICS: Semiconductor manufacturing and related processes
        using fluorinated gases.
    PULP_PAPER: Pulp and paper production processes.
    OTHER: Mineral wool, carbon anode baking, food & drink, and
        processes not classified elsewhere.
    """

    MINERAL = "mineral"
    CHEMICAL = "chemical"
    METAL = "metal"
    ELECTRONICS = "electronics"
    PULP_PAPER = "pulp_paper"
    OTHER = "other"


class ProcessType(str, Enum):
    """Specific industrial process type identifiers for Scope 1 process emissions.

    Covers all major non-combustion industrial processes encountered in
    Scope 1 GHG inventories. Each process type has associated default
    emission factors, applicable gases, and regulatory references.

    Naming follows IPCC 2006 Guidelines Volume 3 (Industrial Processes
    and Product Use) and EPA 40 CFR Part 98 Subpart conventions for
    cross-framework compatibility.
    """

    # Mineral processes (IPCC Vol 3, Ch 2)
    CEMENT_PRODUCTION = "cement_production"
    LIME_PRODUCTION = "lime_production"
    GLASS_PRODUCTION = "glass_production"
    CERAMICS = "ceramics"
    SODA_ASH = "soda_ash"

    # Chemical processes (IPCC Vol 3, Ch 3)
    AMMONIA_PRODUCTION = "ammonia_production"
    NITRIC_ACID = "nitric_acid"
    ADIPIC_ACID = "adipic_acid"
    CARBIDE_PRODUCTION = "carbide_production"
    PETROCHEMICAL = "petrochemical"
    HYDROGEN_PRODUCTION = "hydrogen_production"
    PHOSPHORIC_ACID = "phosphoric_acid"
    TITANIUM_DIOXIDE = "titanium_dioxide"

    # Metal processes (IPCC Vol 3, Ch 4)
    IRON_STEEL = "iron_steel"
    ALUMINUM_SMELTING = "aluminum_smelting"
    FERROALLOY = "ferroalloy"
    LEAD_PRODUCTION = "lead_production"
    ZINC_PRODUCTION = "zinc_production"
    MAGNESIUM_PRODUCTION = "magnesium_production"
    COPPER_SMELTING = "copper_smelting"

    # Electronics (IPCC Vol 3, Ch 6)
    SEMICONDUCTOR = "semiconductor"

    # Pulp & paper
    PULP_PAPER = "pulp_paper"

    # Other
    MINERAL_WOOL = "mineral_wool"
    CARBON_ANODE = "carbon_anode"
    FOOD_DRINK = "food_drink"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in industrial process emission calculations.

    CO2: Carbon dioxide - primary process emission from carbonate
        decomposition, chemical reactions, and carbon electrode consumption.
    CH4: Methane - by-product of some chemical and metal processes.
    N2O: Nitrous oxide - major emission from nitric acid and adipic acid
        production.
    CF4: Carbon tetrafluoride (PFC-14) - emitted during aluminum smelting
        anode effects and semiconductor manufacturing.
    C2F6: Hexafluoroethane (PFC-116) - emitted during aluminum smelting
        anode effects and semiconductor manufacturing.
    SF6: Sulfur hexafluoride - used as cover gas in magnesium production
        and in semiconductor manufacturing.
    NF3: Nitrogen trifluoride - used in semiconductor chamber cleaning.
    HFC: Hydrofluorocarbons (aggregate) - used in semiconductor and
        electronics processes.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CF4 = "CF4"
    C2F6 = "C2F6"
    SF6 = "SF6"
    NF3 = "NF3"
    HFC = "HFC"


class CalculationMethod(str, Enum):
    """Methodology for calculating process emissions.

    EMISSION_FACTOR: Applies published emission factors to activity data
        (production quantities, material inputs). Simplest approach.
    MASS_BALANCE: Tracks carbon mass across all inputs and outputs to
        derive emissions by difference. Requires comprehensive material
        flow data. Preferred for chemical and metal processes.
    STOICHIOMETRIC: Uses balanced chemical reaction equations to calculate
        theoretical emissions from input quantities. Applicable to
        well-characterized mineral decomposition reactions.
    DIRECT_MEASUREMENT: Uses continuous emission monitoring systems (CEMS)
        or periodic stack testing to directly measure emissions. Highest
        accuracy but most data-intensive.
    """

    EMISSION_FACTOR = "EMISSION_FACTOR"
    MASS_BALANCE = "MASS_BALANCE"
    STOICHIOMETRIC = "STOICHIOMETRIC"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"


class CalculationTier(str, Enum):
    """IPCC calculation methodology tier level for process emissions.

    TIER_1: Default emission factors from IPCC or national inventories.
        Uses production-based factors (e.g. tCO2 per tonne clinker).
        Lowest accuracy, simplest data requirements.
    TIER_2: Country-specific or process-specific emission factors.
        May include plant-specific material composition data. Moderate
        accuracy and data requirements.
    TIER_3: Facility-level measurements, mass balance with plant-specific
        data, or CEMS. Highest accuracy, most data-intensive. Required
        for EU ETS installations above threshold.
    """

    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class EmissionFactorSource(str, Enum):
    """Source authority for process emission factor values.

    EPA: US Environmental Protection Agency (40 CFR Part 98, Subparts F-Z).
    IPCC: IPCC 2006 Guidelines for National GHG Inventories, Volume 3.
    DEFRA: UK Department for Environment, Food and Rural Affairs conversion
        factors for company reporting.
    EU_ETS: European Union Emissions Trading System Monitoring and Reporting
        Regulation (EU MRR) Annex IV factors.
    CUSTOM: Organization-specific or facility-measured factors.
    """

    EPA = "EPA"
    IPCC = "IPCC"
    DEFRA = "DEFRA"
    EU_ETS = "EU_ETS"
    CUSTOM = "CUSTOM"


class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for GWP conversion factors.

    AR4: Fourth Assessment Report (2007). GWP-100yr.
    AR5: Fifth Assessment Report (2014). GWP-100yr.
    AR6: Sixth Assessment Report (2021). GWP-100yr.
    AR6_20YR: Sixth Assessment Report (2021). GWP-20yr timeframe.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class MaterialType(str, Enum):
    """Raw material types used as inputs in industrial process calculations.

    Covers carbonates, oxides, ores, fuels-as-feedstock, and other
    materials whose chemical transformation releases greenhouse gases.
    Material composition data is critical for mass balance and
    stoichiometric calculation methods.
    """

    # Carbonates (decompose to release CO2)
    CALCIUM_CARBONATE = "calcium_carbonate"
    MAGNESIUM_CARBONATE = "magnesium_carbonate"
    IRON_CARBONATE = "iron_carbonate"

    # Oxides and hydroxides
    CALCIUM_OXIDE = "calcium_oxide"
    CALCIUM_HYDROXIDE = "calcium_hydroxide"

    # Cement/clinker materials
    CLINKER = "clinker"
    LIMESTONE = "limestone"
    DOLOMITE = "dolomite"
    CHALK = "chalk"

    # Metal production materials
    CITE = "cite"
    BAUXITE = "bauxite"
    ALUMINA = "alumina"
    IRON_ORE = "iron_ore"
    SCRAP_METAL = "scrap_metal"

    # Carbon sources
    COKE = "coke"
    COAL = "coal"

    # Petrochemical feedstocks
    NATURAL_GAS_FEEDSTOCK = "natural_gas_feedstock"
    NAPHTHA = "naphtha"
    ETHANE = "ethane"

    # Catch-all
    OTHER = "other"


class AbatementType(str, Enum):
    """Types of emission abatement measures for industrial processes.

    Abatement technologies reduce gross process emissions. The abatement
    efficiency (fraction of emissions eliminated) varies by technology,
    operating conditions, and maintenance status. Tracking abatement
    enables accurate net emission reporting and compliance verification.
    """

    CATALYTIC_REDUCTION = "catalytic_reduction"
    THERMAL_DESTRUCTION = "thermal_destruction"
    SCRUBBING = "scrubbing"
    CARBON_CAPTURE = "carbon_capture"
    PFC_ANODE_CONTROL = "pfc_anode_control"
    SF6_RECOVERY = "sf6_recovery"
    NSCR = "nscr"
    SCR = "scr"
    EXTENDED_ABSORPTION = "extended_absorption"
    OTHER = "other"


class ProcessUnitType(str, Enum):
    """Classification of industrial process equipment units.

    Process unit type determines applicable emission factors, abatement
    options, and regulatory reporting requirements. Equipment-level
    tracking supports Tier 2/3 calculations and facility-level
    aggregation.
    """

    KILN = "kiln"
    FURNACE = "furnace"
    SMELTER = "smelter"
    REACTOR = "reactor"
    ELECTROLYSIS_CELL = "electrolysis_cell"
    REFORMER = "reformer"
    CONVERTER = "converter"
    CALCINER = "calciner"
    DRYER = "dryer"
    OTHER = "other"


class ProcessMode(str, Enum):
    """Operating mode of an industrial process unit.

    BATCH: Discrete batches with start/stop cycles.
    CONTINUOUS: Steady-state operation with continuous feed/output.
    SEMI_CONTINUOUS: Hybrid mode with periodic charging and continuous
        discharge or vice versa.
    """

    BATCH = "batch"
    CONTINUOUS = "continuous"
    SEMI_CONTINUOUS = "semi_continuous"


class ComplianceStatus(str, Enum):
    """Compliance check result status for a regulatory framework.

    COMPLIANT: All applicable requirements are fully met.
    NON_COMPLIANT: One or more requirements are not met.
    PARTIAL: Some requirements are met, others require attention.
    NOT_CHECKED: Compliance has not been evaluated against this framework.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_CHECKED = "not_checked"


class ReportingPeriod(str, Enum):
    """Temporal granularity for emission reporting aggregation.

    MONTHLY: Calendar month aggregation.
    QUARTERLY: Calendar quarter (Q1-Q4) aggregation.
    ANNUAL: Full calendar or fiscal year aggregation.
    """

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class UnitType(str, Enum):
    """Physical unit categories for material quantities and production output.

    MASS: Weight-based units (kg, tonnes, lbs, short tons).
    VOLUME: Volumetric units (liters, cubic meters, gallons).
    ENERGY: Energy content units (GJ, MWh, mmBtu).
    AREA: Surface area units (m2, ft2).
    COUNT: Discrete units (pieces, wafers, batches).
    """

    MASS = "mass"
    VOLUME = "volume"
    ENERGY = "energy"
    AREA = "area"
    COUNT = "count"


class ProductionRoute(str, Enum):
    """Production route for processes with multiple manufacturing pathways.

    Iron and steel production routes:
        BF_BOF: Blast Furnace - Basic Oxygen Furnace (integrated steelmaking).
        EAF: Electric Arc Furnace (scrap-based steelmaking).
        DRI: Direct Reduced Iron (gas or coal-based reduction).
        OHF: Open Hearth Furnace (legacy process).

    Aluminum smelting routes:
        PREBAKE: Pre-baked carbon anode technology.
        SODERBERG_VSS: Soderberg Vertical Stud technology.
        SODERBERG_HSS: Soderberg Horizontal Stud technology.
        CWPB: Centre-Worked Pre-Bake technology.
        SWPB: Side-Worked Pre-Bake technology.
    """

    # Iron & steel
    BF_BOF = "bf_bof"
    EAF = "eaf"
    DRI = "dri"
    OHF = "ohf"

    # Aluminum
    PREBAKE = "prebake"
    SODERBERG_VSS = "soderberg_vss"
    SODERBERG_HSS = "soderberg_hss"
    CWPB = "cwpb"
    SWPB = "swpb"


class CarbonateType(str, Enum):
    """Carbonate mineral types with known stoichiometric CO2 emission factors.

    CALCITE: Calcium carbonate (CaCO3). Primary mineral in limestone.
    DOLOMITE: Calcium magnesium carbonate (CaMg(CO3)2). Mixed carbonate.
    MAGNESITE: Magnesium carbonate (MgCO3). Used in refractory production.
    SIDERITE: Iron carbonate (FeCO3). Minor carbonate in iron ore.
    ANKERITE: Calcium iron magnesium carbonate (Ca(Fe,Mg)(CO3)2).
    OTHER: Unspecified or mixed carbonate materials.
    """

    CALCITE = "calcite"
    DOLOMITE = "dolomite"
    MAGNESITE = "magnesite"
    SIDERITE = "siderite"
    ANKERITE = "ankerite"
    OTHER = "other"


# =============================================================================
# GWP Values Lookup Table
# =============================================================================

#: Global Warming Potential values for all tracked gases by IPCC AR edition.
#: Units: kg CO2e per kg gas (dimensionless multiplier).
#: Sources:
#:   AR4: IPCC Fourth Assessment Report (2007), Table 2.14.
#:   AR5: IPCC Fifth Assessment Report (2014), Table 8.A.1.
#:   AR6: IPCC Sixth Assessment Report (2021), Table 7.15.
#:   AR6_20YR: AR6 GWP-20yr timeframe, Table 7.15.
GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR4": {
        "CO2": 1.0,
        "CH4": 25.0,
        "N2O": 298.0,
        "CF4": 7390.0,
        "C2F6": 12200.0,
        "SF6": 22800.0,
        "NF3": 17200.0,
        "HFC": 1430.0,
    },
    "AR5": {
        "CO2": 1.0,
        "CH4": 28.0,
        "N2O": 265.0,
        "CF4": 6630.0,
        "C2F6": 11100.0,
        "SF6": 23500.0,
        "NF3": 16100.0,
        "HFC": 1300.0,
    },
    "AR6": {
        "CO2": 1.0,
        "CH4": 29.8,
        "N2O": 273.0,
        "CF4": 7380.0,
        "C2F6": 12400.0,
        "SF6": 25200.0,
        "NF3": 17400.0,
        "HFC": 1530.0,
    },
    "AR6_20YR": {
        "CO2": 1.0,
        "CH4": 82.5,
        "N2O": 273.0,
        "CF4": 5300.0,
        "C2F6": 8940.0,
        "SF6": 18300.0,
        "NF3": 12800.0,
        "HFC": 4140.0,
    },
}


# =============================================================================
# Carbonate Emission Factors Lookup Table
# =============================================================================

#: Stoichiometric CO2 emission factors for carbonate decomposition.
#: Units: tonnes CO2 released per tonne of carbonate mineral.
#: Source: IPCC 2006 Guidelines, Volume 3, Table 2.1.
#:   CaCO3  -> CaO  + CO2    (molar: 100 -> 56 + 44)    => 0.440
#:   CaMg(CO3)2 -> CaO + MgO + 2CO2 (molar: 184 -> 96 + 88) => 0.477
#:   MgCO3  -> MgO  + CO2    (molar: 84.3 -> 40.3 + 44) => 0.522
#:   FeCO3  -> FeO  + CO2    (molar: 115.9 -> 71.9 + 44) => 0.380
#:   Ca(Fe,Mg)(CO3)2 -> oxides + 2CO2 (approx)           => 0.407
CARBONATE_EMISSION_FACTORS: Dict[str, Decimal] = {
    "calcite": Decimal("0.440"),
    "dolomite": Decimal("0.477"),
    "magnesite": Decimal("0.522"),
    "siderite": Decimal("0.380"),
    "ankerite": Decimal("0.407"),
}

#: Mapping from ProcessCategory to the set of applicable ProcessType values.
PROCESS_CATEGORY_MAP: Dict[str, List[str]] = {
    "mineral": [
        "cement_production",
        "lime_production",
        "glass_production",
        "ceramics",
        "soda_ash",
    ],
    "chemical": [
        "ammonia_production",
        "nitric_acid",
        "adipic_acid",
        "carbide_production",
        "petrochemical",
        "hydrogen_production",
        "phosphoric_acid",
        "titanium_dioxide",
    ],
    "metal": [
        "iron_steel",
        "aluminum_smelting",
        "ferroalloy",
        "lead_production",
        "zinc_production",
        "magnesium_production",
        "copper_smelting",
    ],
    "electronics": [
        "semiconductor",
    ],
    "pulp_paper": [
        "pulp_paper",
    ],
    "other": [
        "mineral_wool",
        "carbon_anode",
        "food_drink",
    ],
}

#: Default gases emitted by each process type. Used when specific gas
#: profiles are not provided at the facility level.
PROCESS_DEFAULT_GASES: Dict[str, List[str]] = {
    "cement_production": ["CO2"],
    "lime_production": ["CO2"],
    "glass_production": ["CO2"],
    "ceramics": ["CO2"],
    "soda_ash": ["CO2"],
    "ammonia_production": ["CO2"],
    "nitric_acid": ["N2O"],
    "adipic_acid": ["N2O"],
    "carbide_production": ["CO2", "CH4"],
    "petrochemical": ["CO2", "CH4"],
    "hydrogen_production": ["CO2"],
    "phosphoric_acid": ["CO2"],
    "titanium_dioxide": ["CO2"],
    "iron_steel": ["CO2", "CH4"],
    "aluminum_smelting": ["CO2", "CF4", "C2F6"],
    "ferroalloy": ["CO2", "CH4"],
    "lead_production": ["CO2"],
    "zinc_production": ["CO2"],
    "magnesium_production": ["CO2", "SF6"],
    "copper_smelting": ["CO2"],
    "semiconductor": ["CF4", "C2F6", "SF6", "NF3", "HFC"],
    "pulp_paper": ["CO2"],
    "mineral_wool": ["CO2"],
    "carbon_anode": ["CO2"],
    "food_drink": ["CO2"],
}


# =============================================================================
# Data Models (17)
# =============================================================================


class ProcessTypeInfo(BaseModel):
    """Metadata record describing an industrial process type.

    Provides reference information about a process type including its
    category, applicable gases, default emission factors, and regulatory
    references. Used for process type registration and lookup.

    Attributes:
        process_type: Identifier of the industrial process.
        category: Broad sector classification.
        display_name: Human-readable display name.
        description: Detailed description of the process and its emissions.
        applicable_gases: List of greenhouse gas species emitted by
            this process type.
        ipcc_reference: IPCC 2006 Guidelines chapter and section reference.
        epa_subpart: EPA 40 CFR Part 98 subpart identifier (if applicable).
        default_tier: Recommended default calculation tier.
        default_method: Recommended default calculation method.
        supports_mass_balance: Whether mass balance method is applicable.
        supports_stoichiometric: Whether stoichiometric method is applicable.
    """

    model_config = ConfigDict(frozen=True)

    process_type: ProcessType = Field(
        ...,
        description="Identifier of the industrial process",
    )
    category: ProcessCategory = Field(
        ...,
        description="Broad sector classification",
    )
    display_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable display name",
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed description of the process and its emissions",
    )
    applicable_gases: List[EmissionGas] = Field(
        default_factory=list,
        description="Greenhouse gas species emitted by this process",
    )
    ipcc_reference: Optional[str] = Field(
        default=None,
        description="IPCC 2006 Guidelines chapter and section reference",
    )
    epa_subpart: Optional[str] = Field(
        default=None,
        description="EPA 40 CFR Part 98 subpart identifier",
    )
    default_tier: CalculationTier = Field(
        default=CalculationTier.TIER_1,
        description="Recommended default calculation tier",
    )
    default_method: CalculationMethod = Field(
        default=CalculationMethod.EMISSION_FACTOR,
        description="Recommended default calculation method",
    )
    supports_mass_balance: bool = Field(
        default=False,
        description="Whether mass balance method is applicable",
    )
    supports_stoichiometric: bool = Field(
        default=False,
        description="Whether stoichiometric method is applicable",
    )


class RawMaterialInfo(BaseModel):
    """Reference data for a raw material used in process calculations.

    Attributes:
        material_type: Identifier of the raw material.
        display_name: Human-readable name.
        chemical_formula: Chemical formula (e.g. CaCO3, Al2O3).
        carbon_content_fraction: Mass fraction of carbon (0.0-1.0).
        carbonate_type: Carbonate mineral type if the material is a
            carbonate; None otherwise.
        co2_emission_factor: Stoichiometric CO2 emission factor
            (tonnes CO2 per tonne material), if applicable.
        molecular_weight: Molecular weight in g/mol.
    """

    model_config = ConfigDict(frozen=True)

    material_type: MaterialType = Field(
        ...,
        description="Identifier of the raw material",
    )
    display_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name",
    )
    chemical_formula: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Chemical formula (e.g. CaCO3, Al2O3)",
    )
    carbon_content_fraction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mass fraction of carbon (0.0-1.0)",
    )
    carbonate_type: Optional[CarbonateType] = Field(
        default=None,
        description="Carbonate mineral type if applicable",
    )
    co2_emission_factor: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Stoichiometric CO2 EF (tCO2/t material)",
    )
    molecular_weight: Optional[float] = Field(
        default=None,
        gt=0,
        description="Molecular weight in g/mol",
    )


class EmissionFactorRecord(BaseModel):
    """A single emission factor record for a process-gas combination.

    Emission factors define the mass of GHG released per unit of
    production output or material input. Each record is scoped to a
    specific process type, greenhouse gas, source authority, tier
    level, and geographic jurisdiction.

    Attributes:
        factor_id: Unique identifier for this emission factor record.
        process_type: Industrial process this factor applies to.
        gas: Greenhouse gas species this factor quantifies.
        value: Emission factor numeric value.
        unit: Unit of measurement for the factor (e.g. tCO2/t clinker,
            kg N2O/t HNO3).
        source: Authority that published this emission factor.
        tier: Calculation tier this factor is appropriate for.
        geography: ISO 3166 country/region code or GLOBAL.
        production_route: Production route this factor applies to
            (for iron/steel and aluminum). None for route-agnostic factors.
        effective_date: Date from which this factor is valid.
        expiry_date: Date after which this factor is superseded.
        reference: Bibliographic reference or document ID.
        uncertainty_pct: Percentage uncertainty of this factor (0-100).
        notes: Optional notes about applicability or limitations.
    """

    factor_id: str = Field(
        default_factory=lambda: f"pef_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this emission factor record",
    )
    process_type: ProcessType = Field(
        ...,
        description="Industrial process this factor applies to",
    )
    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species this factor quantifies",
    )
    value: float = Field(
        ...,
        gt=0,
        description="Emission factor numeric value",
    )
    unit: str = Field(
        ...,
        min_length=1,
        description="Unit of measurement (e.g. tCO2/t clinker)",
    )
    source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
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
    production_route: Optional[ProductionRoute] = Field(
        default=None,
        description="Production route if route-specific (iron/steel, aluminum)",
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
    uncertainty_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage uncertainty of this factor (0-100)",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notes about applicability or limitations",
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


class ProcessUnitRecord(BaseModel):
    """Registration record for an industrial process equipment unit.

    Tracks equipment-level metadata needed for Tier 2/3 calculations
    including process type, operating mode, capacity, production route,
    and installed abatement technology.

    Attributes:
        unit_id: Unique identifier for this process unit.
        name: Human-readable name or asset tag.
        process_type: Industrial process operated by this unit.
        unit_type: Equipment classification.
        process_mode: Operating mode (batch, continuous, semi-continuous).
        production_route: Production pathway (for iron/steel, aluminum).
        rated_capacity_tonnes_yr: Annual production capacity in tonnes.
        location: Facility or site identifier.
        installation_year: Year the unit was installed.
        abatement_types: List of installed abatement technologies.
        notes: Optional notes about the unit.
    """

    unit_id: str = Field(
        default_factory=lambda: f"pu_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this process unit",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name or asset tag",
    )
    process_type: ProcessType = Field(
        ...,
        description="Industrial process operated by this unit",
    )
    unit_type: ProcessUnitType = Field(
        ...,
        description="Equipment classification",
    )
    process_mode: ProcessMode = Field(
        default=ProcessMode.CONTINUOUS,
        description="Operating mode",
    )
    production_route: Optional[ProductionRoute] = Field(
        default=None,
        description="Production pathway (for iron/steel, aluminum)",
    )
    rated_capacity_tonnes_yr: Optional[float] = Field(
        default=None,
        gt=0,
        description="Annual production capacity in tonnes",
    )
    location: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Facility or site identifier",
    )
    installation_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Year the unit was installed",
    )
    abatement_types: List[AbatementType] = Field(
        default_factory=list,
        description="List of installed abatement technologies",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notes about the unit",
    )


class MaterialInputRecord(BaseModel):
    """A single material input consumed in a process emission calculation.

    Records the quantity and composition of a raw material input to an
    industrial process. Used by mass balance and stoichiometric
    calculation methods to derive process emissions.

    Attributes:
        material_type: Type of material consumed.
        quantity_tonnes: Quantity consumed in metric tonnes.
        carbon_content_fraction: Mass fraction of carbon in this batch
            (0.0-1.0). Overrides the material default when provided.
        carbonate_type: Carbonate mineral type if the material is a
            carbonate (enables stoichiometric CO2 calculation).
        purity_fraction: Purity of the material (0.0-1.0). Defaults to
            1.0 (100% pure).
        moisture_fraction: Moisture content (0.0-1.0). Defaults to 0.0.
        source_description: Description of the material source or supplier.
    """

    material_type: MaterialType = Field(
        ...,
        description="Type of material consumed",
    )
    quantity_tonnes: float = Field(
        ...,
        gt=0,
        description="Quantity consumed in metric tonnes",
    )
    carbon_content_fraction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mass fraction of carbon (0.0-1.0)",
    )
    carbonate_type: Optional[CarbonateType] = Field(
        default=None,
        description="Carbonate mineral type if applicable",
    )
    purity_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Purity of the material (0.0-1.0)",
    )
    moisture_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Moisture content (0.0-1.0)",
    )
    source_description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Description of the material source or supplier",
    )


class CalculationRequest(BaseModel):
    """Input data for a single process emission calculation.

    Represents one production record for a specific time period and
    industrial process, optionally linked to a specific process unit
    and facility. The calculation engine uses this input together with
    emission factors and material data to compute GHG emissions.

    Attributes:
        process_type: Type of industrial process.
        production_quantity_tonnes: Total production output in tonnes for
            the reporting period.
        unit_id: Optional link to a specific process unit for Tier 2/3.
        facility_id: Optional facility identifier for aggregation.
        period_start: Start of the production reporting period.
        period_end: End of the production reporting period.
        calculation_method: Calculation methodology to apply.
        tier: Calculation tier to apply.
        emission_factor_source: Source authority for emission factors.
        gwp_source: IPCC AR edition for GWP values.
        production_route: Production pathway (for iron/steel, aluminum).
        material_inputs: List of raw material inputs for mass balance
            or stoichiometric methods.
        custom_emission_factors: Optional override emission factors keyed
            by gas (e.g. {"CO2": 0.525, "CH4": 0.001}).
        geography: Optional ISO 3166 code for region-specific factors.
        notes: Optional notes for the calculation record.
    """

    process_type: ProcessType = Field(
        ...,
        description="Type of industrial process",
    )
    production_quantity_tonnes: float = Field(
        ...,
        gt=0,
        description="Total production output in tonnes",
    )
    unit_id: Optional[str] = Field(
        default=None,
        description="Optional link to a specific process unit",
    )
    facility_id: Optional[str] = Field(
        default=None,
        description="Optional facility identifier",
    )
    period_start: datetime = Field(
        ...,
        description="Start of the production reporting period",
    )
    period_end: datetime = Field(
        ...,
        description="End of the production reporting period",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.EMISSION_FACTOR,
        description="Calculation methodology to apply",
    )
    tier: CalculationTier = Field(
        default=CalculationTier.TIER_1,
        description="Calculation tier to apply",
    )
    emission_factor_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="Source authority for emission factors",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC AR edition for GWP values",
    )
    production_route: Optional[ProductionRoute] = Field(
        default=None,
        description="Production pathway (for iron/steel, aluminum)",
    )
    material_inputs: List[MaterialInputRecord] = Field(
        default_factory=list,
        max_length=MAX_MATERIAL_INPUTS_PER_CALC,
        description="Raw material inputs for mass balance/stoichiometric",
    )
    custom_emission_factors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Override emission factors keyed by gas",
    )
    geography: Optional[str] = Field(
        default=None,
        description="ISO 3166 code for region-specific factors",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Notes for the calculation record",
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


class GasEmissionResult(BaseModel):
    """Emission result for a single greenhouse gas from a process.

    Captures the calculated emissions in both native mass units and
    CO2-equivalent, along with the emission factor and GWP used for
    full traceability.

    Attributes:
        gas: Greenhouse gas species.
        emissions_kg: Calculated emissions in kilograms of the specific gas.
        emissions_tonnes: Calculated emissions in metric tonnes of the gas.
        emissions_tco2e: Calculated emissions in tonnes of CO2-equivalent.
        emission_factor_value: Numeric value of the emission factor applied.
        emission_factor_unit: Unit of the emission factor applied.
        emission_factor_source: Source authority for the emission factor.
        gwp_applied: Global Warming Potential multiplier applied.
        gwp_source: IPCC AR edition for the GWP value.
    """

    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species",
    )
    emissions_kg: float = Field(
        ...,
        ge=0,
        description="Emissions in kilograms of the specific gas",
    )
    emissions_tonnes: float = Field(
        ...,
        ge=0,
        description="Emissions in metric tonnes of the gas",
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
    gwp_source: str = Field(
        default="AR6",
        description="IPCC AR edition for the GWP value",
    )


class CalculationResult(BaseModel):
    """Complete result of a single process emission calculation.

    Contains all calculated emissions by gas, total CO2e, the methodology
    parameters used, abatement adjustments, and a SHA-256 provenance hash
    for audit trail integrity.

    Attributes:
        calculation_id: Unique identifier for this calculation result.
        process_type: Industrial process type.
        process_category: Sector category of the process.
        production_quantity_tonnes: Production output in tonnes.
        calculation_method: Methodology used.
        tier_used: Calculation tier applied.
        production_route: Production route used (if applicable).
        emissions_by_gas: Itemized emissions for each greenhouse gas.
        total_co2e_kg: Total CO2-equivalent emissions in kilograms.
        total_co2e_tonnes: Total CO2-equivalent emissions in tonnes.
        gross_co2e_tonnes: Gross emissions before abatement (tonnes CO2e).
        abatement_co2e_tonnes: Emissions removed by abatement (tonnes CO2e).
        net_co2e_tonnes: Net emissions after abatement (tonnes CO2e).
        by_product_credit_co2e_tonnes: By-product emission credits (tonnes).
        provenance_hash: SHA-256 hash for audit trail integrity.
        calculation_trace: Ordered list of human-readable calculation steps.
        timestamp: UTC timestamp when the calculation was performed.
        facility_id: Facility identifier (if provided).
        unit_id: Process unit identifier (if provided).
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        gwp_source: GWP source used.
        emission_factor_source: EF source authority used.
    """

    calculation_id: str = Field(
        default_factory=lambda: f"pecalc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this calculation result",
    )
    process_type: ProcessType = Field(
        ...,
        description="Industrial process type",
    )
    process_category: ProcessCategory = Field(
        ...,
        description="Sector category of the process",
    )
    production_quantity_tonnes: float = Field(
        ...,
        gt=0,
        description="Production output in tonnes",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Methodology used",
    )
    tier_used: CalculationTier = Field(
        ...,
        description="Calculation tier applied",
    )
    production_route: Optional[ProductionRoute] = Field(
        default=None,
        description="Production route used (if applicable)",
    )
    emissions_by_gas: List[GasEmissionResult] = Field(
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
        description="Total CO2-equivalent emissions in tonnes",
    )
    gross_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Gross emissions before abatement (tonnes CO2e)",
    )
    abatement_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Emissions removed by abatement (tonnes CO2e)",
    )
    net_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Net emissions after abatement (tonnes CO2e)",
    )
    by_product_credit_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="By-product emission credits (tonnes CO2e)",
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
        description="Facility identifier",
    )
    unit_id: Optional[str] = Field(
        default=None,
        description="Process unit identifier",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the reporting period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the reporting period",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source used",
    )
    emission_factor_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="EF source authority used",
    )


class CalculationDetailResult(BaseModel):
    """Extended calculation result with detailed step-by-step breakdown.

    Wraps a :class:`CalculationResult` with additional detail fields
    including full material balance data, intermediate values, and
    per-step audit entries for comprehensive traceability.

    Attributes:
        result: The core calculation result.
        material_balance: Material balance data (input/output carbon
            tonnes) when mass balance method is used.
        stoichiometric_details: Stoichiometric calculation intermediate
            values when stoichiometric method is used.
        emission_factor_details: Details of emission factors selected
            and applied.
        abatement_details: Details of abatement measures applied.
        audit_entries: Ordered list of step-by-step audit records.
    """

    result: CalculationResult = Field(
        ...,
        description="The core calculation result",
    )
    material_balance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Material balance data (mass balance method)",
    )
    stoichiometric_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Stoichiometric intermediate values",
    )
    emission_factor_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Emission factor selection details",
    )
    abatement_details: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Abatement measures applied",
    )
    audit_entries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Step-by-step audit records",
    )


class AbatementRecord(BaseModel):
    """Record of an emission abatement measure applied to a process.

    Tracks the abatement technology, its efficiency, the emissions
    reduced, and the time period of application. Used to calculate
    net emissions from gross process emissions.

    Attributes:
        abatement_id: Unique identifier for this abatement record.
        unit_id: Process unit where the abatement is applied.
        abatement_type: Type of abatement technology.
        efficiency: Abatement efficiency as a fraction (0.0-1.0).
            Represents the fraction of target gas emissions eliminated.
        target_gas: The greenhouse gas targeted by this abatement.
        emissions_reduced_tco2e: Emissions reduced in tonnes CO2e.
        operational_status: Current status (active, inactive, maintenance).
        period_start: Start of the abatement application period.
        period_end: End of the abatement application period.
        verification_status: Whether the abatement has been verified
            by a third party (verified, unverified, pending).
        notes: Optional notes about the abatement record.
    """

    abatement_id: str = Field(
        default_factory=lambda: f"abate_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this abatement record",
    )
    unit_id: str = Field(
        ...,
        min_length=1,
        description="Process unit where the abatement is applied",
    )
    abatement_type: AbatementType = Field(
        ...,
        description="Type of abatement technology",
    )
    efficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Abatement efficiency as a fraction (0.0-1.0)",
    )
    target_gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas targeted by this abatement",
    )
    emissions_reduced_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Emissions reduced in tonnes CO2e",
    )
    operational_status: str = Field(
        default="active",
        description="Current status (active, inactive, maintenance)",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the abatement application period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the abatement application period",
    )
    verification_status: str = Field(
        default="unverified",
        description="Verification status (verified, unverified, pending)",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notes about the abatement record",
    )

    @field_validator("operational_status")
    @classmethod
    def validate_operational_status(cls, v: str) -> str:
        """Normalize and validate operational status."""
        normalised = v.strip().lower()
        valid = {"active", "inactive", "maintenance"}
        if normalised not in valid:
            raise ValueError(
                f"operational_status must be one of {sorted(valid)}, "
                f"got '{v}'"
            )
        return normalised

    @field_validator("verification_status")
    @classmethod
    def validate_verification_status(cls, v: str) -> str:
        """Normalize and validate verification status."""
        normalised = v.strip().lower()
        valid = {"verified", "unverified", "pending"}
        if normalised not in valid:
            raise ValueError(
                f"verification_status must be one of {sorted(valid)}, "
                f"got '{v}'"
            )
        return normalised


class ComplianceCheckResult(BaseModel):
    """Result of a regulatory compliance check against a specific framework.

    Captures the assessment of whether a process emission calculation
    meets the requirements of a regulatory framework, with detailed
    findings and recommendations.

    Attributes:
        framework: Regulatory framework checked.
        status: Overall compliance status.
        requirement_count: Total requirements evaluated.
        met_count: Number of requirements fully met.
        partially_met_count: Number of requirements partially met.
        not_met_count: Number of requirements not met.
        findings: List of compliance findings with details.
        recommendations: List of corrective action recommendations.
        checked_at: UTC timestamp when the check was performed.
        calculation_id: The calculation this check applies to.
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
    requirement_count: int = Field(
        default=0,
        ge=0,
        description="Total requirements evaluated",
    )
    met_count: int = Field(
        default=0,
        ge=0,
        description="Number of requirements fully met",
    )
    partially_met_count: int = Field(
        default=0,
        ge=0,
        description="Number of requirements partially met",
    )
    not_met_count: int = Field(
        default=0,
        ge=0,
        description="Number of requirements not met",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Compliance findings with details",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Corrective action recommendations",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the check was performed",
    )
    calculation_id: Optional[str] = Field(
        default=None,
        description="Calculation this check applies to",
    )


class BatchCalculationRequest(BaseModel):
    """Request model for batch process emission calculations.

    Groups multiple calculation inputs for processing as a single
    batch, sharing common parameters like GWP source and compliance
    framework preferences.

    Attributes:
        calculations: List of individual calculation requests.
        gwp_source: IPCC AR edition for GWP values (shared by batch).
        enable_compliance: Whether to run compliance checks on results.
        compliance_frameworks: Frameworks to check against.
        organization_id: Organization identifier for aggregation.
        reporting_period: Temporal granularity for the batch.
    """

    calculations: List[CalculationRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_CALCULATIONS_PER_BATCH,
        description="List of individual calculation requests",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC AR edition for GWP values",
    )
    enable_compliance: bool = Field(
        default=False,
        description="Whether to run compliance checks on results",
    )
    compliance_frameworks: List[str] = Field(
        default_factory=list,
        description="Regulatory frameworks to check against",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization identifier for aggregation",
    )
    reporting_period: Optional[ReportingPeriod] = Field(
        default=None,
        description="Temporal granularity for the batch",
    )


class BatchCalculationResult(BaseModel):
    """Response model for a batch process emission calculation.

    Aggregates individual calculation results with batch-level totals,
    emissions breakdown by process type and gas, and processing metadata.

    Attributes:
        success: Whether all calculations in the batch succeeded.
        results: List of individual calculation results.
        total_co2e_tonnes: Batch total CO2-equivalent in metric tonnes.
        total_gross_co2e_tonnes: Batch gross total before abatement.
        total_abatement_co2e_tonnes: Batch total abatement reduction.
        total_net_co2e_tonnes: Batch net total after abatement.
        emissions_by_process_type: Emissions by process type (tCO2e).
        emissions_by_gas: Emissions by gas species (tCO2e).
        emissions_by_category: Emissions by process category (tCO2e).
        calculation_count: Number of successful calculations.
        failed_count: Number of failed calculations.
        processing_time_ms: Total batch processing time in ms.
        provenance_hash: SHA-256 hash covering the entire batch.
        gwp_source: GWP source used for this batch.
        compliance_results: Compliance check results (if requested).
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
    total_gross_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch gross total before abatement (tonnes CO2e)",
    )
    total_abatement_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch total abatement reduction (tonnes CO2e)",
    )
    total_net_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Batch net total after abatement (tonnes CO2e)",
    )
    emissions_by_process_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions aggregated by process type (tCO2e)",
    )
    emissions_by_gas: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions aggregated by gas species (tCO2e)",
    )
    emissions_by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions aggregated by process category (tCO2e)",
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
        description="Total batch processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash covering the entire batch result",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source used for this batch",
    )
    compliance_results: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="Compliance check results (if requested)",
    )


class UncertaintyRequest(BaseModel):
    """Request model for uncertainty quantification of a process emission.

    Attributes:
        calculation_request: The base calculation request to quantify.
        iterations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.
        confidence_levels: Confidence levels for interval calculation.
        include_contributions: Whether to compute parameter contribution
            analysis.
    """

    calculation_request: CalculationRequest = Field(
        ...,
        description="The base calculation request to quantify",
    )
    iterations: int = Field(
        default=5000,
        gt=0,
        le=1_000_000,
        description="Number of Monte Carlo iterations",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
    confidence_levels: List[float] = Field(
        default_factory=lambda: [90.0, 95.0, 99.0],
        description="Confidence levels for interval calculation",
    )
    include_contributions: bool = Field(
        default=True,
        description="Whether to compute parameter contribution analysis",
    )

    @field_validator("confidence_levels")
    @classmethod
    def validate_confidence_levels(cls, v: List[float]) -> List[float]:
        """Validate all confidence levels are in (0, 100)."""
        for lvl in v:
            if not (0.0 < lvl < 100.0):
                raise ValueError(
                    f"Each confidence level must be in (0, 100), got {lvl}"
                )
        return v


class UncertaintyResult(BaseModel):
    """Monte Carlo uncertainty quantification result for a process emission.

    Provides statistical characterization of emission estimate uncertainty
    including mean, standard deviation, confidence intervals, and
    parameter contribution analysis.

    Attributes:
        mean_co2e_tonnes: Mean CO2-equivalent emission estimate (tonnes).
        std_dev_tonnes: Standard deviation (tonnes).
        coefficient_of_variation: CV = std_dev / mean (dimensionless).
        confidence_intervals: Confidence intervals keyed by level string
            (e.g. "90" -> (lower, upper) in tonnes CO2e).
        iterations: Number of Monte Carlo iterations performed.
        seed_used: Random seed used for reproducibility.
        data_quality_score: Overall data quality indicator (1-5 scale).
        tier: Calculation tier used for this uncertainty analysis.
        contributions: Parameter contribution to total variance, keyed
            by parameter name (e.g. "emission_factor" -> 0.45).
        process_type: Process type analyzed.
        calculation_method: Calculation method used.
    """

    mean_co2e_tonnes: float = Field(
        ...,
        ge=0,
        description="Mean CO2-equivalent emission estimate (tonnes)",
    )
    std_dev_tonnes: float = Field(
        ...,
        ge=0,
        description="Standard deviation (tonnes)",
    )
    coefficient_of_variation: float = Field(
        ...,
        ge=0,
        description="CV = std_dev / mean (dimensionless)",
    )
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Confidence intervals keyed by level",
    )
    iterations: int = Field(
        ...,
        gt=0,
        description="Number of Monte Carlo iterations performed",
    )
    seed_used: int = Field(
        default=42,
        ge=0,
        description="Random seed used for reproducibility",
    )
    data_quality_score: Optional[float] = Field(
        default=None,
        ge=1,
        le=5,
        description="Data quality indicator (1-5 scale)",
    )
    tier: CalculationTier = Field(
        ...,
        description="Calculation tier used",
    )
    contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameter contribution to total variance",
    )
    process_type: ProcessType = Field(
        ...,
        description="Process type analyzed",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Calculation method used",
    )


class AggregationRequest(BaseModel):
    """Request model for aggregating process emissions.

    Defines the scope and grouping parameters for rolling up individual
    calculation results into aggregate totals by facility, process type,
    category, or organization.

    Attributes:
        facility_id: Optional facility filter.
        organization_id: Optional organization filter.
        process_types: Optional list of process types to include.
        categories: Optional list of process categories to include.
        period_start: Start of the aggregation period.
        period_end: End of the aggregation period.
        reporting_period: Temporal granularity.
        group_by: Grouping dimensions (process_type, category, facility,
            gas, production_route).
        include_abatement: Whether to include abatement in aggregation.
    """

    facility_id: Optional[str] = Field(
        default=None,
        description="Facility filter",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization filter",
    )
    process_types: Optional[List[ProcessType]] = Field(
        default=None,
        description="Process types to include",
    )
    categories: Optional[List[ProcessCategory]] = Field(
        default=None,
        description="Process categories to include",
    )
    period_start: datetime = Field(
        ...,
        description="Start of the aggregation period",
    )
    period_end: datetime = Field(
        ...,
        description="End of the aggregation period",
    )
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Temporal granularity",
    )
    group_by: List[str] = Field(
        default_factory=lambda: ["process_type"],
        description="Grouping dimensions",
    )
    include_abatement: bool = Field(
        default=True,
        description="Whether to include abatement in aggregation",
    )

    @field_validator("period_end")
    @classmethod
    def agg_end_after_start(
        cls, v: datetime, info: Any
    ) -> datetime:
        """Validate that period_end is after period_start."""
        period_start = info.data.get("period_start")
        if period_start is not None and v <= period_start:
            raise ValueError(
                "period_end must be after period_start"
            )
        return v

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, v: List[str]) -> List[str]:
        """Validate group_by dimensions are recognized."""
        valid_dims = {
            "process_type",
            "category",
            "facility",
            "gas",
            "production_route",
            "unit_id",
            "tier",
            "method",
        }
        for dim in v:
            if dim not in valid_dims:
                raise ValueError(
                    f"group_by dimension '{dim}' not recognized; "
                    f"valid dimensions: {sorted(valid_dims)}"
                )
        return v


class AggregationResult(BaseModel):
    """Result of an emission aggregation across process calculations.

    Rolls up individual calculation results into aggregate totals with
    breakdowns by the requested grouping dimensions.

    Attributes:
        facility_id: Facility filter applied (if any).
        organization_id: Organization filter applied (if any).
        period_start: Start of the aggregation period.
        period_end: End of the aggregation period.
        reporting_period: Temporal granularity.
        total_co2e_tonnes: Aggregate total CO2e (metric tonnes).
        total_gross_co2e_tonnes: Aggregate gross CO2e (metric tonnes).
        total_abatement_co2e_tonnes: Aggregate abatement (metric tonnes).
        total_net_co2e_tonnes: Aggregate net CO2e (metric tonnes).
        by_product_credit_co2e_tonnes: Aggregate by-product credits.
        emissions_by_group: Emissions keyed by grouping dimension values.
        emissions_by_gas: Emissions keyed by gas species (tCO2e).
        calculation_count: Number of calculations aggregated.
        process_types_included: Distinct process types in aggregation.
        categories_included: Distinct process categories in aggregation.
        provenance_hash: SHA-256 hash for audit trail integrity.
    """

    facility_id: Optional[str] = Field(
        default=None,
        description="Facility filter applied",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization filter applied",
    )
    period_start: datetime = Field(
        ...,
        description="Start of the aggregation period",
    )
    period_end: datetime = Field(
        ...,
        description="End of the aggregation period",
    )
    reporting_period: ReportingPeriod = Field(
        ...,
        description="Temporal granularity",
    )
    total_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Aggregate total CO2e (metric tonnes)",
    )
    total_gross_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Aggregate gross CO2e before abatement (metric tonnes)",
    )
    total_abatement_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Aggregate abatement reduction (metric tonnes)",
    )
    total_net_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Aggregate net CO2e after abatement (metric tonnes)",
    )
    by_product_credit_co2e_tonnes: float = Field(
        default=0.0,
        ge=0,
        description="Aggregate by-product credits (metric tonnes)",
    )
    emissions_by_group: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Emissions keyed by grouping dimension values",
    )
    emissions_by_gas: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions keyed by gas species (tCO2e)",
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations aggregated",
    )
    process_types_included: List[str] = Field(
        default_factory=list,
        description="Distinct process types in aggregation",
    )
    categories_included: List[str] = Field(
        default_factory=list,
        description="Distinct process categories in aggregation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail integrity",
    )

    @field_validator("period_end")
    @classmethod
    def result_end_after_start(
        cls, v: datetime, info: Any
    ) -> datetime:
        """Validate that period_end is after period_start."""
        period_start = info.data.get("period_start")
        if period_start is not None and v <= period_start:
            raise ValueError(
                "period_end must be after period_start"
            )
        return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_MATERIAL_INPUTS_PER_CALC",
    "GWP_VALUES",
    "CARBONATE_EMISSION_FACTORS",
    "PROCESS_CATEGORY_MAP",
    "PROCESS_DEFAULT_GASES",
    # Enums
    "ProcessCategory",
    "ProcessType",
    "EmissionGas",
    "CalculationMethod",
    "CalculationTier",
    "EmissionFactorSource",
    "GWPSource",
    "MaterialType",
    "AbatementType",
    "ProcessUnitType",
    "ProcessMode",
    "ComplianceStatus",
    "ReportingPeriod",
    "UnitType",
    "ProductionRoute",
    "CarbonateType",
    # Data models
    "ProcessTypeInfo",
    "RawMaterialInfo",
    "EmissionFactorRecord",
    "ProcessUnitRecord",
    "MaterialInputRecord",
    "CalculationRequest",
    "GasEmissionResult",
    "CalculationResult",
    "CalculationDetailResult",
    "AbatementRecord",
    "ComplianceCheckResult",
    "BatchCalculationRequest",
    "BatchCalculationResult",
    "UncertaintyRequest",
    "UncertaintyResult",
    "AggregationRequest",
    "AggregationResult",
]
