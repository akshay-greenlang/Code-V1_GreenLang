"""
Waste Generated in Operations Agent Models (AGENT-MRV-018)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 5
(Waste Generated in Operations) emissions calculations.

Supports:
- 5 waste treatment pathways (landfill, incineration, recycling, composting/AD, wastewater)
- IPCC First Order Decay (FOD) model for landfill emissions
- 4 calculation methods (supplier-specific, waste-type-specific, average-data, spend-based)
- 14 waste categories, 11 treatment methods, 6 waste streams
- EPA WARM v16 factors (61 materials x 6 disposal methods)
- DEFRA/BEIS waste conversion factors
- IPCC 2006/2019 waste sector guidance
- European Waste Catalogue (EWC) classification
- Basel Convention hazard classes (H1-H13)
- Biogenic vs fossil carbon separation
- CSRD ESRS E5 circular economy disclosures
- GHG Protocol, ISO 14064, CDP, SBTi compliance
- Data quality indicators (DQI) and uncertainty quantification

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.waste_generated.models import WasteStreamInput, LandfillInput
    >>> waste = WasteStreamInput(
    ...     stream_id="WS-2026-001",
    ...     facility_id="FAC-001",
    ...     waste_category=WasteCategory.PAPER_CARDBOARD,
    ...     treatment_method=WasteTreatmentMethod.LANDFILL,
    ...     mass_tonnes=Decimal("15.5")
    ... )
    >>> landfill = LandfillInput(
    ...     landfill_type=LandfillType.MANAGED_ANAEROBIC,
    ...     climate_zone=ClimateZone.TEMPERATE_WET,
    ...     gas_collection=GasCollectionSystem.ACTIVE_GEOMEMBRANE
    ... )
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator, field_validator, model_validator
from pydantic import ConfigDict
import hashlib

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-005"
AGENT_COMPONENT: str = "AGENT-MRV-018"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_wg_"

# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class CalculationMethod(str, Enum):
    """Calculation method for waste emissions per GHG Protocol Technical Guidance."""

    SUPPLIER_SPECIFIC = "supplier_specific"  # Waste contractor-reported emissions
    WASTE_TYPE_SPECIFIC = "waste_type_specific"  # Mass × waste type × treatment method EF
    AVERAGE_DATA = "average_data"  # Total mass × average EF per treatment
    SPEND_BASED = "spend_based"  # Waste management spend × EEIO factor


class WasteTreatmentMethod(str, Enum):
    """Waste treatment methods per GHG Protocol and IPCC Vol 5."""

    LANDFILL = "landfill"  # Standard landfill (no gas capture)
    LANDFILL_WITH_GAS_CAPTURE = "landfill_with_gas_capture"  # Active gas collection
    LANDFILL_WITH_ENERGY_RECOVERY = "landfill_with_energy_recovery"  # LFG to energy
    INCINERATION = "incineration"  # Thermal treatment (no energy recovery)
    INCINERATION_WITH_ENERGY_RECOVERY = "incineration_with_energy_recovery"  # Waste-to-energy
    RECYCLING_OPEN_LOOP = "recycling_open_loop"  # Material recycled to different product
    RECYCLING_CLOSED_LOOP = "recycling_closed_loop"  # Material recycled to same product
    COMPOSTING = "composting"  # Aerobic composting
    ANAEROBIC_DIGESTION = "anaerobic_digestion"  # Anaerobic digestion with biogas
    WASTEWATER_TREATMENT = "wastewater_treatment"  # Liquid effluent treatment
    OTHER = "other"  # Other disposal/treatment


class WasteCategory(str, Enum):
    """Waste material categories aligned with EPA WARM and DEFRA."""

    PAPER_CARDBOARD = "paper_cardboard"
    PLASTICS_HDPE = "plastics_hdpe"  # High-density polyethylene
    PLASTICS_LDPE = "plastics_ldpe"  # Low-density polyethylene
    PLASTICS_PET = "plastics_pet"  # Polyethylene terephthalate
    PLASTICS_PP = "plastics_pp"  # Polypropylene
    PLASTICS_MIXED = "plastics_mixed"
    GLASS = "glass"
    METALS_ALUMINUM = "metals_aluminum"
    METALS_STEEL = "metals_steel"
    METALS_MIXED = "metals_mixed"
    FOOD_WASTE = "food_waste"
    GARDEN_WASTE = "garden_waste"  # Yard trimmings
    TEXTILES = "textiles"
    WOOD = "wood"
    RUBBER_LEATHER = "rubber_leather"
    ELECTRONICS = "electronics"  # WEEE (Waste Electrical and Electronic Equipment)
    CONSTRUCTION_DEMOLITION = "construction_demolition"  # C&D waste
    HAZARDOUS = "hazardous"
    MIXED_MSW = "mixed_msw"  # Mixed municipal solid waste
    OTHER = "other"


class WasteStream(str, Enum):
    """Waste stream types per EU Waste Framework Directive."""

    MUNICIPAL_SOLID_WASTE = "municipal_solid_waste"  # MSW
    COMMERCIAL_INDUSTRIAL = "commercial_industrial"  # Commercial/industrial waste
    CONSTRUCTION_DEMOLITION = "construction_demolition"  # C&D waste
    HAZARDOUS = "hazardous"  # Hazardous waste
    WASTEWATER = "wastewater"  # Liquid effluent
    SPECIAL = "special"  # Special waste (e.g., medical, radioactive)


class LandfillType(str, Enum):
    """Landfill types per IPCC 2006 Vol 5 Table 3.1."""

    MANAGED_ANAEROBIC = "managed_anaerobic"  # MCF = 1.0
    MANAGED_SEMI_AEROBIC = "managed_semi_aerobic"  # MCF = 0.5
    UNMANAGED_DEEP = "unmanaged_deep"  # MCF = 0.8 (depth > 5m)
    UNMANAGED_SHALLOW = "unmanaged_shallow"  # MCF = 0.4 (depth < 5m)
    UNCATEGORIZED = "uncategorized"  # MCF = 0.6 (default)
    ACTIVE_AERATION = "active_aeration"  # MCF = 0.4 (semi-aerobic with forced air)


class ClimateZone(str, Enum):
    """IPCC climate zones for landfill decay rate constants (k)."""

    BOREAL_TEMPERATE_DRY = "boreal_temperate_dry"  # MAT < 20°C, MAP/PET < 1
    TEMPERATE_WET = "temperate_wet"  # MAT < 20°C, MAP/PET > 1
    TROPICAL_DRY = "tropical_dry"  # MAT >= 20°C, MAP/PET < 1
    TROPICAL_WET = "tropical_wet"  # MAT >= 20°C, MAP/PET > 1


class IncineratorType(str, Enum):
    """Incinerator types per IPCC 2006 Vol 5 Table 5.3."""

    CONTINUOUS_STOKER = "continuous_stoker"  # Modern mass-burn incinerator
    SEMI_CONTINUOUS = "semi_continuous"  # Semi-continuous feed
    BATCH = "batch"  # Batch-fed incinerator
    FLUIDIZED_BED = "fluidized_bed"  # Fluidized bed combustor
    OPEN_BURNING = "open_burning"  # Uncontrolled open burning (not recommended)


class RecyclingType(str, Enum):
    """Recycling types per GHG Protocol."""

    OPEN_LOOP = "open_loop"  # Material recycled to different/lower-grade product
    CLOSED_LOOP = "closed_loop"  # Material recycled to same product type


class WastewaterSystem(str, Enum):
    """Wastewater treatment systems per IPCC 2006 Vol 5 Table 6.3."""

    CENTRALIZED_AEROBIC_GOOD = "centralized_aerobic_good"  # MCF = 0.00 (well managed)
    CENTRALIZED_AEROBIC_POOR = "centralized_aerobic_poor"  # MCF = 0.03 (overloaded)
    CENTRALIZED_ANAEROBIC = "centralized_anaerobic"  # MCF = 0.80 (anaerobic reactor)
    ANAEROBIC_REACTOR = "anaerobic_reactor"  # MCF = 0.80 (UASB, anaerobic filter)
    LAGOON_SHALLOW = "lagoon_shallow"  # MCF = 0.20 (< 2m depth)
    LAGOON_DEEP = "lagoon_deep"  # MCF = 0.80 (> 2m depth)
    SEPTIC = "septic"  # MCF = 0.50 (septic tank + drain field)
    OPEN_SEWER = "open_sewer"  # MCF = 0.10 (untreated discharge)
    CONSTRUCTED_WETLAND = "constructed_wetland"  # MCF = 0.05 (engineered wetland)


class GasCollectionSystem(str, Enum):
    """Landfill gas collection system types."""

    NONE = "none"  # No gas collection (capture efficiency = 0%)
    ACTIVE_OPERATING_CELL = "active_operating_cell"  # Active with operating cover (75%)
    ACTIVE_TEMP_COVER = "active_temp_cover"  # Active with temporary cover (50%)
    ACTIVE_CLAY_COVER = "active_clay_cover"  # Active with clay cap (65%)
    ACTIVE_GEOMEMBRANE = "active_geomembrane"  # Active with synthetic cap (90%)
    PASSIVE_VENTING = "passive_venting"  # Passive vent pipes (20%)
    FLARE_ONLY = "flare_only"  # Gas collected and flared (35%)


class EFSource(str, Enum):
    """Emission factor source."""

    EPA_WARM = "epa_warm"  # EPA WARM v16
    DEFRA_BEIS = "defra_beis"  # DEFRA/DESNZ conversion factors
    IPCC_2006 = "ipcc_2006"  # IPCC 2006 Guidelines Vol 5
    IPCC_2019 = "ipcc_2019"  # IPCC 2019 Refinement
    CUSTOM = "custom"  # Custom/contractor-specific


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 + E5
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    EU_WASTE_DIRECTIVE = "eu_waste_directive"  # EU Waste Framework Directive 2008/98/EC
    EPA_40CFR98 = "epa_40cfr98"  # EPA 40 CFR Part 98 Subpart HH/TT


class DataQualityTier(str, Enum):
    """IPCC data quality tiers."""

    TIER_1 = "tier_1"  # Default emission factors
    TIER_2 = "tier_2"  # Country/region-specific emission factors
    TIER_3 = "tier_3"  # Facility-specific emission factors


class WasteDataSource(str, Enum):
    """Waste data source types (affects DQI reliability score)."""

    WASTE_AUDIT = "waste_audit"  # Physical waste audit (DQI reliability = 1)
    TRANSFER_NOTES = "transfer_notes"  # Waste transfer notes/invoices (DQI = 2)
    PROCUREMENT_ESTIMATE = "procurement_estimate"  # Procurement data estimate (DQI = 3)
    SPEND_ESTIMATE = "spend_estimate"  # Spend-based estimate (DQI = 4)


class ProvenanceStage(str, Enum):
    """Processing pipeline stages."""

    VALIDATE = "validate"  # Input validation
    CLASSIFY = "classify"  # Waste classification
    NORMALIZE = "normalize"  # Unit normalization
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution
    CALCULATE_TREATMENT = "calculate_treatment"  # Treatment emissions calculation
    CALCULATE_TRANSPORT = "calculate_transport"  # Transport to facility calculation
    ALLOCATE = "allocate"  # Multi-facility allocation
    COMPLIANCE = "compliance"  # Compliance checks
    AGGREGATE = "aggregate"  # Aggregation
    SEAL = "seal"  # Provenance sealing


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method."""

    IPCC_DEFAULT = "ipcc_default"  # IPCC default uncertainty ranges
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    ERROR_PROPAGATION = "error_propagation"  # Analytical error propagation


class HazardClass(str, Enum):
    """Basel Convention hazard classes (Annex I)."""

    H1 = "h1"  # Explosive
    H2 = "h2"  # Oxidizing
    H3 = "h3"  # Flammable liquids
    H4_1 = "h4_1"  # Flammable solids
    H4_2 = "h4_2"  # Spontaneously combustible
    H4_3 = "h4_3"  # Water-reactive
    H5_1 = "h5_1"  # Oxidizing (supporting combustion)
    H5_2 = "h5_2"  # Organic peroxides
    H6_1 = "h6_1"  # Poisonous (acute)
    H6_2 = "h6_2"  # Infectious substances
    H8 = "h8"  # Corrosives
    H10 = "h10"  # Liberation of toxic gases
    H11 = "h11"  # Toxic (delayed or chronic)
    H12 = "h12"  # Ecotoxic
    H13 = "h13"  # Capable of yielding another material after disposal


class GWPVersion(str, Enum):
    """IPCC Global Warming Potential source."""

    AR4 = "ar4"  # Fourth Assessment Report (100-year)
    AR5 = "ar5"  # Fifth Assessment Report (100-year)
    AR6 = "ar6"  # Sixth Assessment Report (100-year)
    AR6_20YR = "ar6_20yr"  # Sixth Assessment Report (20-year)


class IndustryWastewaterType(str, Enum):
    """Industry-specific wastewater types (IPCC 2006 Vol 5 Table 6.9)."""

    STARCH = "starch"  # Starch production
    ALCOHOL = "alcohol"  # Alcohol/spirits production
    BEER_MALT = "beer_malt"  # Beer and malt beverages
    PULP_PAPER = "pulp_paper"  # Pulp and paper
    FOOD_PROCESSING = "food_processing"  # Food processing (general)
    MEAT_POULTRY = "meat_poultry"  # Meat and poultry processing
    VEGETABLES_FRUITS = "vegetables_fruits"  # Vegetable and fruit processing
    DAIRY = "dairy"  # Dairy processing
    SUGAR = "sugar"  # Sugar refining
    TEXTILE = "textile"  # Textile dyeing/finishing
    PHARMACEUTICAL = "pharmaceutical"  # Pharmaceutical manufacturing
    OTHER = "other"  # Other industrial wastewater


class EmissionGas(str, Enum):
    """Greenhouse gas types."""

    CO2_FOSSIL = "co2_fossil"  # Fossil-origin CO2
    CO2_BIOGENIC = "co2_biogenic"  # Biogenic-origin CO2 (memo item)
    CH4 = "ch4"  # Methane
    N2O = "n2o"  # Nitrous oxide
    CO2E = "co2e"  # Total CO2 equivalent


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol."""

    TEMPORAL = "temporal"  # Temporal correlation
    GEOGRAPHICAL = "geographical"  # Geographical correlation
    TECHNOLOGICAL = "technological"  # Technological correlation
    COMPLETENESS = "completeness"  # Data completeness
    RELIABILITY = "reliability"  # Source reliability


class DQIScore(str, Enum):
    """Data Quality Indicator scores (1-5 scale)."""

    VERY_GOOD = "very_good"  # 1
    GOOD = "good"  # 2
    FAIR = "fair"  # 3
    POOR = "poor"  # 4
    VERY_POOR = "very_poor"  # 5


class ComplianceStatus(str, Enum):
    """Compliance check status."""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for spend-based method."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"


class ExportFormat(str, Enum):
    """Export format for results."""

    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    PDF = "pdf"


class BatchStatus(str, Enum):
    """Batch calculation status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some records failed


# ==============================================================================
# CONSTANT TABLES
# ==============================================================================

# Global Warming Potential (100-year unless stated)
GWP_VALUES: Dict[GWPVersion, Dict[str, Decimal]] = {
    GWPVersion.AR4: {
        "co2": Decimal("1"),
        "ch4": Decimal("25"),
        "n2o": Decimal("298"),
    },
    GWPVersion.AR5: {
        "co2": Decimal("1"),
        "ch4": Decimal("28"),
        "n2o": Decimal("265"),
    },
    GWPVersion.AR6: {
        "co2": Decimal("1"),
        "ch4": Decimal("27.9"),  # Fossil CH4 with climate-carbon feedback
        "n2o": Decimal("273"),
    },
    GWPVersion.AR6_20YR: {
        "co2": Decimal("1"),
        "ch4": Decimal("82.5"),  # Fossil CH4, 20-year
        "n2o": Decimal("273"),  # Same as 100-year
    },
}

# Degradable Organic Carbon (DOC) fraction per waste type (IPCC 2006 Vol 5 Table 2.4)
DOC_VALUES: Dict[WasteCategory, Decimal] = {
    WasteCategory.FOOD_WASTE: Decimal("0.150"),
    WasteCategory.GARDEN_WASTE: Decimal("0.200"),
    WasteCategory.PAPER_CARDBOARD: Decimal("0.400"),
    WasteCategory.WOOD: Decimal("0.430"),
    WasteCategory.TEXTILES: Decimal("0.240"),
    WasteCategory.RUBBER_LEATHER: Decimal("0.390"),
    WasteCategory.PLASTICS_HDPE: Decimal("0.000"),  # Not degradable
    WasteCategory.PLASTICS_LDPE: Decimal("0.000"),
    WasteCategory.PLASTICS_PET: Decimal("0.000"),
    WasteCategory.PLASTICS_PP: Decimal("0.000"),
    WasteCategory.PLASTICS_MIXED: Decimal("0.000"),
    WasteCategory.GLASS: Decimal("0.000"),
    WasteCategory.METALS_ALUMINUM: Decimal("0.000"),
    WasteCategory.METALS_STEEL: Decimal("0.000"),
    WasteCategory.METALS_MIXED: Decimal("0.000"),
    WasteCategory.ELECTRONICS: Decimal("0.050"),  # Small organic fraction
    WasteCategory.CONSTRUCTION_DEMOLITION: Decimal("0.080"),
    WasteCategory.MIXED_MSW: Decimal("0.160"),  # Developed countries
    WasteCategory.HAZARDOUS: Decimal("0.100"),  # Variable, conservative estimate
    WasteCategory.OTHER: Decimal("0.100"),  # Conservative estimate
}

# Methane Correction Factor (MCF) per landfill type (IPCC 2006 Vol 5 Table 3.1)
MCF_VALUES: Dict[LandfillType, Decimal] = {
    LandfillType.MANAGED_ANAEROBIC: Decimal("1.0"),
    LandfillType.MANAGED_SEMI_AEROBIC: Decimal("0.5"),
    LandfillType.UNMANAGED_DEEP: Decimal("0.8"),
    LandfillType.UNMANAGED_SHALLOW: Decimal("0.4"),
    LandfillType.UNCATEGORIZED: Decimal("0.6"),
    LandfillType.ACTIVE_AERATION: Decimal("0.4"),
}

# Decay rate constants (k, yr-1) by climate zone and waste type (IPCC 2006 Vol 5 Table 3.3)
DECAY_RATE_CONSTANTS: Dict[ClimateZone, Dict[str, Decimal]] = {
    ClimateZone.BOREAL_TEMPERATE_DRY: {
        "food_waste": Decimal("0.06"),
        "garden_waste": Decimal("0.03"),
        "paper_cardboard": Decimal("0.04"),
        "wood": Decimal("0.02"),
        "textiles": Decimal("0.04"),
        "other": Decimal("0.05"),
    },
    ClimateZone.TEMPERATE_WET: {
        "food_waste": Decimal("0.185"),
        "garden_waste": Decimal("0.10"),
        "paper_cardboard": Decimal("0.06"),
        "wood": Decimal("0.03"),
        "textiles": Decimal("0.06"),
        "other": Decimal("0.09"),
    },
    ClimateZone.TROPICAL_DRY: {
        "food_waste": Decimal("0.085"),
        "garden_waste": Decimal("0.05"),
        "paper_cardboard": Decimal("0.045"),
        "wood": Decimal("0.025"),
        "textiles": Decimal("0.045"),
        "other": Decimal("0.065"),
    },
    ClimateZone.TROPICAL_WET: {
        "food_waste": Decimal("0.40"),
        "garden_waste": Decimal("0.17"),
        "paper_cardboard": Decimal("0.07"),
        "wood": Decimal("0.035"),
        "textiles": Decimal("0.07"),
        "other": Decimal("0.17"),
    },
}

# Gas collection system capture efficiency
GAS_CAPTURE_EFFICIENCY: Dict[GasCollectionSystem, Decimal] = {
    GasCollectionSystem.NONE: Decimal("0.00"),
    GasCollectionSystem.ACTIVE_OPERATING_CELL: Decimal("0.75"),
    GasCollectionSystem.ACTIVE_TEMP_COVER: Decimal("0.50"),
    GasCollectionSystem.ACTIVE_CLAY_COVER: Decimal("0.65"),
    GasCollectionSystem.ACTIVE_GEOMEMBRANE: Decimal("0.90"),
    GasCollectionSystem.PASSIVE_VENTING: Decimal("0.20"),
    GasCollectionSystem.FLARE_ONLY: Decimal("0.35"),
}

# Oxidation factor (OX) by landfill cover type
OXIDATION_FACTORS: Dict[str, Decimal] = {
    "no_cover": Decimal("0.00"),
    "soil_cover": Decimal("0.10"),
    "biocover": Decimal("0.20"),  # Compost cover
    "geomembrane": Decimal("0.10"),
}

# Incineration parameters per waste type (IPCC 2006 Vol 5 Table 5.2)
# dm = dry matter fraction, CF = carbon fraction of dry matter, FCF = fossil carbon fraction, OF = oxidation factor
INCINERATION_PARAMS: Dict[WasteCategory, Dict[str, Decimal]] = {
    WasteCategory.PAPER_CARDBOARD: {
        "dm": Decimal("0.90"),
        "cf": Decimal("0.46"),
        "fcf": Decimal("0.01"),  # Mostly biogenic
        "of": Decimal("1.00"),
    },
    WasteCategory.TEXTILES: {
        "dm": Decimal("0.80"),
        "cf": Decimal("0.46"),
        "fcf": Decimal("0.50"),  # Mix of synthetic (1.0) and natural (0.0)
        "of": Decimal("1.00"),
    },
    WasteCategory.FOOD_WASTE: {
        "dm": Decimal("0.40"),
        "cf": Decimal("0.38"),
        "fcf": Decimal("0.00"),  # Biogenic
        "of": Decimal("1.00"),
    },
    WasteCategory.WOOD: {
        "dm": Decimal("0.85"),
        "cf": Decimal("0.50"),
        "fcf": Decimal("0.05"),  # Mostly biogenic, some treated wood
        "of": Decimal("1.00"),
    },
    WasteCategory.GARDEN_WASTE: {
        "dm": Decimal("0.40"),
        "cf": Decimal("0.49"),
        "fcf": Decimal("0.00"),  # Biogenic
        "of": Decimal("1.00"),
    },
    WasteCategory.PLASTICS_HDPE: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.75"),
        "fcf": Decimal("1.00"),  # Fossil
        "of": Decimal("1.00"),
    },
    WasteCategory.PLASTICS_LDPE: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.75"),
        "fcf": Decimal("1.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.PLASTICS_PET: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.63"),
        "fcf": Decimal("1.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.PLASTICS_PP: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.75"),
        "fcf": Decimal("1.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.PLASTICS_MIXED: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.70"),
        "fcf": Decimal("1.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.RUBBER_LEATHER: {
        "dm": Decimal("0.84"),
        "cf": Decimal("0.67"),
        "fcf": Decimal("0.50"),  # Mix
        "of": Decimal("1.00"),
    },
    WasteCategory.ELECTRONICS: {
        "dm": Decimal("0.90"),
        "cf": Decimal("0.05"),
        "fcf": Decimal("1.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.MIXED_MSW: {
        "dm": Decimal("0.69"),  # Developed countries
        "cf": Decimal("0.33"),
        "fcf": Decimal("0.40"),
        "of": Decimal("1.00"),
    },
    WasteCategory.HAZARDOUS: {
        "dm": Decimal("0.75"),
        "cf": Decimal("0.50"),
        "fcf": Decimal("0.80"),
        "of": Decimal("1.00"),
    },
    WasteCategory.CONSTRUCTION_DEMOLITION: {
        "dm": Decimal("0.80"),
        "cf": Decimal("0.35"),
        "fcf": Decimal("0.30"),
        "of": Decimal("1.00"),
    },
    WasteCategory.OTHER: {
        "dm": Decimal("0.70"),
        "cf": Decimal("0.35"),
        "fcf": Decimal("0.50"),
        "of": Decimal("1.00"),
    },
}

# CH4 emission factors for incineration by incinerator type (kg CH4 / Gg waste) (IPCC 2006 Vol 5 Table 5.3)
CH4_EF_INCINERATION: Dict[IncineratorType, Decimal] = {
    IncineratorType.CONTINUOUS_STOKER: Decimal("0.2"),
    IncineratorType.SEMI_CONTINUOUS: Decimal("6.0"),
    IncineratorType.BATCH: Decimal("60.0"),
    IncineratorType.FLUIDIZED_BED: Decimal("0.1"),
    IncineratorType.OPEN_BURNING: Decimal("6500.0"),  # Uncontrolled
}

# N2O emission factors for incineration by incinerator type (kg N2O / Gg waste) (IPCC 2006 Vol 5 Table 5.3)
N2O_EF_INCINERATION: Dict[IncineratorType, Decimal] = {
    IncineratorType.CONTINUOUS_STOKER: Decimal("50"),
    IncineratorType.SEMI_CONTINUOUS: Decimal("50"),
    IncineratorType.BATCH: Decimal("50"),
    IncineratorType.FLUIDIZED_BED: Decimal("56"),
    IncineratorType.OPEN_BURNING: Decimal("150"),
}

# Composting emission factors (g gas / kg wet waste) (IPCC 2006 Vol 5 Table 4.1)
COMPOSTING_EF: Dict[str, Decimal] = {
    "ch4_industrial_wet": Decimal("4.0"),  # Industrial composting, wet waste
    "ch4_industrial_dry": Decimal("10.0"),  # Industrial composting, dry waste
    "n2o_industrial_wet": Decimal("0.30"),
    "n2o_industrial_dry": Decimal("0.60"),
    "ch4_home_wet": Decimal("10.0"),  # Home composting
    "ch4_home_dry": Decimal("20.0"),
    "n2o_home_wet": Decimal("0.60"),
    "n2o_home_dry": Decimal("1.20"),
}

# Anaerobic digestion methane leakage rates (fraction of biogas produced) (IPCC 2019 Refinement)
AD_LEAKAGE_RATES: Dict[str, Decimal] = {
    "wastewater": Decimal("0.07"),  # 7% leakage
    "manure": Decimal("0.037"),  # 3.7% leakage
    "biowaste": Decimal("0.028"),  # 2.8% leakage (modern enclosed plants)
    "energy_crop": Decimal("0.019"),  # 1.9% leakage
}

# Wastewater treatment system MCF values (IPCC 2006 Vol 5 Table 6.3)
WASTEWATER_MCF: Dict[WastewaterSystem, Decimal] = {
    WastewaterSystem.CENTRALIZED_AEROBIC_GOOD: Decimal("0.00"),
    WastewaterSystem.CENTRALIZED_AEROBIC_POOR: Decimal("0.03"),
    WastewaterSystem.CENTRALIZED_ANAEROBIC: Decimal("0.80"),
    WastewaterSystem.ANAEROBIC_REACTOR: Decimal("0.80"),
    WastewaterSystem.LAGOON_SHALLOW: Decimal("0.20"),
    WastewaterSystem.LAGOON_DEEP: Decimal("0.80"),
    WastewaterSystem.SEPTIC: Decimal("0.50"),
    WastewaterSystem.OPEN_SEWER: Decimal("0.10"),
    WastewaterSystem.CONSTRUCTED_WETLAND: Decimal("0.05"),
}

# Maximum CH4 producing capacity (Bo) (IPCC 2006 Vol 5 Ch 6)
WASTEWATER_Bo: Dict[str, Decimal] = {
    "cod_basis": Decimal("0.25"),  # kg CH4 / kg COD
    "bod_basis": Decimal("0.60"),  # kg CH4 / kg BOD
}

# Industry-specific wastewater organic loads (IPCC 2006 Vol 5 Table 6.9)
# COD and BOD in kg/m3, wastewater volume in m3/tonne product
INDUSTRY_WASTEWATER_LOADS: Dict[IndustryWastewaterType, Dict[str, Decimal]] = {
    IndustryWastewaterType.STARCH: {
        "cod_kg_per_m3": Decimal("10.0"),
        "bod_kg_per_m3": Decimal("6.0"),
        "volume_m3_per_tonne": Decimal("9.0"),
    },
    IndustryWastewaterType.ALCOHOL: {
        "cod_kg_per_m3": Decimal("15.0"),
        "bod_kg_per_m3": Decimal("8.0"),
        "volume_m3_per_tonne": Decimal("24.0"),
    },
    IndustryWastewaterType.BEER_MALT: {
        "cod_kg_per_m3": Decimal("3.0"),
        "bod_kg_per_m3": Decimal("1.5"),
        "volume_m3_per_tonne": Decimal("6.3"),
    },
    IndustryWastewaterType.PULP_PAPER: {
        "cod_kg_per_m3": Decimal("7.0"),
        "bod_kg_per_m3": Decimal("3.5"),
        "volume_m3_per_tonne": Decimal("85.0"),
    },
    IndustryWastewaterType.FOOD_PROCESSING: {
        "cod_kg_per_m3": Decimal("5.0"),
        "bod_kg_per_m3": Decimal("2.5"),
        "volume_m3_per_tonne": Decimal("20.0"),
    },
    IndustryWastewaterType.MEAT_POULTRY: {
        "cod_kg_per_m3": Decimal("4.1"),
        "bod_kg_per_m3": Decimal("1.5"),
        "volume_m3_per_tonne": Decimal("13.0"),
    },
    IndustryWastewaterType.VEGETABLES_FRUITS: {
        "cod_kg_per_m3": Decimal("5.0"),
        "bod_kg_per_m3": Decimal("2.5"),
        "volume_m3_per_tonne": Decimal("20.0"),
    },
    IndustryWastewaterType.DAIRY: {
        "cod_kg_per_m3": Decimal("2.7"),
        "bod_kg_per_m3": Decimal("1.5"),
        "volume_m3_per_tonne": Decimal("7.0"),
    },
    IndustryWastewaterType.SUGAR: {
        "cod_kg_per_m3": Decimal("3.2"),
        "bod_kg_per_m3": Decimal("1.6"),
        "volume_m3_per_tonne": Decimal("15.0"),
    },
    IndustryWastewaterType.TEXTILE: {
        "cod_kg_per_m3": Decimal("1.5"),
        "bod_kg_per_m3": Decimal("0.5"),
        "volume_m3_per_tonne": Decimal("100.0"),
    },
    IndustryWastewaterType.PHARMACEUTICAL: {
        "cod_kg_per_m3": Decimal("4.0"),
        "bod_kg_per_m3": Decimal("2.0"),
        "volume_m3_per_tonne": Decimal("50.0"),
    },
    IndustryWastewaterType.OTHER: {
        "cod_kg_per_m3": Decimal("5.0"),
        "bod_kg_per_m3": Decimal("2.5"),
        "volume_m3_per_tonne": Decimal("20.0"),
    },
}

# EPA WARM v16 emission factors (MTCO2e per short ton) - converted to kgCO2e/tonne internally
# Conversion: MTCO2e/short_ton * 1000 / 0.90718 = kgCO2e/tonne (multiply by 1102.31)
EPA_WARM_FACTORS: Dict[WasteCategory, Dict[str, Decimal]] = {
    WasteCategory.PAPER_CARDBOARD: {
        "landfill": Decimal("0.18"),  # MTCO2e/short ton
        "combustion": Decimal("0.04"),
        "recycling": Decimal("-3.11"),  # Negative = avoided (memo item only)
        "composting": Decimal("-0.18"),
    },
    WasteCategory.PLASTICS_HDPE: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("1.27"),
        "recycling": Decimal("-0.78"),
    },
    WasteCategory.PLASTICS_LDPE: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("1.27"),
        "recycling": Decimal("-0.89"),
    },
    WasteCategory.PLASTICS_PET: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("1.55"),
        "recycling": Decimal("-1.55"),
    },
    WasteCategory.PLASTICS_MIXED: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("1.18"),
        "recycling": Decimal("-0.86"),
    },
    WasteCategory.GLASS: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("0.02"),
        "recycling": Decimal("-0.28"),
    },
    WasteCategory.METALS_ALUMINUM: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("0.02"),
        "recycling": Decimal("-9.13"),
    },
    WasteCategory.METALS_STEEL: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("-1.52"),
        "recycling": Decimal("-1.83"),
    },
    WasteCategory.METALS_MIXED: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("-0.75"),
        "recycling": Decimal("-4.49"),
    },
    WasteCategory.FOOD_WASTE: {
        "landfill": Decimal("0.52"),
        "combustion": Decimal("0.04"),
        "composting": Decimal("-0.18"),
        "anaerobic_digestion": Decimal("-0.08"),
    },
    WasteCategory.GARDEN_WASTE: {
        "landfill": Decimal("-0.16"),  # Carbon sequestration in some scenarios
        "combustion": Decimal("-0.14"),
        "composting": Decimal("-0.18"),
    },
    WasteCategory.MIXED_MSW: {
        "landfill": Decimal("0.36"),
        "combustion": Decimal("0.04"),
    },
    WasteCategory.WOOD: {
        "landfill": Decimal("-0.14"),
        "combustion": Decimal("-0.41"),
        "recycling": Decimal("-2.47"),
    },
    WasteCategory.TEXTILES: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("1.09"),
        "recycling": Decimal("-2.29"),
    },
    WasteCategory.CONSTRUCTION_DEMOLITION: {
        "landfill": Decimal("0.02"),
        "combustion": Decimal("0.16"),
        "recycling": Decimal("-0.44"),
    },
}

# DEFRA/BEIS waste emission factors (kgCO2e per tonne) - 2025 factors
DEFRA_WASTE_FACTORS: Dict[WasteCategory, Dict[str, Decimal]] = {
    WasteCategory.PAPER_CARDBOARD: {
        "landfill": Decimal("1042"),
        "incineration": Decimal("21"),
        "incineration_energy": Decimal("21"),
        "recycling": Decimal("21"),
    },
    WasteCategory.PLASTICS_MIXED: {
        "landfill": Decimal("9"),
        "incineration": Decimal("2129"),
        "incineration_energy": Decimal("2129"),
        "recycling": Decimal("21"),
    },
    WasteCategory.PLASTICS_HDPE: {
        "landfill": Decimal("9"),
        "incineration": Decimal("2106"),
        "recycling": Decimal("21"),
    },
    WasteCategory.PLASTICS_PET: {
        "landfill": Decimal("9"),
        "incineration": Decimal("2153"),
        "recycling": Decimal("21"),
    },
    WasteCategory.GLASS: {
        "landfill": Decimal("9"),
        "incineration": Decimal("9"),
        "recycling": Decimal("21"),
    },
    WasteCategory.METALS_MIXED: {
        "landfill": Decimal("9"),
        "incineration": Decimal("9"),
        "recycling": Decimal("21"),
    },
    WasteCategory.METALS_ALUMINUM: {
        "landfill": Decimal("9"),
        "incineration": Decimal("9"),
        "recycling": Decimal("21"),
    },
    WasteCategory.METALS_STEEL: {
        "landfill": Decimal("9"),
        "incineration": Decimal("9"),
        "recycling": Decimal("21"),
    },
    WasteCategory.FOOD_WASTE: {
        "landfill": Decimal("586"),
        "incineration": Decimal("21"),
        "composting": Decimal("116"),
    },
    WasteCategory.GARDEN_WASTE: {
        "landfill": Decimal("578"),
        "incineration": Decimal("21"),
        "composting": Decimal("116"),
    },
    WasteCategory.WOOD: {
        "landfill": Decimal("843"),
        "incineration": Decimal("21"),
        "recycling": Decimal("21"),
    },
    WasteCategory.TEXTILES: {
        "landfill": Decimal("868"),
        "incineration": Decimal("1413"),
        "recycling": Decimal("21"),
    },
    WasteCategory.ELECTRONICS: {
        "landfill": Decimal("9"),
        "incineration": Decimal("9"),
        "recycling": Decimal("21"),
    },
    WasteCategory.CONSTRUCTION_DEMOLITION: {
        "landfill": Decimal("9"),
        "incineration": Decimal("9"),
        "recycling": Decimal("21"),
    },
    WasteCategory.MIXED_MSW: {
        "landfill": Decimal("578"),
        "incineration": Decimal("445"),
        "incineration_energy": Decimal("445"),
        "recycling": Decimal("21"),
    },
    WasteCategory.HAZARDOUS: {
        "landfill": Decimal("9"),
        "incineration": Decimal("1500"),
    },
}

# EEIO waste sector emission factors (kgCO2e per USD) - EPA USEEIO v1.2
EEIO_WASTE_FACTORS: Dict[str, Decimal] = {
    "naics_562111_solid_waste_collection": Decimal("0.480"),
    "naics_562119_other_waste_collection": Decimal("0.420"),
    "naics_562211_hazardous_waste_treatment": Decimal("0.650"),
    "naics_562212_solid_waste_landfill": Decimal("0.580"),
    "naics_562213_solid_waste_combustors": Decimal("0.720"),
    "naics_562219_other_nonhazardous_treatment": Decimal("0.380"),
    "naics_562910_remediation_services": Decimal("0.310"),
    "naics_562920_materials_recovery_facilities": Decimal("0.280"),
    "naics_562991_septic_tank_services": Decimal("0.350"),
    "naics_562998_other_waste_management": Decimal("0.400"),
}

# MSW composition by region (fraction of total) - IPCC 2019, World Bank
MSW_COMPOSITION: Dict[str, Dict[str, Decimal]] = {
    "developed": {
        "food_waste": Decimal("0.28"),
        "garden_waste": Decimal("0.10"),
        "paper_cardboard": Decimal("0.25"),
        "plastics": Decimal("0.12"),
        "glass": Decimal("0.05"),
        "metals": Decimal("0.05"),
        "textiles": Decimal("0.03"),
        "wood": Decimal("0.04"),
        "rubber_leather": Decimal("0.01"),
        "other": Decimal("0.07"),
    },
    "developing": {
        "food_waste": Decimal("0.55"),
        "garden_waste": Decimal("0.05"),
        "paper_cardboard": Decimal("0.08"),
        "plastics": Decimal("0.10"),
        "glass": Decimal("0.03"),
        "metals": Decimal("0.03"),
        "textiles": Decimal("0.03"),
        "wood": Decimal("0.02"),
        "rubber_leather": Decimal("0.01"),
        "other": Decimal("0.10"),
    },
}

# DQI scoring matrix (score 1-5 for each dimension)
DQI_SCORING: Dict[DQIDimension, Dict[int, str]] = {
    DQIDimension.TEMPORAL: {
        1: "Data from reporting year",
        2: "Data within 3 years",
        3: "Data within 6 years",
        4: "Data within 10 years",
        5: "Data older than 10 years",
    },
    DQIDimension.GEOGRAPHICAL: {
        1: "Same facility/region",
        2: "Same country",
        3: "Same continent",
        4: "Global average (relevant climate)",
        5: "Global average (different climate)",
    },
    DQIDimension.TECHNOLOGICAL: {
        1: "Same waste type and treatment method",
        2: "Same waste category, similar treatment",
        3: "Related waste category",
        4: "Generic waste stream average",
        5: "Unrelated or highly aggregated",
    },
    DQIDimension.COMPLETENESS: {
        1: "All waste streams included (100%)",
        2: "80-99% of waste covered",
        3: "50-79% of waste covered",
        4: "20-49% of waste covered",
        5: "Less than 20% covered",
    },
    DQIDimension.RELIABILITY: {
        1: "Waste audit or verified contractor data",
        2: "Transfer notes/invoices",
        3: "Established database (DEFRA/EPA)",
        4: "Industry estimates",
        5: "Assumption or rough estimate",
    },
}

# Uncertainty ranges by data quality tier and method (± % at 95% confidence)
UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    "supplier_specific": {
        "tier_1": Decimal("0.05"),  # ±5%
        "tier_2": Decimal("0.10"),  # ±10%
        "tier_3": Decimal("0.15"),  # ±15%
    },
    "waste_type_specific": {
        "tier_1": Decimal("0.15"),  # ±15%
        "tier_2": Decimal("0.25"),  # ±25%
        "tier_3": Decimal("0.35"),  # ±35%
    },
    "average_data": {
        "tier_1": Decimal("0.40"),  # ±40%
        "tier_2": Decimal("0.55"),  # ±55%
        "tier_3": Decimal("0.70"),  # ±70%
    },
    "spend_based": {
        "tier_1": Decimal("0.50"),  # ±50%
        "tier_2": Decimal("0.75"),  # ±75%
        "tier_3": Decimal("1.00"),  # ±100%
    },
    "landfill_fod": {
        "tier_1": Decimal("0.30"),  # ±30% (IPCC Tier 1 defaults)
        "tier_2": Decimal("0.15"),  # ±15% (country-specific)
        "tier_3": Decimal("0.05"),  # ±5% (facility-specific)
    },
}

# Framework-specific required disclosures
FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_category5_emissions_tco2e",
        "calculation_methodology",
        "waste_by_treatment_method",
        "data_quality_assessment",
        "exclusions_with_justification",
    ],
    ComplianceFramework.ISO_14064: [
        "waste_activity_data_by_type",
        "emission_factors_with_source",
        "uncertainty_quantification",
        "system_boundary_description",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "scope3_category5_absolute_emissions",
        "e5_5_waste_by_type_and_treatment",
        "diversion_rate",
        "hazardous_waste_mass",
        "waste_reduction_targets",
        "circular_economy_initiatives",
    ],
    ComplianceFramework.CDP: [
        "scope3_cat5_total",
        "percentage_of_total_scope3",
        "waste_by_disposal_method",
        "waste_reduction_initiatives",
        "year_over_year_explanation",
    ],
    ComplianceFramework.SBTI: [
        "scope3_cat5_baseline_year",
        "scope3_cat5_target_year",
        "reduction_trajectory",
        "diversion_rate_improvement",
    ],
    ComplianceFramework.EU_WASTE_DIRECTIVE: [
        "waste_generation_total",
        "waste_by_ewc_code",
        "hazardous_waste_separate",
        "recycling_rate",
        "landfill_diversion_progress",
    ],
    ComplianceFramework.EPA_40CFR98: [
        "landfill_ch4_generation",
        "gas_collection_efficiency",
        "landfill_type_classification",
        "waste_in_place_by_year",
    ],
}

# IPCC constants for landfill FOD model
IPCC_LANDFILL_CONSTANTS: Dict[str, Decimal] = {
    "docf": Decimal("0.50"),  # Fraction of DOC that decomposes (default)
    "f_ch4": Decimal("0.50"),  # Fraction of CH4 in landfill gas (default)
    "molecular_ratio_ch4_c": Decimal("1.333"),  # 16/12 (CH4/C molecular weight ratio)
    "molecular_ratio_co2_c": Decimal("3.667"),  # 44/12 (CO2/C molecular weight ratio)
    "molecular_ratio_n2o_n": Decimal("1.571"),  # 44/28 (N2O/N molecular weight ratio)
}

# Net Calorific Values for energy recovery calculation (MJ/kg wet weight)
NET_CALORIFIC_VALUES: Dict[WasteCategory, Decimal] = {
    WasteCategory.MIXED_MSW: Decimal("10.0"),
    WasteCategory.PAPER_CARDBOARD: Decimal("14.0"),
    WasteCategory.PLASTICS_MIXED: Decimal("35.0"),
    WasteCategory.PLASTICS_HDPE: Decimal("40.0"),
    WasteCategory.PLASTICS_PET: Decimal("22.0"),
    WasteCategory.FOOD_WASTE: Decimal("4.5"),
    WasteCategory.WOOD: Decimal("15.0"),
    WasteCategory.TEXTILES: Decimal("16.5"),
    WasteCategory.RUBBER_LEATHER: Decimal("22.5"),
    WasteCategory.GARDEN_WASTE: Decimal("6.0"),
}


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================


class WasteCompositionInput(BaseModel):
    """
    Waste composition breakdown by material type.

    Used for mixed waste streams to specify fraction of each waste category.
    """

    waste_category: WasteCategory
    fraction: Decimal = Field(..., ge=0, le=1, description="Fraction of total waste (0-1)")
    moisture_content: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Moisture content (wet basis, 0-1)"
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("fraction")
    @classmethod
    def validate_fraction(cls, v: Decimal) -> Decimal:
        """Ensure fraction is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("fraction must be between 0 and 1")
        return v


class WasteStreamInput(BaseModel):
    """
    Waste stream input record.

    Primary input for waste emissions calculation.
    """

    stream_id: str = Field(..., description="Unique waste stream identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    facility_id: str = Field(..., description="Source facility identifier")

    # Waste classification
    waste_category: WasteCategory
    waste_stream: WasteStream = Field(default=WasteStream.COMMERCIAL_INDUSTRIAL)
    treatment_method: WasteTreatmentMethod

    # Composition (for mixed streams)
    composition: List[WasteCompositionInput] = Field(
        default_factory=list,
        description="Waste composition breakdown (sum of fractions should = 1.0)"
    )

    # Quantity
    mass_tonnes: Decimal = Field(..., gt=0, description="Mass in tonnes (wet weight)")
    wet_dry_basis: str = Field(default="wet", description="'wet' or 'dry' weight basis")

    # Classification codes
    ewc_code: Optional[str] = Field(
        None, description="European Waste Catalogue code (6-digit)"
    )
    hazardous: bool = Field(default=False, description="Is hazardous waste?")
    hazard_classes: List[HazardClass] = Field(
        default_factory=list, description="Basel Convention hazard classes"
    )

    # Period and source
    reporting_year: int = Field(..., ge=2000, le=2100)
    reporting_period: Optional[str] = Field(None, description="e.g., '2024-Q1'")
    data_source: WasteDataSource = Field(default=WasteDataSource.TRANSFER_NOTES)

    # Contractor
    contractor_id: Optional[str] = Field(None, description="Waste contractor ID")
    contractor_name: Optional[str] = None

    # Distance to treatment facility (for transport emissions)
    distance_to_facility_km: Optional[Decimal] = Field(None, ge=0)

    # Supplier-specific emissions (if using supplier-specific method)
    contractor_reported_emissions_kgco2e: Optional[Decimal] = Field(None, ge=0)

    # Metadata
    notes: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode='after')
    def validate_composition_sum(self) -> 'WasteStreamInput':
        """Ensure composition fractions sum to 1.0 if provided."""
        if self.composition:
            total = sum(c.fraction for c in self.composition)
            if abs(total - Decimal("1.0")) > Decimal("0.01"):
                raise ValueError(
                    f"Composition fractions must sum to 1.0, got {total}"
                )
        return self


class LandfillInput(BaseModel):
    """
    Landfill-specific calculation parameters for FOD model.
    """

    mass_tonnes: Decimal = Field(..., gt=0, description="Mass deposited in landfill")
    waste_category: WasteCategory

    # Landfill characteristics
    landfill_type: LandfillType
    climate_zone: ClimateZone
    gas_collection: GasCollectionSystem = Field(default=GasCollectionSystem.NONE)
    has_cover: bool = Field(default=True, description="Has engineered soil cover?")

    # Parameter overrides (if facility-specific data available)
    doc_override: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Override DOC value"
    )
    k_override: Optional[Decimal] = Field(
        None, gt=0, description="Override decay rate constant (yr-1)"
    )
    mcf_override: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Override MCF value"
    )
    oxidation_factor_override: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Override OX value"
    )

    # Multi-year projection
    years_projection: int = Field(
        default=1, ge=1, le=100, description="Years to project decay (1 = single year)"
    )

    model_config = ConfigDict(frozen=True)


class IncinerationInput(BaseModel):
    """
    Incineration-specific calculation parameters.
    """

    mass_tonnes: Decimal = Field(..., gt=0, description="Mass incinerated")
    waste_category: WasteCategory
    incinerator_type: IncineratorType

    # Energy recovery
    energy_recovery: bool = Field(default=False, description="Waste-to-energy plant?")
    thermal_efficiency: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Thermal/electrical conversion efficiency"
    )

    # Composition (for mixed waste)
    waste_composition: List[WasteCompositionInput] = Field(
        default_factory=list,
        description="Detailed composition for accurate fossil/biogenic split"
    )

    # Parameter overrides
    dm_override: Optional[Decimal] = Field(None, ge=0, le=1, description="Override dry matter")
    cf_override: Optional[Decimal] = Field(None, ge=0, le=1, description="Override carbon fraction")
    fcf_override: Optional[Decimal] = Field(None, ge=0, le=1, description="Override fossil carbon fraction")
    of_override: Optional[Decimal] = Field(None, ge=0, le=1, description="Override oxidation factor")

    model_config = ConfigDict(frozen=True)


class RecyclingInput(BaseModel):
    """
    Recycling-specific calculation parameters.
    """

    mass_tonnes: Decimal = Field(..., gt=0, description="Mass recycled")
    waste_category: WasteCategory
    recycling_type: RecyclingType

    # Quality factor for downcycling (0-1, where 1 = no quality loss)
    quality_factor: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Quality factor for open-loop recycling"
    )

    # Calculate avoided emissions as memo item?
    calculate_avoided_emissions: bool = Field(
        default=True,
        description="Calculate avoided emissions (memo item only, not deducted)"
    )

    model_config = ConfigDict(frozen=True)


class CompostingInput(BaseModel):
    """
    Composting-specific calculation parameters.
    """

    mass_tonnes: Decimal = Field(..., gt=0, description="Mass composted (wet weight)")
    waste_category: WasteCategory

    # Composting type
    is_home_composting: bool = Field(
        default=False, description="Home composting (higher emissions than industrial)?"
    )
    dry_weight_basis: bool = Field(
        default=False, description="Is mass on dry weight basis?"
    )

    # Emission factor overrides
    ch4_ef_override: Optional[Decimal] = Field(
        None, gt=0, description="Override CH4 EF (g CH4/kg wet waste)"
    )
    n2o_ef_override: Optional[Decimal] = Field(
        None, gt=0, description="Override N2O EF (g N2O/kg wet waste)"
    )

    model_config = ConfigDict(frozen=True)


class AnaerobicDigestionInput(BaseModel):
    """
    Anaerobic digestion (AD) specific calculation parameters.
    """

    mass_tonnes: Decimal = Field(..., gt=0, description="Mass sent to AD (wet weight)")
    waste_category: WasteCategory

    # Plant type (affects leakage rate)
    plant_type: str = Field(
        default="biowaste",
        description="Plant type: 'biowaste', 'wastewater', 'manure', 'energy_crop'"
    )

    # Biogas characteristics
    biogas_ch4_content: Decimal = Field(
        default=Decimal("0.60"), ge=0, le=1,
        description="CH4 fraction in biogas (default 60%)"
    )

    # Leakage rate override
    leakage_rate_override: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Override CH4 leakage rate"
    )

    # Storage
    gastight_storage: bool = Field(
        default=True, description="Gastight biogas storage?"
    )

    model_config = ConfigDict(frozen=True)


class WastewaterInput(BaseModel):
    """
    Wastewater treatment emissions calculation parameters.
    """

    # Organic load
    organic_load_kg: Decimal = Field(..., gt=0, description="Organic load in kg COD or BOD")
    measurement_basis: str = Field(
        default="cod", description="'cod' or 'bod' measurement basis"
    )

    # Treatment system
    treatment_system: WastewaterSystem

    # Nitrogen load (for N2O calculation)
    nitrogen_load_kg: Optional[Decimal] = Field(
        None, ge=0, description="Nitrogen in effluent (kg N)"
    )

    # Industry type (for automatic organic load calculation if not provided)
    industry_type: Optional[IndustryWastewaterType] = None
    production_volume_tonnes: Optional[Decimal] = Field(
        None, gt=0, description="Production volume for industry-specific calc"
    )

    # Parameter overrides
    bo_override: Optional[Decimal] = Field(
        None, gt=0, description="Override Bo value (kg CH4/kg COD)"
    )
    mcf_override: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Override MCF value"
    )

    model_config = ConfigDict(frozen=True)


class WasteEmissionFactor(BaseModel):
    """
    Waste emission factor record.

    Stored in database for EF hierarchy resolution.
    """

    ef_id: str = Field(..., description="Unique EF identifier")
    waste_category: WasteCategory
    treatment_method: WasteTreatmentMethod

    # Emission factor
    ef_kgco2e_per_tonne: Decimal = Field(..., ge=0, description="kgCO2e per tonne waste")

    # Metadata
    source: EFSource
    source_year: int = Field(..., ge=2000, le=2030)
    region: str = Field(default="global", description="Geographic region")
    data_quality_tier: DataQualityTier = Field(default=DataQualityTier.TIER_1)

    # Validity
    valid_from: date
    valid_to: Optional[date] = None

    model_config = ConfigDict(frozen=True)


class WasteClassificationResult(BaseModel):
    """
    Waste classification engine output.
    """

    waste_category: WasteCategory
    waste_stream: WasteStream
    ewc_code: Optional[str] = None
    ewc_chapter: Optional[str] = Field(None, description="EWC chapter (01-20)")

    # Treatment compatibility
    compatible_treatments: List[WasteTreatmentMethod] = Field(
        default_factory=list,
        description="Compatible treatment methods for this waste type"
    )

    # Hazard classification
    hazardous: bool
    hazard_classes: List[HazardClass] = Field(default_factory=list)

    # Data quality
    data_quality_tier: DataQualityTier

    model_config = ConfigDict(frozen=True)


class LandfillEmissionsResult(BaseModel):
    """
    Landfill emissions calculation output (FOD model).
    """

    # CH4 generation and fate
    ch4_generated_tonnes: Decimal = Field(..., ge=0, description="Total CH4 generated")
    ch4_recovered_tonnes: Decimal = Field(..., ge=0, description="CH4 recovered (flared or energy)")
    ch4_oxidized_tonnes: Decimal = Field(..., ge=0, description="CH4 oxidized in cover")
    ch4_emitted_tonnes: Decimal = Field(..., ge=0, description="Net CH4 emitted to atmosphere")

    # CO2e total
    co2e_total: Decimal = Field(..., ge=0, description="Total CO2e emissions")

    # Parameters used
    doc_used: Decimal
    mcf_used: Decimal
    k_used: Decimal
    gas_capture_efficiency: Decimal
    oxidation_factor: Decimal

    # Multi-year projection (if years_projection > 1)
    decay_projection: Optional[List[Dict[str, Decimal]]] = Field(
        None,
        description="Year-by-year CH4 emissions projection [{year, ch4_tonnes, co2e_tonnes}, ...]"
    )

    model_config = ConfigDict(frozen=True)


class IncinerationEmissionsResult(BaseModel):
    """
    Incineration emissions calculation output.
    """

    # Fossil CO2 (reported in Category 5)
    co2_fossil_tonnes: Decimal = Field(..., ge=0, description="Fossil-origin CO2")

    # Biogenic CO2 (memo item only, not counted in Category 5)
    co2_biogenic_tonnes: Decimal = Field(..., ge=0, description="Biogenic-origin CO2 (memo)")

    # CH4 and N2O
    ch4_tonnes: Decimal = Field(..., ge=0, description="CH4 emissions")
    n2o_tonnes: Decimal = Field(..., ge=0, description="N2O emissions")

    # CO2e total
    co2e_total: Decimal = Field(..., ge=0, description="Total CO2e (fossil CO2 + CH4 + N2O)")

    # Energy recovery
    energy_recovered_kwh: Optional[Decimal] = Field(
        None, ge=0, description="Energy recovered (kWh)"
    )
    avoided_co2e_memo: Optional[Decimal] = Field(
        None, description="Avoided emissions from energy recovery (memo item only)"
    )

    model_config = ConfigDict(frozen=True)


class RecyclingCompostingResult(BaseModel):
    """
    Recycling, composting, or anaerobic digestion calculation output.
    """

    # Process emissions (cut-off approach for recycling)
    treatment_emissions_co2e: Decimal = Field(
        ..., ge=0, description="Treatment process emissions (Category 5)"
    )

    # Avoided emissions (memo item only, NOT deducted from Category 5)
    avoided_emissions_memo_co2e: Optional[Decimal] = Field(
        None, description="Avoided emissions from recycling (memo item only, not deducted)"
    )

    # Net emissions (for composting/AD, includes CH4 and N2O)
    net_emissions_co2e: Decimal = Field(..., ge=0, description="Net emissions")

    # Gas breakdown (for composting/AD)
    ch4_tonnes: Optional[Decimal] = Field(None, ge=0, description="CH4 emissions")
    n2o_tonnes: Optional[Decimal] = Field(None, ge=0, description="N2O emissions")

    # Method detail
    method_detail: str = Field(
        ..., description="'recycling', 'composting', 'anaerobic_digestion'"
    )

    model_config = ConfigDict(frozen=True)


class WastewaterEmissionsResult(BaseModel):
    """
    Wastewater treatment emissions calculation output.
    """

    # CH4 from organic degradation
    ch4_from_treatment_tonnes: Decimal = Field(..., ge=0, description="CH4 from treatment")

    # N2O from effluent nitrogen
    n2o_from_effluent_tonnes: Decimal = Field(..., ge=0, description="N2O from effluent")

    # CH4 recovered (if AD with biogas capture)
    ch4_recovered_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="CH4 recovered for energy"
    )

    # CO2e total
    co2e_total: Decimal = Field(..., ge=0, description="Total CO2e emissions")

    # Parameters used
    organic_load_used_kg: Decimal
    mcf_used: Decimal
    bo_used: Decimal
    treatment_system: WastewaterSystem

    model_config = ConfigDict(frozen=True)


class WasteCalculationResult(BaseModel):
    """
    Complete waste emissions calculation result for a single waste stream.
    """

    calculation_id: str = Field(..., description="Unique calculation identifier")
    tenant_id: str
    facility_id: str

    # Waste details
    waste_category: WasteCategory
    treatment_method: WasteTreatmentMethod
    calculation_method: CalculationMethod

    # Mass
    mass_tonnes: Decimal = Field(..., gt=0)

    # Total emissions
    total_co2e: Decimal = Field(..., ge=0, description="Total CO2e emissions (tonnes)")

    # Treatment-specific breakdown
    breakdown: Union[
        LandfillEmissionsResult,
        IncinerationEmissionsResult,
        RecyclingCompostingResult,
        WastewaterEmissionsResult,
        Dict[str, Any]  # For spend-based or supplier-specific
    ] = Field(..., description="Treatment-specific calculation breakdown")

    # Emission factor used
    ef_source: EFSource
    ef_kgco2e_per_tonne: Optional[Decimal] = Field(None, ge=0)

    # Data quality
    data_quality_tier: DataQualityTier
    dqi_composite_score: Optional[Decimal] = Field(
        None, ge=1, le=5, description="Composite DQI score (1-5)"
    )
    uncertainty_pct: Optional[Decimal] = Field(
        None, ge=0, description="Uncertainty as ± percentage"
    )

    # GWP version
    gwp_version: GWPVersion = Field(default=GWPVersion.AR5)

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")

    # Timing
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(None, ge=0)

    model_config = ConfigDict(frozen=True)


class WasteBatchResult(BaseModel):
    """
    Batch waste emissions calculation output.
    """

    batch_id: str = Field(..., description="Unique batch identifier")
    results: List[WasteCalculationResult]

    # Totals
    total_co2e: Decimal = Field(..., ge=0, description="Total CO2e across all streams")
    total_mass_tonnes: Decimal = Field(..., ge=0, description="Total mass across all streams")

    # Counts
    success_count: int = Field(..., ge=0)
    failure_count: int = Field(..., ge=0)

    # Errors
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of errors [{stream_id, error_message}, ...]"
    )

    # Timing
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(None, ge=0)

    # Status
    status: BatchStatus

    model_config = ConfigDict(frozen=True)


class WasteAggregation(BaseModel):
    """
    Single aggregation dimension (e.g., by treatment method, by waste type).
    """

    dimension: str = Field(..., description="Aggregation dimension (e.g., 'treatment_method')")
    key: str = Field(..., description="Dimension key (e.g., 'landfill')")
    co2e_tonnes: Decimal = Field(..., ge=0)
    mass_tonnes: Decimal = Field(..., ge=0)
    fraction_of_total: Decimal = Field(..., ge=0, le=1, description="Fraction of total emissions")

    model_config = ConfigDict(frozen=True)


class WasteAggregationResult(BaseModel):
    """
    Aggregated waste emissions by multiple dimensions.
    """

    aggregation_id: str = Field(..., description="Unique aggregation identifier")
    tenant_id: str

    # Period
    reporting_period: str = Field(..., description="e.g., '2024-Q1', '2024'")
    period_start: date
    period_end: date

    # Total emissions
    total_co2e: Decimal = Field(..., ge=0)
    total_mass_tonnes: Decimal = Field(..., ge=0)

    # By treatment method
    by_treatment_method: List[WasteAggregation] = Field(default_factory=list)

    # By waste category
    by_waste_category: List[WasteAggregation] = Field(default_factory=list)

    # By waste stream
    by_waste_stream: List[WasteAggregation] = Field(default_factory=list)

    # By facility
    by_facility: List[WasteAggregation] = Field(default_factory=list)

    # Intensity metrics
    intensity_per_tonne_waste: Optional[Decimal] = Field(
        None, description="kgCO2e per tonne waste"
    )
    intensity_per_revenue: Optional[Decimal] = Field(
        None, description="tCO2e per $M revenue"
    )

    # Diversion rate
    diversion_rate: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Waste diversion rate (0-1)"
    )

    # Timing
    calculated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class ComplianceCheckInput(BaseModel):
    """
    Compliance check request input.
    """

    calculation_ids: List[str] = Field(..., min_length=1, description="Calculation IDs to check")
    frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: list(ComplianceFramework),
        description="Frameworks to check against"
    )
    reporting_year: int = Field(..., ge=2000, le=2100)

    # Organization context
    organization_name: Optional[str] = None
    industry_sector: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class ComplianceCheckResult(BaseModel):
    """
    Compliance check output for a single framework.
    """

    result_id: str = Field(..., description="Unique result identifier")
    framework: ComplianceFramework

    # Status
    status: ComplianceStatus
    compliance_score: Decimal = Field(..., ge=0, le=100, description="Compliance score (0-100)")

    # Findings
    findings: List[str] = Field(
        default_factory=list,
        description="List of compliance findings"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="List of data gaps or non-compliant items"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )

    # Missing disclosures
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Required fields that are missing"
    )

    # Timestamp
    checked_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class UncertaintyInput(BaseModel):
    """
    Uncertainty analysis request input.
    """

    calculation_result: WasteCalculationResult
    method: UncertaintyMethod = Field(default=UncertaintyMethod.IPCC_DEFAULT)

    # Monte Carlo parameters
    iterations: int = Field(default=10000, ge=100, le=100000, description="Monte Carlo iterations")
    confidence_level: Decimal = Field(
        default=Decimal("0.95"), ge=0, le=1, description="Confidence level (0-1)"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """
    Uncertainty analysis output.
    """

    result_id: str = Field(..., description="Unique result identifier")
    method: UncertaintyMethod

    # Statistics
    mean_co2e: Decimal = Field(..., ge=0)
    median_co2e: Decimal = Field(..., ge=0)
    std_dev_co2e: Decimal = Field(..., ge=0)

    # Confidence interval
    lower_bound_co2e: Decimal = Field(..., ge=0, description="Lower bound at confidence level")
    upper_bound_co2e: Decimal = Field(..., ge=0, description="Upper bound at confidence level")
    confidence_pct: Decimal = Field(..., ge=0, le=100)

    # Relative uncertainty
    relative_uncertainty_pct: Decimal = Field(
        ..., ge=0, description="± % relative to mean"
    )

    # Timestamp
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class DataQualityInput(BaseModel):
    """
    Data quality assessment input.
    """

    calculation_id: str

    # DQI dimension scores (1-5)
    temporal_score: int = Field(..., ge=1, le=5, description="Temporal correlation (1-5)")
    geographical_score: int = Field(..., ge=1, le=5, description="Geographical correlation (1-5)")
    technological_score: int = Field(..., ge=1, le=5, description="Technological correlation (1-5)")
    completeness_score: int = Field(..., ge=1, le=5, description="Data completeness (1-5)")
    reliability_score: int = Field(..., ge=1, le=5, description="Source reliability (1-5)")

    model_config = ConfigDict(frozen=True)


class DataQualityResult(BaseModel):
    """
    Data quality assessment output.
    """

    result_id: str = Field(..., description="Unique result identifier")

    # Composite DQI score (average of 5 dimensions)
    composite_score: Decimal = Field(..., ge=1, le=5, description="Composite DQI (1-5)")

    # Classification
    classification: str = Field(
        ..., description="'very_good', 'good', 'fair', 'poor', 'very_poor'"
    )

    # Dimension scores
    dimension_scores: Dict[str, int] = Field(
        ..., description="Scores by dimension {temporal: 2, geographical: 3, ...}"
    )

    # Recommendations
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions to improve DQI"
    )

    # Timestamp
    assessed_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(BaseModel):
    """
    Provenance chain record for a single pipeline stage.
    """

    record_id: str = Field(..., description="Unique record identifier")
    stage: ProvenanceStage

    # Hashes
    input_hash: str = Field(..., description="SHA-256 hash of stage input")
    output_hash: str = Field(..., description="SHA-256 hash of stage output")

    # Metadata
    parameters_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used in this stage"
    )

    # Agent info
    agent_id: str = Field(default=AGENT_ID)
    agent_version: str = Field(default=VERSION)

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class ProvenanceChainResult(BaseModel):
    """
    Complete provenance chain for a calculation.
    """

    chain_id: str = Field(..., description="Unique chain identifier")
    calculation_id: str

    # Chain records (ordered by stage)
    records: List[ProvenanceRecord] = Field(..., min_length=1)

    # Final chain hash (hash of all record hashes)
    chain_hash: str = Field(..., description="SHA-256 hash of complete chain")

    # Validation
    is_valid: bool = Field(..., description="Is chain valid and unbroken?")

    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class WasteDiversionAnalysis(BaseModel):
    """
    Waste diversion rate analysis output.
    """

    analysis_id: str = Field(..., description="Unique analysis identifier")
    tenant_id: str

    # Period
    reporting_period: str
    period_start: date
    period_end: date

    # Total waste
    total_generated_tonnes: Decimal = Field(..., ge=0)

    # Diverted (recycling, composting, AD, reuse)
    diverted_mass_tonnes: Decimal = Field(..., ge=0)
    diverted_by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Mass diverted by method {recycling: 100, composting: 50, ...}"
    )

    # Disposed (landfill, incineration)
    disposed_mass_tonnes: Decimal = Field(..., ge=0)
    disposed_by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Mass disposed by method {landfill: 200, incineration: 50, ...}"
    )

    # Diversion rate
    diversion_rate: Decimal = Field(..., ge=0, le=1, description="Diversion rate (0-1)")
    diversion_rate_pct: Decimal = Field(..., ge=0, le=100, description="Diversion rate (%)")

    # By facility
    by_facility: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Diversion analysis by facility"
    )

    # Timestamp
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class WasteCompositionProfile(BaseModel):
    """
    Waste composition profile for a region or facility.
    """

    profile_id: str = Field(..., description="Unique profile identifier")
    region: str = Field(..., description="Region or facility identifier")
    waste_stream: WasteStream

    # Composition
    components: List[WasteCompositionInput] = Field(
        ..., min_length=1,
        description="Waste composition breakdown"
    )

    # Source
    source: str = Field(..., description="Data source (e.g., 'waste_audit', 'IPCC_default')")

    # Validity
    valid_from: date
    valid_to: Optional[date] = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode='after')
    def validate_composition_sum(self) -> 'WasteCompositionProfile':
        """Ensure composition fractions sum to 1.0."""
        total = sum(c.fraction for c in self.components)
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Composition fractions must sum to 1.0, got {total}"
            )
        return self


class SpendBasedInput(BaseModel):
    """
    Spend-based calculation input for waste management spend.
    """

    record_id: str = Field(..., description="Unique record identifier")
    tenant_id: str

    # Spend details
    spend_amount: Decimal = Field(..., gt=0, description="Waste management spend")
    currency: CurrencyCode = Field(default=CurrencyCode.USD)
    spend_year: int = Field(..., ge=2000, le=2030)

    # Service type
    waste_service_type: str = Field(
        ..., description="NAICS code or service description (e.g., 'naics_562212_solid_waste_landfill')"
    )

    # EEIO factor
    eeio_factor_kgco2e_per_usd: Optional[Decimal] = Field(
        None, gt=0, description="Custom EEIO factor"
    )
    eeio_source: str = Field(default="EPA_USEEIO_v1.2")

    # Context
    facility_id: Optional[str] = None
    contractor_id: Optional[str] = None

    # Metadata
    reporting_period: Optional[str] = None
    data_source: WasteDataSource = Field(default=WasteDataSource.SPEND_ESTIMATE)

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from inputs.

    Args:
        *inputs: Variable number of input objects to hash

    Returns:
        Hexadecimal SHA-256 hash string
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, BaseModel):
            hash_input += inp.model_dump_json(sort_keys=True)
        else:
            hash_input += str(inp)

    return hashlib.sha256(hash_input.encode()).hexdigest()


def get_dqi_classification(composite_score: Decimal) -> str:
    """
    Get DQI classification from composite score.

    Args:
        composite_score: Composite DQI score (1-5)

    Returns:
        Classification string: 'very_good', 'good', 'fair', 'poor', 'very_poor'
    """
    if composite_score <= Decimal("1.5"):
        return "very_good"
    elif composite_score <= Decimal("2.5"):
        return "good"
    elif composite_score <= Decimal("3.5"):
        return "fair"
    elif composite_score <= Decimal("4.5"):
        return "poor"
    else:
        return "very_poor"


def convert_warm_factor_to_kgco2e_per_tonne(mtco2e_per_short_ton: Decimal) -> Decimal:
    """
    Convert EPA WARM factor from MTCO2e/short ton to kgCO2e/tonne.

    Args:
        mtco2e_per_short_ton: EPA WARM factor in MTCO2e per short ton

    Returns:
        Factor in kgCO2e per tonne (metric)
    """
    # 1 short ton = 0.90718 metric tonnes
    # 1 MTCO2e = 1000 kgCO2e
    # Factor = MTCO2e/short_ton * 1000 / 0.90718 = MTCO2e/short_ton * 1102.31
    return mtco2e_per_short_ton * Decimal("1102.31")


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",

    # Enums
    "CalculationMethod",
    "WasteTreatmentMethod",
    "WasteCategory",
    "WasteStream",
    "LandfillType",
    "ClimateZone",
    "IncineratorType",
    "RecyclingType",
    "WastewaterSystem",
    "GasCollectionSystem",
    "EFSource",
    "ComplianceFramework",
    "DataQualityTier",
    "WasteDataSource",
    "ProvenanceStage",
    "UncertaintyMethod",
    "HazardClass",
    "GWPVersion",
    "IndustryWastewaterType",
    "EmissionGas",
    "DQIDimension",
    "DQIScore",
    "ComplianceStatus",
    "CurrencyCode",
    "ExportFormat",
    "BatchStatus",

    # Constants
    "GWP_VALUES",
    "DOC_VALUES",
    "MCF_VALUES",
    "DECAY_RATE_CONSTANTS",
    "GAS_CAPTURE_EFFICIENCY",
    "OXIDATION_FACTORS",
    "INCINERATION_PARAMS",
    "CH4_EF_INCINERATION",
    "N2O_EF_INCINERATION",
    "COMPOSTING_EF",
    "AD_LEAKAGE_RATES",
    "WASTEWATER_MCF",
    "WASTEWATER_Bo",
    "INDUSTRY_WASTEWATER_LOADS",
    "EPA_WARM_FACTORS",
    "DEFRA_WASTE_FACTORS",
    "EEIO_WASTE_FACTORS",
    "MSW_COMPOSITION",
    "DQI_SCORING",
    "UNCERTAINTY_RANGES",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    "IPCC_LANDFILL_CONSTANTS",
    "NET_CALORIFIC_VALUES",

    # Models
    "WasteCompositionInput",
    "WasteStreamInput",
    "LandfillInput",
    "IncinerationInput",
    "RecyclingInput",
    "CompostingInput",
    "AnaerobicDigestionInput",
    "WastewaterInput",
    "WasteEmissionFactor",
    "WasteClassificationResult",
    "LandfillEmissionsResult",
    "IncinerationEmissionsResult",
    "RecyclingCompostingResult",
    "WastewaterEmissionsResult",
    "WasteCalculationResult",
    "WasteBatchResult",
    "WasteAggregation",
    "WasteAggregationResult",
    "ComplianceCheckInput",
    "ComplianceCheckResult",
    "UncertaintyInput",
    "UncertaintyResult",
    "DataQualityInput",
    "DataQualityResult",
    "ProvenanceRecord",
    "ProvenanceChainResult",
    "WasteDiversionAnalysis",
    "WasteCompositionProfile",
    "SpendBasedInput",

    # Helper functions
    "calculate_provenance_hash",
    "get_dqi_classification",
    "convert_warm_factor_to_kgco2e_per_tonne",
]
