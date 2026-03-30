"""
End-of-Life Treatment of Sold Products Agent Models (AGENT-MRV-025)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 12
(End-of-Life Treatment of Sold Products) emissions calculations.

Supports:
- 5 calculation methods (waste-type-specific, average-data, spend-based,
  producer-specific EPD, hybrid)
- 7 treatment pathways (landfill, incineration, recycling, composting,
  anaerobic digestion, open burning, wastewater)
- 15 material types with treatment-specific emission factors
- 20 product categories with default bill-of-material compositions
- 12 regional treatment mix profiles (US, EU, UK, DE, FR, JP, CN, IN, BR, AU, KR, GLOBAL)
- IPCC First Order Decay (FOD) landfill model with 4 climate zones
- Incineration fossil/biogenic carbon split
- Recycling avoided emissions (cut-off, closed-loop, substitution)
- Composting and anaerobic digestion CH4/N2O process emissions
- 8 double-counting prevention rules (vs Cat 1/Cat 5/Scope 1)
- EPA WARM v16, DEFRA, IPCC, Ecoinvent, producer EPD emission factor sources
- Data quality indicators (DQI) with 5-dimension scoring across 3 tiers
- Uncertainty quantification (Monte Carlo, analytical, IPCC Tier 2)
- Compliance checking for 7 frameworks (GHG Protocol, ISO 14064, CSRD ESRS,
  CDP, SBTi, SB 253, GRI)
- Circularity metrics (recycling rate, diversion rate, circularity index)
- SHA-256 provenance chain with 10-stage pipeline
- Waste hierarchy alignment (prevention > reuse > recycling > recovery > disposal)

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.end_of_life_treatment.models import (
    ...     ProductEOLInput, ProductCategory, MaterialType, TreatmentMethod
    ... )
    >>> product = ProductEOLInput(
    ...     product_id="PROD-2026-001",
    ...     org_id="ORG-001",
    ...     category=ProductCategory.ELECTRONICS,
    ...     units_sold=10000,
    ...     weight_per_unit_kg=Decimal("2.5"),
    ...     region=RegionalTreatmentProfile.US,
    ...     reporting_year=2025
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import Field, validator, model_validator
from greenlang.schemas import GreenLangBase, utcnow, new_uuid

import hashlib
import json

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-012"
AGENT_COMPONENT: str = "AGENT-MRV-025"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_eol_"

# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class MaterialType(str, Enum):
    """Material types for product end-of-life treatment emission factors."""

    PLASTIC = "plastic"  # Mixed plastics (fossil-derived)
    METAL = "metal"  # Mixed metals
    ALUMINUM = "aluminum"  # Aluminum / aluminium
    STEEL = "steel"  # Steel / iron
    GLASS = "glass"  # Glass (all types)
    PAPER = "paper"  # Paper (office, newsprint, etc.)
    CARDBOARD = "cardboard"  # Corrugated cardboard / paperboard
    WOOD = "wood"  # Wood / timber / MDF
    TEXTILE = "textile"  # Textiles (natural + synthetic)
    ELECTRONICS = "electronics"  # PCBs, semiconductors, connectors
    ORGANIC = "organic"  # Organic / biodegradable fraction
    RUBBER = "rubber"  # Rubber / elastomers
    CERAMIC = "ceramic"  # Ceramic / porcelain
    CONCRETE = "concrete"  # Concrete / cement / aggregate
    MIXED = "mixed"  # Mixed / unclassified materials


class TreatmentMethod(str, Enum):
    """End-of-life waste treatment methods per GHG Protocol and IPCC Vol 5."""

    LANDFILL = "landfill"  # Managed / unmanaged landfill disposal
    INCINERATION = "incineration"  # Thermal treatment (with or without energy recovery)
    RECYCLING = "recycling"  # Material recovery and reprocessing
    COMPOSTING = "composting"  # Aerobic biological treatment
    ANAEROBIC_DIGESTION = "anaerobic_digestion"  # Anaerobic biological treatment with biogas
    OPEN_BURNING = "open_burning"  # Uncontrolled open burning (developing regions)
    WASTEWATER = "wastewater"  # Wastewater treatment pathway


class ProductCategory(str, Enum):
    """Product categories for default material composition and weight lookups."""

    ELECTRONICS = "electronics"  # Consumer electronics (phones, laptops, tablets)
    APPLIANCES = "appliances"  # Large and small household appliances
    FURNITURE = "furniture"  # Furniture (home, office)
    PACKAGING = "packaging"  # Product packaging (primary, secondary, tertiary)
    CLOTHING = "clothing"  # Apparel and fashion textiles
    AUTOMOTIVE_PARTS = "automotive_parts"  # Automotive components and accessories
    BUILDING_MATERIALS = "building_materials"  # Construction and building products
    TOYS = "toys"  # Toys and games
    MEDICAL_DEVICES = "medical_devices"  # Medical devices and equipment
    BATTERIES = "batteries"  # Batteries (Li-ion, NiMH, lead-acid, etc.)
    TIRES = "tires"  # Tires and rubber products
    FOOD_PRODUCTS = "food_products"  # Food items and ingredients
    BEVERAGES = "beverages"  # Beverages and liquid products
    CHEMICALS = "chemicals"  # Chemical products (paints, solvents, etc.)
    COSMETICS = "cosmetics"  # Cosmetics and personal care products
    OFFICE_SUPPLIES = "office_supplies"  # Office and stationery supplies
    SPORTING_GOODS = "sporting_goods"  # Sports equipment and accessories
    TOOLS = "tools"  # Hand and power tools
    LIGHTING = "lighting"  # Lighting products (bulbs, fixtures, LEDs)
    MIXED_PRODUCTS = "mixed_products"  # Mixed / unclassified product portfolio


class RegionalTreatmentProfile(str, Enum):
    """Regional profiles for default end-of-life treatment mix distributions."""

    US = "US"  # United States
    EU = "EU"  # European Union (27-member average)
    UK = "UK"  # United Kingdom
    DE = "DE"  # Germany
    FR = "FR"  # France
    JP = "JP"  # Japan
    CN = "CN"  # China
    IN = "IN"  # India
    BR = "BR"  # Brazil
    AU = "AU"  # Australia
    KR = "KR"  # South Korea
    GLOBAL = "GLOBAL"  # Global weighted average


class CalculationMethod(str, Enum):
    """Calculation method for end-of-life treatment emissions per GHG Protocol."""

    WASTE_TYPE_SPECIFIC = "waste_type_specific"  # Material x treatment x EF
    AVERAGE_DATA = "average_data"  # Product-category average EFs
    SPEND_BASED = "spend_based"  # Waste management spend x EEIO factor
    PRODUCER_SPECIFIC = "producer_specific"  # EPD / LCA end-of-life module D
    HYBRID = "hybrid"  # Blended multi-method aggregation


class LandfillType(str, Enum):
    """Landfill types per IPCC 2006 Vol 5 Table 3.1, affecting MCF values."""

    MANAGED_ANAEROBIC = "managed_anaerobic"  # MCF = 1.0
    MANAGED_SEMI_AEROBIC = "managed_semi_aerobic"  # MCF = 0.5
    UNMANAGED_DEEP = "unmanaged_deep"  # MCF = 0.8 (depth > 5m)
    UNMANAGED_SHALLOW = "unmanaged_shallow"  # MCF = 0.4 (depth < 5m)
    ENGINEERED_WITH_GAS = "engineered_with_gas"  # MCF = 1.0 with gas collection
    ENGINEERED_WITHOUT_GAS = "engineered_without_gas"  # MCF = 1.0 no gas collection


class ClimateZone(str, Enum):
    """IPCC climate zones for landfill decay rate constants (k)."""

    BOREAL_TEMPERATE_DRY = "boreal_temperate_dry"  # MAT < 20C, MAP/PET < 1
    BOREAL_TEMPERATE_WET = "boreal_temperate_wet"  # MAT < 20C, MAP/PET > 1
    TROPICAL_DRY = "tropical_dry"  # MAT >= 20C, MAP/PET < 1
    TROPICAL_WET = "tropical_wet"  # MAT >= 20C, MAP/PET > 1


class IncinerationType(str, Enum):
    """Incineration technology types affecting emission profiles."""

    MASS_BURN = "mass_burn"  # Modern mass-burn stoker incinerator
    REFUSE_DERIVED = "refuse_derived"  # Refuse-derived fuel (RDF) plant
    WASTE_TO_ENERGY = "waste_to_energy"  # WtE with electricity/heat recovery
    OPEN_BURNING = "open_burning"  # Uncontrolled open burning


class RecyclingApproach(str, Enum):
    """Recycling accounting approach per GHG Protocol."""

    CUT_OFF = "cut_off"  # Cut-off: only processing emissions counted
    CLOSED_LOOP = "closed_loop"  # Closed-loop: material returns to same product
    SUBSTITUTION = "substitution"  # Substitution: avoided virgin material credit


class EFSource(str, Enum):
    """Emission factor data source for audit trail and provenance."""

    EPA_WARM = "epa_warm"  # EPA Waste Reduction Model v16
    DEFRA = "defra"  # DEFRA/DESNZ conversion factors
    IPCC = "ipcc"  # IPCC 2006/2019 Guidelines Vol 5
    ECOINVENT = "ecoinvent"  # Ecoinvent LCI database
    PRODUCER_EPD = "producer_epd"  # Environmental Product Declaration
    CUSTOM = "custom"  # Custom / user-provided factor


class DataQualityTier(str, Enum):
    """IPCC data quality tiers for emission factor classification."""

    TIER_1 = "tier_1"  # Default / global average emission factors
    TIER_2 = "tier_2"  # Country or region-specific emission factors
    TIER_3 = "tier_3"  # Facility or product-specific emission factors


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol Corporate Value Chain."""

    TEMPORAL = "temporal"  # Temporal correlation to reporting year
    GEOGRAPHICAL = "geographical"  # Geographical correlation to activity region
    TECHNOLOGICAL = "technological"  # Technology correlation to actual process
    COMPLETENESS = "completeness"  # Data completeness coverage
    RELIABILITY = "reliability"  # Source reliability and verification


class ComplianceFramework(str, Enum):
    """Regulatory and reporting frameworks for compliance checking."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 Climate Change
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    SB_253 = "sb_253"  # California SB 253 Climate Corporate Data Accountability Act
    GRI = "gri"  # GRI 305 Emissions / GRI 306 Waste


class ComplianceStatus(str, Enum):
    """Compliance assessment status result."""

    COMPLIANT = "compliant"  # Fully meets framework requirements
    NON_COMPLIANT = "non_compliant"  # Does not meet requirements
    PARTIAL = "partial"  # Partially meets requirements
    NOT_ASSESSED = "not_assessed"  # Not yet evaluated


class PipelineStage(str, Enum):
    """Processing pipeline stages for end-of-life treatment calculations."""

    VALIDATE = "validate"  # Input validation
    CLASSIFY = "classify"  # Product and material classification
    NORMALIZE = "normalize"  # Unit and weight normalization
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution
    CALCULATE = "calculate"  # Core emissions calculation
    ALLOCATE = "allocate"  # Treatment pathway allocation
    AGGREGATE = "aggregate"  # Multi-product aggregation
    COMPLIANCE = "compliance"  # Compliance checking
    PROVENANCE = "provenance"  # Provenance hash chain
    SEAL = "seal"  # Final sealing and output


class ProvenanceStage(str, Enum):
    """Provenance chain stages mirroring the processing pipeline."""

    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE = "calculate"
    ALLOCATE = "allocate"
    AGGREGATE = "aggregate"
    COMPLIANCE = "compliance"
    PROVENANCE = "provenance"
    SEAL = "seal"


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method for emissions estimates."""

    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation (10K+ iterations)
    ANALYTICAL = "analytical"  # Analytical error propagation (Gaussian)
    IPCC_TIER2 = "ipcc_tier2"  # IPCC Tier 2 default uncertainty ranges


class BatchStatus(str, Enum):
    """Batch processing status for multi-product calculations."""

    PENDING = "pending"  # Queued for processing
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Processing failed


class GWPSource(str, Enum):
    """IPCC Assessment Report source for Global Warming Potential values."""

    AR5 = "ar5"  # Fifth Assessment Report (100-year GWP)
    AR6 = "ar6"  # Sixth Assessment Report (100-year GWP)


class EmissionGas(str, Enum):
    """Greenhouse gas types tracked in end-of-life treatment calculations."""

    CO2_FOSSIL = "co2_fossil"  # Fossil-origin CO2 (reported in inventory)
    CO2_BIOGENIC = "co2_biogenic"  # Biogenic-origin CO2 (memo item only)
    CH4 = "ch4"  # Methane (from anaerobic decomposition)
    N2O = "n2o"  # Nitrous oxide (from nitrogen in waste)


class WasteHierarchyLevel(str, Enum):
    """EU Waste Framework Directive waste hierarchy levels (most to least preferred)."""

    PREVENTION = "prevention"  # Waste prevention / source reduction
    REUSE = "reuse"  # Product reuse / refurbishment
    RECYCLING = "recycling"  # Material recycling / recovery
    RECOVERY = "recovery"  # Energy recovery (WtE, AD biogas)
    DISPOSAL = "disposal"  # Landfill, incineration without recovery


class CircularityMetric(str, Enum):
    """Circular economy metrics for end-of-life performance assessment."""

    RECYCLING_RATE = "recycling_rate"  # Mass recycled / total mass
    DIVERSION_RATE = "diversion_rate"  # Mass diverted from landfill / total mass
    CIRCULARITY_INDEX = "circularity_index"  # Ellen MacArthur MCI-derived index
    MATERIAL_RECOVERY_RATE = "material_recovery_rate"  # Material recovered / total mass


# ==============================================================================
# GLOBAL WARMING POTENTIALS
# ==============================================================================

GWP_VALUES: Dict[GWPSource, Dict[str, Decimal]] = {
    GWPSource.AR5: {
        "co2": Decimal("1"),
        "ch4": Decimal("28"),
        "n2o": Decimal("265"),
    },
    GWPSource.AR6: {
        "co2": Decimal("1"),
        "ch4": Decimal("27.9"),
        "n2o": Decimal("273"),
    },
}


# ==============================================================================
# CONSTANT TABLE 1: MATERIAL TREATMENT EMISSION FACTORS
# ==============================================================================
# kgCO2e per kg of material by treatment method.
# Negative values represent avoided emissions (recycling credits, energy recovery).
# Sources: EPA WARM v16, DEFRA 2024, IPCC 2006 Vol 5, Ecoinvent 3.10

MATERIAL_TREATMENT_EFS: Dict[str, Dict[str, Decimal]] = {
    MaterialType.PLASTIC: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("2.760"),
        TreatmentMethod.RECYCLING: Decimal("-1.440"),
        TreatmentMethod.OPEN_BURNING: Decimal("3.100"),
    },
    MaterialType.METAL: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("0.021"),
        TreatmentMethod.RECYCLING: Decimal("-2.500"),
    },
    MaterialType.ALUMINUM: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("0.021"),
        TreatmentMethod.RECYCLING: Decimal("-9.120"),
    },
    MaterialType.STEEL: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("-1.520"),
        TreatmentMethod.RECYCLING: Decimal("-1.820"),
    },
    MaterialType.GLASS: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("0.021"),
        TreatmentMethod.RECYCLING: Decimal("-0.315"),
    },
    MaterialType.PAPER: {
        TreatmentMethod.LANDFILL: Decimal("1.095"),
        TreatmentMethod.INCINERATION: Decimal("0.040"),
        TreatmentMethod.RECYCLING: Decimal("-0.680"),
        TreatmentMethod.COMPOSTING: Decimal("-0.180"),
    },
    MaterialType.CARDBOARD: {
        TreatmentMethod.LANDFILL: Decimal("1.095"),
        TreatmentMethod.INCINERATION: Decimal("0.040"),
        TreatmentMethod.RECYCLING: Decimal("-0.750"),
        TreatmentMethod.COMPOSTING: Decimal("-0.180"),
    },
    MaterialType.WOOD: {
        TreatmentMethod.LANDFILL: Decimal("0.729"),
        TreatmentMethod.INCINERATION: Decimal("-0.410"),
        TreatmentMethod.RECYCLING: Decimal("-2.470"),
        TreatmentMethod.COMPOSTING: Decimal("-0.180"),
    },
    MaterialType.TEXTILE: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("2.760"),
        TreatmentMethod.RECYCLING: Decimal("-2.290"),
    },
    MaterialType.ELECTRONICS: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("0.050"),
        TreatmentMethod.RECYCLING: Decimal("-2.500"),
    },
    MaterialType.ORGANIC: {
        TreatmentMethod.LANDFILL: Decimal("1.824"),
        TreatmentMethod.INCINERATION: Decimal("0.040"),
        TreatmentMethod.COMPOSTING: Decimal("0.116"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.028"),
    },
    MaterialType.RUBBER: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("2.430"),
        TreatmentMethod.RECYCLING: Decimal("-1.200"),
    },
    MaterialType.CERAMIC: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("0.021"),
        TreatmentMethod.RECYCLING: Decimal("-0.100"),
    },
    MaterialType.CONCRETE: {
        TreatmentMethod.LANDFILL: Decimal("0.021"),
        TreatmentMethod.INCINERATION: Decimal("0.021"),
        TreatmentMethod.RECYCLING: Decimal("-0.070"),
    },
    MaterialType.MIXED: {
        TreatmentMethod.LANDFILL: Decimal("0.587"),
        TreatmentMethod.INCINERATION: Decimal("0.445"),
        TreatmentMethod.RECYCLING: Decimal("-0.860"),
        TreatmentMethod.COMPOSTING: Decimal("0.116"),
    },
}


# ==============================================================================
# CONSTANT TABLE 2: PRODUCT MATERIAL COMPOSITIONS
# ==============================================================================
# Default bill-of-materials (BOM) for each product category.
# Fractions sum to 1.0 per product category.
# Sources: Industry averages, WEEE Directive Annex, EPA product composition studies.

PRODUCT_MATERIAL_COMPOSITIONS: Dict[str, Dict[str, Decimal]] = {
    ProductCategory.ELECTRONICS: {
        MaterialType.PLASTIC: Decimal("0.40"),
        MaterialType.METAL: Decimal("0.35"),
        MaterialType.GLASS: Decimal("0.15"),
        MaterialType.MIXED: Decimal("0.10"),
    },
    ProductCategory.APPLIANCES: {
        MaterialType.METAL: Decimal("0.60"),
        MaterialType.PLASTIC: Decimal("0.30"),
        MaterialType.MIXED: Decimal("0.10"),
    },
    ProductCategory.FURNITURE: {
        MaterialType.WOOD: Decimal("0.60"),
        MaterialType.METAL: Decimal("0.20"),
        MaterialType.TEXTILE: Decimal("0.15"),
        MaterialType.MIXED: Decimal("0.05"),
    },
    ProductCategory.PACKAGING: {
        MaterialType.PAPER: Decimal("0.50"),
        MaterialType.PLASTIC: Decimal("0.30"),
        MaterialType.MIXED: Decimal("0.20"),
    },
    ProductCategory.CLOTHING: {
        MaterialType.TEXTILE: Decimal("0.90"),
        MaterialType.MIXED: Decimal("0.10"),
    },
    ProductCategory.AUTOMOTIVE_PARTS: {
        MaterialType.STEEL: Decimal("0.50"),
        MaterialType.ALUMINUM: Decimal("0.15"),
        MaterialType.PLASTIC: Decimal("0.20"),
        MaterialType.RUBBER: Decimal("0.10"),
        MaterialType.MIXED: Decimal("0.05"),
    },
    ProductCategory.BUILDING_MATERIALS: {
        MaterialType.CONCRETE: Decimal("0.40"),
        MaterialType.STEEL: Decimal("0.25"),
        MaterialType.WOOD: Decimal("0.20"),
        MaterialType.GLASS: Decimal("0.10"),
        MaterialType.MIXED: Decimal("0.05"),
    },
    ProductCategory.TOYS: {
        MaterialType.PLASTIC: Decimal("0.60"),
        MaterialType.METAL: Decimal("0.10"),
        MaterialType.TEXTILE: Decimal("0.15"),
        MaterialType.MIXED: Decimal("0.15"),
    },
    ProductCategory.MEDICAL_DEVICES: {
        MaterialType.PLASTIC: Decimal("0.45"),
        MaterialType.METAL: Decimal("0.30"),
        MaterialType.GLASS: Decimal("0.10"),
        MaterialType.MIXED: Decimal("0.15"),
    },
    ProductCategory.BATTERIES: {
        MaterialType.METAL: Decimal("0.55"),
        MaterialType.PLASTIC: Decimal("0.20"),
        MaterialType.MIXED: Decimal("0.25"),
    },
    ProductCategory.TIRES: {
        MaterialType.RUBBER: Decimal("0.70"),
        MaterialType.STEEL: Decimal("0.15"),
        MaterialType.TEXTILE: Decimal("0.10"),
        MaterialType.MIXED: Decimal("0.05"),
    },
    ProductCategory.FOOD_PRODUCTS: {
        MaterialType.ORGANIC: Decimal("0.85"),
        MaterialType.MIXED: Decimal("0.15"),
    },
    ProductCategory.BEVERAGES: {
        MaterialType.ORGANIC: Decimal("0.70"),
        MaterialType.GLASS: Decimal("0.15"),
        MaterialType.PLASTIC: Decimal("0.10"),
        MaterialType.MIXED: Decimal("0.05"),
    },
    ProductCategory.CHEMICALS: {
        MaterialType.PLASTIC: Decimal("0.40"),
        MaterialType.METAL: Decimal("0.20"),
        MaterialType.MIXED: Decimal("0.40"),
    },
    ProductCategory.COSMETICS: {
        MaterialType.PLASTIC: Decimal("0.50"),
        MaterialType.GLASS: Decimal("0.25"),
        MaterialType.MIXED: Decimal("0.25"),
    },
    ProductCategory.OFFICE_SUPPLIES: {
        MaterialType.PAPER: Decimal("0.50"),
        MaterialType.PLASTIC: Decimal("0.25"),
        MaterialType.METAL: Decimal("0.15"),
        MaterialType.MIXED: Decimal("0.10"),
    },
    ProductCategory.SPORTING_GOODS: {
        MaterialType.PLASTIC: Decimal("0.30"),
        MaterialType.METAL: Decimal("0.25"),
        MaterialType.TEXTILE: Decimal("0.20"),
        MaterialType.RUBBER: Decimal("0.15"),
        MaterialType.MIXED: Decimal("0.10"),
    },
    ProductCategory.TOOLS: {
        MaterialType.STEEL: Decimal("0.55"),
        MaterialType.PLASTIC: Decimal("0.25"),
        MaterialType.RUBBER: Decimal("0.10"),
        MaterialType.MIXED: Decimal("0.10"),
    },
    ProductCategory.LIGHTING: {
        MaterialType.GLASS: Decimal("0.35"),
        MaterialType.METAL: Decimal("0.30"),
        MaterialType.PLASTIC: Decimal("0.25"),
        MaterialType.MIXED: Decimal("0.10"),
    },
    ProductCategory.MIXED_PRODUCTS: {
        MaterialType.PLASTIC: Decimal("0.30"),
        MaterialType.METAL: Decimal("0.20"),
        MaterialType.PAPER: Decimal("0.15"),
        MaterialType.GLASS: Decimal("0.10"),
        MaterialType.WOOD: Decimal("0.10"),
        MaterialType.MIXED: Decimal("0.15"),
    },
}


# ==============================================================================
# CONSTANT TABLE 3: REGIONAL TREATMENT MIXES
# ==============================================================================
# Default end-of-life treatment fractions by region, summing to 1.0.
# Sources: World Bank What A Waste 2.0, Eurostat, EPA Facts and Figures,
#          OECD Environment Statistics, national waste statistics.

REGIONAL_TREATMENT_MIXES: Dict[str, Dict[str, Decimal]] = {
    RegionalTreatmentProfile.US: {
        TreatmentMethod.LANDFILL: Decimal("0.50"),
        TreatmentMethod.INCINERATION: Decimal("0.14"),
        TreatmentMethod.RECYCLING: Decimal("0.32"),
        TreatmentMethod.COMPOSTING: Decimal("0.04"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.00"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.00"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.EU: {
        TreatmentMethod.LANDFILL: Decimal("0.24"),
        TreatmentMethod.INCINERATION: Decimal("0.27"),
        TreatmentMethod.RECYCLING: Decimal("0.44"),
        TreatmentMethod.COMPOSTING: Decimal("0.05"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.00"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.00"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.UK: {
        TreatmentMethod.LANDFILL: Decimal("0.28"),
        TreatmentMethod.INCINERATION: Decimal("0.30"),
        TreatmentMethod.RECYCLING: Decimal("0.38"),
        TreatmentMethod.COMPOSTING: Decimal("0.04"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.00"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.00"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.DE: {
        TreatmentMethod.LANDFILL: Decimal("0.01"),
        TreatmentMethod.INCINERATION: Decimal("0.32"),
        TreatmentMethod.RECYCLING: Decimal("0.60"),
        TreatmentMethod.COMPOSTING: Decimal("0.06"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.01"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.00"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.FR: {
        TreatmentMethod.LANDFILL: Decimal("0.22"),
        TreatmentMethod.INCINERATION: Decimal("0.34"),
        TreatmentMethod.RECYCLING: Decimal("0.38"),
        TreatmentMethod.COMPOSTING: Decimal("0.05"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.01"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.00"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.JP: {
        TreatmentMethod.LANDFILL: Decimal("0.05"),
        TreatmentMethod.INCINERATION: Decimal("0.72"),
        TreatmentMethod.RECYCLING: Decimal("0.20"),
        TreatmentMethod.COMPOSTING: Decimal("0.03"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.00"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.00"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.CN: {
        TreatmentMethod.LANDFILL: Decimal("0.45"),
        TreatmentMethod.INCINERATION: Decimal("0.40"),
        TreatmentMethod.RECYCLING: Decimal("0.10"),
        TreatmentMethod.COMPOSTING: Decimal("0.02"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.00"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.03"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.IN: {
        TreatmentMethod.LANDFILL: Decimal("0.60"),
        TreatmentMethod.INCINERATION: Decimal("0.05"),
        TreatmentMethod.RECYCLING: Decimal("0.10"),
        TreatmentMethod.COMPOSTING: Decimal("0.05"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.00"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.20"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.BR: {
        TreatmentMethod.LANDFILL: Decimal("0.58"),
        TreatmentMethod.INCINERATION: Decimal("0.02"),
        TreatmentMethod.RECYCLING: Decimal("0.13"),
        TreatmentMethod.COMPOSTING: Decimal("0.02"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.00"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.25"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.AU: {
        TreatmentMethod.LANDFILL: Decimal("0.42"),
        TreatmentMethod.INCINERATION: Decimal("0.05"),
        TreatmentMethod.RECYCLING: Decimal("0.45"),
        TreatmentMethod.COMPOSTING: Decimal("0.07"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.01"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.00"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.KR: {
        TreatmentMethod.LANDFILL: Decimal("0.10"),
        TreatmentMethod.INCINERATION: Decimal("0.25"),
        TreatmentMethod.RECYCLING: Decimal("0.60"),
        TreatmentMethod.COMPOSTING: Decimal("0.04"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.01"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.00"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
    RegionalTreatmentProfile.GLOBAL: {
        TreatmentMethod.LANDFILL: Decimal("0.40"),
        TreatmentMethod.INCINERATION: Decimal("0.16"),
        TreatmentMethod.RECYCLING: Decimal("0.20"),
        TreatmentMethod.COMPOSTING: Decimal("0.05"),
        TreatmentMethod.ANAEROBIC_DIGESTION: Decimal("0.01"),
        TreatmentMethod.OPEN_BURNING: Decimal("0.18"),
        TreatmentMethod.WASTEWATER: Decimal("0.00"),
    },
}


# ==============================================================================
# CONSTANT TABLE 4: LANDFILL FOD PARAMETERS
# ==============================================================================
# First Order Decay parameters by material and climate zone.
# DOC = Degradable Organic Carbon fraction, DOCf = fraction of DOC that decomposes,
# MCF = Methane Correction Factor, k = decay rate (yr-1),
# F = CH4 fraction in landfill gas, OX = oxidation factor.
# Source: IPCC 2006 Vol 5 Tables 2.4, 2.5, 3.1, 3.3

LANDFILL_FOD_PARAMETERS: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    MaterialType.PAPER: {
        ClimateZone.BOREAL_TEMPERATE_DRY: {
            "DOC": Decimal("0.400"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.040"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.BOREAL_TEMPERATE_WET: {
            "DOC": Decimal("0.400"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.060"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_DRY: {
            "DOC": Decimal("0.400"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.045"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_WET: {
            "DOC": Decimal("0.400"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.070"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
    },
    MaterialType.CARDBOARD: {
        ClimateZone.BOREAL_TEMPERATE_DRY: {
            "DOC": Decimal("0.440"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.040"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.BOREAL_TEMPERATE_WET: {
            "DOC": Decimal("0.440"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.060"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_DRY: {
            "DOC": Decimal("0.440"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.045"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_WET: {
            "DOC": Decimal("0.440"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.070"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
    },
    MaterialType.WOOD: {
        ClimateZone.BOREAL_TEMPERATE_DRY: {
            "DOC": Decimal("0.430"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.020"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.BOREAL_TEMPERATE_WET: {
            "DOC": Decimal("0.430"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.030"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_DRY: {
            "DOC": Decimal("0.430"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.025"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_WET: {
            "DOC": Decimal("0.430"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.035"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
    },
    MaterialType.TEXTILE: {
        ClimateZone.BOREAL_TEMPERATE_DRY: {
            "DOC": Decimal("0.240"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.040"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.BOREAL_TEMPERATE_WET: {
            "DOC": Decimal("0.240"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.060"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_DRY: {
            "DOC": Decimal("0.240"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.045"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_WET: {
            "DOC": Decimal("0.240"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.070"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
    },
    MaterialType.ORGANIC: {
        ClimateZone.BOREAL_TEMPERATE_DRY: {
            "DOC": Decimal("0.150"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.060"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.BOREAL_TEMPERATE_WET: {
            "DOC": Decimal("0.150"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.185"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_DRY: {
            "DOC": Decimal("0.150"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.085"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
        ClimateZone.TROPICAL_WET: {
            "DOC": Decimal("0.150"), "DOCf": Decimal("0.500"), "MCF": Decimal("1.0"),
            "k": Decimal("0.400"), "F": Decimal("0.500"), "OX": Decimal("0.10"),
        },
    },
}


# ==============================================================================
# CONSTANT TABLE 5: INCINERATION PARAMETERS
# ==============================================================================
# Parameters for incineration CO2 calculation per IPCC 2006 Vol 5 Table 5.2.
# dry_matter_fraction, carbon_fraction, fossil_carbon_fraction, oxidation_factor.

INCINERATION_PARAMETERS: Dict[str, Dict[str, Decimal]] = {
    MaterialType.PLASTIC: {
        "dry_matter_fraction": Decimal("1.00"),
        "carbon_fraction": Decimal("0.75"),
        "fossil_carbon_fraction": Decimal("1.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    MaterialType.PAPER: {
        "dry_matter_fraction": Decimal("0.90"),
        "carbon_fraction": Decimal("0.46"),
        "fossil_carbon_fraction": Decimal("0.01"),
        "oxidation_factor": Decimal("1.00"),
    },
    MaterialType.CARDBOARD: {
        "dry_matter_fraction": Decimal("0.90"),
        "carbon_fraction": Decimal("0.46"),
        "fossil_carbon_fraction": Decimal("0.01"),
        "oxidation_factor": Decimal("1.00"),
    },
    MaterialType.WOOD: {
        "dry_matter_fraction": Decimal("0.85"),
        "carbon_fraction": Decimal("0.50"),
        "fossil_carbon_fraction": Decimal("0.05"),
        "oxidation_factor": Decimal("1.00"),
    },
    MaterialType.TEXTILE: {
        "dry_matter_fraction": Decimal("0.80"),
        "carbon_fraction": Decimal("0.46"),
        "fossil_carbon_fraction": Decimal("0.50"),
        "oxidation_factor": Decimal("1.00"),
    },
    MaterialType.ORGANIC: {
        "dry_matter_fraction": Decimal("0.40"),
        "carbon_fraction": Decimal("0.38"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    MaterialType.RUBBER: {
        "dry_matter_fraction": Decimal("0.84"),
        "carbon_fraction": Decimal("0.67"),
        "fossil_carbon_fraction": Decimal("0.50"),
        "oxidation_factor": Decimal("1.00"),
    },
    MaterialType.ELECTRONICS: {
        "dry_matter_fraction": Decimal("0.90"),
        "carbon_fraction": Decimal("0.05"),
        "fossil_carbon_fraction": Decimal("1.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    MaterialType.MIXED: {
        "dry_matter_fraction": Decimal("0.69"),
        "carbon_fraction": Decimal("0.33"),
        "fossil_carbon_fraction": Decimal("0.40"),
        "oxidation_factor": Decimal("1.00"),
    },
}


# ==============================================================================
# CONSTANT TABLE 6: GAS COLLECTION EFFICIENCY
# ==============================================================================
# Landfill gas collection efficiency by landfill type (0.0 to 0.75).
# Source: EPA AP-42 Ch 2.4, IPCC 2006 Vol 5 Ch 3.

GAS_COLLECTION_EFFICIENCY: Dict[str, Decimal] = {
    LandfillType.MANAGED_ANAEROBIC: Decimal("0.20"),
    LandfillType.MANAGED_SEMI_AEROBIC: Decimal("0.10"),
    LandfillType.UNMANAGED_DEEP: Decimal("0.00"),
    LandfillType.UNMANAGED_SHALLOW: Decimal("0.00"),
    LandfillType.ENGINEERED_WITH_GAS: Decimal("0.75"),
    LandfillType.ENGINEERED_WITHOUT_GAS: Decimal("0.00"),
}


# ==============================================================================
# CONSTANT TABLE 7: ENERGY RECOVERY FACTORS
# ==============================================================================
# Waste-to-energy (WtE) efficiency and displaced grid emission factor by region.
# wte_efficiency = thermal-to-electrical conversion efficiency (fraction).
# displaced_grid_ef = kgCO2e per kWh of displaced grid electricity.
# Source: IEA Electricity Maps, national grid averages.

ENERGY_RECOVERY_FACTORS: Dict[str, Dict[str, Decimal]] = {
    RegionalTreatmentProfile.US: {
        "wte_efficiency": Decimal("0.22"),
        "displaced_grid_ef": Decimal("0.386"),
    },
    RegionalTreatmentProfile.EU: {
        "wte_efficiency": Decimal("0.25"),
        "displaced_grid_ef": Decimal("0.276"),
    },
    RegionalTreatmentProfile.UK: {
        "wte_efficiency": Decimal("0.24"),
        "displaced_grid_ef": Decimal("0.207"),
    },
    RegionalTreatmentProfile.DE: {
        "wte_efficiency": Decimal("0.26"),
        "displaced_grid_ef": Decimal("0.338"),
    },
    RegionalTreatmentProfile.FR: {
        "wte_efficiency": Decimal("0.25"),
        "displaced_grid_ef": Decimal("0.052"),
    },
    RegionalTreatmentProfile.JP: {
        "wte_efficiency": Decimal("0.20"),
        "displaced_grid_ef": Decimal("0.462"),
    },
    RegionalTreatmentProfile.CN: {
        "wte_efficiency": Decimal("0.18"),
        "displaced_grid_ef": Decimal("0.555"),
    },
    RegionalTreatmentProfile.IN: {
        "wte_efficiency": Decimal("0.15"),
        "displaced_grid_ef": Decimal("0.708"),
    },
    RegionalTreatmentProfile.BR: {
        "wte_efficiency": Decimal("0.16"),
        "displaced_grid_ef": Decimal("0.074"),
    },
    RegionalTreatmentProfile.AU: {
        "wte_efficiency": Decimal("0.22"),
        "displaced_grid_ef": Decimal("0.656"),
    },
    RegionalTreatmentProfile.KR: {
        "wte_efficiency": Decimal("0.22"),
        "displaced_grid_ef": Decimal("0.415"),
    },
    RegionalTreatmentProfile.GLOBAL: {
        "wte_efficiency": Decimal("0.20"),
        "displaced_grid_ef": Decimal("0.442"),
    },
}


# ==============================================================================
# CONSTANT TABLE 8: RECYCLING PROCESSING EMISSION FACTORS
# ==============================================================================
# Transport to MRF + MRF processing emissions per kg of material (kgCO2e/kg).
# These are the positive emissions incurred during the recycling process itself.
# Source: Ecoinvent 3.10, EPA WARM v16.

RECYCLING_PROCESSING_EFS: Dict[str, Decimal] = {
    MaterialType.PLASTIC: Decimal("0.180"),
    MaterialType.METAL: Decimal("0.120"),
    MaterialType.ALUMINUM: Decimal("0.150"),
    MaterialType.STEEL: Decimal("0.110"),
    MaterialType.GLASS: Decimal("0.085"),
    MaterialType.PAPER: Decimal("0.095"),
    MaterialType.CARDBOARD: Decimal("0.090"),
    MaterialType.WOOD: Decimal("0.070"),
    MaterialType.TEXTILE: Decimal("0.200"),
    MaterialType.ELECTRONICS: Decimal("0.350"),
    MaterialType.ORGANIC: Decimal("0.050"),
    MaterialType.RUBBER: Decimal("0.160"),
    MaterialType.CERAMIC: Decimal("0.060"),
    MaterialType.CONCRETE: Decimal("0.040"),
    MaterialType.MIXED: Decimal("0.130"),
}


# ==============================================================================
# CONSTANT TABLE 9: AVOIDED EMISSION FACTORS
# ==============================================================================
# Virgin material substitution credit (negative kgCO2e/kg) for recycling.
# Represents emissions avoided by displacing virgin material production.
# These are memo items, NOT deducted from Category 12 totals per GHG Protocol.
# Source: Ecoinvent 3.10, EPA WARM v16.

AVOIDED_EMISSION_FACTORS: Dict[str, Decimal] = {
    MaterialType.PLASTIC: Decimal("-1.440"),
    MaterialType.METAL: Decimal("-2.500"),
    MaterialType.ALUMINUM: Decimal("-9.120"),
    MaterialType.STEEL: Decimal("-1.820"),
    MaterialType.GLASS: Decimal("-0.315"),
    MaterialType.PAPER: Decimal("-0.680"),
    MaterialType.CARDBOARD: Decimal("-0.750"),
    MaterialType.WOOD: Decimal("-2.470"),
    MaterialType.TEXTILE: Decimal("-2.290"),
    MaterialType.ELECTRONICS: Decimal("-2.500"),
    MaterialType.ORGANIC: Decimal("-0.180"),
    MaterialType.RUBBER: Decimal("-1.200"),
    MaterialType.CERAMIC: Decimal("-0.100"),
    MaterialType.CONCRETE: Decimal("-0.070"),
    MaterialType.MIXED: Decimal("-0.860"),
}


# ==============================================================================
# CONSTANT TABLE 10: PRODUCT WEIGHT DEFAULTS
# ==============================================================================
# Average weight per unit (kg) for each product category.
# Used as fallback when product-specific weight is not provided.
# Source: Industry averages, WEEE Directive Annex III, EPA product studies.

PRODUCT_WEIGHT_DEFAULTS: Dict[str, Decimal] = {
    ProductCategory.ELECTRONICS: Decimal("2.5"),
    ProductCategory.APPLIANCES: Decimal("35.0"),
    ProductCategory.FURNITURE: Decimal("25.0"),
    ProductCategory.PACKAGING: Decimal("0.5"),
    ProductCategory.CLOTHING: Decimal("0.8"),
    ProductCategory.AUTOMOTIVE_PARTS: Decimal("15.0"),
    ProductCategory.BUILDING_MATERIALS: Decimal("50.0"),
    ProductCategory.TOYS: Decimal("1.5"),
    ProductCategory.MEDICAL_DEVICES: Decimal("3.0"),
    ProductCategory.BATTERIES: Decimal("0.5"),
    ProductCategory.TIRES: Decimal("10.0"),
    ProductCategory.FOOD_PRODUCTS: Decimal("1.0"),
    ProductCategory.BEVERAGES: Decimal("1.2"),
    ProductCategory.CHEMICALS: Decimal("5.0"),
    ProductCategory.COSMETICS: Decimal("0.3"),
    ProductCategory.OFFICE_SUPPLIES: Decimal("0.4"),
    ProductCategory.SPORTING_GOODS: Decimal("3.0"),
    ProductCategory.TOOLS: Decimal("2.0"),
    ProductCategory.LIGHTING: Decimal("0.5"),
    ProductCategory.MIXED_PRODUCTS: Decimal("2.0"),
}


# ==============================================================================
# CONSTANT TABLE 11: COMPOSTING EMISSION FACTORS
# ==============================================================================
# CH4 and N2O emission factors for composting (g gas per kg wet organic waste).
# Source: IPCC 2006 Vol 5 Table 4.1.

COMPOSTING_EMISSION_FACTORS: Dict[str, Decimal] = {
    "ch4_ef_industrial": Decimal("4.0"),  # g CH4 per kg wet waste (industrial)
    "n2o_ef_industrial": Decimal("0.30"),  # g N2O per kg wet waste (industrial)
    "ch4_ef_home": Decimal("10.0"),  # g CH4 per kg wet waste (home composting)
    "n2o_ef_home": Decimal("0.60"),  # g N2O per kg wet waste (home composting)
}


# ==============================================================================
# CONSTANT TABLE 12: ANAEROBIC DIGESTION EMISSION FACTORS
# ==============================================================================
# Fugitive CH4 emissions per m3 biogas produced and capture efficiency by AD type.
# Source: IPCC 2019 Refinement, industry averages.

AD_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "enclosed_modern": {
        "fugitive_ch4_per_m3_biogas": Decimal("0.012"),  # kg CH4/m3 biogas
        "capture_efficiency": Decimal("0.98"),
    },
    "enclosed_standard": {
        "fugitive_ch4_per_m3_biogas": Decimal("0.019"),
        "capture_efficiency": Decimal("0.95"),
    },
    "open_lagoon": {
        "fugitive_ch4_per_m3_biogas": Decimal("0.045"),
        "capture_efficiency": Decimal("0.70"),
    },
    "covered_lagoon": {
        "fugitive_ch4_per_m3_biogas": Decimal("0.028"),
        "capture_efficiency": Decimal("0.85"),
    },
}


# ==============================================================================
# CONSTANT TABLE 13: DOUBLE-COUNTING PREVENTION RULES
# ==============================================================================
# Rules to prevent double-counting between Scope 3 Category 12 and other categories.
# Source: GHG Protocol Scope 3 Technical Guidance, Chapter 12.

DC_RULES: List[Dict[str, str]] = [
    {
        "id": "DC-EOL-001",
        "description": "Do not double-count with Category 5 (Waste Generated in Operations). "
                       "Cat 5 covers waste from the company's OWN operations. Cat 12 covers "
                       "end-of-life treatment of SOLD PRODUCTS by downstream consumers.",
        "scope_boundary": "cat5_vs_cat12",
    },
    {
        "id": "DC-EOL-002",
        "description": "Do not double-count with Category 1 (Purchased Goods and Services). "
                       "If a product is purchased and then disposed of within the same reporting "
                       "period, emissions should be reported in Cat 5, not Cat 12.",
        "scope_boundary": "cat1_vs_cat12",
    },
    {
        "id": "DC-EOL-003",
        "description": "Do not double-count with Scope 1 direct emissions. If the reporting "
                       "company operates its own waste treatment facilities for sold products, "
                       "those emissions belong in Scope 1, not Cat 12.",
        "scope_boundary": "scope1_vs_cat12",
    },
    {
        "id": "DC-EOL-004",
        "description": "Recycling avoided emissions are memo items only. Do NOT deduct avoided "
                       "emissions from Category 12 totals. Report them separately as informational.",
        "scope_boundary": "avoided_emissions_memo",
    },
    {
        "id": "DC-EOL-005",
        "description": "Do not double-count biogenic CO2 from incineration of biogenic materials. "
                       "Report biogenic CO2 as a memo item, not in the GHG inventory total.",
        "scope_boundary": "biogenic_co2_memo",
    },
    {
        "id": "DC-EOL-006",
        "description": "Do not double-count with Category 11 (Use of Sold Products). "
                       "Cat 11 covers emissions during the use phase; Cat 12 covers emissions "
                       "at the end of the product's useful life.",
        "scope_boundary": "cat11_vs_cat12",
    },
    {
        "id": "DC-EOL-007",
        "description": "Energy recovery credits from waste-to-energy are memo items only. "
                       "Do NOT deduct WtE displaced grid emissions from Category 12 totals.",
        "scope_boundary": "energy_recovery_memo",
    },
    {
        "id": "DC-EOL-008",
        "description": "Do not double-count with Category 10 (Processing of Sold Products). "
                       "Cat 10 covers intermediate processing; Cat 12 covers final disposal "
                       "at end of life.",
        "scope_boundary": "cat10_vs_cat12",
    },
]


# ==============================================================================
# CONSTANT TABLE 14: COMPLIANCE FRAMEWORK RULES
# ==============================================================================
# Required disclosures and rules per compliance framework for Category 12.
# Source: GHG Protocol, ISO 14064, CSRD ESRS E1/E5, CDP, SBTi, SB 253, GRI 305/306.

COMPLIANCE_FRAMEWORK_RULES: Dict[str, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_category12_emissions_tco2e",
        "calculation_methodology_documented",
        "products_included_with_justification",
        "exclusions_with_justification_and_percentage",
        "data_quality_assessment_per_product",
        "treatment_scenario_assumptions_documented",
        "regional_treatment_mix_source_cited",
        "avoided_emissions_reported_separately",
    ],
    ComplianceFramework.ISO_14064: [
        "activity_data_by_product_category",
        "emission_factors_with_source_and_year",
        "uncertainty_quantification_provided",
        "system_boundary_description",
        "base_year_emissions_for_recalculation",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "scope3_category12_absolute_emissions",
        "e1_6_gross_scope3_ghg_emissions",
        "e5_5_waste_by_type_and_treatment",
        "circular_economy_metrics",
        "product_eol_reduction_targets",
        "waste_hierarchy_alignment",
    ],
    ComplianceFramework.CDP: [
        "scope3_cat12_total_mtco2e",
        "percentage_of_total_scope3",
        "products_by_disposal_method",
        "eol_reduction_initiatives",
        "year_over_year_change_explanation",
    ],
    ComplianceFramework.SBTI: [
        "scope3_cat12_baseline_year_emissions",
        "scope3_cat12_target_year_emissions",
        "reduction_trajectory_documented",
        "product_design_for_recyclability",
    ],
    ComplianceFramework.SB_253: [
        "scope3_cat12_total_reported",
        "methodology_and_ef_sources",
        "third_party_assurance_status",
        "data_gaps_and_estimation_methods",
    ],
    ComplianceFramework.GRI: [
        "gri_305_scope3_cat12_emissions",
        "gri_306_waste_by_composition",
        "gri_306_waste_diverted_from_disposal",
        "gri_306_waste_directed_to_disposal",
        "reduction_actions_and_results",
    ],
}


# ==============================================================================
# CONSTANT TABLE 15: DQI SCORING
# ==============================================================================
# Data Quality Indicator scoring matrix: 5 dimensions x 3 tiers.
# Score is on a 1-5 scale (1 = best, 5 = worst).
# Source: GHG Protocol Corporate Value Chain Guidance, Table 7.1.

DQI_SCORING: Dict[str, Dict[str, Decimal]] = {
    DQIDimension.TEMPORAL: {
        DataQualityTier.TIER_1: Decimal("4.0"),  # Default global factor, not time-specific
        DataQualityTier.TIER_2: Decimal("2.5"),  # Regional factor within 5 years
        DataQualityTier.TIER_3: Decimal("1.0"),  # Product-specific, current year
    },
    DQIDimension.GEOGRAPHICAL: {
        DataQualityTier.TIER_1: Decimal("4.0"),  # Global average
        DataQualityTier.TIER_2: Decimal("2.0"),  # Country-specific
        DataQualityTier.TIER_3: Decimal("1.0"),  # Site/product-specific
    },
    DQIDimension.TECHNOLOGICAL: {
        DataQualityTier.TIER_1: Decimal("4.0"),  # Generic material category
        DataQualityTier.TIER_2: Decimal("2.5"),  # Material-specific, similar treatment
        DataQualityTier.TIER_3: Decimal("1.0"),  # Exact material and treatment process
    },
    DQIDimension.COMPLETENESS: {
        DataQualityTier.TIER_1: Decimal("3.5"),  # < 50% of product portfolio
        DataQualityTier.TIER_2: Decimal("2.0"),  # 50-90% of product portfolio
        DataQualityTier.TIER_3: Decimal("1.0"),  # > 90% of product portfolio
    },
    DQIDimension.RELIABILITY: {
        DataQualityTier.TIER_1: Decimal("4.0"),  # Published database, unverified
        DataQualityTier.TIER_2: Decimal("2.5"),  # Published, peer-reviewed
        DataQualityTier.TIER_3: Decimal("1.0"),  # Verified EPD or measured data
    },
}


# ==============================================================================
# CONSTANT TABLE 16: UNCERTAINTY RANGES
# ==============================================================================
# Uncertainty ranges (plus/minus fraction at 95% confidence) by calculation method
# and data quality tier.
# Source: IPCC 2006 Vol 1 Ch 3 Uncertainties, GHG Protocol guidance.

UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    CalculationMethod.WASTE_TYPE_SPECIFIC: {
        DataQualityTier.TIER_1: Decimal("0.35"),  # +/- 35%
        DataQualityTier.TIER_2: Decimal("0.20"),  # +/- 20%
        DataQualityTier.TIER_3: Decimal("0.10"),  # +/- 10%
    },
    CalculationMethod.AVERAGE_DATA: {
        DataQualityTier.TIER_1: Decimal("0.60"),  # +/- 60%
        DataQualityTier.TIER_2: Decimal("0.40"),  # +/- 40%
        DataQualityTier.TIER_3: Decimal("0.25"),  # +/- 25%
    },
    CalculationMethod.SPEND_BASED: {
        DataQualityTier.TIER_1: Decimal("0.80"),  # +/- 80%
        DataQualityTier.TIER_2: Decimal("0.60"),  # +/- 60%
        DataQualityTier.TIER_3: Decimal("0.40"),  # +/- 40%
    },
    CalculationMethod.PRODUCER_SPECIFIC: {
        DataQualityTier.TIER_1: Decimal("0.15"),  # +/- 15%
        DataQualityTier.TIER_2: Decimal("0.10"),  # +/- 10%
        DataQualityTier.TIER_3: Decimal("0.05"),  # +/- 5%
    },
    CalculationMethod.HYBRID: {
        DataQualityTier.TIER_1: Decimal("0.45"),  # +/- 45%
        DataQualityTier.TIER_2: Decimal("0.30"),  # +/- 30%
        DataQualityTier.TIER_3: Decimal("0.15"),  # +/- 15%
    },
}


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================


class ProductEOLInput(GreenLangBase):
    """
    Primary input for end-of-life treatment emissions calculation of a sold product.

    Represents a single product or product line with its material composition,
    treatment scenario, and regional context for Category 12 emissions calculation.

    Example:
        >>> product = ProductEOLInput(
        ...     product_id="PROD-001",
        ...     org_id="ORG-001",
        ...     category=ProductCategory.ELECTRONICS,
        ...     units_sold=10000,
        ...     weight_per_unit_kg=Decimal("2.5"),
        ...     region=RegionalTreatmentProfile.US,
        ...     reporting_year=2025
        ... )
    """

    product_id: str = Field(..., description="Unique product or product-line identifier")
    org_id: str = Field(..., description="Organization / tenant identifier")
    category: ProductCategory = Field(..., description="Product category for default BOM lookup")
    units_sold: int = Field(..., gt=0, description="Number of units sold in reporting year")
    weight_per_unit_kg: Optional[Decimal] = Field(
        None, gt=0, description="Weight per unit in kg (uses default if not provided)"
    )
    material_composition: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Custom material composition {MaterialType: fraction}. "
                    "Fractions must sum to 1.0. Overrides default BOM if provided."
    )
    treatment_scenario: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Custom treatment scenario {TreatmentMethod: fraction}. "
                    "Fractions must sum to 1.0. Overrides regional default if provided."
    )
    region: RegionalTreatmentProfile = Field(
        default=RegionalTreatmentProfile.GLOBAL,
        description="Region for default treatment mix lookup"
    )
    lifetime_years: Optional[int] = Field(
        None, ge=0, le=100,
        description="Expected product lifetime in years (informational)"
    )
    reporting_year: int = Field(..., ge=2000, le=2100, description="GHG reporting year")
    reporting_period: Optional[str] = Field(None, description="e.g., '2025-Q1', '2025-H1'")
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.WASTE_TYPE_SPECIFIC,
        description="Calculation method to use"
    )
    notes: Optional[str] = Field(None, description="Free-text notes for audit trail")

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_composition_sum(self) -> "ProductEOLInput":
        """Ensure material composition fractions sum to 1.0 if provided."""
        if self.material_composition is not None:
            total = sum(self.material_composition.values())
            if abs(total - Decimal("1.0")) > Decimal("0.01"):
                raise ValueError(
                    f"material_composition fractions must sum to 1.0, got {total}"
                )
        return self

    @model_validator(mode="after")
    def validate_treatment_sum(self) -> "ProductEOLInput":
        """Ensure treatment scenario fractions sum to 1.0 if provided."""
        if self.treatment_scenario is not None:
            total = sum(self.treatment_scenario.values())
            if abs(total - Decimal("1.0")) > Decimal("0.01"):
                raise ValueError(
                    f"treatment_scenario fractions must sum to 1.0, got {total}"
                )
        return self


class MaterialComposition(GreenLangBase):
    """
    Material composition entry for a product's bill-of-materials.

    Represents a single material within a product, with its type,
    fractional contribution, and absolute weight.
    """

    material_type: MaterialType = Field(..., description="Material type")
    fraction: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Fraction of total product weight (0-1)"
    )
    weight_kg: Decimal = Field(..., ge=Decimal("0"), description="Absolute weight in kg")

    model_config = ConfigDict(frozen=True)


class TreatmentScenario(GreenLangBase):
    """
    Treatment scenario for a specific end-of-life pathway.

    Defines the treatment method, its fractional allocation, and any
    treatment-specific parameters (e.g., landfill type, gas collection).
    """

    treatment_method: TreatmentMethod = Field(..., description="Treatment method")
    fraction: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Fraction of waste going to this treatment (0-1)"
    )
    landfill_type: Optional[LandfillType] = Field(
        None, description="Landfill type (required if treatment is landfill)"
    )
    climate_zone: Optional[ClimateZone] = Field(
        None, description="IPCC climate zone (for landfill decay rate)"
    )
    gas_collection: Optional[bool] = Field(
        None, description="Whether landfill has gas collection system"
    )
    energy_recovery: Optional[bool] = Field(
        None, description="Whether treatment includes energy recovery"
    )
    incineration_type: Optional[IncinerationType] = Field(
        None, description="Incineration technology type"
    )
    recycling_approach: Optional[RecyclingApproach] = Field(
        None, description="Recycling accounting approach"
    )

    model_config = ConfigDict(frozen=True)


class WasteTypeResult(GreenLangBase):
    """
    Waste-type-specific calculation result for a single material-treatment pair.

    Contains the emissions calculation for one material type sent to one
    treatment method, including the emission factor used and gas breakdown.
    """

    material: MaterialType = Field(..., description="Material type")
    treatment: TreatmentMethod = Field(..., description="Treatment method applied")
    weight_kg: Decimal = Field(..., ge=Decimal("0"), description="Weight processed in kg")
    emissions_kgco2e: Decimal = Field(..., description="Total emissions in kgCO2e")
    ef_used: Decimal = Field(..., description="Emission factor used (kgCO2e/kg)")
    ef_source: EFSource = Field(..., description="Source of emission factor")
    gas_breakdown: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by gas {co2_fossil, co2_biogenic, ch4, n2o} in kgCO2e"
    )

    model_config = ConfigDict(frozen=True)


class AverageDataResult(GreenLangBase):
    """
    Average-data calculation result for a product category.

    Uses product-category-level average emission factors rather than
    material-specific waste-type calculations.
    """

    product_category: ProductCategory = Field(..., description="Product category")
    units: int = Field(..., gt=0, description="Number of units")
    weight_kg: Decimal = Field(..., ge=Decimal("0"), description="Total weight in kg")
    emissions_kgco2e: Decimal = Field(..., ge=Decimal("0"), description="Total emissions kgCO2e")
    composite_ef: Decimal = Field(..., description="Composite EF used (kgCO2e/kg)")

    model_config = ConfigDict(frozen=True)


class ProducerSpecificResult(GreenLangBase):
    """
    Producer-specific (EPD) calculation result.

    Uses Environmental Product Declaration end-of-life module data
    for product-specific emissions calculation.
    """

    product_id: str = Field(..., description="Product identifier")
    epd_reference: str = Field(..., description="EPD document reference number")
    eol_ef: Decimal = Field(..., description="EPD end-of-life emission factor (kgCO2e/unit)")
    verification_status: str = Field(
        ..., description="EPD verification status: 'verified', 'self_declared', 'expired'"
    )
    emissions_kgco2e: Decimal = Field(..., ge=Decimal("0"), description="Total emissions kgCO2e")

    model_config = ConfigDict(frozen=True)


class AvoidedEmissions(GreenLangBase):
    """
    Avoided emissions summary (memo items only, NOT deducted from totals).

    Per GHG Protocol, avoided emissions from recycling and energy recovery
    are reported separately as informational items.
    """

    recycling_credit_tco2e: Decimal = Field(
        ..., description="Avoided emissions from recycling (tCO2e, negative)"
    )
    energy_recovery_tco2e: Decimal = Field(
        ..., description="Avoided emissions from energy recovery (tCO2e, negative)"
    )
    total_avoided_tco2e: Decimal = Field(
        ..., description="Total avoided emissions (tCO2e, negative)"
    )

    model_config = ConfigDict(frozen=True)


class CircularityScore(GreenLangBase):
    """
    Circularity metrics for the product portfolio's end-of-life performance.

    Evaluates how well the product portfolio aligns with circular economy
    principles and the EU Waste Framework Directive waste hierarchy.
    """

    recycling_rate: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Mass recycled / total mass (0-1)"
    )
    diversion_rate: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Mass diverted from landfill / total mass (0-1)"
    )
    circularity_index: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Weighted circularity index (0-1, higher is better)"
    )
    waste_hierarchy_compliance: str = Field(
        ..., description="Waste hierarchy alignment level: 'excellent', 'good', 'fair', 'poor'"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityScore(GreenLangBase):
    """
    Data Quality Indicator (DQI) assessment with 5-dimension scoring.

    Each dimension scored on a 1-5 scale (1 = best, 5 = worst).
    Overall score is the arithmetic mean of the five dimensions.
    """

    temporal: Decimal = Field(
        ..., ge=Decimal("1"), le=Decimal("5"),
        description="Temporal correlation score (1-5)"
    )
    geographical: Decimal = Field(
        ..., ge=Decimal("1"), le=Decimal("5"),
        description="Geographical correlation score (1-5)"
    )
    technological: Decimal = Field(
        ..., ge=Decimal("1"), le=Decimal("5"),
        description="Technological correlation score (1-5)"
    )
    completeness: Decimal = Field(
        ..., ge=Decimal("1"), le=Decimal("5"),
        description="Data completeness score (1-5)"
    )
    reliability: Decimal = Field(
        ..., ge=Decimal("1"), le=Decimal("5"),
        description="Source reliability score (1-5)"
    )
    overall: Decimal = Field(
        ..., ge=Decimal("1"), le=Decimal("5"),
        description="Overall DQI score (arithmetic mean, 1-5)"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(GreenLangBase):
    """
    Uncertainty analysis result for emissions estimates.

    Provides statistical measures of uncertainty including confidence
    intervals and the quantification method used.
    """

    method: UncertaintyMethod = Field(..., description="Uncertainty quantification method")
    mean: Decimal = Field(..., ge=Decimal("0"), description="Mean emissions (tCO2e)")
    std_dev: Decimal = Field(..., ge=Decimal("0"), description="Standard deviation (tCO2e)")
    ci_lower: Decimal = Field(..., ge=Decimal("0"), description="Lower bound of 95% CI (tCO2e)")
    ci_upper: Decimal = Field(..., ge=Decimal("0"), description="Upper bound of 95% CI (tCO2e)")
    confidence_level: Decimal = Field(
        default=Decimal("0.95"), ge=Decimal("0"), le=Decimal("1"),
        description="Confidence level (0-1)"
    )
    relative_uncertainty_pct: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Relative uncertainty as +/- percentage"
    )

    model_config = ConfigDict(frozen=True)


class CalculationResult(GreenLangBase):
    """
    Complete calculation result for end-of-life treatment of sold products.

    This is the primary output model aggregating all product-level results
    into an organization-wide Category 12 emissions total.
    """

    calc_id: str = Field(..., description="Unique calculation identifier")
    org_id: str = Field(..., description="Organization / tenant identifier")
    year: int = Field(..., ge=2000, le=2100, description="Reporting year")

    # Totals
    total_tco2e: Decimal = Field(..., ge=Decimal("0"), description="Total Category 12 emissions (tCO2e)")

    # Breakdowns
    by_treatment: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by treatment method {treatment: tCO2e}"
    )
    by_material: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by material type {material: tCO2e}"
    )
    by_product: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by product {product_id: tCO2e}"
    )

    # Avoided emissions (memo item)
    avoided_emissions: Optional[AvoidedEmissions] = Field(
        None, description="Avoided emissions summary (memo item only)"
    )

    # Data quality
    dqi: Optional[DataQualityScore] = Field(None, description="Data quality assessment")

    # Uncertainty
    uncertainty: Optional[UncertaintyResult] = Field(
        None, description="Uncertainty quantification"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for complete audit trail")

    # Timing
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(None, ge=0, description="Processing duration in ms")

    model_config = ConfigDict(frozen=True)


class AggregationResult(GreenLangBase):
    """
    Aggregated emissions result across multiple products, periods, or regions.

    Provides multi-dimensional roll-up of Category 12 emissions for
    reporting dashboards and compliance disclosures.
    """

    period: str = Field(..., description="Aggregation period (e.g., '2025', '2025-Q1')")
    total_tco2e: Decimal = Field(..., ge=Decimal("0"), description="Total aggregated emissions")
    by_treatment: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by treatment method"
    )
    by_material: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by material type"
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by product category"
    )
    by_region: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by region"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceResult(GreenLangBase):
    """
    Compliance check result for a single regulatory framework.

    Evaluates whether the Category 12 emissions calculation meets
    the disclosure requirements of the specified framework.
    """

    framework: ComplianceFramework = Field(..., description="Compliance framework evaluated")
    status: ComplianceStatus = Field(..., description="Overall compliance status")
    rules_checked: int = Field(..., ge=0, description="Number of rules checked")
    rules_passed: int = Field(..., ge=0, description="Number of rules passed")
    rules_failed: int = Field(..., ge=0, description="Number of rules failed")
    findings: List[str] = Field(
        default_factory=list,
        description="Detailed findings (issues, gaps, recommendations)"
    )
    checked_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(GreenLangBase):
    """
    Provenance chain record for a single pipeline stage.

    Captures the SHA-256 input/output hashes at each processing stage
    to create an immutable audit trail for regulatory assurance.
    """

    stage: ProvenanceStage = Field(..., description="Pipeline stage")
    input_hash: str = Field(..., description="SHA-256 hash of stage input data")
    output_hash: str = Field(..., description="SHA-256 hash of stage output data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Stage completion time")
    agent_id: str = Field(default=AGENT_ID, description="Agent identifier")
    agent_version: str = Field(default=VERSION, description="Agent version")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional stage metadata (parameters, EF sources, etc.)"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_material_ef(
    material: str,
    treatment: str,
) -> Optional[Decimal]:
    """
    Look up the emission factor for a material-treatment pair.

    Args:
        material: MaterialType value (e.g., 'plastic', 'metal').
        treatment: TreatmentMethod value (e.g., 'landfill', 'recycling').

    Returns:
        Emission factor in kgCO2e/kg, or None if the combination is not defined.

    Example:
        >>> get_material_ef('plastic', 'landfill')
        Decimal('0.021')
        >>> get_material_ef('aluminum', 'recycling')
        Decimal('-9.120')
    """
    material_efs = MATERIAL_TREATMENT_EFS.get(material)
    if material_efs is None:
        return None
    return material_efs.get(treatment)


def get_product_composition(
    category: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Get the default material composition for a product category.

    Args:
        category: ProductCategory value (e.g., 'electronics', 'packaging').

    Returns:
        Dictionary of {MaterialType: fraction} summing to 1.0, or None if unknown.

    Example:
        >>> comp = get_product_composition('electronics')
        >>> comp['plastic']
        Decimal('0.40')
    """
    return PRODUCT_MATERIAL_COMPOSITIONS.get(category)


def get_regional_treatment_mix(
    region: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Get the default end-of-life treatment mix for a region.

    Args:
        region: RegionalTreatmentProfile value (e.g., 'US', 'EU', 'GLOBAL').

    Returns:
        Dictionary of {TreatmentMethod: fraction} summing to 1.0, or None if unknown.

    Example:
        >>> mix = get_regional_treatment_mix('US')
        >>> mix['landfill']
        Decimal('0.50')
    """
    return REGIONAL_TREATMENT_MIXES.get(region)


def get_landfill_params(
    material: str,
    climate_zone: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Get IPCC FOD landfill parameters for a material-climate combination.

    Args:
        material: MaterialType value (e.g., 'paper', 'wood', 'organic').
        climate_zone: ClimateZone value (e.g., 'boreal_temperate_dry').

    Returns:
        Dictionary with keys {DOC, DOCf, MCF, k, F, OX}, or None if not defined.

    Example:
        >>> params = get_landfill_params('paper', 'boreal_temperate_wet')
        >>> params['k']
        Decimal('0.060')
    """
    material_params = LANDFILL_FOD_PARAMETERS.get(material)
    if material_params is None:
        return None
    return material_params.get(climate_zone)


def get_incineration_params(
    material: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Get incineration parameters for a material type.

    Args:
        material: MaterialType value (e.g., 'plastic', 'paper', 'textile').

    Returns:
        Dictionary with keys {dry_matter_fraction, carbon_fraction,
        fossil_carbon_fraction, oxidation_factor}, or None if not defined.

    Example:
        >>> params = get_incineration_params('plastic')
        >>> params['fossil_carbon_fraction']
        Decimal('1.00')
    """
    return INCINERATION_PARAMETERS.get(material)


def get_gas_collection_eff(
    landfill_type: str,
) -> Optional[Decimal]:
    """
    Get landfill gas collection efficiency for a landfill type.

    Args:
        landfill_type: LandfillType value (e.g., 'engineered_with_gas').

    Returns:
        Gas collection efficiency as a fraction (0.0-0.75), or None if not defined.

    Example:
        >>> get_gas_collection_eff('engineered_with_gas')
        Decimal('0.75')
        >>> get_gas_collection_eff('unmanaged_deep')
        Decimal('0.00')
    """
    return GAS_COLLECTION_EFFICIENCY.get(landfill_type)


def get_energy_recovery(
    region: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Get waste-to-energy recovery factors for a region.

    Args:
        region: RegionalTreatmentProfile value (e.g., 'US', 'EU').

    Returns:
        Dictionary with keys {wte_efficiency, displaced_grid_ef}, or None.

    Example:
        >>> factors = get_energy_recovery('US')
        >>> factors['displaced_grid_ef']
        Decimal('0.386')
    """
    return ENERGY_RECOVERY_FACTORS.get(region)


def get_recycling_processing_ef(
    material: str,
) -> Optional[Decimal]:
    """
    Get recycling processing emission factor for a material type.

    Args:
        material: MaterialType value (e.g., 'plastic', 'glass').

    Returns:
        Processing emission factor in kgCO2e/kg, or None if not defined.

    Example:
        >>> get_recycling_processing_ef('aluminum')
        Decimal('0.150')
    """
    return RECYCLING_PROCESSING_EFS.get(material)


def get_avoided_ef(
    material: str,
) -> Optional[Decimal]:
    """
    Get avoided emission factor (virgin substitution credit) for a material.

    Args:
        material: MaterialType value (e.g., 'aluminum', 'steel').

    Returns:
        Avoided emission factor in kgCO2e/kg (negative value), or None.

    Example:
        >>> get_avoided_ef('aluminum')
        Decimal('-9.120')
    """
    return AVOIDED_EMISSION_FACTORS.get(material)


def get_product_weight(
    category: str,
) -> Optional[Decimal]:
    """
    Get default weight per unit for a product category.

    Args:
        category: ProductCategory value (e.g., 'electronics', 'furniture').

    Returns:
        Default weight in kg per unit, or None if not defined.

    Example:
        >>> get_product_weight('appliances')
        Decimal('35.0')
    """
    return PRODUCT_WEIGHT_DEFAULTS.get(category)


def get_composting_ef(
    is_home: bool = False,
) -> Dict[str, Decimal]:
    """
    Get composting CH4 and N2O emission factors.

    Args:
        is_home: True for home composting (higher emissions), False for industrial.

    Returns:
        Dictionary with keys {ch4_ef, n2o_ef} in g gas per kg wet waste.

    Example:
        >>> efs = get_composting_ef(is_home=False)
        >>> efs['ch4_ef']
        Decimal('4.0')
    """
    if is_home:
        return {
            "ch4_ef": COMPOSTING_EMISSION_FACTORS["ch4_ef_home"],
            "n2o_ef": COMPOSTING_EMISSION_FACTORS["n2o_ef_home"],
        }
    return {
        "ch4_ef": COMPOSTING_EMISSION_FACTORS["ch4_ef_industrial"],
        "n2o_ef": COMPOSTING_EMISSION_FACTORS["n2o_ef_industrial"],
    }


def get_ad_ef(
    ad_type: str = "enclosed_standard",
) -> Optional[Dict[str, Decimal]]:
    """
    Get anaerobic digestion emission parameters by AD plant type.

    Args:
        ad_type: AD type ('enclosed_modern', 'enclosed_standard',
                 'open_lagoon', 'covered_lagoon').

    Returns:
        Dictionary with keys {fugitive_ch4_per_m3_biogas, capture_efficiency},
        or None if not defined.

    Example:
        >>> params = get_ad_ef('enclosed_modern')
        >>> params['capture_efficiency']
        Decimal('0.98')
    """
    return AD_EMISSION_FACTORS.get(ad_type)


def get_dc_rule(
    rule_id: str,
) -> Optional[Dict[str, str]]:
    """
    Get a double-counting prevention rule by its identifier.

    Args:
        rule_id: Rule identifier (e.g., 'DC-EOL-001').

    Returns:
        Dictionary with keys {id, description, scope_boundary}, or None.

    Example:
        >>> rule = get_dc_rule('DC-EOL-004')
        >>> 'memo' in rule['description'].lower()
        True
    """
    for rule in DC_RULES:
        if rule["id"] == rule_id:
            return rule
    return None


def get_framework_rules(
    framework: str,
) -> Optional[List[str]]:
    """
    Get compliance framework disclosure requirements.

    Args:
        framework: ComplianceFramework value (e.g., 'ghg_protocol', 'csrd_esrs').

    Returns:
        List of required disclosure items, or None if framework not found.

    Example:
        >>> rules = get_framework_rules('ghg_protocol')
        >>> 'total_category12_emissions_tco2e' in rules
        True
    """
    return COMPLIANCE_FRAMEWORK_RULES.get(framework)


def get_dqi_score(
    dimension: str,
    tier: str,
) -> Optional[Decimal]:
    """
    Get the DQI score for a dimension-tier combination.

    Args:
        dimension: DQIDimension value (e.g., 'temporal', 'geographical').
        tier: DataQualityTier value (e.g., 'tier_1', 'tier_2', 'tier_3').

    Returns:
        DQI score (Decimal 1.0-5.0), or None if not defined.

    Example:
        >>> get_dqi_score('temporal', 'tier_3')
        Decimal('1.0')
    """
    dim_scores = DQI_SCORING.get(dimension)
    if dim_scores is None:
        return None
    return dim_scores.get(tier)


def get_uncertainty_range(
    method: str,
    tier: str,
) -> Optional[Decimal]:
    """
    Get the uncertainty range for a calculation method and data quality tier.

    Args:
        method: CalculationMethod value (e.g., 'waste_type_specific', 'average_data').
        tier: DataQualityTier value (e.g., 'tier_1', 'tier_2', 'tier_3').

    Returns:
        Uncertainty range as a fraction (e.g., 0.35 means +/- 35%), or None.

    Example:
        >>> get_uncertainty_range('producer_specific', 'tier_3')
        Decimal('0.05')
    """
    method_ranges = UNCERTAINTY_RANGES.get(method)
    if method_ranges is None:
        return None
    return method_ranges.get(tier)


def calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Produces a deterministic hash from any combination of Pydantic models,
    dictionaries, and primitive values for audit trail integrity.

    Args:
        *inputs: Variable number of input objects to hash. Pydantic models are
                 serialized to sorted JSON. Other types are converted via str().

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).

    Example:
        >>> h = calculate_provenance_hash("input_data", Decimal("42.5"))
        >>> len(h)
        64
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, BaseModel):
            hash_input += inp.model_dump_json(sort_keys=True)
        elif isinstance(inp, dict):
            hash_input += json.dumps(inp, sort_keys=True, default=str)
        else:
            hash_input += str(inp)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",

    # Enumerations
    "MaterialType",
    "TreatmentMethod",
    "ProductCategory",
    "RegionalTreatmentProfile",
    "CalculationMethod",
    "LandfillType",
    "ClimateZone",
    "IncinerationType",
    "RecyclingApproach",
    "EFSource",
    "DataQualityTier",
    "DQIDimension",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "ProvenanceStage",
    "UncertaintyMethod",
    "BatchStatus",
    "GWPSource",
    "EmissionGas",
    "WasteHierarchyLevel",
    "CircularityMetric",

    # Constants
    "GWP_VALUES",
    "MATERIAL_TREATMENT_EFS",
    "PRODUCT_MATERIAL_COMPOSITIONS",
    "REGIONAL_TREATMENT_MIXES",
    "LANDFILL_FOD_PARAMETERS",
    "INCINERATION_PARAMETERS",
    "GAS_COLLECTION_EFFICIENCY",
    "ENERGY_RECOVERY_FACTORS",
    "RECYCLING_PROCESSING_EFS",
    "AVOIDED_EMISSION_FACTORS",
    "PRODUCT_WEIGHT_DEFAULTS",
    "COMPOSTING_EMISSION_FACTORS",
    "AD_EMISSION_FACTORS",
    "DC_RULES",
    "COMPLIANCE_FRAMEWORK_RULES",
    "DQI_SCORING",
    "UNCERTAINTY_RANGES",

    # Pydantic Models
    "ProductEOLInput",
    "MaterialComposition",
    "TreatmentScenario",
    "WasteTypeResult",
    "AverageDataResult",
    "ProducerSpecificResult",
    "CalculationResult",
    "AvoidedEmissions",
    "CircularityScore",
    "AggregationResult",
    "ComplianceResult",
    "ProvenanceRecord",
    "DataQualityScore",
    "UncertaintyResult",

    # Helper Functions
    "get_material_ef",
    "get_product_composition",
    "get_regional_treatment_mix",
    "get_landfill_params",
    "get_incineration_params",
    "get_gas_collection_eff",
    "get_energy_recovery",
    "get_recycling_processing_ef",
    "get_avoided_ef",
    "get_product_weight",
    "get_composting_ef",
    "get_ad_ef",
    "get_dc_rule",
    "get_framework_rules",
    "get_dqi_score",
    "get_uncertainty_range",
    "calculate_provenance_hash",
]
