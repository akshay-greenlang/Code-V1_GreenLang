"""
Processing of Sold Products Agent Models (AGENT-MRV-023)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 10
(Processing of Sold Products) emissions calculations.

Supports:
- 5 calculation methods (site-specific direct/energy/fuel, average-data, spend-based)
- 12 intermediate product categories (metals, plastics, chemicals, food, textiles, etc.)
- 18 processing types (machining, molding, casting, refining, sintering, etc.)
- Processing chain modeling for multi-step downstream transformations
- Site-specific energy and fuel consumption tracking with grid/fuel EFs
- Average-data emission factors per product category (kgCO2e/tonne processed)
- EEIO spend-based factors for 12 NAICS downstream sectors
- Multi-currency conversion with CPI deflation (2015-2025)
- 16-region grid emission factors (kgCO2e/kWh) for site-specific energy method
- 6 fuel types with combustion emission factors (kgCO2e/kWh thermal)
- 8 multi-step processing chains with combined emission factors
- Product-to-processing-type applicability mapping
- Energy intensity ranges (low/mid/high kWh/tonne) for 18 processing types
- 4 allocation methods (mass, revenue, units, equal)
- Data quality indicators (DQI) with 5-dimension scoring
- Uncertainty quantification (analytical, Monte Carlo, bootstrap)
- 8 double-counting prevention rules (DC-PSP-001 through DC-PSP-008)
- Compliance checking for 7 frameworks (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, SB 253, GRI)
- SHA-256 provenance chain with 10-stage pipeline
- Frozen (immutable) Pydantic models for audit trail integrity

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.processing_sold_products.models import (
    ...     IntermediateProductInput, IntermediateProductCategory, ProcessingType, ProductUnit
    ... )
    >>> product = IntermediateProductInput(
    ...     product_id="PROD-2026-001",
    ...     category=IntermediateProductCategory.METALS_FERROUS,
    ...     processing_type=ProcessingType.MACHINING,
    ...     quantity=Decimal("500"),
    ...     unit=ProductUnit.TONNE,
    ...     customer_id="CUST-001",
    ...     customer_country=GridRegion.DE
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict
import hashlib
import json

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"

# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class IntermediateProductCategory(str, Enum):
    """Intermediate product categories sold to downstream processors.

    These represent the 12 primary categories of intermediate goods that
    undergo further processing after sale, per GHG Protocol Scope 3
    Category 10 guidance.
    """

    METALS_FERROUS = "metals_ferrous"  # Steel, iron, cast iron
    METALS_NON_FERROUS = "metals_non_ferrous"  # Aluminum, copper, zinc, titanium
    PLASTICS_THERMOPLASTIC = "plastics_thermoplastic"  # PE, PP, PET, PVC, ABS
    PLASTICS_THERMOSET = "plastics_thermoset"  # Epoxy, phenolic, polyurethane
    CHEMICALS = "chemicals"  # Industrial chemicals, solvents, resins
    FOOD_INGREDIENTS = "food_ingredients"  # Flour, sugar, oils, starch
    TEXTILES = "textiles"  # Fibers, yarns, greige fabrics
    ELECTRONICS = "electronics"  # Wafers, PCBs, components, ICs
    GLASS_CERAMICS = "glass_ceramics"  # Glass cullet, ceramic powders
    WOOD_PAPER = "wood_paper"  # Lumber, pulp, paper stock
    MINERALS = "minerals"  # Cement clinker, calcium carbonate, silica
    AGRICULTURAL = "agricultural"  # Raw agricultural commodities


class ProcessingType(str, Enum):
    """Types of downstream processing operations applied to intermediate products.

    These 18 processing types cover the major industrial transformation steps
    with distinct energy intensity profiles and emission factor characteristics.
    """

    MACHINING = "machining"  # CNC milling, turning, drilling, grinding
    STAMPING = "stamping"  # Metal stamping, pressing, blanking
    WELDING = "welding"  # Arc, MIG, TIG, spot, laser welding
    HEAT_TREATMENT = "heat_treatment"  # Annealing, tempering, case hardening
    INJECTION_MOLDING = "injection_molding"  # Thermoplastic injection molding
    EXTRUSION = "extrusion"  # Plastic/metal extrusion
    BLOW_MOLDING = "blow_molding"  # Hollow plastic part forming
    CASTING = "casting"  # Sand, die, investment casting
    FORGING = "forging"  # Open-die, closed-die, upset forging
    COATING = "coating"  # Paint, powder coat, plating, anodizing
    ASSEMBLY = "assembly"  # Mechanical assembly, bonding, fastening
    CHEMICAL_REACTION = "chemical_reaction"  # Synthesis, polymerization, cracking
    REFINING = "refining"  # Purification, distillation, electrolysis
    MILLING = "milling"  # Grain milling, mineral grinding, ball milling
    DRYING = "drying"  # Thermal drying, spray drying, freeze drying
    SINTERING = "sintering"  # Powder metallurgy, ceramic sintering
    FERMENTATION = "fermentation"  # Biological fermentation, brewing
    TEXTILE_FINISHING = "textile_finishing"  # Dyeing, bleaching, mercerizing


class CalculationMethod(str, Enum):
    """Calculation methods for processing of sold products per GHG Protocol.

    The GHG Protocol Technical Guidance for Category 10 defines these five
    approaches in order of data quality preference.
    """

    SITE_SPECIFIC_DIRECT = "site_specific_direct"  # Customer-reported total processing emissions
    SITE_SPECIFIC_ENERGY = "site_specific_energy"  # Energy consumption x grid/fuel EF
    SITE_SPECIFIC_FUEL = "site_specific_fuel"  # Fuel consumption x combustion EF
    AVERAGE_DATA = "average_data"  # Product category x processing type EF
    SPEND_BASED = "spend_based"  # Revenue x EEIO sector factor


class EnergyType(str, Enum):
    """Energy types consumed during downstream processing operations."""

    ELECTRICITY = "electricity"  # Grid or renewable electricity
    NATURAL_GAS = "natural_gas"  # Pipeline natural gas
    DIESEL = "diesel"  # Diesel fuel
    HFO = "hfo"  # Heavy fuel oil
    LPG = "lpg"  # Liquefied petroleum gas
    COAL = "coal"  # Coal / anthracite / lignite


class FuelType(str, Enum):
    """Fuel types for site-specific fuel-based calculations.

    Includes the six primary fossil fuels plus biomass for thermal
    energy generation at downstream processing facilities.
    """

    NATURAL_GAS = "natural_gas"  # kgCO2e per kWh thermal
    DIESEL = "diesel"  # kgCO2e per kWh thermal
    HFO = "hfo"  # Heavy fuel oil
    LPG = "lpg"  # Liquefied petroleum gas
    COAL = "coal"  # Coal / anthracite
    BIOMASS = "biomass"  # Wood pellets, bagasse, biogas (biogenic)


class GridRegion(str, Enum):
    """Grid regions with distinct electricity emission factors.

    Covers 15 major industrial countries plus a global average fallback.
    Factors sourced from IEA 2024, eGRID 2024, BEIS 2024.
    """

    US = "US"  # United States (eGRID national average)
    GB = "GB"  # United Kingdom (BEIS)
    DE = "DE"  # Germany (UBA)
    FR = "FR"  # France (ADEME)
    CN = "CN"  # China (MEE)
    IN = "IN"  # India (CEA)
    JP = "JP"  # Japan (MOE)
    KR = "KR"  # South Korea (KEEI)
    BR = "BR"  # Brazil (MCTIC)
    CA = "CA"  # Canada (ECCC)
    AU = "AU"  # Australia (CER)
    MX = "MX"  # Mexico (SEMARNAT)
    IT = "IT"  # Italy (ISPRA)
    ES = "ES"  # Spain (MITECO)
    PL = "PL"  # Poland (KOBiZE)
    GLOBAL = "GLOBAL"  # IEA world average fallback


class NAICSSector(str, Enum):
    """NAICS sectors for downstream processing EEIO factor lookup.

    These 12 sectors cover the primary downstream manufacturing industries
    that process intermediate products, sourced from EPA USEEIO v2.0.
    """

    IRON_STEEL_MILLS = "331"  # Iron and Steel Mills and Ferroalloy Mfg
    FABRICATED_METAL = "332"  # Fabricated Metal Product Manufacturing
    CHEMICAL_MFG = "325"  # Chemical Manufacturing
    PLASTICS_RUBBER = "326"  # Plastics and Rubber Products Manufacturing
    FOOD_MFG = "311"  # Food Manufacturing
    TEXTILE_MILLS = "313"  # Textile Mills
    COMPUTER_ELECTRONIC = "334"  # Computer and Electronic Product Mfg
    NONMETALLIC_MINERAL = "327"  # Nonmetallic Mineral Product Mfg
    WOOD_PRODUCT = "321"  # Wood Product Manufacturing
    PAPER_MFG = "322"  # Paper Manufacturing
    TRANSPORT_EQUIP = "336"  # Transportation Equipment Manufacturing
    ELECTRICAL_EQUIP = "335"  # Electrical Equipment Manufacturing


class Currency(str, Enum):
    """ISO 4217 currency codes for spend-based calculations."""

    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    CNY = "CNY"  # Chinese Yuan
    INR = "INR"  # Indian Rupee
    CAD = "CAD"  # Canadian Dollar
    AUD = "AUD"  # Australian Dollar
    KRW = "KRW"  # South Korean Won
    BRL = "BRL"  # Brazilian Real
    MXN = "MXN"  # Mexican Peso
    CHF = "CHF"  # Swiss Franc


class ProcessingChainType(str, Enum):
    """Pre-defined multi-step processing chains for common product flows.

    Each chain represents a typical downstream transformation pathway
    with a combined emission factor covering all processing steps.
    """

    METALS_AUTOMOTIVE = "metals_automotive"  # Steel -> stamping -> welding -> coating -> assembly
    ALUMINUM_PACKAGING = "aluminum_packaging"  # Al sheet -> stamping -> coating -> assembly
    PLASTIC_PACKAGING = "plastic_packaging"  # Resin -> injection molding -> assembly
    SEMICONDUCTOR = "semiconductor"  # Wafer -> etching -> assembly -> testing
    FOOD_PRODUCTS = "food_products"  # Ingredient -> milling -> drying -> packaging
    TEXTILE_GARMENTS = "textile_garments"  # Yarn -> textile finishing -> assembly
    GLASS_BOTTLES = "glass_bottles"  # Cullet -> casting -> coating
    PAPER_PRODUCTS = "paper_products"  # Pulp -> milling -> drying -> coating


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges and DQI scoring."""

    TIER_1 = "tier_1"  # Site-specific / primary data from customer
    TIER_2 = "tier_2"  # Regional / industry-specific secondary data
    TIER_3 = "tier_3"  # Global average / spend-based estimates


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol Scope 3 guidance."""

    RELIABILITY = "reliability"  # Source reliability and verification status
    COMPLETENESS = "completeness"  # Fraction of data coverage across products
    TEMPORAL = "temporal"  # Temporal correlation to reporting year
    GEOGRAPHICAL = "geographical"  # Geographical correlation to processing location
    TECHNOLOGICAL = "technological"  # Technological correlation to actual process


class ComplianceFramework(str, Enum):
    """Regulatory/reporting frameworks for compliance validation."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard (Cat 10)
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 Climate Change
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    SB_253 = "sb_253"  # California SB 253 (Climate Corporate Data Accountability Act)
    GRI = "gri"  # GRI 305 Emissions Standard


class ComplianceStatus(str, Enum):
    """Result status from a compliance check against a framework."""

    PASS = "pass"  # Fully compliant with all applicable rules
    FAIL = "fail"  # Non-compliant with one or more mandatory rules
    WARNING = "warning"  # Compliant but with recommendations
    NOT_APPLICABLE = "not_applicable"  # Framework does not apply to this scope


class PipelineStage(str, Enum):
    """Processing pipeline stages for the 10-stage calculation workflow."""

    VALIDATE = "validate"  # Input validation and schema conformance
    CLASSIFY = "classify"  # Product and processing type classification
    NORMALIZE = "normalize"  # Unit normalization (mass, currency, energy)
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution and selection
    CALCULATE = "calculate"  # Core emissions calculation
    ALLOCATE = "allocate"  # Product allocation across customers
    AGGREGATE = "aggregate"  # Aggregation by category, method, country
    COMPLIANCE = "compliance"  # Compliance checks against frameworks
    PROVENANCE = "provenance"  # Provenance hash chain generation
    SEAL = "seal"  # Final chain sealing and audit record


class ProvenanceStage(str, Enum):
    """Provenance tracking stages mirroring the pipeline stages."""

    VALIDATE = "validate"  # Input validation hash
    CLASSIFY = "classify"  # Classification hash
    NORMALIZE = "normalize"  # Normalization hash
    RESOLVE_EFS = "resolve_efs"  # EF resolution hash
    CALCULATE = "calculate"  # Calculation hash
    ALLOCATE = "allocate"  # Allocation hash
    AGGREGATE = "aggregate"  # Aggregation hash
    COMPLIANCE = "compliance"  # Compliance check hash
    PROVENANCE = "provenance"  # Provenance generation hash
    SEAL = "seal"  # Final seal hash


class AllocationMethod(str, Enum):
    """Methods for allocating processing emissions across products."""

    MASS = "mass"  # Allocate by mass fraction (kg/tonne)
    REVENUE = "revenue"  # Allocate by revenue fraction
    UNITS = "units"  # Allocate by unit count fraction
    EQUAL = "equal"  # Equal allocation across all products


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods for emissions estimates."""

    ANALYTICAL = "analytical"  # Analytical error propagation (Gaussian)
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation (N iterations)
    BOOTSTRAP = "bootstrap"  # Bootstrap resampling


class BatchStatus(str, Enum):
    """Batch calculation processing status."""

    PENDING = "pending"  # Awaiting processing
    RUNNING = "running"  # Currently processing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Processing failed


class AuditAction(str, Enum):
    """Audit trail action types for change tracking."""

    CREATE = "create"  # New record creation
    UPDATE = "update"  # Record update / recalculation
    DELETE = "delete"  # Record deletion / invalidation
    CALCULATE = "calculate"  # Emissions calculation performed
    VALIDATE = "validate"  # Compliance validation performed
    EXPORT = "export"  # Data export / report generation


class ProductUnit(str, Enum):
    """Units of measure for intermediate product quantities."""

    TONNE = "tonne"  # Metric tonnes (1000 kg)
    KG = "kg"  # Kilograms
    UNIT = "unit"  # Discrete units / pieces
    M2 = "m2"  # Square metres (sheet/panel products)
    M3 = "m3"  # Cubic metres (volume products)


# ==============================================================================
# CONSTANT TABLES
# ==============================================================================

# Quantization constants for Decimal arithmetic
_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")

# ---------------------------------------------------------------------------
# 1. Processing Emission Factors (average-data method)
# ---------------------------------------------------------------------------
# kgCO2e per tonne of intermediate product processed downstream.
# Source: GHG Protocol Cat 10 Technical Guidance, EPA USEEIO, Ecoinvent 3.10
PROCESSING_EMISSION_FACTORS: Dict[str, Decimal] = {
    IntermediateProductCategory.METALS_FERROUS.value: Decimal("280"),
    IntermediateProductCategory.METALS_NON_FERROUS.value: Decimal("380"),
    IntermediateProductCategory.PLASTICS_THERMOPLASTIC.value: Decimal("520"),
    IntermediateProductCategory.PLASTICS_THERMOSET.value: Decimal("450"),
    IntermediateProductCategory.CHEMICALS.value: Decimal("680"),
    IntermediateProductCategory.FOOD_INGREDIENTS.value: Decimal("130"),
    IntermediateProductCategory.TEXTILES.value: Decimal("350"),
    IntermediateProductCategory.ELECTRONICS.value: Decimal("950"),
    IntermediateProductCategory.GLASS_CERAMICS.value: Decimal("580"),
    IntermediateProductCategory.WOOD_PAPER.value: Decimal("190"),
    IntermediateProductCategory.MINERALS.value: Decimal("250"),
    IntermediateProductCategory.AGRICULTURAL.value: Decimal("110"),
}

# ---------------------------------------------------------------------------
# 2. Energy Intensity Factors
# ---------------------------------------------------------------------------
# kWh of energy consumed per tonne of product processed, by processing type.
# Source: US DOE Manufacturing Energy Consumption Survey (MECS), IEA 2024
ENERGY_INTENSITY_FACTORS: Dict[str, Decimal] = {
    ProcessingType.MACHINING.value: Decimal("280"),
    ProcessingType.STAMPING.value: Decimal("140"),
    ProcessingType.WELDING.value: Decimal("220"),
    ProcessingType.HEAT_TREATMENT.value: Decimal("380"),
    ProcessingType.INJECTION_MOLDING.value: Decimal("520"),
    ProcessingType.EXTRUSION.value: Decimal("340"),
    ProcessingType.BLOW_MOLDING.value: Decimal("400"),
    ProcessingType.CASTING.value: Decimal("750"),
    ProcessingType.FORGING.value: Decimal("580"),
    ProcessingType.COATING.value: Decimal("120"),
    ProcessingType.ASSEMBLY.value: Decimal("45"),
    ProcessingType.CHEMICAL_REACTION.value: Decimal("1100"),
    ProcessingType.REFINING.value: Decimal("900"),
    ProcessingType.MILLING.value: Decimal("190"),
    ProcessingType.DRYING.value: Decimal("310"),
    ProcessingType.SINTERING.value: Decimal("1200"),
    ProcessingType.FERMENTATION.value: Decimal("160"),
    ProcessingType.TEXTILE_FINISHING.value: Decimal("420"),
}

# ---------------------------------------------------------------------------
# 3. Grid Emission Factors
# ---------------------------------------------------------------------------
# kgCO2e per kWh of electricity consumed, by grid region.
# Source: IEA CO2 Emissions from Fuel Combustion 2024, eGRID 2024, BEIS 2024
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    GridRegion.US.value: Decimal("0.417"),
    GridRegion.GB.value: Decimal("0.233"),
    GridRegion.DE.value: Decimal("0.348"),
    GridRegion.FR.value: Decimal("0.052"),
    GridRegion.CN.value: Decimal("0.555"),
    GridRegion.IN.value: Decimal("0.708"),
    GridRegion.JP.value: Decimal("0.462"),
    GridRegion.KR.value: Decimal("0.424"),
    GridRegion.BR.value: Decimal("0.075"),
    GridRegion.CA.value: Decimal("0.120"),
    GridRegion.AU.value: Decimal("0.656"),
    GridRegion.MX.value: Decimal("0.431"),
    GridRegion.IT.value: Decimal("0.256"),
    GridRegion.ES.value: Decimal("0.175"),
    GridRegion.PL.value: Decimal("0.635"),
    GridRegion.GLOBAL.value: Decimal("0.475"),
}

# ---------------------------------------------------------------------------
# 4. Fuel Emission Factors
# ---------------------------------------------------------------------------
# kgCO2e per kWh thermal output, by fuel type.
# Source: IPCC 2006 Vol 2 Ch 2, DEFRA 2024 conversion factors
FUEL_EMISSION_FACTORS: Dict[str, Decimal] = {
    FuelType.NATURAL_GAS.value: Decimal("2.024"),
    FuelType.DIESEL.value: Decimal("2.706"),
    FuelType.HFO.value: Decimal("3.114"),
    FuelType.LPG.value: Decimal("1.557"),
    FuelType.COAL.value: Decimal("2.883"),
    FuelType.BIOMASS.value: Decimal("0.015"),
}

# ---------------------------------------------------------------------------
# 5. EEIO Sector Factors (spend-based method)
# ---------------------------------------------------------------------------
# EF: kgCO2e per USD of sector output (producer price)
# margin: producer-to-purchaser margin fraction for margin removal
# Source: EPA USEEIO v2.0, Exiobase 3.8
EEIO_SECTOR_FACTORS: Dict[str, Dict[str, Decimal]] = {
    NAICSSector.IRON_STEEL_MILLS.value: {
        "ef": Decimal("0.820"),
        "margin": Decimal("0.15"),
    },
    NAICSSector.FABRICATED_METAL.value: {
        "ef": Decimal("0.540"),
        "margin": Decimal("0.20"),
    },
    NAICSSector.CHEMICAL_MFG.value: {
        "ef": Decimal("0.710"),
        "margin": Decimal("0.18"),
    },
    NAICSSector.PLASTICS_RUBBER.value: {
        "ef": Decimal("0.630"),
        "margin": Decimal("0.22"),
    },
    NAICSSector.FOOD_MFG.value: {
        "ef": Decimal("0.390"),
        "margin": Decimal("0.25"),
    },
    NAICSSector.TEXTILE_MILLS.value: {
        "ef": Decimal("0.480"),
        "margin": Decimal("0.20"),
    },
    NAICSSector.COMPUTER_ELECTRONIC.value: {
        "ef": Decimal("0.350"),
        "margin": Decimal("0.30"),
    },
    NAICSSector.NONMETALLIC_MINERAL.value: {
        "ef": Decimal("0.750"),
        "margin": Decimal("0.16"),
    },
    NAICSSector.WOOD_PRODUCT.value: {
        "ef": Decimal("0.420"),
        "margin": Decimal("0.18"),
    },
    NAICSSector.PAPER_MFG.value: {
        "ef": Decimal("0.560"),
        "margin": Decimal("0.17"),
    },
    NAICSSector.TRANSPORT_EQUIP.value: {
        "ef": Decimal("0.460"),
        "margin": Decimal("0.22"),
    },
    NAICSSector.ELECTRICAL_EQUIP.value: {
        "ef": Decimal("0.380"),
        "margin": Decimal("0.25"),
    },
}

# ---------------------------------------------------------------------------
# 6. Processing Chains (multi-step downstream)
# ---------------------------------------------------------------------------
# Each chain defines the ordered processing steps and a combined EF
# (kgCO2e per tonne through the full chain).
# Source: Ecoinvent 3.10 lifecycle modules, GHG Protocol Cat 10 examples
PROCESSING_CHAINS: Dict[str, Dict[str, Any]] = {
    ProcessingChainType.METALS_AUTOMOTIVE.value: {
        "steps": ["stamping", "welding", "heat_treatment", "coating", "assembly"],
        "combined_ef": Decimal("920"),
        "description": "Steel to automotive body panel assembly",
    },
    ProcessingChainType.ALUMINUM_PACKAGING.value: {
        "steps": ["stamping", "coating", "assembly"],
        "combined_ef": Decimal("480"),
        "description": "Aluminum sheet to beverage can production",
    },
    ProcessingChainType.PLASTIC_PACKAGING.value: {
        "steps": ["injection_molding", "assembly"],
        "combined_ef": Decimal("590"),
        "description": "Plastic resin to injection-molded packaging",
    },
    ProcessingChainType.SEMICONDUCTOR.value: {
        "steps": ["chemical_reaction", "coating", "assembly"],
        "combined_ef": Decimal("1350"),
        "description": "Silicon wafer to packaged semiconductor device",
    },
    ProcessingChainType.FOOD_PRODUCTS.value: {
        "steps": ["milling", "drying", "assembly"],
        "combined_ef": Decimal("380"),
        "description": "Food ingredient to packaged food product",
    },
    ProcessingChainType.TEXTILE_GARMENTS.value: {
        "steps": ["textile_finishing", "assembly"],
        "combined_ef": Decimal("510"),
        "description": "Raw textile to finished garment",
    },
    ProcessingChainType.GLASS_BOTTLES.value: {
        "steps": ["casting", "coating"],
        "combined_ef": Decimal("870"),
        "description": "Glass cullet to coated glass bottle",
    },
    ProcessingChainType.PAPER_PRODUCTS.value: {
        "steps": ["milling", "drying", "coating"],
        "combined_ef": Decimal("440"),
        "description": "Paper pulp to coated paper product",
    },
}

# ---------------------------------------------------------------------------
# 7. Currency Conversion Rates (to USD)
# ---------------------------------------------------------------------------
# Approximate mid-market rates for nominal conversion.
# Source: ECB, Federal Reserve, mid-2025 rates
CURRENCIES: Dict[str, Decimal] = {
    Currency.USD.value: Decimal("1.0000"),
    Currency.EUR.value: Decimal("1.0850"),
    Currency.GBP.value: Decimal("1.2650"),
    Currency.JPY.value: Decimal("0.006667"),
    Currency.CNY.value: Decimal("0.1378"),
    Currency.INR.value: Decimal("0.01198"),
    Currency.CAD.value: Decimal("0.7410"),
    Currency.AUD.value: Decimal("0.6520"),
    Currency.KRW.value: Decimal("0.000750"),
    Currency.BRL.value: Decimal("0.1990"),
    Currency.MXN.value: Decimal("0.0580"),
    Currency.CHF.value: Decimal("1.1280"),
}

# ---------------------------------------------------------------------------
# 8. CPI Deflators (base year 2021 = 1.0000)
# ---------------------------------------------------------------------------
# Source: US BLS CPI-U / OECD CPI
CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.8490"),
    2016: Decimal("0.8597"),
    2017: Decimal("0.8781"),
    2018: Decimal("0.8997"),
    2019: Decimal("0.9153"),
    2020: Decimal("0.9271"),
    2021: Decimal("1.0000"),
    2022: Decimal("1.0800"),
    2023: Decimal("1.1152"),
    2024: Decimal("1.1490"),
    2025: Decimal("1.1780"),
}

# ---------------------------------------------------------------------------
# 9. DQI Scoring Matrix
# ---------------------------------------------------------------------------
# Score 1-5 per dimension per tier (5 = best).
# Source: GHG Protocol Scope 3 Technical Guidance, Table 7.1
DQI_SCORING: Dict[str, Dict[str, int]] = {
    DQIDimension.RELIABILITY.value: {
        DataQualityTier.TIER_1.value: 5,
        DataQualityTier.TIER_2.value: 3,
        DataQualityTier.TIER_3.value: 1,
    },
    DQIDimension.COMPLETENESS.value: {
        DataQualityTier.TIER_1.value: 5,
        DataQualityTier.TIER_2.value: 3,
        DataQualityTier.TIER_3.value: 2,
    },
    DQIDimension.TEMPORAL.value: {
        DataQualityTier.TIER_1.value: 5,
        DataQualityTier.TIER_2.value: 4,
        DataQualityTier.TIER_3.value: 2,
    },
    DQIDimension.GEOGRAPHICAL.value: {
        DataQualityTier.TIER_1.value: 5,
        DataQualityTier.TIER_2.value: 3,
        DataQualityTier.TIER_3.value: 1,
    },
    DQIDimension.TECHNOLOGICAL.value: {
        DataQualityTier.TIER_1.value: 5,
        DataQualityTier.TIER_2.value: 4,
        DataQualityTier.TIER_3.value: 2,
    },
}

# ---------------------------------------------------------------------------
# 10. Uncertainty Ranges
# ---------------------------------------------------------------------------
# Half-width of the 95% confidence interval as a fraction, by method.
# Source: IPCC 2006 Vol 1 Ch 3 Table 3.2, GHG Protocol Cat 10 guidance
UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    CalculationMethod.SITE_SPECIFIC_DIRECT.value: {
        "min": Decimal("0.05"),
        "default": Decimal("0.10"),
        "max": Decimal("0.15"),
    },
    CalculationMethod.SITE_SPECIFIC_ENERGY.value: {
        "min": Decimal("0.08"),
        "default": Decimal("0.15"),
        "max": Decimal("0.20"),
    },
    CalculationMethod.SITE_SPECIFIC_FUEL.value: {
        "min": Decimal("0.08"),
        "default": Decimal("0.15"),
        "max": Decimal("0.22"),
    },
    CalculationMethod.AVERAGE_DATA.value: {
        "min": Decimal("0.20"),
        "default": Decimal("0.30"),
        "max": Decimal("0.50"),
    },
    CalculationMethod.SPEND_BASED.value: {
        "min": Decimal("0.30"),
        "default": Decimal("0.50"),
        "max": Decimal("0.70"),
    },
}

# ---------------------------------------------------------------------------
# 11. Product Category to Processing Type Applicability Map
# ---------------------------------------------------------------------------
# Maps each product category to its list of applicable processing types.
PRODUCT_CATEGORY_PROCESSING_MAP: Dict[str, List[str]] = {
    IntermediateProductCategory.METALS_FERROUS.value: [
        ProcessingType.MACHINING.value,
        ProcessingType.STAMPING.value,
        ProcessingType.WELDING.value,
        ProcessingType.HEAT_TREATMENT.value,
        ProcessingType.CASTING.value,
        ProcessingType.FORGING.value,
        ProcessingType.COATING.value,
        ProcessingType.ASSEMBLY.value,
    ],
    IntermediateProductCategory.METALS_NON_FERROUS.value: [
        ProcessingType.MACHINING.value,
        ProcessingType.STAMPING.value,
        ProcessingType.WELDING.value,
        ProcessingType.HEAT_TREATMENT.value,
        ProcessingType.CASTING.value,
        ProcessingType.FORGING.value,
        ProcessingType.COATING.value,
        ProcessingType.EXTRUSION.value,
        ProcessingType.ASSEMBLY.value,
    ],
    IntermediateProductCategory.PLASTICS_THERMOPLASTIC.value: [
        ProcessingType.INJECTION_MOLDING.value,
        ProcessingType.EXTRUSION.value,
        ProcessingType.BLOW_MOLDING.value,
        ProcessingType.COATING.value,
        ProcessingType.ASSEMBLY.value,
    ],
    IntermediateProductCategory.PLASTICS_THERMOSET.value: [
        ProcessingType.INJECTION_MOLDING.value,
        ProcessingType.CASTING.value,
        ProcessingType.COATING.value,
        ProcessingType.ASSEMBLY.value,
    ],
    IntermediateProductCategory.CHEMICALS.value: [
        ProcessingType.CHEMICAL_REACTION.value,
        ProcessingType.REFINING.value,
        ProcessingType.MILLING.value,
        ProcessingType.DRYING.value,
    ],
    IntermediateProductCategory.FOOD_INGREDIENTS.value: [
        ProcessingType.MILLING.value,
        ProcessingType.DRYING.value,
        ProcessingType.FERMENTATION.value,
        ProcessingType.ASSEMBLY.value,
    ],
    IntermediateProductCategory.TEXTILES.value: [
        ProcessingType.TEXTILE_FINISHING.value,
        ProcessingType.DRYING.value,
        ProcessingType.COATING.value,
        ProcessingType.ASSEMBLY.value,
    ],
    IntermediateProductCategory.ELECTRONICS.value: [
        ProcessingType.CHEMICAL_REACTION.value,
        ProcessingType.COATING.value,
        ProcessingType.ASSEMBLY.value,
        ProcessingType.SINTERING.value,
    ],
    IntermediateProductCategory.GLASS_CERAMICS.value: [
        ProcessingType.CASTING.value,
        ProcessingType.SINTERING.value,
        ProcessingType.COATING.value,
        ProcessingType.HEAT_TREATMENT.value,
    ],
    IntermediateProductCategory.WOOD_PAPER.value: [
        ProcessingType.MILLING.value,
        ProcessingType.DRYING.value,
        ProcessingType.COATING.value,
        ProcessingType.ASSEMBLY.value,
    ],
    IntermediateProductCategory.MINERALS.value: [
        ProcessingType.MILLING.value,
        ProcessingType.SINTERING.value,
        ProcessingType.HEAT_TREATMENT.value,
        ProcessingType.DRYING.value,
    ],
    IntermediateProductCategory.AGRICULTURAL.value: [
        ProcessingType.MILLING.value,
        ProcessingType.DRYING.value,
        ProcessingType.FERMENTATION.value,
        ProcessingType.ASSEMBLY.value,
    ],
}

# ---------------------------------------------------------------------------
# 12. Energy Intensity Ranges
# ---------------------------------------------------------------------------
# Low / mid / high ranges for each processing type (kWh/tonne).
# Source: US DOE Industrial Assessment Center data, IEA IETS annex studies
ENERGY_INTENSITY_RANGES: Dict[str, Dict[str, Decimal]] = {
    ProcessingType.MACHINING.value: {
        "low": Decimal("180"),
        "mid": Decimal("280"),
        "high": Decimal("420"),
    },
    ProcessingType.STAMPING.value: {
        "low": Decimal("90"),
        "mid": Decimal("140"),
        "high": Decimal("210"),
    },
    ProcessingType.WELDING.value: {
        "low": Decimal("140"),
        "mid": Decimal("220"),
        "high": Decimal("340"),
    },
    ProcessingType.HEAT_TREATMENT.value: {
        "low": Decimal("250"),
        "mid": Decimal("380"),
        "high": Decimal("550"),
    },
    ProcessingType.INJECTION_MOLDING.value: {
        "low": Decimal("350"),
        "mid": Decimal("520"),
        "high": Decimal("750"),
    },
    ProcessingType.EXTRUSION.value: {
        "low": Decimal("220"),
        "mid": Decimal("340"),
        "high": Decimal("500"),
    },
    ProcessingType.BLOW_MOLDING.value: {
        "low": Decimal("260"),
        "mid": Decimal("400"),
        "high": Decimal("580"),
    },
    ProcessingType.CASTING.value: {
        "low": Decimal("500"),
        "mid": Decimal("750"),
        "high": Decimal("1100"),
    },
    ProcessingType.FORGING.value: {
        "low": Decimal("380"),
        "mid": Decimal("580"),
        "high": Decimal("850"),
    },
    ProcessingType.COATING.value: {
        "low": Decimal("75"),
        "mid": Decimal("120"),
        "high": Decimal("180"),
    },
    ProcessingType.ASSEMBLY.value: {
        "low": Decimal("25"),
        "mid": Decimal("45"),
        "high": Decimal("80"),
    },
    ProcessingType.CHEMICAL_REACTION.value: {
        "low": Decimal("700"),
        "mid": Decimal("1100"),
        "high": Decimal("1600"),
    },
    ProcessingType.REFINING.value: {
        "low": Decimal("600"),
        "mid": Decimal("900"),
        "high": Decimal("1300"),
    },
    ProcessingType.MILLING.value: {
        "low": Decimal("120"),
        "mid": Decimal("190"),
        "high": Decimal("290"),
    },
    ProcessingType.DRYING.value: {
        "low": Decimal("200"),
        "mid": Decimal("310"),
        "high": Decimal("460"),
    },
    ProcessingType.SINTERING.value: {
        "low": Decimal("800"),
        "mid": Decimal("1200"),
        "high": Decimal("1750"),
    },
    ProcessingType.FERMENTATION.value: {
        "low": Decimal("100"),
        "mid": Decimal("160"),
        "high": Decimal("240"),
    },
    ProcessingType.TEXTILE_FINISHING.value: {
        "low": Decimal("280"),
        "mid": Decimal("420"),
        "high": Decimal("620"),
    },
}

# ---------------------------------------------------------------------------
# 13. Double-Counting Prevention Rules
# ---------------------------------------------------------------------------
# Rules to prevent double-counting between Category 10 and other categories.
# Source: GHG Protocol Scope 3 Standard, Chapter 10 boundary guidance
DC_RULES: Dict[str, Dict[str, str]] = {
    "DC-PSP-001": {
        "title": "Exclude Scope 1 / Scope 2 of reporter",
        "description": (
            "Processing emissions occurring at the reporting company's own "
            "facilities are Scope 1/2, not Category 10."
        ),
        "overlapping_category": "Scope 1 / Scope 2",
        "resolution": "Exclude from Category 10 if processing is at reporter's facility.",
    },
    "DC-PSP-002": {
        "title": "Exclude Category 1 (Purchased Goods & Services)",
        "description": (
            "Emissions from purchased raw materials (upstream) are Category 1, "
            "not Category 10. Only downstream processing counts."
        ),
        "overlapping_category": "Category 1",
        "resolution": "Include only post-sale processing; exclude upstream extraction/production.",
    },
    "DC-PSP-003": {
        "title": "Exclude Category 2 (Capital Goods)",
        "description": (
            "Processing equipment (machines, tools) at customer sites is Category 2 "
            "for the customer, not Category 10 for the seller of intermediate products."
        ),
        "overlapping_category": "Category 2",
        "resolution": "Exclude capital equipment emissions; include only energy/fuel for processing.",
    },
    "DC-PSP-004": {
        "title": "Exclude Category 4 (Upstream Transportation)",
        "description": (
            "Transportation of sold products to customer processing sites "
            "is Category 4 or 9, not Category 10."
        ),
        "overlapping_category": "Category 4 / Category 9",
        "resolution": "Exclude transport emissions between seller and downstream processor.",
    },
    "DC-PSP-005": {
        "title": "Exclude Category 11 (Use of Sold Products)",
        "description": (
            "Emissions from use of the final product by end consumers "
            "are Category 11, not Category 10."
        ),
        "overlapping_category": "Category 11",
        "resolution": "Include only processing/transformation; exclude end-use phase.",
    },
    "DC-PSP-006": {
        "title": "Exclude Category 12 (End-of-Life Treatment)",
        "description": (
            "Waste treatment of the final product is Category 12, "
            "not Category 10."
        ),
        "overlapping_category": "Category 12",
        "resolution": "Exclude end-of-life disposal; include only transformation processing.",
    },
    "DC-PSP-007": {
        "title": "Avoid multi-tier double counting",
        "description": (
            "If a product passes through multiple processing tiers before reaching "
            "end use, only include the first downstream processing step unless the "
            "reporter has visibility into the full chain."
        ),
        "overlapping_category": "Category 10 (multi-tier)",
        "resolution": "Count only known processing steps; document boundary assumptions.",
    },
    "DC-PSP-008": {
        "title": "Avoid overlap with franchisee Scope 1/2",
        "description": (
            "If the downstream processor is a franchisee, their processing emissions "
            "may be reported under Category 14. Avoid counting in both categories."
        ),
        "overlapping_category": "Category 14",
        "resolution": "Assign to Category 10 or Category 14, not both; document allocation.",
    },
}

# ---------------------------------------------------------------------------
# 14. Compliance Framework Rules
# ---------------------------------------------------------------------------
# Number of rules and key requirements per compliance framework.
COMPLIANCE_FRAMEWORK_RULES: Dict[str, Dict[str, Any]] = {
    ComplianceFramework.GHG_PROTOCOL.value: {
        "rule_count": 12,
        "required_disclosures": [
            "total_co2e",
            "method_used",
            "ef_sources",
            "product_categories",
            "exclusions",
            "dqi_score",
        ],
        "description": "GHG Protocol Scope 3 Standard, Category 10",
    },
    ComplianceFramework.ISO_14064.value: {
        "rule_count": 8,
        "required_disclosures": [
            "total_co2e",
            "uncertainty_analysis",
            "base_year",
            "methodology",
        ],
        "description": "ISO 14064-1:2018 indirect emissions quantification",
    },
    ComplianceFramework.CSRD_ESRS.value: {
        "rule_count": 10,
        "required_disclosures": [
            "total_co2e",
            "category_breakdown",
            "methodology",
            "targets",
            "actions",
        ],
        "description": "CSRD ESRS E1 Climate Change, Scope 3 downstream",
    },
    ComplianceFramework.CDP.value: {
        "rule_count": 9,
        "required_disclosures": [
            "total_co2e",
            "method_breakdown",
            "verification_status",
            "product_category_detail",
        ],
        "description": "CDP Climate Change Questionnaire C6.5",
    },
    ComplianceFramework.SBTI.value: {
        "rule_count": 7,
        "required_disclosures": [
            "total_co2e",
            "target_coverage",
            "progress_tracking",
            "base_year_recalculation",
        ],
        "description": "SBTi Corporate Net-Zero Standard",
    },
    ComplianceFramework.SB_253.value: {
        "rule_count": 6,
        "required_disclosures": [
            "total_co2e",
            "methodology",
            "assurance_opinion",
        ],
        "description": "California SB 253 Climate Corporate Data Accountability Act",
    },
    ComplianceFramework.GRI.value: {
        "rule_count": 5,
        "required_disclosures": [
            "total_co2e",
            "gases_included",
            "base_year",
            "standards_used",
        ],
        "description": "GRI 305 Emissions Standard",
    },
}


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================


class IntermediateProductInput(BaseModel):
    """Input model for a single intermediate product sold to a downstream processor.

    Represents one line item of sold product with its downstream processing
    characteristics. Optional fields allow site-specific data when available.

    Example:
        >>> product = IntermediateProductInput(
        ...     product_id="PROD-2026-001",
        ...     category=IntermediateProductCategory.METALS_FERROUS,
        ...     processing_type=ProcessingType.MACHINING,
        ...     quantity=Decimal("500"),
        ...     unit=ProductUnit.TONNE,
        ...     customer_id="CUST-001",
        ...     customer_country=GridRegion.DE
        ... )
    """

    product_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Unique identifier for the intermediate product"
    )
    category: IntermediateProductCategory = Field(
        ...,
        description="Product category classification"
    )
    processing_type: ProcessingType = Field(
        ...,
        description="Expected downstream processing type"
    )
    quantity: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Quantity of product sold"
    )
    unit: ProductUnit = Field(
        default=ProductUnit.TONNE,
        description="Unit of measure for quantity"
    )
    customer_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Unique identifier for the downstream customer/processor"
    )
    customer_country: GridRegion = Field(
        default=GridRegion.GLOBAL,
        description="Country where downstream processing occurs (for grid EF)"
    )
    processing_energy_kwh: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"),
        description="Site-specific energy consumed for processing (kWh)"
    )
    processing_emissions_kg: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"),
        description="Site-specific direct processing emissions (kgCO2e)"
    )
    fuel_type: Optional[FuelType] = Field(
        default=None,
        description="Fuel type used in downstream processing (if known)"
    )
    fuel_quantity_kwh: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"),
        description="Fuel consumed for processing (kWh thermal)"
    )

    model_config = ConfigDict(frozen=True)

    @validator("quantity")
    def validate_quantity(cls, v: Decimal) -> Decimal:
        """Validate quantity is positive."""
        if v <= Decimal("0"):
            raise ValueError(f"Quantity must be positive, got {v}")
        return v

    @validator("product_id")
    def validate_product_id(cls, v: str) -> str:
        """Validate product_id is non-empty and stripped."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Product ID must not be empty or whitespace")
        return stripped


class SiteSpecificInput(BaseModel):
    """Input model for site-specific calculation method.

    Used when the reporting company has access to customer-level processing
    data including energy consumption, fuel use, or direct emission reports.

    Example:
        >>> site_input = SiteSpecificInput(
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ...     products=[product1, product2],
        ...     method=CalculationMethod.SITE_SPECIFIC_ENERGY,
        ...     customer_data={"CUST-001": {"grid_region": "DE"}}
        ... )
    """

    org_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Reporting organization identifier"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year for the calculation"
    )
    products: List[IntermediateProductInput] = Field(
        ..., min_length=1,
        description="List of intermediate products sold during the reporting period"
    )
    method: CalculationMethod = Field(
        ...,
        description="Site-specific calculation method variant"
    )
    customer_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional customer-specific data for EF resolution"
    )

    model_config = ConfigDict(frozen=True)

    @validator("method")
    def validate_site_specific_method(cls, v: CalculationMethod) -> CalculationMethod:
        """Ensure method is one of the site-specific variants."""
        valid = {
            CalculationMethod.SITE_SPECIFIC_DIRECT,
            CalculationMethod.SITE_SPECIFIC_ENERGY,
            CalculationMethod.SITE_SPECIFIC_FUEL,
        }
        if v not in valid:
            raise ValueError(
                f"SiteSpecificInput requires a site-specific method, got '{v.value}'. "
                f"Valid options: {[m.value for m in valid]}"
            )
        return v


class AverageDataInput(BaseModel):
    """Input model for average-data calculation method.

    Used when site-specific data is unavailable. Applies product category
    and processing type emission factors to the sold quantities.

    Example:
        >>> avg_input = AverageDataInput(
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ...     products=[product1, product2],
        ...     processing_types=[ProcessingType.MACHINING, ProcessingType.STAMPING]
        ... )
    """

    org_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Reporting organization identifier"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year for the calculation"
    )
    products: List[IntermediateProductInput] = Field(
        ..., min_length=1,
        description="List of intermediate products sold during the reporting period"
    )
    processing_types: List[ProcessingType] = Field(
        default_factory=list,
        description="Expected processing types (overrides product-level if non-empty)"
    )

    model_config = ConfigDict(frozen=True)


class SpendBasedInput(BaseModel):
    """Input model for spend-based (EEIO) calculation method.

    Used as a screening-level estimate when neither site-specific nor
    average-data approaches are feasible. Applies sector-level EEIO
    factors to revenue after currency conversion and CPI deflation.

    Example:
        >>> spend_input = SpendBasedInput(
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ...     revenue=Decimal("5000000"),
        ...     currency=Currency.EUR,
        ...     sector=NAICSSector.FABRICATED_METAL,
        ...     year=2025
        ... )
    """

    org_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Reporting organization identifier"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year for the calculation"
    )
    revenue: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Total revenue from products sold to downstream processors"
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Currency of the revenue amount"
    )
    sector: NAICSSector = Field(
        ...,
        description="NAICS sector of the downstream processing industry"
    )
    year: int = Field(
        ..., ge=2015, le=2030,
        description="Year of the revenue data for CPI deflation"
    )

    model_config = ConfigDict(frozen=True)

    @validator("revenue")
    def validate_revenue(cls, v: Decimal) -> Decimal:
        """Validate revenue is positive."""
        if v <= Decimal("0"):
            raise ValueError(f"Revenue must be positive, got {v}")
        return v


class ProcessingChainInput(BaseModel):
    """Input model for multi-step processing chain calculation.

    Applies a pre-defined processing chain's combined emission factor
    to the product quantity, accounting for all transformation steps.

    Example:
        >>> chain_input = ProcessingChainInput(
        ...     product_id="PROD-2026-001",
        ...     chain_type=ProcessingChainType.METALS_AUTOMOTIVE,
        ...     quantity=Decimal("200"),
        ...     customer_country=GridRegion.DE
        ... )
    """

    product_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Product identifier for the intermediate product"
    )
    chain_type: ProcessingChainType = Field(
        ...,
        description="Pre-defined multi-step processing chain type"
    )
    quantity: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Quantity of product entering the processing chain (tonnes)"
    )
    customer_country: GridRegion = Field(
        default=GridRegion.GLOBAL,
        description="Country where the processing chain is located"
    )

    model_config = ConfigDict(frozen=True)

    @validator("quantity")
    def validate_quantity(cls, v: Decimal) -> Decimal:
        """Validate quantity is positive."""
        if v <= Decimal("0"):
            raise ValueError(f"Quantity must be positive, got {v}")
        return v


class ProductBreakdown(BaseModel):
    """Emissions breakdown for a single product within a calculation result.

    Provides detailed per-product emissions including the emission factor
    used, calculation method, and data quality score.
    """

    product_id: str = Field(
        ..., description="Product identifier"
    )
    category: IntermediateProductCategory = Field(
        ..., description="Product category"
    )
    processing_type: ProcessingType = Field(
        ..., description="Processing type applied"
    )
    quantity: Decimal = Field(
        ..., description="Quantity processed"
    )
    emissions_kg: Decimal = Field(
        ..., description="Emissions in kgCO2e"
    )
    ef_used: Decimal = Field(
        ..., description="Emission factor applied (kgCO2e per unit)"
    )
    method: CalculationMethod = Field(
        ..., description="Calculation method used"
    )
    dqi: Decimal = Field(
        ..., description="Data quality indicator score (1-5)"
    )

    model_config = ConfigDict(frozen=True)


class CalculationResult(BaseModel):
    """Complete calculation result with provenance and quality metadata.

    The primary output model from any calculation method, containing
    total emissions, per-product breakdowns, and audit trail data.

    Example:
        >>> result = CalculationResult(
        ...     calc_id="CALC-2026-001",
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ...     method=CalculationMethod.AVERAGE_DATA,
        ...     total_emissions_kg=Decimal("140000"),
        ...     total_emissions_tco2e=Decimal("140.000"),
        ...     product_breakdowns=[breakdown1, breakdown2],
        ...     dqi_score=Decimal("3.8"),
        ...     uncertainty=Decimal("0.30"),
        ...     provenance_hash="abc123...",
        ...     timestamp="2026-02-28T12:00:00Z"
        ... )
    """

    calc_id: str = Field(
        ..., description="Unique calculation identifier"
    )
    org_id: str = Field(
        ..., description="Reporting organization identifier"
    )
    reporting_year: int = Field(
        ..., description="Reporting year"
    )
    method: CalculationMethod = Field(
        ..., description="Calculation method used"
    )
    total_emissions_kg: Decimal = Field(
        ..., description="Total emissions in kgCO2e"
    )
    total_emissions_tco2e: Decimal = Field(
        ..., description="Total emissions in tCO2e (tonnes)"
    )
    product_breakdowns: List[ProductBreakdown] = Field(
        default_factory=list,
        description="Per-product emissions breakdowns"
    )
    dqi_score: Decimal = Field(
        ..., description="Weighted data quality indicator score (1-5)"
    )
    uncertainty: Decimal = Field(
        ..., description="Uncertainty half-width (fraction of total)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash for audit trail"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )

    model_config = ConfigDict(frozen=True)


class AggregationResult(BaseModel):
    """Aggregated emissions across multiple dimensions.

    Provides rollup views by product category, calculation method,
    and customer country for reporting and analysis.
    """

    period: str = Field(
        ..., description="Reporting period (e.g., '2025', '2025-Q3')"
    )
    total_tco2e: Decimal = Field(
        ..., description="Total aggregated emissions in tCO2e"
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by product category (tCO2e)"
    )
    by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by calculation method (tCO2e)"
    )
    by_country: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by customer country (tCO2e)"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceResult(BaseModel):
    """Result of a compliance check against a single regulatory framework.

    Contains the check status, rule pass/fail counts, and specific
    findings with recommendations.
    """

    framework: ComplianceFramework = Field(
        ..., description="Framework that was checked"
    )
    status: ComplianceStatus = Field(
        ..., description="Overall compliance status"
    )
    rules_checked: int = Field(
        ..., ge=0, description="Total number of rules evaluated"
    )
    rules_passed: int = Field(
        ..., ge=0, description="Number of rules passed"
    )
    rules_failed: int = Field(
        ..., ge=0, description="Number of rules failed"
    )
    findings: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Specific findings (rule_id, severity, message)"
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(BaseModel):
    """Single record in the SHA-256 provenance chain.

    Each pipeline stage produces a provenance record linking the
    input hash to the output hash with a cumulative chain hash.
    """

    stage: ProvenanceStage = Field(
        ..., description="Pipeline stage that produced this record"
    )
    input_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the stage input"
    )
    output_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the stage output"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of record creation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-specific metadata (method, EF source, etc.)"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityScore(BaseModel):
    """Data quality assessment result across five GHG Protocol dimensions.

    Each dimension is scored 1-5 (5 = best). The overall score is a
    weighted average of the five dimensions.
    """

    reliability: int = Field(
        ..., ge=1, le=5,
        description="Source reliability score (1-5)"
    )
    completeness: int = Field(
        ..., ge=1, le=5,
        description="Data completeness score (1-5)"
    )
    temporal: int = Field(
        ..., ge=1, le=5,
        description="Temporal correlation score (1-5)"
    )
    geographical: int = Field(
        ..., ge=1, le=5,
        description="Geographical correlation score (1-5)"
    )
    technological: int = Field(
        ..., ge=1, le=5,
        description="Technological correlation score (1-5)"
    )
    overall: Decimal = Field(
        ..., ge=Decimal("1"), le=Decimal("5"),
        description="Weighted overall DQI score (1-5)"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """Result of uncertainty quantification for an emissions estimate.

    Provides the mean estimate with confidence interval bounds
    and the method/parameters used.
    """

    method: UncertaintyMethod = Field(
        ..., description="Uncertainty quantification method used"
    )
    mean: Decimal = Field(
        ..., description="Mean emissions estimate (kgCO2e)"
    )
    std_dev: Decimal = Field(
        ..., ge=Decimal("0"),
        description="Standard deviation (kgCO2e)"
    )
    ci_lower: Decimal = Field(
        ..., description="95% confidence interval lower bound (kgCO2e)"
    )
    ci_upper: Decimal = Field(
        ..., description="95% confidence interval upper bound (kgCO2e)"
    )
    iterations: int = Field(
        ..., ge=0,
        description="Number of iterations (Monte Carlo/bootstrap) or 0 for analytical"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_processing_ef(category: IntermediateProductCategory) -> Decimal:
    """Get the average-data processing emission factor for a product category.

    Args:
        category: Intermediate product category.

    Returns:
        Emission factor in kgCO2e per tonne of product processed.

    Raises:
        ValueError: If category is not found in PROCESSING_EMISSION_FACTORS.

    Example:
        >>> get_processing_ef(IntermediateProductCategory.METALS_FERROUS)
        Decimal('280')
    """
    ef = PROCESSING_EMISSION_FACTORS.get(category.value)
    if ef is None:
        raise ValueError(
            f"No processing emission factor found for category '{category.value}'. "
            f"Available categories: {sorted(PROCESSING_EMISSION_FACTORS.keys())}"
        )
    return ef


def get_energy_intensity(processing_type: ProcessingType) -> Decimal:
    """Get the energy intensity factor for a processing type.

    Args:
        processing_type: Type of downstream processing operation.

    Returns:
        Energy intensity in kWh per tonne of product processed.

    Raises:
        ValueError: If processing type is not found.

    Example:
        >>> get_energy_intensity(ProcessingType.MACHINING)
        Decimal('280')
    """
    intensity = ENERGY_INTENSITY_FACTORS.get(processing_type.value)
    if intensity is None:
        raise ValueError(
            f"No energy intensity factor found for processing type '{processing_type.value}'. "
            f"Available types: {sorted(ENERGY_INTENSITY_FACTORS.keys())}"
        )
    return intensity


def get_grid_ef(region: GridRegion) -> Decimal:
    """Get the grid electricity emission factor for a region.

    Falls back to GLOBAL if the specific region is not found.

    Args:
        region: Grid region for electricity emission factor lookup.

    Returns:
        Grid emission factor in kgCO2e per kWh.

    Example:
        >>> get_grid_ef(GridRegion.DE)
        Decimal('0.348')
        >>> get_grid_ef(GridRegion.GLOBAL)
        Decimal('0.475')
    """
    ef = GRID_EMISSION_FACTORS.get(region.value)
    if ef is None:
        return GRID_EMISSION_FACTORS[GridRegion.GLOBAL.value]
    return ef


def get_fuel_ef(fuel_type: FuelType) -> Decimal:
    """Get the combustion emission factor for a fuel type.

    Args:
        fuel_type: Fuel type for emission factor lookup.

    Returns:
        Fuel emission factor in kgCO2e per kWh thermal.

    Raises:
        ValueError: If fuel type is not found.

    Example:
        >>> get_fuel_ef(FuelType.NATURAL_GAS)
        Decimal('2.024')
    """
    ef = FUEL_EMISSION_FACTORS.get(fuel_type.value)
    if ef is None:
        raise ValueError(
            f"No fuel emission factor found for fuel type '{fuel_type.value}'. "
            f"Available types: {sorted(FUEL_EMISSION_FACTORS.keys())}"
        )
    return ef


def get_eeio_factor(sector: NAICSSector) -> Tuple[Decimal, Decimal]:
    """Get the EEIO emission factor and margin for a NAICS sector.

    Args:
        sector: NAICS sector code for downstream processing.

    Returns:
        Tuple of (ef_kgco2e_per_usd, margin_fraction).

    Raises:
        ValueError: If sector is not found in EEIO_SECTOR_FACTORS.

    Example:
        >>> ef, margin = get_eeio_factor(NAICSSector.FABRICATED_METAL)
        >>> ef
        Decimal('0.540')
        >>> margin
        Decimal('0.20')
    """
    entry = EEIO_SECTOR_FACTORS.get(sector.value)
    if entry is None:
        raise ValueError(
            f"No EEIO factor found for sector '{sector.value}'. "
            f"Available sectors: {sorted(EEIO_SECTOR_FACTORS.keys())}"
        )
    return entry["ef"], entry["margin"]


def get_processing_chain(chain_type: ProcessingChainType) -> Dict[str, Any]:
    """Get the full processing chain definition for a chain type.

    Args:
        chain_type: Pre-defined processing chain type.

    Returns:
        Dictionary with 'steps' (list), 'combined_ef' (Decimal), 'description' (str).

    Raises:
        ValueError: If chain type is not found.

    Example:
        >>> chain = get_processing_chain(ProcessingChainType.METALS_AUTOMOTIVE)
        >>> chain["combined_ef"]
        Decimal('920')
        >>> chain["steps"]
        ['stamping', 'welding', 'heat_treatment', 'coating', 'assembly']
    """
    chain = PROCESSING_CHAINS.get(chain_type.value)
    if chain is None:
        raise ValueError(
            f"No processing chain found for type '{chain_type.value}'. "
            f"Available chains: {sorted(PROCESSING_CHAINS.keys())}"
        )
    return chain


def get_currency_rate(currency: Currency) -> Decimal:
    """Get the USD conversion rate for a currency.

    Args:
        currency: Source currency code.

    Returns:
        Conversion rate to USD (multiply source amount by this rate).

    Raises:
        ValueError: If currency is not found.

    Example:
        >>> get_currency_rate(Currency.EUR)
        Decimal('1.0850')
    """
    rate = CURRENCIES.get(currency.value)
    if rate is None:
        raise ValueError(
            f"No conversion rate found for currency '{currency.value}'. "
            f"Available currencies: {sorted(CURRENCIES.keys())}"
        )
    return rate


def get_cpi_deflator(year: int) -> Decimal:
    """Get the CPI deflator for a given year (base year 2021 = 1.0).

    Used to convert nominal revenue to real (2021 base) USD for
    consistent EEIO factor application.

    Args:
        year: Year of the revenue data.

    Returns:
        CPI deflator value.

    Raises:
        ValueError: If year is not available in CPI_DEFLATORS.

    Example:
        >>> get_cpi_deflator(2024)
        Decimal('1.1490')
    """
    deflator = CPI_DEFLATORS.get(year)
    if deflator is None:
        raise ValueError(
            f"CPI deflator not available for year {year}. "
            f"Available years: {sorted(CPI_DEFLATORS.keys())}"
        )
    return deflator


def get_dqi_score(dimension: DQIDimension, level: DataQualityTier) -> int:
    """Get the DQI score for a dimension at a given data quality tier.

    Args:
        dimension: DQI dimension to score.
        level: Data quality tier.

    Returns:
        Integer score (1-5).

    Raises:
        ValueError: If dimension or level is not found.

    Example:
        >>> get_dqi_score(DQIDimension.RELIABILITY, DataQualityTier.TIER_1)
        5
    """
    dim_scores = DQI_SCORING.get(dimension.value)
    if dim_scores is None:
        raise ValueError(
            f"No DQI scoring found for dimension '{dimension.value}'. "
            f"Available dimensions: {sorted(DQI_SCORING.keys())}"
        )
    score = dim_scores.get(level.value)
    if score is None:
        raise ValueError(
            f"No DQI score found for tier '{level.value}' in dimension '{dimension.value}'. "
            f"Available tiers: {sorted(dim_scores.keys())}"
        )
    return score


def get_uncertainty_range(method: CalculationMethod) -> Tuple[Decimal, Decimal, Decimal]:
    """Get the uncertainty range (min, default, max) for a calculation method.

    Args:
        method: Calculation method.

    Returns:
        Tuple of (min_uncertainty, default_uncertainty, max_uncertainty) as fractions.

    Raises:
        ValueError: If method is not found in UNCERTAINTY_RANGES.

    Example:
        >>> get_uncertainty_range(CalculationMethod.AVERAGE_DATA)
        (Decimal('0.20'), Decimal('0.30'), Decimal('0.50'))
    """
    entry = UNCERTAINTY_RANGES.get(method.value)
    if entry is None:
        raise ValueError(
            f"No uncertainty range found for method '{method.value}'. "
            f"Available methods: {sorted(UNCERTAINTY_RANGES.keys())}"
        )
    return entry["min"], entry["default"], entry["max"]


def get_applicable_processing_types(
    category: IntermediateProductCategory,
) -> List[str]:
    """Get the list of applicable processing types for a product category.

    Args:
        category: Intermediate product category.

    Returns:
        List of processing type value strings applicable to the category.

    Raises:
        ValueError: If category is not found in PRODUCT_CATEGORY_PROCESSING_MAP.

    Example:
        >>> types = get_applicable_processing_types(
        ...     IntermediateProductCategory.METALS_FERROUS
        ... )
        >>> "machining" in types
        True
    """
    types = PRODUCT_CATEGORY_PROCESSING_MAP.get(category.value)
    if types is None:
        raise ValueError(
            f"No processing type mapping found for category '{category.value}'. "
            f"Available categories: {sorted(PRODUCT_CATEGORY_PROCESSING_MAP.keys())}"
        )
    return list(types)


def get_energy_range(
    processing_type: ProcessingType,
) -> Tuple[Decimal, Decimal, Decimal]:
    """Get the energy intensity range (low, mid, high) for a processing type.

    Args:
        processing_type: Processing type to look up.

    Returns:
        Tuple of (low_kwh_per_tonne, mid_kwh_per_tonne, high_kwh_per_tonne).

    Raises:
        ValueError: If processing type is not found.

    Example:
        >>> low, mid, high = get_energy_range(ProcessingType.MACHINING)
        >>> low
        Decimal('180')
        >>> mid
        Decimal('280')
        >>> high
        Decimal('420')
    """
    entry = ENERGY_INTENSITY_RANGES.get(processing_type.value)
    if entry is None:
        raise ValueError(
            f"No energy intensity range found for processing type '{processing_type.value}'. "
            f"Available types: {sorted(ENERGY_INTENSITY_RANGES.keys())}"
        )
    return entry["low"], entry["mid"], entry["high"]


def get_dc_rule(rule_id: str) -> Dict[str, str]:
    """Get a double-counting prevention rule by its ID.

    Args:
        rule_id: Rule identifier (e.g., 'DC-PSP-001').

    Returns:
        Dictionary with 'title', 'description', 'overlapping_category', 'resolution'.

    Raises:
        ValueError: If rule_id is not found.

    Example:
        >>> rule = get_dc_rule("DC-PSP-001")
        >>> rule["title"]
        'Exclude Scope 1 / Scope 2 of reporter'
    """
    rule = DC_RULES.get(rule_id)
    if rule is None:
        raise ValueError(
            f"No double-counting rule found for ID '{rule_id}'. "
            f"Available rules: {sorted(DC_RULES.keys())}"
        )
    return dict(rule)


def get_framework_rules(framework: ComplianceFramework) -> Dict[str, Any]:
    """Get the compliance rule definition for a regulatory framework.

    Args:
        framework: Compliance framework to look up.

    Returns:
        Dictionary with 'rule_count', 'required_disclosures', 'description'.

    Raises:
        ValueError: If framework is not found.

    Example:
        >>> rules = get_framework_rules(ComplianceFramework.GHG_PROTOCOL)
        >>> rules["rule_count"]
        12
    """
    entry = COMPLIANCE_FRAMEWORK_RULES.get(framework.value)
    if entry is None:
        raise ValueError(
            f"No compliance rules found for framework '{framework.value}'. "
            f"Available frameworks: {sorted(COMPLIANCE_FRAMEWORK_RULES.keys())}"
        )
    return dict(entry)


def calculate_provenance_hash(*inputs: Any) -> str:
    """Calculate SHA-256 provenance hash from variable inputs.

    Supports Pydantic models (serialized to sorted JSON), Decimal values,
    and any other stringifiable objects. Used throughout the pipeline
    to build the provenance chain.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).

    Example:
        >>> h = calculate_provenance_hash("PROD-001", Decimal("500"))
        >>> len(h)
        64
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, BaseModel):
            hash_input += json.dumps(
                inp.model_dump(mode="json"), sort_keys=True, default=str
            )
        elif isinstance(inp, Decimal):
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
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
    "IntermediateProductCategory",
    "ProcessingType",
    "CalculationMethod",
    "EnergyType",
    "FuelType",
    "GridRegion",
    "NAICSSector",
    "Currency",
    "ProcessingChainType",
    "DataQualityTier",
    "DQIDimension",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "ProvenanceStage",
    "AllocationMethod",
    "UncertaintyMethod",
    "BatchStatus",
    "AuditAction",
    "ProductUnit",

    # Constant tables
    "PROCESSING_EMISSION_FACTORS",
    "ENERGY_INTENSITY_FACTORS",
    "GRID_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "EEIO_SECTOR_FACTORS",
    "PROCESSING_CHAINS",
    "CURRENCIES",
    "CPI_DEFLATORS",
    "DQI_SCORING",
    "UNCERTAINTY_RANGES",
    "PRODUCT_CATEGORY_PROCESSING_MAP",
    "ENERGY_INTENSITY_RANGES",
    "DC_RULES",
    "COMPLIANCE_FRAMEWORK_RULES",

    # Input models
    "IntermediateProductInput",
    "SiteSpecificInput",
    "AverageDataInput",
    "SpendBasedInput",
    "ProcessingChainInput",

    # Result models
    "CalculationResult",
    "ProductBreakdown",
    "AggregationResult",
    "ComplianceResult",
    "ProvenanceRecord",
    "DataQualityScore",
    "UncertaintyResult",

    # Helper functions
    "get_processing_ef",
    "get_energy_intensity",
    "get_grid_ef",
    "get_fuel_ef",
    "get_eeio_factor",
    "get_processing_chain",
    "get_currency_rate",
    "get_cpi_deflator",
    "get_dqi_score",
    "get_uncertainty_range",
    "get_applicable_processing_types",
    "get_energy_range",
    "get_dc_rule",
    "get_framework_rules",
    "calculate_provenance_hash",
]
