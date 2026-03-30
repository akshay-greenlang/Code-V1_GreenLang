# -*- coding: utf-8 -*-
"""
Purchased Goods & Services Agent Data Models - AGENT-MRV-014

Pydantic v2 data models for the Purchased Goods & Services Agent SDK
covering GHG Protocol Scope 3 Category 1 emissions from upstream
production of purchased goods and services including:

- Four calculation methods: spend-based (EEIO), average-data (physical
  EFs), supplier-specific (EPD/PCF/CDP), and hybrid (multi-method)
- Spend classification across NAICS, NACE, ISIC, and UNSPSC systems
  with cross-mapping and confidence scoring
- EEIO emission factor lookup from EPA USEEIO (1,016 commodities),
  EXIOBASE (9,800+ factors), WIOD, and GTAP databases
- Physical emission factor lookup for 30+ materials from ecoinvent,
  GaBi, DEFRA, ICE, World Steel, IAI, PlasticsEurope, and CEPI
- Supplier-specific data integration from EPDs, PCFs, CDP Supply
  Chain, EcoVadis, PACT Network, and direct disclosure
- Currency conversion (20 currencies), inflation deflation (CPI base
  year adjustment), and margin removal (producer vs purchaser price)
- 5-dimension data quality indicator (DQI) scoring per GHG Protocol
  Scope 3 Standard Chapter 7 (temporal, geographical, technological,
  completeness, reliability) on a 1-5 scale
- 8-level emission factor selection hierarchy from supplier EPD
  (highest) to global average EEIO (lowest)
- Hot-spot analysis with Pareto ranking and materiality quadrant
  classification (prioritize, monitor, improve data, low priority)
- Double-counting prevention against Categories 2-8 with boundary
  enforcement and overlap detection
- Compliance checking against 7 regulatory frameworks: GHG Protocol,
  CSRD/ESRS E1, California SB 253, CDP, SBTi, GRI 305, ISO 14064
- Uncertainty quantification via Monte Carlo, analytical propagation,
  pedigree matrix, and expert judgment methods
- SHA-256 provenance chain for complete audit trails
- Batch multi-period processing and multi-facility aggregation
- Export in JSON, CSV, and Excel formats

Enumerations (20):
    - CalculationMethod, SpendClassificationSystem, EEIODatabase,
      PhysicalEFSource, SupplierDataSource, AllocationMethod,
      MaterialCategory, CurrencyCode, DQIDimension, DQIScore,
      UncertaintyMethod, ComplianceFramework, ComplianceStatus,
      PipelineStage, ReportFormat, BatchStatus, GWPSource,
      EmissionGas, ProcurementType, CoverageLevel

Constants (9):
    - EEIO_EMISSION_FACTORS: 50+ NAICS sectors with kgCO2e/USD
    - PHYSICAL_EMISSION_FACTORS: 30+ materials with kgCO2e/kg
    - CURRENCY_EXCHANGE_RATES: 20 currencies to USD
    - INDUSTRY_MARGIN_PERCENTAGES: 20 NAICS 2-digit sectors
    - DQI_SCORE_VALUES: 5 quality scores (1.0-5.0)
    - DQI_QUALITY_TIERS: 5 quality tier labels with ranges
    - UNCERTAINTY_RANGES: 4 calculation methods with min/max %
    - COVERAGE_THRESHOLDS: 5 coverage levels with percentages
    - EF_HIERARCHY_PRIORITY: 8 data source priorities

Data Models (25):
    - ProcurementItem, SpendRecord, PhysicalRecord, SupplierRecord,
      SpendBasedResult, AverageDataResult, SupplierSpecificResult,
      HybridResult, EEIOFactor, PhysicalEF, SupplierEF,
      DQIAssessment, MaterialityItem, CoverageReport,
      ComplianceRequirement, ComplianceCheckResult,
      CalculationRequest, CalculationResult, BatchRequest,
      BatchResult, ExportRequest, AggregationResult,
      HotSpotAnalysis, CategoryBoundaryCheck, PipelineContext

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-S3-001)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, field_validator

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module-level Constants
# ---------------------------------------------------------------------------

#: Agent identifier for registry integration.
AGENT_ID: str = "GL-MRV-S3-001"

#: Agent component identifier.
AGENT_COMPONENT: str = "AGENT-MRV-014"

#: Service version string.
VERSION: str = "1.0.0"

#: Database table prefix for all PGS tables.
TABLE_PREFIX: str = "gl_pgs_"

#: Maximum number of procurement items per calculation request.
MAX_PROCUREMENT_ITEMS: int = 100_000

#: Maximum number of periods in a batch request.
MAX_BATCH_PERIODS: int = 120

#: Maximum number of facilities per aggregation.
MAX_FACILITIES: int = 50_000

#: Maximum number of suppliers per request.
MAX_SUPPLIERS: int = 10_000

#: Maximum number of frameworks per compliance check.
MAX_FRAMEWORKS: int = 20

#: Maximum number of compliance requirements per framework.
MAX_REQUIREMENTS_PER_FRAMEWORK: int = 200

#: Maximum number of hot-spot items per analysis.
MAX_HOTSPOT_ITEMS: int = 5_000

#: Default confidence level for uncertainty quantification.
DEFAULT_CONFIDENCE_LEVEL: Decimal = Decimal("95.0")

#: Positive infinity sentinel for Decimal comparisons.
DECIMAL_INF: Decimal = Decimal("Infinity")

#: Number of decimal places for Decimal quantization.
DECIMAL_PLACES: int = 8

#: Decimal zero constant for arithmetic operations.
ZERO: Decimal = Decimal("0")

#: Decimal one constant.
ONE: Decimal = Decimal("1")

#: Decimal one hundred constant for percentage calculations.
ONE_HUNDRED: Decimal = Decimal("100")

#: Decimal one thousand constant for unit conversions.
ONE_THOUSAND: Decimal = Decimal("1000")

# =============================================================================
# Enumerations (20)
# =============================================================================

class CalculationMethod(str, Enum):
    """GHG Protocol Scope 3 Category 1 calculation methods.

    The GHG Protocol Technical Guidance defines four methods for
    Category 1 emissions, listed from most to least accurate.
    Organizations should use the most accurate method for which
    data is available, and may combine methods via the HYBRID
    approach.

    SUPPLIER_SPECIFIC: Uses primary data from suppliers on
        cradle-to-gate GHG emissions of their products. Highest
        accuracy (+/- 10-30%). Requires EPDs, PCFs, or CDP data.
    HYBRID: Combines all three methods, using the most accurate
        data available for each supplier or procurement category.
        High accuracy (+/- 20-50%).
    AVERAGE_DATA: Uses physical quantities multiplied by industry-
        average cradle-to-gate emission factors from LCA databases.
        Medium accuracy (+/- 30-60%).
    SPEND_BASED: Estimates emissions by multiplying economic value
        of each procurement category by EEIO emission factors.
        Lowest accuracy (+/- 50-100%) but broadest coverage.
    """

    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"

class SpendClassificationSystem(str, Enum):
    """Industry classification systems for spend categorization.

    The agent supports four classification systems with cross-mapping
    capabilities. Each system has different granularity and geographic
    coverage. NAICS is the primary key for EPA USEEIO mapping, NACE
    for EXIOBASE mapping, ISIC as bridge standard, and UNSPSC for
    product-level classification.

    NAICS: North American Industry Classification System (2022).
        6-digit codes, 20 sectors, 1,057 industries. Primary use
        for EPA USEEIO emission factor mapping.
    NACE: Statistical Classification of Economic Activities in the
        European Community, Rev 2.1. Letter + 4-digit codes, 21
        sections, 615 activities. Primary for EXIOBASE and CSRD.
    ISIC: International Standard Industrial Classification of All
        Economic Activities, Rev 4.1. UN global standard used as
        bridge between NAICS and NACE systems.
    UNSPSC: United Nations Standard Products and Services Code,
        v28.0. 8-digit codes, 55 segments, 60,000+ commodities.
        Most granular, used for product-level classification.
    """

    NAICS = "naics"
    NACE = "nace"
    ISIC = "isic"
    UNSPSC = "unspsc"

class EEIODatabase(str, Enum):
    """Environmentally-Extended Input-Output databases for spend-based EFs.

    EEIO models link economic input-output tables with environmental
    satellite accounts to provide emission factors per unit of
    economic output. Each database covers different regions, sectors,
    and base years.

    EPA_USEEIO: US EPA Environmentally-Extended Input-Output Model
        v1.2/v1.3. 1,016 commodities by NAICS-6, base year 2019
        in 2021 USD. Most granular for US-based procurement.
    EXIOBASE: EXIOBASE 3.8 multi-regional IO database. 163 product
        groups x 49 regions (~9,800 factors), base year in EUR.
        Best for EU and global multi-regional analysis.
    WIOD: World Input-Output Database (2016 release). 43 countries,
        56 sectors. Older and less granular but broad coverage.
    GTAP: Global Trade Analysis Project version 11. 141 regions,
        65 sectors. Requires subscription; strong for trade-linked
        supply chain analysis.
    DEFRA_EEIO: UK DEFRA/DESNZ Environmentally-Extended Input-Output
        factors. Sector-level factors in GBP. Used for UK-specific
        spend-based calculations.
    """

    EPA_USEEIO = "epa_useeio"
    EXIOBASE = "exiobase"
    WIOD = "wiod"
    GTAP = "gtap"
    DEFRA_EEIO = "defra_eeio"

class PhysicalEFSource(str, Enum):
    """Sources of physical (quantity-based) emission factors.

    Physical emission factors express cradle-to-gate emissions per
    unit of physical product (e.g. kgCO2e per kg of steel). These
    are used in the average-data calculation method and are more
    accurate than EEIO factors but require physical quantity data.

    ECOINVENT: ecoinvent v3.11 LCA database. 21,000+ unit process
        datasets with global and regional coverage.
    GABI: GaBi/Sphera LCA database. 15,000+ datasets with strong
        coverage for chemicals and automotive supply chains.
    DEFRA: UK DEFRA/DESNZ emission conversion factors. 200+
        materials with annual updates. Free and widely used.
    ICE: Inventory of Carbon and Energy v3.0. 200+ construction
        materials. University of Bath database.
    WORLD_STEEL: World Steel Association environmental data.
        Authoritative source for steel emission factors.
    PLASTICS_EUROPE: PlasticsEurope eco-profile data. Industry-
        average LCA data for major plastic resins.
    IAI: International Aluminium Institute. Primary and secondary
        aluminium production emission factors.
    CEPI: Confederation of European Paper Industries. Pulp and
        paper emission factors.
    CUSTOM: User-defined or custom emission factor source.
    """

    ECOINVENT = "ecoinvent"
    GABI = "gabi"
    DEFRA = "defra"
    ICE = "ice"
    WORLD_STEEL = "world_steel"
    PLASTICS_EUROPE = "plastics_europe"
    IAI = "iai"
    CEPI = "cepi"
    CUSTOM = "custom"

class SupplierDataSource(str, Enum):
    """Sources of supplier-specific emission data.

    Supplier-specific data represents the highest quality emission
    factors available, derived from primary data provided by
    suppliers about their own operations and products.

    CDP_SUPPLY_CHAIN: CDP Supply Chain Program data. 35,000+
        reporting suppliers with Scope 1, 2, and 3 data.
    ECOVADIS: EcoVadis sustainability ratings. 130,000+ rated
        companies with environmental performance scores.
    EPD: Environmental Product Declarations per ISO 14025.
        Third-party verified cradle-to-gate LCA results.
    SUSTAINABILITY_REPORT: Supplier sustainability or annual
        reports with self-reported emission data.
    PACT_NETWORK: WBCSD Partnership for Carbon Transparency.
        Product-level PCF exchange network.
    DIRECT_MEASUREMENT: Direct supplier disclosure of measured
        emissions data for specific products or facilities.
    CUSTOM: User-defined or custom supplier data source.
    """

    CDP_SUPPLY_CHAIN = "cdp_supply_chain"
    ECOVADIS = "ecovadis"
    EPD = "epd"
    SUSTAINABILITY_REPORT = "sustainability_report"
    PACT_NETWORK = "pact_network"
    DIRECT_MEASUREMENT = "direct_measurement"
    CUSTOM = "custom"

class AllocationMethod(str, Enum):
    """Allocation methods for supplier facility-level data.

    When supplier-specific emission data is available only at the
    facility level rather than the product level, allocation methods
    are used to attribute a portion of total facility emissions to
    the reporting company's purchases.

    REVENUE_BASED: Allocate by ratio of revenue from customer to
        total facility revenue. Best for services and diverse
        product mixes.
    MASS_BASED: Allocate by ratio of mass purchased to total
        facility output mass. Best for commodities and bulk
        materials.
    ECONOMIC: Allocate by ratio of economic value of products
        purchased to total economic output. Best for multi-product
        facilities with varying product values.
    PHYSICAL_UNIT: Allocate by ratio of units purchased to total
        facility unit output. Best for single-product production
        lines.
    EQUAL: Equal allocation across all customers. Simplest method,
        used when no differentiation data is available.
    """

    REVENUE_BASED = "revenue_based"
    MASS_BASED = "mass_based"
    ECONOMIC = "economic"
    PHYSICAL_UNIT = "physical_unit"
    EQUAL = "equal"

class MaterialCategory(str, Enum):
    """Material categories for physical emission factor classification.

    Covers the 20 major procurement material and service categories
    that organizations typically purchase. Each category maps to one
    or more physical emission factors in the PHYSICAL_EMISSION_FACTORS
    constant table.

    RAW_METALS: Primary and secondary metals including steel,
        aluminium, copper, zinc, lead, and other base metals.
    PLASTICS: Thermoplastic and thermoset resins including HDPE,
        LDPE, PP, PET, PVC, PS, and engineering plastics.
    CHEMICALS: Industrial chemicals, solvents, adhesives, coatings,
        and specialty chemical products.
    PAPER: Paper and pulp products including kraft, recycled paper,
        corrugated cardboard, and tissue products.
    TEXTILES: Natural and synthetic fibers including cotton, polyester,
        nylon, and wool for textile manufacturing.
    ELECTRONICS: Electronic components, semiconductors, PCBs, and
        assembled electronic products.
    FOOD: Agricultural products, processed food, beverages, and food
        ingredients for manufacturing or consumption.
    PACKAGING: Packaging materials including corrugated, plastic film,
        glass containers, and metal cans.
    CONSTRUCTION: Construction materials including cement, concrete,
        bricks, glass, insulation, and aggregates.
    MACHINERY: Industrial machinery, tools, equipment, and mechanical
        components (below capitalization threshold).
    FUELS: Fuels and lubricants not for own combustion (resale or
        feedstock use only; own-use fuels are Category 3).
    MINERALS: Non-metallic minerals including sand, gravel, clay,
        limestone, and mineral-based products.
    GLASS: Flat glass, container glass, and specialty glass products.
    RUBBER: Natural and synthetic rubber products including tires,
        seals, and industrial rubber goods.
    WOOD: Timber products including sawn lumber, plywood, MDF,
        glulam, and other engineered wood products.
    AGRICULTURE: Agricultural inputs including fertilizers, seeds,
        animal feed, and crop protection products.
    SERVICES_IT: Information technology services including cloud,
        software licenses, and IT consulting.
    SERVICES_PROFESSIONAL: Professional services including legal,
        accounting, management consulting, and engineering.
    SERVICES_FINANCIAL: Financial services including banking,
        insurance, and investment management.
    OTHER: Materials and services not classified in the above
        categories.
    """

    RAW_METALS = "raw_metals"
    PLASTICS = "plastics"
    CHEMICALS = "chemicals"
    PAPER = "paper"
    TEXTILES = "textiles"
    ELECTRONICS = "electronics"
    FOOD = "food"
    PACKAGING = "packaging"
    CONSTRUCTION = "construction"
    MACHINERY = "machinery"
    FUELS = "fuels"
    MINERALS = "minerals"
    GLASS = "glass"
    RUBBER = "rubber"
    WOOD = "wood"
    AGRICULTURE = "agriculture"
    SERVICES_IT = "services_it"
    SERVICES_PROFESSIONAL = "services_professional"
    SERVICES_FINANCIAL = "services_financial"
    OTHER = "other"

class CurrencyCode(str, Enum):
    """ISO 4217 currency codes supported for spend-based calculations.

    The agent supports 20 major currencies for procurement spend
    data. All amounts are converted to USD (base currency) using
    annual average exchange rates before applying EEIO factors.

    USD: United States Dollar (base currency).
    EUR: Euro (Eurozone).
    GBP: British Pound Sterling.
    JPY: Japanese Yen.
    CNY: Chinese Yuan Renminbi.
    INR: Indian Rupee.
    CAD: Canadian Dollar.
    AUD: Australian Dollar.
    CHF: Swiss Franc.
    KRW: South Korean Won.
    BRL: Brazilian Real.
    MXN: Mexican Peso.
    SGD: Singapore Dollar.
    HKD: Hong Kong Dollar.
    SEK: Swedish Krona.
    NOK: Norwegian Krone.
    DKK: Danish Krone.
    PLN: Polish Zloty.
    CZK: Czech Koruna.
    ZAR: South African Rand.
    """

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    KRW = "KRW"
    BRL = "BRL"
    MXN = "MXN"
    SGD = "SGD"
    HKD = "HKD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    PLN = "PLN"
    CZK = "CZK"
    ZAR = "ZAR"

class DQIDimension(str, Enum):
    """Data quality indicator dimensions per GHG Protocol Scope 3.

    Per GHG Protocol Scope 3 Standard Chapter 7, five data quality
    indicators are assessed on a 1-5 scale. Lower scores indicate
    higher quality. The composite DQI is the arithmetic mean of
    all five dimension scores.

    TEMPORAL: Timeliness of the data relative to the reporting year.
        Score 1 = data from reporting year, Score 5 = older than
        10 years.
    GEOGRAPHICAL: Geographic representativeness of the emission
        factor relative to the activity location. Score 1 = same
        country/region, Score 5 = global average.
    TECHNOLOGICAL: Technology representativeness of the emission
        factor relative to the actual technology used. Score 1 =
        identical technology, Score 5 = unrelated category.
    COMPLETENESS: Degree to which all relevant emission sources
        are included. Score 1 = all sources included, Score 5 =
        less than 20% covered.
    RELIABILITY: Trustworthiness of the data source and method.
        Score 1 = third-party verified, Score 5 = estimate or
        assumption.
    """

    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    TECHNOLOGICAL = "technological"
    COMPLETENESS = "completeness"
    RELIABILITY = "reliability"

class DQIScore(str, Enum):
    """Data quality score levels (1-5 scale, lower is better).

    Qualitative labels for the 1-5 numeric data quality score.
    Used in reporting and display; the numeric value is stored
    in DQI_SCORE_VALUES.

    VERY_GOOD: Score 1 -- highest quality, verified primary data.
    GOOD: Score 2 -- high quality, established databases.
    FAIR: Score 3 -- acceptable quality, industry averages.
    POOR: Score 4 -- low quality, broad estimates.
    VERY_POOR: Score 5 -- lowest quality, unverified assumptions.
    """

    VERY_GOOD = "very_good"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

class UncertaintyMethod(str, Enum):
    """Methods for quantifying uncertainty in emission calculations.

    The GHG Protocol Quantitative Uncertainty Guidance recommends
    using Monte Carlo simulation for complex inventories, and
    analytical propagation or pedigree matrix for simpler cases.

    MONTE_CARLO: Monte Carlo simulation with 10,000+ iterations.
        Most rigorous method for propagating uncertainty through
        complex calculation chains.
    ANALYTICAL: Analytical error propagation using the law of
        propagation of uncertainty (root-sum-of-squares).
    PEDIGREE_MATRIX: Pedigree matrix approach using DQI scores
        to assign uncertainty factors per the ecoinvent methodology.
    EXPERT_JUDGMENT: Expert elicitation of uncertainty ranges.
        Used when statistical data is unavailable.
    """

    MONTE_CARLO = "monte_carlo"
    ANALYTICAL = "analytical"
    PEDIGREE_MATRIX = "pedigree_matrix"
    EXPERT_JUDGMENT = "expert_judgment"

class ComplianceFramework(str, Enum):
    """Regulatory and voluntary reporting frameworks for Category 1.

    Seven frameworks that require or recommend Scope 3 Category 1
    reporting. Each has specific requirements for methodology,
    coverage, data quality, and disclosure format.

    GHG_PROTOCOL: GHG Protocol Corporate Value Chain (Scope 3)
        Standard (2011). Chapter 5 Category 1, Chapter 7 DQI,
        Chapter 9 reporting requirements.
    CSRD_ESRS: EU Corporate Sustainability Reporting Directive,
        ESRS E1. E1-6 para 44a/44b/44c Scope 3 by category.
        Mandatory for EU large companies from FY2025.
    CDP: Carbon Disclosure Project climate questionnaire.
        C6.5 Category 1 relevance and calculation.
    SBTI: Science Based Targets initiative v5.3.
        Scope 3 target required if >40% of total emissions.
        67% coverage threshold.
    SB253: California Senate Bill 253. Mandatory Scope 3 by
        FY2027 for entities with >$1B revenue. Safe harbor
        provision through 2030.
    GRI: Global Reporting Initiative Standard 305.
        Scope 3 disclosure if significant.
    ISO_14064: ISO 14064-1:2018 Category 4 indirect emissions.
        Methodology-neutral with uncertainty quantification
        and third-party verification support.
    """

    GHG_PROTOCOL = "ghg_protocol"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB253 = "sb253"
    GRI = "gri"
    ISO_14064 = "iso_14064"

class ComplianceStatus(str, Enum):
    """Result of a regulatory compliance check.

    COMPLIANT: All requirements of the regulatory framework are met.
    NON_COMPLIANT: One or more mandatory requirements are not met.
    PARTIAL: Some requirements are met but others are missing or
        incomplete.
    NOT_APPLICABLE: The framework's requirements do not apply to
        this entity or reporting period.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"

class PipelineStage(str, Enum):
    """Stages in the Purchased Goods & Services calculation pipeline.

    The pipeline processes calculation requests through ten
    sequential stages. Each stage must complete before the next
    one begins.

    INGEST: Ingest and parse procurement data records from
        the input request.
    CLASSIFY: Classify procurement items into NAICS/NACE/ISIC/
        UNSPSC codes and assign material categories.
    BOUNDARY_CHECK: Enforce category boundaries to prevent
        double-counting against Categories 2-8.
    SPEND_CALC: Execute spend-based calculations using EEIO
        factors with currency conversion and margin adjustment.
    AVGDATA_CALC: Execute average-data calculations using
        physical emission factors with unit conversion.
    SUPPLIER_CALC: Execute supplier-specific calculations
        using EPD/PCF/CDP data with allocation.
    AGGREGATE: Aggregate results across methods using hybrid
        prioritization and gap filling.
    DQI_SCORE: Score data quality across five GHG Protocol
        dimensions and compute composite DQI.
    COMPLIANCE_CHECK: Validate results against each requested
        framework's reporting requirements.
    EXPORT: Format and export results in JSON, CSV, or Excel.
    """

    INGEST = "ingest"
    CLASSIFY = "classify"
    BOUNDARY_CHECK = "boundary_check"
    SPEND_CALC = "spend_calc"
    AVGDATA_CALC = "avgdata_calc"
    SUPPLIER_CALC = "supplier_calc"
    AGGREGATE = "aggregate"
    DQI_SCORE = "dqi_score"
    COMPLIANCE_CHECK = "compliance_check"
    EXPORT = "export"

class BatchStatus(str, Enum):
    """Status of a batch calculation job.

    PENDING: Batch job has been created but not started.
    RUNNING: Batch job is actively processing periods.
    COMPLETED: All periods in the batch completed successfully.
    FAILED: All periods in the batch failed.
    PARTIAL: Some periods completed successfully while others failed.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class GWPSource(str, Enum):
    """IPCC Assessment Report source for Global Warming Potential values.

    AR4: Fourth Assessment Report (2007) - 100-year GWP values.
        CH4 = 25, N2O = 298.
    AR5: Fifth Assessment Report (2014) - 100-year GWP values.
        CH4 = 28, N2O = 265.
    AR6: Sixth Assessment Report (2021) - 100-year GWP values.
        CH4 = 27.9, N2O = 273.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"

class EmissionGas(str, Enum):
    """Greenhouse gases tracked in Scope 3 emission calculations.

    CO2: Carbon dioxide - primary gas from fossil fuel combustion
        and industrial processes in supply chains.
    CH4: Methane - emitted from agricultural supply chains, waste
        processing, and upstream fuel extraction.
    N2O: Nitrous oxide - emitted from agricultural processes,
        chemical manufacturing, and combustion.
    CO2E: Carbon dioxide equivalent - aggregate metric combining
        all gases using GWP values.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO2E = "CO2e"

class ProcurementType(str, Enum):
    """Type of procurement for Category 1 boundary classification.

    Distinguishes between goods and services procurement to apply
    appropriate emission factor databases and margin adjustments.
    Goods typically use physical EFs or goods-specific EEIO factors,
    while services use service-specific EEIO factors.

    GOODS: Physical goods purchased including raw materials,
        components, finished goods, packaging, and consumables.
    SERVICES: Services purchased including professional, IT,
        outsourced operations, and other service categories.
    MIXED: Mixed procurement containing both goods and services
        that cannot be cleanly separated.
    """

    GOODS = "goods"
    SERVICES = "services"
    MIXED = "mixed"

class CoverageLevel(str, Enum):
    """Coverage level of spend data for credibility assessment.

    Classifies the percentage of total procurement spend that has
    been covered by emission calculations. Higher coverage levels
    indicate more complete and credible Category 1 inventories.

    FULL: 100% of procurement spend is covered by calculations.
    HIGH: >= 95% of spend covered. Best practice per CDP/SBTi.
    MEDIUM: >= 90% of spend covered. Good practice.
    LOW: >= 80% of spend covered. Minimum viable for credible
        reporting per GHG Protocol.
    MINIMAL: < 80% of spend covered. Below minimum threshold;
        credibility of inventory is at risk.
    """

    FULL = "full"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

# =============================================================================
# Constant Tables (all Decimal for deterministic arithmetic)
# =============================================================================

# ---------------------------------------------------------------------------
# GWP values by IPCC Assessment Report
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[GWPSource, Dict[EmissionGas, Decimal]] = {
    GWPSource.AR4: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("25"),
        EmissionGas.N2O: Decimal("298"),
        EmissionGas.CO2E: Decimal("1"),
    },
    GWPSource.AR5: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("28"),
        EmissionGas.N2O: Decimal("265"),
        EmissionGas.CO2E: Decimal("1"),
    },
    GWPSource.AR6: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("27.9"),
        EmissionGas.N2O: Decimal("273"),
        EmissionGas.CO2E: Decimal("1"),
    },
}

# ---------------------------------------------------------------------------
# DQI score numeric values (1=best, 5=worst)
# ---------------------------------------------------------------------------

DQI_SCORE_VALUES: Dict[DQIScore, Decimal] = {
    DQIScore.VERY_GOOD: Decimal("1.0"),
    DQIScore.GOOD: Decimal("2.0"),
    DQIScore.FAIR: Decimal("3.0"),
    DQIScore.POOR: Decimal("4.0"),
    DQIScore.VERY_POOR: Decimal("5.0"),
}

# ---------------------------------------------------------------------------
# DQI quality tier labels with composite score ranges
# (min_score inclusive, max_score exclusive)
# ---------------------------------------------------------------------------

DQI_QUALITY_TIERS: Dict[str, Tuple[Decimal, Decimal]] = {
    "Very Good": (Decimal("1.0"), Decimal("1.6")),
    "Good": (Decimal("1.6"), Decimal("2.6")),
    "Fair": (Decimal("2.6"), Decimal("3.6")),
    "Poor": (Decimal("3.6"), Decimal("4.6")),
    "Very Poor": (Decimal("4.6"), Decimal("5.1")),
}

# ---------------------------------------------------------------------------
# Uncertainty ranges by calculation method (min%, max%)
# ---------------------------------------------------------------------------

UNCERTAINTY_RANGES: Dict[CalculationMethod, Tuple[Decimal, Decimal]] = {
    CalculationMethod.SUPPLIER_SPECIFIC: (Decimal("10"), Decimal("30")),
    CalculationMethod.HYBRID: (Decimal("20"), Decimal("50")),
    CalculationMethod.AVERAGE_DATA: (Decimal("30"), Decimal("60")),
    CalculationMethod.SPEND_BASED: (Decimal("50"), Decimal("100")),
}

# ---------------------------------------------------------------------------
# Coverage thresholds by level (minimum spend percentage)
# ---------------------------------------------------------------------------

COVERAGE_THRESHOLDS: Dict[CoverageLevel, Decimal] = {
    CoverageLevel.FULL: Decimal("100.0"),
    CoverageLevel.HIGH: Decimal("95.0"),
    CoverageLevel.MEDIUM: Decimal("90.0"),
    CoverageLevel.LOW: Decimal("80.0"),
    CoverageLevel.MINIMAL: Decimal("0.0"),
}

# ---------------------------------------------------------------------------
# Emission factor hierarchy priority (1=best, 8=worst)
# Per GHG Protocol Scope 3 Technical Guidance Section 1.4
# ---------------------------------------------------------------------------

EF_HIERARCHY_PRIORITY: Dict[str, int] = {
    "supplier_epd_verified": 1,
    "supplier_cdp_unverified": 2,
    "product_lca_ecoinvent": 3,
    "material_avg_ice_defra": 4,
    "industry_avg_physical": 5,
    "regional_eeio_exiobase": 6,
    "national_eeio_useeio": 7,
    "global_avg_eeio_fallback": 8,
}

# ---------------------------------------------------------------------------
# Pedigree matrix uncertainty factors by DQI score
# Per ecoinvent pedigree matrix methodology
# ---------------------------------------------------------------------------

PEDIGREE_UNCERTAINTY_FACTORS: Dict[DQIScore, Decimal] = {
    DQIScore.VERY_GOOD: Decimal("1.00"),
    DQIScore.GOOD: Decimal("1.05"),
    DQIScore.FAIR: Decimal("1.10"),
    DQIScore.POOR: Decimal("1.20"),
    DQIScore.VERY_POOR: Decimal("1.50"),
}

# ---------------------------------------------------------------------------
# Currency exchange rates to USD (annual average 2024 estimates)
# All rates expressed as units of foreign currency per 1 USD
# ---------------------------------------------------------------------------

CURRENCY_EXCHANGE_RATES: Dict[CurrencyCode, Decimal] = {
    CurrencyCode.USD: Decimal("1.00000000"),
    CurrencyCode.EUR: Decimal("0.92410000"),
    CurrencyCode.GBP: Decimal("0.79250000"),
    CurrencyCode.JPY: Decimal("151.35000000"),
    CurrencyCode.CNY: Decimal("7.24500000"),
    CurrencyCode.INR: Decimal("83.35000000"),
    CurrencyCode.CAD: Decimal("1.36200000"),
    CurrencyCode.AUD: Decimal("1.53500000"),
    CurrencyCode.CHF: Decimal("0.88150000"),
    CurrencyCode.KRW: Decimal("1345.60000000"),
    CurrencyCode.BRL: Decimal("4.97200000"),
    CurrencyCode.MXN: Decimal("17.14500000"),
    CurrencyCode.SGD: Decimal("1.34400000"),
    CurrencyCode.HKD: Decimal("7.82600000"),
    CurrencyCode.SEK: Decimal("10.51200000"),
    CurrencyCode.NOK: Decimal("10.72800000"),
    CurrencyCode.DKK: Decimal("6.89400000"),
    CurrencyCode.PLN: Decimal("4.03100000"),
    CurrencyCode.CZK: Decimal("23.21500000"),
    CurrencyCode.ZAR: Decimal("18.65200000"),
}

# ---------------------------------------------------------------------------
# Industry margin percentages by NAICS 2-digit sector
# Used to convert purchaser price to producer/basic price for EEIO
# Margin = wholesale + retail + transport margin
# ---------------------------------------------------------------------------

INDUSTRY_MARGIN_PERCENTAGES: Dict[str, Decimal] = {
    # NAICS 2-digit sector code -> margin percentage
    "11": Decimal("25.0"),   # Agriculture, Forestry, Fishing
    "21": Decimal("15.0"),   # Mining, Quarrying, Oil/Gas
    "22": Decimal("10.0"),   # Utilities
    "23": Decimal("20.0"),   # Construction
    "31": Decimal("35.0"),   # Manufacturing - Food
    "32": Decimal("30.0"),   # Manufacturing - Non-metallic
    "33": Decimal("30.0"),   # Manufacturing - Metals/Machinery
    "42": Decimal("25.0"),   # Wholesale Trade
    "44": Decimal("40.0"),   # Retail Trade - Store
    "45": Decimal("40.0"),   # Retail Trade - Non-store
    "48": Decimal("15.0"),   # Transportation
    "49": Decimal("15.0"),   # Warehousing
    "51": Decimal("10.0"),   # Information
    "52": Decimal("8.0"),    # Finance and Insurance
    "53": Decimal("12.0"),   # Real Estate
    "54": Decimal("10.0"),   # Professional Services
    "55": Decimal("8.0"),    # Management of Companies
    "56": Decimal("12.0"),   # Administrative/Waste Services
    "61": Decimal("8.0"),    # Educational Services
    "62": Decimal("10.0"),   # Health Care
    "71": Decimal("15.0"),   # Arts, Entertainment
    "72": Decimal("25.0"),   # Accommodation and Food Services
    "81": Decimal("15.0"),   # Other Services
    "92": Decimal("5.0"),    # Public Administration
}

# ---------------------------------------------------------------------------
# EEIO emission factors by NAICS-6 sector code
# Source: EPA USEEIO v1.2, base year 2019, factors in kgCO2e per USD
# (purchaser price, 2021 USD). Top 50+ sectors by economic activity.
# ---------------------------------------------------------------------------

EEIO_EMISSION_FACTORS: Dict[str, Decimal] = {
    # Agriculture, Forestry, Fishing and Hunting (NAICS 11)
    "111110": Decimal("0.7890"),   # Soybean Farming
    "111150": Decimal("0.6540"),   # Corn Farming
    "111199": Decimal("0.5870"),   # All Other Grain Farming
    "111310": Decimal("0.4230"),   # Orange Groves
    "111920": Decimal("0.3560"),   # Cotton Farming
    "112111": Decimal("1.8340"),   # Beef Cattle Ranching
    "112120": Decimal("0.9870"),   # Dairy Cattle and Milk Production
    "112210": Decimal("0.6230"),   # Hog and Pig Farming
    "112310": Decimal("0.4560"),   # Chicken Egg Production
    "113110": Decimal("0.2340"),   # Timber Tract Operations
    # Mining, Quarrying, and Oil and Gas (NAICS 21)
    "211120": Decimal("0.8910"),   # Crude Petroleum Extraction
    "211130": Decimal("0.7650"),   # Natural Gas Extraction
    "212210": Decimal("0.5430"),   # Iron Ore Mining
    "212230": Decimal("0.4870"),   # Copper, Nickel, Lead Mining
    "212310": Decimal("0.3120"),   # Stone Mining and Quarrying
    # Utilities (NAICS 22)
    "221110": Decimal("1.2340"),   # Electric Power Generation
    "221210": Decimal("0.4560"),   # Natural Gas Distribution
    # Construction (NAICS 23)
    "236110": Decimal("0.3450"),   # Residential Building Construction
    "236220": Decimal("0.3780"),   # Commercial Building Construction
    "237310": Decimal("0.4120"),   # Highway, Street Construction
    # Manufacturing - Food (NAICS 311)
    "311111": Decimal("0.4560"),   # Dog and Cat Food Manufacturing
    "311210": Decimal("0.3890"),   # Flour Milling
    "311410": Decimal("0.2980"),   # Frozen Fruit/Vegetable Manufacturing
    "311513": Decimal("0.5670"),   # Cheese Manufacturing
    "311615": Decimal("0.8920"),   # Poultry Processing
    "311710": Decimal("0.3450"),   # Seafood Processing
    "311920": Decimal("0.2340"),   # Coffee and Tea Manufacturing
    # Manufacturing - Chemicals (NAICS 325)
    "325110": Decimal("0.9870"),   # Petrochemical Manufacturing
    "325180": Decimal("0.7650"),   # Other Basic Inorganic Chemical Mfg
    "325211": Decimal("0.8430"),   # Plastics Material and Resin Mfg
    "325311": Decimal("0.5670"),   # Nitrogenous Fertilizer Mfg
    "325411": Decimal("0.2340"),   # Medicinal and Botanical Mfg
    "325510": Decimal("0.3450"),   # Paint and Coating Manufacturing
    "325611": Decimal("0.1890"),   # Soap and Cleaning Compound Mfg
    # Manufacturing - Metals (NAICS 331)
    "331110": Decimal("1.2340"),   # Iron and Steel Mills
    "331313": Decimal("0.9870"),   # Alumina Refining/Primary Aluminum
    "331420": Decimal("0.6780"),   # Copper Rolling and Drawing
    "331511": Decimal("0.5430"),   # Iron Foundries
    # Manufacturing - Machinery/Equipment (NAICS 333)
    "333111": Decimal("0.2340"),   # Farm Machinery Manufacturing
    "333120": Decimal("0.2560"),   # Construction Machinery Mfg
    "333310": Decimal("0.1890"),   # Commercial/Service Machinery Mfg
    # Manufacturing - Electronics (NAICS 334)
    "334111": Decimal("0.1230"),   # Electronic Computer Manufacturing
    "334210": Decimal("0.0980"),   # Telephone Apparatus Manufacturing
    "334413": Decimal("0.1560"),   # Semiconductor Device Manufacturing
    "334510": Decimal("0.0870"),   # Electromedical Apparatus Mfg
    # Manufacturing - Paper (NAICS 322)
    "322110": Decimal("0.6780"),   # Pulp Mills
    "322121": Decimal("0.5430"),   # Paper (except Newsprint) Mills
    "322130": Decimal("0.4560"),   # Paperboard Mills
    "322211": Decimal("0.3450"),   # Corrugated Container Mfg
    # Wholesale Trade (NAICS 42)
    "423110": Decimal("0.0560"),   # Auto Parts Wholesalers
    "423400": Decimal("0.0450"),   # Professional Equipment Wholesalers
    "423510": Decimal("0.0670"),   # Metal Service Centers
    # Retail Trade (NAICS 44-45)
    "445110": Decimal("0.0340"),   # Supermarkets/Grocery Stores
    "452210": Decimal("0.0280"),   # Department Stores
    # Transportation (NAICS 48)
    "481111": Decimal("0.5670"),   # Scheduled Passenger Air Transport
    "484110": Decimal("0.3450"),   # General Freight Trucking, Local
    # Information (NAICS 51)
    "511210": Decimal("0.0230"),   # Software Publishers
    "518210": Decimal("0.0450"),   # Data Processing/Hosting Services
    # Finance and Insurance (NAICS 52)
    "522110": Decimal("0.0120"),   # Commercial Banking
    "524114": Decimal("0.0150"),   # Direct Health/Medical Insurance
    # Professional Services (NAICS 54)
    "541110": Decimal("0.0180"),   # Offices of Lawyers
    "541211": Decimal("0.0160"),   # Offices of CPAs
    "541310": Decimal("0.0210"),   # Architectural Services
    "541330": Decimal("0.0230"),   # Engineering Services
    "541511": Decimal("0.0190"),   # Custom Computer Programming
    "541611": Decimal("0.0170"),   # Management Consulting
    "541711": Decimal("0.0340"),   # R&D in Biotechnology
    # Health Care (NAICS 62)
    "621111": Decimal("0.0230"),   # Offices of Physicians
    "622110": Decimal("0.0450"),   # General Medical Hospitals
    # Accommodation and Food (NAICS 72)
    "721110": Decimal("0.0560"),   # Hotels and Motels
    "722511": Decimal("0.0670"),   # Full-Service Restaurants
}

# ---------------------------------------------------------------------------
# Physical emission factors by material
# kgCO2e per kg of material (cradle-to-gate)
# Sources: World Steel 2023, IAI 2023, GCCA 2023, PlasticsEurope 2022,
# ICE v3.0, CEPI 2022, Textile Exchange 2023, ICA 2022, IEA 2023
# ---------------------------------------------------------------------------

PHYSICAL_EMISSION_FACTORS: Dict[str, Decimal] = {
    # Metals
    "steel_primary_bof": Decimal("2.33"),
    "steel_secondary_eaf": Decimal("0.67"),
    "steel_world_avg": Decimal("1.37"),
    "steel_virgin_100pct": Decimal("2.89"),
    "aluminum_primary_global": Decimal("16.70"),
    "aluminum_secondary": Decimal("0.50"),
    "aluminum_33pct_recycled": Decimal("6.67"),
    "copper_primary": Decimal("4.10"),
    "lead": Decimal("1.57"),
    "zinc": Decimal("3.86"),
    "lithium_carbonate": Decimal("7.30"),
    # Plastics
    "hdpe": Decimal("1.80"),
    "ldpe": Decimal("2.08"),
    "pp_polypropylene": Decimal("1.63"),
    "pet": Decimal("2.15"),
    "pvc": Decimal("2.41"),
    "ps_polystyrene": Decimal("3.43"),
    "abs": Decimal("3.76"),
    "nylon_6": Decimal("9.10"),
    # Construction
    "cement_portland_global": Decimal("0.63"),
    "cement_portland_cem_i": Decimal("0.91"),
    "concrete_readymix_30mpa": Decimal("0.13"),
    "concrete_high_50mpa": Decimal("0.17"),
    "float_glass": Decimal("1.20"),
    "glass_general": Decimal("1.22"),
    "bricks_general": Decimal("0.24"),
    "timber_softwood_sawn": Decimal("0.31"),
    "timber_hardwood_sawn": Decimal("0.42"),
    "timber_glulam": Decimal("0.51"),
    # Paper and Packaging
    "corrugated_cardboard": Decimal("0.79"),
    "kraft_paper": Decimal("1.06"),
    "recycled_paper": Decimal("0.61"),
    # Textiles
    "cotton_conventional": Decimal("5.90"),
    "cotton_organic": Decimal("3.80"),
    "polyester_fiber": Decimal("3.40"),
    "nylon_fiber": Decimal("6.80"),
    "wool": Decimal("23.80"),
    # Electronics
    "silicon_wafer_solar": Decimal("50.00"),
    "pcb_printed_circuit": Decimal("12.50"),
    # Chemicals
    "ammonia": Decimal("2.10"),
    "ethylene": Decimal("1.28"),
    "propylene": Decimal("1.47"),
    "methanol": Decimal("0.73"),
    # Rubber
    "natural_rubber": Decimal("1.40"),
    "synthetic_rubber_sbr": Decimal("2.85"),
}

# ---------------------------------------------------------------------------
# Framework required disclosures for Category 1 compliance checking
# ---------------------------------------------------------------------------

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "category_1_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "spend_coverage_percentage",
        "category_boundary_description",
        "double_counting_prevention",
        "base_year_recalculation_policy",
        "exclusions_and_limitations",
        "gwp_values_used",
        "uncertainty_assessment",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "category_1_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "spend_coverage_percentage",
        "supplier_engagement_strategy",
        "intensity_per_revenue",
        "significant_changes_explanation",
        "base_year_emissions",
        "reduction_targets",
        "value_chain_boundary",
        "gwp_values_used",
    ],
    ComplianceFramework.CDP: [
        "category_1_total_tco2e",
        "category_1_relevance",
        "calculation_methodology",
        "percentage_calculated_primary_data",
        "percentage_calculated_secondary_data",
        "emission_factor_sources",
        "spend_coverage_percentage",
        "supplier_engagement_scoring",
        "verification_status",
        "boundary_description",
    ],
    ComplianceFramework.SBTI: [
        "category_1_total_tco2e",
        "base_year_category_1_tco2e",
        "target_year",
        "reduction_percentage",
        "coverage_percentage",
        "supplier_engagement_targets",
        "calculation_methodology",
        "emission_factor_sources",
    ],
    ComplianceFramework.SB253: [
        "category_1_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "spend_coverage_percentage",
        "assurance_provider",
        "reporting_entity_revenue",
        "gwp_values_used",
    ],
    ComplianceFramework.GRI: [
        "category_1_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "gwp_values_used",
        "consolidation_approach",
        "base_year_information",
        "standards_and_methodologies",
        "significant_changes",
    ],
    ComplianceFramework.ISO_14064: [
        "category_1_total_tco2e",
        "method_justification",
        "emission_by_gas_co2",
        "emission_by_gas_ch4",
        "emission_by_gas_n2o",
        "emission_factor_sources",
        "gwp_values_used",
        "uncertainty_assessment",
        "organizational_boundary",
        "reporting_period",
        "base_year_information",
        "data_quality_assessment",
    ],
}

# =============================================================================
# Data Models (25) -- Pydantic v2, frozen=True
# =============================================================================

# ---------------------------------------------------------------------------
# 1. ProcurementItem
# ---------------------------------------------------------------------------

class ProcurementItem(GreenLangBase):
    """A single purchased item or service with spend and quantity data.

    Represents one line item from a procurement record. Contains
    spend amount, optional physical quantity, classification codes,
    and supplier information. This is the primary input unit for
    all four calculation methods.

    Attributes:
        item_id: Unique identifier for this procurement item.
        description: Human-readable description of the item.
        procurement_type: Whether this is a good, service, or mixed.
        spend_amount: Total spend in the original currency.
        currency: Currency of the spend amount.
        quantity: Optional physical quantity purchased.
        quantity_unit: Unit of the physical quantity (e.g. kg, tonnes).
        naics_code: NAICS 2022 classification code (6-digit).
        nace_code: NACE Rev 2.1 classification code.
        isic_code: ISIC Rev 4.1 classification code.
        unspsc_code: UNSPSC v28 classification code (8-digit).
        material_category: Material category for physical EF lookup.
        supplier_id: Unique identifier of the supplier.
        supplier_name: Human-readable supplier name.
        facility_id: Facility identifier for multi-site reporting.
        period_start: Start date of the procurement period.
        period_end: End date of the procurement period.
        is_capital_good: Whether this item is a capital good (Cat 2).
        is_fuel_energy: Whether this is fuel/energy for own use (Cat 3).
        is_transport: Whether this is upstream transport (Cat 4).
        is_business_travel: Whether this is business travel (Cat 6).
        is_intercompany: Whether this is an intercompany transaction.
        is_credit_return: Whether this is a credit memo or return.
        metadata: Additional key-value pairs for extensibility.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this procurement item",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Human-readable description of the item",
    )
    procurement_type: ProcurementType = Field(
        default=ProcurementType.GOODS,
        description="Whether this is a good, service, or mixed",
    )
    spend_amount: Decimal = Field(
        ...,
        description="Total spend in the original currency",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of the spend amount",
    )
    quantity: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Optional physical quantity purchased",
    )
    quantity_unit: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Unit of the physical quantity (e.g. kg, tonnes)",
    )
    naics_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="NAICS 2022 classification code (6-digit)",
    )
    nace_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="NACE Rev 2.1 classification code",
    )
    isic_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="ISIC Rev 4.1 classification code",
    )
    unspsc_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="UNSPSC v28 classification code (8-digit)",
    )
    material_category: Optional[MaterialCategory] = Field(
        default=None,
        description="Material category for physical EF lookup",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Unique identifier of the supplier",
    )
    supplier_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable supplier name",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Facility identifier for multi-site reporting",
    )
    period_start: Optional[date] = Field(
        default=None,
        description="Start date of the procurement period",
    )
    period_end: Optional[date] = Field(
        default=None,
        description="End date of the procurement period",
    )
    is_capital_good: bool = Field(
        default=False,
        description="Whether this item is a capital good (Cat 2)",
    )
    is_fuel_energy: bool = Field(
        default=False,
        description="Whether this is fuel/energy for own use (Cat 3)",
    )
    is_transport: bool = Field(
        default=False,
        description="Whether this is upstream transport (Cat 4)",
    )
    is_business_travel: bool = Field(
        default=False,
        description="Whether this is business travel (Cat 6)",
    )
    is_intercompany: bool = Field(
        default=False,
        description="Whether this is an intercompany transaction",
    )
    is_credit_return: bool = Field(
        default=False,
        description="Whether this is a credit memo or return",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for extensibility",
    )

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: Optional[date], info: Any) -> Optional[date]:
        """Validate that period_end is on or after period_start."""
        if v is None:
            return v
        start = info.data.get("period_start")
        if start is not None and v < start:
            raise ValueError(
                f"period_end ({v}) must be on or after "
                f"period_start ({start})"
            )
        return v

# ---------------------------------------------------------------------------
# 2. SpendRecord
# ---------------------------------------------------------------------------

class SpendRecord(GreenLangBase):
    """Spend-based input record for EEIO calculation.

    A procurement item enriched with spend-specific fields needed
    for the spend-based calculation method: currency conversion
    parameters, EEIO database selection, and margin adjustment.

    Attributes:
        item: The underlying procurement item.
        spend_usd: Spend amount converted to USD.
        spend_deflated_usd: Spend adjusted for inflation to
            EEIO base year.
        spend_producer_usd: Spend adjusted to producer/basic
            price after margin removal.
        eeio_database: EEIO database to use for factor lookup.
        eeio_sector_code: Resolved EEIO sector code (NAICS-6).
        margin_rate: Margin rate applied for price adjustment.
        cpi_ratio: CPI ratio for inflation deflation.
        fx_rate: Foreign exchange rate used for conversion.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: ProcurementItem = Field(
        ...,
        description="The underlying procurement item",
    )
    spend_usd: Decimal = Field(
        default=Decimal("0"),
        description="Spend amount converted to USD",
    )
    spend_deflated_usd: Decimal = Field(
        default=Decimal("0"),
        description="Spend adjusted for inflation to EEIO base year",
    )
    spend_producer_usd: Decimal = Field(
        default=Decimal("0"),
        description=(
            "Spend adjusted to producer/basic price after "
            "margin removal"
        ),
    )
    eeio_database: EEIODatabase = Field(
        default=EEIODatabase.EPA_USEEIO,
        description="EEIO database to use for factor lookup",
    )
    eeio_sector_code: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Resolved EEIO sector code (NAICS-6)",
    )
    margin_rate: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Margin rate applied for price adjustment",
    )
    cpi_ratio: Decimal = Field(
        default=Decimal("1.0"),
        gt=Decimal("0"),
        description="CPI ratio for inflation deflation",
    )
    fx_rate: Decimal = Field(
        default=Decimal("1.0"),
        gt=Decimal("0"),
        description="Foreign exchange rate used for conversion",
    )

# ---------------------------------------------------------------------------
# 3. PhysicalRecord
# ---------------------------------------------------------------------------

class PhysicalRecord(GreenLangBase):
    """Quantity-based input record for average-data calculation.

    A procurement item enriched with physical quantity data and
    material classification for average-data emission factor lookup.

    Attributes:
        item: The underlying procurement item.
        quantity_kg: Physical quantity normalized to kilograms.
        material_key: Material key for PHYSICAL_EMISSION_FACTORS
            lookup.
        ef_source: Source database for the physical emission factor.
        includes_transport: Whether the EF includes transport to
            gate; if False, a transport adder is applied.
        transport_distance_km: Distance for transport adder if
            applicable.
        transport_mode: Transport mode for transport adder EF
            lookup (e.g. road, rail, sea).
        waste_loss_factor: Factor to account for material waste
            and loss in production (default 1.0, no loss).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: ProcurementItem = Field(
        ...,
        description="The underlying procurement item",
    )
    quantity_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Physical quantity normalized to kilograms",
    )
    material_key: Optional[str] = Field(
        default=None,
        max_length=100,
        description=(
            "Material key for PHYSICAL_EMISSION_FACTORS lookup"
        ),
    )
    ef_source: PhysicalEFSource = Field(
        default=PhysicalEFSource.DEFRA,
        description="Source database for the physical emission factor",
    )
    includes_transport: bool = Field(
        default=True,
        description=(
            "Whether the EF includes transport to gate; if False "
            "a transport adder is applied"
        ),
    )
    transport_distance_km: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Distance for transport adder if applicable",
    )
    transport_mode: Optional[str] = Field(
        default=None,
        max_length=50,
        description=(
            "Transport mode for transport adder EF lookup "
            "(e.g. road, rail, sea)"
        ),
    )
    waste_loss_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("1.0"),
        description=(
            "Factor to account for material waste and loss "
            "in production (default 1.0, no loss)"
        ),
    )

# ---------------------------------------------------------------------------
# 4. SupplierRecord
# ---------------------------------------------------------------------------

class SupplierRecord(GreenLangBase):
    """Supplier-specific input record for supplier-level calculation.

    Contains primary emission data from a specific supplier,
    including the data source, allocation method, and verification
    status for the supplier-specific calculation method.

    Attributes:
        item: The underlying procurement item.
        supplier_emissions_tco2e: Total supplier facility emissions
            in tonnes CO2e.
        product_ef_kgco2e_per_unit: Product-level emission factor
            from EPD/PCF if available (kgCO2e per unit).
        allocation_method: Method used to allocate facility-level
            emissions to this purchase.
        allocation_factor: Calculated allocation factor (0-1).
        data_source: Source of the supplier emission data.
        verification_status: Whether data is third-party verified.
        epd_number: EPD registration number if applicable.
        pcf_value_kgco2e: Product carbon footprint value if
            available.
        reporting_year: Year the supplier data was reported.
        boundary: System boundary of supplier data (cradle-to-gate
            or cradle-to-grave).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: ProcurementItem = Field(
        ...,
        description="The underlying procurement item",
    )
    supplier_emissions_tco2e: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description=(
            "Total supplier facility emissions in tonnes CO2e"
        ),
    )
    product_ef_kgco2e_per_unit: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description=(
            "Product-level emission factor from EPD/PCF "
            "(kgCO2e per unit)"
        ),
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.REVENUE_BASED,
        description=(
            "Method used to allocate facility-level emissions"
        ),
    )
    allocation_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Calculated allocation factor (0-1)",
    )
    data_source: SupplierDataSource = Field(
        default=SupplierDataSource.DIRECT_MEASUREMENT,
        description="Source of the supplier emission data",
    )
    verification_status: str = Field(
        default="unverified",
        max_length=50,
        description="Whether data is third-party verified",
    )
    epd_number: Optional[str] = Field(
        default=None,
        max_length=100,
        description="EPD registration number if applicable",
    )
    pcf_value_kgco2e: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Product carbon footprint value if available",
    )
    reporting_year: Optional[int] = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Year the supplier data was reported",
    )
    boundary: str = Field(
        default="cradle_to_gate",
        max_length=50,
        description=(
            "System boundary of supplier data "
            "(cradle_to_gate or cradle_to_grave)"
        ),
    )

# ---------------------------------------------------------------------------
# 5. SpendBasedResult
# ---------------------------------------------------------------------------

class SpendBasedResult(GreenLangBase):
    """Result of a spend-based emission calculation for one item.

    Contains the calculated emissions using the EEIO method,
    including intermediate values for the currency conversion,
    inflation adjustment, and margin removal steps.

    Attributes:
        item_id: Reference to the source procurement item.
        emissions_kgco2e: Calculated emissions in kgCO2e.
        emissions_tco2e: Calculated emissions in tonnes CO2e.
        spend_original: Original spend amount in source currency.
        spend_usd: Spend converted to USD.
        spend_deflated_usd: Spend adjusted for inflation.
        spend_producer_usd: Spend adjusted to producer price.
        eeio_factor_kgco2e_per_usd: EEIO factor applied.
        eeio_database: EEIO database used.
        eeio_sector_code: EEIO sector code matched.
        currency: Source currency.
        fx_rate: Exchange rate used.
        cpi_ratio: CPI ratio for inflation adjustment.
        margin_rate: Margin rate for price adjustment.
        dqi_scores: Data quality scores for this calculation.
        provenance_hash: SHA-256 hash of the calculation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source procurement item",
    )
    emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Calculated emissions in kgCO2e",
    )
    emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Calculated emissions in tonnes CO2e",
    )
    spend_original: Decimal = Field(
        ...,
        description="Original spend amount in source currency",
    )
    spend_usd: Decimal = Field(
        ...,
        description="Spend converted to USD",
    )
    spend_deflated_usd: Decimal = Field(
        ...,
        description="Spend adjusted for inflation",
    )
    spend_producer_usd: Decimal = Field(
        ...,
        description="Spend adjusted to producer price",
    )
    eeio_factor_kgco2e_per_usd: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="EEIO factor applied (kgCO2e per USD)",
    )
    eeio_database: EEIODatabase = Field(
        ...,
        description="EEIO database used",
    )
    eeio_sector_code: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="EEIO sector code matched",
    )
    currency: CurrencyCode = Field(
        ...,
        description="Source currency",
    )
    fx_rate: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Exchange rate used",
    )
    cpi_ratio: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="CPI ratio for inflation adjustment",
    )
    margin_rate: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Margin rate for price adjustment",
    )
    dqi_scores: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Data quality scores for this calculation",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the calculation",
    )

# ---------------------------------------------------------------------------
# 6. AverageDataResult
# ---------------------------------------------------------------------------

class AverageDataResult(GreenLangBase):
    """Result of an average-data emission calculation for one item.

    Contains the calculated emissions using physical quantity
    multiplied by industry-average emission factors, including
    optional transport adder and waste/loss factor.

    Attributes:
        item_id: Reference to the source procurement item.
        emissions_kgco2e: Calculated emissions in kgCO2e.
        emissions_tco2e: Calculated emissions in tonnes CO2e.
        quantity_kg: Physical quantity in kilograms.
        ef_kgco2e_per_kg: Emission factor applied (kgCO2e/kg).
        ef_source: Source of the emission factor.
        material_key: Material key used for EF lookup.
        transport_emissions_kgco2e: Additional transport emissions
            if EF excludes transport to gate.
        waste_loss_factor: Waste/loss factor applied.
        total_with_transport_kgco2e: Total emissions including
            transport adder.
        dqi_scores: Data quality scores for this calculation.
        provenance_hash: SHA-256 hash of the calculation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source procurement item",
    )
    emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Calculated emissions in kgCO2e",
    )
    emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Calculated emissions in tonnes CO2e",
    )
    quantity_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Physical quantity in kilograms",
    )
    ef_kgco2e_per_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor applied (kgCO2e per kg)",
    )
    ef_source: PhysicalEFSource = Field(
        ...,
        description="Source of the emission factor",
    )
    material_key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Material key used for EF lookup",
    )
    transport_emissions_kgco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description=(
            "Additional transport emissions if EF excludes "
            "transport to gate"
        ),
    )
    waste_loss_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("1.0"),
        description="Waste/loss factor applied",
    )
    total_with_transport_kgco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total emissions including transport adder",
    )
    dqi_scores: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Data quality scores for this calculation",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the calculation",
    )

# ---------------------------------------------------------------------------
# 7. SupplierSpecificResult
# ---------------------------------------------------------------------------

class SupplierSpecificResult(GreenLangBase):
    """Result of a supplier-specific emission calculation for one item.

    Contains the calculated emissions using primary supplier data,
    including the allocation method and factor if facility-level
    data was used.

    Attributes:
        item_id: Reference to the source procurement item.
        emissions_kgco2e: Calculated emissions in kgCO2e.
        emissions_tco2e: Calculated emissions in tonnes CO2e.
        supplier_id: Identifier of the supplier.
        supplier_name: Name of the supplier.
        data_source: Source of the supplier emission data.
        allocation_method: Allocation method if facility-level.
        allocation_factor: Allocation factor applied (0-1).
        supplier_total_tco2e: Total supplier facility emissions.
        product_ef_kgco2e_per_unit: Product-level EF if available.
        quantity: Quantity purchased.
        verification_status: Third-party verification status.
        dqi_scores: Data quality scores for this calculation.
        provenance_hash: SHA-256 hash of the calculation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source procurement item",
    )
    emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Calculated emissions in kgCO2e",
    )
    emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Calculated emissions in tonnes CO2e",
    )
    supplier_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Identifier of the supplier",
    )
    supplier_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Name of the supplier",
    )
    data_source: SupplierDataSource = Field(
        ...,
        description="Source of the supplier emission data",
    )
    allocation_method: AllocationMethod = Field(
        ...,
        description="Allocation method if facility-level",
    )
    allocation_factor: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Allocation factor applied (0-1)",
    )
    supplier_total_tco2e: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Total supplier facility emissions in tCO2e",
    )
    product_ef_kgco2e_per_unit: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Product-level EF if available (kgCO2e/unit)",
    )
    quantity: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Quantity purchased",
    )
    verification_status: str = Field(
        default="unverified",
        max_length=50,
        description="Third-party verification status",
    )
    dqi_scores: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Data quality scores for this calculation",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the calculation",
    )

# ---------------------------------------------------------------------------
# 8. HybridResult
# ---------------------------------------------------------------------------

class HybridResult(GreenLangBase):
    """Aggregated result combining all calculation methods.

    The hybrid method combines supplier-specific, average-data,
    and spend-based results, using the highest-quality data
    available for each procurement item. This model holds the
    aggregated totals and coverage breakdown.

    Attributes:
        calculation_id: Unique identifier for this calculation run.
        total_emissions_kgco2e: Total Category 1 emissions in kgCO2e.
        total_emissions_tco2e: Total Category 1 emissions in tCO2e.
        spend_based_emissions_tco2e: Emissions from spend-based items.
        average_data_emissions_tco2e: Emissions from average-data.
        supplier_specific_emissions_tco2e: Emissions from supplier
            items.
        spend_based_coverage_pct: Percentage of total spend covered
            by spend-based method.
        average_data_coverage_pct: Percentage covered by average-data.
        supplier_specific_coverage_pct: Percentage covered by
            supplier-specific.
        total_coverage_pct: Total spend coverage percentage.
        coverage_level: Qualitative coverage level classification.
        total_spend_usd: Total procurement spend in USD.
        item_count: Total number of procurement items processed.
        spend_based_count: Number of items using spend-based.
        average_data_count: Number of items using average-data.
        supplier_specific_count: Number of items using supplier data.
        excluded_count: Number of items excluded (boundary checks).
        weighted_dqi: Emission-weighted composite DQI score.
        provenance_hash: SHA-256 hash of the aggregated result.
        timestamp: UTC timestamp of calculation.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this calculation run",
    )
    total_emissions_kgco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total Category 1 emissions in kgCO2e",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total Category 1 emissions in tCO2e",
    )
    spend_based_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions from spend-based items in tCO2e",
    )
    average_data_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions from average-data items in tCO2e",
    )
    supplier_specific_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions from supplier-specific items in tCO2e",
    )
    spend_based_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of spend covered by spend-based",
    )
    average_data_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of spend covered by average-data",
    )
    supplier_specific_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of spend covered by supplier-specific",
    )
    total_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Total spend coverage percentage",
    )
    coverage_level: CoverageLevel = Field(
        default=CoverageLevel.MINIMAL,
        description="Qualitative coverage level classification",
    )
    total_spend_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total procurement spend in USD",
    )
    item_count: int = Field(
        default=0,
        ge=0,
        description="Total number of procurement items processed",
    )
    spend_based_count: int = Field(
        default=0,
        ge=0,
        description="Number of items using spend-based method",
    )
    average_data_count: int = Field(
        default=0,
        ge=0,
        description="Number of items using average-data method",
    )
    supplier_specific_count: int = Field(
        default=0,
        ge=0,
        description="Number of items using supplier-specific method",
    )
    excluded_count: int = Field(
        default=0,
        ge=0,
        description="Number of items excluded by boundary checks",
    )
    weighted_dqi: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Emission-weighted composite DQI score",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the aggregated result",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of calculation",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )

# ---------------------------------------------------------------------------
# 9. EEIOFactor
# ---------------------------------------------------------------------------

class EEIOFactor(GreenLangBase):
    """An EEIO emission factor entry from a reference database.

    Represents a single row in the EEIO factor table, mapping a
    sector classification code to an emission factor expressed in
    kgCO2e per unit of economic output.

    Attributes:
        sector_code: NAICS or NACE sector classification code.
        sector_name: Human-readable sector name.
        factor_kgco2e_per_unit: Emission factor in kgCO2e per
            currency unit.
        database: EEIO database source.
        database_version: Version of the database.
        base_year: Base year for the economic data.
        base_currency: Currency of the base economic data.
        region: Geographic region the factor applies to.
        margin_type: Price type (basic or purchaser).
        classification_system: Classification system used.
        last_updated: Date the factor was last updated.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    sector_code: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="NAICS or NACE sector classification code",
    )
    sector_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable sector name",
    )
    factor_kgco2e_per_unit: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor in kgCO2e per currency unit",
    )
    database: EEIODatabase = Field(
        ...,
        description="EEIO database source",
    )
    database_version: str = Field(
        default="",
        max_length=50,
        description="Version of the database",
    )
    base_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Base year for the economic data",
    )
    base_currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of the base economic data",
    )
    region: str = Field(
        default="US",
        max_length=20,
        description="Geographic region the factor applies to",
    )
    margin_type: str = Field(
        default="purchaser",
        max_length=50,
        description="Price type (basic or purchaser)",
    )
    classification_system: SpendClassificationSystem = Field(
        default=SpendClassificationSystem.NAICS,
        description="Classification system used",
    )
    last_updated: Optional[date] = Field(
        default=None,
        description="Date the factor was last updated",
    )

# ---------------------------------------------------------------------------
# 10. PhysicalEF
# ---------------------------------------------------------------------------

class PhysicalEF(GreenLangBase):
    """A physical (quantity-based) emission factor entry.

    Represents a single row in the physical emission factor table,
    mapping a material to a cradle-to-gate emission factor in
    kgCO2e per kg.

    Attributes:
        material_key: Unique key for the material.
        material_name: Human-readable material name.
        factor_kgco2e_per_kg: Emission factor in kgCO2e per kg.
        source: Source database for the factor.
        source_year: Year of the source data.
        region: Geographic region the factor applies to.
        material_category: Category of the material.
        includes_transport: Whether EF includes transport to gate.
        system_boundary: System boundary description.
        uncertainty_pct: Uncertainty percentage (+/-).
        last_updated: Date the factor was last updated.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    material_key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique key for the material",
    )
    material_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable material name",
    )
    factor_kgco2e_per_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor in kgCO2e per kg",
    )
    source: PhysicalEFSource = Field(
        ...,
        description="Source database for the factor",
    )
    source_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Year of the source data",
    )
    region: str = Field(
        default="GLOBAL",
        max_length=20,
        description="Geographic region the factor applies to",
    )
    material_category: Optional[MaterialCategory] = Field(
        default=None,
        description="Category of the material",
    )
    includes_transport: bool = Field(
        default=True,
        description="Whether EF includes transport to gate",
    )
    system_boundary: str = Field(
        default="cradle_to_gate",
        max_length=50,
        description="System boundary description",
    )
    uncertainty_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Uncertainty percentage (+/-)",
    )
    last_updated: Optional[date] = Field(
        default=None,
        description="Date the factor was last updated",
    )

# ---------------------------------------------------------------------------
# 11. SupplierEF
# ---------------------------------------------------------------------------

class SupplierEF(GreenLangBase):
    """A supplier-specific emission factor entry.

    Represents emission data provided by a specific supplier for
    their products or facilities. Highest quality EF source.

    Attributes:
        supplier_id: Unique identifier of the supplier.
        supplier_name: Name of the supplier.
        product_name: Product or product group name.
        factor_kgco2e_per_unit: Emission factor per product unit.
        factor_unit: Unit of the EF denominator.
        data_source: Source of the supplier data.
        verification_status: Third-party verification status.
        epd_number: EPD registration number if applicable.
        boundary: System boundary of the data.
        reporting_year: Year the data was reported.
        region: Geographic region of the supplier facility.
        confidence_level: Confidence level percentage.
        last_updated: Date the factor was last updated.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique identifier of the supplier",
    )
    supplier_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Name of the supplier",
    )
    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product or product group name",
    )
    factor_kgco2e_per_unit: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor per unit of product",
    )
    factor_unit: str = Field(
        default="kg",
        max_length=50,
        description="Unit of the emission factor denominator",
    )
    data_source: SupplierDataSource = Field(
        ...,
        description="Source of the supplier data",
    )
    verification_status: str = Field(
        default="unverified",
        max_length=50,
        description="Third-party verification status",
    )
    epd_number: Optional[str] = Field(
        default=None,
        max_length=100,
        description="EPD registration number if applicable",
    )
    boundary: str = Field(
        default="cradle_to_gate",
        max_length=50,
        description="System boundary of the data",
    )
    reporting_year: Optional[int] = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Year the data was reported",
    )
    region: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Geographic region of the supplier facility",
    )
    confidence_level: Decimal = Field(
        default=Decimal("95.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Confidence level percentage",
    )
    last_updated: Optional[date] = Field(
        default=None,
        description="Date the factor was last updated",
    )

# ---------------------------------------------------------------------------
# 12. DQIAssessment
# ---------------------------------------------------------------------------

class DQIAssessment(GreenLangBase):
    """Data quality indicator assessment for a calculation result.

    Scores data quality across the five GHG Protocol dimensions
    and computes a composite score (arithmetic mean). Lower scores
    indicate higher quality.

    Attributes:
        item_id: Reference to the procurement item or calculation.
        calculation_method: Calculation method used.
        temporal_score: Temporal representativeness score (1-5).
        geographical_score: Geographical representativeness (1-5).
        technological_score: Technological representativeness (1-5).
        completeness_score: Data completeness score (1-5).
        reliability_score: Data reliability score (1-5).
        composite_score: Arithmetic mean of all five scores.
        quality_tier: Qualitative quality tier label.
        uncertainty_factor: Combined pedigree uncertainty factor.
        findings: List of findings and recommendations.
        ef_hierarchy_level: EF hierarchy level used (1-8).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the procurement item",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Calculation method used",
    )
    temporal_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Temporal representativeness score (1-5)",
    )
    geographical_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Geographical representativeness score (1-5)",
    )
    technological_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Technological representativeness score (1-5)",
    )
    completeness_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Data completeness score (1-5)",
    )
    reliability_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Data reliability score (1-5)",
    )
    composite_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Arithmetic mean of all five dimension scores",
    )
    quality_tier: str = Field(
        default="",
        max_length=50,
        description="Qualitative quality tier label",
    )
    uncertainty_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("1.0"),
        description="Combined pedigree uncertainty factor",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="List of findings and recommendations",
    )
    ef_hierarchy_level: int = Field(
        default=8,
        ge=1,
        le=8,
        description="EF hierarchy level used (1=best, 8=worst)",
    )

# ---------------------------------------------------------------------------
# 13. MaterialityItem
# ---------------------------------------------------------------------------

class MaterialityItem(GreenLangBase):
    """A single item in the hot-spot materiality analysis.

    Represents one procurement category or supplier ranked by
    emission contribution for Pareto analysis and materiality
    quadrant classification.

    Attributes:
        category: Category or supplier identifier.
        category_name: Human-readable category name.
        emissions_tco2e: Emissions in tonnes CO2e.
        emissions_pct: Percentage of total Category 1 emissions.
        cumulative_pct: Cumulative percentage for Pareto ranking.
        spend_usd: Total spend in USD for this category.
        spend_pct: Percentage of total spend.
        ef_intensity_kgco2e_per_usd: Emission intensity per USD.
        quadrant: Materiality quadrant classification.
        recommended_method: Recommended calculation method.
        rank: Pareto rank (1 = highest emitter).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    category: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Category or supplier identifier",
    )
    category_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable category name",
    )
    emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emissions in tonnes CO2e",
    )
    emissions_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of total Category 1 emissions",
    )
    cumulative_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Cumulative percentage for Pareto ranking",
    )
    spend_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total spend in USD for this category",
    )
    spend_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of total spend",
    )
    ef_intensity_kgco2e_per_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emission intensity per USD of spend",
    )
    quadrant: str = Field(
        default="low_priority",
        max_length=50,
        description=(
            "Materiality quadrant: prioritize, monitor, "
            "improve_data, or low_priority"
        ),
    )
    recommended_method: CalculationMethod = Field(
        default=CalculationMethod.SPEND_BASED,
        description="Recommended calculation method for this category",
    )
    rank: int = Field(
        default=0,
        ge=0,
        description="Pareto rank (1 = highest emitter)",
    )

# ---------------------------------------------------------------------------
# 14. CoverageReport
# ---------------------------------------------------------------------------

class CoverageReport(GreenLangBase):
    """Method coverage analysis for the Category 1 inventory.

    Summarizes the breakdown of spend and emissions by calculation
    method, including coverage percentages and gap identification.

    Attributes:
        total_spend_usd: Total procurement spend in USD.
        supplier_specific_spend_usd: Spend covered by supplier data.
        average_data_spend_usd: Spend covered by average-data.
        spend_based_spend_usd: Spend covered by spend-based.
        uncovered_spend_usd: Spend not covered by any method.
        supplier_specific_pct: Supplier-specific coverage percentage.
        average_data_pct: Average-data coverage percentage.
        spend_based_pct: Spend-based coverage percentage.
        total_coverage_pct: Total coverage percentage.
        coverage_level: Qualitative coverage level.
        gap_categories: List of categories with no coverage.
        coverage_by_category: Coverage breakdown by category.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    total_spend_usd: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total procurement spend in USD",
    )
    supplier_specific_spend_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Spend covered by supplier-specific data",
    )
    average_data_spend_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Spend covered by average-data method",
    )
    spend_based_spend_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Spend covered by spend-based method",
    )
    uncovered_spend_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Spend not covered by any method",
    )
    supplier_specific_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Supplier-specific coverage percentage",
    )
    average_data_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Average-data coverage percentage",
    )
    spend_based_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Spend-based coverage percentage",
    )
    total_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Total coverage percentage",
    )
    coverage_level: CoverageLevel = Field(
        default=CoverageLevel.MINIMAL,
        description="Qualitative coverage level",
    )
    gap_categories: List[str] = Field(
        default_factory=list,
        description="List of categories with no coverage",
    )
    coverage_by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Coverage percentage breakdown by category",
    )

# ---------------------------------------------------------------------------
# 15. ComplianceRequirement
# ---------------------------------------------------------------------------

class ComplianceRequirement(GreenLangBase):
    """A single compliance requirement for a regulatory framework.

    Represents one data point or disclosure requirement that must
    be satisfied for compliance with a specific framework.

    Attributes:
        framework: The regulatory framework.
        requirement_id: Machine-readable requirement identifier.
        description: Human-readable description of the requirement.
        is_mandatory: Whether this requirement is mandatory.
        is_met: Whether this requirement has been satisfied.
        evidence: Evidence or data supporting compliance.
        gap_description: Description of the gap if not met.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    framework: ComplianceFramework = Field(
        ...,
        description="The regulatory framework",
    )
    requirement_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Machine-readable requirement identifier",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Human-readable description of the requirement",
    )
    is_mandatory: bool = Field(
        default=True,
        description="Whether this requirement is mandatory",
    )
    is_met: bool = Field(
        default=False,
        description="Whether this requirement has been satisfied",
    )
    evidence: str = Field(
        default="",
        max_length=2000,
        description="Evidence or data supporting compliance",
    )
    gap_description: str = Field(
        default="",
        max_length=2000,
        description="Description of the gap if not met",
    )

# ---------------------------------------------------------------------------
# 16. ComplianceCheckResult
# ---------------------------------------------------------------------------

class ComplianceCheckResult(GreenLangBase):
    """Result of a compliance check against one regulatory framework.

    Aggregates the individual requirement results into an overall
    compliance status for a specific framework.

    Attributes:
        framework: The regulatory framework checked.
        status: Overall compliance status.
        total_requirements: Total number of requirements checked.
        met_requirements: Number of requirements met.
        unmet_requirements: Number of requirements not met.
        compliance_pct: Percentage of requirements met.
        requirements: Detailed results for each requirement.
        recommendations: List of recommendations for improvement.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    framework: ComplianceFramework = Field(
        ...,
        description="The regulatory framework checked",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status",
    )
    total_requirements: int = Field(
        default=0,
        ge=0,
        description="Total number of requirements checked",
    )
    met_requirements: int = Field(
        default=0,
        ge=0,
        description="Number of requirements met",
    )
    unmet_requirements: int = Field(
        default=0,
        ge=0,
        description="Number of requirements not met",
    )
    compliance_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of requirements met",
    )
    requirements: List[ComplianceRequirement] = Field(
        default_factory=list,
        description="Detailed results for each requirement",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of recommendations for improvement",
    )

# ---------------------------------------------------------------------------
# 17. CalculationRequest
# ---------------------------------------------------------------------------

class CalculationRequest(GreenLangBase):
    """Main calculation request for Category 1 emissions.

    Primary input to the pipeline. Contains procurement items,
    configuration for calculation methods, and options for
    compliance checking and export.

    Attributes:
        request_id: Unique identifier for this request.
        tenant_id: Tenant identifier for multi-tenancy.
        items: List of procurement items to calculate.
        calculation_method: Preferred calculation method.
        eeio_database: EEIO database for spend-based method.
        gwp_source: IPCC AR version for GWP values.
        base_currency: Base currency for spend normalization.
        reporting_year: Reporting year for temporal DQI scoring.
        period_start: Start date of the reporting period.
        period_end: End date of the reporting period.
        facility_id: Facility identifier for site-level reporting.
        compliance_frameworks: Frameworks to check against.
        include_dqi: Whether to score data quality.
        include_hotspot: Whether to run hot-spot analysis.
        include_boundary_check: Whether to run boundary checks.
        uncertainty_method: Method for uncertainty quantification.
        export_format: Requested export format.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy",
    )
    items: List[ProcurementItem] = Field(
        ...,
        min_length=1,
        description="List of procurement items to calculate",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.HYBRID,
        description="Preferred calculation method",
    )
    eeio_database: EEIODatabase = Field(
        default=EEIODatabase.EPA_USEEIO,
        description="EEIO database for spend-based method",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR5,
        description="IPCC AR version for GWP values",
    )
    base_currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Base currency for spend normalization",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year for temporal DQI scoring",
    )
    period_start: date = Field(
        ...,
        description="Start date of the reporting period",
    )
    period_end: date = Field(
        ...,
        description="End date of the reporting period",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Facility identifier for site-level reporting",
    )
    compliance_frameworks: Optional[List[ComplianceFramework]] = Field(
        default=None,
        description="Frameworks to check compliance against",
    )
    include_dqi: bool = Field(
        default=True,
        description="Whether to score data quality",
    )
    include_hotspot: bool = Field(
        default=True,
        description="Whether to run hot-spot analysis",
    )
    include_boundary_check: bool = Field(
        default=True,
        description="Whether to run category boundary checks",
    )
    uncertainty_method: UncertaintyMethod = Field(
        default=UncertaintyMethod.PEDIGREE_MATRIX,
        description="Method for uncertainty quantification",
    )
    export_format: Optional[ReportFormat] = Field(
        default=None,
        description="Requested export format",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

    @field_validator("items")
    @classmethod
    def _validate_items_count(
        cls, v: List[ProcurementItem]
    ) -> List[ProcurementItem]:
        """Validate that items do not exceed maximum."""
        if len(v) > MAX_PROCUREMENT_ITEMS:
            raise ValueError(
                f"Maximum {MAX_PROCUREMENT_ITEMS} items per "
                f"request, got {len(v)}"
            )
        return v

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: date, info: Any) -> date:
        """Validate that period_end is on or after period_start."""
        start = info.data.get("period_start")
        if start is not None and v < start:
            raise ValueError(
                f"period_end ({v}) must be on or after "
                f"period_start ({start})"
            )
        return v

    @field_validator("compliance_frameworks")
    @classmethod
    def _validate_frameworks_count(
        cls, v: Optional[List[ComplianceFramework]]
    ) -> Optional[List[ComplianceFramework]]:
        """Validate that frameworks do not exceed maximum."""
        if v is not None and len(v) > MAX_FRAMEWORKS:
            raise ValueError(
                f"Maximum {MAX_FRAMEWORKS} frameworks per "
                f"request, got {len(v)}"
            )
        return v

# ---------------------------------------------------------------------------
# 18. CalculationResult
# ---------------------------------------------------------------------------

class CalculationResult(GreenLangBase):
    """Complete output of a Category 1 emission calculation.

    The primary output of the calculation pipeline, containing
    aggregated emissions, line-item details, data quality scores,
    hot-spot analysis, compliance results, and provenance chain.

    Attributes:
        calculation_id: Unique identifier for this calculation.
        request_id: Reference to the originating request.
        tenant_id: Tenant identifier.
        status: Calculation status.
        calculation_method: Method used (or HYBRID for combined).
        period: Human-readable period label.
        period_start: Start date of the reporting period.
        period_end: End date of the reporting period.
        hybrid_result: Aggregated hybrid result with totals.
        spend_based_results: Line-item spend-based results.
        average_data_results: Line-item average-data results.
        supplier_specific_results: Line-item supplier results.
        dqi_assessments: DQI assessments per line item.
        hotspot_analysis: Hot-spot analysis if requested.
        coverage_report: Coverage analysis report.
        compliance_results: Compliance check results per framework.
        boundary_checks: Category boundary check results.
        provenance_hash: SHA-256 hash over the entire result.
        timestamp: UTC timestamp of calculation completion.
        processing_time_ms: Total processing duration.
        pipeline_stages_completed: Completed pipeline stages.
        warnings: List of warning messages.
        errors: List of error messages.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this calculation",
    )
    request_id: str = Field(
        default="",
        max_length=200,
        description="Reference to the originating request",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier",
    )
    status: BatchStatus = Field(
        default=BatchStatus.PENDING,
        description="Calculation status",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.HYBRID,
        description="Method used (or HYBRID for combined)",
    )
    period: str = Field(
        default="",
        max_length=100,
        description="Human-readable period label",
    )
    period_start: Optional[date] = Field(
        default=None,
        description="Start date of the reporting period",
    )
    period_end: Optional[date] = Field(
        default=None,
        description="End date of the reporting period",
    )
    hybrid_result: Optional[HybridResult] = Field(
        default=None,
        description="Aggregated hybrid result with totals",
    )
    spend_based_results: List[SpendBasedResult] = Field(
        default_factory=list,
        description="Line-item spend-based results",
    )
    average_data_results: List[AverageDataResult] = Field(
        default_factory=list,
        description="Line-item average-data results",
    )
    supplier_specific_results: List[SupplierSpecificResult] = Field(
        default_factory=list,
        description="Line-item supplier-specific results",
    )
    dqi_assessments: List[DQIAssessment] = Field(
        default_factory=list,
        description="DQI assessments per line item",
    )
    hotspot_analysis: Optional[HotSpotAnalysis] = Field(
        default=None,
        description="Hot-spot analysis if requested",
    )
    coverage_report: Optional[CoverageReport] = Field(
        default=None,
        description="Coverage analysis report",
    )
    compliance_results: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="Compliance check results per framework",
    )
    boundary_checks: List[CategoryBoundaryCheck] = Field(
        default_factory=list,
        description="Category boundary check results",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash over the entire result",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of calculation completion",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )
    pipeline_stages_completed: List[str] = Field(
        default_factory=list,
        description="List of completed pipeline stage names",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

# ---------------------------------------------------------------------------
# 19. BatchRequest
# ---------------------------------------------------------------------------

class BatchRequest(GreenLangBase):
    """Request to perform calculations across multiple periods.

    Enables batch processing of Category 1 calculations for
    multiple reporting periods in a single request.

    Attributes:
        batch_id: Unique identifier for this batch job.
        tenant_id: Tenant identifier for multi-tenancy.
        periods: List of (period_start, period_end) date pairs.
        items: Procurement items (shared across all periods).
        calculation_method: Preferred calculation method.
        eeio_database: EEIO database for spend-based method.
        gwp_source: IPCC AR version for GWP values.
        compliance_frameworks: Frameworks to check against.
        include_dqi: Whether to score data quality.
        include_hotspot: Whether to run hot-spot analysis.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch job identifier",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy",
    )
    periods: List[Dict[str, date]] = Field(
        ...,
        min_length=1,
        description=(
            "List of period definitions; each dict must contain "
            "'period_start' and 'period_end' date keys"
        ),
    )
    items: List[ProcurementItem] = Field(
        ...,
        min_length=1,
        description="Procurement items (shared across all periods)",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.HYBRID,
        description="Preferred calculation method",
    )
    eeio_database: EEIODatabase = Field(
        default=EEIODatabase.EPA_USEEIO,
        description="EEIO database for spend-based method",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR5,
        description="IPCC AR version for GWP values",
    )
    compliance_frameworks: Optional[List[ComplianceFramework]] = Field(
        default=None,
        description="Frameworks to check against",
    )
    include_dqi: bool = Field(
        default=True,
        description="Whether to score data quality",
    )
    include_hotspot: bool = Field(
        default=True,
        description="Whether to run hot-spot analysis",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

    @field_validator("periods")
    @classmethod
    def _validate_periods(
        cls, v: List[Dict[str, date]]
    ) -> List[Dict[str, date]]:
        """Validate periods count and structure."""
        if len(v) > MAX_BATCH_PERIODS:
            raise ValueError(
                f"Maximum {MAX_BATCH_PERIODS} periods per batch, "
                f"got {len(v)}"
            )
        for idx, period in enumerate(v):
            if "period_start" not in period or "period_end" not in period:
                raise ValueError(
                    f"Period at index {idx} must contain "
                    f"'period_start' and 'period_end' keys"
                )
            if period["period_end"] < period["period_start"]:
                raise ValueError(
                    f"Period at index {idx}: period_end "
                    f"({period['period_end']}) must be on or after "
                    f"period_start ({period['period_start']})"
                )
        return v

# ---------------------------------------------------------------------------
# 20. BatchResult
# ---------------------------------------------------------------------------

class BatchResult(GreenLangBase):
    """Result of a batch calculation across multiple periods.

    Attributes:
        batch_id: Unique identifier of the batch job.
        status: Overall batch status.
        total_periods: Total number of periods in the batch.
        completed: Number of periods completed successfully.
        failed: Number of periods that failed.
        results: List of calculation results per period.
        failed_periods: List of failed period labels with errors.
        total_emissions_tco2e: Sum of emissions across all periods.
        processing_time_ms: Total processing duration.
        timestamp: UTC timestamp of batch completion.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    batch_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the batch job",
    )
    status: BatchStatus = Field(
        default=BatchStatus.PENDING,
        description="Overall batch status",
    )
    total_periods: int = Field(
        default=0,
        ge=0,
        description="Total number of periods in the batch",
    )
    completed: int = Field(
        default=0,
        ge=0,
        description="Number of periods completed successfully",
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Number of periods that failed",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of calculation results per period",
    )
    failed_periods: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of failed period labels with errors",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Sum of emissions across all periods",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of batch completion",
    )

# ---------------------------------------------------------------------------
# 21. ExportRequest
# ---------------------------------------------------------------------------

class ExportRequest(GreenLangBase):
    """Request to export calculation results in a specific format.

    Attributes:
        calculation_id: Identifier of the calculation to export.
        export_format: Desired export format.
        include_details: Whether to include line-item details.
        include_dqi: Whether to include DQI assessments.
        include_compliance: Whether to include compliance results.
        include_hotspot: Whether to include hot-spot analysis.
        filename_prefix: Optional filename prefix for the export.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the calculation to export",
    )
    export_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Desired export format",
    )
    include_details: bool = Field(
        default=True,
        description="Whether to include line-item details",
    )
    include_dqi: bool = Field(
        default=True,
        description="Whether to include DQI assessments",
    )
    include_compliance: bool = Field(
        default=True,
        description="Whether to include compliance results",
    )
    include_hotspot: bool = Field(
        default=True,
        description="Whether to include hot-spot analysis",
    )
    filename_prefix: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional filename prefix for the export",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

# ---------------------------------------------------------------------------
# 22. AggregationResult
# ---------------------------------------------------------------------------

class AggregationResult(GreenLangBase):
    """Multi-facility or multi-period aggregation of Category 1 results.

    Combines multiple calculation results into a single aggregated
    view with totals, averages, and breakdowns.

    Attributes:
        aggregation_id: Unique identifier for this aggregation.
        tenant_id: Tenant identifier.
        total_emissions_tco2e: Aggregated total emissions.
        total_spend_usd: Aggregated total spend.
        facility_count: Number of facilities included.
        period_count: Number of periods included.
        weighted_dqi: Emission-weighted DQI across all results.
        coverage_level: Overall coverage level.
        by_facility: Emissions breakdown by facility.
        by_method: Emissions breakdown by calculation method.
        by_category: Emissions breakdown by material category.
        intensity_per_revenue: Emission intensity per $M revenue.
        intensity_per_fte: Emission intensity per FTE.
        intensity_per_spend: Emission intensity per $M spend.
        provenance_hash: SHA-256 hash of the aggregation.
        timestamp: UTC timestamp of aggregation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    aggregation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this aggregation",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregated total emissions in tCO2e",
    )
    total_spend_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregated total spend in USD",
    )
    facility_count: int = Field(
        default=0,
        ge=0,
        description="Number of facilities included",
    )
    period_count: int = Field(
        default=0,
        ge=0,
        description="Number of periods included",
    )
    weighted_dqi: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Emission-weighted DQI across all results",
    )
    coverage_level: CoverageLevel = Field(
        default=CoverageLevel.MINIMAL,
        description="Overall coverage level",
    )
    by_facility: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by facility (tCO2e)",
    )
    by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by calculation method",
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by material category",
    )
    intensity_per_revenue: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Emission intensity (tCO2e per $M revenue)",
    )
    intensity_per_fte: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Emission intensity (tCO2e per FTE)",
    )
    intensity_per_spend: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Emission intensity (tCO2e per $M spend)",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the aggregation",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of aggregation",
    )

# ---------------------------------------------------------------------------
# 23. HotSpotAnalysis
# ---------------------------------------------------------------------------

class HotSpotAnalysis(GreenLangBase):
    """Pareto hot-spot analysis of Category 1 emission contributors.

    Identifies the top procurement categories and suppliers by
    emission contribution using 80/20 Pareto analysis and assigns
    materiality quadrant classifications.

    Attributes:
        calculation_id: Reference to the calculation.
        total_emissions_tco2e: Total emissions analysed.
        total_categories: Number of categories analysed.
        top_80_pct_count: Number of categories in the top 80%.
        items: Ranked materiality items.
        quadrant_summary: Count of items by materiality quadrant.
        recommendations: List of prioritization recommendations.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the calculation",
    )
    total_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions analysed in tCO2e",
    )
    total_categories: int = Field(
        default=0,
        ge=0,
        description="Number of categories analysed",
    )
    top_80_pct_count: int = Field(
        default=0,
        ge=0,
        description="Number of categories in the top 80%",
    )
    items: List[MaterialityItem] = Field(
        default_factory=list,
        description="Ranked materiality items",
    )
    quadrant_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of items by materiality quadrant",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of prioritization recommendations",
    )

    @field_validator("items")
    @classmethod
    def _validate_items_count(
        cls, v: List[MaterialityItem]
    ) -> List[MaterialityItem]:
        """Validate that hot-spot items do not exceed maximum."""
        if len(v) > MAX_HOTSPOT_ITEMS:
            raise ValueError(
                f"Maximum {MAX_HOTSPOT_ITEMS} hot-spot items, "
                f"got {len(v)}"
            )
        return v

# ---------------------------------------------------------------------------
# 24. CategoryBoundaryCheck
# ---------------------------------------------------------------------------

class CategoryBoundaryCheck(GreenLangBase):
    """Result of a category boundary check for double-counting prevention.

    Checks whether a procurement item should be excluded from
    Category 1 because it belongs to another Scope 3 category
    (Category 2 capital goods, Category 3 fuel/energy, Category 4
    transport, Category 6 business travel, etc.) or represents
    an intercompany transaction or credit/return.

    Attributes:
        item_id: Reference to the procurement item checked.
        excluded: Whether the item was excluded from Category 1.
        exclusion_reason: Reason for exclusion if applicable.
        target_category: The Scope 3 category the item belongs
            to if excluded.
        confidence: Confidence level of the boundary decision.
        description: Human-readable explanation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the procurement item checked",
    )
    excluded: bool = Field(
        ...,
        description="Whether the item was excluded from Category 1",
    )
    exclusion_reason: str = Field(
        default="",
        max_length=500,
        description="Reason for exclusion if applicable",
    )
    target_category: Optional[str] = Field(
        default=None,
        max_length=100,
        description=(
            "The Scope 3 category the item belongs to if "
            "excluded (e.g. Category 2, Category 3)"
        ),
    )
    confidence: Decimal = Field(
        default=Decimal("100.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Confidence level of the boundary decision",
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Human-readable explanation",
    )

# ---------------------------------------------------------------------------
# 25. PipelineContext
# ---------------------------------------------------------------------------

class PipelineContext(GreenLangBase):
    """Pipeline execution context carrying state across stages.

    Holds the mutable pipeline state as the calculation request
    progresses through the ten pipeline stages. Each stage reads
    from and writes to this context.

from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import ReportFormat

    Attributes:
        pipeline_id: Unique identifier for this pipeline run.
        request: The originating calculation request.
        current_stage: Current pipeline stage being executed.
        completed_stages: List of completed stage names.
        spend_records: Enriched spend records after classification.
        physical_records: Enriched physical records.
        supplier_records: Enriched supplier records.
        excluded_items: Items excluded by boundary checks.
        spend_results: Spend-based calculation results.
        avgdata_results: Average-data calculation results.
        supplier_results: Supplier-specific calculation results.
        hybrid_result: Aggregated hybrid result.
        dqi_assessments: DQI assessment results.
        hotspot: Hot-spot analysis result.
        coverage: Coverage analysis result.
        compliance_results: Compliance check results.
        boundary_checks: Boundary check results.
        warnings: Accumulated warnings.
        errors: Accumulated errors.
        start_time: Pipeline start timestamp.
        stage_timings: Duration per stage in milliseconds.
        provenance_inputs: Inputs for provenance hash computation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this pipeline run",
    )
    request: Optional[CalculationRequest] = Field(
        default=None,
        description="The originating calculation request",
    )
    current_stage: Optional[PipelineStage] = Field(
        default=None,
        description="Current pipeline stage being executed",
    )
    completed_stages: List[str] = Field(
        default_factory=list,
        description="List of completed stage names",
    )
    spend_records: List[SpendRecord] = Field(
        default_factory=list,
        description="Enriched spend records after classification",
    )
    physical_records: List[PhysicalRecord] = Field(
        default_factory=list,
        description="Enriched physical records",
    )
    supplier_records: List[SupplierRecord] = Field(
        default_factory=list,
        description="Enriched supplier records",
    )
    excluded_items: List[ProcurementItem] = Field(
        default_factory=list,
        description="Items excluded by boundary checks",
    )
    spend_results: List[SpendBasedResult] = Field(
        default_factory=list,
        description="Spend-based calculation results",
    )
    avgdata_results: List[AverageDataResult] = Field(
        default_factory=list,
        description="Average-data calculation results",
    )
    supplier_results: List[SupplierSpecificResult] = Field(
        default_factory=list,
        description="Supplier-specific calculation results",
    )
    hybrid_result: Optional[HybridResult] = Field(
        default=None,
        description="Aggregated hybrid result",
    )
    dqi_assessments: List[DQIAssessment] = Field(
        default_factory=list,
        description="DQI assessment results",
    )
    hotspot: Optional[HotSpotAnalysis] = Field(
        default=None,
        description="Hot-spot analysis result",
    )
    coverage: Optional[CoverageReport] = Field(
        default=None,
        description="Coverage analysis result",
    )
    compliance_results: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="Compliance check results",
    )
    boundary_checks: List[CategoryBoundaryCheck] = Field(
        default_factory=list,
        description="Boundary check results",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Accumulated warnings across stages",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Accumulated errors across stages",
    )
    start_time: datetime = Field(
        default_factory=utcnow,
        description="Pipeline start timestamp",
    )
    stage_timings: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Duration per stage in milliseconds",
    )
    provenance_inputs: List[str] = Field(
        default_factory=list,
        description="Inputs for provenance hash computation",
    )

# =============================================================================
# Type Aliases (backward-compatible names)
# =============================================================================

#: Alias for MaterialityItem (backward compatibility).
HotSpotItem = MaterialityItem

#: Alias for ComplianceRequirement (backward compatibility).
ComplianceRule = ComplianceRequirement

# =============================================================================
# __all__ -- Public API
# =============================================================================

__all__ = [
    # Module-level constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "MAX_PROCUREMENT_ITEMS",
    "MAX_BATCH_PERIODS",
    "MAX_FACILITIES",
    "MAX_SUPPLIERS",
    "MAX_FRAMEWORKS",
    "MAX_REQUIREMENTS_PER_FRAMEWORK",
    "MAX_HOTSPOT_ITEMS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DECIMAL_INF",
    "DECIMAL_PLACES",
    "ZERO",
    "ONE",
    "ONE_HUNDRED",
    "ONE_THOUSAND",
    # Enumerations (20)
    "CalculationMethod",
    "SpendClassificationSystem",
    "EEIODatabase",
    "PhysicalEFSource",
    "SupplierDataSource",
    "AllocationMethod",
    "MaterialCategory",
    "CurrencyCode",
    "DQIDimension",
    "DQIScore",
    "UncertaintyMethod",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "ReportFormat",
    "BatchStatus",
    "GWPSource",
    "EmissionGas",
    "ProcurementType",
    "CoverageLevel",
    # Constant tables (9)
    "GWP_VALUES",
    "DQI_SCORE_VALUES",
    "DQI_QUALITY_TIERS",
    "UNCERTAINTY_RANGES",
    "COVERAGE_THRESHOLDS",
    "EF_HIERARCHY_PRIORITY",
    "PEDIGREE_UNCERTAINTY_FACTORS",
    "CURRENCY_EXCHANGE_RATES",
    "INDUSTRY_MARGIN_PERCENTAGES",
    "EEIO_EMISSION_FACTORS",
    "PHYSICAL_EMISSION_FACTORS",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    # Data models (25)
    "ProcurementItem",
    "SpendRecord",
    "PhysicalRecord",
    "SupplierRecord",
    "SpendBasedResult",
    "AverageDataResult",
    "SupplierSpecificResult",
    "HybridResult",
    "EEIOFactor",
    "PhysicalEF",
    "SupplierEF",
    "DQIAssessment",
    "MaterialityItem",
    "CoverageReport",
    "ComplianceRequirement",
    "ComplianceCheckResult",
    "CalculationRequest",
    "CalculationResult",
    "BatchRequest",
    "BatchResult",
    "ExportRequest",
    "AggregationResult",
    "HotSpotAnalysis",
    "CategoryBoundaryCheck",
    "PipelineContext",
    # Type aliases (backward-compatible names)
    "HotSpotItem",
    "ComplianceRule",
    # Helper function
    "_utcnow",
]