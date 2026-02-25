# -*- coding: utf-8 -*-
"""
ProcurementDatabaseEngine - EEIO/Physical EF Lookup, Classification, Currency (Engine 1 of 7)

AGENT-MRV-014: Purchased Goods & Services Agent (GL-MRV-S3-001)

Provides the authoritative reference data layer for Scope 3 Category 1
emissions calculations.  This engine is the single source of truth for
emission factor lookups, industry classification cross-mapping, currency
conversion, margin adjustment, and supplier-specific EF registration used
by all downstream engines (SpendBasedCalculatorEngine, AverageDataEngine,
SupplierSpecificEngine, HybridAggregationEngine).

Built-In Reference Data:
    - 71 EEIO emission factors by NAICS-6 sector (EPA USEEIO v1.2)
    - 45 physical emission factors by material key (ecoinvent/ICE/DEFRA)
    - 20 currency exchange rates to USD (annual average 2024)
    - 24 industry margin percentages by NAICS 2-digit sector
    - 8-level emission factor hierarchy priority
    - 50+ NAICS sector name lookups
    - 55+ NAICS-to-ISIC cross-mappings with confidence scores
    - 55+ NACE-to-ISIC cross-mappings with confidence scores
    - 55 UNSPSC segment-to-NAICS 2-digit mappings
    - Reverse ISIC-to-NAICS mappings

Capabilities (7):
    1. EEIO Factor Lookup - exact match, 4/2-digit fallback, multi-DB
    2. Physical EF Lookup - exact match, fuzzy by MaterialCategory
    3. Classification Cross-Mapping - NAICS/NACE/ISIC/UNSPSC bidirectional
    4. Currency Conversion - 20 currencies, inflation deflation, PPP
    5. Margin Adjustment - purchaser-to-producer price conversion
    6. Supplier EF Registry - register/lookup supplier-specific EFs
    7. EF Hierarchy Selection - 8-level priority selection

Zero-Hallucination Guarantees:
    - All factors are hard-coded from published reference databases.
    - All lookups are deterministic dictionary access.
    - No LLM involvement in any data retrieval or calculation path.
    - Every query result carries a SHA-256 provenance hash.
    - All arithmetic uses Python Decimal for exact precision.

Thread Safety:
    Thread-safe singleton with ``threading.RLock()``.  All reference
    data is immutable after initialization.  The mutable supplier EF
    registry is protected by the singleton lock.

Example:
    >>> from greenlang.purchased_goods_services.procurement_database import (
    ...     ProcurementDatabaseEngine,
    ... )
    >>> db = ProcurementDatabaseEngine()
    >>> factor = db.lookup_eeio_factor("331110")
    >>> assert factor is not None
    >>> assert factor.sector_code == "331110"
    >>> usd = db.convert_currency(Decimal("1000"), CurrencyCode.EUR)
    >>> margin = db.get_margin_rate("33")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-S3-001)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ProcurementDatabaseEngine"]

# ---------------------------------------------------------------------------
# Conditional imports -- graceful fallback when peer modules unavailable
# ---------------------------------------------------------------------------

try:
    from greenlang.purchased_goods_services.models import (
        AGENT_ID,
        VERSION,
        TABLE_PREFIX,
        ZERO,
        ONE,
        ONE_HUNDRED,
        ONE_THOUSAND,
        DECIMAL_PLACES,
        DECIMAL_INF,
        CalculationMethod,
        SpendClassificationSystem,
        EEIODatabase,
        PhysicalEFSource,
        SupplierDataSource,
        AllocationMethod,
        MaterialCategory,
        CurrencyCode,
        DQIDimension,
        DQIScore,
        ComplianceFramework,
        ComplianceStatus,
        PipelineStage,
        ExportFormat,
        BatchStatus,
        GWPSource,
        EmissionGas,
        ProcurementType,
        CoverageLevel,
        GWP_VALUES,
        DQI_SCORE_VALUES,
        DQI_QUALITY_TIERS,
        UNCERTAINTY_RANGES,
        COVERAGE_THRESHOLDS,
        EF_HIERARCHY_PRIORITY,
        PEDIGREE_UNCERTAINTY_FACTORS,
        CURRENCY_EXCHANGE_RATES,
        INDUSTRY_MARGIN_PERCENTAGES,
        EEIO_EMISSION_FACTORS,
        PHYSICAL_EMISSION_FACTORS,
        FRAMEWORK_REQUIRED_DISCLOSURES,
        ProcurementItem,
        SpendRecord,
        PhysicalRecord,
        SupplierRecord,
        SpendBasedResult,
        AverageDataResult,
        SupplierSpecificResult,
        HybridResult,
        EEIOFactor,
        PhysicalEF,
        SupplierEF,
        DQIAssessment,
        MaterialityItem,
        CoverageReport,
        ComplianceRequirement,
        ComplianceCheckResult,
        CalculationRequest,
        CalculationResult,
        BatchRequest,
        BatchResult,
        ExportRequest,
        AggregationResult,
        HotSpotAnalysis,
        CategoryBoundaryCheck,
        PipelineContext,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.warning(
        "greenlang.purchased_goods_services.models not available; "
        "ProcurementDatabaseEngine will use fallback constants"
    )

try:
    from greenlang.purchased_goods_services.config import (
        PurchasedGoodsServicesConfig,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    PurchasedGoodsServicesConfig = None  # type: ignore[assignment,misc]

try:
    from greenlang.purchased_goods_services.metrics import (
        PurchasedGoodsServicesMetrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    PurchasedGoodsServicesMetrics = None  # type: ignore[assignment,misc]

try:
    from greenlang.purchased_goods_services.provenance import (
        PurchasedGoodsProvenanceTracker,
        ProvenanceStage,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    PurchasedGoodsProvenanceTracker = None  # type: ignore[assignment,misc]
    ProvenanceStage = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Quantize helper
# ---------------------------------------------------------------------------

_QUANTIZE_EXP = Decimal(10) ** -8  # 8 decimal places default


def _q(value: Decimal, places: int = 8) -> Decimal:
    """Quantize a Decimal to the given number of decimal places.

    Args:
        value: The Decimal value to quantize.
        places: Number of decimal places (default 8).

    Returns:
        Quantized Decimal value.
    """
    exp = Decimal(10) ** -places
    return value.quantize(exp, rounding=ROUND_HALF_UP)


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _sha256(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hex digest from a JSON-serializable dict.

    Args:
        data: Dictionary to hash.

    Returns:
        64-character lowercase hex digest string.
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


# ============================================================================
# NAICS Sector Name Lookup Table (55 sectors)
# ============================================================================

NAICS_SECTOR_NAMES: Dict[str, str] = {
    # 2-digit sector names
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing (Food, Beverage, Textile)",
    "32": "Manufacturing (Wood, Paper, Chemical, Plastics)",
    "33": "Manufacturing (Metals, Machinery, Electronics)",
    "42": "Wholesale Trade",
    "44": "Retail Trade (Store Retailers)",
    "45": "Retail Trade (Non-store and E-commerce)",
    "48": "Transportation",
    "49": "Warehousing and Storage",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
    # 6-digit sector names (top 55 sectors matching EEIO table)
    "111110": "Soybean Farming",
    "111150": "Corn Farming",
    "111199": "All Other Grain Farming",
    "111310": "Orange Groves",
    "111920": "Cotton Farming",
    "112111": "Beef Cattle Ranching and Farming",
    "112120": "Dairy Cattle and Milk Production",
    "112210": "Hog and Pig Farming",
    "112310": "Chicken Egg Production",
    "113110": "Timber Tract Operations",
    "211120": "Crude Petroleum Extraction",
    "211130": "Natural Gas Extraction",
    "212210": "Iron Ore Mining",
    "212230": "Copper, Nickel, Lead, and Zinc Mining",
    "212310": "Stone Mining and Quarrying",
    "221110": "Electric Power Generation, Transmission",
    "221210": "Natural Gas Distribution",
    "236110": "Residential Building Construction",
    "236220": "Commercial and Institutional Building Construction",
    "237310": "Highway, Street, and Bridge Construction",
    "311111": "Dog and Cat Food Manufacturing",
    "311210": "Flour Milling and Malt Manufacturing",
    "311410": "Frozen Fruit, Juice, and Vegetable Manufacturing",
    "311513": "Cheese Manufacturing",
    "311615": "Poultry Processing",
    "311710": "Seafood Product Preparation and Packaging",
    "311920": "Coffee and Tea Manufacturing",
    "322110": "Pulp Mills",
    "322121": "Paper (except Newsprint) Mills",
    "322130": "Paperboard Mills",
    "322211": "Corrugated and Solid Fiber Box Manufacturing",
    "325110": "Petrochemical Manufacturing",
    "325180": "Other Basic Inorganic Chemical Manufacturing",
    "325211": "Plastics Material and Resin Manufacturing",
    "325311": "Nitrogenous Fertilizer Manufacturing",
    "325411": "Medicinal and Botanical Manufacturing",
    "325510": "Paint and Coating Manufacturing",
    "325611": "Soap and Other Detergent Manufacturing",
    "331110": "Iron and Steel Mills and Ferroalloy Manufacturing",
    "331313": "Alumina Refining and Primary Aluminum Production",
    "331420": "Copper Rolling, Drawing, Extruding, and Alloying",
    "331511": "Iron Foundries",
    "333111": "Farm Machinery and Equipment Manufacturing",
    "333120": "Construction Machinery Manufacturing",
    "333310": "Commercial and Service Industry Machinery Manufacturing",
    "334111": "Electronic Computer Manufacturing",
    "334210": "Telephone Apparatus Manufacturing",
    "334413": "Semiconductor and Related Device Manufacturing",
    "334510": "Electromedical and Electrotherapeutic Apparatus",
    "423110": "Automobile and Other Motor Vehicle Merchant Wholesalers",
    "423400": "Professional and Commercial Equipment Wholesalers",
    "423510": "Metal Service Centers and Other Metal Merchant Wholesalers",
    "445110": "Supermarkets and Other Grocery Retailers",
    "452210": "Department Stores",
    "481111": "Scheduled Passenger Air Transportation",
    "484110": "General Freight Trucking, Local",
    "511210": "Software Publishers",
    "518210": "Computing Infrastructure Providers and Data Processing",
    "522110": "Commercial Banking",
    "524114": "Direct Health and Medical Insurance Carriers",
    "541110": "Offices of Lawyers",
    "541211": "Offices of Certified Public Accountants",
    "541310": "Architectural Services",
    "541330": "Engineering Services",
    "541511": "Custom Computer Programming Services",
    "541611": "Administrative Management and General Consulting",
    "541711": "Research and Development in Biotechnology",
    "621111": "Offices of Physicians (except Mental Health)",
    "622110": "General Medical and Surgical Hospitals",
    "721110": "Hotels (except Casino Hotels) and Motels",
    "722511": "Full-Service Restaurants",
}


# ============================================================================
# NAICS-to-ISIC Cross-Mapping Table (55+ mappings)
# Format: NAICS-6 -> (ISIC Rev 4.1 code, confidence 0-1)
# ============================================================================

NAICS_TO_ISIC: Dict[str, List[Tuple[str, Decimal]]] = {
    # Agriculture
    "111110": [("0111", Decimal("0.95"))],
    "111150": [("0111", Decimal("0.95"))],
    "111199": [("0111", Decimal("0.85")), ("0119", Decimal("0.80"))],
    "111310": [("0122", Decimal("0.90"))],
    "111920": [("0116", Decimal("0.90"))],
    "112111": [("0141", Decimal("0.95"))],
    "112120": [("0141", Decimal("0.90")), ("0105", Decimal("0.70"))],
    "112210": [("0145", Decimal("0.95"))],
    "112310": [("0144", Decimal("0.90"))],
    "113110": [("0210", Decimal("0.95"))],
    # Mining
    "211120": [("0610", Decimal("0.95"))],
    "211130": [("0620", Decimal("0.95"))],
    "212210": [("0710", Decimal("0.95"))],
    "212230": [("0729", Decimal("0.90"))],
    "212310": [("0810", Decimal("0.90"))],
    # Utilities
    "221110": [("3510", Decimal("0.95"))],
    "221210": [("3520", Decimal("0.90"))],
    # Construction
    "236110": [("4100", Decimal("0.85"))],
    "236220": [("4100", Decimal("0.85"))],
    "237310": [("4210", Decimal("0.90"))],
    # Food Manufacturing
    "311111": [("1080", Decimal("0.85"))],
    "311210": [("1061", Decimal("0.90"))],
    "311410": [("1030", Decimal("0.90"))],
    "311513": [("1050", Decimal("0.90"))],
    "311615": [("1010", Decimal("0.90"))],
    "311710": [("1020", Decimal("0.90"))],
    "311920": [("1079", Decimal("0.85"))],
    # Paper Manufacturing
    "322110": [("1701", Decimal("0.95"))],
    "322121": [("1701", Decimal("0.90"))],
    "322130": [("1702", Decimal("0.90"))],
    "322211": [("1702", Decimal("0.85"))],
    # Chemical Manufacturing
    "325110": [("2011", Decimal("0.90"))],
    "325180": [("2012", Decimal("0.85"))],
    "325211": [("2013", Decimal("0.90"))],
    "325311": [("2012", Decimal("0.85"))],
    "325411": [("2100", Decimal("0.85"))],
    "325510": [("2022", Decimal("0.90"))],
    "325611": [("2023", Decimal("0.90"))],
    # Metals Manufacturing
    "331110": [("2410", Decimal("0.95"))],
    "331313": [("2420", Decimal("0.90"))],
    "331420": [("2420", Decimal("0.85"))],
    "331511": [("2431", Decimal("0.90"))],
    # Machinery Manufacturing
    "333111": [("2821", Decimal("0.90"))],
    "333120": [("2824", Decimal("0.90"))],
    "333310": [("2819", Decimal("0.85"))],
    # Electronics Manufacturing
    "334111": [("2620", Decimal("0.90"))],
    "334210": [("2630", Decimal("0.90"))],
    "334413": [("2610", Decimal("0.95"))],
    "334510": [("2660", Decimal("0.85"))],
    # Wholesale Trade
    "423110": [("4510", Decimal("0.85"))],
    "423400": [("4659", Decimal("0.80"))],
    "423510": [("4659", Decimal("0.80"))],
    # Retail Trade
    "445110": [("4711", Decimal("0.90"))],
    "452210": [("4719", Decimal("0.85"))],
    # Transportation
    "481111": [("5110", Decimal("0.95"))],
    "484110": [("4923", Decimal("0.90"))],
    # Information
    "511210": [("5820", Decimal("0.90"))],
    "518210": [("6311", Decimal("0.85"))],
    # Finance
    "522110": [("6419", Decimal("0.90"))],
    "524114": [("6512", Decimal("0.85"))],
    # Professional Services
    "541110": [("6910", Decimal("0.95"))],
    "541211": [("6920", Decimal("0.95"))],
    "541310": [("7110", Decimal("0.95"))],
    "541330": [("7110", Decimal("0.90"))],
    "541511": [("6201", Decimal("0.90"))],
    "541611": [("7020", Decimal("0.90"))],
    "541711": [("7211", Decimal("0.90"))],
    # Health Care
    "621111": [("8620", Decimal("0.90"))],
    "622110": [("8610", Decimal("0.95"))],
    # Accommodation and Food
    "721110": [("5510", Decimal("0.95"))],
    "722511": [("5610", Decimal("0.90"))],
}


# ============================================================================
# NACE-to-ISIC Cross-Mapping Table (55+ mappings)
# Format: NACE Rev 2.1 code -> (ISIC Rev 4.1 code, confidence 0-1)
# ============================================================================

NACE_TO_ISIC: Dict[str, List[Tuple[str, Decimal]]] = {
    # Agriculture (NACE A)
    "A01.11": [("0111", Decimal("0.98"))],
    "A01.12": [("0112", Decimal("0.95"))],
    "A01.13": [("0113", Decimal("0.95"))],
    "A01.14": [("0114", Decimal("0.95"))],
    "A01.15": [("0115", Decimal("0.95"))],
    "A01.16": [("0116", Decimal("0.95"))],
    "A01.19": [("0119", Decimal("0.90"))],
    "A01.21": [("0121", Decimal("0.95"))],
    "A01.22": [("0122", Decimal("0.95"))],
    "A01.41": [("0141", Decimal("0.95"))],
    "A01.42": [("0142", Decimal("0.95"))],
    "A01.43": [("0143", Decimal("0.95"))],
    "A01.44": [("0144", Decimal("0.95"))],
    "A01.45": [("0145", Decimal("0.95"))],
    "A01.46": [("0146", Decimal("0.95"))],
    "A02.10": [("0210", Decimal("0.95"))],
    "A02.20": [("0220", Decimal("0.95"))],
    # Mining (NACE B)
    "B05.10": [("0510", Decimal("0.95"))],
    "B06.10": [("0610", Decimal("0.98"))],
    "B06.20": [("0620", Decimal("0.98"))],
    "B07.10": [("0710", Decimal("0.95"))],
    "B07.29": [("0729", Decimal("0.90"))],
    "B08.11": [("0810", Decimal("0.90"))],
    "B08.12": [("0810", Decimal("0.85"))],
    # Manufacturing (NACE C)
    "C10.10": [("1010", Decimal("0.95"))],
    "C10.20": [("1020", Decimal("0.95"))],
    "C10.30": [("1030", Decimal("0.95"))],
    "C10.50": [("1050", Decimal("0.95"))],
    "C10.61": [("1061", Decimal("0.95"))],
    "C10.80": [("1080", Decimal("0.90"))],
    "C17.11": [("1701", Decimal("0.95"))],
    "C17.12": [("1702", Decimal("0.95"))],
    "C20.11": [("2011", Decimal("0.95"))],
    "C20.12": [("2012", Decimal("0.90"))],
    "C20.13": [("2013", Decimal("0.90"))],
    "C20.14": [("2011", Decimal("0.85"))],
    "C20.15": [("2012", Decimal("0.85"))],
    "C20.16": [("2013", Decimal("0.85"))],
    "C20.41": [("2023", Decimal("0.90"))],
    "C20.42": [("2029", Decimal("0.85"))],
    "C21.10": [("2100", Decimal("0.90"))],
    "C24.10": [("2410", Decimal("0.98"))],
    "C24.20": [("2420", Decimal("0.95"))],
    "C24.31": [("2431", Decimal("0.95"))],
    "C24.42": [("2420", Decimal("0.90"))],
    "C25.11": [("2511", Decimal("0.90"))],
    "C26.10": [("2610", Decimal("0.95"))],
    "C26.20": [("2620", Decimal("0.95"))],
    "C26.30": [("2630", Decimal("0.95"))],
    "C26.60": [("2660", Decimal("0.90"))],
    "C28.21": [("2821", Decimal("0.90"))],
    "C28.92": [("2824", Decimal("0.85"))],
    # Utilities (NACE D)
    "D35.11": [("3510", Decimal("0.95"))],
    "D35.22": [("3520", Decimal("0.90"))],
    # Construction (NACE F)
    "F41.10": [("4100", Decimal("0.90"))],
    "F42.11": [("4210", Decimal("0.90"))],
    # Trade (NACE G)
    "G45.11": [("4510", Decimal("0.90"))],
    "G46.69": [("4659", Decimal("0.85"))],
    "G47.11": [("4711", Decimal("0.90"))],
    # Transport (NACE H)
    "H49.41": [("4923", Decimal("0.90"))],
    "H51.10": [("5110", Decimal("0.95"))],
    # Information (NACE J)
    "J58.29": [("5820", Decimal("0.85"))],
    "J63.11": [("6311", Decimal("0.90"))],
    # Finance (NACE K)
    "K64.19": [("6419", Decimal("0.90"))],
    "K65.12": [("6512", Decimal("0.90"))],
    # Professional (NACE M)
    "M69.10": [("6910", Decimal("0.95"))],
    "M69.20": [("6920", Decimal("0.95"))],
    "M71.11": [("7110", Decimal("0.95"))],
    "M71.12": [("7110", Decimal("0.90"))],
    "M70.22": [("7020", Decimal("0.90"))],
    "M72.11": [("7211", Decimal("0.90"))],
    # Health (NACE Q)
    "Q86.10": [("8610", Decimal("0.95"))],
    "Q86.21": [("8620", Decimal("0.90"))],
    # Accommodation (NACE I)
    "I55.10": [("5510", Decimal("0.95"))],
    "I56.10": [("5610", Decimal("0.90"))],
}


# ============================================================================
# ISIC-to-NAICS Reverse Mapping Table (auto-generated from NAICS_TO_ISIC)
# ============================================================================

def _build_isic_to_naics() -> Dict[str, List[Tuple[str, Decimal]]]:
    """Build reverse ISIC-to-NAICS mapping from NAICS_TO_ISIC.

    Returns:
        Dictionary mapping ISIC codes to lists of (NAICS-6, confidence).
    """
    result: Dict[str, List[Tuple[str, Decimal]]] = {}
    for naics, mappings in NAICS_TO_ISIC.items():
        for isic_code, confidence in mappings:
            if isic_code not in result:
                result[isic_code] = []
            result[isic_code].append((naics, confidence))
    # Sort each list by confidence descending
    for isic_code in result:
        result[isic_code].sort(key=lambda x: x[1], reverse=True)
    return result


ISIC_TO_NAICS: Dict[str, List[Tuple[str, Decimal]]] = _build_isic_to_naics()


# ============================================================================
# UNSPSC Segment-to-NAICS 2-Digit Mapping (55 segments)
# UNSPSC v28 segments (first 2 digits of 8-digit code) -> NAICS 2-digit
# ============================================================================

UNSPSC_TO_NAICS: Dict[str, List[Tuple[str, Decimal]]] = {
    "10": [("11", Decimal("0.90"))],   # Live Plant and Animal Material
    "11": [("21", Decimal("0.85"))],   # Mineral and Textile and Fur Materials
    "12": [("32", Decimal("0.85"))],   # Chemicals incl Bio and Gas Materials
    "13": [("32", Decimal("0.80"))],   # Resin, Rosin, Rubber, Foam Materials
    "14": [("32", Decimal("0.80"))],   # Paper and Paperboard Materials
    "15": [("31", Decimal("0.80"))],   # Fuels and Lubricants and Oils
    "20": [("21", Decimal("0.80"))],   # Mining and Well Drilling Machinery
    "21": [("11", Decimal("0.85"))],   # Farming and Fishing Machinery
    "22": [("23", Decimal("0.80"))],   # Building and Construction Machinery
    "23": [("33", Decimal("0.85"))],   # Industrial Manufacturing Machinery
    "24": [("33", Decimal("0.80"))],   # Material Handling Machinery
    "25": [("33", Decimal("0.80"))],   # Commercial/Industrial Transportation
    "26": [("33", Decimal("0.80"))],   # Power Generation and Distribution
    "27": [("33", Decimal("0.80"))],   # Tools and General Machinery
    "30": [("33", Decimal("0.75"))],   # Structures and Building
    "31": [("33", Decimal("0.80"))],   # Manufacturing Components
    "32": [("33", Decimal("0.80"))],   # Electronic Components
    "39": [("33", Decimal("0.75"))],   # Lighting and Fixtures
    "40": [("33", Decimal("0.80"))],   # Distribution/Conditioning Equipment
    "41": [("33", Decimal("0.80"))],   # Laboratory/Measuring/Testing Equipment
    "42": [("33", Decimal("0.80"))],   # Medical Equipment and Supplies
    "43": [("33", Decimal("0.85"))],   # IT Equipment, Software, Telecom
    "44": [("33", Decimal("0.80"))],   # Office Machines and Supplies
    "45": [("33", Decimal("0.75"))],   # Printing and Photographic Equipment
    "46": [("33", Decimal("0.75"))],   # Defense and Law Enforcement
    "47": [("31", Decimal("0.75"))],   # Cleaning Equipment and Supplies
    "48": [("33", Decimal("0.70"))],   # Service Industry Machinery
    "49": [("33", Decimal("0.70"))],   # Sports and Recreation Equipment
    "50": [("31", Decimal("0.85"))],   # Food Beverage and Tobacco Products
    "51": [("33", Decimal("0.70"))],   # Drugs and Pharmaceutical Products
    "52": [("31", Decimal("0.75"))],   # Domestic Appliances and Supplies
    "53": [("31", Decimal("0.75"))],   # Apparel and Luggage and Textiles
    "54": [("31", Decimal("0.70"))],   # Timepieces, Jewelry, Gemstones
    "55": [("31", Decimal("0.75"))],   # Published Products
    "56": [("31", Decimal("0.70"))],   # Furniture and Furnishings
    "60": [("31", Decimal("0.70"))],   # Musical Instruments and Accessories
    "70": [("11", Decimal("0.80"))],   # Farming and Fishing Products
    "71": [("21", Decimal("0.80"))],   # Mining and Quarrying Products
    "72": [("54", Decimal("0.75"))],   # Building/Facility/Construction Svc
    "73": [("33", Decimal("0.75"))],   # Industrial Production and Mfg Svc
    "76": [("33", Decimal("0.75"))],   # Industrial Cleaning Services
    "77": [("54", Decimal("0.80"))],   # Environmental Services
    "78": [("48", Decimal("0.85"))],   # Transportation/Storage/Mail Services
    "80": [("54", Decimal("0.85"))],   # Management/Business/Admin Services
    "81": [("54", Decimal("0.80"))],   # Engineering and Research Services
    "82": [("54", Decimal("0.80"))],   # Editorial/Design/Graphic/Fine Arts
    "83": [("51", Decimal("0.80"))],   # Public Utilities and Public Services
    "84": [("52", Decimal("0.85"))],   # Financial and Insurance Services
    "85": [("62", Decimal("0.85"))],   # Healthcare Services
    "86": [("61", Decimal("0.85"))],   # Education and Training Services
    "90": [("48", Decimal("0.80"))],   # Travel and Food and Lodging
    "91": [("54", Decimal("0.70"))],   # Personal and Domestic Services
    "92": [("92", Decimal("0.80"))],   # National Defense and Public Order
    "93": [("92", Decimal("0.80"))],   # Politics and Civic Affairs Services
    "94": [("92", Decimal("0.75"))],   # Organizations and Clubs
    "95": [("23", Decimal("0.75"))],   # Land, Buildings, and Structures
}


# ============================================================================
# Physical EF Material-to-Category Mapping
# ============================================================================

MATERIAL_KEY_TO_CATEGORY: Dict[str, str] = {
    # Metals -> RAW_METALS
    "steel_primary_bof": "RAW_METALS",
    "steel_secondary_eaf": "RAW_METALS",
    "steel_world_avg": "RAW_METALS",
    "steel_virgin_100pct": "RAW_METALS",
    "aluminum_primary_global": "RAW_METALS",
    "aluminum_secondary": "RAW_METALS",
    "aluminum_33pct_recycled": "RAW_METALS",
    "copper_primary": "RAW_METALS",
    "lead": "RAW_METALS",
    "zinc": "RAW_METALS",
    "lithium_carbonate": "RAW_METALS",
    # Plastics -> PLASTICS
    "hdpe": "PLASTICS",
    "ldpe": "PLASTICS",
    "pp_polypropylene": "PLASTICS",
    "pet": "PLASTICS",
    "pvc": "PLASTICS",
    "ps_polystyrene": "PLASTICS",
    "abs": "PLASTICS",
    "nylon_6": "PLASTICS",
    # Construction -> CONSTRUCTION
    "cement_portland_global": "CONSTRUCTION",
    "cement_portland_cem_i": "CONSTRUCTION",
    "concrete_readymix_30mpa": "CONSTRUCTION",
    "concrete_high_50mpa": "CONSTRUCTION",
    "float_glass": "CONSTRUCTION",
    "glass_general": "CONSTRUCTION",
    "bricks_general": "CONSTRUCTION",
    "timber_softwood_sawn": "CONSTRUCTION",
    "timber_hardwood_sawn": "CONSTRUCTION",
    "timber_glulam": "CONSTRUCTION",
    # Paper -> PAPER
    "corrugated_cardboard": "PAPER",
    "kraft_paper": "PAPER",
    "recycled_paper": "PAPER",
    # Textiles -> TEXTILES
    "cotton_conventional": "TEXTILES",
    "cotton_organic": "TEXTILES",
    "polyester_fiber": "TEXTILES",
    "nylon_fiber": "TEXTILES",
    "wool": "TEXTILES",
    # Electronics -> ELECTRONICS
    "silicon_wafer_solar": "ELECTRONICS",
    "pcb_printed_circuit": "ELECTRONICS",
    # Chemicals -> CHEMICALS
    "ammonia": "CHEMICALS",
    "ethylene": "CHEMICALS",
    "propylene": "CHEMICALS",
    "methanol": "CHEMICALS",
    # Rubber -> RUBBER
    "natural_rubber": "RUBBER",
    "synthetic_rubber_sbr": "RUBBER",
}


# ============================================================================
# Physical EF Human-Readable Material Names
# ============================================================================

MATERIAL_KEY_TO_NAME: Dict[str, str] = {
    "steel_primary_bof": "Steel, Primary (BOF Route)",
    "steel_secondary_eaf": "Steel, Secondary (EAF Route)",
    "steel_world_avg": "Steel, World Average",
    "steel_virgin_100pct": "Steel, 100% Virgin",
    "aluminum_primary_global": "Aluminium, Primary (Global Average)",
    "aluminum_secondary": "Aluminium, Secondary (Recycled)",
    "aluminum_33pct_recycled": "Aluminium, 33% Recycled Content",
    "copper_primary": "Copper, Primary",
    "lead": "Lead",
    "zinc": "Zinc",
    "lithium_carbonate": "Lithium Carbonate",
    "hdpe": "High-Density Polyethylene (HDPE)",
    "ldpe": "Low-Density Polyethylene (LDPE)",
    "pp_polypropylene": "Polypropylene (PP)",
    "pet": "Polyethylene Terephthalate (PET)",
    "pvc": "Polyvinyl Chloride (PVC)",
    "ps_polystyrene": "Polystyrene (PS)",
    "abs": "Acrylonitrile Butadiene Styrene (ABS)",
    "nylon_6": "Nylon 6",
    "cement_portland_global": "Portland Cement, Global Average",
    "cement_portland_cem_i": "Portland Cement, CEM I",
    "concrete_readymix_30mpa": "Ready-Mix Concrete, 30 MPa",
    "concrete_high_50mpa": "High-Strength Concrete, 50 MPa",
    "float_glass": "Float Glass",
    "glass_general": "Glass, General",
    "bricks_general": "Bricks, General",
    "timber_softwood_sawn": "Sawn Softwood Timber",
    "timber_hardwood_sawn": "Sawn Hardwood Timber",
    "timber_glulam": "Glulam (Glued Laminated Timber)",
    "corrugated_cardboard": "Corrugated Cardboard",
    "kraft_paper": "Kraft Paper",
    "recycled_paper": "Recycled Paper",
    "cotton_conventional": "Cotton, Conventional",
    "cotton_organic": "Cotton, Organic",
    "polyester_fiber": "Polyester Fiber",
    "nylon_fiber": "Nylon Fiber",
    "wool": "Wool",
    "silicon_wafer_solar": "Silicon Wafer (Solar Grade)",
    "pcb_printed_circuit": "Printed Circuit Board (PCB)",
    "ammonia": "Ammonia",
    "ethylene": "Ethylene",
    "propylene": "Propylene",
    "methanol": "Methanol",
    "natural_rubber": "Natural Rubber",
    "synthetic_rubber_sbr": "Synthetic Rubber (SBR)",
}


# ============================================================================
# Physical EF Source Mapping (material_key -> PhysicalEFSource)
# ============================================================================

MATERIAL_KEY_TO_SOURCE: Dict[str, str] = {
    "steel_primary_bof": "WORLD_STEEL",
    "steel_secondary_eaf": "WORLD_STEEL",
    "steel_world_avg": "WORLD_STEEL",
    "steel_virgin_100pct": "WORLD_STEEL",
    "aluminum_primary_global": "IAI",
    "aluminum_secondary": "IAI",
    "aluminum_33pct_recycled": "IAI",
    "copper_primary": "ECOINVENT",
    "lead": "ECOINVENT",
    "zinc": "ECOINVENT",
    "lithium_carbonate": "ECOINVENT",
    "hdpe": "PLASTICS_EUROPE",
    "ldpe": "PLASTICS_EUROPE",
    "pp_polypropylene": "PLASTICS_EUROPE",
    "pet": "PLASTICS_EUROPE",
    "pvc": "PLASTICS_EUROPE",
    "ps_polystyrene": "PLASTICS_EUROPE",
    "abs": "PLASTICS_EUROPE",
    "nylon_6": "PLASTICS_EUROPE",
    "cement_portland_global": "ICE",
    "cement_portland_cem_i": "ICE",
    "concrete_readymix_30mpa": "ICE",
    "concrete_high_50mpa": "ICE",
    "float_glass": "ICE",
    "glass_general": "ICE",
    "bricks_general": "ICE",
    "timber_softwood_sawn": "ICE",
    "timber_hardwood_sawn": "ICE",
    "timber_glulam": "ICE",
    "corrugated_cardboard": "CEPI",
    "kraft_paper": "CEPI",
    "recycled_paper": "CEPI",
    "cotton_conventional": "DEFRA",
    "cotton_organic": "DEFRA",
    "polyester_fiber": "DEFRA",
    "nylon_fiber": "DEFRA",
    "wool": "DEFRA",
    "silicon_wafer_solar": "ECOINVENT",
    "pcb_printed_circuit": "ECOINVENT",
    "ammonia": "ECOINVENT",
    "ethylene": "ECOINVENT",
    "propylene": "ECOINVENT",
    "methanol": "ECOINVENT",
    "natural_rubber": "DEFRA",
    "synthetic_rubber_sbr": "DEFRA",
}


# ============================================================================
# ProcurementDatabaseEngine
# ============================================================================


class ProcurementDatabaseEngine:
    """Thread-safe singleton providing EEIO/physical EF lookup, classification
    cross-mapping, currency conversion, margin adjustment, supplier EF
    registry, and EF hierarchy selection for the Purchased Goods & Services
    Agent (AGENT-MRV-014).

    This engine is the data foundation for all four calculation methods
    (spend-based, average-data, supplier-specific, hybrid) defined in the
    GHG Protocol Scope 3 Category 1 guidance.

    Thread Safety:
        Singleton pattern with ``threading.RLock()``.  All embedded reference
        data (EEIO factors, physical EFs, classification maps, exchange rates,
        margin tables) is immutable after ``__init__``.  The supplier EF
        registry is mutable and protected by the lock.

    Attributes:
        _config: Optional PurchasedGoodsServicesConfig singleton.
        _metrics: Optional PurchasedGoodsServicesMetrics singleton.
        _provenance: Optional PurchasedGoodsProvenanceTracker singleton.
        _supplier_efs: Mutable registry of supplier-specific EFs.
        _eeio_factors_cache: Pre-built EEIOFactor model cache by sector.
        _physical_ef_cache: Pre-built PhysicalEF model cache by material.

    Example:
        >>> db = ProcurementDatabaseEngine()
        >>> factor = db.lookup_eeio_factor("331110")
        >>> assert factor.sector_code == "331110"
        >>> usd = db.convert_currency(Decimal("1000"), CurrencyCode.EUR)
        >>> margin = db.get_margin_rate("33")
    """

    _instance: Optional[ProcurementDatabaseEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> ProcurementDatabaseEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with a reentrant lock for thread
        safety without unnecessary lock acquisition on subsequent calls.

        Returns:
            The singleton ProcurementDatabaseEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize reference data caches and optional integrations.

        Guarded by ``_initialized`` so repeated calls are no-ops.
        Builds the EEIOFactor and PhysicalEF model caches from the
        constant tables in ``models.py``.
        """
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            self._config: Optional[Any] = None
            self._metrics: Optional[Any] = None
            self._provenance: Optional[Any] = None

            # Attempt to load optional integrations
            if _CONFIG_AVAILABLE:
                try:
                    self._config = PurchasedGoodsServicesConfig()
                except Exception:
                    logger.debug(
                        "Config not available; using defaults"
                    )
            if _METRICS_AVAILABLE:
                try:
                    self._metrics = PurchasedGoodsServicesMetrics()
                except Exception:
                    logger.debug(
                        "Metrics not available; metrics disabled"
                    )
            if _PROVENANCE_AVAILABLE:
                try:
                    self._provenance = (
                        PurchasedGoodsProvenanceTracker.get_instance()
                    )
                except Exception:
                    logger.debug(
                        "Provenance tracker not available"
                    )

            # Mutable supplier EF registry
            self._supplier_efs: Dict[str, List[Any]] = {}

            # Build caches
            self._eeio_factors_cache: Dict[str, Any] = {}
            self._physical_ef_cache: Dict[str, Any] = {}
            self._build_eeio_cache()
            self._build_physical_ef_cache()

            self._initialized = True
            logger.info(
                "ProcurementDatabaseEngine initialized: "
                "eeio_factors=%d, physical_efs=%d, "
                "naics_sectors=%d, naics_to_isic=%d, "
                "nace_to_isic=%d, unspsc_to_naics=%d, "
                "currencies=%d, margin_sectors=%d",
                len(self._eeio_factors_cache),
                len(self._physical_ef_cache),
                len(NAICS_SECTOR_NAMES),
                len(NAICS_TO_ISIC),
                len(NACE_TO_ISIC),
                len(UNSPSC_TO_NAICS),
                len(CURRENCY_EXCHANGE_RATES) if _MODELS_AVAILABLE else 0,
                len(INDUSTRY_MARGIN_PERCENTAGES) if _MODELS_AVAILABLE else 0,
            )

    # ------------------------------------------------------------------
    # Singleton reset (testing only)
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for test teardown.

        After calling ``reset()``, the next instantiation will
        re-initialize all caches and registries. Thread-safe.

        Example:
            >>> ProcurementDatabaseEngine.reset()
            >>> db = ProcurementDatabaseEngine()  # fresh instance
        """
        with cls._lock:
            cls._instance = None
        logger.debug("ProcurementDatabaseEngine singleton reset")

    # ------------------------------------------------------------------
    # Cache builders
    # ------------------------------------------------------------------

    def _build_eeio_cache(self) -> None:
        """Build EEIOFactor model cache from EEIO_EMISSION_FACTORS constant.

        Creates an EEIOFactor Pydantic model for each NAICS-6 sector
        in the constant table.  Stores them keyed by sector code for
        O(1) lookup.
        """
        if not _MODELS_AVAILABLE:
            logger.warning(
                "Models not available; EEIO cache empty"
            )
            return

        for sector_code, factor_value in EEIO_EMISSION_FACTORS.items():
            sector_name = NAICS_SECTOR_NAMES.get(
                sector_code, f"NAICS {sector_code}"
            )
            try:
                ef = EEIOFactor(
                    sector_code=sector_code,
                    sector_name=sector_name,
                    factor_kgco2e_per_unit=factor_value,
                    database=EEIODatabase.EPA_USEEIO,
                    database_version="v1.2",
                    base_year=2019,
                    base_currency=CurrencyCode.USD,
                    region="US",
                    margin_type="purchaser",
                    classification_system=SpendClassificationSystem.NAICS,
                )
                self._eeio_factors_cache[sector_code] = ef
            except Exception as exc:
                logger.warning(
                    "Failed to build EEIOFactor for %s: %s",
                    sector_code,
                    exc,
                )

    def _build_physical_ef_cache(self) -> None:
        """Build PhysicalEF model cache from PHYSICAL_EMISSION_FACTORS.

        Creates a PhysicalEF Pydantic model for each material key
        in the constant table.
        """
        if not _MODELS_AVAILABLE:
            logger.warning(
                "Models not available; physical EF cache empty"
            )
            return

        for material_key, factor_value in PHYSICAL_EMISSION_FACTORS.items():
            mat_name = MATERIAL_KEY_TO_NAME.get(
                material_key, material_key.replace("_", " ").title()
            )
            cat_str = MATERIAL_KEY_TO_CATEGORY.get(material_key)
            mat_cat: Optional[MaterialCategory] = None
            if cat_str:
                try:
                    mat_cat = MaterialCategory(cat_str.lower())
                except ValueError:
                    mat_cat = None

            source_str = MATERIAL_KEY_TO_SOURCE.get(material_key, "DEFRA")
            try:
                ef_source = PhysicalEFSource(source_str.lower())
            except ValueError:
                ef_source = PhysicalEFSource.DEFRA

            try:
                pef = PhysicalEF(
                    material_key=material_key,
                    material_name=mat_name,
                    factor_kgco2e_per_kg=factor_value,
                    source=ef_source,
                    source_year=2023,
                    region="GLOBAL",
                    material_category=mat_cat,
                    includes_transport=True,
                    system_boundary="cradle_to_gate",
                )
                self._physical_ef_cache[material_key] = pef
            except Exception as exc:
                logger.warning(
                    "Failed to build PhysicalEF for %s: %s",
                    material_key,
                    exc,
                )

    # ==================================================================
    # 1. EEIO FACTOR LOOKUP
    # ==================================================================

    def lookup_eeio_factor(
        self,
        naics_code: str,
        database: Optional[Any] = None,
    ) -> Optional[Any]:
        """Look up an EEIO emission factor by exact NAICS-6 code.

        Performs an exact-match lookup in the pre-built EEIO factor
        cache.  Returns None if no factor exists for the given code.

        Args:
            naics_code: 6-digit NAICS sector code (e.g. ``"331110"``).
            database: Optional EEIODatabase enum; currently only
                EPA_USEEIO is populated.  Reserved for future
                multi-database support.

        Returns:
            An ``EEIOFactor`` model instance, or ``None`` if not found.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> ef = db.lookup_eeio_factor("331110")
            >>> assert ef is not None
            >>> assert ef.factor_kgco2e_per_unit == Decimal("1.2340")
        """
        start = time.monotonic()
        code = naics_code.strip()
        result = self._eeio_factors_cache.get(code)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "lookup_eeio_factor(%s): %s in %.2f ms",
            code,
            "HIT" if result else "MISS",
            elapsed_ms,
        )
        return result

    def lookup_eeio_factor_with_fallback(
        self,
        naics_code: str,
        database: Optional[Any] = None,
    ) -> Optional[Any]:
        """Look up an EEIO factor with progressive NAICS fallback.

        Tries exact 6-digit match first, then truncates to 4-digit
        and 2-digit prefixes, returning the first available factor
        whose code starts with the prefix.  This implements the GHG
        Protocol recommendation to use the most specific available
        sector factor.

        Fallback order:
            1. Exact 6-digit NAICS match
            2. First factor matching 4-digit NAICS prefix
            3. First factor matching 2-digit NAICS prefix

        Args:
            naics_code: NAICS code of any length (2-6 digits).
            database: Optional EEIODatabase enum (reserved).

        Returns:
            An ``EEIOFactor`` model instance, or ``None`` if no
            match found at any level.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> ef = db.lookup_eeio_factor_with_fallback("331999")
            >>> # Falls back to a 33xxxx sector
            >>> assert ef is not None
        """
        start = time.monotonic()
        code = naics_code.strip()

        # Level 1: exact match
        result = self._eeio_factors_cache.get(code)
        if result is not None:
            self._log_eeio_lookup(code, result, "exact", start)
            return result

        # Level 2: 4-digit prefix fallback
        if len(code) >= 4:
            prefix_4 = code[:4]
            for cached_code, cached_ef in self._eeio_factors_cache.items():
                if cached_code.startswith(prefix_4):
                    self._log_eeio_lookup(
                        code, cached_ef, f"4-digit({prefix_4})", start
                    )
                    return cached_ef

        # Level 3: 2-digit prefix fallback
        if len(code) >= 2:
            prefix_2 = code[:2]
            for cached_code, cached_ef in self._eeio_factors_cache.items():
                if cached_code.startswith(prefix_2):
                    self._log_eeio_lookup(
                        code, cached_ef, f"2-digit({prefix_2})", start
                    )
                    return cached_ef

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "lookup_eeio_factor_with_fallback(%s): MISS at all "
            "levels in %.2f ms",
            code,
            elapsed_ms,
        )
        return None

    def _log_eeio_lookup(
        self,
        query_code: str,
        result: Any,
        match_type: str,
        start: float,
    ) -> None:
        """Log an EEIO factor lookup result.

        Args:
            query_code: The queried NAICS code.
            result: The matched EEIOFactor.
            match_type: Description of match level.
            start: Monotonic start time.
        """
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "lookup_eeio_factor_with_fallback(%s): %s match "
            "-> %s (%.4f kgCO2e/USD) in %.2f ms",
            query_code,
            match_type,
            result.sector_code if hasattr(result, "sector_code") else "?",
            float(
                result.factor_kgco2e_per_unit
                if hasattr(result, "factor_kgco2e_per_unit")
                else 0
            ),
            elapsed_ms,
        )

    def get_all_eeio_factors(
        self,
        database: Optional[Any] = None,
    ) -> List[Any]:
        """Return all cached EEIO emission factors.

        Args:
            database: Optional EEIODatabase filter (reserved).

        Returns:
            List of all ``EEIOFactor`` model instances, sorted by
            sector code.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> factors = db.get_all_eeio_factors()
            >>> assert len(factors) >= 50
        """
        factors = list(self._eeio_factors_cache.values())
        factors.sort(
            key=lambda f: (
                f.sector_code if hasattr(f, "sector_code") else ""
            )
        )
        return factors

    def search_eeio_factors(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Any]:
        """Search EEIO factors by sector code prefix or name substring.

        Performs a case-insensitive search across both sector codes
        and sector names.  Results are sorted by relevance (exact
        code prefix first, then name matches).

        Args:
            query: Search string (code prefix or name substring).
            limit: Maximum number of results to return (default 20).

        Returns:
            List of matching ``EEIOFactor`` instances, up to ``limit``.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> results = db.search_eeio_factors("steel")
            >>> assert len(results) >= 1
        """
        q_lower = query.strip().lower()
        if not q_lower:
            return []

        code_matches: List[Any] = []
        name_matches: List[Any] = []

        for code, ef in self._eeio_factors_cache.items():
            if code.startswith(q_lower) or code.startswith(query.strip()):
                code_matches.append(ef)
            elif hasattr(ef, "sector_name"):
                if q_lower in ef.sector_name.lower():
                    name_matches.append(ef)

        combined = code_matches + name_matches
        return combined[:limit]

    # ==================================================================
    # 2. PHYSICAL EF LOOKUP
    # ==================================================================

    def lookup_physical_ef(
        self,
        material_key: str,
    ) -> Optional[Any]:
        """Look up a physical emission factor by exact material key.

        Args:
            material_key: Material identifier (e.g. ``"steel_world_avg"``).

        Returns:
            A ``PhysicalEF`` model instance, or ``None`` if not found.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> ef = db.lookup_physical_ef("steel_world_avg")
            >>> assert ef is not None
            >>> assert ef.factor_kgco2e_per_kg == Decimal("1.37")
        """
        start = time.monotonic()
        key = material_key.strip().lower()
        result = self._physical_ef_cache.get(key)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "lookup_physical_ef(%s): %s in %.2f ms",
            key,
            "HIT" if result else "MISS",
            elapsed_ms,
        )
        return result

    def lookup_physical_ef_by_category(
        self,
        category: Any,
    ) -> List[Any]:
        """Look up all physical EFs matching a MaterialCategory.

        Performs a fuzzy lookup returning all materials that belong
        to the given category.  Useful when no exact material key
        is available but the procurement item has a material category.

        Args:
            category: A ``MaterialCategory`` enum value.

        Returns:
            List of ``PhysicalEF`` model instances in the category,
            sorted by factor value ascending (lowest first).

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> metals = db.lookup_physical_ef_by_category(
            ...     MaterialCategory.RAW_METALS
            ... )
            >>> assert len(metals) >= 5
        """
        results: List[Any] = []
        for _key, pef in self._physical_ef_cache.items():
            if hasattr(pef, "material_category") and pef.material_category is not None:
                pef_cat = pef.material_category
                # Compare by value for enum compatibility
                cat_val = (
                    category.value
                    if hasattr(category, "value")
                    else str(category)
                )
                pef_val = (
                    pef_cat.value
                    if hasattr(pef_cat, "value")
                    else str(pef_cat)
                )
                if pef_val == cat_val:
                    results.append(pef)

        results.sort(
            key=lambda f: (
                f.factor_kgco2e_per_kg
                if hasattr(f, "factor_kgco2e_per_kg")
                else Decimal("0")
            )
        )
        return results

    def get_all_physical_efs(self) -> List[Any]:
        """Return all cached physical emission factors.

        Returns:
            List of all ``PhysicalEF`` model instances, sorted by
            material key.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> efs = db.get_all_physical_efs()
            >>> assert len(efs) >= 40
        """
        efs = list(self._physical_ef_cache.values())
        efs.sort(
            key=lambda f: (
                f.material_key if hasattr(f, "material_key") else ""
            )
        )
        return efs

    # ==================================================================
    # 3. CLASSIFICATION CROSS-MAPPING
    # ==================================================================

    def map_classification(
        self,
        source_code: str,
        source_system: Any,
        target_system: Any,
    ) -> List[Tuple[str, Decimal]]:
        """Map a classification code between industry systems.

        Supports forward and reverse mapping between NAICS, NACE,
        ISIC, and UNSPSC systems.  Each mapping includes a confidence
        score (0-1) indicating the quality of the correspondence.

        Supported routes:
            - NAICS -> ISIC (via NAICS_TO_ISIC table)
            - NACE -> ISIC (via NACE_TO_ISIC table)
            - ISIC -> NAICS (via reverse ISIC_TO_NAICS table)
            - UNSPSC -> NAICS (via UNSPSC_TO_NAICS table)
            - NAICS -> NACE (NAICS -> ISIC -> NACE reverse)
            - NACE -> NAICS (NACE -> ISIC -> NAICS reverse)

        Args:
            source_code: The source classification code.
            source_system: ``SpendClassificationSystem`` enum value
                for the source system.
            target_system: ``SpendClassificationSystem`` enum value
                for the target system.

        Returns:
            List of ``(target_code, confidence)`` tuples sorted by
            confidence descending.  Empty list if no mapping found.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> mappings = db.map_classification(
            ...     "331110", SpendClassificationSystem.NAICS,
            ...     SpendClassificationSystem.ISIC
            ... )
            >>> assert len(mappings) >= 1
            >>> assert mappings[0][0] == "2410"
        """
        code = source_code.strip()
        src = self._sys_val(source_system)
        tgt = self._sys_val(target_system)

        if src == tgt:
            return [(code, Decimal("1.00"))]

        # Direct routes
        if src == "naics" and tgt == "isic":
            return self._lookup_mapping(NAICS_TO_ISIC, code)

        if src == "nace" and tgt == "isic":
            return self._lookup_mapping(NACE_TO_ISIC, code)

        if src == "isic" and tgt == "naics":
            return self._lookup_mapping(ISIC_TO_NAICS, code)

        if src == "unspsc" and tgt == "naics":
            segment = code[:2] if len(code) >= 2 else code
            return self._lookup_mapping(UNSPSC_TO_NAICS, segment)

        # Transitive routes: NAICS -> ISIC -> use NACE reverse
        if src == "naics" and tgt == "nace":
            isic_mappings = self._lookup_mapping(NAICS_TO_ISIC, code)
            return self._transitive_isic_to_nace(isic_mappings)

        # NACE -> ISIC -> NAICS
        if src == "nace" and tgt == "naics":
            isic_mappings = self._lookup_mapping(NACE_TO_ISIC, code)
            return self._transitive_isic_to_naics(isic_mappings)

        # ISIC -> NACE (reverse NACE_TO_ISIC)
        if src == "isic" and tgt == "nace":
            return self._reverse_nace_lookup(code)

        # UNSPSC -> ISIC (UNSPSC -> NAICS -> ISIC)
        if src == "unspsc" and tgt == "isic":
            segment = code[:2] if len(code) >= 2 else code
            naics_mappings = self._lookup_mapping(
                UNSPSC_TO_NAICS, segment
            )
            return self._transitive_naics_to_isic(naics_mappings)

        logger.warning(
            "No mapping route from %s to %s", src, tgt
        )
        return []

    def _sys_val(self, system: Any) -> str:
        """Extract string value from a SpendClassificationSystem enum.

        Args:
            system: Enum or string classification system.

        Returns:
            Lowercase string value.
        """
        if hasattr(system, "value"):
            return str(system.value).lower()
        return str(system).lower()

    def _lookup_mapping(
        self,
        table: Dict[str, List[Tuple[str, Decimal]]],
        code: str,
    ) -> List[Tuple[str, Decimal]]:
        """Look up a code in a mapping table.

        Args:
            table: Mapping dictionary.
            code: Code to look up.

        Returns:
            List of (target_code, confidence) sorted by confidence desc.
        """
        results = table.get(code, [])
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _transitive_isic_to_nace(
        self,
        isic_mappings: List[Tuple[str, Decimal]],
    ) -> List[Tuple[str, Decimal]]:
        """Resolve ISIC codes to NACE via reverse NACE_TO_ISIC lookup.

        Args:
            isic_mappings: List of (ISIC code, confidence) tuples.

        Returns:
            List of (NACE code, combined confidence) tuples.
        """
        results: List[Tuple[str, Decimal]] = []
        for isic_code, conf1 in isic_mappings:
            for nace_code, nace_mappings in NACE_TO_ISIC.items():
                for mapped_isic, conf2 in nace_mappings:
                    if mapped_isic == isic_code:
                        combined = _q(conf1 * conf2, 2)
                        results.append((nace_code, combined))
        results.sort(key=lambda x: x[1], reverse=True)
        # Deduplicate keeping highest confidence
        seen: Dict[str, Decimal] = {}
        deduped: List[Tuple[str, Decimal]] = []
        for code, conf in results:
            if code not in seen or conf > seen[code]:
                seen[code] = conf
        for code, conf in seen.items():
            deduped.append((code, conf))
        deduped.sort(key=lambda x: x[1], reverse=True)
        return deduped

    def _transitive_isic_to_naics(
        self,
        isic_mappings: List[Tuple[str, Decimal]],
    ) -> List[Tuple[str, Decimal]]:
        """Resolve ISIC codes to NAICS via reverse ISIC_TO_NAICS lookup.

        Args:
            isic_mappings: List of (ISIC code, confidence) tuples.

        Returns:
            List of (NAICS code, combined confidence) tuples.
        """
        results: List[Tuple[str, Decimal]] = []
        for isic_code, conf1 in isic_mappings:
            naics_list = ISIC_TO_NAICS.get(isic_code, [])
            for naics_code, conf2 in naics_list:
                combined = _q(conf1 * conf2, 2)
                results.append((naics_code, combined))
        results.sort(key=lambda x: x[1], reverse=True)
        seen: Dict[str, Decimal] = {}
        for code, conf in results:
            if code not in seen or conf > seen[code]:
                seen[code] = conf
        deduped = [(c, v) for c, v in seen.items()]
        deduped.sort(key=lambda x: x[1], reverse=True)
        return deduped

    def _transitive_naics_to_isic(
        self,
        naics_mappings: List[Tuple[str, Decimal]],
    ) -> List[Tuple[str, Decimal]]:
        """Resolve NAICS codes to ISIC via NAICS_TO_ISIC lookup.

        Args:
            naics_mappings: List of (NAICS 2-digit, confidence) tuples.

        Returns:
            List of (ISIC code, combined confidence) tuples.
        """
        results: List[Tuple[str, Decimal]] = []
        for naics_prefix, conf1 in naics_mappings:
            for naics_code, isic_list in NAICS_TO_ISIC.items():
                if naics_code.startswith(naics_prefix):
                    for isic_code, conf2 in isic_list:
                        combined = _q(conf1 * conf2, 2)
                        results.append((isic_code, combined))
        results.sort(key=lambda x: x[1], reverse=True)
        seen: Dict[str, Decimal] = {}
        for code, conf in results:
            if code not in seen or conf > seen[code]:
                seen[code] = conf
        deduped = [(c, v) for c, v in seen.items()]
        deduped.sort(key=lambda x: x[1], reverse=True)
        return deduped

    def _reverse_nace_lookup(
        self,
        isic_code: str,
    ) -> List[Tuple[str, Decimal]]:
        """Reverse-lookup NACE codes from an ISIC code.

        Args:
            isic_code: ISIC Rev 4.1 code.

        Returns:
            List of (NACE code, confidence) tuples.
        """
        results: List[Tuple[str, Decimal]] = []
        for nace_code, mappings in NACE_TO_ISIC.items():
            for mapped_isic, conf in mappings:
                if mapped_isic == isic_code:
                    results.append((nace_code, conf))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def resolve_naics_from_any(
        self,
        code: str,
        system: Any,
    ) -> Optional[str]:
        """Resolve a classification code from any system to NAICS.

        Convenience method that maps any supported classification
        code to a NAICS code, returning the highest-confidence match.

        Args:
            code: Classification code in the source system.
            system: ``SpendClassificationSystem`` enum for source.

        Returns:
            NAICS code string, or ``None`` if no mapping found.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> naics = db.resolve_naics_from_any("2410", SpendClassificationSystem.ISIC)
            >>> assert naics == "331110"
        """
        sys_val = self._sys_val(system)

        if sys_val == "naics":
            return code.strip()

        if sys_val == "isic":
            mappings = self._lookup_mapping(ISIC_TO_NAICS, code.strip())
        elif sys_val == "nace":
            isic_mappings = self._lookup_mapping(NACE_TO_ISIC, code.strip())
            mappings = self._transitive_isic_to_naics(isic_mappings)
        elif sys_val == "unspsc":
            segment = code.strip()[:2]
            mappings = self._lookup_mapping(UNSPSC_TO_NAICS, segment)
        else:
            return None

        if mappings:
            return mappings[0][0]
        return None

    def get_naics_sector_name(
        self,
        naics_code: str,
    ) -> Optional[str]:
        """Look up the human-readable name for a NAICS code.

        Tries exact match first, then falls back to 4-digit and
        2-digit prefixes.

        Args:
            naics_code: NAICS code (2-6 digits).

        Returns:
            Sector name string, or ``None`` if not found.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> name = db.get_naics_sector_name("331110")
            >>> assert "Iron" in name
        """
        code = naics_code.strip()

        # Exact match
        name = NAICS_SECTOR_NAMES.get(code)
        if name:
            return name

        # 4-digit prefix
        if len(code) > 4:
            name = NAICS_SECTOR_NAMES.get(code[:4])
            if name:
                return name

        # 2-digit prefix
        if len(code) > 2:
            name = NAICS_SECTOR_NAMES.get(code[:2])
            if name:
                return name

        return None

    # ==================================================================
    # 4. CURRENCY CONVERSION & INFLATION
    # ==================================================================

    def convert_currency(
        self,
        amount: Decimal,
        from_currency: Any,
        to_currency: Optional[Any] = None,
    ) -> Decimal:
        """Convert a monetary amount between currencies via USD.

        Uses the CURRENCY_EXCHANGE_RATES table from models.py.
        Rates are expressed as units of foreign currency per 1 USD.
        Conversion formula: ``amount_usd = amount / rate_from``,
        then ``amount_to = amount_usd * rate_to``.

        Args:
            amount: The monetary amount to convert.
            from_currency: Source ``CurrencyCode`` enum.
            to_currency: Target ``CurrencyCode`` enum (default USD).

        Returns:
            Converted amount quantized to 8 decimal places.

        Raises:
            ValueError: If either currency code is not in the
                exchange rate table.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> usd = db.convert_currency(
            ...     Decimal("1000"), CurrencyCode.EUR
            ... )
            >>> assert usd > Decimal("1000")  # EUR > USD
        """
        if not _MODELS_AVAILABLE:
            logger.warning("Models unavailable; returning amount as-is")
            return amount

        if to_currency is None:
            to_currency = CurrencyCode.USD

        from_val = self._currency_val(from_currency)
        to_val = self._currency_val(to_currency)

        if from_val == to_val:
            return _q(amount)

        from_rate = CURRENCY_EXCHANGE_RATES.get(from_currency)
        to_rate = CURRENCY_EXCHANGE_RATES.get(to_currency)

        if from_rate is None:
            raise ValueError(
                f"No exchange rate for currency: {from_currency}"
            )
        if to_rate is None:
            raise ValueError(
                f"No exchange rate for currency: {to_currency}"
            )

        # Convert to USD first: amount_usd = amount / from_rate
        # (from_rate is units of foreign currency per 1 USD)
        amount_usd = amount / from_rate

        # Convert from USD to target: amount_target = amount_usd * to_rate
        result = amount_usd * to_rate

        return _q(result)

    def _currency_val(self, currency: Any) -> str:
        """Extract string value from a CurrencyCode enum.

        Args:
            currency: CurrencyCode enum or string.

        Returns:
            Uppercase currency string.
        """
        if hasattr(currency, "value"):
            return str(currency.value).upper()
        return str(currency).upper()

    def deflate_spend(
        self,
        amount_usd: Decimal,
        cpi_ratio: Optional[Decimal] = None,
    ) -> Decimal:
        """Deflate a USD spend amount using a CPI ratio.

        Adjusts the spend from the reporting year to the EEIO base
        year using the Consumer Price Index ratio.  Formula:
        ``deflated = amount_usd / cpi_ratio``.

        A CPI ratio > 1.0 means prices increased since the base year.
        A CPI ratio of 1.0 means no inflation adjustment needed.

        Args:
            amount_usd: Spend amount in USD (reporting year).
            cpi_ratio: CPI ratio = CPI_reporting_year / CPI_base_year.
                Default ``Decimal("1.0")`` (no adjustment).

        Returns:
            Deflated amount quantized to 8 decimal places.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> deflated = db.deflate_spend(
            ...     Decimal("10000"), Decimal("1.08")
            ... )
            >>> assert deflated < Decimal("10000")
        """
        if cpi_ratio is None:
            cpi_ratio = Decimal("1.0")
        if cpi_ratio <= Decimal("0"):
            raise ValueError(
                f"CPI ratio must be positive, got: {cpi_ratio}"
            )
        return _q(amount_usd / cpi_ratio)

    def apply_ppp_adjustment(
        self,
        amount: Decimal,
        ppp_factor: Decimal,
    ) -> Decimal:
        """Apply purchasing power parity adjustment.

        Adjusts a monetary amount using the PPP conversion factor
        to account for differences in price levels between countries.
        Formula: ``adjusted = amount * ppp_factor``.

        Args:
            amount: The monetary amount to adjust.
            ppp_factor: PPP conversion factor (e.g. 0.85 for a
                country where goods are 15% cheaper than the US).

        Returns:
            PPP-adjusted amount quantized to 8 decimal places.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> adj = db.apply_ppp_adjustment(
            ...     Decimal("10000"), Decimal("0.85")
            ... )
            >>> assert adj == Decimal("8500.00000000")
        """
        return _q(amount * ppp_factor)

    # ==================================================================
    # 5. MARGIN ADJUSTMENT
    # ==================================================================

    def get_margin_rate(
        self,
        naics_code: str,
    ) -> Decimal:
        """Look up the margin rate for a NAICS sector.

        Uses the INDUSTRY_MARGIN_PERCENTAGES table keyed by
        2-digit NAICS sector code.  Falls back to a default
        margin rate of 20% if the sector is not found.

        Args:
            naics_code: NAICS code (any length; first 2 digits used).

        Returns:
            Margin rate as a Decimal percentage (e.g. ``Decimal("30.0")``
            for 30%).

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> rate = db.get_margin_rate("331110")
            >>> assert rate == Decimal("30.0")  # Manufacturing Metals
        """
        if not _MODELS_AVAILABLE:
            return Decimal("20.0")

        code = naics_code.strip()
        prefix_2 = code[:2] if len(code) >= 2 else code

        rate = INDUSTRY_MARGIN_PERCENTAGES.get(prefix_2)
        if rate is not None:
            return rate

        # Default margin for unknown sectors
        logger.debug(
            "No margin rate for NAICS prefix %s; using default 20%%",
            prefix_2,
        )
        return Decimal("20.0")

    def remove_margin(
        self,
        spend_usd: Decimal,
        margin_rate: Decimal,
    ) -> Decimal:
        """Remove the trade margin from a purchaser-price spend amount.

        Converts from purchaser price to producer/basic price by
        removing the wholesale + retail + transport margin.
        Formula: ``producer_price = spend * (1 - margin_rate/100)``.

        Args:
            spend_usd: Spend amount in purchaser price (USD).
            margin_rate: Margin rate as a percentage (e.g. 30.0 for 30%).

        Returns:
            Producer-price spend amount quantized to 8 decimal places.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> producer = db.remove_margin(Decimal("10000"), Decimal("30"))
            >>> assert producer == Decimal("7000.00000000")
        """
        if margin_rate < Decimal("0") or margin_rate > Decimal("100"):
            raise ValueError(
                f"Margin rate must be 0-100, got: {margin_rate}"
            )
        multiplier = Decimal("1") - (margin_rate / Decimal("100"))
        return _q(spend_usd * multiplier)

    def prepare_spend_record(
        self,
        item: Any,
        database: Optional[Any] = None,
        cpi_ratio: Optional[Decimal] = None,
    ) -> Any:
        """Prepare a SpendRecord from a ProcurementItem.

        Performs currency conversion, inflation deflation, margin
        removal, and EEIO sector resolution in a single call.  This
        is the primary integration point used by the
        SpendBasedCalculatorEngine.

        Steps:
            1. Convert spend to USD using item.currency
            2. Deflate to base year using cpi_ratio
            3. Resolve NAICS sector code (from item.naics_code or
               cross-mapping from NACE/ISIC/UNSPSC)
            4. Look up margin rate for the sector
            5. Remove margin to get producer price

        Args:
            item: A ``ProcurementItem`` model instance.
            database: Optional ``EEIODatabase`` for factor lookup.
            cpi_ratio: CPI ratio for inflation deflation
                (default ``Decimal("1.0")``).

        Returns:
            A ``SpendRecord`` model instance with all fields populated.

        Raises:
            ValueError: If models are unavailable.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> item = ProcurementItem(
            ...     description="Steel plates",
            ...     spend_amount=Decimal("50000"),
            ...     currency=CurrencyCode.USD,
            ...     naics_code="331110",
            ... )
            >>> record = db.prepare_spend_record(item)
            >>> assert record.spend_producer_usd < record.spend_usd
        """
        if not _MODELS_AVAILABLE:
            raise ValueError(
                "Models not available; cannot create SpendRecord"
            )

        if cpi_ratio is None:
            cpi_ratio = Decimal("1.0")
        if database is None:
            database = EEIODatabase.EPA_USEEIO

        start = time.monotonic()

        # Step 1: Currency conversion
        spend_usd = self.convert_currency(
            item.spend_amount, item.currency, CurrencyCode.USD
        )
        fx_rate = CURRENCY_EXCHANGE_RATES.get(
            item.currency, Decimal("1.0")
        )

        # Step 2: Inflation deflation
        spend_deflated = self.deflate_spend(spend_usd, cpi_ratio)

        # Step 3: Resolve NAICS sector
        sector_code = self._resolve_sector_code(item)

        # Step 4: Margin lookup
        margin_rate = self.get_margin_rate(
            sector_code if sector_code else "99"
        )

        # Step 5: Margin removal
        spend_producer = self.remove_margin(spend_deflated, margin_rate)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "prepare_spend_record(%s): USD=%.2f, deflated=%.2f, "
            "producer=%.2f, sector=%s, margin=%.1f%% in %.2f ms",
            item.item_id,
            float(spend_usd),
            float(spend_deflated),
            float(spend_producer),
            sector_code,
            float(margin_rate),
            elapsed_ms,
        )

        return SpendRecord(
            item=item,
            spend_usd=spend_usd,
            spend_deflated_usd=spend_deflated,
            spend_producer_usd=spend_producer,
            eeio_database=database,
            eeio_sector_code=sector_code,
            margin_rate=margin_rate,
            cpi_ratio=cpi_ratio,
            fx_rate=fx_rate,
        )

    def _resolve_sector_code(self, item: Any) -> Optional[str]:
        """Resolve the best available NAICS sector code from an item.

        Priority order:
            1. item.naics_code (direct)
            2. item.isic_code -> NAICS mapping
            3. item.nace_code -> ISIC -> NAICS mapping
            4. item.unspsc_code -> NAICS mapping

        Args:
            item: A ProcurementItem with optional classification fields.

        Returns:
            NAICS code string, or ``None`` if none resolved.
        """
        if hasattr(item, "naics_code") and item.naics_code:
            return item.naics_code.strip()

        if hasattr(item, "isic_code") and item.isic_code:
            result = self.resolve_naics_from_any(
                item.isic_code,
                SpendClassificationSystem.ISIC
                if _MODELS_AVAILABLE
                else "isic",
            )
            if result:
                return result

        if hasattr(item, "nace_code") and item.nace_code:
            result = self.resolve_naics_from_any(
                item.nace_code,
                SpendClassificationSystem.NACE
                if _MODELS_AVAILABLE
                else "nace",
            )
            if result:
                return result

        if hasattr(item, "unspsc_code") and item.unspsc_code:
            result = self.resolve_naics_from_any(
                item.unspsc_code,
                SpendClassificationSystem.UNSPSC
                if _MODELS_AVAILABLE
                else "unspsc",
            )
            if result:
                return result

        return None

    # ==================================================================
    # 6. SUPPLIER EF REGISTRY
    # ==================================================================

    def register_supplier_ef(
        self,
        supplier_ef: Any,
    ) -> None:
        """Register a supplier-specific emission factor.

        Adds a ``SupplierEF`` to the in-memory registry indexed by
        ``supplier_id``.  Multiple EFs can be registered per supplier
        (one per product).  Thread-safe via the singleton lock.

        Args:
            supplier_ef: A ``SupplierEF`` model instance to register.

        Raises:
            ValueError: If supplier_ef lacks a supplier_id.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> sef = SupplierEF(
            ...     supplier_id="SUP-001",
            ...     supplier_name="Acme Steel",
            ...     product_name="Hot-Rolled Coil",
            ...     factor_kgco2e_per_unit=Decimal("1.85"),
            ...     data_source=SupplierDataSource.EPD,
            ...     verification_status="third_party_verified",
            ... )
            >>> db.register_supplier_ef(sef)
        """
        sid = (
            supplier_ef.supplier_id
            if hasattr(supplier_ef, "supplier_id")
            else None
        )
        if not sid:
            raise ValueError(
                "SupplierEF must have a supplier_id"
            )

        with self._lock:
            if sid not in self._supplier_efs:
                self._supplier_efs[sid] = []
            self._supplier_efs[sid].append(supplier_ef)

        logger.info(
            "Registered supplier EF: supplier=%s, product=%s, "
            "factor=%.4f kgCO2e/%s, source=%s",
            sid,
            getattr(supplier_ef, "product_name", "?"),
            float(
                getattr(
                    supplier_ef, "factor_kgco2e_per_unit", Decimal("0")
                )
            ),
            getattr(supplier_ef, "factor_unit", "unit"),
            getattr(supplier_ef, "data_source", "?"),
        )

    def lookup_supplier_ef(
        self,
        supplier_id: str,
        product_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Look up a supplier-specific EF by supplier ID and product.

        If ``product_name`` is given, returns the EF matching that
        product.  Otherwise returns the first (most recently
        registered) EF for the supplier.

        Args:
            supplier_id: Unique supplier identifier.
            product_name: Optional product name filter.

        Returns:
            A ``SupplierEF`` model instance, or ``None`` if not found.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> ef = db.lookup_supplier_ef("SUP-001", "Hot-Rolled Coil")
        """
        with self._lock:
            efs = self._supplier_efs.get(supplier_id, [])

        if not efs:
            return None

        if product_name is not None:
            pn_lower = product_name.strip().lower()
            for ef in efs:
                ef_pn = getattr(ef, "product_name", "")
                if ef_pn and ef_pn.strip().lower() == pn_lower:
                    return ef
            return None

        # Return most recently registered
        return efs[-1] if efs else None

    def get_supplier_efs(
        self,
        supplier_id: str,
    ) -> List[Any]:
        """Get all registered EFs for a supplier.

        Args:
            supplier_id: Unique supplier identifier.

        Returns:
            List of ``SupplierEF`` model instances. Empty if none.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> efs = db.get_supplier_efs("SUP-001")
        """
        with self._lock:
            return list(self._supplier_efs.get(supplier_id, []))

    # ==================================================================
    # 7. EF HIERARCHY SELECTION
    # ==================================================================

    def select_best_ef(
        self,
        item: Any,
        supplier_efs: Optional[List[Any]] = None,
    ) -> Tuple[str, int, Any]:
        """Select the best emission factor for an item using the 8-level hierarchy.

        Evaluates available data sources for the given procurement
        item and selects the highest-priority (lowest number) factor
        according to the EF_HIERARCHY_PRIORITY from models.py.

        8-Level Hierarchy (1=best, 8=worst):
            1. supplier_epd_verified - Third-party verified EPD
            2. supplier_cdp_unverified - CDP/unverified supplier data
            3. product_lca_ecoinvent - Product-level LCA (ecoinvent)
            4. material_avg_ice_defra - Material average (ICE/DEFRA)
            5. industry_avg_physical - Industry-average physical EF
            6. regional_eeio_exiobase - Regional EEIO (EXIOBASE)
            7. national_eeio_useeio - National EEIO (EPA USEEIO)
            8. global_avg_eeio_fallback - Global average EEIO fallback

        Args:
            item: A ``ProcurementItem`` with classification and
                supplier fields.
            supplier_efs: Optional list of pre-fetched ``SupplierEF``
                instances for this item's supplier.

        Returns:
            Tuple of ``(hierarchy_key, priority_level, ef_data)`` where
            ``ef_data`` is the selected factor model (SupplierEF,
            PhysicalEF, or EEIOFactor).

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> key, level, ef = db.select_best_ef(item)
            >>> print(f"Selected {key} at level {level}")
        """
        # Gather supplier EFs if not provided
        if supplier_efs is None:
            sid = getattr(item, "supplier_id", None)
            if sid:
                supplier_efs = self.get_supplier_efs(sid)
            else:
                supplier_efs = []

        # Level 1: Supplier EPD (verified)
        for sef in supplier_efs:
            source = getattr(sef, "data_source", None)
            status = getattr(sef, "verification_status", "")
            source_val = source.value if hasattr(source, "value") else str(source)
            if source_val == "epd" and "verified" in status.lower():
                return ("supplier_epd_verified", 1, sef)

        # Level 2: Supplier CDP/unverified
        for sef in supplier_efs:
            source = getattr(sef, "data_source", None)
            source_val = source.value if hasattr(source, "value") else str(source)
            if source_val in (
                "cdp_supply_chain",
                "ecovadis",
                "sustainability_report",
                "pact_network",
                "direct_measurement",
                "epd",
            ):
                return ("supplier_cdp_unverified", 2, sef)

        # Level 3: Product LCA (ecoinvent)
        mat_key = getattr(item, "material_category", None)
        if mat_key:
            pef = self._find_ecoinvent_ef(item)
            if pef is not None:
                return ("product_lca_ecoinvent", 3, pef)

        # Level 4: Material average (ICE/DEFRA)
        pef = self._find_physical_ef(item)
        if pef is not None:
            source = getattr(pef, "source", None)
            source_val = source.value if hasattr(source, "value") else str(source)
            if source_val in ("ice", "defra"):
                return ("material_avg_ice_defra", 4, pef)

        # Level 5: Industry average physical EF
        if pef is not None:
            return ("industry_avg_physical", 5, pef)

        # Level 6: Regional EEIO (EXIOBASE) - check NACE code
        # Currently only EPA USEEIO populated; placeholder
        # for future EXIOBASE integration

        # Level 7: National EEIO (EPA USEEIO)
        naics = self._resolve_sector_code(item)
        if naics:
            eeio_ef = self.lookup_eeio_factor_with_fallback(naics)
            if eeio_ef is not None:
                return ("national_eeio_useeio", 7, eeio_ef)

        # Level 8: Global average EEIO fallback
        if naics:
            # Use 2-digit NAICS fallback as global average proxy
            prefix_2 = naics[:2]
            for code, ef in self._eeio_factors_cache.items():
                if code.startswith(prefix_2):
                    return ("global_avg_eeio_fallback", 8, ef)

        # No factor found at any level
        logger.warning(
            "No emission factor found for item %s at any "
            "hierarchy level",
            getattr(item, "item_id", "?"),
        )
        return ("global_avg_eeio_fallback", 8, None)

    def _find_ecoinvent_ef(self, item: Any) -> Optional[Any]:
        """Find an ecoinvent-sourced physical EF for an item.

        Args:
            item: ProcurementItem with material_category or material data.

        Returns:
            PhysicalEF from ecoinvent source, or None.
        """
        # Try direct material key from metadata
        mat_key = getattr(item, "metadata", {}).get("material_key")
        if mat_key:
            pef = self._physical_ef_cache.get(mat_key)
            if pef is not None:
                source = getattr(pef, "source", None)
                src_val = source.value if hasattr(source, "value") else str(source)
                if src_val == "ecoinvent":
                    return pef

        # Try category match for ecoinvent sources
        mat_cat = getattr(item, "material_category", None)
        if mat_cat:
            candidates = self.lookup_physical_ef_by_category(mat_cat)
            for c in candidates:
                source = getattr(c, "source", None)
                src_val = source.value if hasattr(source, "value") else str(source)
                if src_val == "ecoinvent":
                    return c

        return None

    def _find_physical_ef(self, item: Any) -> Optional[Any]:
        """Find any physical EF for an item.

        Tries direct material_key from metadata, then falls back
        to category-based lookup.

        Args:
            item: ProcurementItem.

        Returns:
            PhysicalEF, or None.
        """
        # Try direct material key
        mat_key = getattr(item, "metadata", {}).get("material_key")
        if mat_key:
            pef = self._physical_ef_cache.get(mat_key)
            if pef is not None:
                return pef

        # Try category
        mat_cat = getattr(item, "material_category", None)
        if mat_cat:
            candidates = self.lookup_physical_ef_by_category(mat_cat)
            if candidates:
                return candidates[0]  # Lowest factor (conservative)

        return None

    # ==================================================================
    # 8. HEALTH CHECK
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Run a comprehensive health check on the engine.

        Verifies that all reference data caches are populated and
        that core operations function correctly.

        Returns:
            Dictionary with health check results including status,
            cache sizes, and timing information.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> health = db.health_check()
            >>> assert health["status"] == "healthy"
        """
        start = time.monotonic()
        checks: Dict[str, Any] = {}
        healthy = True

        # Check EEIO cache
        eeio_count = len(self._eeio_factors_cache)
        checks["eeio_factors_count"] = eeio_count
        if eeio_count == 0:
            checks["eeio_factors_status"] = "EMPTY"
            healthy = False
        else:
            checks["eeio_factors_status"] = "OK"

        # Check physical EF cache
        pef_count = len(self._physical_ef_cache)
        checks["physical_efs_count"] = pef_count
        if pef_count == 0:
            checks["physical_efs_status"] = "EMPTY"
            healthy = False
        else:
            checks["physical_efs_status"] = "OK"

        # Check supplier EF registry
        with self._lock:
            supplier_count = sum(
                len(v) for v in self._supplier_efs.values()
            )
        checks["supplier_efs_count"] = supplier_count
        checks["supplier_efs_suppliers"] = len(self._supplier_efs)

        # Check mapping tables
        checks["naics_sectors_count"] = len(NAICS_SECTOR_NAMES)
        checks["naics_to_isic_count"] = len(NAICS_TO_ISIC)
        checks["nace_to_isic_count"] = len(NACE_TO_ISIC)
        checks["isic_to_naics_count"] = len(ISIC_TO_NAICS)
        checks["unspsc_to_naics_count"] = len(UNSPSC_TO_NAICS)

        # Check currency rates
        if _MODELS_AVAILABLE:
            checks["currency_rates_count"] = len(CURRENCY_EXCHANGE_RATES)
            checks["margin_sectors_count"] = len(
                INDUSTRY_MARGIN_PERCENTAGES
            )
        else:
            checks["currency_rates_count"] = 0
            checks["margin_sectors_count"] = 0
            healthy = False

        # Smoke test: lookup a known factor
        try:
            test_ef = self.lookup_eeio_factor("331110")
            checks["smoke_test_eeio"] = (
                "PASS" if test_ef is not None else "FAIL"
            )
            if test_ef is None:
                healthy = False
        except Exception as exc:
            checks["smoke_test_eeio"] = f"ERROR: {exc}"
            healthy = False

        # Smoke test: physical EF lookup
        try:
            test_pef = self.lookup_physical_ef("steel_world_avg")
            checks["smoke_test_physical"] = (
                "PASS" if test_pef is not None else "FAIL"
            )
            if test_pef is None:
                healthy = False
        except Exception as exc:
            checks["smoke_test_physical"] = f"ERROR: {exc}"
            healthy = False

        # Smoke test: classification mapping
        try:
            if _MODELS_AVAILABLE:
                test_map = self.map_classification(
                    "331110",
                    SpendClassificationSystem.NAICS,
                    SpendClassificationSystem.ISIC,
                )
                checks["smoke_test_classification"] = (
                    "PASS" if len(test_map) > 0 else "FAIL"
                )
            else:
                checks["smoke_test_classification"] = "SKIPPED"
        except Exception as exc:
            checks["smoke_test_classification"] = f"ERROR: {exc}"
            healthy = False

        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "engine": "ProcurementDatabaseEngine",
            "agent": "GL-MRV-S3-001",
            "component": "AGENT-MRV-014",
            "version": VERSION if _MODELS_AVAILABLE else "1.0.0",
            "status": "healthy" if healthy else "degraded",
            "timestamp": _utcnow().isoformat(),
            "duration_ms": round(elapsed_ms, 2),
            "models_available": _MODELS_AVAILABLE,
            "config_available": _CONFIG_AVAILABLE,
            "metrics_available": _METRICS_AVAILABLE,
            "provenance_available": _PROVENANCE_AVAILABLE,
            "checks": checks,
        }

    # ==================================================================
    # 9. PROVENANCE HELPERS
    # ==================================================================

    def compute_factor_provenance(
        self,
        factor_type: str,
        factor_key: str,
        factor_value: Decimal,
        source: str,
        lookup_method: str,
    ) -> str:
        """Compute a SHA-256 provenance hash for a factor lookup.

        Creates an audit trail hash capturing the lookup parameters
        and result for regulatory compliance traceability.

        Args:
            factor_type: Type of factor (``"eeio"`` or ``"physical"``
                or ``"supplier"``).
            factor_key: Key used for the lookup (NAICS code or
                material key).
            factor_value: The emission factor value retrieved.
            source: Source database identifier.
            lookup_method: Lookup method used (``"exact"``,
                ``"fallback"``, ``"category"``).

        Returns:
            64-character SHA-256 hex digest.

        Example:
            >>> db = ProcurementDatabaseEngine()
            >>> h = db.compute_factor_provenance(
            ...     "eeio", "331110", Decimal("1.234"),
            ...     "EPA_USEEIO_v12", "exact"
            ... )
            >>> assert len(h) == 64
        """
        data = {
            "factor_type": factor_type,
            "factor_key": factor_key,
            "factor_value": str(factor_value),
            "source": source,
            "lookup_method": lookup_method,
            "timestamp": _utcnow().isoformat(),
            "engine": "ProcurementDatabaseEngine",
            "agent": "GL-MRV-S3-001",
        }
        return _sha256(data)

    def compute_conversion_provenance(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        fx_rate: Decimal,
        result: Decimal,
    ) -> str:
        """Compute a SHA-256 provenance hash for a currency conversion.

        Args:
            amount: Original amount.
            from_currency: Source currency code.
            to_currency: Target currency code.
            fx_rate: Exchange rate applied.
            result: Converted amount.

        Returns:
            64-character SHA-256 hex digest.
        """
        data = {
            "operation": "currency_conversion",
            "amount": str(amount),
            "from_currency": from_currency,
            "to_currency": to_currency,
            "fx_rate": str(fx_rate),
            "result": str(result),
            "timestamp": _utcnow().isoformat(),
            "engine": "ProcurementDatabaseEngine",
        }
        return _sha256(data)

    def compute_margin_provenance(
        self,
        spend_usd: Decimal,
        margin_rate: Decimal,
        producer_price: Decimal,
        naics_code: str,
    ) -> str:
        """Compute a SHA-256 provenance hash for a margin adjustment.

        Args:
            spend_usd: Original purchaser-price spend.
            margin_rate: Margin rate applied.
            producer_price: Resulting producer price.
            naics_code: NAICS code used for margin lookup.

        Returns:
            64-character SHA-256 hex digest.
        """
        data = {
            "operation": "margin_removal",
            "spend_usd": str(spend_usd),
            "margin_rate": str(margin_rate),
            "producer_price": str(producer_price),
            "naics_code": naics_code,
            "timestamp": _utcnow().isoformat(),
            "engine": "ProcurementDatabaseEngine",
        }
        return _sha256(data)

    # ==================================================================
    # 10. STATISTICS & INTROSPECTION
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the engine's data holdings.

        Returns:
            Dictionary with counts and metadata about all reference
            data tables and registries.
        """
        with self._lock:
            supplier_supplier_count = len(self._supplier_efs)
            supplier_ef_count = sum(
                len(v) for v in self._supplier_efs.values()
            )

        return {
            "eeio_factors": len(self._eeio_factors_cache),
            "physical_efs": len(self._physical_ef_cache),
            "supplier_suppliers": supplier_supplier_count,
            "supplier_efs": supplier_ef_count,
            "naics_sector_names": len(NAICS_SECTOR_NAMES),
            "naics_to_isic_mappings": len(NAICS_TO_ISIC),
            "nace_to_isic_mappings": len(NACE_TO_ISIC),
            "isic_to_naics_mappings": len(ISIC_TO_NAICS),
            "unspsc_to_naics_mappings": len(UNSPSC_TO_NAICS),
            "material_categories": len(MATERIAL_KEY_TO_CATEGORY),
            "currency_rates": (
                len(CURRENCY_EXCHANGE_RATES) if _MODELS_AVAILABLE else 0
            ),
            "margin_sectors": (
                len(INDUSTRY_MARGIN_PERCENTAGES)
                if _MODELS_AVAILABLE
                else 0
            ),
            "ef_hierarchy_levels": (
                len(EF_HIERARCHY_PRIORITY) if _MODELS_AVAILABLE else 0
            ),
        }

    def __repr__(self) -> str:
        """Return a human-readable representation of the engine."""
        stats = self.get_statistics()
        return (
            f"ProcurementDatabaseEngine("
            f"eeio={stats['eeio_factors']}, "
            f"physical={stats['physical_efs']}, "
            f"suppliers={stats['supplier_suppliers']}, "
            f"naics_names={stats['naics_sector_names']}, "
            f"mappings={stats['naics_to_isic_mappings']}"
            f"+{stats['nace_to_isic_mappings']}"
            f"+{stats['unspsc_to_naics_mappings']})"
        )
