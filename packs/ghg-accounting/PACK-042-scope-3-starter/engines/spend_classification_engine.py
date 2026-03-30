# -*- coding: utf-8 -*-
"""
SpendClassificationEngine - PACK-042 Scope 3 Starter Pack Engine 2
====================================================================

Deterministic spend data classification into Scope 3 categories and EEIO
sectors.  Processes procurement transactions by mapping NAICS, ISIC,
UNSPSC, or GL account codes to the appropriate Scope 3 category and
emission intensity sector, with a keyword-based fallback for unclassified
spend.

The engine supports split transaction handling (one invoice allocated
across multiple categories), currency normalisation (50+ currencies),
and CPI-based inflation adjustment for multi-year analysis.

Calculation Methodology:
    Classification Flow:
        1. Attempt NAICS code match (HIGH confidence)
        2. Attempt ISIC code match (HIGH confidence)
        3. Attempt UNSPSC code match (HIGH confidence)
        4. Attempt GL account range match (MEDIUM confidence)
        5. Attempt keyword match (LOW confidence)

    Currency Normalisation:
        spend_eur = spend_original * fx_rate[currency]

    CPI Inflation Adjustment:
        spend_adjusted = spend_base_year * (cpi_target / cpi_base)

    Confidence Scoring:
        HIGH   = exact code match (NAICS/ISIC/UNSPSC)
        MEDIUM = partial code or GL account range match
        LOW    = keyword-based classification

Regulatory References:
    - GHG Protocol Scope 3 Standard, Chapter 7 (Collecting Data)
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions
    - NAICS 2022 (North American Industry Classification System)
    - ISIC Rev. 4 (International Standard Industrial Classification)
    - UNSPSC v26 (United Nations Standard Products and Services Code)
    - Exiobase 3 sector concordance tables

Zero-Hallucination:
    - All classification uses deterministic lookup tables
    - No LLM involvement in classification or calculation
    - SHA-256 provenance hash on every result
    - Confidence levels reflect data quality accurately

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-042 Scope 3 Starter
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serialisable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serialisable = data
    else:
        serialisable = str(data)
    if isinstance(serialisable, dict):
        serialisable = {
            k: v for k, v in serialisable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serialisable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope3Category(str, Enum):
    """All 15 Scope 3 categories per GHG Protocol."""
    CAT_01 = "cat_01_purchased_goods"
    CAT_02 = "cat_02_capital_goods"
    CAT_03 = "cat_03_fuel_energy"
    CAT_04 = "cat_04_upstream_transport"
    CAT_05 = "cat_05_waste"
    CAT_06 = "cat_06_business_travel"
    CAT_07 = "cat_07_employee_commuting"
    CAT_08 = "cat_08_upstream_leased"
    CAT_09 = "cat_09_downstream_transport"
    CAT_10 = "cat_10_processing_sold"
    CAT_11 = "cat_11_use_sold_products"
    CAT_12 = "cat_12_eol_sold_products"
    CAT_13 = "cat_13_downstream_leased"
    CAT_14 = "cat_14_franchises"
    CAT_15 = "cat_15_investments"

class ClassificationConfidence(str, Enum):
    """Confidence level of classification.

    HIGH:   Exact code match (NAICS, ISIC, UNSPSC).
    MEDIUM: Partial code match or GL account range.
    LOW:    Keyword-based classification.
    UNCLASSIFIED: Could not classify.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCLASSIFIED = "unclassified"

class ClassificationMethod(str, Enum):
    """Method used for classification.

    NAICS:      NAICS code lookup.
    ISIC:       ISIC code lookup.
    UNSPSC:     UNSPSC code lookup.
    GL_ACCOUNT: GL account range mapping.
    KEYWORD:    Keyword-based fallback.
    MANUAL:     Manually assigned classification.
    SPLIT:      Transaction split across multiple categories.
    """
    NAICS = "naics"
    ISIC = "isic"
    UNSPSC = "unspsc"
    GL_ACCOUNT = "gl_account"
    KEYWORD = "keyword"
    MANUAL = "manual"
    SPLIT = "split"

class ClassificationStatus(str, Enum):
    """Status of the classification run."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"

# ---------------------------------------------------------------------------
# NAICS-to-Scope 3 Category Mapping (Top 100+ entries)
# ---------------------------------------------------------------------------
# Maps NAICS codes (2-6 digit prefixes) to Scope 3 categories and EEIO sectors.
# More specific codes take precedence over shorter prefixes.

NAICS_TO_SCOPE3: Dict[str, Dict[str, str]] = {
    # Agriculture, Forestry (11xxxx)
    "111": {"category": "cat_01_purchased_goods", "eeio_sector": "agriculture_forestry"},
    "112": {"category": "cat_01_purchased_goods", "eeio_sector": "agriculture_forestry"},
    "113": {"category": "cat_01_purchased_goods", "eeio_sector": "agriculture_forestry"},
    "114": {"category": "cat_01_purchased_goods", "eeio_sector": "fishing_aquaculture"},
    "115": {"category": "cat_01_purchased_goods", "eeio_sector": "agriculture_forestry"},
    # Mining (21xxxx)
    "211": {"category": "cat_01_purchased_goods", "eeio_sector": "oil_gas_extraction"},
    "212": {"category": "cat_01_purchased_goods", "eeio_sector": "mining_quarrying"},
    "213": {"category": "cat_01_purchased_goods", "eeio_sector": "mining_quarrying"},
    # Utilities (22xxxx)
    "2211": {"category": "cat_03_fuel_energy", "eeio_sector": "electricity_gas_steam"},
    "2212": {"category": "cat_03_fuel_energy", "eeio_sector": "electricity_gas_steam"},
    "2213": {"category": "cat_01_purchased_goods", "eeio_sector": "water_supply_sewerage"},
    # Construction (23xxxx)
    "236": {"category": "cat_02_capital_goods", "eeio_sector": "construction"},
    "237": {"category": "cat_02_capital_goods", "eeio_sector": "civil_engineering"},
    "238": {"category": "cat_01_purchased_goods", "eeio_sector": "construction"},
    # Manufacturing - Food (311-312)
    "311": {"category": "cat_01_purchased_goods", "eeio_sector": "food_beverages_tobacco"},
    "312": {"category": "cat_01_purchased_goods", "eeio_sector": "food_beverages_tobacco"},
    # Manufacturing - Textile (313-316)
    "313": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    "314": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    "315": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    "316": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    # Manufacturing - Wood/Paper (321-323)
    "321": {"category": "cat_01_purchased_goods", "eeio_sector": "wood_paper_products"},
    "322": {"category": "cat_01_purchased_goods", "eeio_sector": "wood_paper_products"},
    "323": {"category": "cat_01_purchased_goods", "eeio_sector": "wood_paper_products"},
    # Manufacturing - Petroleum/Chemical (324-326)
    "324": {"category": "cat_03_fuel_energy", "eeio_sector": "petroleum_refining"},
    "325": {"category": "cat_01_purchased_goods", "eeio_sector": "chemicals_pharmaceuticals"},
    "326": {"category": "cat_01_purchased_goods", "eeio_sector": "rubber_plastics"},
    # Manufacturing - Non-metallic minerals (327)
    "327": {"category": "cat_01_purchased_goods", "eeio_sector": "non_metallic_minerals"},
    # Manufacturing - Metals (331-332)
    "331": {"category": "cat_01_purchased_goods", "eeio_sector": "basic_metals"},
    "332": {"category": "cat_01_purchased_goods", "eeio_sector": "fabricated_metals"},
    # Manufacturing - Machinery/Electronics (333-335)
    "333": {"category": "cat_02_capital_goods", "eeio_sector": "machinery_equipment"},
    "334": {"category": "cat_01_purchased_goods", "eeio_sector": "electronics_optical"},
    "3341": {"category": "cat_02_capital_goods", "eeio_sector": "electronics_optical"},
    "335": {"category": "cat_01_purchased_goods", "eeio_sector": "electrical_equipment"},
    # Manufacturing - Transport Equipment (336)
    "336": {"category": "cat_02_capital_goods", "eeio_sector": "motor_vehicles"},
    "3361": {"category": "cat_02_capital_goods", "eeio_sector": "motor_vehicles"},
    "3364": {"category": "cat_02_capital_goods", "eeio_sector": "other_transport_equipment"},
    "3366": {"category": "cat_02_capital_goods", "eeio_sector": "other_transport_equipment"},
    # Manufacturing - Other (337, 339)
    "337": {"category": "cat_01_purchased_goods", "eeio_sector": "furniture_other_manufacturing"},
    "339": {"category": "cat_01_purchased_goods", "eeio_sector": "furniture_other_manufacturing"},
    # Wholesale Trade (42xxxx)
    "423": {"category": "cat_01_purchased_goods", "eeio_sector": "wholesale_trade"},
    "424": {"category": "cat_01_purchased_goods", "eeio_sector": "wholesale_trade"},
    "425": {"category": "cat_01_purchased_goods", "eeio_sector": "wholesale_trade"},
    # Retail Trade (44-45)
    "441": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "442": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "443": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "444": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "445": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "446": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "447": {"category": "cat_03_fuel_energy", "eeio_sector": "retail_trade"},
    "448": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "451": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "452": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "453": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "454": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    # Transportation (48-49)
    "481": {"category": "cat_06_business_travel", "eeio_sector": "air_transport"},
    "482": {"category": "cat_04_upstream_transport", "eeio_sector": "land_transport"},
    "483": {"category": "cat_04_upstream_transport", "eeio_sector": "water_transport"},
    "484": {"category": "cat_04_upstream_transport", "eeio_sector": "land_transport"},
    "485": {"category": "cat_07_employee_commuting", "eeio_sector": "land_transport"},
    "486": {"category": "cat_04_upstream_transport", "eeio_sector": "land_transport"},
    "488": {"category": "cat_04_upstream_transport", "eeio_sector": "warehousing_support"},
    "491": {"category": "cat_04_upstream_transport", "eeio_sector": "postal_courier"},
    "492": {"category": "cat_04_upstream_transport", "eeio_sector": "postal_courier"},
    "493": {"category": "cat_04_upstream_transport", "eeio_sector": "warehousing_support"},
    # Information (51xxxx)
    "511": {"category": "cat_01_purchased_goods", "eeio_sector": "publishing_broadcasting"},
    "512": {"category": "cat_01_purchased_goods", "eeio_sector": "publishing_broadcasting"},
    "515": {"category": "cat_01_purchased_goods", "eeio_sector": "telecommunications"},
    "517": {"category": "cat_01_purchased_goods", "eeio_sector": "telecommunications"},
    "518": {"category": "cat_01_purchased_goods", "eeio_sector": "it_services"},
    "519": {"category": "cat_01_purchased_goods", "eeio_sector": "it_services"},
    # Finance and Insurance (52xxxx)
    "521": {"category": "cat_15_investments", "eeio_sector": "financial_services"},
    "522": {"category": "cat_01_purchased_goods", "eeio_sector": "financial_services"},
    "523": {"category": "cat_15_investments", "eeio_sector": "financial_services"},
    "524": {"category": "cat_01_purchased_goods", "eeio_sector": "insurance"},
    # Real Estate (53xxxx)
    "531": {"category": "cat_08_upstream_leased", "eeio_sector": "real_estate"},
    "532": {"category": "cat_08_upstream_leased", "eeio_sector": "rental_leasing"},
    "533": {"category": "cat_08_upstream_leased", "eeio_sector": "rental_leasing"},
    # Professional Services (54xxxx)
    "5411": {"category": "cat_01_purchased_goods", "eeio_sector": "legal_accounting"},
    "5412": {"category": "cat_01_purchased_goods", "eeio_sector": "legal_accounting"},
    "5413": {"category": "cat_01_purchased_goods", "eeio_sector": "architecture_engineering"},
    "5414": {"category": "cat_01_purchased_goods", "eeio_sector": "architecture_engineering"},
    "5415": {"category": "cat_01_purchased_goods", "eeio_sector": "it_services"},
    "5416": {"category": "cat_01_purchased_goods", "eeio_sector": "management_consulting"},
    "5417": {"category": "cat_01_purchased_goods", "eeio_sector": "scientific_rd"},
    "5418": {"category": "cat_01_purchased_goods", "eeio_sector": "advertising_market_research"},
    # Administrative Services (56xxxx)
    "561": {"category": "cat_01_purchased_goods", "eeio_sector": "office_admin"},
    "5611": {"category": "cat_01_purchased_goods", "eeio_sector": "office_admin"},
    "5612": {"category": "cat_01_purchased_goods", "eeio_sector": "office_admin"},
    "5613": {"category": "cat_01_purchased_goods", "eeio_sector": "employment_services"},
    "5614": {"category": "cat_01_purchased_goods", "eeio_sector": "office_admin"},
    "5615": {"category": "cat_06_business_travel", "eeio_sector": "office_admin"},
    "5616": {"category": "cat_01_purchased_goods", "eeio_sector": "security_services"},
    "5617": {"category": "cat_01_purchased_goods", "eeio_sector": "cleaning_services"},
    "562": {"category": "cat_05_waste", "eeio_sector": "waste_management_remediation"},
    # Education (61xxxx)
    "611": {"category": "cat_01_purchased_goods", "eeio_sector": "education"},
    # Healthcare (62xxxx)
    "621": {"category": "cat_01_purchased_goods", "eeio_sector": "healthcare"},
    "622": {"category": "cat_01_purchased_goods", "eeio_sector": "healthcare"},
    "623": {"category": "cat_01_purchased_goods", "eeio_sector": "social_work"},
    "624": {"category": "cat_01_purchased_goods", "eeio_sector": "social_work"},
    # Accommodation and Food (72xxxx)
    "721": {"category": "cat_06_business_travel", "eeio_sector": "accommodation"},
    "722": {"category": "cat_01_purchased_goods", "eeio_sector": "food_service"},
    # Other Services (81xxxx)
    "811": {"category": "cat_01_purchased_goods", "eeio_sector": "repair_maintenance"},
    "812": {"category": "cat_01_purchased_goods", "eeio_sector": "personal_services"},
    "813": {"category": "cat_01_purchased_goods", "eeio_sector": "membership_organisations"},
}

# ---------------------------------------------------------------------------
# ISIC Rev 4 to Scope 3 Category Mapping (selected entries)
# ---------------------------------------------------------------------------

ISIC_TO_SCOPE3: Dict[str, Dict[str, str]] = {
    "01": {"category": "cat_01_purchased_goods", "eeio_sector": "agriculture_forestry"},
    "02": {"category": "cat_01_purchased_goods", "eeio_sector": "agriculture_forestry"},
    "03": {"category": "cat_01_purchased_goods", "eeio_sector": "fishing_aquaculture"},
    "05": {"category": "cat_01_purchased_goods", "eeio_sector": "mining_quarrying"},
    "06": {"category": "cat_03_fuel_energy", "eeio_sector": "oil_gas_extraction"},
    "07": {"category": "cat_01_purchased_goods", "eeio_sector": "mining_quarrying"},
    "08": {"category": "cat_01_purchased_goods", "eeio_sector": "mining_quarrying"},
    "10": {"category": "cat_01_purchased_goods", "eeio_sector": "food_beverages_tobacco"},
    "11": {"category": "cat_01_purchased_goods", "eeio_sector": "food_beverages_tobacco"},
    "13": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    "14": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    "15": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    "16": {"category": "cat_01_purchased_goods", "eeio_sector": "wood_paper_products"},
    "17": {"category": "cat_01_purchased_goods", "eeio_sector": "wood_paper_products"},
    "19": {"category": "cat_03_fuel_energy", "eeio_sector": "petroleum_refining"},
    "20": {"category": "cat_01_purchased_goods", "eeio_sector": "chemicals_pharmaceuticals"},
    "21": {"category": "cat_01_purchased_goods", "eeio_sector": "chemicals_pharmaceuticals"},
    "22": {"category": "cat_01_purchased_goods", "eeio_sector": "rubber_plastics"},
    "23": {"category": "cat_01_purchased_goods", "eeio_sector": "non_metallic_minerals"},
    "24": {"category": "cat_01_purchased_goods", "eeio_sector": "basic_metals"},
    "25": {"category": "cat_01_purchased_goods", "eeio_sector": "fabricated_metals"},
    "26": {"category": "cat_01_purchased_goods", "eeio_sector": "electronics_optical"},
    "27": {"category": "cat_01_purchased_goods", "eeio_sector": "electrical_equipment"},
    "28": {"category": "cat_02_capital_goods", "eeio_sector": "machinery_equipment"},
    "29": {"category": "cat_02_capital_goods", "eeio_sector": "motor_vehicles"},
    "30": {"category": "cat_02_capital_goods", "eeio_sector": "other_transport_equipment"},
    "31": {"category": "cat_01_purchased_goods", "eeio_sector": "furniture_other_manufacturing"},
    "33": {"category": "cat_01_purchased_goods", "eeio_sector": "repair_maintenance"},
    "35": {"category": "cat_03_fuel_energy", "eeio_sector": "electricity_gas_steam"},
    "36": {"category": "cat_01_purchased_goods", "eeio_sector": "water_supply_sewerage"},
    "37": {"category": "cat_05_waste", "eeio_sector": "waste_management_remediation"},
    "38": {"category": "cat_05_waste", "eeio_sector": "waste_management_remediation"},
    "41": {"category": "cat_02_capital_goods", "eeio_sector": "construction"},
    "42": {"category": "cat_02_capital_goods", "eeio_sector": "civil_engineering"},
    "43": {"category": "cat_01_purchased_goods", "eeio_sector": "construction"},
    "45": {"category": "cat_01_purchased_goods", "eeio_sector": "motor_vehicle_trade"},
    "46": {"category": "cat_01_purchased_goods", "eeio_sector": "wholesale_trade"},
    "47": {"category": "cat_01_purchased_goods", "eeio_sector": "retail_trade"},
    "49": {"category": "cat_04_upstream_transport", "eeio_sector": "land_transport"},
    "50": {"category": "cat_04_upstream_transport", "eeio_sector": "water_transport"},
    "51": {"category": "cat_06_business_travel", "eeio_sector": "air_transport"},
    "52": {"category": "cat_04_upstream_transport", "eeio_sector": "warehousing_support"},
    "53": {"category": "cat_04_upstream_transport", "eeio_sector": "postal_courier"},
    "55": {"category": "cat_06_business_travel", "eeio_sector": "accommodation"},
    "56": {"category": "cat_01_purchased_goods", "eeio_sector": "food_service"},
    "58": {"category": "cat_01_purchased_goods", "eeio_sector": "publishing_broadcasting"},
    "61": {"category": "cat_01_purchased_goods", "eeio_sector": "telecommunications"},
    "62": {"category": "cat_01_purchased_goods", "eeio_sector": "it_services"},
    "64": {"category": "cat_15_investments", "eeio_sector": "financial_services"},
    "65": {"category": "cat_01_purchased_goods", "eeio_sector": "insurance"},
    "68": {"category": "cat_08_upstream_leased", "eeio_sector": "real_estate"},
    "69": {"category": "cat_01_purchased_goods", "eeio_sector": "legal_accounting"},
    "70": {"category": "cat_01_purchased_goods", "eeio_sector": "management_consulting"},
    "71": {"category": "cat_01_purchased_goods", "eeio_sector": "architecture_engineering"},
    "72": {"category": "cat_01_purchased_goods", "eeio_sector": "scientific_rd"},
    "73": {"category": "cat_01_purchased_goods", "eeio_sector": "advertising_market_research"},
    "77": {"category": "cat_08_upstream_leased", "eeio_sector": "rental_leasing"},
    "78": {"category": "cat_01_purchased_goods", "eeio_sector": "employment_services"},
    "80": {"category": "cat_01_purchased_goods", "eeio_sector": "security_services"},
    "81": {"category": "cat_01_purchased_goods", "eeio_sector": "cleaning_services"},
    "82": {"category": "cat_01_purchased_goods", "eeio_sector": "office_admin"},
    "85": {"category": "cat_01_purchased_goods", "eeio_sector": "education"},
    "86": {"category": "cat_01_purchased_goods", "eeio_sector": "healthcare"},
    "87": {"category": "cat_01_purchased_goods", "eeio_sector": "social_work"},
    "90": {"category": "cat_01_purchased_goods", "eeio_sector": "arts_entertainment"},
    "93": {"category": "cat_01_purchased_goods", "eeio_sector": "sports_recreation"},
    "94": {"category": "cat_01_purchased_goods", "eeio_sector": "membership_organisations"},
    "95": {"category": "cat_01_purchased_goods", "eeio_sector": "repair_maintenance"},
    "96": {"category": "cat_01_purchased_goods", "eeio_sector": "personal_services"},
}

# ---------------------------------------------------------------------------
# UNSPSC to Scope 3 Category Mapping (selected 2-digit segments)
# ---------------------------------------------------------------------------

UNSPSC_TO_SCOPE3: Dict[str, Dict[str, str]] = {
    "10": {"category": "cat_01_purchased_goods", "eeio_sector": "mining_quarrying"},
    "11": {"category": "cat_01_purchased_goods", "eeio_sector": "chemicals_pharmaceuticals"},
    "12": {"category": "cat_01_purchased_goods", "eeio_sector": "chemicals_pharmaceuticals"},
    "13": {"category": "cat_01_purchased_goods", "eeio_sector": "rubber_plastics"},
    "14": {"category": "cat_01_purchased_goods", "eeio_sector": "wood_paper_products"},
    "15": {"category": "cat_03_fuel_energy", "eeio_sector": "petroleum_refining"},
    "20": {"category": "cat_02_capital_goods", "eeio_sector": "machinery_equipment"},
    "21": {"category": "cat_02_capital_goods", "eeio_sector": "machinery_equipment"},
    "22": {"category": "cat_02_capital_goods", "eeio_sector": "machinery_equipment"},
    "23": {"category": "cat_02_capital_goods", "eeio_sector": "machinery_equipment"},
    "24": {"category": "cat_01_purchased_goods", "eeio_sector": "fabricated_metals"},
    "25": {"category": "cat_02_capital_goods", "eeio_sector": "motor_vehicles"},
    "26": {"category": "cat_02_capital_goods", "eeio_sector": "electrical_equipment"},
    "27": {"category": "cat_01_purchased_goods", "eeio_sector": "electronics_optical"},
    "30": {"category": "cat_02_capital_goods", "eeio_sector": "construction"},
    "31": {"category": "cat_01_purchased_goods", "eeio_sector": "fabricated_metals"},
    "32": {"category": "cat_01_purchased_goods", "eeio_sector": "electronics_optical"},
    "39": {"category": "cat_01_purchased_goods", "eeio_sector": "electrical_equipment"},
    "40": {"category": "cat_01_purchased_goods", "eeio_sector": "electronics_optical"},
    "41": {"category": "cat_01_purchased_goods", "eeio_sector": "electronics_optical"},
    "42": {"category": "cat_01_purchased_goods", "eeio_sector": "healthcare"},
    "43": {"category": "cat_01_purchased_goods", "eeio_sector": "it_services"},
    "44": {"category": "cat_01_purchased_goods", "eeio_sector": "office_admin"},
    "46": {"category": "cat_05_waste", "eeio_sector": "waste_management_remediation"},
    "47": {"category": "cat_01_purchased_goods", "eeio_sector": "cleaning_services"},
    "48": {"category": "cat_01_purchased_goods", "eeio_sector": "food_service"},
    "49": {"category": "cat_01_purchased_goods", "eeio_sector": "sports_recreation"},
    "50": {"category": "cat_01_purchased_goods", "eeio_sector": "food_beverages_tobacco"},
    "51": {"category": "cat_01_purchased_goods", "eeio_sector": "chemicals_pharmaceuticals"},
    "52": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    "53": {"category": "cat_01_purchased_goods", "eeio_sector": "textiles_leather"},
    "55": {"category": "cat_01_purchased_goods", "eeio_sector": "publishing_broadcasting"},
    "56": {"category": "cat_01_purchased_goods", "eeio_sector": "furniture_other_manufacturing"},
    "60": {"category": "cat_01_purchased_goods", "eeio_sector": "personal_services"},
    "70": {"category": "cat_01_purchased_goods", "eeio_sector": "agriculture_forestry"},
    "71": {"category": "cat_02_capital_goods", "eeio_sector": "mining_quarrying"},
    "72": {"category": "cat_02_capital_goods", "eeio_sector": "construction"},
    "73": {"category": "cat_02_capital_goods", "eeio_sector": "machinery_equipment"},
    "76": {"category": "cat_03_fuel_energy", "eeio_sector": "electricity_gas_steam"},
    "77": {"category": "cat_01_purchased_goods", "eeio_sector": "water_supply_sewerage"},
    "78": {"category": "cat_04_upstream_transport", "eeio_sector": "land_transport"},
    "80": {"category": "cat_01_purchased_goods", "eeio_sector": "management_consulting"},
    "81": {"category": "cat_01_purchased_goods", "eeio_sector": "architecture_engineering"},
    "82": {"category": "cat_01_purchased_goods", "eeio_sector": "advertising_market_research"},
    "83": {"category": "cat_01_purchased_goods", "eeio_sector": "it_services"},
    "84": {"category": "cat_01_purchased_goods", "eeio_sector": "financial_services"},
    "85": {"category": "cat_01_purchased_goods", "eeio_sector": "healthcare"},
    "86": {"category": "cat_01_purchased_goods", "eeio_sector": "education"},
    "90": {"category": "cat_06_business_travel", "eeio_sector": "accommodation"},
    "91": {"category": "cat_01_purchased_goods", "eeio_sector": "personal_services"},
    "92": {"category": "cat_01_purchased_goods", "eeio_sector": "security_services"},
    "93": {"category": "cat_01_purchased_goods", "eeio_sector": "arts_entertainment"},
    "94": {"category": "cat_01_purchased_goods", "eeio_sector": "membership_organisations"},
    "95": {"category": "cat_01_purchased_goods", "eeio_sector": "personal_services"},
}

# ---------------------------------------------------------------------------
# GL Account Range to Scope 3 Category Mapping
# ---------------------------------------------------------------------------
# Maps GL account number ranges to Scope 3 categories.
# Format: (start, end, category, description)

GL_ACCOUNT_RANGES: List[Tuple[int, int, str, str]] = [
    # Cost of goods sold / raw materials
    (40000, 40999, "cat_01_purchased_goods", "Raw materials purchases"),
    (41000, 41999, "cat_01_purchased_goods", "Components and parts"),
    (42000, 42999, "cat_01_purchased_goods", "Packaging materials"),
    (43000, 43499, "cat_01_purchased_goods", "Subcontracted manufacturing"),
    (43500, 43999, "cat_01_purchased_goods", "Other purchased goods"),
    # Capital expenditure
    (44000, 44499, "cat_02_capital_goods", "Buildings and improvements"),
    (44500, 44999, "cat_02_capital_goods", "Machinery and equipment"),
    (45000, 45499, "cat_02_capital_goods", "IT equipment and hardware"),
    (45500, 45999, "cat_02_capital_goods", "Vehicles (owned)"),
    (46000, 46499, "cat_02_capital_goods", "Furniture and fixtures"),
    # Energy and utilities
    (50000, 50499, "cat_03_fuel_energy", "Electricity purchases"),
    (50500, 50999, "cat_03_fuel_energy", "Natural gas purchases"),
    (51000, 51499, "cat_03_fuel_energy", "Fuel purchases"),
    (51500, 51999, "cat_03_fuel_energy", "Steam and heating"),
    # Transport and logistics
    (52000, 52499, "cat_04_upstream_transport", "Inbound freight"),
    (52500, 52999, "cat_04_upstream_transport", "Third-party logistics"),
    (53000, 53499, "cat_09_downstream_transport", "Outbound freight"),
    (53500, 53999, "cat_04_upstream_transport", "Courier and postal"),
    # Waste management
    (54000, 54499, "cat_05_waste", "Waste disposal"),
    (54500, 54999, "cat_05_waste", "Recycling services"),
    (55000, 55499, "cat_05_waste", "Hazardous waste"),
    # Business travel
    (60000, 60499, "cat_06_business_travel", "Air travel"),
    (60500, 60999, "cat_06_business_travel", "Rail travel"),
    (61000, 61499, "cat_06_business_travel", "Hotel accommodation"),
    (61500, 61999, "cat_06_business_travel", "Car rental"),
    (62000, 62499, "cat_06_business_travel", "Travel agency fees"),
    (62500, 62999, "cat_06_business_travel", "Per diem and meals"),
    # Employee-related (commuting proxies)
    (63000, 63499, "cat_07_employee_commuting", "Commuter benefits"),
    (63500, 63999, "cat_07_employee_commuting", "Parking benefits"),
    (64000, 64499, "cat_07_employee_commuting", "Transit subsidies"),
    # Leased assets
    (65000, 65499, "cat_08_upstream_leased", "Building leases"),
    (65500, 65999, "cat_08_upstream_leased", "Equipment leases"),
    (66000, 66499, "cat_08_upstream_leased", "Vehicle leases"),
    (66500, 66999, "cat_13_downstream_leased", "Lease income - property"),
    (67000, 67499, "cat_13_downstream_leased", "Lease income - equipment"),
    # Franchise-related
    (68000, 68499, "cat_14_franchises", "Franchise fees received"),
    (68500, 68999, "cat_14_franchises", "Franchise support costs"),
    # Professional services (general purchased services)
    (70000, 70499, "cat_01_purchased_goods", "Legal services"),
    (70500, 70999, "cat_01_purchased_goods", "Accounting and audit"),
    (71000, 71499, "cat_01_purchased_goods", "Consulting services"),
    (71500, 71999, "cat_01_purchased_goods", "IT services"),
    (72000, 72499, "cat_01_purchased_goods", "Marketing and advertising"),
    (72500, 72999, "cat_01_purchased_goods", "Insurance premiums"),
    (73000, 73499, "cat_01_purchased_goods", "Maintenance and repairs"),
    (73500, 73999, "cat_01_purchased_goods", "Security services"),
    (74000, 74499, "cat_01_purchased_goods", "Cleaning services"),
    (74500, 74999, "cat_01_purchased_goods", "Temporary staffing"),
    # Investment-related
    (80000, 80499, "cat_15_investments", "Equity investments"),
    (80500, 80999, "cat_15_investments", "Debt investments"),
    (81000, 81499, "cat_15_investments", "Project finance"),
]

# ---------------------------------------------------------------------------
# Keyword Classification Dictionary (200+ keywords)
# ---------------------------------------------------------------------------
# Maps keywords found in transaction descriptions to Scope 3 categories.
# Keywords are checked in order; first match wins.

KEYWORD_TO_SCOPE3: List[Tuple[List[str], str, str]] = [
    # Category 1: Purchased Goods and Services
    (["raw material", "raw_material", "commodity", "ingredient"], "cat_01_purchased_goods", "Raw materials"),
    (["packaging", "carton", "box", "pallet", "shrink wrap"], "cat_01_purchased_goods", "Packaging"),
    (["office supplies", "stationery", "toner", "printer"], "cat_01_purchased_goods", "Office supplies"),
    (["cleaning", "janitorial", "sanitation"], "cat_01_purchased_goods", "Cleaning services"),
    (["catering", "canteen", "cafeteria", "food service"], "cat_01_purchased_goods", "Food services"),
    (["security guard", "surveillance", "alarm system"], "cat_01_purchased_goods", "Security"),
    (["consulting", "advisory", "management consulting"], "cat_01_purchased_goods", "Consulting"),
    (["legal", "law firm", "attorney", "solicitor"], "cat_01_purchased_goods", "Legal"),
    (["accounting", "audit", "bookkeeping", "tax prep"], "cat_01_purchased_goods", "Accounting"),
    (["marketing", "advertising", "media buy", "pr agency"], "cat_01_purchased_goods", "Marketing"),
    (["insurance", "premium", "policy renewal"], "cat_01_purchased_goods", "Insurance"),
    (["it service", "software", "cloud", "saas", "hosting"], "cat_01_purchased_goods", "IT services"),
    (["telecom", "telephone", "mobile plan", "internet"], "cat_01_purchased_goods", "Telecoms"),
    (["maintenance", "repair", "service contract"], "cat_01_purchased_goods", "Maintenance"),
    (["temporary staff", "temp agency", "contractor"], "cat_01_purchased_goods", "Staffing"),
    (["training", "education", "seminar", "workshop"], "cat_01_purchased_goods", "Training"),
    (["uniform", "workwear", "ppe", "safety gear"], "cat_01_purchased_goods", "PPE/Workwear"),
    (["water supply", "water treatment", "sewage"], "cat_01_purchased_goods", "Water"),
    (["chemical", "reagent", "laboratory", "lab supply"], "cat_01_purchased_goods", "Chemicals"),
    (["paper", "printing", "publication"], "cat_01_purchased_goods", "Paper/Printing"),
    (["component", "part", "subassembly", "assembly"], "cat_01_purchased_goods", "Components"),
    (["textile", "fabric", "clothing", "garment"], "cat_01_purchased_goods", "Textiles"),
    (["steel", "aluminium", "aluminum", "copper", "metal"], "cat_01_purchased_goods", "Metals"),
    (["plastic", "polymer", "resin", "pvc"], "cat_01_purchased_goods", "Plastics"),
    (["concrete", "cement", "aggregate", "gravel"], "cat_01_purchased_goods", "Construction materials"),
    (["glass", "ceramic", "brick", "tile"], "cat_01_purchased_goods", "Minerals"),
    (["medical supply", "pharmaceutical", "drug", "medicine"], "cat_01_purchased_goods", "Medical supplies"),
    (["electronics", "computer", "laptop", "monitor", "phone"], "cat_01_purchased_goods", "Electronics"),
    (["furniture", "desk", "chair", "filing cabinet"], "cat_01_purchased_goods", "Furniture"),
    # Category 2: Capital Goods
    (["machinery", "heavy equipment", "production line"], "cat_02_capital_goods", "Machinery"),
    (["building construction", "renovation", "extension"], "cat_02_capital_goods", "Construction"),
    (["server", "data center", "data centre", "rack"], "cat_02_capital_goods", "IT infrastructure"),
    (["vehicle purchase", "fleet purchase", "truck purchase"], "cat_02_capital_goods", "Vehicles"),
    (["capital expenditure", "capex", "asset purchase"], "cat_02_capital_goods", "Capital expenditure"),
    (["hvac system", "cooling system", "boiler install"], "cat_02_capital_goods", "HVAC"),
    (["solar panel", "wind turbine", "battery storage"], "cat_02_capital_goods", "Renewable energy assets"),
    # Category 3: Fuel and Energy
    (["electricity", "power supply", "grid electricity"], "cat_03_fuel_energy", "Electricity"),
    (["natural gas", "gas supply", "methane"], "cat_03_fuel_energy", "Natural gas"),
    (["diesel", "petrol", "gasoline", "fuel oil"], "cat_03_fuel_energy", "Transport fuel"),
    (["heating oil", "lpg", "propane", "kerosene"], "cat_03_fuel_energy", "Heating fuel"),
    (["coal", "coke", "solid fuel"], "cat_03_fuel_energy", "Solid fuels"),
    (["renewable energy certificate", "rec", "go cert"], "cat_03_fuel_energy", "RECs"),
    (["steam", "district heating", "chilled water"], "cat_03_fuel_energy", "District energy"),
    # Category 4: Upstream Transportation
    (["freight", "shipping", "haulage", "trucking"], "cat_04_upstream_transport", "Freight"),
    (["courier", "parcel", "express delivery", "fedex"], "cat_04_upstream_transport", "Courier"),
    (["warehousing", "storage", "fulfilment", "fulfillment"], "cat_04_upstream_transport", "Warehousing"),
    (["customs", "import duty", "brokerage"], "cat_04_upstream_transport", "Customs"),
    (["logistics", "3pl", "supply chain"], "cat_04_upstream_transport", "Logistics"),
    # Category 5: Waste
    (["waste disposal", "landfill", "skip hire"], "cat_05_waste", "Waste disposal"),
    (["recycling", "recycle", "material recovery"], "cat_05_waste", "Recycling"),
    (["hazardous waste", "toxic waste", "clinical waste"], "cat_05_waste", "Hazardous waste"),
    (["wastewater", "effluent", "sewage treatment"], "cat_05_waste", "Wastewater"),
    (["composting", "organic waste", "food waste"], "cat_05_waste", "Composting"),
    (["incineration", "waste to energy", "wte"], "cat_05_waste", "Incineration"),
    # Category 6: Business Travel
    (["air travel", "flight", "airline", "airfare"], "cat_06_business_travel", "Air travel"),
    (["rail travel", "train ticket", "eurostar", "amtrak"], "cat_06_business_travel", "Rail"),
    (["hotel", "accommodation", "lodging"], "cat_06_business_travel", "Accommodation"),
    (["car rental", "hire car", "rental car"], "cat_06_business_travel", "Car rental"),
    (["taxi", "uber", "ride share", "rideshare", "cab"], "cat_06_business_travel", "Taxi"),
    (["travel agent", "travel management", "tmc"], "cat_06_business_travel", "Travel management"),
    (["conference", "event attendance", "exhibition"], "cat_06_business_travel", "Events"),
    (["per diem", "meal allowance", "subsistence"], "cat_06_business_travel", "Per diem"),
    # Category 7: Employee Commuting
    (["commuter pass", "transit pass", "bus pass"], "cat_07_employee_commuting", "Transit passes"),
    (["parking", "car park", "employee parking"], "cat_07_employee_commuting", "Parking"),
    (["bicycle scheme", "bike to work", "cycle"], "cat_07_employee_commuting", "Cycling"),
    (["shuttle bus", "employee transport"], "cat_07_employee_commuting", "Shuttle"),
    (["work from home", "remote work", "telework"], "cat_07_employee_commuting", "Remote work"),
    # Category 8: Upstream Leased Assets
    (["office lease", "building rent", "property rent"], "cat_08_upstream_leased", "Building lease"),
    (["equipment lease", "equipment rental"], "cat_08_upstream_leased", "Equipment lease"),
    (["vehicle lease", "fleet lease", "car lease"], "cat_08_upstream_leased", "Vehicle lease"),
    # Category 9: Downstream Transportation
    (["outbound freight", "delivery", "last mile"], "cat_09_downstream_transport", "Outbound freight"),
    (["distribution", "customer delivery"], "cat_09_downstream_transport", "Distribution"),
    # Category 13: Downstream Leased Assets
    (["rental income", "lease income", "tenant"], "cat_13_downstream_leased", "Lease income"),
    # Category 14: Franchises
    (["franchise fee", "franchise royalty", "franchisee"], "cat_14_franchises", "Franchise"),
    # Category 15: Investments
    (["equity investment", "share purchase", "stock"], "cat_15_investments", "Equity"),
    (["bond", "debt investment", "fixed income"], "cat_15_investments", "Debt"),
    (["project finance", "venture capital", "private equity"], "cat_15_investments", "Project finance"),
    (["real estate investment", "reit", "property fund"], "cat_15_investments", "Real estate investment"),
]

# ---------------------------------------------------------------------------
# Currency Exchange Rates (to EUR, approximate mid-2025)
# ---------------------------------------------------------------------------

CURRENCY_TO_EUR: Dict[str, Decimal] = {
    "EUR": Decimal("1.000000"),
    "USD": Decimal("0.920000"),
    "GBP": Decimal("1.160000"),
    "CHF": Decimal("1.040000"),
    "JPY": Decimal("0.006100"),
    "CNY": Decimal("0.127000"),
    "INR": Decimal("0.011000"),
    "KRW": Decimal("0.000690"),
    "BRL": Decimal("0.183000"),
    "CAD": Decimal("0.680000"),
    "AUD": Decimal("0.610000"),
    "NZD": Decimal("0.560000"),
    "SEK": Decimal("0.087000"),
    "NOK": Decimal("0.085000"),
    "DKK": Decimal("0.134000"),
    "PLN": Decimal("0.232000"),
    "CZK": Decimal("0.040000"),
    "HUF": Decimal("0.002600"),
    "RON": Decimal("0.201000"),
    "BGN": Decimal("0.511000"),
    "HRK": Decimal("0.133000"),
    "TRY": Decimal("0.028000"),
    "ZAR": Decimal("0.050000"),
    "MXN": Decimal("0.054000"),
    "ARS": Decimal("0.001100"),
    "CLP": Decimal("0.001000"),
    "COP": Decimal("0.000230"),
    "PEN": Decimal("0.245000"),
    "THB": Decimal("0.026000"),
    "MYR": Decimal("0.197000"),
    "SGD": Decimal("0.690000"),
    "HKD": Decimal("0.118000"),
    "TWD": Decimal("0.029000"),
    "PHP": Decimal("0.016400"),
    "IDR": Decimal("0.000058"),
    "VND": Decimal("0.000037"),
    "AED": Decimal("0.250000"),
    "SAR": Decimal("0.245000"),
    "QAR": Decimal("0.253000"),
    "KWD": Decimal("2.990000"),
    "BHD": Decimal("2.440000"),
    "OMR": Decimal("2.390000"),
    "EGP": Decimal("0.019000"),
    "NGN": Decimal("0.000600"),
    "KES": Decimal("0.006400"),
    "GHS": Decimal("0.063000"),
    "TZS": Decimal("0.000360"),
    "UGX": Decimal("0.000240"),
    "PKR": Decimal("0.003300"),
    "BDT": Decimal("0.008400"),
    "LKR": Decimal("0.003100"),
    "RUB": Decimal("0.010000"),
    "UAH": Decimal("0.022000"),
    "ILS": Decimal("0.250000"),
}

# CPI indices (2020 = 100) for inflation adjustment
CPI_INDICES: Dict[int, Decimal] = {
    2018: Decimal("96.0"),
    2019: Decimal("97.8"),
    2020: Decimal("100.0"),
    2021: Decimal("103.2"),
    2022: Decimal("111.4"),
    2023: Decimal("116.8"),
    2024: Decimal("120.5"),
    2025: Decimal("123.0"),
    2026: Decimal("125.5"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class SpendTransaction(BaseModel):
    """A single spend/procurement transaction.

    Attributes:
        transaction_id: Unique transaction identifier.
        supplier_name: Supplier name.
        description: Transaction description.
        amount: Transaction amount in original currency.
        currency: Currency code (ISO 4217).
        transaction_date: Transaction date.
        fiscal_year: Fiscal year.
        naics_code: NAICS code (if available).
        isic_code: ISIC code (if available).
        unspsc_code: UNSPSC code (if available).
        gl_account: GL account number (if available).
        cost_center: Cost center (if available).
        supplier_country: Supplier country code.
        manual_category: Manually assigned Scope 3 category (if any).
    """
    transaction_id: str = Field(default_factory=_new_uuid, description="Transaction ID")
    supplier_name: str = Field(default="", max_length=500, description="Supplier name")
    description: str = Field(default="", max_length=2000, description="Description")
    amount: Decimal = Field(default=Decimal("0"), description="Amount in original currency")
    currency: str = Field(default="EUR", max_length=3, description="Currency code")
    transaction_date: Optional[str] = Field(default=None, description="Transaction date")
    fiscal_year: int = Field(default=2025, description="Fiscal year")
    naics_code: Optional[str] = Field(default=None, max_length=6, description="NAICS code")
    isic_code: Optional[str] = Field(default=None, max_length=4, description="ISIC code")
    unspsc_code: Optional[str] = Field(default=None, max_length=8, description="UNSPSC code")
    gl_account: Optional[int] = Field(default=None, description="GL account number")
    cost_center: Optional[str] = Field(default=None, description="Cost center")
    supplier_country: str = Field(default="", max_length=2, description="Supplier country")
    manual_category: Optional[str] = Field(default=None, description="Manual category override")

class SplitAllocation(BaseModel):
    """Allocation of a split transaction across multiple categories.

    Attributes:
        category: Scope 3 category for this allocation.
        fraction: Fraction of transaction amount (0 to 1).
        amount_eur: Allocated amount in EUR.
        reason: Reason for the allocation.
    """
    category: str = Field(..., description="Scope 3 category")
    fraction: Decimal = Field(default=Decimal("1"), ge=0, le=1, description="Fraction")
    amount_eur: Decimal = Field(default=Decimal("0"), description="Amount EUR")
    reason: str = Field(default="", description="Allocation reason")

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ClassifiedTransaction(BaseModel):
    """A spend transaction with classification result.

    Attributes:
        transaction_id: Original transaction identifier.
        supplier_name: Supplier name.
        description: Transaction description.
        original_amount: Amount in original currency.
        original_currency: Original currency code.
        amount_eur: Amount normalised to EUR.
        amount_eur_cpi_adjusted: Amount adjusted for CPI inflation.
        scope3_category: Assigned Scope 3 category.
        eeio_sector: Assigned EEIO sector.
        classification_method: Method used for classification.
        confidence: Confidence level.
        confidence_reason: Reason for confidence level.
        is_split: Whether the transaction is split.
        split_allocations: Allocations for split transactions.
        fiscal_year: Fiscal year.
        supplier_country: Supplier country code.
    """
    transaction_id: str = Field(default="", description="Transaction ID")
    supplier_name: str = Field(default="", description="Supplier name")
    description: str = Field(default="", description="Description")
    original_amount: Decimal = Field(default=Decimal("0"), description="Original amount")
    original_currency: str = Field(default="EUR", description="Original currency")
    amount_eur: Decimal = Field(default=Decimal("0"), description="Amount EUR")
    amount_eur_cpi_adjusted: Decimal = Field(
        default=Decimal("0"), description="CPI-adjusted amount EUR"
    )
    scope3_category: str = Field(default="", description="Scope 3 category")
    eeio_sector: str = Field(default="", description="EEIO sector")
    classification_method: ClassificationMethod = Field(
        default=ClassificationMethod.KEYWORD, description="Classification method"
    )
    confidence: ClassificationConfidence = Field(
        default=ClassificationConfidence.UNCLASSIFIED, description="Confidence"
    )
    confidence_reason: str = Field(default="", description="Confidence reason")
    is_split: bool = Field(default=False, description="Is split transaction")
    split_allocations: List[SplitAllocation] = Field(
        default_factory=list, description="Split allocations"
    )
    fiscal_year: int = Field(default=2025, description="Fiscal year")
    supplier_country: str = Field(default="", description="Supplier country")

class CategorySpendSummary(BaseModel):
    """Summary of classified spend for a single Scope 3 category.

    Attributes:
        category: Scope 3 category.
        total_spend_eur: Total spend in EUR.
        total_spend_eur_cpi_adjusted: CPI-adjusted total spend.
        transaction_count: Number of transactions classified to this category.
        high_confidence_count: Transactions with HIGH confidence.
        medium_confidence_count: Transactions with MEDIUM confidence.
        low_confidence_count: Transactions with LOW confidence.
        top_eeio_sectors: Top EEIO sectors by spend.
        share_of_total_pct: Share of total classified spend (%).
    """
    category: str = Field(default="", description="Scope 3 category")
    total_spend_eur: Decimal = Field(default=Decimal("0"), description="Total spend EUR")
    total_spend_eur_cpi_adjusted: Decimal = Field(
        default=Decimal("0"), description="CPI-adjusted total"
    )
    transaction_count: int = Field(default=0, description="Transaction count")
    high_confidence_count: int = Field(default=0, description="High confidence count")
    medium_confidence_count: int = Field(default=0, description="Medium confidence count")
    low_confidence_count: int = Field(default=0, description="Low confidence count")
    top_eeio_sectors: Dict[str, Decimal] = Field(
        default_factory=dict, description="Top EEIO sectors by spend"
    )
    share_of_total_pct: Decimal = Field(default=Decimal("0"), description="Share of total %")

class ClassificationResult(BaseModel):
    """Complete spend classification result.

    Attributes:
        result_id: Unique result identifier.
        total_transactions: Total transactions processed.
        classified_count: Successfully classified count.
        unclassified_count: Unclassified count.
        classification_rate_pct: Classification rate (%).
        total_spend_eur: Total spend in EUR.
        classified_spend_eur: Classified spend in EUR.
        classified_transactions: Per-transaction classification results.
        category_summaries: Per-category summaries.
        confidence_distribution: Distribution across confidence levels.
        method_distribution: Distribution across classification methods.
        target_year: CPI adjustment target year.
        warnings: Warnings generated.
        status: Classification status.
        calculated_at: Timestamp.
        processing_time_ms: Processing time ms.
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    total_transactions: int = Field(default=0, description="Total transactions")
    classified_count: int = Field(default=0, description="Classified count")
    unclassified_count: int = Field(default=0, description="Unclassified count")
    classification_rate_pct: Decimal = Field(
        default=Decimal("0"), description="Classification rate %"
    )
    total_spend_eur: Decimal = Field(default=Decimal("0"), description="Total spend EUR")
    classified_spend_eur: Decimal = Field(
        default=Decimal("0"), description="Classified spend EUR"
    )
    classified_transactions: List[ClassifiedTransaction] = Field(
        default_factory=list, description="Classified transactions"
    )
    category_summaries: List[CategorySpendSummary] = Field(
        default_factory=list, description="Category summaries"
    )
    confidence_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Confidence distribution"
    )
    method_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Method distribution"
    )
    target_year: int = Field(default=2025, description="CPI target year")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    status: ClassificationStatus = Field(
        default=ClassificationStatus.COMPLETE, description="Status"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(default=Decimal("0"), description="Processing time ms")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

SpendTransaction.model_rebuild()
SplitAllocation.model_rebuild()
ClassifiedTransaction.model_rebuild()
CategorySpendSummary.model_rebuild()
ClassificationResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SpendClassificationEngine:
    """Deterministic spend data classification into Scope 3 categories.

    Classifies procurement transactions using a cascading lookup approach:
    NAICS -> ISIC -> UNSPSC -> GL account -> keyword fallback.

    Each classification receives a confidence score reflecting the
    specificity of the match.  The engine also normalises currencies to
    EUR and applies CPI-based inflation adjustment.

    Attributes:
        _target_year: Target year for CPI adjustment.
        _custom_fx_rates: Custom FX rate overrides.
        _custom_gl_ranges: Custom GL account range mappings.
        _warnings: Warnings generated during classification.

    Example:
        >>> engine = SpendClassificationEngine()
        >>> txns = [SpendTransaction(description="Office supplies", amount=Decimal("500"))]
        >>> result = engine.classify_transactions(txns)
        >>> print(result.classified_count)
    """

    def __init__(
        self,
        target_year: int = 2025,
        custom_fx_rates: Optional[Dict[str, Decimal]] = None,
        custom_gl_ranges: Optional[List[Tuple[int, int, str, str]]] = None,
    ) -> None:
        """Initialise SpendClassificationEngine.

        Args:
            target_year: Target year for CPI adjustment.
            custom_fx_rates: Custom FX rate overrides (currency -> EUR).
            custom_gl_ranges: Custom GL account ranges to append.
        """
        self._target_year = target_year
        self._custom_fx_rates = custom_fx_rates or {}
        self._gl_ranges = list(GL_ACCOUNT_RANGES)
        if custom_gl_ranges:
            self._gl_ranges.extend(custom_gl_ranges)
        self._warnings: List[str] = []
        logger.info(
            "SpendClassificationEngine v%s initialised (target_year=%d)",
            _MODULE_VERSION,
            self._target_year,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_transactions(
        self,
        transactions: List[SpendTransaction],
    ) -> ClassificationResult:
        """Classify a list of spend transactions into Scope 3 categories.

        Main entry point.  Iterates through all transactions and applies
        the classification cascade.

        Args:
            transactions: List of spend transactions.

        Returns:
            ClassificationResult with per-transaction results and summaries.

        Raises:
            ValueError: If transactions list is empty.
        """
        t0 = time.perf_counter()
        self._warnings = []

        if not transactions:
            raise ValueError("At least one transaction is required")

        logger.info("Classifying %d transactions", len(transactions))

        classified: List[ClassifiedTransaction] = []
        classified_count = 0
        unclassified_count = 0
        total_spend_eur = Decimal("0")
        classified_spend_eur = Decimal("0")

        for txn in transactions:
            result = self._classify_single(txn)
            classified.append(result)
            total_spend_eur += result.amount_eur

            if result.confidence != ClassificationConfidence.UNCLASSIFIED:
                classified_count += 1
                classified_spend_eur += result.amount_eur
            else:
                unclassified_count += 1

        # Build summaries
        cat_summaries = self._build_category_summaries(classified, total_spend_eur)
        conf_dist = self._build_confidence_distribution(classified)
        method_dist = self._build_method_distribution(classified)

        classification_rate = _safe_divide(
            _decimal(classified_count) * Decimal("100"),
            _decimal(len(transactions)),
        )

        elapsed_ms = Decimal(str((time.perf_counter() - t0) * 1000))
        result = ClassificationResult(
            total_transactions=len(transactions),
            classified_count=classified_count,
            unclassified_count=unclassified_count,
            classification_rate_pct=_round_val(classification_rate, 2),
            total_spend_eur=_round_val(total_spend_eur, 2),
            classified_spend_eur=_round_val(classified_spend_eur, 2),
            classified_transactions=classified,
            category_summaries=cat_summaries,
            confidence_distribution=conf_dist,
            method_distribution=method_dist,
            target_year=self._target_year,
            warnings=list(self._warnings),
            status=ClassificationStatus.COMPLETE,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = self._compute_provenance(result)

        logger.info(
            "Classification complete: %d/%d classified (%.1f%%)",
            classified_count,
            len(transactions),
            classification_rate,
        )
        return result

    def classify_single(
        self,
        transaction: SpendTransaction,
    ) -> ClassifiedTransaction:
        """Classify a single spend transaction.

        Convenience method for classifying one transaction.

        Args:
            transaction: Spend transaction.

        Returns:
            ClassifiedTransaction.
        """
        self._warnings = []
        return self._classify_single(transaction)

    def normalise_currency(
        self,
        amount: Decimal,
        currency: str,
    ) -> Decimal:
        """Convert an amount to EUR using stored FX rates.

        Args:
            amount: Amount in original currency.
            currency: ISO 4217 currency code.

        Returns:
            Amount in EUR.
        """
        return self._convert_to_eur(amount, currency)

    def adjust_for_inflation(
        self,
        amount_eur: Decimal,
        base_year: int,
        target_year: Optional[int] = None,
    ) -> Decimal:
        """Adjust an EUR amount for CPI inflation.

        Args:
            amount_eur: Amount in EUR.
            base_year: Base year of the amount.
            target_year: Target year for adjustment (defaults to engine target).

        Returns:
            CPI-adjusted amount in EUR.
        """
        ty = target_year if target_year is not None else self._target_year
        return self._cpi_adjust(amount_eur, base_year, ty)

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _classify_single(self, txn: SpendTransaction) -> ClassifiedTransaction:
        """Classify a single transaction using the cascading lookup.

        Args:
            txn: Spend transaction.

        Returns:
            ClassifiedTransaction with category and confidence.
        """
        # Step 0: Currency normalisation
        amount_eur = self._convert_to_eur(txn.amount, txn.currency)
        amount_eur_adj = self._cpi_adjust(amount_eur, txn.fiscal_year, self._target_year)

        # Step 0.5: Check for manual override
        if txn.manual_category:
            return ClassifiedTransaction(
                transaction_id=txn.transaction_id,
                supplier_name=txn.supplier_name,
                description=txn.description,
                original_amount=txn.amount,
                original_currency=txn.currency,
                amount_eur=_round_val(amount_eur, 2),
                amount_eur_cpi_adjusted=_round_val(amount_eur_adj, 2),
                scope3_category=txn.manual_category,
                eeio_sector="",
                classification_method=ClassificationMethod.MANUAL,
                confidence=ClassificationConfidence.HIGH,
                confidence_reason="Manual category override",
                fiscal_year=txn.fiscal_year,
                supplier_country=txn.supplier_country,
            )

        # Step 1: NAICS lookup
        if txn.naics_code:
            result = self._classify_by_naics(txn.naics_code)
            if result:
                return self._build_classified(
                    txn, amount_eur, amount_eur_adj, result,
                    ClassificationMethod.NAICS,
                    ClassificationConfidence.HIGH,
                    f"NAICS code {txn.naics_code} matched",
                )

        # Step 2: ISIC lookup
        if txn.isic_code:
            result = self._classify_by_isic(txn.isic_code)
            if result:
                return self._build_classified(
                    txn, amount_eur, amount_eur_adj, result,
                    ClassificationMethod.ISIC,
                    ClassificationConfidence.HIGH,
                    f"ISIC code {txn.isic_code} matched",
                )

        # Step 3: UNSPSC lookup
        if txn.unspsc_code:
            result = self._classify_by_unspsc(txn.unspsc_code)
            if result:
                return self._build_classified(
                    txn, amount_eur, amount_eur_adj, result,
                    ClassificationMethod.UNSPSC,
                    ClassificationConfidence.HIGH,
                    f"UNSPSC code {txn.unspsc_code} matched",
                )

        # Step 4: GL account lookup
        if txn.gl_account is not None:
            result = self._classify_by_gl_account(txn.gl_account)
            if result:
                return self._build_classified(
                    txn, amount_eur, amount_eur_adj, result,
                    ClassificationMethod.GL_ACCOUNT,
                    ClassificationConfidence.MEDIUM,
                    f"GL account {txn.gl_account} matched range: {result.get('description', '')}",
                )

        # Step 5: Keyword fallback
        if txn.description:
            result = self._classify_by_keywords(txn.description)
            if result:
                return self._build_classified(
                    txn, amount_eur, amount_eur_adj, result,
                    ClassificationMethod.KEYWORD,
                    ClassificationConfidence.LOW,
                    f"Keyword match: {result.get('matched_keyword', '')}",
                )

        # Unclassified
        return ClassifiedTransaction(
            transaction_id=txn.transaction_id,
            supplier_name=txn.supplier_name,
            description=txn.description,
            original_amount=txn.amount,
            original_currency=txn.currency,
            amount_eur=_round_val(amount_eur, 2),
            amount_eur_cpi_adjusted=_round_val(amount_eur_adj, 2),
            scope3_category="unclassified",
            eeio_sector="",
            classification_method=ClassificationMethod.KEYWORD,
            confidence=ClassificationConfidence.UNCLASSIFIED,
            confidence_reason="No match found in any lookup table",
            fiscal_year=txn.fiscal_year,
            supplier_country=txn.supplier_country,
        )

    def _classify_by_naics(self, code: str) -> Optional[Dict[str, str]]:
        """Classify using NAICS code lookup.

        Tries longest prefix match first (6, 5, 4, 3 digits).

        Args:
            code: NAICS code string.

        Returns:
            Dict with category and eeio_sector, or None.
        """
        clean = code.strip()
        # Try longest prefix first
        for length in range(len(clean), 2, -1):
            prefix = clean[:length]
            if prefix in NAICS_TO_SCOPE3:
                return dict(NAICS_TO_SCOPE3[prefix])
        return None

    def _classify_by_isic(self, code: str) -> Optional[Dict[str, str]]:
        """Classify using ISIC Rev 4 code lookup.

        Tries 4-digit, then 2-digit prefix.

        Args:
            code: ISIC code string.

        Returns:
            Dict with category and eeio_sector, or None.
        """
        clean = code.strip()
        for length in range(len(clean), 1, -1):
            prefix = clean[:length]
            if prefix in ISIC_TO_SCOPE3:
                return dict(ISIC_TO_SCOPE3[prefix])
        return None

    def _classify_by_unspsc(self, code: str) -> Optional[Dict[str, str]]:
        """Classify using UNSPSC code lookup.

        Uses 2-digit segment prefix.

        Args:
            code: UNSPSC code string.

        Returns:
            Dict with category and eeio_sector, or None.
        """
        clean = code.strip()
        if len(clean) >= 2:
            prefix = clean[:2]
            if prefix in UNSPSC_TO_SCOPE3:
                return dict(UNSPSC_TO_SCOPE3[prefix])
        return None

    def _classify_by_gl_account(self, account: int) -> Optional[Dict[str, str]]:
        """Classify using GL account range lookup.

        Checks if the account number falls within any defined range.

        Args:
            account: GL account number.

        Returns:
            Dict with category, eeio_sector, and description, or None.
        """
        for start, end, category, description in self._gl_ranges:
            if start <= account <= end:
                return {
                    "category": category,
                    "eeio_sector": "",
                    "description": description,
                }
        return None

    def _classify_by_keywords(self, description: str) -> Optional[Dict[str, str]]:
        """Classify using keyword matching on transaction description.

        Scans through the keyword dictionary and returns the first match.

        Args:
            description: Transaction description text.

        Returns:
            Dict with category, eeio_sector, and matched_keyword, or None.
        """
        desc_lower = description.lower()
        for keywords, category, group in KEYWORD_TO_SCOPE3:
            for kw in keywords:
                if kw.lower() in desc_lower:
                    return {
                        "category": category,
                        "eeio_sector": "",
                        "matched_keyword": kw,
                        "keyword_group": group,
                    }
        return None

    def _build_classified(
        self,
        txn: SpendTransaction,
        amount_eur: Decimal,
        amount_eur_adj: Decimal,
        classification: Dict[str, str],
        method: ClassificationMethod,
        confidence: ClassificationConfidence,
        reason: str,
    ) -> ClassifiedTransaction:
        """Build a ClassifiedTransaction from classification result.

        Args:
            txn: Original transaction.
            amount_eur: EUR amount.
            amount_eur_adj: CPI-adjusted EUR amount.
            classification: Classification dict.
            method: Classification method used.
            confidence: Confidence level.
            reason: Reason for classification.

        Returns:
            ClassifiedTransaction.
        """
        return ClassifiedTransaction(
            transaction_id=txn.transaction_id,
            supplier_name=txn.supplier_name,
            description=txn.description,
            original_amount=txn.amount,
            original_currency=txn.currency,
            amount_eur=_round_val(amount_eur, 2),
            amount_eur_cpi_adjusted=_round_val(amount_eur_adj, 2),
            scope3_category=classification.get("category", ""),
            eeio_sector=classification.get("eeio_sector", ""),
            classification_method=method,
            confidence=confidence,
            confidence_reason=reason,
            fiscal_year=txn.fiscal_year,
            supplier_country=txn.supplier_country,
        )

    def _convert_to_eur(self, amount: Decimal, currency: str) -> Decimal:
        """Convert amount to EUR.

        Args:
            amount: Amount in original currency.
            currency: ISO 4217 currency code.

        Returns:
            Amount in EUR.
        """
        if currency == "EUR":
            return amount

        # Check custom rates first
        rate = self._custom_fx_rates.get(currency)
        if rate is None:
            rate = CURRENCY_TO_EUR.get(currency)
        if rate is None:
            self._warnings.append(f"No FX rate for currency '{currency}'; using 1:1")
            rate = Decimal("1")

        return amount * rate

    def _cpi_adjust(
        self,
        amount_eur: Decimal,
        base_year: int,
        target_year: int,
    ) -> Decimal:
        """Adjust amount for CPI inflation.

        Args:
            amount_eur: Amount in EUR.
            base_year: Base year.
            target_year: Target year.

        Returns:
            CPI-adjusted amount.
        """
        if base_year == target_year:
            return amount_eur

        cpi_base = CPI_INDICES.get(base_year)
        cpi_target = CPI_INDICES.get(target_year)

        if cpi_base is None or cpi_target is None:
            # No CPI data; return unadjusted
            return amount_eur

        return amount_eur * _safe_divide(cpi_target, cpi_base, Decimal("1"))

    def _build_category_summaries(
        self,
        classified: List[ClassifiedTransaction],
        total_spend_eur: Decimal,
    ) -> List[CategorySpendSummary]:
        """Build per-category spend summaries.

        Args:
            classified: List of classified transactions.
            total_spend_eur: Total spend in EUR.

        Returns:
            List of CategorySpendSummary.
        """
        cat_data: Dict[str, Dict[str, Any]] = {}

        for txn in classified:
            cat = txn.scope3_category
            if cat not in cat_data:
                cat_data[cat] = {
                    "total_eur": Decimal("0"),
                    "total_adj": Decimal("0"),
                    "count": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "sectors": {},
                }

            d = cat_data[cat]
            d["total_eur"] += txn.amount_eur
            d["total_adj"] += txn.amount_eur_cpi_adjusted
            d["count"] += 1

            if txn.confidence == ClassificationConfidence.HIGH:
                d["high"] += 1
            elif txn.confidence == ClassificationConfidence.MEDIUM:
                d["medium"] += 1
            elif txn.confidence == ClassificationConfidence.LOW:
                d["low"] += 1

            if txn.eeio_sector:
                d["sectors"][txn.eeio_sector] = (
                    d["sectors"].get(txn.eeio_sector, Decimal("0")) + txn.amount_eur
                )

        summaries: List[CategorySpendSummary] = []
        for cat, d in sorted(cat_data.items()):
            share = _safe_divide(
                d["total_eur"] * Decimal("100"), total_spend_eur
            )
            # Top 5 sectors by spend
            sorted_sectors = sorted(
                d["sectors"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            top_sectors = {k: _round_val(v, 2) for k, v in sorted_sectors}

            summaries.append(CategorySpendSummary(
                category=cat,
                total_spend_eur=_round_val(d["total_eur"], 2),
                total_spend_eur_cpi_adjusted=_round_val(d["total_adj"], 2),
                transaction_count=d["count"],
                high_confidence_count=d["high"],
                medium_confidence_count=d["medium"],
                low_confidence_count=d["low"],
                top_eeio_sectors=top_sectors,
                share_of_total_pct=_round_val(share, 2),
            ))

        return summaries

    def _build_confidence_distribution(
        self,
        classified: List[ClassifiedTransaction],
    ) -> Dict[str, int]:
        """Build confidence level distribution.

        Args:
            classified: Classified transactions.

        Returns:
            Dict mapping confidence level to count.
        """
        dist: Dict[str, int] = {
            "high": 0, "medium": 0, "low": 0, "unclassified": 0,
        }
        for txn in classified:
            key = txn.confidence.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def _build_method_distribution(
        self,
        classified: List[ClassifiedTransaction],
    ) -> Dict[str, int]:
        """Build classification method distribution.

        Args:
            classified: Classified transactions.

        Returns:
            Dict mapping method to count.
        """
        dist: Dict[str, int] = {}
        for txn in classified:
            key = txn.classification_method.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def _compute_provenance(self, result: ClassificationResult) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            result: Classification result.

        Returns:
            SHA-256 hex digest.
        """
        return _compute_hash(result)
