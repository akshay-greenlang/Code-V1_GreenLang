# -*- coding: utf-8 -*-
"""
Scope3ScreeningEngine - PACK-042 Scope 3 Starter Pack Engine 1
================================================================

Rapid screening-level Scope 3 assessment per GHG Protocol Scope 3 Standard
Chapter 6.  Performs an initial materiality screening across all 15 Scope 3
categories using sector-specific EEIO (environmentally extended input-output)
emission intensities, revenue-based estimates, and relevance scoring.

The engine is designed for organisations beginning their Scope 3 journey that
need to identify which categories are material before investing in primary
data collection.  It follows a three-step process:

    1. Map the organisation's sector profile to EEIO intensity factors.
    2. Estimate each category's emissions using spend, revenue, or
       headcount proxies depending on the category type.
    3. Score each category for relevance (magnitude, data availability,
       stakeholder interest, outsourcing potential) and flag those exceeding
       the significance threshold.

Calculation Methodology:
    Screening Estimate per Category:
        E_cat = Activity_proxy * EEIO_intensity_factor

    Revenue-Based Downstream (Cat 10-12):
        E_downstream = Revenue * downstream_intensity_per_EUR

    Relevance Score (0-100):
        R = w_mag * S_magnitude + w_data * S_data_avail
            + w_stake * S_stakeholder + w_out * S_outsourcing
        where w_mag=0.40, w_data=0.20, w_stake=0.20, w_out=0.20

    Significance Threshold:
        Category is significant if E_cat / E_total >= threshold (default 1%)

Regulatory References:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard, Ch 6 (Screening)
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions (2013)
    - GHG Protocol Scope 3 Evaluator Tool methodology
    - Exiobase 3 MRIO database (emission intensity factors)
    - IPCC AR5 GWP-100 values

Zero-Hallucination:
    - All emission estimates use deterministic EEIO intensity lookup tables
    - Relevance scoring uses weighted formula with fixed coefficients
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-042 Scope 3 Starter
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope3Category(str, Enum):
    """All 15 Scope 3 categories per GHG Protocol.

    Upstream categories (1-8) cover emissions from purchased goods,
    capital goods, fuel and energy, transportation, waste, travel,
    commuting, and leased assets.

    Downstream categories (9-15) cover emissions from distribution,
    product processing, product use, end-of-life, leased assets,
    franchises, and investments.
    """
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

class RelevanceTier(str, Enum):
    """Relevance tier for a Scope 3 category.

    HIGH:       Category is clearly significant and must be reported.
    MEDIUM:     Category may be significant; further investigation needed.
    LOW:        Category is unlikely to be material.
    NOT_APPLICABLE: Category does not apply to this organisation.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_APPLICABLE = "not_applicable"

class DataAvailabilityLevel(str, Enum):
    """Level of data availability for a Scope 3 category.

    READILY_AVAILABLE:  Primary data or detailed records exist.
    PARTIALLY_AVAILABLE: Some data exists, proxies needed for gaps.
    LIMITED:            Only high-level proxies available.
    NONE:               No data available.
    """
    READILY_AVAILABLE = "readily_available"
    PARTIALLY_AVAILABLE = "partially_available"
    LIMITED = "limited"
    NONE = "none"

class ScreeningStatus(str, Enum):
    """Status of the screening assessment.

    COMPLETE:   All 15 categories screened successfully.
    PARTIAL:    Some categories could not be screened.
    ERROR:      Screening failed due to errors.
    """
    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"

# ---------------------------------------------------------------------------
# EEIO Emission Intensity Factors (kgCO2e per EUR of spend)
# ---------------------------------------------------------------------------
# Source: Derived from Exiobase 3 MRIO database, 2022 base year.
# These are representative sector-level intensities for screening purposes.
# Actual EEIO models would use full multi-regional tables.

EEIO_INTENSITIES: Dict[str, Decimal] = {
    # Primary and extractive sectors
    "agriculture_forestry": Decimal("2.85"),
    "mining_quarrying": Decimal("1.42"),
    "oil_gas_extraction": Decimal("2.10"),
    "fishing_aquaculture": Decimal("2.30"),
    # Manufacturing sectors
    "food_beverages_tobacco": Decimal("1.65"),
    "textiles_leather": Decimal("1.20"),
    "wood_paper_products": Decimal("0.98"),
    "chemicals_pharmaceuticals": Decimal("1.55"),
    "rubber_plastics": Decimal("1.78"),
    "non_metallic_minerals": Decimal("2.40"),
    "basic_metals": Decimal("2.95"),
    "fabricated_metals": Decimal("1.10"),
    "electronics_optical": Decimal("0.45"),
    "electrical_equipment": Decimal("0.62"),
    "machinery_equipment": Decimal("0.58"),
    "motor_vehicles": Decimal("0.72"),
    "other_transport_equipment": Decimal("0.68"),
    "furniture_other_manufacturing": Decimal("0.85"),
    "petroleum_refining": Decimal("3.20"),
    "coke_products": Decimal("4.10"),
    # Utilities
    "electricity_gas_steam": Decimal("3.50"),
    "water_supply_sewerage": Decimal("0.95"),
    "waste_management_remediation": Decimal("1.80"),
    # Construction
    "construction": Decimal("1.15"),
    "civil_engineering": Decimal("1.35"),
    # Services - trade
    "wholesale_trade": Decimal("0.25"),
    "retail_trade": Decimal("0.30"),
    "motor_vehicle_trade": Decimal("0.35"),
    # Services - transport and logistics
    "land_transport": Decimal("1.90"),
    "water_transport": Decimal("2.60"),
    "air_transport": Decimal("3.80"),
    "warehousing_support": Decimal("0.55"),
    "postal_courier": Decimal("0.70"),
    # Services - hospitality
    "accommodation": Decimal("0.48"),
    "food_service": Decimal("0.65"),
    # Services - ICT
    "telecommunications": Decimal("0.22"),
    "it_services": Decimal("0.18"),
    "publishing_broadcasting": Decimal("0.15"),
    # Services - financial
    "financial_services": Decimal("0.12"),
    "insurance": Decimal("0.10"),
    "real_estate": Decimal("0.20"),
    # Services - professional
    "legal_accounting": Decimal("0.14"),
    "management_consulting": Decimal("0.16"),
    "architecture_engineering": Decimal("0.19"),
    "scientific_rd": Decimal("0.25"),
    "advertising_market_research": Decimal("0.13"),
    # Services - administrative
    "rental_leasing": Decimal("0.28"),
    "employment_services": Decimal("0.11"),
    "security_services": Decimal("0.17"),
    "cleaning_services": Decimal("0.35"),
    "office_admin": Decimal("0.12"),
    # Public and social
    "public_administration": Decimal("0.22"),
    "education": Decimal("0.18"),
    "healthcare": Decimal("0.32"),
    "social_work": Decimal("0.20"),
    # Other
    "arts_entertainment": Decimal("0.15"),
    "sports_recreation": Decimal("0.22"),
    "membership_organisations": Decimal("0.10"),
    "repair_maintenance": Decimal("0.40"),
    "personal_services": Decimal("0.18"),
}

# ---------------------------------------------------------------------------
# NAICS 2-digit Sector-to-EEIO Mapping
# ---------------------------------------------------------------------------
# Maps NAICS 2-digit sector codes to EEIO intensity keys and typical
# Scope 3 category relevance profiles.

NAICS_SECTOR_PROFILES: Dict[str, Dict[str, Any]] = {
    "11": {
        "name": "Agriculture, Forestry, Fishing and Hunting",
        "eeio_key": "agriculture_forestry",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.25"), Scope3Category.CAT_02: Decimal("0.05"),
            Scope3Category.CAT_03: Decimal("0.12"), Scope3Category.CAT_04: Decimal("0.10"),
            Scope3Category.CAT_05: Decimal("0.08"), Scope3Category.CAT_06: Decimal("0.02"),
            Scope3Category.CAT_07: Decimal("0.03"), Scope3Category.CAT_08: Decimal("0.02"),
            Scope3Category.CAT_09: Decimal("0.10"), Scope3Category.CAT_10: Decimal("0.05"),
            Scope3Category.CAT_11: Decimal("0.08"), Scope3Category.CAT_12: Decimal("0.05"),
            Scope3Category.CAT_13: Decimal("0.01"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.03"),
        },
    },
    "21": {
        "name": "Mining, Quarrying, and Oil and Gas Extraction",
        "eeio_key": "mining_quarrying",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.15"), Scope3Category.CAT_02: Decimal("0.10"),
            Scope3Category.CAT_03: Decimal("0.15"), Scope3Category.CAT_04: Decimal("0.12"),
            Scope3Category.CAT_05: Decimal("0.05"), Scope3Category.CAT_06: Decimal("0.03"),
            Scope3Category.CAT_07: Decimal("0.02"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.08"), Scope3Category.CAT_10: Decimal("0.10"),
            Scope3Category.CAT_11: Decimal("0.05"), Scope3Category.CAT_12: Decimal("0.03"),
            Scope3Category.CAT_13: Decimal("0.02"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.04"),
        },
    },
    "22": {
        "name": "Utilities",
        "eeio_key": "electricity_gas_steam",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.10"), Scope3Category.CAT_02: Decimal("0.08"),
            Scope3Category.CAT_03: Decimal("0.35"), Scope3Category.CAT_04: Decimal("0.05"),
            Scope3Category.CAT_05: Decimal("0.05"), Scope3Category.CAT_06: Decimal("0.02"),
            Scope3Category.CAT_07: Decimal("0.02"), Scope3Category.CAT_08: Decimal("0.01"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.20"), Scope3Category.CAT_12: Decimal("0.02"),
            Scope3Category.CAT_13: Decimal("0.03"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.04"),
        },
    },
    "23": {
        "name": "Construction",
        "eeio_key": "construction",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.35"), Scope3Category.CAT_02: Decimal("0.10"),
            Scope3Category.CAT_03: Decimal("0.08"), Scope3Category.CAT_04: Decimal("0.12"),
            Scope3Category.CAT_05: Decimal("0.08"), Scope3Category.CAT_06: Decimal("0.03"),
            Scope3Category.CAT_07: Decimal("0.04"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.05"), Scope3Category.CAT_12: Decimal("0.03"),
            Scope3Category.CAT_13: Decimal("0.02"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.02"),
        },
    },
    "31": {
        "name": "Manufacturing - Food, Beverage, Textile",
        "eeio_key": "food_beverages_tobacco",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.40"), Scope3Category.CAT_02: Decimal("0.05"),
            Scope3Category.CAT_03: Decimal("0.08"), Scope3Category.CAT_04: Decimal("0.10"),
            Scope3Category.CAT_05: Decimal("0.05"), Scope3Category.CAT_06: Decimal("0.02"),
            Scope3Category.CAT_07: Decimal("0.02"), Scope3Category.CAT_08: Decimal("0.01"),
            Scope3Category.CAT_09: Decimal("0.08"), Scope3Category.CAT_10: Decimal("0.03"),
            Scope3Category.CAT_11: Decimal("0.08"), Scope3Category.CAT_12: Decimal("0.04"),
            Scope3Category.CAT_13: Decimal("0.01"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.02"),
        },
    },
    "32": {
        "name": "Manufacturing - Wood, Paper, Chemical, Plastics",
        "eeio_key": "chemicals_pharmaceuticals",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.35"), Scope3Category.CAT_02: Decimal("0.06"),
            Scope3Category.CAT_03: Decimal("0.10"), Scope3Category.CAT_04: Decimal("0.08"),
            Scope3Category.CAT_05: Decimal("0.06"), Scope3Category.CAT_06: Decimal("0.02"),
            Scope3Category.CAT_07: Decimal("0.02"), Scope3Category.CAT_08: Decimal("0.01"),
            Scope3Category.CAT_09: Decimal("0.07"), Scope3Category.CAT_10: Decimal("0.05"),
            Scope3Category.CAT_11: Decimal("0.10"), Scope3Category.CAT_12: Decimal("0.04"),
            Scope3Category.CAT_13: Decimal("0.01"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.02"),
        },
    },
    "33": {
        "name": "Manufacturing - Metal, Electronics, Transport Equipment",
        "eeio_key": "fabricated_metals",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.30"), Scope3Category.CAT_02: Decimal("0.08"),
            Scope3Category.CAT_03: Decimal("0.07"), Scope3Category.CAT_04: Decimal("0.08"),
            Scope3Category.CAT_05: Decimal("0.04"), Scope3Category.CAT_06: Decimal("0.02"),
            Scope3Category.CAT_07: Decimal("0.02"), Scope3Category.CAT_08: Decimal("0.01"),
            Scope3Category.CAT_09: Decimal("0.08"), Scope3Category.CAT_10: Decimal("0.05"),
            Scope3Category.CAT_11: Decimal("0.15"), Scope3Category.CAT_12: Decimal("0.05"),
            Scope3Category.CAT_13: Decimal("0.01"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.03"),
        },
    },
    "42": {
        "name": "Wholesale Trade",
        "eeio_key": "wholesale_trade",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.45"), Scope3Category.CAT_02: Decimal("0.03"),
            Scope3Category.CAT_03: Decimal("0.04"), Scope3Category.CAT_04: Decimal("0.15"),
            Scope3Category.CAT_05: Decimal("0.05"), Scope3Category.CAT_06: Decimal("0.03"),
            Scope3Category.CAT_07: Decimal("0.03"), Scope3Category.CAT_08: Decimal("0.03"),
            Scope3Category.CAT_09: Decimal("0.08"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.03"), Scope3Category.CAT_12: Decimal("0.02"),
            Scope3Category.CAT_13: Decimal("0.02"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.03"),
        },
    },
    "44": {
        "name": "Retail Trade",
        "eeio_key": "retail_trade",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.55"), Scope3Category.CAT_02: Decimal("0.03"),
            Scope3Category.CAT_03: Decimal("0.03"), Scope3Category.CAT_04: Decimal("0.12"),
            Scope3Category.CAT_05: Decimal("0.04"), Scope3Category.CAT_06: Decimal("0.02"),
            Scope3Category.CAT_07: Decimal("0.03"), Scope3Category.CAT_08: Decimal("0.02"),
            Scope3Category.CAT_09: Decimal("0.08"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.02"), Scope3Category.CAT_12: Decimal("0.02"),
            Scope3Category.CAT_13: Decimal("0.01"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.02"),
        },
    },
    "45": {
        "name": "Retail Trade (continued)",
        "eeio_key": "retail_trade",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.55"), Scope3Category.CAT_02: Decimal("0.03"),
            Scope3Category.CAT_03: Decimal("0.03"), Scope3Category.CAT_04: Decimal("0.12"),
            Scope3Category.CAT_05: Decimal("0.04"), Scope3Category.CAT_06: Decimal("0.02"),
            Scope3Category.CAT_07: Decimal("0.03"), Scope3Category.CAT_08: Decimal("0.02"),
            Scope3Category.CAT_09: Decimal("0.08"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.02"), Scope3Category.CAT_12: Decimal("0.02"),
            Scope3Category.CAT_13: Decimal("0.01"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.02"),
        },
    },
    "48": {
        "name": "Transportation and Warehousing",
        "eeio_key": "land_transport",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.10"), Scope3Category.CAT_02: Decimal("0.08"),
            Scope3Category.CAT_03: Decimal("0.30"), Scope3Category.CAT_04: Decimal("0.05"),
            Scope3Category.CAT_05: Decimal("0.03"), Scope3Category.CAT_06: Decimal("0.03"),
            Scope3Category.CAT_07: Decimal("0.05"), Scope3Category.CAT_08: Decimal("0.10"),
            Scope3Category.CAT_09: Decimal("0.08"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.05"), Scope3Category.CAT_12: Decimal("0.03"),
            Scope3Category.CAT_13: Decimal("0.05"), Scope3Category.CAT_14: Decimal("0.02"),
            Scope3Category.CAT_15: Decimal("0.03"),
        },
    },
    "49": {
        "name": "Transportation and Warehousing (continued)",
        "eeio_key": "warehousing_support",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.10"), Scope3Category.CAT_02: Decimal("0.08"),
            Scope3Category.CAT_03: Decimal("0.30"), Scope3Category.CAT_04: Decimal("0.05"),
            Scope3Category.CAT_05: Decimal("0.03"), Scope3Category.CAT_06: Decimal("0.03"),
            Scope3Category.CAT_07: Decimal("0.05"), Scope3Category.CAT_08: Decimal("0.10"),
            Scope3Category.CAT_09: Decimal("0.08"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.05"), Scope3Category.CAT_12: Decimal("0.03"),
            Scope3Category.CAT_13: Decimal("0.05"), Scope3Category.CAT_14: Decimal("0.02"),
            Scope3Category.CAT_15: Decimal("0.03"),
        },
    },
    "51": {
        "name": "Information",
        "eeio_key": "it_services",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.25"), Scope3Category.CAT_02: Decimal("0.10"),
            Scope3Category.CAT_03: Decimal("0.05"), Scope3Category.CAT_04: Decimal("0.03"),
            Scope3Category.CAT_05: Decimal("0.03"), Scope3Category.CAT_06: Decimal("0.10"),
            Scope3Category.CAT_07: Decimal("0.05"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.20"), Scope3Category.CAT_12: Decimal("0.05"),
            Scope3Category.CAT_13: Decimal("0.02"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.04"),
        },
    },
    "52": {
        "name": "Finance and Insurance",
        "eeio_key": "financial_services",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.15"), Scope3Category.CAT_02: Decimal("0.05"),
            Scope3Category.CAT_03: Decimal("0.02"), Scope3Category.CAT_04: Decimal("0.02"),
            Scope3Category.CAT_05: Decimal("0.02"), Scope3Category.CAT_06: Decimal("0.08"),
            Scope3Category.CAT_07: Decimal("0.05"), Scope3Category.CAT_08: Decimal("0.03"),
            Scope3Category.CAT_09: Decimal("0.00"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.00"), Scope3Category.CAT_12: Decimal("0.00"),
            Scope3Category.CAT_13: Decimal("0.03"), Scope3Category.CAT_14: Decimal("0.02"),
            Scope3Category.CAT_15: Decimal("0.53"),
        },
    },
    "53": {
        "name": "Real Estate and Rental and Leasing",
        "eeio_key": "real_estate",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.15"), Scope3Category.CAT_02: Decimal("0.10"),
            Scope3Category.CAT_03: Decimal("0.05"), Scope3Category.CAT_04: Decimal("0.03"),
            Scope3Category.CAT_05: Decimal("0.03"), Scope3Category.CAT_06: Decimal("0.03"),
            Scope3Category.CAT_07: Decimal("0.03"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.10"), Scope3Category.CAT_12: Decimal("0.02"),
            Scope3Category.CAT_13: Decimal("0.30"), Scope3Category.CAT_14: Decimal("0.02"),
            Scope3Category.CAT_15: Decimal("0.07"),
        },
    },
    "54": {
        "name": "Professional, Scientific, and Technical Services",
        "eeio_key": "management_consulting",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.20"), Scope3Category.CAT_02: Decimal("0.08"),
            Scope3Category.CAT_03: Decimal("0.03"), Scope3Category.CAT_04: Decimal("0.03"),
            Scope3Category.CAT_05: Decimal("0.03"), Scope3Category.CAT_06: Decimal("0.15"),
            Scope3Category.CAT_07: Decimal("0.10"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.10"), Scope3Category.CAT_12: Decimal("0.03"),
            Scope3Category.CAT_13: Decimal("0.05"), Scope3Category.CAT_14: Decimal("0.03"),
            Scope3Category.CAT_15: Decimal("0.10"),
        },
    },
    "55": {
        "name": "Management of Companies and Enterprises",
        "eeio_key": "management_consulting",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.18"), Scope3Category.CAT_02: Decimal("0.05"),
            Scope3Category.CAT_03: Decimal("0.03"), Scope3Category.CAT_04: Decimal("0.02"),
            Scope3Category.CAT_05: Decimal("0.02"), Scope3Category.CAT_06: Decimal("0.10"),
            Scope3Category.CAT_07: Decimal("0.08"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.01"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.02"), Scope3Category.CAT_12: Decimal("0.02"),
            Scope3Category.CAT_13: Decimal("0.05"), Scope3Category.CAT_14: Decimal("0.05"),
            Scope3Category.CAT_15: Decimal("0.32"),
        },
    },
    "56": {
        "name": "Administrative and Support Services",
        "eeio_key": "office_admin",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.25"), Scope3Category.CAT_02: Decimal("0.05"),
            Scope3Category.CAT_03: Decimal("0.03"), Scope3Category.CAT_04: Decimal("0.05"),
            Scope3Category.CAT_05: Decimal("0.05"), Scope3Category.CAT_06: Decimal("0.08"),
            Scope3Category.CAT_07: Decimal("0.10"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.03"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.10"), Scope3Category.CAT_12: Decimal("0.05"),
            Scope3Category.CAT_13: Decimal("0.05"), Scope3Category.CAT_14: Decimal("0.02"),
            Scope3Category.CAT_15: Decimal("0.09"),
        },
    },
    "61": {
        "name": "Educational Services",
        "eeio_key": "education",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.25"), Scope3Category.CAT_02: Decimal("0.10"),
            Scope3Category.CAT_03: Decimal("0.05"), Scope3Category.CAT_04: Decimal("0.03"),
            Scope3Category.CAT_05: Decimal("0.04"), Scope3Category.CAT_06: Decimal("0.08"),
            Scope3Category.CAT_07: Decimal("0.15"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.01"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.05"), Scope3Category.CAT_12: Decimal("0.03"),
            Scope3Category.CAT_13: Decimal("0.05"), Scope3Category.CAT_14: Decimal("0.03"),
            Scope3Category.CAT_15: Decimal("0.08"),
        },
    },
    "62": {
        "name": "Health Care and Social Assistance",
        "eeio_key": "healthcare",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.35"), Scope3Category.CAT_02: Decimal("0.10"),
            Scope3Category.CAT_03: Decimal("0.05"), Scope3Category.CAT_04: Decimal("0.08"),
            Scope3Category.CAT_05: Decimal("0.08"), Scope3Category.CAT_06: Decimal("0.05"),
            Scope3Category.CAT_07: Decimal("0.08"), Scope3Category.CAT_08: Decimal("0.03"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.05"), Scope3Category.CAT_12: Decimal("0.05"),
            Scope3Category.CAT_13: Decimal("0.02"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.03"),
        },
    },
    "71": {
        "name": "Arts, Entertainment, and Recreation",
        "eeio_key": "arts_entertainment",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.20"), Scope3Category.CAT_02: Decimal("0.08"),
            Scope3Category.CAT_03: Decimal("0.05"), Scope3Category.CAT_04: Decimal("0.05"),
            Scope3Category.CAT_05: Decimal("0.05"), Scope3Category.CAT_06: Decimal("0.08"),
            Scope3Category.CAT_07: Decimal("0.10"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.10"), Scope3Category.CAT_12: Decimal("0.05"),
            Scope3Category.CAT_13: Decimal("0.08"), Scope3Category.CAT_14: Decimal("0.03"),
            Scope3Category.CAT_15: Decimal("0.06"),
        },
    },
    "72": {
        "name": "Accommodation and Food Services",
        "eeio_key": "accommodation",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.40"), Scope3Category.CAT_02: Decimal("0.05"),
            Scope3Category.CAT_03: Decimal("0.08"), Scope3Category.CAT_04: Decimal("0.08"),
            Scope3Category.CAT_05: Decimal("0.10"), Scope3Category.CAT_06: Decimal("0.02"),
            Scope3Category.CAT_07: Decimal("0.05"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.05"), Scope3Category.CAT_12: Decimal("0.05"),
            Scope3Category.CAT_13: Decimal("0.02"), Scope3Category.CAT_14: Decimal("0.01"),
            Scope3Category.CAT_15: Decimal("0.02"),
        },
    },
    "81": {
        "name": "Other Services (except Public Administration)",
        "eeio_key": "personal_services",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.25"), Scope3Category.CAT_02: Decimal("0.05"),
            Scope3Category.CAT_03: Decimal("0.05"), Scope3Category.CAT_04: Decimal("0.05"),
            Scope3Category.CAT_05: Decimal("0.05"), Scope3Category.CAT_06: Decimal("0.05"),
            Scope3Category.CAT_07: Decimal("0.10"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.03"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.12"), Scope3Category.CAT_12: Decimal("0.05"),
            Scope3Category.CAT_13: Decimal("0.05"), Scope3Category.CAT_14: Decimal("0.03"),
            Scope3Category.CAT_15: Decimal("0.07"),
        },
    },
    "92": {
        "name": "Public Administration",
        "eeio_key": "public_administration",
        "typical_profile": {
            Scope3Category.CAT_01: Decimal("0.30"), Scope3Category.CAT_02: Decimal("0.10"),
            Scope3Category.CAT_03: Decimal("0.05"), Scope3Category.CAT_04: Decimal("0.05"),
            Scope3Category.CAT_05: Decimal("0.05"), Scope3Category.CAT_06: Decimal("0.05"),
            Scope3Category.CAT_07: Decimal("0.10"), Scope3Category.CAT_08: Decimal("0.05"),
            Scope3Category.CAT_09: Decimal("0.02"), Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.05"), Scope3Category.CAT_12: Decimal("0.03"),
            Scope3Category.CAT_13: Decimal("0.05"), Scope3Category.CAT_14: Decimal("0.03"),
            Scope3Category.CAT_15: Decimal("0.07"),
        },
    },
}

# Downstream revenue-based intensity factors (kgCO2e per EUR revenue).
# Used for categories 10-12 when spend data is not available.
DOWNSTREAM_INTENSITY_BY_SECTOR: Dict[str, Dict[str, Decimal]] = {
    "manufacturing_heavy": {
        "cat_10": Decimal("0.55"),
        "cat_11": Decimal("1.20"),
        "cat_12": Decimal("0.35"),
    },
    "manufacturing_light": {
        "cat_10": Decimal("0.30"),
        "cat_11": Decimal("0.80"),
        "cat_12": Decimal("0.25"),
    },
    "retail": {
        "cat_10": Decimal("0.00"),
        "cat_11": Decimal("0.40"),
        "cat_12": Decimal("0.15"),
    },
    "technology": {
        "cat_10": Decimal("0.00"),
        "cat_11": Decimal("0.60"),
        "cat_12": Decimal("0.10"),
    },
    "financial_services": {
        "cat_10": Decimal("0.00"),
        "cat_11": Decimal("0.00"),
        "cat_12": Decimal("0.00"),
    },
    "utilities": {
        "cat_10": Decimal("0.00"),
        "cat_11": Decimal("2.50"),
        "cat_12": Decimal("0.05"),
    },
    "transportation": {
        "cat_10": Decimal("0.00"),
        "cat_11": Decimal("0.90"),
        "cat_12": Decimal("0.15"),
    },
    "healthcare": {
        "cat_10": Decimal("0.00"),
        "cat_11": Decimal("0.30"),
        "cat_12": Decimal("0.20"),
    },
    "construction": {
        "cat_10": Decimal("0.10"),
        "cat_11": Decimal("0.70"),
        "cat_12": Decimal("0.25"),
    },
    "services_general": {
        "cat_10": Decimal("0.00"),
        "cat_11": Decimal("0.15"),
        "cat_12": Decimal("0.05"),
    },
}

# Relevance scoring weights per GHG Protocol Scope 3 Standard Chapter 6
RELEVANCE_WEIGHTS: Dict[str, Decimal] = {
    "magnitude": Decimal("0.40"),
    "data_availability": Decimal("0.20"),
    "stakeholder_interest": Decimal("0.20"),
    "outsourcing_potential": Decimal("0.20"),
}

# Default significance threshold (1% of estimated total Scope 3)
DEFAULT_SIGNIFICANCE_THRESHOLD_PCT: Decimal = Decimal("1.0")

# Scope 3 category descriptions for reporting
CATEGORY_DESCRIPTIONS: Dict[Scope3Category, str] = {
    Scope3Category.CAT_01: "Purchased goods and services",
    Scope3Category.CAT_02: "Capital goods",
    Scope3Category.CAT_03: "Fuel- and energy-related activities not included in Scope 1 or 2",
    Scope3Category.CAT_04: "Upstream transportation and distribution",
    Scope3Category.CAT_05: "Waste generated in operations",
    Scope3Category.CAT_06: "Business travel",
    Scope3Category.CAT_07: "Employee commuting",
    Scope3Category.CAT_08: "Upstream leased assets",
    Scope3Category.CAT_09: "Downstream transportation and distribution",
    Scope3Category.CAT_10: "Processing of sold products",
    Scope3Category.CAT_11: "Use of sold products",
    Scope3Category.CAT_12: "End-of-life treatment of sold products",
    Scope3Category.CAT_13: "Downstream leased assets",
    Scope3Category.CAT_14: "Franchises",
    Scope3Category.CAT_15: "Investments",
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class OrgProfile(BaseModel):
    """Organisation profile for Scope 3 screening.

    Attributes:
        org_id: Organisation identifier.
        org_name: Organisation name.
        naics_2_digit: Primary NAICS 2-digit sector code.
        secondary_naics: Additional NAICS codes (if multi-sector).
        annual_revenue_eur: Annual revenue in EUR.
        annual_spend_eur: Total annual procurement spend in EUR.
        employee_count: Number of full-time equivalent employees.
        scope1_tco2e: Total Scope 1 emissions (tCO2e).
        scope2_tco2e: Total Scope 2 emissions (tCO2e).
        has_franchises: Whether the org operates franchises.
        has_investments: Whether the org has significant equity investments.
        has_leased_assets_upstream: Whether the org leases assets (as lessee).
        has_leased_assets_downstream: Whether the org leases assets (as lessor).
        produces_physical_products: Whether the org manufactures physical goods.
        downstream_sector_key: Sector key for downstream intensity lookup.
        custom_eeio_overrides: Custom EEIO intensity overrides by category.
        reporting_year: Reporting year.
        currency: Currency code for spend/revenue figures.
    """
    org_id: str = Field(default_factory=_new_uuid, description="Organisation ID")
    org_name: str = Field(default="", max_length=500, description="Organisation name")
    naics_2_digit: str = Field(default="", max_length=2, description="NAICS 2-digit code")
    secondary_naics: List[str] = Field(default_factory=list, description="Secondary NAICS codes")
    annual_revenue_eur: Decimal = Field(default=Decimal("0"), ge=0, description="Annual revenue EUR")
    annual_spend_eur: Decimal = Field(default=Decimal("0"), ge=0, description="Annual spend EUR")
    employee_count: int = Field(default=0, ge=0, description="Employee headcount")
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 1 tCO2e")
    scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 2 tCO2e")
    has_franchises: bool = Field(default=False, description="Has franchises")
    has_investments: bool = Field(default=False, description="Has equity investments")
    has_leased_assets_upstream: bool = Field(default=False, description="Leases assets (lessee)")
    has_leased_assets_downstream: bool = Field(default=False, description="Leases assets (lessor)")
    produces_physical_products: bool = Field(default=True, description="Produces physical products")
    downstream_sector_key: str = Field(
        default="services_general", description="Downstream sector for intensity lookup"
    )
    custom_eeio_overrides: Dict[str, Decimal] = Field(
        default_factory=dict, description="Custom EEIO overrides"
    )
    reporting_year: int = Field(default=2025, ge=2000, le=2100, description="Reporting year")
    currency: str = Field(default="EUR", max_length=3, description="Currency code")

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class RelevanceScore(BaseModel):
    """Multi-dimensional relevance score for a Scope 3 category.

    Attributes:
        magnitude_score: Score based on estimated emission magnitude (0-100).
        data_availability_score: Score based on data availability (0-100).
        stakeholder_interest_score: Score based on stakeholder interest (0-100).
        outsourcing_potential_score: Score based on outsourcing potential (0-100).
        weighted_total: Weighted total relevance score (0-100).
    """
    magnitude_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    data_availability_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    stakeholder_interest_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    outsourcing_potential_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    weighted_total: Decimal = Field(default=Decimal("0"), ge=0, le=100)

class CategoryScreening(BaseModel):
    """Screening result for a single Scope 3 category.

    Attributes:
        category: Scope 3 category.
        category_number: Category number (1-15).
        category_description: Human-readable category description.
        estimated_tco2e: Screening-level emission estimate (tCO2e).
        share_of_total_pct: Share of estimated total Scope 3 (%).
        relevance_tier: Relevance tier (HIGH, MEDIUM, LOW, NOT_APPLICABLE).
        relevance_score: Multi-dimensional relevance score.
        is_significant: Whether the category exceeds significance threshold.
        estimation_method: Method used for the estimate.
        eeio_intensity_used: EEIO intensity factor used (kgCO2e/EUR).
        proxy_data_used: Description of proxy data used.
        data_availability: Data availability level.
        recommended_methodology: Recommended calculation methodology for next steps.
        notes: Additional notes.
    """
    category: Scope3Category = Field(..., description="Scope 3 category")
    category_number: int = Field(default=0, ge=0, le=15, description="Category number")
    category_description: str = Field(default="", description="Category description")
    estimated_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Estimated tCO2e")
    share_of_total_pct: Decimal = Field(default=Decimal("0"), ge=0, description="Share of total %")
    relevance_tier: RelevanceTier = Field(default=RelevanceTier.LOW, description="Relevance tier")
    relevance_score: RelevanceScore = Field(
        default_factory=RelevanceScore, description="Relevance score"
    )
    is_significant: bool = Field(default=False, description="Exceeds significance threshold")
    estimation_method: str = Field(default="eeio_spend_based", description="Estimation method")
    eeio_intensity_used: Decimal = Field(default=Decimal("0"), description="EEIO intensity")
    proxy_data_used: str = Field(default="", description="Proxy data description")
    data_availability: DataAvailabilityLevel = Field(
        default=DataAvailabilityLevel.LIMITED, description="Data availability"
    )
    recommended_methodology: str = Field(default="", description="Recommended methodology")
    notes: str = Field(default="", description="Notes")

class ScreeningResult(BaseModel):
    """Complete Scope 3 screening result for an organisation.

    Attributes:
        result_id: Unique result identifier.
        org_id: Organisation identifier.
        org_name: Organisation name.
        reporting_year: Reporting year.
        total_estimated_scope3_tco2e: Total estimated Scope 3 (tCO2e).
        scope12_total_tco2e: Scope 1 + Scope 2 total for context.
        scope3_share_of_total_pct: Scope 3 as % of total carbon footprint.
        upstream_total_tco2e: Upstream (Cat 1-8) total estimated tCO2e.
        downstream_total_tco2e: Downstream (Cat 9-15) total estimated tCO2e.
        category_screenings: Per-category screening results.
        significant_categories: List of significant category numbers.
        significant_category_count: Number of significant categories.
        screening_methodology: Overall screening methodology description.
        significance_threshold_pct: Significance threshold used (%).
        naics_sector_used: NAICS sector used for screening.
        eeio_base_intensity: Base EEIO intensity used.
        warnings: Warnings generated during screening.
        status: Screening status.
        calculated_at: Timestamp.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    org_id: str = Field(default="", description="Organisation ID")
    org_name: str = Field(default="", description="Organisation name")
    reporting_year: int = Field(default=2025, description="Reporting year")
    total_estimated_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total estimated Scope 3 tCO2e"
    )
    scope12_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 1+2 total tCO2e"
    )
    scope3_share_of_total_pct: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 3 share of total %"
    )
    upstream_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Upstream total tCO2e"
    )
    downstream_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Downstream total tCO2e"
    )
    category_screenings: List[CategoryScreening] = Field(
        default_factory=list, description="Per-category screenings"
    )
    significant_categories: List[int] = Field(
        default_factory=list, description="Significant category numbers"
    )
    significant_category_count: int = Field(
        default=0, ge=0, description="Number of significant categories"
    )
    screening_methodology: str = Field(
        default="EEIO spend-based with sector profiles", description="Methodology"
    )
    significance_threshold_pct: Decimal = Field(
        default=Decimal("1.0"), description="Significance threshold %"
    )
    naics_sector_used: str = Field(default="", description="NAICS sector used")
    eeio_base_intensity: Decimal = Field(default=Decimal("0"), description="Base EEIO intensity")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    status: ScreeningStatus = Field(default=ScreeningStatus.COMPLETE, description="Status")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(default=Decimal("0"), description="Processing time ms")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

OrgProfile.model_rebuild()
RelevanceScore.model_rebuild()
CategoryScreening.model_rebuild()
ScreeningResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Scope3ScreeningEngine:
    """Rapid Scope 3 screening engine per GHG Protocol Chapter 6.

    Performs a screening-level assessment of all 15 Scope 3 categories
    using EEIO emission intensity factors, sector profiles, and
    organisational proxy data (spend, revenue, headcount).

    The engine follows GHG Protocol's recommended screening approach:
    1. Estimate magnitude of each category using available proxies.
    2. Score relevance using four dimensions.
    3. Identify significant categories for detailed calculation.

    Attributes:
        _significance_threshold_pct: Threshold for flagging significant categories.
        _relevance_weights: Weights for relevance scoring dimensions.
        _warnings: Warnings generated during screening.

    Example:
        >>> engine = Scope3ScreeningEngine()
        >>> profile = OrgProfile(
        ...     naics_2_digit="31",
        ...     annual_spend_eur=Decimal("50000000"),
        ...     annual_revenue_eur=Decimal("100000000"),
        ...     employee_count=500,
        ...     scope1_tco2e=Decimal("5000"),
        ...     scope2_tco2e=Decimal("3000"),
        ... )
        >>> result = engine.screen_all_categories(profile)
        >>> print(result.total_estimated_scope3_tco2e)
    """

    def __init__(
        self,
        significance_threshold_pct: Optional[Decimal] = None,
        relevance_weights: Optional[Dict[str, Decimal]] = None,
    ) -> None:
        """Initialise Scope3ScreeningEngine.

        Args:
            significance_threshold_pct: Threshold for significance (default 1%).
            relevance_weights: Custom weights for relevance scoring.
        """
        self._significance_threshold_pct = (
            significance_threshold_pct
            if significance_threshold_pct is not None
            else DEFAULT_SIGNIFICANCE_THRESHOLD_PCT
        )
        self._relevance_weights = relevance_weights or dict(RELEVANCE_WEIGHTS)
        self._warnings: List[str] = []
        logger.info(
            "Scope3ScreeningEngine v%s initialised (threshold=%.1f%%)",
            _MODULE_VERSION,
            self._significance_threshold_pct,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen_all_categories(
        self,
        org_profile: OrgProfile,
    ) -> ScreeningResult:
        """Screen all 15 Scope 3 categories for the organisation.

        Main entry point.  Estimates emissions for each category using
        EEIO intensities and sector profiles, then scores relevance and
        identifies significant categories.

        Args:
            org_profile: Organisation profile with sector and financial data.

        Returns:
            ScreeningResult with per-category screenings.

        Raises:
            ValueError: If org_profile has no sector or spend data.
        """
        t0 = time.perf_counter()
        self._warnings = []

        # Validate inputs
        self._validate_profile(org_profile)

        # Resolve sector profile
        sector_profile = self._resolve_sector_profile(org_profile)
        base_intensity = self._resolve_base_eeio_intensity(org_profile)

        # Screen each category
        category_screenings: List[CategoryScreening] = []
        for category in Scope3Category:
            screening = self._screen_category(
                category=category,
                org_profile=org_profile,
                sector_profile=sector_profile,
                base_intensity=base_intensity,
            )
            category_screenings.append(screening)

        # Calculate total estimated Scope 3
        total_scope3 = sum(
            (cs.estimated_tco2e for cs in category_screenings), Decimal("0")
        )

        # Calculate upstream/downstream split
        upstream_total = sum(
            (cs.estimated_tco2e for cs in category_screenings
             if cs.category_number <= 8),
            Decimal("0"),
        )
        downstream_total = sum(
            (cs.estimated_tco2e for cs in category_screenings
             if cs.category_number > 8),
            Decimal("0"),
        )

        # Calculate shares and significance
        scope12_total = org_profile.scope1_tco2e + org_profile.scope2_tco2e
        scope3_share = _safe_pct(total_scope3, total_scope3 + scope12_total)

        for cs in category_screenings:
            cs.share_of_total_pct = _round_val(_safe_pct(cs.estimated_tco2e, total_scope3), 2)
            cs.is_significant = cs.share_of_total_pct >= self._significance_threshold_pct

        # Score relevance for each category
        for cs in category_screenings:
            cs.relevance_score = self._score_relevance(cs, org_profile, total_scope3)
            cs.relevance_tier = self._determine_tier(cs)

        # Identify significant categories
        significant_cats = [
            cs.category_number for cs in category_screenings if cs.is_significant
        ]

        # Build result
        elapsed_ms = Decimal(str((time.perf_counter() - t0) * 1000))
        result = ScreeningResult(
            org_id=org_profile.org_id,
            org_name=org_profile.org_name,
            reporting_year=org_profile.reporting_year,
            total_estimated_scope3_tco2e=_round_val(total_scope3, 2),
            scope12_total_tco2e=_round_val(scope12_total, 2),
            scope3_share_of_total_pct=_round_val(scope3_share, 2),
            upstream_total_tco2e=_round_val(upstream_total, 2),
            downstream_total_tco2e=_round_val(downstream_total, 2),
            category_screenings=category_screenings,
            significant_categories=significant_cats,
            significant_category_count=len(significant_cats),
            significance_threshold_pct=self._significance_threshold_pct,
            naics_sector_used=org_profile.naics_2_digit,
            eeio_base_intensity=base_intensity,
            warnings=list(self._warnings),
            status=ScreeningStatus.COMPLETE,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = self._compute_provenance(result)

        logger.info(
            "Scope 3 screening complete: %.2f tCO2e total, %d significant categories",
            total_scope3,
            len(significant_cats),
        )
        return result

    def screen_single_category(
        self,
        category: Scope3Category,
        org_profile: OrgProfile,
    ) -> CategoryScreening:
        """Screen a single Scope 3 category.

        Convenience method to screen one category without running the
        full 15-category assessment.

        Args:
            category: The Scope 3 category to screen.
            org_profile: Organisation profile.

        Returns:
            CategoryScreening for the specified category.
        """
        self._warnings = []
        sector_profile = self._resolve_sector_profile(org_profile)
        base_intensity = self._resolve_base_eeio_intensity(org_profile)
        return self._screen_category(
            category=category,
            org_profile=org_profile,
            sector_profile=sector_profile,
            base_intensity=base_intensity,
        )

    def get_sector_profile(self, naics_2_digit: str) -> Optional[Dict[str, Any]]:
        """Retrieve the sector profile for a NAICS 2-digit code.

        Args:
            naics_2_digit: NAICS 2-digit sector code.

        Returns:
            Sector profile dict or None if not found.
        """
        return NAICS_SECTOR_PROFILES.get(naics_2_digit)

    def get_eeio_intensity(self, sector_key: str) -> Optional[Decimal]:
        """Retrieve the EEIO emission intensity for a sector.

        Args:
            sector_key: EEIO sector key.

        Returns:
            Intensity in kgCO2e/EUR or None if not found.
        """
        return EEIO_INTENSITIES.get(sector_key)

    def get_downstream_intensities(self, sector_key: str) -> Optional[Dict[str, Decimal]]:
        """Retrieve downstream revenue-based intensities for a sector.

        Args:
            sector_key: Downstream sector key.

        Returns:
            Dict with cat_10, cat_11, cat_12 intensities or None.
        """
        return DOWNSTREAM_INTENSITY_BY_SECTOR.get(sector_key)

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _validate_profile(self, profile: OrgProfile) -> None:
        """Validate the organisation profile has minimum required data.

        Args:
            profile: Organisation profile.

        Raises:
            ValueError: If critical data is missing.
        """
        if not profile.naics_2_digit and not profile.custom_eeio_overrides:
            raise ValueError(
                "OrgProfile must have either naics_2_digit or custom_eeio_overrides"
            )
        if profile.annual_spend_eur <= Decimal("0") and profile.annual_revenue_eur <= Decimal("0"):
            raise ValueError(
                "OrgProfile must have positive annual_spend_eur or annual_revenue_eur"
            )
        if profile.naics_2_digit and profile.naics_2_digit not in NAICS_SECTOR_PROFILES:
            self._warnings.append(
                f"NAICS code '{profile.naics_2_digit}' not in sector profiles; "
                "using generic services profile"
            )

    def _resolve_sector_profile(
        self,
        profile: OrgProfile,
    ) -> Dict[Scope3Category, Decimal]:
        """Resolve the sector profile to category-level emission fractions.

        Maps the NAICS code to a typical Scope 3 category distribution.
        Falls back to a generic services profile if sector not found.

        Args:
            profile: Organisation profile.

        Returns:
            Dict mapping each Scope3Category to its fraction of total.
        """
        naics_data = NAICS_SECTOR_PROFILES.get(profile.naics_2_digit)
        if naics_data and "typical_profile" in naics_data:
            return dict(naics_data["typical_profile"])

        # Fallback: generic services profile
        logger.warning(
            "No sector profile for NAICS '%s'; using generic services profile",
            profile.naics_2_digit,
        )
        return {
            Scope3Category.CAT_01: Decimal("0.25"),
            Scope3Category.CAT_02: Decimal("0.08"),
            Scope3Category.CAT_03: Decimal("0.05"),
            Scope3Category.CAT_04: Decimal("0.05"),
            Scope3Category.CAT_05: Decimal("0.04"),
            Scope3Category.CAT_06: Decimal("0.08"),
            Scope3Category.CAT_07: Decimal("0.08"),
            Scope3Category.CAT_08: Decimal("0.04"),
            Scope3Category.CAT_09: Decimal("0.03"),
            Scope3Category.CAT_10: Decimal("0.00"),
            Scope3Category.CAT_11: Decimal("0.10"),
            Scope3Category.CAT_12: Decimal("0.04"),
            Scope3Category.CAT_13: Decimal("0.05"),
            Scope3Category.CAT_14: Decimal("0.03"),
            Scope3Category.CAT_15: Decimal("0.08"),
        }

    def _resolve_base_eeio_intensity(self, profile: OrgProfile) -> Decimal:
        """Resolve the base EEIO intensity for the organisation's sector.

        Looks up the EEIO intensity by NAICS-to-EEIO mapping. Falls back
        to a weighted average across all sectors.

        Args:
            profile: Organisation profile.

        Returns:
            Base EEIO intensity in kgCO2e/EUR.
        """
        naics_data = NAICS_SECTOR_PROFILES.get(profile.naics_2_digit)
        if naics_data:
            eeio_key = naics_data.get("eeio_key", "")
            intensity = EEIO_INTENSITIES.get(eeio_key)
            if intensity is not None:
                return intensity

        # Fallback: median intensity across all sectors
        all_intensities = sorted(EEIO_INTENSITIES.values())
        mid = len(all_intensities) // 2
        median_intensity = all_intensities[mid]
        self._warnings.append(
            f"Using median EEIO intensity ({median_intensity} kgCO2e/EUR) as fallback"
        )
        return median_intensity

    def _screen_category(
        self,
        category: Scope3Category,
        org_profile: OrgProfile,
        sector_profile: Dict[Scope3Category, Decimal],
        base_intensity: Decimal,
    ) -> CategoryScreening:
        """Screen a single Scope 3 category.

        Estimates emissions using the appropriate proxy and EEIO intensity,
        adjusted by the sector profile fraction.

        Args:
            category: Scope 3 category to screen.
            org_profile: Organisation profile.
            sector_profile: Category-level emission fractions for sector.
            base_intensity: Base EEIO intensity in kgCO2e/EUR.

        Returns:
            CategoryScreening with estimated emissions.
        """
        cat_number = self._category_number(category)
        cat_desc = CATEGORY_DESCRIPTIONS.get(category, "")
        fraction = sector_profile.get(category, Decimal("0"))

        # Check custom overrides
        override_key = f"cat_{cat_number:02d}"
        custom_intensity = org_profile.custom_eeio_overrides.get(override_key)

        # Determine estimation method and calculate
        estimated_tco2e = Decimal("0")
        method = "eeio_spend_based"
        proxy_desc = ""
        intensity_used = base_intensity

        if custom_intensity is not None:
            intensity_used = custom_intensity

        # Category-specific estimation logic
        if cat_number in (1, 2, 3, 4, 5):
            # Upstream spend-based categories
            estimated_tco2e = self._estimate_spend_based(
                org_profile.annual_spend_eur, fraction, intensity_used
            )
            proxy_desc = f"Annual spend EUR {org_profile.annual_spend_eur} * fraction {fraction}"
            method = "eeio_spend_based"

        elif cat_number == 6:
            # Business travel: estimate from headcount
            estimated_tco2e = self._estimate_business_travel(
                org_profile.employee_count, fraction, org_profile.annual_spend_eur
            )
            proxy_desc = f"Employee count {org_profile.employee_count} + spend fraction"
            method = "headcount_proxy"

        elif cat_number == 7:
            # Employee commuting: estimate from headcount
            estimated_tco2e = self._estimate_employee_commuting(
                org_profile.employee_count, fraction, org_profile.annual_spend_eur
            )
            proxy_desc = f"Employee count {org_profile.employee_count} + commute assumptions"
            method = "headcount_proxy"

        elif cat_number == 8:
            # Upstream leased assets
            if not org_profile.has_leased_assets_upstream:
                return self._not_applicable_screening(
                    category, cat_number, cat_desc,
                    "Organisation does not lease upstream assets"
                )
            estimated_tco2e = self._estimate_spend_based(
                org_profile.annual_spend_eur, fraction, intensity_used
            )
            proxy_desc = "Lease spend fraction"
            method = "eeio_spend_based"

        elif cat_number in (10, 11, 12):
            # Downstream revenue-based
            if not org_profile.produces_physical_products and cat_number in (10, 12):
                return self._not_applicable_screening(
                    category, cat_number, cat_desc,
                    "Organisation does not produce physical products"
                )
            estimated_tco2e = self._estimate_downstream_revenue(
                org_profile, cat_number
            )
            proxy_desc = (
                f"Revenue EUR {org_profile.annual_revenue_eur} * "
                f"downstream intensity for {org_profile.downstream_sector_key}"
            )
            method = "revenue_based_downstream"

        elif cat_number == 9:
            # Downstream transport
            estimated_tco2e = self._estimate_spend_based(
                org_profile.annual_revenue_eur, fraction, base_intensity
            )
            proxy_desc = "Revenue-based with transport fraction"
            method = "revenue_based"

        elif cat_number == 13:
            # Downstream leased assets
            if not org_profile.has_leased_assets_downstream:
                return self._not_applicable_screening(
                    category, cat_number, cat_desc,
                    "Organisation does not have downstream leased assets"
                )
            estimated_tco2e = self._estimate_spend_based(
                org_profile.annual_revenue_eur, fraction, base_intensity
            )
            proxy_desc = "Revenue fraction for leased assets"
            method = "revenue_based"

        elif cat_number == 14:
            # Franchises
            if not org_profile.has_franchises:
                return self._not_applicable_screening(
                    category, cat_number, cat_desc,
                    "Organisation does not operate franchises"
                )
            estimated_tco2e = self._estimate_spend_based(
                org_profile.annual_revenue_eur, fraction, base_intensity
            )
            proxy_desc = "Revenue fraction for franchises"
            method = "revenue_based"

        elif cat_number == 15:
            # Investments
            if not org_profile.has_investments:
                return self._not_applicable_screening(
                    category, cat_number, cat_desc,
                    "Organisation does not have significant equity investments"
                )
            estimated_tco2e = self._estimate_spend_based(
                org_profile.annual_revenue_eur, fraction, base_intensity
            )
            proxy_desc = "Revenue fraction for investments"
            method = "revenue_based"

        # Convert kgCO2e to tCO2e
        estimated_tco2e_t = _round_val(estimated_tco2e / Decimal("1000"), 2)

        # Determine data availability
        data_avail = self._assess_data_availability(cat_number, org_profile)

        # Recommend methodology
        recommended = self._recommend_methodology(cat_number, estimated_tco2e_t)

        return CategoryScreening(
            category=category,
            category_number=cat_number,
            category_description=cat_desc,
            estimated_tco2e=estimated_tco2e_t,
            estimation_method=method,
            eeio_intensity_used=intensity_used,
            proxy_data_used=proxy_desc,
            data_availability=data_avail,
            recommended_methodology=recommended,
        )

    def _estimate_spend_based(
        self,
        total_spend: Decimal,
        fraction: Decimal,
        intensity: Decimal,
    ) -> Decimal:
        """Estimate emissions using spend-based EEIO approach.

        Formula: E = spend * fraction * intensity (kgCO2e/EUR)

        Args:
            total_spend: Total annual spend or revenue in EUR.
            fraction: Fraction allocated to this category.
            intensity: EEIO intensity in kgCO2e/EUR.

        Returns:
            Estimated emissions in kgCO2e.
        """
        return total_spend * fraction * intensity

    def _estimate_business_travel(
        self,
        employee_count: int,
        fraction: Decimal,
        annual_spend: Decimal,
    ) -> Decimal:
        """Estimate business travel emissions.

        Uses a hybrid of headcount-based average and spend fraction.
        Average business travel: 1.5 tCO2e/employee/year (industry average).
        Adjusted by sector fraction.

        Args:
            employee_count: Number of employees.
            fraction: Sector fraction for business travel.
            annual_spend: Annual spend for spend-based fallback.

        Returns:
            Estimated emissions in kgCO2e.
        """
        # Industry average: 1.5 tCO2e per employee per year = 1500 kgCO2e
        avg_per_employee_kg = Decimal("1500")
        headcount_estimate = _decimal(employee_count) * avg_per_employee_kg

        # Blend with spend-based if spend available
        if annual_spend > Decimal("0"):
            spend_estimate = annual_spend * fraction * Decimal("0.16")
            # Weight: 60% headcount, 40% spend
            return headcount_estimate * Decimal("0.6") + spend_estimate * Decimal("0.4")

        return headcount_estimate

    def _estimate_employee_commuting(
        self,
        employee_count: int,
        fraction: Decimal,
        annual_spend: Decimal,
    ) -> Decimal:
        """Estimate employee commuting emissions.

        Average commuting: 1.0 tCO2e/employee/year (industry average).
        Assumes average commute of 30 km/day, 230 working days, mixed mode.

        Args:
            employee_count: Number of employees.
            fraction: Sector fraction for commuting.
            annual_spend: Annual spend for spend-based fallback.

        Returns:
            Estimated emissions in kgCO2e.
        """
        # Industry average: 1.0 tCO2e per employee per year = 1000 kgCO2e
        avg_per_employee_kg = Decimal("1000")
        headcount_estimate = _decimal(employee_count) * avg_per_employee_kg

        if annual_spend > Decimal("0"):
            spend_estimate = annual_spend * fraction * Decimal("0.12")
            return headcount_estimate * Decimal("0.7") + spend_estimate * Decimal("0.3")

        return headcount_estimate

    def _estimate_downstream_revenue(
        self,
        org_profile: OrgProfile,
        cat_number: int,
    ) -> Decimal:
        """Estimate downstream emissions using revenue-based intensity.

        Looks up the downstream intensity factor for the organisation's
        sector and multiplies by annual revenue.

        Args:
            org_profile: Organisation profile.
            cat_number: Category number (10, 11, or 12).

        Returns:
            Estimated emissions in kgCO2e.
        """
        intensities = DOWNSTREAM_INTENSITY_BY_SECTOR.get(
            org_profile.downstream_sector_key
        )
        if not intensities:
            intensities = DOWNSTREAM_INTENSITY_BY_SECTOR.get(
                "services_general", {}
            )

        key = f"cat_{cat_number}"
        intensity = intensities.get(key, Decimal("0"))
        return org_profile.annual_revenue_eur * intensity

    def _not_applicable_screening(
        self,
        category: Scope3Category,
        cat_number: int,
        cat_desc: str,
        reason: str,
    ) -> CategoryScreening:
        """Create a NOT_APPLICABLE screening result.

        Args:
            category: Scope 3 category.
            cat_number: Category number.
            cat_desc: Category description.
            reason: Reason for not-applicable determination.

        Returns:
            CategoryScreening with NOT_APPLICABLE tier.
        """
        return CategoryScreening(
            category=category,
            category_number=cat_number,
            category_description=cat_desc,
            estimated_tco2e=Decimal("0"),
            relevance_tier=RelevanceTier.NOT_APPLICABLE,
            is_significant=False,
            estimation_method="not_applicable",
            notes=reason,
        )

    def _score_relevance(
        self,
        screening: CategoryScreening,
        org_profile: OrgProfile,
        total_scope3: Decimal,
    ) -> RelevanceScore:
        """Score the relevance of a Scope 3 category.

        Uses four dimensions weighted per GHG Protocol guidance:
        - Magnitude: based on share of total estimated emissions.
        - Data availability: based on available data sources.
        - Stakeholder interest: based on sector and category type.
        - Outsourcing potential: based on reduction leverage.

        Args:
            screening: Category screening result.
            org_profile: Organisation profile.
            total_scope3: Total estimated Scope 3 emissions.

        Returns:
            RelevanceScore with weighted total.
        """
        # Magnitude score (0-100): linear scale from 0% to 20% share
        mag_score = min(
            _safe_divide(screening.share_of_total_pct, Decimal("20")) * Decimal("100"),
            Decimal("100"),
        )

        # Data availability score (0-100)
        data_scores = {
            DataAvailabilityLevel.READILY_AVAILABLE: Decimal("90"),
            DataAvailabilityLevel.PARTIALLY_AVAILABLE: Decimal("60"),
            DataAvailabilityLevel.LIMITED: Decimal("30"),
            DataAvailabilityLevel.NONE: Decimal("10"),
        }
        data_score = data_scores.get(screening.data_availability, Decimal("30"))

        # Stakeholder interest score (0-100): higher for upstream categories
        stakeholder_scores = {
            1: Decimal("85"), 2: Decimal("60"), 3: Decimal("70"),
            4: Decimal("75"), 5: Decimal("65"), 6: Decimal("70"),
            7: Decimal("55"), 8: Decimal("40"), 9: Decimal("65"),
            10: Decimal("50"), 11: Decimal("80"), 12: Decimal("60"),
            13: Decimal("45"), 14: Decimal("55"), 15: Decimal("70"),
        }
        stake_score = stakeholder_scores.get(
            screening.category_number, Decimal("50")
        )

        # Outsourcing potential score (0-100): categories where the org can
        # influence reductions in the value chain
        outsource_scores = {
            1: Decimal("80"), 2: Decimal("60"), 3: Decimal("50"),
            4: Decimal("75"), 5: Decimal("70"), 6: Decimal("65"),
            7: Decimal("40"), 8: Decimal("35"), 9: Decimal("60"),
            10: Decimal("45"), 11: Decimal("70"), 12: Decimal("55"),
            13: Decimal("40"), 14: Decimal("50"), 15: Decimal("30"),
        }
        outsource_score = outsource_scores.get(
            screening.category_number, Decimal("50")
        )

        # Weighted total
        w = self._relevance_weights
        weighted_total = (
            w["magnitude"] * mag_score
            + w["data_availability"] * data_score
            + w["stakeholder_interest"] * stake_score
            + w["outsourcing_potential"] * outsource_score
        )

        return RelevanceScore(
            magnitude_score=_round_val(mag_score, 2),
            data_availability_score=_round_val(data_score, 2),
            stakeholder_interest_score=_round_val(stake_score, 2),
            outsourcing_potential_score=_round_val(outsource_score, 2),
            weighted_total=_round_val(min(weighted_total, Decimal("100")), 2),
        )

    def _determine_tier(self, screening: CategoryScreening) -> RelevanceTier:
        """Determine the relevance tier based on score and significance.

        Args:
            screening: Category screening with relevance score.

        Returns:
            RelevanceTier.
        """
        if screening.relevance_tier == RelevanceTier.NOT_APPLICABLE:
            return RelevanceTier.NOT_APPLICABLE

        score = screening.relevance_score.weighted_total
        if screening.is_significant or score >= Decimal("60"):
            return RelevanceTier.HIGH
        elif score >= Decimal("35"):
            return RelevanceTier.MEDIUM
        else:
            return RelevanceTier.LOW

    def _assess_data_availability(
        self,
        cat_number: int,
        org_profile: OrgProfile,
    ) -> DataAvailabilityLevel:
        """Assess data availability for a category.

        Args:
            cat_number: Category number.
            org_profile: Organisation profile.

        Returns:
            DataAvailabilityLevel.
        """
        # Categories where spend data is typically available
        if cat_number in (1, 2, 4) and org_profile.annual_spend_eur > Decimal("0"):
            return DataAvailabilityLevel.PARTIALLY_AVAILABLE

        # Categories where utility/fuel data is typically available
        if cat_number == 3 and org_profile.scope2_tco2e > Decimal("0"):
            return DataAvailabilityLevel.PARTIALLY_AVAILABLE

        # Categories where headcount is available
        if cat_number in (6, 7) and org_profile.employee_count > 0:
            return DataAvailabilityLevel.PARTIALLY_AVAILABLE

        # Waste data
        if cat_number == 5:
            return DataAvailabilityLevel.LIMITED

        # Downstream categories
        if cat_number in (9, 10, 11, 12):
            if org_profile.produces_physical_products:
                return DataAvailabilityLevel.LIMITED
            return DataAvailabilityLevel.NONE

        # Leased assets, franchises, investments
        if cat_number in (8, 13, 14, 15):
            return DataAvailabilityLevel.LIMITED

        return DataAvailabilityLevel.LIMITED

    def _recommend_methodology(
        self,
        cat_number: int,
        estimated_tco2e: Decimal,
    ) -> str:
        """Recommend a calculation methodology for the next steps.

        Args:
            cat_number: Category number.
            estimated_tco2e: Estimated emissions from screening.

        Returns:
            Recommended methodology string.
        """
        recommendations = {
            1: "Supplier-specific or hybrid method (primary data from key suppliers + EEIO for rest)",
            2: "Spend-based with sector-specific emission factors",
            3: "Average-data method using grid emission factors and T&D loss rates",
            4: "Distance-based method (tkm) or spend-based with logistics EFs",
            5: "Waste-type-specific method using disposal method emission factors",
            6: "Distance-based method using travel booking data and mode-specific EFs",
            7: "Average-data method using commuting surveys and mode-specific EFs",
            8: "Asset-specific method using building energy data",
            9: "Distance-based or spend-based with logistics EFs",
            10: "Average-data method using industry processing energy data",
            11: "Direct use-phase method with product lifetime and energy consumption data",
            12: "Waste-type-specific method based on product material composition",
            13: "Asset-specific method using tenant energy data",
            14: "Franchise-specific method using energy and operational data",
            15: "Investment-specific method (PCAF) using financed emissions data",
        }
        return recommendations.get(cat_number, "Spend-based or average-data method")

    @staticmethod
    def _category_number(category: Scope3Category) -> int:
        """Extract the category number from a Scope3Category enum.

        Args:
            category: Scope 3 category enum value.

        Returns:
            Integer category number (1-15).
        """
        # Extract number from enum value string, e.g. "cat_01_purchased_goods" -> 1
        parts = category.value.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 0

    def _compute_provenance(self, result: ScreeningResult) -> str:
        """Compute SHA-256 provenance hash for the screening result.

        Args:
            result: Complete screening result.

        Returns:
            SHA-256 hex digest.
        """
        return _compute_hash(result)
