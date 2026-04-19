# -*- coding: utf-8 -*-
"""
SectorClassificationEngine - PACK-028 Sector Pathway Engine 1
================================================================

Automatic sector classification using NACE Rev.2, GICS, ISIC Rev.4
codes with SBTi SDA sector mapping, revenue-weighted prioritisation,
multi-sector company handling, and SDA eligibility validation.

The engine classifies companies into one or more of 15+ sector
categories, maps them to SBTi SDA sectors (12 sectors) and IEA NZE
sectors (15+ sectors), and determines pathway eligibility.

Classification Methodology:
    1. Code Lookup: NACE Rev.2 / GICS / ISIC Rev.4 code -> sector mapping
    2. Revenue Weighting: For multi-sector companies, weight by revenue share
    3. SDA Eligibility: Validate coverage (95% Scope 1+2), sector match
    4. Primary Sector: Highest revenue-weighted sector
    5. Multi-sector: Handle conglomerates with 2+ sectors

Sector Taxonomy:
    - 12 SBTi SDA Sectors: power, steel, cement, aluminum, pulp_paper,
      chemicals, aviation, shipping, road_transport, rail,
      buildings_residential, buildings_commercial
    - 3 Extended IEA Sectors: agriculture, food_beverage, oil_gas
    - 1 Fallback: cross_sector (ACA approach)

Regulatory References:
    - SBTi Sectoral Decarbonization Approach (SDA) Methodology
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - NACE Rev.2 Statistical Classification (Eurostat)
    - GICS (Global Industry Classification Standard) - MSCI/S&P
    - ISIC Rev.4 (International Standard Industrial Classification) - UN
    - IEA Net Zero by 2050 Roadmap (2023) - Sector definitions

Zero-Hallucination:
    - All sector classifications use deterministic code lookups
    - Revenue weighting uses arithmetic Decimal operations
    - SDA eligibility is rule-based (no ML/LLM)
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-028 Sector Pathway
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
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

def _round3(value: float) -> float:
    """Round to 3 decimal places."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SectorCode(str, Enum):
    """Unified sector classification covering SBTi SDA + IEA NZE sectors.

    12 SBTi SDA sectors plus 3 extended IEA sectors and 1 fallback.
    """
    POWER_GENERATION = "power_generation"
    STEEL = "steel"
    CEMENT = "cement"
    ALUMINUM = "aluminum"
    PULP_PAPER = "pulp_paper"
    CHEMICALS = "chemicals"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    ROAD_TRANSPORT = "road_transport"
    RAIL = "rail"
    BUILDINGS_RESIDENTIAL = "buildings_residential"
    BUILDINGS_COMMERCIAL = "buildings_commercial"
    AGRICULTURE = "agriculture"
    FOOD_BEVERAGE = "food_beverage"
    OIL_GAS = "oil_gas"
    CROSS_SECTOR = "cross_sector"

class ClassificationSystem(str, Enum):
    """Industry classification system identifiers."""
    NACE_REV2 = "nace_rev2"
    GICS = "gics"
    ISIC_REV4 = "isic_rev4"
    MANUAL = "manual"

class SDAEligibility(str, Enum):
    """SBTi SDA eligibility status."""
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    PARTIAL = "partial"
    REQUIRES_REVIEW = "requires_review"

class PathwayApproach(str, Enum):
    """Decarbonization pathway approach based on sector classification."""
    SDA = "sda"
    ACA = "aca"
    FLAG = "flag"
    HYBRID = "hybrid"

class SectorPriority(str, Enum):
    """Priority level for sector classification."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    MINOR = "minor"

class DataQuality(str, Enum):
    """Data quality tier for classification inputs."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Constants -- NACE Rev.2 -> Sector Mapping
# ---------------------------------------------------------------------------

# NACE Rev.2 section and division codes mapped to SectorCode.
# Source: Eurostat NACE Rev.2 structure, SBTi SDA sector guidance.
NACE_TO_SECTOR: Dict[str, SectorCode] = {
    # Power generation (NACE D35)
    "D35.1": SectorCode.POWER_GENERATION,
    "D35.11": SectorCode.POWER_GENERATION,
    "D35.12": SectorCode.POWER_GENERATION,
    "D35.13": SectorCode.POWER_GENERATION,
    "D35.14": SectorCode.POWER_GENERATION,
    "D35.2": SectorCode.POWER_GENERATION,
    "D35.3": SectorCode.POWER_GENERATION,
    # Steel (NACE C24.1)
    "C24.1": SectorCode.STEEL,
    "C24.10": SectorCode.STEEL,
    "C24.2": SectorCode.STEEL,
    "C24.3": SectorCode.STEEL,
    "C24.31": SectorCode.STEEL,
    "C24.32": SectorCode.STEEL,
    "C24.33": SectorCode.STEEL,
    "C24.34": SectorCode.STEEL,
    # Cement (NACE C23.5)
    "C23.5": SectorCode.CEMENT,
    "C23.51": SectorCode.CEMENT,
    "C23.52": SectorCode.CEMENT,
    "C23.6": SectorCode.CEMENT,
    "C23.61": SectorCode.CEMENT,
    "C23.63": SectorCode.CEMENT,
    # Aluminum (NACE C24.4)
    "C24.4": SectorCode.ALUMINUM,
    "C24.42": SectorCode.ALUMINUM,
    "C24.43": SectorCode.ALUMINUM,
    "C24.44": SectorCode.ALUMINUM,
    "C24.45": SectorCode.ALUMINUM,
    # Pulp & Paper (NACE C17)
    "C17": SectorCode.PULP_PAPER,
    "C17.1": SectorCode.PULP_PAPER,
    "C17.11": SectorCode.PULP_PAPER,
    "C17.12": SectorCode.PULP_PAPER,
    "C17.2": SectorCode.PULP_PAPER,
    "C17.21": SectorCode.PULP_PAPER,
    "C17.22": SectorCode.PULP_PAPER,
    "C17.23": SectorCode.PULP_PAPER,
    "C17.24": SectorCode.PULP_PAPER,
    "C17.29": SectorCode.PULP_PAPER,
    # Chemicals (NACE C20)
    "C20": SectorCode.CHEMICALS,
    "C20.1": SectorCode.CHEMICALS,
    "C20.11": SectorCode.CHEMICALS,
    "C20.12": SectorCode.CHEMICALS,
    "C20.13": SectorCode.CHEMICALS,
    "C20.14": SectorCode.CHEMICALS,
    "C20.15": SectorCode.CHEMICALS,
    "C20.16": SectorCode.CHEMICALS,
    "C20.17": SectorCode.CHEMICALS,
    "C20.2": SectorCode.CHEMICALS,
    "C20.3": SectorCode.CHEMICALS,
    "C20.4": SectorCode.CHEMICALS,
    "C20.5": SectorCode.CHEMICALS,
    "C20.6": SectorCode.CHEMICALS,
    # Aviation (NACE H51)
    "H51": SectorCode.AVIATION,
    "H51.1": SectorCode.AVIATION,
    "H51.10": SectorCode.AVIATION,
    "H51.2": SectorCode.AVIATION,
    "H51.21": SectorCode.AVIATION,
    "H51.22": SectorCode.AVIATION,
    # Shipping (NACE H50)
    "H50": SectorCode.SHIPPING,
    "H50.1": SectorCode.SHIPPING,
    "H50.10": SectorCode.SHIPPING,
    "H50.2": SectorCode.SHIPPING,
    "H50.20": SectorCode.SHIPPING,
    "H50.3": SectorCode.SHIPPING,
    "H50.4": SectorCode.SHIPPING,
    # Road transport (NACE H49)
    "H49": SectorCode.ROAD_TRANSPORT,
    "H49.1": SectorCode.ROAD_TRANSPORT,
    "H49.10": SectorCode.ROAD_TRANSPORT,
    "H49.2": SectorCode.ROAD_TRANSPORT,
    "H49.3": SectorCode.ROAD_TRANSPORT,
    "H49.31": SectorCode.ROAD_TRANSPORT,
    "H49.32": SectorCode.ROAD_TRANSPORT,
    "H49.39": SectorCode.ROAD_TRANSPORT,
    "H49.4": SectorCode.ROAD_TRANSPORT,
    "H49.41": SectorCode.ROAD_TRANSPORT,
    "H49.42": SectorCode.ROAD_TRANSPORT,
    "H49.5": SectorCode.ROAD_TRANSPORT,
    # Rail (NACE H49.1, H49.2 -- specifically rail)
    # Note: H49.1 is rail passenger, H49.2 is rail freight
    # These overlap with ROAD_TRANSPORT above; the engine resolves
    # via revenue weighting and activity description.
    # Agriculture (NACE A01)
    "A01": SectorCode.AGRICULTURE,
    "A01.1": SectorCode.AGRICULTURE,
    "A01.11": SectorCode.AGRICULTURE,
    "A01.12": SectorCode.AGRICULTURE,
    "A01.13": SectorCode.AGRICULTURE,
    "A01.14": SectorCode.AGRICULTURE,
    "A01.15": SectorCode.AGRICULTURE,
    "A01.16": SectorCode.AGRICULTURE,
    "A01.19": SectorCode.AGRICULTURE,
    "A01.2": SectorCode.AGRICULTURE,
    "A01.21": SectorCode.AGRICULTURE,
    "A01.22": SectorCode.AGRICULTURE,
    "A01.23": SectorCode.AGRICULTURE,
    "A01.24": SectorCode.AGRICULTURE,
    "A01.25": SectorCode.AGRICULTURE,
    "A01.26": SectorCode.AGRICULTURE,
    "A01.27": SectorCode.AGRICULTURE,
    "A01.28": SectorCode.AGRICULTURE,
    "A01.29": SectorCode.AGRICULTURE,
    "A01.3": SectorCode.AGRICULTURE,
    "A01.4": SectorCode.AGRICULTURE,
    "A01.41": SectorCode.AGRICULTURE,
    "A01.42": SectorCode.AGRICULTURE,
    "A01.43": SectorCode.AGRICULTURE,
    "A01.44": SectorCode.AGRICULTURE,
    "A01.45": SectorCode.AGRICULTURE,
    "A01.46": SectorCode.AGRICULTURE,
    "A01.47": SectorCode.AGRICULTURE,
    "A01.49": SectorCode.AGRICULTURE,
    "A01.5": SectorCode.AGRICULTURE,
    "A01.6": SectorCode.AGRICULTURE,
    # Food & Beverage (NACE C10, C11)
    "C10": SectorCode.FOOD_BEVERAGE,
    "C10.1": SectorCode.FOOD_BEVERAGE,
    "C10.11": SectorCode.FOOD_BEVERAGE,
    "C10.12": SectorCode.FOOD_BEVERAGE,
    "C10.13": SectorCode.FOOD_BEVERAGE,
    "C10.2": SectorCode.FOOD_BEVERAGE,
    "C10.3": SectorCode.FOOD_BEVERAGE,
    "C10.31": SectorCode.FOOD_BEVERAGE,
    "C10.32": SectorCode.FOOD_BEVERAGE,
    "C10.39": SectorCode.FOOD_BEVERAGE,
    "C10.4": SectorCode.FOOD_BEVERAGE,
    "C10.41": SectorCode.FOOD_BEVERAGE,
    "C10.42": SectorCode.FOOD_BEVERAGE,
    "C10.5": SectorCode.FOOD_BEVERAGE,
    "C10.51": SectorCode.FOOD_BEVERAGE,
    "C10.52": SectorCode.FOOD_BEVERAGE,
    "C10.6": SectorCode.FOOD_BEVERAGE,
    "C10.7": SectorCode.FOOD_BEVERAGE,
    "C10.71": SectorCode.FOOD_BEVERAGE,
    "C10.72": SectorCode.FOOD_BEVERAGE,
    "C10.73": SectorCode.FOOD_BEVERAGE,
    "C10.8": SectorCode.FOOD_BEVERAGE,
    "C10.81": SectorCode.FOOD_BEVERAGE,
    "C10.82": SectorCode.FOOD_BEVERAGE,
    "C10.83": SectorCode.FOOD_BEVERAGE,
    "C10.84": SectorCode.FOOD_BEVERAGE,
    "C10.85": SectorCode.FOOD_BEVERAGE,
    "C10.86": SectorCode.FOOD_BEVERAGE,
    "C10.89": SectorCode.FOOD_BEVERAGE,
    "C10.9": SectorCode.FOOD_BEVERAGE,
    "C11": SectorCode.FOOD_BEVERAGE,
    "C11.01": SectorCode.FOOD_BEVERAGE,
    "C11.02": SectorCode.FOOD_BEVERAGE,
    "C11.03": SectorCode.FOOD_BEVERAGE,
    "C11.04": SectorCode.FOOD_BEVERAGE,
    "C11.05": SectorCode.FOOD_BEVERAGE,
    "C11.06": SectorCode.FOOD_BEVERAGE,
    "C11.07": SectorCode.FOOD_BEVERAGE,
    # Oil & Gas (NACE B06, B09, C19)
    "B06": SectorCode.OIL_GAS,
    "B06.1": SectorCode.OIL_GAS,
    "B06.10": SectorCode.OIL_GAS,
    "B06.2": SectorCode.OIL_GAS,
    "B06.20": SectorCode.OIL_GAS,
    "B09": SectorCode.OIL_GAS,
    "B09.1": SectorCode.OIL_GAS,
    "B09.10": SectorCode.OIL_GAS,
    "C19": SectorCode.OIL_GAS,
    "C19.1": SectorCode.OIL_GAS,
    "C19.10": SectorCode.OIL_GAS,
    "C19.2": SectorCode.OIL_GAS,
    "C19.20": SectorCode.OIL_GAS,
    # Buildings - Residential (NACE F41, L68 residential)
    "F41.1": SectorCode.BUILDINGS_RESIDENTIAL,
    "F41.10": SectorCode.BUILDINGS_RESIDENTIAL,
    "L68.1": SectorCode.BUILDINGS_RESIDENTIAL,
    "L68.10": SectorCode.BUILDINGS_RESIDENTIAL,
    # Buildings - Commercial (NACE F41.2, L68.2, L68.3)
    "F41.2": SectorCode.BUILDINGS_COMMERCIAL,
    "F41.20": SectorCode.BUILDINGS_COMMERCIAL,
    "L68.2": SectorCode.BUILDINGS_COMMERCIAL,
    "L68.20": SectorCode.BUILDINGS_COMMERCIAL,
    "L68.3": SectorCode.BUILDINGS_COMMERCIAL,
    "L68.31": SectorCode.BUILDINGS_COMMERCIAL,
    "L68.32": SectorCode.BUILDINGS_COMMERCIAL,
}

# ---------------------------------------------------------------------------
# Constants -- GICS -> Sector Mapping
# ---------------------------------------------------------------------------

# GICS sub-industry codes (8-digit) mapped to SectorCode.
# Source: MSCI/S&P Global Industry Classification Standard.
GICS_TO_SECTOR: Dict[str, SectorCode] = {
    # Energy
    "10101010": SectorCode.OIL_GAS,      # Oil & Gas Drilling
    "10101020": SectorCode.OIL_GAS,      # Oil & Gas E&P
    "10102010": SectorCode.OIL_GAS,      # Integrated Oil & Gas
    "10102020": SectorCode.OIL_GAS,      # Oil & Gas Refining
    "10102030": SectorCode.OIL_GAS,      # Oil & Gas Storage & Transport
    "10102040": SectorCode.OIL_GAS,      # Oil & Gas Equipment & Services
    "10102050": SectorCode.OIL_GAS,      # Coal & Consumable Fuels
    # Utilities
    "55101010": SectorCode.POWER_GENERATION,  # Electric Utilities
    "55102010": SectorCode.POWER_GENERATION,  # Gas Utilities
    "55103010": SectorCode.POWER_GENERATION,  # Multi-Utilities
    "55104010": SectorCode.POWER_GENERATION,  # Water Utilities
    "55105010": SectorCode.POWER_GENERATION,  # IPP & Energy Traders
    "55105020": SectorCode.POWER_GENERATION,  # Renewable Electricity
    # Materials - Metals & Mining
    "15104010": SectorCode.STEEL,             # Steel
    "15104020": SectorCode.ALUMINUM,          # Aluminum
    "15104025": SectorCode.ALUMINUM,          # Diversified Metals
    "15104030": SectorCode.STEEL,             # Copper
    "15104040": SectorCode.STEEL,             # Precious Metals
    "15104045": SectorCode.STEEL,             # Silver
    "15104050": SectorCode.STEEL,             # Diversified Mining
    # Materials - Chemicals
    "15101010": SectorCode.CHEMICALS,         # Commodity Chemicals
    "15101020": SectorCode.CHEMICALS,         # Diversified Chemicals
    "15101030": SectorCode.CHEMICALS,         # Fertilizers & Ag Chem
    "15101040": SectorCode.CHEMICALS,         # Industrial Gases
    "15101050": SectorCode.CHEMICALS,         # Specialty Chemicals
    # Materials - Construction
    "15102010": SectorCode.CEMENT,            # Construction Materials
    # Materials - Paper & Forest
    "15105010": SectorCode.PULP_PAPER,        # Forest Products
    "15105020": SectorCode.PULP_PAPER,        # Paper Products
    # Industrials - Transportation
    "20301010": SectorCode.AVIATION,          # Air Freight & Logistics
    "20302010": SectorCode.AVIATION,          # Airlines
    "20303010": SectorCode.SHIPPING,          # Marine
    "20304010": SectorCode.ROAD_TRANSPORT,    # Road & Rail
    "20304020": SectorCode.RAIL,              # Railroads
    "20305010": SectorCode.ROAD_TRANSPORT,    # Trucking
    # Real Estate
    "60101010": SectorCode.BUILDINGS_COMMERCIAL,  # Diversified REITs
    "60101020": SectorCode.BUILDINGS_COMMERCIAL,  # Industrial REITs
    "60101030": SectorCode.BUILDINGS_COMMERCIAL,  # Hotel & Resort REITs
    "60101040": SectorCode.BUILDINGS_COMMERCIAL,  # Office REITs
    "60101050": SectorCode.BUILDINGS_COMMERCIAL,  # Health Care REITs
    "60101060": SectorCode.BUILDINGS_RESIDENTIAL,  # Residential REITs
    "60101070": SectorCode.BUILDINGS_COMMERCIAL,  # Retail REITs
    "60101080": SectorCode.BUILDINGS_COMMERCIAL,  # Specialized REITs
    "60102010": SectorCode.BUILDINGS_COMMERCIAL,  # RE Mgmt & Dev
    "60102020": SectorCode.BUILDINGS_COMMERCIAL,  # RE Operating Companies
    "60102030": SectorCode.BUILDINGS_COMMERCIAL,  # RE Services
    # Consumer Staples - Food
    "30201010": SectorCode.FOOD_BEVERAGE,     # Agricultural Products
    "30201020": SectorCode.FOOD_BEVERAGE,     # Packaged Foods & Meats
    "30201030": SectorCode.FOOD_BEVERAGE,     # Brewers
    "30201040": SectorCode.FOOD_BEVERAGE,     # Distillers & Vintners
    "30201050": SectorCode.FOOD_BEVERAGE,     # Soft Drinks
    "30202010": SectorCode.AGRICULTURE,       # Agricultural Products & Svc
    "30202030": SectorCode.FOOD_BEVERAGE,     # Tobacco
}

# ---------------------------------------------------------------------------
# Constants -- ISIC Rev.4 -> Sector Mapping
# ---------------------------------------------------------------------------

# ISIC Rev.4 division/class codes mapped to SectorCode.
# Source: UN ISIC Rev.4.
ISIC_TO_SECTOR: Dict[str, SectorCode] = {
    # Electricity, gas, steam
    "351": SectorCode.POWER_GENERATION,
    "3510": SectorCode.POWER_GENERATION,
    "352": SectorCode.POWER_GENERATION,
    "353": SectorCode.POWER_GENERATION,
    # Steel
    "241": SectorCode.STEEL,
    "2410": SectorCode.STEEL,
    "242": SectorCode.STEEL,
    "243": SectorCode.STEEL,
    # Cement
    "239": SectorCode.CEMENT,
    "2394": SectorCode.CEMENT,
    "2395": SectorCode.CEMENT,
    "2396": SectorCode.CEMENT,
    # Aluminum
    "2420": SectorCode.ALUMINUM,
    # Pulp & Paper
    "170": SectorCode.PULP_PAPER,
    "1701": SectorCode.PULP_PAPER,
    "1702": SectorCode.PULP_PAPER,
    "1709": SectorCode.PULP_PAPER,
    # Chemicals
    "201": SectorCode.CHEMICALS,
    "2011": SectorCode.CHEMICALS,
    "2012": SectorCode.CHEMICALS,
    "2013": SectorCode.CHEMICALS,
    "202": SectorCode.CHEMICALS,
    "2021": SectorCode.CHEMICALS,
    "2022": SectorCode.CHEMICALS,
    "2023": SectorCode.CHEMICALS,
    "2029": SectorCode.CHEMICALS,
    "2030": SectorCode.CHEMICALS,
    # Aviation
    "511": SectorCode.AVIATION,
    "5110": SectorCode.AVIATION,
    # Shipping
    "501": SectorCode.SHIPPING,
    "5011": SectorCode.SHIPPING,
    "5012": SectorCode.SHIPPING,
    "502": SectorCode.SHIPPING,
    # Road transport
    "491": SectorCode.ROAD_TRANSPORT,
    "4911": SectorCode.ROAD_TRANSPORT,
    "4912": SectorCode.ROAD_TRANSPORT,
    "4921": SectorCode.ROAD_TRANSPORT,
    "4922": SectorCode.ROAD_TRANSPORT,
    "4923": SectorCode.ROAD_TRANSPORT,
    # Rail
    "4910": SectorCode.RAIL,
    # Agriculture
    "011": SectorCode.AGRICULTURE,
    "012": SectorCode.AGRICULTURE,
    "013": SectorCode.AGRICULTURE,
    "014": SectorCode.AGRICULTURE,
    "015": SectorCode.AGRICULTURE,
    "016": SectorCode.AGRICULTURE,
    # Food & Beverage
    "101": SectorCode.FOOD_BEVERAGE,
    "102": SectorCode.FOOD_BEVERAGE,
    "103": SectorCode.FOOD_BEVERAGE,
    "104": SectorCode.FOOD_BEVERAGE,
    "105": SectorCode.FOOD_BEVERAGE,
    "106": SectorCode.FOOD_BEVERAGE,
    "107": SectorCode.FOOD_BEVERAGE,
    "108": SectorCode.FOOD_BEVERAGE,
    "110": SectorCode.FOOD_BEVERAGE,
    # Oil & Gas
    "061": SectorCode.OIL_GAS,
    "062": SectorCode.OIL_GAS,
    "091": SectorCode.OIL_GAS,
    "192": SectorCode.OIL_GAS,
    # Buildings
    "410": SectorCode.BUILDINGS_COMMERCIAL,
    "411": SectorCode.BUILDINGS_RESIDENTIAL,
    "412": SectorCode.BUILDINGS_COMMERCIAL,
    "681": SectorCode.BUILDINGS_RESIDENTIAL,
    "682": SectorCode.BUILDINGS_COMMERCIAL,
}

# ---------------------------------------------------------------------------
# Constants -- SDA Sector Metadata
# ---------------------------------------------------------------------------

# SBTi SDA sector definitions including intensity metric, coverage,
# and pathway approach requirements.
SDA_SECTOR_METADATA: Dict[str, Dict[str, Any]] = {
    SectorCode.POWER_GENERATION: {
        "name": "Power Generation",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "gCO2/kWh",
        "activity_metric": "kWh generated",
        "iea_chapter": "Chapter 3: Electricity",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Power",
        "key_technologies": [
            "Solar PV", "Wind (onshore/offshore)", "Hydropower",
            "Nuclear (baseload/SMR)", "Battery storage",
            "Grid-scale hydrogen", "CCS (fossil)",
        ],
    },
    SectorCode.STEEL: {
        "name": "Steel",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "tCO2e/tonne crude steel",
        "activity_metric": "tonnes crude steel produced",
        "iea_chapter": "Chapter 5: Industry (Steel)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Steel",
        "key_technologies": [
            "Electric Arc Furnace (EAF)", "DRI with green H2",
            "CCS for BF-BOF", "Scrap recycling", "Waste heat recovery",
        ],
    },
    SectorCode.CEMENT: {
        "name": "Cement",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "tCO2e/tonne cement",
        "activity_metric": "tonnes cement produced",
        "iea_chapter": "Chapter 5: Industry (Cement)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Cement",
        "key_technologies": [
            "Clinker substitution", "Alternative fuels",
            "CCUS", "High-efficiency kilns", "Low-carbon cement",
        ],
    },
    SectorCode.ALUMINUM: {
        "name": "Aluminum",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "tCO2e/tonne aluminum",
        "activity_metric": "tonnes aluminum produced",
        "iea_chapter": "Chapter 5: Industry (Aluminum)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Aluminum",
        "key_technologies": [
            "Inert anode", "Renewable electricity for smelting",
            "Secondary aluminum expansion", "Low-carbon alumina",
        ],
    },
    SectorCode.PULP_PAPER: {
        "name": "Pulp & Paper",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "tCO2e/tonne pulp",
        "activity_metric": "tonnes pulp produced",
        "iea_chapter": "Chapter 5: Industry (Pulp)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Pulp",
        "key_technologies": [
            "Biomass CHP", "Energy efficiency",
            "Black liquor gasification", "Electrification",
        ],
    },
    SectorCode.CHEMICALS: {
        "name": "Chemicals",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "tCO2e/tonne product",
        "activity_metric": "tonnes chemical product",
        "iea_chapter": "Chapter 5: Industry (Chemicals)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Chemicals",
        "key_technologies": [
            "Electrification", "Green hydrogen",
            "Catalytic efficiency", "CCS", "Circular chemistry",
        ],
    },
    SectorCode.AVIATION: {
        "name": "Aviation",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "gCO2/pkm",
        "activity_metric": "passenger-kilometers",
        "iea_chapter": "Chapter 4: Transport (Aviation)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Aviation",
        "key_technologies": [
            "Sustainable Aviation Fuel (SAF)", "Fleet renewal",
            "Operational efficiency", "Hydrogen aircraft",
            "Electric aircraft (<500km)",
        ],
    },
    SectorCode.SHIPPING: {
        "name": "Shipping",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "gCO2/tkm",
        "activity_metric": "tonne-kilometers",
        "iea_chapter": "Chapter 4: Transport (Shipping)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Shipping",
        "key_technologies": [
            "LNG/methanol/ammonia fuels", "Wind-assisted propulsion",
            "Hull efficiency", "Slow steaming", "Shore power",
        ],
    },
    SectorCode.ROAD_TRANSPORT: {
        "name": "Road Transport",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "gCO2/vkm",
        "activity_metric": "vehicle-kilometers",
        "iea_chapter": "Chapter 4: Transport (Road)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Transport",
        "key_technologies": [
            "BEV fleet transition", "FCEV (heavy-duty)",
            "Route optimisation", "Autonomous driving efficiency",
        ],
    },
    SectorCode.RAIL: {
        "name": "Rail",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "gCO2/pkm",
        "activity_metric": "passenger-kilometers",
        "iea_chapter": "Chapter 4: Transport (Rail)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Rail",
        "key_technologies": [
            "Electrification of rail lines",
            "Hydrogen-powered trains",
            "Regenerative braking",
        ],
    },
    SectorCode.BUILDINGS_RESIDENTIAL: {
        "name": "Buildings (Residential)",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "kgCO2/m2/year",
        "activity_metric": "square meters floor area",
        "iea_chapter": "Chapter 2: Buildings (Residential)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Buildings",
        "key_technologies": [
            "Heat pumps", "Building envelope retrofit",
            "District heating", "Rooftop solar", "Smart controls",
        ],
    },
    SectorCode.BUILDINGS_COMMERCIAL: {
        "name": "Buildings (Commercial)",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "kgCO2/m2/year",
        "activity_metric": "square meters floor area",
        "iea_chapter": "Chapter 2: Buildings (Commercial)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Buildings",
        "key_technologies": [
            "Heat pumps", "HVAC optimisation",
            "LED / efficient lighting", "Building envelope",
            "On-site renewable", "Smart BMS",
        ],
    },
    SectorCode.AGRICULTURE: {
        "name": "Agriculture",
        "sda_eligible": False,
        "approach": PathwayApproach.FLAG,
        "intensity_metric": "tCO2e/tonne food",
        "activity_metric": "tonnes food produced",
        "iea_chapter": "Chapter 6: Agriculture",
        "scope_coverage_required_pct": Decimal("67"),
        "sbti_methodology": "FLAG",
        "key_technologies": [
            "Precision farming", "Low-emission fertilizers",
            "Methane reduction (enteric)", "Soil carbon sequestration",
            "Agroforestry",
        ],
    },
    SectorCode.FOOD_BEVERAGE: {
        "name": "Food & Beverage",
        "sda_eligible": True,
        "approach": PathwayApproach.SDA,
        "intensity_metric": "tCO2e/tonne product",
        "activity_metric": "tonnes product produced",
        "iea_chapter": "Chapter 5: Industry (Food)",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "SDA-Food",
        "key_technologies": [
            "Heat recovery", "Electrification",
            "Refrigerant transition", "Process efficiency",
        ],
    },
    SectorCode.OIL_GAS: {
        "name": "Oil & Gas",
        "sda_eligible": False,
        "approach": PathwayApproach.ACA,
        "intensity_metric": "gCO2/MJ energy produced",
        "activity_metric": "MJ energy produced",
        "iea_chapter": "Chapter 1: Energy Supply",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "ACA (Oil & Gas)",
        "key_technologies": [
            "Methane leak reduction", "Flaring elimination",
            "CCS", "Renewable energy diversification",
            "Hydrogen production",
        ],
    },
    SectorCode.CROSS_SECTOR: {
        "name": "Cross-Sector (Generic)",
        "sda_eligible": False,
        "approach": PathwayApproach.ACA,
        "intensity_metric": "tCO2e/M revenue",
        "activity_metric": "revenue (M EUR/USD)",
        "iea_chapter": "Multiple chapters",
        "scope_coverage_required_pct": Decimal("95"),
        "sbti_methodology": "ACA",
        "key_technologies": [
            "Energy efficiency", "Renewable procurement",
            "Electrification", "Supply chain engagement",
        ],
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class IndustryCodeEntry(BaseModel):
    """A single industry classification code with optional metadata.

    Attributes:
        system: Classification system (NACE, GICS, ISIC, or manual).
        code: The classification code string.
        description: Human-readable description of the activity.
        revenue_share_pct: Revenue share for this code (0-100).
        is_primary: Whether this is the primary activity code.
    """
    system: ClassificationSystem = Field(
        ..., description="Classification system"
    )
    code: str = Field(
        ..., min_length=1, max_length=20, description="Classification code"
    )
    description: str = Field(
        default="", max_length=500, description="Activity description"
    )
    revenue_share_pct: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="Revenue share percentage"
    )
    is_primary: bool = Field(
        default=False, description="Primary activity indicator"
    )

class ManualSectorOverride(BaseModel):
    """Manual sector assignment for cases where code lookup is insufficient.

    Attributes:
        sector: Target sector code.
        justification: Reason for manual override.
        revenue_share_pct: Revenue share for this sector.
    """
    sector: SectorCode = Field(..., description="Sector code")
    justification: str = Field(
        default="", max_length=1000, description="Override justification"
    )
    revenue_share_pct: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="Revenue share"
    )

class EmissionsCoverage(BaseModel):
    """Emissions coverage data for SDA eligibility assessment.

    Attributes:
        total_scope1_tco2e: Total Scope 1 emissions.
        total_scope2_tco2e: Total Scope 2 emissions.
        covered_scope1_tco2e: Scope 1 included in SDA boundary.
        covered_scope2_tco2e: Scope 2 included in SDA boundary.
        total_scope3_tco2e: Total Scope 3 emissions.
        covered_scope3_tco2e: Scope 3 included in boundary.
    """
    total_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Total Scope 1 (tCO2e)"
    )
    total_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Total Scope 2 (tCO2e)"
    )
    covered_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Covered Scope 1 (tCO2e)"
    )
    covered_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Covered Scope 2 (tCO2e)"
    )
    total_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Total Scope 3 (tCO2e)"
    )
    covered_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Covered Scope 3 (tCO2e)"
    )

class ClassificationInput(BaseModel):
    """Input for sector classification.

    Attributes:
        entity_name: Company or entity name.
        entity_id: Optional unique entity identifier.
        industry_codes: One or more industry classification codes.
        manual_overrides: Optional manual sector assignments.
        total_revenue: Total annual revenue (any currency, for weighting).
        revenue_currency: Currency of revenue.
        emissions_coverage: Emissions data for SDA eligibility.
        reporting_year: Year of the data.
        country: Country of primary operations (ISO 3166-1 alpha-2).
        include_sda_validation: Whether to run SDA eligibility checks.
        include_iea_mapping: Whether to include IEA sector mapping.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    entity_id: str = Field(
        default="", max_length=100, description="Entity identifier"
    )
    industry_codes: List[IndustryCodeEntry] = Field(
        default_factory=list, description="Industry codes"
    )
    manual_overrides: List[ManualSectorOverride] = Field(
        default_factory=list, description="Manual sector overrides"
    )
    total_revenue: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Total revenue"
    )
    revenue_currency: str = Field(
        default="EUR", max_length=3, description="Revenue currency"
    )
    emissions_coverage: Optional[EmissionsCoverage] = Field(
        default=None, description="Emissions coverage"
    )
    reporting_year: int = Field(
        default=2024, ge=2015, le=2035, description="Reporting year"
    )
    country: str = Field(
        default="", max_length=2, description="Country (ISO alpha-2)"
    )
    include_sda_validation: bool = Field(
        default=True, description="Run SDA eligibility checks"
    )
    include_iea_mapping: bool = Field(
        default=True, description="Include IEA sector mapping"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class SectorMatch(BaseModel):
    """A single sector match from classification.

    Attributes:
        sector: Matched sector code.
        sector_name: Human-readable sector name.
        source_system: Classification system that produced this match.
        source_code: Original code that matched.
        revenue_share_pct: Revenue-weighted share.
        confidence: Confidence level of the match (0-1).
        priority: Priority level (primary, secondary, etc.).
        intensity_metric: Sector-specific intensity metric.
        activity_metric: Sector-specific activity metric.
        pathway_approach: Recommended pathway approach (SDA/ACA/FLAG).
    """
    sector: str = Field(default="")
    sector_name: str = Field(default="")
    source_system: str = Field(default="")
    source_code: str = Field(default="")
    revenue_share_pct: Decimal = Field(default=Decimal("0"))
    confidence: Decimal = Field(default=Decimal("0"))
    priority: str = Field(default=SectorPriority.MINOR.value)
    intensity_metric: str = Field(default="")
    activity_metric: str = Field(default="")
    pathway_approach: str = Field(default=PathwayApproach.ACA.value)

class SDAValidation(BaseModel):
    """SDA eligibility validation result.

    Attributes:
        eligibility: Overall SDA eligibility status.
        scope12_coverage_pct: Coverage of Scope 1+2 emissions.
        scope12_threshold_pct: Required coverage threshold.
        scope12_met: Whether coverage threshold is met.
        scope3_coverage_pct: Coverage of Scope 3 emissions.
        scope3_threshold_pct: Required Scope 3 threshold.
        scope3_met: Whether Scope 3 threshold is met.
        sector_is_sda: Whether primary sector has SDA methodology.
        methodology: Recommended methodology.
        validation_notes: Detailed validation notes.
    """
    eligibility: str = Field(default=SDAEligibility.REQUIRES_REVIEW.value)
    scope12_coverage_pct: Decimal = Field(default=Decimal("0"))
    scope12_threshold_pct: Decimal = Field(default=Decimal("95"))
    scope12_met: bool = Field(default=False)
    scope3_coverage_pct: Decimal = Field(default=Decimal("0"))
    scope3_threshold_pct: Decimal = Field(default=Decimal("67"))
    scope3_met: bool = Field(default=False)
    sector_is_sda: bool = Field(default=False)
    methodology: str = Field(default="")
    validation_notes: List[str] = Field(default_factory=list)

class IEASectorMapping(BaseModel):
    """IEA sector mapping for a matched sector.

    Attributes:
        sector: Source sector.
        iea_chapter: IEA NZE 2050 chapter reference.
        iea_sector_name: IEA sector naming.
        has_nze_pathway: Whether IEA provides a NZE pathway.
        has_aps_pathway: Whether IEA provides an APS pathway.
        has_steps_pathway: Whether IEA provides a STEPS pathway.
        key_milestones_available: Number of IEA milestones for this sector.
        technology_count: Number of key technologies mapped.
    """
    sector: str = Field(default="")
    iea_chapter: str = Field(default="")
    iea_sector_name: str = Field(default="")
    has_nze_pathway: bool = Field(default=True)
    has_aps_pathway: bool = Field(default=True)
    has_steps_pathway: bool = Field(default=True)
    key_milestones_available: int = Field(default=0)
    technology_count: int = Field(default=0)

class MultiSectorSummary(BaseModel):
    """Summary for multi-sector companies.

    Attributes:
        is_multi_sector: Whether company operates in 2+ sectors.
        sector_count: Number of distinct sectors identified.
        primary_sector: Primary sector by revenue.
        primary_revenue_share_pct: Primary sector revenue share.
        secondary_sectors: List of secondary sectors.
        requires_split_pathway: Whether split pathway analysis is needed.
        recommended_approach: Recommended overall approach.
    """
    is_multi_sector: bool = Field(default=False)
    sector_count: int = Field(default=0)
    primary_sector: str = Field(default="")
    primary_revenue_share_pct: Decimal = Field(default=Decimal("0"))
    secondary_sectors: List[str] = Field(default_factory=list)
    requires_split_pathway: bool = Field(default=False)
    recommended_approach: str = Field(default="")

class ClassificationResult(BaseModel):
    """Complete sector classification result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        entity_id: Entity identifier.
        reporting_year: Reporting year.
        country: Country of primary operations.
        sector_matches: All sector matches from classification.
        primary_sector: Primary sector determination.
        primary_sector_name: Human-readable primary sector name.
        primary_pathway_approach: Recommended pathway approach.
        sda_validation: SDA eligibility validation (if requested).
        iea_mappings: IEA sector mappings (if requested).
        multi_sector_summary: Multi-sector analysis.
        data_quality: Data quality assessment.
        recommendations: Classification recommendations.
        warnings: Classification warnings.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    reporting_year: int = Field(default=0)
    country: str = Field(default="")
    sector_matches: List[SectorMatch] = Field(default_factory=list)
    primary_sector: str = Field(default=SectorCode.CROSS_SECTOR.value)
    primary_sector_name: str = Field(default="Cross-Sector (Generic)")
    primary_pathway_approach: str = Field(default=PathwayApproach.ACA.value)
    sda_validation: Optional[SDAValidation] = Field(default=None)
    iea_mappings: List[IEASectorMapping] = Field(default_factory=list)
    multi_sector_summary: Optional[MultiSectorSummary] = Field(default=None)
    data_quality: str = Field(default=DataQuality.MEDIUM.value)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SectorClassificationEngine:
    """Sector classification engine for PACK-028 Sector Pathway Pack.

    Classifies companies into SBTi SDA, IEA NZE, and FLAG sectors
    using NACE Rev.2, GICS, ISIC Rev.4 codes and revenue-weighted
    multi-sector analysis.

    All classifications use deterministic code lookups. No LLM in
    any path.

    Usage::

        engine = SectorClassificationEngine()
        result = engine.calculate(classification_input)
        print(f"Primary: {result.primary_sector}")
        for m in result.sector_matches:
            print(f"  {m.sector_name}: {m.revenue_share_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: ClassificationInput) -> ClassificationResult:
        """Run complete sector classification.

        Args:
            data: Validated classification input.

        Returns:
            ClassificationResult with sector matches, SDA validation,
            IEA mappings, and multi-sector analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Sector classification: entity=%s, codes=%d, overrides=%d",
            data.entity_name, len(data.industry_codes),
            len(data.manual_overrides),
        )

        # Step 1: Classify via industry codes
        code_matches = self._classify_from_codes(data.industry_codes)

        # Step 2: Apply manual overrides
        override_matches = self._apply_manual_overrides(data.manual_overrides)

        # Step 3: Merge and deduplicate
        all_matches = self._merge_matches(code_matches, override_matches)

        # Step 4: Revenue-weighted prioritization
        all_matches = self._prioritize_by_revenue(all_matches)

        # Step 5: Determine primary sector
        primary_sector, primary_name, primary_approach = (
            self._determine_primary(all_matches)
        )

        # Step 6: SDA eligibility validation
        sda_validation: Optional[SDAValidation] = None
        if data.include_sda_validation:
            sda_validation = self._validate_sda_eligibility(
                primary_sector, data.emissions_coverage
            )

        # Step 7: IEA sector mapping
        iea_mappings: List[IEASectorMapping] = []
        if data.include_iea_mapping:
            iea_mappings = self._build_iea_mappings(all_matches)

        # Step 8: Multi-sector summary
        multi_summary = self._build_multi_sector_summary(
            all_matches, primary_sector
        )

        # Step 9: Data quality assessment
        data_quality = self._assess_data_quality(data, all_matches)

        # Step 10: Recommendations
        recommendations = self._generate_recommendations(
            data, all_matches, primary_sector, sda_validation,
            multi_summary
        )

        # Step 11: Warnings
        warnings = self._generate_warnings(
            data, all_matches, primary_sector, sda_validation
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ClassificationResult(
            entity_name=data.entity_name,
            entity_id=data.entity_id,
            reporting_year=data.reporting_year,
            country=data.country,
            sector_matches=all_matches,
            primary_sector=primary_sector,
            primary_sector_name=primary_name,
            primary_pathway_approach=primary_approach,
            sda_validation=sda_validation,
            iea_mappings=iea_mappings,
            multi_sector_summary=multi_summary,
            data_quality=data_quality,
            recommendations=recommendations,
            warnings=warnings,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Classification complete: entity=%s, primary=%s, "
            "matches=%d, sda=%s",
            data.entity_name, primary_sector,
            len(all_matches),
            sda_validation.eligibility if sda_validation else "skipped",
        )
        return result

    # ------------------------------------------------------------------ #
    # Code Classification                                                 #
    # ------------------------------------------------------------------ #

    def _classify_from_codes(
        self,
        codes: List[IndustryCodeEntry],
    ) -> List[SectorMatch]:
        """Classify from industry codes using lookup tables.

        Args:
            codes: Industry code entries.

        Returns:
            List of sector matches.
        """
        matches: List[SectorMatch] = []

        for entry in codes:
            sector: Optional[SectorCode] = None
            confidence = Decimal("0")

            if entry.system == ClassificationSystem.NACE_REV2:
                sector = self._lookup_nace(entry.code)
                confidence = Decimal("0.95") if sector else Decimal("0")
            elif entry.system == ClassificationSystem.GICS:
                sector = self._lookup_gics(entry.code)
                confidence = Decimal("0.90") if sector else Decimal("0")
            elif entry.system == ClassificationSystem.ISIC_REV4:
                sector = self._lookup_isic(entry.code)
                confidence = Decimal("0.92") if sector else Decimal("0")
            elif entry.system == ClassificationSystem.MANUAL:
                # Try to parse manual code as sector directly
                try:
                    sector = SectorCode(entry.code)
                    confidence = Decimal("0.80")
                except ValueError:
                    sector = None

            if sector is not None:
                meta = SDA_SECTOR_METADATA.get(sector, {})
                matches.append(SectorMatch(
                    sector=sector.value,
                    sector_name=meta.get("name", sector.value),
                    source_system=entry.system.value,
                    source_code=entry.code,
                    revenue_share_pct=entry.revenue_share_pct,
                    confidence=confidence,
                    priority=SectorPriority.PRIMARY.value
                    if entry.is_primary
                    else SectorPriority.SECONDARY.value,
                    intensity_metric=meta.get("intensity_metric", ""),
                    activity_metric=meta.get("activity_metric", ""),
                    pathway_approach=meta.get(
                        "approach", PathwayApproach.ACA
                    ).value if isinstance(meta.get("approach"), PathwayApproach)
                    else str(meta.get("approach", PathwayApproach.ACA.value)),
                ))
            else:
                logger.debug(
                    "No sector match for %s code: %s",
                    entry.system.value, entry.code,
                )

        return matches

    def _lookup_nace(self, code: str) -> Optional[SectorCode]:
        """Look up NACE Rev.2 code in mapping table.

        Tries exact match first, then progressively shorter prefixes.
        """
        normalized = code.strip().upper()
        # Try exact match
        if normalized in NACE_TO_SECTOR:
            return NACE_TO_SECTOR[normalized]
        # Try without leading letter section if it has one
        # Try progressively shorter codes
        parts = normalized
        while len(parts) > 1:
            if parts in NACE_TO_SECTOR:
                return NACE_TO_SECTOR[parts]
            # Remove last character or last segment
            if "." in parts:
                parts = parts.rsplit(".", 1)[0]
            else:
                parts = parts[:-1]
        return None

    def _lookup_gics(self, code: str) -> Optional[SectorCode]:
        """Look up GICS code in mapping table.

        Tries 8-digit, 6-digit, 4-digit, 2-digit.
        """
        normalized = code.strip()
        # Try exact match
        if normalized in GICS_TO_SECTOR:
            return GICS_TO_SECTOR[normalized]
        # Try truncating to sub-industry (8), industry (6), group (4), sector (2)
        for length in [8, 6, 4, 2]:
            prefix = normalized[:length]
            if prefix in GICS_TO_SECTOR:
                return GICS_TO_SECTOR[prefix]
        return None

    def _lookup_isic(self, code: str) -> Optional[SectorCode]:
        """Look up ISIC Rev.4 code in mapping table."""
        normalized = code.strip()
        if normalized in ISIC_TO_SECTOR:
            return ISIC_TO_SECTOR[normalized]
        # Try progressively shorter
        for length in range(len(normalized), 0, -1):
            prefix = normalized[:length]
            if prefix in ISIC_TO_SECTOR:
                return ISIC_TO_SECTOR[prefix]
        return None

    # ------------------------------------------------------------------ #
    # Manual Overrides                                                    #
    # ------------------------------------------------------------------ #

    def _apply_manual_overrides(
        self,
        overrides: List[ManualSectorOverride],
    ) -> List[SectorMatch]:
        """Convert manual overrides to sector matches."""
        matches: List[SectorMatch] = []
        for override in overrides:
            meta = SDA_SECTOR_METADATA.get(override.sector, {})
            matches.append(SectorMatch(
                sector=override.sector.value,
                sector_name=meta.get("name", override.sector.value),
                source_system=ClassificationSystem.MANUAL.value,
                source_code=override.sector.value,
                revenue_share_pct=override.revenue_share_pct,
                confidence=Decimal("0.85"),
                priority=SectorPriority.PRIMARY.value,
                intensity_metric=meta.get("intensity_metric", ""),
                activity_metric=meta.get("activity_metric", ""),
                pathway_approach=meta.get(
                    "approach", PathwayApproach.ACA
                ).value if isinstance(meta.get("approach"), PathwayApproach)
                else str(meta.get("approach", PathwayApproach.ACA.value)),
            ))
        return matches

    # ------------------------------------------------------------------ #
    # Merge & Deduplicate                                                 #
    # ------------------------------------------------------------------ #

    def _merge_matches(
        self,
        code_matches: List[SectorMatch],
        override_matches: List[SectorMatch],
    ) -> List[SectorMatch]:
        """Merge and deduplicate sector matches.

        Manual overrides take precedence over code-based matches.
        When the same sector appears multiple times, keep the one
        with the highest revenue share.
        """
        # Overrides first (higher priority)
        all_raw = override_matches + code_matches
        seen: Dict[str, SectorMatch] = {}
        for match in all_raw:
            key = match.sector
            if key not in seen:
                seen[key] = match
            else:
                existing = seen[key]
                # Keep the one with higher revenue share
                if match.revenue_share_pct > existing.revenue_share_pct:
                    seen[key] = match
                # Or higher confidence if revenue is equal
                elif (
                    match.revenue_share_pct == existing.revenue_share_pct
                    and match.confidence > existing.confidence
                ):
                    seen[key] = match
        return list(seen.values())

    # ------------------------------------------------------------------ #
    # Revenue-Weighted Prioritization                                     #
    # ------------------------------------------------------------------ #

    def _prioritize_by_revenue(
        self,
        matches: List[SectorMatch],
    ) -> List[SectorMatch]:
        """Sort and assign priority based on revenue share.

        Primary: >= 50% revenue
        Secondary: >= 20% revenue
        Tertiary: >= 5% revenue
        Minor: < 5% revenue
        """
        # Sort by revenue share descending
        sorted_matches = sorted(
            matches,
            key=lambda m: float(m.revenue_share_pct),
            reverse=True,
        )

        for i, match in enumerate(sorted_matches):
            rev = match.revenue_share_pct
            if i == 0 or rev >= Decimal("50"):
                match.priority = SectorPriority.PRIMARY.value
            elif rev >= Decimal("20"):
                match.priority = SectorPriority.SECONDARY.value
            elif rev >= Decimal("5"):
                match.priority = SectorPriority.TERTIARY.value
            else:
                match.priority = SectorPriority.MINOR.value

        return sorted_matches

    # ------------------------------------------------------------------ #
    # Primary Sector Determination                                        #
    # ------------------------------------------------------------------ #

    def _determine_primary(
        self,
        matches: List[SectorMatch],
    ) -> Tuple[str, str, str]:
        """Determine primary sector from sorted matches.

        Returns:
            Tuple of (sector_code, sector_name, pathway_approach).
        """
        if not matches:
            return (
                SectorCode.CROSS_SECTOR.value,
                "Cross-Sector (Generic)",
                PathwayApproach.ACA.value,
            )

        primary = matches[0]
        return (
            primary.sector,
            primary.sector_name,
            primary.pathway_approach,
        )

    # ------------------------------------------------------------------ #
    # SDA Eligibility Validation                                          #
    # ------------------------------------------------------------------ #

    def _validate_sda_eligibility(
        self,
        primary_sector: str,
        emissions_coverage: Optional[EmissionsCoverage],
    ) -> SDAValidation:
        """Validate SDA eligibility for the primary sector.

        SDA requires:
        1. Sector must have SDA methodology
        2. 95% of Scope 1+2 emissions must be covered
        3. 67% of Scope 3 must be covered (for near-term)
        """
        validation = SDAValidation()

        # Check sector SDA availability
        try:
            sector_enum = SectorCode(primary_sector)
        except ValueError:
            validation.eligibility = SDAEligibility.INELIGIBLE.value
            validation.validation_notes.append(
                f"Unknown sector: {primary_sector}"
            )
            return validation

        meta = SDA_SECTOR_METADATA.get(sector_enum, {})
        validation.sector_is_sda = meta.get("sda_eligible", False)
        validation.methodology = meta.get("sbti_methodology", "ACA")
        validation.scope12_threshold_pct = meta.get(
            "scope_coverage_required_pct", Decimal("95")
        )

        if not validation.sector_is_sda:
            validation.eligibility = SDAEligibility.INELIGIBLE.value
            validation.validation_notes.append(
                f"Sector '{meta.get('name', primary_sector)}' does not have "
                f"SDA methodology. Recommended approach: "
                f"{meta.get('approach', PathwayApproach.ACA).value if isinstance(meta.get('approach'), PathwayApproach) else meta.get('approach', 'ACA')}"
            )
            return validation

        # Check emissions coverage
        if emissions_coverage is None:
            validation.eligibility = SDAEligibility.REQUIRES_REVIEW.value
            validation.validation_notes.append(
                "No emissions coverage data provided. Cannot validate "
                "Scope 1+2 coverage threshold."
            )
            return validation

        # Scope 1+2 coverage
        total_s12 = (
            emissions_coverage.total_scope1_tco2e
            + emissions_coverage.total_scope2_tco2e
        )
        covered_s12 = (
            emissions_coverage.covered_scope1_tco2e
            + emissions_coverage.covered_scope2_tco2e
        )
        if total_s12 > Decimal("0"):
            validation.scope12_coverage_pct = _round_val(
                _safe_pct(covered_s12, total_s12), 2
            )
        validation.scope12_met = (
            validation.scope12_coverage_pct
            >= validation.scope12_threshold_pct
        )

        # Scope 3 coverage
        if emissions_coverage.total_scope3_tco2e > Decimal("0"):
            validation.scope3_coverage_pct = _round_val(
                _safe_pct(
                    emissions_coverage.covered_scope3_tco2e,
                    emissions_coverage.total_scope3_tco2e,
                ),
                2,
            )
        validation.scope3_met = (
            validation.scope3_coverage_pct
            >= validation.scope3_threshold_pct
        )

        # Determine eligibility
        if validation.scope12_met and validation.sector_is_sda:
            validation.eligibility = SDAEligibility.ELIGIBLE.value
            validation.validation_notes.append(
                f"SDA eligible: Scope 1+2 coverage "
                f"{validation.scope12_coverage_pct}% >= "
                f"{validation.scope12_threshold_pct}% threshold."
            )
        elif validation.sector_is_sda and not validation.scope12_met:
            validation.eligibility = SDAEligibility.PARTIAL.value
            validation.validation_notes.append(
                f"SDA partially eligible: Scope 1+2 coverage "
                f"{validation.scope12_coverage_pct}% < "
                f"{validation.scope12_threshold_pct}% threshold. "
                f"Increase boundary coverage to qualify."
            )
        else:
            validation.eligibility = SDAEligibility.INELIGIBLE.value

        # Scope 3 notes
        if not validation.scope3_met:
            validation.validation_notes.append(
                f"Scope 3 coverage {validation.scope3_coverage_pct}% < "
                f"{validation.scope3_threshold_pct}% threshold. "
                f"Expand Scope 3 measurement for full SBTi compliance."
            )

        return validation

    # ------------------------------------------------------------------ #
    # IEA Sector Mapping                                                  #
    # ------------------------------------------------------------------ #

    def _build_iea_mappings(
        self,
        matches: List[SectorMatch],
    ) -> List[IEASectorMapping]:
        """Build IEA sector mappings for each matched sector."""
        mappings: List[IEASectorMapping] = []
        seen_sectors: set = set()

        # IEA milestone counts per sector (approximate from IEA NZE 2050)
        iea_milestone_counts: Dict[str, int] = {
            SectorCode.POWER_GENERATION.value: 55,
            SectorCode.STEEL.value: 35,
            SectorCode.CEMENT.value: 30,
            SectorCode.ALUMINUM.value: 25,
            SectorCode.PULP_PAPER.value: 20,
            SectorCode.CHEMICALS.value: 35,
            SectorCode.AVIATION.value: 30,
            SectorCode.SHIPPING.value: 25,
            SectorCode.ROAD_TRANSPORT.value: 40,
            SectorCode.RAIL.value: 20,
            SectorCode.BUILDINGS_RESIDENTIAL.value: 35,
            SectorCode.BUILDINGS_COMMERCIAL.value: 35,
            SectorCode.AGRICULTURE.value: 25,
            SectorCode.FOOD_BEVERAGE.value: 20,
            SectorCode.OIL_GAS.value: 30,
            SectorCode.CROSS_SECTOR.value: 10,
        }

        for match in matches:
            if match.sector in seen_sectors:
                continue
            seen_sectors.add(match.sector)

            try:
                sector_enum = SectorCode(match.sector)
            except ValueError:
                continue

            meta = SDA_SECTOR_METADATA.get(sector_enum, {})
            tech_count = len(meta.get("key_technologies", []))
            milestone_count = iea_milestone_counts.get(match.sector, 0)

            mappings.append(IEASectorMapping(
                sector=match.sector,
                iea_chapter=meta.get("iea_chapter", ""),
                iea_sector_name=meta.get("name", match.sector),
                has_nze_pathway=True,
                has_aps_pathway=True,
                has_steps_pathway=True,
                key_milestones_available=milestone_count,
                technology_count=tech_count,
            ))

        return mappings

    # ------------------------------------------------------------------ #
    # Multi-Sector Summary                                                #
    # ------------------------------------------------------------------ #

    def _build_multi_sector_summary(
        self,
        matches: List[SectorMatch],
        primary_sector: str,
    ) -> MultiSectorSummary:
        """Build multi-sector summary analysis."""
        unique_sectors = list({m.sector for m in matches})
        is_multi = len(unique_sectors) > 1

        primary_rev = Decimal("0")
        secondary_sectors: List[str] = []
        for m in matches:
            if m.sector == primary_sector:
                primary_rev = m.revenue_share_pct
            else:
                secondary_sectors.append(m.sector_name)

        # Determine if split pathway is needed
        # Split pathway recommended when no single sector > 80% revenue
        requires_split = is_multi and primary_rev < Decimal("80")

        # Recommended approach for multi-sector
        if not is_multi:
            recommended = "single_sector_pathway"
        elif requires_split:
            recommended = "split_sector_pathway"
        else:
            recommended = "primary_sector_dominant"

        return MultiSectorSummary(
            is_multi_sector=is_multi,
            sector_count=len(unique_sectors),
            primary_sector=primary_sector,
            primary_revenue_share_pct=primary_rev,
            secondary_sectors=secondary_sectors,
            requires_split_pathway=requires_split,
            recommended_approach=recommended,
        )

    # ------------------------------------------------------------------ #
    # Data Quality                                                        #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(
        self,
        data: ClassificationInput,
        matches: List[SectorMatch],
    ) -> str:
        """Assess classification data quality."""
        score = 0

        # Industry codes provided
        if len(data.industry_codes) >= 1:
            score += 2
        if len(data.industry_codes) >= 2:
            score += 1

        # Multiple classification systems used
        systems = {c.system for c in data.industry_codes}
        if len(systems) >= 2:
            score += 2

        # Revenue shares provided
        has_revenue = any(
            c.revenue_share_pct != Decimal("100")
            for c in data.industry_codes
        )
        if has_revenue:
            score += 1

        # Emissions coverage provided
        if data.emissions_coverage is not None:
            score += 2

        # Country provided
        if data.country:
            score += 1

        # At least one match found
        if matches:
            score += 1

        if score >= 8:
            return DataQuality.HIGH.value
        elif score >= 5:
            return DataQuality.MEDIUM.value
        elif score >= 2:
            return DataQuality.LOW.value
        else:
            return DataQuality.ESTIMATED.value

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: ClassificationInput,
        matches: List[SectorMatch],
        primary_sector: str,
        sda_validation: Optional[SDAValidation],
        multi_summary: MultiSectorSummary,
    ) -> List[str]:
        """Generate sector classification recommendations."""
        recs: List[str] = []

        # No codes provided
        if not data.industry_codes and not data.manual_overrides:
            recs.append(
                "Provide at least one NACE Rev.2, GICS, or ISIC Rev.4 "
                "code for automated sector classification."
            )

        # Only one classification system
        systems = {c.system for c in data.industry_codes}
        if len(systems) == 1:
            recs.append(
                "Cross-reference with a second classification system "
                "(NACE + GICS recommended) for higher confidence."
            )

        # Multi-sector
        if multi_summary.requires_split_pathway:
            recs.append(
                f"Multi-sector company with {multi_summary.sector_count} "
                f"sectors. Consider split pathway analysis with separate "
                f"SDA targets for each sector."
            )

        # SDA eligibility gaps
        if sda_validation and sda_validation.eligibility == SDAEligibility.PARTIAL.value:
            recs.append(
                "Increase Scope 1+2 boundary coverage to >= "
                f"{sda_validation.scope12_threshold_pct}% "
                "to fully qualify for SDA pathway."
            )

        # Missing emissions data
        if data.emissions_coverage is None:
            recs.append(
                "Provide emissions coverage data (Scope 1, 2, 3) to "
                "validate SDA eligibility thresholds."
            )

        # Revenue weighting
        if len(data.industry_codes) > 1:
            all_100 = all(
                c.revenue_share_pct == Decimal("100")
                for c in data.industry_codes
            )
            if all_100:
                recs.append(
                    "Set revenue_share_pct for each code to enable "
                    "revenue-weighted sector prioritization."
                )

        # Agriculture/FLAG
        if primary_sector == SectorCode.AGRICULTURE.value:
            recs.append(
                "Agriculture sector requires separate FLAG pathway. "
                "Ensure land-use change and agricultural emissions are "
                "accounted per SBTi FLAG guidance."
            )

        # Oil & Gas
        if primary_sector == SectorCode.OIL_GAS.value:
            recs.append(
                "Oil & Gas sector uses ACA approach (not SDA). "
                "Include Scope 3 Category 11 (Use of Sold Products) "
                "which typically dominates total emissions."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Warnings                                                            #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        data: ClassificationInput,
        matches: List[SectorMatch],
        primary_sector: str,
        sda_validation: Optional[SDAValidation],
    ) -> List[str]:
        """Generate classification warnings."""
        warnings: List[str] = []

        # No matches found
        if not matches:
            warnings.append(
                "No sector matches found from provided codes. "
                "Defaulting to cross-sector (ACA) approach."
            )

        # Low confidence
        low_confidence = [
            m for m in matches
            if m.confidence < Decimal("0.80")
        ]
        if low_confidence:
            for m in low_confidence:
                warnings.append(
                    f"Low confidence ({m.confidence}) for sector "
                    f"'{m.sector_name}' from {m.source_system} "
                    f"code '{m.source_code}'."
                )

        # SDA coverage gap
        if sda_validation and not sda_validation.scope12_met:
            gap = (
                sda_validation.scope12_threshold_pct
                - sda_validation.scope12_coverage_pct
            )
            if gap > Decimal("0"):
                warnings.append(
                    f"Scope 1+2 coverage gap: {gap}% below SDA threshold. "
                    f"SDA pathway may not be accepted by SBTi."
                )

        # Revenue shares exceed 100%
        total_rev = sum(
            m.revenue_share_pct for m in matches
        )
        if total_rev > Decimal("100"):
            warnings.append(
                f"Total revenue share ({total_rev}%) exceeds 100%. "
                f"Verify revenue allocation across sectors."
            )

        return warnings

    # ------------------------------------------------------------------ #
    # Batch Classification                                                #
    # ------------------------------------------------------------------ #

    def calculate_batch(
        self,
        inputs: List[ClassificationInput],
    ) -> List[ClassificationResult]:
        """Classify multiple entities in batch.

        Args:
            inputs: List of classification inputs.

        Returns:
            List of classification results, one per input.
        """
        results: List[ClassificationResult] = []
        for data in inputs:
            try:
                result = self.calculate(data)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch classification error for %s: %s",
                    data.entity_name, exc,
                )
                # Return a minimal error result
                results.append(ClassificationResult(
                    entity_name=data.entity_name,
                    entity_id=data.entity_id,
                    warnings=[f"Classification error: {exc}"],
                ))
        return results

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_supported_sectors(self) -> List[Dict[str, Any]]:
        """Return list of all supported sectors with metadata."""
        sectors: List[Dict[str, Any]] = []
        for sector_code, meta in SDA_SECTOR_METADATA.items():
            sectors.append({
                "sector_code": sector_code.value,
                "name": meta["name"],
                "sda_eligible": meta["sda_eligible"],
                "approach": meta["approach"].value
                if isinstance(meta["approach"], PathwayApproach)
                else str(meta["approach"]),
                "intensity_metric": meta["intensity_metric"],
                "activity_metric": meta["activity_metric"],
                "iea_chapter": meta["iea_chapter"],
                "sbti_methodology": meta["sbti_methodology"],
                "key_technologies": meta["key_technologies"],
            })
        return sectors

    def get_nace_codes(self) -> Dict[str, str]:
        """Return NACE Rev.2 code -> sector mapping."""
        return {k: v.value for k, v in NACE_TO_SECTOR.items()}

    def get_gics_codes(self) -> Dict[str, str]:
        """Return GICS code -> sector mapping."""
        return {k: v.value for k, v in GICS_TO_SECTOR.items()}

    def get_isic_codes(self) -> Dict[str, str]:
        """Return ISIC Rev.4 code -> sector mapping."""
        return {k: v.value for k, v in ISIC_TO_SECTOR.items()}
