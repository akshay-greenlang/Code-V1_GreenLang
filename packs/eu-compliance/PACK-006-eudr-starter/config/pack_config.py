"""
PACK-006 EUDR Starter Pack - Configuration Manager

This module implements the EUDRPackConfig and PackConfig classes that load,
merge, and validate all configuration for the EUDR Starter Pack. It supports
layered configuration with base manifest, size presets, sector presets, and
environment overrides.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Size preset (large_operator / mid_market / sme / first_time)
    3. Sector preset (palm_oil / timber_wood / cocoa_coffee / soy_cattle / rubber)
    4. Environment overrides (EUDR_PACK_* environment variables)
    5. Explicit runtime overrides

Regulatory Context:
    - EUDR: Regulation (EU) 2023/1115
    - Articles: 3, 4, 8, 9, 10, 11, 12, 13, 29, 33
    - Cutoff Date: 31 December 2020
    - Commodities: 7 (cattle, cocoa, coffee, oil palm, rubber, soya, wood)
    - Annex I CN Codes: 400+ product classifications

Example:
    >>> config = PackConfig.load(
    ...     size_preset="mid_market",
    ...     sector_preset="palm_oil",
    ... )
    >>> print(config.pack.metadata.display_name)
    'EUDR Starter Pack'
    >>> print(config.active_agents)
    ['AGENT-EUDR-001', 'AGENT-EUDR-002', ...]
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - EUDR-specific enumeration types
# =============================================================================


class EUDRCommodity(str, Enum):
    """EUDR Article 1 regulated commodities."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class OperatorType(str, Enum):
    """EUDR operator classification per Article 2."""

    OPERATOR = "OPERATOR"
    TRADER = "TRADER"


class CompanySize(str, Enum):
    """Company size classification for EUDR obligations."""

    SME = "SME"
    MID_MARKET = "MID_MARKET"
    LARGE = "LARGE"


class DDSType(str, Enum):
    """Due Diligence Statement type per Articles 4 and 13."""

    STANDARD = "STANDARD"
    SIMPLIFIED = "SIMPLIFIED"


class DDSStatus(str, Enum):
    """Due Diligence Statement lifecycle status."""

    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    AMENDED = "AMENDED"


class RiskLevel(str, Enum):
    """Risk level classification for suppliers and commodities."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CountryBenchmark(str, Enum):
    """Country benchmarking classification per Article 29."""

    LOW_RISK = "LOW_RISK"
    STANDARD_RISK = "STANDARD_RISK"
    HIGH_RISK = "HIGH_RISK"


class CertificationScheme(str, Enum):
    """Recognized voluntary certification schemes."""

    FSC = "FSC"
    PEFC = "PEFC"
    RSPO = "RSPO"
    ISCC = "ISCC"
    RAINFOREST_ALLIANCE = "RAINFOREST_ALLIANCE"
    UTZ = "UTZ"
    FAIRTRADE = "FAIRTRADE"
    RTRS = "RTRS"
    PROTERRA = "PROTERRA"
    FOUR_C = "FOUR_C"
    ORGANIC = "ORGANIC"
    SFI = "SFI"
    MSPO = "MSPO"
    ISPO = "ISPO"


class ChainOfCustodyModel(str, Enum):
    """Chain of custody models per industry standard."""

    IDENTITY_PRESERVED = "IDENTITY_PRESERVED"
    SEGREGATED = "SEGREGATED"
    MASS_BALANCE = "MASS_BALANCE"
    CONTROLLED_SOURCES = "CONTROLLED_SOURCES"


class CoordinateFormat(str, Enum):
    """GPS coordinate format types."""

    DECIMAL_DEGREES = "DECIMAL_DEGREES"
    DMS = "DMS"
    UTM = "UTM"


class SupplierDDStatus(str, Enum):
    """Supplier due diligence completion status."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    VERIFIED = "VERIFIED"
    EXPIRED = "EXPIRED"


class AreaUnit(str, Enum):
    """Area measurement units for plot boundaries."""

    HECTARES = "HECTARES"
    SQUARE_METERS = "SQUARE_METERS"
    ACRES = "ACRES"
    SQUARE_KILOMETERS = "SQUARE_KILOMETERS"


class AuthType(str, Enum):
    """Authentication type for EU Information System."""

    OAUTH2 = "OAUTH2"
    EIDAS = "EIDAS"
    CERTIFICATE = "CERTIFICATE"


# =============================================================================
# Reference Data Constants - EUDR regulatory reference data
# =============================================================================

# Cutoff date per Article 1(1) - no deforestation after this date
CUTOFF_DATE: date = date(2020, 12, 31)

# Polygon area threshold (hectares) - above this, polygon required; below, point OK
POLYGON_AREA_THRESHOLD_HA: float = 4.0

# EUDR commodities with display names and Annex I article references
EUDR_COMMODITIES: Dict[str, Dict[str, Any]] = {
    "cattle": {
        "display_name": "Cattle",
        "article": "Article 1(1)(a)",
        "cn_code_count": 12,
        "description": "Live bovine animals, beef, leather, hides, tallow",
    },
    "cocoa": {
        "display_name": "Cocoa",
        "article": "Article 1(1)(b)",
        "cn_code_count": 6,
        "description": "Cocoa beans, paste, butter, powder, chocolate",
    },
    "coffee": {
        "display_name": "Coffee",
        "article": "Article 1(1)(c)",
        "cn_code_count": 2,
        "description": "Coffee beans, roasted, ground, instant, extracts",
    },
    "oil_palm": {
        "display_name": "Oil Palm",
        "article": "Article 1(1)(d)",
        "cn_code_count": 9,
        "description": "Palm oil, palm kernel oil, oleochemicals, biodiesel",
    },
    "rubber": {
        "display_name": "Rubber",
        "article": "Article 1(1)(e)",
        "cn_code_count": 12,
        "description": "Natural rubber, tyres, gloves, industrial goods",
    },
    "soya": {
        "display_name": "Soya",
        "article": "Article 1(1)(f)",
        "cn_code_count": 5,
        "description": "Soybeans, soybean oil, meal, lecithin, animal feed",
    },
    "wood": {
        "display_name": "Wood",
        "article": "Article 1(1)(g)",
        "cn_code_count": 65,
        "description": "Timber, charcoal, pulp, paper, furniture, cork",
    },
}

# Comprehensive Annex I CN code database per commodity
ANNEX_I_CN_CODES: Dict[str, List[Dict[str, str]]] = {
    "cattle": [
        {"code": "0102", "description": "Live bovine animals"},
        {"code": "0201", "description": "Meat of bovine animals, fresh or chilled"},
        {"code": "0202", "description": "Meat of bovine animals, frozen"},
        {"code": "0206 10", "description": "Edible offal of bovine animals, fresh or chilled"},
        {"code": "0206 22", "description": "Edible livers of bovine animals, frozen"},
        {"code": "0206 29", "description": "Edible offal of bovine animals, frozen (other)"},
        {"code": "0210 20", "description": "Meat of bovine animals, salted, dried or smoked"},
        {"code": "1501", "description": "Pig fat and poultry fat (bovine tallow included)"},
        {"code": "1502", "description": "Fats of bovine animals, sheep or goats"},
        {"code": "1602 50", "description": "Prepared or preserved meat of bovine animals"},
        {"code": "4101", "description": "Raw hides and skins of bovine or equine animals"},
        {"code": "4104", "description": "Tanned or crust hides and skins of bovine or equine"},
        {"code": "4107", "description": "Leather further prepared after tanning, bovine"},
        {"code": "4301", "description": "Raw furskins"},
    ],
    "cocoa": [
        {"code": "1801 00 00", "description": "Cocoa beans, whole or broken, raw or roasted"},
        {"code": "1802 00 00", "description": "Cocoa shells, husks, skins and other cocoa waste"},
        {"code": "1803", "description": "Cocoa paste, whether or not defatted"},
        {"code": "1804 00 00", "description": "Cocoa butter, fat and oil"},
        {"code": "1805 00 00", "description": "Cocoa powder, not containing added sugar"},
        {"code": "1806", "description": "Chocolate and other food preparations containing cocoa"},
    ],
    "coffee": [
        {"code": "0901", "description": "Coffee, whether or not roasted or decaffeinated"},
        {"code": "2101 11", "description": "Extracts, essences and concentrates of coffee"},
        {"code": "2101 12", "description": "Preparations with a basis of coffee extracts"},
    ],
    "oil_palm": [
        {"code": "1207 10", "description": "Palm nuts and kernels"},
        {"code": "1511", "description": "Palm oil and its fractions"},
        {"code": "1513 21", "description": "Crude palm kernel or babassu oil"},
        {"code": "1513 29", "description": "Palm kernel or babassu oil, refined"},
        {"code": "1516 20", "description": "Vegetable fats and oils, hydrogenated (palm)"},
        {"code": "2915", "description": "Saturated acyclic monocarboxylic acids (palm fatty acids)"},
        {"code": "2916", "description": "Unsaturated acyclic monocarboxylic acids (palm oleic)"},
        {"code": "3401", "description": "Soap, organic surface-active products (palm-derived)"},
        {"code": "3823", "description": "Industrial monocarboxylic fatty acids (palm)"},
        {"code": "3826 00", "description": "Biodiesel (palm-derived FAME)"},
    ],
    "rubber": [
        {"code": "4001", "description": "Natural rubber in primary forms or plates, sheets, strip"},
        {"code": "4005", "description": "Compounded rubber, unvulcanised, in primary forms"},
        {"code": "4006", "description": "Other forms of unvulcanised rubber (rods, tubes)"},
        {"code": "4007 00 00", "description": "Vulcanised rubber thread and cord"},
        {"code": "4008", "description": "Plates, sheets, strip of vulcanised rubber (non-cellular)"},
        {"code": "4010", "description": "Conveyor or transmission belts of vulcanised rubber"},
        {"code": "4011", "description": "New pneumatic tyres, of rubber"},
        {"code": "4012", "description": "Retreaded or used pneumatic tyres of rubber"},
        {"code": "4013", "description": "Inner tubes, of rubber"},
        {"code": "4015", "description": "Articles of apparel (gloves, mittens) of vulcanised rubber"},
        {"code": "4016", "description": "Other articles of vulcanised rubber (non-hard)"},
        {"code": "4017 00", "description": "Hard rubber (ebonite) in all forms"},
    ],
    "soya": [
        {"code": "1201", "description": "Soya beans, whether or not broken"},
        {"code": "1208 10", "description": "Soya bean flour and meal"},
        {"code": "1507", "description": "Soya-bean oil and its fractions"},
        {"code": "2304 00 00", "description": "Oil-cake and other solid residues of soya-bean"},
        {"code": "2309", "description": "Preparations for animal feeding (soy-based)"},
    ],
    "wood": [
        {"code": "4401", "description": "Fuel wood, wood in chips, sawdust, waste and scrap"},
        {"code": "4402", "description": "Wood charcoal"},
        {"code": "4403", "description": "Wood in the rough"},
        {"code": "4404", "description": "Hoopwood, split poles, piles, pickets, stakes"},
        {"code": "4405 00 00", "description": "Wood wool; wood flour"},
        {"code": "4406", "description": "Railway or tramway sleepers (cross-ties) of wood"},
        {"code": "4407", "description": "Wood sawn or chipped lengthwise, >6 mm thick"},
        {"code": "4408", "description": "Sheets for veneering, plywood, sawn wood <=6 mm"},
        {"code": "4409", "description": "Wood continuously shaped along any edge or face"},
        {"code": "4410", "description": "Particle board, OSB and similar board of wood"},
        {"code": "4411", "description": "Fibreboard of wood or other ligneous materials"},
        {"code": "4412", "description": "Plywood, veneered panels, similar laminated wood"},
        {"code": "4413 00 00", "description": "Densified wood, in blocks, plates, strips, profiles"},
        {"code": "4414 00", "description": "Wooden frames for paintings, photographs, mirrors"},
        {"code": "4415", "description": "Packing cases, boxes, crates, drums of wood"},
        {"code": "4416 00 00", "description": "Casks, barrels, vats, tubs of wood"},
        {"code": "4417 00", "description": "Tools, tool bodies, handles, broom bodies of wood"},
        {"code": "4418", "description": "Builders joinery and carpentry of wood"},
        {"code": "4419", "description": "Tableware and kitchenware, of wood"},
        {"code": "4420", "description": "Wood marquetry and inlaid wood; caskets, cases"},
        {"code": "4421", "description": "Other articles of wood (clothes hangers, spools)"},
        {"code": "4501", "description": "Natural cork, raw or simply prepared"},
        {"code": "4502 00 00", "description": "Natural cork, debacked or roughly squared"},
        {"code": "4503", "description": "Articles of natural cork"},
        {"code": "4504", "description": "Agglomerated cork and articles thereof"},
        {"code": "4701 00", "description": "Mechanical wood pulp"},
        {"code": "4702 00 00", "description": "Chemical wood pulp, dissolving grades"},
        {"code": "4703", "description": "Chemical wood pulp, soda or sulphate (not dissolving)"},
        {"code": "4704", "description": "Chemical wood pulp, sulphite (not dissolving)"},
        {"code": "4705 00 00", "description": "Wood pulp from mechanical and chemical combined"},
        {"code": "4706", "description": "Pulps of fibres from recovered (waste, scrap) paper"},
        {"code": "4707", "description": "Recovered (waste and scrap) paper or paperboard"},
        {"code": "4801 00 00", "description": "Newsprint, in rolls or sheets"},
        {"code": "4802", "description": "Uncoated paper for writing, printing (not newsprint)"},
        {"code": "4803 00", "description": "Toilet or facial tissue stock, towel stock"},
        {"code": "4804", "description": "Uncoated kraft paper and paperboard"},
        {"code": "4805", "description": "Other uncoated paper and paperboard"},
        {"code": "4806", "description": "Vegetable parchment, greaseproof, tracing papers"},
        {"code": "4807 00", "description": "Composite paper and paperboard (not surface-coated)"},
        {"code": "4808", "description": "Paper and paperboard, corrugated, creped, embossed"},
        {"code": "4809", "description": "Carbon paper, self-copy paper and transfer papers"},
        {"code": "4810", "description": "Paper and paperboard, coated with kaolin"},
        {"code": "4811", "description": "Paper, paperboard, cellulose wadding, coated"},
        {"code": "4812 00 00", "description": "Filter blocks, slabs, plates of paper pulp"},
        {"code": "4813", "description": "Cigarette paper"},
        {"code": "4814", "description": "Wallpaper and similar wall coverings of paper"},
        {"code": "4816", "description": "Carbon paper, self-copy paper in rolls/sheets"},
        {"code": "4817", "description": "Envelopes, letter cards, plain postcards of paper"},
        {"code": "4818", "description": "Toilet paper, tissues, towels, napkins of paper"},
        {"code": "4819", "description": "Cartons, boxes, cases, bags of paper/paperboard"},
        {"code": "4820", "description": "Registers, account books, notebooks, diaries"},
        {"code": "4821", "description": "Paper or paperboard labels of all kinds"},
        {"code": "4822", "description": "Bobbins, spools, cops of paper pulp/paperboard"},
        {"code": "4823", "description": "Other paper, paperboard, cellulose wadding, articles"},
        {"code": "4901", "description": "Printed books, brochures, leaflets"},
        {"code": "4902", "description": "Newspapers, journals and periodicals"},
        {"code": "4903 00 00", "description": "Children's picture, drawing or colouring books"},
        {"code": "4904 00 00", "description": "Music, printed or in manuscript"},
        {"code": "4905", "description": "Maps and hydrographic or similar charts"},
        {"code": "4906 00 00", "description": "Plans and drawings for architectural purposes"},
        {"code": "4907 00", "description": "Unused postage, revenue or similar stamps"},
        {"code": "4908", "description": "Transfers (decalcomanias)"},
        {"code": "4909 00", "description": "Printed or illustrated postcards"},
        {"code": "4910 00 00", "description": "Calendars of any kind, printed"},
        {"code": "4911", "description": "Other printed matter, including pictures, designs"},
        {"code": "9401", "description": "Seats (wooden), whether or not convertible into beds"},
        {"code": "9403", "description": "Other furniture and parts thereof (wooden)"},
        {"code": "9406 10", "description": "Prefabricated buildings of wood"},
    ],
}

# Certification scheme registries - API/URL references
CERTIFICATION_REGISTRIES: Dict[str, Dict[str, str]] = {
    "FSC": {
        "name": "Forest Stewardship Council",
        "url": "https://info.fsc.org/",
        "api": "https://info.fsc.org/certificate-api",
        "commodities": "wood, rubber",
    },
    "PEFC": {
        "name": "Programme for the Endorsement of Forest Certification",
        "url": "https://www.pefc.org/",
        "api": "https://www.pefc.org/find-certified",
        "commodities": "wood, rubber",
    },
    "RSPO": {
        "name": "Roundtable on Sustainable Palm Oil",
        "url": "https://www.rspo.org/",
        "api": "https://rspo.org/members/all",
        "commodities": "oil_palm",
    },
    "ISCC": {
        "name": "International Sustainability and Carbon Certification",
        "url": "https://www.iscc-system.org/",
        "api": "https://www.iscc-system.org/certificates/",
        "commodities": "oil_palm, soya",
    },
    "RAINFOREST_ALLIANCE": {
        "name": "Rainforest Alliance Certified",
        "url": "https://www.rainforest-alliance.org/",
        "api": "https://www.rainforest-alliance.org/find-certified",
        "commodities": "cocoa, coffee",
    },
    "UTZ": {
        "name": "UTZ (merged with Rainforest Alliance)",
        "url": "https://www.rainforest-alliance.org/",
        "api": "https://www.rainforest-alliance.org/find-certified",
        "commodities": "cocoa, coffee",
    },
    "FAIRTRADE": {
        "name": "Fairtrade International",
        "url": "https://www.fairtrade.net/",
        "api": "https://www.fairtrade.net/product-search",
        "commodities": "cocoa, coffee",
    },
    "RTRS": {
        "name": "Round Table on Responsible Soy",
        "url": "https://responsiblesoy.org/",
        "api": "https://responsiblesoy.org/certified-companies",
        "commodities": "soya",
    },
    "PROTERRA": {
        "name": "ProTerra Foundation",
        "url": "https://www.proterrafoundation.org/",
        "api": "https://www.proterrafoundation.org/certified",
        "commodities": "soya",
    },
    "FOUR_C": {
        "name": "4C (Common Code for the Coffee Community)",
        "url": "https://www.4c-services.org/",
        "api": "https://www.4c-services.org/about/licence-holders/",
        "commodities": "coffee",
    },
    "SFI": {
        "name": "Sustainable Forestry Initiative",
        "url": "https://forests.org/",
        "api": "https://forests.org/find-certified",
        "commodities": "wood",
    },
    "MSPO": {
        "name": "Malaysian Sustainable Palm Oil",
        "url": "https://www.mpocc.org.my/",
        "api": "https://www.mpocc.org.my/mspo-certification",
        "commodities": "oil_palm",
    },
    "ISPO": {
        "name": "Indonesian Sustainable Palm Oil",
        "url": "https://ispo.go.id/",
        "api": "https://ispo.go.id/certified",
        "commodities": "oil_palm",
    },
}

# High-risk countries list (ISO-3166-1 alpha-3)
HIGH_RISK_COUNTRIES: List[str] = [
    "BRA", "IDN", "COD", "BOL", "PRY", "MYS", "PNG", "COG",
    "MMR", "CMR", "CIV", "GHA", "NGA", "LAO", "KHM", "NIC",
    "SLE", "LBR", "GAB", "VEN", "GUY", "SUR", "GTM", "HND",
    "TZA", "MOZ", "MDG", "AGO",
]

# Low-risk countries list (EU-27 + EEA/EFTA + select OECD)
LOW_RISK_COUNTRIES: List[str] = [
    # EU-27
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
    "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK",
    "SVN", "ESP", "SWE",
    # EEA / EFTA
    "ISL", "LIE", "NOR", "CHE",
    # Other OECD with strong governance
    "GBR", "CAN", "AUS", "NZL", "JPN", "KOR", "SGP",
]

# Country risk database with Article 29 benchmarking details
COUNTRY_RISK_DATABASE: Dict[str, Dict[str, Any]] = {
    # HIGH RISK
    "BRA": {"name": "Brazil", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 59.0, "annual_deforestation_rate": 0.45, "governance_score": 0.42},
    "IDN": {"name": "Indonesia", "benchmark": "HIGH_RISK", "region": "Southeast Asia", "forest_cover_pct": 49.1, "annual_deforestation_rate": 0.75, "governance_score": 0.40},
    "COD": {"name": "Democratic Republic of Congo", "benchmark": "HIGH_RISK", "region": "Central Africa", "forest_cover_pct": 67.3, "annual_deforestation_rate": 0.40, "governance_score": 0.18},
    "BOL": {"name": "Bolivia", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 50.6, "annual_deforestation_rate": 0.50, "governance_score": 0.35},
    "PRY": {"name": "Paraguay", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 38.6, "annual_deforestation_rate": 0.80, "governance_score": 0.38},
    "MYS": {"name": "Malaysia", "benchmark": "HIGH_RISK", "region": "Southeast Asia", "forest_cover_pct": 54.6, "annual_deforestation_rate": 0.40, "governance_score": 0.52},
    "PNG": {"name": "Papua New Guinea", "benchmark": "HIGH_RISK", "region": "Oceania", "forest_cover_pct": 74.1, "annual_deforestation_rate": 0.35, "governance_score": 0.25},
    "COG": {"name": "Republic of Congo", "benchmark": "HIGH_RISK", "region": "Central Africa", "forest_cover_pct": 65.4, "annual_deforestation_rate": 0.20, "governance_score": 0.22},
    "MMR": {"name": "Myanmar", "benchmark": "HIGH_RISK", "region": "Southeast Asia", "forest_cover_pct": 42.9, "annual_deforestation_rate": 0.85, "governance_score": 0.15},
    "CMR": {"name": "Cameroon", "benchmark": "HIGH_RISK", "region": "Central Africa", "forest_cover_pct": 39.8, "annual_deforestation_rate": 0.30, "governance_score": 0.28},
    "CIV": {"name": "Cote d'Ivoire", "benchmark": "HIGH_RISK", "region": "West Africa", "forest_cover_pct": 8.9, "annual_deforestation_rate": 2.60, "governance_score": 0.30},
    "GHA": {"name": "Ghana", "benchmark": "HIGH_RISK", "region": "West Africa", "forest_cover_pct": 21.0, "annual_deforestation_rate": 1.20, "governance_score": 0.48},
    "NGA": {"name": "Nigeria", "benchmark": "HIGH_RISK", "region": "West Africa", "forest_cover_pct": 7.2, "annual_deforestation_rate": 3.70, "governance_score": 0.32},
    "LAO": {"name": "Laos", "benchmark": "HIGH_RISK", "region": "Southeast Asia", "forest_cover_pct": 58.0, "annual_deforestation_rate": 0.70, "governance_score": 0.28},
    "KHM": {"name": "Cambodia", "benchmark": "HIGH_RISK", "region": "Southeast Asia", "forest_cover_pct": 46.3, "annual_deforestation_rate": 1.30, "governance_score": 0.25},
    "NIC": {"name": "Nicaragua", "benchmark": "HIGH_RISK", "region": "Central America", "forest_cover_pct": 25.9, "annual_deforestation_rate": 1.10, "governance_score": 0.28},
    "SLE": {"name": "Sierra Leone", "benchmark": "HIGH_RISK", "region": "West Africa", "forest_cover_pct": 24.5, "annual_deforestation_rate": 0.90, "governance_score": 0.22},
    "LBR": {"name": "Liberia", "benchmark": "HIGH_RISK", "region": "West Africa", "forest_cover_pct": 43.4, "annual_deforestation_rate": 0.60, "governance_score": 0.20},
    "GAB": {"name": "Gabon", "benchmark": "HIGH_RISK", "region": "Central Africa", "forest_cover_pct": 88.0, "annual_deforestation_rate": 0.10, "governance_score": 0.32},
    "VEN": {"name": "Venezuela", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 52.3, "annual_deforestation_rate": 0.40, "governance_score": 0.15},
    "GUY": {"name": "Guyana", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 84.0, "annual_deforestation_rate": 0.05, "governance_score": 0.35},
    "SUR": {"name": "Suriname", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 93.0, "annual_deforestation_rate": 0.05, "governance_score": 0.30},
    "GTM": {"name": "Guatemala", "benchmark": "HIGH_RISK", "region": "Central America", "forest_cover_pct": 33.0, "annual_deforestation_rate": 1.00, "governance_score": 0.32},
    "HND": {"name": "Honduras", "benchmark": "HIGH_RISK", "region": "Central America", "forest_cover_pct": 41.0, "annual_deforestation_rate": 1.50, "governance_score": 0.30},
    "TZA": {"name": "Tanzania", "benchmark": "HIGH_RISK", "region": "East Africa", "forest_cover_pct": 50.6, "annual_deforestation_rate": 0.80, "governance_score": 0.35},
    "MOZ": {"name": "Mozambique", "benchmark": "HIGH_RISK", "region": "East Africa", "forest_cover_pct": 36.4, "annual_deforestation_rate": 0.40, "governance_score": 0.25},
    "MDG": {"name": "Madagascar", "benchmark": "HIGH_RISK", "region": "East Africa", "forest_cover_pct": 21.3, "annual_deforestation_rate": 1.10, "governance_score": 0.22},
    "AGO": {"name": "Angola", "benchmark": "HIGH_RISK", "region": "Southern Africa", "forest_cover_pct": 46.0, "annual_deforestation_rate": 0.20, "governance_score": 0.20},
    # LOW RISK - EU-27
    "AUT": {"name": "Austria", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 47.3, "annual_deforestation_rate": 0.0, "governance_score": 0.92},
    "BEL": {"name": "Belgium", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 22.8, "annual_deforestation_rate": 0.0, "governance_score": 0.89},
    "BGR": {"name": "Bulgaria", "benchmark": "LOW_RISK", "region": "Eastern Europe", "forest_cover_pct": 35.5, "annual_deforestation_rate": 0.0, "governance_score": 0.70},
    "HRV": {"name": "Croatia", "benchmark": "LOW_RISK", "region": "Southern Europe", "forest_cover_pct": 34.4, "annual_deforestation_rate": 0.0, "governance_score": 0.72},
    "CYP": {"name": "Cyprus", "benchmark": "LOW_RISK", "region": "Southern Europe", "forest_cover_pct": 18.7, "annual_deforestation_rate": 0.0, "governance_score": 0.78},
    "CZE": {"name": "Czech Republic", "benchmark": "LOW_RISK", "region": "Central Europe", "forest_cover_pct": 34.7, "annual_deforestation_rate": 0.0, "governance_score": 0.82},
    "DNK": {"name": "Denmark", "benchmark": "LOW_RISK", "region": "Northern Europe", "forest_cover_pct": 14.7, "annual_deforestation_rate": 0.0, "governance_score": 0.95},
    "EST": {"name": "Estonia", "benchmark": "LOW_RISK", "region": "Northern Europe", "forest_cover_pct": 52.1, "annual_deforestation_rate": 0.0, "governance_score": 0.85},
    "FIN": {"name": "Finland", "benchmark": "LOW_RISK", "region": "Northern Europe", "forest_cover_pct": 73.1, "annual_deforestation_rate": 0.0, "governance_score": 0.96},
    "FRA": {"name": "France", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 31.4, "annual_deforestation_rate": 0.0, "governance_score": 0.88},
    "DEU": {"name": "Germany", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 32.8, "annual_deforestation_rate": 0.0, "governance_score": 0.93},
    "GRC": {"name": "Greece", "benchmark": "LOW_RISK", "region": "Southern Europe", "forest_cover_pct": 31.5, "annual_deforestation_rate": 0.0, "governance_score": 0.75},
    "HUN": {"name": "Hungary", "benchmark": "LOW_RISK", "region": "Central Europe", "forest_cover_pct": 22.9, "annual_deforestation_rate": 0.0, "governance_score": 0.73},
    "IRL": {"name": "Ireland", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 11.0, "annual_deforestation_rate": 0.0, "governance_score": 0.91},
    "ITA": {"name": "Italy", "benchmark": "LOW_RISK", "region": "Southern Europe", "forest_cover_pct": 32.0, "annual_deforestation_rate": 0.0, "governance_score": 0.82},
    "LVA": {"name": "Latvia", "benchmark": "LOW_RISK", "region": "Northern Europe", "forest_cover_pct": 54.1, "annual_deforestation_rate": 0.0, "governance_score": 0.80},
    "LTU": {"name": "Lithuania", "benchmark": "LOW_RISK", "region": "Northern Europe", "forest_cover_pct": 34.8, "annual_deforestation_rate": 0.0, "governance_score": 0.80},
    "LUX": {"name": "Luxembourg", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 34.0, "annual_deforestation_rate": 0.0, "governance_score": 0.94},
    "MLT": {"name": "Malta", "benchmark": "LOW_RISK", "region": "Southern Europe", "forest_cover_pct": 1.1, "annual_deforestation_rate": 0.0, "governance_score": 0.78},
    "NLD": {"name": "Netherlands", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 11.2, "annual_deforestation_rate": 0.0, "governance_score": 0.93},
    "POL": {"name": "Poland", "benchmark": "LOW_RISK", "region": "Central Europe", "forest_cover_pct": 30.9, "annual_deforestation_rate": 0.0, "governance_score": 0.78},
    "PRT": {"name": "Portugal", "benchmark": "LOW_RISK", "region": "Southern Europe", "forest_cover_pct": 36.2, "annual_deforestation_rate": 0.0, "governance_score": 0.82},
    "ROU": {"name": "Romania", "benchmark": "LOW_RISK", "region": "Eastern Europe", "forest_cover_pct": 30.0, "annual_deforestation_rate": 0.01, "governance_score": 0.72},
    "SVK": {"name": "Slovakia", "benchmark": "LOW_RISK", "region": "Central Europe", "forest_cover_pct": 40.1, "annual_deforestation_rate": 0.0, "governance_score": 0.78},
    "SVN": {"name": "Slovenia", "benchmark": "LOW_RISK", "region": "Central Europe", "forest_cover_pct": 62.0, "annual_deforestation_rate": 0.0, "governance_score": 0.82},
    "ESP": {"name": "Spain", "benchmark": "LOW_RISK", "region": "Southern Europe", "forest_cover_pct": 37.4, "annual_deforestation_rate": 0.0, "governance_score": 0.85},
    "SWE": {"name": "Sweden", "benchmark": "LOW_RISK", "region": "Northern Europe", "forest_cover_pct": 68.9, "annual_deforestation_rate": 0.0, "governance_score": 0.96},
    # LOW RISK - EEA/EFTA
    "ISL": {"name": "Iceland", "benchmark": "LOW_RISK", "region": "Northern Europe", "forest_cover_pct": 0.5, "annual_deforestation_rate": 0.0, "governance_score": 0.93},
    "LIE": {"name": "Liechtenstein", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 43.1, "annual_deforestation_rate": 0.0, "governance_score": 0.92},
    "NOR": {"name": "Norway", "benchmark": "LOW_RISK", "region": "Northern Europe", "forest_cover_pct": 37.4, "annual_deforestation_rate": 0.0, "governance_score": 0.97},
    "CHE": {"name": "Switzerland", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 31.7, "annual_deforestation_rate": 0.0, "governance_score": 0.96},
    # LOW RISK - OECD
    "GBR": {"name": "United Kingdom", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 13.0, "annual_deforestation_rate": 0.0, "governance_score": 0.90},
    "CAN": {"name": "Canada", "benchmark": "LOW_RISK", "region": "North America", "forest_cover_pct": 38.7, "annual_deforestation_rate": 0.01, "governance_score": 0.92},
    "AUS": {"name": "Australia", "benchmark": "LOW_RISK", "region": "Oceania", "forest_cover_pct": 17.1, "annual_deforestation_rate": 0.02, "governance_score": 0.91},
    "NZL": {"name": "New Zealand", "benchmark": "LOW_RISK", "region": "Oceania", "forest_cover_pct": 38.6, "annual_deforestation_rate": 0.0, "governance_score": 0.94},
    "JPN": {"name": "Japan", "benchmark": "LOW_RISK", "region": "East Asia", "forest_cover_pct": 68.4, "annual_deforestation_rate": 0.0, "governance_score": 0.90},
    "KOR": {"name": "South Korea", "benchmark": "LOW_RISK", "region": "East Asia", "forest_cover_pct": 63.7, "annual_deforestation_rate": 0.0, "governance_score": 0.85},
    "SGP": {"name": "Singapore", "benchmark": "LOW_RISK", "region": "Southeast Asia", "forest_cover_pct": 23.1, "annual_deforestation_rate": 0.0, "governance_score": 0.95},
    # STANDARD RISK - Americas
    "USA": {"name": "United States", "benchmark": "STANDARD_RISK", "region": "North America", "forest_cover_pct": 33.9, "annual_deforestation_rate": 0.01, "governance_score": 0.85},
    "MEX": {"name": "Mexico", "benchmark": "STANDARD_RISK", "region": "Central America", "forest_cover_pct": 33.6, "annual_deforestation_rate": 0.20, "governance_score": 0.50},
    "ARG": {"name": "Argentina", "benchmark": "STANDARD_RISK", "region": "South America", "forest_cover_pct": 10.7, "annual_deforestation_rate": 0.50, "governance_score": 0.52},
    "CHL": {"name": "Chile", "benchmark": "STANDARD_RISK", "region": "South America", "forest_cover_pct": 24.5, "annual_deforestation_rate": 0.05, "governance_score": 0.72},
    "COL": {"name": "Colombia", "benchmark": "STANDARD_RISK", "region": "South America", "forest_cover_pct": 52.7, "annual_deforestation_rate": 0.35, "governance_score": 0.48},
    "PER": {"name": "Peru", "benchmark": "STANDARD_RISK", "region": "South America", "forest_cover_pct": 57.8, "annual_deforestation_rate": 0.15, "governance_score": 0.45},
    "ECU": {"name": "Ecuador", "benchmark": "STANDARD_RISK", "region": "South America", "forest_cover_pct": 48.6, "annual_deforestation_rate": 0.25, "governance_score": 0.42},
    "URY": {"name": "Uruguay", "benchmark": "STANDARD_RISK", "region": "South America", "forest_cover_pct": 11.3, "annual_deforestation_rate": 0.0, "governance_score": 0.72},
    "CRI": {"name": "Costa Rica", "benchmark": "STANDARD_RISK", "region": "Central America", "forest_cover_pct": 54.0, "annual_deforestation_rate": 0.0, "governance_score": 0.68},
    "PAN": {"name": "Panama", "benchmark": "STANDARD_RISK", "region": "Central America", "forest_cover_pct": 62.1, "annual_deforestation_rate": 0.10, "governance_score": 0.55},
    "SLV": {"name": "El Salvador", "benchmark": "STANDARD_RISK", "region": "Central America", "forest_cover_pct": 12.4, "annual_deforestation_rate": 0.05, "governance_score": 0.45},
    "DOM": {"name": "Dominican Republic", "benchmark": "STANDARD_RISK", "region": "Caribbean", "forest_cover_pct": 43.6, "annual_deforestation_rate": 0.10, "governance_score": 0.50},
    # STANDARD RISK - Asia
    "CHN": {"name": "China", "benchmark": "STANDARD_RISK", "region": "East Asia", "forest_cover_pct": 23.3, "annual_deforestation_rate": 0.0, "governance_score": 0.55},
    "IND": {"name": "India", "benchmark": "STANDARD_RISK", "region": "South Asia", "forest_cover_pct": 24.3, "annual_deforestation_rate": 0.0, "governance_score": 0.52},
    "THA": {"name": "Thailand", "benchmark": "STANDARD_RISK", "region": "Southeast Asia", "forest_cover_pct": 31.6, "annual_deforestation_rate": 0.15, "governance_score": 0.48},
    "VNM": {"name": "Vietnam", "benchmark": "STANDARD_RISK", "region": "Southeast Asia", "forest_cover_pct": 42.0, "annual_deforestation_rate": 0.05, "governance_score": 0.45},
    "PHL": {"name": "Philippines", "benchmark": "STANDARD_RISK", "region": "Southeast Asia", "forest_cover_pct": 24.1, "annual_deforestation_rate": 0.20, "governance_score": 0.45},
    "BGD": {"name": "Bangladesh", "benchmark": "STANDARD_RISK", "region": "South Asia", "forest_cover_pct": 11.2, "annual_deforestation_rate": 0.15, "governance_score": 0.38},
    "PAK": {"name": "Pakistan", "benchmark": "STANDARD_RISK", "region": "South Asia", "forest_cover_pct": 2.2, "annual_deforestation_rate": 0.10, "governance_score": 0.35},
    "LKA": {"name": "Sri Lanka", "benchmark": "STANDARD_RISK", "region": "South Asia", "forest_cover_pct": 29.7, "annual_deforestation_rate": 0.20, "governance_score": 0.45},
    "TUR": {"name": "Turkey", "benchmark": "STANDARD_RISK", "region": "Western Asia", "forest_cover_pct": 28.6, "annual_deforestation_rate": 0.0, "governance_score": 0.55},
    "ISR": {"name": "Israel", "benchmark": "STANDARD_RISK", "region": "Western Asia", "forest_cover_pct": 7.6, "annual_deforestation_rate": 0.0, "governance_score": 0.78},
    "GEO": {"name": "Georgia", "benchmark": "STANDARD_RISK", "region": "Western Asia", "forest_cover_pct": 40.6, "annual_deforestation_rate": 0.05, "governance_score": 0.58},
    "KAZ": {"name": "Kazakhstan", "benchmark": "STANDARD_RISK", "region": "Central Asia", "forest_cover_pct": 1.2, "annual_deforestation_rate": 0.0, "governance_score": 0.45},
    # STANDARD RISK - Africa
    "ZAF": {"name": "South Africa", "benchmark": "STANDARD_RISK", "region": "Southern Africa", "forest_cover_pct": 7.6, "annual_deforestation_rate": 0.05, "governance_score": 0.62},
    "KEN": {"name": "Kenya", "benchmark": "STANDARD_RISK", "region": "East Africa", "forest_cover_pct": 6.1, "annual_deforestation_rate": 0.30, "governance_score": 0.42},
    "ETH": {"name": "Ethiopia", "benchmark": "STANDARD_RISK", "region": "East Africa", "forest_cover_pct": 15.5, "annual_deforestation_rate": 0.50, "governance_score": 0.35},
    "UGA": {"name": "Uganda", "benchmark": "STANDARD_RISK", "region": "East Africa", "forest_cover_pct": 8.9, "annual_deforestation_rate": 1.00, "governance_score": 0.38},
    "RWA": {"name": "Rwanda", "benchmark": "STANDARD_RISK", "region": "East Africa", "forest_cover_pct": 18.4, "annual_deforestation_rate": 0.10, "governance_score": 0.55},
    "ZMB": {"name": "Zambia", "benchmark": "STANDARD_RISK", "region": "Southern Africa", "forest_cover_pct": 60.0, "annual_deforestation_rate": 0.50, "governance_score": 0.40},
    "SEN": {"name": "Senegal", "benchmark": "STANDARD_RISK", "region": "West Africa", "forest_cover_pct": 42.4, "annual_deforestation_rate": 0.10, "governance_score": 0.52},
    "EGY": {"name": "Egypt", "benchmark": "STANDARD_RISK", "region": "North Africa", "forest_cover_pct": 0.1, "annual_deforestation_rate": 0.0, "governance_score": 0.45},
    "MAR": {"name": "Morocco", "benchmark": "STANDARD_RISK", "region": "North Africa", "forest_cover_pct": 12.6, "annual_deforestation_rate": 0.0, "governance_score": 0.52},
    "TUN": {"name": "Tunisia", "benchmark": "STANDARD_RISK", "region": "North Africa", "forest_cover_pct": 6.6, "annual_deforestation_rate": 0.0, "governance_score": 0.55},
    # STANDARD RISK - Middle East
    "SAU": {"name": "Saudi Arabia", "benchmark": "STANDARD_RISK", "region": "Western Asia", "forest_cover_pct": 0.5, "annual_deforestation_rate": 0.0, "governance_score": 0.55},
    "ARE": {"name": "United Arab Emirates", "benchmark": "STANDARD_RISK", "region": "Western Asia", "forest_cover_pct": 3.8, "annual_deforestation_rate": 0.0, "governance_score": 0.72},
    # STANDARD RISK - Europe (non-EU)
    "UKR": {"name": "Ukraine", "benchmark": "STANDARD_RISK", "region": "Eastern Europe", "forest_cover_pct": 16.7, "annual_deforestation_rate": 0.02, "governance_score": 0.48},
    "RUS": {"name": "Russia", "benchmark": "STANDARD_RISK", "region": "Eastern Europe", "forest_cover_pct": 49.8, "annual_deforestation_rate": 0.02, "governance_score": 0.35},
    "SRB": {"name": "Serbia", "benchmark": "STANDARD_RISK", "region": "Southern Europe", "forest_cover_pct": 31.1, "annual_deforestation_rate": 0.0, "governance_score": 0.58},
    "BIH": {"name": "Bosnia and Herzegovina", "benchmark": "STANDARD_RISK", "region": "Southern Europe", "forest_cover_pct": 42.7, "annual_deforestation_rate": 0.0, "governance_score": 0.45},
    "ALB": {"name": "Albania", "benchmark": "STANDARD_RISK", "region": "Southern Europe", "forest_cover_pct": 28.8, "annual_deforestation_rate": 0.05, "governance_score": 0.48},
    # STANDARD RISK - Oceania
    "FJI": {"name": "Fiji", "benchmark": "STANDARD_RISK", "region": "Oceania", "forest_cover_pct": 55.7, "annual_deforestation_rate": 0.10, "governance_score": 0.52},
    "SLB": {"name": "Solomon Islands", "benchmark": "STANDARD_RISK", "region": "Oceania", "forest_cover_pct": 77.6, "annual_deforestation_rate": 0.30, "governance_score": 0.30},
    "TLS": {"name": "Timor-Leste", "benchmark": "STANDARD_RISK", "region": "Southeast Asia", "forest_cover_pct": 46.1, "annual_deforestation_rate": 0.20, "governance_score": 0.32},
    "BRN": {"name": "Brunei", "benchmark": "STANDARD_RISK", "region": "Southeast Asia", "forest_cover_pct": 72.1, "annual_deforestation_rate": 0.05, "governance_score": 0.62},
}


# =============================================================================
# Pydantic Sub-Config Models (12 models)
# =============================================================================


class OperatorConfig(BaseModel):
    """Operator/trader identification per EUDR Article 2.

    Defines the company placing products on the EU market or trading them.
    The operator_type determines obligation level (operators have full DD,
    traders may use simplified DD for SME traders).
    """

    company_name: str = Field(
        "",
        description="Legal name of the operator or trader",
    )
    eori_number: str = Field(
        "",
        description="Economic Operators Registration and Identification number",
    )
    registration_country: str = Field(
        "DEU",
        description="ISO-3166-1 alpha-3 country code of registration",
    )
    operator_type: OperatorType = Field(
        OperatorType.OPERATOR,
        description="Whether entity is an operator or trader per Article 2",
    )
    company_size: CompanySize = Field(
        CompanySize.LARGE,
        description="Company size classification affecting DD obligations",
    )
    contact_email: str = Field(
        "",
        description="Primary contact email for EUDR compliance",
    )
    vat_number: str = Field(
        "",
        description="EU VAT identification number",
    )

    @field_validator("registration_country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is 3-letter ISO format."""
        if len(v) != 3 or not v.isalpha() or not v.isupper():
            raise ValueError(
                f"Country code must be 3-letter uppercase ISO-3166-1 alpha-3: {v}"
            )
        return v


class CommodityConfig(BaseModel):
    """Configuration for a single EUDR commodity.

    Controls which commodity is active, its CN codes, high-risk origins,
    and applicable certification schemes.
    """

    commodity_type: EUDRCommodity = Field(
        ...,
        description="EUDR commodity type",
    )
    enabled: bool = Field(
        True,
        description="Whether this commodity is enabled for compliance tracking",
    )
    cn_codes: List[str] = Field(
        default_factory=list,
        description="Annex I CN codes applicable to this commodity",
    )
    high_risk_origins: List[str] = Field(
        default_factory=list,
        description="ISO-3166-1 alpha-3 codes of high-risk origin countries",
    )
    certification_schemes: List[CertificationScheme] = Field(
        default_factory=list,
        description="Applicable voluntary certification schemes",
    )
    annual_volume_tonnes: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated annual import volume in metric tonnes",
    )
    priority: int = Field(
        1,
        ge=1,
        le=10,
        description="Priority ranking for compliance focus (1=highest)",
    )


class GeolocationConfig(BaseModel):
    """Geolocation verification configuration per Article 9(1)(d).

    Controls coordinate precision, polygon handling, CRS settings,
    and batch processing parameters for plot verification.
    """

    coordinate_precision: int = Field(
        6,
        ge=4,
        le=10,
        description="Decimal places for coordinate precision (6 = ~0.11m)",
    )
    polygon_max_vertices: int = Field(
        10000,
        ge=100,
        le=100000,
        description="Maximum vertices allowed per polygon boundary",
    )
    polygon_area_threshold_ha: float = Field(
        4.0,
        ge=0,
        description="Hectare threshold above which polygon is required (Article 9)",
    )
    area_unit: AreaUnit = Field(
        AreaUnit.HECTARES,
        description="Default area measurement unit",
    )
    crs: str = Field(
        "EPSG:4326",
        description="Coordinate Reference System (WGS 84 default)",
    )
    allowed_crs_list: List[str] = Field(
        default_factory=lambda: ["EPSG:4326", "EPSG:3857", "EPSG:32601"],
        description="Allowed coordinate reference systems for input",
    )
    batch_size: int = Field(
        500,
        ge=10,
        le=10000,
        description="Batch size for bulk coordinate verification",
    )
    overlap_detection_enabled: bool = Field(
        True,
        description="Enable detection of overlapping plot boundaries",
    )
    satellite_overlay_enabled: bool = Field(
        True,
        description="Enable satellite imagery overlay for visual verification",
    )
    coordinate_format: CoordinateFormat = Field(
        CoordinateFormat.DECIMAL_DEGREES,
        description="Default input coordinate format",
    )


class RiskAssessmentConfig(BaseModel):
    """Risk assessment weights and thresholds per Article 10.

    Defines the weighted scoring model for composite risk calculation
    and the threshold values for risk level classification.
    """

    country_weight: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Weight for country risk in composite score (Article 29)",
    )
    supplier_weight: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Weight for supplier risk in composite score",
    )
    commodity_weight: float = Field(
        0.20,
        ge=0.0,
        le=1.0,
        description="Weight for commodity-specific risk factors",
    )
    document_weight: float = Field(
        0.20,
        ge=0.0,
        le=1.0,
        description="Weight for documentation completeness and quality",
    )
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "low_max": 25.0,
            "medium_max": 50.0,
            "high_max": 75.0,
            "critical_min": 75.0,
        },
        description="Score thresholds for risk level classification",
    )
    auto_escalation_enabled: bool = Field(
        True,
        description="Auto-escalate when risk exceeds threshold",
    )
    reassessment_interval_days: int = Field(
        90,
        ge=30,
        le=365,
        description="Days between mandatory risk reassessments",
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "RiskAssessmentConfig":
        """Validate that all weights sum to 1.0."""
        total = (
            self.country_weight
            + self.supplier_weight
            + self.commodity_weight
            + self.document_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Risk assessment weights must sum to 1.0, got {total:.2f}"
            )
        return self


class DDSConfig(BaseModel):
    """Due Diligence Statement configuration per Articles 4 and 13.

    Controls DDS generation type, review requirements, template versions,
    and auto-generation settings.
    """

    dds_type: DDSType = Field(
        DDSType.STANDARD,
        description="DDS type: STANDARD (Article 4) or SIMPLIFIED (Article 13)",
    )
    auto_generate: bool = Field(
        False,
        description="Auto-generate DDS when all data is complete",
    )
    review_required: bool = Field(
        True,
        description="Require human review before DDS submission",
    )
    template_version: str = Field(
        "1.0",
        description="DDS template format version",
    )
    annex_ii_version: str = Field(
        "2023",
        description="Annex II specification version for DDS content",
    )
    retention_years: int = Field(
        5,
        ge=5,
        le=10,
        description="Years to retain DDS records (minimum 5 per Article 4(6))",
    )
    include_supporting_documents: bool = Field(
        True,
        description="Include supporting evidence package with DDS",
    )
    multi_commodity_dds: bool = Field(
        False,
        description="Allow single DDS to cover multiple commodities",
    )


class EUISConfig(BaseModel):
    """EU Information System configuration per Article 33.

    Controls connection to the EU Information System for DDS submission,
    reference number generation, and competent authority queries.
    """

    api_url: str = Field(
        "https://eudr-euis.ec.europa.eu/api/v1",
        description="EU Information System API endpoint",
    )
    sandbox_mode: bool = Field(
        True,
        description="Use sandbox/test environment instead of production",
    )
    sandbox_api_url: str = Field(
        "https://eudr-euis-sandbox.ec.europa.eu/api/v1",
        description="Sandbox API endpoint for testing",
    )
    auth_type: AuthType = Field(
        AuthType.OAUTH2,
        description="Authentication method for EU IS",
    )
    submission_retry_count: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of retry attempts for DDS submission",
    )
    submission_retry_delay_seconds: int = Field(
        30,
        ge=5,
        le=300,
        description="Delay between retry attempts in seconds",
    )
    polling_interval_seconds: int = Field(
        60,
        ge=10,
        le=600,
        description="Polling interval for submission status checks",
    )
    timeout_seconds: int = Field(
        120,
        ge=30,
        le=600,
        description="HTTP request timeout for EU IS API calls",
    )
    batch_submission_enabled: bool = Field(
        False,
        description="Enable batch DDS submission mode",
    )


class SupplyChainConfig(BaseModel):
    """Supply chain traceability configuration.

    Controls the depth of supply chain mapping, chain of custody model,
    and traceability requirements for EUDR compliance.
    """

    max_tier_depth: int = Field(
        2,
        ge=1,
        le=10,
        description="Maximum supply chain tier depth to trace",
    )
    chain_of_custody_model: ChainOfCustodyModel = Field(
        ChainOfCustodyModel.MASS_BALANCE,
        description="Default chain of custody model",
    )
    trace_level: str = Field(
        "shipment",
        description="Traceability granularity: shipment, batch, lot, item",
    )
    require_plot_linkage: bool = Field(
        True,
        description="Require each shipment to be linked to a production plot",
    )
    mixed_shipment_handling: str = Field(
        "split",
        description="How to handle mixed-commodity shipments: split, reject, flag",
    )
    upstream_data_sharing: bool = Field(
        True,
        description="Enable upstream data sharing requests to suppliers",
    )


class SupplierConfig(BaseModel):
    """Supplier management configuration.

    Controls supplier onboarding, engagement, data collection,
    and compliance tracking settings.
    """

    bulk_import_limit: int = Field(
        500,
        ge=10,
        le=10000,
        description="Maximum suppliers per bulk import operation",
    )
    engagement_auto_reminders: bool = Field(
        True,
        description="Send automatic reminders for pending supplier data",
    )
    reminder_interval_days: int = Field(
        14,
        ge=7,
        le=90,
        description="Days between automatic reminder emails",
    )
    data_completeness_threshold: int = Field(
        80,
        ge=50,
        le=100,
        description="Minimum data completeness percentage to proceed with DDS",
    )
    require_certification_proof: bool = Field(
        True,
        description="Require uploaded proof for claimed certifications",
    )
    supplier_self_assessment: bool = Field(
        True,
        description="Enable supplier self-assessment questionnaire",
    )
    auto_risk_flag: bool = Field(
        True,
        description="Automatically flag suppliers from high-risk countries",
    )
    tier1_mandatory_fields: List[str] = Field(
        default_factory=lambda: [
            "company_name",
            "country",
            "commodity",
            "geolocation",
            "production_date",
        ],
        description="Mandatory data fields for tier-1 suppliers",
    )


class ComplianceConfig(BaseModel):
    """Compliance rules and scoring configuration.

    Controls which compliance rules are active, scoring thresholds,
    and simplified due diligence eligibility.
    """

    rules_enabled: List[str] = Field(
        default_factory=lambda: [
            "deforestation_free",
            "legal_compliance",
            "geolocation_required",
            "cutoff_date_check",
            "cn_code_classification",
            "country_risk_assessment",
            "supplier_risk_assessment",
            "dds_completeness",
            "document_authentication",
            "chain_of_custody",
        ],
        description="Active compliance rule identifiers",
    )
    compliance_score_threshold: int = Field(
        70,
        ge=0,
        le=100,
        description="Minimum compliance score to proceed with DDS submission",
    )
    simplified_dd_enabled: bool = Field(
        True,
        description="Enable simplified DD for low-risk country sourcing (Article 13)",
    )
    simplified_dd_country_requirement: CountryBenchmark = Field(
        CountryBenchmark.LOW_RISK,
        description="Country benchmark required for simplified DD eligibility",
    )
    penalty_tracking: bool = Field(
        True,
        description="Track and report potential penalty exposure",
    )
    max_penalty_pct_turnover: float = Field(
        4.0,
        ge=0,
        le=100,
        description="Maximum penalty as percentage of turnover (Article 25)",
    )
    competent_authority_check_pct: float = Field(
        9.0,
        ge=0,
        le=100,
        description="Expected percentage of DDS checked by competent authorities",
    )
    competent_authority_high_risk_check_pct: float = Field(
        15.0,
        ge=0,
        le=100,
        description="Check percentage for high-risk country sourcing",
    )


class CutoffDateConfig(BaseModel):
    """Cutoff date verification configuration per Article 1(1).

    Controls the deforestation cutoff date and evidence requirements
    for temporal verification of production plots.
    """

    cutoff_date: date = Field(
        default=date(2020, 12, 31),
        description="Deforestation cutoff date (31 December 2020)",
    )
    evidence_required: bool = Field(
        True,
        description="Require satellite or documentary evidence for cutoff compliance",
    )
    temporal_buffer_days: int = Field(
        0,
        ge=0,
        le=365,
        description="Buffer days before cutoff date for evidence tolerance",
    )
    satellite_verification_enabled: bool = Field(
        True,
        description="Use satellite imagery for cutoff date verification",
    )
    satellite_sources: List[str] = Field(
        default_factory=lambda: [
            "global_forest_watch",
            "copernicus",
            "glad_alerts",
            "planet_labs",
        ],
        description="Satellite data sources for deforestation monitoring",
    )
    min_satellite_resolution_m: int = Field(
        30,
        ge=1,
        le=250,
        description="Minimum satellite resolution in meters for verification",
    )


class ReportingConfig(BaseModel):
    """Reporting and dashboard configuration.

    Controls report generation settings, dashboard refresh intervals,
    output formats, and automated reporting schedules.
    """

    dashboard_refresh_interval_seconds: int = Field(
        300,
        ge=60,
        le=3600,
        description="Dashboard auto-refresh interval in seconds",
    )
    report_formats: List[str] = Field(
        default_factory=lambda: ["pdf", "html", "json", "csv"],
        description="Available report output formats",
    )
    auto_generate_quarterly: bool = Field(
        True,
        description="Auto-generate quarterly compliance reports",
    )
    auto_generate_annual: bool = Field(
        True,
        description="Auto-generate annual compliance summary",
    )
    include_risk_heatmap: bool = Field(
        True,
        description="Include geographic risk heatmap in reports",
    )
    include_supplier_scorecard: bool = Field(
        True,
        description="Include supplier compliance scorecard in reports",
    )
    executive_summary_enabled: bool = Field(
        True,
        description="Generate executive summary for board reporting",
    )
    report_language: str = Field(
        "en",
        description="Primary report language (ISO 639-1)",
    )


class DemoConfig(BaseModel):
    """Demo mode configuration.

    Controls demo/sandbox behavior for testing, training, and
    demonstration purposes without affecting production data.
    """

    enabled: bool = Field(
        False,
        description="Whether demo mode is active",
    )
    demo_suppliers_count: int = Field(
        10,
        ge=5,
        le=100,
        description="Number of demo suppliers to generate",
    )
    demo_plots_count: int = Field(
        20,
        ge=10,
        le=500,
        description="Number of demo plot geolocations to generate",
    )
    use_sample_data: bool = Field(
        False,
        description="Load bundled sample data for demonstration",
    )
    skip_external_apis: bool = Field(
        False,
        description="Skip calls to external APIs (satellite, EU IS)",
    )
    mock_euis_responses: bool = Field(
        False,
        description="Use mocked EU Information System responses",
    )
    fast_execution: bool = Field(
        False,
        description="Skip delays and accelerate processing for demos",
    )
    sample_suppliers_file: str = Field(
        "config/demo/demo_suppliers.json",
        description="Path to demo suppliers data file",
    )
    sample_plots_file: str = Field(
        "config/demo/demo_plots.geojson",
        description="Path to demo plot geolocations file",
    )


# =============================================================================
# Agent Component Configuration
# =============================================================================


class AgentComponentConfig(BaseModel):
    """Configuration for a single agent component in the pack."""

    id: str = Field(..., description="Agent identifier (e.g., AGENT-EUDR-001)")
    name: str = Field("", description="Human-readable agent name")
    description: str = Field("", description="Agent description")
    path: str = Field("", description="Path to agent module directory")
    required: bool = Field(True, description="Whether this agent is required")
    enabled: bool = Field(True, description="Whether this agent is enabled")
    version: str = Field("1.0.0", description="Agent version")
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific configuration overrides",
    )


class ComponentsConfig(BaseModel):
    """All agent components included in the EUDR pack."""

    eudr_agents: List[AgentComponentConfig] = Field(default_factory=list)
    data_agents: List[AgentComponentConfig] = Field(default_factory=list)
    foundation_agents: List[AgentComponentConfig] = Field(default_factory=list)

    def get_all_agent_ids(self) -> List[str]:
        """Return all agent IDs across all component groups."""
        ids: List[str] = []
        for group in [self.eudr_agents, self.data_agents, self.foundation_agents]:
            ids.extend(agent.id for agent in group)
        return ids

    def get_enabled_agent_ids(self) -> List[str]:
        """Return only enabled agent IDs across all component groups."""
        ids: List[str] = []
        for group in [self.eudr_agents, self.data_agents, self.foundation_agents]:
            ids.extend(agent.id for agent in group if agent.enabled)
        return ids

    def get_required_agent_ids(self) -> List[str]:
        """Return only required agent IDs across all component groups."""
        ids: List[str] = []
        for group in [self.eudr_agents, self.data_agents, self.foundation_agents]:
            ids.extend(agent.id for agent in group if agent.required)
        return ids

    def find_agent(self, agent_id: str) -> Optional[AgentComponentConfig]:
        """Find an agent by its ID across all groups."""
        for group in [self.eudr_agents, self.data_agents, self.foundation_agents]:
            for agent in group:
                if agent.id == agent_id:
                    return agent
        return None


# =============================================================================
# Workflow Configuration
# =============================================================================


class WorkflowPhaseConfig(BaseModel):
    """Configuration for a single phase within a workflow."""

    name: str = Field(..., description="Phase identifier")
    description: str = Field("", description="Phase description")
    agents: List[str] = Field(
        default_factory=list, description="Agent IDs in this phase"
    )
    duration_days: int = Field(1, ge=1, description="Estimated duration in days")


class WorkflowConfig(BaseModel):
    """Configuration for a workflow orchestration."""

    display_name: str = Field(..., description="Human-readable workflow name")
    description: str = Field("", description="Workflow description")
    schedule: str = Field(
        "on_demand", description="Schedule: annual, quarterly, on_demand"
    )
    estimated_duration_days: int = Field(
        1, ge=1, description="Total estimated duration"
    )
    phases: List[WorkflowPhaseConfig] = Field(
        default_factory=list, description="Ordered list of workflow phases"
    )
    enabled: bool = Field(True, description="Whether this workflow is enabled")

    def get_all_agent_ids(self) -> List[str]:
        """Return all unique agent IDs used across all phases."""
        ids: Set[str] = set()
        for phase in self.phases:
            ids.update(phase.agents)
        return sorted(ids)


# =============================================================================
# Template Configuration
# =============================================================================


class TemplateConfig(BaseModel):
    """Configuration for a report template."""

    id: str = Field(..., description="Template identifier")
    display_name: str = Field(..., description="Human-readable template name")
    description: str = Field("", description="Template description")
    format: str = Field("pdf", description="Output format (pdf, html, json, csv)")
    template_file: str = Field(..., description="Path to template file")
    enabled: bool = Field(True, description="Whether this template is enabled")


# =============================================================================
# Performance Targets
# =============================================================================


class PerformanceTargets(BaseModel):
    """Performance targets for the EUDR pack."""

    data_ingestion_rps: int = Field(
        5000, description="Records per second for data ingestion"
    )
    geolocation_single_max_ms: int = Field(
        50, description="Max ms for single coordinate verification"
    )
    geolocation_polygon_max_ms: int = Field(
        500, description="Max ms for polygon verification"
    )
    geolocation_batch_1000_max_seconds: int = Field(
        30, description="Max seconds for batch of 1000 coordinates"
    )
    risk_single_supplier_max_seconds: int = Field(
        5, description="Max seconds for single supplier risk assessment"
    )
    risk_batch_100_max_seconds: int = Field(
        60, description="Max seconds for batch of 100 risk assessments"
    )
    dds_single_max_seconds: int = Field(
        30, description="Max seconds for single DDS generation"
    )
    dds_batch_50_max_seconds: int = Field(
        300, description="Max seconds for batch of 50 DDS"
    )
    api_p50_ms: int = Field(100, description="API p50 latency target")
    api_p95_ms: int = Field(500, description="API p95 latency target")
    api_p99_ms: int = Field(2000, description="API p99 latency target")
    availability_percent: float = Field(
        99.9, description="Target uptime percentage"
    )


# =============================================================================
# System Requirements
# =============================================================================


class RequirementsConfig(BaseModel):
    """System requirements for the EUDR pack."""

    python_version: str = Field(">=3.11", description="Minimum Python version")
    postgresql_version: str = Field(">=14", description="Minimum PostgreSQL version")
    redis_version: str = Field(">=7", description="Minimum Redis version")
    min_cpu_cores: int = Field(4, description="Minimum CPU cores")
    min_memory_gb: int = Field(16, description="Minimum memory in GB")
    min_storage_gb: int = Field(100, description="Minimum storage in GB")
    recommended_cpu_cores: int = Field(8, description="Recommended CPU cores")
    recommended_memory_gb: int = Field(32, description="Recommended memory in GB")
    recommended_storage_gb: int = Field(500, description="Recommended storage in GB")
    database_extensions: List[str] = Field(
        default_factory=lambda: ["pgvector", "timescaledb", "postgis"],
        description="Required database extensions",
    )
    min_db_connections: int = Field(20, description="Minimum database connections")


# =============================================================================
# Compliance Reference
# =============================================================================


class ComplianceReference(BaseModel):
    """Reference to a regulatory compliance standard."""

    id: str = Field(..., description="Short identifier for the regulation")
    name: str = Field(..., description="Full name of the regulation")
    regulation: str = Field(..., description="Official regulation number")
    effective_date: str = Field(
        ..., description="Date regulation became effective"
    )
    description: str = Field("", description="Brief description of the regulation")


# =============================================================================
# Pack Metadata
# =============================================================================


class PackMetadata(BaseModel):
    """Pack manifest metadata."""

    name: str = Field(..., description="Pack identifier slug")
    version: str = Field(..., description="Semantic version string")
    display_name: str = Field(..., description="Human-readable pack name")
    description: str = Field("", description="Pack description")
    category: str = Field(..., description="Pack category (e.g., eu-compliance)")
    tier: str = Field("starter", description="Pack tier level")
    author: str = Field("", description="Pack author or team")
    license: str = Field("Proprietary", description="License type")
    min_platform_version: str = Field(
        "2.0.0", description="Minimum GreenLang platform version"
    )
    release_date: str = Field("", description="Release date ISO string")
    support_tier: str = Field("enterprise", description="Support tier level")
    documentation_url: str = Field("", description="URL to pack documentation")
    changelog_url: str = Field("", description="URL to changelog")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    compliance_references: List[ComplianceReference] = Field(
        default_factory=list, description="Regulatory compliance references"
    )


# =============================================================================
# Preset Configuration
# =============================================================================


class PresetConfig(BaseModel):
    """Merged configuration from size and sector presets."""

    size_preset_id: str = Field("", description="Active size preset identifier")
    sector_preset_id: str = Field("", description="Active sector preset identifier")
    operator: OperatorConfig = Field(
        default_factory=OperatorConfig, description="Operator configuration"
    )
    commodities: List[CommodityConfig] = Field(
        default_factory=list, description="Commodity configurations"
    )
    geolocation: GeolocationConfig = Field(
        default_factory=GeolocationConfig, description="Geolocation configuration"
    )
    risk_assessment: RiskAssessmentConfig = Field(
        default_factory=RiskAssessmentConfig,
        description="Risk assessment configuration",
    )
    dds: DDSConfig = Field(
        default_factory=DDSConfig, description="DDS configuration"
    )
    euis: EUISConfig = Field(
        default_factory=EUISConfig, description="EU IS configuration"
    )
    supply_chain: SupplyChainConfig = Field(
        default_factory=SupplyChainConfig,
        description="Supply chain configuration",
    )
    supplier: SupplierConfig = Field(
        default_factory=SupplierConfig, description="Supplier configuration"
    )
    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig, description="Compliance configuration"
    )
    cutoff_date: CutoffDateConfig = Field(
        default_factory=CutoffDateConfig,
        description="Cutoff date configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig, description="Reporting configuration"
    )
    demo: DemoConfig = Field(
        default_factory=DemoConfig, description="Demo mode configuration"
    )
    agent_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Agent-specific configuration overrides by agent ID",
    )
    sector_specific: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sector-specific configuration values",
    )


# =============================================================================
# EUDRPackConfig - Top-level pack configuration
# =============================================================================


class EUDRPackConfig(BaseModel):
    """
    Top-level EUDR Pack configuration model.

    This model represents the fully merged and validated configuration
    for an EUDR Starter Pack deployment. It combines the base manifest,
    size preset, sector preset, and any runtime overrides.

    Attributes:
        metadata: Pack metadata (name, version, description).
        components: Agent component configurations.
        workflows: Workflow orchestration configurations.
        templates: Report template configurations.
        performance: Performance target settings.
        requirements: System requirements.
        presets: Active preset configurations (operator, commodities, risk, etc.).
    """

    metadata: PackMetadata = Field(..., description="Pack metadata")
    components: ComponentsConfig = Field(
        default_factory=ComponentsConfig, description="Component configurations"
    )
    workflows: Dict[str, WorkflowConfig] = Field(
        default_factory=dict, description="Workflow configurations"
    )
    templates: List[TemplateConfig] = Field(
        default_factory=list, description="Template configurations"
    )
    performance: PerformanceTargets = Field(
        default_factory=PerformanceTargets, description="Performance targets"
    )
    requirements: RequirementsConfig = Field(
        default_factory=RequirementsConfig, description="System requirements"
    )
    presets: PresetConfig = Field(
        default_factory=PresetConfig, description="Active preset configuration"
    )

    @field_validator("workflows", mode="before")
    @classmethod
    def parse_workflows(cls, v: Any) -> Dict[str, WorkflowConfig]:
        """Parse workflow definitions from YAML structure."""
        if isinstance(v, dict):
            parsed: Dict[str, WorkflowConfig] = {}
            for key, val in v.items():
                if isinstance(val, WorkflowConfig):
                    parsed[key] = val
                elif isinstance(val, dict):
                    parsed[key] = WorkflowConfig(**val)
                else:
                    parsed[key] = val
            return parsed
        return v

    def get_enabled_workflows(self) -> Dict[str, WorkflowConfig]:
        """Return only enabled workflows."""
        return {k: v for k, v in self.workflows.items() if v.enabled}

    def get_active_agent_ids(self) -> List[str]:
        """Return all enabled agent IDs from components."""
        return self.components.get_enabled_agent_ids()

    def get_enabled_commodities(self) -> List[CommodityConfig]:
        """Return only enabled commodity configurations."""
        return [c for c in self.presets.commodities if c.enabled]

    def get_high_risk_countries(self) -> List[str]:
        """Return list of high-risk country ISO codes."""
        return [
            iso3
            for iso3, data in COUNTRY_RISK_DATABASE.items()
            if data["benchmark"] == "HIGH_RISK"
        ]

    def get_country_benchmark(self, iso3: str) -> CountryBenchmark:
        """Look up country benchmark classification."""
        entry = COUNTRY_RISK_DATABASE.get(iso3)
        if entry is None:
            return CountryBenchmark.STANDARD_RISK
        return CountryBenchmark(entry["benchmark"])

    def is_simplified_dd_eligible(self, country_iso3: str) -> bool:
        """Check if sourcing from a country qualifies for simplified DD."""
        if not self.presets.compliance.simplified_dd_enabled:
            return False
        benchmark = self.get_country_benchmark(country_iso3)
        return benchmark == CountryBenchmark.LOW_RISK

    def get_cn_codes_for_commodity(
        self, commodity: EUDRCommodity
    ) -> List[Dict[str, str]]:
        """Return all Annex I CN codes for a given commodity."""
        return ANNEX_I_CN_CODES.get(commodity.value, [])


# =============================================================================
# Utility Functions
# =============================================================================


def get_country_risk(iso3: str) -> CountryBenchmark:
    """
    Look up Article 29 country benchmark classification.

    Args:
        iso3: ISO-3166-1 alpha-3 country code.

    Returns:
        Country benchmark classification (LOW_RISK, STANDARD_RISK, HIGH_RISK).
        Defaults to STANDARD_RISK for unknown countries.
    """
    entry = COUNTRY_RISK_DATABASE.get(iso3)
    if entry is None:
        return CountryBenchmark.STANDARD_RISK
    return CountryBenchmark(entry["benchmark"])


def is_eudr_commodity(cn_code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a CN code falls within EUDR Annex I scope.

    Args:
        cn_code: Combined Nomenclature code (2-digit or full).

    Returns:
        Tuple of (is_in_scope, commodity_name_or_none).
    """
    cn_prefix = cn_code.replace(" ", "")[:4]
    for commodity, codes in ANNEX_I_CN_CODES.items():
        for code_entry in codes:
            code_clean = code_entry["code"].replace(" ", "")
            if cn_prefix.startswith(code_clean[:4]) or code_clean.startswith(cn_prefix):
                return True, commodity
    return False, None


def get_all_cn_codes() -> List[Dict[str, str]]:
    """Return all Annex I CN codes across all commodities."""
    all_codes: List[Dict[str, str]] = []
    for commodity, codes in ANNEX_I_CN_CODES.items():
        for code in codes:
            all_codes.append({
                "commodity": commodity,
                "code": code["code"],
                "description": code["description"],
            })
    return all_codes


def calculate_config_hash(config_data: Dict[str, Any]) -> str:
    """
    Calculate SHA-256 hash of configuration for provenance tracking.

    Args:
        config_data: Configuration dictionary to hash.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    config_json = json.dumps(config_data, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


# =============================================================================
# PackConfig - Main configuration manager
# =============================================================================


class PackConfig:
    """
    Configuration manager for PACK-006 EUDR Starter Pack.

    Loads and merges configuration from multiple sources in the following
    priority order (later sources override earlier):

        1. Base pack.yaml manifest
        2. Size preset (large_operator, mid_market, sme, first_time)
        3. Sector preset (palm_oil, timber_wood, cocoa_coffee, soy_cattle, rubber)
        4. Environment variables (EUDR_PACK_* prefix)
        5. Runtime overrides

    Attributes:
        pack: The fully resolved EUDRPackConfig instance.
        config_hash: SHA-256 hash of the resolved configuration for provenance.
        loaded_at: Timestamp when configuration was loaded.
        source_files: List of source files that were loaded.

    Example:
        >>> config = PackConfig.load(
        ...     size_preset="mid_market",
        ...     sector_preset="palm_oil",
        ... )
        >>> print(config.pack.metadata.version)
        '1.0.0'
        >>> print(config.active_agents)
        ['AGENT-EUDR-001', ...]
    """

    VALID_SIZE_PRESETS = {"large_operator", "mid_market", "sme", "first_time"}
    VALID_SECTOR_PRESETS = {
        "palm_oil",
        "timber_wood",
        "cocoa_coffee",
        "soy_cattle",
        "rubber",
    }

    def __init__(
        self,
        pack: EUDRPackConfig,
        config_hash: str,
        loaded_at: datetime,
        source_files: List[str],
    ) -> None:
        """
        Initialize PackConfig with resolved configuration.

        Args:
            pack: Fully resolved pack configuration.
            config_hash: SHA-256 hash of the configuration.
            loaded_at: Timestamp of configuration loading.
            source_files: List of source files that were loaded.
        """
        self.pack = pack
        self.config_hash = config_hash
        self.loaded_at = loaded_at
        self.source_files = source_files

    @property
    def active_agents(self) -> List[str]:
        """Return list of active (enabled) agent IDs."""
        return self.pack.get_active_agent_ids()

    @property
    def enabled_commodities(self) -> List[EUDRCommodity]:
        """Return list of enabled commodity types."""
        return [c.commodity_type for c in self.pack.get_enabled_commodities()]

    @classmethod
    def load(
        cls,
        pack_dir: Optional[Union[str, Path]] = None,
        size_preset: Optional[str] = None,
        sector_preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        demo_mode: bool = False,
    ) -> "PackConfig":
        """
        Load and merge pack configuration from all sources.

        Args:
            pack_dir: Path to the pack root directory. Defaults to PACK_BASE_DIR.
            size_preset: Size preset to apply (large_operator, mid_market, sme, first_time).
            sector_preset: Sector preset to apply (palm_oil, timber_wood, etc.).
            overrides: Dictionary of runtime configuration overrides.
            demo_mode: Whether to enable demo mode.

        Returns:
            Fully resolved PackConfig instance.

        Raises:
            FileNotFoundError: If pack.yaml or preset file is not found.
            ValueError: If preset name is invalid.
        """
        start_time = datetime.now()
        pack_dir = Path(pack_dir) if pack_dir else PACK_BASE_DIR
        source_files: List[str] = []

        # Step 1: Load base manifest
        manifest_path = pack_dir / "pack.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Pack manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        source_files.append(str(manifest_path))
        logger.info("Loaded base manifest: %s", manifest_path)

        # Step 2: Load size preset
        preset_data: Dict[str, Any] = {}
        if size_preset:
            if size_preset not in cls.VALID_SIZE_PRESETS:
                raise ValueError(
                    f"Invalid size preset '{size_preset}'. "
                    f"Valid options: {cls.VALID_SIZE_PRESETS}"
                )
            preset_path = pack_dir / "config" / "presets" / f"{size_preset}.yaml"
            if preset_path.exists():
                with open(preset_path, "r", encoding="utf-8") as f:
                    preset_data = yaml.safe_load(f) or {}
                source_files.append(str(preset_path))
                logger.info("Loaded size preset: %s", size_preset)

        # Step 3: Load sector preset
        sector_data: Dict[str, Any] = {}
        if sector_preset:
            if sector_preset not in cls.VALID_SECTOR_PRESETS:
                raise ValueError(
                    f"Invalid sector preset '{sector_preset}'. "
                    f"Valid options: {cls.VALID_SECTOR_PRESETS}"
                )
            sector_path = pack_dir / "config" / "sectors" / f"{sector_preset}.yaml"
            if sector_path.exists():
                with open(sector_path, "r", encoding="utf-8") as f:
                    sector_data = yaml.safe_load(f) or {}
                source_files.append(str(sector_path))
                logger.info("Loaded sector preset: %s", sector_preset)

        # Step 4: Load demo config if requested
        demo_data: Dict[str, Any] = {}
        if demo_mode:
            demo_path = pack_dir / "config" / "demo" / "demo_config.yaml"
            if demo_path.exists():
                with open(demo_path, "r", encoding="utf-8") as f:
                    demo_data = yaml.safe_load(f) or {}
                source_files.append(str(demo_path))
                logger.info("Loaded demo config")

        # Step 5: Load environment overrides
        env_overrides = cls._load_env_overrides()

        # Step 6: Merge all configuration layers
        merged_config = cls._merge_configs(
            base_config, preset_data, sector_data, demo_data, env_overrides
        )

        # Step 7: Apply runtime overrides
        if overrides:
            merged_config = cls._deep_merge(merged_config, overrides)

        # Step 8: Build typed configuration
        pack_config = cls._build_pack_config(
            merged_config, size_preset, sector_preset
        )

        # Step 9: Calculate provenance hash
        config_hash = calculate_config_hash(merged_config)

        loaded_at = datetime.now()
        load_time_ms = (loaded_at - start_time).total_seconds() * 1000
        logger.info(
            "PackConfig loaded in %.1f ms (hash: %s...)",
            load_time_ms,
            config_hash[:12],
        )

        return cls(
            pack=pack_config,
            config_hash=config_hash,
            loaded_at=loaded_at,
            source_files=source_files,
        )

    @classmethod
    def _load_env_overrides(cls) -> Dict[str, Any]:
        """Load configuration overrides from EUDR_PACK_* environment variables."""
        overrides: Dict[str, Any] = {}
        prefix = "EUDR_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Parse boolean values
                if value.lower() in ("true", "1", "yes"):
                    overrides[config_key] = True
                elif value.lower() in ("false", "0", "no"):
                    overrides[config_key] = False
                else:
                    # Try numeric, fall back to string
                    try:
                        overrides[config_key] = int(value)
                    except ValueError:
                        try:
                            overrides[config_key] = float(value)
                        except ValueError:
                            overrides[config_key] = value
        return overrides

    @classmethod
    def _merge_configs(
        cls,
        base: Dict[str, Any],
        preset: Dict[str, Any],
        sector: Dict[str, Any],
        demo: Dict[str, Any],
        env: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge configuration layers with proper precedence."""
        result = dict(base)
        for layer in [preset, sector, demo, env]:
            if layer:
                result = cls._deep_merge(result, layer)
        return result

    @classmethod
    def _deep_merge(
        cls, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence.

        Lists are replaced entirely (not appended). Nested dicts are merged
        recursively.
        """
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def _build_pack_config(
        cls,
        merged: Dict[str, Any],
        size_preset: Optional[str],
        sector_preset: Optional[str],
    ) -> EUDRPackConfig:
        """Build typed EUDRPackConfig from merged dictionary."""
        metadata_raw = merged.get("metadata", {})
        metadata = PackMetadata(
            name=metadata_raw.get("name", "PACK-006-eudr-starter"),
            version=metadata_raw.get("version", "1.0.0"),
            display_name=metadata_raw.get("display_name", "EUDR Starter Pack"),
            description=metadata_raw.get("description", ""),
            category=metadata_raw.get("category", "eu-compliance"),
            tier=metadata_raw.get("tier", "starter"),
            author=metadata_raw.get("author", "GreenLang Platform Team"),
            license=metadata_raw.get("license", "Proprietary"),
            min_platform_version=metadata_raw.get("min_platform_version", "2.0.0"),
            release_date=metadata_raw.get("release_date", ""),
            support_tier=metadata_raw.get("support_tier", "enterprise"),
            documentation_url=metadata_raw.get("documentation_url", ""),
            changelog_url=metadata_raw.get("changelog_url", ""),
            tags=metadata_raw.get("tags", []),
            compliance_references=[
                ComplianceReference(**ref)
                for ref in metadata_raw.get("compliance_references", [])
            ],
        )

        presets = PresetConfig(
            size_preset_id=size_preset or "",
            sector_preset_id=sector_preset or "",
        )

        return EUDRPackConfig(
            metadata=metadata,
            presets=presets,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete configuration to a dictionary."""
        return {
            "pack": self.pack.model_dump(mode="json"),
            "config_hash": self.config_hash,
            "loaded_at": self.loaded_at.isoformat(),
            "source_files": self.source_files,
        }

    def get_provenance_hash(self) -> str:
        """Return SHA-256 hash of the current configuration state."""
        return self.config_hash

    def get_country_risk(self, iso3: str) -> CountryBenchmark:
        """Look up country risk benchmark."""
        return self.pack.get_country_benchmark(iso3)

    def is_commodity_in_scope(self, cn_code: str) -> Tuple[bool, Optional[str]]:
        """Check if a CN code falls within EUDR Annex I scope."""
        return is_eudr_commodity(cn_code)

    def __repr__(self) -> str:
        """String representation of PackConfig."""
        return (
            f"PackConfig("
            f"name='{self.pack.metadata.name}', "
            f"version='{self.pack.metadata.version}', "
            f"agents={len(self.active_agents)}, "
            f"hash='{self.config_hash[:12]}...'"
            f")"
        )
