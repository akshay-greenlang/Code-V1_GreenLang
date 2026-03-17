"""
PACK-004 CBAM Readiness Pack - Configuration Manager

This module implements the CBAMPackConfig class and all supporting Pydantic
models for configuring the CBAM Readiness Pack. It provides typed, validated
configuration for importers of CBAM-covered goods (cement, iron & steel,
aluminium, fertilizers, electricity, hydrogen).

Configuration Merge Order (later overrides earlier):
    1. Base defaults defined in Pydantic models
    2. Commodity preset (steel_importer / aluminum_importer / cement_importer / etc.)
    3. Sector preset (heavy_industry / chemicals / energy_trading)
    4. Environment overrides (CBAM_PACK_* environment variables)
    5. Explicit runtime overrides

Regulatory References:
    - CBAM Regulation (EU) 2023/956
    - CBAM Implementing Regulation (EU) 2023/1773
    - EU ETS Directive 2003/87/EC

Example:
    >>> config = CBAMPackConfig.from_preset("steel_importer")
    >>> print(config.goods.enabled_categories)
    [<CBAMGoodsCategory.IRON_STEEL: 'iron_steel'>]
    >>> print(config.certificate.ets_price_source)
    'AUCTION'
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent
PRESETS_DIR = CONFIG_DIR / "presets"
SECTORS_DIR = CONFIG_DIR / "sectors"
DEMO_DIR = CONFIG_DIR / "demo"


# =============================================================================
# Enumerations
# =============================================================================


class CBAMGoodsCategory(str, Enum):
    """
    CBAM goods categories as defined in Annex I of Regulation (EU) 2023/956.

    Each category corresponds to a group of products identified by Combined
    Nomenclature (CN) codes that are subject to CBAM obligations.
    """

    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILIZERS = "fertilizers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


class CalculationMethod(str, Enum):
    """
    Methods for calculating embedded emissions per CBAM methodology.

    - ACTUAL: Installation-specific actual emission data from the producer.
    - DEFAULT: EU default values published by the European Commission.
    - COUNTRY_DEFAULT: Country-specific default values based on national
      grid/industry emission factors and a markup percentage.
    """

    ACTUAL = "actual"
    DEFAULT = "default"
    COUNTRY_DEFAULT = "country_default"


class ReportingPeriod(str, Enum):
    """
    CBAM reporting period phase.

    - TRANSITIONAL: Oct 2023 - Dec 2025 (reporting only, no certificates).
    - DEFINITIVE: From Jan 2026 (certificates required, verified emissions).
    """

    TRANSITIONAL = "transitional"
    DEFINITIVE = "definitive"


class CostScenario(str, Enum):
    """
    Cost projection scenarios for CBAM certificate price forecasting.

    - LOW: Conservative EU ETS price trajectory.
    - MID: Central/baseline EU ETS price trajectory.
    - HIGH: Aggressive EU ETS price trajectory.
    """

    LOW = "low"
    MID = "mid"
    HIGH = "high"


class VerificationFrequency(str, Enum):
    """
    Frequency of third-party verification of embedded emissions.

    - ANNUAL: Verification every year (required for large importers).
    - BIENNIAL: Verification every two years (may apply to smaller importers).
    """

    ANNUAL = "annual"
    BIENNIAL = "biennial"


class ETSPriceSource(str, Enum):
    """Source for EU ETS / CBAM certificate pricing."""

    AUCTION = "auction"
    SPOT = "spot"
    MANUAL = "manual"


class EmissionFactorSource(str, Enum):
    """Source database for emission factors."""

    EU_DEFAULT = "eu_default"
    IPCC = "ipcc"
    INDUSTRY = "industry"


class DataSubmissionFormat(str, Enum):
    """Supported formats for supplier data submission."""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    EXCEL = "excel"


class ReportLanguage(str, Enum):
    """Supported languages for CBAM reports."""

    EN = "EN"
    DE = "DE"
    FR = "FR"
    IT = "IT"
    ES = "ES"
    NL = "NL"
    PL = "PL"
    PT = "PT"


class EUMemberState(str, Enum):
    """EU member states for importer registration."""

    AT = "AT"   # Austria
    BE = "BE"   # Belgium
    BG = "BG"   # Bulgaria
    HR = "HR"   # Croatia
    CY = "CY"   # Cyprus
    CZ = "CZ"   # Czech Republic
    DK = "DK"   # Denmark
    EE = "EE"   # Estonia
    FI = "FI"   # Finland
    FR = "FR"   # France
    DE = "DE"   # Germany
    GR = "GR"   # Greece
    HU = "HU"   # Hungary
    IE = "IE"   # Ireland
    IT = "IT"   # Italy
    LV = "LV"   # Latvia
    LT = "LT"   # Lithuania
    LU = "LU"   # Luxembourg
    MT = "MT"   # Malta
    NL = "NL"   # Netherlands
    PL = "PL"   # Poland
    PT = "PT"   # Portugal
    RO = "RO"   # Romania
    SK = "SK"   # Slovakia
    SI = "SI"   # Slovenia
    ES = "ES"   # Spain
    SE = "SE"   # Sweden


# =============================================================================
# CN Codes Reference - Complete CBAM Annex I mapping
# =============================================================================

# Comprehensive mapping of CBAM goods categories to their CN codes.
# Source: Annex I of Regulation (EU) 2023/956

CN_CODES: Dict[str, List[Dict[str, str]]] = {
    "cement": [
        {"code": "2523 10 00", "description": "Cement clinkers"},
        {"code": "2523 21 00", "description": "White Portland cement"},
        {"code": "2523 29 00", "description": "Other Portland cement"},
        {"code": "2523 30 00", "description": "Aluminous cement"},
        {"code": "2523 90 00", "description": "Other hydraulic cements"},
    ],
    "iron_steel": [
        # Chapter 72 - Iron and steel
        {"code": "7201", "description": "Pig iron and spiegeleisen in pigs, blocks or other primary forms"},
        {"code": "7202 11", "description": "Ferro-manganese containing >2% carbon"},
        {"code": "7202 19", "description": "Other ferro-manganese"},
        {"code": "7202 21", "description": "Ferro-silicon containing >55% silicon"},
        {"code": "7202 29", "description": "Other ferro-silicon"},
        {"code": "7202 30", "description": "Ferro-silico-manganese"},
        {"code": "7202 41", "description": "Ferro-chromium containing >4% carbon"},
        {"code": "7202 49", "description": "Other ferro-chromium"},
        {"code": "7202 50", "description": "Ferro-silico-chromium"},
        {"code": "7202 60", "description": "Ferro-nickel"},
        {"code": "7202 70", "description": "Ferro-molybdenum"},
        {"code": "7202 80", "description": "Ferro-tungsten and ferro-silico-tungsten"},
        {"code": "7202 91", "description": "Ferro-titanium and ferro-silico-titanium"},
        {"code": "7202 92", "description": "Ferro-vanadium"},
        {"code": "7202 93", "description": "Ferro-niobium"},
        {"code": "7202 99", "description": "Other ferro-alloys"},
        {"code": "7203", "description": "Ferrous products obtained by direct reduction of iron ore"},
        {"code": "7204", "description": "Ferrous waste and scrap; remelting scrap ingots"},
        {"code": "7205", "description": "Granules and powders of pig iron, spiegeleisen, iron or steel"},
        {"code": "7206", "description": "Iron and non-alloy steel in ingots or other primary forms"},
        {"code": "7207", "description": "Semi-finished products of iron or non-alloy steel"},
        {"code": "7208", "description": "Flat-rolled products of iron/non-alloy steel, hot-rolled, width >=600mm"},
        {"code": "7209", "description": "Flat-rolled products of iron/non-alloy steel, cold-rolled, width >=600mm"},
        {"code": "7210", "description": "Flat-rolled products of iron/non-alloy steel, clad/plated/coated, width >=600mm"},
        {"code": "7211", "description": "Flat-rolled products of iron or non-alloy steel, width <600mm"},
        {"code": "7212", "description": "Flat-rolled products of iron/non-alloy steel, clad/plated/coated, width <600mm"},
        {"code": "7213", "description": "Bars and rods, hot-rolled, in irregularly wound coils, of iron or non-alloy steel"},
        {"code": "7214", "description": "Other bars and rods of iron or non-alloy steel, not further worked than forged"},
        {"code": "7215", "description": "Other bars and rods of iron or non-alloy steel"},
        {"code": "7216", "description": "Angles, shapes and sections of iron or non-alloy steel"},
        {"code": "7217", "description": "Wire of iron or non-alloy steel"},
        {"code": "7218", "description": "Stainless steel in ingots or other primary forms; semi-finished products"},
        {"code": "7219", "description": "Flat-rolled products of stainless steel, width >=600mm"},
        {"code": "7220", "description": "Flat-rolled products of stainless steel, width <600mm"},
        {"code": "7221", "description": "Bars and rods of stainless steel, hot-rolled, irregularly wound"},
        {"code": "7222", "description": "Other bars and rods of stainless steel; angles, shapes and sections"},
        {"code": "7223", "description": "Wire of stainless steel"},
        {"code": "7224", "description": "Other alloy steel in ingots or other primary forms; semi-finished products"},
        {"code": "7225", "description": "Flat-rolled products of other alloy steel, width >=600mm"},
        {"code": "7226", "description": "Flat-rolled products of other alloy steel, width <600mm"},
        {"code": "7227", "description": "Bars and rods of other alloy steel, hot-rolled, irregularly wound"},
        {"code": "7228", "description": "Other bars and rods of other alloy steel; angles, shapes, sections; hollow drill bars"},
        {"code": "7229", "description": "Wire of other alloy steel"},
        # Chapter 73 - Articles of iron or steel
        {"code": "7301", "description": "Sheet piling of iron or steel"},
        {"code": "7302", "description": "Railway or tramway track construction material of iron or steel"},
        {"code": "7303", "description": "Tubes, pipes and hollow profiles of cast iron"},
        {"code": "7304", "description": "Tubes, pipes and hollow profiles, seamless, of iron (other than cast iron) or steel"},
        {"code": "7305", "description": "Other tubes and pipes, having circular cross-sections, external diameter >406.4mm"},
        {"code": "7306", "description": "Other tubes, pipes and hollow profiles of iron or steel"},
        {"code": "7307", "description": "Tube or pipe fittings of iron or steel"},
        {"code": "7308", "description": "Structures and parts of structures of iron or steel"},
        {"code": "7309", "description": "Reservoirs, tanks, vats and similar containers of iron or steel, >300 litres"},
        {"code": "7310", "description": "Tanks, casks, drums, cans, boxes of iron or steel, capacity <=300 litres"},
        {"code": "7311", "description": "Containers for compressed or liquefied gas, of iron or steel"},
        {"code": "7312", "description": "Stranded wire, ropes, cables, plaited bands, slings of iron or steel"},
        {"code": "7313", "description": "Barbed wire of iron or steel; twisted hoop or single flat wire"},
        {"code": "7318", "description": "Screws, bolts, nuts, coach screws, screw hooks, rivets of iron or steel"},
        {"code": "7326", "description": "Other articles of iron or steel"},
    ],
    "aluminium": [
        {"code": "7601", "description": "Unwrought aluminium"},
        {"code": "7602", "description": "Aluminium waste and scrap"},
        {"code": "7603", "description": "Aluminium powders and flakes"},
        {"code": "7604", "description": "Aluminium bars, rods and profiles"},
        {"code": "7605", "description": "Aluminium wire"},
        {"code": "7606", "description": "Aluminium plates, sheets and strip, thickness >0.2mm"},
        {"code": "7607", "description": "Aluminium foil (whether or not printed), thickness <=0.2mm"},
        {"code": "7608", "description": "Aluminium tubes and pipes"},
        {"code": "7609", "description": "Aluminium tube or pipe fittings"},
        {"code": "7610", "description": "Aluminium structures and parts of structures"},
        {"code": "7611", "description": "Aluminium reservoirs, tanks, vats and similar containers, >300 litres"},
        {"code": "7612", "description": "Aluminium casks, drums, cans, boxes and similar containers, <=300 litres"},
        {"code": "7613", "description": "Aluminium containers for compressed or liquefied gas"},
        {"code": "7614", "description": "Stranded wire, cables, plaited bands and the like, of aluminium"},
        {"code": "7616", "description": "Other articles of aluminium"},
    ],
    "fertilizers": [
        {"code": "2808 00 00", "description": "Nitric acid; sulphonitric acids"},
        {"code": "2814 10 00", "description": "Anhydrous ammonia"},
        {"code": "2814 20 00", "description": "Ammonia in aqueous solution"},
        {"code": "2834 10 00", "description": "Nitrites"},
        {"code": "3102 10", "description": "Urea, whether or not in aqueous solution"},
        {"code": "3102 21 00", "description": "Ammonium sulphate"},
        {"code": "3102 29 00", "description": "Double salts and mixtures of ammonium sulphate and ammonium nitrate"},
        {"code": "3102 30", "description": "Ammonium nitrate, whether or not in aqueous solution"},
        {"code": "3102 40", "description": "Mixtures of ammonium nitrate with calcium carbonate or non-fertilizing substances"},
        {"code": "3102 50 00", "description": "Sodium nitrate"},
        {"code": "3102 60 00", "description": "Double salts and mixtures of calcium nitrate and ammonium nitrate"},
        {"code": "3102 80 00", "description": "Mixtures of urea and ammonium nitrate in aqueous or ammoniacal solution"},
        {"code": "3102 90", "description": "Other mineral or chemical fertilizers, nitrogenous, including mixtures"},
        {"code": "3103 11 00", "description": "Superphosphates containing >=35% diphosphorus pentaoxide"},
        {"code": "3103 19 00", "description": "Other superphosphates"},
        {"code": "3103 90 00", "description": "Other mineral or chemical fertilizers, phosphatic"},
        {"code": "3104 20", "description": "Potassium chloride"},
        {"code": "3104 30 00", "description": "Potassium sulphate"},
        {"code": "3104 90", "description": "Other mineral or chemical fertilizers, potassic"},
        {"code": "3105 10 00", "description": "Goods of this chapter in tablets or similar forms, or in packages <=10 kg"},
        {"code": "3105 20", "description": "Mineral or chemical fertilizers containing nitrogen, phosphorus and potassium (NPK)"},
        {"code": "3105 30 00", "description": "Diammonium hydrogenorthophosphate (diammonium phosphate, DAP)"},
        {"code": "3105 40 00", "description": "Ammonium dihydrogenorthophosphate (monoammonium phosphate, MAP)"},
        {"code": "3105 51 00", "description": "Mineral or chemical fertilizers containing nitrates and phosphates"},
        {"code": "3105 59 00", "description": "Other mineral or chemical fertilizers containing nitrogen and phosphorus"},
        {"code": "3105 60 00", "description": "Mineral or chemical fertilizers containing phosphorus and potassium"},
        {"code": "3105 90", "description": "Other mineral or chemical fertilizers"},
    ],
    "electricity": [
        {"code": "2716 00 00", "description": "Electrical energy"},
    ],
    "hydrogen": [
        {"code": "2804 10 00", "description": "Hydrogen"},
    ],
}

# Flat lookup: CN code prefix -> goods category
CN_CODE_TO_CATEGORY: Dict[str, CBAMGoodsCategory] = {}


def _build_cn_code_lookup() -> None:
    """Build the flat CN code to category lookup table."""
    category_enum_map = {
        "cement": CBAMGoodsCategory.CEMENT,
        "iron_steel": CBAMGoodsCategory.IRON_STEEL,
        "aluminium": CBAMGoodsCategory.ALUMINIUM,
        "fertilizers": CBAMGoodsCategory.FERTILIZERS,
        "electricity": CBAMGoodsCategory.ELECTRICITY,
        "hydrogen": CBAMGoodsCategory.HYDROGEN,
    }
    for cat_key, cn_list in CN_CODES.items():
        category = category_enum_map[cat_key]
        for entry in cn_list:
            code = entry["code"].replace(" ", "")
            CN_CODE_TO_CATEGORY[code] = category
            # Also map the 4-digit prefix for lookup by heading
            prefix_4 = code[:4]
            if prefix_4 not in CN_CODE_TO_CATEGORY:
                CN_CODE_TO_CATEGORY[prefix_4] = category


_build_cn_code_lookup()


# =============================================================================
# EU Default Emission Factors
# =============================================================================

EU_DEFAULT_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "cement": {
        "clinker": 0.84,                # tCO2e/tonne clinker
        "portland_cement": 0.68,         # tCO2e/tonne cement
        "aluminous_cement": 0.72,        # tCO2e/tonne
        "other_hydraulic_cement": 0.60,  # tCO2e/tonne
    },
    "iron_steel": {
        "pig_iron_bf_bof": 1.85,         # tCO2e/tonne
        "pig_iron_bf_bof_pellets": 1.92, # tCO2e/tonne (with pelletizing)
        "crude_steel_bof": 1.85,         # tCO2e/tonne
        "crude_steel_eaf": 0.45,         # tCO2e/tonne
        "crude_steel_dri_eaf": 1.10,     # tCO2e/tonne
        "hot_rolled_flat": 1.98,         # tCO2e/tonne
        "cold_rolled_flat": 2.15,        # tCO2e/tonne
        "coated_flat": 2.25,             # tCO2e/tonne
        "long_products": 1.95,           # tCO2e/tonne
        "tubes_pipes": 2.10,             # tCO2e/tonne
        "stainless_steel": 2.80,         # tCO2e/tonne
        "alloy_steel": 2.50,             # tCO2e/tonne
        "ferro_alloys": 3.50,            # tCO2e/tonne (varies widely)
    },
    "aluminium": {
        "unwrought_primary": 8.00,       # tCO2e/tonne
        "unwrought_secondary": 0.50,     # tCO2e/tonne
        "bars_rods_profiles": 8.50,      # tCO2e/tonne
        "plates_sheets": 8.80,           # tCO2e/tonne
        "foil": 9.20,                    # tCO2e/tonne
        "tubes_pipes": 8.60,             # tCO2e/tonne
        "structures": 9.00,              # tCO2e/tonne
    },
    "fertilizers": {
        "ammonia_anhydrous": 2.10,       # tCO2e/tonne
        "ammonia_aqueous": 1.80,         # tCO2e/tonne
        "urea": 1.60,                    # tCO2e/tonne
        "ammonium_nitrate": 3.10,        # tCO2e/tonne (includes N2O)
        "ammonium_sulphate": 1.50,       # tCO2e/tonne
        "nitric_acid": 1.70,             # tCO2e/tonne
        "npk_fertilizer": 2.40,          # tCO2e/tonne
        "dap": 1.90,                     # tCO2e/tonne
        "map": 1.70,                     # tCO2e/tonne
        "uan_solution": 2.00,            # tCO2e/tonne
    },
    "electricity": {
        "eu_average": 0.23,              # tCO2e/MWh
    },
    "hydrogen": {
        "grey_smr": 10.00,              # tCO2e/tonne H2
        "blue_smr_ccs": 3.00,           # tCO2e/tonne H2
        "green_electrolysis": 0.50,     # tCO2e/tonne H2
        "turquoise_pyrolysis": 4.00,    # tCO2e/tonne H2
    },
}


# Country-specific default emission factors for major exporting countries
COUNTRY_DEFAULT_FACTORS: Dict[str, Dict[str, float]] = {
    "TR": {  # Turkey
        "iron_steel_bf_bof": 1.95,
        "iron_steel_eaf": 0.52,
        "cement_clinker": 0.88,
        "aluminium_primary": 9.50,
        "electricity_grid": 0.48,
    },
    "CN": {  # China
        "iron_steel_bf_bof": 2.15,
        "iron_steel_eaf": 0.65,
        "cement_clinker": 0.92,
        "aluminium_primary": 12.50,
        "fertilizer_ammonia": 2.80,
        "electricity_grid": 0.58,
        "hydrogen_grey": 12.00,
    },
    "RU": {  # Russia
        "iron_steel_bf_bof": 2.05,
        "iron_steel_eaf": 0.55,
        "cement_clinker": 0.85,
        "aluminium_primary": 5.50,
        "fertilizer_ammonia": 2.20,
        "electricity_grid": 0.42,
        "hydrogen_grey": 10.50,
    },
    "IN": {  # India
        "iron_steel_bf_bof": 2.50,
        "iron_steel_eaf": 0.70,
        "cement_clinker": 0.95,
        "aluminium_primary": 14.00,
        "fertilizer_ammonia": 2.90,
        "electricity_grid": 0.72,
    },
    "UA": {  # Ukraine
        "iron_steel_bf_bof": 2.20,
        "iron_steel_eaf": 0.58,
        "cement_clinker": 0.90,
        "electricity_grid": 0.45,
    },
    "EG": {  # Egypt
        "iron_steel_eaf": 0.60,
        "fertilizer_ammonia": 2.40,
        "electricity_grid": 0.50,
    },
    "BR": {  # Brazil
        "iron_steel_bf_bof": 1.90,
        "iron_steel_eaf": 0.48,
        "aluminium_primary": 6.00,
        "electricity_grid": 0.08,
    },
    "ZA": {  # South Africa
        "iron_steel_bf_bof": 2.30,
        "cement_clinker": 0.92,
        "aluminium_primary": 13.00,
        "electricity_grid": 0.95,
    },
    "KR": {  # South Korea
        "iron_steel_bf_bof": 1.88,
        "iron_steel_eaf": 0.50,
        "cement_clinker": 0.80,
        "aluminium_primary": 8.50,
        "electricity_grid": 0.42,
    },
    "NO": {  # Norway (low-carbon aluminium)
        "aluminium_primary": 2.50,
        "electricity_grid": 0.01,
    },
    "IS": {  # Iceland (geothermal-powered aluminium)
        "aluminium_primary": 2.00,
        "electricity_grid": 0.00,
    },
}


# EU ETS free allocation phase-out schedule for CBAM sectors (2026-2034)
FREE_ALLOCATION_PHASEOUT: Dict[int, float] = {
    2026: 97.5,
    2027: 95.0,
    2028: 90.0,
    2029: 82.5,
    2030: 75.0,
    2031: 60.0,
    2032: 45.0,
    2033: 30.0,
    2034: 0.0,
}


# Default CN codes grouped by category for the GoodsCategoryConfig defaults
DEFAULT_CN_CODES_BY_CATEGORY: Dict[str, List[str]] = {
    cat: [entry["code"] for entry in entries]
    for cat, entries in CN_CODES.items()
}


# =============================================================================
# Pydantic Configuration Models
# =============================================================================


class ImporterConfig(BaseModel):
    """
    Configuration for the EU importer / authorized CBAM declarant.

    Contains company identification, EORI number, CBAM registry ID,
    and EU member state of establishment.
    """

    company_name: str = Field(
        "",
        description="Legal name of the importing company",
    )
    eori_number: str = Field(
        "",
        description=(
            "Economic Operators Registration and Identification number. "
            "Required for all EU importers. Format: CC + up to 15 alphanumeric characters."
        ),
    )
    authorized_declarant: str = Field(
        "",
        description="Name of the authorized CBAM declarant (natural person or customs representative)",
    )
    eu_member_state: Optional[EUMemberState] = Field(
        None,
        description="EU member state where the importer is established",
    )
    cbam_registry_id: str = Field(
        "",
        description="CBAM Transitional Registry or Definitive Registry account identifier",
    )
    contact_email: str = Field(
        "",
        description="Primary contact email for CBAM correspondence",
    )
    contact_phone: str = Field(
        "",
        description="Primary contact phone number",
    )
    customs_office_code: str = Field(
        "",
        description="Code of the customs office of import (for cross-referencing customs declarations)",
    )

    @field_validator("eori_number")
    @classmethod
    def validate_eori_format(cls, v: str) -> str:
        """Validate EORI number format: 2-letter country code + up to 15 characters."""
        if v and len(v) < 3:
            raise ValueError(
                "EORI number must be at least 3 characters (2-letter country code + identifier)"
            )
        if v and not v[:2].isalpha():
            raise ValueError("EORI number must start with a 2-letter country code")
        return v.upper() if v else v


class GoodsCategoryConfig(BaseModel):
    """
    Configuration for CBAM goods categories and CN code mapping.

    Defines which goods categories are active, their associated CN codes,
    and whether default CN codes from Annex I should be pre-populated.
    """

    enabled_categories: List[CBAMGoodsCategory] = Field(
        default_factory=lambda: list(CBAMGoodsCategory),
        description="List of enabled CBAM goods categories",
    )
    cn_codes_per_category: Dict[str, List[str]] = Field(
        default_factory=lambda: dict(DEFAULT_CN_CODES_BY_CATEGORY),
        description=(
            "Mapping of goods category to list of CN codes. "
            "Pre-populated with all Annex I CN codes by default."
        ),
    )
    custom_cn_codes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Additional CN codes added by the user beyond Annex I defaults",
    )
    precursor_tracking: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "cement": ["clinker"],
            "iron_steel": ["pig_iron", "direct_reduced_iron", "crude_steel", "ferro_alloys"],
            "aluminium": ["alumina", "unwrought_aluminium"],
            "fertilizers": ["ammonia", "nitric_acid", "urea"],
            "electricity": [],
            "hydrogen": [],
        },
        description="Precursor products tracked per goods category",
    )

    def get_cn_codes_for_category(self, category: CBAMGoodsCategory) -> List[str]:
        """Return all CN codes for a given goods category."""
        cat_key = category.value
        base_codes = self.cn_codes_per_category.get(cat_key, [])
        custom_codes = self.custom_cn_codes.get(cat_key, [])
        return base_codes + custom_codes

    def get_all_enabled_cn_codes(self) -> List[str]:
        """Return all CN codes across all enabled categories."""
        codes: List[str] = []
        for category in self.enabled_categories:
            codes.extend(self.get_cn_codes_for_category(category))
        return codes

    @field_validator("enabled_categories")
    @classmethod
    def validate_at_least_one_category(cls, v: List[CBAMGoodsCategory]) -> List[CBAMGoodsCategory]:
        """Ensure at least one goods category is enabled (unless small importer)."""
        # Allow empty for small_importer preset - validated at pack level
        return v


class EmissionConfig(BaseModel):
    """
    Configuration for embedded emission calculation methodology.

    Defines the preferred calculation method, markup percentages for
    country defaults, whether indirect emissions are included, and
    the source database for emission factors.
    """

    calculation_method: CalculationMethod = Field(
        CalculationMethod.ACTUAL,
        description=(
            "Preferred calculation method for embedded emissions. "
            "ACTUAL is preferred when installation-specific data is available."
        ),
    )
    fallback_method: CalculationMethod = Field(
        CalculationMethod.COUNTRY_DEFAULT,
        description="Fallback method when preferred method data is unavailable",
    )
    default_markup_percentage: float = Field(
        25.0,
        ge=0.0,
        le=100.0,
        description=(
            "Markup percentage applied to country default values when actual data "
            "is unavailable. Typical values: 10% (low risk), 25% (medium), 50% (high risk)."
        ),
    )
    indirect_emissions_included: bool = Field(
        True,
        description=(
            "Whether indirect emissions (from electricity consumption) are included "
            "in the embedded emission calculation. Required for cement, steel, aluminium, "
            "and fertilizers."
        ),
    )
    precursor_tracking_enabled: bool = Field(
        True,
        description=(
            "Whether precursor product emissions are tracked and allocated. "
            "Required for complex goods (e.g., steel from pig iron, cement from clinker)."
        ),
    )
    emission_factor_source: EmissionFactorSource = Field(
        EmissionFactorSource.EU_DEFAULT,
        description="Primary source database for emission factors",
    )
    emission_factor_vintage_year: int = Field(
        2024,
        ge=2020,
        le=2030,
        description="Reference year for emission factors",
    )
    carbon_price_deduction_enabled: bool = Field(
        True,
        description=(
            "Whether carbon prices paid in the country of origin can be deducted "
            "from the CBAM certificate obligation."
        ),
    )
    country_carbon_price_sources: List[str] = Field(
        default_factory=lambda: [
            "eu_ets",
            "uk_ets",
            "china_ets",
            "korea_ets",
            "carbon_tax_registry",
        ],
        description="Recognized carbon pricing mechanisms for price deduction",
    )


class CertificateConfig(BaseModel):
    """
    Configuration for CBAM certificate management.

    Manages certificate pricing, purchase planning, holding targets,
    surrender deadlines, and cost scenario modeling.
    """

    ets_price_source: ETSPriceSource = Field(
        ETSPriceSource.AUCTION,
        description=(
            "Source for CBAM certificate pricing. AUCTION uses weekly average "
            "EU ETS auction price. SPOT uses daily spot price. MANUAL allows "
            "override."
        ),
    )
    manual_price_eur_per_tco2e: Optional[float] = Field(
        None,
        ge=0.0,
        description="Manual certificate price (EUR/tCO2e) when ets_price_source is MANUAL",
    )
    free_allocation_enabled: bool = Field(
        True,
        description=(
            "Whether free allocation phase-out deduction is applied to CBAM "
            "certificate obligations. True during 2026-2034 transition."
        ),
    )
    carbon_deduction_enabled: bool = Field(
        True,
        description=(
            "Whether carbon price paid in country of origin is deducted from "
            "CBAM certificate obligation."
        ),
    )
    cost_scenario: CostScenario = Field(
        CostScenario.MID,
        description="Default cost scenario for certificate price forecasting",
    )
    cost_scenario_prices: Dict[str, Dict[int, float]] = Field(
        default_factory=lambda: {
            "low": {
                2026: 55.0, 2027: 58.0, 2028: 60.0, 2029: 62.0, 2030: 65.0,
                2031: 68.0, 2032: 72.0, 2033: 75.0, 2034: 78.0, 2035: 80.0,
            },
            "mid": {
                2026: 75.0, 2027: 80.0, 2028: 85.0, 2029: 90.0, 2030: 95.0,
                2031: 100.0, 2032: 108.0, 2033: 115.0, 2034: 120.0, 2035: 125.0,
            },
            "high": {
                2026: 100.0, 2027: 110.0, 2028: 120.0, 2029: 130.0, 2030: 140.0,
                2031: 155.0, 2032: 170.0, 2033: 185.0, 2034: 200.0, 2035: 220.0,
            },
        },
        description="Price projections (EUR/tCO2e) per scenario per year",
    )
    quarterly_holding_target_pct: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description=(
            "Target percentage of estimated annual obligation to hold in "
            "certificates at each quarter end. 80% is the regulatory minimum "
            "by end of Q2; 50% is a prudent planning default."
        ),
    )
    surrender_deadline_month: int = Field(
        5,
        ge=1,
        le=12,
        description="Month of the annual certificate surrender deadline (May = 5)",
    )
    surrender_deadline_day: int = Field(
        31,
        ge=1,
        le=31,
        description="Day of month for the annual certificate surrender deadline",
    )
    repurchase_enabled: bool = Field(
        True,
        description=(
            "Whether repurchase of certificates is enabled when price "
            "decreases, allowing buy-back within one year of purchase."
        ),
    )
    repurchase_max_pct: float = Field(
        33.33,
        ge=0.0,
        le=100.0,
        description="Maximum percentage of certificates eligible for repurchase (one-third)",
    )

    @model_validator(mode="after")
    def validate_manual_price(self) -> "CertificateConfig":
        """Ensure manual price is provided when price source is MANUAL."""
        if self.ets_price_source == ETSPriceSource.MANUAL and self.manual_price_eur_per_tco2e is None:
            raise ValueError(
                "manual_price_eur_per_tco2e must be provided when ets_price_source is MANUAL"
            )
        return self


class QuarterlyConfig(BaseModel):
    """
    Configuration for quarterly CBAM report generation and submission.

    Controls scheduling, deadline management, amendment windows, XML
    validation, and report language settings.
    """

    auto_schedule: bool = Field(
        True,
        description=(
            "Whether quarterly reports are automatically scheduled based on "
            "calendar quarters. When True, the system creates report tasks "
            "at the start of each quarter for the previous quarter's data."
        ),
    )
    submission_deadline_buffer_days: int = Field(
        7,
        ge=0,
        le=30,
        description=(
            "Number of days before the official submission deadline to target "
            "internal completion. Provides buffer for review and corrections."
        ),
    )
    amendment_window_days: int = Field(
        60,
        ge=0,
        le=120,
        description=(
            "Number of days after initial submission during which amendments "
            "can be filed. Per CBAM IR, amendments are allowed within 2 months."
        ),
    )
    xml_validation_enabled: bool = Field(
        True,
        description="Whether XML schema validation is run before submission",
    )
    report_language: ReportLanguage = Field(
        ReportLanguage.EN,
        description="Language for quarterly report generation",
    )
    additional_languages: List[ReportLanguage] = Field(
        default_factory=list,
        description="Additional languages for parallel report generation",
    )
    include_supporting_documents: bool = Field(
        True,
        description="Whether to attach supporting documents (invoices, BOLs) to report package",
    )
    quarterly_deadlines: Dict[str, str] = Field(
        default_factory=lambda: {
            "Q1": "April 30",
            "Q2": "July 31",
            "Q3": "October 31",
            "Q4": "January 31",
        },
        description="Submission deadlines per quarter",
    )
    archive_reports: bool = Field(
        True,
        description="Whether submitted reports are archived with SHA-256 provenance hashes",
    )
    max_retries_on_submission_failure: int = Field(
        3,
        ge=0,
        le=10,
        description="Maximum retry attempts for registry submission failures",
    )


class SupplierConfig(BaseModel):
    """
    Configuration for supplier emission data management.

    Controls questionnaire dispatch frequency, data quality thresholds,
    installation limits, EORI validation, and data submission formats.
    """

    auto_request_frequency_months: int = Field(
        3,
        ge=1,
        le=12,
        description=(
            "Frequency in months for automatic questionnaire dispatch to "
            "registered suppliers. Quarterly (3) aligns with reporting cycle."
        ),
    )
    quality_threshold: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description=(
            "Minimum data quality score (0-100) for supplier data to be "
            "accepted for actual emission calculations. Below this threshold, "
            "default values are used with a markup."
        ),
    )
    max_installations_per_supplier: int = Field(
        20,
        ge=1,
        le=100,
        description="Maximum number of production installations per supplier",
    )
    eori_validation_enabled: bool = Field(
        True,
        description="Whether supplier EORI numbers are validated against EU registry",
    )
    data_submission_format: DataSubmissionFormat = Field(
        DataSubmissionFormat.JSON,
        description="Preferred format for supplier data submission",
    )
    accepted_formats: List[DataSubmissionFormat] = Field(
        default_factory=lambda: list(DataSubmissionFormat),
        description="All accepted data submission formats",
    )
    questionnaire_template_version: str = Field(
        "1.0.0",
        description="Version of the supplier emission data questionnaire template",
    )
    reminder_days_before_deadline: List[int] = Field(
        default_factory=lambda: [30, 14, 7, 3, 1],
        description="Days before deadline to send reminder notifications to suppliers",
    )
    auto_fallback_to_defaults: bool = Field(
        True,
        description=(
            "When True, automatically falls back to country/EU default values "
            "if supplier actual data is not received by the deadline."
        ),
    )
    supplier_portal_enabled: bool = Field(
        True,
        description="Whether the supplier self-service data portal is enabled",
    )
    data_retention_years: int = Field(
        10,
        ge=5,
        le=20,
        description="Number of years to retain supplier emission data",
    )

    @field_validator("quality_threshold")
    @classmethod
    def validate_quality_threshold(cls, v: float) -> float:
        """Ensure quality threshold is reasonable."""
        if v < 30.0:
            logger.warning(
                "Quality threshold %.1f%% is very low; consider 70%% or higher for regulatory compliance",
                v,
            )
        return v


class DeMinimisConfig(BaseModel):
    """
    Configuration for de minimis threshold monitoring.

    Under CBAM, small import quantities may be exempt or subject to
    simplified reporting. This config controls threshold monitoring,
    alerting, and exemption assessment.
    """

    monitoring_enabled: bool = Field(
        True,
        description="Whether de minimis threshold monitoring is active",
    )
    threshold_tonnes: float = Field(
        50.0,
        ge=0.0,
        description=(
            "Annual import tonnage threshold below which simplified reporting "
            "or exemption may apply. Default 50 tonnes per goods category."
        ),
    )
    threshold_value_eur: float = Field(
        150.0,
        ge=0.0,
        description=(
            "Per-consignment value threshold (EUR) below which CBAM obligations "
            "do not apply. Per Article 2(4)."
        ),
    )
    alert_thresholds: List[float] = Field(
        default_factory=lambda: [80.0, 90.0, 95.0, 100.0],
        description=(
            "Percentage thresholds at which alerts are triggered as cumulative "
            "imports approach the de minimis limit."
        ),
    )
    auto_exemption: bool = Field(
        False,
        description=(
            "Whether to automatically generate exemption documentation when "
            "the importer remains below thresholds for the reporting year."
        ),
    )
    sector_grouping: bool = Field(
        True,
        description=(
            "Whether de minimis thresholds are assessed per goods category "
            "(True) or across all categories combined (False)."
        ),
    )
    monitoring_frequency: str = Field(
        "per_import",
        description="Frequency of threshold checks: per_import, daily, weekly",
    )
    cumulative_tracking_start: str = Field(
        "january_1",
        description="Start date for cumulative annual volume tracking",
    )
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email", "dashboard", "webhook"],
        description="Channels for threshold alert notifications",
    )

    @field_validator("alert_thresholds")
    @classmethod
    def validate_alert_thresholds(cls, v: List[float]) -> List[float]:
        """Ensure alert thresholds are sorted and in valid range."""
        for threshold in v:
            if threshold < 0.0 or threshold > 200.0:
                raise ValueError(f"Alert threshold {threshold} must be between 0 and 200")
        return sorted(v)


class VerificationConfig(BaseModel):
    """
    Configuration for third-party verification of embedded emissions.

    Controls verification frequency, materiality thresholds, verifier
    accreditation requirements, and evidence retention policies.
    """

    frequency: VerificationFrequency = Field(
        VerificationFrequency.ANNUAL,
        description="Frequency of third-party verification",
    )
    materiality_threshold_pct: float = Field(
        5.0,
        ge=0.0,
        le=25.0,
        description=(
            "Materiality threshold as percentage of total embedded emissions. "
            "Discrepancies above this threshold require corrective action."
        ),
    )
    verifier_accreditation_required: bool = Field(
        True,
        description=(
            "Whether the verifier must be accredited per CBAM requirements "
            "(ISO 14065 or equivalent national accreditation body)."
        ),
    )
    accepted_accreditation_bodies: List[str] = Field(
        default_factory=lambda: [
            "DAkkS",           # Germany
            "UKAS",            # UK
            "COFRAC",          # France
            "ACCREDIA",        # Italy
            "ENAC",            # Spain
            "RvA",             # Netherlands
            "SAS",             # Switzerland
            "JAS-ANZ",         # Australia/New Zealand
        ],
        description="List of accepted accreditation bodies for verifiers",
    )
    evidence_retention_years: int = Field(
        10,
        ge=5,
        le=20,
        description=(
            "Number of years to retain verification evidence and opinions. "
            "CBAM requires minimum 10 years."
        ),
    )
    verification_standards: List[str] = Field(
        default_factory=lambda: [
            "ISO 14064-3",
            "ISO 14065",
            "CBAM Delegated Regulation",
        ],
        description="Applicable verification standards",
    )
    site_visit_required: bool = Field(
        True,
        description="Whether physical site visits are required as part of verification",
    )
    remote_verification_allowed: bool = Field(
        False,
        description="Whether remote/virtual verification is permitted (post-COVID flexibility)",
    )
    sampling_methodology: str = Field(
        "risk_based",
        description="Methodology for selecting installations/goods for detailed verification",
    )
    max_findings_before_rejection: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of major findings before a verification report is rejected",
    )
    corrective_action_deadline_days: int = Field(
        30,
        ge=7,
        le=90,
        description="Days allowed for addressing verification findings",
    )


# =============================================================================
# Main Pack Configuration
# =============================================================================


class CBAMPackConfig(BaseModel):
    """
    Root configuration for the CBAM Readiness Pack.

    Aggregates all sub-configurations for importer details, goods categories,
    emission calculation, certificate management, quarterly reporting, supplier
    management, de minimis monitoring, and verification.

    Example:
        >>> config = CBAMPackConfig.from_preset("steel_importer")
        >>> print(config.goods.enabled_categories)
        [<CBAMGoodsCategory.IRON_STEEL: 'iron_steel'>]

        >>> config = CBAMPackConfig.from_yaml("path/to/config.yaml")
        >>> config.validate_config()
    """

    # Sub-configurations
    importer: ImporterConfig = Field(
        default_factory=ImporterConfig,
        description="Importer company identification and contact details",
    )
    goods: GoodsCategoryConfig = Field(
        default_factory=GoodsCategoryConfig,
        description="CBAM goods categories and CN code configuration",
    )
    emission: EmissionConfig = Field(
        default_factory=EmissionConfig,
        description="Embedded emission calculation methodology configuration",
    )
    certificate: CertificateConfig = Field(
        default_factory=CertificateConfig,
        description="CBAM certificate management configuration",
    )
    quarterly: QuarterlyConfig = Field(
        default_factory=QuarterlyConfig,
        description="Quarterly report generation and submission configuration",
    )
    supplier: SupplierConfig = Field(
        default_factory=SupplierConfig,
        description="Supplier emission data management configuration",
    )
    deminimis: DeMinimisConfig = Field(
        default_factory=DeMinimisConfig,
        description="De minimis threshold monitoring configuration",
    )
    verification: VerificationConfig = Field(
        default_factory=VerificationConfig,
        description="Third-party verification configuration",
    )

    # Pack-level settings
    reporting_year: int = Field(
        2026,
        ge=2023,
        le=2040,
        description="The calendar year for which CBAM reporting is being prepared",
    )
    reporting_period: ReportingPeriod = Field(
        ReportingPeriod.DEFINITIVE,
        description=(
            "CBAM reporting period phase. TRANSITIONAL (2023-2025) requires "
            "quarterly reports only. DEFINITIVE (2026+) requires certificates."
        ),
    )
    transitional_mode: bool = Field(
        False,
        description=(
            "Legacy flag: True enables transitional period behavior (reporting only). "
            "Prefer using reporting_period=TRANSITIONAL instead."
        ),
    )
    demo_mode: bool = Field(
        False,
        description="When True, uses sample data and skips external API calls",
    )
    pack_version: str = Field(
        "1.0.0",
        description="Version of the CBAM Readiness Pack",
    )
    log_level: str = Field(
        "INFO",
        description="Logging level for pack operations",
    )
    provenance_enabled: bool = Field(
        True,
        description="Whether SHA-256 provenance hashing is enabled for all outputs",
    )

    # Class-level constants
    AVAILABLE_PRESETS: ClassVar[List[str]] = [
        "steel_importer",
        "aluminum_importer",
        "cement_importer",
        "fertilizer_importer",
        "multi_commodity",
        "small_importer",
    ]

    AVAILABLE_SECTORS: ClassVar[List[str]] = [
        "heavy_industry",
        "chemicals",
        "energy_trading",
    ]

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "CBAMPackConfig":
        """
        Load CBAMPackConfig from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Fully validated CBAMPackConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content fails validation.
        """
        path = Path(yaml_path)
        if not path.is_absolute():
            path = CONFIG_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        logger.info("Loading CBAM pack configuration from: %s", path)

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Empty configuration file: {path}")

        config = cls.model_validate(raw)
        logger.info(
            "Loaded CBAM config: %d categories enabled, reporting_year=%d, period=%s",
            len(config.goods.enabled_categories),
            config.reporting_year,
            config.reporting_period.value,
        )
        return config

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        sector_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "CBAMPackConfig":
        """
        Load CBAMPackConfig from a named preset with optional sector overlay.

        Args:
            preset_name: Name of the commodity preset (e.g., "steel_importer").
            sector_name: Optional sector preset to merge on top (e.g., "heavy_industry").
            overrides: Optional dict of field overrides applied last.

        Returns:
            Fully validated CBAMPackConfig instance.

        Raises:
            ValueError: If the preset name is not recognized.
            FileNotFoundError: If the preset file does not exist.
        """
        if preset_name not in cls.AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {cls.AVAILABLE_PRESETS}"
            )

        # Load commodity preset
        preset_path = PRESETS_DIR / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")

        logger.info("Loading CBAM preset: %s", preset_name)
        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Merge sector preset if provided
        if sector_name:
            if sector_name not in cls.AVAILABLE_SECTORS:
                raise ValueError(
                    f"Unknown sector '{sector_name}'. Available: {cls.AVAILABLE_SECTORS}"
                )
            sector_path = SECTORS_DIR / f"{sector_name}.yaml"
            if not sector_path.exists():
                raise FileNotFoundError(f"Sector file not found: {sector_path}")

            logger.info("Merging sector preset: %s", sector_name)
            with open(sector_path, "r", encoding="utf-8") as f:
                sector_data = yaml.safe_load(f) or {}
            preset_data = cls._deep_merge(preset_data, sector_data)

        # Apply runtime overrides
        if overrides:
            logger.info("Applying %d runtime overrides", len(overrides))
            preset_data = cls._deep_merge(preset_data, overrides)

        # Apply environment variable overrides (CBAM_PACK_*)
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            logger.info("Applying %d environment variable overrides", len(env_overrides))
            preset_data = cls._deep_merge(preset_data, env_overrides)

        config = cls.model_validate(preset_data)
        logger.info(
            "Loaded CBAM config from preset '%s': %d categories, year=%d",
            preset_name,
            len(config.goods.enabled_categories),
            config.reporting_year,
        )
        return config

    @classmethod
    def from_demo(cls) -> "CBAMPackConfig":
        """
        Load the demo configuration for EuroSteel Imports GmbH.

        Returns:
            CBAMPackConfig configured for the demo scenario.
        """
        demo_path = DEMO_DIR / "demo_config.yaml"
        config = cls.from_yaml(demo_path)
        config.demo_mode = True
        return config

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        """
        Run comprehensive validation on the configuration.

        Returns:
            List of validation warning/error messages. Empty list means valid.
        """
        issues: List[str] = []

        # Check reporting period consistency
        if self.reporting_year <= 2025 and self.reporting_period == ReportingPeriod.DEFINITIVE:
            issues.append(
                f"reporting_year={self.reporting_year} is in transitional period but "
                f"reporting_period is set to DEFINITIVE. Consider setting to TRANSITIONAL."
            )

        if self.reporting_year >= 2026 and self.reporting_period == ReportingPeriod.TRANSITIONAL:
            issues.append(
                f"reporting_year={self.reporting_year} is in definitive period but "
                f"reporting_period is set to TRANSITIONAL."
            )

        # Check importer config
        if not self.demo_mode:
            if not self.importer.company_name:
                issues.append("importer.company_name is not set")
            if not self.importer.eori_number:
                issues.append("importer.eori_number is not set (required for CBAM registry)")

        # Check goods categories
        if not self.goods.enabled_categories:
            issues.append(
                "No goods categories are enabled. At least one category is required "
                "unless operating under de minimis."
            )

        # Check certificate config for definitive period
        if self.reporting_period == ReportingPeriod.DEFINITIVE:
            if (
                self.certificate.ets_price_source == ETSPriceSource.MANUAL
                and self.certificate.manual_price_eur_per_tco2e is None
            ):
                issues.append(
                    "Certificate price source is MANUAL but no manual price is set"
                )

        # Validate CN codes against enabled categories
        for category in self.goods.enabled_categories:
            cn_codes = self.goods.get_cn_codes_for_category(category)
            if not cn_codes:
                issues.append(
                    f"No CN codes configured for enabled category: {category.value}"
                )

        # Check supplier config
        if self.supplier.quality_threshold < 50.0:
            issues.append(
                f"supplier.quality_threshold={self.supplier.quality_threshold}% is very low. "
                f"Recommended minimum is 70% for regulatory compliance."
            )

        # Check verification config
        if self.reporting_period == ReportingPeriod.DEFINITIVE:
            if not self.verification.verifier_accreditation_required:
                issues.append(
                    "Verifier accreditation is not required, but CBAM definitive period "
                    "mandates accredited verifiers."
                )

        if issues:
            for issue in issues:
                logger.warning("Configuration issue: %s", issue)

        return issues

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_free_allocation_factor(self, year: Optional[int] = None) -> float:
        """
        Get the EU ETS free allocation percentage for a given year.

        Returns the percentage of free allocation still available (0-100).
        After 2034, this returns 0.0 (no free allocation = full CBAM).

        Args:
            year: Calendar year. Defaults to self.reporting_year.

        Returns:
            Free allocation percentage (0.0 to 100.0).
        """
        target_year = year or self.reporting_year
        return FREE_ALLOCATION_PHASEOUT.get(target_year, 0.0 if target_year > 2034 else 100.0)

    def get_cbam_coverage_factor(self, year: Optional[int] = None) -> float:
        """
        Get the CBAM coverage factor (inverse of free allocation) for a year.

        Returns the percentage of emissions subject to CBAM certificates.

        Args:
            year: Calendar year. Defaults to self.reporting_year.

        Returns:
            CBAM coverage percentage (0.0 to 100.0).
        """
        return 100.0 - self.get_free_allocation_factor(year)

    def get_eu_default_factor(
        self,
        category: CBAMGoodsCategory,
        product_type: str,
    ) -> Optional[float]:
        """
        Look up EU default emission factor for a goods category and product type.

        Args:
            category: CBAM goods category.
            product_type: Specific product type within the category.

        Returns:
            Emission factor in tCO2e/tonne (or tCO2e/MWh for electricity), or None if not found.
        """
        cat_factors = EU_DEFAULT_EMISSION_FACTORS.get(category.value, {})
        return cat_factors.get(product_type)

    def get_country_default_factor(
        self,
        country_code: str,
        product_type: str,
    ) -> Optional[float]:
        """
        Look up country-specific default emission factor.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g., "CN", "TR").
            product_type: Specific product type key.

        Returns:
            Emission factor in tCO2e/tonne, or None if not found.
        """
        country_factors = COUNTRY_DEFAULT_FACTORS.get(country_code, {})
        return country_factors.get(product_type)

    def classify_cn_code(self, cn_code: str) -> Optional[CBAMGoodsCategory]:
        """
        Classify a CN code to its CBAM goods category.

        Args:
            cn_code: Combined Nomenclature code (e.g., "7208 51 00" or "7208").

        Returns:
            Matching CBAMGoodsCategory, or None if the code is not a CBAM good.
        """
        clean_code = cn_code.replace(" ", "").replace(".", "")

        # Try exact match first
        if clean_code in CN_CODE_TO_CATEGORY:
            return CN_CODE_TO_CATEGORY[clean_code]

        # Try progressively shorter prefixes (8-digit -> 6 -> 4)
        for length in [8, 6, 4]:
            prefix = clean_code[:length]
            if prefix in CN_CODE_TO_CATEGORY:
                return CN_CODE_TO_CATEGORY[prefix]

        return None

    def estimate_certificate_cost(
        self,
        embedded_emissions_tco2e: float,
        year: Optional[int] = None,
        scenario: Optional[CostScenario] = None,
        country_carbon_price_eur: float = 0.0,
    ) -> Dict[str, float]:
        """
        Estimate CBAM certificate cost for a given volume of embedded emissions.

        Applies free allocation phase-out and carbon price deduction.

        Args:
            embedded_emissions_tco2e: Total embedded emissions in tCO2e.
            year: Calendar year for pricing. Defaults to self.reporting_year.
            scenario: Cost scenario. Defaults to self.certificate.cost_scenario.
            country_carbon_price_eur: Carbon price already paid in country of origin (EUR/tCO2e).

        Returns:
            Dict with keys: gross_obligation_tco2e, net_obligation_tco2e,
            price_per_tco2e, gross_cost_eur, net_cost_eur, carbon_deduction_eur.
        """
        target_year = year or self.reporting_year
        target_scenario = scenario or self.certificate.cost_scenario

        # Get CBAM coverage factor (how much is NOT covered by free allocation)
        cbam_coverage = self.get_cbam_coverage_factor(target_year) / 100.0

        # Gross obligation = embedded emissions * CBAM coverage
        gross_obligation = embedded_emissions_tco2e * cbam_coverage

        # Carbon price deduction
        carbon_deduction_per_tonne = country_carbon_price_eur if self.certificate.carbon_deduction_enabled else 0.0

        # Certificate price
        scenario_prices = self.certificate.cost_scenario_prices.get(target_scenario.value, {})
        price_per_tco2e = scenario_prices.get(target_year, 80.0)

        # Net price after carbon deduction
        net_price = max(0.0, price_per_tco2e - carbon_deduction_per_tonne)

        # Cost calculation
        gross_cost = gross_obligation * price_per_tco2e
        carbon_deduction_total = gross_obligation * carbon_deduction_per_tonne
        net_cost = gross_obligation * net_price

        return {
            "gross_obligation_tco2e": round(gross_obligation, 4),
            "net_obligation_tco2e": round(gross_obligation, 4),
            "price_per_tco2e_eur": round(price_per_tco2e, 2),
            "carbon_deduction_per_tco2e_eur": round(carbon_deduction_per_tonne, 2),
            "gross_cost_eur": round(gross_cost, 2),
            "carbon_deduction_total_eur": round(carbon_deduction_total, 2),
            "net_cost_eur": round(net_cost, 2),
            "free_allocation_pct": round(self.get_free_allocation_factor(target_year), 2),
            "cbam_coverage_pct": round(cbam_coverage * 100, 2),
            "year": target_year,
            "scenario": target_scenario.value,
        }

    def compute_provenance_hash(self) -> str:
        """
        Compute SHA-256 hash of the entire configuration for audit provenance.

        Returns:
            Hex-encoded SHA-256 hash of the JSON-serialized config.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Write the current configuration to a YAML file.

        Args:
            output_path: Path where the YAML file will be written.
        """
        path = Path(output_path)
        data = self.model_dump(mode="json")

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info("Configuration written to: %s", path)

    def summary(self) -> Dict[str, Any]:
        """
        Generate a concise summary of the configuration.

        Returns:
            Dict with key configuration parameters for display.
        """
        return {
            "pack_version": self.pack_version,
            "reporting_year": self.reporting_year,
            "reporting_period": self.reporting_period.value,
            "importer": self.importer.company_name or "(not configured)",
            "eori": self.importer.eori_number or "(not configured)",
            "member_state": self.importer.eu_member_state.value if self.importer.eu_member_state else "(not set)",
            "enabled_categories": [c.value for c in self.goods.enabled_categories],
            "total_cn_codes": len(self.goods.get_all_enabled_cn_codes()),
            "calculation_method": self.emission.calculation_method.value,
            "ets_price_source": self.certificate.ets_price_source.value,
            "cost_scenario": self.certificate.cost_scenario.value,
            "supplier_quality_threshold": self.supplier.quality_threshold,
            "deminimis_monitoring": self.deminimis.monitoring_enabled,
            "verification_frequency": self.verification.frequency.value,
            "demo_mode": self.demo_mode,
            "provenance_hash": self.compute_provenance_hash()[:16] + "...",
        }

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep-merge two dictionaries. Values from overlay take precedence.

        Args:
            base: Base dictionary.
            overlay: Dictionary to merge on top of base.

        Returns:
            New merged dictionary.
        """
        result = dict(base)
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = CBAMPackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variables follow the pattern CBAM_PACK_{SECTION}_{KEY}.
        Example: CBAM_PACK_REPORTING_YEAR=2027

        Returns:
            Dict of overrides parsed from environment variables.
        """
        overrides: Dict[str, Any] = {}
        prefix = "CBAM_PACK_"

        env_mapping: Dict[str, Tuple[str, type]] = {
            "REPORTING_YEAR": ("reporting_year", int),
            "DEMO_MODE": ("demo_mode", bool),
            "LOG_LEVEL": ("log_level", str),
            "TRANSITIONAL_MODE": ("transitional_mode", bool),
            "IMPORTER_COMPANY_NAME": ("importer.company_name", str),
            "IMPORTER_EORI": ("importer.eori_number", str),
            "EMISSION_METHOD": ("emission.calculation_method", str),
            "EMISSION_MARKUP_PCT": ("emission.default_markup_percentage", float),
            "CERTIFICATE_PRICE_SOURCE": ("certificate.ets_price_source", str),
            "SUPPLIER_QUALITY_THRESHOLD": ("supplier.quality_threshold", float),
            "VERIFICATION_FREQUENCY": ("verification.frequency", str),
        }

        for env_suffix, (config_path, value_type) in env_mapping.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if value_type == bool:
                        parsed = env_value.lower() in ("true", "1", "yes")
                    elif value_type == int:
                        parsed = int(env_value)
                    elif value_type == float:
                        parsed = float(env_value)
                    else:
                        parsed = env_value

                    # Set nested key
                    parts = config_path.split(".")
                    current = overrides
                    for part in parts[:-1]:
                        current = current.setdefault(part, {})
                    current[parts[-1]] = parsed

                    logger.info("Applied env override: %s = %s", env_var, parsed)
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid env override %s=%s: %s", env_var, env_value, e)

        return overrides
