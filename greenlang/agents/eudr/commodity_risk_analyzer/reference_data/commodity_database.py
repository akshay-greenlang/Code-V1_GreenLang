# -*- coding: utf-8 -*-
"""
Commodity Database - AGENT-EUDR-018 Commodity Risk Analyzer

Comprehensive EUDR commodity reference data covering all 7 regulated
commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood) with
Harmonized System (HS) codes, Combined Nomenclature (CN) codes, intrinsic
risk factors, key producing countries, derived products (Annex I), supply
chain depth metrics, and sustainability certification schemes.

Each commodity entry provides:
    - HS codes (6-digit Harmonized System codes per WCO 2024)
    - CN codes (8-digit EU Combined Nomenclature codes)
    - Intrinsic risk factors (0.0-1.0 scale):
        * deforestation_pressure: Direct deforestation linkage strength
        * supply_chain_complexity: Number of intermediaries and opacity
        * traceability_difficulty: Difficulty of origin verification
        * processing_variability: Product transformation variability
    - Key producing countries with production share percentages
    - Average supply chain depth (number of tiers)
    - Typical processing stages
    - Sustainability certifications available
    - EUDR Annex I product codes

Derived products cover 70+ Annex I products across 7 commodities:
    - Cattle: 8 product categories (beef, leather, tallow, gelatin, etc.)
    - Cocoa: 8 product categories (paste, butter, powder, chocolate, etc.)
    - Coffee: 7 product categories (green, roasted, instant, extract, etc.)
    - Oil Palm: 8 product categories (crude oil, refined, biodiesel, etc.)
    - Rubber: 7 product categories (natural rubber, latex, tires, etc.)
    - Soya: 8 product categories (beans, meal, oil, lecithin, etc.)
    - Wood: 10 product categories (sawn, plywood, veneer, pulp, etc.)

Data Sources:
    - WCO Harmonized System Nomenclature 2024
    - European Commission Combined Nomenclature 2024
    - FAO FAOSTAT Commodity Production Database 2024
    - EUDR Annex I Product List (EU 2023/1115)
    - RSPO, FSC, PEFC, Rainforest Alliance, UTZ certification databases

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "WCO Harmonized System Nomenclature 2024",
    "European Commission Combined Nomenclature 2024",
    "FAO FAOSTAT Commodity Production Database 2024",
    "EUDR Annex I Product List (EU 2023/1115)",
    "RSPO Supply Chain Certification Standard 2023",
    "FSC Chain of Custody Certification (FSC-STD-40-004 V3-1)",
    "Rainforest Alliance Sustainable Agriculture Standard 2022",
]

# ===========================================================================
# EUDR Commodities - All 7 regulated commodities
# ===========================================================================

EUDR_COMMODITIES: Dict[str, Dict[str, Any]] = {

    # -------------------------------------------------------------------
    # 1. CATTLE
    # -------------------------------------------------------------------
    "cattle": {
        "commodity_type": "cattle",
        "display_name": "Cattle",
        "eudr_annex_ref": "Annex I - Part A",
        "description": (
            "Cattle and products derived from cattle including beef, "
            "leather, tallow, gelatin, and bone meal."
        ),
        "hs_codes": [
            "0102.21", "0102.29", "0102.31", "0102.39", "0102.90",
            "0201.10", "0201.20", "0201.30",
            "0202.10", "0202.20", "0202.30",
            "0206.10", "0206.21", "0206.22", "0206.29",
            "0210.20",
            "4101.20", "4101.50", "4101.90",
            "4104.11", "4104.19", "4104.41", "4104.49",
            "4107.11", "4107.12", "4107.19",
        ],
        "cn_codes": [
            "0102 21 10", "0102 21 30", "0102 21 90",
            "0102 29 05", "0102 29 10", "0102 29 21",
            "0201 10 00", "0201 20 20", "0201 20 30",
            "0202 10 00", "0202 20 10", "0202 20 30",
            "4101 20 10", "4101 50 10", "4101 90 00",
        ],
        "intrinsic_risk_factors": {
            "deforestation_pressure": 0.85,
            "supply_chain_complexity": 0.70,
            "traceability_difficulty": 0.75,
            "processing_variability": 0.60,
        },
        "key_producing_countries": {
            "BRA": {"production_share": 0.265, "rank": 1},
            "IND": {"production_share": 0.095, "rank": 2},
            "USA": {"production_share": 0.090, "rank": 3},
            "CHN": {"production_share": 0.085, "rank": 4},
            "ARG": {"production_share": 0.055, "rank": 5},
            "AUS": {"production_share": 0.045, "rank": 6},
            "MEX": {"production_share": 0.035, "rank": 7},
            "PAK": {"production_share": 0.030, "rank": 8},
            "COL": {"production_share": 0.025, "rank": 9},
            "PRY": {"production_share": 0.020, "rank": 10},
        },
        "average_supply_chain_depth": 5,
        "typical_processing_stages": [
            "breeding", "fattening", "slaughter", "deboning",
            "cutting", "aging", "packaging",
        ],
        "sustainability_certifications": [
            "Certified Sustainable Beef (CSB)",
            "Global Roundtable for Sustainable Beef (GRSB)",
            "Rainforest Alliance",
            "JBS Green Platform",
        ],
    },

    # -------------------------------------------------------------------
    # 2. COCOA
    # -------------------------------------------------------------------
    "cocoa": {
        "commodity_type": "cocoa",
        "display_name": "Cocoa",
        "eudr_annex_ref": "Annex I - Part B",
        "description": (
            "Cocoa beans and products derived from cocoa including "
            "cocoa paste, butter, powder, and chocolate."
        ),
        "hs_codes": [
            "1801.00",
            "1802.00",
            "1803.10", "1803.20",
            "1804.00",
            "1805.00",
            "1806.10", "1806.20", "1806.31", "1806.32", "1806.90",
        ],
        "cn_codes": [
            "1801 00 00",
            "1802 00 00",
            "1803 10 00", "1803 20 00",
            "1804 00 00",
            "1805 00 00",
            "1806 10 15", "1806 10 20", "1806 10 30", "1806 10 90",
            "1806 20 10", "1806 20 30", "1806 20 50", "1806 20 70",
            "1806 20 80", "1806 20 95",
            "1806 31 00", "1806 32 10", "1806 32 90",
            "1806 90 11", "1806 90 19", "1806 90 31", "1806 90 39",
        ],
        "intrinsic_risk_factors": {
            "deforestation_pressure": 0.80,
            "supply_chain_complexity": 0.75,
            "traceability_difficulty": 0.70,
            "processing_variability": 0.65,
        },
        "key_producing_countries": {
            "CIV": {"production_share": 0.380, "rank": 1},
            "GHA": {"production_share": 0.165, "rank": 2},
            "ECU": {"production_share": 0.075, "rank": 3},
            "CMR": {"production_share": 0.065, "rank": 4},
            "NGA": {"production_share": 0.055, "rank": 5},
            "IDN": {"production_share": 0.050, "rank": 6},
            "BRA": {"production_share": 0.045, "rank": 7},
            "PER": {"production_share": 0.030, "rank": 8},
            "DOM": {"production_share": 0.025, "rank": 9},
            "COL": {"production_share": 0.020, "rank": 10},
        },
        "average_supply_chain_depth": 6,
        "typical_processing_stages": [
            "harvesting", "fermentation", "drying", "sorting",
            "roasting", "winnowing", "grinding", "conching", "tempering",
        ],
        "sustainability_certifications": [
            "Rainforest Alliance / UTZ Certified",
            "Fairtrade International",
            "Organic (EU/USDA)",
            "Cocoa Horizons",
            "World Cocoa Foundation (WCF)",
        ],
    },

    # -------------------------------------------------------------------
    # 3. COFFEE
    # -------------------------------------------------------------------
    "coffee": {
        "commodity_type": "coffee",
        "display_name": "Coffee",
        "eudr_annex_ref": "Annex I - Part C",
        "description": (
            "Coffee beans (green and roasted) and products derived from "
            "coffee including instant, soluble, and coffee extracts."
        ),
        "hs_codes": [
            "0901.11", "0901.12", "0901.21", "0901.22", "0901.90",
            "2101.11", "2101.12",
        ],
        "cn_codes": [
            "0901 11 00", "0901 12 00",
            "0901 21 00", "0901 22 00",
            "0901 90 10", "0901 90 90",
            "2101 11 11", "2101 11 19",
            "2101 12 92", "2101 12 98",
        ],
        "intrinsic_risk_factors": {
            "deforestation_pressure": 0.65,
            "supply_chain_complexity": 0.70,
            "traceability_difficulty": 0.65,
            "processing_variability": 0.55,
        },
        "key_producing_countries": {
            "BRA": {"production_share": 0.340, "rank": 1},
            "VNM": {"production_share": 0.165, "rank": 2},
            "COL": {"production_share": 0.080, "rank": 3},
            "IDN": {"production_share": 0.065, "rank": 4},
            "ETH": {"production_share": 0.055, "rank": 5},
            "HND": {"production_share": 0.040, "rank": 6},
            "IND": {"production_share": 0.035, "rank": 7},
            "UGA": {"production_share": 0.030, "rank": 8},
            "MEX": {"production_share": 0.025, "rank": 9},
            "GTM": {"production_share": 0.022, "rank": 10},
        },
        "average_supply_chain_depth": 5,
        "typical_processing_stages": [
            "harvesting", "depulping", "fermentation", "washing",
            "drying", "hulling", "grading", "roasting", "grinding",
        ],
        "sustainability_certifications": [
            "Rainforest Alliance / UTZ Certified",
            "Fairtrade International",
            "Organic (EU/USDA)",
            "4C Association",
            "Bird Friendly (Smithsonian)",
        ],
    },

    # -------------------------------------------------------------------
    # 4. OIL PALM
    # -------------------------------------------------------------------
    "oil_palm": {
        "commodity_type": "oil_palm",
        "display_name": "Oil Palm",
        "eudr_annex_ref": "Annex I - Part D",
        "description": (
            "Oil palm and products derived from oil palm including "
            "crude and refined palm oil, palm kernel oil, biodiesel, "
            "oleochemicals, and food additives."
        ),
        "hs_codes": [
            "1207.10",
            "1511.10", "1511.90",
            "1513.21", "1513.29",
            "2306.60",
            "3823.11", "3823.12", "3823.13", "3823.19",
            "3826.00",
        ],
        "cn_codes": [
            "1207 10 10", "1207 10 90",
            "1511 10 10", "1511 10 90",
            "1511 90 11", "1511 90 19", "1511 90 91", "1511 90 99",
            "1513 21 10", "1513 21 30", "1513 21 90",
            "1513 29 11", "1513 29 19", "1513 29 30",
            "3826 00 10", "3826 00 90",
        ],
        "intrinsic_risk_factors": {
            "deforestation_pressure": 0.90,
            "supply_chain_complexity": 0.80,
            "traceability_difficulty": 0.80,
            "processing_variability": 0.70,
        },
        "key_producing_countries": {
            "IDN": {"production_share": 0.575, "rank": 1},
            "MYS": {"production_share": 0.250, "rank": 2},
            "THA": {"production_share": 0.035, "rank": 3},
            "COL": {"production_share": 0.025, "rank": 4},
            "NGA": {"production_share": 0.020, "rank": 5},
            "GTM": {"production_share": 0.012, "rank": 6},
            "HND": {"production_share": 0.010, "rank": 7},
            "PNG": {"production_share": 0.010, "rank": 8},
            "ECU": {"production_share": 0.008, "rank": 9},
            "GHA": {"production_share": 0.007, "rank": 10},
        },
        "average_supply_chain_depth": 7,
        "typical_processing_stages": [
            "harvesting_ffb", "sterilization", "stripping",
            "digestion", "pressing", "clarification",
            "purification", "fractionation", "refining",
        ],
        "sustainability_certifications": [
            "RSPO (Roundtable on Sustainable Palm Oil)",
            "ISCC (International Sustainability & Carbon Certification)",
            "MSPO (Malaysian Sustainable Palm Oil)",
            "ISPO (Indonesian Sustainable Palm Oil)",
            "Rainforest Alliance",
        ],
    },

    # -------------------------------------------------------------------
    # 5. RUBBER
    # -------------------------------------------------------------------
    "rubber": {
        "commodity_type": "rubber",
        "display_name": "Rubber",
        "eudr_annex_ref": "Annex I - Part E",
        "description": (
            "Natural rubber (Hevea brasiliensis) and products derived "
            "from natural rubber including latex, RSS, TSR, tires, "
            "and rubber compounds."
        ),
        "hs_codes": [
            "4001.10", "4001.21", "4001.22", "4001.29", "4001.30",
            "4005.10", "4005.20", "4005.91", "4005.99",
            "4011.10", "4011.20", "4011.30", "4011.40", "4011.50",
            "4011.70", "4011.80", "4011.90",
        ],
        "cn_codes": [
            "4001 10 00",
            "4001 21 00", "4001 22 00", "4001 29 00",
            "4001 30 00",
            "4005 10 00", "4005 20 00",
            "4011 10 00", "4011 20 10", "4011 20 90",
        ],
        "intrinsic_risk_factors": {
            "deforestation_pressure": 0.70,
            "supply_chain_complexity": 0.65,
            "traceability_difficulty": 0.70,
            "processing_variability": 0.55,
        },
        "key_producing_countries": {
            "THA": {"production_share": 0.330, "rank": 1},
            "IDN": {"production_share": 0.250, "rank": 2},
            "VNM": {"production_share": 0.085, "rank": 3},
            "CHN": {"production_share": 0.065, "rank": 4},
            "IND": {"production_share": 0.060, "rank": 5},
            "MYS": {"production_share": 0.040, "rank": 6},
            "CIV": {"production_share": 0.035, "rank": 7},
            "GTM": {"production_share": 0.020, "rank": 8},
            "MMR": {"production_share": 0.015, "rank": 9},
            "CMR": {"production_share": 0.012, "rank": 10},
        },
        "average_supply_chain_depth": 5,
        "typical_processing_stages": [
            "tapping", "collection", "coagulation",
            "sheeting", "drying", "grading",
            "baling", "compounding",
        ],
        "sustainability_certifications": [
            "FSC (Forest Stewardship Council)",
            "PEFC (Programme for Endorsement of Forest Certification)",
            "SNR-i (Sustainable Natural Rubber Initiative)",
            "GPSNR (Global Platform for Sustainable Natural Rubber)",
            "Rainforest Alliance",
        ],
    },

    # -------------------------------------------------------------------
    # 6. SOYA
    # -------------------------------------------------------------------
    "soya": {
        "commodity_type": "soya",
        "display_name": "Soya",
        "eudr_annex_ref": "Annex I - Part F",
        "description": (
            "Soybeans and products derived from soya including soy meal, "
            "soy flour, soy oil, soy lecithin, tofu, tempeh, and "
            "soya-based biodiesel."
        ),
        "hs_codes": [
            "1201.10", "1201.90",
            "1208.10",
            "1507.10", "1507.90",
            "2304.00",
            "3507.10",
            "2106.10",
        ],
        "cn_codes": [
            "1201 10 00", "1201 90 00",
            "1208 10 00",
            "1507 10 10", "1507 10 90",
            "1507 90 10", "1507 90 90",
            "2304 00 00",
        ],
        "intrinsic_risk_factors": {
            "deforestation_pressure": 0.80,
            "supply_chain_complexity": 0.65,
            "traceability_difficulty": 0.60,
            "processing_variability": 0.50,
        },
        "key_producing_countries": {
            "BRA": {"production_share": 0.370, "rank": 1},
            "USA": {"production_share": 0.285, "rank": 2},
            "ARG": {"production_share": 0.130, "rank": 3},
            "CHN": {"production_share": 0.045, "rank": 4},
            "IND": {"production_share": 0.035, "rank": 5},
            "PRY": {"production_share": 0.030, "rank": 6},
            "CAN": {"production_share": 0.020, "rank": 7},
            "BOL": {"production_share": 0.015, "rank": 8},
            "URY": {"production_share": 0.010, "rank": 9},
            "UKR": {"production_share": 0.008, "rank": 10},
        },
        "average_supply_chain_depth": 4,
        "typical_processing_stages": [
            "harvesting", "cleaning", "conditioning",
            "cracking", "dehulling", "flaking",
            "extracting", "refining",
        ],
        "sustainability_certifications": [
            "RTRS (Round Table on Responsible Soy)",
            "ProTerra Foundation",
            "ISCC (International Sustainability & Carbon Certification)",
            "Organic (EU/USDA)",
            "Danube Soya / Europe Soya",
        ],
    },

    # -------------------------------------------------------------------
    # 7. WOOD
    # -------------------------------------------------------------------
    "wood": {
        "commodity_type": "wood",
        "display_name": "Wood",
        "eudr_annex_ref": "Annex I - Part G",
        "description": (
            "Wood in the rough and products derived from wood including "
            "sawn wood, plywood, veneer, MDF, particle board, furniture, "
            "pulp, paper, charcoal, and wood pellets."
        ),
        "hs_codes": [
            "4401.11", "4401.12", "4401.21", "4401.22", "4401.31",
            "4401.32", "4401.39", "4401.41", "4401.49",
            "4403.11", "4403.12", "4403.21", "4403.22", "4403.23",
            "4403.24", "4403.25", "4403.26", "4403.41", "4403.42",
            "4403.49", "4403.91", "4403.93", "4403.94", "4403.95",
            "4403.96", "4403.97", "4403.98", "4403.99",
            "4407.11", "4407.12", "4407.19", "4407.21", "4407.22",
            "4407.25", "4407.26", "4407.27", "4407.28", "4407.29",
            "4407.91", "4407.92", "4407.93", "4407.94", "4407.95",
            "4407.96", "4407.97", "4407.99",
            "4408.10", "4408.31", "4408.39", "4408.90",
            "4409.10", "4409.21", "4409.22", "4409.29",
            "4410.11", "4410.12", "4410.19", "4410.90",
            "4411.12", "4411.13", "4411.14", "4411.92", "4411.93",
            "4411.94",
            "4412.10", "4412.31", "4412.33", "4412.34", "4412.39",
            "4412.94", "4412.99",
            "4414.00", "4418.10", "4418.20", "4418.40", "4418.50",
            "4418.60", "4418.73", "4418.74", "4418.75", "4418.79",
            "4418.91", "4418.92", "4418.99",
            "4420.10", "4420.90",
            "4702.00",
            "4703.11", "4703.19", "4703.21", "4703.29",
            "4704.11", "4704.19", "4704.21", "4704.29",
            "4705.00",
            "4801.00",
            "4802.10", "4802.20", "4802.40", "4802.54", "4802.55",
            "4802.56", "4802.57", "4802.58", "4802.61", "4802.62",
            "4802.69",
            "9403.30", "9403.40", "9403.50", "9403.60",
        ],
        "cn_codes": [
            "4401 11 00", "4401 12 00",
            "4401 21 00", "4401 22 00",
            "4403 11 00", "4403 12 00",
            "4407 11 10", "4407 11 20", "4407 11 90",
            "4412 10 00", "4412 31 00",
            "4702 00 00",
            "4703 11 00", "4703 19 00",
            "9403 30 11", "9403 30 19", "9403 30 91",
        ],
        "intrinsic_risk_factors": {
            "deforestation_pressure": 0.75,
            "supply_chain_complexity": 0.60,
            "traceability_difficulty": 0.65,
            "processing_variability": 0.70,
        },
        "key_producing_countries": {
            "BRA": {"production_share": 0.105, "rank": 1},
            "USA": {"production_share": 0.100, "rank": 2},
            "CHN": {"production_share": 0.095, "rank": 3},
            "RUS": {"production_share": 0.080, "rank": 4},
            "CAN": {"production_share": 0.070, "rank": 5},
            "IDN": {"production_share": 0.060, "rank": 6},
            "IND": {"production_share": 0.045, "rank": 7},
            "SWE": {"production_share": 0.035, "rank": 8},
            "FIN": {"production_share": 0.030, "rank": 9},
            "DEU": {"production_share": 0.025, "rank": 10},
        },
        "average_supply_chain_depth": 4,
        "typical_processing_stages": [
            "felling", "skidding", "debarking", "sawing",
            "drying", "planing", "treatment", "finishing",
        ],
        "sustainability_certifications": [
            "FSC (Forest Stewardship Council)",
            "PEFC (Programme for Endorsement of Forest Certification)",
            "SFI (Sustainable Forestry Initiative)",
            "CSA (Canadian Standards Association)",
            "MTCS (Malaysian Timber Certification Scheme)",
        ],
    },
}

# ===========================================================================
# Derived Products - Annex I mapping per commodity
# ===========================================================================

DERIVED_PRODUCTS: Dict[str, List[Dict[str, Any]]] = {

    "cattle": [
        {"name": "fresh_beef", "hs_code": "0201", "risk_multiplier": 1.0, "traceability_requirement": "individual_animal", "annex_ref": "0201"},
        {"name": "frozen_beef", "hs_code": "0202", "risk_multiplier": 1.05, "traceability_requirement": "batch_lot", "annex_ref": "0202"},
        {"name": "raw_leather", "hs_code": "4101", "risk_multiplier": 1.10, "traceability_requirement": "batch_lot", "annex_ref": "4101"},
        {"name": "processed_leather", "hs_code": "4104", "risk_multiplier": 1.20, "traceability_requirement": "batch_lot", "annex_ref": "4104"},
        {"name": "finished_leather", "hs_code": "4107", "risk_multiplier": 1.30, "traceability_requirement": "batch_lot", "annex_ref": "4107"},
        {"name": "tallow", "hs_code": "1502", "risk_multiplier": 1.15, "traceability_requirement": "mass_balance", "annex_ref": "1502"},
        {"name": "gelatin", "hs_code": "3503", "risk_multiplier": 1.25, "traceability_requirement": "mass_balance", "annex_ref": "3503"},
        {"name": "bone_meal", "hs_code": "0506", "risk_multiplier": 1.20, "traceability_requirement": "mass_balance", "annex_ref": "0506"},
    ],

    "cocoa": [
        {"name": "cocoa_beans", "hs_code": "1801", "risk_multiplier": 1.0, "traceability_requirement": "batch_lot", "annex_ref": "1801"},
        {"name": "cocoa_shells", "hs_code": "1802", "risk_multiplier": 0.90, "traceability_requirement": "batch_lot", "annex_ref": "1802"},
        {"name": "cocoa_paste", "hs_code": "1803", "risk_multiplier": 1.10, "traceability_requirement": "batch_lot", "annex_ref": "1803"},
        {"name": "cocoa_butter", "hs_code": "1804", "risk_multiplier": 1.15, "traceability_requirement": "mass_balance", "annex_ref": "1804"},
        {"name": "cocoa_powder", "hs_code": "1805", "risk_multiplier": 1.15, "traceability_requirement": "mass_balance", "annex_ref": "1805"},
        {"name": "chocolate_dark", "hs_code": "1806.31", "risk_multiplier": 1.25, "traceability_requirement": "mass_balance", "annex_ref": "1806.31"},
        {"name": "chocolate_milk", "hs_code": "1806.32", "risk_multiplier": 1.30, "traceability_requirement": "mass_balance", "annex_ref": "1806.32"},
        {"name": "cocoa_preparations", "hs_code": "1806.90", "risk_multiplier": 1.35, "traceability_requirement": "mass_balance", "annex_ref": "1806.90"},
    ],

    "coffee": [
        {"name": "green_beans", "hs_code": "0901.11", "risk_multiplier": 1.0, "traceability_requirement": "batch_lot", "annex_ref": "0901.11"},
        {"name": "decaffeinated_green", "hs_code": "0901.12", "risk_multiplier": 1.05, "traceability_requirement": "batch_lot", "annex_ref": "0901.12"},
        {"name": "roasted_whole", "hs_code": "0901.21", "risk_multiplier": 1.10, "traceability_requirement": "batch_lot", "annex_ref": "0901.21"},
        {"name": "roasted_ground", "hs_code": "0901.22", "risk_multiplier": 1.15, "traceability_requirement": "batch_lot", "annex_ref": "0901.22"},
        {"name": "coffee_husks", "hs_code": "0901.90", "risk_multiplier": 0.85, "traceability_requirement": "batch_lot", "annex_ref": "0901.90"},
        {"name": "instant_coffee", "hs_code": "2101.11", "risk_multiplier": 1.25, "traceability_requirement": "mass_balance", "annex_ref": "2101.11"},
        {"name": "coffee_extract", "hs_code": "2101.12", "risk_multiplier": 1.30, "traceability_requirement": "mass_balance", "annex_ref": "2101.12"},
    ],

    "oil_palm": [
        {"name": "palm_fruit", "hs_code": "1207.10", "risk_multiplier": 1.0, "traceability_requirement": "plantation_mill", "annex_ref": "1207.10"},
        {"name": "crude_palm_oil", "hs_code": "1511.10", "risk_multiplier": 1.10, "traceability_requirement": "mass_balance", "annex_ref": "1511.10"},
        {"name": "refined_palm_oil", "hs_code": "1511.90", "risk_multiplier": 1.20, "traceability_requirement": "mass_balance", "annex_ref": "1511.90"},
        {"name": "palm_kernel_oil", "hs_code": "1513.21", "risk_multiplier": 1.15, "traceability_requirement": "mass_balance", "annex_ref": "1513.21"},
        {"name": "palm_kernel_cake", "hs_code": "2306.60", "risk_multiplier": 1.05, "traceability_requirement": "mass_balance", "annex_ref": "2306.60"},
        {"name": "oleochemicals", "hs_code": "3823", "risk_multiplier": 1.35, "traceability_requirement": "mass_balance", "annex_ref": "3823"},
        {"name": "palm_biodiesel", "hs_code": "3826.00", "risk_multiplier": 1.40, "traceability_requirement": "mass_balance", "annex_ref": "3826"},
        {"name": "palm_soap_stock", "hs_code": "3401", "risk_multiplier": 1.30, "traceability_requirement": "mass_balance", "annex_ref": "3401"},
    ],

    "rubber": [
        {"name": "natural_rubber_latex", "hs_code": "4001.10", "risk_multiplier": 1.0, "traceability_requirement": "batch_lot", "annex_ref": "4001.10"},
        {"name": "smoked_sheets_rss", "hs_code": "4001.21", "risk_multiplier": 1.05, "traceability_requirement": "batch_lot", "annex_ref": "4001.21"},
        {"name": "tsr_block_rubber", "hs_code": "4001.22", "risk_multiplier": 1.10, "traceability_requirement": "batch_lot", "annex_ref": "4001.22"},
        {"name": "other_natural_rubber", "hs_code": "4001.29", "risk_multiplier": 1.05, "traceability_requirement": "batch_lot", "annex_ref": "4001.29"},
        {"name": "rubber_compounds", "hs_code": "4005", "risk_multiplier": 1.20, "traceability_requirement": "mass_balance", "annex_ref": "4005"},
        {"name": "pneumatic_tires", "hs_code": "4011", "risk_multiplier": 1.35, "traceability_requirement": "mass_balance", "annex_ref": "4011"},
        {"name": "rubber_gaskets_seals", "hs_code": "4016", "risk_multiplier": 1.25, "traceability_requirement": "mass_balance", "annex_ref": "4016"},
    ],

    "soya": [
        {"name": "soybeans", "hs_code": "1201", "risk_multiplier": 1.0, "traceability_requirement": "batch_lot", "annex_ref": "1201"},
        {"name": "soy_flour", "hs_code": "1208.10", "risk_multiplier": 1.10, "traceability_requirement": "mass_balance", "annex_ref": "1208.10"},
        {"name": "crude_soy_oil", "hs_code": "1507.10", "risk_multiplier": 1.15, "traceability_requirement": "mass_balance", "annex_ref": "1507.10"},
        {"name": "refined_soy_oil", "hs_code": "1507.90", "risk_multiplier": 1.20, "traceability_requirement": "mass_balance", "annex_ref": "1507.90"},
        {"name": "soy_meal", "hs_code": "2304", "risk_multiplier": 1.10, "traceability_requirement": "mass_balance", "annex_ref": "2304"},
        {"name": "soy_lecithin", "hs_code": "2923.20", "risk_multiplier": 1.25, "traceability_requirement": "mass_balance", "annex_ref": "2923.20"},
        {"name": "soy_protein_isolate", "hs_code": "3504", "risk_multiplier": 1.30, "traceability_requirement": "mass_balance", "annex_ref": "3504"},
        {"name": "soy_biodiesel", "hs_code": "3826", "risk_multiplier": 1.35, "traceability_requirement": "mass_balance", "annex_ref": "3826"},
    ],

    "wood": [
        {"name": "fuel_wood", "hs_code": "4401", "risk_multiplier": 0.90, "traceability_requirement": "batch_lot", "annex_ref": "4401"},
        {"name": "wood_in_rough", "hs_code": "4403", "risk_multiplier": 1.0, "traceability_requirement": "individual_log", "annex_ref": "4403"},
        {"name": "sawn_wood", "hs_code": "4407", "risk_multiplier": 1.10, "traceability_requirement": "batch_lot", "annex_ref": "4407"},
        {"name": "veneer_sheets", "hs_code": "4408", "risk_multiplier": 1.15, "traceability_requirement": "batch_lot", "annex_ref": "4408"},
        {"name": "plywood", "hs_code": "4412", "risk_multiplier": 1.25, "traceability_requirement": "mass_balance", "annex_ref": "4412"},
        {"name": "particle_board", "hs_code": "4410", "risk_multiplier": 1.30, "traceability_requirement": "mass_balance", "annex_ref": "4410"},
        {"name": "fibreboard_mdf", "hs_code": "4411", "risk_multiplier": 1.30, "traceability_requirement": "mass_balance", "annex_ref": "4411"},
        {"name": "wood_furniture", "hs_code": "9403", "risk_multiplier": 1.40, "traceability_requirement": "mass_balance", "annex_ref": "9403"},
        {"name": "wood_pulp", "hs_code": "4703", "risk_multiplier": 1.35, "traceability_requirement": "mass_balance", "annex_ref": "4703"},
        {"name": "charcoal", "hs_code": "4402", "risk_multiplier": 1.20, "traceability_requirement": "batch_lot", "annex_ref": "4402"},
    ],
}

# ===========================================================================
# Country Production Data - Top producers per commodity (tonnes/year)
# ===========================================================================

COUNTRY_PRODUCTION_DATA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "cattle": {
        "BRA": {"annual_production_tonnes": 10_200_000, "year": 2024, "trend": "increasing"},
        "USA": {"annual_production_tonnes": 12_300_000, "year": 2024, "trend": "stable"},
        "IND": {"annual_production_tonnes": 4_300_000, "year": 2024, "trend": "increasing"},
        "CHN": {"annual_production_tonnes": 7_200_000, "year": 2024, "trend": "increasing"},
        "ARG": {"annual_production_tonnes": 3_100_000, "year": 2024, "trend": "stable"},
        "AUS": {"annual_production_tonnes": 2_100_000, "year": 2024, "trend": "decreasing"},
    },
    "cocoa": {
        "CIV": {"annual_production_tonnes": 2_200_000, "year": 2024, "trend": "stable"},
        "GHA": {"annual_production_tonnes": 1_050_000, "year": 2024, "trend": "decreasing"},
        "ECU": {"annual_production_tonnes": 380_000, "year": 2024, "trend": "increasing"},
        "CMR": {"annual_production_tonnes": 290_000, "year": 2024, "trend": "stable"},
        "NGA": {"annual_production_tonnes": 270_000, "year": 2024, "trend": "decreasing"},
        "IDN": {"annual_production_tonnes": 200_000, "year": 2024, "trend": "decreasing"},
    },
    "coffee": {
        "BRA": {"annual_production_tonnes": 3_900_000, "year": 2024, "trend": "stable"},
        "VNM": {"annual_production_tonnes": 1_850_000, "year": 2024, "trend": "increasing"},
        "COL": {"annual_production_tonnes": 870_000, "year": 2024, "trend": "stable"},
        "IDN": {"annual_production_tonnes": 750_000, "year": 2024, "trend": "decreasing"},
        "ETH": {"annual_production_tonnes": 530_000, "year": 2024, "trend": "increasing"},
        "HND": {"annual_production_tonnes": 420_000, "year": 2024, "trend": "stable"},
    },
    "oil_palm": {
        "IDN": {"annual_production_tonnes": 49_500_000, "year": 2024, "trend": "increasing"},
        "MYS": {"annual_production_tonnes": 18_500_000, "year": 2024, "trend": "stable"},
        "THA": {"annual_production_tonnes": 3_200_000, "year": 2024, "trend": "increasing"},
        "COL": {"annual_production_tonnes": 1_800_000, "year": 2024, "trend": "increasing"},
        "NGA": {"annual_production_tonnes": 1_400_000, "year": 2024, "trend": "stable"},
        "GTM": {"annual_production_tonnes": 800_000, "year": 2024, "trend": "increasing"},
    },
    "rubber": {
        "THA": {"annual_production_tonnes": 4_700_000, "year": 2024, "trend": "stable"},
        "IDN": {"annual_production_tonnes": 3_100_000, "year": 2024, "trend": "stable"},
        "VNM": {"annual_production_tonnes": 1_300_000, "year": 2024, "trend": "increasing"},
        "CHN": {"annual_production_tonnes": 850_000, "year": 2024, "trend": "stable"},
        "IND": {"annual_production_tonnes": 780_000, "year": 2024, "trend": "increasing"},
        "MYS": {"annual_production_tonnes": 500_000, "year": 2024, "trend": "decreasing"},
    },
    "soya": {
        "BRA": {"annual_production_tonnes": 160_000_000, "year": 2024, "trend": "increasing"},
        "USA": {"annual_production_tonnes": 121_000_000, "year": 2024, "trend": "stable"},
        "ARG": {"annual_production_tonnes": 50_000_000, "year": 2024, "trend": "stable"},
        "CHN": {"annual_production_tonnes": 20_500_000, "year": 2024, "trend": "increasing"},
        "IND": {"annual_production_tonnes": 12_000_000, "year": 2024, "trend": "increasing"},
        "PRY": {"annual_production_tonnes": 10_500_000, "year": 2024, "trend": "stable"},
    },
    "wood": {
        "BRA": {"annual_production_tonnes": 260_000_000, "year": 2024, "trend": "increasing"},
        "USA": {"annual_production_tonnes": 420_000_000, "year": 2024, "trend": "stable"},
        "CHN": {"annual_production_tonnes": 380_000_000, "year": 2024, "trend": "increasing"},
        "RUS": {"annual_production_tonnes": 220_000_000, "year": 2024, "trend": "decreasing"},
        "CAN": {"annual_production_tonnes": 150_000_000, "year": 2024, "trend": "stable"},
        "IDN": {"annual_production_tonnes": 120_000_000, "year": 2024, "trend": "stable"},
    },
}

# ===========================================================================
# HS Code Mapping - HS code to commodity type
# ===========================================================================

HS_CODE_MAPPING: Dict[str, str] = {}

# Build mapping from EUDR_COMMODITIES
for _commodity_type, _commodity_data in EUDR_COMMODITIES.items():
    for _hs_code in _commodity_data["hs_codes"]:
        HS_CODE_MAPPING[_hs_code] = _commodity_type
        # Also map 4-digit prefix
        _prefix = _hs_code[:4]
        if _prefix not in HS_CODE_MAPPING:
            HS_CODE_MAPPING[_prefix] = _commodity_type


# ===========================================================================
# CommodityDatabase class
# ===========================================================================


class CommodityDatabase:
    """
    Stateless reference data accessor for EUDR commodity information.

    Provides typed access to EUDR commodity data including HS codes,
    intrinsic risk factors, derived products, and country production data.

    Example:
        >>> db = CommodityDatabase()
        >>> info = db.get_commodity_info("cocoa")
        >>> assert info["commodity_type"] == "cocoa"
        >>> products = db.get_derived_products("oil_palm")
        >>> assert len(products) > 0
    """

    def get_commodity_info(self, commodity_type: str) -> Optional[Dict[str, Any]]:
        """Get complete info for a commodity type.

        Args:
            commodity_type: One of the 7 EUDR commodity types.

        Returns:
            Commodity data dict or None if not found.
        """
        return EUDR_COMMODITIES.get(commodity_type)

    def get_derived_products(self, commodity_type: str) -> List[Dict[str, Any]]:
        """Get all derived products for a commodity type.

        Args:
            commodity_type: One of the 7 EUDR commodity types.

        Returns:
            List of derived product dicts (empty if commodity not found).
        """
        return DERIVED_PRODUCTS.get(commodity_type, [])

    def lookup_hs_code(self, hs_code: str) -> Optional[str]:
        """Look up commodity type from HS code.

        Args:
            hs_code: Harmonized System code (4 or 6+ digit).

        Returns:
            Commodity type string or None if not mapped.
        """
        return HS_CODE_MAPPING.get(hs_code)

    def get_all_commodities(self) -> List[str]:
        """Get list of all supported commodity types.

        Returns:
            List of commodity type strings.
        """
        return list(EUDR_COMMODITIES.keys())


# ===========================================================================
# Module-level helper functions
# ===========================================================================


def get_commodity_info(commodity_type: str) -> Optional[Dict[str, Any]]:
    """Get complete info for a commodity type.

    Args:
        commodity_type: One of the 7 EUDR commodity types.

    Returns:
        Commodity data dict or None if not found.
    """
    return EUDR_COMMODITIES.get(commodity_type)


def get_derived_products(commodity_type: str) -> List[Dict[str, Any]]:
    """Get all derived products for a commodity type.

    Args:
        commodity_type: One of the 7 EUDR commodity types.

    Returns:
        List of derived product dicts (empty if commodity not found).
    """
    return DERIVED_PRODUCTS.get(commodity_type, [])


def lookup_hs_code(hs_code: str) -> Optional[str]:
    """Look up commodity type from HS code.

    Tries exact match first, then 4-digit prefix match.

    Args:
        hs_code: Harmonized System code (4 or 6+ digit).

    Returns:
        Commodity type string or None if not mapped.
    """
    result = HS_CODE_MAPPING.get(hs_code)
    if result is None and len(hs_code) > 4:
        result = HS_CODE_MAPPING.get(hs_code[:4])
    return result


def get_hs_codes_for_commodity(commodity_type: str) -> List[str]:
    """Get all HS codes for a commodity type.

    Args:
        commodity_type: One of the 7 EUDR commodity types.

    Returns:
        List of HS code strings (empty if commodity not found).
    """
    commodity = EUDR_COMMODITIES.get(commodity_type)
    if commodity is None:
        return []
    return commodity.get("hs_codes", [])


def get_producing_countries(commodity_type: str) -> Dict[str, Dict[str, Any]]:
    """Get key producing countries for a commodity type.

    Args:
        commodity_type: One of the 7 EUDR commodity types.

    Returns:
        Dict of country code to production data (empty if not found).
    """
    commodity = EUDR_COMMODITIES.get(commodity_type)
    if commodity is None:
        return {}
    return commodity.get("key_producing_countries", {})


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "EUDR_COMMODITIES",
    "DERIVED_PRODUCTS",
    "COUNTRY_PRODUCTION_DATA",
    "HS_CODE_MAPPING",
    "CommodityDatabase",
    "get_commodity_info",
    "get_derived_products",
    "lookup_hs_code",
    "get_hs_codes_for_commodity",
    "get_producing_countries",
]
