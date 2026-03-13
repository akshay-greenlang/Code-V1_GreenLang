# -*- coding: utf-8 -*-
"""
Processing Chains - AGENT-EUDR-018 Commodity Risk Analyzer

Processing chain definitions for all 7 EUDR-regulated commodities describing
transformation pathways from raw commodity to final derived product. Each
chain includes per-stage metadata: name, typical location, risk addition,
traceability loss, waste percentage, and input-to-output transformation ratios.

Processing chains are critical for:
    - Traceability scoring: Calculating cumulative traceability loss across
      processing stages to determine how verifiable the origin claim is
    - Risk multiplication: Each processing stage adds incremental risk due
      to mixing, co-processing, and intermediary handling
    - Transformation ratios: Mass balance verification (e.g., 10 tonnes of
      fresh fruit bunches -> 2 tonnes of crude palm oil)
    - Waste tracking: Monitoring material loss at each stage for
      reconciliation with declared quantities

Chains covered:
    - Cattle: Farm -> Slaughterhouse -> Deboning -> Cutting -> Products
    - Cocoa: Farm -> Fermentation -> Drying -> Roasting -> Grinding -> Chocolate
    - Coffee: Farm -> Depulping -> Drying -> Hulling -> Roasting -> Products
    - Oil Palm: Plantation -> Mill -> Refinery -> Fractionation -> Products
    - Rubber: Plantation -> Collection -> Processing -> Grading -> Products
    - Soya: Farm -> Cleaning -> Crushing -> Extraction -> Products
    - Wood: Forest -> Sawmill -> Kiln Drying -> Processing -> Products

Data Sources:
    - FAO Commodity Processing Guides
    - UNIDO Industrial Processing Standards
    - European Commission Best Available Techniques (BAT)
    - Industry-specific processing manuals

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
    "FAO Commodity Processing Technical Guides 2024",
    "UNIDO Industrial Processing Standards",
    "European Commission BAT Reference Documents",
    "RSPO Supply Chain Certification Standard 2023",
    "FSC Chain of Custody Standard (FSC-STD-40-004 V3-1)",
]

# ===========================================================================
# Processing Chains - Transformation pathways per commodity
# ===========================================================================

PROCESSING_CHAINS: Dict[str, Dict[str, Any]] = {

    # -------------------------------------------------------------------
    # CATTLE Processing Chains
    # -------------------------------------------------------------------
    "cattle": {
        "commodity_type": "cattle",
        "chains": {
            "beef_fresh": {
                "source_commodity": "cattle",
                "final_product": "fresh_beef",
                "overall_transformation_ratio": 0.42,
                "overall_traceability_retention": 0.70,
                "stages": [
                    {"name": "breeding", "typical_location": "farm", "risk_addition": 0.05, "traceability_loss": 0.02, "waste_pct": 0.0, "duration_days": 730},
                    {"name": "fattening", "typical_location": "feedlot", "risk_addition": 0.08, "traceability_loss": 0.05, "waste_pct": 0.0, "duration_days": 180},
                    {"name": "transport_to_slaughter", "typical_location": "transit", "risk_addition": 0.10, "traceability_loss": 0.05, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "slaughter", "typical_location": "abattoir", "risk_addition": 0.05, "traceability_loss": 0.03, "waste_pct": 0.38, "duration_days": 1},
                    {"name": "deboning", "typical_location": "processing_plant", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.15, "duration_days": 1},
                    {"name": "cutting", "typical_location": "processing_plant", "risk_addition": 0.02, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "aging", "typical_location": "cold_storage", "risk_addition": 0.01, "traceability_loss": 0.02, "waste_pct": 0.02, "duration_days": 21},
                    {"name": "packaging", "typical_location": "packing_facility", "risk_addition": 0.01, "traceability_loss": 0.03, "waste_pct": 0.01, "duration_days": 1},
                ],
            },
            "leather_processed": {
                "source_commodity": "cattle",
                "final_product": "processed_leather",
                "overall_transformation_ratio": 0.065,
                "overall_traceability_retention": 0.50,
                "stages": [
                    {"name": "slaughter", "typical_location": "abattoir", "risk_addition": 0.05, "traceability_loss": 0.03, "waste_pct": 0.38, "duration_days": 1},
                    {"name": "hide_removal", "typical_location": "abattoir", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "salting_preservation", "typical_location": "hide_warehouse", "risk_addition": 0.05, "traceability_loss": 0.08, "waste_pct": 0.02, "duration_days": 30},
                    {"name": "soaking_liming", "typical_location": "tannery", "risk_addition": 0.08, "traceability_loss": 0.10, "waste_pct": 0.15, "duration_days": 3},
                    {"name": "tanning", "typical_location": "tannery", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.10, "duration_days": 14},
                    {"name": "finishing", "typical_location": "tannery", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 7},
                ],
            },
        },
    },

    # -------------------------------------------------------------------
    # COCOA Processing Chains
    # -------------------------------------------------------------------
    "cocoa": {
        "commodity_type": "cocoa",
        "chains": {
            "chocolate_dark": {
                "source_commodity": "cocoa",
                "final_product": "dark_chocolate",
                "overall_transformation_ratio": 0.35,
                "overall_traceability_retention": 0.45,
                "stages": [
                    {"name": "harvesting", "typical_location": "plantation", "risk_addition": 0.03, "traceability_loss": 0.02, "waste_pct": 0.60, "duration_days": 1},
                    {"name": "fermentation", "typical_location": "farm_or_cooperative", "risk_addition": 0.08, "traceability_loss": 0.08, "waste_pct": 0.05, "duration_days": 6},
                    {"name": "drying", "typical_location": "farm_or_cooperative", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.10, "duration_days": 7},
                    {"name": "sorting_grading", "typical_location": "warehouse", "risk_addition": 0.05, "traceability_loss": 0.08, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "roasting", "typical_location": "processing_plant", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "winnowing", "typical_location": "processing_plant", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.12, "duration_days": 1},
                    {"name": "grinding_liquor", "typical_location": "processing_plant", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "pressing_butter", "typical_location": "processing_plant", "risk_addition": 0.03, "traceability_loss": 0.08, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "conching", "typical_location": "chocolate_factory", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.02, "duration_days": 3},
                    {"name": "tempering", "typical_location": "chocolate_factory", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.01, "duration_days": 1},
                ],
            },
            "cocoa_butter": {
                "source_commodity": "cocoa",
                "final_product": "cocoa_butter",
                "overall_transformation_ratio": 0.22,
                "overall_traceability_retention": 0.50,
                "stages": [
                    {"name": "harvesting", "typical_location": "plantation", "risk_addition": 0.03, "traceability_loss": 0.02, "waste_pct": 0.60, "duration_days": 1},
                    {"name": "fermentation", "typical_location": "farm_or_cooperative", "risk_addition": 0.08, "traceability_loss": 0.08, "waste_pct": 0.05, "duration_days": 6},
                    {"name": "drying", "typical_location": "farm_or_cooperative", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.10, "duration_days": 7},
                    {"name": "roasting", "typical_location": "processing_plant", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "winnowing", "typical_location": "processing_plant", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.12, "duration_days": 1},
                    {"name": "grinding", "typical_location": "processing_plant", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "pressing", "typical_location": "processing_plant", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.45, "duration_days": 1},
                ],
            },
        },
    },

    # -------------------------------------------------------------------
    # COFFEE Processing Chains
    # -------------------------------------------------------------------
    "coffee": {
        "commodity_type": "coffee",
        "chains": {
            "roasted_ground": {
                "source_commodity": "coffee",
                "final_product": "roasted_ground_coffee",
                "overall_transformation_ratio": 0.18,
                "overall_traceability_retention": 0.55,
                "stages": [
                    {"name": "harvesting", "typical_location": "farm", "risk_addition": 0.03, "traceability_loss": 0.02, "waste_pct": 0.50, "duration_days": 1},
                    {"name": "depulping", "typical_location": "wet_mill", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.40, "duration_days": 1},
                    {"name": "fermentation", "typical_location": "wet_mill", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.02, "duration_days": 2},
                    {"name": "washing", "typical_location": "wet_mill", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.01, "duration_days": 1},
                    {"name": "drying", "typical_location": "farm_or_station", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.12, "duration_days": 14},
                    {"name": "hulling", "typical_location": "dry_mill", "risk_addition": 0.05, "traceability_loss": 0.08, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "grading_sorting", "typical_location": "dry_mill", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.08, "duration_days": 1},
                    {"name": "roasting", "typical_location": "roastery", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.15, "duration_days": 1},
                    {"name": "grinding", "typical_location": "roastery", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.02, "duration_days": 1},
                ],
            },
            "instant_coffee": {
                "source_commodity": "coffee",
                "final_product": "instant_soluble_coffee",
                "overall_transformation_ratio": 0.10,
                "overall_traceability_retention": 0.35,
                "stages": [
                    {"name": "harvesting", "typical_location": "farm", "risk_addition": 0.03, "traceability_loss": 0.02, "waste_pct": 0.50, "duration_days": 1},
                    {"name": "processing_green", "typical_location": "mill", "risk_addition": 0.05, "traceability_loss": 0.08, "waste_pct": 0.20, "duration_days": 7},
                    {"name": "roasting", "typical_location": "factory", "risk_addition": 0.05, "traceability_loss": 0.08, "waste_pct": 0.15, "duration_days": 1},
                    {"name": "grinding", "typical_location": "factory", "risk_addition": 0.02, "traceability_loss": 0.05, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "extraction", "typical_location": "factory", "risk_addition": 0.08, "traceability_loss": 0.12, "waste_pct": 0.40, "duration_days": 1},
                    {"name": "concentration", "typical_location": "factory", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.10, "duration_days": 1},
                    {"name": "spray_drying", "typical_location": "factory", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.05, "duration_days": 1},
                ],
            },
        },
    },

    # -------------------------------------------------------------------
    # OIL PALM Processing Chains
    # -------------------------------------------------------------------
    "oil_palm": {
        "commodity_type": "oil_palm",
        "chains": {
            "crude_palm_oil": {
                "source_commodity": "oil_palm",
                "final_product": "crude_palm_oil",
                "overall_transformation_ratio": 0.21,
                "overall_traceability_retention": 0.55,
                "stages": [
                    {"name": "harvesting_ffb", "typical_location": "plantation", "risk_addition": 0.05, "traceability_loss": 0.03, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "transport_to_mill", "typical_location": "transit", "risk_addition": 0.08, "traceability_loss": 0.08, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "sterilization", "typical_location": "palm_oil_mill", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "stripping", "typical_location": "palm_oil_mill", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.25, "duration_days": 1},
                    {"name": "digestion", "typical_location": "palm_oil_mill", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "pressing", "typical_location": "palm_oil_mill", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.35, "duration_days": 1},
                    {"name": "clarification", "typical_location": "palm_oil_mill", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.10, "duration_days": 1},
                    {"name": "purification", "typical_location": "palm_oil_mill", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.03, "duration_days": 1},
                ],
            },
            "refined_palm_oil": {
                "source_commodity": "oil_palm",
                "final_product": "refined_palm_oil",
                "overall_transformation_ratio": 0.19,
                "overall_traceability_retention": 0.35,
                "stages": [
                    {"name": "harvesting_ffb", "typical_location": "plantation", "risk_addition": 0.05, "traceability_loss": 0.03, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "milling", "typical_location": "palm_oil_mill", "risk_addition": 0.08, "traceability_loss": 0.10, "waste_pct": 0.79, "duration_days": 1},
                    {"name": "degumming", "typical_location": "refinery", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "bleaching", "typical_location": "refinery", "risk_addition": 0.03, "traceability_loss": 0.08, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "deodorization", "typical_location": "refinery", "risk_addition": 0.03, "traceability_loss": 0.08, "waste_pct": 0.03, "duration_days": 1},
                    {"name": "fractionation", "typical_location": "refinery", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.05, "duration_days": 1},
                ],
            },
        },
    },

    # -------------------------------------------------------------------
    # RUBBER Processing Chains
    # -------------------------------------------------------------------
    "rubber": {
        "commodity_type": "rubber",
        "chains": {
            "smoked_sheets": {
                "source_commodity": "rubber",
                "final_product": "ribbed_smoked_sheets",
                "overall_transformation_ratio": 0.30,
                "overall_traceability_retention": 0.65,
                "stages": [
                    {"name": "tapping", "typical_location": "plantation", "risk_addition": 0.03, "traceability_loss": 0.02, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "collection", "typical_location": "plantation", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "coagulation", "typical_location": "processing_center", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.55, "duration_days": 1},
                    {"name": "milling_sheeting", "typical_location": "processing_center", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "smoking_drying", "typical_location": "smokehouse", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 7},
                    {"name": "grading_baling", "typical_location": "warehouse", "risk_addition": 0.03, "traceability_loss": 0.08, "waste_pct": 0.02, "duration_days": 1},
                ],
            },
            "tires": {
                "source_commodity": "rubber",
                "final_product": "pneumatic_tires",
                "overall_transformation_ratio": 0.15,
                "overall_traceability_retention": 0.25,
                "stages": [
                    {"name": "tapping", "typical_location": "plantation", "risk_addition": 0.03, "traceability_loss": 0.02, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "collection_processing", "typical_location": "factory", "risk_addition": 0.08, "traceability_loss": 0.10, "waste_pct": 0.60, "duration_days": 3},
                    {"name": "compounding", "typical_location": "tire_factory", "risk_addition": 0.10, "traceability_loss": 0.15, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "calendering", "typical_location": "tire_factory", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.03, "duration_days": 1},
                    {"name": "building", "typical_location": "tire_factory", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "vulcanization", "typical_location": "tire_factory", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "inspection", "typical_location": "tire_factory", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.05, "duration_days": 1},
                ],
            },
        },
    },

    # -------------------------------------------------------------------
    # SOYA Processing Chains
    # -------------------------------------------------------------------
    "soya": {
        "commodity_type": "soya",
        "chains": {
            "soy_oil": {
                "source_commodity": "soya",
                "final_product": "refined_soy_oil",
                "overall_transformation_ratio": 0.18,
                "overall_traceability_retention": 0.40,
                "stages": [
                    {"name": "harvesting", "typical_location": "farm", "risk_addition": 0.03, "traceability_loss": 0.02, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "cleaning", "typical_location": "silo", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "conditioning", "typical_location": "crushing_plant", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.01, "duration_days": 1},
                    {"name": "cracking_dehulling", "typical_location": "crushing_plant", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.08, "duration_days": 1},
                    {"name": "flaking", "typical_location": "crushing_plant", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.01, "duration_days": 1},
                    {"name": "solvent_extraction", "typical_location": "crushing_plant", "risk_addition": 0.08, "traceability_loss": 0.10, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "degumming", "typical_location": "refinery", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.03, "duration_days": 1},
                    {"name": "refining", "typical_location": "refinery", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.05, "duration_days": 1},
                ],
            },
            "soy_meal": {
                "source_commodity": "soya",
                "final_product": "soy_meal_animal_feed",
                "overall_transformation_ratio": 0.78,
                "overall_traceability_retention": 0.45,
                "stages": [
                    {"name": "harvesting", "typical_location": "farm", "risk_addition": 0.03, "traceability_loss": 0.02, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "cleaning", "typical_location": "silo", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "crushing", "typical_location": "crushing_plant", "risk_addition": 0.05, "traceability_loss": 0.08, "waste_pct": 0.08, "duration_days": 1},
                    {"name": "solvent_extraction", "typical_location": "crushing_plant", "risk_addition": 0.08, "traceability_loss": 0.10, "waste_pct": 0.18, "duration_days": 1},
                    {"name": "desolventizing", "typical_location": "crushing_plant", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.02, "duration_days": 1},
                    {"name": "toasting_grinding", "typical_location": "crushing_plant", "risk_addition": 0.03, "traceability_loss": 0.08, "waste_pct": 0.02, "duration_days": 1},
                ],
            },
        },
    },

    # -------------------------------------------------------------------
    # WOOD Processing Chains
    # -------------------------------------------------------------------
    "wood": {
        "commodity_type": "wood",
        "chains": {
            "sawn_timber": {
                "source_commodity": "wood",
                "final_product": "sawn_timber",
                "overall_transformation_ratio": 0.50,
                "overall_traceability_retention": 0.70,
                "stages": [
                    {"name": "felling", "typical_location": "forest", "risk_addition": 0.05, "traceability_loss": 0.02, "waste_pct": 0.10, "duration_days": 1},
                    {"name": "skidding_transport", "typical_location": "forest_road", "risk_addition": 0.08, "traceability_loss": 0.05, "waste_pct": 0.0, "duration_days": 3},
                    {"name": "debarking", "typical_location": "sawmill", "risk_addition": 0.03, "traceability_loss": 0.03, "waste_pct": 0.10, "duration_days": 1},
                    {"name": "sawing", "typical_location": "sawmill", "risk_addition": 0.05, "traceability_loss": 0.05, "waste_pct": 0.25, "duration_days": 1},
                    {"name": "kiln_drying", "typical_location": "sawmill", "risk_addition": 0.03, "traceability_loss": 0.03, "waste_pct": 0.05, "duration_days": 14},
                    {"name": "planing", "typical_location": "sawmill", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "grading", "typical_location": "sawmill", "risk_addition": 0.02, "traceability_loss": 0.03, "waste_pct": 0.05, "duration_days": 1},
                ],
            },
            "plywood": {
                "source_commodity": "wood",
                "final_product": "plywood_sheets",
                "overall_transformation_ratio": 0.40,
                "overall_traceability_retention": 0.45,
                "stages": [
                    {"name": "felling", "typical_location": "forest", "risk_addition": 0.05, "traceability_loss": 0.02, "waste_pct": 0.10, "duration_days": 1},
                    {"name": "log_conditioning", "typical_location": "plywood_mill", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.0, "duration_days": 2},
                    {"name": "peeling_rotary", "typical_location": "plywood_mill", "risk_addition": 0.05, "traceability_loss": 0.08, "waste_pct": 0.15, "duration_days": 1},
                    {"name": "drying_veneer", "typical_location": "plywood_mill", "risk_addition": 0.03, "traceability_loss": 0.05, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "gluing", "typical_location": "plywood_mill", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.0, "duration_days": 1},
                    {"name": "pressing", "typical_location": "plywood_mill", "risk_addition": 0.03, "traceability_loss": 0.08, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "trimming_sanding", "typical_location": "plywood_mill", "risk_addition": 0.02, "traceability_loss": 0.05, "waste_pct": 0.10, "duration_days": 1},
                ],
            },
            "wood_pulp": {
                "source_commodity": "wood",
                "final_product": "chemical_wood_pulp",
                "overall_transformation_ratio": 0.45,
                "overall_traceability_retention": 0.30,
                "stages": [
                    {"name": "felling", "typical_location": "forest", "risk_addition": 0.05, "traceability_loss": 0.02, "waste_pct": 0.10, "duration_days": 1},
                    {"name": "chipping", "typical_location": "pulp_mill", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.10, "duration_days": 1},
                    {"name": "cooking_digestion", "typical_location": "pulp_mill", "risk_addition": 0.08, "traceability_loss": 0.15, "waste_pct": 0.45, "duration_days": 1},
                    {"name": "washing_screening", "typical_location": "pulp_mill", "risk_addition": 0.03, "traceability_loss": 0.10, "waste_pct": 0.05, "duration_days": 1},
                    {"name": "bleaching", "typical_location": "pulp_mill", "risk_addition": 0.05, "traceability_loss": 0.10, "waste_pct": 0.03, "duration_days": 1},
                    {"name": "drying_sheeting", "typical_location": "pulp_mill", "risk_addition": 0.03, "traceability_loss": 0.08, "waste_pct": 0.02, "duration_days": 1},
                ],
            },
        },
    },
}

# ===========================================================================
# Processing Risk Factors - Risk multipliers per processing type
# ===========================================================================

PROCESSING_RISK_FACTORS: Dict[str, float] = {
    "primary_processing": 1.0,
    "secondary_processing": 1.15,
    "chemical_processing": 1.25,
    "thermal_processing": 1.10,
    "mechanical_processing": 1.05,
    "blending_mixing": 1.30,
    "refining": 1.20,
    "fermentation": 1.10,
    "extraction": 1.25,
    "drying": 1.05,
    "packaging": 1.02,
    "transport": 1.08,
    "storage": 1.03,
    "multi_ingredient": 1.40,
}

# ===========================================================================
# Traceability Loss Factors
# ===========================================================================

TRACEABILITY_LOSS_FACTORS: Dict[str, float] = {
    "identity_preserved": 0.02,
    "segregated": 0.05,
    "mass_balance": 0.15,
    "book_and_claim": 0.30,
    "no_chain_of_custody": 0.50,
    "mixed_sources": 0.25,
    "co_processing": 0.35,
    "chemical_transformation": 0.20,
    "multi_site_processing": 0.15,
    "third_party_processing": 0.20,
}


# ===========================================================================
# ProcessingChainDatabase class
# ===========================================================================


class ProcessingChainDatabase:
    """
    Stateless accessor for processing chain reference data.

    Provides methods to query processing chains, calculate cumulative
    risk, and determine transformation ratios for EUDR commodity
    traceability analysis.

    Example:
        >>> db = ProcessingChainDatabase()
        >>> chain = db.get_processing_chain("cocoa", "chocolate_dark")
        >>> assert chain["final_product"] == "dark_chocolate"
    """

    def get_processing_chain(
        self, commodity_type: str, chain_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get a specific processing chain.

        Args:
            commodity_type: EUDR commodity type.
            chain_name: Chain identifier within the commodity.

        Returns:
            Processing chain dict or None if not found.
        """
        commodity = PROCESSING_CHAINS.get(commodity_type)
        if commodity is None:
            return None
        return commodity.get("chains", {}).get(chain_name)

    def get_all_chains(self, commodity_type: str) -> Dict[str, Any]:
        """Get all processing chains for a commodity.

        Args:
            commodity_type: EUDR commodity type.

        Returns:
            Dict of chain_name to chain data (empty if not found).
        """
        commodity = PROCESSING_CHAINS.get(commodity_type)
        if commodity is None:
            return {}
        return commodity.get("chains", {})

    def calculate_chain_risk(
        self, commodity_type: str, chain_name: str
    ) -> Optional[float]:
        """Calculate cumulative risk for a processing chain.

        Args:
            commodity_type: EUDR commodity type.
            chain_name: Chain identifier.

        Returns:
            Cumulative risk score (0.0-1.0) or None if chain not found.
        """
        chain = self.get_processing_chain(commodity_type, chain_name)
        if chain is None:
            return None
        return sum(s["risk_addition"] for s in chain.get("stages", []))


# ===========================================================================
# Module-level helper functions
# ===========================================================================


def get_processing_chain(
    commodity_type: str, chain_name: str
) -> Optional[Dict[str, Any]]:
    """Get a specific processing chain for a commodity.

    Args:
        commodity_type: EUDR commodity type.
        chain_name: Chain identifier within the commodity.

    Returns:
        Processing chain dict or None if not found.
    """
    commodity = PROCESSING_CHAINS.get(commodity_type)
    if commodity is None:
        return None
    return commodity.get("chains", {}).get(chain_name)


def calculate_chain_risk(commodity_type: str, chain_name: str) -> Optional[float]:
    """Calculate cumulative risk addition for a processing chain.

    Sums the risk_addition values across all stages in the chain.

    Args:
        commodity_type: EUDR commodity type.
        chain_name: Chain identifier.

    Returns:
        Cumulative risk score (float) or None if chain not found.
    """
    chain = get_processing_chain(commodity_type, chain_name)
    if chain is None:
        return None
    return sum(stage["risk_addition"] for stage in chain.get("stages", []))


def get_transformation_ratio(
    commodity_type: str, chain_name: str
) -> Optional[float]:
    """Get the overall transformation ratio for a processing chain.

    The transformation ratio represents the mass ratio of output to input
    (e.g., 0.21 means 1 tonne input produces 0.21 tonnes output).

    Args:
        commodity_type: EUDR commodity type.
        chain_name: Chain identifier.

    Returns:
        Transformation ratio (0.0-1.0) or None if chain not found.
    """
    chain = get_processing_chain(commodity_type, chain_name)
    if chain is None:
        return None
    return chain.get("overall_transformation_ratio")


def get_processing_stages(
    commodity_type: str, chain_name: str
) -> List[Dict[str, Any]]:
    """Get the list of processing stages for a chain.

    Args:
        commodity_type: EUDR commodity type.
        chain_name: Chain identifier.

    Returns:
        List of stage dicts (empty if chain not found).
    """
    chain = get_processing_chain(commodity_type, chain_name)
    if chain is None:
        return []
    return chain.get("stages", [])


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "PROCESSING_CHAINS",
    "PROCESSING_RISK_FACTORS",
    "TRACEABILITY_LOSS_FACTORS",
    "ProcessingChainDatabase",
    "get_processing_chain",
    "calculate_chain_risk",
    "get_transformation_ratio",
    "get_processing_stages",
]
