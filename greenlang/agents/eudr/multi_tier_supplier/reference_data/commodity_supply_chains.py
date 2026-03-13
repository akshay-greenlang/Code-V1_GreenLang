# -*- coding: utf-8 -*-
"""
Commodity Supply Chain Reference Data - AGENT-EUDR-008 Multi-Tier Supplier Tracker

Provides typical supply chain structures, tier definitions, actor types,
common supply chain patterns, and industry-average tier visibility benchmarks
for all 7 EUDR-regulated commodities. Used for tier depth assessment,
gap analysis, and benchmark comparison without external API dependencies.

Commodity Coverage (7 EUDR commodities):
    cocoa, coffee, palm_oil, soya, rubber, cattle, wood

Per Commodity Data:
    - Typical supply chain tiers (farmer to importer)
    - Actor types per tier level
    - Average and max tier depths
    - Common supply chain patterns (linear, branching, hub-spoke)
    - Industry visibility benchmarks (% of operators with N-tier visibility)
    - Typical volume flow patterns
    - Traceability challenges per tier

Data Sources:
    EUDR Regulation (EU) 2023/1115 Annex I
    FAO Global Food Value Chain Analysis (2024)
    UNCTAD Commodities at a Glance (2024)
    World Cocoa Foundation Supply Chain Mapping (2024)
    RSPO Supply Chain Analysis (2024)
    GFW Commodity Traceability Reports (2024)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Commodity Supply Chain Structures (7 EUDR commodities)
# ---------------------------------------------------------------------------

COMMODITY_SUPPLY_CHAINS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Cocoa (Theobroma cacao)
    # ------------------------------------------------------------------
    "cocoa": {
        "name": "Cocoa",
        "eudr_annex_reference": "Annex I, item 4",
        "hs_codes": ["1801", "1802", "1803", "1804", "1805", "1806"],
        "typical_tier_depth": 7,
        "min_tier_depth": 5,
        "max_tier_depth": 10,
        "chain_pattern": "branching_convergent",
        "chain_description": (
            "Highly fragmented at farm level (5-6 million smallholders globally), "
            "converging through cooperatives and aggregators to a small number "
            "of processors and traders. Top 3 processors control ~60% of volume."
        ),
        "tiers": [
            {
                "tier_level": 1,
                "tier_name": "EU Importer / First Placer",
                "actor_types": ["importer", "first_placer", "brand_owner"],
                "typical_actor_count": "1",
                "data_availability": "high",
                "eudr_obligation": "Full DDS required",
                "traceability_challenge": "Low -- direct business relationship",
            },
            {
                "tier_level": 2,
                "tier_name": "International Trader / Exporter",
                "actor_types": ["trader", "exporter", "broker"],
                "typical_actor_count": "1-3",
                "data_availability": "high",
                "eudr_obligation": "Trader obligation (Article 10)",
                "traceability_challenge": "Low -- contractual relationship",
            },
            {
                "tier_level": 3,
                "tier_name": "Processor / Grinder",
                "actor_types": ["processor", "grinder", "refiner"],
                "typical_actor_count": "1-5",
                "data_availability": "medium",
                "eudr_obligation": "Operator obligation if in EU",
                "traceability_challenge": "Medium -- bulk processing, blending",
            },
            {
                "tier_level": 4,
                "tier_name": "Regional Aggregator / Buying Station",
                "actor_types": ["aggregator", "buying_station", "warehouse"],
                "typical_actor_count": "5-50",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- informal, cash transactions",
            },
            {
                "tier_level": 5,
                "tier_name": "Cooperative / Farmer Organization",
                "actor_types": ["cooperative", "farmer_organization", "association"],
                "typical_actor_count": "10-200",
                "data_availability": "low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- limited record keeping",
            },
            {
                "tier_level": 6,
                "tier_name": "Village Collector / Pisteur",
                "actor_types": ["collector", "pisteur", "traitant", "intermediary"],
                "typical_actor_count": "50-1000",
                "data_availability": "very_low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Very high -- informal, no registration",
            },
            {
                "tier_level": 7,
                "tier_name": "Smallholder Farmer",
                "actor_types": ["farmer", "smallholder", "plot_owner"],
                "typical_actor_count": "100-100000",
                "data_availability": "very_low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Very high -- no GPS, no registration",
            },
        ],
        "major_origin_countries": ["CI", "GH", "EC", "CM", "NG", "ID", "BR"],
        "visibility_benchmarks": {
            "tier_1": 100.0,
            "tier_2": 95.0,
            "tier_3": 70.0,
            "tier_4": 35.0,
            "tier_5": 15.0,
            "tier_6": 5.0,
            "tier_7": 2.0,
        },
        "certification_coverage_pct": 35.0,
        "key_traceability_gaps": [
            "Village collector identity and GPS",
            "Cooperative member farmer list with plot GPS",
            "Blending at aggregation points",
            "Informal cash transactions below cooperative level",
        ],
    },
    # ------------------------------------------------------------------
    # Coffee (Coffea arabica / Coffea canephora)
    # ------------------------------------------------------------------
    "coffee": {
        "name": "Coffee",
        "eudr_annex_reference": "Annex I, item 3",
        "hs_codes": ["0901"],
        "typical_tier_depth": 6,
        "min_tier_depth": 4,
        "max_tier_depth": 9,
        "chain_pattern": "branching_convergent",
        "chain_description": (
            "Similar to cocoa with millions of smallholders. Key difference: "
            "wet/dry processing at origin adds a distinct processing tier. "
            "Specialty vs commodity segments have different chain depths."
        ),
        "tiers": [
            {
                "tier_level": 1,
                "tier_name": "EU Importer / Roaster",
                "actor_types": ["importer", "roaster", "brand_owner"],
                "typical_actor_count": "1",
                "data_availability": "high",
                "eudr_obligation": "Full DDS required",
                "traceability_challenge": "Low",
            },
            {
                "tier_level": 2,
                "tier_name": "International Trader / Exporter",
                "actor_types": ["trader", "exporter", "green_coffee_dealer"],
                "typical_actor_count": "1-3",
                "data_availability": "high",
                "eudr_obligation": "Trader obligation",
                "traceability_challenge": "Low -- contractual",
            },
            {
                "tier_level": 3,
                "tier_name": "Dry Mill / Export Processor",
                "actor_types": ["dry_mill", "export_processor", "cupping_lab"],
                "typical_actor_count": "1-5",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Medium -- lot blending",
            },
            {
                "tier_level": 4,
                "tier_name": "Wet Mill / Washing Station",
                "actor_types": ["wet_mill", "washing_station", "pulpery"],
                "typical_actor_count": "5-100",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Medium -- multi-farmer lots",
            },
            {
                "tier_level": 5,
                "tier_name": "Cooperative / Farmer Organization",
                "actor_types": ["cooperative", "farmer_group", "association"],
                "typical_actor_count": "10-500",
                "data_availability": "low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- variable record keeping",
            },
            {
                "tier_level": 6,
                "tier_name": "Smallholder Farmer",
                "actor_types": ["farmer", "smallholder", "estate"],
                "typical_actor_count": "100-50000",
                "data_availability": "very_low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Very high -- no digital records",
            },
        ],
        "major_origin_countries": [
            "BR", "VN", "CO", "ID", "ET", "HN", "IN", "UG", "PE", "MX",
        ],
        "visibility_benchmarks": {
            "tier_1": 100.0,
            "tier_2": 95.0,
            "tier_3": 75.0,
            "tier_4": 45.0,
            "tier_5": 20.0,
            "tier_6": 5.0,
        },
        "certification_coverage_pct": 40.0,
        "key_traceability_gaps": [
            "Washing station farmer registry completeness",
            "Lot blending at dry mill level",
            "Cooperative member plot GPS coordinates",
            "Cherry pickers (seasonal workers) identification",
        ],
    },
    # ------------------------------------------------------------------
    # Palm Oil (Elaeis guineensis)
    # ------------------------------------------------------------------
    "palm_oil": {
        "name": "Palm Oil",
        "eudr_annex_reference": "Annex I, item 1",
        "hs_codes": ["1511", "1513"],
        "typical_tier_depth": 6,
        "min_tier_depth": 4,
        "max_tier_depth": 8,
        "chain_pattern": "hub_spoke_convergent",
        "chain_description": (
            "Mill-centric supply chain where each palm oil mill sources "
            "from 500-3000 smallholders within a 50km radius. Refineries "
            "aggregate from multiple mills. High concentration at refining level."
        ),
        "tiers": [
            {
                "tier_level": 1,
                "tier_name": "EU Importer / User",
                "actor_types": ["importer", "food_manufacturer", "oleochemical_company"],
                "typical_actor_count": "1",
                "data_availability": "high",
                "eudr_obligation": "Full DDS required",
                "traceability_challenge": "Low",
            },
            {
                "tier_level": 2,
                "tier_name": "International Trader",
                "actor_types": ["trader", "broker", "commodity_house"],
                "typical_actor_count": "1-3",
                "data_availability": "high",
                "eudr_obligation": "Trader obligation",
                "traceability_challenge": "Low -- contractual",
            },
            {
                "tier_level": 3,
                "tier_name": "Refinery",
                "actor_types": ["refinery", "fractionation_plant", "oleochemical_plant"],
                "typical_actor_count": "1-5",
                "data_availability": "medium",
                "eudr_obligation": "Operator if in EU",
                "traceability_challenge": "High -- multi-source blending",
            },
            {
                "tier_level": 4,
                "tier_name": "Palm Oil Mill",
                "actor_types": ["mill", "crude_palm_oil_mill", "kernel_crusher"],
                "typical_actor_count": "5-50",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Medium -- known GPS, RSPO traceability",
            },
            {
                "tier_level": 5,
                "tier_name": "Collection Point / Dealer",
                "actor_types": ["collection_point", "dealer", "agent"],
                "typical_actor_count": "20-500",
                "data_availability": "low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- informal intermediaries",
            },
            {
                "tier_level": 6,
                "tier_name": "Smallholder / Plantation",
                "actor_types": ["smallholder", "plantation", "estate", "outgrower"],
                "typical_actor_count": "500-5000",
                "data_availability": "very_low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Very high -- smallholders lack GPS/ID",
            },
        ],
        "major_origin_countries": ["ID", "MY", "TH", "CO", "NG", "GH", "PG"],
        "visibility_benchmarks": {
            "tier_1": 100.0,
            "tier_2": 95.0,
            "tier_3": 85.0,
            "tier_4": 60.0,
            "tier_5": 25.0,
            "tier_6": 8.0,
        },
        "certification_coverage_pct": 19.0,
        "key_traceability_gaps": [
            "Independent smallholder identification and GPS",
            "Dealer/agent intermediaries between farmer and mill",
            "Refinery multi-source blending makes origin attribution difficult",
            "Kernel vs CPO supply chain divergence",
        ],
    },
    # ------------------------------------------------------------------
    # Soya (Glycine max)
    # ------------------------------------------------------------------
    "soya": {
        "name": "Soya",
        "eudr_annex_reference": "Annex I, item 2",
        "hs_codes": ["1201", "1507", "2304"],
        "typical_tier_depth": 5,
        "min_tier_depth": 3,
        "max_tier_depth": 7,
        "chain_pattern": "linear_convergent",
        "chain_description": (
            "More concentrated than cocoa/coffee. Fewer but larger farms. "
            "Silos aggregate from multiple farms. Crushers convert to meal/oil. "
            "Key challenge: silo-level blending from hundreds of farms."
        ),
        "tiers": [
            {
                "tier_level": 1,
                "tier_name": "EU Importer / Feed Compounder",
                "actor_types": ["importer", "feed_compounder", "food_processor"],
                "typical_actor_count": "1",
                "data_availability": "high",
                "eudr_obligation": "Full DDS required",
                "traceability_challenge": "Low",
            },
            {
                "tier_level": 2,
                "tier_name": "International Trader",
                "actor_types": ["trader", "commodity_house", "exporter"],
                "typical_actor_count": "1-3",
                "data_availability": "high",
                "eudr_obligation": "Trader obligation",
                "traceability_challenge": "Low",
            },
            {
                "tier_level": 3,
                "tier_name": "Crusher / Processor",
                "actor_types": ["crusher", "oil_extractor", "meal_processor"],
                "typical_actor_count": "1-10",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Medium -- volume blending",
            },
            {
                "tier_level": 4,
                "tier_name": "Silo / Elevator / Originator",
                "actor_types": ["silo", "elevator", "grain_originator", "cooperative"],
                "typical_actor_count": "5-50",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- bulk storage, multi-farm blending",
            },
            {
                "tier_level": 5,
                "tier_name": "Farm",
                "actor_types": ["farm", "ranch", "estate"],
                "typical_actor_count": "10-5000",
                "data_availability": "low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Medium -- larger farms with GPS available",
            },
        ],
        "major_origin_countries": ["BR", "AR", "PY", "US", "BO", "UY"],
        "visibility_benchmarks": {
            "tier_1": 100.0,
            "tier_2": 95.0,
            "tier_3": 80.0,
            "tier_4": 50.0,
            "tier_5": 15.0,
        },
        "certification_coverage_pct": 5.0,
        "key_traceability_gaps": [
            "Silo-level farm identification (blending from hundreds of farms)",
            "Land conversion history for individual farm plots",
            "CAR (Cadastro Ambiental Rural) linkage in Brazil",
            "Cross-border sourcing in Mercosur region",
        ],
    },
    # ------------------------------------------------------------------
    # Rubber (Hevea brasiliensis)
    # ------------------------------------------------------------------
    "rubber": {
        "name": "Rubber",
        "eudr_annex_reference": "Annex I, item 6",
        "hs_codes": ["4001", "4002", "4003", "4004", "4005"],
        "typical_tier_depth": 6,
        "min_tier_depth": 4,
        "max_tier_depth": 8,
        "chain_pattern": "branching_convergent",
        "chain_description": (
            "Smallholder-dominated (85% of production). Daily tapping creates "
            "frequent micro-transactions. Dealers aggregate from hundreds of "
            "smallholders. Factory processing converts to TSR/RSS grades."
        ),
        "tiers": [
            {
                "tier_level": 1,
                "tier_name": "EU Importer / Tire Manufacturer",
                "actor_types": ["importer", "tire_manufacturer", "rubber_goods_maker"],
                "typical_actor_count": "1",
                "data_availability": "high",
                "eudr_obligation": "Full DDS required",
                "traceability_challenge": "Low",
            },
            {
                "tier_level": 2,
                "tier_name": "International Trader",
                "actor_types": ["trader", "exporter", "commodity_house"],
                "typical_actor_count": "1-3",
                "data_availability": "high",
                "eudr_obligation": "Trader obligation",
                "traceability_challenge": "Low",
            },
            {
                "tier_level": 3,
                "tier_name": "Rubber Factory / Processor",
                "actor_types": ["factory", "processor", "tsr_plant", "rss_plant"],
                "typical_actor_count": "1-10",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Medium -- multi-source processing",
            },
            {
                "tier_level": 4,
                "tier_name": "Regional Dealer / Collector",
                "actor_types": ["dealer", "collector", "middleman", "agent"],
                "typical_actor_count": "10-200",
                "data_availability": "low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- informal, cash-based",
            },
            {
                "tier_level": 5,
                "tier_name": "Village Dealer / Sub-Dealer",
                "actor_types": ["village_dealer", "sub_dealer", "tapper_agent"],
                "typical_actor_count": "50-1000",
                "data_availability": "very_low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Very high -- daily micro-transactions",
            },
            {
                "tier_level": 6,
                "tier_name": "Smallholder Rubber Farmer",
                "actor_types": ["smallholder", "tapper", "farmer", "plantation"],
                "typical_actor_count": "500-10000",
                "data_availability": "very_low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Very high -- no registration, no GPS",
            },
        ],
        "major_origin_countries": ["TH", "ID", "VN", "CI", "MY", "IN", "CM"],
        "visibility_benchmarks": {
            "tier_1": 100.0,
            "tier_2": 95.0,
            "tier_3": 70.0,
            "tier_4": 30.0,
            "tier_5": 10.0,
            "tier_6": 3.0,
        },
        "certification_coverage_pct": 2.0,
        "key_traceability_gaps": [
            "Village dealer identity and GPS -- unregistered intermediaries",
            "Smallholder tapper identification and plot GPS",
            "Daily tapping micro-transactions (cash) traceability",
            "Cross-border smuggling (Thailand-Myanmar, CI-Liberia)",
        ],
    },
    # ------------------------------------------------------------------
    # Cattle (Bos taurus / Bos indicus)
    # ------------------------------------------------------------------
    "cattle": {
        "name": "Cattle",
        "eudr_annex_reference": "Annex I, item 5",
        "hs_codes": ["0102", "0201", "0202", "4101", "4104", "4107"],
        "typical_tier_depth": 4,
        "min_tier_depth": 3,
        "max_tier_depth": 6,
        "chain_pattern": "linear_with_transit",
        "chain_description": (
            "Unique among EUDR commodities: animals move between properties "
            "during their lifecycle (calf-cow, backgrounding, feedlot, slaughter). "
            "Each transfer creates a new tier. Key challenge: tracking animal "
            "movement history across multiple ranches."
        ),
        "tiers": [
            {
                "tier_level": 1,
                "tier_name": "EU Importer / Meat Processor",
                "actor_types": ["importer", "meat_processor", "leather_tanner"],
                "typical_actor_count": "1",
                "data_availability": "high",
                "eudr_obligation": "Full DDS required",
                "traceability_challenge": "Low",
            },
            {
                "tier_level": 2,
                "tier_name": "Slaughterhouse / Meatpacker",
                "actor_types": ["slaughterhouse", "meatpacker", "abattoir"],
                "typical_actor_count": "1-5",
                "data_availability": "high",
                "eudr_obligation": "Trader obligation",
                "traceability_challenge": "Low -- GTA/animal passport systems",
            },
            {
                "tier_level": 3,
                "tier_name": "Feedlot / Fattening Operation",
                "actor_types": ["feedlot", "fattening_farm", "finishing_operation"],
                "typical_actor_count": "1-20",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "Medium -- animal transit records",
            },
            {
                "tier_level": 4,
                "tier_name": "Cow-Calf Ranch / Breeder",
                "actor_types": ["ranch", "breeder", "cow_calf_operation", "farm"],
                "typical_actor_count": "5-500",
                "data_availability": "low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- indirect suppliers, laundering risk",
            },
        ],
        "major_origin_countries": ["BR", "AR", "PY", "UY", "CO", "AU"],
        "visibility_benchmarks": {
            "tier_1": 100.0,
            "tier_2": 95.0,
            "tier_3": 65.0,
            "tier_4": 15.0,
        },
        "certification_coverage_pct": 1.0,
        "key_traceability_gaps": [
            "Indirect suppliers (cow-calf ranches selling to feedlots)",
            "Animal laundering through intermediate ranches",
            "GTA (Guia de Transito Animal) digital availability",
            "Deforestation on cow-calf ranch land (pre-feedlot)",
        ],
    },
    # ------------------------------------------------------------------
    # Wood / Timber (various species)
    # ------------------------------------------------------------------
    "wood": {
        "name": "Wood / Timber",
        "eudr_annex_reference": "Annex I, item 7",
        "hs_codes": ["4401", "4403", "4407", "4408", "4409", "4410",
                      "4411", "4412", "4418", "9401", "9403"],
        "typical_tier_depth": 5,
        "min_tier_depth": 3,
        "max_tier_depth": 7,
        "chain_pattern": "linear_branching",
        "chain_description": (
            "More traceable than agricultural commodities due to log marking "
            "and chain of custody certification (FSC/PEFC). However, illegal "
            "logging and mixing at sawmill level remain major challenges. "
            "Wide product range from raw logs to finished furniture."
        ),
        "tiers": [
            {
                "tier_level": 1,
                "tier_name": "EU Importer / Distributor",
                "actor_types": ["importer", "distributor", "retailer", "furniture_maker"],
                "typical_actor_count": "1",
                "data_availability": "high",
                "eudr_obligation": "Full DDS required",
                "traceability_challenge": "Low",
            },
            {
                "tier_level": 2,
                "tier_name": "International Trader / Exporter",
                "actor_types": ["trader", "exporter", "log_broker"],
                "typical_actor_count": "1-3",
                "data_availability": "high",
                "eudr_obligation": "Trader obligation",
                "traceability_challenge": "Low -- shipping documentation",
            },
            {
                "tier_level": 3,
                "tier_name": "Processor / Manufacturer",
                "actor_types": ["sawmill", "plywood_mill", "pulp_mill", "furniture_factory"],
                "typical_actor_count": "1-10",
                "data_availability": "medium",
                "eudr_obligation": "Operator if in EU",
                "traceability_challenge": "Medium -- multi-source log intake",
            },
            {
                "tier_level": 4,
                "tier_name": "Primary Sawmill / Log Yard",
                "actor_types": ["sawmill", "log_yard", "wood_depot"],
                "typical_actor_count": "5-50",
                "data_availability": "medium",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- illegal log mixing risk",
            },
            {
                "tier_level": 5,
                "tier_name": "Forest Concession / Forest Owner",
                "actor_types": [
                    "forest_concession", "forest_owner", "community_forest",
                    "logging_company",
                ],
                "typical_actor_count": "1-100",
                "data_availability": "low",
                "eudr_obligation": "Not directly regulated",
                "traceability_challenge": "High -- concession boundary verification",
            },
        ],
        "major_origin_countries": [
            "BR", "ID", "MY", "CM", "CG", "CD", "GA", "PG", "RU",
        ],
        "visibility_benchmarks": {
            "tier_1": 100.0,
            "tier_2": 95.0,
            "tier_3": 80.0,
            "tier_4": 55.0,
            "tier_5": 30.0,
        },
        "certification_coverage_pct": 12.0,
        "key_traceability_gaps": [
            "Illegal logging mixed with legal at sawmill level",
            "Community forest land tenure documentation",
            "Concession boundary vs actual harvesting area",
            "Species identification and CITES compliance",
        ],
    },
}

# Totals
TOTAL_COMMODITIES: int = len(COMMODITY_SUPPLY_CHAINS)


# ---------------------------------------------------------------------------
# Industry Visibility Benchmarks (aggregate)
# ---------------------------------------------------------------------------

INDUSTRY_VISIBILITY_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "eu_operators_2024": {
        "description": (
            "Industry average visibility by tier level across EU operators "
            "as of 2024, based on surveys and EUDR readiness assessments."
        ),
        "tier_1_visibility_pct": 100.0,
        "tier_2_visibility_pct": 92.0,
        "tier_3_visibility_pct": 55.0,
        "tier_4_visibility_pct": 22.0,
        "tier_5_visibility_pct": 8.0,
        "tier_6_plus_visibility_pct": 3.0,
        "avg_mapped_depth": 2.8,
        "target_mapped_depth_eudr": 5.0,
        "operators_with_full_chain_pct": 4.0,
    },
    "eudr_target_2026": {
        "description": (
            "Target visibility levels for EUDR compliance by June 2026. "
            "Operators should aim to exceed these benchmarks."
        ),
        "tier_1_visibility_pct": 100.0,
        "tier_2_visibility_pct": 100.0,
        "tier_3_visibility_pct": 90.0,
        "tier_4_visibility_pct": 70.0,
        "tier_5_visibility_pct": 50.0,
        "tier_6_plus_visibility_pct": 20.0,
        "avg_mapped_depth": 4.5,
        "target_mapped_depth_eudr": 5.0,
        "operators_with_full_chain_pct": 30.0,
    },
}


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------


def get_typical_chain(commodity: str) -> Optional[Dict[str, Any]]:
    """Get the typical supply chain structure for a commodity.

    Args:
        commodity: EUDR commodity key (e.g. 'cocoa', 'palm_oil').

    Returns:
        Dictionary with full supply chain structure or None if not found.

    Example:
        >>> chain = get_typical_chain("cocoa")
        >>> assert chain["typical_tier_depth"] == 7
    """
    return COMMODITY_SUPPLY_CHAINS.get(commodity.lower()) if commodity else None


def get_industry_benchmark(
    benchmark_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Get industry visibility benchmarks.

    Args:
        benchmark_key: Optional benchmark key (e.g. 'eu_operators_2024').
            If None, returns all benchmarks.

    Returns:
        Dictionary with benchmark data.

    Example:
        >>> bench = get_industry_benchmark("eu_operators_2024")
        >>> assert bench["avg_mapped_depth"] == 2.8
    """
    if benchmark_key is not None:
        return INDUSTRY_VISIBILITY_BENCHMARKS.get(benchmark_key, {})
    return dict(INDUSTRY_VISIBILITY_BENCHMARKS)


def get_actor_types_for_tier(
    commodity: str,
    tier_level: int,
) -> List[str]:
    """Get the typical actor types at a specific tier level for a commodity.

    Args:
        commodity: EUDR commodity key.
        tier_level: Tier level number (1-based).

    Returns:
        List of actor type strings, or empty list if not found.

    Example:
        >>> types = get_actor_types_for_tier("cocoa", 5)
        >>> assert "cooperative" in types
    """
    chain = get_typical_chain(commodity)
    if chain is None:
        return []
    for tier in chain.get("tiers", []):
        if tier["tier_level"] == tier_level:
            return tier.get("actor_types", [])
    return []


def get_avg_tier_depth(commodity: str) -> Optional[int]:
    """Get the typical (average) tier depth for a commodity.

    Args:
        commodity: EUDR commodity key.

    Returns:
        Integer tier depth or None if commodity not found.

    Example:
        >>> depth = get_avg_tier_depth("cocoa")
        >>> assert depth == 7
    """
    chain = get_typical_chain(commodity)
    if chain is None:
        return None
    return chain.get("typical_tier_depth")


def get_visibility_benchmark(
    commodity: str,
    tier_level: int,
) -> Optional[float]:
    """Get the industry-average visibility percentage at a tier level.

    Args:
        commodity: EUDR commodity key.
        tier_level: Tier level number (1-based).

    Returns:
        Visibility percentage (0-100) or None if not found.

    Example:
        >>> vis = get_visibility_benchmark("cocoa", 4)
        >>> assert vis == 35.0
    """
    chain = get_typical_chain(commodity)
    if chain is None:
        return None
    benchmarks = chain.get("visibility_benchmarks", {})
    return benchmarks.get(f"tier_{tier_level}")


def get_tier_info(
    commodity: str,
    tier_level: int,
) -> Optional[Dict[str, Any]]:
    """Get detailed tier information for a commodity at a specific level.

    Args:
        commodity: EUDR commodity key.
        tier_level: Tier level number (1-based).

    Returns:
        Dictionary with tier details or None if not found.

    Example:
        >>> info = get_tier_info("palm_oil", 4)
        >>> assert info["tier_name"] == "Palm Oil Mill"
    """
    chain = get_typical_chain(commodity)
    if chain is None:
        return None
    for tier in chain.get("tiers", []):
        if tier["tier_level"] == tier_level:
            return dict(tier)
    return None


def get_traceability_gaps(commodity: str) -> List[str]:
    """Get key traceability gaps for a commodity supply chain.

    Args:
        commodity: EUDR commodity key.

    Returns:
        List of traceability gap descriptions, or empty list.

    Example:
        >>> gaps = get_traceability_gaps("soya")
        >>> assert len(gaps) > 0
    """
    chain = get_typical_chain(commodity)
    if chain is None:
        return []
    return chain.get("key_traceability_gaps", [])


def get_all_commodities() -> List[Dict[str, Any]]:
    """Get a summary list of all EUDR commodity supply chains.

    Returns:
        List of commodity summaries sorted alphabetically by name.

    Example:
        >>> commodities = get_all_commodities()
        >>> assert len(commodities) == 7
    """
    results = []
    for key, chain in COMMODITY_SUPPLY_CHAINS.items():
        results.append({
            "key": key,
            "name": chain["name"],
            "typical_tier_depth": chain["typical_tier_depth"],
            "min_tier_depth": chain["min_tier_depth"],
            "max_tier_depth": chain["max_tier_depth"],
            "chain_pattern": chain["chain_pattern"],
            "tier_count": len(chain["tiers"]),
            "major_origin_countries": chain["major_origin_countries"],
            "certification_coverage_pct": chain["certification_coverage_pct"],
        })
    results.sort(key=lambda x: x["name"])
    return results
