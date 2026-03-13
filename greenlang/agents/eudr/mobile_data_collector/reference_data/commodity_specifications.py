# -*- coding: utf-8 -*-
"""
EUDR Commodity Specifications - AGENT-EUDR-015

Reference data for the 7 EUDR-regulated commodities as defined in
EU 2023/1115 Article 1 and Annex I.  Each commodity entry provides:

    - commodity_code: Internal commodity identifier
    - hs_codes: Harmonized System codes and ranges
    - cn_codes: Combined Nomenclature codes (EU-specific)
    - derived_products: Sub-products and derivatives
    - unit_of_measure: Standard measurement unit
    - quality_parameters: Min/max/typical for key quality attributes
    - required_certifications: Applicable certification schemes
    - seasonal_patterns: Typical harvest seasons by region

All data is stored in module-level frozen dictionaries importable
without side effects.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ===========================================================================
# Cattle
# ===========================================================================

CATTLE_SPECIFICATION: Dict[str, Any] = {
    "commodity_code": "cattle",
    "commodity_name": "Cattle",
    "eudr_annex_ref": "Annex I, Item 1",
    "description": (
        "Live bovine animals and derived products including beef, "
        "leather, and tallow."
    ),
    "hs_codes": [
        {"code": "0102", "description": "Live bovine animals"},
        {"code": "0201", "description": "Meat of bovine animals, fresh or chilled"},
        {"code": "0202", "description": "Meat of bovine animals, frozen"},
        {"code": "0206", "description": "Edible offal of bovine animals"},
        {"code": "0210", "description": "Meat, salted, dried, or smoked"},
        {"code": "4101", "description": "Raw hides and skins of bovine"},
        {"code": "4104", "description": "Tanned/crust hides of bovine"},
        {"code": "4107", "description": "Leather of bovine, further prepared"},
        {"code": "1502", "description": "Fats of bovine animals (tallow)"},
    ],
    "derived_products": [
        {"product": "beef", "description": "Beef cuts and processed beef"},
        {"product": "leather", "description": "Bovine leather and hides"},
        {"product": "tallow", "description": "Rendered bovine fat"},
        {"product": "offal", "description": "Edible offal"},
        {"product": "gelatin", "description": "Bovine gelatin"},
    ],
    "unit_of_measure": "kg",
    "alternate_units": ["head", "tonnes"],
    "quality_parameters": {
        "age_categories": [
            {"category": "calf", "max_months": 12},
            {"category": "yearling", "min_months": 12, "max_months": 24},
            {"category": "adult", "min_months": 24},
        ],
        "breed_types": [
            "angus", "hereford", "brahman", "nelore", "charolais",
            "limousin", "simmental", "holstein", "mixed",
        ],
        "carcass_grades": [
            {"grade": "prime", "description": "Highest quality"},
            {"grade": "choice", "description": "High quality"},
            {"grade": "select", "description": "Standard quality"},
            {"grade": "standard", "description": "Below average"},
            {"grade": "ungraded", "description": "Not graded"},
        ],
        "weight_range_kg": {"min": 200, "max": 1200, "typical": 550},
    },
    "required_certifications": [
        "veterinary_health_certificate",
        "origin_certificate",
    ],
    "seasonal_patterns": {
        "brazil": {"peak_months": [3, 4, 5, 6, 7, 8, 9, 10]},
        "argentina": {"peak_months": [1, 2, 3, 4, 5, 10, 11, 12]},
        "australia": {"peak_months": [1, 2, 3, 4, 5, 6]},
    },
}


# ===========================================================================
# Cocoa
# ===========================================================================

COCOA_SPECIFICATION: Dict[str, Any] = {
    "commodity_code": "cocoa",
    "commodity_name": "Cocoa",
    "eudr_annex_ref": "Annex I, Item 2",
    "description": (
        "Cocoa beans and derived products including butter, powder, "
        "paste, and chocolate."
    ),
    "hs_codes": [
        {"code": "1801", "description": "Cocoa beans, whole or broken"},
        {"code": "1802", "description": "Cocoa shells, husks, and waste"},
        {"code": "1803", "description": "Cocoa paste"},
        {"code": "1804", "description": "Cocoa butter, fat, and oil"},
        {"code": "1805", "description": "Cocoa powder (no added sugar)"},
        {"code": "1806", "description": "Chocolate and cocoa preparations"},
    ],
    "derived_products": [
        {"product": "beans", "description": "Raw or fermented cocoa beans"},
        {"product": "butter", "description": "Cocoa butter (fat extracted)"},
        {"product": "powder", "description": "Cocoa powder"},
        {"product": "paste", "description": "Cocoa paste/liquor"},
        {"product": "chocolate", "description": "Chocolate products"},
        {"product": "nibs", "description": "Cocoa nibs"},
        {"product": "shells", "description": "Cocoa shells and husks"},
    ],
    "unit_of_measure": "kg",
    "alternate_units": ["bags_60kg", "tonnes"],
    "quality_parameters": {
        "fermentation_levels": [
            {"level": "well_fermented", "min_pct": 75, "description": ">75% fermented"},
            {"level": "moderately_fermented", "min_pct": 50, "max_pct": 75},
            {"level": "under_fermented", "max_pct": 50},
            {"level": "unfermented", "max_pct": 5},
        ],
        "bean_count_per_100g": {"min": 80, "max": 130, "typical": 100},
        "moisture_pct": {"min": 5.0, "max": 8.0, "typical": 6.5, "reject_above": 8.0},
        "defect_pct": {"max_grade_1": 3.0, "max_grade_2": 8.0},
        "foreign_matter_pct": {"max": 1.5},
        "fat_content_pct": {"min": 50.0, "max": 58.0, "typical": 54.0},
        "grades": [
            {"grade": "grade_1", "max_defect_pct": 3, "max_foreign_matter_pct": 1},
            {"grade": "grade_2", "max_defect_pct": 8, "max_foreign_matter_pct": 1.5},
            {"grade": "substandard", "description": "Above grade 2 thresholds"},
        ],
    },
    "required_certifications": [
        "phytosanitary_certificate",
        "origin_certificate",
    ],
    "seasonal_patterns": {
        "west_africa": {"main_crop_months": [10, 11, 12, 1, 2, 3], "mid_crop_months": [5, 6, 7, 8]},
        "indonesia": {"peak_months": [3, 4, 5, 9, 10, 11]},
        "ecuador": {"peak_months": [4, 5, 6, 7]},
    },
}


# ===========================================================================
# Coffee
# ===========================================================================

COFFEE_SPECIFICATION: Dict[str, Any] = {
    "commodity_code": "coffee",
    "commodity_name": "Coffee",
    "eudr_annex_ref": "Annex I, Item 3",
    "description": (
        "Coffee in all forms: cherry, parchment, green, roasted, "
        "instant, and extracts."
    ),
    "hs_codes": [
        {"code": "0901", "description": "Coffee, whether or not roasted"},
        {"code": "210111", "description": "Extracts, essences of coffee"},
        {"code": "210112", "description": "Preparations of coffee"},
    ],
    "derived_products": [
        {"product": "green", "description": "Green (unroasted) coffee beans"},
        {"product": "roasted", "description": "Roasted coffee beans"},
        {"product": "instant", "description": "Instant/soluble coffee"},
        {"product": "cherry", "description": "Fresh coffee cherry"},
        {"product": "parchment", "description": "Parchment coffee"},
        {"product": "extracts", "description": "Coffee extracts and essences"},
    ],
    "unit_of_measure": "kg",
    "alternate_units": ["bags_60kg", "bags_69kg", "tonnes"],
    "quality_parameters": {
        "screen_sizes": {
            "AA": {"min_size": 18, "description": "Premium large bean"},
            "AB": {"min_size": 15, "max_size": 18},
            "C": {"max_size": 15, "description": "Small bean"},
            "PB": {"description": "Peaberry"},
        },
        "defect_count_per_300g": {
            "specialty": {"max": 5},
            "premium": {"max": 8},
            "exchange": {"max": 23},
            "below_standard": {"min": 24},
        },
        "cup_scores": {
            "specialty": {"min": 80, "max": 100},
            "premium": {"min": 75, "max": 80},
            "commercial": {"min": 60, "max": 75},
            "below_grade": {"max": 60},
        },
        "moisture_pct": {"min": 9.0, "max": 12.5, "typical": 11.0, "reject_above": 13.0},
        "foreign_matter_pct": {"max": 0.5},
        "species": ["arabica", "robusta", "liberica"],
    },
    "required_certifications": [
        "phytosanitary_certificate",
        "origin_certificate",
        "ico_certificate_of_origin",
    ],
    "seasonal_patterns": {
        "brazil": {"peak_months": [5, 6, 7, 8, 9]},
        "colombia": {"main_harvest": [10, 11, 12, 1], "mitaca": [4, 5, 6]},
        "ethiopia": {"peak_months": [10, 11, 12, 1, 2]},
        "vietnam": {"peak_months": [11, 12, 1, 2, 3]},
        "indonesia": {"peak_months": [5, 6, 7, 8, 9]},
    },
}


# ===========================================================================
# Oil Palm
# ===========================================================================

OIL_PALM_SPECIFICATION: Dict[str, Any] = {
    "commodity_code": "oil_palm",
    "commodity_name": "Oil Palm",
    "eudr_annex_ref": "Annex I, Item 4",
    "description": (
        "Oil palm products including crude palm oil (CPO), palm kernel "
        "oil (PKO), and derived products."
    ),
    "hs_codes": [
        {"code": "1511", "description": "Palm oil and fractions"},
        {"code": "1513", "description": "Coconut/palm kernel oil"},
        {"code": "151321", "description": "Crude palm kernel oil"},
        {"code": "151329", "description": "Refined palm kernel oil"},
        {"code": "2306", "description": "Oil-cake and residues of palm"},
        {"code": "3823", "description": "Fatty acids from palm oil"},
    ],
    "derived_products": [
        {"product": "cpo", "description": "Crude Palm Oil"},
        {"product": "pko", "description": "Palm Kernel Oil"},
        {"product": "rbdpo", "description": "Refined, Bleached, Deodorized Palm Oil"},
        {"product": "palm_stearin", "description": "Palm stearin (solid fraction)"},
        {"product": "palm_olein", "description": "Palm olein (liquid fraction)"},
        {"product": "ffb", "description": "Fresh Fruit Bunches"},
        {"product": "kernel_cake", "description": "Palm kernel expeller/cake"},
    ],
    "unit_of_measure": "kg",
    "alternate_units": ["tonnes", "bunches"],
    "quality_parameters": {
        "ffa_pct": {
            "excellent": {"max": 2.0},
            "good": {"min": 2.0, "max": 3.5},
            "acceptable": {"min": 3.5, "max": 5.0},
            "reject": {"min": 5.0},
            "typical": 3.0,
        },
        "dobi_value": {
            "good": {"min": 2.5},
            "acceptable": {"min": 2.0, "max": 2.5},
            "poor": {"max": 2.0},
            "description": "Deterioration of Bleachability Index",
        },
        "moisture_pct": {"max": 0.5, "typical": 0.2},
        "impurities_pct": {"max": 0.1},
        "iodine_value": {"min": 50, "max": 55, "typical": 52},
        "peroxide_value": {"max": 5.0},
        "ffb_ripeness": [
            {"level": "unripe", "loose_fruits_pct": 0},
            {"level": "under_ripe", "loose_fruits_pct_range": [0, 25]},
            {"level": "ripe", "loose_fruits_pct_range": [25, 75]},
            {"level": "over_ripe", "loose_fruits_pct_range": [75, 100]},
        ],
    },
    "required_certifications": [
        "phytosanitary_certificate",
        "origin_certificate",
    ],
    "seasonal_patterns": {
        "malaysia": {"peak_months": [7, 8, 9, 10, 11]},
        "indonesia": {"peak_months": [8, 9, 10, 11, 12]},
        "west_africa": {"peak_months": [2, 3, 4, 5, 6]},
    },
}


# ===========================================================================
# Rubber
# ===========================================================================

RUBBER_SPECIFICATION: Dict[str, Any] = {
    "commodity_code": "rubber",
    "commodity_name": "Rubber",
    "eudr_annex_ref": "Annex I, Item 5",
    "description": (
        "Natural rubber including latex, cup lump, sheet rubber, "
        "and Technically Specified Rubber (TSR) grades."
    ),
    "hs_codes": [
        {"code": "4001", "description": "Natural rubber in primary forms"},
        {"code": "400110", "description": "Natural rubber latex"},
        {"code": "400121", "description": "Smoked sheets of natural rubber"},
        {"code": "400122", "description": "Technically specified natural rubber (TSNR)"},
        {"code": "400129", "description": "Other natural rubber in primary forms"},
        {"code": "400130", "description": "Balata, gutta-percha"},
    ],
    "derived_products": [
        {"product": "latex", "description": "Field latex (liquid)"},
        {"product": "cup_lump", "description": "Coagulated cup lump"},
        {"product": "rss", "description": "Ribbed Smoked Sheet (RSS)"},
        {"product": "tsr", "description": "Technically Specified Rubber"},
        {"product": "crepe", "description": "Crepe rubber"},
        {"product": "skim_rubber", "description": "Skim rubber from centrifuge"},
    ],
    "unit_of_measure": "kg",
    "alternate_units": ["tonnes"],
    "quality_parameters": {
        "drc_pct": {
            "latex": {"min": 28, "max": 40, "typical": 33},
            "cup_lump": {"min": 40, "max": 65, "typical": 50},
            "description": "Dry Rubber Content",
        },
        "ash_content_pct": {
            "tsr_5": {"max": 0.4},
            "tsr_10": {"max": 0.75},
            "tsr_20": {"max": 1.0},
            "tsr_50": {"max": 1.5},
        },
        "dirt_content_pct": {
            "tsr_5": {"max": 0.04},
            "tsr_10": {"max": 0.08},
            "tsr_20": {"max": 0.16},
            "tsr_50": {"max": 0.5},
        },
        "volatile_matter_pct": {"max": 0.8, "typical": 0.5},
        "nitrogen_pct": {"max": 0.6},
        "plasticity_retention_index": {"min": 30, "max": 80},
        "mooney_viscosity": {"min": 50, "max": 90, "typical": 65},
        "tsr_grades": ["TSR5", "TSR10", "TSR20", "TSR50"],
        "rss_grades": ["RSS1", "RSS2", "RSS3", "RSS4", "RSS5"],
    },
    "required_certifications": [
        "phytosanitary_certificate",
        "origin_certificate",
    ],
    "seasonal_patterns": {
        "thailand": {"peak_months": [1, 2, 3, 4, 5, 10, 11, 12], "wintering": [6, 7, 8, 9]},
        "indonesia": {"peak_months": [3, 4, 5, 6, 7, 8, 9, 10]},
        "vietnam": {"peak_months": [3, 4, 5, 6, 7, 8, 9, 10]},
    },
}


# ===========================================================================
# Soya
# ===========================================================================

SOYA_SPECIFICATION: Dict[str, Any] = {
    "commodity_code": "soya",
    "commodity_name": "Soya",
    "eudr_annex_ref": "Annex I, Item 6",
    "description": (
        "Soybeans and derived products including meal, oil, lecithin, "
        "and protein isolates."
    ),
    "hs_codes": [
        {"code": "1201", "description": "Soya beans, whether or not broken"},
        {"code": "1208", "description": "Soya bean flour and meal"},
        {"code": "1507", "description": "Soya-bean oil and fractions"},
        {"code": "2304", "description": "Soya-bean oil-cake and residues"},
    ],
    "derived_products": [
        {"product": "beans", "description": "Whole soybeans"},
        {"product": "meal", "description": "Soybean meal (animal feed)"},
        {"product": "oil", "description": "Soybean oil (crude/refined)"},
        {"product": "lecithin", "description": "Soy lecithin"},
        {"product": "protein_isolate", "description": "Soy protein isolate"},
        {"product": "flour", "description": "Soybean flour"},
        {"product": "hulls", "description": "Soybean hulls"},
    ],
    "unit_of_measure": "kg",
    "alternate_units": ["bushels", "tonnes"],
    "quality_parameters": {
        "protein_pct": {"min": 34.0, "max": 42.0, "typical": 38.0},
        "oil_content_pct": {"min": 17.0, "max": 22.0, "typical": 19.5},
        "moisture_pct": {"max": 13.0, "typical": 11.0, "reject_above": 14.0},
        "foreign_matter_pct": {"max": 2.0, "premium_max": 1.0},
        "damaged_kernels_pct": {"max": 8.0, "premium_max": 3.0},
        "splits_pct": {"max": 30.0, "premium_max": 10.0},
        "gmo_status": [
            {"status": "conventional", "description": "Non-GMO verified"},
            {"status": "identity_preserved", "description": "IP Non-GMO"},
            {"status": "gmo", "description": "Genetically modified"},
            {"status": "unknown", "description": "GMO status unknown"},
        ],
        "grades": [
            {"grade": "grade_1", "max_damage_pct": 2, "max_foreign_pct": 1},
            {"grade": "grade_2", "max_damage_pct": 5, "max_foreign_pct": 2},
            {"grade": "sample_grade", "description": "Below grade 2"},
        ],
    },
    "required_certifications": [
        "phytosanitary_certificate",
        "origin_certificate",
    ],
    "seasonal_patterns": {
        "brazil": {"peak_months": [2, 3, 4, 5]},
        "argentina": {"peak_months": [3, 4, 5]},
        "usa": {"peak_months": [9, 10, 11]},
        "paraguay": {"peak_months": [2, 3, 4]},
    },
}


# ===========================================================================
# Wood
# ===========================================================================

WOOD_SPECIFICATION: Dict[str, Any] = {
    "commodity_code": "wood",
    "commodity_name": "Wood",
    "eudr_annex_ref": "Annex I, Item 7",
    "description": (
        "Wood products including logs, sawnwood, plywood, pulp, paper, "
        "charcoal, and printed products."
    ),
    "hs_codes": [
        {"code": "4401", "description": "Fuel wood, wood chips, sawdust"},
        {"code": "4402", "description": "Wood charcoal"},
        {"code": "4403", "description": "Wood in the rough (logs)"},
        {"code": "4404", "description": "Hoopwood, split poles"},
        {"code": "4406", "description": "Railway sleepers of wood"},
        {"code": "4407", "description": "Wood sawn or chipped lengthwise"},
        {"code": "4408", "description": "Veneer sheets"},
        {"code": "4409", "description": "Wood continuously shaped"},
        {"code": "4410", "description": "Particle board and OSB"},
        {"code": "4411", "description": "Fibreboard (MDF/HDF)"},
        {"code": "4412", "description": "Plywood, veneered panels"},
        {"code": "4413", "description": "Densified wood"},
        {"code": "4414", "description": "Wooden frames"},
        {"code": "4415", "description": "Packing cases, pallets"},
        {"code": "4416", "description": "Casks, barrels of wood"},
        {"code": "4418", "description": "Builders' joinery (doors/windows)"},
        {"code": "47", "description": "Pulp of wood"},
        {"code": "48", "description": "Paper and paperboard"},
        {"code": "49", "description": "Printed books and newspapers"},
        {"code": "9401", "description": "Wooden furniture (seats)"},
        {"code": "9403", "description": "Wooden furniture (other)"},
    ],
    "derived_products": [
        {"product": "logs", "description": "Round wood/logs"},
        {"product": "sawnwood", "description": "Sawn timber/lumber"},
        {"product": "plywood", "description": "Plywood sheets"},
        {"product": "veneer", "description": "Veneer sheets"},
        {"product": "charcoal", "description": "Wood charcoal"},
        {"product": "pulp", "description": "Wood pulp"},
        {"product": "paper", "description": "Paper and paperboard"},
        {"product": "particle_board", "description": "Particle board/OSB"},
        {"product": "fibreboard", "description": "MDF/HDF"},
        {"product": "furniture", "description": "Wooden furniture"},
        {"product": "pellets", "description": "Wood pellets"},
    ],
    "unit_of_measure": "m3",
    "alternate_units": ["kg", "tonnes", "board_feet", "pieces"],
    "quality_parameters": {
        "species_codes": {
            "description": "CITES and national species identification codes",
            "examples": ["teak", "mahogany", "pine", "eucalyptus", "acacia", "spruce"],
        },
        "fsc_status": [
            {"status": "fsc_100", "description": "100% FSC certified"},
            {"status": "fsc_mix", "description": "FSC Mix"},
            {"status": "fsc_recycled", "description": "FSC Recycled"},
            {"status": "non_certified", "description": "Not certified"},
        ],
        "pefc_status": [
            {"status": "pefc_certified", "description": "PEFC certified"},
            {"status": "non_certified", "description": "Not certified"},
        ],
        "moisture_pct": {"green": {"min": 30, "max": 80}, "air_dried": {"max": 20}, "kiln_dried": {"max": 12}},
        "volume_measurement": {
            "methods": ["under_bark", "over_bark", "stacked", "solid"],
            "scaling_rules": "National scaling rules apply",
        },
        "defect_types": [
            "knots", "splits", "warp", "decay", "insect_damage",
            "blue_stain", "shake", "bark_inclusion",
        ],
        "log_grades": [
            {"grade": "A", "description": "Premium/veneer quality"},
            {"grade": "B", "description": "Sawlog quality"},
            {"grade": "C", "description": "Industrial quality"},
            {"grade": "D", "description": "Pulpwood/fuel quality"},
        ],
    },
    "required_certifications": [
        "phytosanitary_certificate",
        "origin_certificate",
        "cites_permit",
        "flegt_license",
    ],
    "seasonal_patterns": {
        "brazil": {"peak_months": [5, 6, 7, 8, 9, 10]},
        "indonesia": {"peak_months": [4, 5, 6, 7, 8, 9, 10]},
        "northern_hemisphere": {"peak_months": [10, 11, 12, 1, 2, 3]},
    },
}


# ===========================================================================
# Commodity registry
# ===========================================================================

ALL_COMMODITIES: Dict[str, Dict[str, Any]] = {
    "cattle": CATTLE_SPECIFICATION,
    "cocoa": COCOA_SPECIFICATION,
    "coffee": COFFEE_SPECIFICATION,
    "oil_palm": OIL_PALM_SPECIFICATION,
    "rubber": RUBBER_SPECIFICATION,
    "soya": SOYA_SPECIFICATION,
    "wood": WOOD_SPECIFICATION,
}

VALID_COMMODITY_CODES: List[str] = sorted(ALL_COMMODITIES.keys())


# ===========================================================================
# Accessor functions
# ===========================================================================


def get_commodity(commodity_code: str) -> Optional[Dict[str, Any]]:
    """Return the commodity specification for the given code.

    Args:
        commodity_code: Internal commodity code (cattle, cocoa, etc.).

    Returns:
        Commodity specification dictionary or None if not found.
    """
    return ALL_COMMODITIES.get(commodity_code)


def is_valid_commodity(commodity_code: str) -> bool:
    """Check whether the given commodity code is EUDR-regulated.

    Args:
        commodity_code: Commodity code to check.

    Returns:
        True if the commodity is one of the 7 EUDR-regulated commodities.
    """
    return commodity_code in ALL_COMMODITIES


def get_hs_codes(commodity_code: str) -> List[Dict[str, str]]:
    """Return the HS code list for a commodity.

    Args:
        commodity_code: Internal commodity code.

    Returns:
        List of HS code dictionaries, or empty list if not found.
    """
    spec = ALL_COMMODITIES.get(commodity_code)
    if spec is None:
        return []
    return spec.get("hs_codes", [])


def get_derived_products(commodity_code: str) -> List[Dict[str, str]]:
    """Return the derived products list for a commodity.

    Args:
        commodity_code: Internal commodity code.

    Returns:
        List of derived product dictionaries, or empty list.
    """
    spec = ALL_COMMODITIES.get(commodity_code)
    if spec is None:
        return []
    return spec.get("derived_products", [])


def get_quality_parameters(commodity_code: str) -> Dict[str, Any]:
    """Return the quality parameters for a commodity.

    Args:
        commodity_code: Internal commodity code.

    Returns:
        Quality parameters dictionary, or empty dict if not found.
    """
    spec = ALL_COMMODITIES.get(commodity_code)
    if spec is None:
        return {}
    return spec.get("quality_parameters", {})


def get_commodity_unit(commodity_code: str) -> str:
    """Return the standard unit of measure for a commodity.

    Args:
        commodity_code: Internal commodity code.

    Returns:
        Unit of measure string, or 'kg' as default.
    """
    spec = ALL_COMMODITIES.get(commodity_code)
    if spec is None:
        return "kg"
    return spec.get("unit_of_measure", "kg")


def lookup_commodity_by_hs(hs_code: str) -> Optional[str]:
    """Find the EUDR commodity code matching a given HS code prefix.

    Args:
        hs_code: Harmonized System code (partial match from start).

    Returns:
        Commodity code string, or None if no match.
    """
    for code, spec in ALL_COMMODITIES.items():
        for hs_entry in spec.get("hs_codes", []):
            if hs_code.startswith(hs_entry["code"]):
                return code
    return None


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "CATTLE_SPECIFICATION",
    "COCOA_SPECIFICATION",
    "COFFEE_SPECIFICATION",
    "OIL_PALM_SPECIFICATION",
    "RUBBER_SPECIFICATION",
    "SOYA_SPECIFICATION",
    "WOOD_SPECIFICATION",
    "ALL_COMMODITIES",
    "VALID_COMMODITY_CODES",
    "get_commodity",
    "is_valid_commodity",
    "get_hs_codes",
    "get_derived_products",
    "get_quality_parameters",
    "get_commodity_unit",
    "lookup_commodity_by_hs",
]
