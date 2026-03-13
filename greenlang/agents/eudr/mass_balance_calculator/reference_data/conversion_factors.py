# -*- coding: utf-8 -*-
"""
Conversion Factors Reference Data - AGENT-EUDR-011 Mass Balance Calculator

Commodity-specific conversion/yield ratios for mass balance calculations.
These factors define the expected output-to-input ratios for every major
processing step across the seven EUDR commodity classes.  Each entry
includes an acceptable range so the ConversionFactorValidator engine can
flag deviations as warnings or rejections.

Datasets:
    CONVERSION_FACTORS:
        List of dictionaries, each describing one commodity conversion
        step.  Keys: commodity, process, input_material, output_material,
        yield_ratio (expected), acceptable_range_min, acceptable_range_max,
        source (authoritative body), notes, data_year.

    COMMODITY_PROCESS_INDEX:
        Two-level lookup ``{commodity: {process: factor_dict}}`` built
        automatically from CONVERSION_FACTORS for O(1) access.

Lookup helpers:
    get_reference_factor(commodity, process) -> dict | None
    get_all_factors(commodity) -> list[dict]
    get_factor_range(commodity, process) -> tuple[float, float] | None
    is_within_range(commodity, process, actual_ratio) -> bool

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011 (Mass Balance Calculator) - Appendix A
Agent ID: GL-EUDR-MBC-011
Regulation: EU 2023/1115 (EUDR) Articles 4, 10(2)(f), 14
Standard: ISO 22095:2020 Chain of Custody - Mass Balance
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversion factors by commodity and processing step
# ---------------------------------------------------------------------------

CONVERSION_FACTORS: List[Dict[str, Any]] = [
    # ======================================================================
    # COCOA  (Source: ICCO - International Cocoa Organization)
    # ======================================================================
    {
        "commodity": "cocoa",
        "process": "shelling",
        "input_material": "cocoa_beans",
        "output_material": "cocoa_nibs",
        "yield_ratio": 0.87,
        "acceptable_range_min": 0.82,
        "acceptable_range_max": 0.92,
        "source": "ICCO",
        "notes": "Shell removal; shell fraction 10-18% depending on origin",
        "data_year": 2024,
    },
    {
        "commodity": "cocoa",
        "process": "grinding",
        "input_material": "cocoa_nibs",
        "output_material": "cocoa_liquor",
        "yield_ratio": 0.80,
        "acceptable_range_min": 0.75,
        "acceptable_range_max": 0.85,
        "source": "ICCO",
        "notes": "Nib grinding to cocoa mass/liquor; moisture loss 2-5%",
        "data_year": 2024,
    },
    {
        "commodity": "cocoa",
        "process": "pressing_butter",
        "input_material": "cocoa_liquor",
        "output_material": "cocoa_butter",
        "yield_ratio": 0.45,
        "acceptable_range_min": 0.40,
        "acceptable_range_max": 0.50,
        "source": "ICCO",
        "notes": "Hydraulic pressing; co-product is cocoa press cake",
        "data_year": 2024,
    },
    {
        "commodity": "cocoa",
        "process": "pressing_powder",
        "input_material": "cocoa_liquor",
        "output_material": "cocoa_powder",
        "yield_ratio": 0.55,
        "acceptable_range_min": 0.50,
        "acceptable_range_max": 0.60,
        "source": "ICCO",
        "notes": "Press cake pulverization; complementary to butter yield",
        "data_year": 2024,
    },
    {
        "commodity": "cocoa",
        "process": "alkalization",
        "input_material": "cocoa_nibs",
        "output_material": "alkalized_nibs",
        "yield_ratio": 0.96,
        "acceptable_range_min": 0.92,
        "acceptable_range_max": 0.99,
        "source": "ICCO",
        "notes": "Dutch process alkalization; minor mass change from added potash",
        "data_year": 2024,
    },
    {
        "commodity": "cocoa",
        "process": "couverture_blending",
        "input_material": "cocoa_liquor",
        "output_material": "couverture_chocolate",
        "yield_ratio": 0.65,
        "acceptable_range_min": 0.55,
        "acceptable_range_max": 0.75,
        "source": "ICCO",
        "notes": "Cocoa liquor fraction in couverture; varies by cocoa %",
        "data_year": 2024,
    },

    # ======================================================================
    # PALM OIL  (Source: MPOB - Malaysian Palm Oil Board)
    # ======================================================================
    {
        "commodity": "oil_palm",
        "process": "milling_cpo",
        "input_material": "fresh_fruit_bunches",
        "output_material": "crude_palm_oil",
        "yield_ratio": 0.215,
        "acceptable_range_min": 0.18,
        "acceptable_range_max": 0.25,
        "source": "MPOB",
        "notes": "OER (Oil Extraction Rate); varies by variety and age",
        "data_year": 2024,
    },
    {
        "commodity": "oil_palm",
        "process": "milling_pko",
        "input_material": "fresh_fruit_bunches",
        "output_material": "palm_kernel_oil",
        "yield_ratio": 0.035,
        "acceptable_range_min": 0.025,
        "acceptable_range_max": 0.045,
        "source": "MPOB",
        "notes": "Kernel extraction and pressing; kernel 5-7% of FFB",
        "data_year": 2024,
    },
    {
        "commodity": "oil_palm",
        "process": "refining",
        "input_material": "crude_palm_oil",
        "output_material": "rbd_palm_oil",
        "yield_ratio": 0.92,
        "acceptable_range_min": 0.88,
        "acceptable_range_max": 0.96,
        "source": "MPOB",
        "notes": "RBD (Refined Bleached Deodorized); loss from FFA removal",
        "data_year": 2024,
    },
    {
        "commodity": "oil_palm",
        "process": "fractionation_olein",
        "input_material": "rbd_palm_oil",
        "output_material": "palm_olein",
        "yield_ratio": 0.70,
        "acceptable_range_min": 0.65,
        "acceptable_range_max": 0.78,
        "source": "MPOB",
        "notes": "Dry fractionation; liquid fraction at ambient temperature",
        "data_year": 2024,
    },
    {
        "commodity": "oil_palm",
        "process": "fractionation_stearin",
        "input_material": "rbd_palm_oil",
        "output_material": "palm_stearin",
        "yield_ratio": 0.30,
        "acceptable_range_min": 0.22,
        "acceptable_range_max": 0.35,
        "source": "MPOB",
        "notes": "Dry fractionation; solid fraction complement to olein",
        "data_year": 2024,
    },
    {
        "commodity": "oil_palm",
        "process": "kernel_crushing",
        "input_material": "palm_kernels",
        "output_material": "palm_kernel_oil",
        "yield_ratio": 0.45,
        "acceptable_range_min": 0.40,
        "acceptable_range_max": 0.52,
        "source": "MPOB",
        "notes": "Mechanical pressing; co-product is palm kernel cake",
        "data_year": 2024,
    },

    # ======================================================================
    # COFFEE  (Source: ICO - International Coffee Organization)
    # ======================================================================
    {
        "commodity": "coffee",
        "process": "wet_processing",
        "input_material": "coffee_cherry",
        "output_material": "green_coffee",
        "yield_ratio": 0.185,
        "acceptable_range_min": 0.15,
        "acceptable_range_max": 0.22,
        "source": "ICO",
        "notes": "Wet/washed process: pulping, fermenting, washing, drying",
        "data_year": 2024,
    },
    {
        "commodity": "coffee",
        "process": "dry_processing",
        "input_material": "coffee_cherry",
        "output_material": "green_coffee",
        "yield_ratio": 0.20,
        "acceptable_range_min": 0.17,
        "acceptable_range_max": 0.24,
        "source": "ICO",
        "notes": "Natural/dry process: whole cherry sun-dried then hulled",
        "data_year": 2024,
    },
    {
        "commodity": "coffee",
        "process": "roasting",
        "input_material": "green_coffee",
        "output_material": "roasted_coffee",
        "yield_ratio": 0.825,
        "acceptable_range_min": 0.78,
        "acceptable_range_max": 0.87,
        "source": "ICO",
        "notes": "Moisture and volatile loss during roasting; 12-20% weight loss",
        "data_year": 2024,
    },
    {
        "commodity": "coffee",
        "process": "decaffeination",
        "input_material": "green_coffee",
        "output_material": "decaf_green_coffee",
        "yield_ratio": 0.96,
        "acceptable_range_min": 0.93,
        "acceptable_range_max": 0.99,
        "source": "ICO",
        "notes": "Caffeine extraction via solvent/CO2/water; ~2-4% mass loss",
        "data_year": 2024,
    },
    {
        "commodity": "coffee",
        "process": "instant_extraction",
        "input_material": "roasted_coffee",
        "output_material": "instant_coffee",
        "yield_ratio": 0.38,
        "acceptable_range_min": 0.32,
        "acceptable_range_max": 0.44,
        "source": "ICO",
        "notes": "Spray or freeze-dried extraction; soluble yield varies by bean",
        "data_year": 2024,
    },

    # ======================================================================
    # SOYA  (Source: NOPA - National Oilseed Processors Association)
    # ======================================================================
    {
        "commodity": "soya",
        "process": "oil_extraction",
        "input_material": "soybeans",
        "output_material": "crude_soya_oil",
        "yield_ratio": 0.19,
        "acceptable_range_min": 0.16,
        "acceptable_range_max": 0.22,
        "source": "NOPA",
        "notes": "Hexane solvent extraction; oil content 18-22% of bean",
        "data_year": 2024,
    },
    {
        "commodity": "soya",
        "process": "meal_production",
        "input_material": "soybeans",
        "output_material": "soya_meal",
        "yield_ratio": 0.795,
        "acceptable_range_min": 0.75,
        "acceptable_range_max": 0.84,
        "source": "NOPA",
        "notes": "Defatted meal after extraction; complementary to oil yield",
        "data_year": 2024,
    },
    {
        "commodity": "soya",
        "process": "oil_refining",
        "input_material": "crude_soya_oil",
        "output_material": "refined_soya_oil",
        "yield_ratio": 0.95,
        "acceptable_range_min": 0.92,
        "acceptable_range_max": 0.98,
        "source": "NOPA",
        "notes": "Degumming, neutralization, bleaching, deodorization",
        "data_year": 2024,
    },
    {
        "commodity": "soya",
        "process": "lecithin_recovery",
        "input_material": "crude_soya_oil",
        "output_material": "soya_lecithin",
        "yield_ratio": 0.03,
        "acceptable_range_min": 0.02,
        "acceptable_range_max": 0.05,
        "source": "NOPA",
        "notes": "Water degumming phospholipid recovery; 2-3% of crude oil",
        "data_year": 2024,
    },
    {
        "commodity": "soya",
        "process": "protein_concentrate",
        "input_material": "soya_meal",
        "output_material": "soya_protein_concentrate",
        "yield_ratio": 0.65,
        "acceptable_range_min": 0.58,
        "acceptable_range_max": 0.72,
        "source": "NOPA",
        "notes": "Aqueous ethanol wash to remove soluble sugars; 65-70% protein",
        "data_year": 2024,
    },

    # ======================================================================
    # RUBBER  (Source: IRSG - International Rubber Study Group)
    # ======================================================================
    {
        "commodity": "rubber",
        "process": "sheet_drying",
        "input_material": "field_latex",
        "output_material": "ribbed_smoked_sheet",
        "yield_ratio": 0.325,
        "acceptable_range_min": 0.28,
        "acceptable_range_max": 0.37,
        "source": "IRSG",
        "notes": "Coagulation, sheeting, smoking; DRC of field latex ~30-35%",
        "data_year": 2024,
    },
    {
        "commodity": "rubber",
        "process": "block_production",
        "input_material": "field_latex",
        "output_material": "technically_specified_rubber",
        "yield_ratio": 0.30,
        "acceptable_range_min": 0.25,
        "acceptable_range_max": 0.35,
        "source": "IRSG",
        "notes": "Cup lump/cuplump processing to TSR (e.g. STR-20, SMR-20)",
        "data_year": 2024,
    },
    {
        "commodity": "rubber",
        "process": "concentrate_centrifuge",
        "input_material": "field_latex",
        "output_material": "latex_concentrate",
        "yield_ratio": 0.40,
        "acceptable_range_min": 0.35,
        "acceptable_range_max": 0.48,
        "source": "IRSG",
        "notes": "Centrifugal concentration to 60% DRC; skim as co-product",
        "data_year": 2024,
    },
    {
        "commodity": "rubber",
        "process": "crepe_production",
        "input_material": "field_latex",
        "output_material": "crepe_rubber",
        "yield_ratio": 0.31,
        "acceptable_range_min": 0.26,
        "acceptable_range_max": 0.36,
        "source": "IRSG",
        "notes": "Coagulation and creping rolls; pale crepe from fresh latex",
        "data_year": 2024,
    },

    # ======================================================================
    # WOOD  (Source: ITTO - International Tropical Timber Organization)
    # ======================================================================
    {
        "commodity": "wood",
        "process": "sawmilling",
        "input_material": "roundwood_logs",
        "output_material": "sawn_timber",
        "yield_ratio": 0.50,
        "acceptable_range_min": 0.40,
        "acceptable_range_max": 0.60,
        "source": "ITTO",
        "notes": "Log-to-lumber recovery; varies by species, log diameter, equipment",
        "data_year": 2024,
    },
    {
        "commodity": "wood",
        "process": "veneer_peeling",
        "input_material": "roundwood_logs",
        "output_material": "veneer_sheets",
        "yield_ratio": 0.55,
        "acceptable_range_min": 0.45,
        "acceptable_range_max": 0.65,
        "source": "ITTO",
        "notes": "Rotary peeling; core waste 10-15%, clipping loss 10-15%",
        "data_year": 2024,
    },
    {
        "commodity": "wood",
        "process": "plywood_layup",
        "input_material": "veneer_sheets",
        "output_material": "plywood_panels",
        "yield_ratio": 0.85,
        "acceptable_range_min": 0.78,
        "acceptable_range_max": 0.92,
        "source": "ITTO",
        "notes": "Veneer-to-plywood; adhesive adds ~2-5% mass, trim loss 5-10%",
        "data_year": 2024,
    },
    {
        "commodity": "wood",
        "process": "chipping",
        "input_material": "roundwood_logs",
        "output_material": "wood_chips",
        "yield_ratio": 0.88,
        "acceptable_range_min": 0.82,
        "acceptable_range_max": 0.95,
        "source": "ITTO",
        "notes": "Chipping for pulp/paper; bark and fines as co-products",
        "data_year": 2024,
    },
    {
        "commodity": "wood",
        "process": "pulping",
        "input_material": "wood_chips",
        "output_material": "wood_pulp",
        "yield_ratio": 0.48,
        "acceptable_range_min": 0.42,
        "acceptable_range_max": 0.55,
        "source": "ITTO",
        "notes": "Chemical (kraft) pulping; dissolves lignin ~50% of wood mass",
        "data_year": 2024,
    },
    {
        "commodity": "wood",
        "process": "kiln_drying",
        "input_material": "sawn_timber",
        "output_material": "kiln_dried_timber",
        "yield_ratio": 0.90,
        "acceptable_range_min": 0.85,
        "acceptable_range_max": 0.95,
        "source": "ITTO",
        "notes": "Moisture removal from green to 8-12% MC; 5-15% weight loss",
        "data_year": 2024,
    },

    # ======================================================================
    # CATTLE  (Source: FAO, ICHSLTA)
    # ======================================================================
    {
        "commodity": "cattle",
        "process": "slaughter_carcass",
        "input_material": "live_cattle",
        "output_material": "carcass",
        "yield_ratio": 0.55,
        "acceptable_range_min": 0.48,
        "acceptable_range_max": 0.62,
        "source": "FAO",
        "notes": "Dressing percentage; varies by breed and finish weight",
        "data_year": 2024,
    },
    {
        "commodity": "cattle",
        "process": "hide_recovery",
        "input_material": "live_cattle",
        "output_material": "raw_hide",
        "yield_ratio": 0.07,
        "acceptable_range_min": 0.05,
        "acceptable_range_max": 0.09,
        "source": "ICHSLTA",
        "notes": "Green salted hide weight relative to live weight",
        "data_year": 2024,
    },
    {
        "commodity": "cattle",
        "process": "deboning",
        "input_material": "carcass",
        "output_material": "boneless_meat",
        "yield_ratio": 0.72,
        "acceptable_range_min": 0.65,
        "acceptable_range_max": 0.78,
        "source": "FAO",
        "notes": "Bone-out yield; bone fraction 20-30% of carcass",
        "data_year": 2024,
    },
    {
        "commodity": "cattle",
        "process": "tanning",
        "input_material": "raw_hide",
        "output_material": "finished_leather",
        "yield_ratio": 0.55,
        "acceptable_range_min": 0.45,
        "acceptable_range_max": 0.65,
        "source": "ICHSLTA",
        "notes": "Chrome tanning yield; significant mass loss from fleshing/liming",
        "data_year": 2024,
    },
    {
        "commodity": "cattle",
        "process": "offal_recovery",
        "input_material": "live_cattle",
        "output_material": "edible_offal",
        "yield_ratio": 0.12,
        "acceptable_range_min": 0.08,
        "acceptable_range_max": 0.16,
        "source": "FAO",
        "notes": "Liver, heart, kidney, tongue, tripe as fraction of live weight",
        "data_year": 2024,
    },
    {
        "commodity": "cattle",
        "process": "rendering",
        "input_material": "slaughter_byproducts",
        "output_material": "tallow",
        "yield_ratio": 0.35,
        "acceptable_range_min": 0.28,
        "acceptable_range_max": 0.42,
        "source": "FAO",
        "notes": "Fat rendering from trimmings, bones, offal waste",
        "data_year": 2024,
    },
]

# ---------------------------------------------------------------------------
# Total entries counter
# ---------------------------------------------------------------------------

TOTAL_CONVERSION_FACTORS: int = len(CONVERSION_FACTORS)

# ---------------------------------------------------------------------------
# Two-level lookup index: {commodity: {process: factor_dict}}
# ---------------------------------------------------------------------------

COMMODITY_PROCESS_INDEX: Dict[str, Dict[str, Dict[str, Any]]] = {}

for _factor in CONVERSION_FACTORS:
    _commodity = _factor["commodity"]
    _process = _factor["process"]
    if _commodity not in COMMODITY_PROCESS_INDEX:
        COMMODITY_PROCESS_INDEX[_commodity] = {}
    COMMODITY_PROCESS_INDEX[_commodity][_process] = _factor

TOTAL_COMMODITIES: int = len(COMMODITY_PROCESS_INDEX)

# ---------------------------------------------------------------------------
# Lookup helper functions
# ---------------------------------------------------------------------------


def get_reference_factor(
    commodity: str,
    process: str,
) -> Optional[Dict[str, Any]]:
    """Return the reference conversion factor for a commodity-process pair.

    Looks up the canonical yield ratio and acceptable range for a
    specific commodity processing step.

    Args:
        commodity: EUDR commodity (cocoa, oil_palm, coffee, soya,
            rubber, wood, cattle).
        process: Processing step identifier (e.g. "shelling",
            "milling_cpo", "wet_processing").

    Returns:
        Dictionary with yield_ratio, acceptable_range_min/max, source,
        and notes.  Returns None if the combination is not found.

    Example:
        >>> factor = get_reference_factor("cocoa", "shelling")
        >>> factor["yield_ratio"]
        0.87
    """
    commodity_factors = COMMODITY_PROCESS_INDEX.get(commodity)
    if commodity_factors is None:
        return None
    return commodity_factors.get(process)


def get_all_factors(commodity: str) -> List[Dict[str, Any]]:
    """Return all conversion factors for a given commodity.

    Args:
        commodity: EUDR commodity identifier.

    Returns:
        List of factor dictionaries for the commodity.
        Returns empty list if commodity is not found.

    Example:
        >>> factors = get_all_factors("coffee")
        >>> len(factors) >= 3
        True
    """
    return [
        f for f in CONVERSION_FACTORS
        if f["commodity"] == commodity
    ]


def get_factor_range(
    commodity: str,
    process: str,
) -> Optional[Tuple[float, float]]:
    """Return the acceptable range (min, max) for a commodity-process pair.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.

    Returns:
        Tuple of (acceptable_range_min, acceptable_range_max).
        Returns None if the combination is not found.

    Example:
        >>> get_factor_range("oil_palm", "milling_cpo")
        (0.18, 0.25)
    """
    factor = get_reference_factor(commodity, process)
    if factor is None:
        return None
    return (
        float(factor["acceptable_range_min"]),
        float(factor["acceptable_range_max"]),
    )


def is_within_range(
    commodity: str,
    process: str,
    actual_ratio: float,
) -> bool:
    """Check if an actual yield ratio falls within the acceptable range.

    This is the primary validation used by the ConversionFactorValidator
    engine.  A ratio outside this range triggers a warning or rejection
    depending on the deviation magnitude.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.
        actual_ratio: The actual yield ratio observed in production.

    Returns:
        True if ``acceptable_range_min <= actual_ratio <= acceptable_range_max``.
        Returns False if the commodity-process combination is unknown
        (fail-safe: unknown conversion treated as out-of-range).

    Example:
        >>> is_within_range("cocoa", "shelling", 0.85)
        True
        >>> is_within_range("cocoa", "shelling", 0.70)
        False
    """
    factor_range = get_factor_range(commodity, process)
    if factor_range is None:
        return False
    range_min, range_max = factor_range
    return range_min <= actual_ratio <= range_max


def get_expected_yield(commodity: str, process: str) -> Optional[float]:
    """Return the expected (reference) yield ratio for a commodity-process pair.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.

    Returns:
        Expected yield ratio as float, or None if not found.

    Example:
        >>> get_expected_yield("soya", "oil_extraction")
        0.19
    """
    factor = get_reference_factor(commodity, process)
    if factor is None:
        return None
    return float(factor["yield_ratio"])


def get_source(commodity: str, process: str) -> Optional[str]:
    """Return the authoritative source for a conversion factor.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.

    Returns:
        Source organization abbreviation, or None if not found.

    Example:
        >>> get_source("rubber", "sheet_drying")
        'IRSG'
    """
    factor = get_reference_factor(commodity, process)
    if factor is None:
        return None
    return str(factor.get("source", ""))


def get_all_commodities() -> List[str]:
    """Return all commodity identifiers that have conversion factors.

    Returns:
        Sorted list of commodity identifier strings.

    Example:
        >>> commodities = get_all_commodities()
        >>> "cocoa" in commodities
        True
    """
    return sorted(COMMODITY_PROCESS_INDEX.keys())


def get_all_processes(commodity: str) -> List[str]:
    """Return all processing step identifiers for a commodity.

    Args:
        commodity: EUDR commodity identifier.

    Returns:
        Sorted list of process identifier strings.
        Returns empty list if commodity is not found.

    Example:
        >>> processes = get_all_processes("oil_palm")
        >>> "milling_cpo" in processes
        True
    """
    commodity_factors = COMMODITY_PROCESS_INDEX.get(commodity)
    if commodity_factors is None:
        return []
    return sorted(commodity_factors.keys())


def compute_deviation_pct(
    commodity: str,
    process: str,
    actual_ratio: float,
) -> Optional[float]:
    """Compute the percentage deviation from the reference yield ratio.

    Positive values indicate the actual ratio is higher than expected;
    negative values indicate it is lower.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.
        actual_ratio: The actual yield ratio observed.

    Returns:
        Deviation as a percentage (e.g. 5.0 means 5% above reference).
        Returns None if the commodity-process is unknown or the
        reference ratio is zero.

    Example:
        >>> compute_deviation_pct("cocoa", "shelling", 0.90)
        3.45  # approximately (0.90 - 0.87) / 0.87 * 100
    """
    factor = get_reference_factor(commodity, process)
    if factor is None:
        return None
    ref = float(factor["yield_ratio"])
    if ref == 0.0:
        return None
    return round(((actual_ratio - ref) / ref) * 100.0, 2)


# ---------------------------------------------------------------------------
# Module-level logging on import
# ---------------------------------------------------------------------------

logger.debug(
    "Conversion factors reference data loaded: "
    "%d factors across %d commodities",
    TOTAL_CONVERSION_FACTORS,
    TOTAL_COMMODITIES,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "CONVERSION_FACTORS",
    "COMMODITY_PROCESS_INDEX",
    # Counts
    "TOTAL_CONVERSION_FACTORS",
    "TOTAL_COMMODITIES",
    # Lookup helpers
    "get_reference_factor",
    "get_all_factors",
    "get_factor_range",
    "is_within_range",
    "get_expected_yield",
    "get_source",
    "get_all_commodities",
    "get_all_processes",
    "compute_deviation_pct",
]
