# -*- coding: utf-8 -*-
"""
Loss Tolerances Reference Data - AGENT-EUDR-011 Mass Balance Calculator

Commodity-specific loss and waste tolerance limits for mass balance
compliance.  Each entry defines the expected and maximum allowable
loss percentage for a specific commodity processing step, transport
mode, or storage condition.  Values exceeding the maximum tolerance
trigger an investigation flag in the LossWasteTracker engine.

Datasets:
    PROCESSING_LOSS_TOLERANCES:
        Loss tolerances for commodity processing/transformation steps.
        Each entry: commodity, process, loss_type, expected_loss_pct,
        max_loss_pct, source, notes.

    TRANSPORT_LOSS_TOLERANCES:
        Loss tolerances during transportation by mode and commodity.
        Each entry: commodity, transport_mode, expected_loss_pct,
        max_loss_pct, distance_factor_per_1000km, source, notes.

    STORAGE_LOSS_TOLERANCES:
        Loss tolerances during storage by condition and commodity.
        Each entry: commodity, storage_condition, expected_loss_monthly_pct,
        max_loss_monthly_pct, max_storage_months, source, notes.

    BYPRODUCT_RECOVERY_RATES:
        Expected by-product recovery fractions for processing steps.
        Each entry: commodity, process, byproduct, recovery_rate,
        acceptable_range_min, acceptable_range_max, source.

Lookup helpers:
    get_loss_tolerance(commodity, process) -> dict | None
    is_within_tolerance(commodity, process, actual_loss_pct) -> bool
    get_all_tolerances(commodity) -> list[dict]
    get_expected_loss(commodity, process) -> float

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011 (Mass Balance Calculator) - Appendix B
Agent ID: GL-EUDR-MBC-011
Regulation: EU 2023/1115 (EUDR) Articles 4, 10(2)(f), 14
Standard: ISO 22095:2020 Chain of Custody - Mass Balance
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Processing loss tolerances by commodity and process
# ---------------------------------------------------------------------------

PROCESSING_LOSS_TOLERANCES: List[Dict[str, Any]] = [
    # ======================================================================
    # COCOA  (Source: ICCO)
    # ======================================================================
    {
        "commodity": "cocoa",
        "process": "shelling",
        "loss_type": "shell_removal",
        "expected_loss_pct": 13.0,
        "max_loss_pct": 20.0,
        "source": "ICCO",
        "notes": "Shell fraction 10-18%; losses include dust and fines",
    },
    {
        "commodity": "cocoa",
        "process": "grinding",
        "loss_type": "moisture_and_fines",
        "expected_loss_pct": 2.0,
        "max_loss_pct": 5.0,
        "source": "ICCO",
        "notes": "Grinding loss from moisture evaporation and fine dust",
    },
    {
        "commodity": "cocoa",
        "process": "pressing",
        "loss_type": "press_residue",
        "expected_loss_pct": 1.5,
        "max_loss_pct": 4.0,
        "source": "ICCO",
        "notes": "Residual liquor in press equipment and filter cloth",
    },
    {
        "commodity": "cocoa",
        "process": "alkalization",
        "loss_type": "chemical_reaction",
        "expected_loss_pct": 2.0,
        "max_loss_pct": 5.0,
        "source": "ICCO",
        "notes": "Washing and neutralization losses in Dutch processing",
    },
    {
        "commodity": "cocoa",
        "process": "conching",
        "loss_type": "moisture_evaporation",
        "expected_loss_pct": 1.0,
        "max_loss_pct": 3.0,
        "source": "ICCO",
        "notes": "Moisture and volatile acid loss during conching",
    },

    # ======================================================================
    # PALM OIL  (Source: MPOB)
    # ======================================================================
    {
        "commodity": "oil_palm",
        "process": "milling",
        "loss_type": "extraction_loss",
        "expected_loss_pct": 78.5,
        "max_loss_pct": 82.0,
        "source": "MPOB",
        "notes": "Non-oil fraction: EFB, fibre, shell, POME; OER 18-25%",
    },
    {
        "commodity": "oil_palm",
        "process": "refining",
        "loss_type": "refining_loss",
        "expected_loss_pct": 5.0,
        "max_loss_pct": 10.0,
        "source": "MPOB",
        "notes": "FFA removal, bleaching earth absorption, deodorizer distillate",
    },
    {
        "commodity": "oil_palm",
        "process": "fractionation",
        "loss_type": "separation_loss",
        "expected_loss_pct": 1.5,
        "max_loss_pct": 4.0,
        "source": "MPOB",
        "notes": "Filter cake and mid-fraction losses during crystallization",
    },
    {
        "commodity": "oil_palm",
        "process": "kernel_processing",
        "loss_type": "cracking_loss",
        "expected_loss_pct": 8.0,
        "max_loss_pct": 15.0,
        "source": "MPOB",
        "notes": "Shell separation, kernel breakage, and moisture",
    },

    # ======================================================================
    # COFFEE  (Source: ICO)
    # ======================================================================
    {
        "commodity": "coffee",
        "process": "wet_processing",
        "loss_type": "pulp_and_mucilage",
        "expected_loss_pct": 81.5,
        "max_loss_pct": 85.0,
        "source": "ICO",
        "notes": "Pulp 40-45%, mucilage 15-22%, parchment 5-8%, moisture",
    },
    {
        "commodity": "coffee",
        "process": "dry_processing",
        "loss_type": "husk_and_moisture",
        "expected_loss_pct": 80.0,
        "max_loss_pct": 83.0,
        "source": "ICO",
        "notes": "Dried cherry husk removal; natural process whole-fruit drying",
    },
    {
        "commodity": "coffee",
        "process": "roasting",
        "loss_type": "moisture_and_volatiles",
        "expected_loss_pct": 17.5,
        "max_loss_pct": 22.0,
        "source": "ICO",
        "notes": "Weight loss 12-20% depending on roast level (light to dark)",
    },
    {
        "commodity": "coffee",
        "process": "grinding",
        "loss_type": "dust_and_fines",
        "expected_loss_pct": 0.5,
        "max_loss_pct": 2.0,
        "source": "ICO",
        "notes": "Fine particles lost during milling; equipment retention",
    },

    # ======================================================================
    # SOYA  (Source: NOPA)
    # ======================================================================
    {
        "commodity": "soya",
        "process": "oil_extraction",
        "loss_type": "solvent_extraction_loss",
        "expected_loss_pct": 1.0,
        "max_loss_pct": 3.0,
        "source": "NOPA",
        "notes": "Combined oil+meal yield is 98-99%; residual solvent loss",
    },
    {
        "commodity": "soya",
        "process": "dehulling",
        "loss_type": "hull_removal",
        "expected_loss_pct": 8.0,
        "max_loss_pct": 12.0,
        "source": "NOPA",
        "notes": "Hull fraction 6-10%; hulls used as fiber feed additive",
    },
    {
        "commodity": "soya",
        "process": "conditioning",
        "loss_type": "moisture_adjustment",
        "expected_loss_pct": 1.5,
        "max_loss_pct": 3.0,
        "source": "NOPA",
        "notes": "Moisture tempering and flaking preparation losses",
    },

    # ======================================================================
    # RUBBER  (Source: IRSG)
    # ======================================================================
    {
        "commodity": "rubber",
        "process": "coagulation_drying",
        "loss_type": "serum_loss",
        "expected_loss_pct": 67.5,
        "max_loss_pct": 72.0,
        "source": "IRSG",
        "notes": "Field latex DRC ~30-35%; serum water 60-70%",
    },
    {
        "commodity": "rubber",
        "process": "smoking",
        "loss_type": "moisture_evaporation",
        "expected_loss_pct": 3.0,
        "max_loss_pct": 6.0,
        "source": "IRSG",
        "notes": "Further drying during smokehouse curing; 3-7 days",
    },
    {
        "commodity": "rubber",
        "process": "crumb_production",
        "loss_type": "washing_loss",
        "expected_loss_pct": 2.0,
        "max_loss_pct": 5.0,
        "source": "IRSG",
        "notes": "Contaminant removal during crumbing and washing",
    },

    # ======================================================================
    # WOOD  (Source: ITTO)
    # ======================================================================
    {
        "commodity": "wood",
        "process": "sawmilling",
        "loss_type": "sawdust_and_offcuts",
        "expected_loss_pct": 50.0,
        "max_loss_pct": 60.0,
        "source": "ITTO",
        "notes": "Sawdust 10-15%, slabs/offcuts 25-35%, bark 5-10%",
    },
    {
        "commodity": "wood",
        "process": "veneer_peeling",
        "loss_type": "core_and_clipping",
        "expected_loss_pct": 45.0,
        "max_loss_pct": 55.0,
        "source": "ITTO",
        "notes": "Peeler core 10-15%, clipping trim 10-15%, roundup waste 10-15%",
    },
    {
        "commodity": "wood",
        "process": "kiln_drying",
        "loss_type": "moisture_removal",
        "expected_loss_pct": 10.0,
        "max_loss_pct": 15.0,
        "source": "ITTO",
        "notes": "Green MC 40-80% to target 8-12%; degrade losses 1-5%",
    },
    {
        "commodity": "wood",
        "process": "planing",
        "loss_type": "surface_removal",
        "expected_loss_pct": 8.0,
        "max_loss_pct": 12.0,
        "source": "ITTO",
        "notes": "Surface planing shavings; depends on surface quality required",
    },

    # ======================================================================
    # CATTLE  (Source: FAO)
    # ======================================================================
    {
        "commodity": "cattle",
        "process": "slaughter",
        "loss_type": "non_carcass_fraction",
        "expected_loss_pct": 45.0,
        "max_loss_pct": 52.0,
        "source": "FAO",
        "notes": "Blood 3-4%, gut content 8-12%, head 3-4%, hooves 1-2%, offal variable",
    },
    {
        "commodity": "cattle",
        "process": "deboning",
        "loss_type": "bone_and_trim",
        "expected_loss_pct": 28.0,
        "max_loss_pct": 35.0,
        "source": "FAO",
        "notes": "Bone 20-25%, fat trim 5-10% depending on grade specification",
    },
    {
        "commodity": "cattle",
        "process": "chilling",
        "loss_type": "drip_loss",
        "expected_loss_pct": 1.5,
        "max_loss_pct": 3.0,
        "source": "FAO",
        "notes": "Evaporative and drip loss during 24-48 hour chilling",
    },
    {
        "commodity": "cattle",
        "process": "tanning",
        "loss_type": "fleshing_and_liming",
        "expected_loss_pct": 45.0,
        "max_loss_pct": 55.0,
        "source": "ICHSLTA",
        "notes": "Hair removal, fleshing, liming; significant mass reduction",
    },
]

# ---------------------------------------------------------------------------
# Transport loss tolerances
# ---------------------------------------------------------------------------

TRANSPORT_LOSS_TOLERANCES: List[Dict[str, Any]] = [
    # -- Bulk commodities: road transport --
    {
        "commodity": "cocoa",
        "transport_mode": "road_truck",
        "expected_loss_pct": 0.2,
        "max_loss_pct": 0.5,
        "distance_factor_per_1000km": 0.05,
        "source": "ICCO",
        "notes": "Bagged transport; moisture absorption and spillage",
    },
    {
        "commodity": "oil_palm",
        "transport_mode": "road_truck",
        "expected_loss_pct": 0.5,
        "max_loss_pct": 1.5,
        "distance_factor_per_1000km": 0.10,
        "source": "MPOB",
        "notes": "FFB bruising and loose fruit spillage en route to mill",
    },
    {
        "commodity": "coffee",
        "transport_mode": "road_truck",
        "expected_loss_pct": 0.15,
        "max_loss_pct": 0.4,
        "distance_factor_per_1000km": 0.03,
        "source": "ICO",
        "notes": "Bagged green coffee; moisture change risk",
    },
    {
        "commodity": "soya",
        "transport_mode": "road_truck",
        "expected_loss_pct": 0.3,
        "max_loss_pct": 0.8,
        "distance_factor_per_1000km": 0.05,
        "source": "NOPA",
        "notes": "Bulk bean transport; spillage and dust loss",
    },
    {
        "commodity": "rubber",
        "transport_mode": "road_truck",
        "expected_loss_pct": 0.1,
        "max_loss_pct": 0.3,
        "distance_factor_per_1000km": 0.02,
        "source": "IRSG",
        "notes": "Baled rubber; minimal loss during transport",
    },
    {
        "commodity": "wood",
        "transport_mode": "road_truck",
        "expected_loss_pct": 0.5,
        "max_loss_pct": 1.0,
        "distance_factor_per_1000km": 0.08,
        "source": "ITTO",
        "notes": "Log/timber transport; bark loss and breakage",
    },
    {
        "commodity": "cattle",
        "transport_mode": "road_truck",
        "expected_loss_pct": 2.0,
        "max_loss_pct": 5.0,
        "distance_factor_per_1000km": 0.50,
        "source": "FAO",
        "notes": "Live animal weight loss (shrink) during transit",
    },
    # -- Bulk commodities: sea transport --
    {
        "commodity": "cocoa",
        "transport_mode": "sea_container",
        "expected_loss_pct": 0.3,
        "max_loss_pct": 1.0,
        "distance_factor_per_1000km": 0.02,
        "source": "ICCO",
        "notes": "Container shipping; moisture migration risk",
    },
    {
        "commodity": "oil_palm",
        "transport_mode": "sea_tanker",
        "expected_loss_pct": 0.1,
        "max_loss_pct": 0.3,
        "distance_factor_per_1000km": 0.01,
        "source": "MPOB",
        "notes": "Bulk liquid CPO/RBD tanker transport; heel loss",
    },
    {
        "commodity": "coffee",
        "transport_mode": "sea_container",
        "expected_loss_pct": 0.2,
        "max_loss_pct": 0.5,
        "distance_factor_per_1000km": 0.02,
        "source": "ICO",
        "notes": "Green coffee in GrainPro liners; moisture gain/loss risk",
    },
    {
        "commodity": "soya",
        "transport_mode": "sea_bulk",
        "expected_loss_pct": 0.4,
        "max_loss_pct": 1.0,
        "distance_factor_per_1000km": 0.03,
        "source": "NOPA",
        "notes": "Bulk carrier; moisture and dust losses",
    },
    {
        "commodity": "wood",
        "transport_mode": "sea_container",
        "expected_loss_pct": 0.2,
        "max_loss_pct": 0.5,
        "distance_factor_per_1000km": 0.02,
        "source": "ITTO",
        "notes": "Sawn timber in containers; moisture variation",
    },
    # -- Rail transport --
    {
        "commodity": "soya",
        "transport_mode": "rail",
        "expected_loss_pct": 0.2,
        "max_loss_pct": 0.5,
        "distance_factor_per_1000km": 0.03,
        "source": "NOPA",
        "notes": "Hopper car bulk transport; less handling than truck",
    },
    {
        "commodity": "wood",
        "transport_mode": "rail",
        "expected_loss_pct": 0.3,
        "max_loss_pct": 0.7,
        "distance_factor_per_1000km": 0.04,
        "source": "ITTO",
        "notes": "Log/lumber flat car transport; bark and chip loss",
    },
]

# ---------------------------------------------------------------------------
# Storage loss tolerances
# ---------------------------------------------------------------------------

STORAGE_LOSS_TOLERANCES: List[Dict[str, Any]] = [
    {
        "commodity": "cocoa",
        "storage_condition": "warehouse_ambient",
        "expected_loss_monthly_pct": 0.1,
        "max_loss_monthly_pct": 0.3,
        "max_storage_months": 12,
        "source": "ICCO",
        "notes": "Dried beans at <8% MC; pest damage risk if uncontrolled",
    },
    {
        "commodity": "cocoa",
        "storage_condition": "warehouse_controlled",
        "expected_loss_monthly_pct": 0.05,
        "max_loss_monthly_pct": 0.15,
        "max_storage_months": 18,
        "source": "ICCO",
        "notes": "Temperature and humidity controlled; fumigated",
    },
    {
        "commodity": "oil_palm",
        "storage_condition": "tank_heated",
        "expected_loss_monthly_pct": 0.05,
        "max_loss_monthly_pct": 0.15,
        "max_storage_months": 6,
        "source": "MPOB",
        "notes": "CPO in heated tanks; oxidation and FFA increase risk",
    },
    {
        "commodity": "oil_palm",
        "storage_condition": "ffb_ramp",
        "expected_loss_monthly_pct": 5.0,
        "max_loss_monthly_pct": 10.0,
        "max_storage_months": 0,
        "source": "MPOB",
        "notes": "FFB must be processed within 24-48h; rapid FFA increase",
    },
    {
        "commodity": "coffee",
        "storage_condition": "warehouse_ambient",
        "expected_loss_monthly_pct": 0.1,
        "max_loss_monthly_pct": 0.25,
        "max_storage_months": 12,
        "source": "ICO",
        "notes": "Green coffee at 10-12% MC; must avoid moisture absorption",
    },
    {
        "commodity": "coffee",
        "storage_condition": "warehouse_controlled",
        "expected_loss_monthly_pct": 0.05,
        "max_loss_monthly_pct": 0.10,
        "max_storage_months": 24,
        "source": "ICO",
        "notes": "Climate-controlled warehouse with GrainPro storage",
    },
    {
        "commodity": "soya",
        "storage_condition": "silo_aerated",
        "expected_loss_monthly_pct": 0.1,
        "max_loss_monthly_pct": 0.3,
        "max_storage_months": 12,
        "source": "NOPA",
        "notes": "Aerated silo at <14% MC; insect and mould risk",
    },
    {
        "commodity": "soya",
        "storage_condition": "warehouse_ambient",
        "expected_loss_monthly_pct": 0.15,
        "max_loss_monthly_pct": 0.4,
        "max_storage_months": 6,
        "source": "NOPA",
        "notes": "Bagged storage; higher pest/moisture risk than silo",
    },
    {
        "commodity": "rubber",
        "storage_condition": "warehouse_ambient",
        "expected_loss_monthly_pct": 0.02,
        "max_loss_monthly_pct": 0.10,
        "max_storage_months": 24,
        "source": "IRSG",
        "notes": "Baled RSS/TSR; minimal degradation if dry",
    },
    {
        "commodity": "wood",
        "storage_condition": "yard_open",
        "expected_loss_monthly_pct": 0.5,
        "max_loss_monthly_pct": 1.5,
        "max_storage_months": 6,
        "source": "ITTO",
        "notes": "Open log yard; fungal stain, end-split, insect damage risk",
    },
    {
        "commodity": "wood",
        "storage_condition": "shed_covered",
        "expected_loss_monthly_pct": 0.1,
        "max_loss_monthly_pct": 0.3,
        "max_storage_months": 18,
        "source": "ITTO",
        "notes": "Covered shed for kiln-dried timber; minimal degradation",
    },
    {
        "commodity": "cattle",
        "storage_condition": "cold_chain",
        "expected_loss_monthly_pct": 0.3,
        "max_loss_monthly_pct": 0.8,
        "max_storage_months": 3,
        "source": "FAO",
        "notes": "Chilled carcass/primals; drip loss and trim oxidation",
    },
    {
        "commodity": "cattle",
        "storage_condition": "frozen",
        "expected_loss_monthly_pct": 0.1,
        "max_loss_monthly_pct": 0.3,
        "max_storage_months": 12,
        "source": "FAO",
        "notes": "Frozen meat/offal; sublimation loss over extended storage",
    },
]

# ---------------------------------------------------------------------------
# By-product recovery rates
# ---------------------------------------------------------------------------

BYPRODUCT_RECOVERY_RATES: List[Dict[str, Any]] = [
    {
        "commodity": "cocoa",
        "process": "shelling",
        "byproduct": "cocoa_shell",
        "recovery_rate": 0.12,
        "acceptable_range_min": 0.08,
        "acceptable_range_max": 0.18,
        "source": "ICCO",
        "notes": "Shell used for mulch, biofuel, or theobromine extraction",
    },
    {
        "commodity": "oil_palm",
        "process": "milling",
        "byproduct": "empty_fruit_bunches",
        "recovery_rate": 0.23,
        "acceptable_range_min": 0.20,
        "acceptable_range_max": 0.28,
        "source": "MPOB",
        "notes": "EFB used for mulching, composting, or biomass fuel",
    },
    {
        "commodity": "oil_palm",
        "process": "milling",
        "byproduct": "palm_fibre",
        "recovery_rate": 0.14,
        "acceptable_range_min": 0.11,
        "acceptable_range_max": 0.17,
        "source": "MPOB",
        "notes": "Mesocarp fibre used as boiler fuel",
    },
    {
        "commodity": "oil_palm",
        "process": "milling",
        "byproduct": "palm_shell",
        "recovery_rate": 0.06,
        "acceptable_range_min": 0.04,
        "acceptable_range_max": 0.08,
        "source": "MPOB",
        "notes": "Endocarp shell used as biomass fuel or activated carbon",
    },
    {
        "commodity": "soya",
        "process": "oil_extraction",
        "byproduct": "soya_hulls",
        "recovery_rate": 0.08,
        "acceptable_range_min": 0.06,
        "acceptable_range_max": 0.10,
        "source": "NOPA",
        "notes": "Hulls used as animal feed fibre supplement",
    },
    {
        "commodity": "wood",
        "process": "sawmilling",
        "byproduct": "sawdust",
        "recovery_rate": 0.12,
        "acceptable_range_min": 0.08,
        "acceptable_range_max": 0.18,
        "source": "ITTO",
        "notes": "Sawdust used for particle board, pellets, or bedding",
    },
    {
        "commodity": "wood",
        "process": "sawmilling",
        "byproduct": "wood_chips_slabs",
        "recovery_rate": 0.30,
        "acceptable_range_min": 0.22,
        "acceptable_range_max": 0.38,
        "source": "ITTO",
        "notes": "Slab wood and edgings chipped for pulp or bioenergy",
    },
    {
        "commodity": "cattle",
        "process": "slaughter",
        "byproduct": "blood",
        "recovery_rate": 0.035,
        "acceptable_range_min": 0.025,
        "acceptable_range_max": 0.045,
        "source": "FAO",
        "notes": "Blood collected for blood meal, albumin, or pharmaceutical use",
    },
    {
        "commodity": "cattle",
        "process": "slaughter",
        "byproduct": "bone_meal",
        "recovery_rate": 0.10,
        "acceptable_range_min": 0.07,
        "acceptable_range_max": 0.14,
        "source": "FAO",
        "notes": "Rendered bone for animal feed or fertilizer (non-BSE markets)",
    },
    {
        "commodity": "rubber",
        "process": "concentrate_centrifuge",
        "byproduct": "skim_latex",
        "recovery_rate": 0.55,
        "acceptable_range_min": 0.48,
        "acceptable_range_max": 0.62,
        "source": "IRSG",
        "notes": "Low-DRC skim from centrifuging; re-coagulated for TSR",
    },
    {
        "commodity": "coffee",
        "process": "wet_processing",
        "byproduct": "coffee_pulp",
        "recovery_rate": 0.42,
        "acceptable_range_min": 0.35,
        "acceptable_range_max": 0.50,
        "source": "ICO",
        "notes": "Pulp used for composting, cascara tea, or biogas",
    },
]

# ---------------------------------------------------------------------------
# Total entry counts
# ---------------------------------------------------------------------------

TOTAL_PROCESSING_TOLERANCES: int = len(PROCESSING_LOSS_TOLERANCES)
TOTAL_TRANSPORT_TOLERANCES: int = len(TRANSPORT_LOSS_TOLERANCES)
TOTAL_STORAGE_TOLERANCES: int = len(STORAGE_LOSS_TOLERANCES)
TOTAL_BYPRODUCT_RATES: int = len(BYPRODUCT_RECOVERY_RATES)

# ---------------------------------------------------------------------------
# Internal lookup index: {commodity: {process: tolerance_dict}}
# ---------------------------------------------------------------------------

_PROCESSING_INDEX: Dict[str, Dict[str, Dict[str, Any]]] = {}

for _tol in PROCESSING_LOSS_TOLERANCES:
    _commodity = _tol["commodity"]
    _process = _tol["process"]
    if _commodity not in _PROCESSING_INDEX:
        _PROCESSING_INDEX[_commodity] = {}
    _PROCESSING_INDEX[_commodity][_process] = _tol

_TRANSPORT_INDEX: Dict[str, Dict[str, Dict[str, Any]]] = {}

for _tol in TRANSPORT_LOSS_TOLERANCES:
    _commodity = _tol["commodity"]
    _mode = _tol["transport_mode"]
    if _commodity not in _TRANSPORT_INDEX:
        _TRANSPORT_INDEX[_commodity] = {}
    _TRANSPORT_INDEX[_commodity][_mode] = _tol

_STORAGE_INDEX: Dict[str, Dict[str, Dict[str, Any]]] = {}

for _tol in STORAGE_LOSS_TOLERANCES:
    _commodity = _tol["commodity"]
    _condition = _tol["storage_condition"]
    if _commodity not in _STORAGE_INDEX:
        _STORAGE_INDEX[_commodity] = {}
    _STORAGE_INDEX[_commodity][_condition] = _tol

# ---------------------------------------------------------------------------
# Lookup helper functions
# ---------------------------------------------------------------------------


def get_loss_tolerance(
    commodity: str,
    process: str,
) -> Optional[Dict[str, Any]]:
    """Return the processing loss tolerance for a commodity-process pair.

    Args:
        commodity: EUDR commodity (cocoa, oil_palm, coffee, soya,
            rubber, wood, cattle).
        process: Processing step identifier (e.g. "shelling",
            "milling", "wet_processing").

    Returns:
        Dictionary with expected_loss_pct, max_loss_pct, source,
        and notes.  Returns None if the combination is not found.

    Example:
        >>> tol = get_loss_tolerance("cocoa", "shelling")
        >>> tol["expected_loss_pct"]
        13.0
    """
    commodity_tols = _PROCESSING_INDEX.get(commodity)
    if commodity_tols is None:
        return None
    return commodity_tols.get(process)


def is_within_tolerance(
    commodity: str,
    process: str,
    actual_loss_pct: float,
) -> bool:
    """Check if an actual loss percentage is within the maximum tolerance.

    This is the primary validation used by the LossWasteTracker engine.
    A loss exceeding the maximum triggers an investigation flag.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.
        actual_loss_pct: The actual loss percentage observed.

    Returns:
        True if ``actual_loss_pct <= max_loss_pct``.
        Returns False if the commodity-process combination is unknown
        (fail-safe: unknown loss treated as out-of-tolerance).

    Example:
        >>> is_within_tolerance("cocoa", "shelling", 15.0)
        True
        >>> is_within_tolerance("cocoa", "shelling", 25.0)
        False
    """
    tol = get_loss_tolerance(commodity, process)
    if tol is None:
        return False
    return actual_loss_pct <= float(tol["max_loss_pct"])


def get_all_tolerances(commodity: str) -> List[Dict[str, Any]]:
    """Return all processing loss tolerances for a given commodity.

    Args:
        commodity: EUDR commodity identifier.

    Returns:
        List of tolerance dictionaries for the commodity.
        Returns empty list if commodity is not found.

    Example:
        >>> tols = get_all_tolerances("oil_palm")
        >>> len(tols) >= 3
        True
    """
    return [
        t for t in PROCESSING_LOSS_TOLERANCES
        if t["commodity"] == commodity
    ]


def get_expected_loss(commodity: str, process: str) -> float:
    """Return the expected (typical) loss percentage for a commodity-process.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.

    Returns:
        Expected loss percentage as float.
        Returns 0.0 if the combination is not found.

    Example:
        >>> get_expected_loss("coffee", "wet_processing")
        81.5
    """
    tol = get_loss_tolerance(commodity, process)
    if tol is None:
        return 0.0
    return float(tol["expected_loss_pct"])


def get_max_loss(commodity: str, process: str) -> float:
    """Return the maximum allowable loss percentage for a commodity-process.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.

    Returns:
        Maximum loss percentage as float.
        Returns 0.0 if the combination is not found.

    Example:
        >>> get_max_loss("wood", "sawmilling")
        60.0
    """
    tol = get_loss_tolerance(commodity, process)
    if tol is None:
        return 0.0
    return float(tol["max_loss_pct"])


def get_transport_tolerance(
    commodity: str,
    transport_mode: str,
) -> Optional[Dict[str, Any]]:
    """Return the transport loss tolerance for a commodity-mode pair.

    Args:
        commodity: EUDR commodity identifier.
        transport_mode: Transport mode (road_truck, sea_container,
            sea_tanker, sea_bulk, rail).

    Returns:
        Dictionary with expected_loss_pct, max_loss_pct, and
        distance_factor_per_1000km.  Returns None if not found.

    Example:
        >>> tol = get_transport_tolerance("cocoa", "road_truck")
        >>> tol["max_loss_pct"]
        0.5
    """
    commodity_tols = _TRANSPORT_INDEX.get(commodity)
    if commodity_tols is None:
        return None
    return commodity_tols.get(transport_mode)


def get_storage_tolerance(
    commodity: str,
    storage_condition: str,
) -> Optional[Dict[str, Any]]:
    """Return the storage loss tolerance for a commodity-condition pair.

    Args:
        commodity: EUDR commodity identifier.
        storage_condition: Storage condition (warehouse_ambient,
            warehouse_controlled, tank_heated, silo_aerated,
            yard_open, cold_chain, frozen, etc.).

    Returns:
        Dictionary with expected_loss_monthly_pct, max_loss_monthly_pct,
        and max_storage_months.  Returns None if not found.

    Example:
        >>> tol = get_storage_tolerance("rubber", "warehouse_ambient")
        >>> tol["max_storage_months"]
        24
    """
    commodity_tols = _STORAGE_INDEX.get(commodity)
    if commodity_tols is None:
        return None
    return commodity_tols.get(storage_condition)


def get_byproduct_recovery(
    commodity: str,
    process: str,
    byproduct: str,
) -> Optional[Dict[str, Any]]:
    """Return the by-product recovery rate for a specific by-product.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.
        byproduct: By-product material name.

    Returns:
        Dictionary with recovery_rate and acceptable range.
        Returns None if not found.

    Example:
        >>> bp = get_byproduct_recovery("oil_palm", "milling", "empty_fruit_bunches")
        >>> bp["recovery_rate"]
        0.23
    """
    for entry in BYPRODUCT_RECOVERY_RATES:
        if (
            entry["commodity"] == commodity
            and entry["process"] == process
            and entry["byproduct"] == byproduct
        ):
            return entry
    return None


def get_all_byproducts(
    commodity: str,
    process: str,
) -> List[Dict[str, Any]]:
    """Return all by-product recovery rates for a commodity-process pair.

    Args:
        commodity: EUDR commodity identifier.
        process: Processing step identifier.

    Returns:
        List of by-product recovery dictionaries.

    Example:
        >>> bps = get_all_byproducts("oil_palm", "milling")
        >>> len(bps)
        3
    """
    return [
        entry for entry in BYPRODUCT_RECOVERY_RATES
        if entry["commodity"] == commodity and entry["process"] == process
    ]


# ---------------------------------------------------------------------------
# Module-level logging on import
# ---------------------------------------------------------------------------

logger.debug(
    "Loss tolerances reference data loaded: "
    "processing=%d, transport=%d, storage=%d, byproduct=%d",
    TOTAL_PROCESSING_TOLERANCES,
    TOTAL_TRANSPORT_TOLERANCES,
    TOTAL_STORAGE_TOLERANCES,
    TOTAL_BYPRODUCT_RATES,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "PROCESSING_LOSS_TOLERANCES",
    "TRANSPORT_LOSS_TOLERANCES",
    "STORAGE_LOSS_TOLERANCES",
    "BYPRODUCT_RECOVERY_RATES",
    # Counts
    "TOTAL_PROCESSING_TOLERANCES",
    "TOTAL_TRANSPORT_TOLERANCES",
    "TOTAL_STORAGE_TOLERANCES",
    "TOTAL_BYPRODUCT_RATES",
    # Lookup helpers - processing
    "get_loss_tolerance",
    "is_within_tolerance",
    "get_all_tolerances",
    "get_expected_loss",
    "get_max_loss",
    # Lookup helpers - transport
    "get_transport_tolerance",
    # Lookup helpers - storage
    "get_storage_tolerance",
    # Lookup helpers - by-product
    "get_byproduct_recovery",
    "get_all_byproducts",
]
