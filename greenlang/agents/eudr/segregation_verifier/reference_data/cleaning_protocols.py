# -*- coding: utf-8 -*-
"""
Cleaning & Changeover Protocols Reference Data - AGENT-EUDR-010

Cleaning and changeover requirements for transport vehicles and
processing lines when switching between non-compliant and EUDR-compliant
material.  Deterministic lookup tables used by the TransportSegregationTracker
(Engine 3) and ProcessingLineVerifier (Engine 4) for zero-hallucination
verification.

Datasets:
    CLEANING_PROTOCOLS:
        Per-transport-type cleaning requirements including methods,
        duration, verification approach, and certificate requirements.
        10 transport types covering road, sea, rail, barge, and air.

    PROCESSING_CHANGEOVER_PROTOCOLS:
        Per-processing-line-type changeover requirements including
        flush volumes, purge methods, duration, and first-run handling.
        15 processing line types covering all EUDR commodity transformations.

    VERIFICATION_METHODS:
        Details of cleaning verification methods (visual, swab test,
        lab analysis, rinse water test, ATP test) with descriptions,
        sensitivity levels, and result interpretation.

    CLEANING_AGENT_COMPATIBILITY:
        Compatibility matrix of cleaning agents vs commodity residue
        types, ensuring the correct cleaning agent is selected.

Lookup helpers:
    get_cleaning_protocol(transport_type) -> dict | None
    get_changeover_protocol(line_type) -> dict | None
    get_verification_method(method_name) -> dict | None
    is_cleaning_sufficient(transport_type, method, duration_minutes) -> bool
    is_changeover_sufficient(line_type, duration_minutes, flush_liters) -> bool
    get_all_transport_types() -> list[str]
    get_all_line_types() -> list[str]

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier (GL-EUDR-SGV-010)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transport cleaning protocols (10 transport types)
# ---------------------------------------------------------------------------

CLEANING_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "bulk_truck": {
        "name": "Bulk Truck (Tipper/Flatbed)",
        "methods": ["power_wash", "steam_clean"],
        "min_duration_minutes": 120,
        "verification": "visual_swab",
        "certificate_required": True,
        "drying_required": True,
        "drying_time_minutes": 60,
        "residue_tolerance_ppm": 50,
        "applicable_commodities": [
            "cocoa", "coffee", "soya", "wood", "rubber",
        ],
        "notes": "Full interior wash; chassis and tailgate seals inspected.",
    },
    "container_truck": {
        "name": "Container Truck (20ft/40ft)",
        "methods": ["power_wash", "sweep_wash"],
        "min_duration_minutes": 90,
        "verification": "visual_swab",
        "certificate_required": True,
        "drying_required": True,
        "drying_time_minutes": 45,
        "residue_tolerance_ppm": 100,
        "applicable_commodities": [
            "cocoa", "coffee", "soya", "wood", "rubber",
        ],
        "notes": "Interior walls, floor, and door seals cleaned.",
    },
    "tanker": {
        "name": "Road Tanker (Liquid)",
        "methods": ["cip_wash", "steam_clean", "chemical_wash"],
        "min_duration_minutes": 180,
        "verification": "rinse_water_test",
        "certificate_required": True,
        "drying_required": False,
        "drying_time_minutes": 0,
        "residue_tolerance_ppm": 10,
        "applicable_commodities": ["oil_palm"],
        "notes": "CIP (Clean-In-Place) mandatory; rinse water tested for oil residue.",
    },
    "dry_bulk_vessel": {
        "name": "Dry Bulk Vessel (Handymax/Panamax)",
        "methods": ["power_wash", "sweep_wash", "fumigation"],
        "min_duration_minutes": 480,
        "verification": "visual_swab",
        "certificate_required": True,
        "drying_required": True,
        "drying_time_minutes": 120,
        "residue_tolerance_ppm": 100,
        "applicable_commodities": [
            "cocoa", "coffee", "soya", "wood",
        ],
        "notes": "Hold sweeping, washing, and SGS/Bureau Veritas inspection.",
    },
    "container_vessel": {
        "name": "Container Vessel (ISO Container)",
        "methods": ["power_wash", "sweep_wash"],
        "min_duration_minutes": 90,
        "verification": "visual_swab",
        "certificate_required": True,
        "drying_required": True,
        "drying_time_minutes": 60,
        "residue_tolerance_ppm": 100,
        "applicable_commodities": [
            "cocoa", "coffee", "soya", "wood", "rubber",
        ],
        "notes": "Container interior and door gaskets inspected.",
    },
    "tanker_vessel": {
        "name": "Tanker Vessel (Chemical/Product)",
        "methods": ["cip_wash", "butterworth_wash", "chemical_wash"],
        "min_duration_minutes": 360,
        "verification": "lab_analysis",
        "certificate_required": True,
        "drying_required": False,
        "drying_time_minutes": 0,
        "residue_tolerance_ppm": 5,
        "applicable_commodities": ["oil_palm"],
        "notes": "Tank cleaning per MARPOL Annex II; wall wash test required.",
    },
    "rail_hopper": {
        "name": "Rail Hopper Car",
        "methods": ["power_wash", "sweep_wash"],
        "min_duration_minutes": 120,
        "verification": "visual_swab",
        "certificate_required": True,
        "drying_required": True,
        "drying_time_minutes": 60,
        "residue_tolerance_ppm": 100,
        "applicable_commodities": [
            "cocoa", "coffee", "soya", "wood",
        ],
        "notes": "Interior hopper and discharge valves cleaned.",
    },
    "rail_container": {
        "name": "Rail Container (Intermodal)",
        "methods": ["power_wash", "sweep_wash"],
        "min_duration_minutes": 90,
        "verification": "visual_swab",
        "certificate_required": True,
        "drying_required": True,
        "drying_time_minutes": 45,
        "residue_tolerance_ppm": 100,
        "applicable_commodities": [
            "cocoa", "coffee", "soya", "wood", "rubber",
        ],
        "notes": "Standard ISO container cleaning protocol.",
    },
    "barge": {
        "name": "Inland Barge",
        "methods": ["power_wash", "sweep_wash"],
        "min_duration_minutes": 240,
        "verification": "visual_swab",
        "certificate_required": True,
        "drying_required": True,
        "drying_time_minutes": 90,
        "residue_tolerance_ppm": 100,
        "applicable_commodities": [
            "cocoa", "coffee", "soya", "wood",
        ],
        "notes": "Hold cleaning and bilge pump verified.",
    },
    "air_freight": {
        "name": "Air Freight (ULD/Pallet)",
        "methods": ["wipe_clean", "sweep_wash"],
        "min_duration_minutes": 30,
        "verification": "visual",
        "certificate_required": False,
        "drying_required": False,
        "drying_time_minutes": 0,
        "residue_tolerance_ppm": 200,
        "applicable_commodities": [
            "cocoa", "coffee", "rubber",
        ],
        "notes": "ULD netting and floor liner replaced; wipe clean of frame.",
    },
}

TOTAL_TRANSPORT_TYPES: int = len(CLEANING_PROTOCOLS)

# ---------------------------------------------------------------------------
# Processing line changeover protocols (15 line types)
# ---------------------------------------------------------------------------

PROCESSING_CHANGEOVER_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "extraction": {
        "name": "Solvent Extraction Line",
        "flush_required": True,
        "min_flush_volume_liters": 500,
        "min_duration_minutes": 90,
        "purge_method": "product_flush",
        "first_run_discard": True,
        "first_run_discard_kg": 200,
        "cleaning_agent": "food_grade_solvent",
        "verification": "lab_analysis",
        "applicable_commodities": ["soya", "oil_palm"],
        "notes": "Solvent lines purged; hexane residue tested < 5 ppm.",
    },
    "pressing": {
        "name": "Mechanical Press Line",
        "flush_required": True,
        "min_flush_volume_liters": 300,
        "min_duration_minutes": 60,
        "purge_method": "product_flush",
        "first_run_discard": True,
        "first_run_discard_kg": 100,
        "cleaning_agent": "neutral_oil",
        "verification": "rinse_water_test",
        "applicable_commodities": ["oil_palm", "soya"],
        "notes": "Press cage and oil collection tray flushed.",
    },
    "milling": {
        "name": "Milling / Grinding Line",
        "flush_required": True,
        "min_flush_volume_liters": 100,
        "min_duration_minutes": 45,
        "purge_method": "air_purge",
        "first_run_discard": True,
        "first_run_discard_kg": 50,
        "cleaning_agent": "compressed_air",
        "verification": "visual_swab",
        "applicable_commodities": ["cocoa", "coffee", "soya", "wood"],
        "notes": "Mill chamber and conveyor belt vacuumed and air-purged.",
    },
    "refining": {
        "name": "Oil Refining Line",
        "flush_required": True,
        "min_flush_volume_liters": 1000,
        "min_duration_minutes": 120,
        "purge_method": "product_flush",
        "first_run_discard": True,
        "first_run_discard_kg": 500,
        "cleaning_agent": "refined_oil",
        "verification": "lab_analysis",
        "applicable_commodities": ["oil_palm", "soya"],
        "notes": "Full refining train (degumming/bleaching/deodorizing) flushed.",
    },
    "roasting": {
        "name": "Roasting Line",
        "flush_required": False,
        "min_flush_volume_liters": 0,
        "min_duration_minutes": 30,
        "purge_method": "air_purge",
        "first_run_discard": True,
        "first_run_discard_kg": 25,
        "cleaning_agent": "compressed_air",
        "verification": "visual",
        "applicable_commodities": ["cocoa", "coffee"],
        "notes": "Roasting drum and cooling tray air-purged; first batch sampled.",
    },
    "fermenting": {
        "name": "Fermentation Line",
        "flush_required": True,
        "min_flush_volume_liters": 200,
        "min_duration_minutes": 60,
        "purge_method": "water_flush",
        "first_run_discard": False,
        "first_run_discard_kg": 0,
        "cleaning_agent": "potable_water",
        "verification": "visual_swab",
        "applicable_commodities": ["cocoa", "coffee"],
        "notes": "Fermentation boxes and drains washed.",
    },
    "drying": {
        "name": "Drying Line (Mechanical/Solar)",
        "flush_required": False,
        "min_flush_volume_liters": 0,
        "min_duration_minutes": 30,
        "purge_method": "air_purge",
        "first_run_discard": False,
        "first_run_discard_kg": 0,
        "cleaning_agent": "compressed_air",
        "verification": "visual",
        "applicable_commodities": ["cocoa", "coffee", "rubber", "wood"],
        "notes": "Drying trays and conveyor surfaces swept and air-blown.",
    },
    "cutting": {
        "name": "Sawing / Cutting Line",
        "flush_required": False,
        "min_flush_volume_liters": 0,
        "min_duration_minutes": 20,
        "purge_method": "manual_clean",
        "first_run_discard": False,
        "first_run_discard_kg": 0,
        "cleaning_agent": "none",
        "verification": "visual",
        "applicable_commodities": ["wood"],
        "notes": "Blade change or blade cleaning; sawdust collection emptied.",
    },
    "tanning": {
        "name": "Tanning Line (Leather)",
        "flush_required": True,
        "min_flush_volume_liters": 500,
        "min_duration_minutes": 120,
        "purge_method": "water_flush",
        "first_run_discard": True,
        "first_run_discard_kg": 50,
        "cleaning_agent": "potable_water",
        "verification": "rinse_water_test",
        "applicable_commodities": ["cattle"],
        "notes": "Tanning drums and chemical feed lines flushed.",
    },
    "spinning": {
        "name": "Spinning / Extrusion Line",
        "flush_required": True,
        "min_flush_volume_liters": 100,
        "min_duration_minutes": 45,
        "purge_method": "product_flush",
        "first_run_discard": True,
        "first_run_discard_kg": 30,
        "cleaning_agent": "neutral_polymer",
        "verification": "visual_swab",
        "applicable_commodities": ["rubber"],
        "notes": "Extruder barrel and die head purged with neutral compound.",
    },
    "smelting": {
        "name": "Smelting / Rendering Line",
        "flush_required": True,
        "min_flush_volume_liters": 200,
        "min_duration_minutes": 90,
        "purge_method": "product_flush",
        "first_run_discard": True,
        "first_run_discard_kg": 100,
        "cleaning_agent": "neutral_fat",
        "verification": "lab_analysis",
        "applicable_commodities": ["cattle"],
        "notes": "Rendering vessel and fat collection piping flushed.",
    },
    "fractionation": {
        "name": "Fractionation Line",
        "flush_required": True,
        "min_flush_volume_liters": 800,
        "min_duration_minutes": 120,
        "purge_method": "product_flush",
        "first_run_discard": True,
        "first_run_discard_kg": 400,
        "cleaning_agent": "refined_oil",
        "verification": "lab_analysis",
        "applicable_commodities": ["oil_palm"],
        "notes": "Crystallizer, filter press, and olein/stearin piping flushed.",
    },
    "blending_line": {
        "name": "Blending / Mixing Line",
        "flush_required": True,
        "min_flush_volume_liters": 300,
        "min_duration_minutes": 60,
        "purge_method": "product_flush",
        "first_run_discard": True,
        "first_run_discard_kg": 150,
        "cleaning_agent": "neutral_oil",
        "verification": "visual_swab",
        "applicable_commodities": ["oil_palm", "soya", "cocoa"],
        "notes": "Mixing vessel, dosing pumps, and inline filters flushed.",
    },
    "packaging": {
        "name": "Packaging Line",
        "flush_required": False,
        "min_flush_volume_liters": 0,
        "min_duration_minutes": 20,
        "purge_method": "manual_clean",
        "first_run_discard": False,
        "first_run_discard_kg": 0,
        "cleaning_agent": "none",
        "verification": "visual",
        "applicable_commodities": [
            "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
        ],
        "notes": "Hopper emptied; label and packaging material changed.",
    },
    "grading": {
        "name": "Grading / Sorting Line",
        "flush_required": False,
        "min_flush_volume_liters": 0,
        "min_duration_minutes": 15,
        "purge_method": "manual_clean",
        "first_run_discard": False,
        "first_run_discard_kg": 0,
        "cleaning_agent": "none",
        "verification": "visual",
        "applicable_commodities": [
            "cocoa", "coffee", "rubber", "soya", "wood",
        ],
        "notes": "Sorting belt and collection bins emptied and swept.",
    },
}

TOTAL_LINE_TYPES: int = len(PROCESSING_CHANGEOVER_PROTOCOLS)

# ---------------------------------------------------------------------------
# Verification methods
# ---------------------------------------------------------------------------

VERIFICATION_METHODS: Dict[str, Dict[str, Any]] = {
    "visual": {
        "name": "Visual Inspection",
        "sensitivity_level": "low",
        "description": (
            "Trained inspector visually checks for residue, stains, "
            "odours, and foreign material."
        ),
        "detection_limit_ppm": None,
        "turnaround_minutes": 5,
        "certification_required": False,
        "suitable_for": ["dry_goods", "solid_materials"],
    },
    "visual_swab": {
        "name": "Visual + Swab Test",
        "sensitivity_level": "medium",
        "description": (
            "Visual inspection plus swab sampling of surfaces.  "
            "Swab tested for residue using rapid test kits."
        ),
        "detection_limit_ppm": 50,
        "turnaround_minutes": 15,
        "certification_required": False,
        "suitable_for": ["dry_goods", "solid_materials", "oily_surfaces"],
    },
    "rinse_water_test": {
        "name": "Rinse Water Analysis",
        "sensitivity_level": "medium",
        "description": (
            "Final rinse water collected and tested for residue "
            "concentration using field test kits or pH/turbidity meters."
        ),
        "detection_limit_ppm": 10,
        "turnaround_minutes": 30,
        "certification_required": False,
        "suitable_for": ["liquid_systems", "piping", "tanks"],
    },
    "lab_analysis": {
        "name": "Laboratory Analysis",
        "sensitivity_level": "high",
        "description": (
            "Samples sent to accredited laboratory for GC-MS, HPLC, "
            "or ICP-OES analysis.  Most sensitive and legally defensible."
        ),
        "detection_limit_ppm": 1,
        "turnaround_minutes": 1440,
        "certification_required": True,
        "suitable_for": [
            "liquid_systems", "oily_surfaces",
            "allergen_detection", "chemical_residue",
        ],
    },
    "atp_test": {
        "name": "ATP Bioluminescence Test",
        "sensitivity_level": "medium",
        "description": (
            "Adenosine triphosphate (ATP) surface hygiene test.  "
            "Rapid (10-second) detection of biological contamination."
        ),
        "detection_limit_ppm": None,
        "turnaround_minutes": 1,
        "certification_required": False,
        "suitable_for": [
            "food_contact_surfaces", "organic_residue",
        ],
    },
}

TOTAL_VERIFICATION_METHODS: int = len(VERIFICATION_METHODS)

# ---------------------------------------------------------------------------
# Cleaning agent compatibility
# ---------------------------------------------------------------------------

CLEANING_AGENT_COMPATIBILITY: Dict[str, Dict[str, Any]] = {
    "potable_water": {
        "name": "Potable Water",
        "effective_against": [
            "dust", "loose_particles", "sugar_residue",
            "water_soluble_compounds",
        ],
        "not_effective_against": ["oil", "fat", "wax", "resin"],
        "food_safe": True,
        "environmental_impact": "none",
        "cost_level": "low",
    },
    "food_grade_solvent": {
        "name": "Food-Grade Solvent (e.g. ethanol, isopropanol)",
        "effective_against": [
            "oil", "fat", "wax", "resin",
            "hexane_residue", "solvent_residue",
        ],
        "not_effective_against": ["protein", "mineral_deposits"],
        "food_safe": True,
        "environmental_impact": "low",
        "cost_level": "medium",
    },
    "neutral_oil": {
        "name": "Neutral Edible Oil (e.g. refined palm olein)",
        "effective_against": [
            "oil_residue", "fat_residue", "color_residue",
        ],
        "not_effective_against": [
            "protein", "mineral_deposits", "sugar_residue",
        ],
        "food_safe": True,
        "environmental_impact": "none",
        "cost_level": "medium",
    },
    "refined_oil": {
        "name": "Refined Oil (product-grade flush)",
        "effective_against": [
            "crude_oil_residue", "fatty_acid_residue",
            "color_bodies", "oxidized_material",
        ],
        "not_effective_against": [
            "protein", "mineral_deposits",
        ],
        "food_safe": True,
        "environmental_impact": "none",
        "cost_level": "high",
    },
    "compressed_air": {
        "name": "Compressed Air (dry purge)",
        "effective_against": [
            "dust", "loose_particles", "powder_residue",
            "coffee_chaff", "cocoa_shell",
        ],
        "not_effective_against": [
            "oil", "fat", "sticky_residue", "caked_material",
        ],
        "food_safe": True,
        "environmental_impact": "none",
        "cost_level": "low",
    },
    "neutral_polymer": {
        "name": "Neutral Polymer Compound (purge compound)",
        "effective_against": [
            "rubber_residue", "polymer_residue",
            "color_concentrate", "additive_residue",
        ],
        "not_effective_against": [
            "oil", "fat", "mineral_deposits",
        ],
        "food_safe": False,
        "environmental_impact": "low",
        "cost_level": "medium",
    },
    "neutral_fat": {
        "name": "Neutral Fat / Tallow",
        "effective_against": [
            "rendered_fat_residue", "protein_residue",
            "bone_meal_residue",
        ],
        "not_effective_against": [
            "mineral_deposits", "chemical_residue",
        ],
        "food_safe": True,
        "environmental_impact": "none",
        "cost_level": "medium",
    },
    "none": {
        "name": "No Cleaning Agent (manual sweep/wipe only)",
        "effective_against": [
            "dust", "loose_particles", "sawdust",
        ],
        "not_effective_against": [
            "oil", "fat", "sticky_residue",
            "chemical_residue", "protein",
        ],
        "food_safe": True,
        "environmental_impact": "none",
        "cost_level": "none",
    },
}

TOTAL_CLEANING_AGENTS: int = len(CLEANING_AGENT_COMPATIBILITY)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_cleaning_protocol(
    transport_type: str,
) -> Optional[Dict[str, Any]]:
    """Return cleaning protocol for a transport type.

    Args:
        transport_type: Transport type identifier (bulk_truck,
            container_truck, tanker, etc.).

    Returns:
        Dictionary with cleaning protocol data, or None if not found.
    """
    return CLEANING_PROTOCOLS.get(transport_type)


def get_changeover_protocol(
    line_type: str,
) -> Optional[Dict[str, Any]]:
    """Return changeover protocol for a processing line type.

    Args:
        line_type: Processing line type identifier (extraction,
            pressing, milling, etc.).

    Returns:
        Dictionary with changeover protocol data, or None if not found.
    """
    return PROCESSING_CHANGEOVER_PROTOCOLS.get(line_type)


def get_verification_method(
    method_name: str,
) -> Optional[Dict[str, Any]]:
    """Return verification method details.

    Args:
        method_name: Verification method identifier (visual,
            visual_swab, rinse_water_test, lab_analysis, atp_test).

    Returns:
        Dictionary with method details, or None if not found.
    """
    return VERIFICATION_METHODS.get(method_name)


def is_cleaning_sufficient(
    transport_type: str,
    method: str,
    duration_minutes: int,
) -> bool:
    """Check if a cleaning operation meets the minimum protocol.

    Validates that (a) the method is in the protocol's accepted
    method list and (b) the duration meets or exceeds the minimum.

    Args:
        transport_type: Transport type identifier.
        method: Cleaning method applied (power_wash, steam_clean,
            cip_wash, etc.).
        duration_minutes: Actual cleaning duration in minutes.

    Returns:
        True if cleaning meets minimum requirements.
    """
    protocol = CLEANING_PROTOCOLS.get(transport_type)
    if protocol is None:
        return False
    if method not in protocol["methods"]:
        return False
    return duration_minutes >= protocol["min_duration_minutes"]


def is_changeover_sufficient(
    line_type: str,
    duration_minutes: int,
    flush_liters: float = 0.0,
) -> bool:
    """Check if a changeover operation meets the minimum protocol.

    Validates duration and, if flush is required, volume.

    Args:
        line_type: Processing line type identifier.
        duration_minutes: Actual changeover duration in minutes.
        flush_liters: Actual flush volume in liters (ignored if
            flush is not required for this line type).

    Returns:
        True if changeover meets minimum requirements.
    """
    protocol = PROCESSING_CHANGEOVER_PROTOCOLS.get(line_type)
    if protocol is None:
        return False
    if duration_minutes < protocol["min_duration_minutes"]:
        return False
    if protocol["flush_required"]:
        if flush_liters < protocol["min_flush_volume_liters"]:
            return False
    return True


def get_all_transport_types() -> List[str]:
    """Return all supported transport types.

    Returns:
        Sorted list of transport type identifier strings.
    """
    return sorted(CLEANING_PROTOCOLS.keys())


def get_all_line_types() -> List[str]:
    """Return all supported processing line types.

    Returns:
        Sorted list of processing line type identifier strings.
    """
    return sorted(PROCESSING_CHANGEOVER_PROTOCOLS.keys())


def get_first_run_discard_kg(line_type: str) -> float:
    """Return the first-run discard quantity for a line type.

    Args:
        line_type: Processing line type identifier.

    Returns:
        Discard quantity in kilograms.  Returns 0.0 if line type
        is unknown or first-run discard is not required.
    """
    protocol = PROCESSING_CHANGEOVER_PROTOCOLS.get(line_type)
    if protocol is None:
        return 0.0
    if not protocol.get("first_run_discard", False):
        return 0.0
    return float(protocol.get("first_run_discard_kg", 0.0))


def get_residue_tolerance(transport_type: str) -> float:
    """Return the maximum residue tolerance in ppm for a transport type.

    Args:
        transport_type: Transport type identifier.

    Returns:
        Residue tolerance in parts per million.
        Returns 0.0 if transport type is unknown.
    """
    protocol = CLEANING_PROTOCOLS.get(transport_type)
    if protocol is None:
        return 0.0
    return float(protocol.get("residue_tolerance_ppm", 0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "CLEANING_PROTOCOLS",
    "PROCESSING_CHANGEOVER_PROTOCOLS",
    "VERIFICATION_METHODS",
    "CLEANING_AGENT_COMPATIBILITY",
    # Counts
    "TOTAL_TRANSPORT_TYPES",
    "TOTAL_LINE_TYPES",
    "TOTAL_VERIFICATION_METHODS",
    "TOTAL_CLEANING_AGENTS",
    # Lookup helpers
    "get_cleaning_protocol",
    "get_changeover_protocol",
    "get_verification_method",
    "is_cleaning_sufficient",
    "is_changeover_sufficient",
    "get_all_transport_types",
    "get_all_line_types",
    "get_first_run_discard_kg",
    "get_residue_tolerance",
]
