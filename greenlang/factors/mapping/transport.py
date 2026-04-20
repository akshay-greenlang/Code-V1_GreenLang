# -*- coding: utf-8 -*-
"""Transport taxonomy — mode × vehicle class × payload basis.

Aligned with GLEC Framework v3 mode codes + ISO 14083 boundary terms.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from greenlang.factors.mapping.base import (
    BaseMapping,
    MappingConfidence,
    MappingResult,
    normalize_text,
)

TRANSPORT_MODES = {
    "road": {
        "synonyms": ["truck", "road freight", "over-the-road", "otr", "lorry", "hgv"],
        "meta": {"boundary_preference": "WTW"},
    },
    "rail": {
        "synonyms": ["train", "rail freight", "intermodal rail", "railway"],
        "meta": {"boundary_preference": "WTW"},
    },
    "sea": {
        "synonyms": ["maritime", "ocean freight", "sea freight", "container shipping", "deep sea"],
        "meta": {"boundary_preference": "WTW"},
    },
    "inland_waterway": {
        "synonyms": ["barge", "inland navigation", "river freight"],
        "meta": {"boundary_preference": "WTW"},
    },
    "air": {
        "synonyms": ["air freight", "aviation", "cargo flight", "belly cargo"],
        "meta": {"boundary_preference": "WTW"},
    },
    "pipeline": {
        "synonyms": ["gas pipeline", "oil pipeline", "pipeline transport"],
        "meta": {"boundary_preference": "WTW"},
    },
    "passenger_rail": {
        "synonyms": ["commuter rail", "metro", "subway", "light rail", "high speed rail"],
        "meta": {"boundary_preference": "WTW"},
    },
    "passenger_road": {
        "synonyms": ["car", "bus", "coach", "taxi", "ride hail", "rideshare"],
        "meta": {"boundary_preference": "WTW"},
    },
    "passenger_air": {
        "synonyms": ["short haul", "long haul", "domestic flight", "international flight"],
        "meta": {"boundary_preference": "WTW"},
    },
}


VEHICLE_CLASSES = {
    # --- Road freight ---
    "heavy_truck_40t": {
        "synonyms": [
            "40t truck", "40-tonne truck", "40 tonne truck", "40 ton truck",
            "articulated truck", "semi-trailer", "semitrailer", "hgv 40t",
            "long haul truck", "long haul", "heavy truck", "heavy duty truck",
            "class 8 truck", "hgv",
        ],
        "meta": {"mode": "road", "max_payload_t": 40.0, "fuel_default": "diesel"},
    },
    "medium_truck_7_5_18t": {
        "synonyms": ["7.5 to 18 tonne truck", "medium truck", "rigid truck"],
        "meta": {"mode": "road", "max_payload_t": 18.0, "fuel_default": "diesel"},
    },
    "light_commercial": {
        "synonyms": ["lcv", "van", "light commercial vehicle", "delivery van"],
        "meta": {"mode": "road", "max_payload_t": 3.5, "fuel_default": "diesel"},
    },
    # --- Rail ---
    "diesel_freight_rail": {
        "synonyms": ["diesel rail", "diesel train freight"],
        "meta": {"mode": "rail", "fuel_default": "diesel"},
    },
    "electric_freight_rail": {
        "synonyms": ["electric rail", "electrified train"],
        "meta": {"mode": "rail", "fuel_default": "electricity"},
    },
    # --- Sea ---
    "container_ship": {
        "synonyms": ["container vessel", "container feeder", "panamax", "post-panamax"],
        "meta": {"mode": "sea", "fuel_default": "fuel_oil"},
    },
    "bulk_carrier": {
        "synonyms": ["dry bulk ship", "ore carrier"],
        "meta": {"mode": "sea", "fuel_default": "fuel_oil"},
    },
    "ro_ro": {
        "synonyms": ["ro-ro", "roll-on roll-off"],
        "meta": {"mode": "sea", "fuel_default": "fuel_oil"},
    },
    # --- Air ---
    "narrow_body": {
        "synonyms": ["narrow body", "single aisle", "737", "a320"],
        "meta": {"mode": "air", "fuel_default": "jet_fuel"},
    },
    "wide_body": {
        "synonyms": ["wide body", "twin aisle", "777", "787", "a350"],
        "meta": {"mode": "air", "fuel_default": "jet_fuel"},
    },
    "freighter": {
        "synonyms": ["air freighter", "cargo aircraft"],
        "meta": {"mode": "air", "fuel_default": "jet_fuel"},
    },
}


class TransportModeMapping(BaseMapping):
    TAXONOMY = TRANSPORT_MODES


class VehicleClassMapping(BaseMapping):
    TAXONOMY = VEHICLE_CLASSES


def map_transport(
    description: str,
    *,
    payload_tonnes: Optional[float] = None,
    distance_km: Optional[float] = None,
) -> MappingResult:
    """Resolve a transport description into (mode, vehicle_class)."""
    mode_result = TransportModeMapping._lookup(description)
    vehicle_result = VehicleClassMapping._lookup(description)

    # Prefer vehicle-class match because it also pins the mode.
    if vehicle_result is not None and vehicle_result.confidence >= 0.6:
        vc = vehicle_result.canonical
        meta = VEHICLE_CLASSES[vc]["meta"]
        mode = meta["mode"]
        canonical: Dict[str, Any] = {
            "mode": mode,
            "vehicle_class": vc,
            "fuel_default": meta.get("fuel_default"),
            "max_payload_t": meta.get("max_payload_t"),
            "payload_tonnes": payload_tonnes,
            "distance_km": distance_km,
        }
        rationale = (
            f"Matched vehicle class '{vc}' via {vehicle_result.rationale}; "
            f"implied mode={mode}."
        )
        return MappingResult(
            canonical=canonical,
            confidence=vehicle_result.confidence,
            band=vehicle_result.band,
            rationale=rationale,
            matched_pattern=vehicle_result.matched_pattern,
            raw_input=description,
        )

    if mode_result is not None:
        canonical = {
            "mode": mode_result.canonical,
            "vehicle_class": None,
            "payload_tonnes": payload_tonnes,
            "distance_km": distance_km,
        }
        return MappingResult(
            canonical=canonical,
            confidence=mode_result.confidence,
            band=mode_result.band,
            rationale=f"Matched mode '{mode_result.canonical}' (no specific vehicle class)",
            matched_pattern=mode_result.matched_pattern,
            raw_input=description,
        )

    return MappingResult(
        canonical={"mode": None, "vehicle_class": None},
        confidence=0.0,
        band=MappingConfidence.UNKNOWN,
        rationale=f"No transport match for '{description}'",
        raw_input=description,
    )


__all__ = [
    "TRANSPORT_MODES",
    "VEHICLE_CLASSES",
    "TransportModeMapping",
    "VehicleClassMapping",
    "map_transport",
]
