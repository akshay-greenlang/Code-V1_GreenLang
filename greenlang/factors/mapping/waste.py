# -*- coding: utf-8 -*-
"""Waste route taxonomy — Scope 3 Cat 5 + Cat 12."""
from __future__ import annotations

from greenlang.factors.mapping.base import (
    BaseMapping,
    MappingConfidence,
    MappingResult,
)

WASTE_ROUTES = {
    "landfill": {
        "synonyms": ["landfilled", "sanitary landfill", "msw landfill", "dumped"],
        "meta": {"family": "disposal", "generates_ch4": True},
    },
    "open_burning": {
        "synonyms": ["open burn", "open-air burning"],
        "meta": {"family": "disposal"},
    },
    "incineration_no_recovery": {
        "synonyms": ["incinerated", "combusted waste", "mass burn"],
        "meta": {"family": "treatment", "energy_recovery": False},
    },
    "incineration_energy_recovery": {
        "synonyms": ["waste to energy", "wte", "energy from waste", "efw"],
        "meta": {"family": "treatment", "energy_recovery": True},
    },
    "composting": {
        "synonyms": ["composted", "compost facility", "organic composting"],
        "meta": {"family": "biological", "generates_ch4": False},
    },
    "anaerobic_digestion": {
        "synonyms": ["ad", "anaerobic digester", "biogas plant", "digested waste"],
        "meta": {"family": "biological", "energy_recovery": True, "generates_ch4": True},
    },
    "recycling": {
        "synonyms": ["recycled", "material recycling", "recovery"],
        "meta": {"family": "recycling"},
    },
    "reuse": {
        "synonyms": ["reused", "repurposed", "repaired"],
        "meta": {"family": "circular"},
    },
    "wastewater_treatment_aerobic": {
        "synonyms": ["sewage treatment plant", "stp", "aerobic wastewater", "activated sludge"],
        "meta": {"family": "wastewater"},
    },
    "wastewater_treatment_anaerobic": {
        "synonyms": ["anaerobic wastewater", "lagoon"],
        "meta": {"family": "wastewater", "generates_ch4": True},
    },
}


class WasteMapping(BaseMapping):
    TAXONOMY = WASTE_ROUTES


def map_waste(description: str) -> MappingResult:
    result = WasteMapping._lookup(description)
    if result is None:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=f"No waste route match for '{description}'",
            raw_input=description,
        )
    meta = WASTE_ROUTES[result.canonical]["meta"]
    result.rationale = (
        f"{result.rationale}; family={meta.get('family')}; "
        f"generates_ch4={meta.get('generates_ch4', False)}"
    )
    return result


__all__ = ["WASTE_ROUTES", "WasteMapping", "map_waste"]
