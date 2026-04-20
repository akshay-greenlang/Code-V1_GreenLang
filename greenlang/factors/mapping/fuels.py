# -*- coding: utf-8 -*-
"""Fuel taxonomy — canonical fuel types + synonyms.

Canonical keys match the ``fuel_type`` column on existing factor YAMLs.
Synonyms include common industry names, regional spellings, fuel-grade
designators, and abbreviations the matcher's gold set contains.
"""
from __future__ import annotations

from greenlang.factors.mapping.base import (
    BaseMapping,
    MappingConfidence,
    MappingResult,
    normalize_text,
)

# Canonical fuel → synonyms + metadata.
FUEL_TAXONOMY = {
    "diesel": {
        "synonyms": [
            "diesel fuel", "distillate", "distillate fuel oil", "no 2 diesel",
            "no. 2 distillate", "gas oil", "adb", "diesel oil", "automotive diesel",
            "asb", "euro-diesel", "bs-vi diesel", "biodiesel b5", "hsd",
            "high speed diesel", "fuel oil no 2",
        ],
        "meta": {
            "fuel_family": "liquid_fossil",
            "typical_unit": "liters",
            "biogenic_share": 0.0,
        },
    },
    "gasoline": {
        "synonyms": [
            "petrol", "motor gasoline", "motor spirit", "mogas", "unleaded",
            "unleaded gasoline", "regular unleaded", "premium unleaded",
            "rula", "rul", "motor vehicle gasoline",
        ],
        "meta": {"fuel_family": "liquid_fossil", "typical_unit": "liters"},
    },
    "natural_gas": {
        "synonyms": [
            "ng", "nat gas", "pipeline natural gas", "city gas", "piped gas",
            "lpg", "cng", "compressed natural gas",  # CNG variants intentional
            "lng", "liquefied natural gas", "fuel gas", "sweet gas",
        ],
        "meta": {"fuel_family": "gaseous_fossil", "typical_unit": "kWh"},
    },
    "coal": {
        "synonyms": [
            "bituminous coal", "sub-bituminous coal", "sub bituminous coal",
            "anthracite", "lignite", "brown coal", "hard coal", "steam coal",
            "thermal coal", "coking coal", "metallurgical coal", "met coal",
            "coke", "pulverized coal",
        ],
        "meta": {"fuel_family": "solid_fossil", "typical_unit": "tonnes"},
    },
    "propane": {
        "synonyms": ["lpg", "liquid petroleum gas", "liquefied petroleum gas", "hd-5", "hd5"],
        "meta": {"fuel_family": "liquid_fossil", "typical_unit": "liters"},
    },
    "butane": {
        "synonyms": ["n-butane", "iso-butane", "isobutane"],
        "meta": {"fuel_family": "liquid_fossil", "typical_unit": "liters"},
    },
    "kerosene": {
        "synonyms": ["kero", "paraffin", "jet-a1", "jet a1", "jet a", "aviation kerosene"],
        "meta": {"fuel_family": "liquid_fossil", "typical_unit": "liters"},
    },
    "jet_fuel": {
        "synonyms": ["jet a", "jet a-1", "jet a1", "avtur", "atf", "aviation turbine fuel"],
        "meta": {"fuel_family": "liquid_fossil", "typical_unit": "liters"},
    },
    "avgas": {
        "synonyms": ["aviation gasoline", "100ll", "100 ll", "avgas 100"],
        "meta": {"fuel_family": "liquid_fossil", "typical_unit": "liters"},
    },
    "fuel_oil": {
        "synonyms": [
            "heavy fuel oil", "hfo", "residual fuel oil", "bunker c", "no 6 fuel oil",
            "no. 6 fuel oil", "marine fuel oil", "mfo",
        ],
        "meta": {"fuel_family": "liquid_fossil", "typical_unit": "liters"},
    },
    "biodiesel": {
        "synonyms": ["b100", "biodiesel b100", "fame", "fatty acid methyl ester"],
        "meta": {"fuel_family": "biofuel", "typical_unit": "liters", "biogenic_share": 1.0},
    },
    "ethanol": {
        "synonyms": ["bioethanol", "e100", "etoh", "grain ethanol", "cellulosic ethanol"],
        "meta": {"fuel_family": "biofuel", "typical_unit": "liters", "biogenic_share": 1.0},
    },
    "hydrogen": {
        "synonyms": ["h2", "green hydrogen", "grey hydrogen", "blue hydrogen", "pink hydrogen"],
        "meta": {"fuel_family": "gaseous_alt", "typical_unit": "kg"},
    },
    "electricity": {
        "synonyms": [
            "grid electricity", "purchased electricity", "mains electricity",
            "electrical energy", "power", "elec", "electricity consumption",
        ],
        "meta": {"fuel_family": "electricity", "typical_unit": "kWh"},
    },
    "steam": {
        "synonyms": ["district steam", "process steam", "purchased steam"],
        "meta": {"fuel_family": "thermal_energy", "typical_unit": "GJ"},
    },
    "biomass": {
        "synonyms": ["wood", "wood pellets", "wood chips", "agricultural residue", "bagasse", "rice husk"],
        "meta": {"fuel_family": "biomass", "typical_unit": "tonnes", "biogenic_share": 1.0},
    },
    "msw": {
        "synonyms": ["municipal solid waste", "rdf", "refuse-derived fuel"],
        "meta": {"fuel_family": "waste_fuel", "typical_unit": "tonnes"},
    },
}


class FuelMapping(BaseMapping):
    TAXONOMY = FUEL_TAXONOMY


def map_fuel(description: str) -> MappingResult:
    """Map free-text to a canonical fuel type."""
    result = FuelMapping._lookup(description)
    if result is None:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=f"No fuel match for '{description}'",
            raw_input=description,
        )
    # Enrich alternates with meta from the taxonomy.
    canonical = result.canonical
    meta = FUEL_TAXONOMY.get(canonical, {}).get("meta", {})
    result.alternates = [
        {"key": other, "fuel_family": FUEL_TAXONOMY[other]["meta"].get("fuel_family")}
        for other in FUEL_TAXONOMY
        if other != canonical
    ][:5]
    result.rationale = (
        f"{result.rationale}; fuel_family={meta.get('fuel_family')}; "
        f"typical_unit={meta.get('typical_unit')}"
    )
    return result


__all__ = ["FUEL_TAXONOMY", "FuelMapping", "map_fuel"]
