# -*- coding: utf-8 -*-
"""Material taxonomy — for product carbon + Scope 3 Cat 1."""
from __future__ import annotations

from greenlang.factors.mapping.base import (
    BaseMapping,
    MappingConfidence,
    MappingResult,
)

MATERIAL_TAXONOMY = {
    "steel_hot_rolled_coil": {
        "synonyms": ["hot rolled coil", "hrc", "hot-rolled steel coil", "hr coil"],
        "meta": {"family": "ferrous", "cbam": True, "cn_code_prefix": "7208"},
    },
    "steel_cold_rolled_coil": {
        "synonyms": ["cold rolled coil", "crc", "cold-rolled steel"],
        "meta": {"family": "ferrous", "cbam": True, "cn_code_prefix": "7209"},
    },
    "steel_rebar": {
        "synonyms": ["rebar", "reinforcing bar", "reinforcement steel", "tmt bar"],
        "meta": {"family": "ferrous", "cbam": True, "cn_code_prefix": "7214"},
    },
    "aluminium_ingot_primary": {
        "synonyms": ["primary aluminium", "aluminum ingot", "aluminium ingot", "primary al"],
        "meta": {"family": "non_ferrous", "cbam": True, "cn_code_prefix": "7601"},
    },
    "aluminium_ingot_secondary": {
        "synonyms": ["secondary aluminium", "recycled aluminium", "aluminium alloy secondary"],
        "meta": {"family": "non_ferrous", "cbam": True, "cn_code_prefix": "7601"},
    },
    "cement_portland": {
        "synonyms": ["portland cement", "opc", "ordinary portland cement"],
        "meta": {"family": "cement", "cbam": True, "cn_code_prefix": "2523"},
    },
    "cement_blended": {
        "synonyms": ["ppc", "ggbs cement", "fly ash cement", "slag cement", "blended cement"],
        "meta": {"family": "cement", "cbam": True, "cn_code_prefix": "2523"},
    },
    "clinker": {
        "synonyms": ["cement clinker", "clinker cement"],
        "meta": {"family": "cement", "cbam": True, "cn_code_prefix": "2523"},
    },
    "fertilizer_urea": {
        "synonyms": ["urea", "urea fertilizer"],
        "meta": {"family": "fertilizer", "cbam": True, "cn_code_prefix": "3102"},
    },
    "fertilizer_nitrate": {
        "synonyms": ["nitrate fertilizer", "calcium ammonium nitrate", "can", "ammonium nitrate"],
        "meta": {"family": "fertilizer", "cbam": True, "cn_code_prefix": "3102"},
    },
    "concrete_ready_mix": {
        "synonyms": ["ready mix concrete", "rmc", "cast-in-place concrete"],
        "meta": {"family": "concrete"},
    },
    "glass_float": {
        "synonyms": ["float glass", "flat glass", "architectural glass"],
        "meta": {"family": "glass"},
    },
    "glass_container": {
        "synonyms": ["container glass", "bottle glass", "jar glass"],
        "meta": {"family": "glass"},
    },
    "plastic_pet": {
        "synonyms": ["pet", "polyethylene terephthalate"],
        "meta": {"family": "plastic"},
    },
    "plastic_hdpe": {
        "synonyms": ["hdpe", "high-density polyethylene"],
        "meta": {"family": "plastic"},
    },
    "plastic_ldpe": {
        "synonyms": ["ldpe", "low-density polyethylene"],
        "meta": {"family": "plastic"},
    },
    "plastic_pp": {
        "synonyms": ["polypropylene", "pp plastic"],
        "meta": {"family": "plastic"},
    },
    "paper_kraft": {
        "synonyms": ["kraft paper", "kraft liner"],
        "meta": {"family": "paper"},
    },
    "paper_recycled": {
        "synonyms": ["recycled paper", "rcp", "recovered paper"],
        "meta": {"family": "paper"},
    },
    "cotton_fibre": {
        "synonyms": ["cotton", "cotton fiber", "raw cotton"],
        "meta": {"family": "textile"},
    },
    "polyester_fibre": {
        "synonyms": ["polyester", "pet fibre", "polyester yarn"],
        "meta": {"family": "textile"},
    },
    "copper_cathode": {
        "synonyms": ["copper", "cathode copper", "refined copper"],
        "meta": {"family": "non_ferrous"},
    },
    "chemical_ammonia": {
        "synonyms": ["nh3", "ammonia", "anhydrous ammonia"],
        "meta": {"family": "chemical"},
    },
    "chemical_methanol": {
        "synonyms": ["methanol", "methyl alcohol", "ch3oh"],
        "meta": {"family": "chemical"},
    },
}


class MaterialMapping(BaseMapping):
    TAXONOMY = MATERIAL_TAXONOMY


def map_material(description: str) -> MappingResult:
    result = MaterialMapping._lookup(description)
    if result is None:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=f"No material match for '{description}'",
            raw_input=description,
        )
    meta = MATERIAL_TAXONOMY[result.canonical]["meta"]
    result.rationale = (
        f"{result.rationale}; family={meta.get('family')}; "
        f"cbam_covered={meta.get('cbam', False)}; "
        f"cn_code_prefix={meta.get('cn_code_prefix')}"
    )
    return result


__all__ = ["MATERIAL_TAXONOMY", "MaterialMapping", "map_material"]
