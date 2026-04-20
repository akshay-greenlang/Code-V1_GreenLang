# -*- coding: utf-8 -*-
"""Industry / trade classification cross-map.

Supports NAICS ↔ ISIC ↔ HS/CN ↔ GICS.  Only a small, high-signal slice
is hard-coded here — enough to route common CBAM + Scope 3 Cat 1 activities.
Full coverage lands in F9 / F10 via an ingested cross-walk table.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from greenlang.factors.mapping.base import MappingConfidence, MappingResult


#: Canonical cross-map rows.  Each row carries every code we know for the
#: same economic activity, plus a suggested factor family.
CROSS_WALK: List[Dict[str, Any]] = [
    {
        "label": "Electric power generation",
        "naics": "221112",
        "isic": "D3510",
        "gics": "55101010",  # Electric Utilities
        "hs_cn_prefix": None,
        "factor_family": "grid_intensity",
    },
    {
        "label": "Iron and steel basic manufacturing",
        "naics": "331110",
        "isic": "C2410",
        "gics": "15104020",  # Steel
        "hs_cn_prefix": "72",
        "factor_family": "material_embodied",
    },
    {
        "label": "Aluminium production",
        "naics": "331313",
        "isic": "C2420",
        "gics": "15104030",  # Aluminum
        "hs_cn_prefix": "7601",
        "factor_family": "material_embodied",
    },
    {
        "label": "Cement manufacturing",
        "naics": "327310",
        "isic": "C2394",
        "gics": "15102020",  # Construction Materials
        "hs_cn_prefix": "2523",
        "factor_family": "material_embodied",
    },
    {
        "label": "Fertilizer manufacturing",
        "naics": "325311",
        "isic": "C2012",
        "gics": "15101030",  # Fertilizers & Agricultural Chemicals
        "hs_cn_prefix": "3102",
        "factor_family": "material_embodied",
    },
    {
        "label": "Road freight transportation",
        "naics": "484",
        "isic": "H4923",
        "gics": "20305020",
        "hs_cn_prefix": None,
        "factor_family": "transport_lane",
    },
    {
        "label": "Rail freight transportation",
        "naics": "482",
        "isic": "H4912",
        "gics": "20304020",
        "hs_cn_prefix": None,
        "factor_family": "transport_lane",
    },
    {
        "label": "Maritime freight",
        "naics": "483",
        "isic": "H5012",
        "gics": "20302010",
        "hs_cn_prefix": None,
        "factor_family": "transport_lane",
    },
    {
        "label": "Air freight",
        "naics": "4812",
        "isic": "H5120",
        "gics": "20302020",
        "hs_cn_prefix": None,
        "factor_family": "transport_lane",
    },
    {
        "label": "Petroleum refining",
        "naics": "324110",
        "isic": "C1920",
        "gics": "10102030",
        "hs_cn_prefix": "2710",
        "factor_family": "emissions",
    },
    {
        "label": "Natural gas distribution",
        "naics": "2212",
        "isic": "D3520",
        "gics": "55102010",
        "hs_cn_prefix": None,
        "factor_family": "emissions",
    },
    {
        "label": "Waste management — landfill",
        "naics": "562212",
        "isic": "E3821",
        "gics": "20201050",
        "hs_cn_prefix": None,
        "factor_family": "waste_treatment",
    },
    {
        "label": "Commercial real estate",
        "naics": "531120",
        "isic": "L6810",
        "gics": "60101010",
        "hs_cn_prefix": None,
        "factor_family": "finance_proxy",
    },
]


#: System → field map so we can look up any row by any code.  BICS is a
#: reduced GICS variant used by Bloomberg; we treat it as an alias so
#: callers can pass either identifier.
_SYSTEMS = {
    "naics": "naics",
    "isic": "isic",
    "hs": "hs_cn_prefix",
    "cn": "hs_cn_prefix",
    "gics": "gics",
    "bics": "gics",           # alias — Bloomberg maintains a GICS-compatible subset
}


def _norm_code(system: str, code: str) -> str:
    return str(code).strip().upper().replace("-", "").replace(".", "")


def map_classification(system: str, code: str) -> MappingResult:
    """Return the cross-walk row for a single ``(system, code)`` pair."""
    sys_lower = system.lower().strip()
    if sys_lower not in _SYSTEMS:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=f"Unknown classification system '{system}'; expected one of {sorted(_SYSTEMS)}",
            raw_input=f"{system}:{code}",
        )
    field = _SYSTEMS[sys_lower]
    needle = _norm_code(sys_lower, code)
    best: Optional[Dict[str, Any]] = None
    matched_pattern: Optional[str] = None
    for row in CROSS_WALK:
        value = row.get(field)
        if value is None:
            continue
        cmp = _norm_code(sys_lower, value)
        if cmp == needle:
            best = row
            matched_pattern = cmp
            break
        if cmp and needle.startswith(cmp):
            # HS-prefix match (e.g., '7208' matches '72' prefix)
            if sys_lower in ("hs", "cn") and len(cmp) >= 2:
                best = row
                matched_pattern = cmp
                break
    if best is None:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=f"No cross-walk row for {sys_lower}:{code}",
            raw_input=f"{system}:{code}",
        )
    return MappingResult(
        canonical=dict(best),
        confidence=1.0 if matched_pattern == needle else 0.8,
        band=MappingConfidence.EXACT if matched_pattern == needle else MappingConfidence.HIGH,
        rationale=f"Matched {sys_lower}:{matched_pattern} → {best['label']}",
        matched_pattern=matched_pattern,
        raw_input=f"{system}:{code}",
    )


def cross_map_classification(
    system: str, code: str, target_systems: Optional[List[str]] = None
) -> MappingResult:
    """Map a (system, code) pair and return the full cross-walk row.

    When ``target_systems`` is provided, the result's ``canonical`` dict
    is filtered to those systems plus the label + factor_family.
    """
    base = map_classification(system, code)
    if base.canonical is None or target_systems is None:
        return base
    filtered: Dict[str, Any] = {
        "label": base.canonical["label"],
        "factor_family": base.canonical.get("factor_family"),
    }
    for sys_ in target_systems:
        sys_lower = sys_.lower().strip()
        field = _SYSTEMS.get(sys_lower)
        if field is None:
            continue
        filtered[sys_lower] = base.canonical.get(field)
    base.canonical = filtered
    return base


__all__ = ["CROSS_WALK", "map_classification", "cross_map_classification"]
