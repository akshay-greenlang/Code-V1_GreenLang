# -*- coding: utf-8 -*-
"""Industry / trade classification cross-map.

Supports NAICS <-> ISIC <-> HS/CN <-> GICS, plus the newer systems added in
GAP-7 (CPC, ANZSIC, JSIC). The tiny hard-coded CROSS_WALK remains the
authoritative route for high-signal CBAM + Scope 3 Cat 1 activities.
Fuller coverage for the industry classifications (ISIC / NACE / NAICS /
ANZSIC / JSIC) ships via :mod:`greenlang.factors.mapping.industry_codes`,
which loads YAML taxonomies. For richer trade-code crosswalks (HS / CN /
CPC) refer to the EU CBAM tables in :mod:`eu_policy`.

This module keeps the narrow surface used by the resolution engine:

- :func:`map_classification` for a single code
- :func:`cross_map_classification` for selective projection
- :class:`TradeCodeSystem` enum / :func:`parse_trade_code` for the new
  HS / CN / CPC trade systems
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from greenlang.factors.mapping.base import MappingConfidence, MappingError, MappingResult

logger = logging.getLogger(__name__)


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


#: System -> field map so we can look up any row by any code. BICS is a
#: reduced GICS variant used by Bloomberg; we treat it as an alias so
#: callers can pass either identifier. NACE is also 2-digit-aligned with
#: ISIC, so we reuse the same column.
_SYSTEMS = {
    "naics": "naics",
    "isic": "isic",
    "nace": "isic",           # NACE Rev.2 aligns 1:1 with ISIC Rev.4 at 2-digit
    "hs": "hs_cn_prefix",
    "cn": "hs_cn_prefix",
    "cpc": "hs_cn_prefix",    # CPC prefixes broadly align with HS at 2-digit
    "gics": "gics",
    "bics": "gics",           # alias - Bloomberg maintains a GICS-compatible subset
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


# ---------------------------------------------------------------------------
# Trade-code extensions (HS / CN / CPC)
# ---------------------------------------------------------------------------
#
# We deliberately keep this small: the heavy lifting for CBAM code parsing
# lives in greenlang.factors.mapping.eu_policy. This section provides:
#
#   - a TradeCodeSystem enum so callers can be type-safe about which system
#     they are passing, and
#   - a shallow parse_trade_code() helper that returns the code family
#     (chapter, heading, subheading) so downstream consumers can route to
#     the correct lookup table without depending on eu_policy directly.
#
# Supported systems:
#   HS    - World Customs Organization Harmonized System (2/4/6/8/10 digit)
#   CN    - EU Combined Nomenclature (8 digit; first 6 align with HS)
#   CPC  - UN Central Product Classification (5-digit groups; first 2-4
#          align with HS chapters/headings conceptually though not literally)


class TradeCodeSystem(str, Enum):
    """Supported international product-trade classification systems."""

    HS = "hs"
    CN = "cn"
    CPC = "cpc"

    @classmethod
    def from_str(cls, name: str) -> "TradeCodeSystem":
        if not name:
            raise MappingError("Empty trade code system name")
        needle = name.strip().lower()
        for member in cls:
            if member.value == needle:
                return member
        raise MappingError(
            "Unknown trade code system %r; expected one of %s"
            % (name, [m.value for m in cls])
        )


def parse_trade_code(system: str, code: str) -> Dict[str, Any]:
    """Decompose a trade code into chapter / heading / subheading fields.

    Args:
        system: ``"hs"``, ``"cn"``, or ``"cpc"``.
        code: digit string (dots / hyphens tolerated).

    Returns:
        dict: ``{system, code, chapter, heading, subheading, length}``.

    Raises:
        MappingError: if the system is unknown or the code is non-numeric.
    """
    sys_enum = TradeCodeSystem.from_str(system)
    raw = str(code).strip().upper().replace(".", "").replace("-", "")
    if not raw.isdigit():
        raise MappingError("Trade code must be all digits; got %r" % code)

    # Normalise length. HS is most often reported at 6 digits (WCO global
    # subheading). CN extends to 8. CPC uses 5-digit groups.
    length = len(raw)
    if sys_enum in (TradeCodeSystem.HS, TradeCodeSystem.CN):
        chapter = raw[:2] if length >= 2 else raw
        heading = raw[:4] if length >= 4 else (raw + "?" * (4 - length))
        subheading = raw[:6] if length >= 6 else (raw + "?" * (6 - length))
    else:  # CPC: 1-digit section, 2-digit division, 3-digit group, 5-digit class
        chapter = raw[:2] if length >= 2 else raw
        heading = raw[:3] if length >= 3 else raw
        subheading = raw[:5] if length >= 5 else raw
    return {
        "system": sys_enum.value,
        "code": raw,
        "chapter": chapter,
        "heading": heading,
        "subheading": subheading,
        "length": length,
    }


def map_trade_code(system: str, code: str) -> MappingResult:
    """Minimal HS/CN/CPC resolver.

    Delegates to :func:`map_classification` using the existing HS-prefix
    match logic (HS/CN/CPC share the same column in :data:`CROSS_WALK`).
    Extended CBAM coverage lives in :mod:`eu_policy`.
    """
    try:
        parsed = parse_trade_code(system, code)
    except MappingError as exc:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=str(exc),
            raw_input="%s:%s" % (system, code),
        )
    # Re-use map_classification which already handles HS prefix matching.
    result = map_classification(parsed["system"], parsed["code"])
    if result.canonical is None:
        return result
    # Enrich with parsed components for downstream consumers.
    if isinstance(result.canonical, dict):
        result.canonical = dict(result.canonical)
        result.canonical["parsed"] = parsed
    return result


__all__ = [
    "CROSS_WALK",
    "TradeCodeSystem",
    "cross_map_classification",
    "map_classification",
    "map_trade_code",
    "parse_trade_code",
]
