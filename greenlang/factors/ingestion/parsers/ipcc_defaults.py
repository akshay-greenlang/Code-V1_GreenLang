# -*- coding: utf-8 -*-
"""
IPCC Tier 1 default emission factors parser (F013).

Parses IPCC 2006 Guidelines + 2019 Refinement default emission factors
covering Energy, Industrial Processes, Agriculture, LULUCF, and Waste.

Expected JSON structure:
{
  "metadata": {"source": "IPCC", "version": "2019_refinement", ...},
  "energy_stationary": [ ... ],
  "energy_mobile": [ ... ],
  "industrial_processes": [ ... ],
  "agriculture": [ ... ],
  "lulucf": [ ... ],
  "waste": [ ... ]
}
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Dict, Iterator, List

from greenlang.data.emission_factor_record import (
    Boundary,
    GWPSet,
    GeographyLevel,
    Methodology,
    Scope,
)

logger = logging.getLogger(__name__)

_LICENSE = {
    "license": "IPCC-Guideline",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "IPCC_2006", "ISO_14064"]


def _slug(s: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    return x or "unknown"


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _stamp(rec: Dict[str, Any]) -> Dict[str, Any]:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rec.setdefault("created_at", now)
    rec.setdefault("updated_at", now)
    rec.setdefault("created_by", "greenlang_factors_etl")
    rec.setdefault("compliance_frameworks", _COMPLIANCE)
    gwp = dict(rec.get("gwp_100yr") or {})
    gwp.pop("co2e_total", None)
    rec["gwp_100yr"] = gwp
    return rec


def _provenance(meta: Dict[str, Any], year: int) -> Dict[str, Any]:
    return {
        "source_org": "IPCC",
        "source_publication": str(
            meta.get("source_publication", "IPCC 2006 Guidelines / 2019 Refinement")
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(meta.get("source_url", "https://www.ipcc-nggip.iges.or.jp/")),
        "version": str(meta.get("version", "2019_refinement")),
    }


def _dqs_ipcc() -> Dict[str, int]:
    """IPCC defaults: global averages with lower geographic specificity."""
    return {
        "temporal": 3,
        "geographical": 2,
        "technological": 3,
        "representativeness": 3,
        "methodological": 5,
    }


def _iter_section(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int,
    prefix: str, scope: str, boundary: str, default_unit: str,
    extra_tags: List[str],
) -> Iterator[Dict[str, Any]]:
    """Generic IPCC section parser."""
    for row in rows:
        category = _slug(str(row.get("category") or row.get("fuel_type") or row.get("process") or "unknown"))
        sub = _slug(str(row.get("subcategory") or row.get("subtype") or ""))
        geo = str(row.get("geography") or row.get("region") or "GLOBAL").upper()
        unit = _slug(str(row.get("unit") or default_unit))

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        slug_parts = [prefix, category]
        if sub:
            slug_parts.append(sub)
        slug_parts.append(unit)
        fid = f"EF:IPCC:{'_'.join(slug_parts)}:{geo}:{year}:v1"

        tags = ["ipcc", "tier1"] + extra_tags + [category]
        if sub:
            tags.append(sub)

        yield _stamp({
            "factor_id": fid,
            "fuel_type": category,
            "unit": unit.replace("_", " "),
            "geography": geo,
            "geography_level": GeographyLevel.GLOBAL.value if geo == "GLOBAL" else GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR6_100.value, "CH4_gwp": 28, "N2O_gwp": 273},
            "scope": scope,
            "boundary": boundary,
            "provenance": _provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": _safe_float(row.get("uncertainty_95ci"), 0.15),
            "dqs": _dqs_ipcc(),
            "license_info": _LICENSE,
            "tags": tags,
            "notes": row.get("notes"),
        })


_SECTIONS = {
    "energy_stationary": ("en_stat", Scope.SCOPE_1.value, Boundary.COMBUSTION.value, "gj", ["energy", "stationary"]),
    "energy_mobile": ("en_mob", Scope.SCOPE_1.value, Boundary.COMBUSTION.value, "gj", ["energy", "mobile"]),
    "industrial_processes": ("ind", Scope.SCOPE_1.value, Boundary.CRADLE_TO_GATE.value, "tonnes", ["industrial"]),
    "agriculture": ("ag", Scope.SCOPE_1.value, Boundary.COMBUSTION.value, "kg", ["agriculture"]),
    "lulucf": ("lu", Scope.SCOPE_1.value, Boundary.CRADLE_TO_GATE.value, "kg", ["lulucf", "land_use"]),
    "waste": ("waste", Scope.SCOPE_3.value, Boundary.CRADLE_TO_GATE.value, "tonnes", ["waste"]),
}


def parse_ipcc_defaults(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse IPCC default emission factors payload.

    Returns:
        List of factor dicts ready for QA validation.
    """
    meta = data.get("metadata") or {}
    year = 2019
    try:
        raw = str(meta.get("version") or meta.get("year") or "2019")
        year = int(re.search(r"\d{4}", raw).group())  # type: ignore[union-attr]
    except (TypeError, ValueError, AttributeError):
        year = 2019

    factors: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for section_key, (prefix, scope, boundary, default_unit, extra_tags) in _SECTIONS.items():
        section_rows = data.get(section_key) or []
        if not isinstance(section_rows, list):
            continue
        for f in _iter_section(section_rows, meta, year, prefix, scope, boundary, default_unit, extra_tags):
            fid = f.get("factor_id", "")
            if fid not in seen_ids:
                seen_ids.add(fid)
                factors.append(f)

    logger.info("IPCC parser produced %d factors (year=%d)", len(factors), year)
    return factors
