# -*- coding: utf-8 -*-
"""
The Climate Registry (TCR) default factors parser (F016).

Parses TCR General Reporting Protocol default emission factor tables
and US-specific GHG reporting defaults.

Expected JSON structure:
{
  "metadata": {"source": "TCR", "version": "2024", ...},
  "stationary_combustion": [ ... ],
  "mobile_combustion": [ ... ],
  "electricity": [ ... ],
  "refrigerants": [ ... ]
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
    "license": "TCR-Registry-Terms",
    "redistribution_allowed": False,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "IPCC_2006", "TCR_GRP"]


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
        "source_org": "The Climate Registry",
        "source_publication": str(
            meta.get("source_publication", "TCR General Reporting Protocol")
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(meta.get("source_url", "https://www.theclimateregistry.org/")),
        "version": str(meta.get("version", f"{year}")),
    }


def _iter_section(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int,
    prefix: str, scope: str, boundary: str, default_unit: str, extra_tags: List[str],
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        fuel = _slug(str(row.get("fuel_type") or row.get("type") or row.get("category") or "unknown"))
        geo = str(row.get("geography") or "US").upper()
        unit = _slug(str(row.get("unit") or default_unit))

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:TCR:{prefix}_{fuel}_{unit}:{geo}:{year}:v1"
        yield _stamp({
            "factor_id": fid,
            "fuel_type": fuel,
            "unit": unit.replace("_", " "),
            "geography": geo,
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR5_100.value, "CH4_gwp": 28, "N2O_gwp": 265},
            "scope": scope,
            "boundary": boundary,
            "provenance": _provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.07,
            "dqs": {"temporal": 4, "geographical": 4, "technological": 4, "representativeness": 4, "methodological": 4},
            "license_info": _LICENSE,
            "tags": ["tcr", "grp"] + extra_tags + [fuel],
            "notes": row.get("notes"),
        })


_SECTIONS = {
    "stationary_combustion": ("stat", Scope.SCOPE_1.value, Boundary.COMBUSTION.value, "mmbtu", ["stationary"]),
    "mobile_combustion": ("mob", Scope.SCOPE_1.value, Boundary.COMBUSTION.value, "gallons", ["mobile"]),
    "electricity": ("elec", Scope.SCOPE_2.value, Boundary.COMBUSTION.value, "kwh", ["electricity"]),
    "refrigerants": ("ref", Scope.SCOPE_1.value, Boundary.COMBUSTION.value, "kg", ["refrigerants"]),
}


def parse_tcr(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse TCR default emission factor tables."""
    meta = data.get("metadata") or {}
    year = 2024
    try:
        year = int(str(meta.get("version") or meta.get("year") or "2024").split(".")[0])
    except (TypeError, ValueError):
        year = 2024

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

    logger.info("TCR parser produced %d factors (year=%d)", len(factors), year)
    return factors
