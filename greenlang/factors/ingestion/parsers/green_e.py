# -*- coding: utf-8 -*-
"""
Green-e residual mix emission factors parser (F017).

Parses annual Green-e residual mix emission rates for US + Canada —
the standard source for Scope 2 market-based electricity factors.

Expected JSON structure:
{
  "metadata": {"source": "Green-e", "version": "2023", ...},
  "residual_mix": [
    {"region": "US-NEPOOL", "state": "MA", "co2_lb_mwh": 680, ...},
    ...
  ]
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

_LB_MWH_TO_KG_KWH = 0.453592 / 1000.0

_LICENSE = {
    "license": "Green-e-Terms",
    "redistribution_allowed": False,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "Scope2_Guidance", "Green_e"]


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


def iter_residual_mix(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse residual mix entries."""
    for row in rows:
        region = str(row.get("region") or row.get("grid_zone") or "").upper()
        state = str(row.get("state") or row.get("state_code") or "").upper()
        geo = region or state or "US"

        geo_level = GeographyLevel.GRID_ZONE.value
        if state and len(state) == 2 and not region:
            geo_level = GeographyLevel.STATE.value
        elif geo in ("US", "CA"):
            geo_level = GeographyLevel.COUNTRY.value

        co2 = _safe_float(row.get("co2_lb_mwh") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_lb_mwh") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_lb_mwh") or row.get("n2o"))

        # Convert from lb/MWh to kg/kWh if values seem large (>1 = lb/MWh)
        if co2 > 1.0:
            co2 = co2 * _LB_MWH_TO_KG_KWH
            ch4 = ch4 * _LB_MWH_TO_KG_KWH
            n2o = n2o * _LB_MWH_TO_KG_KWH

        geo_slug = _slug(geo)
        fid = f"EF:GreenE:residual_{geo_slug}:{geo}:{year}:v1"

        yield _stamp({
            "factor_id": fid,
            "fuel_type": "electricity_residual_mix",
            "unit": "kwh",
            "geography": geo,
            "geography_level": geo_level,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR6_100.value, "CH4_gwp": 28, "N2O_gwp": 273},
            "scope": Scope.SCOPE_2.value,
            "boundary": Boundary.COMBUSTION.value,
            "provenance": {
                "source_org": "Green-e / Center for Resource Solutions",
                "source_publication": str(meta.get("source_publication", f"Green-e Residual Mix {year}")),
                "source_year": year,
                "methodology": Methodology.DIRECT_MEASUREMENT.value,
                "source_url": str(meta.get("source_url", "https://www.green-e.org/")),
                "version": str(meta.get("version", f"{year}")),
            },
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.05,
            "dqs": {"temporal": 5, "geographical": 4, "technological": 4, "representativeness": 4, "methodological": 5},
            "license_info": _LICENSE,
            "tags": ["green_e", "residual_mix", "scope2", "market_based", geo_slug],
            "notes": row.get("notes"),
        })


def parse_green_e(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse Green-e residual mix emission factors."""
    meta = data.get("metadata") or {}
    year = 2023
    try:
        year = int(str(meta.get("version") or meta.get("year") or "2023").split(".")[0])
    except (TypeError, ValueError):
        year = 2023

    rows = data.get("residual_mix") or []
    if not isinstance(rows, list):
        return []

    factors: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for f in iter_residual_mix(rows, meta, year):
        fid = f.get("factor_id", "")
        if fid not in seen_ids:
            seen_ids.add(fid)
            factors.append(f)

    logger.info("Green-e parser produced %d factors (year=%d)", len(factors), year)
    return factors
