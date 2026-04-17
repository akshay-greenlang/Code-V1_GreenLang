# -*- coding: utf-8 -*-
"""
GHG Protocol standard factors parser (F015).

Parses GHG Protocol Scope 3 calculation guidance factors:
Cat 1 (purchased goods/EEIO), Cat 4-9 (transport), Cat 6 (travel),
Cat 7 (commuting), Cat 13 (downstream leased).

Expected JSON structure:
{
  "metadata": {"source": "GHG Protocol", "version": "2024", ...},
  "cat1_purchased_goods": [ ... ],
  "cat4_upstream_transport": [ ... ],
  "cat5_waste": [ ... ],
  "cat6_business_travel": [ ... ],
  "cat7_commuting": [ ... ],
  "cat9_downstream_transport": [ ... ],
  "cat13_downstream_leased": [ ... ]
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
    "license": "WRI-WBCSD-Terms",
    "redistribution_allowed": False,
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
        "source_org": "WRI/WBCSD",
        "source_publication": str(
            meta.get("source_publication", "GHG Protocol Scope 3 Calculation Guidance")
        ),
        "source_year": year,
        "methodology": Methodology.SPEND_BASED.value,
        "source_url": str(meta.get("source_url", "https://ghgprotocol.org/")),
        "version": str(meta.get("version", f"{year}")),
    }


def _iter_category(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int,
    prefix: str, cat_num: int, boundary: str, default_unit: str,
    methodology: str, extra_tags: List[str],
) -> Iterator[Dict[str, Any]]:
    """Generic GHG Protocol category parser."""
    for row in rows:
        activity = _slug(str(row.get("activity") or row.get("sector") or row.get("category") or "unknown"))
        geo = str(row.get("geography") or row.get("region") or "GLOBAL").upper()
        unit = _slug(str(row.get("unit") or default_unit))

        co2 = _safe_float(row.get("co2_factor") or row.get("co2") or row.get("co2e"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:GHGP:{prefix}_{activity}_{unit}:{geo}:{year}:v1"

        prov = _provenance(meta, year)
        prov["methodology"] = methodology

        yield _stamp({
            "factor_id": fid,
            "fuel_type": activity,
            "unit": unit.replace("_", " "),
            "geography": geo,
            "geography_level": GeographyLevel.GLOBAL.value if geo == "GLOBAL" else GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR6_100.value, "CH4_gwp": 28, "N2O_gwp": 273},
            "scope": Scope.SCOPE_3.value,
            "boundary": boundary,
            "provenance": prov,
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": _safe_float(row.get("uncertainty_95ci"), 0.20),
            "dqs": {
                "temporal": 3,
                "geographical": 2 if geo == "GLOBAL" else 4,
                "technological": 3,
                "representativeness": 3,
                "methodological": 4,
            },
            "license_info": _LICENSE,
            "tags": ["ghg_protocol", "scope3", f"cat{cat_num}"] + extra_tags + [activity],
            "notes": row.get("notes"),
        })


_CATEGORIES = {
    "cat1_purchased_goods": ("c1", 1, Boundary.CRADLE_TO_GATE.value, "usd", Methodology.SPEND_BASED.value, ["purchased_goods", "eeio"]),
    "cat4_upstream_transport": ("c4", 4, Boundary.WTW.value, "tonne_km", Methodology.IPCC_TIER_1.value, ["transport", "upstream"]),
    "cat5_waste": ("c5", 5, Boundary.CRADLE_TO_GATE.value, "tonnes", Methodology.IPCC_TIER_1.value, ["waste"]),
    "cat6_business_travel": ("c6", 6, Boundary.WTW.value, "km", Methodology.IPCC_TIER_1.value, ["business_travel"]),
    "cat7_commuting": ("c7", 7, Boundary.WTW.value, "km", Methodology.IPCC_TIER_1.value, ["commuting"]),
    "cat9_downstream_transport": ("c9", 9, Boundary.WTW.value, "tonne_km", Methodology.IPCC_TIER_1.value, ["transport", "downstream"]),
    "cat13_downstream_leased": ("c13", 13, Boundary.CRADLE_TO_GATE.value, "m3", Methodology.IPCC_TIER_1.value, ["downstream_leased"]),
}


def parse_ghg_protocol(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse GHG Protocol Scope 3 guidance factors.

    Returns:
        List of factor dicts ready for QA validation.
    """
    meta = data.get("metadata") or {}
    year = 2024
    try:
        year = int(str(meta.get("version") or meta.get("year") or "2024").split(".")[0])
    except (TypeError, ValueError):
        year = 2024

    factors: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for section_key, (prefix, cat_num, boundary, default_unit, methodology, extra_tags) in _CATEGORIES.items():
        section_rows = data.get(section_key) or []
        if not isinstance(section_rows, list):
            continue
        for f in _iter_category(
            section_rows, meta, year, prefix, cat_num, boundary, default_unit, methodology, extra_tags
        ):
            fid = f.get("factor_id", "")
            if fid not in seen_ids:
                seen_ids.add(fid)
                factors.append(f)

    logger.info("GHG Protocol parser produced %d factors (year=%d)", len(factors), year)
    return factors
