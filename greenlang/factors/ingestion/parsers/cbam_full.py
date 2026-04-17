# -*- coding: utf-8 -*-
"""
CBAM full coverage parser (F014).

Expands the existing CBAM parser to cover all CBAM-regulated products:
iron & steel, aluminum, cement, fertilizers, electricity, hydrogen.

Expected JSON structure:
{
  "metadata": {"source": "EU Commission", "version": "2024", ...},
  "products": {
    "iron_steel": {"categories": [...], "by_country": {...}},
    "aluminum": {"categories": [...], "by_country": {...}},
    "cement": {"categories": [...], "by_country": {...}},
    "fertilizers": {"categories": [...], "by_country": {...}},
    "electricity": {"by_country": {...}},
    "hydrogen": {"categories": [...], "by_country": {...}}
  }
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
    "license": "EU-Publication",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["EU_CBAM", "GHG_Protocol", "IPCC_2006"]


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
        "source_org": "EU Commission",
        "source_publication": str(
            meta.get("source_publication", "CBAM default emission values")
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get(
                "source_url",
                "https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en",
            )
        ),
        "version": str(meta.get("version", f"{year}")),
    }


def _iter_product_entries(
    product_key: str, product_data: Dict[str, Any], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse entries for a single CBAM product group."""
    categories = product_data.get("categories") or [{"name": product_key}]
    by_country = product_data.get("by_country") or {}

    for cat in categories:
        cat_name = _slug(str(cat.get("name") or cat.get("category") or product_key))
        unit = str(cat.get("unit") or "kg_product")

        for country, row in by_country.items():
            if not isinstance(row, dict):
                continue
            country = str(country).upper()
            direct = _safe_float(row.get("direct_emissions_factor") or row.get("direct"))
            indirect = _safe_float(row.get("indirect_emissions_factor") or row.get("indirect"))
            total = direct + indirect

            fid = f"EF:CBAM:{_slug(product_key)}_{cat_name}:{country}:{year}:v1"

            yield _stamp({
                "factor_id": fid,
                "fuel_type": f"cbam_{_slug(product_key)}_{cat_name}",
                "unit": unit,
                "geography": country,
                "geography_level": GeographyLevel.COUNTRY.value,
                "vectors": {"CO2": total, "CH4": 0.0, "N2O": 0.0},
                "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR6_100.value, "CH4_gwp": 28, "N2O_gwp": 273},
                "scope": Scope.SCOPE_3.value,
                "boundary": Boundary.CRADLE_TO_GATE.value,
                "provenance": _provenance(meta, year),
                "valid_from": date(year, 1, 1).isoformat(),
                "uncertainty_95ci": 0.15,
                "dqs": {
                    "temporal": 4,
                    "geographical": 4,
                    "technological": 3,
                    "representativeness": 3,
                    "methodological": 4,
                },
                "license_info": _LICENSE,
                "tags": ["cbam", "cbam_2026", product_key, cat_name, "CBAM_2026"],
                "notes": f"CBAM default: direct={direct}, indirect={indirect}",
            })


def parse_cbam_full(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse a full CBAM payload covering all regulated product groups.

    Returns:
        List of factor dicts ready for QA validation.
    """
    meta = data.get("metadata") or {}
    year = 2024
    try:
        year = int(str(meta.get("version") or meta.get("year") or "2024").split(".")[0])
    except (TypeError, ValueError):
        year = 2024

    products = data.get("products") or {}
    factors: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for product_key, product_data in products.items():
        if not isinstance(product_data, dict):
            continue
        for f in _iter_product_entries(product_key, product_data, meta, year):
            fid = f.get("factor_id", "")
            if fid not in seen_ids:
                seen_ids.add(fid)
                factors.append(f)

    logger.info(
        "CBAM full parser produced %d factors from %d product groups (year=%d)",
        len(factors), len(products), year,
    )
    return factors
