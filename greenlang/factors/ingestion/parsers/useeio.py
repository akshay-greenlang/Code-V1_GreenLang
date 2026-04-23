# -*- coding: utf-8 -*-
"""
USEEIO v2 parser (Wave 4-B catalog expansion).

Parses the US EPA USEEIO (Environmentally-Extended Input-Output) v2.0
commodity-level GHG intensity factors. USEEIO publishes kg-CO2e/USD
spend for NAICS 6-digit US sectors and is **US Federal Government work,
public domain** — safe to redistribute with EPA attribution.

The parser covers the ``finance_proxy`` / ``purchased_goods_proxy`` family
for Scope 3 Category 1 spend-based screening. Factor IDs match the gold
set pattern ``EF:EEIO:<naics_code>:US:v2.0:v1``.

This file ONLY shapes pre-extracted rows. The seed input lives at
``catalog_seed/_inputs/useeio.json`` and is authoritatively sourced from
the EPA USEEIO v2 Supply Chain GHG Factors dataset. Rows lacking a
verified published value MUST carry ``illustrative_value: true`` with a
TODO pointing to the bulk import.

Expected JSON structure::

    {
      "metadata": {"source": "USEEIO", "version": "2.0", "vintage_year": 2022, ...},
      "commodities": [
        {
          "naics_code": "453210",
          "commodity_name": "Office supplies and stationery stores",
          "kg_co2e_per_usd": 0.251,
          "illustrative_value": false
        },
        ...
      ]
    }
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterator, List

from greenlang.data.canonical_v2 import FactorFamily
from greenlang.data.emission_factor_record import (
    Boundary,
    GWPSet,
    GeographyLevel,
    Methodology,
    Scope,
)

logger = logging.getLogger(__name__)

_LICENSE = {
    "license": "US-Public-Domain",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "GHG_Protocol_Scope3", "USEPA_SCGHG"]


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stamp(rec: Dict[str, Any]) -> Dict[str, Any]:
    now = _now_iso()
    rec.setdefault("created_at", now)
    rec.setdefault("updated_at", now)
    rec.setdefault("created_by", "greenlang_factors_etl")
    rec.setdefault("compliance_frameworks", _COMPLIANCE)
    gwp = dict(rec.get("gwp_100yr") or {})
    gwp.pop("co2e_total", None)
    rec["gwp_100yr"] = gwp
    return rec


def _provenance(meta: Dict[str, Any], year: int, version: str) -> Dict[str, Any]:
    return {
        "source_org": "US EPA (Office of Research and Development)",
        "source_publication": str(
            meta.get(
                "source_publication",
                "USEEIO v2.0 Supply Chain GHG Emission Factors",
            )
        ),
        "source_year": year,
        "methodology": Methodology.IO_MODEL.value if hasattr(Methodology, "IO_MODEL") else Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get(
                "source_url",
                "https://www.epa.gov/land-research/us-environmentally-extended-input-output-useeio-models",
            )
        ),
        "version": version,
    }


def _dqs(illustrative: bool) -> Dict[str, int]:
    # USEEIO is a national IO model — geographic specificity is US-wide only.
    # When rows are marked illustrative, drop methodological one notch.
    return {
        "temporal": 3,
        "geographical": 3,
        "technological": 3,
        "representativeness": 3,
        "methodological": 3 if illustrative else 4,
    }


def _iter_commodities(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    year: int,
    version: str,
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        naics = str(row.get("naics_code") or row.get("naics") or "").strip()
        if not naics:
            continue
        name = str(row.get("commodity_name") or row.get("name") or naics)
        kg_co2e_per_usd = _safe_float(row.get("kg_co2e_per_usd") or row.get("ef"))
        illustrative = bool(row.get("illustrative_value", False))

        fid = f"EF:EEIO:{naics}:US:v{version}:v1"

        notes_bits = [f"USEEIO v{version} sector: {name}"]
        if illustrative:
            notes_bits.append(
                "illustrative_value=true; TODO: bulk-import full USEEIO v2 "
                "before GA"
            )
        if row.get("notes"):
            notes_bits.append(str(row["notes"]))

        tags = ["useeio", "spend_based", "scope3_cat1", f"naics_{naics}"]
        if illustrative:
            tags.append("illustrative")

        yield _stamp({
            "factor_id": fid,
            "fuel_type": _slug(name)[:40],
            "unit": "usd",
            "geography": "US",
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": kg_co2e_per_usd, "CH4": 0.0, "N2O": 0.0},
            "gwp_100yr": {
                "gwp_set": GWPSet.IPCC_AR5_100.value,
                "CH4_gwp": 28,
                "N2O_gwp": 265,
            },
            "scope": Scope.SCOPE_3.value,
            "boundary": Boundary.CRADLE_TO_GATE.value,
            "provenance": _provenance(meta, year, version),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.30,
            "dqs": _dqs(illustrative),
            "license_info": _LICENSE,
            "tags": tags,
            "notes": "; ".join(notes_bits),
            "factor_family": FactorFamily.FINANCE_PROXY.value,
            "activity_tags": ["spend_based", "purchased_goods_and_services"],
            "sector_tags": [f"naics_{naics[:2]}"],
            "source_id": "useeio_v2",
            "source_release": f"v{version}",
            "release_version": f"useeio-v{version}",
        })


def parse_useeio(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse USEEIO v2 payload into normalized factor dicts.

    Args:
        data: Dict with ``metadata`` and ``commodities``.

    Returns:
        List of factor dicts — one per NAICS commodity.
    """
    meta = data.get("metadata") or {}
    version = str(meta.get("version") or "2.0")
    try:
        year = int(meta.get("vintage_year") or 2022)
    except (TypeError, ValueError):
        year = 2022

    rows = data.get("commodities") or []
    if not isinstance(rows, list):
        logger.warning("USEEIO: commodities is not a list; got %s", type(rows))
        return []

    factors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for f in _iter_commodities(rows, meta, year, version):
        fid = f.get("factor_id", "")
        if fid and fid not in seen:
            seen.add(fid)
            factors.append(f)

    logger.info(
        "USEEIO parser produced %d factors (version=%s, vintage=%d)",
        len(factors), version, year,
    )
    return factors


__all__ = ["parse_useeio"]
