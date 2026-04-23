# -*- coding: utf-8 -*-
"""
EPA WARM v15 parser (Wave 4-B catalog expansion).

Parses the US EPA Waste Reduction Model (WARM) v15 end-of-life GHG
factors covering landfill, incineration (WTE), composting, anaerobic
digestion, and recycling across ~20 material streams.

Source: US EPA, **public domain** (US Federal Government work).
Attribution: "U.S. EPA Waste Reduction Model (WARM) v15".

Factor IDs match the gold-set pattern
``EF:EPA:waste_<treatment>_<material>:US:v15:v1``.

Seed input lives at ``catalog_seed/_inputs/epa_warm.json``.

Expected JSON structure::

    {
      "metadata": {"source": "EPA WARM", "version": "15", ...},
      "factors": [
        {
          "treatment": "landfill",           # landfill | incineration | compost | recycling | anaerobic_digestion
          "material":  "msw",                # msw | paper | plastic | cardboard | food | aluminum | glass | steel | ...
          "co2e_per_short_ton": 0.53,        # MT CO2e per short ton of material
          "unit": "short_ton",
          "notes": "..."
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

_COMPLIANCE = ["GHG_Protocol", "GHG_Protocol_Scope3", "EPA_WARM"]

# 1 short ton = 0.907185 tonnes
_SHORT_TON_TO_TONNE = 0.907185


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
        "source_org": "US EPA",
        "source_publication": str(
            meta.get("source_publication", f"WARM v{version} Documentation")
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get(
                "source_url",
                "https://www.epa.gov/warm",
            )
        ),
        "version": f"v{version}",
    }


def _dqs(illustrative: bool) -> Dict[str, int]:
    return {
        "temporal": 4,
        "geographical": 4,
        "technological": 3,
        "representativeness": 3,
        "methodological": 3 if illustrative else 4,
    }


def _iter_factors(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    year: int,
    version: str,
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        treatment = _slug(str(row.get("treatment") or "unknown"))
        material = _slug(str(row.get("material") or "unknown"))
        if treatment == "unknown" or material == "unknown":
            continue

        unit_in = _slug(str(row.get("unit") or "short_ton"))
        co2e_val = _safe_float(row.get("co2e_per_short_ton") or row.get("co2e"))

        # Canonicalize to per-tonne for the catalog. If the seed value was
        # per short ton, convert.
        if unit_in == "short_ton":
            co2e_per_tonne = co2e_val / _SHORT_TON_TO_TONNE if co2e_val else 0.0
            out_unit = "tonnes"
        else:
            co2e_per_tonne = co2e_val
            out_unit = "tonnes"

        illustrative = bool(row.get("illustrative_value", False))
        fid = f"EF:EPA:waste_{treatment}_{material}:US:v{version}:v1"

        notes_bits = [
            f"EPA WARM v{version} end-of-life factor: {treatment} {material}"
        ]
        if illustrative:
            notes_bits.append(
                "illustrative_value=true; TODO: reconcile against full WARM "
                "v15 workbook before GA"
            )
        if row.get("notes"):
            notes_bits.append(str(row["notes"]))

        tags = [
            "epa_warm", "waste", f"treatment_{treatment}",
            f"material_{material}", "scope3_cat5",
        ]
        if illustrative:
            tags.append("illustrative")

        # Biogenic CO2 flag for composting / organics
        biogenic_flag = treatment in {"compost", "anaerobic_digestion"}

        yield _stamp({
            "factor_id": fid,
            "fuel_type": f"waste_{material}",
            "unit": out_unit,
            "geography": "US",
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2e_per_tonne, "CH4": 0.0, "N2O": 0.0},
            "gwp_100yr": {
                "gwp_set": GWPSet.IPCC_AR5_100.value,
                "CH4_gwp": 28,
                "N2O_gwp": 265,
            },
            "scope": Scope.SCOPE_3.value,
            "boundary": Boundary.CRADLE_TO_GATE.value,
            "provenance": _provenance(meta, year, version),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.25,
            "dqs": _dqs(illustrative),
            "license_info": _LICENSE,
            "tags": tags,
            "notes": "; ".join(notes_bits),
            "biogenic_flag": biogenic_flag,
            "factor_family": FactorFamily.WASTE_TREATMENT.value,
            "activity_tags": [f"waste_{treatment}", "end_of_life"],
            "sector_tags": ["waste_management"],
            "source_id": "epa_warm",
            "source_release": f"v{version}",
            "release_version": f"warm-v{version}",
        })


def parse_epa_warm(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse EPA WARM payload into factor dicts."""
    meta = data.get("metadata") or {}
    version = str(meta.get("version") or "15")
    try:
        year = int(meta.get("vintage_year") or 2020)
    except (TypeError, ValueError):
        year = 2020

    rows = data.get("factors") or []
    if not isinstance(rows, list):
        logger.warning("EPA WARM: factors is not a list; got %s", type(rows))
        return []

    factors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for f in _iter_factors(rows, meta, year, version):
        fid = f.get("factor_id", "")
        if fid and fid not in seen:
            seen.add(fid)
            factors.append(f)

    logger.info(
        "EPA WARM parser produced %d factors (version=%s, vintage=%d)",
        len(factors), version, year,
    )
    return factors


__all__ = ["parse_epa_warm"]
