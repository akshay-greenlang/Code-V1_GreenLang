# -*- coding: utf-8 -*-
"""
eGRID electricity grid emission factors parser (F011).

Parses the EPA eGRID annual dataset — the authoritative source for US
electricity grid emission factors at subregion, state, and national levels.

Expected JSON structure (curated from eGRID Excel):
{
  "metadata": {"source": "eGRID", "version": "2022", ...},
  "subregions": [
    {"acronym": "AKGD", "name": "ASCC Alaska Grid", "co2_lb_mwh": 1050.2, ...},
    ...
  ],
  "states": [
    {"state": "AL", "co2_lb_mwh": 820.5, ...},
    ...
  ],
  "national": {"co2_lb_mwh": 852.3, ...},
  "generation_mix": [
    {"region": "CAMX", "coal_pct": 0.02, "gas_pct": 0.45, ...},
    ...
  ]
}
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Dict, Iterator, List, Optional

from greenlang.data.emission_factor_record import (
    Boundary,
    DataQualityScore,
    GWPSet,
    GeographyLevel,
    LicenseInfo,
    Methodology,
    Scope,
    SourceProvenance,
)

logger = logging.getLogger(__name__)

# Conversion: lb/MWh -> kg/kWh  (1 lb = 0.453592 kg, 1 MWh = 1000 kWh)
_LB_MWH_TO_KG_KWH = 0.453592 / 1000.0

_LICENSE = {
    "license": "US-Public-Domain",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "IPCC_2006", "EPA_MRR", "eGRID"]


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _slug(s: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    return x or "unknown"


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


def _make_provenance(meta: Dict[str, Any], year: int) -> Dict[str, Any]:
    return {
        "source_org": "EPA",
        "source_publication": str(
            meta.get("source_publication", f"eGRID{year}")
        ),
        "source_year": year,
        "methodology": Methodology.DIRECT_MEASUREMENT.value,
        "source_url": str(meta.get("source_url", "https://www.epa.gov/egrid")),
        "version": str(meta.get("version", f"{year}")),
    }


def _convert_lb_mwh_to_kg_kwh(lb_per_mwh: float) -> float:
    """Convert lb/MWh to kg/kWh."""
    return lb_per_mwh * _LB_MWH_TO_KG_KWH


def iter_subregions(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse eGRID subregion-level emission factors."""
    for row in rows:
        acronym = str(row.get("acronym") or row.get("subregion") or "").upper()
        if not acronym:
            continue
        name = str(row.get("name") or row.get("subregion_name") or acronym)

        # Convert from lb/MWh to kg/kWh
        co2 = _convert_lb_mwh_to_kg_kwh(
            _safe_float(row.get("co2_lb_mwh") or row.get("co2"))
        )
        ch4 = _convert_lb_mwh_to_kg_kwh(
            _safe_float(row.get("ch4_lb_mwh") or row.get("ch4"))
        )
        n2o = _convert_lb_mwh_to_kg_kwh(
            _safe_float(row.get("n2o_lb_mwh") or row.get("n2o"))
        )

        fid = f"EF:eGRID:sub_{acronym}:{acronym}:{year}:v1"
        tags = ["egrid", "electricity", "subregion", acronym.lower()]
        notes = f"eGRID subregion {acronym} ({name})"

        # Attach generation mix if present
        gen_mix = row.get("generation_mix") or {}
        if gen_mix:
            notes += f"; coal={gen_mix.get('coal_pct', '?')}%, gas={gen_mix.get('gas_pct', '?')}%"

        yield _stamp({
            "factor_id": fid,
            "fuel_type": "electricity_grid",
            "unit": "kwh",
            "geography": acronym,
            "geography_level": GeographyLevel.GRID_ZONE.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {
                "gwp_set": GWPSet.IPCC_AR6_100.value,
                "CH4_gwp": 28,
                "N2O_gwp": 273,
            },
            "scope": Scope.SCOPE_2.value,
            "boundary": Boundary.COMBUSTION.value,
            "provenance": _make_provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.03,
            "dqs": {
                "temporal": 5,
                "geographical": 5,
                "technological": 5,
                "representativeness": 5,
                "methodological": 5,
            },
            "license_info": _LICENSE,
            "tags": tags,
            "notes": notes,
        })


def iter_states(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse state-level electricity emission factors."""
    for row in rows:
        state = str(row.get("state") or row.get("state_code") or "").upper()
        if not state or len(state) != 2:
            continue

        co2 = _convert_lb_mwh_to_kg_kwh(
            _safe_float(row.get("co2_lb_mwh") or row.get("co2"))
        )
        ch4 = _convert_lb_mwh_to_kg_kwh(
            _safe_float(row.get("ch4_lb_mwh") or row.get("ch4"))
        )
        n2o = _convert_lb_mwh_to_kg_kwh(
            _safe_float(row.get("n2o_lb_mwh") or row.get("n2o"))
        )

        fid = f"EF:eGRID:state_{state}:US-{state}:{year}:v1"
        yield _stamp({
            "factor_id": fid,
            "fuel_type": "electricity_grid",
            "unit": "kwh",
            "geography": f"US-{state}",
            "geography_level": GeographyLevel.STATE.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {
                "gwp_set": GWPSet.IPCC_AR6_100.value,
                "CH4_gwp": 28,
                "N2O_gwp": 273,
            },
            "scope": Scope.SCOPE_2.value,
            "boundary": Boundary.COMBUSTION.value,
            "provenance": _make_provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.03,
            "dqs": {
                "temporal": 5,
                "geographical": 4,
                "technological": 5,
                "representativeness": 4,
                "methodological": 5,
            },
            "license_info": _LICENSE,
            "tags": ["egrid", "electricity", "state", state.lower()],
            "notes": f"eGRID state-level average for {state}",
        })


def parse_national(
    data: Dict[str, Any], meta: Dict[str, Any], year: int
) -> Optional[Dict[str, Any]]:
    """Parse national average electricity emission factor."""
    if not data:
        return None
    co2 = _convert_lb_mwh_to_kg_kwh(
        _safe_float(data.get("co2_lb_mwh") or data.get("co2"))
    )
    ch4 = _convert_lb_mwh_to_kg_kwh(
        _safe_float(data.get("ch4_lb_mwh") or data.get("ch4"))
    )
    n2o = _convert_lb_mwh_to_kg_kwh(
        _safe_float(data.get("n2o_lb_mwh") or data.get("n2o"))
    )
    fid = f"EF:eGRID:national:US:{year}:v1"
    return _stamp({
        "factor_id": fid,
        "fuel_type": "electricity_grid",
        "unit": "kwh",
        "geography": "US",
        "geography_level": GeographyLevel.COUNTRY.value,
        "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
        "gwp_100yr": {
            "gwp_set": GWPSet.IPCC_AR6_100.value,
            "CH4_gwp": 28,
            "N2O_gwp": 273,
        },
        "scope": Scope.SCOPE_2.value,
        "boundary": Boundary.COMBUSTION.value,
        "provenance": _make_provenance(meta, year),
        "valid_from": date(year, 1, 1).isoformat(),
        "uncertainty_95ci": 0.02,
        "dqs": {
            "temporal": 5,
            "geographical": 3,
            "technological": 5,
            "representativeness": 5,
            "methodological": 5,
        },
        "license_info": _LICENSE,
        "tags": ["egrid", "electricity", "national"],
        "notes": "eGRID US national average",
    })


def parse_egrid(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse a full eGRID JSON payload into normalized factor dicts.

    Args:
        data: JSON-decoded dict with ``metadata``, ``subregions``, ``states``, ``national``.

    Returns:
        List of factor dicts ready for QA validation + EmissionFactorRecord.from_dict.
    """
    meta = data.get("metadata") or {}
    year = 2022
    try:
        year = int(str(meta.get("version") or meta.get("year") or "2022").split(".")[0])
    except (TypeError, ValueError):
        year = 2022

    factors: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add(f: Dict[str, Any]) -> None:
        fid = f.get("factor_id", "")
        if fid not in seen_ids:
            seen_ids.add(fid)
            factors.append(f)

    # Subregions
    subregions = data.get("subregions") or []
    if isinstance(subregions, list):
        for f in iter_subregions(subregions, meta, year):
            _add(f)

    # States
    states = data.get("states") or []
    if isinstance(states, list):
        for f in iter_states(states, meta, year):
            _add(f)

    # National
    national = data.get("national")
    if isinstance(national, dict):
        nf = parse_national(national, meta, year)
        if nf:
            _add(nf)

    logger.info(
        "eGRID parser produced %d factors (%d subregions, %d states) year=%d",
        len(factors),
        len(subregions) if isinstance(subregions, list) else 0,
        len(states) if isinstance(states, list) else 0,
        year,
    )
    return factors
