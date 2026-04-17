# -*- coding: utf-8 -*-
"""
DESNZ/DEFRA full UK GHG conversion factors parser (F012).

Parses the full UK Department for Energy Security and Net Zero (DESNZ)
GHG conversion factors spreadsheet — covering Scope 1-3.

Expected JSON structure (curated from DESNZ Excel):
{
  "metadata": {"source": "DESNZ", "version": "2024", "region": "UK", ...},
  "scope1_fuels": [ ... ],           # Stationary combustion fuels
  "scope1_bioenergy": [ ... ],       # Bioenergy fuels
  "scope2_electricity": [ ... ],     # UK grid + generation
  "scope2_heat_steam": [ ... ],      # District heat/steam
  "scope3_wtt": [ ... ],             # Well-to-tank
  "scope3_freight": [ ... ],         # Freight transport
  "scope3_business_travel": [ ... ], # Business travel (air, rail, car, hotel)
  "scope3_water": [ ... ],           # Water supply/treatment
  "scope3_waste": [ ... ],           # Waste disposal
  "scope3_materials": [ ... ]        # Embodied carbon of materials
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
    "license": "OGL-UK-v3",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "IPCC_2006", "UK_SECR", "UK_ESOS"]


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
        "source_org": "DESNZ",
        "source_publication": str(
            meta.get("source_publication", f"UK GHG Conversion Factors {year}")
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get(
                "source_url",
                "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting-of-greenhouse-gas-emissions",
            )
        ),
        "version": str(meta.get("version", f"{year}")),
    }


def _dqs(temporal: int = 5, geographical: int = 5, technological: int = 4,
         representativeness: int = 4, methodological: int = 5) -> Dict[str, int]:
    return {
        "temporal": temporal,
        "geographical": geographical,
        "technological": technological,
        "representativeness": representativeness,
        "methodological": methodological,
    }


# ---- Section Parsers ----

def _iter_scope1_fuels(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int, geo: str
) -> Iterator[Dict[str, Any]]:
    """Scope 1 stationary combustion fuels (natural gas, oils, coal, LPG, etc.)."""
    for row in rows:
        fuel = _slug(str(row.get("fuel_type") or row.get("fuel") or "unknown"))
        unit = _slug(str(row.get("unit") or "kwh"))

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:DESNZ:s1_{fuel}_{unit}:{geo}:{year}:v1"
        yield _stamp({
            "factor_id": fid,
            "fuel_type": fuel,
            "unit": unit.replace("_", " "),
            "geography": geo,
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR5_100.value, "CH4_gwp": 28, "N2O_gwp": 265},
            "scope": Scope.SCOPE_1.value,
            "boundary": Boundary.COMBUSTION.value,
            "provenance": _provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.05,
            "dqs": _dqs(),
            "license_info": _LICENSE,
            "tags": ["desnz", "scope1", "stationary", fuel],
            "notes": row.get("notes"),
        })


def _iter_scope1_bioenergy(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int, geo: str
) -> Iterator[Dict[str, Any]]:
    """Scope 1 bioenergy fuels (wood, biogas, biomethane, etc.)."""
    for row in rows:
        fuel = _slug(str(row.get("fuel_type") or "bioenergy"))
        unit = _slug(str(row.get("unit") or "kwh"))

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))
        bio_co2 = _safe_float(row.get("biogenic_co2"))

        rec = {
            "factor_id": f"EF:DESNZ:s1_bio_{fuel}_{unit}:{geo}:{year}:v1",
            "fuel_type": fuel,
            "unit": unit.replace("_", " "),
            "geography": geo,
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR5_100.value, "CH4_gwp": 28, "N2O_gwp": 265},
            "scope": Scope.SCOPE_1.value,
            "boundary": Boundary.COMBUSTION.value,
            "provenance": _provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.07,
            "dqs": _dqs(technological=3),
            "license_info": _LICENSE,
            "tags": ["desnz", "scope1", "bioenergy", fuel],
            "notes": f"Biogenic CO2={bio_co2}" if bio_co2 else row.get("notes"),
        }
        yield _stamp(rec)


def _iter_scope2_electricity(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int, geo: str
) -> Iterator[Dict[str, Any]]:
    """Scope 2 UK grid electricity (generation + T&D)."""
    for row in rows:
        elec_type = _slug(str(row.get("type") or row.get("grid_type") or "grid_average"))

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:DESNZ:s2_elec_{elec_type}:{geo}:{year}:v1"
        yield _stamp({
            "factor_id": fid,
            "fuel_type": f"electricity_{elec_type}",
            "unit": "kwh",
            "geography": geo,
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR5_100.value, "CH4_gwp": 28, "N2O_gwp": 265},
            "scope": Scope.SCOPE_2.value,
            "boundary": Boundary.COMBUSTION.value,
            "provenance": _provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.03,
            "dqs": _dqs(geographical=5, technological=5),
            "license_info": _LICENSE,
            "tags": ["desnz", "scope2", "electricity", elec_type],
            "notes": row.get("notes"),
        })


def _iter_scope2_heat_steam(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int, geo: str
) -> Iterator[Dict[str, Any]]:
    """Scope 2 district heat/steam/cooling factors."""
    for row in rows:
        heat_type = _slug(str(row.get("type") or "district_heat"))
        unit = _slug(str(row.get("unit") or "kwh"))

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:DESNZ:s2_heat_{heat_type}:{geo}:{year}:v1"
        yield _stamp({
            "factor_id": fid,
            "fuel_type": heat_type,
            "unit": unit.replace("_", " "),
            "geography": geo,
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR5_100.value, "CH4_gwp": 28, "N2O_gwp": 265},
            "scope": Scope.SCOPE_2.value,
            "boundary": Boundary.COMBUSTION.value,
            "provenance": _provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": 0.05,
            "dqs": _dqs(temporal=4),
            "license_info": _LICENSE,
            "tags": ["desnz", "scope2", "heat_steam", heat_type],
            "notes": row.get("notes"),
        })


def _iter_scope3_section(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int, geo: str,
    prefix: str, boundary: str, default_tags: List[str],
    default_unit: str = "kwh",
) -> Iterator[Dict[str, Any]]:
    """Generic Scope 3 section parser (WTT, freight, travel, waste, water, materials)."""
    for row in rows:
        activity = _slug(str(row.get("activity") or row.get("type") or row.get("category") or "unknown"))
        unit = _slug(str(row.get("unit") or default_unit))

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:DESNZ:{prefix}_{activity}_{unit}:{geo}:{year}:v1"
        yield _stamp({
            "factor_id": fid,
            "fuel_type": activity,
            "unit": unit.replace("_", " "),
            "geography": geo,
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "gwp_100yr": {"gwp_set": GWPSet.IPCC_AR5_100.value, "CH4_gwp": 28, "N2O_gwp": 265},
            "scope": Scope.SCOPE_3.value,
            "boundary": boundary,
            "provenance": _provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "uncertainty_95ci": _safe_float(row.get("uncertainty_95ci"), 0.10),
            "dqs": _dqs(temporal=4, geographical=4, technological=3, representativeness=3),
            "license_info": _LICENSE,
            "tags": ["desnz", "scope3"] + default_tags + [activity],
            "notes": row.get("notes"),
        })


# ---- Section dispatch ----
_SCOPE3_SECTIONS = {
    "scope3_wtt": ("s3_wtt", Boundary.WTT.value, ["wtt"], "kwh"),
    "scope3_freight": ("s3_freight", Boundary.WTW.value, ["freight", "transport"], "tonne_km"),
    "scope3_business_travel": ("s3_travel", Boundary.WTW.value, ["business_travel"], "km"),
    "scope3_water": ("s3_water", Boundary.CRADLE_TO_GATE.value, ["water"], "m3"),
    "scope3_waste": ("s3_waste", Boundary.CRADLE_TO_GATE.value, ["waste"], "tonnes"),
    "scope3_materials": ("s3_material", Boundary.CRADLE_TO_GATE.value, ["materials", "embodied"], "kg"),
}


def parse_desnz_uk(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse a full DESNZ UK GHG conversion factors payload.

    Returns:
        List of factor dicts ready for QA validation.
    """
    meta = data.get("metadata") or {}
    geo = str(meta.get("region") or "UK").upper()
    year = 2024
    try:
        year = int(str(meta.get("version") or meta.get("year") or "2024").split(".")[0])
    except (TypeError, ValueError):
        year = 2024

    factors: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add(f: Dict[str, Any]) -> None:
        fid = f.get("factor_id", "")
        if fid not in seen_ids:
            seen_ids.add(fid)
            factors.append(f)

    # Scope 1 fuels
    s1 = data.get("scope1_fuels") or []
    if isinstance(s1, list):
        for f in _iter_scope1_fuels(s1, meta, year, geo):
            _add(f)

    # Scope 1 bioenergy
    s1b = data.get("scope1_bioenergy") or []
    if isinstance(s1b, list):
        for f in _iter_scope1_bioenergy(s1b, meta, year, geo):
            _add(f)

    # Scope 2 electricity
    s2e = data.get("scope2_electricity") or []
    if isinstance(s2e, list):
        for f in _iter_scope2_electricity(s2e, meta, year, geo):
            _add(f)

    # Scope 2 heat/steam
    s2h = data.get("scope2_heat_steam") or []
    if isinstance(s2h, list):
        for f in _iter_scope2_heat_steam(s2h, meta, year, geo):
            _add(f)

    # Scope 3 sections
    for section_key, (prefix, boundary, tags, default_unit) in _SCOPE3_SECTIONS.items():
        section_rows = data.get(section_key) or []
        if isinstance(section_rows, list):
            for f in _iter_scope3_section(
                section_rows, meta, year, geo, prefix, boundary, tags, default_unit
            ):
                _add(f)

    logger.info("DESNZ parser produced %d factors for %s year=%d", len(factors), geo, year)
    return factors
