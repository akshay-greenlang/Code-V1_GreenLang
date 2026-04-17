# -*- coding: utf-8 -*-
"""
EPA GHG Emission Factors Hub parser (F010).

Parses the EPA GHG Emission Factors Hub data — the primary US federal source
for Scope 1 stationary combustion, mobile combustion, and related factors.

The parser expects a JSON structure exported/curated from the Hub tables:
{
  "metadata": {"source": "EPA", "version": "2024", ...},
  "stationary_combustion": [ ... ],
  "mobile_combustion": [ ... ],
  "electricity": [ ... ],
  "steam_and_heat": [ ... ],
  "scope3_upstream": [ ... ]
}

Each section contains row dicts with fuel-specific emission factors.
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

# ---- Unit mapping: EPA native -> GreenLang ontology ----
_EPA_UNIT_MAP: Dict[str, str] = {
    "scf": "scf",
    "short ton": "tonnes",
    "short tons": "tonnes",
    "gallon": "gallons",
    "gallons": "gallons",
    "mmBtu": "mmbtu",
    "mmbtu": "mmbtu",
    "bbl": "gallons",  # barrel -> gallons (42 gal/bbl handled in factor value)
    "barrel": "gallons",
    "mcf": "scf",  # 1 mcf = 1000 scf, factor values already per-mcf
    "1000 scf": "scf",
    "therm": "therms",
    "therms": "therms",
    "kwh": "kwh",
    "mwh": "mwh",
    "kg": "kg",
    "pound": "lb",
    "lb": "lb",
    "liter": "litres",
    "litre": "litres",
    "litres": "litres",
    "liters": "litres",
    "gj": "gj",
    "m3": "m3",
    "ton-mile": "tonne_km",
    "vehicle-mile": "miles",
    "passenger-mile": "miles",
    "mile": "miles",
    "miles": "miles",
}

# Fuel type slug normalization
_FUEL_SLUG_MAP: Dict[str, str] = {
    "motor gasoline": "gasoline",
    "distillate fuel oil no. 2": "diesel",
    "distillate fuel oil no 2": "diesel",
    "diesel fuel": "diesel",
    "residual fuel oil no. 6": "residual_fuel_oil",
    "residual fuel oil no 6": "residual_fuel_oil",
    "natural gas": "natural_gas",
    "propane": "propane",
    "butane": "butane",
    "kerosene": "kerosene",
    "jet fuel": "jet_fuel",
    "kerosene-type jet fuel": "jet_fuel",
    "aviation gasoline": "aviation_gasoline",
    "lpg": "lpg",
    "liquefied petroleum gas": "lpg",
    "crude oil": "crude_oil",
    "anthracite coal": "anthracite_coal",
    "bituminous coal": "bituminous_coal",
    "sub-bituminous coal": "sub_bituminous_coal",
    "subbituminous coal": "sub_bituminous_coal",
    "lignite coal": "lignite_coal",
    "lignite": "lignite_coal",
    "mixed coal": "mixed_coal",
    "coal coke": "coal_coke",
    "petroleum coke": "petroleum_coke",
    "municipal solid waste": "msw",
    "landfill gas": "landfill_gas",
    "wood and wood waste": "wood_waste",
    "wood": "wood_waste",
    "biomass - wood": "wood_waste",
    "agricultural byproducts": "agricultural_byproducts",
    "ethanol": "ethanol",
    "biodiesel": "biodiesel",
    "biogas": "biogas",
    "blast furnace gas": "blast_furnace_gas",
    "coke oven gas": "coke_oven_gas",
    "still gas": "still_gas",
    "fuel gas": "fuel_gas",
}


def _slug(s: str) -> str:
    """Normalize a string to a URL-safe slug."""
    x = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    return x or "unknown"


def _normalize_fuel(raw: str) -> str:
    """Normalize fuel name via lookup table, fallback to slug."""
    lower = raw.strip().lower()
    return _FUEL_SLUG_MAP.get(lower, _slug(raw))


def _normalize_unit(raw: str) -> str:
    """Map EPA unit string to GreenLang ontology unit."""
    lower = raw.strip().lower()
    return _EPA_UNIT_MAP.get(lower, _slug(raw))


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _make_provenance(meta: Dict[str, Any], year: int) -> Dict[str, Any]:
    return {
        "source_org": "EPA",
        "source_publication": str(
            meta.get("source_publication", "EPA GHG Emission Factors Hub")
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get("source_url", "https://www.epa.gov/ghgemissionfactors")
        ),
        "version": str(meta.get("version", f"{year}")),
    }


def _make_dqs(
    temporal: int = 5,
    geographical: int = 4,
    technological: int = 4,
    representativeness: int = 4,
    methodological: int = 5,
) -> Dict[str, int]:
    return {
        "temporal": temporal,
        "geographical": geographical,
        "technological": technological,
        "representativeness": representativeness,
        "methodological": methodological,
    }


_LICENSE = {
    "license": "US-Public-Domain",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "IPCC_2006", "EPA_MRR"]


def _stamp(rec: Dict[str, Any], year: int) -> Dict[str, Any]:
    """Add standard timestamps and defaults."""
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


# --------------- Section Parsers ---------------


def iter_stationary_combustion(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse stationary combustion factors (boilers, furnaces, turbines)."""
    for row in rows:
        fuel_raw = str(row.get("fuel_type") or row.get("fuel") or "unknown")
        fuel = _normalize_fuel(fuel_raw)
        unit_raw = str(row.get("unit") or row.get("per_unit") or "mmbtu")
        unit = _normalize_unit(unit_raw)
        geo = str(row.get("geography") or row.get("country") or "US").upper()

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:EPA:stat_{fuel}:{geo}:{year}:v1"
        yield _stamp(
            {
                "factor_id": fid,
                "fuel_type": fuel,
                "unit": unit,
                "geography": geo,
                "geography_level": GeographyLevel.COUNTRY.value,
                "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
                "gwp_100yr": {
                    "gwp_set": GWPSet.IPCC_AR6_100.value,
                    "CH4_gwp": 28,
                    "N2O_gwp": 273,
                },
                "scope": Scope.SCOPE_1.value,
                "boundary": Boundary.COMBUSTION.value,
                "provenance": _make_provenance(meta, year),
                "valid_from": date(year, 1, 1).isoformat(),
                "uncertainty_95ci": _safe_float(row.get("uncertainty_95ci"), 0.05),
                "dqs": _make_dqs(temporal=5, geographical=4, technological=4),
                "license_info": _LICENSE,
                "tags": ["epa", "stationary_combustion", fuel],
                "notes": row.get("notes"),
            },
            year,
        )


def iter_mobile_combustion(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse mobile combustion factors (on-road, non-road, rail, aviation, marine)."""
    for row in rows:
        fuel_raw = str(row.get("fuel_type") or row.get("fuel") or "unknown")
        fuel = _normalize_fuel(fuel_raw)
        vehicle = _slug(str(row.get("vehicle_type") or row.get("mode") or "all"))
        unit_raw = str(row.get("unit") or row.get("per_unit") or "gallons")
        unit = _normalize_unit(unit_raw)
        geo = str(row.get("geography") or "US").upper()

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:EPA:mob_{fuel}_{vehicle}:{geo}:{year}:v1"
        yield _stamp(
            {
                "factor_id": fid,
                "fuel_type": fuel,
                "unit": unit,
                "geography": geo,
                "geography_level": GeographyLevel.COUNTRY.value,
                "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
                "gwp_100yr": {
                    "gwp_set": GWPSet.IPCC_AR6_100.value,
                    "CH4_gwp": 28,
                    "N2O_gwp": 273,
                },
                "scope": Scope.SCOPE_1.value,
                "boundary": Boundary.COMBUSTION.value,
                "provenance": _make_provenance(meta, year),
                "valid_from": date(year, 1, 1).isoformat(),
                "uncertainty_95ci": _safe_float(row.get("uncertainty_95ci"), 0.05),
                "dqs": _make_dqs(temporal=5, geographical=4, technological=4),
                "license_info": _LICENSE,
                "tags": ["epa", "mobile_combustion", fuel, vehicle],
                "notes": row.get("notes"),
            },
            year,
        )


def iter_electricity(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse electricity grid emission factors (eGRID subregions, state, national)."""
    for row in rows:
        region = str(
            row.get("subregion") or row.get("region") or row.get("state") or "US"
        ).upper()
        geo_level_raw = str(row.get("geography_level") or "grid_zone")
        geo_level = geo_level_raw if geo_level_raw in (
            "country", "state", "grid_zone"
        ) else "grid_zone"

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:EPA:elec:{region}:{year}:v1"
        yield _stamp(
            {
                "factor_id": fid,
                "fuel_type": "electricity_grid",
                "unit": "kwh",
                "geography": region,
                "geography_level": geo_level,
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
                "uncertainty_95ci": _safe_float(row.get("uncertainty_95ci"), 0.03),
                "dqs": _make_dqs(temporal=5, geographical=5, technological=4),
                "license_info": _LICENSE,
                "tags": ["epa", "electricity", "grid", region.lower()],
                "notes": row.get("notes"),
            },
            year,
        )


def iter_steam_and_heat(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse steam and purchased heat factors."""
    for row in rows:
        fuel_raw = str(row.get("fuel_type") or row.get("type") or "steam")
        fuel = _normalize_fuel(fuel_raw) if fuel_raw.lower() not in (
            "steam", "heat", "cooling"
        ) else _slug(fuel_raw)
        unit_raw = str(row.get("unit") or "mmbtu")
        unit = _normalize_unit(unit_raw)
        geo = str(row.get("geography") or "US").upper()

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:EPA:heat_{fuel}:{geo}:{year}:v1"
        yield _stamp(
            {
                "factor_id": fid,
                "fuel_type": fuel,
                "unit": unit,
                "geography": geo,
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
                "uncertainty_95ci": _safe_float(row.get("uncertainty_95ci"), 0.05),
                "dqs": _make_dqs(temporal=4, geographical=4, technological=4),
                "license_info": _LICENSE,
                "tags": ["epa", "steam_heat", fuel],
                "notes": row.get("notes"),
            },
            year,
        )


def iter_scope3_upstream(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], year: int
) -> Iterator[Dict[str, Any]]:
    """Parse Scope 3 upstream (well-to-tank) factors."""
    for row in rows:
        fuel_raw = str(row.get("fuel_type") or row.get("fuel") or "unknown")
        fuel = _normalize_fuel(fuel_raw)
        unit_raw = str(row.get("unit") or row.get("per_unit") or "mmbtu")
        unit = _normalize_unit(unit_raw)
        geo = str(row.get("geography") or "US").upper()

        co2 = _safe_float(row.get("co2_factor") or row.get("co2"))
        ch4 = _safe_float(row.get("ch4_factor") or row.get("ch4"))
        n2o = _safe_float(row.get("n2o_factor") or row.get("n2o"))

        fid = f"EF:EPA:wtt_{fuel}:{geo}:{year}:v1"
        yield _stamp(
            {
                "factor_id": fid,
                "fuel_type": fuel,
                "unit": unit,
                "geography": geo,
                "geography_level": GeographyLevel.COUNTRY.value,
                "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
                "gwp_100yr": {
                    "gwp_set": GWPSet.IPCC_AR6_100.value,
                    "CH4_gwp": 28,
                    "N2O_gwp": 273,
                },
                "scope": Scope.SCOPE_3.value,
                "boundary": Boundary.WTT.value,
                "provenance": _make_provenance(meta, year),
                "valid_from": date(year, 1, 1).isoformat(),
                "uncertainty_95ci": _safe_float(row.get("uncertainty_95ci"), 0.10),
                "dqs": _make_dqs(temporal=4, geographical=4, technological=3),
                "license_info": _LICENSE,
                "tags": ["epa", "scope3", "wtt", fuel],
                "notes": row.get("notes"),
            },
            year,
        )


# ---- Section dispatch ----
_SECTION_PARSERS = {
    "stationary_combustion": iter_stationary_combustion,
    "mobile_combustion": iter_mobile_combustion,
    "electricity": iter_electricity,
    "steam_and_heat": iter_steam_and_heat,
    "scope3_upstream": iter_scope3_upstream,
}


def parse_epa_ghg_hub(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse a full EPA GHG Hub JSON payload into normalized factor dicts.

    Args:
        data: JSON-decoded dict with ``metadata`` and section arrays.

    Returns:
        List of factor dicts ready for ``validate_factor_dict`` + ``EmissionFactorRecord.from_dict``.
    """
    meta = data.get("metadata") or {}
    year = 2024
    try:
        year = int(str(meta.get("version") or meta.get("year") or "2024").split(".")[0])
    except (TypeError, ValueError):
        year = 2024

    factors: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for section_key, parser_fn in _SECTION_PARSERS.items():
        section_rows = data.get(section_key)
        if not section_rows or not isinstance(section_rows, list):
            continue
        for factor_dict in parser_fn(section_rows, meta, year):
            fid = factor_dict.get("factor_id", "")
            if fid in seen_ids:
                logger.warning("Duplicate factor_id %s in EPA Hub, skipping", fid)
                continue
            seen_ids.add(fid)
            factors.append(factor_dict)

    logger.info(
        "EPA GHG Hub parser produced %d factors from %d sections (year=%d)",
        len(factors),
        sum(1 for k in _SECTION_PARSERS if isinstance(data.get(k), list)),
        year,
    )
    return factors
