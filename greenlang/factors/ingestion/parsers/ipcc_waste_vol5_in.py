# -*- coding: utf-8 -*-
"""
IPCC 2006 GL Vol 5 (Waste) — India-parameterized default factors.

Wave 5 catalog expansion. Emits India-specific default values from the
IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Volume 5
(Waste), chapters 2 (solid waste disposal), 3 (biological treatment),
4 (incineration and open burning), 5 (wastewater), chapter-6 defaults
parameterised for the Indian climate zone (tropical / wet) with
country-specific DOC fractions, MCF, and degradation constants.

Source: IPCC 2006 GL Vol 5 — default values; climate zone parameters
and MSW composition defaults pulled from Vol 5 Ch 2-3 tables and
Vol 5 Ch 6 wastewater defaults, parameterised for India (tropical wet).

License: ``ipcc_reference`` — IPCC Guidelines are public international
publications; numerical factual default values carry no copyright
restriction and may be redistributed with attribution to the IPCC Task
Force on National GHG Inventories (TFI).

Factor IDs follow the gold-pattern shape:
    ``EF:IPCC_WASTE:<category>_<stream>:IN:<vintage>:v1``

for example::

    EF:IPCC_WASTE:swds_msw_food:IN:2006:v1
    EF:IPCC_WASTE:incineration_msw:IN:2006:v1
    EF:IPCC_WASTE:wastewater_domestic:IN:2006:v1
    EF:IPCC_WASTE:biological_compost_msw:IN:2006:v1

Method-pack compatibility: ``corporate`` (Scope-3 Cat.5 waste),
``national_inventory`` (IPCC-aligned reporting), ``eu_policy`` (when
used as a fallback in CDM/Verra waste-sector baselines).

Expected payload::

    {
      "metadata": {
        "source": "IPCC 2006 GL Vol 5",
        "vintage_year": 2006,
        "country": "IN",
        "climate_zone": "tropical_wet"
      },
      "categories": {
        "swds": [
          {
            "stream": "msw_food",
            "doc_fraction": 0.15,
            "doc_f_fraction": 0.5,
            "mcf": 0.8,
            "methane_correction": 1.0,
            "kg_ch4_per_tonne": 55.0
          },
          ...
        ],
        "incineration": [
          {"stream": "msw", "kg_co2e_per_tonne": 900.0, "fossil_fraction": 0.39},
          ...
        ],
        "biological": [
          {"stream": "compost_msw", "kg_ch4_per_tonne": 4.0,
           "kg_n2o_per_tonne": 0.3},
          ...
        ],
        "wastewater": [
          {"stream": "domestic", "mcf": 0.8, "b0_kg_ch4_per_kg_bod": 0.6,
           "kg_ch4_per_m3": 0.18, "kg_n2o_per_person_per_yr": 0.005},
          ...
        ]
      }
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
    "license": "IPCC-Guideline",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = [
    "GHG_Protocol",
    "GHG_Protocol_Scope3",
    "IPCC_2006",
    "UNFCCC_National_Inventory",
]

# AR6 100yr GWPs for post-conversion CO2e back-of-envelope where the seed
# gives CH4/N2O explicitly.
_GWP_CH4 = 28
_GWP_N2O = 273

# Canonical category -> (id_prefix, factor_family, unit, activity_tag).
_CATEGORY_META: Dict[str, Dict[str, str]] = {
    "swds": {
        "id_prefix": "swds",
        "family": FactorFamily.WASTE_TREATMENT.value,
        "unit": "tonne",
        "activity": "waste_landfill",
        "boundary": Boundary.CRADLE_TO_GATE.value,
    },
    "incineration": {
        "id_prefix": "incineration",
        "family": FactorFamily.WASTE_TREATMENT.value,
        "unit": "tonne",
        "activity": "waste_incineration",
        "boundary": Boundary.CRADLE_TO_GATE.value,
    },
    "biological": {
        "id_prefix": "biological",
        "family": FactorFamily.WASTE_TREATMENT.value,
        "unit": "tonne",
        "activity": "waste_biological_treatment",
        "boundary": Boundary.CRADLE_TO_GATE.value,
    },
    "wastewater": {
        "id_prefix": "wastewater",
        "family": FactorFamily.WASTE_TREATMENT.value,
        "unit": "m3",
        "activity": "wastewater_treatment",
        "boundary": Boundary.CRADLE_TO_GATE.value,
    },
}


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


def _provenance(
    meta: Dict[str, Any], year: int, category: str
) -> Dict[str, Any]:
    chapter_map = {
        "swds": "Vol 5 Ch 2 — Solid Waste Disposal",
        "incineration": "Vol 5 Ch 5 — Incineration and Open Burning",
        "biological": "Vol 5 Ch 4 — Biological Treatment of Solid Waste",
        "wastewater": "Vol 5 Ch 6 — Wastewater Treatment and Discharge",
    }
    chapter = chapter_map.get(category, "Vol 5 — Waste")
    return {
        "source_org": "IPCC Task Force on National GHG Inventories",
        "source_publication": str(
            meta.get(
                "source_publication",
                f"IPCC 2006 Guidelines for National GHG Inventories, {chapter}",
            )
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get(
                "source_url",
                "https://www.ipcc-nggip.iges.or.jp/public/2006gl/vol5.html",
            )
        ),
        "version": f"ipcc2006-vol5-{year}",
    }


def _dqs(illustrative: bool) -> Dict[str, int]:
    # IPCC Tier-1 defaults are medium temporal/tech representativeness for
    # any one country; India-parameterised geography raises the geo score.
    return {
        "temporal": 3,
        "geographical": 4,
        "technological": 3,
        "representativeness": 3,
        "methodological": 3 if illustrative else 4,
    }


def _ch4_n2o_to_co2e(
    kg_ch4: float, kg_n2o: float, extra_co2: float = 0.0
) -> float:
    return extra_co2 + kg_ch4 * _GWP_CH4 + kg_n2o * _GWP_N2O


def _swds_factor(
    row: Dict[str, Any],
    meta: Dict[str, Any],
    year: int,
    country: str,
    climate_zone: str,
) -> Dict[str, Any]:
    stream = _slug(str(row.get("stream") or "msw"))
    doc = _safe_float(row.get("doc_fraction"))
    doc_f = _safe_float(row.get("doc_f_fraction"), 0.5)
    mcf = _safe_float(row.get("mcf"), 0.8)
    # Primary emission vector is CH4 from the first-order-decay proxy; the
    # seed may pre-compute the first-year equivalent kg CH4/t, or we fall
    # back to F = MCF * DOC * DOC_F * (16/12) * 0.5 * 1000 kg CH4/t.
    kg_ch4 = _safe_float(row.get("kg_ch4_per_tonne"))
    if kg_ch4 <= 0.0 and doc > 0.0:
        kg_ch4 = mcf * doc * doc_f * (16.0 / 12.0) * 0.5 * 1000.0
    if kg_ch4 <= 0.0:
        return {}

    illustrative = bool(row.get("illustrative_value", False))
    meta_cat = _CATEGORY_META["swds"]
    fid = f"EF:IPCC_WASTE:{meta_cat['id_prefix']}_{stream}:{country}:{year}:v1"
    co2e = _ch4_n2o_to_co2e(kg_ch4, 0.0)

    notes_parts = [
        f"IPCC 2006 Vol 5 Ch 2 SWDS default for {stream} ({country}, "
        f"climate={climate_zone}); DOC={doc}, DOC_F={doc_f}, MCF={mcf}; "
        f"kg CH4/t={kg_ch4:.2f}"
    ]
    if illustrative:
        notes_parts.append(
            "illustrative_value=true; TODO: reconcile with India national "
            "inventory NATCOM-4 waste-sector tables"
        )
    if row.get("notes"):
        notes_parts.append(str(row["notes"]))

    tags = [
        "ipcc_2006", "vol5", "waste", "swds", "landfill",
        f"stream_{stream}", f"country_{country.lower()}",
        f"climate_{climate_zone}", "scope3_cat5",
    ]
    if illustrative:
        tags.append("illustrative")

    return _stamp({
        "factor_id": fid,
        "fuel_type": f"waste_{stream}",
        "unit": meta_cat["unit"],
        "geography": country,
        "geography_level": GeographyLevel.COUNTRY.value,
        "vectors": {"CO2": co2e, "CH4": kg_ch4, "N2O": 0.0},
        "gwp_100yr": {
            "gwp_set": GWPSet.IPCC_AR6_100.value,
            "CH4_gwp": _GWP_CH4,
            "N2O_gwp": _GWP_N2O,
        },
        "scope": Scope.SCOPE_3.value,
        "boundary": meta_cat["boundary"],
        "provenance": _provenance(meta, year, "swds"),
        "valid_from": date(year, 1, 1).isoformat(),
        "valid_to": date(9999, 12, 31).isoformat(),
        "uncertainty_95ci": 0.40,
        "dqs": _dqs(illustrative),
        "license_info": _LICENSE,
        "tags": tags,
        "notes": "; ".join(notes_parts),
        "biogenic_flag": bool(row.get("biogenic", True)),
        "factor_family": meta_cat["family"],
        "activity_tags": [meta_cat["activity"], f"waste_{stream}"],
        "sector_tags": ["waste_management", "solid_waste_disposal"],
        "source_id": "ipcc_waste_vol5_in",
        "source_release": f"ipcc2006-vol5-{year}",
        "release_version": f"ipcc2006-vol5-{year}",
        "validation_flags": {
            "method_pack_compat": [
                "corporate", "national_inventory", "eu_policy",
            ],
            "ipcc_chapter": "Vol 5 Ch 2",
            "stream": stream,
            "climate_zone": climate_zone,
            "doc_fraction": doc,
            "doc_f_fraction": doc_f,
            "mcf": mcf,
            "assumptions": [
                "IPCC 2006 GL Vol 5 Ch 2 defaults",
                f"India climate zone: {climate_zone}",
                f"Waste stream: {stream}",
                f"DOC={doc}, DOC_F={doc_f}, MCF={mcf}",
            ],
        },
    })


def _incineration_factor(
    row: Dict[str, Any],
    meta: Dict[str, Any],
    year: int,
    country: str,
) -> Dict[str, Any]:
    stream = _slug(str(row.get("stream") or "msw"))
    kg_co2e = _safe_float(row.get("kg_co2e_per_tonne"))
    fossil_fraction = _safe_float(row.get("fossil_fraction"), 1.0)
    kg_ch4 = _safe_float(row.get("kg_ch4_per_tonne"))
    kg_n2o = _safe_float(row.get("kg_n2o_per_tonne"))
    kg_co2_fossil = _safe_float(row.get("kg_co2_fossil_per_tonne"))
    if kg_co2_fossil <= 0.0 and kg_co2e > 0.0 and fossil_fraction > 0.0:
        # Back out the fossil CO2 share if only the composite is given.
        kg_co2_fossil = kg_co2e * fossil_fraction
    if kg_co2_fossil <= 0.0 and kg_ch4 <= 0.0 and kg_n2o <= 0.0:
        return {}
    total_co2e = _ch4_n2o_to_co2e(kg_ch4, kg_n2o, kg_co2_fossil)
    if total_co2e <= 0.0:
        return {}

    illustrative = bool(row.get("illustrative_value", False))
    meta_cat = _CATEGORY_META["incineration"]
    fid = f"EF:IPCC_WASTE:{meta_cat['id_prefix']}_{stream}:{country}:{year}:v1"

    notes_parts = [
        f"IPCC 2006 Vol 5 Ch 5 incineration default for {stream} "
        f"({country}); fossil_fraction={fossil_fraction}, "
        f"kg CO2 fossil/t={kg_co2_fossil:.1f}"
    ]
    if illustrative:
        notes_parts.append(
            "illustrative_value=true; TODO: reconcile with Indian Central "
            "Pollution Control Board WTE emissions inventory"
        )

    tags = [
        "ipcc_2006", "vol5", "waste", "incineration", "open_burning",
        f"stream_{stream}", f"country_{country.lower()}", "scope3_cat5",
    ]
    if illustrative:
        tags.append("illustrative")

    return _stamp({
        "factor_id": fid,
        "fuel_type": f"waste_{stream}",
        "unit": meta_cat["unit"],
        "geography": country,
        "geography_level": GeographyLevel.COUNTRY.value,
        "vectors": {
            "CO2": total_co2e,
            "CH4": kg_ch4,
            "N2O": kg_n2o,
        },
        "gwp_100yr": {
            "gwp_set": GWPSet.IPCC_AR6_100.value,
            "CH4_gwp": _GWP_CH4,
            "N2O_gwp": _GWP_N2O,
        },
        "scope": Scope.SCOPE_3.value,
        "boundary": meta_cat["boundary"],
        "provenance": _provenance(meta, year, "incineration"),
        "valid_from": date(year, 1, 1).isoformat(),
        "valid_to": date(9999, 12, 31).isoformat(),
        "uncertainty_95ci": 0.35,
        "dqs": _dqs(illustrative),
        "license_info": _LICENSE,
        "tags": tags,
        "notes": "; ".join(notes_parts),
        "biogenic_flag": fossil_fraction < 1.0,
        "factor_family": meta_cat["family"],
        "activity_tags": [meta_cat["activity"], f"waste_{stream}"],
        "sector_tags": ["waste_management", "incineration"],
        "source_id": "ipcc_waste_vol5_in",
        "source_release": f"ipcc2006-vol5-{year}",
        "release_version": f"ipcc2006-vol5-{year}",
        "validation_flags": {
            "method_pack_compat": [
                "corporate", "national_inventory", "eu_policy",
            ],
            "ipcc_chapter": "Vol 5 Ch 5",
            "stream": stream,
            "fossil_fraction": fossil_fraction,
            "assumptions": [
                "IPCC 2006 GL Vol 5 Ch 5 incineration defaults",
                f"Waste stream: {stream}",
                f"Fossil fraction: {fossil_fraction}",
            ],
        },
    })


def _biological_factor(
    row: Dict[str, Any],
    meta: Dict[str, Any],
    year: int,
    country: str,
) -> Dict[str, Any]:
    stream = _slug(str(row.get("stream") or "compost_msw"))
    kg_ch4 = _safe_float(row.get("kg_ch4_per_tonne"))
    kg_n2o = _safe_float(row.get("kg_n2o_per_tonne"))
    if kg_ch4 <= 0.0 and kg_n2o <= 0.0:
        return {}
    co2e = _ch4_n2o_to_co2e(kg_ch4, kg_n2o)

    illustrative = bool(row.get("illustrative_value", False))
    meta_cat = _CATEGORY_META["biological"]
    fid = f"EF:IPCC_WASTE:{meta_cat['id_prefix']}_{stream}:{country}:{year}:v1"

    notes = (
        f"IPCC 2006 Vol 5 Ch 4 biological treatment default for {stream} "
        f"({country}); kg CH4/t={kg_ch4}, kg N2O/t={kg_n2o}"
    )
    if illustrative:
        notes += (
            "; illustrative_value=true; TODO: reconcile with India NATCOM-4 "
            "composting sector values"
        )

    tags = [
        "ipcc_2006", "vol5", "waste", "biological_treatment",
        "composting", f"stream_{stream}", f"country_{country.lower()}",
        "scope3_cat5",
    ]
    if illustrative:
        tags.append("illustrative")

    return _stamp({
        "factor_id": fid,
        "fuel_type": f"waste_{stream}",
        "unit": meta_cat["unit"],
        "geography": country,
        "geography_level": GeographyLevel.COUNTRY.value,
        "vectors": {"CO2": co2e, "CH4": kg_ch4, "N2O": kg_n2o},
        "gwp_100yr": {
            "gwp_set": GWPSet.IPCC_AR6_100.value,
            "CH4_gwp": _GWP_CH4,
            "N2O_gwp": _GWP_N2O,
        },
        "scope": Scope.SCOPE_3.value,
        "boundary": meta_cat["boundary"],
        "provenance": _provenance(meta, year, "biological"),
        "valid_from": date(year, 1, 1).isoformat(),
        "valid_to": date(9999, 12, 31).isoformat(),
        "uncertainty_95ci": 0.50,
        "dqs": _dqs(illustrative),
        "license_info": _LICENSE,
        "tags": tags,
        "notes": notes,
        "biogenic_flag": True,
        "factor_family": meta_cat["family"],
        "activity_tags": [meta_cat["activity"], f"waste_{stream}"],
        "sector_tags": ["waste_management", "composting"],
        "source_id": "ipcc_waste_vol5_in",
        "source_release": f"ipcc2006-vol5-{year}",
        "release_version": f"ipcc2006-vol5-{year}",
        "validation_flags": {
            "method_pack_compat": [
                "corporate", "national_inventory",
            ],
            "ipcc_chapter": "Vol 5 Ch 4",
            "stream": stream,
            "assumptions": [
                "IPCC 2006 GL Vol 5 Ch 4 biological treatment defaults",
                f"Waste stream: {stream}",
            ],
        },
    })


def _wastewater_factor(
    row: Dict[str, Any],
    meta: Dict[str, Any],
    year: int,
    country: str,
    climate_zone: str,
) -> Dict[str, Any]:
    stream = _slug(str(row.get("stream") or "domestic"))
    kg_ch4 = _safe_float(row.get("kg_ch4_per_m3"))
    kg_n2o_per_m3 = _safe_float(row.get("kg_n2o_per_m3"))
    if kg_ch4 <= 0.0 and kg_n2o_per_m3 <= 0.0:
        return {}
    co2e = _ch4_n2o_to_co2e(kg_ch4, kg_n2o_per_m3)
    mcf = _safe_float(row.get("mcf"), 0.8)
    b0 = _safe_float(row.get("b0_kg_ch4_per_kg_bod"), 0.6)

    illustrative = bool(row.get("illustrative_value", False))
    meta_cat = _CATEGORY_META["wastewater"]
    fid = f"EF:IPCC_WASTE:{meta_cat['id_prefix']}_{stream}:{country}:{year}:v1"

    notes_parts = [
        f"IPCC 2006 Vol 5 Ch 6 wastewater default for {stream} "
        f"({country}, climate={climate_zone}); MCF={mcf}, B0={b0} kg CH4/kg BOD; "
        f"kg CH4/m3={kg_ch4}"
    ]
    if illustrative:
        notes_parts.append(
            "illustrative_value=true; TODO: pin to India CPCB wastewater "
            "effluent characterisation tables"
        )

    tags = [
        "ipcc_2006", "vol5", "waste", "wastewater", f"stream_{stream}",
        f"country_{country.lower()}", f"climate_{climate_zone}",
        "scope3_cat5",
    ]
    if illustrative:
        tags.append("illustrative")

    return _stamp({
        "factor_id": fid,
        "fuel_type": f"wastewater_{stream}",
        "unit": meta_cat["unit"],
        "geography": country,
        "geography_level": GeographyLevel.COUNTRY.value,
        "vectors": {
            "CO2": co2e,
            "CH4": kg_ch4,
            "N2O": kg_n2o_per_m3,
        },
        "gwp_100yr": {
            "gwp_set": GWPSet.IPCC_AR6_100.value,
            "CH4_gwp": _GWP_CH4,
            "N2O_gwp": _GWP_N2O,
        },
        "scope": Scope.SCOPE_3.value,
        "boundary": meta_cat["boundary"],
        "provenance": _provenance(meta, year, "wastewater"),
        "valid_from": date(year, 1, 1).isoformat(),
        "valid_to": date(9999, 12, 31).isoformat(),
        "uncertainty_95ci": 0.45,
        "dqs": _dqs(illustrative),
        "license_info": _LICENSE,
        "tags": tags,
        "notes": "; ".join(notes_parts),
        "biogenic_flag": True,
        "factor_family": meta_cat["family"],
        "activity_tags": [meta_cat["activity"], f"wastewater_{stream}"],
        "sector_tags": ["waste_management", "wastewater_treatment"],
        "source_id": "ipcc_waste_vol5_in",
        "source_release": f"ipcc2006-vol5-{year}",
        "release_version": f"ipcc2006-vol5-{year}",
        "validation_flags": {
            "method_pack_compat": [
                "corporate", "national_inventory", "eu_policy",
            ],
            "ipcc_chapter": "Vol 5 Ch 6",
            "stream": stream,
            "climate_zone": climate_zone,
            "mcf": mcf,
            "b0_kg_ch4_per_kg_bod": b0,
            "assumptions": [
                "IPCC 2006 GL Vol 5 Ch 6 wastewater defaults",
                f"India climate zone: {climate_zone}",
                f"Stream: {stream}",
                f"MCF={mcf}, B0={b0}",
            ],
        },
    })


def _iter_factors(
    data: Dict[str, Any],
    meta: Dict[str, Any],
    year: int,
    country: str,
    climate_zone: str,
) -> Iterator[Dict[str, Any]]:
    categories = data.get("categories") or {}
    if not isinstance(categories, dict):
        logger.warning(
            "IPCC Waste: categories is not a dict; got %s", type(categories)
        )
        return

    for cat, rows in categories.items():
        if not isinstance(rows, list):
            continue
        cat_slug = _slug(cat)
        for row in rows:
            if not isinstance(row, dict):
                continue
            if cat_slug == "swds":
                f = _swds_factor(row, meta, year, country, climate_zone)
            elif cat_slug == "incineration":
                f = _incineration_factor(row, meta, year, country)
            elif cat_slug == "biological":
                f = _biological_factor(row, meta, year, country)
            elif cat_slug == "wastewater":
                f = _wastewater_factor(
                    row, meta, year, country, climate_zone
                )
            else:
                continue
            if f:
                yield f


def parse_ipcc_waste_vol5_in(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse IPCC 2006 Vol 5 India-parameterised waste defaults.

    Returns ~40-80 factors covering SWDS, incineration, biological
    treatment, and wastewater categories with India (tropical wet)
    climate-zone parameters.
    """
    meta = data.get("metadata") or {}
    try:
        year = int(meta.get("vintage_year") or 2006)
    except (TypeError, ValueError):
        year = 2006
    country = str(meta.get("country") or "IN").strip().upper()
    climate_zone = _slug(str(meta.get("climate_zone") or "tropical_wet"))

    factors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for f in _iter_factors(data, meta, year, country, climate_zone):
        fid = f.get("factor_id", "")
        if fid and fid not in seen:
            seen.add(fid)
            factors.append(f)

    logger.info(
        "IPCC Waste Vol5 (%s) parser produced %d factors (vintage=%d)",
        country, len(factors), year,
    )
    return factors


__all__ = ["parse_ipcc_waste_vol5_in"]
