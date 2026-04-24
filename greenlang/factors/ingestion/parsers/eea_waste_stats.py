# -*- coding: utf-8 -*-
"""
EEA European waste statistics parser — Wave 5 catalog expansion.

Parses per-stream emission intensities for waste treatment (landfill,
incineration, recycling, composting, anaerobic digestion) across EU27
member states from the European Environment Agency (EEA) waste
statistics series.

Source: EEA — European waste statistics indicators combined with the
JRC per-treatment CO2e factors published under the EU Waste Framework
Directive evaluation reports.

License: ``public_eu`` / EU-Publication (Decision 2011/833/EU).
Redistribution allowed with attribution.

Factor IDs follow the gold-pattern shape:
    ``EF:EEA:waste_<treatment>_<material>:<country>:<vintage>:v1``

for example::

    EF:EEA:waste_recycling_plastic:EU:2022:v1
    EF:EEA:waste_incineration_msw:EU:2022:v1
    EF:EEA:wastewater_aerobic:EU:2022:v1

Method-pack compatibility: ``corporate`` (Scope-3 Cat.5), ``eu_policy``.

Expected payload::

    {
      "metadata": {"source": "EEA", "vintage_year": 2022, ...},
      "streams": [
        {
          "treatment": "recycling",
          "material": "plastic",
          "country": "EU",            # ISO-2 or "EU" aggregate
          "kg_co2e_per_tonne": 320.0,
          "biogenic": false,
          "illustrative_value": true
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
    "license": "EU-Publication",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "GHG_Protocol_Scope3", "EU_Waste_Framework_Directive"]


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


def _provenance(meta: Dict[str, Any], year: int) -> Dict[str, Any]:
    return {
        "source_org": "European Environment Agency (EEA)",
        "source_publication": str(
            meta.get(
                "source_publication",
                f"EEA European waste statistics — treatment GHG intensity "
                f"({year})",
            )
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get(
                "source_url",
                "https://www.eea.europa.eu/en/topics/in-depth/waste-and-recycling",
            )
        ),
        "version": f"eea-waste-{year}",
    }


def _dqs(illustrative: bool) -> Dict[str, int]:
    return {
        "temporal": 3,
        "geographical": 4,
        "technological": 3,
        "representativeness": 3,
        "methodological": 3 if illustrative else 4,
    }


def _iter_streams(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    year: int,
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        treatment = _slug(str(row.get("treatment") or "unknown"))
        if treatment == "unknown":
            continue
        material = _slug(str(row.get("material") or ""))
        country = str(row.get("country") or "EU").strip().upper()
        kg_tonne = _safe_float(row.get("kg_co2e_per_tonne"))
        unit = str(row.get("unit") or "tonne")
        illustrative = bool(row.get("illustrative_value", False))
        biogenic = bool(row.get("biogenic", False))

        if kg_tonne == 0.0 and not row.get("allow_zero", False):
            continue

        # Wastewater is treated distinctly — id uses ``wastewater_<kind>``
        # rather than ``waste_<treatment>_<material>``.
        if treatment in {"aerobic", "anaerobic", "wastewater"}:
            mat_token = material or "generic"
            kind = (
                mat_token if treatment == "wastewater"
                else f"wastewater_{treatment}"
            )
            fid_body = f"wastewater_{treatment}"
            unit_out = "m3"
        else:
            if not material:
                material = "generic"
            fid_body = f"waste_{treatment}_{material}"
            unit_out = "tonne" if unit.lower() in {"tonne", "t", "mt"} else unit

        fid = f"EF:EEA:{fid_body}:{country}:{year}:v1"

        notes = (
            f"EEA waste-stream factor: {treatment} {material} "
            f"({country}, {year})"
        )
        if illustrative:
            notes += (
                "; illustrative_value=true; TODO: reconcile with the "
                "EEA dashboard XLSX release"
            )
        if row.get("notes"):
            notes += f"; {row['notes']}"

        tags = [
            "eea", "waste", f"treatment_{treatment}",
            f"material_{material or 'generic'}", "scope3_cat5", "eu27",
        ]
        if illustrative:
            tags.append("illustrative")

        yield _stamp({
            "factor_id": fid,
            "fuel_type": f"waste_{material or 'generic'}",
            "unit": unit_out,
            "geography": country,
            "geography_level": (
                GeographyLevel.COUNTRY.value
                if country != "EU"
                else GeographyLevel.GLOBAL.value
            ),
            "vectors": {"CO2": kg_tonne, "CH4": 0.0, "N2O": 0.0},
            "gwp_100yr": {
                "gwp_set": GWPSet.IPCC_AR6_100.value,
                "CH4_gwp": 28,
                "N2O_gwp": 273,
            },
            "scope": Scope.SCOPE_3.value,
            "boundary": Boundary.CRADLE_TO_GATE.value,
            "provenance": _provenance(meta, year),
            "valid_from": date(year, 1, 1).isoformat(),
            "valid_to": date(9999, 12, 31).isoformat(),
            "uncertainty_95ci": 0.30,
            "dqs": _dqs(illustrative),
            "license_info": _LICENSE,
            "tags": tags,
            "notes": notes,
            "biogenic_flag": biogenic,
            "factor_family": FactorFamily.WASTE_TREATMENT.value,
            "activity_tags": [
                f"waste_{treatment}",
                "end_of_life",
            ],
            "sector_tags": ["waste_management"],
            "source_id": "eea_waste_stats",
            "source_release": f"eea-waste-{year}",
            "release_version": f"eea-waste-{year}",
            "validation_flags": {
                "method_pack_compat": ["corporate", "eu_policy"],
                "eea_treatment": treatment,
                "eea_material": material or "generic",
                "assumptions": [
                    f"EEA European waste statistics {year}",
                    f"Treatment: {treatment}",
                    f"Material: {material or 'generic'}",
                    f"Country: {country}",
                ],
            },
        })


def parse_eea_waste_stats(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse EEA waste statistics payload into factor dicts."""
    meta = data.get("metadata") or {}
    try:
        year = int(meta.get("vintage_year") or 2022)
    except (TypeError, ValueError):
        year = 2022

    rows = data.get("streams") or []
    if not isinstance(rows, list):
        logger.warning("EEA waste: streams is not a list; got %s", type(rows))
        return []

    factors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for f in _iter_streams(rows, meta, year):
        fid = f.get("factor_id", "")
        if fid and fid not in seen:
            seen.add(fid)
            factors.append(f)

    logger.info(
        "EEA waste-stats parser produced %d factors (vintage=%d)",
        len(factors), year,
    )
    return factors


__all__ = ["parse_eea_waste_stats"]
