# -*- coding: utf-8 -*-
"""
IPCC refrigerant GWP promotion parser (Wave 4-B catalog expansion).

Promotes HFC / PFC / SF6 / NF3 100-year GWP values from IPCC AR5 and
AR6 assessment reports into the GreenLang default catalog as
**certified** records (the legacy ``ipcc_defaults.py`` emits them as
``preview``, and the gold-set ceiling-bound entries need concrete
certified rows to resolve).

Factor IDs match the gold-set pattern exactly::

    EF:IPCC:r_<gas_slug>:GLOBAL:ipcc_ar5_100:v1
    EF:IPCC:r_<gas_slug>:GLOBAL:ipcc_ar6_100:v1

Two records per gas — one per GWP basis — so the resolver can pick the
right set depending on the reporter's GWP assessment choice.

Gas coverage (12 common refrigerants + 4 high-GWP utility gases):
R-22, R-32, R-134a, R-410A, R-404A, R-507, R-1234yf, R-1234ze, R-452A,
R-513A, R-448A, R-407F, R-23, SF6, NF3, and a CF4 / C2F6 PFC pair.

GWP values are sourced from the IPCC Assessment Reports (AR5 WG1 Ch 8,
AR6 WG1 Ch 7); attribution matches the existing IPCC source. The
``factor_family`` is stamped as ``refrigerant_gwp`` so the Wave 3
family_inference resolver step can match.

Seed input: ``catalog_seed/_inputs/ipcc_refrigerants.json`` (GWP values
only — no computed values; just the published IPCC GWP numbers).
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

_COMPLIANCE = ["GHG_Protocol", "IPCC_2006", "ISO_14064", "IPCC_AR5", "IPCC_AR6"]


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


def _provenance(gwp_basis: str, assessment_year: int) -> Dict[str, Any]:
    ar_label = "AR5" if gwp_basis == "ipcc_ar5_100" else "AR6"
    return {
        "source_org": "IPCC",
        "source_publication": (
            f"IPCC {ar_label} WGI — 100-year GWP values for halogenated gases"
        ),
        "source_year": assessment_year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": "https://www.ipcc.ch/",
        "version": ar_label.lower() + "_100yr",
    }


def _iter_refrigerants(
    rows: List[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        gas_id = str(row.get("gas_id") or row.get("slug") or "").strip().lower()
        if not gas_id:
            continue
        display = str(row.get("display_name") or gas_id.upper())
        formula = str(row.get("formula") or "")
        gwp_ar5 = _safe_float(row.get("gwp_ar5_100"))
        gwp_ar6 = _safe_float(row.get("gwp_ar6_100"))
        kind = str(row.get("kind") or "hfc")
        notes_extra = str(row.get("notes") or "")

        # Emit TWO records per gas — AR5 and AR6 GWP bases.
        for gwp_basis, gwp_val, ar_label, assessment_year in (
            ("ipcc_ar5_100", gwp_ar5, "AR5", 2013),
            ("ipcc_ar6_100", gwp_ar6, "AR6", 2021),
        ):
            if gwp_val <= 0:
                continue

            fid = f"EF:IPCC:{gas_id}:GLOBAL:{gwp_basis}:v1"

            notes_bits = [
                f"{display} ({formula}) — 100-year GWP per IPCC {ar_label}: "
                f"{gwp_val} kgCO2e/kg",
                f"{kind.upper()}; fugitive emission accounting (GHG Protocol "
                f"Scope 1, refrigerant leak/service/EOL disposal)",
            ]
            if kind == "hcfc":
                notes_bits.append("HCFC, phase-out under Montreal Protocol")
            if notes_extra:
                notes_bits.append(notes_extra)

            tags = [
                "ipcc", "refrigerant", f"gwp_basis_{gwp_basis}",
                gas_id, kind,
            ]

            # For refrigerants we use a direct-CO2e representation: the
            # vector's CO2 slot carries the GWP number (kgCO2e/kg).
            # Callers apply quantity [kg] * CO2 [kgCO2e/kg] = kgCO2e. CH4 and
            # N2O are zero since the gas itself is the forcing agent.
            yield _stamp({
                "factor_id": fid,
                "fuel_type": f"refrigerant_{gas_id}",
                "unit": "kg",
                "geography": "GLOBAL",
                "geography_level": GeographyLevel.GLOBAL.value,
                "vectors": {"CO2": gwp_val, "CH4": 0.0, "N2O": 0.0},
                "gwp_100yr": {
                    "gwp_set": (
                        GWPSet.IPCC_AR5_100.value
                        if gwp_basis == "ipcc_ar5_100"
                        else GWPSet.IPCC_AR6_100.value
                    ),
                    "CH4_gwp": 28 if gwp_basis == "ipcc_ar5_100" else 27,
                    "N2O_gwp": 265 if gwp_basis == "ipcc_ar5_100" else 273,
                },
                "scope": Scope.SCOPE_1.value,
                "boundary": Boundary.FUGITIVE.value
                    if hasattr(Boundary, "FUGITIVE")
                    else Boundary.COMBUSTION.value,
                "provenance": _provenance(gwp_basis, assessment_year),
                "valid_from": date(assessment_year, 1, 1).isoformat(),
                "uncertainty_95ci": 0.10,
                "dqs": {
                    "temporal": 5,
                    "geographical": 5,
                    "technological": 5,
                    "representativeness": 5,
                    "methodological": 5,
                },
                "license_info": _LICENSE,
                "tags": tags,
                "notes": "; ".join(notes_bits),
                "factor_family": FactorFamily.REFRIGERANT_GWP.value,
                "activity_tags": ["fugitive_emission", "refrigerant", "scope1"],
                "sector_tags": ["refrigeration", "hvac"],
                "source_id": "ipcc_refrigerants_promoted",
                "source_release": gwp_basis,
                "release_version": f"ipcc-{ar_label.lower()}",
            })


def parse_ipcc_refrigerants(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse IPCC refrigerant GWP payload -> dicts.

    Args:
        data: Dict with ``metadata`` and ``gases`` (list of refrigerant
            rows, each carrying ``gas_id``, ``gwp_ar5_100``,
            ``gwp_ar6_100``).

    Returns:
        List of factor dicts (2 records per gas — AR5 + AR6).
    """
    rows = data.get("gases") or []
    if not isinstance(rows, list):
        logger.warning("IPCC refrigerants: gases is not a list")
        return []

    factors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for f in _iter_refrigerants(rows):
        fid = f.get("factor_id", "")
        if fid and fid not in seen:
            seen.add(fid)
            factors.append(f)

    logger.info(
        "IPCC refrigerants promoted: %d factors across %d gases",
        len(factors), len(rows),
    )
    return factors


__all__ = ["parse_ipcc_refrigerants"]
