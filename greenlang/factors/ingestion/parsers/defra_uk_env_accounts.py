# -*- coding: utf-8 -*-
"""
DEFRA / ONS UK Environmental Accounts (atmospheric emissions by industry)
parser — Wave 5 catalog expansion.

Parses per-SIC-2007 GHG intensity values from the UK Environmental
Accounts (atmospheric emissions by industry), published jointly by the
Office for National Statistics (ONS) and DEFRA. Closes the UK
purchased-goods-proxy gap identified in Wave 4 gold-eval analysis.

Source: UK Office for National Statistics / DEFRA, Environmental
Accounts — "Atmospheric emissions: greenhouse gases by industry and
gas" series.

License: UK Open Government Licence v3.0 (``uk_open_government``).
Redistribution allowed with attribution; values are factual regulatory
statistics.

Factor IDs follow the gold-pattern shape:
    ``EF:DEFRA_IO:<sic_code>:UK:<vintage>:v1``

for example::

    EF:DEFRA_IO:69:UK:2022:v1      # Legal & accounting services
    EF:DEFRA_IO:42:UK:2022:v1      # Civil engineering
    EF:DEFRA_IO:46:UK:2022:v1      # Wholesale trade

Method-pack compatibility: ``corporate`` (Scope-3 Cat.1 spend proxy),
``finance_proxy``.

Expected payload::

    {
      "metadata": {"source": "ONS / DEFRA", "vintage_year": 2022, ...},
      "sectors": [
        {
          "sic_code": "69",
          "sic_name": "Legal and accounting activities",
          "kg_co2e_per_gbp": 0.05,
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
    "license": "OGL-UK-v3",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "GHG_Protocol_Scope3", "UK_SECR"]


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
        "source_org": "UK Office for National Statistics / DEFRA",
        "source_publication": str(
            meta.get(
                "source_publication",
                "UK Environmental Accounts — atmospheric emissions by "
                "industry (SIC 2007)",
            )
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get(
                "source_url",
                "https://www.ons.gov.uk/economy/environmentalaccounts",
            )
        ),
        "version": f"ea-{year}",
    }


def _dqs(illustrative: bool) -> Dict[str, int]:
    return {
        "temporal": 4,
        "geographical": 5,
        "technological": 3,
        "representativeness": 4,
        "methodological": 3 if illustrative else 4,
    }


def _iter_sectors(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    year: int,
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        sic_code = str(row.get("sic_code") or "").strip()
        if not sic_code:
            continue
        name = str(row.get("sic_name") or sic_code)
        kg_gbp = _safe_float(row.get("kg_co2e_per_gbp"))
        if kg_gbp <= 0.0:
            continue
        illustrative = bool(row.get("illustrative_value", False))

        fid = f"EF:DEFRA_IO:{sic_code}:UK:{year}:v1"

        notes = (
            f"ONS/DEFRA UK Environmental Accounts {year} spend-intensity "
            f"for SIC {sic_code} — {name}"
        )
        if illustrative:
            notes += (
                "; illustrative_value=true; TODO: pin to the ONS XLSX "
                "publication before GA"
            )
        if row.get("notes"):
            notes += f"; {row['notes']}"

        tags = [
            "defra_io", "ons", "uk", "spend_based", "scope3_cat1",
            f"sic_{sic_code}", "gbp",
        ]
        if illustrative:
            tags.append("illustrative")

        yield _stamp({
            "factor_id": fid,
            "fuel_type": _slug(name)[:40],
            "unit": "gbp",
            "geography": "GB",
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": kg_gbp, "CH4": 0.0, "N2O": 0.0},
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
            "uncertainty_95ci": 0.25,
            "dqs": _dqs(illustrative),
            "license_info": _LICENSE,
            "tags": tags,
            "notes": notes,
            "factor_family": FactorFamily.FINANCE_PROXY.value,
            "activity_tags": [
                "spend_based",
                "purchased_goods_and_services",
                f"sic_{sic_code}",
            ],
            "sector_tags": [f"sic_{sic_code}"],
            "source_id": "defra_uk_env_accounts",
            "source_release": f"ea-{year}",
            "release_version": f"defra-ea-{year}",
            "validation_flags": {
                "method_pack_compat": ["corporate", "finance_proxy"],
                "sic_code": sic_code,
                "sic_name": name,
                "assumptions": [
                    f"ONS UK Environmental Accounts {year}",
                    f"SIC 2007 code {sic_code} ({name})",
                    "GBP spend basis, UK purchased-goods proxy",
                ],
            },
        })


def parse_defra_uk_env_accounts(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse UK Environmental Accounts IO payload into factor dicts.

    Returns one factor per SIC 2007 2-digit code (roughly 90-110 sectors
    depending on the vintage).
    """
    meta = data.get("metadata") or {}
    try:
        year = int(meta.get("vintage_year") or 2022)
    except (TypeError, ValueError):
        year = 2022

    rows = data.get("sectors") or []
    if not isinstance(rows, list):
        logger.warning(
            "DEFRA UK Env Accounts: sectors is not a list; got %s", type(rows)
        )
        return []

    factors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for f in _iter_sectors(rows, meta, year):
        fid = f.get("factor_id", "")
        if fid and fid not in seen:
            seen.add(fid)
            factors.append(f)

    logger.info(
        "DEFRA UK Env Accounts parser produced %d factors (vintage=%d)",
        len(factors), year,
    )
    return factors


__all__ = ["parse_defra_uk_env_accounts"]
