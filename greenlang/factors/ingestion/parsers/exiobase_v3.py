# -*- coding: utf-8 -*-
"""
EXIOBASE v3 multi-regional EE-IO parser — Wave 5 catalog expansion.

Parses the per-region per-sector GHG-intensity vectors extracted from
the EXIOBASE v3.8.x environmentally-extended multi-regional input-output
model. The values are only used as purchased-goods proxies (Scope-3
Cat.1 spend-based screening) for EU27 + rest-of-world regions.

Source: EXIOBASE Consortium (TU Wien / NTNU / UTwente), CC-BY-4.0 for
the v3.8 base release (moved to CC-BY from earlier academic-use-only
license, see https://www.exiobase.eu/index.php/data-download/).

Factor IDs follow the gold-pattern shape:
    ``EF:EXIOBASE:<sector_code>:<region>:<vintage>:v1``

for example::

    EF:EXIOBASE:1712:EU:v3.8:v1
    EF:EXIOBASE:J62:EU:v3.8:v1
    EF:EXIOBASE:J63:IN:v3.8:v1

Method-pack compatibility: ``corporate`` (Scope-3 Cat.1 spend proxy),
``finance_proxy``. Not for product-carbon Tier-1 use.

Expected payload::

    {
      "metadata": {"source": "EXIOBASE", "version": "3.8.2", "vintage_year": 2022},
      "sectors": [
        {
          "sector_code": "1712",                  # NACE-like or EXIOBASE native
          "sector_name": "Pulp, paper and paper products",
          "region": "EU",                          # ISO-2 or aggregate (EU, EU27, RoW)
          "kg_co2e_per_eur": 0.65,
          "kg_co2e_per_usd": 0.70,
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
    "license": "CC-BY-4.0",
    "redistribution_allowed": True,
    "commercial_use_allowed": True,
    "attribution_required": True,
}

_COMPLIANCE = ["GHG_Protocol", "GHG_Protocol_Scope3", "ESRS_E1"]

# EU27 member-state ISO-2 codes; anything not in this set that isn't a
# named aggregate gets mapped through as-is.
_EU27 = frozenset({
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL",
    "PT", "RO", "SE", "SI", "SK",
})

_AGGREGATE_REGIONS = frozenset({"EU", "EU27", "ROW", "WORLD", "GLOBAL"})


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
        "source_org": "EXIOBASE Consortium",
        "source_publication": str(
            meta.get(
                "source_publication",
                f"EXIOBASE v{version} multi-regional EE-IO",
            )
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get("source_url", "https://www.exiobase.eu/")
        ),
        "version": f"v{version}",
    }


def _dqs(illustrative: bool) -> Dict[str, int]:
    return {
        "temporal": 3,
        "geographical": 4,
        "technological": 3,
        "representativeness": 3,
        "methodological": 3 if illustrative else 4,
    }


def _iter_sectors(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    year: int,
    version: str,
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        sector_code = str(row.get("sector_code") or "").strip()
        if not sector_code:
            continue
        region_raw = str(row.get("region") or "").strip().upper()
        if not region_raw:
            continue
        name = str(row.get("sector_name") or sector_code)
        kg_eur = _safe_float(row.get("kg_co2e_per_eur"))
        kg_usd = _safe_float(row.get("kg_co2e_per_usd"))
        illustrative = bool(row.get("illustrative_value", False))

        # Pick the primary denominator. EUR first for EU/EU27; USD for
        # US/row. Rows may carry both — we emit one factor per currency
        # keyed off the slug so downstream picks the nearest match.
        currency_specs: List[tuple[str, float]] = []
        if kg_eur > 0.0:
            currency_specs.append(("EUR", kg_eur))
        if kg_usd > 0.0:
            currency_specs.append(("USD", kg_usd))
        if not currency_specs:
            continue

        # Region normalisation: pass through ISO-2 / aggregates.
        region = region_raw
        if region not in _EU27 and region not in _AGGREGATE_REGIONS:
            # Treat anything else as-is (ISO-2 for non-EU countries).
            pass

        # Version tag on the factor id; keep raw "3.8" without the "v"
        # prefix so it reads ``EF:EXIOBASE:1712:EU:v3.8:v1``.
        version_tag = f"v{version}"

        for currency, val in currency_specs:
            fid = f"EF:EXIOBASE:{sector_code}:{region}:{version_tag}:v1"
            if currency == "USD":
                # Disambiguate the USD-denominated variant.
                fid = (
                    f"EF:EXIOBASE:{sector_code}:{region}:{version_tag}:usd:v1"
                )

            notes = (
                f"EXIOBASE v{version} spend-intensity for {name} "
                f"({currency})"
            )
            if illustrative:
                notes += (
                    "; illustrative_value=true; TODO: replace with "
                    "EXIOBASE v3.8 full-bulk import before GA"
                )

            tags = [
                "exiobase", "spend_based", "scope3_cat1",
                f"sector_{sector_code}", f"region_{region}", currency.lower(),
            ]
            if region in _EU27 or region in _AGGREGATE_REGIONS:
                tags.append("eu27")
            if illustrative:
                tags.append("illustrative")

            geography_level = (
                GeographyLevel.GLOBAL.value
                if region in {"WORLD", "GLOBAL", "ROW"}
                else GeographyLevel.COUNTRY.value
            )

            # Store the EU-aggregate rows under country code "EU" (non-ISO
            # but the resolver normalises) so the gold pattern
            # ``EF:EXIOBASE:1712:EU:.*`` matches.
            geo = region

            yield _stamp({
                "factor_id": fid,
                "fuel_type": _slug(name)[:40],
                "unit": currency.lower(),
                "geography": geo,
                "geography_level": geography_level,
                "vectors": {"CO2": val, "CH4": 0.0, "N2O": 0.0},
                "gwp_100yr": {
                    "gwp_set": GWPSet.IPCC_AR6_100.value,
                    "CH4_gwp": 28,
                    "N2O_gwp": 273,
                },
                "scope": Scope.SCOPE_3.value,
                "boundary": Boundary.CRADLE_TO_GATE.value,
                "provenance": _provenance(meta, year, version),
                "valid_from": date(year, 1, 1).isoformat(),
                "valid_to": date(9999, 12, 31).isoformat(),
                "uncertainty_95ci": 0.35,
                "dqs": _dqs(illustrative),
                "license_info": _LICENSE,
                "tags": tags,
                "notes": notes,
                "factor_family": FactorFamily.FINANCE_PROXY.value,
                "activity_tags": [
                    "spend_based",
                    "purchased_goods_and_services",
                    f"sector_{sector_code}",
                ],
                "sector_tags": [f"exiobase_{sector_code}"],
                "source_id": "exiobase_v3",
                "source_release": version_tag,
                "release_version": f"exiobase-{version_tag}",
                "validation_flags": {
                    "method_pack_compat": ["corporate", "finance_proxy"],
                    "exiobase_sector_code": sector_code,
                    "exiobase_region": region,
                    "assumptions": [
                        f"EXIOBASE v{version} EE-IO model",
                        f"Sector {sector_code} ({name})",
                        f"Region {region}",
                        f"{currency} basis",
                    ],
                },
            })


def parse_exiobase_v3(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse EXIOBASE v3 payload into normalized factor dicts.

    Returns a flat list — one per (sector_code, region, currency). EU27
    member states + aggregates (EU, EU27, RoW) are the focus set.
    """
    meta = data.get("metadata") or {}
    version = str(meta.get("version") or "3.8.2")
    try:
        year = int(meta.get("vintage_year") or 2022)
    except (TypeError, ValueError):
        year = 2022

    rows = data.get("sectors") or []
    if not isinstance(rows, list):
        logger.warning("EXIOBASE: sectors is not a list; got %s", type(rows))
        return []

    factors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for f in _iter_sectors(rows, meta, year, version):
        fid = f.get("factor_id", "")
        if fid and fid not in seen:
            seen.add(fid)
            factors.append(f)

    logger.info(
        "EXIOBASE v3 parser produced %d factors (version=%s, vintage=%d)",
        len(factors), version, year,
    )
    return factors


__all__ = ["parse_exiobase_v3"]
