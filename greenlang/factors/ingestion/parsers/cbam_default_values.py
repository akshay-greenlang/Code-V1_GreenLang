# -*- coding: utf-8 -*-
"""
CBAM default values (flat sector-level) parser — Wave 5 catalog expansion.

The existing ``cbam_full`` parser emits deep per-category ids
(``EF:CBAM:iron_steel_crude_steel_bof:CN:2024:v1``). The gold-eval gate
expects the flat *sector-level* rollup shape that the resolver's
ID-pattern matcher can hit directly:

    ``EF:CBAM:steel:CN:2024:v1``
    ``EF:CBAM:aluminium:IN:2024:v1``
    ``EF:CBAM:cement:TR:2024:v1``
    ``EF:CBAM:fertilizer:RU:2024:v1``
    ``EF:CBAM:hydrogen:SA:2024:v1``
    ``EF:CBAM:electricity:RS:2024:v1``

Source: EU Commission Implementing Regulation (EU) 2023/1773 Annex IV —
default embedded-emission values for CBAM goods (steel, aluminium,
cement, fertilizer, hydrogen, electricity, iron).

License: ``public_eu`` / EU-Publication (Decision 2011/833/EU). Values
are factual regulatory defaults; redistribution is allowed with
attribution to the European Commission.

Method-pack compatibility: ``eu_policy`` (primary), ``product_carbon``
(for Scope-3 upstream embedded-emissions use).

Expected payload::

    {
      "metadata": {"source": "EU Commission", "version": "2024"},
      "sectors": {
        "steel":       {"by_country": {"CN": {"direct": 2.20, "indirect": 0.60}, ...}},
        "aluminium":   {"by_country": {...}},
        "cement":      {"by_country": {...}},
        "fertilizer":  {"by_country": {...}},
        "hydrogen":    {"by_country": {...}},
        "electricity": {"by_country": {...}},
        "iron":        {"by_country": {...}}
      }
    }
"""
from __future__ import annotations

import logging
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

_COMPLIANCE = ["EU_CBAM", "GHG_Protocol", "IPCC_2006"]

# CBAM 7-sector rollup. Each sector carries a canonical "unit" used in
# factor_name. The parser always emits per kg-product; the sector-level
# value is the country average of the per-product defaults.
_DEFAULT_UNIT = "kg_product"


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
        "source_org": "EU Commission",
        "source_publication": str(
            meta.get(
                "source_publication",
                "CBAM Implementing Regulation (EU) 2023/1773 Annex IV "
                "default embedded-emission values",
            )
        ),
        "source_year": year,
        "methodology": Methodology.IPCC_TIER_1.value,
        "source_url": str(
            meta.get(
                "source_url",
                "https://taxation-customs.ec.europa.eu/"
                "carbon-border-adjustment-mechanism_en",
            )
        ),
        "version": str(meta.get("version", f"{year}")),
    }


def _iter_sector_entries(
    sector: str,
    sector_data: Dict[str, Any],
    meta: Dict[str, Any],
    year: int,
) -> Iterator[Dict[str, Any]]:
    """Yield one factor per (sector, country) pair at sector-rollup level."""
    by_country = sector_data.get("by_country") or {}
    sector_slug = sector.strip().lower()
    unit = str(sector_data.get("unit") or _DEFAULT_UNIT)

    for country, row in by_country.items():
        if not isinstance(row, dict):
            continue
        country = str(country).upper()
        if country == "DEFAULT":
            # Skip DEFAULT — the gold set expects real ISO-2 codes. A
            # default fallback is still useful but lives under a
            # dedicated `ROW` code if the seed provides one.
            continue
        direct = _safe_float(row.get("direct") or row.get("direct_emissions_factor"))
        indirect = _safe_float(
            row.get("indirect") or row.get("indirect_emissions_factor")
        )
        total = direct + indirect
        if total <= 0.0:
            continue

        illustrative = bool(row.get("illustrative_value", False))
        fid = f"EF:CBAM:{sector_slug}:{country}:{year}:v1"

        notes_bits = [
            f"CBAM Annex IV default — {sector_slug} from {country}, "
            f"direct={direct} + indirect={indirect} = {total} tCO2e/t-product"
        ]
        if illustrative:
            notes_bits.append(
                "illustrative_value=true; TODO: reconcile with official "
                "EU Commission Annex IV release before GA"
            )

        tags = [
            "cbam",
            "cbam_default",
            "cbam_2026",
            f"sector_{sector_slug}",
            sector_slug,
            "embedded_emissions",
        ]
        if illustrative:
            tags.append("illustrative")

        assumptions = [
            f"CBAM default value for {sector_slug}",
            f"CBAM default direct {direct} + indirect {indirect}",
            "embedded emissions",
        ]
        if row.get("cn_code"):
            assumptions.append(f"CN code {row['cn_code']}")

        # Values on seed are in tCO2e/t-product — convert to kg/kg-product
        # (dimensionally identical) and store in CO2 vector; CH4/N2O are
        # not broken out at the regulatory-default granularity.
        yield _stamp({
            "factor_id": fid,
            "fuel_type": f"cbam_{sector_slug}",
            "unit": unit,
            "geography": country,
            "geography_level": GeographyLevel.COUNTRY.value,
            "vectors": {"CO2": total, "CH4": 0.0, "N2O": 0.0},
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
            "uncertainty_95ci": 0.15,
            "dqs": {
                "temporal": 4,
                "geographical": 4,
                "technological": 3,
                "representativeness": 4,
                "methodological": 4,
            },
            "license_info": _LICENSE,
            "tags": tags,
            "notes": "; ".join(notes_bits),
            "factor_family": FactorFamily.MATERIAL_EMBODIED.value,
            "activity_tags": [
                "embedded_emissions",
                "cbam_goods",
                f"cbam_{sector_slug}",
            ],
            "sector_tags": [f"cbam_{sector_slug}"],
            "source_id": "cbam_default_values",
            "source_release": str(meta.get("version", str(year))),
            "release_version": f"cbam-annex-iv-{year}",
            "validation_flags": {
                "method_pack_compat": ["eu_policy", "product_carbon"],
                "cbam_default_source": "EU Commission Implementing Regulation 2023/1773 Annex IV",
                "assumptions": assumptions,
            },
        })


def parse_cbam_default_values(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse CBAM flat per-sector default values.

    Emits factor_ids of the gold-pattern shape
    ``EF:CBAM:<sector>:<country>:<year>:v1``. ~40-80 records expected
    (7 sectors x ~6-12 origin countries).
    """
    meta = data.get("metadata") or {}
    year = 2024
    try:
        year = int(
            str(meta.get("version") or meta.get("year") or "2024")
            .split(".")[0]
        )
    except (TypeError, ValueError):
        year = 2024

    sectors = data.get("sectors") or {}
    factors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for sector, sector_data in sectors.items():
        if not isinstance(sector_data, dict):
            continue
        for f in _iter_sector_entries(sector, sector_data, meta, year):
            fid = f.get("factor_id", "")
            if fid and fid not in seen:
                seen.add(fid)
                factors.append(f)

    logger.info(
        "CBAM default values parser produced %d factors from %d sectors (year=%d)",
        len(factors), len(sectors), year,
    )
    return factors


__all__ = ["parse_cbam_default_values"]
