# -*- coding: utf-8 -*-
"""Spend-category mapper — routes a spend line to a factor family + scope.

Used by the Scope 3 Corporate Pack (Category 1 — Purchased Goods &
Services) and the PCAF Finance Proxy Pack.  Input is a spend category
string (e.g. "IT services — cloud compute", "Office supplies") plus an
optional amount. Output is a factor family + suggested method profile.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from greenlang.factors.mapping.base import (
    BaseMapping,
    MappingConfidence,
    MappingResult,
)

SPEND_TAXONOMY = {
    "purchased_electricity": {
        "synonyms": ["electricity", "utility bills electricity", "power bills"],
        "meta": {
            "factor_family": "grid_intensity",
            "method_profile": "corporate_scope2_location_based",
            "scope": "2",
        },
    },
    "purchased_natural_gas": {
        "synonyms": ["natural gas bill", "gas utility", "pipeline gas"],
        "meta": {
            "factor_family": "emissions",
            "method_profile": "corporate_scope1",
            "scope": "1",
        },
    },
    "road_freight": {
        "synonyms": ["truck freight", "trucking", "trucking services", "road transport spend"],
        "meta": {
            "factor_family": "transport_lane",
            "method_profile": "freight_iso_14083",
            "scope": "3",
        },
    },
    "rail_freight": {
        "synonyms": ["rail freight spend", "train freight"],
        "meta": {
            "factor_family": "transport_lane",
            "method_profile": "freight_iso_14083",
            "scope": "3",
        },
    },
    "air_freight": {
        "synonyms": ["air freight spend", "airfreight"],
        "meta": {
            "factor_family": "transport_lane",
            "method_profile": "freight_iso_14083",
            "scope": "3",
        },
    },
    "ocean_freight": {
        "synonyms": ["ocean freight spend", "sea freight spend", "maritime freight"],
        "meta": {
            "factor_family": "transport_lane",
            "method_profile": "freight_iso_14083",
            "scope": "3",
        },
    },
    "business_travel_air": {
        "synonyms": ["air travel", "business flights", "employee flights", "corporate travel air"],
        "meta": {
            "factor_family": "transport_lane",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 6,
        },
    },
    "business_travel_rail": {
        "synonyms": ["business travel rail", "business rail", "employee rail"],
        "meta": {
            "factor_family": "transport_lane",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 6,
        },
    },
    "office_supplies": {
        "synonyms": ["stationery", "office supplies spend", "paper supplies"],
        "meta": {
            "factor_family": "material_embodied",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 1,
        },
    },
    "it_hardware": {
        "synonyms": ["laptops", "desktop computers", "servers", "it equipment"],
        "meta": {
            "factor_family": "material_embodied",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 2,
        },
    },
    "cloud_compute": {
        "synonyms": [
            "cloud services", "cloud compute", "aws services", "azure services",
            "gcp services", "saas subscription", "infrastructure as a service", "iaas",
        ],
        "meta": {
            "factor_family": "grid_intensity",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 1,
        },
    },
    "professional_services": {
        "synonyms": [
            "consulting", "legal services", "accounting services", "audit fees",
            "advisory services",
        ],
        "meta": {
            "factor_family": "finance_proxy",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 1,
        },
    },
    "construction_services": {
        "synonyms": ["construction", "building works", "capex construction"],
        "meta": {
            "factor_family": "material_embodied",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 2,
        },
    },
    "waste_services": {
        "synonyms": ["waste collection", "waste management", "sanitation services"],
        "meta": {
            "factor_family": "waste_treatment",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 5,
        },
    },
    "employee_commuting": {
        "synonyms": ["commuting allowance", "employee transport subsidy"],
        "meta": {
            "factor_family": "transport_lane",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 7,
        },
    },
    "fuel_purchases_upstream": {
        "synonyms": ["fuel purchases", "gasoline purchases", "diesel purchases"],
        "meta": {
            "factor_family": "emissions",
            "method_profile": "corporate_scope3",
            "scope": "3",
            "scope3_category": 3,
        },
    },
}


class SpendMapping(BaseMapping):
    TAXONOMY = SPEND_TAXONOMY


def map_spend(
    description: str,
    *,
    amount_usd: Optional[float] = None,
    currency: str = "USD",
) -> MappingResult:
    """Map a spend line to a factor family + method profile."""
    inner = SpendMapping._lookup(description)
    if inner is None:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=f"No spend-category match for '{description}'",
            raw_input=description,
        )
    meta = SPEND_TAXONOMY[inner.canonical]["meta"]
    canonical: Dict[str, Any] = {
        "spend_category": inner.canonical,
        "factor_family": meta["factor_family"],
        "method_profile": meta["method_profile"],
        "scope": meta["scope"],
        "scope3_category": meta.get("scope3_category"),
        "amount_usd": amount_usd,
        "currency": currency,
    }
    return MappingResult(
        canonical=canonical,
        confidence=inner.confidence,
        band=inner.band,
        rationale=(
            f"{inner.rationale}; factor_family={meta['factor_family']}; "
            f"method_profile={meta['method_profile']}; scope={meta['scope']}"
        ),
        matched_pattern=inner.matched_pattern,
        raw_input=description,
    )


__all__ = ["SPEND_TAXONOMY", "SpendMapping", "map_spend"]
