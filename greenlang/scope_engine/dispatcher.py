# -*- coding: utf-8 -*-
"""Dispatcher — routes ActivityRecord → MRV agent via Category Mapper.

Stub implementation: maps by `activity_type` prefix. Production wiring to the
30 AGENT-MRV agents (scope 1: 001-008, scope 2: 009-013, scope 3: 014-028,
cross-cutting: 029-030) lives in SCOPE-ENG 2 (task #22).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from greenlang.data.emission_factor_record import Scope
from greenlang.scope_engine.models import ActivityRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MRVRoute:
    scope: Scope
    agent_id: str
    scope3_category: int | None = None


# Canonical activity-type prefix → route.
# Extended at runtime via AGENT-MRV-029 (Category Mapper) before GA.
_ROUTES: dict[str, MRVRoute] = {
    "stationary_combustion": MRVRoute(Scope.SCOPE_1, "mrv.001.stationary"),
    "refrigerant": MRVRoute(Scope.SCOPE_1, "mrv.002.refrigerant"),
    "mobile_combustion": MRVRoute(Scope.SCOPE_1, "mrv.003.mobile"),
    "process_emissions": MRVRoute(Scope.SCOPE_1, "mrv.004.process"),
    "fugitive": MRVRoute(Scope.SCOPE_1, "mrv.005.fugitive"),
    "land_use": MRVRoute(Scope.SCOPE_1, "mrv.006.land_use"),
    "waste_direct": MRVRoute(Scope.SCOPE_1, "mrv.007.waste"),
    "agricultural": MRVRoute(Scope.SCOPE_1, "mrv.008.agricultural"),
    "purchased_electricity_location": MRVRoute(Scope.SCOPE_2, "mrv.009.location"),
    "purchased_electricity_market": MRVRoute(Scope.SCOPE_2, "mrv.010.market"),
    "purchased_steam": MRVRoute(Scope.SCOPE_2, "mrv.011.steam"),
    "purchased_cooling": MRVRoute(Scope.SCOPE_2, "mrv.012.cooling"),
}

# Scope 3: categories 1-15
_SCOPE3_CATEGORIES: dict[int, str] = {
    1: "mrv.014.purchased_goods",
    2: "mrv.015.capital_goods",
    3: "mrv.016.fuel_energy_upstream",
    4: "mrv.017.upstream_transport",
    5: "mrv.018.operational_waste",
    6: "mrv.019.business_travel",
    7: "mrv.020.employee_commuting",
    8: "mrv.021.upstream_leased",
    9: "mrv.022.downstream_transport",
    10: "mrv.023.processing",
    11: "mrv.024.use_of_sold",
    12: "mrv.025.eol_sold",
    13: "mrv.026.downstream_leased",
    14: "mrv.027.franchises",
    15: "mrv.028.investments",
}


# Fallback fuel-type → canonical activity_type mapping (for records that pass a
# raw fuel name instead of the semantic category).
_FUEL_FALLBACK: dict[str, str] = {
    "diesel": "stationary_combustion",
    "gasoline": "mobile_combustion",
    "natural_gas": "stationary_combustion",
    "propane": "stationary_combustion",
    "coal": "stationary_combustion",
    "fuel_oil": "stationary_combustion",
    "lpg": "stationary_combustion",
    "jet_fuel": "mobile_combustion",
    "electricity": "purchased_electricity_location",
    "steam": "purchased_steam",
    "cooling": "purchased_cooling",
}


def resolve(record: ActivityRecord) -> MRVRoute:
    """Resolve activity -> MRV route.

    Precedence: scope3_category > explicit activity_type > fuel_type fallback
    > scope_hint fallback.
    """
    if record.scope3_category:
        agent = _SCOPE3_CATEGORIES.get(record.scope3_category)
        if agent:
            return MRVRoute(Scope.SCOPE_3, agent, record.scope3_category)

    route = _ROUTES.get(record.activity_type)
    if route:
        return route

    # Fallback: activity_type might actually be a fuel name (diesel, natural_gas)
    fallback_type = _FUEL_FALLBACK.get(record.activity_type) or (
        _FUEL_FALLBACK.get(record.fuel_type) if record.fuel_type else None
    )
    if fallback_type:
        fallback = _ROUTES.get(fallback_type)
        if fallback:
            return fallback

    if record.scope_hint:
        return MRVRoute(record.scope_hint, f"mrv.generic.{record.scope_hint.value}")

    raise ValueError(
        f"Unable to resolve MRV route for activity {record.activity_id} "
        f"(type={record.activity_type}, fuel_type={record.fuel_type}, "
        f"scope_hint={record.scope_hint})"
    )
