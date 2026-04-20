# -*- coding: utf-8 -*-
"""
GreenLang Factors — Mapping Layer (Phase F4).

Customers almost never start with factor-ready labels. They hand us
strings like "diesel for our Indian truck fleet" or "aluminium ingot
imports from Norway" or NAICS-221112 spend lines. The mapping layer
turns those into canonical labels the resolution engine can match.

Every sub-module returns a :class:`MappingResult` that carries:

- the canonical key(s) it matched (fuel_type, transport_mode, naics, etc.)
- a confidence score (0.0–1.0)
- a short rationale (matched pattern, fuzzy rank, etc.)
- the raw source attributes that drove the decision (for the Explain
  endpoint).

Public entry points re-exported here. Individual taxonomies live in
``fuels.py``, ``transport.py``, ``materials.py``, ``waste.py``,
``electricity_market.py``, ``classifications.py``, ``spend.py``.

Usage::

    from greenlang.factors.mapping import map_fuel, map_transport, map_spend

    fuel = map_fuel("No. 2 distillate diesel, India")
    # MappingResult(canonical='diesel', confidence=0.92, ...)

    spend = map_spend("Purchased goods — aluminum extrusions", amount_usd=42000)
    # MappingResult(canonical={'factor_family': 'material_embodied', ...}, ...)

All mapping modules are **pure** — no network calls, no database reads.
Callers that need persistence wrap the results.
"""
from __future__ import annotations

from greenlang.factors.mapping.base import (
    BaseMapping,
    MappingConfidence,
    MappingError,
    MappingResult,
    normalize_text,
)
from greenlang.factors.mapping.classifications import (
    cross_map_classification,
    map_classification,
)
from greenlang.factors.mapping.electricity_market import (
    ElectricityMarketCategory,
    map_electricity_market,
)
from greenlang.factors.mapping.fuels import FUEL_TAXONOMY, map_fuel
from greenlang.factors.mapping.materials import (
    MATERIAL_TAXONOMY,
    map_material,
)
from greenlang.factors.mapping.spend import map_spend
from greenlang.factors.mapping.transport import (
    TRANSPORT_MODES,
    VEHICLE_CLASSES,
    map_transport,
)
from greenlang.factors.mapping.waste import WASTE_ROUTES, map_waste

__all__ = [
    # Base
    "BaseMapping",
    "MappingConfidence",
    "MappingError",
    "MappingResult",
    "normalize_text",
    # Fuel
    "FUEL_TAXONOMY",
    "map_fuel",
    # Transport
    "TRANSPORT_MODES",
    "VEHICLE_CLASSES",
    "map_transport",
    # Materials
    "MATERIAL_TAXONOMY",
    "map_material",
    # Waste
    "WASTE_ROUTES",
    "map_waste",
    # Electricity market
    "ElectricityMarketCategory",
    "map_electricity_market",
    # Classifications
    "map_classification",
    "cross_map_classification",
    # Spend
    "map_spend",
]
