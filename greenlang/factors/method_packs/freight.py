# -*- coding: utf-8 -*-
"""Freight method pack (ISO 14083 + GLEC Framework)."""
from __future__ import annotations

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    BoundaryRule,
    DEFAULT_FALLBACK,
    DeprecationRule,
    MarketInstrumentTreatment,
    MethodPack,
    SelectionRule,
)
from greenlang.factors.method_packs.registry import register_pack


FREIGHT_ISO_14083 = MethodPack(
    profile=MethodProfile.FREIGHT_ISO_14083,
    name="Freight — ISO 14083:2023 + GLEC Framework",
    description=(
        "Transport-chain emissions per ISO 14083:2023 implemented via the "
        "Smart Freight Centre GLEC Framework. Supports shipment-level and "
        "consignment-level calculation modes across road, sea, air, rail, and "
        "inland waterway with WTW / TTW labelling."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.TRANSPORT_LANE,
            FactorFamily.EMISSIONS,
            FactorFamily.ENERGY_CONVERSION,
        ),
        allowed_formula_types=(
            FormulaType.TRANSPORT_CHAIN,
            FormulaType.DIRECT_FACTOR,
        ),
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("3",),
        allowed_boundaries=("WTW", "WTT"),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=DeprecationRule(max_age_days=365 * 3, grace_period_days=180),
    reporting_labels=("ISO_14083", "GLEC"),
    audit_text_template=(
        "Transport chain per ISO 14083. Leg: {leg_id} ({mode}, {distance_km} km, "
        "payload {payload_t} t). Factor: {factor_id}. WTW basis."
    ),
    pack_version="1.0.0",
    tags=("freight", "licensed"),    # Freight Pack is a premium SKU
)


register_pack(FREIGHT_ISO_14083)


__all__ = ["FREIGHT_ISO_14083"]
