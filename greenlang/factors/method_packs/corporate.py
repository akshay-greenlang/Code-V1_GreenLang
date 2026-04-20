# -*- coding: utf-8 -*-
"""Corporate Inventory method packs (GHG Protocol + IFRS S2)."""
from __future__ import annotations

from greenlang.data.canonical_v2 import (
    ElectricityBasis,
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

# Shared deprecation policy for corporate reporting packs — 4-year window
# captures the typical audit cycle + restatement horizon.
_DEPRECATION = DeprecationRule(max_age_days=365 * 4, grace_period_days=365)


CORPORATE_SCOPE1 = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE1,
    name="Corporate Inventory — Scope 1 (GHG Protocol)",
    description=(
        "Direct emissions from sources owned or controlled by the reporting "
        "entity. Fossil CO2 only; biogenic CO2 reported separately per GHG "
        "Protocol Corporate Standard."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.EMISSIONS,
            FactorFamily.HEATING_VALUE,
            FactorFamily.REFRIGERANT_GWP,
            FactorFamily.OXIDATION,
            FactorFamily.CARBON_CONTENT,
        ),
        allowed_formula_types=(FormulaType.DIRECT_FACTOR, FormulaType.COMBUSTION),
        allowed_statuses=("certified",),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("1",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=("GHG_Protocol", "IFRS_S2", "ISO_14064"),
    audit_text_template=(
        "Scope 1 direct emissions computed per GHG Protocol Corporate Standard. "
        "Factor: {factor_id} (source: {source_org}, vintage: {source_year}). "
        "GWP basis: {gwp_basis}. Biogenic CO2 reported separately."
    ),
    pack_version="1.0.0",
    tags=("open_core",),
)

CORPORATE_SCOPE2_LOCATION = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
    name="Corporate Inventory — Scope 2 Location-Based (GHG Protocol)",
    description=(
        "Purchased electricity, steam, heating, cooling using the location-"
        "based method. Grid-average factors; market instruments (RECs, GOs, "
        "PPAs) are explicitly disallowed under this method."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.GRID_INTENSITY, FactorFamily.ENERGY_CONVERSION),
        allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
        allowed_statuses=("certified",),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("2",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.PROHIBITED,
        include_transmission_losses=False,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=("GHG_Protocol_Scope2", "IFRS_S2"),
    audit_text_template=(
        "Scope 2 location-based emissions per GHG Protocol Scope 2 Guidance. "
        "Grid factor: {factor_id} ({geography}, {source_year}). RECs / GOs / "
        "PPAs are not applied under this method."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.LOCATION_BASED,
    tags=("open_core",),
)

CORPORATE_SCOPE2_MARKET = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Corporate Inventory — Scope 2 Market-Based (GHG Protocol)",
    description=(
        "Purchased electricity using market-based method. Supplier-specific "
        "contracts, RECs, GOs, and residual-mix factors are allowed per GHG "
        "Protocol Scope 2 Guidance."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.GRID_INTENSITY,
            FactorFamily.RESIDUAL_MIX,
            FactorFamily.ENERGY_CONVERSION,
        ),
        allowed_formula_types=(FormulaType.DIRECT_FACTOR, FormulaType.RESIDUAL_MIX),
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("2",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.ALLOWED,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=("GHG_Protocol_Scope2", "IFRS_S2", "RE100"),
    audit_text_template=(
        "Scope 2 market-based emissions per GHG Protocol. Supplier contract: "
        "{supplier}; certificate: {certificate}. When no contract applies, "
        "residual mix used: {residual_mix_factor_id}."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.MARKET_BASED,
    tags=("open_core",),
)

CORPORATE_SCOPE3 = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE3,
    name="Corporate Inventory — Scope 3 (GHG Protocol 15 categories)",
    description=(
        "Value-chain emissions across 15 categories. Spend-based, average-"
        "data, supplier-specific, and hybrid methods allowed. Biogenic carbon "
        "reported separately."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.EMISSIONS,
            FactorFamily.MATERIAL_EMBODIED,
            FactorFamily.TRANSPORT_LANE,
            FactorFamily.WASTE_TREATMENT,
            FactorFamily.FINANCE_PROXY,
            FactorFamily.ENERGY_CONVERSION,
        ),
        allowed_formula_types=(
            FormulaType.DIRECT_FACTOR,
            FormulaType.SPEND_PROXY,
            FormulaType.LCA,
            FormulaType.TRANSPORT_CHAIN,
        ),
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("3",),
        allowed_boundaries=("cradle_to_gate", "cradle_to_grave", "WTW", "WTT"),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=("GHG_Protocol_Scope3", "IFRS_S2"),
    audit_text_template=(
        "Scope 3 Cat {scope3_category} per GHG Protocol Scope 3 Standard. "
        "Method: {calculation_method}. Factor: {factor_id}. Data quality: "
        "{dqs_score}/100."
    ),
    pack_version="1.0.0",
    tags=("open_core",),
)


# Register all four corporate packs on import.
for _pack in (
    CORPORATE_SCOPE1,
    CORPORATE_SCOPE2_LOCATION,
    CORPORATE_SCOPE2_MARKET,
    CORPORATE_SCOPE3,
):
    register_pack(_pack)


__all__ = [
    "CORPORATE_SCOPE1",
    "CORPORATE_SCOPE2_LOCATION",
    "CORPORATE_SCOPE2_MARKET",
    "CORPORATE_SCOPE3",
]
