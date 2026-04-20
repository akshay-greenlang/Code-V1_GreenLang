# -*- coding: utf-8 -*-
"""Phase F2 — Method Pack Library tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from greenlang.data.canonical_v2 import (
    ElectricityBasis,
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs import (
    MethodPack,
    get_pack,
    list_packs,
    register_pack,
    registered_profiles,
)
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    BoundaryRule,
    DEFAULT_FALLBACK,
    DeprecationRule,
    MarketInstrumentTreatment,
    SelectionRule,
)
from greenlang.factors.method_packs.registry import MethodPackNotFound


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class TestRegistry:
    def test_get_known_pack(self):
        pack = get_pack(MethodProfile.CORPORATE_SCOPE1)
        assert pack.profile == MethodProfile.CORPORATE_SCOPE1
        assert "Scope 1" in pack.name

    def test_get_unknown_profile_raises(self):
        # Use a valid enum value that we deliberately un-register for the test.
        with pytest.raises(MethodPackNotFound):
            get_pack("nonexistent")  # type: ignore[arg-type]

    def test_seven_core_profiles_registered(self):
        expected = {
            MethodProfile.CORPORATE_SCOPE1,
            MethodProfile.CORPORATE_SCOPE2_LOCATION,
            MethodProfile.CORPORATE_SCOPE2_MARKET,
            MethodProfile.CORPORATE_SCOPE3,
            MethodProfile.PRODUCT_CARBON,
            MethodProfile.FREIGHT_ISO_14083,
            MethodProfile.LAND_REMOVALS,
            MethodProfile.FINANCE_PROXY,
            MethodProfile.EU_CBAM,
            MethodProfile.EU_DPP,
        }
        profiles = set(registered_profiles())
        assert expected.issubset(profiles), sorted(expected - profiles)

    def test_list_packs_returns_all(self):
        packs = list_packs()
        assert len(packs) >= 10

    def test_register_custom_pack(self):
        custom = MethodPack(
            profile=MethodProfile.CORPORATE_SCOPE1,           # overwrite path
            name="Test override",
            description="unit test",
            selection_rule=SelectionRule(
                allowed_families=(FactorFamily.EMISSIONS,),
                allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
            ),
            boundary_rule=BoundaryRule(
                allowed_scopes=("1",),
                allowed_boundaries=("combustion",),
            ),
            gwp_basis="IPCC_AR6_100",
            region_hierarchy=DEFAULT_FALLBACK,
            deprecation=DeprecationRule(max_age_days=1000),
            reporting_labels=("test",),
            audit_text_template="",
            pack_version="9.9.9",
        )
        register_pack(custom)
        assert get_pack(MethodProfile.CORPORATE_SCOPE1).pack_version == "9.9.9"
        # Restore the real pack so downstream tests don't see the override.
        from greenlang.factors.method_packs.corporate import CORPORATE_SCOPE1

        register_pack(CORPORATE_SCOPE1)


# --------------------------------------------------------------------------
# Corporate Packs
# --------------------------------------------------------------------------


class TestCorporatePacks:
    def test_scope1_rejects_scope2_record(self):
        pack = get_pack(MethodProfile.CORPORATE_SCOPE1)
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.GRID_INTENSITY.value,  # scope 2
            formula_type=FormulaType.DIRECT_FACTOR.value,
        )
        # Family is GRID_INTENSITY, not in Scope 1 pack's allowed list.
        assert pack.selection_rule.accepts(rec) is False

    def test_scope1_accepts_combustion_record(self):
        pack = get_pack(MethodProfile.CORPORATE_SCOPE1)
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.EMISSIONS.value,
            formula_type=FormulaType.COMBUSTION.value,
        )
        assert pack.selection_rule.accepts(rec) is True

    def test_scope2_location_rejects_market_instruments(self):
        pack = get_pack(MethodProfile.CORPORATE_SCOPE2_LOCATION)
        assert (
            pack.boundary_rule.market_instruments
            == MarketInstrumentTreatment.PROHIBITED
        )

    def test_scope2_market_allows_instruments(self):
        pack = get_pack(MethodProfile.CORPORATE_SCOPE2_MARKET)
        assert (
            pack.boundary_rule.market_instruments
            == MarketInstrumentTreatment.ALLOWED
        )
        assert pack.electricity_basis == ElectricityBasis.MARKET_BASED

    def test_scope3_allows_lca_and_spend_proxy(self):
        pack = get_pack(MethodProfile.CORPORATE_SCOPE3)
        rec_lca = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.MATERIAL_EMBODIED.value,
            formula_type=FormulaType.LCA.value,
        )
        rec_spend = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.FINANCE_PROXY.value,
            formula_type=FormulaType.SPEND_PROXY.value,
        )
        assert pack.selection_rule.accepts(rec_lca) is True
        assert pack.selection_rule.accepts(rec_spend) is True

    def test_biogenic_treatment_consistent_across_scope12(self):
        s1 = get_pack(MethodProfile.CORPORATE_SCOPE1)
        s2 = get_pack(MethodProfile.CORPORATE_SCOPE2_LOCATION)
        for pack in (s1, s2):
            assert pack.boundary_rule.biogenic_treatment in (
                BiogenicTreatment.REPORTED_SEPARATELY,
                BiogenicTreatment.EXCLUDED,
            )


# --------------------------------------------------------------------------
# EU Policy Packs
# --------------------------------------------------------------------------


class TestEUPolicyPacks:
    def test_cbam_requires_verification(self):
        cbam = get_pack(MethodProfile.EU_CBAM)
        assert cbam.selection_rule.require_verification is True

    def test_cbam_rejects_unverified_record(self):
        cbam = get_pack(MethodProfile.EU_CBAM)
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.EMISSIONS.value,
            formula_type=FormulaType.COMBUSTION.value,
            verification=SimpleNamespace(status="unverified"),
        )
        assert cbam.selection_rule.accepts(rec) is False

    def test_cbam_accepts_regulator_approved(self):
        cbam = get_pack(MethodProfile.EU_CBAM)
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.EMISSIONS.value,
            formula_type=FormulaType.COMBUSTION.value,
            verification=SimpleNamespace(status="regulator_approved"),
        )
        assert cbam.selection_rule.accepts(rec) is True

    def test_cbam_biogenic_excluded(self):
        cbam = get_pack(MethodProfile.EU_CBAM)
        assert cbam.boundary_rule.biogenic_treatment == BiogenicTreatment.EXCLUDED

    def test_dpp_pack_registered(self):
        dpp = get_pack(MethodProfile.EU_DPP)
        assert "ESPR" in dpp.reporting_labels
        assert dpp.pack_version == "0.1.0"


# --------------------------------------------------------------------------
# Product / Freight / Land / Finance packs
# --------------------------------------------------------------------------


class TestDomainPacks:
    def test_product_carbon_supports_pact(self):
        pack = get_pack(MethodProfile.PRODUCT_CARBON)
        assert "PACT" in pack.reporting_labels
        assert pack.boundary_rule.allowed_boundaries == ("cradle_to_gate", "cradle_to_grave")

    def test_freight_uses_transport_chain_formula(self):
        pack = get_pack(MethodProfile.FREIGHT_ISO_14083)
        assert FormulaType.TRANSPORT_CHAIN in pack.selection_rule.allowed_formula_types
        assert pack.boundary_rule.allowed_boundaries == ("WTW", "WTT")

    def test_land_removals_includes_biogenic(self):
        pack = get_pack(MethodProfile.LAND_REMOVALS)
        assert pack.boundary_rule.biogenic_treatment == BiogenicTreatment.INCLUDED

    def test_finance_proxy_is_spend_based(self):
        pack = get_pack(MethodProfile.FINANCE_PROXY)
        assert FormulaType.SPEND_PROXY in pack.selection_rule.allowed_formula_types
        assert "PCAF" in pack.reporting_labels


# --------------------------------------------------------------------------
# SelectionRule filter logic
# --------------------------------------------------------------------------


class TestSelectionRule:
    def test_rejects_deprecated_by_default(self):
        rule = SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
        )
        rec = SimpleNamespace(
            factor_status="deprecated",
            factor_family=FactorFamily.EMISSIONS.value,
            formula_type=FormulaType.DIRECT_FACTOR.value,
        )
        assert rule.accepts(rec) is False

    def test_allows_preview_when_configured(self):
        rule = SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
            allowed_statuses=("certified", "preview"),
        )
        rec = SimpleNamespace(
            factor_status="preview",
            factor_family=FactorFamily.EMISSIONS.value,
            formula_type=FormulaType.DIRECT_FACTOR.value,
        )
        assert rule.accepts(rec) is True

    def test_legacy_record_without_factor_family_defaults_to_emissions(self):
        rule = SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(),
        )
        rec = SimpleNamespace(factor_status="certified")
        assert rule.accepts(rec) is True          # defaults to EMISSIONS

    def test_primary_data_required(self):
        rule = SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(),
            require_primary_data=True,
        )
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.EMISSIONS.value,
            primary_data_flag="secondary",
        )
        assert rule.accepts(rec) is False
        rec_primary = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.EMISSIONS.value,
            primary_data_flag="primary",
        )
        assert rule.accepts(rec_primary) is True


# --------------------------------------------------------------------------
# India CEA parser
# --------------------------------------------------------------------------


class TestIndiaCEAParser:
    def test_all_india_row_parses(self):
        from greenlang.factors.ingestion.parsers.india_cea import parse_india_cea_rows

        rows = [
            {
                "grid": "All India",
                "financial_year": "2023-24",
                "co2_intensity_t_per_mwh": 0.727,
                "publication_version": "v20.0",
            }
        ]
        records = parse_india_cea_rows(rows)
        assert len(records) == 1
        rec = records[0]
        assert rec.geography == "IN"
        assert rec.factor_family == FactorFamily.GRID_INTENSITY.value
        assert rec.method_profile == MethodProfile.CORPORATE_SCOPE2_LOCATION.value
        assert rec.jurisdiction.country == "IN"
        assert rec.parameters.electricity_basis == ElectricityBasis.LOCATION_BASED
        assert float(rec.vectors.CO2) == pytest.approx(0.727, abs=1e-6)

    def test_regional_grid_records_grid_region(self):
        from greenlang.factors.ingestion.parsers.india_cea import parse_india_cea_rows

        rows = [
            {"grid": "S", "financial_year": "2023-24", "co2_intensity_t_per_mwh": 0.692}
        ]
        rec = parse_india_cea_rows(rows)[0]
        assert rec.jurisdiction.grid_region == "S"

    def test_skips_invalid_rows(self):
        from greenlang.factors.ingestion.parsers.india_cea import parse_india_cea_rows

        rows = [
            {"missing_grid_field": True},                    # skipped
            {"grid": "All India", "financial_year": "2023-24",
             "co2_intensity_t_per_mwh": 0.727},              # kept
        ]
        records = parse_india_cea_rows(rows)
        assert len(records) == 1


# --------------------------------------------------------------------------
# AIB residual-mix parser
# --------------------------------------------------------------------------


class TestAIBResidualMixParser:
    def test_germany_row_parses(self):
        from greenlang.factors.ingestion.parsers.aib_residual_mix import (
            parse_aib_residual_mix_rows,
        )

        rows = [
            {
                "country": "DE",
                "calendar_year": 2023,
                "residual_mix_g_co2_per_kwh": 498.0,
                "version": "AIB-2024-v1",
            }
        ]
        rec = parse_aib_residual_mix_rows(rows)[0]
        assert rec.geography == "DE"
        assert float(rec.vectors.CO2) == pytest.approx(0.498, abs=1e-6)
        assert rec.factor_family == FactorFamily.RESIDUAL_MIX.value
        assert rec.method_profile == MethodProfile.CORPORATE_SCOPE2_MARKET.value
        assert rec.formula_type == FormulaType.RESIDUAL_MIX.value
        assert rec.parameters.residual_mix_applicable is True

    def test_redistribution_class_restricted(self):
        from greenlang.data.canonical_v2 import RedistributionClass
        from greenlang.factors.ingestion.parsers.aib_residual_mix import (
            parse_aib_residual_mix_rows,
        )

        rows = [
            {"country": "FR", "calendar_year": 2023, "residual_mix_g_co2_per_kwh": 50.0}
        ]
        rec = parse_aib_residual_mix_rows(rows)[0]
        assert rec.redistribution_class == RedistributionClass.RESTRICTED.value
