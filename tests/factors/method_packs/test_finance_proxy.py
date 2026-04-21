# -*- coding: utf-8 -*-
"""Tests for GAP-8 — PCAF Finance Proxy method packs."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs import (
    MethodPack,
    MethodPackNotFound,
    get_pack,
    list_packs,
)
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    MarketInstrumentTreatment,
)
from greenlang.factors.method_packs.finance_proxy import (
    FINANCE_PROXY,
    PCAF_ATTRIBUTION_HIERARCHY,
    PCAF_BUSINESS_LOANS,
    PCAF_COMMERCIAL_REAL_ESTATE,
    PCAF_CORPORATE_BONDS,
    PCAF_DQS_RUBRIC,
    PCAF_LISTED_EQUITY,
    PCAF_MORTGAGES,
    PCAF_MOTOR_VEHICLE_LOANS,
    PCAF_PROJECT_FINANCE,
    PCAFAssetClass,
    PCAFAttributionMethod,
    PCAFBusinessLoansPack,
    PCAFCommercialRealEstatePack,
    PCAFCorporateBondsPack,
    PCAFDataQualityScore,
    PCAFIntensityMode,
    PCAFListedEquityPack,
    PCAFMortgagesPack,
    PCAFMotorVehicleLoansPack,
    PCAFPackMetadata,
    PCAFProjectFinancePack,
    get_pcaf_metadata,
    get_pcaf_variant,
    list_pcaf_variants,
    register_pcaf_variant,
)


# ---------------------------------------------------------------------------
# Registry + variant resolution
# ---------------------------------------------------------------------------


class TestPCAFRegistry:
    def test_seven_variants_registered(self):
        names = set(list_pcaf_variants())
        expected = {
            "pcaf_listed_equity",
            "pcaf_corporate_bonds",
            "pcaf_business_loans",
            "pcaf_project_finance",
            "pcaf_commercial_real_estate",
            "pcaf_mortgages",
            "pcaf_motor_vehicle_loans",
        }
        assert expected.issubset(names), sorted(expected - names)

    def test_get_pack_by_string_name(self):
        pack = get_pack("pcaf_listed_equity")
        assert isinstance(pack, MethodPack)
        assert pack.profile == MethodProfile.FINANCE_PROXY
        assert "Listed Equity" in pack.name

    def test_get_pack_unknown_string_raises(self):
        with pytest.raises(MethodPackNotFound):
            get_pack("pcaf_does_not_exist")

    def test_get_pack_with_method_profile_returns_umbrella(self):
        pack = get_pack(MethodProfile.FINANCE_PROXY)
        # Umbrella pack registered under FINANCE_PROXY profile.
        assert pack is FINANCE_PROXY
        assert "PCAF" in pack.name or "Financed Emissions" in pack.name

    def test_umbrella_has_scope3_allowed(self):
        assert "3" in FINANCE_PROXY.boundary_rule.allowed_scopes

    def test_get_pcaf_variant_helper(self):
        pack = get_pcaf_variant("pcaf_mortgages")
        assert isinstance(pack, MethodPack)
        assert "Mortgages" in pack.name

    def test_get_pcaf_variant_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_pcaf_variant("no_such_variant")

    def test_list_packs_contains_finance_profile(self):
        all_packs = list_packs()
        profiles = [p.profile for p in all_packs]
        assert MethodProfile.FINANCE_PROXY in profiles

    def test_register_duplicate_variant_version_bump_logs(self, caplog):
        meta = get_pcaf_metadata("pcaf_mortgages")
        original = get_pcaf_variant("pcaf_mortgages")
        new_pack = MethodPack(
            profile=original.profile,
            name=original.name,
            description=original.description,
            selection_rule=original.selection_rule,
            boundary_rule=original.boundary_rule,
            gwp_basis=original.gwp_basis,
            region_hierarchy=original.region_hierarchy,
            deprecation=original.deprecation,
            reporting_labels=original.reporting_labels,
            audit_text_template=original.audit_text_template,
            pack_version="9.9.9",
            tags=original.tags,
        )
        register_pcaf_variant("pcaf_mortgages", new_pack, meta)
        assert get_pcaf_variant("pcaf_mortgages").pack_version == "9.9.9"
        # Restore
        register_pcaf_variant("pcaf_mortgages", original, meta)

    def test_register_empty_variant_name_raises(self):
        meta = get_pcaf_metadata("pcaf_mortgages")
        pack = get_pcaf_variant("pcaf_mortgages")
        with pytest.raises(ValueError):
            register_pcaf_variant("", pack, meta)


# ---------------------------------------------------------------------------
# Class wrappers
# ---------------------------------------------------------------------------


class TestPCAFClassWrappers:
    def test_listed_equity_class(self):
        p = PCAFListedEquityPack()
        assert p.variant_name == "pcaf_listed_equity"
        assert p.pack is PCAF_LISTED_EQUITY
        assert p.metadata.asset_class == PCAFAssetClass.LISTED_EQUITY
        assert p.metadata.attribution_method == PCAFAttributionMethod.EVIC

    def test_corporate_bonds_class(self):
        p = PCAFCorporateBondsPack()
        assert p.metadata.asset_class == PCAFAssetClass.CORPORATE_BONDS
        assert p.metadata.attribution_method == PCAFAttributionMethod.EVIC

    def test_business_loans_class(self):
        p = PCAFBusinessLoansPack()
        assert p.metadata.asset_class == PCAFAssetClass.BUSINESS_LOANS
        assert (
            p.metadata.attribution_method
            == PCAFAttributionMethod.OUTSTANDING_AMOUNT_PLUS_EQUITY
        )
        # Revenue-proxy fallback mentioned in the formula.
        assert "revenue" in p.metadata.attribution_formula.lower()

    def test_project_finance_class(self):
        p = PCAFProjectFinancePack()
        assert p.metadata.asset_class == PCAFAssetClass.PROJECT_FINANCE
        assert (
            p.metadata.attribution_method
            == PCAFAttributionMethod.COMMITTED_CAPITAL
        )

    def test_commercial_real_estate_class(self):
        p = PCAFCommercialRealEstatePack()
        assert p.metadata.asset_class == PCAFAssetClass.COMMERCIAL_REAL_ESTATE
        assert (
            PCAFIntensityMode.PHYSICAL_INTENSITY_FLOOR_AREA
            in p.metadata.intensity_modes
        )

    def test_mortgages_class(self):
        p = PCAFMortgagesPack()
        assert p.metadata.asset_class == PCAFAssetClass.MORTGAGES
        assert p.pack.boundary_rule.allowed_scopes == ("1", "2")

    def test_motor_vehicle_loans_class(self):
        p = PCAFMotorVehicleLoansPack()
        assert p.metadata.asset_class == PCAFAssetClass.MOTOR_VEHICLE_LOANS
        assert (
            PCAFIntensityMode.PHYSICAL_INTENSITY_DISTANCE
            in p.metadata.intensity_modes
        )


# ---------------------------------------------------------------------------
# Selection + boundary rules
# ---------------------------------------------------------------------------


class TestPCAFSelectionRules:
    def test_spend_proxy_accepted(self):
        pack = PCAF_BUSINESS_LOANS
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.FINANCE_PROXY.value,
            formula_type=FormulaType.SPEND_PROXY.value,
        )
        assert pack.selection_rule.accepts(rec) is True

    def test_direct_factor_accepted(self):
        pack = PCAF_MORTGAGES
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.GRID_INTENSITY.value,
            formula_type=FormulaType.DIRECT_FACTOR.value,
        )
        assert pack.selection_rule.accepts(rec) is True

    def test_deprecated_rejected(self):
        pack = PCAF_LISTED_EQUITY
        rec = SimpleNamespace(
            factor_status="deprecated",
            factor_family=FactorFamily.FINANCE_PROXY.value,
            formula_type=FormulaType.SPEND_PROXY.value,
        )
        assert pack.selection_rule.accepts(rec) is False

    def test_preview_accepted(self):
        pack = PCAF_CORPORATE_BONDS
        rec = SimpleNamespace(
            factor_status="preview",
            factor_family=FactorFamily.FINANCE_PROXY.value,
            formula_type=FormulaType.SPEND_PROXY.value,
        )
        assert pack.selection_rule.accepts(rec) is True

    def test_wrong_family_rejected(self):
        pack = PCAF_LISTED_EQUITY
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.REFRIGERANT_GWP.value,
            formula_type=FormulaType.DIRECT_FACTOR.value,
        )
        assert pack.selection_rule.accepts(rec) is False


class TestPCAFBoundaryRules:
    @pytest.mark.parametrize(
        "pack",
        [
            PCAF_LISTED_EQUITY,
            PCAF_CORPORATE_BONDS,
            PCAF_BUSINESS_LOANS,
            PCAF_PROJECT_FINANCE,
            PCAF_COMMERCIAL_REAL_ESTATE,
            PCAF_MORTGAGES,
            PCAF_MOTOR_VEHICLE_LOANS,
        ],
    )
    def test_biogenic_reported_separately(self, pack):
        assert pack.boundary_rule.biogenic_treatment == BiogenicTreatment.REPORTED_SEPARATELY

    @pytest.mark.parametrize(
        "pack",
        [
            PCAF_LISTED_EQUITY,
            PCAF_CORPORATE_BONDS,
            PCAF_BUSINESS_LOANS,
            PCAF_PROJECT_FINANCE,
            PCAF_COMMERCIAL_REAL_ESTATE,
            PCAF_MORTGAGES,
            PCAF_MOTOR_VEHICLE_LOANS,
        ],
    )
    def test_market_instruments_na(self, pack):
        assert pack.boundary_rule.market_instruments == MarketInstrumentTreatment.NOT_APPLICABLE

    def test_real_estate_scope_limited(self):
        assert PCAF_COMMERCIAL_REAL_ESTATE.boundary_rule.allowed_scopes == ("1", "2")

    def test_mortgages_scope_limited(self):
        assert PCAF_MORTGAGES.boundary_rule.allowed_scopes == ("1", "2")

    def test_business_loans_all_scopes(self):
        assert PCAF_BUSINESS_LOANS.boundary_rule.allowed_scopes == ("1", "2", "3")


# ---------------------------------------------------------------------------
# DQS rubric + uncertainty
# ---------------------------------------------------------------------------


class TestPCAFDataQuality:
    def test_dqs_rubric_has_five_tiers(self):
        assert set(PCAF_DQS_RUBRIC.keys()) == {1, 2, 3, 4, 5}

    def test_dqs_enum_values(self):
        assert PCAFDataQualityScore.SCORE_1_VERIFIED.value == 1
        assert PCAFDataQualityScore.SCORE_5_ASSET_CLASS_DEFAULT.value == 5

    def test_all_metadata_require_uncertainty_at_dqs_4_or_higher(self):
        for name in list_pcaf_variants():
            meta = get_pcaf_metadata(name)
            # Per spec: DQS >= 4 requires uncertainty band disclosure.
            assert meta.uncertainty_band_required_dqs == 4

    def test_metadata_has_full_dqs_scale(self):
        meta = get_pcaf_metadata("pcaf_listed_equity")
        assert len(meta.dqs_scale) == 5

    def test_dqs_rubric_descriptive(self):
        assert "verified" in PCAF_DQS_RUBRIC[1].lower()
        assert "unverified" in PCAF_DQS_RUBRIC[2].lower()
        assert "physical" in PCAF_DQS_RUBRIC[3].lower()
        assert "economic" in PCAF_DQS_RUBRIC[4].lower()
        assert "asset-class-average proxy" in PCAF_DQS_RUBRIC[5].lower()


# ---------------------------------------------------------------------------
# Attribution hierarchy
# ---------------------------------------------------------------------------


class TestPCAFAttributionHierarchy:
    def test_five_step_hierarchy(self):
        assert len(PCAF_ATTRIBUTION_HIERARCHY) == 5

    def test_ranks_monotonic(self):
        ranks = [step.rank for step in PCAF_ATTRIBUTION_HIERARCHY]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1
        assert ranks[-1] == 5

    def test_customer_specific_first(self):
        assert PCAF_ATTRIBUTION_HIERARCHY[0].label == "customer_specific"

    def test_asset_class_default_last(self):
        assert PCAF_ATTRIBUTION_HIERARCHY[-1].label == "asset_class_default"

    def test_every_metadata_uses_hierarchy(self):
        for name in list_pcaf_variants():
            meta = get_pcaf_metadata(name)
            assert meta.attribution_hierarchy == PCAF_ATTRIBUTION_HIERARCHY


# ---------------------------------------------------------------------------
# Scope 3 trigger sectors
# ---------------------------------------------------------------------------


class TestPCAFScope3Sectors:
    def test_high_emitting_sectors_included(self):
        meta = get_pcaf_metadata("pcaf_listed_equity")
        triggers = set(meta.requires_scope3_for_sectors)
        must_have = {
            "oil_and_gas",
            "coal_mining",
            "power_generation",
            "steel",
            "cement",
        }
        assert must_have.issubset(triggers)

    def test_real_estate_no_scope3_sectors(self):
        meta = get_pcaf_metadata("pcaf_commercial_real_estate")
        # CRE emissions come from the building itself, not a counterparty.
        assert meta.requires_scope3_for_sectors == ()


# ---------------------------------------------------------------------------
# Intensity modes
# ---------------------------------------------------------------------------


class TestPCAFIntensityModes:
    def test_absolute_mode_always_available(self):
        for name in list_pcaf_variants():
            meta = get_pcaf_metadata(name)
            assert PCAFIntensityMode.ABSOLUTE in meta.intensity_modes

    def test_mortgages_has_floor_area_mode(self):
        meta = get_pcaf_metadata("pcaf_mortgages")
        assert PCAFIntensityMode.PHYSICAL_INTENSITY_FLOOR_AREA in meta.intensity_modes

    def test_motor_vehicle_has_distance_mode(self):
        meta = get_pcaf_metadata("pcaf_motor_vehicle_loans")
        assert PCAFIntensityMode.PHYSICAL_INTENSITY_DISTANCE in meta.intensity_modes

    def test_project_finance_has_energy_mode(self):
        meta = get_pcaf_metadata("pcaf_project_finance")
        assert PCAFIntensityMode.PHYSICAL_INTENSITY_ENERGY in meta.intensity_modes


# ---------------------------------------------------------------------------
# Reporting labels + GWP + versioning
# ---------------------------------------------------------------------------


class TestPCAFReportingLabels:
    @pytest.mark.parametrize(
        "pack",
        [
            PCAF_LISTED_EQUITY,
            PCAF_CORPORATE_BONDS,
            PCAF_BUSINESS_LOANS,
            PCAF_PROJECT_FINANCE,
            PCAF_COMMERCIAL_REAL_ESTATE,
            PCAF_MORTGAGES,
            PCAF_MOTOR_VEHICLE_LOANS,
        ],
    )
    def test_has_pcaf_label(self, pack):
        labels = set(pack.reporting_labels)
        assert "PCAF" in labels
        assert "GHG_Protocol_Scope3_Cat15" in labels

    def test_ifrs_s2_included(self):
        assert "IFRS_S2" in PCAF_LISTED_EQUITY.reporting_labels

    def test_gwp_basis_ar6(self):
        for name in list_pcaf_variants():
            pack = get_pcaf_variant(name)
            assert pack.gwp_basis == "IPCC_AR6_100"

    def test_pack_version_set(self):
        for name in list_pcaf_variants():
            pack = get_pcaf_variant(name)
            # Every production pack advanced past stub v0.x.
            major = int(pack.pack_version.split(".")[0])
            assert major >= 1


# ---------------------------------------------------------------------------
# Audit templates
# ---------------------------------------------------------------------------


class TestPCAFAuditTemplates:
    def test_templates_reference_attribution_factor(self):
        for name in list_pcaf_variants():
            pack = get_pcaf_variant(name)
            assert "attribution_factor" in pack.audit_text_template

    def test_templates_reference_dqs(self):
        for name in list_pcaf_variants():
            pack = get_pcaf_variant(name)
            assert "pcaf_dqs" in pack.audit_text_template


# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------


class TestPCAFConfigYAML:
    @pytest.fixture(scope="class")
    def config(self):
        cfg_path = (
            Path(__file__).resolve().parents[3]
            / "greenlang"
            / "factors"
            / "data"
            / "method_packs"
            / "pcaf_config.yaml"
        )
        assert cfg_path.is_file(), f"missing {cfg_path}"
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def test_config_has_seven_asset_classes(self, config):
        asset_classes = config["asset_classes"]
        assert len(asset_classes) == 7

    def test_config_asset_class_ids_unique_and_sequential(self, config):
        ids = [ac["asset_class_id"] for ac in config["asset_classes"]]
        assert sorted(ids) == list(range(1, 8))

    def test_config_variant_names_match_registered(self, config):
        yaml_names = {ac["variant_name"] for ac in config["asset_classes"]}
        registered = set(list_pcaf_variants())
        assert yaml_names == registered

    def test_config_dqs_has_five_tiers(self, config):
        assert set(config["dqs_rubric"].keys()) == {1, 2, 3, 4, 5}

    def test_config_dqs4_and_dqs5_require_uncertainty_band(self, config):
        assert config["dqs_rubric"][4]["requires_uncertainty_band"] is True
        assert config["dqs_rubric"][5]["requires_uncertainty_band"] is True

    def test_config_proxy_hierarchy_five_steps(self, config):
        assert len(config["proxy_hierarchy"]) == 5
        assert config["proxy_hierarchy"][0]["label"] == "customer_specific"
        assert config["proxy_hierarchy"][-1]["label"] == "asset_class_default"

    def test_config_sector_crosswalk_nonempty(self, config):
        assert len(config["sector_crosswalk"]) >= 15

    def test_config_sector_crosswalk_has_nace_and_naics(self, config):
        for entry in config["sector_crosswalk"]:
            assert entry["nace_rev2"], entry
            assert entry["naics_2022"], entry

    def test_config_scope3_sectors_match_metadata(self, config):
        yaml_sectors = set(config["scope3_required_sectors"])
        meta = get_pcaf_metadata("pcaf_listed_equity")
        runtime_sectors = set(meta.requires_scope3_for_sectors)
        assert yaml_sectors == runtime_sectors


# ---------------------------------------------------------------------------
# End-to-end resolution smoke test
# ---------------------------------------------------------------------------


class TestPCAFResolutionSmoke:
    def test_get_pack_returns_same_instance_twice(self):
        a = get_pack("pcaf_listed_equity")
        b = get_pack("pcaf_listed_equity")
        assert a is b

    def test_variant_is_independent_of_umbrella(self):
        variant = get_pack("pcaf_listed_equity")
        umbrella = get_pack(MethodProfile.FINANCE_PROXY)
        assert variant is not umbrella
