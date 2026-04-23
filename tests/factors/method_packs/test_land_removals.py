# -*- coding: utf-8 -*-
"""Tests for GAP-9 — GHG Protocol Land Sector & Removals method packs."""
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
from greenlang.factors.method_packs.land_removals import (
    DEFAULT_BUFFER_POOL,
    GHG_LSR_LAND_MANAGEMENT,
    GHG_LSR_LAND_USE_EMISSIONS,
    GHG_LSR_REMOVALS,
    GHG_LSR_STORAGE,
    GHGLSRLandManagementPack,
    GHGLSRLandUseEmissionsPack,
    GHGLSRRemovalsPack,
    GHGLSRStoragePack,
    LAND_REMOVALS,
    LSR_FALLBACK_HIERARCHY,
    LSRPackMetadata,
    BiogenicAccountingTreatment,
    PermanenceClass,
    RISK_BUFFER_MULTIPLIER,
    RemovalCategory,
    RemovalType,
    ReportingFrequency,
    ReversalRiskLevel,
    VerificationStandard,
    compute_buffer_pool_pct,
    get_lsr_metadata,
    get_lsr_variant,
    list_lsr_variants,
    register_lsr_variant,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestLSRRegistry:
    def test_four_variants_registered(self):
        names = set(list_lsr_variants())
        expected = {
            "lsr_land_use_emissions",
            "lsr_land_management",
            "lsr_removals",
            "lsr_storage",
        }
        assert expected.issubset(names), sorted(expected - names)

    def test_get_pack_by_string_name(self):
        pack = get_pack("lsr_removals")
        assert isinstance(pack, MethodPack)
        assert pack.profile == MethodProfile.LAND_REMOVALS

    def test_get_pack_unknown_variant_raises(self):
        with pytest.raises(MethodPackNotFound):
            get_pack("lsr_does_not_exist")

    def test_umbrella_registered_on_method_profile(self):
        pack = get_pack(MethodProfile.LAND_REMOVALS)
        assert pack is LAND_REMOVALS

    def test_get_lsr_variant_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_lsr_variant("nope")

    def test_register_empty_name_raises(self):
        meta = get_lsr_metadata("lsr_removals")
        pack = get_lsr_variant("lsr_removals")
        with pytest.raises(ValueError):
            register_lsr_variant("", pack, meta)

    def test_list_packs_contains_land_profile(self):
        all_packs = list_packs()
        assert MethodProfile.LAND_REMOVALS in [p.profile for p in all_packs]


# ---------------------------------------------------------------------------
# Class wrappers
# ---------------------------------------------------------------------------


class TestLSRClassWrappers:
    def test_land_use_emissions_wrapper(self):
        p = GHGLSRLandUseEmissionsPack()
        assert p.variant_name == "lsr_land_use_emissions"
        assert p.pack is GHG_LSR_LAND_USE_EMISSIONS
        assert p.metadata.iluc_included is True
        assert p.metadata.soc_tracked is True

    def test_land_management_wrapper(self):
        p = GHGLSRLandManagementPack()
        assert p.metadata.permanence_class == PermanenceClass.MEDIUM
        assert p.metadata.is_active_removal is False

    def test_removals_wrapper(self):
        p = GHGLSRRemovalsPack()
        assert p.metadata.is_active_removal is True
        assert RemovalType.DACCS in p.metadata.allowed_removal_types
        assert RemovalType.AFFORESTATION in p.metadata.allowed_removal_types
        assert RemovalType.BECCS in p.metadata.allowed_removal_types

    def test_storage_wrapper(self):
        p = GHGLSRStoragePack()
        assert p.metadata.is_active_removal is False
        assert p.metadata.permanence_class == PermanenceClass.LONG
        assert (
            p.metadata.biogenic_treatment
            == BiogenicAccountingTreatment.STORAGE_TRACKED
        )


# ---------------------------------------------------------------------------
# Selection + boundary rules
# ---------------------------------------------------------------------------


class TestLSRSelectionRules:
    def test_land_use_removals_family_accepted(self):
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.LAND_USE_REMOVALS.value,
            formula_type=FormulaType.CARBON_BUDGET.value,
        )
        assert GHG_LSR_REMOVALS.selection_rule.accepts(rec) is True

    def test_deprecated_rejected(self):
        rec = SimpleNamespace(
            factor_status="deprecated",
            factor_family=FactorFamily.LAND_USE_REMOVALS.value,
            formula_type=FormulaType.DIRECT_FACTOR.value,
        )
        assert GHG_LSR_LAND_MANAGEMENT.selection_rule.accepts(rec) is False

    def test_preview_accepted(self):
        rec = SimpleNamespace(
            factor_status="preview",
            factor_family=FactorFamily.LAND_USE_REMOVALS.value,
            formula_type=FormulaType.DIRECT_FACTOR.value,
        )
        assert GHG_LSR_REMOVALS.selection_rule.accepts(rec) is True

    def test_wrong_family_rejected(self):
        rec = SimpleNamespace(
            factor_status="certified",
            factor_family=FactorFamily.REFRIGERANT_GWP.value,
            formula_type=FormulaType.DIRECT_FACTOR.value,
        )
        assert GHG_LSR_STORAGE.selection_rule.accepts(rec) is False


class TestLSRBoundaryRules:
    @pytest.mark.parametrize(
        "pack",
        [
            GHG_LSR_LAND_USE_EMISSIONS,
            GHG_LSR_LAND_MANAGEMENT,
            GHG_LSR_REMOVALS,
            GHG_LSR_STORAGE,
        ],
    )
    def test_biogenic_included(self, pack):
        assert pack.boundary_rule.biogenic_treatment == BiogenicTreatment.INCLUDED

    def test_land_use_emissions_market_instruments_na(self):
        assert (
            GHG_LSR_LAND_USE_EMISSIONS.boundary_rule.market_instruments
            == MarketInstrumentTreatment.NOT_APPLICABLE
        )

    def test_land_management_market_instruments_na(self):
        assert (
            GHG_LSR_LAND_MANAGEMENT.boundary_rule.market_instruments
            == MarketInstrumentTreatment.NOT_APPLICABLE
        )

    def test_removals_require_certificate(self):
        assert (
            GHG_LSR_REMOVALS.boundary_rule.market_instruments
            == MarketInstrumentTreatment.REQUIRE_CERTIFICATE
        )

    def test_storage_requires_certificate(self):
        assert (
            GHG_LSR_STORAGE.boundary_rule.market_instruments
            == MarketInstrumentTreatment.REQUIRE_CERTIFICATE
        )

    def test_cradle_to_grave_boundary(self):
        for pack in (
            GHG_LSR_LAND_USE_EMISSIONS,
            GHG_LSR_LAND_MANAGEMENT,
            GHG_LSR_REMOVALS,
            GHG_LSR_STORAGE,
        ):
            assert "cradle_to_grave" in pack.boundary_rule.allowed_boundaries


# ---------------------------------------------------------------------------
# Permanence, reversal-risk and buffer pool
# ---------------------------------------------------------------------------


class TestLSRPermanence:
    def test_permanence_class_enum_values(self):
        assert PermanenceClass.SHORT.value == "short"
        assert PermanenceClass.MEDIUM.value == "medium"
        assert PermanenceClass.LONG.value == "long"

    def test_default_buffer_pool_ordering(self):
        # Shorter permanence => larger buffer.
        assert (
            DEFAULT_BUFFER_POOL[PermanenceClass.SHORT]
            > DEFAULT_BUFFER_POOL[PermanenceClass.MEDIUM]
            > DEFAULT_BUFFER_POOL[PermanenceClass.LONG]
        )

    def test_risk_multiplier_monotonic(self):
        assert (
            RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.LOW]
            < RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.MEDIUM]
            < RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.HIGH]
        )

    def test_compute_buffer_pool_long_low(self):
        assert compute_buffer_pool_pct(PermanenceClass.LONG, ReversalRiskLevel.LOW) == pytest.approx(0.10)

    def test_compute_buffer_pool_medium_medium(self):
        result = compute_buffer_pool_pct(PermanenceClass.MEDIUM, ReversalRiskLevel.MEDIUM)
        assert result == pytest.approx(0.30)

    def test_compute_buffer_pool_short_high(self):
        result = compute_buffer_pool_pct(PermanenceClass.SHORT, ReversalRiskLevel.HIGH)
        assert result == pytest.approx(0.70)

    def test_compute_buffer_pool_capped_at_1(self):
        # Even the worst combo cannot exceed 1.0 (100%).
        val = compute_buffer_pool_pct(PermanenceClass.SHORT, ReversalRiskLevel.HIGH)
        assert 0.0 <= val <= 1.0

    def test_land_use_emissions_buffer_pool_zero(self):
        # Emissions already in the atmosphere — no buffer pool.
        meta = get_lsr_metadata("lsr_land_use_emissions")
        assert meta.buffer_pool_pct == 0.0


# ---------------------------------------------------------------------------
# Removal categorisation
# ---------------------------------------------------------------------------


class TestLSRRemovalCategorisation:
    def test_nature_based_types_registered(self):
        meta = get_lsr_metadata("lsr_removals")
        allowed = set(meta.allowed_removal_types)
        nature_based = {
            RemovalType.AFFORESTATION,
            RemovalType.REFORESTATION,
            RemovalType.PEATLAND_REWETTING,
            RemovalType.SOIL_CARBON_SEQUESTRATION,
            RemovalType.BLUE_CARBON_MANGROVE,
            RemovalType.BLUE_CARBON_SEAGRASS,
            RemovalType.BLUE_CARBON_SALTMARSH,
        }
        assert nature_based.issubset(allowed)

    def test_technology_based_types_registered(self):
        meta = get_lsr_metadata("lsr_removals")
        allowed = set(meta.allowed_removal_types)
        tech_based = {
            RemovalType.BECCS,
            RemovalType.DACCS,
            RemovalType.ENHANCED_ROCK_WEATHERING,
            RemovalType.MINERAL_CARBONATION,
            RemovalType.OCEAN_ALKALINITY_ENHANCEMENT,
        }
        assert tech_based.issubset(allowed)

    def test_biochar_classified_hybrid(self):
        # Biochar spans nature-based (biomass) and tech (pyrolysis) — hybrid.
        meta = get_lsr_metadata("lsr_removals")
        assert RemovalType.BIOCHAR in meta.allowed_removal_types
        assert meta.removal_category == RemovalCategory.HYBRID

    def test_storage_pack_allows_durable_tech_only(self):
        meta = get_lsr_metadata("lsr_storage")
        # Storage is the post-capture lock — only durable tech types apply.
        for rt in meta.allowed_removal_types:
            assert rt in (
                RemovalType.BIOCHAR,
                RemovalType.MINERAL_CARBONATION,
                RemovalType.ENHANCED_ROCK_WEATHERING,
            )


# ---------------------------------------------------------------------------
# Sequestration vs storage distinction
# ---------------------------------------------------------------------------


class TestLSRSequestrationVsStorage:
    def test_removals_is_active(self):
        meta = get_lsr_metadata("lsr_removals")
        assert meta.is_active_removal is True

    def test_storage_is_not_active(self):
        meta = get_lsr_metadata("lsr_storage")
        assert meta.is_active_removal is False

    def test_land_use_emissions_not_active_removal(self):
        # Emissions are never removals.
        meta = get_lsr_metadata("lsr_land_use_emissions")
        assert meta.is_active_removal is False

    def test_land_management_tracks_sequestration(self):
        meta = get_lsr_metadata("lsr_land_management")
        assert (
            meta.biogenic_treatment == BiogenicAccountingTreatment.SEQUESTRATION_TRACKED
        )

    def test_storage_tracks_stock(self):
        meta = get_lsr_metadata("lsr_storage")
        assert meta.biogenic_treatment == BiogenicAccountingTreatment.STORAGE_TRACKED


# ---------------------------------------------------------------------------
# MRV requirements
# ---------------------------------------------------------------------------


class TestLSRMRVRequirements:
    def test_removals_accept_major_standards(self):
        meta = get_lsr_metadata("lsr_removals")
        standards = set(meta.verification_standards)
        assert VerificationStandard.VCS in standards
        assert VerificationStandard.GOLD_STANDARD in standards
        assert VerificationStandard.PURO_EARTH in standards
        assert VerificationStandard.ISOMETRIC in standards
        assert VerificationStandard.CLIMEWORKS_VERIFIED in standards

    def test_removals_include_icvcm_alignment(self):
        meta = get_lsr_metadata("lsr_removals")
        assert VerificationStandard.ICVCM_CCP_APPROVED in meta.verification_standards

    def test_reporting_frequency_defaults_annual(self):
        for name in list_lsr_variants():
            meta = get_lsr_metadata(name)
            assert meta.reporting_frequency == ReportingFrequency.ANNUAL

    def test_storage_uses_puro_and_isometric(self):
        meta = get_lsr_metadata("lsr_storage")
        standards = set(meta.verification_standards)
        assert VerificationStandard.PURO_EARTH in standards
        assert VerificationStandard.ISOMETRIC in standards


# ---------------------------------------------------------------------------
# Boundary scope flags
# ---------------------------------------------------------------------------


class TestLSRBoundaryScope:
    def test_land_use_emissions_iluc_included(self):
        meta = get_lsr_metadata("lsr_land_use_emissions")
        assert meta.iluc_included is True

    def test_land_use_emissions_soc_tracked(self):
        meta = get_lsr_metadata("lsr_land_use_emissions")
        assert meta.soc_tracked is True

    def test_land_management_soc_tracked(self):
        meta = get_lsr_metadata("lsr_land_management")
        assert meta.soc_tracked is True

    def test_removals_soc_tracked(self):
        meta = get_lsr_metadata("lsr_removals")
        assert meta.soc_tracked is True

    def test_storage_does_not_track_soc(self):
        meta = get_lsr_metadata("lsr_storage")
        assert meta.soc_tracked is False


# ---------------------------------------------------------------------------
# Fallback hierarchy + GWP basis
# ---------------------------------------------------------------------------


class TestLSRFallbackAndGWP:
    def test_four_step_fallback(self):
        assert len(LSR_FALLBACK_HIERARCHY) == 4

    def test_project_specific_first(self):
        assert LSR_FALLBACK_HIERARCHY[0].label == "project_specific"

    def test_global_default_last(self):
        assert LSR_FALLBACK_HIERARCHY[-1].label == "global_default"

    def test_gwp_basis_ar6(self):
        for name in list_lsr_variants():
            pack = get_lsr_variant(name)
            assert pack.gwp_basis == "IPCC_AR6_100"

    def test_all_packs_v1_plus(self):
        for name in list_lsr_variants():
            pack = get_lsr_variant(name)
            # v0.2 (Wave 4-G preview window): LSR packs ship at 0.2.x
            # during the preview promotion before methodology sign-off
            # lifts them to v1.0 certified. Accept any populated semver
            # at >= 0.2.0.
            parts = pack.pack_version.split(".")
            assert len(parts) == 3, f"pack.version must be semver; got {pack.pack_version!r}"
            major, minor, patch = (int(p) for p in parts)
            assert (major, minor) >= (0, 2), (
                f"{name}: expected >=0.2.0 during Wave 4-G; got {pack.pack_version}"
            )


# ---------------------------------------------------------------------------
# Audit templates
# ---------------------------------------------------------------------------


class TestLSRAuditTemplates:
    def test_removals_template_mentions_buffer(self):
        assert "buffer" in GHG_LSR_REMOVALS.audit_text_template.lower()

    def test_storage_template_mentions_lifetime(self):
        assert "lifetime" in GHG_LSR_STORAGE.audit_text_template.lower()

    def test_land_use_template_mentions_iluc(self):
        assert "iluc" in GHG_LSR_LAND_USE_EMISSIONS.audit_text_template.lower()

    def test_removals_template_mentions_permanence(self):
        assert "permanence" in GHG_LSR_REMOVALS.audit_text_template.lower()


# ---------------------------------------------------------------------------
# Reporting labels
# ---------------------------------------------------------------------------


class TestLSRReportingLabels:
    @pytest.mark.parametrize(
        "pack",
        [
            GHG_LSR_LAND_USE_EMISSIONS,
            GHG_LSR_LAND_MANAGEMENT,
            GHG_LSR_REMOVALS,
            GHG_LSR_STORAGE,
        ],
    )
    def test_ghg_protocol_lsr_label(self, pack):
        assert "GHG_Protocol_LSR" in pack.reporting_labels

    def test_ipcc_references(self):
        labels = set(GHG_LSR_REMOVALS.reporting_labels)
        assert "IPCC_2006_GL" in labels
        assert "IPCC_2019_Refinement" in labels


# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------


class TestLSRConfigYAML:
    @pytest.fixture(scope="class")
    def config(self):
        cfg_path = (
            Path(__file__).resolve().parents[3]
            / "greenlang"
            / "factors"
            / "data"
            / "method_packs"
            / "ghg_lsr_config.yaml"
        )
        assert cfg_path.is_file(), f"missing {cfg_path}"
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def test_config_has_three_permanence_classes(self, config):
        pc = config["permanence_classes"]
        assert set(pc.keys()) == {"short", "medium", "long"}

    def test_config_permanence_buffer_pools_match_runtime(self, config):
        pc = config["permanence_classes"]
        assert pc["short"]["default_buffer_pool_pct"] == pytest.approx(
            DEFAULT_BUFFER_POOL[PermanenceClass.SHORT]
        )
        assert pc["medium"]["default_buffer_pool_pct"] == pytest.approx(
            DEFAULT_BUFFER_POOL[PermanenceClass.MEDIUM]
        )
        assert pc["long"]["default_buffer_pool_pct"] == pytest.approx(
            DEFAULT_BUFFER_POOL[PermanenceClass.LONG]
        )

    def test_config_risk_multipliers_match_runtime(self, config):
        mult = config["reversal_risk_multiplier"]
        assert mult["low"] == pytest.approx(RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.LOW])
        assert mult["medium"] == pytest.approx(
            RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.MEDIUM]
        )
        assert mult["high"] == pytest.approx(RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.HIGH])

    def test_config_has_four_variants(self, config):
        variants = config["variants"]
        assert len(variants) == 4

    def test_config_variant_names_match_registry(self, config):
        yaml_names = {v["variant_name"] for v in config["variants"]}
        assert yaml_names == set(list_lsr_variants())

    def test_config_removal_types_cover_nature_and_tech(self, config):
        rts = {rt["removal_type"]: rt for rt in config["removal_types"]}
        # Spot-check major categories.
        assert "afforestation" in rts
        assert rts["afforestation"]["category"] == "nature_based"
        assert "daccs" in rts
        assert rts["daccs"]["category"] == "technology_based"
        assert "biochar" in rts
        assert rts["biochar"]["category"] == "hybrid"

    def test_config_removal_types_have_mrv_tier(self, config):
        for rt in config["removal_types"]:
            assert "mrv_tier" in rt
            assert rt["verification_standards"], rt
            assert rt["reporting_frequency"] in (
                "annual",
                "biennial",
                "quinquennial",
                "project_lifecycle",
            )

    def test_config_verification_standards_complete(self, config):
        ref = config["verification_standards_reference"]
        for needed in ("verra_vcs", "gold_standard", "puro_earth", "isometric"):
            assert needed in ref

    def test_config_selection_hierarchy_ranks_monotonic(self, config):
        ranks = [s["rank"] for s in config["selection_hierarchy"]]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1


# ---------------------------------------------------------------------------
# End-to-end resolution
# ---------------------------------------------------------------------------


class TestLSRResolutionSmoke:
    def test_get_pack_returns_same_instance_twice(self):
        a = get_pack("lsr_removals")
        b = get_pack("lsr_removals")
        assert a is b

    def test_variant_is_independent_of_umbrella(self):
        variant = get_pack("lsr_removals")
        umbrella = get_pack(MethodProfile.LAND_REMOVALS)
        assert variant is not umbrella

    def test_all_variants_have_metadata(self):
        for name in list_lsr_variants():
            meta = get_lsr_metadata(name)
            assert isinstance(meta, LSRPackMetadata)
            assert meta.variant_name == name
