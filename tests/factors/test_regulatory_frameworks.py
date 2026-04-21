# -*- coding: utf-8 -*-
"""Tests for the regulatory-framework tagger."""

from __future__ import annotations

import pytest

from greenlang.factors.mapping.regulatory_frameworks import (
    BUILTIN_RULES,
    FrameworkApplicability,
    FrameworkIndex,
    FrameworkScope,
    RegulatoryFramework,
    tag_factor,
    tag_factor_batch,
)


def _tags(**factor_attrs):
    return set(tag_factor(factor_attrs))


class TestScope2Disambiguation:
    def test_market_based_method_profile_only_tags_market(self):
        tags = _tags(
            scope=2,
            activity_family="electricity",
            geography="FR",
            method_profile="ghg_corporate_scope2_market",
        )
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE2_MB.value in tags
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE2_LB.value not in tags

    def test_location_based_method_profile_only_tags_location(self):
        tags = _tags(
            scope=2,
            activity_family="electricity",
            geography="FR",
            method_profile="ghg_corporate_scope2_location",
        )
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE2_LB.value in tags
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE2_MB.value not in tags

    def test_no_method_profile_matches_both_variants(self):
        """Without a declared method profile we cannot disambiguate; both rules apply."""
        tags = _tags(scope=2, activity_family="electricity", geography="FR")
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE2_LB.value in tags
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE2_MB.value in tags


class TestJurisdictionFilter:
    def test_cbam_applies_to_eu_member_state(self):
        tags = _tags(scope=1, activity_family="process", geography="DE", method_profile="eu_cbam")
        assert RegulatoryFramework.CBAM.value in tags

    def test_cbam_rejects_non_eu_geography(self):
        tags = _tags(scope=1, activity_family="process", geography="US", method_profile="eu_cbam")
        assert RegulatoryFramework.CBAM.value not in tags

    def test_cbam_accepts_explicit_eu_tag(self):
        tags = _tags(scope=1, activity_family="process", geography="EU")
        assert RegulatoryFramework.CBAM.value in tags

    def test_ca_sb253_only_for_us_entities(self):
        tags_ca = _tags(scope=1, activity_family="stationary_combustion", geography="US-CA")
        tags_fr = _tags(scope=1, activity_family="stationary_combustion", geography="FR")
        assert RegulatoryFramework.CA_SB253.value in tags_ca
        assert RegulatoryFramework.CA_SB253.value not in tags_fr

    def test_missing_geography_rejects_jurisdictional_rules(self):
        tags = _tags(scope=1, activity_family="process", method_profile="eu_cbam")
        assert RegulatoryFramework.CBAM.value not in tags


class TestUniversalFrameworks:
    def test_ghg_corporate_always_applies_when_scope_known(self):
        tags = _tags(scope=1, activity_family="stationary_combustion")
        assert RegulatoryFramework.GHG_PROTOCOL_CORPORATE.value in tags

    def test_iso_14064_1_applies_to_all_scopes(self):
        for sc in (1, 2, 3):
            tags = _tags(scope=sc, activity_family="electricity")
            assert RegulatoryFramework.ISO_14064_1.value in tags

    def test_sbti_universal(self):
        tags = _tags(scope=3, activity_family="purchased_goods")
        assert RegulatoryFramework.SBTi.value in tags


class TestProductAndFinance:
    def test_iso_14067_product_carbon(self):
        tags = _tags(scope="product", activity_family="processing")
        assert RegulatoryFramework.ISO_14067.value in tags

    def test_pcaf_financed_emissions(self):
        tags = _tags(factor_family="finance_proxy", method_profile="pcaf", geography="US")
        assert RegulatoryFramework.PCAF.value in tags
        assert RegulatoryFramework.GHG_PROTOCOL_CORPORATE.value not in tags

    def test_iso_14083_freight(self):
        tags = _tags(scope=3, activity_family="freight", method_profile="freight_iso_14083")
        assert RegulatoryFramework.ISO_14083.value in tags


class TestInputNormalisation:
    def test_scope_as_string(self):
        tags = _tags(scope="scope_1", activity_family="combustion")
        assert RegulatoryFramework.GHG_PROTOCOL_CORPORATE.value in tags

    def test_activity_tags_list(self):
        tags = _tags(scope=3, activity_tags=["freight", "road"])
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE3.value in tags

    def test_dataclass_like_object(self):
        class Factor:
            scope = 1
            activity_family = "refrigerants"
            geography = "US"
            method_profile = None
        tags = set(tag_factor(Factor()))
        assert RegulatoryFramework.GHG_PROTOCOL_CORPORATE.value in tags


class TestExtraRules:
    def test_caller_can_inject_custom_rule(self):
        rule = FrameworkApplicability(
            framework=RegulatoryFramework.CDP,  # reuse enum slot
            scopes=frozenset({FrameworkScope.ENTITY}),
            activity_families=frozenset({"custom_family"}),
        )
        tags = tag_factor(
            {"scope": "entity", "activity_family": "custom_family"},
            extra_rules=[rule],
        )
        assert RegulatoryFramework.CDP.value in tags


class TestBatchAndIndex:
    def test_batch_tagging(self):
        factors = [
            {"scope": 1, "activity_family": "combustion"},
            {"scope": 3, "activity_family": "freight", "method_profile": "freight_iso_14083"},
        ]
        batch = tag_factor_batch(factors)
        assert len(batch) == 2
        assert RegulatoryFramework.GHG_PROTOCOL_CORPORATE.value in batch[0]
        assert RegulatoryFramework.ISO_14083.value in batch[1]

    def test_framework_index_builds_and_queries(self):
        factors = [
            {"factor_id": "f1", "scope": 1, "activity_family": "combustion"},
            {"factor_id": "f2", "scope": 2, "activity_family": "electricity",
             "method_profile": "ghg_corporate_scope2_market"},
            {"factor_id": "f3", "scope": 3, "activity_family": "freight",
             "method_profile": "freight_iso_14083"},
            {"factor_id": "f4", "scope": 1, "activity_family": "process",
             "geography": "DE", "method_profile": "eu_cbam"},
        ]
        index = FrameworkIndex.build(factors)
        # CBAM picks up only the DE process factor.
        assert index.factors_for(RegulatoryFramework.CBAM.value) == ["f4"]
        # ISO 14083 picks up only the freight factor.
        assert index.factors_for(RegulatoryFramework.ISO_14083.value) == ["f3"]
        # GHG Corporate picks up all four.
        assert len(index.factors_for(RegulatoryFramework.GHG_PROTOCOL_CORPORATE.value)) == 4
        # factor-id reverse lookup works.
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE2_MB.value in index.frameworks_for("f2")
        # summary totals are consistent.
        assert index.factor_count(RegulatoryFramework.CBAM.value) == 1


class TestBuiltinRuleCatalog:
    def test_every_framework_has_at_least_one_rule(self):
        covered = {r.framework for r in BUILTIN_RULES}
        # Expect near-complete coverage; allow leaving IFRS_S2 etc. implicit
        # only if they are listed separately.
        assert RegulatoryFramework.CBAM in covered
        assert RegulatoryFramework.ESRS_E1 in covered
        assert RegulatoryFramework.TCFD in covered
        assert RegulatoryFramework.PCAF in covered
        assert RegulatoryFramework.CA_SB253 in covered

    def test_all_rules_have_valid_framework_enum(self):
        for rule in BUILTIN_RULES:
            assert isinstance(rule.framework, RegulatoryFramework)
