# -*- coding: utf-8 -*-
"""Tests for tier-based factor visibility enforcement (F030)."""

from __future__ import annotations

import pytest

from greenlang.factors.tier_enforcement import (
    Tier,
    TierVisibility,
    enforce_tier_on_request,
    factor_visible_for_tier,
    filter_factors_by_tier,
    resolve_tier,
)


# ---- Tier enum ----

def test_tier_values():
    assert Tier.COMMUNITY.value == "community"
    assert Tier.PRO.value == "pro"
    assert Tier.ENTERPRISE.value == "enterprise"
    assert Tier.INTERNAL.value == "internal"


# ---- resolve_tier ----

def test_resolve_tier_none():
    assert resolve_tier(None) == "community"


def test_resolve_tier_empty_dict():
    assert resolve_tier({}) == "community"


def test_resolve_tier_valid():
    assert resolve_tier({"tier": "pro"}) == "pro"
    assert resolve_tier({"tier": "enterprise"}) == "enterprise"
    assert resolve_tier({"tier": "internal"}) == "internal"


def test_resolve_tier_case_insensitive():
    assert resolve_tier({"tier": "PRO"}) == "pro"
    assert resolve_tier({"tier": "Enterprise"}) == "enterprise"


def test_resolve_tier_unknown_defaults():
    assert resolve_tier({"tier": "platinum"}) == "community"


# ---- TierVisibility.from_tier ----

def test_community_visibility():
    tv = TierVisibility.from_tier("community")
    assert not tv.include_preview
    assert not tv.include_connector
    assert not tv.include_deprecated
    assert not tv.audit_bundle_allowed
    assert not tv.bulk_export_allowed
    assert tv.max_export_rows == 1_000


def test_pro_visibility():
    tv = TierVisibility.from_tier("pro")
    assert tv.include_preview
    assert not tv.include_connector
    assert not tv.include_deprecated
    assert not tv.audit_bundle_allowed
    assert tv.bulk_export_allowed
    assert tv.max_export_rows == 10_000


def test_enterprise_visibility():
    tv = TierVisibility.from_tier("enterprise")
    assert tv.include_preview
    assert tv.include_connector
    assert not tv.include_deprecated
    assert tv.audit_bundle_allowed
    assert tv.bulk_export_allowed
    assert tv.max_export_rows == 100_000


def test_internal_visibility():
    tv = TierVisibility.from_tier("internal")
    assert tv.include_preview
    assert tv.include_connector
    assert tv.include_deprecated
    assert tv.audit_bundle_allowed
    assert tv.bulk_export_allowed
    assert tv.max_export_rows == 0  # unlimited


# ---- factor_visible_for_tier ----

def test_certified_always_visible():
    for t in ("community", "pro", "enterprise", "internal"):
        tv = TierVisibility.from_tier(t)
        assert factor_visible_for_tier("certified", tv)


def test_preview_not_visible_community():
    tv = TierVisibility.from_tier("community")
    assert not factor_visible_for_tier("preview", tv)


def test_preview_visible_pro():
    tv = TierVisibility.from_tier("pro")
    assert factor_visible_for_tier("preview", tv)


def test_connector_not_visible_pro():
    tv = TierVisibility.from_tier("pro")
    assert not factor_visible_for_tier("connector_only", tv)


def test_connector_visible_enterprise():
    tv = TierVisibility.from_tier("enterprise")
    assert factor_visible_for_tier("connector_only", tv)


def test_deprecated_only_internal():
    for t in ("community", "pro", "enterprise"):
        tv = TierVisibility.from_tier(t)
        assert not factor_visible_for_tier("deprecated", tv)
    tv = TierVisibility.from_tier("internal")
    assert factor_visible_for_tier("deprecated", tv)


# ---- enforce_tier_on_request ----

def test_enforce_clamps_preview_for_community():
    tv = enforce_tier_on_request({"tier": "community"}, requested_preview=True)
    assert not tv.include_preview


def test_enforce_allows_preview_for_pro():
    tv = enforce_tier_on_request({"tier": "pro"}, requested_preview=True)
    assert tv.include_preview


def test_enforce_clamps_connector_for_pro():
    tv = enforce_tier_on_request({"tier": "pro"}, requested_connector=True)
    assert not tv.include_connector


def test_enforce_allows_connector_for_enterprise():
    tv = enforce_tier_on_request(
        {"tier": "enterprise"}, requested_preview=True, requested_connector=True
    )
    assert tv.include_preview
    assert tv.include_connector


def test_enforce_no_request_flags():
    tv = enforce_tier_on_request({"tier": "enterprise"})
    assert not tv.include_preview
    assert not tv.include_connector


# ---- filter_factors_by_tier ----

def test_filter_dicts_community():
    factors = [
        {"factor_id": "1", "factor_status": "certified"},
        {"factor_id": "2", "factor_status": "preview"},
        {"factor_id": "3", "factor_status": "connector_only"},
        {"factor_id": "4", "factor_status": "deprecated"},
    ]
    tv = TierVisibility.from_tier("community")
    result = filter_factors_by_tier(factors, tv)
    assert len(result) == 1
    assert result[0]["factor_id"] == "1"


def test_filter_dicts_pro():
    factors = [
        {"factor_id": "1", "factor_status": "certified"},
        {"factor_id": "2", "factor_status": "preview"},
        {"factor_id": "3", "factor_status": "connector_only"},
    ]
    tv = TierVisibility.from_tier("pro")
    tv.include_preview = True
    result = filter_factors_by_tier(factors, tv)
    assert len(result) == 2


def test_filter_dicts_enterprise():
    factors = [
        {"factor_id": "1", "factor_status": "certified"},
        {"factor_id": "2", "factor_status": "preview"},
        {"factor_id": "3", "factor_status": "connector_only"},
        {"factor_id": "4", "factor_status": "deprecated"},
    ]
    tv = TierVisibility.from_tier("enterprise")
    tv.include_preview = True
    tv.include_connector = True
    result = filter_factors_by_tier(factors, tv)
    assert len(result) == 3  # no deprecated


def test_filter_empty():
    tv = TierVisibility.from_tier("enterprise")
    assert filter_factors_by_tier([], tv) == []


def test_filter_none_status_defaults_certified():
    factors = [{"factor_id": "1", "factor_status": None}]
    tv = TierVisibility.from_tier("community")
    result = filter_factors_by_tier(factors, tv)
    assert len(result) == 1
