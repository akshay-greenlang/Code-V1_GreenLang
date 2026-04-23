# -*- coding: utf-8 -*-
"""
Tests for the public 4-SKU catalog used by the FY27 Pricing Page.

Mirrors the contract in `greenlang/factors/billing/skus.py` and the
public surface consumed by `greenlang/factors/billing/api.py`.

These tests are intentionally narrow: they assert the *shape* of
`get_skus()` (what the Pricing Page renders) rather than re-asserting
the rich internal `CATALOG` (which already has its own tests in
`tests/factors/billing/test_skus.py`).
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.factors.billing.skus import (
    PUBLIC_SKUS,
    Tier,
    get_skus,
    get_sku_by_id,
)


EXPECTED_PLAN_IDS: tuple = (
    "community",
    "developer_pro",
    "consulting_platform",
    "enterprise",
)


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------


def test_get_skus_returns_exactly_four_plans() -> None:
    """The Pricing Page surface MUST be exactly four SKUs."""
    plans = get_skus()
    assert isinstance(plans, list)
    assert len(plans) == 4


def test_get_skus_returns_canonical_plan_ids_in_order() -> None:
    """Plan ids must match the agreed order Community -> Pro -> Consulting -> Enterprise."""
    plans = get_skus()
    assert tuple(p["plan_id"] for p in plans) == EXPECTED_PLAN_IDS


def test_get_skus_returns_fresh_copies() -> None:
    """`get_skus()` must hand back new dicts so callers cannot mutate the catalog."""
    a = get_skus()
    b = get_skus()
    a[0]["display_name"] = "MUTATED"
    assert b[0]["display_name"] != "MUTATED"
    assert PUBLIC_SKUS[0]["display_name"] != "MUTATED"


@pytest.mark.parametrize("plan_id", EXPECTED_PLAN_IDS)
def test_get_sku_by_id_round_trips(plan_id: str) -> None:
    sku = get_sku_by_id(plan_id)
    assert sku is not None
    assert sku["plan_id"] == plan_id


def test_get_sku_by_id_unknown_returns_none() -> None:
    assert get_sku_by_id("nonexistent") is None
    assert get_sku_by_id("") is None


def test_get_sku_by_id_is_case_insensitive() -> None:
    assert get_sku_by_id("DEVELOPER_PRO") is not None
    assert get_sku_by_id("  community  ") is not None


# ---------------------------------------------------------------------------
# Per-SKU semantics (the "what does each plan get" contract)
# ---------------------------------------------------------------------------


def test_community_is_free_self_serve_and_open_class_only() -> None:
    sku = get_sku_by_id("community")
    assert sku is not None
    assert sku["price_usd_monthly"] == "0.00"
    assert sku["self_serve"] is True
    assert sku["contact_sales"] is False
    assert sku["license_classes"] == ["redistribute_open"]
    assert sku["included_premium_packs"] == []
    assert sku["oem_redistribution"] is False
    # Rate-limit caps
    assert sku["rate_limit"]["requests_per_minute"] == 60
    assert sku["rate_limit"]["requests_per_month_included"] == 1_000


def test_developer_pro_is_self_serve_with_overage() -> None:
    sku = get_sku_by_id("developer_pro")
    assert sku is not None
    assert sku["self_serve"] is True
    assert sku["contact_sales"] is False
    assert Decimal(sku["price_usd_monthly"]) == Decimal("499.00")
    assert sku["overage_unit_price_usd"] == "0.001"
    assert sku["rate_limit"]["requests_per_minute"] == 1_000
    assert sku["rate_limit"]["requests_per_month_included"] == 100_000
    assert "redistribute_restricted" in sku["license_classes"]
    assert sku["oem_redistribution"] is False


def test_consulting_platform_includes_sub_tenants_and_packs() -> None:
    sku = get_sku_by_id("consulting_platform")
    assert sku is not None
    assert sku["self_serve"] is True
    assert sku["contact_sales"] is False
    assert sku["included_sub_tenants"] == 5
    assert "redistribute_restricted" in sku["license_classes"]
    assert "customer_private" in sku["license_classes"]
    assert len(sku["included_premium_packs"]) >= 1
    assert sku["oem_redistribution"] is False


def test_enterprise_is_contact_sales_with_oem_rights() -> None:
    sku = get_sku_by_id("enterprise")
    assert sku is not None
    assert sku["self_serve"] is False
    assert sku["contact_sales"] is True
    assert sku["price_usd_monthly"] is None
    assert sku["oem_redistribution"] is True
    # Enterprise must include every redistribution class.
    expected_classes = {
        "redistribute_open",
        "redistribute_restricted",
        "connector_only",
        "customer_private",
        "internal_only",
    }
    assert expected_classes.issubset(set(sku["license_classes"]))


# ---------------------------------------------------------------------------
# Internal-tier mapping integrity
# ---------------------------------------------------------------------------


def test_each_public_sku_maps_to_a_known_internal_tier() -> None:
    """Every public SKU must reference a Tier enum value the catalog knows about."""
    for sku in get_skus():
        # ValueError if the tier_name is not a Tier enum member.
        Tier(sku["tier_name"])


@pytest.mark.parametrize(
    "plan_id,expected_tier",
    [
        ("community", Tier.COMMUNITY),
        ("developer_pro", Tier.PRO),
        ("consulting_platform", Tier.PLATFORM),
        ("enterprise", Tier.ENTERPRISE),
    ],
)
def test_plan_id_maps_to_expected_tier(plan_id: str, expected_tier: Tier) -> None:
    sku = get_sku_by_id(plan_id)
    assert sku is not None
    assert Tier(sku["tier_name"]) == expected_tier


# ---------------------------------------------------------------------------
# Smoke: the FastAPI router model still consumes our SKU dicts
# ---------------------------------------------------------------------------


def test_skus_are_compatible_with_planview_pydantic_model() -> None:
    """The FastAPI endpoint serialises every SKU through `PlanView`.

    If a future edit to `get_skus()` removes a required field, this test
    fires before the API does.
    """
    from greenlang.factors.billing.api import PlanView

    for sku in get_skus():
        plan = PlanView(**sku)
        assert plan.plan_id == sku["plan_id"]
        assert plan.display_name == sku["display_name"]
