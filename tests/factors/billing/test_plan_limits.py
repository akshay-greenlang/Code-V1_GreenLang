# -*- coding: utf-8 -*-
"""
Unit tests for :mod:`greenlang.factors.billing.plan_limits` (Agent W4-E / C7).

Coverage:
    * Every tier produces a :class:`PlanLimits` bundle.
    * Community is hard-capped (no overage allowed).
    * Paid tiers (Pro, Consulting, Platform) allow soft-limit overage but
      refuse once ``hard_limit`` is exceeded.
    * ``check_quota`` raises for unknown dimensions and negative usage.
    * Explain-history retention strictly increases with tier.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.factors.billing.plan_limits import (
    DualLimit,
    PLAN_LIMITS,
    PlanLimits,
    QuotaDecision,
    check_quota,
    plan_limits_for,
)
from greenlang.factors.billing.skus import Tier


class TestPlanLimitsCatalog:
    def test_every_tier_has_a_bundle(self) -> None:
        """All 5 tiers must have a PlanLimits bundle."""
        for tier in Tier:
            assert tier in PLAN_LIMITS
            assert isinstance(PLAN_LIMITS[tier], PlanLimits)

    def test_plan_limits_for_accepts_string(self) -> None:
        assert plan_limits_for("pro").tier == Tier.PRO
        assert plan_limits_for(Tier.PLATFORM).tier == Tier.PLATFORM

    def test_plan_limits_for_invalid(self) -> None:
        with pytest.raises(ValueError):
            plan_limits_for("bogus_tier")

    def test_explain_history_increases_with_tier(self) -> None:
        """Retention must strictly increase community -> pro -> consulting."""
        assert (
            PLAN_LIMITS[Tier.COMMUNITY].explain_history_days
            < PLAN_LIMITS[Tier.PRO].explain_history_days
            < PLAN_LIMITS[Tier.CONSULTING].explain_history_days
            < PLAN_LIMITS[Tier.ENTERPRISE].explain_history_days
        )


class TestCommunityHardCap:
    def test_community_is_hard_capped(self) -> None:
        limits = plan_limits_for(Tier.COMMUNITY)
        assert limits.api_calls_per_month.is_hard_capped is True

    def test_community_refuses_overage(self) -> None:
        """Community tier must refuse requests above the 1,000/mo cap."""
        limits = plan_limits_for(Tier.COMMUNITY)
        cap = limits.api_calls_per_month.soft_limit
        assert cap is not None
        # Within cap: allowed.
        decision = limits.api_calls_per_month.check(cap)
        assert decision.allowed is True
        assert decision.refused is False
        # Over cap: refused (hard cap == soft cap for community).
        decision = limits.api_calls_per_month.check(cap + 1)
        assert decision.allowed is False
        assert decision.refused is True

    def test_community_webhook_zero(self) -> None:
        limits = plan_limits_for(Tier.COMMUNITY)
        assert limits.webhooks.soft_limit == 0
        assert limits.webhooks.hard_limit == 0
        assert limits.webhooks.check(0).allowed is True
        assert limits.webhooks.check(1).refused is True


class TestPaidTierOverage:
    @pytest.mark.parametrize(
        "tier",
        [Tier.PRO, Tier.CONSULTING, Tier.PLATFORM],
    )
    def test_paid_tier_permits_overage(self, tier: Tier) -> None:
        """Paid tiers should allow overage between soft and hard caps."""
        limits = plan_limits_for(tier)
        dual = limits.api_calls_per_month
        assert dual.soft_limit is not None and dual.hard_limit is not None
        # Overage > 0, not yet hard.
        over = dual.soft_limit + 1
        decision = dual.check(over)
        assert decision.allowed is True
        assert decision.overage == 1

    @pytest.mark.parametrize(
        "tier",
        [Tier.PRO, Tier.CONSULTING, Tier.PLATFORM],
    )
    def test_paid_tier_refuses_past_hard_cap(self, tier: Tier) -> None:
        """Once hard_limit is exceeded, request is refused."""
        limits = plan_limits_for(tier)
        dual = limits.api_calls_per_month
        assert dual.hard_limit is not None
        decision = dual.check(dual.hard_limit + 1)
        assert decision.refused is True
        assert decision.allowed is False

    def test_pro_overage_price_nonzero(self) -> None:
        """Pro must carry a positive overage price; community must not."""
        assert (
            plan_limits_for(Tier.PRO).api_calls_per_month.overage_unit_price_usd
            > Decimal("0")
        )
        assert (
            plan_limits_for(
                Tier.COMMUNITY
            ).api_calls_per_month.overage_unit_price_usd
            == Decimal("0")
        )


class TestEnterpriseFairUse:
    def test_enterprise_tenants_unlimited(self) -> None:
        """Enterprise has no hard tenant cap (contract-negotiated)."""
        limits = plan_limits_for(Tier.ENTERPRISE)
        # Enterprise's multiplier is 100x on api_calls so hard is still set;
        # webhooks are None (unlimited).
        assert limits.webhooks.soft_limit is None
        assert limits.webhooks.hard_limit is None
        # Any webhook count is allowed.
        assert limits.webhooks.check(10_000).allowed is True


class TestCheckQuotaAPI:
    def test_check_quota_returns_decision(self) -> None:
        decision = check_quota("pro", "api_calls_per_month", 1)
        assert isinstance(decision, QuotaDecision)
        assert decision.allowed is True

    def test_check_quota_unknown_dimension(self) -> None:
        with pytest.raises(KeyError):
            check_quota("pro", "does_not_exist", 1)

    def test_check_quota_negative_refused(self) -> None:
        with pytest.raises(ValueError):
            check_quota("pro", "api_calls_per_month", -1)

    def test_check_quota_string_tier(self) -> None:
        """String tier names are accepted."""
        decision = check_quota("community", "api_calls_per_month", 500)
        assert decision.allowed is True


class TestInvariants:
    def test_hard_cap_geq_soft_cap(self) -> None:
        """Wherever both are set, hard_limit >= soft_limit."""
        for tier, bundle in PLAN_LIMITS.items():
            for dim in (
                "api_calls_per_month",
                "batch_rows_per_day",
                "tenants",
                "webhooks",
            ):
                lim: DualLimit = getattr(bundle, dim)
                if lim.soft_limit is not None and lim.hard_limit is not None:
                    assert lim.hard_limit >= lim.soft_limit, (
                        f"{tier} {dim}: hard < soft"
                    )

    def test_community_annual_contract_not_required(self) -> None:
        assert (
            plan_limits_for(Tier.COMMUNITY).annual_contract_required is False
        )
        assert plan_limits_for(Tier.PRO).annual_contract_required is False
        assert (
            plan_limits_for(Tier.ENTERPRISE).annual_contract_required is True
        )
