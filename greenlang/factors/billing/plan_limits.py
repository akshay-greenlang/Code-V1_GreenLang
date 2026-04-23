# -*- coding: utf-8 -*-
"""
Plan limits for the GreenLang Factors FY27 pricing surface (Agent W4-E / C7).

Hard + soft quota limits per tier. Pulled together from
:mod:`greenlang.factors.billing.skus` (authoritative catalog) plus the
publicly-proposed pricing in the W4-E brief. All numbers here feed the
plan-limits enforcement middleware so the factors API refuses requests
that would exceed a tier's capacity BEFORE they hit Stripe for overage.

Dual limit model
----------------
Every dimension has two thresholds:

* ``soft_limit`` -- request is allowed, Stripe receives a usage record
  and will bill the overage price on the next invoice.
* ``hard_limit`` -- request is refused with HTTP 402 ``quota_exceeded``.
  Community is hard-capped (no overage). Developer Pro / Consulting /
  Platform hard-stop at N x the soft limit as a blast-radius guard.

When ``overage_unit_price_usd`` for a tier is 0 (community, enterprise
contract-inclusive metering) soft and hard collapse to the same value.

Dimensions tracked
------------------
* ``api_calls_per_month`` -- billed via :class:`Meter.API_CALLS`.
* ``batch_rows_per_day``  -- anti-DOS guard; billed per-month via
  :class:`Meter.BATCH_ROWS`.
* ``tenants``             -- multi-tenant seats (Consulting / Platform).
* ``webhooks``            -- outgoing webhook endpoints per tenant.
* ``explain_history_days``-- ``/explain`` payload retention.
* ``private_registry_entries`` -- overlay / private-registry factor
  override count.

Usage
-----
>>> from greenlang.factors.billing.plan_limits import plan_limits_for
>>> limits = plan_limits_for("pro")
>>> limits.api_calls_per_month.soft_limit
100_000
>>> limits.is_hard_capped("api_calls_per_month")
False

Limit enforcement happens in two places:

1. ``greenlang.factors.middleware.rate_limiter.RateLimitMiddleware`` calls
   :func:`check_quota` on every request.
2. ``greenlang.factors.billing.overage.flush_period_overage`` reads
   :attr:`PlanLimits.api_calls_per_month.overage_unit_price_usd` to emit
   Stripe usage records at period end.

NOTE: pricing figures are marked "v1 - subject to CTO/Commercial approval".
The Commercial lead signs off on ``data/commercial/pricing_proposal_v1.md``
before this module ships to staging billing.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional

from greenlang.factors.billing.skus import (
    CATALOG,
    Meter,
    Tier,
)


# ---------------------------------------------------------------------------
# Dual-threshold limit dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualLimit:
    """A single metered dimension with soft + hard thresholds.

    Attributes:
        soft_limit: Included in the base tier fee; exceeding it starts
            Stripe-metered overage. ``None`` means "unlimited / fair-use".
        hard_limit: Absolute cut-off; requests are refused at this
            threshold with HTTP 402. ``None`` means "no hard cap".
        overage_unit_price_usd: Per-unit price charged between
            ``soft_limit`` and ``hard_limit``. ``Decimal('0')`` means
            the tier is hard-capped (soft == hard), no overage sold.
    """

    soft_limit: Optional[int]
    hard_limit: Optional[int]
    overage_unit_price_usd: Decimal

    @property
    def is_hard_capped(self) -> bool:
        """``True`` iff overage is not sold (community tier pattern)."""
        return self.overage_unit_price_usd == Decimal("0") or (
            self.soft_limit is not None
            and self.hard_limit is not None
            and self.soft_limit == self.hard_limit
        )

    def check(self, used: int) -> "QuotaDecision":
        """Classify ``used`` into allowed / overage / refused."""
        if used < 0:
            raise ValueError("used must be non-negative")
        if self.soft_limit is None:
            return QuotaDecision(allowed=True, overage=0, refused=False)
        if used <= self.soft_limit:
            return QuotaDecision(allowed=True, overage=0, refused=False)
        overage = used - self.soft_limit
        if self.hard_limit is not None and used > self.hard_limit:
            return QuotaDecision(
                allowed=False,
                overage=overage,
                refused=True,
            )
        if self.is_hard_capped:
            return QuotaDecision(
                allowed=False,
                overage=overage,
                refused=True,
            )
        return QuotaDecision(allowed=True, overage=overage, refused=False)


@dataclass(frozen=True)
class QuotaDecision:
    """Outcome of a plan-limit check."""

    allowed: bool
    overage: int
    refused: bool


# ---------------------------------------------------------------------------
# Per-tier limit bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanLimits:
    """Full plan-limits bundle for a tier."""

    tier: Tier
    api_calls_per_month: DualLimit
    batch_rows_per_day: DualLimit
    tenants: DualLimit
    webhooks: DualLimit
    explain_history_days: int
    private_registry_entries: Optional[int]
    annual_contract_required: bool

    def is_hard_capped(self, dimension: str) -> bool:
        """Return ``True`` if ``dimension`` is hard-capped (no overage)."""
        limit = getattr(self, dimension, None)
        if isinstance(limit, DualLimit):
            return limit.is_hard_capped
        return True


# ---------------------------------------------------------------------------
# Catalog-derived limit bundles
# ---------------------------------------------------------------------------


def _meter_overage(tier: Tier, meter: Meter) -> Decimal:
    """Pull the overage price from the SKU catalog."""
    cfg = CATALOG.tier(tier).meter(meter)
    return cfg.overage_unit_price_usd if cfg else Decimal("0")


def _meter_included(tier: Tier, meter: Meter) -> Optional[int]:
    """Return the tier's included allowance for ``meter``, or ``None``."""
    cfg = CATALOG.tier(tier).meter(meter)
    return cfg.included_per_month if cfg else None


# Blast-radius multipliers: hard cap = soft_limit * BLAST_RADIUS[tier].
# Community is 1.0 (hard cap == soft cap, i.e. no overage permitted).
# Pro / Consulting / Platform allow up to 10x soft before refusing -- beyond
# that we want a human to approve the scale.
_BLAST_RADIUS: Dict[Tier, float] = {
    Tier.COMMUNITY: 1.0,
    Tier.PRO: 10.0,
    Tier.CONSULTING: 10.0,
    Tier.PLATFORM: 10.0,
    Tier.ENTERPRISE: 100.0,  # effectively "unlimited, billed later"
}


def _compute(
    tier: Tier,
    meter: Meter,
    hard_cap_override: Optional[int] = None,
) -> DualLimit:
    """Build a :class:`DualLimit` for ``(tier, meter)`` from the catalog."""
    included = _meter_included(tier, meter)
    overage = _meter_overage(tier, meter)
    if included is None:
        return DualLimit(
            soft_limit=None, hard_limit=None, overage_unit_price_usd=overage
        )
    if overage == Decimal("0"):
        # Hard cap == soft cap (community).
        return DualLimit(
            soft_limit=included,
            hard_limit=included,
            overage_unit_price_usd=Decimal("0"),
        )
    if hard_cap_override is not None:
        hard = hard_cap_override
    else:
        multiplier = _BLAST_RADIUS.get(tier, 10.0)
        hard = int(included * multiplier)
    return DualLimit(
        soft_limit=included,
        hard_limit=hard,
        overage_unit_price_usd=overage,
    )


# ---------------------------------------------------------------------------
# Tier-specific extras (not in the Meter catalog)
# ---------------------------------------------------------------------------
#
# Webhooks, explain-history retention, and batch-rows-per-day are surface
# concerns that don't map onto a Stripe meter but still need per-tier
# limits so the middleware can enforce them.

_WEBHOOKS_BY_TIER: Dict[Tier, DualLimit] = {
    Tier.COMMUNITY: DualLimit(
        soft_limit=0, hard_limit=0, overage_unit_price_usd=Decimal("0")
    ),
    Tier.PRO: DualLimit(
        soft_limit=5, hard_limit=10, overage_unit_price_usd=Decimal("10.00")
    ),
    Tier.CONSULTING: DualLimit(
        soft_limit=25, hard_limit=50, overage_unit_price_usd=Decimal("5.00")
    ),
    Tier.PLATFORM: DualLimit(
        soft_limit=100, hard_limit=200, overage_unit_price_usd=Decimal("5.00")
    ),
    Tier.ENTERPRISE: DualLimit(
        soft_limit=None, hard_limit=None, overage_unit_price_usd=Decimal("0")
    ),
}

# ``batch_rows_per_day`` is a daily anti-abuse guardrail on top of the
# monthly BATCH_ROWS meter. We allow roughly 1/25 of the monthly allowance
# per day (~business days in a month), hard-cap at 3x that, and defer
# overage billing to the monthly Stripe meter.
_BATCH_ROWS_PER_DAY_FACTOR = 25


def _batch_rows_per_day(tier: Tier) -> DualLimit:
    monthly = _meter_included(tier, Meter.BATCH_ROWS)
    overage = _meter_overage(tier, Meter.BATCH_ROWS)
    if monthly is None:
        return DualLimit(
            soft_limit=None, hard_limit=None, overage_unit_price_usd=overage
        )
    soft = max(1, monthly // _BATCH_ROWS_PER_DAY_FACTOR)
    hard = soft * 3
    return DualLimit(
        soft_limit=soft,
        hard_limit=hard,
        overage_unit_price_usd=overage,
    )


_EXPLAIN_HISTORY_DAYS: Dict[Tier, int] = {
    Tier.COMMUNITY: 7,
    Tier.PRO: 90,
    Tier.CONSULTING: 365,
    Tier.PLATFORM: 365,
    Tier.ENTERPRISE: 2555,  # 7 years (SOC2 / regulated retention)
}


# ---------------------------------------------------------------------------
# Public catalog
# ---------------------------------------------------------------------------


def _build_plan_limits() -> Dict[Tier, PlanLimits]:
    bundles: Dict[Tier, PlanLimits] = {}
    for tier in Tier:
        cfg = CATALOG.tier(tier)
        bundles[tier] = PlanLimits(
            tier=tier,
            api_calls_per_month=_compute(tier, Meter.API_CALLS),
            batch_rows_per_day=_batch_rows_per_day(tier),
            tenants=_compute(tier, Meter.TENANTS),
            webhooks=_WEBHOOKS_BY_TIER[tier],
            explain_history_days=_EXPLAIN_HISTORY_DAYS[tier],
            private_registry_entries=cfg.private_registry_entries,
            annual_contract_required=cfg.annual_contract_required,
        )
    return bundles


PLAN_LIMITS: Dict[Tier, PlanLimits] = _build_plan_limits()


def plan_limits_for(tier) -> PlanLimits:
    """Look up :class:`PlanLimits` by :class:`Tier` or string name."""
    if isinstance(tier, Tier):
        return PLAN_LIMITS[tier]
    return PLAN_LIMITS[Tier(str(tier).lower().strip())]


# ---------------------------------------------------------------------------
# Enforcement helpers
# ---------------------------------------------------------------------------


def check_quota(
    tier,
    dimension: str,
    used: int,
) -> QuotaDecision:
    """Check ``used`` against the plan's ``dimension`` limit.

    Args:
        tier: Tier enum or string (community, pro, consulting, platform,
            enterprise).
        dimension: One of the ``DualLimit``-typed attributes of
            :class:`PlanLimits`: ``api_calls_per_month``,
            ``batch_rows_per_day``, ``tenants``, ``webhooks``.
        used: Current usage count to test.

    Returns:
        :class:`QuotaDecision` describing whether the request is allowed,
        the overage qty to report to Stripe, and whether the hard cap
        would be breached.

    Raises:
        KeyError: if ``dimension`` is not a known :class:`DualLimit`
            attribute.
        ValueError: if ``used`` is negative.
    """
    limits = plan_limits_for(tier)
    limit = getattr(limits, dimension, None)
    if not isinstance(limit, DualLimit):
        raise KeyError(
            f"Unknown plan-limit dimension {dimension!r}; "
            "must be one of: api_calls_per_month, batch_rows_per_day, "
            "tenants, webhooks"
        )
    return limit.check(used)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def _assert_invariants() -> None:
    """Fire invariants at import so a typo breaks CI not prod."""
    # Every tier has a PlanLimits bundle.
    for tier in Tier:
        assert tier in PLAN_LIMITS, f"no PlanLimits for {tier}"

    # Community MUST be hard-capped.
    assert PLAN_LIMITS[Tier.COMMUNITY].api_calls_per_month.is_hard_capped, (
        "Community tier api_calls must be hard-capped (no overage allowed)"
    )

    # Hard limit >= soft limit when both are set.
    for tier, bundle in PLAN_LIMITS.items():
        for dim in ("api_calls_per_month", "batch_rows_per_day", "tenants", "webhooks"):
            lim = getattr(bundle, dim)
            if lim.soft_limit is not None and lim.hard_limit is not None:
                assert lim.hard_limit >= lim.soft_limit, (
                    f"{tier} {dim}: hard < soft ({lim.hard_limit} < "
                    f"{lim.soft_limit})"
                )

    # explain_history_days strictly increases with tier.
    # (community < pro < consulting <= platform < enterprise)
    assert (
        _EXPLAIN_HISTORY_DAYS[Tier.COMMUNITY]
        < _EXPLAIN_HISTORY_DAYS[Tier.PRO]
        < _EXPLAIN_HISTORY_DAYS[Tier.CONSULTING]
        <= _EXPLAIN_HISTORY_DAYS[Tier.PLATFORM]
        < _EXPLAIN_HISTORY_DAYS[Tier.ENTERPRISE]
    )


_assert_invariants()


__all__ = [
    "DualLimit",
    "PlanLimits",
    "QuotaDecision",
    "PLAN_LIMITS",
    "plan_limits_for",
    "check_quota",
]
