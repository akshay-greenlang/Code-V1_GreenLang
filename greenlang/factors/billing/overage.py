# -*- coding: utf-8 -*-
"""
Period-end overage billing for the FY27 Factors commercial surface.

Read the meter counters populated by the request-time middleware, diff
them against each tier's included allowance, and emit Stripe usage
records for the overage quantities.

This module is deliberately thin. It calls into:

* :mod:`greenlang.factors.billing.aggregator` (reads the SQLite usage
  sink populated by ``AuthMeteringMiddleware``),
* :mod:`greenlang.factors.billing.skus` (authoritative tier <-> meter
  mapping), and
* :mod:`greenlang.factors.billing.stripe_provider` (hits the Stripe
  usage-record endpoint, or no-ops when ``STRIPE_API_KEY`` is unset).

Typical cron wiring (see ``deployment/cron/factors_overage.yaml``):

    # Every night at 00:05 UTC, flush yesterday's overage.
    from greenlang.factors.billing.overage import flush_period_overage
    result = flush_period_overage(tenant_id="tenant_acme", tier="pro")

The function is **idempotent per (tenant, subscription_item, period)** --
the Stripe ``Idempotency-Key`` we send is derived from those three
inputs, so running the cron twice in a row will not double-bill.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.factors.billing.aggregator import UsageAggregator
from greenlang.factors.billing.plan_limits import plan_limits_for
from greenlang.factors.billing.skus import (
    CATALOG,
    Meter,
    Tier,
    meter_price_id,
    overage_price,
)
from greenlang.factors.billing.stripe_provider import (
    StripeApiError,
    StripeBillingProvider,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OverageFlushResult:
    """Outcome of one :func:`flush_period_overage` call."""

    tenant_id: str
    tier: str
    subscription_id: Optional[str]
    period_start: str
    period_end: str
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    total_amount_usd: str = "0.00"
    stripe_usage_records: List[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "tier": self.tier,
            "subscription_id": self.subscription_id,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "line_items": list(self.line_items),
            "total_amount_usd": self.total_amount_usd,
            "stripe_usage_records": list(self.stripe_usage_records),
            "dry_run": self.dry_run,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _period_bounds() -> tuple[str, str]:
    """Return (start, end) ISO-8601 strings for the current calendar month."""
    now = datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start.isoformat(), now.isoformat()


def _idempotency_key(
    tenant_id: str,
    subscription_id: str,
    meter: Meter,
    period_end: str,
) -> str:
    """Stable Stripe idempotency key for one usage-record write."""
    payload = f"{tenant_id}|{subscription_id}|{meter.value}|{period_end}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def flush_period_overage(
    *,
    tenant_id: str,
    tier: str,
    subscription_id: Optional[str] = None,
    aggregator: Optional[UsageAggregator] = None,
    provider: Optional[StripeBillingProvider] = None,
    dry_run: bool = False,
    api_key_hash: Optional[str] = None,
) -> OverageFlushResult:
    """Read meters for ``tenant_id`` and push overage to Stripe.

    Args:
        tenant_id: GreenLang tenant id whose usage we're flushing.
        tier: Tier name (community, pro, consulting, platform, enterprise).
        subscription_id: Stripe subscription id the overage is billed
            against. Optional when ``dry_run=True`` or for Community
            (which is hard-capped and cannot accrue overage).
        aggregator: Optional :class:`UsageAggregator`; default constructs
            one pointed at ``GL_FACTORS_USAGE_SQLITE`` (falls back to
            ``/tmp/factors_usage.db``).
        provider: Optional :class:`StripeBillingProvider`; default loads
            from the environment.
        dry_run: When True, compute line items but do NOT call Stripe.
        api_key_hash: Usage sink key. When omitted defaults to a hash
            derived from ``tenant_id`` so the sink lookup is stable.

    Returns:
        :class:`OverageFlushResult` describing the emitted line items.

    Raises:
        ValueError: for unknown tier names.
    """
    try:
        resolved_tier = Tier(tier.lower().strip())
    except ValueError as exc:
        raise ValueError(
            f"Unknown tier {tier!r}; must be one of "
            f"{[t.value for t in Tier]}"
        ) from exc

    plan = plan_limits_for(resolved_tier)

    # Community hard-capped -- never bills.
    if plan.api_calls_per_month.is_hard_capped and resolved_tier == Tier.COMMUNITY:
        logger.info(
            "Skipping overage flush for community tier tenant=%s (hard-capped)",
            tenant_id,
        )
        start, end = _period_bounds()
        return OverageFlushResult(
            tenant_id=tenant_id,
            tier=resolved_tier.value,
            subscription_id=subscription_id,
            period_start=start,
            period_end=end,
            dry_run=dry_run,
        )

    aggregator = aggregator or UsageAggregator(
        os.getenv("GL_FACTORS_USAGE_SQLITE", "/tmp/factors_usage.db")
    )
    provider = provider or StripeBillingProvider.from_environment()

    # Stable per-tenant sink key for UsageAggregator lookups. In production
    # the middleware stamps this hash on every row; here we reconstruct
    # from tenant_id when the caller did not provide one.
    if api_key_hash is None:
        api_key_hash = hashlib.sha256(
            f"tenant:{tenant_id}".encode("utf-8")
        ).hexdigest()[:16]

    try:
        summary = aggregator.aggregate_by_period(api_key_hash, period="monthly")
        total_requests = summary.total_requests
        period_start = summary.period_start.isoformat()
        period_end = summary.period_end.isoformat()
    except Exception as exc:  # noqa: BLE001 -- sink may be empty
        logger.warning(
            "Could not aggregate usage for tenant=%s: %s; assuming zero",
            tenant_id,
            exc,
        )
        total_requests = 0
        period_start, period_end = _period_bounds()

    result = OverageFlushResult(
        tenant_id=tenant_id,
        tier=resolved_tier.value,
        subscription_id=subscription_id,
        period_start=period_start,
        period_end=period_end,
        dry_run=dry_run,
    )

    # ---- API calls meter -----------------------------------------------------
    tier_cfg = CATALOG.tier(resolved_tier)
    api_meter_def = tier_cfg.meter(Meter.API_CALLS)
    if api_meter_def is None:
        logger.debug("Tier %s has no API_CALLS meter; skipping.", resolved_tier)
    else:
        included = api_meter_def.included_per_month
        overage_qty = max(0, total_requests - included)
        if overage_qty == 0:
            logger.info(
                "No API overage for tenant=%s tier=%s (used=%d, included=%d)",
                tenant_id,
                resolved_tier.value,
                total_requests,
                included,
            )
        else:
            amount = overage_price(resolved_tier, Meter.API_CALLS, total_requests)
            line = {
                "meter": Meter.API_CALLS.value,
                "price_id": meter_price_id(resolved_tier, Meter.API_CALLS),
                "used": total_requests,
                "included": included,
                "overage_qty": overage_qty,
                "amount_usd": str(amount),
            }
            result.line_items.append(line)
            result.total_amount_usd = str(
                (
                    _decimal(result.total_amount_usd) + amount
                ).quantize(amount)
            )

            if not dry_run and subscription_id and provider.configured:
                idem = _idempotency_key(
                    tenant_id, subscription_id, Meter.API_CALLS, period_end
                )
                try:
                    usage = provider.record_usage(
                        subscription_id=subscription_id,
                        quantity=overage_qty,
                        idempotency_key=idem,
                    )
                    rec_id = usage.get("usage_record_id")
                    if rec_id:
                        result.stripe_usage_records.append(str(rec_id))
                    logger.info(
                        "Recorded %d overage units for tenant=%s tier=%s "
                        "idempotency_key=%s",
                        overage_qty,
                        tenant_id,
                        resolved_tier.value,
                        idem[:12],
                    )
                except StripeApiError as exc:
                    logger.error(
                        "Failed to record overage usage tenant=%s tier=%s: %s",
                        tenant_id,
                        resolved_tier.value,
                        exc,
                    )
                    raise
            elif dry_run:
                logger.info(
                    "[dry-run] Would record overage: tenant=%s tier=%s qty=%d",
                    tenant_id,
                    resolved_tier.value,
                    overage_qty,
                )

    return result


def _decimal(value: str):
    """Local Decimal helper to avoid importing decimal at module top."""
    from decimal import Decimal

    return Decimal(value)


__all__ = [
    "OverageFlushResult",
    "flush_period_overage",
]
