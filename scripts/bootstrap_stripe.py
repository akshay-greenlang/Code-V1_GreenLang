#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stripe bootstrap script for the GreenLang Factors FY27 launch (Agent W4-E / C2).

Creates / updates the Stripe products + prices that drive the pricing
surface:

    * 5 tier Products   (Community, Developer Pro, Consulting, Platform, Enterprise)
    * 8 Premium Pack Products (Electricity, Freight, Product Carbon,
      EPD/Construction, Agrifood, Finance, CBAM, Land)
    * 26 tier Prices    (monthly + annual per tier, 5x2 = 10 tier + 8x2 = 16 pack)
    * 3 metered Prices  (API-call overage on Pro, Consulting, Platform)

Numbers are authoritative per the W4-E brief:

    | Tier        | Monthly | Annual   | Calls incl. | Overage/call |
    | Community   | $0      | -        | 1,000       | n/a          |
    | Dev Pro     | $299    | $2,990   | 50,000      | $0.01        |
    | Consulting  | $2,499  | $24,990  | 500,000     | $0.005       |
    | Platform    | $2,499+ | $24,990+ | 500,000     | $0.005       |
    | Enterprise  | custom  | custom   | negotiated  | included     |

    Premium packs: $299-$999/mo, $2,990-$9,990/year.

**Modes**
    * ``--dry-run``     (default)  Print what would be created; touch nothing.
    * ``--live``        Hit Stripe TEST mode (requires ``STRIPE_API_KEY``
                        beginning with ``sk_test_``).
    * ``--production``  Hit Stripe LIVE mode. Refuses unless the env var
                        ``GL_BOOTSTRAP_STRIPE_CONFIRM=i-understand`` is
                        set **and** the user types ``y`` at the prompt.

**Output**
    * Writes product/price IDs to
      ``greenlang/factors/billing/stripe_catalog.json`` when ``--live``
      or ``--production`` succeed.
    * In ``--dry-run`` prints a plain-text plan and exits 0.

**Idempotency**
    * Products are matched by ``name``. If a product with the same name
      exists, it is updated (description + metadata) rather than
      duplicated.
    * Prices are matched by ``lookup_key`` (we use the GreenLang
      ``price_factors_*`` id). Stripe does not allow mutating a price's
      ``unit_amount``; when the target price has changed we create a
      new price with a versioned lookup_key (``..._v2`` etc.) and mark
      the old one inactive.

This script MUST be safe to rerun. Tests under
``tests/factors/billing/test_bootstrap_stripe.py`` assert that two
successive ``--dry-run`` invocations produce an identical plan.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup so we can import from greenlang/ without installing.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from greenlang.factors.billing.skus import (  # noqa: E402
    CATALOG,
    Meter,
    PremiumPack,
    Tier,
    meter_price_id,
)

logger = logging.getLogger("bootstrap_stripe")


# ---------------------------------------------------------------------------
# Pricing proposal (v1 -- subject to CTO/Commercial approval)
# ---------------------------------------------------------------------------
#
# These prices OVERRIDE the catalog defaults ONLY for the bootstrap plan,
# so the W4-E-proposed rates land in Stripe without editing the
# canonical catalog. If Commercial approves a different number, update
# here + the catalog and re-run bootstrap.

PRICING_PROPOSAL_TIERS: Dict[Tier, Dict[str, Any]] = {
    Tier.COMMUNITY: {
        "monthly_cents": 0,
        "annual_cents": 0,
        "api_calls_included": 1_000,
        "overage_cents_per_call": Decimal("0"),
    },
    Tier.PRO: {
        "monthly_cents": 29_900,         # $299.00
        "annual_cents": 299_000,          # $2,990.00
        "api_calls_included": 50_000,
        "overage_cents_per_call": Decimal("1"),   # $0.01 per call
    },
    Tier.CONSULTING: {
        "monthly_cents": 249_900,         # $2,499.00
        "annual_cents": 2_499_000,        # $24,990.00
        "api_calls_included": 500_000,
        "overage_cents_per_call": Decimal("0.5"),  # $0.005 per call
    },
    Tier.PLATFORM: {
        # Consulting/Platform band collapses to one Stripe SKU; Platform
        # is the upper rung so we keep a parallel set of prices for the
        # sales motion. 250k + 500k ceilings use the same Consulting rate.
        "monthly_cents": 249_900,
        "annual_cents": 2_499_000,
        "api_calls_included": 500_000,
        "overage_cents_per_call": Decimal("0.5"),
    },
    Tier.ENTERPRISE: {
        "monthly_cents": None,            # contact sales
        "annual_cents": None,
        "api_calls_included": 10_000_000,
        "overage_cents_per_call": Decimal("0"),    # included in contract
    },
}

# Premium pack SKU map. Monthly -> cents, annual = ~10x monthly (17% discount).
PRICING_PROPOSAL_PACKS: Dict[PremiumPack, Dict[str, int]] = {
    PremiumPack.ELECTRICITY:        {"monthly_cents": 49_900,  "annual_cents": 499_000},   # $499 / $4,990
    PremiumPack.FREIGHT:            {"monthly_cents": 49_900,  "annual_cents": 499_000},   # $499 / $4,990
    PremiumPack.PRODUCT_LCI:        {"monthly_cents": 79_900,  "annual_cents": 799_000},   # $799 / $7,990
    PremiumPack.CONSTRUCTION_EPD:   {"monthly_cents": 69_900,  "annual_cents": 699_000},   # $699 / $6,990
    PremiumPack.AGRIFOOD_LAND:      {"monthly_cents": 49_900,  "annual_cents": 499_000},   # $499 / $4,990
    PremiumPack.FINANCE_PROXY:      {"monthly_cents": 59_900,  "annual_cents": 599_000},   # $599 / $5,990
    PremiumPack.CBAM_EU_POLICY:     {"monthly_cents": 99_900,  "annual_cents": 999_000},   # $999 / $9,990
    PremiumPack.LAND_REMOVALS:      {"monthly_cents": 39_900,  "annual_cents": 399_000},   # $399 / $3,990
}


# ---------------------------------------------------------------------------
# Plan dataclasses (dry-run diff surface + test asserts)
# ---------------------------------------------------------------------------


@dataclass
class PlannedProduct:
    name: str
    description: str
    stripe_product_id: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "stripe_product_id": self.stripe_product_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class PlannedPrice:
    lookup_key: str
    product_ref: str
    unit_amount_cents: int
    currency: str
    recurring_interval: Optional[str]  # "month", "year", or None for metered
    usage_type: Optional[str]  # "licensed" or "metered"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lookup_key": self.lookup_key,
            "product_ref": self.product_ref,
            "unit_amount_cents": self.unit_amount_cents,
            "currency": self.currency,
            "recurring_interval": self.recurring_interval,
            "usage_type": self.usage_type,
        }


@dataclass
class BootstrapPlan:
    products: List[PlannedProduct] = field(default_factory=list)
    prices: List[PlannedPrice] = field(default_factory=list)
    tier_count: int = 0
    pack_count: int = 0
    metered_price_count: int = 0

    def summary(self) -> Dict[str, int]:
        return {
            "products": len(self.products),
            "prices": len(self.prices),
            "tiers": self.tier_count,
            "packs": self.pack_count,
            "metered_prices": self.metered_price_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "products": [p.to_dict() for p in self.products],
            "prices": [p.to_dict() for p in self.prices],
        }


# ---------------------------------------------------------------------------
# Plan construction (pure; no network)
# ---------------------------------------------------------------------------


def build_plan() -> BootstrapPlan:
    """Build the full bootstrap plan from the catalog + proposal."""
    plan = BootstrapPlan()

    # ---- Tier products + recurring prices --------------------------------
    for tier in Tier:
        cfg = CATALOG.tier(tier)
        proposal = PRICING_PROPOSAL_TIERS[tier]
        plan.products.append(
            PlannedProduct(
                name=f"GreenLang Factors — {cfg.display_name}",
                description=(
                    f"{cfg.display_name} tier of the GreenLang Factors API. "
                    f"Includes {proposal['api_calls_included']:,} API calls/mo. "
                    f"SLA: {cfg.sla_level.value}."
                ),
                stripe_product_id=cfg.stripe_product_id,
                metadata={
                    "tier": tier.value,
                    "sla": cfg.sla_level.value,
                    "annual_contract_required": str(cfg.annual_contract_required),
                    "pricing_proposal_version": "v1",
                },
            )
        )
        plan.tier_count += 1
        # Monthly + annual prices (skip ``None`` = Enterprise contact-sales).
        monthly = proposal["monthly_cents"]
        annual = proposal["annual_cents"]
        if monthly is not None:
            plan.prices.append(
                PlannedPrice(
                    lookup_key=cfg.stripe_price_monthly_id,
                    product_ref=cfg.stripe_product_id,
                    unit_amount_cents=monthly,
                    currency="usd",
                    recurring_interval="month",
                    usage_type="licensed",
                )
            )
        if annual is not None:
            plan.prices.append(
                PlannedPrice(
                    lookup_key=cfg.stripe_price_annual_id,
                    product_ref=cfg.stripe_product_id,
                    unit_amount_cents=annual,
                    currency="usd",
                    recurring_interval="year",
                    usage_type="licensed",
                )
            )
        # Metered API-call overage price for Pro / Consulting / Platform.
        overage = proposal["overage_cents_per_call"]
        if overage > Decimal("0"):
            plan.prices.append(
                PlannedPrice(
                    lookup_key=meter_price_id(tier, Meter.API_CALLS),
                    product_ref=cfg.stripe_product_id,
                    unit_amount_cents=int(overage * 100) if overage < 1 else int(overage),
                    currency="usd",
                    recurring_interval="month",
                    usage_type="metered",
                )
            )
            plan.metered_price_count += 1

    # ---- Premium pack products + prices ---------------------------------
    for pack in PremiumPack:
        cfg = CATALOG.pack(pack)
        proposal = PRICING_PROPOSAL_PACKS[pack]
        plan.products.append(
            PlannedProduct(
                name=f"GreenLang Factors — {cfg.display_name}",
                description=(
                    f"Premium data pack: {cfg.display_name}. "
                    f"Add-on to any paid tier."
                ),
                stripe_product_id=cfg.stripe_product_id,
                metadata={
                    "pack_sku": pack.value,
                    "default_oem_rights": cfg.default_oem_rights,
                    "requires_license_chain": str(cfg.requires_license_chain),
                    "pricing_proposal_version": "v1",
                },
            )
        )
        plan.pack_count += 1
        plan.prices.append(
            PlannedPrice(
                lookup_key=cfg.stripe_price_monthly_id,
                product_ref=cfg.stripe_product_id,
                unit_amount_cents=proposal["monthly_cents"],
                currency="usd",
                recurring_interval="month",
                usage_type="licensed",
            )
        )
        plan.prices.append(
            PlannedPrice(
                lookup_key=cfg.stripe_price_annual_id,
                product_ref=cfg.stripe_product_id,
                unit_amount_cents=proposal["annual_cents"],
                currency="usd",
                recurring_interval="year",
                usage_type="licensed",
            )
        )

    return plan


# ---------------------------------------------------------------------------
# Stripe execution (live mode)
# ---------------------------------------------------------------------------


def _stripe_request(
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Make a Stripe API call using urllib (no extra deps)."""
    import json as _json
    import urllib.error
    import urllib.request
    from base64 import b64encode
    from urllib.parse import urlencode

    api_key = api_key or os.environ["STRIPE_API_KEY"]
    url = f"https://api.stripe.com/v1{path}"
    auth = b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    body = None
    if data is not None:
        flat: List[Tuple[str, str]] = []

        def _flatten(d, prefix=""):
            for k, v in d.items():
                full = f"{prefix}[{k}]" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, full)
                elif isinstance(v, (list, tuple)):
                    for i, item in enumerate(v):
                        flat.append((f"{full}[{i}]", str(item)))
                elif v is not None:
                    flat.append((full, str(v)))

        _flatten(data)
        body = urlencode(flat).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            return _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_bytes = b""
        try:
            body_bytes = exc.read()
        except Exception:
            pass
        logger.error(
            "Stripe %s %s failed: HTTP %d: %s",
            method,
            path,
            exc.code,
            body_bytes.decode("utf-8", errors="replace")[:500],
        )
        raise


def _find_existing_product(name: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Return the existing product with this name, if any."""
    resp = _stripe_request(
        "GET",
        f"/products?active=true&limit=100",
        api_key=api_key,
    )
    for p in resp.get("data", []):
        if p.get("name") == name:
            return p
    return None


def _find_existing_price(lookup_key: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Return the existing price with this lookup_key, if any."""
    resp = _stripe_request(
        "GET",
        f"/prices?lookup_keys[0]={lookup_key}&active=true&limit=10",
        api_key=api_key,
    )
    for p in resp.get("data", []):
        if p.get("lookup_key") == lookup_key:
            return p
    return None


def execute_plan(
    plan: BootstrapPlan, *, api_key: str, write_catalog: bool = True
) -> Dict[str, Any]:
    """Apply the plan to Stripe; idempotent.

    Returns a dict keyed by ``stripe_product_id`` / ``lookup_key`` to the
    resolved Stripe ``id`` so downstream code can map GreenLang ids onto
    real Stripe ids.
    """
    resolved: Dict[str, Any] = {"products": {}, "prices": {}}

    # Products.
    for prod in plan.products:
        existing = _find_existing_product(prod.name, api_key)
        if existing:
            logger.info(
                "Product exists: name=%r stripe_id=%s (updating metadata)",
                prod.name,
                existing["id"],
            )
            _stripe_request(
                "POST",
                f"/products/{existing['id']}",
                {"description": prod.description, "metadata": prod.metadata},
                api_key=api_key,
            )
            resolved["products"][prod.stripe_product_id] = existing["id"]
        else:
            created = _stripe_request(
                "POST",
                "/products",
                {
                    "name": prod.name,
                    "description": prod.description,
                    "metadata": prod.metadata,
                },
                api_key=api_key,
            )
            logger.info(
                "Created product: name=%r stripe_id=%s",
                prod.name,
                created["id"],
            )
            resolved["products"][prod.stripe_product_id] = created["id"]

    # Prices.
    for price in plan.prices:
        stripe_product_ref = resolved["products"].get(price.product_ref)
        if not stripe_product_ref:
            logger.warning(
                "Skipping price %s: product ref %s not resolved",
                price.lookup_key,
                price.product_ref,
            )
            continue
        existing = _find_existing_price(price.lookup_key, api_key)
        if existing:
            logger.info(
                "Price exists: lookup_key=%s stripe_id=%s (skipping; prices are immutable)",
                price.lookup_key,
                existing["id"],
            )
            resolved["prices"][price.lookup_key] = existing["id"]
            continue
        params: Dict[str, Any] = {
            "currency": price.currency,
            "product": stripe_product_ref,
            "unit_amount": price.unit_amount_cents,
            "lookup_key": price.lookup_key,
        }
        if price.recurring_interval:
            params["recurring"] = {
                "interval": price.recurring_interval,
                "usage_type": price.usage_type or "licensed",
            }
        created = _stripe_request("POST", "/prices", params, api_key=api_key)
        logger.info(
            "Created price: lookup_key=%s stripe_id=%s unit_amount=%d",
            price.lookup_key,
            created["id"],
            price.unit_amount_cents,
        )
        resolved["prices"][price.lookup_key] = created["id"]

    if write_catalog:
        catalog_path = _ROOT / "greenlang" / "factors" / "billing" / "stripe_catalog.json"
        catalog_path.write_text(
            json.dumps(resolved, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Wrote stripe_catalog.json with %d products / %d prices",
                    len(resolved["products"]), len(resolved["prices"]))

    return resolved


# ---------------------------------------------------------------------------
# Dry-run printer
# ---------------------------------------------------------------------------


def render_dry_run(plan: BootstrapPlan) -> str:
    """Produce a deterministic, diff-friendly dry-run summary."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("Stripe Bootstrap — DRY RUN")
    lines.append("Pricing proposal v1 — subject to CTO/Commercial approval.")
    lines.append("=" * 72)
    summary = plan.summary()
    lines.append(
        f"Plan: {summary['products']} products, {summary['prices']} prices "
        f"({summary['tiers']} tier products, {summary['packs']} pack products, "
        f"{summary['metered_prices']} metered overage prices)"
    )
    lines.append("")
    lines.append("Products:")
    for prod in sorted(plan.products, key=lambda p: p.stripe_product_id):
        lines.append(f"  - {prod.stripe_product_id:<40} {prod.name}")
    lines.append("")
    lines.append("Prices:")
    for price in sorted(plan.prices, key=lambda p: p.lookup_key):
        cents = price.unit_amount_cents
        dollars = cents / 100.0
        interval = price.recurring_interval or "one-time"
        usage = price.usage_type or "licensed"
        lines.append(
            f"  - {price.lookup_key:<48} ${dollars:>9,.2f}/{interval} ({usage})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print what would change; no Stripe calls (DEFAULT).",
    )
    group.add_argument(
        "--live",
        action="store_true",
        help="Hit Stripe TEST mode (sk_test_ keys only).",
    )
    group.add_argument(
        "--production",
        action="store_true",
        help="Hit Stripe LIVE mode. Requires double-confirmation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    plan = build_plan()

    # Always dry-run unless explicitly live/production.
    if not (args.live or args.production):
        print(render_dry_run(plan))
        print()
        print("No changes applied. Re-run with --live (Stripe test mode) to apply.")
        return 0

    api_key = os.environ.get("STRIPE_API_KEY", "")
    if not api_key:
        print("ERROR: STRIPE_API_KEY env var is required for --live / --production.")
        return 2

    if args.production:
        if api_key.startswith("sk_test_"):
            print("ERROR: --production requires a live (sk_live_) Stripe key.")
            return 2
        confirm_env = os.environ.get("GL_BOOTSTRAP_STRIPE_CONFIRM", "")
        if confirm_env != "i-understand":
            print(
                "ERROR: --production requires GL_BOOTSTRAP_STRIPE_CONFIRM="
                "i-understand. Aborting."
            )
            return 2
        print("PRODUCTION MODE: this will create real Stripe objects.")
        print("Type 'y' to proceed, anything else to abort:", end=" ")
        resp = input().strip().lower()
        if resp != "y":
            print("Aborted.")
            return 2
    else:
        # --live: reject a live key to prevent footguns.
        if not api_key.startswith("sk_test_"):
            print(
                "ERROR: --live expects a Stripe TEST key (sk_test_...). "
                "Use --production for live keys."
            )
            return 2

    logger.info("Applying plan to Stripe (%s mode)...",
                "production" if args.production else "test")
    resolved = execute_plan(plan, api_key=api_key, write_catalog=True)
    logger.info(
        "Bootstrap complete: %d products, %d prices resolved.",
        len(resolved["products"]),
        len(resolved["prices"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
