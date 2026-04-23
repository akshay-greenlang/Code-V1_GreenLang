#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sync GreenLang Factors Stripe products + prices from skus.py.

Reads :mod:`greenlang.factors.billing.skus` and creates/updates the
matching Stripe Product + Price objects via the Stripe API. Runs as the
glue between the SKU source-of-truth in code and the Stripe dashboard.

Usage::

    # Dry-run (no writes; prints what would change):
    python scripts/sync_stripe_products.py --dry-run

    # Real sync (requires STRIPE_API_KEY in environment):
    export STRIPE_API_KEY=sk_live_...
    python scripts/sync_stripe_products.py --commit

    # Sync against Stripe test mode:
    export STRIPE_API_KEY=sk_test_...
    python scripts/sync_stripe_products.py --commit --test-mode

Idempotent: safe to re-run. Uses Stripe's lookup_key on prices so the
SDK never has to hard-code Stripe price IDs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sync_stripe")


def load_skus() -> List[Dict[str, Any]]:
    from greenlang.factors.billing.skus import get_skus
    raw = get_skus()
    out: List[Dict[str, Any]] = []
    for sku in raw:
        if hasattr(sku, "model_dump"):
            out.append(sku.model_dump())
        elif hasattr(sku, "to_dict"):
            out.append(sku.to_dict())
        elif isinstance(sku, dict):
            out.append(sku)
        else:
            out.append(getattr(sku, "__dict__", {"id": str(sku)}))
    return out


def upsert_product(stripe, sku: Dict[str, Any], dry_run: bool) -> Optional[str]:
    """Create or update the Stripe Product for this SKU."""
    sku_id = sku.get("id") or sku.get("sku_id")
    if not sku_id:
        logger.warning("SKU has no id; skipping: %s", sku)
        return None

    existing = stripe.Product.search(query=f'metadata["gl_sku_id"]:"{sku_id}"')
    payload: Dict[str, Any] = {
        "name": sku.get("name") or sku_id,
        "description": sku.get("description") or "",
        "active": bool(sku.get("active", True)),
        "metadata": {
            "gl_sku_id": sku_id,
            "gl_tier": sku.get("tier", ""),
            "gl_source_of_truth": "greenlang.factors.billing.skus",
        },
    }

    if existing.data:
        product = existing.data[0]
        if dry_run:
            print(f"DRY: would update product {product.id} ({sku_id})")
            return product.id
        stripe.Product.modify(product.id, **payload)
        print(f"UPDATED product {product.id} ({sku_id})")
        return product.id

    if dry_run:
        print(f"DRY: would create product for {sku_id}")
        return None
    product = stripe.Product.create(**payload)
    print(f"CREATED product {product.id} ({sku_id})")
    return product.id


def upsert_price(stripe, product_id: str, sku: Dict[str, Any], dry_run: bool) -> None:
    """Create the Stripe Price (immutable; new prices replace old via lookup_key)."""
    sku_id = sku.get("id") or sku.get("sku_id")
    monthly_cents = sku.get("monthly_cents")
    if monthly_cents in (None, 0):
        if dry_run:
            print(f"DRY: skip price for {sku_id} (free tier or contact-sales)")
        return

    lookup_key = f"gl_factors_{sku_id}_monthly"
    existing = stripe.Price.list(lookup_keys=[lookup_key], active=True, limit=1)
    if existing.data:
        if dry_run:
            print(f"DRY: price {existing.data[0].id} for {sku_id} already exists")
        else:
            print(f"OK: price {existing.data[0].id} for {sku_id} already current")
        return

    payload = {
        "currency": sku.get("currency", "usd"),
        "unit_amount": int(monthly_cents),
        "product": product_id,
        "recurring": {"interval": "month"},
        "lookup_key": lookup_key,
        "metadata": {"gl_sku_id": sku_id},
    }

    if dry_run:
        print(f"DRY: would create price {sku.get('currency','usd')} {monthly_cents}c for {sku_id}")
        return
    price = stripe.Price.create(**payload)
    print(f"CREATED price {price.id} ({sku_id})")

    # Optional usage-based meter for overages
    overage = sku.get("overage_per_request_cents")
    if overage:
        usage_lookup = f"gl_factors_{sku_id}_usage"
        if not stripe.Price.list(lookup_keys=[usage_lookup], active=True, limit=1).data:
            stripe.Price.create(
                currency=sku.get("currency", "usd"),
                product=product_id,
                recurring={"interval": "month", "usage_type": "metered"},
                billing_scheme="per_unit",
                unit_amount=int(overage),
                lookup_key=usage_lookup,
                metadata={"gl_sku_id": sku_id, "kind": "overage"},
            )
            print(f"CREATED metered price for {sku_id} overages")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true", help="Apply changes (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writes")
    parser.add_argument("--test-mode", action="store_true", help="Use Stripe test mode (sk_test_ key required)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    dry_run = args.dry_run or not args.commit

    api_key = os.getenv("STRIPE_API_KEY")
    if not api_key:
        print("ERROR: STRIPE_API_KEY not set", file=sys.stderr)
        return 2
    if args.test_mode and api_key.startswith("sk_live_"):
        print("ERROR: --test-mode set but STRIPE_API_KEY is a live key", file=sys.stderr)
        return 2

    try:
        import stripe  # type: ignore
    except ImportError:
        print("ERROR: 'stripe' package not installed; run: pip install stripe", file=sys.stderr)
        return 2

    stripe.api_key = api_key

    skus = load_skus()
    print(f"Loaded {len(skus)} SKUs from greenlang.factors.billing.skus")
    print(f"Mode: {'DRY-RUN' if dry_run else 'COMMIT'}\n")

    for sku in skus:
        print(f"--- {sku.get('id') or sku.get('name')} ---")
        product_id = upsert_product(stripe, sku, dry_run=dry_run)
        if product_id:
            upsert_price(stripe, product_id, sku, dry_run=dry_run)
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
