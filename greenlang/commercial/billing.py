# -*- coding: utf-8 -*-
"""Stripe subscription webhook handler (skeleton).

Production wiring requires:
- STRIPE_WEBHOOK_SECRET env var
- stripe-python package
- Tenant <-> Stripe customer ID mapping (stored in PG table `tenant_billing`)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubscriptionEvent:
    event_type: str  # customer.subscription.created|updated|deleted, invoice.paid, etc.
    tenant_id: str
    stripe_customer_id: str
    stripe_subscription_id: Optional[str]
    new_tier: Optional[str]
    raw_payload: dict[str, Any]


class StripeWebhookHandler:
    def __init__(self, webhook_secret: str, tenant_mapper: "TenantMapper") -> None:
        self._secret = webhook_secret
        self._mapper = tenant_mapper

    def verify_signature(self, payload: bytes, signature_header: str) -> bool:
        # Real impl: stripe.Webhook.construct_event(payload, sig, self._secret)
        # Returns the parsed event object. Here we just verify presence.
        return bool(signature_header) and bool(self._secret)

    def parse(self, payload: bytes) -> SubscriptionEvent:
        data = json.loads(payload.decode("utf-8"))
        event_type = data.get("type", "")
        stripe_obj = data.get("data", {}).get("object", {})
        stripe_customer_id = stripe_obj.get("customer", "")
        tenant_id = self._mapper.tenant_for_customer(stripe_customer_id)
        tier = self._tier_for_product(stripe_obj)
        return SubscriptionEvent(
            event_type=event_type,
            tenant_id=tenant_id,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=stripe_obj.get("id"),
            new_tier=tier,
            raw_payload=data,
        )

    def handle(self, event: SubscriptionEvent) -> None:
        logger.info(
            "Stripe event %s for tenant %s (tier=%s, sub=%s)",
            event.event_type, event.tenant_id, event.new_tier, event.stripe_subscription_id,
        )
        # Real impl: update tenant_billing table, emit tier-change audit event,
        # invalidate tier cache for this tenant.

    @staticmethod
    def _tier_for_product(stripe_obj: dict[str, Any]) -> Optional[str]:
        product = (stripe_obj.get("items", {}).get("data") or [{}])[0].get("price", {})
        product_id = product.get("product")
        # Real impl: lookup product_id -> Tier mapping from env/config
        mapping = {
            "prod_community": "community",
            "prod_pro": "pro",
            "prod_enterprise": "enterprise",
        }
        return mapping.get(product_id)


class TenantMapper:
    """In-memory placeholder. Production backs with PG `tenant_billing` table."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._m = mapping or {}

    def tenant_for_customer(self, stripe_customer_id: str) -> str:
        return self._m.get(stripe_customer_id, f"unknown:{stripe_customer_id}")

    def set(self, stripe_customer_id: str, tenant_id: str) -> None:
        self._m[stripe_customer_id] = tenant_id
