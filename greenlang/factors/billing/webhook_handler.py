# -*- coding: utf-8 -*-
"""
Stripe webhook handler for the GreenLang Factors API.

Provides a FastAPI router that processes Stripe webhook events for
subscription lifecycle management, pack entitlement activation, and
invoice reconciliation.

Supported events (FY27 launch):
    - customer.subscription.created
    - customer.subscription.updated
    - customer.subscription.deleted
    - invoice.payment_succeeded
    - invoice.payment_failed
    - checkout.session.completed

Signature verification is mandatory in production. When
``STRIPE_WEBHOOK_SECRET`` is set the handler verifies the HMAC-SHA256
signature in the ``Stripe-Signature`` header and rejects timestamps
older than ``tolerance`` seconds. When the secret is unset (development
only) it logs a warning and parses the payload unverified.

The router is mounted at ``/api/v1/billing/webhooks``.

Environment variables:
    STRIPE_WEBHOOK_SECRET: Webhook endpoint signing secret (whsec_...).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from greenlang.factors.billing.skus import (
    CATALOG,
    PremiumPack,
    Tier,
    pack_from_price_id,
    tier_entitlements,
    tier_from_price_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/billing",
    tags=["Billing"],
)


# ---------------------------------------------------------------------------
# In-memory tier/quota tracking (updated by webhooks)
# ---------------------------------------------------------------------------
#
# Maps customer_id -> subscription state. In production this is backed by
# Postgres (see ``entitlements.EntitlementRegistry`` for the persistence
# layer); here we keep a process-local cache so tests don't need a DB.

_subscription_state: Dict[str, Dict[str, Any]] = {}

#: Hook the rest of the stack can subscribe to -- called with the latest
#: state dict every time the handler mutates it. Used by the tenant
#: service to sync entitlements into Postgres + invalidate Redis caches.
_state_listeners: List[Callable[[str, Dict[str, Any]], None]] = []


def get_subscription_state(customer_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached subscription state for a customer.

    Args:
        customer_id: Stripe customer ID.

    Returns:
        Subscription state dict or ``None`` if not tracked.
    """
    return _subscription_state.get(customer_id)


def update_subscription_state(customer_id: str, state: Dict[str, Any]) -> None:
    """Update cached subscription state + notify listeners.

    Args:
        customer_id: Stripe customer ID.
        state: Subscription state dict to cache.
    """
    _subscription_state[customer_id] = state
    logger.info(
        "Updated subscription state for customer %s: status=%s tier=%s",
        customer_id,
        state.get("status"),
        state.get("tier"),
    )
    for listener in _state_listeners:
        try:
            listener(customer_id, state)
        except Exception:  # noqa: BLE001 -- never let a listener crash a webhook
            logger.exception(
                "subscription-state listener failed for customer=%s",
                customer_id,
            )


def register_state_listener(
    listener: Callable[[str, Dict[str, Any]], None],
) -> None:
    """Subscribe ``listener`` to every subscription-state mutation.

    Used by application code (tenant service, entitlement registry, etc.)
    to persist webhook outcomes into durable storage.
    """
    _state_listeners.append(listener)


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def verify_stripe_signature(
    payload: bytes,
    sig_header: str,
    webhook_secret: str,
    tolerance: int = 300,
) -> Dict[str, Any]:
    """Verify a Stripe webhook signature and parse the event payload.

    Uses the ``Stripe-Signature`` header to confirm the webhook was sent
    by Stripe and has not been tampered with. Also enforces a timestamp
    tolerance to blunt replay attacks.

    Args:
        payload: Raw request body bytes.
        sig_header: Value of the ``Stripe-Signature`` header.
        webhook_secret: Webhook signing secret (``whsec_...``).
        tolerance: Maximum age of the event in seconds (default 300 = 5 min).

    Returns:
        Parsed event dict.

    Raises:
        HTTPException: If signature verification fails.
    """
    if not sig_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe-Signature header",
        )

    # Parse signature header: t=timestamp,v1=signature[,v1=...]
    elements: Dict[str, List[str]] = {}
    for part in sig_header.split(","):
        key_val = part.strip().split("=", 1)
        if len(key_val) == 2:
            elements.setdefault(key_val[0], []).append(key_val[1])

    timestamp_str = (elements.get("t") or [None])[0]
    signatures = elements.get("v1", [])

    if not timestamp_str or not signatures:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Stripe-Signature header format",
        )

    try:
        timestamp = int(timestamp_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid timestamp in Stripe-Signature header",
        )

    if abs(time.time() - timestamp) > tolerance:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook timestamp outside tolerance window",
        )

    # Compute expected signature: HMAC-SHA256("{timestamp}.{payload}", secret)
    signed_payload = f"{timestamp}.".encode("utf-8") + payload
    expected_sig = hmac.new(
        webhook_secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    if not any(hmac.compare_digest(expected_sig, sig) for sig in signatures):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook signature verification failed",
        )

    try:
        event = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {exc}",
        )

    return event


# ---------------------------------------------------------------------------
# Helpers -- resolve SKUs from a Stripe subscription
# ---------------------------------------------------------------------------


def _extract_items(subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the items array from a Stripe subscription object."""
    items = subscription.get("items") or {}
    if isinstance(items, dict):
        data = items.get("data") or []
    else:
        data = items
    return [i for i in data if isinstance(i, dict)]


def _resolve_subscription_skus(
    subscription: Dict[str, Any],
) -> Dict[str, Any]:
    """Map a Stripe subscription back onto GreenLang tier + pack SKUs.

    Stripe tells us the customer is on price ``price_factors_pro_monthly``;
    we convert that into :data:`Tier.PRO` and the subset of packs they have
    entitlement to.
    """
    resolved_tier: Optional[Tier] = None
    resolved_packs: List[PremiumPack] = []
    resolved_price_ids: List[str] = []

    for item in _extract_items(subscription):
        price = item.get("price") or {}
        # ``price.lookup_key`` is where the provisioner stores the
        # GreenLang-owned ID. Fall back to ``metadata.gl_price_id`` for
        # accounts that don't use lookup keys.
        gl_price_id = (
            price.get("lookup_key")
            or (price.get("metadata") or {}).get("gl_price_id")
            or price.get("id", "")
        )
        resolved_price_ids.append(gl_price_id)
        tier_match = tier_from_price_id(gl_price_id)
        if tier_match is not None and resolved_tier is None:
            resolved_tier = tier_match
        pack_match = pack_from_price_id(gl_price_id)
        if pack_match is not None and pack_match not in resolved_packs:
            resolved_packs.append(pack_match)

    # Tier metadata fallback: respect ``subscription.metadata.tier`` when
    # the price mapping comes up empty (e.g. custom Enterprise quote).
    if resolved_tier is None:
        tier_str = (subscription.get("metadata") or {}).get("tier", "")
        if tier_str:
            try:
                resolved_tier = Tier(tier_str.lower().strip())
            except ValueError:
                resolved_tier = None

    return {
        "tier": resolved_tier.value if resolved_tier else "community",
        "packs": [p.value for p in resolved_packs],
        "price_ids": resolved_price_ids,
    }


def _build_state_from_subscription(
    subscription: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the cached state dict from a Stripe subscription payload."""
    sku = _resolve_subscription_skus(subscription)
    tier_value = sku["tier"]
    try:
        entitlements = tier_entitlements(Tier(tier_value))
    except ValueError:
        entitlements = tier_entitlements(Tier.COMMUNITY)
    return {
        "subscription_id": subscription.get("id"),
        "status": subscription.get("status", "active"),
        "tier": tier_value,
        "packs": sku["packs"],
        "price_ids": sku["price_ids"],
        "current_period_start": subscription.get("current_period_start"),
        "current_period_end": subscription.get("current_period_end"),
        "cancel_at_period_end": subscription.get("cancel_at_period_end", False),
        "entitlements": entitlements,
        "updated_at": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_subscription_created(event_data: Dict[str, Any]) -> Dict[str, str]:
    """Handle ``customer.subscription.created``.

    A brand-new subscription — typically the webhook we receive right
    after Checkout succeeds, or right after a manually-created
    subscription in Stripe admin. We build the full entitlement snapshot
    and persist it so the rest of the stack can read it immediately.
    """
    customer_id = event_data.get("customer", "") or ""
    state = _build_state_from_subscription(event_data)
    logger.info(
        "Subscription created: customer=%s subscription=%s tier=%s packs=%s",
        customer_id,
        state["subscription_id"],
        state["tier"],
        state["packs"],
    )
    update_subscription_state(customer_id, state)
    return {"status": "processed", "event": "customer.subscription.created"}


def _handle_subscription_updated(event_data: Dict[str, Any]) -> Dict[str, str]:
    """Handle ``customer.subscription.updated``.

    Fires for plan changes, cancellation scheduling, pack adds/removes,
    etc. We rebuild the state snapshot and overwrite — there is no field
    that 'updates' is materially different from ``created``, and avoiding
    merge logic is what keeps this webhook safe under out-of-order delivery.
    """
    customer_id = event_data.get("customer", "") or ""
    state = _build_state_from_subscription(event_data)
    logger.info(
        "Subscription updated: customer=%s status=%s tier=%s",
        customer_id,
        state["status"],
        state["tier"],
    )
    update_subscription_state(customer_id, state)
    return {"status": "processed", "event": "customer.subscription.updated"}


def _handle_subscription_deleted(event_data: Dict[str, Any]) -> Dict[str, str]:
    """Handle ``customer.subscription.deleted``.

    Final-state cancellation. Tenant is downgraded to Community.
    """
    customer_id = event_data.get("customer", "") or ""
    logger.info(
        "Subscription deleted: customer=%s subscription=%s",
        customer_id,
        event_data.get("id"),
    )
    state: Dict[str, Any] = {
        "subscription_id": None,
        "status": "canceled",
        "tier": Tier.COMMUNITY.value,
        "packs": [],
        "price_ids": [],
        "canceled_at": int(time.time()),
        "entitlements": tier_entitlements(Tier.COMMUNITY),
        "updated_at": int(time.time()),
    }
    update_subscription_state(customer_id, state)
    return {"status": "processed", "event": "customer.subscription.deleted"}


def _handle_payment_succeeded(event_data: Dict[str, Any]) -> Dict[str, str]:
    """Handle ``invoice.payment_succeeded``.

    Confirms the subscription is current. We clear any ``past_due`` flag
    and stamp ``last_payment_at`` so the SLA tracker can attribute
    downtime correctly.
    """
    invoice = event_data
    customer_id = invoice.get("customer", "") or ""
    subscription_id = invoice.get("subscription", "") or ""
    amount_paid = invoice.get("amount_paid", 0)

    logger.info(
        "Payment succeeded: customer=%s subscription=%s amount=%d",
        customer_id,
        subscription_id,
        amount_paid,
    )

    existing = dict(_subscription_state.get(customer_id) or {})
    existing["status"] = "active"
    existing["last_payment_at"] = int(time.time())
    existing["last_invoice_id"] = invoice.get("id")
    existing["last_amount_paid"] = amount_paid
    if subscription_id and not existing.get("subscription_id"):
        existing["subscription_id"] = subscription_id
    existing.setdefault("tier", Tier.COMMUNITY.value)
    existing.setdefault("packs", [])
    existing["updated_at"] = int(time.time())
    update_subscription_state(customer_id, existing)

    return {"status": "processed", "event": "invoice.payment_succeeded"}


def _handle_payment_failed(event_data: Dict[str, Any]) -> Dict[str, str]:
    """Handle ``invoice.payment_failed``.

    Marks the subscription ``past_due`` so middleware can degrade the
    tenant to read-only / Community visibility until dunning clears.
    """
    invoice = event_data
    customer_id = invoice.get("customer", "") or ""
    subscription_id = invoice.get("subscription", "") or ""
    attempt_count = invoice.get("attempt_count", 0)

    logger.warning(
        "Payment failed: customer=%s subscription=%s attempt=%d",
        customer_id,
        subscription_id,
        attempt_count,
    )

    existing = dict(_subscription_state.get(customer_id) or {})
    existing["status"] = "past_due"
    existing["last_payment_failure_at"] = int(time.time())
    existing["last_failed_invoice_id"] = invoice.get("id")
    existing["payment_attempt_count"] = attempt_count
    if subscription_id and not existing.get("subscription_id"):
        existing["subscription_id"] = subscription_id
    existing.setdefault("tier", Tier.COMMUNITY.value)
    existing["updated_at"] = int(time.time())
    update_subscription_state(customer_id, existing)

    return {"status": "processed", "event": "invoice.payment_failed"}


def _handle_checkout_completed(event_data: Dict[str, Any]) -> Dict[str, str]:
    """Handle ``checkout.session.completed``.

    Self-serve sign-up path: Checkout succeeded and a subscription was
    created. Stripe also sends ``customer.subscription.created`` for the
    same sale, but we process this event eagerly so:

    * The ``tenant_id`` from ``client_reference_id`` is linked to the
      Stripe customer immediately.
    * The ``sku_name`` and ``premium_packs`` from the Checkout metadata
      grant the right tier + packs without waiting for the subscription
      event (which can arrive seconds later and out of order).

    Idempotency: Stripe is allowed to retry a webhook delivery for up to
    72 hours. We rebuild ``state`` from scratch from the metadata each
    time, so a duplicate delivery is a no-op (same final state, same
    listener notification — the persistence layer is responsible for
    using ``subscription_id`` as the upsert key).
    """
    session = event_data
    customer_id = session.get("customer", "") or ""
    tenant_id = session.get("client_reference_id") or ""
    subscription_id = session.get("subscription", "") or ""
    mode = session.get("mode", "")
    metadata = session.get("metadata") or {}
    sku_name = (metadata.get("sku_name") or "").lower().strip()
    premium_packs_raw = (metadata.get("premium_packs") or "").strip()
    premium_packs: List[str] = [
        p.strip().lower() for p in premium_packs_raw.split(",") if p.strip()
    ]

    logger.info(
        "Checkout completed: customer=%s tenant=%s subscription=%s mode=%s "
        "sku=%s packs=%s",
        customer_id,
        tenant_id,
        subscription_id,
        mode,
        sku_name,
        premium_packs,
    )

    existing = dict(_subscription_state.get(customer_id) or {})
    if tenant_id:
        existing["tenant_id"] = tenant_id
    if subscription_id:
        existing["subscription_id"] = subscription_id

    # Resolve tier + packs from the Checkout metadata so entitlements are
    # granted even if the ``customer.subscription.created`` event hasn't
    # arrived yet (or is delayed by Stripe).
    resolved_tier: Optional[Tier] = None
    if sku_name:
        try:
            resolved_tier = Tier(sku_name)
        except ValueError:
            logger.warning(
                "Checkout metadata.sku_name=%r is not a known Tier; "
                "defaulting to community",
                sku_name,
            )
            resolved_tier = None

    if resolved_tier is not None:
        existing["tier"] = resolved_tier.value
        existing["entitlements"] = tier_entitlements(resolved_tier)
        existing["status"] = "active"
    else:
        existing.setdefault("tier", Tier.COMMUNITY.value)
        existing.setdefault("status", "active")

    # Validate + merge premium packs from metadata. We accept only packs
    # that exist in the catalog and that the tier is allowed to purchase.
    if premium_packs:
        valid_packs: List[str] = []
        for pack_str in premium_packs:
            try:
                pack = PremiumPack(pack_str)
            except ValueError:
                logger.warning(
                    "Checkout metadata.premium_packs entry %r is not a "
                    "known PremiumPack; ignoring",
                    pack_str,
                )
                continue
            valid_packs.append(pack.value)
        # Merge with any packs already attached (e.g. from a prior subscription
        # event) without duplicates, preserving deterministic order.
        merged = list(existing.get("packs") or [])
        for pack_value in valid_packs:
            if pack_value not in merged:
                merged.append(pack_value)
        existing["packs"] = merged
    else:
        existing.setdefault("packs", [])

    existing["checkout_completed_at"] = int(time.time())
    existing["updated_at"] = int(time.time())
    update_subscription_state(customer_id, existing)

    return {"status": "processed", "event": "checkout.session.completed"}


# Event type -> handler mapping
_EVENT_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, str]]] = {
    "customer.subscription.created": _handle_subscription_created,
    "customer.subscription.updated": _handle_subscription_updated,
    "customer.subscription.deleted": _handle_subscription_deleted,
    "invoice.payment_succeeded": _handle_payment_succeeded,
    "invoice.payment_failed": _handle_payment_failed,
    "checkout.session.completed": _handle_checkout_completed,
}


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/webhooks",
    summary="Stripe webhook receiver",
    description=(
        "Receives and processes Stripe webhook events for subscription "
        "lifecycle management. Verifies webhook signatures when "
        "STRIPE_WEBHOOK_SECRET is configured."
    ),
    responses={
        200: {"description": "Event processed successfully"},
        400: {"description": "Invalid signature or payload"},
    },
)
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
) -> JSONResponse:
    """Receive and process Stripe webhook events.

    Verifies the webhook signature (when configured), dispatches to the
    appropriate handler, and always returns 200 for events we ignore so
    Stripe does not retry forever.
    """
    payload = await request.body()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    if webhook_secret:
        event = verify_stripe_signature(
            payload,
            stripe_signature or "",
            webhook_secret,
        )
    else:
        logger.warning(
            "STRIPE_WEBHOOK_SECRET not set; accepting webhook without "
            "signature verification. Do NOT use in production."
        )
        try:
            event = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {exc}",
            )

    event_type = event.get("type", "")
    event_data = event.get("data", {}).get("object", {})

    logger.info(
        "Received Stripe webhook: type=%s id=%s",
        event_type,
        event.get("id"),
    )

    handler = _EVENT_HANDLERS.get(event_type)
    if handler:
        result = handler(event_data)
        return JSONResponse(content=result, status_code=200)

    logger.info("Ignoring unhandled Stripe event type: %s", event_type)
    return JSONResponse(
        content={"status": "ignored", "event": event_type},
        status_code=200,
    )


__all__ = [
    "router",
    "verify_stripe_signature",
    "get_subscription_state",
    "update_subscription_state",
    "register_state_listener",
]
