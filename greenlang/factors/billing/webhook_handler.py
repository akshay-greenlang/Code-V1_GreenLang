# -*- coding: utf-8 -*-
"""
Stripe webhook handler for the GreenLang Factors API.

Provides a FastAPI router that processes Stripe webhook events for
subscription lifecycle management. Handles payment confirmations,
subscription updates, and cancellations.

Supported events:
    - invoice.payment_succeeded
    - customer.subscription.updated
    - customer.subscription.deleted

The router is mounted at /api/v1/billing/webhooks.

Environment variables:
    STRIPE_WEBHOOK_SECRET: Webhook endpoint signing secret for signature verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

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

# Maps customer_id -> subscription state
# In production, this would be backed by a database.
_subscription_state: Dict[str, Dict[str, Any]] = {}


def get_subscription_state(customer_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached subscription state for a customer.

    Args:
        customer_id: Stripe customer ID.

    Returns:
        Subscription state dict or None if not tracked.
    """
    return _subscription_state.get(customer_id)


def update_subscription_state(customer_id: str, state: Dict[str, Any]) -> None:
    """
    Update cached subscription state for a customer.

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


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def verify_stripe_signature(
    payload: bytes,
    sig_header: str,
    webhook_secret: str,
    tolerance: int = 300,
) -> Dict[str, Any]:
    """
    Verify a Stripe webhook signature and parse the event payload.

    Uses the Stripe-Signature header to verify that the webhook was sent
    by Stripe and has not been tampered with.

    Args:
        payload: Raw request body bytes.
        sig_header: Value of the Stripe-Signature header.
        webhook_secret: Webhook signing secret (whsec_...).
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

    # Parse signature header: t=timestamp,v1=signature
    elements = {}
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

    # Check tolerance
    if abs(time.time() - timestamp) > tolerance:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook timestamp outside tolerance window",
        )

    # Compute expected signature
    signed_payload = f"{timestamp}.".encode("utf-8") + payload
    expected_sig = hmac.new(
        webhook_secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    # Verify at least one signature matches
    if not any(hmac.compare_digest(expected_sig, sig) for sig in signatures):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook signature verification failed",
        )

    # Parse and return event
    try:
        event = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {exc}",
        )

    return event


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_payment_succeeded(event_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Handle invoice.payment_succeeded event.

    Updates local tracking to confirm the subscription is active and paid.

    Args:
        event_data: The event data.object from Stripe.

    Returns:
        Status dict.
    """
    invoice = event_data
    customer_id = invoice.get("customer", "")
    subscription_id = invoice.get("subscription", "")
    amount_paid = invoice.get("amount_paid", 0)

    logger.info(
        "Payment succeeded: customer=%s subscription=%s amount=%d",
        customer_id,
        subscription_id,
        amount_paid,
    )

    # Update local state
    existing = _subscription_state.get(customer_id, {})
    existing["status"] = "active"
    existing["last_payment_at"] = int(time.time())
    existing["subscription_id"] = subscription_id
    _subscription_state[customer_id] = existing

    return {"status": "processed", "event": "invoice.payment_succeeded"}


def _handle_subscription_updated(event_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Handle customer.subscription.updated event.

    Updates local tier and status tracking when a subscription changes.

    Args:
        event_data: The event data.object from Stripe (subscription object).

    Returns:
        Status dict.
    """
    subscription = event_data
    customer_id = subscription.get("customer", "")
    sub_status = subscription.get("status", "unknown")
    metadata = subscription.get("metadata", {})
    tier = metadata.get("tier", "unknown")

    logger.info(
        "Subscription updated: customer=%s status=%s tier=%s",
        customer_id,
        sub_status,
        tier,
    )

    update_subscription_state(customer_id, {
        "subscription_id": subscription.get("id"),
        "status": sub_status,
        "tier": tier,
        "current_period_start": subscription.get("current_period_start"),
        "current_period_end": subscription.get("current_period_end"),
        "cancel_at_period_end": subscription.get("cancel_at_period_end", False),
    })

    return {"status": "processed", "event": "customer.subscription.updated"}


def _handle_subscription_deleted(event_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Handle customer.subscription.deleted event.

    Downgrades the customer to community tier when their subscription is cancelled.

    Args:
        event_data: The event data.object from Stripe (subscription object).

    Returns:
        Status dict.
    """
    subscription = event_data
    customer_id = subscription.get("customer", "")

    logger.info(
        "Subscription deleted: customer=%s subscription=%s",
        customer_id,
        subscription.get("id"),
    )

    # Downgrade to community
    update_subscription_state(customer_id, {
        "subscription_id": None,
        "status": "canceled",
        "tier": "community",
        "canceled_at": int(time.time()),
    })

    return {"status": "processed", "event": "customer.subscription.deleted"}


# Event type -> handler mapping
_EVENT_HANDLERS = {
    "invoice.payment_succeeded": _handle_payment_succeeded,
    "customer.subscription.updated": _handle_subscription_updated,
    "customer.subscription.deleted": _handle_subscription_deleted,
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
    """
    Receive and process Stripe webhook events.

    Verifies the webhook signature (when configured), dispatches to the
    appropriate handler, and returns a 200 response.
    """
    import os

    payload = await request.body()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    # Verify signature if secret is configured
    if webhook_secret:
        event = verify_stripe_signature(
            payload,
            stripe_signature or "",
            webhook_secret,
        )
    else:
        # Development mode: parse without verification
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

    logger.info("Received Stripe webhook: type=%s id=%s", event_type, event.get("id"))

    # Dispatch to handler
    handler = _EVENT_HANDLERS.get(event_type)
    if handler:
        result = handler(event_data)
        return JSONResponse(content=result, status_code=200)

    # Unhandled event type -- acknowledge but log
    logger.info("Ignoring unhandled Stripe event type: %s", event_type)
    return JSONResponse(
        content={"status": "ignored", "event": event_type},
        status_code=200,
    )
