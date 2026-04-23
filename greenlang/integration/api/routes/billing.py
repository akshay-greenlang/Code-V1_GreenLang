# -*- coding: utf-8 -*-
"""
GreenLang API - Billing / Checkout Routes (FY27 Pricing Page)
==============================================================

Customer-facing billing endpoints invoked by the public Pricing Page.

Routes:
    * ``POST /api/v1/billing/checkout``
        Create a Stripe Checkout Session for a tier subscription, optionally
        with one or more Premium Data Pack add-ons. Returns the hosted
        Checkout URL the frontend should redirect the browser to.

The webhook handler (``/api/v1/billing/webhooks``) is registered separately
in :mod:`greenlang.factors.billing.webhook_handler` and grants entitlements
once payment succeeds. This module covers only the customer-initiated
**create** flow.

Security
--------
* Authentication is required (``Depends(get_current_user)``). Anonymous
  Pricing Page traffic must first pass through the public sign-up flow to
  obtain a session token.
* The signed-receipts middleware (mounted on ``/api/v1/factors``) must NOT
  cover ``/api/v1/billing`` — billing responses are not factor data and
  must not carry GreenLang factor receipts.
* The Stripe Idempotency-Key prevents the frontend from double-creating
  a Checkout session on accidental retry / double-click.

Author: GreenLang Backend Team
Date: 2026-04-22
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import Field, HttpUrl

from greenlang.factors.billing.skus import CATALOG, PremiumPack, Tier
from greenlang.factors.billing.stripe_provider import StripeBillingProvider
from greenlang.integration.api.dependencies import get_current_user
from greenlang.schemas import GreenLangRequest, GreenLangResponse
from greenlang.utilities.exceptions.integration import BillingProviderError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/billing", tags=["Billing"])


# ---------------------------------------------------------------------------
# Provider singleton
# ---------------------------------------------------------------------------
#
# A process-wide StripeBillingProvider; lazy so unit tests can monkey-patch
# the env vars before the first call.

_provider: Optional[StripeBillingProvider] = None


def get_stripe_provider() -> StripeBillingProvider:
    """Return the process-wide :class:`StripeBillingProvider` (lazy)."""
    global _provider
    if _provider is None:
        _provider = StripeBillingProvider.from_environment()
    return _provider


def reset_stripe_provider_for_tests() -> None:
    """Test-only hook: drop the cached provider so the next call rebuilds it."""
    global _provider
    _provider = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CheckoutRequest(GreenLangRequest):
    """Customer request to create a Stripe Checkout session.

    The frontend Pricing Page POSTs this when the user clicks
    "Subscribe". The contract is intentionally minimal so the page does
    not need to know Stripe price IDs — it just sends GreenLang SKU names.
    """

    sku_name: str = Field(
        ...,
        description=(
            "Tier SKU name (one of: pro, consulting, platform, enterprise). "
            "Community is free and is rejected here — provision community "
            "tenants directly without Checkout."
        ),
        examples=["pro", "platform", "enterprise"],
    )
    success_url: HttpUrl = Field(
        ...,
        description=(
            "Absolute URL Stripe redirects to on successful payment. "
            "Stripe rejects relative paths and non-HTTPS URLs in production."
        ),
        examples=["https://greenlang.io/billing/success"],
    )
    cancel_url: HttpUrl = Field(
        ...,
        description="Absolute URL Stripe redirects to if the user abandons.",
        examples=["https://greenlang.io/pricing"],
    )
    premium_packs: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of Premium Data Pack SKU strings (e.g. "
            "'electricity_premium', 'freight_premium', 'cbam_premium'). "
            "Each entry adds a recurring add-on line item to the session."
        ),
        examples=[["electricity_premium", "freight_premium"]],
    )


class CheckoutResponse(GreenLangResponse):
    """Stripe Checkout session created — frontend redirects to ``url``."""

    session_id: str = Field(
        ...,
        description="Stripe Checkout Session ID (cs_test_... or cs_live_...).",
        examples=["cs_test_a1b2c3d4e5f6g7h8"],
    )
    url: HttpUrl = Field(
        ...,
        description="Hosted Checkout URL — redirect the browser here.",
        examples=["https://checkout.stripe.com/c/pay/cs_test_..."],
    )


# ---------------------------------------------------------------------------
# POST /api/v1/billing/checkout
# ---------------------------------------------------------------------------


@router.post(
    "/checkout",
    response_model=CheckoutResponse,
    status_code=status.HTTP_200_OK,
    summary="Create a Stripe Checkout session",
    description=(
        "Create a hosted Stripe Checkout session for the given tier SKU "
        "and any premium-pack add-ons. Returns the session ID and the "
        "redirect URL the Pricing Page should send the browser to.\n\n"
        "The frontend should call this endpoint with the user's chosen "
        "plan, then `window.location = response.url` to hand off to "
        "Stripe-hosted Checkout. After payment, Stripe redirects back to "
        "`success_url` and fires `checkout.session.completed` to "
        "`/api/v1/billing/webhooks` which grants entitlements server-side."
    ),
    responses={
        200: {
            "description": "Checkout session created",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "cs_test_a1b2c3d4e5f6g7h8i9j0",
                        "url": (
                            "https://checkout.stripe.com/c/pay/"
                            "cs_test_a1b2c3d4e5f6g7h8i9j0"
                        ),
                    }
                }
            },
        },
        400: {"description": "Invalid SKU name or premium pack"},
        401: {"description": "Authentication required"},
        502: {
            "description": (
                "Upstream billing provider error — see Retry-After header"
            )
        },
    },
)
async def create_checkout_session(
    request: Request,
    body: CheckoutRequest,
    current_user: dict = Depends(get_current_user),
) -> CheckoutResponse:
    """Create a Stripe Checkout session for the authenticated tenant.

    Resolves ``tenant_id`` from the authenticated user, validates the
    requested SKU and premium packs against the GreenLang catalog, and
    delegates to :meth:`StripeBillingProvider.create_checkout_session`.

    Args:
        request: FastAPI Request (used to pull tenant_id from auth state
            when set there by middleware).
        body: Validated :class:`CheckoutRequest` payload.
        current_user: Auth context from :func:`get_current_user`.

    Returns:
        :class:`CheckoutResponse` with the Stripe session id + URL.

    Raises:
        HTTPException: 400 for invalid SKU/pack, 401 for missing auth,
            502 for upstream Stripe failures.
    """
    # Resolve tenant_id: prefer request.state (set by tenant-resolver
    # middleware), then auth payload, then default. We never let it be
    # empty -- Stripe stores it on the session and the webhook needs it
    # to grant entitlements.
    tenant_id = (
        getattr(request.state, "tenant_id", None)
        or (current_user or {}).get("tenant_id")
        or "default"
    )
    customer_email = (current_user or {}).get("email")

    # ----- Validate SKU is in the catalog (fast-fail before hitting Stripe)
    sku_lower = (body.sku_name or "").lower().strip()
    try:
        tier = Tier(sku_lower)
    except ValueError as exc:
        valid = ", ".join(t.value for t in Tier if t != Tier.COMMUNITY)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown sku_name '{body.sku_name}'. "
                f"Valid SKU names: {valid}."
            ),
        ) from exc

    if tier == Tier.COMMUNITY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Community tier is free and does not require checkout. "
                "Sign up via /api/v1/auth/signup instead."
            ),
        )

    # Make sure the tier actually has a Stripe price configured. This is
    # an env-config issue, not a customer issue, so fail fast with 503.
    if not CATALOG.tier(tier).stripe_price_monthly_id:
        logger.error(
            "Tier '%s' has no Stripe price configured; cannot create checkout",
            sku_lower,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Billing not configured for tier '{sku_lower}'. "
                "Contact GreenLang support."
            ),
        )

    # ----- Validate premium packs
    premium_packs = body.premium_packs or []
    for pack_str in premium_packs:
        try:
            PremiumPack(str(pack_str).lower().strip())
        except ValueError as exc:
            valid_packs = ", ".join(p.value for p in PremiumPack)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unknown premium pack '{pack_str}'. "
                    f"Valid packs: {valid_packs}."
                ),
            ) from exc

    # ----- Delegate to the Stripe provider
    provider = get_stripe_provider()
    try:
        result = provider.create_checkout_session(
            sku_name=sku_lower,
            tenant_id=tenant_id,
            success_url=str(body.success_url),
            cancel_url=str(body.cancel_url),
            premium_packs=premium_packs or None,
            customer_email=customer_email,
        )
    except ValueError as exc:
        # Provider-side validation (e.g. unknown SKU snuck past us, or a
        # pack a tier isn't allowed to buy) — surface as 400.
        logger.warning(
            "Checkout validation rejected: tenant=%s sku=%s err=%s",
            tenant_id,
            sku_lower,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except BillingProviderError as exc:
        # Upstream Stripe failure — return 502 with a Retry-After hint.
        logger.error(
            "BillingProviderError on checkout create tenant=%s sku=%s "
            "status=%s msg=%s",
            tenant_id,
            sku_lower,
            exc.status_code,
            exc,
        )
        headers = {}
        if exc.retry_after_seconds:
            headers["Retry-After"] = str(exc.retry_after_seconds)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Billing provider unavailable. "
                "Please retry; if the problem persists contact support."
            ),
            headers=headers or None,
        ) from exc

    logger.info(
        "Created checkout session %s tenant=%s sku=%s packs=%s",
        result["session_id"],
        tenant_id,
        sku_lower,
        premium_packs,
    )
    return CheckoutResponse(
        session_id=result["session_id"],
        url=result["url"],
    )


__all__ = [
    "router",
    "CheckoutRequest",
    "CheckoutResponse",
    "get_stripe_provider",
    "reset_stripe_provider_for_tests",
]
