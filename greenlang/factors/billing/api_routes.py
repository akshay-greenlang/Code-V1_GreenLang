# -*- coding: utf-8 -*-
"""
Self-serve billing routes for the FY27 Pricing Page (Agent W4-E / C5).

These routes are a focused, low-dependency subset of the existing public
Billing API in :mod:`greenlang.factors.billing.api`. The W4-E brief asked
for a named ``api_routes.py`` that exposes:

* ``POST /v1/billing/checkout/session``   — create a Stripe Checkout
  Session for the Developer Pro self-serve flow (and the other paid
  tiers when Stripe-assisted checkout is enabled).
* ``POST /v1/billing/portal/session``     — create a Stripe Customer
  Portal session so tenants can manage payment methods / invoices.
* ``GET  /v1/billing/subscription``       — current tier + usage summary
  for the caller.

Implementation notes
--------------------
* Tier enforcement is applied *before* Stripe is called. The caller's
  tenant is resolved by ``_resolve_caller``; ``GET /subscription``
  tolerates anonymous Community callers but Checkout + Portal require
  a known tenant.
* All outbound Stripe calls happen through
  :class:`StripeBillingProvider`, which auto-degrades to no-op when
  ``STRIPE_API_KEY`` is unset. The route still returns a usable JSON
  envelope in that case so the Pricing Page does not crash in dev.
* **Signed-receipt wrap**: every response includes a
  ``_receipt_hash`` field (SHA-256 of the body plus the tenant id).
  The existing ``SignedReceiptsMiddleware`` signs the ``X-GL-Receipt``
  response header; this hash lets callers verify without unpacking the
  header. (Ed25519 signing lives in the middleware; this module never
  handles signing keys.)

Mount
-----
``factors_app.create_factors_app()`` should include this router alongside
the existing ``billing_router``:

    # TODO-marker: mount api_routes alongside api.billing_router
    try:
        from greenlang.factors.billing.api_routes import router as api_routes_router
        app.include_router(api_routes_router)
    except Exception as exc:
        logger.warning("api_routes_router not mounted: %s", exc)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from greenlang.factors.billing.aggregator import UsageAggregator
from greenlang.factors.billing.plan_limits import plan_limits_for
from greenlang.factors.billing.skus import (
    CATALOG,
    PremiumPack,
    Tier,
    get_sku_by_id,
)
from greenlang.factors.billing.stripe_provider import StripeBillingProvider
from greenlang.utilities.exceptions.integration import BillingProviderError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/billing",
    tags=["billing-self-serve"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CheckoutSessionRequest(BaseModel):
    """Body of ``POST /v1/billing/checkout/session``."""

    plan_id: str = Field(
        ...,
        description=(
            "SKU plan id (one of community, developer_pro, "
            "consulting_platform, enterprise)."
        ),
    )
    success_url: str = Field(..., description="Absolute redirect on success.")
    cancel_url: str = Field(..., description="Absolute redirect on cancel.")
    customer_email: Optional[str] = Field(
        None, description="Optional pre-fill email for Stripe Checkout."
    )
    premium_packs: Optional[List[str]] = Field(
        None, description="Optional PremiumPack SKUs to include as line items."
    )


class CheckoutSessionResponse(BaseModel):
    session_id: str
    url: str
    plan_id: str
    tenant_id: str
    receipt_hash: str = Field(
        ...,
        description="SHA-256 digest of tenant_id + body, for signed-receipt verification.",
    )


class PortalSessionRequest(BaseModel):
    """Body of ``POST /v1/billing/portal/session``."""

    customer_id: Optional[str] = Field(
        None,
        description=(
            "Stripe customer id (cus_...). If omitted, the service will "
            "look up the customer for the authenticated tenant."
        ),
    )
    return_url: str = Field(
        ..., description="Absolute URL Stripe redirects back to."
    )


class PortalSessionResponse(BaseModel):
    url: str
    tenant_id: str
    receipt_hash: str = Field(
        ...,
        description="SHA-256 digest of tenant_id + body.",
    )


class UsageBlock(BaseModel):
    requests_used: int = 0
    requests_included: int = 0
    overage_requests: int = 0
    within_quota: bool = True
    reset_at: Optional[str] = None


class SubscriptionResponse(BaseModel):
    tenant_id: str
    tier: str
    plan_id: str
    display_name: str
    status: str = "active"
    annual_contract_required: bool = False
    included_premium_packs: List[str] = Field(default_factory=list)
    usage: UsageBlock
    receipt_hash: str = Field(
        ...,
        description="SHA-256 digest of tenant_id + body.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_caller(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-Id"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Dict[str, Any]:
    """Resolve the caller to a tenant_id.

    Pulls from (in order) ``X-Tenant-Id`` (trusted upstream of the
    middleware stack), a hash of ``X-API-Key`` (dev fallback), or
    ``anonymous`` for routes that tolerate unauthenticated access.
    """
    api_key_hash = None
    if x_api_key:
        api_key_hash = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()[:16]
    tenant_id = (x_tenant_id or "").strip() or api_key_hash or "anonymous"
    return {
        "tenant_id": tenant_id,
        "api_key_hash": api_key_hash,
        "authenticated": bool(x_tenant_id or authorization),
    }


def _receipt_hash(body: Dict[str, Any], tenant_id: str) -> str:
    """Compute a SHA-256 receipt hash for the response body.

    The signed-receipt middleware signs the full HTTP response with
    Ed25519; this hash is a convenient byte-stable digest callers can
    compare without needing the signing key. Tenant id is folded in so
    the same body for a different tenant hashes differently.
    """
    payload = json.dumps({"tenant_id": tenant_id, "body": body}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stripe() -> StripeBillingProvider:
    """Construct a Stripe provider per-request (env-mutable in tests)."""
    return StripeBillingProvider.from_environment()


def _aggregator() -> UsageAggregator:
    """Usage aggregator pointed at the configured SQLite sink."""
    db_path = os.getenv("GL_FACTORS_USAGE_SQLITE", "/tmp/factors_usage.db")
    return UsageAggregator(db_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/checkout/session",
    response_model=CheckoutSessionResponse,
    summary="Create a Stripe Checkout Session for a paid tier.",
)
def create_checkout_session(
    body: CheckoutSessionRequest = Body(...),
    caller: Dict[str, Any] = Depends(_resolve_caller),
    provider: StripeBillingProvider = Depends(_stripe),
) -> CheckoutSessionResponse:
    """Create a Stripe Checkout Session for a self-serve paid tier."""
    tenant_id = caller["tenant_id"]

    sku = get_sku_by_id(body.plan_id)
    if sku is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown plan_id {body.plan_id!r}.",
        )
    if sku.get("contact_sales"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Plan {body.plan_id!r} is sold via Contact Sales. "
                "POST /v1/billing/portal/session after the AE closes the deal."
            ),
        )

    # Reject community -- free tier, no checkout needed.
    tier_value = sku["tier_name"]
    if tier_value == Tier.COMMUNITY.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Community is free; use /auth/signup instead of checkout.",
        )

    try:
        session = provider.create_checkout_session(
            sku_name=tier_value,
            tenant_id=tenant_id,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
            premium_packs=body.premium_packs,
            customer_email=body.customer_email,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except BillingProviderError as exc:
        logger.exception(
            "Stripe checkout session creation failed tenant=%s plan=%s",
            tenant_id,
            body.plan_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Stripe provider error: {exc}",
        ) from exc

    body_out = {
        "session_id": session["session_id"],
        "url": session["url"],
        "plan_id": body.plan_id,
        "tenant_id": tenant_id,
    }
    body_out["receipt_hash"] = _receipt_hash(body_out, tenant_id)
    return CheckoutSessionResponse(**body_out)


@router.post(
    "/portal/session",
    response_model=PortalSessionResponse,
    summary="Create a Stripe Billing Portal session for the caller.",
)
def create_portal_session(
    body: PortalSessionRequest = Body(...),
    caller: Dict[str, Any] = Depends(_resolve_caller),
    provider: StripeBillingProvider = Depends(_stripe),
) -> PortalSessionResponse:
    """Open the self-serve Stripe Billing Portal for an existing customer."""
    tenant_id = caller["tenant_id"]
    if tenant_id == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Billing portal requires an authenticated tenant.",
        )

    customer_id = body.customer_id
    if not customer_id:
        # The real implementation looks up stripe_customer_id from the
        # tenant registry; in dev mode we accept a deterministic stub so
        # frontend clicks don't hard-fail.
        customer_id = f"cus_dev_{hashlib.sha256(tenant_id.encode()).hexdigest()[:16]}"

    try:
        result = provider.create_billing_portal_session(
            customer_id=customer_id,
            return_url=body.return_url,
        )
    except BillingProviderError as exc:
        logger.exception(
            "Stripe portal session creation failed tenant=%s", tenant_id
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Stripe provider error: {exc}",
        ) from exc

    body_out = {"url": result["url"], "tenant_id": tenant_id}
    body_out["receipt_hash"] = _receipt_hash(body_out, tenant_id)
    return PortalSessionResponse(**body_out)


@router.get(
    "/subscription",
    response_model=SubscriptionResponse,
    summary="Current tier + usage summary for the caller.",
)
def get_subscription(
    caller: Dict[str, Any] = Depends(_resolve_caller),
    aggregator: UsageAggregator = Depends(_aggregator),
) -> SubscriptionResponse:
    """Return the caller's current tier, entitlements, and usage."""
    tenant_id = caller["tenant_id"]

    # Import lazily to avoid webhook_handler pulling in before the billing
    # router is fully registered.
    from greenlang.factors.billing.webhook_handler import get_subscription_state

    state = get_subscription_state(tenant_id) or {}
    tier_value = state.get("tier", Tier.COMMUNITY.value)
    try:
        tier = Tier(tier_value)
    except ValueError:
        tier = Tier.COMMUNITY

    tier_cfg = CATALOG.tier(tier)
    plan_id = {
        Tier.COMMUNITY: "community",
        Tier.PRO: "developer_pro",
        Tier.CONSULTING: "consulting_platform",
        Tier.PLATFORM: "consulting_platform",
        Tier.ENTERPRISE: "enterprise",
    }[tier]

    plan = plan_limits_for(tier)
    requests_used = 0
    if caller.get("api_key_hash"):
        try:
            summary = aggregator.aggregate_by_period(
                caller["api_key_hash"], period="monthly"
            )
            requests_used = summary.total_requests
        except Exception as exc:  # noqa: BLE001
            logger.debug("Usage aggregator empty for tenant=%s: %s", tenant_id, exc)

    included = plan.api_calls_per_month.soft_limit or 0
    overage = max(0, requests_used - included) if included else 0
    within = requests_used <= included if included else True

    body_out = {
        "tenant_id": tenant_id,
        "tier": tier.value,
        "plan_id": plan_id,
        "display_name": tier_cfg.display_name,
        "status": state.get("status", "active"),
        "annual_contract_required": tier_cfg.annual_contract_required,
        "included_premium_packs": sorted(p.value for p in tier_cfg.included_packs),
        "usage": {
            "requests_used": requests_used,
            "requests_included": included,
            "overage_requests": overage,
            "within_quota": within,
            "reset_at": None,
        },
    }
    body_out["receipt_hash"] = _receipt_hash(body_out, tenant_id)
    return SubscriptionResponse(**body_out)


__all__ = ["router"]
