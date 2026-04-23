# -*- coding: utf-8 -*-
"""FastAPI router that exposes the public Billing API.

This module is the **public surface** of the GreenLang Factors billing
system used by the FY27 Pricing Page and Developer Portal:

* ``GET  /v1/billing/plans``            -> public catalog of the 4 SKUs
* ``POST /v1/billing/checkout/{plan}``  -> creates a Stripe Checkout Session
* ``POST /v1/billing/webhook``          -> Stripe webhook receiver
* ``GET  /v1/billing/me``               -> caller's plan + usage summary
* ``POST /v1/billing/portal``           -> create a Stripe Billing Portal session

The router is intended to auto-mount in :mod:`greenlang.factors.factors_app`
via:

    try:
        from greenlang.factors.billing.api import billing_router
        app.include_router(billing_router)
    except Exception:
        pass

so the rest of the Factors app does not require a billing import at boot.

All requests are tagged ``billing`` for the OpenAPI grouping and use the
``/v1/billing`` prefix.  Per-route authentication is delegated to FastAPI
dependencies that resolve the caller's tenant identity (we accept both
``X-API-Key`` and ``Authorization: Bearer ...`` headers).

The handler is **failure-tolerant by design**: webhook dispatch and
checkout session creation always return JSON with a clear ``error`` field
on failure rather than raising 500s, because Stripe will retry webhooks
forever on non-2xx responses.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from greenlang.factors.billing.aggregator import UsageAggregator
from greenlang.factors.billing.skus import (
    PUBLIC_SKUS,
    Tier,
    get_skus,
    get_sku_by_id,
)
from greenlang.factors.billing.stripe_provider import (
    StripeApiError,
    StripeBillingProvider,
)
from greenlang.factors.billing import webhook_handler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

billing_router = APIRouter(
    prefix="/v1/billing",
    tags=["billing"],
)


# ---------------------------------------------------------------------------
# Pydantic surface models -- thin DTOs decoupled from the catalog dataclasses.
# ---------------------------------------------------------------------------


class RateLimitView(BaseModel):
    """Public rate-limit description shown on the Pricing Page."""

    requests_per_minute: int = Field(..., ge=0)
    requests_per_month_included: int = Field(..., ge=0)


class PlanView(BaseModel):
    """Public plan SKU as exposed by ``GET /v1/billing/plans``."""

    plan_id: str = Field(..., description="Stable plan identifier; safe to use as URL slug")
    display_name: str
    tagline: str
    price_usd_monthly: Optional[str] = Field(
        None,
        description=(
            "Monthly headline price as a decimal string. ``None`` for "
            "contact-sales SKUs (Enterprise)."
        ),
    )
    price_usd_annual: Optional[str] = Field(
        None,
        description="Annual price as a decimal string, when published.",
    )
    contact_sales: bool = Field(
        False,
        description="When True, the Pricing Page should render Contact Sales instead of Self-Serve Checkout.",
    )
    self_serve: bool = Field(
        True,
        description="When True, Stripe Checkout is available for this SKU.",
    )
    rate_limit: RateLimitView
    overage_unit_price_usd: Optional[str] = Field(
        None,
        description="Per-call overage price (USD) once included quota is exhausted.",
    )
    license_classes: List[str] = Field(
        default_factory=list,
        description=(
            "Factor license classes the SKU is entitled to access "
            "(e.g. open, restricted, customer_private)."
        ),
    )
    included_premium_packs: List[str] = Field(default_factory=list)
    included_sub_tenants: int = Field(0, ge=0)
    oem_redistribution: bool = False
    sla: Optional[str] = None
    features: List[str] = Field(default_factory=list)


class PlansResponse(BaseModel):
    """Envelope for the plans endpoint."""

    plans: List[PlanView]
    currency: str = "USD"
    stripe_publishable_key: Optional[str] = None


class CheckoutRequest(BaseModel):
    """Body of ``POST /v1/billing/checkout/{plan_id}``."""

    success_url: str = Field(
        ...,
        description="Absolute URL Stripe will redirect to on successful payment.",
    )
    cancel_url: str = Field(
        ...,
        description="Absolute URL Stripe will redirect to if the user abandons checkout.",
    )
    customer_email: Optional[str] = Field(
        None,
        description="Optional customer email for Checkout pre-fill.",
    )
    premium_packs: Optional[List[str]] = Field(
        default=None,
        description="Optional list of premium pack SKUs to include as line items.",
    )


class CheckoutResponse(BaseModel):
    session_id: str
    url: str
    plan_id: str


class PortalRequest(BaseModel):
    """Body for ``POST /v1/billing/portal``."""

    return_url: str = Field(
        ...,
        description="Absolute URL Stripe will redirect to after the customer leaves the portal.",
    )


class PortalResponse(BaseModel):
    url: str


class UsageSummaryView(BaseModel):
    """Public usage summary returned by ``/v1/billing/me``."""

    period: str = "monthly"
    requests_used: int = 0
    requests_included: int = 0
    requests_remaining: int = 0
    overage_requests: int = 0
    within_quota: bool = True
    reset_at: Optional[str] = None


class MeResponse(BaseModel):
    tenant_id: Optional[str] = None
    plan_id: str
    plan: PlanView
    status: str = "active"
    usage: UsageSummaryView


# ---------------------------------------------------------------------------
# Caller identity helpers
# ---------------------------------------------------------------------------


class CallerContext(BaseModel):
    """Resolved caller identity for billing routes."""

    tenant_id: str
    api_key_hash: Optional[str] = None
    bearer_present: bool = False


async def _resolve_caller(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-Id"),
) -> CallerContext:
    """Resolve caller identity for billing endpoints.

    We do not own auth here -- the higher-level ``factors_app`` middleware
    is responsible for full JWT verification. We only need to know which
    tenant to attribute the request to.

    Resolution order:
      1. ``X-Tenant-Id`` header (set by upstream auth middleware after JWT verify).
      2. SHA-256 prefix of the API key (for keyless dev mode this still
         gives us a stable per-key identity).
      3. ``"anonymous"`` fallback for routes that tolerate it.
    """
    import hashlib

    api_key_hash: Optional[str] = None
    if x_api_key:
        api_key_hash = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()[:16]

    tenant_id: str = (
        (x_tenant_id or "").strip()
        or api_key_hash
        or "anonymous"
    )
    return CallerContext(
        tenant_id=tenant_id,
        api_key_hash=api_key_hash,
        bearer_present=bool(authorization),
    )


def _stripe_provider() -> StripeBillingProvider:
    """Construct a Stripe provider from environment variables on each call.

    Constructed per-request rather than at import so tests can mutate
    ``STRIPE_API_KEY`` in-place via ``monkeypatch.setenv``.
    """
    return StripeBillingProvider.from_environment()


def _usage_aggregator() -> UsageAggregator:
    """Return a UsageAggregator pointed at the configured SQLite sink.

    Falls back to an in-memory path so the route returns zeros gracefully
    when the usage sink has never been initialised (e.g. fresh dev box).
    """
    db_path = os.getenv("GL_FACTORS_USAGE_SQLITE", "/tmp/factors_usage.db")
    return UsageAggregator(db_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@billing_router.get(
    "/plans",
    response_model=PlansResponse,
    summary="List the public Factors API plans.",
    description=(
        "Returns the four canonical SKUs (Community, Developer Pro, "
        "Consulting/Platform, Enterprise) with prices, rate limits, and "
        "license-class entitlements. Drives the pricing page."
    ),
)
def list_plans() -> PlansResponse:
    """Return the public SKU catalog."""
    plans = [PlanView(**sku) for sku in get_skus()]
    publishable_key = os.getenv("STRIPE_PUBLISHABLE_KEY") or None
    return PlansResponse(plans=plans, stripe_publishable_key=publishable_key)


@billing_router.post(
    "/checkout/{plan_id}",
    response_model=CheckoutResponse,
    summary="Create a Stripe Checkout Session for a plan.",
    description=(
        "Creates a hosted Stripe Checkout Session for the given SKU. "
        "Returns the redirect URL the browser should send the user to. "
        "Only self-serve plans (Community is auto-provisioned, Developer Pro "
        "is real Stripe checkout) are accepted; Consulting and Enterprise "
        "must be sold via Contact Sales."
    ),
)
def create_checkout(
    plan_id: str = Path(..., description="One of: community, developer_pro, consulting_platform, enterprise"),
    body: CheckoutRequest = Body(...),
    caller: CallerContext = Depends(_resolve_caller),
) -> CheckoutResponse:
    """Create a Stripe Checkout Session for ``plan_id``."""
    sku = get_sku_by_id(plan_id)
    if sku is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown plan_id: {plan_id!r}",
        )
    if not sku.get("self_serve", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Plan {plan_id!r} is not self-serve. Use 'Contact Sales' to "
                "purchase Consulting or Enterprise SKUs."
            ),
        )

    # Community auto-provisions: no Stripe Checkout needed; we return a
    # synthetic session pointing straight at the success URL so the UI
    # flow is identical to a paid sign-up.
    if plan_id == "community":
        return CheckoutResponse(
            session_id="cs_community_freeprovision",
            url=body.success_url,
            plan_id=plan_id,
        )

    provider = _stripe_provider()
    tier_name = sku["tier_name"]

    try:
        session = provider.create_checkout_session(
            sku_name=tier_name,
            tenant_id=caller.tenant_id,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
            premium_packs=body.premium_packs or [],
            customer_email=body.customer_email,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except StripeApiError as exc:
        logger.error(
            "Stripe checkout failed for plan=%s tenant=%s status=%s",
            plan_id,
            caller.tenant_id,
            exc.status_code,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Stripe checkout failed: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - safety net
        logger.exception(
            "Unexpected billing checkout failure plan=%s tenant=%s",
            plan_id,
            caller.tenant_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected billing checkout failure: {exc}",
        ) from exc

    return CheckoutResponse(
        session_id=session["session_id"],
        url=session["url"],
        plan_id=plan_id,
    )


@billing_router.post(
    "/webhook",
    summary="Stripe webhook receiver.",
    description=(
        "Receives Stripe webhook events and dispatches them to the "
        "billing webhook handler with HMAC-SHA256 signature verification "
        "(via STRIPE_WEBHOOK_SECRET). Always returns 200 for events we "
        "ignore so Stripe does not retry forever."
    ),
)
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
) -> JSONResponse:
    """Receive a Stripe webhook event and dispatch to ``webhook_handler.handle``."""
    payload = await request.body()
    try:
        result = webhook_handler.handle(
            payload=payload,
            signature_header=stripe_signature or "",
            webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET", ""),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - convert to 400 not 500
        logger.exception("Unexpected webhook dispatch failure")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "error": str(exc)},
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content=result)


@billing_router.get(
    "/me",
    response_model=MeResponse,
    summary="Caller's current plan + usage summary.",
    description=(
        "Returns the caller's tenant, the SKU they are subscribed to, "
        "and a snapshot of monthly API usage (used / included / remaining)."
    ),
)
def me(
    caller: CallerContext = Depends(_resolve_caller),
) -> MeResponse:
    """Return the caller's current plan + usage."""
    state = webhook_handler.get_subscription_state(caller.tenant_id) or {}
    tier_value = (state.get("tier") or "community").lower()

    # Map internal Tier enum (5 values) -> public 4-SKU plan_id
    plan_id = _tier_to_plan_id(tier_value)
    sku = get_sku_by_id(plan_id) or get_sku_by_id("community")
    plan = PlanView(**sku)  # type: ignore[arg-type]

    usage_view = UsageSummaryView(
        period="monthly",
        requests_used=0,
        requests_included=plan.rate_limit.requests_per_month_included,
        requests_remaining=plan.rate_limit.requests_per_month_included,
        within_quota=True,
    )
    if caller.api_key_hash:
        try:
            aggregator = _usage_aggregator()
            quota = aggregator.check_quota(
                api_key_hash=caller.api_key_hash,
                tier=tier_value if tier_value in ("community", "pro", "enterprise") else "community",
            )
            usage_view = UsageSummaryView(
                period="monthly",
                requests_used=quota.used,
                requests_included=quota.allowed,
                requests_remaining=quota.remaining,
                overage_requests=quota.overage_amount,
                within_quota=quota.within_quota,
                reset_at=quota.reset_at.isoformat(),
            )
        except Exception as exc:  # noqa: BLE001 - never break /me on bad sink
            logger.warning("Usage aggregator failure: %s", exc)

    return MeResponse(
        tenant_id=caller.tenant_id,
        plan_id=plan_id,
        plan=plan,
        status=state.get("status", "active"),
        usage=usage_view,
    )


@billing_router.post(
    "/portal",
    response_model=PortalResponse,
    summary="Create a Stripe Billing Portal session for the caller.",
    description=(
        "Creates a Stripe Billing Portal session and returns the redirect "
        "URL. The portal lets customers update payment methods, view "
        "invoices, and cancel subscriptions self-service."
    ),
)
def create_portal_session(
    body: PortalRequest = Body(...),
    caller: CallerContext = Depends(_resolve_caller),
) -> PortalResponse:
    """Create a Stripe Billing Portal session for ``caller``."""
    state = webhook_handler.get_subscription_state(caller.tenant_id) or {}
    customer_id = state.get("customer_id") or state.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No Stripe customer linked to this tenant. The Billing "
                "Portal is only available after a successful checkout."
            ),
        )

    provider = _stripe_provider()
    try:
        session = provider.create_billing_portal_session(
            customer_id=customer_id,
            return_url=body.return_url,
        )
    except StripeApiError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Stripe Billing Portal failed: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Billing Portal creation failed for tenant=%s", caller.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return PortalResponse(url=session["url"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_TIER_TO_PLAN: Dict[str, str] = {
    "community": "community",
    "pro": "developer_pro",
    "consulting": "consulting_platform",
    "platform": "consulting_platform",
    "enterprise": "enterprise",
}


def _tier_to_plan_id(tier_value: str) -> str:
    """Collapse an internal tier (5 values) into a public plan id (4 values).

    The catalog distinguishes Consulting (entry rung) from Platform (upper
    band) to keep ACV negotiations clean, but the Pricing Page surfaces
    one combined "Consulting / Platform" SKU.
    """
    return _TIER_TO_PLAN.get((tier_value or "").lower(), "community")


__all__ = [
    "billing_router",
    "PlansResponse",
    "PlanView",
    "CheckoutRequest",
    "CheckoutResponse",
    "MeResponse",
    "PortalRequest",
    "PortalResponse",
]
