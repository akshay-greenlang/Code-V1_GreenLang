# -*- coding: utf-8 -*-
"""
Stripe billing provider for the GreenLang Factors API.

Implements abstract BillingProvider interface with a concrete Stripe
implementation. Supports subscription management, usage-based billing,
and invoice retrieval.

Tier pricing:
    - Community: Free ($0/mo, 1,000 req/mo)
    - Pro: $299/mo (50,000 req/mo, $0.01/overage request)
    - Enterprise: $999/mo (500,000 req/mo, $0.005/overage request)

Environment variables:
    STRIPE_API_KEY: Stripe secret API key (sk_...)
    STRIPE_WEBHOOK_SECRET: Webhook endpoint signing secret (whsec_...)

Example:
    >>> provider = StripeBillingProvider.from_environment()
    >>> sub = provider.create_subscription("cus_abc123", "pro")
    >>> print(sub["subscription_id"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from base64 import b64encode
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from greenlang.utilities.exceptions.integration import BillingProviderError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierConfig:
    """Pricing and quota configuration for a billing tier."""

    name: str
    display_name: str
    monthly_price_cents: int  # in cents
    monthly_quota: int
    overage_price_cents: float  # per request, in cents
    stripe_price_id: Optional[str] = None
    stripe_overage_price_id: Optional[str] = None


# Default tier configurations (price IDs set via environment or Stripe dashboard)
TIER_CONFIGS: Dict[str, TierConfig] = {
    "community": TierConfig(
        name="community",
        display_name="Community",
        monthly_price_cents=0,
        monthly_quota=1_000,
        overage_price_cents=0.0,
        stripe_price_id=os.getenv("STRIPE_PRICE_COMMUNITY"),
        stripe_overage_price_id=None,
    ),
    "pro": TierConfig(
        name="pro",
        display_name="Pro",
        monthly_price_cents=29_900,  # $299.00
        monthly_quota=50_000,
        overage_price_cents=1.0,  # $0.01
        stripe_price_id=os.getenv("STRIPE_PRICE_PRO"),
        stripe_overage_price_id=os.getenv("STRIPE_OVERAGE_PRICE_PRO"),
    ),
    "enterprise": TierConfig(
        name="enterprise",
        display_name="Enterprise",
        monthly_price_cents=99_900,  # $999.00
        monthly_quota=500_000,
        overage_price_cents=0.5,  # $0.005
        stripe_price_id=os.getenv("STRIPE_PRICE_ENTERPRISE"),
        stripe_overage_price_id=os.getenv("STRIPE_OVERAGE_PRICE_ENTERPRISE"),
    ),
}


# ---------------------------------------------------------------------------
# Abstract billing provider
# ---------------------------------------------------------------------------


class BillingProvider(ABC):
    """
    Abstract base class for billing integrations.

    All billing backends (Stripe, manual, etc.) must implement this interface
    to provide subscription lifecycle, usage recording, and invoice management.
    """

    @abstractmethod
    def create_subscription(
        self,
        customer_id: str,
        tier: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new subscription for a customer.

        Args:
            customer_id: External billing system customer ID.
            tier: Pricing tier name (community, pro, enterprise).
            metadata: Optional key-value pairs for the subscription.

        Returns:
            Dict with subscription_id, status, tier, and created_at.
        """

    @abstractmethod
    def record_usage(
        self,
        subscription_id: str,
        quantity: int,
        timestamp: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record metered usage for overage billing.

        Args:
            subscription_id: Active subscription ID.
            quantity: Number of overage requests to record.
            timestamp: Unix epoch timestamp (defaults to now).
            idempotency_key: Prevents duplicate recording.

        Returns:
            Dict with usage_record_id and quantity.
        """

    @abstractmethod
    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """
        Retrieve an invoice by ID.

        Args:
            invoice_id: The invoice identifier.

        Returns:
            Dict with invoice details (amount, status, line items, etc.).
        """

    @abstractmethod
    def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> Dict[str, Any]:
        """
        Cancel a subscription.

        Args:
            subscription_id: The subscription to cancel.
            at_period_end: If True, cancel at end of billing period.

        Returns:
            Dict with subscription_id and cancellation status.
        """

    @abstractmethod
    def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """
        Retrieve subscription details.

        Args:
            subscription_id: The subscription identifier.

        Returns:
            Dict with subscription details.
        """

    @abstractmethod
    def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new customer in the billing system.

        Args:
            email: Customer email address.
            name: Optional customer display name.
            metadata: Optional key-value pairs.

        Returns:
            Dict with customer_id and email.
        """

    @abstractmethod
    def list_invoices(
        self,
        customer_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        List invoices for a customer.

        Args:
            customer_id: The customer identifier.
            limit: Maximum number of invoices to return.

        Returns:
            List of invoice dicts.
        """


# ---------------------------------------------------------------------------
# Stripe implementation
# ---------------------------------------------------------------------------


class StripeBillingProvider(BillingProvider):
    """
    Stripe billing provider for the Factors API.

    Uses Stripe's REST API directly via urllib (no stripe-python dependency).
    Degrades gracefully when STRIPE_API_KEY is not set.

    Attributes:
        api_key: Stripe secret API key.
        webhook_secret: Stripe webhook signing secret.
        configured: Whether Stripe credentials are available.
    """

    STRIPE_API_BASE = "https://api.stripe.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ) -> None:
        """
        Initialize the Stripe billing provider.

        Args:
            api_key: Stripe secret key (sk_...). Falls back to STRIPE_API_KEY env var.
            webhook_secret: Webhook signing secret. Falls back to STRIPE_WEBHOOK_SECRET env var.
        """
        self.api_key = api_key or os.getenv("STRIPE_API_KEY", "")
        self.webhook_secret = webhook_secret or os.getenv("STRIPE_WEBHOOK_SECRET", "")
        self.configured = bool(self.api_key)

        if not self.configured:
            logger.warning(
                "STRIPE_API_KEY not set; Stripe billing provider will operate "
                "in no-op mode. Set the environment variable to enable billing."
            )

    @classmethod
    def from_environment(cls) -> StripeBillingProvider:
        """Create a StripeBillingProvider from environment variables."""
        return cls(
            api_key=os.getenv("STRIPE_API_KEY"),
            webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET"),
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _stripe_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the Stripe API.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: API path (e.g., "/customers").
            data: Form-encoded body data for POST requests.
            idempotency_key: Optional value for the ``Idempotency-Key`` header.
                When set Stripe returns the original response for any retry
                using the same key (within a 24-hour window), preventing
                duplicate side effects.

        Returns:
            Parsed JSON response dict.

        Raises:
            RuntimeError: If Stripe is not configured.
            StripeApiError: If Stripe returns a non-2xx response.
        """
        if not self.configured:
            raise RuntimeError(
                "Stripe billing provider is not configured. "
                "Set STRIPE_API_KEY environment variable."
            )

        url = f"{self.STRIPE_API_BASE}{path}"
        auth = b64encode(f"{self.api_key}:".encode("utf-8")).decode("ascii")
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        body = None
        if data:
            body = urlencode(self._flatten_params(data)).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            logger.error(
                "Stripe API error: %s %s -> HTTP %d: %s",
                method,
                path,
                e.code,
                error_body[:500],
            )
            raise StripeApiError(e.code, error_body) from e

    @staticmethod
    def _flatten_params(
        data: Dict[str, Any], prefix: str = ""
    ) -> List[tuple[str, str]]:
        """
        Flatten nested dicts into Stripe-style form params.

        Example: {"metadata": {"org": "acme"}} -> [("metadata[org]", "acme")]
        """
        items: List[tuple[str, str]] = []
        for key, value in data.items():
            full_key = f"{prefix}[{key}]" if prefix else key
            if isinstance(value, dict):
                items.extend(
                    StripeBillingProvider._flatten_params(value, full_key)
                )
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    items.append((f"{full_key}[{i}]", str(item)))
            elif value is not None:
                items.append((full_key, str(value)))
        return items

    # ------------------------------------------------------------------
    # No-op fallback for unconfigured state
    # ------------------------------------------------------------------

    def _noop_response(self, operation: str, **kwargs: Any) -> Dict[str, Any]:
        """Return a no-op response when Stripe is not configured."""
        logger.info(
            "Stripe not configured; returning no-op response for %s",
            operation,
        )
        return {
            "status": "noop",
            "message": f"Stripe not configured; {operation} skipped",
            **kwargs,
        }

    # ------------------------------------------------------------------
    # BillingProvider implementation
    # ------------------------------------------------------------------

    def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a Stripe customer."""
        if not self.configured:
            return self._noop_response("create_customer", email=email)

        params: Dict[str, Any] = {"email": email}
        if name:
            params["name"] = name
        if metadata:
            params["metadata"] = metadata

        result = self._stripe_request("POST", "/customers", params)
        logger.info("Created Stripe customer: %s", result.get("id"))
        return {
            "customer_id": result["id"],
            "email": result.get("email", email),
            "created_at": result.get("created"),
        }

    def create_subscription(
        self,
        customer_id: str,
        tier: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a Stripe subscription for the given tier."""
        tier_lower = tier.lower()
        tier_cfg = TIER_CONFIGS.get(tier_lower)
        if not tier_cfg:
            raise ValueError(
                f"Unknown tier: {tier}. Must be one of: {list(TIER_CONFIGS.keys())}"
            )

        if not self.configured:
            return self._noop_response(
                "create_subscription",
                customer_id=customer_id,
                tier=tier_lower,
            )

        # Community tier: no Stripe subscription needed
        if tier_lower == "community":
            logger.info(
                "Community tier does not require a Stripe subscription for %s",
                customer_id,
            )
            return {
                "subscription_id": None,
                "status": "active",
                "tier": "community",
                "message": "Community tier is free; no subscription created.",
            }

        if not tier_cfg.stripe_price_id:
            raise ValueError(
                f"Stripe price ID not configured for tier '{tier_lower}'. "
                f"Set STRIPE_PRICE_{tier_lower.upper()} environment variable."
            )

        params: Dict[str, Any] = {
            "customer": customer_id,
            "items[0][price]": tier_cfg.stripe_price_id,
        }
        # Add overage metered item if configured
        if tier_cfg.stripe_overage_price_id:
            params["items[1][price]"] = tier_cfg.stripe_overage_price_id

        if metadata:
            params["metadata"] = metadata

        result = self._stripe_request("POST", "/subscriptions", params)

        logger.info(
            "Created subscription %s for customer %s (tier=%s)",
            result.get("id"),
            customer_id,
            tier_lower,
        )

        return {
            "subscription_id": result["id"],
            "status": result.get("status", "active"),
            "tier": tier_lower,
            "created_at": result.get("created"),
            "current_period_end": result.get("current_period_end"),
        }

    # ------------------------------------------------------------------
    # Checkout Session creation (FY27 Pricing Page customer flow)
    # ------------------------------------------------------------------

    @staticmethod
    def _create_checkout_session_idempotency_key(
        tenant_id: str,
        sku_name: str,
        premium_packs: Optional[List[str]] = None,
    ) -> str:
        """Build a stable idempotency key for ``create_checkout_session``.

        The Pricing Page may double-fire the request on flaky networks or
        eager retries. By giving Stripe the same idempotency key for the
        same logical request we guarantee a single session is created and
        identical responses are returned.

        We deliberately do NOT include ``success_url`` / ``cancel_url`` —
        the user might bounce between localhost and production while
        debugging, but it's still the same purchase intent for the same
        tenant + SKU + packs combo.

        Args:
            tenant_id: GreenLang tenant identifier.
            sku_name: Tier SKU name (community, pro, platform, enterprise).
            premium_packs: Optional list of premium pack SKU strings.

        Returns:
            A 64-char hex digest suitable for Stripe ``Idempotency-Key``.
        """
        packs_normalized = ",".join(sorted(premium_packs or []))
        payload = f"{tenant_id}|{sku_name.lower()}|{packs_normalized}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def create_checkout_session(
        self,
        sku_name: str,
        tenant_id: str,
        success_url: str,
        cancel_url: str,
        premium_packs: Optional[List[str]] = None,
        customer_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Stripe Checkout Session for the FY27 Pricing Page.

        Resolves the tier SKU and any premium-pack add-ons via
        :data:`greenlang.factors.billing.skus.CATALOG`, builds the
        ``line_items`` array, and asks Stripe for a hosted Checkout
        session.  The session ``client_reference_id`` is the GreenLang
        tenant ID so the webhook handler can grant entitlements.

        The webhook ``checkout.session.completed`` handler reads
        ``metadata.sku_name`` and ``metadata.premium_packs`` to grant the
        right tier + packs after payment succeeds — see
        :func:`greenlang.factors.billing.webhook_handler._handle_checkout_completed`.

        Args:
            sku_name: Tier SKU name (e.g. ``"pro"``, ``"platform"``,
                ``"enterprise"``). Case-insensitive; must resolve to a
                tier in :data:`CATALOG`. Community is rejected (free
                tier needs no checkout).
            tenant_id: GreenLang tenant ID. Stored as Stripe
                ``client_reference_id`` and in session metadata.
            success_url: Absolute URL to redirect to on successful payment.
                Must be HTTPS in production; Stripe rejects relative paths.
            cancel_url: Absolute URL to redirect to if the user abandons.
            premium_packs: Optional list of premium pack SKU strings (the
                ``PremiumPack`` enum values, e.g. ``"electricity_premium"``).
                Each pack adds an extra line item billed at the Pro/Consulting
                monthly add-on rate.
            customer_email: Optional pre-filled customer email; Stripe will
                show it on the Checkout page and use it on the new Customer.

        Returns:
            ``{"session_id": <stripe session id>, "url": <hosted checkout url>}``.

        Raises:
            ValueError: If ``sku_name`` is unknown, if a ``premium_packs``
                entry is unknown, or if a required Stripe price ID is not
                configured for the resolved SKU.
            BillingProviderError: If the Stripe API call fails.
        """
        # Lazy import to avoid a circular import: skus.py reads from
        # entitlements which can pull in billing globals at module-load time.
        from greenlang.factors.billing.skus import (
            CATALOG,
            PremiumPack,
            Tier,
            allowed_for,
        )

        # ---- Resolve tier SKU ------------------------------------------------
        sku_lower = (sku_name or "").lower().strip()
        try:
            tier = Tier(sku_lower)
        except ValueError as exc:
            valid = ", ".join(t.value for t in Tier)
            raise ValueError(
                f"Unknown SKU '{sku_name}'. Valid SKU names: {valid}"
            ) from exc

        if tier == Tier.COMMUNITY:
            # Community is free — no Stripe Checkout needed. The Pricing
            # Page should never call us for the free tier; reject loudly.
            raise ValueError(
                "Community tier is free and does not require Stripe Checkout. "
                "Provision the tenant directly without a checkout session."
            )

        tier_cfg = CATALOG.tier(tier)
        tier_price_id = tier_cfg.stripe_price_monthly_id
        if not tier_price_id:
            raise ValueError(
                f"No Stripe price ID configured for tier '{sku_lower}'. "
                "Run the Stripe provisioning script before serving Checkout."
            )

        line_items: List[Dict[str, Any]] = [
            {"price": tier_price_id, "quantity": 1},
        ]

        # ---- Resolve premium packs ------------------------------------------
        packs_resolved: List[str] = []
        for pack_str in premium_packs or []:
            try:
                pack = PremiumPack(str(pack_str).lower().strip())
            except ValueError as exc:
                valid_packs = ", ".join(p.value for p in PremiumPack)
                raise ValueError(
                    f"Unknown premium pack '{pack_str}'. "
                    f"Valid packs: {valid_packs}"
                ) from exc

            if not allowed_for(tier, pack):
                # Community can't buy packs; everything else can.
                raise ValueError(
                    f"Tier '{sku_lower}' is not allowed to purchase pack "
                    f"'{pack.value}'."
                )

            pack_cfg = CATALOG.pack(pack)
            pack_price_id = pack_cfg.stripe_price_monthly_id
            if not pack_price_id:
                raise ValueError(
                    f"No Stripe price ID configured for pack '{pack.value}'."
                )
            line_items.append({"price": pack_price_id, "quantity": 1})
            packs_resolved.append(pack.value)

        # ---- No-op fallback when Stripe key is unset (dev mode) -------------
        if not self.configured:
            fake_session_id = (
                "cs_test_noop_"
                + self._create_checkout_session_idempotency_key(
                    tenant_id, sku_lower, packs_resolved
                )[:24]
            )
            logger.info(
                "Stripe not configured; returning no-op Checkout session "
                "tenant=%s sku=%s packs=%s",
                tenant_id,
                sku_lower,
                packs_resolved,
            )
            return {
                "session_id": fake_session_id,
                "url": f"{success_url}?noop_session={fake_session_id}",
            }

        # ---- Build the Stripe params ----------------------------------------
        params: Dict[str, Any] = {
            "mode": "subscription",
            "line_items": line_items,
            "client_reference_id": tenant_id,
            "success_url": success_url,
            "cancel_url": cancel_url,
            "allow_promotion_codes": "true",
            "metadata": {
                "tenant_id": tenant_id,
                "sku_name": sku_lower,
                "premium_packs": ",".join(packs_resolved),
            },
            # Mirror metadata onto the resulting subscription so the
            # ``customer.subscription.created`` webhook can also see it
            # without round-tripping through the session object.
            "subscription_data": {
                "metadata": {
                    "tenant_id": tenant_id,
                    "tier": sku_lower,
                    "premium_packs": ",".join(packs_resolved),
                },
            },
        }
        if customer_email:
            params["customer_email"] = customer_email

        idempotency_key = self._create_checkout_session_idempotency_key(
            tenant_id, sku_lower, packs_resolved
        )

        try:
            result = self._stripe_request(
                "POST",
                "/checkout/sessions",
                params,
                idempotency_key=idempotency_key,
            )
        except StripeApiError as exc:
            logger.error(
                "Stripe checkout session creation failed: tenant=%s sku=%s "
                "packs=%s status=%s",
                tenant_id,
                sku_lower,
                packs_resolved,
                exc.status_code,
            )
            # Suggest a retry-after of 30 seconds for transient (5xx) errors.
            retry_after = 30 if exc.status_code >= 500 else None
            raise BillingProviderError(
                message=f"Stripe checkout session creation failed: {exc}",
                provider="stripe",
                operation="create_checkout_session",
                status_code=exc.status_code,
                retry_after_seconds=retry_after,
                cause=exc,
                context={
                    "tenant_id": tenant_id,
                    "sku_name": sku_lower,
                    "premium_packs": packs_resolved,
                },
            ) from exc
        except Exception as exc:  # noqa: BLE001 -- final safety net
            logger.exception(
                "Unexpected error creating Stripe checkout session "
                "tenant=%s sku=%s",
                tenant_id,
                sku_lower,
            )
            raise BillingProviderError(
                message=(
                    "Unexpected error creating Stripe checkout session: "
                    f"{exc}"
                ),
                provider="stripe",
                operation="create_checkout_session",
                cause=exc,
                context={
                    "tenant_id": tenant_id,
                    "sku_name": sku_lower,
                    "premium_packs": packs_resolved,
                },
            ) from exc

        session_id = result.get("id")
        session_url = result.get("url")
        if not session_id or not session_url:
            raise BillingProviderError(
                message=(
                    "Stripe checkout session response missing id/url: "
                    f"{result}"
                ),
                provider="stripe",
                operation="create_checkout_session",
            )

        logger.info(
            "Created Stripe checkout session %s tenant=%s sku=%s packs=%s",
            session_id,
            tenant_id,
            sku_lower,
            packs_resolved,
        )
        return {"session_id": session_id, "url": session_url}

    def create_billing_portal_session(
        self,
        customer_id: str,
        return_url: str,
        configuration: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Stripe Billing Portal session for a customer.

        The Billing Portal is Stripe's hosted self-service surface where
        customers can update payment methods, see invoice history, change
        plans (when allowed by the configuration), and cancel.

        Args:
            customer_id: Stripe customer id (``cus_...``).
            return_url: Absolute URL Stripe redirects the user back to
                when they leave the portal.
            configuration: Optional Stripe Billing Portal configuration id
                (``bpc_...``). When omitted, Stripe uses the default
                configuration on the account.

        Returns:
            ``{"session_id": ..., "url": ...}`` -- redirect the browser
            to ``url`` to deliver the customer into the portal.

        Raises:
            BillingProviderError: when the Stripe call fails.
        """
        if not self.configured:
            return {
                "session_id": "bps_test_noop_" + (customer_id or "")[:24],
                "url": return_url,
                "status": "noop",
            }

        params: Dict[str, Any] = {
            "customer": customer_id,
            "return_url": return_url,
        }
        if configuration:
            params["configuration"] = configuration

        try:
            result = self._stripe_request(
                "POST",
                "/billing_portal/sessions",
                params,
            )
        except StripeApiError as exc:
            raise BillingProviderError(
                message=f"Stripe billing portal session creation failed: {exc}",
                provider="stripe",
                operation="create_billing_portal_session",
                status_code=exc.status_code,
                cause=exc,
                context={"customer_id": customer_id},
            ) from exc

        session_id = result.get("id")
        url = result.get("url")
        if not session_id or not url:
            raise BillingProviderError(
                message=(
                    "Stripe billing portal response missing id/url: "
                    f"{result}"
                ),
                provider="stripe",
                operation="create_billing_portal_session",
            )
        logger.info(
            "Created Stripe Billing Portal session %s for customer %s",
            session_id,
            customer_id,
        )
        return {"session_id": session_id, "url": url}

    def report_usage(
        self,
        subscription_id: str,
        quantity: int,
        *,
        timestamp: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Report metered usage to Stripe (alias for :meth:`record_usage`).

        Provides a name aligned with Stripe's own ``UsageRecord`` API so
        the FY27 billing surface reads naturally. All semantics, including
        idempotency and the metered-item lookup, are inherited from
        :meth:`record_usage`.
        """
        return self.record_usage(
            subscription_id=subscription_id,
            quantity=quantity,
            timestamp=timestamp,
            idempotency_key=idempotency_key,
        )

    def record_usage(
        self,
        subscription_id: str,
        quantity: int,
        timestamp: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record metered usage on a subscription item for overage billing."""
        if not self.configured:
            return self._noop_response(
                "record_usage",
                subscription_id=subscription_id,
                quantity=quantity,
            )

        # First, fetch subscription to find the metered item
        sub = self._stripe_request("GET", f"/subscriptions/{subscription_id}")
        items = sub.get("items", {}).get("data", [])

        # Find the metered (overage) subscription item
        metered_item_id = None
        for item in items:
            price = item.get("price", {})
            if price.get("recurring", {}).get("usage_type") == "metered":
                metered_item_id = item["id"]
                break

        if not metered_item_id:
            logger.warning(
                "No metered subscription item found for %s; skipping usage record",
                subscription_id,
            )
            return {
                "status": "skipped",
                "message": "No metered item on subscription",
            }

        params: Dict[str, Any] = {
            "quantity": quantity,
            "action": "increment",
        }
        if timestamp:
            params["timestamp"] = timestamp

        headers_extra = {}
        if idempotency_key:
            headers_extra["Idempotency-Key"] = idempotency_key

        result = self._stripe_request(
            "POST",
            f"/subscription_items/{metered_item_id}/usage_records",
            params,
        )

        logger.info(
            "Recorded %d usage units for subscription %s (item %s)",
            quantity,
            subscription_id,
            metered_item_id,
        )

        return {
            "usage_record_id": result.get("id"),
            "quantity": result.get("quantity", quantity),
            "subscription_item_id": metered_item_id,
        }

    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """Retrieve a Stripe invoice by ID."""
        if not self.configured:
            return self._noop_response("get_invoice", invoice_id=invoice_id)

        result = self._stripe_request("GET", f"/invoices/{invoice_id}")

        return {
            "invoice_id": result["id"],
            "customer_id": result.get("customer"),
            "status": result.get("status"),
            "amount_due": result.get("amount_due"),
            "amount_paid": result.get("amount_paid"),
            "currency": result.get("currency"),
            "period_start": result.get("period_start"),
            "period_end": result.get("period_end"),
            "hosted_invoice_url": result.get("hosted_invoice_url"),
            "pdf_url": result.get("invoice_pdf"),
        }

    def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> Dict[str, Any]:
        """Cancel a Stripe subscription."""
        if not self.configured:
            return self._noop_response(
                "cancel_subscription",
                subscription_id=subscription_id,
            )

        if at_period_end:
            # Update to cancel at period end
            result = self._stripe_request(
                "POST",
                f"/subscriptions/{subscription_id}",
                {"cancel_at_period_end": "true"},
            )
        else:
            # Immediate cancellation
            result = self._stripe_request(
                "DELETE",
                f"/subscriptions/{subscription_id}",
            )

        logger.info(
            "Cancelled subscription %s (at_period_end=%s)",
            subscription_id,
            at_period_end,
        )

        return {
            "subscription_id": result["id"],
            "status": result.get("status"),
            "cancel_at_period_end": result.get("cancel_at_period_end", at_period_end),
            "canceled_at": result.get("canceled_at"),
        }

    def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """Retrieve Stripe subscription details."""
        if not self.configured:
            return self._noop_response(
                "get_subscription",
                subscription_id=subscription_id,
            )

        result = self._stripe_request("GET", f"/subscriptions/{subscription_id}")

        return {
            "subscription_id": result["id"],
            "customer_id": result.get("customer"),
            "status": result.get("status"),
            "tier": result.get("metadata", {}).get("tier", "unknown"),
            "current_period_start": result.get("current_period_start"),
            "current_period_end": result.get("current_period_end"),
            "cancel_at_period_end": result.get("cancel_at_period_end", False),
        }

    def list_invoices(
        self,
        customer_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List invoices for a Stripe customer."""
        if not self.configured:
            return []

        result = self._stripe_request(
            "GET",
            f"/invoices?customer={customer_id}&limit={limit}",
        )

        return [
            {
                "invoice_id": inv["id"],
                "status": inv.get("status"),
                "amount_due": inv.get("amount_due"),
                "amount_paid": inv.get("amount_paid"),
                "currency": inv.get("currency"),
                "created": inv.get("created"),
            }
            for inv in result.get("data", [])
        ]


# ---------------------------------------------------------------------------
# Stripe API error
# ---------------------------------------------------------------------------


class StripeApiError(Exception):
    """Raised when Stripe returns a non-2xx HTTP response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        # Try to extract Stripe error message
        message = f"Stripe API error (HTTP {status_code})"
        try:
            parsed = json.loads(body)
            err = parsed.get("error", {})
            message = f"Stripe API error: {err.get('message', body[:200])}"
        except (json.JSONDecodeError, AttributeError):
            pass
        super().__init__(message)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def get_tier_config(tier: str) -> TierConfig:
    """
    Look up tier configuration by name.

    Args:
        tier: Tier name (community, pro, enterprise).

    Returns:
        TierConfig for the requested tier.

    Raises:
        ValueError: If tier is unknown.
    """
    tier_lower = tier.lower()
    if tier_lower not in TIER_CONFIGS:
        raise ValueError(
            f"Unknown tier: {tier}. Must be one of: {list(TIER_CONFIGS.keys())}"
        )
    return TIER_CONFIGS[tier_lower]
