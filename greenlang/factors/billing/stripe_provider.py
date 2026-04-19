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
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the Stripe API.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: API path (e.g., "/customers").
            data: Form-encoded body data for POST requests.

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
