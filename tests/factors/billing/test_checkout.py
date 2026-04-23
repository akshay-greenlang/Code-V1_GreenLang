# -*- coding: utf-8 -*-
"""
Tests for the FY27 Pricing Page Stripe Checkout creation endpoint.

Covers:
    * StripeBillingProvider.create_checkout_session — basic + premium pack +
      invalid SKU + idempotency.
    * POST /api/v1/billing/checkout — auth, happy path, validation errors.
    * Webhook checkout.session.completed — grants the right tier + packs
      from session metadata.

Stripe is fully mocked at the urllib boundary; no real API calls are made.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.factors.billing import skus as skus_module
from greenlang.factors.billing.skus import CATALOG, PremiumPack, Tier
from greenlang.factors.billing.stripe_provider import (
    StripeApiError,
    StripeBillingProvider,
)
from greenlang.factors.billing.webhook_handler import (
    _handle_checkout_completed,
    _subscription_state,
    get_subscription_state,
    router as webhook_router,
)
from greenlang.utilities.exceptions.integration import BillingProviderError


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockStripeCall:
    """Records each call to ``StripeBillingProvider._stripe_request``.

    Use as ``side_effect`` of a patch on ``_stripe_request``. Returns the
    queued response for each call; raises ``StopIteration`` if exhausted.
    """

    def __init__(
        self, responses: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        self.calls: List[Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]] = []
        self._responses = list(responses or [])

    def __call__(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.calls.append((method, path, data, idempotency_key))
        if not self._responses:
            return {
                "id": "cs_test_default_session_id",
                "url": "https://checkout.stripe.com/c/pay/cs_test_default",
            }
        return self._responses.pop(0)


@pytest.fixture()
def configured_provider(monkeypatch) -> StripeBillingProvider:
    """A StripeBillingProvider that thinks it's configured (no real key)."""
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_dummy_for_tests")
    return StripeBillingProvider.from_environment()


@pytest.fixture(autouse=True)
def _clear_subscription_state():
    """Clear in-memory subscription state between tests."""
    _subscription_state.clear()
    yield
    _subscription_state.clear()


# ---------------------------------------------------------------------------
# Provider unit tests
# ---------------------------------------------------------------------------


class TestCreateCheckoutSession:
    def test_returns_session_id_and_url(self, configured_provider):
        """Happy path: returns the session_id and url Stripe gave us."""
        mock = MockStripeCall(
            responses=[
                {
                    "id": "cs_test_basic_abc123",
                    "url": "https://checkout.stripe.com/c/pay/cs_test_basic_abc123",
                }
            ]
        )
        with patch.object(configured_provider, "_stripe_request", side_effect=mock):
            result = configured_provider.create_checkout_session(
                sku_name="pro",
                tenant_id="tenant_acme",
                success_url="https://example.com/success",
                cancel_url="https://example.com/cancel",
            )
        assert result["session_id"] == "cs_test_basic_abc123"
        assert "checkout.stripe.com" in result["url"]
        assert len(mock.calls) == 1
        method, path, data, _idem = mock.calls[0]
        assert method == "POST"
        assert path == "/checkout/sessions"
        # Tier price ID present in line_items[0]
        pro_price = CATALOG.tier(Tier.PRO).stripe_price_monthly_id
        assert data["line_items"][0]["price"] == pro_price
        # client_reference_id and metadata.tenant_id both stamped
        assert data["client_reference_id"] == "tenant_acme"
        assert data["metadata"]["tenant_id"] == "tenant_acme"
        assert data["metadata"]["sku_name"] == "pro"
        assert data["mode"] == "subscription"
        assert data["allow_promotion_codes"] == "true"

    def test_with_premium_pack_adds_line_item(self, configured_provider):
        """Each premium pack becomes an extra line item."""
        mock = MockStripeCall(
            responses=[
                {
                    "id": "cs_test_packs",
                    "url": "https://checkout.stripe.com/c/pay/cs_test_packs",
                }
            ]
        )
        with patch.object(configured_provider, "_stripe_request", side_effect=mock):
            result = configured_provider.create_checkout_session(
                sku_name="pro",
                tenant_id="tenant_xyz",
                success_url="https://example.com/success",
                cancel_url="https://example.com/cancel",
                premium_packs=["electricity_premium", "freight_premium"],
            )
        assert result["session_id"] == "cs_test_packs"
        _method, _path, data, _idem = mock.calls[0]
        # 1 tier + 2 packs = 3 line items
        assert len(data["line_items"]) == 3
        prices = [li["price"] for li in data["line_items"]]
        assert CATALOG.tier(Tier.PRO).stripe_price_monthly_id in prices
        assert (
            CATALOG.pack(PremiumPack.ELECTRICITY).stripe_price_monthly_id in prices
        )
        assert CATALOG.pack(PremiumPack.FREIGHT).stripe_price_monthly_id in prices
        # Metadata includes the comma-joined pack list
        assert (
            data["metadata"]["premium_packs"]
            == "electricity_premium,freight_premium"
        )

    def test_invalid_sku_raises_value_error(self, configured_provider):
        """Unknown sku_name raises ValueError before hitting Stripe."""
        mock = MockStripeCall()
        with patch.object(configured_provider, "_stripe_request", side_effect=mock):
            with pytest.raises(ValueError, match="Unknown SKU"):
                configured_provider.create_checkout_session(
                    sku_name="ultra_mega_premium",
                    tenant_id="t1",
                    success_url="https://example.com/s",
                    cancel_url="https://example.com/c",
                )
        assert mock.calls == []  # No Stripe call made

    def test_community_sku_rejected(self, configured_provider):
        """Community is free — must reject before talking to Stripe."""
        mock = MockStripeCall()
        with patch.object(configured_provider, "_stripe_request", side_effect=mock):
            with pytest.raises(ValueError, match="Community"):
                configured_provider.create_checkout_session(
                    sku_name="community",
                    tenant_id="t1",
                    success_url="https://example.com/s",
                    cancel_url="https://example.com/c",
                )
        assert mock.calls == []

    def test_invalid_premium_pack_raises_value_error(self, configured_provider):
        mock = MockStripeCall()
        with patch.object(configured_provider, "_stripe_request", side_effect=mock):
            with pytest.raises(ValueError, match="Unknown premium pack"):
                configured_provider.create_checkout_session(
                    sku_name="pro",
                    tenant_id="t1",
                    success_url="https://example.com/s",
                    cancel_url="https://example.com/c",
                    premium_packs=["nonexistent_pack"],
                )
        assert mock.calls == []

    def test_idempotent_same_args_same_key(self, configured_provider):
        """Two calls with the same args produce the same Idempotency-Key."""
        mock = MockStripeCall(
            responses=[
                {
                    "id": "cs_test_idem_1",
                    "url": "https://checkout.stripe.com/c/pay/cs_test_idem_1",
                },
                # Stripe would return the SAME session for the same key in
                # production -- mock can return identical bytes here.
                {
                    "id": "cs_test_idem_1",
                    "url": "https://checkout.stripe.com/c/pay/cs_test_idem_1",
                },
            ]
        )
        with patch.object(configured_provider, "_stripe_request", side_effect=mock):
            r1 = configured_provider.create_checkout_session(
                sku_name="pro",
                tenant_id="tenant_idem",
                success_url="https://example.com/success",
                cancel_url="https://example.com/cancel",
                premium_packs=["electricity_premium"],
            )
            r2 = configured_provider.create_checkout_session(
                sku_name="pro",
                tenant_id="tenant_idem",
                success_url="https://example.com/success",
                cancel_url="https://example.com/cancel",
                premium_packs=["electricity_premium"],
            )
        assert r1["session_id"] == r2["session_id"]
        # Both calls passed the same Idempotency-Key
        key1 = mock.calls[0][3]
        key2 = mock.calls[1][3]
        assert key1 is not None
        assert key1 == key2

    def test_idempotency_key_pure_helper(self):
        """Same logical inputs -> same key; diff inputs -> diff keys."""
        k1 = StripeBillingProvider._create_checkout_session_idempotency_key(
            "tenant_a", "pro", ["electricity_premium", "freight_premium"]
        )
        k2 = StripeBillingProvider._create_checkout_session_idempotency_key(
            "tenant_a", "pro", ["freight_premium", "electricity_premium"]
        )  # order shouldn't matter
        k3 = StripeBillingProvider._create_checkout_session_idempotency_key(
            "tenant_b", "pro", ["electricity_premium", "freight_premium"]
        )
        assert k1 == k2  # pack order is normalized
        assert k1 != k3  # different tenant => different key
        assert len(k1) == 64  # SHA-256 hex

    def test_stripe_5xx_raises_billing_provider_error(self, configured_provider):
        """Stripe 5xx surface as BillingProviderError with retry hint."""
        def _raise(method, path, data=None, idempotency_key=None):
            raise StripeApiError(
                503,
                json.dumps({"error": {"message": "Stripe down"}}),
            )

        with patch.object(configured_provider, "_stripe_request", side_effect=_raise):
            with pytest.raises(BillingProviderError) as exc_info:
                configured_provider.create_checkout_session(
                    sku_name="pro",
                    tenant_id="tenant_err",
                    success_url="https://example.com/s",
                    cancel_url="https://example.com/c",
                )
        err = exc_info.value
        assert err.provider == "stripe"
        assert err.operation == "create_checkout_session"
        assert err.status_code == 503
        assert err.retry_after_seconds == 30

    def test_unconfigured_provider_returns_noop(self, monkeypatch):
        """When STRIPE_API_KEY is unset, return a deterministic noop session."""
        monkeypatch.delenv("STRIPE_API_KEY", raising=False)
        provider = StripeBillingProvider.from_environment()
        assert provider.configured is False
        result = provider.create_checkout_session(
            sku_name="pro",
            tenant_id="tenant_noop",
            success_url="https://example.com/success",
            cancel_url="https://example.com/cancel",
        )
        assert result["session_id"].startswith("cs_test_noop_")
        assert "noop_session=" in result["url"]


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def billing_app(monkeypatch) -> FastAPI:
    """Minimal FastAPI app mounting just the billing router (no auth coupling)."""
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_dummy_for_tests")
    # Reset cached provider so our env var takes effect.
    from greenlang.integration.api.routes import billing as billing_route_mod

    billing_route_mod.reset_stripe_provider_for_tests()
    app = FastAPI()
    app.include_router(billing_route_mod.router)
    return app


@pytest.fixture()
def auth_override(billing_app) -> Dict[str, Any]:
    """Override the auth dependency with a stub user; return it for mutation."""
    from greenlang.integration.api.dependencies import get_current_user

    fake_user: Dict[str, Any] = {
        "user_id": "u_test_001",
        "tenant_id": "tenant_test",
        "email": "buyer@example.com",
        "tier": "community",
    }

    def _override():
        return fake_user

    billing_app.dependency_overrides[get_current_user] = _override
    yield fake_user
    billing_app.dependency_overrides.pop(get_current_user, None)


def _patched_stripe_request(
    response_id: str = "cs_test_endpoint_001",
):
    """Return a function that always responds with the given session id."""
    def _fn(method, path, data=None, idempotency_key=None):
        return {
            "id": response_id,
            "url": f"https://checkout.stripe.com/c/pay/{response_id}",
        }
    return _fn


class TestCheckoutEndpoint:
    def test_unauthenticated_returns_401(self, billing_app):
        """No Authorization header -> 401."""
        client = TestClient(billing_app)
        resp = client.post(
            "/api/v1/billing/checkout",
            json={
                "sku_name": "pro",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
        )
        # FastAPI's HTTPBearer returns 403 if no header is present unless
        # auto_error is True (default). Either 401 or 403 is acceptable
        # signal of "no auth".
        assert resp.status_code in (401, 403)

    def test_happy_path(self, billing_app, auth_override):
        """Authenticated POST returns session_id + url."""
        from greenlang.integration.api.routes import billing as billing_route_mod

        provider = billing_route_mod.get_stripe_provider()
        with patch.object(
            provider,
            "_stripe_request",
            side_effect=_patched_stripe_request("cs_test_happy_001"),
        ):
            client = TestClient(billing_app)
            resp = client.post(
                "/api/v1/billing/checkout",
                json={
                    "sku_name": "pro",
                    "success_url": "https://example.com/success",
                    "cancel_url": "https://example.com/cancel",
                    "premium_packs": ["electricity_premium"],
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == "cs_test_happy_001"
        assert body["url"].startswith("https://checkout.stripe.com/")

    def test_invalid_sku_returns_400(self, billing_app, auth_override):
        """Unknown SKU is rejected with a 400 + actionable error."""
        client = TestClient(billing_app)
        resp = client.post(
            "/api/v1/billing/checkout",
            json={
                "sku_name": "ultra_pro_max",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
        )
        assert resp.status_code == 400
        assert "Unknown sku_name" in resp.json()["detail"]

    def test_community_sku_returns_400(self, billing_app, auth_override):
        """Community is free -> reject without bothering Stripe."""
        client = TestClient(billing_app)
        resp = client.post(
            "/api/v1/billing/checkout",
            json={
                "sku_name": "community",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
            },
        )
        assert resp.status_code == 400
        assert "Community" in resp.json()["detail"]

    def test_invalid_premium_pack_returns_400(self, billing_app, auth_override):
        client = TestClient(billing_app)
        resp = client.post(
            "/api/v1/billing/checkout",
            json={
                "sku_name": "pro",
                "success_url": "https://example.com/success",
                "cancel_url": "https://example.com/cancel",
                "premium_packs": ["fictional_pack"],
            },
        )
        assert resp.status_code == 400
        assert "Unknown premium pack" in resp.json()["detail"]

    def test_billing_provider_error_returns_502(self, billing_app, auth_override):
        """Upstream Stripe failure -> 502 with Retry-After header."""
        from greenlang.integration.api.routes import billing as billing_route_mod

        provider = billing_route_mod.get_stripe_provider()

        def _fail(method, path, data=None, idempotency_key=None):
            raise StripeApiError(
                503,
                json.dumps({"error": {"message": "Stripe is down"}}),
            )

        with patch.object(provider, "_stripe_request", side_effect=_fail):
            client = TestClient(billing_app)
            resp = client.post(
                "/api/v1/billing/checkout",
                json={
                    "sku_name": "pro",
                    "success_url": "https://example.com/success",
                    "cancel_url": "https://example.com/cancel",
                },
            )
        assert resp.status_code == 502
        assert resp.headers.get("retry-after") == "30"


# ---------------------------------------------------------------------------
# Webhook integration: checkout.session.completed grants tier + packs
# ---------------------------------------------------------------------------


class TestWebhookGrantsAfterCheckout:
    def test_checkout_completed_grants_tier_and_packs(self):
        """The webhook must read metadata.sku_name + .premium_packs and
        populate the tenant's entitlements accordingly."""
        session_event = {
            "id": "cs_test_webhook_001",
            "customer": "cus_webhook_001",
            "client_reference_id": "tenant_webhook",
            "subscription": "sub_webhook_001",
            "mode": "subscription",
            "metadata": {
                "tenant_id": "tenant_webhook",
                "sku_name": "platform",
                "premium_packs": "electricity_premium,freight_premium",
            },
        }
        result = _handle_checkout_completed(session_event)
        assert result["status"] == "processed"

        state = get_subscription_state("cus_webhook_001")
        assert state is not None
        assert state["tenant_id"] == "tenant_webhook"
        assert state["subscription_id"] == "sub_webhook_001"
        assert state["tier"] == Tier.PLATFORM.value
        assert "electricity_premium" in state["packs"]
        assert "freight_premium" in state["packs"]
        # entitlements snapshot was filled in
        assert state["entitlements"]["tier"] == Tier.PLATFORM.value
        assert state["status"] == "active"

    def test_checkout_completed_idempotent_on_retry(self):
        """Re-firing the same webhook produces the same final state."""
        session_event = {
            "id": "cs_test_idem_webhook",
            "customer": "cus_idem_001",
            "client_reference_id": "tenant_idem",
            "subscription": "sub_idem_001",
            "mode": "subscription",
            "metadata": {
                "tenant_id": "tenant_idem",
                "sku_name": "pro",
                "premium_packs": "electricity_premium",
            },
        }
        _handle_checkout_completed(session_event)
        state1 = dict(get_subscription_state("cus_idem_001") or {})
        _handle_checkout_completed(session_event)
        state2 = dict(get_subscription_state("cus_idem_001") or {})

        # Tier + packs must be identical and packs must NOT have duplicated.
        assert state1["tier"] == state2["tier"] == Tier.PRO.value
        assert state1["packs"] == state2["packs"] == ["electricity_premium"]
        assert state1["subscription_id"] == state2["subscription_id"]

    def test_checkout_completed_unknown_sku_falls_back_to_community(self):
        """A garbage sku_name in metadata must not crash the webhook."""
        session_event = {
            "id": "cs_test_bad_sku",
            "customer": "cus_bad_sku",
            "client_reference_id": "tenant_bad",
            "subscription": "sub_bad_001",
            "mode": "subscription",
            "metadata": {
                "tenant_id": "tenant_bad",
                "sku_name": "not_a_real_tier",
                "premium_packs": "",
            },
        }
        result = _handle_checkout_completed(session_event)
        assert result["status"] == "processed"
        state = get_subscription_state("cus_bad_sku")
        assert state is not None
        assert state["tier"] == Tier.COMMUNITY.value

    def test_checkout_completed_filters_unknown_packs(self):
        """Unknown packs in metadata are dropped silently."""
        session_event = {
            "id": "cs_test_filter_packs",
            "customer": "cus_filter_001",
            "client_reference_id": "tenant_filter",
            "subscription": "sub_filter_001",
            "mode": "subscription",
            "metadata": {
                "tenant_id": "tenant_filter",
                "sku_name": "pro",
                "premium_packs": "electricity_premium,not_a_pack,freight_premium",
            },
        }
        _handle_checkout_completed(session_event)
        state = get_subscription_state("cus_filter_001")
        assert state is not None
        assert state["packs"] == ["electricity_premium", "freight_premium"]
