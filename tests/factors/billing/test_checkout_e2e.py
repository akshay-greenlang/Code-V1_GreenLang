# -*- coding: utf-8 -*-
"""
End-to-end Stripe Checkout tests (Agent W4-E / C11).

Covers the Developer Pro self-serve checkout flow from the Pricing Page
all the way through the webhook-driven entitlement grant:

    1. Client calls ``POST /v1/billing/checkout/session`` with
       ``plan_id=developer_pro``.
    2. We mock Stripe's hosted-Checkout endpoint; assert session id +
       URL are returned.
    3. Stripe POSTs ``checkout.session.completed`` back at us.
    4. ``_handle_checkout_completed`` grants the Pro tier + selected
       premium packs to the caller's tenant.
    5. ``GET /v1/billing/subscription`` reflects the new tier.

Stripe is entirely mocked at the urllib boundary; no network calls.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.factors.billing.api_routes import router as api_routes_router
from greenlang.factors.billing.skus import CATALOG, PremiumPack, Tier
from greenlang.factors.billing.stripe_provider import StripeBillingProvider
from greenlang.factors.billing.webhook_handler import (
    _handle_checkout_completed,
    _subscription_state,
    get_subscription_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockStripe:
    """Queue of canned Stripe responses."""

    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None) -> None:
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
        if self._responses:
            return self._responses.pop(0)
        return {
            "id": "cs_test_generic",
            "url": "https://checkout.stripe.com/c/pay/cs_test_generic",
        }


@pytest.fixture(autouse=True)
def _clear_state():
    _subscription_state.clear()
    yield
    _subscription_state.clear()


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_dummy_for_tests")
    app = FastAPI()
    app.include_router(api_routes_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckoutSessionEndpoint:
    def test_developer_pro_checkout_returns_session(self, client) -> None:
        """POST /checkout/session with developer_pro creates a Stripe session."""
        mock = MockStripe(
            responses=[
                {
                    "id": "cs_test_pro_e2e_1",
                    "url": "https://checkout.stripe.com/c/pay/cs_test_pro_e2e_1",
                }
            ]
        )
        with patch.object(StripeBillingProvider, "_stripe_request", side_effect=mock):
            resp = client.post(
                "/v1/billing/checkout/session",
                json={
                    "plan_id": "developer_pro",
                    "success_url": "https://app.greenlang.io/success",
                    "cancel_url": "https://app.greenlang.io/cancel",
                },
                headers={"X-Tenant-Id": "tenant_e2e_1"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == "cs_test_pro_e2e_1"
        assert body["tenant_id"] == "tenant_e2e_1"
        assert body["plan_id"] == "developer_pro"
        assert "receipt_hash" in body
        assert len(body["receipt_hash"]) == 64  # SHA-256 hex

    def test_checkout_with_premium_packs(self, client) -> None:
        """Selecting premium packs adds them as extra line items."""
        mock = MockStripe(
            responses=[
                {
                    "id": "cs_test_pro_with_packs",
                    "url": "https://checkout.stripe.com/c/pay/cs_test_pro_with_packs",
                }
            ]
        )
        with patch.object(StripeBillingProvider, "_stripe_request", side_effect=mock):
            resp = client.post(
                "/v1/billing/checkout/session",
                json={
                    "plan_id": "developer_pro",
                    "success_url": "https://x.com/s",
                    "cancel_url": "https://x.com/c",
                    "premium_packs": ["electricity_premium", "cbam_premium"],
                },
                headers={"X-Tenant-Id": "tenant_packs"},
            )
        assert resp.status_code == 200, resp.text
        # Line items sent to Stripe include both packs.
        assert len(mock.calls) == 1
        _, _, data, _ = mock.calls[0]
        assert len(data["line_items"]) == 3  # 1 tier + 2 packs

    def test_community_plan_rejected(self, client) -> None:
        """Community is auto-provisioned; checkout rejects it."""
        resp = client.post(
            "/v1/billing/checkout/session",
            json={
                "plan_id": "community",
                "success_url": "https://x.com/s",
                "cancel_url": "https://x.com/c",
            },
            headers={"X-Tenant-Id": "tenant_free"},
        )
        assert resp.status_code == 400
        assert "community" in resp.json()["detail"].lower()

    def test_contact_sales_plan_rejected(self, client) -> None:
        """Enterprise is contact-sales; no self-serve checkout."""
        resp = client.post(
            "/v1/billing/checkout/session",
            json={
                "plan_id": "enterprise",
                "success_url": "https://x.com/s",
                "cancel_url": "https://x.com/c",
            },
            headers={"X-Tenant-Id": "tenant_ent"},
        )
        assert resp.status_code == 400
        assert "contact sales" in resp.json()["detail"].lower()

    def test_unknown_plan_400(self, client) -> None:
        resp = client.post(
            "/v1/billing/checkout/session",
            json={
                "plan_id": "totally_fake_plan",
                "success_url": "https://x.com/s",
                "cancel_url": "https://x.com/c",
            },
            headers={"X-Tenant-Id": "tenant"},
        )
        assert resp.status_code == 400


class TestPortalEndpoint:
    def test_portal_requires_authentication(self, client) -> None:
        """Portal is blocked for anonymous callers."""
        resp = client.post(
            "/v1/billing/portal/session",
            json={"return_url": "https://x.com/back"},
        )
        assert resp.status_code == 401

    def test_portal_returns_url_for_authenticated(self, client) -> None:
        mock = MockStripe(
            responses=[
                {
                    "id": "bps_test_portal",
                    "url": "https://billing.stripe.com/p/session/test_portal",
                }
            ]
        )
        with patch.object(StripeBillingProvider, "_stripe_request", side_effect=mock):
            resp = client.post(
                "/v1/billing/portal/session",
                json={"return_url": "https://app.greenlang.io/billing"},
                headers={"X-Tenant-Id": "tenant_portal"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "billing.stripe.com" in body["url"]
        assert body["tenant_id"] == "tenant_portal"


class TestWebhookCheckoutCompleted:
    def test_checkout_completed_grants_pro_tier(self) -> None:
        """Webhook ``checkout.session.completed`` grants the Pro tier."""
        event_data = {
            "id": "cs_test_completed",
            "object": "checkout.session",
            "customer": "cus_tenant_e2e_1",
            "client_reference_id": "tenant_e2e_1",
            "subscription": "sub_test_pro",
            "mode": "subscription",
            "metadata": {
                "tenant_id": "tenant_e2e_1",
                "sku_name": "pro",
                "premium_packs": "",
            },
        }
        result = _handle_checkout_completed(event_data)
        assert result["status"] == "processed"
        state = get_subscription_state("cus_tenant_e2e_1")
        assert state is not None
        assert state["tier"] == Tier.PRO.value
        assert state["status"] == "active"
        assert state["subscription_id"] == "sub_test_pro"
        assert state["tenant_id"] == "tenant_e2e_1"

    def test_checkout_completed_grants_premium_packs(self) -> None:
        """Premium packs from metadata are granted on checkout completion."""
        event_data = {
            "id": "cs_test_with_packs",
            "object": "checkout.session",
            "customer": "cus_with_packs",
            "client_reference_id": "tenant_with_packs",
            "subscription": "sub_test_platform",
            "mode": "subscription",
            "metadata": {
                "tenant_id": "tenant_with_packs",
                "sku_name": "platform",
                "premium_packs": "electricity_premium,cbam_premium",
            },
        }
        result = _handle_checkout_completed(event_data)
        assert result["status"] == "processed"
        state = get_subscription_state("cus_with_packs")
        assert state["tier"] == Tier.PLATFORM.value
        assert "electricity_premium" in state["packs"]
        assert "cbam_premium" in state["packs"]

    def test_checkout_completed_is_idempotent(self) -> None:
        """Replaying the webhook produces the same state (no duplicates)."""
        event_data = {
            "id": "cs_idem",
            "object": "checkout.session",
            "customer": "cus_idem",
            "client_reference_id": "tenant_idem",
            "subscription": "sub_idem",
            "mode": "subscription",
            "metadata": {
                "tenant_id": "tenant_idem",
                "sku_name": "pro",
                "premium_packs": "freight_premium",
            },
        }
        _handle_checkout_completed(event_data)
        first = dict(get_subscription_state("cus_idem"))
        _handle_checkout_completed(event_data)
        second = dict(get_subscription_state("cus_idem"))
        # updated_at and checkout_completed_at may tick; everything else stable.
        first.pop("updated_at", None)
        first.pop("checkout_completed_at", None)
        second.pop("updated_at", None)
        second.pop("checkout_completed_at", None)
        assert first == second

    def test_checkout_completed_ignores_unknown_pack(self) -> None:
        """Invalid premium pack in metadata is logged and skipped, not crashed."""
        event_data = {
            "id": "cs_unknown_pack",
            "object": "checkout.session",
            "customer": "cus_unknown",
            "client_reference_id": "tenant_unknown",
            "subscription": "sub_unknown",
            "mode": "subscription",
            "metadata": {
                "tenant_id": "tenant_unknown",
                "sku_name": "pro",
                "premium_packs": "electricity_premium,totally_fake_pack",
            },
        }
        result = _handle_checkout_completed(event_data)
        assert result["status"] == "processed"
        state = get_subscription_state("cus_unknown")
        assert "electricity_premium" in state["packs"]
        assert "totally_fake_pack" not in state["packs"]


class TestEndToEndFlow:
    def test_pricing_page_to_webhook_to_subscription_view(self, client) -> None:
        """Full self-serve flow: checkout -> webhook -> GET /subscription."""
        # 1. Client kicks off checkout.
        mock = MockStripe(
            responses=[
                {
                    "id": "cs_test_e2e_full",
                    "url": "https://checkout.stripe.com/c/pay/cs_test_e2e_full",
                }
            ]
        )
        with patch.object(StripeBillingProvider, "_stripe_request", side_effect=mock):
            resp = client.post(
                "/v1/billing/checkout/session",
                json={
                    "plan_id": "developer_pro",
                    "success_url": "https://app.greenlang.io/success",
                    "cancel_url": "https://app.greenlang.io/cancel",
                },
                headers={"X-Tenant-Id": "tenant_full_flow"},
            )
        assert resp.status_code == 200

        # 2. Stripe fires checkout.session.completed; we process it.
        _handle_checkout_completed(
            {
                "id": "cs_test_e2e_full",
                "object": "checkout.session",
                "customer": "tenant_full_flow",   # route uses tenant_id as cache key
                "client_reference_id": "tenant_full_flow",
                "subscription": "sub_test_e2e_full",
                "mode": "subscription",
                "metadata": {
                    "tenant_id": "tenant_full_flow",
                    "sku_name": "pro",
                    "premium_packs": "",
                },
            }
        )

        # 3. GET /subscription reflects the new tier for the tenant.
        resp = client.get(
            "/v1/billing/subscription",
            headers={"X-Tenant-Id": "tenant_full_flow"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tenant_id"] == "tenant_full_flow"
        assert body["tier"] == Tier.PRO.value
        assert body["plan_id"] == "developer_pro"
        assert body["status"] == "active"
        assert body["usage"]["requests_included"] > 0
