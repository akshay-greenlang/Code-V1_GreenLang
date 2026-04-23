# -*- coding: utf-8 -*-
"""
N7 — Open-core boundary: Community tier never sees Premium / Private / OEM.

A Community-tier caller must NEVER receive any factor whose source
belongs to a Premium Pack SKU, a Customer-Private factor, or an
OEM-Redistributable factor.

Parametrized over the 8 premium SKUs declared in
:class:`greenlang.factors.entitlements.PackSKU`:

    ELECTRICITY_PREMIUM
    FREIGHT_PREMIUM
    PRODUCT_CARBON_PREMIUM
    EPD_PREMIUM
    AGRIFOOD_PREMIUM
    FINANCE_PREMIUM
    CBAM_PREMIUM
    LAND_PREMIUM

Plus a direct check that CUSTOMER_PRIVATE and OEM_REDISTRIBUTABLE
classes are rejected at the Community tier.

Run standalone::

    pytest tests/factors/gates/test_n7_open_core_boundary.py -v
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from greenlang.factors.entitlements import PackSKU
from greenlang.factors.middleware.licensing_guard import LicensingGuardMiddleware
from greenlang.factors.tier_enforcement import (
    Tier,
    TierVisibility,
    filter_factors_by_tier,
)


# ---------------------------------------------------------------------------
# Helpers (mirror test_n4)
# ---------------------------------------------------------------------------


def _make_request(user: Dict[str, Any]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/factors/resolve",
        "headers": [],
        "state": {"user": user},
    }

    async def _receive() -> Dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    req = Request(scope, _receive)
    req.state.user = user
    return req


async def _invoke_guard(
    user: Dict[str, Any], payload: Dict[str, Any]
) -> Response:
    mw = LicensingGuardMiddleware(app=None)

    async def _call_next(_request: Request) -> Response:
        return JSONResponse(content=payload, status_code=200)

    return await mw.dispatch(_make_request(user), _call_next)


def _invoke_sync(user, payload) -> Response:
    return asyncio.new_event_loop().run_until_complete(
        _invoke_guard(user, payload)
    )


# ---------------------------------------------------------------------------
# Premium pack factor metadata. Community must be blocked from every one.
# ---------------------------------------------------------------------------


def _premium_factor(pack_sku: str) -> Dict[str, Any]:
    """A minimal factor payload that carries premium-pack metadata.

    Premium packs are commercial-license data, so they ship with
    ``redistribution_class='licensed'`` and a ``pack_sku`` tag.
    """
    return {
        "factor_id": f"F:{pack_sku}:1",
        "redistribution_class": "licensed",
        "pack_sku": pack_sku,
    }


# ---------------------------------------------------------------------------
# Gate: community caller → 402 for every premium pack.
# ---------------------------------------------------------------------------


class TestN7CommunityBlockedFromPremiumPacks:
    """One parametrized case per premium SKU."""

    @pytest.mark.parametrize("pack_sku", PackSKU.ALL)
    def test_community_blocked_from_premium_pack(self, pack_sku, caller_contexts):
        user = caller_contexts["community_anon"].as_request_state_user()
        resp = _invoke_sync(user, _premium_factor(pack_sku))
        assert resp.status_code == 402, (
            f"N7 violation: community-tier caller was allowed to read a "
            f"factor from premium pack {pack_sku!r}. Expected 402. "
            f"Got status={resp.status_code}. Open-core boundary is broken."
        )

    def test_community_with_premium_sku_granted_ok(self, caller_contexts):
        """Sanity: a community caller with the premium grant succeeds."""
        user = caller_contexts["community_plus_freight"].as_request_state_user()
        # Caller bought freight_premium; an open factor should still pass (200).
        open_payload = {
            "factor_id": "F:open:1",
            "redistribution_class": "open",
        }
        resp = _invoke_sync(user, open_payload)
        assert resp.status_code == 200

    def test_all_eight_packs_enumerated(self):
        """PackSKU.ALL must list all 8 premium packs — N7 covers every one."""
        assert len(PackSKU.ALL) == 8, (
            "N7 invariant: PackSKU.ALL must have exactly 8 entries (the 8 "
            f"premium packs). Got {len(PackSKU.ALL)}: {PackSKU.ALL}"
        )


# ---------------------------------------------------------------------------
# Gate: community caller can never access Customer-Private or OEM.
# ---------------------------------------------------------------------------


class TestN7CommunityBlockedFromPrivateAndOEM:
    def test_community_blocked_from_customer_private(self, caller_contexts):
        user = caller_contexts["community_anon"].as_request_state_user()
        payload = {
            "factor_id": "F:private:tenantA:1",
            "redistribution_class": "customer_private",
            "tenant_id": "tenant-a",
        }
        resp = _invoke_sync(user, payload)
        assert resp.status_code in (402, 403), (
            "N7 violation: community-tier caller was allowed to read a "
            f"customer_private factor. Got status={resp.status_code}."
        )

    def test_community_blocked_from_oem_redistributable(self, caller_contexts):
        user = caller_contexts["community_anon"].as_request_state_user()
        payload = {
            "factor_id": "F:oem:1",
            "redistribution_class": "oem_redistributable",
        }
        resp = _invoke_sync(user, payload)
        assert resp.status_code in (402, 403), (
            "N7 violation: community-tier caller was allowed to read an "
            f"oem_redistributable factor. Got status={resp.status_code}."
        )


# ---------------------------------------------------------------------------
# Gate: tier_enforcement must hide preview/connector_only rows from community.
# ---------------------------------------------------------------------------


class TestN7TierVisibilityForCommunity:
    """Community tier cannot see preview / connector_only / deprecated rows."""

    def test_community_tier_visibility_is_certified_only(self):
        vis = TierVisibility.from_tier(Tier.COMMUNITY.value)
        assert vis.include_preview is False, (
            "N7 violation: Community tier must not see preview-status factors."
        )
        assert vis.include_connector is False, (
            "N7 violation: Community tier must not see connector_only factors."
        )
        assert vis.include_deprecated is False, (
            "N7 violation: Community tier must not see deprecated factors."
        )
        assert vis.audit_bundle_allowed is False, (
            "N7 violation: Community tier must not receive audit bundles."
        )

    def test_filter_factors_by_tier_drops_preview_and_connector(
        self, make_record, make_vectors
    ):
        certified = make_record(
            factor_id="F:cert:1",
            family="emissions",
            vectors=make_vectors(CO2=1.0, CH4=0.001, N2O=0.0001),
            co2e_total=1.03,
            factor_status="certified",
        )
        preview = make_record(
            factor_id="F:preview:1",
            family="emissions",
            vectors=make_vectors(CO2=1.0, CH4=0.001, N2O=0.0001),
            co2e_total=1.03,
            factor_status="preview",
        )
        connector = make_record(
            factor_id="F:connector:1",
            family="emissions",
            vectors=make_vectors(CO2=1.0, CH4=0.001, N2O=0.0001),
            co2e_total=1.03,
            factor_status="connector_only",
        )
        vis = TierVisibility.from_tier(Tier.COMMUNITY.value)
        kept = filter_factors_by_tier([certified, preview, connector], vis)
        kept_ids = {getattr(f, "factor_id", None) for f in kept}
        assert kept_ids == {"F:cert:1"}, (
            "N7 violation: filter_factors_by_tier for Community returned "
            f"{kept_ids}. It must return ONLY the certified factor."
        )
