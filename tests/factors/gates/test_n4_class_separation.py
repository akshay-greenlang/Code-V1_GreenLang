# -*- coding: utf-8 -*-
"""
N4 — Never mix licensing classes (open / licensed / customer-private / OEM).

:class:`greenlang.factors.middleware.licensing_guard.LicensingGuardMiddleware`
must return:

    * 402 Payment Required — caller lacks the entitlement / plan.
    * 403 Forbidden        — customer-private factor for a different tenant.

Four parameterized cases (at minimum):

    Open           → any caller OK
    Licensed       → only Pro+/entitled caller OK
    Customer-Private → only same-tenant caller OK (cross-tenant = 403)
    OEM-Redistributable → only OEM-entitled caller OK

Run standalone::

    pytest tests/factors/gates/test_n4_class_separation.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from greenlang.factors.middleware.licensing_guard import (
    LicensingGuardMiddleware,
    _caller_grants,
    _walk_for_violation,
    _CLASS_RANK,
    _max_class,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(user: Dict[str, Any]) -> Request:
    """Build a minimal Starlette Request with ``request.state.user`` set."""
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


def _factor_payload(
    *,
    factor_id: str,
    redistribution_class: str,
    owner_tenant_id: str = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "factor_id": factor_id,
        "redistribution_class": redistribution_class,
    }
    if owner_tenant_id is not None:
        body["tenant_id"] = owner_tenant_id
    return body


async def _invoke_guard(
    user: Dict[str, Any], payload: Dict[str, Any]
) -> Response:
    """Wire the middleware around a simple handler returning ``payload`` as JSON."""
    mw = LicensingGuardMiddleware(app=None)

    async def _call_next(request: Request) -> Response:
        return JSONResponse(content=payload, status_code=200)

    request = _make_request(user)
    return await mw.dispatch(request, _call_next)


def _invoke_guard_sync(user, payload) -> Response:
    return asyncio.get_event_loop().run_until_complete(
        _invoke_guard(user, payload)
    ) if False else asyncio.new_event_loop().run_until_complete(
        _invoke_guard(user, payload)
    )


# ---------------------------------------------------------------------------
# Gate: 4 canonical license-class cases.
# ---------------------------------------------------------------------------


class TestN4LicenseClassSeparation:
    """402 / 403 enforcement across the 4 redistribution classes."""

    # -- Open data: any caller OK (200) -----------------------------------

    def test_open_class_any_caller_ok(self, caller_contexts):
        for ctx_name, ctx in caller_contexts.items():
            user = ctx.as_request_state_user()
            resp = _invoke_guard_sync(
                user,
                _factor_payload(factor_id="F:open:1", redistribution_class="open"),
            )
            assert resp.status_code == 200, (
                f"N4 violation: Open-class factor rejected for caller "
                f"tier={ctx.tier!r} ctx={ctx_name!r}. Open data must be "
                f"accessible to every tier. Got status={resp.status_code}."
            )

    # -- Licensed: community denied (402), pro+entitled OK ---------------

    def test_licensed_class_community_denied_402(self, caller_contexts):
        user = caller_contexts["community_anon"].as_request_state_user()
        resp = _invoke_guard_sync(
            user,
            _factor_payload(factor_id="F:licensed:1", redistribution_class="licensed"),
        )
        assert resp.status_code == 402, (
            "N4 violation: community-tier caller was allowed to read a "
            "'licensed' factor. Expected 402 Payment Required. Got "
            f"status={resp.status_code}."
        )
        body = json.loads(bytes(resp.body))
        assert body.get("error") == "licensing_gap", (
            f"N4 violation: 402 body should identify error=licensing_gap. Got {body}"
        )
        assert body.get("required_class") == "licensed", (
            "N4 violation: 402 body must name the blocked class as "
            f"'licensed'. Got {body}"
        )

    def test_licensed_class_enterprise_entitled_ok(self, caller_contexts):
        user = caller_contexts["enterprise"].as_request_state_user()
        resp = _invoke_guard_sync(
            user,
            _factor_payload(factor_id="F:licensed:1", redistribution_class="licensed"),
        )
        assert resp.status_code == 200, (
            "N4 violation: enterprise caller with 'licensed' entitlement was "
            f"denied a 'licensed' factor. Got status={resp.status_code}."
        )

    # -- Customer-Private: cross-tenant denied (403) ---------------------

    def test_customer_private_cross_tenant_denied_403(self, caller_contexts):
        user = caller_contexts["enterprise_tenant_b"].as_request_state_user()
        resp = _invoke_guard_sync(
            user,
            _factor_payload(
                factor_id="F:private:A:1",
                redistribution_class="customer_private",
                owner_tenant_id="tenant-a",     # factor belongs to tenant-a
            ),
        )
        assert resp.status_code == 403, (
            "N4 violation: tenant-b was allowed to read a customer_private "
            "factor owned by tenant-a. Cross-tenant reads must 403. "
            f"Got status={resp.status_code}."
        )
        body = json.loads(bytes(resp.body))
        assert body.get("error") == "tenant_forbidden", (
            f"N4 violation: 403 body should say error=tenant_forbidden. Got {body}"
        )

    def test_customer_private_same_tenant_ok(self, caller_contexts):
        user = caller_contexts["enterprise_tenant_a"].as_request_state_user()
        resp = _invoke_guard_sync(
            user,
            _factor_payload(
                factor_id="F:private:A:1",
                redistribution_class="customer_private",
                owner_tenant_id="tenant-a",
            ),
        )
        assert resp.status_code == 200, (
            "N4 violation: tenant-a (owner) was denied its own customer_private "
            f"factor. Got status={resp.status_code}."
        )

    # -- OEM-Redistributable: only OEM-entitled caller ------------------

    def test_oem_class_non_oem_caller_denied(self, caller_contexts):
        # A 'licensed' caller WITHOUT oem_id should NOT see an OEM-only factor.
        user = caller_contexts["enterprise"].as_request_state_user()
        # Make sure oem_id is absent.
        user.pop("oem_id", None)
        resp = _invoke_guard_sync(
            user,
            _factor_payload(
                factor_id="F:oem:1",
                redistribution_class="oem_redistributable",
            ),
        )
        assert resp.status_code in (402, 403), (
            "N4 violation: non-OEM enterprise caller was allowed to read an "
            "oem_redistributable factor. This must 402/403."
        )

    def test_oem_class_oem_caller_ok(self, caller_contexts):
        """When the caller is a registered OEM partner, the read is allowed."""
        user = caller_contexts["oem_partner"].as_request_state_user()
        resp = _invoke_guard_sync(
            user,
            _factor_payload(
                factor_id="F:oem:1",
                redistribution_class="oem_redistributable",
            ),
        )
        assert resp.status_code == 200, (
            "N4 violation: OEM partner (with oem_id + licensed entitlement) "
            f"was denied an oem_redistributable factor. Got status={resp.status_code}."
        )

    # -- OEM auto-grant: three explicit cases covering Fix 3 -----------

    def test_oem_autogrant_matching_oem_caller_granted(self, caller_contexts):
        """(a) OEM caller whose oem_id matches the factor's allowed_oem_ids
        receives the oem_redistributable grant and reads successfully."""
        user = {
            "tier": "enterprise",
            "entitlements": ["licensed"],
            "oem_id": "oem-partner-1",
        }
        # Factor explicitly allows oem-partner-1
        payload = {
            "factor_id": "F:oem:1",
            "redistribution_class": "oem_redistributable",
            "allowed_oem_ids": ["oem-partner-1"],
            "oem_redistributable_allowed": True,
        }
        resp = _invoke_guard_sync(user, payload)
        assert resp.status_code == 200, (
            "OEM auto-grant: matching oem_id + allowed flag must succeed. "
            f"Got {resp.status_code}."
        )

    def test_oem_autogrant_non_oem_caller_denied(self, caller_contexts):
        """(b) Enterprise caller WITHOUT oem_id is denied an OEM factor
        even though they have the 'licensed' grant."""
        user = {
            "tier": "enterprise",
            "entitlements": ["licensed"],
            # NOTE: no oem_id
        }
        payload = {
            "factor_id": "F:oem:1",
            "redistribution_class": "oem_redistributable",
            "oem_redistributable_allowed": True,
        }
        resp = _invoke_guard_sync(user, payload)
        assert resp.status_code in (402, 403), (
            "OEM auto-grant: caller without oem_id must be denied. "
            f"Got {resp.status_code}."
        )

    def test_oem_autogrant_oem_caller_without_matching_flag_denied(
        self, caller_contexts
    ):
        """(c) OEM caller against a factor that does NOT carry the
        oem_redistributable_allowed flag (nor a matching oem_id /
        allowed_oem_ids entry) is denied."""
        user = {
            "tier": "enterprise",
            "entitlements": ["licensed"],
            "oem_id": "oem-partner-99",
        }
        # Factor has NO oem_redistributable_allowed flag, NO oem_id match.
        payload = {
            "factor_id": "F:oem:1",
            "redistribution_class": "oem_redistributable",
        }
        resp = _invoke_guard_sync(user, payload)
        assert resp.status_code in (402, 403), (
            "OEM auto-grant: OEM caller without matching factor flag must "
            f"be denied. Got {resp.status_code}."
        )


# ---------------------------------------------------------------------------
# Sanity: the rank table reflects the declared ordering.
# ---------------------------------------------------------------------------


class TestN4RankInvariants:
    def test_customer_private_outranks_licensed(self):
        assert _CLASS_RANK["customer_private"] > _CLASS_RANK["licensed"], (
            "N4 invariant: customer_private MUST outrank licensed in "
            "_CLASS_RANK or cross-tenant reads won't 403."
        )
        assert _CLASS_RANK["licensed"] > _CLASS_RANK["open"], (
            "N4 invariant: licensed MUST outrank open in _CLASS_RANK."
        )

    def test_oem_redistributable_has_dedicated_rank(self):
        """N4 invariant: oem_redistributable must be in _CLASS_RANK with
        rank >= licensed (otherwise a community caller reads it as open)."""
        assert "oem_redistributable" in _CLASS_RANK, (
            "N4 invariant: oem_redistributable MUST appear in _CLASS_RANK."
        )
        assert _CLASS_RANK["oem_redistributable"] >= _CLASS_RANK["licensed"], (
            "N4 invariant: oem_redistributable rank must be >= licensed "
            "so that lower-tier callers cannot read it as open data."
        )
        assert _CLASS_RANK["customer_private"] > _CLASS_RANK["oem_redistributable"], (
            "N4 invariant: customer_private must outrank oem_redistributable."
        )
