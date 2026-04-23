# -*- coding: utf-8 -*-
"""
LicensingGuardMiddleware — enforces the CTO non-negotiable
"never mix licensing classes (open / licensed / customer-private)".

A request that asks for a factor whose ``redistribution_class`` exceeds the
caller's entitlement returns:

  * ``402 Payment Required`` — caller's plan does not include the licensed
    pack the factor belongs to (e.g. Developer Pro asking for ecoinvent).
  * ``403 Forbidden`` — factor is customer-private to a different tenant,
    or OEM cross-tenant boundary violated.

Detection is heuristic: the middleware scans the JSON body for a
``redistribution_class`` (or ``licensing.redistribution_class``) field on
any factor returned. Routes that need to bypass scanning (e.g. /coverage
counts that don't return whole records) annotate themselves by setting
``request.state.skip_licensing_scan = True``.

The decorator form ``licensing_guard(required=...)`` lets a route declare
its required entitlement upfront; missing entitlement short-circuits with
a clean 402 before any factor work runs.
"""

from __future__ import annotations

import json
import logging
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Redistribution classes ranked from least to most restrictive.
#
# Semantic rank (higher rank = more access required):
#     open                 = 0   public, redistributable, every caller
#     licensed_embedded    = 1   previously "restricted" — embedded licensee
#     licensed/commercial  = 2   requires a paid entitlement / pack SKU
#     oem_redistributable  = 2   same rank as licensed but only granted
#                                 when the caller presents a matching oem_id
#                                 AND the factor's oem_redistributable_allowed
#                                 flag is True (see ``_caller_grants``).
#     customer_private     = 3   tenant-scoped data; never granted by class
#                                 alone — only auto-granted to the owning
#                                 tenant via ``_caller_grants``.
#
# A caller granted "licensed" implicitly receives "open".  "customer_private"
# and "oem_redistributable" require the same-tenant / OEM auto-grant logic.
_CLASS_RANK = {
    "open": 0,
    "public": 0,
    "restricted": 1,
    "licensed_embedded": 1,
    "licensed": 2,
    "commercial": 2,
    "oem_redistributable": 2,
    "oem-redistributable": 2,
    "customer_private": 3,
    "customer-private": 3,
    "private": 3,
}

UPGRADE_URL = "https://greenlang.ai/pricing"


class LicensingGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if getattr(request.state, "skip_licensing_scan", False):
            return response
        if response.status_code >= 400:
            return response
        if not response.headers.get("content-type", "").startswith("application/json"):
            return response
        return await _scan_response(request, response)


def licensing_guard(required: Iterable[str]) -> Callable:
    """Route decorator. Short-circuits with 402 if caller lacks the entitlement."""

    required_set: Set[str] = {r.strip().lower() for r in required if r.strip()}

    def _decorate(func: Callable) -> Callable:
        @wraps(func)
        async def _wrapped(*args, **kwargs):
            request: Optional[Request] = kwargs.get("request") or _find_request(args)
            if request is not None:
                granted = _caller_grants(request)
                missing = required_set - granted
                if missing:
                    return JSONResponse(
                        status_code=402,
                        content={
                            "error": "licensing_gap",
                            "message": (
                                "This route requires entitlements your plan does "
                                "not include."
                            ),
                            "required_entitlements": sorted(missing),
                            "upgrade_url": UPGRADE_URL,
                        },
                    )
            return await func(*args, **kwargs)

        return _wrapped

    return _decorate


# --- helpers ---------------------------------------------------------------


def _find_request(args: tuple) -> Optional[Request]:
    for a in args:
        if isinstance(a, Request):
            return a
    return None


def _caller_grants(
    request: Request, factor: Optional[dict] = None
) -> Set[str]:
    """Compute the set of redistribution classes the caller may read.

    The base set is tier-driven (Community → open only; Pro → +restricted;
    Enterprise / Internal → +licensed / +commercial). Two per-factor
    auto-grants extend the base set when the caller is looking at a
    specific factor:

    1. **Same-tenant auto-grant for ``customer_private``** — if
       ``user.tenant_id`` is set AND equals the factor's ``tenant_id`` /
       ``owning_tenant_id`` / ``owner_tenant_id`` field, the caller
       receives the ``customer_private`` grant for that factor only.

    2. **OEM auto-grant for ``oem_redistributable``** — if
       ``user.oem_id`` is set AND the factor's class is
       ``oem_redistributable`` AND the factor carries
       ``oem_redistributable_allowed=True`` (or a matching
       ``oem_id`` / ``allowed_oem_ids`` entry), the caller receives the
       ``oem_redistributable`` grant for that factor only.

    **N7 invariant**: Community tier NEVER receives ``customer_private``
    or ``oem_redistributable``, regardless of the auto-grant rules. This
    is also enforced by :mod:`greenlang.factors.tier_enforcement` but
    re-checked here so a mis-wired middleware stack cannot leak premium
    data to community callers.
    """
    user = getattr(request.state, "user", None) or {}
    grants = user.get("entitlements") or user.get("packs") or []
    base = {"open", "public"}  # everyone gets open data
    base.update(g.strip().lower() for g in grants if isinstance(g, str))
    tier = (user.get("tier") or "").lower()
    if tier in ("enterprise", "internal"):
        base.update({"licensed", "commercial", "restricted", "licensed_embedded"})
    elif tier in ("consulting_platform", "consulting", "platform"):
        base.update({"licensed", "restricted", "licensed_embedded"})
    elif tier in ("developer_pro", "pro", "developer"):
        base.update({"restricted", "licensed_embedded"})

    # Per-factor auto-grants (same-tenant + OEM). Only applied when a
    # factor payload is supplied so the decorator path (no factor) stays
    # pure-tier based.
    if factor is not None and isinstance(factor, dict):
        # Same-tenant: owning tenant reads its own customer_private data.
        user_tenant = user.get("tenant_id")
        factor_owner = (
            factor.get("tenant_id")
            or factor.get("owning_tenant_id")
            or factor.get("owner_tenant_id")
        )
        if user_tenant and factor_owner and user_tenant == factor_owner:
            base.add("customer_private")

        # OEM partner: factor's oem_redistributable flag is live + caller
        # presents matching oem_id.
        user_oem = user.get("oem_id")
        klass = _redistribution_class(factor)
        if user_oem and klass and klass.lower().replace("-", "_") == "oem_redistributable":
            allowed_flag = bool(
                factor.get("oem_redistributable_allowed")
                or factor.get("oem_allowed")
            )
            allowed_oem_ids = factor.get("allowed_oem_ids") or []
            factor_oem_id = factor.get("oem_id")
            oem_match = (
                allowed_flag
                or user_oem == factor_oem_id
                or user_oem in allowed_oem_ids
            )
            if oem_match:
                base.add("oem_redistributable")

    # N7 invariant: community tier is *never* granted oem / private,
    # even if an upstream entitlement row accidentally contained them.
    if tier in ("community", ""):
        base.discard("customer_private")
        base.discard("customer-private")
        base.discard("private")
        base.discard("oem_redistributable")
        base.discard("oem-redistributable")

    return base


def _max_class(grants: Set[str]) -> int:
    return max((_CLASS_RANK.get(g, 0) for g in grants), default=0)


async def _scan_response(request: Request, response: Response) -> Response:
    body = b""
    if hasattr(response, "body") and response.body is not None:
        body = response.body
    else:
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            body += chunk

    try:
        payload = json.loads(body or b"null")
    except (TypeError, ValueError):
        return _rebuild(response, body)

    user = getattr(request.state, "user", None) or {}
    tenant_id = user.get("tenant_id") or user.get("oem_id")

    violation = _walk_for_violation(payload, request, tenant_id)
    if violation is not None:
        klass, factor_id, reason = violation
        status = 403 if reason == "tenant_mismatch" else 402
        return JSONResponse(
            status_code=status,
            content={
                "error": "licensing_gap" if status == 402 else "tenant_forbidden",
                "message": (
                    f"Factor {factor_id or '<unknown>'} is class '{klass}' which "
                    f"your entitlement does not cover."
                ),
                "required_class": klass,
                "factor_id": factor_id,
                "upgrade_url": UPGRADE_URL,
            },
        )
    return _rebuild(response, body)


def _walk_for_violation(
    payload: Any, request: "Request", tenant_id: Optional[str]
):
    """Walk ``payload`` looking for a factor the caller cannot read.

    Grants are recomputed **per-factor** so same-tenant and OEM auto-grants
    can fire only when the current factor's tenant/OEM metadata matches.
    """
    if isinstance(payload, dict):
        klass = _redistribution_class(payload)
        if klass is not None:
            klass_key = klass.lower().replace("-", "_")
            rank = _CLASS_RANK.get(klass.lower(), _CLASS_RANK.get(klass_key, 0))
            owner = payload.get("tenant_id") or payload.get("owner_tenant_id")
            # N7 invariant: community tier is forbidden from oem/private
            # classes regardless of any per-caller grant. Check this BEFORE
            # computing grants so nothing else can override it.
            user = getattr(request.state, "user", None) or {}
            tier = (user.get("tier") or "").lower()
            try:
                from greenlang.factors.tier_enforcement import tier_allows_class
                if not tier_allows_class(tier, klass):
                    return klass, payload.get("factor_id"), "class_above_grant"
            except ImportError:  # pragma: no cover — defensive
                pass
            # Per-factor grants (may include customer_private / oem_*).
            grants = _caller_grants(request, factor=payload)
            max_allowed = _max_class(grants)
            # Tenant mismatch on customer_private factors = 403 regardless
            # of any class-rank-based grant the caller may carry.
            if klass_key in ("customer_private", "private") and owner and owner != tenant_id:
                return klass, payload.get("factor_id"), "tenant_mismatch"
            # oem_redistributable requires an explicit, literal grant — the
            # tier-driven auto-grants (licensed / commercial) share the same
            # rank as oem but do NOT imply OEM redistribution rights. A
            # non-OEM caller with "licensed" must be denied the OEM row.
            if klass_key == "oem_redistributable":
                if "oem_redistributable" not in grants and "oem-redistributable" not in grants:
                    return klass, payload.get("factor_id"), "class_above_grant"
            if rank > max_allowed:
                return klass, payload.get("factor_id"), "class_above_grant"
        for v in payload.values():
            hit = _walk_for_violation(v, request, tenant_id)
            if hit:
                return hit
    elif isinstance(payload, list):
        for item in payload:
            hit = _walk_for_violation(item, request, tenant_id)
            if hit:
                return hit
    return None


def _redistribution_class(d: dict) -> Optional[str]:
    if "redistribution_class" in d and isinstance(d["redistribution_class"], str):
        return d["redistribution_class"]
    licensing = d.get("licensing") or d.get("license_info")
    if isinstance(licensing, dict):
        for key in ("redistribution_class", "redistribution", "class"):
            v = licensing.get(key)
            if isinstance(v, str):
                return v
    return None


def _rebuild(response: Response, body: bytes) -> Response:
    headers = dict(response.headers)
    headers.pop("content-length", None)
    return Response(
        content=body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
    )
