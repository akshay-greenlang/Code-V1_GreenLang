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
# A caller granted "licensed" implicitly receives "open"; "customer_private"
# is per-tenant and never granted by class alone.
_CLASS_RANK = {
    "open": 0,
    "public": 0,
    "restricted": 1,
    "licensed": 2,
    "commercial": 2,
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


def _caller_grants(request: Request) -> Set[str]:
    user = getattr(request.state, "user", None) or {}
    grants = user.get("entitlements") or user.get("packs") or []
    base = {"open", "public"}  # everyone gets open data
    base.update(g.strip().lower() for g in grants if isinstance(g, str))
    tier = (user.get("tier") or "").lower()
    if tier in ("enterprise", "internal"):
        base.update({"licensed", "commercial", "restricted"})
    elif tier in ("consulting_platform", "consulting", "platform"):
        base.update({"licensed", "restricted"})
    elif tier in ("developer_pro", "pro", "developer"):
        base.add("restricted")
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

    grants = _caller_grants(request)
    max_allowed = _max_class(grants)
    user = getattr(request.state, "user", None) or {}
    tenant_id = user.get("tenant_id") or user.get("oem_id")

    violation = _walk_for_violation(payload, max_allowed, tenant_id)
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


def _walk_for_violation(payload: Any, max_allowed: int, tenant_id: Optional[str]):
    if isinstance(payload, dict):
        klass = _redistribution_class(payload)
        if klass is not None:
            rank = _CLASS_RANK.get(klass.lower(), 0)
            owner = payload.get("tenant_id") or payload.get("owner_tenant_id")
            if rank == 3 and owner and owner != tenant_id:
                return klass, payload.get("factor_id"), "tenant_mismatch"
            if rank > max_allowed:
                return klass, payload.get("factor_id"), "class_above_grant"
        for v in payload.values():
            hit = _walk_for_violation(v, max_allowed, tenant_id)
            if hit:
                return hit
    elif isinstance(payload, list):
        for item in payload:
            hit = _walk_for_violation(item, max_allowed, tenant_id)
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
