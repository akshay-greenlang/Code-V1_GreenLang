# -*- coding: utf-8 -*-
"""
FastAPI router for OEM white-label onboarding (Track C-5).

Mounts under ``/v1/oem`` and exposes the surface a third-party platform
needs to:

* sign up as a GreenLang Factors OEM partner,
* configure white-label branding (logo, colours, custom domain,
  support contact),
* provision and revoke sub-tenants under its redistribution grant,
* read the OEM context that decorates downstream API responses.

The router is included by Agent 1's main FastAPI app via:

.. code-block:: python

    try:
        from greenlang.factors.onboarding.api import oem_router
        app.include_router(oem_router)
    except Exception:
        pass

so a missing OEM module never breaks the rest of the API surface.

Authentication strategy
-----------------------
Routes that mutate or expose OEM context (``/me``, ``/branding``,
``/subtenants``, ``/redistribution``) require the caller to identify
which OEM they are acting as. We accept the OEM id either via:

* the ``X-OEM-Id`` header (preferred for SDK calls), or
* the request's ``oem_id`` query parameter (operator console fallback).

Real production uses the API gateway's signed-token middleware (see
Agent 1's ``api_auth.py``). To keep this module decoupled and unit-
testable the router here only requires the OEM id to identify the
context; the auth middleware in front of it is responsible for proving
the caller actually owns that id.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from greenlang.factors.entitlements import EntitlementError
from greenlang.factors.onboarding.branding_config import BrandingConfig
from greenlang.factors.onboarding.oem_export import (
    OemExportError,
    OemExportQuotaError,
    build_oem_export,
)
from greenlang.factors.onboarding.partner_setup import (
    OEM_ELIGIBLE_PARENT_PLANS,
    OEM_GRANT_CLASSES,
    OemError,
    OemPartner,
    create_oem_partner,
    get_oem_partner,
    get_redistribution_grant,
    list_subtenants,
    provision_subtenant,
    revoke_subtenant,
    update_branding,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

oem_router = APIRouter(
    prefix="/v1/oem",
    tags=["OEM"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class OemSignupRequest(BaseModel):
    """Payload for ``POST /v1/oem/signup``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=128)
    contact_email: EmailStr = Field(...)
    redistribution_grants: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "License-class strings the OEM is licensed to resell. "
            "Must be subset of OEM_GRANT_CLASSES."
        ),
    )
    parent_plan: str = Field(
        ...,
        description="Plan slug; one of OEM_ELIGIBLE_PARENT_PLANS.",
    )
    branding: Optional[BrandingConfig] = Field(
        default=None,
        description="Optional initial branding payload.",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1024,
    )


class SubTenantCreateRequest(BaseModel):
    """Payload for ``POST /v1/oem/subtenants``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=128)
    entitlements: List[str] = Field(
        default_factory=list,
        description=(
            "Subset of the OEM grant classes this sub-tenant may resolve."
        ),
    )
    branding: Optional[BrandingConfig] = Field(default=None)


class BrandingUpdateRequest(BaseModel):
    """Payload for ``POST /v1/oem/branding``."""

    model_config = ConfigDict(extra="forbid")

    branding: BrandingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_oem_id(
    header_id: Optional[str],
    query_id: Optional[str],
) -> str:
    """Pull the OEM id off the request; raise 401 if neither is present."""
    oem_id = (header_id or query_id or "").strip()
    if not oem_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OEM context required (provide X-OEM-Id header or oem_id query param).",
        )
    return oem_id


def _partner_payload(partner: OemPartner) -> Dict[str, Any]:
    """Serialise an OemPartner for the API response.

    We intentionally do NOT echo the OEM API key in any GET response;
    the key is only returned on signup. ``to_dict()`` already redacts
    to a prefix; we strip it entirely on subsequent reads.
    """
    payload = partner.to_dict()
    payload.pop("api_key_prefix", None)
    return payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@oem_router.post(
    "/signup",
    status_code=status.HTTP_201_CREATED,
    summary="Provision a new OEM partner.",
    response_description="The newly created OEM partner record (includes API key once).",
)
def signup_oem(payload: OemSignupRequest) -> Dict[str, Any]:
    """Create a new OEM partner and return its credentials.

    The returned ``api_key`` is shown ONCE; subsequent reads only return
    the prefix. Callers should persist the key in their own secret
    store.
    """
    try:
        partner = create_oem_partner(
            name=payload.name,
            contact_email=payload.contact_email,
            redistribution_grants=payload.redistribution_grants,
            parent_plan=payload.parent_plan,
            branding=payload.branding,
            notes=payload.notes,
        )
    except OemError as exc:
        logger.warning("OEM signup rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    body = partner.to_dict()
    # First-time signup is the only place we leak the full API key.
    body["api_key"] = partner.api_key
    body["api_key_prefix"] = partner.api_key[:20] + "..."
    return body


@oem_router.get(
    "/me",
    summary="Return the current OEM context (branding, grants, sub-tenants).",
)
def get_me(
    x_oem_id: Optional[str] = Header(default=None, alias="X-OEM-Id"),
    oem_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return the OEM partner identified by the request context."""
    resolved = _resolve_oem_id(x_oem_id, oem_id)
    try:
        partner = get_oem_partner(resolved)
    except OemError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return _partner_payload(partner)


@oem_router.post(
    "/branding",
    summary="Replace the OEM's white-label branding payload.",
)
def update_branding_route(
    body: BrandingUpdateRequest,
    x_oem_id: Optional[str] = Header(default=None, alias="X-OEM-Id"),
    oem_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Set / replace the OEM's branding configuration."""
    resolved = _resolve_oem_id(x_oem_id, oem_id)
    try:
        partner = update_branding(resolved, body.branding)
    except OemError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return {
        "oem_id": partner.id,
        "branding": body.branding.to_response_metadata(),
    }


@oem_router.post(
    "/subtenants",
    status_code=status.HTTP_201_CREATED,
    summary="Provision a sub-tenant under the OEM.",
)
def create_subtenant_route(
    body: SubTenantCreateRequest,
    x_oem_id: Optional[str] = Header(default=None, alias="X-OEM-Id"),
    oem_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Create a new sub-tenant; entitlements must be a subset of the grant."""
    resolved = _resolve_oem_id(x_oem_id, oem_id)
    try:
        sub = provision_subtenant(
            oem_id=resolved,
            subtenant_name=body.name,
            branding=body.branding,
            entitlements=body.entitlements,
        )
    except EntitlementError as exc:
        # 403 - the OEM grant does not cover the requested entitlements.
        logger.info("Sub-tenant rejected (entitlement): oem=%s err=%s", resolved, exc)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    except OemError as exc:
        logger.warning("Sub-tenant rejected (oem): oem=%s err=%s", resolved, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    payload = sub.to_dict()
    # Surface the freshly minted sub-tenant API key once.
    payload["api_key"] = sub.api_key
    return payload


@oem_router.get(
    "/subtenants",
    summary="List sub-tenants for the current OEM.",
)
def list_subtenants_route(
    x_oem_id: Optional[str] = Header(default=None, alias="X-OEM-Id"),
    oem_id: Optional[str] = Query(default=None),
    active_only: bool = Query(default=False),
) -> Dict[str, Any]:
    """Return every sub-tenant scoped to the OEM."""
    resolved = _resolve_oem_id(x_oem_id, oem_id)
    try:
        subs = list_subtenants(resolved, active_only=active_only)
    except OemError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return {
        "oem_id": resolved,
        "count": len(subs),
        "subtenants": [s.to_dict() for s in subs],
    }


@oem_router.delete(
    "/subtenants/{subtenant_id}",
    summary="Revoke a sub-tenant (soft delete).",
)
def revoke_subtenant_route(
    subtenant_id: str,
    x_oem_id: Optional[str] = Header(default=None, alias="X-OEM-Id"),
    oem_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Soft-revoke a sub-tenant; idempotent."""
    resolved = _resolve_oem_id(x_oem_id, oem_id)
    try:
        revoked = revoke_subtenant(resolved, subtenant_id)
    except OemError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    if not revoked:
        # Either the sub-tenant did not exist, or it was already revoked.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sub-tenant not found or already revoked",
        )
    return {"oem_id": resolved, "subtenant_id": subtenant_id, "revoked": True}


@oem_router.get(
    "/redistribution",
    summary="Return the OEM's redistribution grant.",
)
def get_redistribution_route(
    x_oem_id: Optional[str] = Header(default=None, alias="X-OEM-Id"),
    oem_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return the OEM's :class:`RedistributionGrant` payload."""
    resolved = _resolve_oem_id(x_oem_id, oem_id)
    try:
        grant = get_redistribution_grant(resolved)
    except OemError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    payload = grant.to_dict()
    payload["known_classes"] = list(OEM_GRANT_CLASSES)
    payload["eligible_parent_plans"] = list(OEM_ELIGIBLE_PARENT_PLANS)
    return payload


# ---------------------------------------------------------------------------
# Bulk export (Wave 5 - Track C-5.2)
# ---------------------------------------------------------------------------
#
# POST /v1/oem/export returns a signed JSONL artifact filtered to the
# factors this OEM is licensed to redistribute. A non-OEM tenant hitting
# the endpoint (i.e. a caller whose ``X-OEM-Id`` is not registered)
# receives a 403 via the ``OemError`` branch below.


class OemExportRequest(BaseModel):
    """Payload for ``POST /v1/oem/export``."""

    model_config = ConfigDict(extra="forbid")

    edition_id: Optional[str] = Field(
        default=None,
        description=(
            "Edition slug to export from. Defaults to the request's pinned "
            "edition (X-GL-Edition header or the service default)."
        ),
    )
    include_preview: bool = Field(
        default=False,
        description="Include preview-status factors in the dump.",
    )
    include_connector: bool = Field(
        default=False,
        description="Include connector-only factors in the dump.",
    )
    max_rows: Optional[int] = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Optional row ceiling; defaults to the service cap.",
    )


def _factors_service(request: Request) -> Any:
    svc = getattr(request.app.state, "factors_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="factors_service not configured; cannot export.",
        )
    return svc


def _resolve_edition(request: Request, svc: Any, requested: Optional[str]) -> str:
    if requested:
        try:
            return svc.repo.resolve_edition(requested)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown edition {requested!r}: {exc}",
            ) from exc
    pinned = getattr(request.state, "edition_id", None)
    if pinned:
        return pinned
    try:
        return svc.repo.resolve_edition(None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No default edition: {exc}",
        ) from exc


def _fetch_rows_for_export(
    svc: Any,
    edition_id: str,
    *,
    include_preview: bool,
    include_connector: bool,
    max_rows: int,
) -> List[Dict[str, Any]]:
    """Pull a capped list of factor records for the export filter.

    Repository surfaces vary across the memory / sqlite / pg back-ends so
    we probe ``list_factors`` first (canonical records) and fall back to
    ``list_factor_summaries`` when that isn't available.
    """
    repo = svc.repo
    if hasattr(repo, "list_factors"):
        try:
            rows, _total = repo.list_factors(
                edition_id,
                page=1,
                limit=max_rows,
                include_preview=include_preview,
                include_connector=include_connector,
            )
            return list(rows)
        except TypeError:
            try:
                rows, _total = repo.list_factors(edition_id, page=1, limit=max_rows)
                return list(rows)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    if hasattr(repo, "list_factor_summaries"):
        try:
            rows = repo.list_factor_summaries(edition_id)
            return [dict(r) for r in rows][:max_rows]
        except Exception:  # noqa: BLE001
            pass
    return []


@oem_router.post(
    "/export",
    summary="Export a signed bulk dump of factors this OEM may redistribute.",
    response_description=(
        "Signed JSONL artifact + manifest filtered by the OEM's grant."
    ),
)
def export_oem_bulk(
    request: Request,
    body: OemExportRequest,
    x_oem_id: Optional[str] = Header(default=None, alias="X-OEM-Id"),
    oem_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Produce a :class:`SignedArtifact` for the identified OEM.

    Authorisation:
      * The caller MUST identify as a known OEM (header or query param).
        An unknown or missing id surfaces as HTTP 403 (not 401) because
        the auth layer in front of this route has already proven the
        API key; the 403 reflects "you are not an OEM partner".
      * The OEM's ``redistribution_grant`` determines which rows are
        eligible. Rows outside the grant are silently filtered.
    """
    # A non-OEM tenant hitting this endpoint gets 403 (NOT 401). The
    # canonical failure mode is "authed but not an OEM".
    if not (x_oem_id or oem_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "not_oem",
                "message": (
                    "OEM context required (provide X-OEM-Id header). "
                    "This endpoint is only available to registered OEM partners."
                ),
            },
        )
    resolved = (x_oem_id or oem_id or "").strip()
    try:
        get_oem_partner(resolved)
    except OemError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "not_oem",
                "message": str(exc),
            },
        ) from exc

    svc = _factors_service(request)
    edition_id = _resolve_edition(request, svc, body.edition_id)
    max_rows = int(body.max_rows or 100_000)

    rows = _fetch_rows_for_export(
        svc,
        edition_id,
        include_preview=body.include_preview,
        include_connector=body.include_connector,
        max_rows=max_rows,
    )

    try:
        artifact = build_oem_export(
            resolved,
            edition_id=edition_id,
            rows=rows,
        )
    except OemExportQuotaError as exc:
        logger.info("OEM export quota hit: oem=%s err=%s", resolved, exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "oem_export_quota", "message": str(exc)},
        ) from exc
    except OemExportError as exc:
        logger.info("OEM export rejected: oem=%s err=%s", resolved, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "oem_export_invalid", "message": str(exc)},
        ) from exc

    return artifact.to_envelope()


__all__ = [
    "oem_router",
    "OemSignupRequest",
    "SubTenantCreateRequest",
    "BrandingUpdateRequest",
    "OemExportRequest",
]
