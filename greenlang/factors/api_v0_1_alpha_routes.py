# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 Alpha — read-only public router.

Mounted by :func:`greenlang.factors.factors_app.create_factors_app`
ONLY when :func:`greenlang.factors.release_profile.is_alpha` is True.
Implements the exact 5 GETs from CTO doc §19.1 and nothing else.

The five routes:

    GET /v1/healthz       -- service health + edition id (unauthenticated)
    GET /v1/factors       -- list factors (cursor-paginated, filtered)
    GET /v1/factors/{urn} -- get one factor by URL-encoded URN
    GET /v1/sources       -- list registered alpha sources
    GET /v1/packs         -- list factor packs grouped by source

Plus a /api/v1/{path:path} -> 410 Gone catch-all so any client still
hitting the legacy prefix gets a clear migration message.

The router intentionally does NOT use the resolve / explain code path.
It pulls factor records from the existing FactorCatalogService and
coerces them into the v0.1 shape (see ``_coerce_v0_1``). Until the
Postgres DDL lands (task #2), this is the only backing store.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse

from greenlang.factors.api_v0_1_alpha_models import (
    ALPHA_CATEGORY_ENUM,
    ErrorResponse,
    FactorListResponse,
    FactorV0_1,
    HealthzResponse,
    PackListResponse,
    PackV0_1,
    SourceListResponse,
    SourceV0_1,
)
from greenlang.factors.release_profile import current_profile

logger = logging.getLogger(__name__)


def _stamp_release_profile_header(response: Response) -> None:
    """Router-level dependency: stamp ``X-GL-Release-Profile`` on every response.

    Used as a router-level dependency so all five alpha endpoints carry the
    header without per-route boilerplate. Errors raised inside the handler
    will still return without this header (those go through HTTPException
    JSONResponses, which bypass the dependency Response object) — that's
    acceptable per the contract: only successful responses are guaranteed
    to advertise the profile. The catch-all 410 router declares its own
    explicit header below.
    """
    response.headers["X-GL-Release-Profile"] = current_profile().value


router = APIRouter(
    prefix="/v1",
    tags=["factors-v0.1-alpha"],
    dependencies=[Depends(_stamp_release_profile_header)],
)
deprecated_router = APIRouter(prefix="/api/v1", tags=["factors-v0.1-alpha-deprecated"])

ALPHA_SCHEMA_ID = "https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json"

ALPHA_ENDPOINTS_PUBLIC = [
    "/v1/healthz",
    "/v1/factors",
    "/v1/factors/{urn}",
    "/v1/sources",
    "/v1/packs",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service(request: Request):
    """Return the FactorCatalogService or None if not configured."""
    return getattr(request.app.state, "factors_service", None)


def _alpha_repo(request: Request):
    """Return the real :class:`AlphaFactorRepository` or None.

    Wave D / TaskCreate #31: when ``app.state.alpha_factor_repo`` is wired
    by ``create_factors_app()`` (under ``release_profile=alpha-v0.1``), the
    router prefers it over the legacy ``factors_service`` and skips
    ``_coerce_v0_1`` entirely — records are returned verbatim.
    """
    return getattr(request.app.state, "alpha_factor_repo", None)


def _edition_id(request: Request, svc: Any) -> Optional[str]:
    """Resolve the active edition; never raises (returns None on failure)."""
    if svc is None:
        return None
    pinned = getattr(request.state, "edition_id", None)
    if pinned:
        return pinned
    try:
        return svc.repo.resolve_edition(None)
    except Exception:  # noqa: BLE001
        return None


def _git_commit() -> Optional[str]:
    """Return git short SHA if available (env var first, then `git`)."""
    sha = (
        os.getenv("GIT_COMMIT")
        or os.getenv("GIT_SHA")
        or os.getenv("SOURCE_COMMIT")
    )
    if sha:
        return sha.strip()[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _slugify(value: Optional[str]) -> str:
    """Cheap URN-safe slug. Never returns empty (falls back to 'unknown')."""
    if not value:
        return "unknown"
    out = []
    prev_dash = False
    for ch in value.strip().lower():
        if ch.isalnum() or ch in (".", "-"):
            out.append(ch)
            prev_dash = False
        elif ch in (" ", "_", "/", ":"):
            if not prev_dash:
                out.append("-")
                prev_dash = True
        # silently drop other chars
    slug = "".join(out).strip(".-")
    return slug or "unknown"


def _strip_internal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop explain/alternates/signed_receipt keys before returning."""
    if not isinstance(payload, dict):
        return payload
    payload = dict(payload)
    for key in ("explain", "alternates", "signed_receipt", "_compact"):
        payload.pop(key, None)
    return payload


def _coerce_v0_1(record: Any, source_version: str = "0.1") -> Dict[str, Any]:
    """Coerce a legacy ``EmissionFactorRecord`` into the v0.1 shape.

    Best-effort: missing fields are filled with ``None`` rather than
    raising. The returned dict is the model_dump() shape of FactorV0_1,
    so the caller can wrap it in ``FactorV0_1.model_validate(...)`` if
    strict typing is needed.
    """
    if record is None:
        return {}

    # Common attribute access pattern: dataclass / pydantic alike.
    def get(attr: str, default: Any = None) -> Any:
        return getattr(record, attr, default)

    factor_id = get("factor_id") or ""
    source_id = get("source_id") or "unknown"
    fuel_type = get("fuel_type") or ""
    geography = get("geography") or ""
    family = get("factor_family")
    family = getattr(family, "value", family) if family is not None else None
    method_profile = get("method_profile")
    method_profile = (
        getattr(method_profile, "value", method_profile)
        if method_profile is not None
        else None
    )
    factor_name = get("factor_name") or fuel_type or factor_id
    unit = get("unit") or ""
    valid_from = get("valid_from")
    valid_to = get("valid_to")
    license_class = get("license_class") or get("license_info")
    if license_class is not None and not isinstance(license_class, str):
        license_class = getattr(license_class, "license_name", str(license_class))

    src_slug = _slugify(source_id)
    namespace = _slugify(family or fuel_type or "default")
    leaf = _slugify(factor_id.replace("EF:", "")) if factor_id else "unknown"

    # If factor_id already begins urn:gl:factor: keep it; else build a URN.
    if isinstance(factor_id, str) and factor_id.startswith("urn:gl:factor:"):
        urn = factor_id
        alias = None
    else:
        urn = f"urn:gl:factor:{src_slug}:{namespace}:{leaf}:v1"
        alias = factor_id if factor_id else None

    # CO2-equivalent scalar — collapse from gwp_100yr.total_co2e.
    value: Optional[float] = None
    gwp = get("gwp_100yr")
    if gwp is not None:
        for attr in ("total_co2e", "co2e", "value"):
            v = getattr(gwp, attr, None)
            if isinstance(v, (int, float)):
                value = float(v)
                break

    # Geography URN: country code -> urn:gl:geo:country:<lc>; else region.
    geo_urn = None
    if geography:
        if len(geography) <= 3 and geography.isalpha():
            geo_urn = f"urn:gl:geo:country:{geography.lower()}"
        else:
            geo_urn = f"urn:gl:geo:region:{_slugify(geography)}"

    # Methodology URN.
    methodology_urn = None
    if method_profile:
        methodology_urn = f"urn:gl:methodology:{_slugify(str(method_profile))}"

    # Citations — pull from license_info or provenance if present.
    citations: List[Dict[str, Any]] = []
    prov = get("provenance")
    if prov is not None:
        url = getattr(prov, "source_url", None) or getattr(prov, "url", None)
        cite_text = getattr(prov, "citation", None) or getattr(prov, "citation_text", None)
        if url or cite_text:
            citations.append(
                {
                    "url": url,
                    "citation_text": cite_text or f"Source: {source_id}",
                }
            )

    # Pack URN (synthetic until task #2 publishes real pack ids).
    pack_urn = f"urn:gl:pack:{src_slug}:default:v1"

    return {
        "urn": urn,
        "factor_id_alias": alias,
        "source_urn": f"urn:gl:source:{src_slug}",
        "factor_pack_urn": pack_urn,
        "name": factor_name,
        "description": get("notes") or None,
        "category": _normalize_category(family, get("scope")),
        "value": value,
        "unit_urn": f"urn:gl:unit:{_slugify(unit)}" if unit else None,
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": geo_urn,
        "vintage_start": str(valid_from) if valid_from else None,
        "vintage_end": str(valid_to) if valid_to else None,
        "resolution": "annual",
        "methodology_urn": methodology_urn,
        "boundary": str(get("boundary") or "") or None,
        "licence": str(license_class) if license_class else None,
        "citations": citations,
        "published_at": (
            str(get("created_at")) if get("created_at") else None
        ),
        "extraction": {
            "source_url": getattr(prov, "source_url", None) if prov else None,
            "source_version": source_version,
            "ingested_at": str(get("created_at")) if get("created_at") else None,
        },
        "review": {
            "review_status": "approved",
        },
        "tags": list(get("tags") or []),
    }


def _normalize_category(family: Optional[str], scope: Any) -> Optional[str]:
    """Map legacy factor_family / scope to the alpha v0.1 category enum."""
    if isinstance(family, str):
        f = family.lower()
        if f in ("electricity", "grid"):
            scope_val = getattr(scope, "value", scope) if scope is not None else None
            if str(scope_val) in ("2", "scope_2"):
                return "scope2_location_based"
            return "grid_intensity"
        if f in ("combustion", "stationary_combustion", "mobile_combustion"):
            return "fuel"
        if f in ("refrigerant", "refrigerants"):
            return "refrigerant"
        if f in ("fugitive",):
            return "fugitive"
        if f in ("process",):
            return "process"
        if "cbam" in f:
            return "cbam_default"
    # Fall back to scope1 if scope == 1.
    scope_val = getattr(scope, "value", scope) if scope is not None else None
    if str(scope_val) in ("1", "scope_1"):
        return "scope1"
    return None


# ---------------------------------------------------------------------------
# /v1/healthz
# ---------------------------------------------------------------------------


@router.get(
    "/healthz",
    response_model=HealthzResponse,
    summary="Service health + edition id",
    description=(
        "Unauthenticated liveness probe for v0.1 alpha. Returns the active "
        "release profile, the schema id every factor record validates "
        "against, the active edition id (when a catalog is loaded), and "
        "the build commit SHA when available."
    ),
)
def healthz(request: Request) -> HealthzResponse:
    request.state.skip_licensing_scan = True
    svc = _service(request)
    edition = _edition_id(request, svc)
    # When the real alpha repo is wired and the legacy service has no
    # edition pinned, surface the alpha repo edition tag so /v1/healthz
    # never returns null in alpha mode.
    if edition is None and _alpha_repo(request) is not None:
        edition = os.getenv("GL_FACTORS_ALPHA_EDITION_ID", "alpha-v0.1")
    return HealthzResponse(
        status="ok",
        service="greenlang-factors",
        release_profile=current_profile().value,
        schema_id=ALPHA_SCHEMA_ID,
        edition=edition,
        git_commit=_git_commit(),
        version="0.1.0",
    )


# ---------------------------------------------------------------------------
# /v1/factors  — list
# ---------------------------------------------------------------------------


@router.get(
    "/factors",
    response_model=FactorListResponse,
    summary="List factors (cursor-paginated, filtered)",
    description=(
        "Read-only listing of every factor visible to the alpha tier. "
        "Filtering is the alpha-spec subset only: by geography URN, "
        "source URN, pack URN, category enum, and vintage window. "
        "Pagination is cursor-based (opaque next_cursor; pass it back "
        "verbatim). Resolve / explain payloads are intentionally excluded."
    ),
)
def list_factors(
    request: Request,
    geography_urn: Optional[str] = Query(None, description="Filter by geography URN."),
    source_urn: Optional[str] = Query(None, description="Filter by source URN."),
    pack_urn: Optional[str] = Query(None, description="Filter by factor pack URN."),
    category: Optional[str] = Query(
        None,
        description=f"One of {ALPHA_CATEGORY_ENUM}.",
    ),
    vintage_start_after: Optional[str] = Query(
        None, description="ISO date; only factors with vintage_start > this."
    ),
    vintage_end_before: Optional[str] = Query(
        None, description="ISO date; only factors with vintage_end < this."
    ),
    cursor: Optional[str] = Query(None, description="Opaque pagination cursor."),
    limit: int = Query(50, ge=1, le=200, description="Page size (1-200)."),
) -> FactorListResponse:
    request.state.skip_licensing_scan = True

    if category is not None and category not in ALPHA_CATEGORY_ENUM:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_category",
                "message": (
                    f"category={category!r} is not in the alpha v0.1 enum."
                ),
                "allowed": ALPHA_CATEGORY_ENUM,
            },
            headers={"X-GL-Release-Profile": current_profile().value},
        )

    repo = _alpha_repo(request)
    if repo is not None:
        # Real repo path — read records verbatim, no coercion.
        try:
            rows, next_cursor = repo.list_factors(
                geography_urn=geography_urn,
                source_urn=source_urn,
                pack_urn=pack_urn,
                category=category,
                vintage_start_after=vintage_start_after,
                vintage_end_before=vintage_end_before,
                cursor=cursor,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("alpha_factor_repo.list_factors failed: %s", exc)
            rows, next_cursor = [], None
        # Phase 1 rights gate: drop commercial / private / pending /
        # blocked / errored records before model validation. Audit
        # events for non-community_open sources are emitted inside
        # _phase1_rights_filter_list.
        rows = _phase1_rights_filter_list(request, rows)
        edition_tag = os.getenv("GL_FACTORS_ALPHA_EDITION_ID", "alpha-v0.1")
        # When the legacy service is also bound, prefer its edition id so
        # the SDK e2e demo's pinned edition surfaces correctly.
        svc_for_edition = _service(request)
        legacy_edition = _edition_id(request, svc_for_edition)
        if legacy_edition is not None:
            edition_tag = legacy_edition
        return FactorListResponse(
            data=[FactorV0_1.model_validate(r) for r in rows],
            next_cursor=next_cursor,
            edition=edition_tag,
        )

    svc = _service(request)
    edition = _edition_id(request, svc)
    if svc is None or edition is None:
        # Bare service: still return a well-formed empty list.
        return FactorListResponse(data=[], next_cursor=None, edition=None)

    # Cursor is just the page index, base64 isn't worth the round-trip
    # noise for alpha. We accept either an int-string or our prefixed form.
    page = _decode_cursor(cursor)

    try:
        rows, total = svc.repo.list_factors(
            edition,
            page=page,
            limit=limit,
            include_preview=False,
            include_connector=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_factors backend failure: %s", exc)
        return FactorListResponse(data=[], next_cursor=None, edition=edition)

    # Apply the alpha filters in Python (cheap at alpha scale).
    coerced_rows: List[Dict[str, Any]] = []
    for r in rows:
        coerced = _coerce_v0_1(r)
        if geography_urn and coerced.get("geography_urn") != geography_urn:
            continue
        if source_urn and coerced.get("source_urn") != source_urn:
            continue
        if pack_urn and coerced.get("factor_pack_urn") != pack_urn:
            continue
        if category and coerced.get("category") != category:
            continue
        if vintage_start_after and (coerced.get("vintage_start") or "") <= vintage_start_after:
            continue
        if vintage_end_before:
            vend = coerced.get("vintage_end")
            if not vend or vend >= vintage_end_before:
                continue
        coerced_rows.append(coerced)

    # Phase 1 rights gate: drop denied / metadata-only / errored
    # records before model validation.
    coerced_rows = _phase1_rights_filter_list(request, coerced_rows)
    out: List[FactorV0_1] = [FactorV0_1.model_validate(c) for c in coerced_rows]

    next_cursor = None
    # Cheap "more pages?" check — total may be unreliable, so we also keep
    # paginating while the backend returned a full page.
    if len(rows) >= limit and (total is None or page * limit < int(total)):
        next_cursor = _encode_cursor(page + 1)

    return FactorListResponse(data=out, next_cursor=next_cursor, edition=edition)


def _encode_cursor(page: int) -> str:
    return f"page:{page}"


def _decode_cursor(cursor: Optional[str]) -> int:
    if not cursor:
        return 1
    if cursor.startswith("page:"):
        try:
            return max(1, int(cursor.split(":", 1)[1]))
        except ValueError:
            return 1
    try:
        return max(1, int(cursor))
    except ValueError:
        return 1


# ---------------------------------------------------------------------------
# /v1/factors/{urn}
# ---------------------------------------------------------------------------


@router.get(
    "/factors/{urn:path}",
    response_model=FactorV0_1,
    summary="Get one factor by URN",
    description=(
        "URL-encoded canonical URN (urn:gl:factor:<source>:<ns>:<id>:v<n>). "
        "Returns a 404 with a stable JSON error body if the URN is "
        "unknown in the active edition."
    ),
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Factor not found in the active edition.",
        }
    },
)
def get_factor(
    request: Request,
    urn: str = Path(..., description="URL-encoded factor URN."),
) -> FactorV0_1:
    request.state.skip_licensing_scan = True
    # FastAPI's {path} converter already decodes percent-escapes once.
    decoded_urn = urn
    profile_header = {"X-GL-Release-Profile": current_profile().value}

    repo = _alpha_repo(request)
    if repo is not None:
        try:
            rec = repo.get_by_urn(decoded_urn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alpha_factor_repo.get_by_urn failed: %s", exc)
            rec = None
        if rec is not None:
            denied = _phase1_rights_filter_one(request, rec)
            if denied is not None:
                return denied
            return FactorV0_1.model_validate(rec)
        # Fall through to legacy lookup so a partially-populated repo can
        # still surface records the legacy service knows about.

    svc = _service(request)
    edition = _edition_id(request, svc)
    if svc is None or edition is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "factor_not_found",
                "message": "No edition is loaded; the catalog is empty.",
                "urn": decoded_urn,
            },
            headers=profile_header,
        )

    record = _lookup_factor(svc, edition, decoded_urn)
    if record is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "factor_not_found",
                "message": (
                    f"Factor {decoded_urn!r} not found in edition {edition!r}."
                ),
                "urn": decoded_urn,
            },
            headers=profile_header,
        )

    coerced = _strip_internal(_coerce_v0_1(record))
    denied = _phase1_rights_filter_one(request, coerced)
    if denied is not None:
        return denied
    return FactorV0_1.model_validate(coerced)


def _phase1_rights_evaluate(
    request: Request, record: Dict[str, Any], *, action: str
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Core Phase 1 source-rights evaluation shared by single-read +
    list paths.

    Returns ``(verdict, error_payload_or_none, audit_meta)``. The
    ``verdict`` is one of:

      * ``"allow"``    — caller may surface the record verbatim.
      * ``"deny"``     — caller must refuse / filter out (audit emitted).
      * ``"metadata"`` — caller must refuse on bulk paths (audit emitted).
      * ``"error"``    — rights service raised at runtime; caller MUST
        fail closed (return 503 in production / hide the record in
        list paths).

    The ``error_payload`` is a JSON dict suitable for the response body
    when verdict is ``deny`` / ``metadata`` / ``error`` and the caller
    is the single-read path. List callers ignore it and just drop the
    record.

    Audit emission is performed inline for non-``community_open``
    sources (allow / deny / metadata / error) so callers don't need
    to remember to fire audit events.

    Fail-closed contract: any exception raised by the rights service
    OR an import failure of the rights module yields verdict
    ``"error"`` (NOT ``"allow"``). The *unknown source* case (no row
    in the rights registry) is a deliberate ``"allow"`` per the
    SourceRightsService docstring — the provenance gate is the
    canonical "is this a registered source" check.
    """
    audit_meta: Dict[str, Any] = {}
    source_urn = (
        record.get("source_urn") if isinstance(record, dict) else None
    )
    factor_urn = record.get("urn") if isinstance(record, dict) else None
    audit_meta["source_urn"] = source_urn
    audit_meta["factor_urn"] = factor_urn
    if not isinstance(source_urn, str) or not source_urn:
        # No source URN to evaluate — fail open (provenance gate
        # rejects records without a source_urn).
        return ("allow", None, audit_meta)

    user = getattr(request.state, "user", None) or {}
    tenant_id = user.get("tenant_id") if isinstance(user, dict) else None
    api_key_id = user.get("api_key_id") if isinstance(user, dict) else None
    audit_meta["tenant_id"] = tenant_id
    audit_meta["api_key_id"] = api_key_id

    try:
        from greenlang.factors.rights import (
            audit_licensed_access,
            default_service,
        )
        from greenlang.factors.rights.audit import AuditDecision
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "phase1 rights gate import failed: %s — failing CLOSED", exc
        )
        return (
            "error",
            {
                "error": "rights_unavailable",
                "message": "source-rights subsystem unavailable",
                "source_urn": source_urn,
                "factor_urn": factor_urn,
            },
            audit_meta,
        )
    try:
        decision = default_service().check_factor_read_allowed(
            tenant_id, source_urn, action=action
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "phase1 rights gate raised on %s: %s — failing CLOSED",
            source_urn, exc,
        )
        try:
            audit_licensed_access(
                tenant_id=tenant_id,
                api_key_id=api_key_id,
                source_urn=source_urn,
                factor_urn=factor_urn,
                licence_class=None,
                decision=AuditDecision.DENY,
                reason=f"rights gate runtime error: {exc}",
                request_id=request.headers.get("X-Request-Id"),
                action=action,
            )
        except Exception:  # noqa: BLE001
            pass
        return (
            "error",
            {
                "error": "rights_unavailable",
                "message": "source-rights evaluation failed",
                "source_urn": source_urn,
                "factor_urn": factor_urn,
            },
            audit_meta,
        )

    licence_class = decision.licence_class
    audit_meta["licence_class"] = licence_class
    audit_meta["reason"] = decision.reason

    # Audit every non-community_open access (allow / deny / metadata).
    if licence_class and licence_class != "community_open":
        audit_licensed_access(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            source_urn=source_urn,
            factor_urn=factor_urn,
            licence_class=licence_class,
            decision=(
                AuditDecision.ALLOW if decision.allowed
                else (
                    AuditDecision.METADATA_ONLY if decision.metadata_only
                    else AuditDecision.DENY
                )
            ),
            reason=decision.reason,
            request_id=request.headers.get("X-Request-Id"),
            action=action,
        )

    if decision.denied:
        return (
            "deny",
            {
                "error": "rights_denied",
                "message": decision.reason,
                "source_urn": source_urn,
                "factor_urn": factor_urn,
            },
            audit_meta,
        )
    if decision.metadata_only:
        return (
            "metadata",
            {
                "error": "rights_metadata_only",
                "message": decision.reason,
                "source_urn": source_urn,
            },
            audit_meta,
        )
    return ("allow", None, audit_meta)


def _phase1_rights_filter_one(
    request: Request, record: Dict[str, Any]
) -> Optional[JSONResponse]:
    """Single-factor read gate.

    Returns ``None`` when allowed; a 403 (rights_denied / rights_metadata_only)
    or 503 (rights_unavailable) JSONResponse otherwise.
    """
    verdict, payload, _meta = _phase1_rights_evaluate(
        request, record, action="read"
    )
    if verdict == "allow":
        return None
    status = 503 if verdict == "error" else 403
    return JSONResponse(status_code=status, content=payload or {})


def _phase1_rights_filter_list(
    request: Request, records: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """List-query gate.

    Drops every record that the rights service denies / flags as
    metadata-only / errors out on. Audit events are emitted inside
    :func:`_phase1_rights_evaluate` so the route layer just consumes
    the filtered list.

    Critical: this MUST be applied before the records are model_validated
    or returned, otherwise commercial / private / pending-source rows
    can leak through ``GET /v1/factors``.
    """
    out: List[Dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        verdict, _payload, _meta = _phase1_rights_evaluate(
            request, r, action="list"
        )
        if verdict == "allow":
            out.append(r)
        # deny / metadata / error → silently drop. Audit was emitted
        # inside the evaluator for non-community_open sources; for the
        # error case the evaluator already logged at ERROR.
    return out


def _lookup_factor(svc: Any, edition: str, urn: str) -> Any:
    """Try urn -> legacy alias -> raw factor_id -> None."""
    repo = svc.repo
    # 1. Direct (the catalog is mostly EF: ids today).
    try:
        rec = repo.get_factor(edition, urn)
        if rec is not None:
            return rec
    except Exception:  # noqa: BLE001
        pass
    # 2. Treat the leaf segment as a fallback.
    if urn.startswith("urn:gl:factor:"):
        leaf = urn.rsplit(":", 1)[0].rsplit(":", 1)[-1]
        try:
            rec = repo.get_factor(edition, leaf)
            if rec is not None:
                return rec
        except Exception:  # noqa: BLE001
            pass
    return None


# ---------------------------------------------------------------------------
# /v1/sources
# ---------------------------------------------------------------------------


@router.get(
    "/sources",
    response_model=SourceListResponse,
    summary="List registered alpha sources",
    description=(
        "Returns the 6 alpha-flagged sources from "
        "greenlang/factors/data/source_registry.yaml. "
        "Connector-only and beta+ sources are excluded."
    ),
)
def list_sources(request: Request) -> SourceListResponse:
    request.state.skip_licensing_scan = True
    repo = _alpha_repo(request)
    if repo is not None:
        try:
            repo_rows = repo.list_sources()
            rows = {str(r.get("source_id") or i): r for i, r in enumerate(repo_rows)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("alpha_factor_repo.list_sources failed: %s", exc)
            rows = {}
    else:
        try:
            from greenlang.factors.source_registry import alpha_v0_1_sources
            rows = alpha_v0_1_sources()
        except Exception as exc:  # noqa: BLE001
            logger.warning("alpha_v0_1_sources() failed: %s", exc)
            rows = {}

    out: List[SourceV0_1] = []
    for source_id, item in sorted(rows.items()):
        out.append(
            SourceV0_1(
                urn=str(item.get("urn") or f"urn:gl:source:{_slugify(source_id)}"),
                source_id=str(source_id),
                display_name=item.get("display_name"),
                publisher=item.get("publisher") or item.get("source_owner"),
                jurisdiction=item.get("jurisdiction"),
                license_class=item.get("license_class"),
                cadence=item.get("cadence"),
                publication_url=item.get("publication_url"),
                citation_text=item.get("citation_text"),
                source_version=item.get("source_version"),
                latest_ingestion_at=item.get("latest_ingestion_at"),
                provenance_completeness_score=item.get(
                    "provenance_completeness_score"
                ),
            )
        )
    return SourceListResponse(data=out, count=len(out))


# ---------------------------------------------------------------------------
# /v1/packs
# ---------------------------------------------------------------------------


@router.get(
    "/packs",
    response_model=PackListResponse,
    summary="List factor packs grouped by source",
    description=(
        "One synthetic pack per alpha source until task #2 publishes the "
        "full pack registry. Filtering by source_urn returns only that "
        "source's packs."
    ),
)
def list_packs(
    request: Request,
    source_urn: Optional[str] = Query(None, description="Filter by source URN."),
) -> PackListResponse:
    request.state.skip_licensing_scan = True
    repo = _alpha_repo(request)
    if repo is not None:
        try:
            repo_packs = repo.list_packs(source_urn=source_urn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alpha_factor_repo.list_packs failed: %s", exc)
            repo_packs = []
        out: List[PackV0_1] = [
            PackV0_1(
                urn=str(p.get("urn") or ""),
                source_urn=str(p.get("source_urn") or ""),
                pack_id=str(p.get("pack_id") or "default"),
                version=str(p.get("version") or "0.1"),
                display_name=p.get("display_name"),
                factor_count=p.get("factor_count"),
            )
            for p in repo_packs
        ]
        return PackListResponse(data=out, count=len(out))

    try:
        from greenlang.factors.source_registry import alpha_v0_1_sources
        rows = alpha_v0_1_sources()
    except Exception as exc:  # noqa: BLE001
        logger.warning("alpha_v0_1_sources() failed: %s", exc)
        rows = {}

    out: List[PackV0_1] = []
    for source_id, item in sorted(rows.items()):
        s_urn = str(item.get("urn") or f"urn:gl:source:{_slugify(source_id)}")
        if source_urn and s_urn != source_urn:
            continue
        version_str = str(item.get("source_version") or "0.1")
        urn = f"urn:gl:pack:{_slugify(source_id)}:default:v1"
        out.append(
            PackV0_1(
                urn=urn,
                source_urn=s_urn,
                pack_id="default",
                version=version_str,
                display_name=item.get("display_name"),
                factor_count=None,
            )
        )
    return PackListResponse(data=out, count=len(out))


# ---------------------------------------------------------------------------
# /api/v1/{path} -> 410 Gone (alpha-profile only)
# ---------------------------------------------------------------------------


@deprecated_router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
def deprecated_api_v1(request: Request, path: str) -> JSONResponse:
    """Catch-all 410 for ``/api/v1/...`` while in alpha profile."""
    return JSONResponse(
        status_code=410,
        content={
            "error": "endpoint_gone",
            "message": (
                "/api/v1 is not part of the v0.1 alpha contract; "
                "use /v1/..."
            ),
            "alpha_endpoints": ALPHA_ENDPOINTS_PUBLIC,
            "requested_path": f"/api/v1/{path}",
        },
        headers={"X-GL-Release-Profile": current_profile().value},
    )


__all__ = [
    "router",
    "deprecated_router",
    "ALPHA_ENDPOINTS_PUBLIC",
    "ALPHA_SCHEMA_ID",
]
