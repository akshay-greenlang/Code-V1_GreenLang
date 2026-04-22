# -*- coding: utf-8 -*-
"""
GreenLang API - Emission Factor Routes (FULLY IMPLEMENTED)

REST API endpoints for the GreenLang Factors catalog.

Routes (ordered: static before parametrized):
- GET  /api/v1/factors            - List factors with pagination
- GET  /api/v1/factors/search     - Basic text search
- POST /api/v1/factors/search/v2  - Advanced search with sort/pagination
- GET  /api/v1/factors/search/facets - Facet counts for filter UIs
- POST /api/v1/factors/match      - Activity-to-factor matching
- GET  /api/v1/factors/export     - Bulk export (Pro/Enterprise)
- GET  /api/v1/factors/coverage   - Coverage statistics
- GET  /api/v1/factors/{factor_id}              - Get single factor
- GET  /api/v1/factors/{factor_id}/audit-bundle - Audit bundle (Enterprise)
- GET  /api/v1/factors/{factor_id}/diff         - Factor diff between editions
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)

from greenlang.integration.api.models import (
    CoverageStats,
    EmissionFactorResponse,
    EmissionFactorSummary,
    ErrorResponse,
    FactorListResponse,
    FactorMatchRequest,
    FactorMatchCandidate,
    FactorMatchResponse,
    FactorSearchFacetsResponse,
    FactorSearchResponse,
    StatsResponse,
)
from greenlang.integration.api.dependencies import (
    get_current_user,
    get_factor_service,
)
from greenlang.factors.middleware.rate_limiter import (
    apply_export_rate_limit,
    apply_rate_limit,
)
from greenlang.factors.observability.prometheus import (
    record_match_score,
    record_search_results,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/factors", tags=["Factors"])


# ==================== Helpers ====================


def _resolve_edition(svc, request: Request, edition: Optional[str] = None) -> str:
    """Resolve edition from header -> query -> default."""
    from greenlang.factors.service import resolve_edition_id

    header_edition = request.headers.get("X-Factors-Edition")
    resolved, _source = resolve_edition_id(svc.repo, header_edition, edition)
    return resolved


def _resolve_tier(user: dict) -> str:
    """Extract tier string from JWT user context."""
    from greenlang.factors.tier_enforcement import resolve_tier

    return resolve_tier(user)


def _get_visibility(user: dict, include_preview: bool = False, include_connector: bool = False):
    """Get tier-clamped visibility flags."""
    from greenlang.factors.tier_enforcement import enforce_tier_on_request

    return enforce_tier_on_request(
        user,
        requested_preview=include_preview,
        requested_connector=include_connector,
    )


def _factor_to_summary(f) -> dict:
    """Convert EmissionFactorRecord to EmissionFactorSummary-compatible dict."""
    return {
        "factor_id": f.factor_id,
        "fuel_type": f.fuel_type,
        "unit": f.unit,
        "geography": f.geography,
        "scope": f.scope.value,
        "boundary": f.boundary.value,
        "co2e_per_unit": f.gwp_100yr.co2e_total,
        "source": f.provenance.source_org,
        "source_year": f.provenance.source_year,
        "data_quality_score": f.dqs.overall_score,
        "uncertainty_percent": (f.uncertainty_95ci or 0.0) * 100,
        "factor_status": getattr(f, "factor_status", "certified") or "certified",
        "source_id": getattr(f, "source_id", None),
        "release_version": getattr(f, "release_version", None),
        "license_class": getattr(f, "license_class", None),
        "activity_tags": list(getattr(f, "activity_tags", []) or []),
        "sector_tags": list(getattr(f, "sector_tags", []) or []),
    }


def _extract_gwp_set(f) -> str:
    """Locate the GWP set enum either on the record or on gwp_100yr."""
    gs = getattr(f, "gwp_set", None)
    if gs is None:
        gs = getattr(getattr(f, "gwp_100yr", None), "gwp_set", None)
    if gs is None:
        return ""
    return gs.value if hasattr(gs, "value") else str(gs)


def _factor_to_detailed(f) -> dict:
    """Convert EmissionFactorRecord to EmissionFactorResponse-compatible dict."""
    return {
        "factor_id": f.factor_id,
        "fuel_type": f.fuel_type,
        "unit": f.unit,
        "geography": f.geography,
        "geography_level": getattr(f, "geography_level", "country") or "country",
        "scope": f.scope.value,
        "boundary": f.boundary.value,
        # Per-gas emissions live on the GHGVectors (``vectors``) —
        # ``gwp_100yr`` is the GWPValues multipliers object, not the
        # per-unit emissions.  Fall back gracefully when a test fixture
        # predates the schema split.
        "co2_per_unit": float(getattr(f, "vectors", f.gwp_100yr).CO2),
        "ch4_per_unit": float(getattr(f, "vectors", f.gwp_100yr).CH4),
        "n2o_per_unit": float(getattr(f, "vectors", f.gwp_100yr).N2O),
        "co2e_per_unit": f.gwp_100yr.co2e_total,
        # gwp_set lives on the gwp_100yr container; callers may also
        # supply it at the record level on older fixtures.
        "gwp_set": _extract_gwp_set(f),
        "ch4_gwp": getattr(f.gwp_100yr, "CH4_gwp", getattr(f.gwp_100yr, "ch4_gwp", 28)),
        "n2o_gwp": getattr(f.gwp_100yr, "N2O_gwp", getattr(f.gwp_100yr, "n2o_gwp", 273)),
        "data_quality": {
            "temporal": f.dqs.temporal,
            "geographical": f.dqs.geographical,
            "technological": f.dqs.technological,
            "representativeness": f.dqs.representativeness,
            "methodological": f.dqs.methodological,
            "overall_score": f.dqs.overall_score,
            "rating": f.dqs.rating.value if hasattr(f.dqs.rating, "value") else str(f.dqs.rating),
        },
        "source": {
            "organization": f.provenance.source_org,
            "publication": f.provenance.source_publication,
            "year": f.provenance.source_year,
            "url": getattr(f.provenance, "source_url", None),
            "methodology": (
                f.provenance.methodology.value
                if hasattr(f.provenance.methodology, "value")
                else str(f.provenance.methodology)
            ),
            "version": f.provenance.version,
        },
        "uncertainty_95ci": f.uncertainty_95ci or 0.0,
        "valid_from": str(f.valid_from),
        "valid_to": str(f.valid_to) if f.valid_to else None,
        "license": f.license_info.license,
        "compliance_frameworks": list(f.compliance_frameworks or []),
        "tags": list(f.tags or []),
        "notes": f.notes,
        "factor_status": getattr(f, "factor_status", "certified") or "certified",
        "source_id": getattr(f, "source_id", None),
        "source_release": getattr(f, "source_release", None),
        "source_record_id": getattr(f, "source_record_id", None),
        "release_version": getattr(f, "release_version", None),
        "replacement_factor_id": getattr(f, "replacement_factor_id", None),
        "license_class": getattr(f, "license_class", None),
        "activity_tags": list(getattr(f, "activity_tags", []) or []),
        "sector_tags": list(getattr(f, "sector_tags", []) or []),
        "redistribution_allowed": getattr(f.license_info, "redistribution_allowed", None),
    }


# ==================== STATIC ROUTES (before parametrized) ====================


@router.get(
    "",
    response_model=FactorListResponse,
    summary="List emission factors",
    description="Retrieve paginated list of emission factors with optional filters",
    responses={
        200: {"description": "Successfully retrieved factors"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
    },
)
async def list_factors(
    request: Request,
    response: Response,
    fuel_type: Optional[str] = Query(None, description="Filter by fuel type"),
    geography: Optional[str] = Query(None, description="Filter by geography (ISO code)"),
    scope: Optional[str] = Query(None, description="Filter by scope (1, 2, or 3)"),
    boundary: Optional[str] = Query(None, description="Filter by boundary"),
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    include_preview: bool = Query(False, description="Include preview-status factors"),
    include_connector: bool = Query(False, description="Include connector-only factors"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(100, ge=1, le=500, description="Items per page (max 500)"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> FactorListResponse:
    """List emission factors with filtering and pagination."""
    apply_rate_limit(request, response, current_user)

    from greenlang.factors.api_endpoints import (
        cache_control_for_list,
        check_etag_match,
        compute_list_etag,
    )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    vis = _get_visibility(current_user, include_preview, include_connector)

    factors, total = svc.repo.list_factors(
        edition_id,
        fuel_type=fuel_type,
        geography=geography,
        scope=scope,
        boundary=boundary,
        page=page,
        limit=limit,
        include_preview=vis.include_preview,
        include_connector=vis.include_connector,
    )

    # ETag support for list endpoint
    etag = compute_list_etag(factors, edition_id)
    if_none_match = request.headers.get("If-None-Match")
    if check_etag_match(if_none_match, etag):
        return Response(status_code=304)

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_list()
    response.headers["X-Factors-Edition"] = edition_id

    summaries = [EmissionFactorSummary(**_factor_to_summary(f)) for f in factors]
    total_pages = max(1, math.ceil(total / limit))

    return FactorListResponse(
        factors=summaries,
        total_count=total,
        page=page,
        page_size=limit,
        total_pages=total_pages,
        edition_id=edition_id,
    )


@router.get(
    "/search",
    response_model=FactorSearchResponse,
    summary="Search emission factors",
    description="Full-text search across emission factors",
    responses={
        200: {"description": "Search results"},
        400: {"model": ErrorResponse, "description": "Invalid search query"},
    },
)
async def search_factors(
    request: Request,
    response: Response,
    q: str = Query(..., min_length=1, description="Search query string"),
    geography: Optional[str] = Query(None, description="Filter by geography"),
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    include_preview: bool = Query(False, description="Include preview-status factors"),
    include_connector: bool = Query(False, description="Include connector-only factors"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> FactorSearchResponse:
    """Search emission factors with full-text search."""
    apply_rate_limit(request, response, current_user)

    from greenlang.factors.api_endpoints import (
        cache_control_for_list,
        check_etag_match,
        compute_search_etag,
    )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    vis = _get_visibility(current_user, include_preview, include_connector)
    t0 = time.monotonic()

    results = svc.repo.search_factors(
        edition_id,
        query=q,
        geography=geography,
        limit=limit,
        include_preview=vis.include_preview,
        include_connector=vis.include_connector,
    )

    # ETag support for search endpoint
    etag = compute_search_etag(q, results, edition_id)
    if_none_match = request.headers.get("If-None-Match")
    if check_etag_match(if_none_match, etag):
        return Response(status_code=304)

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_list()
    response.headers["X-Factors-Edition"] = edition_id

    elapsed_ms = (time.monotonic() - t0) * 1000
    summaries = [EmissionFactorSummary(**_factor_to_summary(f)) for f in results]

    # Record Prometheus metrics for search result count
    record_search_results(len(summaries))

    return FactorSearchResponse(
        query=q,
        factors=summaries,
        count=len(summaries),
        search_time_ms=round(elapsed_ms, 2),
        edition_id=edition_id,
    )


@router.post(
    "/search/v2",
    summary="Advanced factor search",
    description="POST-body search with sort, pagination, and advanced filters",
    responses={
        200: {"description": "Search results with pagination metadata"},
        400: {"model": ErrorResponse, "description": "Invalid search request"},
    },
)
async def search_factors_v2(
    request: Request,
    response: Response,
    body: Dict[str, Any],
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
):
    """Enhanced search with post-filtering, sorting, and offset-based pagination."""
    apply_rate_limit(request, response, current_user)

    from greenlang.factors.api_endpoints import (
        SearchV2Request,
        cache_control_for_list,
        check_etag_match,
        compute_search_etag,
        search_v2,
    )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    vis = _get_visibility(
        current_user,
        body.get("include_preview", False),
        body.get("include_connector", False),
    )

    try:
        req = SearchV2Request(
            query=body.get("query", ""),
            geography=body.get("geography"),
            fuel_type=body.get("fuel_type"),
            scope=body.get("scope"),
            source_id=body.get("source_id"),
            factor_status=body.get("factor_status"),
            license_class=body.get("license_class"),
            dqs_min=body.get("dqs_min"),
            valid_on_date=body.get("valid_on_date"),
            sector_tags=body.get("sector_tags"),
            activity_tags=body.get("activity_tags"),
            sort_by=body.get("sort_by", "relevance"),
            sort_order=body.get("sort_order", "desc"),
            offset=body.get("offset", 0),
            limit=body.get("limit", 20),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid search request: {exc}")

    if not req.query:
        raise HTTPException(status_code=400, detail="query field is required")

    result = search_v2(
        svc.repo,
        edition_id,
        req,
        include_preview=vis.include_preview,
        include_connector=vis.include_connector,
    )

    # Record Prometheus metrics for search v2 result count
    result_count = len(result.factors) if hasattr(result, "factors") else 0
    record_search_results(result_count)

    # ETag support for search v2 endpoint
    result_dict = result.to_dict()
    etag = compute_search_etag(req.query, result.factors, edition_id)
    if_none_match = request.headers.get("If-None-Match")
    if check_etag_match(if_none_match, etag):
        return Response(status_code=304)

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_list()
    response.headers["X-Factors-Edition"] = edition_id

    return result_dict


@router.get(
    "/search/facets",
    response_model=FactorSearchFacetsResponse,
    summary="Get facet counts for filter UIs",
    description="Returns value counts for geography, scope, fuel_type, etc.",
    responses={200: {"description": "Facet counts"}},
)
async def search_facets(
    request: Request,
    response: Response,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    include_preview: bool = Query(False, description="Include preview factors"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> FactorSearchFacetsResponse:
    """Get facet value counts for building filter UIs."""
    apply_rate_limit(request, response, current_user)

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    vis = _get_visibility(current_user, include_preview, False)
    facets_data = svc.repo.search_facets(
        edition_id,
        include_preview=vis.include_preview,
        include_connector=vis.include_connector,
    )

    return FactorSearchFacetsResponse(
        edition_id=facets_data.get("edition_id", edition_id),
        facets=facets_data.get("facets", {}),
    )


# ==================== Public: three-label catalog status (Phase 5.3) ====================


@router.get(
    "/status/summary",
    summary="Catalog status summary (three-label dashboard)",
    description=(
        "Public (unauthenticated) counts of factors by coverage label "
        "(Certified / Preview / Connector-only / Deprecated), plus per-source "
        "breakdown. Backs the FactorsCatalogStatus frontend page. "
        "Cacheable for 5 minutes."
    ),
    responses={200: {"description": "Counts by label and source"}},
)
async def status_summary(
    request: Request,
    response: Response,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Return count-by-label aggregation for the public dashboard.

    Intentionally unauthenticated: the coverage matrix is a public trust
    signal and must be reachable without credentials.
    """
    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    summary = svc.status_summary(edition_id)

    response.headers["Cache-Control"] = "public, max-age=300"
    response.headers["X-Factors-Edition"] = edition_id
    # Row-count hint for the auth/metering middleware (not authenticated here).
    response.headers["X-Row-Count"] = str(summary["totals"].get("all", 0))
    return summary


# ==================== Public: watch pipeline status (Phase 5.4) ====================


@router.get(
    "/watch/status",
    summary="Watch-pipeline status (per-source results)",
    description=(
        "Public (unauthenticated) snapshot of the source-watch pipeline. "
        "Returns up to N recent check results per source (default 10). "
        "Backed by the ``watch_results`` table (migration V430). "
        "Cacheable for 5 minutes."
    ),
    responses={200: {"description": "Per-source watch result snapshot"}},
)
async def watch_status(
    request: Request,
    response: Response,
    limit_per_source: int = Query(
        10, ge=1, le=50, description="Recent results per source (max 50)"
    ),
) -> Dict[str, Any]:
    """Return the last N watch-result rows per source.

    Intentionally unauthenticated: the watch status is a public trust
    signal ("are our factor sources healthy?").  Local-dev uses the
    ``GL_FACTORS_WATCH_SQLITE`` path; production pulls from Postgres via
    ``GL_FACTORS_DATABASE_URL``.
    """
    from greenlang.factors.watch.status_api import collect_watch_status

    summary = collect_watch_status(limit_per_source=limit_per_source)
    response.headers["Cache-Control"] = "public, max-age=300"
    response.headers["X-Row-Count"] = str(summary.get("source_count", 0))
    return summary


@router.post(
    "/match",
    response_model=FactorMatchResponse,
    summary="Match activity to emission factors",
    description="Takes a natural language activity description and returns best-matching factors",
    responses={
        200: {"description": "Matched factors with scores"},
        400: {"model": ErrorResponse, "description": "Invalid match request"},
    },
)
async def match_factors(
    request: Request,
    response: Response,
    body: FactorMatchRequest,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> FactorMatchResponse:
    """Match an activity description to emission factors using hybrid search."""
    apply_rate_limit(request, response, current_user)

    from greenlang.factors.matching.pipeline import MatchRequest, run_match

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    vis = _get_visibility(current_user, True, True)

    match_req = MatchRequest(
        activity_description=body.activity_description,
        geography=body.geography,
        fuel_type=body.fuel_type,
        scope=body.scope,
        limit=body.limit,
    )

    candidates = run_match(
        svc.repo,
        edition_id,
        match_req,
        include_preview=vis.include_preview,
        include_connector=vis.include_connector,
    )

    # Record Prometheus metrics for match results
    record_search_results(len(candidates))
    if candidates:
        record_match_score(candidates[0]["score"])

    return FactorMatchResponse(
        edition_id=edition_id,
        candidates=[
            FactorMatchCandidate(
                factor_id=c["factor_id"],
                score=c["score"],
                explanation=c.get("explanation", {}),
            )
            for c in candidates
        ],
    )


@router.get(
    "/export",
    summary="Bulk export factors",
    description="Export full factor datasets as JSON Lines (Pro/Enterprise)",
    responses={
        200: {"description": "JSON Lines stream of factors"},
        403: {"model": ErrorResponse, "description": "Tier insufficient for export"},
    },
)
async def export_factors(
    request: Request,
    response: Response,
    format: str = Query("json", description="Export format: json"),
    geography: Optional[str] = Query(None, description="Filter by geography"),
    fuel_type: Optional[str] = Query(None, description="Filter by fuel type"),
    scope: Optional[str] = Query(None, description="Filter by scope"),
    factor_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    source_id: Optional[str] = Query(None, description="Filter by source"),
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
):
    """Bulk export factors as list of dicts."""
    apply_export_rate_limit(request, response, current_user)

    from greenlang.factors.api_endpoints import bulk_export_factors, bulk_export_manifest
    from greenlang.factors.tier_enforcement import TierVisibility

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    tier = _resolve_tier(current_user)
    vis = TierVisibility.from_tier(tier)

    if not vis.bulk_export_allowed:
        raise HTTPException(
            status_code=403,
            detail="Bulk export requires Pro or Enterprise tier",
        )

    rows = bulk_export_factors(
        svc.repo,
        edition_id,
        status_filter=factor_status,
        geography=geography,
        fuel_type=fuel_type,
        scope=scope,
        source_id=source_id,
        include_preview=vis.include_preview,
        include_connector=vis.include_connector,
        max_rows=vis.max_export_rows,
    )

    manifest = bulk_export_manifest(svc.repo, edition_id, len(rows))
    response.headers["X-Factors-Edition"] = edition_id
    response.headers["X-Factors-Manifest-Hash"] = manifest.get("manifest_hash", "")

    return {"manifest": manifest, "factors": rows}


@router.get(
    "/coverage",
    response_model=CoverageStats,
    summary="Get coverage statistics",
    description="Statistics about emission factor coverage by geography, scope, etc.",
    responses={200: {"description": "Coverage statistics"}},
)
async def get_coverage(
    request: Request,
    response: Response,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> CoverageStats:
    """Get emission factor coverage statistics."""
    apply_rate_limit(request, response, current_user)

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    stats = svc.repo.coverage_stats(edition_id)
    return CoverageStats(**stats)


# ==================== PARAMETRIZED ROUTES ====================


@router.get(
    "/{factor_id}",
    response_model=EmissionFactorResponse,
    summary="Get emission factor by ID",
    description="Retrieve detailed information for a specific emission factor",
    responses={
        200: {"description": "Successfully retrieved factor"},
        304: {"description": "Not modified (ETag match)"},
        404: {"model": ErrorResponse, "description": "Factor not found"},
    },
)
async def get_factor(
    request: Request,
    response: Response,
    factor_id: str,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> EmissionFactorResponse:
    """Get detailed emission factor by ID with ETag/Cache-Control support."""
    apply_rate_limit(request, response, current_user)

    from greenlang.factors.api_endpoints import (
        cache_control_for_status,
        check_etag_match,
        compute_etag,
    )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    factor = svc.repo.get_factor(edition_id, factor_id)
    if not factor:
        raise HTTPException(
            status_code=404,
            detail=f"Factor {factor_id!r} not found in edition {edition_id}",
        )

    # ETag support
    etag = compute_etag(factor)
    if_none_match = request.headers.get("If-None-Match")
    if check_etag_match(if_none_match, etag):
        return Response(status_code=304)

    factor_status = getattr(factor, "factor_status", "certified") or "certified"
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_status(factor_status)
    response.headers["X-Factors-Edition"] = edition_id

    return EmissionFactorResponse(**_factor_to_detailed(factor))


@router.get(
    "/{factor_id}/audit-bundle",
    summary="Get audit bundle for a factor",
    description="Full provenance and verification bundle (Enterprise only)",
    responses={
        200: {"description": "Audit bundle with provenance chain"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Factor not found"},
    },
)
async def get_audit_bundle(
    request: Request,
    response: Response,
    factor_id: str,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
):
    """Get audit bundle with full provenance for auditors."""
    apply_rate_limit(request, response, current_user)

    from greenlang.factors.api_endpoints import build_audit_bundle
    from greenlang.factors.tier_enforcement import TierVisibility

    tier = _resolve_tier(current_user)
    vis = TierVisibility.from_tier(tier)
    if not vis.audit_bundle_allowed:
        raise HTTPException(
            status_code=403,
            detail="Audit bundle requires Enterprise tier",
        )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    bundle = build_audit_bundle(svc.repo, edition_id, factor_id)
    if bundle is None:
        raise HTTPException(
            status_code=404,
            detail=f"Factor {factor_id!r} not found in edition {edition_id}",
        )

    return bundle


@router.get(
    "/{factor_id}/diff",
    summary="Diff a factor between editions",
    description="Compare a specific factor field-by-field between two editions",
    responses={
        200: {"description": "Field-level diff"},
        400: {"model": ErrorResponse, "description": "Missing edition parameters"},
    },
)
async def get_factor_diff(
    request: Request,
    response: Response,
    factor_id: str,
    left_edition: str = Query(..., description="Left (older) edition ID"),
    right_edition: str = Query(..., description="Right (newer) edition ID"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
):
    """Compare a specific factor between two editions, field by field."""
    apply_rate_limit(request, response, current_user)

    from greenlang.factors.api_endpoints import diff_factor_between_editions

    try:
        svc.repo.resolve_edition(left_edition)
        svc.repo.resolve_edition(right_edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return diff_factor_between_editions(svc.repo, factor_id, left_edition, right_edition)


# ==================== GAP-2: Explain endpoints (Pro+ tier) ====================
#
# These routes expose the full 7-step resolution cascade + alternates so
# consultants and auditors can justify *why* a given factor was chosen
# over its peers.  Every response includes:
#
#   * chosen_factor + source/version/vintage
#   * fallback_rank (1..7) + step_label + why_chosen
#   * alternates_considered (top-N, configurable, max 20)
#   * quality_score + uncertainty_band
#   * gas_breakdown (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3, biogenic_CO2
#     kept as *separate* components — CTO non-negotiable, never rolled up)
#   * co2e_basis (which GWP set was applied)
#   * assumptions (list of text assumption strings)
#   * deprecation_status + deprecation_replacement
#
# Every response carries ``X-GreenLang-Edition`` and
# ``X-GreenLang-Method-Profile`` headers so downstream caches + audit
# logs can pin on both dimensions.
#
# Tier gate: Pro / Consulting / Enterprise / Internal.  Community tier
# receives HTTP 403.


def _require_pro_tier(current_user: dict) -> str:
    """Enforce Pro+ tier on explain routes.

    Returns the resolved tier string, or raises HTTP 403 for community.
    """
    from greenlang.factors.tier_enforcement import Tier

    tier = _resolve_tier(current_user)
    allowed = {
        Tier.PRO.value,
        Tier.CONSULTING.value,
        Tier.ENTERPRISE.value,
        Tier.INTERNAL.value,
    }
    if tier not in allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                "Factor explain endpoints require Pro, Consulting, "
                "Enterprise, or Internal tier."
            ),
        )
    return tier


def _set_explain_headers(
    response: Response,
    *,
    edition_id: str,
    method_profile: str,
) -> None:
    """Apply the GreenLang explain headers required by the CTO spec."""
    response.headers["X-GreenLang-Edition"] = edition_id
    response.headers["X-GreenLang-Method-Profile"] = method_profile
    # Keep the legacy X-Factors-Edition header for backward compatibility
    # with the rest of the factors routes.
    response.headers["X-Factors-Edition"] = edition_id


@router.get(
    "/{factor_id}/explain",
    summary="Explain a factor (full resolution payload)",
    description=(
        "Return the complete ``ResolvedFactor`` payload for a specific "
        "factor_id, showing why this factor would win for a default "
        "activity context.\n\n"
        "Required tier: **Pro, Consulting, Enterprise, or Internal**.\n\n"
        "Response includes the chosen factor, alternates considered, "
        "tie-break reasons, quality score, uncertainty band, gas "
        "breakdown (CO2/CH4/N2O/HFCs/PFCs/SF6/NF3/biogenic_CO2 kept "
        "separate per CTO non-negotiable), assumptions, deprecation "
        "status, and fallback_rank (which of the 7 cascade steps "
        "produced the winner).\n\n"
        "Example:\n\n"
        "```bash\n"
        "curl -H 'Authorization: Bearer $TOKEN' \\\n"
        "     https://api.greenlang.io/api/v1/factors/EF:US:diesel:2024:v1/explain"
        "?method_profile=corporate_scope1&limit=5\n"
        "```"
    ),
    responses={
        200: {"description": "Resolved factor explain payload"},
        304: {"description": "Not modified (ETag match)"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Factor not found"},
    },
)
async def explain_factor(
    request: Request,
    response: Response,
    factor_id: str,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    method_profile: Optional[str] = Query(
        None,
        description=(
            "Override the default method profile.  Defaults to the factor's "
            "own ``method_profile`` or one derived from its scope."
        ),
    ),
    limit: Optional[int] = Query(
        None,
        ge=1,
        le=20,
        description="Number of alternates to return (default 5, max 20).",
    ),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Return the full ``ResolvedFactor`` payload for a specific factor_id."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.api_endpoints import (
        build_factor_explain,
        cache_control_for_explain,
        check_etag_match,
        clamp_alternates_limit,
        compute_explain_etag,
    )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    from greenlang.factors.resolution.engine import ResolutionError

    try:
        payload = build_factor_explain(
            svc.repo,
            edition_id,
            factor_id,
            method_profile=method_profile,
            alternates_limit=clamp_alternates_limit(limit),
        )
    except ResolutionError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Factor {factor_id!r} cannot be resolved under method "
                f"profile {method_profile!r}: {exc}"
            ),
        )
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Factor {factor_id!r} not found in edition {edition_id}",
        )

    # ETag support.
    etag = compute_explain_etag(payload, edition_id)
    if_none_match = request.headers.get("If-None-Match")
    if check_etag_match(if_none_match, etag):
        return Response(status_code=304)

    profile = payload.get("method_profile") or "corporate_scope1"
    _set_explain_headers(response, edition_id=edition_id, method_profile=profile)
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_explain()

    return payload


@router.post(
    "/resolve-explain",
    summary="Resolve + explain from a full ResolutionRequest",
    description=(
        "Body: ``ResolutionRequest`` (activity, method_profile, "
        "jurisdiction, reporting_date, supplier_id, facility_id, "
        "utility_or_grid_region, preferred_sources, extras, …).\n\n"
        "Required tier: **Pro, Consulting, Enterprise, or Internal**.\n\n"
        "Returns the full ``ResolvedFactor`` payload with the chosen "
        "factor, up to 20 alternates (configurable via ``?limit=N``), "
        "tie-break reasons, quality & uncertainty, gas breakdown, "
        "assumptions, deprecation status, and the fallback_rank of the "
        "winning cascade step.\n\n"
        "Example body:\n\n"
        "```json\n"
        "{\n"
        "  \"activity\": \"diesel combustion stationary\",\n"
        "  \"method_profile\": \"corporate_scope1\",\n"
        "  \"jurisdiction\": \"US\",\n"
        "  \"reporting_date\": \"2026-06-01\"\n"
        "}\n"
        "```"
    ),
    responses={
        200: {"description": "Full resolution + explain payload"},
        400: {"model": ErrorResponse, "description": "Invalid request body"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        422: {"model": ErrorResponse, "description": "No factor could be resolved"},
    },
)
async def resolve_explain(
    request: Request,
    response: Response,
    body: Dict[str, Any],
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    limit: Optional[int] = Query(
        None,
        ge=1,
        le=20,
        description="Number of alternates to return (default 5, max 20).",
    ),
    include_preview: bool = Query(
        False, description="Include preview-status factors."
    ),
    include_connector: bool = Query(
        False, description="Include connector-only factors (Enterprise only)."
    ),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Run the 7-step cascade on a user-supplied ResolutionRequest."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.api_endpoints import (
        build_resolution_explain,
        cache_control_for_explain,
        clamp_alternates_limit,
        compute_explain_etag,
    )
    from greenlang.factors.resolution.engine import ResolutionError

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    vis = _get_visibility(current_user, include_preview, include_connector)

    try:
        payload = build_resolution_explain(
            svc.repo,
            edition_id,
            body,
            alternates_limit=clamp_alternates_limit(limit),
            include_preview=vis.include_preview,
            include_connector=vis.include_connector,
        )
    except ResolutionError as exc:
        raise HTTPException(status_code=422, detail=f"Resolution failed: {exc}")
    except Exception as exc:
        # Pydantic ValidationError + any other request-shape error.
        raise HTTPException(
            status_code=400, detail=f"Invalid ResolutionRequest: {exc}"
        )

    profile = payload.get("method_profile") or body.get(
        "method_profile", "corporate_scope1"
    )
    etag = compute_explain_etag(payload, edition_id)
    _set_explain_headers(response, edition_id=edition_id, method_profile=profile)
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_explain()

    return payload


@router.get(
    "/{factor_id}/alternates",
    summary="List alternative factors for the same activity",
    description=(
        "Return the top-N alternate factors the resolution engine would "
        "also consider for the activity represented by ``factor_id``. "
        "Useful for consultants doing methodology justification or "
        "sensitivity analysis.\n\n"
        "Required tier: **Pro, Consulting, Enterprise, or Internal**.\n\n"
        "Example:\n\n"
        "```bash\n"
        "curl -H 'Authorization: Bearer $TOKEN' \\\n"
        "     https://api.greenlang.io/api/v1/factors/"
        "EF:US:diesel:2024:v1/alternates?limit=10\n"
        "```"
    ),
    responses={
        200: {"description": "List of alternate factor candidates"},
        304: {"description": "Not modified (ETag match)"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Factor not found"},
    },
)
async def factor_alternates(
    request: Request,
    response: Response,
    factor_id: str,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    method_profile: Optional[str] = Query(
        None,
        description="Override the default method profile for ranking.",
    ),
    limit: Optional[int] = Query(
        None,
        ge=1,
        le=20,
        description="Number of alternates to return (default 5, max 20).",
    ),
    include_preview: bool = Query(False),
    include_connector: bool = Query(False),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Return alternative factors that could resolve for the same activity."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.api_endpoints import (
        build_factor_alternates,
        cache_control_for_explain,
        check_etag_match,
        clamp_alternates_limit,
        compute_explain_etag,
    )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    vis = _get_visibility(current_user, include_preview, include_connector)

    payload = build_factor_alternates(
        svc.repo,
        edition_id,
        factor_id,
        method_profile=method_profile,
        alternates_limit=clamp_alternates_limit(limit),
        include_preview=vis.include_preview,
        include_connector=vis.include_connector,
    )
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Factor {factor_id!r} not found in edition {edition_id}",
        )

    etag = compute_explain_etag(payload, edition_id)
    if_none_match = request.headers.get("If-None-Match")
    if check_etag_match(if_none_match, etag):
        return Response(status_code=304)

    profile = payload.get("method_profile") or "corporate_scope1"
    _set_explain_headers(response, edition_id=edition_id, method_profile=profile)
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_explain()

    return payload


# ==================== Factor Quality Score (0-100) surface ====================


@router.get(
    "/{factor_id}/quality",
    summary="Composite Factor Quality Score (FQS) on a 0-100 scale",
    description=(
        "Return the 5 component scores (temporal, geographical, "
        "technological, representativeness, methodological) on both the "
        "native DQS 1-5 scale and the CTO-spec 0-100 scale, plus a "
        "weighted composite ``composite_fqs`` (0-100), a ``rating`` "
        "label (excellent / good / fair / poor) and a "
        "``promotion_eligibility`` label (certified / preview / "
        "connector_only).\n\n"
        "Public endpoint: no tier gate.  Exposes the same FQS the "
        "release-signoff pipeline uses so external users can filter "
        "and sort by quality consistently with the GreenLang release "
        "gate."
    ),
    responses={
        200: {"description": "Composite FQS payload"},
        304: {"description": "Not modified (ETag match)"},
        404: {"model": ErrorResponse, "description": "Factor not found"},
    },
)
async def factor_quality(
    request: Request,
    response: Response,
    factor_id: str,
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Return the composite FQS payload for a factor."""
    apply_rate_limit(request, response, current_user)

    from greenlang.factors.api_endpoints import (
        cache_control_for_status,
        check_etag_match,
        compute_etag,
    )
    from greenlang.factors.quality.composite_fqs import compute_fqs

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    factor = svc.repo.get_factor(edition_id, factor_id)
    if not factor:
        raise HTTPException(
            status_code=404,
            detail=f"Factor {factor_id!r} not found in edition {edition_id}",
        )

    dqs = getattr(factor, "dqs", None)
    if dqs is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Factor {factor_id!r} has no DataQualityScore; "
                "cannot compute FQS."
            ),
        )

    fqs = compute_fqs(dqs)
    payload: Dict[str, Any] = {
        "factor_id": factor_id,
        "edition_id": edition_id,
        "dqs_overall_5": getattr(dqs, "overall_score", None),
        "dqs_rating": getattr(
            getattr(dqs, "rating", None), "value", None
        ),
        "fqs": fqs.to_dict(),
    }

    # ETag + cache: quality is stable per (factor, edition).
    etag = compute_etag(factor)
    if_none_match = request.headers.get("If-None-Match")
    if check_etag_match(if_none_match, etag):
        return Response(status_code=304)

    factor_status = getattr(factor, "factor_status", "certified") or "certified"
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_status(factor_status)
    response.headers["X-GreenLang-Edition"] = edition_id
    response.headers["X-Factors-Edition"] = edition_id

    return payload


# ==================== GAP-5: Rollback endpoints (Enterprise+) ====================
#
# The rollback flow is a two-signature change-control gate on top of the
# append-only factor version chain.  All four endpoints share the same
# tier + auth pattern established for the explain endpoints.


def _require_enterprise_tier(current_user: dict) -> str:
    """Enforce Enterprise+ tier for the rollback plan/execute endpoints."""
    from greenlang.factors.tier_enforcement import Tier

    tier = _resolve_tier(current_user)
    allowed = {Tier.ENTERPRISE.value, Tier.INTERNAL.value}
    if tier not in allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                "Factor rollback plan/execute endpoints require Enterprise "
                "or Internal tier."
            ),
        )
    return tier


def _get_rollback_service():
    """Return the process-wide ``RollbackService`` instance.

    Lazily constructed from env-configured SQLite paths so route
    imports stay lightweight.  Production wiring (Postgres-backed store
    + real cascade lookup) replaces this via
    ``greenlang.factors.quality.rollback_wiring.get_rollback_service``
    once that module lands.
    """
    import os

    from greenlang.factors.quality.impact_simulator import ImpactSimulator
    from greenlang.factors.quality.rollback import (
        RollbackService,
        RollbackStore,
    )
    from greenlang.factors.quality.versioning import FactorVersionChain

    chain_path = os.environ.get(
        "GL_FACTORS_VERSION_CHAIN_PATH", "var/factors/version_chain.sqlite"
    )
    store_path = os.environ.get(
        "GL_FACTORS_ROLLBACK_STORE_PATH", "var/factors/rollback.sqlite"
    )
    chain = FactorVersionChain(chain_path)
    store = RollbackStore(store_path)
    # Impact simulator gets empty collections by default — the real ledger
    # wiring injects them via greenlang.factors.quality.rollback_wiring.
    simulator = ImpactSimulator(ledger_entries=[], evidence_records=[])
    return RollbackService(
        version_chain=chain,
        impact_simulator=simulator,
        store=store,
    )


@router.post(
    "/{factor_id}/rollback/plan",
    summary="Plan a factor rollback (impact preview)",
    description=(
        "Build a rollback plan against an existing factor version, "
        "including a full impact preview (affected tenants, computations, "
        "evidence bundles).  Required tier: **Enterprise** or **Internal**.\n\n"
        "The plan is persisted in PLANNED state; execution requires the "
        "two-signature approval gate (methodology_lead + compliance_lead) "
        "via POST /rollback/execute."
    ),
    responses={
        200: {"description": "RollbackPlan with impact preview"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Factor not found"},
    },
)
async def plan_factor_rollback(
    request: Request,
    response: Response,
    factor_id: str,
    body: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Plan a rollback; returns the impact preview."""
    apply_rate_limit(request, response, current_user)
    _require_enterprise_tier(current_user)

    from greenlang.factors.quality.rollback import RollbackError

    to_version = body.get("to_version")
    reason = body.get("reason")
    if not to_version or not reason:
        raise HTTPException(
            status_code=400,
            detail="Request body must include 'to_version' and 'reason'.",
        )

    service = _get_rollback_service()
    created_by = (
        current_user.get("user_id") if current_user else None
    ) or "anonymous"
    try:
        plan = service.plan_rollback(
            factor_id=factor_id,
            to_version=str(to_version),
            reason=str(reason),
            created_by=str(created_by),
            value_map=body.get("value_map"),
        )
    except RollbackError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return plan.to_dict()


@router.post(
    "/{factor_id}/rollback/execute",
    summary="Execute an approved factor rollback",
    description=(
        "Execute a previously-approved rollback.  Requires the "
        "two-signature approval gate (methodology_lead + compliance_lead). "
        "Returns 403 when fewer than two distinct signers have signed.  "
        "Required tier: **Enterprise** or **Internal**.\n\n"
        "Request body::\n\n"
        "    {\n"
        "      \"rollback_id\": \"...\",\n"
        "      \"approver_id\": \"user@company\",\n"
        "      \"approver_role\": \"methodology_lead\" | \"compliance_lead\",\n"
        "      \"approval_signature\": \"<sig>\"\n"
        "    }"
    ),
    responses={
        200: {"description": "RollbackRecord after execution"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        403: {"model": ErrorResponse, "description": "Missing approvals / tier"},
        404: {"model": ErrorResponse, "description": "Rollback not found"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def execute_factor_rollback(
    request: Request,
    response: Response,
    factor_id: str,
    body: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Add a signature (and execute once both signatures are present)."""
    apply_rate_limit(request, response, current_user)
    _require_enterprise_tier(current_user)

    from greenlang.factors.quality.rollback import (
        REQUIRED_APPROVAL_ROLES,
        RollbackApprovalError,
        RollbackError,
        RollbackNotFoundError,
        RollbackStateError,
        RollbackStatus,
    )

    rollback_id = body.get("rollback_id")
    approver_id = body.get("approver_id") or (
        current_user.get("user_id") if current_user else None
    )
    approver_role = body.get("approver_role")
    approval_signature = body.get("approval_signature")
    if not all([rollback_id, approver_id, approver_role, approval_signature]):
        raise HTTPException(
            status_code=400,
            detail=(
                "Request must include rollback_id, approver_id, approver_role "
                "and approval_signature."
            ),
        )

    service = _get_rollback_service()

    # Factor_id in path must match the persisted record.
    existing = service.get_rollback(str(rollback_id))
    if existing is None:
        raise HTTPException(
            status_code=404, detail=f"Rollback {rollback_id!r} not found"
        )
    if existing.factor_id != factor_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Rollback {rollback_id!r} belongs to factor "
                f"{existing.factor_id!r}, not {factor_id!r}."
            ),
        )

    try:
        record = service.approve_rollback(
            rollback_id=str(rollback_id),
            approver_id=str(approver_id),
            approver_role=str(approver_role),
            signature=str(approval_signature),
        )
    except RollbackApprovalError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except RollbackStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RollbackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # If we now have enough signatures, execute the rollback.
    if (
        record.status == RollbackStatus.APPROVED
        and len(record.approvals) >= len(REQUIRED_APPROVAL_ROLES)
    ):
        try:
            record = service.execute_rollback(rollback_id=str(rollback_id))
        except RollbackStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except RollbackError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    else:
        # Only one signature so far — return 202-like success payload with
        # a status=approved_pending signal for the client.
        pass

    return record.to_dict()


@router.get(
    "/{factor_id}/rollback/history",
    summary="List rollback records for a factor",
    description=(
        "Return the chronological list of rollback records for "
        "``factor_id`` (newest first).  Required tier: **Pro**+."
    ),
    responses={
        200: {"description": "Rollback history"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
    },
)
async def factor_rollback_history(
    request: Request,
    response: Response,
    factor_id: str,
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """List rollback records for a factor."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    service = _get_rollback_service()
    records = service.list_for_factor(factor_id)
    return {
        "factor_id": factor_id,
        "count": len(records),
        "rollbacks": [r.to_dict() for r in records],
    }


@router.get(
    "/rollback/{rollback_id}",
    summary="Get a single rollback record",
    description="Fetch one rollback record by ID.  Required tier: **Pro**+.",
    responses={
        200: {"description": "Rollback record"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Rollback not found"},
    },
)
async def get_rollback_record(
    request: Request,
    response: Response,
    rollback_id: str,
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Fetch a rollback record by ID."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    service = _get_rollback_service()
    record = service.get_rollback(rollback_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Rollback {rollback_id!r} not found"
        )
    return record.to_dict()


# ==================== GAP-6: Impact Simulation endpoints ====================


def _require_consulting_tier(current_user: dict) -> str:
    """Enforce Consulting+ tier for impact simulation endpoints."""
    from greenlang.factors.tier_enforcement import Tier

    tier = _resolve_tier(current_user)
    allowed = {
        Tier.CONSULTING.value,
        Tier.ENTERPRISE.value,
        Tier.INTERNAL.value,
    }
    if tier not in allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                "Impact simulation endpoints require Consulting, Enterprise, "
                "or Internal tier."
            ),
        )
    return tier


def _require_enterprise_tier_impact(current_user: dict) -> str:
    """Enforce Enterprise+ tier on the batch impact endpoint."""
    return _require_enterprise_tier(current_user)


def _load_impact_ledger() -> List[Dict[str, Any]]:
    """Best-effort load of the climate ledger for impact simulation.

    Uses ``GL_FACTORS_LEDGER_SQLITE`` when set; falls back to an empty
    list so the endpoint still responds (listing_only mode).
    """
    import os

    from greenlang.factors.quality.impact_simulator import (
        load_ledger_entries_from_sqlite,
    )

    path = os.environ.get("GL_FACTORS_LEDGER_SQLITE")
    if not path:
        return []
    try:
        return load_ledger_entries_from_sqlite(path)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.warning("Impact simulator: ledger load failed: %s", exc)
        return []


@router.post(
    "/impact-simulation",
    summary="Simulate the impact of a factor change",
    description=(
        "Preview every computation and tenant affected by a hypothetical "
        "factor change.  Supports three modes:\n\n"
        "1. ``hypothetical_value: {co2e_total: 1.23}`` — explicit new value\n"
        "2. ``hypothetical_value: \"deprecation\"`` — flag every consumer\n"
        "3. ``hypothetical_value: null`` — listing only (no numeric delta)\n\n"
        "Required tier: **Consulting**+."
    ),
    responses={
        200: {"description": "ImpactReport with affected computations + deltas"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
    },
)
async def impact_simulation(
    request: Request,
    response: Response,
    body: Dict[str, Any],
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Run an impact simulation for a single hypothetical factor change."""
    apply_rate_limit(request, response, current_user)
    _require_consulting_tier(current_user)

    from greenlang.factors.api_endpoints import (
        build_impact_simulation,
        cache_control_for_impact,
    )

    factor_id = body.get("factor_id")
    if not factor_id:
        raise HTTPException(
            status_code=400, detail="factor_id is required"
        )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    tenant_scope = body.get("tenant_scope")
    payload = build_impact_simulation(
        svc.repo,
        factor_id=str(factor_id),
        hypothetical_value=body.get("hypothetical_value"),
        tenant_scope=list(tenant_scope) if tenant_scope else None,
        edition_id=edition_id,
        ledger_entries=_load_impact_ledger(),
    )
    response.headers["Cache-Control"] = cache_control_for_impact()
    response.headers["X-Factors-Edition"] = edition_id
    return payload


@router.post(
    "/impact-simulation/batch",
    summary="Simulate the impact of multiple factor changes at once",
    description=(
        "Accepts a list of factor-change items and returns a combined "
        "report plus per-factor subreports.  Required tier: "
        "**Enterprise**+."
    ),
    responses={
        200: {"description": "Aggregated impact report"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
    },
)
async def impact_simulation_batch(
    request: Request,
    response: Response,
    body: Dict[str, Any],
    edition: Optional[str] = Query(None, description="Pin to specific edition"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Run impact simulations for multiple factor changes simultaneously."""
    apply_rate_limit(request, response, current_user)
    _require_enterprise_tier_impact(current_user)

    from greenlang.factors.api_endpoints import (
        build_impact_simulation_batch,
        cache_control_for_impact,
    )

    items = body.get("items") or []
    if not isinstance(items, list) or not items:
        raise HTTPException(
            status_code=400,
            detail="Request body must include a non-empty 'items' array.",
        )
    # Hard cap to avoid abuse even for Enterprise.
    if len(items) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch impact simulation is capped at 100 items per request.",
        )

    try:
        edition_id = _resolve_edition(svc, request, edition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload = build_impact_simulation_batch(
        svc.repo,
        items=items,
        edition_id=edition_id,
        ledger_entries=_load_impact_ledger(),
    )
    response.headers["Cache-Control"] = cache_control_for_impact()
    response.headers["X-Factors-Edition"] = edition_id
    return payload


@router.get(
    "/{factor_id}/dependent-computations",
    summary="List computations that depend on this factor",
    description=(
        "Return every computation in the climate ledger that consumed "
        "``factor_id``.  Helper for the impact-simulation UI.  "
        "Required tier: **Pro**+."
    ),
    responses={
        200: {"description": "Dependent computations list"},
        304: {"description": "Not modified (ETag match)"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
    },
)
async def factor_dependent_computations(
    request: Request,
    response: Response,
    factor_id: str,
    tenant_scope: Optional[str] = Query(
        None,
        description="Comma-separated list of tenant IDs to restrict the scan.",
    ),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """List computations that depend on a factor."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.api_endpoints import (
        cache_control_for_impact,
        check_etag_match,
        compute_impact_etag,
        list_dependent_computations,
    )

    ts_list: Optional[List[str]] = None
    if tenant_scope:
        ts_list = [t.strip() for t in tenant_scope.split(",") if t.strip()]

    ledger = _load_impact_ledger()
    deps = list_dependent_computations(
        factor_id=factor_id,
        ledger_entries=ledger,
        tenant_scope=ts_list,
    )
    payload = {
        "factor_id": factor_id,
        "tenant_scope": ts_list,
        "count": len(deps),
        "computations": deps,
    }

    etag = compute_impact_etag(payload)
    if_none_match = request.headers.get("If-None-Match")
    if check_etag_match(if_none_match, etag):
        return Response(status_code=304)

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control_for_impact()
    return payload


# ==================== GAP-11: Batch Job endpoints (Pro+) ====================


@router.post(
    "/batch/submit",
    summary="Submit a batch resolution job",
    description=(
        "Queue a batch resolution job and return a handle the caller "
        "polls via GET /batch/{job_id}.  Rate limit per batch: "
        "Pro=100, Consulting=1000, Enterprise=10000.  "
        "Required tier: **Pro**+."
    ),
    responses={
        202: {"description": "Batch job accepted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        413: {"model": ErrorResponse, "description": "Batch exceeds tier cap"},
    },
    status_code=202,
)
async def submit_batch_job(
    request: Request,
    response: Response,
    body: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Queue a batch resolution job."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.batch_jobs import (
        BatchJobError,
        BatchJobLimitError,
        BatchJobType,
        get_default_queue,
        submit_batch_resolution,
    )

    requests_payload = body.get("requests") or []
    if not isinstance(requests_payload, list) or not requests_payload:
        raise HTTPException(
            status_code=400,
            detail="Request body must include a non-empty 'requests' array.",
        )
    job_type = body.get("job_type", "resolve")
    try:
        job_type_enum = BatchJobType(job_type)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Unknown job_type {job_type!r}"
        )

    tenant_id = (current_user or {}).get("tenant_id") or "default"
    tier = _resolve_tier(current_user)
    created_by = (current_user or {}).get("user_id") or "anonymous"

    queue = get_default_queue()
    try:
        handle = submit_batch_resolution(
            queue,
            requests=list(requests_payload),
            tenant_id=str(tenant_id),
            tier=tier,
            created_by=str(created_by),
            job_type=job_type_enum,
            webhook_url=body.get("webhook_url"),
            webhook_secret_ref=body.get("webhook_secret_ref"),
        )
    except BatchJobLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except BatchJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return handle.to_dict()


@router.get(
    "/batch",
    summary="List batch jobs for the current tenant",
    description="List batch jobs submitted by the caller's tenant.  Required tier: **Pro**+.",
    responses={
        200: {"description": "Batch job list"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
    },
)
async def list_batch_jobs(
    request: Request,
    response: Response,
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """List batch jobs for the caller's tenant."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.batch_jobs import BatchJobStatus, get_default_queue

    tenant_id = (current_user or {}).get("tenant_id") or "default"
    queue = get_default_queue()

    status_enum: Optional[BatchJobStatus] = None
    if status_filter:
        try:
            status_enum = BatchJobStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Unknown status {status_filter!r}"
            )

    jobs, total = queue.list_for_tenant(
        str(tenant_id), status=status_enum, limit=limit, offset=offset
    )
    return {
        "tenant_id": tenant_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "jobs": [j.to_dict() for j in jobs],
    }


@router.get(
    "/batch/{job_id}",
    summary="Get batch job status",
    description="Return current status + progress for a batch job.  Required tier: **Pro**+.",
    responses={
        200: {"description": "Batch job status"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_batch_job(
    request: Request,
    response: Response,
    job_id: str,
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Return current status for a batch job."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.batch_jobs import (
        BatchJobNotFoundError,
        get_batch_job_status,
        get_default_queue,
    )

    queue = get_default_queue()
    try:
        job = get_batch_job_status(queue, job_id)
    except BatchJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Tenant isolation — don't let tenant A peek at tenant B's jobs.
    tenant_id = (current_user or {}).get("tenant_id") or "default"
    if job.tenant_id != tenant_id and _resolve_tier(current_user) != "internal":
        raise HTTPException(status_code=404, detail="Job not found")

    return job.to_dict()


@router.get(
    "/batch/{job_id}/results",
    summary="Download batch job results (paginated)",
    description=(
        "Stream batch job results with cursor-based pagination.  Required "
        "tier: **Pro**+."
    ),
    responses={
        200: {"description": "Batch job results page"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_batch_job_results_endpoint(
    request: Request,
    response: Response,
    job_id: str,
    cursor: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10_000),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Return a paginated results page for a batch job."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.batch_jobs import (
        BatchJobNotFoundError,
        get_batch_job_results,
        get_batch_job_status,
        get_default_queue,
    )

    queue = get_default_queue()
    try:
        job = get_batch_job_status(queue, job_id)
    except BatchJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    tenant_id = (current_user or {}).get("tenant_id") or "default"
    if job.tenant_id != tenant_id and _resolve_tier(current_user) != "internal":
        raise HTTPException(status_code=404, detail="Job not found")

    return get_batch_job_results(queue, job_id, cursor=cursor, limit=limit)


@router.post(
    "/batch/{job_id}/cancel",
    summary="Cancel a queued or running batch job",
    description="Cancel a QUEUED or RUNNING job.  Required tier: **Pro**+.",
    responses={
        200: {"description": "Cancelled job record"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job not cancellable"},
    },
)
async def cancel_batch_job_endpoint(
    request: Request,
    response: Response,
    job_id: str,
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Cancel a queued or running batch job."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.batch_jobs import (
        BatchJobNotFoundError,
        BatchJobStateError,
        cancel_batch_job,
        get_batch_job_status,
        get_default_queue,
    )

    queue = get_default_queue()
    try:
        job = get_batch_job_status(queue, job_id)
    except BatchJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    tenant_id = (current_user or {}).get("tenant_id") or "default"
    if job.tenant_id != tenant_id and _resolve_tier(current_user) != "internal":
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        job = cancel_batch_job(queue, job_id)
    except BatchJobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return job.to_dict()


@router.delete(
    "/batch/{job_id}",
    summary="Delete a completed batch job",
    description=(
        "Delete a COMPLETED / FAILED / CANCELLED batch job and its stored "
        "results.  Required tier: **Pro**+."
    ),
    responses={
        200: {"description": "Deletion confirmation"},
        403: {"model": ErrorResponse, "description": "Tier insufficient"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job still running"},
    },
)
async def delete_batch_job_endpoint(
    request: Request,
    response: Response,
    job_id: str,
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> Dict[str, Any]:
    """Delete a completed batch job."""
    apply_rate_limit(request, response, current_user)
    _require_pro_tier(current_user)

    from greenlang.factors.batch_jobs import (
        BatchJobNotFoundError,
        BatchJobStateError,
        delete_batch_job,
        get_batch_job_status,
        get_default_queue,
    )

    queue = get_default_queue()
    try:
        job = get_batch_job_status(queue, job_id)
    except BatchJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    tenant_id = (current_user or {}).get("tenant_id") or "default"
    if job.tenant_id != tenant_id and _resolve_tier(current_user) != "internal":
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        deleted = delete_batch_job(queue, job_id)
    except BatchJobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"job_id": job_id, "deleted": bool(deleted)}


__all__ = ["router"]
