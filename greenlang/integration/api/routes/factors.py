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
        "co2_per_unit": f.gwp_100yr.CO2,
        "ch4_per_unit": f.gwp_100yr.CH4,
        "n2o_per_unit": f.gwp_100yr.N2O,
        "co2e_per_unit": f.gwp_100yr.co2e_total,
        "gwp_set": f.gwp_set.value if hasattr(f.gwp_set, "value") else str(f.gwp_set),
        "ch4_gwp": getattr(f.gwp_100yr, "ch4_gwp", 28),
        "n2o_gwp": getattr(f.gwp_100yr, "n2o_gwp", 273),
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


__all__ = ["router"]
