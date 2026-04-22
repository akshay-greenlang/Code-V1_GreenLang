# -*- coding: utf-8 -*-
"""
greenlang/api/main.py

Production-grade FastAPI application for emission factor queries and calculations.

Performance requirements:
- <50ms response time (95th percentile)
- 1000 requests/second capacity
- 99.9% uptime
- Horizontal scaling support
"""

import hashlib
import json as jsonlib

from fastapi import FastAPI, HTTPException, Query, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional, List
from datetime import datetime, date
import logging
import time
import uuid
import asyncio

from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.data.emission_factor_record import EmissionFactorRecord
from greenlang.utilities.determinism import deterministic_uuid, DeterministicClock
from .models import (
    CalculationRequest,
    CalculationResponse,
    BatchCalculationRequest,
    BatchCalculationResponse,
    Scope1Request,
    Scope2Request,
    Scope3Request,
    EmissionResult,
    EmissionFactorResponse,
    EmissionFactorSummary,
    FactorListResponse,
    FactorSearchResponse,
    FactorSearchFacetsResponse,
    FactorProvenanceResponse,
    EditionSummary,
    EditionListResponse,
    EditionChangelogResponse,
    EditionCompareResponse,
    FactorReplacementsResponse,
    FactorMatchRequest,
    FactorMatchResponse,
    FactorMatchCandidate,
    SourceRegistryListResponse,
    SourceRegistryEntryResponse,
    StatsResponse,
    CoverageStats,
    HealthResponse,
    ErrorResponse,
    GHGBreakdown,
    DataQuality,
    SourceInfo,
    CacheStats,
)
from greenlang.factors.metering import GLOBAL_METER
from greenlang.factors.service import FactorCatalogService, resolve_edition_id

# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== APPLICATION INITIALIZATION ====================

app = FastAPI(
    title="GreenLang Emission Factor API",
    description="""
    Production REST API for emission factor queries and calculations.

    ## Features

    - 327+ emission factors (US, EU, UK, and more)
    - Multi-gas breakdown (CO2, CH4, N2O)
    - Full provenance tracking
    - Data quality scoring
    - Scope 1, 2, 3 calculations
    - Batch processing
    - Historical queries
    - Rate limiting and caching

    ## Performance

    - <50ms response time (95th percentile)
    - 1000 requests/second capacity
    - 99.9% uptime guarantee
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    contact={
        "name": "GreenLang Support",
        "email": "support@greenlang.io",
        "url": "https://greenlang.io"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    }
)

# ==================== MIDDLEWARE ====================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.greenlang.io",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"]
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.greenlang.io", "localhost", "127.0.0.1", "testserver"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer(auto_error=False)

# ==================== FACTORS — SIGNED RECEIPTS MIDDLEWARE ====================
# Install on the /api/v1/factors prefix so every successful response carries
# a signed receipt + edition pin.  The auth/metering path in this file
# already populates request.state.tier via get_current_user; the signing
# middleware reads that for algorithm selection (HMAC for community/pro,
# Ed25519 for consulting/platform/enterprise with graceful fallback).
#
# Must install AFTER the CORSMiddleware / TrustedHostMiddleware above so
# the receipt covers the final response bytes the client will see.
try:
    from greenlang.factors.middleware.signed_receipts import (
        install_signing_middleware,
    )

    install_signing_middleware(app, protected_prefix="/api/v1/factors")
    logger.info("Signed-receipts middleware installed on /api/v1/factors")
except Exception as _sign_exc:  # noqa: BLE001
    # Never block app startup on the signing middleware — sign every
    # response we can, but if the wiring itself fails for an unexpected
    # reason (import error, etc.) log and continue.
    logger.warning(
        "Signed-receipts middleware NOT installed: %s", _sign_exc
    )

# ==================== GLOBAL STATE ====================

# Emission factor database (initialized on startup)
emission_db: Optional[EmissionFactorDatabase] = None
catalog_service: Optional[FactorCatalogService] = None

# Application metrics
app_start_time = time.time()
calculation_counter = 0

# ==================== STARTUP / SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global emission_db, catalog_service

    logger.info("Starting GreenLang Emission Factor API v1.0.0")

    # Initialize emission factor database with caching
    emission_db = EmissionFactorDatabase(
        enable_cache=True,
        cache_size=1000,
        cache_ttl=3600  # 1 hour
    )
    catalog_service = FactorCatalogService.from_environment(emission_db)

    logger.info("Loaded %s emission factors", len(emission_db.factors))
    logger.info("Cache enabled: %s", emission_db.enable_cache)
    logger.info(
        "Factor catalog backend: %s",
        type(catalog_service.repo).__name__,
    )
    logger.info("API ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down GreenLang Emission Factor API")

    # Clear cache
    if emission_db:
        emission_db.clear_cache()

    logger.info("Shutdown complete")


# ==================== AUTHENTICATION ====================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Validate JWT token or API key.

    Implements basic token validation with environment-aware security:
    - Development/staging: Allows anonymous access with warning
    - Production: Requires valid JWT or API key

    Returns:
        dict: User context with user_id and tenant_id
    """
    import os
    import jwt

    env = os.getenv("GL_ENV", "development")
    jwt_secret = os.getenv("GL_JWT_SECRET", "")

    # Production requires proper authentication
    if env == "production" and not jwt_secret:
        logger.error("SECURITY: GL_JWT_SECRET not configured in production")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured"
        )

    # Extract and validate token
    if credentials and credentials.credentials:
        token = credentials.credentials

        # Check for API key format (gl_*)
        if token.startswith("gl_"):
            # API key validation - check prefix and minimum length
            if len(token) >= 32:
                allowed = {
                    x.strip()
                    for x in os.getenv("GL_API_KEYS", "").split(",")
                    if x.strip()
                }
                if allowed and token not in allowed:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Unknown API key",
                    )
                logger.info("API key authentication: prefix=%s...", token[:8])
                tier = os.getenv("GL_FACTORS_TIER", "community")
                return {
                    "user_id": f"apikey:{token[:12]}",
                    "tenant_id": "default",
                    "tier": tier,
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key format"
                )

        # JWT validation
        if jwt_secret:
            try:
                payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
                return {
                    "user_id": payload.get("sub", "unknown"),
                    "tenant_id": payload.get("tenant_id", "default"),
                    "tier": payload.get("tier", os.getenv("GL_FACTORS_TIER", "community")),
                }
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            except jwt.InvalidTokenError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token: {e}"
                )

    # Development/staging: allow anonymous with warning
    if env in ("development", "staging", "test"):
        logger.warning("SECURITY: Anonymous access allowed in non-production environment")
        return {
            "user_id": "anonymous",
            "tenant_id": "default",
            "tier": os.getenv("GL_FACTORS_TIER", "community"),
        }

    # Production without valid credentials
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )


# ==================== MIDDLEWARE FOR REQUEST TRACKING ====================

@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    """Add request ID and response time tracking"""
    request_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000  # Convert to ms

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{process_time:.2f}ms"
    GLOBAL_METER.record(request.url.path)
    try:
        from greenlang.factors.billing.usage_sink import record_path_hit

        record_path_hit(
            str(request.url.path),
            None,
            request.headers.get("x-gl-tier") or request.headers.get("X-GL-Tier"),
        )
    except Exception:
        pass

    return response


# ==================== FACTOR QUERY ENDPOINTS ====================

def _factor_catalog_service() -> FactorCatalogService:
    """Return catalog service; lazily wire if startup did not run (e.g. partial tests)."""
    global catalog_service, emission_db
    if catalog_service is None:
        if emission_db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Factor catalog not initialized",
            )
        catalog_service = FactorCatalogService.from_environment(emission_db)
    return catalog_service


@app.get(
    "/api/v1/editions",
    response_model=EditionListResponse,
    summary="List factor catalog editions",
    tags=["Factors", "Editions"],
)
@limiter.limit("200/minute")
async def list_editions(
    request: Request,
    include_pending: bool = Query(True),
    current_user: dict = Depends(get_current_user),
) -> EditionListResponse:
    svc = _factor_catalog_service()
    rows = svc.repo.list_editions(include_pending=include_pending)
    editions = [
        EditionSummary(
            edition_id=r.edition_id,
            status=r.status,
            label=r.label,
            manifest_hash=r.manifest_hash,
        )
        for r in rows
    ]
    return EditionListResponse(
        editions=editions,
        default_edition_id=svc.repo.get_default_edition_id(),
    )


@app.get(
    "/api/v1/editions/{edition_id}/changelog",
    response_model=EditionChangelogResponse,
    summary="Changelog for a catalog edition",
    tags=["Factors", "Editions"],
)
@limiter.limit("200/minute")
async def edition_changelog(
    request: Request,
    edition_id: str,
    current_user: dict = Depends(get_current_user),
) -> EditionChangelogResponse:
    svc = _factor_catalog_service()
    return EditionChangelogResponse(
        edition_id=edition_id,
        changelog=svc.repo.get_changelog(edition_id),
    )


@app.get(
    "/api/v1/editions/compare",
    response_model=EditionCompareResponse,
    summary="Compare two catalog editions",
    tags=["Factors", "Editions"],
)
@limiter.limit("60/minute")
async def compare_editions(
    request: Request,
    left: str = Query(..., description="Left edition id"),
    right: str = Query(..., description="Right edition id"),
    current_user: dict = Depends(get_current_user),
) -> EditionCompareResponse:
    svc = _factor_catalog_service()
    diff = svc.compare_editions(left, right)
    return EditionCompareResponse(**diff)


@app.get(
    "/api/v1/factors/source-registry",
    response_model=SourceRegistryListResponse,
    summary="CTO source registry (rights + cadence metadata)",
    tags=["Factors"],
)
@limiter.limit("120/minute")
async def list_source_registry(
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> SourceRegistryListResponse:
    from greenlang.factors.source_registry import load_source_registry

    rows = load_source_registry()
    return SourceRegistryListResponse(
        sources=[
            SourceRegistryEntryResponse(
                source_id=e.source_id,
                display_name=e.display_name,
                connector_only=e.connector_only,
                license_class=e.license_class,
                cadence=e.cadence,
                watch_mechanism=e.watch_mechanism,
                watch_url=e.watch_url,
                watch_file_type=e.watch_file_type,
                redistribution_allowed=e.redistribution_allowed,
                derivative_works_allowed=e.derivative_works_allowed,
                commercial_use_allowed=e.commercial_use_allowed,
                attribution_required=e.attribution_required,
                citation_text=e.citation_text,
                approval_required_for_certified=e.approval_required_for_certified,
                legal_signoff_artifact=e.legal_signoff_artifact,
                legal_signoff_version=e.legal_signoff_version,
                public_bulk_export_allowed=e.public_bulk_export_allowed(),
            )
            for e in rows
        ]
    )


@app.get(
    "/api/v1/factors",
    response_model=FactorListResponse,
    summary="List emission factors",
    description="Retrieve paginated list of emission factors with optional filters",
    tags=["Factors"],
    responses={
        200: {"description": "Successfully retrieved factors"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        429: {"description": "Rate limit exceeded"}
    }
)
@limiter.limit("1000/minute")
async def list_factors(
    request: Request,
    response: Response,
    fuel_type: Optional[str] = Query(None, description="Filter by fuel type"),
    geography: Optional[str] = Query(None, description="Filter by geography"),
    scope: Optional[str] = Query(None, description="Filter by scope (1, 2, or 3)"),
    boundary: Optional[str] = Query(None, description="Filter by boundary"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(100, ge=1, le=500, description="Items per page (max 500)"),
    include_preview: bool = Query(False, description="Include preview-status factors"),
    include_connector: bool = Query(
        False,
        description="Include connector_only rows (enterprise / licensed use)",
    ),
    edition: Optional[str] = Query(
        None,
        description="Catalog edition id (overridden by X-Factors-Edition header when set)",
    ),
    current_user: dict = Depends(get_current_user),
) -> FactorListResponse:
    """
    List emission factors with filtering and pagination.

    Returns paginated list of factors matching the specified filters.
    """
    try:
        svc = _factor_catalog_service()
        hdr = request.headers.get("x-factors-edition") or request.headers.get(
            "X-Factors-Edition"
        )
        try:
            edition_id, _src = resolve_edition_id(svc.repo, hdr, edition)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        tier = (current_user or {}).get("tier", "community")
        if include_connector and tier not in ("enterprise", "internal"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="include_connector requires enterprise tier",
            )

        page_factors, total_count = svc.repo.list_factors(
            edition_id,
            fuel_type=fuel_type,
            geography=geography,
            scope=scope,
            boundary=boundary,
            page=page,
            limit=limit,
            include_preview=include_preview,
            include_connector=include_connector,
        )
        total_pages = (total_count + limit - 1) // limit if total_count else 0
        factor_summaries = [_factor_to_summary(factor) for factor in page_factors]

        response.headers["X-Factors-Edition"] = edition_id
        return FactorListResponse(
            factors=factor_summaries,
            total_count=total_count,
            page=page,
            page_size=len(factor_summaries),
            total_pages=total_pages,
            edition_id=edition_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error retrieving factors"
        )


@app.get(
    "/api/v1/factors/search",
    response_model=FactorSearchResponse,
    summary="Search emission factors",
    description="Full-text search across emission factors",
    tags=["Factors"],
    responses={
        200: {"description": "Search results"},
        400: {"model": ErrorResponse, "description": "Invalid search query"}
    }
)
@limiter.limit("500/minute")
async def search_factors(
    request: Request,
    response: Response,
    q: str = Query(..., min_length=2, description="Search query"),
    geography: Optional[str] = Query(None, description="Filter by geography"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    include_preview: bool = Query(False),
    include_connector: bool = Query(False),
    factor_status: Optional[str] = Query(None, description="Exact factor_status filter"),
    source_id: Optional[str] = Query(None, description="Filter by source_id column when set"),
    edition: Optional[str] = Query(None, description="Catalog edition id"),
    current_user: dict = Depends(get_current_user),
) -> FactorSearchResponse:
    """
    Search emission factors by text query.

    Searches across fuel_type, geography, tags, and notes fields.
    """
    try:
        start_time = time.time()
        svc = _factor_catalog_service()
        hdr = request.headers.get("x-factors-edition") or request.headers.get(
            "X-Factors-Edition"
        )
        try:
            edition_id, _src = resolve_edition_id(svc.repo, hdr, edition)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        tier = (current_user or {}).get("tier", "community")
        if include_connector and tier not in ("enterprise", "internal"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="include_connector requires enterprise tier",
            )

        results = svc.repo.search_factors(
            edition_id,
            query=q,
            geography=geography,
            limit=limit,
            include_preview=include_preview,
            include_connector=include_connector,
            factor_status=factor_status,
            source_id=source_id,
        )
        search_time_ms = (time.time() - start_time) * 1000
        response.headers["X-Factors-Edition"] = edition_id
        return FactorSearchResponse(
            query=q,
            factors=[_factor_to_summary(f) for f in results],
            count=len(results),
            search_time_ms=search_time_ms,
            edition_id=edition_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error searching factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error searching factors"
        )


@app.get(
    "/api/v1/factors/search/v2",
    response_model=FactorSearchResponse,
    summary="Faceted search (A1): v2 filters on top of lexical search",
    tags=["Factors"],
    responses={
        200: {"description": "Search results"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
    },
)
@limiter.limit("500/minute")
async def search_factors_v2(
    request: Request,
    response: Response,
    q: str = Query(..., min_length=2, description="Search query"),
    geography: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    include_preview: bool = Query(False),
    include_connector: bool = Query(False),
    factor_status: Optional[str] = Query(None),
    source_id: Optional[str] = Query(None),
    unit: Optional[str] = Query(None, description="Filter by activity unit denominator"),
    scope: Optional[str] = Query(None, description="Filter by GHG scope 1/2/3"),
    methodology: Optional[str] = Query(None, description="Substring match on provenance methodology"),
    valid_year: Optional[int] = Query(None, ge=1990, le=2050, description="Match valid_from year"),
    edition: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
) -> FactorSearchResponse:
    """Search v2: same lexical core as /search with optional post-filters (M2)."""
    try:
        start_time = time.time()
        svc = _factor_catalog_service()
        hdr = request.headers.get("x-factors-edition") or request.headers.get("X-Factors-Edition")
        try:
            edition_id, _src = resolve_edition_id(svc.repo, hdr, edition)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        tier = (current_user or {}).get("tier", "community")
        if include_connector and tier not in ("enterprise", "internal"):
            raise HTTPException(status_code=403, detail="include_connector requires enterprise tier")
        wide = min(500, max(limit * 15, limit))
        results = svc.repo.search_factors(
            edition_id,
            query=q,
            geography=geography,
            limit=wide,
            include_preview=include_preview,
            include_connector=include_connector,
            factor_status=factor_status,
            source_id=source_id,
        )
        filtered = []
        for f in results:
            if unit and f.unit.lower() != unit.lower():
                continue
            if scope and f.scope.value != scope:
                continue
            if methodology and methodology.lower() not in (f.provenance.methodology.value or "").lower():
                continue
            if valid_year is not None and f.valid_from.year != valid_year:
                continue
            filtered.append(f)
            if len(filtered) >= limit:
                break
        search_time_ms = (time.time() - start_time) * 1000
        response.headers["X-Factors-Edition"] = edition_id
        return FactorSearchResponse(
            query=q,
            factors=[_factor_to_summary(f) for f in filtered],
            count=len(filtered),
            search_time_ms=search_time_ms,
            edition_id=edition_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in search v2: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error searching factors",
        )


@app.get(
    "/api/v1/factors/search/facets",
    response_model=FactorSearchFacetsResponse,
    summary="Facet counts for factor filters (M2)",
    tags=["Factors"],
)
@limiter.limit("240/minute")
async def search_factor_facets(
    request: Request,
    include_preview: bool = Query(False),
    include_connector: bool = Query(False),
    max_values: int = Query(80, ge=10, le=200),
    edition: Optional[str] = Query(None, description="Catalog edition id"),
    current_user: dict = Depends(get_current_user),
) -> FactorSearchFacetsResponse:
    svc = _factor_catalog_service()
    hdr = request.headers.get("x-factors-edition") or request.headers.get("X-Factors-Edition")
    try:
        edition_id, _ = resolve_edition_id(svc.repo, hdr, edition)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tier = (current_user or {}).get("tier", "community")
    if include_connector and tier not in ("enterprise", "internal"):
        raise HTTPException(
            status_code=403,
            detail="include_connector requires enterprise tier",
        )
    raw = svc.repo.search_facets(
        edition_id,
        include_preview=include_preview,
        include_connector=include_connector,
        max_values=max_values,
    )
    return FactorSearchFacetsResponse(
        edition_id=str(raw.get("edition_id") or edition_id),
        facets=dict(raw.get("facets") or {}),
    )


@app.post(
    "/api/v1/factors/match",
    response_model=FactorMatchResponse,
    summary="Hybrid activity → factor match (deterministic core)",
    tags=["Factors"],
)
@limiter.limit("120/minute")
async def match_factors(
    request: Request,
    response: Response,
    body: FactorMatchRequest,
    edition: Optional[str] = Query(None),
    include_preview: bool = Query(False),
    include_connector: bool = Query(False),
    current_user: dict = Depends(get_current_user),
) -> FactorMatchResponse:
    from greenlang.factors.matching.pipeline import MatchRequest, run_match

    svc = _factor_catalog_service()
    hdr = request.headers.get("x-factors-edition") or request.headers.get("X-Factors-Edition")
    try:
        edition_id, _ = resolve_edition_id(svc.repo, hdr, edition)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tier = (current_user or {}).get("tier", "community")
    if include_connector and tier not in ("enterprise", "internal"):
        raise HTTPException(status_code=403, detail="include_connector requires enterprise tier")
    mreq = MatchRequest(
        activity_description=body.activity_description,
        geography=body.geography,
        fuel_type=body.fuel_type,
        scope=body.scope,
        limit=body.limit,
    )
    raw = run_match(
        svc.repo,
        edition_id,
        mreq,
        include_preview=include_preview,
        include_connector=include_connector,
    )
    response.headers["X-Factors-Edition"] = edition_id
    return FactorMatchResponse(
        edition_id=edition_id,
        candidates=[FactorMatchCandidate(**c) for c in raw],
    )


@app.get(
    "/api/v1/factors/{factor_id}/replacements",
    response_model=FactorReplacementsResponse,
    summary="Deprecation replacement chain",
    tags=["Factors"],
)
@limiter.limit("200/minute")
async def factor_replacements(
    request: Request,
    response: Response,
    factor_id: str,
    edition: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
) -> FactorReplacementsResponse:
    svc = _factor_catalog_service()
    hdr = request.headers.get("x-factors-edition") or request.headers.get("X-Factors-Edition")
    try:
        edition_id, _ = resolve_edition_id(svc.repo, hdr, edition)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    chain = svc.replacement_chain(edition_id, factor_id)
    response.headers["X-Factors-Edition"] = edition_id
    return FactorReplacementsResponse(
        edition_id=edition_id,
        seed_factor_id=factor_id,
        chain=chain,
    )


@app.get(
    "/api/v1/factors/{factor_id}/provenance",
    response_model=FactorProvenanceResponse,
    summary="Provenance and license for a factor",
    tags=["Factors"],
)
@limiter.limit("500/minute")
async def get_factor_provenance(
    request: Request,
    response: Response,
    factor_id: str,
    edition: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
) -> FactorProvenanceResponse:
    svc = _factor_catalog_service()
    hdr = request.headers.get("x-factors-edition") or request.headers.get(
        "X-Factors-Edition"
    )
    try:
        edition_id, _ = resolve_edition_id(svc.repo, hdr, edition)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    factor = svc.repo.get_factor(edition_id, factor_id)
    if not factor:
        raise HTTPException(status_code=404, detail=f"Factor '{factor_id}' not found")
    response.headers["X-Factors-Edition"] = edition_id
    lic = factor.license_info
    return FactorProvenanceResponse(
        factor_id=factor.factor_id,
        content_hash=factor.content_hash,
        provenance={
            "source_org": factor.provenance.source_org,
            "source_publication": factor.provenance.source_publication,
            "source_year": str(factor.provenance.source_year),
            "methodology": factor.provenance.methodology.value,
            "version": factor.provenance.version,
            "citation": factor.provenance.citation,
        },
        license_info={
            "license": lic.license,
            "redistribution_allowed": lic.redistribution_allowed,
            "commercial_use_allowed": lic.commercial_use_allowed,
            "attribution_required": lic.attribution_required,
        },
        edition_id=edition_id,
    )


@app.get(
    "/api/v1/factors/{factor_id}",
    response_model=EmissionFactorResponse,
    summary="Get emission factor by ID",
    description="Retrieve detailed information for a specific emission factor",
    tags=["Factors"],
    responses={
        200: {"description": "Successfully retrieved factor"},
        404: {"model": ErrorResponse, "description": "Factor not found"},
        429: {"description": "Rate limit exceeded"}
    }
)
@limiter.limit("1000/minute")
async def get_factor(
    request: Request,
    response: Response,
    factor_id: str,
    edition: Optional[str] = Query(None, description="Catalog edition id"),
    current_user: dict = Depends(get_current_user),
) -> EmissionFactorResponse:
    """
    Get detailed emission factor by ID.

    Returns complete factor record with all metadata.
    """
    try:
        svc = _factor_catalog_service()
        hdr = request.headers.get("x-factors-edition") or request.headers.get(
            "X-Factors-Edition"
        )
        try:
            edition_id, _src = resolve_edition_id(svc.repo, hdr, edition)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        factor = svc.repo.get_factor(edition_id, factor_id)
        if not factor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Factor '{factor_id}' not found",
            )

        response.headers["X-Factors-Edition"] = edition_id
        resp_model = _factor_to_response(factor)
        dumped = (
            resp_model.model_dump()
            if hasattr(resp_model, "model_dump")
            else resp_model.dict()
        )
        payload = jsonlib.dumps(dumped, sort_keys=True, default=str)
        response.headers["ETag"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return resp_model

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving factor %s: %s", factor_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error retrieving factor"
        )


@app.get(
    "/api/v1/system/factors-metering",
    summary="In-process request counters for Factors routes (ops/debug)",
    tags=["System"],
)
@limiter.limit("60/minute")
async def factors_metering_snapshot(
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> dict:
    return {"paths": GLOBAL_METER.snapshot()}


@app.get(
    "/api/v1/factors/category/{fuel_type}",
    response_model=FactorListResponse,
    summary="Get factors by fuel type",
    description="Retrieve all factors for a specific fuel type",
    tags=["Factors"]
)
@limiter.limit("1000/minute")
async def get_by_fuel_type(
    request: Request,
    response: Response,
    fuel_type: str,
    geography: Optional[str] = Query(None),
    edition: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
) -> FactorListResponse:
    """Get all factors for a specific fuel type"""
    return await list_factors(
        request=request,
        response=response,
        fuel_type=fuel_type,
        geography=geography,
        scope=None,
        boundary=None,
        page=1,
        limit=100,
        include_preview=False,
        include_connector=False,
        edition=edition,
        current_user=current_user,
    )


@app.get(
    "/api/v1/factors/scope/{scope}",
    response_model=FactorListResponse,
    summary="Get factors by scope",
    description="Retrieve all factors for a specific GHG scope",
    tags=["Factors"]
)
@limiter.limit("1000/minute")
async def get_by_scope(
    request: Request,
    response: Response,
    scope: str,
    edition: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
) -> FactorListResponse:
    """Get all factors for a specific scope"""
    return await list_factors(
        request=request,
        response=response,
        fuel_type=None,
        geography=None,
        scope=scope,
        boundary=None,
        page=1,
        limit=100,
        include_preview=False,
        include_connector=False,
        edition=edition,
        current_user=current_user,
    )


# ==================== CALCULATION ENDPOINTS ====================

@app.post(
    "/api/v1/calculate",
    response_model=CalculationResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate emissions",
    description="Calculate GHG emissions for a single activity",
    tags=["Calculations"],
    responses={
        200: {"description": "Calculation successful"},
        400: {"model": ErrorResponse, "description": "Invalid calculation request"},
        404: {"model": ErrorResponse, "description": "Emission factor not found"},
        429: {"description": "Rate limit exceeded"}
    }
)
@limiter.limit("500/minute")
async def calculate_emissions(
    request: Request,
    calc_request: CalculationRequest,
    current_user: dict = Depends(get_current_user)
) -> CalculationResponse:
    """
    Calculate GHG emissions for a single activity.

    Returns total CO2e emissions and breakdown by gas type.
    """
    global calculation_counter

    try:
        # Get emission factor
        factor = emission_db.get_factor_record(
            fuel_type=calc_request.fuel_type,
            unit=calc_request.activity_unit,
            geography=calc_request.geography or "US",
            scope=calc_request.scope.value if calc_request.scope else "1",
            boundary=calc_request.boundary.value if calc_request.boundary else "combustion",
            gwp_set=calc_request.gwp_set.value if calc_request.gwp_set else "IPCC_AR6_100",
            as_of_date=calc_request.calculation_date
        )

        if not factor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No emission factor found for {calc_request.fuel_type} ({calc_request.activity_unit}) in {calc_request.geography or 'US'}"
            )

        # Calculate emissions
        activity = calc_request.activity_amount

        emissions_by_gas = {
            "CO2": factor.vectors.CO2 * activity,
            "CH4": factor.vectors.CH4 * activity,
            "N2O": factor.vectors.N2O * activity
        }

        total_co2e = factor.gwp_100yr.co2e_total * activity

        # Generate calculation ID
        calc_id = f"calc_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:12]}"
        calculation_counter += 1

        return CalculationResponse(
            calculation_id=calc_id,
            emissions_kg_co2e=total_co2e,
            emissions_tonnes_co2e=total_co2e / 1000.0,
            emissions_by_gas=emissions_by_gas,
            factor_used=_factor_to_summary(factor),
            timestamp=DeterministicClock.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error calculating emissions: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error calculating emissions"
        )


@app.post(
    "/api/v1/calculate/batch",
    response_model=BatchCalculationResponse,
    summary="Batch calculate emissions",
    description="Calculate emissions for multiple activities (max 100)",
    tags=["Calculations"]
)
@limiter.limit("100/minute")
async def calculate_batch(
    request: Request,
    batch_request: BatchCalculationRequest,
    current_user: dict = Depends(get_current_user)
) -> BatchCalculationResponse:
    """
    Calculate emissions for multiple activities in a single request.

    Maximum 100 calculations per batch.
    """
    try:
        # Process all calculations
        results = []
        total_emissions = 0.0

        for calc_req in batch_request.calculations:
            calc_response = await calculate_emissions(
                request=request,
                calc_request=calc_req,
                current_user=current_user
            )
            results.append(calc_response)
            total_emissions += calc_response.emissions_kg_co2e

        batch_id = f"batch_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:12]}"

        return BatchCalculationResponse(
            batch_id=batch_id,
            total_emissions_kg_co2e=total_emissions,
            total_emissions_tonnes_co2e=total_emissions / 1000.0,
            calculations=results,
            count=len(results),
            timestamp=DeterministicClock.now()
        )

    except Exception as e:
        logger.error("Error in batch calculation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error in batch calculation"
        )


@app.post(
    "/api/v1/calculate/scope1",
    response_model=EmissionResult,
    summary="Calculate Scope 1 emissions",
    description="Calculate direct emissions (Scope 1)",
    tags=["Calculations"]
)
@limiter.limit("500/minute")
async def calculate_scope1(
    request: Request,
    scope1_request: Scope1Request,
    current_user: dict = Depends(get_current_user)
) -> EmissionResult:
    """Calculate Scope 1 (direct) emissions"""
    try:
        calc_request = CalculationRequest(
            fuel_type=scope1_request.fuel_type,
            activity_amount=scope1_request.consumption,
            activity_unit=scope1_request.unit,
            geography=scope1_request.geography,
            scope="1",
            boundary="combustion"
        )

        calc_response = await calculate_emissions(request, calc_request, current_user)

        return EmissionResult(
            emissions_kg_co2e=calc_response.emissions_kg_co2e,
            emissions_tonnes_co2e=calc_response.emissions_tonnes_co2e,
            gas_breakdown=GHGBreakdown(**calc_response.emissions_by_gas)
        )

    except Exception as e:
        logger.error("Error calculating Scope 1: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error calculating Scope 1 emissions"
        )


@app.post(
    "/api/v1/calculate/scope2",
    response_model=EmissionResult,
    summary="Calculate Scope 2 emissions",
    description="Calculate purchased electricity emissions (Scope 2)",
    tags=["Calculations"]
)
@limiter.limit("500/minute")
async def calculate_scope2(
    request: Request,
    scope2_request: Scope2Request,
    current_user: dict = Depends(get_current_user)
) -> EmissionResult:
    """Calculate Scope 2 (purchased electricity) emissions"""
    try:
        # Use market-based factor if provided, otherwise location-based
        if scope2_request.market_based_factor:
            # Direct calculation with market-based factor
            total_co2e = scope2_request.electricity_kwh * scope2_request.market_based_factor

            return EmissionResult(
                emissions_kg_co2e=total_co2e,
                emissions_tonnes_co2e=total_co2e / 1000.0,
                gas_breakdown=GHGBreakdown(
                    CO2=total_co2e,  # Simplified: assume all CO2
                    CH4=0.0,
                    N2O=0.0
                )
            )
        else:
            # Use location-based factor from database
            calc_request = CalculationRequest(
                fuel_type="electricity",
                activity_amount=scope2_request.electricity_kwh,
                activity_unit="kWh",
                geography=scope2_request.geography,
                scope="2",
                boundary="combustion"
            )

            calc_response = await calculate_emissions(request, calc_request, current_user)

            return EmissionResult(
                emissions_kg_co2e=calc_response.emissions_kg_co2e,
                emissions_tonnes_co2e=calc_response.emissions_tonnes_co2e,
                gas_breakdown=GHGBreakdown(**calc_response.emissions_by_gas)
            )

    except Exception as e:
        logger.error("Error calculating Scope 2: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error calculating Scope 2 emissions"
        )


@app.post(
    "/api/v1/calculate/scope3",
    response_model=EmissionResult,
    summary="Calculate Scope 3 emissions",
    description="Calculate indirect value chain emissions (Scope 3)",
    tags=["Calculations"]
)
@limiter.limit("500/minute")
async def calculate_scope3(
    request: Request,
    scope3_request: Scope3Request,
    current_user: dict = Depends(get_current_user)
) -> EmissionResult:
    """Calculate Scope 3 (indirect) emissions"""
    try:
        # Scope 3 is complex - this is a simplified implementation
        # In production, would have category-specific logic

        # For now, return placeholder

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Scope 3 calculations require category-specific implementation. Coming soon."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error calculating Scope 3: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error calculating Scope 3 emissions"
        )


# ==================== STATISTICS ENDPOINTS ====================

@app.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    summary="Get API statistics",
    description="Retrieve usage statistics and metrics",
    tags=["System"]
)
@limiter.limit("100/minute")
async def get_statistics(
    request: Request,
    current_user: dict = Depends(get_current_user)
) -> StatsResponse:
    """Get API usage statistics"""
    try:
        cache_stats_dict = emission_db.get_cache_stats()

        cache_stats = CacheStats(
            enabled=cache_stats_dict.get("enabled", False),
            hits=cache_stats_dict.get("hits", 0),
            misses=cache_stats_dict.get("misses", 0),
            hit_rate_pct=cache_stats_dict.get("hit_rate_pct", 0.0),
            size=cache_stats_dict.get("size", 0),
            max_size=cache_stats_dict.get("max_size", 0)
        )

        svc = _factor_catalog_service()
        eid = svc.repo.get_default_edition_id()
        cov = svc.repo.coverage_stats(eid)
        total_cat = int(cov.get("total_factors", 0))

        return StatsResponse(
            version="1.0.0",
            total_factors=total_cat or len(emission_db.factors),
            calculations_today=calculation_counter,
            cache_stats=cache_stats,
            uptime_seconds=time.time() - app_start_time,
            timestamp=DeterministicClock.now()
        )

    except Exception as e:
        logger.error("Error retrieving statistics: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error retrieving statistics"
        )


@app.get(
    "/api/v1/stats/coverage",
    response_model=CoverageStats,
    summary="Get coverage statistics",
    description="Retrieve coverage statistics (geographies, fuel types, scopes)",
    tags=["System"]
)
@limiter.limit("100/minute")
async def get_coverage_stats(
    request: Request,
    current_user: dict = Depends(get_current_user)
) -> CoverageStats:
    """Get coverage statistics"""
    try:
        svc = _factor_catalog_service()
        eid = svc.repo.get_default_edition_id()
        cov = svc.repo.coverage_stats(eid)
        return CoverageStats(
            total_factors=int(cov["total_factors"]),
            total_catalog=int(cov.get("total_catalog", cov["total_factors"])),
            certified=int(cov.get("certified", cov["total_factors"])),
            preview=int(cov.get("preview", 0)),
            connector_visible=int(cov.get("connector_visible", 0)),
            geographies=int(cov["geographies"]),
            fuel_types=int(cov["fuel_types"]),
            scopes=cov["scopes"],
            boundaries=cov["boundaries"],
            by_geography=cov["by_geography"],
            by_fuel_type=cov["by_fuel_type"],
        )

    except Exception as e:
        logger.error("Error retrieving coverage stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error retrieving coverage statistics"
        )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status for load balancers",
    tags=["System"]
)
@limiter.limit("1000/minute")
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns 200 if healthy, 503 if unhealthy.
    """
    try:
        # Check database availability
        db_status = "connected" if emission_db and len(emission_db.factors) > 0 else "unavailable"

        # Check cache availability
        cache_status = "available" if emission_db and emission_db.enable_cache else "disabled"

        # Determine overall health
        is_healthy = db_status == "connected"

        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unhealthy"
            )

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=DeterministicClock.now(),
            database=db_status,
            cache=cache_status,
            uptime_seconds=time.time() - app_start_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "timestamp": DeterministicClock.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error("Unhandled exception: %s", exc, exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An internal error occurred",
            "timestamp": DeterministicClock.now().isoformat()
        }
    )


# ==================== HELPER FUNCTIONS ====================

def _factor_to_summary(factor: EmissionFactorRecord) -> EmissionFactorSummary:
    """Convert EmissionFactorRecord to EmissionFactorSummary"""
    return EmissionFactorSummary(
        factor_id=factor.factor_id,
        fuel_type=factor.fuel_type,
        unit=factor.unit,
        geography=factor.geography,
        scope=factor.scope.value,
        boundary=factor.boundary.value,
        co2e_per_unit=factor.gwp_100yr.co2e_total,
        source=factor.provenance.source_org,
        source_year=factor.provenance.source_year,
        data_quality_score=factor.dqs.overall_score,
        uncertainty_percent=factor.uncertainty_95ci * 100,
        factor_status=getattr(factor, "factor_status", "certified") or "certified",
        source_id=getattr(factor, "source_id", None),
        release_version=getattr(factor, "release_version", None),
        license_class=getattr(factor, "license_class", None),
        activity_tags=list(getattr(factor, "activity_tags", []) or []),
        sector_tags=list(getattr(factor, "sector_tags", []) or []),
    )


def _factor_to_response(factor: EmissionFactorRecord) -> EmissionFactorResponse:
    """Convert EmissionFactorRecord to EmissionFactorResponse"""
    return EmissionFactorResponse(
        factor_id=factor.factor_id,
        fuel_type=factor.fuel_type,
        unit=factor.unit,
        geography=factor.geography,
        geography_level=factor.geography_level.value,
        scope=factor.scope.value,
        boundary=factor.boundary.value,
        co2_per_unit=factor.vectors.CO2,
        ch4_per_unit=factor.vectors.CH4,
        n2o_per_unit=factor.vectors.N2O,
        co2e_per_unit=factor.gwp_100yr.co2e_total,
        gwp_set=factor.gwp_100yr.gwp_set.value,
        ch4_gwp=factor.gwp_100yr.CH4_gwp,
        n2o_gwp=factor.gwp_100yr.N2O_gwp,
        data_quality=DataQuality(
            temporal=factor.dqs.temporal,
            geographical=factor.dqs.geographical,
            technological=factor.dqs.technological,
            representativeness=factor.dqs.representativeness,
            methodological=factor.dqs.methodological,
            overall_score=factor.dqs.overall_score,
            rating=factor.dqs.rating.value
        ),
        source=SourceInfo(
            organization=factor.provenance.source_org,
            publication=factor.provenance.source_publication,
            year=factor.provenance.source_year,
            url=factor.provenance.source_url,
            methodology=factor.provenance.methodology.value,
            version=factor.provenance.version
        ),
        uncertainty_95ci=factor.uncertainty_95ci,
        valid_from=factor.valid_from,
        valid_to=factor.valid_to,
        license=factor.license_info.license,
        compliance_frameworks=factor.compliance_frameworks,
        tags=factor.tags,
        notes=factor.notes,
        factor_status=getattr(factor, "factor_status", None),
        source_id=getattr(factor, "source_id", None),
        source_release=getattr(factor, "source_release", None),
        source_record_id=getattr(factor, "source_record_id", None),
        release_version=getattr(factor, "release_version", None),
        replacement_factor_id=getattr(factor, "replacement_factor_id", None),
        license_class=getattr(factor, "license_class", None),
        activity_tags=list(getattr(factor, "activity_tags", []) or []),
        sector_tags=list(getattr(factor, "sector_tags", []) or []),
        redistribution_allowed=factor.license_info.redistribution_allowed,
    )


# ==================== ROOT ENDPOINT ====================

@app.get(
    "/",
    summary="API root",
    description="Welcome message and links",
    tags=["System"]
)
async def root():
    """API root endpoint"""
    return {
        "name": "GreenLang Emission Factor API",
        "version": "1.0.0",
        "description": "Production REST API for emission factor queries and calculations",
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "health": "/api/v1/health",
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "greenlang.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
