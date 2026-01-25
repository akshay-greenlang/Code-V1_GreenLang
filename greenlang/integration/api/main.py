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

from fastapi import FastAPI, HTTPException, Query, Depends, Request, status
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
    StatsResponse,
    CoverageStats,
    HealthResponse,
    ErrorResponse,
    GHGBreakdown,
    DataQuality,
    SourceInfo,
    CacheStats,
)

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
    allowed_hosts=["*.greenlang.io", "localhost", "127.0.0.1"]
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer(auto_error=False)

# ==================== GLOBAL STATE ====================

# Emission factor database (initialized on startup)
emission_db: Optional[EmissionFactorDatabase] = None

# Application metrics
app_start_time = time.time()
calculation_counter = 0

# ==================== STARTUP / SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global emission_db

    logger.info("Starting GreenLang Emission Factor API v1.0.0")

    # Initialize emission factor database with caching
    emission_db = EmissionFactorDatabase(
        enable_cache=True,
        cache_size=1000,
        cache_ttl=3600  # 1 hour
    )

    logger.info(f"Loaded {len(emission_db.factors)} emission factors")
    logger.info(f"Cache enabled: {emission_db.enable_cache}")
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
                # In production, validate against key store
                logger.info(f"API key authentication: {token[:8]}...")
                return {"user_id": f"apikey:{token[:12]}", "tenant_id": "default"}
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
                    "tenant_id": payload.get("tenant_id", "default")
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
        return {"user_id": "anonymous", "tenant_id": "default"}

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

    return response


# ==================== FACTOR QUERY ENDPOINTS ====================

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
    fuel_type: Optional[str] = Query(None, description="Filter by fuel type"),
    geography: Optional[str] = Query(None, description="Filter by geography"),
    scope: Optional[str] = Query(None, description="Filter by scope (1, 2, or 3)"),
    boundary: Optional[str] = Query(None, description="Filter by boundary"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(100, ge=1, le=500, description="Items per page (max 500)"),
    current_user: dict = Depends(get_current_user)
) -> FactorListResponse:
    """
    List emission factors with filtering and pagination.

    Returns paginated list of factors matching the specified filters.
    """
    try:
        # Get all factors matching filters
        all_factors = emission_db.list_factors(
            fuel_type=fuel_type,
            geography=geography
        )

        # Apply additional filters
        filtered_factors = []
        for factor in all_factors:
            if scope and factor.scope.value != scope:
                continue
            if boundary and factor.boundary.value != boundary:
                continue
            filtered_factors.append(factor)

        # Calculate pagination
        total_count = len(filtered_factors)
        total_pages = (total_count + limit - 1) // limit
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        # Get page of factors
        page_factors = filtered_factors[start_idx:end_idx]

        # Convert to summary models
        factor_summaries = [
            _factor_to_summary(factor)
            for factor in page_factors
        ]

        return FactorListResponse(
            factors=factor_summaries,
            total_count=total_count,
            page=page,
            page_size=len(factor_summaries),
            total_pages=total_pages
        )

    except Exception as e:
        logger.error(f"Error listing factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error retrieving factors"
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
    factor_id: str,
    current_user: dict = Depends(get_current_user)
) -> EmissionFactorResponse:
    """
    Get detailed emission factor by ID.

    Returns complete factor record with all metadata.
    """
    try:
        # Find factor by ID
        factor = None
        for f in emission_db.factors.values():
            if f.factor_id == factor_id:
                factor = f
                break

        if not factor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Factor '{factor_id}' not found"
            )

        return _factor_to_response(factor)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving factor {factor_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error retrieving factor"
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
    q: str = Query(..., min_length=2, description="Search query"),
    geography: Optional[str] = Query(None, description="Filter by geography"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    current_user: dict = Depends(get_current_user)
) -> FactorSearchResponse:
    """
    Search emission factors by text query.

    Searches across fuel_type, geography, tags, and notes fields.
    """
    try:
        start_time = time.time()

        # Simple text search (case-insensitive)
        q_lower = q.lower()
        results = []

        for factor in emission_db.factors.values():
            # Apply geography filter
            if geography and factor.geography != geography:
                continue

            # Search in relevant fields
            searchable_text = " ".join([
                factor.fuel_type,
                factor.geography,
                factor.scope.value,
                factor.boundary.value,
                " ".join(factor.tags),
                factor.notes or ""
            ]).lower()

            if q_lower in searchable_text:
                results.append(factor)

                if len(results) >= limit:
                    break

        search_time_ms = (time.time() - start_time) * 1000

        return FactorSearchResponse(
            query=q,
            factors=[_factor_to_summary(f) for f in results],
            count=len(results),
            search_time_ms=search_time_ms
        )

    except Exception as e:
        logger.error(f"Error searching factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error searching factors"
        )


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
    fuel_type: str,
    geography: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
) -> FactorListResponse:
    """Get all factors for a specific fuel type"""
    return await list_factors(
        request=request,
        fuel_type=fuel_type,
        geography=geography,
        scope=None,
        boundary=None,
        page=1,
        limit=100,
        current_user=current_user
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
    scope: str,
    current_user: dict = Depends(get_current_user)
) -> FactorListResponse:
    """Get all factors for a specific scope"""
    return await list_factors(
        request=request,
        fuel_type=None,
        geography=None,
        scope=scope,
        boundary=None,
        page=1,
        limit=100,
        current_user=current_user
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
        logger.error(f"Error calculating emissions: {e}", exc_info=True)
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
        logger.error(f"Error in batch calculation: {e}", exc_info=True)
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
        logger.error(f"Error calculating Scope 1: {e}", exc_info=True)
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
        logger.error(f"Error calculating Scope 2: {e}", exc_info=True)
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
        logger.error(f"Error calculating Scope 3: {e}", exc_info=True)
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

        return StatsResponse(
            version="1.0.0",
            total_factors=len(emission_db.factors),
            calculations_today=calculation_counter,
            cache_stats=cache_stats,
            uptime_seconds=time.time() - app_start_time,
            timestamp=DeterministicClock.now()
        )

    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}", exc_info=True)
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
        geographies = set()
        fuel_types = set()
        scopes = {"1": 0, "2": 0, "3": 0}
        boundaries = {}
        by_geography = {}
        by_fuel_type = {}

        for factor in emission_db.factors.values():
            geographies.add(factor.geography)
            fuel_types.add(factor.fuel_type)
            scopes[factor.scope.value] = scopes.get(factor.scope.value, 0) + 1
            boundaries[factor.boundary.value] = boundaries.get(factor.boundary.value, 0) + 1
            by_geography[factor.geography] = by_geography.get(factor.geography, 0) + 1
            by_fuel_type[factor.fuel_type] = by_fuel_type.get(factor.fuel_type, 0) + 1

        return CoverageStats(
            total_factors=len(emission_db.factors),
            geographies=len(geographies),
            fuel_types=len(fuel_types),
            scopes=scopes,
            boundaries=boundaries,
            by_geography=by_geography,
            by_fuel_type=by_fuel_type
        )

    except Exception as e:
        logger.error(f"Error retrieving coverage stats: {e}", exc_info=True)
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
        logger.error(f"Health check failed: {e}", exc_info=True)
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
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

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
        uncertainty_percent=factor.uncertainty_95ci * 100
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
        notes=factor.notes
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
