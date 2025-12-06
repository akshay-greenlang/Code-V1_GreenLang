"""
Complete FastAPI Application with GraphQL Integration

This example demonstrates how to set up and use the Process Heat GraphQL
schema with FastAPI, including endpoints, error handling, and best practices.

Run with:
    python examples/graphql_fastapi_example.py
    # or
    uvicorn examples.graphql_fastapi_example:app --reload

Then access:
- GraphQL Playground: http://localhost:8000/graphql
- Health Check: http://localhost:8000/health
- Status: http://localhost:8000/graphql/status
- API Docs: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

try:
    from greenlang.infrastructure.api.graphql_integration import (
        setup_graphql,
        GraphQLConfig,
        query_agents,
        query_emissions,
        run_calculation,
        QueryExecutor,
    )
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# STARTUP/SHUTDOWN HOOKS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("Starting Process Heat GraphQL API")
    logger.info("GraphQL endpoint available at /graphql")
    logger.info("GraphQL playground available at /graphql")

    if not GRAPHQL_AVAILABLE:
        logger.warning("GraphQL not available - install strawberry-graphql[fastapi]")

    yield

    # Shutdown
    logger.info("Shutting down Process Heat GraphQL API")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Process Heat GraphQL API",
    description="GraphQL API for Process Heat agents with real-time monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# SETUP GRAPHQL
# ============================================================================

if GRAPHQL_AVAILABLE:
    try:
        graphql_config = GraphQLConfig(
            path="/graphql",
            enable_schema_introspection=True,
            enable_playground=True,
            max_query_depth=10,
            timeout_seconds=30.0,
        )

        setup_graphql(app, graphql_config)
        logger.info("GraphQL integration setup complete")

    except ImportError as e:
        logger.error(f"GraphQL setup failed: {e}")


# ============================================================================
# HEALTH CHECK AND STATUS ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "process_heat_graphql_api",
        "version": "1.0.0",
        "graphql_available": GRAPHQL_AVAILABLE,
    }


@app.get("/", tags=["Info"])
async def root() -> Dict[str, Any]:
    """
    API information endpoint.

    Returns:
        API information and endpoints
    """
    return {
        "name": "Process Heat GraphQL API",
        "version": "1.0.0",
        "endpoints": {
            "graphql": "/graphql",
            "graphql_playground": "/graphql",
            "graphql_status": "/graphql/status",
            "health": "/health",
            "api_docs": "/docs",
            "redoc": "/redoc",
            "agents": "/api/v1/agents",
            "emissions": "/api/v1/emissions",
            "jobs": "/api/v1/jobs",
        },
    }


# ============================================================================
# REST ENDPOINTS (Alternative to GraphQL)
# ============================================================================

@app.get("/api/v1/agents", tags=["Agents"])
async def list_agents(
    status: Optional[str] = Query(None, description="Filter by status"),
) -> Dict[str, Any]:
    """
    List process heat agents.

    Query Args:
        status: Optional status filter (idle, running, completed, failed)

    Returns:
        List of agents with metrics

    Example:
        GET /api/v1/agents?status=idle
    """
    if not GRAPHQL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GraphQL not available - strawberry-graphql[fastapi] required"
        )

    try:
        executor = app.state.graphql_executor
        result = await query_agents(executor, status=status)

        if "errors" in result:
            raise HTTPException(status_code=400, detail=result["errors"])

        agents = result.get("data", {}).get("agents", [])

        return {
            "status": "success",
            "count": len(agents),
            "data": agents,
        }

    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/emissions", tags=["Emissions"])
async def get_emissions(
    facility_id: str = Query(..., description="Facility ID"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
) -> Dict[str, Any]:
    """
    Get emissions for a facility.

    Query Args:
        facility_id: Target facility ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of emission results with provenance hashes

    Example:
        GET /api/v1/emissions?facility_id=fac-1&start_date=2025-01-01&end_date=2025-03-31
    """
    if not GRAPHQL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GraphQL not available"
        )

    try:
        executor = app.state.graphql_executor
        result = await query_emissions(
            executor,
            facility_id=facility_id,
            start_date=start_date,
            end_date=end_date
        )

        if "errors" in result:
            raise HTTPException(status_code=400, detail=result["errors"])

        emissions = result.get("data", {}).get("emissions", [])

        return {
            "status": "success",
            "count": len(emissions),
            "facility_id": facility_id,
            "period": {
                "start": start_date,
                "end": end_date,
            },
            "data": emissions,
        }

    except Exception as e:
        logger.error(f"Failed to get emissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/jobs", tags=["Jobs"])
async def start_calculation(
    agent_id: str = Query(..., description="Agent ID"),
    facility_id: str = Query(..., description="Facility ID"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    priority: str = Query("normal", description="Priority (low, normal, high)"),
) -> Dict[str, Any]:
    """
    Start a calculation job.

    Query Args:
        agent_id: Target agent ID
        facility_id: Target facility ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        priority: Job priority

    Returns:
        Created job with ID and status

    Example:
        POST /api/v1/jobs?agent_id=agent-001&facility_id=fac-1&start_date=2025-01-01&end_date=2025-03-31&priority=high
    """
    if not GRAPHQL_AVAILABLE:
        raise HTTPException(status_code=503, detail="GraphQL not available")

    # Validate priority
    if priority not in ["low", "normal", "high"]:
        raise HTTPException(status_code=400, detail="Invalid priority")

    try:
        executor = app.state.graphql_executor
        result = await run_calculation(
            executor,
            agent_id=agent_id,
            facility_id=facility_id,
            start_date=start_date,
            end_date=end_date,
            priority=priority
        )

        if "errors" in result:
            raise HTTPException(status_code=400, detail=result["errors"])

        job = result.get("data", {}).get("runCalculation", {})

        return {
            "status": "success",
            "job": job,
        }

    except Exception as e:
        logger.error(f"Failed to start calculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MIDDLEWARE AND UTILITIES
# ============================================================================

@app.middleware("http")
async def log_requests(request, call_next):
    """Log HTTP requests and responses."""
    logger.info(f"{request.method} {request.url.path}")

    response = await call_next(request)

    logger.info(f"Response status: {response.status_code}")

    return response


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
        },
    )


# ============================================================================
# EXAMPLE QUERIES (in comments for documentation)
# ============================================================================

"""
EXAMPLE GraphQL QUERIES:

1. List all agents:
   query {
       agents {
           id
           name
           status
           metrics {
               executionTimeMs
               errorCount
           }
       }
   }

2. Get emissions for facility:
   query {
       emissions(
           facilityId: "facility-1"
           dateRange: {
               startDate: "2025-01-01"
               endDate: "2025-03-31"
           }
       ) {
           id
           co2Tonnes
           totalCo2eTonnes
           provenanceHash
       }
   }

3. Start calculation job:
   mutation {
       runCalculation(input: {
           agentId: "agent-001"
           facilityId: "facility-1"
           dateRange: {
               startDate: "2025-01-01"
               endDate: "2025-03-31"
           }
           priority: "high"
       }) {
           id
           status
           progressPercent
       }
   }

4. Monitor job progress:
   subscription {
       jobProgress(jobId: "job-id") {
           jobId
           progressPercent
           status
           message
       }
   }
"""


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Run with: python examples/graphql_fastapi_example.py
    uvicorn.run(
        "examples.graphql_fastapi_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
