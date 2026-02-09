# -*- coding: utf-8 -*-
"""
Data Gateway Service REST API Router - AGENT-DATA-004 (GL-DATA-GW-001)

FastAPI router providing 20 REST API endpoints for data gateway
operations including unified query execution, source management,
schema translation, caching, catalog browsing, and template management.

Endpoints:
    1.  POST /v1/gateway/query              - Execute unified data query
    2.  POST /v1/gateway/query/batch        - Execute batch multi-source query
    3.  GET  /v1/gateway/query/{query_id}   - Get query result by ID
    4.  GET  /v1/gateway/query/{query_id}/lineage - Get query provenance chain
    5.  GET  /v1/gateway/sources            - List all registered data sources
    6.  GET  /v1/gateway/sources/{source_id} - Get source details
    7.  POST /v1/gateway/sources/{source_id}/test - Test source connectivity
    8.  GET  /v1/gateway/sources/{source_id}/schema - Get source schema
    9.  GET  /v1/gateway/catalog            - Browse unified data catalog
    10. GET  /v1/gateway/catalog/search     - Search data catalog
    11. GET  /v1/gateway/schemas            - List registered schemas
    12. POST /v1/gateway/schemas/translate  - Translate between schemas
    13. GET  /v1/gateway/templates          - List query templates
    14. POST /v1/gateway/templates          - Create query template
    15. POST /v1/gateway/templates/{template_id}/execute - Execute from template
    16. GET  /v1/gateway/cache/stats        - Cache statistics
    17. DELETE /v1/gateway/cache            - Invalidate cache
    18. GET  /v1/gateway/health             - Health check
    19. GET  /v1/gateway/health/sources     - Source health statuses
    20. GET  /v1/gateway/statistics         - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/gateway", tags=["Data Gateway"])


# =============================================================================
# Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    agent_id: str = "GL-DATA-GW-001"
    agent_name: str = "API Gateway Agent"
    version: str = "1.0.0"
    timestamp: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""


class QueryResultResponse(BaseModel):
    """Query result response."""
    query_id: str
    source_id: str = ""
    data: List[Dict[str, Any]] = []
    total_count: int = 0
    row_count: int = 0
    errors: List[str] = []
    execution_time_ms: float = 0.0


class SourceResponse(BaseModel):
    """Data source response."""
    source_id: str
    name: str
    source_type: str
    status: str = "active"
    description: str = ""


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    total_size_bytes: int = 0


class TemplateResponse(BaseModel):
    """Query template response."""
    template_id: str
    name: str = ""
    description: str = ""
    created_at: str = ""


# =============================================================================
# Service Dependency
# =============================================================================


def _get_service():
    """Get or create the Data Gateway Service singleton.

    Returns:
        DataGatewayService instance.
    """
    from greenlang.data_gateway.setup import DataGatewayService
    if not hasattr(_get_service, "_instance"):
        _get_service._instance = DataGatewayService()
    return _get_service._instance


# =============================================================================
# 1. POST /v1/gateway/query - Execute unified data query
# =============================================================================


@router.post("/query", tags=["Query"])
async def execute_query(request: Dict[str, Any]):
    """Execute a unified data query across one or more sources.

    Accepts a query request with sources, filters, sorts, aggregations,
    and pagination. Routes to appropriate sources and returns merged results.

    Args:
        request: Query request with sources, filters, sorts, etc.

    Returns:
        QueryResult dictionary.
    """
    try:
        service = _get_service()
        result = service.execute_query(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error executing query: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 2. POST /v1/gateway/query/batch - Execute batch multi-source query
# =============================================================================


@router.post("/query/batch", tags=["Query"])
async def execute_batch_query(request: Dict[str, Any]):
    """Execute a batch of queries across multiple sources.

    Args:
        request: Batch request with 'queries' list.

    Returns:
        List of QueryResult dictionaries.
    """
    try:
        service = _get_service()
        queries = request.get("queries", [])
        if not queries:
            raise ValueError("No queries provided in batch request")
        results = service.execute_batch(queries)
        return {"results": results, "total": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error executing batch query: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 3. GET /v1/gateway/query/{query_id} - Get query result by ID
# =============================================================================


@router.get("/query/{query_id}", tags=["Query"])
async def get_query_result(query_id: str):
    """Get a previously executed query result by ID.

    Args:
        query_id: Query identifier.

    Returns:
        QueryResult dictionary.
    """
    service = _get_service()
    result = service.query_router.get_result(query_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Query result not found: {query_id}"
        )
    return result


# =============================================================================
# 4. GET /v1/gateway/query/{query_id}/lineage - Get query provenance chain
# =============================================================================


@router.get("/query/{query_id}/lineage", tags=["Query"])
async def get_query_lineage(query_id: str):
    """Get the provenance chain for a query execution.

    Args:
        query_id: Query identifier.

    Returns:
        Provenance chain with verification status.
    """
    service = _get_service()
    is_valid, chain = service.provenance.verify_chain(query_id)
    return {
        "query_id": query_id,
        "is_valid": is_valid,
        "chain_length": len(chain),
        "chain": chain,
    }


# =============================================================================
# 5. GET /v1/gateway/sources - List all registered data sources
# =============================================================================


@router.get("/sources", tags=["Sources"])
async def list_sources(
    source_type: Optional[str] = Query(
        None, description="Filter by source type"
    ),
    status: Optional[str] = Query(
        None, description="Filter by source status"
    ),
):
    """List all registered data sources with optional filters.

    Args:
        source_type: Filter by source type (postgresql, redis, etc.).
        status: Filter by status (active, inactive, maintenance).

    Returns:
        List of DataSource dictionaries.
    """
    try:
        service = _get_service()
        results = service.connection_manager.list_sources(
            source_type=source_type,
            status=status,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 6. GET /v1/gateway/sources/{source_id} - Get source details
# =============================================================================


@router.get("/sources/{source_id}", tags=["Sources"])
async def get_source(source_id: str):
    """Get data source details by ID.

    Args:
        source_id: Source identifier.

    Returns:
        DataSource dictionary.
    """
    service = _get_service()
    result = service.connection_manager.get_source(source_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Source not found: {source_id}"
        )
    return result


# =============================================================================
# 7. POST /v1/gateway/sources/{source_id}/test - Test source connectivity
# =============================================================================


@router.post("/sources/{source_id}/test", tags=["Sources"])
async def test_source(source_id: str):
    """Test connectivity to a data source.

    Args:
        source_id: Source identifier.

    Returns:
        Connection test result.
    """
    try:
        service = _get_service()
        result = service.test_source(source_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 8. GET /v1/gateway/sources/{source_id}/schema - Get source schema
# =============================================================================


@router.get("/sources/{source_id}/schema", tags=["Sources"])
async def get_source_schema(source_id: str):
    """Get the schema associated with a data source.

    Looks up the source in the catalog and returns its schema definition.

    Args:
        source_id: Source identifier.

    Returns:
        Schema definition or list of matching schemas.
    """
    try:
        service = _get_service()
        source = service.connection_manager.get_source(source_id)
        if source is None:
            raise HTTPException(
                status_code=404, detail=f"Source not found: {source_id}"
            )
        source_type = source.get("source_type", "")
        schemas = service.schema_translator.list_schemas(
            source_type=source_type,
        )
        return {
            "source_id": source_id,
            "source_type": source_type,
            "schemas": schemas,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 9. GET /v1/gateway/catalog - Browse unified data catalog
# =============================================================================


@router.get("/catalog", tags=["Catalog"])
async def browse_catalog(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    source_type: Optional[str] = Query(
        None, description="Filter by source type"
    ),
    tags: Optional[str] = Query(
        None, description="Comma-separated tags to filter by"
    ),
):
    """Browse the unified data catalog with optional filters.

    Args:
        domain: Filter by business domain.
        source_type: Filter by source type.
        tags: Comma-separated tags filter.

    Returns:
        List of DataCatalogEntry dictionaries.
    """
    try:
        service = _get_service()
        tag_list = (
            [t.strip() for t in tags.split(",") if t.strip()]
            if tags else None
        )
        results = service.data_catalog.list_entries(
            domain=domain,
            source_type=source_type,
            tags=tag_list,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 10. GET /v1/gateway/catalog/search - Search data catalog
# =============================================================================


@router.get("/catalog/search", tags=["Catalog"])
async def search_catalog(
    q: str = Query("", description="Search query string"),
):
    """Search the data catalog by keyword.

    Searches across name, description, domain, tags, and source type.

    Args:
        q: Search query string.

    Returns:
        List of matching DataCatalogEntry dictionaries.
    """
    try:
        service = _get_service()
        results = service.search_catalog(q)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 11. GET /v1/gateway/schemas - List registered schemas
# =============================================================================


@router.get("/schemas", tags=["Schemas"])
async def list_schemas(
    source_type: Optional[str] = Query(
        None, description="Filter by source type"
    ),
):
    """List all registered schema definitions.

    Args:
        source_type: Filter by source type.

    Returns:
        List of SchemaDefinition dictionaries.
    """
    try:
        service = _get_service()
        results = service.schema_translator.list_schemas(
            source_type=source_type,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 12. POST /v1/gateway/schemas/translate - Translate between schemas
# =============================================================================


@router.post("/schemas/translate", tags=["Schemas"])
async def translate_schema(request: Dict[str, Any]):
    """Translate data from one schema format to another.

    Args:
        request: Translation request with data, source_type, target_type.

    Returns:
        Translated data dictionary.
    """
    try:
        service = _get_service()
        result = service.translate_schema(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 13. GET /v1/gateway/templates - List query templates
# =============================================================================


@router.get("/templates", tags=["Templates"])
async def list_templates():
    """List all available query templates.

    Returns:
        List of template dictionaries.
    """
    try:
        service = _get_service()
        return service.list_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 14. POST /v1/gateway/templates - Create query template
# =============================================================================


@router.post("/templates", tags=["Templates"])
async def create_template(request: Dict[str, Any]):
    """Create a new query template.

    Args:
        request: Template definition with name and query structure.

    Returns:
        Created template with template_id.
    """
    try:
        service = _get_service()
        template_id = service.create_template(request)
        template = service.get_template(template_id)
        return template
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 15. POST /v1/gateway/templates/{template_id}/execute - Execute from template
# =============================================================================


@router.post("/templates/{template_id}/execute", tags=["Templates"])
async def execute_template(
    template_id: str,
    request: Optional[Dict[str, Any]] = None,
):
    """Execute a query from a saved template.

    Args:
        template_id: Template identifier.
        request: Optional parameter overrides.

    Returns:
        QueryResult dictionary.
    """
    try:
        service = _get_service()
        parameters = request or {}
        result = service.execute_template(template_id, parameters)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 16. GET /v1/gateway/cache/stats - Cache statistics
# =============================================================================


@router.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """Get cache performance statistics.

    Returns:
        Cache statistics including hits, misses, and hit rate.
    """
    try:
        service = _get_service()
        return service.get_cache_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 17. DELETE /v1/gateway/cache - Invalidate cache
# =============================================================================


@router.delete("/cache", tags=["Cache"])
async def invalidate_cache(
    source_id: Optional[str] = Query(
        None, description="Invalidate by source ID"
    ),
    query_hash: Optional[str] = Query(
        None, description="Invalidate by query hash"
    ),
    invalidate_all: bool = Query(
        False, description="Invalidate all entries"
    ),
):
    """Invalidate cache entries.

    Can invalidate by source ID, query hash, or all entries.

    Args:
        source_id: Invalidate entries for this source.
        query_hash: Invalidate specific query hash.
        invalidate_all: Invalidate all entries.

    Returns:
        Number of entries invalidated.
    """
    try:
        service = _get_service()
        count = service.invalidate_cache({
            "source_id": source_id,
            "query_hash": query_hash,
            "invalidate_all": invalidate_all,
        })
        return {
            "invalidated_count": count,
            "source_id": source_id,
            "query_hash": query_hash,
            "invalidate_all": invalidate_all,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 18. GET /v1/gateway/health - Health check
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check endpoint.

    Returns:
        Health status with agent metadata.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
    )


# =============================================================================
# 19. GET /v1/gateway/health/sources - Source health statuses
# =============================================================================


@router.get("/health/sources", tags=["Health"])
async def source_health_check():
    """Check health status of all registered data sources.

    Returns:
        List of SourceHealthCheck dictionaries.
    """
    try:
        service = _get_service()
        results = service.connection_manager.check_all_health()
        healthy = sum(
            1 for r in results if r.get("status") == "healthy"
        )
        return {
            "total_sources": len(results),
            "healthy": healthy,
            "unhealthy": len(results) - healthy,
            "checks": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 20. GET /v1/gateway/statistics - Service statistics
# =============================================================================


@router.get("/statistics", tags=["Statistics"])
async def get_statistics():
    """Get comprehensive service statistics.

    Returns:
        Statistics from all 7 engines.
    """
    try:
        service = _get_service()
        return service.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Utility
# =============================================================================


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a Pydantic model or dataclass to dictionary.

    Args:
        obj: Object to convert.

    Returns:
        Dictionary representation.
    """
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
