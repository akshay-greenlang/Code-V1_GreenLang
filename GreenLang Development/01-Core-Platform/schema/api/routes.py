# -*- coding: utf-8 -*-
"""
FastAPI Routes for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module provides the REST API endpoints for the Schema Validator service:

Endpoints:
    POST /v1/schema/validate           - Validate single payload
    POST /v1/schema/validate/batch     - Validate multiple payloads
    POST /v1/schema/compile            - Compile schema to IR
    GET  /v1/schema/{schema_id}/versions - List schema versions
    GET  /v1/schema/{schema_id}/{version} - Get schema details
    GET  /health                       - Health check
    GET  /metrics                      - Prometheus metrics

All endpoints include:
    - OpenAPI documentation with examples
    - Request validation via Pydantic
    - Rate limiting
    - Distributed tracing (X-Trace-ID header)
    - GreenLang error codes (GLSCHEMA-API-*)

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.schema.api.routes import router
    >>> app = FastAPI()
    >>> app.include_router(router)

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.3
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from greenlang.schema.api.dependencies import (
    APIConfig,
    MetricsCollector,
    RequestContext,
    SERVICE_VERSION,
    check_api_key,
    check_rate_limit,
    get_compiler,
    get_config,
    get_metrics,
    get_registry,
    get_request_context,
    get_uptime_seconds,
    get_validator,
)
from greenlang.schema.api.models import (
    BatchItemResult,
    BatchSummary,
    BatchValidateRequest,
    BatchValidateResponse,
    CompileRequest,
    CompileResponse,
    ComponentHealth,
    ErrorDetail,
    ErrorResponse,
    GreenLangSchemaErrorCodes,
    HealthResponse,
    MetricsResponse,
    SchemaDetailResponse,
    SchemaVersionInfo,
    SchemaVersionsResponse,
    ValidateRequest,
    ValidateResponse,
)
from greenlang.schema.constants import MAX_BATCH_ITEMS, MAX_PAYLOAD_BYTES
from greenlang.schema.models.config import ValidationOptions
from greenlang.schema.models.report import ValidationSummary


logger = logging.getLogger(__name__)


# =============================================================================
# ROUTER INITIALIZATION
# =============================================================================

# Main API router for schema operations
router = APIRouter(prefix="/v1/schema", tags=["Schema Validation"])

# System router for health and metrics
system_router = APIRouter(tags=["System"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_error_response(
    error_type: str,
    message: str,
    details: Optional[List[ErrorDetail]] = None,
    trace_id: Optional[str] = None,
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error_type,
        message=message,
        details=details or [],
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc),
    )


def compute_schema_hash(content: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of schema content."""
    content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()


# =============================================================================
# VALIDATION ENDPOINTS
# =============================================================================


@router.post(
    "/validate",
    response_model=ValidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate a single payload",
    description="""
Validate a single payload against a GreenLang schema.

The endpoint validates the payload through multiple phases:
1. **Parse** - Parse YAML/JSON payload
2. **Compile** - Compile schema to IR (cached)
3. **Structural** - Validate types, required fields
4. **Constraints** - Validate ranges, patterns, enums
5. **Units** - Validate unit dimensions
6. **Rules** - Evaluate cross-field rules
7. **Lint** - Check for typos, deprecated fields

Returns a complete validation report with findings, normalized payload,
and fix suggestions.
    """,
    responses={
        200: {"description": "Validation completed (check 'valid' field)"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Schema not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def validate_payload(
    request_body: ValidateRequest,
    context: RequestContext = Depends(get_request_context),
    config: APIConfig = Depends(get_config),
    metrics: MetricsCollector = Depends(get_metrics),
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: None = Depends(check_api_key),
) -> ValidateResponse:
    """
    Validate a single payload against a schema.

    The validation is performed synchronously and returns a complete report.
    For large payloads or high-volume validation, consider using the batch
    endpoint.
    """
    start_time = time.perf_counter()

    logger.info(
        f"Validate request: schema={request_body.schema_ref} [{context.trace_id}]"
    )

    try:
        # Get validator
        validator = await get_validator()

        # Perform validation
        result = validator.validate(
            payload=request_body.payload,
            schema_ref=request_body.schema_ref,
            options=request_body.options,
        )

        # Record metrics
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_validation(valid=result.valid, latency_ms=latency_ms)

        # Build response
        response = ValidateResponse(
            valid=result.valid,
            schema_ref=result.schema_ref,
            schema_hash=result.schema_hash,
            summary=result.summary,
            findings=result.findings,
            normalized_payload=result.normalized_payload,
            fix_suggestions=result.fix_suggestions,
            timings_ms=result.timings.to_dict() if result.timings else {},
        )

        logger.info(
            f"Validation completed: valid={result.valid}, "
            f"errors={result.summary.error_count}, latency={latency_ms:.1f}ms "
            f"[{context.trace_id}]"
        )

        return response

    except ValueError as e:
        logger.warning(f"Validation error: {e} [{context.trace_id}]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(
                error_type="validation_error",
                message=str(e),
                details=[
                    ErrorDetail(
                        code=GreenLangSchemaErrorCodes.PARAM_INVALID,
                        message=str(e),
                    )
                ],
                trace_id=context.trace_id,
            ).model_dump(),
        )

    except Exception as e:
        logger.error(f"Validation failed: {e} [{context.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Validation failed",
                details=[
                    ErrorDetail(
                        code=GreenLangSchemaErrorCodes.INTERNAL_ERROR,
                        message="An internal error occurred during validation",
                    )
                ],
                trace_id=context.trace_id,
            ).model_dump(),
        )


@router.post(
    "/validate/batch",
    response_model=BatchValidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate multiple payloads",
    description="""
Validate multiple payloads against the same schema in a single request.

This endpoint is optimized for batch processing:
- Schema is compiled once and reused
- Parallel processing of items
- Aggregate summary with individual results

Maximum batch size is configurable (default: 1000 items).
    """,
    responses={
        200: {"description": "Batch validation completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Schema not found"},
        413: {"model": ErrorResponse, "description": "Batch too large"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def validate_batch(
    request_body: BatchValidateRequest,
    context: RequestContext = Depends(get_request_context),
    config: APIConfig = Depends(get_config),
    metrics: MetricsCollector = Depends(get_metrics),
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: None = Depends(check_api_key),
) -> BatchValidateResponse:
    """
    Validate multiple payloads against the same schema.

    The schema is compiled once and reused for all payloads,
    providing better performance than individual requests.
    """
    start_time = time.perf_counter()

    logger.info(
        f"Batch validate request: schema={request_body.schema_ref}, "
        f"items={len(request_body.payloads)} [{context.trace_id}]"
    )

    # Check batch size limit
    if len(request_body.payloads) > MAX_BATCH_ITEMS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=create_error_response(
                error_type="batch_too_large",
                message=f"Batch size {len(request_body.payloads)} exceeds maximum {MAX_BATCH_ITEMS}",
                details=[
                    ErrorDetail(
                        code=GreenLangSchemaErrorCodes.BATCH_TOO_LARGE,
                        message=f"Maximum batch size is {MAX_BATCH_ITEMS}",
                        field="payloads",
                    )
                ],
                trace_id=context.trace_id,
            ).model_dump(),
        )

    try:
        # Get validator
        validator = await get_validator()

        # Perform batch validation
        batch_result = validator.validate_batch(
            payloads=request_body.payloads,
            schema_ref=request_body.schema_ref,
            options=request_body.options,
        )

        # Convert results to API format
        results: List[BatchItemResult] = []
        total_findings = 0

        for item in batch_result.results:
            results.append(
                BatchItemResult(
                    index=item.index,
                    id=item.id,
                    valid=item.valid,
                    findings=item.findings,
                    normalized_payload=item.normalized_payload,
                )
            )
            total_findings += len(item.findings)

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Build summary
        summary = BatchSummary(
            total_items=batch_result.summary.total_items,
            valid_count=batch_result.summary.valid_count,
            error_count=batch_result.summary.error_count,
            warning_count=batch_result.summary.warning_count,
            total_findings=total_findings,
            processing_time_ms=processing_time_ms,
        )

        # Record metrics
        metrics.record_batch_validation(
            item_count=len(request_body.payloads),
            latency_ms=processing_time_ms,
        )

        # Build response
        response = BatchValidateResponse(
            schema_ref=batch_result.schema_ref,
            schema_hash=batch_result.schema_hash,
            summary=summary,
            results=results,
        )

        logger.info(
            f"Batch validation completed: {summary.valid_count}/{summary.total_items} valid, "
            f"latency={processing_time_ms:.1f}ms [{context.trace_id}]"
        )

        return response

    except ValueError as e:
        logger.warning(f"Batch validation error: {e} [{context.trace_id}]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(
                error_type="validation_error",
                message=str(e),
                details=[
                    ErrorDetail(
                        code=GreenLangSchemaErrorCodes.PARAM_INVALID,
                        message=str(e),
                    )
                ],
                trace_id=context.trace_id,
            ).model_dump(),
        )

    except Exception as e:
        logger.error(
            f"Batch validation failed: {e} [{context.trace_id}]", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Batch validation failed",
                details=[
                    ErrorDetail(
                        code=GreenLangSchemaErrorCodes.INTERNAL_ERROR,
                        message="An internal error occurred during batch validation",
                    )
                ],
                trace_id=context.trace_id,
            ).model_dump(),
        )


# =============================================================================
# COMPILATION ENDPOINT
# =============================================================================


@router.post(
    "/compile",
    response_model=CompileResponse,
    status_code=status.HTTP_200_OK,
    summary="Compile a schema",
    description="""
Compile a schema to intermediate representation (IR).

This endpoint parses and validates the schema, computing:
- Schema hash (SHA-256)
- Property definitions
- Constraint metadata
- Rule bindings

Optionally returns the full compiled IR for inspection.
    """,
    responses={
        200: {"description": "Compilation successful"},
        400: {"model": ErrorResponse, "description": "Invalid schema"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def compile_schema(
    request_body: CompileRequest,
    context: RequestContext = Depends(get_request_context),
    metrics: MetricsCollector = Depends(get_metrics),
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: None = Depends(check_api_key),
) -> CompileResponse:
    """
    Compile a schema to intermediate representation.

    The compiled IR is cached for subsequent validation requests.
    """
    start_time = time.perf_counter()

    logger.info(f"Compile request [{context.trace_id}]")

    try:
        # Get compiler
        compiler = await get_compiler()

        # Parse schema source if string
        schema_source = request_body.schema_source
        if isinstance(schema_source, str):
            # Parse YAML/JSON string
            import yaml

            try:
                schema_source = yaml.safe_load(schema_source)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML/JSON: {e}")

        # Compute schema hash
        schema_hash = compute_schema_hash(schema_source)

        # Determine schema ID and version
        schema_id = "_inline"
        version = "1.0.0"
        if request_body.schema_ref:
            schema_id = request_body.schema_ref.schema_id
            version = request_body.schema_ref.version

        # Compile schema
        result = compiler.compile(
            schema_source=schema_source,
            schema_id=schema_id,
            version=version,
        )

        # Build response
        response = CompileResponse(
            schema_hash=result.ir.schema_hash if result.ir else schema_hash,
            compiled_at=datetime.now(timezone.utc),
            warnings=result.warnings if hasattr(result, "warnings") else [],
            properties_count=len(result.ir.properties) if result.ir else 0,
            rules_count=len(result.ir.rule_bindings) if result.ir else 0,
        )

        # Include IR if requested
        if request_body.include_ir and result.ir:
            response.ir = {
                "schema_id": result.ir.schema_id,
                "version": result.ir.version,
                "schema_hash": result.ir.schema_hash,
                "properties": list(result.ir.properties.keys()) if result.ir.properties else [],
                "required_paths": list(result.ir.required_paths) if result.ir.required_paths else [],
            }

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Compilation completed: hash={schema_hash[:16]}..., "
            f"latency={latency_ms:.1f}ms [{context.trace_id}]"
        )

        return response

    except ValueError as e:
        logger.warning(f"Compilation error: {e} [{context.trace_id}]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(
                error_type="compilation_error",
                message=str(e),
                details=[
                    ErrorDetail(
                        code=GreenLangSchemaErrorCodes.COMPILATION_FAILED,
                        message=str(e),
                    )
                ],
                trace_id=context.trace_id,
            ).model_dump(),
        )

    except Exception as e:
        logger.error(f"Compilation failed: {e} [{context.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Compilation failed",
                details=[
                    ErrorDetail(
                        code=GreenLangSchemaErrorCodes.INTERNAL_ERROR,
                        message="An internal error occurred during compilation",
                    )
                ],
                trace_id=context.trace_id,
            ).model_dump(),
        )


# =============================================================================
# SCHEMA REGISTRY ENDPOINTS
# =============================================================================


@router.get(
    "/{schema_id:path}/versions",
    response_model=SchemaVersionsResponse,
    summary="List schema versions",
    description="Get all available versions of a schema from the registry.",
    responses={
        200: {"description": "List of versions"},
        404: {"model": ErrorResponse, "description": "Schema not found"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def list_schema_versions(
    schema_id: str = Path(..., description="Schema identifier (e.g., 'emissions/activity')"),
    context: RequestContext = Depends(get_request_context),
    _rate_limit: None = Depends(check_rate_limit),
) -> SchemaVersionsResponse:
    """
    List all available versions of a schema.

    Returns version information including deprecation status.
    """
    logger.info(f"List versions for schema: {schema_id} [{context.trace_id}]")

    try:
        # Get registry
        registry = await get_registry()

        if registry is None:
            # No registry configured - return empty
            return SchemaVersionsResponse(
                schema_id=schema_id,
                versions=[],
                latest=None,
            )

        # Get versions from registry
        versions_list = registry.list_versions(schema_id)

        if not versions_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Schema not found: {schema_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangSchemaErrorCodes.SCHEMA_NOT_FOUND,
                            message=f"No schema found with ID '{schema_id}'",
                        )
                    ],
                    trace_id=context.trace_id,
                ).model_dump(),
            )

        # Convert to response format
        versions = []
        latest = None

        for v in versions_list:
            version_info = SchemaVersionInfo(
                version=v.get("version", v) if isinstance(v, dict) else str(v),
                created_at=v.get("created_at") if isinstance(v, dict) else None,
                deprecated=v.get("deprecated", False) if isinstance(v, dict) else False,
                deprecated_message=v.get("deprecated_message") if isinstance(v, dict) else None,
            )
            versions.append(version_info)

            # Track latest non-deprecated version
            if not version_info.deprecated:
                latest = version_info.version

        return SchemaVersionsResponse(
            schema_id=schema_id,
            versions=versions,
            latest=latest,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Failed to list versions: {e} [{context.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to list schema versions",
                trace_id=context.trace_id,
            ).model_dump(),
        )


@router.get(
    "/{schema_id:path}/{version}",
    response_model=SchemaDetailResponse,
    summary="Get schema details",
    description="Get the full schema definition for a specific version.",
    responses={
        200: {"description": "Schema details"},
        404: {"model": ErrorResponse, "description": "Schema or version not found"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def get_schema(
    schema_id: str = Path(..., description="Schema identifier"),
    version: str = Path(..., description="Schema version (e.g., '1.3.0')"),
    context: RequestContext = Depends(get_request_context),
    _rate_limit: None = Depends(check_rate_limit),
) -> SchemaDetailResponse:
    """
    Get the full schema definition for a specific version.

    Returns the complete schema content with metadata.
    """
    logger.info(f"Get schema: {schema_id}@{version} [{context.trace_id}]")

    try:
        # Get registry
        registry = await get_registry()

        if registry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message="No schema registry configured",
                    details=[
                        ErrorDetail(
                            code=GreenLangSchemaErrorCodes.SCHEMA_NOT_FOUND,
                            message="Schema registry is not available",
                        )
                    ],
                    trace_id=context.trace_id,
                ).model_dump(),
            )

        # Get schema from registry
        schema_source = registry.resolve(schema_id, version)

        if schema_source is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Schema version not found: {schema_id}@{version}",
                    details=[
                        ErrorDetail(
                            code=GreenLangSchemaErrorCodes.VERSION_NOT_FOUND,
                            message=f"Version '{version}' not found for schema '{schema_id}'",
                        )
                    ],
                    trace_id=context.trace_id,
                ).model_dump(),
            )

        # Get schema content
        content = schema_source.content if hasattr(schema_source, "content") else schema_source

        # Compute hash
        schema_hash = compute_schema_hash(content)

        return SchemaDetailResponse(
            schema_id=schema_id,
            version=version,
            schema_hash=schema_hash,
            content=content,
            created_at=schema_source.created_at if hasattr(schema_source, "created_at") else None,
            deprecated=schema_source.deprecated if hasattr(schema_source, "deprecated") else False,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Failed to get schema: {e} [{context.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get schema",
                trace_id=context.trace_id,
            ).model_dump(),
        )


# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================


@system_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service and its dependencies.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check(
    context: RequestContext = Depends(get_request_context),
    metrics: MetricsCollector = Depends(get_metrics),
) -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns overall health status and component-level details.
    """
    components: List[ComponentHealth] = []
    overall_status = "healthy"

    # Check validator
    try:
        validator = await get_validator()
        components.append(
            ComponentHealth(
                name="validator",
                status="healthy",
                message="Validator is operational",
            )
        )
    except Exception as e:
        components.append(
            ComponentHealth(
                name="validator",
                status="unhealthy",
                message=str(e),
            )
        )
        overall_status = "degraded"

    # Check registry
    try:
        registry = await get_registry()
        if registry is not None:
            components.append(
                ComponentHealth(
                    name="registry",
                    status="healthy",
                    message="Registry is configured",
                )
            )
        else:
            components.append(
                ComponentHealth(
                    name="registry",
                    status="healthy",
                    message="No registry configured (using inline schemas)",
                )
            )
    except Exception as e:
        components.append(
            ComponentHealth(
                name="registry",
                status="degraded",
                message=str(e),
            )
        )

    # Get metrics for cache info
    metrics_data = metrics.get_metrics()

    return HealthResponse(
        status=overall_status,
        version=SERVICE_VERSION,
        cache_size=metrics_data.get("cache_size", 0),
        uptime_seconds=get_uptime_seconds(),
        components=components,
        timestamp=datetime.now(timezone.utc),
    )


@system_router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get metrics",
    description="Get Prometheus-compatible metrics for monitoring.",
    responses={
        200: {"description": "Current metrics"},
    },
)
async def get_metrics_endpoint(
    metrics: MetricsCollector = Depends(get_metrics),
) -> MetricsResponse:
    """
    Get service metrics for Prometheus/Grafana monitoring.

    Returns validation counts, cache performance, and latency statistics.
    """
    metrics_data = metrics.get_metrics()

    return MetricsResponse(
        validations_total=metrics_data.get("validations_total", 0),
        validations_success=metrics_data.get("validations_success", 0),
        validations_failed=metrics_data.get("validations_failed", 0),
        batch_validations_total=metrics_data.get("batch_validations_total", 0),
        batch_items_total=metrics_data.get("batch_items_total", 0),
        cache_hits=metrics_data.get("cache_hits", 0),
        cache_misses=metrics_data.get("cache_misses", 0),
        cache_size=metrics_data.get("cache_size", 0),
        avg_validation_time_ms=metrics_data.get("avg_validation_time_ms", 0.0),
        p95_validation_time_ms=metrics_data.get("p95_validation_time_ms", 0.0),
        uptime_seconds=metrics_data.get("uptime_seconds", 0.0),
    )


@system_router.get(
    "/metrics/prometheus",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
    description="Get metrics in Prometheus text format.",
    responses={
        200: {"description": "Prometheus format metrics"},
    },
)
async def get_prometheus_metrics(
    metrics: MetricsCollector = Depends(get_metrics),
) -> PlainTextResponse:
    """
    Get metrics in Prometheus text exposition format.

    Compatible with Prometheus scraping.
    """
    metrics_data = metrics.get_metrics()

    # Build Prometheus format
    lines = [
        "# HELP glschema_validations_total Total number of validation requests",
        "# TYPE glschema_validations_total counter",
        f'glschema_validations_total {metrics_data.get("validations_total", 0)}',
        "",
        "# HELP glschema_validations_success Number of successful validations",
        "# TYPE glschema_validations_success counter",
        f'glschema_validations_success {metrics_data.get("validations_success", 0)}',
        "",
        "# HELP glschema_validations_failed Number of failed validations",
        "# TYPE glschema_validations_failed counter",
        f'glschema_validations_failed {metrics_data.get("validations_failed", 0)}',
        "",
        "# HELP glschema_batch_validations_total Total batch validation requests",
        "# TYPE glschema_batch_validations_total counter",
        f'glschema_batch_validations_total {metrics_data.get("batch_validations_total", 0)}',
        "",
        "# HELP glschema_batch_items_total Total items validated in batches",
        "# TYPE glschema_batch_items_total counter",
        f'glschema_batch_items_total {metrics_data.get("batch_items_total", 0)}',
        "",
        "# HELP glschema_cache_hits Schema cache hits",
        "# TYPE glschema_cache_hits counter",
        f'glschema_cache_hits {metrics_data.get("cache_hits", 0)}',
        "",
        "# HELP glschema_cache_misses Schema cache misses",
        "# TYPE glschema_cache_misses counter",
        f'glschema_cache_misses {metrics_data.get("cache_misses", 0)}',
        "",
        "# HELP glschema_cache_size Current number of cached schemas",
        "# TYPE glschema_cache_size gauge",
        f'glschema_cache_size {metrics_data.get("cache_size", 0)}',
        "",
        "# HELP glschema_validation_time_avg_ms Average validation time in milliseconds",
        "# TYPE glschema_validation_time_avg_ms gauge",
        f'glschema_validation_time_avg_ms {metrics_data.get("avg_validation_time_ms", 0.0)}',
        "",
        "# HELP glschema_validation_time_p95_ms 95th percentile validation time",
        "# TYPE glschema_validation_time_p95_ms gauge",
        f'glschema_validation_time_p95_ms {metrics_data.get("p95_validation_time_ms", 0.0)}',
        "",
        "# HELP glschema_uptime_seconds Service uptime in seconds",
        "# TYPE glschema_uptime_seconds gauge",
        f'glschema_uptime_seconds {metrics_data.get("uptime_seconds", 0.0)}',
    ]

    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/plain; version=0.0.4",
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "router",
    "system_router",
    "validate_payload",
    "validate_batch",
    "compile_schema",
    "list_schema_versions",
    "get_schema",
    "health_check",
    "get_metrics_endpoint",
    "get_prometheus_metrics",
]
