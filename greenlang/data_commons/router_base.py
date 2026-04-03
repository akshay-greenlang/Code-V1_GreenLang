# -*- coding: utf-8 -*-
"""
Standard FastAPI Router Base for GreenLang Data Agents

Eliminates ~100-150 lines of identical boilerplate from each of the 15+
``api/router.py`` files across the data-layer agents by providing:

1.  **FASTAPI_AVAILABLE** -- the try/except import guard that every router
    duplicates for ``fastapi``, ``HTTPException``, ``Query``, ``Request``,
    ``JSONResponse``, and ``pydantic`` components.
2.  **create_standard_router()** -- a factory that creates an ``APIRouter``
    with a ``/health`` endpoint and an optional ``/v1/statistics`` endpoint,
    both of which are copy-pasted verbatim in every router module.
3.  **get_service_dependency()** -- a factory that builds the ``_get_service``
    helper function, eliminating the duplicate 10-line function in every
    router that pulls a service from ``request.app.state``.

All 15+ agent routers share this identical pattern::

    try:
        from fastapi import APIRouter, HTTPException, Query, Request
        from fastapi.responses import JSONResponse
        FASTAPI_AVAILABLE = True
    except ImportError:
        FASTAPI_AVAILABLE = False
        APIRouter = None

    # ... router = APIRouter(prefix=..., tags=[...])
    # ... _get_service() that reads request.app.state.{service_name}_service
    # ... @router.get("/health") returning {"status": "healthy", "service": ...}
    # ... @router.get("/v1/statistics") delegating to service.get_statistics()

This module centralises that boilerplate so each agent router file can
focus solely on its domain-specific endpoints.

Typical usage in an agent's ``api/router.py``::

    from greenlang.data_commons.router_base import (
        FASTAPI_AVAILABLE,
        create_standard_router,
        get_service_dependency,
    )

    if FASTAPI_AVAILABLE:
        from fastapi import Request
        from greenlang.data_commons.router_base import APIRouter, HTTPException, Query

    router, _health_registered = create_standard_router(
        service_name="pdf-extractor",
        prefix="/api/v1/pdf-extractor",
        tags=["pdf-extractor"],
    )

    _get_service = get_service_dependency(
        state_attr="pdf_extractor_service",
        service_display_name="PDF extractor",
    )

    # ... only domain-specific endpoints follow ...

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful FastAPI import (shared across all data agent routers)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Provide None stubs so downstream modules can do ``if FASTAPI_AVAILABLE:``
    # without NameError on the references.
    APIRouter = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    Query = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment,misc]
    logger.info(
        "FastAPI not installed; router_base utilities return None routers"
    )


# ---------------------------------------------------------------------------
# Standard router factory
# ---------------------------------------------------------------------------


def create_standard_router(
    service_name: str,
    *,
    prefix: str = "",
    tags: Optional[List[str]] = None,
    service_version: str = "1.0.0",
    include_health: bool = True,
    include_statistics: bool = False,
    statistics_state_attr: str = "",
    statistics_service_display: str = "",
) -> Tuple[Optional[Any], bool]:
    """Create a standard FastAPI ``APIRouter`` with health and optional stats endpoints.

    This factory encapsulates the router creation pattern used identically in
    all 15+ data agent ``api/router.py`` files, including the ``/health``
    endpoint that returns ``{"status": "healthy", "service": service_name}``
    and the optional ``/v1/statistics`` endpoint that delegates to the
    service's ``get_statistics()`` method.

    Args:
        service_name: Kebab-case service identifier (e.g. ``"pdf-extractor"``).
            Returned in the health check response body.
        prefix: URL prefix for the router (e.g. ``"/api/v1/pdf-extractor"``).
        tags: OpenAPI tags for the router.  Defaults to ``[service_name]``.
        service_version: Version string returned in the health response.
        include_health: Whether to register a ``GET /health`` endpoint.
            Defaults to ``True``.
        include_statistics: Whether to register a ``GET /v1/statistics``
            endpoint that calls ``service.get_statistics()``.  Requires
            ``statistics_state_attr`` to be set.
        statistics_state_attr: The ``request.app.state`` attribute name for
            the service instance used by the statistics endpoint.
        statistics_service_display: Human-readable name for the service
            (used in the 503 error message if the service is missing).

    Returns:
        A 2-tuple of ``(router, endpoints_registered)``.  ``router`` is
        ``None`` when FastAPI is not available; ``endpoints_registered`` is
        ``True`` when at least one standard endpoint was added.

    Example:
        >>> router, _ = create_standard_router(
        ...     "pdf-extractor",
        ...     prefix="/api/v1/pdf-extractor",
        ...     tags=["pdf-extractor"],
        ... )
    """
    if not FASTAPI_AVAILABLE:
        return None, False

    resolved_tags = tags if tags is not None else [service_name]
    router = APIRouter(prefix=prefix, tags=resolved_tags)
    registered = False

    if include_health:

        @router.get("/health", tags=["Health"])
        async def health() -> Dict[str, str]:
            """Service health check endpoint."""
            return {
                "status": "healthy",
                "service": service_name,
                "version": service_version,
            }

        registered = True

    if include_statistics and statistics_state_attr:

        @router.get("/v1/statistics", tags=["Statistics"])
        async def get_statistics(request: Request) -> Dict[str, Any]:  # type: ignore[arg-type]
            """Get service statistics."""
            service = getattr(request.app.state, statistics_state_attr, None)
            if service is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"{statistics_service_display or service_name} service not configured",
                )
            stats = service.get_statistics()
            if hasattr(stats, "model_dump"):
                return stats.model_dump(mode="json")
            return stats

        registered = True

    return router, registered


# ---------------------------------------------------------------------------
# Service dependency factory
# ---------------------------------------------------------------------------


def get_service_dependency(
    state_attr: str,
    service_display_name: str = "",
) -> Callable[..., Any]:
    """Create a ``_get_service(request)`` helper for a router module.

    Every data agent router defines an identical ``_get_service()`` function
    that reads the service instance from ``request.app.state.<attr>`` and
    raises a 503 ``HTTPException`` if it is ``None``.  This factory
    produces that function with the correct attribute name baked in.

    Args:
        state_attr: Attribute name on ``request.app.state`` that holds
            the service instance (e.g. ``"pdf_extractor_service"``).
        service_display_name: Human-readable name for 503 error messages
            (e.g. ``"PDF extractor"``).  Defaults to *state_attr* with
            underscores replaced by spaces.

    Returns:
        A callable ``_get_service(request: Request) -> Any``.

    Example:
        >>> _get_service = get_service_dependency(
        ...     "pdf_extractor_service", "PDF extractor",
        ... )
    """
    display = service_display_name or state_attr.replace("_", " ")

    def _get_service(request: Any) -> Any:
        """Extract the service instance from FastAPI app state.

        Args:
            request: FastAPI ``Request`` object.

        Returns:
            The service instance.

        Raises:
            HTTPException: 503 if the service is not configured.
        """
        service = getattr(request.app.state, state_attr, None)
        if service is None:
            if FASTAPI_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail=f"{display} service not configured",
                )
            raise RuntimeError(f"{display} service not configured")
        return service

    _get_service.__qualname__ = f"_get_service[{state_attr}]"
    _get_service.__doc__ = (
        f"Extract {display} service from ``request.app.state.{state_attr}``."
    )
    return _get_service


# ---------------------------------------------------------------------------
# Convenience: safe error response builder
# ---------------------------------------------------------------------------


def error_response(status_code: int, detail: str) -> Any:
    """Raise an ``HTTPException`` if FastAPI is available, else a ``ValueError``.

    This is a convenience wrapper for code that may run both with and without
    FastAPI installed.

    Args:
        status_code: HTTP status code (e.g. 400, 404, 503).
        detail: Error detail message.

    Raises:
        HTTPException: When FastAPI is available.
        ValueError: When FastAPI is not available.
    """
    if FASTAPI_AVAILABLE:
        raise HTTPException(status_code=status_code, detail=detail)
    raise ValueError(detail)


__all__ = [
    "FASTAPI_AVAILABLE",
    "APIRouter",
    "HTTPException",
    "Query",
    "Request",
    "JSONResponse",
    "Field",
    "create_standard_router",
    "get_service_dependency",
    "error_response",
]
