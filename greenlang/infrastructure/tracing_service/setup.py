# -*- coding: utf-8 -*-
"""
configure_tracing - One-liner tracing setup for GreenLang services (OBS-003)

Provides a single function that initialises the entire tracing stack:
1. TracerProvider with OTLP exporter and GreenLangSampler.
2. Auto-instrumentors for FastAPI, httpx, psycopg, Redis, Celery, requests.
3. TracingMiddleware on the FastAPI app (if provided).
4. Shutdown hook registered on the app's lifespan.

Designed to be called once during application startup.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.tracing_service import configure_tracing
    >>>
    >>> app = FastAPI()
    >>> configure_tracing(app, service_name="api-service")

    Or without FastAPI:
    >>> from greenlang.infrastructure.tracing_service import configure_tracing
    >>> config = configure_tracing(service_name="batch-worker")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from greenlang.infrastructure.tracing_service.config import TracingConfig
from greenlang.infrastructure.tracing_service.provider import (
    setup_provider,
    shutdown as shutdown_provider,
    OTEL_AVAILABLE,
)
from greenlang.infrastructure.tracing_service.instrumentors import setup_instrumentors
from greenlang.infrastructure.tracing_service.middleware import TracingMiddleware
from greenlang.infrastructure.tracing_service.span_enrichment import SpanEnricher
from greenlang.infrastructure.tracing_service.metrics_bridge import get_metrics_bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_initialized: bool = False
_active_config: Optional[TracingConfig] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_tracing(
    app: Any = None,
    *,
    service_name: str = "greenlang",
    config: Optional[TracingConfig] = None,
) -> TracingConfig:
    """Initialise the complete GreenLang tracing stack in one call.

    Safe to call multiple times -- subsequent calls return the existing
    config without re-initialising.

    Args:
        app: Optional FastAPI/Starlette application to attach middleware to.
        service_name: Service name for traces (ignored if *config* provides one).
        config: Optional pre-built TracingConfig.  When ``None``, a new
                config is created from environment variables.

    Returns:
        The active TracingConfig.
    """
    global _initialized, _active_config

    if _initialized and _active_config is not None:
        logger.debug("Tracing already initialised; returning existing config")
        return _active_config

    # -- Build config ----------------------------------------------------------
    if config is None:
        config = TracingConfig(service_name=service_name)
    elif config.service_name == "greenlang" and service_name != "greenlang":
        # Caller provided a config but also a service_name override
        config.service_name = service_name

    _active_config = config

    if not config.enabled:
        logger.info("Tracing disabled via configuration")
        _initialized = True
        return config

    # -- 1. Provider -----------------------------------------------------------
    setup_provider(config)

    # -- 2. Auto-instrumentors -------------------------------------------------
    instrumented = setup_instrumentors(config)
    logger.info("Instrumented libraries: %s", instrumented)

    # -- 3. Metrics bridge -----------------------------------------------------
    get_metrics_bridge(service_name=config.service_name)

    # -- 4. FastAPI middleware + shutdown hook ----------------------------------
    if app is not None:
        _attach_to_app(app, config)

    _initialized = True
    logger.info(
        "Tracing fully configured: service=%s env=%s otel=%s",
        config.service_name,
        config.environment,
        OTEL_AVAILABLE,
    )
    return config


def shutdown_tracing() -> None:
    """Flush and shut down the tracing stack.

    Call this during application shutdown if you are not using the
    automatic FastAPI shutdown hook.
    """
    global _initialized, _active_config

    shutdown_provider()
    _initialized = False
    _active_config = None
    logger.info("Tracing shut down")


def is_tracing_enabled() -> bool:
    """Check whether tracing has been initialised and is active.

    Returns:
        True if tracing is initialised and OTel is available.
    """
    return _initialized and OTEL_AVAILABLE


def get_active_config() -> Optional[TracingConfig]:
    """Return the active tracing configuration, if initialised.

    Returns:
        The TracingConfig or None.
    """
    return _active_config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _attach_to_app(app: Any, config: TracingConfig) -> None:
    """Attach TracingMiddleware and shutdown hook to a FastAPI app.

    Args:
        app: The FastAPI/Starlette application instance.
        config: Active tracing configuration.
    """
    enricher = SpanEnricher(environment=config.environment)

    # Attach ASGI middleware
    try:
        app.add_middleware(
            TracingMiddleware,
            service_name=config.service_name,
            enricher=enricher,
        )
        logger.info("TracingMiddleware attached to application")
    except Exception as exc:
        logger.warning("Failed to attach TracingMiddleware: %s", exc)

    # Register shutdown handler
    try:
        @app.on_event("shutdown")
        async def _on_shutdown() -> None:
            shutdown_provider()
            logger.info("Tracing shut down via app shutdown event")
    except Exception as exc:
        logger.debug(
            "Could not register shutdown event (may not be FastAPI): %s", exc
        )


__all__ = [
    "configure_tracing",
    "shutdown_tracing",
    "is_tracing_enabled",
    "get_active_config",
]
