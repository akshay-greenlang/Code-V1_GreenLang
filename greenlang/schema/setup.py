# -*- coding: utf-8 -*-
"""
Schema Service Setup - AGENT-FOUND-002: GreenLang Schema Compiler & Validator

Provides ``configure_schema_service(app)`` which wires up the Schema
validation engine (compiler, validator, registry, IR cache) and mounts
the REST API.

Also exposes ``get_schema_service(app)`` for programmatic access and
the ``SchemaService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.schema.setup import configure_schema_service
    >>> app = FastAPI()
    >>> configure_schema_service(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-002 Schema Compiler & Validator
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

from greenlang.schema.compiler.compiler import SchemaCompiler
from greenlang.schema.models.report import (
    BatchValidationReport,
    ValidationReport,
)
from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.registry.cache import IRCacheService
from greenlang.schema.registry.resolver import SchemaRegistry
from greenlang.schema.sdk import (
    CompiledSchema,
    PayloadInput,
    SchemaInput,
    compile_schema,
    validate,
    validate_batch,
)
from greenlang.schema.metrics import (
    PROMETHEUS_AVAILABLE,
    record_validation,
    record_compilation,
    update_registered_schemas,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# SchemaService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["SchemaService"] = None


class SchemaService:
    """Unified facade over the Schema validation engine.

    Manages schema validation, compilation, registry access, IR
    cache lifecycle, and Prometheus metrics. This class provides a
    single entry point for all schema operations.

    Attributes:
        compiler: Schema compiler engine.
        registry: Schema registry for resolving references.
        ir_cache: IR cache service for compiled schema caching.

    Example:
        >>> service = SchemaService()
        >>> result = service.validate(payload, "gl://schemas/activity@1.0.0")
        >>> if result.valid:
        ...     print("Payload is valid")
    """

    def __init__(
        self,
        registry: Optional[SchemaRegistry] = None,
        ir_cache: Optional[IRCacheService] = None,
    ) -> None:
        """Initialize the Schema Service facade.

        Args:
            registry: Optional schema registry (creates default if None).
            ir_cache: Optional IR cache service (creates default if None).
        """
        self.compiler = SchemaCompiler()
        self.registry = registry or SchemaRegistry()
        self.ir_cache = ir_cache or IRCacheService()
        self._started = False

        logger.info("SchemaService facade created")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        payload: PayloadInput,
        schema: SchemaInput,
        **kwargs: Any,
    ) -> ValidationReport:
        """Validate a payload against a schema.

        Delegates to the SDK validate() function with the configured
        registry for schema resolution.

        Args:
            payload: Payload to validate (YAML/JSON string or dict).
            schema: Schema reference, URI string, or inline dict.
            **kwargs: Additional options passed to sdk.validate().

        Returns:
            ValidationReport with validation results.
        """
        return validate(
            payload=payload,
            schema=schema,
            registry=self.registry,
            **kwargs,
        )

    def validate_batch(
        self,
        payloads: Sequence[PayloadInput],
        schema: SchemaInput,
        **kwargs: Any,
    ) -> BatchValidationReport:
        """Validate multiple payloads against a schema.

        Delegates to the SDK validate_batch() function with the
        configured registry.

        Args:
            payloads: Sequence of payloads to validate.
            schema: Schema reference, URI string, or inline dict.
            **kwargs: Additional options passed to sdk.validate_batch().

        Returns:
            BatchValidationReport with results for all payloads.
        """
        return validate_batch(
            payloads=payloads,
            schema=schema,
            registry=self.registry,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile_schema(
        self,
        schema: SchemaInput,
    ) -> CompiledSchema:
        """Pre-compile a schema for efficient repeated validation.

        Delegates to the SDK compile_schema() function with the
        configured registry.

        Args:
            schema: Schema reference, URI string, or inline dict.

        Returns:
            CompiledSchema object for efficient validation.

        Raises:
            ValueError: If schema compilation fails.
        """
        return compile_schema(
            schema=schema,
            registry=self.registry,
        )

    # ------------------------------------------------------------------
    # Registry access
    # ------------------------------------------------------------------

    def get_registry(self) -> SchemaRegistry:
        """Get the schema registry.

        Returns:
            The SchemaRegistry instance used by this service.
        """
        return self.registry

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get schema service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        cache_metrics = self.ir_cache.get_metrics()
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "ir_cache": {
                "total_entries": cache_metrics.total_entries,
                "total_size_bytes": cache_metrics.total_size_bytes,
                "hit_count": cache_metrics.hit_count,
                "miss_count": cache_metrics.miss_count,
                "hit_rate": cache_metrics.hit_rate,
            },
            "started": self._started,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the schema service.

        Warms up the IR cache with commonly used schemas and sets
        the started flag. Safe to call multiple times.
        """
        if self._started:
            logger.debug("SchemaService already started; skipping startup")
            return

        logger.info("SchemaService starting up...")

        # Warm up IR cache with popular schemas
        try:
            self.ir_cache.start()
            logger.info("IR cache warm-up started")
        except Exception as exc:
            logger.warning(
                "IR cache warm-up failed (non-fatal): %s", exc
            )

        self._started = True
        logger.info("SchemaService startup complete")

    def shutdown(self) -> None:
        """Shutdown the schema service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("SchemaService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> SchemaService:
    """Get or create the singleton SchemaService instance.

    Returns:
        The singleton SchemaService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = SchemaService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


def configure_schema_service(
    app: Any,
    registry: Optional[SchemaRegistry] = None,
    ir_cache: Optional[IRCacheService] = None,
) -> SchemaService:
    """Configure the Schema Service on a FastAPI application.

    Creates the SchemaService, stores it in app.state, mounts
    the schema API router, and starts IR cache warm-up.

    Args:
        app: FastAPI application instance.
        registry: Optional schema registry.
        ir_cache: Optional IR cache service.

    Returns:
        SchemaService instance.
    """
    global _singleton_instance

    service = SchemaService(registry=registry, ir_cache=ir_cache)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.schema_service = service

    # Mount schema API router
    try:
        from greenlang.schema.api.routes import router as schema_router
        if schema_router is not None:
            app.include_router(schema_router)
            logger.info("Schema service API router mounted")
    except ImportError:
        logger.warning("Schema router not available; API not mounted")

    # Start IR cache warm-up
    service.startup()

    logger.info("Schema service configured on app")
    return service


def get_schema_service(app: Any) -> SchemaService:
    """Get the SchemaService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        SchemaService instance.

    Raises:
        RuntimeError: If schema service not configured.
    """
    service = getattr(app.state, "schema_service", None)
    if service is None:
        raise RuntimeError(
            "Schema service not configured. "
            "Call configure_schema_service(app) first."
        )
    return service


__all__ = [
    "SchemaService",
    "configure_schema_service",
    "get_schema_service",
]
