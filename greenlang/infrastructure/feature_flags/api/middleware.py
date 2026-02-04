# -*- coding: utf-8 -*-
"""
Feature Flags Middleware - INFRA-008

FastAPI middleware that automatically evaluates feature flags on each request
and injects the results into the ExecutionContext.features dictionary. This
enables downstream agents and handlers to use ``context.is_feature_enabled()``
without manual flag evaluation.

Request flow:
    1. Extract user_id, tenant_id from JWT claims in Authorization header
       (JWT is decoded without verification - Kong has already verified it)
    2. Build EvaluationContext from the request
    3. Call service.evaluate_all(context)
    4. Inject results into ExecutionContext.features
    5. Add X-Feature-Flags response header with active flag keys

The middleware gracefully handles errors: if flag evaluation fails, the
request continues with an empty features dictionary.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.feature_flags.api.middleware import FeatureFlagMiddleware
    >>> app = FastAPI()
    >>> app.add_middleware(
    ...     FeatureFlagMiddleware,
    ...     enabled=True,
    ...     skip_paths=["/health", "/ready", "/metrics"],
    ... )
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from greenlang.infrastructure.feature_flags.models import EvaluationContext
from greenlang.infrastructure.feature_flags.service import get_feature_flag_service

try:
    from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
    from starlette.requests import Request
    from starlette.responses import Response

    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    BaseHTTPMiddleware = object  # type: ignore[misc, assignment]
    RequestResponseEndpoint = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    Response = None  # type: ignore[misc, assignment]

# Conditional import of ExecutionContext - may not always be available
try:
    from greenlang.execution.core.context import (
        ExecutionContext,
        get_current_context,
        set_current_context,
    )

    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False
    ExecutionContext = None  # type: ignore[misc, assignment]
    get_current_context = None  # type: ignore[assignment]
    set_current_context = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default paths to skip (health checks, readiness, metrics)
DEFAULT_SKIP_PATHS: Set[str] = {
    "/health",
    "/ready",
    "/readyz",
    "/healthz",
    "/livez",
    "/metrics",
    "/openapi.json",
    "/docs",
    "/redoc",
}


class FeatureFlagMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Middleware that evaluates feature flags on each HTTP request.

    Extracts identity from JWT claims, evaluates all active flags, and
    injects results into the ExecutionContext so downstream code can use
    ``context.is_feature_enabled("flag-key")``.

    Attributes:
        enabled: Whether the middleware is active.
        skip_paths: Paths to skip flag evaluation for.
    """

    def __init__(
        self,
        app: Any,
        enabled: bool = True,
        skip_paths: Optional[List[str]] = None,
    ) -> None:
        """Initialize the feature flag middleware.

        Args:
            app: The ASGI application.
            enabled: Whether to enable flag evaluation. Set to False to
                disable without removing the middleware.
            skip_paths: List of URL paths to skip. Defaults to health
                check and documentation endpoints.
        """
        super().__init__(app)
        self.enabled = enabled
        self.skip_paths: Set[str] = set(skip_paths) if skip_paths else DEFAULT_SKIP_PATHS
        logger.info(
            "FeatureFlagMiddleware initialized (enabled=%s, skip_paths=%d)",
            self.enabled, len(self.skip_paths),
        )

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process each request through the feature flag evaluation pipeline.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response with X-Feature-Flags header.
        """
        # Short-circuit if disabled or path should be skipped
        if not self.enabled or self._should_skip(request):
            return await call_next(request)

        start_time = time.monotonic()
        feature_flags: Dict[str, bool] = {}

        try:
            # Extract identity from JWT
            user_id, tenant_id = self._extract_jwt_claims(request)

            # Build evaluation context
            context = EvaluationContext(
                user_id=user_id,
                tenant_id=tenant_id,
                environment=self._get_environment(request),
                user_segments=self._extract_segments(request),
                user_attributes=self._extract_attributes(request),
            )

            # Evaluate all flags
            service = get_feature_flag_service()
            feature_flags = await service.evaluate_all(context)

            # Inject into ExecutionContext if available
            self._inject_into_execution_context(feature_flags)

            duration_ms = (time.monotonic() - start_time) * 1000
            logger.debug(
                "Feature flags evaluated: %d flags in %.2fms (user=%s, tenant=%s)",
                len(feature_flags), duration_ms, user_id, tenant_id,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "Feature flag evaluation failed (%.2fms), continuing with empty flags: %s",
                duration_ms, exc,
            )
            # Continue with empty flags - do not fail the request

        # Process the request
        response = await call_next(request)

        # Add feature flags header to response
        active_flags = [k for k, v in feature_flags.items() if v]
        if active_flags:
            response.headers["X-Feature-Flags"] = json.dumps(active_flags)

        return response

    def _should_skip(self, request: Request) -> bool:
        """Check if the request path should skip flag evaluation.

        Args:
            request: The HTTP request.

        Returns:
            True if the path is in the skip list.
        """
        path = request.url.path.rstrip("/")
        return path in self.skip_paths

    @staticmethod
    def _extract_jwt_claims(request: Request) -> tuple[Optional[str], Optional[str]]:
        """Extract user_id and tenant_id from JWT in Authorization header.

        The JWT is decoded WITHOUT verification because Kong has already
        verified the token signature and validity. We only need to extract
        the claims for feature flag targeting.

        Args:
            request: The HTTP request.

        Returns:
            Tuple of (user_id, tenant_id). Either may be None.
        """
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return None, None

        token = auth_header[7:].strip()
        if not token:
            return None, None

        try:
            # JWT has 3 parts: header.payload.signature
            parts = token.split(".")
            if len(parts) < 2:
                return None, None

            # Decode the payload (base64url)
            payload_b64 = parts[1]
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            claims = json.loads(payload_bytes)

            user_id = claims.get("sub") or claims.get("user_id")
            tenant_id = claims.get("tenant_id") or claims.get("org_id")

            return str(user_id) if user_id else None, str(tenant_id) if tenant_id else None

        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.debug("Failed to decode JWT claims: %s", exc)
            return None, None

    @staticmethod
    def _get_environment(request: Request) -> str:
        """Determine the environment from request headers or defaults.

        Args:
            request: The HTTP request.

        Returns:
            Environment string, defaults to "dev".
        """
        # Check for environment header (set by Kong or load balancer)
        env = request.headers.get("x-environment", "")
        if env:
            return env.strip().lower()

        # Check for host-based detection
        host = request.headers.get("host", "")
        if "staging" in host:
            return "staging"
        if "prod" in host or "api." in host:
            return "prod"

        return "dev"

    @staticmethod
    def _extract_segments(request: Request) -> List[str]:
        """Extract user segments from request headers.

        Kong custom plugins or upstream middleware may inject segment
        information via the X-User-Segments header.

        Args:
            request: The HTTP request.

        Returns:
            List of segment strings.
        """
        segments_header = request.headers.get("x-user-segments", "")
        if not segments_header:
            return []
        return [s.strip().lower() for s in segments_header.split(",") if s.strip()]

    @staticmethod
    def _extract_attributes(request: Request) -> Dict[str, Any]:
        """Extract user attributes from request headers and query params.

        Args:
            request: The HTTP request.

        Returns:
            Dictionary of user attributes for targeting.
        """
        attributes: Dict[str, Any] = {}

        # Extract from X-User-Attributes header (JSON)
        attr_header = request.headers.get("x-user-attributes", "")
        if attr_header:
            try:
                attributes.update(json.loads(attr_header))
            except json.JSONDecodeError:
                pass

        # Add request metadata as attributes
        attributes["request_method"] = request.method
        attributes["request_path"] = request.url.path

        return attributes

    @staticmethod
    def _inject_into_execution_context(flags: Dict[str, bool]) -> None:
        """Inject evaluated flags into the current ExecutionContext.

        Args:
            flags: Mapping of flag_key -> enabled boolean.
        """
        if not CONTEXT_AVAILABLE:
            return

        ctx = get_current_context()
        if ctx is not None:
            ctx.features.update(flags)
            logger.debug(
                "Injected %d feature flags into ExecutionContext", len(flags)
            )


__all__ = ["FeatureFlagMiddleware"]
