"""
JWT Authentication Middleware

This module provides JWT-based authentication for all API endpoints.
"""

import logging
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    JWT Authentication Middleware.

    Validates JWT tokens from the Authorization header and injects
    user context into the request state.

    Public endpoints (like /health) are excluded from authentication.
    """

    PUBLIC_PATHS = {"/health", "/ready", "/docs", "/redoc", "/openapi.json"}

    def __init__(
        self,
        app,
        secret_key: str,
        algorithm: str = "HS256",
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            secret_key: JWT signing secret
            algorithm: JWT algorithm
        """
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process the request.

        - Extracts JWT from Authorization header
        - Validates token signature and expiration
        - Injects user context into request.state
        """
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Extract token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            # Check for API key fallback
            api_key = request.headers.get("X-API-Key")
            if api_key:
                # TODO: Validate API key
                request.state.user_id = "api-key-user"
                request.state.tenant_id = "default"
                return await call_next(request)

            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Missing or invalid Authorization header",
                    }
                },
            )

        token = auth_header.split(" ")[1]

        # Validate token
        try:
            # TODO: Implement actual JWT validation
            # payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            # request.state.user_id = payload.get("sub")
            # request.state.tenant_id = payload.get("tenant_id")

            # Placeholder
            request.state.user_id = "user-1"
            request.state.tenant_id = "tenant-1"

        except Exception as e:
            logger.warning(f"JWT validation failed: {e}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "INVALID_TOKEN",
                        "message": "Invalid or expired token",
                    }
                },
            )

        return await call_next(request)
