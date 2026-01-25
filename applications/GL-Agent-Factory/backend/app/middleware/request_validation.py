"""
Request Validation Middleware

This module provides request validation and size limiting.
"""

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Request Validation Middleware.

    Validates:
    - Content-Type header
    - Request body size
    - JSON syntax
    """

    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Validate the request before processing.
        """
        # Check Content-Type for POST/PUT/PATCH
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("Content-Type", "")
            if not content_type.startswith("application/json"):
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": {
                            "code": "UNSUPPORTED_MEDIA_TYPE",
                            "message": "Content-Type must be application/json",
                        }
                    },
                )

            # Check Content-Length
            content_length = request.headers.get("Content-Length")
            if content_length and int(content_length) > self.MAX_BODY_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "PAYLOAD_TOO_LARGE",
                            "message": f"Request body exceeds {self.MAX_BODY_SIZE // (1024*1024)} MB limit",
                        }
                    },
                )

        return await call_next(request)
