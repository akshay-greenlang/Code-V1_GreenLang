# -*- coding: utf-8 -*-
"""
EditionPinMiddleware — pins each request to a specific edition manifest.

Resolution order: ``X-GL-Edition`` header → ``?edition=`` query → service
default. Stashes the resolved id on ``request.state.edition_id`` and echoes
``X-GreenLang-Edition`` on the response so the signed-receipts middleware
folds it into the receipt signature (CTO non-negotiable: never overwrite a
factor — every response carries its edition).
"""

from __future__ import annotations

import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class EditionPinMiddleware(BaseHTTPMiddleware):
    HEADER_NAME = "X-GL-Edition"
    QUERY_NAME = "edition"
    RESPONSE_HEADER = "X-GreenLang-Edition"

    async def dispatch(self, request: Request, call_next):
        edition_id = self._resolve_edition(request)
        request.state.edition_id = edition_id
        try:
            response: Response = await call_next(request)
        except _UnknownEditionError as exc:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "edition_not_found",
                    "message": str(exc),
                    "requested_edition": exc.edition_id,
                },
            )
        if edition_id:
            response.headers[self.RESPONSE_HEADER] = edition_id
        return response

    def _resolve_edition(self, request: Request) -> Optional[str]:
        header = request.headers.get(self.HEADER_NAME)
        if header:
            return self._validate(request, header.strip(), "header")
        query = request.query_params.get(self.QUERY_NAME)
        if query:
            return self._validate(request, query.strip(), "query")
        return self._service_default(request)

    def _service_default(self, request: Request) -> Optional[str]:
        service = getattr(request.app.state, "factors_service", None)
        if service is None or not hasattr(service, "repo"):
            return None
        try:
            return service.repo.resolve_edition(None)
        except Exception:  # noqa: BLE001
            return None

    def _validate(self, request: Request, edition_id: str, source: str) -> str:
        service = getattr(request.app.state, "factors_service", None)
        if service is None or not hasattr(service, "repo"):
            return edition_id  # cannot validate; trust header (test mode)
        try:
            return service.repo.resolve_edition(edition_id)
        except Exception as exc:  # noqa: BLE001
            logger.info("Edition pin %s=%s rejected: %s", source, edition_id, exc)
            raise _UnknownEditionError(edition_id, str(exc))


class _UnknownEditionError(Exception):
    def __init__(self, edition_id: str, message: str):
        super().__init__(f"Unknown edition '{edition_id}': {message}")
        self.edition_id = edition_id
