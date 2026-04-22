# -*- coding: utf-8 -*-
"""Exception hierarchy for the Factors Python SDK.

All SDK errors inherit from :class:`FactorsAPIError`, which in turn
subclasses :class:`greenlang.utilities.exceptions.base.GreenLangException`
so they surface cleanly in the unified GreenLang error taxonomy.

Mapping HTTP status -> exception class::

    400 -> ValidationError
    401 -> AuthError
    403 -> TierError (or LicenseError for factor license issues)
    404 -> FactorNotFoundError (on /factors/{id}; otherwise FactorsAPIError)
    422 -> ValidationError
    429 -> RateLimitError
    5xx -> FactorsAPIError
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from greenlang.utilities.exceptions.base import GreenLangException

logger = logging.getLogger(__name__)


class FactorsAPIError(GreenLangException):
    """Base class for all Factors SDK errors.

    Carries the HTTP status code and raw response body (when available)
    alongside the GreenLangException context dict.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
        request_id: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx: Dict[str, Any] = dict(context or {})
        if status_code is not None:
            ctx["status_code"] = status_code
        if response_body is not None:
            ctx["response_body"] = response_body
        if request_id is not None:
            ctx["request_id"] = request_id
        super().__init__(message, error_code=error_code, context=ctx)
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id


class AuthError(FactorsAPIError):
    """401 — authentication failed (missing/invalid JWT or API key)."""


class TierError(FactorsAPIError):
    """403 — caller's tier is insufficient for the requested endpoint."""


class LicenseError(FactorsAPIError):
    """403 — factor is ``connector_only`` and caller lacks permission."""


class FactorNotFoundError(FactorsAPIError):
    """404 — requested factor_id does not exist in the edition."""


class ValidationError(FactorsAPIError):
    """400/422 — request payload or query parameters are invalid."""


class RateLimitError(FactorsAPIError):
    """429 — caller exceeded the tier rate limit.

    Exposes the retry-after seconds when the server supplied it so
    the transport layer and user code can back off politely.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[float] = None,
        status_code: Optional[int] = 429,
        response_body: Optional[Any] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = dict(context or {})
        if retry_after is not None:
            ctx["retry_after"] = retry_after
        super().__init__(
            message,
            status_code=status_code,
            response_body=response_body,
            request_id=request_id,
            context=ctx,
        )
        self.retry_after = retry_after


class EditionMismatchError(FactorsAPIError):
    """Client pinned edition X, server returned a different edition Y.

    Raised by :class:`FactorsClient` (and :class:`AsyncFactorsClient`)
    when a response carries an ``X-GreenLang-Edition`` /
    ``X-Factors-Edition`` header whose value disagrees with the pin set
    via :meth:`FactorsClient.pin_edition` or the ``client.edition(...)``
    context manager.

    We deliberately do NOT silently accept the drifted edition: the whole
    point of pinning is reproducibility, so a mismatch is a hard fail.
    The ``pinned_edition`` and ``returned_edition`` attributes let the
    caller decide whether to retry, fall through to the default edition,
    or surface the error to the user.
    """

    def __init__(
        self,
        message: str,
        *,
        pinned_edition: Optional[str] = None,
        returned_edition: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx: Dict[str, Any] = dict(context or {})
        if pinned_edition is not None:
            ctx["pinned_edition"] = pinned_edition
        if returned_edition is not None:
            ctx["returned_edition"] = returned_edition
        super().__init__(
            message,
            status_code=status_code,
            response_body=response_body,
            request_id=request_id,
            context=ctx,
        )
        self.pinned_edition = pinned_edition
        self.returned_edition = returned_edition


def error_from_response(
    *,
    status_code: int,
    url: str,
    body: Any,
    request_id: Optional[str] = None,
    retry_after: Optional[float] = None,
) -> FactorsAPIError:
    """Map an HTTP error response onto the SDK exception hierarchy.

    Args:
        status_code: HTTP status code.
        url: Fully-qualified request URL (for diagnostics).
        body: Decoded response body (may be ``str`` or ``dict``).
        request_id: Opaque request id pulled from ``X-Request-ID``.
        retry_after: Seconds hint from ``Retry-After`` header (429 only).

    Returns:
        The most specific SDK exception subclass for ``status_code``.
    """
    msg = _extract_detail(body) or f"HTTP {status_code} from {url}"

    if status_code == 401:
        return AuthError(
            msg,
            status_code=status_code,
            response_body=body,
            request_id=request_id,
        )
    if status_code == 403:
        detail = (msg or "").lower()
        if "license" in detail or "connector_only" in detail or "redistribution" in detail:
            return LicenseError(
                msg,
                status_code=status_code,
                response_body=body,
                request_id=request_id,
            )
        return TierError(
            msg,
            status_code=status_code,
            response_body=body,
            request_id=request_id,
        )
    if status_code == 404 and "/factors/" in url:
        return FactorNotFoundError(
            msg,
            status_code=status_code,
            response_body=body,
            request_id=request_id,
        )
    if status_code in (400, 422):
        return ValidationError(
            msg,
            status_code=status_code,
            response_body=body,
            request_id=request_id,
        )
    if status_code == 429:
        return RateLimitError(
            msg,
            retry_after=retry_after,
            status_code=status_code,
            response_body=body,
            request_id=request_id,
        )
    return FactorsAPIError(
        msg,
        status_code=status_code,
        response_body=body,
        request_id=request_id,
    )


def _extract_detail(body: Any) -> Optional[str]:
    """Pull a human-readable detail field out of a FastAPI error body."""
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("message")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict):
                return first.get("msg") or first.get("message")
            return str(first)
        if isinstance(detail, dict):
            return detail.get("msg") or detail.get("message")
    if isinstance(body, str) and body.strip():
        return body
    return None


__all__ = [
    "FactorsAPIError",
    "AuthError",
    "TierError",
    "LicenseError",
    "FactorNotFoundError",
    "ValidationError",
    "RateLimitError",
    "EditionMismatchError",
    "error_from_response",
]
