# -*- coding: utf-8 -*-
"""
Error Classifications - AGENT-EUDR-026

Reference data for error classification taxonomy, retry strategies,
and fallback configuration for each EUDR agent. Maps HTTP status codes,
exception types, and error message patterns to error classifications
and appropriate recovery strategies.

Error Classification Matrix:
    Transient (retry with backoff):
        - HTTP 408 Request Timeout
        - HTTP 429 Too Many Requests
        - HTTP 500 Internal Server Error
        - HTTP 502 Bad Gateway
        - HTTP 503 Service Unavailable
        - HTTP 504 Gateway Timeout
        - Connection timeout/reset/refused
        - DNS resolution failure

    Permanent (fail immediately):
        - HTTP 400 Bad Request
        - HTTP 401 Unauthorized
        - HTTP 403 Forbidden
        - HTTP 404 Not Found
        - HTTP 405 Method Not Allowed
        - HTTP 409 Conflict
        - HTTP 410 Gone
        - HTTP 422 Unprocessable Entity
        - Data validation errors

    Degraded (use fallback):
        - Partial results returned
        - Low confidence scores
        - Stale cache data used
        - Incomplete agent output

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    ErrorClassification,
    FallbackStrategy,
)


# ---------------------------------------------------------------------------
# HTTP status code classification
# ---------------------------------------------------------------------------

#: HTTP status codes classified as transient errors (safe to retry).
TRANSIENT_HTTP_CODES: FrozenSet[int] = frozenset({
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
})

#: HTTP status codes classified as permanent errors (do not retry).
PERMANENT_HTTP_CODES: FrozenSet[int] = frozenset({
    400,  # Bad Request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found
    405,  # Method Not Allowed
    409,  # Conflict
    410,  # Gone
    422,  # Unprocessable Entity
})

#: HTTP status codes indicating partial success (degraded).
DEGRADED_HTTP_CODES: FrozenSet[int] = frozenset({
    206,  # Partial Content
    207,  # Multi-Status
    226,  # IM Used (partial)
})


# ---------------------------------------------------------------------------
# Exception type classification
# ---------------------------------------------------------------------------

#: Exception class names classified as transient.
TRANSIENT_EXCEPTIONS: FrozenSet[str] = frozenset({
    "TimeoutError",
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "ConnectionResetError",
    "BrokenPipeError",
    "TemporaryError",
    "ServiceUnavailableError",
    "TooManyRedirects",
    "RetryError",
    "OSError",
    "IOError",
})

#: Exception class names classified as permanent.
PERMANENT_EXCEPTIONS: FrozenSet[str] = frozenset({
    "ValueError",
    "TypeError",
    "KeyError",
    "AttributeError",
    "ValidationError",
    "PermissionError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "SchemaValidationError",
    "DataIntegrityError",
})


# ---------------------------------------------------------------------------
# Error message keyword classification
# ---------------------------------------------------------------------------

#: Keywords in error messages that indicate transient errors.
TRANSIENT_KEYWORDS: List[str] = [
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "temporary",
    "temporarily unavailable",
    "service unavailable",
    "rate limit",
    "too many requests",
    "retry",
    "circuit breaker",
    "overloaded",
    "capacity",
    "dns resolution",
]

#: Keywords in error messages that indicate permanent errors.
PERMANENT_KEYWORDS: List[str] = [
    "invalid",
    "validation failed",
    "not found",
    "does not exist",
    "unauthorized",
    "forbidden",
    "permission denied",
    "authentication failed",
    "bad request",
    "unprocessable",
    "schema mismatch",
    "data integrity",
    "constraint violation",
]

#: Keywords in error messages that indicate degraded service.
DEGRADED_KEYWORDS: List[str] = [
    "partial",
    "incomplete",
    "stale",
    "cached",
    "low confidence",
    "degraded",
    "approximate",
    "estimated",
    "fallback",
    "limited data",
]


# ---------------------------------------------------------------------------
# Per-agent fallback and retry configuration
# ---------------------------------------------------------------------------

#: Per-agent retry and fallback configuration overrides.
#: Agents not in this map use default settings from config.
AGENT_ERROR_CONFIG: Dict[str, Dict[str, Any]] = {
    "EUDR-001": {
        "max_retries": 5,
        "fallback": FallbackStrategy.FAIL,
        "critical": True,
        "description": "Supply Chain Mapping -- critical entry point, no fallback",
    },
    "EUDR-002": {
        "max_retries": 5,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "Geolocation -- can use cached coordinates",
    },
    "EUDR-003": {
        "max_retries": 3,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "Satellite Monitoring -- can use recent cached imagery",
    },
    "EUDR-004": {
        "max_retries": 3,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "Forest Cover -- can use cached analysis",
    },
    "EUDR-005": {
        "max_retries": 3,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "Land Use Change -- can use cached detection",
    },
    "EUDR-006": {
        "max_retries": 3,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "Plot Boundary -- can proceed without polygons",
    },
    "EUDR-007": {
        "max_retries": 5,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "GPS Validation -- can use cached validation",
    },
    "EUDR-008": {
        "max_retries": 5,
        "fallback": FallbackStrategy.FAIL,
        "critical": True,
        "description": "Multi-Tier Supplier -- critical, no fallback",
    },
    "EUDR-009": {
        "max_retries": 3,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "Chain of Custody -- degraded mode available",
    },
    "EUDR-010": {
        "max_retries": 3,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "Segregation Verifier -- degraded mode available",
    },
    "EUDR-011": {
        "max_retries": 3,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "Mass Balance -- degraded mode available",
    },
    "EUDR-012": {
        "max_retries": 3,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "Document Auth -- degraded mode with manual review",
    },
    "EUDR-013": {
        "max_retries": 2,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "Blockchain -- fully optional, degraded acceptable",
    },
    "EUDR-014": {
        "max_retries": 2,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "QR Code -- optional traceability enhancement",
    },
    "EUDR-015": {
        "max_retries": 2,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "Mobile Data -- optional field data collection",
    },
    "EUDR-016": {
        "max_retries": 5,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "Country Risk -- can use cached risk data",
    },
    "EUDR-017": {
        "max_retries": 3,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "Supplier Risk -- can use cached scores",
    },
    "EUDR-018": {
        "max_retries": 3,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "Commodity Risk -- can use cached analysis",
    },
    "EUDR-019": {
        "max_retries": 3,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": False,
        "description": "Corruption Index -- can use cached CPI data",
    },
    "EUDR-020": {
        "max_retries": 5,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": True,
        "description": "Deforestation Alert -- can use cached alerts",
    },
    "EUDR-021": {
        "max_retries": 3,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": False,
        "description": "Indigenous Rights -- can use cached assessment",
    },
    "EUDR-022": {
        "max_retries": 3,
        "fallback": FallbackStrategy.CACHED_RESULT,
        "critical": False,
        "description": "Protected Area -- can use cached boundaries",
    },
    "EUDR-023": {
        "max_retries": 5,
        "fallback": FallbackStrategy.FAIL,
        "critical": True,
        "description": "Legal Compliance -- critical, no fallback",
    },
    "EUDR-024": {
        "max_retries": 3,
        "fallback": FallbackStrategy.DEGRADED_MODE,
        "critical": False,
        "description": "Third-Party Audit -- degraded if no audit available",
    },
    "EUDR-025": {
        "max_retries": 5,
        "fallback": FallbackStrategy.MANUAL_OVERRIDE,
        "critical": True,
        "description": "Risk Mitigation -- manual override if agent fails",
    },
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def classify_http_status(status_code: int) -> ErrorClassification:
    """Classify an HTTP status code into an error classification.

    Args:
        status_code: HTTP response status code.

    Returns:
        ErrorClassification for the status code.
    """
    if status_code in TRANSIENT_HTTP_CODES:
        return ErrorClassification.TRANSIENT
    if status_code in PERMANENT_HTTP_CODES:
        return ErrorClassification.PERMANENT
    if status_code in DEGRADED_HTTP_CODES:
        return ErrorClassification.DEGRADED
    if 200 <= status_code < 300:
        return ErrorClassification.TRANSIENT  # Success, not an error
    return ErrorClassification.UNKNOWN


def classify_exception(exception_name: str) -> ErrorClassification:
    """Classify an exception type name into an error classification.

    Args:
        exception_name: Exception class name.

    Returns:
        ErrorClassification for the exception type.
    """
    if exception_name in TRANSIENT_EXCEPTIONS:
        return ErrorClassification.TRANSIENT
    if exception_name in PERMANENT_EXCEPTIONS:
        return ErrorClassification.PERMANENT
    return ErrorClassification.UNKNOWN


def get_agent_error_config(agent_id: str) -> Dict[str, Any]:
    """Get error handling configuration for a specific agent.

    Args:
        agent_id: EUDR agent identifier.

    Returns:
        Error configuration dictionary with max_retries, fallback, etc.
    """
    return dict(AGENT_ERROR_CONFIG.get(agent_id, {
        "max_retries": 3,
        "fallback": FallbackStrategy.FAIL,
        "critical": True,
        "description": f"Default configuration for {agent_id}",
    }))


def get_agent_fallback(agent_id: str) -> FallbackStrategy:
    """Get the fallback strategy for a specific agent.

    Args:
        agent_id: EUDR agent identifier.

    Returns:
        FallbackStrategy for the agent.
    """
    config = get_agent_error_config(agent_id)
    return config.get("fallback", FallbackStrategy.FAIL)


def get_agent_max_retries(agent_id: str) -> int:
    """Get the maximum retry count for a specific agent.

    Args:
        agent_id: EUDR agent identifier.

    Returns:
        Maximum number of retries.
    """
    config = get_agent_error_config(agent_id)
    return config.get("max_retries", 3)


def is_critical_agent(agent_id: str) -> bool:
    """Check if an agent is critical (workflow fails if it fails).

    Args:
        agent_id: EUDR agent identifier.

    Returns:
        True if the agent is critical.
    """
    config = get_agent_error_config(agent_id)
    return config.get("critical", True)


def get_all_critical_agents() -> List[str]:
    """Get all critical agent IDs.

    Returns:
        List of critical agent identifiers.
    """
    return [
        agent_id for agent_id, config in AGENT_ERROR_CONFIG.items()
        if config.get("critical", True)
    ]


def get_non_critical_agents() -> List[str]:
    """Get all non-critical agent IDs.

    Returns:
        List of non-critical agent identifiers.
    """
    return [
        agent_id for agent_id, config in AGENT_ERROR_CONFIG.items()
        if not config.get("critical", True)
    ]
