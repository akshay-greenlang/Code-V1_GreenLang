# -*- coding: utf-8 -*-
"""
Audit Exclusions - Centralized Audit Logging Service (SEC-005)

Defines path-based exclusion rules and sensitivity classification for the
audit middleware. Controls which HTTP endpoints are audited and at what
sensitivity level.

Features:
    - EXCLUDED_PATHS: Exact path matches that bypass auditing.
    - EXCLUDED_PREFIXES: Prefix patterns that bypass auditing.
    - SENSITIVITY_MAP: Path pattern to sensitivity level mapping.
    - Wildcard path matching with fnmatch-style patterns.
    - Sensitivity levels: public, internal, sensitive, critical.

Security Compliance:
    - SOC 2 CC6.1 (Logical Access)
    - ISO 27001 A.12.4 (Logging and Monitoring)
    - PCI DSS 10.2 (Audit Trail Requirements)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path Exclusions
# ---------------------------------------------------------------------------

EXCLUDED_PATHS: Set[str] = {
    # Health check endpoints (high frequency, no security value)
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/livez",
    # Metrics endpoints (Prometheus scrapes, no user data)
    "/metrics",
    "/metrics/",
    # Static assets
    "/favicon.ico",
    "/robots.txt",
    # OpenAPI documentation (read-only, public)
    "/docs",
    "/redoc",
    "/openapi.json",
}

EXCLUDED_PREFIXES: List[str] = [
    # Static file serving
    "/static/",
    "/_next/",
    "/assets/",
    "/public/",
    # Webpack dev server
    "/__webpack_hmr",
    # Socket.IO polling (handled separately if needed)
    "/socket.io/",
]


# ---------------------------------------------------------------------------
# Sensitivity Classification
# ---------------------------------------------------------------------------

# Sensitivity levels from least to most sensitive:
#   - public: No user-specific data, public endpoints
#   - internal: Internal operations, may contain user IDs
#   - sensitive: Contains PII, financial data, or security-related info
#   - critical: Admin operations, compliance-critical, audit trail itself

SENSITIVITY_LEVELS = ("public", "internal", "sensitive", "critical")

# Default sensitivity for unmatched paths
DEFAULT_SENSITIVITY = "internal"

# Pattern to sensitivity mapping
# Patterns are evaluated in order; first match wins
# Supports fnmatch-style wildcards: *, ?, [seq], [!seq]
SENSITIVITY_MAP: Dict[str, str] = {
    # =========================================================================
    # Critical: Admin and audit operations
    # =========================================================================
    "/api/v1/admin/*": "critical",
    "/api/v1/audit/*": "critical",
    "/api/v1/rbac/*": "critical",
    "/api/v1/encryption/*": "critical",
    "/auth/admin/*": "critical",
    "/api/v1/compliance/*": "critical",
    "/api/v1/flags/*/kill": "critical",
    "/api/v1/flags/*/restore": "critical",

    # =========================================================================
    # Sensitive: Authentication, user data, financial data
    # =========================================================================
    "/auth/*": "sensitive",
    "/api/v1/users/*": "sensitive",
    "/api/v1/accounts/*": "sensitive",
    "/api/v1/tenants/*": "sensitive",
    "/api/v1/billing/*": "sensitive",
    "/api/v1/emissions/*/financial": "sensitive",
    "/api/v1/reports/*": "sensitive",
    "/api/v1/exports/*": "sensitive",

    # =========================================================================
    # Internal: Standard API operations
    # =========================================================================
    "/api/v1/agents/*": "internal",
    "/api/v1/emissions/*": "internal",
    "/api/v1/jobs/*": "internal",
    "/api/v1/factory/*": "internal",
    "/api/v1/flags/*": "internal",
    "/api/v1/vectors/*": "internal",
    "/api/v1/data/*": "internal",
    "/api/*": "internal",

    # =========================================================================
    # Public: Documentation, health checks, etc.
    # =========================================================================
    "/docs*": "public",
    "/redoc*": "public",
    "/openapi*": "public",
}


# ---------------------------------------------------------------------------
# AuditExclusionRules Class
# ---------------------------------------------------------------------------


class AuditExclusionRules:
    """Manages audit exclusion rules and sensitivity classification.

    Provides efficient path matching for determining whether a request
    should be audited and at what sensitivity level.

    Thread-safe: All methods are stateless lookups against class-level
    constants.

    Example:
        >>> rules = AuditExclusionRules()
        >>> rules.should_exclude("/health")
        True
        >>> rules.should_exclude("/api/v1/agents/123")
        False
        >>> rules.get_sensitivity("/api/v1/admin/users")
        'critical'
        >>> rules.get_sensitivity("/api/v1/agents/execute")
        'internal'
    """

    def __init__(
        self,
        excluded_paths: Optional[Set[str]] = None,
        excluded_prefixes: Optional[List[str]] = None,
        sensitivity_map: Optional[Dict[str, str]] = None,
        default_sensitivity: str = DEFAULT_SENSITIVITY,
    ) -> None:
        """Initialize exclusion rules.

        Args:
            excluded_paths: Set of exact paths to exclude.
                Defaults to module-level EXCLUDED_PATHS.
            excluded_prefixes: List of path prefixes to exclude.
                Defaults to module-level EXCLUDED_PREFIXES.
            sensitivity_map: Pattern to sensitivity mapping.
                Defaults to module-level SENSITIVITY_MAP.
            default_sensitivity: Sensitivity for unmatched paths.
                Defaults to "internal".
        """
        self._excluded_paths = excluded_paths or EXCLUDED_PATHS
        self._excluded_prefixes = excluded_prefixes or EXCLUDED_PREFIXES
        self._sensitivity_map = sensitivity_map or SENSITIVITY_MAP
        self._default_sensitivity = default_sensitivity

        # Pre-compile regex patterns for performance
        self._sensitivity_patterns: List[tuple] = []
        for pattern, sensitivity in self._sensitivity_map.items():
            # Convert fnmatch pattern to regex
            regex = self._fnmatch_to_regex(pattern)
            self._sensitivity_patterns.append((regex, sensitivity))

        logger.debug(
            "AuditExclusionRules initialized: %d excluded paths, "
            "%d excluded prefixes, %d sensitivity patterns",
            len(self._excluded_paths),
            len(self._excluded_prefixes),
            len(self._sensitivity_patterns),
        )

    def should_exclude(self, path: str) -> bool:
        """Determine if a path should be excluded from auditing.

        Args:
            path: The request URL path (e.g., "/health", "/api/v1/agents").

        Returns:
            True if the path should NOT be audited, False otherwise.

        Example:
            >>> rules = AuditExclusionRules()
            >>> rules.should_exclude("/health")
            True
            >>> rules.should_exclude("/static/css/main.css")
            True
            >>> rules.should_exclude("/api/v1/agents/123")
            False
        """
        normalised = self._normalise_path(path)

        # Check exact match
        if normalised in self._excluded_paths:
            return True

        # Check prefix match
        for prefix in self._excluded_prefixes:
            if normalised.startswith(prefix):
                return True

        return False

    def get_sensitivity(self, path: str) -> str:
        """Determine the sensitivity level for a path.

        Sensitivity levels control how audit data is handled:
            - public: Minimal logging, no special protection
            - internal: Standard logging, no PII
            - sensitive: Full logging, PII redaction applied
            - critical: Full logging, PII redaction, long retention

        Args:
            path: The request URL path.

        Returns:
            Sensitivity level string: one of SENSITIVITY_LEVELS.

        Example:
            >>> rules = AuditExclusionRules()
            >>> rules.get_sensitivity("/api/v1/admin/users")
            'critical'
            >>> rules.get_sensitivity("/api/v1/agents/123/execute")
            'internal'
            >>> rules.get_sensitivity("/auth/login")
            'sensitive'
        """
        normalised = self._normalise_path(path)

        # Check patterns in order (first match wins)
        for regex, sensitivity in self._sensitivity_patterns:
            if regex.match(normalised):
                return sensitivity

        return self._default_sensitivity

    def is_critical(self, path: str) -> bool:
        """Check if a path is classified as critical.

        Critical paths require immediate audit log persistence and
        extended retention periods.

        Args:
            path: The request URL path.

        Returns:
            True if the path is critical sensitivity.
        """
        return self.get_sensitivity(path) == "critical"

    def is_sensitive(self, path: str) -> bool:
        """Check if a path is classified as sensitive or critical.

        Sensitive paths require PII redaction and careful handling.

        Args:
            path: The request URL path.

        Returns:
            True if the path is sensitive or critical.
        """
        sensitivity = self.get_sensitivity(path)
        return sensitivity in ("sensitive", "critical")

    def get_all_critical_patterns(self) -> List[str]:
        """Get all patterns marked as critical.

        Useful for documentation and configuration validation.

        Returns:
            List of fnmatch patterns with critical sensitivity.
        """
        return [
            pattern
            for pattern, sensitivity in self._sensitivity_map.items()
            if sensitivity == "critical"
        ]

    def get_all_excluded_paths(self) -> Set[str]:
        """Get all excluded paths.

        Returns:
            Set of excluded path strings.
        """
        return self._excluded_paths.copy()

    def get_all_excluded_prefixes(self) -> List[str]:
        """Get all excluded path prefixes.

        Returns:
            List of excluded prefix strings.
        """
        return list(self._excluded_prefixes)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Normalise a URL path for matching.

        Removes trailing slashes and ensures leading slash.

        Args:
            path: Raw path string.

        Returns:
            Normalised path string.
        """
        normalised = path.rstrip("/") or "/"
        if not normalised.startswith("/"):
            normalised = "/" + normalised
        return normalised

    @staticmethod
    def _fnmatch_to_regex(pattern: str) -> re.Pattern:
        """Convert an fnmatch pattern to a compiled regex.

        Supports:
            - * matches any characters except /
            - ** matches any characters including /
            - ? matches a single character
            - [seq] matches any character in seq
            - [!seq] matches any character not in seq

        Args:
            pattern: fnmatch-style pattern string.

        Returns:
            Compiled regex pattern.
        """
        # Escape special regex characters except our wildcards
        regex = re.escape(pattern)

        # Convert escaped wildcards back to regex equivalents
        # Order matters: ** before *
        regex = regex.replace(r"\*\*", ".*")
        regex = regex.replace(r"\*", "[^/]*")
        regex = regex.replace(r"\?", ".")

        # Handle character classes (already escaped, need to unescape)
        regex = regex.replace(r"\[", "[").replace(r"\]", "]")

        # Anchor the pattern
        regex = "^" + regex + "$"

        return re.compile(regex)


# ---------------------------------------------------------------------------
# Singleton instance for convenience
# ---------------------------------------------------------------------------

_default_rules: Optional[AuditExclusionRules] = None


def get_exclusion_rules() -> AuditExclusionRules:
    """Get the default singleton AuditExclusionRules instance.

    Returns:
        Shared AuditExclusionRules instance.
    """
    global _default_rules
    if _default_rules is None:
        _default_rules = AuditExclusionRules()
    return _default_rules


def should_exclude(path: str) -> bool:
    """Convenience function: check if path should be excluded.

    Args:
        path: Request URL path.

    Returns:
        True if the path should not be audited.
    """
    return get_exclusion_rules().should_exclude(path)


def get_sensitivity(path: str) -> str:
    """Convenience function: get sensitivity level for path.

    Args:
        path: Request URL path.

    Returns:
        Sensitivity level string.
    """
    return get_exclusion_rules().get_sensitivity(path)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "EXCLUDED_PATHS",
    "EXCLUDED_PREFIXES",
    "SENSITIVITY_LEVELS",
    "SENSITIVITY_MAP",
    "DEFAULT_SENSITIVITY",
    # Classes
    "AuditExclusionRules",
    # Convenience functions
    "get_exclusion_rules",
    "should_exclude",
    "get_sensitivity",
]
