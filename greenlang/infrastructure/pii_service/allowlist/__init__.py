# -*- coding: utf-8 -*-
"""
PII Allowlist Package - SEC-011 PII Detection/Redaction Enhancements

Manages allowlists for PII detection to reduce false positives. Supports
pattern-based exclusion of known safe values such as test data, reserved
domains, and placeholder values.

Features:
    - Multiple pattern types (regex, exact, prefix, suffix, contains)
    - Per-tenant and global allowlists
    - Default allowlists for common test patterns
    - Compiled regex caching for performance
    - PostgreSQL persistence
    - Prometheus metrics

Public API:
    - AllowlistManager: Main class for managing allowlists
    - AllowlistEntry: Model for individual allowlist entries
    - PatternType: Enum for pattern matching types
    - AllowlistConfig: Configuration for the manager
    - DEFAULT_ALLOWLISTS: Pre-configured safe patterns

Example:
    >>> from greenlang.infrastructure.pii_service.allowlist import (
    ...     AllowlistManager,
    ...     AllowlistEntry,
    ...     AllowlistConfig,
    ...     PatternType,
    ...     DEFAULT_ALLOWLISTS,
    ... )
    >>> config = AllowlistConfig(enable_defaults=True)
    >>> manager = AllowlistManager(config)
    >>> await manager.initialize()
    >>> is_allowed, entry = await manager.is_allowed("test@example.com", PIIType.EMAIL)
    >>> print(f"Allowed: {is_allowed}, Reason: {entry.reason if entry else 'N/A'}")

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
Version: 1.0.0
"""

from __future__ import annotations

import logging

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Patterns Module
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.allowlist.patterns import (
    PatternType,
    AllowlistEntry,
    DEFAULT_ALLOWLISTS,
    get_default_allowlist_count,
    get_allowlist_for_type,
)

# ---------------------------------------------------------------------------
# Manager Module
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.allowlist.manager import (
    AllowlistConfig,
    AllowlistManager,
    AllowlistError,
    InvalidPatternError,
    EntryNotFoundError,
    EntryLimitExceededError,
    get_allowlist_manager,
    reset_allowlist_manager,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Patterns
    "PatternType",
    "AllowlistEntry",
    "DEFAULT_ALLOWLISTS",
    "get_default_allowlist_count",
    "get_allowlist_for_type",
    # Manager
    "AllowlistConfig",
    "AllowlistManager",
    "AllowlistError",
    "InvalidPatternError",
    "EntryNotFoundError",
    "EntryLimitExceededError",
    "get_allowlist_manager",
    "reset_allowlist_manager",
]

logger.debug("PII allowlist package loaded (version %s)", __version__)
