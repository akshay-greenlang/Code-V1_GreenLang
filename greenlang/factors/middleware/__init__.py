# -*- coding: utf-8 -*-
"""
GreenLang Factors Middleware Package
====================================

Provides tier-based rate limiting middleware for the Factors API.

Modules:
    rate_limiter: Sliding-window rate limiter with per-tier limits,
                  standard rate-limit response headers, and optional
                  Redis backend for multi-instance deployments.
"""

from greenlang.factors.middleware.rate_limiter import (
    RateLimitConfig,
    TierRateLimiter,
    apply_export_rate_limit,
    apply_rate_limit,
    configure_limiter,
    get_rate_limiter,
)

__all__ = [
    "RateLimitConfig",
    "TierRateLimiter",
    "apply_export_rate_limit",
    "apply_rate_limit",
    "configure_limiter",
    "get_rate_limiter",
]
