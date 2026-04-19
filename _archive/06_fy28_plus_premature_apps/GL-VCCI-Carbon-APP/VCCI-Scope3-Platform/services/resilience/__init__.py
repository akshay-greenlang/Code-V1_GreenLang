# -*- coding: utf-8 -*-
"""GL-VCCI Resilience Patterns.

Application-specific resilience patterns for the GL-VCCI Scope 3 Platform.

Author: Team 2 - Resilience Patterns
Date: November 2025
"""

from .graceful_degradation import (
    DegradationTier,
    DegradationManager,
    ServiceHealth,
    get_degradation_manager,
    degradation_handler,
)

__all__ = [
    "DegradationTier",
    "DegradationManager",
    "ServiceHealth",
    "get_degradation_manager",
    "degradation_handler",
]
