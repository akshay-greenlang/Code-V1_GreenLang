"""
GreenLang Feature Flags - INFRA-008

Production-grade feature flag system for the GreenLang Climate OS platform.
Provides gradual rollout, kill switches, A/B testing, tenant isolation,
and full audit trails for regulatory compliance.

Public API:
    - FeatureFlagService: Async service class for flag evaluation and management.
    - FeatureFlag: Core flag definition model.
    - FlagType: Supported flag evaluation strategies.
    - FlagStatus: Lifecycle status of a flag.
    - EvaluationContext: Runtime context for flag evaluation.
    - FlagEvaluationResult: Evaluation outcome.
    - get_feature_flag_service: Factory function for singleton access.

Example:
    >>> from greenlang.infrastructure.feature_flags import (
    ...     FeatureFlag, FlagType, FlagStatus, EvaluationContext,
    ...     FlagEvaluationResult, get_feature_flag_service,
    ... )
    >>> service = get_feature_flag_service()
    >>> await service.initialize()
    >>> ctx = EvaluationContext(user_id="u-1", tenant_id="t-acme", environment="prod")
    >>> enabled = await service.is_enabled("enable-scope3-calc", ctx)
"""

from __future__ import annotations

import logging

from greenlang.infrastructure.feature_flags.models import (
    AuditLogEntry,
    EvaluationContext,
    FeatureFlag,
    FlagEvaluationResult,
    FlagOverride,
    FlagRule,
    FlagStatus,
    FlagType,
    FlagVariant,
)
from greenlang.infrastructure.feature_flags.service import (
    FeatureFlagService,
    get_feature_flag_service,
    reset_service as reset_feature_flag_service,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Service
    "FeatureFlagService",
    "get_feature_flag_service",
    "reset_feature_flag_service",
    # Models
    "AuditLogEntry",
    "EvaluationContext",
    "FeatureFlag",
    "FlagEvaluationResult",
    "FlagOverride",
    "FlagRule",
    "FlagStatus",
    "FlagType",
    "FlagVariant",
]
