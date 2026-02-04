"""
Feature Flag Targeting - INFRA-008

Targeting subsystem for the GreenLang feature flag engine. Provides
deterministic percentage rollout via consistent hashing, audience
segment matching with rich operator support, and priority-based
rule evaluation.

Exports:
    PercentageRollout: Consistent-hash-based percentage rollout and variant selection.
    SegmentMatcher: Evaluates user context against segment conditions.
    RuleEvaluator: Evaluates ordered targeting rules against an evaluation context.
"""

from greenlang.infrastructure.feature_flags.targeting.percentage import (
    PercentageRollout,
)
from greenlang.infrastructure.feature_flags.targeting.segments import (
    SegmentMatcher,
)
from greenlang.infrastructure.feature_flags.targeting.rules import (
    RuleEvaluator,
)

__all__ = [
    "PercentageRollout",
    "SegmentMatcher",
    "RuleEvaluator",
]
