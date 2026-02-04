# -*- coding: utf-8 -*-
"""
Feature Flags Lifecycle Management - INFRA-008

Flag lifecycle state machine and stale flag detection for the GreenLang
feature flag system.

Provides:
    - FlagLifecycleManager: State machine for flag lifecycle transitions
    - StaleFlagDetector: Detects flags with no recent evaluations

Example:
    >>> from greenlang.infrastructure.feature_flags.lifecycle import (
    ...     FlagLifecycleManager,
    ...     StaleFlagDetector,
    ... )
    >>> manager = FlagLifecycleManager(service)
    >>> await manager.transition_state("my-flag", FlagStatus.ACTIVE, "engineer")
    >>> detector = StaleFlagDetector(service)
    >>> stale = await detector.detect_stale_flags(days_threshold=30)
"""

from greenlang.infrastructure.feature_flags.lifecycle.manager import (
    FlagLifecycleManager,
)
from greenlang.infrastructure.feature_flags.lifecycle.stale_detector import (
    StaleFlagDetector,
)

__all__ = [
    "FlagLifecycleManager",
    "StaleFlagDetector",
]
