# -*- coding: utf-8 -*-
"""
Retry Policy - AGENT-FOUND-001: GreenLang DAG Orchestrator

Per-node retry policy helpers with integration to the existing
``execution/resilience/retry.py`` RetryConfig infrastructure.

Provides:
- Delay calculation for all four backoff strategies
- Retry eligibility checking
- Merging node-level policy with DAG-level defaults
- Conversion to RetryConfig for decorator integration

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from greenlang.orchestrator.models import RetryPolicy, RetryStrategyType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default policy constant
# ---------------------------------------------------------------------------

DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    strategy=RetryStrategyType.EXPONENTIAL,
    base_delay=1.0,
    max_delay=60.0,
    jitter=True,
    jitter_range=0.1,
    retryable_exceptions=[],
)


# ---------------------------------------------------------------------------
# Delay calculation
# ---------------------------------------------------------------------------


def calculate_delay(policy: RetryPolicy, attempt: int) -> float:
    """Calculate the delay in seconds before the next retry attempt.

    Args:
        policy: RetryPolicy configuration.
        attempt: Zero-indexed attempt number (0 = first retry).

    Returns:
        Delay in seconds (non-negative).
    """
    if policy.strategy == RetryStrategyType.EXPONENTIAL:
        delay = policy.base_delay * (2 ** attempt)
    elif policy.strategy == RetryStrategyType.LINEAR:
        delay = policy.base_delay * (attempt + 1)
    elif policy.strategy == RetryStrategyType.CONSTANT:
        delay = policy.base_delay
    elif policy.strategy == RetryStrategyType.FIBONACCI:
        fib = [1, 1]
        for _ in range(2, attempt + 2):
            fib.append(fib[-1] + fib[-2])
        delay = policy.base_delay * fib[min(attempt, len(fib) - 1)]
    else:
        delay = policy.base_delay

    # Cap at max_delay
    delay = min(delay, policy.max_delay)

    # Apply jitter
    if policy.jitter and delay > 0:
        jitter_amount = delay * policy.jitter_range
        delay += random.uniform(-jitter_amount, jitter_amount)

    return max(0.0, delay)


# ---------------------------------------------------------------------------
# Should retry
# ---------------------------------------------------------------------------


def should_retry(
    policy: RetryPolicy,
    exception: Exception,
    attempt: int,
) -> bool:
    """Determine whether a failed attempt should be retried.

    Args:
        policy: RetryPolicy configuration.
        exception: The exception that occurred.
        attempt: Current attempt number (1-indexed: 1 = first attempt).

    Returns:
        True if the attempt should be retried.
    """
    # Check max retries
    if attempt >= policy.max_retries:
        return False

    # Never retry KeyboardInterrupt or SystemExit
    if isinstance(exception, (KeyboardInterrupt, SystemExit)):
        return False

    # Check retryable exceptions filter
    if policy.retryable_exceptions:
        exception_name = type(exception).__name__
        exception_full = f"{type(exception).__module__}.{exception_name}"
        for allowed in policy.retryable_exceptions:
            if allowed in (exception_name, exception_full):
                return True
        return False

    # Default: retry all exceptions
    return True


# ---------------------------------------------------------------------------
# Policy merging
# ---------------------------------------------------------------------------


def merge_with_default(
    node_policy: Optional[RetryPolicy],
    default_policy: RetryPolicy,
) -> RetryPolicy:
    """Merge a node-level retry policy with DAG-level defaults.

    If node_policy is None, the default_policy is returned.
    If node_policy is provided, it takes full precedence.

    Args:
        node_policy: Optional node-level override.
        default_policy: DAG-level default policy.

    Returns:
        Effective RetryPolicy for the node.
    """
    if node_policy is None:
        return default_policy
    return node_policy


# ---------------------------------------------------------------------------
# RetryConfig bridge
# ---------------------------------------------------------------------------


def to_retry_config(policy: RetryPolicy) -> object:
    """Convert a RetryPolicy to a RetryConfig for decorator integration.

    This bridges the orchestrator's RetryPolicy to the existing
    ``execution/resilience/retry.py`` RetryConfig.

    Args:
        policy: Orchestrator RetryPolicy.

    Returns:
        RetryConfig instance from the resilience module.
    """
    try:
        from greenlang.execution.resilience.retry import (
            RetryConfig,
            RetryStrategy,
        )
        strategy_map = {
            RetryStrategyType.EXPONENTIAL: RetryStrategy.EXPONENTIAL,
            RetryStrategyType.LINEAR: RetryStrategy.LINEAR,
            RetryStrategyType.CONSTANT: RetryStrategy.CONSTANT,
            RetryStrategyType.FIBONACCI: RetryStrategy.FIBONACCI,
        }
        return RetryConfig(
            max_retries=policy.max_retries,
            base_delay=policy.base_delay,
            max_delay=policy.max_delay,
            strategy=strategy_map.get(
                policy.strategy, RetryStrategy.EXPONENTIAL,
            ),
            jitter=policy.jitter,
            jitter_range=policy.jitter_range,
        )
    except ImportError:
        logger.debug(
            "execution.resilience.retry not available; "
            "returning raw RetryPolicy"
        )
        return policy


__all__ = [
    "DEFAULT_RETRY_POLICY",
    "calculate_delay",
    "should_retry",
    "merge_with_default",
    "to_retry_config",
]
