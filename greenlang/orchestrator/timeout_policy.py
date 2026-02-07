# -*- coding: utf-8 -*-
"""
Timeout Policy - AGENT-FOUND-001: GreenLang DAG Orchestrator

Per-node timeout policy helpers. Wraps asyncio.wait_for for async
execution and provides timeout handling actions.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, Optional, TypeVar

from greenlang.orchestrator.models import OnTimeout, TimeoutPolicy

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Default policy constant
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_POLICY = TimeoutPolicy(
    timeout_seconds=60.0,
    on_timeout=OnTimeout.FAIL,
)


# ---------------------------------------------------------------------------
# Apply timeout to coroutine
# ---------------------------------------------------------------------------


async def apply_timeout(
    coro: Coroutine[Any, Any, T],
    policy: TimeoutPolicy,
) -> T:
    """Wrap a coroutine with asyncio.wait_for using the given policy.

    Args:
        coro: Awaitable coroutine to execute.
        policy: TimeoutPolicy specifying the timeout duration.

    Returns:
        Result of the coroutine.

    Raises:
        asyncio.TimeoutError: If the coroutine exceeds the timeout.
    """
    return await asyncio.wait_for(coro, timeout=policy.timeout_seconds)


# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------


def handle_timeout(node_id: str, policy: TimeoutPolicy) -> str:
    """Determine the action to take when a node times out.

    Args:
        node_id: ID of the node that timed out.
        policy: TimeoutPolicy with on_timeout action.

    Returns:
        Action string: "fail", "skip", or "compensate".
    """
    action = policy.on_timeout.value
    logger.warning(
        "Node '%s' timed out after %.1fs; action=%s",
        node_id, policy.timeout_seconds, action,
    )
    return action


# ---------------------------------------------------------------------------
# Policy merging
# ---------------------------------------------------------------------------


def merge_with_default(
    node_policy: Optional[TimeoutPolicy],
    default_policy: TimeoutPolicy,
) -> TimeoutPolicy:
    """Merge a node-level timeout policy with DAG-level defaults.

    If node_policy is None, the default_policy is returned.
    If node_policy is provided, it takes full precedence.

    Args:
        node_policy: Optional node-level override.
        default_policy: DAG-level default policy.

    Returns:
        Effective TimeoutPolicy for the node.
    """
    if node_policy is None:
        return default_policy
    return node_policy


__all__ = [
    "DEFAULT_TIMEOUT_POLICY",
    "apply_timeout",
    "handle_timeout",
    "merge_with_default",
]
