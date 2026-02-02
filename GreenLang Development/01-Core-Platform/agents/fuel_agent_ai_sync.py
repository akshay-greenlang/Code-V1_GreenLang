# -*- coding: utf-8 -*-
"""Backward-compatible sync wrapper for AsyncFuelAgentAI.

This module provides a drop-in replacement for the original FuelAgentAI that
internally uses the async version for better performance and resource management.

Key Benefits:
    1. Zero Breaking Changes: Same API as original FuelAgentAI
    2. Better Performance: Uses async infrastructure underneath
    3. Proper Resource Cleanup: Async context managers ensure cleanup
    4. Easy Migration: Change import path, code works unchanged

Migration Path:
    # Before (sync)
    from greenlang.agents.fuel_agent_ai import FuelAgentAI
    agent = FuelAgentAI()
    result = agent.run(payload)

    # After (async-powered sync wrapper)
    from greenlang.agents.fuel_agent_ai_sync import FuelAgentAISync as FuelAgentAI
    agent = FuelAgentAI()
    result = agent.run(payload)  # Same API!

    # Or use async directly for best performance
    from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI
    async with AsyncFuelAgentAI() as agent:
        result = await agent.run_async(payload)

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from typing import Optional, Dict, Any
import warnings

# DEPRECATION WARNING: This agent is deprecated for CRITICAL PATH emissions calculations
warnings.warn(
    "FuelAgentAISync has been deprecated. "
    "For CRITICAL PATH emissions calculations (Scope 1/2 fuel emissions), use the deterministic version instead: "
    "from greenlang.agents.fuel_agent import FuelAgent. "
    "This AI version should only be used for non-regulatory recommendations. "
    "See AGENT_CATEGORIZATION_AUDIT.md for details.",
    DeprecationWarning,
    stacklevel=2
)

from greenlang.agents.base import AgentResult
from greenlang.agents.sync_wrapper import SyncAgentWrapper
from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI
from .types import FuelInput, FuelOutput
from greenlang.config.schemas import GreenLangConfig


class FuelAgentAISync(SyncAgentWrapper[FuelInput, FuelOutput]):
    """Sync wrapper for AsyncFuelAgentAI with backward-compatible API.

    This class wraps AsyncFuelAgentAI to provide a synchronous interface
    that's 100% compatible with the original FuelAgentAI.

    Internally, it uses asyncio.run() to execute async operations, giving you
    the benefits of the async infrastructure (proper resource management,
    lifecycle hooks) with a sync API.

    For best performance in async applications, use AsyncFuelAgentAI directly.

    Features:
    - Same API as original FuelAgentAI
    - Async infrastructure benefits (resource cleanup, lifecycle)
    - Config injection support
    - Context manager support
    - Zero breaking changes

    Example:
        >>> from greenlang.config import get_config
        >>> agent = FuelAgentAISync(get_config())
        >>> result = agent.run({
        ...     "fuel_type": "natural_gas",
        ...     "amount": 1000,
        ...     "unit": "therms"
        ... })
        >>> print(result["data"]["co2e_emissions_kg"])
        5310.0
    """

    def __init__(
        self,
        config: Optional[GreenLangConfig] = None,
        *,
        budget_usd: Optional[float] = None,
        enable_explanations: Optional[bool] = None,
        enable_recommendations: Optional[bool] = None,
    ) -> None:
        """Initialize the sync wrapper for AsyncFuelAgentAI.

        Args:
            config: GreenLangConfig instance (uses default if None)
            budget_usd: Maximum USD to spend per calculation (from config if None)
            enable_explanations: Enable AI-generated explanations (default: True)
            enable_recommendations: Enable AI recommendations (default: True)
        """
        # Create async agent
        async_agent = AsyncFuelAgentAI(
            config=config,
            budget_usd=budget_usd,
            enable_explanations=enable_explanations,
            enable_recommendations=enable_recommendations,
        )

        # Initialize sync wrapper
        super().__init__(async_agent)

        # Store config for backward compatibility
        self.config = async_agent.config
        self.agent_id = async_agent.agent_id
        self.name = async_agent.name
        self.version = async_agent.version

    def validate(self, payload: FuelInput) -> bool:
        """Validate input payload (sync).

        Args:
            payload: Input data

        Returns:
            bool: True if valid
        """
        # Delegate to underlying fuel_agent
        return self._async_agent.fuel_agent.validate(payload)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dict with AI and tool metrics
        """
        return self._async_agent.get_performance_summary()


# Alias for backward compatibility
FuelAgentAI = FuelAgentAISync
