"""
Agent Registry - Single source of truth for all GreenLang agents.

This module provides a centralized registry for managing agent implementations,
versions, and deprecation status. It handles backward compatibility and
provides migration utilities.

Example:
    >>> from greenlang.agents.registry import AgentRegistry
    >>> registry = AgentRegistry()
    >>> agent = registry.create_agent('FuelAgent', config=config)
"""

import importlib
import warnings
from typing import Dict, Any, Optional, Type, List
from dataclasses import dataclass
from enum import Enum
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent availability status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"


class ExecutionMode(str, Enum):
    """Agent execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    name: str
    canonical_class: str
    module_path: str
    version: str
    status: AgentStatus
    deprecated_in: Optional[str] = None
    removed_in: Optional[str] = None
    migration_guide: Optional[str] = None
    supported_modes: List[ExecutionMode] = None
    ai_enabled: bool = False
    description: str = ""

    @property
    def canonical_import(self) -> str:
        """Get the canonical import statement."""
        return f"from {self.module_path} import {self.canonical_class}"

    @property
    def is_deprecated(self) -> bool:
        """Check if agent is deprecated."""
        return self.status == AgentStatus.DEPRECATED


class AgentRegistryConfig(BaseModel):
    """Configuration for agent registry."""
    suppress_deprecation_warnings: bool = Field(
        default=False,
        description="Suppress deprecation warnings"
    )
    allow_experimental: bool = Field(
        default=False,
        description="Allow loading experimental agents"
    )
    compatibility_mode: bool = Field(
        default=True,
        description="Enable backward compatibility mode"
    )


class AgentRegistry:
    """
    Centralized registry for all GreenLang agents.

    This registry maintains information about all available agents,
    their versions, deprecation status, and provides utilities for
    agent creation and migration.
    """

    # Master registry of all agents
    _AGENT_REGISTRY: Dict[str, AgentInfo] = {
        # Fuel Agent Family
        "FuelAgent": AgentInfo(
            name="FuelAgent",
            canonical_class="FuelAgent",
            module_path="greenlang.agents.fuel_agent_ai_v2",
            version="2.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC, ExecutionMode.ASYNC],
            ai_enabled=True,
            description="Fuel consumption calculation and optimization agent"
        ),
        "FuelAgent_legacy": AgentInfo(
            name="FuelAgent_legacy",
            canonical_class="FuelAgent",
            module_path="greenlang.agents.deprecated.fuel_agent",
            version="1.0.0",
            status=AgentStatus.DEPRECATED,
            deprecated_in="1.8.0",
            removed_in="2.0.0",
            migration_guide="Use FuelAgent with ai_enabled=False for deterministic calculations"
        ),

        # Boiler Replacement Agent Family
        "BoilerReplacementAgent": AgentInfo(
            name="BoilerReplacementAgent",
            canonical_class="BoilerReplacementAgent",
            module_path="greenlang.agents.boiler_replacement_agent_ai_v4",
            version="4.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Boiler replacement analysis and recommendations"
        ),

        # Carbon Agent Family
        "CarbonAgent": AgentInfo(
            name="CarbonAgent",
            canonical_class="CarbonAgent",
            module_path="greenlang.agents.carbon_agent_ai",
            version="2.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Carbon emissions calculation and tracking"
        ),

        # Grid Factor Agent Family
        "GridFactorAgent": AgentInfo(
            name="GridFactorAgent",
            canonical_class="GridFactorAgent",
            module_path="greenlang.agents.grid_factor_agent_ai",
            version="2.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Grid emission factors and renewable mix analysis"
        ),

        # Recommendation Agent Family
        "RecommendationAgent": AgentInfo(
            name="RecommendationAgent",
            canonical_class="RecommendationAgent",
            module_path="greenlang.agents.recommendation_agent_ai_v2",
            version="2.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Energy efficiency and decarbonization recommendations"
        ),

        # Report Agent Family
        "ReportAgent": AgentInfo(
            name="ReportAgent",
            canonical_class="ReportAgent",
            module_path="greenlang.agents.report_agent_ai",
            version="2.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Report generation with narrative capabilities"
        ),

        # Industrial Heat Pump Agent Family
        "IndustrialHeatPumpAgent": AgentInfo(
            name="IndustrialHeatPumpAgent",
            canonical_class="IndustrialHeatPumpAgent",
            module_path="greenlang.agents.industrial_heat_pump_agent_ai_v4",
            version="4.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Industrial heat pump sizing and optimization"
        ),

        # Waste Heat Recovery Agent Family
        "WasteHeatRecoveryAgent": AgentInfo(
            name="WasteHeatRecoveryAgent",
            canonical_class="WasteHeatRecoveryAgent",
            module_path="greenlang.agents.waste_heat_recovery_agent_ai_v3",
            version="3.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Waste heat recovery system analysis"
        ),

        # Decarbonization Roadmap Agent Family
        "DecarbonizationRoadmapAgent": AgentInfo(
            name="DecarbonizationRoadmapAgent",
            canonical_class="DecarbonizationRoadmapAgent",
            module_path="greenlang.agents.decarbonization_roadmap_agent_ai_v3",
            version="3.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Comprehensive decarbonization roadmap generation"
        ),

        # Other Core Agents
        "BenchmarkAgent": AgentInfo(
            name="BenchmarkAgent",
            canonical_class="BenchmarkAgent",
            module_path="greenlang.agents.benchmark_agent_ai",
            version="2.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Performance benchmarking and comparison"
        ),

        "CogenerationCHPAgent": AgentInfo(
            name="CogenerationCHPAgent",
            canonical_class="CogenerationCHPAgent",
            module_path="greenlang.agents.cogeneration_chp_agent_ai",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Combined heat and power system analysis"
        ),

        "ThermalStorageAgent": AgentInfo(
            name="ThermalStorageAgent",
            canonical_class="ThermalStorageAgent",
            module_path="greenlang.agents.thermal_storage_agent_ai",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Thermal energy storage optimization"
        ),

        "IndustrialProcessHeatAgent": AgentInfo(
            name="IndustrialProcessHeatAgent",
            canonical_class="IndustrialProcessHeatAgent",
            module_path="greenlang.agents.industrial_process_heat_agent_ai",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC, ExecutionMode.ASYNC],
            ai_enabled=True,
            description="Industrial process heat optimization"
        ),

        "AnomalyAgent": AgentInfo(
            name="AnomalyAgent",
            canonical_class="AnomalyAgent",
            module_path="greenlang.agents.anomaly_agent_iforest",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=False,
            description="Anomaly detection using Isolation Forest"
        ),

        "AnomalyInvestigationAgent": AgentInfo(
            name="AnomalyInvestigationAgent",
            canonical_class="AnomalyInvestigationAgent",
            module_path="greenlang.agents.anomaly_investigation_agent",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Deep investigation of detected anomalies"
        ),

        "ForecastAgent": AgentInfo(
            name="ForecastAgent",
            canonical_class="ForecastAgent",
            module_path="greenlang.agents.forecast_agent_sarima",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=False,
            description="Time series forecasting using SARIMA"
        ),

        "ForecastExplanationAgent": AgentInfo(
            name="ForecastExplanationAgent",
            canonical_class="ForecastExplanationAgent",
            module_path="greenlang.agents.forecast_explanation_agent",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=True,
            description="Explain and interpret forecast results"
        ),

        "IntensityAgent": AgentInfo(
            name="IntensityAgent",
            canonical_class="IntensityAgent",
            module_path="greenlang.agents.intensity_agent",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            supported_modes=[ExecutionMode.SYNC],
            ai_enabled=False,
            description="Energy and carbon intensity calculations"
        ),
    }

    # Alias mappings for backward compatibility
    _ALIAS_MAPPING = {
        "FuelAgentAI": "FuelAgent",
        "FuelAgentAsync": "FuelAgent",
        "FuelAgentV2": "FuelAgent",
        "BoilerReplacementAgentV4": "BoilerReplacementAgent",
        "BoilerReplacementAgentV3": "BoilerReplacementAgent",
        "CarbonAgentAI": "CarbonAgent",
        "GridFactorAgentAI": "GridFactorAgent",
        "RecommendationAgentV2": "RecommendationAgent",
        "ReportAgentAI": "ReportAgent",
        "IndustrialHeatPumpAgentV4": "IndustrialHeatPumpAgent",
        "WasteHeatRecoveryAgentV3": "WasteHeatRecoveryAgent",
        "DecarbonizationRoadmapAgentV3": "DecarbonizationRoadmapAgent",
    }

    def __init__(self, config: Optional[AgentRegistryConfig] = None):
        """Initialize agent registry."""
        self.config = config or AgentRegistryConfig()
        self._loaded_agents: Dict[str, Type] = {}

    @classmethod
    def list_agents(cls, include_deprecated: bool = False) -> List[str]:
        """
        List all available agents.

        Args:
            include_deprecated: Whether to include deprecated agents

        Returns:
            List of agent names
        """
        agents = []
        for name, info in cls._AGENT_REGISTRY.items():
            if include_deprecated or info.status != AgentStatus.DEPRECATED:
                if not name.endswith("_legacy"):
                    agents.append(name)
        return sorted(agents)

    @classmethod
    def get_agent_info(cls, agent_name: str) -> Optional[AgentInfo]:
        """
        Get information about an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentInfo object or None if not found
        """
        # Check aliases first
        if agent_name in cls._ALIAS_MAPPING:
            agent_name = cls._ALIAS_MAPPING[agent_name]

        return cls._AGENT_REGISTRY.get(agent_name)

    def create_agent(self,
                    agent_name: str,
                    config: Any = None,
                    mode: Optional[ExecutionMode] = None,
                    **kwargs) -> Any:
        """
        Create an agent instance from the registry.

        Args:
            agent_name: Name of the agent to create
            config: Configuration for the agent
            mode: Execution mode (sync/async)
            **kwargs: Additional arguments for agent constructor

        Returns:
            Instantiated agent

        Raises:
            ValueError: If agent not found or configuration invalid
        """
        # Resolve aliases
        if agent_name in self._ALIAS_MAPPING:
            original_name = agent_name
            agent_name = self._ALIAS_MAPPING[agent_name]
            logger.info(f"Resolved alias {original_name} to {agent_name}")

        # Get agent info
        agent_info = self.get_agent_info(agent_name)
        if not agent_info:
            raise ValueError(f"Agent '{agent_name}' not found in registry")

        # Check deprecation
        if agent_info.is_deprecated and not self.config.suppress_deprecation_warnings:
            warnings.warn(
                f"Agent '{agent_name}' is deprecated since version {agent_info.deprecated_in}. "
                f"It will be removed in version {agent_info.removed_in}. "
                f"Migration guide: {agent_info.migration_guide}",
                DeprecationWarning,
                stacklevel=2
            )

        # Check experimental
        if agent_info.status == AgentStatus.EXPERIMENTAL and not self.config.allow_experimental:
            raise ValueError(
                f"Agent '{agent_name}' is experimental. "
                "Set allow_experimental=True to use experimental agents."
            )

        # Check mode support
        if mode and agent_info.supported_modes:
            if mode not in agent_info.supported_modes:
                raise ValueError(
                    f"Agent '{agent_name}' does not support mode '{mode}'. "
                    f"Supported modes: {agent_info.supported_modes}"
                )

        # Load agent class if not already loaded
        if agent_name not in self._loaded_agents:
            try:
                module = importlib.import_module(agent_info.module_path)
                agent_class = getattr(module, agent_info.canonical_class)
                self._loaded_agents[agent_name] = agent_class
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load agent '{agent_name}' from {agent_info.module_path}: {e}"
                )

        # Create agent instance
        agent_class = self._loaded_agents[agent_name]

        # Add mode to kwargs if specified
        if mode:
            kwargs['mode'] = mode.value

        # Add AI enabled flag if applicable
        if agent_info.ai_enabled and 'ai_enabled' not in kwargs:
            kwargs['ai_enabled'] = True

        try:
            if config:
                agent = agent_class(config, **kwargs)
            else:
                agent = agent_class(**kwargs)

            logger.info(f"Created agent '{agent_name}' v{agent_info.version}")
            return agent

        except Exception as e:
            raise ValueError(f"Failed to create agent '{agent_name}': {e}")

    @classmethod
    def check_deprecation(cls, agent_name: str) -> Optional[str]:
        """
        Check if an agent is deprecated and return migration message.

        Args:
            agent_name: Name of the agent

        Returns:
            Migration message or None if not deprecated
        """
        agent_info = cls.get_agent_info(agent_name)
        if agent_info and agent_info.is_deprecated:
            return (
                f"Agent '{agent_name}' is deprecated. {agent_info.migration_guide}"
            )
        return None

    @classmethod
    def get_migration_path(cls, old_agent: str) -> Optional[str]:
        """
        Get migration path for deprecated agent.

        Args:
            old_agent: Name of deprecated agent

        Returns:
            Canonical agent name or None
        """
        # Check if it's an alias
        if old_agent in cls._ALIAS_MAPPING:
            return cls._ALIAS_MAPPING[old_agent]

        # Check if it's a legacy agent
        if old_agent + "_legacy" in cls._AGENT_REGISTRY:
            return old_agent

        return None

    @classmethod
    def export_registry(cls) -> Dict[str, Dict[str, Any]]:
        """
        Export registry as dictionary for documentation.

        Returns:
            Dictionary representation of registry
        """
        registry_export = {}
        for name, info in cls._AGENT_REGISTRY.items():
            if not name.endswith("_legacy"):
                registry_export[name] = {
                    "version": info.version,
                    "status": info.status.value,
                    "ai_enabled": info.ai_enabled,
                    "modes": [m.value for m in info.supported_modes] if info.supported_modes else [],
                    "description": info.description,
                    "import": info.canonical_import,
                    "deprecated": info.is_deprecated,
                }
        return registry_export


# Singleton instance for convenience
_default_registry = AgentRegistry()

# Export convenience functions
list_agents = _default_registry.list_agents
get_agent_info = _default_registry.get_agent_info
create_agent = _default_registry.create_agent
check_deprecation = _default_registry.check_deprecation
get_migration_path = _default_registry.get_migration_path
export_registry = _default_registry.export_registry

__all__ = [
    'AgentRegistry',
    'AgentInfo',
    'AgentStatus',
    'ExecutionMode',
    'AgentRegistryConfig',
    'list_agents',
    'get_agent_info',
    'create_agent',
    'check_deprecation',
    'get_migration_path',
    'export_registry',
]