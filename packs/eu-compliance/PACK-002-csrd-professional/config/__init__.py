"""
PACK-002 CSRD Professional Pack - Configuration Module

This module provides configuration management for the CSRD Professional Pack,
including pack manifest loading, preset resolution, sector-specific
configuration, professional-tier features (consolidation, cross-framework,
scenarios), and environment-based overrides.

Usage:
    >>> from packs.eu_compliance.pack_002_csrd_professional.config import PackConfig
    >>> config = PackConfig.load()
    >>> config = PackConfig.load(
    ...     size_preset="enterprise_group",
    ...     sector_preset="manufacturing_pro",
    ... )
"""

from config.pack_config import (
    AgentComponentConfig,
    ApprovalConfig,
    AssuranceConfig,
    BenchmarkingConfig,
    ConsolidationConfig,
    CrossFrameworkConfig,
    CSRDProfessionalPackConfig,
    DataGovernanceConfig,
    EntityDefinition,
    IntensityMetricsConfig,
    PackConfig,
    PerformanceTargets,
    PresetConfig,
    ProfessionalPackConfig,
    QualityGate,
    QualityGateConfig,
    RegulatoryConfig,
    ScenarioConfig,
    ScenarioDefinition,
    StakeholderConfig,
    WebhookConfig,
    WorkflowConfig,
    WorkflowPhaseConfig,
)

__all__ = [
    "PackConfig",
    "CSRDProfessionalPackConfig",
    "ProfessionalPackConfig",
    "WorkflowConfig",
    "WorkflowPhaseConfig",
    "AgentComponentConfig",
    "PresetConfig",
    "PerformanceTargets",
    "ConsolidationConfig",
    "EntityDefinition",
    "ApprovalConfig",
    "QualityGateConfig",
    "QualityGate",
    "CrossFrameworkConfig",
    "ScenarioConfig",
    "ScenarioDefinition",
    "BenchmarkingConfig",
    "StakeholderConfig",
    "RegulatoryConfig",
    "DataGovernanceConfig",
    "WebhookConfig",
    "AssuranceConfig",
    "IntensityMetricsConfig",
]
