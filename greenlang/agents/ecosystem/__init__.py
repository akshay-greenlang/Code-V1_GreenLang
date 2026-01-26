# -*- coding: utf-8 -*-
"""
GreenLang Ecosystem Layer Agents
================================

The Ecosystem Layer provides developer tools, marketplace integration,
and platform extension capabilities for GreenLang Climate OS.

Agents:
    GL-ECO-X-001: Agent SDK Agent - SDK for building custom agents
    GL-ECO-X-002: Pack Builder Agent - Builds solution packs
    GL-ECO-X-003: Marketplace Agent - GreenLang Hub marketplace integration
    GL-ECO-X-004: Partner Integration Agent - Partner API integration
    GL-ECO-X-005: Documentation Generator - Auto-generates documentation
    GL-ECO-X-006: Training Data Generator - Generates training datasets
    GL-ECO-X-007: Agent Performance Monitor - Monitors agent performance
"""

from greenlang.agents.ecosystem.agent_sdk_agent import (
    AgentSDKAgent,
    AgentSDKInput,
    AgentSDKOutput,
    AgentTemplate,
    AgentDefinition,
    TemplateType,
    ValidationResult,
    CodeGenerationResult,
)

from greenlang.agents.ecosystem.pack_builder_agent import (
    PackBuilderAgent,
    PackBuilderInput,
    PackBuilderOutput,
    SolutionPack,
    PackComponent,
    PackManifest,
    PackType,
    BuildStatus,
)

from greenlang.agents.ecosystem.marketplace_agent import (
    MarketplaceAgent,
    MarketplaceInput,
    MarketplaceOutput,
    MarketplaceListing,
    ListingCategory,
    ListingStatus,
    Review,
    InstallationStatus,
)

from greenlang.agents.ecosystem.partner_integration_agent import (
    PartnerIntegrationAgent,
    PartnerIntegrationInput,
    PartnerIntegrationOutput,
    PartnerConnection,
    IntegrationType,
    ConnectionStatus,
    DataMapping,
    SyncConfiguration,
)

from greenlang.agents.ecosystem.documentation_generator import (
    DocumentationGenerator,
    DocumentationInput,
    DocumentationOutput,
    DocumentationType,
    DocumentationSection,
    APIDocumentation,
    UserGuide,
)

from greenlang.agents.ecosystem.training_data_generator import (
    TrainingDataGenerator,
    TrainingDataInput,
    TrainingDataOutput,
    TrainingDataset,
    DatasetType,
    DataQuality,
    SyntheticDataConfig,
)

from greenlang.agents.ecosystem.agent_performance_monitor import (
    AgentPerformanceMonitor,
    PerformanceMonitorInput,
    PerformanceMonitorOutput,
    AgentPerformanceMetrics,
    PerformanceThreshold,
    PerformanceAlert,
    ResourceUsage,
)

__all__ = [
    # Agent SDK Agent (GL-ECO-X-001)
    "AgentSDKAgent",
    "AgentSDKInput",
    "AgentSDKOutput",
    "AgentTemplate",
    "AgentDefinition",
    "TemplateType",
    "ValidationResult",
    "CodeGenerationResult",
    # Pack Builder Agent (GL-ECO-X-002)
    "PackBuilderAgent",
    "PackBuilderInput",
    "PackBuilderOutput",
    "SolutionPack",
    "PackComponent",
    "PackManifest",
    "PackType",
    "BuildStatus",
    # Marketplace Agent (GL-ECO-X-003)
    "MarketplaceAgent",
    "MarketplaceInput",
    "MarketplaceOutput",
    "MarketplaceListing",
    "ListingCategory",
    "ListingStatus",
    "Review",
    "InstallationStatus",
    # Partner Integration Agent (GL-ECO-X-004)
    "PartnerIntegrationAgent",
    "PartnerIntegrationInput",
    "PartnerIntegrationOutput",
    "PartnerConnection",
    "IntegrationType",
    "ConnectionStatus",
    "DataMapping",
    "SyncConfiguration",
    # Documentation Generator (GL-ECO-X-005)
    "DocumentationGenerator",
    "DocumentationInput",
    "DocumentationOutput",
    "DocumentationType",
    "DocumentationSection",
    "APIDocumentation",
    "UserGuide",
    # Training Data Generator (GL-ECO-X-006)
    "TrainingDataGenerator",
    "TrainingDataInput",
    "TrainingDataOutput",
    "TrainingDataset",
    "DatasetType",
    "DataQuality",
    "SyntheticDataConfig",
    # Agent Performance Monitor (GL-ECO-X-007)
    "AgentPerformanceMonitor",
    "PerformanceMonitorInput",
    "PerformanceMonitorOutput",
    "AgentPerformanceMetrics",
    "PerformanceThreshold",
    "PerformanceAlert",
    "ResourceUsage",
]
