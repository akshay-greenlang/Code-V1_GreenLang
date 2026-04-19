# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Integrations Module
====================================================

This package provides the integration layer for the SFDR Article 9 Pack
(PACK-011). It contains 10 modules covering orchestration, cross-pack
bridges, data intake bridges, regulatory tracking, health verification,
and guided setup.

Modules:
    pack_orchestrator       - 11-phase Article 9 execution pipeline
    article8_pack_bridge    - PACK-010 cross-reference and downgrade detection
    taxonomy_pack_bridge    - PACK-008 EU Taxonomy alignment integration
    mrv_emissions_bridge    - Bridge to 30 MRV agents for PAI 1-6
    benchmark_data_bridge   - CTB/PAB benchmark data intake (Art 9(3))
    impact_data_bridge      - Impact data intake and SDG alignment
    eet_data_bridge         - EET (European ESG Template) import/export
    regulatory_bridge       - SFDR/Taxonomy/BMR regulatory update tracking
    health_check            - 20-category system verification
    setup_wizard            - 8-step guided product configuration

Exported Classes:
    From pack_orchestrator:
        Article9Orchestrator
        Article9OrchestrationConfig
        PipelineResult
        PipelinePhase
        PhaseResult

    From article8_pack_bridge:
        Article8PackBridge
        Article8BridgeConfig
        DowngradeAssessment
        ClassificationComparison
        SharedPAIResult

    From taxonomy_pack_bridge:
        TaxonomyPackBridge
        TaxonomyBridgeConfig
        TaxonomyAlignmentData
        SafeguardsResult

    From mrv_emissions_bridge:
        MRVEmissionsBridge
        MRVBridgeConfig
        PAIRoutingResult
        EmissionsAggregate
        MRVAgentMapping

    From benchmark_data_bridge:
        BenchmarkDataBridge
        BenchmarkDataConfig
        BenchmarkIndex
        UniverseData
        BenchmarkPerformance

    From impact_data_bridge:
        ImpactDataBridge
        ImpactDataConfig
        InvesteeImpactData
        SDGAlignmentData
        ImpactVerification

    From eet_data_bridge:
        EETDataBridge
        EETBridgeConfig
        EETImportResult
        EETExportResult
        Article9EETFields

    From regulatory_bridge:
        RegulatoryBridge
        RegulatoryBridgeConfig
        UpdateCheckResult
        RegulatoryEvent
        ComplianceDeadline

    From health_check:
        HealthCheck
        HealthCheckConfig
        HealthCheckResult
        CategoryResult

    From setup_wizard:
        SetupWizard
        SetupWizardConfig
        WizardStep
        WizardResult
        ProductType

Example:
    >>> from packs.eu_compliance.PACK_011_sfdr_article_9.integrations import (
    ...     Article9Orchestrator,
    ...     MRVEmissionsBridge,
    ...     HealthCheck,
    ...     SetupWizard,
    ... )
    >>> wizard = SetupWizard()
    >>> result = wizard.execute_all_steps(product_data)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Version: 1.0.0
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependencies and heavy module loading at
# import time.  Each bridge/orchestrator is imported on first access via the
# public ``__all__`` list or direct attribute reference.
# ---------------------------------------------------------------------------

# pack_orchestrator
from .pack_orchestrator import (  # noqa: F401
    Article9OrchestrationConfig,
    PhaseResult,
    PipelinePhase,
    PipelineResult,
)

# We alias the orchestrator class if available, but guard against import issues
try:
    from .pack_orchestrator import Article9Orchestrator  # noqa: F401
except ImportError:
    Article9Orchestrator = None  # type: ignore[assignment,misc]

# article8_pack_bridge
from .article8_pack_bridge import (  # noqa: F401
    Article8BridgeConfig,
    Article8PackBridge,
    ClassificationComparison,
    DowngradeAssessment,
    SharedPAIResult,
)

# taxonomy_pack_bridge
from .taxonomy_pack_bridge import (  # noqa: F401
    SafeguardsResult,
    TaxonomyAlignmentData,
    TaxonomyBridgeConfig,
    TaxonomyPackBridge,
)

# mrv_emissions_bridge
from .mrv_emissions_bridge import (  # noqa: F401
    EmissionsAggregate,
    MRVAgentMapping,
    MRVBridgeConfig,
    MRVEmissionsBridge,
    PAIRoutingResult,
)

# benchmark_data_bridge
from .benchmark_data_bridge import (  # noqa: F401
    BenchmarkDataBridge,
    BenchmarkDataConfig,
    BenchmarkIndex,
    BenchmarkPerformance,
    UniverseData,
)

# impact_data_bridge
from .impact_data_bridge import (  # noqa: F401
    ImpactDataBridge,
    ImpactDataConfig,
    ImpactVerification,
    InvesteeImpactData,
    SDGAlignmentData,
)

# eet_data_bridge
from .eet_data_bridge import (  # noqa: F401
    Article9EETFields,
    EETBridgeConfig,
    EETDataBridge,
    EETExportResult,
    EETImportResult,
)

# regulatory_bridge
from .regulatory_bridge import (  # noqa: F401
    ComplianceDeadline,
    RegulatoryBridge,
    RegulatoryBridgeConfig,
    RegulatoryEvent,
    UpdateCheckResult,
)

# health_check
from .health_check import (  # noqa: F401
    CategoryResult,
    HealthCheck,
    HealthCheckConfig,
    HealthCheckResult,
)

# setup_wizard
from .setup_wizard import (  # noqa: F401
    ProductType,
    SetupWizard,
    SetupWizardConfig,
    WizardResult,
    WizardStep,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # pack_orchestrator
    "Article9Orchestrator",
    "Article9OrchestrationConfig",
    "PipelineResult",
    "PipelinePhase",
    "PhaseResult",
    # article8_pack_bridge
    "Article8PackBridge",
    "Article8BridgeConfig",
    "DowngradeAssessment",
    "ClassificationComparison",
    "SharedPAIResult",
    # taxonomy_pack_bridge
    "TaxonomyPackBridge",
    "TaxonomyBridgeConfig",
    "TaxonomyAlignmentData",
    "SafeguardsResult",
    # mrv_emissions_bridge
    "MRVEmissionsBridge",
    "MRVBridgeConfig",
    "PAIRoutingResult",
    "EmissionsAggregate",
    "MRVAgentMapping",
    # benchmark_data_bridge
    "BenchmarkDataBridge",
    "BenchmarkDataConfig",
    "BenchmarkIndex",
    "UniverseData",
    "BenchmarkPerformance",
    # impact_data_bridge
    "ImpactDataBridge",
    "ImpactDataConfig",
    "InvesteeImpactData",
    "SDGAlignmentData",
    "ImpactVerification",
    # eet_data_bridge
    "EETDataBridge",
    "EETBridgeConfig",
    "EETImportResult",
    "EETExportResult",
    "Article9EETFields",
    # regulatory_bridge
    "RegulatoryBridge",
    "RegulatoryBridgeConfig",
    "UpdateCheckResult",
    "RegulatoryEvent",
    "ComplianceDeadline",
    # health_check
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "CategoryResult",
    # setup_wizard
    "SetupWizard",
    "SetupWizardConfig",
    "WizardStep",
    "WizardResult",
    "ProductType",
]

logger.debug(
    "PACK-011 integrations loaded: %d exports from 10 modules",
    len(__all__),
)
