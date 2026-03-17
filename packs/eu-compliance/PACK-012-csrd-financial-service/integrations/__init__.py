# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Integrations Module
============================================================

This package provides the integration layer for the CSRD Financial Service
Pack (PACK-012). It contains 10 modules covering orchestration, cross-pack
bridges (CSRD, SFDR, Taxonomy), agent bridges (MRV, Finance), FI-specific
bridges (Climate Risk, EBA Pillar 3), health verification, and guided setup.

Modules:
    pack_orchestrator       - 11-phase FI CSRD execution pipeline
    csrd_pack_bridge        - PACK-001/002/003 CSRD core integration
    sfdr_pack_bridge        - PACK-010/011 SFDR Article 8/9 integration
    taxonomy_pack_bridge    - PACK-008 EU Taxonomy alignment integration
    mrv_investments_bridge  - AGENT-MRV-028 financed emissions bridge
    finance_agent_bridge    - greenlang.agents.finance green screening bridge
    climate_risk_bridge     - Transition + physical climate risk bridge
    eba_pillar3_bridge      - EBA Pillar 3 ITS template bridge
    health_check            - 22-category system verification
    setup_wizard            - 8-step guided FI configuration

Exported Classes:
    From pack_orchestrator:
        FSCSRDOrchestrator
        FSOrchestrationConfig
        PipelineResult
        PipelinePhase
        PhaseResult

    From csrd_pack_bridge:
        CSRDPackBridge
        CSRDBridgeConfig
        ESRSCoreResult
        QualityGateResult

    From sfdr_pack_bridge:
        SFDRPackBridge
        SFDRBridgeConfig
        PAIDataResult
        CarbonFootprintResult

    From taxonomy_pack_bridge:
        TaxonomyPackBridge
        TaxonomyBridgeConfig
        TaxonomyAssessmentResult
        EligibilityScreenResult

    From mrv_investments_bridge:
        MRVInvestmentsBridge
        MRVInvestmentsBridgeConfig
        AssetClassResult
        FinancedEmissionsResult

    From finance_agent_bridge:
        FinanceAgentBridge
        FinanceAgentBridgeConfig
        GreenScreeningResult
        StrandedAssetResult

    From climate_risk_bridge:
        ClimateRiskBridge
        ClimateRiskBridgeConfig
        TransitionRiskResult
        PhysicalRiskResult
        CombinedClimateRiskResult

    From eba_pillar3_bridge:
        EBAPillar3Bridge
        EBAPillar3BridgeConfig
        TemplateResult
        Pillar3Result

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

Example:
    >>> from packs.eu_compliance.PACK_012_csrd_financial_service.integrations import (
    ...     FSCSRDOrchestrator,
    ...     HealthCheck,
    ...     SetupWizard,
    ... )
    >>> wizard = SetupWizard()
    >>> result = wizard.execute_all_steps(institution_data)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Version: 1.0.0
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from pack_orchestrator
# ---------------------------------------------------------------------------

from .pack_orchestrator import (  # noqa: F401
    FSOrchestrationConfig,
    PhaseResult,
    PipelinePhase,
    PipelineResult,
)

try:
    from .pack_orchestrator import FSCSRDOrchestrator  # noqa: F401
except ImportError:
    FSCSRDOrchestrator = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Imports from csrd_pack_bridge
# ---------------------------------------------------------------------------

from .csrd_pack_bridge import (  # noqa: F401
    CSRDBridgeConfig,
    CSRDPackBridge,
    ESRSCoreResult,
    QualityGateResult,
)

# ---------------------------------------------------------------------------
# Imports from sfdr_pack_bridge
# ---------------------------------------------------------------------------

from .sfdr_pack_bridge import (  # noqa: F401
    CarbonFootprintResult,
    PAIDataResult,
    SFDRBridgeConfig,
    SFDRPackBridge,
)

# ---------------------------------------------------------------------------
# Imports from taxonomy_pack_bridge
# ---------------------------------------------------------------------------

from .taxonomy_pack_bridge import (  # noqa: F401
    EligibilityScreenResult,
    TaxonomyAssessmentResult,
    TaxonomyBridgeConfig,
    TaxonomyPackBridge,
)

# ---------------------------------------------------------------------------
# Imports from mrv_investments_bridge
# ---------------------------------------------------------------------------

from .mrv_investments_bridge import (  # noqa: F401
    AssetClassResult,
    FinancedEmissionsResult,
    MRVInvestmentsBridge,
    MRVInvestmentsBridgeConfig,
)

# ---------------------------------------------------------------------------
# Imports from finance_agent_bridge
# ---------------------------------------------------------------------------

from .finance_agent_bridge import (  # noqa: F401
    FinanceAgentBridge,
    FinanceAgentBridgeConfig,
    GreenScreeningResult,
    StrandedAssetResult,
)

# ---------------------------------------------------------------------------
# Imports from climate_risk_bridge
# ---------------------------------------------------------------------------

from .climate_risk_bridge import (  # noqa: F401
    ClimateRiskBridge,
    ClimateRiskBridgeConfig,
    CombinedClimateRiskResult,
    PhysicalRiskResult,
    TransitionRiskResult,
)

# ---------------------------------------------------------------------------
# Imports from eba_pillar3_bridge
# ---------------------------------------------------------------------------

from .eba_pillar3_bridge import (  # noqa: F401
    EBAPillar3Bridge,
    EBAPillar3BridgeConfig,
    Pillar3Result,
    TemplateResult,
)

# ---------------------------------------------------------------------------
# Imports from health_check
# ---------------------------------------------------------------------------

from .health_check import (  # noqa: F401
    CategoryResult,
    HealthCheck,
    HealthCheckConfig,
    HealthCheckResult,
)

# ---------------------------------------------------------------------------
# Imports from setup_wizard
# ---------------------------------------------------------------------------

from .setup_wizard import (  # noqa: F401
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
    "FSCSRDOrchestrator",
    "FSOrchestrationConfig",
    "PipelineResult",
    "PipelinePhase",
    "PhaseResult",
    # csrd_pack_bridge
    "CSRDPackBridge",
    "CSRDBridgeConfig",
    "ESRSCoreResult",
    "QualityGateResult",
    # sfdr_pack_bridge
    "SFDRPackBridge",
    "SFDRBridgeConfig",
    "PAIDataResult",
    "CarbonFootprintResult",
    # taxonomy_pack_bridge
    "TaxonomyPackBridge",
    "TaxonomyBridgeConfig",
    "TaxonomyAssessmentResult",
    "EligibilityScreenResult",
    # mrv_investments_bridge
    "MRVInvestmentsBridge",
    "MRVInvestmentsBridgeConfig",
    "AssetClassResult",
    "FinancedEmissionsResult",
    # finance_agent_bridge
    "FinanceAgentBridge",
    "FinanceAgentBridgeConfig",
    "GreenScreeningResult",
    "StrandedAssetResult",
    # climate_risk_bridge
    "ClimateRiskBridge",
    "ClimateRiskBridgeConfig",
    "TransitionRiskResult",
    "PhysicalRiskResult",
    "CombinedClimateRiskResult",
    # eba_pillar3_bridge
    "EBAPillar3Bridge",
    "EBAPillar3BridgeConfig",
    "TemplateResult",
    "Pillar3Result",
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
]

logger.debug(
    "PACK-012 integrations loaded: %d exports from 10 modules",
    len(__all__),
)
