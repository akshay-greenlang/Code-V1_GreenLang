# -*- coding: utf-8 -*-
"""
PACK-009 EU Climate Compliance Bundle integration bridges.

This package provides 10 integration bridges for the EU Climate Compliance
Bundle, coordinating data flow across 4 constituent packs (CSRD, CBAM, EUDR,
EU Taxonomy) with cross-framework mapping, shared data pipelines, consolidated
evidence management, health monitoring, and guided setup.

Integration bridges:
1.  BundlePackOrchestrator - 12-phase pipeline orchestration
2.  CSRDPackBridge - Routes data to/from PACK-001 CSRD Starter
3.  CBAMPackBridge - Routes data to/from PACK-004 CBAM Readiness
4.  EUDRPackBridge - Routes data to/from PACK-006 EUDR Starter
5.  TaxonomyPackBridge - Routes data to/from PACK-008 EU Taxonomy Alignment
6.  CrossFrameworkMapperBridge - Unified field lookup across 4 regulations
7.  SharedDataPipelineBridge - Deduplicated data routing to pack pipelines
8.  ConsolidatedEvidenceBridge - Unified evidence management and reuse
9.  BundleHealthCheckIntegration - 25-category health verification
10. BundleSetupWizard - 10-step guided configuration wizard
"""

from .pack_orchestrator import (
    BundlePackOrchestrator,
    BundleOrchestratorConfig,
    BundleOrchestratorResult,
    BundlePhaseResult,
    BundlePipelinePhase,
)
from .csrd_pack_bridge import (
    CSRDPackBridge,
    CSRDPackBridgeConfig,
)
from .cbam_pack_bridge import (
    CBAMPackBridge,
    CBAMPackBridgeConfig,
)
from .eudr_pack_bridge import (
    EUDRPackBridge,
    EUDRPackBridgeConfig,
)
from .taxonomy_pack_bridge import (
    TaxonomyPackBridge,
    TaxonomyPackBridgeConfig,
)
from .cross_framework_mapper_bridge import (
    CrossFrameworkMapperBridge,
    CrossFrameworkMapperConfig,
)
from .shared_data_pipeline_bridge import (
    SharedDataPipelineBridge,
    SharedDataPipelineConfig,
)
from .consolidated_evidence_bridge import (
    ConsolidatedEvidenceBridge,
    ConsolidatedEvidenceConfig,
)
from .bundle_health_check import (
    BundleHealthCheckIntegration,
    BundleHealthCheckConfig,
    BundleHealthCheckResult,
)
from .setup_wizard import (
    BundleSetupWizard,
    BundleSetupWizardConfig,
    WizardResult,
)

__all__ = [
    # 1. Pack Orchestrator
    "BundlePackOrchestrator",
    "BundleOrchestratorConfig",
    "BundleOrchestratorResult",
    "BundlePhaseResult",
    "BundlePipelinePhase",
    # 2. CSRD Pack Bridge
    "CSRDPackBridge",
    "CSRDPackBridgeConfig",
    # 3. CBAM Pack Bridge
    "CBAMPackBridge",
    "CBAMPackBridgeConfig",
    # 4. EUDR Pack Bridge
    "EUDRPackBridge",
    "EUDRPackBridgeConfig",
    # 5. Taxonomy Pack Bridge
    "TaxonomyPackBridge",
    "TaxonomyPackBridgeConfig",
    # 6. Cross-Framework Mapper
    "CrossFrameworkMapperBridge",
    "CrossFrameworkMapperConfig",
    # 7. Shared Data Pipeline
    "SharedDataPipelineBridge",
    "SharedDataPipelineConfig",
    # 8. Consolidated Evidence
    "ConsolidatedEvidenceBridge",
    "ConsolidatedEvidenceConfig",
    # 9. Bundle Health Check
    "BundleHealthCheckIntegration",
    "BundleHealthCheckConfig",
    "BundleHealthCheckResult",
    # 10. Setup Wizard
    "BundleSetupWizard",
    "BundleSetupWizardConfig",
    "WizardResult",
]
