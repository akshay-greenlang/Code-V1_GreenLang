"""
PACK-013 CSRD Manufacturing Pack - Integrations Package.

Exports all integration classes from the 8 integration modules:
  1. pack_orchestrator          -- 11-phase pipeline orchestrator
  2. csrd_pack_bridge           -- Bridge to PACK-001/002/003 CSRD packs
  3. cbam_pack_bridge           -- Bridge to PACK-004/005 CBAM packs
  4. mrv_industrial_bridge      -- Routing to AGENT-MRV 001-030
  5. data_manufacturing_bridge  -- Routing to AGENT-DATA agents
  6. eu_ets_bridge              -- EU ETS compliance integration
  7. taxonomy_bridge            -- EU Taxonomy alignment assessment
  8. health_check               -- 22-category health verification
  9. setup_wizard               -- 8-step guided setup
"""

# ---------------------------------------------------------------------------
# pack_orchestrator
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.pack_orchestrator import (
    CSRDManufacturingOrchestrator,
    OrchestratorConfig,
    RetryPolicy,
    PhaseResult,
    PhaseProvenance,
    PipelineResult,
    PhaseStatus,
    PipelinePhase,
    PHASE_DEPENDENCIES,
)

# ---------------------------------------------------------------------------
# csrd_pack_bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.csrd_pack_bridge import (
    CSRDPackBridge,
    CSRDBridgeConfig,
    CSRDBridgeResult,
    CSRDPackTier,
    ESRSStandard,
    ESRSDisclosureData,
    MaterialityItem,
)

# ---------------------------------------------------------------------------
# cbam_pack_bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.cbam_pack_bridge import (
    CBAMPackBridge,
    CBAMBridgeConfig,
    CBAMBridgeResult,
    CBAMPhase,
    CBAMGoodCategory,
    EmbeddedEmissionsRecord,
)

# ---------------------------------------------------------------------------
# mrv_industrial_bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.mrv_industrial_bridge import (
    MRVIndustrialBridge,
    MRVBridgeConfig,
    MRVRouting,
    DEFAULT_ROUTING_TABLE,
    SUB_SECTOR_AGENTS,
)

# ---------------------------------------------------------------------------
# data_manufacturing_bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.data_manufacturing_bridge import (
    DataManufacturingBridge,
    DataBridgeConfig,
    DataRouting,
    ERP_FIELD_MAP,
)

# ---------------------------------------------------------------------------
# eu_ets_bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.eu_ets_bridge import (
    EUETSBridge,
    ETSBridgeConfig,
    ETSComplianceResult,
    BenchmarkComparison,
    PRODUCT_BENCHMARKS,
)

# ---------------------------------------------------------------------------
# taxonomy_bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.taxonomy_bridge import (
    TaxonomyBridge,
    TaxonomyBridgeConfig,
    TaxonomyAlignmentResult,
    ActivityAssessment,
    ObjectiveResult,
    EnvironmentalObjective,
    AlignmentStatus,
    MANUFACTURING_ACTIVITIES,
)

# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.health_check import (
    ManufacturingHealthCheck,
    HealthCheckResult,
    CategoryResult,
    CheckDetail,
    HealthStatus,
    HealthCategory,
)

# ---------------------------------------------------------------------------
# setup_wizard
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.setup_wizard import (
    ManufacturingSetupWizard,
    WizardState,
    WizardResult,
    SetupStep,
    SubSector,
    ERPSystem,
    CompanyProfile,
    FacilityData,
    DataSourceConfig,
    BaselineData,
    TargetData,
    WorkflowConfig,
    STEP_ORDER,
    STEP_DEPENDENCIES,
    SUB_SECTOR_PRESETS,
)

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
__all__ = [
    # pack_orchestrator
    "CSRDManufacturingOrchestrator",
    "OrchestratorConfig",
    "RetryPolicy",
    "PhaseResult",
    "PhaseProvenance",
    "PipelineResult",
    "PhaseStatus",
    "PipelinePhase",
    "PHASE_DEPENDENCIES",
    # csrd_pack_bridge
    "CSRDPackBridge",
    "CSRDBridgeConfig",
    "CSRDBridgeResult",
    "CSRDPackTier",
    "ESRSStandard",
    "ESRSDisclosureData",
    "MaterialityItem",
    # cbam_pack_bridge
    "CBAMPackBridge",
    "CBAMBridgeConfig",
    "CBAMBridgeResult",
    "CBAMPhase",
    "CBAMGoodCategory",
    "EmbeddedEmissionsRecord",
    # mrv_industrial_bridge
    "MRVIndustrialBridge",
    "MRVBridgeConfig",
    "MRVRouting",
    "DEFAULT_ROUTING_TABLE",
    "SUB_SECTOR_AGENTS",
    # data_manufacturing_bridge
    "DataManufacturingBridge",
    "DataBridgeConfig",
    "DataRouting",
    "ERP_FIELD_MAP",
    # eu_ets_bridge
    "EUETSBridge",
    "ETSBridgeConfig",
    "ETSComplianceResult",
    "BenchmarkComparison",
    "PRODUCT_BENCHMARKS",
    # taxonomy_bridge
    "TaxonomyBridge",
    "TaxonomyBridgeConfig",
    "TaxonomyAlignmentResult",
    "ActivityAssessment",
    "ObjectiveResult",
    "EnvironmentalObjective",
    "AlignmentStatus",
    "MANUFACTURING_ACTIVITIES",
    # health_check
    "ManufacturingHealthCheck",
    "HealthCheckResult",
    "CategoryResult",
    "CheckDetail",
    "HealthStatus",
    "HealthCategory",
    # setup_wizard
    "ManufacturingSetupWizard",
    "WizardState",
    "WizardResult",
    "SetupStep",
    "SubSector",
    "ERPSystem",
    "CompanyProfile",
    "FacilityData",
    "DataSourceConfig",
    "BaselineData",
    "TargetData",
    "WorkflowConfig",
    "STEP_ORDER",
    "STEP_DEPENDENCIES",
    "SUB_SECTOR_PRESETS",
]
