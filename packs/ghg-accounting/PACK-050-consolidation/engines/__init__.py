"""
PACK-050 GHG Consolidation Pack - Engines
====================================================

This package contains the calculation and processing engines for
multi-entity corporate GHG consolidation per GHG Protocol
Corporate Standard Chapter 3.

Engines:
    1. EntityRegistryEngine - Corporate entity hierarchy management
    2. OwnershipStructureEngine - Equity chain resolution and control
    3. BoundaryConsolidationEngine - Organizational boundary consolidation
    4. EquityShareEngine - Equity share approach calculations
    5. ControlApproachEngine - Operational/financial control consolidation
    6. IntercompanyEliminationEngine - Double-counting elimination
    7. AcquisitionDivestitureEngine - M&A event handling
    8. ConsolidationAdjustmentEngine - Manual adjustments and corrections
    9. GroupReportingEngine - Consolidated group GHG reporting
    10. ConsolidationAuditEngine - Complete audit trail
"""

__all__: list = []

# Engine 1: Entity Registry
try:
    from .entity_registry_engine import (
        EntityRegistryEngine,
        EntityRecord,
        EntityHierarchy,
        EntitySearchResult,
        EntityRegistryStats,
        EntityType,
        EntityStatus,
    )
    __all__ += [
        "EntityRegistryEngine",
        "EntityRecord",
        "EntityHierarchy",
        "EntitySearchResult",
        "EntityRegistryStats",
        "EntityType",
        "EntityStatus",
    ]
except ImportError:
    pass

# Engine 2: Ownership Structure
try:
    from .ownership_structure_engine import (
        OwnershipStructureEngine,
        OwnershipRecord,
        EquityChain,
        ControlAssessment,
        OwnershipChange,
        ControlType,
        OwnershipCategory,
        ChangeReason,
    )
    __all__ += [
        "OwnershipStructureEngine",
        "OwnershipRecord",
        "EquityChain",
        "ControlAssessment",
        "OwnershipChange",
        "ControlType",
        "OwnershipCategory",
        "ChangeReason",
    ]
except ImportError:
    pass

# Engine 3: Boundary Consolidation
try:
    from .boundary_consolidation_engine import (
        BoundaryConsolidationEngine,
        BoundaryDefinition,
        EntityInclusion,
        BoundaryComparison,
        BoundaryLock,
        ConsolidationApproach,
        BoundaryStatus,
        ChangeJustification,
    )
    __all__ += [
        "BoundaryConsolidationEngine",
        "BoundaryDefinition",
        "EntityInclusion",
        "BoundaryComparison",
        "BoundaryLock",
        "ConsolidationApproach",
        "BoundaryStatus",
        "ChangeJustification",
    ]
except ImportError:
    pass

# Engine 4: Equity Share
try:
    from .equity_share_engine import (
        EquityShareEngine,
        EquityShareInput,
        EquityShareResult,
        EntityEquityContribution,
        EquityReconciliation,
    )
    __all__ += [
        "EquityShareEngine",
        "EquityShareInput",
        "EquityShareResult",
        "EntityEquityContribution",
        "EquityReconciliation",
    ]
except ImportError:
    pass

# Engine 5: Control Approach
try:
    from .control_approach_engine import (
        ControlApproachEngine,
        ControlInput,
        ControlResult,
        ControlAssessmentDetail,
        FranchiseBoundary,
        LeaseBoundary,
        ControlApproachType,
        FranchiseRole,
        LeaseType,
        LeaseRole,
        OutsourceType,
    )
    __all__ += [
        "ControlApproachEngine",
        "ControlInput",
        "ControlResult",
        "ControlAssessmentDetail",
        "FranchiseBoundary",
        "LeaseBoundary",
        "ControlApproachType",
        "FranchiseRole",
        "LeaseType",
        "LeaseRole",
        "OutsourceType",
    ]
except ImportError:
    pass

# Engine 6: Intercompany Elimination
try:
    from .intercompany_elimination_engine import (
        IntercompanyEliminationEngine,
        TransferRecord,
        EliminationEntry,
        EliminationResult,
        TransferReconciliation,
        TransferType,
        EliminationScope,
        ReconciliationStatus,
    )
    __all__ += [
        "IntercompanyEliminationEngine",
        "TransferRecord",
        "EliminationEntry",
        "EliminationResult",
        "TransferReconciliation",
        "TransferType",
        "EliminationScope",
        "ReconciliationStatus",
    ]
except ImportError:
    pass

# Engine 7: Acquisition / Divestiture
try:
    from .acquisition_divestiture_engine import (
        AcquisitionDivestitureEngine,
        MnAEvent,
        ProRataCalculation,
        BaseYearRestatement,
        StructuralChangeRecord,
        OrganicGrowthAnalysis,
        MnAEventType,
    )
    __all__ += [
        "AcquisitionDivestitureEngine",
        "MnAEvent",
        "ProRataCalculation",
        "BaseYearRestatement",
        "StructuralChangeRecord",
        "OrganicGrowthAnalysis",
        "MnAEventType",
    ]
except ImportError:
    pass

# Engine 8: Consolidation Adjustments
try:
    from .consolidation_adjustment_engine import (
        ConsolidationAdjustmentEngine,
        AdjustmentRecord,
        AdjustmentApproval,
        AdjustmentImpact,
        AdjustmentBatch,
        AdjustmentCategory,
        AdjustmentStatus,
    )
    __all__ += [
        "ConsolidationAdjustmentEngine",
        "AdjustmentRecord",
        "AdjustmentApproval",
        "AdjustmentImpact",
        "AdjustmentBatch",
        "AdjustmentCategory",
        "AdjustmentStatus",
    ]
except ImportError:
    pass

# Engine 9: Group Reporting
try:
    from .group_reporting_engine import (
        GroupReportingEngine,
        GroupReport,
        FrameworkMapping,
        ScopeBreakdown,
        TrendData,
        ContributionWaterfall,
        GeographicBreakdown,
        ReportingFramework,
    )
    __all__ += [
        "GroupReportingEngine",
        "GroupReport",
        "FrameworkMapping",
        "ScopeBreakdown",
        "TrendData",
        "ContributionWaterfall",
        "GeographicBreakdown",
        "ReportingFramework",
    ]
except ImportError:
    pass

# Engine 10: Consolidation Audit
try:
    from .consolidation_audit_engine import (
        ConsolidationAuditEngine,
        AuditEntry,
        ReconciliationResult,
        CompletenessCheck,
        SignOff,
        AuditFinding,
        AssurancePackage,
        AuditStepType,
        FindingSeverity,
        SignOffLevel,
    )
    __all__ += [
        "ConsolidationAuditEngine",
        "AuditEntry",
        "ReconciliationResult",
        "CompletenessCheck",
        "SignOff",
        "AuditFinding",
        "AssurancePackage",
        "AuditStepType",
        "FindingSeverity",
        "SignOffLevel",
    ]
except ImportError:
    pass
