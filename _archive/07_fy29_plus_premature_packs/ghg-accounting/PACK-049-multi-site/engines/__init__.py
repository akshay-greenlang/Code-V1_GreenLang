"""
PACK-049 GHG Multi-Site Management Pack - Engines
====================================================

This package contains the calculation and processing engines for
multi-site GHG emissions management across an organisation's
portfolio of facilities.

Engines:
    1. SiteRegistryEngine - Site registry and portfolio management
    2. SiteDataCollectionEngine - Activity data collection workflows
    3. SiteBoundaryEngine - Organisational boundary management
    4. RegionalFactorEngine - Regional emission factor assignment
    5. SiteConsolidationEngine - Multi-site emissions consolidation
    6. SiteAllocationEngine - Shared services / landlord-tenant allocation
    7. SiteComparisonEngine - Site benchmarking and comparison
    8. SiteCompletionEngine - Completeness tracking and gap analysis
    9. SiteQualityEngine - Data quality assessment (PCAF scoring)
    10. MultiSiteReportingEngine - Multi-format reporting
"""

__all__ = []

# Engine 1: Site Registry
try:
    from .site_registry_engine import (
        SiteRegistryEngine,
        SiteRecord,
        SiteGroup,
        SiteRegistryResult,
        FacilityCharacteristics,
        SiteClassification,
        SiteCompletenessResult,
        FacilityType,
        LifecycleStatus,
        GroupType,
    )
    __all__ += [
        "SiteRegistryEngine",
        "SiteRecord",
        "SiteGroup",
        "SiteRegistryResult",
        "FacilityCharacteristics",
        "SiteClassification",
        "SiteCompletenessResult",
        "FacilityType",
        "LifecycleStatus",
        "GroupType",
    ]
except ImportError:
    pass

# Engine 2: Data Collection
try:
    from .site_data_collection_engine import (
        SiteDataCollectionEngine,
        DataEntry,
        ValidationRule,
        ValidationResult,
        CollectionTemplate,
        SiteSubmission,
        CollectionRound,
        CollectionResult,
        DataSource,
        SubmissionStatus,
        RoundStatus,
        PeriodType,
        EstimationMethod,
    )
    __all__ += [
        "SiteDataCollectionEngine",
        "DataEntry",
        "ValidationRule",
        "ValidationResult",
        "CollectionTemplate",
        "SiteSubmission",
        "CollectionRound",
        "CollectionResult",
        "DataSource",
        "SubmissionStatus",
        "RoundStatus",
        "PeriodType",
        "EstimationMethod",
    ]
except ImportError:
    pass

# Engine 3: Boundary
try:
    from .site_boundary_engine import (
        SiteBoundaryEngine,
        EntityOwnership,
        BoundaryInclusion,
        BoundaryChange,
        BoundaryDefinition,
        MaterialityResult,
        BoundaryComparison,
        ConsolidationApproach,
        OwnershipType,
        ChangeType,
    )
    __all__ += [
        "SiteBoundaryEngine",
        "EntityOwnership",
        "BoundaryInclusion",
        "BoundaryChange",
        "BoundaryDefinition",
        "MaterialityResult",
        "BoundaryComparison",
        "ConsolidationApproach",
        "OwnershipType",
        "ChangeType",
    ]
except ImportError:
    pass

# Engine 4: Regional Factors
try:
    from .regional_factor_engine import (
        RegionalFactorEngine,
        FactorAssignment,
        GridRegion,
        ClimateZone,
        FactorOverride,
        RegionalFactorResult,
        FactorCoverage,
        FactorType,
        FactorTier,
        FactorSource,
        DEFAULT_GRID_FACTORS,
        DEFAULT_FUEL_FACTORS,
        DEFAULT_FACTOR_DATABASES,
    )
    __all__ += [
        "RegionalFactorEngine",
        "FactorAssignment",
        "GridRegion",
        "ClimateZone",
        "FactorOverride",
        "RegionalFactorResult",
        "FactorCoverage",
        "FactorType",
        "FactorTier",
        "FactorSource",
        "DEFAULT_GRID_FACTORS",
        "DEFAULT_FUEL_FACTORS",
        "DEFAULT_FACTOR_DATABASES",
    ]
except ImportError:
    pass

# Engine 5: Consolidation
try:
    from .site_consolidation_engine import (
        SiteConsolidationEngine,
        SiteTotal,
        EliminationEntry,
        EquityAdjustment,
        ReconciliationResult,
        ConsolidationRun,
        ContributionAnalysis,
        ScopeBreakdown,
        EliminationType,
        ScopeType,
    )
    __all__ += [
        "SiteConsolidationEngine",
        "SiteTotal",
        "EliminationEntry",
        "EquityAdjustment",
        "ReconciliationResult",
        "ConsolidationRun",
        "ContributionAnalysis",
        "ScopeBreakdown",
        "EliminationType",
        "ScopeType",
    ]
except ImportError:
    pass

# Engine 6: Allocation
try:
    from .site_allocation_engine import (
        SiteAllocationEngine,
        AllocationConfig,
        AllocationResult,
        LandlordTenantSplit,
        CogenerationAllocation,
        DistrictConsumption,
        VPPACertificate,
        AllocationSummary,
        CompletenessCheck,
    )
    __all__ += [
        "SiteAllocationEngine",
        "AllocationConfig",
        "AllocationResult",
        "LandlordTenantSplit",
        "CogenerationAllocation",
        "DistrictConsumption",
        "VPPACertificate",
        "AllocationSummary",
        "CompletenessCheck",
    ]
except ImportError:
    pass

# Engine 7: Comparison
try:
    from .site_comparison_engine import (
        SiteComparisonEngine,
    )
    __all__ += ["SiteComparisonEngine"]
except ImportError:
    pass

# Engine 8: Completion
try:
    from .site_completion_engine import (
        SiteCompletionEngine,
    )
    __all__ += ["SiteCompletionEngine"]
except ImportError:
    pass

# Engine 9: Quality
try:
    from .site_quality_engine import (
        SiteQualityEngine,
    )
    __all__ += ["SiteQualityEngine"]
except ImportError:
    pass

# Engine 10: Reporting
try:
    from .multi_site_reporting_engine import (
        MultiSiteReportingEngine,
    )
    __all__ += ["MultiSiteReportingEngine"]
except ImportError:
    pass
