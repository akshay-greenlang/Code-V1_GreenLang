# -*- coding: utf-8 -*-
"""
GreenLang MRV (Monitoring, Reporting, Verification) Agents
==========================================================

This package contains agents for measuring, reporting, and verifying
greenhouse gas emissions across various sectors.

Core MRV Agents (GL-MRV-X-001 to GL-MRV-X-030):
-----------------------------------------------
Scope 1 Agents:
    - GL-MRV-X-001: Scope 1 Combustion Calculator
    - GL-MRV-X-002: Refrigerants & F-Gas Agent
    - GL-MRV-X-015: Process Emissions Agent

Scope 2 Agents:
    - GL-MRV-X-003: Scope 2 Location-Based Agent
    - GL-MRV-X-004: Scope 2 Market-Based Agent

Scope 3 Agents:
    - GL-MRV-X-005: Scope 3 Category Mapper
    - GL-MRV-X-016 to GL-MRV-X-028: Category-specific calculators

Cross-Cutting Agents:
    - GL-MRV-X-006: Uncertainty & Data Quality Agent
    - GL-MRV-X-007: Audit Trail & Lineage Agent
    - GL-MRV-X-008: Consolidation & Roll-up Agent
    - GL-MRV-X-009: Baseline & Target Tracker
    - GL-MRV-X-010: Inventory Boundary Agent
    - GL-MRV-X-011: Temporal Alignment Agent
    - GL-MRV-X-012: Activity Data Validation Agent
    - GL-MRV-X-013: Emission Factor Selection Agent
    - GL-MRV-X-014: GWP Application Agent
    - GL-MRV-X-029: Biogenic Carbon Tracker
    - GL-MRV-X-030: Removals & Offsets Tracker

Subpackages:
- industrial: Industrial sector MRV agents (steel, cement, chemicals, etc.)
- transport: Transport sector MRV agents (road, aviation, maritime, rail, etc.)
- agriculture: Agriculture sector MRV agents (crops, livestock, fertilizer, etc.)
- energy: Energy sector MRV agents (power generation, grid, renewable, etc.)

All MRV agents follow the CRITICAL PATH pattern with:
- Zero-hallucination guarantee (no LLM in calculation path)
- Full audit trail with SHA-256 provenance hashing
- GHG Protocol and regulatory compliance
- Deterministic, reproducible calculations
"""

from typing import List

# =============================================================================
# Core MRV Agents (GL-MRV-X-001 to GL-MRV-X-030)
# =============================================================================

# Scope 1 Agents
from greenlang.agents.mrv.scope1_combustion import (
    Scope1CombustionAgent,
    CombustionType,
    FuelType,
    EmissionGas,
    FuelConsumption,
    CombustionCalculationResult,
    Scope1CombustionInput,
    Scope1CombustionOutput,
    STATIONARY_EMISSION_FACTORS,
    MOBILE_EMISSION_FACTORS,
)

from greenlang.agents.mrv.refrigerants_fgas import (
    RefrigerantsFGasAgent,
    RefrigerantType,
    EquipmentType,
    CalculationMethod,
    RefrigerantInventory,
    MassBalanceInput,
    FGasEmissionResult,
    RefrigerantsFGasInput,
    RefrigerantsFGasOutput,
    GWP_REFRIGERANTS,
)

# Scope 2 Agents
from greenlang.agents.mrv.scope2_location_based import (
    Scope2LocationBasedAgent,
    EnergyType,
    EnergyConsumption,
    LocationBasedResult,
    Scope2LocationBasedInput,
    Scope2LocationBasedOutput,
    GRID_EMISSION_FACTORS,
)

from greenlang.agents.mrv.scope2_market_based import (
    Scope2MarketBasedAgent,
    InstrumentType,
    EnergySource,
    ContractualInstrument,
    EnergyPurchase,
    MarketBasedResult,
    Scope2MarketBasedInput,
    Scope2MarketBasedOutput,
    RESIDUAL_MIX_FACTORS,
)

# Scope 3 Category Mapper
from greenlang.agents.mrv.scope3_category_mapper import (
    Scope3CategoryMapperAgent,
    Scope3Category,
    DataSourceType,
    CalculationApproach,
    SpendRecord,
    PurchaseOrder,
    BOMItem,
    CategoryMappingResult,
    Scope3CategoryMapperInput,
    Scope3CategoryMapperOutput,
)

# Cross-Cutting Agents
from greenlang.agents.mrv.uncertainty_data_quality import (
    UncertaintyDataQualityAgent,
    DataQualityIndicator,
    UncertaintyType,
    DistributionType,
    UncertaintyInput,
    DataQualityAssessment,
    UncertaintyResult,
    DataQualityResult,
    UncertaintyDataQualityInput,
    UncertaintyDataQualityOutput,
)

from greenlang.agents.mrv.audit_trail_lineage import (
    AuditTrailLineageAgent,
    LineageEventType,
    DataSource as LineageDataSource,
    LineageNode,
    LineageEdge,
    AuditEntry,
    LineageGraph,
    AuditTrailLineageInput,
    AuditTrailLineageOutput,
)

from greenlang.agents.mrv.consolidation_rollup import (
    ConsolidationRollupAgent,
    ConsolidationApproach,
    EntityType,
    EntityEmissions,
    ConsolidatedResult,
    ConsolidationInput,
    ConsolidationOutput,
)

from greenlang.agents.mrv.baseline_target_tracker import (
    BaselineTargetTrackerAgent,
    TargetType,
    TargetScope,
    TargetStatus,
    BaselineData,
    EmissionsTarget,
    AnnualEmissions,
    TargetProgress,
    BaselineTargetInput,
    BaselineTargetOutput,
)

from greenlang.agents.mrv.inventory_boundary import (
    InventoryBoundaryAgent,
    BoundaryApproach,
    FacilityType,
    ExclusionReason,
    Facility,
    BoundaryScope,
    BoundaryAssessment,
    InventoryBoundaryInput,
    InventoryBoundaryOutput,
)

from greenlang.agents.mrv.temporal_alignment import (
    TemporalAlignmentAgent,
    AlignmentMethod,
    TimeGranularity,
    TimePeriod,
    DataPoint,
    AlignedDataPoint,
    TemporalAlignmentInput,
    TemporalAlignmentOutput,
)

from greenlang.agents.mrv.activity_data_validation import (
    ActivityDataValidationAgent,
    ValidationSeverity,
    ValidationCategory,
    ActivityDataRecord,
    ValidationRule,
    ValidationIssue,
    ValidationResult,
    ActivityDataValidationInput,
    ActivityDataValidationOutput,
)

from greenlang.agents.mrv.emission_factor_selection import (
    EmissionFactorSelectionAgent,
    EFTier,
    EFSource,
    ActivityType,
    EmissionFactor,
    EFSelectionCriteria,
    EFSelectionResult,
    EmissionFactorSelectionInput,
    EmissionFactorSelectionOutput,
)

from greenlang.agents.mrv.gwp_application import (
    GWPApplicationAgent,
    GWPSource,
    GWPHorizon,
    GHGType,
    GHGQuantity,
    GWPConversionResult,
    GWPApplicationInput,
    GWPApplicationOutput,
    GWP_VALUES,
)

from greenlang.agents.mrv.process_emissions import (
    ProcessEmissionsAgent,
    ProcessType,
    ProcessActivity,
    ProcessEmissionResult,
    ProcessEmissionsInput,
    ProcessEmissionsOutput,
    PROCESS_EMISSION_FACTORS,
)

# Additional MRV Agents (GL-MRV-X-016 to GL-MRV-X-030)
from greenlang.agents.mrv.additional_mrv_agents import (
    BusinessTravelAgent,
    EmployeeCommutingAgent,
    WasteEmissionsAgent,
    UpstreamTransportAgent,
    CapitalGoodsAgent,
    FuelEnergyRelatedAgent,
    PurchasedGoodsAgent,
    DownstreamTransportAgent,
    ProductUsePhaseAgent,
    EndOfLifeAgent,
    LeasedAssetsAgent,
    FranchisesAgent,
    InvestmentsAgent,
    BiogenicCarbonAgent,
    RemovalsOffsetsAgent,
    TravelMode,
    CommuteMode,
    WasteType,
    TransportMode,
)

# =============================================================================
# Industrial Sector MRV Agents
# =============================================================================
try:
    from greenlang.agents.mrv.industrial import (
        # Base
        IndustrialMRVBaseAgent,
        IndustrialMRVInput,
        IndustrialMRVOutput,
        # Steel
        SteelProductionMRVAgent,
        SteelMRVInput,
        SteelMRVOutput,
        # Cement
        CementProductionMRVAgent,
        CementMRVInput,
        CementMRVOutput,
        # Chemicals
        ChemicalsProductionMRVAgent,
        ChemicalsMRVInput,
        ChemicalsMRVOutput,
        # Aluminum
        AluminumProductionMRVAgent,
        AluminumMRVInput,
        AluminumMRVOutput,
        # Pulp & Paper
        PulpPaperMRVAgent,
        PulpPaperMRVInput,
        PulpPaperMRVOutput,
        # Glass
        GlassProductionMRVAgent,
        GlassMRVInput,
        GlassMRVOutput,
        # Food Processing
        FoodProcessingMRVAgent,
        FoodProcessingMRVInput,
        FoodProcessingMRVOutput,
        # Additional sectors
        PharmaceuticalMRVAgent,
        ElectronicsMRVAgent,
        AutomotiveMRVAgent,
        TextilesMRVAgent,
        MiningMRVAgent,
        PlasticsMRVAgent,
    )
    _industrial_available = True
except ImportError:
    _industrial_available = False

# =============================================================================
# Exports
# =============================================================================

__all__: List[str] = [
    # Core MRV Agents - Scope 1
    "Scope1CombustionAgent",
    "RefrigerantsFGasAgent",
    "ProcessEmissionsAgent",

    # Core MRV Agents - Scope 2
    "Scope2LocationBasedAgent",
    "Scope2MarketBasedAgent",

    # Core MRV Agents - Scope 3
    "Scope3CategoryMapperAgent",
    "BusinessTravelAgent",
    "EmployeeCommutingAgent",
    "WasteEmissionsAgent",
    "UpstreamTransportAgent",
    "CapitalGoodsAgent",
    "FuelEnergyRelatedAgent",
    "PurchasedGoodsAgent",
    "DownstreamTransportAgent",
    "ProductUsePhaseAgent",
    "EndOfLifeAgent",
    "LeasedAssetsAgent",
    "FranchisesAgent",
    "InvestmentsAgent",

    # Core MRV Agents - Cross-Cutting
    "UncertaintyDataQualityAgent",
    "AuditTrailLineageAgent",
    "ConsolidationRollupAgent",
    "BaselineTargetTrackerAgent",
    "InventoryBoundaryAgent",
    "TemporalAlignmentAgent",
    "ActivityDataValidationAgent",
    "EmissionFactorSelectionAgent",
    "GWPApplicationAgent",
    "BiogenicCarbonAgent",
    "RemovalsOffsetsAgent",

    # Enums - Scope 1
    "CombustionType",
    "FuelType",
    "EmissionGas",
    "RefrigerantType",
    "EquipmentType",
    "CalculationMethod",
    "ProcessType",

    # Enums - Scope 2
    "EnergyType",
    "InstrumentType",
    "EnergySource",

    # Enums - Scope 3
    "Scope3Category",
    "DataSourceType",
    "CalculationApproach",
    "TravelMode",
    "CommuteMode",
    "WasteType",
    "TransportMode",

    # Enums - Cross-Cutting
    "DataQualityIndicator",
    "UncertaintyType",
    "DistributionType",
    "LineageEventType",
    "ConsolidationApproach",
    "EntityType",
    "TargetType",
    "TargetScope",
    "TargetStatus",
    "BoundaryApproach",
    "FacilityType",
    "ExclusionReason",
    "AlignmentMethod",
    "TimeGranularity",
    "ValidationSeverity",
    "ValidationCategory",
    "EFTier",
    "EFSource",
    "ActivityType",
    "GWPSource",
    "GWPHorizon",
    "GHGType",

    # Data Models
    "FuelConsumption",
    "CombustionCalculationResult",
    "RefrigerantInventory",
    "MassBalanceInput",
    "FGasEmissionResult",
    "EnergyConsumption",
    "LocationBasedResult",
    "ContractualInstrument",
    "EnergyPurchase",
    "MarketBasedResult",
    "SpendRecord",
    "PurchaseOrder",
    "BOMItem",
    "CategoryMappingResult",
    "UncertaintyInput",
    "DataQualityAssessment",
    "UncertaintyResult",
    "DataQualityResult",
    "LineageDataSource",
    "LineageNode",
    "LineageEdge",
    "AuditEntry",
    "LineageGraph",
    "EntityEmissions",
    "ConsolidatedResult",
    "BaselineData",
    "EmissionsTarget",
    "AnnualEmissions",
    "TargetProgress",
    "Facility",
    "BoundaryScope",
    "BoundaryAssessment",
    "TimePeriod",
    "DataPoint",
    "AlignedDataPoint",
    "ActivityDataRecord",
    "ValidationRule",
    "ValidationIssue",
    "ValidationResult",
    "EmissionFactor",
    "EFSelectionCriteria",
    "EFSelectionResult",
    "GHGQuantity",
    "GWPConversionResult",
    "ProcessActivity",
    "ProcessEmissionResult",

    # Input/Output Models
    "Scope1CombustionInput",
    "Scope1CombustionOutput",
    "RefrigerantsFGasInput",
    "RefrigerantsFGasOutput",
    "Scope2LocationBasedInput",
    "Scope2LocationBasedOutput",
    "Scope2MarketBasedInput",
    "Scope2MarketBasedOutput",
    "Scope3CategoryMapperInput",
    "Scope3CategoryMapperOutput",
    "UncertaintyDataQualityInput",
    "UncertaintyDataQualityOutput",
    "AuditTrailLineageInput",
    "AuditTrailLineageOutput",
    "ConsolidationInput",
    "ConsolidationOutput",
    "BaselineTargetInput",
    "BaselineTargetOutput",
    "InventoryBoundaryInput",
    "InventoryBoundaryOutput",
    "TemporalAlignmentInput",
    "TemporalAlignmentOutput",
    "ActivityDataValidationInput",
    "ActivityDataValidationOutput",
    "EmissionFactorSelectionInput",
    "EmissionFactorSelectionOutput",
    "GWPApplicationInput",
    "GWPApplicationOutput",
    "ProcessEmissionsInput",
    "ProcessEmissionsOutput",

    # Reference Data
    "STATIONARY_EMISSION_FACTORS",
    "MOBILE_EMISSION_FACTORS",
    "GWP_REFRIGERANTS",
    "GRID_EMISSION_FACTORS",
    "RESIDUAL_MIX_FACTORS",
    "GWP_VALUES",
    "PROCESS_EMISSION_FACTORS",
]

# Add industrial agents if available
if _industrial_available:
    __all__.extend([
        # Base
        "IndustrialMRVBaseAgent",
        "IndustrialMRVInput",
        "IndustrialMRVOutput",
        # Sector agents
        "SteelProductionMRVAgent",
        "SteelMRVInput",
        "SteelMRVOutput",
        "CementProductionMRVAgent",
        "CementMRVInput",
        "CementMRVOutput",
        "ChemicalsProductionMRVAgent",
        "ChemicalsMRVInput",
        "ChemicalsMRVOutput",
        "AluminumProductionMRVAgent",
        "AluminumMRVInput",
        "AluminumMRVOutput",
        "PulpPaperMRVAgent",
        "PulpPaperMRVInput",
        "PulpPaperMRVOutput",
        "GlassProductionMRVAgent",
        "GlassMRVInput",
        "GlassMRVOutput",
        "FoodProcessingMRVAgent",
        "FoodProcessingMRVInput",
        "FoodProcessingMRVOutput",
        "PharmaceuticalMRVAgent",
        "ElectronicsMRVAgent",
        "AutomotiveMRVAgent",
        "TextilesMRVAgent",
        "MiningMRVAgent",
        "PlasticsMRVAgent",
    ])

__version__ = "1.0.0"
