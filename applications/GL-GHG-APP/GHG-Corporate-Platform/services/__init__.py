"""
GL-GHG-APP Corporate Platform -- Services Package

GHG Protocol Corporate Accounting and Reporting Standard implementation.
Integrates 28 MRV agents across Scope 1 (8 categories), Scope 2 (5 agents),
and Scope 3 (15 categories).

Public API:
    GHGPlatform        -- Unified facade composing all engines.
    GHGAppConfig       -- Application configuration.
    InventoryManager   -- Organization and inventory management.
    BaseYearManager    -- Base year and recalculation (Ch 6).
    ScopeAggregator    -- Scope 1/2/3 aggregation from 28 MRV agents.
    IntensityCalculator -- GHG intensity metrics (Ch 12).
    UncertaintyEngine  -- Monte Carlo uncertainty propagation (Ch 11).
    CompletenessChecker -- Mandatory disclosure checking.
    ReportGenerator    -- Multi-format report generation.
    VerificationWorkflow -- Internal review and external assurance.
    TargetTracker      -- Reduction targets and SBTi alignment.

Example:
    >>> from services import GHGPlatform
    >>> platform = GHGPlatform()
    >>> print(platform.health_check())
"""

from .config import (
    ConsolidationApproach,
    DataQualityTier,
    EntityType,
    FindingSeverity,
    FindingType,
    GHGAppConfig,
    GHGGas,
    GWP_AR5,
    IntensityDenominator,
    ReportFormat,
    ReportingPeriod,
    Scope,
    Scope1Category,
    Scope3Category,
    SECTOR_BENCHMARKS,
    TargetType,
    UNCERTAINTY_CV_BY_TIER,
    VerificationLevel,
    VerificationStatus,
)
from .models import (
    BaseYear,
    CompletenessResult,
    DashboardMetrics,
    DataGap,
    Disclosure,
    Entity,
    ExclusionRecord,
    GHGInventory,
    IntensityMetric,
    InventoryBoundary,
    Organization,
    Recalculation,
    Report,
    ReportSection,
    ScopeEmissions,
    ScopeUncertainty,
    Target,
    UncertaintyResult,
    VerificationFinding,
    VerificationRecord,
)
from .inventory_manager import InventoryManager
from .base_year_manager import BaseYearManager
from .scope_aggregator import ScopeAggregator
from .intensity_calculator import IntensityCalculator
from .uncertainty_engine import UncertaintyEngine
from .completeness_checker import CompletenessChecker
from .report_generator import ReportGenerator
from .verification_workflow import VerificationWorkflow
from .target_tracker import TargetTracker
from .setup import GHGPlatform

__all__ = [
    # Facade
    "GHGPlatform",
    # Engines
    "InventoryManager",
    "BaseYearManager",
    "ScopeAggregator",
    "IntensityCalculator",
    "UncertaintyEngine",
    "CompletenessChecker",
    "ReportGenerator",
    "VerificationWorkflow",
    "TargetTracker",
    # Config & Enums
    "GHGAppConfig",
    "ConsolidationApproach",
    "DataQualityTier",
    "EntityType",
    "FindingSeverity",
    "FindingType",
    "GHGGas",
    "GWP_AR5",
    "IntensityDenominator",
    "ReportFormat",
    "ReportingPeriod",
    "Scope",
    "Scope1Category",
    "Scope3Category",
    "SECTOR_BENCHMARKS",
    "TargetType",
    "UNCERTAINTY_CV_BY_TIER",
    "VerificationLevel",
    "VerificationStatus",
    # Domain Models
    "BaseYear",
    "CompletenessResult",
    "DashboardMetrics",
    "DataGap",
    "Disclosure",
    "Entity",
    "ExclusionRecord",
    "GHGInventory",
    "IntensityMetric",
    "InventoryBoundary",
    "Organization",
    "Recalculation",
    "Report",
    "ReportSection",
    "ScopeEmissions",
    "ScopeUncertainty",
    "Target",
    "UncertaintyResult",
    "VerificationFinding",
    "VerificationRecord",
]

__version__ = "1.0.0"
