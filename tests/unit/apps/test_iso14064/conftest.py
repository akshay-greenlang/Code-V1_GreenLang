# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for GL-ISO14064-APP v1.0 test suite.

Provides reusable fixtures for configuration, organizations, entities,
inventories, emission sources, removal sources, and service engine instances
used across all 14 test modules.

Author: GL-TestEngineer
Date: March 2026
"""

import sys
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List

import pytest

# Ensure the services package is importable
_SERVICES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "applications", "GL-ISO14064-APP", "ISO14064-Compliance-Platform",
)
_SERVICES_DIR = os.path.normpath(_SERVICES_DIR)
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

from services.config import (
    ActionCategory,
    ActionStatus,
    ConsolidationApproach,
    DataQualityTier,
    FindingSeverity,
    FindingStatus,
    GHGGas,
    GWPSource,
    GWP_AR5,
    GWP_AR6,
    InventoryStatus,
    ISOCategory,
    ISO14064AppConfig,
    PermanenceLevel,
    QuantificationMethod,
    RemovalType,
    ReportFormat,
    ReportingPeriod,
    SignificanceLevel,
    VerificationLevel,
    VerificationStage,
)
from services.models import (
    CategoryResult,
    EmissionSource,
    Entity,
    GHGGasBreakdown,
    ISOInventory,
    Organization,
    RemovalSource,
    _new_id,
    _now,
    _sha256,
)
from services.boundary_manager import BoundaryManager
from services.quantification_engine import QuantificationEngine
from services.removals_tracker import RemovalsTracker
from services.significance_engine import SignificanceEngine
from services.uncertainty_engine import UncertaintyEngine
from services.quality_management import QualityManager
from services.base_year_manager import BaseYearManager
from services.report_generator import ReportGenerator
from services.management_plan import ManagementPlanEngine
from services.verification_workflow import VerificationWorkflow
from services.crosswalk_engine import CrosswalkEngine


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def default_config() -> ISO14064AppConfig:
    """Default application configuration."""
    return ISO14064AppConfig()


@pytest.fixture
def custom_config() -> ISO14064AppConfig:
    """Custom configuration with adjusted thresholds for testing."""
    return ISO14064AppConfig(
        significance_threshold_percent=Decimal("2.0"),
        recalculation_threshold_percent=Decimal("3.0"),
        monte_carlo_iterations=1000,
        confidence_levels=[90, 95],
        reporting_year=2025,
    )


# ============================================================================
# BOUNDARY MANAGER FIXTURES
# ============================================================================

@pytest.fixture
def boundary_manager(default_config) -> BoundaryManager:
    """Fresh BoundaryManager instance."""
    return BoundaryManager(default_config)


@pytest.fixture
def sample_org(boundary_manager) -> Organization:
    """Create a sample organization."""
    return boundary_manager.create_organization(
        name="Acme Corp",
        industry="manufacturing",
        country="US",
        description="Test manufacturing company",
    )


@pytest.fixture
def sample_entity(boundary_manager, sample_org) -> Entity:
    """Create a sample entity under the sample org."""
    return boundary_manager.add_entity(
        org_id=sample_org.id,
        name="Main Factory",
        entity_type="facility",
        country="US",
        ownership_pct=Decimal("100.0"),
        employees=500,
        revenue=Decimal("50000000"),
    )


@pytest.fixture
def sample_inventory(boundary_manager, sample_org) -> ISOInventory:
    """Create a sample inventory for 2025."""
    return boundary_manager.create_inventory(
        org_id=sample_org.id,
        reporting_year=2025,
    )


# ============================================================================
# QUANTIFICATION ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def quant_engine(default_config) -> QuantificationEngine:
    """Fresh QuantificationEngine instance."""
    return QuantificationEngine(default_config)


# ============================================================================
# REMOVALS TRACKER FIXTURES
# ============================================================================

@pytest.fixture
def removals_tracker(default_config) -> RemovalsTracker:
    """Fresh RemovalsTracker instance."""
    return RemovalsTracker(default_config)


# ============================================================================
# SIGNIFICANCE ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def significance_engine(default_config) -> SignificanceEngine:
    """Fresh SignificanceEngine instance."""
    return SignificanceEngine(default_config)


# ============================================================================
# UNCERTAINTY ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def uncertainty_engine(default_config) -> UncertaintyEngine:
    """Fresh UncertaintyEngine instance."""
    return UncertaintyEngine(default_config)


# ============================================================================
# QUALITY MANAGER FIXTURES
# ============================================================================

@pytest.fixture
def quality_manager(default_config) -> QualityManager:
    """Fresh QualityManager instance."""
    return QualityManager(default_config)


# ============================================================================
# BASE YEAR MANAGER FIXTURES
# ============================================================================

@pytest.fixture
def base_year_manager(default_config) -> BaseYearManager:
    """Fresh BaseYearManager instance."""
    return BaseYearManager(default_config)


# ============================================================================
# REPORT GENERATOR FIXTURES
# ============================================================================

@pytest.fixture
def report_generator(default_config) -> ReportGenerator:
    """Fresh ReportGenerator with empty stores."""
    return ReportGenerator(config=default_config)


# ============================================================================
# MANAGEMENT PLAN FIXTURES
# ============================================================================

@pytest.fixture
def management_engine(default_config) -> ManagementPlanEngine:
    """Fresh ManagementPlanEngine instance."""
    return ManagementPlanEngine(default_config)


# ============================================================================
# VERIFICATION WORKFLOW FIXTURES
# ============================================================================

@pytest.fixture
def verification_workflow(default_config) -> VerificationWorkflow:
    """Fresh VerificationWorkflow instance."""
    return VerificationWorkflow(default_config)


# ============================================================================
# CROSSWALK ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def crosswalk_engine(default_config) -> CrosswalkEngine:
    """Fresh CrosswalkEngine instance."""
    return CrosswalkEngine(default_config)


# ============================================================================
# CATEGORY RESULT FIXTURES
# ============================================================================

@pytest.fixture
def sample_category_results() -> Dict[str, CategoryResult]:
    """Sample category results for testing aggregation and crosswalk."""
    return {
        ISOCategory.CATEGORY_1_DIRECT.value: CategoryResult(
            iso_category=ISOCategory.CATEGORY_1_DIRECT,
            category_name="Category 1 - Direct",
            total_tco2e=Decimal("5000"),
            data_quality_tier=DataQualityTier.TIER_3,
        ),
        ISOCategory.CATEGORY_2_ENERGY.value: CategoryResult(
            iso_category=ISOCategory.CATEGORY_2_ENERGY,
            category_name="Category 2 - Energy",
            total_tco2e=Decimal("3000"),
            data_quality_tier=DataQualityTier.TIER_2,
        ),
        ISOCategory.CATEGORY_3_TRANSPORT.value: CategoryResult(
            iso_category=ISOCategory.CATEGORY_3_TRANSPORT,
            category_name="Category 3 - Transport",
            total_tco2e=Decimal("1500"),
            data_quality_tier=DataQualityTier.TIER_2,
        ),
        ISOCategory.CATEGORY_4_PRODUCTS_USED.value: CategoryResult(
            iso_category=ISOCategory.CATEGORY_4_PRODUCTS_USED,
            category_name="Category 4 - Products Used",
            total_tco2e=Decimal("2000"),
            data_quality_tier=DataQualityTier.TIER_1,
        ),
        ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG.value: CategoryResult(
            iso_category=ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
            category_name="Category 5 - Products from Org",
            total_tco2e=Decimal("800"),
            data_quality_tier=DataQualityTier.TIER_1,
        ),
        ISOCategory.CATEGORY_6_OTHER.value: CategoryResult(
            iso_category=ISOCategory.CATEGORY_6_OTHER,
            category_name="Category 6 - Other",
            total_tco2e=Decimal("200"),
            data_quality_tier=DataQualityTier.TIER_1,
        ),
    }
