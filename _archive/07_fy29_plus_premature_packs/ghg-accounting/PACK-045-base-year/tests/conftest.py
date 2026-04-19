# -*- coding: utf-8 -*-
"""
PACK-045 Base Year Management Pack - Shared Test Fixtures

Provides reusable fixtures for all test modules including sample emissions
data, candidate years, organization IDs, engine instances, and configuration
objects.
"""

import sys
import os
from decimal import Decimal
from pathlib import Path

import pytest

# Ensure the pack root is on sys.path so engines/ etc. are importable
PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

# ---------------------------------------------------------------------------
# Engine imports
# ---------------------------------------------------------------------------
from engines.base_year_selection_engine import (
    BaseYearSelectionEngine,
    CandidateYear,
    SelectionConfig,
    SelectionWeights,
    SectorType as SelectionSectorType,
)
from engines.base_year_inventory_engine import (
    BaseYearInventoryEngine,
    SourceEmission,
    InventoryConfig,
    SourceCategory,
    GasType,
    GWPVersion as InvGWPVersion,
    ScopeType as InvScopeType,
    ConsolidationApproach as InvConsolidationApproach,
)
from engines.recalculation_policy_engine import RecalculationPolicyEngine
from engines.recalculation_trigger_engine import RecalculationTriggerEngine
from engines.significance_assessment_engine import SignificanceAssessmentEngine
from engines.base_year_adjustment_engine import BaseYearAdjustmentEngine
from engines.time_series_consistency_engine import TimeSeriesConsistencyEngine
from engines.target_tracking_engine import TargetTrackingEngine
from engines.base_year_audit_engine import BaseYearAuditEngine
from engines.base_year_reporting_engine import BaseYearReportingEngine

from config.pack_config import (
    PackConfig,
    BaseYearManagementConfig,
    BaseYearSelectionConfig,
    RecalculationPolicyConfig,
    TriggerConfig,
    SignificanceConfig,
    AdjustmentConfig,
    TimeSeriesConfig,
    TargetTrackingConfig,
    AuditConfig,
    ReportingConfig,
    GWPConfig,
    ScopeConfig,
    NotificationConfig,
    PerformanceConfig,
    SecurityConfig,
    IntegrationConfig,
    BaseYearType,
    RecalculationTriggerType,
    SignificanceMethod,
    AdjustmentApproach,
    ConsolidationApproach,
    TargetType,
    SBTiAmbitionLevel,
    AuditLevel,
    ReportingFramework,
    GWPVersion,
    ScopeType,
    OutputFormat,
    NotificationChannel,
    SectorType,
)

# ---------------------------------------------------------------------------
# Organisation / identity fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def organization_id():
    """Sample organisation identifier."""
    return "ORG-TEST-001"


@pytest.fixture
def base_year():
    """Standard test base year."""
    return 2022


# ---------------------------------------------------------------------------
# Selection engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def selection_engine():
    """BaseYearSelectionEngine instance."""
    return BaseYearSelectionEngine()


@pytest.fixture
def candidate_years():
    """List of 5 candidate years with varying quality."""
    return [
        CandidateYear(
            year=2019,
            scope1_tco2e=Decimal("5000"),
            scope2_tco2e=Decimal("3000"),
            scope3_tco2e=Decimal("2000"),
            total_tco2e=Decimal("10000"),
            data_quality_score=Decimal("85"),
            completeness_pct=Decimal("92"),
            methodology_tier=3,
            is_verified=True,
            boundary_changes_count=0,
        ),
        CandidateYear(
            year=2020,
            scope1_tco2e=Decimal("4800"),
            scope2_tco2e=Decimal("2800"),
            scope3_tco2e=Decimal("1800"),
            total_tco2e=Decimal("9400"),
            data_quality_score=Decimal("78"),
            completeness_pct=Decimal("88"),
            methodology_tier=2,
            is_verified=False,
            boundary_changes_count=1,
        ),
        CandidateYear(
            year=2021,
            scope1_tco2e=Decimal("5200"),
            scope2_tco2e=Decimal("3200"),
            scope3_tco2e=Decimal("2200"),
            total_tco2e=Decimal("10600"),
            data_quality_score=Decimal("90"),
            completeness_pct=Decimal("95"),
            methodology_tier=3,
            is_verified=True,
            boundary_changes_count=0,
        ),
        CandidateYear(
            year=2022,
            scope1_tco2e=Decimal("5100"),
            scope2_tco2e=Decimal("3100"),
            scope3_tco2e=Decimal("2100"),
            total_tco2e=Decimal("10300"),
            data_quality_score=Decimal("92"),
            completeness_pct=Decimal("97"),
            methodology_tier=4,
            is_verified=True,
            boundary_changes_count=0,
        ),
        CandidateYear(
            year=2023,
            scope1_tco2e=Decimal("4900"),
            scope2_tco2e=Decimal("2900"),
            scope3_tco2e=Decimal("1900"),
            total_tco2e=Decimal("9700"),
            data_quality_score=Decimal("88"),
            completeness_pct=Decimal("93"),
            methodology_tier=3,
            is_verified=True,
            boundary_changes_count=0,
        ),
    ]


@pytest.fixture
def default_selection_config():
    """Default SelectionConfig."""
    return SelectionConfig()


# ---------------------------------------------------------------------------
# Inventory engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inventory_engine():
    """BaseYearInventoryEngine instance."""
    return BaseYearInventoryEngine()


@pytest.fixture
def sample_sources():
    """List of SourceEmission objects covering multiple scopes."""
    return [
        SourceEmission(
            category=SourceCategory.STATIONARY_COMBUSTION,
            facility_id="FAC-001",
            tco2e=Decimal("3000"),
            gas_type=GasType.CO2,
            data_quality_score=Decimal("80"),
        ),
        SourceEmission(
            category=SourceCategory.MOBILE_COMBUSTION,
            facility_id="FAC-001",
            tco2e=Decimal("1500"),
            gas_type=GasType.CO2,
            data_quality_score=Decimal("75"),
        ),
        SourceEmission(
            category=SourceCategory.FUGITIVE,
            facility_id="FAC-002",
            tco2e=Decimal("500"),
            gas_type=GasType.CH4,
            data_quality_score=Decimal("60"),
        ),
        SourceEmission(
            category=SourceCategory.ELECTRICITY_LOCATION,
            facility_id="FAC-001",
            tco2e=Decimal("2000"),
            gas_type=GasType.CO2,
            data_quality_score=Decimal("85"),
        ),
        SourceEmission(
            category=SourceCategory.ELECTRICITY_MARKET,
            facility_id="FAC-001",
            tco2e=Decimal("1800"),
            gas_type=GasType.CO2,
            data_quality_score=Decimal("85"),
        ),
        SourceEmission(
            category=SourceCategory.SCOPE3_CAT1,
            facility_id="FAC-001",
            tco2e=Decimal("4000"),
            gas_type=GasType.CO2,
            data_quality_score=Decimal("50"),
        ),
        SourceEmission(
            category=SourceCategory.SCOPE3_CAT6,
            facility_id="FAC-001",
            tco2e=Decimal("300"),
            gas_type=GasType.CO2,
            data_quality_score=Decimal("70"),
        ),
    ]


@pytest.fixture
def inventory_config(organization_id, base_year):
    """Standard InventoryConfig."""
    return InventoryConfig(
        organization_id=organization_id,
        base_year=base_year,
        gwp_version=InvGWPVersion.AR5,
        consolidation_approach=InvConsolidationApproach.OPERATIONAL_CONTROL,
    )


@pytest.fixture
def established_inventory(inventory_engine, sample_sources, inventory_config):
    """A fully established BaseYearInventory."""
    return inventory_engine.establish_inventory(sample_sources, inventory_config)


# ---------------------------------------------------------------------------
# Other engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def policy_engine():
    return RecalculationPolicyEngine()


@pytest.fixture
def trigger_engine():
    return RecalculationTriggerEngine()


@pytest.fixture
def significance_engine():
    return SignificanceAssessmentEngine()


@pytest.fixture
def adjustment_engine():
    return BaseYearAdjustmentEngine()


@pytest.fixture
def time_series_engine():
    return TimeSeriesConsistencyEngine()


@pytest.fixture
def target_engine():
    return TargetTrackingEngine()


@pytest.fixture
def audit_engine():
    return BaseYearAuditEngine()


@pytest.fixture
def reporting_engine():
    return BaseYearReportingEngine()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_pack_config():
    """Default PackConfig."""
    return PackConfig()


@pytest.fixture
def default_mgmt_config():
    """Default BaseYearManagementConfig."""
    return BaseYearManagementConfig()


@pytest.fixture
def manufacturing_config():
    """Manufacturing sector config."""
    return BaseYearManagementConfig(
        company_name="Test Manufacturing Co",
        sector_type=SectorType.MANUFACTURING,
    )
