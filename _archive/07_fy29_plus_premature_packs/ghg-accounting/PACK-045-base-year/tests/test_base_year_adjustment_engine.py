# -*- coding: utf-8 -*-
"""
Tests for BaseYearAdjustmentEngine (Engine 6).

Covers adjustment calculation (acquisition, divestiture, error, methodology),
pro-rata, approval workflow, and package management.
Target: ~60 tests.
"""

import pytest
from decimal import Decimal
from datetime import date
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.base_year_adjustment_engine import (
    BaseYearAdjustmentEngine,
    AdjustmentConfig,
    AdjustmentPackage,
    AdjustmentLine,
    BaseYearInventory,
    TriggerInput,
    TriggerType,
    AdjustmentStatus,
    ProRataMethod,
    Scope,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_inventory():
    """A minimal base year inventory for testing."""
    return BaseYearInventory(
        base_year=2022,
        scope1_tco2e=Decimal("50000"),
        scope2_location_tco2e=Decimal("30000"),
        scope2_market_tco2e=Decimal("28000"),
        scope3_tco2e=Decimal("20000"),
    )


@pytest.fixture
def acquisition_trigger():
    return TriggerInput(
        trigger_id="TRIG-001",
        trigger_type=TriggerType.ACQUISITION,
        scope=Scope.SCOPE_1,
        entity_id="ENT-ACQ",
        entity_emissions_tco2e=Decimal("8000"),
        ownership_pct=Decimal("100"),
        effective_date=date(2024, 7, 1),
        description="Acquired subsidiary",
    )


@pytest.fixture
def divestiture_trigger():
    return TriggerInput(
        trigger_id="TRIG-002",
        trigger_type=TriggerType.DIVESTITURE,
        scope=Scope.SCOPE_1,
        entity_id="ENT-DIV",
        entity_emissions_tco2e=Decimal("6000"),
        ownership_pct=Decimal("100"),
        effective_date=date(2024, 6, 1),
        description="Divested division",
    )


@pytest.fixture
def error_trigger():
    return TriggerInput(
        trigger_id="TRIG-003",
        trigger_type=TriggerType.ERROR_CORRECTION,
        scope=Scope.SCOPE_1,
        original_value_tco2e=Decimal("5000"),
        corrected_value_tco2e=Decimal("3500"),
        description="Double-counted fleet fuel",
    )


@pytest.fixture
def methodology_trigger():
    return TriggerInput(
        trigger_id="TRIG-004",
        trigger_type=TriggerType.METHODOLOGY_CHANGE,
        scope=Scope.SCOPE_1,
        activity_data=Decimal("1000"),
        old_emission_factor=Decimal("2.50"),
        new_emission_factor=Decimal("2.70"),
        description="Updated emission factor",
    )


# ============================================================================
# Engine Init
# ============================================================================

class TestBaseYearAdjustmentEngineInit:
    def test_engine_creation(self, adjustment_engine):
        assert adjustment_engine is not None

    def test_engine_is_instance(self, adjustment_engine):
        assert isinstance(adjustment_engine, BaseYearAdjustmentEngine)


# ============================================================================
# Create Adjustment Package
# ============================================================================

class TestCreateAdjustmentPackage:
    def test_create_package(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        assert isinstance(pkg, AdjustmentPackage)
        assert pkg.status == AdjustmentStatus.DRAFT

    def test_package_has_id(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        assert pkg.package_id != ""

    def test_package_has_adjustment_lines(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        assert len(pkg.adjustment_lines) >= 1

    def test_package_with_config(self, adjustment_engine, sample_inventory, acquisition_trigger):
        config = AdjustmentConfig(pro_rata_method=ProRataMethod.DAILY)
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger], config
        )
        assert isinstance(pkg, AdjustmentPackage)

    def test_package_with_multiple_triggers(self, adjustment_engine, sample_inventory,
                                            acquisition_trigger, divestiture_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger, divestiture_trigger]
        )
        assert len(pkg.adjustment_lines) >= 2

    def test_package_has_provenance_hash(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        assert pkg.provenance_hash != ""
        assert len(pkg.provenance_hash) == 64


# ============================================================================
# Calculate Acquisition Adjustment
# ============================================================================

class TestCalculateAcquisitionAdjustment:
    def test_basic_acquisition(self, adjustment_engine):
        line = adjustment_engine.calculate_acquisition_adjustment(
            entity_emissions_tco2e=Decimal("8000"),
            ownership_pct=Decimal("100"),
            effective_date=date(2024, 7, 1),
        )
        assert isinstance(line, AdjustmentLine)
        assert line.adjustment_tco2e > Decimal("0")

    def test_partial_ownership(self, adjustment_engine):
        line = adjustment_engine.calculate_acquisition_adjustment(
            entity_emissions_tco2e=Decimal("10000"),
            ownership_pct=Decimal("60"),
        )
        assert isinstance(line, AdjustmentLine)
        # Should scale by ownership
        assert line.adjustment_tco2e <= Decimal("10000")

    def test_acquisition_with_pro_rata(self, adjustment_engine):
        line = adjustment_engine.calculate_acquisition_adjustment(
            entity_emissions_tco2e=Decimal("12000"),
            ownership_pct=Decimal("100"),
            effective_date=date(2024, 7, 1),
            pro_rata_method=ProRataMethod.MONTHLY,
            base_year=2024,
        )
        assert isinstance(line, AdjustmentLine)

    def test_acquisition_scope(self, adjustment_engine):
        line = adjustment_engine.calculate_acquisition_adjustment(
            entity_emissions_tco2e=Decimal("5000"),
            ownership_pct=Decimal("100"),
            scope=Scope.SCOPE_2_LOCATION,
        )
        assert line.scope == Scope.SCOPE_2_LOCATION


# ============================================================================
# Calculate Divestiture Adjustment
# ============================================================================

class TestCalculateDivestitureAdjustment:
    def test_basic_divestiture(self, adjustment_engine):
        line = adjustment_engine.calculate_divestiture_adjustment(
            entity_emissions_tco2e=Decimal("6000"),
            ownership_pct=Decimal("100"),
            effective_date=date(2024, 6, 1),
        )
        assert isinstance(line, AdjustmentLine)
        # Divestiture should reduce base year
        assert line.adjustment_tco2e < Decimal("0")

    def test_divestiture_partial_ownership(self, adjustment_engine):
        line = adjustment_engine.calculate_divestiture_adjustment(
            entity_emissions_tco2e=Decimal("10000"),
            ownership_pct=Decimal("40"),
        )
        assert isinstance(line, AdjustmentLine)


# ============================================================================
# Calculate Error Correction
# ============================================================================

class TestCalculateErrorCorrection:
    def test_error_correction(self, adjustment_engine):
        line = adjustment_engine.calculate_error_correction(
            original_tco2e=Decimal("5000"),
            corrected_tco2e=Decimal("3500"),
        )
        assert isinstance(line, AdjustmentLine)

    def test_error_correction_with_description(self, adjustment_engine):
        line = adjustment_engine.calculate_error_correction(
            original_tco2e=Decimal("5000"),
            corrected_tco2e=Decimal("3500"),
            error_description="Double-counted fleet fuel",
        )
        assert isinstance(line, AdjustmentLine)

    def test_error_correction_scope(self, adjustment_engine):
        line = adjustment_engine.calculate_error_correction(
            original_tco2e=Decimal("2000"),
            corrected_tco2e=Decimal("1500"),
            scope=Scope.SCOPE_2_LOCATION,
        )
        assert line.scope == Scope.SCOPE_2_LOCATION


# ============================================================================
# Calculate Methodology Restatement
# ============================================================================

class TestCalculateMethodologyRestatement:
    def test_methodology_restatement(self, adjustment_engine):
        line = adjustment_engine.calculate_methodology_restatement(
            activity_data=Decimal("1000"),
            old_factor=Decimal("2.50"),
            new_factor=Decimal("2.70"),
        )
        assert isinstance(line, AdjustmentLine)

    def test_restatement_with_scope(self, adjustment_engine):
        line = adjustment_engine.calculate_methodology_restatement(
            activity_data=Decimal("500"),
            old_factor=Decimal("1.80"),
            new_factor=Decimal("2.00"),
            scope=Scope.SCOPE_1,
            category="stationary_combustion",
        )
        assert isinstance(line, AdjustmentLine)


# ============================================================================
# Apply Adjustments
# ============================================================================

class TestApplyAdjustments:
    def test_apply_adjustments(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        # Must approve before applying
        pkg = adjustment_engine.submit_for_approval(pkg)
        pkg = adjustment_engine.approve_adjustment(pkg, approver="admin@test.com")
        result = adjustment_engine.apply_adjustments(sample_inventory, pkg)
        assert isinstance(result, BaseYearInventory)


# ============================================================================
# Approval Workflow
# ============================================================================

class TestApprovalWorkflow:
    def test_submit_for_approval(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        submitted = adjustment_engine.submit_for_approval(pkg)
        assert submitted.status == AdjustmentStatus.PENDING_APPROVAL

    def test_approve_adjustment(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        pkg = adjustment_engine.submit_for_approval(pkg)
        approved = adjustment_engine.approve_adjustment(pkg, approver="admin@test.com")
        assert approved.status == AdjustmentStatus.APPROVED
        assert approved.approved_by == "admin@test.com"

    def test_reject_adjustment(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        pkg = adjustment_engine.submit_for_approval(pkg)
        rejected = adjustment_engine.reject_adjustment(
            pkg, rejector="reviewer@test.com", reason="Insufficient evidence"
        )
        assert rejected.status == AdjustmentStatus.REJECTED


# ============================================================================
# Package Management
# ============================================================================

class TestPackageManagement:
    def test_list_packages(self, adjustment_engine):
        packages = adjustment_engine.list_packages()
        assert isinstance(packages, list)

    def test_get_package(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        retrieved = adjustment_engine.get_package(pkg.package_id)
        assert retrieved is not None

    def test_get_nonexistent_package(self, adjustment_engine):
        result = adjustment_engine.get_package("nonexistent-id")
        assert result is None

    def test_list_packages_with_filter(self, adjustment_engine, sample_inventory, acquisition_trigger):
        pkg = adjustment_engine.create_adjustment_package(
            sample_inventory, [acquisition_trigger]
        )
        drafts = adjustment_engine.list_packages(status_filter=AdjustmentStatus.DRAFT)
        assert isinstance(drafts, list)


# ============================================================================
# Enums
# ============================================================================

class TestAdjustmentEnums:
    def test_adjustment_status(self):
        assert AdjustmentStatus.DRAFT is not None
        assert AdjustmentStatus.PENDING_APPROVAL is not None
        assert AdjustmentStatus.APPROVED is not None
        assert AdjustmentStatus.APPLIED is not None
        assert AdjustmentStatus.REJECTED is not None
        assert len(AdjustmentStatus) == 5

    def test_pro_rata_method(self):
        assert ProRataMethod.MONTHLY is not None
        assert ProRataMethod.DAILY is not None
        assert ProRataMethod.QUARTERLY is not None
        assert len(ProRataMethod) == 3

    def test_scope(self):
        assert Scope.SCOPE_1 is not None
        assert Scope.SCOPE_2_LOCATION is not None
        assert Scope.SCOPE_2_MARKET is not None
        assert Scope.SCOPE_3 is not None
        assert len(Scope) == 4

    def test_trigger_type(self):
        assert TriggerType.ACQUISITION is not None
        assert TriggerType.DIVESTITURE is not None
        assert len(TriggerType) >= 4


# ============================================================================
# Model Tests
# ============================================================================

class TestAdjustmentConfig:
    def test_create_config_defaults(self):
        config = AdjustmentConfig()
        assert config.pro_rata_method == ProRataMethod.MONTHLY

    def test_config_with_pro_rata(self):
        config = AdjustmentConfig(pro_rata_method=ProRataMethod.DAILY)
        assert config.pro_rata_method == ProRataMethod.DAILY


class TestTriggerInput:
    def test_create_acquisition_trigger(self):
        t = TriggerInput(
            trigger_id="T-001",
            trigger_type=TriggerType.ACQUISITION,
            entity_emissions_tco2e=Decimal("5000"),
            ownership_pct=Decimal("100"),
        )
        assert t.trigger_type == TriggerType.ACQUISITION

    def test_create_error_trigger(self):
        t = TriggerInput(
            trigger_id="T-002",
            trigger_type=TriggerType.ERROR_CORRECTION,
            original_value_tco2e=Decimal("5000"),
            corrected_value_tco2e=Decimal("3500"),
        )
        assert t.trigger_type == TriggerType.ERROR_CORRECTION


class TestBaseYearInventory:
    def test_create_inventory(self):
        inv = BaseYearInventory(
            base_year=2022,
            scope1_tco2e=Decimal("50000"),
            scope2_location_tco2e=Decimal("30000"),
        )
        assert inv.base_year == 2022
        assert inv.scope1_tco2e == Decimal("50000")
