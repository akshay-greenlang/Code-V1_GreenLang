# -*- coding: utf-8 -*-
"""
Tests for RecalculationTriggerEngine (Engine 4).

Covers trigger detection, status management, entity changes, methodology
changes, boundary changes, error corrections, and full detection pipeline.
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

from engines.recalculation_trigger_engine import (
    RecalculationTriggerEngine,
    TriggerDetectionConfig,
    TriggerDetectionResult,
    DetectedTrigger,
    EntityChange,
    MethodologyChange,
    BoundaryChange,
    ErrorCorrection,
    ExternalEvent,
    EntityRegistryEntry,
    InventorySnapshot,
    TriggerType,
    TriggerStatus,
    DetectionMethod,
    Scope,
    CalculationTier,
)


# ============================================================================
# Engine Init
# ============================================================================

class TestRecalculationTriggerEngineInit:
    def test_engine_creation(self, trigger_engine):
        assert trigger_engine is not None

    def test_engine_is_instance(self, trigger_engine):
        assert isinstance(trigger_engine, RecalculationTriggerEngine)

    def test_engine_get_all_triggers_initially_empty(self, trigger_engine):
        triggers = trigger_engine.get_all_triggers()
        assert isinstance(triggers, list)
        assert len(triggers) == 0


# ============================================================================
# Detect Entity Changes (compare registries)
# ============================================================================

class TestDetectEntityChanges:
    def test_detect_acquisition_new_entity(self, trigger_engine):
        """Entity in current but not in previous -> acquisition."""
        current = [
            EntityRegistryEntry(entity_id="ENT-001", entity_name="Original"),
            EntityRegistryEntry(entity_id="ENT-002", entity_name="Acquired Corp",
                                total_emissions_tco2e=Decimal("5000")),
        ]
        previous = [
            EntityRegistryEntry(entity_id="ENT-001", entity_name="Original"),
        ]
        result = trigger_engine.detect_entity_changes(current, previous)
        assert len(result) >= 1
        assert any(t.trigger_type == TriggerType.ACQUISITION for t in result)

    def test_detect_divestiture_removed_entity(self, trigger_engine):
        """Entity in previous but not in current -> divestiture."""
        current = [
            EntityRegistryEntry(entity_id="ENT-001", entity_name="Remaining"),
        ]
        previous = [
            EntityRegistryEntry(entity_id="ENT-001", entity_name="Remaining"),
            EntityRegistryEntry(entity_id="ENT-002", entity_name="Divested Corp",
                                total_emissions_tco2e=Decimal("3000")),
        ]
        result = trigger_engine.detect_entity_changes(current, previous)
        assert len(result) >= 1
        assert any(t.trigger_type == TriggerType.DIVESTITURE for t in result)

    def test_detect_ownership_increase(self, trigger_engine):
        """Same entity with increased ownership -> acquisition trigger."""
        current = [
            EntityRegistryEntry(entity_id="ENT-001", entity_name="Corp A",
                                ownership_pct=Decimal("100"),
                                total_emissions_tco2e=Decimal("10000")),
        ]
        previous = [
            EntityRegistryEntry(entity_id="ENT-001", entity_name="Corp A",
                                ownership_pct=Decimal("50"),
                                total_emissions_tco2e=Decimal("10000")),
        ]
        config = TriggerDetectionConfig(ownership_change_threshold_pct=Decimal("1.0"))
        result = trigger_engine.detect_entity_changes(current, previous, config)
        assert len(result) >= 1

    def test_detect_no_changes(self, trigger_engine):
        """Same registries -> no triggers."""
        registry = [
            EntityRegistryEntry(entity_id="ENT-001", entity_name="Corp A"),
        ]
        result = trigger_engine.detect_entity_changes(registry, registry)
        assert len(result) == 0

    def test_empty_registries(self, trigger_engine):
        result = trigger_engine.detect_entity_changes([], [])
        assert len(result) == 0

    def test_multiple_acquisitions(self, trigger_engine):
        current = [
            EntityRegistryEntry(entity_id="ENT-001", entity_name="A",
                                total_emissions_tco2e=Decimal("1000")),
            EntityRegistryEntry(entity_id="ENT-002", entity_name="B",
                                total_emissions_tco2e=Decimal("2000")),
            EntityRegistryEntry(entity_id="ENT-003", entity_name="C",
                                total_emissions_tco2e=Decimal("3000")),
        ]
        previous = []
        result = trigger_engine.detect_entity_changes(current, previous)
        assert len(result) >= 3


# ============================================================================
# Detect Methodology Changes
# ============================================================================

class TestDetectMethodologyChanges:
    def test_detect_factor_change(self, trigger_engine):
        """Changed emission factor should trigger methodology change."""
        current_factors = {"stationary_combustion": Decimal("2.68")}
        previous_factors = {"stationary_combustion": Decimal("2.50")}
        activity_data = {"stationary_combustion": Decimal("1000")}
        result = trigger_engine.detect_methodology_changes(
            current_factors, previous_factors, activity_data
        )
        assert len(result) >= 1
        assert result[0].trigger_type == TriggerType.METHODOLOGY_CHANGE

    def test_no_factor_change(self, trigger_engine):
        """Same factors -> no triggers."""
        factors = {"stationary_combustion": Decimal("2.68")}
        result = trigger_engine.detect_methodology_changes(factors, factors)
        assert len(result) == 0

    def test_multiple_factor_changes(self, trigger_engine):
        current = {
            "stationary_combustion": Decimal("2.70"),
            "mobile_combustion": Decimal("3.10"),
        }
        previous = {
            "stationary_combustion": Decimal("2.50"),
            "mobile_combustion": Decimal("2.90"),
        }
        result = trigger_engine.detect_methodology_changes(current, previous)
        assert len(result) >= 2

    def test_new_factor_not_detected_as_method_change(self, trigger_engine):
        """New category factor (not in previous) -> handled by boundary detection."""
        current = {"new_category": Decimal("1.0")}
        previous = {}
        result = trigger_engine.detect_methodology_changes(current, previous)
        # Should not create methodology change trigger for new categories
        assert len(result) == 0

    def test_factor_change_with_activity_data(self, trigger_engine):
        current = {"cat": Decimal("3.0")}
        previous = {"cat": Decimal("2.0")}
        activity = {"cat": Decimal("500")}
        result = trigger_engine.detect_methodology_changes(current, previous, activity)
        assert len(result) >= 1
        # impact = |500 * 3.0 - 500 * 2.0| = 500
        assert result[0].emission_impact_tco2e > Decimal("0")


# ============================================================================
# Detect Boundary Changes
# ============================================================================

class TestDetectBoundaryChanges:
    def test_detect_source_added(self, trigger_engine):
        current = ["stationary_combustion", "mobile_combustion", "fugitive"]
        previous = ["stationary_combustion", "mobile_combustion"]
        result = trigger_engine.detect_boundary_changes(current, previous)
        assert len(result) >= 1
        assert result[0].trigger_type == TriggerType.SOURCE_BOUNDARY_CHANGE

    def test_detect_source_removed(self, trigger_engine):
        current = ["stationary_combustion"]
        previous = ["stationary_combustion", "mobile_combustion"]
        result = trigger_engine.detect_boundary_changes(current, previous)
        assert len(result) >= 1

    def test_no_boundary_change(self, trigger_engine):
        boundary = ["stationary_combustion", "mobile_combustion"]
        result = trigger_engine.detect_boundary_changes(boundary, boundary)
        assert len(result) == 0

    def test_boundary_with_emissions_data(self, trigger_engine):
        current = ["cat_a", "cat_b"]
        previous = ["cat_a"]
        current_emissions = {"cat_a": Decimal("5000"), "cat_b": Decimal("2000")}
        previous_emissions = {"cat_a": Decimal("5000")}
        result = trigger_engine.detect_boundary_changes(
            current, previous, current_emissions, previous_emissions
        )
        assert len(result) >= 1
        assert result[0].emission_impact_tco2e >= Decimal("0")


# ============================================================================
# Detect Errors
# ============================================================================

class TestDetectErrors:
    def test_detect_error_correction(self, trigger_engine):
        errors = [
            ErrorCorrection(
                scope=Scope.SCOPE_1,
                category="stationary_combustion",
                original_value_tco2e=Decimal("5000"),
                corrected_value_tco2e=Decimal("3500"),
                error_description="Double-counted fleet fuel",
            )
        ]
        config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("50000"))
        result = trigger_engine.detect_errors(errors, config)
        assert len(result) >= 1
        assert result[0].trigger_type == TriggerType.ERROR_CORRECTION

    def test_detect_minor_error_dismissed(self, trigger_engine):
        """Below de minimis threshold -> auto-dismissed."""
        errors = [
            ErrorCorrection(
                scope=Scope.SCOPE_1,
                category="stationary_combustion",
                original_value_tco2e=Decimal("5000"),
                corrected_value_tco2e=Decimal("4999"),  # ~0.002% impact
                error_description="Trivial rounding",
            )
        ]
        config = TriggerDetectionConfig(
            base_year_total_tco2e=Decimal("100000"),
            de_minimis_threshold_pct=Decimal("0.5"),
            auto_dismiss_below_de_minimis=True,
        )
        result = trigger_engine.detect_errors(errors, config)
        # Should be dismissed as below de minimis
        assert len(result) == 0

    def test_detect_multiple_errors(self, trigger_engine):
        errors = [
            ErrorCorrection(scope=Scope.SCOPE_1,
                            original_value_tco2e=Decimal("5000"),
                            corrected_value_tco2e=Decimal("3500")),
            ErrorCorrection(scope=Scope.SCOPE_2_LOCATION,
                            original_value_tco2e=Decimal("2000"),
                            corrected_value_tco2e=Decimal("1500")),
        ]
        config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("10000"))
        result = trigger_engine.detect_errors(errors, config)
        assert len(result) >= 2


# ============================================================================
# Full Detection Pipeline (detect_triggers)
# ============================================================================

class TestDetectTriggers:
    def test_detect_triggers_with_inventories(self, trigger_engine):
        current = InventorySnapshot(
            year=2024,
            scope1_total_tco2e=Decimal("5000"),
            scope2_location_total_tco2e=Decimal("3000"),
            entity_ids=["FAC-001", "FAC-002", "FAC-003"],
            source_categories=["stationary", "mobile", "fugitive"],
            emission_factors={"stationary": Decimal("2.68")},
        )
        previous = InventorySnapshot(
            year=2023,
            scope1_total_tco2e=Decimal("4500"),
            scope2_location_total_tco2e=Decimal("2800"),
            entity_ids=["FAC-001", "FAC-002"],
            source_categories=["stationary", "mobile"],
            emission_factors={"stationary": Decimal("2.50")},
        )
        config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("50000"))
        result = trigger_engine.detect_triggers(current, previous, [], config)
        assert isinstance(result, TriggerDetectionResult)
        assert result.total_triggers_detected >= 0
        assert result.provenance_hash != ""

    def test_detect_triggers_with_external_events(self, trigger_engine):
        current = InventorySnapshot(year=2024)
        previous = InventorySnapshot(year=2023)
        events = [
            ExternalEvent(
                trigger_type=TriggerType.ACQUISITION,
                description="Acquired new subsidiary",
                estimated_impact_tco2e=Decimal("8000"),
            )
        ]
        config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("50000"))
        result = trigger_engine.detect_triggers(current, previous, events, config)
        assert result.total_triggers_detected >= 1

    def test_detect_triggers_empty_inputs(self, trigger_engine):
        current = InventorySnapshot(year=2024)
        previous = InventorySnapshot(year=2023)
        config = TriggerDetectionConfig()
        result = trigger_engine.detect_triggers(current, previous, [], config)
        assert isinstance(result, TriggerDetectionResult)

    def test_detect_triggers_has_processing_time(self, trigger_engine):
        current = InventorySnapshot(year=2024)
        previous = InventorySnapshot(year=2023)
        result = trigger_engine.detect_triggers(current, previous, [])
        assert result.processing_time_ms >= 0

    def test_detect_triggers_cumulative_impact(self, trigger_engine):
        current = InventorySnapshot(
            year=2024,
            entity_ids=["FAC-001", "FAC-NEW"],
            by_facility={"FAC-001": Decimal("3000"), "FAC-NEW": Decimal("5000")},
        )
        previous = InventorySnapshot(
            year=2023,
            entity_ids=["FAC-001"],
            by_facility={"FAC-001": Decimal("3000")},
        )
        config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("50000"))
        result = trigger_engine.detect_triggers(current, previous, [], config)
        assert result.total_cumulative_impact_tco2e >= Decimal("0")


# ============================================================================
# Assess Trigger
# ============================================================================

class TestAssessTrigger:
    def test_assess_significant_trigger(self, trigger_engine):
        trigger = DetectedTrigger(
            trigger_type=TriggerType.ACQUISITION,
            emission_impact_tco2e=Decimal("6000"),
        )
        assessed = trigger_engine.assess_trigger(trigger, Decimal("100000"))
        assert assessed.significance_pct > Decimal("0")
        assert assessed.requires_recalculation is True

    def test_assess_non_significant_trigger(self, trigger_engine):
        trigger = DetectedTrigger(
            trigger_type=TriggerType.ERROR_CORRECTION,
            emission_impact_tco2e=Decimal("100"),
        )
        assessed = trigger_engine.assess_trigger(trigger, Decimal("100000"))
        assert assessed.requires_recalculation is False

    def test_merger_always_requires_recalculation(self, trigger_engine):
        trigger = DetectedTrigger(
            trigger_type=TriggerType.MERGER,
            emission_impact_tco2e=Decimal("100"),  # Small impact
        )
        assessed = trigger_engine.assess_trigger(trigger, Decimal("100000"))
        assert assessed.requires_recalculation is True

    def test_assess_returns_provenance_hash(self, trigger_engine):
        trigger = DetectedTrigger(
            trigger_type=TriggerType.ACQUISITION,
            emission_impact_tco2e=Decimal("3000"),
        )
        assessed = trigger_engine.assess_trigger(trigger, Decimal("100000"))
        assert assessed.provenance_hash != ""
        assert len(assessed.provenance_hash) == 64


# ============================================================================
# Trigger Status Management
# ============================================================================

class TestTriggerStatus:
    def test_update_trigger_status_not_found(self, trigger_engine):
        result = trigger_engine.update_trigger_status(
            "nonexistent-id", TriggerStatus.CONFIRMED
        )
        assert result is None

    def test_update_trigger_status_after_detection(self, trigger_engine):
        current = InventorySnapshot(
            year=2024,
            entity_ids=["FAC-001", "FAC-NEW"],
            by_facility={"FAC-NEW": Decimal("5000")},
        )
        previous = InventorySnapshot(year=2023, entity_ids=["FAC-001"])
        config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("50000"))
        result = trigger_engine.detect_triggers(current, previous, [], config)
        if result.triggers:
            trigger_id = result.triggers[0].trigger_id
            updated = trigger_engine.update_trigger_status(
                trigger_id, TriggerStatus.CONFIRMED
            )
            assert updated is not None
            assert updated.status == TriggerStatus.CONFIRMED

    def test_get_pending_triggers(self, trigger_engine):
        pending = trigger_engine.get_pending_triggers()
        assert isinstance(pending, list)

    def test_get_confirmed_triggers(self, trigger_engine):
        confirmed = trigger_engine.get_confirmed_triggers()
        assert isinstance(confirmed, list)

    def test_get_all_triggers(self, trigger_engine):
        all_t = trigger_engine.get_all_triggers()
        assert isinstance(all_t, list)


# ============================================================================
# Clear Trigger Store
# ============================================================================

class TestClearTriggerStore:
    def test_clear_trigger_store(self, trigger_engine):
        count = trigger_engine.clear_trigger_store()
        assert count >= 0
        all_triggers = trigger_engine.get_all_triggers()
        assert len(all_triggers) == 0


# ============================================================================
# Enums
# ============================================================================

class TestTriggerEnums:
    def test_trigger_type_values(self):
        assert TriggerType.ACQUISITION is not None
        assert TriggerType.DIVESTITURE is not None
        assert TriggerType.MERGER is not None
        assert TriggerType.METHODOLOGY_CHANGE is not None
        assert TriggerType.ERROR_CORRECTION is not None
        assert TriggerType.SOURCE_BOUNDARY_CHANGE is not None
        assert TriggerType.OUTSOURCING_INSOURCING is not None
        assert len(TriggerType) == 7

    def test_trigger_status_values(self):
        assert TriggerStatus.DETECTED is not None
        assert TriggerStatus.UNDER_REVIEW is not None
        assert TriggerStatus.CONFIRMED is not None
        assert TriggerStatus.DISMISSED is not None
        assert TriggerStatus.PROCESSED is not None
        assert len(TriggerStatus) == 5

    def test_detection_method_values(self):
        assert DetectionMethod.AUTOMATED is not None
        assert DetectionMethod.MANUAL is not None
        assert DetectionMethod.IMPORTED is not None
        assert len(DetectionMethod) == 3

    def test_scope_values(self):
        assert Scope.SCOPE_1 is not None
        assert Scope.SCOPE_2_LOCATION is not None
        assert Scope.SCOPE_2_MARKET is not None
        assert Scope.SCOPE_3 is not None
        assert Scope.ALL is not None
        assert len(Scope) == 5


# ============================================================================
# Config Model
# ============================================================================

class TestTriggerDetectionConfig:
    def test_create_default_config(self):
        config = TriggerDetectionConfig()
        assert config.significance_threshold_pct == Decimal("5.0")
        assert config.de_minimis_threshold_pct == Decimal("0.5")

    def test_config_with_base_year_total(self):
        config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("100000"))
        assert config.base_year_total_tco2e == Decimal("100000")

    def test_config_detection_flags(self):
        config = TriggerDetectionConfig(
            detect_entity_changes=False,
            detect_methodology_changes=True,
            detect_errors=True,
            detect_boundary_changes=False,
        )
        assert config.detect_entity_changes is False
        assert config.detect_methodology_changes is True


# ============================================================================
# Pydantic Models
# ============================================================================

class TestEntityChange:
    def test_create_entity_change(self):
        ec = EntityChange(
            entity_id="ENT-001",
            entity_name="Corp A",
            change_type=TriggerType.ACQUISITION,
            effective_date=date(2024, 1, 1),
            emissions_impact_tco2e=Decimal("5000"),
            ownership_pct_before=Decimal("0"),
            ownership_pct_after=Decimal("100"),
        )
        assert ec.entity_id == "ENT-001"
        assert ec.ownership_delta_pct == Decimal("100")

    def test_ownership_delta_calculation(self):
        ec = EntityChange(
            entity_id="ENT-001",
            entity_name="Corp A",
            change_type=TriggerType.ACQUISITION,
            effective_date=date(2024, 1, 1),
            ownership_pct_before=Decimal("30"),
            ownership_pct_after=Decimal("80"),
        )
        assert ec.ownership_delta_pct == Decimal("50")


class TestErrorCorrection:
    def test_error_impact(self):
        ec = ErrorCorrection(
            scope=Scope.SCOPE_1,
            original_value_tco2e=Decimal("5000"),
            corrected_value_tco2e=Decimal("3500"),
        )
        assert ec.error_impact_tco2e == Decimal("1500")


class TestBoundaryChange:
    def test_net_source_change(self):
        bc = BoundaryChange(
            sources_added=["cat_a", "cat_b"],
            sources_removed=["cat_c"],
        )
        assert bc.net_source_change == 1


class TestInventorySnapshot:
    def test_grand_total(self):
        snap = InventorySnapshot(
            year=2024,
            scope1_total_tco2e=Decimal("5000"),
            scope2_location_total_tco2e=Decimal("3000"),
            scope3_total_tco2e=Decimal("2000"),
        )
        assert snap.grand_total_tco2e == Decimal("10000")


class TestEntityRegistryEntry:
    def test_create_entry(self):
        entry = EntityRegistryEntry(
            entity_id="ENT-001",
            entity_name="Corp A",
            ownership_pct=Decimal("100"),
            total_emissions_tco2e=Decimal("5000"),
        )
        assert entry.entity_id == "ENT-001"
        assert entry.status == "active"


class TestExternalEvent:
    def test_create_external_event(self):
        event = ExternalEvent(
            trigger_type=TriggerType.ACQUISITION,
            description="Acquired subsidiary",
            estimated_impact_tco2e=Decimal("8000"),
        )
        assert event.trigger_type == TriggerType.ACQUISITION
        assert event.detection_method == DetectionMethod.MANUAL
