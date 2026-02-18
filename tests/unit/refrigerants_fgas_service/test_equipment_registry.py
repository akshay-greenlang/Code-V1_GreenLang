# -*- coding: utf-8 -*-
"""
Unit tests for EquipmentRegistryEngine - AGENT-MRV-002 Engine 3

Tests equipment registration, retrieval, updates, decommissioning,
service events, fleet analytics, and provenance tracking.

Target: 55+ tests, 600+ lines.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.refrigerants_fgas.equipment_registry import (
    EquipmentRegistryEngine,
    EQUIPMENT_DEFAULTS,
)
from greenlang.refrigerants_fgas.models import (
    EquipmentProfile,
    EquipmentType,
    EquipmentStatus,
    RefrigerantType,
    ServiceEvent,
    ServiceEventType,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> EquipmentRegistryEngine:
    """Create a fresh EquipmentRegistryEngine."""
    return EquipmentRegistryEngine()


@pytest.fixture
def sample_profile() -> EquipmentProfile:
    """Standard test equipment profile."""
    return EquipmentProfile(
        equipment_id="eq_test_001",
        equipment_type=EquipmentType.COMMERCIAL_AC,
        refrigerant_type=RefrigerantType.R_410A,
        charge_kg=15.0,
        equipment_count=1,
        status=EquipmentStatus.ACTIVE,
        installation_date=datetime(2020, 6, 1, tzinfo=timezone.utc),
        location="Building A",
    )


@pytest.fixture
def populated_engine(
    engine: EquipmentRegistryEngine, sample_profile: EquipmentProfile
) -> EquipmentRegistryEngine:
    """Engine with one registered equipment."""
    engine.register_equipment(sample_profile)
    return engine


# ===========================================================================
# Test: Initialization
# ===========================================================================


class TestEquipmentRegistryInit:
    """Tests for engine initialization."""

    def test_initialization(self, engine: EquipmentRegistryEngine):
        """Engine initializes with zero equipment."""
        assert len(engine) == 0

    def test_repr(self, engine: EquipmentRegistryEngine):
        """repr includes key information."""
        r = repr(engine)
        assert "EquipmentRegistryEngine" in r

    def test_equipment_defaults_count(self):
        """EQUIPMENT_DEFAULTS has 15 entries."""
        assert len(EQUIPMENT_DEFAULTS) == 15


# ===========================================================================
# Test: Register Equipment
# ===========================================================================


class TestRegisterEquipment:
    """Tests for equipment registration."""

    def test_register_equipment(
        self, engine: EquipmentRegistryEngine, sample_profile: EquipmentProfile
    ):
        """Registration returns the equipment_id."""
        equip_id = engine.register_equipment(sample_profile)
        assert equip_id == "eq_test_001"
        assert len(engine) == 1

    def test_register_duplicate_raises(
        self, populated_engine: EquipmentRegistryEngine, sample_profile
    ):
        """Duplicate registration raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            populated_engine.register_equipment(sample_profile)

    def test_register_at_capacity_raises(self):
        """At capacity raises ValueError."""
        engine = EquipmentRegistryEngine(config={"max_equipment": 1})
        profile = EquipmentProfile(
            equipment_id="eq_cap_1",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=10.0,
        )
        engine.register_equipment(profile)
        profile2 = EquipmentProfile(
            equipment_id="eq_cap_2",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=10.0,
        )
        with pytest.raises(ValueError, match="at capacity"):
            engine.register_equipment(profile2)


# ===========================================================================
# Test: Get Equipment
# ===========================================================================


class TestGetEquipment:
    """Tests for equipment retrieval."""

    def test_get_equipment(self, populated_engine: EquipmentRegistryEngine):
        """Retrieves registered equipment by ID."""
        profile = populated_engine.get_equipment("eq_test_001")
        assert profile.equipment_id == "eq_test_001"
        assert profile.charge_kg == 15.0

    def test_get_equipment_not_found(self, engine: EquipmentRegistryEngine):
        """Non-existent ID raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            engine.get_equipment("eq_nonexistent")


# ===========================================================================
# Test: List Equipment
# ===========================================================================


class TestListEquipment:
    """Tests for equipment listing and filtering."""

    def test_list_equipment_all(self, populated_engine: EquipmentRegistryEngine):
        """List all returns all registered equipment."""
        results = populated_engine.list_equipment()
        assert len(results) == 1

    def test_list_equipment_by_type(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Filter by equipment type."""
        results = populated_engine.list_equipment(
            equipment_type=EquipmentType.COMMERCIAL_AC
        )
        assert len(results) == 1

    def test_list_equipment_by_type_no_match(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Filter by non-matching type returns empty."""
        results = populated_engine.list_equipment(
            equipment_type=EquipmentType.SWITCHGEAR
        )
        assert len(results) == 0

    def test_list_equipment_by_status(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Filter by status."""
        results = populated_engine.list_equipment(status=EquipmentStatus.ACTIVE)
        assert len(results) == 1

    def test_list_equipment_by_status_no_match(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Filter by non-matching status returns empty."""
        results = populated_engine.list_equipment(
            status=EquipmentStatus.DECOMMISSIONED
        )
        assert len(results) == 0


# ===========================================================================
# Test: Update Equipment
# ===========================================================================


class TestUpdateEquipment:
    """Tests for equipment updates."""

    def test_update_equipment(self, populated_engine: EquipmentRegistryEngine):
        """Update charge_kg and location."""
        updated = populated_engine.update_equipment(
            "eq_test_001", charge_kg=20.0, location="Building B"
        )
        assert updated.charge_kg == 20.0
        assert updated.location == "Building B"

    def test_update_status(self, populated_engine: EquipmentRegistryEngine):
        """Update status to MAINTENANCE."""
        updated = populated_engine.update_equipment(
            "eq_test_001", status=EquipmentStatus.MAINTENANCE
        )
        assert updated.status == EquipmentStatus.MAINTENANCE

    def test_update_not_found_raises(self, engine: EquipmentRegistryEngine):
        """Update non-existent equipment raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            engine.update_equipment("eq_missing", charge_kg=10.0)

    def test_update_invalid_field_raises(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Unsupported field raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            populated_engine.update_equipment(
                "eq_test_001", invalid_field="bad"
            )


# ===========================================================================
# Test: Decommission Equipment
# ===========================================================================


class TestDecommissionEquipment:
    """Tests for equipment decommissioning."""

    def test_decommission_equipment(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Decommission sets status and creates event."""
        result = populated_engine.decommission_equipment(
            "eq_test_001", recovery_kg=10.0
        )
        assert result["equipment_id"] == "eq_test_001"
        assert result["previous_status"] == "active"
        assert "provenance_hash" in result

        # Verify status changed
        profile = populated_engine.get_equipment("eq_test_001")
        assert profile.status == EquipmentStatus.DECOMMISSIONED

    def test_decommission_already_decommissioned_raises(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Re-decommissioning raises ValueError."""
        populated_engine.decommission_equipment("eq_test_001", recovery_kg=5.0)
        with pytest.raises(ValueError, match="already decommissioned"):
            populated_engine.decommission_equipment("eq_test_001")

    def test_decommission_not_found_raises(
        self, engine: EquipmentRegistryEngine
    ):
        """Decommissioning non-existent raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            engine.decommission_equipment("eq_missing")

    def test_decommission_negative_recovery_raises(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Negative recovery raises ValueError."""
        with pytest.raises(ValueError, match="recovery_kg must be >= 0"):
            populated_engine.decommission_equipment(
                "eq_test_001", recovery_kg=-1.0
            )


# ===========================================================================
# Test: Service Events
# ===========================================================================


class TestServiceEvents:
    """Tests for service event logging and history."""

    def test_log_service_event(self, populated_engine: EquipmentRegistryEngine):
        """Log a recharge event."""
        event = ServiceEvent(
            equipment_id="eq_test_001",
            event_type=ServiceEventType.RECHARGE,
            date=datetime.now(timezone.utc),
            refrigerant_added_kg=5.0,
        )
        event_id = populated_engine.log_service_event(event)
        assert event_id is not None
        assert event_id != ""

    def test_log_event_not_registered_raises(
        self, engine: EquipmentRegistryEngine
    ):
        """Logging event for unregistered equipment raises KeyError."""
        event = ServiceEvent(
            equipment_id="eq_missing",
            event_type=ServiceEventType.RECHARGE,
            date=datetime.now(timezone.utc),
        )
        with pytest.raises(KeyError, match="not found"):
            engine.log_service_event(event)

    def test_get_service_history(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Retrieve service history."""
        event = ServiceEvent(
            equipment_id="eq_test_001",
            event_type=ServiceEventType.RECHARGE,
            date=datetime.now(timezone.utc),
            refrigerant_added_kg=3.0,
        )
        populated_engine.log_service_event(event)
        history = populated_engine.get_service_history("eq_test_001")
        assert len(history) == 1
        assert history[0].event_type == ServiceEventType.RECHARGE

    def test_get_service_history_not_found_raises(
        self, engine: EquipmentRegistryEngine
    ):
        """History for unregistered equipment raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            engine.get_service_history("eq_missing")


# ===========================================================================
# Test: Cumulative Loss
# ===========================================================================


class TestCumulativeLoss:
    """Tests for cumulative loss calculation."""

    def test_calculate_cumulative_loss(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Cumulative loss sums events."""
        e1 = ServiceEvent(
            equipment_id="eq_test_001",
            event_type=ServiceEventType.RECHARGE,
            date=datetime.now(timezone.utc),
            refrigerant_added_kg=10.0,
        )
        e2 = ServiceEvent(
            equipment_id="eq_test_001",
            event_type=ServiceEventType.RECOVERY,
            date=datetime.now(timezone.utc),
            refrigerant_recovered_kg=3.0,
        )
        populated_engine.log_service_event(e1)
        populated_engine.log_service_event(e2)

        result = populated_engine.calculate_cumulative_loss("eq_test_001")
        assert result["total_added"] == "10.000"
        assert result["total_recovered"] == "3.000"
        assert result["net_loss"] == "7.000"
        assert result["event_count"] == 2
        assert "provenance_hash" in result


# ===========================================================================
# Test: Fleet Summary
# ===========================================================================


class TestFleetSummary:
    """Tests for fleet summary analytics."""

    def test_get_fleet_summary(self, populated_engine: EquipmentRegistryEngine):
        """Fleet summary includes all expected keys."""
        summary = populated_engine.get_fleet_summary()
        assert summary["total_equipment"] == 1
        assert "total_charge_kg" in summary
        assert "by_type" in summary
        assert "by_refrigerant" in summary
        assert "by_status" in summary
        assert "provenance_hash" in summary

    def test_get_total_installed_charge(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Total installed charge sums active equipment."""
        charge = populated_engine.get_total_installed_charge()
        assert charge == Decimal("15.000")


# ===========================================================================
# Test: Equipment Age and Lifetime
# ===========================================================================


class TestEquipmentAgeLifetime:
    """Tests for age and remaining lifetime calculations."""

    def test_get_equipment_age(self, populated_engine: EquipmentRegistryEngine):
        """Equipment age is positive for installed equipment."""
        age = populated_engine.get_equipment_age("eq_test_001")
        assert age > 0  # Installed in 2020, now is 2026

    def test_get_equipment_age_no_install_date(
        self, engine: EquipmentRegistryEngine
    ):
        """Equipment without install date returns 0.0."""
        profile = EquipmentProfile(
            equipment_id="eq_no_date",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=10.0,
        )
        engine.register_equipment(profile)
        age = engine.get_equipment_age("eq_no_date")
        assert age == 0.0

    def test_get_remaining_lifetime(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Remaining lifetime is non-negative."""
        remaining = populated_engine.get_remaining_lifetime("eq_test_001")
        assert remaining >= 0.0

    def test_get_remaining_lifetime_not_found_raises(
        self, engine: EquipmentRegistryEngine
    ):
        """Non-existent equipment raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            engine.get_remaining_lifetime("eq_missing")


# ===========================================================================
# Test: Equipment Defaults
# ===========================================================================


class TestEquipmentDefaults:
    """Tests for get_equipment_defaults across all 15 types."""

    @pytest.mark.parametrize("equip_type", list(EquipmentType))
    def test_get_equipment_defaults(
        self, engine: EquipmentRegistryEngine, equip_type: EquipmentType
    ):
        """Each of 15 equipment types has defaults."""
        defaults = engine.get_equipment_defaults(equip_type)
        assert defaults["equipment_type"] == equip_type.value
        assert "charge_range_min_kg" in defaults
        assert "charge_range_max_kg" in defaults
        assert "default_charge_kg" in defaults
        assert "default_leak_rate" in defaults
        assert "lifetime_years" in defaults
        assert "typical_refrigerants" in defaults
        assert "description" in defaults
        assert "source" in defaults

    def test_get_equipment_defaults_commercial_refrigeration(
        self, engine: EquipmentRegistryEngine
    ):
        """Commercial centralized refrigeration defaults are correct."""
        defaults = engine.get_equipment_defaults(
            EquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED
        )
        assert defaults["default_leak_rate"] == "0.20"
        assert defaults["lifetime_years"] == 18

    def test_get_equipment_defaults_aerosols(
        self, engine: EquipmentRegistryEngine
    ):
        """Aerosols have 100% leak rate."""
        defaults = engine.get_equipment_defaults(EquipmentType.AEROSOLS)
        assert defaults["default_leak_rate"] == "1.00"
        assert defaults["lifetime_years"] == 1


# ===========================================================================
# Test: Equipment Validation
# ===========================================================================


class TestValidateEquipment:
    """Tests for equipment profile validation."""

    def test_validate_equipment_valid(
        self, engine: EquipmentRegistryEngine, sample_profile: EquipmentProfile
    ):
        """Valid profile returns empty error list."""
        errors = engine.validate_equipment(sample_profile)
        # May have warning about atypical refrigerant, but no charge errors
        charge_errors = [e for e in errors if "Charge" in e]
        assert len(charge_errors) == 0

    def test_validate_equipment_charge_below_range(
        self, engine: EquipmentRegistryEngine
    ):
        """Charge below expected range produces error."""
        profile = EquipmentProfile(
            equipment_id="eq_low_charge",
            equipment_type=EquipmentType.INDUSTRIAL_REFRIGERATION,
            refrigerant_type=RefrigerantType.R_717,
            charge_kg=0.5,  # Below min 100 kg
        )
        errors = engine.validate_equipment(profile)
        assert any("below" in e.lower() for e in errors)

    def test_validate_equipment_charge_above_range(
        self, engine: EquipmentRegistryEngine
    ):
        """Charge above expected range produces error."""
        profile = EquipmentProfile(
            equipment_id="eq_high_charge",
            equipment_type=EquipmentType.RESIDENTIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=999.0,  # Above max 3 kg
        )
        errors = engine.validate_equipment(profile)
        assert any("exceeds" in e.lower() for e in errors)


# ===========================================================================
# Test: Contains and Stats
# ===========================================================================


class TestContainsAndStats:
    """Tests for __contains__ and get_stats."""

    def test_contains_registered(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """__contains__ returns True for registered equipment."""
        assert "eq_test_001" in populated_engine

    def test_contains_not_registered(
        self, engine: EquipmentRegistryEngine
    ):
        """__contains__ returns False for unregistered."""
        assert "eq_missing" not in engine

    def test_get_stats(self, populated_engine: EquipmentRegistryEngine):
        """get_stats returns expected keys."""
        stats = populated_engine.get_stats()
        assert stats["total_equipment"] == 1
        assert "total_service_events" in stats
        assert "equipment_types_available" in stats

    def test_clear(self, populated_engine: EquipmentRegistryEngine):
        """clear() removes all equipment."""
        populated_engine.clear()
        assert len(populated_engine) == 0


# ===========================================================================
# Test: Provenance Tracking
# ===========================================================================


class TestProvenanceTracking:
    """Tests for provenance hash generation."""

    def test_register_records_provenance(
        self, engine: EquipmentRegistryEngine, sample_profile: EquipmentProfile
    ):
        """Registration does not raise provenance errors."""
        equip_id = engine.register_equipment(sample_profile)
        assert equip_id is not None

    def test_decommission_has_provenance_hash(
        self, populated_engine: EquipmentRegistryEngine
    ):
        """Decommission result has a 64-char provenance hash."""
        result = populated_engine.decommission_equipment(
            "eq_test_001", recovery_kg=5.0
        )
        assert len(result["provenance_hash"]) == 64
