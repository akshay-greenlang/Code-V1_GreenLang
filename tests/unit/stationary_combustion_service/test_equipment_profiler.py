# -*- coding: utf-8 -*-
"""
Unit tests for EquipmentProfilerEngine (Engine 3) - AGENT-MRV-001

Tests all methods of EquipmentProfilerEngine with 50+ tests covering:
- Initialization with empty registry
- Equipment registration (boiler, furnace, turbine)
- Equipment retrieval by ID
- Equipment update (capacity, efficiency, age)
- Equipment deletion
- Equipment listing with type/facility filters
- Efficiency calculation via polynomial curves
- Default efficiency templates for all 13 equipment types
- Age degradation with linear model
- Adjusted emissions incorporating efficiency and age
- Equipment template lookups
- Equipment count tracking
- Clear/reset functionality
- Duplicate registration rejection
- Invalid equipment type validation
- Thread safety for concurrent register/get operations

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal

import pytest

from greenlang.stationary_combustion.equipment_profiler import (
    EquipmentProfilerEngine,
    EQUIPMENT_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def profiler():
    """Create an EquipmentProfilerEngine with provenance disabled."""
    return EquipmentProfilerEngine(config={"enable_provenance": False})


@pytest.fixture
def profiler_with_boiler(profiler):
    """Create a profiler with one registered boiler."""
    profiler.register_equipment(
        equipment_id="boiler_001",
        equipment_type="BOILER_WATER_TUBE",
        name="Main Boiler",
        rated_capacity=100.0,
        age_years=5,
    )
    return profiler


# ---------------------------------------------------------------------------
# TestEquipmentProfilerInit
# ---------------------------------------------------------------------------

class TestEquipmentProfilerInit:
    """Tests for EquipmentProfilerEngine initialization."""

    def test_initializes_with_empty_registry(self, profiler):
        """Engine starts with no registered equipment."""
        assert profiler.get_equipment_count() == 0

    def test_has_13_default_templates(self):
        """Engine provides 13 built-in equipment templates."""
        assert len(EQUIPMENT_DEFAULTS) == 13

    def test_repr_shows_empty(self, profiler):
        """__repr__ shows zero registered equipment."""
        r = repr(profiler)
        assert "registered=0" in r

    def test_len_returns_count(self, profiler):
        """__len__ returns the equipment count."""
        assert len(profiler) == 0


# ---------------------------------------------------------------------------
# TestRegisterEquipment
# ---------------------------------------------------------------------------

class TestRegisterEquipment:
    """Tests for equipment registration."""

    def test_register_boiler(self, profiler):
        """Register a water tube boiler successfully."""
        profile = profiler.register_equipment(
            equipment_id="boiler_001",
            equipment_type="BOILER_WATER_TUBE",
            name="Test Boiler",
            rated_capacity=50.0,
        )
        assert profile.equipment_id == "boiler_001"
        assert profiler.get_equipment_count() == 1

    def test_register_furnace(self, profiler):
        """Register a furnace successfully."""
        profile = profiler.register_equipment(
            equipment_id="furnace_001",
            equipment_type="FURNACE",
            name="Test Furnace",
            rated_capacity=200.0,
        )
        assert profile.equipment_id == "furnace_001"

    def test_register_gas_turbine(self, profiler):
        """Register a gas turbine successfully."""
        profile = profiler.register_equipment(
            equipment_id="turbine_001",
            equipment_type="GAS_TURBINE_SIMPLE",
            name="Test Turbine",
            rated_capacity=50.0,
        )
        assert profile.equipment_id == "turbine_001"

    def test_register_with_facility_id(self, profiler):
        """Register equipment with a facility ID."""
        profile = profiler.register_equipment(
            equipment_id="boiler_f01",
            equipment_type="BOILER_FIRE_TUBE",
            name="Facility Boiler",
            facility_id="FAC-001",
        )
        assert profile.facility_id == "FAC-001"

    def test_register_with_custom_load_factor(self, profiler):
        """Register equipment with a custom load factor."""
        profile = profiler.register_equipment(
            equipment_id="boiler_lf",
            equipment_type="BOILER_WATER_TUBE",
            name="Custom LF Boiler",
            load_factor=0.85,
        )
        assert profile.load_factor == Decimal("0.85")

    def test_register_with_age(self, profiler):
        """Register equipment with age_years."""
        profile = profiler.register_equipment(
            equipment_id="old_boiler",
            equipment_type="BOILER_WATER_TUBE",
            name="Old Boiler",
            age_years=20,
        )
        assert profile.age_years == 20


# ---------------------------------------------------------------------------
# TestGetEquipment
# ---------------------------------------------------------------------------

class TestGetEquipment:
    """Tests for equipment retrieval."""

    def test_get_registered_equipment(self, profiler_with_boiler):
        """Retrieve a registered equipment profile by ID."""
        profile = profiler_with_boiler.get_equipment("boiler_001")
        assert profile.equipment_id == "boiler_001"
        assert profile.name == "Main Boiler"

    def test_get_nonexistent_raises_key_error(self, profiler):
        """Getting non-existent equipment raises KeyError."""
        with pytest.raises(KeyError, match="Equipment not found"):
            profiler.get_equipment("nonexistent")


# ---------------------------------------------------------------------------
# TestUpdateEquipment
# ---------------------------------------------------------------------------

class TestUpdateEquipment:
    """Tests for equipment updates."""

    def test_update_age(self, profiler_with_boiler):
        """Update equipment age_years."""
        updated = profiler_with_boiler.update_equipment(
            "boiler_001", age_years=10,
        )
        assert updated.age_years == 10

    def test_update_efficiency(self, profiler_with_boiler):
        """Update equipment efficiency."""
        updated = profiler_with_boiler.update_equipment(
            "boiler_001", efficiency=0.90,
        )
        assert updated.efficiency == Decimal("0.90")

    def test_update_name(self, profiler_with_boiler):
        """Update equipment name."""
        updated = profiler_with_boiler.update_equipment(
            "boiler_001", name="Renamed Boiler",
        )
        assert updated.name == "Renamed Boiler"

    def test_update_nonexistent_raises_key_error(self, profiler):
        """Updating non-existent equipment raises KeyError."""
        with pytest.raises(KeyError):
            profiler.update_equipment("nonexistent", age_years=5)

    def test_update_refreshes_timestamp(self, profiler_with_boiler):
        """Update refreshes the updated_at timestamp."""
        original = profiler_with_boiler.get_equipment("boiler_001")
        updated = profiler_with_boiler.update_equipment(
            "boiler_001", age_years=10,
        )
        assert updated.updated_at >= original.updated_at


# ---------------------------------------------------------------------------
# TestDeleteEquipment
# ---------------------------------------------------------------------------

class TestDeleteEquipment:
    """Tests for equipment deletion."""

    def test_delete_existing(self, profiler_with_boiler):
        """Delete registered equipment returns True."""
        result = profiler_with_boiler.delete_equipment("boiler_001")
        assert result is True
        assert profiler_with_boiler.get_equipment_count() == 0

    def test_delete_nonexistent(self, profiler):
        """Delete non-existent equipment returns False."""
        result = profiler.delete_equipment("nonexistent")
        assert result is False

    def test_deleted_equipment_not_retrievable(self, profiler_with_boiler):
        """Deleted equipment cannot be retrieved."""
        profiler_with_boiler.delete_equipment("boiler_001")
        with pytest.raises(KeyError):
            profiler_with_boiler.get_equipment("boiler_001")


# ---------------------------------------------------------------------------
# TestListEquipment
# ---------------------------------------------------------------------------

class TestListEquipment:
    """Tests for equipment listing."""

    def test_list_all(self, profiler):
        """List all registered equipment."""
        profiler.register_equipment("b1", "BOILER_WATER_TUBE", "Boiler 1")
        profiler.register_equipment("f1", "FURNACE", "Furnace 1")
        profiles = profiler.list_equipment()
        assert len(profiles) == 2

    def test_filter_by_type(self, profiler):
        """Filter equipment by type."""
        profiler.register_equipment("b1", "BOILER_WATER_TUBE", "Boiler 1")
        profiler.register_equipment("f1", "FURNACE", "Furnace 1")
        boilers = profiler.list_equipment(equipment_type="BOILER_WATER_TUBE")
        assert len(boilers) == 1
        assert boilers[0].equipment_id == "b1"

    def test_filter_by_facility(self, profiler):
        """Filter equipment by facility_id."""
        profiler.register_equipment(
            "b1", "BOILER_WATER_TUBE", "Boiler 1", facility_id="FAC-A",
        )
        profiler.register_equipment(
            "b2", "BOILER_WATER_TUBE", "Boiler 2", facility_id="FAC-B",
        )
        fac_a = profiler.list_equipment(facility_id="FAC-A")
        assert len(fac_a) == 1
        assert fac_a[0].equipment_id == "b1"

    def test_empty_list(self, profiler):
        """List on empty registry returns empty list."""
        assert profiler.list_equipment() == []


# ---------------------------------------------------------------------------
# TestCalculateEfficiency
# ---------------------------------------------------------------------------

class TestCalculateEfficiency:
    """Tests for polynomial efficiency curve evaluation."""

    def test_efficiency_at_default_load(self, profiler_with_boiler):
        """Efficiency at default load factor is in valid range."""
        eff = profiler_with_boiler.calculate_efficiency("boiler_001")
        assert 0.01 <= eff <= 1.0

    def test_efficiency_at_full_load(self, profiler_with_boiler):
        """Efficiency at full load (1.0) is in valid range."""
        eff = profiler_with_boiler.calculate_efficiency("boiler_001", load_factor=1.0)
        assert 0.01 <= eff <= 1.0

    def test_efficiency_at_zero_load(self, profiler_with_boiler):
        """Efficiency at zero load is clamped to 0.01 minimum."""
        eff = profiler_with_boiler.calculate_efficiency("boiler_001", load_factor=0.0)
        assert eff >= 0.01

    def test_efficiency_increases_with_load(self, profiler_with_boiler):
        """Efficiency generally increases from low to moderate load."""
        eff_low = profiler_with_boiler.calculate_efficiency(
            "boiler_001", load_factor=0.2,
        )
        eff_mid = profiler_with_boiler.calculate_efficiency(
            "boiler_001", load_factor=0.5,
        )
        # For most equipment types efficiency increases with load up to ~75%
        assert eff_mid >= eff_low

    def test_invalid_load_factor_raises(self, profiler_with_boiler):
        """Load factor outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="load_factor must be in"):
            profiler_with_boiler.calculate_efficiency("boiler_001", load_factor=1.5)

    def test_negative_load_factor_raises(self, profiler_with_boiler):
        """Negative load factor raises ValueError."""
        with pytest.raises(ValueError):
            profiler_with_boiler.calculate_efficiency("boiler_001", load_factor=-0.1)


# ---------------------------------------------------------------------------
# TestDefaultEfficiency
# ---------------------------------------------------------------------------

class TestDefaultEfficiency:
    """Tests for default templates for all 13 equipment types."""

    @pytest.mark.parametrize("eq_type", sorted(EQUIPMENT_DEFAULTS.keys()))
    def test_template_has_efficiency(self, eq_type):
        """Every template has a default efficiency value."""
        assert "efficiency" in EQUIPMENT_DEFAULTS[eq_type]
        eff = EQUIPMENT_DEFAULTS[eq_type]["efficiency"]
        assert Decimal("0") < eff <= Decimal("1")

    @pytest.mark.parametrize("eq_type", sorted(EQUIPMENT_DEFAULTS.keys()))
    def test_template_has_curve(self, eq_type):
        """Every template has an efficiency curve with 4 coefficients."""
        curve = EQUIPMENT_DEFAULTS[eq_type]["efficiency_curve"]
        assert len(curve) == 4

    @pytest.mark.parametrize("eq_type", sorted(EQUIPMENT_DEFAULTS.keys()))
    def test_register_all_types(self, profiler, eq_type):
        """All 13 equipment types can be registered."""
        profile = profiler.register_equipment(
            equipment_id=f"test_{eq_type.lower()}",
            equipment_type=eq_type,
            name=f"Test {eq_type}",
        )
        assert profile is not None


# ---------------------------------------------------------------------------
# TestAgeDegradation
# ---------------------------------------------------------------------------

class TestAgeDegradation:
    """Tests for age degradation calculation."""

    def test_new_equipment_no_degradation(self, profiler):
        """New equipment (age=0) has degradation factor 1.0."""
        profiler.register_equipment(
            "new_boiler", "BOILER_WATER_TUBE", "New Boiler", age_years=0,
        )
        deg = profiler.calculate_age_degradation("new_boiler")
        assert deg == 1.0

    def test_aged_equipment_degraded(self, profiler):
        """Aged equipment has degradation factor < 1.0."""
        profiler.register_equipment(
            "old_boiler", "BOILER_WATER_TUBE", "Old Boiler", age_years=20,
        )
        deg = profiler.calculate_age_degradation("old_boiler")
        assert deg < 1.0
        # BOILER_WATER_TUBE degradation_rate = 0.004
        # deg = 1.0 - 20 * 0.004 = 1.0 - 0.08 = 0.92
        assert abs(deg - 0.92) < 0.01

    def test_degradation_clamped_at_0_5(self, profiler):
        """Degradation factor is clamped to minimum 0.5."""
        profiler.register_equipment(
            "ancient_boiler", "BOILER_WATER_TUBE", "Ancient Boiler",
            age_years=200,
        )
        deg = profiler.calculate_age_degradation("ancient_boiler")
        assert deg == 0.5


# ---------------------------------------------------------------------------
# TestAdjustedEmissions
# ---------------------------------------------------------------------------

class TestAdjustedEmissions:
    """Tests for efficiency and age adjusted emissions."""

    def test_adjusted_emissions_higher(self, profiler):
        """Adjusted emissions are higher than base (less efficient equipment)."""
        profiler.register_equipment(
            "boiler", "BOILER_WATER_TUBE", "Boiler", age_years=10,
        )
        result = profiler.calculate_adjusted_emissions(
            base_emissions_kg=1000.0,
            equipment_id="boiler",
            load_factor=0.7,
        )
        assert result["adjusted_emissions_kg"] >= result["base_emissions_kg"]

    def test_adjusted_contains_all_fields(self, profiler):
        """Adjusted emissions dict contains all required fields."""
        profiler.register_equipment(
            "boiler", "BOILER_WATER_TUBE", "Boiler", age_years=5,
        )
        result = profiler.calculate_adjusted_emissions(1000.0, "boiler")
        expected_keys = {
            "base_emissions_kg", "adjusted_emissions_kg",
            "adjustment_factor", "curve_efficiency",
            "age_degradation", "actual_efficiency", "equipment_id",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_adjustment_factor_positive(self, profiler):
        """Adjustment factor is always positive."""
        profiler.register_equipment(
            "boiler", "BOILER_WATER_TUBE", "Boiler", age_years=5,
        )
        result = profiler.calculate_adjusted_emissions(1000.0, "boiler")
        assert result["adjustment_factor"] > 0


# ---------------------------------------------------------------------------
# TestEquipmentTemplate
# ---------------------------------------------------------------------------

class TestEquipmentTemplate:
    """Tests for get_equipment_template method."""

    def test_get_boiler_template(self, profiler):
        """Boiler template has correct default efficiency."""
        template = profiler.get_equipment_template("BOILER_WATER_TUBE")
        assert template["efficiency"] == 0.86

    def test_template_contains_all_fields(self, profiler):
        """Template dict contains all expected fields."""
        template = profiler.get_equipment_template("FURNACE")
        expected_keys = {
            "equipment_type", "efficiency", "capacity_range_mw",
            "load_factor", "degradation_rate", "efficiency_curve",
        }
        assert expected_keys.issubset(set(template.keys()))

    def test_unknown_template_raises(self, profiler):
        """Unknown equipment type raises KeyError."""
        with pytest.raises(KeyError, match="Unknown equipment type"):
            profiler.get_equipment_template("UNKNOWN_TYPE")


# ---------------------------------------------------------------------------
# TestEquipmentCount
# ---------------------------------------------------------------------------

class TestEquipmentCount:
    """Tests for equipment count tracking."""

    def test_count_increments(self, profiler):
        """Count increments after registration."""
        assert profiler.get_equipment_count() == 0
        profiler.register_equipment("b1", "BOILER_WATER_TUBE", "B1")
        assert profiler.get_equipment_count() == 1
        profiler.register_equipment("b2", "FURNACE", "B2")
        assert profiler.get_equipment_count() == 2

    def test_count_decrements_on_delete(self, profiler):
        """Count decrements after deletion."""
        profiler.register_equipment("b1", "BOILER_WATER_TUBE", "B1")
        profiler.delete_equipment("b1")
        assert profiler.get_equipment_count() == 0


# ---------------------------------------------------------------------------
# TestClear
# ---------------------------------------------------------------------------

class TestClear:
    """Tests for clear/reset functionality."""

    def test_clear_removes_all(self, profiler):
        """Clear removes all registered equipment."""
        profiler.register_equipment("b1", "BOILER_WATER_TUBE", "B1")
        profiler.register_equipment("b2", "FURNACE", "B2")
        profiler.clear()
        assert profiler.get_equipment_count() == 0


# ---------------------------------------------------------------------------
# TestDuplicateRegistration
# ---------------------------------------------------------------------------

class TestDuplicateRegistration:
    """Tests for duplicate equipment ID rejection."""

    def test_duplicate_raises_value_error(self, profiler):
        """Registering the same equipment_id twice raises ValueError."""
        profiler.register_equipment("b1", "BOILER_WATER_TUBE", "B1")
        with pytest.raises(ValueError, match="already registered"):
            profiler.register_equipment("b1", "BOILER_WATER_TUBE", "B1 Copy")


# ---------------------------------------------------------------------------
# TestInvalidEquipmentType
# ---------------------------------------------------------------------------

class TestInvalidEquipmentType:
    """Tests for invalid equipment type validation."""

    def test_invalid_type_raises_value_error(self, profiler):
        """Unknown equipment type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown equipment type"):
            profiler.register_equipment("x1", "FLYING_SAUCER", "X1")


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Thread safety tests for concurrent operations."""

    def test_concurrent_register_get(self, profiler):
        """Concurrent register and get operations do not corrupt state."""
        errors = []

        def register_and_get(thread_id):
            try:
                eq_id = f"eq_{thread_id}"
                profiler.register_equipment(
                    eq_id, "BOILER_WATER_TUBE", f"Boiler {thread_id}",
                )
                profile = profiler.get_equipment(eq_id)
                assert profile.equipment_id == eq_id
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_and_get, args=(i,))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert profiler.get_equipment_count() == 20
