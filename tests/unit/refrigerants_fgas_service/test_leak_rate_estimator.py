# -*- coding: utf-8 -*-
"""
Unit tests for LeakRateEstimatorEngine - AGENT-MRV-002 Engine 4

Tests leak rate estimation, age/climate/LDAR adjustment factors,
lifecycle rates, annual loss, lifetime emissions, custom rate
registration, and provenance tracking.

Target: 55+ tests, 600+ lines.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.refrigerants_fgas.leak_rate_estimator import (
    LeakRateEstimatorEngine,
    LeakEquipmentType,
    LifecycleStage,
    ClimateZone,
    LDARLevel,
    LeakRateProfile,
    LifetimeEmissionsProfile,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> LeakRateEstimatorEngine:
    """Create a fresh LeakRateEstimatorEngine."""
    return LeakRateEstimatorEngine()


# ===========================================================================
# Test: Initialization
# ===========================================================================


class TestLeakRateEstimatorInit:
    """Tests for engine initialization."""

    def test_initialization(self, engine: LeakRateEstimatorEngine):
        """Engine initializes with zero history."""
        assert len(engine) == 0

    def test_repr(self, engine: LeakRateEstimatorEngine):
        """repr includes key info."""
        r = repr(engine)
        assert "LeakRateEstimatorEngine" in r
        assert "equipment_types=" in r

    def test_list_equipment_types(self, engine: LeakRateEstimatorEngine):
        """15 equipment types are available."""
        types = engine.list_equipment_types()
        assert len(types) == 15


# ===========================================================================
# Test: estimate_leak_rate default (operating, no adjustments)
# ===========================================================================


class TestEstimateLeakRateDefault:
    """Tests for default leak rate estimation."""

    def test_estimate_leak_rate_default(self, engine: LeakRateEstimatorEngine):
        """Default estimation returns a LeakRateProfile."""
        profile = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
        )
        assert isinstance(profile, LeakRateProfile)
        assert profile.effective_rate > Decimal("0")
        assert profile.provenance_hash != ""
        assert len(profile.provenance_hash) == 64

    def test_estimate_leak_rate_default_operating(
        self, engine: LeakRateEstimatorEngine
    ):
        """Default lifecycle stage is OPERATING."""
        profile = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
        )
        assert profile.lifecycle_stage == LifecycleStage.OPERATING.value


# ===========================================================================
# Test: estimate_leak_rate with age adjustment
# ===========================================================================


class TestAgeAdjustment:
    """Tests for age-based leak rate adjustments."""

    def test_estimate_leak_rate_with_age_adjustment(
        self, engine: LeakRateEstimatorEngine
    ):
        """Older equipment has higher effective rate."""
        profile_new = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
            age_years=2,
        )
        profile_old = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
            age_years=12,
        )
        assert profile_old.effective_rate > profile_new.effective_rate

    @pytest.mark.parametrize("age,expected_factor", [
        (0, Decimal("1.00")),
        (2, Decimal("1.00")),
        (4, Decimal("1.00")),
        (5, Decimal("1.15")),
        (7, Decimal("1.15")),
        (9, Decimal("1.15")),
        (10, Decimal("1.35")),
        (12, Decimal("1.35")),
        (14, Decimal("1.35")),
        (15, Decimal("1.60")),
        (20, Decimal("1.60")),
        (30, Decimal("1.60")),
    ])
    def test_calculate_age_factor(
        self, engine: LeakRateEstimatorEngine, age, expected_factor
    ):
        """Age factor matches bracket table."""
        factor = engine.calculate_age_factor(age)
        assert factor == expected_factor

    def test_calculate_age_factor_negative_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """Negative age raises ValueError."""
        with pytest.raises(ValueError, match="age_years must be >= 0"):
            engine.calculate_age_factor(-1)


# ===========================================================================
# Test: estimate_leak_rate with climate zone
# ===========================================================================


class TestClimateZoneAdjustment:
    """Tests for climate zone adjustments."""

    @pytest.mark.parametrize("zone,expected_factor", [
        (ClimateZone.TROPICAL.value, Decimal("1.15")),
        (ClimateZone.SUBTROPICAL.value, Decimal("1.10")),
        (ClimateZone.TEMPERATE.value, Decimal("1.00")),
        (ClimateZone.CONTINENTAL.value, Decimal("0.95")),
        (ClimateZone.POLAR.value, Decimal("0.90")),
    ])
    def test_calculate_climate_factor(
        self, engine: LeakRateEstimatorEngine, zone, expected_factor
    ):
        """Climate factor matches zone table."""
        factor = engine.calculate_climate_factor(zone)
        assert factor == expected_factor

    def test_estimate_with_climate_zone(
        self, engine: LeakRateEstimatorEngine
    ):
        """Tropical zone increases effective rate vs temperate."""
        profile_temp = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
            climate_zone=ClimateZone.TEMPERATE.value,
        )
        profile_trop = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
            climate_zone=ClimateZone.TROPICAL.value,
        )
        assert profile_trop.effective_rate > profile_temp.effective_rate

    def test_invalid_climate_zone_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """Invalid climate zone raises ValueError."""
        with pytest.raises(ValueError, match="Unknown climate_zone"):
            engine.calculate_climate_factor("MARS")


# ===========================================================================
# Test: estimate_leak_rate with LDAR
# ===========================================================================


class TestLDARAdjustment:
    """Tests for LDAR program adjustments."""

    @pytest.mark.parametrize("level,expected_factor", [
        (LDARLevel.NONE.value, Decimal("1.00")),
        (LDARLevel.ANNUAL.value, Decimal("0.85")),
        (LDARLevel.QUARTERLY.value, Decimal("0.70")),
        (LDARLevel.CONTINUOUS.value, Decimal("0.50")),
    ])
    def test_calculate_ldar_factor(
        self, engine: LeakRateEstimatorEngine, level, expected_factor
    ):
        """LDAR factor matches level table."""
        factor = engine.calculate_ldar_factor(level)
        assert factor == expected_factor

    def test_estimate_with_ldar(self, engine: LeakRateEstimatorEngine):
        """Continuous LDAR reduces effective rate vs no LDAR."""
        profile_none = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
            ldar_level=LDARLevel.NONE.value,
        )
        profile_cont = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
            ldar_level=LDARLevel.CONTINUOUS.value,
        )
        assert profile_cont.effective_rate < profile_none.effective_rate

    def test_invalid_ldar_level_raises(self, engine: LeakRateEstimatorEngine):
        """Invalid LDAR level raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ldar_level"):
            engine.calculate_ldar_factor("DAILY")


# ===========================================================================
# Test: Combined factor estimation
# ===========================================================================


class TestCombinedFactors:
    """Tests with multiple adjustment factors applied."""

    def test_estimate_combined_factors(self, engine: LeakRateEstimatorEngine):
        """All factors applied simultaneously."""
        profile = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
            lifecycle_stage=LifecycleStage.OPERATING.value,
            age_years=12,
            climate_zone=ClimateZone.SUBTROPICAL.value,
            ldar_level=LDARLevel.QUARTERLY.value,
        )
        # base=0.20, age=1.35, climate=1.10, ldar=0.70
        expected = Decimal("0.20") * Decimal("1.35") * Decimal("1.10") * Decimal("0.70")
        assert profile.effective_rate == pytest.approx(
            float(expected), abs=0.001
        )

    def test_estimate_capped_at_one(self, engine: LeakRateEstimatorEngine):
        """Effective rate is capped at 1.0 (100%)."""
        profile = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.AEROSOL_MDI.value,
            lifecycle_stage=LifecycleStage.OPERATING.value,
            age_years=20,
            climate_zone=ClimateZone.TROPICAL.value,
        )
        assert profile.effective_rate <= Decimal("1")


# ===========================================================================
# Test: get_default_rate for all 15 equipment types
# ===========================================================================


class TestGetDefaultRate:
    """Tests for get_default_rate."""

    @pytest.mark.parametrize("equip_type", [e.value for e in LeakEquipmentType])
    def test_get_default_rate(
        self, engine: LeakRateEstimatorEngine, equip_type
    ):
        """Each equipment type has a default rate."""
        rate = engine.get_default_rate(equip_type)
        assert isinstance(rate, Decimal)
        assert rate >= Decimal("0")
        assert rate <= Decimal("1")

    def test_get_default_rate_invalid_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown equipment_type"):
            engine.get_default_rate("INVALID_TYPE")


# ===========================================================================
# Test: get_lifecycle_rates
# ===========================================================================


class TestGetLifecycleRates:
    """Tests for lifecycle stage rates."""

    @pytest.mark.parametrize("equip_type", [
        LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
        LeakEquipmentType.CHILLER_CENTRIFUGAL.value,
        LeakEquipmentType.SWITCHGEAR_SF6.value,
    ])
    def test_get_lifecycle_rates(
        self, engine: LeakRateEstimatorEngine, equip_type
    ):
        """Returns rates for all 3 lifecycle stages."""
        rates = engine.get_lifecycle_rates(equip_type)
        assert LifecycleStage.INSTALLATION.value in rates
        assert LifecycleStage.OPERATING.value in rates
        assert LifecycleStage.END_OF_LIFE.value in rates
        for stage, rate in rates.items():
            assert isinstance(rate, Decimal)
            assert rate >= Decimal("0")


# ===========================================================================
# Test: estimate_annual_loss
# ===========================================================================


class TestEstimateAnnualLoss:
    """Tests for annual refrigerant loss estimation."""

    def test_estimate_annual_loss(self, engine: LeakRateEstimatorEngine):
        """Annual loss = charge * effective_rate."""
        loss = engine.estimate_annual_loss(
            charge_kg=Decimal("200"),
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
        )
        # Default rate 0.20, temperate, no LDAR, age 0 -> effective = 0.20
        assert loss == Decimal("40.000")

    def test_estimate_annual_loss_zero_charge_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """Zero charge raises ValueError."""
        with pytest.raises(ValueError, match="charge_kg must be > 0"):
            engine.estimate_annual_loss(
                charge_kg=Decimal("0"),
                equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
            )


# ===========================================================================
# Test: estimate_lifetime_emissions
# ===========================================================================


class TestEstimateLifetimeEmissions:
    """Tests for lifetime emission estimation."""

    def test_estimate_lifetime_emissions(
        self, engine: LeakRateEstimatorEngine
    ):
        """Lifetime emissions cover install + operating + EOL."""
        profile = engine.estimate_lifetime_emissions(
            charge_kg=Decimal("200"),
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
            gwp=Decimal("3922"),
        )
        assert isinstance(profile, LifetimeEmissionsProfile)
        assert profile.installation_loss_kg > Decimal("0")
        assert profile.total_operating_loss_kg > Decimal("0")
        assert profile.end_of_life_loss_kg >= Decimal("0")
        assert profile.total_lifetime_emissions_tco2e > Decimal("0")
        assert len(profile.provenance_hash) == 64

    def test_lifetime_total_equals_sum_of_stages(
        self, engine: LeakRateEstimatorEngine
    ):
        """Total loss = install + operating + eol."""
        profile = engine.estimate_lifetime_emissions(
            charge_kg=Decimal("100"),
            equipment_type=LeakEquipmentType.CHILLER_CENTRIFUGAL.value,
            gwp=Decimal("1530"),
        )
        total = (
            profile.installation_loss_kg
            + profile.total_operating_loss_kg
            + profile.end_of_life_loss_kg
        )
        assert abs(profile.total_lifetime_loss_kg - total) < Decimal("0.01")

    def test_lifetime_zero_charge_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """Zero charge raises ValueError."""
        with pytest.raises(ValueError, match="charge_kg must be > 0"):
            engine.estimate_lifetime_emissions(
                charge_kg=Decimal("0"),
                equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
                gwp=Decimal("2088"),
            )


# ===========================================================================
# Test: Custom Rate Registration
# ===========================================================================


class TestCustomRateRegistration:
    """Tests for custom leak rate registration."""

    def test_register_custom_rate(self, engine: LeakRateEstimatorEngine):
        """Custom rate is registered and used."""
        key = engine.register_custom_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
            lifecycle_stage=LifecycleStage.OPERATING.value,
            rate=Decimal("0.30"),
        )
        assert "COMMERCIAL_REFRIGERATION_CENTRALIZED:OPERATING" == key

        profile = engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value,
        )
        assert profile.base_rate == Decimal("0.30")
        assert profile.rate_source == "registered_custom"

    def test_register_custom_rate_invalid_rate_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """Rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="rate must be in"):
            engine.register_custom_rate(
                equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
                lifecycle_stage=LifecycleStage.OPERATING.value,
                rate=Decimal("1.5"),
            )

    def test_unregister_custom_rate(self, engine: LeakRateEstimatorEngine):
        """Custom rate can be removed."""
        engine.register_custom_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
            lifecycle_stage=LifecycleStage.OPERATING.value,
            rate=Decimal("0.15"),
        )
        result = engine.unregister_custom_rate(
            LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
            LifecycleStage.OPERATING.value,
        )
        assert result is True

    def test_unregister_nonexistent_returns_false(
        self, engine: LeakRateEstimatorEngine
    ):
        """Removing non-existent custom rate returns False."""
        result = engine.unregister_custom_rate("NONEXISTENT", "OPERATING")
        assert result is False


# ===========================================================================
# Test: list_rates
# ===========================================================================


class TestListRates:
    """Tests for list_rates."""

    def test_list_rates_all(self, engine: LeakRateEstimatorEngine):
        """List all rates returns 15 entries."""
        rates = engine.list_rates()
        assert len(rates) == 15

    def test_list_rates_filtered(self, engine: LeakRateEstimatorEngine):
        """Filtered list returns 1 entry."""
        rates = engine.list_rates(
            equipment_type=LeakEquipmentType.SWITCHGEAR_SF6.value
        )
        assert len(rates) == 1
        assert rates[0]["equipment_type"] == LeakEquipmentType.SWITCHGEAR_SF6.value


# ===========================================================================
# Test: History and Stats
# ===========================================================================


class TestHistoryAndStats:
    """Tests for estimation history and stats."""

    def test_history_grows(self, engine: LeakRateEstimatorEngine):
        """History count increases."""
        engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
        )
        assert len(engine.get_history()) == 1

    def test_stats(self, engine: LeakRateEstimatorEngine):
        """Stats returns expected keys."""
        stats = engine.get_stats()
        assert "total_estimations" in stats
        assert "equipment_types_available" in stats

    def test_clear(self, engine: LeakRateEstimatorEngine):
        """Clear removes all history."""
        engine.estimate_leak_rate(
            equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
        )
        engine.clear()
        assert len(engine) == 0


# ===========================================================================
# Test: Validation Errors
# ===========================================================================


class TestValidationErrors:
    """Tests for input validation errors."""

    def test_invalid_equipment_type_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """Invalid equipment type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown equipment_type"):
            engine.estimate_leak_rate(equipment_type="INVALID")

    def test_invalid_lifecycle_stage_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """Invalid lifecycle stage raises ValueError."""
        with pytest.raises(ValueError, match="Unknown lifecycle_stage"):
            engine.estimate_leak_rate(
                equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
                lifecycle_stage="INVALID",
            )

    def test_negative_age_raises(self, engine: LeakRateEstimatorEngine):
        """Negative age_years raises ValueError."""
        with pytest.raises(ValueError, match="age_years must be >= 0"):
            engine.estimate_leak_rate(
                equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
                age_years=-5,
            )

    def test_custom_rate_out_of_range_raises(
        self, engine: LeakRateEstimatorEngine
    ):
        """custom_rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="custom_rate must be in"):
            engine.estimate_leak_rate(
                equipment_type=LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value,
                custom_rate=Decimal("2.0"),
            )
