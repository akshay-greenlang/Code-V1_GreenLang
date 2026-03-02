# -*- coding: utf-8 -*-
"""
Unit tests for TeleworkCalculatorEngine (Engine 4).

Tests telework/WFH emissions calculation including full remote, hybrid patterns,
grid factor resolution, seasonal adjustments, eGRID subregional factors,
equipment lifecycle emissions, and net impact analysis.

Target: ~30 tests covering all telework categories, climate zones, grid
factors, heating fuels, equipment, hybrid worker, and input validation.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUANT = Decimal("0.00000001")


def _q(v: Decimal) -> Decimal:
    return v.quantize(_QUANT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton before each test."""
    from greenlang.employee_commuting.telework_calculator import (
        TeleworkCalculatorEngine,
    )
    TeleworkCalculatorEngine.reset_instance()
    yield
    TeleworkCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh TeleworkCalculatorEngine instance."""
    from greenlang.employee_commuting.telework_calculator import (
        TeleworkCalculatorEngine,
    )
    return TeleworkCalculatorEngine.get_instance()


# ===========================================================================
# 1. SINGLETON AND PROPERTIES
# ===========================================================================

class TestSingletonTelework:
    """Singleton and property tests."""

    def test_singleton_identity(self, engine):
        """Two get_instance calls return the same object."""
        from greenlang.employee_commuting.telework_calculator import (
            TeleworkCalculatorEngine,
        )
        assert engine is TeleworkCalculatorEngine.get_instance()

    def test_engine_id(self, engine):
        """Engine ID is correct."""
        assert engine.engine_id == "telework_calculator_engine"

    def test_engine_version(self, engine):
        """Engine version is 1.0.0."""
        assert engine.engine_version == "1.0.0"

    def test_initial_count(self, engine):
        """Calculation count starts at zero."""
        assert engine.calculation_count == 0


# ===========================================================================
# 2. FULL REMOTE CALCULATION
# ===========================================================================

class TestFullRemote:
    """Tests for full remote telework calculation."""

    def test_full_remote_positive_emissions(self, engine):
        """Full remote worker produces positive telework emissions."""
        result = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="temperate",
        )
        assert result["total_co2e_kg"] > Decimal("0")
        assert result["telework_category"] == "full_remote"

    def test_full_remote_all_components_present(self, engine):
        """Result includes electricity, heating, cooling, equipment components."""
        result = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="temperate",
        )
        assert "electricity_co2e_kg" in result
        assert "heating_co2e_kg" in result
        assert "cooling_co2e_kg" in result
        assert "equipment_co2e_kg" in result

    def test_full_remote_annual_wfh_days(self, engine):
        """Full remote = 5/5 * 240 = 240 WFH days."""
        result = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="temperate",
            working_days=240,
        )
        assert result["annual_wfh_days"] == _q(Decimal("240"))

    def test_full_remote_convenience_method(self, engine):
        """calculate_full_remote is equivalent to full_remote category."""
        direct = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="temperate",
            working_days=240,
        )
        convenience = engine.calculate_full_remote(
            country_code="US",
            climate_zone="temperate",
            working_days=240,
        )
        assert direct["total_co2e_kg"] == convenience["total_co2e_kg"]


# ===========================================================================
# 3. HYBRID PATTERNS
# ===========================================================================

class TestHybridPatterns:
    """Tests for hybrid work patterns (1-4 days)."""

    @pytest.mark.parametrize("category,wfh_days", [
        ("hybrid_4day", 4),
        ("hybrid_3day", 3),
        ("hybrid_2day", 2),
        ("hybrid_1day", 1),
    ])
    def test_hybrid_wfh_days(self, engine, category, wfh_days):
        """Hybrid category maps to correct WFH days per week."""
        result = engine.calculate_telework_emissions(
            telework_category=category,
            country_code="US",
            climate_zone="temperate",
            working_days=250,
        )
        expected_annual = _q(Decimal(str(wfh_days)) / Decimal("5") * Decimal("250"))
        assert result["annual_wfh_days"] == expected_annual

    def test_hybrid_3_less_than_full_remote(self, engine):
        """Hybrid 3-day has lower emissions than full remote."""
        full = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="temperate",
        )
        hybrid3 = engine.calculate_telework_emissions(
            telework_category="hybrid_3day",
            country_code="US",
            climate_zone="temperate",
        )
        assert hybrid3["total_co2e_kg"] < full["total_co2e_kg"]

    def test_more_wfh_days_means_more_emissions(self, engine):
        """More WFH days means higher telework emissions."""
        h1 = engine.calculate_telework_emissions(
            telework_category="hybrid_1day",
            country_code="US",
        )
        h4 = engine.calculate_telework_emissions(
            telework_category="hybrid_4day",
            country_code="US",
        )
        assert h4["total_co2e_kg"] > h1["total_co2e_kg"]


# ===========================================================================
# 4. OFFICE-BASED (ZERO TELEWORK)
# ===========================================================================

class TestOfficeBased:
    """Tests for office-based (zero telework)."""

    def test_office_based_zero_emissions(self, engine):
        """Office-based worker has zero telework emissions."""
        result = engine.calculate_telework_emissions(
            telework_category="office_based",
            country_code="US",
        )
        assert result["total_co2e_kg"] == Decimal("0") or result["total_co2e_kg"] == _q(Decimal("0"))


# ===========================================================================
# 5. GRID FACTOR RESOLUTION
# ===========================================================================

class TestGridFactorResolution:
    """Tests for grid emission factor resolution."""

    @pytest.mark.parametrize("country,expected_grid", [
        ("US", Decimal("0.37938")),
        ("GB", Decimal("0.20705")),
        ("NO", Decimal("0.00760")),
        ("SE", Decimal("0.00830")),
        ("IN", Decimal("0.70767")),
    ])
    def test_country_grid_factor_used(self, engine, country, expected_grid):
        """Correct grid factor is applied for each country."""
        result = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code=country,
            climate_zone="temperate",
        )
        assert result["grid_factor_used"] == expected_grid

    def test_egrid_subregion_overrides_country(self, engine):
        """eGRID subregion overrides national grid factor."""
        from greenlang.employee_commuting.telework_calculator import US_EGRID_FACTORS
        result = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            egrid_subregion="RMPA",
            climate_zone="temperate",
        )
        assert result["grid_factor_used"] == US_EGRID_FACTORS["RMPA"]

    def test_low_grid_country_lower_emissions(self, engine):
        """Norway (low grid) produces far less than South Africa (high grid)."""
        no = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="NO",
            climate_zone="temperate",
        )
        za = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="ZA",
            climate_zone="temperate",
        )
        assert no["electricity_co2e_kg"] < za["electricity_co2e_kg"]


# ===========================================================================
# 6. CLIMATE ZONES AND SEASONAL ADJUSTMENT
# ===========================================================================

class TestClimateZones:
    """Tests for climate zone and seasonal effects."""

    def test_polar_more_heating_than_tropical(self, engine):
        """Polar zone has much higher heating emissions than tropical."""
        polar = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="polar",
        )
        tropical = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="tropical",
        )
        assert polar["heating_co2e_kg"] > tropical["heating_co2e_kg"]

    def test_tropical_more_cooling_than_polar(self, engine):
        """Tropical zone has higher cooling emissions than polar."""
        polar = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="polar",
        )
        tropical = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="tropical",
        )
        assert tropical["cooling_co2e_kg"] > polar["cooling_co2e_kg"]


# ===========================================================================
# 7. EQUIPMENT LIFECYCLE EMISSIONS
# ===========================================================================

class TestEquipmentEmissions:
    """Tests for equipment lifecycle emissions."""

    def test_equipment_included_increases_total(self, engine):
        """Including equipment emissions increases the total."""
        without = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="temperate",
            include_equipment=False,
        )
        with_equip = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            climate_zone="temperate",
            include_equipment=True,
        )
        assert with_equip["total_co2e_kg"] > without["total_co2e_kg"]
        assert with_equip["equipment_co2e_kg"] > Decimal("0")

    def test_equipment_excluded_is_zero(self, engine):
        """Excluding equipment gives zero equipment emissions."""
        result = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
            include_equipment=False,
        )
        assert result["equipment_co2e_kg"] == _q(Decimal("0"))


# ===========================================================================
# 8. HYBRID WORKER NET IMPACT
# ===========================================================================

class TestHybridWorkerNet:
    """Tests for calculate_hybrid_worker with net impact."""

    def test_hybrid_worker_basic(self, engine):
        """Hybrid worker with 3 office days produces telework emissions."""
        result = engine.calculate_hybrid_worker(
            office_days_per_week=3,
            country_code="US",
        )
        assert result["gross_telework_co2e_kg"] > Decimal("0")
        assert result["wfh_days_per_week"] == 2

    def test_hybrid_worker_with_avoided_commute(self, engine):
        """Net impact subtracts avoided commute from telework emissions."""
        result = engine.calculate_hybrid_worker(
            office_days_per_week=2,
            country_code="US",
            commute_avoided_co2e=Decimal("500.0"),
        )
        assert "net_co2e_kg" in result
        assert result["net_co2e_kg"] == _q(
            result["gross_telework_co2e_kg"] - Decimal("500.0")
        )

    def test_hybrid_worker_carbon_saving_flag(self, engine):
        """When avoided commute > telework, is_carbon_saving = True."""
        result = engine.calculate_hybrid_worker(
            office_days_per_week=3,
            country_code="US",
            commute_avoided_co2e=Decimal("5000.0"),
        )
        assert result["is_carbon_saving"] is True
        assert result["net_co2e_kg"] < Decimal("0")

    def test_invalid_office_days_raises(self, engine):
        """Office days > 5 raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_hybrid_worker(office_days_per_week=6)

    def test_invalid_office_days_negative_raises(self, engine):
        """Negative office days raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_hybrid_worker(office_days_per_week=-1)


# ===========================================================================
# 9. INPUT VALIDATION
# ===========================================================================

class TestTeleworkInputValidation:
    """Tests for telework input validation."""

    def test_unknown_category_raises(self, engine):
        """Unknown telework category raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_telework_emissions(
                telework_category="quarterly_remote",
                country_code="US",
            )

    def test_unknown_climate_zone_raises(self, engine):
        """Unknown climate zone raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_telework_emissions(
                telework_category="full_remote",
                climate_zone="martian",
            )

    def test_unknown_heating_fuel_raises(self, engine):
        """Unknown heating fuel raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_telework_emissions(
                telework_category="full_remote",
                heating_fuel="nuclear",
            )


# ===========================================================================
# 10. PROVENANCE AND REPRODUCIBILITY
# ===========================================================================

class TestTeleworkProvenance:
    """Provenance and reproducibility tests."""

    def test_same_inputs_same_hash(self, engine):
        """Identical inputs produce the same provenance hash."""
        kwargs = dict(
            telework_category="full_remote",
            country_code="US",
            climate_zone="temperate",
            working_days=240,
        )
        r1 = engine.calculate_telework_emissions(**kwargs)
        r2 = engine.calculate_telework_emissions(**kwargs)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_provenance_hash_is_64_chars(self, engine):
        """Provenance hash is a 64-character SHA-256 hex string."""
        result = engine.calculate_telework_emissions(
            telework_category="full_remote",
            country_code="US",
        )
        assert len(result["provenance_hash"]) == 64

    def test_calculation_count_increments(self, engine):
        """Each successful calculation increments the counter."""
        engine.calculate_telework_emissions(
            telework_category="full_remote", country_code="US",
        )
        engine.calculate_telework_emissions(
            telework_category="hybrid_3day", country_code="GB",
        )
        assert engine.calculation_count == 2
