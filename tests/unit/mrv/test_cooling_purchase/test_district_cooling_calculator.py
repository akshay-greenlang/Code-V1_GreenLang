# -*- coding: utf-8 -*-
"""
Unit tests for DistrictCoolingCalculatorEngine (Engine 4 of 7)

AGENT-MRV-012: Cooling Purchase Agent

Tests district cooling network calculations, free cooling (seawater/lake/river/air),
thermal energy storage (TES), and multi-source plant emissions with distribution
losses, pump energy, and temporal emission shifting capabilities.

Target: 70 tests, ~600 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.agents.mrv.cooling_purchase.district_cooling_calculator import (
    DistrictCoolingCalculatorEngine,
    get_district_cooling_calculator,
)
from greenlang.agents.mrv.cooling_purchase.models import (
    FreeCoolingSource,
    TESType,
    DataQualityTier,
    GWPSource,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dcc_engine():
    """Create a DistrictCoolingCalculatorEngine instance."""
    engine = DistrictCoolingCalculatorEngine()
    yield engine
    engine.reset()


@pytest.fixture
def district_request_dict() -> Dict[str, Any]:
    """Return a standard district cooling request dictionary."""
    return {
        "cooling_kwh_th": Decimal("100000"),
        "region": "singapore",
        "distribution_loss_pct": Decimal("5.0"),
        "pump_energy_kwh": Decimal("2000"),
        "cop_plant": Decimal("4.5"),
        "grid_ef_kgco2e_kwh": Decimal("0.45"),
        "tier": "TIER_1",
        "gwp_source": "AR6",
    }


@pytest.fixture
def free_cooling_dict() -> Dict[str, Any]:
    """Return a free cooling request dictionary."""
    return {
        "cooling_kwh_th": Decimal("50000"),
        "source": "seawater",
        "cop_override": None,
        "grid_ef_kgco2e_kwh": Decimal("0.40"),
        "tier": "TIER_2",
        "gwp_source": "AR6",
    }


@pytest.fixture
def tes_request_dict() -> Dict[str, Any]:
    """Return a TES request dictionary."""
    return {
        "cooling_kwh_th": Decimal("80000"),
        "tes_type": "ice",
        "capacity_kwh_th": Decimal("20000"),
        "cop_charge": Decimal("3.0"),
        "grid_ef_charge_kgco2e_kwh": Decimal("0.30"),
        "grid_ef_peak_kgco2e_kwh": Decimal("0.60"),
        "cop_peak": Decimal("4.0"),
        "tier": "TIER_2",
        "gwp_source": "AR6",
    }


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestDistrictCoolingEngineInit:
    """Test DistrictCoolingCalculatorEngine initialization."""

    def test_singleton_pattern(self):
        """Test singleton returns same instance."""
        e1 = DistrictCoolingCalculatorEngine()
        e2 = DistrictCoolingCalculatorEngine()
        assert e1 is e2

    def test_get_function_returns_singleton(self):
        """Test get_district_cooling_calculator returns singleton."""
        e1 = get_district_cooling_calculator()
        e2 = get_district_cooling_calculator()
        assert e1 is e2

    def test_reset_clears_state(self, dcc_engine):
        """Test reset clears internal state."""
        _ = dcc_engine.calculate_district_network(Decimal("10000"), "singapore")
        stats_before = dcc_engine.get_statistics()
        dcc_engine.reset()
        stats_after = dcc_engine.get_statistics()
        assert stats_after["total_calculations"] == 0
        assert stats_before["total_calculations"] > 0

    def test_statistics_starts_at_zero(self, dcc_engine):
        """Test statistics counter starts at zero."""
        stats = dcc_engine.get_statistics()
        assert stats["total_calculations"] == 0


# ===========================================================================
# 2. District Network Calculation Tests
# ===========================================================================


class TestDistrictNetworkCalculation:
    """Test calculate_district_network method."""

    def test_basic_district_cooling(self, dcc_engine):
        """Test basic district cooling calculation."""
        result = dcc_engine.calculate_district_network(
            cooling_kwh_th=Decimal("100000"),
            region="singapore",
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.technology == "DISTRICT_COOLING"
        assert len(result.provenance_hash) == 64

    def test_adjusted_cooling_calculation(self, dcc_engine):
        """Test adjusted_cooling = cooling / (1 - loss)."""
        result = dcc_engine.calculate_district_network(
            cooling_kwh_th=Decimal("100000"),
            region="singapore",
            distribution_loss_pct=Decimal("10.0"),
        )
        # Adjusted = 100000 / (1 - 0.10) = 111111.11
        assert result.emissions_kgco2e > Decimal("0")

    def test_distribution_loss(self, dcc_engine):
        """Test calculate_distribution_loss = cooling * loss_pct."""
        loss = dcc_engine.calculate_distribution_loss(
            Decimal("100000"), Decimal("5.0")
        )
        assert loss == Decimal("5000")

    def test_adjusted_cooling_method(self, dcc_engine):
        """Test calculate_adjusted_cooling = cooling / (1 - loss_pct)."""
        adjusted = dcc_engine.calculate_adjusted_cooling(
            Decimal("100000"), Decimal("5.0")
        )
        expected = Decimal("100000") / (Decimal("1") - Decimal("0.05"))
        assert abs(adjusted - expected) < Decimal("0.01")

    def test_pump_energy_calculation(self, dcc_engine):
        """Test calculate_pump_energy method."""
        pump_emissions = dcc_engine.calculate_pump_energy(
            pump_kwh=Decimal("2000"),
            grid_ef=Decimal("0.45"),
        )
        expected = Decimal("2000") * Decimal("0.45")
        assert pump_emissions == expected

    def test_emissions_formula(self, dcc_engine):
        """Test emissions = adjusted/COP*ef + pump*grid_ef."""
        result = dcc_engine.calculate_district_network(
            cooling_kwh_th=Decimal("100000"),
            region="singapore",
            distribution_loss_pct=Decimal("5.0"),
            pump_energy_kwh=Decimal("2000"),
            cop_plant=Decimal("4.5"),
            grid_ef_kgco2e_kwh=Decimal("0.45"),
        )
        # Adjusted = 100000 / 0.95 = 105263.16
        # Gen = 105263.16 / 4.5 * 0.45 = 10526.32
        # Pump = 2000 * 0.45 = 900
        # Total ~11426.32
        assert result.emissions_kgco2e > Decimal("11000")
        assert result.emissions_kgco2e < Decimal("12000")

    def test_provenance_hash_present(self, dcc_engine):
        """Test result has provenance_hash."""
        result = dcc_engine.calculate_district_network(
            Decimal("10000"), "singapore"
        )
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 3. Regional Emission Factor Tests
# ===========================================================================


class TestRegionalEmissionFactors:
    """Test get_regional_ef for all 12 regions."""

    def test_singapore_ef(self, dcc_engine):
        """Test Singapore regional EF."""
        ef = dcc_engine.get_regional_ef("singapore")
        assert ef > Decimal("0")

    def test_dubai_ef(self, dcc_engine):
        """Test Dubai regional EF."""
        ef = dcc_engine.get_regional_ef("dubai")
        assert ef > Decimal("0")

    def test_doha_ef(self, dcc_engine):
        """Test Doha regional EF."""
        ef = dcc_engine.get_regional_ef("doha")
        assert ef > Decimal("0")

    def test_hong_kong_ef(self, dcc_engine):
        """Test Hong Kong regional EF."""
        ef = dcc_engine.get_regional_ef("hong_kong")
        assert ef > Decimal("0")

    def test_abu_dhabi_ef(self, dcc_engine):
        """Test Abu Dhabi regional EF."""
        ef = dcc_engine.get_regional_ef("abu_dhabi")
        assert ef > Decimal("0")

    def test_paris_ef(self, dcc_engine):
        """Test Paris regional EF."""
        ef = dcc_engine.get_regional_ef("paris")
        assert ef > Decimal("0")

    def test_copenhagen_ef(self, dcc_engine):
        """Test Copenhagen regional EF."""
        ef = dcc_engine.get_regional_ef("copenhagen")
        assert ef > Decimal("0")

    def test_stockholm_ef(self, dcc_engine):
        """Test Stockholm regional EF."""
        ef = dcc_engine.get_regional_ef("stockholm")
        assert ef > Decimal("0")

    def test_toronto_ef(self, dcc_engine):
        """Test Toronto regional EF."""
        ef = dcc_engine.get_regional_ef("toronto")
        assert ef > Decimal("0")

    def test_seoul_ef(self, dcc_engine):
        """Test Seoul regional EF."""
        ef = dcc_engine.get_regional_ef("seoul")
        assert ef > Decimal("0")

    def test_riyadh_ef(self, dcc_engine):
        """Test Riyadh regional EF."""
        ef = dcc_engine.get_regional_ef("riyadh")
        assert ef > Decimal("0")

    def test_kuala_lumpur_ef(self, dcc_engine):
        """Test Kuala Lumpur regional EF."""
        ef = dcc_engine.get_regional_ef("kuala_lumpur")
        assert ef > Decimal("0")

    def test_invalid_region_returns_default(self, dcc_engine):
        """Test invalid region returns default EF."""
        ef = dcc_engine.get_regional_ef("unknown_city")
        assert ef > Decimal("0")  # Should return default


# ===========================================================================
# 4. Free Cooling Tests
# ===========================================================================


class TestFreeCooling:
    """Test free cooling calculations for natural heat sinks."""

    def test_seawater_free_cooling(self, dcc_engine):
        """Test seawater free cooling calculation."""
        result = dcc_engine.calculate_free_cooling(
            cooling_kwh_th=Decimal("50000"),
            source=FreeCoolingSource.SEAWATER_FREE,
            grid_ef_kgco2e_kwh=Decimal("0.40"),
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.technology == "SEAWATER_FREE"

    def test_lake_free_cooling(self, dcc_engine):
        """Test lake water free cooling."""
        result = dcc_engine.calculate_free_cooling(
            cooling_kwh_th=Decimal("50000"),
            source=FreeCoolingSource.LAKE_FREE,
            grid_ef_kgco2e_kwh=Decimal("0.40"),
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.technology == "LAKE_FREE"

    def test_river_free_cooling(self, dcc_engine):
        """Test river water free cooling."""
        result = dcc_engine.calculate_free_cooling(
            cooling_kwh_th=Decimal("50000"),
            source=FreeCoolingSource.RIVER_FREE,
            grid_ef_kgco2e_kwh=Decimal("0.40"),
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.technology == "RIVER_FREE"

    def test_ambient_air_free_cooling(self, dcc_engine):
        """Test ambient air free cooling."""
        result = dcc_engine.calculate_free_cooling(
            cooling_kwh_th=Decimal("50000"),
            source=FreeCoolingSource.AMBIENT_AIR_FREE,
            grid_ef_kgco2e_kwh=Decimal("0.40"),
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.technology == "AMBIENT_AIR_FREE"

    def test_free_cooling_low_emissions(self, dcc_engine):
        """Test free cooling has very low emissions due to high COP."""
        result = dcc_engine.calculate_free_cooling(
            cooling_kwh_th=Decimal("100000"),
            source=FreeCoolingSource.SEAWATER_FREE,
            grid_ef_kgco2e_kwh=Decimal("0.45"),
        )
        # With COP ~20, emissions = 100000/20*0.45 = 2250 kgCO2e
        assert result.emissions_kgco2e < Decimal("5000")

    def test_get_free_cooling_cop_seawater(self, dcc_engine):
        """Test default COP for seawater is 20."""
        cop = dcc_engine.get_free_cooling_cop(FreeCoolingSource.SEAWATER_FREE)
        assert cop == Decimal("20")

    def test_get_free_cooling_cop_lake(self, dcc_engine):
        """Test default COP for lake is 18."""
        cop = dcc_engine.get_free_cooling_cop(FreeCoolingSource.LAKE_FREE)
        assert cop == Decimal("18")

    def test_get_free_cooling_cop_river(self, dcc_engine):
        """Test default COP for river is 15."""
        cop = dcc_engine.get_free_cooling_cop(FreeCoolingSource.RIVER_FREE)
        assert cop == Decimal("15")

    def test_get_free_cooling_cop_air(self, dcc_engine):
        """Test default COP for air is 10."""
        cop = dcc_engine.get_free_cooling_cop(FreeCoolingSource.AMBIENT_AIR_FREE)
        assert cop == Decimal("10")


# ===========================================================================
# 5. Thermal Energy Storage (TES) Tests
# ===========================================================================


class TestThermalEnergyStorage:
    """Test TES calculation methods."""

    def test_ice_tes_calculation(self, dcc_engine):
        """Test ice TES calculation."""
        result = dcc_engine.calculate_tes(
            cooling_kwh_th=Decimal("80000"),
            tes_type=TESType.ICE_TES,
            capacity_kwh_th=Decimal("20000"),
            cop_charge=Decimal("3.0"),
            grid_ef_charge_kgco2e_kwh=Decimal("0.30"),
            grid_ef_peak_kgco2e_kwh=Decimal("0.60"),
            cop_peak=Decimal("4.0"),
        )
        assert result.emissions_kgco2e > Decimal("0")
        assert result.tes_savings_kgco2e is not None

    def test_chilled_water_tes(self, dcc_engine):
        """Test chilled water TES calculation."""
        result = dcc_engine.calculate_tes(
            cooling_kwh_th=Decimal("80000"),
            tes_type=TESType.CHILLED_WATER_TES,
            capacity_kwh_th=Decimal("20000"),
            cop_charge=Decimal("5.0"),
            grid_ef_charge_kgco2e_kwh=Decimal("0.30"),
            grid_ef_peak_kgco2e_kwh=Decimal("0.60"),
        )
        assert result.emissions_kgco2e > Decimal("0")

    def test_pcm_tes(self, dcc_engine):
        """Test PCM TES calculation."""
        result = dcc_engine.calculate_tes(
            cooling_kwh_th=Decimal("80000"),
            tes_type=TESType.PCM_TES,
            capacity_kwh_th=Decimal("20000"),
            cop_charge=Decimal("4.0"),
            grid_ef_charge_kgco2e_kwh=Decimal("0.30"),
            grid_ef_peak_kgco2e_kwh=Decimal("0.60"),
        )
        assert result.emissions_kgco2e > Decimal("0")

    def test_tes_charge_energy_calculation(self, dcc_engine):
        """Test calculate_tes_charge_energy = capacity/(cop*rte)."""
        charge_energy = dcc_engine.calculate_tes_charge_energy(
            capacity_kwh_th=Decimal("20000"),
            cop_charge=Decimal("3.0"),
            rte=Decimal("0.85"),
        )
        expected = Decimal("20000") / (Decimal("3.0") * Decimal("0.85"))
        assert abs(charge_energy - expected) < Decimal("0.01")

    def test_tes_savings_calculation(self, dcc_engine):
        """Test calculate_tes_savings = peak_emissions - tes_emissions."""
        savings = dcc_engine.calculate_tes_savings(
            peak_emissions_kgco2e=Decimal("12000"),
            tes_emissions_kgco2e=Decimal("8000"),
        )
        assert savings == Decimal("4000")

    def test_get_round_trip_efficiency_ice(self, dcc_engine):
        """Test round-trip efficiency for ice is 0.85."""
        rte = dcc_engine.get_round_trip_efficiency(TESType.ICE_TES)
        assert rte == Decimal("0.85")

    def test_get_round_trip_efficiency_chilled_water(self, dcc_engine):
        """Test round-trip efficiency for chilled water is 0.95."""
        rte = dcc_engine.get_round_trip_efficiency(TESType.CHILLED_WATER_TES)
        assert rte == Decimal("0.95")

    def test_get_round_trip_efficiency_pcm(self, dcc_engine):
        """Test round-trip efficiency for PCM is 0.90."""
        rte = dcc_engine.get_round_trip_efficiency(TESType.PCM_TES)
        assert rte == Decimal("0.90")


# ===========================================================================
# 6. Multi-Source Plant Tests
# ===========================================================================


class TestMultiSourcePlant:
    """Test multi-source district cooling plant calculations."""

    def test_multi_source_two_plants(self, dcc_engine):
        """Test multi-source plant with two sources."""
        sources = [
            {
                "fraction": Decimal("0.6"),
                "emissions_kgco2e": Decimal("10000"),
            },
            {
                "fraction": Decimal("0.4"),
                "emissions_kgco2e": Decimal("5000"),
            },
        ]
        total = dcc_engine.calculate_multi_source_plant(sources)
        expected = Decimal("0.6") * Decimal("10000") + Decimal("0.4") * Decimal("5000")
        assert total == expected

    def test_multi_source_three_plants(self, dcc_engine):
        """Test multi-source plant with three sources."""
        sources = [
            {"fraction": Decimal("0.5"), "emissions_kgco2e": Decimal("8000")},
            {"fraction": Decimal("0.3"), "emissions_kgco2e": Decimal("6000")},
            {"fraction": Decimal("0.2"), "emissions_kgco2e": Decimal("4000")},
        ]
        total = dcc_engine.calculate_multi_source_plant(sources)
        expected = (
            Decimal("0.5") * Decimal("8000")
            + Decimal("0.3") * Decimal("6000")
            + Decimal("0.2") * Decimal("4000")
        )
        assert total == expected


# ===========================================================================
# 7. Provenance and Trace Tests
# ===========================================================================


class TestProvenanceAndTrace:
    """Test provenance hashing and calculation traces."""

    def test_district_cooling_has_provenance(self, dcc_engine):
        """Test district cooling result has provenance hash."""
        result = dcc_engine.calculate_district_network(
            Decimal("100000"), "singapore"
        )
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_free_cooling_has_provenance(self, dcc_engine):
        """Test free cooling result has provenance hash."""
        result = dcc_engine.calculate_free_cooling(
            Decimal("50000"), FreeCoolingSource.SEAWATER_FREE, Decimal("0.40")
        )
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_tes_has_provenance(self, dcc_engine):
        """Test TES result has provenance hash."""
        result = dcc_engine.calculate_tes(
            cooling_kwh_th=Decimal("80000"),
            tes_type=TESType.ICE_TES,
            capacity_kwh_th=Decimal("20000"),
            cop_charge=Decimal("3.0"),
            grid_ef_charge_kgco2e_kwh=Decimal("0.30"),
            grid_ef_peak_kgco2e_kwh=Decimal("0.60"),
        )
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64
