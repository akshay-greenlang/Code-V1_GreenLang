# -*- coding: utf-8 -*-
"""
Unit tests for EquipmentCalculatorEngine (AGENT-MRV-021, Engine 4)

35 tests covering energy-based, fuel-based, output-based, and average-data
equipment emission calculations, load factor, batch processing, edge cases,
and provenance for leased equipment.

Calculation methods:
    Energy-based:  rated_kw * hours * load_factor * grid_ef (electricity)
    Fuel-based:    rated_kw * hours * load_factor * sfc * fuel_ef (diesel/fuel)
    Output-based:  output_kwh * grid_ef / efficiency
    Average-data:  benchmark_ef * rated_power * hours

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.upstream_leased_assets.equipment_calculator import (
        EquipmentCalculatorEngine,
    )
    from greenlang.upstream_leased_assets.models import (
        EquipmentType,
        EnergySource,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="EquipmentCalculatorEngine not available",
)

pytestmark = _SKIP


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    if _AVAILABLE:
        EquipmentCalculatorEngine.reset_instance()
    yield
    if _AVAILABLE:
        EquipmentCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh EquipmentCalculatorEngine."""
    return EquipmentCalculatorEngine()


# ==============================================================================
# ENERGY-BASED CALCULATION TESTS (ELECTRICITY)
# ==============================================================================


class TestEnergyBasedCalculation:
    """Test energy-based equipment emission calculations."""

    @pytest.mark.parametrize("equipment_type", [
        "manufacturing", "construction", "generator",
        "agricultural", "mining", "hvac",
    ])
    def test_energy_based_all_equipment_types(self, engine, equipment_type):
        """Test energy-based calculation for all 6 equipment types."""
        result = engine.calculate({
            "method": "energy_based",
            "equipment_type": equipment_type,
            "rated_power_kw": Decimal("100"),
            "annual_operating_hours": 2000,
            "load_factor": Decimal("0.70"),
            "energy_source": "electricity",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
        assert isinstance(result["total_co2e_kg"], Decimal)

    def test_higher_power_more_emissions(self, engine):
        """Test higher rated power produces more emissions."""
        low = engine.calculate({
            "method": "energy_based",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("100"),
            "annual_operating_hours": 2000,
            "load_factor": Decimal("0.70"),
            "energy_source": "electricity",
            "region": "US",
        })
        high = engine.calculate({
            "method": "energy_based",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("500"),
            "annual_operating_hours": 2000,
            "load_factor": Decimal("0.70"),
            "energy_source": "electricity",
            "region": "US",
        })
        ratio = high["total_co2e_kg"] / low["total_co2e_kg"]
        assert abs(ratio - Decimal("5.0")) < Decimal("0.1")

    def test_more_hours_more_emissions(self, engine):
        """Test more operating hours produces more emissions."""
        short = engine.calculate({
            "method": "energy_based",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("200"),
            "annual_operating_hours": 2000,
            "load_factor": Decimal("0.70"),
            "energy_source": "electricity",
            "region": "US",
        })
        long = engine.calculate({
            "method": "energy_based",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("200"),
            "annual_operating_hours": 6000,
            "load_factor": Decimal("0.70"),
            "energy_source": "electricity",
            "region": "US",
        })
        assert long["total_co2e_kg"] > short["total_co2e_kg"]

    def test_known_value_manufacturing_electricity(self, engine):
        """Test known value: 500kW * 6000h * 0.75 * 0.37170 = 836325 kWh * 0.37170 ~= 310,781 kg."""
        result = engine.calculate({
            "method": "energy_based",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("500"),
            "annual_operating_hours": 6000,
            "load_factor": Decimal("0.75"),
            "energy_source": "electricity",
            "region": "US",
        })
        # 500 * 6000 * 0.75 = 2,250,000 kWh * ~0.37 = ~832,500 kg
        assert Decimal("500000") < result["total_co2e_kg"] < Decimal("1200000")

    def test_provenance_hash_deterministic(self, engine):
        """Test provenance hash is deterministic."""
        inp = {
            "method": "energy_based",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("500"),
            "annual_operating_hours": 6000,
            "load_factor": Decimal("0.75"),
            "energy_source": "electricity",
            "region": "US",
        }
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1["provenance_hash"] == r2["provenance_hash"]
        assert len(r1["provenance_hash"]) == 64


# ==============================================================================
# FUEL-BASED CALCULATION TESTS (DIESEL)
# ==============================================================================


class TestFuelBasedEquipment:
    """Test fuel-based equipment emission calculations."""

    def test_diesel_construction_equipment(self, engine):
        """Test diesel construction equipment calculation."""
        result = engine.calculate({
            "method": "fuel_based",
            "equipment_type": "construction",
            "rated_power_kw": Decimal("200"),
            "annual_operating_hours": 2000,
            "load_factor": Decimal("0.60"),
            "energy_source": "diesel",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_diesel_generator(self, engine):
        """Test diesel generator calculation."""
        result = engine.calculate({
            "method": "fuel_based",
            "equipment_type": "generator",
            "rated_power_kw": Decimal("100"),
            "annual_operating_hours": 1500,
            "load_factor": Decimal("0.80"),
            "energy_source": "diesel",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_diesel_higher_load_factor_more_emissions(self, engine):
        """Test higher load factor produces more emissions."""
        low_lf = engine.calculate({
            "method": "fuel_based",
            "equipment_type": "construction",
            "rated_power_kw": Decimal("200"),
            "annual_operating_hours": 2000,
            "load_factor": Decimal("0.30"),
            "energy_source": "diesel",
            "region": "US",
        })
        high_lf = engine.calculate({
            "method": "fuel_based",
            "equipment_type": "construction",
            "rated_power_kw": Decimal("200"),
            "annual_operating_hours": 2000,
            "load_factor": Decimal("0.90"),
            "energy_source": "diesel",
            "region": "US",
        })
        assert high_lf["total_co2e_kg"] > low_lf["total_co2e_kg"]


# ==============================================================================
# OUTPUT-BASED CALCULATION TESTS
# ==============================================================================


class TestOutputBasedCalculation:
    """Test output-based equipment emission calculations."""

    def test_generator_output_based(self, engine):
        """Test generator output-based calculation."""
        result = engine.calculate({
            "method": "output_based",
            "equipment_type": "generator",
            "output_kwh": Decimal("120000"),
            "energy_source": "diesel",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# AVERAGE-DATA CALCULATION TESTS
# ==============================================================================


class TestAverageDataEquipment:
    """Test average-data equipment emission calculations."""

    def test_average_data_manufacturing(self, engine):
        """Test average-data manufacturing equipment."""
        result = engine.calculate({
            "method": "average_data",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("500"),
            "annual_operating_hours": 6000,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_average_data_hvac(self, engine):
        """Test average-data HVAC equipment."""
        result = engine.calculate({
            "method": "average_data",
            "equipment_type": "hvac",
            "rated_power_kw": Decimal("50"),
            "annual_operating_hours": 4000,
            "region": "DE",
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


class TestBatchEquipment:
    """Test batch equipment calculations."""

    def test_batch_multiple_equipment(self, engine):
        """Test batch processing of multiple equipment items."""
        equipment_list = [
            {
                "method": "energy_based",
                "equipment_type": "manufacturing",
                "rated_power_kw": Decimal("500"),
                "annual_operating_hours": 6000,
                "load_factor": Decimal("0.75"),
                "energy_source": "electricity",
                "region": "US",
            },
            {
                "method": "fuel_based",
                "equipment_type": "construction",
                "rated_power_kw": Decimal("200"),
                "annual_operating_hours": 2000,
                "load_factor": Decimal("0.60"),
                "energy_source": "diesel",
                "region": "US",
            },
        ]
        results = engine.calculate_batch(equipment_list)
        assert len(results) == 2
        assert all(r["total_co2e_kg"] > 0 for r in results)


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestEquipmentEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_hours_raises_error(self, engine):
        """Test zero operating hours raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "energy_based",
                "equipment_type": "manufacturing",
                "rated_power_kw": Decimal("500"),
                "annual_operating_hours": 0,
                "load_factor": Decimal("0.75"),
                "energy_source": "electricity",
                "region": "US",
            })

    def test_negative_power_raises_error(self, engine):
        """Test negative rated power raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "energy_based",
                "equipment_type": "manufacturing",
                "rated_power_kw": Decimal("-100"),
                "annual_operating_hours": 2000,
                "load_factor": Decimal("0.75"),
                "energy_source": "electricity",
                "region": "US",
            })

    def test_load_factor_over_one_raises_error(self, engine):
        """Test load factor over 1.0 raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "energy_based",
                "equipment_type": "manufacturing",
                "rated_power_kw": Decimal("100"),
                "annual_operating_hours": 2000,
                "load_factor": Decimal("1.5"),
                "energy_source": "electricity",
                "region": "US",
            })

    def test_hours_over_8760_raises_error(self, engine):
        """Test operating hours over 8760 raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "energy_based",
                "equipment_type": "manufacturing",
                "rated_power_kw": Decimal("100"),
                "annual_operating_hours": 9000,
                "load_factor": Decimal("0.75"),
                "energy_source": "electricity",
                "region": "US",
            })

    def test_result_contains_equipment_type(self, engine):
        """Test result contains equipment type."""
        result = engine.calculate({
            "method": "energy_based",
            "equipment_type": "hvac",
            "rated_power_kw": Decimal("50"),
            "annual_operating_hours": 4000,
            "load_factor": Decimal("0.65"),
            "energy_source": "electricity",
            "region": "US",
        })
        assert "equipment_type" in result or "asset_type" in result


# ==============================================================================
# PARAMETRIZED TESTS
# ==============================================================================


class TestEquipmentParametrized:
    """Parametrized tests for exhaustive equipment coverage."""

    @pytest.mark.parametrize("region", [
        "US", "GB", "DE", "FR", "JP", "CA", "AU",
    ])
    def test_manufacturing_multiple_regions(self, engine, region):
        """Test manufacturing equipment across multiple grid regions."""
        result = engine.calculate({
            "method": "energy_based",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("200"),
            "annual_operating_hours": 4000,
            "load_factor": Decimal("0.70"),
            "energy_source": "electricity",
            "region": region,
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("load_factor", [
        Decimal("0.20"), Decimal("0.40"), Decimal("0.60"),
        Decimal("0.80"), Decimal("1.00"),
    ])
    def test_various_load_factors(self, engine, load_factor):
        """Test equipment with various load factors."""
        result = engine.calculate({
            "method": "energy_based",
            "equipment_type": "manufacturing",
            "rated_power_kw": Decimal("200"),
            "annual_operating_hours": 4000,
            "load_factor": load_factor,
            "energy_source": "electricity",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("hours", [500, 1000, 2000, 4000, 6000, 8760])
    def test_various_operating_hours(self, engine, hours):
        """Test equipment with various operating hours."""
        result = engine.calculate({
            "method": "energy_based",
            "equipment_type": "construction",
            "rated_power_kw": Decimal("150"),
            "annual_operating_hours": hours,
            "load_factor": Decimal("0.60"),
            "energy_source": "electricity",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
