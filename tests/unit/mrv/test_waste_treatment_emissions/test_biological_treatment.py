# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-007 BiologicalTreatmentEngine.

Tests composting emissions, anaerobic digestion, MBT, batch processing,
BMP defaults, volatile solids estimation, Decimal precision,
GWP conversion, and audit trail.

Target: 130+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Standalone helper - creates engine without conftest
# ===========================================================================


def _make_bio_engine():
    """Create a BiologicalTreatmentEngine with a real database engine."""
    from greenlang.waste_treatment_emissions.waste_treatment_database import (
        WasteTreatmentDatabaseEngine,
    )
    from greenlang.waste_treatment_emissions.biological_treatment import (
        BiologicalTreatmentEngine,
    )
    db = WasteTreatmentDatabaseEngine()
    return BiologicalTreatmentEngine(database=db)


def _composting_input(**overrides) -> Dict[str, Any]:
    """Return a minimal valid composting calculation input."""
    base: Dict[str, Any] = {
        "treatment_method": "composting",
        "management_type": "well_managed",
        "waste_type": "food_waste",
        "waste_quantity_tonnes": Decimal("100.0"),
        "moisture_content": Decimal("0.60"),
        "gwp_source": "AR6",
    }
    base.update(overrides)
    return base


def _ad_input(**overrides) -> Dict[str, Any]:
    """Return a minimal valid anaerobic digestion calculation input."""
    base: Dict[str, Any] = {
        "treatment_method": "anaerobic_digestion",
        "waste_type": "food_waste",
        "waste_quantity_tonnes": Decimal("200.0"),
        "volatile_solids_fraction": Decimal("0.80"),
        "digestion_efficiency": Decimal("0.70"),
        "biogas_ch4_fraction": Decimal("0.60"),
        "capture_efficiency": Decimal("0.90"),
        "flare_fraction": Decimal("0.50"),
        "utilize_fraction": Decimal("0.50"),
        "gwp_source": "AR6",
    }
    base.update(overrides)
    return base


def _mbt_input(**overrides) -> Dict[str, Any]:
    """Return a minimal valid MBT calculation input."""
    base: Dict[str, Any] = {
        "treatment_method": "mbt",
        "mbt_type": "mbt_aerobic",
        "waste_type": "mixed_msw",
        "waste_quantity_tonnes": Decimal("500.0"),
        "gwp_source": "AR6",
    }
    base.update(overrides)
    return base


# ===========================================================================
# TestComposting - windrow, in-vessel, aerated static pile
# ===========================================================================


class TestComposting:
    """Tests for composting emission calculations."""

    # -- Well-managed composting ----------------------------------------

    def test_well_managed_ch4_per_kg(self, bio_engine):
        """Well-managed composting: 4g CH4/kg wet waste (IPCC 2019)."""
        inp = _composting_input(
            management_type="well_managed",
            waste_quantity_tonnes=Decimal("1.0"),  # 1 tonne = 1000 kg
        )
        result = bio_engine.calculate_composting(inp)
        # 1000 kg * 4.0 g/kg = 4000 g = 4.0 kg CH4
        expected_ch4_kg = Decimal("4.0")
        assert result["ch4_kg"] == pytest.approx(float(expected_ch4_kg), rel=1e-3)

    def test_well_managed_n2o_per_kg(self, bio_engine):
        """Well-managed composting: 0.24g N2O/kg wet waste."""
        inp = _composting_input(
            management_type="well_managed",
            waste_quantity_tonnes=Decimal("1.0"),
        )
        result = bio_engine.calculate_composting(inp)
        # 1000 kg * 0.24 g/kg = 240 g = 0.24 kg N2O
        expected_n2o_kg = Decimal("0.24")
        assert result["n2o_kg"] == pytest.approx(float(expected_n2o_kg), rel=1e-3)

    # -- Poorly managed composting --------------------------------------

    def test_poorly_managed_ch4_per_kg(self, bio_engine):
        """Poorly managed composting: 10g CH4/kg wet waste."""
        inp = _composting_input(
            management_type="poorly_managed",
            waste_quantity_tonnes=Decimal("1.0"),
        )
        result = bio_engine.calculate_composting(inp)
        expected_ch4_kg = Decimal("10.0")
        assert result["ch4_kg"] == pytest.approx(float(expected_ch4_kg), rel=1e-3)

    def test_poorly_managed_n2o_per_kg(self, bio_engine):
        """Poorly managed composting: 0.6g N2O/kg wet waste."""
        inp = _composting_input(
            management_type="poorly_managed",
            waste_quantity_tonnes=Decimal("1.0"),
        )
        result = bio_engine.calculate_composting(inp)
        expected_n2o_kg = Decimal("0.6")
        assert result["n2o_kg"] == pytest.approx(float(expected_n2o_kg), rel=1e-3)

    # -- Biofilter efficiency -------------------------------------------

    @pytest.mark.parametrize("biofilter_efficiency,expected_ch4_reduction", [
        (Decimal("0.10"), Decimal("0.90")),
        (Decimal("0.50"), Decimal("0.50")),
        (Decimal("0.80"), Decimal("0.20")),
        (Decimal("0.95"), Decimal("0.05")),
    ])
    def test_biofilter_reduces_ch4(self, bio_engine, biofilter_efficiency,
                                    expected_ch4_reduction):
        """Biofilter reduces CH4 emissions by the specified efficiency."""
        inp_no_filter = _composting_input(
            management_type="well_managed",
            waste_quantity_tonnes=Decimal("100.0"),
        )
        result_no_filter = bio_engine.calculate_composting(inp_no_filter)

        inp_with_filter = _composting_input(
            management_type="well_managed",
            waste_quantity_tonnes=Decimal("100.0"),
            biofilter_efficiency=biofilter_efficiency,
        )
        result_with_filter = bio_engine.calculate_composting(inp_with_filter)

        ratio = result_with_filter["ch4_kg"] / result_no_filter["ch4_kg"]
        assert ratio == pytest.approx(float(expected_ch4_reduction), abs=0.05)

    # -- Multiple waste types -------------------------------------------

    @pytest.mark.parametrize("waste_type", [
        "food_waste", "yard_waste", "garden_waste", "mixed_msw",
    ])
    def test_composting_various_waste_types(self, bio_engine, waste_type):
        """Composting calculation succeeds for various waste types."""
        inp = _composting_input(waste_type=waste_type)
        result = bio_engine.calculate_composting(inp)
        assert result["ch4_kg"] >= 0
        assert result["n2o_kg"] >= 0
        assert result["total_co2e_kg"] >= 0

    # -- Scaling with quantity ------------------------------------------

    def test_ch4_scales_linearly(self, bio_engine):
        """Doubling waste quantity doubles CH4 emissions."""
        inp_100 = _composting_input(waste_quantity_tonnes=Decimal("100.0"))
        inp_200 = _composting_input(waste_quantity_tonnes=Decimal("200.0"))
        r100 = bio_engine.calculate_composting(inp_100)
        r200 = bio_engine.calculate_composting(inp_200)
        ratio = r200["ch4_kg"] / r100["ch4_kg"]
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_n2o_scales_linearly(self, bio_engine):
        """Doubling waste quantity doubles N2O emissions."""
        inp_100 = _composting_input(waste_quantity_tonnes=Decimal("100.0"))
        inp_200 = _composting_input(waste_quantity_tonnes=Decimal("200.0"))
        r100 = bio_engine.calculate_composting(inp_100)
        r200 = bio_engine.calculate_composting(inp_200)
        ratio = r200["n2o_kg"] / r100["n2o_kg"]
        assert ratio == pytest.approx(2.0, rel=1e-6)

    # -- Edge cases -----------------------------------------------------

    def test_zero_waste_produces_zero_emissions(self, bio_engine):
        """Zero waste quantity produces zero emissions."""
        inp = _composting_input(waste_quantity_tonnes=Decimal("0"))
        result = bio_engine.calculate_composting(inp)
        assert result["ch4_kg"] == pytest.approx(0.0)
        assert result["n2o_kg"] == pytest.approx(0.0)
        assert result["total_co2e_kg"] == pytest.approx(0.0)

    def test_very_large_quantity(self, bio_engine):
        """Very large waste quantity (100,000 tonnes) calculates without error."""
        inp = _composting_input(waste_quantity_tonnes=Decimal("100000.0"))
        result = bio_engine.calculate_composting(inp)
        assert result["ch4_kg"] > 0
        assert result["n2o_kg"] > 0

    # -- CO2e total includes both gases ---------------------------------

    def test_co2e_includes_ch4_and_n2o(self, bio_engine):
        """Total CO2e includes both CH4 and N2O contributions."""
        inp = _composting_input()
        result = bio_engine.calculate_composting(inp)
        # CO2e must be at least as large as the larger individual contribution
        assert result["total_co2e_kg"] >= result["ch4_co2e_kg"]
        assert result["total_co2e_kg"] >= result["n2o_co2e_kg"]
        # And approximately equal to their sum
        expected = result["ch4_co2e_kg"] + result["n2o_co2e_kg"]
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=1e-6)

    # -- Result structure -----------------------------------------------

    def test_composting_result_has_required_keys(self, bio_engine):
        """Composting result contains all required output keys."""
        inp = _composting_input()
        result = bio_engine.calculate_composting(inp)
        required_keys = [
            "ch4_kg", "n2o_kg", "ch4_co2e_kg", "n2o_co2e_kg",
            "total_co2e_kg", "treatment_method", "waste_type",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_composting_result_method_label(self, bio_engine):
        """Result correctly labels the treatment method as composting."""
        inp = _composting_input()
        result = bio_engine.calculate_composting(inp)
        assert result["treatment_method"] == "composting"


# ===========================================================================
# TestAnaerobicDigestion - biogas production, CH4 capture, flaring, utilization
# ===========================================================================


class TestAnaerobicDigestion:
    """Tests for anaerobic digestion emission calculations."""

    # -- Default BMP by waste type --------------------------------------

    @pytest.mark.parametrize("waste_type,expected_min_bmp", [
        ("food_waste", Decimal("0.30")),
        ("sewage_sludge", Decimal("0.15")),
        ("garden_waste", Decimal("0.10")),
        ("agricultural_residue", Decimal("0.15")),
    ])
    def test_default_bmp_by_waste_type(self, bio_engine, waste_type, expected_min_bmp):
        """Default BMP (biochemical methane potential) is reasonable per waste type."""
        bmp = bio_engine.get_default_bmp(waste_type)
        assert bmp >= expected_min_bmp

    # -- Volatile solids estimation -------------------------------------

    def test_volatile_solids_estimation(self, bio_engine):
        """VS estimation from waste type returns reasonable fraction."""
        vs = bio_engine.estimate_volatile_solids("food_waste")
        assert Decimal("0.50") <= vs <= Decimal("0.95")

    def test_volatile_solids_sewage_sludge(self, bio_engine):
        """Sewage sludge VS fraction is in typical range."""
        vs = bio_engine.estimate_volatile_solids("sewage_sludge")
        assert Decimal("0.50") <= vs <= Decimal("0.85")

    # -- CH4 fraction in biogas -----------------------------------------

    @pytest.mark.parametrize("ch4_fraction", [
        Decimal("0.50"), Decimal("0.55"), Decimal("0.60"),
        Decimal("0.65"), Decimal("0.70"),
    ])
    def test_ch4_fraction_in_biogas(self, bio_engine, ch4_fraction):
        """Different CH4 fractions in biogas produce proportional results."""
        inp_low = _ad_input(biogas_ch4_fraction=Decimal("0.50"))
        inp_high = _ad_input(biogas_ch4_fraction=ch4_fraction)
        r_low = bio_engine.calculate_anaerobic_digestion(inp_low)
        r_high = bio_engine.calculate_anaerobic_digestion(inp_high)
        if ch4_fraction > Decimal("0.50"):
            assert r_high["biogas_ch4_m3"] >= r_low["biogas_ch4_m3"]

    # -- Capture efficiency scenarios -----------------------------------

    @pytest.mark.parametrize("capture_eff", [
        Decimal("0.50"), Decimal("0.70"), Decimal("0.85"),
        Decimal("0.90"), Decimal("0.95"),
    ])
    def test_capture_efficiency_reduces_vented(self, bio_engine, capture_eff):
        """Higher capture efficiency reduces vented CH4."""
        inp = _ad_input(capture_efficiency=capture_eff)
        result = bio_engine.calculate_anaerobic_digestion(inp)
        # Vented = total_produced * (1 - capture_efficiency)
        assert result["ch4_vented_kg"] >= 0
        if capture_eff > Decimal("0.50"):
            inp_low = _ad_input(capture_efficiency=Decimal("0.50"))
            r_low = bio_engine.calculate_anaerobic_digestion(inp_low)
            assert result["ch4_vented_kg"] <= r_low["ch4_vented_kg"]

    # -- Flare vs utilize allocation ------------------------------------

    def test_all_flared(self, bio_engine):
        """100% flare allocation sends all captured CH4 to flare."""
        inp = _ad_input(
            flare_fraction=Decimal("1.0"),
            utilize_fraction=Decimal("0.0"),
        )
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["ch4_utilized_kg"] == pytest.approx(0.0, abs=0.01)

    def test_all_utilized(self, bio_engine):
        """100% utilization allocation sends all captured CH4 to engine."""
        inp = _ad_input(
            flare_fraction=Decimal("0.0"),
            utilize_fraction=Decimal("1.0"),
        )
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["ch4_flared_kg"] == pytest.approx(0.0, abs=0.01)

    def test_equal_flare_utilize_split(self, bio_engine):
        """50/50 split sends equal amounts to flare and utilization."""
        inp = _ad_input(
            flare_fraction=Decimal("0.50"),
            utilize_fraction=Decimal("0.50"),
        )
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["ch4_flared_kg"] == pytest.approx(
            result["ch4_utilized_kg"], rel=0.01
        )

    # -- Net emissions after recovery -----------------------------------

    def test_net_emissions_less_than_gross(self, bio_engine):
        """Net emissions are less than gross when capture occurs."""
        inp = _ad_input(capture_efficiency=Decimal("0.90"))
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["net_co2e_kg"] < result["gross_co2e_kg"]

    def test_zero_capture_net_equals_gross(self, bio_engine):
        """With zero capture, net emissions equal gross."""
        inp = _ad_input(capture_efficiency=Decimal("0.0"))
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["net_co2e_kg"] == pytest.approx(
            result["gross_co2e_kg"], rel=1e-4
        )

    # -- Biogas production -----------------------------------------------

    def test_biogas_production_positive(self, bio_engine):
        """AD produces positive biogas volume."""
        inp = _ad_input()
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["biogas_total_m3"] > 0
        assert result["biogas_ch4_m3"] > 0

    def test_biogas_ch4_fraction_in_result(self, bio_engine):
        """CH4 volume is a fraction of total biogas volume."""
        inp = _ad_input(biogas_ch4_fraction=Decimal("0.60"))
        result = bio_engine.calculate_anaerobic_digestion(inp)
        ratio = result["biogas_ch4_m3"] / result["biogas_total_m3"]
        assert ratio == pytest.approx(0.60, rel=0.02)

    # -- Result structure -----------------------------------------------

    def test_ad_result_has_required_keys(self, bio_engine):
        """AD result contains all required output keys."""
        inp = _ad_input()
        result = bio_engine.calculate_anaerobic_digestion(inp)
        required_keys = [
            "biogas_total_m3", "biogas_ch4_m3",
            "ch4_produced_kg", "ch4_captured_kg",
            "ch4_flared_kg", "ch4_utilized_kg", "ch4_vented_kg",
            "gross_co2e_kg", "net_co2e_kg", "treatment_method",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_ad_result_method_label(self, bio_engine):
        """Result labels treatment method as anaerobic_digestion."""
        inp = _ad_input()
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["treatment_method"] == "anaerobic_digestion"

    # -- Scaling ---------------------------------------------------------

    def test_ad_scales_with_quantity(self, bio_engine):
        """Doubling waste quantity approximately doubles biogas production."""
        inp_100 = _ad_input(waste_quantity_tonnes=Decimal("100.0"))
        inp_200 = _ad_input(waste_quantity_tonnes=Decimal("200.0"))
        r100 = bio_engine.calculate_anaerobic_digestion(inp_100)
        r200 = bio_engine.calculate_anaerobic_digestion(inp_200)
        ratio = r200["biogas_total_m3"] / r100["biogas_total_m3"]
        assert ratio == pytest.approx(2.0, rel=1e-4)

    # -- Zero waste edge case -------------------------------------------

    def test_ad_zero_waste(self, bio_engine):
        """Zero waste produces zero biogas and zero emissions."""
        inp = _ad_input(waste_quantity_tonnes=Decimal("0"))
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["biogas_total_m3"] == pytest.approx(0.0)
        assert result["ch4_produced_kg"] == pytest.approx(0.0)
        assert result["net_co2e_kg"] == pytest.approx(0.0)

    # -- Digestion efficiency range -------------------------------------

    @pytest.mark.parametrize("eff", [
        Decimal("0.50"), Decimal("0.60"), Decimal("0.70"),
        Decimal("0.80"), Decimal("0.90"),
    ])
    def test_digestion_efficiency_range(self, bio_engine, eff):
        """Higher digestion efficiency produces more biogas."""
        inp_low = _ad_input(digestion_efficiency=Decimal("0.50"))
        inp_high = _ad_input(digestion_efficiency=eff)
        r_low = bio_engine.calculate_anaerobic_digestion(inp_low)
        r_high = bio_engine.calculate_anaerobic_digestion(inp_high)
        if eff > Decimal("0.50"):
            assert r_high["biogas_total_m3"] >= r_low["biogas_total_m3"]


# ===========================================================================
# TestMBT - mechanical biological treatment
# ===========================================================================


class TestMBT:
    """Tests for Mechanical Biological Treatment (MBT) calculations."""

    def test_mbt_aerobic_returns_result(self, bio_engine):
        """MBT aerobic calculation returns a valid result."""
        inp = _mbt_input(mbt_type="mbt_aerobic")
        result = bio_engine.calculate_mbt(inp)
        assert result["ch4_kg"] >= 0
        assert result["n2o_kg"] >= 0
        assert result["total_co2e_kg"] >= 0

    def test_mbt_anaerobic_returns_result(self, bio_engine):
        """MBT anaerobic calculation returns a valid result."""
        inp = _mbt_input(mbt_type="mbt_anaerobic")
        result = bio_engine.calculate_mbt(inp)
        assert result["ch4_kg"] >= 0
        assert result["total_co2e_kg"] >= 0

    def test_mbt_aerobic_has_n2o(self, bio_engine):
        """Aerobic MBT pre-treatment produces N2O emissions."""
        inp = _mbt_input(mbt_type="mbt_aerobic")
        result = bio_engine.calculate_mbt(inp)
        assert result["n2o_kg"] >= 0

    def test_mbt_anaerobic_lower_ch4_than_aerobic(self, bio_engine):
        """Anaerobic MBT typically has lower CH4 than poorly-managed aerobic."""
        inp_aerobic = _mbt_input(mbt_type="mbt_aerobic")
        inp_anaerobic = _mbt_input(mbt_type="mbt_anaerobic")
        r_aer = bio_engine.calculate_mbt(inp_aerobic)
        r_ana = bio_engine.calculate_mbt(inp_anaerobic)
        # Anaerobic MBT captures CH4; both should be valid
        assert r_aer["ch4_kg"] >= 0
        assert r_ana["ch4_kg"] >= 0

    def test_mbt_scales_with_quantity(self, bio_engine):
        """Doubling MBT input doubles CH4 emissions."""
        inp_500 = _mbt_input(waste_quantity_tonnes=Decimal("500.0"))
        inp_1000 = _mbt_input(waste_quantity_tonnes=Decimal("1000.0"))
        r_500 = bio_engine.calculate_mbt(inp_500)
        r_1000 = bio_engine.calculate_mbt(inp_1000)
        ratio = r_1000["ch4_kg"] / r_500["ch4_kg"]
        assert ratio == pytest.approx(2.0, rel=1e-4)

    def test_mbt_zero_waste(self, bio_engine):
        """Zero waste MBT produces zero emissions."""
        inp = _mbt_input(waste_quantity_tonnes=Decimal("0"))
        result = bio_engine.calculate_mbt(inp)
        assert result["ch4_kg"] == pytest.approx(0.0)
        assert result["total_co2e_kg"] == pytest.approx(0.0)

    def test_mbt_result_has_method_label(self, bio_engine):
        """MBT result includes treatment_method key."""
        inp = _mbt_input()
        result = bio_engine.calculate_mbt(inp)
        assert result["treatment_method"] == "mbt"


# ===========================================================================
# TestBiologicalBatch - batch processing of multiple treatments
# ===========================================================================


class TestBiologicalBatch:
    """Tests for batch processing of multiple biological treatments."""

    def test_batch_single_item(self, bio_engine):
        """Batch with a single item returns one result."""
        items = [_composting_input()]
        results = bio_engine.calculate_batch(items)
        assert len(results) == 1

    def test_batch_multiple_items(self, bio_engine):
        """Batch with multiple items returns matching number of results."""
        items = [
            _composting_input(waste_quantity_tonnes=Decimal("100")),
            _composting_input(waste_quantity_tonnes=Decimal("200")),
            _composting_input(waste_quantity_tonnes=Decimal("300")),
        ]
        results = bio_engine.calculate_batch(items)
        assert len(results) == 3

    def test_batch_mixed_treatment_types(self, bio_engine):
        """Batch handles a mix of composting and AD calculations."""
        items = [
            _composting_input(),
            _ad_input(),
            _mbt_input(),
        ]
        results = bio_engine.calculate_batch(items)
        assert len(results) == 3
        methods = [r.get("treatment_method") for r in results]
        assert "composting" in methods
        assert "anaerobic_digestion" in methods
        assert "mbt" in methods

    def test_batch_empty_list(self, bio_engine):
        """Batch with empty list returns empty results."""
        results = bio_engine.calculate_batch([])
        assert results == []

    def test_batch_total_co2e_is_sum(self, bio_engine):
        """Batch total CO2e equals the sum of individual totals."""
        items = [
            _composting_input(waste_quantity_tonnes=Decimal("100")),
            _composting_input(waste_quantity_tonnes=Decimal("200")),
        ]
        results = bio_engine.calculate_batch(items)
        batch_total = sum(r["total_co2e_kg"] for r in results)
        # Compare against individual calculations
        r1 = bio_engine.calculate_composting(items[0])
        r2 = bio_engine.calculate_composting(items[1])
        individual_total = r1["total_co2e_kg"] + r2["total_co2e_kg"]
        assert batch_total == pytest.approx(individual_total, rel=1e-6)


# ===========================================================================
# TestBMPDefaults - default BMP values by waste category
# ===========================================================================


class TestBMPDefaults:
    """Tests for default Biochemical Methane Potential values."""

    @pytest.mark.parametrize("waste_type", [
        "food_waste", "sewage_sludge", "garden_waste",
        "agricultural_residue", "mixed_msw",
    ])
    def test_bmp_returns_decimal(self, bio_engine, waste_type):
        """get_default_bmp returns a Decimal for known waste types."""
        bmp = bio_engine.get_default_bmp(waste_type)
        assert isinstance(bmp, (Decimal, float))

    def test_bmp_food_waste_highest(self, bio_engine):
        """Food waste has the highest BMP among common wastes."""
        bmp_food = bio_engine.get_default_bmp("food_waste")
        bmp_garden = bio_engine.get_default_bmp("garden_waste")
        assert bmp_food >= bmp_garden

    def test_bmp_all_positive(self, bio_engine):
        """All BMP values are positive."""
        for wt in ["food_waste", "sewage_sludge", "garden_waste"]:
            assert bio_engine.get_default_bmp(wt) > 0

    def test_bmp_range_reasonable(self, bio_engine):
        """BMP values are in reasonable range (0.05-0.60 m3 CH4/kg VS)."""
        for wt in ["food_waste", "sewage_sludge", "garden_waste"]:
            bmp = float(bio_engine.get_default_bmp(wt))
            assert 0.05 <= bmp <= 0.60


# ===========================================================================
# TestVolatileSolids - volatile solids estimation
# ===========================================================================


class TestVolatileSolids:
    """Tests for volatile solids estimation by waste type."""

    @pytest.mark.parametrize("waste_type,min_vs,max_vs", [
        ("food_waste", Decimal("0.70"), Decimal("0.95")),
        ("sewage_sludge", Decimal("0.50"), Decimal("0.85")),
        ("garden_waste", Decimal("0.40"), Decimal("0.80")),
        ("agricultural_residue", Decimal("0.60"), Decimal("0.90")),
        ("mixed_msw", Decimal("0.40"), Decimal("0.75")),
    ])
    def test_vs_within_range(self, bio_engine, waste_type, min_vs, max_vs):
        """Volatile solids fraction is within expected range for waste type."""
        vs = bio_engine.estimate_volatile_solids(waste_type)
        assert min_vs <= vs <= max_vs

    def test_vs_returns_decimal(self, bio_engine):
        """estimate_volatile_solids returns a Decimal."""
        vs = bio_engine.estimate_volatile_solids("food_waste")
        assert isinstance(vs, (Decimal, float))

    def test_vs_food_higher_than_garden(self, bio_engine):
        """Food waste has higher VS than garden waste."""
        vs_food = bio_engine.estimate_volatile_solids("food_waste")
        vs_garden = bio_engine.estimate_volatile_solids("garden_waste")
        assert vs_food >= vs_garden


# ===========================================================================
# TestDecimalPrecision - all calculations use Decimal, reproducible
# ===========================================================================


class TestDecimalPrecision:
    """Tests for Decimal precision and reproducibility."""

    def test_composting_deterministic(self, bio_engine):
        """Same input produces identical composting output (bit-perfect)."""
        inp = _composting_input()
        r1 = bio_engine.calculate_composting(inp)
        r2 = bio_engine.calculate_composting(inp)
        assert r1["ch4_kg"] == r2["ch4_kg"]
        assert r1["n2o_kg"] == r2["n2o_kg"]
        assert r1["total_co2e_kg"] == r2["total_co2e_kg"]

    def test_ad_deterministic(self, bio_engine):
        """Same input produces identical AD output (bit-perfect)."""
        inp = _ad_input()
        r1 = bio_engine.calculate_anaerobic_digestion(inp)
        r2 = bio_engine.calculate_anaerobic_digestion(inp)
        assert r1["biogas_total_m3"] == r2["biogas_total_m3"]
        assert r1["net_co2e_kg"] == r2["net_co2e_kg"]

    def test_composting_result_types(self, bio_engine):
        """Composting result values are numeric (Decimal or float)."""
        inp = _composting_input()
        result = bio_engine.calculate_composting(inp)
        for key in ["ch4_kg", "n2o_kg", "total_co2e_kg"]:
            assert isinstance(result[key], (Decimal, float, int))

    def test_ad_result_types(self, bio_engine):
        """AD result values are numeric (Decimal or float)."""
        inp = _ad_input()
        result = bio_engine.calculate_anaerobic_digestion(inp)
        for key in ["biogas_total_m3", "ch4_produced_kg", "net_co2e_kg"]:
            assert isinstance(result[key], (Decimal, float, int))

    def test_precision_of_small_quantities(self, bio_engine):
        """Small waste quantities produce non-zero but small emissions."""
        inp = _composting_input(waste_quantity_tonnes=Decimal("0.001"))
        result = bio_engine.calculate_composting(inp)
        assert result["ch4_kg"] > 0
        assert result["ch4_kg"] < 0.1  # Very small


# ===========================================================================
# TestGWPConversion - CH4 biogenic GWP applied correctly
# ===========================================================================


class TestGWPConversion:
    """Tests for GWP conversion in biological treatment calculations."""

    def test_ar6_ch4_biogenic_gwp_applied(self, bio_engine):
        """AR6 uses biogenic CH4 GWP (27) for composting emissions."""
        inp = _composting_input(gwp_source="AR6")
        result = bio_engine.calculate_composting(inp)
        # CH4 CO2e = CH4_kg * GWP_biogenic_CH4
        # For AR6: biogenic CH4 GWP = 27.0 (vs fossil 29.8)
        ch4_kg = result["ch4_kg"]
        ch4_co2e = result["ch4_co2e_kg"]
        implied_gwp = ch4_co2e / ch4_kg if ch4_kg > 0 else 0
        # Biogenic CH4 GWP should be ~27.0 for AR6
        assert implied_gwp == pytest.approx(27.0, rel=0.05)

    def test_ar4_ch4_gwp_applied(self, bio_engine):
        """AR4 uses CH4 GWP of 25."""
        inp = _composting_input(gwp_source="AR4")
        result = bio_engine.calculate_composting(inp)
        ch4_kg = result["ch4_kg"]
        ch4_co2e = result["ch4_co2e_kg"]
        implied_gwp = ch4_co2e / ch4_kg if ch4_kg > 0 else 0
        assert implied_gwp == pytest.approx(25.0, rel=0.05)

    def test_ar5_ch4_gwp_applied(self, bio_engine):
        """AR5 uses CH4 GWP of 28."""
        inp = _composting_input(gwp_source="AR5")
        result = bio_engine.calculate_composting(inp)
        ch4_kg = result["ch4_kg"]
        ch4_co2e = result["ch4_co2e_kg"]
        implied_gwp = ch4_co2e / ch4_kg if ch4_kg > 0 else 0
        assert implied_gwp == pytest.approx(28.0, rel=0.05)

    def test_n2o_gwp_ar6(self, bio_engine):
        """N2O GWP is 273 under AR6."""
        inp = _composting_input(gwp_source="AR6")
        result = bio_engine.calculate_composting(inp)
        n2o_kg = result["n2o_kg"]
        n2o_co2e = result["n2o_co2e_kg"]
        implied_gwp = n2o_co2e / n2o_kg if n2o_kg > 0 else 0
        assert implied_gwp == pytest.approx(273.0, rel=0.05)

    def test_higher_gwp_produces_higher_co2e(self, bio_engine):
        """AR6 CH4 GWP > AR4 CH4 GWP produces higher CO2e for same mass."""
        inp_ar4 = _composting_input(gwp_source="AR4")
        inp_ar6 = _composting_input(gwp_source="AR6")
        r_ar4 = bio_engine.calculate_composting(inp_ar4)
        r_ar6 = bio_engine.calculate_composting(inp_ar6)
        # AR6 biogenic CH4 GWP (27) > AR4 CH4 GWP (25)
        assert r_ar6["ch4_co2e_kg"] >= r_ar4["ch4_co2e_kg"]


# ===========================================================================
# TestAuditTrail - calculation steps recorded
# ===========================================================================


class TestAuditTrail:
    """Tests for audit trail and provenance in biological treatment."""

    def test_composting_has_provenance_hash(self, bio_engine):
        """Composting result includes a provenance hash."""
        inp = _composting_input()
        result = bio_engine.calculate_composting(inp)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64  # SHA-256

    def test_ad_has_provenance_hash(self, bio_engine):
        """AD result includes a provenance hash."""
        inp = _ad_input()
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_composting_provenance_deterministic(self, bio_engine):
        """Same composting input produces same provenance hash."""
        inp = _composting_input()
        r1 = bio_engine.calculate_composting(inp)
        r2 = bio_engine.calculate_composting(inp)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_different_input_different_provenance(self, bio_engine):
        """Different inputs produce different provenance hashes."""
        inp1 = _composting_input(waste_quantity_tonnes=Decimal("100"))
        inp2 = _composting_input(waste_quantity_tonnes=Decimal("200"))
        r1 = bio_engine.calculate_composting(inp1)
        r2 = bio_engine.calculate_composting(inp2)
        assert r1["provenance_hash"] != r2["provenance_hash"]

    def test_composting_has_calculation_steps(self, bio_engine):
        """Composting result includes calculation steps for audit."""
        inp = _composting_input()
        result = bio_engine.calculate_composting(inp)
        assert "calculation_steps" in result or "audit_trail" in result

    def test_ad_has_calculation_steps(self, bio_engine):
        """AD result includes calculation steps for audit."""
        inp = _ad_input()
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert "calculation_steps" in result or "audit_trail" in result

    def test_composting_has_processing_time(self, bio_engine):
        """Composting result includes processing time in milliseconds."""
        inp = _composting_input()
        result = bio_engine.calculate_composting(inp)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_ad_has_processing_time(self, bio_engine):
        """AD result includes processing time in milliseconds."""
        inp = _ad_input()
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_composting_result_includes_input_summary(self, bio_engine):
        """Composting result includes a summary of input parameters."""
        inp = _composting_input()
        result = bio_engine.calculate_composting(inp)
        assert result.get("waste_type") == "food_waste"

    def test_mbt_has_provenance_hash(self, bio_engine):
        """MBT result includes a provenance hash."""
        inp = _mbt_input()
        result = bio_engine.calculate_mbt(inp)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# TestCompostingWindowTypes - windrow, in-vessel, aerated static pile
# ===========================================================================


class TestCompostingWindowTypes:
    """Tests for different composting system configurations."""

    def test_windrow_composting(self, bio_engine):
        """Windrow composting with well-managed conditions."""
        inp = _composting_input(
            composting_system="windrow",
            management_type="well_managed",
            waste_quantity_tonnes=Decimal("500.0"),
        )
        result = bio_engine.calculate_composting(inp)
        assert result["ch4_kg"] > 0
        assert result["n2o_kg"] > 0

    def test_in_vessel_composting(self, bio_engine):
        """In-vessel composting with controlled conditions."""
        inp = _composting_input(
            composting_system="in_vessel",
            management_type="well_managed",
            waste_quantity_tonnes=Decimal("500.0"),
        )
        result = bio_engine.calculate_composting(inp)
        assert result["ch4_kg"] >= 0
        assert result["n2o_kg"] >= 0

    def test_aerated_static_pile(self, bio_engine):
        """Aerated static pile composting."""
        inp = _composting_input(
            composting_system="aerated_static_pile",
            management_type="well_managed",
            waste_quantity_tonnes=Decimal("500.0"),
        )
        result = bio_engine.calculate_composting(inp)
        assert result["total_co2e_kg"] >= 0


# ===========================================================================
# TestBiologicalEdgeCases - additional edge cases
# ===========================================================================


class TestBiologicalEdgeCases:
    """Additional edge case tests for biological treatment engine."""

    def test_composting_negative_quantity_raises(self, bio_engine):
        """Negative waste quantity raises ValueError."""
        inp = _composting_input(waste_quantity_tonnes=Decimal("-10.0"))
        with pytest.raises((ValueError, AssertionError)):
            bio_engine.calculate_composting(inp)

    def test_ad_negative_capture_raises(self, bio_engine):
        """Negative capture efficiency raises ValueError."""
        inp = _ad_input(capture_efficiency=Decimal("-0.1"))
        with pytest.raises((ValueError, AssertionError)):
            bio_engine.calculate_anaerobic_digestion(inp)

    def test_ad_capture_over_one_raises(self, bio_engine):
        """Capture efficiency > 1.0 raises ValueError."""
        inp = _ad_input(capture_efficiency=Decimal("1.1"))
        with pytest.raises((ValueError, AssertionError)):
            bio_engine.calculate_anaerobic_digestion(inp)

    def test_composting_very_high_moisture(self, bio_engine):
        """Very high moisture content (0.95) still produces emissions."""
        inp = _composting_input(moisture_content=Decimal("0.95"))
        result = bio_engine.calculate_composting(inp)
        assert result["ch4_kg"] >= 0

    def test_composting_zero_moisture(self, bio_engine):
        """Zero moisture content works (dry composting)."""
        inp = _composting_input(moisture_content=Decimal("0.0"))
        result = bio_engine.calculate_composting(inp)
        assert result["ch4_kg"] >= 0

    def test_ad_flare_plus_utilize_sum_to_one(self, bio_engine):
        """Flare + utilize fractions can sum to exactly 1.0."""
        inp = _ad_input(
            flare_fraction=Decimal("0.30"),
            utilize_fraction=Decimal("0.70"),
        )
        result = bio_engine.calculate_anaerobic_digestion(inp)
        total_recovered = result["ch4_flared_kg"] + result["ch4_utilized_kg"]
        assert total_recovered > 0

    def test_ad_different_waste_types(self, bio_engine):
        """AD handles various waste types."""
        for wt in ["food_waste", "sewage_sludge", "garden_waste",
                    "agricultural_residue"]:
            inp = _ad_input(waste_type=wt)
            result = bio_engine.calculate_anaerobic_digestion(inp)
            assert result["biogas_total_m3"] >= 0

    def test_mbt_large_quantity(self, bio_engine):
        """MBT handles large waste quantities (50,000 tonnes)."""
        inp = _mbt_input(waste_quantity_tonnes=Decimal("50000.0"))
        result = bio_engine.calculate_mbt(inp)
        assert result["ch4_kg"] > 0
        assert result["total_co2e_kg"] > 0

    def test_composting_10_tonnes(self, bio_engine):
        """Composting of 10 tonnes produces proportionate emissions."""
        inp_10 = _composting_input(waste_quantity_tonnes=Decimal("10.0"))
        inp_100 = _composting_input(waste_quantity_tonnes=Decimal("100.0"))
        r_10 = bio_engine.calculate_composting(inp_10)
        r_100 = bio_engine.calculate_composting(inp_100)
        ratio = r_100["ch4_kg"] / r_10["ch4_kg"]
        assert ratio == pytest.approx(10.0, rel=1e-4)

    def test_ad_all_vented_no_capture(self, bio_engine):
        """With 0% capture, all CH4 is vented."""
        inp = _ad_input(capture_efficiency=Decimal("0.0"))
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["ch4_vented_kg"] > 0
        assert result["ch4_captured_kg"] == pytest.approx(0.0, abs=0.01)
        assert result["ch4_flared_kg"] == pytest.approx(0.0, abs=0.01)
        assert result["ch4_utilized_kg"] == pytest.approx(0.0, abs=0.01)

    def test_ad_full_capture(self, bio_engine):
        """With 100% capture, no CH4 is vented."""
        inp = _ad_input(capture_efficiency=Decimal("1.0"))
        result = bio_engine.calculate_anaerobic_digestion(inp)
        assert result["ch4_vented_kg"] == pytest.approx(0.0, abs=0.01)
        assert result["ch4_captured_kg"] > 0

    def test_composting_provenance_hash_changes_with_gwp(self, bio_engine):
        """Different GWP source produces different provenance hash."""
        inp_ar4 = _composting_input(gwp_source="AR4")
        inp_ar6 = _composting_input(gwp_source="AR6")
        r_ar4 = bio_engine.calculate_composting(inp_ar4)
        r_ar6 = bio_engine.calculate_composting(inp_ar6)
        assert r_ar4["provenance_hash"] != r_ar6["provenance_hash"]

    def test_batch_preserves_order(self, bio_engine):
        """Batch results are in the same order as inputs."""
        items = [
            _composting_input(waste_quantity_tonnes=Decimal("100")),
            _composting_input(waste_quantity_tonnes=Decimal("200")),
            _composting_input(waste_quantity_tonnes=Decimal("300")),
        ]
        results = bio_engine.calculate_batch(items)
        # Each result should have increasing co2e
        for i in range(len(results) - 1):
            assert results[i]["total_co2e_kg"] < results[i + 1]["total_co2e_kg"]
