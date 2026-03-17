# -*- coding: utf-8 -*-
"""
Tests for CarbonFootprintEngine - PACK-020 Engine 1
=====================================================

Comprehensive tests for carbon footprint calculation per
EU Battery Regulation Art 7 and Annex II.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-020 Battery Passport Prep
"""

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Dynamic Import
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = PACK_ROOT / "engines"


def _load_module(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ENGINE_DIR / file_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module("carbon_footprint_engine.py", "pack020_test.engines.carbon_footprint")

CarbonFootprintEngine = mod.CarbonFootprintEngine
CarbonFootprintInput = mod.CarbonFootprintInput
CarbonFootprintResult = mod.CarbonFootprintResult
LifecycleEmissions = mod.LifecycleEmissions
LifecycleBreakdown = mod.LifecycleBreakdown
BenchmarkComparison = mod.BenchmarkComparison
LifecycleStage = mod.LifecycleStage
CarbonFootprintClass = mod.CarbonFootprintClass
BatteryCategory = mod.BatteryCategory
BatteryChemistry = mod.BatteryChemistry
PERFORMANCE_CLASS_THRESHOLDS = mod.PERFORMANCE_CLASS_THRESHOLDS
CATEGORY_MAX_THRESHOLDS = mod.CATEGORY_MAX_THRESHOLDS
CHEMISTRY_BENCHMARKS = mod.CHEMISTRY_BENCHMARKS
LIFECYCLE_STAGE_LABELS = mod.LIFECYCLE_STAGE_LABELS
CATEGORY_LABELS = mod.CATEGORY_LABELS
METHODOLOGY_REFERENCES = mod.METHODOLOGY_REFERENCES
_round_val = mod._round_val
_compute_hash = mod._compute_hash
_decimal = mod._decimal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return CarbonFootprintEngine()


@pytest.fixture
def basic_emissions():
    return [
        LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("3000")),
        LifecycleEmissions(stage=LifecycleStage.MANUFACTURING, co2e_kg=Decimal("1500")),
        LifecycleEmissions(stage=LifecycleStage.DISTRIBUTION, co2e_kg=Decimal("300")),
        LifecycleEmissions(stage=LifecycleStage.END_OF_LIFE, co2e_kg=Decimal("200")),
    ]


@pytest.fixture
def basic_input(basic_emissions):
    return CarbonFootprintInput(
        battery_id="BAT-TEST-001",
        category=BatteryCategory.EV,
        chemistry=BatteryChemistry.NMC811,
        energy_kwh=Decimal("75"),
        weight_kg=Decimal("450"),
        lifecycle_emissions=basic_emissions,
    )


# ---------------------------------------------------------------------------
# Test: Engine Initialization
# ---------------------------------------------------------------------------


class TestCarbonFootprintEngineInit:
    def test_init_creates_engine(self):
        engine = CarbonFootprintEngine()
        assert engine is not None

    def test_engine_version(self):
        engine = CarbonFootprintEngine()
        assert engine.engine_version == "1.0.0"

    def test_results_empty_on_init(self):
        engine = CarbonFootprintEngine()
        assert engine.get_results() == []


# ---------------------------------------------------------------------------
# Test: Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_lifecycle_stage_values(self):
        assert LifecycleStage.RAW_MATERIAL_EXTRACTION.value == "raw_material_extraction"
        assert LifecycleStage.MANUFACTURING.value == "manufacturing"
        assert LifecycleStage.DISTRIBUTION.value == "distribution"
        assert LifecycleStage.END_OF_LIFE.value == "end_of_life"
        assert len(LifecycleStage) == 4

    def test_carbon_footprint_class_values(self):
        assert CarbonFootprintClass.CLASS_A.value == "class_a"
        assert CarbonFootprintClass.CLASS_E.value == "class_e"
        assert len(CarbonFootprintClass) == 5

    def test_battery_category_values(self):
        assert BatteryCategory.EV.value == "ev"
        assert BatteryCategory.PORTABLE.value == "portable"
        assert len(BatteryCategory) == 5

    def test_battery_chemistry_values(self):
        assert BatteryChemistry.NMC811.value == "nmc811"
        assert BatteryChemistry.LFP.value == "lfp"
        assert len(BatteryChemistry) == 13


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_performance_class_thresholds(self):
        assert PERFORMANCE_CLASS_THRESHOLDS["class_a"] == Decimal("60")
        assert PERFORMANCE_CLASS_THRESHOLDS["class_b"] == Decimal("80")
        assert PERFORMANCE_CLASS_THRESHOLDS["class_c"] == Decimal("100")
        assert PERFORMANCE_CLASS_THRESHOLDS["class_d"] == Decimal("120")

    def test_category_max_thresholds(self):
        assert CATEGORY_MAX_THRESHOLDS["ev"] == Decimal("150")
        assert CATEGORY_MAX_THRESHOLDS["industrial"] == Decimal("200")
        assert CATEGORY_MAX_THRESHOLDS["lmt"] == Decimal("175")
        assert CATEGORY_MAX_THRESHOLDS["sli"] == Decimal("250")
        assert "portable" not in CATEGORY_MAX_THRESHOLDS

    def test_all_chemistries_have_benchmarks(self):
        for chem in BatteryChemistry:
            assert chem.value in CHEMISTRY_BENCHMARKS

    def test_benchmark_structure(self):
        for chem, data in CHEMISTRY_BENCHMARKS.items():
            assert "low" in data
            assert "typical" in data
            assert "high" in data
            assert data["low"] < data["typical"] < data["high"]


# ---------------------------------------------------------------------------
# Test: calculate_footprint
# ---------------------------------------------------------------------------


class TestCalculateFootprint:
    def test_basic_calculation(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        assert isinstance(result, CarbonFootprintResult)
        assert result.battery_id == "BAT-TEST-001"
        assert result.total_co2e_kg == Decimal("5000.000")

    def test_per_kwh_calculation(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        # 5000 / 75 = 66.667
        expected = _round_val(Decimal("5000") / Decimal("75"), 3)
        assert result.per_kwh_co2e_kg == expected

    def test_per_kg_calculation(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        expected = _round_val(Decimal("5000") / Decimal("450"), 3)
        assert result.per_kg_co2e_kg == expected

    def test_result_has_provenance_hash(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_result_has_processing_time(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        assert result.processing_time_ms >= 0.0

    def test_result_stored_in_engine(self, engine, basic_input):
        engine.calculate_footprint(basic_input)
        assert len(engine.get_results()) == 1

    def test_result_has_methodology(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        assert len(result.methodology) > 0

    def test_result_has_functional_unit(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        assert result.functional_unit != ""

    def test_lifecycle_breakdown_populated(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        assert len(result.lifecycle_breakdown) == 4

    def test_dominant_stage_identified(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        assert result.dominant_stage != ""
        assert result.dominant_stage_pct > Decimal("0")

    def test_data_quality_summary(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        assert "primary" in result.data_quality_summary


# ---------------------------------------------------------------------------
# Test: Performance Class Assignment
# ---------------------------------------------------------------------------


class TestPerformanceClass:
    @pytest.mark.parametrize("per_kwh,expected_class", [
        (Decimal("30"), CarbonFootprintClass.CLASS_A),
        (Decimal("60"), CarbonFootprintClass.CLASS_A),
        (Decimal("60.001"), CarbonFootprintClass.CLASS_B),
        (Decimal("80"), CarbonFootprintClass.CLASS_B),
        (Decimal("80.001"), CarbonFootprintClass.CLASS_C),
        (Decimal("100"), CarbonFootprintClass.CLASS_C),
        (Decimal("100.001"), CarbonFootprintClass.CLASS_D),
        (Decimal("120"), CarbonFootprintClass.CLASS_D),
        (Decimal("120.001"), CarbonFootprintClass.CLASS_E),
        (Decimal("200"), CarbonFootprintClass.CLASS_E),
        (Decimal("0"), CarbonFootprintClass.CLASS_A),
    ])
    def test_performance_class_thresholds(self, engine, per_kwh, expected_class):
        assert engine.assign_performance_class(per_kwh) == expected_class

    def test_class_a_boundary(self, engine):
        assert engine.assign_performance_class(Decimal("60")) == CarbonFootprintClass.CLASS_A
        assert engine.assign_performance_class(Decimal("60.01")) == CarbonFootprintClass.CLASS_B


# ---------------------------------------------------------------------------
# Test: Threshold Compliance
# ---------------------------------------------------------------------------


class TestThresholdCompliance:
    def test_ev_compliant(self, engine):
        result = engine.check_threshold_compliance(Decimal("100"), BatteryCategory.EV)
        assert result["compliant"] is True
        assert result["threshold"] == Decimal("150.000")
        assert result["headroom"] == Decimal("50.000")

    def test_ev_non_compliant(self, engine):
        result = engine.check_threshold_compliance(Decimal("160"), BatteryCategory.EV)
        assert result["compliant"] is False
        assert result["headroom"] == Decimal("-10.000")

    def test_portable_no_threshold(self, engine):
        result = engine.check_threshold_compliance(Decimal("300"), BatteryCategory.PORTABLE)
        assert result["compliant"] is True
        assert result["threshold"] is None
        assert result["headroom"] is None

    def test_ev_at_threshold(self, engine):
        result = engine.check_threshold_compliance(Decimal("150"), BatteryCategory.EV)
        assert result["compliant"] is True

    @pytest.mark.parametrize("category,threshold", [
        (BatteryCategory.EV, Decimal("150")),
        (BatteryCategory.INDUSTRIAL, Decimal("200")),
        (BatteryCategory.LMT, Decimal("175")),
        (BatteryCategory.SLI, Decimal("250")),
    ])
    def test_all_category_thresholds(self, engine, category, threshold):
        result = engine.check_threshold_compliance(threshold, category)
        assert result["compliant"] is True


# ---------------------------------------------------------------------------
# Test: Lifecycle Breakdown
# ---------------------------------------------------------------------------


class TestLifecycleBreakdown:
    def test_breakdown_percentages_sum(self, engine, basic_emissions):
        total = Decimal("5000")
        energy = Decimal("75")
        breakdown = engine.calculate_lifecycle_breakdown(basic_emissions, total, energy)
        pct_sum = sum(b.percentage for b in breakdown)
        assert pct_sum == Decimal("100.00")

    def test_breakdown_per_kwh(self, engine, basic_emissions):
        total = Decimal("5000")
        energy = Decimal("75")
        breakdown = engine.calculate_lifecycle_breakdown(basic_emissions, total, energy)
        for b in breakdown:
            assert b.per_kwh_co2e_kg >= Decimal("0")

    def test_breakdown_has_labels(self, engine, basic_emissions):
        total = Decimal("5000")
        energy = Decimal("75")
        breakdown = engine.calculate_lifecycle_breakdown(basic_emissions, total, energy)
        for b in breakdown:
            assert b.stage_label != ""

    def test_zero_total_emissions(self, engine):
        emissions = [
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("0")),
        ]
        breakdown = engine.calculate_lifecycle_breakdown(emissions, Decimal("0"), Decimal("75"))
        assert breakdown[0].percentage == Decimal("0.00")


# ---------------------------------------------------------------------------
# Test: Batch Processing
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    def test_batch_returns_results(self, engine, basic_input):
        results = engine.calculate_batch([basic_input, basic_input])
        assert len(results) == 2

    def test_batch_empty_input(self, engine):
        results = engine.calculate_batch([])
        assert results == []

    def test_batch_skips_invalid(self, engine, basic_input):
        # Create an input that will fail validation (negative total)
        bad_emissions = [
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("0")),
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("0")),
        ]
        bad_input = CarbonFootprintInput(
            battery_id="BAD",
            category=BatteryCategory.EV,
            chemistry=BatteryChemistry.NMC,
            energy_kwh=Decimal("75"),
            weight_kg=Decimal("450"),
            lifecycle_emissions=bad_emissions,
        )
        results = engine.calculate_batch([basic_input, bad_input])
        # bad_input has duplicate stages so should be skipped
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Test: Compare Footprints
# ---------------------------------------------------------------------------


class TestCompareFootprints:
    def test_compare_empty(self, engine):
        comparison = engine.compare_footprints([])
        assert comparison["count"] == 0

    def test_compare_single(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        comparison = engine.compare_footprints([result])
        assert comparison["count"] == 1
        assert "provenance_hash" in comparison

    def test_compare_has_statistics(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        comparison = engine.compare_footprints([result])
        assert "statistics" in comparison
        assert "min_per_kwh" in comparison["statistics"]

    def test_compare_has_class_distribution(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        comparison = engine.compare_footprints([result])
        assert "class_distribution" in comparison


# ---------------------------------------------------------------------------
# Test: Build Declaration
# ---------------------------------------------------------------------------


class TestBuildDeclaration:
    def test_declaration_structure(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        decl = engine.build_declaration(result)
        assert "declaration_id" in decl
        assert "regulation_reference" in decl
        assert "battery_information" in decl
        assert "carbon_footprint" in decl
        assert "lifecycle_stages" in decl
        assert "performance_class" in decl
        assert "threshold_compliance" in decl
        assert "provenance_hash" in decl

    def test_declaration_has_battery_info(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        decl = engine.build_declaration(result)
        assert decl["battery_information"]["battery_id"] == "BAT-TEST-001"

    def test_declaration_has_lifecycle_stages(self, engine, basic_input):
        result = engine.calculate_footprint(basic_input)
        decl = engine.build_declaration(result)
        assert len(decl["lifecycle_stages"]) == 4


# ---------------------------------------------------------------------------
# Test: Benchmark Lookup
# ---------------------------------------------------------------------------


class TestBenchmarkLookup:
    def test_known_chemistry(self, engine):
        result = engine.get_chemistry_benchmark(BatteryChemistry.NMC811)
        assert result["available"] is True
        assert "low_kgco2e_per_kwh" in result

    def test_all_benchmarks(self, engine):
        all_bm = engine.get_all_benchmarks()
        assert len(all_bm) == len(CHEMISTRY_BENCHMARKS)


# ---------------------------------------------------------------------------
# Test: Threshold Lookup
# ---------------------------------------------------------------------------


class TestThresholdLookup:
    def test_ev_threshold(self, engine):
        result = engine.get_category_threshold(BatteryCategory.EV)
        assert result["has_threshold"] is True
        assert result["max_kgco2e_per_kwh"] == "150"

    def test_portable_no_threshold(self, engine):
        result = engine.get_category_threshold(BatteryCategory.PORTABLE)
        assert result["has_threshold"] is False

    def test_all_thresholds(self, engine):
        all_thresh = engine.get_all_thresholds()
        assert len(all_thresh) == 4  # ev, industrial, lmt, sli

    def test_performance_class_thresholds_method(self, engine):
        result = engine.get_performance_class_thresholds()
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Test: Registry Management
# ---------------------------------------------------------------------------


class TestRegistryManagement:
    def test_clear_results(self, engine, basic_input):
        engine.calculate_footprint(basic_input)
        assert len(engine.get_results()) == 1
        engine.clear_results()
        assert len(engine.get_results()) == 0

    def test_get_results_returns_copy(self, engine, basic_input):
        engine.calculate_footprint(basic_input)
        results = engine.get_results()
        results.clear()
        assert len(engine.get_results()) == 1


# ---------------------------------------------------------------------------
# Test: Validation Errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_duplicate_stages_rejected(self, engine):
        emissions = [
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("3000")),
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("1000")),
        ]
        inp = CarbonFootprintInput(
            battery_id="BAT-DUP",
            category=BatteryCategory.EV,
            chemistry=BatteryChemistry.NMC,
            energy_kwh=Decimal("75"),
            weight_kg=Decimal("450"),
            lifecycle_emissions=emissions,
        )
        with pytest.raises(ValueError, match="Duplicate lifecycle stages"):
            engine.calculate_footprint(inp)

    def test_plausibility_ceiling_exceeded(self, engine):
        emissions = [
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("50000")),
        ]
        inp = CarbonFootprintInput(
            battery_id="BAT-HIGH",
            category=BatteryCategory.EV,
            chemistry=BatteryChemistry.NMC,
            energy_kwh=Decimal("75"),
            weight_kg=Decimal("450"),
            lifecycle_emissions=emissions,
        )
        with pytest.raises(ValueError, match="plausibility ceiling"):
            engine.calculate_footprint(inp)


# ---------------------------------------------------------------------------
# Test: Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_high_class_generates_recommendation(self, engine):
        emissions = [
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("10000")),
        ]
        inp = CarbonFootprintInput(
            battery_id="BAT-HIGH-CLASS",
            category=BatteryCategory.EV,
            chemistry=BatteryChemistry.NMC,
            energy_kwh=Decimal("75"),
            weight_kg=Decimal("450"),
            lifecycle_emissions=emissions,
        )
        result = engine.calculate_footprint(inp)
        assert len(result.recommendations) > 0

    def test_compliant_battery_less_recommendations(self, engine):
        emissions = [
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("1500")),
            LifecycleEmissions(stage=LifecycleStage.MANUFACTURING, co2e_kg=Decimal("500")),
            LifecycleEmissions(stage=LifecycleStage.DISTRIBUTION, co2e_kg=Decimal("100")),
            LifecycleEmissions(stage=LifecycleStage.END_OF_LIFE, co2e_kg=Decimal("100")),
        ]
        inp = CarbonFootprintInput(
            battery_id="BAT-GOOD",
            category=BatteryCategory.EV,
            chemistry=BatteryChemistry.LFP,
            energy_kwh=Decimal("75"),
            weight_kg=Decimal("400"),
            lifecycle_emissions=emissions,
        )
        result = engine.calculate_footprint(inp)
        assert isinstance(result.recommendations, list)


# ---------------------------------------------------------------------------
# Test: Deterministic Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_input_same_result(self, engine, basic_input):
        r1 = engine.calculate_footprint(basic_input)
        r2 = engine.calculate_footprint(basic_input)
        assert r1.total_co2e_kg == r2.total_co2e_kg
        assert r1.per_kwh_co2e_kg == r2.per_kwh_co2e_kg
        assert r1.performance_class == r2.performance_class

    def test_calculation_accuracy_nmc(self, engine):
        emissions = [
            LifecycleEmissions(stage=LifecycleStage.RAW_MATERIAL_EXTRACTION, co2e_kg=Decimal("2812.50")),
            LifecycleEmissions(stage=LifecycleStage.MANUFACTURING, co2e_kg=Decimal("1406.25")),
            LifecycleEmissions(stage=LifecycleStage.DISTRIBUTION, co2e_kg=Decimal("281.25")),
            LifecycleEmissions(stage=LifecycleStage.END_OF_LIFE, co2e_kg=Decimal("1125.00")),
        ]
        inp = CarbonFootprintInput(
            battery_id="BAT-ACCURACY",
            category=BatteryCategory.EV,
            chemistry=BatteryChemistry.NMC811,
            energy_kwh=Decimal("75"),
            weight_kg=Decimal("450"),
            lifecycle_emissions=emissions,
        )
        result = engine.calculate_footprint(inp)
        assert result.total_co2e_kg == Decimal("5625.000")
        expected_per_kwh = _round_val(Decimal("5625") / Decimal("75"), 3)
        assert result.per_kwh_co2e_kg == expected_per_kwh
