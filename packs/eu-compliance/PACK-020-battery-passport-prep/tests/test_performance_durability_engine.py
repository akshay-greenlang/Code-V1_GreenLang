# -*- coding: utf-8 -*-
"""
Tests for PerformanceDurabilityEngine - PACK-020 Engine 4
==========================================================

Comprehensive tests for battery performance and durability
assessment per EU Battery Regulation Art 10 and Annex IV.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-020 Battery Passport Prep
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = PACK_ROOT / "engines"


def _load_module(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ENGINE_DIR / file_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module("performance_durability_engine.py", "pack020_test.engines.performance_durability")

PerformanceDurabilityEngine = mod.PerformanceDurabilityEngine
PerformanceInput = mod.PerformanceInput
PerformanceResult = mod.PerformanceResult
SoHAssessment = mod.SoHAssessment
CycleLifeAssessment = mod.CycleLifeAssessment
EfficiencyAssessment = mod.EfficiencyAssessment
MetricValidation = mod.MetricValidation
PerformanceMetric = mod.PerformanceMetric
DurabilityRating = mod.DurabilityRating
MetricStatus = mod.MetricStatus
BatteryLifeStage = mod.BatteryLifeStage
SOH_THRESHOLDS = mod.SOH_THRESHOLDS
MIN_EFFICIENCY_THRESHOLDS = mod.MIN_EFFICIENCY_THRESHOLDS
CHEMISTRY_CYCLE_LIFE = mod.CHEMISTRY_CYCLE_LIFE
METRIC_LABELS = mod.METRIC_LABELS


@pytest.fixture
def engine():
    return PerformanceDurabilityEngine()


@pytest.fixture
def full_input():
    return PerformanceInput(
        battery_id="BAT-PERF-001",
        category="ev",
        chemistry="nmc811",
        rated_capacity_ah=Decimal("100"),
        current_capacity_ah=Decimal("92"),
        min_capacity_ah=Decimal("80"),
        voltage_nominal=Decimal("400"),
        voltage_min=Decimal("320"),
        voltage_max=Decimal("440"),
        power_capability_w=Decimal("150000"),
        cycle_life_expected=1500,
        cycles_completed=300,
        calendar_life_years=Decimal("10"),
        age_years=Decimal("1"),
        efficiency_pct=Decimal("93.5"),
        internal_resistance_mohm=Decimal("55"),
        initial_resistance_mohm=Decimal("45"),
        soh_pct=Decimal("92"),
        soc_pct=Decimal("80"),
        c_rate_max=Decimal("2.0"),
        temperature_min=Decimal("-20"),
        temperature_max=Decimal("45"),
        energy_capacity_kwh=Decimal("75"),
        weight_kg=Decimal("450"),
    )


@pytest.fixture
def minimal_input():
    return PerformanceInput(battery_id="BAT-MIN-001")


class TestEngineInit:
    def test_init(self):
        engine = PerformanceDurabilityEngine()
        assert engine.engine_version == "1.0.0"

    def test_results_empty(self):
        engine = PerformanceDurabilityEngine()
        assert engine.get_results() == []


class TestEnums:
    def test_performance_metrics(self):
        assert len(PerformanceMetric) == 15

    def test_durability_ratings(self):
        assert len(DurabilityRating) == 5

    def test_metric_statuses(self):
        assert len(MetricStatus) == 4

    def test_life_stages(self):
        assert len(BatteryLifeStage) == 5


class TestConstants:
    def test_soh_thresholds(self):
        assert SOH_THRESHOLDS["excellent"] == Decimal("95")
        assert SOH_THRESHOLDS["good"] == Decimal("85")
        assert SOH_THRESHOLDS["acceptable"] == Decimal("75")
        assert SOH_THRESHOLDS["poor"] == Decimal("65")

    def test_efficiency_thresholds(self):
        assert MIN_EFFICIENCY_THRESHOLDS["ev"] == Decimal("85")

    def test_chemistry_cycle_life(self):
        assert "nmc811" in CHEMISTRY_CYCLE_LIFE
        assert CHEMISTRY_CYCLE_LIFE["lfp"]["typical"] == 4000


class TestCalculateSoH:
    @pytest.mark.parametrize("initial,current,expected", [
        (Decimal("100"), Decimal("92"), Decimal("92.00")),
        (Decimal("100"), Decimal("100"), Decimal("100.00")),
        (Decimal("100"), Decimal("50"), Decimal("50.00")),
        (Decimal("100"), Decimal("0"), Decimal("0.00")),
        (Decimal("100"), Decimal("105"), Decimal("100.00")),  # capped at 100
        (Decimal("0"), Decimal("50"), Decimal("0.00")),  # zero initial
    ])
    def test_soh_calculation(self, engine, initial, current, expected):
        assert engine.calculate_soh(initial, current) == expected


class TestDurabilityRating:
    @pytest.mark.parametrize("soh,expected_rating", [
        (Decimal("100"), DurabilityRating.EXCELLENT),
        (Decimal("95"), DurabilityRating.EXCELLENT),
        (Decimal("94"), DurabilityRating.GOOD),
        (Decimal("85"), DurabilityRating.GOOD),
        (Decimal("84"), DurabilityRating.ACCEPTABLE),
        (Decimal("75"), DurabilityRating.ACCEPTABLE),
        (Decimal("74"), DurabilityRating.POOR),
        (Decimal("65"), DurabilityRating.POOR),
        (Decimal("64"), DurabilityRating.CRITICAL),
        (Decimal("0"), DurabilityRating.CRITICAL),
    ])
    def test_durability_from_soh(self, engine, soh, expected_rating):
        assert engine.assess_durability_from_soh(soh) == expected_rating

    def test_assess_durability_dict_with_soh(self, engine):
        result = engine.assess_durability({"soh_pct": 90})
        assert result == DurabilityRating.GOOD

    def test_assess_durability_dict_with_capacity(self, engine):
        result = engine.assess_durability({
            "rated_capacity_ah": 100,
            "current_capacity_ah": 96,
        })
        assert result == DurabilityRating.EXCELLENT

    def test_assess_durability_dict_no_data(self, engine):
        result = engine.assess_durability({})
        assert result == DurabilityRating.CRITICAL


class TestAssessPerformance:
    def test_full_assessment(self, engine, full_input):
        result = engine.assess_performance(full_input)
        assert isinstance(result, PerformanceResult)
        assert result.battery_id == "BAT-PERF-001"
        assert result.provenance_hash != ""

    def test_minimal_assessment(self, engine, minimal_input):
        result = engine.assess_performance(minimal_input)
        assert result.metrics_not_assessed > 0

    def test_soh_assessment_present(self, engine, full_input):
        result = engine.assess_performance(full_input)
        assert result.soh_assessment is not None
        assert result.soh_assessment.soh_pct == Decimal("92.00")

    def test_cycle_life_assessment(self, engine, full_input):
        result = engine.assess_performance(full_input)
        assert result.cycle_life_assessment is not None
        assert result.cycle_life_assessment.cycles_expected == 1500
        assert result.cycle_life_assessment.cycles_completed == 300

    def test_efficiency_assessment(self, engine, full_input):
        result = engine.assess_performance(full_input)
        assert result.efficiency_assessment is not None
        assert result.efficiency_assessment.above_threshold is True

    def test_compliance_status(self, engine, full_input):
        result = engine.assess_performance(full_input)
        assert result.compliance_status in ("compliant", "compliant_with_warnings", "non_compliant")

    def test_specific_power_calculated(self, engine, full_input):
        result = engine.assess_performance(full_input)
        assert result.specific_power_w_per_kg is not None

    def test_specific_energy_calculated(self, engine, full_input):
        result = engine.assess_performance(full_input)
        assert result.specific_energy_wh_per_kg is not None

    def test_resistance_increase(self, engine, full_input):
        result = engine.assess_performance(full_input)
        assert result.resistance_increase_pct is not None


class TestCycleLifeAssessment:
    def test_early_life(self, engine):
        result = engine.assess_cycle_life(300, 1500)
        assert result.status == MetricStatus.PASS
        assert result.cycles_remaining == 1200

    def test_approaching_end(self, engine):
        result = engine.assess_cycle_life(1300, 1500)
        assert result.status == MetricStatus.WARNING

    def test_exceeded_cycle_life(self, engine):
        result = engine.assess_cycle_life(1600, 1500)
        assert result.status == MetricStatus.WARNING
        assert result.cycles_remaining == 0

    def test_zero_expected(self, engine):
        result = engine.assess_cycle_life(100, 0)
        assert result.status == MetricStatus.NOT_ASSESSED


class TestValidateMetrics:
    def test_validate_all_metrics(self, engine, full_input):
        validations = engine.validate_metrics(full_input)
        assert len(validations) == 13

    def test_missing_metrics_not_assessed(self, engine, minimal_input):
        validations = engine.validate_metrics(minimal_input)
        not_assessed = [v for v in validations if v.status == MetricStatus.NOT_ASSESSED]
        assert len(not_assessed) > 5


class TestBatchProcessing:
    def test_batch_assessment(self, engine, full_input):
        results = engine.assess_batch([full_input])
        assert len(results) == 1

    def test_batch_empty(self, engine):
        results = engine.assess_batch([])
        assert results == []


class TestCompareResults:
    def test_compare_empty(self, engine):
        result = engine.compare_results([])
        assert result["count"] == 0

    def test_compare_single(self, engine, full_input):
        r = engine.assess_performance(full_input)
        comparison = engine.compare_results([r])
        assert comparison["count"] == 1
        assert "rating_distribution" in comparison


class TestBuildDocumentation:
    def test_documentation_structure(self, engine, full_input):
        result = engine.assess_performance(full_input)
        doc = engine.build_documentation(result)
        assert "document_id" in doc
        assert "metrics" in doc
        assert "provenance_hash" in doc
        assert doc["battery_id"] == "BAT-PERF-001"


class TestReferenceData:
    def test_soh_thresholds_method(self, engine):
        thresholds = engine.get_soh_thresholds()
        assert "excellent" in thresholds

    def test_metric_reference(self, engine):
        ref = engine.get_metric_reference()
        assert len(ref) == len(METRIC_LABELS)

    def test_chemistry_cycle_life_known(self, engine):
        result = engine.get_chemistry_cycle_life("lfp")
        assert result["available"] is True
        assert result["typical"] == 4000

    def test_chemistry_cycle_life_unknown(self, engine):
        result = engine.get_chemistry_cycle_life("unknown_chem")
        assert result["available"] is False


class TestRegistryManagement:
    def test_clear_results(self, engine, full_input):
        engine.assess_performance(full_input)
        assert len(engine.get_results()) == 1
        engine.clear_results()
        assert len(engine.get_results()) == 0


class TestInputValidation:
    def test_empty_battery_id(self, engine):
        with pytest.raises(Exception):
            PerformanceInput(battery_id="")

    def test_voltage_min_exceeds_max(self, engine):
        inp = PerformanceInput(
            battery_id="BAT-V",
            voltage_min=Decimal("500"),
            voltage_max=Decimal("400"),
        )
        with pytest.raises(ValueError):
            engine.assess_performance(inp)
