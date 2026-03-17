# -*- coding: utf-8 -*-
"""
Tests for EndOfLifeEngine - PACK-020 Engine 7
================================================

Comprehensive tests for end-of-life collection, recycling, and
material recovery per EU Battery Regulation Art 56-71.

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


mod = _load_module("end_of_life_engine.py", "pack020_test.engines.end_of_life")

EndOfLifeEngine = mod.EndOfLifeEngine
CollectionData = mod.CollectionData
RecyclingData = mod.RecyclingData
MaterialRecoveryData = mod.MaterialRecoveryData
SecondLifeAssessment = mod.SecondLifeAssessment
EOLResult = mod.EOLResult
BatteryCategory = mod.BatteryCategory
RecoveryMaterial = mod.RecoveryMaterial
EOLPhase = mod.EOLPhase
BatteryChemistry = mod.BatteryChemistry
COLLECTION_TARGETS = mod.COLLECTION_TARGETS
MATERIAL_RECOVERY_TARGETS = mod.MATERIAL_RECOVERY_TARGETS
RECYCLING_EFFICIENCY_TARGETS = mod.RECYCLING_EFFICIENCY_TARGETS
EOL_PHASE_DESCRIPTIONS = mod.EOL_PHASE_DESCRIPTIONS
_compute_hash = mod._compute_hash
_safe_divide = mod._safe_divide
_round2 = mod._round2
_round3 = mod._round3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return EndOfLifeEngine()


@pytest.fixture
def portable_collection_meeting_target():
    """Portable batteries meeting 2027 target (63%)."""
    return CollectionData(
        category=BatteryCategory.PORTABLE,
        batteries_placed=10000.0,
        batteries_collected=6500.0,
        year=2027,
    )


@pytest.fixture
def portable_collection_missing_target():
    """Portable batteries missing 2027 target (63%)."""
    return CollectionData(
        category=BatteryCategory.PORTABLE,
        batteries_placed=10000.0,
        batteries_collected=5000.0,
        year=2027,
    )


@pytest.fixture
def lithium_recycling_meeting():
    """Lithium-ion recycling meeting 2027 target (65%)."""
    return RecyclingData(
        chemistry=BatteryChemistry.LITHIUM_ION,
        input_weight_kg=1000.0,
        output_weight_kg=700.0,
        year=2027,
    )


@pytest.fixture
def cobalt_recovery_meeting():
    """Cobalt recovery meeting 2027 target (90%)."""
    return MaterialRecoveryData(
        material=RecoveryMaterial.COBALT,
        input_kg=100.0,
        recovered_kg=92.0,
        year=2027,
    )


@pytest.fixture
def lithium_recovery_missing():
    """Lithium recovery missing 2027 target (50%)."""
    return MaterialRecoveryData(
        material=RecoveryMaterial.LITHIUM,
        input_kg=100.0,
        recovered_kg=40.0,
        year=2027,
    )


# ---------------------------------------------------------------------------
# Test: Engine Initialization
# ---------------------------------------------------------------------------


class TestEndOfLifeEngineInit:
    def test_init_creates_engine(self):
        engine = EndOfLifeEngine()
        assert engine is not None

    def test_engine_version(self):
        engine = EndOfLifeEngine()
        assert engine.engine_version == "1.0.0"


# ---------------------------------------------------------------------------
# Test: Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_battery_category_values(self):
        assert BatteryCategory.PORTABLE.value == "portable"
        assert BatteryCategory.LMT.value == "lmt"
        assert BatteryCategory.SLI.value == "sli"
        assert BatteryCategory.EV.value == "ev"
        assert BatteryCategory.INDUSTRIAL.value == "industrial"

    def test_battery_category_count(self):
        assert len(BatteryCategory) == 5

    def test_recovery_material_values(self):
        assert RecoveryMaterial.COBALT.value == "cobalt"
        assert RecoveryMaterial.LITHIUM.value == "lithium"
        assert RecoveryMaterial.NICKEL.value == "nickel"
        assert RecoveryMaterial.COPPER.value == "copper"
        assert RecoveryMaterial.LEAD.value == "lead"

    def test_recovery_material_count(self):
        assert len(RecoveryMaterial) == 5

    def test_eol_phase_values(self):
        assert EOLPhase.COLLECTION.value == "collection"
        assert EOLPhase.DISMANTLING.value == "dismantling"
        assert EOLPhase.RECYCLING.value == "recycling"
        assert EOLPhase.RECOVERY.value == "recovery"
        assert EOLPhase.DISPOSAL.value == "disposal"

    def test_eol_phase_count(self):
        assert len(EOLPhase) == 5

    def test_battery_chemistry_values(self):
        assert BatteryChemistry.LEAD_ACID.value == "lead_acid"
        assert BatteryChemistry.LITHIUM_ION.value == "lithium_ion"
        assert BatteryChemistry.NICKEL_METAL_HYDRIDE.value == "nickel_metal_hydride"
        assert BatteryChemistry.OTHER.value == "other"

    def test_battery_chemistry_count(self):
        assert len(BatteryChemistry) == 4


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_collection_targets_has_all_categories(self):
        for cat in BatteryCategory:
            assert cat.value in COLLECTION_TARGETS

    def test_portable_2027_target_is_63(self):
        assert COLLECTION_TARGETS["portable"][2027] == 63.0

    def test_portable_2030_target_is_73(self):
        assert COLLECTION_TARGETS["portable"][2030] == 73.0

    def test_lmt_2028_target_is_51(self):
        assert COLLECTION_TARGETS["lmt"][2028] == 51.0

    def test_lmt_2031_target_is_61(self):
        assert COLLECTION_TARGETS["lmt"][2031] == 61.0

    def test_ev_always_100(self):
        for year in COLLECTION_TARGETS["ev"]:
            assert COLLECTION_TARGETS["ev"][year] == 100.0

    def test_sli_always_100(self):
        for year in COLLECTION_TARGETS["sli"]:
            assert COLLECTION_TARGETS["sli"][year] == 100.0

    def test_industrial_always_100(self):
        for year in COLLECTION_TARGETS["industrial"]:
            assert COLLECTION_TARGETS["industrial"][year] == 100.0

    def test_recovery_targets_all_materials(self):
        for mat in RecoveryMaterial:
            assert mat.value in MATERIAL_RECOVERY_TARGETS

    def test_lithium_2027_target_50(self):
        assert MATERIAL_RECOVERY_TARGETS["lithium"][2027] == 50.0

    def test_lithium_2031_target_80(self):
        assert MATERIAL_RECOVERY_TARGETS["lithium"][2031] == 80.0

    def test_cobalt_2027_target_90(self):
        assert MATERIAL_RECOVERY_TARGETS["cobalt"][2027] == 90.0

    def test_cobalt_2031_target_95(self):
        assert MATERIAL_RECOVERY_TARGETS["cobalt"][2031] == 95.0

    def test_recycling_efficiency_all_chemistries(self):
        for chem in BatteryChemistry:
            assert chem.value in RECYCLING_EFFICIENCY_TARGETS

    def test_lead_acid_2025_target_75(self):
        assert RECYCLING_EFFICIENCY_TARGETS["lead_acid"][2025] == 75.0

    def test_lithium_ion_2025_target_65(self):
        assert RECYCLING_EFFICIENCY_TARGETS["lithium_ion"][2025] == 65.0

    def test_lithium_ion_2030_target_70(self):
        assert RECYCLING_EFFICIENCY_TARGETS["lithium_ion"][2030] == 70.0

    def test_eol_phase_descriptions_all_phases(self):
        for phase in EOLPhase:
            assert phase.value in EOL_PHASE_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Test: Collection Target Checking
# ---------------------------------------------------------------------------


class TestCollectionTargets:
    def test_meeting_target(self, engine, portable_collection_meeting_target):
        results = engine.check_collection_targets(
            [portable_collection_meeting_target], 2027
        )
        assert len(results) == 1
        r = results[0]
        assert r.collection_rate_pct == 65.0  # 6500/10000 * 100
        assert r.target_rate_pct == 63.0
        assert r.meets_target is True
        assert r.gap_pct == 2.0

    def test_missing_target(self, engine, portable_collection_missing_target):
        results = engine.check_collection_targets(
            [portable_collection_missing_target], 2027
        )
        r = results[0]
        assert r.collection_rate_pct == 50.0
        assert r.target_rate_pct == 63.0
        assert r.meets_target is False
        assert r.gap_pct == -13.0

    def test_zero_placed_zero_rate(self, engine):
        data = CollectionData(
            category=BatteryCategory.PORTABLE,
            batteries_placed=0.0,
            batteries_collected=0.0,
            year=2027,
        )
        results = engine.check_collection_targets([data], 2027)
        assert results[0].collection_rate_pct == 0.0

    def test_ev_100pct_target(self, engine):
        data = CollectionData(
            category=BatteryCategory.EV,
            batteries_placed=500.0,
            batteries_collected=500.0,
            year=2027,
        )
        results = engine.check_collection_targets([data], 2027)
        assert results[0].meets_target is True
        assert results[0].target_rate_pct == 100.0

    def test_provenance_hash_on_collection(self, engine, portable_collection_meeting_target):
        results = engine.check_collection_targets(
            [portable_collection_meeting_target], 2027
        )
        assert results[0].provenance_hash != ""
        assert len(results[0].provenance_hash) == 64

    def test_empty_list(self, engine):
        results = engine.check_collection_targets([], 2027)
        assert results == []


# ---------------------------------------------------------------------------
# Test: Recycling Efficiency
# ---------------------------------------------------------------------------


class TestRecyclingEfficiency:
    def test_lithium_ion_meeting_target(self, engine, lithium_recycling_meeting):
        results = engine.check_recycling_efficiency(
            [lithium_recycling_meeting], 2027
        )
        r = results[0]
        assert r.efficiency_pct == 70.0  # 700/1000 * 100
        assert r.target_pct == 65.0
        assert r.meets_target is True

    def test_lithium_ion_missing_target(self, engine):
        data = RecyclingData(
            chemistry=BatteryChemistry.LITHIUM_ION,
            input_weight_kg=1000.0,
            output_weight_kg=600.0,
            year=2027,
        )
        results = engine.check_recycling_efficiency([data], 2027)
        r = results[0]
        assert r.efficiency_pct == 60.0
        assert r.meets_target is False

    def test_lead_acid_target(self, engine):
        data = RecyclingData(
            chemistry=BatteryChemistry.LEAD_ACID,
            input_weight_kg=1000.0,
            output_weight_kg=800.0,
            year=2027,
        )
        results = engine.check_recycling_efficiency([data], 2027)
        r = results[0]
        assert r.target_pct == 75.0
        assert r.meets_target is True

    def test_zero_input_zero_efficiency(self, engine):
        data = RecyclingData(
            chemistry=BatteryChemistry.LITHIUM_ION,
            input_weight_kg=0.0,
            output_weight_kg=0.0,
            year=2027,
        )
        results = engine.check_recycling_efficiency([data], 2027)
        assert results[0].efficiency_pct == 0.0

    def test_provenance_hash_on_recycling(self, engine, lithium_recycling_meeting):
        results = engine.check_recycling_efficiency(
            [lithium_recycling_meeting], 2027
        )
        assert results[0].provenance_hash != ""
        assert len(results[0].provenance_hash) == 64

    def test_empty_list(self, engine):
        results = engine.check_recycling_efficiency([], 2027)
        assert results == []


# ---------------------------------------------------------------------------
# Test: Material Recovery
# ---------------------------------------------------------------------------


class TestMaterialRecovery:
    def test_cobalt_meeting_target(self, engine, cobalt_recovery_meeting):
        results = engine.check_material_recovery(
            [cobalt_recovery_meeting], 2027
        )
        r = results[0]
        assert r.recovery_pct == 92.0  # 92/100 * 100
        assert r.target_pct == 90.0
        assert r.meets_target is True

    def test_lithium_missing_target(self, engine, lithium_recovery_missing):
        results = engine.check_material_recovery(
            [lithium_recovery_missing], 2027
        )
        r = results[0]
        assert r.recovery_pct == 40.0
        assert r.target_pct == 50.0
        assert r.meets_target is False
        assert r.gap_pct == -10.0

    def test_zero_input_zero_recovery(self, engine):
        data = MaterialRecoveryData(
            material=RecoveryMaterial.COBALT,
            input_kg=0.0,
            recovered_kg=0.0,
            year=2027,
        )
        results = engine.check_material_recovery([data], 2027)
        assert results[0].recovery_pct == 0.0

    def test_all_materials_can_be_assessed(self, engine):
        data_list = [
            MaterialRecoveryData(
                material=mat,
                input_kg=100.0,
                recovered_kg=95.0,
                year=2027,
            )
            for mat in RecoveryMaterial
        ]
        results = engine.check_material_recovery(data_list, 2027)
        assert len(results) == 5

    def test_provenance_hash_on_recovery(self, engine, cobalt_recovery_meeting):
        results = engine.check_material_recovery(
            [cobalt_recovery_meeting], 2027
        )
        assert results[0].provenance_hash != ""
        assert len(results[0].provenance_hash) == 64

    def test_empty_list(self, engine):
        results = engine.check_material_recovery([], 2027)
        assert results == []


# ---------------------------------------------------------------------------
# Test: Second-Life Assessment
# ---------------------------------------------------------------------------


class TestSecondLifeAssessment:
    def test_high_soh_suitable(self, engine):
        result = engine.assess_second_life("BAT-001", 85.0)
        assert result.suitable_for_second_life is True
        assert result.estimated_remaining_life_years == 8.0
        assert "grid-scale" in result.recommended_application.lower() or \
               "stationary" in result.recommended_application.lower()

    def test_good_soh_suitable(self, engine):
        result = engine.assess_second_life("BAT-002", 75.0)
        assert result.suitable_for_second_life is True
        assert result.estimated_remaining_life_years == 5.0

    def test_marginal_soh_not_suitable(self, engine):
        result = engine.assess_second_life("BAT-003", 55.0)
        assert result.suitable_for_second_life is False
        assert result.estimated_remaining_life_years == 2.0

    def test_low_soh_not_suitable(self, engine):
        result = engine.assess_second_life("BAT-004", 30.0)
        assert result.suitable_for_second_life is False
        assert result.estimated_remaining_life_years == 0.0
        assert "recycling" in result.recommended_application.lower()

    def test_boundary_80(self, engine):
        result = engine.assess_second_life("BAT-B80", 80.0)
        assert result.suitable_for_second_life is True
        assert result.estimated_remaining_life_years == 8.0

    def test_boundary_70(self, engine):
        result = engine.assess_second_life("BAT-B70", 70.0)
        assert result.suitable_for_second_life is True
        assert result.estimated_remaining_life_years == 5.0

    def test_boundary_50(self, engine):
        result = engine.assess_second_life("BAT-B50", 50.0)
        assert result.suitable_for_second_life is False
        assert result.estimated_remaining_life_years == 2.0

    def test_boundary_49(self, engine):
        result = engine.assess_second_life("BAT-B49", 49.0)
        assert result.suitable_for_second_life is False
        assert result.estimated_remaining_life_years == 0.0

    def test_battery_id_set(self, engine):
        result = engine.assess_second_life("MY-BAT-123", 90.0)
        assert result.battery_id == "MY-BAT-123"

    def test_soh_100(self, engine):
        result = engine.assess_second_life("BAT-100", 100.0)
        assert result.suitable_for_second_life is True

    def test_soh_0(self, engine):
        result = engine.assess_second_life("BAT-0", 0.0)
        assert result.suitable_for_second_life is False
        assert result.estimated_remaining_life_years == 0.0

    def test_assessment_has_notes(self, engine):
        result = engine.assess_second_life("BAT-001", 85.0)
        assert result.assessment_notes != ""


# ---------------------------------------------------------------------------
# Test: Full End-of-Life Assessment
# ---------------------------------------------------------------------------


class TestAssessEndOfLife:
    def test_all_compliant(
        self, engine, portable_collection_meeting_target,
        lithium_recycling_meeting, cobalt_recovery_meeting
    ):
        result = engine.assess_end_of_life(
            collection_data=[portable_collection_meeting_target],
            recycling_data=[lithium_recycling_meeting],
            recovery_data=[cobalt_recovery_meeting],
            year=2027,
        )
        assert isinstance(result, EOLResult)
        assert result.collection_compliant is True
        assert result.recycling_compliant is True
        assert result.recovery_compliant is True
        assert result.overall_compliance is True

    def test_collection_non_compliant(
        self, engine, portable_collection_missing_target,
        lithium_recycling_meeting, cobalt_recovery_meeting
    ):
        result = engine.assess_end_of_life(
            collection_data=[portable_collection_missing_target],
            recycling_data=[lithium_recycling_meeting],
            recovery_data=[cobalt_recovery_meeting],
            year=2027,
        )
        assert result.collection_compliant is False
        assert result.overall_compliance is False

    def test_recovery_non_compliant(
        self, engine, portable_collection_meeting_target,
        lithium_recycling_meeting, lithium_recovery_missing
    ):
        result = engine.assess_end_of_life(
            collection_data=[portable_collection_meeting_target],
            recycling_data=[lithium_recycling_meeting],
            recovery_data=[lithium_recovery_missing],
            year=2027,
        )
        assert result.recovery_compliant is False
        assert result.overall_compliance is False

    def test_empty_data_is_compliant(self, engine):
        result = engine.assess_end_of_life([], [], [], year=2027)
        assert result.overall_compliance is True
        assert result.collection_compliant is True
        assert result.recycling_compliant is True
        assert result.recovery_compliant is True

    def test_provenance_hash(self, engine, portable_collection_meeting_target):
        result = engine.assess_end_of_life(
            [portable_collection_meeting_target], [], [], year=2027
        )
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_processing_time(self, engine, portable_collection_meeting_target):
        result = engine.assess_end_of_life(
            [portable_collection_meeting_target], [], [], year=2027
        )
        assert result.processing_time_ms >= 0.0

    def test_total_collected_kg(self, engine):
        c1 = CollectionData(
            category=BatteryCategory.PORTABLE,
            batteries_placed=1000.0,
            batteries_collected=500.0,
            year=2027,
        )
        c2 = CollectionData(
            category=BatteryCategory.EV,
            batteries_placed=2000.0,
            batteries_collected=2000.0,
            year=2027,
        )
        result = engine.assess_end_of_life([c1, c2], [], [], year=2027)
        assert result.total_collected_kg == 2500.0

    def test_total_recycled_kg(self, engine, lithium_recycling_meeting):
        result = engine.assess_end_of_life(
            [], [lithium_recycling_meeting], [], year=2027
        )
        assert result.total_recycled_kg == 700.0

    def test_second_life_assessment_included(self, engine):
        result = engine.assess_end_of_life(
            [], [], [], year=2027,
            second_life_battery_id="BAT-SL-001",
            second_life_soh_pct=85.0,
        )
        assert result.second_life_assessment is not None
        assert result.second_life_assessment.suitable_for_second_life is True

    def test_second_life_not_included_when_not_provided(self, engine):
        result = engine.assess_end_of_life([], [], [], year=2027)
        assert result.second_life_assessment is None

    def test_recommendations_generated(
        self, engine, portable_collection_missing_target
    ):
        result = engine.assess_end_of_life(
            [portable_collection_missing_target], [], [], year=2027
        )
        assert len(result.recommendations) > 0


# ---------------------------------------------------------------------------
# Test: Target Lookup Utilities
# ---------------------------------------------------------------------------


class TestTargetLookups:
    def test_get_collection_target(self, engine):
        result = engine.get_collection_target(BatteryCategory.PORTABLE, 2027)
        assert result["target_pct"] == 63.0
        assert result["is_take_back"] is False
        assert "provenance_hash" in result

    def test_get_collection_target_ev_is_take_back(self, engine):
        result = engine.get_collection_target(BatteryCategory.EV, 2027)
        assert result["is_take_back"] is True
        assert result["target_pct"] == 100.0

    def test_get_recovery_target(self, engine):
        result = engine.get_recovery_target(RecoveryMaterial.LITHIUM, 2027)
        assert result["target_pct"] == 50.0
        assert "provenance_hash" in result

    def test_get_recycling_target(self, engine):
        result = engine.get_recycling_target(BatteryChemistry.LITHIUM_ION, 2027)
        assert result["target_pct"] == 65.0
        assert "provenance_hash" in result

    def test_get_all_targets_for_year(self, engine):
        result = engine.get_all_targets_for_year(2027)
        assert "collection" in result
        assert "recycling" in result
        assert "recovery" in result
        assert result["year"] == 2027
        assert "provenance_hash" in result

    def test_all_targets_for_year_collection_all_categories(self, engine):
        result = engine.get_all_targets_for_year(2027)
        for cat in BatteryCategory:
            assert cat.value in result["collection"]

    def test_all_targets_for_year_recovery_all_materials(self, engine):
        result = engine.get_all_targets_for_year(2027)
        for mat in RecoveryMaterial:
            assert mat.value in result["recovery"]


# ---------------------------------------------------------------------------
# Test: Private Target Lookups (edge cases)
# ---------------------------------------------------------------------------


class TestPrivateTargetLookups:
    def test_collection_exact_year(self, engine):
        assert engine._get_collection_target(BatteryCategory.PORTABLE, 2027) == 63.0

    def test_collection_future_year_uses_latest(self, engine):
        # 2040 not in table; latest <= 2040 is 2031 with value 73.0
        assert engine._get_collection_target(BatteryCategory.PORTABLE, 2040) == 73.0

    def test_collection_early_year_uses_earliest(self, engine):
        # 2020 before any target => returns earliest (2024: 45.0)
        assert engine._get_collection_target(BatteryCategory.PORTABLE, 2020) == 45.0

    def test_recycling_unknown_chemistry_default_50(self, engine):
        # If chemistry not in targets, returns 50.0
        # OTHER chemistry has entries, so test with existing value
        assert engine._get_recycling_target(BatteryChemistry.OTHER, 2027) == 50.0

    def test_recovery_before_2027_returns_zero(self, engine):
        # Lithium recovery starts at 2027; year 2025 should be 0.0
        assert engine._get_recovery_target(RecoveryMaterial.LITHIUM, 2025) == 0.0

    def test_recovery_future_year(self, engine):
        # 2040 not in table; latest <= 2040 is 2031 with value 80.0
        assert engine._get_recovery_target(RecoveryMaterial.LITHIUM, 2040) == 80.0


# ---------------------------------------------------------------------------
# Test: Gap Analysis
# ---------------------------------------------------------------------------


class TestGapAnalysis:
    def test_no_gaps_all_compliant(
        self, engine, portable_collection_meeting_target,
        lithium_recycling_meeting, cobalt_recovery_meeting
    ):
        result = engine.assess_end_of_life(
            [portable_collection_meeting_target],
            [lithium_recycling_meeting],
            [cobalt_recovery_meeting],
            year=2027,
        )
        gaps = engine.gap_analysis(result)
        assert gaps["total_gaps"] == 0
        assert gaps["collection_gaps"] == []
        assert gaps["recycling_gaps"] == []
        assert gaps["recovery_gaps"] == []

    def test_collection_gap_detected(
        self, engine, portable_collection_missing_target
    ):
        result = engine.assess_end_of_life(
            [portable_collection_missing_target], [], [], year=2027
        )
        gaps = engine.gap_analysis(result)
        assert gaps["total_gaps"] > 0
        assert len(gaps["collection_gaps"]) == 1
        assert gaps["collection_gaps"][0]["gap_pct"] < 0

    def test_recovery_gap_detected(self, engine, lithium_recovery_missing):
        result = engine.assess_end_of_life(
            [], [], [lithium_recovery_missing], year=2027
        )
        gaps = engine.gap_analysis(result)
        assert gaps["total_gaps"] > 0
        assert len(gaps["recovery_gaps"]) == 1

    def test_gap_additional_kg_needed(
        self, engine, portable_collection_missing_target
    ):
        result = engine.assess_end_of_life(
            [portable_collection_missing_target], [], [], year=2027
        )
        gaps = engine.gap_analysis(result)
        gap = gaps["collection_gaps"][0]
        assert gap["additional_kg_needed"] > 0

    def test_gap_provenance_hash(
        self, engine, portable_collection_meeting_target
    ):
        result = engine.assess_end_of_life(
            [portable_collection_meeting_target], [], [], year=2027
        )
        gaps = engine.gap_analysis(result)
        assert "provenance_hash" in gaps
        assert len(gaps["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# Test: EOL Summary
# ---------------------------------------------------------------------------


class TestEOLSummary:
    def test_summary_structure(
        self, engine, portable_collection_meeting_target
    ):
        result = engine.assess_end_of_life(
            [portable_collection_meeting_target], [], [], year=2027
        )
        summary = engine.get_eol_summary(result)
        assert "overall_compliance" in summary
        assert "collection_compliant" in summary
        assert "recycling_compliant" in summary
        assert "recovery_compliant" in summary
        assert "total_collected_kg" in summary
        assert "provenance_hash" in summary

    def test_summary_with_second_life(self, engine):
        result = engine.assess_end_of_life(
            [], [], [], year=2027,
            second_life_battery_id="BAT-001",
            second_life_soh_pct=85.0,
        )
        summary = engine.get_eol_summary(result)
        assert summary["has_second_life_assessment"] is True
        assert summary["second_life_suitable"] is True

    def test_summary_without_second_life(self, engine):
        result = engine.assess_end_of_life([], [], [], year=2027)
        summary = engine.get_eol_summary(result)
        assert summary["has_second_life_assessment"] is False
        assert summary["second_life_suitable"] is None


# ---------------------------------------------------------------------------
# Test: Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_all_met_positive_message(self, engine):
        c = CollectionData(
            category=BatteryCategory.PORTABLE,
            batteries_placed=10000.0,
            batteries_collected=7500.0,
            year=2027,
        )
        result = engine.assess_end_of_life([c], [], [], year=2027)
        assert any("met" in r.lower() for r in result.recommendations)

    def test_collection_gap_recommendation(
        self, engine, portable_collection_missing_target
    ):
        result = engine.assess_end_of_life(
            [portable_collection_missing_target], [], [], year=2027
        )
        assert any("collection gap" in r.lower() for r in result.recommendations)

    def test_lithium_recovery_specific_recommendation(
        self, engine, lithium_recovery_missing
    ):
        result = engine.assess_end_of_life(
            [], [], [lithium_recovery_missing], year=2027
        )
        assert any("lithium" in r.lower() for r in result.recommendations)

    def test_future_target_warning(self, engine):
        result = engine.assess_end_of_life([], [], [], year=2028)
        assert any("2031" in r for r in result.recommendations)

    def test_no_future_warning_for_2031(self, engine):
        c = CollectionData(
            category=BatteryCategory.PORTABLE,
            batteries_placed=10000.0,
            batteries_collected=7500.0,
            year=2031,
        )
        result = engine.assess_end_of_life([c], [], [], year=2031)
        # Year >= 2031: no future target warning
        future_warnings = [r for r in result.recommendations if "Targets increase" in r]
        assert len(future_warnings) == 0


# ---------------------------------------------------------------------------
# Test: Edge Cases / Precision
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_collection_rate_exactly_at_target(self, engine):
        data = CollectionData(
            category=BatteryCategory.PORTABLE,
            batteries_placed=10000.0,
            batteries_collected=6300.0,
            year=2027,
        )
        results = engine.check_collection_targets([data], 2027)
        assert results[0].collection_rate_pct == 63.0
        assert results[0].meets_target is True
        assert results[0].gap_pct == 0.0

    def test_recovery_exactly_at_target(self, engine):
        data = MaterialRecoveryData(
            material=RecoveryMaterial.COBALT,
            input_kg=100.0,
            recovered_kg=90.0,
            year=2027,
        )
        results = engine.check_material_recovery([data], 2027)
        assert results[0].recovery_pct == 90.0
        assert results[0].meets_target is True

    def test_recycling_exactly_at_target(self, engine):
        data = RecyclingData(
            chemistry=BatteryChemistry.LITHIUM_ION,
            input_weight_kg=1000.0,
            output_weight_kg=650.0,
            year=2027,
        )
        results = engine.check_recycling_efficiency([data], 2027)
        assert results[0].efficiency_pct == 65.0
        assert results[0].meets_target is True

    def test_very_small_values(self, engine):
        data = CollectionData(
            category=BatteryCategory.PORTABLE,
            batteries_placed=0.001,
            batteries_collected=0.001,
            year=2027,
        )
        results = engine.check_collection_targets([data], 2027)
        assert results[0].collection_rate_pct == 100.0
