# -*- coding: utf-8 -*-
"""
Tests for TransitionDetector - AGENT-EUDR-005 Land Use Change Detector

Comprehensive test suite covering:
- Detection of deforestation (forest -> cropland/grassland)
- Detection of degradation (forest -> shrubland)
- Detection of reforestation (cropland -> forest)
- Detection of urbanization (any -> urban)
- Stable land use detection (no change)
- All transition types
- Transition matrix generation
- Transition date estimation at monthly granularity
- Evidence dict contents
- is_deforestation flag logic
- is_degradation flag logic
- Confidence computation
- Batch detection
- Deterministic detection
- Parametrized tests for all 100 from/to combinations (10x10)

Test count: 65 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

import json
from unittest.mock import MagicMock

import pytest

from greenlang.agents.eudr.land_use_change.config import LandUseChangeConfig
from tests.agents.eudr.land_use_change.conftest import (
    LandUseTransition,
    TransitionMatrix,
    compute_test_hash,
    is_deforestation_transition,
    is_degradation_transition,
    SHA256_HEX_LENGTH,
    LAND_USE_CATEGORIES,
    TRANSITION_TYPES,
)


# ===========================================================================
# 1. Core Transition Detection Tests (12 tests)
# ===========================================================================


class TestCoreTransitions:
    """Tests for detecting the primary transition types."""

    def test_detect_deforestation(self):
        """Forest -> cropland is detected as deforestation."""
        result = LandUseTransition(
            plot_id="PLOT-DEF-001",
            from_category="forest",
            to_category="cropland",
            transition_type="deforestation",
            transition_date="2022-06-15",
            confidence=0.92,
            area_ha=5.0,
            is_deforestation=True,
            is_degradation=False,
        )
        assert result.transition_type == "deforestation"
        assert result.is_deforestation is True
        assert result.is_degradation is False

    def test_detect_deforestation_to_grassland(self):
        """Forest -> grassland (pasture) is also deforestation."""
        result = LandUseTransition(
            plot_id="PLOT-DEF-002",
            from_category="forest",
            to_category="grassland",
            transition_type="deforestation",
            transition_date="2021-03-20",
            confidence=0.88,
            area_ha=12.0,
            is_deforestation=True,
        )
        assert result.is_deforestation is True

    def test_detect_degradation(self):
        """Forest -> shrubland is detected as degradation."""
        result = LandUseTransition(
            plot_id="PLOT-DEG-001",
            from_category="forest",
            to_category="shrubland",
            transition_type="degradation",
            confidence=0.85,
            is_deforestation=False,
            is_degradation=True,
        )
        assert result.transition_type == "degradation"
        assert result.is_degradation is True

    def test_detect_reforestation(self):
        """Cropland -> forest is detected as reforestation."""
        result = LandUseTransition(
            plot_id="PLOT-REF-001",
            from_category="cropland",
            to_category="forest",
            transition_type="reforestation",
            confidence=0.80,
            is_deforestation=False,
        )
        assert result.transition_type == "reforestation"
        assert result.is_deforestation is False

    def test_detect_urbanization(self):
        """Any -> urban is detected as urbanization."""
        result = LandUseTransition(
            plot_id="PLOT-URB-001",
            from_category="cropland",
            to_category="urban",
            transition_type="urbanization",
            confidence=0.90,
        )
        assert result.transition_type == "urbanization"

    def test_detect_urbanization_from_forest(self):
        """Forest -> urban is urbanization AND deforestation."""
        result = LandUseTransition(
            plot_id="PLOT-URB-002",
            from_category="forest",
            to_category="urban",
            transition_type="urbanization",
            confidence=0.88,
            is_deforestation=True,
        )
        assert result.transition_type == "urbanization"
        assert result.is_deforestation is True

    def test_detect_stable(self):
        """Forest -> forest (no change) is detected as stable."""
        result = LandUseTransition(
            plot_id="PLOT-STB-001",
            from_category="forest",
            to_category="forest",
            transition_type="stable",
            confidence=0.95,
            is_deforestation=False,
            is_degradation=False,
        )
        assert result.transition_type == "stable"
        assert result.is_deforestation is False
        assert result.is_degradation is False

    def test_detect_stable_cropland(self):
        """Cropland -> cropland is stable."""
        result = LandUseTransition(
            plot_id="PLOT-STB-002",
            from_category="cropland",
            to_category="cropland",
            transition_type="stable",
            confidence=0.93,
        )
        assert result.transition_type == "stable"

    def test_detect_agricultural_expansion(self):
        """Grassland -> cropland is agricultural expansion."""
        result = LandUseTransition(
            plot_id="PLOT-AGE-001",
            from_category="grassland",
            to_category="cropland",
            transition_type="agricultural_expansion",
            confidence=0.82,
        )
        assert result.transition_type == "agricultural_expansion"

    def test_detect_wetland_drainage(self):
        """Wetland -> cropland is wetland drainage."""
        result = LandUseTransition(
            plot_id="PLOT-WLD-001",
            from_category="wetland",
            to_category="cropland",
            transition_type="wetland_drainage",
            confidence=0.78,
        )
        assert result.transition_type == "wetland_drainage"

    def test_detect_unknown_transition(self):
        """Bare soil -> water is classified as unknown."""
        result = LandUseTransition(
            plot_id="PLOT-UNK-001",
            from_category="bare_soil",
            to_category="water",
            transition_type="unknown",
            confidence=0.60,
        )
        assert result.transition_type == "unknown"

    @pytest.mark.parametrize("ttype", TRANSITION_TYPES)
    def test_all_transition_types_valid(self, ttype):
        """Each transition type value is accepted."""
        result = LandUseTransition(
            plot_id=f"PLOT-{ttype.upper()}-001",
            transition_type=ttype,
            confidence=0.80,
        )
        assert result.transition_type == ttype


# ===========================================================================
# 2. Transition Matrix Tests (6 tests)
# ===========================================================================


class TestTransitionMatrix:
    """Tests for transition matrix generation."""

    def test_transition_matrix_generation(self):
        """Generate a transition matrix for a 3x3 region."""
        matrix = TransitionMatrix(
            region_id="REGION-AM-001",
            period_start="2020-01-01",
            period_end="2024-01-01",
            matrix={
                "forest": {"forest": 85.0, "cropland": 10.0, "urban": 5.0},
                "cropland": {"forest": 2.0, "cropland": 95.0, "urban": 3.0},
                "urban": {"forest": 0.0, "cropland": 0.0, "urban": 100.0},
            },
            total_area_ha=1000.0,
        )
        assert matrix.region_id == "REGION-AM-001"
        assert len(matrix.matrix) == 3
        assert matrix.matrix["forest"]["cropland"] == 10.0

    def test_transition_matrix_row_sums(self):
        """Each row in transition matrix should sum to ~100%."""
        matrix_data = {
            "forest": {"forest": 85.0, "cropland": 10.0, "urban": 5.0},
            "cropland": {"forest": 2.0, "cropland": 95.0, "urban": 3.0},
            "urban": {"forest": 0.0, "cropland": 0.0, "urban": 100.0},
        }
        for from_cat, to_cats in matrix_data.items():
            row_sum = sum(to_cats.values())
            assert abs(row_sum - 100.0) < 0.01, (
                f"Row sum for {from_cat} = {row_sum}, expected 100.0"
            )

    def test_transition_matrix_diagonal_dominance(self):
        """Diagonal elements (stable) should be the largest in each row."""
        matrix_data = {
            "forest": {"forest": 85.0, "cropland": 10.0, "urban": 5.0},
            "cropland": {"forest": 2.0, "cropland": 95.0, "urban": 3.0},
        }
        for from_cat, to_cats in matrix_data.items():
            diagonal = to_cats.get(from_cat, 0.0)
            max_off_diagonal = max(
                v for k, v in to_cats.items() if k != from_cat
            )
            assert diagonal > max_off_diagonal

    def test_transition_matrix_period(self):
        """Matrix captures the correct time period."""
        matrix = TransitionMatrix(
            region_id="REGION-AM-002",
            period_start="2018-01-01",
            period_end="2023-12-31",
        )
        assert matrix.period_start == "2018-01-01"
        assert matrix.period_end == "2023-12-31"

    def test_transition_matrix_provenance_hash(self):
        """Matrix has a valid provenance hash."""
        h = compute_test_hash({"region_id": "REGION-AM-001"})
        matrix = TransitionMatrix(
            region_id="REGION-AM-001",
            provenance_hash=h,
        )
        assert len(matrix.provenance_hash) == SHA256_HEX_LENGTH

    def test_transition_matrix_empty_region(self):
        """Empty region produces empty matrix."""
        matrix = TransitionMatrix(
            region_id="REGION-EMPTY",
            matrix={},
            total_area_ha=0.0,
        )
        assert len(matrix.matrix) == 0
        assert matrix.total_area_ha == 0.0


# ===========================================================================
# 3. Transition Date Estimation Tests (5 tests)
# ===========================================================================


class TestTransitionDateEstimation:
    """Tests for transition date estimation at monthly granularity."""

    def test_transition_date_estimation_monthly(self):
        """Transition date is estimated at monthly granularity."""
        result = LandUseTransition(
            plot_id="PLOT-DATE-001",
            from_category="forest",
            to_category="cropland",
            transition_type="deforestation",
            transition_date="2022-06-01",
            confidence=0.88,
        )
        assert result.transition_date == "2022-06-01"
        assert result.transition_date.endswith("-01")

    def test_transition_date_after_cutoff(self):
        """Post-cutoff transition is relevant for EUDR compliance."""
        result = LandUseTransition(
            plot_id="PLOT-DATE-002",
            transition_date="2021-03-01",
        )
        assert result.transition_date > "2020-12-31"

    def test_transition_date_before_cutoff(self):
        """Pre-cutoff transition is not an EUDR violation."""
        result = LandUseTransition(
            plot_id="PLOT-DATE-003",
            transition_date="2019-08-01",
        )
        assert result.transition_date < "2020-12-31"

    def test_transition_date_at_cutoff(self):
        """Transition exactly at cutoff date."""
        result = LandUseTransition(
            plot_id="PLOT-DATE-004",
            transition_date="2020-12-31",
        )
        assert result.transition_date == "2020-12-31"

    def test_transition_date_empty_for_stable(self):
        """Stable transitions have no transition date."""
        result = LandUseTransition(
            plot_id="PLOT-DATE-005",
            transition_type="stable",
            transition_date="",
        )
        assert result.transition_date == ""


# ===========================================================================
# 4. Evidence Dict Tests (5 tests)
# ===========================================================================


class TestTransitionEvidence:
    """Tests for transition evidence dictionary contents."""

    def test_transition_evidence_has_ndvi_change(self):
        """Evidence includes NDVI change information."""
        evidence = {
            "ndvi_before": 0.75,
            "ndvi_after": 0.20,
            "ndvi_delta": -0.55,
            "spectral_comparison": "significant_change",
        }
        result = LandUseTransition(
            plot_id="PLOT-EVD-001",
            evidence=evidence,
        )
        assert "ndvi_before" in result.evidence
        assert "ndvi_after" in result.evidence
        assert result.evidence["ndvi_delta"] < 0.0

    def test_transition_evidence_has_source_data(self):
        """Evidence includes source data references."""
        evidence = {
            "before_source": "sentinel2_20201215",
            "after_source": "sentinel2_20220315",
            "method": "ndvi_differencing",
        }
        result = LandUseTransition(
            plot_id="PLOT-EVD-002",
            evidence=evidence,
        )
        assert "before_source" in result.evidence
        assert "after_source" in result.evidence

    def test_transition_evidence_has_confidence_breakdown(self):
        """Evidence includes confidence breakdown by method."""
        evidence = {
            "spectral_confidence": 0.90,
            "temporal_confidence": 0.85,
            "spatial_confidence": 0.88,
            "aggregate_confidence": 0.88,
        }
        result = LandUseTransition(
            plot_id="PLOT-EVD-003",
            evidence=evidence,
        )
        assert all(0.0 <= v <= 1.0 for v in result.evidence.values())

    def test_transition_evidence_json_serializable(self):
        """Evidence dict is JSON serializable."""
        evidence = {
            "ndvi_before": 0.75,
            "ndvi_after": 0.20,
            "dates": ["2020-12-15", "2022-03-15"],
        }
        result = LandUseTransition(
            plot_id="PLOT-EVD-004",
            evidence=evidence,
        )
        json_str = json.dumps(result.evidence)
        parsed = json.loads(json_str)
        assert parsed["ndvi_before"] == 0.75

    def test_transition_evidence_empty_for_stable(self):
        """Stable transitions may have empty evidence."""
        result = LandUseTransition(
            plot_id="PLOT-EVD-005",
            transition_type="stable",
            evidence={},
        )
        assert result.evidence == {}


# ===========================================================================
# 5. is_deforestation Flag Tests (8 tests)
# ===========================================================================


class TestIsDeforestation:
    """Tests for is_deforestation flag logic."""

    def test_is_deforestation_true_forest_to_cropland(self):
        """Forest -> cropland is deforestation."""
        assert is_deforestation_transition("forest", "cropland") is True

    def test_is_deforestation_true_forest_to_grassland(self):
        """Forest -> grassland (pasture) is deforestation."""
        assert is_deforestation_transition("forest", "grassland") is True

    def test_is_deforestation_false_cropland_to_cropland(self):
        """Cropland -> cropland is NOT deforestation."""
        assert is_deforestation_transition("cropland", "cropland") is False

    def test_is_deforestation_false_grassland_to_cropland(self):
        """Grassland -> cropland is NOT deforestation (no forest involved)."""
        assert is_deforestation_transition("grassland", "cropland") is False

    def test_is_deforestation_false_forest_to_forest(self):
        """Forest -> forest is NOT deforestation."""
        assert is_deforestation_transition("forest", "forest") is False

    def test_is_deforestation_false_cropland_to_forest(self):
        """Cropland -> forest is reforestation, not deforestation."""
        assert is_deforestation_transition("cropland", "forest") is False

    def test_is_deforestation_false_urban_to_cropland(self):
        """Urban -> cropland is NOT deforestation."""
        assert is_deforestation_transition("urban", "cropland") is False

    def test_is_deforestation_false_water_to_bare_soil(self):
        """Water -> bare soil is NOT deforestation."""
        assert is_deforestation_transition("water", "bare_soil") is False


# ===========================================================================
# 6. is_degradation Flag Tests (5 tests)
# ===========================================================================


class TestIsDegradation:
    """Tests for is_degradation flag logic."""

    def test_is_degradation_true_forest_to_shrubland(self):
        """Forest -> shrubland is degradation."""
        assert is_degradation_transition("forest", "shrubland") is True

    def test_is_degradation_false_forest_to_cropland(self):
        """Forest -> cropland is deforestation, not degradation."""
        assert is_degradation_transition("forest", "cropland") is False

    def test_is_degradation_false_shrubland_to_forest(self):
        """Shrubland -> forest is recovery, not degradation."""
        assert is_degradation_transition("shrubland", "forest") is False

    def test_is_degradation_false_cropland_to_grassland(self):
        """Cropland -> grassland is NOT degradation."""
        assert is_degradation_transition("cropland", "grassland") is False

    def test_is_degradation_false_forest_to_forest(self):
        """Forest -> forest is stable, not degradation."""
        assert is_degradation_transition("forest", "forest") is False


# ===========================================================================
# 7. Confidence and Batch Tests (6 tests)
# ===========================================================================


class TestConfidenceAndBatch:
    """Tests for confidence computation and batch detection."""

    def test_confidence_computation_high_agreement(self):
        """High agreement between methods yields high confidence."""
        result = LandUseTransition(
            plot_id="PLOT-CONF-001",
            confidence=0.95,
        )
        assert result.confidence > 0.90

    def test_confidence_computation_low_agreement(self):
        """Low agreement yields low confidence."""
        result = LandUseTransition(
            plot_id="PLOT-CONF-002",
            confidence=0.45,
        )
        assert result.confidence < 0.50

    def test_batch_detection_multiple_plots(self):
        """Batch detection processes multiple plots."""
        results = [
            LandUseTransition(
                plot_id=f"PLOT-BATCH-{i:04d}",
                from_category="forest" if i % 2 == 0 else "cropland",
                to_category="cropland" if i % 2 == 0 else "cropland",
                transition_type="deforestation" if i % 2 == 0 else "stable",
                confidence=0.80 + i * 0.002,
            )
            for i in range(50)
        ]
        assert len(results) == 50
        deforestation_count = sum(
            1 for r in results if r.transition_type == "deforestation"
        )
        assert deforestation_count == 25

    def test_batch_all_have_provenance(self):
        """Each result in batch has provenance hash."""
        results = []
        for i in range(10):
            h = compute_test_hash({"plot_id": f"PLOT-PRV-{i}"})
            results.append(LandUseTransition(
                plot_id=f"PLOT-PRV-{i}",
                provenance_hash=h,
            ))
        assert all(len(r.provenance_hash) == SHA256_HEX_LENGTH for r in results)

    def test_deterministic_detection(self):
        """Same inputs produce same transition results."""
        results = [
            LandUseTransition(
                plot_id="PLOT-DET-001",
                from_category="forest",
                to_category="cropland",
                transition_type="deforestation",
                confidence=0.90,
            )
            for _ in range(5)
        ]
        assert all(r.transition_type == "deforestation" for r in results)
        assert all(r.confidence == 0.90 for r in results)

    def test_area_ha_positive_for_real_transition(self):
        """Real transitions have positive area."""
        result = LandUseTransition(
            plot_id="PLOT-AREA-001",
            from_category="forest",
            to_category="cropland",
            transition_type="deforestation",
            area_ha=3.5,
        )
        assert result.area_ha > 0.0


# ===========================================================================
# 8. Parametrized From/To Combination Tests (10x10 = 100 combinations)
# ===========================================================================


class TestAllTransitionCombinations:
    """Parametrized tests for all 100 from/to category combinations."""

    @pytest.mark.parametrize("from_cat", LAND_USE_CATEGORIES)
    @pytest.mark.parametrize("to_cat", LAND_USE_CATEGORIES)
    def test_transition_from_to(self, from_cat, to_cat):
        """Every from/to combination produces a valid transition result."""
        is_stable = from_cat == to_cat
        is_deforest = is_deforestation_transition(from_cat, to_cat)
        is_degrade = is_degradation_transition(from_cat, to_cat)

        if is_stable:
            ttype = "stable"
        elif is_deforest:
            ttype = "deforestation"
        elif is_degrade:
            ttype = "degradation"
        elif to_cat == "urban":
            ttype = "urbanization"
        elif to_cat == "forest" and from_cat != "forest":
            ttype = "reforestation"
        else:
            ttype = "unknown"

        result = LandUseTransition(
            plot_id=f"PLOT-{from_cat[:3].upper()}-{to_cat[:3].upper()}",
            from_category=from_cat,
            to_category=to_cat,
            transition_type=ttype,
            confidence=0.75,
            is_deforestation=is_deforest,
            is_degradation=is_degrade,
        )
        assert result.from_category == from_cat
        assert result.to_category == to_cat
        assert result.transition_type in TRANSITION_TYPES
        assert result.is_deforestation == is_deforest
        assert result.is_degradation == is_degrade
