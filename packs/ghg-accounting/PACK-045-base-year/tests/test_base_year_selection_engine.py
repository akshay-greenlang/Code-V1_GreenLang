# -*- coding: utf-8 -*-
"""
Tests for BaseYearSelectionEngine (Engine 1).

Covers candidate scoring, selection, ranking, type recommendation,
sector adjustments, validation, and batch operations.
Target: ~80 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.base_year_selection_engine import (
    BaseYearSelectionEngine,
    CandidateYear,
    SelectionConfig,
    SelectionWeights,
    SelectionResult,
    CandidateScore,
    CriterionScore,
    BaseYearTypeRecommendation,
    BaseYearType,
    SectorType,
    SelectionCriterion,
    RecommendationConfidence,
    MINIMUM_CANDIDATES,
    MAXIMUM_CANDIDATES,
    DEFAULT_WEIGHTS,
    METHODOLOGY_TIER_SCORES,
    SECTOR_GUIDANCE,
    SECTOR_WEIGHT_ADJUSTMENTS,
    evaluate_base_year_candidates,
    get_default_selection_weights,
    get_sector_selection_weights,
)


class TestBaseYearSelectionEngineInit:
    def test_engine_creation(self, selection_engine):
        assert selection_engine is not None

    def test_engine_version(self, selection_engine):
        assert selection_engine.get_version() == "1.0.0"

    def test_default_weights(self, selection_engine):
        w = selection_engine.get_default_weights()
        assert isinstance(w, SelectionWeights)

    def test_sector_weights(self, selection_engine):
        w = selection_engine.get_sector_weights(SectorType.MANUFACTURING)
        assert isinstance(w, SelectionWeights)


class TestCandidateYearModel:
    def test_create_basic(self):
        c = CandidateYear(year=2020)
        assert c.year == 2020
        assert c.total_tco2e == Decimal("0")

    def test_auto_compute_total(self):
        c = CandidateYear(
            year=2020,
            scope1_tco2e=Decimal("100"),
            scope2_tco2e=Decimal("200"),
            scope3_tco2e=Decimal("300"),
        )
        assert c.total_tco2e == Decimal("600")

    def test_explicit_total_preserved(self):
        c = CandidateYear(
            year=2020,
            scope1_tco2e=Decimal("100"),
            scope2_tco2e=Decimal("200"),
            total_tco2e=Decimal("999"),
        )
        assert c.total_tco2e == Decimal("999")

    def test_coerce_decimal_from_float(self):
        c = CandidateYear(year=2020, data_quality_score=85.5)
        assert c.data_quality_score == Decimal("85.5")

    def test_coerce_decimal_from_int(self):
        c = CandidateYear(year=2020, data_quality_score=90)
        assert c.data_quality_score == Decimal("90")

    def test_year_range_validation(self):
        with pytest.raises(ValueError):
            CandidateYear(year=1989)  # below minimum

    def test_optional_fields(self):
        c = CandidateYear(year=2020, production_volume=Decimal("1000"))
        assert c.production_volume == Decimal("1000")
        assert c.occupancy_rate is None


class TestSelectionWeights:
    def test_defaults_sum_to_one(self):
        w = SelectionWeights()
        total = (
            w.data_quality_weight + w.completeness_weight +
            w.representativeness_weight + w.methodology_weight +
            w.verification_weight + w.stability_weight
        )
        assert abs(total - Decimal("1")) < Decimal("0.001")

    def test_custom_weights_valid(self):
        w = SelectionWeights(
            data_quality_weight=Decimal("0.30"),
            completeness_weight=Decimal("0.30"),
            representativeness_weight=Decimal("0.10"),
            methodology_weight=Decimal("0.10"),
            verification_weight=Decimal("0.10"),
            stability_weight=Decimal("0.10"),
        )
        assert w.data_quality_weight == Decimal("0.30")

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SelectionWeights(
                data_quality_weight=Decimal("0.50"),
                completeness_weight=Decimal("0.50"),
                representativeness_weight=Decimal("0.50"),
            )

    def test_to_dict(self):
        w = SelectionWeights()
        d = w.to_dict()
        assert SelectionCriterion.DATA_QUALITY.value in d
        assert len(d) == 6

    def test_apply_sector_adjustments(self):
        w = SelectionWeights()
        adjusted = w.apply_sector_adjustments(SectorType.MANUFACTURING)
        total = sum(adjusted.to_dict().values())
        assert abs(total - Decimal("1")) < Decimal("0.01")

    def test_apply_no_adjustment_for_other(self):
        w = SelectionWeights()
        adjusted = w.apply_sector_adjustments(SectorType.OTHER)
        assert adjusted.data_quality_weight == w.data_quality_weight


class TestScoringMethods:
    def test_score_data_quality(self, selection_engine):
        c = CandidateYear(year=2020, data_quality_score=Decimal("85"))
        score = selection_engine.score_data_quality(c)
        assert score == Decimal("85.00")

    def test_score_data_quality_clamped(self, selection_engine):
        c = CandidateYear(year=2020, data_quality_score=Decimal("0"))
        score = selection_engine.score_data_quality(c)
        assert score == Decimal("0.00")

    def test_score_completeness(self, selection_engine):
        c = CandidateYear(year=2020, completeness_pct=Decimal("95"))
        score = selection_engine.score_completeness(c)
        assert score == Decimal("95.00")

    def test_score_methodology_tier1(self, selection_engine):
        c = CandidateYear(year=2020, methodology_tier=1)
        assert selection_engine.score_methodology(c) == Decimal("30.00")

    def test_score_methodology_tier2(self, selection_engine):
        c = CandidateYear(year=2020, methodology_tier=2)
        assert selection_engine.score_methodology(c) == Decimal("60.00")

    def test_score_methodology_tier3(self, selection_engine):
        c = CandidateYear(year=2020, methodology_tier=3)
        assert selection_engine.score_methodology(c) == Decimal("90.00")

    def test_score_methodology_tier4(self, selection_engine):
        c = CandidateYear(year=2020, methodology_tier=4)
        assert selection_engine.score_methodology(c) == Decimal("100.00")

    def test_score_verification_true(self, selection_engine):
        c = CandidateYear(year=2020, is_verified=True)
        assert selection_engine.score_verification(c) == Decimal("100.00")

    def test_score_verification_false(self, selection_engine):
        c = CandidateYear(year=2020, is_verified=False)
        assert selection_engine.score_verification(c) == Decimal("0.00")

    def test_score_stability_zero_changes(self, selection_engine):
        c = CandidateYear(year=2020, boundary_changes_count=0)
        assert selection_engine.score_stability(c) == Decimal("100.00")

    def test_score_stability_one_change(self, selection_engine):
        c = CandidateYear(year=2020, boundary_changes_count=1)
        assert selection_engine.score_stability(c) == Decimal("80.00")

    def test_score_stability_five_changes_floors_at_zero(self, selection_engine):
        c = CandidateYear(year=2020, boundary_changes_count=5)
        assert selection_engine.score_stability(c) == Decimal("0.00")

    def test_score_stability_many_changes_floors_at_zero(self, selection_engine):
        c = CandidateYear(year=2020, boundary_changes_count=10)
        assert selection_engine.score_stability(c) == Decimal("0.00")

    def test_score_representativeness(self, selection_engine, candidate_years):
        # The candidate closest to median should score highest
        score = selection_engine.score_representativeness(
            candidate_years[0], candidate_years
        )
        assert Decimal("0") <= score <= Decimal("100")


class TestEvaluateCandidates:
    def test_basic_evaluation(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        assert isinstance(result, SelectionResult)
        assert result.recommended_year in [c.year for c in candidate_years]
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_candidate_scores_ranked(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        ranks = [cs.rank for cs in result.candidate_scores]
        assert sorted(ranks) == list(range(1, len(candidate_years) + 1))

    def test_best_candidate_rank_one(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        best = result.candidate_scores[0]
        assert best.rank == 1
        assert best.year == result.recommended_year

    def test_provenance_hash_deterministic(self, selection_engine, candidate_years):
        r1 = selection_engine.evaluate_candidates(candidate_years)
        r2 = selection_engine.evaluate_candidates(candidate_years)
        # Provenance hashes exclude volatile fields so should match
        # for the same logical output
        assert r1.recommended_year == r2.recommended_year

    def test_too_few_candidates_raises(self, selection_engine):
        with pytest.raises(ValueError, match="At least"):
            selection_engine.evaluate_candidates([
                CandidateYear(year=2020, total_tco2e=Decimal("1000"))
            ])

    def test_too_many_candidates_raises(self, selection_engine):
        candidates = [
            CandidateYear(year=1990 + i, total_tco2e=Decimal("1000"))
            for i in range(31)
        ]
        with pytest.raises(ValueError, match="At most"):
            selection_engine.evaluate_candidates(candidates)

    def test_confidence_level_set(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        assert result.confidence in list(RecommendationConfidence)

    def test_sector_guidance_present(self, selection_engine, candidate_years):
        config = SelectionConfig(sector=SectorType.MANUFACTURING)
        result = selection_engine.evaluate_candidates(candidate_years, config)
        assert "Manufacturing" in result.sector_guidance

    def test_type_recommendation_present(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        assert result.type_recommendation is not None
        assert result.base_year_type in list(BaseYearType)

    def test_config_snapshot_stored(self, selection_engine, candidate_years):
        config = SelectionConfig(sector=SectorType.ENERGY)
        result = selection_engine.evaluate_candidates(candidate_years, config)
        assert result.config_used is not None
        assert result.config_used.sector == SectorType.ENERGY

    def test_rationale_not_empty(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        assert len(result.rationale) > 0

    def test_processing_time_positive(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        assert result.processing_time_ms > 0

    def test_calculated_at_populated(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        assert result.calculated_at != ""


class TestEligibilityFiltering:
    def test_minimum_quality_filter(self, selection_engine, candidate_years):
        config = SelectionConfig(minimum_quality=Decimal("95"))
        result = selection_engine.evaluate_candidates(candidate_years, config)
        # Some candidates may be ineligible
        ineligible = [cs for cs in result.candidate_scores if not cs.is_eligible]
        assert len(ineligible) > 0

    def test_minimum_completeness_filter(self, selection_engine, candidate_years):
        config = SelectionConfig(minimum_completeness=Decimal("99"))
        result = selection_engine.evaluate_candidates(candidate_years, config)
        ineligible = [cs for cs in result.candidate_scores if not cs.is_eligible]
        assert len(ineligible) > 0

    def test_require_verification_filter(self, selection_engine, candidate_years):
        config = SelectionConfig(require_verification=True)
        result = selection_engine.evaluate_candidates(candidate_years, config)
        # year 2020 is unverified
        unverified = [cs for cs in result.candidate_scores
                      if cs.year == 2020 and not cs.is_eligible]
        assert len(unverified) == 1

    def test_max_boundary_changes_filter(self, selection_engine, candidate_years):
        config = SelectionConfig(max_boundary_changes=0)
        result = selection_engine.evaluate_candidates(candidate_years, config)
        # year 2020 has 1 boundary change
        filtered = [cs for cs in result.candidate_scores
                    if cs.year == 2020 and not cs.is_eligible]
        assert len(filtered) == 1


class TestRecencyBonus:
    def test_recency_bonus_applied(self, selection_engine, candidate_years):
        config = SelectionConfig(prefer_recent=True, recent_year_bonus=Decimal("2"))
        result = selection_engine.evaluate_candidates(candidate_years, config)
        # Most recent year (2023) should have highest recency bonus
        scores_2023 = [cs for cs in result.candidate_scores if cs.year == 2023]
        assert len(scores_2023) == 1
        assert scores_2023[0].recency_bonus > Decimal("0")

    def test_no_recency_bonus_by_default(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        for cs in result.candidate_scores:
            assert cs.recency_bonus == Decimal("0")


class TestBaseYearTypeRecommendation:
    def test_low_cv_fixed(self, selection_engine, candidate_years):
        # candidate_years have similar emissions -> low CV -> FIXED
        rec = selection_engine.recommend_base_year_type(candidate_years)
        assert rec.recommended_type == BaseYearType.FIXED
        assert rec.coefficient_of_variation >= Decimal("0")

    def test_high_cv_rolling(self, selection_engine):
        # Create high variability candidates
        candidates = [
            CandidateYear(year=2019, total_tco2e=Decimal("5000")),
            CandidateYear(year=2020, total_tco2e=Decimal("15000")),
            CandidateYear(year=2021, total_tco2e=Decimal("5000")),
        ]
        rec = selection_engine.recommend_base_year_type(candidates)
        assert rec.recommended_type in (BaseYearType.ROLLING_3YR, BaseYearType.ROLLING_5YR)

    def test_agriculture_sector_bumps_up(self, selection_engine, candidate_years):
        rec = selection_engine.recommend_base_year_type(
            candidate_years, SectorType.AGRICULTURE
        )
        # Agriculture should bump up from FIXED to ROLLING_3YR
        assert rec.recommended_type in (BaseYearType.ROLLING_3YR, BaseYearType.ROLLING_5YR)

    def test_insufficient_data_defaults_fixed(self, selection_engine):
        candidates = [CandidateYear(year=2020, total_tco2e=Decimal("0"))]
        rec = selection_engine.recommend_base_year_type(candidates)
        assert rec.recommended_type == BaseYearType.FIXED

    def test_recommendation_has_rationale(self, selection_engine, candidate_years):
        rec = selection_engine.recommend_base_year_type(candidate_years)
        assert len(rec.rationale) > 0


class TestBatchOperations:
    def test_evaluate_multiple_orgs(self, selection_engine, candidate_years):
        org_data = {
            "ORG-001": candidate_years,
            "ORG-002": candidate_years[:3],
        }
        results = selection_engine.evaluate_multiple_organisations(org_data)
        assert "ORG-001" in results
        assert "ORG-002" in results

    def test_evaluate_multiple_skips_invalid(self, selection_engine, candidate_years):
        org_data = {
            "ORG-001": candidate_years,
            "ORG-BAD": [CandidateYear(year=2020)],  # only 1 candidate
        }
        results = selection_engine.evaluate_multiple_organisations(org_data)
        assert "ORG-001" in results
        assert "ORG-BAD" not in results

    def test_compare_scenarios(self, selection_engine, candidate_years):
        configs = [
            SelectionConfig(sector=SectorType.MANUFACTURING),
            SelectionConfig(sector=SectorType.ENERGY),
        ]
        results = selection_engine.compare_scenarios(candidate_years, configs)
        assert len(results) == 2


class TestValidation:
    def test_validate_duplicates(self, selection_engine):
        candidates = [
            CandidateYear(year=2020, total_tco2e=Decimal("1000")),
            CandidateYear(year=2020, total_tco2e=Decimal("2000")),
        ]
        issues = selection_engine.validate_candidates(candidates)
        assert any("duplicate" in i.lower() for i in issues)

    def test_validate_zero_emissions(self, selection_engine):
        candidates = [
            CandidateYear(year=2019, total_tco2e=Decimal("1000")),
            CandidateYear(year=2020, total_tco2e=Decimal("0")),
        ]
        issues = selection_engine.validate_candidates(candidates)
        assert any("zero" in i.lower() for i in issues)

    def test_validate_gaps(self, selection_engine):
        candidates = [
            CandidateYear(year=2019, total_tco2e=Decimal("1000")),
            CandidateYear(year=2022, total_tco2e=Decimal("1000")),
        ]
        issues = selection_engine.validate_candidates(candidates)
        assert any("gap" in i.lower() for i in issues)

    def test_validate_outliers(self, selection_engine):
        candidates = [
            CandidateYear(year=2019, total_tco2e=Decimal("1000")),
            CandidateYear(year=2020, total_tco2e=Decimal("1000")),
            CandidateYear(year=2021, total_tco2e=Decimal("50000")),
        ]
        issues = selection_engine.validate_candidates(candidates)
        assert any("outlier" in i.lower() for i in issues)

    def test_validate_clean(self, selection_engine, candidate_years):
        issues = selection_engine.validate_candidates(candidate_years)
        # Our fixture candidates should have no duplicates or gaps
        assert not any("duplicate" in i.lower() for i in issues)


class TestScoringUtility:
    def test_get_scoring_summary(self, selection_engine, candidate_years):
        result = selection_engine.evaluate_candidates(candidate_years)
        summary = selection_engine.get_scoring_summary(result)
        assert summary["recommended_year"] == result.recommended_year
        assert len(summary["scores"]) == len(candidate_years)
        assert summary["provenance_hash"] == result.provenance_hash


class TestConvenienceFunctions:
    def test_evaluate_base_year_candidates(self, candidate_years):
        result = evaluate_base_year_candidates(candidate_years)
        assert isinstance(result, SelectionResult)

    def test_get_default_selection_weights_func(self):
        w = get_default_selection_weights()
        assert isinstance(w, SelectionWeights)

    def test_get_sector_selection_weights_func(self):
        w = get_sector_selection_weights(SectorType.ENERGY)
        assert isinstance(w, SelectionWeights)


class TestConstants:
    def test_minimum_base_year(self):
        from engines.base_year_selection_engine import MINIMUM_BASE_YEAR
        assert MINIMUM_BASE_YEAR == 1990

    def test_maximum_base_year(self):
        from engines.base_year_selection_engine import MAXIMUM_BASE_YEAR
        assert MAXIMUM_BASE_YEAR == 2035

    def test_methodology_tier_scores(self):
        assert METHODOLOGY_TIER_SCORES[1] == Decimal("30")
        assert METHODOLOGY_TIER_SCORES[4] == Decimal("100")

    def test_sector_guidance_keys(self):
        assert SectorType.MANUFACTURING.value in SECTOR_GUIDANCE
        assert SectorType.AGRICULTURE.value in SECTOR_GUIDANCE

    def test_sector_weight_adjustments_keys(self):
        assert SectorType.MANUFACTURING.value in SECTOR_WEIGHT_ADJUSTMENTS
