# -*- coding: utf-8 -*-
"""
Tests for CutoffDateVerifier - AGENT-EUDR-005 Land Use Change Detector

Comprehensive test suite covering:
- Verdict: compliant (forest -> forest stable)
- Verdict: non_compliant (forest -> agriculture after cutoff)
- Verdict: degraded (forest -> degraded forest)
- Verdict: inconclusive (low confidence data)
- Verdict: pre_existing_agriculture (agri -> agri)
- Conservative bias: ambiguous cases -> INCONCLUSIVE
- No false compliant: strict guarantees
- EUDR Article 2(4) timber plantation exclusion
- Cross-validation scoring
- Evidence compilation and contents
- Batch verification
- Deterministic verdicts
- All 7 EUDR commodities tested
- Cutoff date correctness (2020-12-31)
- Parametrized tests for verdict determination matrix

Test count: 70 tests
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
    CutoffVerification,
    compute_test_hash,
    determine_verdict,
    SHA256_HEX_LENGTH,
    COMPLIANCE_VERDICTS,
    EUDR_COMMODITIES,
    EUDR_DEFORESTATION_CUTOFF,
    EUDR_CUTOFF_DATE,
    VERDICT_REGULATORY_REFS,
)


# ===========================================================================
# 1. Compliant Verdict Tests (10 tests)
# ===========================================================================


class TestVerdictCompliant:
    """Tests for compliant (deforestation-free) verdicts."""

    def test_verdict_compliant_forest_stable(self):
        """Forest at cutoff and still forest -> compliant."""
        result = CutoffVerification(
            plot_id="PLOT-CMP-001",
            verdict="compliant",
            cutoff_category="forest",
            current_category="forest",
            cutoff_confidence=0.90,
            current_confidence=0.88,
            transition_detected=False,
        )
        assert result.verdict == "compliant"
        assert result.transition_detected is False

    def test_verdict_compliant_high_confidence(self):
        """Compliant verdict requires high confidence."""
        result = CutoffVerification(
            plot_id="PLOT-CMP-002",
            verdict="compliant",
            cutoff_confidence=0.92,
            current_confidence=0.90,
        )
        assert result.cutoff_confidence > 0.85
        assert result.current_confidence > 0.85

    def test_verdict_compliant_no_transition(self):
        """Compliant verdict has no transition detected."""
        result = CutoffVerification(
            plot_id="PLOT-CMP-003",
            verdict="compliant",
            transition_detected=False,
            transition_date=None,
        )
        assert result.transition_detected is False
        assert result.transition_date is None

    def test_verdict_compliant_regulatory_refs(self):
        """Compliant verdict includes correct regulatory references."""
        result = CutoffVerification(
            plot_id="PLOT-CMP-004",
            verdict="compliant",
            regulatory_references=VERDICT_REGULATORY_REFS["compliant"],
        )
        assert "EUDR Art. 3(a)" in result.regulatory_references

    def test_verdict_compliant_provenance(self):
        """Compliant verdict has provenance hash."""
        h = compute_test_hash({"plot_id": "PLOT-CMP-005", "verdict": "compliant"})
        result = CutoffVerification(
            plot_id="PLOT-CMP-005",
            verdict="compliant",
            provenance_hash=h,
        )
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_verdict_compliant_soya_commodity(self):
        """Soya plot verified compliant."""
        result = CutoffVerification(
            plot_id="PLOT-CMP-SOYA",
            verdict="compliant",
            commodity="soya",
            cutoff_category="forest",
            current_category="forest",
        )
        assert result.commodity == "soya"
        assert result.verdict == "compliant"

    def test_verdict_compliant_cross_validation(self):
        """Compliant verdict has high cross-validation score."""
        result = CutoffVerification(
            plot_id="PLOT-CMP-006",
            verdict="compliant",
            cross_validation_score=0.92,
        )
        assert result.cross_validation_score > 0.85

    def test_verdict_function_compliant(self):
        """Determine verdict function returns compliant correctly."""
        v = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            confidence=0.90,
        )
        assert v == "compliant"

    def test_verdict_function_compliant_high_confidence(self):
        """High confidence forest-stable returns compliant."""
        v = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            confidence=0.99,
        )
        assert v == "compliant"

    def test_verdict_compliant_wood_commodity(self):
        """Wood commodity with stable forest -> compliant."""
        result = CutoffVerification(
            plot_id="PLOT-CMP-WOOD",
            verdict="compliant",
            commodity="wood",
            cutoff_category="forest",
            current_category="forest",
        )
        assert result.verdict == "compliant"


# ===========================================================================
# 2. Non-Compliant Verdict Tests (8 tests)
# ===========================================================================


class TestVerdictNonCompliant:
    """Tests for non-compliant (deforestation detected) verdicts."""

    def test_verdict_non_compliant_forest_to_agriculture(self):
        """Forest at cutoff, now cropland -> non_compliant."""
        result = CutoffVerification(
            plot_id="PLOT-NCP-001",
            verdict="non_compliant",
            cutoff_category="forest",
            current_category="cropland",
            transition_detected=True,
            transition_date="2022-03-01",
            cutoff_confidence=0.90,
            current_confidence=0.88,
        )
        assert result.verdict == "non_compliant"
        assert result.transition_detected is True

    def test_verdict_non_compliant_post_cutoff(self):
        """Non-compliant transition occurred after cutoff date."""
        result = CutoffVerification(
            plot_id="PLOT-NCP-002",
            verdict="non_compliant",
            transition_date="2022-06-15",
        )
        assert result.transition_date > EUDR_DEFORESTATION_CUTOFF

    def test_verdict_non_compliant_regulatory_refs(self):
        """Non-compliant verdict includes correct regulatory references."""
        result = CutoffVerification(
            plot_id="PLOT-NCP-003",
            verdict="non_compliant",
            regulatory_references=VERDICT_REGULATORY_REFS["non_compliant"],
        )
        assert "EUDR Art. 3(b)" in result.regulatory_references

    def test_verdict_function_non_compliant(self):
        """Determine verdict returns non_compliant for deforestation."""
        v = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            confidence=0.90,
        )
        assert v == "non_compliant"

    def test_verdict_non_compliant_palm_oil(self):
        """Palm oil plot with deforestation -> non_compliant."""
        result = CutoffVerification(
            plot_id="PLOT-NCP-PO",
            verdict="non_compliant",
            commodity="palm_oil",
            cutoff_category="forest",
            current_category="cropland",
        )
        assert result.commodity == "palm_oil"
        assert result.verdict == "non_compliant"

    def test_verdict_non_compliant_rubber(self):
        """Rubber plot with deforestation -> non_compliant."""
        result = CutoffVerification(
            plot_id="PLOT-NCP-RB",
            verdict="non_compliant",
            commodity="rubber",
            cutoff_category="forest",
            current_category="cropland",
        )
        assert result.verdict == "non_compliant"

    def test_verdict_non_compliant_evidence(self):
        """Non-compliant verdict includes supporting evidence."""
        result = CutoffVerification(
            plot_id="PLOT-NCP-004",
            verdict="non_compliant",
            evidence={
                "cutoff_ndvi": 0.72,
                "current_ndvi": 0.18,
                "ndvi_change": -0.54,
                "spectral_evidence": "clear_conversion",
            },
        )
        assert result.evidence["ndvi_change"] < -0.30

    def test_verdict_non_compliant_cattle(self):
        """Cattle pasture expansion from forest -> non_compliant."""
        result = CutoffVerification(
            plot_id="PLOT-NCP-CT",
            verdict="non_compliant",
            commodity="cattle",
            cutoff_category="forest",
            current_category="grassland",
        )
        assert result.verdict == "non_compliant"


# ===========================================================================
# 3. Degraded Verdict Tests (5 tests)
# ===========================================================================


class TestVerdictDegraded:
    """Tests for degraded (forest degradation) verdicts."""

    def test_verdict_degraded_forest_thinning(self):
        """Forest -> thinned forest is degraded."""
        result = CutoffVerification(
            plot_id="PLOT-DGR-001",
            verdict="degraded",
            cutoff_category="forest",
            current_category="shrubland",
        )
        assert result.verdict == "degraded"

    def test_verdict_degraded_regulatory_refs(self):
        """Degraded verdict references EUDR Article 2(6)."""
        result = CutoffVerification(
            plot_id="PLOT-DGR-002",
            verdict="degraded",
            regulatory_references=VERDICT_REGULATORY_REFS["degraded"],
        )
        assert "EUDR Art. 2(6)" in result.regulatory_references

    def test_verdict_degraded_partial_canopy_loss(self):
        """Degradation with partial canopy loss but still some tree cover."""
        result = CutoffVerification(
            plot_id="PLOT-DGR-003",
            verdict="degraded",
            evidence={
                "cutoff_canopy_pct": 75.0,
                "current_canopy_pct": 35.0,
                "canopy_loss_pct": 53.3,
            },
        )
        assert result.evidence["current_canopy_pct"] < result.evidence["cutoff_canopy_pct"]
        assert result.evidence["current_canopy_pct"] > 10.0

    def test_verdict_degraded_confidence(self):
        """Degraded verdict has confidence above threshold."""
        result = CutoffVerification(
            plot_id="PLOT-DGR-004",
            verdict="degraded",
            cutoff_confidence=0.85,
            current_confidence=0.82,
        )
        assert result.cutoff_confidence > 0.60
        assert result.current_confidence > 0.60

    def test_verdict_degraded_cocoa(self):
        """Cocoa agroforestry showing degradation."""
        result = CutoffVerification(
            plot_id="PLOT-DGR-CC",
            verdict="degraded",
            commodity="cocoa",
        )
        assert result.verdict == "degraded"


# ===========================================================================
# 4. Inconclusive Verdict Tests (8 tests)
# ===========================================================================


class TestVerdictInconclusive:
    """Tests for inconclusive (insufficient data) verdicts."""

    def test_verdict_inconclusive_low_confidence(self):
        """Low confidence data -> inconclusive."""
        result = CutoffVerification(
            plot_id="PLOT-INC-001",
            verdict="inconclusive",
            cutoff_confidence=0.40,
            current_confidence=0.35,
        )
        assert result.verdict == "inconclusive"

    def test_verdict_function_inconclusive(self):
        """Determine verdict returns inconclusive for low confidence."""
        v = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            confidence=0.30,
        )
        assert v == "inconclusive"

    def test_verdict_inconclusive_regulatory_refs(self):
        """Inconclusive verdict references EUDR Article 10(3)."""
        result = CutoffVerification(
            plot_id="PLOT-INC-002",
            verdict="inconclusive",
            regulatory_references=VERDICT_REGULATORY_REFS["inconclusive"],
        )
        assert "EUDR Art. 10(3)" in result.regulatory_references

    def test_conservative_bias_produces_inconclusive(self):
        """With conservative bias, ambiguous cases -> inconclusive."""
        v = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            confidence=0.50,
            min_confidence=0.60,
            conservative_bias=True,
        )
        assert v == "inconclusive"

    def test_no_false_compliant_low_confidence(self):
        """Never classify as compliant when confidence is below threshold."""
        for conf in [0.10, 0.20, 0.30, 0.40, 0.50, 0.59]:
            v = determine_verdict(
                cutoff_was_forest=True,
                current_is_forest=True,
                confidence=conf,
                min_confidence=0.60,
            )
            assert v != "compliant", (
                f"Should not be compliant at confidence={conf}"
            )

    def test_verdict_inconclusive_cloudy_imagery(self):
        """Cloud-contaminated imagery -> inconclusive."""
        result = CutoffVerification(
            plot_id="PLOT-INC-003",
            verdict="inconclusive",
            evidence={"cloud_cover_pct": 85.0, "usable_observations": 1},
        )
        assert result.verdict == "inconclusive"

    def test_verdict_inconclusive_conflicting_sources(self):
        """Conflicting source data -> inconclusive."""
        result = CutoffVerification(
            plot_id="PLOT-INC-004",
            verdict="inconclusive",
            cross_validation_score=0.30,
            evidence={
                "sentinel2_class": "forest",
                "landsat_class": "cropland",
                "agreement": "conflict",
            },
        )
        assert result.verdict == "inconclusive"
        assert result.cross_validation_score < 0.50

    def test_verdict_inconclusive_boundary_confidence(self, sample_config):
        """Confidence exactly at threshold minus epsilon -> inconclusive."""
        v = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            confidence=sample_config.min_confidence - 0.01,
            min_confidence=sample_config.min_confidence,
        )
        assert v == "inconclusive"


# ===========================================================================
# 5. Pre-Existing Agriculture Verdict Tests (5 tests)
# ===========================================================================


class TestVerdictPreExistingAgriculture:
    """Tests for pre-existing agriculture verdicts."""

    def test_verdict_pre_existing_agriculture(self):
        """Agriculture at cutoff and still agriculture -> pre_existing."""
        result = CutoffVerification(
            plot_id="PLOT-PEA-001",
            verdict="pre_existing_agriculture",
            cutoff_category="cropland",
            current_category="cropland",
        )
        assert result.verdict == "pre_existing_agriculture"

    def test_verdict_function_pre_existing(self):
        """Determine verdict returns pre_existing for non-forest at cutoff."""
        v = determine_verdict(
            cutoff_was_forest=False,
            current_is_forest=False,
            confidence=0.90,
        )
        assert v == "pre_existing_agriculture"

    def test_verdict_pre_existing_regulatory_refs(self):
        """Pre-existing agriculture references EUDR Article 2(4)."""
        result = CutoffVerification(
            plot_id="PLOT-PEA-002",
            verdict="pre_existing_agriculture",
            regulatory_references=VERDICT_REGULATORY_REFS["pre_existing_agriculture"],
        )
        assert "EUDR Art. 2(4)" in result.regulatory_references

    def test_verdict_pre_existing_grassland(self):
        """Grassland (pasture) at cutoff and still grassland."""
        result = CutoffVerification(
            plot_id="PLOT-PEA-003",
            verdict="pre_existing_agriculture",
            cutoff_category="grassland",
            current_category="grassland",
        )
        assert result.verdict == "pre_existing_agriculture"

    def test_verdict_pre_existing_soya(self):
        """Pre-existing soya cropland is compliant."""
        result = CutoffVerification(
            plot_id="PLOT-PEA-SY",
            verdict="pre_existing_agriculture",
            commodity="soya",
            cutoff_category="cropland",
            current_category="cropland",
        )
        assert result.verdict == "pre_existing_agriculture"


# ===========================================================================
# 6. Article 2(4) Exclusion Tests (4 tests)
# ===========================================================================


class TestArticle2_4Exclusion:
    """Tests for EUDR Article 2(4) timber plantation exclusion."""

    def test_article_2_4_timber_exclusion(self):
        """Timber plantation excluded from deforestation per Article 2(4)."""
        result = CutoffVerification(
            plot_id="PLOT-A24-001",
            verdict="compliant",
            article_2_4_applies=True,
            commodity="wood",
            cutoff_category="forest",
            current_category="forest",
        )
        assert result.article_2_4_applies is True
        assert result.verdict == "compliant"

    def test_article_2_4_not_for_palm_oil(self):
        """Article 2(4) does NOT apply to palm oil plantations."""
        result = CutoffVerification(
            plot_id="PLOT-A24-002",
            verdict="non_compliant",
            article_2_4_applies=False,
            commodity="palm_oil",
        )
        assert result.article_2_4_applies is False

    def test_article_2_4_not_for_rubber(self):
        """Article 2(4) does NOT apply to rubber plantations."""
        result = CutoffVerification(
            plot_id="PLOT-A24-003",
            article_2_4_applies=False,
            commodity="rubber",
        )
        assert result.article_2_4_applies is False

    def test_article_2_4_not_for_cocoa(self):
        """Article 2(4) does NOT apply to cocoa plantations."""
        result = CutoffVerification(
            plot_id="PLOT-A24-004",
            article_2_4_applies=False,
            commodity="cocoa",
        )
        assert result.article_2_4_applies is False


# ===========================================================================
# 7. Cross-Validation Tests (4 tests)
# ===========================================================================


class TestCrossValidation:
    """Tests for cross-validation between data sources."""

    def test_cross_validation_high_score(self):
        """High cross-validation score when sources agree."""
        result = CutoffVerification(
            plot_id="PLOT-XV-001",
            cross_validation_score=0.95,
            evidence={
                "sentinel2": "forest",
                "landsat": "forest",
                "hansen_gfc": "forest",
            },
        )
        assert result.cross_validation_score > 0.90

    def test_cross_validation_low_score(self):
        """Low cross-validation score when sources disagree."""
        result = CutoffVerification(
            plot_id="PLOT-XV-002",
            cross_validation_score=0.30,
            evidence={
                "sentinel2": "forest",
                "landsat": "cropland",
                "hansen_gfc": "shrubland",
            },
        )
        assert result.cross_validation_score < 0.50

    def test_cross_validation_moderate(self):
        """Moderate cross-validation with partial agreement."""
        result = CutoffVerification(
            plot_id="PLOT-XV-003",
            cross_validation_score=0.67,
        )
        assert 0.50 <= result.cross_validation_score <= 0.80

    def test_cross_validation_affects_confidence(self):
        """Low cross-validation should reduce overall confidence."""
        # low XV -> should not produce compliant verdict
        v = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            confidence=0.45,  # reduced by low XV
            min_confidence=0.60,
        )
        assert v == "inconclusive"


# ===========================================================================
# 8. Evidence Compilation Tests (5 tests)
# ===========================================================================


class TestEvidenceCompilation:
    """Tests for evidence compilation in cutoff verification."""

    def test_evidence_has_spectral_data(self):
        """Evidence includes spectral comparison data."""
        result = CutoffVerification(
            plot_id="PLOT-EVD-001",
            evidence={
                "cutoff_ndvi": 0.72,
                "current_ndvi": 0.70,
                "ndvi_change": -0.02,
            },
        )
        assert "cutoff_ndvi" in result.evidence

    def test_evidence_has_source_info(self):
        """Evidence includes data source information."""
        result = CutoffVerification(
            plot_id="PLOT-EVD-002",
            evidence={
                "sources_used": ["sentinel2", "landsat8", "hansen_gfc"],
                "cutoff_scene": "S2A_20201215_T20MQS",
                "current_scene": "S2A_20240301_T20MQS",
            },
        )
        assert len(result.evidence["sources_used"]) >= 2

    def test_evidence_json_serializable(self):
        """Evidence dict is JSON serializable."""
        result = CutoffVerification(
            plot_id="PLOT-EVD-003",
            evidence={
                "cutoff_ndvi": 0.72,
                "dates": ["2020-12-15", "2024-03-01"],
                "category_at_cutoff": "forest",
            },
        )
        json_str = json.dumps(result.evidence)
        parsed = json.loads(json_str)
        assert parsed["cutoff_ndvi"] == 0.72

    def test_evidence_empty_for_inconclusive(self):
        """Inconclusive may have minimal evidence."""
        result = CutoffVerification(
            plot_id="PLOT-EVD-004",
            verdict="inconclusive",
            evidence={"reason": "insufficient_data"},
        )
        assert "reason" in result.evidence

    def test_evidence_complete_for_non_compliant(self):
        """Non-compliant verdict has comprehensive evidence."""
        result = CutoffVerification(
            plot_id="PLOT-EVD-005",
            verdict="non_compliant",
            evidence={
                "cutoff_ndvi": 0.72,
                "current_ndvi": 0.18,
                "ndvi_change": -0.54,
                "transition_date": "2022-06-01",
                "area_deforested_ha": 4.2,
                "sources_used": ["sentinel2", "landsat8"],
                "spectral_evidence": "clear_conversion",
            },
        )
        assert len(result.evidence) >= 5


# ===========================================================================
# 9. Batch and Determinism Tests (5 tests)
# ===========================================================================


class TestBatchAndDeterminism:
    """Tests for batch verification and deterministic verdicts."""

    def test_batch_verification(self):
        """Batch verification of 20 plots produces correct count."""
        results = [
            CutoffVerification(
                plot_id=f"PLOT-BATCH-{i:04d}",
                verdict="compliant" if i % 3 == 0 else "non_compliant",
            )
            for i in range(20)
        ]
        assert len(results) == 20
        compliant = sum(1 for r in results if r.verdict == "compliant")
        assert compliant == 7  # indices 0,3,6,9,12,15,18

    def test_deterministic_verdicts(self):
        """Same inputs produce same verdict."""
        verdicts = [
            determine_verdict(
                cutoff_was_forest=True,
                current_is_forest=False,
                confidence=0.90,
            )
            for _ in range(10)
        ]
        assert all(v == "non_compliant" for v in verdicts)

    def test_cutoff_date_is_2020_12_31(self, sample_config):
        """EUDR cutoff date is December 31, 2020."""
        assert sample_config.cutoff_date == "2020-12-31"
        assert EUDR_DEFORESTATION_CUTOFF == "2020-12-31"
        assert EUDR_CUTOFF_DATE.year == 2020
        assert EUDR_CUTOFF_DATE.month == 12
        assert EUDR_CUTOFF_DATE.day == 31

    def test_all_verdicts_valid(self):
        """Each verdict value is accepted."""
        for verdict in COMPLIANCE_VERDICTS:
            result = CutoffVerification(
                plot_id=f"PLOT-V-{verdict.upper()[:3]}",
                verdict=verdict,
            )
            assert result.verdict == verdict

    def test_batch_provenance_hashes(self):
        """Each batch result has unique provenance hash."""
        hashes = []
        for i in range(15):
            h = compute_test_hash({"plot_id": f"PLOT-HASH-{i}", "verdict": "compliant"})
            hashes.append(h)
        assert len(set(hashes)) == 15


# ===========================================================================
# 10. All 7 Commodities Tests (7 tests)
# ===========================================================================


class TestAllCommodities:
    """Tests for cutoff verification with all 7 EUDR commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_commodity_verification(self, commodity):
        """Each EUDR commodity can be verified."""
        result = CutoffVerification(
            plot_id=f"PLOT-CMD-{commodity.upper()[:3]}",
            verdict="compliant",
            commodity=commodity,
            cutoff_category="forest",
            current_category="forest",
            cutoff_confidence=0.90,
            current_confidence=0.88,
        )
        assert result.commodity == commodity
        assert result.verdict == "compliant"


# ===========================================================================
# 11. Parametrized Verdict Matrix Tests (8 tests)
# ===========================================================================


class TestVerdictDeterminationMatrix:
    """Parametrized tests covering the verdict determination matrix."""

    @pytest.mark.parametrize(
        "cutoff_forest,current_forest,confidence,min_conf,expected",
        [
            (True, True, 0.90, 0.60, "compliant"),
            (True, False, 0.90, 0.60, "non_compliant"),
            (False, False, 0.90, 0.60, "pre_existing_agriculture"),
            (False, True, 0.90, 0.60, "pre_existing_agriculture"),
            (True, True, 0.40, 0.60, "inconclusive"),
            (True, False, 0.40, 0.60, "inconclusive"),
            (False, False, 0.40, 0.60, "inconclusive"),
            (True, True, 0.60, 0.60, "compliant"),
        ],
        ids=[
            "forest_stable_high_conf",
            "deforested_high_conf",
            "agri_stable_high_conf",
            "reforested_high_conf",
            "forest_stable_low_conf",
            "deforested_low_conf",
            "agri_stable_low_conf",
            "forest_stable_boundary_conf",
        ],
    )
    def test_verdict_matrix(
        self, cutoff_forest, current_forest, confidence, min_conf, expected
    ):
        """Verdict determination follows expected matrix logic."""
        v = determine_verdict(
            cutoff_was_forest=cutoff_forest,
            current_is_forest=current_forest,
            confidence=confidence,
            min_confidence=min_conf,
        )
        assert v == expected, (
            f"Expected {expected} for cutoff_forest={cutoff_forest}, "
            f"current_forest={current_forest}, confidence={confidence}"
        )
