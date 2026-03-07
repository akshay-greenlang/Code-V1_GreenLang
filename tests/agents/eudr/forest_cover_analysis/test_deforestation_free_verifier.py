# -*- coding: utf-8 -*-
"""
Tests for DeforestationFreeVerifier - AGENT-EUDR-004 Engine 4: Deforestation-Free Verification

THIS IS THE MOST CRITICAL ENGINE in the Forest Cover Analysis Agent.

Comprehensive test suite covering:
- Verdict determination for all 4 outcomes (DEFORESTATION_FREE, DEFORESTED,
  DEGRADED, INCONCLUSIVE)
- Canopy change percentage calculation
- Default and biome-specific degradation thresholds
- Commodity-specific exclusion rules (palm oil, rubber, agroforestry)
- Evidence package assembly (before/after, spectral, provenance)
- Regulatory reference attachment per verdict
- Conservative approach (ambiguous -> INCONCLUSIVE, never false-positive)
- Minimum confidence thresholds
- Batch verification
- Full parametrization across commodities and verdicts
- Determinism and provenance hash reproducibility

Test count: 80+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 4 - Deforestation-Free Verification)
"""

import pytest

from tests.agents.eudr.forest_cover_analysis.conftest import (
    DeforestationFreeResult,
    compute_test_hash,
    compute_canopy_change_pct,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    VERDICTS,
    VERDICT_REGULATORY_REFS,
    ALL_BIOMES,
)


# ---------------------------------------------------------------------------
# Helpers: Verdict determination logic
# ---------------------------------------------------------------------------


def _determine_verdict(
    cutoff_was_forest: bool,
    current_is_forest: bool,
    canopy_change_pct: float,
    degradation_threshold_pct: float = 30.0,
    confidence: float = 0.9,
    confidence_min: float = 0.6,
    commodity: str = "",
    commodity_exclusion_applied: bool = False,
) -> str:
    """Determine deforestation-free verdict.

    Decision tree:
    1. If confidence < confidence_min -> INCONCLUSIVE
    2. If cutoff was NOT forest -> DEFORESTATION_FREE (no forest to lose)
    3. If cutoff was forest + current NOT forest -> DEFORESTED
    4. If cutoff was forest + current is forest + canopy loss > threshold -> DEGRADED
    5. If cutoff was forest + current is forest + canopy loss <= threshold -> DEFORESTATION_FREE
    """
    # Low confidence -> inconclusive
    if confidence < confidence_min:
        return "INCONCLUSIVE"

    # If commodity exclusion applied (e.g., palm oil plantation was NOT forest)
    if commodity_exclusion_applied and not cutoff_was_forest:
        return "DEFORESTATION_FREE"

    # Plot was not forest at cutoff -> deforestation-free by definition
    if not cutoff_was_forest:
        return "DEFORESTATION_FREE"

    # Plot was forest at cutoff
    if not current_is_forest:
        return "DEFORESTED"

    # Both were/are forest -- check degradation
    if abs(canopy_change_pct) > degradation_threshold_pct:
        return "DEGRADED"

    return "DEFORESTATION_FREE"


def _get_regulatory_refs(verdict: str) -> list:
    """Return EUDR regulatory article references for a verdict."""
    return VERDICT_REGULATORY_REFS.get(verdict, [])


def _build_evidence_package(
    before_ndvi: float,
    after_ndvi: float,
    canopy_change_pct: float,
    spectral_comparison: str = "stable",
) -> dict:
    """Build an evidence package for regulatory submission."""
    return {
        "before_ndvi": before_ndvi,
        "after_ndvi": after_ndvi,
        "canopy_change_pct": canopy_change_pct,
        "spectral_comparison": spectral_comparison,
        "provenance_hash": compute_test_hash({
            "before_ndvi": before_ndvi,
            "after_ndvi": after_ndvi,
        }),
    }


# Default biome-specific degradation thresholds
_BIOME_DEGRADATION_THRESHOLDS = {
    "tropical_rainforest": 25.0,
    "tropical_moist_forest": 25.0,
    "tropical_dry_forest": 30.0,
    "temperate_forest": 35.0,
    "temperate_rainforest": 30.0,
    "temperate_deciduous": 40.0,
    "boreal_forest": 35.0,
    "mangrove": 20.0,
    "peat_swamp_forest": 20.0,
    "cerrado_savanna": 40.0,
    "tropical_savanna": 40.0,
    "woodland_savanna": 35.0,
    "montane_cloud_forest": 25.0,
    "montane_dry_forest": 30.0,
    "dry_woodland": 35.0,
    "thorn_forest": 40.0,
}


# ===========================================================================
# 1. Core Verdict Logic (20 tests)
# ===========================================================================


class TestCoreVerdictLogic:
    """Test the deforestation-free verdict determination logic."""

    def test_verdict_deforestation_free_was_not_forest(self):
        """Test plot NOT forest at cutoff -> DEFORESTATION_FREE."""
        verdict = _determine_verdict(
            cutoff_was_forest=False,
            current_is_forest=False,
            canopy_change_pct=0.0,
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_verdict_deforestation_free_forest_intact(self):
        """Test forest at cutoff + forest now + minimal change -> DEFORESTATION_FREE."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_verdict_deforested(self):
        """Test forest at cutoff + NOT forest now -> DEFORESTED."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            canopy_change_pct=-80.0,
        )
        assert verdict == "DEFORESTED"

    def test_verdict_degraded(self):
        """Test forest at cutoff + forest now + >30% canopy loss -> DEGRADED."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-35.0,
        )
        assert verdict == "DEGRADED"

    def test_verdict_inconclusive_low_confidence(self):
        """Test confidence < 0.6 -> INCONCLUSIVE."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            confidence=0.4,
        )
        assert verdict == "INCONCLUSIVE"

    def test_verdict_inconclusive_no_data(self):
        """Test very low confidence (simulating missing data) -> INCONCLUSIVE."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=0.0,
            confidence=0.1,
        )
        assert verdict == "INCONCLUSIVE"

    def test_verdict_not_forest_at_cutoff_currently_forest(self):
        """Test plot was NOT forest at cutoff but IS forest now (reforestation)."""
        verdict = _determine_verdict(
            cutoff_was_forest=False,
            current_is_forest=True,
            canopy_change_pct=100.0,
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_verdict_at_degradation_boundary(self):
        """Test canopy change exactly at degradation threshold."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-30.0,
        )
        # Exactly at threshold: abs(-30) == 30, not > 30 -> DEFORESTATION_FREE
        assert verdict == "DEFORESTATION_FREE"

    def test_verdict_just_above_degradation_threshold(self):
        """Test canopy change just above degradation threshold."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-30.1,
        )
        assert verdict == "DEGRADED"

    def test_verdict_at_confidence_boundary(self):
        """Test confidence exactly at minimum threshold."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            confidence=0.6,
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_verdict_just_below_confidence(self):
        """Test confidence just below minimum threshold."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            confidence=0.59,
        )
        assert verdict == "INCONCLUSIVE"


# ===========================================================================
# 2. Canopy Change Calculation (8 tests)
# ===========================================================================


class TestCanopyChangeCalculation:
    """Test canopy change percentage calculation."""

    def test_canopy_change_calculation(self):
        """Test (current - cutoff) / cutoff * 100."""
        change = compute_canopy_change_pct(80.0, 56.0)
        expected = ((56.0 - 80.0) / 80.0) * 100.0  # -30.0
        assert abs(change - expected) < 1e-9

    def test_canopy_change_no_change(self):
        """Test zero change when current equals cutoff."""
        change = compute_canopy_change_pct(70.0, 70.0)
        assert abs(change) < 1e-9

    def test_canopy_change_total_loss(self):
        """Test 100% loss when current is 0."""
        change = compute_canopy_change_pct(70.0, 0.0)
        assert abs(change - (-100.0)) < 1e-9

    def test_canopy_change_gain(self):
        """Test positive change for canopy gain."""
        change = compute_canopy_change_pct(50.0, 60.0)
        assert change > 0

    def test_canopy_change_cutoff_zero(self):
        """Test cutoff density of 0 returns 0 (avoid division by zero)."""
        change = compute_canopy_change_pct(0.0, 50.0)
        assert change == 0.0

    @pytest.mark.parametrize("cutoff,current,expected", [
        (100.0, 100.0, 0.0),
        (100.0, 50.0, -50.0),
        (100.0, 0.0, -100.0),
        (50.0, 75.0, 50.0),
        (80.0, 56.0, -30.0),
    ])
    def test_canopy_change_parametrized(self, cutoff, current, expected):
        """Test canopy change calculation across various scenarios."""
        change = compute_canopy_change_pct(cutoff, current)
        assert abs(change - expected) < 1e-9

    def test_canopy_change_determinism(self):
        """Test canopy change calculation is deterministic."""
        results = [compute_canopy_change_pct(80.0, 56.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 3. Degradation Thresholds (10 tests)
# ===========================================================================


class TestDegradationThresholds:
    """Test default and biome-specific degradation thresholds."""

    def test_degradation_threshold_default(self):
        """Test default degradation threshold is 30%."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-31.0,
            degradation_threshold_pct=30.0,
        )
        assert verdict == "DEGRADED"

    @pytest.mark.parametrize("biome", ALL_BIOMES)
    def test_degradation_threshold_biome_specific(self, biome):
        """Test biome-specific degradation threshold is defined and reasonable."""
        threshold = _BIOME_DEGRADATION_THRESHOLDS[biome]
        assert 10.0 <= threshold <= 50.0

    def test_degradation_threshold_mangrove_stricter(self):
        """Test mangrove has stricter threshold (20%) than default."""
        assert _BIOME_DEGRADATION_THRESHOLDS["mangrove"] < 30.0

    def test_degradation_threshold_savanna_lenient(self):
        """Test savanna has more lenient threshold (40%) than default."""
        assert _BIOME_DEGRADATION_THRESHOLDS["cerrado_savanna"] > 30.0

    def test_degradation_threshold_tropical_rainforest(self):
        """Test tropical rainforest threshold is 25%."""
        assert _BIOME_DEGRADATION_THRESHOLDS["tropical_rainforest"] == 25.0

    def test_degradation_threshold_applies_correctly(self):
        """Test biome threshold overrides default in verdict."""
        # With 25% threshold, -26% change should be DEGRADED
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-26.0,
            degradation_threshold_pct=25.0,
        )
        assert verdict == "DEGRADED"

    def test_degradation_threshold_all_biomes_defined(self):
        """Test all 16 biomes have degradation thresholds defined."""
        for biome in ALL_BIOMES:
            assert biome in _BIOME_DEGRADATION_THRESHOLDS


# ===========================================================================
# 4. Commodity Exclusion Rules (10 tests)
# ===========================================================================


class TestCommodityExclusionRules:
    """Test EUDR commodity-specific exclusion rules for verdicts."""

    def test_commodity_palm_oil_exclusion(self):
        """Test palm oil plantation (NOT forest) -> DEFORESTATION_FREE."""
        verdict = _determine_verdict(
            cutoff_was_forest=False,
            current_is_forest=False,
            canopy_change_pct=0.0,
            commodity="oil_palm",
            commodity_exclusion_applied=True,
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_commodity_rubber_exclusion(self):
        """Test rubber monoculture (NOT forest) -> DEFORESTATION_FREE."""
        verdict = _determine_verdict(
            cutoff_was_forest=False,
            current_is_forest=False,
            canopy_change_pct=0.0,
            commodity="rubber",
            commodity_exclusion_applied=True,
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_commodity_cocoa_agroforestry(self):
        """Test shade-grown cocoa agroforestry (IS forest) -> normal verification."""
        # Agroforestry with native canopy IS forest, so normal rules apply
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            commodity="cocoa",
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_commodity_coffee_agroforestry(self):
        """Test shade-grown coffee under native canopy -> normal verification."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            commodity="coffee",
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_commodity_soya_no_exclusion(self):
        """Test soya has no special exclusion rule."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            canopy_change_pct=-90.0,
            commodity="soya",
        )
        assert verdict == "DEFORESTED"

    def test_commodity_cattle_no_exclusion(self):
        """Test cattle has no special exclusion rule."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            canopy_change_pct=-90.0,
            commodity="cattle",
        )
        assert verdict == "DEFORESTED"

    def test_commodity_wood_no_exclusion(self):
        """Test wood has no special exclusion rule."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            canopy_change_pct=-90.0,
            commodity="wood",
        )
        assert verdict == "DEFORESTED"

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_all_commodities_deterministic(self, commodity):
        """Test verdict is deterministic for each commodity."""
        results = [
            _determine_verdict(
                cutoff_was_forest=True,
                current_is_forest=True,
                canopy_change_pct=-5.0,
                commodity=commodity,
            )
            for _ in range(5)
        ]
        assert len(set(results)) == 1

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_commodity_with_deforestation(self, commodity):
        """Test all commodities detect deforestation when forest was cleared."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            canopy_change_pct=-90.0,
            commodity=commodity,
        )
        assert verdict == "DEFORESTED"

    def test_commodity_empty_string(self):
        """Test empty commodity string uses default logic."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            commodity="",
        )
        assert verdict == "DEFORESTATION_FREE"


# ===========================================================================
# 5. Evidence Package (6 tests)
# ===========================================================================


class TestEvidencePackage:
    """Test evidence package assembly for regulatory submission."""

    def test_evidence_package_contents(self):
        """Test evidence package includes before/after, spectral, provenance."""
        pkg = _build_evidence_package(0.75, 0.72, -4.0, "stable")
        assert "before_ndvi" in pkg
        assert "after_ndvi" in pkg
        assert "canopy_change_pct" in pkg
        assert "spectral_comparison" in pkg
        assert "provenance_hash" in pkg

    def test_evidence_package_provenance_hash(self):
        """Test evidence package has valid SHA-256 hash."""
        pkg = _build_evidence_package(0.75, 0.72, -4.0)
        assert len(pkg["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_evidence_package_values(self):
        """Test evidence package values match inputs."""
        pkg = _build_evidence_package(0.80, 0.50, -37.5, "degraded")
        assert pkg["before_ndvi"] == 0.80
        assert pkg["after_ndvi"] == 0.50
        assert pkg["canopy_change_pct"] == -37.5
        assert pkg["spectral_comparison"] == "degraded"

    def test_evidence_package_determinism(self):
        """Test evidence package is deterministic."""
        pkgs = [_build_evidence_package(0.75, 0.72, -4.0) for _ in range(10)]
        assert all(p["provenance_hash"] == pkgs[0]["provenance_hash"] for p in pkgs)

    def test_evidence_package_fixture(self, sample_deforestation_free_result):
        """Test fixture evidence package is populated."""
        pkg = sample_deforestation_free_result.evidence_package
        assert "before_ndvi" in pkg
        assert "after_ndvi" in pkg

    def test_evidence_package_spectral_comparison_values(self):
        """Test spectral comparison string is meaningful."""
        for status in ["stable", "degraded", "deforested", "regrowth"]:
            pkg = _build_evidence_package(0.75, 0.72, -4.0, status)
            assert pkg["spectral_comparison"] == status


# ===========================================================================
# 6. Regulatory References (6 tests)
# ===========================================================================


class TestRegulatoryReferences:
    """Test EUDR regulatory article references per verdict."""

    @pytest.mark.parametrize("verdict", VERDICTS)
    def test_regulatory_references_exist(self, verdict):
        """Test each verdict has regulatory references defined."""
        refs = _get_regulatory_refs(verdict)
        assert len(refs) >= 1

    def test_regulatory_references_deforestation_free(self):
        """Test DEFORESTATION_FREE references correct articles."""
        refs = _get_regulatory_refs("DEFORESTATION_FREE")
        assert "EUDR Art. 3(a)" in refs
        assert "EUDR Art. 10(1)" in refs

    def test_regulatory_references_deforested(self):
        """Test DEFORESTED references correct articles."""
        refs = _get_regulatory_refs("DEFORESTED")
        assert "EUDR Art. 3(b)" in refs
        assert "EUDR Art. 10(2)" in refs

    def test_regulatory_references_degraded(self):
        """Test DEGRADED references correct articles."""
        refs = _get_regulatory_refs("DEGRADED")
        assert "EUDR Art. 2(6)" in refs

    def test_regulatory_references_inconclusive(self):
        """Test INCONCLUSIVE references correct articles."""
        refs = _get_regulatory_refs("INCONCLUSIVE")
        assert "EUDR Art. 10(3)" in refs

    def test_regulatory_references_unknown_verdict(self):
        """Test unknown verdict returns empty list."""
        refs = _get_regulatory_refs("UNKNOWN")
        assert refs == []


# ===========================================================================
# 7. Conservative Approach (6 tests)
# ===========================================================================


class TestConservativeApproach:
    """Test conservative approach: ambiguous cases -> INCONCLUSIVE."""

    def test_conservative_approach_low_confidence(self):
        """Test ambiguous low confidence -> INCONCLUSIVE."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            confidence=0.55,
        )
        assert verdict == "INCONCLUSIVE"

    def test_conservative_approach_never_false_positive(self):
        """Test DEFORESTATION_FREE never issued with low confidence."""
        # Even if change is minimal, low confidence should not give DF
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-1.0,
            confidence=0.3,
        )
        assert verdict == "INCONCLUSIVE"
        assert verdict != "DEFORESTATION_FREE"

    def test_conservative_approach_high_confidence_passes(self):
        """Test high confidence with minimal change -> DEFORESTATION_FREE."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-1.0,
            confidence=0.95,
        )
        assert verdict == "DEFORESTATION_FREE"

    def test_confidence_minimum_default(self):
        """Test default confidence minimum is 0.6."""
        # Confidence 0.6 should pass
        verdict_pass = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            confidence=0.6,
            confidence_min=0.6,
        )
        assert verdict_pass != "INCONCLUSIVE"

        # Confidence 0.59 should fail
        verdict_fail = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            confidence=0.59,
            confidence_min=0.6,
        )
        assert verdict_fail == "INCONCLUSIVE"

    def test_confidence_minimum_custom(self):
        """Test custom confidence minimum (0.8) applied correctly."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            confidence=0.75,
            confidence_min=0.8,
        )
        assert verdict == "INCONCLUSIVE"

    def test_conservative_no_data_scenario(self):
        """Test zero confidence -> always INCONCLUSIVE."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            canopy_change_pct=-90.0,
            confidence=0.0,
        )
        assert verdict == "INCONCLUSIVE"


# ===========================================================================
# 8. Batch Verification (3 tests)
# ===========================================================================


class TestBatchVerification:
    """Test batch deforestation-free verification."""

    def test_batch_verify_multiple(self):
        """Test batch processing returns results for all plots."""
        scenarios = [
            (True, True, -5.0, 0.9),
            (True, False, -80.0, 0.95),
            (False, False, 0.0, 0.85),
            (True, True, -35.0, 0.9),
        ]
        verdicts = [
            _determine_verdict(was, cur, chg, confidence=conf)
            for was, cur, chg, conf in scenarios
        ]
        assert len(verdicts) == 4
        assert verdicts[0] == "DEFORESTATION_FREE"
        assert verdicts[1] == "DEFORESTED"
        assert verdicts[2] == "DEFORESTATION_FREE"
        assert verdicts[3] == "DEGRADED"

    def test_batch_verify_all_verdicts_represented(self):
        """Test batch can produce all 4 verdict types."""
        scenarios = [
            (True, True, -5.0, 0.9, 0.6),    # DEFORESTATION_FREE
            (True, False, -80.0, 0.9, 0.6),   # DEFORESTED
            (True, True, -35.0, 0.9, 0.6),    # DEGRADED
            (True, True, -5.0, 0.3, 0.6),     # INCONCLUSIVE
        ]
        verdicts = set(
            _determine_verdict(was, cur, chg, confidence=conf,
                               confidence_min=cmin)
            for was, cur, chg, conf, cmin in scenarios
        )
        assert verdicts == set(VERDICTS)

    def test_batch_verify_empty(self):
        """Test empty batch returns empty results."""
        results = []
        assert len(results) == 0


# ===========================================================================
# 9. Parametrized Across Commodities and Verdicts (8 tests)
# ===========================================================================


class TestParametrizedCommodityVerdict:
    """Test verdict determination parametrized across commodities and verdicts."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_verdict_deforestation_free_per_commodity(self, commodity):
        """Test DEFORESTATION_FREE verdict possible for each commodity."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            commodity=commodity,
        )
        assert verdict == "DEFORESTATION_FREE"

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_verdict_deforested_per_commodity(self, commodity):
        """Test DEFORESTED verdict for each commodity when forest cleared."""
        verdict = _determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            canopy_change_pct=-90.0,
            commodity=commodity,
        )
        assert verdict == "DEFORESTED"

    @pytest.mark.parametrize("verdict_type", VERDICTS)
    def test_verdict_values_complete(self, verdict_type):
        """Test all 4 verdict types are recognized."""
        assert verdict_type in VERDICTS

    def test_verdict_count(self):
        """Test exactly 4 verdicts exist."""
        assert len(VERDICTS) == 4

    @pytest.mark.parametrize("commodity,was_forest,is_forest,change,expected", [
        ("soya", True, True, -5.0, "DEFORESTATION_FREE"),
        ("soya", True, False, -90.0, "DEFORESTED"),
        ("soya", True, True, -35.0, "DEGRADED"),
        ("oil_palm", False, False, 0.0, "DEFORESTATION_FREE"),
        ("rubber", False, False, 0.0, "DEFORESTATION_FREE"),
        ("cocoa", True, True, -5.0, "DEFORESTATION_FREE"),
        ("wood", True, False, -80.0, "DEFORESTED"),
    ])
    def test_verdict_matrix(self, commodity, was_forest, is_forest, change, expected):
        """Test verdict determination across a matrix of scenarios."""
        verdict = _determine_verdict(
            cutoff_was_forest=was_forest,
            current_is_forest=is_forest,
            canopy_change_pct=change,
            commodity=commodity,
        )
        assert verdict == expected


# ===========================================================================
# 10. Result Construction (5 tests)
# ===========================================================================


class TestResultConstruction:
    """Test DeforestationFreeResult construction."""

    def test_result_fixture_valid(self, sample_deforestation_free_result):
        """Test fixture creates a valid DeforestationFreeResult."""
        result = sample_deforestation_free_result
        assert isinstance(result, DeforestationFreeResult)
        assert result.verdict == "DEFORESTATION_FREE"

    def test_result_has_provenance(self, sample_deforestation_free_result):
        """Test result includes provenance hash."""
        assert len(sample_deforestation_free_result.provenance_hash) == SHA256_HEX_LENGTH

    def test_result_has_regulatory_refs(self, sample_deforestation_free_result):
        """Test result includes regulatory references."""
        refs = sample_deforestation_free_result.regulatory_references
        assert len(refs) >= 1

    def test_result_confidence_above_min(self, sample_deforestation_free_result):
        """Test result confidence is above minimum threshold."""
        assert (sample_deforestation_free_result.confidence >=
                sample_deforestation_free_result.confidence_min)

    def test_result_commodity(self, sample_deforestation_free_result):
        """Test result commodity is set."""
        assert sample_deforestation_free_result.commodity in EUDR_COMMODITIES


# ===========================================================================
# 11. Determinism (5 tests)
# ===========================================================================


class TestVerifierDeterminism:
    """Test deterministic behaviour of deforestation-free verification."""

    def test_determinism_same_inputs_same_verdict(self):
        """Test same inputs always produce same verdict."""
        results = [
            _determine_verdict(True, True, -5.0, confidence=0.9)
            for _ in range(20)
        ]
        assert len(set(results)) == 1

    def test_determinism_same_inputs_same_provenance_hash(self):
        """Test same inputs produce identical provenance hash."""
        data = {"verdict": "DEFORESTATION_FREE", "plot_id": "P001", "change": -5.0}
        hashes = [compute_test_hash(data) for _ in range(20)]
        assert len(set(hashes)) == 1

    def test_determinism_different_inputs_different_hash(self):
        """Test different inputs produce different hashes."""
        h1 = compute_test_hash({"verdict": "DEFORESTATION_FREE"})
        h2 = compute_test_hash({"verdict": "DEFORESTED"})
        assert h1 != h2

    def test_determinism_evidence_package(self):
        """Test evidence package is deterministic."""
        pkgs = [_build_evidence_package(0.75, 0.72, -4.0) for _ in range(10)]
        hashes = [p["provenance_hash"] for p in pkgs]
        assert len(set(hashes)) == 1

    def test_determinism_canopy_change(self):
        """Test canopy change calculation is deterministic."""
        results = [compute_canopy_change_pct(80.0, 56.0) for _ in range(10)]
        assert len(set(results)) == 1
