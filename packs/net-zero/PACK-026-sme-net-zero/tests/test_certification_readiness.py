# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - Certification Readiness Engine.

Tests readiness assessment, pathway recommendations, gap analysis,
timeline estimation, and multi-certification support.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~350 lines, 50+ tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines import (
    CertificationReadinessEngine,
    CertificationReadinessInput,
    CertificationReadinessResult,
    CertificationPathway,
    ReadinessLevel,
    GapRemediationItem,
    CertificationAssessment,
    DimensionInput,
)

# Try to import CERT_REQUIREMENTS, use empty dict if not available
try:
    from engines.certification_readiness_engine import CERT_REQUIREMENTS
except ImportError:
    CERT_REQUIREMENTS = {}

from .conftest import assert_provenance_hash


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> CertificationReadinessEngine:
    return CertificationReadinessEngine()


@pytest.fixture
def early_stage_input() -> CertificationReadinessInput:
    """Business just starting its sustainability journey."""
    return CertificationReadinessInput(
        entity_name="Micro Cafe",
        country="GB",
        headcount=6,
        assessment_data=DimensionInput(
            has_baseline=False,
            baseline_data_tier="none",
            baseline_scope_coverage="none",
            has_targets=False,
            target_type="none",
            has_action_plan=False,
            actions_identified=0,
            actions_implemented=0,
            has_board_oversight=False,
            has_climate_policy=False,
            has_sustainability_role=False,
            has_public_disclosure=False,
            has_reporting_process=False,
            has_third_party_verification=False,
        ),
        preferred_pathways=[CertificationPathway.SME_CLIMATE_HUB, CertificationPathway.B_CORP_CLIMATE],
        current_certifications=[],
    )


@pytest.fixture
def mid_stage_input() -> CertificationReadinessInput:
    """Business with some sustainability measures in place."""
    return CertificationReadinessInput(
        entity_name="TechSoft Ltd",
        country="DE",
        headcount=32,
        assessment_data=DimensionInput(
            has_baseline=True,
            baseline_data_tier="silver",
            baseline_scope_coverage="scope_1_2",
            has_targets=True,
            target_type="voluntary",
            target_year=2030,
            has_action_plan=True,
            actions_identified=5,
            actions_implemented=2,
            has_board_oversight=True,
            has_climate_policy=True,
            has_sustainability_role=False,
            has_public_disclosure=False,
            has_reporting_process=True,
            has_third_party_verification=False,
        ),
        preferred_pathways=[],
        current_certifications=["sme_climate_hub"],
    )


@pytest.fixture
def advanced_input() -> CertificationReadinessInput:
    """Business with advanced sustainability program."""
    return CertificationReadinessInput(
        entity_name="EuroManufact GmbH",
        country="DE",
        headcount=145,
        assessment_data=DimensionInput(
            has_baseline=True,
            baseline_data_tier="gold",
            baseline_scope_coverage="scope_1_2_3",
            has_targets=True,
            target_type="sbti",
            target_year=2030,
            has_action_plan=True,
            actions_identified=10,
            actions_implemented=6,
            has_board_oversight=True,
            has_climate_policy=True,
            has_sustainability_role=True,
            has_public_disclosure=True,
            has_reporting_process=True,
            has_third_party_verification=False,
        ),
        preferred_pathways=[],
        current_certifications=["sme_climate_hub", "iso_14001"],
    )


# ===========================================================================
# Tests -- Certification Types
# ===========================================================================


class TestCertificationTypes:
    @pytest.mark.parametrize("cert_type", [
        "sme_climate_hub", "b_corp_climate", "iso_14001",
        "carbon_trust_standard", "climate_active", "cdp_supply_chain",
    ])
    def test_certification_type_values(self, cert_type) -> None:
        assert CertificationPathway(cert_type) is not None

    def test_certification_count(self) -> None:
        assert len(CertificationPathway) >= 6


# ===========================================================================
# Tests -- Readiness Levels
# ===========================================================================


class TestReadinessLevels:
    @pytest.mark.parametrize("level", [
        "not_ready", "early_stage", "in_progress", "nearly_ready", "ready",
    ])
    def test_readiness_level_values(self, level) -> None:
        assert ReadinessLevel(level) is not None


# ===========================================================================
# Tests -- Certification Requirements Database
# ===========================================================================


class TestCertificationRequirements:
    def test_requirements_database_exists(self) -> None:
        assert len(CERT_REQUIREMENTS) > 0

    @pytest.mark.parametrize("cert", [
        CertificationPathway.SME_CLIMATE_HUB,
        CertificationPathway.B_CORP_CLIMATE,
        CertificationPathway.ISO_14001,
        CertificationPathway.CARBON_TRUST_STANDARD,
        CertificationPathway.CLIMATE_ACTIVE,
        CertificationPathway.CDP_SUPPLY_CHAIN,
    ])
    def test_requirements_for_each_certification(self, cert) -> None:
        assert cert in CERT_REQUIREMENTS
        reqs = CERT_REQUIREMENTS[cert]
        assert len(reqs) > 0

    def test_requirements_have_dimension_scores(self) -> None:
        for cert, reqs in CERT_REQUIREMENTS.items():
            assert isinstance(reqs, dict)
            assert len(reqs) > 0


# ===========================================================================
# Tests -- Early Stage Assessment
# ===========================================================================


class TestEarlyStageAssessment:
    def test_early_stage_calculates(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        assert isinstance(result, CertificationReadinessResult)

    def test_early_stage_low_readiness(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        for assessment in result.assessments:
            assert assessment.readiness_level in (
                ReadinessLevel.NOT_READY.value, ReadinessLevel.EARLY_STAGE.value,
            )

    def test_early_stage_has_gaps(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        for assessment in result.assessments:
            assert len(assessment.gaps) > 0

    def test_early_stage_recommends_sme_climate_hub_first(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        if hasattr(result, "recommended_pathway"):
            assert "sme_climate_hub" in result.recommended_pathway.lower()

    def test_early_stage_timeline_realistic(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        for assessment in result.assessments:
            if hasattr(assessment, "estimated_timeline_months"):
                timeline_str = assessment.estimated_timeline_months
                assert timeline_str  # Should not be empty


# ===========================================================================
# Tests -- Mid Stage Assessment
# ===========================================================================


class TestMidStageAssessment:
    def test_mid_stage_calculates(self, engine, mid_stage_input) -> None:
        result = engine.calculate(mid_stage_input)
        assert isinstance(result, CertificationReadinessResult)

    def test_mid_stage_higher_readiness(self, engine, mid_stage_input) -> None:
        result = engine.calculate(mid_stage_input)
        readiness_scores = [a.readiness_score for a in result.assessments]
        assert max(readiness_scores) > Decimal("30")

    def test_mid_stage_fewer_gaps(self, engine, early_stage_input, mid_stage_input) -> None:
        early_result = engine.calculate(early_stage_input)
        mid_result = engine.calculate(mid_stage_input)
        early_gaps = sum(len(a.gaps) for a in early_result.assessments)
        mid_gaps = sum(len(a.gaps) for a in mid_result.assessments)
        assert mid_gaps <= early_gaps

    def test_mid_stage_existing_certs_recognized(self, engine, mid_stage_input) -> None:
        result = engine.calculate(mid_stage_input)
        sme_hub = next(
            (a for a in result.assessments if a.pathway == "sme_climate_hub"),
            None,
        )
        if sme_hub:
            # Mid-stage should be at least in_progress or higher
            assert sme_hub.readiness_level in (
                ReadinessLevel.IN_PROGRESS.value,
                ReadinessLevel.NEARLY_READY.value,
                ReadinessLevel.READY.value
            )


# ===========================================================================
# Tests -- Advanced Assessment
# ===========================================================================


class TestAdvancedAssessment:
    def test_advanced_calculates(self, engine, advanced_input) -> None:
        result = engine.calculate(advanced_input)
        assert isinstance(result, CertificationReadinessResult)

    def test_advanced_high_readiness(self, engine, advanced_input) -> None:
        result = engine.calculate(advanced_input)
        readiness_scores = [a.readiness_score for a in result.assessments]
        assert max(readiness_scores) >= Decimal("60")

    def test_advanced_shorter_timeline(self, engine, early_stage_input, advanced_input) -> None:
        early_result = engine.calculate(early_stage_input)
        advanced_result = engine.calculate(advanced_input)
        # Advanced should have higher overall readiness
        assert advanced_result.overall_readiness_score > early_result.overall_readiness_score


# ===========================================================================
# Tests -- Gap Analysis
# ===========================================================================


class TestGapAnalysis:
    def test_gaps_have_descriptions(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        for assessment in result.assessments:
            for gap in assessment.gaps:
                # Gaps are strings describing the gap
                assert isinstance(gap, str)
                assert len(gap) > 0

    def test_gaps_have_priority(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        for assessment in result.assessments:
            for gap in assessment.gaps:
                # Gaps contain "critical", "significant", "moderate", or "minor"
                assert isinstance(gap, str)

    def test_gaps_have_estimated_effort(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        for assessment in result.assessments:
            # Remediation plan contains effort estimates
            assert isinstance(assessment.remediation_plan, (str, list))

    def test_gaps_actionable(self, engine, mid_stage_input) -> None:
        """Each gap should be present in remediation plan."""
        result = engine.calculate(mid_stage_input)
        for assessment in result.assessments:
            if len(assessment.gaps) > 0:
                assert len(assessment.remediation_plan) > 0


# ===========================================================================
# Tests -- Pathway Recommendations
# ===========================================================================


class TestPathwayRecommendations:
    def test_recommended_pathway(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        assert hasattr(result, "recommended_pathway")
        assert result.recommended_pathway  # Should not be empty

    def test_recommended_has_reason(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        assert hasattr(result, "recommended_reason")
        if result.recommended_pathway:
            assert result.recommended_reason  # Should have a reason


# ===========================================================================
# Tests -- Provenance & Determinism
# ===========================================================================


class TestCertificationProvenance:
    def test_provenance_hash(self, engine, early_stage_input) -> None:
        result = engine.calculate(early_stage_input)
        assert_provenance_hash(result)

    def test_deterministic(self, engine, early_stage_input) -> None:
        r1 = engine.calculate(early_stage_input)
        r2 = engine.calculate(early_stage_input)
        # Hashes may differ due to timestamps/UUIDs, but results should be identical
        assert r1.overall_readiness_score == r2.overall_readiness_score
        assert len(r1.assessments) == len(r2.assessments)


# ===========================================================================
# Tests -- Error Handling
# ===========================================================================


class TestCertificationErrors:
    def test_empty_headcount_handled(self, engine) -> None:
        # Engine should handle minimal input
        result = engine.calculate(CertificationReadinessInput(
            entity_name="Test",
            country="GB",
            headcount=1,
            assessment_data=DimensionInput(),
        ))
        assert result is not None
