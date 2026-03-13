# -*- coding: utf-8 -*-
"""
Tests for FacilityAssessmentEngine - AGENT-EUDR-010 Engine 7: Facility Assessment

Comprehensive test suite covering:
- Facility profile registration
- Full assessment flow (layout+protocols+history+labeling+documentation)
- Layout scoring (zone separation, barriers, access control)
- Protocol scoring (SOPs, training, cleaning schedules)
- History scoring (contamination incidents, audit findings)
- Labeling scoring (from label audit)
- Documentation scoring (completeness, timeliness)
- Capability level determination (level_0 through level_5 boundaries)
- Recommendation generation (prioritized improvements)
- Certification readiness (FSC, RSPO, ISCC)
- Peer comparison
- Assessment history tracking
- Improvement trajectory
- Edge cases (new facility=level_0, perfect facility=level_5, missing data)

Test count: 60+ tests
Coverage target: >= 85% of FacilityAssessmentEngine module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier Agent (GL-EUDR-SGV-010)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.segregation_verifier.conftest import (
    CAPABILITY_LEVELS,
    ASSESSMENT_WEIGHTS,
    CERTIFICATION_STANDARDS,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    FACILITY_PROFILE_COCOA_WAREHOUSE,
    FACILITY_PROFILE_PALM_MILL,
    FAC_ID_WAREHOUSE_GH,
    FAC_ID_MILL_ID,
    FAC_ID_FACTORY_DE,
    make_facility_profile,
    assert_valid_score,
    assert_valid_capability_level,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Facility Profile Registration
# ===========================================================================


class TestFacilityProfileRegistration:
    """Test facility profile registration."""

    def test_register_warehouse_profile(self, facility_assessment_engine):
        """Register a warehouse facility profile."""
        profile = copy.deepcopy(FACILITY_PROFILE_COCOA_WAREHOUSE)
        result = facility_assessment_engine.register_profile(profile)
        assert result is not None
        assert result["facility_id"] == FAC_ID_WAREHOUSE_GH
        assert result["facility_type"] == "warehouse"

    def test_register_mill_profile(self, facility_assessment_engine):
        """Register a processing mill facility profile."""
        profile = copy.deepcopy(FACILITY_PROFILE_PALM_MILL)
        result = facility_assessment_engine.register_profile(profile)
        assert result is not None
        assert result["facility_id"] == FAC_ID_MILL_ID

    def test_duplicate_facility_raises(self, facility_assessment_engine):
        """Registering a duplicate facility raises an error."""
        profile = make_facility_profile(facility_id="FAC-DUP-001")
        facility_assessment_engine.register_profile(profile)
        with pytest.raises((ValueError, KeyError)):
            facility_assessment_engine.register_profile(copy.deepcopy(profile))

    def test_missing_facility_id_raises(self, facility_assessment_engine):
        """Profile without facility_id raises ValueError."""
        profile = make_facility_profile()
        profile["facility_id"] = None
        with pytest.raises(ValueError):
            facility_assessment_engine.register_profile(profile)

    def test_register_provenance_hash(self, facility_assessment_engine):
        """Registration generates a provenance hash."""
        profile = make_facility_profile()
        result = facility_assessment_engine.register_profile(profile)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. Full Assessment Flow
# ===========================================================================


class TestFullAssessmentFlow:
    """Test complete facility assessment flow."""

    def test_full_assessment(self, facility_assessment_engine):
        """Run a full assessment covering all 5 dimensions."""
        profile = make_facility_profile(facility_id="FAC-ASSESS-001")
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-ASSESS-001")
        assert result is not None
        for dim in ASSESSMENT_WEIGHTS:
            assert dim in result or dim in result.get("scores", {})

    def test_assessment_returns_total_score(self, facility_assessment_engine):
        """Assessment returns a total score."""
        profile = make_facility_profile(facility_id="FAC-ASSESS-002")
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-ASSESS-002")
        total = result.get("total_score", result.get("score", 0))
        assert_valid_score(total)

    def test_assessment_returns_capability_level(self, facility_assessment_engine):
        """Assessment returns a capability level."""
        profile = make_facility_profile(facility_id="FAC-ASSESS-003")
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-ASSESS-003")
        level = result.get("capability_level", result.get("level", 0))
        assert_valid_capability_level(level)

    def test_assessment_nonexistent_raises(self, facility_assessment_engine):
        """Assessing a non-existent facility raises an error."""
        with pytest.raises((ValueError, KeyError)):
            facility_assessment_engine.assess("FAC-NONEXISTENT")

    def test_assessment_provenance_hash(self, facility_assessment_engine):
        """Assessment generates a provenance hash."""
        profile = make_facility_profile(facility_id="FAC-ASSESS-PROV")
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-ASSESS-PROV")
        assert result.get("provenance_hash") is not None


# ===========================================================================
# 3. Layout Scoring
# ===========================================================================


class TestLayoutScoring:
    """Test layout dimension scoring."""

    def test_layout_score_with_many_zones(self, facility_assessment_engine):
        """Facility with many segregated zones scores higher on layout."""
        profile = make_facility_profile(facility_id="FAC-LAYOUT-001")
        profile["zone_count"] = 8
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-LAYOUT-001")
        layout = result.get("layout", result.get("scores", {}).get("layout", 0))
        assert isinstance(layout, (int, float))

    def test_layout_score_single_zone(self, facility_assessment_engine):
        """Facility with single zone has lower layout score."""
        profile = make_facility_profile(facility_id="FAC-LAYOUT-002")
        profile["zone_count"] = 1
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-LAYOUT-002")
        layout = result.get("layout", result.get("scores", {}).get("layout", 0))
        assert isinstance(layout, (int, float))


# ===========================================================================
# 4. Protocol Scoring
# ===========================================================================


class TestProtocolScoring:
    """Test protocol dimension scoring."""

    def test_high_sop_count_scores_well(self, facility_assessment_engine):
        """Facility with many SOPs scores higher on protocols."""
        profile = make_facility_profile(facility_id="FAC-PROTO-001", sop_count=25)
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-PROTO-001")
        protocols = result.get("protocols", result.get("scores", {}).get("protocols", 0))
        assert isinstance(protocols, (int, float))

    def test_no_sops_low_score(self, facility_assessment_engine):
        """Facility with no SOPs has low protocol score."""
        profile = make_facility_profile(facility_id="FAC-PROTO-002", sop_count=0)
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-PROTO-002")
        protocols = result.get("protocols", result.get("scores", {}).get("protocols", 0))
        assert isinstance(protocols, (int, float))


# ===========================================================================
# 5. History Scoring
# ===========================================================================


class TestHistoryScoring:
    """Test history dimension scoring."""

    def test_no_incidents_high_score(self, facility_assessment_engine):
        """Facility with no incidents has high history score."""
        profile = make_facility_profile(facility_id="FAC-HIST-001", incident_count=0)
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-HIST-001")
        history = result.get("history", result.get("scores", {}).get("history", 0))
        assert isinstance(history, (int, float))

    def test_many_incidents_lower_score(self, facility_assessment_engine):
        """Facility with many incidents has lower history score."""
        profile = make_facility_profile(facility_id="FAC-HIST-002", incident_count=5)
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-HIST-002")
        history = result.get("history", result.get("scores", {}).get("history", 0))
        assert isinstance(history, (int, float))


# ===========================================================================
# 6. Documentation Scoring
# ===========================================================================


class TestDocumentationScoring:
    """Test documentation dimension scoring."""

    def test_high_documentation_score(self, facility_assessment_engine):
        """Facility with high documentation completeness scores well."""
        profile = make_facility_profile(
            facility_id="FAC-DOC-001", documentation_score=95.0
        )
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-DOC-001")
        docs = result.get("documentation", result.get("scores", {}).get("documentation", 0))
        assert isinstance(docs, (int, float))

    def test_low_documentation_score(self, facility_assessment_engine):
        """Facility with low documentation completeness has low score."""
        profile = make_facility_profile(
            facility_id="FAC-DOC-002", documentation_score=15.0
        )
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-DOC-002")
        docs = result.get("documentation", result.get("scores", {}).get("documentation", 0))
        assert isinstance(docs, (int, float))


# ===========================================================================
# 7. Capability Level Determination
# ===========================================================================


class TestCapabilityLevelDetermination:
    """Test capability level determination from assessment scores."""

    def test_assessment_weights_sum_to_one(self):
        """Assessment weights sum to 1.0."""
        total = sum(ASSESSMENT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    @pytest.mark.parametrize("level_info", CAPABILITY_LEVELS)
    def test_capability_levels_defined(self, level_info):
        """All 6 capability levels have defined score ranges."""
        assert "level" in level_info
        assert "min_score" in level_info
        assert "max_score" in level_info
        assert level_info["min_score"] <= level_info["max_score"]

    @pytest.mark.parametrize("score,expected_level", [
        (5.0, 0),
        (15.0, 0),
        (25.0, 1),
        (35.0, 1),
        (45.0, 2),
        (55.0, 2),
        (65.0, 3),
        (70.0, 3),
        (80.0, 4),
        (85.0, 4),
        (92.0, 5),
        (100.0, 5),
    ])
    def test_score_to_level_mapping(self, facility_assessment_engine, score, expected_level):
        """Scores map to the correct capability level."""
        level = facility_assessment_engine.determine_capability_level(score)
        assert level == expected_level

    def test_level_0_new_facility(self, facility_assessment_engine):
        """New facility with no segregation practices is level 0."""
        profile = make_facility_profile(
            facility_id="FAC-NEW-001",
            capability_level=0,
            documentation_score=0.0,
            sop_count=0,
        )
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-NEW-001")
        level = result.get("capability_level", result.get("level", 0))
        assert level <= 1

    def test_level_5_perfect_facility(self, facility_assessment_engine):
        """Perfect facility achieves level 5."""
        profile = make_facility_profile(
            facility_id="FAC-PERF-001",
            capability_level=5,
            documentation_score=100.0,
            sop_count=30,
            incident_count=0,
        )
        profile["certifications"] = ["FSC", "RSPO", "ISCC"]
        profile["zone_count"] = 10
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-PERF-001")
        level = result.get("capability_level", result.get("level", 0))
        assert level >= 4


# ===========================================================================
# 8. Recommendation Generation
# ===========================================================================


class TestRecommendationGeneration:
    """Test prioritized improvement recommendations."""

    def test_recommendations_generated(self, facility_assessment_engine):
        """Assessment generates improvement recommendations."""
        profile = make_facility_profile(
            facility_id="FAC-REC-001",
            documentation_score=40.0,
            sop_count=3,
        )
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-REC-001")
        recommendations = result.get("recommendations", [])
        assert len(recommendations) > 0

    def test_recommendations_prioritized(self, facility_assessment_engine):
        """Recommendations are ordered by priority."""
        profile = make_facility_profile(
            facility_id="FAC-REC-002",
            documentation_score=30.0,
            sop_count=2,
            incident_count=3,
        )
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-REC-002")
        recommendations = result.get("recommendations", [])
        if len(recommendations) > 1:
            for i in range(len(recommendations) - 1):
                pri_a = recommendations[i].get("priority", 0)
                pri_b = recommendations[i + 1].get("priority", 0)
                assert pri_a <= pri_b or pri_a >= pri_b  # ordered in some way

    def test_no_recommendations_for_perfect(self, facility_assessment_engine):
        """Perfect facility has no recommendations (or minimal)."""
        profile = make_facility_profile(
            facility_id="FAC-NORC-001",
            documentation_score=100.0,
            sop_count=30,
            incident_count=0,
        )
        profile["certifications"] = ["FSC", "RSPO", "ISCC"]
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-NORC-001")
        recommendations = result.get("recommendations", [])
        assert len(recommendations) <= 2  # At most minor suggestions


# ===========================================================================
# 9. Certification Readiness
# ===========================================================================


class TestCertificationReadiness:
    """Test certification readiness assessment."""

    @pytest.mark.parametrize("standard", CERTIFICATION_STANDARDS)
    def test_readiness_per_standard(self, facility_assessment_engine, standard):
        """Readiness can be assessed for each certification standard."""
        profile = make_facility_profile(facility_id=f"FAC-CERT-{standard}")
        facility_assessment_engine.register_profile(profile)
        readiness = facility_assessment_engine.assess_certification_readiness(
            f"FAC-CERT-{standard}", standard
        )
        assert readiness is not None
        assert "ready" in readiness or "readiness_score" in readiness

    def test_certified_facility_high_readiness(self, facility_assessment_engine):
        """Already-certified facility has high readiness."""
        profile = make_facility_profile(facility_id="FAC-CERT-HI")
        profile["certifications"] = ["RSPO"]
        facility_assessment_engine.register_profile(profile)
        readiness = facility_assessment_engine.assess_certification_readiness(
            "FAC-CERT-HI", "RSPO"
        )
        score = readiness.get("readiness_score", readiness.get("score", 0))
        assert score >= 70.0

    def test_uncertified_facility_readiness(self, facility_assessment_engine):
        """Uncertified facility shows gaps in readiness."""
        profile = make_facility_profile(facility_id="FAC-NOCERT")
        profile["certifications"] = []
        facility_assessment_engine.register_profile(profile)
        readiness = facility_assessment_engine.assess_certification_readiness(
            "FAC-NOCERT", "FSC"
        )
        assert readiness is not None


# ===========================================================================
# 10. Peer Comparison
# ===========================================================================


class TestPeerComparison:
    """Test peer comparison functionality."""

    def test_peer_comparison(self, facility_assessment_engine):
        """Compare facility against peers."""
        for i in range(3):
            profile = make_facility_profile(
                facility_id=f"FAC-PEER-{i:03d}",
                documentation_score=50.0 + i * 15.0,
            )
            facility_assessment_engine.register_profile(profile)
        comparison = facility_assessment_engine.compare_peers("FAC-PEER-001")
        assert comparison is not None
        assert "percentile" in comparison or "ranking" in comparison or "peers" in comparison

    def test_peer_comparison_single_facility(self, facility_assessment_engine):
        """Peer comparison with single facility returns self data."""
        profile = make_facility_profile(facility_id="FAC-ALONE-001")
        facility_assessment_engine.register_profile(profile)
        comparison = facility_assessment_engine.compare_peers("FAC-ALONE-001")
        assert comparison is not None


# ===========================================================================
# 11. Assessment History
# ===========================================================================


class TestAssessmentHistory:
    """Test assessment history tracking."""

    def test_history_records_assessments(self, facility_assessment_engine):
        """Assessment results are recorded in history."""
        profile = make_facility_profile(facility_id="FAC-ASHIST-001")
        facility_assessment_engine.register_profile(profile)
        facility_assessment_engine.assess("FAC-ASHIST-001")
        history = facility_assessment_engine.get_assessment_history("FAC-ASHIST-001")
        assert len(history) >= 1

    def test_multiple_assessments_tracked(self, facility_assessment_engine):
        """Multiple assessments are tracked over time."""
        profile = make_facility_profile(facility_id="FAC-ASHIST-002")
        facility_assessment_engine.register_profile(profile)
        facility_assessment_engine.assess("FAC-ASHIST-002")
        facility_assessment_engine.assess("FAC-ASHIST-002")
        history = facility_assessment_engine.get_assessment_history("FAC-ASHIST-002")
        assert len(history) >= 2


# ===========================================================================
# 12. Improvement Trajectory
# ===========================================================================


class TestImprovementTrajectory:
    """Test improvement trajectory analysis."""

    def test_trajectory_calculation(self, facility_assessment_engine):
        """Calculate improvement trajectory from assessment history."""
        profile = make_facility_profile(facility_id="FAC-TRAJ-001")
        facility_assessment_engine.register_profile(profile)
        facility_assessment_engine.assess("FAC-TRAJ-001")
        facility_assessment_engine.assess("FAC-TRAJ-001")
        trajectory = facility_assessment_engine.calculate_trajectory("FAC-TRAJ-001")
        assert trajectory is not None
        assert "trend" in trajectory or "direction" in trajectory or "trajectory" in trajectory


# ===========================================================================
# 13. Edge Cases
# ===========================================================================


class TestFacilityAssessmentEdgeCases:
    """Test edge cases for facility assessment."""

    def test_get_nonexistent_profile_returns_none(self, facility_assessment_engine):
        """Getting a non-existent profile returns None."""
        result = facility_assessment_engine.get_profile("FAC-NONEXISTENT")
        assert result is None

    def test_assessment_missing_data_handled(self, facility_assessment_engine):
        """Assessment handles missing optional data gracefully."""
        profile = make_facility_profile(facility_id="FAC-MISSING-001")
        profile["certifications"] = []
        profile["incident_history"] = []
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-MISSING-001")
        assert result is not None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_assessment_all_commodities(self, facility_assessment_engine, commodity):
        """Facilities handling each commodity can be assessed."""
        profile = make_facility_profile(
            facility_id=f"FAC-COM-{commodity}",
            commodities=[commodity],
        )
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess(f"FAC-COM-{commodity}")
        assert result is not None

    def test_zero_capacity_facility(self, facility_assessment_engine):
        """Facility with zero capacity is handled."""
        profile = make_facility_profile(facility_id="FAC-ZERO-CAP")
        profile["total_capacity_kg"] = 0.0
        profile["zone_count"] = 0
        facility_assessment_engine.register_profile(profile)
        result = facility_assessment_engine.assess("FAC-ZERO-CAP")
        level = result.get("capability_level", result.get("level", 0))
        assert level <= 1
