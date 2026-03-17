# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Stakeholder Engagement Engine Tests
===================================================================================

Unit tests for StakeholderEngagementEngine (Engine 3) covering stakeholder
identification, priority mapping, consultation recording, coverage calculation,
engagement quality assessment, findings synthesis, and provenance hashing.

ESRS 1 Para 22-23: stakeholder engagement in materiality assessment.

Target: 45+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the stakeholder_engagement engine module."""
    return _load_engine("stakeholder_engagement")


@pytest.fixture
def engine(mod):
    """Create a fresh StakeholderEngagementEngine instance."""
    return mod.StakeholderEngagementEngine()


@pytest.fixture
def sample_stakeholder(mod):
    """Create a sample Stakeholder for testing."""
    return mod.Stakeholder(
        id="SH-001",
        name="Employee Works Council",
        category=mod.StakeholderCategory.EMPLOYEES,
        is_affected_stakeholder=True,
        influence_level=4,
        impact_level=5,
        topics_of_interest=["e1_climate", "s1_own_workforce"],
    )


@pytest.fixture
def sample_consultation(mod):
    """Create a sample ConsultationRecord for testing."""
    return mod.ConsultationRecord(
        stakeholder_id="SH-001",
        date="2025-06-15",
        method=mod.EngagementMethod.WORKSHOP,
        topics_discussed=["e1_climate", "s1_own_workforce"],
        key_findings=["Employees prioritise health and safety"],
        attendance_count=25,
        documentation_available=True,
        follow_up_actions=["action1", "action2"],
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestStakeholderEnums:
    """Tests for stakeholder engagement enums."""

    def test_stakeholder_category_count(self, mod):
        """StakeholderCategory has 10 members."""
        assert len(mod.StakeholderCategory) == 10

    def test_stakeholder_category_includes_employees(self, mod):
        """EMPLOYEES is in StakeholderCategory."""
        names = {m.name for m in mod.StakeholderCategory}
        assert "EMPLOYEES" in names

    def test_engagement_method_count(self, mod):
        """EngagementMethod has 8 methods."""
        assert len(mod.EngagementMethod) == 8

    def test_engagement_method_includes_survey(self, mod):
        """SURVEY is in EngagementMethod."""
        names = {m.name for m in mod.EngagementMethod}
        assert "SURVEY" in names

    def test_consultation_status_count(self, mod):
        """ConsultationStatus has 5 statuses."""
        assert len(mod.ConsultationStatus) == 5


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestStakeholderConstants:
    """Tests for stakeholder engagement constants."""

    def test_engagement_method_quality_has_all_methods(self, mod):
        """ENGAGEMENT_METHOD_QUALITY has entries for all EngagementMethod values."""
        for method in mod.EngagementMethod:
            assert method.value in mod.ENGAGEMENT_METHOD_QUALITY, (
                f"Missing quality score for {method.value}"
            )

    def test_engagement_method_quality_range(self, mod):
        """All quality scores are between 0.0 and 1.0."""
        for method, quality in mod.ENGAGEMENT_METHOD_QUALITY.items():
            assert 0.0 <= quality <= 1.0, (
                f"Quality score {quality} for {method} out of range"
            )

    def test_survey_quality_lowest(self, mod):
        """Survey has a low quality score."""
        assert float(mod.ENGAGEMENT_METHOD_QUALITY["survey"]) == pytest.approx(0.40)

    def test_advisory_panel_quality_highest(self, mod):
        """Advisory panel has the highest quality score (0.95)."""
        assert float(mod.ENGAGEMENT_METHOD_QUALITY["advisory_panel"]) == pytest.approx(0.95)

    def test_engagement_quality_criteria_weights_sum(self, mod):
        """ENGAGEMENT_QUALITY_CRITERIA weights sum to 1.0."""
        total = sum(mod.ENGAGEMENT_QUALITY_CRITERIA.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_sector_stakeholder_map_has_default(self, mod):
        """SECTOR_STAKEHOLDER_MAP has a 'default' entry."""
        assert "default" in mod.SECTOR_STAKEHOLDER_MAP

    def test_sector_stakeholder_map_has_14_sectors(self, mod):
        """SECTOR_STAKEHOLDER_MAP has at least 14 sectors plus default."""
        assert len(mod.SECTOR_STAKEHOLDER_MAP) >= 14

    def test_priority_threshold(self, mod):
        """PRIORITY_THRESHOLD is 3."""
        assert mod.PRIORITY_THRESHOLD == 3


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================


class TestStakeholderModel:
    """Tests for Stakeholder Pydantic model."""

    def test_create_valid_stakeholder(self, mod):
        """Create a valid Stakeholder."""
        sh = mod.Stakeholder(
            name="Test Stakeholder",
            category=mod.StakeholderCategory.INVESTORS,
            is_affected_stakeholder=False,
            influence_level=3,
            impact_level=2,
        )
        assert sh.name == "Test Stakeholder"
        assert sh.is_affected_stakeholder is False

    def test_stakeholder_influence_range(self, mod):
        """Influence level must be 1-5."""
        with pytest.raises(Exception):
            mod.Stakeholder(
                name="Test", category=mod.StakeholderCategory.INVESTORS,
                influence_level=0, impact_level=3,
            )

    def test_stakeholder_impact_range(self, mod):
        """Impact level must be 1-5."""
        with pytest.raises(Exception):
            mod.Stakeholder(
                name="Test", category=mod.StakeholderCategory.INVESTORS,
                influence_level=3, impact_level=6,
            )


class TestConsultationRecordModel:
    """Tests for ConsultationRecord Pydantic model."""

    def test_create_valid_consultation(self, mod):
        """Create a valid ConsultationRecord."""
        cr = mod.ConsultationRecord(
            stakeholder_id="SH-001",
            date="2025-06-15",
            method=mod.EngagementMethod.INTERVIEW,
            topics_discussed=["e1_climate"],
            attendance_count=5,
        )
        assert cr.stakeholder_id == "SH-001"
        assert cr.attendance_count == 5

    def test_consultation_requires_stakeholder_id(self, mod):
        """ConsultationRecord requires stakeholder_id."""
        with pytest.raises(Exception):
            mod.ConsultationRecord(
                stakeholder_id="",
                date="2025-06-15",
                method=mod.EngagementMethod.SURVEY,
                topics_discussed=["e1_climate"],
            )


# ===========================================================================
# Identify Stakeholders Tests
# ===========================================================================


class TestIdentifyStakeholders:
    """Tests for identify_stakeholders method."""

    def test_identify_stakeholders_default_sector(self, engine):
        """Default sector returns a list of stakeholder categories."""
        stakeholders = engine.identify_stakeholders(sector="default")
        assert isinstance(stakeholders, list)
        assert len(stakeholders) >= 1

    def test_identify_stakeholders_manufacturing(self, engine):
        """Manufacturing sector returns relevant stakeholders."""
        stakeholders = engine.identify_stakeholders(sector="manufacturing")
        assert isinstance(stakeholders, list)
        assert len(stakeholders) >= 1

    def test_identify_stakeholders_returns_strings(self, engine):
        """Returned items are stakeholder category strings."""
        stakeholders = engine.identify_stakeholders(sector="default")
        for sh in stakeholders:
            assert isinstance(sh, str)

    def test_identify_stakeholders_unknown_sector_uses_default(self, engine):
        """Unknown sector falls back to default mapping."""
        stakeholders = engine.identify_stakeholders(sector="unknown_sector")
        assert isinstance(stakeholders, list)


# ===========================================================================
# Priority Mapping Tests
# ===========================================================================


class TestPriorityMapping:
    """Tests for map_stakeholder_priority method."""

    def test_priority_mapping_returns_dict(self, engine, sample_stakeholder):
        """map_stakeholder_priority returns a dict with 4 quadrants."""
        # map_stakeholder_priority takes List[Stakeholder]
        priority = engine.map_stakeholder_priority([sample_stakeholder])
        assert isinstance(priority, dict)
        assert "high_high" in priority
        assert "high_low" in priority
        assert "low_high" in priority
        assert "low_low" in priority

    def test_high_influence_high_impact(self, engine, mod):
        """High influence (4) and high impact (5) -> high_high quadrant."""
        sh = mod.Stakeholder(
            name="Key Stakeholder",
            category=mod.StakeholderCategory.EMPLOYEES,
            influence_level=4, impact_level=5,
        )
        priority = engine.map_stakeholder_priority([sh])
        assert "Key Stakeholder" in priority["high_high"]

    def test_low_influence_low_impact(self, engine, mod):
        """Low influence (1) and low impact (1) -> low_low quadrant."""
        sh = mod.Stakeholder(
            name="Low Priority",
            category=mod.StakeholderCategory.NGOS,
            influence_level=1, impact_level=1,
        )
        priority = engine.map_stakeholder_priority([sh])
        assert "Low Priority" in priority["low_low"]

    def test_priority_deterministic(self, engine, sample_stakeholder):
        """Same stakeholder always gets same priority."""
        p1 = engine.map_stakeholder_priority([sample_stakeholder])
        p2 = engine.map_stakeholder_priority([sample_stakeholder])
        assert p1 == p2


# ===========================================================================
# Record Consultation Tests
# ===========================================================================


class TestRecordConsultation:
    """Tests for record_consultation method."""

    def test_record_consultation_basic(self, engine, mod):
        """Recording a consultation returns a ConsultationRecord."""
        sh = mod.Stakeholder(
            id="SH-REC-001",
            name="Record Test",
            category=mod.StakeholderCategory.INVESTORS,
            influence_level=3, impact_level=3,
        )
        result = engine.record_consultation(
            stakeholder=sh,
            date="2025-06-15",
            method=mod.EngagementMethod.SURVEY,
            topics_discussed=["e1_climate"],
            key_findings=["Key finding 1"],
            attendance_count=100,
        )
        assert result is not None
        assert result.stakeholder_id == sh.id

    def test_record_consultation_updates_status(self, engine, mod):
        """Recording a consultation updates stakeholder consultation_status."""
        sh = mod.Stakeholder(
            id="SH-REC-002",
            name="Status Test",
            category=mod.StakeholderCategory.INVESTORS,
            influence_level=3, impact_level=3,
            consultation_status=mod.ConsultationStatus.NOT_STARTED,
        )
        engine.record_consultation(
            stakeholder=sh,
            date="2025-06-15",
            method=mod.EngagementMethod.SURVEY,
            topics_discussed=["e1_climate"],
            key_findings=["Finding"],
        )
        # Status should be updated to IN_PROGRESS
        assert sh.consultation_status == mod.ConsultationStatus.IN_PROGRESS


# ===========================================================================
# Coverage Calculation Tests
# ===========================================================================


class TestCoverageCalculation:
    """Tests for calculate_coverage method."""

    def test_coverage_full_engagement(self, engine, mod):
        """All stakeholders engaged = 100% coverage."""
        stakeholders = [
            mod.Stakeholder(
                id=f"COV-{i}", name=f"S{i}",
                category=mod.StakeholderCategory.EMPLOYEES,
                influence_level=3, impact_level=3,
                consultation_status=mod.ConsultationStatus.COMPLETED,
            )
            for i in range(3)
        ]
        # calculate_coverage takes (stakeholders, records)
        coverage = engine.calculate_coverage(stakeholders, [])
        assert coverage is not None
        # COMPLETED status counts as engaged
        assert float(coverage) == pytest.approx(100.0, abs=1.0)

    def test_coverage_no_engagement(self, engine, mod):
        """No stakeholders engaged = 0% coverage."""
        stakeholders = [
            mod.Stakeholder(
                id=f"NOCOV-{i}", name=f"S{i}",
                category=mod.StakeholderCategory.EMPLOYEES,
                influence_level=3, impact_level=3,
                consultation_status=mod.ConsultationStatus.NOT_STARTED,
            )
            for i in range(3)
        ]
        coverage = engine.calculate_coverage(stakeholders, [])
        assert float(coverage) == pytest.approx(0.0, abs=1.0)


# ===========================================================================
# Engagement Quality Tests
# ===========================================================================


class TestEngagementQuality:
    """Tests for assess_engagement_quality method."""

    def test_engagement_quality_basic(self, engine, mod):
        """Assess engagement quality returns a Decimal score."""
        stakeholders = [
            mod.Stakeholder(
                id="EQ-1", name="Employees",
                category=mod.StakeholderCategory.EMPLOYEES,
                influence_level=4, impact_level=5,
            ),
            mod.Stakeholder(
                id="EQ-2", name="Investors",
                category=mod.StakeholderCategory.INVESTORS,
                influence_level=5, impact_level=3,
            ),
        ]
        consultations = [
            mod.ConsultationRecord(
                stakeholder_id="EQ-1",
                date="2025-06-15",
                method=mod.EngagementMethod.WORKSHOP,
                topics_discussed=["e1_climate", "s1_own_workforce"],
                key_findings=["Key finding"],
                attendance_count=25,
                documentation_available=True,
                follow_up_actions=["review_policy"],
            ),
            mod.ConsultationRecord(
                stakeholder_id="EQ-2",
                date="2025-06-20",
                method=mod.EngagementMethod.INTERVIEW,
                topics_discussed=["e1_climate"],
                key_findings=["Another finding"],
                attendance_count=5,
                documentation_available=True,
            ),
        ]
        result = engine.assess_engagement_quality(stakeholders, consultations)
        assert result is not None
        assert isinstance(result, Decimal)

    def test_engagement_quality_no_records(self, engine, mod):
        """No consultation records yields zero quality."""
        stakeholders = [
            mod.Stakeholder(
                id="EQZ-1", name="No Records",
                category=mod.StakeholderCategory.INVESTORS,
                influence_level=5, impact_level=5,
            ),
        ]
        result = engine.assess_engagement_quality(stakeholders, [])
        assert result == Decimal("0.000")


# ===========================================================================
# Synthesize Findings Tests
# ===========================================================================


class TestSynthesizeFindings:
    """Tests for synthesize_findings method."""

    def test_synthesize_findings_basic(self, engine, mod):
        """synthesize_findings returns a StakeholderEngagementResult."""
        stakeholders = [
            mod.Stakeholder(
                id="SF-1", name="Employees",
                category=mod.StakeholderCategory.EMPLOYEES,
                influence_level=4, impact_level=5,
            ),
        ]
        consultations = [
            mod.ConsultationRecord(
                stakeholder_id="SF-1",
                date="2025-06-15",
                method=mod.EngagementMethod.WORKSHOP,
                topics_discussed=["e1_climate"],
                key_findings=["Important finding"],
                attendance_count=25,
            ),
        ]
        result = engine.synthesize_findings(stakeholders, consultations)
        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64

    def test_synthesize_findings_empty_stakeholders_raises(self, engine, mod):
        """Empty stakeholders list raises ValueError."""
        with pytest.raises(ValueError):
            engine.synthesize_findings([], [
                mod.ConsultationRecord(
                    stakeholder_id="X",
                    date="2025-06-15",
                    method=mod.EngagementMethod.SURVEY,
                    topics_discussed=["e1_climate"],
                    key_findings=["Finding"],
                ),
            ])

    def test_synthesize_findings_deterministic_scores(self, engine, mod):
        """Same inputs produce same scores across calls."""
        stakeholders = [
            mod.Stakeholder(
                id="DET-SH-1", name="Det Test",
                category=mod.StakeholderCategory.EMPLOYEES,
                influence_level=3, impact_level=3,
            ),
        ]
        consultations = [
            mod.ConsultationRecord(
                stakeholder_id="DET-SH-1",
                date="2025-06-15",
                method=mod.EngagementMethod.SURVEY,
                topics_discussed=["e1_climate"],
                key_findings=["Finding"],
                attendance_count=50,
            ),
        ]
        r1 = engine.synthesize_findings(stakeholders, consultations)
        r2 = engine.synthesize_findings(stakeholders, consultations)
        assert r1.engagement_quality_score == r2.engagement_quality_score

    def test_synthesize_findings_has_coverage(self, engine, mod):
        """Result includes coverage percentage."""
        stakeholders = [
            mod.Stakeholder(
                id="COV-SH", name="Coverage Test",
                category=mod.StakeholderCategory.EMPLOYEES,
                influence_level=3, impact_level=3,
            ),
        ]
        consultations = [
            mod.ConsultationRecord(
                stakeholder_id="COV-SH",
                date="2025-06-15",
                method=mod.EngagementMethod.SURVEY,
                topics_discussed=["e1_climate"],
                key_findings=["Finding"],
                attendance_count=10,
            ),
        ]
        result = engine.synthesize_findings(stakeholders, consultations)
        assert hasattr(result, "coverage_pct")
        assert float(result.coverage_pct) > 0

    def test_synthesize_findings_has_topic_frequency(self, engine, mod):
        """Result includes topic frequency data."""
        stakeholders = [
            mod.Stakeholder(
                id="TF-SH", name="Topic Test",
                category=mod.StakeholderCategory.EMPLOYEES,
                influence_level=3, impact_level=3,
            ),
        ]
        consultations = [
            mod.ConsultationRecord(
                stakeholder_id="TF-SH",
                date="2025-06-15",
                method=mod.EngagementMethod.WORKSHOP,
                topics_discussed=["e1_climate", "s1_own_workforce"],
                key_findings=["Finding"],
                attendance_count=10,
            ),
        ]
        result = engine.synthesize_findings(stakeholders, consultations)
        assert hasattr(result, "topic_frequency")
        assert "e1_climate" in result.topic_frequency


# ===========================================================================
# Provenance Hash Tests
# ===========================================================================


class TestStakeholderProvenanceHash:
    """Tests for stakeholder engagement provenance hashing."""

    def test_hash_is_64_chars(self, engine, mod):
        """Provenance hash is a 64-character SHA-256 hex string."""
        stakeholders = [
            mod.Stakeholder(
                id="PH-1", name="Hash Test",
                category=mod.StakeholderCategory.EMPLOYEES,
                influence_level=3, impact_level=3,
            ),
        ]
        consultations = [
            mod.ConsultationRecord(
                stakeholder_id="PH-1",
                date="2025-06-15",
                method=mod.EngagementMethod.SURVEY,
                topics_discussed=["e1_climate"],
                key_findings=["Finding"],
                attendance_count=50,
            ),
        ]
        result = engine.synthesize_findings(stakeholders, consultations)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)
