# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - S3 Affected Communities Engine Tests
========================================================================

Unit tests for AffectedCommunitiesEngine (S3) covering policy assessment,
engagement evaluation, grievance resolution metrics, action and impact
assessment, target tracking, full disclosure calculation, completeness
validation, and SHA-256 provenance.

ESRS S3: Affected Communities.

Target: 50+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine, ENGINES_DIR


@pytest.fixture(scope="module")
def mod():
    return _load_engine("s3_affected_communities")


@pytest.fixture
def engine(mod):
    return mod.AffectedCommunitiesEngine()


@pytest.fixture
def comprehensive_policy(mod):
    return mod.CommunityPolicy(
        name="Human Rights and Community Policy",
        scope="Group-wide including all operational sites",
        rights_frameworks_referenced=[
            mod.RightsFramework.UN_GUIDING_PRINCIPLES,
            mod.RightsFramework.UN_DRIP,
            mod.RightsFramework.ILO_169,
        ],
        fpic_commitment=True,
        indigenous_peoples_specific=True,
    )


@pytest.fixture
def basic_policy(mod):
    return mod.CommunityPolicy(
        name="Community Relations Policy",
        scope="Domestic operations only",
        rights_frameworks_referenced=[mod.RightsFramework.UN_GUIDING_PRINCIPLES],
        fpic_commitment=False,
        indigenous_peoples_specific=False,
    )


@pytest.fixture
def indigenous_engagement(mod):
    return mod.CommunityEngagement(
        community_type=mod.CommunityType.INDIGENOUS_PEOPLES,
        location="Amazon Basin, Brazil",
        engagement_level=mod.EngagementLevel.EMPOWER,
        consent_type=mod.ConsentType.FPIC,
        participants_count=120,
        frequency="quarterly",
        outcomes="Land use agreement established",
    )


@pytest.fixture
def local_engagement(mod):
    return mod.CommunityEngagement(
        community_type=mod.CommunityType.LOCAL_COMMUNITIES,
        location="Rhine Valley, Germany",
        engagement_level=mod.EngagementLevel.CONSULT,
        consent_type=mod.ConsentType.CONSULTATION,
        participants_count=80,
        frequency="annual",
        outcomes="Noise mitigation plan agreed",
    )


@pytest.fixture
def resolved_grievance(mod):
    return mod.CommunityGrievance(
        community_type=mod.CommunityType.LOCAL_COMMUNITIES,
        issue_area=mod.ImpactArea.LIVELIHOODS,
        severity=mod.SeverityLevel.MEDIUM,
        date_raised="2025-01-15",
        status=mod.GrievanceStatus.RESOLVED,
        time_to_resolution_days=30,
    )


@pytest.fixture
def open_grievance(mod):
    return mod.CommunityGrievance(
        community_type=mod.CommunityType.INDIGENOUS_PEOPLES,
        issue_area=mod.ImpactArea.LAND_RIGHTS,
        severity=mod.SeverityLevel.HIGH,
        date_raised="2025-03-01",
        status=mod.GrievanceStatus.OPEN,
    )


@pytest.fixture
def sample_action(mod):
    return mod.CommunityAction(
        description="Remediate water contamination near mining site",
        impact_area=mod.ImpactArea.WATER_ACCESS,
        communities_affected=[
            mod.CommunityType.LOCAL_COMMUNITIES,
            mod.CommunityType.INDIGENOUS_PEOPLES,
        ],
        resources_allocated=Decimal("500000"),
        status=mod.ActionStatus.IN_PROGRESS,
    )


@pytest.fixture
def sample_impact(mod):
    return mod.CommunityImpactAssessment(
        site_id="SITE-001",
        community_type=mod.CommunityType.LOCAL_COMMUNITIES,
        impact_area=mod.ImpactArea.LIVELIHOODS,
        severity=mod.SeverityLevel.MEDIUM,
        likelihood="medium",
        people_affected_estimate=500,
        mitigation_measures=["Employment programme", "Skills training"],
    )


@pytest.fixture
def sample_target(mod):
    return mod.CommunityTarget(
        metric="grievance_resolution_rate",
        target_type=mod.TargetType.ABSOLUTE,
        base_year=2023,
        base_value=Decimal("60"),
        target_value=Decimal("90"),
        target_year=2030,
        progress_pct=Decimal("55"),
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestS3Enums:

    def test_community_type_count(self, mod):
        assert len(mod.CommunityType) == 6

    def test_impact_area_count(self, mod):
        assert len(mod.ImpactArea) == 9

    def test_consent_type_count(self, mod):
        assert len(mod.ConsentType) == 4

    def test_engagement_level_count(self, mod):
        assert len(mod.EngagementLevel) == 5

    def test_rights_framework_count(self, mod):
        assert len(mod.RightsFramework) == 5

    def test_grievance_status_count(self, mod):
        assert len(mod.GrievanceStatus) == 6

    def test_action_status_count(self, mod):
        assert len(mod.ActionStatus) == 5

    def test_severity_level_count(self, mod):
        assert len(mod.SeverityLevel) == 4

    def test_target_type_count(self, mod):
        assert len(mod.TargetType) == 2


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestS3Constants:

    def test_all_datapoints_count(self, mod):
        assert len(mod.ALL_S3_DATAPOINTS) == 38

    def test_s3_1_datapoints_count(self, mod):
        assert len(mod.S3_1_DATAPOINTS) == 8

    def test_s3_2_datapoints_count(self, mod):
        assert len(mod.S3_2_DATAPOINTS) == 8


# ===========================================================================
# Policy Assessment Tests (S3-1)
# ===========================================================================


class TestPolicyAssessment:

    def test_policy_count(self, engine, comprehensive_policy):
        result = engine.assess_policy_coverage([comprehensive_policy])
        assert result["policy_count"] == 1

    def test_fpic_committed(self, engine, comprehensive_policy):
        result = engine.assess_policy_coverage([comprehensive_policy])
        assert result["fpic_committed"] is True

    def test_indigenous_specific(self, engine, comprehensive_policy):
        result = engine.assess_policy_coverage([comprehensive_policy])
        assert result["indigenous_specific"] is True

    def test_frameworks_covered(self, engine, comprehensive_policy):
        result = engine.assess_policy_coverage([comprehensive_policy])
        assert len(result["frameworks_covered"]) == 3

    def test_coverage_score_positive(self, engine, comprehensive_policy):
        result = engine.assess_policy_coverage([comprehensive_policy])
        assert result["coverage_score"] > Decimal("0")

    def test_empty_policies(self, engine):
        result = engine.assess_policy_coverage([])
        assert result["policy_count"] == 0

    def test_basic_policy_no_fpic(self, engine, basic_policy):
        result = engine.assess_policy_coverage([basic_policy])
        assert result["fpic_committed"] is False
        assert result["indigenous_specific"] is False

    def test_policy_provenance(self, engine, comprehensive_policy):
        result = engine.assess_policy_coverage([comprehensive_policy])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Engagement Quality Tests (S3-2)
# ===========================================================================


class TestEngagementQuality:

    def test_engagement_count(self, engine, indigenous_engagement, local_engagement):
        result = engine.assess_engagement_quality(
            [indigenous_engagement, local_engagement]
        )
        assert result["engagement_count"] == 2

    def test_community_types_reached(self, engine, indigenous_engagement, local_engagement):
        result = engine.assess_engagement_quality(
            [indigenous_engagement, local_engagement]
        )
        assert len(result["community_types_reached"]) == 2

    def test_total_participants(self, engine, indigenous_engagement, local_engagement):
        result = engine.assess_engagement_quality(
            [indigenous_engagement, local_engagement]
        )
        assert result["total_participants"] == 200

    def test_fpic_count(self, engine, indigenous_engagement):
        result = engine.assess_engagement_quality([indigenous_engagement])
        assert result["fpic_count"] >= 1

    def test_engagement_score_positive(self, engine, indigenous_engagement):
        result = engine.assess_engagement_quality([indigenous_engagement])
        assert result["engagement_score"] > Decimal("0")

    def test_empty_engagements(self, engine):
        result = engine.assess_engagement_quality([])
        assert result["engagement_count"] == 0

    def test_engagement_provenance(self, engine, indigenous_engagement):
        result = engine.assess_engagement_quality([indigenous_engagement])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Grievance Mechanism Tests (S3-3)
# ===========================================================================


class TestGrievanceMechanisms:

    def test_total_cases(self, engine, resolved_grievance, open_grievance):
        result = engine.assess_grievance_mechanisms(
            [resolved_grievance, open_grievance]
        )
        assert result["total_cases"] == 2

    def test_resolved_count(self, engine, resolved_grievance, open_grievance):
        result = engine.assess_grievance_mechanisms(
            [resolved_grievance, open_grievance]
        )
        assert result["resolved_count"] == 1

    def test_resolution_rate(self, engine, resolved_grievance, open_grievance):
        result = engine.assess_grievance_mechanisms(
            [resolved_grievance, open_grievance]
        )
        rate = float(result["resolution_rate"])
        assert rate == pytest.approx(50.0, abs=1.0)

    def test_empty_grievances(self, engine):
        result = engine.assess_grievance_mechanisms([])
        assert result["total_cases"] == 0

    def test_grievance_provenance(self, engine, resolved_grievance):
        result = engine.assess_grievance_mechanisms([resolved_grievance])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Actions and Impacts Tests (S3-4)
# ===========================================================================


class TestActionsAndImpacts:

    def test_actions_counted(self, engine, sample_action, sample_impact):
        result = engine.assess_actions_and_impacts(
            [sample_action], [sample_impact]
        )
        assert result["action_count"] >= 1

    def test_impact_assessments_counted(self, engine, sample_action, sample_impact):
        result = engine.assess_actions_and_impacts(
            [sample_action], [sample_impact]
        )
        assert result["assessment_count"] >= 1

    def test_resources_allocated(self, engine, sample_action, sample_impact):
        result = engine.assess_actions_and_impacts(
            [sample_action], [sample_impact]
        )
        total = Decimal(str(result["total_resources_eur"]))
        assert total == Decimal("500000")

    def test_empty_actions(self, engine):
        result = engine.assess_actions_and_impacts([], [])
        assert result["action_count"] == 0

    def test_actions_provenance(self, engine, sample_action, sample_impact):
        result = engine.assess_actions_and_impacts(
            [sample_action], [sample_impact]
        )
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Target Assessment Tests (S3-5)
# ===========================================================================


class TestTargetAssessment:

    def test_target_count(self, engine, sample_target):
        result = engine.assess_targets([sample_target])
        assert result["target_count"] == 1

    def test_avg_progress(self, engine, sample_target):
        result = engine.assess_targets([sample_target])
        avg = float(result["avg_progress_pct"])
        assert avg == pytest.approx(55.0, abs=0.5)

    def test_empty_targets(self, engine):
        result = engine.assess_targets([])
        assert result["target_count"] == 0

    def test_target_provenance(self, engine, sample_target):
        result = engine.assess_targets([sample_target])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Full Disclosure Tests
# ===========================================================================


class TestS3Disclosure:

    def test_full_disclosure(
        self, engine, comprehensive_policy, indigenous_engagement,
        resolved_grievance, sample_action, sample_impact, sample_target,
    ):
        result = engine.calculate_s3_disclosure(
            policies=[comprehensive_policy],
            engagements=[indigenous_engagement],
            grievances=[resolved_grievance],
            actions=[sample_action],
            impact_assessments=[sample_impact],
            targets=[sample_target],
        )
        assert result.compliance_score > Decimal("0")
        assert result.grievance_cases_total == 1

    def test_disclosure_provenance(
        self, engine, comprehensive_policy, indigenous_engagement,
        resolved_grievance, sample_action, sample_impact, sample_target,
    ):
        result = engine.calculate_s3_disclosure(
            policies=[comprehensive_policy],
            engagements=[indigenous_engagement],
            grievances=[resolved_grievance],
            actions=[sample_action],
            impact_assessments=[sample_impact],
            targets=[sample_target],
        )
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_disclosure_communities_engaged(
        self, engine, comprehensive_policy, indigenous_engagement,
        local_engagement, resolved_grievance, sample_action,
        sample_impact, sample_target,
    ):
        result = engine.calculate_s3_disclosure(
            policies=[comprehensive_policy],
            engagements=[indigenous_engagement, local_engagement],
            grievances=[resolved_grievance],
            actions=[sample_action],
            impact_assessments=[sample_impact],
            targets=[sample_target],
        )
        assert result.communities_engaged_count == 2


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestS3Completeness:

    def test_completeness_structure(
        self, engine, comprehensive_policy, indigenous_engagement,
        resolved_grievance, sample_action, sample_impact, sample_target,
    ):
        result = engine.calculate_s3_disclosure(
            policies=[comprehensive_policy],
            engagements=[indigenous_engagement],
            grievances=[resolved_grievance],
            actions=[sample_action],
            impact_assessments=[sample_impact],
            targets=[sample_target],
        )
        completeness = engine.validate_s3_completeness(result)
        assert completeness["total_datapoints"] == 38
        assert "by_disclosure" in completeness

    def test_partial_missing(
        self, engine, comprehensive_policy, indigenous_engagement,
        resolved_grievance, sample_action, sample_impact, sample_target,
    ):
        result = engine.calculate_s3_disclosure(
            policies=[comprehensive_policy],
            engagements=[],
            grievances=[],
            actions=[],
            impact_assessments=[],
            targets=[],
        )
        completeness = engine.validate_s3_completeness(result)
        assert len(completeness["missing_datapoints"]) > 0

    def test_completeness_provenance(
        self, engine, comprehensive_policy, indigenous_engagement,
        resolved_grievance, sample_action, sample_impact, sample_target,
    ):
        result = engine.calculate_s3_disclosure(
            policies=[comprehensive_policy],
            engagements=[indigenous_engagement],
            grievances=[resolved_grievance],
            actions=[sample_action],
            impact_assessments=[sample_impact],
            targets=[sample_target],
        )
        completeness = engine.validate_s3_completeness(result)
        assert len(completeness["provenance_hash"]) == 64


# ===========================================================================
# Source Code Quality Tests
# ===========================================================================


class TestS3SourceQuality:

    def test_engine_has_docstring(self, mod):
        assert mod.AffectedCommunitiesEngine.__doc__ is not None

    def test_engine_source_has_sha256(self):
        source = (ENGINES_DIR / "s3_affected_communities_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        source = (ENGINES_DIR / "s3_affected_communities_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        source = (ENGINES_DIR / "s3_affected_communities_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    @pytest.mark.parametrize("dr", ["S3-1", "S3-2", "S3-3", "S3-4", "S3-5"])
    def test_all_5_drs_referenced(self, dr):
        source = (ENGINES_DIR / "s3_affected_communities_engine.py").read_text(
            encoding="utf-8"
        )
        normalized = dr.replace("-", "_")
        assert dr in source or normalized in source, (
            f"S3 engine should reference {dr}"
        )
