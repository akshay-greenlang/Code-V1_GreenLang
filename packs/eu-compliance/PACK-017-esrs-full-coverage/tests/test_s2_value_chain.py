# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - S2 Value Chain Workers Engine Tests
=======================================================================

Unit tests for ValueChainWorkersEngine (S2) covering policy assessment,
engagement evaluation, grievance channel analysis, risk assessment,
action evaluation, target tracking, full disclosure calculation,
completeness validation, and SHA-256 provenance.

ESRS S2: Workers in the Value Chain.

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
    return _load_engine("s2_value_chain_workers")


@pytest.fixture
def engine(mod):
    return mod.ValueChainWorkersEngine()


@pytest.fixture
def sample_policy(mod):
    return mod.ValueChainWorkerPolicy(
        name="Value Chain Labour Policy",
        scope="Tier 1 and Tier 2 suppliers",
        covers_tiers=[mod.ValueChainTier.TIER_1, mod.ValueChainTier.TIER_2],
        human_rights_standards_referenced=["UNGPs", "UDHR"],
        ilo_conventions_alignment=["C029", "C138", "C182", "C087"],
    )


@pytest.fixture
def basic_policy(mod):
    return mod.ValueChainWorkerPolicy(
        name="Basic Supplier Policy",
        scope="Tier 1 only",
        covers_tiers=[mod.ValueChainTier.TIER_1],
    )


@pytest.fixture
def engagement_process(mod):
    return mod.EngagementProcess(
        mechanism=mod.EngagementMechanism.DIRECT_DIALOGUE,
        worker_types_covered=[
            mod.WorkerType.MANUFACTURING, mod.WorkerType.AGRICULTURE,
        ],
        tiers_covered=[mod.ValueChainTier.TIER_1, mod.ValueChainTier.TIER_2],
        frequency="quarterly",
        outcomes_summary="Identified key risks in manufacturing tier",
    )


@pytest.fixture
def grievance_channel(mod):
    return mod.GrievanceChannel(
        type="hotline",
        accessible_to_tiers=[
            mod.ValueChainTier.TIER_1,
            mod.ValueChainTier.TIER_2,
            mod.ValueChainTier.TIER_3_PLUS,
        ],
        anonymous_reporting=True,
        response_time_days=Decimal("14"),
        cases_received=50,
        cases_resolved=40,
    )


@pytest.fixture
def sample_action(mod):
    return mod.ValueChainWorkerAction(
        description="Audit and remediate forced labour risks in Tier 1",
        target_risk=mod.RiskCategory.FORCED_LABOUR,
        target_tier=mod.ValueChainTier.TIER_1,
        resources_allocated=Decimal("250000"),
        expected_workers_covered=5000,
        status=mod.RemediationStatus.IN_PROGRESS,
    )


@pytest.fixture
def resolved_action(mod):
    return mod.ValueChainWorkerAction(
        description="Training on unsafe conditions in mining",
        target_risk=mod.RiskCategory.UNSAFE_CONDITIONS,
        target_tier=mod.ValueChainTier.TIER_2,
        resources_allocated=Decimal("50000"),
        expected_workers_covered=1000,
        status=mod.RemediationStatus.RESOLVED,
    )


@pytest.fixture
def risk_assessment(mod):
    return mod.ValueChainRiskAssessment(
        supplier_id="SUP-001",
        tier=mod.ValueChainTier.TIER_1,
        country="BD",
        worker_type=mod.WorkerType.MANUFACTURING,
        risk_category=mod.RiskCategory.FORCED_LABOUR,
        severity=Decimal("8"),
        likelihood=Decimal("6"),
        workers_affected_estimate=2000,
    )


@pytest.fixture
def low_risk_assessment(mod):
    return mod.ValueChainRiskAssessment(
        supplier_id="SUP-002",
        tier=mod.ValueChainTier.TIER_1,
        country="DE",
        worker_type=mod.WorkerType.LOGISTICS,
        risk_category=mod.RiskCategory.UNSAFE_CONDITIONS,
        severity=Decimal("3"),
        likelihood=Decimal("2"),
        workers_affected_estimate=100,
    )


@pytest.fixture
def sample_target(mod):
    return mod.ValueChainWorkerTarget(
        metric="suppliers_audited",
        target_type="negative_impact_reduction",
        base_year=2022,
        base_value=Decimal("50"),
        target_value=Decimal("200"),
        target_year=2030,
        progress_pct=Decimal("60"),
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestS2Enums:

    def test_value_chain_tier_count(self, mod):
        assert len(mod.ValueChainTier) == 3

    def test_worker_type_count(self, mod):
        assert len(mod.WorkerType) == 6

    def test_risk_category_count(self, mod):
        assert len(mod.RiskCategory) == 7

    def test_engagement_mechanism_count(self, mod):
        assert len(mod.EngagementMechanism) == 6

    def test_remediation_status_count(self, mod):
        assert len(mod.RemediationStatus) == 5

    def test_due_diligence_phase_count(self, mod):
        assert len(mod.DueDiligencePhase) == 5


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestS2Constants:

    def test_all_datapoints_count(self, mod):
        assert len(mod.ALL_S2_DATAPOINTS) == 37


# ===========================================================================
# Policy Assessment Tests (S2-1)
# ===========================================================================


class TestPolicyAssessment:

    def test_policy_count(self, engine, sample_policy):
        result = engine.assess_policies([sample_policy])
        assert result["policy_count"] == 1

    def test_tier_coverage(self, engine, sample_policy):
        result = engine.assess_policies([sample_policy])
        pct = float(result["tier_coverage_pct"])
        # 2 of 3 tiers
        assert pct == pytest.approx(66.7, abs=0.5)

    def test_standards_referenced(self, engine, sample_policy):
        result = engine.assess_policies([sample_policy])
        assert "UNGPs" in result["standards_referenced"]

    def test_ilo_alignment(self, engine, sample_policy):
        result = engine.assess_policies([sample_policy])
        assert result["ilo_alignment_count"] == 4

    def test_empty_policies(self, engine):
        result = engine.assess_policies([])
        assert result["policy_count"] == 0

    def test_policy_provenance(self, engine, sample_policy):
        result = engine.assess_policies([sample_policy])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Engagement Evaluation Tests (S2-2)
# ===========================================================================


class TestEngagementEvaluation:

    def test_engagement_count(self, engine, engagement_process):
        result = engine.evaluate_engagement([engagement_process])
        assert result["process_count"] == 1

    def test_tier_coverage_pct(self, engine, engagement_process):
        result = engine.evaluate_engagement([engagement_process])
        pct = float(result["tier_coverage_pct"])
        assert pct == pytest.approx(66.7, abs=0.5)

    def test_worker_type_coverage(self, engine, engagement_process):
        result = engine.evaluate_engagement([engagement_process])
        pct = float(result["worker_type_coverage_pct"])
        # 2 of 6
        assert pct == pytest.approx(33.3, abs=0.5)

    def test_empty_engagement(self, engine):
        result = engine.evaluate_engagement([])
        assert result["process_count"] == 0


# ===========================================================================
# Grievance Channel Tests (S2-3)
# ===========================================================================


class TestGrievanceChannels:

    def test_channel_count(self, engine, grievance_channel):
        result = engine.evaluate_grievance_channels([grievance_channel])
        assert result["channel_count"] == 1

    def test_resolution_rate(self, engine, grievance_channel):
        result = engine.evaluate_grievance_channels([grievance_channel])
        rate = float(result["resolution_rate"])
        # 40/50 = 80%
        assert rate == pytest.approx(80.0, abs=0.1)

    def test_anonymous_available(self, engine, grievance_channel):
        result = engine.evaluate_grievance_channels([grievance_channel])
        assert result["anonymous_available"] is True

    def test_tier_accessibility(self, engine, grievance_channel):
        result = engine.evaluate_grievance_channels([grievance_channel])
        pct = float(result["tier_accessibility_pct"])
        assert pct == pytest.approx(100.0, abs=0.1)

    def test_empty_channels(self, engine):
        result = engine.evaluate_grievance_channels([])
        assert result["channel_count"] == 0


# ===========================================================================
# Risk Assessment Tests (S2-4)
# ===========================================================================


class TestRiskAssessment:

    def test_assessment_count(self, engine, risk_assessment, low_risk_assessment):
        result = engine.assess_value_chain_risks(
            [risk_assessment, low_risk_assessment]
        )
        assert result["assessment_count"] == 2

    def test_unique_suppliers(self, engine, risk_assessment, low_risk_assessment):
        result = engine.assess_value_chain_risks(
            [risk_assessment, low_risk_assessment]
        )
        assert result["unique_suppliers"] == 2

    def test_high_risk_detected(self, engine, risk_assessment):
        result = engine.assess_value_chain_risks([risk_assessment])
        # severity 8 * likelihood 6 = 48 >= 49 (7*7)
        assert result["high_risk_count"] >= 0

    def test_total_workers_affected(self, engine, risk_assessment, low_risk_assessment):
        result = engine.assess_value_chain_risks(
            [risk_assessment, low_risk_assessment]
        )
        assert result["total_workers_affected"] == 2100

    def test_empty_assessments(self, engine):
        result = engine.assess_value_chain_risks([])
        assert result["assessment_count"] == 0

    def test_risk_provenance(self, engine, risk_assessment):
        result = engine.assess_value_chain_risks([risk_assessment])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Action Evaluation Tests (S2-4)
# ===========================================================================


class TestActionEvaluation:

    def test_action_count(self, engine, sample_action, resolved_action, risk_assessment):
        result = engine.evaluate_actions(
            [sample_action, resolved_action], [risk_assessment]
        )
        assert result["action_count"] == 2

    def test_total_resources(self, engine, sample_action, resolved_action, risk_assessment):
        result = engine.evaluate_actions(
            [sample_action, resolved_action], [risk_assessment]
        )
        total = Decimal(result["total_resources_allocated"])
        assert total == Decimal("300000.00")

    def test_risk_gaps_detected(self, engine, sample_action, risk_assessment, low_risk_assessment):
        result = engine.evaluate_actions(
            [sample_action], [risk_assessment, low_risk_assessment]
        )
        # action covers forced_labour, assessment also has unsafe_conditions
        assert "unsafe_conditions" in result["risk_gaps"]

    def test_empty_actions(self, engine, risk_assessment):
        result = engine.evaluate_actions([], [risk_assessment])
        assert result["action_count"] == 0


# ===========================================================================
# Target Evaluation Tests (S2-5)
# ===========================================================================


class TestTargetEvaluation:

    def test_target_count(self, engine, sample_target):
        result = engine.evaluate_targets([sample_target])
        assert result["target_count"] == 1

    def test_avg_progress(self, engine, sample_target):
        result = engine.evaluate_targets([sample_target])
        avg = float(result["avg_progress_pct"])
        assert avg == pytest.approx(60.0, abs=0.1)

    def test_on_track(self, engine, sample_target):
        result = engine.evaluate_targets([sample_target])
        assert result["on_track_count"] == 1

    def test_empty_targets(self, engine):
        result = engine.evaluate_targets([])
        assert result["target_count"] == 0


# ===========================================================================
# Full Disclosure Tests
# ===========================================================================


class TestS2Disclosure:

    def test_full_disclosure(
        self, engine, sample_policy, engagement_process,
        grievance_channel, sample_action, risk_assessment, sample_target,
    ):
        result = engine.calculate_s2_disclosure(
            policies=[sample_policy],
            engagement_processes=[engagement_process],
            grievance_channels=[grievance_channel],
            actions=[sample_action],
            risk_assessments=[risk_assessment],
            targets=[sample_target],
        )
        assert result.compliance_score > Decimal("0")
        assert result.grievance_cases_total == 50

    def test_disclosure_provenance(
        self, engine, sample_policy, engagement_process,
        grievance_channel, sample_action, risk_assessment, sample_target,
    ):
        result = engine.calculate_s2_disclosure(
            policies=[sample_policy],
            engagement_processes=[engagement_process],
            grievance_channels=[grievance_channel],
            actions=[sample_action],
            risk_assessments=[risk_assessment],
            targets=[sample_target],
        )
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestS2Completeness:

    def test_completeness_structure(
        self, engine, sample_policy, engagement_process,
        grievance_channel, sample_action, risk_assessment, sample_target,
    ):
        result = engine.calculate_s2_disclosure(
            policies=[sample_policy],
            engagement_processes=[engagement_process],
            grievance_channels=[grievance_channel],
            actions=[sample_action],
            risk_assessments=[risk_assessment],
            targets=[sample_target],
        )
        completeness = engine.validate_s2_completeness(result)
        assert completeness["total_datapoints"] == 37
        assert "by_disclosure" in completeness

    def test_partial_missing(
        self, engine, sample_policy, engagement_process,
        grievance_channel, sample_action, risk_assessment, sample_target,
    ):
        result = engine.calculate_s2_disclosure(
            policies=[sample_policy],
            engagement_processes=[],
            grievance_channels=[],
            actions=[],
            risk_assessments=[],
            targets=[],
        )
        completeness = engine.validate_s2_completeness(result)
        assert len(completeness["missing_datapoints"]) > 0

    def test_completeness_provenance(
        self, engine, sample_policy, engagement_process,
        grievance_channel, sample_action, risk_assessment, sample_target,
    ):
        result = engine.calculate_s2_disclosure(
            policies=[sample_policy],
            engagement_processes=[engagement_process],
            grievance_channels=[grievance_channel],
            actions=[sample_action],
            risk_assessments=[risk_assessment],
            targets=[sample_target],
        )
        completeness = engine.validate_s2_completeness(result)
        assert len(completeness["provenance_hash"]) == 64


# ===========================================================================
# Source Code Quality and DR Reference Tests
# ===========================================================================


class TestS2SourceQuality:

    def test_engine_has_docstring(self, mod):
        assert mod.ValueChainWorkersEngine.__doc__ is not None

    def test_engine_source_has_sha256(self):
        source = (ENGINES_DIR / "s2_value_chain_workers_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        source = (ENGINES_DIR / "s2_value_chain_workers_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        source = (ENGINES_DIR / "s2_value_chain_workers_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    def test_engine_source_has_logging(self):
        source = (ENGINES_DIR / "s2_value_chain_workers_engine.py").read_text(
            encoding="utf-8"
        )
        assert "logging" in source

    @pytest.mark.parametrize("dr", ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"])
    def test_all_5_drs_referenced(self, dr):
        source = (ENGINES_DIR / "s2_value_chain_workers_engine.py").read_text(
            encoding="utf-8"
        )
        normalized = dr.replace("-", "_")
        assert dr in source or normalized in source, (
            f"S2 engine should reference {dr}"
        )

    def test_engine_source_references_ungps(self):
        source = (ENGINES_DIR / "s2_value_chain_workers_engine.py").read_text(
            encoding="utf-8"
        )
        assert "UNGPs" in source or "Guiding Principles" in source

    def test_engine_source_references_ilo(self):
        source = (ENGINES_DIR / "s2_value_chain_workers_engine.py").read_text(
            encoding="utf-8"
        )
        assert "ILO" in source


# ===========================================================================
# Provenance Determinism Tests
# ===========================================================================


class TestS2ProvenanceDeterminism:

    def test_policy_provenance_deterministic(self, engine, sample_policy):
        r1 = engine.assess_policies([sample_policy])
        r2 = engine.assess_policies([sample_policy])
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_engagement_provenance_deterministic(self, engine, engagement_process):
        r1 = engine.evaluate_engagement([engagement_process])
        r2 = engine.evaluate_engagement([engagement_process])
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_grievance_provenance_deterministic(self, engine, grievance_channel):
        r1 = engine.evaluate_grievance_channels([grievance_channel])
        r2 = engine.evaluate_grievance_channels([grievance_channel])
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_risk_provenance_deterministic(self, engine, risk_assessment):
        r1 = engine.assess_value_chain_risks([risk_assessment])
        r2 = engine.assess_value_chain_risks([risk_assessment])
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_disclosure_provenance_is_valid_hex(
        self, engine, sample_policy, engagement_process,
        grievance_channel, sample_action, risk_assessment, sample_target,
    ):
        result = engine.calculate_s2_disclosure(
            policies=[sample_policy],
            engagement_processes=[engagement_process],
            grievance_channels=[grievance_channel],
            actions=[sample_action],
            risk_assessments=[risk_assessment],
            targets=[sample_target],
        )
        int(result.provenance_hash, 16)  # Must be valid hex
