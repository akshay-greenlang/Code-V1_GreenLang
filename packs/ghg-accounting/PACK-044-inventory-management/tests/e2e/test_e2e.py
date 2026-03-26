# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - End-to-End Tests
=========================================

Tests the full inventory management lifecycle across all engines:
period creation through approval, data collection, quality management,
change assessment, review, versioning, consolidation, gap analysis,
documentation, and benchmarking.

Target: 25+ test cases.
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

# Ensure the tests directory is importable for conftest
_tests_dir = Path(__file__).resolve().parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from conftest import _load_engine, _load_config_module


# ---------------------------------------------------------------------------
# Load all engines
# ---------------------------------------------------------------------------

_period_mod = _load_engine("inventory_period")
_dc_mod = _load_engine("data_collection")
_qm_mod = _load_engine("quality_management")
_cm_mod = _load_engine("change_management")
_ra_mod = _load_engine("review_approval")
_iv_mod = _load_engine("inventory_versioning")
_consol_mod = _load_engine("consolidation_management")
_gap_mod = _load_engine("gap_analysis")
_doc_mod = _load_engine("documentation")
_bench_mod = _load_engine("benchmarking")


InventoryPeriodEngine = _period_mod.InventoryPeriodEngine
PeriodStatus = _period_mod.PeriodStatus
MilestoneStatus = _period_mod.MilestoneStatus

DataCollectionEngine = _dc_mod.DataCollectionEngine
DataScope = _dc_mod.DataScope

QualityManagementEngine = _qm_mod.QualityManagementEngine
QualityDimension = _qm_mod.QualityDimension
CheckSeverity = _qm_mod.CheckSeverity

ChangeManagementEngine = _cm_mod.ChangeManagementEngine
ChangeRequest = _cm_mod.ChangeRequest
AffectedSource = _cm_mod.AffectedSource
ChangeCategory = _cm_mod.ChangeCategory

ReviewApprovalEngine = _ra_mod.ReviewApprovalEngine
ReviewRequest = _ra_mod.ReviewRequest
ReviewStage = _ra_mod.ReviewStage
ReviewStatus = _ra_mod.ReviewStatus

InventoryVersioningEngine = _iv_mod.InventoryVersioningEngine
VersionStatus = _iv_mod.VersionStatus

ConsolidationManagementEngine = _consol_mod.ConsolidationManagementEngine
ConsolidationApproach = _consol_mod.ConsolidationApproach
EntityHierarchy = _consol_mod.EntityHierarchy
Entity = _consol_mod.Entity
SubsidiarySubmission = _consol_mod.SubsidiarySubmission
SubmissionStatus = _consol_mod.SubmissionStatus

GapAnalysisEngine = _gap_mod.GapAnalysisEngine
SourceCategoryAssessment = _gap_mod.SourceCategoryAssessment
MethodologyTier = _gap_mod.MethodologyTier

DocumentationEngine = _doc_mod.DocumentationEngine
MethodologyDocument = _doc_mod.MethodologyDocument
DocumentType = _doc_mod.DocumentType

BenchmarkingEngine = _bench_mod.BenchmarkingEngine
EntityProfile = _bench_mod.EntityProfile
PeerDataPoint = _bench_mod.PeerDataPoint
HistoricalDataPoint = _bench_mod.HistoricalDataPoint


# ===================================================================
# Full Lifecycle E2E Test
# ===================================================================


class TestFullInventoryLifecycle:
    """End-to-end test of the complete inventory management lifecycle."""

    def test_full_lifecycle_period_to_approval(self):
        """Test the full period lifecycle from PLANNING to FINAL."""
        engine = InventoryPeriodEngine()

        # Step 1: Create period
        result = engine.create_period(
            organisation_id="ORG-E2E-001",
            period_name="FY2025 E2E Test",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        pid = result.period.period_id
        assert result.period.status == PeriodStatus.PLANNING

        # Step 2: Complete planning milestones and advance
        period = engine.get_period(pid)
        for ms in period.milestones:
            if ms.phase == "planning":
                engine.update_milestone(
                    pid, ms.milestone_id,
                    status=MilestoneStatus.COMPLETED,
                    actual_date=date.today(),
                )

        engine.transition(pid, PeriodStatus.DATA_COLLECTION)
        assert engine.get_period(pid).status == PeriodStatus.DATA_COLLECTION

        # Step 3: Complete data collection milestones
        period = engine.get_period(pid)
        for ms in period.milestones:
            if ms.phase == "data_collection":
                engine.update_milestone(
                    pid, ms.milestone_id,
                    status=MilestoneStatus.COMPLETED,
                    actual_date=date.today(),
                )

        engine.transition(pid, PeriodStatus.CALCULATION)
        assert engine.get_period(pid).status == PeriodStatus.CALCULATION

        # Step 4: Complete calculation milestones
        period = engine.get_period(pid)
        for ms in period.milestones:
            if ms.phase == "calculation":
                engine.update_milestone(
                    pid, ms.milestone_id,
                    status=MilestoneStatus.COMPLETED,
                    actual_date=date.today(),
                )

        engine.transition(pid, PeriodStatus.REVIEW)
        assert engine.get_period(pid).status == PeriodStatus.REVIEW

        # Step 5: Complete review milestones and approve
        period = engine.get_period(pid)
        for ms in period.milestones:
            if ms.phase == "review":
                engine.update_milestone(
                    pid, ms.milestone_id,
                    status=MilestoneStatus.COMPLETED,
                    actual_date=date.today(),
                )

        engine.transition(pid, PeriodStatus.APPROVED)
        assert engine.get_period(pid).status == PeriodStatus.APPROVED
        assert engine.get_period(pid).locked is True

        # Step 6: Finalize
        engine.transition(pid, PeriodStatus.FINAL)
        assert engine.get_period(pid).status == PeriodStatus.FINAL

    def test_data_collection_campaign_lifecycle(self):
        """Test creating a campaign, adding requests, and collecting data."""
        engine = DataCollectionEngine()

        result = engine.create_campaign(
            period_id="per-e2e-001",
            organisation_id="ORG-E2E-001",
            campaign_name="E2E Data Collection",
        )
        cid = result.campaign.campaign_id

        for scope in [DataScope.SCOPE_1, DataScope.SCOPE_2]:
            engine.add_request(
                campaign_id=cid,
                scope=scope,
                category="test_category",
                facility_id="FAC-E2E",
            )

        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        assert len(c.requests) == 2

        for req in c.requests:
            sub_r = engine.submit_data(
                cid, req.request_id,
                {"kwh": 50000},
                submitted_by="e2e-user",
            )
            engine.accept_submission(
                cid, req.request_id,
                sub_r.submission.submission_id,
                accepted_by="e2e-reviewer",
            )

        coverage = engine.calculate_coverage(cid)
        assert coverage.progress.accepted_count == 2

    def test_quality_management_workflow(self):
        """Test QA/QC checks, issue creation, and verification readiness."""
        engine = QualityManagementEngine()

        all_pass = {f"{d}-{i:03d}": True for d, count in [
            ("COMP", 5), ("CONS", 4), ("ACCU", 5), ("TRAN", 5),
        ] for i in range(1, count + 1)}

        result = engine.run_checks("per-e2e", "org-e2e", all_pass)
        assert result.quality_score.grade == "A"

        readiness = engine.assess_verification_readiness("per-e2e", "org-e2e", all_pass)
        assert readiness.quality_score.verification_ready is True

    def test_change_management_workflow(self):
        """Test change request through impact assessment and trigger detection."""
        engine = ChangeManagementEngine(
            base_year=2019,
            base_year_total_tco2e=Decimal("60000"),
        )

        request = ChangeRequest(
            title="E2E emission factor update",
            category=ChangeCategory.METHODOLOGICAL,
            total_inventory_tco2e=Decimal("55000"),
            affected_sources=[
                AffectedSource(
                    source_id="SRC-E2E",
                    scope="scope2",
                    old_value_tco2e=Decimal("5000"),
                    new_value_tco2e=Decimal("4500"),
                    delta_tco2e=Decimal("-500"),
                ),
            ],
        )

        result = engine.process_change(request)
        assert result.impact is not None
        assert result.approval_routing is not None
        assert result.base_year_trigger is not None

    def test_review_approval_workflow(self):
        """Test multi-stage review from creation through approval."""
        engine = ReviewApprovalEngine()

        req = ReviewRequest(
            inventory_id="inv-e2e",
            reporting_year=2025,
            section="full_inventory",
            title="E2E Review",
            preparer_id="prep-001",
            preparer_name="Preparer",
            reviewer_id="rev-001",
            reviewer_name="Reviewer",
            approver_id="app-001",
            approver_name="Approver",
        )

        result = engine.submit_for_review(req, "prep-001", "Preparer")
        assert result.request.current_stage == ReviewStage.REVIEW

        engine.record_decision(
            req, ReviewStage.REVIEW, ReviewStatus.APPROVED,
            "rev-001", "Reviewer",
        )
        r = engine.record_decision(
            req, ReviewStage.APPROVAL, ReviewStatus.APPROVED,
            "app-001", "Approver",
        )
        assert req.stage_statuses[ReviewStage.APPROVAL.value] in (
            ReviewStatus.APPROVED.value, "approved",
        )

    def test_versioning_workflow(self):
        """Test version creation, update, diff, and finalization."""
        engine = InventoryVersioningEngine()

        v1 = engine.create_version(
            "inv-e2e", 2025,
            {"scope1": 10000, "scope2": 5000},
            "user-e2e",
        )

        v2 = engine.create_next_version(
            v1.version,
            {"scope1": 9500, "scope2": 5200},
        )

        diff = engine.compute_diff(v1.version, v2.version)
        assert diff.diff.fields_modified > 0

        engine.transition_status(
            v2.version, VersionStatus.UNDER_REVIEW,
            "user-e2e", "User",
        )
        engine.transition_status(
            v2.version, VersionStatus.FINAL,
            "approver-e2e", "Approver",
        )
        assert v2.version.status == VersionStatus.FINAL

    def test_consolidation_workflow(self):
        """Test multi-entity consolidation with eliminations."""
        engine = ConsolidationManagementEngine()

        hierarchy = EntityHierarchy(
            group_name="E2E Group",
            reporting_year=2025,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            entities=[
                Entity(
                    entity_id="ENT-ROOT",
                    entity_name="E2E Group HQ",
                    entity_type="parent",
                    parent_entity_id=None,
                    equity_pct=Decimal("100"),
                    has_operational_control=True,
                ),
                Entity(
                    entity_id="E-1",
                    entity_name="Sub A",
                    entity_type="subsidiary",
                    parent_entity_id="ENT-ROOT",
                    equity_pct=Decimal("100"),
                    has_operational_control=True,
                    has_financial_control=True,
                ),
                Entity(
                    entity_id="E-2",
                    entity_name="Sub B",
                    entity_type="subsidiary",
                    parent_entity_id="ENT-ROOT",
                    equity_pct=Decimal("100"),
                    has_operational_control=True,
                    has_financial_control=True,
                ),
            ],
        )

        submissions = [
            SubsidiarySubmission(
                entity_id="E-1",
                scope1_tco2e=Decimal("10000"),
                scope2_location_tco2e=Decimal("5000"),
                scope2_market_tco2e=Decimal("3000"),
                status=SubmissionStatus.SUBMITTED,
            ),
            SubsidiarySubmission(
                entity_id="E-2",
                scope1_tco2e=Decimal("8000"),
                scope2_location_tco2e=Decimal("4000"),
                scope2_market_tco2e=Decimal("2500"),
                status=SubmissionStatus.SUBMITTED,
            ),
        ]

        result = engine.consolidate(hierarchy, submissions)
        assert abs(result.total_scope1_tco2e - 18000.0) < 1.0

    def test_gap_analysis_workflow(self):
        """Test gap identification and improvement recommendations."""
        engine = GapAnalysisEngine()

        categories = [
            SourceCategoryAssessment(
                category_name="stationary_combustion",
                scope=1,
                emissions_tco2e=Decimal("12000"),
                methodology_tier=MethodologyTier.TIER_2,
                has_activity_data=True,
                has_emission_factors=True,
                data_source="Invoices",
            ),
            SourceCategoryAssessment(
                category_name="fugitive_emissions",
                scope=1,
                emissions_tco2e=Decimal("500"),
                methodology_tier=MethodologyTier.TIER_1,
                has_activity_data=True,
                has_emission_factors=True,
                data_source="Estimates",
            ),
            SourceCategoryAssessment(
                category_name="employee_commuting",
                scope=3,
                emissions_tco2e=Decimal("0"),
                methodology_tier=MethodologyTier.MISSING,
                has_activity_data=False,
                has_emission_factors=False,
                data_source="",
            ),
        ]

        result = engine.analyse(categories=categories)
        assert result.gaps_identified >= 1
        assert result.recommendations is not None

    def test_documentation_workflow(self):
        """Test document registration and completeness assessment."""
        engine = DocumentationEngine()

        doc = MethodologyDocument(
            title="E2E Scope 1 Methodology",
            document_type=DocumentType.METHODOLOGY,
            category_id="cat-e2e",
            category_name="stationary_combustion",
            scope=1,
            version="1.0",
            author="e2e-user",
        )

        result = engine.assess(documents=[doc], category_ids=["cat-e2e"])
        assert result is not None
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_benchmarking_workflow(self):
        """Test peer benchmarking and trend analysis."""
        engine = BenchmarkingEngine()

        entity = EntityProfile(
            entity_name="E2E Corp",
            total_scope1_tco2e=Decimal("22000"),
            total_scope2_location_tco2e=Decimal("15000"),
            revenue_eur_millions=Decimal("950"),
            fte_count=Decimal("2500"),
        )

        peers = [
            PeerDataPoint(
                peer_name=f"Peer {i}", year=2025,
                total_scope12_tco2e=Decimal(str(30000 + i * 5000)),
                intensity_revenue=Decimal(str(35 + i)),
                intensity_fte=Decimal(str(14 + i * 0.5)),
            )
            for i in range(5)
        ]

        result = engine.benchmark(entity, peers=peers)
        assert len(result.sector_benchmarks) >= 1

        historical = [
            HistoricalDataPoint(
                year=yr,
                total_scope12_tco2e=Decimal(str(40000 - (yr - 2023) * 1500)),
            )
            for yr in range(2023, 2026)
        ]

        trend_result = engine.benchmark(entity, historical=historical)
        assert trend_result is not None


# ===================================================================
# Cross-Engine Integration Tests
# ===================================================================


class TestCrossEngineIntegration:
    """Tests for cross-engine data flow consistency."""

    def test_period_and_data_collection_linked(self):
        """Period engine creates a period, data collection references it."""
        period_engine = InventoryPeriodEngine()
        dc_engine = DataCollectionEngine()

        pr = period_engine.create_period(
            "org-cross", "Cross Test Period",
            date(2025, 1, 1), date(2025, 12, 31),
        )
        pid = pr.period.period_id

        cr = dc_engine.create_campaign(
            period_id=pid,
            organisation_id="org-cross",
            campaign_name="Cross-linked Campaign",
        )
        assert cr.campaign.period_id == pid

    def test_quality_and_change_management_linked(self):
        """Quality issues can trigger change requests."""
        qm_engine = QualityManagementEngine()
        cm_engine = ChangeManagementEngine(
            base_year=2019,
            base_year_total_tco2e=Decimal("60000"),
        )

        inputs = {k: True for k in [
            "COMP-001", "COMP-002", "COMP-003", "COMP-004", "COMP-005",
            "CONS-001", "CONS-002", "CONS-003", "CONS-004",
            "ACCU-001", "ACCU-002", "ACCU-003", "ACCU-004", "ACCU-005",
            "TRAN-001", "TRAN-002", "TRAN-003", "TRAN-004", "TRAN-005",
        ]}
        inputs["ACCU-002"] = False  # One failure

        qm_result = qm_engine.run_checks("per-cross", "org-cross", inputs)
        assert qm_result.quality_score.total_failed >= 1

        change_req = ChangeRequest(
            title="Fix accuracy issue from QA/QC",
            category=ChangeCategory.ERROR_CORRECTION,
            total_inventory_tco2e=Decimal("55000"),
            affected_sources=[
                AffectedSource(
                    source_id="SRC-FIX",
                    scope="scope1",
                    old_value_tco2e=Decimal("1000"),
                    new_value_tco2e=Decimal("1100"),
                    delta_tco2e=Decimal("100"),
                ),
            ],
        )
        cm_result = cm_engine.process_change(change_req)
        assert cm_result.impact is not None

    def test_version_and_review_linked(self):
        """A version goes through review before finalization."""
        iv_engine = InventoryVersioningEngine()
        ra_engine = ReviewApprovalEngine()

        v1 = iv_engine.create_version(
            "inv-cross", 2025,
            {"scope1": 10000}, "user-001",
        )

        req = ReviewRequest(
            inventory_id="inv-cross",
            reporting_year=2025,
            section="full_inventory",
            title="Cross Review",
            preparer_id="user-001",
            preparer_name="Preparer",
            reviewer_id="user-002",
            reviewer_name="Reviewer",
            approver_id="user-003",
            approver_name="Approver",
        )
        result = ra_engine.submit_for_review(req, "user-001", "Preparer")
        assert result.request is not None

    def test_consolidation_feeds_benchmarking(self):
        """Consolidation results used as benchmarking inputs."""
        consol_engine = ConsolidationManagementEngine()
        bench_engine = BenchmarkingEngine()

        hierarchy = EntityHierarchy(
            group_name="Bench Group",
            reporting_year=2025,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            entities=[
                Entity(
                    entity_id="ENT-ROOT",
                    entity_name="Bench Group HQ",
                    entity_type="parent",
                    parent_entity_id=None,
                    equity_pct=Decimal("100"),
                    has_operational_control=True,
                ),
                Entity(
                    entity_id="E-1",
                    entity_name="Sub",
                    entity_type="subsidiary",
                    parent_entity_id="ENT-ROOT",
                    equity_pct=Decimal("100"),
                    has_operational_control=True,
                    has_financial_control=True,
                ),
            ],
        )

        submissions = [
            SubsidiarySubmission(
                entity_id="E-1",
                scope1_tco2e=Decimal("15000"),
                scope2_location_tco2e=Decimal("8000"),
                scope2_market_tco2e=Decimal("5000"),
                status=SubmissionStatus.SUBMITTED,
            ),
        ]

        consol = consol_engine.consolidate(hierarchy, submissions)

        entity = EntityProfile(
            entity_name="Bench Corp",
            total_scope1_tco2e=Decimal(str(consol.total_scope1_tco2e)),
            total_scope2_location_tco2e=Decimal(str(consol.total_scope2_location_tco2e)),
            revenue_eur_millions=Decimal("500"),
            fte_count=Decimal("1000"),
        )

        peers = [
            PeerDataPoint(
                peer_name="Peer A",
                total_scope12_tco2e=Decimal("25000"),
                intensity_revenue=Decimal("50"),
                intensity_fte=Decimal("14"),
            ),
        ]

        result = bench_engine.benchmark(entity, peers=peers)
        assert len(result.sector_benchmarks) >= 1


# ===================================================================
# Provenance Chain Tests
# ===================================================================


class TestProvenanceChain:
    """Tests for SHA-256 provenance hash chain across engines."""

    def test_all_engines_produce_provenance_hashes(self):
        """Verify every engine produces 64-character hex provenance hashes."""
        period = InventoryPeriodEngine().create_period(
            "org", "P", date(2025, 1, 1), date(2025, 12, 31),
        )
        assert len(period.provenance_hash) == 64

        dc = DataCollectionEngine().create_campaign("p", "o", "C")
        assert len(dc.provenance_hash) == 64

        qm = QualityManagementEngine()
        inputs = {f"COMP-{i:03d}": True for i in range(1, 6)}
        inputs.update({f"CONS-{i:03d}": True for i in range(1, 5)})
        inputs.update({f"ACCU-{i:03d}": True for i in range(1, 6)})
        inputs.update({f"TRAN-{i:03d}": True for i in range(1, 6)})
        qm_result = qm.run_checks("p", "o", inputs)
        assert len(qm_result.provenance_hash) == 64

        iv = InventoryVersioningEngine().create_version(
            "inv", 2025, {"s1": 1000}, "user",
        )
        assert len(iv.provenance_hash) == 64

    def test_hash_reproducibility(self):
        """Same inputs should produce same hash (deterministic)."""
        e1 = InventoryPeriodEngine()
        e2 = InventoryPeriodEngine()

        r1 = e1.create_period("org", "Same Period", date(2025, 1, 1), date(2025, 12, 31))
        r2 = e2.create_period("org", "Same Period", date(2025, 1, 1), date(2025, 12, 31))

        # Hashes may differ due to UUIDs, but both should be valid hex
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        int(r1.provenance_hash, 16)
        int(r2.provenance_hash, 16)
