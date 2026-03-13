# -*- coding: utf-8 -*-
"""
Unit tests for CommodityDueDiligenceEngine (AGENT-EUDR-018 Engine 7).

Tests DD workflow lifecycle: initiation, evidence submission, verification,
completion, escalation, pending workflows, evidence requirements per
commodity, completion percentage, and workflow history.

Coverage target: 85%+
"""

from decimal import Decimal
import pytest

from greenlang.agents.eudr.commodity_risk_analyzer.engines.commodity_due_diligence_engine import (
    CommodityDueDiligenceEngine,
    EUDR_COMMODITIES,
    COMMODITY_EVIDENCE_TEMPLATES,
    WORKFLOW_STATES,
    TERMINAL_STATES,
    VALID_TRIGGERS,
    DDWorkflow,
    EvidenceItem,
)

SEVEN_COMMODITIES = sorted(EUDR_COMMODITIES)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for CommodityDueDiligenceEngine initialization."""

    @pytest.mark.unit
    def test_init_empty_workflows(self):
        """Engine initializes with empty workflow store."""
        engine = CommodityDueDiligenceEngine()
        assert engine._workflows == {}

    @pytest.mark.unit
    def test_init_creates_lock(self):
        """Engine creates a reentrant lock."""
        engine = CommodityDueDiligenceEngine()
        assert engine._lock is not None


# ---------------------------------------------------------------------------
# TestInitiateWorkflow
# ---------------------------------------------------------------------------

class TestInitiateWorkflow:
    """Tests for initiate_workflow method."""

    @pytest.mark.unit
    def test_initiate_basic(self, commodity_dd_engine):
        """Initiating a workflow returns INITIATED status."""
        result = commodity_dd_engine.initiate_workflow("wood", "SUP-001")
        assert result["status"] == "INITIATED"
        assert result["commodity_type"] == "wood"
        assert result["supplier_id"] == "SUP-001"
        assert result["trigger"] == "scheduled"

    @pytest.mark.unit
    def test_initiate_populates_evidence(self, commodity_dd_engine):
        """Initiated workflow has evidence items from template."""
        template_count = len(COMMODITY_EVIDENCE_TEMPLATES["wood"])
        result = commodity_dd_engine.initiate_workflow("wood", "SUP-002")
        assert result["evidence_count"] == template_count

    @pytest.mark.unit
    def test_initiate_manual_trigger(self, commodity_dd_engine):
        """Manual trigger is accepted."""
        result = commodity_dd_engine.initiate_workflow("cocoa", "SUP-003", trigger="manual")
        assert result["trigger"] == "manual"

    @pytest.mark.unit
    def test_invalid_commodity_raises(self, commodity_dd_engine):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="not a valid EUDR commodity"):
            commodity_dd_engine.initiate_workflow("banana", "SUP-004")

    @pytest.mark.unit
    def test_empty_supplier_raises(self, commodity_dd_engine):
        """Empty supplier_id raises ValueError."""
        with pytest.raises(ValueError, match="supplier_id"):
            commodity_dd_engine.initiate_workflow("soya", "")

    @pytest.mark.unit
    def test_invalid_trigger_raises(self, commodity_dd_engine):
        """Invalid trigger type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid trigger"):
            commodity_dd_engine.initiate_workflow("soya", "SUP-005", trigger="invalid")


# ---------------------------------------------------------------------------
# TestWorkflowStatus
# ---------------------------------------------------------------------------

class TestWorkflowStatus:
    """Tests for get_workflow_status method."""

    @pytest.mark.unit
    def test_status_returns_initiated(self, commodity_dd_engine):
        """Newly created workflow returns INITIATED status."""
        wf = commodity_dd_engine.initiate_workflow("coffee", "SUP-010")
        status = commodity_dd_engine.get_workflow_status(wf["workflow_id"])
        assert status["status"] == "INITIATED"
        assert "days_elapsed" in status
        assert "days_remaining" in status

    @pytest.mark.unit
    def test_status_invalid_id_raises(self, commodity_dd_engine):
        """Non-existent workflow_id raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            commodity_dd_engine.get_workflow_status("wf-nonexistent")


# ---------------------------------------------------------------------------
# TestSubmitEvidence
# ---------------------------------------------------------------------------

class TestSubmitEvidence:
    """Tests for submit_evidence method."""

    @pytest.mark.unit
    def test_submit_transitions_to_in_progress(self, commodity_dd_engine):
        """First evidence submission transitions workflow to IN_PROGRESS."""
        wf = commodity_dd_engine.initiate_workflow("wood", "SUP-020")
        result = commodity_dd_engine.submit_evidence(
            wf["workflow_id"],
            "species_identification",
            {"genus": "Tectona", "species": "grandis"},
        )
        assert result["submission_status"] == "ACCEPTED"
        assert result["workflow_status"] == "IN_PROGRESS"

    @pytest.mark.unit
    def test_submit_invalid_evidence_type_raises(self, commodity_dd_engine):
        """Submitting an unknown evidence type raises ValueError."""
        wf = commodity_dd_engine.initiate_workflow("soya", "SUP-021")
        with pytest.raises(ValueError, match="not expected"):
            commodity_dd_engine.submit_evidence(
                wf["workflow_id"],
                "unknown_evidence_type",
                {"data": "value"},
            )

    @pytest.mark.unit
    def test_submit_to_completed_workflow_raises(self, commodity_dd_engine):
        """Submitting evidence to a COMPLETED workflow raises ValueError."""
        wf = commodity_dd_engine.initiate_workflow("cocoa", "SUP-022")
        wf_id = wf["workflow_id"]
        # Submit and verify all mandatory evidence to complete
        for item in wf["evidence_items"]:
            if item["mandatory"]:
                commodity_dd_engine.submit_evidence(
                    wf_id, item["evidence_type"],
                    _make_valid_evidence(item["verification_method"]),
                )
                evidence_id = None
                status = commodity_dd_engine.get_workflow_status(wf_id)
                for ei in status["evidence_items"]:
                    if ei["evidence_type"] == item["evidence_type"]:
                        evidence_id = ei["evidence_id"]
                        break
                if evidence_id:
                    commodity_dd_engine.verify_evidence(wf_id, evidence_id)
        commodity_dd_engine.complete_workflow(wf_id)

        with pytest.raises(ValueError, match="terminal"):
            commodity_dd_engine.submit_evidence(
                wf_id, "satellite_monitoring", {"data": "img"},
            )


# ---------------------------------------------------------------------------
# TestVerifyEvidence
# ---------------------------------------------------------------------------

class TestVerifyEvidence:
    """Tests for verify_evidence method."""

    @pytest.mark.unit
    def test_verify_with_valid_data(self, commodity_dd_engine):
        """Valid GPS evidence passes verification."""
        wf = commodity_dd_engine.initiate_workflow("soya", "SUP-030")
        wf_id = wf["workflow_id"]
        commodity_dd_engine.submit_evidence(
            wf_id, "farm_gps", {"latitude": -15.79, "longitude": -47.88},
        )
        # Get evidence_id
        status = commodity_dd_engine.get_workflow_status(wf_id)
        evi_id = None
        for item in status["evidence_items"]:
            if item["evidence_type"] == "farm_gps":
                evi_id = item["evidence_id"]
                break

        result = commodity_dd_engine.verify_evidence(wf_id, evi_id)
        assert result["verification_status"] == "VERIFIED"
        assert result["issues"] == []

    @pytest.mark.unit
    def test_verify_with_missing_coords_rejected(self, commodity_dd_engine):
        """GPS evidence without lat/lon is rejected."""
        wf = commodity_dd_engine.initiate_workflow("coffee", "SUP-031")
        wf_id = wf["workflow_id"]
        commodity_dd_engine.submit_evidence(wf_id, "farm_gps", {"note": "no coords"})
        status = commodity_dd_engine.get_workflow_status(wf_id)
        evi_id = None
        for item in status["evidence_items"]:
            if item["evidence_type"] == "farm_gps":
                evi_id = item["evidence_id"]
                break
        result = commodity_dd_engine.verify_evidence(wf_id, evi_id)
        assert result["verification_status"] == "REJECTED"
        assert len(result["issues"]) > 0

    @pytest.mark.unit
    def test_verify_unsubmitted_raises(self, commodity_dd_engine):
        """Verifying unsubmitted evidence raises ValueError."""
        wf = commodity_dd_engine.initiate_workflow("rubber", "SUP-032")
        wf_id = wf["workflow_id"]
        evi_id = wf["evidence_items"][0]["evidence_id"]
        with pytest.raises(ValueError, match="not been submitted"):
            commodity_dd_engine.verify_evidence(wf_id, evi_id)

    @pytest.mark.unit
    def test_verify_invalid_evidence_id_raises(self, commodity_dd_engine):
        """Unknown evidence_id raises ValueError."""
        wf = commodity_dd_engine.initiate_workflow("cattle", "SUP-033")
        with pytest.raises(ValueError, match="not found"):
            commodity_dd_engine.verify_evidence(wf["workflow_id"], "evi-nonexistent")


# ---------------------------------------------------------------------------
# TestCompleteWorkflow
# ---------------------------------------------------------------------------

class TestCompleteWorkflow:
    """Tests for complete_workflow method."""

    @pytest.mark.unit
    def test_complete_with_missing_mandatory_fails(self, commodity_dd_engine):
        """Workflow with missing mandatory evidence completes as FAILED."""
        wf = commodity_dd_engine.initiate_workflow("wood", "SUP-040")
        result = commodity_dd_engine.complete_workflow(wf["workflow_id"])
        assert result["status"] == "FAILED"
        assert result["completion_result"] == "FAIL"
        assert len(result["mandatory_missing"]) > 0

    @pytest.mark.unit
    def test_complete_already_terminal_raises(self, commodity_dd_engine):
        """Completing an already-completed workflow raises ValueError."""
        wf = commodity_dd_engine.initiate_workflow("cocoa", "SUP-041")
        wf_id = wf["workflow_id"]
        commodity_dd_engine.complete_workflow(wf_id)  # FAILED first time
        with pytest.raises(ValueError, match="terminal"):
            commodity_dd_engine.complete_workflow(wf_id)

    @pytest.mark.unit
    def test_complete_with_assessor_notes(self, commodity_dd_engine):
        """Assessor notes are recorded in the completion result."""
        wf = commodity_dd_engine.initiate_workflow("soya", "SUP-042")
        result = commodity_dd_engine.complete_workflow(
            wf["workflow_id"], assessor_notes="Reviewed by auditor",
        )
        assert result["assessor_notes"] == "Reviewed by auditor"


# ---------------------------------------------------------------------------
# TestPendingWorkflows
# ---------------------------------------------------------------------------

class TestPendingWorkflows:
    """Tests for get_pending_workflows method."""

    @pytest.mark.unit
    def test_pending_includes_initiated(self, commodity_dd_engine):
        """INITIATED workflows appear in pending list."""
        commodity_dd_engine.initiate_workflow("rubber", "SUP-050")
        pending = commodity_dd_engine.get_pending_workflows()
        assert len(pending) >= 1

    @pytest.mark.unit
    def test_pending_excludes_completed(self, commodity_dd_engine):
        """COMPLETED workflows do not appear in pending list."""
        wf = commodity_dd_engine.initiate_workflow("cattle", "SUP-051")
        commodity_dd_engine.complete_workflow(wf["workflow_id"])
        pending = commodity_dd_engine.get_pending_workflows()
        wf_ids = [p["workflow_id"] for p in pending]
        assert wf["workflow_id"] not in wf_ids

    @pytest.mark.unit
    def test_pending_filter_by_commodity(self, commodity_dd_engine):
        """Filter by commodity returns only matching workflows."""
        commodity_dd_engine.initiate_workflow("cocoa", "SUP-052")
        commodity_dd_engine.initiate_workflow("coffee", "SUP-053")
        pending = commodity_dd_engine.get_pending_workflows(commodity_type="cocoa")
        for p in pending:
            assert p["commodity_type"] == "cocoa"


# ---------------------------------------------------------------------------
# TestEvidenceRequirements
# ---------------------------------------------------------------------------

class TestEvidenceRequirements:
    """Tests for get_evidence_requirements method."""

    @pytest.mark.unit
    @pytest.mark.parametrize("commodity", SEVEN_COMMODITIES)
    def test_evidence_requirements_non_empty(self, commodity_dd_engine, commodity):
        """Every commodity has at least 5 evidence requirements."""
        reqs = commodity_dd_engine.get_evidence_requirements(commodity)
        assert len(reqs) >= 5

    @pytest.mark.unit
    def test_wood_has_species_identification(self, commodity_dd_engine):
        """Wood evidence requires species_identification."""
        reqs = commodity_dd_engine.get_evidence_requirements("wood")
        types = [r["evidence_type"] for r in reqs]
        assert "species_identification" in types

    @pytest.mark.unit
    def test_soya_has_gmo_declaration(self, commodity_dd_engine):
        """Soya evidence requires gmo_declaration."""
        reqs = commodity_dd_engine.get_evidence_requirements("soya")
        types = [r["evidence_type"] for r in reqs]
        assert "gmo_declaration" in types

    @pytest.mark.unit
    def test_invalid_commodity_raises(self, commodity_dd_engine):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError):
            commodity_dd_engine.get_evidence_requirements("banana")


# ---------------------------------------------------------------------------
# TestCompletionPercentage
# ---------------------------------------------------------------------------

class TestCompletionPercentage:
    """Tests for calculate_completion_percentage method."""

    @pytest.mark.unit
    def test_zero_completion_at_start(self, commodity_dd_engine):
        """Newly initiated workflow has 0% completion."""
        wf = commodity_dd_engine.initiate_workflow("oil_palm", "SUP-060")
        pct = commodity_dd_engine.calculate_completion_percentage(wf["workflow_id"])
        assert pct == Decimal("0")

    @pytest.mark.unit
    def test_completion_increases_with_submission(self, commodity_dd_engine):
        """Submitting evidence increases completion percentage."""
        wf = commodity_dd_engine.initiate_workflow("soya", "SUP-061")
        wf_id = wf["workflow_id"]
        commodity_dd_engine.submit_evidence(
            wf_id, "farm_gps", {"latitude": -15.79, "longitude": -47.88},
        )
        pct = commodity_dd_engine.calculate_completion_percentage(wf_id)
        assert pct > Decimal("0")

    @pytest.mark.unit
    def test_invalid_workflow_raises(self, commodity_dd_engine):
        """Non-existent workflow raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            commodity_dd_engine.calculate_completion_percentage("wf-nonexistent")


# ---------------------------------------------------------------------------
# TestEscalateWorkflow
# ---------------------------------------------------------------------------

class TestEscalateWorkflow:
    """Tests for escalate_workflow method."""

    @pytest.mark.unit
    def test_escalate_from_initiated(self, commodity_dd_engine):
        """Escalating from INITIATED transitions to ESCALATED."""
        wf = commodity_dd_engine.initiate_workflow("cattle", "SUP-070")
        result = commodity_dd_engine.escalate_workflow(
            wf["workflow_id"],
            reason="Missing critical evidence",
            escalate_to="Senior Compliance Officer",
        )
        assert result["new_status"] == "ESCALATED"
        assert result["previous_status"] == "INITIATED"

    @pytest.mark.unit
    def test_escalate_empty_reason_raises(self, commodity_dd_engine):
        """Empty reason raises ValueError."""
        wf = commodity_dd_engine.initiate_workflow("cocoa", "SUP-071")
        with pytest.raises(ValueError, match="reason"):
            commodity_dd_engine.escalate_workflow(wf["workflow_id"], "", "Manager")

    @pytest.mark.unit
    def test_escalate_empty_target_raises(self, commodity_dd_engine):
        """Empty escalate_to raises ValueError."""
        wf = commodity_dd_engine.initiate_workflow("coffee", "SUP-072")
        with pytest.raises(ValueError, match="escalate_to"):
            commodity_dd_engine.escalate_workflow(wf["workflow_id"], "Reason", "")

    @pytest.mark.unit
    def test_escalate_terminal_state_raises(self, commodity_dd_engine):
        """Escalating a COMPLETED workflow raises ValueError."""
        wf = commodity_dd_engine.initiate_workflow("rubber", "SUP-073")
        commodity_dd_engine.complete_workflow(wf["workflow_id"])
        with pytest.raises(ValueError, match="terminal"):
            commodity_dd_engine.escalate_workflow(
                wf["workflow_id"], "Reason", "Manager",
            )


# ---------------------------------------------------------------------------
# TestWorkflowHistory
# ---------------------------------------------------------------------------

class TestWorkflowHistory:
    """Tests for get_workflow_history method."""

    @pytest.mark.unit
    def test_history_returns_supplier_workflows(self, commodity_dd_engine):
        """History returns all workflows for a supplier."""
        commodity_dd_engine.initiate_workflow("wood", "SUP-080")
        commodity_dd_engine.initiate_workflow("soya", "SUP-080")
        history = commodity_dd_engine.get_workflow_history("SUP-080")
        assert len(history) >= 2

    @pytest.mark.unit
    def test_history_filter_by_commodity(self, commodity_dd_engine):
        """History can be filtered by commodity."""
        commodity_dd_engine.initiate_workflow("cocoa", "SUP-081")
        commodity_dd_engine.initiate_workflow("coffee", "SUP-081")
        history = commodity_dd_engine.get_workflow_history("SUP-081", commodity_type="cocoa")
        for entry in history:
            assert entry["commodity_type"] == "cocoa"

    @pytest.mark.unit
    def test_history_empty_supplier_raises(self, commodity_dd_engine):
        """Empty supplier_id raises ValueError."""
        with pytest.raises(ValueError, match="supplier_id"):
            commodity_dd_engine.get_workflow_history("")

    @pytest.mark.unit
    def test_history_sorted_descending(self, commodity_dd_engine):
        """History is sorted by created_at descending."""
        commodity_dd_engine.initiate_workflow("cattle", "SUP-082")
        commodity_dd_engine.initiate_workflow("cattle", "SUP-082")
        history = commodity_dd_engine.get_workflow_history("SUP-082")
        dates = [h["created_at"] for h in history]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------

class TestProvenance:
    """Tests for provenance hash integrity."""

    @pytest.mark.unit
    def test_workflow_provenance(self, commodity_dd_engine):
        """Initiated workflow has a 64-char SHA-256 hash."""
        wf = commodity_dd_engine.initiate_workflow("oil_palm", "SUP-090")
        assert len(wf["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_evidence_submission_provenance(self, commodity_dd_engine):
        """Evidence submission result has provenance hash."""
        wf = commodity_dd_engine.initiate_workflow("soya", "SUP-091")
        result = commodity_dd_engine.submit_evidence(
            wf["workflow_id"], "farm_gps", {"latitude": -10.0, "longitude": -50.0},
        )
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.unit
    def test_initiate_invalid_commodity(self, commodity_dd_engine):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError):
            commodity_dd_engine.initiate_workflow("wheat", "SUP-ERR1")

    @pytest.mark.unit
    def test_get_status_nonexistent(self, commodity_dd_engine):
        """Non-existent workflow raises ValueError."""
        with pytest.raises(ValueError):
            commodity_dd_engine.get_workflow_status("wf-does-not-exist")

    @pytest.mark.unit
    def test_history_invalid_commodity_filter(self, commodity_dd_engine):
        """Invalid commodity filter in history raises ValueError."""
        with pytest.raises(ValueError):
            commodity_dd_engine.get_workflow_history("SUP-ERR2", commodity_type="wheat")

    @pytest.mark.unit
    def test_pending_invalid_commodity_filter(self, commodity_dd_engine):
        """Invalid commodity filter in pending raises ValueError."""
        with pytest.raises(ValueError):
            commodity_dd_engine.get_pending_workflows(commodity_type="wheat")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_valid_evidence(verification_method: str) -> dict:
    """Create minimal valid evidence payload for a verification method."""
    evidence_map = {
        "coordinate_validation": {"latitude": -5.0, "longitude": 105.0},
        "polygon_validation": {"coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        "document_review": {"document_id": "DOC-001", "content": "text"},
        "certificate_validation": {"certificate_number": "CERT-001", "expiry_date": "2027-12-31"},
        "chain_of_custody_check": {"stages": ["farm", "mill", "port"]},
        "signature_verification": {"signatory": "John Doe", "date_signed": "2025-06-01"},
        "species_verification": {"genus": "Tectona", "species": "grandis"},
        "facility_verification": {"facility_id": "FAC-001"},
        "license_verification": {"license_number": "LIC-001", "issuing_authority": "Ministry"},
        "satellite_cross_reference": {"imagery_id": "SAT-001", "data": True},
        "registry_lookup": {"registry_id": "REG-001", "data": True},
        "map_validation": {"map_data": True, "data": True},
        "policy_verification": {"policy_id": "POL-001", "data": True},
        "technical_review": {"report_id": "RPT-001", "data": True},
        "permit_verification": {"permit_id": "PER-001", "data": True},
    }
    return evidence_map.get(verification_method, {"data": "generic"})
