# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Consolidation Adjustment Engine Tests

Tests adjustment creation, approval workflow, impact calculation,
reversal, batch adjustments, history retrieval, and summary generation.

Target: 40-60 tests.
"""

import pytest
from decimal import Decimal

from engines.consolidation_adjustment_engine import (
    ConsolidationAdjustmentEngine,
    AdjustmentRecord,
    AdjustmentApproval,
    AdjustmentImpact,
    AdjustmentBatch,
    AdjustmentCategory,
    AdjustmentStatus,
    ScopeTarget,
    _round2,
)


@pytest.fixture
def engine():
    """Fresh ConsolidationAdjustmentEngine."""
    return ConsolidationAdjustmentEngine()


@pytest.fixture
def base_adjustment_data():
    """Standard adjustment creation data."""
    return {
        "reporting_year": 2025,
        "entity_id": "ENT-SUB-001",
        "entity_name": "Subsidiary One",
        "category": "ERROR_CORRECTION",
        "scope_target": "SCOPE_1",
        "adjustment_amount_tco2e": Decimal("-500"),
        "before_value_tco2e": Decimal("15000"),
        "justification": "Correction of natural gas conversion factor",
        "evidence_reference": "EV-001",
        "created_by": "analyst@corp.com",
    }


class TestAdjustmentCreation:
    """Test adjustment creation."""

    def test_create_adjustment(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        assert isinstance(adj, AdjustmentRecord)
        assert adj.entity_id == "ENT-SUB-001"
        assert adj.category == "ERROR_CORRECTION"
        assert adj.adjustment_amount_tco2e == Decimal("-500")
        assert adj.status == AdjustmentStatus.DRAFT.value

    def test_auto_calculate_after_value(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        expected_after = _round2(Decimal("15000") + Decimal("-500"))
        assert adj.after_value_tco2e == expected_after

    def test_create_positive_adjustment(self, engine):
        adj = engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "ENT-A",
            "category": "LATE_SUBMISSION",
            "adjustment_amount_tco2e": Decimal("200"),
            "before_value_tco2e": Decimal("5000"),
            "justification": "Late data submission from remote site",
        })
        assert adj.adjustment_amount_tco2e == Decimal("200")
        assert adj.after_value_tco2e == Decimal("5200.00")

    def test_create_forces_draft_status(self, engine):
        adj = engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "ENT-A",
            "category": "OTHER",
            "adjustment_amount_tco2e": Decimal("100"),
            "justification": "Test",
            "status": "APPROVED",  # Should be overridden
        })
        assert adj.status == AdjustmentStatus.DRAFT.value

    def test_create_with_scope_target(self, engine):
        adj = engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "ENT-A",
            "category": "SCOPE_RECLASSIFICATION",
            "scope_target": "SCOPE_3",
            "adjustment_amount_tco2e": Decimal("300"),
            "justification": "Reclassify from Scope 1 to Scope 3",
        })
        assert adj.scope_target == ScopeTarget.SCOPE_3.value

    def test_provenance_hash_on_create(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        assert len(adj.provenance_hash) == 64

    @pytest.mark.parametrize("category", [
        "METHODOLOGY_CHANGE", "ERROR_CORRECTION", "SCOPE_RECLASSIFICATION",
        "TIMING_ADJUSTMENT", "LATE_SUBMISSION", "DATA_QUALITY",
        "BOUNDARY_CHANGE", "OTHER",
    ])
    def test_all_categories_accepted(self, engine, category):
        adj = engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "ENT-A",
            "category": category,
            "adjustment_amount_tco2e": Decimal("100"),
            "justification": f"Test {category}",
        })
        assert adj.category == category


class TestApprovalWorkflow:
    """Test adjustment approval workflow."""

    def test_submit_adjustment(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        submitted = engine.submit_adjustment(adj.adjustment_id, "submitter@corp.com")
        assert submitted.status == AdjustmentStatus.SUBMITTED.value

    def test_review_adjustment(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "submitter@corp.com")
        reviewed = engine.review_adjustment(adj.adjustment_id, "reviewer@corp.com")
        assert reviewed.status == AdjustmentStatus.REVIEWED.value

    def test_approve_adjustment(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "submitter@corp.com")
        engine.review_adjustment(adj.adjustment_id, "reviewer@corp.com")
        approved = engine.approve_adjustment(adj.adjustment_id, "approver@corp.com")
        assert approved.status == AdjustmentStatus.APPROVED.value

    def test_reject_adjustment_from_submitted(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "submitter@corp.com")
        rejected = engine.reject_adjustment(
            adj.adjustment_id, "reviewer@corp.com", "Insufficient evidence"
        )
        assert rejected.status == AdjustmentStatus.REJECTED.value

    def test_reject_adjustment_from_reviewed(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "submitter@corp.com")
        engine.review_adjustment(adj.adjustment_id, "reviewer@corp.com")
        rejected = engine.reject_adjustment(
            adj.adjustment_id, "approver@corp.com", "Not justified"
        )
        assert rejected.status == AdjustmentStatus.REJECTED.value

    def test_invalid_transition_draft_to_approved(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        with pytest.raises(ValueError, match="Cannot transition"):
            engine.approve_adjustment(adj.adjustment_id, "approver@corp.com")

    def test_invalid_transition_rejected_to_approved(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "submitter@corp.com")
        engine.reject_adjustment(adj.adjustment_id, "reviewer@corp.com", "Bad")
        with pytest.raises(ValueError, match="Cannot transition"):
            engine.approve_adjustment(adj.adjustment_id, "approver@corp.com")

    def test_approval_history_tracked(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "submitter@corp.com")
        engine.review_adjustment(adj.adjustment_id, "reviewer@corp.com")
        engine.approve_adjustment(adj.adjustment_id, "approver@corp.com")
        history = engine.get_approval_history(adj.adjustment_id)
        assert len(history) == 3
        assert history[0].action == AdjustmentStatus.SUBMITTED.value
        assert history[1].action == AdjustmentStatus.REVIEWED.value
        assert history[2].action == AdjustmentStatus.APPROVED.value

    def test_approval_provenance_hash(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "submitter@corp.com")
        history = engine.get_approval_history(adj.adjustment_id)
        assert len(history[0].provenance_hash) == 64

    def test_not_found_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.submit_adjustment("NONEXISTENT", "user@corp.com")


class TestReversal:
    """Test adjustment reversal."""

    def test_reverse_approved_adjustment(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "sub")
        engine.review_adjustment(adj.adjustment_id, "rev")
        engine.approve_adjustment(adj.adjustment_id, "app")

        reversal = engine.reverse_adjustment(
            adj.adjustment_id, "admin@corp.com", "Data was incorrect"
        )
        assert reversal.is_reversal is True
        assert reversal.reversal_of_id == adj.adjustment_id
        assert reversal.adjustment_amount_tco2e == Decimal("500")

    def test_reverse_marks_original_as_reversed(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        engine.submit_adjustment(adj.adjustment_id, "sub")
        engine.review_adjustment(adj.adjustment_id, "rev")
        engine.approve_adjustment(adj.adjustment_id, "app")
        engine.reverse_adjustment(adj.adjustment_id, "admin@corp.com", "Error")

        original = engine.get_adjustment(adj.adjustment_id)
        assert original.status == AdjustmentStatus.REVERSED.value

    def test_cannot_reverse_draft(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        with pytest.raises(ValueError, match="APPROVED"):
            engine.reverse_adjustment(adj.adjustment_id, "admin", "test")

    def test_cannot_reverse_nonexistent(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.reverse_adjustment("NONEXISTENT", "admin", "test")


class TestBatchAdjustments:
    """Test batch adjustment operations."""

    def test_create_batch(self, engine):
        batch = engine.create_batch(
            reporting_year=2025,
            description="Month-end corrections",
            adjustments=[
                {
                    "entity_id": "ENT-A",
                    "category": "ERROR_CORRECTION",
                    "adjustment_amount_tco2e": Decimal("-100"),
                    "justification": "Fix A",
                },
                {
                    "entity_id": "ENT-B",
                    "category": "ERROR_CORRECTION",
                    "adjustment_amount_tco2e": Decimal("200"),
                    "justification": "Fix B",
                },
            ],
            created_by="analyst@corp.com",
        )
        assert isinstance(batch, AdjustmentBatch)
        assert len(batch.adjustment_ids) == 2
        assert batch.total_adjustment_tco2e == Decimal("100.00")

    def test_batch_adjustments_have_batch_id(self, engine):
        batch = engine.create_batch(
            reporting_year=2025,
            description="Test batch",
            adjustments=[
                {
                    "entity_id": "ENT-A",
                    "category": "OTHER",
                    "adjustment_amount_tco2e": Decimal("50"),
                    "justification": "Test",
                },
            ],
        )
        adj = engine.get_adjustment(batch.adjustment_ids[0])
        assert adj.batch_id == batch.batch_id

    def test_batch_provenance_hash(self, engine):
        batch = engine.create_batch(
            reporting_year=2025,
            description="Test",
            adjustments=[
                {
                    "entity_id": "ENT-A",
                    "category": "OTHER",
                    "adjustment_amount_tco2e": Decimal("50"),
                    "justification": "Test",
                },
            ],
        )
        assert len(batch.provenance_hash) == 64

    def test_get_batch(self, engine):
        batch = engine.create_batch(
            reporting_year=2025,
            description="Test",
            adjustments=[
                {
                    "entity_id": "ENT-A",
                    "category": "OTHER",
                    "adjustment_amount_tco2e": Decimal("50"),
                    "justification": "Test",
                },
            ],
        )
        retrieved = engine.get_batch(batch.batch_id)
        assert retrieved.description == "Test"

    def test_get_batch_not_found(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_batch("NONEXISTENT")


class TestImpactCalculation:
    """Test adjustment impact calculation."""

    def test_impact_approved_only(self, engine):
        adj1 = engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "ENT-A",
            "category": "ERROR_CORRECTION",
            "scope_target": "SCOPE_1",
            "adjustment_amount_tco2e": Decimal("-500"),
            "justification": "Fix 1",
        })
        adj2 = engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "ENT-B",
            "category": "LATE_SUBMISSION",
            "scope_target": "SCOPE_1",
            "adjustment_amount_tco2e": Decimal("200"),
            "justification": "Fix 2",
        })
        # Only approve adj1
        engine.submit_adjustment(adj1.adjustment_id, "sub")
        engine.review_adjustment(adj1.adjustment_id, "rev")
        engine.approve_adjustment(adj1.adjustment_id, "app")

        impact = engine.calculate_impact(2025, pre_adjustment_total=Decimal("100000"))
        assert isinstance(impact, AdjustmentImpact)
        assert impact.adjustments_included == 1
        assert impact.total_adjustment_tco2e == Decimal("-500.00")

    def test_impact_scope_breakdown(self, engine):
        adj = engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "ENT-A",
            "category": "ERROR_CORRECTION",
            "scope_target": "SCOPE_1",
            "adjustment_amount_tco2e": Decimal("-300"),
            "justification": "Fix",
        })
        engine.submit_adjustment(adj.adjustment_id, "sub")
        engine.review_adjustment(adj.adjustment_id, "rev")
        engine.approve_adjustment(adj.adjustment_id, "app")

        impact = engine.calculate_impact(2025)
        assert impact.scope1_adjustment == Decimal("-300.00")
        assert impact.scope2_location_adjustment == Decimal("0.00")

    def test_impact_by_category(self, engine):
        for cat in ["ERROR_CORRECTION", "ERROR_CORRECTION", "LATE_SUBMISSION"]:
            adj = engine.create_adjustment({
                "reporting_year": 2025,
                "entity_id": "ENT-A",
                "category": cat,
                "adjustment_amount_tco2e": Decimal("100"),
                "justification": f"Test {cat}",
            })
            engine.submit_adjustment(adj.adjustment_id, "sub")
            engine.review_adjustment(adj.adjustment_id, "rev")
            engine.approve_adjustment(adj.adjustment_id, "app")

        impact = engine.calculate_impact(2025)
        assert impact.by_category["ERROR_CORRECTION"] == Decimal("200.00")
        assert impact.by_category["LATE_SUBMISSION"] == Decimal("100.00")

    def test_impact_net_pct(self, engine):
        adj = engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "ENT-A",
            "category": "ERROR_CORRECTION",
            "adjustment_amount_tco2e": Decimal("-1000"),
            "justification": "Fix",
        })
        engine.submit_adjustment(adj.adjustment_id, "sub")
        engine.review_adjustment(adj.adjustment_id, "rev")
        engine.approve_adjustment(adj.adjustment_id, "app")

        impact = engine.calculate_impact(2025, pre_adjustment_total=Decimal("50000"))
        # -1000/50000 * 100 = -2.00%
        assert impact.net_impact_pct == Decimal("-2.00")
        assert impact.post_adjustment_total == Decimal("49000.00")

    def test_impact_provenance_hash(self, engine):
        impact = engine.calculate_impact(2025)
        assert len(impact.provenance_hash) == 64

    def test_empty_impact(self, engine):
        impact = engine.calculate_impact(2025)
        assert impact.adjustments_included == 0
        assert impact.total_adjustment_tco2e == Decimal("0.00")


class TestHistoryAndAccessors:
    """Test history and accessor methods."""

    def test_get_adjustment(self, engine, base_adjustment_data):
        adj = engine.create_adjustment(base_adjustment_data)
        retrieved = engine.get_adjustment(adj.adjustment_id)
        assert retrieved.entity_id == "ENT-SUB-001"

    def test_get_adjustment_not_found(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_adjustment("NONEXISTENT")

    def test_history_filter_by_entity(self, engine):
        engine.create_adjustment({
            "reporting_year": 2025, "entity_id": "A",
            "category": "OTHER", "adjustment_amount_tco2e": Decimal("100"),
            "justification": "Test",
        })
        engine.create_adjustment({
            "reporting_year": 2025, "entity_id": "B",
            "category": "OTHER", "adjustment_amount_tco2e": Decimal("200"),
            "justification": "Test",
        })
        history = engine.get_adjustment_history(entity_id="A")
        assert len(history) == 1

    def test_history_filter_by_category(self, engine):
        engine.create_adjustment({
            "reporting_year": 2025, "entity_id": "A",
            "category": "ERROR_CORRECTION",
            "adjustment_amount_tco2e": Decimal("100"),
            "justification": "Test",
        })
        engine.create_adjustment({
            "reporting_year": 2025, "entity_id": "A",
            "category": "LATE_SUBMISSION",
            "adjustment_amount_tco2e": Decimal("200"),
            "justification": "Test",
        })
        history = engine.get_adjustment_history(category="ERROR_CORRECTION")
        assert len(history) == 1

    def test_history_filter_by_status(self, engine):
        adj = engine.create_adjustment({
            "reporting_year": 2025, "entity_id": "A",
            "category": "OTHER", "adjustment_amount_tco2e": Decimal("100"),
            "justification": "Test",
        })
        engine.submit_adjustment(adj.adjustment_id, "sub")
        drafts = engine.get_adjustment_history(status="DRAFT")
        submitted = engine.get_adjustment_history(status="SUBMITTED")
        assert len(drafts) == 0
        assert len(submitted) == 1

    def test_adjustments_summary(self, engine):
        adj = engine.create_adjustment({
            "reporting_year": 2025, "entity_id": "A",
            "category": "ERROR_CORRECTION",
            "adjustment_amount_tco2e": Decimal("-500"),
            "justification": "Test",
        })
        engine.submit_adjustment(adj.adjustment_id, "sub")
        engine.review_adjustment(adj.adjustment_id, "rev")
        engine.approve_adjustment(adj.adjustment_id, "app")

        summary = engine.get_adjustments_summary(2025)
        assert summary["total_adjustments"] == 1
        assert summary["by_status"]["APPROVED"] == 1
        assert summary["total_approved_tco2e"] == "-500.00"
        assert "provenance_hash" in summary
