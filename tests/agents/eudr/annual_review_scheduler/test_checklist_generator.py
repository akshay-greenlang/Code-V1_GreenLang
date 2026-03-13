# -*- coding: utf-8 -*-
"""
Unit tests for ChecklistGenerator - AGENT-EUDR-034

Tests checklist generation from templates, item management, completion
tracking, evidence attachment, phase-specific checklists, commodity
customization, and progress calculation.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (Engine 3: Checklist Generator)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
)
from greenlang.agents.eudr.annual_review_scheduler.checklist_generator import (
    ChecklistGenerator,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    ChecklistItem,
    ChecklistItemStatus,
    ChecklistTemplate,
    EUDRCommodity,
    ReviewPhase,
)
from greenlang.agents.eudr.annual_review_scheduler.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return AnnualReviewSchedulerConfig()


@pytest.fixture
def generator(config):
    return ChecklistGenerator(config=config, provenance=ProvenanceTracker())


# ---------------------------------------------------------------------------
# Generate Checklist from Template
# ---------------------------------------------------------------------------

class TestGenerateFromTemplate:
    """Test checklist generation from templates."""

    @pytest.mark.asyncio
    async def test_generate_returns_items(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001",
            template=sample_checklist_template,
        )
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_generated_items_have_cycle_id(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001",
            template=sample_checklist_template,
        )
        for item in items:
            assert item.cycle_id == "cyc-001"

    @pytest.mark.asyncio
    async def test_generated_items_have_unique_ids(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001",
            template=sample_checklist_template,
        )
        ids = [item.item_id for item in items]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_generated_items_all_pending(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001",
            template=sample_checklist_template,
        )
        for item in items:
            assert item.status == ChecklistItemStatus.PENDING

    @pytest.mark.asyncio
    async def test_generated_items_preserve_phase(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001",
            template=sample_checklist_template,
        )
        for item in items:
            assert item.phase == ReviewPhase.DATA_COLLECTION

    @pytest.mark.asyncio
    async def test_generated_items_preserve_priority(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001",
            template=sample_checklist_template,
        )
        priorities = [item.priority for item in items]
        assert priorities == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_generated_items_preserve_required_flag(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001",
            template=sample_checklist_template,
        )
        required_flags = [item.required for item in items]
        assert required_flags == [True, True, False]


# ---------------------------------------------------------------------------
# Generate for Cycle
# ---------------------------------------------------------------------------

class TestGenerateForCycle:
    """Test checklist generation for entire review cycle."""

    @pytest.mark.asyncio
    async def test_generate_for_cycle_covers_all_phases(self, generator, sample_review_cycle):
        items = await generator.generate_for_cycle(sample_review_cycle)
        phases = {item.phase for item in items}
        assert ReviewPhase.PREPARATION in phases
        assert ReviewPhase.DATA_COLLECTION in phases
        assert ReviewPhase.ANALYSIS in phases

    @pytest.mark.asyncio
    async def test_generate_for_cycle_returns_items(self, generator, sample_review_cycle):
        items = await generator.generate_for_cycle(sample_review_cycle)
        assert len(items) > 0

    @pytest.mark.asyncio
    async def test_generate_for_cycle_items_tied_to_cycle(self, generator, sample_review_cycle):
        items = await generator.generate_for_cycle(sample_review_cycle)
        for item in items:
            assert item.cycle_id == sample_review_cycle.cycle_id

    @pytest.mark.asyncio
    async def test_generate_for_multi_commodity_cycle(self, generator, active_review_cycle):
        items = await generator.generate_for_cycle(active_review_cycle)
        assert len(items) > 0


# ---------------------------------------------------------------------------
# Item Status Management
# ---------------------------------------------------------------------------

class TestItemStatusManagement:
    """Test checklist item status transitions."""

    @pytest.mark.asyncio
    async def test_start_item(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        started = await generator.start_item(items[0].item_id)
        assert started.status == ChecklistItemStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_complete_item(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        await generator.start_item(items[0].item_id)
        completed = await generator.complete_item(
            items[0].item_id, completed_by="analyst@company.com",
        )
        assert completed.status == ChecklistItemStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.completed_by == "analyst@company.com"

    @pytest.mark.asyncio
    async def test_skip_item(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        skipped = await generator.skip_item(
            items[2].item_id, reason="Not applicable for this commodity",
        )
        assert skipped.status == ChecklistItemStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_skip_required_item_raises(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        required_item = [i for i in items if i.required][0]
        with pytest.raises(ValueError, match="required"):
            await generator.skip_item(required_item.item_id, reason="Test")

    @pytest.mark.asyncio
    async def test_block_item(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        blocked = await generator.block_item(
            items[0].item_id, reason="Waiting for supplier data",
        )
        assert blocked.status == ChecklistItemStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_unblock_item(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        await generator.block_item(items[0].item_id, reason="Blocked")
        unblocked = await generator.unblock_item(items[0].item_id)
        assert unblocked.status == ChecklistItemStatus.PENDING

    @pytest.mark.asyncio
    async def test_complete_pending_item_directly(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        completed = await generator.complete_item(
            items[0].item_id, completed_by="analyst@company.com",
        )
        assert completed.status == ChecklistItemStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_complete_already_completed_raises(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        await generator.complete_item(items[0].item_id, completed_by="user")
        with pytest.raises(ValueError, match="already completed"):
            await generator.complete_item(items[0].item_id, completed_by="user")


# ---------------------------------------------------------------------------
# Evidence Attachment
# ---------------------------------------------------------------------------

class TestEvidenceAttachment:
    """Test evidence attachment to checklist items."""

    @pytest.mark.asyncio
    async def test_attach_evidence(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        updated = await generator.attach_evidence(
            items[0].item_id,
            evidence_ref="s3://evidence/report_2026.pdf",
            evidence_type="document",
        )
        assert updated is not None

    @pytest.mark.asyncio
    async def test_attach_multiple_evidence(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        await generator.attach_evidence(
            items[0].item_id, evidence_ref="s3://evidence/doc1.pdf", evidence_type="document",
        )
        updated = await generator.attach_evidence(
            items[0].item_id, evidence_ref="s3://evidence/doc2.pdf", evidence_type="certificate",
        )
        assert updated is not None

    @pytest.mark.asyncio
    async def test_attach_evidence_to_nonexistent_item_raises(self, generator):
        with pytest.raises(ValueError, match="not found"):
            await generator.attach_evidence(
                "chk-nonexistent", evidence_ref="s3://evidence/doc.pdf",
                evidence_type="document",
            )


# ---------------------------------------------------------------------------
# Progress Calculation
# ---------------------------------------------------------------------------

class TestProgressCalculation:
    """Test checklist progress calculation."""

    @pytest.mark.asyncio
    async def test_progress_zero_when_none_completed(self, generator, sample_checklist_template):
        await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        progress = await generator.calculate_progress(cycle_id="cyc-001")
        assert progress.total_items > 0
        assert progress.completed_items == 0
        assert progress.completion_percentage == Decimal("0")

    @pytest.mark.asyncio
    async def test_progress_partial_completion(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        await generator.complete_item(items[0].item_id, completed_by="user")
        progress = await generator.calculate_progress(cycle_id="cyc-001")
        assert progress.completed_items == 1
        assert progress.completion_percentage > Decimal("0")

    @pytest.mark.asyncio
    async def test_progress_100_when_all_completed(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        for item in items:
            if item.required:
                await generator.complete_item(item.item_id, completed_by="user")
            else:
                await generator.skip_item(item.item_id, reason="Optional")
        progress = await generator.calculate_progress(cycle_id="cyc-001")
        assert progress.completion_percentage == Decimal("100")

    @pytest.mark.asyncio
    async def test_progress_by_phase(self, generator, sample_checklist_template):
        await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        progress = await generator.calculate_progress(
            cycle_id="cyc-001", phase=ReviewPhase.DATA_COLLECTION,
        )
        assert progress.total_items > 0

    @pytest.mark.asyncio
    async def test_progress_skipped_items_count_as_done(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        non_required = [i for i in items if not i.required]
        if non_required:
            await generator.skip_item(non_required[0].item_id, reason="Not applicable")
        progress = await generator.calculate_progress(cycle_id="cyc-001")
        assert progress.completed_items + progress.skipped_items >= 1


# ---------------------------------------------------------------------------
# Template Management
# ---------------------------------------------------------------------------

class TestTemplateManagement:
    """Test checklist template management."""

    @pytest.mark.asyncio
    async def test_register_template(self, generator, sample_checklist_template):
        result = await generator.register_template(sample_checklist_template)
        assert result is True

    @pytest.mark.asyncio
    async def test_get_template_by_id(self, generator, sample_checklist_template):
        await generator.register_template(sample_checklist_template)
        retrieved = await generator.get_template(sample_checklist_template.template_id)
        assert retrieved.template_id == sample_checklist_template.template_id

    @pytest.mark.asyncio
    async def test_list_templates_by_phase(self, generator, sample_checklist_template):
        await generator.register_template(sample_checklist_template)
        templates = await generator.list_templates(phase=ReviewPhase.DATA_COLLECTION)
        assert len(templates) >= 1

    @pytest.mark.asyncio
    async def test_list_templates_by_commodity(self, generator, sample_checklist_template):
        await generator.register_template(sample_checklist_template)
        templates = await generator.list_templates(commodity=EUDRCommodity.COFFEE)
        assert len(templates) >= 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_template_raises(self, generator):
        with pytest.raises(ValueError, match="not found"):
            await generator.get_template("tpl-nonexistent")

    @pytest.mark.asyncio
    async def test_register_duplicate_template_updates(self, generator, sample_checklist_template):
        await generator.register_template(sample_checklist_template)
        sample_checklist_template.name = "Updated Name"
        result = await generator.register_template(sample_checklist_template)
        assert result is True
        retrieved = await generator.get_template(sample_checklist_template.template_id)
        assert retrieved.name == "Updated Name"


# ---------------------------------------------------------------------------
# List and Filter Items
# ---------------------------------------------------------------------------

class TestListAndFilterItems:
    """Test item listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_items_by_cycle(self, generator, sample_checklist_template):
        await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        items = await generator.list_items(cycle_id="cyc-001")
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_list_items_by_status(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        await generator.complete_item(items[0].item_id, completed_by="user")
        pending = await generator.list_items(
            cycle_id="cyc-001", status=ChecklistItemStatus.PENDING,
        )
        completed = await generator.list_items(
            cycle_id="cyc-001", status=ChecklistItemStatus.COMPLETED,
        )
        assert len(pending) == 2
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_list_items_by_assignee(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        await generator.assign_item(items[0].item_id, assignee="analyst@company.com")
        assigned = await generator.list_items(
            cycle_id="cyc-001", assigned_to="analyst@company.com",
        )
        assert len(assigned) >= 1

    @pytest.mark.asyncio
    async def test_get_item_by_id(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        retrieved = await generator.get_item(items[0].item_id)
        assert retrieved.item_id == items[0].item_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_item_raises(self, generator):
        with pytest.raises(ValueError, match="not found"):
            await generator.get_item("chk-nonexistent")


# ---------------------------------------------------------------------------
# Assign Items
# ---------------------------------------------------------------------------

class TestAssignItems:
    """Test checklist item assignment."""

    @pytest.mark.asyncio
    async def test_assign_item_to_user(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        assigned = await generator.assign_item(
            items[0].item_id, assignee="analyst@company.com",
        )
        assert assigned.assigned_to == "analyst@company.com"

    @pytest.mark.asyncio
    async def test_reassign_item(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        await generator.assign_item(items[0].item_id, assignee="analyst@company.com")
        reassigned = await generator.assign_item(
            items[0].item_id, assignee="lead@company.com",
        )
        assert reassigned.assigned_to == "lead@company.com"

    @pytest.mark.asyncio
    async def test_bulk_assign_items(self, generator, sample_checklist_template):
        items = await generator.generate_from_template(
            cycle_id="cyc-001", template=sample_checklist_template,
        )
        item_ids = [item.item_id for item in items]
        results = await generator.bulk_assign_items(
            item_ids=item_ids, assignee="analyst@company.com",
        )
        assert len(results) == len(item_ids)
        for item in results:
            assert item.assigned_to == "analyst@company.com"
