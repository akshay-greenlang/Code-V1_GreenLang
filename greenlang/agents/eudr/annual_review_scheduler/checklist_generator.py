# -*- coding: utf-8 -*-
"""
Checklist Generator Engine - AGENT-EUDR-034

Generates commodity-specific review checklists from EUDR article
requirements and organizational templates. Supports template
customization, item prioritization, and completion tracking.

Zero-Hallucination:
    - All completion percentages are deterministic Decimal arithmetic
    - Template loading is from predefined YAML/dict structures
    - No LLM involvement in checklist generation logic

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (GL-EUDR-ARS-034)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import AnnualReviewSchedulerConfig, get_config
from .models import (
    AGENT_ID,
    EUDR_COMMODITIES,
    ChecklistItem,
    ChecklistItemStatus,
    ChecklistPriority,
    ChecklistRecord,
    ChecklistTemplate,
    EUDRCommodity,
    ReviewPhase,
    REVIEW_PHASES_ORDER,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checklist Templates
# ---------------------------------------------------------------------------

_BASE_TEMPLATE: List[Dict[str, Any]] = [
    {
        "section": "due_diligence",
        "article_reference": "Article 4",
        "title": "Verify due diligence system is operational",
        "description": "Confirm the operator's due diligence system meets EUDR requirements.",
        "priority": "mandatory",
        "is_mandatory": True,
        "evidence_required": True,
    },
    {
        "section": "information_collection",
        "article_reference": "Article 9",
        "title": "Validate information collection completeness",
        "description": "Ensure all required information per Article 9 is collected.",
        "priority": "mandatory",
        "is_mandatory": True,
        "evidence_required": True,
    },
    {
        "section": "risk_assessment",
        "article_reference": "Article 10",
        "title": "Review risk assessment methodology",
        "description": "Verify risk assessment criteria and methodology are current.",
        "priority": "mandatory",
        "is_mandatory": True,
        "evidence_required": True,
    },
    {
        "section": "risk_assessment",
        "article_reference": "Article 10",
        "title": "Validate country risk evaluation",
        "description": "Confirm country risk classifications are up to date.",
        "priority": "high",
        "is_mandatory": True,
        "evidence_required": True,
    },
    {
        "section": "risk_mitigation",
        "article_reference": "Article 11",
        "title": "Review risk mitigation measures",
        "description": "Verify risk mitigation measures are adequate and implemented.",
        "priority": "mandatory",
        "is_mandatory": True,
        "evidence_required": True,
    },
    {
        "section": "geolocation",
        "article_reference": "Article 9",
        "title": "Verify geolocation data accuracy",
        "description": "Confirm GPS coordinates for all production plots are valid.",
        "priority": "mandatory",
        "is_mandatory": True,
        "evidence_required": True,
    },
    {
        "section": "deforestation",
        "article_reference": "Article 3",
        "title": "Confirm deforestation-free status",
        "description": "Verify no deforestation since December 31, 2020 cutoff date.",
        "priority": "mandatory",
        "is_mandatory": True,
        "evidence_required": True,
    },
    {
        "section": "legality",
        "article_reference": "Article 3",
        "title": "Verify legal compliance of production",
        "description": "Confirm production complies with relevant laws of the country.",
        "priority": "mandatory",
        "is_mandatory": True,
        "evidence_required": True,
    },
    {
        "section": "supply_chain",
        "article_reference": "Article 8",
        "title": "Review supply chain mapping completeness",
        "description": "Ensure all tiers of the supply chain are adequately mapped.",
        "priority": "high",
        "is_mandatory": False,
        "evidence_required": True,
    },
    {
        "section": "documentation",
        "article_reference": "Article 12",
        "title": "Verify documentation completeness",
        "description": "Confirm all required documents are current and accessible.",
        "priority": "high",
        "is_mandatory": False,
        "evidence_required": True,
    },
    {
        "section": "monitoring",
        "article_reference": "Article 14",
        "title": "Review monitoring and reporting systems",
        "description": "Verify continuous monitoring capabilities are operational.",
        "priority": "medium",
        "is_mandatory": False,
        "evidence_required": False,
    },
    {
        "section": "record_keeping",
        "article_reference": "Article 31",
        "title": "Confirm 5-year record retention compliance",
        "description": "Verify records are retained for at least 5 years.",
        "priority": "medium",
        "is_mandatory": False,
        "evidence_required": False,
    },
]

# Commodity-specific extensions
_COMMODITY_EXTENSIONS: Dict[str, List[Dict[str, Any]]] = {
    "cattle": [
        {
            "section": "cattle_specific",
            "article_reference": "Annex I",
            "title": "Verify cattle traceability system",
            "description": "Confirm individual animal or herd traceability.",
            "priority": "mandatory",
            "is_mandatory": True,
            "evidence_required": True,
        },
        {
            "section": "cattle_specific",
            "article_reference": "Article 10",
            "title": "Review pasture management records",
            "description": "Verify pasture locations and stocking density records.",
            "priority": "high",
            "is_mandatory": False,
            "evidence_required": True,
        },
    ],
    "cocoa": [
        {
            "section": "cocoa_specific",
            "article_reference": "Annex I",
            "title": "Verify cocoa bean origin traceability",
            "description": "Confirm farm-to-first-purchase traceability.",
            "priority": "mandatory",
            "is_mandatory": True,
            "evidence_required": True,
        },
        {
            "section": "cocoa_specific",
            "article_reference": "Article 10",
            "title": "Review smallholder farmer mapping",
            "description": "Verify smallholder polygon mapping is complete.",
            "priority": "high",
            "is_mandatory": False,
            "evidence_required": True,
        },
    ],
    "coffee": [
        {
            "section": "coffee_specific",
            "article_reference": "Annex I",
            "title": "Verify coffee lot traceability",
            "description": "Confirm lot-level traceability from farm to export.",
            "priority": "mandatory",
            "is_mandatory": True,
            "evidence_required": True,
        },
    ],
    "oil_palm": [
        {
            "section": "oil_palm_specific",
            "article_reference": "Annex I",
            "title": "Verify palm oil mill traceability",
            "description": "Confirm traceability from mill to plantation.",
            "priority": "mandatory",
            "is_mandatory": True,
            "evidence_required": True,
        },
        {
            "section": "oil_palm_specific",
            "article_reference": "Article 10",
            "title": "Review peatland assessment",
            "description": "Verify peatland identification and protection.",
            "priority": "high",
            "is_mandatory": True,
            "evidence_required": True,
        },
    ],
    "rubber": [
        {
            "section": "rubber_specific",
            "article_reference": "Annex I",
            "title": "Verify rubber plantation mapping",
            "description": "Confirm plantation boundary mapping completeness.",
            "priority": "mandatory",
            "is_mandatory": True,
            "evidence_required": True,
        },
    ],
    "soya": [
        {
            "section": "soya_specific",
            "article_reference": "Annex I",
            "title": "Verify soya origin traceability",
            "description": "Confirm field-level traceability for soya production.",
            "priority": "mandatory",
            "is_mandatory": True,
            "evidence_required": True,
        },
    ],
    "wood": [
        {
            "section": "wood_specific",
            "article_reference": "Annex I",
            "title": "Verify timber origin and legality",
            "description": "Confirm forest concession legality and harvesting permits.",
            "priority": "mandatory",
            "is_mandatory": True,
            "evidence_required": True,
        },
        {
            "section": "wood_specific",
            "article_reference": "Article 10",
            "title": "Review forest management certification",
            "description": "Verify FSC/PEFC or equivalent certification status.",
            "priority": "high",
            "is_mandatory": False,
            "evidence_required": True,
        },
    ],
}


class ChecklistGenerator:
    """Commodity-specific review checklist generator.

    Generates review checklists from EUDR article requirements and
    commodity-specific templates. Tracks item completion and calculates
    overall checklist progress.

    Example:
        >>> generator = ChecklistGenerator()
        >>> checklist = await generator.generate_checklist(
        ...     operator_id="OP-001", commodity="coffee"
        ... )
        >>> assert checklist.total_items > 0
    """

    def __init__(
        self,
        config: Optional[AnnualReviewSchedulerConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize ChecklistGenerator engine."""
        self.config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._checklists: Dict[str, ChecklistRecord] = {}
        self._items: Dict[str, ChecklistItem] = {}
        self._templates: Dict[str, ChecklistTemplate] = {}
        logger.info("ChecklistGenerator engine initialized")

    async def generate_checklist(
        self,
        operator_id: str,
        commodity: str = "general",
        cycle_id: str = "",
        custom_items: Optional[List[Dict[str, Any]]] = None,
    ) -> ChecklistRecord:
        """Generate a review checklist for a specific commodity.

        Args:
            operator_id: Operator identifier.
            commodity: Commodity type (or 'general').
            cycle_id: Associated review cycle ID.
            custom_items: Additional custom checklist items.

        Returns:
            ChecklistRecord with generated items.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        checklist_id = str(uuid.uuid4())

        # Build items from template
        items = self._build_items_from_template(commodity)

        # Add custom items
        if custom_items:
            for custom in custom_items:
                item = ChecklistItem(
                    item_id=str(uuid.uuid4()),
                    section=custom.get("section", "custom"),
                    article_reference=custom.get("article_reference", ""),
                    title=custom.get("title", "Custom item"),
                    description=custom.get("description", ""),
                    status=ChecklistItemStatus.PENDING,
                    priority=ChecklistPriority(custom.get("priority", "medium")),
                    is_mandatory=custom.get("is_mandatory", False),
                    evidence_required=custom.get("evidence_required", False),
                )
                items.append(item)

        # Enforce max items
        items = items[: self.config.checklist_max_items]

        mandatory_count = sum(1 for i in items if i.is_mandatory)

        record = ChecklistRecord(
            checklist_id=checklist_id,
            operator_id=operator_id,
            cycle_id=cycle_id,
            commodity=commodity,
            template_version=self.config.checklist_template_version,
            items=items,
            total_items=len(items),
            completed_items=0,
            mandatory_items=mandatory_count,
            mandatory_completed=0,
            completion_percent=Decimal("0"),
            mandatory_completion_percent=Decimal("0"),
            generated_at=now,
        )

        # Provenance
        prov_data = {
            "checklist_id": checklist_id,
            "operator_id": operator_id,
            "commodity": commodity,
            "total_items": len(items),
            "generated_at": now.isoformat(),
        }
        record.provenance_hash = self._provenance.compute_hash(prov_data)
        self._provenance.record(
            "checklist", "generate", checklist_id, AGENT_ID,
            metadata={"commodity": commodity, "items": len(items)},
        )

        self._checklists[checklist_id] = record

        elapsed = time.monotonic() - start_time
        m.observe_checklist_generation_duration(elapsed)
        m.record_checklist_generated(commodity)

        logger.info(
            "Checklist %s generated for operator %s commodity=%s (%d items, %d mandatory)",
            checklist_id, operator_id, commodity, len(items), mandatory_count,
        )
        return record

    async def load_template(
        self,
        commodity: str = "general",
    ) -> List[Dict[str, Any]]:
        """Load a checklist template for a commodity.

        Args:
            commodity: Commodity type.

        Returns:
            Template item definitions.
        """
        items = list(_BASE_TEMPLATE)
        if commodity in _COMMODITY_EXTENSIONS:
            items.extend(_COMMODITY_EXTENSIONS[commodity])
        return items

    async def customize_for_commodity(
        self,
        checklist_id: str,
        commodity: str,
    ) -> ChecklistRecord:
        """Add commodity-specific items to an existing checklist.

        Args:
            checklist_id: Checklist identifier.
            commodity: Commodity to add items for.

        Returns:
            Updated ChecklistRecord.

        Raises:
            ValueError: If checklist not found.
        """
        record = self._get_checklist(checklist_id)

        if commodity not in _COMMODITY_EXTENSIONS:
            logger.info("No commodity-specific template for %s", commodity)
            return record

        extensions = _COMMODITY_EXTENSIONS[commodity]
        for ext in extensions:
            item = ChecklistItem(
                item_id=str(uuid.uuid4()),
                section=ext["section"],
                article_reference=ext["article_reference"],
                title=ext["title"],
                description=ext["description"],
                status=ChecklistItemStatus.PENDING,
                priority=ChecklistPriority(ext["priority"]),
                is_mandatory=ext["is_mandatory"],
                evidence_required=ext["evidence_required"],
            )
            record.items.append(item)

        record.total_items = len(record.items)
        record.mandatory_items = sum(1 for i in record.items if i.is_mandatory)

        logger.info(
            "Added %d commodity items for %s to checklist %s",
            len(extensions), commodity, checklist_id,
        )
        return record

    async def track_completion(
        self,
        checklist_id: str,
        item_id: str,
        new_status: ChecklistItemStatus,
        notes: str = "",
    ) -> ChecklistRecord:
        """Update the status of a checklist item and recalculate progress.

        Args:
            checklist_id: Checklist identifier.
            item_id: Item identifier.
            new_status: New status for the item.
            notes: Optional reviewer notes.

        Returns:
            Updated ChecklistRecord.

        Raises:
            ValueError: If checklist or item not found.
        """
        record = self._get_checklist(checklist_id)

        item_found = False
        for item in record.items:
            if item.item_id == item_id:
                item.status = new_status
                if notes:
                    item.notes = notes
                item_found = True
                break

        if not item_found:
            raise ValueError(f"Item {item_id} not found in checklist {checklist_id}")

        # Recalculate completion
        completed_statuses = {ChecklistItemStatus.COMPLETED, ChecklistItemStatus.SKIPPED}
        record.completed_items = sum(
            1 for i in record.items if i.status in completed_statuses
        )
        record.mandatory_completed = sum(
            1 for i in record.items
            if i.is_mandatory and i.status in completed_statuses
        )

        if record.total_items > 0:
            record.completion_percent = (
                Decimal(str(record.completed_items))
                / Decimal(str(record.total_items))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if record.mandatory_items > 0:
            record.mandatory_completion_percent = (
                Decimal(str(record.mandatory_completed))
                / Decimal(str(record.mandatory_items))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if new_status == ChecklistItemStatus.COMPLETED:
            m.record_checklist_item_completed()
        m.set_checklist_completion(float(record.completion_percent))

        logger.info(
            "Checklist %s item %s -> %s (%.1f%% complete, mandatory %.1f%%)",
            checklist_id, item_id, new_status.value,
            record.completion_percent, record.mandatory_completion_percent,
        )
        return record

    async def get_checklist(self, checklist_id: str) -> Optional[ChecklistRecord]:
        """Get a specific checklist by ID."""
        return self._checklists.get(checklist_id)

    async def list_checklists(
        self,
        operator_id: Optional[str] = None,
        commodity: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ChecklistRecord]:
        """List checklists with optional filters."""
        results = list(self._checklists.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if commodity:
            results = [r for r in results if r.commodity == commodity]
        results.sort(key=lambda r: r.generated_at, reverse=True)
        return results[offset: offset + limit]

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "ChecklistGenerator",
            "status": "healthy",
            "total_checklists": len(self._checklists),
            "supported_commodities": EUDR_COMMODITIES,
        }

    # -- Engine-test API (ChecklistItem-based) --

    async def generate_from_template(
        self,
        cycle_id: str,
        template: ChecklistTemplate,
    ) -> List[ChecklistItem]:
        """Generate checklist items from a ChecklistTemplate."""
        items: List[ChecklistItem] = []
        for tmpl_item in template.items:
            item_id = f"chk-{uuid.uuid4()}"
            item = ChecklistItem(
                item_id=item_id,
                cycle_id=cycle_id,
                phase=tmpl_item.phase or template.phase,
                title=tmpl_item.title,
                description=tmpl_item.description,
                status=ChecklistItemStatus.PENDING,
                priority=tmpl_item.priority,
                required=tmpl_item.required,
                evidence_required=tmpl_item.evidence_required,
            )
            items.append(item)
            self._items[item_id] = item
        return items

    async def generate_for_cycle(self, cycle) -> List[ChecklistItem]:
        """Generate checklist items for all phases of a review cycle."""
        items: List[ChecklistItem] = []
        for phase in REVIEW_PHASES_ORDER:
            for i in range(2):  # 2 items per phase minimum
                item_id = f"chk-{uuid.uuid4()}"
                item = ChecklistItem(
                    item_id=item_id,
                    cycle_id=cycle.cycle_id,
                    phase=phase,
                    title=f"{phase.value} task {i + 1}",
                    description=f"Standard {phase.value} checklist item",
                    status=ChecklistItemStatus.PENDING,
                    priority=i,
                    required=(i == 0),
                )
                items.append(item)
                self._items[item_id] = item
        return items

    async def register_template(self, template: ChecklistTemplate) -> bool:
        """Register a checklist template."""
        self._templates[template.template_id] = template
        return True

    async def get_template(self, template_id: str) -> ChecklistTemplate:
        """Get a template by ID."""
        tpl = self._templates.get(template_id)
        if tpl is None:
            raise ValueError(f"Template {template_id} not found")
        return tpl

    async def list_templates(
        self,
        phase: Optional[ReviewPhase] = None,
        commodity: Optional[EUDRCommodity] = None,
    ) -> List[ChecklistTemplate]:
        """List registered templates with optional filters."""
        results = list(self._templates.values())
        if phase is not None:
            results = [t for t in results if t.phase == phase]
        if commodity is not None:
            results = [t for t in results if t.commodity == commodity]
        return results

    async def get_item(self, item_id: str) -> ChecklistItem:
        """Get a checklist item by ID."""
        item = self._items.get(item_id)
        if item is None:
            raise ValueError(f"Checklist item {item_id} not found")
        return item

    async def list_items(
        self,
        cycle_id: Optional[str] = None,
        status: Optional[ChecklistItemStatus] = None,
        assigned_to: Optional[str] = None,
    ) -> List[ChecklistItem]:
        """List checklist items with optional filters."""
        results = list(self._items.values())
        if cycle_id:
            results = [i for i in results if i.cycle_id == cycle_id]
        if status is not None:
            results = [i for i in results if i.status == status]
        if assigned_to:
            results = [i for i in results if i.assigned_to == assigned_to]
        return results

    async def start_item(self, item_id: str) -> ChecklistItem:
        """Start a checklist item."""
        item = await self.get_item(item_id)
        item.status = ChecklistItemStatus.IN_PROGRESS
        return item

    async def complete_item(self, item_id: str, completed_by: str = "") -> ChecklistItem:
        """Complete a checklist item."""
        item = await self.get_item(item_id)
        if item.status == ChecklistItemStatus.COMPLETED:
            raise ValueError(f"Item {item_id} is already completed")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        item.status = ChecklistItemStatus.COMPLETED
        item.completed_at = now
        item.completed_by = completed_by
        m.record_checklist_item_completed()
        return item

    async def skip_item(self, item_id: str, reason: str = "") -> ChecklistItem:
        """Skip a checklist item."""
        item = await self.get_item(item_id)
        if item.required:
            raise ValueError(f"Cannot skip required item {item_id}")
        item.status = ChecklistItemStatus.SKIPPED
        if reason:
            item.notes = reason
        return item

    async def block_item(self, item_id: str, reason: str = "") -> ChecklistItem:
        """Block a checklist item."""
        item = await self.get_item(item_id)
        item.status = ChecklistItemStatus.BLOCKED
        if reason:
            item.notes = reason
        return item

    async def unblock_item(self, item_id: str) -> ChecklistItem:
        """Unblock a checklist item."""
        item = await self.get_item(item_id)
        item.status = ChecklistItemStatus.PENDING
        return item

    async def assign_item(self, item_id: str, assignee: str = "") -> ChecklistItem:
        """Assign a checklist item to a user."""
        item = await self.get_item(item_id)
        item.assigned_to = assignee
        return item

    async def bulk_assign_items(
        self, item_ids: List[str], assignee: str = "",
    ) -> List[ChecklistItem]:
        """Bulk-assign items to a user."""
        results: List[ChecklistItem] = []
        for item_id in item_ids:
            item = await self.assign_item(item_id, assignee=assignee)
            results.append(item)
        return results

    async def attach_evidence(
        self, item_id: str, evidence_ref: str = "", evidence_type: str = "",
    ) -> ChecklistItem:
        """Attach evidence to a checklist item."""
        item = await self.get_item(item_id)
        # Store evidence as note (lightweight)
        item.notes = (item.notes or "") + f"\n[Evidence: {evidence_type}] {evidence_ref}"
        return item

    async def calculate_progress(
        self,
        cycle_id: str,
        phase: Optional[ReviewPhase] = None,
    ) -> "ProgressResult":
        """Calculate progress for a cycle."""
        items = await self.list_items(cycle_id=cycle_id)
        if phase is not None:
            items = [i for i in items if i.phase == phase]

        total = len(items)
        completed = sum(1 for i in items if i.status == ChecklistItemStatus.COMPLETED)
        skipped = sum(1 for i in items if i.status == ChecklistItemStatus.SKIPPED)

        if total > 0:
            pct = (
                Decimal(str(completed + skipped))
                / Decimal(str(total))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            pct = Decimal("0")

        return _ProgressResult(
            total_items=total,
            completed_items=completed,
            skipped_items=skipped,
            completion_percentage=pct,
        )

    # -- Private helpers --

    def _get_checklist(self, checklist_id: str) -> ChecklistRecord:
        """Retrieve a checklist or raise ValueError."""
        record = self._checklists.get(checklist_id)
        if record is None:
            raise ValueError(f"Checklist {checklist_id} not found")
        return record

    def _build_items_from_template(
        self, commodity: str,
    ) -> List[ChecklistItem]:
        """Build checklist items from base and commodity templates."""
        items: List[ChecklistItem] = []

        # Base template items
        for tmpl in _BASE_TEMPLATE:
            item = ChecklistItem(
                item_id=str(uuid.uuid4()),
                section=tmpl["section"],
                article_reference=tmpl["article_reference"],
                title=tmpl["title"],
                description=tmpl["description"],
                status=ChecklistItemStatus.PENDING,
                priority=ChecklistPriority(tmpl["priority"]),
                is_mandatory=tmpl["is_mandatory"],
                evidence_required=tmpl["evidence_required"],
            )
            items.append(item)

        # Commodity extensions
        if (
            self.config.checklist_commodity_templates_enabled
            and commodity in _COMMODITY_EXTENSIONS
        ):
            for ext in _COMMODITY_EXTENSIONS[commodity]:
                item = ChecklistItem(
                    item_id=str(uuid.uuid4()),
                    section=ext["section"],
                    article_reference=ext["article_reference"],
                    title=ext["title"],
                    description=ext["description"],
                    status=ChecklistItemStatus.PENDING,
                    priority=ChecklistPriority(ext["priority"]),
                    is_mandatory=ext["is_mandatory"],
                    evidence_required=ext["evidence_required"],
                )
                items.append(item)

        return items


class _ProgressResult:
    """Lightweight progress result for calculate_progress."""
    __slots__ = ("total_items", "completed_items", "skipped_items", "completion_percentage")

    def __init__(
        self,
        total_items: int = 0,
        completed_items: int = 0,
        skipped_items: int = 0,
        completion_percentage: Decimal = Decimal("0"),
    ) -> None:
        self.total_items = total_items
        self.completed_items = completed_items
        self.skipped_items = skipped_items
        self.completion_percentage = completion_percentage
