# -*- coding: utf-8 -*-
"""
Recycled Content Tracking Workflow
========================================

4-phase workflow for tracking and verifying recycled content in batteries
per EU Battery Regulation 2023/1542, Article 8 and Annex XII. Implements
material inventory, recycled content calculation, target comparison against
regulatory minimums, and compliance documentation generation.

Phases:
    1. MaterialInventory     -- Catalogue all materials and their sources
    2. ContentCalculation    -- Calculate recycled content percentages
    3. TargetComparison      -- Compare against regulatory targets
    4. Documentation         -- Generate recycled content documentation

Regulatory references:
    - EU Regulation 2023/1542 Art. 8 (recycled content)
    - EU Regulation 2023/1542 Annex XII (recycled content calculation)
    - Art. 8(1): From 18 Aug 2031 - min recycled content by weight
    - Art. 8(2): From 18 Aug 2036 - increased recycled content targets

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the recycled content tracking workflow."""
    MATERIAL_INVENTORY = "material_inventory"
    CONTENT_CALCULATION = "content_calculation"
    TARGET_COMPARISON = "target_comparison"
    DOCUMENTATION = "documentation"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RecoveredMaterialType(str, Enum):
    """Critical raw materials tracked for recycled content per Art. 8."""
    COBALT = "cobalt"
    LITHIUM = "lithium"
    NICKEL = "nickel"
    LEAD = "lead"
    COPPER = "copper"
    MANGANESE = "manganese"
    GRAPHITE = "graphite"
    OTHER = "other"


class MaterialSourceType(str, Enum):
    """Source classification for material traceability."""
    PRIMARY_VIRGIN = "primary_virgin"
    PRE_CONSUMER_RECYCLED = "pre_consumer_recycled"
    POST_CONSUMER_RECYCLED = "post_consumer_recycled"
    RECOVERED_FROM_BATTERY = "recovered_from_battery"
    RECOVERED_FROM_WASTE = "recovered_from_waste"
    UNKNOWN = "unknown"


class ComplianceTargetPhase(str, Enum):
    """Regulatory phase for recycled content targets."""
    PHASE_1_2031 = "phase_1_2031"
    PHASE_2_2036 = "phase_2_2036"


# =============================================================================
# REGULATORY TARGETS (% by weight per Art. 8)
# =============================================================================


RECYCLED_CONTENT_TARGETS_2031: Dict[str, float] = {
    "cobalt": 16.0,
    "lithium": 6.0,
    "nickel": 6.0,
    "lead": 85.0,
}

RECYCLED_CONTENT_TARGETS_2036: Dict[str, float] = {
    "cobalt": 26.0,
    "lithium": 12.0,
    "nickel": 15.0,
    "lead": 85.0,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class MaterialEntry(BaseModel):
    """Individual material entry within the battery BOM."""
    entry_id: str = Field(default_factory=lambda: f"me-{_new_uuid()[:8]}")
    material_type: RecoveredMaterialType = Field(
        ..., description="Material type category"
    )
    material_name: str = Field(default="", description="Specific material name")
    total_mass_kg: float = Field(default=0.0, ge=0.0, description="Total mass in kg")
    recycled_mass_kg: float = Field(
        default=0.0, ge=0.0, description="Mass from recycled sources in kg"
    )
    source_type: MaterialSourceType = Field(
        default=MaterialSourceType.PRIMARY_VIRGIN
    )
    supplier_name: str = Field(default="", description="Material supplier")
    supplier_country: str = Field(default="", description="Supplier ISO 3166 code")
    certification_standard: str = Field(
        default="", description="e.g., RMI, IRMA, ISO 22095"
    )
    chain_of_custody_verified: bool = Field(default=False)
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class RecycledContentSummary(BaseModel):
    """Aggregated recycled content for a specific material type."""
    material_type: str = Field(..., description="Material type")
    total_mass_kg: float = Field(default=0.0, ge=0.0)
    recycled_mass_kg: float = Field(default=0.0, ge=0.0)
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_2031_pct: float = Field(default=0.0, ge=0.0)
    target_2036_pct: float = Field(default=0.0, ge=0.0)
    meets_2031_target: bool = Field(default=False)
    meets_2036_target: bool = Field(default=False)
    gap_to_2031_pct: float = Field(default=0.0)
    gap_to_2036_pct: float = Field(default=0.0)
    entry_count: int = Field(default=0, ge=0)


class RecycledContentInput(BaseModel):
    """Input data model for RecycledContentWorkflow."""
    battery_id: str = Field(default_factory=lambda: f"bat-{_new_uuid()[:8]}")
    battery_model: str = Field(default="", description="Battery model identifier")
    material_entries: List[MaterialEntry] = Field(default_factory=list)
    target_phase: ComplianceTargetPhase = Field(
        default=ComplianceTargetPhase.PHASE_1_2031,
        description="Regulatory target phase to evaluate against"
    )
    total_battery_mass_kg: float = Field(
        default=0.0, ge=0.0, description="Total battery mass in kg"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class RecycledContentResult(BaseModel):
    """Complete result from recycled content tracking workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="recycled_content_tracking")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    battery_id: str = Field(default="")
    material_summaries: List[RecycledContentSummary] = Field(default_factory=list)
    overall_recycled_content_pct: float = Field(default=0.0, ge=0.0)
    total_material_mass_kg: float = Field(default=0.0, ge=0.0)
    total_recycled_mass_kg: float = Field(default=0.0, ge=0.0)
    materials_meeting_targets: int = Field(default=0, ge=0)
    materials_below_targets: int = Field(default=0, ge=0)
    all_targets_met: bool = Field(default=False)
    target_phase: str = Field(default="phase_1_2031")
    verified_supply_chain_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    reporting_year: int = Field(default=2025)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RecycledContentWorkflow:
    """
    4-phase recycled content tracking workflow per EU Battery Regulation.

    Implements end-to-end tracking of recycled content in batteries
    per EU Regulation 2023/1542 Art. 8 and Annex XII. Catalogues
    materials, calculates recycled content percentages, compares against
    regulatory targets, and generates compliance documentation.

    Zero-hallucination: all percentages computed with deterministic
    arithmetic (recycled_mass / total_mass * 100). No LLM in
    numeric calculation paths.

    Example:
        >>> wf = RecycledContentWorkflow()
        >>> inp = RecycledContentInput(material_entries=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.all_targets_met in (True, False)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize RecycledContentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._entries: List[MaterialEntry] = []
        self._summaries: List[RecycledContentSummary] = []
        self._all_targets_met: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.MATERIAL_INVENTORY.value, "description": "Catalogue all materials and their sources"},
            {"name": WorkflowPhase.CONTENT_CALCULATION.value, "description": "Calculate recycled content percentages"},
            {"name": WorkflowPhase.TARGET_COMPARISON.value, "description": "Compare against regulatory targets"},
            {"name": WorkflowPhase.DOCUMENTATION.value, "description": "Generate recycled content documentation"},
        ]

    def validate_inputs(self, input_data: RecycledContentInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.material_entries:
            issues.append("No material entries provided")
        for entry in input_data.material_entries:
            if entry.recycled_mass_kg > entry.total_mass_kg:
                issues.append(
                    f"Entry {entry.entry_id}: recycled mass exceeds total mass"
                )
            if entry.total_mass_kg < 0:
                issues.append(f"Entry {entry.entry_id}: negative total mass")
        return issues

    async def execute(
        self,
        input_data: Optional[RecycledContentInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RecycledContentResult:
        """
        Execute the 4-phase recycled content tracking workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            RecycledContentResult with material summaries and compliance status.
        """
        if input_data is None:
            input_data = RecycledContentInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting recycled content workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_material_inventory(input_data))
            phases_done += 1
            phase_results.append(await self._phase_content_calculation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_target_comparison(input_data))
            phases_done += 1
            phase_results.append(await self._phase_documentation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Recycled content workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        total_mass = sum(s.total_mass_kg for s in self._summaries)
        recycled_mass = sum(s.recycled_mass_kg for s in self._summaries)
        overall_pct = round(
            (recycled_mass / total_mass * 100) if total_mass > 0 else 0.0, 2
        )
        meeting = sum(1 for s in self._summaries if s.meets_2031_target)
        below = len(self._summaries) - meeting

        # Supply chain verification percentage
        verified_count = sum(1 for e in self._entries if e.chain_of_custody_verified)
        verified_pct = round(
            (verified_count / len(self._entries) * 100) if self._entries else 0.0, 1
        )

        result = RecycledContentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            battery_id=input_data.battery_id,
            material_summaries=self._summaries,
            overall_recycled_content_pct=overall_pct,
            total_material_mass_kg=round(total_mass, 2),
            total_recycled_mass_kg=round(recycled_mass, 2),
            materials_meeting_targets=meeting,
            materials_below_targets=below,
            all_targets_met=self._all_targets_met,
            target_phase=input_data.target_phase.value,
            verified_supply_chain_pct=verified_pct,
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Recycled content %s completed in %.2fs: %.1f%% overall, targets_met=%s",
            self.workflow_id, elapsed, overall_pct, self._all_targets_met,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Material Inventory
    # -------------------------------------------------------------------------

    async def _phase_material_inventory(
        self, input_data: RecycledContentInput,
    ) -> PhaseResult:
        """Catalogue all materials and their source classifications."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._entries = list(input_data.material_entries)

        # Calculate recycled content per entry where not provided
        for entry in self._entries:
            if entry.recycled_content_pct <= 0 and entry.total_mass_kg > 0:
                entry.recycled_content_pct = round(
                    entry.recycled_mass_kg / entry.total_mass_kg * 100, 2
                )

        # Type distribution
        type_counts: Dict[str, int] = {}
        type_mass: Dict[str, float] = {}
        source_counts: Dict[str, int] = {}
        for entry in self._entries:
            mt = entry.material_type.value
            type_counts[mt] = type_counts.get(mt, 0) + 1
            type_mass[mt] = type_mass.get(mt, 0.0) + entry.total_mass_kg
            st = entry.source_type.value
            source_counts[st] = source_counts.get(st, 0) + 1

        outputs["entries_catalogued"] = len(self._entries)
        outputs["material_type_distribution"] = type_counts
        outputs["material_mass_distribution_kg"] = {
            k: round(v, 2) for k, v in type_mass.items()
        }
        outputs["source_type_distribution"] = source_counts
        outputs["total_mass_kg"] = round(
            sum(e.total_mass_kg for e in self._entries), 2
        )
        outputs["unique_suppliers"] = len(set(
            e.supplier_name for e in self._entries if e.supplier_name
        ))

        # Warnings
        unknown_source = [
            e for e in self._entries if e.source_type == MaterialSourceType.UNKNOWN
        ]
        if unknown_source:
            warnings.append(
                f"{len(unknown_source)} entries have unknown source type"
            )

        unverified = [e for e in self._entries if not e.chain_of_custody_verified]
        if unverified:
            warnings.append(
                f"{len(unverified)} entries lack chain-of-custody verification"
            )

        if not self._entries:
            warnings.append("No material entries provided; analysis will be empty")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 MaterialInventory: %d entries, %d material types",
            len(self._entries), len(type_counts),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.MATERIAL_INVENTORY.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Content Calculation
    # -------------------------------------------------------------------------

    async def _phase_content_calculation(
        self, input_data: RecycledContentInput,
    ) -> PhaseResult:
        """Calculate recycled content percentages per material type."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._summaries = []

        # Group entries by material type
        groups: Dict[str, List[MaterialEntry]] = {}
        for entry in self._entries:
            mt = entry.material_type.value
            groups.setdefault(mt, []).append(entry)

        for mat_type, entries in sorted(groups.items()):
            total_mass = sum(e.total_mass_kg for e in entries)
            recycled_mass = sum(e.recycled_mass_kg for e in entries)
            pct = round(
                (recycled_mass / total_mass * 100) if total_mass > 0 else 0.0, 2
            )

            target_2031 = RECYCLED_CONTENT_TARGETS_2031.get(mat_type, 0.0)
            target_2036 = RECYCLED_CONTENT_TARGETS_2036.get(mat_type, 0.0)
            meets_2031 = pct >= target_2031 if target_2031 > 0 else True
            meets_2036 = pct >= target_2036 if target_2036 > 0 else True
            gap_2031 = round(max(0.0, target_2031 - pct), 2)
            gap_2036 = round(max(0.0, target_2036 - pct), 2)

            self._summaries.append(RecycledContentSummary(
                material_type=mat_type,
                total_mass_kg=round(total_mass, 4),
                recycled_mass_kg=round(recycled_mass, 4),
                recycled_content_pct=pct,
                target_2031_pct=target_2031,
                target_2036_pct=target_2036,
                meets_2031_target=meets_2031,
                meets_2036_target=meets_2036,
                gap_to_2031_pct=gap_2031,
                gap_to_2036_pct=gap_2036,
                entry_count=len(entries),
            ))

        overall_total = sum(s.total_mass_kg for s in self._summaries)
        overall_recycled = sum(s.recycled_mass_kg for s in self._summaries)

        outputs["material_types_analyzed"] = len(self._summaries)
        outputs["overall_recycled_content_pct"] = round(
            (overall_recycled / overall_total * 100) if overall_total > 0 else 0.0, 2
        )
        outputs["per_material_pct"] = {
            s.material_type: s.recycled_content_pct for s in self._summaries
        }

        zero_recycled = [s for s in self._summaries if s.recycled_mass_kg <= 0]
        if zero_recycled:
            warnings.append(
                f"{len(zero_recycled)} material types have zero recycled content"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 ContentCalculation: %d types, overall %.1f%%",
            len(self._summaries),
            outputs["overall_recycled_content_pct"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.CONTENT_CALCULATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Target Comparison
    # -------------------------------------------------------------------------

    async def _phase_target_comparison(
        self, input_data: RecycledContentInput,
    ) -> PhaseResult:
        """Compare recycled content against regulatory targets."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        use_2036 = input_data.target_phase == ComplianceTargetPhase.PHASE_2_2036
        targets = RECYCLED_CONTENT_TARGETS_2036 if use_2036 else RECYCLED_CONTENT_TARGETS_2031

        regulated_materials = [
            s for s in self._summaries if s.material_type in targets
        ]
        unregulated_materials = [
            s for s in self._summaries if s.material_type not in targets
        ]

        meeting_target = []
        below_target = []
        for summary in regulated_materials:
            meets = summary.meets_2036_target if use_2036 else summary.meets_2031_target
            if meets:
                meeting_target.append(summary.material_type)
            else:
                below_target.append(summary.material_type)
                gap = summary.gap_to_2036_pct if use_2036 else summary.gap_to_2031_pct
                target_val = summary.target_2036_pct if use_2036 else summary.target_2031_pct
                warnings.append(
                    f"{summary.material_type}: {summary.recycled_content_pct}% vs "
                    f"target {target_val}% (gap: {gap}%)"
                )

        self._all_targets_met = len(below_target) == 0 and len(regulated_materials) > 0

        # Check for missing regulated materials in BOM
        reported_types = set(s.material_type for s in self._summaries)
        missing_regulated = set(targets.keys()) - reported_types
        if missing_regulated:
            warnings.append(
                f"Missing data for regulated materials: {', '.join(sorted(missing_regulated))}"
            )

        outputs["target_phase"] = input_data.target_phase.value
        outputs["regulated_materials_checked"] = len(regulated_materials)
        outputs["unregulated_materials"] = len(unregulated_materials)
        outputs["materials_meeting_target"] = meeting_target
        outputs["materials_below_target"] = below_target
        outputs["all_targets_met"] = self._all_targets_met
        outputs["targets_applied"] = targets
        outputs["missing_regulated_materials"] = sorted(missing_regulated)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 TargetComparison: %d meeting, %d below, all_met=%s",
            len(meeting_target), len(below_target), self._all_targets_met,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_COMPARISON.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Documentation
    # -------------------------------------------------------------------------

    async def _phase_documentation(
        self, input_data: RecycledContentInput,
    ) -> PhaseResult:
        """Generate recycled content compliance documentation."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_mass = sum(s.total_mass_kg for s in self._summaries)
        recycled_mass = sum(s.recycled_mass_kg for s in self._summaries)
        overall_pct = round(
            (recycled_mass / total_mass * 100) if total_mass > 0 else 0.0, 2
        )

        documentation = {
            "document_id": f"rcd-{_new_uuid()[:8]}",
            "battery_id": input_data.battery_id,
            "battery_model": input_data.battery_model,
            "regulation_reference": "EU Regulation 2023/1542 Art. 8",
            "methodology": "EU Battery Regulation Annex XII",
            "reporting_year": input_data.reporting_year,
            "target_phase": input_data.target_phase.value,
            "overall_recycled_content_pct": overall_pct,
            "total_material_mass_kg": round(total_mass, 2),
            "total_recycled_mass_kg": round(recycled_mass, 2),
            "material_breakdown": [
                {
                    "material_type": s.material_type,
                    "total_mass_kg": s.total_mass_kg,
                    "recycled_mass_kg": s.recycled_mass_kg,
                    "recycled_content_pct": s.recycled_content_pct,
                    "target_pct": s.target_2031_pct,
                    "meets_target": s.meets_2031_target,
                }
                for s in self._summaries
            ],
            "all_targets_met": self._all_targets_met,
            "issued_at": _utcnow().isoformat(),
        }

        # Completeness assessment
        completeness_checks = {
            "battery_id": bool(input_data.battery_id),
            "material_entries": len(self._entries) > 0,
            "recycled_data": recycled_mass > 0,
            "target_comparison": len(self._summaries) > 0,
        }
        passed = sum(1 for v in completeness_checks.values() if v)
        completeness_pct = round(passed / len(completeness_checks) * 100, 1)

        if completeness_pct < 100:
            failed = [k for k, v in completeness_checks.items() if not v]
            warnings.append(
                f"Documentation incomplete ({completeness_pct}%): "
                f"missing {', '.join(failed)}"
            )

        outputs["document_id"] = documentation["document_id"]
        outputs["documentation"] = documentation
        outputs["completeness_pct"] = completeness_pct
        outputs["completeness_checks"] = completeness_checks
        outputs["all_targets_met"] = self._all_targets_met

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 Documentation: %s issued, completeness %.1f%%",
            documentation["document_id"], completeness_pct,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.DOCUMENTATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: RecycledContentResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
