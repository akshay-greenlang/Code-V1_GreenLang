# -*- coding: utf-8 -*-
"""
Packaging Compliance Workflow
=================================

4-phase workflow for PPWR (Packaging and Packaging Waste Regulation)
compliance assessment within PACK-014 CSRD Retail Pack.

Phases:
    1. PackagingInventory         -- Catalog all packaging by material and type
    2. RecycledContentAssessment  -- Check recycled content vs PPWR targets
    3. EPRCompliance              -- Calculate EPR fees, check eco-modulation grades
    4. LabelingAudit              -- Verify labeling requirements met

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class PackagingMaterial(str, Enum):
    """Packaging material types."""
    PET = "PET"
    HDPE = "HDPE"
    PP = "PP"
    PS = "PS"
    LDPE = "LDPE"
    OTHER_PLASTIC = "other_plastic"
    GLASS = "glass"
    ALUMINIUM = "aluminium"
    STEEL = "steel"
    CARDBOARD = "cardboard"
    PAPER = "paper"
    WOOD = "wood"
    COMPOSITE = "composite"


class PackagingType(str, Enum):
    """Packaging type classification."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    TRANSPORT = "transport"


class EcoModulationGrade(str, Enum):
    """EPR eco-modulation grade."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class LabelStatus(str, Enum):
    """Labeling compliance status."""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class PackagingItem(BaseModel):
    """Individual packaging item record."""
    item_id: str = Field(default_factory=lambda: f"pkg-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="Packaging item name")
    material: str = Field(..., description="Material type")
    packaging_type: PackagingType = Field(default=PackagingType.PRIMARY)
    weight_grams: float = Field(default=0.0, ge=0.0, description="Weight per unit in grams")
    annual_units: int = Field(default=0, ge=0, description="Annual units placed on market")
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    recyclable: bool = Field(default=False, description="Is the packaging recyclable?")
    reusable: bool = Field(default=False, description="Is the packaging reusable?")
    compostable: bool = Field(default=False, description="Is the packaging compostable?")
    has_correct_label: bool = Field(default=False, description="Has correct recycling label?")
    has_material_identification: bool = Field(default=False)
    has_sorting_instruction: bool = Field(default=False)
    product_category: str = Field(default="", description="Associated product category")
    supplier_id: str = Field(default="")


class RecycledContentTarget(BaseModel):
    """PPWR recycled content targets by material and year."""
    material: str = Field(...)
    target_2025_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_2030_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_2040_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class RecycledContentGap(BaseModel):
    """Gap between actual and target recycled content."""
    material: str = Field(...)
    actual_pct: float = Field(default=0.0)
    target_pct: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    affected_items: int = Field(default=0)
    total_weight_kg: float = Field(default=0.0)
    status: str = Field(default="gap", description="met|gap|excess")


class EPRFeeItem(BaseModel):
    """EPR fee calculation for a packaging item."""
    item_id: str = Field(default="")
    material: str = Field(default="")
    weight_tonnes: float = Field(default=0.0)
    base_fee_eur: float = Field(default=0.0)
    eco_modulation_factor: float = Field(default=1.0)
    final_fee_eur: float = Field(default=0.0)
    grade: EcoModulationGrade = Field(default=EcoModulationGrade.C)


class LabelingResult(BaseModel):
    """Labeling compliance result for a packaging item."""
    item_id: str = Field(default="")
    material_identification: bool = Field(default=False)
    sorting_instruction: bool = Field(default=False)
    recyclability_label: bool = Field(default=False)
    digital_watermark: bool = Field(default=False)
    status: LabelStatus = Field(default=LabelStatus.NOT_ASSESSED)
    missing_elements: List[str] = Field(default_factory=list)


class PackagingInput(BaseModel):
    """Input data model for PackagingComplianceWorkflow."""
    packaging_items: List[PackagingItem] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class PackagingResult(BaseModel):
    """Complete result from packaging compliance workflow."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="packaging_compliance")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    compliance_status: str = Field(default="non_compliant")
    total_items: int = Field(default=0)
    total_weight_tonnes: float = Field(default=0.0)
    recycled_content_gaps: List[RecycledContentGap] = Field(default_factory=list)
    epr_fees: List[EPRFeeItem] = Field(default_factory=list)
    total_epr_fee_eur: float = Field(default=0.0)
    labeling_status: Dict[str, int] = Field(default_factory=dict)
    labeling_results: List[LabelingResult] = Field(default_factory=list)
    recyclability_rate_pct: float = Field(default=0.0)
    reuse_rate_pct: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# PPWR TARGETS AND FEE SCHEDULES
# =============================================================================

PPWR_RECYCLED_CONTENT_TARGETS: Dict[str, Dict[str, float]] = {
    "PET": {"2025": 25.0, "2030": 30.0, "2040": 65.0},
    "HDPE": {"2025": 0.0, "2030": 10.0, "2040": 50.0},
    "PP": {"2025": 0.0, "2030": 10.0, "2040": 50.0},
    "PS": {"2025": 0.0, "2030": 10.0, "2040": 50.0},
    "LDPE": {"2025": 0.0, "2030": 10.0, "2040": 50.0},
    "other_plastic": {"2025": 0.0, "2030": 10.0, "2040": 50.0},
    "glass": {"2025": 0.0, "2030": 0.0, "2040": 0.0},
    "aluminium": {"2025": 0.0, "2030": 0.0, "2040": 0.0},
    "steel": {"2025": 0.0, "2030": 0.0, "2040": 0.0},
    "cardboard": {"2025": 0.0, "2030": 0.0, "2040": 0.0},
    "paper": {"2025": 0.0, "2030": 0.0, "2040": 0.0},
    "wood": {"2025": 0.0, "2030": 0.0, "2040": 0.0},
    "composite": {"2025": 0.0, "2030": 10.0, "2040": 30.0},
}

# EPR base fees (EUR per tonne, indicative)
EPR_BASE_FEES: Dict[str, float] = {
    "PET": 180.0, "HDPE": 165.0, "PP": 170.0, "PS": 250.0,
    "LDPE": 200.0, "other_plastic": 280.0, "glass": 45.0,
    "aluminium": 120.0, "steel": 80.0, "cardboard": 55.0,
    "paper": 50.0, "wood": 35.0, "composite": 350.0,
}

# Eco-modulation factors by grade
ECO_MODULATION: Dict[str, float] = {
    "A": 0.50, "B": 0.75, "C": 1.00, "D": 1.25, "F": 1.50,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PackagingComplianceWorkflow:
    """
    4-phase PPWR packaging compliance workflow.

    Catalogs packaging, assesses recycled content vs PPWR targets,
    calculates EPR fees with eco-modulation, and audits labeling
    compliance.

    Zero-hallucination: all targets from PPWR regulation text,
    all fees deterministic.

    Example:
        >>> wf = PackagingComplianceWorkflow()
        >>> inp = PackagingInput(packaging_items=[...])
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize PackagingComplianceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._gaps: List[RecycledContentGap] = []
        self._epr_fees: List[EPRFeeItem] = []
        self._labeling_results: List[LabelingResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[PackagingInput] = None,
        packaging_items: Optional[List[PackagingItem]] = None,
        reporting_year: int = 2025,
        config: Optional[Dict[str, Any]] = None,
    ) -> PackagingResult:
        """Execute the 4-phase packaging compliance workflow."""
        if input_data is None:
            input_data = PackagingInput(
                packaging_items=packaging_items or [],
                reporting_year=reporting_year,
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting packaging compliance workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_packaging_inventory(input_data))
            phase_results.append(await self._phase_recycled_content(input_data))
            phase_results.append(await self._phase_epr_compliance(input_data))
            phase_results.append(await self._phase_labeling_audit(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Packaging workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        total_weight = sum(
            (p.weight_grams * p.annual_units) / 1_000_000
            for p in input_data.packaging_items
        )
        total_epr = sum(f.final_fee_eur for f in self._epr_fees)
        recyclable_count = sum(1 for p in input_data.packaging_items if p.recyclable)
        reusable_count = sum(1 for p in input_data.packaging_items if p.reusable)
        total_items = len(input_data.packaging_items)

        label_summary: Dict[str, int] = {}
        for lr in self._labeling_results:
            label_summary[lr.status.value] = label_summary.get(lr.status.value, 0) + 1

        gaps_exist = any(g.status == "gap" for g in self._gaps)
        non_compliant_labels = label_summary.get("non_compliant", 0)
        if not gaps_exist and non_compliant_labels == 0:
            compliance = "compliant"
        elif gaps_exist and non_compliant_labels > 0:
            compliance = "non_compliant"
        else:
            compliance = "partially_compliant"

        result = PackagingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            compliance_status=compliance,
            total_items=total_items,
            total_weight_tonnes=round(total_weight, 4),
            recycled_content_gaps=self._gaps,
            epr_fees=self._epr_fees,
            total_epr_fee_eur=round(total_epr, 2),
            labeling_status=label_summary,
            labeling_results=self._labeling_results,
            recyclability_rate_pct=round(recyclable_count / max(total_items, 1) * 100, 2),
            reuse_rate_pct=round(reusable_count / max(total_items, 1) * 100, 2),
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Packaging Inventory
    # -------------------------------------------------------------------------

    async def _phase_packaging_inventory(self, input_data: PackagingInput) -> PhaseResult:
        """Catalog all packaging by material and type."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        material_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        material_weight_kg: Dict[str, float] = {}

        for item in input_data.packaging_items:
            material_counts[item.material] = material_counts.get(item.material, 0) + 1
            type_counts[item.packaging_type.value] = type_counts.get(item.packaging_type.value, 0) + 1
            weight_kg = (item.weight_grams * item.annual_units) / 1000.0
            material_weight_kg[item.material] = material_weight_kg.get(item.material, 0.0) + weight_kg

            if item.weight_grams <= 0:
                warnings.append(f"Item {item.item_id}: weight_grams is zero")
            if item.annual_units <= 0:
                warnings.append(f"Item {item.item_id}: annual_units is zero")

        outputs["total_items"] = len(input_data.packaging_items)
        outputs["material_distribution"] = material_counts
        outputs["type_distribution"] = type_counts
        outputs["material_weight_kg"] = {k: round(v, 2) for k, v in material_weight_kg.items()}

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PackagingInventory: %d items cataloged", len(input_data.packaging_items))
        return PhaseResult(
            phase_name="packaging_inventory", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Recycled Content Assessment
    # -------------------------------------------------------------------------

    async def _phase_recycled_content(self, input_data: PackagingInput) -> PhaseResult:
        """Check recycled content vs PPWR targets."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._gaps = []

        year_key = str(input_data.reporting_year)
        if int(year_key) < 2030:
            year_key = "2025"
        elif int(year_key) < 2040:
            year_key = "2030"
        else:
            year_key = "2040"

        material_groups: Dict[str, List[PackagingItem]] = {}
        for item in input_data.packaging_items:
            material_groups.setdefault(item.material, []).append(item)

        for material, items in material_groups.items():
            targets = PPWR_RECYCLED_CONTENT_TARGETS.get(material, {})
            target_pct = targets.get(year_key, 0.0)
            if target_pct <= 0:
                continue

            total_weight = sum((it.weight_grams * it.annual_units) / 1000.0 for it in items)
            weighted_rc = sum(
                it.recycled_content_pct * (it.weight_grams * it.annual_units) / 1000.0
                for it in items
            )
            actual_pct = weighted_rc / total_weight if total_weight > 0 else 0.0
            gap = target_pct - actual_pct

            self._gaps.append(RecycledContentGap(
                material=material,
                actual_pct=round(actual_pct, 2),
                target_pct=target_pct,
                gap_pct=round(max(gap, 0.0), 2),
                affected_items=len(items),
                total_weight_kg=round(total_weight, 2),
                status="gap" if gap > 0 else "met",
            ))

        outputs["materials_assessed"] = len(material_groups)
        outputs["gaps_found"] = sum(1 for g in self._gaps if g.status == "gap")
        outputs["compliant_materials"] = sum(1 for g in self._gaps if g.status == "met")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 2 RecycledContent: %d gaps found", outputs["gaps_found"])
        return PhaseResult(
            phase_name="recycled_content_assessment", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: EPR Compliance
    # -------------------------------------------------------------------------

    async def _phase_epr_compliance(self, input_data: PackagingInput) -> PhaseResult:
        """Calculate EPR fees and eco-modulation grades."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._epr_fees = []

        for item in input_data.packaging_items:
            weight_tonnes = (item.weight_grams * item.annual_units) / 1_000_000
            base_fee_per_tonne = EPR_BASE_FEES.get(item.material, 200.0)
            base_fee = weight_tonnes * base_fee_per_tonne
            grade = self._determine_eco_grade(item)
            mod_factor = ECO_MODULATION.get(grade.value, 1.0)
            final_fee = base_fee * mod_factor

            self._epr_fees.append(EPRFeeItem(
                item_id=item.item_id, material=item.material,
                weight_tonnes=round(weight_tonnes, 6),
                base_fee_eur=round(base_fee, 2),
                eco_modulation_factor=mod_factor,
                final_fee_eur=round(final_fee, 2),
                grade=grade,
            ))

        total_fee = sum(f.final_fee_eur for f in self._epr_fees)
        grade_dist: Dict[str, int] = {}
        for f in self._epr_fees:
            grade_dist[f.grade.value] = grade_dist.get(f.grade.value, 0) + 1

        outputs["total_epr_fee_eur"] = round(total_fee, 2)
        outputs["grade_distribution"] = grade_dist
        outputs["items_assessed"] = len(self._epr_fees)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 3 EPRCompliance: total_fee=%.2f EUR", total_fee)
        return PhaseResult(
            phase_name="epr_compliance", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    def _determine_eco_grade(self, item: PackagingItem) -> EcoModulationGrade:
        """Determine eco-modulation grade for a packaging item."""
        score = 0
        if item.recyclable:
            score += 3
        if item.reusable:
            score += 3
        if item.compostable:
            score += 1
        if item.recycled_content_pct >= 50:
            score += 3
        elif item.recycled_content_pct >= 25:
            score += 2
        elif item.recycled_content_pct >= 10:
            score += 1

        if score >= 8:
            return EcoModulationGrade.A
        elif score >= 6:
            return EcoModulationGrade.B
        elif score >= 4:
            return EcoModulationGrade.C
        elif score >= 2:
            return EcoModulationGrade.D
        return EcoModulationGrade.F

    # -------------------------------------------------------------------------
    # Phase 4: Labeling Audit
    # -------------------------------------------------------------------------

    async def _phase_labeling_audit(self, input_data: PackagingInput) -> PhaseResult:
        """Verify labeling requirements are met."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._labeling_results = []

        for item in input_data.packaging_items:
            missing: List[str] = []
            if not item.has_material_identification:
                missing.append("material_identification")
            if not item.has_sorting_instruction:
                missing.append("sorting_instruction")
            if not item.has_correct_label:
                missing.append("recyclability_label")

            if not missing:
                status = LabelStatus.COMPLIANT
            elif len(missing) <= 1:
                status = LabelStatus.PARTIAL
            else:
                status = LabelStatus.NON_COMPLIANT

            self._labeling_results.append(LabelingResult(
                item_id=item.item_id,
                material_identification=item.has_material_identification,
                sorting_instruction=item.has_sorting_instruction,
                recyclability_label=item.has_correct_label,
                status=status,
                missing_elements=missing,
            ))

        status_counts: Dict[str, int] = {}
        for lr in self._labeling_results:
            status_counts[lr.status.value] = status_counts.get(lr.status.value, 0) + 1

        outputs["labeling_summary"] = status_counts
        outputs["compliant_count"] = status_counts.get("compliant", 0)
        outputs["non_compliant_count"] = status_counts.get("non_compliant", 0)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 4 LabelingAudit: %d compliant, %d non-compliant",
                         outputs["compliant_count"], outputs["non_compliant_count"])
        return PhaseResult(
            phase_name="labeling_audit", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: PackagingResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
