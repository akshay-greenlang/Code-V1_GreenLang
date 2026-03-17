# -*- coding: utf-8 -*-
"""
End-of-Life Planning Workflow
===================================

4-phase workflow for battery end-of-life management planning per EU Battery
Regulation 2023/1542, Articles 59-62, 71 and Annex XII. Implements collection
assessment, recycling target evaluation, material recovery calculation, and
compliance reporting.

Phases:
    1. CollectionAssessment  -- Assess collection infrastructure and rates
    2. RecyclingTargets      -- Evaluate recycling efficiency targets
    3. RecoveryCalculation   -- Calculate material recovery rates
    4. ComplianceReporting   -- Generate end-of-life compliance report

Regulatory references:
    - EU Regulation 2023/1542 Art. 59 (collection of waste portable batteries)
    - EU Regulation 2023/1542 Art. 60 (collection rate targets)
    - EU Regulation 2023/1542 Art. 61 (collection of waste batteries from EVs)
    - EU Regulation 2023/1542 Art. 62 (treatment and recycling)
    - EU Regulation 2023/1542 Art. 71 (recycling efficiencies)
    - EU Regulation 2023/1542 Annex XII (recycling efficiency formula)

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
    """Phases of the end-of-life planning workflow."""
    COLLECTION_ASSESSMENT = "collection_assessment"
    RECYCLING_TARGETS = "recycling_targets"
    RECOVERY_CALCULATION = "recovery_calculation"
    COMPLIANCE_REPORTING = "compliance_reporting"


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


class BatteryWasteCategory(str, Enum):
    """Battery waste stream categories."""
    PORTABLE = "portable"
    EV = "ev"
    INDUSTRIAL = "industrial"
    LMT = "lmt"
    SLI = "sli"


class CollectionChannel(str, Enum):
    """Collection infrastructure channels."""
    PRODUCER_TAKE_BACK = "producer_take_back"
    MUNICIPAL_COLLECTION = "municipal_collection"
    RETAIL_COLLECTION = "retail_collection"
    DEDICATED_FACILITY = "dedicated_facility"
    OEM_RETURN = "oem_return"
    AUTHORIZED_AGENT = "authorized_agent"


class RecyclingProcess(str, Enum):
    """Battery recycling process types."""
    HYDROMETALLURGICAL = "hydrometallurgical"
    PYROMETALLURGICAL = "pyrometallurgical"
    DIRECT_RECYCLING = "direct_recycling"
    MECHANICAL = "mechanical"
    COMBINED = "combined"


# =============================================================================
# REGULATORY TARGETS (Art. 60, 71)
# =============================================================================


# Collection rate targets for portable batteries (Art. 60)
COLLECTION_RATE_TARGETS: Dict[str, Dict[str, float]] = {
    "portable": {"2024": 45.0, "2027": 63.0, "2030": 73.0},
    "lmt": {"2028": 51.0, "2031": 61.0},
    "ev": {"target": 100.0},
    "industrial": {"target": 100.0},
    "sli": {"target": 100.0},
}

# Recycling efficiency targets (Art. 71)
RECYCLING_EFFICIENCY_TARGETS: Dict[str, Dict[str, float]] = {
    "lithium_ion": {"2026": 65.0, "2031": 70.0},
    "lead_acid": {"2026": 75.0, "2031": 80.0},
    "nickel_cadmium": {"2026": 80.0, "2031": 80.0},
    "other": {"2026": 50.0, "2031": 50.0},
}

# Material recovery targets (Art. 71(3))
MATERIAL_RECOVERY_TARGETS: Dict[str, Dict[str, float]] = {
    "cobalt": {"2028": 90.0, "2032": 95.0},
    "copper": {"2028": 90.0, "2032": 95.0},
    "lead": {"2028": 90.0, "2032": 95.0},
    "lithium": {"2028": 50.0, "2032": 80.0},
    "nickel": {"2028": 90.0, "2032": 95.0},
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


class CollectionPoint(BaseModel):
    """Collection infrastructure point."""
    point_id: str = Field(default_factory=lambda: f"cp-{_new_uuid()[:8]}")
    name: str = Field(default="", description="Collection point name")
    channel: CollectionChannel = Field(default=CollectionChannel.PRODUCER_TAKE_BACK)
    country_code: str = Field(default="", description="ISO 3166-1 alpha-2")
    capacity_tonnes_year: float = Field(default=0.0, ge=0.0)
    actual_collected_tonnes: float = Field(default=0.0, ge=0.0)
    operational: bool = Field(default=True)


class WasteStreamRecord(BaseModel):
    """Battery waste stream record."""
    stream_id: str = Field(default_factory=lambda: f"ws-{_new_uuid()[:8]}")
    waste_category: BatteryWasteCategory = Field(default=BatteryWasteCategory.EV)
    chemistry: str = Field(default="lithium_ion")
    batteries_placed_on_market_tonnes: float = Field(
        default=0.0, ge=0.0, description="Batteries placed on market (tonnes)"
    )
    batteries_collected_tonnes: float = Field(
        default=0.0, ge=0.0, description="Batteries collected (tonnes)"
    )
    batteries_recycled_tonnes: float = Field(
        default=0.0, ge=0.0, description="Batteries sent to recycling (tonnes)"
    )
    recycling_process: RecyclingProcess = Field(
        default=RecyclingProcess.HYDROMETALLURGICAL
    )
    recycler_name: str = Field(default="", description="Recycling facility")
    recycler_country: str = Field(default="")
    reporting_period: str = Field(default="", description="e.g., 2025-Q1")


class MaterialRecoveryRecord(BaseModel):
    """Material recovery data from recycling."""
    recovery_id: str = Field(default_factory=lambda: f"mr-{_new_uuid()[:8]}")
    material_name: str = Field(..., description="Recovered material")
    input_mass_kg: float = Field(default=0.0, ge=0.0, description="Mass entering recycling")
    recovered_mass_kg: float = Field(default=0.0, ge=0.0, description="Mass recovered")
    recovery_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    recycling_process: RecyclingProcess = Field(
        default=RecyclingProcess.HYDROMETALLURGICAL
    )
    purity_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Recovered material purity")


class EndOfLifeInput(BaseModel):
    """Input data model for EndOfLifePlanningWorkflow."""
    battery_id: str = Field(default_factory=lambda: f"bat-{_new_uuid()[:8]}")
    battery_model: str = Field(default="")
    battery_category: str = Field(default="ev")
    battery_chemistry: str = Field(default="lithium_ion")
    collection_points: List[CollectionPoint] = Field(default_factory=list)
    waste_streams: List[WasteStreamRecord] = Field(default_factory=list)
    material_recoveries: List[MaterialRecoveryRecord] = Field(default_factory=list)
    target_year: str = Field(default="2028", description="Regulatory target year")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class CollectionAssessmentSummary(BaseModel):
    """Summary of collection infrastructure assessment."""
    total_collection_points: int = Field(default=0, ge=0)
    total_capacity_tonnes: float = Field(default=0.0, ge=0.0)
    total_collected_tonnes: float = Field(default=0.0, ge=0.0)
    collection_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_rate_pct: float = Field(default=0.0, ge=0.0)
    meets_target: bool = Field(default=False)
    gap_pct: float = Field(default=0.0)
    channel_breakdown: Dict[str, float] = Field(default_factory=dict)


class RecyclingEfficiencySummary(BaseModel):
    """Summary of recycling efficiency assessment."""
    chemistry: str = Field(default="")
    total_input_tonnes: float = Field(default=0.0, ge=0.0)
    total_recycled_tonnes: float = Field(default=0.0, ge=0.0)
    recycling_efficiency_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_pct: float = Field(default=0.0, ge=0.0)
    meets_target: bool = Field(default=False)
    gap_pct: float = Field(default=0.0)


class MaterialRecoverySummary(BaseModel):
    """Summary of material recovery rates."""
    material: str = Field(default="")
    input_mass_kg: float = Field(default=0.0, ge=0.0)
    recovered_mass_kg: float = Field(default=0.0, ge=0.0)
    recovery_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_pct: float = Field(default=0.0, ge=0.0)
    meets_target: bool = Field(default=False)
    gap_pct: float = Field(default=0.0)


class EndOfLifeResult(BaseModel):
    """Complete result from end-of-life planning workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="end_of_life_planning")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    battery_id: str = Field(default="")
    collection_summary: Optional[CollectionAssessmentSummary] = Field(default=None)
    recycling_summary: Optional[RecyclingEfficiencySummary] = Field(default=None)
    material_recovery_summaries: List[MaterialRecoverySummary] = Field(
        default_factory=list
    )
    overall_eol_compliant: bool = Field(default=False)
    compliance_gaps: List[str] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class EndOfLifePlanningWorkflow:
    """
    4-phase end-of-life planning workflow per EU Battery Regulation.

    Implements end-of-life management assessment following EU Regulation
    2023/1542 Art. 59-62 and Art. 71. Assesses collection infrastructure,
    evaluates recycling efficiency targets, calculates material recovery
    rates, and generates compliance reports.

    Zero-hallucination: all rates and efficiencies computed with
    deterministic arithmetic (mass_out / mass_in * 100). No LLM in
    calculation paths.

    Example:
        >>> wf = EndOfLifePlanningWorkflow()
        >>> inp = EndOfLifeInput(waste_streams=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.overall_eol_compliant in (True, False)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize EndOfLifePlanningWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._collection_points: List[CollectionPoint] = []
        self._waste_streams: List[WasteStreamRecord] = []
        self._recoveries: List[MaterialRecoveryRecord] = []
        self._collection_summary: Optional[CollectionAssessmentSummary] = None
        self._recycling_summary: Optional[RecyclingEfficiencySummary] = None
        self._recovery_summaries: List[MaterialRecoverySummary] = []
        self._compliance_gaps: List[str] = []
        self._compliant: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.COLLECTION_ASSESSMENT.value, "description": "Assess collection infrastructure and rates"},
            {"name": WorkflowPhase.RECYCLING_TARGETS.value, "description": "Evaluate recycling efficiency targets"},
            {"name": WorkflowPhase.RECOVERY_CALCULATION.value, "description": "Calculate material recovery rates"},
            {"name": WorkflowPhase.COMPLIANCE_REPORTING.value, "description": "Generate end-of-life compliance report"},
        ]

    def validate_inputs(self, input_data: EndOfLifeInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.waste_streams and not input_data.collection_points:
            issues.append("At least waste streams or collection points are required")
        for ws in input_data.waste_streams:
            if ws.batteries_collected_tonnes > ws.batteries_placed_on_market_tonnes:
                issues.append(
                    f"Waste stream {ws.stream_id}: collected exceeds placed on market"
                )
        return issues

    async def execute(
        self,
        input_data: Optional[EndOfLifeInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EndOfLifeResult:
        """
        Execute the 4-phase end-of-life planning workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            EndOfLifeResult with collection, recycling, and recovery assessments.
        """
        if input_data is None:
            input_data = EndOfLifeInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting end-of-life planning workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_collection_assessment(input_data))
            phases_done += 1
            phase_results.append(await self._phase_recycling_targets(input_data))
            phases_done += 1
            phase_results.append(await self._phase_recovery_calculation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_compliance_reporting(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("End-of-life workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        result = EndOfLifeResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            battery_id=input_data.battery_id,
            collection_summary=self._collection_summary,
            recycling_summary=self._recycling_summary,
            material_recovery_summaries=self._recovery_summaries,
            overall_eol_compliant=self._compliant,
            compliance_gaps=self._compliance_gaps,
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "End-of-life %s completed in %.2fs: compliant=%s, %d gaps",
            self.workflow_id, elapsed, self._compliant, len(self._compliance_gaps),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Collection Assessment
    # -------------------------------------------------------------------------

    async def _phase_collection_assessment(
        self, input_data: EndOfLifeInput,
    ) -> PhaseResult:
        """Assess collection infrastructure and compute collection rates."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._collection_points = list(input_data.collection_points)
        self._waste_streams = list(input_data.waste_streams)

        # Aggregate collection data
        total_capacity = sum(
            cp.capacity_tonnes_year for cp in self._collection_points if cp.operational
        )
        total_collected = sum(
            cp.actual_collected_tonnes for cp in self._collection_points
        )

        # Collection from waste streams
        placed_on_market = sum(
            ws.batteries_placed_on_market_tonnes for ws in self._waste_streams
        )
        collected_from_streams = sum(
            ws.batteries_collected_tonnes for ws in self._waste_streams
        )

        # Use the larger of the two collection sources
        effective_collected = max(total_collected, collected_from_streams)
        effective_base = placed_on_market if placed_on_market > 0 else total_capacity

        collection_rate = round(
            (effective_collected / effective_base * 100)
            if effective_base > 0 else 0.0, 2
        )

        # Determine applicable target
        category = input_data.battery_category
        target_year = input_data.target_year
        category_targets = COLLECTION_RATE_TARGETS.get(category, {})
        target_rate = category_targets.get(target_year, category_targets.get("target", 0.0))

        meets_target = collection_rate >= target_rate
        gap = round(max(0.0, target_rate - collection_rate), 2)

        # Channel breakdown
        channel_breakdown: Dict[str, float] = {}
        for cp in self._collection_points:
            ch = cp.channel.value
            channel_breakdown[ch] = channel_breakdown.get(ch, 0.0) + cp.actual_collected_tonnes

        self._collection_summary = CollectionAssessmentSummary(
            total_collection_points=len(self._collection_points),
            total_capacity_tonnes=round(total_capacity, 2),
            total_collected_tonnes=round(effective_collected, 2),
            collection_rate_pct=collection_rate,
            target_rate_pct=target_rate,
            meets_target=meets_target,
            gap_pct=gap,
            channel_breakdown={k: round(v, 2) for k, v in channel_breakdown.items()},
        )

        if not meets_target and target_rate > 0:
            self._compliance_gaps.append(
                f"Collection rate {collection_rate}% below target {target_rate}% "
                f"(gap: {gap}%)"
            )
            warnings.append(
                f"Collection rate {collection_rate}% does not meet "
                f"{target_year} target of {target_rate}%"
            )

        non_operational = sum(1 for cp in self._collection_points if not cp.operational)
        if non_operational > 0:
            warnings.append(f"{non_operational} collection points are non-operational")

        outputs["collection_rate_pct"] = collection_rate
        outputs["target_rate_pct"] = target_rate
        outputs["meets_target"] = meets_target
        outputs["total_collection_points"] = len(self._collection_points)
        outputs["total_collected_tonnes"] = round(effective_collected, 2)
        outputs["channel_breakdown"] = channel_breakdown

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 CollectionAssessment: %.1f%% rate, target %.1f%%",
            collection_rate, target_rate,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.COLLECTION_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Recycling Targets
    # -------------------------------------------------------------------------

    async def _phase_recycling_targets(
        self, input_data: EndOfLifeInput,
    ) -> PhaseResult:
        """Evaluate recycling efficiency against regulatory targets."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        chemistry = input_data.battery_chemistry
        target_year = input_data.target_year

        total_input = sum(ws.batteries_collected_tonnes for ws in self._waste_streams)
        total_recycled = sum(ws.batteries_recycled_tonnes for ws in self._waste_streams)

        efficiency = round(
            (total_recycled / total_input * 100) if total_input > 0 else 0.0, 2
        )

        chemistry_targets = RECYCLING_EFFICIENCY_TARGETS.get(chemistry, {})
        target_eff = chemistry_targets.get(target_year, 0.0)

        meets_target = efficiency >= target_eff if target_eff > 0 else True
        gap = round(max(0.0, target_eff - efficiency), 2)

        self._recycling_summary = RecyclingEfficiencySummary(
            chemistry=chemistry,
            total_input_tonnes=round(total_input, 2),
            total_recycled_tonnes=round(total_recycled, 2),
            recycling_efficiency_pct=efficiency,
            target_pct=target_eff,
            meets_target=meets_target,
            gap_pct=gap,
        )

        if not meets_target and target_eff > 0:
            self._compliance_gaps.append(
                f"Recycling efficiency {efficiency}% below target {target_eff}% "
                f"for {chemistry} (gap: {gap}%)"
            )
            warnings.append(
                f"Recycling efficiency {efficiency}% does not meet "
                f"{target_year} target of {target_eff}% for {chemistry}"
            )

        # Process breakdown
        process_counts: Dict[str, int] = {}
        for ws in self._waste_streams:
            p = ws.recycling_process.value
            process_counts[p] = process_counts.get(p, 0) + 1

        outputs["recycling_efficiency_pct"] = efficiency
        outputs["target_efficiency_pct"] = target_eff
        outputs["meets_target"] = meets_target
        outputs["chemistry"] = chemistry
        outputs["total_input_tonnes"] = round(total_input, 2)
        outputs["total_recycled_tonnes"] = round(total_recycled, 2)
        outputs["process_distribution"] = process_counts

        if total_input <= 0:
            warnings.append("No waste stream input data; recycling efficiency is 0%")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 RecyclingTargets: %.1f%% efficiency, target %.1f%%",
            efficiency, target_eff,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.RECYCLING_TARGETS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Recovery Calculation
    # -------------------------------------------------------------------------

    async def _phase_recovery_calculation(
        self, input_data: EndOfLifeInput,
    ) -> PhaseResult:
        """Calculate material recovery rates against regulatory targets."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._recovery_summaries = []
        self._recoveries = list(input_data.material_recoveries)

        target_year = input_data.target_year

        # Group recoveries by material
        material_groups: Dict[str, List[MaterialRecoveryRecord]] = {}
        for rec in self._recoveries:
            material_groups.setdefault(rec.material_name.lower(), []).append(rec)

        for material, records in sorted(material_groups.items()):
            total_input = sum(r.input_mass_kg for r in records)
            total_recovered = sum(r.recovered_mass_kg for r in records)
            rate = round(
                (total_recovered / total_input * 100) if total_input > 0 else 0.0, 2
            )

            mat_targets = MATERIAL_RECOVERY_TARGETS.get(material, {})
            target_rate = mat_targets.get(target_year, 0.0)
            meets = rate >= target_rate if target_rate > 0 else True
            gap = round(max(0.0, target_rate - rate), 2)

            self._recovery_summaries.append(MaterialRecoverySummary(
                material=material,
                input_mass_kg=round(total_input, 2),
                recovered_mass_kg=round(total_recovered, 2),
                recovery_rate_pct=rate,
                target_pct=target_rate,
                meets_target=meets,
                gap_pct=gap,
            ))

            if not meets and target_rate > 0:
                self._compliance_gaps.append(
                    f"Material recovery for {material}: {rate}% below target "
                    f"{target_rate}% (gap: {gap}%)"
                )
                warnings.append(
                    f"{material} recovery rate {rate}% below {target_year} "
                    f"target {target_rate}%"
                )

        # Check for missing critical materials
        critical_materials = set(MATERIAL_RECOVERY_TARGETS.keys())
        reported_materials = set(material_groups.keys())
        missing = critical_materials - reported_materials
        if missing:
            warnings.append(
                f"No recovery data for regulated materials: {', '.join(sorted(missing))}"
            )

        outputs["materials_analyzed"] = len(self._recovery_summaries)
        outputs["materials_meeting_targets"] = sum(
            1 for s in self._recovery_summaries if s.meets_target
        )
        outputs["materials_below_targets"] = sum(
            1 for s in self._recovery_summaries if not s.meets_target and s.target_pct > 0
        )
        outputs["per_material_rates"] = {
            s.material: s.recovery_rate_pct for s in self._recovery_summaries
        }
        outputs["missing_critical_materials"] = sorted(missing)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 RecoveryCalculation: %d materials, %d meeting targets",
            len(self._recovery_summaries),
            outputs["materials_meeting_targets"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.RECOVERY_CALCULATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Compliance Reporting
    # -------------------------------------------------------------------------

    async def _phase_compliance_reporting(
        self, input_data: EndOfLifeInput,
    ) -> PhaseResult:
        """Generate end-of-life compliance report."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Overall compliance determination
        collection_ok = (
            self._collection_summary.meets_target
            if self._collection_summary else False
        )
        recycling_ok = (
            self._recycling_summary.meets_target
            if self._recycling_summary else False
        )
        recovery_ok = all(
            s.meets_target for s in self._recovery_summaries
            if s.target_pct > 0
        ) if self._recovery_summaries else False

        self._compliant = collection_ok and recycling_ok and recovery_ok

        report = {
            "report_id": f"eol-{_new_uuid()[:8]}",
            "battery_id": input_data.battery_id,
            "battery_model": input_data.battery_model,
            "battery_category": input_data.battery_category,
            "battery_chemistry": input_data.battery_chemistry,
            "regulation_reference": "EU Regulation 2023/1542 Art. 59-62, 71",
            "reporting_year": input_data.reporting_year,
            "target_year": input_data.target_year,
            "overall_compliant": self._compliant,
            "collection_assessment": {
                "rate_pct": self._collection_summary.collection_rate_pct if self._collection_summary else 0.0,
                "target_pct": self._collection_summary.target_rate_pct if self._collection_summary else 0.0,
                "meets_target": collection_ok,
            },
            "recycling_efficiency": {
                "efficiency_pct": self._recycling_summary.recycling_efficiency_pct if self._recycling_summary else 0.0,
                "target_pct": self._recycling_summary.target_pct if self._recycling_summary else 0.0,
                "meets_target": recycling_ok,
            },
            "material_recovery": [
                {
                    "material": s.material,
                    "rate_pct": s.recovery_rate_pct,
                    "target_pct": s.target_pct,
                    "meets_target": s.meets_target,
                }
                for s in self._recovery_summaries
            ],
            "compliance_gaps": self._compliance_gaps,
            "issued_at": _utcnow().isoformat(),
        }

        outputs["report_id"] = report["report_id"]
        outputs["overall_compliant"] = self._compliant
        outputs["collection_meets_target"] = collection_ok
        outputs["recycling_meets_target"] = recycling_ok
        outputs["recovery_meets_target"] = recovery_ok
        outputs["total_compliance_gaps"] = len(self._compliance_gaps)
        outputs["report_ready"] = True

        if not self._compliant:
            warnings.append(
                f"End-of-life compliance not met: {len(self._compliance_gaps)} gaps identified"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ComplianceReporting: %s, compliant=%s, %d gaps",
            report["report_id"], self._compliant, len(self._compliance_gaps),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.COMPLIANCE_REPORTING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: EndOfLifeResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
