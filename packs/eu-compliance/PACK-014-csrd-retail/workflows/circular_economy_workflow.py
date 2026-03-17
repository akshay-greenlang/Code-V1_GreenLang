# -*- coding: utf-8 -*-
"""
Circular Economy Workflow
==============================

4-phase workflow for circular economy metrics within PACK-014
CSRD Retail and Consumer Goods Pack.

Phases:
    1. TakeBackPrograms       -- Assess collection volumes and recovery rates
    2. MaterialRecovery       -- Track material flows and recycling rates
    3. EPRSchemeCompliance     -- Check compliance by EPR scheme
    4. CircularityMetrics     -- Calculate MCI, waste diversion, recycled content

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


class MaterialStream(str, Enum):
    """Material stream types."""
    PLASTIC = "plastic"
    GLASS = "glass"
    METAL = "metal"
    PAPER_CARDBOARD = "paper_cardboard"
    TEXTILES = "textiles"
    ELECTRONICS = "electronics"
    BATTERIES = "batteries"
    ORGANIC = "organic"
    MIXED = "mixed"


class EPRScheme(str, Enum):
    """Extended Producer Responsibility scheme types."""
    PACKAGING = "packaging"
    WEEE = "weee"
    BATTERIES = "batteries"
    TEXTILES = "textiles"
    FURNITURE = "furniture"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class TakeBackProgram(BaseModel):
    """Take-back program data."""
    program_id: str = Field(default_factory=lambda: f"tbp-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="Program name")
    material_stream: MaterialStream = Field(default=MaterialStream.MIXED)
    collection_points: int = Field(default=0, ge=0)
    items_collected: int = Field(default=0, ge=0)
    weight_collected_tonnes: float = Field(default=0.0, ge=0.0)
    weight_recovered_tonnes: float = Field(default=0.0, ge=0.0)
    weight_recycled_tonnes: float = Field(default=0.0, ge=0.0)
    recovery_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    active: bool = Field(default=True)
    partner_name: str = Field(default="")


class MaterialFlow(BaseModel):
    """Material flow record."""
    flow_id: str = Field(default_factory=lambda: f"mf-{uuid.uuid4().hex[:8]}")
    material: MaterialStream = Field(default=MaterialStream.MIXED)
    input_tonnes: float = Field(default=0.0, ge=0.0, description="Virgin + recycled input")
    recycled_input_tonnes: float = Field(default=0.0, ge=0.0)
    output_product_tonnes: float = Field(default=0.0, ge=0.0)
    waste_tonnes: float = Field(default=0.0, ge=0.0)
    recycled_output_tonnes: float = Field(default=0.0, ge=0.0)
    landfill_tonnes: float = Field(default=0.0, ge=0.0)
    incineration_tonnes: float = Field(default=0.0, ge=0.0)
    reuse_tonnes: float = Field(default=0.0, ge=0.0)


class EPRData(BaseModel):
    """EPR scheme compliance data."""
    scheme: EPRScheme = Field(default=EPRScheme.PACKAGING)
    scheme_name: str = Field(default="")
    country: str = Field(default="", description="ISO 3166-1 alpha-2")
    placed_on_market_tonnes: float = Field(default=0.0, ge=0.0)
    collected_tonnes: float = Field(default=0.0, ge=0.0)
    recycled_tonnes: float = Field(default=0.0, ge=0.0)
    target_collection_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_recycling_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    fee_paid_eur: float = Field(default=0.0, ge=0.0)
    compliant: bool = Field(default=False)


class TakeBackMetrics(BaseModel):
    """Aggregated take-back program metrics."""
    total_programs: int = Field(default=0)
    total_collection_points: int = Field(default=0)
    total_collected_tonnes: float = Field(default=0.0)
    total_recovered_tonnes: float = Field(default=0.0)
    avg_recovery_rate_pct: float = Field(default=0.0)
    by_material: Dict[str, float] = Field(default_factory=dict)


class CircularEconomyInput(BaseModel):
    """Input data model for CircularEconomyWorkflow."""
    take_back_programs: List[TakeBackProgram] = Field(default_factory=list)
    material_flows: List[MaterialFlow] = Field(default_factory=list)
    epr_data: List[EPRData] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class CircularEconomyResult(BaseModel):
    """Complete result from circular economy workflow."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="circular_economy")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    mci_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Material Circularity Indicator")
    epr_compliance: Dict[str, Any] = Field(default_factory=dict)
    take_back_metrics: TakeBackMetrics = Field(default_factory=TakeBackMetrics)
    diversion_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    material_recovery_rates: Dict[str, float] = Field(default_factory=dict)
    epr_total_fees_eur: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CircularEconomyWorkflow:
    """
    4-phase circular economy workflow.

    Assesses take-back programs, tracks material flows and recovery,
    checks EPR scheme compliance, and calculates circularity metrics
    including Material Circularity Indicator (MCI).

    MCI Formula (Ellen MacArthur Foundation):
        MCI = 1 - LFI * F(X)
        where LFI = Linear Flow Index, F(X) = utility factor

    Example:
        >>> wf = CircularEconomyWorkflow()
        >>> inp = CircularEconomyInput(take_back_programs=[...])
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CircularEconomyWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._take_back_metrics: TakeBackMetrics = TakeBackMetrics()
        self._material_recovery: Dict[str, float] = {}
        self._epr_results: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[CircularEconomyInput] = None,
        take_back_programs: Optional[List[TakeBackProgram]] = None,
        material_flows: Optional[List[MaterialFlow]] = None,
        epr_data: Optional[List[EPRData]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> CircularEconomyResult:
        """Execute the 4-phase circular economy workflow."""
        if input_data is None:
            input_data = CircularEconomyInput(
                take_back_programs=take_back_programs or [],
                material_flows=material_flows or [],
                epr_data=epr_data or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting circular economy workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_take_back(input_data))
            phase_results.append(await self._phase_material_recovery(input_data))
            phase_results.append(await self._phase_epr_compliance(input_data))
            phase_results.append(await self._phase_circularity_metrics(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Circular economy workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        mci = self._calculate_mci(input_data)
        diversion = self._calculate_diversion_rate(input_data)
        recycled_content = self._calculate_recycled_content(input_data)
        epr_fees = sum(e.fee_paid_eur for e in input_data.epr_data)

        result = CircularEconomyResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            mci_score=round(mci, 4),
            epr_compliance=self._epr_results,
            take_back_metrics=self._take_back_metrics,
            diversion_rate_pct=round(diversion, 2),
            recycled_content_pct=round(recycled_content, 2),
            material_recovery_rates=self._material_recovery,
            epr_total_fees_eur=round(epr_fees, 2),
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Take-Back Programs
    # -------------------------------------------------------------------------

    async def _phase_take_back(self, input_data: CircularEconomyInput) -> PhaseResult:
        """Assess collection volumes and recovery rates."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        programs = input_data.take_back_programs

        total_collected = sum(p.weight_collected_tonnes for p in programs)
        total_recovered = sum(p.weight_recovered_tonnes for p in programs)
        total_points = sum(p.collection_points for p in programs)

        by_material: Dict[str, float] = {}
        for p in programs:
            by_material[p.material_stream.value] = by_material.get(p.material_stream.value, 0.0) + p.weight_collected_tonnes

        avg_recovery = (total_recovered / total_collected * 100) if total_collected > 0 else 0.0

        self._take_back_metrics = TakeBackMetrics(
            total_programs=len(programs),
            total_collection_points=total_points,
            total_collected_tonnes=round(total_collected, 4),
            total_recovered_tonnes=round(total_recovered, 4),
            avg_recovery_rate_pct=round(avg_recovery, 2),
            by_material={k: round(v, 4) for k, v in by_material.items()},
        )

        outputs["total_programs"] = len(programs)
        outputs["total_collected_tonnes"] = round(total_collected, 4)
        outputs["avg_recovery_rate_pct"] = round(avg_recovery, 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 1 TakeBack: %d programs, %.2f tonnes collected", len(programs), total_collected)
        return PhaseResult(
            phase_name="take_back_programs", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Material Recovery
    # -------------------------------------------------------------------------

    async def _phase_material_recovery(self, input_data: CircularEconomyInput) -> PhaseResult:
        """Track material flows and recycling rates."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._material_recovery = {}

        for flow in input_data.material_flows:
            total_output = flow.recycled_output_tonnes + flow.reuse_tonnes
            total_waste = flow.waste_tonnes + flow.landfill_tonnes + flow.incineration_tonnes
            recovery_rate = (total_output / (total_output + total_waste) * 100) if (total_output + total_waste) > 0 else 0.0
            self._material_recovery[flow.material.value] = round(recovery_rate, 2)

        outputs["materials_tracked"] = len(input_data.material_flows)
        outputs["recovery_rates"] = self._material_recovery

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 2 MaterialRecovery: %d materials tracked", len(input_data.material_flows))
        return PhaseResult(
            phase_name="material_recovery", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: EPR Scheme Compliance
    # -------------------------------------------------------------------------

    async def _phase_epr_compliance(self, input_data: CircularEconomyInput) -> PhaseResult:
        """Check compliance by EPR scheme."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        scheme_results: Dict[str, Dict[str, Any]] = {}

        for epr in input_data.epr_data:
            collection_pct = (epr.collected_tonnes / epr.placed_on_market_tonnes * 100) if epr.placed_on_market_tonnes > 0 else 0.0
            recycling_pct = (epr.recycled_tonnes / epr.placed_on_market_tonnes * 100) if epr.placed_on_market_tonnes > 0 else 0.0

            collection_compliant = collection_pct >= epr.target_collection_pct
            recycling_compliant = recycling_pct >= epr.target_recycling_pct
            overall_compliant = collection_compliant and recycling_compliant

            key = f"{epr.scheme.value}_{epr.country}"
            scheme_results[key] = {
                "scheme": epr.scheme.value,
                "country": epr.country,
                "collection_pct": round(collection_pct, 2),
                "collection_target_pct": epr.target_collection_pct,
                "collection_compliant": collection_compliant,
                "recycling_pct": round(recycling_pct, 2),
                "recycling_target_pct": epr.target_recycling_pct,
                "recycling_compliant": recycling_compliant,
                "overall_compliant": overall_compliant,
                "fee_eur": round(epr.fee_paid_eur, 2),
            }

        self._epr_results = {
            "schemes_assessed": len(scheme_results),
            "compliant_count": sum(1 for v in scheme_results.values() if v["overall_compliant"]),
            "non_compliant_count": sum(1 for v in scheme_results.values() if not v["overall_compliant"]),
            "scheme_details": scheme_results,
        }

        outputs["schemes_assessed"] = len(scheme_results)
        outputs["compliant_count"] = self._epr_results["compliant_count"]

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 3 EPRCompliance: %d schemes, %d compliant", len(scheme_results), self._epr_results["compliant_count"])
        return PhaseResult(
            phase_name="epr_scheme_compliance", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Circularity Metrics
    # -------------------------------------------------------------------------

    async def _phase_circularity_metrics(self, input_data: CircularEconomyInput) -> PhaseResult:
        """Calculate MCI, waste diversion, recycled content."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}

        mci = self._calculate_mci(input_data)
        diversion = self._calculate_diversion_rate(input_data)
        recycled_content = self._calculate_recycled_content(input_data)

        outputs["mci_score"] = round(mci, 4)
        outputs["diversion_rate_pct"] = round(diversion, 2)
        outputs["recycled_content_pct"] = round(recycled_content, 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 4 CircularityMetrics: MCI=%.4f, diversion=%.1f%%", mci, diversion)
        return PhaseResult(
            phase_name="circularity_metrics", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_mci(self, input_data: CircularEconomyInput) -> float:
        """
        Calculate Material Circularity Indicator (0-1).

        MCI = 1 - LFI * F(X)
        LFI = (V + W) / (2 * M)
        V = virgin material input, W = unrecoverable waste
        M = total mass, F(X) = utility factor (simplified to 1.0)
        """
        total_input = sum(f.input_tonnes for f in input_data.material_flows)
        total_recycled_input = sum(f.recycled_input_tonnes for f in input_data.material_flows)
        total_waste = sum(f.landfill_tonnes + f.incineration_tonnes for f in input_data.material_flows)

        if total_input <= 0:
            return 0.0

        virgin = total_input - total_recycled_input
        lfi = (virgin + total_waste) / (2.0 * total_input)
        mci = max(0.0, min(1.0, 1.0 - lfi))
        return mci

    def _calculate_diversion_rate(self, input_data: CircularEconomyInput) -> float:
        """Calculate waste diversion rate (% diverted from landfill)."""
        total_waste = sum(f.waste_tonnes for f in input_data.material_flows)
        total_landfill = sum(f.landfill_tonnes for f in input_data.material_flows)
        if total_waste <= 0:
            return 0.0
        return ((total_waste - total_landfill) / total_waste) * 100

    def _calculate_recycled_content(self, input_data: CircularEconomyInput) -> float:
        """Calculate average recycled content percentage."""
        total_input = sum(f.input_tonnes for f in input_data.material_flows)
        total_recycled = sum(f.recycled_input_tonnes for f in input_data.material_flows)
        if total_input <= 0:
            return 0.0
        return (total_recycled / total_input) * 100

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: CircularEconomyResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
