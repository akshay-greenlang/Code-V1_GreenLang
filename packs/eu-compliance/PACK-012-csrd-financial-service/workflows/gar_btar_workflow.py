# -*- coding: utf-8 -*-
"""
GAR/BTAR Workflow
====================

Four-phase workflow for computing Green Asset Ratio (GAR) and Banking Book
Taxonomy Alignment Ratio (BTAR) under EU Taxonomy Regulation Art. 8 Delegated
Act for credit institutions.

Phases:
    1. AssetClassification - Classify assets into GAR-eligible categories
    2. AlignmentAssessment - Assess taxonomy alignment per technical criteria
    3. KPIComputation - Compute GAR/BTAR KPIs by objective and counterparty
    4. DisclosureGeneration - Generate EBA ITS disclosure tables

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

class PhaseStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)

class WorkflowResult(BaseModel):
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

class TaxonomyObjective(str, Enum):
    """EU Taxonomy environmental objectives."""
    CLIMATE_MITIGATION = "CLIMATE_MITIGATION"
    CLIMATE_ADAPTATION = "CLIMATE_ADAPTATION"
    WATER = "WATER"
    CIRCULAR_ECONOMY = "CIRCULAR_ECONOMY"
    POLLUTION = "POLLUTION"
    BIODIVERSITY = "BIODIVERSITY"

class AssetEligibility(str, Enum):
    """Asset eligibility categories for GAR."""
    ELIGIBLE = "ELIGIBLE"
    NON_ELIGIBLE = "NON_ELIGIBLE"
    EXCLUDED = "EXCLUDED"
    SOVEREIGN = "SOVEREIGN"
    CENTRAL_BANK = "CENTRAL_BANK"
    TRADING_BOOK = "TRADING_BOOK"

# ---------------------------------------------------------------------------
#  Input / Result Models
# ---------------------------------------------------------------------------

class GARAsset(BaseModel):
    """Single asset record for GAR/BTAR computation."""
    asset_id: str = Field(..., description="Asset identifier")
    counterparty_name: str = Field(default="")
    counterparty_nace: str = Field(default="", description="NACE sector code")
    asset_type: str = Field(default="loan", description="loan, bond, equity, etc.")
    gross_carrying_amount: float = Field(..., ge=0.0, description="GCA in EUR")
    is_nfrd_scope: bool = Field(default=True, description="Counterparty in NFRD/CSRD scope")
    is_sme: bool = Field(default=False)
    taxonomy_eligible: Optional[bool] = Field(None)
    taxonomy_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    substantial_contribution_objective: Optional[str] = Field(None)
    dnsh_compliant: bool = Field(default=False)
    minimum_safeguards: bool = Field(default=False)
    is_transitional: bool = Field(default=False)
    is_enabling: bool = Field(default=False)
    country: str = Field(default="")

class GARBTARInput(BaseModel):
    """Input for the GAR/BTAR workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_date: str = Field(..., description="Reporting date YYYY-MM-DD")
    assets: List[GARAsset] = Field(default_factory=list)
    include_btar: bool = Field(default=True)
    flow_period_start: Optional[str] = Field(None, description="Flow period start YYYY-MM-DD")
    total_assets_eur: Optional[float] = Field(None, ge=0.0)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v

class GARBTARResult(WorkflowResult):
    """Result from the GAR/BTAR workflow."""
    gar_pct: float = Field(default=0.0)
    btar_pct: float = Field(default=0.0)
    eligible_assets_eur: float = Field(default=0.0)
    aligned_assets_eur: float = Field(default=0.0)
    total_covered_assets_eur: float = Field(default=0.0)
    gar_by_objective: Dict[str, float] = Field(default_factory=dict)
    counterparties_assessed: int = Field(default=0)
    disclosure_tables_generated: int = Field(default=0)

# ---------------------------------------------------------------------------
#  Phase 1: Asset Classification
# ---------------------------------------------------------------------------

class AssetClassificationPhase:
    """Classify assets into GAR-eligible/non-eligible/excluded categories."""

    PHASE_NAME = "asset_classification"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            assets = config.get("assets", [])

            eligible = []
            non_eligible = []
            excluded = []
            total_gca = 0.0

            for asset in assets:
                gca = asset.get("gross_carrying_amount", 0.0)
                total_gca += gca
                asset_type = asset.get("asset_type", "")
                is_nfrd = asset.get("is_nfrd_scope", True)

                if asset_type in ("sovereign", "central_bank"):
                    excluded.append({**asset, "classification": "EXCLUDED",
                                     "reason": "Sovereign/central bank exposure"})
                elif asset_type == "trading_book":
                    excluded.append({**asset, "classification": "EXCLUDED",
                                     "reason": "Trading book position"})
                elif not is_nfrd:
                    non_eligible.append({**asset, "classification": "NON_ELIGIBLE",
                                         "reason": "Counterparty outside NFRD/CSRD scope"})
                elif asset.get("taxonomy_eligible") is False:
                    non_eligible.append({**asset, "classification": "NON_ELIGIBLE",
                                         "reason": "Non-eligible economic activity"})
                else:
                    eligible.append({**asset, "classification": "ELIGIBLE"})

            outputs["eligible_assets"] = eligible
            outputs["non_eligible_assets"] = non_eligible
            outputs["excluded_assets"] = excluded
            outputs["eligible_count"] = len(eligible)
            outputs["non_eligible_count"] = len(non_eligible)
            outputs["excluded_count"] = len(excluded)
            outputs["total_gca"] = total_gca
            outputs["eligible_gca"] = sum(a.get("gross_carrying_amount", 0.0) for a in eligible)
            outputs["non_eligible_gca"] = sum(a.get("gross_carrying_amount", 0.0) for a in non_eligible)

            status = PhaseStatus.COMPLETED
            records = len(assets)

        except Exception as exc:
            logger.error("AssetClassification failed: %s", exc, exc_info=True)
            errors.append(f"Asset classification failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs), records_processed=records,
        )

# ---------------------------------------------------------------------------
#  Phase 2: Alignment Assessment
# ---------------------------------------------------------------------------

class AlignmentAssessmentPhase:
    """Assess taxonomy alignment per substantial contribution, DNSH, MSS."""

    PHASE_NAME = "alignment_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            classification = context.get_phase_output("asset_classification")
            eligible = classification.get("eligible_assets", [])

            aligned = []
            not_aligned = []

            for asset in eligible:
                aligned_pct = asset.get("taxonomy_aligned_pct", 0.0)
                dnsh = asset.get("dnsh_compliant", False)
                mss = asset.get("minimum_safeguards", False)
                sc_obj = asset.get("substantial_contribution_objective")

                if aligned_pct > 0 and dnsh and mss and sc_obj:
                    gca = asset.get("gross_carrying_amount", 0.0)
                    aligned_amount = gca * aligned_pct / 100.0
                    aligned.append({
                        **asset,
                        "aligned_amount": round(aligned_amount, 2),
                        "alignment_status": "ALIGNED",
                        "sc_objective": sc_obj,
                    })
                else:
                    issues = []
                    if aligned_pct == 0:
                        issues.append("zero_alignment")
                    if not dnsh:
                        issues.append("dnsh_not_met")
                    if not mss:
                        issues.append("mss_not_met")
                    if not sc_obj:
                        issues.append("no_sc_objective")
                    not_aligned.append({
                        **asset,
                        "aligned_amount": 0.0,
                        "alignment_status": "NOT_ALIGNED",
                        "issues": issues,
                    })

            outputs["aligned_assets"] = aligned
            outputs["not_aligned_assets"] = not_aligned
            outputs["aligned_count"] = len(aligned)
            outputs["total_aligned_amount"] = round(
                sum(a.get("aligned_amount", 0.0) for a in aligned), 2
            )

            # By objective
            by_obj: Dict[str, float] = {}
            for a in aligned:
                obj = a.get("sc_objective", "OTHER")
                by_obj[obj] = by_obj.get(obj, 0.0) + a.get("aligned_amount", 0.0)
            outputs["aligned_by_objective"] = {k: round(v, 2) for k, v in by_obj.items()}

            # Transitional / enabling
            outputs["transitional_amount"] = round(
                sum(a.get("aligned_amount", 0.0) for a in aligned if a.get("is_transitional")), 2
            )
            outputs["enabling_amount"] = round(
                sum(a.get("aligned_amount", 0.0) for a in aligned if a.get("is_enabling")), 2
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("AlignmentAssessment failed: %s", exc, exc_info=True)
            errors.append(f"Alignment assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 3: KPI Computation
# ---------------------------------------------------------------------------

class KPIComputationPhase:
    """Compute GAR/BTAR KPIs by objective, counterparty type, flow/stock."""

    PHASE_NAME = "kpi_computation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            classification = context.get_phase_output("asset_classification")
            alignment = context.get_phase_output("alignment_assessment")

            total_gca = classification.get("total_gca", 0.0)
            total_assets_input = config.get("total_assets_eur") or total_gca
            eligible_gca = classification.get("eligible_gca", 0.0)
            aligned_amount = alignment.get("total_aligned_amount", 0.0)
            excluded_gca = sum(
                a.get("gross_carrying_amount", 0.0)
                for a in classification.get("excluded_assets", [])
            )

            covered_assets = total_assets_input - excluded_gca
            gar = round(aligned_amount / max(covered_assets, 1.0) * 100, 4) if covered_assets > 0 else 0.0

            outputs["gar_pct"] = gar
            outputs["covered_assets_eur"] = round(covered_assets, 2)
            outputs["eligible_assets_eur"] = round(eligible_gca, 2)
            outputs["aligned_assets_eur"] = round(aligned_amount, 2)
            outputs["excluded_assets_eur"] = round(excluded_gca, 2)

            # GAR by objective
            by_obj = alignment.get("aligned_by_objective", {})
            gar_by_obj = {}
            for obj, amount in by_obj.items():
                gar_by_obj[obj] = round(amount / max(covered_assets, 1.0) * 100, 4)
            outputs["gar_by_objective"] = gar_by_obj

            # BTAR (includes voluntary disclosures from non-NFRD counterparties)
            if config.get("include_btar", True):
                btar = round(aligned_amount / max(total_gca, 1.0) * 100, 4) if total_gca > 0 else 0.0
                outputs["btar_pct"] = btar
            else:
                outputs["btar_pct"] = 0.0

            # Counterparty breakdown
            cpty_set = set()
            for a in alignment.get("aligned_assets", []) + alignment.get("not_aligned_assets", []):
                cpty_set.add(a.get("asset_id", ""))
            outputs["counterparties_assessed"] = len(cpty_set)

            # Transitional / enabling ratios
            trans = alignment.get("transitional_amount", 0.0)
            enab = alignment.get("enabling_amount", 0.0)
            outputs["transitional_pct"] = round(trans / max(covered_assets, 1.0) * 100, 4)
            outputs["enabling_pct"] = round(enab / max(covered_assets, 1.0) * 100, 4)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("KPIComputation failed: %s", exc, exc_info=True)
            errors.append(f"KPI computation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 4: Disclosure Generation
# ---------------------------------------------------------------------------

class DisclosureGenerationPhase:
    """Generate EBA ITS disclosure tables for GAR/BTAR."""

    PHASE_NAME = "disclosure_generation"

    EBA_TEMPLATES = [
        "Template_1_GAR_Stock",
        "Template_2_GAR_Flow",
        "Template_3_BTAR",
        "Template_4_Qualitative",
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            kpi = context.get_phase_output("kpi_computation")

            tables_generated = 0
            disclosure = {}

            # Template 1: GAR Stock
            disclosure["template_1_gar_stock"] = {
                "gar_pct": kpi.get("gar_pct", 0.0),
                "covered_assets_eur": kpi.get("covered_assets_eur", 0.0),
                "eligible_assets_eur": kpi.get("eligible_assets_eur", 0.0),
                "aligned_assets_eur": kpi.get("aligned_assets_eur", 0.0),
                "by_objective": kpi.get("gar_by_objective", {}),
                "transitional_pct": kpi.get("transitional_pct", 0.0),
                "enabling_pct": kpi.get("enabling_pct", 0.0),
            }
            tables_generated += 1

            # Template 2: GAR Flow
            disclosure["template_2_gar_flow"] = {
                "flow_period_start": config.get("flow_period_start", ""),
                "reporting_date": config.get("reporting_date", ""),
                "new_lending_aligned_pct": kpi.get("gar_pct", 0.0),
                "note": "Flow data computed from same dataset; dedicated flow tracking recommended.",
            }
            tables_generated += 1

            # Template 3: BTAR
            if config.get("include_btar", True):
                disclosure["template_3_btar"] = {
                    "btar_pct": kpi.get("btar_pct", 0.0),
                    "total_assets_eur": kpi.get("covered_assets_eur", 0.0) + kpi.get("excluded_assets_eur", 0.0),
                    "aligned_assets_eur": kpi.get("aligned_assets_eur", 0.0),
                }
                tables_generated += 1

            # Template 4: Qualitative
            disclosure["template_4_qualitative"] = {
                "methodology_description": (
                    "Taxonomy alignment assessed using counterparty-reported data "
                    "against EU Taxonomy Delegated Acts technical screening criteria."
                ),
                "data_sources": ["Counterparty CSRD reports", "EPC registers", "NACE classifications"],
                "limitations": ["SME coverage gaps", "Estimated alignment for non-reporting entities"],
                "improvement_plan": "Expand direct counterparty engagement for taxonomy data.",
            }
            tables_generated += 1

            outputs["disclosure_tables"] = disclosure
            outputs["tables_generated"] = tables_generated
            outputs["reporting_date"] = config.get("reporting_date", "")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("DisclosureGeneration failed: %s", exc, exc_info=True)
            errors.append(f"Disclosure generation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Workflow Orchestrator
# ---------------------------------------------------------------------------

class GARBTARWorkflow:
    """Four-phase GAR/BTAR workflow for EU Taxonomy Art. 8 DA compliance."""

    WORKFLOW_NAME = "gar_btar"
    PHASE_ORDER = ["asset_classification", "alignment_assessment",
                    "kpi_computation", "disclosure_generation"]

    def __init__(self, progress_callback: Optional[Callable[[str, str, float], None]] = None) -> None:
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "asset_classification": AssetClassificationPhase(),
            "alignment_assessment": AlignmentAssessmentPhase(),
            "kpi_computation": KPIComputationPhase(),
            "disclosure_generation": DisclosureGenerationPhase(),
        }

    async def run(self, input_data: GARBTARInput) -> GARBTARResult:
        """Execute the complete 4-phase GAR/BTAR workflow."""
        started_at = utcnow()
        logger.info("Starting GAR/BTAR workflow %s org=%s", self.workflow_id, input_data.organization_id)
        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=input_data.model_dump(),
        )
        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                ))
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue
            if context.is_phase_completed(phase_name):
                continue
            self._notify_progress(phase_name, f"Starting: {phase_name}", idx / len(self.PHASE_ORDER))
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == "asset_classification":
                        overall_status = WorkflowStatus.FAILED
                        break
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=utcnow(), errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                ))
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                         for p in completed_phases)
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })
        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return GARBTARResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            gar_pct=summary.get("gar_pct", 0.0),
            btar_pct=summary.get("btar_pct", 0.0),
            eligible_assets_eur=summary.get("eligible_assets_eur", 0.0),
            aligned_assets_eur=summary.get("aligned_assets_eur", 0.0),
            total_covered_assets_eur=summary.get("total_covered_assets_eur", 0.0),
            gar_by_objective=summary.get("gar_by_objective", {}),
            counterparties_assessed=summary.get("counterparties_assessed", 0),
            disclosure_tables_generated=summary.get("disclosure_tables_generated", 0),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        kpi = context.get_phase_output("kpi_computation")
        disc = context.get_phase_output("disclosure_generation")
        return {
            "gar_pct": kpi.get("gar_pct", 0.0),
            "btar_pct": kpi.get("btar_pct", 0.0),
            "eligible_assets_eur": kpi.get("eligible_assets_eur", 0.0),
            "aligned_assets_eur": kpi.get("aligned_assets_eur", 0.0),
            "total_covered_assets_eur": kpi.get("covered_assets_eur", 0.0),
            "gar_by_objective": kpi.get("gar_by_objective", {}),
            "counterparties_assessed": kpi.get("counterparties_assessed", 0),
            "disclosure_tables_generated": disc.get("tables_generated", 0),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
