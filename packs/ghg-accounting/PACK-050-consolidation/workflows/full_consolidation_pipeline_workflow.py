# -*- coding: utf-8 -*-
"""
Full Consolidation Pipeline Workflow
====================================

8-phase end-to-end orchestrator that chains all sub-workflows for the
complete GHG consolidation lifecycle within PACK-050 GHG Consolidation Pack.

Phases:
    1. EntitySetup              -- Entity mapping (EntityMappingWorkflow)
    2. DataCollection           -- Entity data collection
                                   (EntityDataCollectionWorkflow)
    3. BoundaryDefinition       -- Boundary selection
                                   (BoundarySelectionWorkflow)
    4. OwnershipResolution      -- Ownership chain and equity calculations
    5. ConsolidationExecution   -- Core consolidation execution
                                   (ConsolidationExecutionWorkflow)
    6. EliminationProcessing    -- Intercompany elimination
                                   (EliminationWorkflow)
    7. ReportGeneration         -- Group reporting
                                   (GroupReportingWorkflow)
    8. AuditFinalization        -- Final audit trail, provenance chain,
                                   and sign-off generation

Supports conditional phase execution, checkpoint caching, resume from
failure, and full provenance chain across all sub-workflows.

Regulatory Basis:
    GHG Protocol Corporate Standard -- Full consolidation lifecycle
    ISO 14064-1:2018 -- Organisation-level GHG quantification
    CSRD / ESRS E1 -- Climate change disclosure

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class PipelinePhase(str, Enum):
    ENTITY_SETUP = "entity_setup"
    DATA_COLLECTION = "data_collection"
    BOUNDARY_DEFINITION = "boundary_definition"
    OWNERSHIP_RESOLUTION = "ownership_resolution"
    CONSOLIDATION_EXECUTION = "consolidation_execution"
    ELIMINATION_PROCESSING = "elimination_processing"
    REPORT_GENERATION = "report_generation"
    AUDIT_FINALIZATION = "audit_finalization"


class ReportFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


class CheckpointStatus(str, Enum):
    SAVED = "saved"
    LOADED = "loaded"
    NOT_FOUND = "not_found"


class AuditVerdict(str, Enum):
    CLEAN = "clean"
    QUALIFIED = "qualified"
    ADVERSE = "adverse"
    DISCLAIMER = "disclaimer"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    sub_workflow_id: str = Field(default="", description="Sub-workflow run ID")


class PipelineCheckpoint(BaseModel):
    """Checkpoint state for pipeline resumption."""
    checkpoint_id: str = Field(default_factory=_new_uuid)
    pipeline_id: str = Field("")
    phase: PipelinePhase = Field(...)
    status: CheckpointStatus = Field(CheckpointStatus.SAVED)
    saved_at: str = Field(default_factory=lambda: _utcnow().isoformat())
    data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field("")


class PipelineMilestone(BaseModel):
    """Milestone marker within the pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    milestone_name: str = Field(...)
    phase: PipelinePhase = Field(...)
    achieved_at: str = Field("")
    metric_name: str = Field("")
    metric_value: str = Field("")


class AuditTrailEntry(BaseModel):
    """Audit trail entry for the consolidation pipeline."""
    entry_id: str = Field(default_factory=_new_uuid)
    phase: str = Field("")
    action: str = Field("")
    timestamp: str = Field(default_factory=lambda: _utcnow().isoformat())
    actor: str = Field("system")
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field("")


class PipelineSummaryReport(BaseModel):
    """Summary report generated at pipeline completion."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    report_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    generated_at: str = Field("")
    total_entities: int = Field(0)
    entities_in_boundary: int = Field(0)
    consolidation_approach: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    eliminations_tco2e: Decimal = Field(Decimal("0"))
    data_quality_score: Decimal = Field(Decimal("0"))
    completeness_pct: Decimal = Field(Decimal("0"))
    phases_completed: int = Field(0)
    phases_skipped: int = Field(0)
    phases_failed: int = Field(0)
    audit_verdict: AuditVerdict = Field(AuditVerdict.CLEAN)
    provenance_chain: List[str] = Field(default_factory=list)
    provenance_hash: str = Field("")


class FullConsolidationInput(BaseModel):
    """Input for the full consolidation pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organisation_id: str = Field(...)
    organisation_name: str = Field("")
    reporting_year: int = Field(...)
    base_year: int = Field(0)
    consolidation_approach: str = Field("operational_control")

    # Sub-workflow configs
    entity_mapping_config: Dict[str, Any] = Field(default_factory=dict)
    data_collection_config: Dict[str, Any] = Field(default_factory=dict)
    boundary_selection_config: Dict[str, Any] = Field(default_factory=dict)
    consolidation_exec_config: Dict[str, Any] = Field(default_factory=dict)
    elimination_config: Dict[str, Any] = Field(default_factory=dict)
    reporting_config: Dict[str, Any] = Field(default_factory=dict)

    # Pipeline control
    skip_phases: List[str] = Field(default_factory=list)
    stop_on_failure: bool = Field(True)
    enable_checkpoints: bool = Field(False)
    resume_from_phase: Optional[str] = Field(None)
    report_formats: List[str] = Field(default_factory=lambda: ["json"])

    # Master data
    entity_data: List[Dict[str, Any]] = Field(default_factory=list)
    ownership_links: List[Dict[str, Any]] = Field(default_factory=list)
    entity_emissions: List[Dict[str, Any]] = Field(default_factory=list)
    intercompany_transfers: List[Dict[str, Any]] = Field(default_factory=list)
    stakeholder_votes: List[Dict[str, Any]] = Field(default_factory=list)
    signatories: List[Dict[str, Any]] = Field(default_factory=list)


class FullConsolidationResult(BaseModel):
    """Output from the full consolidation pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    checkpoints: List[PipelineCheckpoint] = Field(default_factory=list)
    milestones: List[PipelineMilestone] = Field(default_factory=list)
    audit_trail: List[AuditTrailEntry] = Field(default_factory=list)
    summary_report: Optional[PipelineSummaryReport] = Field(None)
    sub_workflow_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class FullConsolidationPipelineWorkflow:
    """
    8-phase end-to-end orchestrator for GHG consolidation.

    Chains EntityMapping -> DataCollection -> BoundarySelection ->
    OwnershipResolution -> ConsolidationExecution -> Elimination ->
    GroupReporting -> AuditFinalization with conditional phase execution,
    checkpointing, and full provenance chain.

    Example:
        >>> wf = FullConsolidationPipelineWorkflow()
        >>> inp = FullConsolidationInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     entity_data=[{"entity_name": "Sub A", "jurisdiction": "DE"}],
        ...     entity_emissions=[{"entity_id": "E1", "scope_1_tco2e": "1000"}],
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[PipelinePhase] = [
        PipelinePhase.ENTITY_SETUP,
        PipelinePhase.DATA_COLLECTION,
        PipelinePhase.BOUNDARY_DEFINITION,
        PipelinePhase.OWNERSHIP_RESOLUTION,
        PipelinePhase.CONSOLIDATION_EXECUTION,
        PipelinePhase.ELIMINATION_PROCESSING,
        PipelinePhase.REPORT_GENERATION,
        PipelinePhase.AUDIT_FINALIZATION,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._state: Dict[str, Any] = {}
        self._provenance_chain: List[str] = []
        self._audit_trail: List[AuditTrailEntry] = []

    def execute(self, input_data: FullConsolidationInput) -> FullConsolidationResult:
        """Execute the full 8-phase consolidation pipeline."""
        start = _utcnow()
        result = FullConsolidationResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        self._add_audit_entry("pipeline", "started", {
            "organisation_id": input_data.organisation_id,
            "reporting_year": input_data.reporting_year,
        })

        # Determine resume point
        resume_idx = 0
        if input_data.resume_from_phase:
            for i, p in enumerate(self.PHASE_ORDER):
                if p.value == input_data.resume_from_phase:
                    resume_idx = i
                    break

        phase_methods = {
            PipelinePhase.ENTITY_SETUP: self._phase_entity_setup,
            PipelinePhase.DATA_COLLECTION: self._phase_data_collection,
            PipelinePhase.BOUNDARY_DEFINITION: self._phase_boundary_definition,
            PipelinePhase.OWNERSHIP_RESOLUTION: self._phase_ownership_resolution,
            PipelinePhase.CONSOLIDATION_EXECUTION: self._phase_consolidation_execution,
            PipelinePhase.ELIMINATION_PROCESSING: self._phase_elimination_processing,
            PipelinePhase.REPORT_GENERATION: self._phase_report_generation,
            PipelinePhase.AUDIT_FINALIZATION: self._phase_audit_finalization,
        }

        for idx, phase in enumerate(self.PHASE_ORDER):
            phase_num = idx + 1

            if idx < resume_idx:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=phase_num,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=phase_num,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            logger.info("Pipeline Phase %d/%d: %s",
                        phase_num, len(self.PHASE_ORDER), phase.value)

            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                ph_hash = _compute_hash(str(phase_out))
                self._provenance_chain.append(ph_hash)

                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=phase_num,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=ph_hash,
                    sub_workflow_id=phase_out.get("sub_workflow_id", ""),
                ))

                result.milestones.append(PipelineMilestone(
                    milestone_name=f"{phase.value}_completed",
                    phase=phase,
                    achieved_at=_utcnow().isoformat(),
                    metric_name="duration_s",
                    metric_value=f"{elapsed:.2f}",
                ))

                if input_data.enable_checkpoints:
                    cp = PipelineCheckpoint(
                        pipeline_id=result.pipeline_id,
                        phase=phase,
                        data={"state_keys": list(self._state.keys())},
                        provenance_hash=ph_hash,
                    )
                    result.checkpoints.append(cp)

                self._add_audit_entry(phase.value, "completed", {
                    "duration_seconds": elapsed,
                    "provenance_hash": ph_hash,
                })

            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Pipeline phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=phase_num,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed,
                    errors=[str(exc)],
                ))
                result.errors.append(f"Phase {phase.value}: {exc}")

                self._add_audit_entry(phase.value, "failed", {"error": str(exc)})

                if input_data.stop_on_failure:
                    result.status = WorkflowStatus.FAILED
                    break
                else:
                    result.warnings.append(f"Phase {phase.value} failed but continuing")

        # Determine final status
        if result.status != WorkflowStatus.FAILED:
            completed = sum(1 for p in result.phase_results if p.status == PhaseStatus.COMPLETED)
            failed = sum(1 for p in result.phase_results if p.status == PhaseStatus.FAILED)
            if failed > 0:
                result.status = WorkflowStatus.PARTIAL
            elif completed > 0:
                result.status = WorkflowStatus.COMPLETED
            else:
                result.status = WorkflowStatus.FAILED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.audit_trail = self._audit_trail

        chain_str = "|".join(self._provenance_chain)
        result.provenance_hash = _compute_hash(
            f"{result.pipeline_id}|{chain_str}|{result.completed_at}"
        )

        self._add_audit_entry("pipeline", "completed", {
            "status": result.status.value,
            "duration_seconds": result.duration_seconds,
        })
        result.audit_trail = self._audit_trail

        logger.info(
            "Pipeline %s: %s in %.2fs (%d completed, %d skipped, %d failed)",
            result.pipeline_id, result.status.value, result.duration_seconds,
            sum(1 for p in result.phase_results if p.status == PhaseStatus.COMPLETED),
            sum(1 for p in result.phase_results if p.status == PhaseStatus.SKIPPED),
            sum(1 for p in result.phase_results if p.status == PhaseStatus.FAILED),
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- ENTITY SETUP
    # -----------------------------------------------------------------

    def _phase_entity_setup(
        self, input_data: FullConsolidationInput, result: FullConsolidationResult,
    ) -> Dict[str, Any]:
        """Execute entity mapping sub-workflow."""
        logger.info("Pipeline: Entity Setup phase")
        from .entity_mapping_workflow import EntityMappingWorkflow, EntityMappingInput

        config = input_data.entity_mapping_config
        sub_input = EntityMappingInput(
            organisation_id=input_data.organisation_id,
            organisation_name=input_data.organisation_name,
            reporting_year=input_data.reporting_year,
            entity_data=input_data.entity_data or config.get("entity_data", []),
            ownership_links=input_data.ownership_links or config.get("ownership_links", []),
            control_indicators=config.get("control_indicators", []),
            emissions_estimates=config.get("emissions_estimates", []),
        )

        wf = EntityMappingWorkflow(config=config)
        sub_result = wf.execute(sub_input)

        self._state["entity_setup"] = {
            "total_discovered": sub_result.total_discovered,
            "total_included": sub_result.total_included,
            "total_excluded": sub_result.total_excluded,
            "locked_entities": [
                {"entity_id": e.entity_id, "entity_name": e.entity_name,
                 "is_included": e.is_included}
                for e in sub_result.locked_entities
            ],
        }
        result.sub_workflow_results["entity_setup"] = {
            "workflow_id": sub_result.workflow_id,
            "total_discovered": sub_result.total_discovered,
            "total_included": sub_result.total_included,
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "entities_discovered": sub_result.total_discovered,
            "entities_included": sub_result.total_included,
            "entities_excluded": sub_result.total_excluded,
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- DATA COLLECTION
    # -----------------------------------------------------------------

    def _phase_data_collection(
        self, input_data: FullConsolidationInput, result: FullConsolidationResult,
    ) -> Dict[str, Any]:
        """Execute entity data collection sub-workflow."""
        logger.info("Pipeline: Data Collection phase")
        from .entity_data_collection_workflow import (
            EntityDataCollectionWorkflow, EntityDataCollectionInput,
        )

        config = input_data.data_collection_config
        entities = self._state.get("entity_setup", {}).get("locked_entities", [])
        entity_list = [e for e in entities if e.get("is_included", True)]

        sub_input = EntityDataCollectionInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            entities=entity_list or config.get("entities", []),
            submissions=input_data.entity_emissions or config.get("submissions", []),
        )

        wf = EntityDataCollectionWorkflow(config=config)
        sub_result = wf.execute(sub_input)

        self._state["data_collection"] = {
            "entities_submitted": sub_result.entities_submitted,
            "entities_approved": sub_result.entities_approved,
            "overall_completeness_pct": float(sub_result.overall_completeness_pct),
        }
        result.sub_workflow_results["data_collection"] = {
            "workflow_id": sub_result.workflow_id,
            "submitted": sub_result.entities_submitted,
            "approved": sub_result.entities_approved,
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "entities_submitted": sub_result.entities_submitted,
            "entities_approved": sub_result.entities_approved,
            "completeness_pct": float(sub_result.overall_completeness_pct),
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 3 -- BOUNDARY DEFINITION
    # -----------------------------------------------------------------

    def _phase_boundary_definition(
        self, input_data: FullConsolidationInput, result: FullConsolidationResult,
    ) -> Dict[str, Any]:
        """Execute boundary selection sub-workflow."""
        logger.info("Pipeline: Boundary Definition phase")
        from .boundary_selection_workflow import (
            BoundarySelectionWorkflow, BoundarySelectionInput,
        )

        config = input_data.boundary_selection_config
        sub_input = BoundarySelectionInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            entity_summaries=config.get("entity_summaries", input_data.entity_data),
            entity_emissions=config.get("entity_emissions", input_data.entity_emissions),
            stakeholder_votes=input_data.stakeholder_votes or config.get("stakeholder_votes", []),
            preferred_approach=input_data.consolidation_approach,
        )

        wf = BoundarySelectionWorkflow(config=config)
        sub_result = wf.execute(sub_input)

        selected = sub_result.selected_approach
        self._state["boundary_definition"] = {
            "selected_approach": selected.value if selected else input_data.consolidation_approach,
        }
        result.sub_workflow_results["boundary_definition"] = {
            "workflow_id": sub_result.workflow_id,
            "selected_approach": selected.value if selected else "",
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "selected_approach": selected.value if selected else "",
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 4 -- OWNERSHIP RESOLUTION
    # -----------------------------------------------------------------

    def _phase_ownership_resolution(
        self, input_data: FullConsolidationInput, result: FullConsolidationResult,
    ) -> Dict[str, Any]:
        """Resolve ownership chains and equity calculations."""
        logger.info("Pipeline: Ownership Resolution phase")

        # Ownership resolution uses data from entity_setup
        entity_state = self._state.get("entity_setup", {})
        entities = entity_state.get("locked_entities", [])
        ownership_links = input_data.ownership_links

        # Build effective ownership map
        ownership_map: Dict[str, Decimal] = {}
        for entity in entities:
            eid = entity.get("entity_id", "")
            # Look up ownership from links
            link = next(
                (l for l in ownership_links if l.get("child_entity_id") == eid),
                None
            )
            if link:
                ownership_map[eid] = self._dec(link.get("ownership_pct", "100"))
            else:
                ownership_map[eid] = Decimal("100.00")

        self._state["ownership_resolution"] = {
            "entities_resolved": len(ownership_map),
            "ownership_map": {k: float(v) for k, v in ownership_map.items()},
        }

        avg_ownership = Decimal("0")
        if ownership_map:
            avg_ownership = (
                sum(ownership_map.values()) / Decimal(str(len(ownership_map)))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        logger.info("Ownership resolved: %d entities, avg %.1f%%",
                     len(ownership_map), float(avg_ownership))
        return {
            "entities_resolved": len(ownership_map),
            "average_ownership_pct": float(avg_ownership),
        }

    # -----------------------------------------------------------------
    # PHASE 5 -- CONSOLIDATION EXECUTION
    # -----------------------------------------------------------------

    def _phase_consolidation_execution(
        self, input_data: FullConsolidationInput, result: FullConsolidationResult,
    ) -> Dict[str, Any]:
        """Execute consolidation sub-workflow."""
        logger.info("Pipeline: Consolidation Execution phase")
        from .consolidation_execution_workflow import (
            ConsolidationExecutionWorkflow, ConsolidationExecInput, ConsolidationApproach,
        )

        config = input_data.consolidation_exec_config
        approach_str = self._state.get("boundary_definition", {}).get(
            "selected_approach", input_data.consolidation_approach
        )
        try:
            approach = ConsolidationApproach(approach_str)
        except ValueError:
            approach = ConsolidationApproach.OPERATIONAL_CONTROL

        sub_input = ConsolidationExecInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            consolidation_approach=approach,
            entity_emissions=input_data.entity_emissions or config.get("entity_emissions", []),
            intercompany_transfers=input_data.intercompany_transfers or config.get("intercompany_transfers", []),
            manual_adjustments=config.get("manual_adjustments", []),
            top_down_estimates=config.get("top_down_estimates"),
        )

        wf = ConsolidationExecutionWorkflow(config=config)
        sub_result = wf.execute(sub_input)

        totals_dict: Dict[str, Any] = {}
        if sub_result.consolidated_total:
            ct = sub_result.consolidated_total
            totals_dict = {
                "scope_1": float(ct.scope_1_tco2e),
                "scope_2_location": float(ct.scope_2_location_tco2e),
                "scope_2_market": float(ct.scope_2_market_tco2e),
                "scope_3": float(ct.scope_3_tco2e),
                "total_location": float(ct.total_location_tco2e),
                "total_market": float(ct.total_market_tco2e),
                "eliminations": float(ct.eliminations_tco2e),
                "entities_count": ct.entities_count,
            }

        self._state["consolidation"] = totals_dict
        result.sub_workflow_results["consolidation"] = {
            "workflow_id": sub_result.workflow_id,
            **totals_dict,
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            **totals_dict,
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 6 -- ELIMINATION PROCESSING
    # -----------------------------------------------------------------

    def _phase_elimination_processing(
        self, input_data: FullConsolidationInput, result: FullConsolidationResult,
    ) -> Dict[str, Any]:
        """Execute elimination sub-workflow."""
        logger.info("Pipeline: Elimination Processing phase")
        from .elimination_workflow import EliminationWorkflow, EliminationInput

        config = input_data.elimination_config
        entity_ids = [
            e.get("entity_id", "")
            for e in self._state.get("entity_setup", {}).get("locked_entities", [])
            if e.get("is_included", True)
        ]

        sub_input = EliminationInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            transfers=input_data.intercompany_transfers or config.get("transfers", []),
            entity_ids=entity_ids or config.get("entity_ids", []),
        )

        wf = EliminationWorkflow(config=config)
        sub_result = wf.execute(sub_input)

        self._state["elimination"] = {
            "total_eliminated": float(sub_result.total_eliminated_tco2e),
            "transfers_count": len(sub_result.identified_transfers),
        }
        result.sub_workflow_results["elimination"] = {
            "workflow_id": sub_result.workflow_id,
            "total_eliminated_tco2e": float(sub_result.total_eliminated_tco2e),
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "total_eliminated_tco2e": float(sub_result.total_eliminated_tco2e),
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 7 -- REPORT GENERATION
    # -----------------------------------------------------------------

    def _phase_report_generation(
        self, input_data: FullConsolidationInput, result: FullConsolidationResult,
    ) -> Dict[str, Any]:
        """Execute group reporting sub-workflow."""
        logger.info("Pipeline: Report Generation phase")
        from .group_reporting_workflow import GroupReportingWorkflow, GroupReportingInput

        config = input_data.reporting_config
        cons_state = self._state.get("consolidation", {})

        consolidated_data = {
            "scope_1_tco2e": str(cons_state.get("scope_1", 0)),
            "scope_2_location_tco2e": str(cons_state.get("scope_2_location", 0)),
            "scope_2_market_tco2e": str(cons_state.get("scope_2_market", 0)),
            "scope_3_tco2e": str(cons_state.get("scope_3", 0)),
            "entities_count": cons_state.get("entities_count", 0),
            "eliminations_tco2e": str(cons_state.get("eliminations", 0)),
        }

        sub_input = GroupReportingInput(
            organisation_id=input_data.organisation_id,
            organisation_name=input_data.organisation_name,
            reporting_year=input_data.reporting_year,
            base_year=input_data.base_year,
            consolidation_approach=self._state.get("boundary_definition", {}).get(
                "selected_approach", input_data.consolidation_approach
            ),
            consolidated_data=consolidated_data,
            target_frameworks=config.get("target_frameworks", ["ghg_protocol"]),
            report_formats=input_data.report_formats,
            signatories=input_data.signatories or config.get("signatories", []),
        )

        wf = GroupReportingWorkflow(config=config)
        sub_result = wf.execute(sub_input)

        self._state["reporting"] = {
            "reports_generated": len(sub_result.generated_reports),
            "qa_pass_rate": float(sub_result.qa_pass_rate_pct),
        }
        result.sub_workflow_results["reporting"] = {
            "workflow_id": sub_result.workflow_id,
            "reports_generated": len(sub_result.generated_reports),
            "qa_pass_rate_pct": float(sub_result.qa_pass_rate_pct),
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "reports_generated": len(sub_result.generated_reports),
            "qa_pass_rate_pct": float(sub_result.qa_pass_rate_pct),
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 8 -- AUDIT FINALIZATION
    # -----------------------------------------------------------------

    def _phase_audit_finalization(
        self, input_data: FullConsolidationInput, result: FullConsolidationResult,
    ) -> Dict[str, Any]:
        """Generate final audit trail and summary report."""
        logger.info("Pipeline: Audit Finalization phase")
        now_iso = _utcnow().isoformat()

        cons_state = self._state.get("consolidation", {})
        entity_state = self._state.get("entity_setup", {})
        collection_state = self._state.get("data_collection", {})
        reporting_state = self._state.get("reporting", {})

        # Determine audit verdict
        phases_failed = sum(1 for p in result.phase_results if p.status == PhaseStatus.FAILED)
        qa_pass = reporting_state.get("qa_pass_rate", 0)

        if phases_failed == 0 and qa_pass >= 80:
            verdict = AuditVerdict.CLEAN
        elif phases_failed <= 1 and qa_pass >= 60:
            verdict = AuditVerdict.QUALIFIED
        elif phases_failed <= 2:
            verdict = AuditVerdict.ADVERSE
        else:
            verdict = AuditVerdict.DISCLAIMER

        summary = PipelineSummaryReport(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            generated_at=now_iso,
            total_entities=entity_state.get("total_discovered", 0),
            entities_in_boundary=entity_state.get("total_included", 0),
            consolidation_approach=self._state.get("boundary_definition", {}).get(
                "selected_approach", input_data.consolidation_approach
            ),
            scope_1_tco2e=Decimal(str(cons_state.get("scope_1", 0))),
            scope_2_location_tco2e=Decimal(str(cons_state.get("scope_2_location", 0))),
            scope_2_market_tco2e=Decimal(str(cons_state.get("scope_2_market", 0))),
            scope_3_tco2e=Decimal(str(cons_state.get("scope_3", 0))),
            total_tco2e=Decimal(str(cons_state.get("total_location", 0))),
            eliminations_tco2e=Decimal(str(cons_state.get("eliminations", 0))),
            data_quality_score=Decimal(str(qa_pass)),
            completeness_pct=Decimal(str(collection_state.get("overall_completeness_pct", 0))),
            phases_completed=sum(1 for p in result.phase_results if p.status == PhaseStatus.COMPLETED),
            phases_skipped=sum(1 for p in result.phase_results if p.status == PhaseStatus.SKIPPED),
            phases_failed=phases_failed,
            audit_verdict=verdict,
            provenance_chain=self._provenance_chain.copy(),
        )

        summary.provenance_hash = _compute_hash(
            f"{summary.report_id}|{summary.organisation_id}|"
            f"{summary.reporting_year}|{float(summary.total_tco2e)}|{now_iso}"
        )

        result.summary_report = summary

        logger.info(
            "Audit finalized: verdict=%s, total=%.2f tCO2e, quality=%.1f",
            verdict.value, float(summary.total_tco2e), float(summary.data_quality_score),
        )
        return {
            "audit_verdict": verdict.value,
            "total_tco2e": float(summary.total_tco2e),
            "entities_in_boundary": summary.entities_in_boundary,
            "provenance_hash": summary.provenance_hash,
        }

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _add_audit_entry(
        self, phase: str, action: str, details: Dict[str, Any]
    ) -> None:
        """Add an entry to the audit trail."""
        prov = _compute_hash(f"{phase}|{action}|{_utcnow().isoformat()}")
        entry = AuditTrailEntry(
            phase=phase, action=action, details=details, provenance_hash=prov,
        )
        self._audit_trail.append(entry)

    def _dec(self, value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FullConsolidationPipelineWorkflow",
    "FullConsolidationInput",
    "FullConsolidationResult",
    "PipelinePhase",
    "ReportFormat",
    "CheckpointStatus",
    "AuditVerdict",
    "PipelineCheckpoint",
    "PipelineMilestone",
    "AuditTrailEntry",
    "PipelineSummaryReport",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
