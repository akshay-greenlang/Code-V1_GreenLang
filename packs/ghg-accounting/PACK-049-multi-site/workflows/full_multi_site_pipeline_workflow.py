# -*- coding: utf-8 -*-
"""
Full Multi-Site Pipeline Workflow
====================================

8-phase end-to-end orchestrator that chains all 7 sub-workflows plus
reporting for the complete multi-site GHG management lifecycle within
PACK-049 GHG Multi-Site Management Pack.

Phases:
    1. Registration         -- Site registration (SiteRegistrationWorkflow)
    2. Collection           -- Data collection (DataCollectionWorkflow)
    3. Boundary             -- Boundary definition (BoundaryDefinitionWorkflow)
    4. Consolidation        -- Emissions consolidation (ConsolidationWorkflow)
    5. Allocation           -- Shared service allocation (AllocationWorkflow)
    6. Comparison           -- Site comparison (SiteComparisonWorkflow)
    7. Quality              -- Quality improvement (QualityImprovementWorkflow)
    8. Reporting            -- Final reporting and provenance generation

Supports conditional phase execution, checkpoint caching, and full
provenance chain across all sub-workflows.

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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
    REGISTRATION = "registration"
    COLLECTION = "collection"
    BOUNDARY = "boundary"
    CONSOLIDATION = "consolidation"
    ALLOCATION = "allocation"
    COMPARISON = "comparison"
    QUALITY = "quality"
    REPORTING = "reporting"


class ReportFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


class CheckpointStatus(str, Enum):
    SAVED = "saved"
    LOADED = "loaded"
    NOT_FOUND = "not_found"


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


class PipelineReport(BaseModel):
    """Summary report generated at pipeline completion."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    generated_at: str = Field("")
    total_sites: int = Field(0)
    active_sites: int = Field(0)
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    data_quality_score: Decimal = Field(Decimal("0"))
    completeness_pct: Decimal = Field(Decimal("0"))
    reduction_potential_tco2e: Decimal = Field(Decimal("0"))
    phases_completed: int = Field(0)
    phases_skipped: int = Field(0)
    phases_failed: int = Field(0)
    provenance_chain: List[str] = Field(default_factory=list)
    provenance_hash: str = Field("")


class FullPipelineInput(BaseModel):
    """Input for the full multi-site pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    organisation_id: str = Field(...)
    organisation_name: str = Field("")
    reporting_year: int = Field(...)

    # Sub-workflow inputs (optional -- pipeline builds defaults from master data)
    registration_config: Dict[str, Any] = Field(default_factory=dict)
    collection_config: Dict[str, Any] = Field(default_factory=dict)
    boundary_config: Dict[str, Any] = Field(default_factory=dict)
    consolidation_config: Dict[str, Any] = Field(default_factory=dict)
    allocation_config: Dict[str, Any] = Field(default_factory=dict)
    comparison_config: Dict[str, Any] = Field(default_factory=dict)
    quality_config: Dict[str, Any] = Field(default_factory=dict)

    # Pipeline control
    skip_phases: List[str] = Field(default_factory=list)
    stop_on_failure: bool = Field(True)
    enable_checkpoints: bool = Field(False)
    resume_from_phase: Optional[str] = Field(None)
    report_formats: List[str] = Field(default_factory=lambda: ["json"])

    # Master data
    candidate_sites: List[Dict[str, Any]] = Field(default_factory=list)
    site_metadata: List[Dict[str, Any]] = Field(default_factory=list)
    submissions: List[Dict[str, Any]] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    ownership_links: List[Dict[str, Any]] = Field(default_factory=list)
    site_totals: List[Dict[str, Any]] = Field(default_factory=list)
    shared_services: List[Dict[str, Any]] = Field(default_factory=list)
    site_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    site_quality_data: List[Dict[str, Any]] = Field(default_factory=list)


class FullPipelineResult(BaseModel):
    """Output from the full multi-site pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pipeline_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    checkpoints: List[PipelineCheckpoint] = Field(default_factory=list)
    milestones: List[PipelineMilestone] = Field(default_factory=list)
    report: Optional[PipelineReport] = Field(None)
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


class FullMultiSitePipelineWorkflow:
    """
    8-phase end-to-end orchestrator for multi-site GHG management.

    Chains SiteRegistration -> DataCollection -> BoundaryDefinition ->
    Consolidation -> Allocation -> SiteComparison -> QualityImprovement ->
    Reporting with conditional phase execution, checkpointing, and full
    provenance chain.

    Example:
        >>> wf = FullMultiSitePipelineWorkflow()
        >>> inp = FullPipelineInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     candidate_sites=[{"site_name": "Plant A", "country_code": "DE"}],
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[PipelinePhase] = [
        PipelinePhase.REGISTRATION,
        PipelinePhase.COLLECTION,
        PipelinePhase.BOUNDARY,
        PipelinePhase.CONSOLIDATION,
        PipelinePhase.ALLOCATION,
        PipelinePhase.COMPARISON,
        PipelinePhase.QUALITY,
        PipelinePhase.REPORTING,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._state: Dict[str, Any] = {}
        self._provenance_chain: List[str] = []

    def execute(self, input_data: FullPipelineInput) -> FullPipelineResult:
        """Execute the full 8-phase multi-site pipeline."""
        start = _utcnow()
        result = FullPipelineResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        # Determine resume point
        resume_idx = 0
        if input_data.resume_from_phase:
            for i, p in enumerate(self.PHASE_ORDER):
                if p.value == input_data.resume_from_phase:
                    resume_idx = i
                    break

        phase_methods = {
            PipelinePhase.REGISTRATION: self._phase_registration,
            PipelinePhase.COLLECTION: self._phase_collection,
            PipelinePhase.BOUNDARY: self._phase_boundary,
            PipelinePhase.CONSOLIDATION: self._phase_consolidation,
            PipelinePhase.ALLOCATION: self._phase_allocation,
            PipelinePhase.COMPARISON: self._phase_comparison,
            PipelinePhase.QUALITY: self._phase_quality,
            PipelinePhase.REPORTING: self._phase_reporting,
        }

        for idx, phase in enumerate(self.PHASE_ORDER):
            phase_num = idx + 1

            # Skip if before resume point
            if idx < resume_idx:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=phase_num,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            # Skip if in skip list
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=phase_num,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            logger.info("Pipeline Phase %d/%d: %s", phase_num, len(self.PHASE_ORDER), phase.value)

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

                # Milestone
                result.milestones.append(PipelineMilestone(
                    milestone_name=f"{phase.value}_completed",
                    phase=phase,
                    achieved_at=_utcnow().isoformat(),
                    metric_name="duration_s",
                    metric_value=f"{elapsed:.2f}",
                ))

                # Checkpoint
                if input_data.enable_checkpoints:
                    cp = PipelineCheckpoint(
                        pipeline_id=result.pipeline_id,
                        phase=phase,
                        data={"state_keys": list(self._state.keys())},
                        provenance_hash=ph_hash,
                    )
                    result.checkpoints.append(cp)

            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Pipeline phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=phase_num,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed,
                    errors=[str(exc)],
                ))
                result.errors.append(f"Phase {phase.value}: {exc}")

                if input_data.stop_on_failure:
                    result.status = WorkflowStatus.FAILED
                    break
                else:
                    # Continue with partial success
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

        # Final provenance: chain of all phase hashes
        chain_str = "|".join(self._provenance_chain)
        result.provenance_hash = _compute_hash(
            f"{result.pipeline_id}|{chain_str}|{result.completed_at}"
        )

        logger.info(
            "Pipeline %s: %s in %.2fs (%d completed, %d skipped, %d failed)",
            result.pipeline_id, result.status.value, result.duration_seconds,
            sum(1 for p in result.phase_results if p.status == PhaseStatus.COMPLETED),
            sum(1 for p in result.phase_results if p.status == PhaseStatus.SKIPPED),
            sum(1 for p in result.phase_results if p.status == PhaseStatus.FAILED),
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- REGISTRATION
    # -----------------------------------------------------------------

    def _phase_registration(
        self, input_data: FullPipelineInput, result: FullPipelineResult,
    ) -> Dict[str, Any]:
        """Execute site registration sub-workflow."""
        logger.info("Pipeline: Registration phase")
        from .site_registration_workflow import SiteRegistrationWorkflow, SiteRegistrationInput

        reg_config = input_data.registration_config
        reg_input = SiteRegistrationInput(
            organisation_id=input_data.organisation_id,
            organisation_name=input_data.organisation_name,
            reporting_year=input_data.reporting_year,
            candidate_sites=input_data.candidate_sites or reg_config.get("candidate_sites", []),
            existing_sites=reg_config.get("existing_sites", []),
            auto_classify=reg_config.get("auto_classify", True),
        )

        wf = SiteRegistrationWorkflow(config=reg_config)
        sub_result = wf.execute(reg_input)

        self._state["registration"] = {
            "registered_sites": [
                {"site_id": s.site_id, "site_name": s.site_name, "country_code": s.country_code,
                 "facility_type": s.facility_type.value, "status": s.status.value}
                for s in sub_result.registered_sites
            ],
            "total_activated": sub_result.total_activated,
        }
        result.sub_workflow_results["registration"] = {
            "workflow_id": sub_result.workflow_id,
            "total_activated": sub_result.total_activated,
            "total_excluded": sub_result.total_excluded,
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "sites_activated": sub_result.total_activated,
            "sites_excluded": sub_result.total_excluded,
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- COLLECTION
    # -----------------------------------------------------------------

    def _phase_collection(
        self, input_data: FullPipelineInput, result: FullPipelineResult,
    ) -> Dict[str, Any]:
        """Execute data collection sub-workflow."""
        logger.info("Pipeline: Collection phase")
        from .data_collection_workflow import (
            DataCollectionWorkflow, DataCollectionInput, CollectionRoundConfig,
        )

        col_config = input_data.collection_config
        round_cfg = CollectionRoundConfig(
            reporting_period_start=col_config.get("period_start", f"{input_data.reporting_year}-01-01"),
            reporting_period_end=col_config.get("period_end", f"{input_data.reporting_year}-12-31"),
            submission_deadline=col_config.get("deadline", f"{input_data.reporting_year + 1}-03-31T23:59:59Z"),
        )

        col_input = DataCollectionInput(
            organisation_id=input_data.organisation_id,
            round_config=round_cfg,
            site_metadata=input_data.site_metadata or col_config.get("site_metadata", []),
            submissions=input_data.submissions or col_config.get("submissions", []),
        )

        wf = DataCollectionWorkflow(config=col_config)
        sub_result = wf.execute(col_input)

        self._state["collection"] = {
            "total_entries": sub_result.total_entries,
            "approved_count": sub_result.approved_count,
            "completeness_pct": float(sub_result.overall_completeness_pct),
        }
        result.sub_workflow_results["collection"] = {
            "workflow_id": sub_result.workflow_id,
            "total_entries": sub_result.total_entries,
            "approved": sub_result.approved_count,
            "rejected": sub_result.rejected_count,
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "entries_collected": sub_result.total_entries,
            "approved": sub_result.approved_count,
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 3 -- BOUNDARY
    # -----------------------------------------------------------------

    def _phase_boundary(
        self, input_data: FullPipelineInput, result: FullPipelineResult,
    ) -> Dict[str, Any]:
        """Execute boundary definition sub-workflow."""
        logger.info("Pipeline: Boundary phase")
        from .boundary_definition_workflow import BoundaryDefinitionWorkflow, BoundaryDefinitionInput

        bnd_config = input_data.boundary_config
        bnd_input = BoundaryDefinitionInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            entities=input_data.entities or bnd_config.get("entities", []),
            ownership_links=input_data.ownership_links or bnd_config.get("ownership_links", []),
            entity_facility_map=bnd_config.get("entity_facility_map", []),
        )

        wf = BoundaryDefinitionWorkflow(config=bnd_config)
        sub_result = wf.execute(bnd_input)

        self._state["boundary"] = {
            "included_count": sub_result.included_count,
            "excluded_count": sub_result.excluded_count,
        }
        result.sub_workflow_results["boundary"] = {
            "workflow_id": sub_result.workflow_id,
            "included": sub_result.included_count,
            "excluded": sub_result.excluded_count,
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "included_entities": sub_result.included_count,
            "excluded_entities": sub_result.excluded_count,
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 4 -- CONSOLIDATION
    # -----------------------------------------------------------------

    def _phase_consolidation(
        self, input_data: FullPipelineInput, result: FullPipelineResult,
    ) -> Dict[str, Any]:
        """Execute consolidation sub-workflow."""
        logger.info("Pipeline: Consolidation phase")
        from .consolidation_workflow import ConsolidationWorkflow, ConsolidationInput

        cons_config = input_data.consolidation_config
        cons_input = ConsolidationInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            site_totals=input_data.site_totals or cons_config.get("site_totals", []),
            eliminations=cons_config.get("eliminations", []),
            top_down_estimates=cons_config.get("top_down_estimates"),
        )

        wf = ConsolidationWorkflow(config=cons_config)
        sub_result = wf.execute(cons_input)

        totals_dict: Dict[str, Any] = {}
        if sub_result.consolidated_totals:
            ct = sub_result.consolidated_totals
            totals_dict = {
                "scope_1": float(ct.scope_1_tco2e),
                "scope_2_location": float(ct.scope_2_location_tco2e),
                "scope_2_market": float(ct.scope_2_market_tco2e),
                "scope_3": float(ct.scope_3_tco2e),
                "total_location": float(ct.total_location_tco2e),
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
    # PHASE 5 -- ALLOCATION
    # -----------------------------------------------------------------

    def _phase_allocation(
        self, input_data: FullPipelineInput, result: FullPipelineResult,
    ) -> Dict[str, Any]:
        """Execute allocation sub-workflow."""
        logger.info("Pipeline: Allocation phase")
        from .allocation_workflow import AllocationWorkflow, AllocationInput

        alloc_config = input_data.allocation_config
        alloc_input = AllocationInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            shared_services=input_data.shared_services or alloc_config.get("shared_services", []),
            site_drivers=input_data.site_drivers or alloc_config.get("site_drivers", []),
            method_overrides=alloc_config.get("method_overrides", {}),
        )

        wf = AllocationWorkflow(config=alloc_config)
        sub_result = wf.execute(alloc_input)

        self._state["allocation"] = {
            "total_allocated": float(sub_result.total_allocated_tco2e),
            "services": sub_result.services_count,
        }
        result.sub_workflow_results["allocation"] = {
            "workflow_id": sub_result.workflow_id,
            "total_allocated_tco2e": float(sub_result.total_allocated_tco2e),
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "total_allocated_tco2e": float(sub_result.total_allocated_tco2e),
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 6 -- COMPARISON
    # -----------------------------------------------------------------

    def _phase_comparison(
        self, input_data: FullPipelineInput, result: FullPipelineResult,
    ) -> Dict[str, Any]:
        """Execute site comparison sub-workflow."""
        logger.info("Pipeline: Comparison phase")
        from .site_comparison_workflow import SiteComparisonWorkflow, SiteComparisonInput

        comp_config = input_data.comparison_config
        comp_input = SiteComparisonInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            site_metrics=comp_config.get("site_metrics", input_data.site_totals or []),
            peer_group_criteria=comp_config.get("peer_group_criteria", ["facility_type"]),
            kpi_types=comp_config.get("kpi_types", ["tco2e_per_sqm", "tco2e_per_fte"]),
        )

        wf = SiteComparisonWorkflow(config=comp_config)
        sub_result = wf.execute(comp_input)

        self._state["comparison"] = {
            "peer_groups": len(sub_result.peer_groups),
            "reduction_potential": float(sub_result.total_reduction_potential_tco2e),
        }
        result.sub_workflow_results["comparison"] = {
            "workflow_id": sub_result.workflow_id,
            "peer_groups": len(sub_result.peer_groups),
            "reduction_potential_tco2e": float(sub_result.total_reduction_potential_tco2e),
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "peer_groups": len(sub_result.peer_groups),
            "reduction_potential_tco2e": float(sub_result.total_reduction_potential_tco2e),
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 7 -- QUALITY
    # -----------------------------------------------------------------

    def _phase_quality(
        self, input_data: FullPipelineInput, result: FullPipelineResult,
    ) -> Dict[str, Any]:
        """Execute quality improvement sub-workflow."""
        logger.info("Pipeline: Quality phase")
        from .quality_improvement_workflow import QualityImprovementWorkflow, QualityImprovementInput

        qual_config = input_data.quality_config
        qual_input = QualityImprovementInput(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            site_quality_data=input_data.site_quality_data or qual_config.get("site_quality_data", []),
        )

        wf = QualityImprovementWorkflow(config=qual_config)
        sub_result = wf.execute(qual_input)

        self._state["quality"] = {
            "corporate_score": float(sub_result.corporate_score),
            "sites_below_threshold": sub_result.sites_below_threshold,
            "total_gaps": sub_result.total_gaps,
        }
        result.sub_workflow_results["quality"] = {
            "workflow_id": sub_result.workflow_id,
            "corporate_score": float(sub_result.corporate_score),
            "gaps": sub_result.total_gaps,
            "status": sub_result.status.value,
        }

        return {
            "sub_workflow_id": sub_result.workflow_id,
            "corporate_quality_score": float(sub_result.corporate_score),
            "total_gaps": sub_result.total_gaps,
            "status": sub_result.status.value,
        }

    # -----------------------------------------------------------------
    # PHASE 8 -- REPORTING
    # -----------------------------------------------------------------

    def _phase_reporting(
        self, input_data: FullPipelineInput, result: FullPipelineResult,
    ) -> Dict[str, Any]:
        """Generate final pipeline report."""
        logger.info("Pipeline: Reporting phase")
        now_iso = _utcnow().isoformat()

        reg_state = self._state.get("registration", {})
        cons_state = self._state.get("consolidation", {})
        qual_state = self._state.get("quality", {})
        col_state = self._state.get("collection", {})
        comp_state = self._state.get("comparison", {})

        report = PipelineReport(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            generated_at=now_iso,
            total_sites=reg_state.get("total_activated", len(input_data.candidate_sites)),
            active_sites=reg_state.get("total_activated", 0),
            scope_1_tco2e=Decimal(str(cons_state.get("scope_1", 0))),
            scope_2_location_tco2e=Decimal(str(cons_state.get("scope_2_location", 0))),
            scope_2_market_tco2e=Decimal(str(cons_state.get("scope_2_market", 0))),
            scope_3_tco2e=Decimal(str(cons_state.get("scope_3", 0))),
            total_tco2e=Decimal(str(cons_state.get("total_location", 0))),
            data_quality_score=Decimal(str(qual_state.get("corporate_score", 0))),
            completeness_pct=Decimal(str(col_state.get("completeness_pct", 0))),
            reduction_potential_tco2e=Decimal(str(comp_state.get("reduction_potential", 0))),
            phases_completed=sum(
                1 for p in result.phase_results if p.status == PhaseStatus.COMPLETED
            ),
            phases_skipped=sum(
                1 for p in result.phase_results if p.status == PhaseStatus.SKIPPED
            ),
            phases_failed=sum(
                1 for p in result.phase_results if p.status == PhaseStatus.FAILED
            ),
            provenance_chain=self._provenance_chain.copy(),
        )

        report.provenance_hash = _compute_hash(
            f"{report.report_id}|{report.organisation_id}|{report.reporting_year}|"
            f"{float(report.total_tco2e)}|{now_iso}"
        )

        result.report = report

        logger.info(
            "Report generated: %d sites, %.2f tCO2e, quality %.1f",
            report.active_sites, float(report.total_tco2e),
            float(report.data_quality_score),
        )
        return {
            "report_id": report.report_id,
            "total_sites": report.total_sites,
            "total_tco2e": float(report.total_tco2e),
            "quality_score": float(report.data_quality_score),
            "provenance_hash": report.provenance_hash,
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FullMultiSitePipelineWorkflow",
    "FullPipelineInput",
    "FullPipelineResult",
    "PipelinePhase",
    "ReportFormat",
    "CheckpointStatus",
    "PipelineCheckpoint",
    "PipelineMilestone",
    "PipelineReport",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
